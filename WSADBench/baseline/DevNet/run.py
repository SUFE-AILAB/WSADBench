# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD '19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871

PyTorch reimplementation by Xu Yao 2025-07-20.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
import sys
import os
import copy
from WSADBench.myutils import Utils
from typing import Literal


class DevNetDataset(Dataset):
    """Custom Dataset for DevNet"""

    def __init__(self, X, outlier_indices, inlier_indices, batch_size, rng):
        self.X = X
        self.outlier_indices = outlier_indices
        self.inlier_indices = inlier_indices
        self.batch_size = batch_size
        self.rng = rng
        self.n_inliers = len(inlier_indices)
        self.n_outliers = len(outlier_indices)

    def __len__(self):
        return self.batch_size * 100  # arbitrary large number for continuous generation

    def __getitem__(self, idx):
        if idx % 2 == 0:
            # Inlier
            sid = self.rng.choice(self.n_inliers, 1)[0]
            sample = self.X[self.inliers_indices[sid]]
            label = 0.0
        else:
            # Outlier
            sid = self.rng.choice(self.n_outliers, 1)[0]
            sample = self.X[self.outlier_indices[sid]]
            label = 1.0

        return torch.FloatTensor(sample), torch.FloatTensor([label])


class DevNetworkDeep(nn.Module):
    """Deeper network architecture with three hidden layers"""

    def __init__(self, input_dim):
        super(DevNetworkDeep, self).__init__()
        self.hl1 = nn.Linear(input_dim, 1000)
        self.hl2 = nn.Linear(1000, 250)
        self.hl3 = nn.Linear(250, 20)
        self.score = nn.Linear(20, 1)

        # Apply L2 regularization
        self.weight_decay = 0.01

    def forward(self, x):
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = F.relu(self.hl3(x))
        x = self.score(x)
        return x


class DevNetworkShallow(nn.Module):
    """Network architecture with one hidden layer"""

    def __init__(self, input_dim):
        super(DevNetworkShallow, self).__init__()
        self.hl1 = nn.Linear(input_dim, 20)
        self.score = nn.Linear(20, 1)

        # Apply L2 regularization
        self.weight_decay = 0.01

    def forward(self, x):
        x = F.relu(self.hl1(x))
        x = self.score(x)
        return x


class DevNetworkLinear(nn.Module):
    """Network architecture with no hidden layer"""

    def __init__(self, input_dim):
        super(DevNetworkLinear, self).__init__()
        self.score = nn.Linear(input_dim, 1)

        # Apply L2 regularization
        self.weight_decay = 0.01

    def forward(self, x):
        x = self.score(x)
        return x


class DeviationLoss(nn.Module):
    """z-score-based deviation loss"""

    def __init__(self, device):
        super(DeviationLoss, self).__init__()
        self.device = device
        self.confidence_margin = 5.0
        self.ref = None

    def forward(self, y_pred, y_true):
        # Initialize reference if not exists
        if self.ref is None:
            self.ref = torch.normal(mean=0.0, std=1.0, size=(5000,), device=self.device)

        # Calculate deviation
        ref_mean = torch.mean(self.ref)
        ref_std = torch.std(self.ref)
        dev = (y_pred.squeeze() - ref_mean) / ref_std

        # Calculate losses
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(torch.clamp(self.confidence_margin - dev, min=0.0))

        # Combine losses based on labels
        y_true = y_true.squeeze()
        loss = (1 - y_true) * inlier_loss + y_true * outlier_loss

        return torch.mean(loss)


class DevNet:
    def __init__(
        self,
        seed,
        model_name="DevNet",
        best_model_method:Literal["min_train_loss","last_epoch"]="min_train_loss",
        epochs=50,
        batch_size=512,
        nb_batch=20,
        network_depth=2,
        loss_name:Literal["Deviation", "BCE"]="Deviation",
    ):
        """

        Args:
            seed: random seed for reproducibility
            model_name: name of the model
            best_model_method: method to determine the best model, e.g., "min_train_loss
            epochs: number of training epochs
            batch_size: size of each training batch
            nb_batch: number of batches per epoch
            network_depth: depth of the network architecture (1, 2, or 4)
        """
        self.utils = Utils()
        self.device = self.utils.get_device(True)  # get device
        self.seed = seed
        self.MAX_INT = np.iinfo(np.int32).max
        self.best_model_method = best_model_method
        self.loss_name = loss_name

        self.epochs = epochs
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.model_name = model_name
        self.network_depth = int(network_depth)

        # Initialize model and loss
        self.model = None
        self.best_model = None  # Keep best model in memory
        self.criterion = None
        self.input_dim = None

    def create_deviation_network(self, input_dim, network_depth):
        """
        construct the deviation network-based detection model
        """
        if network_depth == 4:
            model = DevNetworkDeep(input_dim)
        elif network_depth == 2:
            model = DevNetworkShallow(input_dim)
        elif network_depth == 1:
            model = DevNetworkLinear(input_dim)
        else:
            sys.exit("The network depth is not set properly")

        return model.to(self.device)

    def create_data_loader(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        """Create data loader for training"""
        dataset = DevNetDataset(X_train, outlier_indices, inlier_indices, batch_size, rng)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def input_batch_generation_sup(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        """
        Generate batch of samples. This is for csv data.
        Alternates between positive and negative pairs.
        """
        dim = X_train.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)

        for i in range(batch_size):
            if i % 2 == 0:
                sid = rng.choice(n_inliers, 1)
                ref[i] = X_train[inlier_indices[sid]]
                training_labels.append(0)
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = X_train[outlier_indices[sid]]
                training_labels.append(1)

        return torch.FloatTensor(ref).to(self.device), torch.FloatTensor(training_labels).to(self.device)

    def fit(self, X_train, y_train, ratio=None):
        """Train the DevNet model"""
        # Get indices
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]
        n_outliers = len(outlier_indices)
        print("Training size: %d, No. outliers: %d" % (X_train.shape[0], n_outliers))

        # Set seed
        self.utils.set_seed(self.seed)
        rng = np.random.RandomState(self.seed)

        # Setup model
        self.input_dim = X_train.shape[1]
        epochs = self.epochs
        batch_size = self.batch_size
        nb_batch = self.nb_batch

        self.model = self.create_deviation_network(self.input_dim, self.network_depth)
        if self.loss_name == "BCE":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_name == "Deviation":
            self.criterion = DeviationLoss(self.device)
        else:
            raise ValueError("Unsupported loss function: {}".format(self.loss_name))

        # Setup optimizer with weight decay for L2 regularization
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=0.01)

        best_loss = float("inf")

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx in range(nb_batch):
                # Generate batch
                batch_data, batch_labels = self.input_batch_generation_sup(
                    X_train, outlier_indices, inlier_indices, batch_size, rng
                )

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels.unsqueeze(1))

                # Backward pass
                loss.backward()

                # Gradient clipping (equivalent to clipnorm=1 in RMSprop)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= nb_batch

            # Keep best model in memory
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # Deep copy the current model state to keep the best model
                self.best_model = copy.deepcopy(self.model)

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

        return self

    def predict_score(self, X):
        """Predict anomaly scores using the best model"""
        # Use the best model if available, otherwise use current model
        if self.best_model_method  == "last_epoch":
            model_to_use = self.model
        else:
            model_to_use = self.best_model if self.best_model is not None else self.model

        if model_to_use is None:
            raise ValueError("Model has not been trained yet!")

        model_to_use.eval()
        X_tensor = torch.FloatTensor(X)
        all_scores = []
        batch_size = 1024 # 分批推理
        with torch.no_grad():
            for i in range(0,len(X_tensor),batch_size):
                batch_X = X_tensor[i:i+batch_size]  # 切片允许越界，会截断
                batch_X = batch_X.to(self.device)
                batch_score = model_to_use(batch_X)
                all_scores.append(batch_score.cpu().numpy())

        score = np.concatenate(all_scores, axis=0)
        return score

    def input_batch_generation_sup_sparse(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        """
        Generate batch of samples. This is for libsvm stored sparse data.
        Alternates between positive and negative pairs.
        """
        ref = np.empty((batch_size), dtype=int)
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)

        for i in range(batch_size):
            if i % 2 == 0:
                sid = rng.choice(n_inliers, 1)
                ref[i] = inlier_indices[sid]
                training_labels.append(0)
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = outlier_indices[sid]
                training_labels.append(1)

        # Convert sparse matrix to dense
        ref_data = X_train[ref, :].toarray()

        return torch.FloatTensor(ref_data).to(self.device), torch.FloatTensor(training_labels).to(self.device)
