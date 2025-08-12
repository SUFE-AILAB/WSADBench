# -*- coding: utf-8 -*-
"""
@author：Xu Yao
The algorithm was implemented using Python 3.9.21, PyTorch 2.5.1 based on the original Keras code.
This is a PyTorch reimplementation of the FEAWAD algorithm from the paper:
Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu and Lingqiao Liu,
Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection,
in IEEE Transactions on Neural Networks and Learning Systems, 2021.
"""
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from WSADBench.myutils import Utils
from copy import deepcopy

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.ReLU(True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class DevNet(nn.Module):
    def __init__(self, input_dim, confidence_margin=5.):
        super(DevNet, self).__init__()
        self.confidence_margin = confidence_margin
        
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.ReLU(True)
        )
        
        # Anomaly score generator MLP
        # Input: [hidden_rep, normalized_recon_residual, recon_error_norm]
        # Size: 64 + input_dim + 1 = 65 + input_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(64 + input_dim + 1, 256),
            nn.ReLU(True),
            # Input to next layer: [intermediate_vec, recon_error_norm]
            # Size: 256 + 1
            nn.Linear(256 + 1, 32),
            nn.ReLU(True),
            # Input to next layer: [intermediate_vec, recon_error_norm]
            # Size: 32 + 1
            nn.Linear(32 + 1, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_ae_weights(self, ae_model: AutoEncoder):
        self.encoder_net.load_state_dict(ae_model.encoder.state_dict())
        self.decoder_net.load_state_dict(ae_model.decoder.state_dict())

    def forward(self, x):
        # AutoEncoder part
        encoded = self.encoder_net(x)
        decoded = self.decoder_net(encoded)

        # Feature engineering part
        recon_residual = x - decoded
        recon_error_norm = torch.norm(recon_residual, p=2, dim=1, keepdim=True)
        # Add a small epsilon to avoid division by zero
        normalized_recon_residual = recon_residual / (recon_error_norm + 1e-8)
        
        # Anomaly score generator part
        # Concat [hidden_rep, normalized_recon_residual, recon_error_norm]
        s_input = torch.cat([encoded, normalized_recon_residual, recon_error_norm], dim=1)
        
        intermediate = self.score_mlp[0](s_input)
        intermediate = self.score_mlp[1](intermediate) # ReLU
        
        intermediate_cat = torch.cat([intermediate, recon_error_norm], dim=1)
        intermediate = self.score_mlp[2](intermediate_cat)
        intermediate = self.score_mlp[3](intermediate) # ReLU

        intermediate_cat = torch.cat([intermediate, recon_error_norm], dim=1)
        score = self.score_mlp[4](intermediate_cat)
        
        return score, recon_residual

    def loss_function(self, y_true, y_pred, recon_residual):
        dev_score = y_pred
        
        inlier_loss = torch.abs(dev_score)
        outlier_loss = torch.abs(torch.clamp(self.confidence_margin - dev_score, min=0.))
        
        recon_error_norm = torch.norm(recon_residual, p=2, dim=1, keepdim=True)
        outlier_sub_loss = torch.abs(torch.clamp(self.confidence_margin - recon_error_norm, min=0.))
        
        loss = (1 - y_true) * (inlier_loss + recon_error_norm) + y_true * (outlier_loss + outlier_sub_loss)
        
        return torch.mean(loss)

class AEDataset(Dataset):
    def __init__(self, x, inlier_indices):
        self.x = torch.from_numpy(x[inlier_indices]).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.x[idx]

class DevNetDataset(Dataset):
    def __init__(self, x, outlier_indices, inlier_indices):
        self.x = x
        self.outlier_indices = outlier_indices
        self.inlier_indices = inlier_indices
        
        # Create a list of all indices and corresponding labels
        self.indices = np.concatenate([inlier_indices, outlier_indices])
        self.labels = np.concatenate([np.zeros(len(inlier_indices)), np.ones(len(outlier_indices))])
        
        # For alternating sampling, we can prepare pairs
        self.num_inliers = len(inlier_indices)
        self.num_outliers = len(outlier_indices)
        self.total_len = self.num_inliers + self.num_outliers

    def __len__(self):
        # The length should be large enough to allow for multiple epochs of batch generation
        return self.total_len * 20 # Heuristic to match original batch logic

    def __getitem__(self, idx):
        # Alternate between inliers and outliers
        if idx % 2 == 0:
            # Inlier
            sid = np.random.choice(self.num_inliers)
            sample_idx = self.inlier_indices[sid]
            label = 0.0
        else:
            # Outlier
            sid = np.random.choice(self.num_outliers)
            sample_idx = self.outlier_indices[sid]
            label = 1.0
            
        sample = torch.from_numpy(self.x[sample_idx]).float()
        label = torch.tensor(label).float().unsqueeze(0)
        
        return sample, label

class FEAWAD:
    def __init__(self, seed, model_name='FEAWAD', save_suffix='test'):
        self.utils = Utils()
        self.device = self.utils.get_device(True)
        self.seed = seed
        self.utils.set_seed(seed)

        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--nb_batch", type=int, default=20)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--ae_epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.0001)
        self.args, _ = parser.parse_known_args()

        self.save_suffix = save_suffix
        self.modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        if not os.path.exists(self.modelpath):
            os.makedirs(self.modelpath)
            
        self.ae_model = None
        self.dev_model = None

    def fit(self, X_train, y_train, ratio=None):
        self.utils.set_seed(self.seed)
        
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]
        self.input_shape = X_train.shape[1:]
        input_dim = X_train.shape[1]

        # 1. Pre-train AutoEncoder
        print('AutoEncoder pre-training start....')
        self.ae_model = AutoEncoder(input_dim=input_dim).to(self.device)
        ae_dataset = AEDataset(X_train, inlier_indices)
        # Use a large batch size if dataset is small
        ae_batch_size = min(self.args.batch_size, len(ae_dataset))
        ae_dataloader = DataLoader(dataset=ae_dataset, batch_size=ae_batch_size, shuffle=True)
        
        optimizer_ae = torch.optim.Adam(self.ae_model.parameters(), lr=self.args.lr)
        criterion_ae = nn.MSELoss()

        for epoch in range(self.args.ae_epochs):
            self.ae_model.train()
            total_loss = 0
            for x, y in ae_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer_ae.zero_grad()
                _, decoded = self.ae_model(x)
                loss = criterion_ae(decoded, y)
                loss.backward()
                optimizer_ae.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'AE Epoch {epoch+1}/{self.args.ae_epochs}, Loss: {total_loss/len(ae_dataloader):.6f}')
        print('AutoEncoder pre-training finished.')

        # 2. Train Deviation Network
        print('End-to-end training start....')
        self.dev_model = DevNet(input_dim=input_dim).to(self.device)
        self.dev_model.load_ae_weights(self.ae_model) # Load weights from pre-trained AE
        
        dev_dataset = DevNetDataset(X_train, outlier_indices, inlier_indices)
        dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer_dev = torch.optim.Adam(self.dev_model.parameters(), lr=self.args.lr)

        best_loss = float('inf')
        best_state_dict = None
        for epoch in range(self.args.epochs):
            self.dev_model.train()
            total_loss = 0
            steps = 0
            for x, y in dev_dataloader:
                if steps >= self.args.nb_batch:
                    break
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer_dev.zero_grad()
                score, recon_residual = self.dev_model(x)
                loss = self.dev_model.loss_function(y, score, recon_residual)
                loss.backward()
                optimizer_dev.step()
                total_loss += loss.item()
                steps += 1
            
            epoch_loss = total_loss / steps
            if (epoch + 1) % 5 == 0:
                print(f'DevNet Epoch {epoch+1}/{self.args.epochs}, Loss: {epoch_loss:.6f}')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state_dict = deepcopy(self.dev_model.state_dict())
        
        print('End-to-end training finished.')
        # Load best model for prediction
        if best_state_dict is not None:
            self.dev_model.load_state_dict(best_state_dict)
        return self

    def predict_score(self, X):
        if self.dev_model is None:
            raise RuntimeError("The model has not been trained yet. Please call fit() first.")
        
        self.dev_model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        scores = []
        with torch.no_grad():
            # Process in batches to avoid OOM error on large test sets
            test_loader = DataLoader(dataset=X_tensor, batch_size=self.args.batch_size, shuffle=False)
            for batch_x in test_loader:
                score, _ = self.dev_model(batch_x)
                scores.append(score.cpu().numpy())
        
        return np.concatenate(scores, axis=0)
