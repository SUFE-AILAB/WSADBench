# -*- coding: utf-8 -*-
import numpy as np
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import KDTree
from sklearn.utils.random import sample_without_replacement
from copy import deepcopy

warnings.simplefilter("ignore")

MAX_INT = np.iinfo(np.int32).max

def sqr_euclidean_dist(x, y):
    return torch.sum(torch.square(x - y), dim=-1)

def triplet_ranking_loss(input_example, input_positive, input_negative, confidence_margin):
    positive_distances = sqr_euclidean_dist(input_example, input_positive)
    negative_distances = sqr_euclidean_dist(input_example, input_negative)
    loss = torch.mean(torch.clamp(confidence_margin - (negative_distances - positive_distances), min=0.))
    return loss

class Repen_network(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Repen_network, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, input_data):
        return self.activation(self.hidden_layer(input_data))

class Trainer:
    def __init__(self, n_epochs=50, batch_size=256,
                 nb_batch=100, random_seed=42, device='cpu'):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.rng = np.random.RandomState(random_seed)
        self.device = device

    def batch_generator(self, X, positive_weights,
                        negative_weights, inlier_ids,
                        outlier_ids):
        rng = np.random.RandomState(self.rng.randint(MAX_INT, size=1))
        counter = 0
        while counter < self.nb_batch:
            examples_idx = rng.choice(inlier_ids, self.batch_size, p=positive_weights)
            positives_idx = rng.choice(inlier_ids, self.batch_size)
            
            # Ensure example and positive are not the same
            inds_to_change = np.where(examples_idx == positives_idx)[0]
            while len(inds_to_change) > 0:
                new_positives = rng.choice(inlier_ids, len(inds_to_change))
                positives_idx[inds_to_change] = new_positives
                inds_to_change = np.where(examples_idx[inds_to_change] == positives_idx[inds_to_change])[0]

            if isinstance(outlier_ids, list) and len(outlier_ids) == 2:
                neg_1_count = int(self.batch_size / 2)
                neg_2_count = self.batch_size - neg_1_count
                neg_1_idx = rng.choice(outlier_ids[0], neg_1_count, p=negative_weights[0])
                neg_2_idx = rng.choice(outlier_ids[1], neg_2_count, p=negative_weights[1])
                negatives_idx = np.hstack([neg_1_idx, neg_2_idx])
            else:
                negatives_idx = rng.choice(outlier_ids, self.batch_size, p=negative_weights)

            yield (torch.from_numpy(X[examples_idx]).float(),
                   torch.from_numpy(X[positives_idx]).float(),
                   torch.from_numpy(X[negatives_idx]).float())
            counter += 1

    def train(self, network, confidence_margin, x_train,
              positive_weights, negative_weights,
              inlier_indices, outlier_indices,
              verbose=True):

        optimizer = torch.optim.Adadelta(network.parameters())
        
        best_loss = float('inf')
        best_model_state = None

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            network.train()
            
            batch_gen = self.batch_generator(x_train, positive_weights, negative_weights, inlier_indices, outlier_indices)
            
            for i, (input_e, input_p, input_n) in enumerate(batch_gen):
                input_e, input_p, input_n = input_e.to(self.device), input_p.to(self.device), input_n.to(self.device)

                optimizer.zero_grad()

                hidden_e = network(input_e)
                hidden_p = network(input_p)
                hidden_n = network(input_n)

                loss = triplet_ranking_loss(hidden_e, hidden_p, hidden_n, confidence_margin)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            epoch_loss /= self.nb_batch
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = deepcopy(network.state_dict())

        # Load best model
        if best_model_state:
            network.load_state_dict(best_model_state)
        return network


class repen:
    def __init__(self, n_epochs=50, batch_size=256, n_neighbors=2,
                 nb_batch=100, random_seed=42,
                 mode="semi_supervised", known_outliers=10, hidden_dim=20,
                 confidence_margin=1000.0, device='cpu'):

        assert (mode in ["semi_supervised", "unsupervised", "supervised"])
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.known_outliers = known_outliers
        self.confidence_margin = confidence_margin
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.Trainer = Trainer(n_epochs, batch_size, nb_batch, random_seed, device)
        self.network = None # Will be initialized in fit

    def prepare_data(self, x_train, y_train=None):
        if self.mode == "unsupervised":
            outlier_scores = self.lesinn(x_train, x_train)
            ind_scores = np.argsort(outlier_scores.flatten())
            inlier_ids, outlier_ids = ind_scores[:-self.known_outliers], ind_scores[-self.known_outliers:]
            
            transforms = np.sum(outlier_scores[inlier_ids]) - outlier_scores[inlier_ids]
            total_weights_p = np.sum(transforms)
            positive_weights = (transforms / total_weights_p).flatten()
            
            total_weights_n = np.sum(outlier_scores[outlier_ids])
            negative_weights = (outlier_scores[outlier_ids] / total_weights_n).flatten()

        elif self.mode == "semi_supervised":
            outlier_ids_labeled = np.where(y_train == 1)[0]
            outlier_scores = self.lesinn(x_train, x_train)

            if outlier_ids_labeled.shape[0] < self.known_outliers:
                ind_scores = np.argsort(outlier_scores.flatten())
                ind_scores = [elt for elt in ind_scores if elt not in outlier_ids_labeled]
                mn = self.known_outliers - outlier_ids_labeled.shape[0]
                to_add_idx = ind_scores[-mn:]

                total_weights_n = np.sum(outlier_scores[to_add_idx])
                neg_weights_unlabeled = (outlier_scores[to_add_idx] / total_weights_n).flatten()
                neg_weights_labeled = np.ones(outlier_ids_labeled.shape[0]) / outlier_ids_labeled.shape[0]
                
                negative_weights = [neg_weights_unlabeled, neg_weights_labeled]
                outlier_ids = [to_add_idx, outlier_ids_labeled]
            else:
                outlier_ids = outlier_ids_labeled
                negative_weights = np.ones(outlier_ids.shape[0]) / outlier_ids.shape[0]


            all_outliers = np.hstack(outlier_ids) if isinstance(outlier_ids, list) else outlier_ids
            inlier_ids = np.delete(np.arange(len(x_train)), all_outliers, axis=0)
            
            transforms = np.sum(outlier_scores[inlier_ids]) - outlier_scores[inlier_ids]
            total_weights_p = np.sum(transforms)
            positive_weights = (transforms / total_weights_p).flatten()

        else: # supervised
            outlier_ids = np.where(y_train == 1)[0]
            inlier_ids = np.where(y_train == 0)[0]
            
            if outlier_ids.shape[0] > self.known_outliers:
                mn = outlier_ids.shape[0] - self.known_outliers
                remove_idx = self.Trainer.rng.choice(outlier_ids, mn, replace=False)
                outlier_ids = np.setdiff1d(outlier_ids, remove_idx)

            positive_weights = np.ones(inlier_ids.shape[0]) / inlier_ids.shape[0]
            negative_weights = np.ones(outlier_ids.shape[0]) / outlier_ids.shape[0]

        self.inlier_ids = inlier_ids
        self.outlier_ids = outlier_ids
        self.positive_weights = positive_weights
        self.negative_weights = negative_weights

    def lesinn(self, x_train, to_query):
        ensemble_size = 50
        subsample_size = 8
        scores = np.zeros([to_query.shape[0], 1])
        seeds = self.Trainer.rng.randint(MAX_INT, size=ensemble_size)
        for i in range(ensemble_size):
            rs = np.random.RandomState(seeds[i])
            sid = sample_without_replacement(n_population=x_train.shape[0],
                                             n_samples=subsample_size,
                                             random_state=rs)
            subsample = x_train[sid]
            kdt = KDTree(subsample, metric='euclidean')
            dists, _ = kdt.query(to_query, k=self.n_neighbors)
            scores += np.mean(dists, axis=1)[:, np.newaxis]
        return scores / ensemble_size

    def fit(self, x_train, y_train=None, verbose=False):
        self.x_train = x_train
        self.prepare_data(x_train, y_train)
        
        self.network = Repen_network(input_dim=x_train.shape[1], hidden_dim=self.hidden_dim).to(self.device)
        
        self.network = self.Trainer.train(self.network, self.confidence_margin, x_train,
                                          self.positive_weights, self.negative_weights,
                                          self.inlier_ids, self.outlier_ids,
                                          verbose=verbose)

    def decision_function(self, x_val):
        self.network.eval()
        with torch.no_grad():
            x_train_tensor = torch.from_numpy(self.x_train).float().to(self.device)
            x_val_tensor = torch.from_numpy(x_val).float().to(self.device)
            
            hidden_features_tr = self.network(x_train_tensor).cpu().numpy()
            hidden_features_val = self.network(x_val_tensor).cpu().numpy()
            
        scores = self.lesinn(hidden_features_tr, hidden_features_val)
        return scores.flatten()
