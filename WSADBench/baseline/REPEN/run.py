import numpy as np
from WSADBench.baseline.REPEN.model import repen
from WSADBench.myutils import Utils
import os
import torch

class REPEN():
    def __init__(self, seed, model_name='REPEN', save_suffix='test',
                 mode:str='supervised', hidden_dim:int=20, batch_size:int=256, nb_batch:int=50, n_epochs:int=1000,
                 verbose:bool=True):
        self.utils = Utils()
        self.device = self.utils.get_device(True)
        self.seed = seed

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.n_epochs = n_epochs

        self.save_suffix = save_suffix
            
        self.model = None
        self.verbose = verbose

    def fit(self, X_train, y_train, ratio=None):
        self.utils.set_seed(self.seed)

        if sum(y_train) == 0:
            self.mode = 'unsupervised'

        self.model = repen(mode=self.mode, hidden_dim=self.hidden_dim, batch_size=self.batch_size, 
                           nb_batch=self.nb_batch, n_epochs=self.n_epochs, 
                           known_outliers=1000000, device=self.device)

        self.model.fit(X_train, y_train, verbose=self.verbose)
        return self

    def predict_score(self, X):
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self.model.decision_function(X)

    def parameter_count(self):
        if self.model and self.model.network:
            total_params = sum(p.numel() for p in self.model.network.parameters())
            trainable_params = sum(p.numel() for p in self.model.network.parameters() if p.requires_grad)
            return {
                'network_total': total_params,
                'network_trainable': trainable_params,
                'network_non_trainable': total_params - trainable_params,
                'total': total_params
            }
        else:
            # Estimate params before fitting
            # Note: This assumes input_dim is known or can be estimated.
            # Here, we can't know input_dim before fit, so we return a note.
            # A temporary solution could be to create a dummy network if hidden_dim is known.
            # For now, let's indicate it's not available until fit.
            input_dim_placeholder = 100 # A placeholder
            temp_network = torch.nn.Sequential(
                torch.nn.Linear(input_dim_placeholder, self.hidden_dim),
                torch.nn.ReLU()
            )
            total_params = sum(p.numel() for p in temp_network.parameters())
            return {
                'network_total': total_params,
                'network_trainable': total_params,
                'network_non_trainable': 0,
                'total': total_params,
                'note': f'Parameters estimated for a placeholder input_dim={input_dim_placeholder}'
            }

