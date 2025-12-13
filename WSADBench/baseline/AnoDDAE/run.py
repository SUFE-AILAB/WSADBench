import argparse
import yaml
import numpy as np
import torch
from WSADBench.myutils import Utils
# from src.model import AnomalyDetector
from WSADBench.baseline.AnoDDAE.src.data import load_data, split_data
from WSADBench.baseline.AnoDDAE.src.utils import set_seed, normalize_data, evaluate_anomaly_detection, get_batch_size,get_device
from WSADBench.baseline.AnoDDAE.src.model import DDAE, DiffusionScheduler


# def parse_args():
#     parser = argparse.ArgumentParser(description="Run anomaly detection experiment.")
#     parser.add_argument('--config', type=str, default='src/config.yaml',
#                         help='Path to the config file.')
#     return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class AnoDDAE:
    def __init__(self,seed,model_name='AnoDDAE',
                 input_dim=None):
        
        self.model=None
        self.input_dim = input_dim
        self.config = load_config("WSADBench/baseline/AnoDDAE/src/config.yaml")
        # seed = config.get('seed', 111)
        self.seed = seed
        self.utils = Utils()
        self.device = self.utils.get_device(True)


    def fit(self,X_train,y_train,X_test=None,y_test=None):

        #normal only
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

        # Initialize and train model
        self.model = DDAE(
            input_dim= X_train.shape[1],
            hidden_dim= self.config['model']['hidden_dim'],
            activation= self.config['model']['activation'],
            num_timesteps= self.config['diffusion']['num_timesteps'],
            beta_start= self.config['diffusion']['beta_start'],
            beta_end= self.config['diffusion']['beta_end'],
            scheduler= self.config['diffusion']['scheduler'],
            time_emb_dim= self.config['diffusion']['time_emb_dim'],
            time_emb_type= self.config['diffusion']['time_emb_type'],
            epochs= self.config['train']['epochs'],
            batch_size= get_batch_size(X_train.shape[0]),
            learning_rate= self.config['train']['lr'],
            eval_epochs= self.config['train']['eval_epochs'],
            device =self.device
            )
        
        print("Batch size:", get_batch_size(X_train.shape[0]))
        self.model.fit(X_train,x_test=None,y_train=None,y_test=None)
        print("train finished")

        return self
        
    def predict_score(self,X_test):

    # Predict anomaly scores
        scores = self.model.predict(X_test)
        scores = scores.cpu().numpy()

        return scores
