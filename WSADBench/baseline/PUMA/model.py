# -*- coding: utf-8 -*-
"""
PUMA方法模型架构定义
基于" Learning from Positive and Unlabeled Multi-Instance Bags in Anomaly Detection."论文实现
Multiple Instance Learning (MIL) 弱监督异常检测方法
"""

from __future__ import division
from __future__ import print_function

import torch
#from torch import nn
#from multiprocessing import Pool, freeze_support, cpu_count, set_start_method
import torch
import torch.nn as nn

import numpy as np
from pyod.utils.torch_utility import get_activation_by_name

#X_bags -> X  ; Y_bags -> y

class PyODDataset(torch.utils.data.Dataset):

    def __init__(self, X_bags, y=None,mean=np.zeros(1,np.float32), std=np.ones(1,np.float32)):
        super(PyODDataset, self).__init__()

        self.n_bags = X_bags.shape[0]
        self.n_samples = X_bags.shape[1]
        self.input_dim = X_bags.shape[2]
        self.X_bags = X_bags
        self.X_inst = X_bags.reshape(self.n_bags * self.n_samples,self.input_dim)
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X_bags.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X_bags[idx, :, :]
        sample = (sample - self.mean) / self.std

        return torch.from_numpy(sample), idx

class inner_autoencoder(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_neurons=[128, 64],
                 dropout_rate=0.2,
                 batch_norm=True,
                 hidden_activation='relu'):
        super(inner_autoencoder, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation

        self.activation = get_activation_by_name(hidden_activation)

        self.layers_neurons_ = [self.input_dim, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if batch_norm:
                self.encoder.add_module("batch_norm"+str(idx),torch.nn.BatchNorm1d(self.layers_neurons_[idx]))
            self.encoder.add_module("linear"+str(idx),torch.nn.Linear(self.layers_neurons_[idx],self.layers_neurons_[idx+1]))
            self.encoder.add_module(self.hidden_activation+str(idx),self.activation)
            self.encoder.add_module("dropout"+str(idx),torch.nn.Dropout(dropout_rate))

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if batch_norm:
                self.decoder.add_module("batch_norm"+str(idx),torch.nn.BatchNorm1d(self.layers_neurons_decoder_[idx]))
            self.decoder.add_module("linear"+str(idx),torch.nn.Linear(self.layers_neurons_decoder_[idx],
                                                                      self.layers_neurons_decoder_[idx+1]))
            self.encoder.add_module(self.hidden_activation+str(idx),self.activation)
            self.decoder.add_module("dropout"+str(idx),torch.nn.Dropout(dropout_rate))

    def forward(self, x):
        # we could return the latent representation here after the encoder as the latent representation
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def get_vars(self):
        """获取模型参数列表"""
        return self.vars



def init_weights_xavier(module):
    """
    Xavier权重初始化函数
    
    Args:
        module: 神经网络模块
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#兼容
model = inner_autoencoder