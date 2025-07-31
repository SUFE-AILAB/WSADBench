# -*- coding: utf-8 -*-
"""
TargAD方法模型架构定义
基于"A Robust Prioritized Anomaly Detection when Not
 All Anomalies are of Primary Interest"论文实现
弱监督异常检测方法
"""
from enum import auto
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, einsum
from einops import rearrange

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


#自编码，模型第一个板块

class AutoEncoder(nn.Module):

    def __init__(self, input_dim,  num_features):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim       # 输入特征维度
        self.num_features = num_features   # 嵌入特征维度
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, num_features),
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
        
        # -----model initialization----- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # -----feature embedding----- #
        x = x.view(-1, self.input_dim)     #报错
        x_e = self.encoder(x)  #[256,10] -> [256,64]
        x_de = self.decoder(x_e) #[256,64] -> [256,10]
        x_de = x_de.view(-1, 1, self.input_dim)   #[256,1,10]
        return x_e, x_de
    
# TargAD模型，第二个板块，分类输出
class Classifier(nn.Module):

    def __init__(self, input_dim,  num_features, num_classes):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.encoder = nn.Sequential(
#             nn.Dropout(p=0.1),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(256, num_features)
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.3),
            nn.Linear(num_features, 32),
            nn.ReLU(),
#             nn.Dropout(p=0.3),
            nn.Linear(32, num_classes)
        )
        # -----model initialization----- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # -----feature embedding----- #
        x = x.view(-1, self.input_dim)    #[128,10]
        x_e = self.encoder(x) #[128,10] -> [128,64]
        tmp = self.classifier[0](x_e) #[128,64] -> [128,32]
        self.hidden = self.classifier[1](tmp)
        y_logit = self.classifier(x_e)   # [128,64] -> [128,8]

        return x_e, y_logit        #预测输出

#兼容命名
# model1 = AutoEncoder
model = Classifier
autoencoder = AutoEncoder