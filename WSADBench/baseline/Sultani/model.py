# -*- coding: utf-8 -*-
"""
Sultani方法模型架构定义
基于"Real-world Anomaly Detection in Surveillance Videos"论文实现
Multiple Instance Learning (MIL) 弱监督异常检测方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SultaniLearner(nn.Module):
    """
    Sultani MIL学习器
    基于原始实现的深度MIL分类器
    """
    
    def __init__(self, input_dim: int = 2048, drop_p: float = 0.6):
        """
        初始化Sultani学习器
        
        Args:
            input_dim: 输入特征维度（默认2048，对应ResNet特征）
            drop_p: Dropout概率
        """
        super(SultaniLearner, self).__init__()
        
        self.input_dim = input_dim
        self.drop_p = drop_p
        
        # 三层MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(), 
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._weight_init()
        
        # 创建参数列表用于meta learning（可选）
        self.vars = nn.ParameterList()
        for param in self.classifier.parameters():
            self.vars.append(param)
    
    def _weight_init(self):
        """Xavier正态分布初始化权重"""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, vars=None):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]
            vars: 可选的参数列表（用于meta learning）
            
        Returns:
            异常分数 [batch_size, 1] 或 [batch_size, seq_len, 1]
        """
        if vars is None:
            # 标准前向传播
            return self.classifier(x)
        else:
            # 使用自定义参数的前向传播（用于meta learning）
            x = F.linear(x, vars[0], vars[1])
            x = F.relu(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            x = F.linear(x, vars[2], vars[3])
            x = F.relu(x)
            x = F.dropout(x, self.drop_p, training=self.training)
            x = F.linear(x, vars[4], vars[5])
            return torch.sigmoid(x)
    
    def get_vars(self):
        """获取模型参数列表"""
        return self.vars


class SultaniFeatureExtractor(nn.Module):
    """
    特征提取器（可选）
    如果输入不是预提取的特征，可以使用此模块进行特征提取
    """
    
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        """
        初始化特征提取器
        
        Args:
            backbone: 骨干网络类型
            pretrained: 是否使用预训练权重
        """
        super(SultaniFeatureExtractor, self).__init__()
        
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=pretrained)
            # 移除最后的分类层
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 512
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        """
        提取特征
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            特征向量 [batch_size, feature_dim]
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return features


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


# 兼容性别名
Learner = SultaniLearner  # 兼容原始实现的命名
