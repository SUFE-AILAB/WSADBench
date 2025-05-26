# -*- coding: utf-8 -*-
"""
Sultani方法包初始化文件
基于"Real-world Anomaly Detection in Surveillance Videos"论文实现
"""

from .run import Sultani
from .model import SultaniLearner, SultaniFeatureExtractor, Learner
from .fit import fit_sultani, fit_sultani_simple, mil_loss, create_mil_dataloader

__all__ = [
    'Sultani',
    'SultaniLearner', 
    'SultaniFeatureExtractor',
    'Learner',  # 兼容别名
    'fit_sultani',
    'fit_sultani_simple', 
    'mil_loss',
    'create_mil_dataloader'
]

__version__ = "1.0.0"
__author__ = "WSADBench Team"
__description__ = "Sultani方法实现 - 基于MIL的弱监督视频异常检测"
