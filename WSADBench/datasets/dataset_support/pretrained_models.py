#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型模块
包含各种用于视频特征提取的预训练模型
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 检查PytorchVideo可用性
try:
    from pytorchvideo.models.hub import i3d_r50
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False
    logger.warning("PytorchVideo not available")


class I3DFeatureExtractorRGB(nn.Module):
    """I3D RGB模态特征提取器，提取分类层前的特征"""
    
    feature_dim = 2048  # I3D特征维度
    
    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        super().__init__()
        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError("PytorchVideo is required for I3D model")
        
        # 加载完整的I3D模型
        self.full_model = i3d_r50(pretrained=pretrained)
        
        # 如果提供了自定义权重，加载它们
        if weights_path:
            self._load_custom_weights(weights_path)
        
        self.features = None
        self._register_hooks()
        
        logger.info(f"I3D RGB feature extractor initialized (pretrained={pretrained})")
    
    def _load_custom_weights(self, weights_path: str):
        """加载自定义权重"""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.full_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded custom weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load custom weights from {weights_path}: {e}")
    
    def _register_hooks(self):
        """注册hooks来提取分类层前的特征"""
        def feature_hook(module, input, output):
            # 保存特征（在全连接层之前，AvgPool3d之后）
            if len(output.shape) == 2:  # [batch_size, feature_dim]
                self.features = output
            elif len(output.shape) > 2:
                # 如果是多维输出，应用全局平均池化
                dims_to_pool = list(range(2, len(output.shape)))
                self.features = torch.mean(output, dim=dims_to_pool)
        
        # 精确定位到 blocks.6.pool 层
        hook_registered = False
        for name, module in self.full_model.named_modules():
            if name == 'blocks.6.pool':
                module.register_forward_hook(feature_hook)
                logger.info(f"Registered hook at I3D feature extraction layer: {name}")
                hook_registered = True
                break
        
        if not hook_registered:
            raise RuntimeError("Failed to register hook for I3D feature extraction layer")
    
    def forward(self, x):
        """
        前向传播提取特征
        
        Args:
            x: 输入视频张量 [batch_size, channels, frames, height, width]
            
        Returns:
            特征张量 [batch_size, feature_dim] (通常是2048维)
        """
        # 重置特征
        self.features = None
        
        # 前向传播
        _ = self.full_model(x)
        
        if self.features is None:
            raise RuntimeError("Feature extraction failed - hook did not capture features")
        
        return self.features


class I3DFeatureExtractorFlow(nn.Module):
    """I3D Flow模态特征提取器（暂未实现，保留接口）"""
    
    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        super().__init__()
        # TODO: 实现Flow模态的I3D特征提取器
        raise NotImplementedError("Flow modality I3D feature extractor is not implemented yet")
    
    def forward(self, x):
        raise NotImplementedError("Flow modality I3D feature extractor is not implemented yet")


def get_model_class(model_class_path: str):
    """
    动态导入并返回模型类
    
    Args:
        model_class_path: 模型类的完整路径，格式如 'module.submodule.ClassName'
        
    Returns:
        模型类
    """
    try:
        module_path, class_name = model_class_path.rsplit('.', 1)
        
        # 动态导入模块
        import importlib
        module = importlib.import_module(module_path)
        
        # 获取类
        model_class = getattr(module, class_name)
        
        return model_class
        
    except Exception as e:
        raise ImportError(f"Failed to import model class {model_class_path}: {e}")