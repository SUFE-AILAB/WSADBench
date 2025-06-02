# -*- coding: utf-8 -*-
"""
Sultani方法训练逻辑
基于MIL (Multiple Instance Learning) 的弱监督异常检测训练
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, Optional, List
import time


def mil_loss(y_pred, batch_size, sparsity_weight=0.00008, smoothness_weight=0.00008):
    """
    MIL损失函数
    基于原始Sultani论文的损失函数实现
    
    Args:
        y_pred: 预测分数 [batch_size, seq_len]
        batch_size: 批量大小
        sparsity_weight: 稀疏性损失权重
        smoothness_weight: 平滑性损失权重
        
    Returns:
        总损失
    """
    device = y_pred.device
    
    # 重塑预测结果
    y_pred = y_pred.view(batch_size, -1)
    
    loss = torch.tensor(0., device=device)
    sparsity_loss = torch.tensor(0., device=device)
    smoothness_loss = torch.tensor(0., device=device)
    
    for i in range(batch_size):
        # 假设前32个为异常段，后32个为正常段
        seq_len = y_pred.shape[1]
        mid_point = seq_len // 2
        
        # 随机排列增强训练
        anomaly_indices = torch.randperm(mid_point, device=device)
        normal_indices = torch.randperm(seq_len - mid_point, device=device)
        
        y_anomaly = y_pred[i, :mid_point][anomaly_indices]
        y_normal = y_pred[i, mid_point:][normal_indices]
        
        # 获取最大最小值
        y_anomaly_max = torch.max(y_anomaly)
        y_normal_max = torch.max(y_normal)
        
        # MIL ranking loss
        loss += F.relu(1.0 - y_anomaly_max + y_normal_max)
        
        # 稀疏性损失：鼓励异常分数稀疏
        sparsity_loss += torch.sum(y_anomaly) * sparsity_weight
        
        # 平滑性损失：鼓励相邻帧分数平滑
        if seq_len > 1:
            diff = y_pred[i, :-1] - y_pred[i, 1:]
            smoothness_loss += torch.sum(diff ** 2) * smoothness_weight
    
    # 平均损失
    total_loss = (loss + sparsity_loss + smoothness_loss) / batch_size
    
    return total_loss


def fit_sultani(model, optimizer, train_loader, epochs, device, 
                sparsity_weight=0.00008, smoothness_weight=0.00008,
                verbose=True, scheduler=None):
    """
    训练Sultani模型
    
    Args:
        model: Sultani模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        epochs: 训练轮数
        device: 计算设备
        sparsity_weight: 稀疏性损失权重
        smoothness_weight: 平滑性损失权重
        verbose: 是否打印训练信息
        scheduler: 学习率调度器（可选）
        
    Returns:
        训练历史
    """
    model.train()
    
    train_history = {
        'loss': [],
        'epoch_time': []
    }
    
    if verbose:
        print(f"开始训练Sultani模型，共{epochs}轮...")
        print(f"设备: {device}")
        print(f"稀疏性权重: {sparsity_weight}, 平滑性权重: {smoothness_weight}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (normal_data, anomaly_data) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 将数据移到设备
            normal_data = normal_data.to(device)
            anomaly_data = anomaly_data.to(device)
            
            batch_size = normal_data.shape[0]
            
            # 合并正常和异常数据 [batch_size, 64, feature_dim]
            # 前32个为异常，后32个为正常
            inputs = torch.cat([anomaly_data, normal_data], dim=1)
            
            # 前向传播
            outputs = model(inputs)  # [batch_size, 64, 1]
            
            # 计算损失
            loss = mil_loss(outputs, batch_size, sparsity_weight, smoothness_weight)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')
    
    if verbose:
        print("训练完成！")
    
    return train_history


def fit_sultani_main(X_train, y_train, model, optimizer, epochs, batch_size, device,
                      sparsity_weight=0.00008, smoothness_weight=0.00008, verbose=True):
    """
    Args:
        X_train: 训练特征 [n_samples, feature_dim]
        y_train: 训练标签 [n_samples]
        model: Sultani模型
        optimizer: 优化器
        epochs: 训练轮数
        batch_size: 批量大小
        device: 计算设备
        sparsity_weight: 稀疏性损失权重
        smoothness_weight: 平滑性损失权重
        verbose: 是否打印训练信息
        
    Returns:
        训练历史
    """
    model.train()
    
    # 分离正常和异常数据
    normal_mask = y_train == 0
    anomaly_mask = y_train == 1
    
    X_normal = X_train[normal_mask]
    X_anomaly = X_train[anomaly_mask]
    
    if len(X_anomaly) == 0:
        raise ValueError("训练数据中没有异常样本！")
    
    # 创建数据加载器
    # 假设每个视频有32个片段
    segments_per_video = 32
    
    # 重塑数据为视频段格式
    n_normal_videos = len(X_normal) // segments_per_video
    n_anomaly_videos = len(X_anomaly) // segments_per_video
    
    if n_normal_videos == 0 or n_anomaly_videos == 0:
        if verbose:
            print("警告: 样本数量不足，使用重复采样...")
        # 重复采样以确保有足够的数据
        min_videos = max(1, min(len(X_normal), len(X_anomaly)) // segments_per_video)
        
        X_normal_videos = X_normal[:min_videos * segments_per_video].reshape(min_videos, segments_per_video, -1)
        X_anomaly_videos = X_anomaly[:min_videos * segments_per_video].reshape(min_videos, segments_per_video, -1)
    else:
        min_videos = min(n_normal_videos, n_anomaly_videos)
        X_normal_videos = X_normal[:min_videos * segments_per_video].reshape(min_videos, segments_per_video, -1)
        X_anomaly_videos = X_anomaly[:min_videos * segments_per_video].reshape(min_videos, segments_per_video, -1)
    
    # 创建训练数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_normal_videos),
        torch.FloatTensor(X_anomaly_videos)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 调用主训练函数
    return fit_sultani(model, optimizer, train_loader, epochs, device,
                      sparsity_weight, smoothness_weight, verbose)