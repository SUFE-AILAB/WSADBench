# -*- coding: utf-8 -*-
"""
Sultani方法训练逻辑
基于MIL (Multiple Instance Learning) 的弱监督异常检测训练
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, Optional, List
import time
# from WSADBench.baseline.VadClip.clip.myUtils import myLogger as logging
from WSADBench.baseline.VadClip.clip.myUtils import setup_logging
logger = setup_logging(log_dir='/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/logs', name='sultani')
from common_utils.baseline_utils import get_gt


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
        # 前半是异常段，后半为正常段
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

# @staticmethod
def _process_video_scores(scores, video_shape,y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames):
    """处理video分数的特殊逻辑：从clip级别还原到帧级别"""
    _clips_num, _crops_num = video_shape

    # 平均每个crop, 获得每个clip的分数
    scores = scores.reshape(_clips_num, _crops_num)
    scores = np.mean(scores, axis=1)

    # 还原clip级别score为帧级别score
    # y_test_idx = data["y_test_idx"]
    # y_test_gt, y_test_gt_idx = data["y_test_gt"], data["y_test_gt_idx"]
    # num_clip_frames = data["NUM_FRAMES"]

    frame_scores = []
    frame_truth = []
    for i in range(max(y_test_gt_idx) + 1):
        select_gt = y_test_gt[y_test_gt_idx == i]
        select_scores = scores[y_test_idx == i]
        select_scores = select_scores.repeat(num_clip_frames)
        common_length = min(len(select_gt), len(select_scores))

        frame_scores.append(select_scores[:common_length])
        frame_truth.append(select_gt[:common_length])

    frame_scores = np.concatenate(frame_scores, axis=0)
    frame_truth = np.concatenate(frame_truth, axis=0)
    # frame_truth存到本地
    # np.save("frame_label/xd_frame_gt.npy", frame_truth)
    return frame_scores, frame_truth

def _process_tabular_scores(scores, data_shape, y_test_idx, y_test_gt,
                           y_test_gt_idx, n_samples):
        """处理tabular_inexact分数的特殊逻辑：从bag级别还原到样本级别"""
        n_bags, n_samples = data_shape  # 取出 n_bags 和 n_samples

        # 平均每个样本，获得每个袋的分数
        scores = scores.reshape(n_bags, n_samples)
        scores = np.mean(scores, axis=1)

        sample_truth = y_test_gt
        sample_scores = scores.repeat(n_samples)
        # 对齐长度
        common_length = min(len(sample_truth), len(sample_scores))
        sample_scores = sample_scores[:common_length]
        sample_truth = sample_truth[:common_length]

        return sample_scores, sample_truth

def fit_sultani(model, optimizer, epochs, device, X_test, trainer,
                       verbose=True, normal_loader=None, anomaly_loader=None):
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
    sparsity_weight =  trainer.sparsity_weight
    smoothness_weight = trainer.smoothness_weight
    train_history = {
        'loss': [],
        'epoch_time': []
    }
    
    if verbose:
        print(f"开始训练Sultani模型，共{epochs}轮...")
        print(f"设备: {device}")
        print(f"稀疏性权重: {sparsity_weight}, 平滑性权重: {smoothness_weight}")
    # X_test, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames = X_test  # 拆包
    X_test, data_shape, y_test_idx, y_test_gt, y_test_gt_idx, n_samples = X_test  # 拆包
    best_epoch = -1
    best_auc = 0.0
    best_ap = 0
    best_epoch_v2 = -1
    best_auc_v2 = 0.0
    best_ap_v2 = 0

    logger.info('start train ...')
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, (normal_data, anomaly_data) in enumerate(zip(normal_loader, anomaly_loader)):
            optimizer.zero_grad()

            # 将数据移到设备
            normal_data = normal_data.to(device)  # [batch_size, crops_num, seq_len, feature_dim]
            anomaly_data = anomaly_data.to(device)  # [batch_size, crops_num, seq_len, feature_dim]

            batch_size, crops_num, seq_len, feature_dim = normal_data.shape
            # 沿着seq_len维度拼接，前seq_len个为异常，后seq_len个为正常
            inputs = torch.cat([anomaly_data, normal_data],
                               dim=2)  # [batch_size, crops_num, 2*seq_len, feature_dim]

            # 重塑为模型期望的输入格式
            inputs = inputs.view(batch_size * crops_num, 2 * seq_len, feature_dim)

            # 前向传播
            outputs = model(inputs)  # [batch_size * crops_num, 2*seq_len, 1]

            # 修正：重塑输出以匹配MIL损失期望的格式
            outputs = outputs.view(batch_size * crops_num, 2 * seq_len)  # [batch_size * crops_num, 2*seq_len]

            # 计算损失 - 使用实际的batch_size * crops_num
            loss = mil_loss(outputs, batch_size * crops_num, sparsity_weight, smoothness_weight)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        # 学习率调度
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')
        if X_test is not None and trainer.is_test:
            trainer.fitted = True
            # 处理video分数的特殊逻辑：从clip级别还原到帧级别
            with torch.no_grad():
                scores = trainer.predict_proba(X_test)  # 得分696270
                prob = np.repeat(scores, 16)
                gt = get_gt(len(prob))
                test_auc_v2 = roc_auc_score(gt, prob)
                test_ap_v2 = average_precision_score(gt, prob)
                # #为适配tabular_inexact数据集修改
                # n_bags, n_samples,_dim = X_test.shape
                # data_shape = (n_bags, n_samples)

                # frame_scores, frame_truth = _process_video_scores(scores, video_shape, y_test_idx, y_test_gt,
                #                                                   y_test_gt_idx,
                #                                                   num_clip_frames)

                frame_scores, frame_truth = _process_tabular_scores(scores, data_shape, y_test_idx,
                                                                     y_test_gt,y_test_gt_idx, n_samples)
                test_auc = roc_auc_score(frame_truth, frame_scores)
                test_ap = average_precision_score(frame_truth, frame_scores)
                if best_auc < test_auc:
                    best_epoch = epoch
                    best_auc = test_auc
                    best_ap = test_ap
                if best_auc_v2 < test_auc_v2:
                    best_epoch_v2 = epoch
                    best_auc_v2 = test_auc_v2
                    best_ap_v2 = test_ap_v2
                logger.info(
                    f"cur epoch:{epoch} AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f} best epoch:{best_epoch}, best auc:{best_auc:.4f}, best ap:{best_ap:4f}")
                logger.info(
                    f"cur epoch_v2:{epoch} AUCROC_v2: {test_auc_v2:.4f}, AUCPR_v2: {test_ap_v2:.4f} best epoch_v2:{best_epoch_v2}, best auc_v2:{best_auc_v2:.4f}, best ap_v2:{best_ap_v2:4f}")

    if verbose:
        print("训练完成！")
    
    return train_history
