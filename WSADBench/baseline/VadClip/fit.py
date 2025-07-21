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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from WSADBench.baseline.VadClip.model import process_split, process_feat

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1

            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors
def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text
def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                      'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                      'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                      'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})
prompt_text = get_prompt_text(label_map)

def fit_sultani(model, optimizer, norm_loader,anomaly_loader, epochs, device,
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

    # model.device = device  # 设置设备
    if verbose:
        print(f"开始训练VadClip模型，共{epochs}轮...")
        print(f"设备: {device}")
        # print(f"稀疏性权重: {sparsity_weight}, 平滑性权重: {smoothness_weight}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        loss_total1 = 0
        loss_total2 = 0
        for batch_idx, (normal_data, anomaly_data) in enumerate(zip(norm_loader, anomaly_loader)):
            step = 0
            optimizer.zero_grad()
            normal_features, normal_label, normal_lengths = normal_data  # 64个正常视频，64个异常视频，加起来是128个
            anomaly_features, anomaly_label, anomaly_lengths = anomaly_data
            # 将数据移到设备

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # vis_fea: [batch, seg:256, feature:2048], prompt:14个标签组成的list， feat_len:[batch]标签有效长度
            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)

            # loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            # loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            # loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += batch_idx * visual_features.shape[0]
            epoch_loss += loss1.item()
            batch_count += 1
            
            if verbose and batch_idx % 10 == 0:
                # print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss1: {loss1.item():.4f}， Loss2: {loss2.item():.4f}， Loss3: {loss3.item():.4f}')
                print(
                    f'epoch: {epoch + 1}| step: {step},| loss1: {loss_total1 / (batch_idx + 1):.4f} loss2: {loss_total2 / (batch_idx + 1):.4f}, loss3:{loss3.item():.4f}')
            # break
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


# 5. 自定义 Dataset
class VideoDataset(Dataset):
    def __init__(self, features, vid_kind, clip_num, seg, is_test=False):
        self.is_test = is_test
        self.seg = seg
        self.features = features
        self.vid_kind = vid_kind  # 保留为 str list
        self.clip_num = clip_num # 保留为 int list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.is_test:
            feat, _ = process_split(self.features[idx], self.seg)
            return feat, self.vid_kind[idx], self.clip_num[idx]
        else:
            feat, _ = process_feat(self.features[idx], self.seg)
            return feat, self.vid_kind[idx], self.clip_num[idx]

def fit_sultani_main(X_train, y_train, model, optimizer, epochs, batch_size, device,
                       verbose=True, vid_info=None, clip_num=None, vid_kind=None, crops_num=None):
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
        vid_info: 每个片段对应的视频id [n_samples]

    Returns:
        训练历史

    """
    model.train()
    ncrop = crops_num
    seg = model.visual_length
    feature = model.input_dim

    def split_by_seg_list(X, seg_list, feature):
        X = X.reshape(-1, ncrop, feature)  # 【seg, crop:10, 2048]
        segments = []
        start = 0
        for seg_len in seg_list:
            end = start + seg_len
            for i in range(ncrop):
                segment = X[start:end, i]  # shape: [1, seg_len, 2048]
                segments.append(segment)
            start = end
        return segments

    X_train = split_by_seg_list(X_train, clip_num.values(), feature)  # (16100, seg, 2048)
    y_train = split_by_seg_list(y_train, clip_num.values(), 1)  # 16100个(, seg, 1)(mask)
    y_train = [int(item[0, 0]) for item in y_train]  # 16100个0、1标签list
    # y_train = y_train.reshape(-1)  # (16100,)

    # 消除 y_train 的重复，取每组的第一个标签
    # reshape y_train to match the reshaped X_train
    # y_train = y_train.reshape(-1, ncrop)  # [1610*seg, 10]
    # y_train = y_train.reshape(-1, seg, ncrop)  # [num, 32, 10]
    # # y_train = y_train.transpose(0, 2, 1) # [num, 10, 32]
    # y_train = y_train[:, 0, :].reshape(-1)   # shape: (num*10)
    # 按y_train为mask，把X_train拆分为normal和 anomaly两个X；并拆分其对应的vid_kind, clip_num
    # 3. 使用 y_train 作为 mask，划分正常/异常（0=normal，1=anomaly）
    # normal_mask = y_train == 0
    # abnormal_mask = y_train == 1

    X_normal_videos = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
    X_anomaly_videos = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1]
    vid_kind_expand = [vid_kind[i] for i in range(len(vid_kind)) for _ in range(ncrop)]  # 包含str标签，并复制
    clip_num_expand = [min(seg, clip_num[i]) for i in range(len(clip_num)) for _ in range(ncrop)]  # 注意上界
    # 4. 对应扩展 vid_kind 和 clip_num：每个视频被切成了10段
    # 4. 用 mask 提取对应部分
    normal_vid_kind = [vid_kind_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
    abnormal_vid_kind = [vid_kind_expand[i] for i in range(len(y_train)) if y_train[i] == 1]

    normal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
    abnormal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 1]




    # 6. 构建 Dataset 和 DataLoader
    normal_dataset = VideoDataset(X_normal_videos, normal_vid_kind, normal_clip_num, seg)
    anomaly_dataset = VideoDataset(X_anomaly_videos, abnormal_vid_kind, abnormal_clip_num, seg)

    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=6)


    
    # 调用主训练函数
    return fit_sultani(model, optimizer, normal_loader,anomaly_loader, epochs, device,
                       verbose,)