# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

def norm(data):
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)


class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}

        triplet = result["triplet_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]
        b = _label.size(0) // 2
        t = att.size(1)
        anomaly = torch.topk(att, t // 16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, t // 16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())

        A_att = torch.topk(A_att, t // 16 + 1, dim=-1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())

        N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())
        A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())

        cost = anomaly_loss + 0.1 * (
                    A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss

        return cost, loss

def fit(model, optimizer, train_loader, epochs, device,
                verbose=True):
    model.train()

    train_history = {
        'loss': [],
        'epoch_time': []
    }

    if verbose:
        print(f"开始训练URDMU模型，共{epochs}轮...")
        print(f"设备: {device}")
    model.flag = "Train"
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
            inputs = torch.cat([ normal_data,anomaly_data], dim=0)  # [batch_size, 32, feature_dim]

            # 前向传播
            outputs = model(inputs)  # [batch_size, 64, 1]

            # 计算损失
            # loss = mil_loss(outputs, batch_size, sparsity_weight, smoothness_weight)
            criterion = AD_Loss()
            label = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)], dim=0).to(device)
            loss,_ = criterion(outputs, label)
            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0

        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)

        if verbose:  # 打印结果，只练不测。。
            print(f'Epoch {epoch + 1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')

    if verbose:
        print("训练完成！")

    return train_history


def fit_main(X_train, y_train, model, optimizer, epochs, batch_size, device, verbose=True):
    model.train()

    # 分离正常和异常数据
    normal_mask = y_train == 0
    anomaly_mask = y_train == 1

    X_normal = X_train[normal_mask]
    X_anomaly = X_train[anomaly_mask]

    if len(X_anomaly) == 0 or len(X_normal) == 0:
        raise ValueError("训练数据中必须同时包含正常和异常样本")

    normal_clips_num, anomaly_clips_num = X_normal.shape[0], X_anomaly.shape[0]

    # 通过过采样确保正常样本与异常样本数量相同
    if normal_clips_num < anomaly_clips_num:
        # 重复正常样本直到数量与异常样本相同
        repeat_times = (anomaly_clips_num + normal_clips_num - 1) // normal_clips_num
        X_normal = np.tile(X_normal, (repeat_times, 1))[:anomaly_clips_num]
    elif anomaly_clips_num < normal_clips_num:
        # 重复异常样本直到数量与正常样本相同
        repeat_times = (normal_clips_num + anomaly_clips_num - 1) // anomaly_clips_num
        X_anomaly = np.tile(X_anomaly, (repeat_times, 1))[:normal_clips_num]

    assert len(X_normal) == len(X_anomaly), "采样后正常样本和异常样本数量仍不匹配"
    data_len = len(X_normal)

    # 重塑数据为视频段格式
    segments_per_video = 32
    segments_num = data_len // segments_per_video

    X_normal_videos = X_normal[:segments_num * segments_per_video].reshape(segments_num, segments_per_video, -1)
    X_anomaly_videos = X_anomaly[:segments_num * segments_per_video].reshape(segments_num, segments_per_video, -1)

    # 创建训练数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_normal_videos),
        torch.FloatTensor(X_anomaly_videos)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 调用主训练函数
    return fit(model, optimizer, train_loader, epochs, device, verbose)