# -*- coding: utf-8 -*-
"""
MGFN方法训练逻辑
基于MIL (Multiple Instance Learning) 的弱监督异常检测训练
"""
from torch.utils.data import DataLoader, TensorDataset
import time
from WSADBench.baseline.RTFM.model import new_feature
import numpy as np
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import MSELoss



def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin, device):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()
        self.device =device

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.to(self.device)

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total


def fit(model, optimizer, train_loader, epochs, device,
                verbose=True):

    model.train()

    train_history = {
        'loss': [],
        'epoch_time': []
    }

    if verbose:
        print(f"开始训练RTFM模型，共{epochs}轮...")
        print(f"设备: {device}")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, (normal_data, anomaly_data) in enumerate(train_loader):
            optimizer.zero_grad() # 这里是先正常 再 不正常
            new_inputs = torch.cat([new_feature(normal_data.numpy()), new_feature(anomaly_data.numpy())], dim=0)
            new_inputs = new_inputs.to(device)
            # print(f'正常大小：{normal_data.shape}, 异常大小：{anomaly_data.shape}')  # 24报错
            batch_size = normal_data.shape[0]

            # 合并正常和异常数据 [batch_size, 64, feature_dim]
            # 前32个为异常，后32个为正常
            # 前向传播  [60, 1, 32, 2049]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(new_inputs)  # [batch_size, 64, 1]

            # 计算损失
            # loss = get_loss(score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores)
            scores = scores.view(batch_size * 32 * 2, -1)

            scores = scores.squeeze()
            abn_scores = scores[batch_size * 32:]

            nlabel = torch.zeros(batch_size)
            alabel = torch.ones(batch_size)

            loss_criterion = RTFM_loss(0.0001, 100, device)
            loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
            loss_smooth = smooth(abn_scores, 8e-4)
            loss = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal,
                                  feat_select_abn) + loss_smooth + loss_sparse
            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        # print("train finish")
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)

        if verbose:  # 打印结果，只练不测。。
            print(f'Epoch {epoch + 1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')

    if verbose:
        print("训练完成！")

    return train_history


def fit_main(X_train, y_train, model, optimizer, epochs, batch_size, device,
                      verbose=True):
    model.train()

    # 分离正常和异常数据(data)
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
    return fit(model, optimizer, train_loader, epochs, device,
                        verbose)