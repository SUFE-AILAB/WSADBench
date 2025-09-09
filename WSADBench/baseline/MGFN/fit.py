# -*- coding: utf-8 -*-
"""
MGFN方法训练逻辑
基于MIL (Multiple Instance Learning) 的弱监督异常检测训练
"""
import torch
import torch.nn.functional as F
import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, Optional, List
import time
from WSADBench.baseline.MGFN.model import new_feature
from torch import nn
from WSADBench.baseline.VadClip.clip.myUtils import setup_logging

from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset, Dataset

from common_utils.baseline_utils import get_gt

logger = setup_logging(log_dir='/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/logs', name='mgfn')
def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def smooth(arr, lamda1):
    arr1 = arr[:, :-1, :]
    arr2 = arr[:, 1:, :]

    loss = torch.sum((arr2 - arr1) ** 2)

    return lamda1 * loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class SigmoidCrossEntropyLoss(nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class mgfn_loss(torch.nn.Module):
    """
        MGFN 模型的自定义损失函数。
        它结合了分类损失 (Binary Cross Entropy Loss) 和多种对比损失 (Contrastive Loss)，
        旨在同时优化异常分数预测和特征幅度的判别性与聚类性。
        """
    def __init__(self, alpha):
        super(mgfn_loss, self).__init__()
        self.alpha = alpha
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()
        self.contrastive = ContrastiveLoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()
        label = label.cuda()
        seperate = len(abn_feamagnitude) / 2

        loss_cls = self.criterion(score, label)
        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2), torch.norm(nor_feamagnitude, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal
        loss_con_n = self.contrastive(torch.norm(nor_feamagnitude[int(seperate):], p=1, dim=2),
                                      torch.norm(nor_feamagnitude[:int(seperate)], p=1, dim=2),
                                      0)  # try to cluster the same class
        loss_con_a = self.contrastive(torch.norm(abn_feamagnitude[int(seperate):], p=1, dim=2),
                                      torch.norm(abn_feamagnitude[:int(seperate)], p=1, dim=2), 0)
        loss_total = loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n)

        return loss_total
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

def fit(model, optimizer, epochs, device, X_test, trainer,
                       verbose=True, normal_loader=None, anomaly_loader=None):
    # model, optimizer, train_loader, epochs, device,X_test,trainer,
    #             verbose=True):

    model.train()

    train_history = {
        'loss': [],
        'epoch_time': []
    }

    if verbose:
        print(f"开始训练MGFN模型，共{epochs}轮...")
        print(f"设备: {device}")
    logger.info('start train ...')
    X_test, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames = X_test  # 拆包
    best_epoch = -1
    best_auc = 0.0
    best_ap = 0
    best_epoch_v2 = -1
    best_auc_v2 = 0.0
    best_ap_v2 = 0


    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, (normal_data, anomaly_data) in enumerate(zip(normal_loader, anomaly_loader)):
            optimizer.zero_grad()
            # ========== 添加预处理：处理长度不一致问题 ==========
            normal_batch_size = normal_data.shape[0]
            anomaly_batch_size = anomaly_data.shape[0]
            # print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: normal_batch_size={normal_batch_size}, anomaly_batch_size={anomaly_batch_size}")
            # 当长度不一致时，对少的进行重采样
            if normal_batch_size != anomaly_batch_size:
                if normal_batch_size < anomaly_batch_size:
                    # normal样本不够，进行重采样
                    shortage = anomaly_batch_size - normal_batch_size
                    # 随机选择索引进行重复采样
                    repeat_indices = torch.randint(0, normal_batch_size, (shortage,))
                    repeated_normal = normal_data[repeat_indices]
                    normal_data = torch.cat([normal_data, repeated_normal], dim=0)
                    logger.info(f"Normal data upsampled from {normal_batch_size} to {normal_data.shape[0]}")

                elif anomaly_batch_size < normal_batch_size:
                    # anomaly样本不够，进行重采样
                    shortage = normal_batch_size - anomaly_batch_size
                    # 随机选择索引进行重复采样
                    repeat_indices = torch.randint(0, anomaly_batch_size, (shortage,))
                    repeated_anomaly = anomaly_data[repeat_indices]
                    anomaly_data = torch.cat([anomaly_data, repeated_anomaly], dim=0)
                    logger.info(f"Anomaly data upsampled from {anomaly_batch_size} to {anomaly_data.shape[0]}")

            # 确保长度一致
            assert normal_data.shape[0] == anomaly_data.shape[0], \
                f"数据长度仍不一致: normal={normal_data.shape[0]}, anomaly={anomaly_data.shape[0]}"

            new_inputs = torch.cat([new_feature(normal_data.numpy()), new_feature(anomaly_data.numpy())], dim=0)
            new_inputs = new_inputs.to(device)  # 16+16

            batch_size = normal_data.shape[0]  # 注意可能大小不一致

            # 合并正常和异常数据 [batch_size, 64, feature_dim]
            # 前32个为异常，后32个为正常
            # 前向传播  [60, 1, 32, 2049]
            score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = model(new_inputs)  # [batch_size, 64, 1]

            def get_loss(score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores):
                batch_size = score_normal.shape[0]
                loss_sparse = sparsity(scores[:batch_size, :, :].view(-1), batch_size, 8e-3)

                loss_smooth = smooth(scores, 8e-4)

                scores = scores.view(batch_size * 32 * 2, -1)
                scores = scores.squeeze()

                nlabel = torch.zeros(batch_size).to(device)
                alabel = torch.ones(batch_size).to(device)

                loss_criterion = mgfn_loss(0.0001)  # 不是完全的batchsize

                cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude,
                                      abn_feamagnitude) + loss_smooth + loss_sparse
                return cost
            # 计算损失
            loss = get_loss(score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores)
            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            if trainer.use_scheduler:
                trainer.scheduler.step()

            if verbose and batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        if X_test is not None and trainer.is_test:
            trainer.fitted = True
            # 处理video分数的特殊逻辑：从clip级别还原到帧级别
            with torch.no_grad():
                scores = trainer.predict_proba(X_test)  # 得分696270
                prob = np.repeat(scores, 16)
                gt = get_gt(len(prob))
                test_auc_v2 = roc_auc_score(gt, prob)
                test_ap_v2 = average_precision_score(gt, prob)

                frame_scores, frame_truth = _process_video_scores(scores, video_shape, y_test_idx, y_test_gt, y_test_gt_idx,
                                                                  num_clip_frames)
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
                logger.info(f"cur epoch:{epoch} AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f} best epoch:{best_epoch}, best auc:{best_auc:.4f}, best ap:{best_ap:4f}")
                logger.info(
                    f"cur epoch_v2:{epoch} AUCROC_v2: {test_auc_v2:.4f}, AUCPR_v2: {test_ap_v2:.4f} best epoch_v2:{best_epoch_v2}, best auc_v2:{best_auc_v2:.4f}, best ap_v2:{best_ap_v2:4f}")
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)

        if verbose:  # 打印结果，只练不测。。
            print(f'Epoch {epoch + 1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')

    if verbose:
        print("训练完成！")

    return train_history

