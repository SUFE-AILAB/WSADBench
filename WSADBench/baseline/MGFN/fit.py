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
from myUtils import myLogger as logging
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset, Dataset

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

def fit_mgfn_with_crops(model, optimizer, epochs, device, X_test, trainer,
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
    logging.info('start train ...')
    X_test, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames = X_test  # 拆包
    best_epoch = -1
    best_auc = 0.0
    best_ap = 0
    best_epoch_v2 = -1
    best_auc_v2 = 0.0
    best_ap_v2 = 0
    gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/VadCLIP_v1/list/ucf_gt_wsad.npy'
    gt = np.load(gt_path)
    gt = np.repeat(gt, 10)

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
                    logging.info(f"Normal data upsampled from {normal_batch_size} to {normal_data.shape[0]}")

                elif anomaly_batch_size < normal_batch_size:
                    # anomaly样本不够，进行重采样
                    shortage = normal_batch_size - anomaly_batch_size
                    # 随机选择索引进行重复采样
                    repeat_indices = torch.randint(0, anomaly_batch_size, (shortage,))
                    repeated_anomaly = anomaly_data[repeat_indices]
                    anomaly_data = torch.cat([anomaly_data, repeated_anomaly], dim=0)
                    logging.info(f"Anomaly data upsampled from {anomaly_batch_size} to {anomaly_data.shape[0]}")

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

            if verbose and batch_idx % 10 == 0:
                logging.info(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        if X_test is not None:
            trainer.fitted = True
            # 处理video分数的特殊逻辑：从clip级别还原到帧级别
            with torch.no_grad():
                scores = trainer.predict_proba(X_test)  # 得分696270
                prob = np.repeat(scores, 16)
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
                logging.info(f"cur epoch:{epoch} AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f} best epoch:{best_epoch}, best auc:{best_auc:.4f}, best ap:{best_ap:4f}")
                logging.info(
                    f"cur epoch_v2:{epoch} AUCROC_v2: {test_auc_v2:.4f}, AUCPR_v2: {test_ap_v2:.4f} best epoch_v2:{best_epoch_v2}, best auc_v2:{best_auc_v2:.4f}, best ap_v2:{best_ap_v2:4f}")
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)

        if verbose:  # 打印结果，只练不测。。
            print(f'Epoch {epoch + 1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')

    if verbose:
        print("训练完成！")

    return train_history


# def fit_mgfn_main(X_train, y_train, model, optimizer, epochs, batch_size, device, X_test,trainer, verbose=True):
#
#     model.train()
#     # 分离正常和异常数据(data)
#     normal_mask = y_train == 0
#     anomaly_mask = y_train == 1
#
#     X_normal = X_train[normal_mask]
#     X_anomaly = X_train[anomaly_mask]
#
#     if len(X_anomaly) == 0 or len(X_normal) == 0:
#         raise ValueError("训练数据中必须同时包含正常和异常样本")
#
#     normal_clips_num, anomaly_clips_num = X_normal.shape[0], X_anomaly.shape[0]
#
#     # 通过过采样确保正常样本与异常样本数量相同
#     if normal_clips_num < anomaly_clips_num:
#         # 重复正常样本直到数量与异常样本相同
#         repeat_times = (anomaly_clips_num + normal_clips_num - 1) // normal_clips_num
#         X_normal = np.tile(X_normal, (repeat_times, 1))[:anomaly_clips_num]
#     elif anomaly_clips_num < normal_clips_num:
#         # 重复异常样本直到数量与正常样本相同
#         repeat_times = (normal_clips_num + anomaly_clips_num - 1) // anomaly_clips_num
#         X_anomaly = np.tile(X_anomaly, (repeat_times, 1))[:normal_clips_num]
#
#     assert len(X_normal) == len(X_anomaly), "采样后正常样本和异常样本数量仍不匹配"
#     data_len = len(X_normal)
#
#     # 重塑数据为视频段格式
#     segments_per_video = 32
#     segments_num = data_len // segments_per_video
#
#     X_normal_videos = X_normal[:segments_num * segments_per_video].reshape(segments_num, segments_per_video, -1)
#     X_anomaly_videos = X_anomaly[:segments_num * segments_per_video].reshape(segments_num, segments_per_video, -1)
#
#     # 创建训练数据集
#     train_dataset = TensorDataset(
#         torch.FloatTensor(X_normal_videos),
#         torch.FloatTensor(X_anomaly_videos)
#     )
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
#     # 调用主训练函数
#     return fit_mgfn(model, optimizer, train_loader, epochs, device,
#                         X_test, trainer,verbose)
def process_feat(feat, length):
    """处理特征，将其调整为指定长度"""
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int64)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


class VideoDataset(Dataset):
    """视频数据集类"""
    def __init__(self, features,  seg, is_test=False):
        self.is_test = is_test
        self.seg = seg
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.is_test:
            return self.features[idx]
        else:
            # feature = process_feat(self.features[idx], 32)  # divide a video into 32 segments
            # return feature
            divided_features = []
            for feature in self.features[idx]:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            return divided_features

def fit_mgfn_main_with_crops(X_train, y_train, model, optimizer, epochs, batch_size, device, X_test, trainer,
                            verbose=True, clip_num=None, crops_num=None):
    """
    MGFN主训练函数，支持crops数据格式
    """
    model.train()
    ncrop = crops_num
    seg = trainer.seg  # 32表示32seg
    feature = trainer.input_dim  # 使用channels作为特征维度

    def split_by_seg_list(X, seg_list, feature):
        X = X.reshape(-1, ncrop, feature)  # [seg, crop:10, 2048]
        segments = []
        start = 0
        for seg_len in seg_list:
            end = start + seg_len
            for i in range(ncrop):
                segment = X[start:end, i]  # shape: [seg_len, 2048]
                segments.append(segment)
            start = end
        return segments
    clip_num = clip_num.values()
    # 按X的形状，判断X是否是32seg
    if len(clip_num)*32*ncrop == X_train.shape[0]:
        print('32 seg')
        clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版
        # clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版

    # clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版
    X_train = split_by_seg_list(X_train, clip_num, feature)  # (16100, seg, 2048)
    y_train = split_by_seg_list(y_train, clip_num, 1)  # 16100个(seg, 1)(mask)
    y_train = [int(item[0, 0]) for item in y_train]  # 16100个0、1标签list

    def group_into_crops(X_videos, crop_size=10):
        grouped = []
        for i in range(0, len(X_videos), crop_size):
            batch = X_videos[i:i + crop_size]
            if len(batch) == crop_size:
                crop = np.stack(batch, axis=0)  # shape: [10, 32, 2048]
                grouped.append(crop)
        return grouped

    X_normal_videos = group_into_crops([X_train[i] for i in range(len(X_train)) if y_train[i] == 0])
    X_anomaly_videos = group_into_crops([X_train[i] for i in range(len(X_train)) if y_train[i] == 1])

    # clip_num_expand = [min(seg, clip_num[i]) for i in range(len(clip_num)) for _ in range(ncrop)]
    # normal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
    # abnormal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 1]

    # 构建 Dataset 和 DataLoader
    normal_dataset = VideoDataset(X_normal_videos, seg)
    anomaly_dataset = VideoDataset(X_anomaly_videos, seg)

    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # clip_num = clip_num.values()
    # # 按X的形状，判断X是否是32seg
    # if len(clip_num) * 32 * ncrop == X_train.shape[0]:
    #     print('32 seg')
    #     clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版
    #
    # def split_by_seg_list(X, seg_list, feature):
    #     X = X.reshape(-1, ncrop, feature)  # 【seg, crop:10, 2048]
    #     segments = []
    #     start = 0
    #     for seg_len in seg_list:
    #         end = start + seg_len
    #         for i in range(ncrop):
    #             segment = X[start:end, i]  # shape: [1, seg_len, 2048]
    #             segments.append(segment)
    #         start = end
    #     return segments
    #
    # X_train = split_by_seg_list(X_train, clip_num, feature)  # (16100, seg, 2048)
    # y_train = split_by_seg_list(y_train, clip_num, 1)  # 16100个(, seg, 1)(mask)
    # y_train = [int(item[0, 0]) for item in y_train]  # 16100个0、1标签list
    # # y_train = y_train.reshape(-1)  # (16100,)
    #
    # # 消除 y_train 的重复，取每组的第一个标签
    # # reshape y_train to match the reshaped X_train
    # # y_train = y_train.reshape(-1, ncrop)  # [1610*seg, 10]
    # # y_train = y_train.reshape(-1, seg, ncrop)  # [num, 32, 10]
    # # # y_train = y_train.transpose(0, 2, 1) # [num, 10, 32]
    # # y_train = y_train[:, 0, :].reshape(-1)   # shape: (num*10)
    # # 按y_train为mask，把X_train拆分为normal和 anomaly两个X；并拆分其对应的vid_kind, clip_num
    # # 3. 使用 y_train 作为 mask，划分正常/异常（0=normal，1=anomaly）
    # # normal_mask = y_train == 0
    # # abnormal_mask = y_train == 1
    #
    # X_normal_videos = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
    # X_anomaly_videos = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1]
    # # vid_kind_expand = [vid_kind[i] for i in range(len(vid_kind)) for _ in range(ncrop)]  # 包含str标签，并复制
    # clip_num_expand = [min(seg, clip_num[i]) for i in range(len(clip_num)) for _ in range(ncrop)]  # 注意上界
    # # 4. 对应扩展 vid_kind 和 clip_num：每个视频被切成了10段
    # # 4. 用 mask 提取对应部分
    # # normal_vid_kind = [vid_kind_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
    # # abnormal_vid_kind = [vid_kind_expand[i] for i in range(len(y_train)) if y_train[i] == 1]
    #
    # normal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
    # abnormal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 1]
    #
    # # 6. 构建 Dataset 和 DataLoader
    # normal_dataset = VideoDataset(X_normal_videos,  seg)
    # anomaly_dataset = VideoDataset(X_anomaly_videos,  seg)
    #
    # normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # 调用主训练函数
    return fit_mgfn_with_crops(model, optimizer, epochs, device, X_test, trainer,
                              verbose, normal_loader, anomaly_loader)