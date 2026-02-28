# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import time
from WSADBench.baseline.VadClip.clip.myUtils import setup_logging
from common_utils.baseline_utils import get_gt, write_jsonl

logger = setup_logging(log_dir='/gpudata/wsad/working_space/zsy/WSADBench/WSADBench/datasets/logs', name='urdmu')
from sklearn.metrics import roc_auc_score, average_precision_score
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
    model.train()

    train_history = {
        'loss': [],
        'epoch_time': []
    }

    if verbose:
        print(f"开始训练URDMU模型，共{epochs}轮...")
        print(f"设备: {device}")

    logger.info('start train ...')
    X_test, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames = X_test  # 拆包
    best_epoch = -1
    best_auc = 0.0
    best_ap = 0
    best_epoch_v2 = -1
    best_auc_v2 = 0.0
    best_ap_v2 = 0
    cur_step = 0
    step_flag = False
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        model.flag = "Train"
        for batch_idx, (normal_data, anomaly_data) in enumerate(zip(normal_loader, anomaly_loader)):
            optimizer.zero_grad()

            # 将数据移到设备
            normal_data = normal_data.to(device)  # [batch, 10, 32, feature]
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
            cur_step+=1
            if cur_step >= trainer.step:
                step_flag = True
                break
            if verbose and batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}, cur_step:{cur_step}')

        epoch_time = time.time() - epoch_start_time
        if X_test is not None and trainer.is_test and cur_step == trainer.step:
            trainer.fitted= True
            with torch.no_grad():
                scores = trainer.predict_proba(X_test)  # 得分696270
                prob = np.repeat(scores, 16)
                gt = get_gt(len(prob))
                test_auc_v2 = roc_auc_score(gt, prob)  # prob成11141440了。。
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
                write_jsonl(model_name='urdmu', epochs=cur_step, auc=test_auc, ap=test_ap, seed=trainer.seed,res_type='frame')
                write_jsonl(model_name='urdmu', epochs=cur_step, auc=test_auc_v2, ap=test_ap_v2, seed=trainer.seed,res_type='clip')
                logger.info(f"cur epoch:{epoch} AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f} best epoch:{best_epoch}, best auc:{best_auc:.4f}, best ap:{best_ap:4f}")
                logger.info(
                    f"cur epoch_v2:{epoch} AUCROC_v2: {test_auc_v2:.4f}, AUCPR_v2: {test_ap_v2:.4f} best epoch_v2:{best_epoch_v2}, best auc_v2:{best_auc_v2:.4f}, best ap_v2:{best_ap_v2:4f}")

        train_history['epoch_time'].append(epoch_time)

        if verbose:  # 打印结果，只练不测。。
            print(f'Epoch {epoch + 1}/{epochs} 完成 - 平均损失: {loss.item():.6f}, 耗时: {epoch_time:.2f}s')
        if step_flag:  # 步数退出
            break

    if verbose:
        print("训练完成！")

    return train_history


