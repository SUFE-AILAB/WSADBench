# -*- coding: utf-8 -*-
"""
MGFN方法训练逻辑
基于MIL (Multiple Instance Learning) 的弱监督异常检测训练
"""
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset, Dataset
import time

from tqdm import tqdm

from WSADBench.baseline.RTFM.model import new_feature
import numpy as np
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import MSELoss
from WSADBench.baseline.VadClip.clip.myUtils import myLogger as logging




def sparsity(arr,  lamda2):
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
    logging.info('start train ...')
    model.train()

    train_history = {
        'loss': [],
        'epoch_time': []
    }

    if verbose:
        print(f"开始训练RTFM模型，共{epochs}轮...")
        print(f"设备: {device}")
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
            optimizer.zero_grad() # 这里是先正常 再 不正常
            new_inputs = torch.cat([new_feature(normal_data.numpy()), new_feature(anomaly_data.numpy())], dim=0)

            new_inputs = new_inputs.to(device)
            new_inputs = new_inputs.squeeze(1)
            # print(f'形状：{new_inputs.shape}')

            # print(f'正常大小：{normal_data.shape}, 异常大小：{anomaly_data.shape}')  # 24报错
            # batch_size = normal_data.shape[0]

            # 合并正常和异常数据 [batch_size, 64, feature_dim]
            # 前32个为异常，后32个为正常
            # 前向传播  [60, 1, 32, 2049]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(new_inputs)  # [batch_size, 64, 1]

            # 计算损失
            # loss = get_loss(score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores)
            scores = scores.view((len(new_inputs)) * 32, -1)  #

            scores = scores.squeeze()
            abn_scores = scores[len(anomaly_data)* 32:]

            nlabel = torch.zeros(len(normal_data))
            alabel = torch.ones(len(anomaly_data))

            loss_criterion = RTFM_loss(0.0001, 100, device)
            loss_sparse = sparsity(abn_scores,  8e-3)
            loss_smooth = smooth(abn_scores, 8e-4)
            loss = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal,
                                  feat_select_abn) + loss_smooth + loss_sparse
            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if verbose and batch_idx % 10 == 0:
                logging.info(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        # 测试一次

        trainer.fitted = True
        trainer.model.eval()
        if X_test is not None:
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

        # print("train finish")
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)

        if verbose:  # 打印结果，只练不测。。
            print(f'Epoch {epoch + 1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')

    if verbose:
        print("训练完成！")

    return train_history


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int64)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat
# 5. 自定义 Dataset
class VideoDataset(Dataset):
    def __init__(self, features, clip_num, seg, is_test=False):
        self.is_test = is_test
        self.seg = seg
        self.features = features
        # self.vid_kind = vid_kind  # 保留为 str list
        # self.clip_num = clip_num # 保留为 int list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.is_test:
            # feat, _ = process_split(self.features[idx], self.seg)
            return self.features[idx]  # 直接返回
        else:
            divided_features = []
            for feature in self.features[idx]:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features

def fit_main(X_train, y_train, model, optimizer, epochs, batch_size, device,X_test,trainer,
                      verbose=True, clip_num=None,  crops_num=None, ):
    model.train()
    ncrop = crops_num
    seg = trainer.seg # 32表示32seg
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
    clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版
    X_train = split_by_seg_list(X_train, clip_num, feature)  # (16100, seg, 2048)
    y_train = split_by_seg_list(y_train, clip_num, 1)  # 16100个(, seg, 1)(mask)
    # for item in y_train:
    #     print(item.shape)
    y_train = [int(item[0, 0]) for item in y_train]  # 16100个0、1标签list

    def group_into_crops(X_normal_videos, crop_size=10):
        grouped = []
        for i in range(0, len(X_normal_videos), crop_size):
            batch = X_normal_videos[i:i + crop_size]
            if len(batch) == crop_size:
                crop = np.stack(batch, axis=0)  # shape: [10, 32, 2048]
                grouped.append(crop)
        return grouped

    X_normal_videos = group_into_crops([X_train[i] for i in range(len(X_train)) if y_train[i] == 0])
    X_anomaly_videos = group_into_crops([X_train[i] for i in range(len(X_train)) if y_train[i] == 1])
    # y_train = np.concatenate(np.zeros(len(X_normal_videos)), np.ones(len(X_anomaly_videos)), axis=0)
    # vid_kind_expand = [vid_kind[i] for i in range(len(vid_kind)) for _ in range(ncrop)]  # 包含str标签，并复制
    clip_num_expand = [min(seg, clip_num[i]) for i in range(len(clip_num)) for _ in range(ncrop)]  # 注意上界
    # 4. 对应扩展 vid_kind 和 clip_num：每个视频被切成了10段

    normal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
    abnormal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 1]

    # 6. 构建 Dataset 和 DataLoader
    normal_dataset = VideoDataset(X_normal_videos,  normal_clip_num, seg)
    anomaly_dataset = VideoDataset(X_anomaly_videos,  abnormal_clip_num, seg)

    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # 调用主训练函数
    return fit(model, optimizer, epochs, device, X_test,  trainer,
                verbose, normal_loader,anomaly_loader)