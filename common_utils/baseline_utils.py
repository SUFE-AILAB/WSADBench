import numpy as  np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import time

def get_gt(score_len):

    if score_len == 1436800:
        # gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/VadCLIP_v1/list/ucf_gt_wsad.npy'
        gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/VadCLIP_v1/list/shanghaitech_gt_1436800.npy'  # 11141440
        # gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/tmpModel/datasets/gt-ucf.npy'  # 11141440
        gt = np.load(gt_path)
    elif score_len == 11141440:
        gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/tmpModel/datasets/gt-ucf.npy'  # 11141440
        gt = np.load(gt_path)
        gt = gt.repeat(10)

    elif score_len == 11140320:
        gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/VadCLIP_v1/list/ucf_gt_wsad.npy'
        gt = np.load(gt_path)
        gt = gt.repeat(10)
    elif score_len == 23431840:
        gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/VadCLIP_v1/list/xdviolence_gt.npy'
        gt = np.load(gt_path)
    elif score_len == 887520:
        gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/VadCLIP_v1/list/tad_gt.npy'
        gt = np.load(gt_path)
    elif score_len == 21440:
        gt_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/baseline/VadCLIP_v1/list/ucsd_ped2_gt.npy'
        gt = np.load(gt_path)
    else:
        raise Exception('score len error')
    return gt


def process_feat(feat, length):
    if feat.shape[1] == length:  # 跳过
        return feat
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int64)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat

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
### tool.utils
def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat
def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate(
                    [split_feat, feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat,
                                             pad(feat[i * length:i * length + length, :], length).reshape(1, length,
                                                                                                          feat.shape[
                                                                                                              1])],
                                            axis=0)

        return split_feat, clip_length
# VadClip的预处理
def uniform_extract(feat, t_max, avg: bool = True):  # t_max  = 256
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)  # [seg:256, feature:2048]
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)  # 原始feat是[seg:564, feature:2048]
    if avg == True:
        for i in range(t_max):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
    else:
        r = np.linspace(0, feat.shape[0] - 1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]

    return new_feat


def process_feat_VadClip(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        # if is_random:
        #     return random_extract(feat, length), length
        # else:
        return uniform_extract(feat, length), length  # train 时用这个
    else:
        return pad(feat, length), clip_length

class VideoDataset_VadClip(Dataset):
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
            feat, _ = process_feat_VadClip(self.features[idx], self.seg)
            return feat, self.vid_kind[idx], self.clip_num[idx]

def fit_utils(X_train, y_train, model, optimizer, epochs, batch_size, device,X_test,trainer,
                      verbose=True, clip_num=None,  crops_num=None,   # 通用参数
               ):  #  其他参数

    """
        MGFN主训练函数，支持crops数据格式
        """
    model.train()
    ncrop = crops_num
    clip_num = clip_num.values()
    if len(clip_num) * 32 * ncrop == X_train.shape[0]:
        print('32 seg')
        clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版
        seg = 32
    elif len(clip_num) * 200 * ncrop == X_train.shape[0]:
        print('200 seg')
        clip_num = [200 for i in range(len(clip_num))]  # 32 seg 版
        seg = 200
    else:
        print('变长seg')
        seg = 32
    # seg = trainer.seg  # 32表示32seg
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
    def balance_video_samples(normal_videos, anomaly_videos):
        """
        平衡正常和异常视频的数量，使用均匀采样
        只增加样本数量，不减少原有样本

        Args:
            normal_videos: 正常视频列表
            anomaly_videos: 异常视频列表

        Returns:
            balanced_normal_videos, balanced_anomaly_videos: 平衡后的视频列表
        """
        normal_count = len(normal_videos)
        anomaly_count = len(anomaly_videos)

        logging.info(f"原始数据: 正常视频 {normal_count} 个, 异常视频 {anomaly_count} 个")

        if normal_count == 0 or anomaly_count == 0:
            logging.warning("正常或异常视频数量为0，无法平衡采样")
            return normal_videos, anomaly_videos

        # 确定目标数量（取较大值，只增加不减少）
        target_count = max(normal_count, anomaly_count)

        def uniform_upsample(videos, target_size):
            """均匀上采样函数 - 只增加样本，不减少"""
            if len(videos) >= target_size:
                # 如果数量已经足够，直接返回原始数据
                return videos
            else:
                # 如果数量不足，均匀重复采样来增加样本
                original_videos = videos.copy()  # 保留所有原始样本
                additional_needed = target_size - len(videos)

                # 均匀选择要重复的样本
                if additional_needed > 0:
                    repeat_indices = np.linspace(0, len(videos) - 1, additional_needed, dtype=int)
                    additional_videos = [videos[i] for i in repeat_indices]
                    return original_videos + additional_videos
                else:
                    return original_videos

        # 均匀上采样
        balanced_normal = uniform_upsample(normal_videos, target_count)
        balanced_anomaly = uniform_upsample(anomaly_videos, target_count)

        logging.info(f"平衡后数据: 正常视频 {len(balanced_normal)} 个, 异常视频 {len(balanced_anomaly)} 个")

        # 验证没有样本被删除
        assert len(balanced_normal) >= normal_count, "正常视频样本数量不应该减少"
        assert len(balanced_anomaly) >= anomaly_count, "异常视频样本数量不应该减少"
        assert len(balanced_normal) == len(balanced_anomaly), "平衡后两类样本数量应该相等"

        return balanced_normal, balanced_anomaly

    # 执行平衡采样
    X_normal_videos_balanced, X_anomaly_videos_balanced = balance_video_samples(
        X_normal_videos, X_anomaly_videos
    )

    # 构建 Dataset 和 DataLoader
    normal_dataset = VideoDataset(X_normal_videos_balanced, seg)
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    anomaly_dataset = VideoDataset(X_anomaly_videos_balanced, seg)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # 调用主训练函数
    return  {
    "model": model,
    "optimizer": optimizer,
    "epochs": epochs,
    "device": device,
    "X_test": X_test,
    "trainer": trainer,
    "verbose": verbose,
    "normal_loader": normal_loader,
    "anomaly_loader": anomaly_loader,
}



def fit_VadClip(X_train, y_train, model, optimizer, epochs, batch_size, device,X_test,trainer,
                      verbose=True, clip_num=None,  crops_num=None,   # 通用参数
              vid_info=None,  vid_kind=None,  X_test_extra=None):
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

    clip_num = list(clip_num.values())

    # 按X的形状，判断X是否是32seg
    if len(clip_num) * 32 * ncrop == X_train.shape[0]:
        print('32 seg')
        clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版
    elif len(clip_num) * 200 * ncrop == X_train.shape[0]:
        print('200 seg')
        clip_num = [200 for i in range(len(clip_num))]  # 200 seg 版
    X_train = split_by_seg_list(X_train, clip_num, feature)  # (16100, seg, 2048)
    y_train = split_by_seg_list(y_train, clip_num, 1)  # 16100个(, seg, 1)(mask)
    y_train = [int(item[0, 0]) for item in y_train]  # 16100个0、1标签list

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



    # todo:这三者长度一致：X_normal_videos, normal_vid_kind, normal_clip_num；平衡正常和异常视频的数量，使用均匀采样
    #         只增加样本数量，不减少原有样本
    def balance_video_samples(normal_videos, normal_vid_kind, normal_clip_num, anomaly_videos, abnormal_vid_kind, abnormal_clip_num):
        normal_count = len(normal_videos)
        anomaly_count = len(anomaly_videos)

        logging.info(f"原始数据: 正常视频 {normal_count} 个, 异常视频 {anomaly_count} 个")

        if normal_count == 0 or anomaly_count == 0:
            logging.warning("正常或异常视频数量为0，无法平衡采样")
            return normal_videos, anomaly_videos

        # 确定目标数量（取较大值，只增加不减少）
        target_count = max(normal_count, anomaly_count)

        def uniform_upsample(videos, target_size):
            """均匀上采样函数 - 只增加样本，不减少"""
            if len(videos) >= target_size:
                # 如果数量已经足够，直接返回原始数据
                return videos
            else:
                # 如果数量不足，均匀重复采样来增加样本
                original_videos = videos.copy()  # 保留所有原始样本
                additional_needed = target_size - len(videos)

                # 均匀选择要重复的样本
                if additional_needed > 0:
                    repeat_indices = np.linspace(0, len(videos) - 1, additional_needed, dtype=int)
                    additional_videos = [videos[i] for i in repeat_indices]
                    return original_videos + additional_videos
                else:
                    return original_videos

        # 均匀上采样
        balanced_normal = uniform_upsample(normal_videos, target_count)
        balanced_anomaly = uniform_upsample(anomaly_videos, target_count)
        balanced_normal_vid_kind = uniform_upsample(normal_vid_kind , target_count)
        balanced_anomaly_vid_kind = uniform_upsample(abnormal_vid_kind , target_count)
        balanced_normal_clip_num = uniform_upsample(normal_clip_num , target_count)
        balanced_anomaly_clip_num = uniform_upsample(abnormal_clip_num , target_count)
        logging.info(f"平衡后数据: 正常视频 {len(balanced_normal)} 个, 异常视频 {len(balanced_anomaly)} 个")
        return balanced_normal, balanced_normal_vid_kind, balanced_normal_clip_num, balanced_anomaly, balanced_anomaly_vid_kind, balanced_anomaly_clip_num

    # 执行平衡采样
    X_normal_videos, normal_vid_kind, normal_clip_num, X_anomaly_videos, abnormal_vid_kind, abnormal_clip_num = balance_video_samples(
        X_normal_videos, normal_vid_kind, normal_clip_num, X_anomaly_videos, abnormal_vid_kind, abnormal_clip_num
    )

    # 6. 构建 Dataset 和 DataLoader
    normal_dataset = VideoDataset_VadClip(X_normal_videos, normal_vid_kind, normal_clip_num, seg)
    anomaly_dataset = VideoDataset_VadClip(X_anomaly_videos, abnormal_vid_kind, abnormal_clip_num, seg)

    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    return  {
    "model": model,
    "optimizer": optimizer,
    "epochs": epochs,
    "device": device,
    "X_test": X_test,
    "trainer": trainer,
    "verbose": verbose,
    "normal_loader": normal_loader,
    "anomaly_loader": anomaly_loader,
    "X_test_extra": X_test_extra,
}