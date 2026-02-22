import os
from pathlib import Path

import numpy as np
import logging
import gc
import numpy as  np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import time
# 内存占用监控
# import psutil  # 新增
# import threading  # 新增

def write_jsonl(model_name,epochs,seed, auc, ap,  res_type):
    base_dir = Path.cwd()
    file_path = str(base_dir / r'results/video/case_res.txt')
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(f'{model_name},{epochs},{seed},{auc:.4f},{ap:.4f},{res_type}\n')
            f.close()
    else:
        with open(file_path, 'w') as f:
            f.write(f'{model_name},{epochs},{seed},{auc:.4f},{ap:.4f},{res_type}\n')
            f.close()
    print(f'write: {model_name},{epochs},{seed},{auc:.4f},{ap:.4f},{epochs},{res_type}')

def get_gt(score_len):
    """
    base_dir = Path.cwd()
    input_dir = str(base_dir / self.config["PREPROCESS"]["INPUT_DIR"])
    """
    base_dir = Path.cwd()
    if score_len == 1436800:
        gt_path = str(base_dir / r'WSADBench/baseline/VadCLIP_v1/list/shanghaitech_gt_1436800.npy')  # 11141440
        gt = np.load(gt_path)
    elif score_len == 11141440:
        gt_path = str(base_dir / 'WSADBench/baseline/tmpModel/datasets/gt-ucf.npy' ) # 11141440
        gt = np.load(gt_path)
        gt = gt.repeat(10)

    elif score_len == 11140320:
        gt_path = str(base_dir / 'WSADBench/baseline/VadCLIP_v1/list/ucf_gt_wsad.npy')
        gt = np.load(gt_path)
        gt = gt.repeat(10)
    elif score_len == 23431840:
        gt_path = str(base_dir / 'WSADBench/baseline/VadCLIP_v1/list/xdviolence_gt.npy')
        gt = np.load(gt_path)
    elif score_len == 887520:
        gt_path = str(base_dir / 'WSADBench/baseline/VadCLIP_v1/list/tad_gt.npy')
        gt = np.load(gt_path)
    elif score_len == 21440:
        gt_path = str(base_dir / 'WSADBench/baseline/VadCLIP_v1/list/ucsd_ped2_gt.npy')
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





def video_data2tabular_data(data, data_shape, model_name, seed):
    """
    优化版: 视频数据转Tabular数据
    1. 使用 reshape 和 transpose 代替显式循环切割，大幅降低内存占用。
    2. 使用索引数组（Indices）进行样本平衡，避免中间产生大量数据副本。
    """
    # ---------------------------------------------------------
    # 1. 参数解析与形状推断
    # ---------------------------------------------------------
    X_raw = data['X_train']
    y_raw = data['y_train']

    ncrop = data_shape[1] if data_shape else 10
    feature_dim = X_raw.shape[-1]

    # 获取 clip_num (每个视频的片段长度列表)
    clip_num_values = list(data.get("vid_source_clips_num_train", {}).values())
    total_raw_rows = X_raw.shape[0]

    # 推断 seg 长度并处理变长/定长逻辑
    # 注意：为了追求极致速度和内存优化，这里主要针对定长seg进行矩阵化优化
    # 如果是变长，通常无法直接 reshape 成规则张量，必须特殊处理
    is_fixed_seg = False
    seg = 32

    if len(clip_num_values) * 32 * ncrop == total_raw_rows:
        print('Detect: 32 seg (Fixed)')
        seg = 32
        is_fixed_seg = True
    elif len(clip_num_values) * 200 * ncrop == total_raw_rows:
        print('Detect: 200 seg (Fixed)')
        seg = 200
        is_fixed_seg = True
    else:
        print('Detect: 变长 seg (Fallback to 32 logic)')
        seg = 32
        # 如果原始逻辑强行设为32，这里保持一致，但逻辑上可能需要 Padding
        # 为保证代码健壮性，这里假设如果进入 else 分支，数据可能已经被预处理为定长
        # 或者我们通过计算假设它是定长的
        if total_raw_rows % (ncrop * seg) == 0:
            is_fixed_seg = True

    if not is_fixed_seg:
        raise ValueError(
            "内存优化版代码目前要求输入数据必须是规整的 (Total_Rows % (ncrop * seg) == 0)。请检查 clip_num 和数据形状。")

    num_videos = total_raw_rows // (seg * ncrop)

    # ---------------------------------------------------------
    # 2. 数据结构重组 (核心优化: 0内存拷贝)
    # ---------------------------------------------------------
    # 原始数据结构推测: [Total_Time_Steps, ncrop_interleaved, feature]
    # 或者 [Total_Rows, feature] 其中 Total_Rows = Time * ncrop
    # 原始代码 split_by_seg_list 暗示了数据在维度 0 上是 (Time, Crop) 混合的
    # 我们需要将其转换为: [Num_Videos, ncrop, seg, feature]

    # Step A: Reshape 原始数据为 (Num_Videos, Seg, ncrop, Feature)
    # 解释: 原始的一维行向量，每 seg 行代表一个时间段，每行里包含了 ncrop 个数据?
    # 不，原始代码 reshape(-1, ncrop, feature) 说明 X_raw 是 (Total_Time, ncrop, feature)
    # 或者是 (Total_Rows, feature) -> reshape -> (Time, Crop, Feature)

    X_reshaped = X_raw.reshape(num_videos, seg, ncrop, feature_dim)

    # Step B: 调整维度顺序
    # 原始逻辑 group_into_crops 是把 Crop 维度放在 Seg 维度前面
    # 目标单个视频形状: (ncrop, seg, feature)
    X_videos = X_reshaped.transpose(0, 2, 1, 3)  # shape: (Num_Videos, ncrop, seg, feature)

    # Step C: 展平单个视频内部，准备好被最终 flatten
    # 我们希望每个视频变成一行数据 (逻辑上的)，方便后续索引平衡
    # 这里的 "一行" 其实是一个 (ncrop * seg, feature) 的块
    X_videos_flat = X_videos.reshape(num_videos, -1, feature_dim)  # (Num_Videos, Block_Size, Feature)

    # 处理标签 Y
    # 假设每个视频的标签是一致的，取每个视频块的第一个样本的标签
    # y_raw 也是类似结构，取每个视频段的第一个值
    y_reshaped = y_raw.reshape(num_videos, -1)  # (Num_Videos, seg*ncrop...)
    video_labels = y_reshaped[:, 0].astype(int)  # (Num_Videos,)

    # ---------------------------------------------------------
    # 3. 样本平衡 (操作索引，而非数据)
    # ---------------------------------------------------------
    indices_normal = np.where(video_labels == 0)[0]
    indices_anomaly = np.where(video_labels == 1)[0]

    logging.info(f"原始数据: 正常视频 {len(indices_normal)} 个, 异常视频 {len(indices_anomaly)} 个")

    if len(indices_normal) == 0 or len(indices_anomaly) == 0:
        logging.warning("某类样本为0，跳过平衡")
        data['X_train'] = X_videos_flat.reshape(-1, feature_dim)
        data['y_train'] = np.repeat(video_labels, X_videos_flat.shape[1])
        return data

    target_count = max(len(indices_normal), len(indices_anomaly))

    def get_balanced_indices(indices, target_size):
        if len(indices) >= target_size:
            return indices

        # 计算需要补充的数量
        n_needed = target_size - len(indices)
        # 均匀采样索引 (linspace)
        repeat_idx = np.linspace(0, len(indices) - 1, n_needed, dtype=int)
        additional_indices = indices[repeat_idx]
        return np.concatenate([indices, additional_indices])

    balanced_idx_normal = get_balanced_indices(indices_normal, target_count)
    balanced_idx_anomaly = get_balanced_indices(indices_anomaly, target_count)

    logging.info(f"平衡后数据: 正常视频 {len(balanced_idx_normal)} 个, 异常视频 {len(balanced_idx_anomaly)} 个")

    # ---------------------------------------------------------
    # 4. 构建最终数据集
    # ---------------------------------------------------------
    # 合并索引: 先放 Normal，后放 Anomaly (保持原始代码逻辑顺序)
    final_video_indices = np.concatenate([balanced_idx_normal, balanced_idx_anomaly])

    # 使用 Fancy Indexing 提取数据 (这里会发生内存复制，是必要的，但只有这一次)
    # X_videos_flat: (Num_Videos, Block_Size, Feature)
    # Result: (Total_Balanced_Videos, Block_Size, Feature)
    X_final = X_videos_flat[final_video_indices]

    # 最终展平为 Tabular 格式 (N, F)
    data['X_train'] = X_final.reshape(-1, feature_dim)

    # 构建 Y (直接生成，比索引提取更快)
    # Normal 部分全是 0，Anomaly 部分全是 1
    # 每个视频包含 block_size 个样本
    block_size = X_videos_flat.shape[1]  # ncrop * seg
    y_zeros = np.zeros(len(balanced_idx_normal) * block_size, dtype=int)
    y_ones = np.ones(len(balanced_idx_anomaly) * block_size, dtype=int)
    data['y_train'] = np.concatenate([y_zeros, y_ones])

    # ---------------------------------------------------------
    # 5. 特定模型的降采样 (保持原有逻辑)
    # ---------------------------------------------------------
    if model_name in ['XGBOD', 'FTTransformer']:
        max_limit = 10000
        current_rows = data['X_train'].shape[0]
        if current_rows > max_limit:
            rnd = np.random.RandomState(seed)
            idx = rnd.permutation(current_rows)[:max_limit]
            data['X_train'] = data['X_train'][idx]
            data['y_train'] = data['y_train'][idx]
            print(f'采样{max_limit}个样本 (from {current_rows})')

    # 主动释放内存
    # del X_raw, y_raw, X_reshaped, X_videos, X_videos_flat
    # gc.collect()

    return data


def fit_utils(X_train, y_train, model, optimizer, epochs, batch_size, device,X_test,trainer,
                      verbose=True, clip_num=None,  crops_num=None,   # 通用参数
               ):  #  其他参数
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

    X_normal_videos = group_into_crops([X_train[i] for i in range(len(X_train)) if y_train[i] == 0])  # [num,crop:10, seg:32, f:2048]
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


def fit_utils_mil(X_train, y_train, model, optimizer, epochs, batch_size, device, X_test, trainer, exp_note,
              verbose=True, seg=None, clip_num=None, crops_num=None  # 通用参数,加入seg参数
              ):  # 其他参数

    """
        MGFN主训练函数，支持crops数据格式
        """
    model.train()
    def group_into_crops(X_videos, crop_size=1):  # 原版为crop_size=10
        grouped = []
        for i in range(0, len(X_videos), crop_size):
            batch = X_videos[i:i + crop_size]
            if len(batch) == crop_size:
                crop = np.stack(batch, axis=0)  # shape: [10, 32, 2048]
                grouped.append(crop)
        return grouped



    # 为tabular_inexact数据集修改
    _, data_shape, y_test_idx, y_test_gt, y_test_gt_idx, n_samples = X_test  # 拆包

    feature = trainer.input_dim  # 使用channels作为特征维度

    seg = n_samples  # 每个bag的样本数作为seg长度，重赋值
    X_train = X_train.reshape(-1, n_samples, feature)  # (n_bags*n_samples, feature) -> (n_bags, n_samples, feature)
    y_train = y_train[::n_samples]  # 每个bag的标签           #end

    # 为tabular_inexact数据修改
    X_normal_videos = [np.expand_dims(X_train[i], 0) for i in range(len(X_train)) if
                       y_train[i] == 0]  # 扩展一个维度，替代group_into_crops产生的维度
    X_anomaly_videos = [np.expand_dims(X_train[i], 0) for i in range(len(X_train)) if y_train[i] == 1]  # end

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
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    anomaly_dataset = VideoDataset(X_anomaly_videos_balanced, seg)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 调用主训练函数
    return {
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

# def memory_monitor(threshold=90, check_interval=1.0):
#     """
#     后台监控线程函数：
#     每隔 check_interval 秒检查一次系统内存。
#     如果内存使用率超过 threshold (百分比)，则强制终止程序。
#     """
#     print(f"启动内存监控: 阈值={threshold}%, 检查间隔={check_interval}s")
#     while True:
#         try:
#             # 获取当前系统内存使用情况
#             mem = psutil.virtual_memory()
#             if mem.percent > threshold:
#                 print(
#                     f"!!! 内存告警 !!! 当前内存使用率 {mem.percent}% 超过阈值 {threshold}%。 "
#                     f"正在强制终止进程以保护系统稳定性..."
#                 )
#                 # 强制终止当前进程及其所有子线程
#                 # os._exit(1) 比 sys.exit() 更强力，能直接退出而不抛出 SystemExit 异常，
#                 # 这在多线程/多进程环境中更有效。
#                 os._exit(1)
#             time.sleep(check_interval)
#         except Exception as e:
#             print(f"内存监控发生错误: {e}")
#             time.sleep(check_interval)