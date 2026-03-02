# -*- coding: utf-8 -*-
"""
Sultani方法训练逻辑
基于MIL (Multiple Instance Learning) 的弱监督异常检测训练
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from WSADBench.baseline.VadClip.clip.myUtils import setup_logging
from common_utils.baseline_utils import get_gt, write_jsonl

# logger = setup_logging(log_dir='/gpudata/wsad/working_space/zsy/WSADBench/WSADBench/datasets/logs', name='vadclip')
from sklearn.metrics import roc_auc_score, average_precision_score

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
# 需要改标签（ucf是14个）
# label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
#                       'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
#                       'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
#                       'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})



def fit(model, optimizer, epochs, device, X_test, trainer,
                verbose=True, normal_loader=None, anomaly_loader=None,X_test_extra=None):
    model.train()
    
    train_history = {
        'loss': [],
        'epoch_time': []
    }

    # model.device = device  # 设置设备
    if verbose:
        print(f"Starting VadClip model training, total {epochs} epochs...")
        print(f"Device: {device}")
    prompt_text = get_prompt_text(trainer.label_map)
    # logger.info('start train ...')
    X_test, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames = X_test  # 拆包
    vid_kind, vid_source_clips_num,crops_num = X_test_extra
    best_epoch = -1
    best_auc = 0.0
    best_ap = 0
    best_epoch_v2 = -1
    best_auc_v2 = 0.0
    best_ap_v2 = 0
    # gt = np.repeat(gt, 10)  # 应该要按长度重复
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        loss_total1 = 0
        loss_total2 = 0
        for batch_idx, (normal_data, anomaly_data) in enumerate(zip(normal_loader, anomaly_loader)):
            step = 0
            optimizer.zero_grad()
            normal_features, normal_label, normal_lengths = normal_data  # 64个正常视频，64个异常视频，加起来是128个
            anomaly_features, anomaly_label, anomaly_lengths = anomaly_data
            # 将数据移到设备

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, trainer.label_map).to(device)

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
            loss3 = loss3 / (len(trainer.label_map)-1) * model.lam  # 1 × 10−4 and 1 × 10−1 on XD-Violence and UCF-Crime,

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
            #     logger.info(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            # break
        # 学习率调度
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        if X_test is not None and trainer.is_test and epoch == epochs - 1:
            trainer.fitted= True
            with torch.no_grad():
                scores = trainer.predict_proba(X_test, vid_kind, vid_source_clips_num,crops_num)  # 得分696270
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
                # write_jsonl(model_name='vadclip', epochs=epoch, auc=test_auc, ap=test_ap, seed=trainer.seed,
                #             res_type='frame')
                # write_jsonl(model_name='vadclip', epochs=epoch, auc=test_auc_v2, ap=test_ap_v2, seed=trainer.seed,
                #             res_type='clip')
                # logger.info(f"cur epoch:{epoch} AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f} best epoch:{best_epoch}, best auc:{best_auc:.4f}, best ap:{best_ap:4f}")
                # logger.info(
                #     f"cur epoch_v2:{epoch} AUCROC_v2: {test_auc_v2:.4f}, AUCPR_v2: {test_ap_v2:.4f} best epoch_v2:{best_epoch_v2}, best auc_v2:{best_auc_v2:.4f}, best ap_v2:{best_ap_v2:4f}")

        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)
        if verbose:
            print(
                f'Epoch {epoch + 1}/{epochs} completed - Average loss: {avg_epoch_loss:.6f}, Time elapsed: {epoch_time:.2f}s')

    if verbose:
        print("Training completed!")
    
    return train_history


#
#
# def fit_sultani_main(X_train, y_train, model, optimizer, epochs, batch_size, device,
#                        verbose=True, vid_info=None, clip_num=None, vid_kind=None, crops_num=None, X_test=None, trainer=None, X_test_extra=None):
#     """
#     Args:
#         X_train: 训练特征 [n_samples, feature_dim]
#         y_train: 训练标签 [n_samples]
#         model: Sultani模型
#         optimizer: 优化器
#         epochs: 训练轮数
#         batch_size: 批量大小
#         device: 计算设备
#         sparsity_weight: 稀疏性损失权重
#         smoothness_weight: 平滑性损失权重
#         verbose: 是否打印训练信息
#         vid_info: 每个片段对应的视频id [n_samples]
#
#     Returns:
#         训练历史
#
#     """
#     model.train()
#     ncrop = crops_num
#     seg = model.visual_length
#     feature = model.input_dim
#
#     def split_by_seg_list(X, seg_list, feature):
#         X = X.reshape(-1, ncrop, feature)  # 【seg, crop:10, 2048]
#         segments = []
#         start = 0
#         for seg_len in seg_list:
#             end = start + seg_len
#             for i in range(ncrop):
#                 segment = X[start:end, i]  # shape: [1, seg_len, 2048]
#                 segments.append(segment)
#             start = end
#         return segments
#
#     clip_num = list(clip_num.values())
#
#     # 按X的形状，判断X是否是32seg
#     if len(clip_num) * 32 * ncrop == X_train.shape[0]:
#         print('32 seg')
#         clip_num = [32 for i in range(len(clip_num))]  # 32 seg 版
#     X_train = split_by_seg_list(X_train, clip_num, feature)  # (16100, seg, 2048)
#     y_train = split_by_seg_list(y_train, clip_num, 1)  # 16100个(, seg, 1)(mask)
#     y_train = [int(item[0, 0]) for item in y_train]  # 16100个0、1标签list
#
#     X_normal_videos = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
#     X_anomaly_videos = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1]
#
#     vid_kind_expand = [vid_kind[i] for i in range(len(vid_kind)) for _ in range(ncrop)]  # 包含str标签，并复制
#     clip_num_expand = [min(seg, clip_num[i]) for i in range(len(clip_num)) for _ in range(ncrop)]  # 注意上界
#     # 4. 对应扩展 vid_kind 和 clip_num：每个视频被切成了10段
#     # 4. 用 mask 提取对应部分
#     normal_vid_kind = [vid_kind_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
#     abnormal_vid_kind = [vid_kind_expand[i] for i in range(len(y_train)) if y_train[i] == 1]
#
#     normal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 0]
#     abnormal_clip_num = [clip_num_expand[i] for i in range(len(y_train)) if y_train[i] == 1]
#
#
#
#     # todo:这三者长度一致：X_normal_videos, normal_vid_kind, normal_clip_num；平衡正常和异常视频的数量，使用均匀采样
#     #         只增加样本数量，不减少原有样本
#     def balance_video_samples(normal_videos, normal_vid_kind, normal_clip_num, anomaly_videos, abnormal_vid_kind, abnormal_clip_num
# ):
#         """
#         平衡正常和异常视频的数量，使用均匀采样
#         只增加样本数量，不减少原有样本
#
#         Args:
#             normal_videos: 正常视频列表
#             anomaly_videos: 异常视频列表
#
#         Returns:
#             balanced_normal_videos, balanced_anomaly_videos: 平衡后的视频列表
#         """
#         normal_count = len(normal_videos)
#         anomaly_count = len(anomaly_videos)
#
#         logging.info(f"原始数据: 正常视频 {normal_count} 个, 异常视频 {anomaly_count} 个")
#
#         if normal_count == 0 or anomaly_count == 0:
#             logging.warning("正常或异常视频数量为0，无法平衡采样")
#             return normal_videos, anomaly_videos
#
#         # 确定目标数量（取较大值，只增加不减少）
#         target_count = max(normal_count, anomaly_count)
#
#         def uniform_upsample(videos, target_size):
#             """均匀上采样函数 - 只增加样本，不减少"""
#             if len(videos) >= target_size:
#                 # 如果数量已经足够，直接返回原始数据
#                 return videos
#             else:
#                 # 如果数量不足，均匀重复采样来增加样本
#                 original_videos = videos.copy()  # 保留所有原始样本
#                 additional_needed = target_size - len(videos)
#
#                 # 均匀选择要重复的样本
#                 if additional_needed > 0:
#                     repeat_indices = np.linspace(0, len(videos) - 1, additional_needed, dtype=int)
#                     additional_videos = [videos[i] for i in repeat_indices]
#                     return original_videos + additional_videos
#                 else:
#                     return original_videos
#
#         # 均匀上采样
#         balanced_normal = uniform_upsample(normal_videos, target_count)
#         balanced_anomaly = uniform_upsample(anomaly_videos, target_count)
#         balanced_normal_vid_kind = uniform_upsample(normal_vid_kind , target_count)
#         balanced_anomaly_vid_kind = uniform_upsample(abnormal_vid_kind , target_count)
#         balanced_normal_clip_num = uniform_upsample(normal_clip_num , target_count)
#         balanced_anomaly_clip_num = uniform_upsample(abnormal_clip_num , target_count)
#         logging.info(f"平衡后数据: 正常视频 {len(balanced_normal)} 个, 异常视频 {len(balanced_anomaly)} 个")
#         return balanced_normal, balanced_normal_vid_kind, balanced_normal_clip_num, balanced_anomaly, balanced_anomaly_vid_kind, balanced_anomaly_clip_num
#
#     # 执行平衡采样
#     X_normal_videos, normal_vid_kind, normal_clip_num, X_anomaly_videos, abnormal_vid_kind, abnormal_clip_num = balance_video_samples(
#         X_normal_videos, normal_vid_kind, normal_clip_num, X_anomaly_videos, abnormal_vid_kind, abnormal_clip_num
#     )
#
#     # 6. 构建 Dataset 和 DataLoader
#     normal_dataset = VideoDataset(X_normal_videos, normal_vid_kind, normal_clip_num, seg)
#     anomaly_dataset = VideoDataset(X_anomaly_videos, abnormal_vid_kind, abnormal_clip_num, seg)
#
#     normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
#     anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
#
#
#
#     # 调用主训练函数
#     return fit_sultani(model, optimizer, normal_loader,anomaly_loader, epochs, device,
#                        verbose,X_test=X_test, trainer=trainer, X_test_extra=X_test_extra)