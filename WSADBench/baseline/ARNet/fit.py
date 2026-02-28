# -*- coding: utf-8 -*-
"""
ARNet训练/评估逻辑
"""
import gc
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import time
import os
# import WSADBench.baseline.ARNet.options
import argparse

from common_utils.baseline_utils import get_gt, write_jsonl

mseloss = torch.nn.MSELoss(reduction='mean')
mseloss_vector = torch.nn.MSELoss(reduction='none')
binary_CE_loss = torch.nn.BCELoss(reduction='mean')
binary_CE_loss_vector = torch.nn.BCELoss(reduction='none')
# #一些训练参数补充
# parser = argparse.ArgumentParser(description='AR_Net')
# parser.add_argument('--k', type=int, default=4, help='value of k')

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


def hinger_loss(anomaly_score, normal_score):
        return F.relu((1 - anomaly_score + normal_score))

#按原论文要求
#DMIL loss 函数
def KMXMILL_individual(y_pred,                       #y_pred = y_pred
                       seq_len,
                       labels,
                       device,
                       k=4,
                       loss_type='CE',
                       ):

    """
    :param y_pred:
    :param seq_len:
    :param batch_size:
    :param labels:
    :param device:
    :param loss:
    :return:
    """
    seq_len = seq_len.cpu().numpy()
    k = np.ceil(seq_len/k).astype('int32')     # k为1维，size=60，且每个值为8
    instance_logits = torch.zeros(0).to(device)
    real_label = torch.zeros(0).to(device)     #[120,32]
    real_size = int(y_pred.shape[0])       #120
    for i in range(real_size):
        tmp, tmp_index = torch.topk(y_pred[i][:int(seq_len[i])], k=int(k[i]), dim=0) #挑选前k[i]个最大分数（即异常置信度最高的k个clip） #挑选前k[i]个最大分数（即异常置信度最高的k个clip）
        instance_logits = torch.cat((instance_logits, tmp), dim=0)    #[sum(k[i])]
        
        if labels[i] == 1:
            real_label = torch.cat((real_label, torch.ones((int(k[i]), 1)).to(device)), dim=0)
        else:
            real_label = torch.cat((real_label, torch.zeros((int(k[i]), 1)).to(device)), dim=0)
    real_label = real_label.squeeze(1)
    if loss_type == 'CE':
        milloss = binary_CE_loss(input=instance_logits, target=real_label)
        return milloss
    elif loss_type == 'MSE':
        milloss = mseloss(input=instance_logits, target=real_label)
        return milloss
    
#中心损失
def normal_smooth(y_pred, labels, device):

    """
    :param y_pred:
    :param seq_len:
    :param batch_size:
    :param labels:
    :param device:
    :param loss:
    :return:
    """
    normal_smooth_loss = torch.zeros(0).to(device)
    real_size = int(y_pred.shape[0])
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    for i in range(real_size):
        if labels[i] == 0:
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(y_pred[i],unbiased=False).unsqueeze(0)))  #有偏估计计算方差，无标签0样本时返回合理值0而非NaN
    # 处理无标签0样本的情况
        else:
            # 标签非0时添加值为0的1维张量
            zero_value = torch.tensor([0.0], device=device)
            normal_smooth_loss = torch.cat((normal_smooth_loss, zero_value))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)  
    return normal_smooth_loss



#计算总损失
def total_loss(y_pred, batch_size,seq_len,labels,device,k,DMIL_weight=1.000,Center_weight=20.000):
    """
    计算总损失函数，它包括两个部分：KMXMILL_individual 和 normal_smooth

    :param y_pred: 预测的 logits [batch_size, seq_len, 1]
    :param seq_len: 序列的长度 [batch_size,]
    :param labels: 视频标签 [batch_size, 1]
    :param device: 当前使用的设备（例如：cuda 或 cpu）
    :param DMIL_weight: DMIL损失权重
    :param Center_weight:中心损失权重
    :param args: 参数设置
    :return: 计算得到的总损失
    """
    device = y_pred.device
    # # 重塑预测结果
    batch_size = y_pred.shape[0]                                     
    y_pred = y_pred.view(batch_size, -1)            #展平，[120,32,1] -> [120,32] 


    m_loss = KMXMILL_individual(y_pred=y_pred,
                                seq_len=seq_len,
                                labels=labels,
                                device=device,
                                loss_type='CE',
                                k=k
                                )
    n_loss = normal_smooth(y_pred=y_pred,
                            labels=labels,
                            device=device,
                                )

    total_loss = float(DMIL_weight) * m_loss + float(Center_weight) * n_loss

    return total_loss

def fit_ARNet(model, optimizer, epochs, device, X_test, trainer,
                       verbose=True, normal_loader=None, anomaly_loader=None):
    """
    训练ARNeet模型
    
    Args:
        model: ARNet模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        epochs: 训练轮数
        batch_size: 批量大小
        labels: 视频级标签
        model_name: ARNet中模型名称（用于处理不同模型的前向传播）
        device: 计算设备
        DMIL_weight: DMIL损失权重
        Center_weight:中心损失权重
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

    DMIL_weight = trainer.DMIL_weight
    Center_weight = trainer.Center_weight
    model_name = trainer.model_name
    k = trainer.k

    if verbose:
        print(f"开始训练ARNet模型，共{epochs}轮...")
        print(f"设备: {device}")
        print(f"DMIL_loss权重: {DMIL_weight}, 中心损失权重: {Center_weight}")     #DMIL_loss 是计算每个视频中top-k的帧级标签和视频级标签的交叉熵损失
                                                                            #中心损失是计算正常片段分数与平均分数差的平方
    X_test, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames = X_test  # 拆包
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (normal_data, anomaly_data) in enumerate(zip(normal_loader, anomaly_loader)):
            optimizer.zero_grad()
            
            # 将数据移到设备     shape[60,32,2048]  B,T,F
            normal_data = normal_data.view(-1, normal_data.size(2), normal_data.size(3))
            normal_data = normal_data.to(device)  # 从[3,10, 32, 2048]变成[30, 32, 2048]
            anomaly_data = anomaly_data.view(-1, anomaly_data.size(2), anomaly_data.size(3))
            anomaly_data = anomaly_data.to(device)
            batch_size = normal_data.shape[0]
            # 合并正常和异常数据 [batch_size, 32, feature_dim]
            # 前32个为异常，后32个为正常  -> [B*2,32,2048]
            inputs = torch.cat([anomaly_data, normal_data], dim=0)   #改
            
            # 前向传播
            # 动态创建序列长度张量
            seq_len = torch.sum(torch.max(inputs.abs(), dim=2)[0] > 0, dim=1).to(device)  
            #lstm和其他模型不同
            if model_name == 'model_lstm':
                _, y_pred = model(inputs, seq_len)
            else:
                _, y_pred = model(inputs)
            labels = torch.cat([
                        torch.ones(batch_size).to(device),    # 异常视频标签
                        torch.zeros(batch_size).to(device)    # 正常视频标签
                    ])
            # 计算损失
            loss = total_loss(
                y_pred, batch_size,seq_len,labels,device,k=4,DMIL_weight=1.000,Center_weight=20.000
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')

            # 清理内存
            del normal_data, anomaly_data, loss
            torch.cuda.empty_cache()
        
        # 学习率调度
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')
            # --- [新增] 3. 最后一个Epoch进行测试与评估 (参考 Sultani 实现) ---
        if X_test is not None  and epoch == epochs - 1:
            trainer.fitted = True
            model.eval()  # 切换到评估模式
            with torch.no_grad():
                # 注意：这里传入的是解包后的 X_test_data
                scores = trainer.predict_proba(X_test)

                # --- Method 1: Clip/Video-level expanded GT (v2 in reference) ---
                # 直接将 clip 分数重复 16 次 (参考 fit.py 中的实现)
                prob = np.repeat(scores, 16)
                gt = get_gt(len(prob))
                test_auc_v2 = roc_auc_score(gt, prob)
                test_ap_v2 = average_precision_score(gt, prob)

                # --- Method 2: Frame-level Interpolated GT (v1 in reference) ---
                # 使用 _process_video_scores 进行精细化对齐
                frame_scores, frame_truth = _process_video_scores(
                    scores, video_shape, y_test_idx, y_test_gt,
                    y_test_gt_idx, num_clip_frames
                )
                test_auc = roc_auc_score(frame_truth, frame_scores)
                test_ap = average_precision_score(frame_truth, frame_scores)



                # 写入结果文件
                write_jsonl(model_name='ARNet', epochs=epoch, seed=trainer.seed, auc=test_auc, ap=test_ap,
                            res_type='frame')
                write_jsonl(model_name='ARNet', epochs=epoch, seed=trainer.seed, auc=test_auc_v2, ap=test_ap_v2,
                            res_type='clip')

                if verbose:
                    print(f"[Result] Frame-level (v1) - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
                    print(f"[Result] Clip-level  (v2) - AUC: {test_auc_v2:.4f}, AP: {test_ap_v2:.4f}")
                model.train()  # 恢复训练模式
        # # 每个epoch后再清理
        gc.collect()
        torch.cuda.empty_cache()

    
    if verbose:
        print("训练完成！")
    
    return train_history
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
