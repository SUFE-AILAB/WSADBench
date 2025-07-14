# -*- coding: utf-8 -*-
"""
ARNet训练/评估逻辑
"""
import gc
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import os
# import WSADBench.baseline.ARNet.options
import argparse

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
                       args,
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
    # [train_video_name, start_index, len_index] = stastics_data
    seq_len = seq_len.cpu().numpy()
    k = np.ceil(seq_len/args.k).astype('int32')     # k为1维，size=60，且每个值为8
    instance_logits = torch.zeros(0).to(device)
    real_label = torch.zeros(0).to(device)
    # print(f'y_pred shape: {y_pred.shape}')       #[120,32]
    real_size = int(y_pred.shape[0])       #120
    # print(f'real_size:,{real_size}') 
    # print(f"labels: {labels.shape}")  #[120,1]
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    for i in range(real_size):
        tmp, tmp_index = torch.topk(y_pred[i][:seq_len[i]], k=int(k[i]), dim=0) #挑选前k[i]个最大分数（即异常置信度最高的k个clip）
        # top_index = np.zeros(len_index[i].numpy())           #tmp:一维，16个数
        # top_predicts = np.zeros(len_index[i].numpy())
        # top_index[tmp_index.cpu().numpy() + start_index[i].numpy()] = 1
        # if train_video_name[i][0] in log_statics:
        #     log_statics[train_video_name[i][0]] = np.concatenate((log_statics[train_video_name[i][0]], np.expand_dims(top_index, axis=0)),axis=0)
        # else:
        #     log_statics[train_video_name[i][0]] = np.expand_dims(top_index, axis=0)
        instance_logits = torch.cat((instance_logits, tmp), dim=0)    #[sum(k[i])]
        
        # label_i = labels[i].max().item()  # 只要存在一个为1则判为异常,视频级标签
        if labels[i] == 1:
            real_label = torch.cat((real_label, torch.ones((int(k[i]), 1)).to(device)), dim=0)
        else:
            real_label = torch.cat((real_label, torch.zeros((int(k[i]), 1)).to(device)), dim=0)
        #real_label的shape为[480,1]
        #让每个cilp与视频标签对齐  （按原论文要求添加修改）
        # if real_label.ndim == 2:  # [N, 1]
        #     real_label = real_label.unsqueeze(1)  # [N, 1, 1]
        # real_label = real_label.expand(-1, instance_logits.shape[1], -1)  # [N, k, 1]
    # instance_logits = instance_logits.unsqueeze(1)       #[480,] -> [480,1]  对齐和real_label的维度
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
    # print(f'real_size from normal_smooth:,{real_size}')
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    for i in range(real_size):
        # label_i = labels[i].max().item()  # 只要存在一个为1则判为异常,视频级标签!!!!这行代码需要检查
        if labels[i] == 0:
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(y_pred[i],unbiased=False).unsqueeze(0)))  #有偏估计计算方差，无标签0样本时返回合理值0而非NaN
           #未计算出normal_smooth_loss
    # 处理无标签0样本的情况
        else:
            # 标签非0时添加值为0的1维张量
            zero_value = torch.tensor([0.0], device=device)
            normal_smooth_loss = torch.cat((normal_smooth_loss, zero_value))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)  
    return normal_smooth_loss



#计算总损失
def total_loss(y_pred, batch_size,seq_len,labels,device,args,DMIL_weight=1.000,Center_weight=20.000):
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
    #！！！！！这里报错y_pred还不是tensor形式
    # y_pred = y_pred.squeeze(2)    #[60, 32, 1] -> [60, 32]  
    device = y_pred.device
    # # 重塑预测结果
    batch_size = y_pred.shape[0]                                       #插个点，有可能是把y_pred展平
    y_pred = y_pred.view(batch_size, -1)            #!!!这段代码可能没起作用


    # for i, data in enumerate(train_loader):     # i为索引 ， data为加载的视频数据   #报错，train_loader没传入数据
    # # for i in range(batch_size):

    #         # 1. 拼接异常和正常样本特征
    #     [anomaly_features, normaly_features], [anomaly_label, normaly_label], stastics_data = data
    #     features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
    #     # print(f"features shape: {features.shape}")  # [120, 60, 10, 2048]       
    #     videolabels = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
        
    #     # 先在 feature 维上 max（最后一维）
    #     x = torch.max(features.abs(), dim=2)[0]  # [120, 60, 10]
    #     # 再在 frame 维上 max
    #     mask = torch.max(x, dim=1)[0] > 0        # [120, 60]
    #     seq_len = mask.sum(dim=0).cpu().numpy()  # [120,]
    #     print(f"seq_len: {seq_len.shape}")  # [batch_size,]
    #     features = features[:, :np.max(seq_len), :]
    #     print(f"features shape: {features.shape}") 

    #     # features = torch.from_numpy(features).float().to(device)
    #     features = features.float().to(device)
    #     videolabels = videolabels.float().to(device)
    #     # final_features, y_pred_linear, y_pred = model(features)
    #     final_features, y_pred = model(features)
    #     # if args.model_name == 'model_lstm':
    #     #     final_features, y_pred = model(features, seq_len)
    #     # else:
    #     #     final_features, y_pred = model(features)
    #     DMIL_weight=1.000
    #     Center_weight=20.000
    # with torch.no_grad():   #包装不需要梯度计算的部分，释放中间变量
    m_loss = KMXMILL_individual(y_pred=y_pred,
                                seq_len=seq_len,
                                labels=labels,
                                device=device,
                                loss_type='CE',
                                args=args
                                )
    n_loss = normal_smooth(y_pred=y_pred,
                            labels=labels,
                            device=device,
                                )

    total_loss = float(DMIL_weight) * m_loss + float(Center_weight) * n_loss

    return total_loss

def fit_ARNet(model, labels,optimizer, train_loader, epochs,device,args,DMIL_weight=1.000,Center_weight=20.000,
          verbose=True,scheduler=None):
    """
    训练ARNeet模型
    
    Args:
        model: ARNet模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        epochs: 训练轮数
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


    if verbose:
        print(f"开始训练ARNet模型，共{epochs}轮...")
        print(f"设备: {device}")
        print(f"DMIL_loss权重: {DMIL_weight}, 中心损失权重: {Center_weight}")     #DMIL_loss 是计算每个视频中top-k的帧级标签和视频级标签的交叉熵损失
                                                                            #中心损失是计算正常片段分数与平均分数差的平方
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (normal_data, anomaly_data) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 将数据移到设备    注意下，插个眼   shape[60,32,2048]  B,T,F
            normal_data = normal_data.to(device)
            anomaly_data = anomaly_data.to(device)
            
            batch_size = normal_data.shape[0]
            
            # 合并正常和异常数据 [batch_size, 32, feature_dim]
            # 前32个为异常，后32个为正常  -> [B*2,32,2048]
            inputs = torch.cat([anomaly_data, normal_data], dim=0)   #改
            
            # 前向传播
            # final_features, y_pred = model(inputs)  # [batch_size*2, 32, 1]   #发现model(inputs)有两个返回 ,忽略原模型输出x(特征)
            
            # 创建序列长度张量（假设每个视频都是完整的32个片段）
            seq_len = torch.full((batch_size*2,), 32, dtype=torch.int32).to(device)
            #lstm和其他模型不同
            if args.model_name == 'model_lstm':
                _, y_pred = model(inputs, seq_len)
            else:
                _, y_pred = model(inputs)
            
            # print(f"y_pred的shape:{y_pred.shape}") #[120,32,1]
            # ## 确保video_labels不为None
            # if labels is None:
            #     raise ValueError("labels must be provided")

            # 计算损失
            # 计算损失
            loss = total_loss(
                y_pred, batch_size,seq_len,labels,device,args,DMIL_weight=1.000,Center_weight=20.000
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
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        train_history['loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')
        
        # # 每个epoch后再清理
        gc.collect()
        torch.cuda.empty_cache()

    
    if verbose:
        print("训练完成！")
    
    return train_history


def fit_ARNet_main(X_train, y_train, model,optimizer, epochs, batch_size, device,
                      DMIL_weight,Center_weight, verbose=True):
    # global train_loader
    """
    Args:
        X_train: 训练特征 [n_samples, feature_dim]
        y_train: 训练标签 [n_samples]
        labels:拼接后的视频标签
        model: ARNet模型
        optimizer: 优化器
        epochs: 训练轮数
        batch_size: 批量大小
        device: 计算设备
        sparsity_weight: 稀疏性损失权重
        smoothness_weight: 平滑性损失权重
        verbose: 是否打印训练信息
        
    Returns:
        训练历史
    """
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
    #reshape为->[7960,32]
    X_normal_videos = X_normal[:segments_num * segments_per_video].reshape(segments_num, segments_per_video, -1)
    X_anomaly_videos = X_anomaly[:segments_num * segments_per_video].reshape(segments_num, segments_per_video, -1)
    
    # 创建训练数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_normal_videos),
        torch.FloatTensor(X_anomaly_videos)
    )     #torch方法执行后 [7960,32,2048]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建视频级标签
            # 正常视频标签为0，异常视频标签为1
    labels = torch.cat([
        torch.ones(batch_size).to(device),    # 异常视频标签
        torch.zeros(batch_size).to(device)    # 正常视频标签
    ])

    # 创建参数对象
    args = argparse.Namespace(k=4, model_name='model_single')    #可根据需要修改参数，目前model_single是OK的
    # 调用主训练函数
    return fit_ARNet(model ,labels,optimizer, train_loader, epochs, device,args,
                      DMIL_weight, Center_weight,verbose=True)
    