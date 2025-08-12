# -*- coding: utf-8 -*-
"""
TargAD方法模型架构定义
基于"A Robust Prioritized Anomaly Detection when Not
 All Anomalies are of Primary Interest"论文实现
弱监督异常检测方法
"""

from ast import mod
import sys
import math
import time
import copy
import random
import argparse
import logging
import numpy as np
import pandas as pd

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
import pickle
import matplotlib.ticker as ticke

import warnings

from implementing.UMIL.datasets.rand_augment import auto_contrast
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report,f1_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from WSADBench.baseline.TargAD.model import AutoEncoder,Classifier
import gc
from sklearn.model_selection import train_test_split


def shuffle(X, Y, S):
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index], S[random_index]
    
def shuffle_u(X, Y):
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index] 
    
#！！！可能修改增加min(len(X),(i+1)*BATCH_SIZE)
def getBatch(X, Y, BATCH_SIZE):
    while True:
        X, Y = shuffle_u(X, Y)
        for i in range(int(len(X)/BATCH_SIZE)):
            yield X[i*BATCH_SIZE:(i+1)*BATCH_SIZE], Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    
def getBatchWeigt(X, Y, W, BATCH_SIZE):
    while True:
        X, Y, W = shuffle(X, Y, W)
        for i in range(int(len(X)/BATCH_SIZE)):
            yield X[i*BATCH_SIZE:(i+1)*BATCH_SIZE], Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE], W[i*BATCH_SIZE:(i+1)*BATCH_SIZE] 

def calculate_entropy(probs):
    ent = -np.sum(probs * np.log(probs + 1e-8))
    return ent

def calculate_energy(logits):
    energy = torch.logsumexp(logits, dim = 1)
    return energy

def calculate_discrepancy(logits):
    #输入：[n x f]
    #输出：[n x f]
    energy_discrepancy = []
    energy = torch.logsumexp(logits, dim = 1) #[batch_size,]
    for i, i_logit in enumerate(logits):
        i_logit = i_logit.cpu().detach().numpy()
        i_logit = np.array(i_logit, dtype = np.float128)
        delete_max = np.log(np.sum(np.exp(i_logit)) - np.max(np.exp(i_logit)))
        energy_discrepancy.append(energy[i].cpu().detach().numpy()-delete_max)
    return energy_discrepancy

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

from sklearn.metrics import roc_auc_score, precision_recall_curve,auc
def metric(y_true, y_score, pos_label=1):
    aucroc = roc_auc_score(y_true, y_score)
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    aucpr = auc(recall, precision)
    return aucroc, aucpr

#损失函数
class SoftCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, label, weight=None):
        assert logit.size() == label.size(), "logit.size() != label.size()"
        dim = logit.dim()
        # 选出预测概率最大的那一类的概率
        max_logit = logit.max(dim - 1, keepdim=True)[0]
        logit = logit - max_logit
        exp_logit = logit.exp()
        exp_sum = exp_logit.sum(dim - 1, keepdim=True)
        prob = exp_logit / exp_sum
        log_exp_sum = exp_sum.log()
        neg_log_prob = log_exp_sum - logit

        if weight is None:
            weighted_label = label
        else:
            if weight.size() != (logit.size(-1),):
                raise ValueError(
                    "since logit.size() = {}, weight.size() should be ({},), but got {}".format(
                        logit.size(),
                        logit.size(-1),
                        weight.size(),
                    )
                )
            size = [1] * label.dim()
            size[-1] = label.size(-1)
            weighted_label = label * weight.view(size)
        # ctx为context的缩写，自定义的forward和backward第一个参数必须是ctx,上下文管理器
        # save_for_backward能够保存forward()静态方法中的张量,从而可以在backward()静态方法中调用
        ctx.save_for_backward(weighted_label, prob)
        out = (neg_log_prob * weighted_label).sum(dim - 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        weighted_label, prob = ctx.saved_tensors
        old_size = weighted_label.size()
        # num_classes
        K = old_size[-1]
        # batch_size
        B = weighted_label.numel() // K

        grad_output = grad_output.view(B, 1)
        weighted_label = weighted_label.view(B, K)
        prob = prob.view(B, K)
        grad_input = grad_output * (prob * weighted_label.sum(1, True) - weighted_label)
        grad_input = grad_input.view(old_size)
        return grad_input, None, None
    
def soft_cross_entropy(logit, label, weight=None, reduce=None, reduction="mean"):
    if weight is not None and weight.requires_grad:
        raise RuntimeError("gradient for weight is not supported")
    losses = SoftCrossEntropyFunction.apply(logit, label, weight)
    reduction = {
        True: "mean",
        False: "none",
        None: reduction,
    }[reduce]
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        raise ValueError("invalid value for reduction: {}".format(reduction))


def fit_TargAD(X_labelled_anomaly, Y_labelled_anomaly,X_unlabelled, Y_unlabelled,X_val,Y_val,model,
    optimizer,num_centroid,num_anomaly_classes, stage_1_epochs, stage_2_epochs,kmeans_batch,stage_1_batch, stage_2_batch,anomaly_batch,ood_batch,device,input_dim,embedding_dim,loss_oe=0.1,loss_re=1,
    stage_one_lr=0.0001,stage_two_lr=0.00001,weight_decay=1e-6,verbose=True,scheduler=None):


    if verbose:
        print(f"开始训练TargAD模型，第一阶段共{stage_1_epochs}轮，第二阶段{stage_2_epochs}轮...")
        print(f"设备: {device}")
    autoencoder = AutoEncoder(input_dim=input_dim, num_features=embedding_dim).to(device)
    """第一阶段训练：自监督AE预训练 + 聚类 + 重构筛选"""  
    X_filter, Y_filter, X_deleted, Y_deleted, score_delete = autoencoder_pretrain(X_unlabelled,Y_unlabelled, X_labelled_anomaly,autoencoder, input_dim, embedding_dim, num_centroid,
                                                                                   stage_1_epochs,kmeans_batch,stage_1_batch, stage_one_lr,filter=0.05, weight_decay=1e-6)

    """第二阶段训练"""
    model.train()
    train_history = {
        'avg_epoch_loss': [],
        'epoch_time': []
    }

    if verbose:
        print(f"开始训练TargAD模型，第一阶段已完毕，第二阶段{stage_2_epochs}轮...")
        print(f"设备: {device}")

    # 得到已知异常的batch
    if anomaly_batch > len(X_labelled_anomaly):  #避免超出异常
        anomaly_batch = len(X_labelled_anomaly)
        # raise ValueError(f"异常样本数量不足，已知异常样本数量为{len(X_labelled_anomaly)}，请调整anomaly_batch参数，目前已自动调整为现有数量")
    gen_anomaly = getBatch(X_labelled_anomaly, Y_labelled_anomaly, anomaly_batch)
    
    # 真实过滤后的潜在异常是有大部分已知异常、未知异常以及困难正常
    # 若将过滤后的潜在异常全部作为ood，则不合适，所以需要设置权重，对于已知异常权重小，
    ood_data = X_deleted #[346,10]
    ood_data_y_true = Y_deleted  #[346,]  # 无标签，设为0
    # 分数越大，越异常越有可能是属于已知异常，权重越小
    # 权重初始化,使权重在(0,1)之间，分数越小权重越大
    ood_data_w =  (np.max(score_delete) - score_delete) / (np.max(score_delete) - np.min(score_delete))

    # 簇(过滤后的可靠正常)+已知异常的标签
    # 希望过滤后的可靠正常为0，未知异常对应于已知异常为1/已知异常的类 形式如[0, 0, 0, 0, 0, 1/3, 1/3, 1/3]
    ood_data_y = np.hstack((np.zeros((ood_data.shape[0], num_centroid)), np.ones((ood_data.shape[0], num_anomaly_classes)) * (1.0 / num_anomaly_classes)))
    
    # 返回一个batch_size的ood数据
    gen_ood = getBatchWeigt(ood_data, ood_data_y, ood_data_w, ood_batch)
    
    print('Starting Stage_Two...')

    num_subgroups = num_centroid + num_anomaly_classes
    model = Classifier(input_dim, embedding_dim, num_subgroups).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=stage_two_lr, weight_decay=weight_decay)

    for epoch in range(stage_2_epochs):
        epoch_start_time = time.time()
        epoch_final_loss = 0.0
        batch_count = 0
        # scheduler.step()
        # 过滤后的可靠正常
        stage2_nbBatch = X_filter.shape[0] // stage_2_batch
        if stage2_nbBatch == 0:
            stage2_nbBatch += 1      #避免除0情况
        X_filter, Y_filter = shuffle_u(X_filter, Y_filter)
        # model.train()


        for i in range(stage2_nbBatch):

            model.train()
            # 过滤后的可靠正常样本(0~4)，其batch_size为128
            x_normal = X_filter[i * stage_2_batch: min((i + 1) * stage_2_batch, X_filter.shape[0])]
            x_normal = torch.tensor(x_normal).float().to(device)   #[128,10]
            y_normal = Y_filter[i * stage_2_batch: min((i + 1) * stage_2_batch, X_filter.shape[0])]
            y_normal = torch.tensor(y_normal).to(dtype=torch.int64) #[128,]
            x_normal_e, logit_normal = model(x_normal)

            # 已知异常, 其batch_size = 32
            x_vandal, y_vandal = gen_anomaly.__next__()
            x_vandal = torch.tensor(x_vandal).float().to(device)   #[32,10]
            y_vandal = np.reshape(y_vandal, y_vandal.shape[0])
            y_vandal = torch.tensor(y_vandal).to(dtype=torch.int64)   #[32,]


            # 将过滤后的可靠正常和已知异常合并
            # 这里的标签为0～8
            x = torch.cat((x_normal, x_vandal), 0).to(device)   
            y = torch.cat((y_normal, y_vandal), 0).to(device)
            x, y = shuffle_u(x, y)
            x_e, y_pred = model(x)

            # CrossEntropy Loss
            CELoss = nn.functional.cross_entropy(y_pred, y)

            # Regularization Loss:使模型的置信度变高
            y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
            y_pred = F.softmax(y_pred, 1)
            regularityLoss = torch.mean(torch.mean(torch.sum(-y_pred * torch.log(y_pred + 1e-8), 1), 1))

            # 过滤后的潜在异常,其batch_size为32
            # 每一个ood的标签都是(0,0,0,0,0,1/3,1/3,1/3)
            x_ood, y_ood, w_ood = gen_ood.__next__()  #[32,10] , [32,8], [32,]
            x_ood = torch.tensor(x_ood).float().to(device)
            y_ood = torch.tensor(y_ood).to(device)
            w_ood = torch.tensor(w_ood).float().to(device)

            x_ood_e, logits_oe = model(x_ood)


            # 相当于监督Loss
            loss_oe = torch.mul(soft_cross_entropy(logits_oe, y_ood, reduction = "none"), w_ood).mean()
            # loss_oe = torch.mul(soft_cross_entropy(torch.stack(selected_unknown_entropy), y_ood[:math.ceil(len(sorted_entropy_top) * 0.2)], reduction = "none"), torch.stack(selected_unknown_weight)).mean()
            Final_Loss =  CELoss  + loss_oe * loss_oe +  loss_re * regularityLoss

            optimizer.zero_grad()
            Final_Loss.backward()
            optimizer.step()

            epoch_final_loss += Final_Loss.item()
            batch_count += 1
        # print('Epoch {}/{}\t epoch_final_loss: {:.2f}\t'.format(epoch+1,stage_2_epochs,epoch_final_loss/stage2_nbBatch))
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_final_loss / batch_count if batch_count > 0 else 0
        
        train_history['avg_epoch_loss'].append(avg_epoch_loss)
        train_history['epoch_time'].append(epoch_time)
        
        if verbose:
            print(f'Epoch {epoch+1}/{stage_2_epochs} 完成 - 平均损失: {avg_epoch_loss:.6f}, 耗时: {epoch_time:.2f}s')
        

        # # # 每个epoch后再清理
        # gc.collect()
        # torch.cuda.empty_cache()
        
        x_e, y_logit = model(torch.tensor(ood_data).float().to(device))
        con_confidence = torch.max(F.softmax(y_logit, dim = 1), dim=1)[0].cpu().detach().numpy()
        ood_data_w =  (np.max(con_confidence) - con_confidence) / (np.max(con_confidence) - np.min(con_confidence))
        gen_ood = getBatchWeigt(ood_data, ood_data_y, ood_data_w, ood_batch)   

    """验证阶段，计算出最佳阈值，实现关注异常/正划分部分"""
    print("model 进入验证阶段, 开始实现Target功能......")
    model.eval()
    _, y_logit = model(torch.tensor(X_val).float().to(device))

        
    # 未知异常不抛出
    y_val = copy.deepcopy(Y_val)  
    # 将已知异常的标签全部置于1，未知异常和正常置为0， 加载的数据集只有0，1标签
    for i in range(num_anomaly_classes):
        y_val[(y_val == num_centroid + i)] = 1
    y_val[(y_val == -1) | (y_val == -2)] = 0

    prob = torch.max(F.softmax(y_logit, dim = 1)[:,-num_anomaly_classes:], dim=1)[0].cpu().detach().numpy()
        
    # 利用val data选择阈值
    best_threshold= compute_thresholds(model, X_val, Y_val,num_centroid, num_anomaly_classes,device)
    print("best_threshold:{}".format(best_threshold))

    # 用阈值给测试集的预测结果重新赋值
    y_pred = []
    probs = F.softmax(y_logit, dim = 1)[:,-num_anomaly_classes:]
    logits_anomaly = y_logit[:,-num_anomaly_classes:]
    
    probs_normal = F.softmax(y_logit, dim = 1)[:,:num_centroid]
    sum_normal_logits = torch.sum(probs_normal, dim=1).cpu().detach().numpy()
    
    for i, sum_logit in enumerate(sum_normal_logits):
        if sum_logit > num_centroid/num_subgroups:
            pred_label = 0   #正常打0
        else:
            energy = calculate_discrepancy(logits_anomaly[i, -num_anomaly_classes:].unsqueeze(0))
            energy = torch.Tensor(energy).squeeze(0)
            if energy > best_threshold: # 已知异常打上1
                pred_label = 1 
            else:
                pred_label = -2  # 剩余样本的预测标签
        y_pred.append(pred_label)
    
    aucroc, aucpr= metric(y_true=y_val, y_score=y_pred)
    print(f"验证集AUC-ROC: {aucroc:.4f}, AUC-PR: {aucpr:.4f}")
    if verbose:
        print("训练完成！")
    
    return train_history, model,best_threshold


def compute_thresholds(model, inputs, labels,num_centroid, num_anomaly_classes,device):
    
    entropy_all = []
    num_subgroups = num_centroid + num_anomaly_classes
    # 将已知异常的标签赋为1，正常为0，未知异常为-2
    labels_ = copy.deepcopy(labels)         #加载的数据集只有0，1标签 
    for i in range(num_anomaly_classes):
        labels_[(labels_ == num_centroid + i)] = 1
    labels_[(labels_ == -1)] = 0
    labels_[(labels_ == -2)] = -2
    
    
    with torch.no_grad():

        _, logits = model(torch.tensor(inputs).float().to(device))
        logits_anomaly = logits[:,-num_anomaly_classes:] #关注异常类别得分
        
        energy_all = calculate_discrepancy(logits_anomaly[:,:])
        
        # 阈值是关注目标异常，故将正常和未知异常置为0
        target_labels = copy.deepcopy(labels)  
        for i in range(num_anomaly_classes):
            target_labels[(target_labels == num_centroid + i)] = 1
        target_labels[(target_labels == -1)] = 0
        target_labels[(target_labels == -2)] = 0      #这里与使用数据标签对上，变为0，1标签
        
        #前n维正常的和
        probs_normal = F.softmax(logits, dim = 1)[:,:num_centroid]
        sum_normal_logits = torch.sum(probs_normal, dim=1).cpu().detach().numpy()
        #删去被预测为正常的，在剩下的样本中找到阈值
        # 通过entropy小于某个阈值，识别目标异常
        new_energy_all = []
        new_target_labels = []
        for i, sum_logit in enumerate(sum_normal_logits):
            if sum_logit <= num_centroid/num_subgroups:
                new_energy_all.append(energy_all[i])
                new_target_labels.append(target_labels[i])
                        
        fpr_1, tpr_1, thresholds_1 = roc_curve(new_target_labels, new_energy_all)

        optimal_th_1, optimal_point_1 = Find_Optimal_Cutoff(TPR=tpr_1, FPR=fpr_1, threshold=thresholds_1)
        
    return optimal_th_1  

def fit_TargAD_main(X_train, y_train,mask,model, autoencoder,optimizer,num_centroid,num_anomaly_classes,
                    stage_1_epochs, stage_2_epochs,kmeans_batch,stage_1_batch, stage_2_batch,anomaly_batch,ood_batch,device,input_dim,embedding_dim,loss_oe,loss_re,stage_one_lr,stage_two_lr,weight_decay,if_split,split_error, verbose=True):
    """
    整体训练流程函数：包括第一阶段（AE+聚类+筛选）与第二阶段（分类器+软标签训练）
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        mask: 训练集样本的标签掩码，0表示无标签，1表示已知标签
        model: 分类器模型
        autoencoder: 自编码器模型
        optimizer: 优化器
        num_centroid: 聚类簇的数量
        num_anomaly_classes: 异常类别的数量
        stage_1_epochs: 第一阶段训练轮数
        stage_2_epochs: 第二阶段训练轮数
        kmeans_batch: kmeans的batch大小
        stage_1_batch: 第一阶段batch大小
        stage_2_batch: 第二阶段batch大小
        anomaly_batch: 已知异常batch大小
        ood_batch: 过滤后的潜在异常batch大小
        device: 设备(cpu/gpu)
        input_dim: 输入特征维度
        embedding_dim: 嵌入特征维度
        loss_oe: ood_loss的权重系数
        loss_re: 正则化损失的权重系数
        stage_one_lr: 第一阶段学习率
        stage_two_lr: 第二阶段学习率
        weight_decay: 权重衰减系数
        if_split: 是否划分验证集，True/False
        split_error: 划分错误时的处理方式，"raise"/"auto"
        verbose: 是否打印训练信息，True/False
    Returns:
        model: 训练好的分类器模型
        best_threshold: 用于识别已知异常的判决阈值
    """
    model.train()
    autoencoder.train()
    #数据加载及处理
    #根据 if_split确定是否将训练数据划分出训练集和验证集，0的话用训练集计算阈值
    if if_split == True and split_error== "raise":
        X_train, X_val, y_train, Y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)  # 如果类别分布不均匀建议加stratify

    elif if_split == True and split_error== "auto":
        try:
            X_train, X_val, y_train, Y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)  # 如果类别分布不均匀建议加stratify

        except ZeroDivisionError: #发生划分错误就执行自动不划分
            X_val = X_train
            Y_val = y_train
        
    else:
        X_val = X_train
        Y_val = y_train

    # # 分离无标签和有标签数据  
    X_unlabelled = X_train[mask==0]        # [6999,10] 标签为0的无标签样本
    X_labelled_anomaly = X_train[mask==1]     # [1,10] 标签为1
    print(f"最终输入的无标签样本数量: {len(X_unlabelled)}, 已知标签样本数量: {len(X_labelled_anomaly)}")

    Y_unlabelled = y_train[mask==0]         # [6999,] 标签为0的无标签样本
    Y_labelled_anomaly = y_train[mask==1]
    
    # 填充缺失值NAN的部分
    X_unlabelled = np.where(np.isnan(X_unlabelled), 0, X_unlabelled)
    X_labelled_anomaly = np.where(np.isnan(X_labelled_anomaly), 0, X_labelled_anomaly)

    return fit_TargAD(X_labelled_anomaly,Y_labelled_anomaly, X_unlabelled, Y_unlabelled,X_val,Y_val,model,    #添加验证数据
                        optimizer,num_centroid,num_anomaly_classes, stage_1_epochs, stage_2_epochs, kmeans_batch,stage_1_batch, stage_2_batch,anomaly_batch,
                         ood_batch, device,input_dim,embedding_dim,loss_oe,loss_re,stage_one_lr,stage_two_lr,weight_decay, verbose)

#将第一阶段训练改为对数据清洗步骤
def autoencoder_pretrain(X_unlabelled,Y_unlabelled, X_labelled_anomaly,autoencoder, input_dim, embedding_dim, num_centroid, stage_1_epochs,kmeans_batch,stage_1_batch, stage_one_lr,filter=0.05, weight_decay=1e-6):
    """
    第一阶段自监督AE预训练
    Args:
        X_unlabelled: 无标签样本
        Y_unlabelled: 无标签样本的标签:0
        Y_unlabelled_category: 无标签样本的聚类标签
        X_labelled_anomaly: 已知异常样本
        input_dim: 特征维度
        embedding_dim: 嵌入特征维度
        num_centroid: 簇的数量
        stage_1_epochs: 第一阶段的训练轮数
        stage_1_batch: 第一阶段的batch大小
        stage_one_lr: 第一阶段的学习率
        weight_decay: 权重衰减
    Returns:
        AE_models: 训练好的自编码器模型列表
    """
    # clustering
    kmean = MiniBatchKMeans(num_centroid, n_init = 42, batch_size = kmeans_batch)  
    kmean.fit(X_unlabelled)
    #为无标签样本生成聚类标签
    Y_unlabelled_category = kmean.predict(X_unlabelled)
    # Y_unlabelled_category = Y_unlabelled_category   #聚类标签
        
    def shuffle_u(X, Y):
        random_index = np.random.permutation(X.shape[0])
        return X[random_index], Y[random_index]
    
    # AE模型列表
    AE_models = []
    print('Starting Stage_One...') 
    # 以下操作均针对一个簇中的数据
    for k in range(num_centroid):
        
        # x_all求的是每个簇中的样本 + labeled anomailes
        x_all = np.vstack((X_unlabelled[Y_unlabelled_category == k], X_labelled_anomaly))
        # 将unlabeled data标记为1
        y_all = np.ones((x_all.shape[0], 1))
        # 将后300个已知异常的标签记为-1
        y_all[-(X_labelled_anomaly.shape[0]):] = -1
        # 进行shuffle打乱顺序
        x_all, y_all = shuffle_u(x_all, y_all)
        autoencoder = AutoEncoder(input_dim=input_dim, num_features=embedding_dim)
        # optimizer = torch.optim.Adam(autoencoder.parameters(),lr=lr)
        optimizer1 = torch.optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=1e-6)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
        
        loss_func = nn.MSELoss()

        stage1_loss = []
        
        # 先对一个簇中的样本进行计算，每个簇运行stage_one_epoch次，然后再对下一个簇进行计算
        for epoch in range(stage_1_epochs):
            # scheduler.step()
            loss_all = 0.0
            # 一个epoch中所含batch的个数，共有32个batch
            stage1_nbBatch = x_all.shape[0] // stage_1_batch    #发现这里有除0报错，因为簇中样本太少导致不够设置的batch
            if stage1_nbBatch == 0:
                stage1_nbBatch += 1     #防止除0
            autoencoder.train()
            for i in range(stage1_nbBatch):
                # 选出每个簇中的一个batch_size样本
                x = x_all[i * stage_1_batch: min((i + 1) * stage_1_batch, x_all.shape[0])]
                x = torch.tensor(x).float()  #除尾部数据shape为 [256, 10]
                
                # 得到表征和重构后的样本
                x_e, x_de = autoencoder(x)
                y = y_all[i * stage_1_batch: min((i + 1) * stage_1_batch, x_all.shape[0])]
                y = torch.tensor(y).float()

                # x.shape = x_de.shape = [256, 1, 10]
                # 256个(1,10),二维数组的行和列
                if x.shape != x_de.shape:
                    x = np.reshape(x.data.cpu().numpy(), x_de.shape)
                    x = torch.tensor(x)

                # 一个样本的每一个维度相减，所以shape = [256, 1, 10]
                # reduction='none' : (x-x_de)^2
                # (1,10)表示x和x_de每个维度之间的均方差误差
                objective_1 = nn.functional.mse_loss(x, x_de, reduction='none')
                # 求和求出样本与样本之间的重构误差，其shape = (256,1)
                objective_1 = torch.sum(objective_1, dim=2)
                # 幂运算，若为1则为本身，若为-1即已知异常则为其倒数
                objective_1 = objective_1 ** y
                # 每个batch的重构误差
                loss = torch.mean(objective_1)
                loss.requires_grad_(True)
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                # loss_all该epoch中整个batch的loss之和，即每一个簇中的重构误差
                # 该loss_all在每个epoch中不断地更新
                loss_all += loss
                
            # 这里的loss_all是整个epoch的loss,该epoch是由32个batch累加得到，最终求得的是batch的平均loss
            stage1_loss.append(loss_all / stage1_nbBatch)
            print('第{}个簇\t Epoch {}/{}\t loss: {:.2f}\t'.format(k,epoch+1,stage_1_epochs,loss_all / stage1_nbBatch))
            
        # 保存的是最后一个epoch训练好的AE
        AE_models.append(autoencoder)

    autoencoders = AE_models
    recon = torch.zeros_like(torch.tensor(Y_unlabelled_category)).float()
    dist = torch.zeros_like(torch.tensor(Y_unlabelled_category)).float()
    # 对每个簇进行操作
    for k in range(num_centroid):
        
        x_unlabelled = torch.tensor(X_unlabelled[Y_unlabelled_category == k]).float()
        x_unlabelled_e, x_unlabelled_de = autoencoders[k](x_unlabelled) #[batch_size, 64],[batch_size, 1, 10]

        # 测试labelled anomaly
        x_labelled = torch.tensor(X_labelled_anomaly).float()
        x_labelled_e, x_labelled_de = autoencoders[k](x_labelled)

        if x_unlabelled.shape != x_unlabelled_de.shape:
            x_unlabelled = np.reshape(x_unlabelled.data.cpu().numpy(), x_unlabelled_de.shape)
            x_unlabelled = torch.tensor(x_unlabelled)
        
        #计算重构损失
        self_reconstruction_loss = nn.functional.mse_loss(x_unlabelled, x_unlabelled_de, reduction='none')
        self_reconstruction_loss = torch.sum(self_reconstruction_loss, dim=2)    #[batch_size, 1]
        # data与重构data的loss
        self_reconstruction_loss = torch.reshape(self_reconstruction_loss, (self_reconstruction_loss.shape[0],))

        # 用于存放每个unlabeled_e与batch_size个labelled_e的最小距离
        tmp = torch.zeros(x_unlabelled_e.shape[0], )

        for i in range(x_unlabelled_e.shape[0]):
            # 计算样本间的欧式距离
            # x_unlabelled_e[i]表示unlabeled data的表征, 每个有64维
            # x_labelled_e表示labeled anomaly的表征大小为(batch_size,64)
            # 这里计算的是unlabeled_e与每一个labelled_e之间的欧式距离            # 对于每一个unlabeled_e都有batch_size个距离
            distance = nn.functional.pairwise_distance(x_unlabelled_e[i], x_labelled_e, p=2)
            # 取出batch_size个距离中最小的距离
            tmp[i] = torch.min(distance)

        # 存储对应聚类标签的loss
        # 针对第一个簇，将loss传入属于该簇的list中
        recon[Y_unlabelled_category == k] = self_reconstruction_loss
        # 存储对应聚类标签的最小距离
        dist[Y_unlabelled_category == k] = tmp
            
    """第一阶段中filtering实现"""
    # 用重构误差作为异常分数, 去掉梯度的形式
    unlabelled_scores = recon.clone().detach().numpy()
    unlabelled_scores = torch.tensor(unlabelled_scores).float()
    # topk返回的是排名前k的值与对应的下标
    # 这里选取异常分数前5%的作为潜在异常
    scores, indexs = unlabelled_scores.topk(int(unlabelled_scores.shape[0] * filter), largest=True)
    score_delete = unlabelled_scores[indexs.detach().numpy()].detach().numpy()   #异常分数
    
    # 潜在的异常
    X_deleted = X_unlabelled[indexs.detach().numpy()]
    Y_deleted = Y_unlabelled[indexs.detach().numpy()]
    # print("潜在异常共有:",Y_deleted.shape[0])
    # print("潜在异常的标签:",Counter(Y_deleted))
    
    # 过滤后的可靠的正常
    reliable_data = list(i for i in range(len(unlabelled_scores)) if i not in indexs)
    X_filter =  X_unlabelled[reliable_data]  #[6586,10]
    Y_filter = Y_unlabelled_category[reliable_data] #[6586,]

    return X_filter, Y_filter, X_deleted, Y_deleted, score_delete

def predict_TargAD(model,best_threshold,x_test,num_centroid,num_anomaly_classes, device,):
    """
    TargAD模型预测函数

    Args:
        model: 训练好的模型
        x_test: 测试数据
        device: 设备

    Returns:
        anomaly_scores: 异常分数
    """
    # model.to(device)
    model.eval()
    with torch.no_grad():
        num_subgroups = num_centroid + num_anomaly_classes
        x_test = torch.tensor(x_test).float().to(device)
        # 获取分类器的预测结果
        _, y_logit = model(x_test)  #y_pred的shape为[3000,8]
           
        # 得到测试集的概率
        prob = torch.max(F.softmax(y_logit, dim = 1)[:,-num_anomaly_classes:], dim=1)[0].cpu().detach().numpy()

        # # 用阈值给测试集的预测结果重新赋值
        # y_pred = []
        # probs = F.softmax(y_logit, dim = 1)[:,-num_anomaly_classes:]
        # logits_anomaly = y_logit[:,-num_anomaly_classes:]
        
        # probs_normal = F.softmax(y_logit, dim = 1)[:,:num_centroid]
        # sum_normal_logits = torch.sum(probs_normal, dim=1).cpu().detach().numpy()
        
        # energy_all = calculate_discrepancy(logits_anomaly[:, -num_anomaly_classes:])
        
        # for i, sum_logit in enumerate(sum_normal_logits):
        #     if sum_logit > num_centroid/num_subgroups:
        #         pred_label = 0   #正常打0
        #     else:
        #         energy = calculate_discrepancy(logits_anomaly[i, -num_anomaly_classes:].unsqueeze(0))
        #         energy = torch.Tensor(energy).squeeze(0)
        #         # eps = 1e-10         
        #         # reciprocal_ent = 1/(ent+eps)
        #         if energy > best_threshold: # 已知异常打上1
        #             pred_label = 1 
        #         else:
        #             pred_label = -2  # 剩余样本的预测标签
        #     y_pred.append(pred_label)
        
        # cm_test = confusion_matrix(y_test_new, y_pred, labels = [0,1,-2])    
     
        # TP_normal = cm_test[0][0]
        # FP_normal = cm_test[1][0] + cm_test[2][0]
        # FN_normal = cm_test[0][1] + cm_test[0][2]
        
        # precision_normal = TP_normal / (TP_normal + FP_normal)
        # recall_normal = TP_normal / (TP_normal + FN_normal)
        # f1_normal = 2*(precision_normal*recall_normal)/(precision_normal+recall_normal)
        
        # TP_anomaly = cm_test[1][1]
        # FP_anomaly = cm_test[0][1] + cm_test[2][1]
        # FN_anomaly = cm_test[1][0] + cm_test[1][2]
        
        # precision_anomaly = TP_anomaly / (TP_anomaly + FP_anomaly)
        # recall_anomaly = TP_anomaly / (TP_anomaly + FN_anomaly)
        # f1_anomaly = 2*(precision_anomaly*recall_anomaly)/(precision_anomaly+recall_anomaly)
        
        # TP_unknown = cm_test[2][2]
        # FP_unknown = cm_test[0][2] + cm_test[1][2]
        # FN_unknown = cm_test[2][0] + cm_test[2][1]
        
        # precision_unknown = TP_unknown / (TP_unknown + FP_unknown)
        # recall_unknown = TP_unknown / (TP_unknown + FN_unknown)
        # f1_unknown = 2*(precision_unknown*recall_unknown)/(precision_unknown+recall_unknown)
        
        # # 宏平均是直接计算各个类别的平均值，不考虑类别的样本数量。
        # macro_avg_precision = (precision_normal + precision_anomaly + precision_unknown) / 3
        # macro_avg_recall = (recall_normal + recall_anomaly + recall_unknown) / 3
        # macro_avg_f1_score = (f1_normal + f1_anomaly + f1_unknown) / 3
        
        
        # # 加权平均是根据每个类别的样本数量进行加权的平均值。
        # normal_all = cm_test[0][0] + cm_test[0][1] + cm_test[0][2]
        # anomaly_all = cm_test[1][0] + cm_test[1][1] + cm_test[1][2]
        # unknown_all = cm_test[2][0] + cm_test[2][1] + cm_test[2][2]
        # sample_all = normal_all + anomaly_all + unknown_all
        
        # weighted_avg_precision = (precision_normal * normal_all + precision_anomaly * anomaly_all
        #                          + precision_unknown * unknown_all)/sample_all
        
        # weighted_avg_recall = (recall_normal * normal_all + recall_anomaly * anomaly_all
        #                          + recall_unknown * unknown_all)/sample_all
        
        # weighted_avg_f1_score = (f1_normal * normal_all + f1_anomaly * anomaly_all
        #                          + f1_unknown * unknown_all)/sample_all

    return prob
