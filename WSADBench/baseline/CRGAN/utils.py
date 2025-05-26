# -*- coding: utf-8 -*-
"""
CR-GAN实用工具函数 - 简化版本
基于原始CR-GAN实现
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def test_eva(generator, encoder, discriminator, epoch, val_loader, test_loader, device, opt):
    """
    评估函数 - 基于原始CR-GAN的testing1.py实现
    """
    normal_cate = {opt.normal_digit}
    
    generator.eval()
    encoder.eval()
    discriminator.eval()
    
    target_all_val = []
    rec_all_val = []
    z_score_val = []
   
    target_all_test = []
    rec_all_test = []
    z_score_test = []
    
    with torch.no_grad():
        # 验证集评估
        for idx, (image, target) in enumerate(val_loader):
            image = image.to(device)
            target = target.to(device)
            target_all_val.append(target.detach().cpu().numpy())
           
            # 重构误差
            score1 = torch.sum((generator(encoder(image)) - image) ** 2, dim=(1, 2, 3))
            rec_all_val.append(score1.detach().cpu().numpy())
            
            # 潜在空间距离
            score4 = torch.sum(encoder(image) ** 2, dim=1)
            z_score_val.append(score4.detach().cpu().numpy())
            
        # 测试集评估
        for idx, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            target_all_test.append(target.detach().cpu().numpy())
           
            # 重构误差
            score1 = torch.sum((generator(encoder(image)) - image) ** 2, dim=(1, 2, 3))
            rec_all_test.append(score1.detach().cpu().numpy())
            
            # 潜在空间距离
            score4 = torch.sum(encoder(image) ** 2, dim=1)
            z_score_test.append(score4.detach().cpu().numpy())
            
    # 处理验证集结果
    target_all_val = np.concatenate(target_all_val, axis=0)
    rec_all_val = np.concatenate(rec_all_val, axis=0)
    z_score_val = np.concatenate(z_score_val, axis=0)
    
    n_rec_all_val = (rec_all_val - np.min(rec_all_val)) / (np.max(rec_all_val) - np.min(rec_all_val))
    n_z_score_val = (z_score_val - np.min(z_score_val)) / (np.max(z_score_val) - np.min(z_score_val))
    rank_score_val = n_rec_all_val + 16 * n_z_score_val
    
    # 处理测试集结果
    target_all_test = np.concatenate(target_all_test, axis=0)
    rec_all_test = np.concatenate(rec_all_test, axis=0)
    z_score_test = np.concatenate(z_score_test, axis=0)
    
    n_rec_all_test = (rec_all_test - np.min(rec_all_val)) / (np.max(rec_all_val) - np.min(rec_all_val))
    n_z_score_test = (z_score_test - np.min(z_score_val)) / (np.max(z_score_val) - np.min(z_score_val))
    rank_score_test = n_rec_all_test + 16 * n_z_score_test
    
    # 计算AUC
    try:
        # 将正常类标记为0，异常类标记为1
        val_labels = np.where(np.isin(target_all_val, list(normal_cate)), 0, 1)
        test_labels = np.where(np.isin(target_all_test, list(normal_cate)), 0, 1)
        
        val_recon = roc_auc_score(val_labels, rec_all_val)
        test_recon = roc_auc_score(test_labels, rec_all_test)
        val_zs = roc_auc_score(val_labels, z_score_val)
        test_zs = roc_auc_score(test_labels, z_score_test)
        val_rank = roc_auc_score(val_labels, rank_score_val)
        test_rank = roc_auc_score(test_labels, rank_score_test)
        
    except ValueError:
        val_recon = test_recon = val_zs = test_zs = val_rank = test_rank = 0.0
    
    eva_dic = {
        "epoch": epoch,
        "val_recon": val_recon,
        "test_recon": test_recon,
        "val_zs": val_zs,
        "test_zs": test_zs,
        "val_rank": val_rank,
        "test_rank": test_rank
    }
    
    return eva_dic


def compute_anomaly_scores(generator, encoder, X, device, score_type='reconstruction'):
    """
    计算异常分数
    
    Args:
        generator: 生成器模型
        encoder: 编码器模型
        X: 输入数据张量
        device: 计算设备
        score_type: 分数类型 ('reconstruction', 'latent', 'rank')
    
    Returns:
        异常分数数组
    """
    generator.eval()
    encoder.eval()
    
    X = X.to(device)
    
    with torch.no_grad():
        # 编码到潜在空间
        z = encoder(X)
        # 重建
        X_recon = generator(z)
        
        # 重建误差
        recon_errors = torch.sum((X_recon - X) ** 2, dim=(1, 2, 3))
        
        # 潜在空间分数
        z_scores = torch.sum(z ** 2, dim=1)
        
        # 根据分数类型返回不同的异常分数
        if score_type == 'reconstruction':
            anomaly_scores = recon_errors
        elif score_type == 'latent':
            anomaly_scores = z_scores
        elif score_type == 'rank':
            # CR-GAN特有的排序分数组合
            # 标准化分数
            recon_errors_np = recon_errors.cpu().numpy()
            z_scores_np = z_scores.cpu().numpy()
            
            # 归一化到[0, 1]
            recon_norm = (recon_errors_np - np.min(recon_errors_np)) / (np.max(recon_errors_np) - np.min(recon_errors_np) + 1e-8)
            z_norm = (z_scores_np - np.min(z_scores_np)) / (np.max(z_scores_np) - np.min(z_scores_np) + 1e-8)
            # 归一化逻辑与原始代码略微不同，因为原始代码将测试集又划分成了验证集和测试集，使用验证集计算归一化参数，
            # 但验证集对于异常检测任务很可能是不可信的，所以这里实现时使用测试集的分数进行归一化。
            # TODO: 是不是应该记录训练集的分数范围用于归一化?
            
            # CR-GAN的排序分数组合：权重 4 来源于论文中说明的数值
            rank_scores = recon_norm + 4 * z_norm
            
            return rank_scores
        else:
            raise ValueError(f"Unknown score_type: {score_type}")
    
    return anomaly_scores.cpu().numpy()


def normal_init(m, mean=0.0, std=0.02):
    """权重初始化"""
    if isinstance(m, (torch.nn.ConvTranspose2d, torch.nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
