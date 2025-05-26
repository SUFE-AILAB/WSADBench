# -*- coding: utf-8 -*-
"""
CR-GAN训练逻辑
基于原始CR-GAN实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
from typing import Optional


def fit_crgan(X_train, y_train, X_aux, generator, encoder, discriminator,
              optimizer_G, optimizer_E, optimizer_D, epochs, batch_size,
              latent_dim, device, alpha=10.0, beta=10.0, gamma=1.0,
              modal='cv', verbose=True):
    """
    训练CR-GAN模型
    
    Args:
        X_train: 训练数据
        y_train: 训练标签 (0: unlabeled/normal, 1: labeled anomaly)
        X_aux: 辅助异常数据
        generator: 生成器模型
        encoder: 编码器模型
        discriminator: 判别器模型
        optimizer_G, optimizer_E, optimizer_D: 优化器
        epochs: 训练轮数
        batch_size: 批量大小
        latent_dim: 潜在空间维度
        device: 计算设备
        alpha, beta, gamma: 损失权重参数
        modal: 数据模态
        verbose: 是否显示训练信息
    """
    
    # 设置模型为训练模式
    generator.train()
    encoder.train()
    discriminator.train()
    
    # 损失函数
    adversarial_loss = nn.MSELoss()
    
    # 分离数据
    labeled_anomaly_mask = y_train == 1
    unlabeled_mask = y_train == 0
    
    X_labeled_anomaly = X_train[labeled_anomaly_mask]
    X_unlabeled = X_train[unlabeled_mask]
    
    # 检查是否有标记的异常样本
    if len(X_labeled_anomaly) == 0:
        raise ValueError("CR-GAN requires at least one labeled anomaly sample for training. "
                        "Please ensure that y_train contains at least one sample with label 1.")
    
    if len(X_unlabeled) == 0:
        raise ValueError("CR-GAN requires at least one unlabeled sample for training. "
                        "Please ensure that y_train contains at least one sample with label 0.")
    
    if verbose:
        print(f"Found {len(X_labeled_anomaly)} labeled anomaly samples and {len(X_unlabeled)} unlabeled samples.")
    
    # 创建数据加载器
    unlabeled_dataset = TensorDataset(X_unlabeled)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    
    labeled_dataset = TensorDataset(X_labeled_anomaly)
    labeled_loader = DataLoader(labeled_dataset, batch_size=min(batch_size, len(X_labeled_anomaly)), shuffle=True)
    
    aux_loader = None
    if X_aux is not None and len(X_aux) > 0:
        aux_dataset = TensorDataset(X_aux)
        aux_loader = DataLoader(aux_dataset, batch_size=min(batch_size, len(X_aux)), shuffle=True)
    
    # 训练参数
    a, b, c = 1.0, 0.0, 0.75  # 原始CR-GAN中的常数
    alpha_a = 1.0  # 辅助数据权重
    th = 0.8  # 标签翻转阈值
    prob_pz = 1.0  # z空间判别概率
    prob_px = 1.0  # x空间判别概率
    
    # 学习率调度器
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.98)
    scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=100, gamma=0.98)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.98)
    
    if verbose:
        print("Starting CR-GAN training...")
    
    for epoch in range(epochs):
        epoch_d_loss_xz = []
        epoch_d_loss_xx = []
        epoch_g_loss = []
        epoch_e_loss = []
        
        # 创建循环迭代器
        labeled_cycle = cycle(labeled_loader) if len(X_labeled_anomaly) > 0 else None
        aux_cycle = cycle(aux_loader) if aux_loader is not None else None
        
        for batch_idx, (unlabeled_batch,) in enumerate(unlabeled_loader):
            unlabeled_batch = unlabeled_batch.to(device)
            batch_size_actual = unlabeled_batch.size(0)
            
            # 获取标记异常数据
            labeled_batch = None
            if labeled_cycle is not None:
                labeled_batch, = next(labeled_cycle)
                labeled_batch = labeled_batch.to(device)
                # 调整大小以匹配unlabeled批次
                if labeled_batch.size(0) > batch_size_actual:
                    labeled_batch = labeled_batch[:batch_size_actual]
                elif labeled_batch.size(0) < batch_size_actual:
                    # 重复采样
                    repeat_times = (batch_size_actual + labeled_batch.size(0) - 1) // labeled_batch.size(0)
                    labeled_batch = labeled_batch.repeat(repeat_times, *([1] * (len(labeled_batch.shape) - 1)))
                    labeled_batch = labeled_batch[:batch_size_actual]
            
            # 获取辅助数据
            aux_batch = None
            if aux_cycle is not None:
                aux_batch, = next(aux_cycle)
                aux_batch = aux_batch.to(device)
                if aux_batch.size(0) > batch_size_actual:
                    aux_batch = aux_batch[:batch_size_actual]
                elif aux_batch.size(0) < batch_size_actual:
                    repeat_times = (batch_size_actual + aux_batch.size(0) - 1) // aux_batch.size(0)
                    aux_batch = aux_batch.repeat(repeat_times, *([1] * (len(aux_batch.shape) - 1)))
                    aux_batch = aux_batch[:batch_size_actual]
            
            # 标签
            valid = torch.ones(batch_size_actual, 1, device=device)
            fake = torch.zeros(batch_size_actual, 1, device=device)
            
            # ================================================================
            # 训练判别器
            # ================================================================
            optimizer_D.zero_grad()
            
            # 生成随机噪声
            z_fake = torch.randn(batch_size_actual, latent_dim, device=device)
            
            # 编码真实数据
            z_real = encoder(unlabeled_batch)
            
            # 生成假数据
            x_fake = generator(z_fake)
            x_recon = generator(z_real)
            
            # XZ判别器损失
            d_loss_xz = 0.0
            
            # 真实(x, z)对
            d_real_xz = adversarial_loss(discriminator(unlabeled_batch, z_real, 'xz')[0], a * valid)
            
            # 假(x, z)对
            d_fake_xz = adversarial_loss(discriminator(x_fake, z_fake, 'xz')[0], b * valid)
            
            d_loss_xz = d_real_xz + d_fake_xz
            
            # 添加标记异常数据的判别
            if labeled_batch is not None:
                z_labeled = encoder(labeled_batch)
                alpha_valid = alpha_a * torch.ones(labeled_batch.size(0), 1, device=device)
                
                res_labeled = discriminator(labeled_batch, z_labeled, 'xz')[0]
                
                # 动态标签翻转
                prob = np.random.uniform(0, 1)
                if prob <= prob_pz:
                    d_labeled_xz = alpha * adversarial_loss(res_labeled, alpha_valid)
                else:
                    d_labeled_xz = alpha * adversarial_loss(res_labeled, b * alpha_valid)
                
                d_loss_xz += d_labeled_xz
                
                # 更新prob_pz
                lambda_z = torch.mean(res_labeled).item()
                if lambda_z > th and prob_pz > 0.05:
                    prob_pz -= 0.05
                    prob_pz = max(prob_pz, 0.0)
                elif lambda_z <= th and prob_pz < 1.0:
                    prob_pz += 0.05
                    prob_pz = min(prob_pz, 1.0)
            
            # XX判别器损失
            d_loss_xx = 0.0
            
            # 真实(x, x)对
            d_real_xx = adversarial_loss(discriminator(unlabeled_batch, unlabeled_batch, 'xx')[0], a * valid)
            
            # 假(x, x')对
            d_fake_xx = adversarial_loss(discriminator(unlabeled_batch, x_recon, 'xx')[0], b * valid)
            
            d_loss_xx = d_real_xx + d_fake_xx
            
            # 添加标记异常数据的判别
            if labeled_batch is not None:
                res_labeled_xx = discriminator(labeled_batch, labeled_batch, 'xx')[0]
                
                prob = np.random.uniform(0, 1)
                if prob <= prob_px:
                    d_labeled_xx = alpha * adversarial_loss(res_labeled_xx, alpha_valid)
                else:
                    d_labeled_xx = alpha * adversarial_loss(res_labeled_xx, b * alpha_valid)
                
                d_loss_xx += d_labeled_xx
                
                # 更新prob_px
                lambda_x = torch.mean(res_labeled_xx).item()
                if lambda_x > th and prob_px > 0.05:
                    prob_px -= 0.05
                    prob_px = max(prob_px, 0.0)
                elif lambda_x <= th and prob_px < 1.0:
                    prob_px += 0.05
                    prob_px = min(prob_px, 1.0)
            
            # 总判别器损失
            d_loss = d_loss_xz + d_loss_xx
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
            
            # ================================================================
            # 训练生成器
            # ================================================================
            optimizer_G.zero_grad()
            
            # 生成器希望生成的数据被判别为真
            g_loss_fake = adversarial_loss(discriminator(x_fake, z_fake, 'xz')[0], c * valid)
            
            # 循环一致性损失
            cycle_loss = (
                adversarial_loss(discriminator(unlabeled_batch, unlabeled_batch, 'xx')[0], c * valid) +
                adversarial_loss(discriminator(unlabeled_batch, x_recon, 'xx')[0], c * valid)
            )
            
            if labeled_batch is not None:
                cycle_loss += adversarial_loss(discriminator(labeled_batch, labeled_batch, 'xx')[0], c * alpha_valid)
            
            g_loss = g_loss_fake + 0.25 * cycle_loss
            g_loss.backward(retain_graph=True)
            optimizer_G.step()
            
            # ================================================================
            # 训练编码器
            # ================================================================
            optimizer_E.zero_grad()
            
            # 编码器希望编码结果被判别为真
            e_loss_real = adversarial_loss(discriminator(unlabeled_batch, z_real, 'xz')[0], c * valid)
            
            e_loss = e_loss_real
            
            if labeled_batch is not None:
                e_loss_labeled = adversarial_loss(discriminator(labeled_batch, z_labeled, 'xz')[0], c * alpha_valid)
                e_loss += e_loss_labeled
            
            e_loss.backward()
            optimizer_E.step()
            
            # 记录损失
            epoch_d_loss_xz.append(d_loss_xz.item())
            epoch_d_loss_xx.append(d_loss_xx.item())
            epoch_g_loss.append(g_loss.item())
            epoch_e_loss.append(e_loss.item())
        
        # 更新学习率
        scheduler_G.step()
        scheduler_E.step()
        scheduler_D.step()
        
        # 计算重建误差用于监控
        generator.eval()
        encoder.eval()
        with torch.no_grad():
            sample_batch = X_unlabeled[:min(100, len(X_unlabeled))].to(device)
            z_sample = encoder(sample_batch)
            x_recon_sample = generator(z_sample)
            if modal == 'cv':
                recon_error = torch.mean(torch.sum((x_recon_sample - sample_batch) ** 2, dim=(1, 2, 3)))
            else:
                recon_error = torch.mean(torch.sum((x_recon_sample - sample_batch) ** 2, dim=1))
        generator.train()
        encoder.train()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  D_loss_xz: {np.mean(epoch_d_loss_xz):.4f}")
            print(f"  D_loss_xx: {np.mean(epoch_d_loss_xx):.4f}")
            print(f"  G_loss: {np.mean(epoch_g_loss):.4f}")
            print(f"  E_loss: {np.mean(epoch_e_loss):.4f}")
            print(f"  Recon_error: {recon_error.item():.4f}")
            print(f"  prob_pz: {prob_pz:.3f}, prob_px: {prob_px:.3f}")
        
        # 早停条件（可选）
        if epoch > 300 and (np.mean(epoch_d_loss_xz) < 0.015 or np.mean(epoch_d_loss_xx) < 0.015):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break


def compute_anomaly_scores(X, encoder, generator, modal='cv', score_type='rank'):
    """
    计算异常分数 - 基于原始CR-GAN实现
    
    Args:
        X: 输入数据
        encoder: 编码器模型
        generator: 生成器模型
        modal: 数据模态
        score_type: 分数类型 ('reconstruction', 'latent', 'rank')
    
    Returns:
        anomaly_scores: 异常分数 (越高越异常)
    """
    encoder.eval()
    generator.eval()
    
    with torch.no_grad():
        # 编码到潜在空间
        z = encoder(X)
        # 重建
        X_recon = generator(z)
        
        # 重建误差
        if modal == 'cv':
            recon_errors = torch.sum((X_recon - X) ** 2, dim=(1, 2, 3))
        else:
            recon_errors = torch.sum((X_recon - X) ** 2, dim=1)
        
        # 潜在空间分数（L2范数平方）
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
            
            # CR-GAN的排序分数组合：recon + 16 * latent
            rank_scores = recon_norm + 16 * z_norm
            
            return rank_scores
        else:
            raise ValueError(f"Unknown score_type: {score_type}")
    
    return anomaly_scores.cpu().numpy()


def create_auxiliary_data(X_unlabeled, X_labeled_anomaly, aux_ratio=0.2, strategy='duplicate'):
    """
    创建辅助异常数据
    
    Args:
        X_unlabeled: 无标签数据
        X_labeled_anomaly: 已标记异常数据
        aux_ratio: 辅助数据比例
        strategy: 生成策略
    
    Returns:
        X_aux: 辅助异常数据
    """
    if len(X_labeled_anomaly) == 0 or aux_ratio <= 0:
        return None
    
    n_aux = int(len(X_unlabeled) * aux_ratio)
    
    if strategy == 'duplicate':
        # 简单重复采样
        if n_aux <= len(X_labeled_anomaly):
            indices = torch.randperm(len(X_labeled_anomaly))[:n_aux]
            X_aux = X_labeled_anomaly[indices]
        else:
            # 需要重复采样
            repeat_times = (n_aux + len(X_labeled_anomaly) - 1) // len(X_labeled_anomaly)
            X_repeated = X_labeled_anomaly.repeat(repeat_times, *([1] * (len(X_labeled_anomaly.shape) - 1)))
            indices = torch.randperm(len(X_repeated))[:n_aux]
            X_aux = X_repeated[indices]
    
    elif strategy == 'noise':
        # 添加噪声变体
        X_aux = []
        for _ in range(n_aux):
            # 随机选择一个异常样本
            idx = torch.randint(0, len(X_labeled_anomaly), (1,))
            sample = X_labeled_anomaly[idx].clone()
            # 添加高斯噪声
            noise = torch.randn_like(sample) * 0.1
            X_aux.append(sample + noise)
        X_aux = torch.stack(X_aux).squeeze(1)
    
    else:
        raise ValueError(f"Unknown auxiliary data strategy: {strategy}")
    
    return X_aux
