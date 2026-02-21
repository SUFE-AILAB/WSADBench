import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle


def fit_aabigan(
        X_train,
        y_train,
        mask,
        X_aux,
        generator,
        encoder,
        discriminator,
        optimizer_G,
        optimizer_E,
        optimizer_D,
        epochs,
        batch_size,
        latent_dim,
        device,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        modal="tabular",
        verbose=True,
):
    """
    训练AABiGAN模型

    Args:
        X_train: 主要训练数据 (正常+少量异常)
        y_train: 训练标签 (0: unlabeled/normal, 1: labeled anomaly)
        X_aux: 辅助异常数据
        mask: 区分有无标签样本的掩码
        generator: 生成器模型
        encoder: 编码器模型
        discriminator: 判别器模型
        optimizer_G: 生成器优化器
        optimizer_E: 编码器优化器
        optimizer_D: 判别器优化器
        epochs: 训练轮数
        batch_size: 批量大小
        latent_dim: 潜在空间维度
        device: 设备
        alpha, beta, gamma: 损失权重
        modal: 数据模态
        verbose: 是否打印训练信息
    """

    generator.train()
    encoder.train()
    discriminator.train()

    # # 分离标记的异常样本和未标记样本
    # labeled_anomaly_mask = y_train == 1
    # unlabeled_mask = y_train == 0

    X_labeled_anomaly = X_train[mask == 1]  # 有标签的样本，根据设定可含有正常样本
    X_unlabeled = X_train[mask == 0]

    # 创建数据加载器
    if len(X_labeled_anomaly) > 0:
        labeled_dataset = TensorDataset(X_labeled_anomaly)
        labeled_loader = DataLoader(
            labeled_dataset, batch_size=min(batch_size // 4, len(X_labeled_anomaly)), shuffle=True, drop_last=True
        )
    else:
        labeled_loader = None

    num_unlabel = X_unlabeled.size(0)
    # 如果样本数小于 batch_size，则简单重复补足
    if num_unlabel < batch_size:
        repeat_times = (batch_size + num_unlabel - 1) // num_unlabel  # 向上取整
        X_unlabeled = X_unlabeled.repeat((repeat_times, *([1] * (X_unlabeled.dim() - 1))))[:batch_size]

    unlabeled_dataset = TensorDataset(X_unlabeled)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if X_aux is not None and len(X_aux) > 0:
        aux_dataset = TensorDataset(X_aux)
        aux_loader = DataLoader(aux_dataset, batch_size=min(batch_size // 2, len(X_aux)), shuffle=True, drop_last=True)
    else:
        aux_loader = None

    # 损失函数
    adversarial_loss = nn.MSELoss()
    reconstruction_loss = nn.MSELoss()

    # 训练循环
    for epoch in range(epochs):

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_e_loss = 0.0
        n_batches = 0

        # 创建循环迭代器
        unlabeled_iter = iter(unlabeled_loader)
        if labeled_loader is not None:
            labeled_iter = cycle(labeled_loader)
        if aux_loader is not None:
            aux_iter = cycle(aux_loader)

        for batch_idx in range(len(unlabeled_loader)):

            # 获取未标记数据
            try:
                unlabeled_batch = next(unlabeled_iter)[0].to(device)
            except StopIteration:
                break

            current_batch_size = unlabeled_batch.size(0)

            # 获取标记异常数据
            if labeled_loader is not None:
                labeled_batch = next(labeled_iter)[0].to(device)
                if labeled_batch.size(0) > current_batch_size:
                    labeled_batch = labeled_batch[:current_batch_size]
            else:
                labeled_batch = None

            # 获取辅助异常数据
            if aux_loader is not None:
                aux_batch = next(aux_iter)[0].to(device)
                if aux_batch.size(0) > current_batch_size:
                    aux_batch = aux_batch[:current_batch_size]
            else:
                aux_batch = None

            # 生成随机潜在向量
            z_real = torch.randn(current_batch_size, latent_dim).to(device)
            z_fake = torch.randn(current_batch_size, latent_dim).to(device)

            # 真实和虚假标签
            valid = torch.ones(current_batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(current_batch_size, 1, requires_grad=False).to(device)

            # ---------------------
            #  训练判别器
            # ---------------------

            optimizer_D.zero_grad()

            # 真实数据的判别损失 (x, E(x))
            z_encoded = encoder(unlabeled_batch)
            d_real = discriminator(unlabeled_batch, z_encoded)
            d_real_loss = adversarial_loss(d_real, valid)

            # 生成数据的判别损失 (G(z), z)
            x_generated = generator(z_fake)
            d_fake = discriminator(x_generated.detach(), z_fake)
            d_fake_loss = adversarial_loss(d_fake, fake)

            # 总判别器损失
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  训练生成器和编码器
            # ---------------------

            optimizer_G.zero_grad()
            optimizer_E.zero_grad()

            # BiGAN对抗损失
            z_encoded = encoder(unlabeled_batch)
            d_real = discriminator(unlabeled_batch, z_encoded)
            g_loss_adv_real = adversarial_loss(d_real, fake)  # 编码器希望被判别为假

            x_generated = generator(z_fake)
            d_fake = discriminator(x_generated, z_fake)
            g_loss_adv_fake = adversarial_loss(d_fake, valid)  # 生成器希望被判别为真

            # 重建损失
            x_reconstructed = generator(z_encoded)
            g_loss_recon = reconstruction_loss(x_reconstructed, unlabeled_batch)

            # 编码损失（可选）
            z_reconstructed = encoder(x_generated)
            e_loss_recon = reconstruction_loss(z_reconstructed, z_fake)

            # 异常检测损失
            anomaly_loss = 0.0
            if labeled_batch is not None:
                # 标记异常样本的编码应该与正常样本不同
                #修改
                combine_batch = torch.cat([labeled_batch, unlabeled_batch[: labeled_batch.size(0)]], dim=0)
                z_combined = encoder(combine_batch)
                z_anomaly = z_combined[: labeled_batch.size(0)]
                z_normal = z_combined[labeled_batch.size(0):] 

                #注释源码
                # z_anomaly = encoder(labeled_batch)
                # z_normal = encoder(unlabeled_batch[: labeled_batch.size(0)])
                # 结束

                # 最大化异常和正常样本在潜在空间的距离
                anomaly_loss = -torch.mean(torch.norm(z_anomaly - z_normal, dim=1))
            # print(f'anomaly_loss:{anomaly_loss}')
            # 辅助损失
            aux_loss = 0.0
            if aux_batch is not None:
                z_aux = encoder(aux_batch)
                # 辅助数据应该在潜在空间中聚集
                aux_loss = torch.mean(torch.norm(z_aux, dim=1))

            # 总损失
            g_loss = (
                    alpha * (g_loss_adv_real + g_loss_adv_fake)
                    + beta * g_loss_recon
                    + gamma * (e_loss_recon + anomaly_loss + aux_loss)
            )

            g_loss.backward()
            optimizer_G.step()
            optimizer_E.step()

            # 统计
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1

        # 打印训练信息
        if verbose and (epoch + 1) % 10 == 0:
            avg_d_loss = epoch_d_loss / n_batches
            avg_g_loss = epoch_g_loss / n_batches
            print(f"Epoch [{epoch + 1}/{epochs}] " f"D Loss: {avg_d_loss:.4f} " f"G Loss: {avg_g_loss:.4f}")


def compute_anomaly_scores(X, encoder, generator, modal="tabular", score_type="reconstruction"):
    """
    计算异常分数 - 基于原始AABiGAN实现

    Args:
        X: 输入数据
        encoder: 编码器模型
        generator: 生成器模型
        modal: 数据模态
        score_type: 分数类型 ('reconstruction', 'latent')

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

        # 计算重建误差(AABiGAN论文中使用范数，但源码中使用平方和，这里按源码实现)
        if modal == "cv":
            # 对于图像数据，使用像素级平方和
            recon_errors = torch.sum((X - X_recon) ** 2, dim=(1, 2, 3))
        else:
            # 对于表格数据，使用特征级平方和
            recon_errors = torch.sum((X - X_recon) ** 2, dim=1)

        # 可以结合潜在空间的距离作为额外的异常指标
        z_norm = torch.sum(z ** 2, dim=1)

        # 根据分数类型返回不同的异常分数
        if score_type == "reconstruction":
            # 原始实现使用 -1*rec_all_val，但这里直接返回正值
            # 因为WSADBench期望越高越异常
            anomaly_scores = recon_errors
        elif score_type == "latent":
            # 原始实现使用 -1*z_score_val，但这里直接返回正值
            anomaly_scores = z_norm
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

    return anomaly_scores.cpu().numpy()


def create_auxiliary_data(X_unlabeled, X_labeled_anomaly, aux_ratio=0.2, strategy="duplicate"):
    """
    创建辅助异常数据

    Args:
        X_unlabeled: 未标记数据
        X_labeled_anomaly: 标记的异常数据
        aux_ratio: 辅助数据比例
        strategy: 创建策略 ('duplicate', 'noise')

    Returns:
        X_aux: 辅助异常数据
    """

    if len(X_labeled_anomaly) == 0:
        return None

    n_aux = int(len(X_unlabeled) * aux_ratio)

    if strategy == "duplicate":
        # 复制已有的异常样本
        indices = np.random.choice(len(X_labeled_anomaly), n_aux, replace=True)
        X_aux = X_labeled_anomaly[indices]

    elif strategy == "noise":
        # 在异常样本基础上添加噪声
        base_indices = np.random.choice(len(X_labeled_anomaly), n_aux, replace=True)
        X_base = X_labeled_anomaly[base_indices]

        # 添加高斯噪声
        noise_scale = 0.1 * torch.std(X_base)
        noise = torch.randn_like(X_base) * noise_scale
        X_aux = X_base + noise
        X_aux = torch.clamp(X_aux, 0, 1)  # 确保在合理范围内

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return X_aux
