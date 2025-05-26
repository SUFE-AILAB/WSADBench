import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def weights_init(m):
    """
    自定义权重初始化函数
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def compute_gradient_penalty(discriminator, real_samples, fake_samples, z_real, z_fake, device):
    """
    计算梯度惩罚 (WGAN-GP)
    
    Args:
        discriminator: 判别器模型
        real_samples: 真实样本
        fake_samples: 生成样本
        z_real: 真实潜在向量
        z_fake: 生成潜在向量
        device: 设备
    
    Returns:
        gradient_penalty: 梯度惩罚
    """
    
    # 随机插值
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    
    # 调整alpha维度以匹配数据维度
    if len(real_samples.shape) == 4:  # 图像数据 (N, C, H, W)
        alpha = alpha.view(-1, 1, 1, 1)
    elif len(real_samples.shape) == 2:  # 表格数据 (N, D)
        alpha = alpha.view(-1, 1)
    
    # 插值样本
    interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated_samples.requires_grad_(True)
    
    # 插值潜在向量
    alpha_z = torch.rand(z_real.size(0), 1).to(device)
    interpolated_z = alpha_z * z_real + (1 - alpha_z) * z_fake
    interpolated_z.requires_grad_(True)
    
    # 判别器输出
    d_interpolated = discriminator(interpolated_samples, interpolated_z)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=[interpolated_samples, interpolated_z],
        grad_outputs=torch.ones(d_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    
    # 合并梯度
    gradient_samples = gradients[0].view(gradients[0].size(0), -1)
    gradient_z = gradients[1]
    gradient_combined = torch.cat([gradient_samples, gradient_z], dim=1)
    
    # 计算梯度惩罚
    gradient_penalty = ((gradient_combined.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def visualize_latent_space(encoder, X_data, y_data, save_path=None, method='tsne'):
    """
    可视化潜在空间
    
    Args:
        encoder: 编码器模型
        X_data: 输入数据
        y_data: 标签
        save_path: 保存路径
        method: 降维方法 ('tsne' 或 'pca')
    """
    
    encoder.eval()
    
    with torch.no_grad():
        if isinstance(X_data, np.ndarray):
            X_tensor = torch.from_numpy(X_data).float()
        else:
            X_tensor = X_data
        
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            encoder = encoder.cuda()
        
        # 编码到潜在空间
        z_encoded = encoder(X_tensor).cpu().numpy()
    
    # 降维到2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    z_2d = reducer.fit_transform(z_encoded)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    # 分别绘制正常和异常点
    normal_mask = y_data == 0
    anomaly_mask = y_data == 1
    
    plt.scatter(z_2d[normal_mask, 0], z_2d[normal_mask, 1], 
                c='blue', alpha=0.6, label='Normal', s=20)
    plt.scatter(z_2d[anomaly_mask, 0], z_2d[anomaly_mask, 1], 
                c='red', alpha=0.8, label='Anomaly', s=20)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Latent Space Visualization ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space visualization saved to {save_path}")
    
    plt.show()


def visualize_reconstructions(generator, encoder, X_samples, save_path=None, n_samples=8):
    """
    可视化重建结果 (适用于图像数据)
    
    Args:
        generator: 生成器模型
        encoder: 编码器模型  
        X_samples: 输入样本
        save_path: 保存路径
        n_samples: 显示样本数量
    """
    
    generator.eval()
    encoder.eval()
    
    with torch.no_grad():
        if isinstance(X_samples, np.ndarray):
            X_tensor = torch.from_numpy(X_samples).float()
        else:
            X_tensor = X_samples
        
        # 选择样本
        if len(X_tensor) > n_samples:
            indices = torch.randperm(len(X_tensor))[:n_samples]
            X_tensor = X_tensor[indices]
        
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            generator = generator.cuda()
            encoder = encoder.cuda()
        
        # 编码和重建
        z_encoded = encoder(X_tensor)
        X_reconstructed = generator(z_encoded)
        
        # 移动到CPU
        X_original = X_tensor.cpu()
        X_recon = X_reconstructed.cpu()
    
    # 可视化
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # 原始图像
        if X_original.shape[1] == 1:  # 灰度图像
            img_orig = X_original[i, 0]
            img_recon = X_recon[i, 0]
            cmap = 'gray'
        else:  # 彩色图像
            img_orig = X_original[i].permute(1, 2, 0)
            img_recon = X_recon[i].permute(1, 2, 0)
            cmap = None
        
        axes[0, i].imshow(img_orig, cmap=cmap)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(img_recon, cmap=cmap)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction visualization saved to {save_path}")
    
    plt.show()


def compute_reconstruction_error_distribution(generator, encoder, X_normal, X_anomaly):
    """
    计算重建误差分布
    
    Args:
        generator: 生成器模型
        encoder: 编码器模型
        X_normal: 正常样本
        X_anomaly: 异常样本
    
    Returns:
        normal_errors: 正常样本重建误差
        anomaly_errors: 异常样本重建误差
    """
    
    generator.eval()
    encoder.eval()
    
    def compute_errors(X):
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.from_numpy(X).float()
            else:
                X_tensor = X
            
            if torch.cuda.is_available():
                X_tensor = X_tensor.cuda()
            
            z = encoder(X_tensor)
            X_recon = generator(z)
            
            # 计算重建误差
            if len(X_tensor.shape) == 4:  # 图像数据
                errors = torch.mean((X_tensor - X_recon) ** 2, dim=(1, 2, 3))
            else:  # 表格数据
                errors = torch.mean((X_tensor - X_recon) ** 2, dim=1)
            
            return errors.cpu().numpy()
    
    normal_errors = compute_errors(X_normal)
    anomaly_errors = compute_errors(X_anomaly)
    
    return normal_errors, anomaly_errors


def plot_loss_curves(train_losses_d, train_losses_g, save_path=None):
    """
    绘制损失曲线
    
    Args:
        train_losses_d: 判别器损失
        train_losses_g: 生成器损失
        save_path: 保存路径
    """
    
    plt.figure(figsize=(12, 4))
    
    # 判别器损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_d, label='Discriminator Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 生成器损失
    plt.subplot(1, 2, 2)
    plt.plot(train_losses_g, label='Generator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to {save_path}")
    
    plt.show()


def compute_model_complexity(model):
    """
    计算模型复杂度
    
    Args:
        model: PyTorch模型
    
    Returns:
        params: 参数数量
        flops: 估计的FLOPs (仅针对线性层的粗略估计)
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 简单的FLOPs估计 (仅针对线性层)
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += module.in_features * module.out_features
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'estimated_flops': flops
    }


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
