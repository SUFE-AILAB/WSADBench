import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TabularGenerator(nn.Module):
    """表格数据的生成器"""
    def __init__(self, latent_dim=100, input_dim=20, hidden_dims=[128, 64]):
        super(TabularGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        layers = []
        current_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, input_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)


class TabularEncoder(nn.Module):
    """表格数据的编码器"""
    def __init__(self, input_dim=20, latent_dim=100, hidden_dims=[64, 128]):
        super(TabularEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, latent_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class TabularDiscriminator(nn.Module):
    """表格数据的判别器"""
    def __init__(self, input_dim=20, latent_dim=100, hidden_dims=[256, 512]):
        super(TabularDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 图像路径
        self.img_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims[:len(hidden_dims)//2]:
            self.img_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_dim = hidden_dim
        self.img_encoder = nn.Sequential(*self.img_layers)
        
        # 潜在变量路径
        self.z_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 联合判别
        self.joint_layers = nn.Sequential(
            nn.Linear(current_dim + hidden_dims[-1], 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x, z):
        img_feat = self.img_encoder(x)
        z_feat = self.z_encoder(z)
        joint_feat = torch.cat([img_feat, z_feat], dim=1)
        return self.joint_layers(joint_feat)


class CVGenerator(nn.Module):
    """CV数据的生成器 - CIFAR风格"""
    def __init__(self, latent_dim=100, channels=3, img_size=32):
        super(CVGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size
        
        # 计算初始特征图大小
        self.init_size = img_size // 4  # 32 -> 8
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class CVEncoder(nn.Module):
    """CV数据的编码器 - CIFAR风格"""
    def __init__(self, channels=3, img_size=32, latent_dim=100):
        super(CVEncoder, self).__init__()
        
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # 计算展平后的大小
        def conv2d_out_dims(size, kernel_size=3, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        # 计算卷积后的尺寸
        conv_size = img_size
        conv_size = conv2d_out_dims(conv_size)  # 第一个卷积层
        conv_size = conv2d_out_dims(conv_size)  # 第二个卷积层  
        conv_size = conv2d_out_dims(conv_size)  # 第三个卷积层
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * conv_size * conv_size, latent_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CVDiscriminator(nn.Module):
    """CV数据的判别器 - CIFAR风格"""
    def __init__(self, channels=3, img_size=32, latent_dim=100):
        super(CVDiscriminator, self).__init__()
        
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # 图像路径
        def conv2d_out_dims(size, kernel_size=3, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        conv_size = img_size
        conv_size = conv2d_out_dims(conv_size)
        conv_size = conv2d_out_dims(conv_size)
        conv_size = conv2d_out_dims(conv_size)
        
        self.img_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 潜在变量路径
        self.z_encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 联合判别
        self.joint_layers = nn.Sequential(
            nn.Linear(128 * conv_size * conv_size + 512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    
    def forward(self, img, z):
        img_feat = self.img_conv(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        z_feat = self.z_encoder(z)
        joint_feat = torch.cat([img_feat, z_feat], dim=1)
        return self.joint_layers(joint_feat)


class AABiGANModel:
    """AABiGAN模型的工厂类"""
    
    @staticmethod
    def create_models(modal='tabular', **kwargs):
        """
        根据模态创建对应的模型
        
        Args:
            modal: 'tabular' 或 'cv'
            **kwargs: 模型参数
        
        Returns:
            generator, encoder, discriminator
        """
        
        if modal == 'tabular':
            input_dim = kwargs.get('input_dim', 20)
            latent_dim = kwargs.get('latent_dim', 100)
            hidden_dims = kwargs.get('hidden_dims', [128, 64])
            
            generator = TabularGenerator(latent_dim, input_dim, hidden_dims)
            encoder = TabularEncoder(input_dim, latent_dim, hidden_dims[::-1])
            discriminator = TabularDiscriminator(input_dim, latent_dim, [256, 512])
            
        elif modal == 'cv':
            channels = kwargs.get('channels', 3)
            img_size = kwargs.get('img_size', 32)
            latent_dim = kwargs.get('latent_dim', 100)
            
            generator = CVGenerator(latent_dim, channels, img_size)
            encoder = CVEncoder(channels, img_size, latent_dim)
            discriminator = CVDiscriminator(channels, img_size, latent_dim)
            
        else:
            raise ValueError(f"Unsupported modal: {modal}. Choose from ['tabular', 'cv']")
        
        return generator, encoder, discriminator


# 辅助类用于简化模型初始化
class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def get_models(modal, **model_params):
        """获取指定模态的模型"""
        return AABiGANModel.create_models(modal=modal, **model_params)
