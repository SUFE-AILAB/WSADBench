import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union

from WSADBench.myutils import Utils
from WSADBench.baseline.AABiGAN.model import ModelFactory
from WSADBench.baseline.AABiGAN.fit import fit_aabigan, compute_anomaly_scores, create_auxiliary_data


class AABiGAN:
    """
    AA-BiGAN (Adversarial Autoencoder BiGAN) for Weakly Supervised Anomaly Detection
    
    支持表格数据和CV数据两种模态
    """
    
    def __init__(self, 
                 seed: int = 42,
                 modal: str = 'tabular',
                 # 模型参数
                 latent_dim: int = 100,
                 # 训练参数
                 epochs: int = 100,
                 batch_size: int = 64,
                 lr_g: float = 0.0002,
                 lr_e: float = 0.0002,
                 lr_d: float = 0.0001,
                 betas: tuple = (0.5, 0.999),
                 # 损失权重
                 alpha: float = 1.0,
                 beta: float = 10.0,
                 gamma: float = 1.0,
                 # 辅助数据参数
                 aux_ratio: float = 0.2,
                 aux_strategy: str = 'duplicate',
                 # 异常分数类型
                 score_type: str = 'reconstruction',
                 # 模型架构参数 (tabular)
                 hidden_dims: list = None,
                 # 模型架构参数 (cv)
                 channels: int = 3,
                 img_size: int = 32,
                 # 其他参数
                 device: str = 'auto',
                 verbose: bool = True):
        """
        初始化AABiGAN模型
        
        Args:
            seed: 随机种子
            modal: 数据模态 ('tabular' 或 'cv')
            latent_dim: 潜在空间维度
            epochs: 训练轮数
            batch_size: 批量大小
            lr_g, lr_e, lr_d: 生成器、编码器、判别器学习率
            betas: Adam优化器beta参数
            alpha, beta, gamma: 损失函数权重
            aux_ratio: 辅助数据比例
            aux_strategy: 辅助数据生成策略
            score_type: 异常分数类型 ('reconstruction', 'latent', 'combined')
            hidden_dims: 表格数据隐藏层维度
            channels: 图像通道数
            img_size: 图像尺寸
            device: 计算设备
            verbose: 是否显示训练信息
        """
        
        self.seed = seed
        self.modal = modal
        self.utils = Utils()
        
        # 设备配置
        if device == 'auto':
            self.device = self.utils.get_device()
        else:
            self.device = torch.device(device)
        
        # 模型参数
        self.latent_dim = latent_dim
        
        # 训练参数
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_e = lr_e
        self.lr_d = lr_d
        self.betas = betas
        
        # 损失权重
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 辅助数据参数
        self.aux_ratio = aux_ratio
        self.aux_strategy = aux_strategy
        
        # 异常分数类型
        self.score_type = score_type
        
        # 模型架构参数
        if modal == 'tabular':
            self.hidden_dims = hidden_dims or [128, 64]
            self.model_params = {
                'latent_dim': latent_dim,
                'hidden_dims': self.hidden_dims
            }
        elif modal == 'cv':
            self.channels = channels
            self.img_size = img_size
            self.model_params = {
                'latent_dim': latent_dim,
                'channels': channels,
                'img_size': img_size
            }
        else:
            raise ValueError(f"Unsupported modal: {modal}. Choose from ['tabular', 'cv']")
        
        self.verbose = verbose
        
        # 模型组件
        self.generator = None
        self.encoder = None
        self.discriminator = None
        self.is_fitted = False
    
    def _initialize_models(self, input_dim: int):
        """初始化模型"""
        
        # 更新模型参数
        if self.modal == 'tabular':
            self.model_params['input_dim'] = input_dim
        
        # 创建模型
        self.generator, self.encoder, self.discriminator = ModelFactory.get_models(
            modal=self.modal, **self.model_params
        )
        
        # 移动到设备
        self.generator = self.generator.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # 创建优化器
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=self.betas
        )
        self.optimizer_E = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr_e, betas=self.betas
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=self.betas
        )
        
        if self.verbose:
            print(f"AABiGAN models initialized for {self.modal} data")
            print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
            print(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
            print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def fit(self, X_train: Union[np.ndarray, torch.Tensor], 
            y_train: np.ndarray, 
            ratio: Optional[float] = None) -> 'AABiGAN':
        """
        训练AABiGAN模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签 (0: unlabeled/normal, 1: labeled anomaly)
            ratio: 未使用，保持接口一致性
        
        Returns:
            self
        """
        
        # 设置随机种子
        self.utils.set_seed(self.seed)
        
        # 数据预处理
        if isinstance(X_train, np.ndarray):
            X_train_tensor = torch.from_numpy(X_train).float()
        else:
            X_train_tensor = X_train.float()
        
        X_train_tensor = X_train_tensor.to(self.device)
        y_train = np.array(y_train)
        
        # 获取输入维度
        if self.modal == 'tabular':
            input_dim = X_train_tensor.shape[1]
        elif self.modal == 'cv':
            input_dim = X_train_tensor.shape[1:]  # (C, H, W)
        
        # 初始化模型
        self._initialize_models(input_dim if self.modal == 'tabular' else None)
        
        # 分离数据
        labeled_anomaly_mask = y_train == 1
        X_labeled_anomaly = X_train_tensor[labeled_anomaly_mask]
        X_unlabeled = X_train_tensor[y_train == 0]
        
        if self.verbose:
            print(f"Training data: {len(X_train_tensor)} samples")
            print(f"Labeled anomalies: {len(X_labeled_anomaly)} samples")
            print(f"Unlabeled data: {len(X_unlabeled)} samples")
        
        # 创建辅助异常数据
        X_aux = None
        if len(X_labeled_anomaly) > 0 and self.aux_ratio > 0:
            X_aux = create_auxiliary_data(
                X_unlabeled, X_labeled_anomaly, 
                self.aux_ratio, self.aux_strategy
            )
            if X_aux is not None:
                X_aux = X_aux.to(self.device)
                if self.verbose:
                    print(f"Auxiliary anomaly data: {len(X_aux)} samples")
        
        # 训练模型
        fit_aabigan(
            X_train=X_train_tensor,
            y_train=y_train,
            X_aux=X_aux,
            generator=self.generator,
            encoder=self.encoder,
            discriminator=self.discriminator,
            optimizer_G=self.optimizer_G,
            optimizer_E=self.optimizer_E,
            optimizer_D=self.optimizer_D,
            epochs=self.epochs,
            batch_size=self.batch_size,
            latent_dim=self.latent_dim,
            device=self.device,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            modal=self.modal,
            verbose=self.verbose
        )
        
        self.is_fitted = True
        
        if self.verbose:
            print("AABiGAN training completed!")
        
        return self
    
    def predict_proba(self, X_test: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        预测异常概率
        
        Args:
            X_test: 测试数据
        
        Returns:
            anomaly_scores: 异常分数
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        # 数据预处理
        if isinstance(X_test, np.ndarray):
            X_test_tensor = torch.from_numpy(X_test).float()
        else:
            X_test_tensor = X_test.float()
        
        X_test_tensor = X_test_tensor.to(self.device)
        
        # 计算异常分数
        anomaly_scores = compute_anomaly_scores(
            X_test_tensor, self.encoder, self.generator, self.modal, self.score_type
        )
        
        return anomaly_scores
    
    def predict(self, X_test: Union[np.ndarray, torch.Tensor], 
                threshold: Optional[float] = None) -> np.ndarray:
        """
        预测异常标签
        
        Args:
            X_test: 测试数据
            threshold: 异常阈值，如果为None则使用中位数
        
        Returns:
            predictions: 预测标签 (0: normal, 1: anomaly)
        """
        
        scores = self.predict_proba(X_test)
        
        if threshold is None:
            threshold = np.median(scores)
        
        predictions = (scores > threshold).astype(int)
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        
        params = {
            'seed': self.seed,
            'modal': self.modal,
            'latent_dim': self.latent_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr_g': self.lr_g,
            'lr_e': self.lr_e,
            'lr_d': self.lr_d,
            'betas': self.betas,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'aux_ratio': self.aux_ratio,
            'aux_strategy': self.aux_strategy,
            'score_type': self.score_type,
        }
        
        if self.modal == 'tabular':
            params['hidden_dims'] = self.hidden_dims
        elif self.modal == 'cv':
            params['channels'] = self.channels
            params['img_size'] = self.img_size
        
        return params
    
    def set_params(self, **params) -> 'AABiGAN':
        """设置模型参数"""
        
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        
        # 如果模型已经初始化，需要重新初始化
        if self.is_fitted:
            self.is_fitted = False
            self.generator = None
            self.encoder = None
            self.discriminator = None
        
        return self
