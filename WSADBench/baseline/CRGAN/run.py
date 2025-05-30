# -*- coding: utf-8 -*-
"""
CR-GAN (Contrastive Representation GAN) for Weakly Supervised Anomaly Detection
基于原始CR-GAN实现，专注于CV数据模态
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

from WSADBench.myutils import Utils
from WSADBench.baseline.CRGAN.model import (
    Generator,
    Encoder,
    Discriminatorxz,
    Discriminatorxx,
    Discriminator,
    normal_init,
)
from WSADBench.baseline.CRGAN.fit import fit_crgan
from WSADBench.baseline.CRGAN.utils import compute_anomaly_scores


class CRGAN:
    """
    CR-GAN (Contrastive Representation GAN) for Weakly Supervised Anomaly Detection

    基于原始CR-GAN实现，采用对比学习和生成对抗网络结合的方法
    专注于CV数据，移除表格数据支持以简化代码
    """

    def __init__(
        self,
        seed: int = 42,
        # 模型参数
        latent_dim: int = 100,  # 固定为100，与原始实现一致
        # 训练参数
        epochs: int = 100,
        batch_size: int = 64,
        lr_g: float = 0.0001,
        lr_e: float = 0.0001,
        lr_d: float = 0.000025,
        betas: tuple = (0.5, 0.9),
        # 损失权重 (对应原始实现中的alpha, beta参数)
        alpha: float = 10.0,  # 标记异常数据权重
        beta: float = 10.0,  # 未标记异常数据权重
        gamma: float = 1.0,  # 生成器/编码器权重
        # 异常分数类型
        score_type: str = "reconstruction",
        verbose: bool = True,
    ):
        """
        初始化CR-GAN模型

        Args:
            seed: 随机种子
            latent_dim: 潜在空间维度，固定为100
            epochs: 训练轮数
            batch_size: 批量大小
            lr_g, lr_e, lr_d: 各组件学习率
            betas: Adam优化器参数
            alpha, beta, gamma: 损失权重参数
            score_type: 异常分数类型
            verbose: 是否显示详细信息
        """

        # 基础参数
        self.seed = seed
        self.utils = Utils()
        self.device = self.utils.get_device(True)

        self.latent_dim = latent_dim  # 固定为100

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

        # 异常分数类型
        self.score_type = score_type
        self.verbose = verbose

        # 模型组件
        self.generator = None
        self.encoder = None
        self.discriminatorxz = None
        self.discriminatorxx = None
        self.discriminator = None  # 主判别器
        self.is_fitted = False

    def _initialize_models(self):
        """初始化模型 - 使用固定架构，与原始CR-GAN一致"""

        # 创建模型实例
        self.generator = Generator()
        self.encoder = Encoder()
        self.discriminatorxz = Discriminatorxz()
        self.discriminatorxx = Discriminatorxx()
        self.discriminator = Discriminator()  # 主判别器

        # 权重初始化
        self.generator.apply(normal_init)
        self.encoder.apply(normal_init)
        self.discriminatorxz.apply(normal_init)
        self.discriminatorxx.apply(normal_init)
        self.discriminator.apply(normal_init)

        # 移动到设备
        self.generator = self.generator.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.discriminatorxz = self.discriminatorxz.to(self.device)
        self.discriminatorxx = self.discriminatorxx.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        if self.verbose:
            print("CR-GAN models initialized for CV data")
            print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
            print(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
            print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")

    @classmethod
    def from_config(cls, config: Dict[str, Any], ratio: Optional[float] = None) -> "CRGAN":
        """
        从配置字典创建CRGAN实例

        Args:
            config: 配置字典
            ratio: 标记异常样本比例（未使用，为兼容性保留）

        Returns:
            CRGAN实例
        """
        return cls(**config)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_aux: Optional[np.ndarray] = None) -> "CRGAN":
        """
        训练CR-GAN模型

        Args:
            X_train: 训练数据，形状为(N, C, H, W)
            y_train: 训练标签，0表示正常/未标记，1表示标记异常
            X_aux: 辅助异常数据（可选）

        Returns:
            训练后的模型实例
        """

        # 设置随机种子
        self.utils.set_seed(self.seed)

        if self.verbose:
            print(f"Training CR-GAN with {len(X_train)} samples")
            print(f"Labeled anomalies: {np.sum(y_train == 1)}")
            print(f"Unlabeled samples: {np.sum(y_train == 0)}")

        # 转换为张量
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train = np.array(y_train)

        if X_aux is not None:
            X_aux_tensor = torch.from_numpy(X_aux).float()
        else:
            X_aux_tensor = None

        # 初始化模型
        self._initialize_models()

        # 创建优化器
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=self.betas)
        optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=self.lr_e, betas=self.betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=self.betas)

        # 训练模型
        fit_crgan(
            X_train=X_train_tensor,
            y_train=y_train,
            X_aux=X_aux_tensor,
            generator=self.generator,
            encoder=self.encoder,
            discriminator=self.discriminator,
            optimizer_G=optimizer_G,
            optimizer_E=optimizer_E,
            optimizer_D=optimizer_D,
            epochs=self.epochs,
            batch_size=self.batch_size,
            latent_dim=self.latent_dim,
            device=self.device,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            verbose=self.verbose,
        )

        self.is_fitted = True

        if self.verbose:
            print("CR-GAN training completed")

        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测异常概率

        Args:
            X_test: 测试数据，形状为(N, C, H, W)

        Returns:
            异常分数数组，形状为(N,)
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # 转换为张量
        X_test_tensor = torch.from_numpy(X_test).float()

        # 计算异常分数
        scores = compute_anomaly_scores(
            generator=self.generator,
            encoder=self.encoder,
            X=X_test_tensor,
            device=self.device,
            score_type=self.score_type,
        )

        return scores

    def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        预测异常标签

        Args:
            X_test: 测试数据
            threshold: 分类阈值

        Returns:
            预测标签数组
        """

        scores = self.predict_proba(X_test)
        # 简单的基于分位数的阈值
        threshold_value = np.percentile(scores, threshold * 100)
        return (scores > threshold_value).astype(int)

    def parameter_count(self) -> dict:
        """
        计算模型的参数量

        Returns:
            dict: 包含各组件参数量的字典
        """
        if not self.is_fitted:
            # 如果模型未训练，创建临时模型实例来计算参数
            try:
                # 保存当前状态
                current_generator = self.generator
                current_encoder = self.encoder
                current_discriminatorxz = self.discriminatorxz
                current_discriminatorxx = self.discriminatorxx
                current_discriminator = self.discriminator

                # 临时初始化模型
                self._initialize_models()

                # 计算参数
                params = self._count_parameters()

                # 恢复状态
                self.generator = current_generator
                self.encoder = current_encoder
                self.discriminatorxz = current_discriminatorxz
                self.discriminatorxx = current_discriminatorxx
                self.discriminator = current_discriminator

                return params

            except Exception as e:
                return {"error": f"Failed to count parameters: {str(e)}"}
        else:
            return self._count_parameters()

    def _count_parameters(self) -> dict:
        """内部方法：计算各组件的参数量"""
        params = {}

        if self.generator is not None:
            params["generator"] = sum(p.numel() for p in self.generator.parameters())
        else:
            params["generator"] = 0

        if self.encoder is not None:
            params["encoder"] = sum(p.numel() for p in self.encoder.parameters())
        else:
            params["encoder"] = 0

        if self.discriminatorxz is not None:
            params["discriminatorxz"] = sum(p.numel() for p in self.discriminatorxz.parameters())
        else:
            params["discriminatorxz"] = 0

        if self.discriminatorxx is not None:
            params["discriminatorxx"] = sum(p.numel() for p in self.discriminatorxx.parameters())
        else:
            params["discriminatorxx"] = 0

        if self.discriminator is not None:
            params["discriminator"] = sum(p.numel() for p in self.discriminator.parameters())
        else:
            params["discriminator"] = 0

        params["total"] = (
            params["generator"]
            + params["encoder"]
            + params["discriminatorxz"]
            + params["discriminatorxx"]
            + params["discriminator"]
        )

        return params
