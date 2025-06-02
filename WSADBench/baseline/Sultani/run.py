# -*- coding: utf-8 -*-
"""
Sultani (Real-world Anomaly Detection) for Weakly Supervised Video Anomaly Detection
基于"Real-world Anomaly Detection in Surveillance Videos"论文实现
Multiple Instance Learning (MIL) 弱监督异常检测方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from WSADBench.myutils import Utils
from WSADBench.baseline.Sultani.model import SultaniLearner
from WSADBench.baseline.Sultani.fit import fit_sultani_main


class Sultani:
    """
    Sultani方法实现
    基于MIL的弱监督视频异常检测

    论文: "Real-world Anomaly Detection in Surveillance Videos"
    """

    def __init__(
        self,
        seed: int = 42,
        # 模型参数
        input_dim: int = 2048,
        dropout: float = 0.6,
        # 训练参数
        epochs: int = 100,
        batch_size: int = 1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        # 损失权重
        sparsity_weight: float = 0.00008,
        smoothness_weight: float = 0.00008,
        # 其他参数
        segments_per_video: int = 32,
        use_scheduler: bool = True,
        scheduler_milestones: List[int] = None,
        verbose: bool = True,
    ):
        """
        初始化Sultani模型

        Args:
            seed: 随机种子
            input_dim: 输入特征维度（默认2048对应ResNet特征）
            dropout: Dropout概率
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            sparsity_weight: 稀疏性损失权重
            smoothness_weight: 平滑性损失权重
            segments_per_video: 每个视频的段数
            use_scheduler: 是否使用学习率调度器
            scheduler_milestones: 学习率调度里程碑
            verbose: 是否打印详细信息
        """

        self.seed = seed
        self.input_dim = input_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
        self.segments_per_video = segments_per_video
        self.use_scheduler = use_scheduler
        self.scheduler_milestones = scheduler_milestones or [25, 50]
        self.verbose = verbose

        # 内部状态
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.fitted = False
        self.training_history = None

        # 工具类
        self.utils = Utils()

        # 设置随机种子
        self.utils.set_seed(seed)
        self.device = self.utils.get_device(True)

    def _init_model(self):
        """初始化模型"""
        if self.model is None:
            # 创建模型
            self.model = SultaniLearner(input_dim=self.input_dim, drop_p=self.dropout).to(self.device)

            # 创建优化器
            self.optimizer = optim.Adagrad(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

            # 创建学习率调度器
            if self.use_scheduler:
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_milestones)

            if self.verbose:
                print(f"Sultani模型初始化完成")
                print(f"设备: {self.device}")
                print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def fit(self, X, y, X_test=None, y_test=None):
        """
        训练Sultani模型

        Args:
            X: 训练特征 [n_samples, feature_dim]
            y: 训练标签 [n_samples] (0: normal, 1: anomaly)
            X_test: 测试特征（可选）
            y_test: 测试标签（可选）

        Returns:
            self
        """
        start_time = time.time()

        if self.verbose:
            print("=" * 60)
            print("开始训练Sultani模型")
            print("=" * 60)
            print(f"训练样本数: {len(X)}")
            print(f"正常样本: {np.sum(y == 0)}, 异常样本: {np.sum(y == 1)}")

        # 数据预处理
        X = self._preprocess_data(X)

        # 初始化模型
        self._init_model()
        # 训练模型
        try:
            self.training_history = fit_sultani_main(
                X_train=X,
                y_train=y,
                model=self.model,
                optimizer=self.optimizer,
                epochs=self.epochs,
                batch_size=self.batch_size,
                device=self.device,
                sparsity_weight=self.sparsity_weight,
                smoothness_weight=self.smoothness_weight,
                verbose=self.verbose,
            )

            self.fitted = True

        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            # 如果MIL训练失败，尝试简化训练
            if self.verbose:
                print("尝试使用简化训练方法...")
            self._fit_simplified(X, y)

        training_time = time.time() - start_time

        if self.verbose:
            print(f"训练完成，耗时: {training_time:.2f}秒")

            # 如果有测试数据，计算测试性能
            if X_test is not None and y_test is not None:
                test_scores = self.predict_proba(X_test)
                if len(np.unique(y_test)) > 1:
                    test_auc = roc_auc_score(y_test, test_scores)
                    test_ap = average_precision_score(y_test, test_scores)
                    print(f"测试集 AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f}")

        return self

    def _fit_simplified(self, X, y):
        """简化训练方法"""
        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()

        for epoch in range(min(self.epochs, 50)):  # 限制epoch数
            self.optimizer.zero_grad()

            # 简单前向传播
            outputs = self.model(X_tensor)

            # 简化损失：二分类交叉熵
            loss = nn.BCELoss()(outputs.squeeze(), y_tensor)

            loss.backward()
            self.optimizer.step()

            if self.verbose and epoch % 10 == 0:
                print(f"简化训练 Epoch {epoch+1}, Loss: {loss.item():.6f}")

        self.fitted = True
        if self.verbose:
            print("简化训练完成")

    def predict_proba(self, X):
        """
        预测异常概率

        Args:
            X: 输入特征 [n_samples, feature_dim]

        Returns:
            异常概率分数 [n_samples]
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        self.model.eval()

        X = self._preprocess_data(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            scores = outputs.squeeze().cpu().numpy()

        # 确保返回一维数组
        if scores.ndim == 0:
            scores = np.array([scores])

        return scores

    def predict(self, X, threshold=0.5):
        """
        预测异常标签

        Args:
            X: 输入特征
            threshold: 分类阈值

        Returns:
            预测标签 [n_samples]
        """
        scores = self.predict_proba(X)
        return (scores > threshold).astype(int)

    def _preprocess_data(self, X):
        """
        数据预处理

        Args:
            X: 输入数据

        Returns:
            预处理后的数据
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        X = np.array(X, dtype=np.float32)

        # 确保是2D数组
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X

    def get_params(self, deep=True):
        """
        获取模型参数（sklearn兼容）

        Args:
            deep: 是否深度获取参数

        Returns:
            参数字典
        """
        return {
            "seed": self.seed,
            "input_dim": self.input_dim,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "sparsity_weight": self.sparsity_weight,
            "smoothness_weight": self.smoothness_weight,
            "segments_per_video": self.segments_per_video,
            "use_scheduler": self.use_scheduler,
            "scheduler_milestones": self.scheduler_milestones,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """
        设置模型参数（sklearn兼容）

        Args:
            **params: 参数字典

        Returns:
            self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 如果模型已经初始化，需要重新初始化
        if self.model is not None:
            self.model = None
            self.fitted = False

        return self

    def parameter_count(self):
        """
        计算Sultani模型的参数数量

        Returns:
            dict: 包含各个组件参数数量的字典
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                # 计算已初始化模型的参数
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )
                non_trainable_params = total_params - trainable_params

                return {
                    "sultani_total": total_params,
                    "sultani_trainable": trainable_params,
                    "sultani_non_trainable": non_trainable_params,
                    "total": total_params,
                }
            else:
                # 如果模型还没有初始化，创建临时模型来计算参数
                from WSADBench.baseline.Sultani.model import SultaniLearner

                temp_model = SultaniLearner(input_dim=self.input_dim, drop_p=self.dropout)
                total_params = sum(p.numel() for p in temp_model.parameters())
                trainable_params = sum(
                    p.numel() for p in temp_model.parameters() if p.requires_grad
                )
                non_trainable_params = total_params - trainable_params

                return {
                    "sultani_total": total_params,
                    "sultani_trainable": trainable_params,
                    "sultani_non_trainable": non_trainable_params,
                    "total": total_params,
                    "note": f"Parameters counted from temporary model (input_dim={self.input_dim})",
                }
        except Exception as e:
            return {"error": f"Failed to count parameters: {str(e)}", "total": 0}
