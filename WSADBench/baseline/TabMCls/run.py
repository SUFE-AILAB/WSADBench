# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from tabm import TabM


class TabMCls:
    def __init__(self,seed,
                 model_name="TabMCls", n_num_features=None, cat_cardinalities=None, num_embeddings=None,
                 d_out=2, k=32, batch_size=256, device='cuda', learning_rate=0.002,
                 weight_decay=0.0003, arch_type='tabm',epochs = 100
                 ):
        """
        TabM 分类器

        Args:
            n_num_features: 数值特征的数量
            cat_cardinalities: 分类特征的基数列表
            num_embeddings: 数值特征的嵌入模块
            d_out: 输出维度，对于二分类设为2
            k: 集成模型的数量
            batch_size: 批处理大小
            device: 设备 ('cuda' 或 'cpu')
            learning_rate: 学习率
            weight_decay: 权重衰减
            arch_type: 架构类型 ('tabm', 'tabm-mini', 'tabm-packed')
        """
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities if cat_cardinalities else []
        self.num_embeddings = num_embeddings
        self.d_out = d_out
        self.k = k
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.arch_type = arch_type
        self.epochs = epochs
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train, y_train,  verbose=True):
        """
        训练 TabM 分类器

        Args:
            X_train: 训练特征，可以是 (X_num, X_cat) 元组或单一的数值特征数组
            y_train: 训练标签
            epochs: 训练轮数
            validation_data: 验证数据 (X_val, y_val)
            verbose: 是否打印训练过程
        """

        # 准备数据
        if isinstance(X_train, tuple):
            X_num, X_cat = X_train
        else:
            X_num, X_cat = X_train, None

        # 转换为 PyTorch 张量
        if not isinstance(X_num, torch.Tensor):
            X_num = torch.FloatTensor(X_num)
        if X_cat is not None and not isinstance(X_cat, torch.Tensor):
            X_cat = torch.LongTensor(X_cat)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.LongTensor(y_train)

        # 移动到设备
        X_num = X_num.to(self.device)
        if X_cat is not None:
            X_cat = X_cat.to(self.device)
        y_train = y_train.to(self.device)

        # 初始化模型
        if self.model is None:
            self._init_model(X_num.shape[1], X_cat)

        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            # 批量训练
            for i in range(0, len(X_num), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_num))

                # 准备批次数据
                X_num_batch = X_num[i:batch_end]
                if X_cat is not None:
                    X_cat_batch = X_cat[i:batch_end]
                else:
                    X_cat_batch = None
                y_batch = y_train[i:batch_end]

                # 前向传播
                self.optimizer.zero_grad()

                if X_cat_batch is not None:
                    outputs = self.model(X_num_batch, X_cat_batch)
                else:
                    outputs = self.model(X_num_batch)

                # 输出形状: (batch_size, k, d_out)
                # 计算每个集成模型的损失并取平均
                loss = 0
                for j in range(self.k):
                    loss += self.criterion(outputs[:, j, :], y_batch)
                loss = loss / self.k

                # 反向传播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict_score(self, X): 
        """
        预测正类的概率分数

        Args:
            X: 输入特征，可以是 (X_num, X_cat) 元组或单一的数值特征数组

        Returns:
            numpy数组: 正类的概率分数
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        if isinstance(X, tuple):
            X_num, X_cat = X
        else:
            X_num, X_cat = X, None

        # 转换为 PyTorch 张量
        if not isinstance(X_num, torch.Tensor):
            X_num = torch.FloatTensor(X_num)
        if X_cat is not None and not isinstance(X_cat, torch.Tensor):
            X_cat = torch.LongTensor(X_cat)

        # 移动到设备
        X_num = X_num.to(self.device)
        if X_cat is not None:
            X_cat = X_cat.to(self.device)

        scores = []

        with torch.no_grad():
            for i in range(0, len(X_num), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_num))

                X_num_batch = X_num[i:batch_end]
                if X_cat is not None:
                    X_cat_batch = X_cat[i:batch_end]
                else:
                    X_cat_batch = None

                # 前向传播
                if X_cat_batch is not None:
                    outputs = self.model(X_num_batch, X_cat_batch)
                else:
                    outputs = self.model(X_num_batch)

                # 输出形状: (batch_size, k, d_out)
                # 对集成模型的预测进行平均，然后取softmax
                avg_outputs = outputs.mean(dim=1)  # (batch_size, d_out)
                probabilities = torch.softmax(avg_outputs, dim=1)

                # 取正类的概率（假设二分类，索引1为正类）
                batch_scores = probabilities[:, 1].cpu().numpy()
                scores.append(batch_scores)

        return np.concatenate(scores, axis=0)

    def _init_model(self, n_num_features, X_cat=None):
        """初始化 TabM 模型和优化器"""
        if self.cat_cardinalities is None and X_cat is not None:
            # 如果没有提供分类特征基数，从数据中推断
            self.cat_cardinalities = [int(X_cat[:, i].max() + 1) for i in range(X_cat.shape[1])]

        # 创建 TabM 模型
        self.model = TabM.make(
            n_num_features=n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            num_embeddings=self.num_embeddings,
            d_out=self.d_out,
            k=self.k,
            arch_type=self.arch_type
        ).to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
