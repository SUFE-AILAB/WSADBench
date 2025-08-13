# -*- coding: utf-8 -*-
from tracemalloc import start
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import time
from tqdm import tqdm
from WSADBench.myutils import Utils
from WSADBench.baseline.TargAD.model import AutoEncoder, Classifier
from WSADBench.baseline.TargAD.fit import fit_TargAD_main, fit_TargAD, predict_TargAD, shuffle_u
# #主函数
class TargAD:
    """
    TargAD方法实现
    基于自编码器和对比学习的弱监督视频异常检测

    论文: "A Robust Prioritized Anomaly Detection when Not
    All Anomalies are of Primary Interest"
    """

    def __init__(self,
                 seed=42,
                 num_centroid=5,
                 num_anomaly_classes=3,
                 stage_1_epochs=30,
                 stage_2_epochs=30,
                 kmeans_batch=256,
                 stage_1_batch=256,
                 stage_2_batch=128,
                 anomaly_batch=32,
                 ood_batch=32,
                 stage_one_lr=0.0001,
                 stage_two_lr=0.00001,
                 input_dim=None,
                 embedding_dim=64,
                 loss_oe=0.1,
                 loss_re=1,
                 if_split=True,
                 split_error="raise",
                 verbose=True, ):
        """
        TargAD模型参数配置类。

        Args:
        - seed: 随机种子，默认为42。
        - num_centroid: 聚类簇数，默认为5。
        - num_anomaly_classes: 异常类别数，默认为3。
        - stage_1_epochs: 第一阶段训练轮数，默认为30。
        - stage_2_epochs: 第二阶段训练轮数，默认为30。
        - kmeans_batch: KMeans聚类批次大小，默认为256。
        - stage_1_batch: 第一阶段批次大小，默认为256。
        - stage_2_batch: 第二阶段批次大小，默认为128。
        - anomaly_batch: 异常样本批次大小，默认为32。
        - ood_batch: OOD（Out of Distribution）样本批次大小，默认为32。
        - stage_one_lr: 第一阶段学习率，默认为0.0001。
        - stage_two_lr: 第二阶段学习率，默认为0.00001。
        - embedding_dim: 嵌入维度，默认为64。
        - input_dim: 特征维度，默认为196。
        - loss_oe: 异常检测损失系数，默认为0.1。
        - loss_re: 正则化损失系数，默认为1。
        - verbose: 是否打印日志，默认为True。
        """
        
        #工具类
        self.utils = Utils()
        #设备设置
        self.device = self.utils.get_device(True)
        #随机种子设置
        self.seed = seed
        self.utils.set_seed(seed)

        self.num_centroid = num_centroid
        self.num_anomaly_classes = num_anomaly_classes
        self.num_subgroups = self.num_centroid + self.num_anomaly_classes
        self.stage_1_epochs = stage_1_epochs
        self.stage_2_epochs = stage_2_epochs
        self.kmeans_batch = kmeans_batch
        self.stage_1_batch = stage_1_batch
        self.stage_2_batch = stage_2_batch
        self.anomaly_batch = anomaly_batch
        self.ood_batch = ood_batch
        self.stage_one_lr = stage_one_lr
        self.stage_two_lr = stage_two_lr
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.loss_oe = loss_oe
        self.loss_re = loss_re
        self.if_split = if_split
        self.split_error=split_error
        self.verbose = verbose

        #模型内部状态
        self.model = None
        self.autoencoder = None
        self.optimizer = None
        self.scheduler = None

    def _init_model(self):
        """初始化模型"""
        #第一阶段训练所需模型
        self.autoencoder = AutoEncoder(input_dim=self.input_dim, num_features=self.embedding_dim).to(self.device)  
        #第二阶段训练所需模型
        self.model = Classifier(input_dim=self.input_dim, num_features=self.embedding_dim, num_classes=self.num_subgroups).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.stage_two_lr, weight_decay=1e-6
        )


        if self.verbose:
            print(f"TargAD模型初始化完成")
            print(f"设备: {self.device}")
            print(f"model参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def fit(self,X,y,mask):
        """
        训练模型
        Args
            X:训练数据特征
            y:训练数据标签
            X_test:测试数据特征
            y_test:测试数据标签
        return:
            self: 返回当前模型实例
        """
        start_time = time.time()
        if self.verbose:
            print("=" * 60)
            print("开始训练TargAD模型")
            print("=" * 60)
            print(f"训练样本数: {len(X)}")
            print(f"无标签样本: {np.sum(y == 0)}, 异常样本: {np.sum(y == 1)}")
        self._init_model()
        
        #训练模型
        self.training_history, self.model = fit_TargAD_main(
            X_train=X,
            y_train=y,
            mask=mask,
            model=self.model,
            autoencoder=self.autoencoder,
            optimizer=self.optimizer,
            num_centroid=self.num_centroid,
            num_anomaly_classes=self.num_anomaly_classes,
            stage_1_epochs=self.stage_1_epochs,
            stage_2_epochs=self.stage_2_epochs,
            kmeans_batch=self.kmeans_batch,
            stage_1_batch=self.stage_1_batch,
            stage_2_batch=self.stage_2_batch,
            anomaly_batch=self.anomaly_batch,
            ood_batch=self.ood_batch,
            device=self.device,
            input_dim=self.input_dim,
            embedding_dim = self.embedding_dim,
            loss_oe=self.loss_oe,
            loss_re=self.loss_re,
            stage_one_lr=self.stage_one_lr,
            stage_two_lr=self.stage_two_lr,
            weight_decay=1e-6,
            if_split=self.if_split,
            split_error=self.split_error,
            verbose=self.verbose
        )
        self.fitted = True

        training_time = time.time() - start_time

        if self.verbose:
            print(f"训练完成，耗时: {training_time:.2f}秒")


        return self
    
    def predict_score(self, X_test):
        """
        Predict anomaly scores

        Args:
            X_test: test feature data

        Returns:
            scores: anomaly score array
        """
        if self.model is None:
            raise ValueError("Model is not trained yet, please call fit method first")

        # Prediction
        scores = predict_TargAD(self.model, X_test,self.num_centroid,self.num_anomaly_classes, self.device)

        return scores

    def parameter_count(self) -> dict:
        """
        计算模型的参数量

        Returns:
            dict: 包含模型参数量的字典
        """
        if self.model is None:
            # 如果模型未初始化，创建临时模型实例来计算参数
            temp_input_dim = 100  # 默认输入维度用于估算

            try:
                # 保存当前状态
                current_model = self.model
                current_optimizer = self.optimizer
                current_scheduler = self.scheduler

                # 临时初始化模型
                self._init_model(temp_input_dim)

                # 计算参数
                params = {"model": sum(p.numel() for p in self.model.parameters())}
                params["total"] = params["model"]

                # 恢复状态
                self.model = current_model
                self.optimizer = current_optimizer
                self.scheduler = current_scheduler

                return params

            except Exception as e:
                return {"error": f"Failed to count parameters: {str(e)}"}
        else:
            params = {"model": sum(p.numel() for p in self.model.parameters())}
            params["total"] = params["model"]
            return params


