import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from WSADBench.myutils import Utils
from WSADBench.baseline.ARNet.model import model_generater
from WSADBench.baseline.ARNet.model import Filter_Module,CAS_Module,BaS_Net
from WSADBench.baseline.ARNet.fit import fit_ARNet_main

class ARNet:

    """ ARNet方法实现
        基于MIL的弱监督视频异常检测  
        论文："Weakly_Supervised_Video_Anomaly_Detection_via_Center-Guided_Discriminative_Learning"
    """

    def __init__(
            self,
            seed:int=42,
            #模型参数
            input_dim:int=2048,
            dropout: float = 0.7,
            model_name: str =None,
            n_feature: int=2048,
            feature_size:int=2048,
            # 训练参数
            epochs: int = 30,
            batch_size: int = 60,
            lr: float = 0.0001,                   #学习率
            weight_decay: float = 0.0010000000474974513,
            #损失参数
            DMIL_weight: float = 1.000,
            Center_weight: float = 20.000,
            #其他参数
            segments_per_video: int = 32,
            use_scheduler: bool = True,
            scheduler_milestones: List[int] = None,
            verbose: bool = True, 
    ):
        """
        初始化ARNet模型

        Args:
            seed: 随机种子
            input_dim: 输入特征维度（默认2048对应ResNet特征） n_feature / feature_size / len_feature 可能一致 待debug
            model_name: 调用的模型,model_genarater的参数
            dropout: Dropout概率
            epochs: 训练轮数
            batch_size: 批量大小
            lr: 学习率
            weight_decay: 权重衰减
            DMIL_weight: DMIL损失权重
            Center_weight: 中心损失权重 
            segments_per_video: 每个视频的段数
            use_scheduler: 是否使用学习率调度器
            scheduler_milestones: 学习率调度里程碑
            verbose: 是否打印详细信息
        """
        self.seed = seed
        self.input_dim = input_dim
        self.dropout = dropout
        self.model_name = model_name
        self.n_feature = n_feature
        self.feature_size = feature_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.DMIL_weight = DMIL_weight
        self.Center_weight = Center_weight
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
            

            # if self.model_name == 'model_lstm':    #这个模型比较特殊
            #     # 创建序列长度张量（假设每个视频都是完整的32个片段）
            #     seq_len = torch.full((2,), 32, dtype=torch.int32).to(self.device)
            #     self.model = model_generater(model_name=self.model_name,feature_size=self.feature_size,seq_len=seq_len).to(self.device)
            # else:
            self.model = model_generater(model_name=self.model_name,feature_size=self.feature_size).to(self.device)
            

            # 创建优化器
            self.optimizer = optim.Adagrad(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            # 创建学习率调度器
            if self.use_scheduler:
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_milestones)

            if self.verbose:
                print(f"ARNet模型初始化完成")
                print(f"设备: {self.device}")
                print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def fit(self, X, y,X_test=None, y_test=None):
        """
        训练ARNet模型

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
            print("开始训练ARNet模型")
            print("=" * 60)
            print(f"训练样本数: {len(X)}")
            print(f"正常样本: {np.sum(y == 0)}, 异常样本: {np.sum(y == 1)}")

        # 数据预处理
        X = self._preprocess_data(X)   #[507180,2048]  # 可能需要根据实际数据调整维度

        # 初始化模型
        self._init_model()
        # 训练模型
        self.training_history = fit_ARNet_main(
            X_train=X,
            y_train=y,
            model=self.model,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            DMIL_weight=self.DMIL_weight,
            Center_weight=self.Center_weight,
            verbose=self.verbose,
        )

        self.fitted = True

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

    def predict_proba(self, X):           #插眼，留意
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
        X_tensor = torch.FloatTensor(X).to(self.device)     #[696270,2048]
        # X_tensor = X_tensor.reshape(X.shape[0], 1, X.shape[1])
        # print(f"X_tensor_shape:{X_tensor.shape}")

        with torch.no_grad():
            _, y_pred = self.model(X_tensor)
            scores = y_pred.squeeze().cpu().numpy()
            # scores = y_pred.squeeze()
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
            "model_name":self.model_name,
            "n_feature":self.n_feature,
            "feature_size":self.feature_size,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "DMIL_weight": self.DMIL_weight,
            "Center_weight": self.Center_weight,
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
        计算ARNet模型的参数数量

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
                    "ARNet_total": total_params,
                    "ARNet_trainable": trainable_params,
                    "ARNet_non_trainable": non_trainable_params,
                    "total": total_params,
                }
            else:
                # 如果模型还没有初始化，创建临时模型来计算参数
                from WSADBench.baseline.ARNet.model import ARNetLearner

                temp_model = ARNetLearner(input_dim=self.input_dim, drop_p=self.dropout)
                total_params = sum(p.numel() for p in temp_model.parameters())
                trainable_params = sum(
                    p.numel() for p in temp_model.parameters() if p.requires_grad
                )
                non_trainable_params = total_params - trainable_params

                return {
                    "ARNet_total": total_params,
                    "ARNet_trainable": trainable_params,
                    "ARNet_non_trainable": non_trainable_params,
                    "total": total_params,
                    "note": f"Parameters counted from temporary model (input_dim={self.input_dim})",
                }
        except Exception as e:
            return {"error": f"Failed to count parameters: {str(e)}", "total": 0}


