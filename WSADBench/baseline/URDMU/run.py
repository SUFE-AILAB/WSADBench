# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from WSADBench.myutils import Utils
from WSADBench.baseline.URDMU.model import URDMULearner
from WSADBench.baseline.URDMU.fit import fit as fit_main
from common_utils.baseline_utils import fit_utils

class URDMU:
    """
    Sultani方法实现
    基于MIL的弱监督视频异常检测

    论文: "Real-world Anomaly Detection in Surveillance Videos"
    """

    def __init__(
        self,

        seed: int = 42,
        dropout: float = 0.6,
        # 训练参数
        epochs: int = 5,  # 可以改
        batch_size: int = 30,
        learning_rate: float = 0.001,
        weight_decay: float = None,
        # 其他参数
        segments_per_video: int = 32,
        verbose: bool = True,
        flag=None, a_nums=None, n_nums=None,seg=None, step=None,
        use_scheduler= None,
        scheduler_milestones= [33, 66],
        is_test = None,

    ):

        self.seed = seed
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.segments_per_video = segments_per_video
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


        self.flag = flag
        self.a_nums= a_nums
        self.n_nums = n_nums
        self.seg = seg
        self.step = step
        self.use_scheduler = use_scheduler
        self.scheduler_milestones = scheduler_milestones
        self.is_test = is_test

    def _init_model(self):
        """初始化模型"""
        if self.model is None:
            # 创建模型
            self.model = URDMULearner(self.input_dim, self.flag, self.a_nums, self.n_nums).to(self.device)
            # 创建优化器
            self.optimizer = optim.Adagrad(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            # 创建学习率调度器
            if self.use_scheduler:
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_milestones)
            if self.verbose:
                print(f"RTFM model initialization completed")
                print(f"Device: {self.device}")
                print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    def fit(self, X, y,  vid_source_clips_num=None,  X_test=None, y_test=None, crops_num=None):
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
            print("Starting Sultani model training")
            print("=" * 60)
            print(f"Number of training samples: {len(X)}")
            print(f"Normal samples: {np.sum(y == 0)}, Anomalous samples: {np.sum(y == 1)}")

        # 数据预处理
        X = self._preprocess_data(X)
        # 判断input_dim
        self.input_dim = X.shape[1]
        # 初始化模型
        self._init_model()
        # 训练模型

        fit_dict = fit_utils(
            trainer=self,
            X_test=X_test,
            # y_test=y_test,
            X_train=X,
            y_train=y,
            model=self.model,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
            clip_num=vid_source_clips_num, crops_num=crops_num
        )
        self.training_history = fit_main(fit_dict['model'], fit_dict['optimizer'], fit_dict['epochs'], fit_dict['device'], fit_dict['X_test'], fit_dict['trainer'], fit_dict['verbose'], fit_dict['normal_loader'], fit_dict['anomaly_loader'])
        self.fitted = True

        training_time = time.time() - start_time

        if self.verbose:
            print(f"Training completed, time elapsed: {training_time:.2f} seconds")

            # Calculate test performance if test data is available
            if X_test is not None and y_test is not None:
                test_scores = self.predict_proba(X_test)
                if len(np.unique(y_test)) > 1:
                    test_auc = roc_auc_score(y_test, test_scores)
                    test_ap = average_precision_score(y_test, test_scores)
                    print(f"Test set AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f}")

        return self

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
        # X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.flag = "Test"
        batch_size = 5000  # 写死
        all_scores = []
        with torch.no_grad():
            for start in tqdm(range(0, len(X), batch_size)):
                end = min(start + batch_size, len(X))
                # print(f'start:{start}, end:{end}')
                batch = torch.FloatTensor(X[start:end]).to(self.device).unsqueeze(0)  # [B, 2048]
                outputs = self.model(batch)["frame"]

                outputs = outputs.squeeze().cpu().numpy()
                # 累加输出
                all_scores.append(outputs)
        # 合并所有 batch 结果
        scores = np.concatenate(all_scores, axis=0)
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
            "segments_per_video": self.segments_per_video,
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
