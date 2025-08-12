
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from WSADBench.myutils import Utils
from WSADBench.baseline.MGFN.model import mgfn
from WSADBench.baseline.MGFN.fit import fit_mgfn_main_with_crops
from tqdm import tqdm

class MGFN:
    """
    MGFN方法实现
    基于CMLoss的弱监督视频异常检测
    论文: "MGFN : Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection"
    """

    def __init__(  # yaml中出现的，这里必须由
        self,
        seed: int = 42,
        # 模型参数
        input_dim: int = None,
        dropout: float = None,
        # 训练参数
        epochs: int = 5,
        batch_size: int = 30,
        learning_rate: float = 0.001,
        weight_decay: float= None,
        depths: Tuple[int, int, int] = None,  # 写list
        mgfn_types: Tuple[str, str, str] = None,
        # 其他参数
        seg: int = 32,
        verbose: bool = True,
        mag_ratio = None
    ):
        self.seed = seed  # 形成可变参数
        self.dropout = dropout  #
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.verbose = verbose
        # 等yaml参数
        self.mag_ratio =mag_ratio
        self.depths=depths
        self.mgfn_types =mgfn_types
        self.input_dim = input_dim
        self.seg = seg
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
            #  创建模型(传参）
            self.model = mgfn(batch_size= self.batch_size, drop_p=self.dropout,mag_ratio=self.mag_ratio,
                              depths=self.depths, mgfn_types=self.mgfn_types,channels=self.input_dim).to(self.device)

            # 创建优化器
            self.optimizer = optim.Adagrad(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

            # 创建学习率调度器
            # if self.use_scheduler:
            #     self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_milestones)

            if self.verbose:
                print(f"MGFN模型初始化完成")
                print(f"设备: {self.device}")
                print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def fit(self, X, y, X_test=None, y_test=None, vid_source_clips_num=None, crops_num=None):
        start_time = time.time()
        if self.verbose:
            print("=" * 60)
            print("开始训练MGFN模型")
            print("=" * 60)
            print(f"训练样本数: {len(X)}")
            print(f"正常样本: {np.sum(y == 0)}, 异常样本: {np.sum(y == 1)}")
        # 数据预处理
        X = self._preprocess_data(X)
        # 初始化模型
        self._init_model()
        # 训练模型
        self.training_history = fit_mgfn_main_with_crops(
            X_train=X,
            y_train=y,
            X_test = X_test,
            trainer = self,  # 为了调用预测
            model=self.model,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
            clip_num=vid_source_clips_num, crops_num=crops_num,
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
        def new_test_feature(feat: np.ndarray) -> torch.Tensor:
            """
            输入:
                feat: np.ndarray，形状为 [89, 2048]
            输出:
                torch.Tensor，形状为 [1, 1, 89, 2049]
            """
            # 1. 计算 L2 范数 => [89, 1]
            feature_mag = np.linalg.norm(feat, axis=1, keepdims=True)

            # 2. 拼接 => [89, 2049]
            combined_feat = np.concatenate((feat, feature_mag), axis=1)

            # 3. 增加两个维度 => [1, 1, 89, 2049]
            output_feat = np.expand_dims(np.expand_dims(combined_feat, axis=0), axis=0)

            # 4. 转换为 torch.Tensor 返回
            return torch.from_numpy(output_feat).float()
        batch_size = 5000  # 写死
        all_scores = []
        with torch.no_grad():
            for start in tqdm(range(0, len(X), batch_size)):
                end = min(start + batch_size, len(X))
                # print(f'start:{start}, end:{end}')
                batch = X[start:end]  # [B, 2048]

                # 特征增强 & 维度扩展: [B, 2048] -> [1, 1, B, 2049]
                batch_tensor = new_test_feature(batch).to(self.device)  # [1, 1, B, 2049]
                # batch_tensor = batch_tensor.permute(0, 2, 1, 3)  # -> [1, B, 1, 2049]  #

                # 模型前向
                outputs = self.model(batch_tensor)  # [1, B, 1] or [1, B]
                outputs = outputs.squeeze().cpu().numpy()  # [B]

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
                    "mgfn_total": total_params,
                    "mgfn_trainable": trainable_params,
                    "mfgn_non_trainable": non_trainable_params,
                    "total": total_params,
                }
            # else:
            #     # 如果模型还没有初始化，创建临时模型来计算参数
            #     from WSADBench.baseline.MGFN.model import MGFNLearner
            #
            #     temp_model = MGFNLearner(input_dim=self.input_dim)
            #     total_params = sum(p.numel() for p in temp_model.parameters())
            #     trainable_params = sum(
            #         p.numel() for p in temp_model.parameters() if p.requires_grad
            #     )
            #     non_trainable_params = total_params - trainable_params
            #
            #     return {
            #         "mgfn_total": total_params,
            #         "mgfn_trainable": trainable_params,
            #         "mgfn_non_trainable": non_trainable_params,
            #         "total": total_params,
            #         "note": f"Parameters counted from temporary model (input_dim={self.input_dim})",
            #     }
        except Exception as e:
            return {"error": f"Failed to count parameters: {str(e)}", "total": 0}
