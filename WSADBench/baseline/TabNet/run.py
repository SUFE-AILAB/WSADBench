from re import X
from tracemalloc import start
import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
import scipy
from pytorch_tabnet.tab_model import TabNetClassifier
from WSADBench.myutils import Utils
import time
from sklearn.model_selection import train_test_split

class TabNet(TabModel):      #使用的TabNetClassifier
    def __init__(self,seed,n_d,n_a,gamma,batch_size,n_steps=3,weights=1,verbose=True):
        super().__init__()
        self.verbose = verbose
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        #工具类
        self.utils = Utils()
        #设备设置
        self.device = self.utils.get_device(True)
        #随机种子设置
        self.seed = seed
        self.utils.set_seed(seed)
        #设置模型内部状态
        self.weights = weights
        self.model = None
        self.input_dim = None

    def init_model(self):
        """初始化模型"""
        self.model = TabNetClassifier(n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, gamma=self.gamma)

        if self.verbose:
            print(f"TabNet模型初始化完成")
            print(f"设备: {self.device}")
            # print(f"model参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def fit(self, X_train, y_train):
        """
        训练TabNet模型
        Args:
            X: 训练特征 [batch_size, input_dim]
            y: 训练标签 [batch_size, 1] (0: normal, 1: anomaly)
        Returns:
            self
        """
        start_time = time.time()
        
        if self.model is None:
            self.init_model()
        
        #划分训练集，验证集
        X_train, X_val, y_train, y_val = train_test_split(
                                    X_train, y_train,
                                    test_size=0.2,       # 验证集占比 20%
                                    random_state=42,     # 固定随机种子，保证可复现
                                    stratify=y_train     # 保证分类任务类别分布一致
                                )
        
        self.model.fit(
            X_train=X_train, y_train=y_train,eval_set=[(X_val,y_val)],batch_size=self.batch_size,weights=self.weights)

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
        # 确保输入是 numpy，不要转 torch
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()
        # Prediction
        scores = self.model.predict(X_test)

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
                current_criterion = self.criterion
                current_optimizer = self.optimizer
                current_scheduler = self.scheduler

                # 临时初始化模型
                self._init_model(temp_input_dim)

                # 计算参数
                params = {"model": sum(p.numel() for p in self.model.parameters())}
                params["total"] = params["model"]

                # 恢复状态
                self.model = current_model
                self.criterion = current_criterion
                self.optimizer = current_optimizer
                self.scheduler = current_scheduler

                return params

            except Exception as e:
                return {"error": f"Failed to count parameters: {str(e)}"}
        else:
            params = {"model": sum(p.numel() for p in self.model.parameters())}
            params["total"] = params["model"]
            return params
