from __future__ import division
from __future__ import print_function
from tracemalloc import start

import torch
#from torch import nn
#from multiprocessing import Pool, freeze_support, cpu_count, set_start_method

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import time
from tqdm import tqdm
from WSADBench.myutils import Utils
from pyod.models.base import BaseDetector
from pyod.utils.torch_utility import get_activation_by_name
from pyod.utils.stat_models import pairwise_distances_no_broadcast
import io
from torch.utils.tensorboard import SummaryWriter
from WSADBench.baseline.PUMA.model import inner_autoencoder, PyODDataset
from WSADBench.baseline.PUMA.fit import fit_PUMA, _logistic,_weightnoisyor


class PUMA:
    """
    PUMA方法实现
    基于" Learning from Positive and Unlabeled Multi-Instance Bags in Anomaly Detection."论文实现
    Multiple Instance Learning (MIL) 弱监督异常检测方法

    """

    def __init__(self,
                 seed = 42,
                 input_dim=None,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 batch_norm=True,
                 learning_rate=0.01,
                 epochs=100,
                 batch_size=10,
                 n_samples=10,
                 mu1 = 0, 
                 sigma1 = 0.1,
                 mu2 = 1,
                 sigma2 = 0.1,
                 n_neg = 10,
                 dropout_rate=0.1,
                 weight_decay=1e-5,
                 preprocessing=True,
                 contamination=0.1,
                 random_state = 331,
                 verbose = True,
                 cont_factor = 0.1):
        torch.manual_seed(random_state)
        self.contamination = contamination
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.epochs = epochs
        self.batch_size = batch_size

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.n_neg = n_neg
        self.n_samples = n_samples
        self.loss_fn = torch.nn.MSELoss()
        #self.writer = SummaryWriter('/home/lorenzo/projects/MI_Learning/csvfiles/log/')
        self.cont_factor = cont_factor
        
        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

        self.verbose = verbose

                #工具类
        self.utils = Utils()
        #设备设置
        self.device = self.utils.get_device(True)
        #随机种子设置
        self.seed = seed
        self.utils.set_seed(seed)

        #模型内部状态
        self.model = None
        self.optimizer = None
        self.train_history = None

    def _init_model(self):
        """初始化模型"""
        self.model = inner_autoencoder(
            input_dim=self.input_dim,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation).to(self.device)
        
        self.model_A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True).to(self.device)
        self.model_A.grad = torch.tensor(torch.rand(1)).to(self.device)
        self.model_B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True).to(self.device)
        
        # move to device and print model information
        self.model = self.model.to(self.device, non_blocking = True)
        if self.verbose:
            print(self.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,amsgrad = True
        )


        if self.verbose:
            print(f"PUMA模型初始化完成")
            print(f"设备: {self.device}")
            print(f"model参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    # noinspection PyUnresolvedReferences
    def fit(self, X, y):  
        """
        训练PUMA模型
        Args:
            X: 训练特征 [n_bags * n_samples, input_dim]
            y: 训练标签 [n_bags * n_samples] (0: normal, 1: anomaly)
        Returns:
            self
        """
        start_time = time.time()
        if self.verbose:
            print("=" * 60)
            print("开始训练PUMA模型")
            print("=" * 60)
            print(f"训练样本数: {len(X)}")
            print(f"正常样本: {np.sum(y == 0)}, 异常样本: {np.sum(y == 1)}")

        # validate inputs X and y (optional)
        #X_inst = check_array(X_inst)
        #初始化模型
        self._init_model()
        #将X重塑为三维，[n_bags, n_samples, input_dim]
        X_bags = X.reshape(-1, self.n_samples, X.shape[-1])

        self.n_bags = X_bags.shape[0]
        self.n_samples = X_bags.shape[1]
        self.input_dim = X_bags.shape[2]

        n_samples = self.n_samples
        #将y重塑为包标签
        y_bags = y[::n_samples]
        device = self.device
        
        if self.n_neg == -1:
            npos = len(np.where(y_bags == 1)[0])
            self.n_neg = npos
            
        if self.cont_factor>0:
            tot_inst = X.shape[0]
            inst_labeledbags = len(np.where(y_bags == 1)[0])
            inst_unlabeledbags = (tot_inst - len(np.where(y_bags == 1)[0]))
            self.cont_factor = max((self.cont_factor*tot_inst - 0.25*inst_labeledbags)/inst_unlabeledbags,0)
        y_bags = torch.from_numpy(y_bags)
        X_inst = X  # X已在运行脚本中处理为二维
        
        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X_inst, axis=0), np.std(X_inst, axis=0)
            train_set = PyODDataset(X_bags=X_bags, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X_bags=X_bags)
        
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers = 0,
                                                   pin_memory = True)
        
        
        # 训练模型
        self.train_history = fit_PUMA(model = self.model,
                                      model_A = self.model_A,
                                      model_B = self.model_B,
                                      optimizer = self.optimizer,
                                      train_loader = train_loader,
                                      y_bags = y_bags,
                                      epochs = self.epochs,
                                      n_bags = self.n_bags,
                                      n_samples = n_samples,
                                      input_dim = self.input_dim,
                                      n_neg = self.n_neg,
                                      loss_fn = self.loss_fn,
                                      mu1 = self.mu1,
                                      sigma1 = self.sigma1,
                                      mu2 = self.mu2,
                                      sigma2 = self.sigma2,
                                      device = self.device,
                                      cont_factor = self.cont_factor,
                                      verbose=self.verbose)
                                       
        train_history = self.train_history
        self.model.load_state_dict(train_history["best_model_dict"])
        self.fitted = True
        training_time = time.time() - start_time
        if self.verbose:
            print(f"训练完成，耗时: {training_time:.2f}秒")
            
        # self.bag_decision_scores_, self.instance_decision_scores_ = self.decision_function(X)
        # self._process_decision_scores()
        
        return self
    

    def predict_proba(self, X):
            
            if not self.fitted:
                raise ValueError("模型尚未训练，请先调用fit()方法")
            if X.ndim == 2:
                X = X.reshape(-1, self.n_samples, X.shape[-1])

            # note the shuffle may be true but should be False
            if self.preprocessing:
                dataset = PyODDataset(X_bags=X, mean=self.mean, std=self.std)
            else:
                dataset = PyODDataset(X_bags=X)

            dataloader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False)
            # enable the evaluation mode
            self.model.eval()
            
            # construct the vector for holding the reconstruction error
            b_scores = torch.zeros([X.shape[0], 1]).to(self.device, non_blocking = True).float()
            i_scores = torch.zeros([X.shape[0]*X.shape[1], 1]).to(self.device, non_blocking = True).float()

            with torch.no_grad():
                for data, data_idx in dataloader:
                    local_batch_size, _, _ = data.size()
                    data_idx = data_idx.to(self.device, non_blocking = True)
                    mi = data_idx[0]
                    ma = data_idx[local_batch_size-1]+1
                    data_inst = torch.reshape(data, (local_batch_size*self.n_samples, self.input_dim))
                    data_inst = data_inst.to(self.device, non_blocking = True).float()

                    l1 = torch.nn.PairwiseDistance(p=2, eps=0)(data_inst,self.model(data_inst))
                    l1 = torch.reshape(l1, (local_batch_size, self.n_samples, 1))
                    instance_scr = _logistic(l1, self.model_A, self.model_B)
                    i_scores[mi*self.n_samples:ma*self.n_samples]=instance_scr.reshape(local_batch_size*self.n_samples,1)
                    pi = _weightnoisyor(instance_scr, self.mu1, self.sigma1, self.mu2, self.sigma2, self.device)
                    b_scores[data_idx] = pi.reshape(local_batch_size,1)

            return i_scores.cpu().numpy()

    def predict(self, X,threshold=0.5):
            """
            预测异常标签

            Args:
                X: 输入特征 [n_samples, feature_dim]

            Returns:
                异常标签 [n_samples] (0: normal, 1: anomaly)
            """

            scores = self.predict_proba(X)

            return (scores > threshold).astype(int)


    def get_params(self, deep=True):
        """
        获取 PUMA 模型的参数（sklearn 兼容）

        Args:
            deep (bool): 是否深度获取参数（此处仅做接口兼容，不做递归）

        Returns:
            dict: 参数字典
        """
        return {
            # 训练/优化相关
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            "random_state": getattr(self, "random_state", None),
            "gpu": getattr(self, "gpu", -1),

            # 模型结构相关
            "input_dim": getattr(self, "input_dim", None),          # 等同 input_dim
            "hidden_neurons": getattr(self, "hidden_neurons", None),
            "hidden_activation": getattr(self, "hidden_activation", "relu"),
            "batch_norm": getattr(self, "batch_norm", True),
            "dropout_rate": getattr(self, "dropout_rate", 0.1),

            # PUMA 特有超参数
            "mu1": getattr(self, "mu1", 0.0),
            "sigma1": getattr(self, "sigma1", 0.1),
            "mu2": getattr(self, "mu2", 1.0),
            "sigma2": getattr(self, "sigma2", 0.1),
            "n_neg": getattr(self, "n_neg", 10),
            "cont_factor": getattr(self, "cont_factor", 0.1),

            # 预处理/数据相关
            "preprocessing": getattr(self, "preprocessing", True),
            "contamination": getattr(self, "contamination", 0.1),

            # 其他运行中缓存（可选是否暴露）
            # "mean": getattr(self, "mean", None),
            # "std": getattr(self, "std", None),
        }


    def set_params(self, **params):
        """
        设置 PUMA 模型参数（sklearn 兼容）
        - 仅对已存在的属性进行覆盖
        - 如果模型已经初始化，则重置（以便使用新参数重新构建）

        Args:
            **params: 参数字典

        Returns:
            self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 如果模型已经初始化，需要重新初始化
        if hasattr(self, "model") and self.model is not None:
            self.model = None
            # 若类中有 fitted / best_model_dict 之类的状态，也一并重置
            if hasattr(self, "fitted"):
                self.fitted = False
            if hasattr(self, "best_model_dict"):
                self.best_model_dict = None
            if hasattr(self, "best_loss"):
                self.best_loss = float("inf")

        return self


    def parameter_count(self):
        """
        计算 PUMA（InnerAutoEncoder）模型的参数数量

        Returns:
            dict: 各项参数数量统计
        """
        try:
            # 已初始化模型则直接统计
            if hasattr(self, "model") and self.model is not None:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters()
                                    if p.requires_grad)
                non_trainable_params = total_params - trainable_params
                return {
                    "puma_total": total_params,
                    "puma_trainable": trainable_params,
                    "puma_non_trainable": non_trainable_params,
                    "total": total_params,
                }
            else:
                # 未初始化则使用当前参数创建“临时模型”进行统计
                # 需要可用的 input_dim / hidden_neurons / hidden_activation / batch_norm / dropout_rate
                if not hasattr(self, "input_dim") or self.input_dim is None:
                    return {"error": "input_dim is not set; cannot build temp model", "total": 0}

                from model import build_model  # 或者改成你项目中实际的导入路径
                temp_model = build_model(
                    input_dim=self.input_dim,
                    hidden_neurons=getattr(self, "hidden_neurons", None),
                    hidden_activation=getattr(self, "hidden_activation", "relu"),
                    batch_norm=getattr(self, "batch_norm", True),
                    dropout_rate=getattr(self, "dropout_rate", 0.1),
                )

                total_params = sum(p.numel() for p in temp_model.parameters())
                trainable_params = sum(p.numel() for p in temp_model.parameters()
                                    if p.requires_grad)
                non_trainable_params = total_params - trainable_params

                return {
                    "puma_total": total_params,
                    "puma_trainable": trainable_params,
                    "puma_non_trainable": non_trainable_params,
                    "total": total_params,
                    "note": f"Parameters counted from temporary model (input_dim={self.input_dim})",
                }
        except Exception as e:
            return {"error": f"Failed to count parameters: {str(e)}", "total": 0}
