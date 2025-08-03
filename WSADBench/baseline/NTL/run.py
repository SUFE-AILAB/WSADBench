import torch
import numpy as np
from WSADBench.myutils import Utils
from WSADBench.baseline.NTL.model import TabNeutralAD
from WSADBench.baseline.NTL.fit import fit_rosas, predict_rosas


class NTL:
    def __init__(
        self,
        seed,
        nbatch_per_epoch=16,
        epochs=100,
        # batch_size=128,
        # n_emb=128,
        lr=0.005,
        step_size = 200,
        gamma=0.5,
        # margin=1.0,
        # alpha=0.5,
        # beta=1.0,
        # T=2,
        # k=2,
        # score_loss="smooth",
        milestones=None,
        prt_step=10,
        verbose=True,
        train_method=None,
        query_method=None,
    ):
        self.utils = Utils()
        self.device = self.utils.get_device(True)
        self.seed = seed
        self.utils.set_seed(seed)

        # Model parameters
        self.epochs = epochs
        self.nbatch_per_epoch = nbatch_per_epoch
        # self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.step_size = step_size
        self.train_method = train_method
        self.query_method = query_method
        # self.n_emb = n_emb
        # self.margin = margin
        # self.alpha = alpha
        # self.beta = beta
        # self.T = T
        # self.k = k
        # self.score_loss = score_loss
        self.milestones = milestones if milestones is not None else [epochs]
        self.prt_step = prt_step
        self.verbose = verbose
        # Model components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        if self.verbose:
            print(
                f"NTL initialized : epochs={epochs}, lr={lr}"
            )

    def _init_model(self, input_dim):
        """Initialize model components"""

        # Create network
        self.model = TabNeutralAD(input_dim) \
            .to(self.device)

        # Create optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def fit(self, X_train, y_train, ratio=None):
        """
        Train RoSAS model

        Args:
            X_train: training feature data
            y_train: training labels
            ratio: unused, kept for interface consistency

        Returns:
            self: trained model instance
        """
        # self.model = TabNeutralAD(X_train.shape[1]) \
        #     .to(self.device)
        # 单独算batch_size（5等分）
        batch_size = X_train.shape[0] // 5 + 1
        if self.verbose:
            print(
                f"Start training RoSAS model, number of training samples: {X_train.shape[0]}, feature dimension: {X_train.shape[1]}"
            )

        # Semi-supervised setting
        semi_y = y_train.copy()  # All anomalies are known anomalies

        # Statistics
        n_outliers = len(np.where(y_train == 1)[0])
        n_known_outliers = len(np.where(semi_y == 1)[0])

        if self.verbose:
            print(
                f"Training set size: {X_train.shape[0]}, number of anomalies: {n_outliers}, number of known anomalies: {n_known_outliers}"
            )

        # Initialize model
        self._init_model(X_train.shape[1])
        self.model.to(self.device)  # 转GPU
        print(f'x_dim:{self.model.x_dim}')
        # Train model
        self.model = fit_rosas(
            train_x=X_train,
            train_semi_y=semi_y,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=self.epochs,
            nbatch_per_epoch=self.nbatch_per_epoch,
            batch_size=batch_size,
            device=self.device,
            prt_step=self.prt_step,
            verbose=self.verbose,
            train_method=self.train_method,
            query_method = self.query_method
        )

        print("RoSAS model training completed")
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
        scores = predict_rosas(self.model.model,self.model.loss_fun, X_test, self.device)

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
