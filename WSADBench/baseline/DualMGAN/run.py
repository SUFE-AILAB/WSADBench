
import numpy as np
from WSADBench.myutils import Utils
from WSADBench.baseline.DualMGAN.model import Args
from WSADBench.baseline.DualMGAN.fit import fit_dual, predict
# from WSADBench.baseline.DualMGA

class DualMGAN:
    def __init__(
        self,
        seed,
        verbose=True,
        k_means=10,
        max_iter_MGAOS=2000,
        max_iter_MGAAL=1000,
        lr_sg=0.0001,
        lr_sd=0.01,
        lr_d=0.001,
        decay=1e-6,
        batch_size=1000,
        momentum=0.9,
        nnr_MGAOS=0.4,
        nnr_MGAAL=0.2,
    ):
        self.utils = Utils()
        self.device = self.utils.get_device(True)
        self.seed = seed
        self.utils.set_seed(seed)

        self.verbose = verbose
        # Model components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.args = Args(k_means=k_means,
        max_iter_MGAOS=max_iter_MGAOS,
        max_iter_MGAAL=max_iter_MGAAL,
        lr_sg=lr_sg,
        lr_sd=lr_sd,
        lr_d=lr_d,
        decay=decay,
        batch_size=batch_size,
        momentum=momentum,
        nnr_MGAOS= nnr_MGAOS,
        nnr_MGAAL=nnr_MGAAL)

    def _init_model(self, input_dim):
        """Initialize model components"""

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
                f"Start training DualMGAN model, number of training samples: {X_train.shape[0]}, feature dimension: {X_train.shape[1]}"
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
        self.model = fit_dual(
            train_x=X_train,
            train_semi_y=semi_y,
            batch_size=batch_size,
            device=self.device,
            verbose=self.verbose,
            args = self.args
        )

        print("NTL model training completed")
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
        scores = predict(self.model, X_test, self.device)

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
