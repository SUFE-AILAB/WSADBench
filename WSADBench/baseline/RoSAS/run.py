import torch
import numpy as np
from WSADBench.myutils import Utils
from WSADBench.baseline.RoSAS.model import EDOSNet, RoSASLoss
from WSADBench.baseline.RoSAS.fit import fit_rosas, predict_rosas


class RoSAS:
    """
    RoSAS (Robust Supervised Anomaly Segmentation) model implementation

    A semi-supervised anomaly detection method combining triplet loss and Mixup regularization
    """

    def __init__(
        self,
        seed,
        model_name="RoSAS",
        nbatch_per_epoch=16,
        epochs=100,
        batch_size=128,
        network="e1s1",
        n_emb=128,
        lr=0.005,
        margin=1.0,
        alpha=0.5,
        beta=1.0,
        T=2,
        k=2,
        score_loss="smooth",
        milestones=None,
        prt_step=10,
        verbose=True,
    ):
        """
        Initialize RoSAS model

        Args:
            seed: random seed
            model_name: model name
            nbatch_per_epoch: number of batches per epoch
            epochs: number of training epochs
            batch_size: batch size
            network: network structure identifier
            n_emb: embedding dimension
            lr: learning rate
            margin: triplet loss margin
            alpha: Dirichlet distribution parameter
            beta: loss function parameter
            T: temperature parameter
            k: number of Mixup samples
            score_loss: score loss type
            milestones: learning rate scheduler milestones
            prt_step: print step
            verbose: whether to print verbose logs
        """
        self.utils = Utils()
        self.device = self.utils.get_device(True)
        self.seed = seed
        self.utils.set_seed(seed)

        # Model parameters
        self.epochs = epochs
        self.nbatch_per_epoch = nbatch_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_emb = n_emb
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.k = k
        self.score_loss = score_loss
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
                f"RoSAS initialized : epochs={epochs}, batch_size={batch_size}, lr={lr}, n_emb={n_emb}, "
                f"margin={margin}, alpha={alpha}, beta={beta}, T={T}, k={k}, score_loss={score_loss}"
            )

    def _init_model(self, input_dim):
        """Initialize model components"""
        # Network structure parameters
        n_hidden = input_dim + int((self.n_emb - input_dim) * 0.5)
        n_hidden2 = int(0.5 * self.n_emb)

        # Create network
        self.model = EDOSNet(n_feature=input_dim, n_hidden=n_hidden, n_hidden2=n_hidden2, n_emb=self.n_emb).to(
            self.device
        )

        # Create loss function
        self.criterion = RoSASLoss(
            l2_reg_weight=1e-2,
            score_loss=self.score_loss,
            margin=self.margin,
            alpha=self.alpha,
            beta=self.beta,
            T=self.T,
            k=self.k,
            device=self.device,
        )

        # Create optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.4)

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
            batch_size=self.batch_size,
            device=self.device,
            prt_step=self.prt_step,
            verbose=self.verbose,
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
        scores = predict_rosas(self.model, X_test, self.device)

        return scores
