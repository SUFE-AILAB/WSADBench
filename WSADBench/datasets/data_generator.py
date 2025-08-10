import numpy as np
import pandas as pd
import random
import os
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.mixture import GaussianMixture

from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE

from WSADBench.myutils import Utils, import_class
from pathlib import Path
import yaml


# currently, data generator only supports for generating the binary classification datasets
class DataGenerator:
    def __init__(
        self,
        seed: int = 42,
        dataset: str = None,
        test_size: float = 0.3,
        generate_duplicates=True,
        n_samples_threshold=1000,
    ):
        """
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        """

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        self.wd = Path(__file__).resolve().parent.parent.parent

        config_path = self.wd / "WSADBench/datasets/dataset_configs/index.yaml"
        with open(config_path, "r") as f:
            self.configs = yaml.safe_load(f)

        # dataset list
        self.all_dataset_list = self.generate_dataset_list()

        # myutils function
        self.utils = Utils()

    def generate_dataset_list(self):

        all_dataset_list = {}
        for dataset_kind, ds_config in self.configs.items():
            if "DATA_DIR" in ds_config:
                all_dataset_list[dataset_kind] = [
                    f.stem for f in (self.wd / ds_config["DATA_DIR"]).iterdir() if f.suffix in [".npz", ".npy"]
                ]
                continue

            all_dataset_list[dataset_kind] = list(ds_config.keys())

        return all_dataset_list

    def generate_realistic_synthetic(self, X, y, realistic_synthetic_mode, alpha: int, percentage: float):
        """
        Currently, four types of realistic synthetic outliers can be generated:
        1. local outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified covariance
        2. global outliers: where normal data follows the GMM distribuion, and anomalies follow the uniform distribution
        3. dependency outliers: where normal data follows the vine coupula distribution, and anomalies follow the independent distribution captured by GaussianKDE
        4. cluster outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified mean

        :param X: input X
        :param y: input y
        :param realistic_synthetic_mode: the type of generated outliers
        :param alpha: the scaling parameter for controling the generated local and cluster anomalies
        :param percentage: controling the generated global anomalies
        """

        if realistic_synthetic_mode in ["local", "cluster", "dependency", "global"]:
            pass
        else:
            raise NotImplementedError

        # the number of normal data and anomalies
        pts_n = len(np.where(y == 0)[0])
        pts_a = len(np.where(y == 1)[0])

        # only use the normal data to fit the model
        X = X[y == 0]
        y = y[y == 0]

        # generate the synthetic normal data
        if realistic_synthetic_mode in ["local", "cluster", "global"]:
            # select the best n_components based on the BIC value
            metric_list = []
            n_components_list = list(np.arange(1, 10))

            for n_components in n_components_list:
                gm = GaussianMixture(n_components=n_components, random_state=self.seed).fit(X)
                metric_list.append(gm.bic(X))

            best_n_components = n_components_list[np.argmin(metric_list)]

            # refit based on the best n_components
            gm = GaussianMixture(n_components=best_n_components, random_state=self.seed).fit(X)

            # generate the synthetic normal data
            X_synthetic_normal = gm.sample(pts_n)[0]

        # we found that copula function may occur error in some datasets
        elif realistic_synthetic_mode == "dependency":
            # sampling the feature since copulas method may spend too long to fit
            if X.shape[1] > 50:
                idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
                X = X[:, idx]

            copula = VineCopula("center")  # default is the C-vine copula
            copula.fit(pd.DataFrame(X))

            # sample to generate synthetic normal data
            X_synthetic_normal = copula.sample(pts_n).values

        else:
            pass

        # generate the synthetic abnormal data
        if realistic_synthetic_mode == "local":
            # generate the synthetic anomalies (local outliers)
            gm.covariances_ = alpha * gm.covariances_
            X_synthetic_anomalies = gm.sample(pts_a)[0]

        elif realistic_synthetic_mode == "cluster":
            # generate the clustering synthetic anomalies
            gm.means_ = alpha * gm.means_
            X_synthetic_anomalies = gm.sample(pts_a)[0]

        elif realistic_synthetic_mode == "dependency":
            X_synthetic_anomalies = np.zeros((pts_a, X.shape[1]))

            # using the GuassianKDE for generating independent feature
            for i in range(X.shape[1]):
                kde = GaussianKDE()
                kde.fit(X[:, i])
                X_synthetic_anomalies[:, i] = kde.sample(pts_a)

        elif realistic_synthetic_mode == "global":
            # generate the synthetic anomalies (global outliers)
            X_synthetic_anomalies = []

            for i in range(X_synthetic_normal.shape[1]):
                low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
                high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)

                X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))

            X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

        else:
            pass

        X = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
        y = np.append(np.repeat(0, X_synthetic_normal.shape[0]), np.repeat(1, X_synthetic_anomalies.shape[0]))

        return X, y

    """
    Here we also consider the robustness of baseline models, where three types of noise can be added
    1. Duplicated anomalies, which should be added to training and testing set, respectively
    2. Irrelevant features, which should be added to both training and testing set
    3. Annotation errors (Label flips), which should be only added to the training set
    """

    def add_duplicated_anomalies(self, X, y, duplicate_times: int):
        if duplicate_times <= 1:
            pass
        else:
            # index of normal and anomaly data
            idx_n = np.where(y == 0)[0]
            idx_a = np.where(y == 1)[0]

            # generate duplicated anomalies
            idx_a = np.random.choice(idx_a, int(len(idx_a) * duplicate_times))

            idx = np.append(idx_n, idx_a)
            random.shuffle(idx)
            X = X[idx]
            y = y[idx]

        return X, y

    def add_irrelevant_features(self, X, y, noise_ratio: float):
        # adding uniform noise
        if noise_ratio == 0.0:
            pass
        else:
            noise_dim = int(noise_ratio / (1 - noise_ratio) * X.shape[1])
            if noise_dim > 0:
                X_noise = []
                for i in range(noise_dim):
                    idx = np.random.choice(np.arange(X.shape[1]), 1)
                    X_min = np.min(X[:, idx])
                    X_max = np.max(X[:, idx])

                    X_noise.append(np.random.uniform(X_min, X_max, size=(X.shape[0], 1)))

                # concat the irrelevant noise feature
                X_noise = np.hstack(X_noise)
                X = np.concatenate((X, X_noise), axis=1)
                # shuffle the dimension
                idx = np.random.choice(np.arange(X.shape[1]), X.shape[1], replace=False)
                X = X[:, idx]

        return X, y

    def add_label_contamination(self, X, y, noise_ratio: float):
        if noise_ratio == 0.0:
            pass
        else:
            # here we consider the label flips situation: a label is randomly filpped to another class with probability p (i.e., noise ratio)
            idx_flips = np.random.choice(np.arange(len(y)), int(len(y) * noise_ratio), replace=False)
            y[idx_flips] = 1 - y[idx_flips]  # change 0 to 1 and 1 to 0

        return X, y

    def generator(
        self,
        X=None,
        y=None,
        minmax=True,
        la=None,
        labeled_abnormal_ratio=None,
        at_least_one_labeled=False,
        realistic_synthetic_mode=None,
        alpha: int = 5,
        percentage: float = 0.1,
        noise_type=None,
        duplicate_times: int = 2,
        contam_ratio=1.00,
        noise_ratio: float = 0.05,
        la_shortage_mode="ignore",
        unlabeled_processing = "0"
    ):
        """
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        la_shortage_mode: behavior when anomaly count < required la
        - 'raise': raise error
        - 'ignore': return all available anomalies only
        """

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        data = None
        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
            print("Testing on customized dataset...")
        else:

            for kind, dataset_list in self.all_dataset_list.items():
                if kind in ["video"]:
                    real_ds_name = self.dataset.split(".")[0]  # for parameterized dataset
                else:
                    real_ds_name = self.dataset
                    
                if real_ds_name not in dataset_list:
                    continue

                if "DATA_DIR" in self.configs[kind]:
                    data_path = (
                        self.wd / self.configs[kind]["DATA_DIR"] / (self.dataset + self.configs[kind]["END_WITH"])
                    )
                    data = np.load(data_path, allow_pickle=True)

                    X, y = data["X"], data["y"]
                    break

                ManagerClass = import_class(self.configs[kind][real_ds_name]["MANAGER_CLASS"])
                ds_config = self.configs[kind][real_ds_name]
                ds_config["working_dir"] = self.wd
                ds_config["seed"] = self.seed
                data_manager = ManagerClass(ds_config)
                ds_param = self.dataset.split(".")[1:]
                _params = {}
                for param in ds_param:
                    if param[0] == 's' and param[1:].isdigit():
                        _params["num_segments"] = int(param[1:])
                    if param[0] == 'l' and param[1:].isdigit():
                        _params["limit"] = int(param[1:])
                data = data_manager.load_data(**_params)

            if data is None:
                raise NotImplementedError(f"Dataset {self.dataset} not found in the available datasets.")

        # 如果已经划分好了训练和测试集，测跳过划分：TODO: 可能需要修改，因为也跳过了异常注入
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(data.keys()):
            X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
        else:
            # if the dataset is too small, generating duplicate smaples up to n_samples_threshold
            if len(y) < self.n_samples_threshold and self.generate_duplicates:
                print(f"generating duplicate samples for dataset {self.dataset}...")
                self.utils.set_seed(self.seed)
                idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_threshold, replace=True)
                X = X[idx_duplicate]
                y = y[idx_duplicate]

            # if the dataset is too large, subsampling for considering the computational cost
            if len(y) > 10000:
                print(f"subsampling for dataset {self.dataset}...")
                self.utils.set_seed(self.seed)
                idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
                X = X[idx_sample]
                y = y[idx_sample]

            # whether to generate realistic synthetic outliers
            if realistic_synthetic_mode is not None:
                # we save the generated dependency anomalies, since the Vine Copula could spend too long for generation
                if realistic_synthetic_mode == "dependency":
                    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")
                    filename = "dependency_anomalies_" + self.dataset + "_" + str(self.seed) + ".npz"

                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    try:
                        data_dependency = np.load(os.path.join(filepath, filename), allow_pickle=True)
                        X = data_dependency["X"]
                        y = data_dependency["y"]
                    except:
                        # raise NotImplementedError
                        print(f"Generating dependency anomalies...")
                        X, y = self.generate_realistic_synthetic(
                            X, y, realistic_synthetic_mode=realistic_synthetic_mode, alpha=alpha, percentage=percentage
                        )
                        np.savez_compressed(os.path.join(filepath, filename), X=X, y=y)
                        pass

                else:
                    X, y = self.generate_realistic_synthetic(
                        X, y, realistic_synthetic_mode=realistic_synthetic_mode, alpha=alpha, percentage=percentage
                    )

            # whether to add different types of noise for testing the robustness of benchmark models
            if noise_type is None:
                pass

            elif noise_type == "duplicated_anomalies":
                # X, y = self.add_duplicated_anomalies(X, y, duplicate_times=duplicate_times)
                pass

            elif noise_type == "irrelevant_features":
                X, y = self.add_irrelevant_features(X, y, noise_ratio=noise_ratio)

            elif noise_type == "label_contamination":
                pass

            else:
                raise NotImplementedError

            print(f"current noise type: {noise_type}")

            # show the statistic
            self.utils.data_description(X=X, y=y)

            # spliting the current data to the training set and testing set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=True, stratify=y
            )

            # we respectively generate the duplicated anomalies for the training and testing set
            if noise_type == "duplicated_anomalies":
                X_train, y_train = self.add_duplicated_anomalies(X_train, y_train, duplicate_times=duplicate_times)
                X_test, y_test = self.add_duplicated_anomalies(X_test, y_test, duplicate_times=duplicate_times)

            # notice that label contamination can only be added in the training set
            elif noise_type == "label_contamination":
                X_train, y_train = self.add_label_contamination(X_train, y_train, noise_ratio=noise_ratio)

            # minmax scaling
            if minmax:
                scaler = MinMaxScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

        # idx of normal samples and unlabeled/labeled anomalies
        # idx of normal samples and anomaly samples
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        # Step 1: 计算有标签异常数量
        if isinstance(labeled_abnormal_ratio, float) and at_least_one_labeled: 
            num_labeled_abnormal = ceil(np.round(la * labeled_abnormal_ratio))

        elif isinstance(labeled_abnormal_ratio, float) and not at_least_one_labeled:
            num_labeled_abnormal = int(np.round(la * labeled_abnormal_ratio))

        elif isinstance(labeled_abnormal_ratio, int):
            num_labeled_abnormal = labeled_abnormal_ratio
        else:
            raise NotImplementedError("labeled_abnormal_ratio 必须为 float 或 int 类型")

        # 边界保护，不能超过现有异常数
        if num_labeled_abnormal > len(idx_anomaly) and la_shortage_mode == "ignore":
            num_labeled_abnormal = min(num_labeled_abnormal, len(idx_anomaly))
            raise "有标签异常数量超过现有异常样本数量，已调整为现有异常样本数量"
        elif num_labeled_abnormal > len(idx_anomaly) and la_shortage_mode == "raise":
            raise ValueError("有标签异常数量超过现有异常样本数量，请调整参数或数据集")

        # Step 2: 有标签正常数量 = 总数 - 有标签异常
        num_labeled_normal = la - num_labeled_abnormal
        # 不能超过现有正常样本
        if num_labeled_normal > len(idx_normal) and la_shortage_mode == "ignore":
            num_labeled_normal = min(num_labeled_normal, len(idx_normal))
        elif num_labeled_normal > len(idx_normal) and la_shortage_mode == "raise":
            raise ValueError("有标签正常数量超过现有正常样本数量，请调整参数或数据集")

        # Step 3: 采样已知标签的异常和正常索引
        idx_labeled_anomaly = np.random.choice(idx_anomaly, num_labeled_abnormal, replace=False) if num_labeled_abnormal > 0 else np.array([], dtype=int)
        idx_labeled_normal = np.random.choice(idx_normal, num_labeled_normal, replace=False) if num_labeled_normal > 0 else np.array([], dtype=int)

        # Step 4: 剩下的异常和正常样本都作为unlabeled
        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        idx_unlabeled_normal = np.setdiff1d(idx_normal, idx_labeled_normal)

        # 可选：去除unlabeled中的异常污染
        if noise_type == "anomaly_contamination":
            idx_unlabeled_anomaly = self.remove_anomaly_contamination(idx_unlabeled_anomaly, contam_ratio)

        # 组装无标签集（正常+未标注异常）
        idx_unlabeled = np.append(idx_unlabeled_normal, idx_unlabeled_anomaly)

        # 清理
        del idx_anomaly, idx_normal, idx_unlabeled_anomaly, idx_unlabeled_normal

        # idx_labeled_anomaly 和 idx_labeled_normal 是有标签训练集，idx_unlabeled 是无标签训练集


        # 设置掩码来区分有标签和无标签样本
        mask = np.ones_like(y_train, dtype=int)  # 默认全 1（有标签）
        mask[idx_unlabeled] = 0                   # 无标签索引处的mask设为 0

        #处理无标签样本方式：目前将无标签样本的标签设为0
        if unlabeled_processing == "0":
            y_train[mask==0] = 0

        result = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "mask": mask
        }

        if data is not None and isinstance(data, dict):
            data.update(result)
            result = data
        return result
