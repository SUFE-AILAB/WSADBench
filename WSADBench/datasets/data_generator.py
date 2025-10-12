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
import torch


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
    #增加标签污染，inaccurate
    def add_label_contamination(self, X, y,flip_normal_ratio, flip_abnormal_ratio):
        
        """
        Add label noise by flipping some normal samples to anomaly and some anomaly samples to normal.
        
        Args:
            X: feature matrix
            y: label array (0 = normal, 1 = anomaly)
            flip_normal_ratio: float, percentage of normal samples (0) to flip to anomalies (1)
            flip_abnormal_ratio: float, percentage of anomaly samples (1) to flip to normal (0)

        Returns:
            X, y with noisy labels
        """
        # 先转整型，防止后续下游 safe-casting 报错
        y_noisy = y.astype(np.int64, copy=True)
        n_samples = len(y)

        # 找到正常和异常样本索引
        normal_idx = np.where(y == 0)[0]
        abnormal_idx = np.where(y == 1)[0]

        # 需要翻转的样本数量
        if type(flip_abnormal_ratio) == float:
            n_flip_normal = ceil(len(normal_idx) * flip_normal_ratio)
            n_flip_abnormal = ceil(len(abnormal_idx) * flip_abnormal_ratio)

            # 随机采样要翻转的索引
            flip_normals = np.random.choice(normal_idx, size=n_flip_normal, replace=False)
            flip_abnormals = np.random.choice(abnormal_idx, size=n_flip_abnormal, replace=False)

        elif type(flip_abnormal_ratio) == int:
            flip_normals = np.random.choice(normal_idx, size=flip_normal_ratio, replace=False)
            flip_abnormals = np.random.choice(abnormal_idx, size=flip_abnormal_ratio, replace=False)
        
        if len(flip_normals) + len(flip_abnormals) != 0:
            # 执行标签翻转
            y_noisy[flip_normals] = 1
            y_noisy[flip_abnormals] = 0
        
        else:
            raise ValueError("No samples were selected for label flipping. Please check the flip ratios.")
        
        print(f"Label contamination: flipped {len(flip_normals)} normal samples to anomalies and {len(flip_abnormals)} anomalies to normal.")
        
        return X, y_noisy

    def generator(
            self,
            X=None,
            y=None,
            minmax=True,
            la=None,
            eln=None,
            ru=None,
            target_for_unlabeled=None,
            at_least_one_labeled=False,
            realistic_synthetic_mode=None,
            alpha: int = 5,
            percentage: float = 0.1,
            noise_type = None,
            flip_normal_ratio: float = 0.0,
            flip_abnormal_ratio: float = 0.0,
            duplicate_times: int = 2,
            contam_ratio=1.00,
            noise_ratio: float = 0.05,
            shortage_mode="ignore",
            data_type=None,
            # seg_num=None,
            # pretrain_model=None,
    ):
        """
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        eln: relative labeled normal samples, can be either the ratio of labeled normal samples or the number of labeled normal samples
        target_for_unlabeled: how to handle the unlabeled data, can be:"use_0", "use_-1", "use_1", "ignore"
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        shortage_mode: behavior when anomaly count < required la
        - 'raise': raise error
        - 'ignore': return all available anomalies only
        """

        # set seed for reproducible results
        self.utils.set_seed(self.seed)
        if '_' in data_type:
            data_type = data_type.split('_', 1)[1]

        data = None
        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
            print("Testing on customized dataset...")
        else:
            if data_type == "video":
                real_ds_name = self.dataset.split(".")[0]  # for parameterized dataset

                if "DATA_DIR" in self.configs[data_type]:
                    data_path = (
                            self.wd / self.configs[data_type]["DATA_DIR"] / (self.dataset + self.configs[data_type]["END_WITH"])
                    )
                    data = np.load(data_path, allow_pickle=True)

                    X, y = data["X"], data["y"]

                ManagerClass = import_class(self.configs[data_type][real_ds_name]["MANAGER_CLASS"])
                ds_config = self.configs[data_type][real_ds_name]
                ds_config["working_dir"] = self.wd
                ds_config["seed"] = self.seed

                ds_param = self.dataset.split(".")[1:]  # .s32.mi3d
                _params = {}
                for param in ds_param:
                    if param[0] == 's' and param[1:].isdigit():
                        _params["num_segments"] = int(param[1:])
                    if param[0] == 'l' and param[1:].isdigit():
                        _params["limit"] = int(param[1:])
                    if param[0] == 'm' and param[1:].isalpha():
                        ds_config['pretrain_model'] = param[1:]

                data_manager = ManagerClass(ds_config)
                data = data_manager.load_data(**_params)

            else:
                real_ds_name = self.dataset

                if "DATA_DIR" in self.configs[data_type]:
                    data_path = (
                            self.wd / self.configs[data_type]["DATA_DIR"] / (self.dataset + self.configs[data_type]["END_WITH"])
                    )
                    data = np.load(data_path, allow_pickle=True)

                    X, y = data["X"], data["y"]
                    #for tabular_inexact
                    y_inst_gt = None
                    if "inexact" in data_type:
                        y_inst_gt = data["y_gt"]

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
                if y_inst_gt is not None:
                    n_samples = X.shape[1]
                    y_inst_gt = np.concatenate([y_inst_gt[i*n_samples:(i+1)*n_samples] for i in idx_duplicate])

                

            # if the dataset is too large, subsampling for considering the computational cost
            if len(y) > 10000:
                print(f"subsampling for dataset {self.dataset}...")
                self.utils.set_seed(self.seed)
                idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
                X = X[idx_sample]
                y = y[idx_sample]
                if y_inst_gt is not None:
                    n_samples = X.shape[1]
                    y_inst_gt = np.concatenate([y_inst_gt[i*n_samples:(i+1)*n_samples] for i in idx_sample])

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

#----------------------------------------------------------------------------------------------------------------------------------

            # whether to add different types of noise for testing the robustness of benchmark models
            # if noise_type is None:
            #     pass

            # elif noise_type == "duplicated_anomalies":
            #     X, y = self.add_duplicated_anomalies(X, y, duplicate_times=duplicate_times)


            # elif noise_type == "irrelevant_features":
            #     X, y = self.add_irrelevant_features(X, y, noise_ratio=noise_ratio)

            # elif noise_type == "label_contamination":
            #     X, y = self.add_label_contamination(X, y, noise_ratio=noise_ratio, flip_normal_ratio=flip_normal_ratio, flip_abnormal_ratio=flip_abnormal_ratio)

            # else:
            #     raise NotImplementedError

            # print(f"current noise type: {noise_type}")

            # show the statistic
            self.utils.data_description(X=X, y=y)
            
            # 原始包索引
            bag_indices = np.arange(X.shape[0]) #为tabular_inexact增加
            # spliting the current data to the training set and testing set   目前为包形式，增加索引记录
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, bag_indices,test_size=self.test_size, shuffle=True, stratify=y
            )
            y_test_gt_idx = None
            y_test_gt = None
            if X_test.ndim ==3:
                n_samples = X_test.shape[1]
                #获得X_test实例级样本索引
                y_test_gt_idx = torch.cat([torch.arange(i*n_samples, (i+1)*n_samples) for i in idx_test])
                y_test_gt = y_inst_gt[y_test_gt_idx]

            if noise_type is None:
                pass
            # we respectively generate the duplicated anomalies for the training and testing set
            # if noise_type == "duplicated_anomalies":
            #     X_train, y_train = self.add_duplicated_anomalies(X_train, y_train, duplicate_times=duplicate_times)
            #     X_test, y_test = self.add_duplicated_anomalies(X_test, y_test, duplicate_times=duplicate_times)
            # # whether to remove the anomaly contamination in the unlabeled data
            # elif noise_type == "anomaly_contamination":
            #     idx_unlabeled_anomaly = self.remove_anomaly_contamination(idx_unlabeled_anomaly, contam_ratio)
            #对训练集进行inaccurate setting处理
            # notice that label contamination can only be added in the training set
            elif noise_type == "label_contamination":
                X_train, y_train = self.add_label_contamination(X_train, y_train,flip_normal_ratio=flip_normal_ratio, flip_abnormal_ratio=flip_abnormal_ratio)

            # minmax scaling1
            if minmax:
                if X_train.ndim <= 2:
                    # 原逻辑
                    scaler = MinMaxScaler().fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)

                elif X_train.ndim == 3:
                    # 三维包形式 [n_bags, n_samples, n_features] → 展平到二维
                    n_bags, n_samples, n_features = X_train.shape

                    # 展平
                    X_train_flat = X_train.reshape(n_bags * n_samples, n_features)
                    X_test_flat = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])

                    # MinMaxScaler
                    scaler = MinMaxScaler().fit(X_train_flat)
                    X_train_scaled = scaler.transform(X_train_flat)
                    X_test_scaled = scaler.transform(X_test_flat)

                    # 恢复三维
                    X_train = X_train_scaled.reshape(n_bags, n_samples, n_features)
                    X_test = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if type(la) == float:
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
        elif type(la) == int:
            if la > len(idx_anomaly):
                if shortage_mode == "raise":
                    raise AssertionError(
                        f"the number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !"
                        f'Please set a smaller la or change the shortage_mode to "ignore" or "duplicate".'
                    )
                elif shortage_mode == "ignore":
                    idx_labeled_anomaly = idx_anomaly
                else:
                    raise NotImplementedError(f"shortage_mode {shortage_mode} is not implemented!")
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        else:
            raise NotImplementedError

        if type(ru) == float:
            idx_using_normal = np.random.choice(idx_normal, ceil(ru * len(idx_normal)), replace=False)
        elif type(ru) == int:
            if ru > len(idx_normal):
                if shortage_mode == "raise":
                    raise AssertionError(
                        f"the number of using unlabeled samples are greater than the total unlabeled samples: {len(idx_normal)} !"
                        f'Please set a smaller ru or change the shortage_mode to "ignore" or "duplicate".'
                    )
                elif shortage_mode == "ignore":
                    idx_using_normal = idx_normal
                else:
                    raise NotImplementedError(f"shortage_mode {shortage_mode} is not implemented!")
            else:
                idx_using_normal = np.random.choice(idx_normal, ru, replace=False)


        if type(eln) == float:
            if at_least_one_labeled:
                idx_labeled_normal = np.random.choice(idx_using_normal, ceil(eln * len(idx_labeled_anomaly)), replace=False)
            else:
                idx_labeled_normal = np.random.choice(idx_using_normal, int(eln * len(idx_labeled_anomaly)), replace=False)
        elif type(eln) == int:
            if eln > len(idx_using_normal):
                if shortage_mode == "raise":
                    raise AssertionError(
                        f"the number of labeled anomalies are greater than the total anomalies: {len(idx_normal)} !"
                        f'Please set a smaller la or change the shortage_mode to "ignore" or "duplicate".'
                    )
                elif shortage_mode == "ignore":
                    idx_labeled_normal = idx_using_normal
                else:
                    raise NotImplementedError(f"shortage_mode {shortage_mode} is not implemented!")
            else:
                idx_labeled_normal = np.random.choice(idx_using_normal, eln, replace=False)
        else:
            raise NotImplementedError
        

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        idx_unlabeled_normal = np.setdiff1d(idx_using_normal, idx_labeled_normal)
        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_unlabeled_normal, idx_unlabeled_anomaly)
        final_indices = np.concatenate([idx_labeled_anomaly, idx_labeled_normal, idx_unlabeled])

        # 根据 final_indices 重新取 X_train / y_train
        X_train = X_train[final_indices]
        y_train = y_train[final_indices]

        # 构建 mask 实现eln相关功能
        mask = np.ones_like(y_train, dtype=int)
        # unlabeled 样本 mask=0
        mask[np.isin(final_indices, idx_unlabeled)] = 0  
        # 保证 labeled anomaly 的标签是 1
        y_train[np.isin(final_indices, idx_labeled_anomaly)] = 1

        if target_for_unlabeled == "fill_unlabel_0":
            y_train[mask == 0] = 0  # 无标签样本的标签为0
        elif target_for_unlabeled == "delete_sample":  # 只使用有标签样本
            X_train = X_train[mask == 1]
            y_train = y_train[mask == 1]

        result = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_test_idx":idx_test,
            "y_test_gt_idx":y_test_gt_idx,
            "y_test_gt":y_test_gt,
            "mask": mask
        }

        if data is not None and isinstance(data, dict):
            data.update(result)
            result = data
        return result
