import re

import numpy as np
import pandas as pd
import random
import pickle
import os
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.mixture import GaussianMixture
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE

from WSADBench.myutils import Utils, import_class
# from py_vis.vis_ood_scatter import visualize_embedding_distribution
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA


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
                # 1. 先构建完整的绝对路径对象
                data_dir_path = self.wd / ds_config["DATA_DIR"]

                # 2. 关键修复：检查路径是否存在，且确实是一个目录
                if data_dir_path.exists() and data_dir_path.is_dir():
                    all_dataset_list[dataset_kind] = [
                        f.stem for f in data_dir_path.iterdir()
                        if f.suffix in [".npz", ".npy", ".pkl"]
                    ]
                else:
                    # 3. 如果目录不存在，给一个空列表，防止报错，也不影响其他已存在的数据集运行
                    # 你可以在这里加个 print 提示，或者保持静默
                    all_dataset_list[dataset_kind] = []

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

    # 增加标签污染，inaccurate
    def add_label_contamination(self, X, y, flip_normal_ratio, flip_abnormal_ratio):

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

        print(
            f"Label contamination: flipped {len(flip_normals)} normal samples to anomalies and {len(flip_abnormals)} anomalies to normal.")

        return X, y_noisy

    def _load_ood_dataset(self, data_type, la, exp_note):
        real_ds_name = self.dataset.split(".")[0]  # for parameterized dataset
        if "DATA_DIR" in self.configs[data_type]:
            data_path = (
                    self.wd / self.configs[data_type]["DATA_DIR"] / (
                    real_ds_name + self.configs[data_type]["END_WITH"])
            )
        try:
            with open(data_path, "rb") as f:
                # X_train_dict = pickle.load(f)
                # X_test_dict = pickle.load(f)

                data = pickle.load(f)
                X_train_dict = data["X_train_dict"]
                X_test_dict = data["X_test_dict"]
                if 'pure_umap' in real_ds_name:
                    X_reduced_dict = data["X_reduced_dict"]['pure_umap']  # 写死
                elif 'pca' in real_ds_name:
                    X_reduced_dict = data["X_reduced_dict"]['pca']
                else:
                    X_reduced_dict = data["X_reduced_dict"]['umap']
        except Exception as e:
            print(f'{e}, load一次失败')
            try:
                with open(data_path, "rb") as f:
                    X_train_dict = pickle.load(f)
                    X_test_dict = pickle.load(f)

                    # data = pickle.load(f)
                    # X_train_dict = data["X_train_dict"]
                    # X_test_dict = data["X_test_dict"]
            except Exception as e:
                print(f'{e}, load2次失败')

                pass

        def reshape_ood_data(X_train_dict, X_test_dict, dataset, rla=1.0, exp_note=None, X_reduced_dict=None):
            """
            将 OOD 数据字典转换为训练集和测试集

            Args:
                X_train_dict: 训练集字典 {file_path: feature_array}
                X_test_dict: 测试集字典 {file_path: feature_array}
                dataset: 数据集名称，格式为 "class_name.known_class.rate"
                rla: labeled anomaly ratio（已标注异常的比例）

            Returns:
                dict: 包含 X_train, y_train, X_test, y_test, mask, y_train_cls, y_test_cls
            """
            # 解析数据集参数
            parts = dataset.split(".")
            class_name = parts[0]
            known_class = parts[1]
            rate = float(parts[2]) / 100  # 百分比

            seed = 0  # 固定种子
            rnd = random.Random(seed)

            normal_class = 'good'
            normal_list = []
            normal_keys = []  # 保存normal样本的key
            abnormal_dict = {}
            abnormal_keys_dict = {}  # 保存abnormal样本的key
            abnormal_num = 0

            # 分离正常和异常样本
            for key, val in X_train_dict.items():
                normal_list.append(val)
                normal_keys.append(key)

            for key, val in X_test_dict.items():
                data_class = key.split('/')[-2]
                if data_class == normal_class:
                    normal_list.append(val)
                    normal_keys.append(key)
                else:
                    if data_class not in abnormal_dict:
                        abnormal_dict[data_class] = []
                        abnormal_keys_dict[data_class] = []
                    abnormal_dict[data_class].append(val)
                    abnormal_keys_dict[data_class].append(key)
                    abnormal_num += 1

            # 在这里通过排序得出最近的簇作为簇心
            def find_abnormal_cluster(X_reduced_dict, normal_keys, abnormal_keys_dict, top_k=1, is_ascend=None):
                """
                找出离正常簇中心最近的异常类别簇中心
                """
                # 计算正常簇中心
                normal_center = np.mean([X_reduced_dict[key] for key in normal_keys], axis=0)
                # 这里的异常类别需要合规
                ds_name = list(X_reduced_dict.keys())[0].split('/')[-4]
                class_dict = {'aitex': ['broken_pick', 'fuzzyball', 'weft_crack', ],
                              'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
                              'elpv': ['mono', ],
                              'hyperkvasir': ['barretts', 'barretts-short-segment', 'esophagitis-b-d', ],
                              'mastcam': ['broken-rock', 'drill-hole', 'drt', 'dump-pile', 'float', 'meteorite',
                                          'veins'],
                              'metal_nut': ['bent', 'color', 'flip', 'scratch']}
                ab_key_list = class_dict[ds_name]
                # 计算每个异常类别的簇中心
                ab_center_dict = {}
                for key in ab_key_list:
                    ab_keys = abnormal_keys_dict[key]
                    if ab_keys:  # 确保列表不为空
                        ab_center_dict[key] = np.mean([X_reduced_dict[ab_name] for ab_name in ab_keys],
                                                      axis=0)

                # 计算每个异常簇中心到正常簇中心的距离
                distances = []
                for ab_type, ab_center in ab_center_dict.items():
                    distance = np.linalg.norm(ab_center - normal_center)
                    distances.append((ab_type, distance, ab_center))

                # 按距离排序，找出最近的异常类别
                if is_ascend:
                    distances.sort(key=lambda x: x[1])
                else:
                    distances.sort(key=lambda x: x[1], reverse=True)

                # 返回前top_k个最近的异常类别
                closest_clusters = [(ab_type, dist) for ab_type, dist, _ in distances[:top_k]]
                # print(f"正常簇中心与各异常簇中心的距离:")
                # for i, (ab_type, dist, center) in enumerate(distances):
                #     print(f"  {ab_type}: {dist:.4f}")

                # print(f"\n最近的{top_k}个异常类别:")
                # for ab_type, dist in closest_clusters:
                #     print(f"  {ab_type}: 距离 = {dist:.4f}")

                return closest_clusters[0][0]

            if exp_note in ['rla_emb_near_inc', 'rla_emb_know_near_inc']:
                known_class = find_abnormal_cluster(X_reduced_dict, normal_keys, abnormal_keys_dict, is_ascend=True)
            elif exp_note in ['rla_emb_know_far_inc']:
                known_class = find_abnormal_cluster(X_reduced_dict, normal_keys, abnormal_keys_dict, is_ascend=False)

            # 计算训练集和测试集的异常样本数量
            if known_class not in abnormal_dict:
                raise ValueError(
                    f"Known class '{known_class}' not found in abnormal classes: {list(abnormal_dict.keys())}")

            n_known_total = len(abnormal_dict[known_class])
            max_ab_train = int(n_known_total * 0.7)  # 训练边界：用于训练的已知类异常上限
            n_ab_test = n_known_total - max_ab_train  # 测试集中来自已知类的异常总数（不受 rla/nla 影响）
            if 'rla' in exp_note:  # 按比例选择“标注的异常数”
                n_ab_train = max(1, int(rla * max_ab_train))
            elif 'nla' in exp_note:  # 按固定个数选择“标注的异常数”
                n_ab_train = int(rla)
            else:
                raise NotImplementedError

            # 计算测试集中已知和未知异常的比例
            n_unseen = int(n_ab_test * rate)
            n_seen = n_ab_test - n_unseen

            # 构建未知异常列表
            unseen_list = []
            unseen_keys = []

            # 同步shuffle已知异常的特征和key
            combined_known = list(zip(abnormal_dict[known_class], abnormal_keys_dict[known_class]))
            rnd.shuffle(combined_known)
            abnormal_dict[known_class], abnormal_keys_dict[known_class] = zip(*combined_known)
            abnormal_dict[known_class] = list(abnormal_dict[known_class])
            abnormal_keys_dict[known_class] = list(abnormal_keys_dict[known_class])

            for key in abnormal_dict:
                if key != known_class:
                    unseen_list += abnormal_dict[key]
                    unseen_keys += abnormal_keys_dict[key]

            # 同步shuffle未知异常
            combined_unseen = list(zip(unseen_list, unseen_keys))
            rnd.shuffle(combined_unseen)
            unseen_list, unseen_keys = zip(*combined_unseen) if combined_unseen else ([], [])
            unseen_list, unseen_keys = list(unseen_list), list(unseen_keys)

            # 同步shuffle正常样本
            combined_normal = list(zip(normal_list, normal_keys))
            rnd.shuffle(combined_normal)
            normal_list, normal_keys = zip(*combined_normal)
            normal_list, normal_keys = list(normal_list), list(normal_keys)

            # 构建训练集：正常 + 已知类异常（训练边界内全量 = 标注 + 未标注）
            n_normal_train = int(len(normal_list) * 0.7)
            # 处理rate
            X_train = normal_list[:n_normal_train] + abnormal_dict[known_class][:max_ab_train]
            y_train_cls = normal_keys[:n_normal_train] + abnormal_keys_dict[known_class][:max_ab_train]

            # 构建测试集
            n_normal_test = len(normal_list) - n_normal_train

            if 'emb' in exp_note:  # 需要按距离取
                # 用训练边界内的已知类异常（含标注与未标注）求簇心
                abnormal_reduced_list = [X_reduced_dict[key] for key in
                                         abnormal_keys_dict[known_class][:max_ab_train]]
                abnormal_reduced_arr = np.stack(abnormal_reduced_list, axis=0)

                centroid = abnormal_reduced_arr.mean(axis=0)
                dists = np.sum((abnormal_reduced_arr - centroid) ** 2, axis=1)
                center_idx = int(np.argmin(dists))
                ab_core = abnormal_reduced_arr[center_idx]

                # 已知类剩余异常（测试候选：训练边界外）
                known_remain_vals = abnormal_dict[known_class][max_ab_train:]
                known_remain_keys = abnormal_keys_dict[known_class][max_ab_train:]
                kr_pairs = [(k, v) for k, v in zip(known_remain_keys, known_remain_vals) if k in X_reduced_dict]

                if kr_pairs and n_seen > 0:
                    kr_keys = [k for k, _ in kr_pairs]
                    kr_vals = [v for _, v in kr_pairs]
                    kr_emb = np.stack([X_reduced_dict[k] for k in kr_keys], axis=0)
                    kr_d = np.sum((kr_emb - ab_core) ** 2, axis=1)
                    k = min(n_seen, kr_d.shape[0])
                    if k > 0:
                        if exp_note in ['rla_emb_near_inc', 'rla_emb_kwon_near_inc']:
                            sel_idx = np.argpartition(kr_d, kth=k - 1)[:k]  # 取最近 k 个
                        else:
                            sel_idx = np.argpartition(-kr_d, kth=k - 1)[:k]  # 取最远 k 个
                        sel_idx = sel_idx[np.argsort(kr_d[sel_idx])]  # 对已选 k 个按距离降序
                        sel_known_vals = [kr_vals[i] for i in sel_idx]
                        sel_known_keys = [kr_keys[i] for i in sel_idx]
                    else:
                        sel_known_vals, sel_known_keys = [], []
                else:
                    sel_known_vals, sel_known_keys = [], []

                # 未知类异常（候选：其它类），下方分支按原逻辑
                # if exp_note in ['nla_emb_know_far_inc', 'rla_emb_know_near_inc', 'rla_emb_near_inc']:
                ur_pairs = [(k, v) for k, v in zip(unseen_keys, unseen_list) if k in X_reduced_dict]
                if ur_pairs:
                    ur_keys = [k for k, _ in ur_pairs]
                    ur_vals = [v for _, v in ur_pairs]
                    ur_emb = np.stack([X_reduced_dict[k] for k in ur_keys], axis=0)
                    ur_d = np.sum((ur_emb - ab_core) ** 2, axis=1)
                    u = min(n_unseen, ur_d.shape[0])
                    if u > 0:
                        if exp_note in ['rla_emb_near_inc']:
                            # 取最近 u 个
                            sel_idx_u = np.argpartition(ur_d, kth=u - 1)[:u]
                        else:
                            sel_idx_u = np.argpartition(-ur_d, kth=u - 1)[:u]
                        sel_idx_u = sel_idx_u[np.argsort(ur_d[sel_idx_u])]
                        sel_unseen_vals = [ur_vals[i] for i in sel_idx_u]
                        sel_unseen_keys = [ur_keys[i] for i in sel_idx_u]
                    else:
                        sel_unseen_vals, sel_unseen_keys = [], []
                else:
                    sel_unseen_vals, sel_unseen_keys = [], []

                # 组装测试集
                X_test = normal_list[n_normal_train:] + sel_known_vals + sel_unseen_vals
                y_test_cls = normal_keys[n_normal_train:] + sel_known_keys + sel_unseen_keys

                # 补一个umap的降维可视化结果，要求对整个数据集降维（从原始维度），然后标出X_test_seen, X_test_unseen，X_train_ab,X_train_ab_core.用点的形状区分
                # 再用颜色区分实际的类别，比如good，'broken_pick','fuzzyball','weft_crack'
                # 在选取簇中心后，补充核心样本 key
                show_emb = False
                if show_emb and rate == 0.5:  # 只看对半
                    save_path = self.wd
                    # 用pickle存一下中间数据，方便重载
                    data_res = {
                        'X_reduced_dict': X_reduced_dict,
                        'normal_keys': normal_keys,
                        'abnormal_keys_dict': abnormal_keys_dict,
                        'sel_known_keys': sel_known_keys,
                        'sel_unseen_keys': sel_unseen_keys,
                        'known_class': known_class,
                        'class_name': class_name,
                        'save_root_dir': save_path,
                        'rate': rate,
                        'max_ab_train': max_ab_train,
                        'n_ab_train': n_ab_train,
                        'center_idx': center_idx,
                        'exp_note': exp_note
                    }
                    with open(os.path.join(save_path, f'myRes/{real_ds_name}_{exp_note}.pkl'), 'wb') as f:
                        pickle.dump(data_res, f)
                        print(f'============myRes/{real_ds_name}_{exp_note}.pkl saved======')
                    # visualize_embedding_distribution(
                    #     X_reduced_dict=X_reduced_dict,  # 你的特征字典
                    #     normal_keys=normal_keys,  # 正常样本Key列表
                    #     abnormal_keys_dict=abnormal_keys_dict,  # 异常样本字典
                    #     sel_known_keys=sel_known_keys,  # 选中的已知异常Key
                    #     sel_unseen_keys=sel_unseen_keys,  # 选中的未知异常Key
                    #     known_class=known_class,  # 当前已知异常类名 (str)
                    #     class_name=class_name,  # 总类名 (str)
                    #     save_root_dir=save_path,  # 保存根路径
                    #     rate=rate,  # 异常率 (float)
                    #     max_ab_train=max_ab_train,  # (int)
                    #     n_ab_train=n_ab_train,  # (int)
                    #     center_idx=center_idx,  # (int)
                    #     exp_note = exp_note
                    # )



            else:
                # 非 emb：按训练边界拆分
                X_test = (
                        normal_list[n_normal_train:] +
                        abnormal_dict[known_class][max_ab_train:max_ab_train + n_seen] +
                        unseen_list[:n_unseen]
                )
                y_test_cls = (
                        normal_keys[n_normal_train:] +
                        abnormal_keys_dict[known_class][max_ab_train:max_ab_train + n_seen] +
                        unseen_keys[:n_unseen]
                )  # 生成标签

            # 生成标签
            y_train = np.zeros(len(X_train), dtype=int)
            y_train[n_normal_train:n_normal_train + n_ab_train] = 1  # 中间为1
            if '_no_' in exp_note:
                # 正则：_no_0_la_0
                res = re.match(r'.*?_no_(\d+)_la_(\d+)', exp_note)
                no_rate = int(res.group(1)) / 100.0
                la_rate = int(res.group(2)) / 100.0
                print(f'exp_note: {exp_note}, before: {len(X_train)}, no: {no_rate}, la: {la_rate}')
                # 对X_train、y_train、y_train_cls进行筛选
                X_train = (X_train[:n_normal_train] + X_train[
                    n_normal_train:n_normal_train + max(int(n_ab_train * la_rate), 1)] +
                           X_train[n_normal_train + n_ab_train:n_normal_train + n_ab_train + max(
                               int((max_ab_train - n_ab_train) * no_rate), 1)])
                selected_y = [
                    y_train[:n_normal_train],
                    y_train[n_normal_train: n_normal_train + max(int(n_ab_train * la_rate), 1)],
                    y_train[n_normal_train + n_ab_train: n_normal_train + n_ab_train + max(int(
                        (max_ab_train - n_ab_train) * no_rate), 1)]
                ]
                y_train = np.concatenate(selected_y, axis=0)
                y_train_cls = (y_train_cls[:n_normal_train] + y_train_cls[
                    n_normal_train:n_normal_train + max(int(n_ab_train * la_rate), 1)] +
                               y_train_cls[n_normal_train + n_ab_train:n_normal_train + n_ab_train + max(
                                   int((max_ab_train - n_ab_train) * no_rate), 1)])
                print(f'exp_note: {exp_note}, after: {len(X_train)}')

            y_test = np.ones(len(X_test), dtype=int)
            # n_normal_test = len(normal_list) - n_normal_train
            y_test[:n_normal_test] = 0
            # 打印训练集上：总数，正常和异常分别的总数，类别标签的名称和占比
            # 打印测试集上：总数，正常和异常分别的总数，类别标签的名称和占比
            # 在生成标签后添加统计打印代码
            # 生成 mask（标记已标注样本）
            # 生成 mask（仅“已标注异常”为 1；正常与“未标注异常”为 0）
            mask = y_train.copy()
            # 转换为 numpy 数组
            X_train = np.array(X_train)
            X_test = np.array(X_test)

            # 统计训练集信息
            train_normal_count = np.sum((y_train == 0) & (mask == 0))
            train_anomaly_count = np.sum((y_train == 1) & (mask == 1))
            # train_inc_count = np.sum((y_train==1) & (mask==0))
            # assert train_normal_count + train_anomaly_count + train_inc_count == len(y_train)
            train_total = len(y_train)

            # 统计训练集各类别的数量和占比
            train_cls_counter = {}
            for cls_key in y_train_cls:
                cls_name = cls_key.split('/')[-2]
                train_cls_counter[cls_name] = train_cls_counter.get(cls_name, 0) + 1

            print("=" * 60)
            print(f"Training Set Statistics for {class_name}:")
            print(f"  Total samples: {train_total}")
            print(f"  Normal samples: {train_normal_count} ({train_normal_count / train_total * 100:.2f}%)")
            print(f"  Anomaly samples: {train_anomaly_count} ({train_anomaly_count / train_total * 100:.2f}%)")
            # print(f'  Noisy anomaly samples: {train_inc_count} ({train_inc_count / train_total * 100:.2f}%)')
            print(f"  Class distribution:")
            for cls_name, count in sorted(train_cls_counter.items()):
                print(f"    {cls_name}: {count} ({count / train_total * 100:.2f}%)")

            # 统计测试集信息
            test_normal_count = np.sum(y_test == 0)
            test_anomaly_count = np.sum(y_test == 1)
            test_total = len(y_test)

            # 统计测试集各类别的数量和占比
            test_cls_counter = {}
            for cls_key in y_test_cls:
                cls_name = cls_key.split('/')[-2]
                test_cls_counter[cls_name] = test_cls_counter.get(cls_name, 0) + 1

            print(f"\nTest Set Statistics for {class_name}:")
            print(f"  Total samples: {test_total}")
            print(f"  Normal samples: {test_normal_count} ({test_normal_count / test_total * 100:.2f}%)")
            print(f"  Anomaly samples: {test_anomaly_count} ({test_anomaly_count / test_total * 100:.2f}%)")
            print(f"  Class distribution:")
            for cls_name, count in sorted(test_cls_counter.items()):
                print(f"    {cls_name}: {count} ({count / test_total * 100:.2f}%)")
            print("=" * 60)

            # 构建X_train_dict和X_test_dict
            X_train_dict = {y_train_cls[i]: X_train[i] for i in range(len(X_train))}
            X_test_dict = {y_test_cls[i]: X_test[i] for i in range(len(X_test))}

            # 数据增强：样本数 < 1000 时复制，> 10000 时采样
            if len(X_train) < 1000:
                print(f"Generating duplicate samples for dataset {class_name}...")
                idx_duplicate = np.random.choice(np.arange(len(X_train)), 1000, replace=True)
                X_train = X_train[idx_duplicate]
                y_train = y_train[idx_duplicate]
                y_train_cls = [y_train_cls[i] for i in idx_duplicate]
                mask = mask[idx_duplicate]

            if len(X_train) > 10000:
                print(f"Subsampling for dataset {class_name}...")
                idx_sample = np.random.choice(np.arange(len(X_train)), 10000, replace=False)
                X_train = X_train[idx_sample]
                y_train = y_train[idx_sample]
                y_train_cls = [y_train_cls[i] for i in idx_sample]
                mask = mask[idx_sample]

            result = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'mask': mask,
                'emb_vis': {
                    'X_train_dict': X_train_dict,
                    'X_test_dict': X_test_dict,
                    'dataset': real_ds_name,
                }
            }

            print(f'Dataset: {class_name}, Known class: {known_class}, Rate: {rate}')
            print(f'Train samples: {len(X_train)}, Test samples: {len(X_test)}')
            print(f'Train anomalies: {np.sum(y_train)}, Test anomalies: {np.sum(y_test)}')

            return result

        # 调用5次rate,只有X_test不同，y_test相同
        rate_list = [0, 25, 50, 75, 100]  # [0, 25, 50, 75, 100]
        data_list = []
        for rate in rate_list:
            dataset = f'{self.dataset}.{rate}'
            data = reshape_ood_data(X_train_dict, X_test_dict, dataset, rla=la, exp_note=exp_note,
                                    X_reduced_dict=X_reduced_dict)
            data_list.append(data)
        # 保存数据成npy

        data["X_test_dict"] = {rate: d['X_test'] for rate, d in zip(rate_list, data_list)}
        return data
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
            noise_type=None,
            flip_normal_ratio: float = 0.0,
            flip_abnormal_ratio: float = 0.0,
            duplicate_times: int = 2,
            contam_ratio=1.00,
            noise_ratio: float = 0.05,
            shortage_mode="ignore",
            data_type=None,
            exp_note = None,
    ):
        # set seed for reproducible results
        self.utils.set_seed(self.seed)
        if '_' in data_type and 'inexact' not in data_type:
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
                            self.wd / self.configs[data_type]["DATA_DIR"] / (
                                self.dataset + self.configs[data_type]["END_WITH"])
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
                return data
            if "OOD" in data_type:  # OOD数据
                data = self._load_ood_dataset(data_type,la, exp_note)
                return data

            else:    # 感觉是下面这段不一致
                real_ds_name = self.dataset

                if "DATA_DIR" in self.configs[data_type]:
                    data_path = (
                            self.wd / self.configs[data_type]["DATA_DIR"] / (
                            self.dataset + self.configs[data_type]["END_WITH"])
                    )
                    data = np.load(data_path, allow_pickle=True)

                    X, y = data["X"], data["y"]  # 这里加载数据
                    # for tabular_inexact
                    y_inst_gt = None
                    if "inexact" in data_type:
                        y_inst_gt = data["y_gt"]

                if data is None:
                    raise NotImplementedError(f"Dataset {self.dataset} not found in the available datasets.")

            # 这里是tab inexcat
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
                        y_inst_gt = np.concatenate(
                            [y_inst_gt[i * n_samples:(i + 1) * n_samples] for i in idx_duplicate])

                # if the dataset is too large, subsampling for considering the computational cost
                if len(y) > 10000:
                    print(f"subsampling for dataset {self.dataset}...")
                    self.utils.set_seed(self.seed)
                    idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
                    X = X[idx_sample]
                    y = y[idx_sample]
                    if y_inst_gt is not None:
                        n_samples = X.shape[1]
                        y_inst_gt = np.concatenate(
                            [y_inst_gt[i * n_samples:(i + 1) * n_samples] for i in idx_sample])

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
                                X, y, realistic_synthetic_mode=realistic_synthetic_mode, alpha=alpha,
                                percentage=percentage
                            )
                            np.savez_compressed(os.path.join(filepath, filename), X=X, y=y)
                            pass

                    else:
                        X, y = self.generate_realistic_synthetic(
                            X, y, realistic_synthetic_mode=realistic_synthetic_mode, alpha=alpha,
                            percentage=percentage
                        )

                # show the statistic
                self.utils.data_description(X=X, y=y)

                # 原始包索引
                bag_indices = np.arange(X.shape[0])  # 为tabular_inexact增加
                # spliting the current data to the training set and testing set   目前为包形式，增加索引记录
                X_train, X_test, y_train, y_test, bag_idx_train, bag_idx_test = train_test_split(  # 7比3划分
                    X, y, bag_indices, test_size=self.test_size, shuffle=True, stratify=y
                )

                y_test_gt_idx = None
                y_test_gt = None
                if X_test.ndim == 3:
                    n_samples = X_test.shape[1]
                    # 获得X_test实例级样本索引
                    y_test_gt_idx = torch.cat(
                        [torch.arange(i * n_samples, (i + 1) * n_samples) for i in bag_idx_test])
                    y_test_gt = y_inst_gt[y_test_gt_idx]

                if noise_type is None:
                    pass
                elif noise_type == "label_contamination":
                    X_train, y_train = self.add_label_contamination(X_train, y_train,
                                                                    flip_normal_ratio=flip_normal_ratio,
                                                                    flip_abnormal_ratio=flip_abnormal_ratio)

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

            if eln == 0.0:
                # idx of normal samples and unlabeled/labeled anomalies
                idx_normal = np.where(y_train == 0)[0]
                idx_anomaly = np.where(y_train == 1)[0]

                if type(la) == float:
                    if at_least_one_labeled:
                        idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)),
                                                               replace=False)
                    else:
                        idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)),
                                                               replace=False)
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

                idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
                # ---- 分层采样逻辑 ----
                n_normal = len(idx_normal)
                n_unlabeled_anom = len(idx_unlabeled_anomaly)
                total_unlabeled = n_normal + n_unlabeled_anom

                # 计算每类比例
                p_normal = n_normal / total_unlabeled
                p_unlabeled_anom = n_unlabeled_anom / total_unlabeled

                # 对无标签的异常样本和正常样本进行分层采用；控制unlabeled数据使用比例
                if type(ru) == float:
                    idx_using_normal = np.random.choice(idx_normal, ceil(ru * len(idx_normal)), replace=False)
                    idx_using_unlabeled_anom = np.random.choice(idx_unlabeled_anomaly,
                                                                ceil(ru * len(idx_unlabeled_anomaly)),
                                                                replace=False)

                elif type(ru) == int:
                    if ru > total_unlabeled:
                        if shortage_mode == "raise":
                            raise AssertionError(
                                f"the number of using unlabeled samples are greater than the total unlabeled samples: {len(idx_normal)} !"
                                f'Please set a smaller ru or change the shortage_mode to "ignore" or "duplicate".'
                            )
                        elif shortage_mode == "ignore":
                            idx_using_unlabeled_anom = idx_unlabeled_anomaly
                            idx_using_normal = idx_normal
                        else:
                            raise NotImplementedError(f"shortage_mode {shortage_mode} is not implemented!")
                    else:
                        idx_using_unlabeled_anom = np.random.choice(idx_unlabeled_anomaly, ceil(
                            min(p_unlabeled_anom * ru, len(idx_unlabeled_anomaly))), replace=False)
                        idx_using_normal = np.random.choice(idx_normal, ru - len(idx_using_unlabeled_anom),
                                                            replace=False)

                idx_using_unlabeled = np.append(idx_using_normal, idx_using_unlabeled_anom)
                final_indices = np.concatenate([idx_labeled_anomaly, idx_using_unlabeled])

            else:
                # idx of normal samples and unlabeled/labeled anomalies
                idx_normal = np.where(y_train == 0)[0]
                idx_anomaly = np.where(y_train == 1)[0]

                if type(la) == float:
                    if at_least_one_labeled:
                        idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)),
                                                               replace=False)
                    else:
                        idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)),
                                                               replace=False)
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

                if type(eln) == float:
                    if at_least_one_labeled:
                        if len(idx_normal) < ceil(eln * len(idx_normal)):
                            print(
                                f"[Warning] normal number of samples lack ({len(idx_normal)} < {ceil(eln * len(idx_normal))},using resample.")
                            shortage = ceil(eln * len(idx_normal)) - len(idx_normal)
                            extra_idx_normal = np.random.choice(idx_normal, shortage, replace=True)
                            idx_labeled_normal = np.append(idx_normal, extra_idx_normal)
                        else:
                            idx_labeled_normal = np.random.choice(idx_normal, ceil(eln * len(idx_normal)),
                                                                  replace=False)
                    else:
                        if len(idx_normal) < int(eln * len(idx_normal)):
                            print(
                                f"[Warning] normal number of samples lack ({len(idx_normal)} < {int(eln * len(idx_normal))},using resample.。")
                            shortage = int(eln * len(idx_normal)) - len(idx_normal)
                            extra_idx_normal = np.random.choice(idx_normal, shortage, replace=True)
                            idx_labeled_normal = np.append(idx_normal, extra_idx_normal)
                        else:
                            idx_labeled_normal = np.random.choice(idx_normal, int(eln * len(idx_normal)),
                                                                  replace=False)

                elif type(eln) == int:
                    if eln > len(idx_normal):
                        if shortage_mode == "raise":
                            raise AssertionError(
                                f"the number of labeled anomalies are greater than the total anomalies: {len(idx_normal)} !"
                                f'Please set a smaller la or change the shortage_mode to "ignore" or "duplicate".'
                            )
                        elif shortage_mode == "ignore":
                            shortage = eln - len(idx_normal)
                            extra_idx_normal = np.random.choice(idx_normal, shortage, replace=True)
                            idx_labeled_normal = np.append(idx_normal, extra_idx_normal)
                        else:
                            raise NotImplementedError(f"shortage_mode {shortage_mode} is not implemented!")
                    else:
                        idx_labeled_normal = np.random.choice(idx_normal, eln, replace=False)
                else:
                    raise NotImplementedError

                idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
                idx_unlabeled_normal = np.setdiff1d(idx_normal, idx_labeled_normal)
                idx_using_unlabeled = np.append(idx_unlabeled_anomaly, idx_unlabeled_normal)
                final_indices = np.concatenate([idx_labeled_anomaly, idx_labeled_normal, idx_using_unlabeled])

            # 根据 final_indices 重新取 X_train / y_train
            X_train = X_train[final_indices]
            y_train = y_train[final_indices]

            # 构建 mask 实现eln相关功能
            mask = np.ones_like(y_train, dtype=int)
            # unlabeled 样本 mask=0
            mask[np.isin(final_indices, idx_using_unlabeled)] = 0
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
            "y_test_idx": bag_idx_test,
            "y_test_gt_idx": y_test_gt_idx,
            "y_test_gt": y_test_gt,
            "mask": mask,
            "bag_info_train":bag_idx_train,
            "bag_info_test":bag_idx_test,
        }
        if X_train.ndim ==3:
            result["NUM_FRAMES"] = X_train.shape[1]  # 这里是tabular_inexact的每个包内样本数，为了减少代码改动，沿用这个名字,tabular_inexact专用

        if data is not None and isinstance(data, dict):
            data.update(result)
            result = data
        return result