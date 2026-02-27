# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from WSADBench.myutils import Utils
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import os
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'


class TabPFN:
    def __init__(
        self,
        seed,
        model_name="TabPFN",
        batch_size=10000,  # 改参数(推理batch
    ):
        """

        Args:
            seed: random seed for reproducibility
            model_name: name of the model
            best_model_method: method to determine the best model, e.g., "min_train_loss
            epochs: number of training epochs
            batch_size: size of each training batch
            nb_batch: number of batches per epoch
            network_depth: depth of the network architecture (1, 2, or 4)
        """
        self.utils = Utils()
        self.device = self.utils.get_device(True)  # get device
        self.seed = seed

        self.clf = TabPFNClassifier(device='cuda', model_path="ckpt/tabpfn-v2.5-classifier-v2.5_default.ckpt",
                                    random_state=seed)  # Uses TabPFN 2.5 weights, finetuned on real data.
        self.batch_size = batch_size
        self.model_name = model_name
        self.use_batch = False
    def fit(self, X_train, y_train):
        # 设置训练集上限 ValueError: Number of samples 134400 in the input data is greater than the maximum number of samples 50000 officially supported by TabPFN. Set `ignore_pretraining_limits=True` to override this error!
        # 临时给一个rnd的seed
        max_limit = 50000
        if X_train.shape[0] > max_limit:
            rnd = np.random.RandomState(self.seed)
            idx = rnd.permutation(X_train.shape[0])[:max_limit]
            X_train = X_train[idx]
            y_train = y_train[idx]
            self.use_batch = True
            print("分批处理")
            # 同时对齐y_train


        self.clf.fit(X_train, y_train)
        print(f'train finished')



        return self

    def predict_score(self, X):
        if self.use_batch:
            with torch.no_grad():
                # 分batch预测
                scores = []
                batch_size = self.batch_size  # 使用训练时的batch_size

                for i in tqdm(range(0, len(X), batch_size)):
                    batch_end = min(i + batch_size, len(X))
                    X_batch = X[i:batch_end]
                    # X_tensor = torch.FloatTensor(X_batch).to(self.device)

                    batch_score = self.clf.predict_proba(X_batch)[:, 1]
                    scores.append(batch_score)

                # 合并所有batch的结果
                score = np.concatenate(scores, axis=0)
        else:
            with torch.no_grad():
                score = self.clf.predict_proba(X)[:, 1]  # 官方说一次性

        return score