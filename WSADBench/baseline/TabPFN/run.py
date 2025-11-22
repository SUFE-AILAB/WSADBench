# -*- coding: utf-8 -*-
import numpy as np
import torch
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
        batch_size=1024,  # 改参数(推理batch
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

        self.clf = TabPFNClassifier(device='cuda', model_path="myRes/ckpt/tabpfn-v2.5-classifier-v2.5_default.ckpt",
                                    random_state=seed)  # Uses TabPFN 2.5 weights, finetuned on real data.
        self.batch_size = batch_size
        self.model_name = model_name
    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        print(f'train finished')



        return self

    def predict_score(self, X):
        # with torch.no_grad():
        #     # 分batch预测
        #     scores = []
        #     batch_size = self.batch_size  # 使用训练时的batch_size
        #
        #     for i in tqdm(range(0, len(X), batch_size)):
        #         batch_end = min(i + batch_size, len(X))
        #         X_batch = X[i:batch_end]
        #         # X_tensor = torch.FloatTensor(X_batch).to(self.device)
        #
        #         batch_score = self.clf.predict_proba(X_batch)[:, 1]
        #         scores.append(batch_score)
        #
        #     # 合并所有batch的结果
        #     score = np.concatenate(scores, axis=0)

        with torch.no_grad():
            score = self.clf.predict_proba(X)[:, 1]  # 官方说一次性

        return score