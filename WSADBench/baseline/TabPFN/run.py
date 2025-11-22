# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
import sys
import os
import copy

from tqdm import tqdm

from WSADBench.myutils import Utils
from typing import Literal
# from common_utils.ood_dataset.train import show_emb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import os
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'


class TabPFN:
    def __init__(
        self,
        seed,
        model_name="TabPFN",
        best_model_method:Literal["min_train_loss","last_epoch"]="min_train_loss",
        batch_size=1024,  # 改参数(推理batch
        loss_name:Literal["Deviation", "BCE"]="Deviation",
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
        self.MAX_INT = np.iinfo(np.int32).max
        self.best_model_method = best_model_method
        self.loss_name = loss_name

        self.batch_size = batch_size
        self.model_name = model_name

        # Initialize model and loss
        self.model = None
        self.best_model = None  # Keep best model in memory
        self.criterion = None
        self.input_dim = None
        self.is_frozen = True




    # 这里的X_train可以是图片本身（比较规则？并不规则，需要transform）
    def get_emb(self, X_dict):
        """高效提取特征embedding"""
        self.model.eval()
        with torch.no_grad():
            # 预先转换为列表，避免重复操作
            batch_size = self.batch_size
            keys = list(X_dict.keys())
            X_values = np.array(list(X_dict.values()))  # 直接转为numpy数组

            emb_list = []
            total_batches = (len(X_values) + batch_size - 1) // batch_size

            for i in tqdm(range(0, len(X_values), batch_size), total=total_batches,
                          desc="Extracting embeddings"):
                batch_end = min(i + batch_size, len(X_values))
                X_batch = X_values[i:batch_end]

                # 直接在GPU上操作
                x = torch.from_numpy(X_batch).float().to(self.device, non_blocking=True)
                x = self.model.feature_extractor(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)

                # 累积在CPU上，减少GPU-CPU传输次数
                emb_list.append(x.cpu().numpy())

            # 一次性拼接所有batch
            emb_array = np.concatenate(emb_list, axis=0)

            # 使用字典推导式构建结果
            X_all = {keys[i]: emb_array[i] for i in range(len(keys))}

            return X_all
    def fit(self, X_train, y_train): # X_test_tmp

        # Initialize a classifier
        self.clf = TabPFNClassifier(device='cuda', model_path="myRes/ckpt/tabpfn-v2.5-classifier-v2.5_default.ckpt")  # Uses TabPFN 2.5 weights, finetuned on real data.
        self.clf.fit(X_train, y_train)
        print(f'train finished')
        # X_test = X_test_tmp[0]
        # y_test = X_test_tmp[1]
        # # Predict probabilities
        # prediction_probabilities = self.clf.predict_proba(X_test)
        # print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
        # print("AUC PR:", average_precision_score(y_test, prediction_probabilities[:, 1]))
        # # Predict labels
        # predictions = self.clf.predict(X_test)
        # print("Accuracy", accuracy_score(y_test, predictions))



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