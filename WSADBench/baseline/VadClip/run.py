# -*- coding: utf-8 -*-
"""
基于"VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection"论文实现
Multiple Instance Learning (MIL) 弱监督异常检测方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from WSADBench.myutils import Utils
from WSADBench.baseline.VadClip.model import CLIPVAD, get_batch_mask
from WSADBench.baseline.VadClip.fit import fit as fit_main, get_prompt_text
from common_utils.baseline_utils import VideoDataset_VadClip as VideoDataset, fit_VadClip
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
class VadClip:
    def __init__(
        self,
        seed: int = 42,
        # 模型参数
        # 训练参数
        epochs: int = 75,
        batch_size: int = 30,
        learning_rate: float = 0.001,
        # 其他参数
        segments_per_video: int = 32,
        use_scheduler: bool = True,
        scheduler_milestones: List[int] = None,
        verbose: bool = True,
        scheduler_gamma=0.1,
        visual_length = None,
        pt_path = None,
        lam = None,
        is_test=None,
    ):
        """
        初始化Sultani模型

        Args:
            seed: 随机种子
            input_dim: 输入特征维度（默认2048对应ResNet特征）
            dropout: Dropout概率
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            sparsity_weight: 稀疏性损失权重
            smoothness_weight: 平滑性损失f权重
            segments_per_video: 每个视频的段数
            use_scheduler: 是否使用学习率调度器
            scheduler_milestones: 学习率调度里程碑
            verbose: 是否打印详细信息
        """

        self.seed = seed
        # self.input_dim = input_dim  # 2048？
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.segments_per_video = segments_per_video
        self.use_scheduler = use_scheduler
        self.scheduler_milestones = scheduler_milestones
        self.verbose = verbose
        self.scheduler_gamma = scheduler_gamma

        self.visual_length = visual_length
        self.pt_path = pt_path
        self.lam = lam
        self.label_map = dict()
        self.is_test = is_test
        # 内部状态
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.fitted = False
        self.training_history = None

        # 工具类
        self.utils = Utils()

        # 设置随机种子
        self.utils.set_seed(seed)
        self.device = self.utils.get_device(True)

    def _init_model(self):
        """初始化模型"""
        if self.model is None:
            # 创建模型
            self.model = CLIPVAD(input_dim=self.input_dim, device=self.device, visual_length = self.visual_length, pt_path=self.pt_path, num_class=self.num_class, lam=self.lam).to(self.device)

            # 创建优化器
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.learning_rate)

            # 创建学习率调度器
            if self.use_scheduler:
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_milestones, gamma=self.scheduler_gamma)

            if self.verbose:
                print(f"VadClip model initialization completed")
                print(f"Device: {self.device}")
                print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters()):,}")


    def fit(self, X, y, vid_info=None, vid_source_clips_num=None, vid_kind=None, X_test=None,
            y_test=None, crops_num=None, X_test_extra=None):
        """
        训练Sultani模型

        Args:
            X: 训练特征 [n_samples, feature_dim]
            y: 训练标签 [n_samples] (0: normal, 1: anomaly)
            X_test: 测试特征（可选）
            y_test: 测试标签（可选）

        Returns:
            self
        """
        start_time = time.time()

        if self.verbose:
            print("=" * 60)
            print("Starting VadClip model training")
            print("=" * 60)
            print(f"Number of training samples: {len(X)}")
            print(f"Normal samples: {np.sum(y == 0)}, Anomalous samples: {np.sum(y == 1)}")

        # 数据预处理
        self.input_dim = X.shape[-1]
        # 取出标签
        """
        label_map = dict({'Normal': 'normal', 'Abnormal': 'abnormal'})
        prompt_text = get_prompt_text(label_map)
        """
        if 'Normal' in vid_kind.values():
            self.label_map['Normal'] = 'normal'
        elif 'A' in vid_kind.values():
            self.label_map['A'] = 'a'
        for val in vid_kind.values():
            if val not in self.label_map:
                self.label_map[val] = val.lower()
        self.num_class = len(self.label_map)


        # 初始化模型
        self._init_model()
        # 训练模型
        fit_dict = fit_VadClip(
            trainer=self,
            X_test=X_test,
            # y_test=y_test,
            X_train=X,
            y_train=y,
            model=self.model,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
            clip_num=vid_source_clips_num, crops_num=crops_num, vid_info=vid_info, vid_kind=vid_kind, X_test_extra=X_test_extra
        )
        self.training_history = fit_main(fit_dict['model'], fit_dict['optimizer'], fit_dict['epochs'],  #
                                         fit_dict['device'], fit_dict['X_test'], fit_dict['trainer'],
                                         fit_dict['verbose'], fit_dict['normal_loader'], fit_dict['anomaly_loader'], fit_dict['X_test_extra'])

        self.fitted = True

        training_time = time.time() - start_time

        if self.verbose:
            print(f"Training completed, time elapsed: {training_time:.2f} seconds")

            # Calculate test performance if test data is available
            if X_test is not None and y_test is not None:
                test_scores = self.predict_proba(X_test)
                if len(np.unique(y_test)) > 1:
                    test_auc = roc_auc_score(y_test, test_scores)
                    test_ap = average_precision_score(y_test, test_scores)
                    print(f"Test set AUCROC: {test_auc:.4f}, AUCPR: {test_ap:.4f}")

        return self

    def predict_proba(self, X, vid_kind, vid_source_clips_num,crops_num):

        self.model.eval()
        # 手搓测试集
        ncrop = crops_num
        seg = self.visual_length
        feature = self.input_dim
        # for i in range(len(vid_source_clips_num)):
        #     print(f'{i},{vid_source_clips_num[i]}')
          # 注意上界
        def split_by_seg_list(X, seg_list):
            X = X.reshape(-1, ncrop, feature) # 【seg, crop:10, 2048]
            segments = []
            start = 0
            for seg_len in seg_list:
                end = start + seg_len
                for i in range(ncrop):
                    segment = X[start:end,i]  # shape: [1, seg_len, 2048]
                    segments.append(segment)
                start = end
            return segments

        X = split_by_seg_list(X, vid_source_clips_num.values()) # (2900, seg, 2048)
        vid_kind_expand = [vid_kind[i] for i in range(len(vid_kind)) for _ in range(ncrop)]  # 包含str标签
        clip_num_expand = [vid_source_clips_num[i] for i in range(len(vid_source_clips_num)) for _ in range(ncrop)]
        assert len(X) == len(vid_kind_expand) and len(X) == len(clip_num_expand)
        test_dataset = VideoDataset(X, vid_kind_expand, clip_num_expand, self.visual_length, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        pre = 0
        gt_list = []
        total = 0
        with torch.no_grad():
            for i, item in tqdm(enumerate(test_loader), total=len(test_loader)):
                visual = item[0].squeeze(0)
                length = item[2]

                length = int(length)  # bug:len_cur为96时，lengths输出为[32,32,32,32]
                len_cur = length
                maxlen = self.visual_length
                if len_cur < maxlen:
                    visual = visual.unsqueeze(0)  # 这段啥意思？
                visual = visual.to(self.device)
                lengths = torch.zeros(int(length / maxlen) + 1)
                for j in range(int(length / maxlen) + 1):
                    if j == 0 and length < maxlen:
                        lengths[j] = length
                    elif j == 0 and length > maxlen:
                        lengths[j] = maxlen
                        length -= maxlen
                    elif length > maxlen:
                        lengths[j] = maxlen
                        length -= maxlen
                    else:
                        lengths[j] = length
                lengths = lengths.to(int)  # 这么做是为了分段，0为有效帧，1为padding（无效部分
                padding_mask = get_batch_mask(lengths, maxlen).to(self.device)
                _, logits1, logits2 = self.model(visual, padding_mask, get_prompt_text(self.label_map), lengths)
                logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
                prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
                if i == 0:
                    ap1 = prob1
                else:
                    ap1 = torch.cat([ap1, prob1], dim=0)
                    # if i % 10 == 9:
                    #     # 制作gt
                    #     gt_list.append(np.tile(gt[pre:pre + len_cur * 16], 10))
                    #     pre += len_cur * 16  # 之前没乘16
                    #     total += len_cur * 16 * 10
            scores = ap1.squeeze().cpu().numpy()  # [696270]

        # 确保返回一维数组
        if scores.ndim == 0:
            scores = np.array([scores])
        return scores

    def predict(self, X, vid_kind, vid_source_clips_num, crops_num, threshold=0.5):
        """
        预测异常标签

        Args:
            X: 输入特征
            threshold: 分类阈值

        Returns:
            预测标签 [n_samples]
        """
        scores = self.predict_proba(X,vid_kind, vid_source_clips_num,crops_num)
        return (scores > threshold).astype(int)

    def _preprocess_data(self, X):
        """
        数据预处理

        Args:
            X: 输入数据

        Returns:
            预处理后的数据
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        X = np.array(X, dtype=np.float32)

        # 确保是2D数组
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X

    def get_params(self, deep=True):
        """
        获取模型参数（sklearn兼容）

        Args:
            deep: 是否深度获取参数

        Returns:
            参数字典
        """
        return {
            "seed": self.seed,
            "input_dim": self.input_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "segments_per_video": self.segments_per_video,
            "use_scheduler": self.use_scheduler,
            "scheduler_milestones": self.scheduler_milestones,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """
        设置模型参数（sklearn兼容）

        Args:
            **params: 参数字典

        Returns:
            self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 如果模型已经初始化，需要重新初始化
        if self.model is not None:
            self.model = None
            self.fitted = False

        return self

    def parameter_count(self):
        """
        计算Sultani模型的参数数量

        Returns:
            dict: 包含各个组件参数数量的字典
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                # 计算已初始化模型的参数
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )
                non_trainable_params = total_params - trainable_params

                return {
                    "sultani_total": total_params,
                    "sultani_trainable": trainable_params,
                    "sultani_non_trainable": non_trainable_params,
                    "total": total_params,
                }
            else:
                # 如果模型还没有初始化，创建临时模型来计算参数
                # from WSADBench.baseline.Sultani.model import SultaniLearner

                temp_model = CLIPVAD(input_dim=self.input_dim, device=self.device, visual_length = self.visual_length)
                total_params = sum(p.numel() for p in temp_model.parameters())
                trainable_params = sum(
                    p.numel() for p in temp_model.parameters() if p.requires_grad
                )
                non_trainable_params = total_params - trainable_params

                return {
                    "sultani_total": total_params,
                    "sultani_trainable": trainable_params,
                    "sultani_non_trainable": non_trainable_params,
                    "total": total_params,
                    "note": f"Parameters counted from temporary model (input_dim={self.input_dim})",
                }
        except Exception as e:
            return {"error": f"Failed to count parameters: {str(e)}", "total": 0}
