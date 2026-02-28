# -*- coding: utf-8 -*-
"""
ZhongGCNAD主入口类
基于"Graph Convolutional Label Noise Cleaner"论文实现的视频异常检测模型 (修正版)
此版本支持多crop输入，并允许自定义GCN超参数。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter
from math import sqrt
import torch.nn.functional as F
from WSADBench.myutils import Utils
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
from common_utils.baseline_utils import get_gt, write_jsonl

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. GCN核心层和模型定义
# =============================================================================

class GraphConvolution(nn.Module):
    """
    简单的GCN层，类似于 https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重和偏置"""
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_tensor: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """GCN层的前向传播"""
        support = torch.matmul(input_tensor, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class NoiseFilter(nn.Module):
    """
    GCN标签噪声清洁器模型.
    包含两个分支: 时间一致性图 和 特征相似性图.
    """

    def __init__(self, nfeat: int, nclass: int = 1, dropout_rate: float = 0.6):
        super(NoiseFilter, self).__init__()
        # 特征压缩层
        self.fc1 = nn.Linear(nfeat, 512)
        self.fc2 = nn.Linear(512, 128)

        # 分支1: 时间一致性图
        self.gc1 = GraphConvolution(128, 32)
        self.gc2 = GraphConvolution(32, nclass)

        # 分支2: 特征相似性图
        self.gc3 = GraphConvolution(128, 32)
        self.gc4 = GraphConvolution(32, nclass)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        GCN模型的前向传播.
        Args:
            x (torch.Tensor): 输入特征 (batch_size, num_nodes, feat_dim).
            adj (torch.Tensor): 预先计算的时间一致性邻接矩阵 (batch_size, num_nodes, num_nodes).
        """
        # 特征压缩
        x_compressed = F.relu(self.fc1(x))
        x_compressed = F.dropout(x_compressed, self.dropout_rate, training=self.training)
        x_compressed = F.relu(self.fc2(x_compressed))
        x_compressed = F.dropout(x_compressed, self.dropout_rate, training=self.training)

        # 分支1: 时间一致性图 (使用预计算的adj)
        x1 = F.relu(self.gc1(x_compressed, adj))
        x1 = self.gc2(x1, adj)

        # 分支2: 特征相似性图 (动态计算adj)
        # 假设 batch_size 为 1
        x2_flat = x_compressed.squeeze(0)

        # 基于余弦相似度动态计算相似度矩阵
        x2_norm = F.normalize(x2_flat, p=2, dim=1)
        sim_matrix = torch.matmul(x2_norm, x2_norm.t())

        # 归一化
        d_inv_sqrt2 = torch.pow(torch.sum(sim_matrix, dim=1).clamp(min=1e-12), -0.5).diag()
        adj_hat2 = torch.matmul(torch.matmul(d_inv_sqrt2, sim_matrix), d_inv_sqrt2)

        y2 = F.relu(self.gc3(x_compressed, adj_hat2.unsqueeze(0)))
        y2 = self.gc4(y2, adj_hat2.unsqueeze(0))

        # 平均融合两个分支的结果
        return (x1 + y2) / 2.0


# =============================================================================
# 2. 分类器头部、损失函数和数据集定义
# =============================================================================

class ActionClassifierHead(nn.Module):
    """用于初步分类和最终预测的简单MLP分类器"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
        super(ActionClassifierHead, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SigmoidCrossEntropyLoss(nn.Module):
    """带Sigmoid的二元交叉熵损失 (用于有监督部分)"""

    def forward(self, x, target):
        return F.binary_cross_entropy_with_logits(x, target)


class SigmoidMAELoss(nn.Module):
    """带Sigmoid的平均绝对误差损失 (用于无监督部分)"""

    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1_loss(torch.sigmoid(pred), target)


class CustomDataset(Dataset):
    """自定义数据集，按视频ID组织数据"""

    def __init__(self, features, labels, vid_info):
        self.features = features
        self.labels = labels
        self.vid_info = vid_info if vid_info is not None else np.arange(len(features))

        # 按视频ID对特征和标签的索引进行分组
        self.video_indices: Dict[Any, List[int]] = {}
        for i, vid in enumerate(self.vid_info):
            vid = vid.item() if isinstance(vid, np.generic) else vid
            if vid not in self.video_indices:
                self.video_indices[vid] = []
            self.video_indices[vid].append(i)

        self.video_ids = list(self.video_indices.keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        indices = self.video_indices[vid]

        video_features = self.features[indices]
        video_labels = self.labels[indices]

        return torch.FloatTensor(video_features), torch.FloatTensor(video_labels), vid
# --- [新增辅助函数] 从 fit.py 复制而来 ---
def _process_video_scores(scores, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames):
    """处理video分数的特殊逻辑：从clip级别还原到帧级别"""
    _clips_num, _crops_num = video_shape

    # 平均每个crop, 获得每个clip的分数
    scores = scores.reshape(_clips_num, _crops_num)
    scores = np.mean(scores, axis=1)

    frame_scores = []
    frame_truth = []
    for i in range(max(y_test_gt_idx) + 1):
        select_gt = y_test_gt[y_test_gt_idx == i]
        select_scores = scores[y_test_idx == i]
        select_scores = select_scores.repeat(num_clip_frames)
        common_length = min(len(select_gt), len(select_scores))

        frame_scores.append(select_scores[:common_length])
        frame_truth.append(select_gt[:common_length])

    frame_scores = np.concatenate(frame_scores, axis=0)
    frame_truth = np.concatenate(frame_truth, axis=0)
    return frame_scores, frame_truth

# =============================================================================
# 3. 主类实现
# =============================================================================
class ZhongGCNAD:
    """
    ZhongGCNAD方法实现
    基于图卷积神经网络的标签噪声清理方法用于视频异常检测

    论文: "Graph Convolutional Label Noise Cleaner: Train a Plug-and-play Action Classifier for Anomaly Detection"
    """

    def __init__(
            self,
            seed: int = 42,
            input_dim: int = 4096,
            hidden_dims: List[int] = None,
            dropout: float = 0.6,
            optimization_rounds: int = 3,
            classifier_epochs: int = 10,
            gcn_epochs: int = 8,
            classifier_lr: float = 0.001,
            gcn_lr: float = 0.0001,
            gcn_momentum: float = 0.9,
            gcn_weight_decay: float = 0.0005,
            gcn_loss_i_weight: float = 0.8,
            batch_size: int = 64,
            verbose: bool = True,
    ):
        self.seed = seed
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [512, 128]
        self.gcn_feat_dim = input_dim
        self.dropout = dropout
        self.optimization_rounds = optimization_rounds
        self.classifier_epochs = classifier_epochs
        self.gcn_epochs = gcn_epochs
        self.classifier_lr = classifier_lr
        self.gcn_lr = gcn_lr
        self.gcn_momentum = gcn_momentum
        self.gcn_weight_decay = gcn_weight_decay
        self.gcn_loss_i_weight = gcn_loss_i_weight
        self.batch_size = batch_size
        self.verbose = verbose

        self.utils = Utils()  # 初始化工具类
        self.utils.set_seed(self.seed)  # 设置随机种子
        self.device = self.utils.get_device(True)  # 获取设备 (GPU或CPU)

        self.action_classifier = None
        self.gcn_model = None
        self.fitted = False

    def _build_temporal_adj(self, num_nodes: int) -> np.ndarray:
        """
        构建并归一化时间一致性邻接矩阵 A_hat = D^{-1/2} @ A @ D^{-1/2}
        其中 A[i, j] = exp(-|i-j|)
        """
        adj = np.zeros((num_nodes, num_nodes))
        indices = np.arange(num_nodes)
        # 使用广播高效计算距离
        adj = np.exp(-np.abs(indices[:, np.newaxis] - indices[np.newaxis, :]))

        # 归一化
        sum_adj = np.sum(adj, axis=1)
        sum_adj[sum_adj <= 1e-12] = 1  # 防止除以零

        d_inv_sqrt = np.diag(np.power(sum_adj, -0.5))
        adj_hat = d_inv_sqrt @ adj @ d_inv_sqrt
        return adj_hat

    def _soft_uniform_sampling(self, video_features: np.ndarray, video_scores: np.ndarray,
                               video_variances: np.ndarray) -> Tuple:
        """
        根据方差进行软均匀采样，并构建时间邻接矩阵.
        返回采样后的数据、邻接矩阵和相关索引.
        """
        # 1. 选择高置信度样本 (方差最小的前30%)
        var_threshold = np.quantile(video_variances, 0.3)
        high_conf_indices = np.where(video_variances <= var_threshold)[0]

        if len(high_conf_indices) < 2:  # 保证至少有2个高置信度样本
            high_conf_indices = np.argsort(video_variances)[:max(2, int(len(video_variances) * 0.1))]

        # 2. 扩展采样范围以包含上下文
        sample_index_set = set(high_conf_indices)
        local_samples = 8
        for i in high_conf_indices:
            start = max(0, i - local_samples // 2)
            end = min(len(video_features), i + local_samples // 2 + 1)
            sample_index_set.update(range(start, end))

        sample_index = sorted(list(sample_index_set))

        # 3. 构建时间一致性邻接矩阵
        adj_hat = self._build_temporal_adj(len(sample_index))

        # 4. 准备输出
        sampled_features = video_features[sample_index]
        sampled_scores = video_scores[sample_index]

        # 获取高置信度节点在子图中的相对索引
        high_conf_mask = np.isin(sample_index, high_conf_indices)
        high_conf_indices_in_graph = np.where(high_conf_mask)[0]

        return sampled_features, adj_hat, sampled_scores, high_conf_indices_in_graph, sample_index

    def _clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _train_classifier(self, X: np.ndarray, y: np.ndarray, epochs: int):
        """内部函数：训练或再训练动作分类器头部"""
        if self.action_classifier is None:
            self.action_classifier = ActionClassifierHead(self.input_dim, self.hidden_dims).to(self.device)

        self.action_classifier.train()
        optimizer = torch.optim.Adam(self.action_classifier.parameters(), lr=self.classifier_lr)
        criterion = nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.action_classifier(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.verbose:
                logger.info(f"  Classifier Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    def _train_gcn(self, train_dataset: CustomDataset, variances: np.ndarray, epochs: int):
        """内部函数：训练GCN标签清洁器"""
        if self.gcn_model is None:
            self.gcn_model = NoiseFilter(nfeat=self.gcn_feat_dim, dropout_rate=self.dropout).to(self.device)
        self.gcn_model.train()

        optimizer = torch.optim.SGD(
            self.gcn_model.parameters(),
            lr=self.gcn_lr,
            momentum=self.gcn_momentum,
            weight_decay=self.gcn_weight_decay
        )

        criterion_supervised = SigmoidCrossEntropyLoss().to(self.device)
        criterion_unsupervised = SigmoidMAELoss().to(self.device)

        vid_mean_preds = {
            vid: torch.from_numpy(train_dataset.labels[train_dataset.video_indices[vid]]).float()
            for vid in train_dataset.video_ids
        }

        for epoch in range(epochs):
            total_loss = 0
            loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

            for features, scores, vid_tensor in loader:
                vid = vid_tensor.item()
                video_features_np = features.squeeze(0).cpu().numpy()
                video_scores_np = scores.squeeze(0).cpu().numpy()

                # 获取当前视频对应的方差
                video_clip_indices = train_dataset.video_indices[vid]
                video_variances = variances[video_clip_indices]

                # 软均匀采样
                sampled_feats_np, adj_np, sampled_scores_np, high_conf_indices_in_graph, sample_index = self._soft_uniform_sampling(
                    video_features_np, video_scores_np, video_variances
                )

                sampled_feats = torch.FloatTensor(sampled_feats_np).unsqueeze(0).to(self.device)
                adj = torch.FloatTensor(adj_np).unsqueeze(0).to(self.device)

                optimizer.zero_grad()
                output = self.gcn_model(sampled_feats, adj).squeeze()

                # 1. 直接监督损失
                target_scores = torch.FloatTensor(sampled_scores_np[high_conf_indices_in_graph]).to(self.device)
                loss_d = criterion_supervised(output[high_conf_indices_in_graph], target_scores)

                # 2. 间接监督损失
                mean_pred_full = vid_mean_preds[vid].to(self.device)
                mean_pred_sampled = mean_pred_full[sample_index]
                loss_i = criterion_unsupervised(output, mean_pred_sampled)

                loss = loss_d + self.gcn_loss_i_weight * loss_i
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # 更新历史平均预测并立即移回CPU
                with torch.no_grad():
                    new_mean_pred_full = vid_mean_preds[vid].to(self.device)
                    new_mean_pred_full[sample_index] = 0.5 * new_mean_pred_full[sample_index] + 0.5 * torch.sigmoid(
                        output).detach()
                    vid_mean_preds[vid] = new_mean_pred_full.cpu()

            if self.verbose:
                logger.info(f"  GCN Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    def fit(self, X: np.ndarray, y: np.ndarray, vid_info: np.ndarray = None, crops_num: int = None, X_test=None):
        """
        训练ZhongGCNAD模型
        Args:
            X (np.ndarray): 输入特征，形状为 (total_rows, feat_dim)，其中 total_rows = clips_num * crops_num.
            y (np.ndarray): 输入标签，形状为 (total_rows,).
            vid_info (np.ndarray): 视频ID信息，形状为 (total_rows,).
            crops_num (int): 每个clip的crop数量，必须提供以便正确重塑数据.
        """
        assert crops_num is not None, "crops_num 必须被提供以便正确重塑数据."  # 没有类别标签
        clips_num = X.shape[0] // crops_num
        # 注意：这里解包会覆盖 X_test 变量，下面的 X_test 指的是测试集特征数组
        X_test_feat, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames = X_test  # 拆包
        # 重塑X并计算每个clip的平均特征
        X_reshaped = X.reshape(clips_num, crops_num, -1)
        X_clips = X_reshaped.mean(axis=1)

        # 提取每个clip的标签和视频ID (假设同一clip的所有crop共享相同信息)
        y_clips = y.reshape(clips_num, crops_num)[:, 0]
        vid_info_clips = vid_info.reshape(clips_num, crops_num)[:, 0]

        self.input_dim = X_clips.shape[1]
        self.gcn_feat_dim = X_clips.shape[1]
        current_labels = y_clips.copy()

        if self.verbose:
            logger.info("--- 初始分类器训练 ---")
        self._train_classifier(X_clips, current_labels, self.classifier_epochs)

        for i in range(self.optimization_rounds):
            if self.verbose:
                logger.info(f"\n--- 交替优化轮次 {i + 1}/{self.optimization_rounds} ---")

            # 1. 使用当前分类器生成软标签和方差
            if self.verbose:
                logger.info("  步骤1: 生成当前软标签和方差...")
            self.action_classifier.eval()
            with torch.no_grad():
                # 分批处理以避免内存溢出
                soft_labels_list = []
                variances_list = []
                batch_size_for_inference = min(self.batch_size, clips_num)

                for start_idx in range(0, len(X), batch_size_for_inference * crops_num):
                    end_idx = min(start_idx + batch_size_for_inference * crops_num, len(X))
                    batch_X = X[start_idx:end_idx]

                    # 对当前批次的所有crop进行预测
                    batch_preds = self.action_classifier(torch.FloatTensor(batch_X).to(self.device))
                    batch_clips = (end_idx - start_idx) // crops_num
                    batch_preds_reshaped = torch.sigmoid(batch_preds).reshape(batch_clips, crops_num)

                    # 计算当前批次的平均预测和方差
                    batch_soft_labels = batch_preds_reshaped.mean(dim=1).cpu().numpy()
                    batch_variances = torch.var(batch_preds_reshaped, dim=1).cpu().numpy()

                    soft_labels_list.append(batch_soft_labels)
                    variances_list.append(batch_variances)

                soft_labels = np.concatenate(soft_labels_list).flatten()
                variances = np.concatenate(variances_list).flatten()

                # 清理列表
                del soft_labels_list, variances_list
                self._clear_gpu_memory()

            # 2. 训练GCN清洁器
            if self.verbose:
                logger.info("  步骤2: 训练GCN标签清洁器...")
            gcn_train_dataset = CustomDataset(X_clips, soft_labels, vid_info_clips)
            self._train_gcn(gcn_train_dataset, variances, self.gcn_epochs)

            # 3. 使用训练好的GCN生成清洁后的软标签
            if self.verbose:
                logger.info("  步骤3: 生成清洁后的软标签...")
            self.gcn_model.eval()
            new_labels_list = []

            inference_loader = DataLoader(gcn_train_dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                for features, _, _ in inference_loader:
                    features = features.to(self.device)
                    num_nodes = features.shape[1]
                    adj_full_video = self._build_temporal_adj(num_nodes)
                    adj_tensor = torch.FloatTensor(adj_full_video).unsqueeze(0).to(self.device)
                    cleaned_scores = torch.sigmoid(self.gcn_model(features, adj_tensor)).cpu().numpy().flatten()
                    new_labels_list.extend(cleaned_scores)

            current_labels = np.array(new_labels_list)
            del new_labels_list, gcn_train_dataset, inference_loader
            self._clear_gpu_memory()

            # 4. 使用清洁后的标签再训练分类器
            if self.verbose:
                logger.info("  步骤4: 使用清洁标签再训练分类器...")
            self._train_classifier(X_clips, current_labels, self.classifier_epochs)

        self.fitted = True
        if self.verbose:
            logger.info("\n--- 训练完成 ---")
            """
            todo:添加两种gt的计算
            """
            # --- [修改开始] 添加两种GT的计算 ---
            if X_test_feat is not None:
                logger.info("开始计算测试集指标 (Frame-level & Clip-level)...")

                # 1. 获取预测分数
                scores = self.predict_proba(X_test_feat)

                # 2. 计算 Frame-level GT 指标 (v1)
                frame_scores, frame_truth = _process_video_scores(
                    scores, video_shape, y_test_idx, y_test_gt,
                    y_test_gt_idx, num_clip_frames
                )
                test_auc = roc_auc_score(frame_truth, frame_scores)
                test_ap = average_precision_score(frame_truth, frame_scores)

                # 3. 计算 Clip-level GT 指标 (v2)
                # 假设每个clip重复16次 (参考fit.py逻辑)
                prob = np.repeat(scores, 16)
                gt = get_gt(len(prob))
                test_auc_v2 = roc_auc_score(gt, prob)
                test_ap_v2 = average_precision_score(gt, prob)

                # 4. 写入结果文件
                write_jsonl(model_name='ZhongGCNAD', epochs=self.optimization_rounds, seed=self.seed, auc=test_auc,
                            ap=test_ap, res_type='frame')
                write_jsonl(model_name='ZhongGCNAD', epochs=self.optimization_rounds, seed=self.seed, auc=test_auc_v2,
                            ap=test_ap_v2, res_type='clip')

                logger.info(f"[Result] Frame-level (v1) - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
                logger.info(f"[Result] Clip-level  (v2) - AUC: {test_auc_v2:.4f}, AP: {test_ap_v2:.4f}")
            # --- [修改结束] ---
            # # 统计模型参数量
            # if self.verbose:
            #     logger.info("\n=== 模型参数统计 ===")
            #
            #     # 统计ActionClassifierHead参数
            #     classifier_params = sum(p.numel() for p in self.action_classifier.parameters())
            #     logger.info(f"ActionClassifierHead参数量: {classifier_params:,}")
            #
            #     # 统计NoiseFilter(GCN)参数
            #     gcn_params = sum(p.numel() for p in self.gcn_model.parameters())
            #     logger.info(f"NoiseFilter(GCN)参数量: {gcn_params:,}")
            #
            #     # 总参数量
            #     total_params = classifier_params + gcn_params
            #     logger.info(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
            #     logger.info(f"分类器占比: {classifier_params / total_params * 100:.1f}%")
            #     logger.info(f"GCN占比: {gcn_params / total_params * 100:.1f}%")

        # 最终清理
        self._clear_gpu_memory()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        使用最终训练好的分类器预测异常概率.
        注意: 这里的输入X应该是clip-level的特征 (即每个clip一个特征向量).
        如果测试数据也有crops，需要先取平均.
        """
        if not self.fitted:
            raise RuntimeError("模型在预测前需要先进行训练. 请调用 .fit() 方法.")

        self.action_classifier.eval()
        with torch.no_grad():
            probas = []
            test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size * 2, shuffle=False)
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(self.device)
                scores = self.action_classifier(batch_X)
                proba_batch = torch.sigmoid(scores).cpu().numpy()
                probas.append(proba_batch)

        return np.vstack(probas).flatten()
