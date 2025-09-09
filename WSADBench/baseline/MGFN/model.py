# -*- coding: utf-8 -*-
"""
Sultani方法模型架构定义
基于"Real-world Anomaly Detection in Surveillance Videos"论文实现
Multiple Instance Learning (MIL) 弱监督异常检测方法
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, einsum
from einops import rearrange

def init_weights_xavier(module):
    """
    Xavier权重初始化函数

    Args:
        module: 神经网络模块
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def count_parameters(model):
    """
    统计模型参数数量
    Args:
        model: PyTorch模型
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




import numpy as np
import torch

def new_feature(feat: np.ndarray) -> torch.Tensor:
    """
    根据给定的特征批次，计算每个特征段的幅度信息，
    并将其拼接回特征，最终调整维度。

    Args:
        feat (np.ndarray): 输入特征数组，形状为 (batch, ncrop, seg, feature_dim)。
                           例如 (30, 10, 32, 2048)

    Returns:
        torch.Tensor: 处理后的特征数组，形状为 (batch, ncrop, seg, feature_dim + 1)。
                      例如 (30, 10, 32, 2049)
    """
    # 计算每个特征向量的幅度（最后一个维度是 feature_dim）
    feature_mag = np.linalg.norm(feat, axis=-1, keepdims=True)  # shape: (batch, ncrop, seg, 1)

    # 拼接幅度到原特征
    combined_feat = np.concatenate((feat, feature_mag), axis=-1)  # shape: (batch, ncrop, seg, feature_dim + 1)

    # 转换为 torch.Tensor
    return torch.from_numpy(combined_feat)


def exists(val):
    return val is not None


def attention(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

def MSNSD(features,scores,bs,batch_size,drop_out,ncrops,k):
    #magnitude selection and score prediction
    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)
    scores = scores.unsqueeze(dim=2)  # (B,32,1)

    normal_features = features[0:batch_size * ncrops]  # [b/2*ten,32,1024]  # 这里不能写10
    normal_scores = scores[0:batch_size]  # [b/2, 32,1]

    abnormal_features = features[batch_size * ncrops:]
    abnormal_scores = scores[batch_size:]

    feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]
    feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]
    nfea_magnitudes = feat_magnitudes[0:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2

    if nfea_magnitudes.shape[0] == 1:  # this is for inference
        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(nfea_magnitudes).cuda()  # 上GPU
    select_idx = drop_out(select_idx)


    afea_magnitudes_drop = afea_magnitudes * select_idx
    idx_abn = torch.topk(afea_magnitudes_drop, k, dim=1)[1]
    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
    abnormal_features = abnormal_features.permute(1, 0, 2, 3)

    total_select_abn_feature = torch.zeros(0).cuda()  # bug  Expected all tensors to be on the same device
    for abnormal_feature in abnormal_features:
        feat_select_abn = torch.gather(abnormal_feature, 1,
                                       idx_abn_feat)
        total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))  #

    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #
    score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                dim=1)


    select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
    select_idx_normal = drop_out(select_idx_normal)
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]
    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, ncrops, t, f)
    normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.zeros(0).cuda()
    for nor_fea in normal_features:
        feat_select_normal = torch.gather(nor_fea, 1,
                                          idx_normal_feat)
        total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores

class Backbone(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mgfn_type = 'gb',
        kernel = 5,
        dim_headnumber = 64,
        ff_repe = 4,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads = heads, dim_head = dim_headnumber, local_aggr_kernel = kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads = heads, dim_head = dim_headnumber, dropout = attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding = 1),
                attention,
                FeedForward(dim, repe = ff_repe, dropout = dropout),
            ]))

    def forward(self, x):
        for scc, attention, ff in self.layers:
            x = scc(x) + x
            x = attention(x) + x
            x = ff(x) + x

        return x

# main class

class mgfn(nn.Module):
    def __init__(
        self,
        batch_size, drop_p,
        *,
        dims = (64, 128, 1024),
        depths,
        mgfn_types,  # 写死
        lokernel = 5,
        input_dim = None,
        ff_repe = 4,
        dim_head = 64,
        attention_dropout = 0.,
        mag_ratio  # 必须是在mgfn初始化的时候把参数扔进去
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.input_dim  =  input_dim
        self.to_tokens = nn.Conv1d(input_dim, init_dim, kernel_size=3, stride = 1, padding = 1)
        self.mag_ratio = mag_ratio
        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mgfn_type = mgfn_types,
                    ff_repe = ff_repe,
                    dropout = drop_p,
                    attention_dropout = attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride = 1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(last_dim)
        )
        self.batch_size =  batch_size
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(drop_p)

        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)
    def forward(self, video):
        # print(f'vedio.shape:{video.shape}')
        k = 3
        bs, ncrops, t, c = video.size()
        x = video.view(bs * ncrops, t, c).permute(0, 2, 1)
        x_f = x[:,:self.input_dim,:]
        x_m = x[:,self.input_dim:,:]
        x_f = self.to_tokens(x_f)
        x_m = self.to_mag(x_m)
        x_f = x_f+self.mag_ratio*x_m

        for backbone, conv in self.stages:
            x_f = backbone(x_f)
            if exists(conv):
                x_f = conv(x_f)

        x_f = x_f.permute(0, 2, 1)
        x =  self.to_logits(x_f)
        scores = self.sigmoid(self.fc(x))  # (B*10crop,32,1)
        if bs >1:
            score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores  = MSNSD(x,scores,bs,int(bs/2),self.drop_out,ncrops,k)
            return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores
        else:  # 特判bs为1时，（test）
            scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)
            scores = scores.unsqueeze(dim=2)  # (B,32,1)
            return scores





class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


def FeedForward(dim, repe = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * repe, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * repe, dim, 1)
    )

# MHRAs (multi-head relation aggregators)
class FOCUS(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, bias = False)
        self.rel_pos = nn.Conv1d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x): #
        x = self.norm(x) #(b*crop,c,t)
        b, c, *_, h = *x.shape, self.heads
        v = self.to_v(x) #(b*crop,c,t)
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h) #(b*ten*64,c/64,32)
        out = self.rel_pos(v)
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        return self.to_out(out)


class GLANCE(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.attn =0

    def forward(self, x):
        x = self.norm(x)
        shape, h = x.shape, self.heads
        x = rearrange(x, 'b c ... -> b c (...)')
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        self.attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', self.attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h = h)
        out = self.to_out(out)

        return out.view(*shape)
# 兼容性别名
# Learner = MGFNLearner  # 兼容原始实现的命名
