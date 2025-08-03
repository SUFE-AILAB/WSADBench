import numpy as np
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class CustomDataset(Dataset):
    def __init__(self, samples, labels):  # tensor， narray
        self.labels = labels
        self.samples = samples
        self.dim_features = samples.shape[1]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sample = self.samples[idx]
        data = {"sample": sample, "label": label}
        return data

class TabNeutralAD(torch.nn.Module):  # 模型入口
    """
    RoSAS网络结构：包含嵌入层和评分层
    """

    def __init__(self, x_dim):  # 这个x_dim要匹配特征数。。
        super(TabNeutralAD, self).__init__()
        self.x_dim = x_dim
        self.enc, self.trans = TabNets()._make_nets(x_dim)
        self.num_trans = 9  # config['num_trans']
        self.trans_type = 'residual'  # config['trans_type']
        self.device = 'cuda'  # config['device']
        if 32 <= x_dim <= 300:
            self.z_dim = 32
        elif x_dim < 32:
            self.z_dim = 2 * x_dim
        else:
            self.z_dim = 64


    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)

        x_T = torch.empty(x.shape[0], self.num_trans, x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1), x_T], 1)
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.num_trans + 1, self.z_dim)
        return zs



class TabTransformNet(nn.Module):
    def __init__(self, x_dim,h_dim,bias,num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=bias))
            net.append(nn.BatchNorm1d(h_dim, affine=bias))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=bias))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)

        return out


class TabEncoder(nn.Module):
    def __init__(self, x_dim,h_dim,z_dim,bias,num_layers,batch_norm):

        super(TabEncoder, self).__init__()

        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, h_dim,bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(h_dim,affine=bias))
            enc.append(nn.ReLU())
            input_dim = h_dim

        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, z_dim,bias=bias)
    def forward(self, x):

        z = self.enc(x)
        z = self.fc(z)

        return z

class TabNets():

    def _make_nets(self,x_dim):
        enc_nlayers = 3#config['enc_nlayers']
        # try:
        #     hdim = config['enc_hdim']
        #     zdim = config['latent_dim']
        #     trans_dim = config['trans_hdim']
        # except:
        if 32<=x_dim <= 300:
            zdim = 32
            hdim = 64
            trans_dim = x_dim
        elif x_dim<32:
            zdim = 2 * x_dim
            hdim = 2 * x_dim
            trans_dim = x_dim
        else:
            zdim = 64
            hdim = 256
            trans_dim = x_dim
        trans_nlayers = 3 #config['trans_nlayers']
        num_trans = 9#config['num_trans']
        batch_norm =False # config['batch_norm']

        enc = TabEncoder(x_dim, hdim,zdim, False,enc_nlayers,batch_norm)
        trans = nn.ModuleList(
            [TabTransformNet(x_dim, trans_dim, False, trans_nlayers) for _ in range(num_trans)])

        return enc,trans
# class RoSASLoss(torch.nn.Module):
#     """
#     RoSAS损失函数：结合三元组损失和Mixup正则化
#     """
#
#     def __init__(
#         self, l2_reg_weight=0.0, margin=1.0, alpha=1.0, beta=2.0, T=2, k=2, score_loss="smooth", device="cuda"
#     ):
#         super(RoSASLoss, self).__init__()
#         self.l2_reg_weight = l2_reg_weight
#         self.loss_tri = torch.nn.TripletMarginLoss(margin=margin)
#         self.T = T
#         self.alpha = alpha
#         self.k = k
#
#         if score_loss == "mse":
#             self.loss_reg = torch.nn.MSELoss(reduction="none")
#         elif score_loss == "mae":
#             self.loss_reg = torch.nn.L1Loss(reduction="none")
#         elif score_loss == "smooth":
#             self.loss_reg = torch.nn.SmoothL1Loss(reduction="none")
#         else:
#             raise ValueError("unsupported loss")
#
#         self.device = device
#
#     def forward(self, basenet, anchor, pos, neg, pre_emb_loss, pre_score_loss):
#         anchor_emb, anchor_s = basenet(anchor)
#         pos_emb, pos_s = basenet(pos)
#         neg_emb, neg_s = basenet(neg)
#
#         # 嵌入损失
#         loss_emb = self.loss_tri(anchor_emb, pos_emb, neg_emb)
#         l2_reg = torch.norm(anchor_emb + pos_emb + neg_emb, p=2)
#
#         # Mixup正则化
#         if self.k == 2:
#             x_i = torch.cat((anchor, pos, neg), 0)
#             target_i = torch.cat(
#                 (torch.ones_like(anchor_s) * -1, torch.ones_like(anchor_s) * -1, torch.ones_like(neg_s)), 0
#             )
#
#             indices_j = torch.randperm(x_i.size(0)).to(self.device)
#             x_j = x_i[indices_j]
#             target_j = target_i[indices_j]
#
#             Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
#             lambdas = Beta.sample(target_i.flatten().shape).to(self.device)[:, 1]
#
#             x_tilde = x_i * lambdas.view(lambdas.size(0), 1) + x_j * (1 - lambdas.view(lambdas.size(0), 1))
#             _, score_tilde = basenet(x_tilde)
#
#             _, score_xi = basenet(x_i)
#             _, score_xj = basenet(x_j)
#
#             score_mix = score_xi * lambdas.view(lambdas.size(0), 1) + score_xj * (1 - lambdas.view(lambdas.size(0), 1))
#             y_tilde = target_i * lambdas.view(lambdas.size(0), 1) + target_j * (1 - lambdas.view(lambdas.size(0), 1))
#             loss_out = self.loss_reg(score_tilde, y_tilde)
#             loss_intra = self.loss_reg(score_tilde, score_mix)
#
#             loss_score = loss_out + loss_intra
#             loss_score = loss_score.mean()
#             loss_out = loss_out.mean()
#             loss_intra = loss_intra.mean()
#
#         else:
#             # n-samples mixup
#             x_i = torch.cat((anchor, pos, neg), 0)
#             target_i = torch.cat(
#                 (torch.ones_like(anchor_s) * -1, torch.ones_like(anchor_s) * -1, torch.ones_like(neg_s)), 0
#             )
#             _, score_xi = basenet(x_i)
#
#             x_dup = [x_i]
#             target_dup = [target_i]
#             score_dup = [score_xi]
#             for k in range(1, self.k):
#                 indices_j = torch.randperm(x_i.size(0)).to(self.device)
#                 x_j = x_i[indices_j]
#                 target_j = target_i[indices_j]
#                 _, score_xj = basenet(x_j)
#
#                 x_dup.append(x_j)
#                 target_dup.append(target_j)
#                 score_dup.append(score_xj)
#
#             Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
#             lambdas_dup = Beta.sample((target_i.flatten().shape[0], self.k)).to(self.device)[:, :, 1]
#
#             s = torch.sum(lambdas_dup, 1).unsqueeze(0).T.repeat(1, self.k)
#             lambdas_dup = lambdas_dup / s
#
#             x_tilde = lambdas_dup[:, 0].unsqueeze(0).T * x_i
#             y_tilde = lambdas_dup[:, 0].unsqueeze(0).T * target_i
#             score_mix = lambdas_dup[:, 0].unsqueeze(0).T * score_xi
#             for k in range(1, self.k):
#                 x_tilde += lambdas_dup[:, k].unsqueeze(0).T * x_dup[k]
#                 y_tilde += lambdas_dup[:, k].unsqueeze(0).T * target_dup[k]
#                 score_mix += lambdas_dup[:, k].unsqueeze(0).T * score_dup[k]
#
#             _, score_tilde = basenet(x_tilde)
#
#             loss_out = self.loss_reg(score_tilde, y_tilde)
#             loss_intra = self.loss_reg(score_tilde, score_mix)
#
#             loss_score = loss_out + loss_intra
#             loss_score = loss_score.mean()
#             loss_out = loss_out.mean()
#             loss_intra = loss_intra.mean()
#
#         # 动态权重调整
#         k1 = torch.exp((loss_emb / pre_emb_loss) / self.T) if pre_emb_loss != 0 else torch.tensor(1.0).to(self.device)
#         k2 = (
#             torch.exp((loss_score / pre_score_loss) / self.T)
#             if pre_score_loss != 0
#             else torch.tensor(1.0).to(self.device)
#         )
#         loss = (k1 / (k1 + k2)) * loss_emb + (k2 / (k1 + k2)) * loss_score + self.l2_reg_weight * l2_reg
#
#         return loss, loss_emb, loss_score, loss_out, loss_intra
#
#
# class DataGenerator:
#     """
#     数据生成器：生成三元组训练数据
#     """
#
#     def __init__(self, x, y, batch_size=256):
#         self.x = x
#         self.y = y
#
#         self.anom_idx = np.where(self.y == 1)[0]
#         self.anom_x = self.x[self.anom_idx]
#         self.norm_idx = np.where(self.y == 0)[0]
#         self.norm_x = self.x[self.norm_idx]
#
#         self.batch_size = batch_size
#
#     def load_batches(self, n_batches=10):
#         import numpy as np
#
#         batch_set = []
#
#         for i in range(n_batches):
#             anom_idx = np.random.choice(len(self.anom_x), self.batch_size)
#             anchor_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)
#             pos_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)
#
#             batch = [[self.norm_x[a], self.norm_x[p], self.anom_x[n]] for a, p, n in zip(anchor_idx, pos_idx, anom_idx)]
#             batch_set.append(batch)
#         return np.array(batch_set)
