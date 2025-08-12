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
