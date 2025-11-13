from pyexpat import model
import torch
#from torch import nn
#from multiprocessing import Pool, freeze_support, cpu_count, set_start_method

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyod.models.base import BaseDetector
from pyod.utils.torch_utility import get_activation_by_name
from pyod.utils.stat_models import pairwise_distances_no_broadcast
import io
from torch.utils.tensorboard import SummaryWriter
from WSADBench.baseline.PUMA.model import PyODDataset, inner_autoencoder
import os


def _ss_loss(l1, y,model_A,model_B,mu1, sigma1, mu2, sigma2,device):
        loss_index = torch.where(y != 0)[0]
        if loss_index.nelement()>0:
            l1 = l1[loss_index,:,:]
            y = y[loss_index]
            pij = _logistic(l1,model_A,model_B)
            pi = _weightnoisyor(pij,mu1, sigma1, mu2, sigma2,device)
            likelihood = -1*(_log_diverse_density(pi, y,device)+1e-10) + 0.01*(model_A**2 + model_B**2)[0]
        else:
            likelihood =  0.01*( model_A**2+ model_B**2)[0]
        return likelihood
    
def _logistic(loss,model_A,model_B):
    return torch.sigmoid( model_A * loss + model_B)

def _noisy_or(pij):
    # instance_prob contains the probability of being positive
    noisyor = 1 - torch.prod(1-pij, dim = 1)
    return noisyor


def _weightnoisyor(pij, mu1, sigma1, mu2, sigma2,device):
    rv1 = torch.distributions.normal.Normal(loc=torch.tensor(mu1), scale=torch.tensor(sigma1))
    rv2 = torch.distributions.normal.Normal(loc=torch.tensor(mu2), scale=torch.tensor(sigma2))
    nbags = pij.size()[0]
    ninstances = pij.size()[1]
    pij = pij.reshape(nbags,ninstances)
    ranks = torch.empty((nbags, ninstances), dtype = torch.float)
    tmp = torch.argsort(pij, dim=1, descending=False)
    for i in range(nbags):
        ranks[i,tmp[i,:]] = torch.arange(0,ninstances)/(ninstances-1)
    w = torch.exp(rv1.log_prob(ranks))+torch.exp(rv2.log_prob(ranks))
    w = torch.div(w,torch.sum(w, dim = 1).reshape(nbags,1))
    pij = pij.to(device, non_blocking = True).float()
    w = w.to(device, non_blocking = True).float()
    noisyor = 1 - torch.prod(torch.pow(1-pij+1e-10,w).clip(min = 0, max = 1), dim = 1)
    return noisyor

def _log_diverse_density(pi, y,device):
    # Compute the likelihood given bag labels y and bag probabilities pi
    z = torch.where(y == -1)[0]
    if z.nelement() > 0:
        zero_sum = torch.sum(torch.log(1-pi[z]+1e-10))
    else:
        zero_sum = torch.tensor(0).to(device, non_blocking = True).float()
        
    o = torch.where(y == 1)[0]
    if o.nelement() > 0:
        one_sum = torch.sum(torch.log(pi[o]+1e-10))
    else:
        one_sum = torch.tensor(0).to(device, non_blocking = True).float()
    return zero_sum+one_sum


def fit_PUMA(model,model_A,model_B, optimizer,train_loader, y_bags,epochs,n_bags,n_samples,input_dim,n_neg,loss_fn,
             mu1, sigma1, mu2, sigma2,device,cont_factor = 0.1,verbose=True):

    """训练PUMA模型
    Args:
        model: PUMA模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        y_bags: 训练包标签
        epochs: 训练轮数
        n_bags: 包的数量
        n_neg: 每个包中负样本的数量
        device: 计算设备
        cont_factor: 连续性因子
        verbose: 是否打印训练信息
    Returns:
        best_model_dict: 最佳模型参数字典
    
    """
    model.train()

    train_history = {
        'best_loss': float('inf'),
        'best_model_dict': None
    }

    if verbose:
        print(f"开始训练PUMA模型，共{epochs}轮...")
        print(f"设备: {device}")
        print(f"包的数量: {n_bags}, 每个包中的负样本数量: {n_neg}")
    
    y_tmp = torch.clone(y_bags).to(device, non_blocking = True)
    neg_idx = torch.where(y_bags == 0)[0].to(device)
    y_tmp[neg_idx[torch.randperm(neg_idx.size(0))[:n_neg]]] = -1
        
    
    for epoch in range(epochs):
        overall_loss = []
        loss1 = []
        loss2 = []
        bag_scores = torch.zeros([n_bags, 1]).to(device, non_blocking = True).float()
        for data, data_idx in train_loader:
            data = data.to(device, non_blocking = True).float()   #将数据转移到CUDA设备 [10,10,input_dim]
            data_idx = data_idx.to(device, non_blocking = True)
            idx_l1 = torch.where(y_tmp[data_idx] != 1)[0]
            local_batch_size,_,_ = data.size()
            optimizer.zero_grad()
            data_inst = torch.reshape(data, (local_batch_size * n_samples, input_dim))
            data_inst = data_inst.to(device, non_blocking = True).float()
            if idx_l1.shape[0] >0:
                data_inst_l1 = torch.reshape(data[idx_l1,:,:],(len(idx_l1)*n_samples, input_dim))
                data_inst_l1 = data_inst_l1.to(device, non_blocking = True).float()
                if cont_factor>0:
                    l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1,model(data_inst_l1))
                    data_inst_l1 = data_inst_l1[torch.where(l1<torch.quantile(l1, 1 - cont_factor, dim=0))]
                loss = loss_fn(data_inst_l1, model(data_inst_l1))
            else:
                loss = torch.tensor(0, dtype = torch.float).to(device, non_blocking = True).float()
                
            loss1.append(loss.item())
            l1 = torch.nn.PairwiseDistance(p=2)(data_inst,model(data_inst)).reshape(local_batch_size, #unsqueeze
                                                                                            n_samples, 1)
            l2 = _ss_loss(l1, y_tmp[data_idx],model_A,model_B,mu1, sigma1, mu2, sigma2,device)
            loss2.append(l2.item())
            #print("Loss1:", loss)
            #print("Loss2:", l2)
            loss += l2
            #print("L2:",l2)
            loss.backward()
            #print("Model A:", self.model_A, self.model_A.grad)
            #print(sum([torch.isfinite(x).all() for x in self.model.parameters()]))
            #print(sum([torch.isfinite(x.grad).all() for x in self.model.parameters()]))

            #torch.nn.utils.clip_grad_norm_(self.model_A, max_norm = 100, error_if_nonfinite = True)
            #print([x.grad for x in self.model.parameters()])
            optimizer.step()
            overall_loss.append(loss.item())
            
            #Need the following to compute the most reliable negative
            instance_scr = _logistic(l1,model_A,model_B)
            pi = _weightnoisyor(instance_scr,mu1, sigma1, mu2, sigma2,device)
            bag_scores[data_idx] = pi.reshape(local_batch_size,1)
        
        del y_tmp
        torch.cuda.empty_cache() #释放显存
        
        # Get the most reliable negatives:
        nonpos_idx = torch.where(y_bags == 0)[0].to(device) 
        sorted_idx = torch.argsort(bag_scores[nonpos_idx], dim=0)[:n_neg]
        y_tmp = torch.clone(y_bags).to(device, non_blocking = True)
        y_tmp[nonpos_idx[sorted_idx]] = -1     
        
        if verbose:
            print('epoch {epoch}: training loss {l} = (Lu {l1} + Lp {l2})'.format(epoch=epoch, l=np.mean(overall_loss),
                                                                                    l1=np.mean(loss1), l2=np.mean(loss2)))
        if np.mean(overall_loss) <= train_history["best_loss"]:
            train_history["best_loss"] = np.mean(overall_loss)
            train_history["best_model_dict"] = model.state_dict()

    if verbose:
        print(f"保存最佳模型，当前最佳损失: {train_history['best_loss']:.6f}")
    
    return train_history

# def fit_PUMA_main(X_train, y, model, model_A, model_B, optimizer, epochs, batch_size, n_bags, n_samples,
#                   input_dim, n_neg, device, mu1=0.3, sigma1=0.1, mu2=0.7, sigma2=0.1,
#                   cont_factor=0.1, verbose=True):
#     """
#     训练PUMA模型的主函数

#     Args:
#         X_train: 训练特征 [n_bags * n_samples, input_dim]
#         y: 训练标签 [n_bags*samples]
#         model: PUMA模型
#         model_A: PUMA模型中的参数A
#         model_B: PUMA模型中的参数B
#         optimizer: 优化器
#         epochs: 训练轮数
#         batch_size: 批量大小
#         n_bags: 包的数量
#         n_samples: 每个包中的样本数量
#         input_dim: 输入特征维度
#         n_neg: 每个包中负样本的数量
#         device: 计算设备
#         mu1: Noisy-OR模型的均值参数1
#         sigma1: Noisy-OR模型的标准差参数1
#         mu2: Noisy-OR模型的均值参数2
#         sigma2: Noisy-OR模型的标准差参数2
#         cont_factor: 连续性因子
#         verbose: 是否打印训练信息

#     Returns:
#         train_history: 训练历史字典，包含最佳损失和最佳模型参数字典
#     """
#     # 将X_train转换为三维
#     X_train = X_train.reshape(n_bags, n_samples, input_dim)
#     train_dataset = PyODDataset(X_train, y, n_samples)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     # 定义损失函数（均方误差）
#     loss_fn = torch.nn.MSELoss(reduction='mean')

#     # 训练PUMA模型
#     return fit_PUMA(model,model_A,model_B, optimizer,train_loader, y,epochs,n_bags,n_samples,input_dim,n_neg,loss_fn,
#              mu1, sigma1, mu2, sigma2,device,cont_factor = 0.1,verbose=True)