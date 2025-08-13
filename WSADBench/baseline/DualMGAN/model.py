
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

class Args:
    def __init__(
        self,
        k_means=10,
        max_iter_MGAOS=2000,
        max_iter_MGAAL=1000,
        lr_sg=0.0001,
        lr_sd=0.01,
        lr_d=0.001,
        decay=1e-6,
        batch_size=1000,
        momentum=0.9,
        nnr_MGAOS=0.4,
        nnr_MGAAL=0.2,
    ):
        self.k_means = k_means
        self.max_iter_MGAOS = max_iter_MGAOS
        self.max_iter_MGAAL = max_iter_MGAAL
        self.lr_sg = lr_sg
        self.lr_sd = lr_sd
        self.lr_d = lr_d
        self.decay = decay
        self.batch_size = batch_size
        self.momentum = momentum
        self.nnr_MGAOS = nnr_MGAOS
        self.nnr_MGAAL = nnr_MGAAL

def get_auc(p_value, data_out_x, data_unl_x):
    """Calculate the AUC score for the detector.
    :param p_value:
    :param data_out_x:
    :param data_unl_x:
    :return:
    """
    p_value = pd.DataFrame(p_value)
    data_y_ide = np.array([1] * (int(data_out_x.shape[0])) + [0] * (data_unl_x.shape[0]))
    data_y_ide = pd.DataFrame(data_y_ide)
    auc = roc_auc_score(data_y_ide, p_value)
    return auc

def get_score(p_value, test_y):
    # The number of the outliers in test
    top_n = 0
    for i in range(test_y.shape[0]):
        top_n = test_y[i]+top_n

    p_value = [i for j in p_value for i in j]
    p_valuen = p_value[:]
    p_value2n = p_value[:]
    p_value_list = np.argsort(p_value)  # 升序
    p_value_n = p_value[p_value_list[-top_n]]  # 最值
    p_value_2n = p_value[p_value_list[-2 * top_n]]
    for idx1 in range(len(p_value)):
        if p_valuen[idx1] >= p_value_n:
            p_valuen[idx1] = 1
        else:
            p_valuen[idx1] = 0
    for idx2 in range(len(p_value)):
        if p_value2n[idx2] >= p_value_2n:
            p_value2n[idx2] = 1
        else:
            p_value2n[idx2] = 0
    p_value = pd.DataFrame(p_value)
    p_valuen = pd.DataFrame(p_valuen)
    test_y = pd.DataFrame(test_y)
    auc = roc_auc_score(test_y, p_value)  # 整体一起算
    ap = average_precision_score(test_y, p_value)
    p_valuen = pd.DataFrame(p_valuen)
    precision_n = precision_score(test_y, p_valuen)
    recall_n = recall_score(test_y, p_valuen)
    f1score_n = f1_score(test_y, p_valuen)
    return auc, ap, precision_n, recall_n, f1score_n

class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size)
        self._init_weights()
        self.activation = nn.ReLU()

    def _init_weights(self):
        init.eye_(self.fc1.weight)
        init.eye_(self.fc2.weight)
        if self.fc1.bias is not None:
            self.fc1.bias.data.zero_()
        if self.fc2.bias is not None:
            self.fc2.bias.data.zero_()

    def forward(self, z):
        x = self.activation(self.fc1(z))
        x = self.activation(self.fc2(x))
        return x

class SubDiscriminator(nn.Module):
    def __init__(self, latent_size, data_size):
        super(SubDiscriminator, self).__init__()
        hidden_size = min(data_size, 1000)
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.fc3 = nn.Linear(10, 1)
        self._init_weights()

    def _init_weights(self):
        # 等价于 keras VarianceScaling(scale=1.0, mode='fan_in', dist='normal')
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc2.bias)
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))


class Detector(nn.Module):
    def __init__(self, latent_size, data_size):
        super(Detector, self).__init__()
        hidden_size = min(data_size, 1000)

        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(10, 1)

        self._init_weights()

    def _init_weights(self):
        # 使用 kaiming_normal_ 初始化，等价于 keras 的 VarianceScaling
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)

        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc2.bias)

        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x


def train_discriminator(model, x_batch, y_batch, optimizer, loss_fn):
    model.train()
    y_batch = y_batch.view(-1, 1)  # 保证 shape 正确
    optimizer.zero_grad()
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_generator(generator, discriminator, optimizer, noise_batch, loss_fn):
    generator.train()
    discriminator.eval()  # 冻结判别器

    optimizer.zero_grad()
    fake_data = generator(noise_batch)
    preds = discriminator(fake_data)
    real_labels = torch.ones_like(preds)  # 欺骗判别器
    loss = loss_fn(preds, real_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_detector(model, x_batch, y_batch, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    outputs = model(x_batch).squeeze(-1)  # shape: (batch_size,)
    y_batch = y_batch.float().view_as(outputs)  # reshape 以匹配预测输出

    loss = loss_fn(outputs, y_batch)
    loss.backward()
    optimizer.step()

    return loss.item()

def get_sample(data, num_samples):
    # 检查数据类型
    if isinstance(data, torch.Tensor):
        # 如果是 Tensor，使用 torch.randint 随机选择行
        indices = torch.randint(0, data.shape[0], (num_samples,))
        return data[indices]
    elif isinstance(data, pd.DataFrame):
        # 如果是 DataFrame，使用 pandas 的 iloc 进行行索引
        indices = torch.randint(0, data.shape[0], (num_samples,))
        return data.iloc[indices.numpy()]  # 将 tensor 转为 numpy 数组用于 iloc
    else:
        raise TypeError("Unsupported data type. Data must be a Tensor or a DataFrame.")

