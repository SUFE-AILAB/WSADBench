import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from WSADBench.baseline.NTL.model import CustomDataset
from WSADBench.baseline.NTL.models.Losses import DCL
from WSADBench.baseline.NTL.models.NeutralAD_active import ActiveAD_trainer


def fit_ntl(
    train_x,
    train_semi_y,
    model,
    optimizer,
    scheduler,
    epochs,
    batch_size,
    train_method=None,
    query_method=None,
    query_num=None
):
    # train_x: [num, feature_dim], y: [num, 1]
    # 生成query
    # 计算污染率
    rate = sum(train_semi_y)/(len(train_semi_y)- sum(train_semi_y))
    train_data = CustomDataset(torch.from_numpy(train_x).float(), train_semi_y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              drop_last=False)

    trainer = ActiveAD_trainer(model, loss_function=DCL(), epochs=epochs, train_method=train_method, query_method=query_method,)


    trainer.train(train_loader=train_loader,
                  contamination=rate, query_num=query_num,
                  optimizer=optimizer, scheduler=scheduler)
    return trainer


def predict_rosas(model, loss_fun, x_test, device):
    """
    RoSAS模型预测函数

    Args:
        model: 训练好的模型
        x_test: 测试数据
        device: 设备

    Returns:
        anomaly_scores: 异常分数
    """
    model.eval()
    with torch.no_grad():
        xx = torch.from_numpy(x_test).float().to(device)
        xx_s = model(xx)
        loss_n, loss_a = loss_fun(xx_s)

        score = loss_n - loss_a
        score = score.data.cpu().numpy()
    return score
