import torch
import time
import numpy as np
from WSADBench.baseline.RoSAS.model import DataGenerator


def fit_rosas(
    train_x,
    train_semi_y,
    model,
    criterion,
    optimizer,
    scheduler,
    epochs,
    nbatch_per_epoch,
    batch_size,
    device,
    prt_step=10,
    verbose=True,
):
    """
    RoSAS模型训练函数

    Args:
        train_x: 训练数据特征
        train_semi_y: 半监督标签
        val_x: 验证数据特征
        val_y: 验证数据标签
        model: 神经网络模型
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数
        nbatch_per_epoch: 每轮的批次数
        batch_size: 批次大小
        device: 设备
        prt_step: 打印步长
        use_es: 是否使用早停
    """

    # 数据生成器
    data_generator = DataGenerator(train_x, train_semi_y, batch_size=batch_size)
    pre_loss_emb, pre_loss_score = 1, 1
    for step in range(epochs):
        start = time.time()

        # 生成批次数据
        batch_triplets = data_generator.load_batches(n_batches=nbatch_per_epoch)
        batch_triplets = torch.from_numpy(batch_triplets).float().to(device)

        losses, losses1, losses2 = [], [], []
        losses_out, losses_intra = [], []

        model.train()
        for batch_triplet in batch_triplets:
            anchor, pos, neg = batch_triplet[:, 0], batch_triplet[:, 1], batch_triplet[:, 2]

            loss, loss1, loss2, loss_out, loss_intra = criterion(model, anchor, pos, neg, pre_loss_emb, pre_loss_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.data.cpu().item())
            losses1.append(loss1.data.cpu().item())
            losses2.append(loss2.data.cpu().item())
            losses_out.append(loss_out.data.cpu().item())
            losses_intra.append(loss_intra.data.cpu().item())

        end = time.time()

        # 打印信息
        t = end - start
        losses, losses1, losses2 = np.array(losses), np.array(losses1), np.array(losses2)
        losses_out, losses_intra = np.array(losses_out), np.array(losses_intra)

        if verbose and (step + 1) % prt_step == 0 or step == 0:
            print(
                f"epoch {step+1}, "
                f"loss (combine/emb/score): {losses.mean():.4f} / {losses1.mean():.4f} / {losses2.mean():.4f}, "
                f"loss (out/intra): {losses_out.mean():.4f} / {losses_intra.mean():.4f}, "
                f"time: {t:.2f}s"
            )

        scheduler.step()
        pre_loss_emb = losses1.mean()
        pre_loss_score = losses2.mean()

    return model


def predict_rosas(model, x_test, device):
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
        _, xx_s = model(xx)
        xx_s = xx_s.flatten()
        xx_s = xx_s.data.cpu().numpy()
    return xx_s
