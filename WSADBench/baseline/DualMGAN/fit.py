import copy
import math

import pandas as pd
import torch
import time
import numpy as np
from sklearn.cluster import KMeans
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from WSADBench.baseline.DualMGAN.model import SubDiscriminator, Detector, Generator, train_discriminator, \
    train_generator, get_sample, train_detector, get_auc
# from WSADBench.baseline.DualMGAN.models.Losses import DCL
# from WSADBench.baseline.DualMGAN.models.NeutralAD_active import ActiveAD_trainer



def to_tensor(x,device):  # array 转 tensor
    return torch.tensor(x, dtype=torch.float32, device=device)

def sample_noise(batch_size, latent_size, device):
    return torch.rand(batch_size, latent_size, device=device)

def fit_dual(
    train_x,
    train_semi_y,
    batch_size,
    device,
    verbose=True,
    args=None
):
    # 拆分成unlabel和ano
    names = locals()
    mask = train_semi_y == 0  # 获取值为 0 的掩码
    data_unl_x = train_x[mask]
    data_out_x = train_x[~mask]
    data_out_y = np.ones(data_out_x.shape[0])
    data_unl_y = np.zeros(data_unl_x.shape[0])
    data_x = np.concatenate((data_out_x, data_unl_x), axis=0)
    data_y = np.concatenate((data_out_y, data_unl_y), axis=0)
    data_out_size = data_out_x.shape[0]
    data_unl_size = data_unl_x.shape[0]
    data_size = data_out_size + data_unl_size
    latent_size = data_x.shape[1]
    batch_size = min(batch_size, data_size)
    mul = math.ceil(data_unl_size / data_out_size) - 1
    # print("The dimensions of the outliers:{}*{}".format(data_out_size, latent_size))
    # print("The dimensions of the unlabeled data:{}*{}".format(data_unl_size, latent_size))
    k_out = min(data_out_size, args.k_means)
    k_unl = min(data_unl_size, args.k_means)
    while True:
    # k-means

        if data_out_size <= args.k_means:
            kmeans_cen_out = data_out_x
        else:  # sk库里的 KMeans
            kmeans_out = KMeans(n_clusters=k_out, random_state=0, max_iter=1000).fit(data_out_x)
            kmeans_cen_out = pd.DataFrame(kmeans_out.cluster_centers_)  # 获取聚类中心
            kmeans_cen_out = kmeans_cen_out.to_numpy()
        kmeans_unl = KMeans(n_clusters=k_unl, random_state=0, max_iter=1000).fit(data_unl_x)
        kmeans_cen_unl = pd.DataFrame(kmeans_unl.cluster_centers_)
        kmeans_cen_unl = kmeans_cen_unl.to_numpy()
        for i in range(k_out):
            names['data_out_x_' + str(i)] = []
            names['data_out_num_x_' + str(i)] = 0
        for idx in range(data_out_size):
            dists_out = np.sqrt(np.sum((data_out_x[idx,] - kmeans_cen_out) ** 2, axis=1))
            index = np.argsort(dists_out)
            for i in range(k_out):
                if index[0] == i:
                    if names['data_out_x_' + str(i)] is None or len(names['data_out_x_' + str(i)]) == 0:
                        names['data_out_x_' + str(i)] = data_out_x[idx,].reshape(1, latent_size)
                    else:
                        names['data_out_x_' + str(i)] = np.concatenate(
                            (names['data_out_x_' + str(i)], data_out_x[idx,].reshape(1, latent_size)), axis=0)
            names['data_out_num_x_' + str(i)] = names['data_out_num_x_' + str(i)] + 1  # 统计每个簇的样本数（但貌似后面没有用到）？
        flag = False
        for i in range(k_out):
            if len(names['data_out_x_' + str(i)]) == 0:
                k_out-=1  # 如果某个簇没有样本，则减少簇的数量
                flag = True

                print("Warning: Cluster {} has no samples, reducing k_out to {}".format(i, k_out))
                break
        if not flag:
            break
        for i in range(k_out+1):
            del names['data_out_x_' + str(i)]  # 删除空的簇
            del names['data_out_num_x_' + str(i)]  # 删除空的簇样本数统计
    for i in range(k_unl):
        names['data_unl_x_' + str(i)] = []
    for idx in range(data_unl_size):
        dists_unl = np.sqrt(np.sum((data_unl_x[idx,] - kmeans_cen_unl) ** 2, axis=1))
        index = np.argsort(dists_unl)
        for i in range(k_unl):
            if index[0] == i:
                if names['data_unl_x_' + str(i)] is None or len(names['data_unl_x_' + str(i)]) == 0:
                    names['data_unl_x_' + str(i)] = data_unl_x[idx,].reshape(1, latent_size)
                else:
                    names['data_unl_x_' + str(i)] = np.concatenate(
                        (names['data_unl_x_' + str(i)], data_unl_x[idx,].reshape(1, latent_size)), axis=0)
    # Create sub-discriminator
    for i in range(k_out):
        names['discriminator_out_' + str(i)] = SubDiscriminator(latent_size, min(data_size, 1000)).to(device)
        names['optimizer_d_out_' + str(i)] = optim.SGD(
            names['discriminator_out_' + str(i)].parameters(),
            lr=args.lr_sd,
            momentum=args.momentum,
            weight_decay=args.decay
        )
    for i in range(k_unl):
        names['discriminator_unl_' + str(i)] = SubDiscriminator(latent_size, min(data_size, 1000)).to(device)
        names['optimizer_d_unl_' + str(i)] = optim.SGD(
            names['discriminator_unl_' + str(i)].parameters(),
            lr=args.lr_sd,
            momentum=args.momentum,
            weight_decay=args.decay
        )
    discriminator_all = Detector(latent_size, min(data_size, 1000)).to(device)
    optimizer_d_all = optim.SGD(
        discriminator_all.parameters(),
        lr=args.lr_d,
        momentum=args.momentum,
        weight_decay=args.decay
    )
    # 创建子生成器 + 对应判别器冻结下的优化器（即 combine_model 训练）
    for i in range(k_out):
        # 子生成器
        names['generator_out_' + str(i)] = Generator(latent_size).to(device)

        # 组合模型：其实就是用 D 作为判别器训练 G，目标是“骗过 D”
        # 判别器已经提前创建好了，参考上一步
        names['optimizer_combine_out_' + str(i)] = optim.SGD(
            names['generator_out_' + str(i)].parameters(),
            lr=args.lr_sg, momentum=args.momentum, weight_decay=args.decay)

    for i in range(k_unl):
        names['generator_unl_' + str(i)] = Generator(latent_size).to(device)

        names['optimizer_combine_unl_' + str(i)] = optim.SGD(
            names['generator_unl_' + str(i)].parameters(),
            lr=args.lr_sg, momentum=args.momentum, weight_decay=args.decay)
    names['loss_fn'] = nn.BCELoss()
    # pre-training the MGAOS
    for i in range(k_out):
        names['stop_out_' + str(i)] = 0
        names['dis_out_' + str(i)] = 0
        names['generated_data_out_all_' + str(i)] = []
        names['nash_out_' + str(i)] = 0
        if names['data_out_x_' + str(i)].shape[0] == 1:
            dists_out = np.sqrt(np.sum((names['data_out_x_' + str(i)] - data_x) ** 2, axis=1))
            index = np.argsort(dists_out)
            names['dis_out_' + str(i)] = dists_out[index[4]]
        elif names['data_out_x_' + str(i)].shape[0] <= 10:
            for idx in range(names['data_out_x_' + str(i)].shape[0]):
                dists_out = np.sum(
                    np.sqrt(np.sum((names['data_out_x_' + str(i)][idx,] - names['data_out_x_' + str(i)]) ** 2, axis=1)),
                    axis=0)
                names['dis_out_' + str(i)] = names['dis_out_' + str(i)] + dists_out
            names['dis_out_' + str(i)] = names['dis_out_' + str(i)] / (
                        names['data_out_x_' + str(i)].shape[0] * (names['data_out_x_' + str(i)].shape[0] - 1))
    stop_out = 0
    my_dist_list = [[] for i in range(k_out)]
    for epoch in tqdm(range(args.max_iter_MGAOS)):  #
        # print('Epoch_out {} of {}'.format(epoch + 1, args.max_iter_MGAOS))
        for i in range(k_out):
            if names['stop_out_' + str(i)] == 0:
                names['data_out_x_' + str(i)] = pd.DataFrame(names['data_out_x_' + str(i)])
                noise = np.random.uniform(0, 1, (int(names['data_out_x_' + str(i)].shape[0]), latent_size))
                # names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)].predict(noise, verbose=0)
                # noise 是一个 shape 为 [batch_size, latent_size] 的 Tensor
                with torch.no_grad():
                    names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)](
                        to_tensor(noise, device)).cpu().numpy()
                names['x_out_' + str(i)] = np.concatenate(
                    (names['data_out_x_' + str(i)], names['generated_data_out_' + str(i)]), axis=0)
                names['y_out_' + str(i)] = np.array([1] * (int(names['data_out_x_' + str(i)].shape[0])) + [0] * (
                    int(names['data_out_x_' + str(i)].shape[0])))
                # names['discriminator_out' + str(i)] = names['discriminator_out_' + str(i)].train_on_batch(
                #     names['x_out_' + str(i)], names['y_out_' + str(i)])
                loss_val = train_discriminator(
                    model=names['discriminator_out_' + str(i)],
                    x_batch=to_tensor(names['x_out_' + str(i)], device),
                    y_batch=to_tensor(names['y_out_' + str(i)], device),
                    optimizer=names['optimizer_d_out_' + str(i)],
                    loss_fn=names['loss_fn']
                )
                # trick_out = np.array([1] * (names['data_out_x_' + str(i)].shape[0]))
                # 冻结判别器，训练生成器
                loss_val_gen = train_generator(
                    generator=names['generator_out_' + str(i)],
                    discriminator=names['discriminator_out_' + str(i)],
                    optimizer=names['optimizer_combine_out_' + str(i)],
                    noise_batch=to_tensor(noise, device),
                    loss_fn=names['loss_fn']
                )
                # if epoch % 10 == 0:
                #     print(f'Epoch_out {epoch + 1}, id:{i} ,loss: {loss_val:.4f}, loss_gen: {loss_val_gen:.4f}')
                # The evaluation of sub-GANs
                if names['data_out_x_' + str(i)].shape[0] == 1:
                    dis = np.linalg.norm(names['generated_data_out_' + str(i)] - names['data_out_x_' + str(i)])
                    my_dist_list[i].append(dis)
                    if dis <= names['dis_out_' + str(i)]:
                        if names['generated_data_out_all_' + str(i)] is None or len(
                                names['generated_data_out_all_' + str(i)]) == 0:
                            names['generated_data_out_all_' + str(i)] = names['generated_data_out_' + str(i)].reshape(1,
                                                                                                                      latent_size)
                        else:
                            names['generated_data_out_all_' + str(i)] = np.concatenate((names[
                                                                                            'generated_data_out_all_' + str(
                                                                                                i)], names[
                                                                                            'generated_data_out_' + str(
                                                                                                i)].reshape(1,
                                                                                                            latent_size)),
                                                                                       axis=0)
                        if names['generated_data_out_all_' + str(i)].shape[0] >= mul:
                            names['stop_out_' + str(i)] = 1
                            stop_out = stop_out + names['stop_out_' + str(i)]
                            break
                        noise = np.random.uniform(0, 1, (mul, latent_size))
                        with torch.no_grad():
                            names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)](
                                to_tensor(noise,device)).cpu().numpy()
                        for idx in range(mul):
                            dis = np.linalg.norm(
                                names['generated_data_out_' + str(i)][idx,] - names['data_out_x_' + str(i)])
                            if dis <= names['dis_out_' + str(i)]:
                                names['generated_data_out_all_' + str(i)] = np.concatenate((names[
                                                                                                'generated_data_out_all_' + str(
                                                                                                    i)], names[
                                                                                                'generated_data_out_' + str(
                                                                                                    i)][idx,].reshape(1,
                                                                                                                      latent_size)),
                                                                                           axis=0)
                                if names['generated_data_out_all_' + str(i)].shape[0] >= mul:
                                    names['stop_out_' + str(i)] = 1
                                    stop_out = stop_out + names['stop_out_' + str(i)]
                                    break
                elif names['data_out_x_' + str(i)].shape[0] <= 10:
                    go_on_gen = 0
                    for idx in range(names['data_out_x_' + str(i)].shape[0]):
                        dis = (np.sum(np.sqrt(
                            np.sum((names['generated_data_out_' + str(i)][idx,] - names['data_out_x_' + str(i)]) ** 2,
                                   axis=1)), axis=0)) / names['data_out_x_' + str(i)].shape[0]
                        my_dist_list[i].append(dis)
                        # if idx
                        # print('sample 2,i:{} dis:{}, need:{}'.format(i,dis, names['dis_out_' + str(i)]))
                        if dis <= names['dis_out_' + str(i)]:
                            go_on_gen = 1
                            if names['generated_data_out_all_' + str(i)] is None or len(
                                    names['generated_data_out_all_' + str(i)]) == 0:
                                names['generated_data_out_all_' + str(i)] = names['generated_data_out_' + str(i)][
                                    idx,].reshape(1, latent_size)
                            else:
                                names['generated_data_out_all_' + str(i)] = np.concatenate((names[
                                                                                                'generated_data_out_all_' + str(
                                                                                                    i)], names[
                                                                                                'generated_data_out_' + str(
                                                                                                    i)][idx,].reshape(1,
                                                                                                                      latent_size)),
                                                                                           axis=0)
                            if names['generated_data_out_all_' + str(i)].shape[0] >= \
                                    names['data_out_x_' + str(i)].shape[0] * mul:
                                names['stop_out_' + str(i)] = 1
                                stop_out = stop_out + names['stop_out_' + str(i)]
                                break
                    if go_on_gen == 1:
                        noise = np.random.uniform(0, 1, (mul * names['data_out_x_' + str(i)].shape[0], latent_size))
                        # names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)].predict(noise,
                        #                                                                                  verbose=0)
                        with torch.no_grad():
                            names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)](
                                to_tensor(noise,device)).cpu().numpy()
                        for idx in range(names['data_out_x_' + str(i)].shape[0] * mul):
                            dis = (np.sum(np.sqrt(np.sum(
                                (names['generated_data_out_' + str(i)][idx,] - names['data_out_x_' + str(i)]) ** 2,
                                axis=1)), axis=0)) / names['data_out_x_' + str(i)].shape[0]
                            if dis <= names['dis_out_' + str(i)]:
                                names['generated_data_out_all_' + str(i)] = np.concatenate((names[
                                                                                                'generated_data_out_all_' + str(
                                                                                                    i)], names[
                                                                                                'generated_data_out_' + str(
                                                                                                    i)][idx,].reshape(1,
                                                                                                                      latent_size)),
                                                                                           axis=0)
                                if names['generated_data_out_all_' + str(i)].shape[0] >= \
                                        names['data_out_x_' + str(i)].shape[0] * mul:
                                    names['stop_out_' + str(i)] = 1
                                    stop_out = stop_out + names['stop_out_' + str(i)]
                                    break
                else:
                    names['generated_data_out_' + str(i)] = pd.DataFrame(names['generated_data_out_' + str(i)])
                    sample_num = min(20, names['data_out_x_' + str(i)].shape[0])
                    # names['eva_nash_data_out_' + str(i)] = names['data_out_x_' + str(i)].sample(sample_num,
                    #                                                                             replace=False,
                    #                                                                             random_state=None,
                    #                                                                             axis=0)
                    names['eva_nash_data_out_' + str(i)] = get_sample(names['data_out_x_' + str(i)], sample_num, )
                    names['eva_nash_data_out_' + str(i)] = names['eva_nash_data_out_' + str(i)].to_numpy()
                    names['Nnr_MGAOS_' + str(i)] = 0
                    for idx in range(sample_num):
                        real = 0
                        dis = np.sqrt(
                            np.sum((names['eva_nash_data_out_' + str(i)][idx,] - names['x_out_' + str(i)]) ** 2,
                                   axis=1))
                        dis = pd.DataFrame(dis)
                        names['y_out_' + str(i)] = pd.DataFrame(names['y_out_' + str(i)])
                        dis = np.concatenate((dis, names['y_out_' + str(i)]), axis=1)
                        dis = pd.DataFrame(dis, columns=['d', 'y'])
                        dis = dis.sort_values('d', ascending=True)
                        dis = dis.to_numpy()
                        for index in range(sample_num):
                            if dis[index, 1] == 0:
                                real = real + 1
                        nnr = real / sample_num
                        if nnr > args.nnr_MGAOS:
                            names['Nnr_MGAOS_' + str(i)] = names['Nnr_MGAOS_' + str(i)] + 1
                    names['Nnr_MGAOS_' + str(i)] = names['Nnr_MGAOS_' + str(i)] / sample_num
                    if names['Nnr_MGAOS_' + str(i)] > args.nnr_MGAOS:
                        names['stop_out_' + str(i)] = 1
                        stop_out = stop_out + 1
        if stop_out == k_out:
            break
    new_data_out_size = 0
    for i in range(k_out):
        if names['data_out_x_' + str(i)].shape[0] <= 10:
            if names['generated_data_out_all_' + str(i)] is None or len(names['generated_data_out_all_' + str(i)]) == 0:
                names['new_data_out_x_' + str(i)] = names['data_out_x_' + str(i)]
            else:
                names['new_data_out_x_' + str(i)] = np.concatenate(
                    (names['generated_data_out_all_' + str(i)], names['data_out_x_' + str(i)]), axis=0)
            if names['new_data_out_x_' + str(i)].shape[0] < names['data_out_x_' + str(i)].shape[0] * (mul + 1):
                names['new_data_out_x_' + str(i)] = pd.DataFrame(names['new_data_out_x_' + str(i)])

                names['new_data_out_x_' + str(i)] = get_sample(names['data_out_x_' + str(i)], math.ceil(
                    names['data_out_x_' + str(i)].shape[0] * (mul + 1)))
        else:
            if names['stop_out_' + str(i)] == 1:
                noise = np.random.uniform(0, 1, (int(names['data_out_x_' + str(i)].shape[0] * mul), latent_size))
                with torch.no_grad():
                    names['generated_data_out_all_' + str(i)] = names['generator_out_' + str(i)](
                        to_tensor(noise,device)).cpu().numpy()
                # names['generated_data_out_all_' + str(i)] = names['generator_out_' + str(i)].predict(noise, verbose=0)
                names['new_data_out_x_' + str(i)] = np.concatenate(
                    (names['generated_data_out_all_' + str(i)], names['data_out_x_' + str(i)]), axis=0)
            else:
                names['data_out_x_' + str(i)] = pd.DataFrame(names['data_out_x_' + str(i)])
                names['new_data_out_x_' + str(i)] = get_sample(names['data_out_x_' + str(i)], math.ceil(
                    names['data_out_x_' + str(i)].shape[0] * (mul + 1)))
                # names['new_data_out_x_' + str(i)] = names['data_out_x_' + str(i)].sample(
                #     n=math.ceil(names['data_out_x_' + str(i)].shape[0] * (mul + 1)), replace=True, random_state=None,
                #     axis=0)
        new_data_out_size = new_data_out_size + names['new_data_out_x_' + str(i)].shape[0]
    # Start iteration
    for i in range(k_unl):
        names['stop_unl_' + str(i)] = 0
        if names['data_unl_x_' + str(i)].shape[0] == 1:
            names['change_' + str(i)] = 0
            dists_unl = np.sqrt(np.sum((names['data_unl_x_' + str(i)] - data_x) ** 2, axis=1))
            index = np.argsort(dists_unl)
            names['dis_unl_' + str(i)] = dists_unl[index[4]]
    for epoch in tqdm(range(args.max_iter_MGAAL)):
        # print('Epoch {} of {}'.format(epoch + 1, args.max_iter_MGAAL))

        # Sample mini-batch date
        for i in range(k_out):
            names['new_data_out_x_' + str(i)] = pd.DataFrame(names['new_data_out_x_' + str(i)])
            names['data_out_batch_x_' + str(i)] = get_sample(names['new_data_out_x_' + str(i)], math.ceil(
                names['new_data_out_x_' + str(i)].shape[0] * (batch_size / 2) / new_data_out_size))
            # names['data_out_batch_x_' + str(i)] = names['new_data_out_x_' + str(i)].sample(n=math.ceil(names['new_data_out_x_' + str(i)].shape[0] * (batch_size/2) / new_data_out_size), replace=False, random_state=None, axis=0)
        for i in range(k_unl):
            names['data_unl_x_' + str(i)] = pd.DataFrame(names['data_unl_x_' + str(i)])
            batch_num = math.ceil(names['data_unl_x_' + str(i)].shape[0] * (batch_size / 2) / data_unl_size)
            if batch_num == 1:
                batch_num = 2
            names['data_unl_batch_x_' + str(i)] = get_sample(names['data_unl_x_' + str(i)], batch_num)
            # batch太少
            # print(f"data_unl_x_{i} shape: {math.ceil(names['data_unl_x_' + str(i)].shape[0] * (batch_size / 2) / data_unl_size)}")
            # names['data_unl_batch_x_' + str(i)] = names['data_unl_x_' + str(i)].sample(n=math.ceil((names['data_unl_x_' + str(i)].shape[0] * (batch_size/2)) / data_unl_size), replace=False, random_state=None, axis=0)
        best_auc = 0
        best_model = None
        # Train sub-generators and sub-discriminators
        for i in range(k_unl):
            names['data_unl_batch_x_' + str(i)] = pd.DataFrame(names['data_unl_batch_x_' + str(i)])
            if names['stop_unl_' + str(i)] == 0:
                noise = np.random.uniform(0, 1, (int(names['data_unl_batch_x_' + str(i)].shape[0]), latent_size))
                with torch.no_grad():
                    names['generated_data_unl_' + str(i)] = names['generator_unl_' + str(i)](
                        to_tensor(noise,device)).cpu().numpy()
                # names['generated_data_unl_' + str(i)] = names['generator_unl_' + str(i)].predict(noise, verbose=0)
                names['x_unl_' + str(i)] = np.concatenate(
                    (names['data_unl_batch_x_' + str(i)], names['generated_data_unl_' + str(i)]), axis=0)
                names['y_unl_' + str(i)] = np.array([1] * (int(names['data_unl_batch_x_' + str(i)].shape[0])) + [0] * (
                    int(names['data_unl_batch_x_' + str(i)].shape[0])))
                # 训练子判别器
                loss_val = train_discriminator(
                    model=names['discriminator_unl_' + str(i)],
                    x_batch=to_tensor(names['x_unl_' + str(i)],device),
                    y_batch=to_tensor(names['y_unl_' + str(i)],device),
                    optimizer=names['optimizer_d_unl_' + str(i)],
                    loss_fn=names['loss_fn']
                )
                # names['discriminator_unl' + str(i)] = names['discriminator_unl_' + str(i)].train_on_batch(names['x_unl_' + str(i)], names['y_unl_' + str(i)])
                trick_unl = np.array([1] * (int(names['data_unl_batch_x_' + str(i)].shape[0])))

                # names['generator_unl' + str(i)] = names['combine_model_unl_' + str(i)].train_on_batch(noise, trick_unl)
                loss_val_gen = train_generator(
                    generator=names['generator_unl_' + str(i)],
                    discriminator=names['discriminator_unl_' + str(i)],
                    optimizer=names['optimizer_combine_unl_' + str(i)],
                    noise_batch=to_tensor(noise,device),
                    loss_fn=names['loss_fn']
                )
                # The evaluation of sub-GANs
                names['generated_data_unl_' + str(i)] = pd.DataFrame(names['generated_data_unl_' + str(i)])
                sample_num = min(20, names['data_unl_batch_x_' + str(i)].shape[0])
                # if sample_num == 1:  # 尝试加1
                #     sample_num = 2
                # names['eva_nash_data_unl_' + str(i)] = names['data_unl_batch_x_' + str(i)].sample(sample_num, replace=False, random_state=None, axis=0)
                names['eva_nash_data_unl_' + str(i)] = get_sample(names['data_unl_batch_x_' + str(i)], sample_num)
                names['eva_nash_data_unl_' + str(i)] = names['eva_nash_data_unl_' + str(i)].to_numpy()
                names['Nnr_unl_' + str(i)] = 0
                if sample_num >= 2:
                    for idx in (range(sample_num)):
                        real = 0
                        dis = np.sqrt(
                            np.sum((names['eva_nash_data_unl_' + str(i)][idx,] - names['x_unl_' + str(i)]) ** 2,
                                   axis=1))
                        dis = pd.DataFrame(dis)
                        names['y_unl_' + str(i)] = pd.DataFrame(names['y_unl_' + str(i)])
                        dis = np.concatenate((dis, names['y_unl_' + str(i)]), axis=1)
                        dis = pd.DataFrame(dis, columns=['d', 'y'])
                        dis = dis.sort_values('d', ascending=True)
                        dis = dis.to_numpy()
                        for index in range(sample_num):
                            if dis[index, 1] == 0:
                                real = real + 1
                        nnr = real / sample_num
                        if nnr > args.nnr_MGAAL:
                            names['Nnr_unl_' + str(i)] = names['Nnr_unl_' + str(i)] + 1
                    names['Nnr_unl_' + str(i)] = names['Nnr_unl_' + str(i)] / sample_num
                    if names['Nnr_unl_' + str(i)] > args.nnr_MGAAL:
                        names['stop_unl_' + str(i)] = 1
                    # print(
                    #     "The {}th subset contains {} samples, the evaluation of the sub-GAN is {}".format(i, sample_num,
                    #                                                                                       names[
                    #                                                                                           'Nnr_unl_' + str(
                    #                                                                                               i)]))
                else:
                    dis = np.linalg.norm(names['eva_nash_data_unl_' + str(i)] - names['generated_data_unl_' + str(i)])
                    if dis <= names['dis_unl_' + str(i)]:  # bug:key Error
                        names['change_' + str(i)] = names['change_' + str(i)] + 1
                        if names['change_' + str(i)] > 5:
                            names['stop_unl_' + str(i)] = 1
                    # print(
                    #     "The {}th subset contains {} samples, the evaluation of the sub-GAN is {}".format(i, sample_num,
                    #                                                                                       names[
                    #                                                                                           'change_' + str(
                    #                                                                                               i)]))

        # Train the detector
        for i in range(k_out):
            if i == 0:
                data_out_batch = names['data_out_batch_x_' + str(i)]
            else:
                data_out_batch = np.concatenate((data_out_batch, names['data_out_batch_x_' + str(i)]), axis=0)
        for i in range(k_unl):
            if i == 0:
                data_unl_batch = names['data_unl_batch_x_' + str(i)]
            else:
                data_unl_batch = np.concatenate((data_unl_batch, names['data_unl_batch_x_' + str(i)]), axis=0)
        for i in range(k_unl):
            noise_all = np.random.uniform(0, 1, (math.ceil(batch_size / (2 * k_unl)), latent_size))
            with torch.no_grad():
                names['generated_data_unl_all_' + str(i)] = names['generator_unl_' + str(i)](
                    to_tensor(noise_all,device)).cpu().numpy()
            # names['generated_data_unl_all_' + str(i)] = names['generator_unl_' + str(i)].predict(noise_all, verbose=0)
            if i == 0:
                x_unl_all = np.concatenate((data_unl_batch, names['generated_data_unl_all_' + str(i)]), axis=0)
            else:
                x_unl_all = np.concatenate((x_unl_all, names['generated_data_unl_all_' + str(i)]), axis=0)

        x_all = np.concatenate((x_unl_all, data_out_batch), axis=0)
        y_all = np.array([1] * (int(data_unl_batch.shape[0])) + [0] * (x_all.shape[0] - data_unl_batch.shape[0]))
        # discriminator_all.train_on_batch(x_all, y_all)
        # 调用训练函数
        loss_val = train_detector(
            model=discriminator_all,
            x_batch=to_tensor(x_all,device),
            y_batch=to_tensor(y_all,device),
            optimizer=optimizer_d_all,
            loss_fn=nn.BCELoss()
        )
        # if epoch % 10 == 0:
        # print(f"Epoch {epoch}, Loss: {loss_val:.4f}")
        # The selection of optimal model(auc)
        with torch.no_grad():
            p_value = 1 - discriminator_all(to_tensor(data_x,device)).cpu().numpy()
        auc = get_auc(p_value, data_out_x, data_unl_x)
        if auc > best_auc:
            best_auc = auc
            best_model = copy.deepcopy(discriminator_all)
    return best_model

def predict(model,  x_test, device):
    model.eval()
    with torch.no_grad():
        xx = torch.from_numpy(x_test).float().to(device)
        score = 1- model(xx)
        score = score.data.cpu().numpy()
    return score
