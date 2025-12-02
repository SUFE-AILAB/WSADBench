import torch
import numpy as np


def create_semisupervised_setting(labels,mask, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    """
    #为label noraml 实验添加
    labeldata_idx = np.where(mask==1)[0]  #有标签数据索引（含有标签异常和正常）
    unlabeldata_idx = np.where(mask==0)[0]  #无标签数据索引

    label_normal_idx = np.where((mask==1)&(labels==0))[0]  #有标签正常数据索引
    #额外有标签正常样本
    extra_label_normal = labels[label_normal_idx]
    
    # idx_normal = np.argwhere(np.isin(labels[unlabeldata_idx], normal_classes)).flatten() #可得原始索引
    idx_normal = unlabeldata_idx  #针对无标签样本标签为0的有效
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    # a = np.array([[1, 1, 0, 0],
    #               [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
    #               [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
    #               [0, -ratio_pollution, (1-ratio_pollution), 0]])
    # b = np.array([n_normal, 0, 0, 0])
    # x = np.linalg.solve(a, b)

    # Get number of samples
    # n_known_normal = int(x[0])
    # n_unlabeled_normal = int(x[1])
    # n_unlabeled_outlier = int(x[2])
    # n_known_outlier = int(x[3])
    n_known_normal = int(n_normal * ratio_known_normal)
    n_unlabeled_normal = n_normal - n_known_normal
    n_unlabeled_outlier = int(len(idx_outlier) * ratio_pollution)
    n_known_outlier = len(idx_outlier) - n_unlabeled_outlier
    

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    #拼接有标签正常数据索引
    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()

    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()

    # Get original class labels
    labels_known_normal = np.append(extra_label_normal,labels[idx_known_normal]).tolist()

    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # Get semi-supervised setting labels
    n_label_normal = len(labels_known_normal)
    idx_label_normal = np.concatenate((label_normal_idx, np.array(idx_known_normal).astype(int))).tolist()  #后者为空
    
    semi_labels_known_normal = np.ones(n_label_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_label_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)

    return list_idx, list_labels, list_semi_labels
