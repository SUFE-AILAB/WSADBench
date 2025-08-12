
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist

def kmeans_diverse(embs, K,tau=0.01, labels=None):  # 没有label
    pos_indices = np.where(labels == 1)[0]  # 找出 label=1 的索引  异常是一模一样的？
    normal_indices = np.where(labels == 0)[0]  # 找出 label=0 的索引
    embs_pos = embs[pos_indices]  # 筛选出对应的 embedding
    embs_normal = embs[normal_indices]  # 筛选出正常样本的 embedding
    def get_embs(embs_pos, indices,K):  # 锁K
        idx_active = []
        dist_matrix = torch.cdist(embs_pos, embs_pos, p=2).cpu().numpy() # 全是nan
        dist_matrix = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min() + 1e-6)  # 避免除以0(所有的pos都一样，导致min和max相等)
        dist_matrix = dist_matrix.astype(np.float64)
        dist_matrix = np.exp(dist_matrix / tau)
        idx_ = np.argmin(np.mean(dist_matrix, 0))
        # idx_ = np.random.choice(np.arange(embs.shape[0]),1)[0]
        idx_active.append(idx_)

        while len(idx_active) < K:
            p = dist_matrix[idx_active].min(0)
            p = p / p.sum()
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(p)), p))
            idx_ = customDist.rvs(size=1)[0]
            while idx_ in idx_active: idx_ = customDist.rvs(size=1)[0]
            idx_active.append(idx_)
        return indices[idx_active].tolist()
    def check(x):

        # 找出唯一行和每行的索引映射
        unique_rows, inverse_indices, counts = torch.unique(x, dim=0, return_inverse=True, return_counts=True)

        # 找出重复的行索引（重复行的原始索引）
        duplicated_mask = counts[inverse_indices] > 1
        print(f'总行数：{len(x)},  重复行数：{duplicated_mask.sum().item()}')
        print("重复行索引:", torch.nonzero(duplicated_mask).squeeze())
        print("重复的行:")
        print(x[duplicated_mask])
        pass

    # check(embs_pos)
    if len(embs_pos) < K:  # 修正策略为，加入正样本
        # K = len(embs_pos)  # 确保 K 不超过正样本的数量
        print(f"error:############个数不足，K缩减为:{K}#############")
        normal_embs = get_embs(embs_normal, normal_indices, K - len(embs_pos))  # 获取正常样本的索引
        pos_indices = pos_indices.tolist() + normal_embs  # 合并正样本和正常样本的索引
        return pos_indices  # 不用筛了。。
    else:
        return get_embs(embs_pos, pos_indices, K)

def pos_diverse(scores,embs, K):
    def min_max_normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    idx_active = []

    scores = min_max_normalize(scores)
    most_pos_idx = torch.argmax(scores).item()
    idx_active.append(most_pos_idx)
    scores[most_pos_idx] = 0.

    dist_matrix = torch.cdist(embs, embs, p=2)
    dist_matrix = min_max_normalize(dist_matrix)
    K -= 1
    for _ in range(K):
        dist_to_active = dist_matrix[idx_active].min(0)[0]
        # print(dist_to_active)
        score = scores + dist_to_active
        idx = torch.argmax(score).item()
        idx_active.append(idx)
        scores[idx] = 0.

    return torch.tensor(idx_active)

def margin_diverse(scores,embs, K,contamination):
    num_knn = int(np.ceil(scores.shape[0]/K))
    dist_matrix = torch.cdist(embs, embs, p=2)
    nearest_dist, _ = torch.topk(dist_matrix, num_knn, largest=False,
                              sorted=False)
    nearest_dist = nearest_dist.cpu().numpy()
    anchor_dist = np.max(nearest_dist,1,keepdims=True)
    score_pos, _ = torch.topk(scores, int(scores.shape[0] * contamination), largest=True,
                              sorted=False)
    anchor_score = score_pos.min()
    boundary_score = torch.abs(scores - anchor_score).cpu().numpy()
    boundary_score = (boundary_score-boundary_score.min())/(boundary_score.max()-boundary_score.min())
    idx_ = np.argmin(boundary_score)
    idx_active = [idx_]
    embs = embs.cpu().numpy()
    for _ in range(K-1):
        score = 0.5+(anchor_dist>=cdist(embs,embs[idx_active],'euclidean')).sum(1)/(2*num_knn)
        score = score+boundary_score
        score[idx_active] = 1e3
        idx_ = np.argmin(score)
        idx_active.append(idx_)
    return idx_active

def margin(scores,K,contamination):
    score_pos, _ = torch.topk(scores, int(scores.shape[0] * contamination), largest=True,
                              sorted=False)
    anchor_score = score_pos.min()
    _, idx_active = torch.topk(torch.abs(scores - anchor_score), K, largest=False, sorted=False)
    idx_active = idx_active.cpu()
    return idx_active

def pos_random(scores,K):
    _, idx_active = torch.topk(scores, int(scores.shape[0] / 2), largest=True, sorted=False)
    perm = torch.randperm(idx_active.size(0))
    idx_active = idx_active[perm[:K]]
    idx_active = idx_active.cpu()
    return idx_active
