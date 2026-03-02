import torch
import numpy as np
from torch.nn import functional as F


class EDOSNet(torch.nn.Module):
    """
    RoSAS网络结构：包含嵌入层和评分层
    """

    def __init__(self, n_feature, n_hidden, n_emb, n_hidden2):
        super(EDOSNet, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden, bias=False)
        self.emb_layer = torch.nn.Linear(n_hidden, n_emb, bias=False)

        self.hidden_layer2 = torch.nn.Linear(n_emb, n_hidden2, bias=False)
        self.hidden_layer2_dup = torch.nn.Linear(n_emb, n_hidden2, bias=False)
        self.out_layer = torch.nn.Linear(n_hidden2, 1)

    def forward(self, x, dup=False):
        x = F.leaky_relu(self.hidden_layer(x))
        emb_x = self.emb_layer(x)

        s = F.leaky_relu(self.hidden_layer2(emb_x))
        s = torch.tanh(self.out_layer(s))

        s2 = F.leaky_relu(self.hidden_layer2_dup(emb_x))
        s2 = torch.tanh(self.out_layer(s2))

        if not dup:
            return emb_x, s
        else:
            return emb_x, s, s2


class RoSASLoss(torch.nn.Module):
    """
    RoSAS损失函数：结合三元组损失和Mixup正则化
    """

    def __init__(
            self, l2_reg_weight=0.0, margin=1.0, alpha=1.0, beta=2.0, T=2, k=2, score_loss="smooth", device="cuda"
    ):
        super(RoSASLoss, self).__init__()
        self.l2_reg_weight = l2_reg_weight
        self.loss_tri = torch.nn.TripletMarginLoss(margin=margin)
        self.T = T
        self.alpha = alpha
        self.k = k

        if score_loss == "mse":
            self.loss_reg = torch.nn.MSELoss(reduction="none")
        elif score_loss == "mae":
            self.loss_reg = torch.nn.L1Loss(reduction="none")
        elif score_loss == "smooth":
            self.loss_reg = torch.nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError("unsupported loss")

        self.device = device

    def forward(self, basenet, anchor, pos, neg, pre_emb_loss, pre_score_loss):
        anchor_emb, anchor_s = basenet(anchor)
        pos_emb, pos_s = basenet(pos)
        neg_emb, neg_s = basenet(neg)

        # 嵌入损失
        loss_emb = self.loss_tri(anchor_emb, pos_emb, neg_emb)
        l2_reg = torch.norm(anchor_emb + pos_emb + neg_emb, p=2)

        # Mixup正则化
        if self.k == 2:
            x_i = torch.cat((anchor, pos, neg), 0)
            target_i = torch.cat(
                (torch.ones_like(anchor_s) * -1, torch.ones_like(anchor_s) * -1, torch.ones_like(neg_s)), 0
            )

            indices_j = torch.randperm(x_i.size(0)).to(self.device)
            x_j = x_i[indices_j]
            target_j = target_i[indices_j]

            Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
            lambdas = Beta.sample(target_i.flatten().shape).to(self.device)[:, 1]

            x_tilde = x_i * lambdas.view(lambdas.size(0), 1) + x_j * (1 - lambdas.view(lambdas.size(0), 1))
            _, score_tilde = basenet(x_tilde)

            _, score_xi = basenet(x_i)
            _, score_xj = basenet(x_j)

            score_mix = score_xi * lambdas.view(lambdas.size(0), 1) + score_xj * (1 - lambdas.view(lambdas.size(0), 1))
            y_tilde = target_i * lambdas.view(lambdas.size(0), 1) + target_j * (1 - lambdas.view(lambdas.size(0), 1))
            loss_out = self.loss_reg(score_tilde, y_tilde)
            loss_intra = self.loss_reg(score_tilde, score_mix)

            loss_score = loss_out + loss_intra
            loss_score = loss_score.mean()
            loss_out = loss_out.mean()
            loss_intra = loss_intra.mean()

        else:
            # n-samples mixup
            x_i = torch.cat((anchor, pos, neg), 0)
            target_i = torch.cat(
                (torch.ones_like(anchor_s) * -1, torch.ones_like(anchor_s) * -1, torch.ones_like(neg_s)), 0
            )
            _, score_xi = basenet(x_i)

            x_dup = [x_i]
            target_dup = [target_i]
            score_dup = [score_xi]
            for k in range(1, self.k):
                indices_j = torch.randperm(x_i.size(0)).to(self.device)
                x_j = x_i[indices_j]
                target_j = target_i[indices_j]
                _, score_xj = basenet(x_j)

                x_dup.append(x_j)
                target_dup.append(target_j)
                score_dup.append(score_xj)

            Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
            lambdas_dup = Beta.sample((target_i.flatten().shape[0], self.k)).to(self.device)[:, :, 1]

            s = torch.sum(lambdas_dup, 1).unsqueeze(0).T.repeat(1, self.k)
            lambdas_dup = lambdas_dup / s

            x_tilde = lambdas_dup[:, 0].unsqueeze(0).T * x_i
            y_tilde = lambdas_dup[:, 0].unsqueeze(0).T * target_i
            score_mix = lambdas_dup[:, 0].unsqueeze(0).T * score_xi
            for k in range(1, self.k):
                x_tilde += lambdas_dup[:, k].unsqueeze(0).T * x_dup[k]
                y_tilde += lambdas_dup[:, k].unsqueeze(0).T * target_dup[k]
                score_mix += lambdas_dup[:, k].unsqueeze(0).T * score_dup[k]

            _, score_tilde = basenet(x_tilde)

            loss_out = self.loss_reg(score_tilde, y_tilde)
            loss_intra = self.loss_reg(score_tilde, score_mix)

            loss_score = loss_out + loss_intra
            loss_score = loss_score.mean()
            loss_out = loss_out.mean()
            loss_intra = loss_intra.mean()

        # --动态权重调整  源码------------------------
        # k1 = torch.exp((loss_emb / pre_emb_loss) / self.T) if pre_emb_loss != 0 else torch.tensor(1.0).to(self.device)
        # k2 = (
        #     torch.exp((loss_score / pre_score_loss) / self.T)
        #     if pre_score_loss != 0
        #     else torch.tensor(1.0).to(self.device)
        # )
        # loss = (k1 / (k1 + k2)) * loss_emb + (k2 / (k1 + k2)) * loss_score + self.l2_reg_weight * l2_reg
        # -----------------结束-----------------------

        # ==========================================
        # 修复核心：动态权重调整的数值稳定性处理
        # ==========================================

        # 1. 设置一个极小值 eps，防止分母为 0
        eps = 1e-8

        # 2. 计算指数的输入，并进行截断 (clamp)，防止 exp() 结果溢出为 Inf
        # float32 的 exp(88) 左右就会溢出，我们限制在 [-10, 10] 之间通常足够且安全
        val_emb = (loss_emb / (pre_emb_loss + eps)) / self.T
        val_emb = torch.clamp(val_emb, max=10.0)

        val_score = (loss_score / (pre_score_loss + eps)) / self.T
        val_score = torch.clamp(val_score, max=10.0)

        # 3. 计算 k1, k2
        k1 = torch.exp(val_emb) if pre_emb_loss != 0 else torch.tensor(1.0).to(self.device)
        k2 = torch.exp(val_score) if pre_score_loss != 0 else torch.tensor(1.0).to(self.device)

        # 4. 计算最终 loss，再次加上 eps 防止 k1+k2 为 0 (虽然概率很小)
        loss = (k1 / (k1 + k2 + eps)) * loss_emb + (k2 / (k1 + k2 + eps)) * loss_score + self.l2_reg_weight * l2_reg

        return loss, loss_emb, loss_score, loss_out, loss_intra


class DataGenerator:
    """
    数据生成器：生成三元组训练数据
    """

    def __init__(self, x, y, batch_size=256):
        self.x = x
        self.y = y

        self.anom_idx = np.where(self.y == 1)[0]
        self.anom_x = self.x[self.anom_idx]
        self.norm_idx = np.where(self.y == 0)[0]
        self.norm_x = self.x[self.norm_idx]

        self.batch_size = batch_size

    def load_batches(self, n_batches=10):
        import numpy as np

        batch_set = []

        for i in range(n_batches):
            anom_idx = np.random.choice(len(self.anom_x), self.batch_size)
            anchor_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)
            pos_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)

            batch = [[self.norm_x[a], self.norm_x[p], self.anom_x[n]] for a, p, n in zip(anchor_idx, pos_idx, anom_idx)]
            batch_set.append(batch)
        return np.array(batch_set)