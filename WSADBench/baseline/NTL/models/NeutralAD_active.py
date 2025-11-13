
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from WSADBench.baseline.NTL.model import CustomDataset
from sklearn.neighbors import KernelDensity
from .Query_strategies import *
from sklearn.metrics import precision_recall_fscore_support
from KDEpy import NaiveKDE

# def estimate_ratio(y_score,y_true,active_idx):
#     p = NaiveKDE(kernel='gaussian', bw='silverman').fit(y_score).evaluate(y_score[active_idx])
#     q = NaiveKDE(kernel='gaussian', bw='silverman').fit(y_score[active_idx]).evaluate(y_score[active_idx])
#     ratio = (y_true[active_idx]*p/q).mean()
#     return ratio
def compute_pre_recall_f1(target, score):
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')

    return f1
def estimate_ratio(y_score,y_true,active_idx):
    bw = (y_score.max()-y_score.min())/len(active_idx)
    kde_p = KernelDensity(kernel="gaussian", bandwidth=bw).fit(y_score[:, np.newaxis])
    kde_q = KernelDensity(kernel="gaussian", bandwidth=bw).fit(y_score[active_idx, np.newaxis])
    log_p = kde_p.score_samples(y_score[active_idx, np.newaxis])
    log_q = kde_q.score_samples(y_score[active_idx, np.newaxis])
    ratio = np.mean(np.exp(log_p - log_q) * y_true[active_idx])
    return ratio

class ActiveAD_trainer:

    def __init__(self, model, loss_function, epochs, train_method='loe_est', query_method='kmeans'):

        self.loss_fun = loss_function
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.train_method = train_method
        self.query_method = query_method
        self.max_epochs = epochs

    def _train_initial(self, optimizer,unknown_loader):
        self.model.train()
        for data in unknown_loader:
            samples = data['sample']
            z = self.model(samples)
            loss_n, _ = self.loss_fun(z)
            loss = loss_n
            loss_mean = loss.mean()
            self.model.zero_grad()
            loss_mean.backward()
            optimizer.step()

    def _train_gt(self, optimizer,unknown_loader):
        self.model.train()
        for data in unknown_loader:
            samples = data['sample']
            labels = data['label'].float().to(self.device)
            z_p = self.model(samples)
            loss_n, loss_a = self.loss_fun(z_p)
            loss = loss_n*(1-labels)+loss_a*labels
            loss_mean = loss.mean()

            self.model.zero_grad()
            loss_mean.backward()
            optimizer.step()

        loss_mean = loss_mean.item()
        return loss_mean

    def _train_sup(self, optimizer,query_data):

        self.model.train()
        samples = query_data.samples
        labels = torch.from_numpy(query_data.labels).float().to(self.device)
        z = self.model(samples)
        loss_n, loss_a = self.loss_fun(z)
        loss = ((1. - labels) * loss_n + labels * loss_a)
        loss_mean = loss.mean()
        self.model.zero_grad()
        loss_mean.backward()
        optimizer.step()

        return loss_mean.item()

    def _train_oc(self, optimizer,unknown_loader,query_data):
        self.model.train()
        for data in unknown_loader:
            samples = query_data.samples
            labels = torch.from_numpy(query_data.labels).float().to(self.device)
            z_q= self.model(samples)
            loss_n, loss_a = self.loss_fun(z_q)
            loss = ((1. - labels) * loss_n + labels * loss_a)
            loss_sup = loss.mean()

            samples = data['sample']
            z = self.model(samples)
            loss_n, loss_a = self.loss_fun(z)
            loss_mean = loss_n.mean()+loss_sup

            self.model.zero_grad()
            loss_mean.backward()
            optimizer.step()

        loss_mean = loss_mean.item()
        return loss_mean

    def _train_loe(self, optimizer,unknown_loader,query_data,ratio):
        self.model.train()
        for data in unknown_loader:
            samples = query_data.samples
            labels = torch.from_numpy(query_data.labels).float().to(self.device)
            z_q= self.model(samples)
            loss_n, loss_a = self.loss_fun(z_q)
            loss = ((1. - labels) * loss_n + labels * loss_a)
            loss_sup = loss.mean()
            self.model.zero_grad()
            loss_sup.backward()
            optimizer.step()

            samples = data['sample']
            z = self.model(samples)
            loss_n, loss_a = self.loss_fun(z)

            score = loss_n-loss_a
            # print(f'need:{int(score.shape[0] * (1 - ratio))}, limit:{score.shape[0] - 1}')
            # k = int(score.shape[0] * (1 - ratio))
            # k = min(k, )  # 防止越界
            neg_num = int(score.shape[0] * (1 - ratio))
            pos_num = int(score.shape[0] * ratio)
            # 加个约束，防止个数越上下界
            # 加个约束，防止个数越上下界
            neg_num = max(1, min(neg_num, score.shape[0] - 1))  # 至少1个，最多N-1个
            pos_num = max(1, min(pos_num, score.shape[0] - neg_num))  # 至少1个，且不超过剩下的

            _, idx_neg = torch.topk(score, neg_num, largest=False,
                                                 sorted=False)
            _, idx_pos = torch.topk(score, pos_num, largest=True, sorted=False)
            loss = torch.cat([loss_n[idx_neg], 0.5 * loss_a[idx_pos] + 0.5 * loss_n[idx_pos]], 0)
            loss_mean = loss.mean()

            self.model.zero_grad()
            loss_mean.backward()
            optimizer.step()

        loss_mean = loss_mean.item()
        return loss_mean

    def detect_outliers(self, loader):  # 离群值
        self.model.eval()
        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                samples = data['sample']
                labels = data['label']
                z = self.model(samples)
                loss_n,loss_a = self.loss_fun(z)
                score = loss_n
                loss_in += loss_n[labels == 0].sum()
                loss_out += loss_n[labels == 1].sum()
                target_all.append(labels)
                score_all.append(score.cpu().numpy())

        score_all = np.concatenate(score_all)
        target_all = np.concatenate(target_all)
        auc = roc_auc_score(target_all, score_all)
        f1 = compute_pre_recall_f1(target_all, score_all)
        ap = average_precision_score(target_all, score_all)
        return auc, ap, f1, score_all,loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum()

    def compute_scores(self, train_data):
        self.model.eval()
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        ad_scores = []
        for data in loader:
            with torch.no_grad():
                samples = data['sample']
                z = self.model(samples)
                loss_neg, loss_pos = self.loss_fun(z)
                score = loss_neg
                ad_scores.append(score)
        ad_scores = torch.cat(ad_scores, 0)
        return ad_scores

    def get_embs(self, train_data):

        self.model.eval()
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        embs = []
        for data in loader:
            with torch.no_grad():
                samples = data['sample']
                z = self.model(samples)
                embs.append(z.reshape(z.shape[0],-1))
        embs = torch.cat(embs,0)
        return embs

    def query_samples(self,train_data, active_rand_idx):
        query_samples = train_data.samples[active_rand_idx]
        query_labels = train_data.labels[active_rand_idx]
        query_data = CustomDataset(query_samples,query_labels)

        num_known_pos = train_data.labels[active_rand_idx].sum()
        msg = f'Number of queried anomalies / Total query samples: {num_known_pos}/{query_labels.shape[0]}'
        if self.logger is not None:
            self.logger.log(msg)
            print(msg)
        else:
            print(msg)

        unknown_idx = np.delete(np.arange(train_data.samples.shape[0]),active_rand_idx)
        unknown_samples = train_data.samples[unknown_idx]
        unknown_labels = train_data.labels[unknown_idx]
        unknown_data = CustomDataset(unknown_samples,unknown_labels)

        return query_data,unknown_data


    def train(self, train_loader, contamination, query_num=0,optimizer=None, scheduler=None):

        # early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, = -1, -1
        test_auc, test_f1, test_score = None, None, None,
        torch.cuda.empty_cache()
        self.logger = None
        # self.contamination_rate = contamination
        # self.anomal_label = 0.5
        self.num_data = len(train_loader.dataset)
        K = int(query_num)  # for active learning
        self.batch_size = train_loader.batch_size

        num_data = len(train_loader.dataset)
        msg = f'num data: {num_data}'
        if self.logger is not None:
            self.logger.log(msg)
            print(msg)
        else:
            print(msg)

        if self.train_method == 'gt':
            pass
        else:
            self._train_initial(optimizer,train_loader)
            scores = self.compute_scores(train_loader.dataset)
            if self.query_method == 'kmeans':
                embs = self.get_embs(train_loader.dataset)
                # active_rand_idx = kmeans_diverse(embs, K, labels = train_loader.dataset.labels)
                # 恢复成旧 的提问
                print(f'恢复成旧 的提问')
                active_rand_idx = kmeans_diverse_v1(embs, K,)
                print("Assign K=%d true labels under kmeans++ strategy " % K)
            elif self.query_method == 'random':
                active_rand_idx = torch.randperm(len(train_loader.dataset))[:K]
                active_rand_idx = active_rand_idx.cpu()
                print("Randomly assign K=%d true labels" % K)
            elif self.query_method == 'positive':
                _, active_rand_idx = torch.topk(scores, K, largest=True, sorted=False)
                active_rand_idx = active_rand_idx.cpu()
                print("Assign K=%d true labels under most-positive strategy " % K)
            elif self.query_method == 'pos_random':
                active_rand_idx = pos_random(scores,K)
                print("Assign K=%d true labels under positive-random strategy " % K)
            elif self.query_method == 'pos_diverse':
                embs = self.get_embs(train_loader.dataset)
                active_rand_idx = pos_diverse(scores,embs, K)
                print("Assign K=%d true labels under pos_diverse strategy " % K)
            elif self.query_method == 'margin':
                active_rand_idx = margin(scores,K,contamination)
                print("Assign K=%d true labels under margin strategy " % K)
            elif self.query_method == 'mar_diverse':
                embs = self.get_embs(train_loader.dataset)
                active_rand_idx = margin_diverse(scores,embs, K,contamination)
                print("Assign K=%d true labels under margin_diverse strategy " % K)

            query_data,unknown_data = self.query_samples(train_loader.dataset, active_rand_idx)
            unknown_loader = DataLoader(unknown_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
            if self.train_method=='loe_true':
                ratio = (contamination*len(train_loader.dataset)-train_loader.dataset.labels[active_rand_idx].sum())/(
                    len(train_loader.dataset)-len(active_rand_idx))
            elif self.train_method=='loe_est':
                ratio = estimate_ratio(scores.cpu().numpy(), train_loader.dataset.labels, active_rand_idx)
                # ratio_error = np.abs(ratio-contamination)
                msg = f'estimated ratio / true ratio: {ratio}/{contamination}'
                if self.logger is not None:
                    self.logger.log(msg)
                    print(msg)
                else:
                    print(msg)
                ratio = (ratio * len(train_loader.dataset) - train_loader.dataset.labels[active_rand_idx].sum()) / (
                        len(train_loader.dataset) - len(active_rand_idx))

        for epoch in tqdm(range(1, self.max_epochs + 1)):
            if self.train_method == 'sup':
                for _ in range(np.ceil(num_data/self.batch_size).astype(int)):
                    train_loss = self._train_sup(optimizer,query_data)
            elif self.train_method == 'gt':
                train_loss = self._train_gt(optimizer,train_loader)
            elif self.train_method == 'oc':
                train_loss = self._train_oc(optimizer,unknown_loader,query_data)
            else:
                train_loss = self._train_loe(optimizer, unknown_loader, query_data,ratio)

            if scheduler is not None:
                scheduler.step()

            # if test_loader is not None:
            #     test_auc, test_ap,test_f1, test_score, testin_loss, testout_loss = self.detect_outliers(test_loader)
            #
            # if validation_loader is not None:
            #     val_auc, val_ap,val_f1, _, valin_loss, valout_loss = self.detect_outliers(validation_loader)
            #
            #     if early_stopper is not None and early_stopper.stop(epoch, valin_loss, val_auc, testin_loss,
            #                                                         test_auc, test_ap, test_f1,
            #                                                         test_score,train_loss):
            #         break

            if epoch % 2 == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss} '

                if self.logger is not None:
                    self.logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        # if early_stopper is not None:
        #     train_loss, val_loss, val_auc, test_loss, test_auc, test_ap, test_f1, test_score, best_epoch \
        #         = early_stopper.get_best_vl_metrics()
        #
        #     msg = f'Stopping at epoch {best_epoch}, TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc} ,' \
        #           f'TS loss: {test_loss}, TS auc: {test_auc} TS ap: {test_ap} TS f1: {test_f1}'
        #     if logger is not None:
        #         logger.log(msg)
        #         print(msg)
        #     else:
        #         print(msg)


        return  self.model