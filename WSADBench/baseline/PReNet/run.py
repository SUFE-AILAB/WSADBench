import torch
from torch import nn
import numpy as np

from WSADBench.myutils import Utils
from WSADBench.baseline.PReNet.model import prenet
from WSADBench.baseline.PReNet.fit import fit

'''
The unofficial implement (with PyTorch) of the PReNet model in the paper "Deep Weakly-supervised Anomaly Detection"
The default hyper-parameter is the same as in the original paper
'''

class PReNet():
    def __init__(self, seed:int, model_name='PReNet', epochs:int=50, batch_num:int=20, batch_size:int=512,
                act_fun=nn.ReLU(), lr:float=1e-3, weight_decay:float=1e-2,
                s_a_a=8, s_a_u=4, s_u_u=0, norm="none"):

        self.seed = seed
        self.utils = Utils()
        self.device = self.utils.get_device(True)

        # hyper-parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.weight_decay = weight_decay

        self.s_a_a = s_a_a
        self.s_a_u = s_a_u
        self.s_u_u = s_u_u

        self.norm = norm

    def fit(self, X_train, y_train, ratio=None):

        input_size = X_train.shape[1]  # input size
        self.X_train_tensor = torch.from_numpy(X_train).float()  # testing set
        self.y_train = y_train

        self.utils.set_seed(self.seed)
        self.model = prenet(input_size=input_size, act_fun=self.act_fun, norm=self.norm)
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # optimizer

        # training
        fit(X_train_tensor=self.X_train_tensor, y_train=y_train, model=self.model, optimizer=optimizer,
            epochs=self.epochs, batch_num=self.batch_num, batch_size=self.batch_size,
            s_a_a=self.s_a_a, s_a_u=self.s_a_u, s_u_u=self.s_u_u, device=self.device)

        return self


    def predict_score(self, X, num=30):
        self.model = self.model.eval()

        if torch.is_tensor(X):
            pass
        else:
            X = torch.from_numpy(X)

        X = X.float()
        X = X.to(self.device)
        self.X_train_tensor = self.X_train_tensor.to(self.device)

        #修改
        data_size = X.size(0)
        batch_size = 512

        score = []
        for i in range(0,data_size,batch_size):
            end_index = min(i+batch_size, data_size)
            X_batch = X[i:end_index]
            current_batch_size = X_batch.size(0)
            batch_accumulated_scores = torch.zeros(current_batch_size).to(self.device)
            
            for j in range(num):
                index_a = np.random.choice(np.where(self.y_train==1)[0], current_batch_size, replace=True) #postive sample in training set
                index_u = np.random.choice(np.where(self.y_train==0)[0], current_batch_size, replace=True) #negative sample in training set

                X_train_a_tensor = self.X_train_tensor[index_a]
                X_train_u_tensor = self.X_train_tensor[index_u]

                with torch.no_grad():
                    score_a_x = self.model(X_train_a_tensor, X_batch)
                    score_x_u = self.model(X_batch, X_train_u_tensor)

                # 累加得分 (保留样本维度)
                batch_accumulated_scores += (score_a_x + score_x_u)
            #得到每个样本分数
            batch_final_scores = batch_accumulated_scores / (num * 2)
            batch_final_scores = batch_final_scores.cpu().numpy().flatten()

            # entire score
            score.append(batch_final_scores)
        all_scores = np.concatenate(score)

        return all_scores

    def parameter_count(self) -> dict:
        """
        计算模型的参数量
        
        Returns:
            dict: 包含模型参数量的字典
        """
        if not hasattr(self, 'model') or self.model is None:
            # 如果模型未训练，创建临时模型实例来计算参数
            try:
                # 创建临时输入尺寸用于初始化模型
                temp_input_size = 100  # 默认输入尺寸
                
                # 保存当前状态
                current_model = getattr(self, 'model', None)
                
                # 临时初始化模型
                temp_model = prenet(input_size=temp_input_size, act_fun=self.act_fun)
                
                # 计算参数
                params = {"model": sum(p.numel() for p in temp_model.parameters())}
                params["total"] = params["model"]
                
                # 清理临时模型
                del temp_model
                
                # 恢复状态
                if current_model is not None:
                    self.model = current_model
                
                return params
                
            except Exception as e:
                return {"error": f"Failed to count parameters: {str(e)}"}
        else:
            params = {"model": sum(p.numel() for p in self.model.parameters())}
            params["total"] = params["model"]
            return params






