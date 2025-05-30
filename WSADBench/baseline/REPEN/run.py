import numpy as np
from WSADBench.baseline.REPEN.model import repen
from WSADBench.myutils import Utils
import os
from keras import backend as K

# we change the training epochs to 1000 since we find that the default setting (epochs=30) cannot guarantee
class REPEN():
    def __init__(self, seed, model_name='REPEN', save_suffix='test',
                 mode:str='supervised', hidden_dim:int=20, batch_size:int=256, nb_batch:int=50, n_epochs:int=1000):
        self.utils = Utils()
        self.device = self.utils.get_device(True)  # get device
        self.seed = seed

        self.MAX_INT = np.iinfo(np.int32).max
        self.MAX_FLOAT = np.finfo(np.float32).max

        # self.sess = tf.Session()
        # K.set_session(self.sess)

        # hyper-parameters
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.n_epochs = n_epochs

        self.save_suffix = save_suffix
        self.modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        if not os.path.exists(self.modelpath):
            os.makedirs(self.modelpath)

    def fit(self, X_train, y_train, ratio=None):
        # initialization the network
        self.utils.set_seed(self.seed)

        # change the model type when no label information is available
        if sum(y_train) == 0:
            self.mode = 'unsupervised'

        # model initialization
        self.model = repen(mode=self.mode, hidden_dim=self.hidden_dim, batch_size=self.batch_size, nb_batch=self.nb_batch,
                           n_epochs=self.n_epochs, known_outliers=1000000,
                           path_model=self.modelpath, save_suffix=self.save_suffix)


        # fitting
        self.model.fit(X_train, y_train)

        return self

    def predict_score(self, X):
        score = self.model.decision_function(X)
        return score

    def parameter_count(self):
        """
        计算REPEN模型的参数数量
        
        Returns:
            dict: 包含各个组件参数数量的字典
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                # 获取主网络模型
                if hasattr(self.model, 'network') and self.model.network is not None:
                    network_model = self.model.network.model
                    if network_model is not None:
                        total_params = network_model.count_params()
                        trainable_params = sum([K.count_params(w) for w in network_model.trainable_weights])
                        non_trainable_params = total_params - trainable_params
                        
                        return {
                            'network_total': total_params,
                            'network_trainable': trainable_params,
                            'network_non_trainable': non_trainable_params,
                            'total': total_params
                        }
                else:
                    # 如果模型还没有训练，创建临时网络来计算参数
                    from WSADBench.baseline.REPEN.model import Repen_network
                    temp_network = Repen_network(hidden_dim=self.hidden_dim)
                    # 假设输入维度为100（这是一个默认值，实际使用时会根据数据调整）
                    temp_network.compile_model(input_dim=100)
                    total_params = temp_network.model.count_params()
                    trainable_params = sum([K.count_params(w) for w in temp_network.model.trainable_weights])
                    non_trainable_params = total_params - trainable_params
                    
                    return {
                        'network_total': total_params,
                        'network_trainable': trainable_params,
                        'network_non_trainable': non_trainable_params,
                        'total': total_params,
                        'note': 'Parameters counted from temporary model (input_dim=100)'
                    }
            else:
                # 模型未初始化时的默认计算
                from WSADBench.baseline.REPEN.model import Repen_network
                temp_network = Repen_network(hidden_dim=self.hidden_dim)
                temp_network.compile_model(input_dim=100)
                total_params = temp_network.model.count_params()
                trainable_params = sum([K.count_params(w) for w in temp_network.model.trainable_weights])
                non_trainable_params = total_params - trainable_params
                
                return {
                    'network_total': total_params,
                    'network_trainable': trainable_params,
                    'network_non_trainable': non_trainable_params,
                    'total': total_params,
                    'note': 'Parameters counted from temporary model (input_dim=100)'
                }
                
        except Exception as e:
            return {
                'error': f'Failed to count parameters: {str(e)}',
                'total': 0
            }