from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from WSADBench.myutils import Utils

class supervised():
    def __init__(self, seed:int, model_name:str=None):
        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {'LR':LogisticRegression,
                           'NB':GaussianNB,
                           'SVM':SVC,
                           'MLP':MLPClassifier,
                           'RF':RandomForestClassifier,
                           'LGB':lgb.LGBMClassifier,
                           'XGB':xgb.XGBClassifier,
                           'CatB':CatBoostClassifier}

    def fit(self, X_train, y_train, ratio=None):
        if self.model_name == 'NB':
            self.model = self.model_dict[self.model_name]()
        elif self.model_name == 'SVM':
            self.model = self.model_dict[self.model_name](probability=True)
        elif self.model_name == 'CatB':
            # 在这里写死 CPU 个数，例如 thread_count=4
            self.model = self.model_dict[self.model_name](
                random_state=self.seed,
                thread_count=10,  # <--- 限制为 4 个线程
                verbose=0  # 可选：关闭训练日志输出
            )
        else:
            self.model = self.model_dict[self.model_name](random_state=self.seed)

        # fitting
        self.model.fit(X_train, y_train)

        return self

    def predict_score(self, X):
        score = self.model.predict_proba(X)[:, 1]
        return score