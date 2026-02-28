# -*- coding: utf-8 -*-
import numpy as np
import torch
from WSADBench.myutils import Utils
from typing import Optional, List
import os
from WSADBench.baseline.LimiX.LimiXmain.inference.predictor import LimiXPredictor

class LimiX16M:
    def __init__(self,
                 device ='cuda:0',
                 input_dim = None,
                 model_path = 'ckpt/LimiX-16M.ckpt',
                 mix_precision:bool=True,
                 inference_config: Optional[str]='WSADBench/baseline/LimiX/LimiXmain/config/cls_default_16M_retrieval.json',
                 categorical_features_indices: Optional[List[int]] = None,
                 outlier_remove_std: float=12,
                 softmax_temperature:float=0.9,
                 task_type: str = "Classifiaction",
                 mask_prediction:bool=False,
                 inference_with_DDP: bool = False,
                 seed: int = None):
        """
        :param self: 说明
        :param device: The hardware that loads the model
        :type device: torch.device
        :param model_path: 说明
        :type model_path: str
        :param mix_precision: 说明
        :type mix_precision: bool
        :param inference_config: 说明
        :type inference_config: str
        :param categorical_features_indices: 说明
        :type categorical_features_indices: list[int]
        :param outlier_remove_std: 说明
        :type outlier_remove_std: float
        :param softmax_temperature: 说明
        :type softmax_temperature: float
        :param task_type: 说明
        :type task_type: str
        :param mask_prediction: 说明
        :type mask_prediction: bool
        :param inference_with_DDP: 说明
        :type inference_with_DDP: bool
        :param seed: 说明
        :type seed: int
        """
        #其他参数
        self.seed = seed
        self.utils = Utils()
        self.device = self.utils.get_device(True)

        #模型初始化参数
        self.model = None
        self.input_dim = input_dim

        self.model_path = model_path
        self.mix_precision = mix_precision
        self.inference_config = inference_config
        self.categorical_features_indices = categorical_features_indices
        self.outlier_remove_std =outlier_remove_std
        self.softmax_temperature = softmax_temperature
        self.task_type = task_type
        self.mask_prediction = mask_prediction
        self.inference_with_DDP = inference_with_DDP


    def fit(self,X_train,y_train):
        
        self.model = LimiXPredictor(
            device = self.device,
            model_path= self.model_path,  #启用16M对预训练模型
            mix_precision=True,  # 启用混合精度加速
            inference_config= self.inference_config,
            task_type= self.task_type
        )

        print("模型初始化完成,启用LimiX-16M...")



        return self

    def predict_score(self,X_train,y_train,X_test):

        scores = self.model.predict(X_train,y_train,X_test)

        return scores

