from torch.utils.data import DataLoader
from WSADBench.baseline.DeepSAD.src.base import BaseADDataset
from WSADBench.baseline.DeepSAD.src.base.odds_dataset import ODDSDataset
from .preprocessing import create_semisupervised_setting

import torch


class ODDSADDataset(BaseADDataset):

    def __init__(self, data,mask, train,n_known_outlier_classes: int = 1, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 1.0, ratio_pollution: float = 0.0):    #ratio_known_outlier实际用rla控制了，这里设为1.0即可
        super().__init__(self)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        # training or testing dataset
        self.train = train

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1,)


        if self.train:
            # Get training set
            self.train_set = ODDSDataset(data=data,train=True)

                # Create semi-supervised setting
            idx, _, semi_targets = create_semisupervised_setting(self.train_set.targets.cpu().data.numpy(),mask,self.normal_classes,
                                                                self.outlier_classes, self.known_outlier_classes,
                                                                ratio_known_normal, ratio_known_outlier, ratio_pollution)
            self.train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels
            print("值为1的数量：", (self.train_set.semi_targets == 1).sum().item())
            print("值为-1的数量：", (self.train_set.semi_targets == -1).sum().item())
            print("值为0的数量：", (self.train_set.semi_targets == 0).sum().item())

        else:
            # Get testing set
            self.test_set = ODDSDataset(data=data, train=False)

        

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):

        if self.train:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers, drop_last=True)
            return train_loader
        else:
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers, drop_last=False)
            return test_loader