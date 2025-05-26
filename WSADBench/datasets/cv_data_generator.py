import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
from typing import Tuple, Dict, Any, Optional, List, Union
from sklearn.model_selection import train_test_split

from .data_generator import DataGenerator
from WSADBench.myutils import Utils
from .dataset_support import (
    load_ucf_crime_dataset,
    load_shanghaitech_dataset,
    load_vis_dataset,
    get_cv_video_dataset_info
)


class CVDataGenerator(DataGenerator):
    """
    CV数据生成器，继承自DataGenerator，支持原生图片数据加载和视频异常检测数据集
    支持CIFAR-10, MNIST, Fashion-MNIST等图像数据集
    支持UCF-Crime, ShanghaiTech, VIS等视频异常检测数据集
    """
    
    def __init__(self, seed: int = 42, dataset: str = None, test_size: float = 0.3,
                 generate_duplicates=True, n_samples_threshold=1000,
                 data_root: str = None, image_size: int = 32, modality: str = 'TWO'):
        """
        初始化CVDataGenerator
        
        Args:
            seed: 随机种子
            dataset: 数据集名称 ('cifar10', 'mnist', 'fashion_mnist', 'ucf_crime', 'shanghaitech', 'vis')
            test_size: 测试集比例
            generate_duplicates: 是否生成重复样本
            n_samples_threshold: 重复样本阈值
            data_root: 数据根目录，如果为None则使用默认路径
            image_size: 图片尺寸，默认32x32（仅对图像数据集有效）
            modality: 视频数据集的模态 ('RGB', 'FLOW', 'TWO')，仅对视频数据集有效
        """
        super().__init__(seed, dataset, test_size, generate_duplicates, n_samples_threshold)
        
        self.image_size = image_size
        self.modality = modality
        self.data_root = data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CV')
        self.utils = Utils()
        
        # 支持的CV数据集列表
        self.supported_datasets = ['cifar10', 'mnist', 'fashion_mnist', 'ucf_crime', 'shanghaitech', 'vis']
        
        # 图像数据集配置
        self.image_dataset_configs = {
            'cifar10': {
                'dataset_class': datasets.CIFAR10,
                'channels': 3,
                'num_classes': 10,
                'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                               'dog', 'frog', 'horse', 'ship', 'truck']
            },
            'mnist': {
                'dataset_class': datasets.MNIST,
                'channels': 1,
                'num_classes': 10,
                'class_names': [str(i) for i in range(10)]
            },
            'fashion_mnist': {
                'dataset_class': datasets.FashionMNIST,
                'channels': 1,
                'num_classes': 10,
                'class_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            }
        }
        
        # 视频异常检测数据集配置
        self.video_dataset_configs = {
            'ucf_crime': get_cv_video_dataset_info('ucf_crime'),
            'shanghaitech': get_cv_video_dataset_info('shanghaitech'),
            'vis': get_cv_video_dataset_info('vis')
        }
    
    def is_video_dataset(self, dataset_name: str) -> bool:
        """
        判断是否为视频数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            True if video dataset, False if image dataset
        """
        return dataset_name.lower() in self.video_dataset_configs
    
    def is_image_dataset(self, dataset_name: str) -> bool:
        """
        判断是否为图像数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            True if image dataset, False if video dataset
        """
        return dataset_name.lower() in self.image_dataset_configs
    
    def get_transform(self) -> transforms.Compose:
        """
        获取数据预处理transform
        
        Returns:
            torchvision transforms
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def load_raw_dataset(self, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载原始数据集（支持图像和视频数据集）
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {self.supported_datasets}")
        
        # 判断数据集类型并分别处理
        if self.is_image_dataset(dataset_name):
            return self._load_image_dataset(dataset_name)
        elif self.is_video_dataset(dataset_name):
            return self._load_video_dataset(dataset_name)
        else:
            raise ValueError(f"Unknown dataset type for: {dataset_name}")
    
    def _load_image_dataset(self, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载图像数据集
        
        Args:
            dataset_name: 图像数据集名称
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        config = self.image_dataset_configs[dataset_name]
        dataset_class = config['dataset_class']
        transform = self.get_transform()
        
        # 创建数据集特定的目录
        dataset_path = os.path.join(self.data_root, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # 加载训练集
        train_dataset = dataset_class(
            root=dataset_path,
            train=True,
            download=True,
            transform=transform
        )
        
        # 加载测试集
        test_dataset = dataset_class(
            root=dataset_path,
            train=False,
            download=True,
            transform=transform
        )
        
        # 转换为numpy数组
        X_train = train_dataset.data
        y_train = np.array(train_dataset.targets)
        X_test = test_dataset.data
        y_test = np.array(test_dataset.targets)
        
        # 处理CIFAR-10数据格式
        if dataset_name == 'cifar10':
            # CIFAR-10数据是numpy array格式 (N, H, W, C)
            X_train = torch.from_numpy(X_train).float()
            X_test = torch.from_numpy(X_test).float()
            # 转换为 (N, C, H, W) 格式
            X_train = X_train.permute(0, 3, 1, 2)
            X_test = X_test.permute(0, 3, 1, 2)
        else:
            # MNIST和Fashion-MNIST数据是torch tensor格式 (N, H, W)
            if not isinstance(X_train, torch.Tensor):
                X_train = torch.from_numpy(X_train)
            if not isinstance(X_test, torch.Tensor):
                X_test = torch.from_numpy(X_test)
            
            X_train = X_train.float().unsqueeze(1)  # 添加channel维度 (N, 1, H, W)
            X_test = X_test.float().unsqueeze(1)
        
        # 确保targets是numpy array
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.numpy()
        
        # 归一化到[0,1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # 调整尺寸
        if X_train.shape[-1] != self.image_size or X_train.shape[-2] != self.image_size:
            resize_transform = transforms.Resize((self.image_size, self.image_size))
            X_train = torch.stack([resize_transform(img) for img in X_train])
            X_test = torch.stack([resize_transform(img) for img in X_test])
        
        return X_train, y_train, X_test, y_test
    
    def _load_video_dataset(self, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载视频异常检测数据集
        
        Args:
            dataset_name: 视频数据集名称
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        # 获取数据集根目录
        dataset_path = os.path.join(self.data_root, dataset_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Video dataset path not found: {dataset_path}")
        
        # 根据数据集类型调用相应的加载函数
        if dataset_name == 'ucf_crime':
            X_train, y_train, X_test, y_test = load_ucf_crime_dataset(dataset_path, self.modality)
        elif dataset_name == 'shanghaitech':
            X_train, y_train, X_test, y_test = load_shanghaitech_dataset(dataset_path, self.modality)
        elif dataset_name == 'vis':
            X_train, y_train, X_test, y_test = load_vis_dataset(dataset_path, self.modality)
        else:
            raise ValueError(f"Unknown video dataset: {dataset_name}")
        
        # 转换为torch tensor
        if len(X_train) > 0:
            X_train = torch.from_numpy(X_train).float()
        else:
            X_train = torch.empty(0)
            
        if len(X_test) > 0:
            X_test = torch.from_numpy(X_test).float()
        else:
            X_test = torch.empty(0)
        
        # 确保标签是numpy array格式
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        return X_train, y_train, X_test, y_test
    
    def create_binary_anomaly_detection_dataset(self, 
                                              X_train: torch.Tensor, 
                                              y_train: np.ndarray,
                                              X_test: torch.Tensor, 
                                              y_test: np.ndarray,
                                              normal_class: int = 0,
                                              anomaly_classes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        创建二分类异常检测数据集
        
        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            normal_class: 正常类别（对视频数据集无效，已经是二分类）
            anomaly_classes: 异常类别列表（对视频数据集无效，已经是二分类）
            
        Returns:
            包含训练和测试数据的字典
        """
        # 检查是否为视频数据集（已经是二分类格式）
        if self.dataset and self.is_video_dataset(self.dataset):
            # 视频数据集已经是二分类格式，直接返回
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'normal_class': 0,  # 固定为0
                'anomaly_classes': [1]  # 固定为1
            }
        
        # 图像数据集需要转换为二分类格式
        # 设置异常类别
        if anomaly_classes is None:
            all_classes = np.unique(np.concatenate([y_train, y_test]))
            anomaly_classes = [c for c in all_classes if c != normal_class]
        
        # 处理训练集
        normal_mask_train = y_train == normal_class
        anomaly_mask_train = np.isin(y_train, anomaly_classes)
        
        X_train_normal = X_train[normal_mask_train]
        X_train_anomaly = X_train[anomaly_mask_train]
        
        # 创建二分类标签 (0: normal, 1: anomaly)
        y_train_binary = np.concatenate([
            np.zeros(len(X_train_normal)),  # normal = 0
            np.ones(len(X_train_anomaly))   # anomaly = 1
        ])
        
        X_train_binary = torch.cat([X_train_normal, X_train_anomaly], dim=0)
        
        # 处理测试集
        normal_mask_test = y_test == normal_class
        anomaly_mask_test = np.isin(y_test, anomaly_classes)
        
        X_test_normal = X_test[normal_mask_test]
        X_test_anomaly = X_test[anomaly_mask_test]
        
        y_test_binary = np.concatenate([
            np.zeros(len(X_test_normal)),   # normal = 0
            np.ones(len(X_test_anomaly))    # anomaly = 1
        ])
        
        X_test_binary = torch.cat([X_test_normal, X_test_anomaly], dim=0)
        
        return {
            'X_train': X_train_binary,
            'y_train': y_train_binary,
            'X_test': X_test_binary,
            'y_test': y_test_binary,
            'normal_class': normal_class,
            'anomaly_classes': anomaly_classes
        }
    
    def generator(self, X=None, y=None, minmax=False,
                  la=None, at_least_one_labeled=False,
                  normal_class: int = 0,
                  anomaly_classes: Optional[List[int]] = None,
                  return_tensors: bool = True,
                  **kwargs) -> Dict[str, Any]:
        """
        生成CV异常检测数据集
        
        Args:
            X, y: 自定义数据，如果提供则使用自定义数据
            minmax: 是否进行MinMax缩放（对图像数据通常不需要）
            la: 标记异常样本的比例或数量
            at_least_one_labeled: 是否保证至少有一个标记异常样本
            normal_class: 正常类别
            anomaly_classes: 异常类别列表
            return_tensors: 是否返回torch tensor格式
            **kwargs: 其他参数
            
        Returns:
            包含训练和测试数据的字典
        """
        # 设置随机种子
        self.utils.set_seed(self.seed)
        
        # 加载数据
        if X is None or y is None:
            if self.dataset is None:
                raise ValueError("必须提供dataset参数或自定义X, y数据")
            
            print(f'Loading CV dataset: {self.dataset}...')
            X_train_raw, y_train_raw, X_test_raw, y_test_raw = self.load_raw_dataset(self.dataset)
        else:
            # 使用自定义数据
            print('Using custom CV dataset...')
            # 假设自定义数据已经是正确格式
            X_combined = X
            y_combined = y
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_combined, y_combined, test_size=self.test_size, 
                shuffle=True, stratify=y_combined, random_state=self.seed
            )
        
        # 创建二分类异常检测数据集
        dataset_dict = self.create_binary_anomaly_detection_dataset(
            X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            normal_class=normal_class, anomaly_classes=anomaly_classes
        )
        
        X_train = dataset_dict['X_train']
        y_train = dataset_dict['y_train']
        X_test = dataset_dict['X_test']
        y_test = dataset_dict['y_test']
        
        # 数据统计
        print(f'Dataset: {self.dataset}')
        print(f'Normal class: {normal_class}')
        print(f'Anomaly classes: {dataset_dict["anomaly_classes"]}')
        print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
        print(f'Train anomaly ratio: {np.mean(y_train):.3f}')
        print(f'Test anomaly ratio: {np.mean(y_test):.3f}')
        
        # 处理标记异常样本
        if la is not None:
            idx_normal = np.where(y_train == 0)[0]
            idx_anomaly = np.where(y_train == 1)[0]
            
            if len(idx_anomaly) == 0:
                print("Warning: No anomalies in training set!")
                idx_labeled_anomaly = np.array([], dtype=int)
            else:
                if isinstance(la, float):
                    if at_least_one_labeled:
                        n_labeled = max(1, int(np.ceil(la * len(idx_anomaly))))
                    else:
                        n_labeled = int(la * len(idx_anomaly))
                elif isinstance(la, int):
                    if la > len(idx_anomaly):
                        raise ValueError(f'标记异常数量 {la} 大于总异常数量 {len(idx_anomaly)}!')
                    n_labeled = la
                else:
                    raise NotImplementedError("la must be float or int")
                
                idx_labeled_anomaly = np.random.choice(idx_anomaly, n_labeled, replace=False)
            
            idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
            idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)
            
            # 重新标记：unlabeled设为0，labeled anomalies设为1
            y_train_new = y_train.copy()
            y_train_new[idx_unlabeled] = 0
            y_train_new[idx_labeled_anomaly] = 1
            y_train = y_train_new
        
        # 转换为numpy格式（如果需要）
        if not return_tensors:
            if isinstance(X_train, torch.Tensor):
                X_train = X_train.numpy()
            if isinstance(X_test, torch.Tensor):
                X_test = X_test.numpy()
        
        result = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'dataset_info': self._get_dataset_info_dict(dataset_dict, normal_class)
        }
        
        return result
    
    def _get_dataset_info_dict(self, dataset_dict: Dict[str, Any], normal_class: int) -> Dict[str, Any]:
        """
        获取数据集信息字典
        
        Args:
            dataset_dict: 数据集字典
            normal_class: 正常类别
            
        Returns:
            数据集信息字典
        """
        if self.dataset and self.is_image_dataset(self.dataset):
            # 图像数据集
            return {
                'dataset_name': self.dataset,
                'dataset_type': 'image',
                'normal_class': normal_class,
                'anomaly_classes': dataset_dict["anomaly_classes"],
                'image_size': self.image_size,
                'channels': self.image_dataset_configs.get(self.dataset, {}).get('channels', 3),
                'modality': None
            }
        elif self.dataset and self.is_video_dataset(self.dataset):
            # 视频数据集
            video_config = self.video_dataset_configs.get(self.dataset, {})
            return {
                'dataset_name': self.dataset,
                'dataset_type': 'video',
                'normal_class': 0,
                'anomaly_classes': [1],
                'image_size': None,
                'channels': None,
                'modality': self.modality,
                'feature_dim': video_config.get('feature_dim_combined' if self.modality == 'TWO' 
                                               else f'feature_dim_{self.modality.lower()}', 4096)
            }
        else:
            # 默认配置
            return {
                'dataset_name': self.dataset,
                'dataset_type': 'unknown',
                'normal_class': normal_class,
                'anomaly_classes': dataset_dict.get("anomaly_classes", []),
                'image_size': self.image_size,
                'channels': 3,
                'modality': self.modality
            }
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集信息字典
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        if self.is_image_dataset(dataset_name):
            return self.image_dataset_configs[dataset_name]
        elif self.is_video_dataset(dataset_name):
            return self.video_dataset_configs[dataset_name]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_name}")
    
    def list_supported_datasets(self) -> List[str]:
        """
        列出支持的数据集
        
        Returns:
            支持的数据集名称列表
        """
        return self.supported_datasets.copy()
    
    def list_image_datasets(self) -> List[str]:
        """
        列出支持的图像数据集
        
        Returns:
            图像数据集名称列表
        """
        return list(self.image_dataset_configs.keys())
    
    def list_video_datasets(self) -> List[str]:
        """
        列出支持的视频异常检测数据集
        
        Returns:
            视频数据集名称列表
        """
        return list(self.video_dataset_configs.keys())
