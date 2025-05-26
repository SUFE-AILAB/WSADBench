#!/usr/bin/env python3
"""
CVDataGenerator测试脚本
测试CV数据生成器的基本功能
"""

import sys
import os
import numpy as np
import torch

# 添加WSADBench到Python路径
sys.path.append('/data/coding/yx/WSADBench')

from WSADBench.datasets import CVDataGenerator


def test_cv_data_generator():
    """测试CVDataGenerator的基本功能"""
    
    print("=" * 60)
    print("Testing CVDataGenerator")
    print("=" * 60)
    
    # 测试支持的数据集
    cv_gen = CVDataGenerator(seed=42)
    print("Supported datasets:", cv_gen.list_supported_datasets())
    print()
    
    # 测试每个数据集
    for dataset_name in ['mnist', 'fashion_mnist', 'cifar10']:
        print(f"\n--- Testing {dataset_name} ---")
        
        try:
            # 创建数据生成器
            cv_gen = CVDataGenerator(
                seed=42,
                dataset=dataset_name,
                test_size=0.3,
                image_size=32
            )
            
            # 获取数据集信息
            info = cv_gen.get_dataset_info(dataset_name)
            print(f"Dataset info: {info}")
            
            # 生成异常检测数据集
            # 使用类别0作为正常类，其他作为异常类
            data_dict = cv_gen.generator(
                normal_class=0,
                la=0.1,  # 10%的异常样本被标记
                at_least_one_labeled=True,
                return_tensors=True
            )
            
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            dataset_info = data_dict['dataset_info']
            
            print(f"Training set shape: {X_train.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Test labels shape: {y_test.shape}")
            print(f"Training anomaly ratio: {np.mean(y_train):.3f}")
            print(f"Test anomaly ratio: {np.mean(y_test):.3f}")
            print(f"Data type: {type(X_train)}")
            print(f"Data dtype: {X_train.dtype}")
            print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
            print(f"Dataset info: {dataset_info}")
            
            # 测试numpy格式输出
            data_dict_numpy = cv_gen.generator(
                normal_class=0,
                la=0.05,
                return_tensors=False
            )
            print(f"Numpy format shape: {data_dict_numpy['X_train'].shape}")
            print(f"Numpy data type: {type(data_dict_numpy['X_train'])}")
            
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("CVDataGenerator test completed!")
    print("=" * 60)


def test_custom_data():
    """测试自定义数据功能"""
    print("\n--- Testing Custom Data ---")
    
    # 创建模拟的自定义图像数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟 100个样本，3通道，32x32图像
    X_custom = torch.randn(100, 3, 32, 32) * 0.5 + 0.5  # 归一化到[0,1]
    y_custom = np.random.randint(0, 5, 100)  # 5个类别
    
    cv_gen = CVDataGenerator(seed=42, test_size=0.3)
    
    data_dict = cv_gen.generator(
        X=X_custom,
        y=y_custom,
        normal_class=0,
        la=0.1,
        return_tensors=True
    )
    
    print(f"Custom data - Training shape: {data_dict['X_train'].shape}")
    print(f"Custom data - Test shape: {data_dict['X_test'].shape}")
    print(f"Custom data - Training anomaly ratio: {np.mean(data_dict['y_train']):.3f}")


if __name__ == "__main__":
    test_cv_data_generator()
    test_custom_data()
