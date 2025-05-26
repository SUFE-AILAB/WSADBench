#!/usr/bin/env python3
"""
AABiGAN模型测试脚本
测试表格数据和CV数据两种模态
"""

import sys
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# 添加WSADBench到Python路径
sys.path.append('/data/coding/yx/WSADBench')

from WSADBench.baseline.AABiGAN import AABiGAN
from WSADBench.datasets import CVDataGenerator, DataGenerator


def test_tabular_data():
    """测试表格数据"""
    print("=" * 60)
    print("Testing AABiGAN with Tabular Data")
    print("=" * 60)
    
    # 使用WSADBench的经典数据集
    data_gen = DataGenerator(seed=42, dataset='arrhythmia', test_size=0.3)
    
    try:
        data_dict = data_gen.generator(
            la=0.1,  # 10%的异常样本被标记
            at_least_one_labeled=True
        )
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training anomaly ratio: {np.mean(y_train):.3f}")
        print(f"Test anomaly ratio: {np.mean(y_test):.3f}")
        
        # 创建AABiGAN模型
        model = AABiGAN(
            seed=42,
            modal='tabular',
            latent_dim=50,
            epochs=20,  # 减少epoch用于快速测试
            batch_size=32,
            hidden_dims=[64, 32],
            lr_g=0.001,
            lr_e=0.001,
            lr_d=0.0005,
            alpha=1.0,
            beta=5.0,
            gamma=1.0,
            verbose=True
        )
        
        # 训练模型
        print("\nTraining AABiGAN...")
        model.fit(X_train, y_train)
        
        # 预测
        print("\nMaking predictions...")
        scores = model.predict_proba(X_test)
        predictions = model.predict(X_test)
        
        # 评估
        auc_score = roc_auc_score(y_test, scores)
        ap_score = average_precision_score(y_test, scores)
        
        print(f"\nResults:")
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"AUC-PR: {ap_score:.4f}")
        print(f"Test predictions: {np.bincount(predictions)}")
        
        print("✓ Tabular data test completed successfully!")
        
    except Exception as e:
        print(f"✗ Tabular data test failed: {e}")
        import traceback
        traceback.print_exc()


def test_cv_data():
    """测试CV数据"""
    print("\n" + "=" * 60)
    print("Testing AABiGAN with CV Data")
    print("=" * 60)
    
    try:
        # 使用CVDataGenerator
        cv_gen = CVDataGenerator(
            seed=42,
            dataset='mnist',
            test_size=0.3,
            image_size=32
        )
        
        # 生成异常检测数据集 (类别0为正常，其他为异常)
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
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training anomaly ratio: {np.mean(y_train):.3f}")
        print(f"Test anomaly ratio: {np.mean(y_test):.3f}")
        print(f"Dataset info: {dataset_info}")
        
        # 创建AABiGAN模型
        model = AABiGAN(
            seed=42,
            modal='cv',
            latent_dim=100,
            epochs=20,  # 减少epoch用于快速测试
            batch_size=64,
            channels=dataset_info['channels'],
            img_size=dataset_info['image_size'],
            lr_g=0.0002,
            lr_e=0.0002,
            lr_d=0.0001,
            alpha=1.0,
            beta=10.0,
            gamma=1.0,
            verbose=True
        )
        
        # 训练模型
        print("\nTraining AABiGAN...")
        model.fit(X_train, y_train)
        
        # 预测
        print("\nMaking predictions...")
        scores = model.predict_proba(X_test)
        predictions = model.predict(X_test)
        
        # 评估
        auc_score = roc_auc_score(y_test, scores)
        ap_score = average_precision_score(y_test, scores)
        
        print(f"\nResults:")
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"AUC-PR: {ap_score:.4f}")
        print(f"Test predictions: {np.bincount(predictions)}")
        
        print("✓ CV data test completed successfully!")
        
    except Exception as e:
        print(f"✗ CV data test failed: {e}")
        import traceback
        traceback.print_exc()


def test_custom_data():
    """测试自定义数据"""
    print("\n" + "=" * 60)
    print("Testing AABiGAN with Custom Data")
    print("=" * 60)
    
    try:
        # 生成模拟数据
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 正常数据：高斯分布
        n_normal = 800
        X_normal = np.random.normal(0, 1, (n_normal, 10))
        
        # 异常数据：均匀分布
        n_anomaly = 200
        X_anomaly = np.random.uniform(-3, 3, (n_anomaly, 10))
        
        # 合并数据
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
        
        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # 模拟弱监督场景：只有10%的异常样本被标记
        anomaly_mask = y_train == 1
        labeled_ratio = 0.1
        n_labeled = int(np.sum(anomaly_mask) * labeled_ratio)
        
        # 随机选择要标记的异常样本
        anomaly_indices = np.where(anomaly_mask)[0]
        labeled_indices = np.random.choice(anomaly_indices, n_labeled, replace=False)
        
        # 创建新的标签：0=未标记，1=标记异常
        y_train_weak = np.zeros_like(y_train)
        y_train_weak[labeled_indices] = 1
        
        print(f"Custom data shape: {X.shape}")
        print(f"Training: {len(X_train)} samples, {np.sum(y_train_weak)} labeled anomalies")
        print(f"Test: {len(X_test)} samples, {np.sum(y_test)} true anomalies")
        
        # 创建AABiGAN模型
        model = AABiGAN(
            seed=42,
            modal='tabular',
            latent_dim=32,
            epochs=30,
            batch_size=32,
            hidden_dims=[64, 32],
            verbose=True
        )
        
        # 训练模型
        print("\nTraining AABiGAN...")
        model.fit(X_train, y_train_weak)
        
        # 预测
        print("\nMaking predictions...")
        scores = model.predict_proba(X_test)
        
        # 评估
        auc_score = roc_auc_score(y_test, scores)
        ap_score = average_precision_score(y_test, scores)
        
        print(f"\nResults:")
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"AUC-PR: {ap_score:.4f}")
        
        print("✓ Custom data test completed successfully!")
        
    except Exception as e:
        print(f"✗ Custom data test failed: {e}")
        import traceback
        traceback.print_exc()


def test_model_parameters():
    """测试模型参数设置"""
    print("\n" + "=" * 60)
    print("Testing Model Parameters")
    print("=" * 60)
    
    try:
        # 测试参数获取和设置
        model = AABiGAN(
            modal='tabular',
            latent_dim=50,
            epochs=10,
            batch_size=32
        )
        
        # 获取参数
        params = model.get_params()
        print("Model parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 设置参数
        model.set_params(latent_dim=64, epochs=20)
        new_params = model.get_params()
        print(f"\nUpdated latent_dim: {new_params['latent_dim']}")
        print(f"Updated epochs: {new_params['epochs']}")
        
        print("✓ Parameter test completed successfully!")
        
    except Exception as e:
        print(f"✗ Parameter test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting AABiGAN comprehensive tests...")
    
    # 测试表格数据
    test_tabular_data()
    
    # 测试CV数据
    test_cv_data()
    
    # 测试自定义数据
    test_custom_data()
    
    # 测试模型参数
    test_model_parameters()
    
    print("\n" + "=" * 60)
    print("All AABiGAN tests completed!")
    print("=" * 60)
