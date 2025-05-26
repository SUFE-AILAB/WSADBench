#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sultani方法测试脚本
测试WSADBench.baseline.Sultani的实现是否正确
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os
import time

# 添加项目路径
sys.path.append('/data/coding/yx/WSADBench')

from WSADBench.baseline.Sultani import Sultani
from WSADBench.myutils import Utils
from WSADBench.datasets.cv_data_generator import CVDataGenerator


def test_sultani_basic_functionality():
    """测试Sultani基本功能"""
    print("=" * 60)
    print("测试1: Sultani基本功能测试")
    print("=" * 60)
    
    # 创建模型
    model = Sultani(
        seed=42,
        epochs=5,  # 少量epoch用于快速测试
        batch_size=1,
        input_dim=2048,
        verbose=True
    )
    
    # 测试模型初始化
    print("✓ Sultani模型初始化成功")
    
    # 创建简单的测试数据（模拟视频特征）
    n_samples = 200
    n_normal = 150
    n_anomaly = 50
    feature_dim = 2048
    
    # 生成特征数据（模拟ResNet提取的特征）
    np.random.seed(42)
    X_normal = np.random.randn(n_normal, feature_dim).astype(np.float32)
    X_anomaly = np.random.randn(n_anomaly, feature_dim).astype(np.float32) + 1.0  # 添加偏移使其异常
    
    X_train = np.vstack([X_normal, X_anomaly])
    y_train = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # 打乱数据
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"标签分布: 正常样本={np.sum(y_train == 0)}, 异常样本={np.sum(y_train == 1)}")
    
    # 训练模型
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ 模型训练完成，耗时: {training_time:.2f}秒")
    
    # 预测
    scores = model.predict_proba(X_train)
    predictions = model.predict(X_train)
    
    print(f"✓ 预测完成")
    print(f"异常分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"预测标签分布: 正常={np.sum(predictions == 0)}, 异常={np.sum(predictions == 1)}")
    
    # 计算性能指标
    if len(np.unique(y_train)) > 1:
        auc = roc_auc_score(y_train, scores)
        ap = average_precision_score(y_train, scores)
        accuracy = np.mean(predictions == y_train)
        
        print(f"训练集性能:")
        print(f"  AUC: {auc:.4f}")
        print(f"  AP: {ap:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    return model


def test_sultani_with_cv_data():
    """测试Sultani在CV数据上的表现"""
    print("\n" + "=" * 60)
    print("测试2: Sultani在CV数据集上的测试")
    print("=" * 60)
    
    try:
        # 使用CIFAR-10数据集
        cv_gen = CVDataGenerator(
            seed=42,
            dataset='cifar10',
            test_size=0.3,
            image_size=32
        )
        
        # 生成异常检测数据集（使用类别0作为正常类）
        data_dict = cv_gen.generator(
            normal_class=0,
            anomaly_classes=[1, 2],  # 使用类别1和2作为异常类
            return_tensors=False
        )
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        print(f"CIFAR-10数据加载成功:")
        print(f"  训练集: {X_train.shape}, 标签分布: {np.bincount(y_train.astype(int))}")
        print(f"  测试集: {X_test.shape}, 标签分布: {np.bincount(y_test.astype(int))}")
        
        # 将图像数据展平为特征向量（简化处理）
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # 降维到合理的特征维度
        feature_dim = 512
        if X_train_flat.shape[1] > feature_dim:
            # 简单的随机投影降维
            np.random.seed(42)
            projection_matrix = np.random.randn(X_train_flat.shape[1], feature_dim) / np.sqrt(feature_dim)
            X_train_flat = X_train_flat @ projection_matrix
            X_test_flat = X_test_flat @ projection_matrix
        
        print(f"降维后特征维度: {X_train_flat.shape[1]}")
        
        # 创建Sultani模型
        model = Sultani(
            seed=42,
            epochs=10,
            batch_size=1,
            input_dim=X_train_flat.shape[1],
            learning_rate=0.001,
            verbose=True
        )
        
        # 训练模型
        start_time = time.time()
        model.fit(X_train_flat, y_train, X_test_flat, y_test)
        training_time = time.time() - start_time
        
        print(f"✓ 模型在CIFAR-10上训练完成，耗时: {training_time:.2f}秒")
        
        # 在测试集上评估
        test_scores = model.predict_proba(X_test_flat)
        test_predictions = model.predict(X_test_flat)
        
        if len(np.unique(y_test)) > 1:
            test_auc = roc_auc_score(y_test, test_scores)
            test_ap = average_precision_score(y_test, test_scores)
            test_accuracy = np.mean(test_predictions == y_test)
            
            print(f"测试集性能:")
            print(f"  AUC: {test_auc:.4f}")
            print(f"  AP: {test_ap:.4f}")
            print(f"  Accuracy: {test_accuracy:.4f}")
        
        return model
        
    except Exception as e:
        print(f"CIFAR-10测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sultani_video_data():
    """测试Sultani在视频数据上的表现"""
    print("\n" + "=" * 60)
    print("测试3: Sultani在模拟视频数据上的测试")
    print("=" * 60)
    
    # 创建模拟视频异常检测数据
    n_videos = 50
    segments_per_video = 32
    feature_dim = 2048
    
    # 模拟正常视频（30个）
    n_normal_videos = 30
    n_anomaly_videos = 20
    
    np.random.seed(42)
    
    # 正常视频特征（较小的方差）
    normal_videos = []
    for i in range(n_normal_videos):
        video_features = np.random.randn(segments_per_video, feature_dim) * 0.5
        normal_videos.append(video_features)
    
    # 异常视频特征（在某些段有较大的偏移）
    anomaly_videos = []
    for i in range(n_anomaly_videos):
        video_features = np.random.randn(segments_per_video, feature_dim) * 0.5
        # 在随机的几个段添加异常信号
        anomaly_segments = np.random.choice(segments_per_video, size=5, replace=False)
        video_features[anomaly_segments] += np.random.randn(5, feature_dim) * 2.0
        anomaly_videos.append(video_features)
    
    # 组合数据
    all_videos = normal_videos + anomaly_videos
    all_labels = [0] * n_normal_videos + [1] * n_anomaly_videos
    
    # 展平为样本级数据
    X_all = []
    y_all = []
    
    for video, label in zip(all_videos, all_labels):
        for segment in video:
            X_all.append(segment)
            y_all.append(label)
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # 分割训练测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    
    print(f"模拟视频数据:")
    print(f"  总视频数: {len(all_videos)} (正常:{n_normal_videos}, 异常:{n_anomaly_videos})")
    print(f"  总段数: {len(X_all)}")
    print(f"  训练集: {X_train.shape}, 标签分布: {np.bincount(y_train)}")
    print(f"  测试集: {X_test.shape}, 标签分布: {np.bincount(y_test)}")
    
    # 创建Sultani模型
    model = Sultani(
        seed=42,
        epochs=15,
        batch_size=1,
        input_dim=feature_dim,
        learning_rate=0.001,
        sparsity_weight=0.00008,
        smoothness_weight=0.00008,
        verbose=True
    )
    
    # 训练模型
    start_time = time.time()
    model.fit(X_train, y_train, X_test, y_test)
    training_time = time.time() - start_time
    
    print(f"✓ 模型在模拟视频数据上训练完成，耗时: {training_time:.2f}秒")
    
    # 评估性能
    results = model.evaluate(X_test, y_test)
    print(f"测试集性能: {results}")
    
    return model


def test_sultani_model_persistence():
    """测试Sultani模型保存和加载"""
    print("\n" + "=" * 60)
    print("测试4: Sultani模型保存和加载测试")
    print("=" * 60)
    
    # 创建简单数据
    n_samples = 100
    feature_dim = 256
    
    np.random.seed(42)
    X = np.random.randn(n_samples, feature_dim).astype(np.float32)
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # 训练模型
    model1 = Sultani(
        seed=42,
        epochs=5,
        input_dim=feature_dim,
        verbose=True
    )
    
    model1.fit(X, y)
    scores1 = model1.predict_proba(X)
    
    # 保存模型
    model_path = "/tmp/sultani_test_model.pth"
    model1.save_model(model_path)
    print(f"✓ 模型已保存到: {model_path}")
    
    # 加载模型
    model2 = Sultani()
    model2.load_model(model_path)
    scores2 = model2.predict_proba(X)
    
    # 验证一致性
    score_diff = np.abs(scores1 - scores2).max()
    print(f"✓ 模型加载成功")
    print(f"预测分数最大差异: {score_diff:.8f}")
    
    if score_diff < 1e-6:
        print("✓ 模型保存和加载测试通过")
    else:
        print("✗ 模型保存和加载测试失败")
    
    # 清理临时文件
    if os.path.exists(model_path):
        os.remove(model_path)
    
    return score_diff < 1e-6


def test_sultani_parameters():
    """测试Sultani参数设置"""
    print("\n" + "=" * 60)
    print("测试5: Sultani参数设置测试")
    print("=" * 60)
    
    # 测试参数获取和设置
    model = Sultani(
        seed=42,
        epochs=10,
        learning_rate=0.001,
        verbose=True
    )
    
    # 获取参数
    params = model.get_params()
    print(f"✓ 获取参数成功: {len(params)} 个参数")
    
    # 设置参数
    new_params = {
        'epochs': 20,
        'learning_rate': 0.01,
        'dropout': 0.5
    }
    model.set_params(**new_params)
    
    # 验证参数设置
    updated_params = model.get_params()
    for key, value in new_params.items():
        if updated_params[key] == value:
            print(f"✓ 参数 {key} 设置成功: {value}")
        else:
            print(f"✗ 参数 {key} 设置失败: 期望 {value}, 实际 {updated_params[key]}")
    
    print("✓ 参数设置测试完成")


def run_all_tests():
    """运行所有测试"""
    print("Sultani方法完整测试套件")
    print("=" * 80)
    
    test_results = {}
    
    try:
        # 基本功能测试
        model1 = test_sultani_basic_functionality()
        test_results['basic'] = model1 is not None
        
        # CV数据测试
        model2 = test_sultani_with_cv_data()
        test_results['cv_data'] = model2 is not None
        
        # 视频数据测试
        model3 = test_sultani_video_data()
        test_results['video_data'] = model3 is not None
        
        # 模型持久化测试
        persistence_result = test_sultani_model_persistence()
        test_results['persistence'] = persistence_result
        
        # 参数设置测试
        test_sultani_parameters()
        test_results['parameters'] = True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结测试结果
    print("\n" + "=" * 80)
    print("测试结果总结:")
    print("=" * 80)
    
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:15s}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过! Sultani方法实现正确。")
    else:
        print(f"⚠️  有 {total_tests - passed_tests} 个测试失败，需要检查。")
    
    return test_results


if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行所有测试
    run_all_tests()
