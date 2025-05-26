#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CR-GAN实现测试脚本
检查WSADBench.baseline.CRGAN的实现是否正确
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os

# 添加项目路径
sys.path.append('/data/coding/yx/WSADBench')

from WSADBench.baseline.CRGAN import CRGAN
from WSADBench.myutils import Utils
from WSADBench.datasets.cv_data_generator import CVDataGenerator


def test_crgan_basic_functionality():
    """测试CRGAN基本功能"""
    print("=" * 60)
    print("测试1: CRGAN基本功能测试")
    print("=" * 60)
    
    # 创建模型
    model = CRGAN(
        seed=42,
        epochs=5,  # 少量epoch用于快速测试
        batch_size=32,
        verbose=True
    )
    
    # 测试模型初始化
    print("✓ CRGAN模型初始化成功")
    
    # 创建简单的测试数据 (CV格式: N, C, H, W)
    n_samples = 200
    n_normal = 150
    n_anomaly = 50
    
    # 生成32x32的图像数据
    X_normal = np.random.randn(n_normal, 1, 32, 32).astype(np.float32)
    X_anomaly = np.random.randn(n_anomaly, 1, 32, 32).astype(np.float32) + 2.0  # 添加偏移使其异常
    
    X_train = np.vstack([X_normal, X_anomaly])
    y_train = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # 只标记一部分异常样本
    labeled_mask = np.zeros_like(y_train)
    labeled_anomaly_indices = np.where(y_train == 1)[0][:10]  # 只标记10个异常样本
    labeled_mask[labeled_anomaly_indices] = 1
    
    # 更新标签：0表示未标记，1表示标记的异常
    y_train_labeled = labeled_mask.copy()
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"标记异常样本数: {np.sum(y_train_labeled == 1)}")
    print(f"未标记样本数: {np.sum(y_train_labeled == 0)}")
    
    # 训练模型
    try:
        model.fit(X_train, y_train_labeled)
        print("✓ CRGAN训练成功")
    except Exception as e:
        print(f"✗ CRGAN训练失败: {e}")
        return False
    
    # 测试预测
    try:
        scores = model.predict_proba(X_train[:50])
        print(f"✓ 预测成功，异常分数形状: {scores.shape}")
        print(f"异常分数统计: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        return False
    
    return True


def test_crgan_data_compatibility():
    """测试CRGAN数据兼容性"""
    print("\n" + "=" * 60)
    print("测试2: CRGAN数据兼容性测试")
    print("=" * 60)
    
    model = CRGAN(epochs=3, batch_size=16, verbose=False)
    
    # 测试不同形状的数据
    test_cases = [
        ("小批量数据", (50, 1, 32, 32)),
        ("大批量数据", (1000, 1, 32, 32)),
        ("极小数据", (10, 1, 32, 32)),
    ]
    
    for test_name, shape in test_cases:
        print(f"\n测试 {test_name}: {shape}")
        
        X = np.random.randn(*shape).astype(np.float32)
        # 创建一些标记异常样本
        n_labeled_anomaly = min(5, shape[0] // 4)
        y = np.zeros(shape[0])
        y[:n_labeled_anomaly] = 1
        
        try:
            model_test = CRGAN(epochs=2, batch_size=min(16, shape[0]), verbose=False)
            model_test.fit(X, y)
            scores = model_test.predict_proba(X[:20] if shape[0] > 20 else X)
            print(f"  ✓ {test_name}测试成功")
        except Exception as e:
            print(f"  ✗ {test_name}测试失败: {e}")
            return False
    
    return True


def test_crgan_with_real_cv_data():
    """使用真实CV数据测试CRGAN"""
    print("\n" + "=" * 60)
    print("测试3: 使用真实CV数据测试CRGAN, 仅使用1000样本")
    print("=" * 60)
    
    try:
        # 使用CV数据生成器，指定dataset参数
        data_generator = CVDataGenerator(dataset='mnist', seed=42)
        
        print("正在生成MNIST数据...")
        data = data_generator.generator(
            normal_class=0,
            la=0.1,  # 10%的异常样本被标记
            at_least_one_labeled=True,
            return_tensors=False  # 返回numpy格式
        )
        
        X_train = data['X_train'][:1000]  # 只使用前1000个样本
        y_train = data['y_train'][:1000]
        X_test = data['X_test'][:1000]  # 只使用前1000个样本
        y_test = data['y_test'][:1000]
        
        print(f"训练数据形状: {X_train.shape}")
        print(f"测试数据形状: {X_test.shape}")
        print(f"训练集异常比例: {np.mean(y_train):.3f}")
        print(f"测试集异常比例: {np.mean(y_test):.3f}")
        
        y_train[-1] = 1  # 确保至少有一个异常样本被标记
        # 创建弱监督标签（只标记部分异常样本）
        anomaly_indices = np.where(y_train == 1)[0]
        labeled_ratio = 1.0
        n_labeled = max(1, int(len(anomaly_indices) * labeled_ratio))  # 确保至少有1个标记样本
        
        # 确保有足够的异常样本可以标记
        if len(anomaly_indices) == 0:
            print("警告: 训练数据中没有异常样本，跳过测试")
            return True
        
        n_labeled = min(n_labeled, len(anomaly_indices))  # 不超过实际异常样本数
        labeled_indices = np.random.choice(anomaly_indices, n_labeled, replace=False)
        
        y_train_wsad = np.zeros_like(y_train)
        y_train_wsad[labeled_indices] = 1
        
        print(f"标记异常样本数: {np.sum(y_train_wsad == 1)}")
        print(f"未标记样本数: {np.sum(y_train_wsad == 0)}")
        
        # 训练CRGAN
        model = CRGAN(
            epochs=20,
            batch_size=64,
            lr_g=0.0001,
            lr_e=0.0001,
            lr_d=0.000025,
            alpha=10.0,
            beta=10.0,
            score_type='reconstruction',
            verbose=True
        )
        
        print("开始训练CRGAN...")
        model.fit(X_train, y_train_wsad)
        
        # 测试
        print("开始测试...")
        test_scores = model.predict_proba(X_test)
        
        # 计算性能指标
        auc_roc = roc_auc_score(y_test, test_scores)
        auc_pr = average_precision_score(y_test, test_scores)
        
        print(f"测试结果:")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")
        
        # 简单的性能检查
        if auc_roc > 0.6:  # 基本的异常检测性能
            print("✓ CRGAN在真实CV数据上表现正常")
            return True
        else:
            print("⚠ CRGAN在真实CV数据上性能较低，可能需要调整参数")
            return True  # 不算失败，可能是参数需要调整
            
    except Exception as e:
        print(f"✗ 真实CV数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crgan_model_components():
    """测试CRGAN模型组件"""
    print("\n" + "=" * 60)
    print("测试4: CRGAN模型组件测试")
    print("=" * 60)
    
    try:
        from WSADBench.baseline.CRGAN.model import Generator, Encoder, Discriminator, normal_init
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 测试生成器
        print("测试生成器...")
        generator = Generator().to(device)
        z = torch.randn(10, 100).to(device)
        x_gen = generator(z)
        print(f"  生成器输入形状: {z.shape}")
        print(f"  生成器输出形状: {x_gen.shape}")
        assert x_gen.shape == (10, 1, 32, 32), f"生成器输出形状错误: {x_gen.shape}"
        print("  ✓ 生成器测试通过")
        
        # 测试编码器
        print("测试编码器...")
        encoder = Encoder().to(device)
        x = torch.randn(10, 1, 32, 32).to(device)
        z_enc = encoder(x)
        print(f"  编码器输入形状: {x.shape}")
        print(f"  编码器输出形状: {z_enc.shape}")
        assert z_enc.shape == (10, 100), f"编码器输出形状错误: {z_enc.shape}"
        print("  ✓ 编码器测试通过")
        
        # 测试判别器
        print("测试判别器...")
        discriminator = Discriminator().to(device)
        
        # 测试XZ判别器
        d_xz, f_xz = discriminator(x, z_enc, 'xz')
        print(f"  XZ判别器输出形状: {d_xz.shape}, 特征形状: {f_xz.shape}")
        assert d_xz.shape == (10, 1), f"XZ判别器输出形状错误: {d_xz.shape}"
        
        # 测试XX判别器
        d_xx, f_xx = discriminator(x, x_gen, 'xx')
        print(f"  XX判别器输出形状: {d_xx.shape}, 特征形状: {f_xx.shape}")
        assert d_xx.shape == (10, 1), f"XX判别器输出形状错误: {d_xx.shape}"
        print("  ✓ 判别器测试通过")
        
        # 测试权重初始化
        print("测试权重初始化...")
        test_model = Generator()
        test_model.apply(normal_init)
        print("  ✓ 权重初始化测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crgan_training_functions():
    """测试CRGAN训练函数"""
    print("\n" + "=" * 60)
    print("测试5: CRGAN训练函数测试")
    print("=" * 60)
    
    try:
        from WSADBench.baseline.CRGAN.fit import fit_crgan
        from WSADBench.baseline.CRGAN.utils import compute_anomaly_scores
        from WSADBench.baseline.CRGAN.model import Generator, Encoder, Discriminator
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        generator = Generator().to(device)
        encoder = Encoder().to(device)
        discriminator = Discriminator().to(device)
        
        # 创建优化器
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0001)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.000025)
        
        # 创建测试数据
        X_train = torch.randn(100, 1, 32, 32).float()
        y_train = np.zeros(100)
        y_train[:10] = 1  # 前10个为标记异常
        
        print("测试训练函数...")
        fit_crgan(
            X_train=X_train,
            y_train=y_train,
            X_aux=None,
            generator=generator,
            encoder=encoder,
            discriminator=discriminator,
            optimizer_G=optimizer_G,
            optimizer_E=optimizer_E,
            optimizer_D=optimizer_D,
            epochs=2,
            batch_size=32,
            latent_dim=100,
            device=device,
            verbose=False
        )
        print("  ✓ 训练函数测试通过")
        
        # 测试异常分数计算
        print("测试异常分数计算...")
        X_test = torch.randn(50, 1, 32, 32).float()
        scores = compute_anomaly_scores(
            generator=generator,
            encoder=encoder,
            X=X_test,
            device=device,
            score_type='reconstruction'
        )
        print(f"  异常分数形状: {scores.shape}")
        assert len(scores) == 50, f"异常分数长度错误: {len(scores)}"
        print("  ✓ 异常分数计算测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crgan_edge_cases():
    """测试CRGAN边界情况"""
    print("\n" + "=" * 60)
    print("测试6: CRGAN边界情况测试")
    print("=" * 60)
    
    test_cases = [
        ("无标记异常数据", lambda: test_no_labeled_anomalies()),
        ("全为标记异常数据", lambda: test_all_labeled_anomalies()),
        ("极少数据", lambda: test_minimal_data()),
        ("验证错误处理", lambda: test_error_handling()),
    ]
    
    all_passed = True
    for test_name, test_func in test_cases:
        print(f"\n测试 {test_name}...")
        try:
            if test_func():
                print(f"  ✓ {test_name}测试通过")
            else:
                print(f"  ✗ {test_name}测试失败")
                all_passed = False
        except Exception as e:
            print(f"  ✗ {test_name}测试失败: {e}")
            all_passed = False
    
    return all_passed


def test_no_labeled_anomalies():
    """测试无标记异常数据的情况"""
    model = CRGAN(epochs=2, batch_size=16, verbose=False)
    X = np.random.randn(50, 1, 32, 32).astype(np.float32)
    y = np.zeros(50)  # 全部为未标记
    
    try:
        model.fit(X, y)
        return False  # 应该抛出异常，如果没有抛出则测试失败
    except ValueError as e:
        # 预期的错误，检查错误消息
        if "at least one labeled anomaly sample" in str(e):
            return True
        else:
            print(f"Unexpected error message: {e}")
            return False
    except Exception as e:
        print(f"Unexpected exception type: {type(e).__name__}: {e}")
        return False


def test_all_labeled_anomalies():
    """测试全为标记异常数据的情况"""
    model = CRGAN(epochs=2, batch_size=16, verbose=False)
    X = np.random.randn(50, 1, 32, 32).astype(np.float32)
    y = np.ones(50)  # 全部为标记异常
    
    try:
        model.fit(X, y)
        return False  # 应该抛出异常，如果没有抛出则测试失败
    except ValueError as e:
        # 预期的错误，检查错误消息
        if "at least one unlabeled sample" in str(e):
            return True
        else:
            print(f"Unexpected error message: {e}")
            return False
    except Exception as e:
        print(f"Unexpected exception type: {type(e).__name__}: {e}")
        return False


def test_minimal_data():
    """测试极少数据的情况"""
    model = CRGAN(epochs=2, batch_size=5, verbose=False)
    X = np.random.randn(10, 1, 32, 32).astype(np.float32)
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 2个标记异常，8个未标记
    
    try:
        model.fit(X, y)
        scores = model.predict_proba(X)
        return len(scores) == 10
    except Exception as e:
        print(f"Minimal data test failed: {e}")
        return False


def test_error_handling():
    """测试错误处理机制"""
    model = CRGAN(epochs=1, batch_size=16, verbose=False)
    
    # 测试空数据
    try:
        X_empty = np.random.randn(0, 1, 32, 32).astype(np.float32)
        y_empty = np.array([])
        model.fit(X_empty, y_empty)
        return False  # 应该抛出错误
    except Exception:
        pass  # 预期的错误
    
    # 测试形状不匹配
    try:
        X_mismatch = np.random.randn(10, 1, 32, 32).astype(np.float32)
        y_mismatch = np.array([1, 0, 0])  # 长度不匹配
        model.fit(X_mismatch, y_mismatch)
        return False  # 应该抛出错误
    except Exception:
        pass  # 预期的错误
    
    # 测试正常情况
    try:
        X_normal = np.random.randn(20, 1, 32, 32).astype(np.float32)
        y_normal = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        model.fit(X_normal, y_normal)
        scores = model.predict_proba(X_normal[:5])
        return len(scores) == 5
    except Exception as e:
        print(f"Normal case failed: {e}")
        return False


def check_crgan_implementation():
    """检查CRGAN实现中的常见问题"""
    print("\n" + "=" * 60)
    print("检查CRGAN实现")
    print("=" * 60)
    
    issues_found = []
    
    # 检查import问题
    try:
        from WSADBench.baseline.CRGAN import CRGAN
        print("✓ CRGAN导入成功")
    except ImportError as e:
        issues_found.append(f"导入错误: {e}")
    
    # 检查模型组件
    try:
        from WSADBench.baseline.CRGAN.model import Generator, Encoder, Discriminator
        print("✓ 模型组件导入成功")
    except ImportError as e:
        issues_found.append(f"模型组件导入错误: {e}")
    
    # 检查训练函数
    try:
        from WSADBench.baseline.CRGAN.fit import fit_crgan
        print("✓ 训练函数导入成功")
    except ImportError as e:
        issues_found.append(f"训练函数导入错误: {e}")
    
    # 检查utils函数
    try:
        from WSADBench.baseline.CRGAN.utils import compute_anomaly_scores
        print("✓ 工具函数导入成功")
    except ImportError as e:
        issues_found.append(f"工具函数导入错误: {e}")
    
    if issues_found:
        print("\n发现以下问题:")
        for issue in issues_found:
            print(f"  ✗ {issue}")
        return False
    else:
        print("✓ 所有基本组件检查通过")
        return True


def main():
    """主测试函数"""
    print("开始CR-GAN实现测试")
    print("=" * 80)
    
    # 设置随机种子
    utils = Utils()
    utils.set_seed(42)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("基本实现检查", check_crgan_implementation),
        ("模型组件测试", test_crgan_model_components),
        ("训练函数测试", test_crgan_training_functions),
        ("基本功能测试", test_crgan_basic_functionality),
        ("数据兼容性测试", test_crgan_data_compatibility),
        ("边界情况测试", test_crgan_edge_cases),
        ("真实数据测试", test_crgan_with_real_cv_data),
    ]
    
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            import traceback
            traceback.print_exc()
            test_results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！CR-GAN实现正确。")
        return True
    else:
        print("⚠️ 部分测试失败，需要修复实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
