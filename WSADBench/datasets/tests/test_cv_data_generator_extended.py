# -*- coding: utf-8 -*-
"""
测试CVDataGenerator的功能，包括对Sultani视频数据集的支持
"""

import sys
import os
sys.path.append('/data/coding/yx/WSADBench')

from WSADBench.datasets.cv_data_generator import CVDataGenerator

def test_image_datasets():
    """测试图像数据集"""
    print("=" * 50)
    print("测试图像数据集")
    print("=" * 50)
    
    # 测试CIFAR-10
    try:
        generator = CVDataGenerator(dataset='cifar10', seed=42, image_size=32)
        print(f"支持的数据集: {generator.list_supported_datasets()}")
        print(f"图像数据集: {generator.list_image_datasets()}")
        print(f"视频数据集: {generator.list_video_datasets()}")
        
        # 测试是否正确识别数据集类型
        print(f"CIFAR-10是图像数据集: {generator.is_image_dataset('cifar10')}")
        print(f"CIFAR-10是视频数据集: {generator.is_video_dataset('cifar10')}")
        
        # 获取数据集信息
        info = generator.get_dataset_info('cifar10')
        print(f"CIFAR-10信息: {info}")
        
        print("CIFAR-10测试成功!")
        
    except Exception as e:
        print(f"CIFAR-10测试失败: {e}")

def test_video_datasets():
    """测试视频数据集"""
    print("=" * 50)
    print("测试视频数据集")
    print("=" * 50)
    
    # 测试UCF-Crime
    try:
        generator = CVDataGenerator(dataset='ucf_crime', seed=42, modality='TWO')
        
        # 测试是否正确识别数据集类型
        print(f"UCF-Crime是图像数据集: {generator.is_image_dataset('ucf_crime')}")
        print(f"UCF-Crime是视频数据集: {generator.is_video_dataset('ucf_crime')}")
        
        # 获取数据集信息
        info = generator.get_dataset_info('ucf_crime')
        print(f"UCF-Crime信息: {info}")
        
        print("UCF-Crime配置测试成功!")
        
    except Exception as e:
        print(f"UCF-Crime测试失败: {e}")
    
    # 测试ShanghaiTech
    try:
        generator = CVDataGenerator(dataset='shanghaitech', seed=42, modality='RGB')
        info = generator.get_dataset_info('shanghaitech')
        print(f"ShanghaiTech信息: {info}")
        print("ShanghaiTech配置测试成功!")
        
    except Exception as e:
        print(f"ShanghaiTech测试失败: {e}")

def test_dataset_loading():
    """测试数据集加载（仅配置，不加载实际数据）"""
    print("=" * 50)
    print("测试数据集加载功能")
    print("=" * 50)
    
    # 测试MNIST（小数据集，应该能下载）
    try:
        generator = CVDataGenerator(dataset='mnist', seed=42, image_size=28)
        
        # 这里只测试配置，不实际加载数据
        print("MNIST数据集配置测试成功!")
        
    except Exception as e:
        print(f"MNIST配置测试失败: {e}")

if __name__ == "__main__":
    test_image_datasets()
    test_video_datasets()
    test_dataset_loading()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
