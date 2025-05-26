# -*- coding: utf-8 -*-
"""
CV视频异常检测数据集支持模块
处理UCF-Crime、ShanghaiTech、VIS等视频异常检测数据集的加载
以及其他CV相关的特殊数据集处理函数
"""

import os
import numpy as np
import torch
import pickle
from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_ucf_crime_dataset(data_root: str, modality: str = 'TWO') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载UCF-Crime数据集
    
    Args:
        data_root: UCF-Crime数据集根目录
        modality: 数据模态 ('RGB', 'FLOW', 'TWO')
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    print(f"Loading UCF-Crime dataset from {data_root} with modality {modality}")
    
    # 检查必要文件是否存在
    required_files = [
        'train_normal.txt', 'train_anomaly.txt', 
        'test_normalv2.txt', 'test_anomalyv2.txt'
    ]
    
    for file_name in required_files:
        if not os.path.exists(os.path.join(data_root, file_name)):
            raise FileNotFoundError(f"Required file {file_name} not found in {data_root}")
    
    # 加载训练数据
    X_train_normal = _load_ucf_crime_file_list(
        os.path.join(data_root, 'train_normal.txt'), data_root, modality, is_train=True
    )
    X_train_anomaly = _load_ucf_crime_file_list(
        os.path.join(data_root, 'train_anomaly.txt'), data_root, modality, is_train=True
    )
    
    # 加载测试数据
    X_test_normal, _ = _load_ucf_crime_test_file_list(
        os.path.join(data_root, 'test_normalv2.txt'), data_root, modality
    )
    X_test_anomaly, _ = _load_ucf_crime_test_file_list(
        os.path.join(data_root, 'test_anomalyv2.txt'), data_root, modality
    )
    
    # 合并数据
    X_train = np.concatenate([X_train_normal, X_train_anomaly], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_train_normal)),  # normal = 0
        np.ones(len(X_train_anomaly))   # anomaly = 1
    ])
    
    X_test = np.concatenate([X_test_normal, X_test_anomaly], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_test_normal)),   # normal = 0
        np.ones(len(X_test_anomaly))    # anomaly = 1
    ])
    
    print(f"UCF-Crime loaded: Train {X_train.shape}, Test {X_test.shape}")
    print(f"Train anomaly ratio: {np.mean(y_train):.3f}, Test anomaly ratio: {np.mean(y_test):.3f}")
    
    return X_train, y_train, X_test, y_test


def load_shanghaitech_dataset(data_root: str, modality: str = 'TWO') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载ShanghaiTech数据集
    
    Args:
        data_root: ShanghaiTech数据集根目录
        modality: 数据模态 ('RGB', 'FLOW', 'TWO')
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    print(f"Loading ShanghaiTech dataset from {data_root} with modality {modality}")
    
    # ShanghaiTech数据集结构检查
    training_dir = os.path.join(data_root, 'training')
    testing_dir = os.path.join(data_root, 'testing')
    
    if not os.path.exists(training_dir) or not os.path.exists(testing_dir):
        # 尝试其他可能的目录结构
        training_dir = data_root
        testing_dir = data_root
    
    # 加载训练数据（ShanghaiTech通常只有正常数据用于训练）
    X_train_normal = _load_shanghaitech_training_data(training_dir, data_root, modality)
    
    # 加载测试数据
    X_test_normal, X_test_anomaly = _load_shanghaitech_testing_data(testing_dir, data_root, modality)
    
    # 合并数据
    X_train = X_train_normal
    y_train = np.zeros(len(X_train_normal))  # 训练集全是正常数据
    
    X_test = np.concatenate([X_test_normal, X_test_anomaly], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_test_normal)),   # normal = 0
        np.ones(len(X_test_anomaly))    # anomaly = 1
    ])
    
    print(f"ShanghaiTech loaded: Train {X_train.shape}, Test {X_test.shape}")
    print(f"Train anomaly ratio: {np.mean(y_train):.3f}, Test anomaly ratio: {np.mean(y_test):.3f}")
    
    return X_train, y_train, X_test, y_test


def load_vis_dataset(data_root: str, modality: str = 'TWO') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载VIS数据集
    
    Args:
        data_root: VIS数据集根目录
        modality: 数据模态 ('RGB', 'FLOW', 'TWO')
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    print(f"Loading VIS dataset from {data_root} with modality {modality}")
    
    # VIS数据集的具体加载逻辑需要根据实际数据结构调整
    # 这里提供一个基本框架
    
    try:
        # 尝试加载VIS特定的文件结构
        X_train, y_train = _load_vis_training_data(data_root, modality)
        X_test, y_test = _load_vis_testing_data(data_root, modality)
        
        print(f"VIS loaded: Train {X_train.shape}, Test {X_test.shape}")
        print(f"Train anomaly ratio: {np.mean(y_train):.3f}, Test anomaly ratio: {np.mean(y_test):.3f}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.warning(f"Failed to load VIS dataset: {e}")
        # 返回空数据集
        return np.array([]), np.array([]), np.array([]), np.array([])


def _load_video_features(file_name: str, data_root: str, modality: str) -> Optional[np.ndarray]:
    """
    加载视频特征文件
    
    Args:
        file_name: 视频文件名（不含扩展名）
        data_root: 数据根目录
        modality: 数据模态
        
    Returns:
        特征数组或None
    """
    # 去掉可能的扩展名
    base_name = os.path.splitext(file_name)[0]
    
    rgb_path = os.path.join(data_root, 'all_rgbs', f'{base_name}.npy')
    flow_path = os.path.join(data_root, 'all_flows', f'{base_name}.npy')
    
    try:
        if modality == 'RGB':
            if os.path.exists(rgb_path):
                return np.load(rgb_path)
        elif modality == 'FLOW':
            if os.path.exists(flow_path):
                return np.load(flow_path)
        else:  # TWO (RGB + FLOW)
            if os.path.exists(rgb_path) and os.path.exists(flow_path):
                rgb_data = np.load(rgb_path)
                flow_data = np.load(flow_path)
                return np.concatenate([rgb_data, flow_data], axis=1)
        
        return None
        
    except Exception as e:
        logger.warning(f"Error loading features for {file_name}: {e}")
        return None


def _load_ucf_crime_file_list(list_file: str, data_root: str, modality: str, is_train: bool = True) -> np.ndarray:
    """加载UCF-Crime文件列表"""
    with open(list_file, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    
    data_list = []
    for file_name in file_list:
        data = _load_video_features(file_name, data_root, modality)
        if data is not None:
            data_list.append(data)
    
    if not data_list:
        logger.warning(f"No valid data found in {list_file}")
        return np.array([])
    
    return np.array(data_list)


def _load_ucf_crime_test_file_list(list_file: str, data_root: str, modality: str) -> Tuple[np.ndarray, List[Dict]]:
    """加载UCF-Crime测试文件列表（包含元信息）"""
    data_list = []
    metadata = []
    
    with open(list_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if '|' in line:  # 异常文件格式
                parts = line.split('|')
                file_name = parts[0]
                frames = int(parts[1]) if len(parts) > 1 else -1
                anomaly_segments = eval(parts[2]) if len(parts) > 2 else []
            else:  # 正常文件格式
                parts = line.split(' ')
                file_name = parts[0]
                frames = int(parts[1]) if len(parts) > 1 else -1
                anomaly_segments = []
            
            data = _load_video_features(file_name, data_root, modality)
            if data is not None:
                data_list.append(data)
                metadata.append({
                    'file_name': file_name,
                    'frames': frames,
                    'anomaly_segments': anomaly_segments
                })
    
    return np.array(data_list), metadata


def _load_shanghaitech_training_data(training_dir: str, data_root: str, modality: str) -> np.ndarray:
    """加载ShanghaiTech训练数据"""
    # 尝试从训练目录或RGB目录扫描文件
    data_list = []
    
    # 检查是否有专门的训练文件列表
    train_list_file = os.path.join(training_dir, 'training_videos.txt')
    if os.path.exists(train_list_file):
        with open(train_list_file, 'r') as f:
            file_list = [line.strip() for line in f.readlines()]
        
        for file_name in file_list:
            data = _load_video_features(file_name, data_root, modality)
            if data is not None:
                data_list.append(data)
    else:
        # 扫描RGB目录
        rgb_dir = os.path.join(data_root, 'all_rgbs')
        if os.path.exists(rgb_dir):
            for file_name in os.listdir(rgb_dir):
                if file_name.endswith('.npy'):
                    base_name = os.path.splitext(file_name)[0]
                    data = _load_video_features(base_name, data_root, modality)
                    if data is not None:
                        data_list.append(data)
    
    return np.array(data_list) if data_list else np.array([])


def _load_shanghaitech_testing_data(testing_dir: str, data_root: str, modality: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载ShanghaiTech测试数据"""
    # ShanghaiTech的测试数据需要根据ground truth划分正常和异常
    # 这里需要读取GT_anomaly.pkl文件来获取异常标注
    
    gt_file = os.path.join(data_root, 'GT_anomaly.pkl')
    normal_data = []
    anomaly_data = []
    
    if os.path.exists(gt_file):
        try:
            with open(gt_file, 'rb') as f:
                gt_data = pickle.load(f)
            
            # 根据GT信息加载数据
            for video_name, gt_info in gt_data.items():
                data = _load_video_features(video_name, data_root, modality)
                if data is not None:
                    # 根据GT信息判断是否包含异常
                    has_anomaly = any(gt_info) if isinstance(gt_info, list) else bool(gt_info)
                    if has_anomaly:
                        anomaly_data.append(data)
                    else:
                        normal_data.append(data)
                        
        except Exception as e:
            logger.warning(f"Error reading GT file: {e}")
    
    # 如果没有GT文件，尝试其他方法
    if not normal_data and not anomaly_data:
        # 扫描测试目录
        test_rgb_dir = os.path.join(testing_dir, 'frames')
        if not os.path.exists(test_rgb_dir):
            test_rgb_dir = os.path.join(data_root, 'all_rgbs')
        
        if os.path.exists(test_rgb_dir):
            for file_name in os.listdir(test_rgb_dir):
                if file_name.endswith('.npy'):
                    base_name = os.path.splitext(file_name)[0]
                    data = _load_video_features(base_name, data_root, modality)
                    if data is not None:
                        # 默认分配（可能需要调整）
                        normal_data.append(data)
    
    normal_array = np.array(normal_data) if normal_data else np.array([])
    anomaly_array = np.array(anomaly_data) if anomaly_data else np.array([])
    
    return normal_array, anomaly_array


def _load_vis_training_data(data_root: str, modality: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载VIS训练数据"""
    # VIS数据集的具体结构需要根据实际情况调整
    # 这里提供一个基本框架
    
    data_list = []
    labels = []
    
    # 扫描数据目录
    rgb_dir = os.path.join(data_root, 'all_rgbs')
    if os.path.exists(rgb_dir):
        for file_name in os.listdir(rgb_dir):
            if file_name.endswith('.npy'):
                base_name = os.path.splitext(file_name)[0]
                data = _load_video_features(base_name, data_root, modality)
                if data is not None:
                    data_list.append(data)
                    # 默认标签分配逻辑（需要根据实际情况调整）
                    labels.append(0)  # 默认为正常
    
    return np.array(data_list), np.array(labels)


def _load_vis_testing_data(data_root: str, modality: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载VIS测试数据"""
    # 类似的逻辑，需要根据VIS数据集的实际结构调整
    return _load_vis_training_data(data_root, modality)


def get_cv_video_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    获取CV视频异常检测数据集信息
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        数据集信息字典
    """
    dataset_configs = {
        'ucf_crime': {
            'full_name': 'UCF-Crime',
            'type': 'video_anomaly_detection',
            'modalities': ['RGB', 'FLOW', 'TWO'],
            'default_modality': 'TWO',
            'num_classes': 2,  # normal, anomaly
            'feature_dim_rgb': 2048,
            'feature_dim_flow': 2048,
            'feature_dim_combined': 4096,
            'description': 'Real-world anomaly detection in surveillance videos'
        },
        'shanghaitech': {
            'full_name': 'ShanghaiTech',
            'type': 'video_anomaly_detection',
            'modalities': ['RGB', 'FLOW', 'TWO'],
            'default_modality': 'TWO',
            'num_classes': 2,  # normal, anomaly
            'feature_dim_rgb': 2048,
            'feature_dim_flow': 2048,
            'feature_dim_combined': 4096,
            'description': 'Campus surveillance video anomaly detection'
        },
        'vis': {
            'full_name': 'VIS',
            'type': 'video_anomaly_detection',
            'modalities': ['RGB', 'FLOW', 'TWO'],
            'default_modality': 'TWO',
            'num_classes': 2,  # normal, anomaly
            'feature_dim_rgb': 2048,
            'feature_dim_flow': 2048,
            'feature_dim_combined': 4096,
            'description': 'Video surveillance anomaly detection'
        }
    }
    
    return dataset_configs.get(dataset_name.lower(), {})
