# WSADBench/baseline/VadClip/clip/myUtils.py
import numpy as np
from tqdm import tqdm
import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging(log_dir, name='exp'):
    """
    配置日志系统，避免重复初始化
    支持按日期滚动的文件日志和控制台输出
    """
    # 创建专用的logger，避免影响root logger
    logger_name = f"{name}_logger"
    logger = logging.getLogger(logger_name)

    # 如果logger已经配置过，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名
    log_filename = os.path.join(
        log_dir,
        f"{name}_{datetime.now().strftime('%Y%m')}.log"
    )

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建文件处理器（按天滚动）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',
        interval=1,
        backupCount=30,  # 保留30天的日志
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 防止日志传播到root logger
    logger.propagate = False

    return logger


# myLogger = setup_logging(log_dir='/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/logs')

# with open(r'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/logs/exp_202508.log', 'r') as f:
#     log_text = f.read()
#     log_text = log_text.split('start train RTFM...')[-1]
