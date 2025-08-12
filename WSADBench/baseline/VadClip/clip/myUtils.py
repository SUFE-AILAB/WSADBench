import numpy as np
from tqdm import tqdm


def read_np():
    file_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/CV_by_I3D/UCF_Crime/all_rgbs/Burglary/Burglary044_x264.mp4.npy'
    # file_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/CV_by_I3D/UCF_Crime/all_rgbs/Burglary/Burglary045_x264.mp4.npy'
    data = np.load(file_path)
    pass

# config/logging_config.py
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging(log_dir):
    """
    配置日志系统
    支持按日期滚动的文件日志和控制台输出
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名
    log_filename = os.path.join(
        log_dir,
        f"exp_{datetime.now().strftime('%Y%m')}.log"
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

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 配置特定模块的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('celery').setLevel(logging.INFO)
    return root_logger

myLogger = setup_logging(log_dir='/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/logs')

# with open(r'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/logs/exp_202508.log', 'r') as f:
#     log_text = f.read()
#     log_text = log_text.split('start train RTFM...')[-1]
