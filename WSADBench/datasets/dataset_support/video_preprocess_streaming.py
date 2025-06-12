# -*- coding: utf-8 -*-
"""
流式并行视频预处理模块
使用真正的生产者-消费者模式，实现CPU和GPU的完全并行处理
CPU处理完一个视频立即交给GPU，不等待批次完成
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional, Union, Any, Generator
import logging
from pathlib import Path
import shutil
from tqdm import tqdm
import queue
import threading
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import signal
import atexit
import gc

# PytorchVideo imports
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import TenCrop, FiveCrop, Normalize, Resize, ToTensor, ToPILImage, CenterCrop

# 导入动态模型加载函数
from WSADBench.myutils import import_class
from decord import VideoReader, cpu, gpu

from numpy.lib.format import open_memmap

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理器 - 生成segment级别的任务"""

    def __init__(self, config: Dict, segment_len: int = 50, output_dir: str = None, max_read_segment_in_block: int = 2):
        self.config = config
        self.num_frames = config["PREPROCESS"]["NUM_FRAMES"]
        self.num_clips = config["PREPROCESS"]["NUM_CLIPS"]
        self.resize_dims = config["PREPROCESS"]["RESIZE"]
        self.segment_len = segment_len  # 每段包含的clip数量
        self.output_dir = output_dir or config["PREPROCESS"]["OUTPUT_DIR"]
        self.max_read_segment_in_block = (
            max_read_segment_in_block  # 每次视频读取的最大段数，越大占用内存越多，但速度快。反之会需要更多磁盘io。
        )
        self.crop_size = config["PREPROCESS"].get("CROP_SIZE", 224)
        self._create_transforms()

    def _create_transforms(self):
        """创建数据转换"""
        target_size = self.crop_size

        # 获取配置的模态和Crop类型
        self.modality = self.config["PREPROCESS"].get("MODALITY", "RGB").upper()
        self.crop_type = self.config["PREPROCESS"].get("CROPS", "Ten")

        # 根据crop类型创建crop变换
        if self.crop_type == "Ten":
            self.crop_transform = TenCrop(target_size)
            self.num_crops = 10
        elif self.crop_type == "Five":
            self.crop_transform = FiveCrop(target_size)
            self.num_crops = 5
        elif self.crop_type == "None":
            self.crop_transform = CenterCrop(target_size)
            self.num_crops = 1
        else:
            self.crop_transform = TenCrop(target_size)
            self.num_crops = 10
            logger.warning(f"Unknown crop type {self.crop_type}, using TenCrop")

        # 根据模态设置归一化参数
        if self.modality == "RGB":
            self.normalize = Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        elif self.modality == "FLOW":
            self.normalize = Normalize(mean=[0.5, 0.5], std=[0.226, 0.226])
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        # 基础转换工具
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        self.resize_transform = Resize(self.resize_dims)

    def process_video(self, video_info: Tuple[str, str]) -> Generator[Dict, None, None]:
        """处理单个视频 - 生成器模式，每次yield一个segment级别的任务"""
        video_path, relative_path = video_info
        video_name = Path(video_path).name

        segment_frames = self.segment_len * self.num_frames
        max_frames_in_block = self.max_read_segment_in_block * segment_frames

        try:
            video_reader = VideoReader(video_path, ctx=cpu())
            total_frames = len(video_reader)
            all_clips = int(np.ceil(total_frames / self.num_frames)) if self.num_clips == -1 else self.num_clips

            start_read_frame = 0
            while start_read_frame < total_frames:
                end_read_frame = min(start_read_frame + max_frames_in_block, total_frames)
                frames_npy = video_reader.get_batch(range(start_read_frame, end_read_frame)).asnumpy()
                frames = torch.from_numpy(frames_npy).float()
                frames = frames.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]

                del frames_npy
                gc.collect()

                logger.info(f"Processing video {video_name} with {total_frames} frames")

                for i in range(start_read_frame, end_read_frame, segment_frames):
                    curr_i = i - start_read_frame  # 相对位置
                    segment = frames[:, curr_i : curr_i + segment_frames, :, :]

                    # 提取segment的clips
                    segment_clips = self.extract_dense_clips(segment)
                    num_clips = segment_clips.shape[0]

                    # segment在整个视频中的索引
                    segment_idx = i // segment_frames
                    # 计算在整个视频中的clip索引范围
                    clip_start_idx = int(i / self.num_frames)
                    clip_end_idx = clip_start_idx + num_clips

                    logger.debug(
                        f"Processing segment {segment_idx} for {video_name}, clips {clip_start_idx}:{clip_end_idx}"
                    )

                    # 生成结果文件路径
                    output_subdir = os.path.join(
                        self.output_dir, self.config["PREPROCESS"]["MODALITY_SAVE_DIR"], relative_path
                    )
                    os.makedirs(output_subdir, exist_ok=True)
                    video_memmap_path = os.path.join(output_subdir, f"{video_name}.npy")

                    yield {
                        "video_path": video_path,
                        "relative_path": relative_path,
                        "video_name": video_name,
                        "segment_idx": segment_idx,
                        "segment_clips": segment_clips,
                        "all_clips": all_clips,
                        "clip_start_idx": clip_start_idx,
                        "clip_end_idx": clip_end_idx,
                        "success": True,
                        "num_crops": self.num_crops,
                        "video_memmap_path": video_memmap_path,
                    }

                    del segment, segment_clips
                    gc.collect()

                start_read_frame = end_read_frame

                del frames
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing video {video_name}: {e}")
            yield {
                "video_path": video_path,
                "relative_path": relative_path,
                "video_name": video_name,
                "error": str(e),
                "success": False,
            }

    def extract_dense_clips(self, frames) -> torch.Tensor:
        """提取dense sliding window clips"""
        total_frames = frames.shape[1]

        # 优化后的处理流程
        processed_frames = self._apply_transforms_to_all_frames(frames)
        clips_tensor = self._extract_non_overlapping_clips(processed_frames, total_frames)

        return clips_tensor

    def _apply_transforms_to_all_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """对所有帧批量应用变换"""
        # 将时间维度转换为批量维度以便批处理
        frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
        # 步骤1: 批量应用resize transform
        frames = self.resize_transform(frames)  # [T, C, H, W]
        if self.modality == "RGB":
            frames = frames / 255.0  # 将像素值归一化到[0, 1]
        frames = self.normalize(frames)  # [T, C, H, W]

        all_crops = self.crop_transform(frames)  # [num_crops, T, C, H, W]

        T, C, _, _ = frames.shape

        del frames  # 释放内存
        gc.collect()

        if self.num_crops == 1:
            if len(all_crops.shape) == 4:
                # CenterCrop返回单个图像
                frames = all_crops.unsqueeze(0)
        else:
            frames = torch.empty((self.num_crops, T, C, self.crop_size, self.crop_size), dtype=torch.float32)
            torch.stack(all_crops, out=frames)

            del all_crops  # 释放内存
            gc.collect()

        frames = frames.permute(0, 2, 1, 3, 4)  # [num_crops, C, T, H, W]
        return frames

    def _extract_non_overlapping_clips(self, processed_frames: torch.Tensor, total_frames: int) -> torch.Tensor:
        """提取non-overlapping clips"""
        # 如果总帧数不能被num_frames整除，padding最后一帧
        if total_frames % self.num_frames != 0:
            last_frame = processed_frames[:, :, -1:, :, :]
            padding_frames = self.num_frames - (total_frames % self.num_frames)
            padding = last_frame.repeat(1, 1, padding_frames, 1, 1)
            processed_frames = torch.cat([processed_frames, padding], dim=2)

        # 使用unfold进行non-overlapping clip提取
        clips_tensor = processed_frames.unfold(2, self.num_frames, self.num_frames)
        clips_tensor = clips_tensor.permute(2, 0, 1, 5, 3, 4)  # [num_clip, num_crops, C, num_frames, H, W]

        num_possible_clips = clips_tensor.shape[0]

        # 如果num_clips为-1，返回所有clips
        if self.num_clips == -1:
            return clips_tensor

        # 如果可用clips数量足够，采样目标数量
        if num_possible_clips >= self.num_clips:
            indices = torch.linspace(0, num_possible_clips - 1, self.num_clips).long()
            return clips_tensor[indices]

        # 如果可用clips数量不足，重复最后一个clip
        remaining = self.num_clips - num_possible_clips
        last_clip = clips_tensor[-1:].repeat(remaining, 1, 1, 1, 1, 1)
        return torch.cat([clips_tensor, last_clip], dim=0)


class GPUWorker:
    """GPU工作线程 - 处理segment级别的任务并写入memmap"""

    def __init__(
        self,
        device_id: int,
        config: Dict,
        task_queue: queue.Queue,
        should_stop: threading.Event,
        global_video_locks: Dict,
        global_video_processed_tags: Dict,
        global_locks_lock: threading.Lock,
    ):
        self.device_id = device_id
        self.config = config
        self.task_queue = task_queue
        self.should_stop = should_stop
        self.device = torch.device(f"cuda:{device_id}")
        self.model = None
        self.modality = config["PREPROCESS"].get("MODALITY", "RGB")

        # 使用全局共享的视频状态跟踪
        self.global_video_locks = global_video_locks
        self.global_video_processed_tags = global_video_processed_tags
        self.global_locks_lock = global_locks_lock

        self._init_model()

    def _init_model(self):
        """初始化模型"""
        model_class_path = self.config["PREPROCESS"].get("MODEL")
        ModelClass = import_class(model_class_path)
        weights_path = self.config["PREPROCESS"].get("WEIGHTS_PATH")
        self.model = ModelClass(pretrained=True, weights_path=weights_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"GPU {self.device_id}: Model initialized")

    def _get_or_create_video_memmap(self, task: Dict, end_with=".processing") -> np.memmap:
        """获取或创建视频级别的memmap数组"""
        video_name = task["video_name"]

        # 确保线程安全 - 使用全局锁来保护锁字典的访问
        with self.global_locks_lock:
            if video_name not in self.global_video_locks:
                self.global_video_locks[video_name] = threading.Lock()

        # 使用视频专用锁来保护memmap和状态的操作
        with self.global_video_locks[video_name]:
            video_memmap_path = task["video_memmap_path"] + end_with
            mode = "r+" if os.path.exists(video_memmap_path) else "w+"

            all_clips = task["all_clips"]
            num_crops = task["num_crops"]
            hidden_dim = self.model.feature_dim
            target_shape = (all_clips, num_crops, hidden_dim)

            memmap_array = open_memmap(video_memmap_path, dtype=np.float32, mode=mode, shape=target_shape)

            # 线程安全地初始化video_processed_tags
            if video_name not in self.global_video_processed_tags:
                self.global_video_processed_tags[video_name] = np.zeros(all_clips, dtype=bool)

            return memmap_array

    def extract_features(self, clips: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        clip_num, num_crops, channels, frames, height, width = clips.shape
        clips_reshaped = clips.reshape(-1, channels, frames, height, width)

        batch_size = self.config["PREPROCESS"].get("MODEL_BATCH_SIZE", 32)
        features_list = []

        with torch.no_grad():
            for i in range(0, clips_reshaped.shape[0], batch_size):
                batch = clips_reshaped[i : i + batch_size].to(self.device)
                batch_features = self.model(batch)
                features_list.append(batch_features.cpu())

        all_features = torch.cat(features_list, dim=0)
        embedding_dim = all_features.shape[1]
        features = all_features.view(clip_num, num_crops, embedding_dim)
        return features

    def run(self):
        """运行GPU工作线程"""
        logger.info(f"GPU {self.device_id}: Worker started")

        while not self.should_stop.is_set():
            try:
                task = self.task_queue.get(timeout=1)

                if task is None:  # 结束信号
                    self.task_queue.task_done()
                    logger.info(f"GPU {self.device_id}: Received stop signal")
                    break

                if not task["success"]:
                    # 处理失败的任务，记录错误信息
                    logger.error(f"GPU {self.device_id}: Skipping failed task: {task.get('error', 'Unknown error')}")
                    self.task_queue.task_done()
                    continue

                # 处理segment任务
                video_name = task["video_name"]
                segment_clips = task["segment_clips"]
                clip_start_idx = task["clip_start_idx"]
                clip_end_idx = task["clip_end_idx"]

                try:
                    # 获取或创建视频memmap
                    video_memmap = self._get_or_create_video_memmap(task)

                    # 提取特征
                    features = self.extract_features(segment_clips)

                    # 写入memmap的对应位置
                    video_memmap[clip_start_idx:clip_end_idx] = features.numpy()
                    video_memmap.flush()  # 确保写入磁盘

                    del video_memmap  # 释放内存
                    gc.collect()

                    logger.debug(
                        f"GPU {self.device_id}: Processed segment {task['segment_idx']} of {video_name}, "
                        f"clips {clip_start_idx}:{clip_end_idx}, features shape: {features.shape}"
                    )

                    # 线程安全地更新处理状态和检查完成
                    with self.global_video_locks[video_name]:
                        self.global_video_processed_tags[video_name][clip_start_idx:clip_end_idx] = True
                        if np.all(self.global_video_processed_tags[video_name]):
                            # 该视频已经处理完毕
                            del self.global_video_processed_tags[video_name]  # 清理内存
                            os.rename(task["video_memmap_path"] + ".processing", task["video_memmap_path"])
                            logger.info(f"GPU {self.device_id}: Completed video {video_name}")

                    # 释放内存
                    del segment_clips, features
                    gc.collect()

                except Exception as e:
                    logger.error(f"GPU {self.device_id}: Error processing segment of {video_name}: {e}")

                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                if self.should_stop.is_set():
                    logger.info(f"GPU {self.device_id}: Stop signal detected during timeout")
                    break
                continue
            except Exception as e:
                logger.error(f"GPU {self.device_id}: Error processing task: {e}")
                if self.should_stop.is_set():
                    break

        logger.info(f"GPU {self.device_id}: Worker finished")


class StreamingVideoPreprocessor:
    """流式视频预处理器"""

    def __init__(
        self,
        config_path: str,
        max_queue: int = 10,
        resume: bool = False,
        memory_threshold: float = 0.8,
        segment_len: int = 50,
        max_read_segment_in_block: int = 2,
    ):
        """
        Args:
            config_path: 配置文件路径
            max_queue: 最大队列长度
            resume: 是否跳过已存在的输出文件
            memory_threshold: 内存使用阈值（0-1），超过此值时停止提交新任务
            segment_len: 每段包含的clip数量
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.resume = resume
        self.max_queue = max_queue
        self.memory_threshold = memory_threshold
        self.segment_len = segment_len  # 添加segment_len参数
        self.max_read_segment_in_block = max_read_segment_in_block

        # 信号处理相关
        self.should_stop = threading.Event()
        self.memory_warning = threading.Event()  # 内存警告标志
        self.gpu_threads = []
        self.gpu_workers = []
        self.child_processes = []  # 追踪所有子进程，用于强制清理

        # 全局共享的视频状态跟踪（所有GPU workers共享）
        self.global_video_locks = {}  # video_name -> threading.Lock()
        self.global_video_processed_tags = {}  # video_name -> np.array(bool)
        self.global_locks_lock = threading.Lock()  # 保护global_video_locks字典的锁

        # 注册信号处理器和退出处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)

        # 设置worker数量
        # 对于视频处理这种I/O+CPU密集型任务，使用更多的worker
        # 通常设置为CPU核心数的1.5-2倍较为合适
        self.num_workers = min(psutil.cpu_count() * 2, 32)  # 最多不超过32个worker

        # 添加强制清理标志，防止重复清理
        self._cleanup_started = threading.Event()
        self._force_exit_timer = None

        # 获取可用的GPU设备
        self.device_ids = self._get_available_gpus()

        # 计算队列大小 - 使用max_queue参数
        queue_size = self.max_queue

        logger.info(f"Using {self.num_workers} CPU workers and {len(self.device_ids)} GPUs: {self.device_ids}")
        logger.info(f"Queue buffer size: {queue_size}")
        logger.info(f"Memory usage threshold: {self.memory_threshold:.1%}")
        logger.info(f"CPU cores available: {psutil.cpu_count()}")
        logger.info(f"Segment length: {self.segment_len} clips per segment")

        # 验证配置
        self._validate_config()

        # 创建队列 - 只需要task_queue
        self.task_queue = queue.Queue(maxsize=queue_size)

        self.error_processed_videos = []  # 用于记录处理失败的视频

    def _signal_handler(self, signum, frame):
        """信号处理器，处理Ctrl+C和SIGTERM"""
        logger.info(f"Received signal {signum}, initiating IMMEDIATE shutdown...")
        self.should_stop.set()

        # 立即调用清理，不等待
        self._cleanup()

        # 强制退出程序
        logger.info("Force exiting after cleanup...")
        os._exit(1)  # 使用os._exit强制退出，不执行cleanup handlers

    def _cleanup(self):
        """清理资源，确保所有子进程和线程正确终止"""
        # 防止重复清理
        if self._cleanup_started.is_set():
            logger.info("Cleanup already in progress, skipping...")
            return

        self._cleanup_started.set()
        logger.info("Starting cleanup process...")

        # 设置停止标志
        self.should_stop.set()

        # 3. 清理GPU工作线程 - 强制模式
        try:
            logger.info("Force signaling GPU threads to stop...")

            # 清空任务队列并发送停止信号
            try:
                # 快速清空队列
                while not self.task_queue.empty():
                    try:
                        self.task_queue.get_nowait()
                    except queue.Empty:
                        break

                # 发送停止信号
                for _ in self.device_ids:
                    try:
                        self.task_queue.put(None, timeout=0.1)
                    except queue.Full:
                        break
            except Exception as e:
                logger.error(f"Error clearing task queue: {e}")

            # 强制终止GPU线程 - 缩短等待时间
            for thread in self.gpu_threads:
                if thread.is_alive():
                    logger.info(f"Force terminating {thread.name}...")
                    thread.join(timeout=0.5)  # 只等待0.5秒
                    if thread.is_alive():
                        logger.warning(f"GPU thread {thread.name} did not terminate, forcing...")
                        # 对于Python线程，我们不能直接kill，但可以设置daemon让其随主进程退出
                        thread.daemon = True
                    else:
                        logger.info(f"GPU thread {thread.name} terminated successfully")
        except Exception as e:
            logger.error(f"Error force cleaning up GPU threads: {e}")

        logger.info("Force cleanup process completed")

    def _check_memory_usage(self) -> bool:
        """检查内存使用率，如果超过阈值返回True"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0

            if memory_percent > self.memory_threshold:
                if not self.memory_warning.is_set():
                    self.memory_warning.set()
                    logger.warning(f"Memory usage high: {memory_percent:.1%} > {self.memory_threshold:.1%}")
                return True
            else:
                if self.memory_warning.is_set():
                    self.memory_warning.clear()
                    logger.info(f"Memory usage normal: {memory_percent:.1%}")
                return False
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return False

    def _log_memory_details(self):
        """记录详细的内存使用情况，用于调试内存泄漏"""
        try:
            import psutil
            import os

            # 系统内存
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # 当前进程内存
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()

            logger.info("=== Memory Details ===")
            logger.info(
                f"System Memory - Total: {memory.total / (1024**3):.1f}GB, "
                f"Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%), "
                f"Available: {memory.available / (1024**3):.1f}GB"
            )
            logger.info(
                f"Swap Memory - Total: {swap.total / (1024**3):.1f}GB, "
                f"Used: {swap.used / (1024**3):.1f}GB ({swap.percent:.1f}%)"
            )
            logger.info(
                f"Process Memory - RSS: {process_memory.rss / (1024**3):.1f}GB, "
                f"VMS: {process_memory.vms / (1024**3):.1f}GB"
            )
            logger.info(f"Queue Size - Task: {self.task_queue.qsize()}")
            logger.info("=====================")
        except Exception as e:
            logger.error(f"Error logging memory details: {e}")

    def _wait_for_queue_space_and_put(self, result: Dict):
        """无限等待直到队列有空间且内存使用正常，然后放入任务"""
        video_name = result.get("video_name", "unknown")
        wait_start_time = time.time()
        wait_count = 0
        memory_wait_logged = False
        queue_wait_logged = False

        while not self.should_stop.is_set():
            try:
                # 检查内存使用情况
                memory_high = self._check_memory_usage()

                if self.task_queue.qsize() > 0 and memory_high:
                    # self.task_queue.qsize()>0 的限制确保有任务在GPU处理，防止死锁
                    if not memory_wait_logged:
                        logger.warning(
                            f"Memory usage high, waiting indefinitely for memory to decrease before queuing {video_name}"
                        )
                        memory_wait_logged = True

                    time.sleep(1)
                    wait_count += 1

                    # 每30秒记录一次等待状态
                    if wait_count % 30 == 0:
                        logger.info(f"Still waiting for memory to decrease (waited {wait_count}s) for {video_name}")
                    continue

                # 内存正常，尝试放入队列
                try:
                    self.task_queue.put(result, timeout=0.1)
                    # 成功放入队列
                    total_wait_time = time.time() - wait_start_time
                    if total_wait_time > 1:  # 只记录等待时间超过1秒的情况
                        logger.info(f"Successfully queued {video_name} after waiting {total_wait_time:.1f}s")
                    return

                except queue.Full:
                    # 队列满了，继续等待
                    if not queue_wait_logged:
                        logger.warning(f"Queue full, waiting indefinitely for queue space to put {video_name}")
                        queue_wait_logged = True

                    time.sleep(0.1)  # 队列满的等待间隔稍短一些
                    wait_count += 1

                    # 每30秒记录一次等待状态
                    if wait_count % 300 == 0:  # 队列等待每30秒记录一次 (300 * 0.1s = 30s)
                        logger.info(f"Still waiting for queue space (waited {wait_count * 0.1:.1f}s) for {video_name}")
                    continue

            except Exception as e:
                logger.error(f"Error while waiting to queue {video_name}: {e}")
                # 遇到异常也要等待一下再重试
                time.sleep(1)
                wait_count += 1

        # 如果收到停止信号，记录一下
        if self.should_stop.is_set():
            logger.info(f"Received stop signal while waiting to queue {video_name}, exiting")

    def _get_available_gpus(self) -> List[int]:
        """获取可用的GPU设备ID"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # 获取CUDA_VISIBLE_DEVICES环境变量指定的GPU
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None:
            device_ids = [int(x) for x in visible_devices.split(",") if x.strip()]
            # 重新映射到0, 1, 2, ...
            device_ids = list(range(len(device_ids)))
        else:
            device_ids = list(range(torch.cuda.device_count()))

        if not device_ids:
            raise RuntimeError("No CUDA devices available")

        logger.info(f"Available CUDA devices: {device_ids}")
        return device_ids

    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """验证配置文件"""
        required_keys = ["PREPROCESS"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        preprocess_config = self.config["PREPROCESS"]
        required_preprocess_keys = [
            "INPUT_DIR",
            "OUTPUT_DIR",
            "RAW_DATA_DIR",
            "NUM_FRAMES",
            "NUM_CLIPS",
            "CROPS",
            "RESIZE",
            "MODEL",
        ]
        for key in required_preprocess_keys:
            if key not in preprocess_config:
                raise ValueError(f"Missing required preprocess config key: {key}")

    def _check_output_exists(self, video_path: str, output_dir: str, relative_path: str = "") -> bool:
        """检查输出文件是否已经存在"""
        try:
            video_name = Path(video_path).name
            modality_save_dir = self.config["PREPROCESS"]["MODALITY_SAVE_DIR"]
            output_subdir = os.path.join(output_dir, modality_save_dir)

            if relative_path:
                output_subdir = os.path.join(output_subdir, relative_path)

            output_path = os.path.join(output_subdir, f"{video_name}.npy")

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    return True
                else:
                    os.remove(output_path)
                    return False

            return False

        except Exception as e:
            logger.error(f"Error checking output file for {video_path}: {e}")
            return False

    def cpu_producer(self, video_files: List[Tuple[str, str]], output_dir: str):
        """CPU生产者 - 简化的两层for循环，直接生成segment级别任务"""
        logger.info(f"CPU producer started, processing {len(video_files)} videos")

        try:
            for video_info in tqdm(video_files, desc="Processing videos", unit="video"):
                # 检查是否需要停止
                if self.should_stop.is_set():
                    logger.info("CPU producer received stop signal")
                    break

                video_path, relative_path = video_info
                video_name = Path(video_path).name
                logger.info(f"Processing video: {video_name}")

                try:
                    # 创建具有正确输出目录的VideoProcessor
                    video_processor = VideoProcessor(
                        self.config, self.segment_len, output_dir, self.max_read_segment_in_block
                    )

                    # 使用VideoProcessor的生成器来处理视频
                    for task in video_processor.process_video(video_info):
                        # 检查是否需要停止
                        if self.should_stop.is_set():
                            logger.info("CPU producer received stop signal during video processing")
                            break

                        # 将任务放入队列
                        self._wait_for_queue_space_and_put(task)

                        if task["success"]:
                            logger.debug(f"CPU: Queued segment {task['segment_idx']} of {video_name}")
                        else:
                            logger.error(
                                f"CPU: Queued failed task for {video_name}: {task.get('error', 'Unknown error')}"
                            )

                except Exception as e:
                    logger.error(f"CPU: Error processing video {video_name}: {e}")
                    self.error_processed_videos.append(video_name)
                    continue

        except Exception as e:
            logger.error(f"Error in CPU producer: {e}")

        finally:
            # 发送结束信号给所有GPU workers
            if not self.should_stop.is_set():
                for _ in self.device_ids:
                    try:
                        self.task_queue.put(None, timeout=1)
                        logger.debug("CPU: Sent stop signal to GPU worker")
                    except queue.Full:
                        logger.warning("CPU: Failed to send stop signal, queue full")

            logger.info("CPU producer finished")

    def resource_monitor(self, interval: int = 10):
        """资源监控线程，定期输出CPU和内存使用情况，并监控内存阈值"""
        logger.info("Resource monitor started")

        while not self.should_stop.is_set():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                task_queue_size = self.task_queue.qsize()

                # 检查内存使用情况
                self._check_memory_usage()

                # 内存状态指示器
                memory_status = "⚠️ HIGH" if memory_percent > self.memory_threshold * 100 else "✅ OK"

                logger.info(
                    f"Resource Status - CPU: {cpu_percent:.1f}%, "
                    f"Memory: {memory_percent:.1f}% ({memory_status}), "
                    f"Task Queue: {task_queue_size}"
                )

                # 如果内存使用过高，额外记录详细信息
                if memory_percent > self.memory_threshold * 100:
                    self._log_memory_details()

                # 使用事件等待而不是sleep，这样可以立即响应停止信号
                self.should_stop.wait(timeout=interval)

            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                break

        logger.info("Resource monitor finished")

    def process_dataset(self):
        """处理整个数据集"""
        input_dir = self.config["PREPROCESS"]["INPUT_DIR"]
        raw_data_dir = self.config["PREPROCESS"]["RAW_DATA_DIR"]
        output_dir = self.config["PREPROCESS"]["OUTPUT_DIR"]

        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 复制指定的文件和目录
            if "COPY" in self.config["PREPROCESS"]:
                for src, dst in self.config["PREPROCESS"]["COPY"]:
                    src_path = os.path.join(input_dir, src)
                    dst_path = os.path.join(output_dir, dst)

                    if os.path.exists(src_path):
                        if os.path.isdir(src_path):
                            if os.path.exists(dst_path):
                                shutil.rmtree(dst_path)
                            shutil.copytree(src_path, dst_path)
                        else:
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                            shutil.copy2(src_path, dst_path)
                        logger.info(f"Copied {src_path} to {dst_path}")

            # 查找所有视频文件
            video_files = []
            video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

            for root, dirs, files in os.walk(raw_data_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(os.path.dirname(full_path), raw_data_dir)
                        if relative_path == ".":
                            relative_path = ""

                        # 如果启用resume模式，检查是否需要跳过
                        if self.resume and self._check_output_exists(full_path, output_dir, relative_path):
                            logger.debug(f"Skipping {file} - output file already exists")
                            continue

                        video_files.append((full_path, relative_path))

            logger.info(f"Found {len(video_files)} video files to process")

            if not video_files:
                logger.info("No video files to process")
                return

            # 启动GPU工作线程
            for device_id in self.device_ids:
                worker = GPUWorker(
                    device_id,
                    self.config,
                    self.task_queue,
                    self.should_stop,
                    self.global_video_locks,
                    self.global_video_processed_tags,
                    self.global_locks_lock,
                )
                self.gpu_workers.append(worker)
                thread = threading.Thread(target=worker.run, name=f"GPU-{device_id}")
                thread.start()
                self.gpu_threads.append(thread)

            # 启动资源监控线程（可选）
            monitor_thread = threading.Thread(target=self.resource_monitor, daemon=True)
            monitor_thread.start()

            # 启动CPU生产者（在主线程中运行）
            self.cpu_producer(video_files, output_dir)

            # 先等待队列中所有任务完成
            logger.info("CPU producer finished. Waiting for all tasks in queue to complete...")
            try:
                self.task_queue.join()
                logger.info("All tasks completed successfully")
            except Exception as e:
                logger.error(f"Error while waiting for tasks: {e}")

            logger.info("Dataset processing completed")
            logger.info(f"Processed {len(video_files) - len(self.error_processed_videos)} videos successfully")
            if self.error_processed_videos:
                logger.error(
                    f"Failed to process {len(self.error_processed_videos)} videos: {self.error_processed_videos}"
                )

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, FORCE stopping...")
            self.should_stop.set()
            self._cleanup()
            logger.info("Force exiting after KeyboardInterrupt...")
            os._exit(1)  # 强制退出
        except Exception as e:
            logger.error(f"Error in process_dataset: {e}")
            self.should_stop.set()
            self._cleanup()
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Streaming parallel video preprocessing with multi-GPU support")
    parser.add_argument("--config_path", required=True, help="Path to the configuration YAML file")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--max_queue",
        type=int,
        default=10,
        help="Maximum queue size for task buffering (default: 10)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume processing by skipping videos with existing output files"
    )
    parser.add_argument(
        "--memory_limit",
        type=float,
        default=0.8,
        help="Memory usage threshold (0.0-1.0). Processing will pause when memory exceeds this limit (default: 0.8)",
    )
    parser.add_argument(
        "--segment_len",
        type=int,
        default=1000,
        help="Number of clips per segment for memory management (default: 1000)",
    )
    parser.add_argument(
        "--max_read_segment_in_block",
        type=int,
        default=2,
        help="Maximum number of segments to read in a single block (default: 2)",
    )

    args = parser.parse_args()

    # 验证参数
    if not (0.0 <= args.memory_limit <= 1.0):
        parser.error("memory_limit must be between 0.0 and 1.0")

    if args.segment_len <= 0:
        parser.error("segment_len must be greater than 0")

    if args.max_queue <= 0:
        parser.error("max_queue must be greater than 0")

    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 设置multiprocessing启动方法
    mp.set_start_method("spawn", force=True)

    # 创建流式预处理器并运行
    preprocessor = None
    try:
        preprocessor = StreamingVideoPreprocessor(
            args.config_path,
            max_queue=args.max_queue,
            resume=args.resume,
            memory_threshold=args.memory_limit,
            segment_len=args.segment_len,
            max_read_segment_in_block=args.max_read_segment_in_block,
        )
        preprocessor.process_dataset()
        print("Streaming video preprocessing completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        if preprocessor:
            preprocessor._cleanup()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        if preprocessor:
            preprocessor._cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Example usage:

# 使用两块GPU和所有CPU核心, 设置内存阈值为80%
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python WSADBench/datasets/dataset_support/video_preprocess_streaming.py \
    --config_path WSADBench/datasets/dataset_configs/CV_by_I3D/UCF_Crime.prep.rgb.yaml \
    --resume --max_queue 10 --memory_limit 0.8 --segment_len 1000

注意：
- max_queue 控制待处理任务的最大队列长度, 默认值为10. 大队列可以保证GPU运行效率, 不出现GPU一直等待CPU处理的情况, 但会占用更多内存
根据可用内存和GPU数量调整 max_queue 值以达到最佳性能. 测试下来一般GPU数量的1-3倍即可。

如果队列中任务消耗过快, 而内存又充足，可以适当调大 segment_len 参数, 该参数控制每次处理视频中多少个clip。
而如果内存不足, 减小 segment_len 同时增多 max_queue 也是个可能的解决方案。

"""
