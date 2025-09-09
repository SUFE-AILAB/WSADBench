import cv2
import os
from pathlib import Path
from typing import List, Tuple
import tempfile
import shutil
import atexit
from tqdm import tqdm
class FrameToVideoConverter:
    """将帧序列转换为视频文件的工具类"""

    def __init__(self, fps: int = 30, temp_dir: str = None, keep_videos: bool = True):
        self.fps = fps
        self.keep_videos = keep_videos  # 是否保留已处理的视频
        self.temp_dir = '/tmp/shanghai_videos'

        # 如果选择不保留视频，才删除已存在的临时目录
        if not self.keep_videos and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Temporary video directory: {self.temp_dir}")

        if self.keep_videos:
            print("Note: Videos will be kept after processing completes")
        else:
            print("Note: Videos will be cleaned up after processing completes")

        # 只有在不保留视频时才注册清理函数
        if not self.keep_videos:
            atexit.register(self.cleanup)

    def cleanup(self):
        """清理临时文件夹"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")

    def convert_frames_to_video(self, frames_dir: str, output_video_path: str, frame_files_list: List[str] = None) -> bool:
        """
        将帧序列转换为视频文件

        Args:
            frames_dir: 包含帧图片的目录
            output_video_path: 输出视频文件路径

        Returns:
            bool: 转换成功返回True，失败返回False
        """
        try:
            if frame_files_list is not None:
                # 使用提供的文件列表（已包含完整路径）
                frame_paths = frame_files_list
            else:
                # 原有逻辑：从单个目录获取文件
                frame_files = []
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

                for file in os.listdir(frames_dir):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        frame_files.append(file)

                if not frame_files:
                    print(f"No image files found in {frames_dir}")
                    return False

                # 对帧文件进行数字排序
                frame_files.sort(key=lambda x: self._extract_frame_number(x))
                frame_paths = [os.path.join(frames_dir, f) for f in frame_files]

            if not frame_paths:
                print(f"No image files found")
                return False

            # 调试输出：显示前几个和后几个文件名以确认排序
            if len(frame_paths) > 10:
                print(f"Frame sorting check - First 5: {[os.path.basename(p) for p in frame_paths[:5]]}, "
                      f"Last 5: {[os.path.basename(p) for p in frame_paths[-5:]]}")
            else:
                print(f"Frame files: {[os.path.basename(p) for p in frame_paths]}")

            # 读取第一帧以获取视频尺寸
            first_frame = cv2.imread(frame_paths[0])
            if first_frame is None:
                print(f"Cannot read first frame: {frame_paths[0]}")
                return False

            height, width, _ = first_frame.shape

            # 创建输出目录
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

            # 创建视频写入器 - 使用XVID编解码器以确保兼容性
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (width, height))

            if not video_writer.isOpened():
                print(f"Cannot open video writer for {output_video_path}")
                return False

            # 写入每一帧
            frames_written = 0
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # 确保帧尺寸一致
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    video_writer.write(frame)
                    frames_written += 1
                else:
                    print(f"Warning: Cannot read frame: {frame_path}")

            video_writer.release()

            # 验证视频文件是否创建成功
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                print(f"Successfully created video: {output_video_path} ({frames_written} frames)")
                return True
            else:
                print(f"Failed to create video file: {output_video_path}")
                return False

        except Exception as e:
            print(f"Error converting frames to video {output_video_path}: {e}")
            return False

    def _extract_frame_number(self, filename: str) -> int:
        """从文件名中提取帧号用于排序，专门优化TAD数据集的帧命名"""
        import re

        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]

        # TAD数据集的帧通常命名为: 0.jpg, 1.jpg, 2.jpg, ...
        # 或者: frame_0001.jpg, frame_0002.jpg, ...

        # 首先尝试提取纯数字的文件名
        if name_without_ext.isdigit():
            return int(name_without_ext)

        # 尝试提取所有数字，使用最后一个作为帧号
        numbers = re.findall(r'\d+', name_without_ext)
        if numbers:
            # 对于TAD数据集，通常最后一个数字是帧号
            return int(numbers[-1])

        # 如果没有找到数字，尝试按字符串排序
        return hash(filename) % 1000000  # 返回一个基于文件名的稳定数字

    def _video_exists_and_valid(self, video_path: str, expected_frames: int = None) -> bool:
        """
        检查视频文件是否存在且有效

        Args:
            video_path: 视频文件路径
            expected_frames: 期望的帧数，如果提供则会验证帧数是否匹配

        Returns:
            bool: 视频存在且有效返回True
        """
        if not os.path.exists(video_path):
            return False

        # 检查文件大小，空文件视为无效
        if os.path.getsize(video_path) == 0:
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            # 如果提供了期望帧数，验证是否匹配
            if expected_frames is not None:
                actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # 允许小幅差异（±1帧）
                return abs(actual_frames - expected_frames) <= 1

            cap.release()
            return True

        except Exception as e:
            print(f"Error checking video validity {video_path}: {e}")
            return False

    def process_shanghai_dataset(self, raw_data_dir: str) -> List[Tuple[str, str]]:
        """
        处理Shanghai数据集，将帧序列转换为视频文件

        Args:
            raw_data_dir: 原始数据目录

        Returns:
            List[Tuple[str, str]]: 视频文件路径和相对路径的列表
        """
        video_files = []

        # 首先统计包含图片文件的目录总数
        print("Scanning directories for frame sequences...")
        frame_dirs = []

        for root, dirs, files in os.walk(raw_data_dir):
            # 检查当前目录是否包含图片文件
            image_files = [f for f in files if any(f.lower().endswith(ext)
                                                   for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])]

            if image_files:
                frame_dirs.append((root, len(image_files)))

        print(f"Found {len(frame_dirs)} directories containing frame sequences")

        if not frame_dirs:
            print("No frame sequences found to convert")
            return video_files

        # 使用进度条处理每个包含帧的目录
        for root, num_frames in tqdm(frame_dirs, desc="Converting frame sequences to videos", unit="video"):
            # 这是一个包含帧的目录，需要转换为视频
            relative_root = os.path.relpath(root, raw_data_dir)
            video_name = os.path.basename(root) + ".avi"

            # 构建临时视频文件路径，保持相对目录结构
            temp_video_dir = os.path.join(self.temp_dir, os.path.dirname(relative_root))
            os.makedirs(temp_video_dir, exist_ok=True)
            temp_video_path = os.path.join(temp_video_dir, video_name)

            tqdm.write(f"Converting {num_frames} frames from {os.path.basename(root)} to {video_name}")

            if self.convert_frames_to_video(root, temp_video_path):
                # 转换成功，添加到视频文件列表
                relative_path = os.path.dirname(relative_root) if os.path.dirname(relative_root) != "." else ""
                video_files.append((temp_video_path, relative_path))
                tqdm.write(f"✅ Successfully converted {video_name}")
            else:
                tqdm.write(f"❌ Failed to convert frames from {os.path.basename(root)}")

        print(f"Frame to video conversion completed: {len(video_files)} videos created")
        return video_files


    def process_tad_dataset(self, raw_data_dir: str) -> List[Tuple[str, str]]:
        """
        处理TAD数据集，将帧序列转换为视频文件
        TAD数据集格式：TAD/raw_data/frames_low/video_name/0.jpg, 1.jpg, ...
        raw_data文件夹下有3个子文件夹：frames_low, frames_out, new
        这些子文件夹下存着视频文件夹

        Args:
            raw_data_dir: 原始数据目录

        Returns:
            List[Tuple[str, str]]: 视频文件路径和相对路径的列表
        """
        video_files = []

        # TAD数据集的帧存储在raw_data下的三个子目录中
        frame_subdirs = ["frames_low", "frames_out", "new"]

        print("Scanning TAD frame directories...")
        frame_dirs = []
        name_list = []  # 排除多余视频
        # 遍历所有帧子目录
        for subdir in frame_subdirs:
            frames_dir = os.path.join(raw_data_dir, subdir)

            if not os.path.exists(frames_dir):
                print(f"TAD frames directory not found: {frames_dir}")
                continue

            print(f"Processing frames in: {frames_dir}, video_num:{len(os.listdir(frames_dir))}")

            # 遍历该子目录下的所有视频目录
            for video_dir in os.listdir(frames_dir):
                video_dir_path = os.path.join(frames_dir, video_dir)
                name_list.append(video_dir)
                # 检查是否为目录且包含图片文件
                if os.path.isdir(video_dir_path):
                    jpg_files = []
                    for root, dirs, files in os.walk(video_dir_path):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                jpg_files.append(os.path.join(root, file))

                    # 按文件名排序以确保正确的帧顺序
                    if jpg_files:
                        jpg_files.sort(key=lambda x: self._extract_frame_number(os.path.basename(x)))
                        # 添加到处理列表，包含子目录信息
                        frame_dirs.append((video_dir_path, len(jpg_files), video_dir, subdir))

        print(f"Found {len(frame_dirs)} TAD video directories containing frame sequences")

        if not frame_dirs:
            print("No TAD frame sequences found to convert")
            return video_files

        # 排查多余视频
        raw_list = []
        try:
            with open(
                    r'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/source_datasets/TAD/splits/Anomaly_train.txt',
                    'r') as f:
                for line in f:
                    raw_list.append(line.strip())
            with open(r'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/source_datasets/TAD/splits/Anomaly_test.txt',
                      'r') as f:
                for line in f:
                    raw_list.append(line.strip())
        except FileNotFoundError as e:
            print(f"Warning: Split files not found: {e}")
            print("Proceeding without filtering videos")

        # 对比raw_list和name_list的差异
        if raw_list:
            print("\nAnalyzing video list differences...")

            # 统一格式：从raw_list中提取视频文件名（去掉子目录前缀）
            # raw_list格式：frames_out/01_Accident_004.mp4
            # name_list格式：Normal_001.mp4
            raw_video_names = set()
            for raw_path in raw_list:
                # 提取文件名部分（去掉路径前缀）
                video_filename = os.path.basename(raw_path)
                raw_video_names.add(video_filename)

            # name_list已经是视频文件名格式
            found_video_names = set(name_list)

            # 统计信息
            print(f"Videos in split files: {len(raw_video_names)}")
            print(f"Videos found in directories: {len(found_video_names)}")

            # 找出差异
            missing_videos = raw_video_names - found_video_names  # 在split文件中但目录中没有的
            extra_videos = found_video_names - raw_video_names  # 在目录中但split文件中没有的
            common_videos = raw_video_names & found_video_names  # 共同的视频

            print(f"Common videos: {len(common_videos)}")
            print(f"Missing videos (in splits but not in directories): {len(missing_videos)}")
            print(f"Extra videos (in directories but not in splits): {len(extra_videos)}")

            if missing_videos:
                print(f"Missing videos: {sorted(list(missing_videos))[:10]}...")  # 只显示前10个

            if extra_videos:
                print(f"Extra videos: {sorted(list(extra_videos))[:10]}...")  # 只显示前10个

            # 过滤frame_dirs，只保留在split文件中的视频
            filtered_frame_dirs = []
            for video_dir_path, num_frames, video_name, subdir in frame_dirs:
                if video_name in raw_video_names:
                    filtered_frame_dirs.append((video_dir_path, num_frames, video_name, subdir))
                else:
                    print(f"Skipping extra video: {subdir}/{video_name}")

            frame_dirs = filtered_frame_dirs
            print(f"After filtering: {len(frame_dirs)} videos will be processed")

        # 统计已存在和需要转换的视频
        existing_videos = 0
        to_convert = []

        for video_dir_path, num_frames, video_name, subdir in frame_dirs:
            # 构建临时视频文件路径
            temp_video_dir = os.path.join(self.temp_dir, subdir)
            os.makedirs(temp_video_dir, exist_ok=True)
            temp_video_path = os.path.join(temp_video_dir, video_name)

            # 检查视频是否已存在且有效
            if self._video_exists_and_valid(temp_video_path, num_frames):
                existing_videos += 1
                video_files.append((temp_video_path, subdir))
                print(f"⏭️  Found existing video: {subdir}/{video_name} ({num_frames} frames)")
            else:
                to_convert.append((video_dir_path, num_frames, video_name, subdir))

        print(f"\nVideo processing summary:")
        print(f"  Already converted: {existing_videos} videos")
        print(f"  To be converted: {len(to_convert)} videos")
        print(f"  Total: {len(frame_dirs)} videos")

        # 处理需要转换的视频
        if to_convert:
            for video_dir_path, num_frames, video_name, subdir in tqdm(to_convert,
                                                                       desc="Converting TAD frame sequences to videos",
                                                                       unit="video"):

                clean_video_name = video_name

                # 构建相对路径（包含子目录信息）
                relative_path = subdir

                # 构建临时视频文件路径
                temp_video_dir = os.path.join(self.temp_dir, relative_path)
                os.makedirs(temp_video_dir, exist_ok=True)
                temp_video_path = os.path.join(temp_video_dir, clean_video_name)

                display_name = f"{subdir}/{video_name}"
                tqdm.write(f"Converting {num_frames} frames from {display_name} to {clean_video_name}")

                # 获取该视频目录下的所有图片文件（递归）
                jpg_files = []
                for root, dirs, files in os.walk(video_dir_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            jpg_files.append(os.path.join(root, file))

                # 按文件名排序以确保正确的帧顺序
                if jpg_files:
                    jpg_files.sort(key=lambda x: self._extract_frame_number(os.path.basename(x)))

                    # 使用文件列表调用转换函数
                    if self.convert_frames_to_video(video_dir_path, temp_video_path, jpg_files):
                        # 转换成功
                        video_files.append((temp_video_path, relative_path))
                        tqdm.write(f"✅ Successfully converted {clean_video_name}")
                    else:
                        tqdm.write(f"❌ Failed to convert frames from {display_name}")

        print(f"TAD frame to video conversion completed: {len(video_files)} videos available")
        return video_files

    def process_ucsd_ped2_dataset(self, raw_data_dir: str) -> List[Tuple[str, str]]:
        """处理UCSD_Ped2数据集，将tiff图片序列转换为视频文件

        Args:
            raw_data_dir: UCSD_Ped2原始数据目录路径

        Returns:
            转换后的视频文件列表，每个元素为(视频路径, 相对路径)元组
        """
        video_files = []
        raw_data_path = Path(raw_data_dir)

        if not raw_data_path.exists():
            print(f"Raw data directory not found: {raw_data_dir}")
            return video_files

        # UCSD_Ped2数据集结构: Train/Test -> 视频文件夹 -> tiff文件
        for split_dir in ['Train', 'Test']:
            split_path = raw_data_path / split_dir
            if not split_path.exists():
                print(f"Split directory not found: {split_path}")
                continue

            print(f"Processing {split_dir} split...")

            for video_dir in split_path.iterdir():
                if not video_dir.is_dir():
                    continue

                try:
                    # 查找tiff文件
                    tiff_files = sorted(list(video_dir.glob("*.tif")) + list(video_dir.glob("*.tiff")))

                    if not tiff_files:
                        print(f"No tiff files found in {video_dir}")
                        continue

                    print(f"Converting {len(tiff_files)} tiff files from {video_dir.name}")

                    # 读取第一帧以获取图像尺寸
                    first_frame = cv2.imread(str(tiff_files[0]), cv2.IMREAD_GRAYSCALE)
                    if first_frame is None:
                        print(f"Cannot read first frame: {tiff_files[0]}")
                        continue

                    height, width = first_frame.shape

                    # 创建输出视频路径
                    relative_path = str(Path(split_dir) / video_dir.name)
                    output_subdir = f'{self.temp_dir}/{split_dir}'
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = f'{output_subdir}/{video_dir.name}.mp4'

                    # 创建视频编写器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        self.fps,
                        (width, height),
                        isColor=False  # UCSD_Ped2通常是灰度图像
                    )

                    if not out.isOpened():
                        print(f"Cannot create video writer for {output_path}")
                        continue

                    # 写入帧
                    frames_written = 0
                    for tiff_file in tiff_files:
                        frame = cv2.imread(str(tiff_file), cv2.IMREAD_GRAYSCALE)
                        if frame is None:
                            print(f"Cannot read frame: {tiff_file}")
                            continue

                        # 确保帧尺寸一致
                        if frame.shape != (height, width):
                            frame = cv2.resize(frame, (width, height))

                        out.write(frame)
                        frames_written += 1

                    out.release()

                    if frames_written > 0:
                        video_files.append((str(output_path), relative_path))
                        print(
                            f"Successfully converted {video_dir.name}: {frames_written} frames -> {output_path}")
                    else:
                        print(f"No frames written for {video_dir.name}")
                        if output_path.exists():
                            output_path.unlink()  # 删除空视频文件

                except Exception as e:
                    print(f"Error processing video directory {video_dir}: {e}")
                    continue

        print(f"UCSD_Ped2 conversion completed. Converted {len(video_files)} videos.")
        return video_files