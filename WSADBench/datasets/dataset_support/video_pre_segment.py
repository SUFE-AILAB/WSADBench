from argparse import ArgumentParser
from pathlib import Path
import yaml
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# 由于配置文件路径从项目根目录开始，所以脚本也应从根目录运行，并设置正确的PYTHONPATH


def process_video_file(video_file, segmentation_dir, source_processed_dir, exclude_list, segment_num):
    """处理单个视频文件"""
    class_name = video_file.parent.name
    if class_name in exclude_list:
        return f"Skipped {video_file.name} (excluded class: {class_name})"
    
    try:
        arr = np.load(video_file, mmap_mode="r")
        if len(arr) == 0:
            return f"Skipped {video_file.name} (empty array)"
        
        segmented_arr = segment_average(arr, segment_num)
        
        # 保存处理后的数据
        save_path = segmentation_dir / video_file.relative_to(source_processed_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, segmented_arr)
        
        return f"Processed {video_file.name}"
    except Exception as e:
        return f"Error processing {video_file.name}: {str(e)}"


def segment_average(arr, num_segments):
    segs = np.linspace(0, len(arr), num_segments + 1, dtype=int)
    return np.stack(
        [
            np.mean(arr[segs[i] : segs[i + 1]], axis=0) if segs[i] != segs[i + 1] else arr[segs[i]]
            for i in range(num_segments)
        ],
        axis=0,
    )


def parse_args():
    parser = ArgumentParser(description="Video Pre-segment Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the dataset configuration file. eg: WSADBench/datasets/dataset_configs/CV_by_I3D/UCF_Crime.prep.rgb.yaml",
    )
    parser.add_argument("--segment_num", type=int, required=True, help="Number of segments to divide the video into.")
    parser.add_argument("--n_jobs", type=int, default=32, help="Number of parallel jobs (default: 32)")
    args = parser.parse_args()
    return args

"""
Example usage:
PYTHONPATH=. python WSADBench/datasets/dataset_support/video_pre_segment.py --config WSADBench/datasets/dataset_configs/CV_by_I3D/UCF_Crime.prep.rgb.yaml --segment_num 32 --n_jobs 64
"""
if __name__ == "__main__":
    args = parse_args()
    ds_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["PREPROCESS"]

    source_processed_dir = Path(ds_config["OUTPUT_DIR"]) / ds_config["MODALITY_SAVE_DIR"]
    segmentation_dir = (
        Path(ds_config["OUTPUT_DIR"])
        / ds_config["SEGMENTATION"]["SEGMENTATION_DIR"]
        / ds_config["MODALITY_SAVE_DIR"]
        / str(args.segment_num)
    )
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有需要处理的视频文件
    video_files = list(source_processed_dir.rglob("*.npy"))
    exclude_list = ds_config["SEGMENTATION"]["EXCLUDE_SEGMENT_LIST"]
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Using {args.n_jobs} parallel jobs")
    
    # 使用joblib并行处理，显示进度条
    results = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(process_video_file)(
            video_file, segmentation_dir, source_processed_dir, exclude_list, args.segment_num
        ) for video_file in tqdm(video_files, desc="Processing videos")
    )
    
    # 统计处理结果
    processed_count = sum(1 for result in results if result.startswith("Processed"))
    skipped_count = sum(1 for result in results if result.startswith("Skipped"))
    error_count = sum(1 for result in results if result.startswith("Error"))
    
    print(f"\nProcessing completed:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    
    # 显示错误详情（如果有）
    errors = [result for result in results if result.startswith("Error")]
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")
