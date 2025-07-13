from pathlib import Path
import numpy as np
from tqdm import tqdm
from WSADBench.datasets.dataset_support.video_pre_segment import segment_average

def segment_average(arr, num_segments):
    segs = np.linspace(0, len(arr), num_segments + 1, dtype=int)
    return np.stack(
        [
            np.mean(arr[segs[i] : segs[i + 1]], axis=0) if segs[i] != segs[i + 1] else arr[segs[i]]
            for i in range(num_segments)
        ],
        axis=0,
    )

class UCFCrimeManager:
    def __init__(self, ds_config):
        self.ds_config = ds_config
        self.wd = Path(ds_config["working_dir"])
        self.data_dir = self.wd / ds_config["DATA_DIR"]
        self.processed_dir = self.data_dir / ds_config["MODALITY_DIR"]
        self.segmentation_dir = self.data_dir / ds_config["SEGMENTATION_DIR"] / ds_config["MODALITY_DIR"]
        self.seed = ds_config.get("seed", 0)

    def get_split(self):
        split_dir = self.data_dir / self.ds_config["SPLIT_DIR"]
        split_file = self.ds_config["SPLIT_FILE"]
        if isinstance(split_file, list):
            split_file = split_file[self.seed % len(split_file)]

        splits = {}
        for split_tag, file_name in split_file.items():
            with open(split_dir / file_name, "r") as f:
                splits[split_tag] = [line.strip() for line in f.readlines()]

        return splits

    def load_data(self, limit=None, num_segments=None):
        if num_segments is not None:
            load_dir = self.segmentation_dir / str(num_segments)
            if not self.segmentation_dir.exists():
                raise FileNotFoundError(f"Segmented dir {load_dir} does not exist. You may need to run video_pre_segment.py first.")
        else:
            load_dir = self.processed_dir
                    
        def load_train_split(files):
            arrs, ys, vid_ids = [], [], []
            for video_idx, file_name in enumerate(tqdm(files, desc="Loading training data")):
                file_path = load_dir / f"{file_name}.npy"
                kind = file_path.parent.name
                if not file_path.exists():
                    continue

                arr = np.load(file_path, mmap_mode="r")
                if limit is not None:
                    arr = arr[:limit]

                label = 0 if 'Normal' in kind else 1
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), video_idx, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                vid_ids.append(vid_id)
            print(f"\nLoaded {len(arrs)} training files from {load_dir}")
            return {
                "X_train": np.concatenate(arrs, axis=0),
                "y_train": np.concatenate(ys, axis=0),
                "vid_train": np.concatenate(vid_ids, axis=0),
            }

        def load_test_split(files):
            arrs, ys, y_gt, y_idx, y_gt_idx, vid_ids = [], [], [], [], [], []
            for i, file_name in enumerate(tqdm(files, desc="Loading test data")):
                file_path = self.processed_dir / f"{file_name}.npy"
                kind = file_path.parent.name
                if not file_path.exists():
                    continue

                arr = np.load(file_path, mmap_mode="r")
                idx = np.full(len(arr), i, dtype=np.int32)
                

                label = 0 if 'Normal' in kind else 1
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), i, dtype=np.int32)  # 测试集中视频ID就是文件索引

                name = file_path.stem
                truth = ground_truth_dict[name]["truth"]
                gt_idx = np.full(len(truth), i, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                y_gt.append(truth)
                y_idx.append(idx)
                y_gt_idx.append(gt_idx)
                vid_ids.append(vid_id)
            print(f"\nLoaded {len(arrs)} test files from {self.processed_dir}")
            return {
                "X_test": np.concatenate(arrs, axis=0),
                "y_test": np.concatenate(ys, axis=0),
                "y_test_gt": np.concatenate(y_gt, axis=0),
                "y_test_idx": np.concatenate(y_idx, axis=0),
                "y_test_gt_idx": np.concatenate(y_gt_idx, axis=0),
                "vid_test": np.concatenate(vid_ids, axis=0),
            }

        # 主体逻辑
        splits = self.get_split()
        ground_truth_dict = self.load_ground_truth()
        data = {}

        if "train" in splits:
            data.update(load_train_split(splits["train"]))

        if "test" in splits:
            data.update(load_test_split(splits["test"]))

        data['NUM_FRAMES'] = self.ds_config["NUM_FRAMES"]
        return data

    def load_ground_truth(self):
        # Logic to load ground truth for UCF Crime dataset
        with open(self.data_dir / self.ds_config["LABEL_LIST"], "r") as f:
            ground_truth_list = [line.strip().split() for line in f.readlines()]
        groud_truth = {}
        for row in ground_truth_list:
            file_path = Path(row[0])
            file_name = file_path.name
            all_frames = int(row[1])
            label_kind = row[2]
            truth = np.zeros(all_frames, dtype=np.int32)

            for i in range(3, len(row), 2):
                start_frame = int(row[i])
                end_frame = int(row[i + 1])
                truth[start_frame : end_frame + 1] = 1
            groud_truth[file_name] = {
                "file_path": file_path,
                "all_frames": all_frames,
                "label_kind": label_kind,
                "truth": truth,
            }

        return groud_truth


if __name__ == "__main__":
    ds_config = {
        "working_dir": "/data/coding/wsad/yx/WSADBench/",
        "DATA_DIR": "WSADBench/datasets/CV_by_I3D/UCF_Crime/",
        "SPLIT_DIR": "splits",
        "SPLIT_FILE": {"train": "Anomaly_Train.txt", "test": "Anomaly_Test.txt"},
        "LABEL_LIST": "UCF_Annotation.txt",
        "seed": 0,
        "MODALITY": "RGB",
        "MODALITY_DIR": "all_rgbs",
        "NUM_FRAMES": 16,
    }
    manager = UCFCrimeManager(ds_config)
    data = manager.load_data(limit=32)
    # data = manager.load_data(num_segments=200)
    print(data.keys())
    print(data["X_train"].shape, data["y_train"].shape, data["vid_train"].shape)
    print(data["X_test"].shape, data["y_test"].shape, data["vid_test"].shape)
    
    # 显示视频ID分布示例
    print("\n训练集视频ID分布示例:")
    print("vid_train前20个:", data["vid_train"][:20])
    print("训练集包含视频数量:", len(np.unique(data["vid_train"])))
    
    print("\n测试集视频ID分布示例:")
    print("vid_test前20个:", data["vid_test"][:20])
    print("测试集包含视频数量:", len(np.unique(data["vid_test"])))
