from pathlib import Path
import numpy as np
from tqdm import tqdm

class UCFCrimeManager:
    def __init__(self, ds_config):
        self.ds_config = ds_config
        self.wd = Path(ds_config["working_dir"])
        self.data_dir = self.wd / ds_config["DATA_DIR"]
        self.processed_dir = self.data_dir / ds_config["MODALITY_DIR"]
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

    def load_data(self, limit_clips=None, num_segments=None):
        if limit_clips is not None and num_segments is not None:
            raise ValueError("Only one of limit_clips or num_segments can be specified.")

        def segment_average(arr, num_segments):
            segs = np.linspace(0, len(arr), num_segments + 1, dtype=int)
            return np.stack(
                [
                    np.mean(arr[segs[i] : segs[i + 1]], axis=0) if segs[i] != segs[i + 1] else arr[segs[i]]
                    for i in range(num_segments)
                ],
                axis=0,
            )

        def load_train_split(files):
            arrs, ys = [], []
            for file_name in tqdm(files, desc="Loading training data"):
                file_path = self.processed_dir / f"{file_name}.npy"
                kind = file_path.parent.name
                if not file_path.exists():
                    continue

                arr = np.load(file_path, mmap_mode="r")
                if limit_clips is not None:
                    arr = arr[:limit_clips]
                elif num_segments is not None:
                    arr = segment_average(arr, num_segments)

                label = 0 if kind in NormalKind else 1
                y = np.full(len(arr), label, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
            return {
                "X_train": np.concatenate(arrs, axis=0),
                "y_train": np.concatenate(ys, axis=0),
            }

        def load_test_split(files):
            arrs, ys, y_gt, y_idx, y_gt_idx = [], [], [], [], []
            for i, file_name in enumerate(tqdm(files, desc="Loading test data")):
                file_path = self.processed_dir / f"{file_name}.npy"
                kind = file_path.parent.name
                if not file_path.exists():
                    continue

                arr = np.load(file_path, mmap_mode="r")
                idx = np.full(len(arr), i, dtype=np.int32)
                

                label = 0 if kind in NormalKind else 1
                y = np.full(len(arr), label, dtype=np.int32)

                name = file_path.stem
                truth = ground_truth_dict[name]["truth"]
                gt_idx = np.full(len(truth), i, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                y_gt.append(truth)
                y_idx.append(idx)
                y_gt_idx.append(gt_idx)
            return {
                "X_test": np.concatenate(arrs, axis=0),
                "y_test": np.concatenate(ys, axis=0),
                "y_test_gt": np.concatenate(y_gt, axis=0),
                "y_test_idx": np.concatenate(y_idx, axis=0),
                "y_test_gt_idx": np.concatenate(y_gt_idx, axis=0),
            }

        # 主体逻辑
        NormalKind = {"Testing_Normal_Video_Anomaly", "Training_Normal_Video_Anomaly", "Normal_Videos_event"}
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
        "working_dir": "/data/coding/yx/WSADBench/",
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
    data = manager.load_data(limit_clips=32)
    # data = manager.load_data(num_segments=200)
    print(data.keys())
    print(data["X_train"].shape, data["y_train"].shape)
    print(data["X_test"].shape, data["y_test"].shape)
