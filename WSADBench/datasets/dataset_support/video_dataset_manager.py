import re
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


def get_pretrain_model_path(path, pretrain_model):
    param_dict = {'i3d': 'CV_by_I3D', 'mvit': 'CV_by_MViT_32', 'sf': 'CV_by_SlowFast'}

    if pretrain_model not in param_dict:
        raise ValueError(f"Pretrain model {pretrain_model} not found in {param_dict}")

    path_str = str(path)
    parts = path_str.split('/')

    # 替换倒数第二个部分
    if len(parts) >= 2:
        parts[-2] = param_dict[pretrain_model]
        return Path('/'.join(parts))
    else:
        raise ValueError(f"Path {path} doesn't have enough parts to replace model directory")

class UCFCrimeManager:
    def __init__(self, ds_config):
        self.ds_config = ds_config
        self.wd = Path(ds_config["working_dir"])
        self.data_dir = self.wd / ds_config[
            "DATA_DIR"] if 'pretrain_model' not in ds_config else get_pretrain_model_path(
            self.wd / ds_config["DATA_DIR"], ds_config["pretrain_model"])
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

    def load_data(self, limit=None, num_segments=32):
        if num_segments is not None:
            load_dir = self.segmentation_dir / str(num_segments)
            if not self.segmentation_dir.exists():
                raise FileNotFoundError(f"Segmented dir {load_dir} does not exist. You may need to run video_pre_segment.py first.")
        else:
            load_dir = self.processed_dir

        def fast_concatenate_axis0_variable(arrs):
            if not arrs:
                return np.array([])

            base_shape = arrs[0].shape[1:]
            for a in arrs:
                if a.shape[1:] != base_shape:
                    raise ValueError("All arrays must have the same shape except along axis=0.")

            total_rows = sum(a.shape[0] for a in arrs)
            result = np.empty((total_rows, *base_shape), dtype=arrs[0].dtype)

            idx = 0
            for a in arrs:
                n = a.shape[0]
                result[idx:idx + n] = a
                idx += n

            return result
        def load_train_split(files):
            arrs, ys, vid_ids = [], [], []
            vid_kind = {}  # 视频标签种类字典
            vid_source_clips_num = {}  # 视频原始片段数量字典
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

                vid_kind[video_idx] = kind if 'Normal' not in kind else 'Normal'
                source_arr = np.load(self.processed_dir / f"{file_name}.npy", mmap_mode="r")
                vid_source_clips_num[video_idx] = source_arr.shape[0]
            print(f"Loaded {len(arrs)} training files from {load_dir}")
            return {
                "X_train": fast_concatenate_axis0_variable(arrs),  # 10min, 7min
                "y_train": fast_concatenate_axis0_variable(ys),  #
                "vid_train": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_train": vid_kind,
                "vid_source_clips_num_train": vid_source_clips_num,
            }

        def load_test_split(files):
            arrs, ys, y_gt, y_idx, y_gt_idx, vid_ids = [], [], [], [], [], []
            vid_kind = {}
            vid_source_clips_num = {}
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

                vid_kind[i] = kind if 'Normal' not in kind else 'Normal'
                source_arr = np.load(self.processed_dir / f"{file_name}.npy", mmap_mode="r")
                vid_source_clips_num[i] = source_arr.shape[0]
            print(f"Loaded {len(arrs)} test files from {self.processed_dir}")
            return {
                "X_test": fast_concatenate_axis0_variable(arrs),
                "y_test": fast_concatenate_axis0_variable(ys),
                "y_test_gt": fast_concatenate_axis0_variable(y_gt),
                "y_test_idx": fast_concatenate_axis0_variable(y_idx),
                "y_test_gt_idx": fast_concatenate_axis0_variable(y_gt_idx),
                "vid_test": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_test": vid_kind,
                "vid_source_clips_num_test": vid_source_clips_num,
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


class ShanghaitechManager:
    def __init__(self, ds_config):
        self.ds_config = ds_config
        self.wd = Path(ds_config["working_dir"])
        self.data_dir = self.wd / ds_config["DATA_DIR"] if 'pretrain_model' not in ds_config else get_pretrain_model_path(self.wd / ds_config["DATA_DIR"], ds_config["pretrain_model"])
        self.processed_dir = self.data_dir / ds_config["MODALITY_DIR"]
        self.segmentation_dir = self.data_dir / ds_config["SEGMENTATION_DIR"] / ds_config["MODALITY_DIR"]
        # self.frame_labels_dir = self.data_dir / ds_config["FRAME_LABELS_DIR"]
        self.seed = ds_config.get("seed", 0)

    def get_split(self):
        """获取数据集划分信息"""
        split_dir = self.data_dir / self.ds_config["SPLIT_DIR"]
        split_file = self.ds_config["SPLIT_FILE"]
        if isinstance(split_file, list):
            split_file = split_file[self.seed % len(split_file)]

        splits = {}
        for split_tag, file_name in split_file.items():
            split_path = split_dir / file_name
            if split_path.exists():
                with open(split_path, "r") as f:
                    splits[split_tag] = [line.strip() for line in f.readlines()]
            else:
                # 如果分割文件不存在，根据文件夹结构自动生成
                splits[split_tag] = self._auto_generate_split(split_tag)

        return splits

    def _auto_generate_split(self, split_tag):
        """自动生成数据集划分（如果分割文件不存在）"""
        video_files = []
        raw_data_dir = self.data_dir.parent.parent / "source_datasets/Shanghai/raw_data"

        # 遍历training和testing文件夹
        if split_tag == "train":
            search_dirs = ["training"]
        else:  # test
            search_dirs = ["testing"]

        for dir_name in search_dirs:
            dir_path = raw_data_dir / dir_name
            if dir_path.exists():
                for subdir in dir_path.iterdir():
                    if subdir.is_dir():
                        for video_file in subdir.glob("*.avi"):
                            # 保存相对于raw_data的路径
                            relative_path = video_file.relative_to(raw_data_dir)
                            video_files.append(str(relative_path.with_suffix('')))

        return video_files

    def load_ground_truth(self):
        """加载Shanghai数据集的帧级别标注"""
        ground_truth = {}

        with open(self.data_dir / self.ds_config["LABEL_LIST"], "r") as f:
            ground_truth_list = [line.strip().split() for line in f.readlines()]
        groud_truth = {}
        for row in ground_truth_list:
            file_path = Path(row[0])
            # 修正：使用 Path 对象的方法来检查路径中是否包含 'frames'
            if 'frames' in file_path.parts or 'frames' in file_path.stem:
                file_name = file_path.stem + '.avi'  # 如果路径包含 frames，添加 .avi 后缀
            else:
                file_name = file_path.name  # 否则直接使用文件名
            all_frames = int(row[1])
            label_kind = row[2]
            truth = np.zeros(all_frames, dtype=np.int32)

            for i in range(3, len(row), 2):
                start_frame = int(row[i])
                end_frame = int(row[i + 1])
                truth[start_frame: end_frame + 1] = 1
            groud_truth[file_name] = {
                "file_path": file_path,
                "all_frames": all_frames,
                "label_kind": label_kind,
                "truth": truth,
            }

        return groud_truth



    def load_data(self, limit=None, num_segments=None):
        """加载Shanghai数据集数据"""
        if num_segments is not None:
            load_dir = self.segmentation_dir / str(num_segments)
            if not load_dir.exists():
                raise FileNotFoundError(
                    f"Segmented dir {load_dir} does not exist. You may need to run video_pre_segment.py first.")
        else:
            load_dir = self.processed_dir

        def fast_concatenate_axis0_variable(arrs):
            if not arrs:
                return np.array([])

            base_shape = arrs[0].shape[1:]
            for a in arrs:
                if a.shape[1:] != base_shape:
                    raise ValueError("All arrays must have the same shape except along axis=0.")

            total_rows = sum(a.shape[0] for a in arrs)
            result = np.empty((total_rows, *base_shape), dtype=arrs[0].dtype)

            idx = 0
            for a in arrs:
                n = a.shape[0]
                result[idx:idx + n] = a
                idx += n

            return result

        def load_train_split(files):
            arrs, ys, vid_ids = [], [], []
            vid_kind = {}
            vid_source_clips_num = {}

            for video_idx, file_name in enumerate(tqdm(files, desc="Loading Shanghai training data")):
                # 处理可能的路径分隔符
                file_parts = Path(file_name).parts

                # 构建完整的文件路径
                file_path = load_dir
                for part in file_parts: # 用 / 拼接路径是 pathlib 的语法糖，等价于 file_path.joinpath(part)。
                    file_path = file_path / part
                file_path = file_path.with_suffix('.avi.npy')  # 即使原路径有别的后缀（比如 .txt），它也会被替换成 .npy。

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                if limit is not None:
                    arr = arr[:limit]

                # Shanghai数据集标签：training文件夹包含正常和异常视频
                # 根据文件路径判断类别
                if 'videos' in str(file_path):
                    label = 0  # training数据作为正常类别
                else:
                    label = 1  # 其他作为异常类别

                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), video_idx, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                vid_ids.append(vid_id)

                vid_kind[video_idx] = 'Normal' if label == 0 else 'Abnormal'

                # 获取原始clip数量
                source_file_path = self.processed_dir
                for part in file_parts:
                    source_file_path = source_file_path / part
                source_file_path = source_file_path.with_suffix('.npy')

                if source_file_path.exists():
                    source_arr = np.load(source_file_path, mmap_mode="r")
                    vid_source_clips_num[video_idx] = source_arr.shape[0]
                else:
                    vid_source_clips_num[video_idx] = len(arr)

            print(f"Loaded {len(arrs)} Shanghai training files from {load_dir}")
            return {
                "X_train": fast_concatenate_axis0_variable(arrs),
                "y_train": fast_concatenate_axis0_variable(ys),
                "vid_train": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_train": vid_kind,
                "vid_source_clips_num_train": vid_source_clips_num,
            }

        def load_test_split(files):
            arrs, ys, y_gt, y_idx, y_gt_idx, vid_ids = [], [], [], [], [], []
            vid_kind = {}
            vid_source_clips_num = {}
            ground_truth_dict = self.load_ground_truth()

            for i, file_name in enumerate(tqdm(files, desc="Loading Shanghai test data")):
                file_parts = Path(file_name).parts

                file_path = self.processed_dir
                for part in file_parts:
                    file_path = file_path / part
                file_path = file_path.with_suffix('.avi.npy')  # 即使原路径有别的后缀（比如 .txt），它也会被替换成 .npy。

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                idx = np.full(len(arr), i, dtype=np.int32)
                name = file_path.stem
                # 测试集包含正常和异常视频，根据ground truth确定标签
                # video_stem = Path(file_name).stem
                # if video_stem in ground_truth_dict:
                #     # 使用ground truth
                #     truth = ground_truth_dict[video_stem]["truth"]
                #     # 视频级别标签：如果包含任何异常帧则为异常
                #     label = 1 if np.any(truth == 1) else 0
                # else:
                #     # 如果没有ground truth，根据路径判断
                #     label = 0 if 'testing' in str(file_path) else 1
                #     # 创建默认的frame-level truth
                #     truth = np.full(len(arr) * self.ds_config["NUM_FRAMES"], label, dtype=np.int32)
                label = 0 if 'videos' in str(file_path) else 1
                truth = ground_truth_dict[name]["truth"]
                gt_idx = np.full(len(truth), i, dtype=np.int32)
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), i, dtype=np.int32)
                arrs.append(arr)
                ys.append(y)
                y_gt.append(truth)
                y_idx.append(idx)
                y_gt_idx.append(gt_idx)
                vid_ids.append(vid_id)

                # y = np.full(len(arr), label, dtype=np.int32)

                # gt_idx = np.full(len(truth), i, dtype=np.int32)
                #
                # arrs.append(arr)
                # ys.append(y)
                # y_gt.append(truth)
                # y_idx.append(idx)
                # y_gt_idx.append(gt_idx)
                # vid_ids.append(vid_id)

                vid_kind[i] = 'Normal' if label == 0 else 'Abnormal'
                vid_source_clips_num[i] = arr.shape[0]

            print(f"Loaded {len(arrs)} Shanghai test files from {self.processed_dir}")
            return {
                "X_test": fast_concatenate_axis0_variable(arrs),
                "y_test": fast_concatenate_axis0_variable(ys),
                "y_test_gt": fast_concatenate_axis0_variable(y_gt),
                "y_test_idx": fast_concatenate_axis0_variable(y_idx),
                "y_test_gt_idx": fast_concatenate_axis0_variable(y_gt_idx),
                "vid_test": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_test": vid_kind,
                "vid_source_clips_num_test": vid_source_clips_num,
            }

        # 主要逻辑
        splits = self.get_split()
        data = {}

        if "train" in splits:
            data.update(load_train_split(splits["train"]))

        if "test" in splits:
            data.update(load_test_split(splits["test"]))

        data['NUM_FRAMES'] = self.ds_config["NUM_FRAMES"]
        return data


class XDViolenceManager:
    def __init__(self, ds_config):
        self.ds_config = ds_config
        self.wd = Path(ds_config["working_dir"])
        self.data_dir = self.wd / ds_config[
            "DATA_DIR"] if 'pretrain_model' not in ds_config else get_pretrain_model_path(
            self.wd / ds_config["DATA_DIR"], ds_config["pretrain_model"])
        self.processed_dir = self.data_dir / ds_config["MODALITY_DIR"]
        self.segmentation_dir = self.data_dir / ds_config["SEGMENTATION_DIR"] / ds_config["MODALITY_DIR"]
        self.seed = ds_config.get("seed", 0)

    def get_split(self):
        """获取数据集划分信息"""
        split_dir = self.data_dir / self.ds_config["SPLIT_DIR"]
        split_file = self.ds_config["SPLIT_FILE"]
        if isinstance(split_file, list):
            split_file = split_file[self.seed % len(split_file)]

        splits = {}  # /data/coding/wsad/zsy/WSADBench/WSADBench/datasets/CV_by_I3D/XD-violence/splits/Anomaly_test.txt
        for split_tag, file_name in split_file.items():
            split_path = split_dir / file_name
            if split_path.exists():
                with open(split_path, "r") as f:
                    splits[split_tag] = [line.strip() for line in f.readlines()]
            else:
                # 如果分割文件不存在，根据文件夹结构自动生成
                splits[split_tag] = self._auto_generate_split(split_tag)

        return splits

    def _auto_generate_split(self, split_tag):
        """自动生成数据集划分（如果分割文件不存在）"""
        video_files = []
        raw_data_dir = self.data_dir.parent.parent / "source_datasets/XDViolence/raw_data"

        # XDViolence数据集通常有train和test文件夹
        if split_tag == "train":
            search_dirs = ["train"]
        else:  # test
            search_dirs = ["test"]

        for dir_name in search_dirs:
            dir_path = raw_data_dir / dir_name
            if dir_path.exists():
                # 遍历Normal和Violent子文件夹
                for category in ['Normal', 'Violent']:
                    category_path = dir_path / category
                    if category_path.exists():
                        for video_file in category_path.glob("*.mp4"):
                            # 保存相对于raw_data的路径
                            relative_path = video_file.relative_to(raw_data_dir)
                            video_files.append(str(relative_path.with_suffix('')))

        return video_files

    def load_ground_truth(self):
        """加载XDViolence数据集的帧级别标注"""
        ground_truth = {}

        annotation_file = self.data_dir / self.ds_config["LABEL_LIST"]
        if not annotation_file.exists():
            print(f"Annotation file {annotation_file} not found, generating default annotations")
            return self._generate_default_ground_truth()

        with open(annotation_file, "r") as f:
            ground_truth_list = [line.strip().split() for line in f.readlines()]

        groud_truth = {}
        for row in ground_truth_list:
            file_path = Path(row[0])
            file_name = file_path.name.replace('.mp4', '')
            # 从原始视频中计算帧数
            assert '.mp4' not in file_name and '.npy' not in file_name
            all_frames = int(row[1])
            if 'label_A' in row[0]:
                label_kind = 'A'
            else:
                match = re.search(r'label_([^-]+)', str(file_path))
                if match:
                    label_kind = match.group(1)
                else:  # 如果没有匹配到，可以设置默认值或者报错
                    raise ValueError(f"Could not extract label from file path: {file_path}")

            truth = np.zeros(all_frames, dtype=np.int32)

            # 解析异常时间段
            for i in range(2, len(row), 2):
                start_frame = int(row[i])
                end_frame = int(row[i + 1])
                truth[start_frame:end_frame + 1] = 1

            groud_truth[file_name] = {
                "file_path": file_path,
                "all_frames": all_frames,
                "label_kind": label_kind,
                "truth": truth,
            }

        return groud_truth

    def _generate_default_ground_truth(self):
        """生成默认的ground truth（如果没有详细标注文件）"""
        ground_truth = {}

        # 遍历处理后的视频文件
        for video_file in self.processed_dir.rglob("*.npy"):
            video_name = video_file.name

            # 根据路径判断是否为异常视频
            is_violent = 'label_A' not in str(video_file)

            try:
                # 加载视频特征以获取帧数
                features = np.load(video_file, mmap_mode='r')
                num_clips = features.shape[0]
                num_frames = num_clips * self.ds_config["NUM_FRAMES"]

                if is_violent:
                    # 对于暴力视频，假设整个视频都是异常
                    truth = np.ones(num_frames, dtype=np.int32)
                    label_kind = "Violent"
                else:
                    # 对于正常视频，整个视频都是正常
                    truth = np.zeros(num_frames, dtype=np.int32)
                    label_kind = "Normal"

                ground_truth[video_name] = {
                    "file_path": Path(video_name),
                    "all_frames": num_frames,
                    "label_kind": label_kind,
                    "truth": truth,
                }

            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

        return ground_truth

    def load_data(self, limit=None, num_segments=32):
        """加载XDViolence数据集数据"""
        if num_segments is not None:
            load_dir = self.segmentation_dir / str(num_segments)
            if not load_dir.exists():
                raise FileNotFoundError(
                    f"Segmented dir {load_dir} does not exist. You may need to run video_pre_segment.py first.")
        else:
            load_dir = self.processed_dir

        def fast_concatenate_axis0_variable(arrs):
            if not arrs:
                return np.array([])

            base_shape = arrs[0].shape[1:]
            for a in arrs:
                if a.shape[1:] != base_shape:
                    raise ValueError("All arrays must have the same shape except along axis=0.")

            total_rows = sum(a.shape[0] for a in arrs)
            result = np.empty((total_rows, *base_shape), dtype=arrs[0].dtype)

            idx = 0
            for a in arrs:
                n = a.shape[0]
                result[idx:idx + n] = a
                idx += n

            return result

        def load_train_split(files):
            arrs, ys, vid_ids = [], [], []
            vid_kind = {}
            vid_source_clips_num = {}

            for video_idx, file_name in enumerate(tqdm(files, desc="Loading XDViolence training data")):
                # 处理可能的路径分隔符
                file_parts = Path(file_name).parts

                # 构建完整的文件路径
                file_path = load_dir
                for part in file_parts:
                    file_path = file_path / part
                old_path = file_path
                file_path = Path(str(file_path).replace('.mp4', '')+'.mp4.npy')
                # file_path = file_path.with_suffix('.mp4.npy')

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                if limit is not None:
                    arr = arr[:limit]

                # XDViolence数据集标签：根据文件路径判断类别
                if 'label_A' in str(file_path):
                    label = 0  # 正常视频
                else:
                    label = 1  # 暴力视频

                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), video_idx, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                vid_ids.append(vid_id)
                if label == 0:
                    vid_kind[video_idx] = 'A'
                else:
                    match = re.search(r'label_([^-]+)', str(file_path))
                    if match:
                        vid_kind[video_idx] = match.group(1).replace('.mp4.npy','')
                    else:
                        # 如果没有匹配到，可以设置默认值或者报错
                        raise ValueError(f"Could not extract label from file path: {file_path}")

                # 获取原始clip数量
                source_file_path = self.processed_dir
                for part in file_parts:
                    source_file_path = source_file_path / part
                source_file_path = source_file_path.with_suffix('.mp4.npy')

                if source_file_path.exists():
                    source_arr = np.load(source_file_path, mmap_mode="r")
                    vid_source_clips_num[video_idx] = source_arr.shape[0]
                else:
                    vid_source_clips_num[video_idx] = len(arr)

            print(f"Loaded {len(arrs)} XDViolence training files from {load_dir}")
            return {
                "X_train": fast_concatenate_axis0_variable(arrs),
                "y_train": fast_concatenate_axis0_variable(ys),
                "vid_train": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_train": vid_kind,
                "vid_source_clips_num_train": vid_source_clips_num,
            }

        def load_test_split(files):
            arrs, ys, y_gt, y_idx, y_gt_idx, vid_ids = [], [], [], [], [], []
            vid_kind = {}
            vid_source_clips_num = {}
            ground_truth_dict = self.load_ground_truth()

            for i, file_name in enumerate(tqdm(files, desc="Loading XDViolence test data")):
                file_parts = Path(file_name).parts

                file_path = self.processed_dir
                for part in file_parts:
                    file_path = file_path / part
                file_path = file_path.with_suffix('.mp4.npy')

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                idx = np.full(len(arr), i, dtype=np.int32)
                name = file_path.name.replace('.mp4.npy', '')
                assert '.mp4' not in name and '.npy' not in name
                # 根据文件路径判断视频级别标签
                if 'label_A' in str(file_path):
                    label = 0
                else:
                    label = 1
                # 获取ground truth
                if name in ground_truth_dict:
                    truth = ground_truth_dict[name]["truth"]  # 只拿真值
                else:
                    # 如果没有详细的ground truth，根据视频级别标签生成
                    num_frames = len(arr) * self.ds_config["NUM_FRAMES"]
                    truth = np.full(num_frames, label, dtype=np.int32)

                gt_idx = np.full(len(truth), i, dtype=np.int32)
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), i, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                y_gt.append(truth)
                y_idx.append(idx)
                y_gt_idx.append(gt_idx)
                vid_ids.append(vid_id)
                if label == 0:
                    vid_kind[i] = 'A'
                else:
                    match = re.search(r'label_([^-]+)', str(file_path))
                    if match:
                        vid_kind[i] = match.group(1)
                    else:  # 如果没有匹配到，可以设置默认值或者报错
                        raise ValueError(f"Could not extract label from file path: {file_path}")
                vid_source_clips_num[i] = arr.shape[0]

            print(f"Loaded {len(arrs)} XDViolence test files from {self.processed_dir}")
            return {
                "X_test": fast_concatenate_axis0_variable(arrs),
                "y_test": fast_concatenate_axis0_variable(ys),
                "y_test_gt": fast_concatenate_axis0_variable(y_gt),
                "y_test_idx": fast_concatenate_axis0_variable(y_idx),
                "y_test_gt_idx": fast_concatenate_axis0_variable(y_gt_idx),
                "vid_test": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_test": vid_kind,
                "vid_source_clips_num_test": vid_source_clips_num,
            }

        # 主要逻辑
        splits = self.get_split()
        self.splits = splits
        data = {}

        if "train" in splits:
            data.update(load_train_split(splits["train"]))

        if "test" in splits:
            data.update(load_test_split(splits["test"]))

        data['NUM_FRAMES'] = self.ds_config["NUM_FRAMES"]
        return data


class TADManager:
    def __init__(self, ds_config):
        self.ds_config = ds_config
        self.wd = Path(ds_config["working_dir"])
        self.data_dir = self.wd / ds_config[
            "DATA_DIR"] if 'pretrain_model' not in ds_config else get_pretrain_model_path(
            self.wd / ds_config["DATA_DIR"], ds_config["pretrain_model"])
        self.processed_dir = self.data_dir / ds_config["MODALITY_DIR"]
        self.segmentation_dir = self.data_dir / ds_config["SEGMENTATION_DIR"] / ds_config["MODALITY_DIR"]
        self.seed = ds_config.get("seed", 0)

    def get_split(self):
        """获取数据集划分信息"""
        split_dir = self.data_dir / self.ds_config["SPLIT_DIR"]
        split_file = self.ds_config["SPLIT_FILE"]
        if isinstance(split_file, list):
            split_file = split_file[self.seed % len(split_file)]

        splits = {}
        for split_tag, file_name in split_file.items():
            split_path = split_dir / file_name
            if split_path.exists():
                with open(split_path, "r") as f:
                    splits[split_tag] = [line.strip() for line in f.readlines()]
            else:
                # 如果分割文件不存在，根据文件夹结构自动生成
                splits[split_tag] = self._auto_generate_split(split_tag)

        return splits

    def _auto_generate_split(self, split_tag):
        """自动生成数据集划分（如果分割文件不存在）"""
        video_files = []
        raw_data_dir = self.data_dir.parent.parent / "source_datasets/TAD/raw_data"

        # TAD数据集通常有train和test文件夹
        if split_tag == "train":
            search_dirs = ["train"]
        else:  # test
            search_dirs = ["test"]

        for dir_name in search_dirs:
            dir_path = raw_data_dir / dir_name
            if dir_path.exists():
                for video_file in dir_path.rglob("*.mp4"):
                    # 保存相对于raw_data的路径
                    relative_path = video_file.relative_to(raw_data_dir)
                    video_files.append(str(relative_path.with_suffix('')))

        return video_files

    def load_ground_truth(self):
        """加载TAD数据集的帧级别标注"""
        ground_truth = {}

        annotation_file = self.data_dir / self.ds_config["LABEL_LIST"]
        if not annotation_file.exists():
            print(f"Annotation file {annotation_file} not found, generating default annotations")
            return self._generate_default_ground_truth()

        with open(annotation_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            file_path = Path(parts[0])
            file_name = file_path.name
            all_frames = int(parts[1])

            # TAD的标注格式可能包含多个异常时间段
            truth = np.zeros(all_frames, dtype=np.int32)

            # 解析异常时间段（以帧为单位）
            i = 2
            while i < len(parts):
                if i + 1 < len(parts):
                    start_frame = int(parts[i])
                    end_frame = int(parts[i + 1])
                    truth[start_frame:end_frame + 1] = 1
                    i += 2
                else:
                    break

            ground_truth[file_name] = {
                "file_path": file_path,
                "all_frames": all_frames,
                "truth": truth,
            }

        return ground_truth

    def _generate_default_ground_truth(self):
        """生成默认的ground truth（如果没有详细标注文件）"""
        ground_truth = {}

        # 遍历处理后的视频文件
        for video_file in self.processed_dir.rglob("*.npy"):
            video_name = video_file.name

            try:
                # 加载视频特征以获取帧数
                features = np.load(video_file, mmap_mode='r')
                num_clips = features.shape[0]
                num_frames = num_clips * self.ds_config["NUM_FRAMES"]

                # 对于TAD数据集，假设所有视频都可能包含异常，根据具体需求调整
                truth = np.zeros(num_frames, dtype=np.int32)

                ground_truth[video_name] = {
                    "file_path": Path(video_name),
                    "all_frames": num_frames,
                    "truth": truth,
                }

            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

        return ground_truth

    def load_data(self, limit=None, num_segments=32):
        """加载TAD数据集数据"""
        if num_segments is not None:
            load_dir = self.segmentation_dir / str(num_segments)
            if not load_dir.exists():
                raise FileNotFoundError(
                    f"Segmented dir {load_dir} does not exist. You may need to run video_pre_segment.py first.")
        else:
            load_dir = self.processed_dir

        def fast_concatenate_axis0_variable(arrs):
            if not arrs:
                return np.array([])

            base_shape = arrs[0].shape[1:]
            for a in arrs:
                if a.shape[1:] != base_shape:
                    raise ValueError("All arrays must have the same shape except along axis=0.")

            total_rows = sum(a.shape[0] for a in arrs)
            result = np.empty((total_rows, *base_shape), dtype=arrs[0].dtype)

            idx = 0
            for a in arrs:
                n = a.shape[0]
                result[idx:idx + n] = a
                idx += n

            return result

        def load_train_split(files):
            arrs, ys, vid_ids = [], [], []
            vid_kind = {}
            vid_source_clips_num = {}

            for video_idx, file_name in enumerate(tqdm(files, desc="Loading TAD training data")):
                # 处理可能的路径分隔符
                file_parts = Path(file_name).parts

                # 构建完整的文件路径
                file_path = load_dir
                for part in file_parts:
                    file_path = file_path / part
                file_path = file_path.with_suffix('.mp4.npy')

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                if limit is not None:
                    arr = arr[:limit]

                # TAD数据集假设为异常检测任务，这里简化为0（可根据实际情况调整）
                label = 0 if 'Normal' in file_name else 1
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), video_idx, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                vid_ids.append(vid_id)
                if 'Normal' in file_name:
                    vid_kind[video_idx] = 'Normal'
                else:
                    vid_kind[video_idx] = file_name.split('/')[-1].split('_')[1]
                # vid_kind[video_idx] = 'Unknown'  # TAD数据集可能没有明确的正常/异常标签

                # 获取原始clip数量
                source_file_path = self.processed_dir
                for part in file_parts:
                    source_file_path = source_file_path / part
                source_file_path = source_file_path.with_suffix('.mp4.npy')

                if source_file_path.exists():
                    source_arr = np.load(source_file_path, mmap_mode="r")
                    vid_source_clips_num[video_idx] = source_arr.shape[0]
                else:
                    vid_source_clips_num[video_idx] = len(arr)

            print(f"Loaded {len(arrs)} TAD training files from {load_dir}")
            return {
                "X_train": fast_concatenate_axis0_variable(arrs),
                "y_train": fast_concatenate_axis0_variable(ys),
                "vid_train": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_train": vid_kind,
                "vid_source_clips_num_train": vid_source_clips_num,
            }

        def load_test_split(files):
            arrs, ys, y_gt, y_idx, y_gt_idx, vid_ids = [], [], [], [], [], []
            vid_kind = {}
            vid_source_clips_num = {}
            ground_truth_dict = self.load_ground_truth()

            for i, file_name in enumerate(tqdm(files, desc="Loading TAD test data")):
                file_parts = Path(file_name).parts

                file_path = self.processed_dir
                for part in file_parts:
                    file_path = file_path / part
                file_path = file_path.with_suffix('.mp4.npy')

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                idx = np.full(len(arr), i, dtype=np.int32)
                name = file_path.name.replace('.npy', '')

                # TAD数据集假设为异常检测任务，这里简化为0（可根据实际情况调整）
                label = 0 if 'Normal' in file_name else 1

                # 获取ground truth
                if name in ground_truth_dict:
                    truth = ground_truth_dict[name]["truth"]
                else:
                    raise ValueError(f"Ground truth not found for {name}")

                gt_idx = np.full(len(truth), i, dtype=np.int32)
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), i, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                y_gt.append(truth)
                y_idx.append(idx)
                y_gt_idx.append(gt_idx)
                vid_ids.append(vid_id)

                if 'Normal' in file_name:
                    vid_kind[i] = 'Normal'
                else:
                    vid_kind[i] = file_name.split('/')[-1].split('_')[1]
                vid_source_clips_num[i] = arr.shape[0]

            print(f"Loaded {len(arrs)} TAD test files from {self.processed_dir}")
            return {
                "X_test": fast_concatenate_axis0_variable(arrs),
                "y_test": fast_concatenate_axis0_variable(ys),
                "y_test_gt": fast_concatenate_axis0_variable(y_gt),
                "y_test_idx": fast_concatenate_axis0_variable(y_idx),
                "y_test_gt_idx": fast_concatenate_axis0_variable(y_gt_idx),
                "vid_test": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_test": vid_kind,
                "vid_source_clips_num_test": vid_source_clips_num,
            }

        # 主要逻辑
        splits = self.get_split()

        data = {}

        if "train" in splits:
            data.update(load_train_split(splits["train"]))

        if "test" in splits:
            data.update(load_test_split(splits["test"]))

        data['NUM_FRAMES'] = self.ds_config["NUM_FRAMES"]
        return data


class UCSD_Ped2Manager:
    def __init__(self, ds_config):
        self.ds_config = ds_config
        self.wd = Path(ds_config["working_dir"])
        self.data_dir = self.wd / ds_config["DATA_DIR"]
        self.processed_dir = self.data_dir / ds_config["MODALITY_DIR"]
        self.segmentation_dir = self.data_dir / ds_config["SEGMENTATION_DIR"] / ds_config["MODALITY_DIR"]
        self.seed = ds_config.get("seed", 0)

    def get_split(self):
        """获取数据集划分信息"""
        split_dir = self.data_dir / self.ds_config["SPLIT_DIR"]
        split_file = self.ds_config["SPLIT_FILE"]
        if isinstance(split_file, list):
            split_file = split_file[self.seed % len(split_file)]

        splits = {}
        for split_tag, file_name in split_file.items():
            split_path = split_dir / file_name
            if split_path.exists():
                with open(split_path, "r") as f:
                    splits[split_tag] = [line.strip() for line in f.readlines()]
            else:
                # 如果分割文件不存在，根据文件夹结构自动生成
                splits[split_tag] = self._auto_generate_split(split_tag)

        return splits

    def _auto_generate_split(self, split_tag):
        """自动生成数据集划分（如果分割文件不存在）"""
        video_files = []
        raw_data_dir = self.data_dir.parent.parent / "source_datasets/UCSD_Ped2/raw_data"

        # UCSD_Ped2数据集通常有Train和Test文件夹
        if split_tag == "train":
            search_dirs = ["Train"]
        else:  # test
            search_dirs = ["Test"]

        for dir_name in search_dirs:
            dir_path = raw_data_dir / dir_name
            if dir_path.exists():
                # 查找视频文件夹（每个文件夹包含tiff序列）
                for video_dir in dir_path.iterdir():
                    if video_dir.is_dir():
                        # 检查文件夹内是否有tiff文件
                        tiff_files = list(video_dir.glob("*.tif")) + list(video_dir.glob("*.tiff"))
                        if tiff_files:
                            relative_path = video_dir.relative_to(raw_data_dir)
                            video_files.append(str(relative_path))

        return video_files

    def load_ground_truth(self):
        """加载UCSD_Ped2数据集的帧级别标注"""
        ground_truth = {}

        annotation_file = self.data_dir / self.ds_config["LABEL_LIST"]
        if not annotation_file.exists():
            print(f"Annotation file {annotation_file} not found, generating default annotations")
            return self._generate_default_ground_truth()

        with open(annotation_file, "r") as f:
            lines = f.readlines()

        for line in lines:  # Test/Test008 180 01 0 179 -1 -1
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            video_name = parts[0]  # 视频文件夹名称
            all_frames = int(parts[1])

            # UCSD_Ped2的标注格式包含异常时间段
            truth = np.zeros(all_frames, dtype=np.int32)

            # 解析异常时间段（以帧为单位）
            i = 3
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] != '-1' and parts[i + 1] != '-1':
                    start_frame = int(parts[i])
                    end_frame = int(parts[i + 1])
                    truth[start_frame:end_frame + 1] = 1
                    i += 2
                else:
                    break

            ground_truth[video_name] = {
                "file_path": Path(video_name),
                "all_frames": all_frames,
                "truth": truth,
            }

        return ground_truth

    def _generate_default_ground_truth(self):
        """生成默认的ground truth（如果没有详细标注文件）"""
        ground_truth = {}

        # 遍历处理后的视频文件
        for video_file in self.processed_dir.rglob("*.npy"):
            video_name = video_file.stem  # 去掉.npy扩展名

            try:
                # 加载视频特征以获取帧数
                features = np.load(video_file, mmap_mode='r')
                num_clips = features.shape[0]
                num_frames = num_clips * self.ds_config["NUM_FRAMES"]

                # 对于UCSD_Ped2，Train文件夹中的视频为正常，Test中的视频可能包含异常
                # 这里简化处理，根据具体需求调整
                truth = np.zeros(num_frames, dtype=np.int32)

                ground_truth[video_name] = {
                    "file_path": Path(video_name),
                    "all_frames": num_frames,
                    "truth": truth,
                }

            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

        return ground_truth

    def load_data(self, limit=None, num_segments=None):
        """加载UCSD_Ped2数据集数据"""
        if num_segments is not None:
            load_dir = self.segmentation_dir / str(num_segments)
            if not load_dir.exists():
                raise FileNotFoundError(
                    f"Segmented dir {load_dir} does not exist. You may need to run video_pre_segment.py first.")
        else:
            load_dir = self.processed_dir

        def fast_concatenate_axis0_variable(arrs):
            if not arrs:
                return np.array([])

            base_shape = arrs[0].shape[1:]
            for a in arrs:
                if a.shape[1:] != base_shape:
                    raise ValueError("All arrays must have the same shape except along axis=0.")

            total_rows = sum(a.shape[0] for a in arrs)
            result = np.empty((total_rows, *base_shape), dtype=arrs[0].dtype)

            idx = 0
            for a in arrs:
                n = a.shape[0]
                result[idx:idx + n] = a
                idx += n

            return result

        def load_train_split(files):
            arrs, ys, vid_ids = [], [], []
            vid_kind = {}
            vid_source_clips_num = {}

            for video_idx, file_name in enumerate(tqdm(files, desc="Loading UCSD_Ped2 training data")):
                # 处理可能的路径分隔符
                file_parts = Path(file_name).parts

                # 构建完整的文件路径
                file_path = load_dir
                for part in file_parts:
                    file_path = file_path / part
                file_path = Path(f"{str(file_path)}/{str(file_path).split('/')[-1]}.mp4.npy")
                # file_path = file_path.with_suffix('.mp4.npy')

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                if limit is not None:
                    arr = arr[:limit]

                # UCSD_Ped2训练集全为正常视频
                label = 0
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), video_idx, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                vid_ids.append(vid_id)

                # 设置视频类别（训练集都是正常的）
                vid_kind[video_idx] = 'Unknown'

                # 获取原始clip数量
                source_file_path = self.processed_dir
                for part in file_parts:
                    source_file_path = source_file_path / part
                source_file_path = source_file_path.with_suffix('.mp4.npy')

                if source_file_path.exists():
                    source_arr = np.load(source_file_path, mmap_mode="r")
                    vid_source_clips_num[video_idx] = source_arr.shape[0]
                else:
                    vid_source_clips_num[video_idx] = len(arr)

            print(f"Loaded {len(arrs)} UCSD_Ped2 training files from {load_dir}")
            return {
                "X_train": fast_concatenate_axis0_variable(arrs),
                "y_train": fast_concatenate_axis0_variable(ys),
                "vid_train": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_train": vid_kind,
                "vid_source_clips_num_train": vid_source_clips_num,
            }

        def load_test_split(files):
            arrs, ys, y_gt, y_idx, y_gt_idx, vid_ids = [], [], [], [], [], []
            vid_kind = {}
            vid_source_clips_num = {}
            ground_truth_dict = self.load_ground_truth()

            for i, file_name in enumerate(tqdm(files, desc="Loading UCSD_Ped2 test data")):
                file_parts = Path(file_name).parts

                file_path = self.processed_dir
                for part in file_parts:
                    file_path = file_path / part
                file_path = Path(f"{str(file_path)}/{str(file_path).split('/')[-1]}.mp4.npy")

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                arr = np.load(file_path, mmap_mode="r")
                idx = np.full(len(arr), i, dtype=np.int32)

                # 使用文件路径构建视频名称
                video_name = '/'.join(file_parts)

                # UCSD_Ped2测试集的视频级别标签（可能包含异常）
                label = 0  # 视频级别标签，可根据需要调整

                # 获取ground truth
                if video_name in ground_truth_dict:
                    truth = ground_truth_dict[video_name]["truth"]
                else:
                    # 如果没有详细的ground truth，生成默认值
                    num_frames = len(arr) * self.ds_config["NUM_FRAMES"]
                    truth = np.zeros(num_frames, dtype=np.int32)
                    print(f"Warning: No ground truth found for {video_name}, using default (all normal)")

                gt_idx = np.full(len(truth), i, dtype=np.int32)
                y = np.full(len(arr), label, dtype=np.int32)
                vid_id = np.full(len(arr), i, dtype=np.int32)

                arrs.append(arr)
                ys.append(y)
                y_gt.append(truth)
                y_idx.append(idx)
                y_gt_idx.append(gt_idx)
                vid_ids.append(vid_id)

                # 设置视频类别
                vid_kind[i] = 'Unknown'
                vid_source_clips_num[i] = arr.shape[0]

            print(f"Loaded {len(arrs)} UCSD_Ped2 test files from {self.processed_dir}")
            return {
                "X_test": fast_concatenate_axis0_variable(arrs),
                "y_test": fast_concatenate_axis0_variable(ys),
                "y_test_gt": fast_concatenate_axis0_variable(y_gt),
                "y_test_idx": fast_concatenate_axis0_variable(y_idx),
                "y_test_gt_idx": fast_concatenate_axis0_variable(y_gt_idx),
                "vid_test": fast_concatenate_axis0_variable(vid_ids),
                "vid_kind_test": vid_kind,
                "vid_source_clips_num_test": vid_source_clips_num,
            }

        # 主要逻辑
        splits = self.get_split()
        self.splits = splits
        data = {}

        if "train" in splits:
            data.update(load_train_split(splits["train"]))

        if "test" in splits:
            data.update(load_test_split(splits["test"]))

        data['NUM_FRAMES'] = self.ds_config["NUM_FRAMES"]
        return data
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
