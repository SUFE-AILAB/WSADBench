import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

from filelock import FileLock


class EarlyStopManager:
    """
    Early stop
    """

    def __init__(self, mode="min", patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.mode = mode

        # 存储内置状态
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.early_stop = False
            return False

        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

    def is_best(self):
        return self.counter == 0  # 如果counter为0，说明当前是最好的

    def reset(self):
        self.best_score = None
        self.counter = 0
        self.early_stop = False


def classification_metric(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class PickleUtils:
    @staticmethod
    def save(obj, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def gen_current_tag():
        # 当前时间 - pid - 随机数
        import os
        import random
        import time

        return f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{os.getpid()}_{random.randint(0, 1000)}"


def visualize_model_grad_graph(loss, model, save_name="graph"):
    """
    可视化模型的计算图
    :param model: 模型
    :param save_name: 保存的文件名
    :return:
    """
    from torchviz import make_dot

    # 计算图
    dot = make_dot(loss, params=dict(model.named_parameters()))
    # 保存
    dot.render(save_name, format="png")
    return dot


def extended_serializer(obj):
    """支持常见非JSON原生类型的扩展序列化器"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()  # ISO 8601 格式
    elif hasattr(obj, "__dict__"):
        return obj.__dict__  # 处理自定义类
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def get_param_hash(params):
    """带类型处理的参数哈希生成"""
    param_str = json.dumps(
        params,
        sort_keys=True,
        default=extended_serializer,  # 使用自定义序列化
        ensure_ascii=False,  # 保留Unicode字符
        separators=(",", ":"),  # 移除多余空格
    )
    return hashlib.sha256(param_str.encode()).hexdigest()


def check_or_add_hash(hash_value, file_path="experiments.json", lock_path="experiments.lock"):
    file_path = Path(file_path)
    lock_path = Path(lock_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用文件锁避免并发写入
    with FileLock(lock_path):
        try:
            with open(file_path, "r") as f:
                hashes = set(json.load(f))
        except FileNotFoundError:
            hashes = set()

        if hash_value in hashes:
            return True

        hashes.add(hash_value)
        with open(file_path, "w") as f:
            json.dump(list(hashes), f)
        return False

def add_hashs(hash_values, file_path="experiments.json", lock_path="experiments.lock"):
    file_path = Path(file_path)
    lock_path = Path(lock_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用文件锁避免并发写入
    with FileLock(lock_path):
        try:
            with open(file_path, "r") as f:
                hashes = set(json.load(f))
        except FileNotFoundError:
            hashes = set()

        hashes = hashes | set(hash_values)

        with open(file_path, "w") as f:
            json.dump(list(hashes), f)