import time
import gc
import argparse
import numpy as np
import pandas as pd
import yaml
import json
import os
import math
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import logging
from typing import Dict, List, Optional, Any
import torch
import torch.multiprocessing as tmp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from WSADBench.datasets.data_generator import DataGenerator
from WSADBench.myutils import Utils, import_class
import resource
import inspect


class ModelRegistry:
    """模型注册器，用于管理和创建不同的模型"""

    @staticmethod
    def get_model_class(model_class_path: str):
        """根据模型类路径获取模型类"""
        return import_class(model_class_path)

    @staticmethod
    def get_default_model_class_path(model_name: str):
        """获取模型的默认类路径"""
        default_model_map = {
            "RoSAS": "WSADBench.baseline.RoSAS.run.RoSAS",
            "AABiGAN": "WSADBench.baseline.AABiGAN.run.AABiGAN",
            "CRGAN": "WSADBench.baseline.CRGAN.run.CRGAN",
            "DAGMM": "WSADBench.baseline.DAGMM.run.DAGMM",
            "DevNet": "WSADBench.baseline.DevNet.run.DevNet",
            "FEAWAD": "WSADBench.baseline.FEAWAD.run.FEAWAD",
            "FTTransformer": "WSADBench.baseline.FTTransformer.run.FTTransformer",
            "GANomaly": "WSADBench.baseline.GANomaly.run.GANomaly",
            "PReNet": "WSADBench.baseline.PReNet.run.PReNet",
            "REPEN": "WSADBench.baseline.REPEN.run.REPEN",
            "Sultani": "WSADBench.baseline.Sultani.run.Sultani",
            "MGFN": "WSADBench.baseline.MGFN.run.MGFN",
            "URDMU": "WSADBench.baseline.URDMU.run.URDMU",
            "RTFM": "WSADBench.baseline.RTFM.run.RTFM",
            "PyOD": "WSADBench.baseline.PyOD.PYOD",
            "Supervised": "WSADBench.baseline.Supervised.supervised",
            "IForest": "WSADBench.baseline.PyOD.PYOD",
            "ZhongGCNAD": "WSADBench.baseline.ZhongGCNAD.run.ZhongGCNAD",
            "VadClip": "WSADBench.baseline.VadClip.run.VadClip",
        }
        return default_model_map.get(model_name, None)


class ExperimentRunner:
    """通用实验运行器，支持video和tabular数据集"""

    def __init__(
        self,
        models: List[str],
        data_type: str,
        n_jobs=1,
        output_dir=None,
        parameter_config_path=None,
        datasets=None,
        rla_list=None,
        seed_list=None,
        gpu_list=None,
        DEBUG=False,
    ):
        """
        初始化运行器

        Args:
            models: 要运行的模型列表
            data_type: 数据类型，'video' 或 'tabular'
            n_jobs: 并行作业数量，-1表示使用所有CPU核心
            output_dir: 输出目录
            parameter_config_path: 参数配置文件路径
            datasets: 数据集列表
            rla_list: 标注异常比例列表
            seed_list: 随机种子列表
            gpu_list: 指定使用的GPU列表，如[0,1,2]或"0,1,2"，None表示自动检测
        """
        self.DEBUG = DEBUG
        if data_type not in ["video", "tabular"]:
            raise ValueError(f"data_type must be 'video' or 'tabular', got '{data_type}'")

        self.models = models
        self.data_type = data_type
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        # 初始化GPU管理器
        self.gpu_manager = GPUManager(gpu_list, self.n_jobs)

        # 设置默认输出目录和配置路径
        default_output_dir = f"results/{data_type}"
        default_config_path = f"WSADBench/model_configs/{data_type}"

        self.output_dir = Path(output_dir) if output_dir else Path(default_output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.parameter_config_path = Path(parameter_config_path) if parameter_config_path else Path(default_config_path)
        self.parameter_config_path.mkdir(exist_ok=True, parents=True)

        # 创建输出目录结构
        self.detail_dir = self.output_dir / "detail"
        self.summary_dir = self.output_dir / "summary"
        self.detail_dir.mkdir(exist_ok=True, parents=True)
        self.summary_dir.mkdir(exist_ok=True, parents=True)

        # 为每个模型创建子目录
        self.model_dirs = {}
        for model_name in self.models:
            model_dir = self.detail_dir / model_name
            model_dir.mkdir(exist_ok=True, parents=True)
            self.model_dirs[model_name] = model_dir

        # 工具类
        self.utils = Utils()

        # 数据生成器
        self.data_generator = DataGenerator(generate_duplicates=True, n_samples_threshold=1000)

        # 实验参数
        self.seed_list = seed_list if seed_list is not None else list(range(1, 11))
        self.rla_list = rla_list if rla_list is not None else [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

        # 获取数据集列表
        if datasets is None:
            self.datasets = self.get_datasets()
        else:
            self.datasets = datasets

        # 模型参数字典
        self.model_params = self._load_model_parameters()

        # 保存模型参数统计
        self._save_model_stats()

        logger.info(f"初始化模型: {self.models}")
        logger.info(f"数据类型: {self.data_type}")
        logger.info(f"并行作业数: {self.n_jobs}")
        logger.info(f"GPU配置: {self.gpu_manager.get_gpu_assignment_summary()}")
        logger.info(f"输出目录: {self.output_dir.absolute()}")
        logger.info(f"参数配置路径: {self.parameter_config_path.absolute()}")
        logger.info(f"数据集数量: {len(self.datasets)}")
        logger.info(f"RLA设置: {self.rla_list}")
        logger.info(f"Seeds: {self.seed_list}")

    def _load_model_parameters(self) -> Dict[str, Dict[str, Any]]:
        """加载模型参数配置"""
        model_params = {}

        for model_name in self.models:
            # 默认的模型类路径
            default_model_class_path = ModelRegistry.get_default_model_class_path(model_name)

            # 初始化配置
            config = {"model_class": default_model_class_path, "parameters": {}}

            # 尝试加载YAML配置文件
            config_file = self.parameter_config_path / f"{model_name}.yaml"
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        yaml_config = yaml.safe_load(f)
                        if yaml_config:
                            # 如果配置文件中指定了model_class，则使用它
                            if "model_class" in yaml_config:
                                config["model_class"] = yaml_config["model_class"]

                            # 如果配置文件中有parameters，则更新参数
                            if "parameters" in yaml_config:
                                config["parameters"].update(yaml_config["parameters"])
                            else:
                                # 向后兼容：将其他所有字段视为参数
                                yaml_params = {k: v for k, v in yaml_config.items() if k != "model_class"}
                                if yaml_params:
                                    config["parameters"].update(yaml_params)

                            logger.info(f"从 {config_file} 加载了 {model_name} 的配置")
                            logger.info(f"模型类: {config['model_class']}")
                except Exception as e:
                    logger.warning(f"加载 {config_file} 失败: {e}")
            else:
                logger.info(f"未找到 {model_name} 的配置文件 {config_file}，使用默认配置")
                logger.info(f"默认模型类: {config['model_class']}")

            if config["model_class"] is None:
                raise ValueError(f"Unknown model: {model_name} Please check the model name or configuration file.")

            model_params[model_name] = config

        return model_params

    def _save_model_stats(self):  # TODO: 将这部分统计修改为拟合完毕之后再保存
        """保存模型参数统计信息"""
        for model_name in self.models:
            model_config = self.model_params.get(model_name, {})

            # 准备统计信息
            stats = {
                "model_name": model_name,
                "model_class": model_config.get("model_class", "Unknown"),
                "parameters": model_config.get("parameters", {}),
                "config_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_type": self.data_type,
            }

            # 尝试计算参数大小
            try:
                temp_model = self.create_model(self.model_params, model_name, seed=1)

                if hasattr(temp_model, "parameter_count"):
                    try:
                        param_stats = temp_model.parameter_count()
                        if isinstance(param_stats, dict) and "total" in param_stats:
                            stats["parameter_count"] = param_stats["total"]
                            stats["parameter_stats"] = param_stats
                        else:
                            logger.warning(f"{model_name} 的 parameter_count 方法返回格式不符合预期: {param_stats}")
                            stats["parameter_count"] = "Unknown"
                    except Exception as e:
                        logger.warning(f"调用 {model_name} 的 parameter_count 方法失败: {e}")
                        stats["parameter_count"] = "Unknown"
                elif hasattr(temp_model, "get_params"):
                    # sklearn风格的模型
                    model_params = temp_model.get_params()
                    stats["sklearn_params"] = model_params
                    stats["parameter_count"] = len(model_params)
                elif hasattr(temp_model, "state_dict"):
                    # PyTorch模型
                    state_dict = temp_model.state_dict()
                    total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, "numel"))
                    stats["parameter_count"] = int(total_params)
                else:
                    stats["parameter_count"] = "Unknown"

                del temp_model
                gc.collect()

            except Exception as e:
                logger.warning(f"无法计算 {model_name} 的参数大小: {e}")
                stats["parameter_count"] = "Unknown"
                stats["error"] = str(e)

            # 保存统计信息
            stats_file = self.model_dirs[model_name] / "model_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"保存 {model_name} 模型统计信息到: {stats_file}")

    def get_datasets(self):
        """获取数据集列表, 这里用于没有指定数据集时获取最合适的默认测试数据集"""
        if self.data_type == "video":
            datasets = self.data_generator.generate_dataset_list()["video"]
            logger.info(f"找到 {len(datasets)} 个Video数据集")
        else:  # tabular
            datasets = self.data_generator.generate_dataset_list()["classical"]
            logger.info(f"找到 {len(datasets)} 个Classical数据集")
        return datasets

    @staticmethod
    def create_model(model_params, model_name: str, seed: int, feature_shape: tuple = None, **kwargs):
        """创建模型实例"""
        # 获取模型配置
        model_config = model_params.get(model_name, {})

        # 获取模型类路径
        model_class_path = model_config.get("model_class")
        if not model_class_path:
            raise ValueError(f"No model class path found for {model_name}")

        # 获取模型类
        model_class = ModelRegistry.get_model_class(model_class_path)

        # 获取模型参数
        model_params = model_config.get("parameters", {}).copy()
        model_params.update(kwargs)

        # 如果模型的 __init__ 方法有 input_dim 参数，且 feature_shape 不为 None，则更新 input_dim
        init_signature = inspect.signature(model_class.__init__)
        if "input_dim" in init_signature.parameters and feature_shape is not None:
            model_params["input_dim"] = feature_shape[-1]

        # 创建模型
        return model_class(seed=seed, **model_params)

    @staticmethod
    def _process_video_data(data):
        """处理video数据的特殊逻辑"""
        # 训练数据reshape
        _clips_num, _crops_num, _dim = data["X_train"].shape
        data["X_train"] = data["X_train"].reshape(_clips_num * _crops_num, _dim)
        data["y_train"] = data["y_train"].repeat(_crops_num)

        # 保留视频ID信息并扩展到crops
        if "vid_train" in data:
            data["vid_train"] = data["vid_train"].repeat(_crops_num)

        # 测试数据reshape
        _clips_num, _crops_num, _dim = data["X_test"].shape
        data["X_test"] = data["X_test"].reshape(_clips_num * _crops_num, _dim)

        # 保留视频ID信息并扩展到crops
        if "vid_test" in data:
            data["vid_test"] = data["vid_test"].repeat(_crops_num)

        return data, (_clips_num, _crops_num)

    @staticmethod
    def _process_video_scores(scores, video_shape, data):
        """处理video分数的特殊逻辑：从clip级别还原到帧级别"""
        _clips_num, _crops_num = video_shape

        # 平均每个crop, 获得每个clip的分数
        scores = scores.reshape(_clips_num, _crops_num)
        scores = np.mean(scores, axis=1)

        # 还原clip级别score为帧级别score
        y_test_idx = data["y_test_idx"]
        y_test_gt, y_test_gt_idx = data["y_test_gt"], data["y_test_gt_idx"]
        num_clip_frames = data["NUM_FRAMES"]

        frame_scores = []
        frame_truth = []
        for i in range(max(y_test_gt_idx) + 1):
            select_gt = y_test_gt[y_test_gt_idx == i]
            select_scores = scores[y_test_idx == i]
            select_scores = select_scores.repeat(num_clip_frames)
            common_length = min(len(select_gt), len(select_scores))

            frame_scores.append(select_scores[:common_length])
            frame_truth.append(select_gt[:common_length])

        frame_scores = np.concatenate(frame_scores, axis=0)
        frame_truth = np.concatenate(frame_truth, axis=0)

        return frame_scores, frame_truth

    def _save_single_result(self, result):
        """保存单个实验结果"""
        if result is None:
            return

        model_name = result["model"]

        # 生成结果文件路径
        result_file = self.model_dirs[model_name] / f"{model_name}_results.csv"

        # 转换为DataFrame
        df = pd.DataFrame([result])

        # 如果文件已存在，追加写入；否则创建新文件
        if result_file.exists():
            df.to_csv(result_file, mode="a", header=False, index=False)
        else:
            df.to_csv(result_file, index=False)

        logger.debug(f"保存 {model_name} 实验结果: {result['dataset']}, seed={result['seed']}, rla={result['rla']}")

    def run_experiments(self):
        """运行所有实验"""
        # 生成实验参数组合
        experiment_params = []
        for model_name in self.models:
            for dataset in self.datasets:
                for rla in self.rla_list:
                    for seed in self.seed_list:
                        experiment_params.append((model_name, dataset, rla, seed))

        logger.info(f"总共 {len(experiment_params)} 个实验")
        logger.info(f"模型: {self.models}")
        logger.info(f"数据集数量: {len(self.datasets)}")
        logger.info(f"RLA设置: {self.rla_list}")
        logger.info(f"Seeds: {self.seed_list}")

        # 准备传递给子进程的实验配置（不包含完整的runner）
        experiment_config = {
            "data_type": self.data_type,
            "model_params": self.model_params,  # 传递模型参数配置
        }

        # 为每个任务分配GPU
        experiment_params_with_gpu = []
        for i, params in enumerate(experiment_params):
            gpu_id = self.gpu_manager.get_gpu_for_task(i)
            experiment_params_with_gpu.append((params, gpu_id, experiment_config, self.DEBUG))

        # 运行实验
        results = []
        if self.n_jobs == 1:
            # 串行运行
            for params in tqdm(experiment_params_with_gpu, desc="运行实验"):
                # 即使串行也设置GPU
                result = run_single_experiment_with_gpu(params)
                if result is not None:
                    results.append(result)
                    self._save_single_result(result)
        else:
            # 并行运行 - 使用torch.multiprocessing
            # 设置multiprocessing启动方法
            if tmp.get_start_method(allow_none=True) != "spawn":
                tmp.set_start_method("spawn", force=True)

            # 使用torch.multiprocessing.Pool执行
            with tmp.Pool(self.n_jobs) as pool:
                for result in tqdm(
                    pool.imap(run_single_experiment_with_gpu, experiment_params_with_gpu),
                    total=len(experiment_params_with_gpu),
                    desc="运行实验",
                ):
                    if result is not None:
                        results.append(result)
                        self._save_single_result(result)

        logger.info(f"完成 {len(results)} 个实验")

        # 生成汇总报告
        self.generate_summary()

        return results

    def generate_summary(self):
        """生成汇总报告"""
        logger.info("开始生成汇总报告...")
        generate_summary_only(self.output_dir)
        logger.info("汇总报告生成完成。")


def generate_summary_statistics(df, model_stats, summary_file):
    """通用的汇总统计函数"""
    # 汇总的效果
    dataset_summary = {}
    for metric in ["aucroc", "aucpr"]:
        dataset_summary[f"{metric}_mean"] = df.groupby(["dataset", "model", "rla"])[metric].mean()
        dataset_summary[f"{metric}_std"] = df.groupby(["dataset", "model", "rla"])[metric].std()

    # 按模型汇总的效果
    model_summary = {}
    for metric in ["aucroc", "aucpr"]:
        model_summary[f"{metric}_mean"] = df.groupby("model")[metric].mean()
        model_summary[f"{metric}_std"] = df.groupby("model")[metric].std()

    # 汇总的时间
    dataset_time_summary = {}
    for metric in ["fit_time", "inference_time"]:
        dataset_time_summary[f"{metric}_mean"] = df.groupby(["dataset", "model", "rla"])[metric].mean()
        dataset_time_summary[f"{metric}_std"] = df.groupby(["dataset", "model", "rla"])[metric].std()

    # 按模型汇总的时间
    model_time_summary = {}
    for metric in ["fit_time", "inference_time"]:
        model_time_summary[f"{metric}_mean"] = df.groupby("model")[metric].mean()
        model_time_summary[f"{metric}_std"] = df.groupby("model")[metric].std()

    # 保存汇总结果到Excel
    with pd.ExcelWriter(summary_file) as writer:
        # 原始详细结果
        df.to_excel(writer, sheet_name="详细结果", index=False)

        # 按数据集汇总的效果
        for metric_name, metric_data in dataset_summary.items():
            metric_data.to_excel(writer, sheet_name=f"效果_{metric_name}")

        # 按模型汇总的效果
        for metric_name, metric_data in model_summary.items():
            metric_data.to_excel(writer, sheet_name=f"模型_{metric_name}")

        # 按数据集汇总的时间
        for metric_name, metric_data in dataset_time_summary.items():
            metric_data.to_excel(writer, sheet_name=f"时间_{metric_name}")

        # 按模型汇总的时间
        for metric_name, metric_data in model_time_summary.items():
            metric_data.to_excel(writer, sheet_name=f"模型时间_{metric_name}")

        # 模型参数统计
        if model_stats:
            stats_df = pd.DataFrame.from_dict(model_stats, orient="index")
            stats_df.to_excel(writer, sheet_name="模型参数统计")

    logger.info(f"汇总结果已保存到: {summary_file}")


def print_summary_statistics(df):
    """通用的打印汇总统计函数"""
    logger.info("\n" + "=" * 50)
    logger.info("实验结果汇总统计")
    logger.info("=" * 50)

    # 总体统计
    valid_results = df.dropna(subset=["aucroc", "aucpr"])
    logger.info(f"有效实验数量: {len(valid_results)}/{len(df)}")

    if len(valid_results) > 0:
        # 按模型统计
        logger.info("\n按模型统计:")
        for model in sorted(valid_results["model"].unique()):
            model_data = valid_results[valid_results["model"] == model]
            logger.info(
                f"{model}: AUCROC={model_data['aucroc'].mean():.4f} ± {model_data['aucroc'].std():.4f}, "
                f"AUCPR={model_data['aucpr'].mean():.4f} ± {model_data['aucpr'].std():.4f}"
            )


def generate_summary_only(output_dir):
    """独立的汇总函数，仅用于汇总现有结果"""
    output_dir = Path(output_dir)
    detail_dir = output_dir / "detail"
    summary_dir = output_dir / "summary"

    if not detail_dir.exists():
        logger.error(f"详细结果目录不存在: {detail_dir}")
        return

    summary_dir.mkdir(exist_ok=True, parents=True)

    # 收集所有模型目录和结果文件
    all_results = []
    model_stats = {}

    for model_dir in detail_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name

            # 读取模型统计信息
            stats_file = model_dir / "model_stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    model_stats[model_name] = json.load(f)

            # 读取结果文件
            result_file = model_dir / f"{model_name}_results.csv"
            if result_file.exists():
                try:
                    df = pd.read_csv(result_file)
                    all_results.append(df)
                    logger.info(f"读取结果文件: {result_file} ({len(df)} 条记录)")
                except Exception as e:
                    logger.warning(f"读取结果文件失败 {result_file}: {e}")
            else:
                # 备用：查找带时间戳的旧格式文件
                old_result_files = list(model_dir.glob(f"{model_name}_results_*.csv"))
                for old_result_file in old_result_files:
                    try:
                        df = pd.read_csv(old_result_file)
                        all_results.append(df)
                        logger.info(f"读取旧格式结果文件: {old_result_file} ({len(df)} 条记录)")
                    except Exception as e:
                        logger.warning(f"读取结果文件失败 {old_result_file}: {e}")

    if not all_results:
        logger.warning("没有找到任何结果文件")
        return

    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"合并了 {len(all_results)} 个结果文件，总共 {len(combined_df)} 个实验结果")

    # 生成汇总统计
    summary_file = summary_dir / "summary.xlsx"

    # 使用通用汇总函数
    generate_summary_statistics(combined_df, model_stats, summary_file)

    # 打印简要统计
    print_summary_statistics(combined_df)


class GPUManager:
    """GPU资源管理器"""

    def __init__(self, gpu_list=None, n_jobs=1):
        """
        初始化GPU管理器

        Args:
            gpu_list: 指定使用的GPU列表，如[0,1,2]，None表示自动检测
            n_jobs: 总并发任务数
        """
        self.available_gpus = self._detect_gpus(gpu_list)
        self.n_jobs = n_jobs
        self.num_gpus = len(self.available_gpus)

        if self.num_gpus == 0:
            logger.warning("未检测到可用GPU，将使用CPU模式")
        else:
            logger.info(f"检测到 {self.num_gpus} 个可用GPU: {self.available_gpus}")
            logger.info(f"并发任务数: {n_jobs}, 每个GPU最多同时运行: {math.ceil(n_jobs / self.num_gpus)} 个任务")

    def _detect_gpus(self, gpu_list):
        """检测可用GPU"""
        if gpu_list is not None:
            # 用户指定GPU列表
            if isinstance(gpu_list, str):
                # 支持 "0,1,2" 格式
                return [int(x.strip()) for x in gpu_list.split(",")]
            elif isinstance(gpu_list, list):
                return gpu_list
            else:
                return [gpu_list]
        else:
            # 自动检测所有可用GPU
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
            else:
                return []

    def get_gpu_for_task(self, task_index):
        """获取任务应该使用的GPU ID"""
        if self.num_gpus == 0:
            return None
        return self.available_gpus[task_index % self.num_gpus]

    def get_gpu_assignment_summary(self):
        """获取GPU分配摘要"""
        if self.num_gpus == 0:
            return "CPU模式"

        tasks_per_gpu = {}
        for i in range(self.n_jobs):
            gpu_id = self.get_gpu_for_task(i)
            tasks_per_gpu[gpu_id] = tasks_per_gpu.get(gpu_id, 0) + 1

        return f"GPU分配: {dict(sorted(tasks_per_gpu.items()))}"


def run_single_experiment_with_gpu(params_with_config):
    """
    带GPU分配的实验执行函数
    """
    params, gpu_id, experiment_config, DEBUG = params_with_config
    model_name, dataset_name, rla, seed = params

    # 设置GPU环境
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"任务 {model_name}-{dataset_name}(seed={seed}, rla={rla}) 分配到 GPU {gpu_id}")
    else:
        # CPU模式
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info(f"任务 {model_name}-{dataset_name}(seed={seed}, rla={rla}) 使用 CPU 模式")

    try:
        # 从配置中获取所需的组件
        data_type = experiment_config["data_type"]
        model_params = experiment_config["model_params"]
        utils = Utils()

        # 创建数据生成器
        data_generator = DataGenerator(generate_duplicates=True, n_samples_threshold=1000)
        data_generator.seed = seed
        data_generator.dataset = dataset_name

        # 生成数据
        data = data_generator.generator(
            la=rla,
            at_least_one_labeled=True,
            la_shortage_mode="ignore",
        )

        # 检查数据有效性
        if len(data["y_train"]) == 0 or np.sum(data["y_train"]) == 0:
            logger.warning(f"数据集 {dataset_name} (model={model_name}, seed={seed}, rla={rla}) 没有标注异常，跳过")
            return None

        # 创建模型
        feature_shape = data["X_train"].shape
        model = ExperimentRunner.create_model(model_params, model_name, seed=seed, feature_shape=feature_shape)

        # 根据数据类型处理数据
        data_shape = None
        if data_type == "video":
            data, data_shape = ExperimentRunner._process_video_data(data)

        # 训练时间
        start_time = time.time()
        def has_param(func, param_name):
            """检查函数是否有指定参数"""
            return param_name in inspect.signature(func).parameters
        
        
        train_input = {}
        if has_param(model.fit, "X"):
            train_input["X"] = data["X_train"]
        if has_param(model.fit, "y"):
            train_input["y"] = data["y_train"]
        if has_param(model.fit, "X_train"):
            train_input["X_train"] = data["X_train"]
        if has_param(model.fit, "y_train"):
            train_input["y_train"] = data["y_train"]
        if has_param(model.fit, "vid_info"):
            train_input["vid_info"] = data.get("vid_train", None)
        if has_param(model.fit, "crops_num"):
            train_input["crops_num"] = data_shape[1] if data_shape else None
        if has_param(model.fit, "vid_kind"):
            train_input["vid_kind"] = data.get("vid_kind_train", None)
        if has_param(model.fit, "vid_source_clips_num"):
            train_input["vid_source_clips_num"] = data.get("vid_source_clips_num_train", None)
        
        pred_func = None
        if hasattr(model, "predict_score"):
            pred_func = model.predict_score
        elif hasattr(model, "decision_function"):
            pred_func = model.decision_function
        elif hasattr(model, "predict_proba"):
            pred_func = model.predict_proba
        else:
            raise AttributeError(f"模型 {model_name} 没有可用的评分方法")


        test_input = {}
        if has_param(pred_func, "X"):
            test_input["X"] = data["X_test"]
        if has_param(pred_func, "X_test"):
            test_input["X_test"] = data["X_test"]
        if has_param(pred_func, "vid_info"):
            test_input["vid_info"] = data.get("vid_test", None)
        if has_param(pred_func, "crops_num"):
            test_input["crops_num"] = data_shape[1] if data_shape else None
        if has_param(pred_func, "vid_kind"):
            test_input["vid_kind"] = data.get("vid_kind_test", None)
        if has_param(pred_func, "vid_source_clips_num"):
            test_input["vid_source_clips_num"] = data.get("vid_source_clips_num_test", None)
        
        model.fit(**train_input)
        
        fit_time = time.time() - start_time

        # 推理时间
        start_time = time.time()
        proba = pred_func(**test_input)
        if proba.ndim == 1:
            scores = proba
        else:
            scores = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()

        inference_time = time.time() - start_time

        # 根据数据类型处理分数和计算指标
        if data_type == "video":
            frame_scores, frame_truth = ExperimentRunner._process_video_scores(scores, data_shape, data)
            metrics = utils.metric(y_true=frame_truth, y_score=frame_scores, pos_label=1)
        else:  # tabular
            metrics = utils.metric(y_true=data["y_test"], y_score=scores, pos_label=1)

        result = {
            "model": model_name,
            "dataset": dataset_name,
            "rla": rla,
            "seed": seed,
            "aucroc": metrics["aucroc"],
            "aucpr": metrics["aucpr"],
            "fit_time": fit_time,
            "inference_time": inference_time,
            "n_train": len(data["y_train"]),
            "n_test": len(data["y_test"]),
            "n_train_anomalies": np.sum(data["y_train"]),
            "n_test_anomalies": np.sum(data["y_test"]),
            "error": "",
            "data_type": data_type,
        }

        logger.info(
            f"完成 {model_name} - {dataset_name} (seed={seed}, rla={rla}): "
            f"AUCROC={metrics['aucroc']:.4f}, AUCPR={metrics['aucpr']:.4f}"
        )

        # 清理内存
        del model, data, scores
        gc.collect()

        return result

    except Exception as e:
        if DEBUG:
            raise e
        logger.error(f"实验失败 {model_name} - {dataset_name} (seed={seed}, rla={rla}): {str(e)}")
        return {
            "model": model_name,
            "dataset": dataset_name,
            "rla": rla,
            "seed": seed,
            "aucroc": np.nan,
            "aucpr": np.nan,
            "fit_time": np.nan,
            "inference_time": np.nan,
            "n_train": np.nan,
            "n_test": np.nan,
            "n_train_anomalies": np.nan,
            "n_test_anomalies": np.nan,
            "error": str(e).replace("\n", " ").replace(",", " "),
            "data_type": data_type,
        }


def main():
    """解开线程限制"""
    # 解开限制
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"原始限制: soft={soft}, hard={hard}")
    # 设置为 2048（注意不能超过 hard limit，否则会报错）
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, hard))
    # 验证修改结果
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"修改后: soft={soft}, hard={hard}")

    """主函数"""
    parser = argparse.ArgumentParser(description="统一的异常检测实验运行器")

    # 必需参数
    parser.add_argument("--data_type", choices=["video", "tabular"], required=True, help="数据类型：video 或 tabular")

    parser.add_argument("--models", nargs="+", help="要运行的模型名称列表")

    # 可选参数
    parser.add_argument("--n_jobs", type=int, default=1, help="并行作业数量，-1表示使用所有CPU核心 (默认: 1)")

    parser.add_argument("--output_dir", type=str, help="输出目录 (默认: results/{data_type})")

    parser.add_argument(
        "--parameter_config_path",
        type=str,
        help="模型参数配置文件目录 (默认: WSADBench/model_configs/{data_type})",
    )

    parser.add_argument("--datasets", nargs="+", default=None, help="指定运行的数据集名称，默认运行所有数据集")

    parser.add_argument(
        "--rla_list",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
        help="标注异常比例列表 (默认: 0.01 0.05 0.1 0.25 0.5 0.75 1.0)",
    )

    parser.add_argument(
        "--seed_list",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="随机种子列表 (默认: 1 2 3 4 5 6 7 8 9 10)",
    )

    parser.add_argument(
        "--dry_summary",
        action="store_true",
        help="仅进行汇总，不运行实验",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="指定使用的GPU，格式：0,1,2 或 auto（自动检测所有GPU），默认：auto",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="开启调试模式，捕获所有异常并打印详细错误信息",
    )

    args = parser.parse_args()

    # 如果只是要汇总
    if args.dry_summary:
        logger.info("仅进行汇总操作...")
        output_dir = args.output_dir if args.output_dir else f"results/{args.data_type}"
        generate_summary_only(output_dir)
        return

    # 如果不是dry_summary模式，则检查必需的参数
    if not args.models:
        parser.error("--models is required when not using --dry_summary. Please specify at least one model.")

    # 预处理RLA列表
    _rla_list = []
    for rla in args.rla_list:
        if rla > 1:
            _rla_list.append(int(rla))
        else:
            _rla_list.append(rla)
    args.rla_list = _rla_list

    # 处理GPU参数
    gpu_list = None
    if args.gpus is not None:
        if args.gpus.lower() == "auto":
            gpu_list = None  # 自动检测
        else:
            gpu_list = args.gpus  # 用户指定

    # 创建运行器
    runner = ExperimentRunner(
        models=args.models,
        data_type=args.data_type,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        parameter_config_path=args.parameter_config_path,
        datasets=args.datasets,
        rla_list=args.rla_list,
        seed_list=args.seed_list,
        gpu_list=gpu_list,
        DEBUG=args.DEBUG,
    )

    # 运行实验
    logger.info(f"开始运行{args.data_type}实验，模型: {args.models}")
    start_time = time.time()

    results = runner.run_experiments()

    total_time = time.time() - start_time
    logger.info(f"所有实验完成，总耗时: {total_time:.2f}秒")


if __name__ == "__main__":
    main()


"""
使用示例:

# 运行video实验，自动检测GPU
python run_experiment.py --data_type video --models RoSAS --n_jobs 4 --gpus auto

# 运行tabular实验，指定使用GPU 0,1
python run_experiment.py --data_type tabular --models RoSAS AABiGAN --n_jobs 8 --gpus 0,1

# 使用所有GPU，高并发任务
python run_experiment.py --data_type video --models Sultani --n_jobs 16 --gpus auto

# 仅使用特定GPU
python run_experiment.py --data_type video --models RoSAS --n_jobs 4 --gpus 0,2

# CPU模式（不使用GPU）
python run_experiment.py --data_type tabular --models RoSAS --n_jobs 4

# 指定特定数据集和RLA
python run_experiment.py --data_type video --models RoSAS --datasets cardio thyroid --rla_list 0.1 0.5 1.0 --gpus auto

# 指定随机种子
python run_experiment.py --data_type tabular --models RoSAS --seed_list 1 2 3 4 5 --gpus 0,1

# 使用自定义配置目录
python run_experiment.py --data_type video --models RoSAS --parameter_config_path ./my_configs --gpus auto

# 并行运行，使用所有CPU核心和GPU
python run_experiment.py --data_type tabular --models RoSAS AABiGAN --n_jobs -1 --gpus auto

# 仅进行汇总（从已有的detail目录生成summary）
python run_experiment.py --data_type video --dry_summary

# GPU超分：8个任务使用2个GPU（每个GPU运行4个任务）
python run_experiment.py --data_type video --models Sultani --n_jobs 8 --gpus 0,1

# 快速测试
python run_experiment.py --data_type video --models AABiGAN --datasets 10_cover --seed_list 1 --rla_list 0.1 --n_jobs 1 --gpus 0
"""
