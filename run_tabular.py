#!/usr/bin/env python3
"""
通用模型在Classical数据集上的并行运行脚本

支持:
- 通过命令行指定模型名称，支持运行多个模型
- 从YAML配置文件加载模型特定参数
- 在Classical数据集上运行10个seeds
- 将AUCROC、AUCPR、运行时间等结果写入Excel
- 支持并行运行多个设置
"""

import os
import sys
import time
import gc
import argparse
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from typing import Dict, List, Optional, Any
import threading
import glob

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from WSADBench.datasets.data_generator import DataGenerator
from WSADBench.myutils import Utils, import_class 


class ModelRegistry:
    """模型注册器，用于管理和创建不同的模型"""

    @staticmethod
    def get_model_class(model_class_path: str):
        """根据模型类路径获取模型类"""
        # 动态导入模型类
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
            "PyOD": "WSADBench.baseline.PyOD.PYOD",
            "Supervised": "WSADBench.baseline.Supervised.supervised",
            "IForest": "WSADBench.baseline.PyOD.PYOD",  # 默认PyOD的一个变体
        }

        return default_model_map.get(model_name, None)


class GeneralClassicalRunner:
    """通用模型在Classical数据集上的运行器"""

    def __init__(
        self,
        models: List[str],
        n_jobs=1,
        output_dir=None,
        parameter_config_path=None,
        datasets=None,
        rla_list=None,
        seed_list=None,
    ):
        """
        初始化运行器

        Args:
            models: 要运行的模型列表
            n_jobs: 并行作业数量，-1表示使用所有CPU核心
            output_dir: 输出目录，默认为当前目录下的results文件夹
            parameter_config_path: 参数配置文件路径
            datasets: 数据集列表，默认使用所有Classical数据集
            rla_list: 标注异常比例列表，默认使用预设值
            seed_list: 随机种子列表，默认1-10
        """
        self.models = models
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.parameter_config_path = Path(parameter_config_path) if parameter_config_path else Path("model_configs")
        self.parameter_config_path.mkdir(exist_ok=True)

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
        self.seed_list = seed_list if seed_list is not None else list(range(1, 11))  # 默认10个seeds
        self.rla_list = rla_list if rla_list is not None else [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]  # 标注异常比例

        # 获取数据集列表
        if datasets is None:
            self.datasets = self.get_classical_datasets()
        else:
            self.datasets = datasets

        # 模型参数字典
        self.model_params = self._load_model_parameters()

        # 保存模型参数统计
        self._save_model_stats()

        logger.info(f"初始化通用Classical运行器，模型: {self.models}")
        logger.info(f"并行作业数: {self.n_jobs}")
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
            if default_model_class_path is None:
                raise ValueError(f"Unknown model: {model_name}. No default class path found.")

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
                                # 如果没有parameters字段，则将其他所有字段视为参数（向后兼容）
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

            model_params[model_name] = config

        return model_params

    def _save_model_stats(self):
        """保存模型参数统计信息"""
        for model_name in self.models:
            model_config = self.model_params.get(model_name, {})

            # 准备统计信息
            stats = {
                "model_name": model_name,
                "model_class": model_config.get("model_class", "Unknown"),
                "parameters": model_config.get("parameters", {}),
                "config_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 尝试计算参数大小（如果可能的话）
            try:
                # 创建一个临时模型实例来计算参数
                temp_model = self.create_model(model_name, seed=1)
                
                # 优先使用模型的parameter_count方法
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
                        # 继续使用下面的备用方法
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
                    stats["parameter_size_mb"] = total_params * 4 / (1024 * 1024)  # 假设float32
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

    def get_classical_datasets(self):
        """获取Classical数据集列表"""
        datasets = self.data_generator.generate_dataset_list()['classical']
        logger.info(f"找到 {len(datasets)} 个Classical数据集")
        return datasets

    def create_model(self, model_name: str, seed: int, **kwargs):
        """创建模型实例"""
        # 获取模型配置
        model_config = self.model_params.get(model_name, {})

        # 获取模型类路径
        model_class_path = model_config.get("model_class")
        if not model_class_path:
            raise ValueError(f"No model class path found for {model_name}")

        # 获取模型类
        model_class = ModelRegistry.get_model_class(model_class_path)

        # 获取模型参数
        model_params = model_config.get("parameters", {}).copy()
        model_params.update(kwargs)

        # 特殊处理某些模型
        base_class_name = model_class_path.split(".")[-1]
        if base_class_name in ["PYOD", "supervised"]:
            # 这些模型需要额外的model_name参数，这里设置调用的默认模型
            if "model_name" not in model_params:
                if base_class_name == "PYOD":
                    model_params["model_name"] = "IForest"  # 默认使用IForest
                elif base_class_name == "supervised":
                    model_params["model_name"] = "LR"  # 默认使用LogisticRegression

        # 创建模型
        return model_class(seed=seed, **model_params)

    def run_single_experiment(self, params):
        """
        运行单个实验

        Args:
            params: 实验参数 (model_name, dataset, rla, seed)

        Returns:
            dict: 实验结果
        """
        model_name, dataset_name, rla, seed = params

        try:
            # 设置数据生成器
            self.data_generator.seed = seed
            self.data_generator.dataset = dataset_name

            # 生成数据
            data = self.data_generator.generator(
                la=rla,
                at_least_one_labeled=True,
                la_shortage_mode="ignore",
            )

            # 检查数据有效性
            if len(data["y_train"]) == 0 or np.sum(data["y_train"]) == 0:
                logger.warning(f"数据集 {dataset_name} (model={model_name}, seed={seed}, rla={rla}) 没有标注异常，跳过")
                return None

            # 创建模型
            model = self.create_model(model_name, seed)

            # 训练时间
            start_time = time.time()
            model.fit(data["X_train"], data["y_train"])
            fit_time = time.time() - start_time

            # 推理时间
            start_time = time.time()
            if hasattr(model, "predict_score"):
                scores = model.predict_score(data["X_test"])
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(data["X_test"])
            elif hasattr(model, "predict_proba"):
                proba = model.predict_proba(data["X_test"])
                scores = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
            else:
                raise AttributeError(f"模型 {model_name} 没有可用的评分方法")

            inference_time = time.time() - start_time

            # 计算性能指标
            metrics = self.utils.metric(y_true=data["y_test"], y_score=scores, pos_label=1)

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
                "error": str(e),
            }

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
        """
        运行所有实验

        使用初始化时设置的数据集和RLA列表
        """
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

        # 运行实验
        results = []
        if self.n_jobs == 1:
            # 串行运行
            for params in tqdm(experiment_params, desc="运行实验"):
                result = self.run_single_experiment(params)
                if result is not None:
                    results.append(result)
                    self._save_single_result(result)
        else:
            # 并行运行
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(self.run_single_experiment, params): params for params in experiment_params}

                for future in tqdm(as_completed(futures), total=len(futures), desc="运行实验"):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        self._save_single_result(result)

        logger.info(f"完成 {len(results)} 个实验")

        # 生成汇总报告
        self.generate_summary()

        return results

    def print_summary_stats(self, df):
        """打印简要统计信息"""
        print_summary_statistics(df)

    def generate_summary(self):
        """生成汇总报告"""
        logger.info("开始生成汇总报告...")

        generate_summary_only(self.output_dir)
        
        logger.info("汇总报告生成完成。")


def generate_summary_statistics(df, model_stats, summary_file):
    """
    通用的汇总统计函数

    Args:
        df: 实验结果DataFrame
        model_stats: 模型统计信息字典
        summary_file: 输出文件路径
    """
    # 按数据集汇总的效果
    dataset_summary = {}
    for metric in ["aucroc", "aucpr"]:
        dataset_summary[f"{metric}_mean"] = df.groupby("dataset")[metric].mean()
        dataset_summary[f"{metric}_std"] = df.groupby("dataset")[metric].std()

    # 按模型汇总的效果
    model_summary = {}
    for metric in ["aucroc", "aucpr"]:
        model_summary[f"{metric}_mean"] = df.groupby("model")[metric].mean()
        model_summary[f"{metric}_std"] = df.groupby("model")[metric].std()

    # 按数据集汇总的时间
    dataset_time_summary = {}
    for metric in ["fit_time", "inference_time"]:
        dataset_time_summary[f"{metric}_mean"] = df.groupby("dataset")[metric].mean()
        dataset_time_summary[f"{metric}_std"] = df.groupby("dataset")[metric].std()

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
            metric_data.to_excel(writer, sheet_name=f"数据集_{metric_name}")

        # 按模型汇总的效果
        for metric_name, metric_data in model_summary.items():
            metric_data.to_excel(writer, sheet_name=f"模型_{metric_name}")

        # 按数据集汇总的时间
        for metric_name, metric_data in dataset_time_summary.items():
            metric_data.to_excel(writer, sheet_name=f"数据集时间_{metric_name}")

        # 按模型汇总的时间
        for metric_name, metric_data in model_time_summary.items():
            metric_data.to_excel(writer, sheet_name=f"模型时间_{metric_name}")

        # 模型参数统计
        if model_stats:
            stats_df = pd.DataFrame.from_dict(model_stats, orient="index")
            stats_df.to_excel(writer, sheet_name="模型参数统计")

    logger.info(f"汇总结果已保存到: {summary_file}")


def print_summary_statistics(df):
    """
    通用的打印汇总统计函数

    Args:
        df: 实验结果DataFrame
    """
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

            # 读取结果文件（新格式使用固定文件名）
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
    summary_file = summary_dir / f"summary.xlsx"

    # 使用通用汇总函数
    generate_summary_statistics(combined_df, model_stats, summary_file)

    # 打印简要统计
    print_summary_statistics(combined_df)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通用模型在Classical数据集上的并行运行")

    parser.add_argument("--models", nargs="+", help="要运行的模型名称列表")

    parser.add_argument("--n_jobs", type=int, default=1, help="并行作业数量，-1表示使用所有CPU核心 (默认: 1)")

    parser.add_argument("--output_dir", type=str, default="results/tabular", help="输出目录 (默认: results)")

    parser.add_argument(
        "--parameter_config_path", type=str, default="WSADBench/model_configs/tabular", help="模型参数配置文件目录 (默认: model_configs)"
    )

    parser.add_argument("--datasets", nargs="+", default=None, help="指定运行的数据集名称，默认运行所有Classical数据集")

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

    args = parser.parse_args()

    # 如果只是要汇总
    if args.dry_summary:
        logger.info("仅进行汇总操作...")
        generate_summary_only(args.output_dir)
        return

    # 如果不是dry_summary模式，则检查必需的参数
    if not args.models:
        parser.error("--models is empty. Please specify at least one model.")

    # 预处理RLA列表
    _rla_list = []
    for rla in args.rla_list:
        if rla > 1:
            _rla_list.append(int(rla))
        else:
            _rla_list.append(rla)
    args.rla_list = _rla_list

    # 创建运行器
    runner = GeneralClassicalRunner(
        models=args.models,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        parameter_config_path=args.parameter_config_path,
        datasets=args.datasets,
        rla_list=args.rla_list,
        seed_list=args.seed_list,
    )

    # 运行实验
    logger.info(f"开始运行通用实验，模型: {args.models}")
    start_time = time.time()

    results = runner.run_experiments()

    total_time = time.time() - start_time
    logger.info(f"所有实验完成，总耗时: {total_time:.2f}秒")


if __name__ == "__main__":
    main()


"""
使用示例:

# 运行RoSAS模型
python run_general_classical.py --models RoSAS --n_jobs 4

# 运行多个模型
python run_general_classical.py --models RoSAS AABiGAN --n_jobs 8

# 指定特定数据集和RLA
python run_general_classical.py --models RoSAS --datasets cardio thyroid --rla_list 0.1 0.5 1.0

# 指定随机种子
python run_general_classical.py --models RoSAS --seed_list 1 2 3 4 5

# 使用自定义配置目录
python run_general_classical.py --models RoSAS --parameter_config_path ./my_configs

# 并行运行，使用所有CPU核心
python run_general_classical.py --models RoSAS AABiGAN --n_jobs -1

# 仅进行汇总（从已有的detail目录生成summary，不需要指定models参数）
python run_general_classical.py --dry_summary

# 指定输出目录进行汇总
python run_general_classical.py --dry_summary --output_dir ./results

# 快速测试
python run_general_classical.py --models AABiGAN --datasets 10_cover --seed_list 1 --rla_list 0.1 --n_jobs 1
"""
