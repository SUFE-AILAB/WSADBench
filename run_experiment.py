import os
from common_utils.baseline_utils import video_data2tabular_data
import re
import time
import gc
import argparse
import numpy as np
import pandas as pd
import yaml
import json

import math
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import logging
from typing import Dict, List, Optional, Any
import torch
import torch.multiprocessing as tmp
from itertools import product
from common_utils.argTypes import int_or_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from WSADBench.datasets.data_generator import DataGenerator
from WSADBench.myutils import Utils, import_class
import resource
import inspect
import cleanlab
from cleanlab import Datalab
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.ensemble import RandomForestClassifier


class ModelRegistry:
    """Model registrar for managing and creating different models."""
    @staticmethod
    def get_model_class(model_class_path: str):
        """Get the model class according to the model class path."""
        return import_class(model_class_path)

    @staticmethod
    def get_default_model_class_path(model_name: str):
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
            "TargAD": "WSADBench.baseline.TargAD.run.TargAD",
            "PUMA": "WSADBench.baseline.PUMA.run.PUMA",
            "TabNet": "WSADBench.baseline.TabNet.run.TabNet",
            "DualMGAN": "WSADBench.baseline.DualMGAN.run.DualMGAN",
            "TabPFN": "WSADBench.baseline.TabPFN.run.TabPFN",
            "TabMCls": "WSADBench.baseline.TabMCls.run.TabMCls",
            "TabR_S": "WSADBench.baseline.TabR_S.run.TabR_S",
            "AnoDDAE": "WSADBench.baseline.AnoDDAE.run.AnoDDAE",
            "LimiX": "WSADBench.baseline.LimiX.run.LimiX16M",
            # Unsupervised
            "PyOD": "WSADBench.baseline.PyOD.PYOD",
            "IForest": "WSADBench.baseline.PyOD.PYOD",
            "LOF": "WSADBench.baseline.PyOD.PYOD",
            "LUNAR": "WSADBench.baseline.PyOD.PYOD",
            "AutoEncoder": "WSADBench.baseline.PyOD.PYOD",
            "ECOD": "WSADBench.baseline.PyOD.PYOD",
            "PCA": "WSADBench.baseline.PyOD.PYOD",
            "CBLOF": "WSADBench.baseline.PyOD.PYOD",
            "VAE": "WSADBench.baseline.PyOD.PYOD",

            "OCSVM": "WSADBench.baseline.PyOD.PYOD",
            "KNN": "WSADBench.baseline.PyOD.PYOD",
            "HBOS": "WSADBench.baseline.PyOD.PYOD",
            "MCD": "WSADBench.baseline.PyOD.PYOD",
            "SOS": "WSADBench.baseline.PyOD.PYOD",
            "AAE": "WSADBench.baseline.PyOD.PYOD",
            "DeepSVDD": "WSADBench.baseline.PyOD.PYOD",

        }
        return default_model_map.get(model_name, None)


class ExperimentRunner:
    def __init__(
            self,
            models: List[str],
            data_type: str,
            n_jobs=1,
            output_dir=None,
            parameter_config_path=None,
            datasets=None,
            rla_list=None,
            eln_list=None,
            ru_list=None,
            flip_nr_list=None,
            flip_ar_list=None,
            target_for_unlabeled=None,
            noise_type=None,
            seed_list=None,
            gpu_list=None,
            DEBUG=False,
            NO_RESUME=False,
            is_cleanlab=False,
            exp_note=None,
    ):
        self.DEBUG = DEBUG
        self.NO_RESUME = NO_RESUME
        if data_type not in [
            "video",
            "tabular_classical",
            "tabular_CV_by_ResNet18",
            "tabular_CV_by_ViT",
            "tabular_NLP_by_BERT",
            "tabular_NLP_by_RoBERTa",
            # tabular_inexact
            "classical_bags_inexact",
            # OOD
            "tabular_CV_by_ResNet18_OOD",

        ]:
            raise ValueError(f"data_type must have 'video' or 'tabular'...... in it, got '{data_type}'")

        self.models = models
        self.data_type = data_type
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        # initial GPUManager
        self.gpu_manager = GPUManager(gpu_list, self.n_jobs)

        # Set default output directory and configuration path
        default_output_dir = f"results/{data_type}"
        if "tabular" in data_type:
            default_config_path = f"WSADBench/model_configs/tabular"
        elif "bags" in data_type:
            default_config_path = f"WSADBench/model_configs/tabular_bags_inexact"
        else:
            default_config_path = f"WSADBench/model_configs/{data_type}"

        self.output_dir = Path(output_dir) if output_dir else Path(default_output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.parameter_config_path = Path(parameter_config_path) if parameter_config_path else Path(default_config_path)
        self.parameter_config_path.mkdir(exist_ok=True, parents=True)

        self.detail_dir = self.output_dir / "detail"
        self.summary_dir = self.output_dir / "summary"
        self.detail_dir.mkdir(exist_ok=True, parents=True)
        self.summary_dir.mkdir(exist_ok=True, parents=True)

        # Create a subdirectory for each model
        self.model_dirs = {}
        for model_name in self.models:
            model_dir = self.detail_dir / model_name
            model_dir.mkdir(exist_ok=True, parents=True)
            self.model_dirs[model_name] = model_dir

        # Tool class
        self.utils = Utils()

        # Data generator
        self.data_generator = DataGenerator(generate_duplicates=True, n_samples_threshold=1000)

        # Experiment parameters
        self.seed_list = seed_list if seed_list is not None else list(range(1, 11))
        self.rla_list = rla_list if rla_list is not None else [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.eln_list = eln_list if eln_list is not None else [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.ru_list = ru_list if ru_list is not None else [1.0]
        self.flip_nr_list = flip_nr_list if flip_nr_list is not None else [0.01, 0.05, 0.1, 0.25, 0.5]
        self.flip_ar_list = flip_ar_list if flip_ar_list is not None else [0.01, 0.05, 0.1, 0.25, 0.5]
        self.target_for_unlabeled = target_for_unlabeled if target_for_unlabeled is not None else "fill_unlabel_0"
        self.noise_type = noise_type if noise_type is not None else None
        self.is_cleanlab = is_cleanlab
        self.exp_note = exp_note

        # Get dataset list
        if datasets is None:
            self.datasets = self.get_datasets()
        else:
            self.datasets = datasets

        # Model parameter dictionary
        self.model_params = self._load_model_parameters()

        # Save model parameter statistics
        self._save_model_stats()

        logger.info(f"Initializing models: {self.models}")
        logger.info(f"Data type: {self.data_type}")
        logger.info(f"Number of parallel jobs: {self.n_jobs}")
        logger.info(f"GPU configuration: {self.gpu_manager.get_gpu_assignment_summary()}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info(f"Parameter config path: {self.parameter_config_path.absolute()}")
        logger.info(f"Number of datasets: {len(self.datasets)}")
        logger.info(f"RLA settings: {self.rla_list}")
        logger.info(f"ELN settings: {self.eln_list}")
        logger.info(f"Unlabeled sample ratio settings: {self.ru_list}")
        logger.info(f"Normal sample mislabeling ratio settings: {self.flip_nr_list}")
        logger.info(f"Anomaly sample mislabeling ratio settings: {self.flip_ar_list}")
        logger.info(f"Unlabeled sample processing method: {self.target_for_unlabeled}")
        logger.info(f"Noise type: {self.noise_type}")
        logger.info(f"Whether to enable data noise cleaning: {self.is_cleanlab}")
        logger.info(f"Seeds: {self.seed_list}")

    def _load_model_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Load model parameter configuration"""
        model_params = {}

        for model_name in self.models:
            default_model_class_path = ModelRegistry.get_default_model_class_path(model_name)
            config = {"model_class": default_model_class_path, "parameters": {}}

            # Get the root path of the base model_configs
            # If self.parameter_config_path is WSADBench/model_configs/video,
            # then .parent will automatically truncate it to WSADBench/model_configs
            if self.parameter_config_path.name != "model_configs":
                base_config_path = self.parameter_config_path.parent
            else:
                base_config_path = self.parameter_config_path

            # Recursively find the corresponding YAML files in the model_configs root directory and all its subdirectories
            yaml_files = list(base_config_path.rglob(f"{model_name}.yaml"))

            if yaml_files:
                # Take the first found YAML file
                config_file = yaml_files[0]
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        yaml_config = yaml.safe_load(f)
                        if yaml_config:
                            # Use the model_class specified in the config file if present
                            if "model_class" in yaml_config:
                                config["model_class"] = yaml_config["model_class"]

                            # Update parameters if the config file contains a "parameters" field
                            if "parameters" in yaml_config:
                                config["parameters"].update(yaml_config["parameters"])
                            else:
                                # Backward compatibility: treat all other fields as parameters
                                yaml_params = {k: v for k, v in yaml_config.items() if k != "model_class"}
                                if yaml_params:
                                    config["parameters"].update(yaml_params)

                            logger.info(f"Loaded configuration for {model_name} from {config_file}")
                            logger.info(f"Model class: {config['model_class']}")
                except Exception as e:
                    logger.warning(f"Failed to load {config_file}: {e}")
            else:
                logger.info(
                        f"No configuration file found for {model_name} in {base_config_path} and its subdirectories, using default configuration")
                logger.info(f"Default model class: {config['model_class']}")

            if config["model_class"] is None:
                raise ValueError(f"Unknown model: {model_name} Please check the model name or configuration file.")

            model_params[model_name] = config

        return model_params

    def _save_model_stats(self):
        """Save statistical information of model parameters"""
        for model_name in self.models:
            model_config = self.model_params.get(model_name, {})

            # Prepare statistical information
            stats = {
                "model_name": model_name,
                "model_class": model_config.get("model_class", "Unknown"),
                "parameters": model_config.get("parameters", {}),
                "config_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_type": self.data_type,
            }

            # Try to calculate the size of parameters
            try:
                temp_model = self.create_model(self.model_params, model_name, seed=1)

                if hasattr(temp_model, "parameter_count"):
                    try:
                        param_stats = temp_model.parameter_count()
                        if isinstance(param_stats, dict) and "total" in param_stats:
                            stats["parameter_count"] = param_stats["total"]
                            stats["parameter_stats"] = param_stats
                        else:
                            logger.warning(
                                f"The return format of the parameter_count method for {model_name} does not meet expectations: {param_stats}")
                            stats["parameter_count"] = "Unknown"
                    except Exception as e:
                        logger.warning(f"Failed to call the parameter_count method of {model_name}: {e}")
                        stats["parameter_count"] = "Unknown"
                elif hasattr(temp_model, "get_params"):
                    # sklearn-style model
                    model_params = temp_model.get_params()
                    stats["sklearn_params"] = model_params
                    stats["parameter_count"] = len(model_params)
                elif hasattr(temp_model, "state_dict"):
                    # PyTorch model
                    state_dict = temp_model.state_dict()
                    total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, "numel"))
                    stats["parameter_count"] = int(total_params)
                else:
                    stats["parameter_count"] = "Unknown"

                del temp_model
                gc.collect()

            except Exception as e:
                logger.warning(f"Failed to calculate the parameter size of {model_name}: {e}")
                stats["parameter_count"] = "Unknown"
                stats["error"] = str(e)

            # Save statistical information
            stats_file = self.model_dirs[model_name] / "model_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Saved statistical information of {model_name} model to: {stats_file}")

    def get_datasets(self):
        """Get dataset list, used here to obtain the most suitable default test datasets when no datasets are specified"""
        if self.data_type == "video":
            datasets = self.data_generator.generate_dataset_list()["video"]
            logger.info(f"Found {len(datasets)} Video datasets")
        elif self.data_type == "tabular_classical":
            datasets = self.data_generator.generate_dataset_list()["classical"]
            logger.info(f"Found {len(datasets)} Classical datasets")
        elif self.data_type == "tabular_CV_by_ResNet18":
            datasets = self.data_generator.generate_dataset_list()["CV_by_ResNet18"]
            logger.info(f"Found {len(datasets)} CV_by_ResNet18 datasets")
        elif self.data_type == "tabular_CV_by_ViT":
            datasets = self.data_generator.generate_dataset_list()["CV_by_ViT"]
            logger.info(f"Found {len(datasets)} CV_by_ViT datasets")
        elif self.data_type == "tabular_NLP_by_BERT":
            datasets = self.data_generator.generate_dataset_list()["NLP_by_BERT"]
            logger.info(f"Found {len(datasets)} NLP_by_BERT datasets")
        elif self.data_type == "tabular_NLP_by_RoBERTa":
            datasets = self.data_generator.generate_dataset_list()["NLP_by_RoBERTa"]
            logger.info(f"Found {len(datasets)} NLP_by_RoBERTa datasets")
        elif self.data_type == "classical_bags_inexact":
            datasets = self.data_generator.generate_dataset_list()["classical_bags_inexact"]
            logger.info(f"Found {len(datasets)} classical_bags_inexact datasets")
        elif self.data_type == "tabular_CV_by_ResNet18_OOD":
            datasets = self.data_generator.generate_dataset_list()["CV_by_ResNet18_OOD"]
            logger.info(f"Found {len(datasets)} CV_by_ResNet18_OOD datasets")
        return datasets

    @staticmethod
    def create_model(model_params, model_name: str, seed: int, feature_shape: tuple = None, **kwargs):
        """Create model instance"""
        model_config = model_params.get(model_name, {})
        model_class_path = model_config.get("model_class")
        if not model_class_path:
            raise ValueError(f"No model class path found for {model_name}")
        model_class = ModelRegistry.get_model_class(model_class_path)
        model_params = model_config.get("parameters", {}).copy()
        model_params.update(kwargs)

        # Remove duplicate model_name
        if "model_name" in model_params:
            del model_params["model_name"]

        # If the __init__ method of the model has an input_dim parameter and feature_shape is not None, update input_dim
        init_signature = inspect.signature(model_class.__init__)
        if "input_dim" in init_signature.parameters and feature_shape is not None:
            model_params["input_dim"] = feature_shape[-1]
        # Check if the constructor has a model_name parameter
        if "model_name" in init_signature.parameters:
            model = model_class(model_name=model_name, seed=seed, **model_params)
        else:
            model = model_class(seed=seed, **model_params)
        # Create model
        return model

    @staticmethod
    def _process_video_data(data):
        """Special logic for processing video data"""
        # Reshape training data
        _clips_num, _crops_num, _dim = data["X_train"].shape
        data["X_train"] = data["X_train"].reshape(_clips_num * _crops_num, _dim)
        data["y_train"] = data["y_train"].repeat(_crops_num)

        # Preserve video ID information and extend to crops
        if "vid_train" in data:
            data["vid_train"] = data["vid_train"].repeat(_crops_num)

        # Reshape test data
        _clips_num, _crops_num, _dim = data["X_test"].shape
        data["X_test"] = data["X_test"].reshape(_clips_num * _crops_num, _dim)

        # Preserve video ID information and extend to crops
        if "vid_test" in data:
            data["vid_test"] = data["vid_test"].repeat(_crops_num)

        return data, (_clips_num, _crops_num)

    @staticmethod
    def _process_video_scores(scores, video_shape, data):
        """Special logic for processing video scores: restore from clip-level to frame-level"""
        _clips_num, _crops_num = video_shape  # tabular_inexact: [n_bags,n_samples,n_features]

        # Average over each crop to get the score for each clip
        scores = scores.reshape(_clips_num, _crops_num)
        scores = np.mean(scores, axis=1)

        # Restore clip-level scores to frame-level scores
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

    @staticmethod
    def _process_tabular_data(data):
        """Special logic for processing tabular_inexact data"""
        # Reshape training data
        n_bags, n_samples, _dim = data["X_train"].shape
        data["X_train"] = data["X_train"].reshape(n_bags * n_samples, _dim)
        # Broadcast bag labels to instance-level labels
        data["y_train"] = data["y_train"].repeat(n_samples)
        data["mask"] = data["mask"].repeat(n_samples)  # Mask must be broadcasted simultaneously

        # Preserve bag ID information and extend to samples
        if "bag_info_train" in data:
            data["bag_info_train"] = data["bag_info_train"].repeat(n_samples)

        # Reshape test data
        n_bags, n_samples, _dim = data["X_test"].shape
        data["X_test"] = data["X_test"].reshape(n_bags * n_samples, _dim)

        # Preserve bag ID information and extend to samples
        if "bag_info_test" in data:
            data["bag_info_test"] = data["bag_info_test"].repeat(n_samples)

        return data, (n_bags, n_samples)

    @staticmethod
    def _process_tabular_scores(scores, data_shape, data):
        """Special logic for processing tabular_inexact scores: restore from bag-level to sample-level"""
        n_bags, n_samples = data_shape  # Extract n_bags and n_samples

        # Average over each sample to get the score for each bag
        scores = scores.reshape(n_bags, n_samples)
        scores = np.mean(scores, axis=1)

        # Restore bag-level scores to sample-level scores
        y_test_idx = data["y_test_idx"]
        y_test_gt, y_test_gt_idx = data["y_test_gt"], data["y_test_gt_idx"]

        sample_truth = y_test_gt
        sample_scores = scores.repeat(n_samples)
        # Align lengths
        common_length = min(len(sample_truth), len(sample_scores))
        sample_scores = sample_scores[:common_length]
        sample_truth = sample_truth[:common_length]

        return sample_scores, sample_truth

    def _save_single_result(self, result):
        """Save a single experimental result"""
        if result is None:
            return

        model_name = result["model"]

        # Generate result file path
        old_result_file = self.model_dirs[model_name] / f"{model_name}_results.csv"
        result_file = self.model_dirs[model_name] / f"{model_name}_results.jsonl"

        if old_result_file.exists():
            logger.warning(
                f"Detected old result file {old_result_file}. Note that new results will be saved as {result_file} in JSONL format. It is recommended to delete the old file to avoid confusion. The old results will be backed up as a .bak file."
            )
            backup_file = old_result_file.with_suffix(".csv.bak")
            old_df = read_result_file(old_result_file)
            old_df.to_json(result_file, orient="records", lines=True)
            os.rename(old_result_file, backup_file)
        df = pd.DataFrame([result])
        if result_file.exists():
            df.to_json(result_file, mode="a", orient="records", lines=True)
        else:
            df.to_json(result_file, orient="records", lines=True)

        logger.debug(
            f"save {model_name} experiment result: "
            f"{result.get('dataset')}, "
            f"seed={result.get('seed')}, "
            f"rla={result.get('rla')}, "
            f"eln={result.get('eln')}, "
            f"ru={result.get('ru')}, "
            f"flip_normal_ratio={result.get('flip_normal_ratio')}, "
            f"flip_abnormal_ratio={result.get('flip_abnormal_ratio')}, "
            f"target_for_unlabeled={result.get('target_for_unlabeled')}, "
            f"is_cleanlab={result.get('is_cleanlab')}, "
            f"noise_type={result.get('noise_type')}, "
            f"exp_note={result.get('exp_note', '')}"
        )

    def _load_finish_exp(self):

        main_keys = ["model", "dataset", "rla", "eln", "ru", "flip_normal_ratio", "flip_abnormal_ratio",
                     "target_for_unlabeled", "seed", "is_cleanlab", "exp_note"]
        finished_experiments = set()
        for model_name in self.models:
            detail_file = self.model_dirs[model_name] / f"{model_name}_results.jsonl"

            if detail_file.exists():
                try:
                    df = read_result_file(detail_file)
                    df = df[df["aucroc"].notna() & (df["aucroc"] > 0)]  # Only consider experiments with valid results and values greater than 0
                    main_setting = df[main_keys].drop_duplicates().to_numpy().tolist()
                    finished_experiments.update([tuple(row) for row in main_setting])
                except Exception as e:
                    pass
        return finished_experiments

    def run_experiments(self):
        if self.data_type == "video":
            try:
                vid_info = self.datasets[-1]  # seg_32_200_pm_i3d_mvit
                self.datasets = self.datasets[:-1]
                seg_list = [seg for seg in vid_info.split("pm")[0].split("_")[1:] if seg]  # Filter out empty strings
                pm_list = [pm for pm in vid_info.split("pm")[1].split("_")[1:] if pm]  # Filter out empty strings
                # Generate individual Cartesian product for the dataset
                self.datasets = [f"{ds}.s{seg}.m{pm}" for ds in self.datasets for seg in seg_list for pm in pm_list]
            except Exception as e:
                logger.error(f"Failed to process the pretrained models and number of seg for the video dataset: {e}")
                raise e
        elif self.data_type == "tabular_CV_by_ResNet18_OOD":
            try:
                ds_list = [ds.split('_')[0] if 'metal' not in ds else 'metal_nut' for ds in self.datasets]
                class_dict = {'aitex': ['broken_pick', 'fuzzyball', 'weft_crack', ],
                              'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
                              'elpv': ['mono', ],
                              'hyperkvasir': ['barretts', 'barretts-short-segment', 'esophagitis-b-d', ],
                              'mastcam': ['broken-rock', 'drill-hole', 'drt', 'dump-pile', 'float', 'meteorite',
                                          'veins'],
                              'metal_nut': ['bent', 'color', 'flip', 'scratch']}
                if len(self.exp_note) == 1 and self.exp_note[0] not in ['rla_emb_near_inc', 'rla_emb_know_far_inc',
                                                                        'rla_emb_know_near_inc']:
                    self.datasets = [f"{ds}.{class_type}" for ds, ds_key in zip(self.datasets, ds_list) for class_type
                                     in class_dict[ds_key]]
                else:
                    self.datasets = [f"{ds}._" for ds in self.datasets]  # Use underscore _ to represent class_type
                    pass
            except Exception as e:
                logger.error(f"Failed to process the OOD dataset: {e}")
                raise e

        # Generate combinations of experimental parameters
        experiment_params = list(
            product(self.models, self.datasets, self.rla_list, self.eln_list, self.ru_list, self.flip_nr_list,
                    self.flip_ar_list, self.target_for_unlabeled, self.seed_list, self.is_cleanlab, self.exp_note))

        finished_experiments = self._load_finish_exp()
        if finished_experiments and not self.NO_RESUME:
            num_before_skip = len(experiment_params)
            experiment_params = [params for params in experiment_params if params not in finished_experiments]
            logger.info(f"Skipping {num_before_skip - len(experiment_params)} completed experiments")

        if not experiment_params:
            logger.info("No experiments need to be run; all experiments are completed or skipped")
            return []

        logger.info(f"Total number of experiments: {len(experiment_params)}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Number of datasets: {len(self.datasets)}")
        logger.info(f"RLA settings: {self.rla_list}")
        logger.info(f"ELN settings: {self.eln_list}")
        logger.info(f"Unlabeled sample ratio settings: {self.ru_list}")
        logger.info(f"Normal sample mislabeling ratio settings: {self.flip_nr_list}")
        logger.info(f"Anomalous sample mislabeling ratio settings: {self.flip_ar_list}")
        logger.info(f"Unlabeled sample processing method: {self.target_for_unlabeled}")
        logger.info(f"Noise type: {self.noise_type}")
        logger.info(f"Whether to enable data noise cleaning function: {self.is_cleanlab}")
        logger.info(f"Seeds: {self.seed_list}")

        # Prepare experimental configuration to pass to child processes (excluding the complete runner)
        experiment_config = {
            "data_type": self.data_type,
            "model_params": self.model_params,
        }

        # Assign GPUs to each task
        experiment_params_with_gpu = []
        for i, params in enumerate(experiment_params):
            gpu_id = self.gpu_manager.get_gpu_for_task(i)
            experiment_params_with_gpu.append((params, gpu_id, experiment_config, self.DEBUG))

        # run experiment
        results = []
        if self.n_jobs == 1:
            # Run serially
            for params in tqdm(experiment_params_with_gpu, desc="run exp"):
                result = run_single_experiment_with_gpu(params)
                if result is not None:
                    results.append(result)
                    self._save_single_result(result)
        else:
            # Run in parallel - using torch.multiprocessing
            # Set multiprocessing start method
            if tmp.get_start_method(allow_none=True) != "spawn":
                tmp.set_start_method("spawn", force=True)

            # Execute using torch.multiprocessing.Pool
            with tmp.Pool(self.n_jobs) as pool:
                for result in tqdm(
                        pool.imap(run_single_experiment_with_gpu, experiment_params_with_gpu),
                        total=len(experiment_params_with_gpu),
                        desc="Running experiments",
                ):
                    if result is not None:
                        results.append(result)
                        self._save_single_result(result)

        logger.info(f"Completed {len(results)} experiments")
        return results

def generate_summary_statistics(df, model_stats, summary_file):
    """General summary statistics function"""
    # Performance summary by dataset
    dataset_summary = {}
    for metric in ["aucroc", "aucpr"]:
        dataset_summary[f"{metric}_mean"] = df.groupby(["dataset", "model", "rla"])[metric].mean()
        dataset_summary[f"{metric}_std"] = df.groupby(["dataset", "model", "rla"])[metric].std()

    # Performance summary by model
    model_summary = {}
    for metric in ["aucroc", "aucpr"]:
        model_summary[f"{metric}_mean"] = df.groupby("model")[metric].mean()
        model_summary[f"{metric}_std"] = df.groupby("model")[metric].std()

    # Time summary by dataset
    dataset_time_summary = {}
    for metric in ["fit_time", "inference_time"]:
        dataset_time_summary[f"{metric}_mean"] = df.groupby(
            ["dataset", "model", "rla"])[metric].mean()
        dataset_time_summary[f"{metric}_std"] = df.groupby(
            ["dataset", "model", "rla"])[metric].std()

    # Time summary by model
    model_time_summary = {}
    for metric in ["fit_time", "inference_time"]:
        model_time_summary[f"{metric}_mean"] = df.groupby("model")[metric].mean()
        model_time_summary[f"{metric}_std"] = df.groupby("model")[metric].std()

    # Save summary results to Excel
    with pd.ExcelWriter(summary_file) as writer:
        # Raw detailed results
        df.to_excel(writer, sheet_name="Detailed_Results", index=False)

        # Performance summary by dataset
        for metric_name, metric_data in dataset_summary.items():
            metric_data.to_excel(writer, sheet_name=f"Performance_{metric_name}")

        # Performance summary by model
        for metric_name, metric_data in model_summary.items():
            metric_data.to_excel(writer, sheet_name=f"Model_{metric_name}")

        # Time summary by dataset
        for metric_name, metric_data in dataset_time_summary.items():
            metric_data.to_excel(writer, sheet_name=f"Time_{metric_name}")

        # Time summary by model
        for metric_name, metric_data in model_time_summary.items():
            metric_data.to_excel(writer, sheet_name=f"Model_Time_{metric_name}")

        # Model parameter statistics
        if model_stats:
            stats_df = pd.DataFrame.from_dict(model_stats, orient="index")
            stats_df.to_excel(writer, sheet_name="Model_Parameters")

    logger.info(f"Summary results saved to: {summary_file}")


def read_result_file(result_file):
    result_file = Path(result_file)
    suffix = result_file.suffix.lower()

    # Target dtype: object
    target_object_cols = [
        "rla", "eln", "ru",
        "flip_normal_ratio", "flip_abnormal_ratio",
        "target_for_unlabeled", "seed"
    ]

    # === 1. Read the file ===
    if suffix == ".csv":
        df = pd.read_csv(result_file, dtype={col: 'object' for col in target_object_cols})
    elif suffix in [".jsonl", ".json"]:
        # pd.read_json does not support the dtype parameter
        df = pd.read_json(result_file, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # === 2. Uniformly convert dtype for columns read from json/jsonl ===
    for col in target_object_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # === 3. Automatically handle inconsistent fields across different model results ===
    # Fill missing fields with NaN (pd.NA)
    for col in target_object_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df

def print_summary_statistics(df):
    """General function to print summary statistics."""
    logger.info("\n" + "=" * 50)
    logger.info("Summary Statistics of Experimental Results")
    logger.info("=" * 50)

    # Overall statistics
    valid_results = df.dropna(subset=["aucroc", "aucpr"])
    logger.info(f"Number of valid experiments: {len(valid_results)}/{len(df)}")

    if len(valid_results) > 0:
        # Statistics by model
        logger.info("\nStatistics by model:")
        for model in sorted(valid_results["model"].unique()):
            model_data = valid_results[valid_results["model"] == model]
            logger.info(
                f"{model}: AUCROC={model_data['aucroc'].mean():.4f} ± {model_data['aucroc'].std():.4f}, "
                f"AUCPR={model_data['aucpr'].mean():.4f} ± {model_data['aucpr'].std():.4f}"
            )


def generate_summary_only(output_dir):
    """Independent summary function, solely for aggregating existing results."""
    output_dir = Path(output_dir)
    detail_dir = output_dir / "detail"
    summary_dir = output_dir / "summary"

    if not detail_dir.exists():
        logger.error(f"Detailed results directory does not exist: {detail_dir}")
        return

    summary_dir.mkdir(exist_ok=True, parents=True)

    # Collect all model directories and result files
    all_results = []
    model_stats = {}

    for model_dir in detail_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name

            # Read model statistics
            stats_file = model_dir / "model_stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    model_stats[model_name] = json.load(f)

            # Read result files
            result_file = model_dir / f"{model_name}_results.jsonl"

            if not result_file.exists():
                if result_file.with_suffix(".csv").exists():
                    # Compatible with old CSV result files: convert them to JSONL format and back up the old file
                    old_result_file = result_file.with_suffix(".csv")
                    df = read_result_file(old_result_file)
                    df.to_json(result_file, orient="records", lines=True)
                    backup_file = old_result_file.with_suffix(".csv.bak")
                    os.rename(old_result_file, backup_file)
                    logger.info(f"Converted old result file {old_result_file} to {result_file}, and backed it up as {backup_file}")

            if result_file.exists():
                try:
                    df = read_result_file(result_file)
                    all_results.append(df)
                    logger.info(f"Read result file: {result_file} ({len(df)} records)")
                except Exception as e:
                    logger.warning(f"Failed to read result file {result_file}: {e}")

    if not all_results:
        logger.warning("No result files found")
        return

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"Combined {len(all_results)} result files, totaling {len(combined_df)} experimental results")

    # Deduplicate
    before = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    after = len(combined_df)
    logger.info(f"Deduplication complete: removed {before - after} duplicate records, {after} records remaining")

    # Clean invalid numeric data
    target_object_cols = [
        'fit_time', 'inference_time']
    for col in target_object_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    # Generate summary statistics
    summary_file = summary_dir / "summary.xlsx"

    # Use the general summary function
    generate_summary_statistics(combined_df, model_stats, summary_file)

    # Print brief statistics
    print_summary_statistics(combined_df)
class GPUManager:
    """GPU Resource Manager"""

    def __init__(self, gpu_list=None, n_jobs=1):
        """
        Initialize GPU Manager

        Args:
            gpu_list: Specify the list of GPUs to use, e.g., [0,1,2]. None means auto-detection
            n_jobs: Total number of concurrent tasks
        """
        self.available_gpus = self._detect_gpus(gpu_list)
        self.n_jobs = n_jobs
        self.num_gpus = len(self.available_gpus)

        if self.num_gpus == 0:
            logger.warning("No available GPUs detected, will use CPU mode")
        else:
            logger.info(f"Detected {self.num_gpus} available GPUs: {self.available_gpus}")
            logger.info(f"Number of concurrent tasks: {n_jobs}, maximum concurrent tasks per GPU: {math.ceil(n_jobs / self.num_gpus)}")

    def _detect_gpus(self, gpu_list):
        """Detect available GPUs"""
        if gpu_list is not None:
            # User-specified GPU list
            if isinstance(gpu_list, str):
                # Support "0,1,2" format
                return [int(x.strip()) for x in gpu_list.split(",")]
            elif isinstance(gpu_list, list):
                return gpu_list
            else:
                return [gpu_list]
        else:
            # Auto-detect all available GPUs
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
            else:
                return []

    def get_gpu_for_task(self, task_index):
        """Get the GPU ID that should be assigned to the task"""
        if self.num_gpus == 0:
            return None
        return self.available_gpus[task_index % self.num_gpus]

    def get_gpu_assignment_summary(self):
        """Get a summary of the GPU assignments"""
        if self.num_gpus == 0:
            return "CPU Mode"

        tasks_per_gpu = {}
        for i in range(self.n_jobs):
            gpu_id = self.get_gpu_for_task(i)
            tasks_per_gpu[gpu_id] = tasks_per_gpu.get(gpu_id, 0) + 1

        return f"GPU Assignment: {dict(sorted(tasks_per_gpu.items()))}"

# Wrapper class added to adapt for cleanlab
class CleanlabWSADWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # Your custom training logic
        self.model.train_loop(X, y)
        return self

    def predict_proba(self, X):
        logits = self.model.predict_score(X)  # or model.forward
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
def run_single_experiment_with_gpu(params_with_config):
    """
    Experiment execution function with GPU assignment
    """
    params, gpu_id, experiment_config, DEBUG = params_with_config
    model_name, dataset_name, rla, eln, ru, flip_normal_ratio, flip_abnormal_ratio, target_for_unlabeled, seed, is_cleanlab, exp_note = params

    if flip_normal_ratio > 0 or flip_abnormal_ratio > 0:
        noise_type = "label_contamination"
    else:
        noise_type = None

    # Set GPU environment
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(
            f"Task {model_name}-{dataset_name}(seed={seed}, rla={rla},eln={eln},ru={ru},noise_type={noise_type},exp_note={exp_note}) assigned to GPU {gpu_id}")
    else:
        # CPU mode
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info(
            f"Task {model_name}-{dataset_name}(seed={seed}, rla={rla},eln={eln},ru={ru},noise_type={noise_type},exp_note={exp_note}) using CPU mode")

    try:
        # Retrieve required components from the configuration
        data_type = experiment_config["data_type"]
        model_params = experiment_config["model_params"]
        utils = Utils()

        # Create data generator
        data_generator = DataGenerator(generate_duplicates=True, n_samples_threshold=1000)
        data_generator.seed = seed
        data_generator.dataset = dataset_name
        # Generate data
        data = data_generator.generator(
            la=rla,
            eln=eln,
            ru=ru,
            flip_normal_ratio=flip_normal_ratio,
            flip_abnormal_ratio=flip_abnormal_ratio,
            target_for_unlabeled=target_for_unlabeled,
            noise_type=noise_type,
            at_least_one_labeled=True,
            shortage_mode="ignore",
            data_type=data_type,
            exp_note=exp_note
        )

        # Check data validity
        if len(data["y_train"]) == 0:
            logger.warning(
                f"Dataset {dataset_name} (model={model_name}, seed={seed}, rla={rla},eln={eln},ru={ru},flip_normal_ratio={flip_normal_ratio},flip_abnormal_ratio={flip_abnormal_ratio},target_for_unlabeled={target_for_unlabeled},noise_type={noise_type},exp_note={exp_note}) has no labeled anomalies, skipping")
            return None

        # Create model
        feature_shape = data["X_train"].shape
        model = ExperimentRunner.create_model(model_params, model_name, seed=seed, feature_shape=feature_shape)

        # Process data based on data type
        data_shape = None
        if data_type == "video":
            data, data_shape = ExperimentRunner._process_video_data(data)
        elif "inexact" in data_type:
            data, data_shape = ExperimentRunner._process_tabular_data(data)  # -> tabular_inexact

        # Training start time
        start_time = time.time()

        def has_param(func, param_name):
            """Check if a function has a specific parameter"""
            return param_name in inspect.signature(func).parameters

        # Determine whether to trigger the transformation of X_train and y_train
        if data_type == "video" and model_name not in ["ARNet", "MGFN", "RTFM", "Sultani", "VadClip", "URDMU",
                                                       "ZhongGCNAD"]:
            data = video_data2tabular_data(data, data_shape, model_name, seed)

        train_input = {}
        if has_param(model.fit, "X"):
            train_input["X"] = data["X_train"]
        if has_param(model.fit, "y"):
            train_input["y"] = data["y_train"]
        if has_param(model.fit, "X_train"):
            train_input["X_train"] = data["X_train"]
        if has_param(model.fit, "y_train"):
            train_input["y_train"] = data["y_train"]
        if has_param(model.fit, "mask"):
            train_input["mask"] = data["mask"]
        if has_param(model.fit, "bags_info"):
            train_input["bags_info"] = data.get("bag_info_train", None)
        if has_param(model.fit, "vid_info"):
            train_input["vid_info"] = data.get("vid_train", None)
        if has_param(model.fit, "crops_num"):
            train_input["crops_num"] = data_shape[1] if data_shape else None
        if has_param(model.fit, "vid_kind"):
            train_input["vid_kind"] = data.get("vid_kind_train", None)
        if has_param(model.fit, "vid_source_clips_num"):
            train_input["vid_source_clips_num"] = data.get("vid_source_clips_num_train", None)

        if has_param(model.fit, "X_test"):  # X_test, video_shape, y_test_idx, y_test_gt, y_test_gt_idx, num_clip_frames

            if "inexact" in data_type or 'video' in data_type:

                train_input["X_test"] = [
                    data["X_test"],
                    data_shape,
                    data["y_test_idx"],
                    data["y_test_gt"],
                    data["y_test_gt_idx"],
                    data["NUM_FRAMES"],
                ]
            else:
                train_input["X_test"] = [
                    data["X_test"],
                    data["y_test"]
                ]

        if has_param(model.fit, "X_test_extra"):  # vid_kind, vid_source_clips_num,crops_num
            train_input["X_test_extra"] = [
                data.get("vid_kind_test", None),
                data.get("vid_source_clips_num_test", None),
                data_shape[1] if data_shape else None,
            ]
        # visualization
        if has_param(model.fit, "emb_vis"):
            train_input["emb_vis"] = data.get("emb_vis", None)

        pred_func = None
        if hasattr(model, "predict_score"):
            pred_func = model.predict_score
        elif hasattr(model, "decision_function"):
            pred_func = model.decision_function
        elif hasattr(model, "predict_proba"):
            pred_func = model.predict_proba
        else:
            raise AttributeError(f"Model {model_name} has no available scoring method")

        test_input = {}
        if has_param(pred_func, "X_train"):
            test_input["X_train"] = data["X_train"]
        if has_param(pred_func, "y_train"):
            test_input["y_train"] = data["y_train"]
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

        if is_cleanlab == "true":
            print("Enabling CleanLearning for data cleaning...")
            if "X_train" in train_input:
                X, y = train_input["X_train"], train_input["y_train"]
            else:
                X, y = train_input["X"], train_input["y"]

            clf = RandomForestClassifier()  # Use Random Forest as the cleaning model
            cl = CleanLearning(clf, cv_n_folds=5, seed=seed)
            cl.fit(X, y)
            # Get the label indices after cleaning
            label_issues = cl.find_label_issues(X, y)
            issues = np.array(label_issues['is_label_issue'])
            label_quality = np.array(label_issues["label_quality"])

            # 1) Find all noisy sample indices
            noise_idx = np.where(issues)[0]  # Samples where is_label_issue == True

            # If there are no noisy samples, return empty directly
            if len(noise_idx) == 0:
                low_quality_noise_idx = np.array([], dtype=int)
            else:
                # 2) Get the label_quality of these samples
                noise_label_quality = label_quality[noise_idx]

                # 3) Select the lowest 30% label_quality among noisy samples
                k = int(len(noise_idx) * 0.30)
                k = max(k, 1)  # Avoid k = 0

                # 4) Find the indices of the internally sorted noisy samples
                local_sorted_idx = np.argsort(noise_label_quality)[:k]

                # 5) Map back to the original sample indices
                low_quality_noise_idx = noise_idx[local_sorted_idx]

            y[low_quality_noise_idx] = 1 - y[low_quality_noise_idx]  # Correct the noisy sample labels

            logger.info(
                f"CleanLearning detected and corrected {len(low_quality_noise_idx)} high-confidence low-quality noisy label samples")

            if 'X' in train_input:  # Update training data
                train_input["X"], train_input["y"] = X, y
            else:
                train_input["X_train"], train_input["y_train"] = X, y

        model.fit(**train_input)

        fit_time = time.time() - start_time  
 
        if data_type == "tabular_CV_by_ResNet18_OOD":  # operate ood_res
            res_list = []
            
            rate_list = [0, 25, 50, 75, 100]  # [0, 25, 50, 75, 100]
            for rate in rate_list:
                test_input = {}
                if has_param(pred_func, "X"):
                    test_input["X"] = data["X_test_dict"][rate]  # get X_test for each rate
                if has_param(pred_func, "X_test"):
                    test_input["X_test"] = data["X_test_dict"][rate]
                start_time = time.time()
                proba = pred_func(**test_input)
                if proba.ndim == 1:
                    scores = proba
                else:
                    scores = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()

                inference_time = time.time() - start_time


                metrics = utils.metric(y_true=data["y_test"], y_score=scores, pos_label=1)

                result = {
                    "model": model_name,
                    "dataset": f'{dataset_name}.{rate}',  #
                    "rla": rla,
                    "eln": eln,
                    "ru": ru,
                    "flip_normal_ratio": flip_normal_ratio,
                    "flip_abnormal_ratio": flip_abnormal_ratio,
                    "target_for_unlabeled": target_for_unlabeled,
                    "seed": seed,
                    "aucroc": metrics["aucroc"],
                    "aucpr": metrics["aucpr"],
                    "noise_type": noise_type,
                    "fit_time": fit_time,
                    "inference_time": inference_time,
                    "n_train": len(data["y_train"]),
                    "n_test": len(data["y_test"]),
                    "n_train_anomalies": np.sum(data["y_train"]),
                    "n_test_anomalies": np.sum(data["y_test"]),
                    "error": "",
                    "data_type": data_type,
                    "exp_note": exp_note,
                }   
                res_list.append(result)
                logger.info(
                    f"finish {model_name} - {result['dataset']} (seed={seed}, rla={rla},eln={eln},ru={ru},flip_normal_ratio={flip_normal_ratio},flip_abnormal_ratio={flip_abnormal_ratio},target_for_unlabeled={target_for_unlabeled},noise_type={noise_type}): \n"
                    f"AUCROC={metrics['aucroc']:.4f}, AUCPR={metrics['aucpr']:.4f}"
                )
            save_path_dir = r'./results/tabular_CV_by_ResNet18_OOD_new_res/detail'
            os.makedirs(save_path_dir, exist_ok=True)
            file_folder = os.path.join(save_path_dir, model_name)
            os.makedirs(file_folder, exist_ok=True)
            jsonl_path = os.path.join(file_folder, f"{model_name}_results.jsonl")
            df = pd.DataFrame(res_list) # add 5 res
            if os.path.exists(jsonl_path):
                df.to_json(jsonl_path, orient="records", lines=True, mode="a")
            else:
                df.to_json(jsonl_path, orient="records", lines=True)

            logger.debug(f"save {model_name} ood results to: {jsonl_path}, num of results: {len(res_list)}")

            del model, data, scores
            gc.collect()
            result['dataset'] = dataset_name
            return result
        else:  # not ood predict

            start_time = time.time()
            proba = pred_func(**test_input)
            if proba.ndim == 1:
                scores = proba
            else:
                scores = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()

            inference_time = time.time() - start_time
            if data_type == "video":
                frame_scores, frame_truth = ExperimentRunner._process_video_scores(scores, data_shape, data)
                metrics = utils.metric(y_true=frame_truth, y_score=frame_scores, pos_label=1)
            elif "inexact" in data_type:
                sample_scores, sample_truth = ExperimentRunner._process_tabular_scores(scores, data_shape, data)
                metrics = utils.metric(y_true=sample_truth, y_score=sample_scores, pos_label=1)
            else:
                metrics = utils.metric(y_true=data["y_test"], y_score=scores, pos_label=1)

            result = {
                "model": model_name,
                "dataset": dataset_name,
                "rla": rla,
                "eln": eln,
                "ru": ru,
                "flip_normal_ratio": flip_normal_ratio,
                "flip_abnormal_ratio": flip_abnormal_ratio,
                "target_for_unlabeled": target_for_unlabeled,
                "seed": seed,
                "aucroc": metrics["aucroc"],
                "aucpr": metrics["aucpr"],
                "noise_type": noise_type,
                "is_cleanlab": is_cleanlab,
                "fit_time": fit_time,
                "inference_time": inference_time,
                "n_train": len(data["y_train"]),
                "n_test": len(data["y_test"]),
                "n_train_anomalies": np.sum(data["y_train"]),
                "n_test_anomalies": np.sum(data["y_test"]),
                "error": "",
                "data_type": data_type,
                "exp_note": exp_note
            }

            logger.info(
                f"finish {model_name} - {dataset_name} (seed={seed}, rla={rla},eln={eln},ru={ru},flip_normal_ratio={flip_normal_ratio},flip_abnormal_ratio={flip_abnormal_ratio},target_for_unlabeled={target_for_unlabeled},noise_type={noise_type},exp_note={exp_note}): "
                f"AUCROC={metrics['aucroc']:.4f}, AUCPR={metrics['aucpr']:.4f}"
            )

            del model, data, scores
            gc.collect()

            return result

    except Exception as e:
        if DEBUG:
            raise e
        logger.error(
            f"fail {model_name} - {dataset_name} (seed={seed}, rla={rla},eln={eln},ru={ru},flip_normal_ratio={flip_normal_ratio},flip_abnormal_ratio={flip_abnormal_ratio},target_for_unlabeled={target_for_unlabeled},noise_type={noise_type}): {str(e)}")
        return {
            "model": model_name,
            "dataset": dataset_name,
            "rla": rla,
            "eln": eln,
            "ru": ru,
            "flip_normal_ratio": flip_normal_ratio,
            "flip_abnormal_ratio": flip_abnormal_ratio,
            "target_for_unlabeled": target_for_unlabeled,
            "seed": seed,
            "aucroc": np.nan,
            "aucpr": np.nan,
            "noise_type": noise_type,
            "is_cleanlab": is_cleanlab,
            "fit_time": np.nan,
            "inference_time": np.nan,
            "n_train": np.nan,
            "n_test": np.nan,
            "n_train_anomalies": np.nan,
            "n_test_anomalies": np.nan,
            "error": str(e).replace("\n", " ").replace(",", " "),
            "data_type": data_type,
            "exp_note": exp_note
        }


def main():
    """Unlock file descriptor limits"""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Original limits: soft={soft}, hard={hard}")
    # Set to 4096 (Note: cannot exceed hard limit, otherwise an error will occur)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
    # Verify modification results
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Modified limits: soft={soft}, hard={hard}")

    parser = argparse.ArgumentParser(description="Unified anomaly detection experiment runner")


    parser.add_argument(
        "--data_type",
        choices=[
            "video",
            "tabular_classical",
            "tabular_CV_by_ResNet18",
            "tabular_CV_by_ViT",
            "tabular_NLP_by_BERT",
            "tabular_NLP_by_RoBERTa",
            "classical_bags_inexact",
            "tabular_CV_by_ResNet18_OOD",
        ],
        required=True,
        default=["tabular_classical"],
        help="type of data",
    )

    parser.add_argument("--models", nargs="+", help="List of model names to run")

    # Optional parameters
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs, -1 means using all CPU cores (Default: 1)")

    parser.add_argument("--output_dir", type=str, help="Output directory (Default: results/{data_type})")

    parser.add_argument(
        "--parameter_config_path",
        type=str,
        help="Directory path for model parameter configuration files (Default: WSADBench/model_configs/{data_type})",
        # video / tabular
    )

    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specify the names of the datasets to run; defaults to running all datasets")

    parser.add_argument(
        "--rla_list",
        nargs="+",
        type=int_or_float,
        default=[1.0],
        help="List of ratios for labeled anomalies (Default: 0.01 0.05 0.1 0.25 0.5 0.75 1.0)",
    )

    parser.add_argument(
        "--eln_list",
        nargs="+",
        type=int_or_float,
        default=[0.0],  # 0.0 means abnormal only; 1.0 means equal number of abnormal and normal samples
        help="List of ratios for expected labeled normal samples, relative to the number of labeled anomalies (Default: 0.01 0.05 0.1 0.25 0.5 0.75 1.0)",
    )

    parser.add_argument(
        "--ru_list",
        nargs="+",
        type=int_or_float,
        default=[1.0],
        help="List of ratios for unlabeled samples (Default: 1.0)",
    )

    parser.add_argument(
        "--flip_nr_list",
        nargs="+",
        type=int_or_float,
        default=[0.0],
        help="List of error labeling ratios for normal samples / false positive rate (Default: 0.01 0.05 0.1 0.25 0.5)",
    )

    parser.add_argument(
        "--flip_ar_list",
        nargs="+",
        type=int_or_float,
        default=[0.0],
        help="List of error labeling ratios for abnormal samples / false negative rate (Default: 0.01 0.05 0.1 0.25 0.5)",
    )

    parser.add_argument(
        "--target_for_unlabeled",
        nargs="+",
        type=str,
        choices=["fill_unlabel_0", "keep_label", "delete_sample"],
        default=["fill_unlabel_0"],
        help="Target handling method for unlabeled data (Default: fill_unlabel_0, Options: fill_unlabel_0, keep_label, delete_sample... to be supplemented)",
    )

    parser.add_argument(
        "--noise_type",
        nargs="+",
        type=str,
        choices=[None, "label_contamination"],
        default=[None],
        help="Noise type (Default: None, Options: label_contamination... to be supplemented)",
    )

    parser.add_argument(
        "--is_cleanlab",
        nargs="+",
        type=str,
        choices=["true", "false"],
        default=["false"],
        help="Switch parameter to enable data cleaning. " \
             "Default: false, consistent with previous experimental conditions.",
    )

    parser.add_argument(
        "--seed_list",
        nargs="+",
        type=int,
        default=list(range(0, 5)),
        help="Random seed list (Default: 0 1 2 3 4)",
    )

    parser.add_argument(
        "--dry_summary",
        action="store_true",
        help="Only perform summarization, do not run experiments",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='auto',
        help="Specify GPUs (format: 0,1,2 or auto for auto-detect all GPUs; default: auto)",
    )

    parser.add_argument(
        "--exp_note",
        nargs="+",
        type=str,
        default=['None'],
        help="Experiment notes: used to distinguish different experiments.",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="Enable debug mode to catch all exceptions and print detailed error information.",
    )

    parser.add_argument(
        "--NO_RESUME",
        action="store_true",
        help="If this option is set, completed experiments will not be skipped, and all experiments will be forcibly re-run.",
    )

    args = parser.parse_args()

    # If you only need to summarize
    if args.dry_summary:
        logger.info("Only perform summary operations...")
        output_dir = args.output_dir if args.output_dir else f"results/{args.data_type}"
        generate_summary_only(output_dir)
        return

    # If it is not in dry_summary mode, check the required parameters.
    if not args.models:
        parser.error("--models is required when not using --dry_summary. Please specify at least one model.")

    # gpu param
    gpu_list = None
    if args.gpus is not None:
        if args.gpus.lower() == "auto":
            gpu_list = None  # auto detect
        else:
            gpu_list = args.gpus.strip()
            if not re.match(r'^[\d,]+$', gpu_list):
                raise ValueError(f"wrong GPU index: {gpu_list}, format should be like '2' or '0,1'")
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # new Experiment
    runner = ExperimentRunner(
        models=args.models,
        data_type=args.data_type,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        parameter_config_path=args.parameter_config_path,
        datasets=args.datasets,
        rla_list=args.rla_list,
        eln_list=args.eln_list,
        ru_list=args.ru_list,
        flip_nr_list=args.flip_nr_list,
        flip_ar_list=args.flip_ar_list,
        target_for_unlabeled=args.target_for_unlabeled,
        noise_type=args.noise_type,
        is_cleanlab=args.is_cleanlab,
        seed_list=args.seed_list,
        gpu_list=gpu_list,
        DEBUG=args.DEBUG,
        NO_RESUME=args.NO_RESUME,
        exp_note=args.exp_note
    )

    logger.info(f"start running {args.data_type} experiment, model: {args.models}")
    start_time = time.time()

    results = runner.run_experiments()

    total_time = time.time() - start_time
    logger.info(f"All experiments completed, with the total time consumed: {total_time:.2f}s")


if __name__ == "__main__":
    main()
