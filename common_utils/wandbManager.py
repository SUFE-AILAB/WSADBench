import os
import yaml
import wandb
import copy
import torch
from pathlib import Path


class WandbManager:
    def __init__(self, config_path: str, args, exclude_keys=None):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        exclude_keys = exclude_keys or []

        os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]
        wandb.login()

        args_copy = copy.deepcopy(args)
        self.wandb_config = {k: self._convert_value(v) for k, v in vars(args_copy).items() if k not in exclude_keys}
        self.project = config["project"]
        self.entity = config["entity"]

        self.run = wandb.init(project=self.project, entity=self.entity, config=self.wandb_config)

    def reinit(self):
        self.run.finish()
        self.run = wandb.init(project=self.project, entity=self.entity, config=self.wandb_config)

    def _convert_value(self, value):
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        else:
            return value

    def watch(self, model, criterion=None, log="all", log_freq=100):
        wandb.watch(model, criterion=criterion, log=log, log_freq=log_freq)

    def log(self, log_dict: dict, step: int = None):
        """
        记录训练日志，格式：
        {
            'train': {'loss': 0.1, 'aucpr': 0.88},
            'val': {'loss': 0.12},
            'test': {'loss': 0.11}
        }
        """
        flat_dict = {}
        for phase, metrics in log_dict.items():
            for key, val in metrics.items():
                flat_dict[f"{phase}/{key}"] = val
        if step is not None:
            flat_dict["epoch"] = step
        wandb.log(flat_dict)

    def save_model(self, model, name="model.pth", save_dir="wandb_models/"):
        """
        保存模型权重，并上传为 artifact
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, name)
        torch.save(model.state_dict(), save_path)
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(save_path)
        self.run.log_artifact(artifact)

    def log_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        记录混淆矩阵图
        """
        if labels is None:
            labels = list(set(y_true) | set(y_pred))
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=y_true, preds=y_pred, class_names=labels
                )
            }
        )

    def log_table(self, name, columns, data):
        """
        上传表格，如结果展示、样本分布等
        """
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})

    def log_image(self, name, images, captions=None):
        """
        记录图像数据，适用于可视化 reconstruction、attention heatmap 等
        """
        wandb_images = [wandb.Image(img, caption=(captions[i] if captions else None)) for i, img in enumerate(images)]
        wandb.log({name: wandb_images})
