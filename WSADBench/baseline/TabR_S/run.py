# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import yaml
from dataclasses import asdict
# >>>
if __name__ == '__main__':

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
import WSADBench.baseline.TabR_S.lib as lib
from WSADBench.baseline.TabR_S.lib import KWArgs, TaskType

@dataclass(frozen=False)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]


@torch.inference_mode()
def evaluate(C, train_indices, device, model, X, Y, eval_batch_size: int):
    model.eval()
    predictions = {}
    # 1. 替换数据集大小获取方式（移除 dataset.size(part)）
    part_sizes = {
        # 'train': len(X_train),
        'test': len(X)
    }
    parts = ['test']
    for part in parts:
        # 检查输入的part是否合法
        if part not in part_sizes:
            raise ValueError(f"Invalid part: {part}, must be 'train' or 'test'")

        while eval_batch_size:
            try:
                # 使用预定义的part_sizes获取数据大小
                predictions[part] = (
                    torch.cat(
                        [
                            apply_model(device ,C, model, train_indices, X, Y, part, idx, False)
                            for idx in torch.arange(
                            part_sizes[part], device=device
                        ).split(eval_batch_size)
                        ]
                    )
                    .cpu()
                    .numpy()
                )
            except RuntimeError as err:
                if not lib.is_oom_exception(err):
                    raise
                eval_batch_size //= 2
                logger.warning(f'eval_batch_size = {eval_batch_size}')
            else:
                break
        if not eval_batch_size:
            RuntimeError('Not enough memory even for eval_batch_size=1')

    return predictions, eval_batch_size

# 子函数
def zero_wd_condition(
    module_name: str,
    module: nn.Module,
    parameter_name: str,
    parameter: nn.parameter.Parameter,
):
    return (
        'label_encoder' in module_name
        or 'label_encoder' in parameter_name
        or lib.default_zero_weight_decay_condition(
            module_name, module, parameter_name, parameter
        )
    )

def get_Xy(device, X, Y, part: str, idx) -> tuple[dict[str, Tensor], Tensor]:

    x_tensor = torch.from_numpy(X).float().to(device)
    if idx is not None:
        x_tensor = x_tensor[idx]
    x_dict = {'num': x_tensor}
    # print(f'sum y1:{torch.sum(Y).item()}')
    # 处理标签
    if part == 'train':
        y_tensor = Y.to(torch.long)
        # print(f'sum y2:{torch.sum(Y).item()}')
        if idx is not None:
            y_tensor = y_tensor[idx]

        return x_dict, y_tensor
    return x_dict  # 可以缺失Y

def apply_model(device, C, model, train_indices, X,Y,part: str, idx: Tensor, training: bool):
    if part == 'train':
        x, y = get_Xy(device, X,Y,part, idx)  # X是float，Y是int
    else:
        x = get_Xy(device, X,None,part, idx)  # 测试的时候，不要Y输入
    candidate_indices = train_indices
    is_train = part == 'train'
    if is_train:
        # NOTE: here, the training batch is removed from the candidates.
        # It will be added back inside the model's forward pass.
        candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
    candidate_x, candidate_y = get_Xy(device, _global_X_train, _global_y_train,  # 这里是两个train
        'train',
        # This condition is here for historical reasons, it could be just
        # the unconditional `candidate_indices`.
        None if candidate_indices is train_indices else candidate_indices,
    )

    return model(
        x_=x,
        y=y if is_train else None,
        candidate_x_=candidate_x,
        candidate_y=candidate_y,
        context_size=C.context_size,
        is_train=is_train,
    ).squeeze(-1)


class Model(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E
        d_in = (
            n_num_features
            * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, lib.get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        *,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x_)
        if is_train:

            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if is_train else 0)
            )
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x


def main(
    output: Union[str, Path], *, force: bool = False, seed: int = 42
) -> Optional[lib.JSONDict]:
    output = Path(output)
    def load_config_from_yaml(filepath: str) -> Config:
        """从 YAML 文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return Config(**config_dict)

    C = load_config_from_yaml('WSADBench/baseline/TabR_S/bin/config.yaml')
    C.seed = seed
    C.data['seed'] = seed
    delu.random.seed(C.seed)
    device = lib.get_device()

    global _global_X_train, _global_y_train
    X_train = _global_X_train

    _global_y_train = torch.from_numpy(_global_y_train).to(torch.float).to(device)
    Y_train = _global_y_train
    model = Model(
        n_num_features=X_train.shape[-1],
        n_bin_features= 0,
        cat_cardinalities=[],  #  dataset.cat_cardinalities()
        n_classes=2,  # dataset.n_classes()
        **C.model,
    )
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # type: ignore[code]

    optimizer = lib.make_optimizer(
        model, **C.optimizer, zero_weight_decay_condition=zero_wd_condition
    )
    loss_fn = lib.get_loss_fn(TaskType.BINCLASS)

    train_size = len(X_train)
    train_indices = torch.arange(train_size, device=device)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None

    print()
    while epoch < C.n_epochs:  # 在这里训练
        print(f'[...] {lib.try_get_relative_path(output)} ')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: loss_fn(apply_model(device, C, model, train_indices, X_train, Y_train,'train', idx, True), Y_train[idx]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        print(f'epoch:{epoch}, loss:{mean_loss:.6f}')
        epoch += 1



    return C, train_indices, device, model, eval_batch_size
# 在脚本顶部定义全局变量
_global_X_train = None
_global_y_train = None


def run_tabr(X_train, y_train,seed):
    global _global_X_train, _global_y_train
    _global_X_train = X_train
    _global_y_train = y_train

    # 配置 libraries（可选）
    lib.configure_libraries()

    # 可选：动态覆盖部分参数（如 seed, batch_size 等）
    # config_dict['seed'] = 42
    # config_dict['batch_size'] = 256

    output_dir = Path('./tmp_tabr_output')
    output_dir.mkdir(exist_ok=True)

    # 直接调用 main，不通过 CLI
    result = main(output=output_dir, force=True, seed=seed)
    return result  # 应为 (C, train_indices, device, model, eval_batch_size)


class TabR_S:
    def __init__(self,seed,
                 model_name="TabR_S",
                 ):
        self.model = None
        self.seed = seed

    def fit(self, X_train, y_train,  verbose=True):
        self.C, self.train_indices, self.device, self.model, self.eval_batch_size = run_tabr(X_train, y_train, seed=self.seed)
        return None


    def predict_score(self, X):
        res = evaluate(self.C, self.train_indices, self.device, self.model, X,None,  self.eval_batch_size,)
        return res[0]['test']  # 返回正类的概率分数




