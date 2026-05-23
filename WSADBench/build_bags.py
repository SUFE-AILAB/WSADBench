#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
目录模式：遍历 input-dir 下的所有 .npz 文件，逐一处理：
- 输入 .npz（含 X, y）
- 构袋
- 输出 .npz 到指定输出根目录（结构镜像）
- 支持 --no-resume：输出存在则跳过
- 支持 --gpus：控制可见 GPU（如 0 或 0,1,2；cpu/-1 为禁用 GPU）

示例：
python your_script.py \
  --input-dir ./classical \
  --output-dir ./classical_bags \
  --bag-size 10 --bag-prob 0.3 --seed 331 \
  --no-resume \
  --gpus 0,1
"""

import argparse
import json
import os
from collections import Counter
import numpy as np


def create_bags(X, y, g=10, bagprob=0.3, seed=331):
    np.random.seed(seed)
    anom_idx = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    n_anom = len(anom_idx)
    n_norm = len(norm_idx)

    bags, instance_labels = [], []
    m = np.shape(X)[1]

    while n_anom > 0 or n_norm > 0:
        p = np.random.uniform(0, 1, 1)[0]
        if p <= bagprob and n_anom > 0:
            hi = max(2, g // 2)
            n_anom_instances = np.random.randint(1, hi, 1)[0]
            if n_anom_instances > n_anom:
                n_anom_instances = n_anom
            if g - n_anom_instances > n_norm:
                break
            anom_instances = np.random.choice(anom_idx, n_anom_instances, replace=False)
            norm_instances = np.random.choice(norm_idx, g - n_anom_instances, replace=False)
            bags.append(np.concatenate((X[norm_instances, :], X[anom_instances, :])))
            instance_labels.append(np.concatenate((y[norm_instances], y[anom_instances])))
            n_anom -= n_anom_instances
            n_norm -= g - n_anom_instances
            anom_idx = anom_idx[~np.isin(anom_idx, anom_instances)]
            norm_idx = norm_idx[~np.isin(norm_idx, norm_instances)]
        else:
            if g > n_norm:
                break
            norm_instances = np.random.choice(norm_idx, g, replace=False)
            bags.append(X[norm_instances, :])
            instance_labels.append(y[norm_instances])
            n_norm -= g
            norm_idx = norm_idx[~np.isin(norm_idx, norm_instances)]

    if len(bags) == 0:
        raise RuntimeError("无法构造任何袋；请检查参数与数据")

    bags = np.array(bags).reshape(-1, g, m)
    instance_labels = np.array(instance_labels).reshape(-1, g)
    bags_labels = np.sum(instance_labels, axis=1)
    bags_labels[bags_labels > 0] = 1
    bags_labels = bags_labels.astype(int)
    y_inst = instance_labels.reshape(-1).astype(int)
    print("Bag summary:", Counter(bags_labels))
    return bags, bags_labels, y_inst


def _load_npz(path: str):
    data = np.load(path, allow_pickle=False)
    if 'X' not in data or 'y' not in data:
        raise KeyError("输入 .npz 必须包含 X 和 y")
    return data['X'], data['y'].astype(int)


def _save_npz(path: str, bags, bags_labels, y_inst, meta):
    meta_json = json.dumps(meta, ensure_ascii=False)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path,
                        X=bags,
                        y=bags_labels,
                        y_gt=y_inst,
                        meta=np.asarray(meta_json))


def parse_args():
    p = argparse.ArgumentParser(description="二维 tabular .npz → MIL 袋数据 .npz（仅目录遍历模式）")
    p.add_argument('--input-dir', required=True, help='输入文件夹，遍历其中的 .npz 文件')
    p.add_argument('--output-dir', help='输出文件夹（未指定则默认在输入目录旁创建 <dir>_bags）')
    p.add_argument('--bag-size', type=int, default=10, help='每袋实例数 g，默认 10')
    p.add_argument('--bag-prob', type=float, default=0.3, help='正袋概率，默认 0.3')
    p.add_argument('--seed', type=int, default=331, help='随机种子，默认 331')
    p.add_argument('--limit', type=int, default=None, help='可选：仅取前 limit 条样本')
    p.add_argument('--no-resume', action='store_true',
                   help='若输出文件已存在则跳过处理（默认覆盖重处理）')
    # 新增 GPU 选择
    p.add_argument('--gpus', type=str, default=None,
                   help="设置可见 GPU，如 '0' 或 '0,1,2'；传 'cpu' 或 '-1' 则禁用 GPU")
    return p.parse_args()


def _process_one(in_path: str, out_path: str, args) -> None:
    # no-resume: 如果输出已存在且要求跳过，则直接返回
    if args.no_resume and os.path.exists(out_path):
        print(f"[SKIP] 目标已存在且指定 --no-resume：{out_path}")
        return

    print(f"[INFO] Load: {in_path}")
    X, y = _load_npz(in_path)

    if args.limit is not None:
        n = min(int(args.limit), X.shape[0])
        X, y = X[:n], y[:n]
        print(f"[INFO] limit={n}")

    # 与原始逻辑一致：不 shuffle
    print(f"[INFO] params: g={args.bag_size}, bagprob={args.bag_prob}, seed={args.seed} (no-shuffle)")

    bags, bags_labels, y_inst = create_bags(X, y, g=args.bag_size, bagprob=args.bag_prob, seed=args.seed)

    meta = {
        'input': in_path,
        'g': int(args.bag_size),
        'bagprob': float(args.bag_prob),
        'seed': int(args.seed),
        'n_bags': int(bags.shape[0]),
        'feature_dim': int(bags.shape[2]),
        'class_counts_bag': {int(k): int(v) for k, v in Counter(bags_labels.tolist()).items()},
        'class_counts_inst': {
            'pos': int(y_inst.sum()),
            'neg': int(y_inst.size - y_inst.sum()),
        },
        'limit': None if args.limit is None else int(args.limit),
        'no_resume': bool(args.no_resume),
        'gpus': os.environ.get("CUDA_VISIBLE_DEVICES", None),
    }

    _save_npz(out_path, bags, bags_labels, y_inst, meta)
    print(f"[INFO] Saved: {out_path}")
    print(f"[INFO] bags: {bags.shape}; bags_labels: {bags_labels.shape}; y_inst: {y_inst.shape}")


def main():
    args = parse_args()

    # 先配置 GPU 可见性
    if args.gpus is not None:
        if args.gpus.strip().lower() in ("-1", "cpu"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("[INFO] GPUs: disabled (CPU only)")
        else:
            # 仅做简单合法性清洗（去空格）
            gpus = ",".join([tok.strip() for tok in args.gpus.split(",") if tok.strip() != ""])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            print(f"[INFO] GPUs visible: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        # 不设置时，沿用系统默认
        print("[INFO] GPUs: use system default visibility")

    # 目录遍历模式
    in_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(in_dir):
        raise NotADirectoryError(f"输入目录不存在：{in_dir}")

    out_root = args.output_dir
    if out_root is None:
        out_root = in_dir.rstrip(os.sep) + '_bags'
    os.makedirs(out_root, exist_ok=True)

    n_total = 0
    n_ok = 0
    n_fail = 0
    n_skip = 0

    for root, _, files in os.walk(in_dir):
        for fname in files:
            if not fname.lower().endswith('.npz'):
                continue
            n_total += 1
            in_path = os.path.join(root, fname)
            # 在输出根目录下复刻相对子路径
            rel = os.path.relpath(in_path, in_dir)
            rel_dir = os.path.dirname(rel)
            os.makedirs(os.path.join(out_root, rel_dir), exist_ok=True)
            base, _ = os.path.splitext(os.path.basename(rel))
            out_path = os.path.join(out_root, rel_dir, base + '_bags.npz')

            # 目录模式下也先做 no-resume 判断（可快速跳过）
            if args.no_resume and os.path.exists(out_path):
                print(f"[SKIP] 目标已存在且指定 --no-resume：{out_path}")
                n_skip += 1
                continue

            try:
                _process_one(in_path, out_path, args)
                n_ok += 1
            except Exception as e:
                print(f"[ERROR] {in_path}: {e}")
                n_fail += 1
                continue

    print("[SUMMARY] total=", n_total, "ok=", n_ok, "skip=", n_skip, "fail=", n_fail)


if __name__ == '__main__':
    main()

#测试代码，处理classical目录下的文件
#python WSADBench/build_bags.py --input-dir WSADBench/datasets/Classical --output-dir WSADBench/datasets/Classical_bags_inexact_prob01 --bag-size 10 --bag-prob 0.3 --seed 331 --no-resume --gpus 0
#python WSADBench/build_bags.py --input-dir WSADBench/datasets/CV_by_ResNet18 --output-dir WSADBench/datasets/CV_by_ResNet18_bags_inexact --bag-size 10 --bag-prob 0.3 --seed 331 --no-resume --gpus 0

# 对照实验1
# python WSADBench/build_bags.py --input-dir WSADBench/datasets/Classical --output-dir WSADBench/datasets/Classical_bags_inexact_prob01 --bag-size 10 --bag-prob 0.1 --seed 331 --no-resume --gpus 0

# 对照实验2
# python WSADBench/build_bags.py --input-dir WSADBench/datasets/Classical --output-dir WSADBench/datasets/Classical_bags_inexact_bag20 --bag-size 20 --bag-prob 0.3 --seed 331 --no-resume --gpus 0

# 对照实验3
# python WSADBench/build_bags.py --input-dir WSADBench/datasets/Classical --output-dir WSADBench/datasets/Classical_bags_inexact_bag30 --bag-size 30 --bag-prob 0.3 --seed 331 --no-resume --gpus 0

# 对照实验4 不用
# python WSADBench/build_bags.py --input-dir WSADBench/datasets/Classical --output-dir WSADBench/datasets/Classical_bags_inexact_bag20prob01 --bag-size 20 --bag-prob 0.1 --seed 331 --no-resume --gpus 0

# 对照实验5 不用
# python WSADBench/build_bags.py --input-dir WSADBench/datasets/Classical --output-dir WSADBench/datasets/Classical_bags_inexact_bag30prob01 --bag-size 30 --bag-prob 0.1 --seed 331 --no-resume --gpus 1

# 对照实验6
# python WSADBench/build_bags.py --input-dir WSADBench/datasets/Classical --output-dir WSADBench/datasets/Classical_bags_inexact_bag10prob02 --bag-size 10 --bag-prob 0.2 --seed 331 --no-resume --gpus 1