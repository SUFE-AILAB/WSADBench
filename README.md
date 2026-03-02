# WSADBench

**Rethinking Weak Supervision in Anomaly Detection: A Comprehensive Benchmark**

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

WSADBench is a comprehensive benchmark for weakly-supervised anomaly detection, supporting multiple data modalities including tabular data (classical, CV features, NLP embeddings), video data, and inexact supervision (MIL bags).

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Data Preparation](#-data-preparation)
- [Supported Models](#-supported-models)
- [Project Structure](#-project-structure)
- [Advanced Usage](#-advanced-usage)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Key Features

- **Multi-Modal Support**: Tabular (classical, CV features, NLP embeddings), Video, and MIL bags
- **30+ Baseline Models**: Weak supervision, semi-supervised, and unsupervised methods
- **Flexible Supervision Settings**: Configurable labeled anomaly ratios (RLA), labeled normal ratios (ELN), unlabeled ratios, and label noise
- **Parallel Execution**: Multi-GPU support with automatic GPU assignment
- **Reproducible Experiments**: Built-in result logging, resume capability, and statistical reporting

---

## 🛠️ Prerequisites

- Python 3.9
- CUDA 11.5+ (for GPU support)



## 🚀 Quick Start

Get `WSADBench`  quickly with this step-by-step guide.

### 1. Installation & Environment

Clone the repository and set up the environment with one block of commands:

```
# 1. Clone the specific branch and enter directory
git clone https://github.com/SUFE-AILAB/WSADBench.git
cd WSADBench
# 2. Create and activate conda environment (Python 3.9)
conda create --name wsad_env python=3.9.21 -y
conda activate wsad_env
# 3. Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Sample Data

Download two lightweight tabular datasets (`musk` and `satellite`) from the mirror to verify the installation.

```
mkdir -p WSADBench/datasets/Classical
wget -P WSADBench/datasets/Classical/ https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/master/Classical/25_musk.npz
wget -P WSADBench/datasets/Classical/ https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/master/Classical/30_satellite.npz
```

### 3. Run Demo Experiment

Run a simple experiment using the **DevNet** model on the downloaded tabular datasets.

```
python run_experiment.py --data_type tabular_classical --models DevNet --seed_list 102
```

### 4. Expected Output

If the experiment runs successfully, you will see the results printed in the console and saved to `results/`.

**Console Log / Result File Content:**

> File location: `WSADBench/results/tabular_classical/detail/DevNet/DevNet_results.jsonl`

```
{"model":"DevNet","dataset":"25_musk","rla":1.0,"eln":0.0,"ru":1.0,"flip_normal_ratio":0.0,"flip_abnormal_ratio":0.0,"target_for_unlabeled":"fill_unlabel_0","seed":102,"aucroc":1.0,"aucpr":1.0,"noise_type":null,"is_cleanlab":"false","fit_time":12.0407865047,"inference_time":0.0013232231,"n_train":2143,"n_test":919,"n_train_anomalies":68,"n_test_anomalies":29,"error":"","data_type":"tabular_classical","exp_note":"None"}
{"model":"DevNet","dataset":"30_satellite","rla":1.0,"eln":0.0,"ru":1.0,"flip_normal_ratio":0.0,"flip_abnormal_ratio":0.0,"target_for_unlabeled":"fill_unlabel_0","seed":102,"aucroc":0.8361974905,"aucpr":0.8213418463,"noise_type":null,"is_cleanlab":"false","fit_time":9.7461748123,"inference_time":0.0014472008,"n_train":4504,"n_test":1931,"n_train_anomalies":1425,"n_test_anomalies":611,"error":"","data_type":"tabular_classical","exp_note":"None"}
```



## 💾 Data Preparation

Our benchmark datasets are collected and integrated from two primary sources. Please download them to reproduce the experiments:

* **[ADBench Datasets](https://jihulab.com/BraudoCC/ADBench_datasets):** A portion of our data (tabular, image, and text features) is integrated from [ADBench](https://github.com/Minqi824/ADBench), a comprehensive benchmark for unsupervised and supervised outlier detection. 

* **[WSADBench Official Datasets](https://modelscope.cn/datasets/mac4mac/WSADBench-Datasets/files):** The remaining datasets, including Video Anomaly Detection (VAD), Out-of-Distribution (OOD) scenarios, and classical tabular MIL bags, are exclusively provided and hosted on our ModelScope repository.

To make dataset preparation seamless and space-efficient, we provide a unified Python script (`download_dataset.py`). This smart script automatically handles:

- **ADBench Datasets (from JihuLab):** Directly pulls the ready-to-use `.npz` files via HTTP without requiring Git LFS.

- **WSADBench Official Datasets (from ModelScope):** Downloads, verifies split chunks, extracts, and immediately deletes the original `.tar` archives to save your disk space.

```
# Download all WSADBench datasets and all ADBench datasets at once.
python WSADBench/datasets/download_dataset.py --datasets WSAD ADBench

# Download all VAD datasets extracted by a specific pretrained model (e.g., all 4 datasets for MViT_32).
python WSADBench/datasets/download_dataset.py --datasets CV_by_MViT_32

# Download only one specific dataset for a specific pretrained model (e.g., only shanghaitech for MViT_32).
python WSADBench/datasets/download_dataset.py --datasets CV_by_MViT_32/shanghaitech

# Download specific tabular benchmarks (e.g., Classical, CV_by_ResNet18) directly from the ADBench repository.
python WSADBench/datasets/download_dataset.py --datasets Classical
```





## 🧩 Extra operation for some models

some model have conflict in the main pip environment, need to be run in the extra pip environment.

some model need ckpts

### TabR-S

```
# 2. Create and activate conda environment (Python 3.9)
conda create --name wsad_tabr python=3.9.21 -y
conda activate wsad_tabr
# 3. Install dependencies 
pip install -r requirements/req_tabr.txt

# run TabR-S
python run_experiment.py --data_type tabular_classical --models TabR_S --seed_list 102
```



### LimiX

```
# new ckpt folder in main folder, if ckpt folder not exist
mkdir ckpt
# pull ckpt
wget "https://modelscope.cn/api/v1/datasets/mac4mac/WSADBench-Datasets/repo?Revision=master&FilePath=ckpt/LimiX-16M.ckpt" -O ckpt/LimiX-16M.ckpt

# 2. Create and activate conda environment (Python 3.9)
conda create --name wsad_limix python=3.12.7 -y
conda activate wsad_limix

# 3. Install dependencies (using Tsinghua mirror for speed)
# get the wheel. if too slow, use: wget "https://modelscope.cn/api/v1/datasets/mac4mac/WSADBench-Datasets/repo?Revision=master&FilePath=env/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" -O flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
wget -O flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install scikit-learn  einops  huggingface-hub matplotlib networkx numpy pandas  scipy tqdm typing_extensions xgboost kditransform hyperopt copulas cleanlab

# remove wheel file
rm flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# run LimiX
python run_experiment.py --data_type tabular_classical --models LimiX --seed_list 102
```



### PyOD

```
# Create and activate conda environment (Python 3.9)
conda create --name wsad_pyod python=3.9.21 -y
conda activate wsad_pyod

pip install -r requirements/req_pyod.txt

# run pyod (unsupervised)
 python run_experiment.py --data_type tabular_classical --models  IForest AutoEncoder --seed_list 102
```



### TabPFN

```
# this model can be run in env wsad_env, the main env for WSADBenchmark.
# new ckpt folder in main folder, if ckpt folder not exist
mkdir ckpt
# pull ckpt
wget "https://modelscope.cn/api/v1/datasets/mac4mac/WSADBench-Datasets/repo?Revision=master&FilePath=ckpt/tabpfn-v2.5-classifier-v2.5_default.ckpt" -O ckpt/tabpfn-v2.5-classifier-v2.5_default.ckpt
```







## 🔬 Reproduce Different Setting

### Anomaly Detection (Tabular, CV, NLP and VAD) and Multiple Instance Learning (MIL) Paradigm 

See Section 4.1 "Basic WSAD Experiments" and Section 4.2.5 "Can Methods Transfer Across Supervision Types?" for details

```Shell
# --- 1. Classical Tabular Datasets ---
# Evaluate a single model on all classical tabular datasets
python -m run_experiment --data_type tabular_classical --models DevNet

# --- 2. Computer Vision (CV) Datasets ---
# Evaluate a model on CV datasets using ResNet18 extracted features
python -m run_experiment --data_type tabular_CV_by_ResNet18 --models DevNet

# Evaluate a model on CV datasets using Vision Transformer (ViT) extracted features
python -m run_experiment --data_type tabular_CV_by_ViT --models DevNet

# --- 3. Natural Language Processing (NLP) Datasets ---
# Evaluate a model on NLP datasets using BERT extracted features
python -m run_experiment --data_type tabular_NLP_by_BERT --models DevNet

# Evaluate a model on NLP datasets using RoBERTa extracted features
python -m run_experiment --data_type tabular_NLP_by_RoBERTa --models DevNet

# --- 4. Dataset-Specific Execution ---
# Execute a model on a specific target dataset (applicable to any data_type above)
python -m run_experiment --data_type tabular_classical --models DevNet --dataset 10_cover


# Multiple Instance Learning (MIL) Paradigm
# This paradigm evaluates models under Inexact Supervision, where labels are provided at the "bag" level rather than for individual instances. Our benchmark supports MIL execution for classical tabular bags.
python -m run_experiment --data_type classical_bags_inexact --models Sultani DevNet 

# VAD
# A. Single Model Run: Evaluate one model on a specific dataset using fixed segmentation and features.
python -m run_experiment --data_type video --models DevNet  --dataset TAD seg_32_pm_mvit

# B. Batch Execution: Evaluate multiple baselines across all datasets, segmentation scales, and pre-trained features.
python -m run_experiment --data_type video --models Sultani ARNet --dataset TAD shanghaitech UCF_Crime XD-violence  seg_32_200_pm_mvit_sf_i3d_sf50_x3d  
```

### Foundation Models in Anomaly Detection

See Section 4.2.1 Foundation Models for details.

```Shell
# --- 1. TabPFN ---
python -m run_experiment --data_type tabular_classical --models TabPFN

# --- 2. LimiX ---
# WARNING: LimiX requires a specific Python environment (e.g., Python 3.9+) due to 
# package conflicts with other baselines. We use a dedicated Conda environment.
python -m run_experiment --data_type tabular_classical --models LimiX
```

### Sensitivity Analysis: Incomplete and Inaccurate Supervision

This section evaluates model robustness under varying degrees of supervision completeness and label quality, corresponding to Section 4.2.2 (The Value of Unlabeled Data) and Section 4.2.3 (Sensitivity to Label Noise).

```Shell
# The Value of Unlabeled Data (Incomplete Supervision)
# --- 1. Varying Labeled Anomaly Ratio (RLA) ---
# Evaluate DevNet on classical tabular data with labeled anomaly ratios ranging from 1% to 100%.
python -m run_experiment --data_type tabular_classical --models DevNet --rla_list 0.01 0.05 0.1 0.25 0.5 1.0 

# Evaluate DeepSAD on CV (ViT) features with a specific list of labeled sample counts/ratios.(nla)
python -m run_experiment --data_type tabular_CV_by_ViT --models DeepSAD --rla_list 1 3 5 10 15 20 50 

# --- 2. Varying Unlabeled Data Ratio (RU) ---
# Evaluate REPEN on NLP (RoBERTa) features, varying both labeled anomalies and unlabeled data size.
# This tests the model's ability to leverage unlabeled data (See Section 4.2.2).
python -m run_experiment --data_type tabular_NLP_by_RoBERTa --models REPEN --rla_list 1 10 20 50 --ru_list 20 50 200 1000


# Sensitivity to Label Noise (Inaccurate Supervision)
# --- 3. Noise in Normal Labels (False Positives) ---
# Simulate scenarios where normal samples are wrongly labeled as anomalies (flip_nr).
python -m run_experiment --data_type tabular_classical --models RoSAS --flip_nr_list 0.01 0.05 0.1 0.25 0.5 --noise_type label_contamination 

# --- 4. Noise in Anomaly Labels (False Negatives) ---
# Simulate scenarios where actual anomalies are wrongly labeled as normal (flip_ar).
python -m run_experiment --data_type tabular_classical --models RoSAS --flip_ar_list 0.01 0.05 0.1 0.25 0.5 --noise_type label_contamination 

# --- 5. Mixed Label Noise (Symmetric/Asymmetric) ---
# Evaluate robustness when both types of label errors exist simultaneously.
python -m run_experiment --data_type tabular_classical --models DevNet --flip_nr_list 0.01 0.05 0.1 0.25 0.5 --flip_ar_list 0.01 0.05 0.1 0.25 0.5 --noise_type label_contamination
```

### OOD

See Section 4.2.4 for details

```Shell
# Setting I (ID Far, OOD Near) 
python -m run_experiment  --data_type tabular_CV_by_ResNet18_OOD --models DevNet  --exp_note rla_emb_know_far_inc --dataset metal_nut
# Setting II (ID Near, OOD Far) 
python -m run_experiment  --data_type tabular_CV_by_ResNet18_OOD --models DevNet  --exp_note rla_emb_know_near_inc --dataset metal_nut  
# Setting III (ID Near, OOD Near)
python -m run_experiment  --data_type tabular_CV_by_ResNet18_OOD --models DevNet  --exp_note rla_emb_near_inc --dataset metal_nut 
# --- Semantic-level OOD (See Appendix for details) ---
# Semantic-Class OOD: Evaluate generalization to unseen anomaly categories without explicit distance constraints.
python -m run_experiment  --data_type tabular_CV_by_ResNet18_OOD --models DevNet  --exp_note rla_inc --dataset metal_nut 

# --- Comprehensive Batch Run ---# Execute multiple models across all OOD scenarios, rla rates and available datasets.
python -m run_experiment  --data_type tabular_CV_by_ResNet18_OOD --models DevNet CatB  --exp_note rla_emb_near_inc rla_emb_know_near_inc rla_emb_know_far_inc rla_inc --dataset carpet metal_nut aitex hyperkvasir  elpv mastcam --rla_list 0.1 0.5 1.0
```









## 📊Supported Data Types

| Data Type | CLI Flag | Description |
|-----------|----------|-------------|
| Classical Tabular | `tabular_classical` | Traditional AD benchmarks (47 datasets) |
| CV Features (ResNet18) | `tabular_CV_by_ResNet18` | Image features extracted by ResNet18 |
| CV Features (ViT) | `tabular_CV_by_ViT` | Image features extracted by ViT |
| NLP Features (BERT) | `tabular_NLP_by_BERT` | Text embeddings from BERT |
| NLP Features (RoBERTa) | `tabular_NLP_by_RoBERTa` | Text embeddings from RoBERTa |
| Video | `video` | Video anomaly detection (I3D features) |
| MIL Bags | `classical_bags_inexact` | Classical data in MIL bag format |

---

## 🤖 Supported Models

### Weakly-Supervised (Instance)

| Model | CLI Flag | Category | Description |
|-------|----------|-------------|-------------|
| DevNet | DevNet | Score Learning | Deviation networks for anomaly detection with limited supervision |
| DeepSAD | DeepSAD | Score Learning | Deep semi-supervised anomaly detection via one-class classification |
| PReNet | PReNet | Score Learning | Pairwise relation network for anomaly detection |
| REPEN | REPEN | Repr. Learning | Representation learning for PU learning |
| XGBOD | XGBOD | Repr. Learning | Feature augmentation for outlier detection |
| RoSAS | RoSAS | Data Aug. | Robust semi-supervised anomaly segmentation |
| Dual-MGAN | Dual-MGAN | Data Aug. | Dual-MGAN for anomaly detection |
| FEAWAD | FEAWAD | Reconstruction | Feature encoding with autoencoders for weakly-supervised AD |
| DDAE | AnoDDAE | Diffusion DAE | Anomaly detection with denoising diffusion autoencoders |
| SOEL-NTL | NTL | Pseudo-Labeling | Self-training with outlier exposure |
| AA-BiGAN | AABiGAN | GAN-based | Adversarially learned anomaly detection with BiGAN |
| GANomaly | GANomaly | GAN-based | GAN-based anomaly detection |

### Unsupervised (Instance)

| Model | CLI Flag | Category | Description |
|-------|----------|-------------|-------------|
| IForest | IForest | Isolation-based | Isolation Forest - classical baseline |
| AutoEncoder | AutoEncoder | Reconstruction | Autoencoder reconstruction error |
| VAE | VAE | Reconstruction | Variational Autoencoder |
| PCA | PCA | Reconstruction | Principal Component Analysis |
| DeepSVDD | DeepSVDD | Deep One-class | Deep Support Vector Data Description |
| ECOD | ECOD | Probabilistic | Empirical Cumulative Distribution |
| CBLOF | CBLOF | Cluster-based | Cluster-based Local Outlier Factor |
| LOF | LOF | Density-based | Local Outlier Factor |
| LUNAR | LUNAR | GNN-based | Graph neural network for anomaly detection |

### Weakly-Supervised (Bag)

| Model | CLI Flag | Category | Description |
|-------|----------|-------------|-------------|
| Sultani | Sultani | Vanilla MIL | MIL-based weakly supervised video anomaly detection |
| RTFM | RTFM | Magnitude MIL | Robust temporal feature magnitude |
| MGFN | MGFN | Magnitude MIL | Multi-graph fusion network |
| AR-Net | ARNet | Dynamic MIL | Dynamic MIL for video anomaly detection |
| VadCLIP | VadClip | Language-Guided MIL | Vision-language video anomaly detection |
| UR-DMU | URDMU | Uncertainty-Aware MIL | Unified representation for detection of multiple anomalies |
| GCN-Anomaly | ZhongGCNAD | Label Denoising | Graph convolutional network for anomaly detection |
| PUMA | PUMA | PU MIL | PU-learning based multi-model anomaly detection |

### Supervised (Instance)

| Model | CLI Flag | Category | Description |
|-------|----------|-------------|-------------|
| XGBoost | XGB | GBDT | Gradient boosting decision trees |
| CatBoost | CatB | GBDT | Categorical boosting |
| FTTransformer | FTTransformer | Deep (Sup.) | Feature-wise transformer for tabular data |
| TabM | TabMCls | Deep (Sup.) | Tabular deep learning model |
| TabR-S | TabR_S | Deep (Sup.) | Tabular regression with scaled embeddings |

### Foundation Models (Instance)

| Model | CLI Flag | Category | Description |
|-------|----------|-------------|-------------|
| TabPFN | TabPFN | Found. Model | Descriminative Foundation Model |
| LimiX | LimiX | Found. Model | Generative Foundation Model |

---

## 📁 Project Structure

```
WSADBench/
├── common_utils/                            # Shared utilities
│   ├── argTypes.py                          # Argument type parsing
│   └── baseline_utils.py                    # Video-specific utilities
├── requirements/                            # Model-specific environment configurations
│   ├── req_pyod.txt                         # Dependencies for PyOD models
│   └── req_tabr.txt                         # Dependencies for TabR model
├── WSADBench/                               # Core package
│   ├── baseline/                            # Model implementations
│   │   ├── AABiGAN/                         # AABiGAN implementation
│   │   ├── AnoDDAE/                         # AnoDDAE implementation
│   │   ├── ARNet/                           # ARNet implementation
│   │   ...                                  # 30+ other models
│   │   ├── PyOD.py                          # PyOD wrapper (20+ models)
│   │   └── Supervised.py                    # Supervised learning baselines
│   ├── datasets/                            # Dataset handling
│   │   ├── dataset_configs/                 # Dataset configuration (YAML)
│   │   ├── dataset_support/                 # Video preprocessing utilities
│   │   ├── cv_data_generator.py             # CV dataset handling
│   │   ├── data_generator.py                # Data generation & loading
│   │   └── download_dataset.py              # Automated dataset download script
│   ├── model_configs/                       # Model hyperparameters (YAML)
│   │   ├── tabular/                         # Tabular model configs
│   │   └── video/                           # Video model configs
│   ├── build_bags.py                        # Instance → MIL bag conversion
│   └── myutils.py                           # Utility functions
├── .gitattributes                           # Git attribute settings
├── .gitignore                               # Ignored files and directories
├── DATASETS.md                              # Dataset preparation guide
├── LICENSE                                  # MIT License
├── README.md                                # This file
├── requirements.txt                         # Main Python dependencies
└── run_experiment.py                        # Main entry point
```

---

## ⚙️ Advanced Usage

### Key CLI Arguments

| **Argument**              | **Description**                                    | **Default**                           | **Possible Values / Choices**                                |
| ------------------------- | -------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| `--data_type`             | Type of data / Modality (Required)                 | `tabular_classical`                   | `video`, `tabular_classical`, `tabular_CV_by_ResNet18`, `tabular_CV_by_ViT`, `tabular_NLP_by_BERT`, `tabular_NLP_by_RoBERTa`, `classical_bags_inexact`, `tabular_CV_by_ResNet18_OOD` |
| `--models`                | List of model names to run                         | -                                     | Any model names (e.g., `DeepSAD`, `DevNet`)                  |
| `--datasets`              | Specify the names of the datasets to run           | All available (`None`)                | dataset names (e.g., `25_musk`, `30_satellite`)              |
| `--rla_list`              | List of ratios for labeled anomalies               | `[1.0]`                               | Float list (e.g., `0.01 0.1 1.0`)                            |
| `--eln_list`              | List of ratios for expected labeled normal samples | `[0.0]`                               | Float list (e.g., `0.0 0.5 1.0`)                             |
| `--ru_list`               | List of ratios for unlabeled samples               | `[1.0]`                               | Float list (e.g., `0.1 0.5 1.0`)                             |
| `--flip_nr_list`          | Error labeling ratios for normal samples (FPR)     | `[0.0]`                               | Float list (e.g., `0.0 0.05 0.1`)                            |
| `--flip_ar_list`          | Error labeling ratios for abnormal samples (FNR)   | `[0.0]`                               | Float list (e.g., `0.0 0.05 0.1`)                            |
| `--target_for_unlabeled`  | Target handling method for unlabeled data          | `["fill_unlabel_0"]`                  | `fill_unlabel_0`, `keep_label`, `delete_sample`              |
| `--noise_type`            | Noise type for experiments                         | `[None]`                              | `None`, `label_contamination`                                |
| `--is_cleanlab`           | Switch parameter to enable data cleaning           | `["false"]`                           | `true`, `false`                                              |
| `--seed_list`             | Random seed list                                   | `[0, 1, 2, 3, 4]`                     | List of Integers (e.g., `42`, ` 102`)                        |
| `--n_jobs`                | Number of parallel jobs                            | `1`                                   | Integer(e.g. `0`, `1`, `2`)                                  |
| `--gpus`                  | Specify GPUs                                       | `'auto'`                              | `'auto'`, or GPU IDs (e.g., `0`, `0,1`)                      |
| `--output_dir`            | Output directory                                   | `results/{data_type}`                 | Any valid directory path string                              |
| `--parameter_config_path` | Directory path for model param config files        | `WSADBench/model_configs/{data_type}` | Any valid directory path string                              |
| `--NO_RESUME`             | Forcibly re-run completed experiments              | `False`                               | Flag (Omit to disable, include to enable)                    |
| `--dry_summary`           | Only perform summarization, do not run             | `False`                               | Flag (Omit to disable, include to enable)                    |
| `--DEBUG`                 | Enable debug mode                                  | `False`                               | Flag (Omit to disable, include to enable)                    |
| `--exp_note`              | Experiment notes for tracking                      | `['None']`                            | Any string list (e.g., `test_run_1`, `test_vad`)             |



### Weak Supervision Settings Explained

WSADBench supports comprehensive weak supervision configurations:

- **RLA (Ratio of Labeled Anomalies)**: Proportion of anomalies that are labeled in training data
- **ELN (Ratio of Labeled Normal samples)**: Proportion of labeled normal samples relative to labeled anomalies
- **RU (Ratio of Unlabeled)**: Proportion of unlabeled samples in training data
- **Label Contamination**: Simulate annotation errors with `flip_nr_list` and `flip_ar_list`

```bash
# Example: 10% labeled anomalies, 50% unlabeled data, 5% label noise
python run_experiment.py \
    --data_type tabular_classical \
    --models DevNet \
    --rla_list 0.1 \
    --ru_list 0.5 \
    --flip_nr_list 0.05 \
    --flip_ar_list 0.05
```

### Custom Model Configuration

Model hyperparameters are stored in `WSADBench/model_configs/{data_type}/{model_name}.yaml`:

```yaml
# Example: WSADBench/model_configs/tabular/DeepSAD.yaml
model_class: "WSADBench.baseline.DeepSAD.run.DeepSAD"
parameters:
  latent_dim: 32
  hidden_dims: [64, 32]
  epochs: 100
  batch_size: 256
  lr: 0.001
```

### Adding New Models

1. Create a new directory in `WSADBench/baseline/YourModel/`
2. Implement `run.py` with a class that has:
   - `__init__(self, seed, **kwargs)`: Initialize model
   - `fit(self, X, y, ...)`: Training method
   - `predict_score(self, X, ...)`: Return anomaly scores
3. Create config file `WSADBench/model_configs/{data_type}/YourModel.yaml`
4. Add model to `ModelRegistry` in `run_experiment.py`

### Output Format

Results are saved in JSONL format:

```
results/
└── {data_type}/
    ├── detail/
        └── {model_name}/
            ├── {model_name}_results.jsonl  # Individual results
            └── model_stats.json            # Model statistics
```

<!-- ---

## 📝 Citation

If you use WSADBench in your research, please cite:

```bibtex
@article{wsadbench2025,
  title={Rethinking Weak Supervision in Anomaly Detection: A Comprehensive Benchmark},
  author={WSADBench Authors},
  journal={arXiv preprint},
  year={2025}
}
``` -->

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- [PyOD](https://github.com/yzhao062/pyod) - Python Outlier Detection library
- [ADBench](https://github.com/Minqi824/ADBench) - Anomaly Detection Benchmark

---

## 📫 Contact

For questions and issues, please open an issue on GitHub.
