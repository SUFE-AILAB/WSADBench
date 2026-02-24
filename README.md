# WSADBench

**Rethinking Weak Supervision in Anomaly Detection: A Comprehensive Benchmark**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

WSADBench is a comprehensive benchmark for weakly-supervised anomaly detection, supporting multiple data modalities including tabular data (classical, CV features, NLP embeddings), video data, and inexact supervision (MIL bags).

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Preparation](#-data-preparation)
- [Supported Models](#-supported-models)
- [Project Structure](#-project-structure)
- [Advanced Usage](#-advanced-usage)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Key Features

- **Multi-Modal Support**: Tabular (classical, CV features, NLP embeddings), Video, and MIL bags
- **30+ Baseline Models**: Weak supervision, semi-supervised, and unsupervised methods
- **Flexible Supervision Settings**: Configurable labeled anomaly ratios (RLA), labeled normal ratios (ELN), unlabeled ratios, and label noise
- **Parallel Execution**: Multi-GPU support with automatic GPU assignment
- **Reproducible Experiments**: Built-in result logging, resume capability, and statistical reporting

---

## 📦 Installation

### Prerequisites

- Python 3.9
- CUDA 11.5+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/WSADBench.git
cd WSADBench

# Create conda environment
conda create -n wsad python=3.9 -y
conda activate wsad

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
pip install pytorchvideo opencv-python
```

Alternatively, use the provided setup script:

```bash
bash setup.sh
```

---

## 🏃 Quick Start

Get `WSADBench` with this step-by-step guide.

### 1. Installation & Environment

Clone the repository (using the `zsy_fix` branch) and set up the environment with one block of commands:

```
# 1. Clone the specific branch and enter directory
git clone -b zsy_fix https://github.com/SUFE-AILAB/WSADBench.git
cd WSADBench
# 2. Create and activate conda environment (Python 3.9)
conda create --name wsad_env python=3.9.21 -y
conda activate wsad_env
# 3. Install dependencies (using Tsinghua mirror for speed)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Prepare Sample Data

Download two lightweight tabular datasets (`musk` and `cardio`) from the mirror to verify the installation.

```
mkdir -p WSADBench/datasets/Classical
wget -P WSADBench/datasets/Classical/ https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/master/Classical/25_musk.npz
wget -P WSADBench/datasets/Classical/ https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/master/Classical/6_cardio.npz
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
{"model":"DevNet","dataset":"25_musk","rla":1.0,"eln":0.0,"ru":1.0,"flip_normal_ratio":0.0,"flip_abnormal_ratio":0.0,"target_for_unlabeled":"fill_unlabel_0","seed":102,"aucroc":1.0,"aucpr":1.0,"noise_type":null,"is_cleanlab":"false","fit_time":12.1113946438,"inference_time":0.0011568069,"n_train":2143,"n_test":919,"n_train_anomalies":68,"n_test_anomalies":29,"error":"","data_type":"tabular_classical","exp_note":"None"}
{"model":"DevNet","dataset":"6_cardio","rla":1.0,"eln":0.0,"ru":1.0,"flip_normal_ratio":0.0,"flip_abnormal_ratio":0.0,"target_for_unlabeled":"fill_unlabel_0","seed":102,"aucroc":0.9899016742,"aucpr":0.9362706439,"noise_type":null,"is_cleanlab":"false","fit_time":9.5926368237,"inference_time":0.0004396439,"n_train":1281,"n_test":550,"n_train_anomalies":123,"n_test_anomalies":53,"error":"","data_type":"tabular_classical","exp_note":"None"}
```



---



## 🏃🏃 Reproduce Different Setting

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

This section evaluates model robustness under varying degrees of supervision completeness and label quality, corresponding to **Section 4.2.2 (The Value of Unlabeled Data)** and **Section 4.2.3 (Sensitivity to Label Noise)**.

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







## 📊 Data Preparation

> **Note**: The complete benchmark datasets (including pre-extracted features for all modalities) will be released after the paper is accepted. For video datasets, we have unified the pretrained models used for feature extraction and re-extracted all features from the original videos to ensure consistency. The feature extraction code is available in this repository.

Datasets should be prepared as symbolic links in the `WSADBench/datasets/` directory. See **[DATASETS.md](DATASETS.md)** for detailed instructions on:

- Download links for all supported datasets
- Preprocessing instructions for each data type
- Directory structure requirements
- Feature extraction scripts (for CV/NLP features)

**Quick Setup:**
```bash
# After downloading datasets, create symlinks
ln -s /path/to/your/classical_datasets WSADBench/datasets/Classical
ln -s /path/to/your/video_features WSADBench/datasets/CV_by_I3D
ln -s /path/to/your/cv_features WSADBench/datasets/CV_by_ResNet18
```

### Supported Data Types

| Data Type | CLI Flag | Description |
|-----------|----------|-------------|
| Classical Tabular | `tabular_classical` | Traditional AD benchmarks (47 datasets) |
| CV Features (ResNet18) | `tabular_CV_by_ResNet18` | Image features extracted by ResNet18 |
| CV Features (ViT) | `tabular_CV_by_ViT` | Image features extracted by ViT |
| NLP Features (BERT) | `tabular_NLP_by_BERT` | Text embeddings from BERT |
| NLP Features (RoBERTa) | `tabular_NLP_by_RoBERTa` | Text embeddings from RoBERTa |
| Video | `video` | Video anomaly detection (I3D features) |
| MIL Bags (Classical) | `classical_bags_inexact` | Classical data in MIL bag format |
| MIL Bags (CV) | `CV_by_ViT_bags_inexact` | CV features in MIL bag format |

---

## 🤖 Supported Models

### Weakly-Supervised (Instance)

| Model | Category | Description |
|-------|----------|-------------|
| DevNet | Score Learning | Deviation networks for anomaly detection with limited supervision |
| DeepSAD | Score Learning | Deep semi-supervised anomaly detection via one-class classification |
| PReNet | Score Learning | Pairwise relation network for anomaly detection |
| REPEN | Repr. Learning | Representation learning for PU learning |
| XGBOD | Repr. Learning | Feature augmentation for outlier detection |
| RoSAS | Data Aug. | Robust semi-supervised anomaly segmentation |
| Dual-MGAN | Data Aug. | Dual-MGAN for anomaly detection |
| FEAWAD | Reconstruction | Feature encoding with autoencoders for weakly-supervised AD |
| DDAE | Diffusion DAE | Anomaly detection with denoising diffusion autoencoders |
| SOEL-NTL | Pseudo-Labeling | Self-training with outlier exposure |
| AA-BiGAN | GAN-based | Adversarially learned anomaly detection with BiGAN |
| GAnomaly | GAN-based | GAN-based anomaly detection |

### Unsupervised (Instance)

| Model | Category | Description |
|-------|----------|-------------|
| IForest | Isolation-based | Isolation Forest - classical baseline |
| AutoEncoder | Reconstruction | Autoencoder reconstruction error |
| VAE | Reconstruction | Variational Autoencoder |
| PCA | Reconstruction | Principal Component Analysis |
| DeepSVDD | Deep One-class | Deep Support Vector Data Description |
| ECOD | Probabilistic | Empirical Cumulative Distribution |
| CBLOF | Cluster-based | Cluster-based Local Outlier Factor |
| LOF | Density-based | Local Outlier Factor |
| LUNAR | GNN-based | Graph neural network for anomaly detection |

### Weakly-Supervised (Bag)

| Model | Category | Description |
|-------|----------|-------------|
| Sultani | Vanilla MIL | MIL-based weakly supervised video anomaly detection |
| RTFM | Magnitude MIL | Robust temporal feature magnitude |
| MGFN | Magnitude MIL | Multi-graph fusion network |
| AR-Net | Dynamic MIL | Dynamic MIL for video anomaly detection |
| VadCLIP | Language-Guided MIL | Vision-language video anomaly detection |
| UR-DMU | Uncertainty-Aware MIL | Unified representation for detection of multiple anomalies |
| GCN-Anomaly | Label Denoising | Graph convolutional network for anomaly detection |
| PUMA | PU MIL | PU-learning based multi-model anomaly detection |

### Supervised (Instance)

| Model | Category | Description |
|-------|----------|-------------|
| XGBoost | GBDT | Gradient boosting decision trees |
| CatBoost | GBDT | Categorical boosting |
| FTTransformer | Deep (Sup.) | Feature-wise transformer for tabular data |
| TabM | Deep (Sup.) | Tabular deep learning model |
| TabR-S | Deep (Sup.) | Tabular regression with scaled embeddings |

### Foundation Models (Instance)

| Model | Category | Description |
|-------|----------|-------------|
| TabPFN | Found. Model | Descriminative Foundation Model |
| LimiX | Found. Model | Generative Foundation Model |

---

## 📁 Project Structure

```
WSADBench/
├── run_experiment.py          # Main entry point
├── requirements.txt           # Python dependencies
├── setup.sh                   # Environment setup script
├── LICENSE                    # MIT License
├── README.md                  # This file
├── DATASETS.md                # Dataset preparation guide
│
├── WSADBench/                 # Core package
│   ├── baseline/              # Model implementations
│   │   ├── DeepSAD/           # DeepSAD implementation
│   │   ├── DevNet/            # DevNet implementation
│   │   ├── FEAWAD/            # FEAWAD implementation
│   │   ├── Sultani/           # Sultani video AD
│   │   ├── PyOD.py            # PyOD wrapper (20+ models)
│   │   └── ...                # 30+ other models
│   │
│   ├── datasets/              # Dataset handling
│   │   ├── data_generator.py  # Data generation & loading
│   │   ├── cv_data_generator.py # CV dataset handling
│   │   ├── dataset_configs/   # Dataset configuration (YAML)
│   │   └── dataset_support/   # Video preprocessing utilities
│   │
│   ├── model_configs/         # Model hyperparameters (YAML)
│   │   ├── tabular/           # Tabular model configs
│   │   ├── video/             # Video model configs
│   │   └── tabular_bags_inexact/ # MIL bag configs
│   │
│   ├── myutils.py             # Utility functions
│   └── build_bags.py          # Instance → MIL bag conversion
│
├── common_utils/              # Shared utilities
│   ├── baseline_utils.py      # Video-specific utilities
│   └── argTypes.py            # Argument type parsing
│
└── results/                   # Experiment outputs (git-ignored)
```

---

## ⚙️ Advanced Usage

### Key CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_type` | Data modality (required) | - |
| `--models` | Model names to run | - |
| `--datasets` | Specific datasets | All available |
| `--rla_list` | Labeled anomaly ratios | [1.0] |
| `--eln_list` | Labeled normal ratios (relative to RLA) | [0.0, 0.01, ...] |
| `--ru_list` | Unlabeled sample ratios | [1.0] |
| `--flip_nr_list` | Label noise (normal→anomaly) | [0.0] |
| `--flip_ar_list` | Label noise (anomaly→normal) | [0.0] |
| `--target_for_unlabeled` | How to handle unlabeled samples | `fill_unlabel_0` |
| `--noise_type` | Noise type for experiments | None |
| `--is_cleanlab` | Enable cleanlab data cleaning | `false` |
| `--seed_list` | Random seeds | [1-10] |
| `--n_jobs` | Parallel jobs | 1 |
| `--gpus` | GPU IDs (e.g., "0,1,2") | All available |
| `--output_dir` | Results directory | results/{data_type} |
| `--NO_RESUME` | Force re-run completed experiments | False |
| `--dry_summary` | Only generate summary | False |
| `--DEBUG` | Enable debug mode | False |
| `--exp_note` | Experiment note for tracking | None |

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
    │   └── {model_name}/
    │       ├── {model_name}_results.jsonl  # Individual results
    │       └── model_stats.json            # Model statistics
    └── summary/
        └── summary.xlsx                     # Aggregated statistics
```

---

## 📝 Citation

If you use WSADBench in your research, please cite:

```bibtex
@article{wsadbench2025,
  title={Rethinking Weak Supervision in Anomaly Detection: A Comprehensive Benchmark},
  author={WSADBench Authors},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [PyOD](https://github.com/yzhao062/pyod) - Python Outlier Detection library
- [ADBench](https://github.com/Minqi824/ADBench) - Anomaly Detection Benchmark

---

## 📞 Contact

For questions and issues, please open an issue on GitHub.
