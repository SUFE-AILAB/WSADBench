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

- Python 3.9+
- CUDA 11.8+ (for GPU support)

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

### Tabular Experiments

```bash
# Run a single model on classical tabular datasets
python run_experiment.py --data_type tabular_classical --models DevNet --rla_list 1.0

# Run multiple models with different labeled anomaly ratios
python run_experiment.py \
    --data_type tabular_classical \
    --models DeepSAD DevNet FEAWAD \
    --rla_list 0.01 0.05 0.1 0.5 1.0 \
    --n_jobs 4

# Run with custom seeds
python run_experiment.py \
    --data_type tabular_classical \
    --models DevNet \
    --seed_list 1 2 3 4 5
```

### Video Experiments

```bash
# Run video anomaly detection
python run_experiment.py \
    --data_type video \
    --models Sultani \
    --datasets UCF_Crime \
    --rla_list 1.0 \
    --n_jobs 1 \
    --gpus 0

# Multi-GPU parallel execution
python run_experiment.py \
    --data_type video \
    --models Sultani \
    --datasets UCF_Crime \
    --n_jobs 2 \
    --rla_list 1.0 \
    --gpus 0,1
```

### Resume Interrupted Experiments

```bash
# WSADBench automatically skips completed experiments
python run_experiment.py --data_type tabular_classical --models DevNet

# Force re-run all experiments
python run_experiment.py --data_type tabular_classical --models DevNet --NO_RESUME
```

### Generate Summary Only

```bash
# Generate summary from existing results without running experiments
python run_experiment.py --data_type tabular_classical --dry_summary
```

---

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

### Key Highlight Models

| Model | Category | Paper | Description |
|-------|----------|-------|-------------|
| **DeepSAD** | Semi-supervised | [ICLR 2020](https://arxiv.org/abs/1906.02614) | Deep semi-supervised anomaly detection via one-class classification |
| **DevNet** | Weakly-supervised | [KDD 2019](https://arxiv.org/abs/1908.03909) | Deviation networks for anomaly detection with limited supervision |
| **FEAWAD** | Weakly-supervised | [KDD 2021](https://arxiv.org/abs/2106.07051) | Feature encoding with autoencoders for weakly-supervised AD |
| **Sultani** | Video MIL | [CVPR 2018](https://arxiv.org/abs/1801.04264) | Real-world anomaly detection in surveillance videos |
| **IForest** | Unsupervised | [ICDM 2008] | Isolation Forest - classical baseline |

### All Supported Models

#### Weakly-Supervised & Semi-Supervised Methods (Tabular)

| Model | Description |
|-------|-------------|
| DeepSAD | Deep semi-supervised anomaly detection |
| DevNet | Deep anomaly detection with deviation networks |
| FEAWAD | Feature encoding with autoencoders |
| PReNet | Pairwise relation network for anomaly detection |
| REPEN | Representation learning for PU learning |
| RoSAS | Robust semi-supervised anomaly segmentation |
| AABiGAN | Adversarially learned anomaly detection with BiGAN |
| CRGAN | Consistent regularization for GANs |
| DAGMM | Deep autoencoding Gaussian mixture model |
| DualMGAN | Dual-MGAN for anomaly detection |
| NTL | Neutral active learning |
| TargAD | Targeted anomaly detection |
| PUMA | PU-learning based multi-model anomaly detection |
| LimiX | 16M retrieval-based model |

#### Video Anomaly Detection

| Model | Description |
|-------|-------------|
| Sultani | MIL-based weakly supervised video anomaly detection |
| RTFM | Robust temporal feature magnitude |
| MGFN | Multi-graph fusion network |
| URDMU | Unified representation for detection of multiple anomalies |
| VadClip | Vision-language video anomaly detection |
| ZhongGCNAD | Graph convolutional network for AD |
| GANomaly | GAN-based anomaly detection |

#### Tabular Specialized Models

| Model | Description |
|-------|-------------|
| FTTransformer | Feature-wise transformer for tabular data |
| TabNet | Tabular neural network with sequential attention |
| TabPFN | Tabular prior-data fitted network |
| TabR_S | Tabular regression with scaled embeddings |
| TabMCls | Tabular multi-classifier approach |
| AnoDDAE | Anomaly detection with denoising diffusion autoencoders |

#### Unsupervised Methods (via PyOD)

| Model | Description |
|-------|-------------|
| IForest | Isolation Forest |
| LOF | Local Outlier Factor |
| OCSVM | One-Class SVM |
| KNN | k-Nearest Neighbors |
| HBOS | Histogram-based Outlier Score |
| PCA | Principal Component Analysis |
| AutoEncoder | Autoencoder reconstruction |
| VAE | Variational Autoencoder |
| ECOD | Empirical Cumulative Distribution |
| COPOD | Copula-based Outlier Detection |
| CBLOF | Cluster-based Local Outlier Factor |
| LUNAR | Linear Unbounded Anomaly Rating |
| MCD | Minimum Covariance Determinant |
| SOS | Stochastic Outlier Selection |
| AAE | Adversarial Autoencoder |
| DeepSVDD | Deep Support Vector Data Description |

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
| `--eln_list` | Labeled normal ratios (relative to RLA) | [0.01, 0.05, ...] |
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
