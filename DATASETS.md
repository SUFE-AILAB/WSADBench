# Dataset Preparation Guide

This document provides instructions for downloading and preparing datasets used in WSADBench.

> **Note**: The complete benchmark datasets (including pre-extracted features for all modalities) will be released after the paper is accepted. Currently, users can prepare datasets following the instructions below or use their own datasets.

---

## 📋 Overview

WSADBench supports the following data types:

| Data Type | CLI Flag | Description |
|-----------|----------|-------------|
| Classical Tabular | `tabular_classical` | Traditional AD benchmarks (47 datasets) |
| CV Features (ResNet18) | `tabular_CV_by_ResNet18` | Image features extracted by ResNet18 |
| CV Features (ViT) | `tabular_CV_by_ViT` | Image features extracted by ViT |
| NLP Features (BERT) | `tabular_NLP_by_BERT` | Text embeddings from BERT |
| NLP Features (RoBERTa) | `tabular_NLP_by_RoBERTa` | Text embeddings from RoBERTa |
| Video | `video` | Video anomaly detection (I3D/MViT features) |
| MIL Bags (Classical) | `classical_bags_inexact` | Classical data in MIL bag format |
| MIL Bags (CV) | `CV_by_ViT_bags_inexact` | CV features in MIL bag format |

---

## 🗂️ Directory Structure

After setup, your directory should look like:

```
WSADBench/
└── WSADBench/
    └── datasets/
        ├── Classical/           # -> symlink to classical datasets
        ├── CV_by_ResNet18/      # -> symlink to ResNet18 features
        ├── CV_by_ViT/           # -> symlink to ViT features
        ├── CV_by_I3D/           # -> symlink to video features
        ├── NLP_by_BERT/         # -> symlink to BERT embeddings
        ├── NLP_by_RoBERTa/      # -> symlink to RoBERTa embeddings
        ├── Classical_bags_inexact/
        └── CV_by_ViT_bags_inexact/
```

---

## 📥 Classical Tabular Datasets

**Sources**: 
- [ADBench](https://github.com/Minqi824/ADBench)
- [ODDS](https://odds.cs.stonybrook.edu/)
- [UCI ML Repository](https://archive.ics.uci.edu/)

**Available datasets** (47 total):
- `1_ALOI`, `2_annthyroid`, `3_breastw`, `4_cardio`, `5_cardiotocography`
- `6_celeba`, `7_cnt`, `8_cover`, `9_donors`, `10_fault`
- `11_fraud`, `12_glass`, `13_Hepatitis`, `14_http`, `15_InternetAds`
- `16_Ionosphere`, `17_landsat`, `18_letter`, `19_Lymphography`, `20_magic.gamma`
- `21_mammography`, `22_mnist`, `23_musk`, `24_optdigits`, `25_pageblocks`
- `26_pendigits`, `27_pima`, `28_satellite`, `29_satimage-2`, `30_shuttle`
- `31_skin`, `32_sofar`, `33_solar`, `34_spam`, `35_spamassassin`
- `36_Spect`, `37_stamp`, `38_thyroid`, `39_ver2`, `40_ver1`
- `41_vowels`, `42_waveform`, `43_WBC`, `44_WDBC`, `45_WPBC`
- `46_yeast`, `47_Zoo`

**Format**: `.npz` files with keys:
- `X` (features): numpy array of shape `(n_samples, n_features)`
- `y` (labels): numpy array of shape `(n_samples,)` where 0=normal, 1=anomaly

**Setup**:
```bash
# Download from ADBench (recommended)
# See: https://github.com/Minqi824/ADBench/tree/main/adbench/datasets

# Alternative: Download from JiHuLab (faster in China)
# See: https://jihulab.com/BraudoCC/ADBench_datasets/

# After downloading, create symlink
ln -s /path/to/Classical WSADBench/datasets/Classical
```

---

## 🖼️ CV Feature Datasets

### ResNet18 Features (`tabular_CV_by_ResNet18`)

Pre-extracted features from image datasets using ResNet18.

**Source datasets**: CIFAR-10, MNIST, Fashion-MNIST, MVTec-AD, etc.

**Format**: `.npz` files with `X` and `y` keys

**Setup**:
```bash
# After extracting features (see scripts below)
ln -s /path/to/cv_resnet18_features WSADBench/datasets/CV_by_ResNet18
```

### ViT Features (`tabular_CV_by_ViT`)

Pre-extracted features using Vision Transformer (ViT).

**Setup**:
```bash
ln -s /path/to/cv_vit_features WSADBench/datasets/CV_by_ViT
```

### Feature Extraction Script

If you need to extract features from your own images:

```python
# scripts/extract_resnet18_features.py
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Extract 512-dim feature vector from an image."""
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(x)
    return features.squeeze().numpy()

# Example usage
# features = extract_features("path/to/image.jpg")
# print(features.shape)  # (512,)
```

---

## 📝 NLP Feature Datasets

### BERT Embeddings (`tabular_NLP_by_BERT`)

Text datasets encoded with BERT-base-uncased.

**Source datasets**: Reuters, Amazon Reviews, etc.

**Setup**:
```bash
ln -s /path/to/nlp_bert_embeddings WSADBench/datasets/NLP_by_BERT
```

### RoBERTa Embeddings (`tabular_NLP_by_RoBERTa`)

Text datasets encoded with RoBERTa-base.

**Setup**:
```bash
ln -s /path/to/nlp_roberta_embeddings WSADBench/datasets/NLP_by_RoBERTa
```

### Text Embedding Extraction

```python
# scripts/extract_bert_embeddings.py
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load pretrained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def extract_embedding(text):
    """Extract 768-dim BERT embedding from text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding
    return outputs.last_hidden_state[0, 0, :].numpy()

# Example usage
# embedding = extract_embedding("This is a sample text.")
# print(embedding.shape)  # (768,)
```

---

## 🎬 Video Datasets

### UCF-Crime

**Paper**: [Real-World Anomaly Detection in Surveillance Videos (CVPR 2018)](https://arxiv.org/abs/1801.04264)

**Download**: [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)

**Categories**: Abuse, Arrest, Arson, Assault, Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, Vandalism, and Normal videos.

**Structure**:
```
UCF_Crime/
├── Anomaly-Videos-Part-1/
├── Anomaly-Videos-Part-2/
├── Normal-Videos-Part-1/
├── Normal-Videos-Part-2/
├── Testing_Annotation.txt
└── All_Videos/
```

### ShanghaiTech

**Paper**: [Abnormal Event Detection in Videos using Spatio-Temporal Autoencoder (ICDM 2018)](https://arxiv.org/abs/1801.04264)

**Download**: [ShanghaiTech Dataset](https://github.com/StevenLauHKHK/Abnormal-event-detection)

**Structure**: 13 scenes with different camera angles

### XD-Violence

**Paper**: [Towards Reliable Violence Detection in Videos (ECCV 2020)](https://arxiv.org/abs/2007.12094)

**Download**: [XD-Violence Dataset](https://roc-ng.github.io/XD-Violence/)

**Categories**: Abuse, Car Accident, Explosion, Fighting, Riot, Shooting, etc.

### UCSD Ped2

**Download**: [UCSD Anomaly Detection](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)

**Description**: Pedestrian walkway surveillance with anomalies like bikes, cars, skateboards.

### TAD (Tuberculosis)

**Description**: Medical video dataset for tuberculosis detection.

---

## 🔧 Video Feature Extraction

WSADBench expects pre-extracted video features. The framework supports I3D and MViT features.

> **Important**: We have unified the pretrained models used for feature extraction across all video datasets and re-extracted features directly from the original videos to ensure consistency and reproducibility. The feature extraction code is open-sourced in this repository under `WSADBench/datasets/dataset_support/`.

### Unified Feature Extraction Pipeline

Our video feature extraction pipeline provides:
- **Unified pretrained models**: Consistent I3D/MViT backbones across all datasets
- **Reproducible features**: All features extracted from original videos using the same pipeline
- **Open-source code**: Feature extraction scripts available in this repository

### Using Provided Preprocessing Script

```bash
# Video preprocessing with streaming (recommended for large datasets)
python WSADBench/datasets/dataset_support/video_preprocess_streaming.py \
    --video_dir /path/to/videos \
    --output_dir /path/to/features \
    --feature_type i3d \
    --segment_length 16
```

### I3D Feature Extraction

```bash
# Install PyTorchVideo
pip install pytorchvideo

# Extract I3D features
python scripts/extract_i3d_features.py \
    --video_dir /path/to/videos \
    --output_dir /path/to/features \
    --segment_length 16
```

### Expected Video Feature Format

```
CV_by_I3D/
└── UCF_Crime/
    ├── splits/
    │   ├── Anomaly_Train.txt
    │   └── Anomaly_Test.txt
    ├── segmentation/
    │   ├── video1.npy    # Shape: (n_segments, feature_dim)
    │   └── video2.npy
    └── Annotation.txt
```

---

## 📦 MIL Bag Datasets (Inexact Supervision)

For experiments with Multiple Instance Learning (bag-level) supervision:

### Convert Existing Data to Bags

```bash
# Convert classical tabular data to MIL bags
python WSADBench/build_bags.py \
    --input-dir ./WSADBench/datasets/Classical \
    --output-dir ./WSADBench/datasets/Classical_bags_inexact \
    --bag-size 10 \
    --anomaly-prob 0.3 \
    --seed 42
```

### Bag Format

Each `.npz` file contains:
- `X`: Shape `(n_bags, n_instances, n_features)`
- `y`: Bag-level labels (0=normal bag, 1=contains at least one anomaly)

**Bag Generation Logic**:
- Normal bag: All instances are normal
- Anomaly bag: At least one instance is anomalous (configurable ratio)

---

## ✅ Verification

After setting up datasets, verify with:

```bash
# List available datasets
python -c "from WSADBench.datasets.data_generator import DataGenerator; dg = DataGenerator(); print(dg.all_dataset_list)"

# Quick test run
python run_experiment.py \
    --data_type tabular_classical \
    --models DevNet \
    --datasets thyroid \
    --rla_list 1.0 \
    --seed_list 1
```

---

## 🆘 Common Issues

### Symlink Permission Denied

```bash
# Use absolute paths
ln -s /absolute/path/to/datasets WSADBench/datasets/Classical

# On Windows (Admin PowerShell)
New-Item -ItemType SymbolicLink -Path "WSADBench\datasets\Classical" -Target "C:\path\to\datasets"
```

### Dataset Not Found

- Check the symlink points to the correct directory
- Verify `.npz` files exist in the target directory
- Check file permissions: `ls -la WSADBench/datasets/`

### Video Features Mismatch

- Ensure segment length matches configuration (default: 16 frames)
- Verify feature dimension matches model expectations (I3D: 2048-dim)
- Check that video annotation files are present

### Memory Issues with Large Datasets

- Use streaming preprocessing for video data
- Reduce batch size in model configs
- Process datasets in smaller chunks

---

## 📚 References

1. **ADBench**: Zhao et al. "ADBench: Anomaly Detection Benchmark." NeurIPS 2023 Datasets Track.
2. **UCF-Crime**: Sultani et al. "Real-World Anomaly Detection in Surveillance Videos." CVPR 2018.
3. **ShanghaiTech**: Liu et al. "Abnormal Event Detection in Videos using Spatio-Temporal Autoencoder." ICDM 2018.
4. **XD-Violence**: Wu et al. "Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision." ECCV 2020.
