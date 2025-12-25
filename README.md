# WSADBench: Weakly-Supervised Anomaly Detection Benchmark

WSADBench是一个专门用于弱监督异常检测的综合基准测试平台，支持表格数据（tabular）、视频数据（video）以及图像数据的异常检测研究。

## 📋 项目概述

WSADBench提供了一个统一的框架来评估和比较各种弱监督异常检测算法。该项目包含了多种模态的数据集、多个基线模型以及完整的实验管理工具。

### 主要特性

- **多模态支持**: 支持表格数据、视频数据、图像数据
- **统一的实验框架**: 提供标准化的训练、测试和评估流程
- **丰富的基线模型**: 包含传统机器学习和深度学习方法
- **并行处理**: 支持大规模实验的并行执行
- **配置化管理**: 通过YAML配置文件管理数据集和模型参数

## 🚀 快速开始
### 环境配置

1. 安装依赖：
```bash
conda create -n ad python=3.9 -y
conda activate ad
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio
pip install tf-nightly
pip install -r requirements.txt
pip install pytorchvideo
pip install opencv-python
```

2. 配置数据集路径（详见下方数据集配置说明）

### 运行实验
#### 表格数据实验

```bash
# 运行单个模型
python run_experiment.py --models IForest --data_type tabular_CV_by_ResNet18
python run_experiment.py --models TargAD --data_type tabular_classical
# 快速测试
python run_experiment.py --data_type tabular --models TargAD --seed_list 1 --rla_list 0.1 --n_jobs 1 --gpus 0
python run_experiment.py --data_type tabular --models TargAD --datasets 30_satellite --rla_list 0.1 --n_jobs 1 --gpus 0
python run_experiment.py --data_type tabular_NLP_by_BERT --models GANomaly AABiGAN REPEN FEAWAD NTL PReNet RoSAS DeepSAD CatB DevNet LGB XGB XGBOD DualMGAN --rla_list 1 --eln_list 0.0 --seed_list 0 --n_jobs 1 --gpus 1 --target_for_unlabeled fill_unlabel_0 --NO_RESUME

#incomplete实验
CUDA_VISIBLE_DEVICES=5 python run_experiment.py --data_type tabular_classical --models RoSAS --rla_list 1 3 5 10 15 20 50 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.0 --flip_ar_list 0.0  --seed_list 0 1 2 3 4 --n_jobs 3 --gpus 5 --target_for_unlabeled fill_unlabel_0 --exp_note incomplete_nla
#AABiGAN PReNet RoSAS DeepSAD CatB DevNet GANomaly LGB XGB XGBOD TargAD
 AABiGAN REPEN FEAWAD NTL PReNet RoSAS DeepSAD CatB DevNet LGB XGB XGBOD DualMGAN
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --data_type tabular_CV_by_ViT --models REPEN FEAWAD NTL PReNet RoSAS DeepSAD DevNet XGBOD GANomaly AABiGAN --rla_list 1 3 5 10 15 20 50 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.0 --flip_ar_list 0.0  --seed_list 0 1 2 3 4 --n_jobs 3 --gpus 1 --target_for_unlabeled fill_unlabel_0

CUDA_VISIBLE_DEVICES=6 python run_experiment.py --data_type tabular_classical --models AABiGAN --rla_list 0.01 0.05 0.1 0.25 0.5 1.0 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.0 --flip_ar_list 0.0  --seed_list 0 1 2 3 4 --n_jobs 3 --gpus 6 --target_for_unlabeled fill_unlabel_0 --exp_note incomplete_rla


#rln实验
CUDA_VISIBLE_DEVICES=7 python run_experiment.py --data_type tabular_classical --models DeepSAD --rla_list 0.05 0.1 0.25 0.5 --eln_list 0.01 0.05 0.1 0.25 0.5 0.9 --ru_list 1.0 --flip_nr_list 0.0 --flip_ar_list 0.0  --seed_list 0 1 2 3 4 --n_jobs 3 --gpus 7 --target_for_unlabeled fill_unlabel_0 --exp_note rln_setting_eta05

#-------------------------------------------------------------------------------------------------------------------------------
#inaccurate 实验
PReNet FEAWAD NTL  AABiGAN XGB DualMGAN FTTransformer
RoSAS DeepSAD REPEN DevNet CatB XGBOD

CUDA_VISIBLE_DEVICES=5 python run_experiment.py --data_type tabular_classical --models LimiX --rla_list 1.0 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.01 0.05 0.1 0.25 0.5 --flip_ar_list 0.01 0.05 0.1 0.25 0.5 --seed_list 0 1 2 3 4 --n_jobs 2 --gpus 5 --target_for_unlabeled fill_unlabel_0 --noise_type label_contamination --is_cleanlab false --exp_note inaccurate_double

CUDA_VISIBLE_DEVICES=7 python run_experiment.py --data_type tabular_CV_by_ViT --models LimiX --rla_list 1.0 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.01 0.05 0.1 0.25 0.5 --flip_ar_list 0.0 --seed_list 0 1 2 3 4 --n_jobs 2 --gpus 7 --target_for_unlabeled fill_unlabel_0 --noise_type label_contamination --exp_note inaccurate_fnr

CUDA_VISIBLE_DEVICES=7 python run_experiment.py --data_type tabular_classical --models LimiX --rla_list 1.0 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.0  --flip_ar_list 0.01 0.05 0.1 0.25 0.5 --seed_list 0 1 2 3 4 --n_jobs 3 --gpus 7 --target_for_unlabeled fill_unlabel_0 --noise_type label_contamination --exp_note inaccurate_far

CUDA_VISIBLE_DEVICES=5 python run_experiment.py --data_type tabular_classical --models REPEN DeepSAD --rla_list 1.0 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.5 0.25 0.1 0.05 0.01 0.0 --flip_ar_list 0.0 --seed_list 0 1 2 3 4 --n_jobs 1 --gpus 5 --target_for_unlabeled fill_unlabel_0 --noise_type label_contamination --is_cleanlab true --exp_note inacc_with_clean

#unlabel 实验
 DevNet GANomaly AABiGAN REPEN FEAWAD NTL RoSAS DeepSAD CatB PReNet XGBOD 
 0.01 0.1 0.5 1.0 / 0.01 0.1 0.5 1.0
CUDA_VISIBLE_DEVICES=6 python run_experiment.py --data_type tabular_classical --models LimiX --rla_list 1 10 20 50 --eln_list 0.0 --ru_list 20 50 200 1000 --flip_nr_list 0.0 --flip_ar_list 0.0 --seed_list 0 1 2 3 4 --n_jobs 2 --gpus 6 --target_for_unlabeled fill_unlabel_0 --exp_note unlabel_nlanu
CatB NTL RoSAS DeepSAD PReNet XGBOD
DevNet GANomaly AABiGAN REPEN FEAWAD

1 10 20 50  /  20 50 200 1000
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --data_type tabular_classical --models DualMGAN --datasets 23_mammography --rla_list 1 --eln_list 0.0 --ru_list 20 --flip_nr_list 0.0 --flip_ar_list 0.0 --seed_list 0 --n_jobs 1 --gpus 0 --target_for_unlabeled fill_unlabel_0 --exp_note unlabel_setting

#tabular_inexact 实验
CUDA_VISIBLE_DEVICES=5 python run_experiment.py --data_type classical_bags_inexact --models LimiX --rla_list 0.01 0.05 0.1 0.25 0.5 1.0 --eln_list 0.0 --ru_list 1.0 --flip_nr_list 0.0 --flip_ar_list 0.0  --seed_list 0 1 2 3 4 --n_jobs 1 --gpus 5 --target_for_unlabeled fill_unlabel_0 --exp_note tabular_inexact

 
# 运行多个模型
python run_experiment.py --models IForest AABiGAN CRGAN --data_type tabular

# 指定其他参数
python run_experiment.py --models IForest --seeds 5 --processes 4 --data_type tabular
```

#### 视频数据实验
```bash
python run_experiment.py --models Sultani --datasets UCF_Crime --n_jobs 1 --rla_list 1.0 --eln_list 0.0  --seed_list 1  --data_type video --gpus 2

python run_experiment.py --models ARNet --datasets UCF_Crime.s32 --n_jobs 1 --rla_list 1.0 --eln_list 0.0 --seed_list 1  --data_type video --gpus 2
# 使用6号和7号两块gpu，并行跑2个任务跑Sultani 这个模型设置，跑10个seed。
python run_experiment.py --models Sultani  --datasets UCF_Crime --n_jobs 2 --rla_list 1.0  --data_type video  --gpus 6,7

```

## 📊 支持的模型

### 表格数据模型（已完成）

#### 传统机器学习模型
- **PyOD包模型**: IForest, OCSVM, ABOD, CBLOF, COF, COPOD, ECOD, FeatureBagging, HBOS, KNN, LOF, PCA, AutoEncoder等20+种模型

#### 深度学习模型
- **AABiGAN**: Adversarially Learned Anomaly Detection with BiGAN
- **CRGAN**: Consistent Regularization for Generative Adversarial Networks  
- **RoSAS**: Deep Semi-Supervised Anomaly Detection
- **FEAWAD**: Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection
- **DeepSAD**: Deep Semi-supervised Anomaly Detection
- **DevNet**: Deep Anomaly Detection with Deviation Networks
- **DAGMM**: Deep Autoencoding Gaussian Mixture Model
- **PReNet**: Pairwise Relation Network

### 视频数据模型

#### 已实现
- **Sultani**: Real-world Anomaly Detection in Surveillance Videos (MIL-based)

#### TODO
- 更多基于深度学习的视频异常检测模型

## ✅ 已完成功能

### 表格数据（Tabular）
- ✅ **完整的运行框架**: `run_tabular.py`脚本支持大规模并行实验
- ✅ **结果整理优化**: 自动生成Excel报告，包含统计分析
- ✅ **传统模型**: 完整的PyOD包集成（20+种算法）
- ✅ **深度学习模型**: AABiGAN、CRGAN、RoSAS等弱监督模型
- ✅ **配置化管理**: YAML配置文件支持模型参数管理

### 视频数据（Video）
- ✅ **数据预处理**: 并行流式预处理脚本，支持内存和负载平衡控制
- ✅ **Sultani模型**: 初步实现并可运行，基于MIL的弱监督异常检测
- ✅ **数据集支持**: UCF-Crime数据集完整支持

## 🔄 TODO列表

### 高优先级
- [ ] **Sultani模型优化**: 调整参数以保持与原文一致的性能
- [ ] **更多视频数据集**: 添加ShanghaiTech、Avenue等数据集支持
- [ ] **更多视频模型**: 集成GANomaly、MNAD等视频异常检测模型
- [ ] **原始图像模态支持**: 打通AABiGAN、CRGAN、RoSAS在原始图像数据集上的加载与运行

### 中优先级
- [ ] **模型性能基准**: 建立各模型在标准数据集上的性能基准
- [ ] **实验可视化**: 添加训练过程可视化和结果分析工具
- [ ] **超参数优化**: 自动化超参数搜索功能

### 低优先级
- [ ] **多模态融合**: 支持文本+图像、视频+音频等多模态异常检测
- [ ] **在线学习**: 支持流式数据的在线异常检测
- [ ] **OOD**: 支持样本外异常检测



## 📊 实验设置

### 标准实验协议
- **数据划分**: 训练集70%，测试集30%
- **交叉验证**: 10个不同随机种子
- **评估指标**: AUCROC、AUCPR、训练时间、推理时间
- **弱监督设置**: 支持不同标记异常样本比例（1%, 5%, 10%等）

### 数据集统计
- **表格数据集**: 30+个经典异常检测数据集
- **视频数据集**: UCF-Crime（已支持），ShanghaiTech（规划中）
- **图像数据集**: MNIST、CIFAR-10、MVTec-AD等（规划中）

## 🤝 贡献指南

欢迎贡献新的模型实现、数据集支持或功能改进：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`) 
5. 创建Pull Request

### 添加新模型

1. 在`WSADBench/baseline/`下创建模型目录
2. 实现标准接口：`fit()`、`predict_score()`方法
3. 添加配置文件到`WSADBench/model_configs/`
4. 更新文档和测试

## 📝 引用

如果您在研究中使用了WSADBench，请引用：

```bibtex
@misc{wsadbench2024,
  title={WSADBench: A Comprehensive Benchmark for Weakly-Supervised Anomaly Detection},
  author={WSADBench Team},
  year={2024},
  howpublished={\url{https://github.com/your-repo/WSADBench}}
}
```

## 📄 许可证

本项目基于MIT许可证开源 - 详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue到GitHub仓库
- 发邮件至项目维护者


