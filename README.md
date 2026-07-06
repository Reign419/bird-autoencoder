可以，下面是一版**英文 + 中文双语 README.md**，你可以直接复制到 GitHub 的 `README.md` 里。

````markdown
# Bird Image Autoencoder / 鸟类图像自编码器

## English Version

### 1. Project Overview

This repository contains autoencoder baselines for image reconstruction on the CUB-200-2011 bird image dataset.

The main goal is to learn a compact image representation using a **single-layer latent vector bottleneck** and reconstruct the original input image from this latent representation.

The basic pipeline is:

```text
Input image → Encoder → Single latent vector → Decoder → Reconstructed image
````

This project is an early-stage baseline for studying image representation learning. The long-term motivation is to explore reliable, interpretable, and potentially causally meaningful representations for trustworthy machine learning, with possible future applications in medical imaging.

---

### 2. Current Task

The current task is to improve image reconstruction quality while keeping the latent representation as a single vector.

In particular, the code supports experiments with:

* Different autoencoder architectures
* Different latent dimensions
* Different reconstruction losses
* Training and validation reconstruction visualization
* Loss curves and metric curves
* Difference maps between original and reconstructed images

---

### 3. Repository Structure

```text
bird-autoencoder/
├── main.py                    # Main training script
├── data.py                    # Dataset loading and preprocessing
├── losses.py                  # Loss functions and evaluation metrics
├── train_utils.py             # Training callbacks
├── visualize.py               # Reconstruction and loss visualization
├── model_cnn.py               # CNN autoencoder baseline
├── model_residual.py          # Residual autoencoder
├── model_residual_lite.py     # Lighter residual autoencoder
├── model_resnet50.py          # ResNet-50-based autoencoder encoder
├── set_tf_gpu.sh              # TensorFlow GPU environment setup script
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files excluded from Git tracking
└── README.md
```

---

### 4. Models

The repository currently includes the following models:

#### CNN Autoencoder

A simple convolutional autoencoder baseline.

```text
Image → CNN Encoder → Dense latent vector → CNN Decoder → Reconstruction
```

#### Residual Autoencoder

A stronger autoencoder using residual blocks to improve optimization and feature refinement.

#### Residual-lite Autoencoder

A lighter residual version that downsamples less aggressively. This is useful for preserving more spatial information before the latent layer.

#### ResNet-50 Autoencoder

A ResNet-50-based encoder followed by a single Dense latent vector and a custom decoder.

This model is used as a stronger encoder baseline, but it may not always improve reconstruction because ResNet-50 was originally designed for classification rather than image reconstruction.

---

### 5. Dataset

The experiments use the CUB-200-2011 bird image dataset.

The dataset is **not included** in this repository.

Expected image directory structure:

```text
CUB_200_2011/
└── images/
    ├── 001.Black_footed_Albatross/
    │   ├── image_0001.jpg
    │   └── ...
    ├── 002.Laysan_Albatross/
    │   ├── image_0001.jpg
    │   └── ...
    └── ...
```

In `main.py`, update the dataset path:

```python
dataset_path = "/path/to/CUB_200_2011/images"
```

---

### 6. Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The required packages include:

```text
tensorflow
numpy
pandas
matplotlib
pillow
scikit-learn
```

---

### 7. Running Experiments

Run the main training script:

```bash
python main.py
```

The main experimental settings can be modified in `main.py`:

```python
model_name = "resnet50"
latent_dims = [128, 256, 512, 1024]
```

Available model names:

```python
"cnn"
"residual"
"residual_lite"
"resnet50"
```

The default image size is:

```python
img_size = (64, 64)
```

The default train-validation split is:

```text
80% training / 20% validation
```

---

### 8. Loss Functions and Metrics

The repository supports several reconstruction losses:

* MSE loss
* L1 + SSIM loss
* L1 + SSIM + Edge loss
* MSE + SSIM loss
* MSE + SSIM + Edge loss

The following metrics are recorded:

* MSE
* L1
* SSIM
* SSIM loss
* Edge loss
* PSNR

The current recommended loss for sharper reconstruction is:

```python
l1_ssim_edge_loss
```

---

### 9. Outputs

Each experiment saves:

```text
config.json
model_summary.txt
encoder_summary.txt
decoder_summary.txt
log_*.csv
loss_*.png
metric curves
train reconstruction grid
validation reconstruction grid
train difference map
validation difference map
result.json
```

These outputs help answer the following questions:

* What training configuration was used?
* How long was the model trained?
* What does the loss curve look like?
* Are both training and validation reconstructions blurry?
* What do the decoded images look like?
* Where are the reconstruction errors located?

---

### 10. Research Notes

This project intentionally avoids U-Net-style skip connections because the goal is to force all reconstructed information to pass through the single latent vector.

Therefore, the model structure is constrained as:

```text
Image → Encoder → Single latent vector → Decoder → Reconstruction
```

This constraint makes the latent representation more meaningful for later analysis.

Future directions may include:

* Comparing full-image training with bounding-box-cropped bird images
* Testing larger image resolutions such as 128×128
* Studying whether the latent vector encodes bird-related features or background shortcuts
* Using the latent representation for classification or robustness experiments
* Connecting the learned latent space to causal representation learning

---

## 中文版本

### 1. 项目简介

本仓库包含基于 CUB-200-2011 鸟类图像数据集的自编码器图像重建实验代码。

本项目的核心目标是：在保持 **单层 latent vector 瓶颈结构** 的前提下，学习一个紧凑的图像表示，并从该 latent representation 重建原始输入图像。

基本流程为：

```text
输入图像 → Encoder → 单层 latent vector → Decoder → 重建图像
```

这是一个早期 baseline 项目，主要用于学习和分析图像表示。长期研究动机是探索更加可靠、可解释、具有潜在因果意义的表示学习方法，并为未来可信机器学习和医学影像应用打基础。

---

### 2. 当前任务

当前任务是改进图像重建质量，同时保持 latent representation 是一个单层向量。

目前代码支持以下实验：

* 不同 autoencoder 结构对比
* 不同 latent dimension 对比
* 不同 reconstruction loss 对比
* 训练集和验证集重建图像可视化
* loss curve 和 metric curve 保存
* 原图与重建图之间的 difference map 可视化

---

### 3. 仓库结构

```text
bird-autoencoder/
├── main.py                    # 主训练脚本
├── data.py                    # 数据读取和预处理
├── losses.py                  # 损失函数和评价指标
├── train_utils.py             # 训练 callbacks
├── visualize.py               # 可视化函数
├── model_cnn.py               # CNN autoencoder baseline
├── model_residual.py          # Residual autoencoder
├── model_residual_lite.py     # 轻量版 residual autoencoder
├── model_resnet50.py          # 基于 ResNet-50 encoder 的 autoencoder
├── set_tf_gpu.sh              # TensorFlow GPU 环境设置脚本
├── requirements.txt           # Python 依赖
├── .gitignore                 # Git 忽略文件
└── README.md
```

---

### 4. 模型结构

当前仓库包含以下模型。

#### CNN Autoencoder

一个简单的卷积自编码器 baseline。

```text
图像 → CNN Encoder → Dense latent vector → CNN Decoder → 重建图像
```

#### Residual Autoencoder

使用 residual block 的更强自编码器，用于增强模型表达能力和训练稳定性。

#### Residual-lite Autoencoder

一个更轻量的 residual autoencoder，下采样不那么激进，有助于在进入 latent vector 之前保留更多空间信息。

#### ResNet-50 Autoencoder

使用 ResNet-50 作为 encoder，然后接单层 Dense latent vector 和自定义 decoder。

该模型主要作为更强 encoder baseline，但 ResNet-50 原本是分类网络，不一定总是适合图像重建任务。

---

### 5. 数据集

本项目使用 CUB-200-2011 鸟类图像数据集。

数据集**不包含**在本仓库中。

期望的数据结构如下：

```text
CUB_200_2011/
└── images/
    ├── 001.Black_footed_Albatross/
    │   ├── image_0001.jpg
    │   └── ...
    ├── 002.Laysan_Albatross/
    │   ├── image_0001.jpg
    │   └── ...
    └── ...
```

需要在 `main.py` 中修改数据路径：

```python
dataset_path = "/path/to/CUB_200_2011/images"
```

---

### 6. 环境配置

创建并激活 Python 虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
tensorflow
numpy
pandas
matplotlib
pillow
scikit-learn
```

---

### 7. 运行实验

运行主训练脚本：

```bash
python main.py
```

可以在 `main.py` 中修改主要实验设置：

```python
model_name = "resnet50"
latent_dims = [128, 256, 512, 1024]
```

当前支持的模型名称：

```python
"cnn"
"residual"
"residual_lite"
"resnet50"
```

默认图像大小：

```python
img_size = (64, 64)
```

默认训练/验证划分：

```text
80% 训练集 / 20% 验证集
```

---

### 8. 损失函数和评价指标

当前支持的 reconstruction loss 包括：

* MSE loss
* L1 + SSIM loss
* L1 + SSIM + Edge loss
* MSE + SSIM loss
* MSE + SSIM + Edge loss

训练过程中记录以下指标：

* MSE
* L1
* SSIM
* SSIM loss
* Edge loss
* PSNR

当前为了改善图像边缘和结构清晰度，推荐使用：

```python
l1_ssim_edge_loss
```

---

### 9. 输出结果

每个实验会保存：

```text
config.json
model_summary.txt
encoder_summary.txt
decoder_summary.txt
log_*.csv
loss_*.png
metric curves
training reconstruction grid
validation reconstruction grid
training difference map
validation difference map
result.json
```

这些输出用于回答以下问题：

* 当前训练配置是什么？
* 模型训练了多久？
* loss curve 长什么样？
* 训练集和验证集的重建结果是否都模糊？
* decoded images / reconstructed images 的视觉效果如何？
* 重建误差主要出现在图像哪些区域？

---

### 10. 研究说明

本项目有意避免使用 U-Net 式 skip connection，因为当前目标是让所有重建信息都必须通过单层 latent vector。

因此模型结构被限制为：

```text
图像 → Encoder → 单层 latent vector → Decoder → 重建图像
```

这个约束有助于后续分析 latent representation 中到底编码了什么信息。

未来可以继续探索：

* full image 和 bounding-box-cropped bird image 的对比
* 64×64 与 128×128 图像分辨率对比
* latent vector 是否主要编码鸟本身，还是编码背景 shortcut
* 使用 latent representation 做分类或 robustness 实验
* 将 learned latent space 与 causal representation learning 联系起来

````
