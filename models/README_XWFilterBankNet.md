# XWFilterBankNet 使用指南

## 模型简介

**XWFilterBankNet** 是一个专为 Filter Bank EEG 数据设计的模块化深度学习模型，具有以下特点：

1. **多频带处理**: 支持任意数量的 Filter Bank（频带）
2. **跨频带注意力**: 自动学习不同频带的重要性（如 Mu 节律、Beta 节律）
3. **模块化架构**: Backbone (特征提取) + Classifier (分类器) 分离
4. **跨数据集兼容**: 适用于 XWStroke、BCICIV2a 等多种 EEG 数据集

## 模型变体

### 1. XWFilterBankNet (标准版)
- 完整功能，包含跨频带注意力机制
- 参数量约 6400 万（使用默认配置）
- 适合大型数据集

### 2. XWFilterBankNetLight (轻量版)
- 精简参数，参数量约 5000
- 训练速度快
- 适合小型数据集或快速实验

## 快速开始

### 训练 XWStroke 数据集（多频带配置）

```bash
python train.py --config configs/experiment/xwstroke_filterbanknet.yaml
```

### 训练 BCICIV2a 数据集

```bash
python train.py --config configs/experiment/bciciv2a_filterbanknet.yaml
```

### 使用轻量版模型

```bash
python train.py --config configs/experiment/xwstroke_filterbanknet_light.yaml
```

## 配置文件说明

### 数据集配置 (configs/dataset/XWStroke_multiband.yaml)

```yaml
filter_bank:
  filter_banks:
    - [4, 8]      # Theta
    - [8, 12]     # Mu (运动皮层活动)
    - [12, 16]    # Low Beta
    - [16, 20]    # Mid Beta
    - [20, 24]    # High Beta
    - [24, 30]    # Low Gamma
    - [30, 36]    # Mid Gamma
    - [36, 40]    # High Gamma
```

### 模型配置 (configs/experiment/xwstroke_filterbanknet.yaml)

```yaml
model:
  type: "XWFilterBankNet"
  args:
    band_channels: 32        # 每个频带的特征通道数
    temporal_kernel: 64      # 时间卷积核大小
    dropout: 0.3             # Dropout 率
    use_attention: true      # 启用跨频带注意力
    classifier_hidden: 0     # 分类器隐藏层（0=线性分类器）
```

## 模型参数自动推导

模型会自动从数据集配置中推导以下参数：
- `num_channels`: 从 `channel_selection` 获取
- `num_bands`: 从 `filter_banks` 数量获取
- `input_length`: `target_sr * window_length_sec`
- `num_classes`: 从数据集配置获取

## 扩展到新数据集

要在一个新的数据集上使用 XWFilterBankNet：

1. **创建数据集配置文件** (configs/dataset/MyDataset.yaml):
```yaml
dataset:
  name: "MyDataset"
  # ... 其他配置

preprocessing:
  filter_bank:
    filter_banks:
      - [low1, high1]
      - [low2, high2]
      # ... 更多频带
```

2. **创建实验配置文件** (configs/experiment/myexperiment.yaml):
```yaml
experiment:
  name: "MyExperiment"

data:
  dataset: "MyDataset"
  info_path: "configs/dataset/MyDataset.yaml"

model:
  type: "XWFilterBankNet"
  args:
    use_attention: true
    # 其他参数...
```

3. **运行训练**:
```bash
python train.py --config configs/experiment/myexperiment.yaml
```

## 可解释性功能

获取频带注意力权重（用于分析模型关注哪些频段）：

```python
import torch
from models import register_all_models
from core.registry import MODELS

register_all_models()

model_cls = MODELS.get('XWFilterBankNet')
model = model_cls(num_classes=2, num_channels=30, num_bands=8, input_length=500)

# 输入数据
x = torch.randn(1, 8, 30, 500)

# 获取注意力权重
attention_weights = model.get_band_attention(x)
print(f"各频带注意力权重: {attention_weights.squeeze().tolist()}")
```

## 与现有模型对比

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| XWFilterBankNet | ~64M | 跨频带注意力 | 大型数据集 |
| XWFilterBankNetLight | ~5K | 轻量快速 | 小型数据集 |
| EEGNet | ~4K | 经典轻量 | 通用场景 |
| FBCNet | ~100K | Filter Bank | 频带分析 |

## 注意事项

1. **CUDA 内存**: 标准版模型参数量较大，确保 GPU 内存充足
2. **Filter Bank 数量**: 频带越多，计算量越大，建议根据任务选择合适的频段
3. **注意力机制**: 可以关闭 `use_attention: false` 以减少参数量
