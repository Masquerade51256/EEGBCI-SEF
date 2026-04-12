# Transfer Learning 使用指南

本文档介绍如何在 EEG-BCI 框架中使用迁移学习功能。

## 概述

迁移学习允许你利用在一个数据集（源域）上预训练的模型，将其知识迁移到另一个数据集（目标域）。这在以下场景特别有用：

- **跨被试迁移**：在一个被试上训练，迁移到其他被试
- **跨数据集迁移**：在一个数据集上预训练，迁移到另一个数据集
- **数据受限场景**：目标域数据较少时，利用源域的先验知识

## 快速开始

### 1. 准备预训练模型

首先需要一个预训练好的模型作为起点：

```bash
# 运行标准训练获得预训练模型
python train.py --config configs/experiment/xwstroke_eegnet.yaml --name pretrain_source
```

预训练模型会保存在 `experiments/pretrain_source/checkpoints/` 目录下。

### 2. 创建迁移学习配置

```bash
# 复制示例配置
Copy-Item configs/experiment/transfer_learning_example.yaml configs/experiment/my_tl_experiment.yaml
```

编辑配置，指定 `pretrained_path`：

```yaml
trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "experiments/pretrain_source/checkpoints/subject_1_fold_0_acc_0.8500.pt"
    freeze_strategy: "gradual"
```

### 3. 运行迁移学习训练

```bash
python train.py --config configs/experiment/my_tl_experiment.yaml --name tl_experiment
```

## 配置参数说明

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `pretrained_path` | string | 预训练模型检查点路径 |
| `freeze_strategy` | string | 冻结策略："none", "all", "gradual", "selective" |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `freeze_epochs` | int | 50 | 冻结阶段训练轮数（gradual策略） |
| `unfreeze_epochs` | int | 100 | 解冻后训练轮数（gradual策略） |
| `learning_rate_frozen` | float | 0.001 | 冻结阶段学习率 |
| `learning_rate_finetune` | float | 0.0001 | 微调阶段学习率 |
| `layers_to_unfreeze` | list | [] | 要解冻的层名列表（selective策略） |
| `warmup_epochs` | int | 0 | 预热轮数 |

## 冻结策略详解

### 1. none - 直接微调

不冻结任何层，直接使用预训练权重进行训练。

```yaml
trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "path/to/checkpoint.pt"
    freeze_strategy: "none"
```

**适用场景**：
- 源域和目标域数据分布相似
- 目标域数据充足
- 希望快速收敛

### 2. all - 特征提取器

冻结所有特征层，只训练最终的分类头。

```yaml
trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "path/to/checkpoint.pt"
    freeze_strategy: "all"
```

**适用场景**：
- 目标域数据很少
- 源域和目标域差异较大
- 防止过拟合

### 3. gradual - 渐进解冻（推荐）

分两阶段训练：
1. **阶段1**：冻结特征层，训练分类头
2. **阶段2**：解冻所有层，整体微调

```yaml
trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "path/to/checkpoint.pt"
    freeze_strategy: "gradual"
    freeze_epochs: 50      # 阶段1轮数
    unfreeze_epochs: 100   # 阶段2轮数
    learning_rate_frozen: 0.001    # 阶段1学习率
    learning_rate_finetune: 0.0001 # 阶段2学习率（较小）
```

**适用场景**：
- 大多数迁移学习任务
- 需要平衡训练速度和性能
- 避免灾难性遗忘

### 4. selective - 选择性解冻

根据层名模式选择性解冻特定层。

```yaml
trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "path/to/checkpoint.pt"
    freeze_strategy: "selective"
    layers_to_unfreeze:
      - "conv"        # 解冻包含 "conv" 的层
      - "classifier"  # 解冻包含 "classifier" 的层
```

**适用场景**：
- 需要精细控制哪些层可训练
- 部分层需要保持源域特征

## 配置示例

### 示例1：跨被试迁移（推荐 gradual 策略）

```yaml
# 在 Subject 1 上预训练，迁移到 Subject 2-5
experiment:
  name: "CrossSubject_TL"

data:
  dataset: "XWStroke"
  subjects: [2, 3, 4, 5]  # 目标被试

model:
  type: "EEGNet"

trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "experiments/subject1_pretrain/checkpoints/subject_1_fold_0_acc_0.8500.pt"
    freeze_strategy: "gradual"
    freeze_epochs: 30
    unfreeze_epochs: 70
    learning_rate_frozen: 0.001
    learning_rate_finetune: 0.0001
```

### 示例2：轻量级迁移（all 策略）

```yaml
# 数据受限，只训练分类头
experiment:
  name: "Lightweight_TL"

trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "experiments/pretrain/checkpoints/best_model.pt"
    freeze_strategy: "all"
    
training:
  epochs: 50  # 只需要少量轮数
```

### 示例3：直接微调（none 策略）

```yaml
# 数据充足，直接微调
experiment:
  name: "Direct_Finetune"

trainer:
  type: "TransferLearningTrainer"
  args:
    pretrained_path: "experiments/pretrain/checkpoints/best_model.pt"
    freeze_strategy: "none"
    
training:
  epochs: 100
  optimizer:
    lr: 0.0005  # 使用较小的学习率
```

## 查看可用训练器

```bash
python train.py --list-trainers
```

输出应包含：
```
Available trainers:
  - SupervisedTrainer
  - TransferLearningTrainer
```

## 与标准训练对比

| 特性 | SupervisedTrainer | TransferLearningTrainer |
|------|-------------------|-------------------------|
| 预训练权重 | 随机初始化 | 从检查点加载 |
| 层冻结 | 不支持 | 多种策略 |
| 分阶段训练 | 不支持 | gradual 策略支持 |
| 适用场景 | 从头训练 | 利用已有知识 |

## 最佳实践

### 1. 预训练模型选择

- 选择与目标域**相似度高**的源域
- 预训练模型**充分训练**（不要欠拟合的模型）
- 使用**相同架构**的模型

### 2. 学习率设置

- **冻结阶段**：可以使用较大学习率（如 1e-3）
- **微调阶段**：使用较小学习率（如 1e-4）
- 避免使用太大学习率破坏预训练权重

### 3. 训练轮数

- **freeze_epochs**：通常 30-50 轮
- **unfreeze_epochs**：通常 50-100 轮
- 可根据收敛情况调整

### 4. 验证策略

- 使用与标准训练相同的 k-fold 交叉验证
- 观察验证准确率是否比从头训练更高
- 对比不同冻结策略的效果

## 故障排除

### 问题1：模型架构不匹配

**错误信息**：
```
Skipped X layers: shape mismatch
```

**解决方案**：
- 确保预训练模型和目标模型架构相同
- 检查 `num_classes` 是否匹配（不匹配会自动跳过分类层）

### 问题2：预训练文件不存在

**错误信息**：
```
FileNotFoundError: Pretrained checkpoint not found
```

**解决方案**：
- 检查 `pretrained_path` 是否正确
- 使用绝对路径或确保相对路径正确

### 问题3：忘记解冻层

**现象**：
- 第二阶段训练时准确率不提升
- 可训练参数很少

**解决方案**：
- 确认 `freeze_strategy` 设置正确
-  gradual 策略会自动解冻，其他策略需要手动配置

## 高级用法

### 自定义层冻结逻辑

如需更复杂的冻结逻辑，可以继承 `TransferLearningTrainer`：

```python
from trainers.transfer_learning_trainer import TransferLearningTrainer
from core.registry import TRAINERS

@TRAINERS.register('MyCustomTLTrainer')
class MyCustomTLTrainer(TransferLearningTrainer):
    def _freeze_all_layers(self, model):
        # 自定义冻结逻辑
        for name, param in model.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
```

### 多阶段迁移

可以串联多个迁移学习实验：

```bash
# 阶段1：Dataset A -> Dataset B
python train.py --config tl_a_to_b.yaml --name tl_stage1

# 阶段2：Dataset B -> Dataset C (使用阶段1的输出)
python train.py --config tl_b_to_c.yaml --name tl_stage2
```

## 参考

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [How to Transfer Learning with PyTorch](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
