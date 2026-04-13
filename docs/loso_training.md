# LOSO (Leave One Subject Out) 训练指南

本文档介绍如何在 EEG-BCI 框架中使用 LOSO 交叉验证进行被试无关（subject-independent）评估。

## 什么是 LOSO？

**Leave One Subject Out (LOSO)** 是一种严格的交叉验证策略：
- 每次**留出一个被试**作为测试集
- 使用**其余所有被试**的数据作为训练集
- 遍历所有被试，确保每个被试都被测试一次

LOSO 是评估 EEG-BCI 模型**跨被试泛化能力**的金标准。

## LOSO vs K-Fold 对比

| 特性 | K-Fold (Subject-Dependent) | LOSO (Subject-Independent) |
|------|---------------------------|---------------------------|
| 训练数据 | 同一被试的部分数据 | 其他所有被试的数据 |
| 测试数据 | 同一被试的剩余数据 | 全新的被试 |
| 评估目标 | 模型拟合能力 | 跨被试泛化能力 |
| 实际应用 | 需为每个新用户校准 | 可能无需用户校准 |
| 难度 | 相对容易 | 更具挑战性 |

## 快速开始

### 1. 创建 LOSO 配置文件

```yaml
# configs/experiment/my_loso_experiment.yaml
experiment:
  name: "LOSO_XWStroke_EEGNet"
  seed: 42

data:
  dataset: "XWStroke"
  subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 包含所有被试
  info_path: "configs/dataset/XWStroke.yaml"

model:
  type: "EEGNet"

training:
  epochs: 100
  batch_size: 64  # 可以用更大的 batch size（数据量更大）
  optimizer:
    lr: 0.001

trainer:
  type: "LOSOTrainer"  # 关键：指定 LOSO 训练器
  args:
    inner_k_folds: 0    # 可选：训练集内部的交叉验证折数
```

### 2. 运行 LOSO 训练

```bash
python train.py --config configs/experiment/my_loso_experiment.yaml --name loso_exp
```

### 3. 查看结果

训练完成后，结果保存在：
```
experiments/loso_exp/
├── checkpoints/              # 每个被试的最佳模型
│   ├── loso_subject1_acc0.7234.pt
│   ├── loso_subject2_acc0.8123.pt
│   └── ...
├── visualizations/
│   ├── loso_comparison.png   # 所有被试结果对比图
│   └── subject_1/
│       └── loso_training_history.png
├── logs/
│   └── training.log
└── results/
    └── results.json          # 详细结果数据
```

## 配置参数说明

### LOSO 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inner_k_folds` | int | 0 | 训练集内部交叉验证折数（0表示不使用） |
| `balance_classes` | bool | false | 是否平衡不同被试的类别分布（实验性） |

### 注意事项

- **k_folds 不适用于 LOSO**：LOSO 使用被试作为自然的交叉验证划分
- **batch_size 可以更大**：LOSO 的训练集包含多个被试的数据，通常样本量更大
- **训练时间更长**：需要训练 N 次（N=被试数），每次使用大量数据

## 结果解读

### 控制台输出示例

```
============================================================
Starting LOSO (Leave One Subject Out) Training
Total subjects: 10
Each subject will be tested once, trained on 9 others
============================================================

[1/10] Test Subject: 1
----------------------------------------
Loading training data from subjects: [2, 3, 4, 5, 6, 7, 8, 9, 10]
Training set: 4500 samples from 9 subjects
Test set: 500 samples from subject 1
    Model: 1,234,567 params (1,234,567 trainable)
LOSO Sub1 |████████████████████| 100/100 [02:15<00:00] LR:1.0e-04 Tr:0.234/0.912 Te:0.456/0.823
    Best: 0.8234 (epoch 87), Final: 0.8123

...

============================================================
LOSO Training Complete
Mean Accuracy: 0.7543 ± 0.0892
Range: [0.6123, 0.8912]
============================================================
```

### 结果 JSON 结构

```json
{
  "subjects": [
    {
      "test_subject_id": 1,
      "test_acc": 0.8123,
      "test_loss": 0.4567,
      "best_acc": 0.8234,
      "best_epoch": 87,
      "train_samples": 4500,
      "test_samples": 500
    },
    ...
  ],
  "overall_mean": 0.7543,
  "overall_std": 0.0892,
  "overall_min": 0.6123,
  "overall_max": 0.8912,
  "loso_config": {
    "num_subjects": 10,
    "subject_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  }
}
```

## 可视化结果

### 1. LOSO 对比图 (`loso_comparison.png`)

显示每个被试的测试准确率：
- **绿色条**: 准确率 ≥ 80%（良好泛化）
- **黄色条**: 准确率 60-80%（中等泛化）
- **红色条**: 准确率 < 60%（较差泛化）
- **红色虚线**: 平均准确率

### 2. 训练历史图 (`loso_training_history.png`)

每个被试的训练曲线：
- 蓝色：训练集（其他被试）
- 红色：测试集（当前被试）

## 与其他训练策略结合

### LOSO + 迁移学习

可以进行更复杂的实验，如：
1. 先在源数据集上预训练
2. 使用 LOSO 评估在目标数据集上的跨被试性能

```yaml
# configs/experiment/loso_transfer_learning.yaml
trainer:
  type: "TransferLOSOTrainer"  # 需要自定义实现
  args:
    pretrained_path: "experiments/pretrain/checkpoints/best.pt"
    freeze_strategy: "gradual"
```

> **注意**: 这需要创建自定义训练器，继承 LOSOTrainer 并添加迁移学习功能。

## 最佳实践

### 1. 数据准备

- **确保所有被试数据格式一致**
- **检查类别分布**：不同被试的类别分布应该相似
- **足够的被试数**：至少 5+ 被试才能获得稳定的 LOSO 估计

### 2. 超参数调整

- **学习率**：LOSO 训练数据更多，可以尝试稍大的学习率
- **Batch size**：可以使用更大的 batch size（如 64, 128）
- **训练轮数**：通常需要更多轮数（数据量大，收敛慢）

### 3. 结果分析

- **识别"困难被试"**：某些被试可能数据质量较差或与群体差异大
- **异常值处理**：如果某个被试准确率明显低于其他，检查数据质量
- **统计显著性**：如果做方法对比，使用配对 t 检验

## 故障排除

### 问题1：内存不足

**现象**：CUDA out of memory

**解决方案**：
- 减小 `batch_size`
- 使用数据加载器的 `num_workers`（如果支持）
- 使用更小的模型

```yaml
training:
  batch_size: 32  # 从 64 减小到 32
```

### 问题2：训练数据不平衡

**现象**：某些被试样本数远多于其他

**解决方案**：
- 启用 `balance_classes`（实验性功能）
- 手动筛选被试，确保样本数相近

```yaml
trainer:
  args:
    balance_classes: true
```

### 问题3：某些被试准确率特别低

**可能原因**：
- 该被试数据质量差
- 该被试与其他被试差异大（如患病程度不同）
- 模型过拟合到训练被试的特征

**解决方案**：
- 检查并清洗数据
- 考虑使用领域自适应（Domain Adaptation）技术
- 分析该被试的混淆矩阵，了解错误模式

## 与论文方法对比

在论文中报告 LOSO 结果时，建议包含：

| 指标 | 说明 |
|------|------|
| Mean ± Std | 所有被试的平均准确率和标准差 |
| Range | 最小和最大准确率 |
| Individual Results | 每个被试的具体结果（附录或表格） |
| Confusion Matrix | 总体混淆矩阵 |
| Statistical Test | 与基线方法的统计显著性检验 |

## 参考

- [Subject-Independent BCI: A Review](https://www.frontiersin.org/articles/10.3389/fnins.2019.00262/full)
- [Transfer Learning for BCI](https://arxiv.org/abs/1902.05253)
