# 大规模数据集 LOSO 实验指南

针对 XW Stroke 数据集（50被试）等高被试数量、高异质性数据集，本文档介绍三种替代方案。

## 问题分析

### 挑战1：内存限制
- **标准LOSO**：需同时加载49个被试的数据
- **假设**：每被试500样本 × 32通道 × 750时间点 × 8频段 × 4字节 ≈ **1.8GB/被试**
- **总内存需求**：49 × 1.8GB ≈ **88GB**（远超普通GPU显存）

### 挑战2：被试异质性
- 中风患者数据个体差异大（病程、病灶位置、严重程度）
- 传统LOSO假设被试间分布相似，可能不适用
- 某些"困难被试"会显著拉低整体性能

---

## 方案一：流式LOSO (StreamingLOSOTrainer)

**核心思想**：每次只加载1个被试，逐被试训练

### 特点
| 特性 | 说明 |
|------|------|
| 内存占用 | **O(1)** - 常数级别（1-2个被试） |
| 训练方式 | 逐个被试流式训练，梯度累积模拟大批次 |
| 速度 | 略慢（频繁IO），但可通过缓存优化 |
| 精度 | 与标准LOSO基本一致 |

### 适用场景
- 被试数量 > 20
- 内存受限（显存 < 16GB）
- 被试间差异中等

### 配置文件

```yaml
# configs/experiment/xwstroke_streaming_loso.yaml
trainer:
  type: "StreamingLOSOTrainer"
  args:
    subject_buffer_size: 1        # 内存中保持的被试数
    gradient_accumulation_steps: 4  # 梯度累积步数
    mixed_precision: true         # 混合精度加速
```

### 运行指令

```bash
python train.py --config configs/experiment/xwstroke_adfcnn_streaming_loso.yaml \
                --name xwstroke_streaming_loso
```

### 内存对比

| 方法 | 50被试数据集内存需求 |
|------|---------------------|
| 标准LOSO | ~88 GB |
| StreamingLOSO | ~2-4 GB |

---

## 方案二：分组LOGO (LOGOTrainer)

**核心思想**：将被试按临床特征分组，留一组测试

### 特点
| 特性 | 说明 |
|------|------|
| 分组策略 | 按严重程度/年龄/病灶位置等 |
| 评估目标 | 组间泛化能力（临床更相关） |
| 内存占用 | 中等（可用streaming模式） |
| 结果解释 | 更直观（轻/中/重度组间差异） |

### 适用场景
- 被试有明显亚群（如轻/中/重度）
- 关注模型对特定亚群的适应性
- 临床转化应用

### 数据配置要求

需在数据集配置中添加 `subject_info`：

```yaml
# configs/dataset/XWStroke.yaml
dataset:
  name: "XWStroke"
  
  # 被试元数据（用于分组）
  subject_info:
    "1": {severity: "mild", age_group: "young", lesion: "cortical"}
    "2": {severity: "severe", age_group: "old", lesion: "subcortical"}
    "3": {severity: "moderate", age_group: "middle", lesion: "cortical"}
    # ... 更多被试

preprocessing:
  # ... 其他预处理配置
```

### 配置文件

```yaml
# configs/experiment/xwstroke_logo.yaml
trainer:
  type: "LOGOTrainer"
  args:
    grouping_key: "severity"      # 按严重程度分组
    streaming: true               # 启用流式加载
    gradient_accumulation_steps: 4
```

### 分组示例

假设50名被试按 `severity` 分组：
- **mild**: 15人
- **moderate**: 20人  
- **severe**: 15人

LOGO将执行3轮：
1. Train: moderate + severe, Test: mild
2. Train: mild + severe, Test: moderate
3. Train: mild + moderate, Test: severe

---

## 方案三：轻量级LOSO（随机子集）

**核心思想**：每次从训练被试中随机采样K个，而非使用全部

### 特点
- 内存可控（选K=5-10）
- 训练速度极快
- 适合快速实验和超参调优
- 牺牲部分精度换取效率

### 实现建议

修改配置文件，限制训练被试数：

```yaml
# configs/experiment/xwstroke_light_loso.yaml
trainer:
  type: "StreamingLOSOTrainer"
  args:
    max_train_subjects: 10    # 每轮只用10个被试训练
    random_selection: true    # 随机选择
```

> **注意**：此功能需要额外实现，当前版本未内置。

---

## 方案对比总结

| 方案 | 内存需求 | 训练时间 | 精度 | 适用场景 |
|------|---------|---------|------|---------|
| **标准LOSO** | 极高(88GB) | 快 | 基准 | 被试少(<20)，内存充足 |
| **StreamingLOSO** | 低(2-4GB) | 中等 | ~基准 | **推荐**：被试多，内存受限 |
| **LOGO** | 可调 | 中等 | 高* | 被试有异质性亚群 |
| **轻量级LOSO** | 很低 | 很快 | 略低 | 快速实验，超参搜索 |

*LOGO精度评估更贴近临床实际

---

## 推荐配置（XW Stroke数据集）

### 推荐1：StreamingLOSO（平衡方案）

```yaml
# xwstroke_streaming_loso.yaml
experiment:
  name: "XWStroke_StreamingLOSO"

data:
  dataset: "XWStroke"
  subjects: [1, 2, ..., 50]  # 全部50人

trainer:
  type: "StreamingLOSOTrainer"
  args:
    subject_buffer_size: 2      # 保持2个被试在内存
    gradient_accumulation_steps: 4
    mixed_precision: true
```

**预期结果**：
- 内存占用：~4GB
- 每轮LOSO训练时间：~30分钟（50轮×30秒）
- 总实验时间：~25小时（50被试×30分钟）

### 推荐2：LOGO（异质性问题）

如果已知被试有明确亚群（如轻/中/重度）：

```yaml
# xwstroke_logo.yaml
trainer:
  type: "LOGOTrainer"
  args:
    grouping_key: "severity"
    streaming: true
```

**优势**：
- 识别模型对哪类患者效果好/差
- 临床解释性更强
- 可比传统LOSO减少计算量（组数<被试数）

---

## 性能优化技巧

### 1. 增加梯度累积

```yaml
trainer:
  args:
    gradient_accumulation_steps: 8  # 模拟batch_size=256
```

### 2. 使用混合精度

```yaml
trainer:
  args:
    mixed_precision: true  # 速度提升~1.5-2x，内存减少~40%
```

### 3. 调整缓存大小

```yaml
trainer:
  args:
    subject_buffer_size: 3  # 如果内存允许，增加缓存减少IO
```

### 4. 预处理数据保存

建议先运行预处理，保存处理后的数据：

```python
# 预处理脚本示例
for subject_id in range(1, 51):
    ds = XWStrokeDataset(subject_id=subject_id, dataset_info=info)
    np.save(f'cache/subject_{subject_id}_data.npy', ds.data)
    np.save(f'cache/subject_{subject_id}_labels.npy', ds.labels)
```

---

## 故障排除

### 问题1：训练速度过慢

**原因**：频繁加载被试数据

**解决方案**：
- 增大 `subject_buffer_size`（如果内存允许）
- 使用SSD存储数据
- 预处理并保存为.npy格式

### 问题2：某些被试准确率特别低

**原因**：异质性导致的"困难被试"

**解决方案**：
- 切换到LOGO，看是否是某组整体困难
- 检查被试数据质量
- 考虑使用Domain Adaptation技术

### 问题3：显存仍不足

**解决方案**：
- 减小 `batch_size`（如从32降到16）
- 增大 `gradient_accumulation_steps` 补偿
- 关闭 `mixed_precision`（如不稳定）

---

## 下一步建议

1. **先用StreamingLOSO跑通全流程**（验证可行性）
2. **分析结果识别困难被试**
3. **收集被试元数据（severity等）**
4. **切换到LOGO进行临床评估**

需要我帮你实现任何特定功能吗？例如：
- 数据预处理缓存脚本
- 被试元数据收集工具
- 结果分析脚本（识别困难被试）
