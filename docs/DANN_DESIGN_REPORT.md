# DANN 域自适应跨被试迁移：设计与实现报告

> **版本**: v1.0  
> **日期**: 2026-05-18  
> **目标**: 通过 Domain-Adversarial Neural Network (DANN) 降低 stroke 患者间的跨被试域差异，提升 LOSO 准确率

---

## 1. 研究背景与动机

### 1.1 当前瓶颈

在先前的实验中，Aff+Align（患健侧标签 + 半球对齐）将全 50 人 LOSO 从 **49.63%** 提升至 **51.46%**（+1.83%, p=0.062），效果有限且未突破 ~55% 的天花板。

分层分析揭示了一个关键规律：
- **皮层下病灶 (Subcortical)** 患者受益最多 (+3.6%)
- **皮层病灶 (Cortical)** 患者被系统性损害 (-3.6%)
- **重度 NIHSS (≥8)** 患者被损害 (-5.5%)

这说明：**不同病灶位置的患者具有本质上不同的 MI 神经表征"域"**。简单的全局标签映射和通道翻转不足以对齐这些异质性域。

### 1.2 核心假设

> **如果让特征提取器学习"病灶位置无关"的表示（即域判别器无法根据特征判断患者是皮层下还是皮层病灶），那么跨被试分类的泛化性能将显著提升。**

这就是 **Domain-Adversarial Neural Network (DANN)** 的核心思想：通过梯度反转层 (Gradient Reversal Layer, GRL) 让特征提取器在优化分类任务的同时，"欺骗"域判别器。

---

## 2. DANN 方法介绍

### 2.1 经典 DANN 架构

```
Input X
  |
  v
[Feature Extractor G]  --->  [Task Classifier C]  --->  Task Prediction Y
  |
  v
[Gradient Reversal Layer]
  |
  v
[Domain Discriminator D]  --->  Domain Prediction D
```

**损失函数**:
```
L = L_task(Y, Y_true) - λ * L_domain(D, D_true)
```

注意负号：通过 GRL，反向传播时梯度被反转并缩放 λ，等价于：
```
L = L_task + λ * L_domain
```

但优化方向相反：
- 域判别器 D 试图最小化 L_domain（准确区分域）
- 特征提取器 G 通过 GRL 接收反向梯度 `-λ * ∂L_domain/∂G`，试图最大化 L_domain（让 D 分不清）

### 2.2 λ 调度策略

遵循 Ganin et al. (2016) 的原始设计：

```
p = epoch / total_epochs
λ(p) = 2 / (1 + exp(-γ * p)) - 1
```

- **训练初期 (p≈0)**: λ≈0，模型专注于学习任务分类
- **训练末期 (p≈1)**: λ≈1，模型强制学习域无关特征
- **γ 参数**: 控制过渡陡峭程度，默认 γ=10

这种渐进式调度避免了训练初期域对抗信号过强干扰任务学习。

---

## 3. 域标签设计

### 3.1 域分组策略

基于 `participants.tsv` 中的临床元数据，我们实现了 **4 种域分组策略**：

| 策略 | 域数量 | 域定义 | 科学依据 |
|:---|:---:|:---|:---|
| **stroke_location_4class** (默认) | 4 | Subcortical, Brainstem, Cortical, Mixed | 分层分析最强调节变量 |
| **stroke_location_2class** | 2 | Subcortical vs Non-subcortical | 简化版，突出受益群体 |
| **paralysis_side** | 2 | Left vs Right | 验证偏瘫侧是否是隐性域 |
| **nihss_3class** | 3 | Mild(1-3), Moderate(4-7), Severe(≥8) | 验证损伤严重度域效应 |

### 3.2 默认策略：stroke_location_4class

**域分布**（基于 50 被试）：

| 域 ID | 类别 | 人数 | 占比 |
|:---:|:---|:---:|:---:|
| 0 | Subcortical | ~20 | 40% |
| 1 | Brainstem | ~18 | 36% |
| 2 | Cortical | ~4 | 8% |
| 3 | Mixed | ~8 | 16% |

**注意**：Cortical 人数较少（4人），在域判别中可能样本不足。如果实验发现 Cortical 域不稳定，建议切换到 `stroke_location_2class`。

### 3.3 病灶位置分类规则

复用 `scripts/run_stratified_analysis.py` 中的关键词匹配逻辑：

```python
# 皮层关键词
cortical_kw = ['cortex', 'frontal', 'parietal', 'temporal', 'occipital', 'insula', 'watershed']

# 皮层下关键词
subcortical_kw = ['basal ganglia', 'thalamus', 'internal capsule', 'corona radiata',
                  'centrum semiovale', 'paraventricular', 'subcortical']

# 脑干关键词
brainstem_kw = ['pons', 'medulla oblongata']

# 小脑关键词
cerebellar_kw = ['cerebellum', 'cerebellar']

# 匹配多个类别 → Mixed
# 仅匹配一个类别 → 该类别
```

---

## 4. 模型架构：DANNADFCNN

### 4.1 与 ADFCNN 的关系

```python
DANNADFCNN(
    backbone=ADFCNN.backbone,      # 完全复用，预训练权重可迁移
    classifier=ADFCNN.classifier,  # 完全复用
    domain_discriminator=DomainDiscriminator(
        input_dim=flattened_features,
        hidden_dims=[256, 128],
        num_domains=4
    )
)
```

### 4.2 Domain Discriminator 结构

```
Input: [B, feature_dim]
  |
  v
Linear(feature_dim -> 256) + ReLU + Dropout(0.5)
  |
  v
Linear(256 -> 128) + ReLU + Dropout(0.5)
  |
  v
Linear(128 -> num_domains)
```

**设计理由**：
- 隐藏层 [256, 128] 足够表达域分布差异，但不过深（避免过拟合小样本域）
- Dropout 0.5 防止域判别器过强导致特征提取器难以欺骗

### 4.3 输入适配

与 `ADFCNNAdapter` 保持一致：
- 原始输入：`[B, num_bands, C, T]`
- reshape 后：`[B, 1, num_bands*C, T]`
- backbone 接收的 `num_channels = num_channels * num_bands`

---

## 5. 训练流程

### 5.1 DANNStreamingLOSOTrainer

继承自 `StreamingLOSOTrainer`，核心修改：

| 方法 | 修改内容 |
|:---|:---|
| `train()` | 初始化 `DomainInfoManager`，解析域标签 |
| `_build_cached_train_loader()` | 为每个样本附加 `domain_id`，DataLoader 返回三元组 `(data, label, domain)` |
| `_train_epoch_cached()` | 双损失前向传播：task_loss + λ * domain_loss |
| `_compute_grl_lambda()` | λ 调度：按 epoch 比例渐进增大 |

### 5.2 训练伪代码

```python
for each test_subject in subjects:
    train_subjects = all - {test_subject}
    train_loader = build_cached_train_loader(train_subjects)  # 带 domain labels
    
    for epoch in epochs:
        lambda_p = compute_grl_lambda(epoch)
        
        for batch in train_loader:
            x, y_task, y_domain = batch
            
            # Forward
            task_pred, domain_pred = model(x, alpha=lambda_p)
            
            # Losses
            L_task = CrossEntropy(task_pred, y_task)
            L_domain = CrossEntropy(domain_pred, y_domain)
            L_total = L_task + domain_loss_weight * L_domain
            
            # Backward (GRL自动反转域判别梯度)
            L_total.backward()
            optimizer.step()
        
        # Evaluate on test subject (只使用 task_pred)
        test_acc = evaluate(model, test_loader)
```

### 5.3 早停与模型选择

沿用 `StreamingLOSOTrainer` 的验证早停机制：
- 从 N-1 训练被试中随机选 1 个作为验证集
- 模型选择基于 **验证集 task accuracy**（绝不偷看测试集）
- Early stopping patience = 15 epochs

---

## 6. 实验配置

### 6.1 主要实验

| 实验 | 配置文件 | 被试数 | 域策略 | 用途 |
|:---|:---|:---:|:---|:---|
| **DANN Full50** | `configs/experiment/xwstroke_dann_loso.yaml` | 50 | stroke_location_4class | **主实验** |
| **DANN Subset15** | `configs/experiment/xwstroke_dann_subset15.yaml` | 15 | stroke_location_4class | 快速验证 |

### 6.2 超参数

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| model | DANNADFCNN | 域自适应 ADFCNN |
| num_domains | 4 | 与域策略对应 |
| domain_hidden_dims | [256, 128] | 域判别器隐藏层 |
| domain_loss_weight | 1.0 | 域损失系数 |
| grl_gamma | 10.0 | λ 调度陡峭度 |
| epochs | 100 | 与基线一致 |
| batch_size | 512 | StreamingLOSO 默认 |
| lr | 0.001 | Adam，余弦退火 |

### 6.3 运行命令

```bash
# 主实验（全 50 人，预计运行 6-10 小时）
python train.py --config configs/experiment/xwstroke_dann_loso.yaml

# 子集快速验证（预计 1-2 小时）
python train.py --config configs/experiment/xwstroke_dann_subset15.yaml

# 覆盖域策略
python scripts/run_dann_experiment.py \
    --config configs/experiment/xwstroke_dann_loso.yaml \
    --domain-grouping stroke_location_2class

# 超参数网格搜索（18 组配置）
python scripts/run_dann_experiment.py \
    --config configs/experiment/xwstroke_dann_loso.yaml \
    --grid-search
```

---

## 7. 预期结果与判读标准

### 7.1 成功标准

| 优先级 | 标准 | 意义 |
|:---:|:---|:---|
| **P0** | DANN Full50 > 53%（vs 基线 51.46% 提升 ≥ 1.5%）| 证明域自适应有效 |
| **P1** | stroke_location_2class > stroke_location_4class | 简化域标签更有效，暗示域间边界模糊 |
| **P2** | Subcortical 域内被试的跨域迁移显著优于 Cortical | 验证"皮层下病灶更可控"的假设 |
| **P3** | domain_loss 随训练下降，task_loss 同步下降 | 证明域对抗和任务学习没有冲突 |

### 7.2 对照实验设计

为严格评估 DANN 的贡献，建议运行以下对照：

| 对照 | 配置修改 | 目的 |
|:---|:---|:---|
| **Baseline** | `model: ADFCNN`, `trainer: StreamingLOSOTrainer` | 原始 Aff+Align 基线 |
| **DANN (本实验)** | `model: DANNADFCNN`, `trainer: DANNStreamingLOSOTrainer` | 域自适应 |
| **Ablation: λ=0** | `domain_loss_weight: 0` | 验证域损失本身的贡献 |
| **Ablation: 随机域标签** | 修改 `domain_info.py` 为随机分配 | 验证域标签的生理学意义 |

### 7.3 结果判读框架

**如果 DANN 显著优于基线 (>53%)**：
- 核心结论：病灶位置是跨被试异质性的主要来源
- 下一步：尝试更精细的域分组（如结合病灶位置 + NIHSS）

**如果 DANN 与基线持平 (~51-52%)**：
- 可能原因：
  1. 域标签粒度不够（需要 sub-domain 或连续域表示）
  2. 域间差异非线性，简单的对抗对齐不足
  3. 训练数据量不足以支撑域判别器学习
- 下一步：尝试 **DeepCORAL**（协方差对齐）或 **MMD** 最小化

**如果 DANN 劣于基线 (<50%)**：
- 可能原因：域对抗信号过强，破坏了任务相关特征
- 下一步：降低 `domain_loss_weight`（如 0.1-0.5），或使用动态加权

---

## 8. 局限性与后续扩展

### 8.1 当前局限

1. **离散域标签**：DANN 假设域是离散类别。但 stroke 患者的异质性可能是连续的（如病灶体积、CST 损伤程度）。
2. **固定 λ 调度**：所有被试共享同一 λ 曲线，未考虑个体进度差异。
3. **单一域维度**：只使用病灶位置，未联合 NIHSS/病程等多维域信息。

### 8.2 扩展方向

| 方向 | 方法 | 预期改进 |
|:---|:---|:---|
| **多域 DANN** | 多个并行的 domain discriminator（病灶位置 + NIHSS + 病程） | 更全面地消除域差异 |
| **条件 DANN** | 将临床特征作为条件输入到 domain discriminator | 利用域内结构信息 |
| **Wasserstein DANN** | 用 Wasserstein 距离替代对抗分类 | 更稳定的训练，尤其对小样本域 |
| **DANN + 预训练** | 在 Scheme C 子集上预训练 DANN，全人群微调 | 源域选择 + 域自适应联合优化 |

---

## 9. 文件清单

| 文件 | 说明 |
|:---|:---|
| `models/DANNADFCNN.py` | DANN 模型（backbone + classifier + domain_discriminator + GRL） |
| `trainers/dann_trainer.py` | DANN Streaming LOSO Trainer |
| `utils/domain_info.py` | 域标签解析工具（多种分组策略） |
| `configs/experiment/xwstroke_dann_loso.yaml` | 全 50 人 DANN 实验配置 |
| `configs/experiment/xwstroke_dann_subset15.yaml` | 子集 15 快速验证配置 |
| `scripts/run_dann_experiment.py` | 运行脚本（支持参数覆盖和网格搜索） |
| `docs/DANN_DESIGN_REPORT.md` | 本设计报告 |

---

## 10. 参考文献

1. Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." *Journal of Machine Learning Research*, 17(59), 1-35.
2. He, H., & Wu, D. (2020). "Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach." *IEEE TNSRE*.
3. 本项目分层分析结果：`results/stratified/STRATIFIED_ANALYSIS_REPORT.md`
