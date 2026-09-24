# 审稿回复补充实验 — 结果报告

> 对应论文：The Limits of Clinical Metadata for Cross-Subject Stroke Motor Imagery Decoding
> 对应计划：`supplementary_experiments_plan.md` / `review_response_guide.md`
> 创建日期：2026-08-27
> 状态：进行中（FDR 已完成，其余实验待确定后补充）

---

## 结果总览

| 实验 | 状态 | 关键结果 |
|------|------|---------|
| FDR 多重比较校正 | ✅ 已完成 | 6 个 p 值 BH 校正后**均不显著**（最小 q = 0.069），与计划预期不符，回复文字需相应改写 |
| 多 K 值源选择敏感性分析 | ✅ 已完成 | K∈{5,…,49} 准确率 49.2%–50.7%，无 K 优于全源；距离-准确率相关性均不显著 → 负面结果对 K 稳健 |
| 额外 Backbone 验证（ShallowConvNet + ADFCNN） | ✅ 已完成 | 三 backbone × 4 pipeline 对照：准确率均 49%–53%，Aff pipeline 一致占优，oracle 差距 3.6–4.6pp → 核心结论跨架构稳健 |

---

## 实验三：FDR 多重比较校正（已完成）

### 方法

对论文 Table 4 中 6 个探索性病灶位置比较的 p 值应用 Benjamini–Hochberg FDR 校正（`statsmodels.stats.multitest.multipletests`，`method='fdr_bh'`，α=0.05，m=6）。

### 结果

| 比较 | 原始 p 值 | BH 校正 q 值 | α=0.05 显著？ |
|------|----------|-------------|---------------|
| NIHSS vs Aff benefit | 0.509 | 0.844 | 否 |
| NIHSS vs EA benefit | 0.942 | 0.942 | 否 |
| Chronic vs acute (Aff) | 0.563 | 0.844 | 否 |
| Chronic vs acute (EA) | 0.763 | 0.916 | 否 |
| Subcortical vs other (Aff) | 0.013 | **0.069** | **否** |
| Subcortical vs other (EA) | 0.023 | **0.069** | **否** |

### ⚠️ 与计划/回复指南预期的偏差（重要）

`supplementary_experiments_plan.md` §4.4 和 `review_response_guide.md` vf3F #2 的拟稿均预期 subcortical 两项 "q≈0.039/0.046，仍显著"——**该预期有误，不能写进论文或回复信**。按 BH 程序，最小 q 值为 0.023×6/2 = 0.069，校正后两项均不显著。

### 对论文/回复信的建议表述

- 校正后两个 subcortical 关联不再达到 α=0.05（q≈0.07），但它们仍是全部 6 项比较中 q 值最小的，且效应量可观（d = 0.749 / 0.679）。
- 这与论文既有的 "hypothesis-generating rather than confirmatory" 定位完全一致，也呼应了两位审稿人对小样本亚组（cortical n=4）的关切，客观上强化了"临床元数据指导能力有限"的主线结论。
- 回复信中 "The subcortical associations remain significant after correction" 的表述必须删除，建议改为：

> We applied Benjamini–Hochberg FDR correction to the six exploratory lesion-location comparisons. The two subcortical associations, which remain the smallest q-values (both q = 0.069) with moderate-to-large effect sizes (d = 0.749 and 0.679), no longer reach α = 0.05 after correction; all other associations remain non-significant. This is consistent with our framing of these associations as hypothesis-generating, and we have made the corrected q-values explicit in the revised manuscript.

### 复现代码

```python
from statsmodels.stats.multitest import multipletests

pvals = [0.509, 0.942, 0.563, 0.763, 0.013, 0.023]
names = ['NIHSS vs Aff benefit', 'NIHSS vs EA benefit',
         'Chronic vs acute (Aff)', 'Chronic vs acute (EA)',
         'Subcortical vs other (Aff)', 'Subcortical vs other (EA)']
rejected, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
for n, p, q, r in zip(names, pvals, qvals, rejected):
    print(f"{n:28s} p={p:.3f} -> q={q:.3f}  significant={r}")
```

---

## 实验一：多 K 值源选择敏感性分析（已完成）

> 计划见 `supplementary_experiments_plan.md` §二。实际执行规模按审稿指南选项 A：K ∈ {5, 10, 15, 20, 25} + 复用现有 K=49 全源基线，EEGNet + LR 标签，LR+noEA / LR+EA 两条 pipeline，LOSO 50 目标被试。
> 执行日期：2026-08-27。配置由 `scripts/generate_multik_source_selection_configs.py` 生成，跑批脚本 `scripts/run_multik_source_selection.sh`，汇总脚本 `scripts/aggregate_multik_source_selection.py`。

### 结果（EEGNet，LOSO 50 被试平均准确率）

| K | LR+noEA | LR+EA |
|---|---------|-------|
| 5 | 50.19% ± 4.93% | 50.55% ± 5.39% |
| 10 | 49.73% ± 4.78% | 50.24% ± 5.70% |
| 15 | 49.85% ± 5.65% | 50.20% ± 5.54% |
| 20 | 49.62% ± 5.64% | 50.68% ± 5.29% |
| 25 | 49.16% ± 5.09% | 49.88% ± 6.06% |
| 49（全源基线） | 49.40% ± 4.97% | 50.49% ± 6.26% |

### 临床距离-准确率相关性（每个目标被试到所选 K 个源的平均临床距离 vs 测试准确率）

| K | LR+noEA r (p) | LR+EA r (p) |
|---|---------------|-------------|
| 5 | −0.063 (0.664) | −0.102 (0.483) |
| 10 | −0.090 (0.534) | +0.082 (0.570) |
| 15 | +0.207 (0.149) | +0.065 (0.654) |
| 20 | +0.047 (0.748) | −0.039 (0.786) |
| 25 | +0.114 (0.430) | +0.051 (0.725) |
| 49 | +0.035 (0.810) | +0.233 (0.103) |

### 结论

- 所有 K 值的准确率都落在 49.2%–50.7% 区间，围绕全源基线（49.40% / 50.49%）上下波动 ≤1 pp，**没有任何 K 值系统性地优于使用全部 49 个源**。
- 所有 K 值下临床距离与测试准确率的相关性均接近零且不显著（|r| ≤ 0.21，p ≥ 0.10）。
- 负面结果对 K 值稳健，支持回复 Reviewer m6Dh #3。
- 注：本组实验使用 EEGNet + LR 标签，同时统一了论文原 K=10 结果（50.66%，ADFCNN + AffAlign）的模型口径问题。

### 输出文件

- `results/comparisons/multik_source_selection_summary.csv` — 汇总表（K, pipeline, mean/std, r, p）
- `results/comparisons/multik_source_selection_per_subject.csv` — 逐被试准确率与源距离
- 原始实验目录：`experiments/XWStroke_EEGNet_20260827_{112220,113501,115423,122109,125451,133543,134753,140733,143441,150852}`

---

## 实验二：额外 Backbone 验证（运行中）

> 计划见 `supplementary_experiments_plan.md` §三。经与审稿指南核对，两位审稿人（m6Dh #4、vf3F #4）并未要求补 backbone 实验，此项为加分项。实际执行方案：**ShallowConvNet**（Schirrmeister et al. 2017，FBCSP 启发的经典架构，与 EEGNet 架构差异大）4-pipeline LOSO + **ADFCNN** 协议补齐。

### 执行内容

1. **ShallowConvNet 4-pipeline LOSO**（运行中，2026-08-27 启动）：
   - 模型适配：`models/ShallowConvNet.py` 修复了失效的 `from model.layers` 导入，在 `models/__init__.py` 注册适配器（backbone + LazyLinear 分类头，输出原始 logits 兼容 CrossEntropyLoss；51K 参数）。
   - 配置：`configs/experiment/xwstroke_shallowconvnet_full_loso_{lr_noEA,lr_ea,affalign_noEA,ea_affalign}.yaml`，由 `scripts/generate_altbackbone_loso_configs.py` 生成，超参与 EEGNet 完全一致（Adam, lr=1e-3, wd=1e-2, batch=512, epochs=100, mp=false），复用论文 4 个数据集配置。
2. **ADFCNN 协议补齐**（排队中）：现有 ADFCNN 结果中 LR+EA（51.24% ± 5.10%）和 Aff+EA（52.75% ± 5.64%）协议与论文一致，直接复用；LR+noEA（49.63%）和 Aff+noEA（51.46%）当时使用 augmentation=0.3 + FP16，与论文 noaug 协议不符，已用 `XWStroke_noEA_noaug.yaml` / `XWStroke_affected_aligned_noEA_noaug.yaml` + mp=false 重跑。
   - 顺带明确：论文原 K=10 源选择结果（50.66% ± 3.87%）为 ADFCNN + AffAlign 产生，本次已在实验一中用 EEGNet + LR 重跑统一口径。

<!-- 结果表格待补充（跑批完成后由 scripts/aggregate_altbackbone_loso.py 生成）：
| Pipeline | EEGNet | ShallowConvNet | ADFCNN |
-->

### 结果一：ShallowConvNet vs EEGNet 4-pipeline LOSO（2026-08-28 完成）

| Pipeline | EEGNet | ShallowConvNet | paired t (p) |
|----------|--------|----------------|--------------|
| LR+noEA | 49.40% ± 4.97% | 50.49% ± 5.70% | +1.01 (0.319) |
| LR+EA | 50.49% ± 6.26% | 50.72% ± 5.90% | +0.22 (0.824) |
| Aff+noEA | 49.55% ± 4.84% | 52.21% ± 4.78% | +2.96 (0.005) |
| Aff+EA | 51.32% ± 4.08% | 52.14% ± 4.81% | +0.90 (0.373) |
| **Oracle（逐被试最优 pipeline）** | 55.68% ± 3.81% | 56.85% ± 3.92% | — |

要点：

- **绝对水平可比**：ShallowConvNet 各 pipeline 准确率 50.5%–52.2%，与 EEGNet 处于同一区间，论文的 ~51% 基线不是 EEGNet 特有现象。
- **定性模式一致**：ShallowConvNet 上同样是两个 Aff（解剖学重标定）pipeline 居前（52.21% / 52.14%），LR pipeline 居后——Aff 重标定的相对优势跨架构成立。
- **Oracle 差距一致**：ShallowConvNet oracle 与最佳固定 pipeline 之差 ≈4.6 pp，与 EEGNet 的 ≈4.4 pp 几乎相同——"逐被试存在更优 pipeline、但无法事先预知"的核心现象跨架构复现。
- 唯一显著差异是 Aff+noEA 上 ShallowConvNet 优于 EEGNet（p=0.005），方向上是 backbone 差异，不影响临床元数据相关结论。
- 注：validation-based selection（clinical-nearest / random）在新 backbone 上未重跑——该结论的架构稳健性可由"oracle 差距一致 + pipeline 排序一致"间接支持；如审稿人坚持可后续补跑。

输出文件：`results/comparisons/altbackbone_4pipeline_per_subject.csv`、`results/comparisons/altbackbone_summary.json`；实验目录 `experiments/XWStroke_ShallowConvNet_20260827_{155028,184835,205141,231801}`。

### 结果二：ADFCNN 协议补齐（2026-08-28 完成）

- 补跑完成（noaug + mp=false）：LR+noEA = 49.20% ± 5.66%（原 aug 版 49.63%），Aff+noEA = 51.43% ± 5.00%（原 aug 版 51.46%），与原结果几乎一致。
- 实验目录：`experiments/XWStroke_ADFCNN_20260828_{021138,034959}`。

### 三 Backbone × 4 Pipeline 完整对照表（LOSO 50 被试平均准确率）

| Pipeline | EEGNet | ShallowConvNet | ADFCNN |
|----------|--------|----------------|--------|
| LR+noEA | 49.40% ± 4.97% | 50.49% ± 5.70% | 49.20% ± 5.66% |
| LR+EA | 50.49% ± 6.26% | 50.72% ± 5.90% | 51.24% ± 5.10% |
| Aff+noEA | 49.55% ± 4.84% | 52.21% ± 4.78% | 51.43% ± 5.00% |
| Aff+EA | **51.32% ± 4.08%** | 52.14% ± 4.81% | **52.75% ± 5.64%** |
| Oracle（逐被试最优） | 55.68% ± 3.81% | 56.85% ± 3.92% | 56.36% ± 4.14% |

（ShallowConvNet 的最佳固定 pipeline 为 Aff+noEA 52.21%，Aff+EA 52.14% 紧随其后；EEGNet 与 ADFCNN 的最佳固定 pipeline 均为 Aff+EA。）

### 跨架构稳健性结论（回应 m6Dh #4 + vf3F #4）

1. **绝对水平架构无关**：三个架构差异显著的 backbone（深度可分离卷积 / FBCSP 启发浅层卷积 / 自适应卷积）准确率全部落在 49%–53%，~51% 的基线不是 EEGNet 特有。
2. **Aff 重标定的相对优势跨架构成立**：三个 backbone 上 Aff pipeline 均不劣于对应 LR pipeline，最佳固定 pipeline 均为 Aff 类。
3. **Oracle 差距跨架构一致**：oracle 与最佳固定 pipeline 之差分别为 4.4 / 4.6 / 3.6 pp——"逐被试存在更优 pipeline 但无法事先预知"的核心现象稳定复现，而论文已证明临床元数据无法填补该差距。
4. ADFCNN 的 4 个 pipeline 全部使用与论文一致的协议（noaug、mp=false、相同超参），其中 2 个为本次补跑，2 个复用历史结果。

输出文件：`results/comparisons/three_backbone_4pipeline_summary.csv`、`altbackbone_shallowconvnet_summary.json`、`altbackbone_adfcnn_summary.json` 及对应 per-subject CSV。

