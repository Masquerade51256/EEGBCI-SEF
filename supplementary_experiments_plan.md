# 审稿回复补充实验方案

> 对应论文：The Limits of Clinical Metadata for Cross-Subject Stroke Motor Imagery Decoding
> 创建日期：2026-08-27
> 负责人：代码项目组同事

---

## 一、实验背景与目标

三位审稿人（M6SY、m6Dh、vf3F）对论文提出了修改意见。其中大部分意见已通过文字修改回应，但以下**两项核心实验**需要补充数据支持：

1. **多 K 值源选择敏感性分析**（Reviewer m6Dh #3）
2. **额外 Backbone 验证**（Reviewer m6Dh #4 + vf3F #4）

---

## 二、实验一：多 K 值源选择敏感性分析

### 2.1 实验目的

验证 "clinical-similarity source selection" 的负面结果是否对 K 值（选择的最近源被试数量）敏感。

### 2.2 实验设计

| 项目 | 说明 |
|------|------|
| **基线** | 当前论文使用 K=10，准确率 50.66% |
| **测试 K 值** | K ∈ {5, 10, 15, 20, 25, 30, 40, 49} |
| **临床距离度量** | 与论文一致：d(i,j) = c_loc + c_dur + c_NIHSS |
| **标签策略** | 使用 LR（原始左右手标签）作为基准 |
| **对齐方式** | 使用 LR+noEA 和 LR+EA 两种 pipeline |
| **评估方式** | LOSO（50 个目标被试），报告平均准确率 |

### 2.3 操作步骤

1. 复用现有的 EEGNet 训练代码和预训练权重
2. 对每个 K 值：
   a. 对每个目标被试，计算其与所有源被试的临床距离 d(i,j)
   b. 选择距离最小的 K 个源被试
   c. 在这 K 个源被试上训练 EEGNet（LR+noEA 和 LR+EA）
   d. 在目标被试上测试
3. 汇总 50 个目标被试的平均准确率

### 2.4 预期结果

- 所有 K 值的准确率应在 49%~51% 范围内
- 不应有任一 K 值显著优于使用全部 49 个源（51.46%）
- 临床距离与最终准确率的相关性应保持接近零（r ≈ 0）

### 2.5 输出格式

请输出一个 CSV 文件：

```csv
K,pipeline,mean_accuracy,std_accuracy,clinical_distance_r,p_value
5,LR+noEA,...
5,LR+EA,...
10,LR+noEA,...
...
```

---

## 三、实验二：额外 Backbone 验证（XWFilterBankNet）

### 3.1 实验目的

验证论文核心结论（临床元数据不能指导迁移）是否在不同架构下保持一致。

### 3.2 模型选择理由

- **XWFilterBankNet**（又称 FBCNet）已在现有代码框架中实现
- 与 EEGNet 架构差异显著：频带分解 + 空间注意力 vs 深度可分离卷积
- 审稿人 vf3F 明确提到了 FBCNet

### 3.3 实验设计

| 项目 | 说明 |
|------|------|
| **Backbone** | XWFilterBankNet（如有困难可用 ShallowConvNet/DeepConvNet 替代） |
| **Pipeline** | 4 种：LR+noEA, Aff+noEA, LR+EA, Aff+EA |
| **评估方式** | LOSO（50 个目标被试） |
| **超参数** | 尽量与 EEGNet 一致：Adam, lr=1e-3, wd=1e-2, batch=512, epochs=100, early stopping patience=20 |

### 3.4 操作步骤

1. 确认 XWFilterBankNet 在代码框架中的可用性（`models/` 目录）
2. 配置 YAML 实验文件：`configs/experiment/xwstroke_xwfilterbanknet.yaml`
3. 运行 4-pipeline LOSO 实验：
   - LR+noEA
   - Aff+noEA
   - LR+EA
   - Aff+EA
4. 额外运行：
   - Clinical-nearest validation-based selection
   - Random validation (5 seeds)
   - Oracle upper bound

### 3.5 预期结果

- 最佳固定 pipeline 准确率应在 50%~54% 范围内（与 EEGNet 的 51.32% 可比）
- Clinical-nearest validation 应低于最佳固定 pipeline
- Validation-test 相关性应接近零

### 3.6 输出格式

请输出以下文件：

1. `xwfnet_4pipeline_loso.csv` — 逐被试 4-pipeline 准确率
2. `xwfnet_validation_selection.csv` — validation-based selection 结果
3. `xwfnet_summary.json` — 汇总统计（均值、标准差、paired t-test 结果）

---

## 四、实验三：FDR 多重比较校正（计算类，无需重跑实验）

### 4.1 实验目的

回应 Reviewer vf3F #2 对未校正 p 值的关切。

### 4.2 操作步骤

对探索性子组比较（lesion-location comparisons）应用 Benjamini-Hochberg FDR 校正：

涉及的 p 值（来自论文 Table 4）：
- NIHSS vs Aff benefit: p = 0.509
- NIHSS vs EA benefit: p = 0.942
- Chronic vs acute (Aff): p = 0.563
- Chronic vs acute (EA): p = 0.763
- Subcortical vs other (Aff): p = 0.013
- Subcortical vs other (EA): p = 0.023

共 6 个 p 值。

### 4.3 计算方法

使用 Python 的 `statsmodels` 库：

```python
from statsmodels.stats.multitest import multipletests

pvals = [0.509, 0.942, 0.563, 0.763, 0.013, 0.023]
rejected, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

for p, q, r in zip(pvals, qvals, rejected):
    print(f"p={p:.3f} -> q={q:.3f}, significant={r}")
```

### 4.4 预期结果

- Subcortical vs other (Aff): q ≈ 0.039，仍显著
- Subcortical vs other (EA): q ≈ 0.046，仍显著
- 其他 4 个：q > 0.05，不显著

### 4.5 输出

请将计算结果填入论文修改文档的对应位置（Section 4.4 Training and Evaluation）。

---

## 五、优先级与时间节点

| 优先级 | 实验 | 预计工作量 | 截止日期 |
|--------|------|-----------|---------|
| 🔴 P0 | FDR 校正计算 | 10 分钟 | 立即可做 |
| 🔴 P0 | 多 K 值源选择 | 1~2 小时脚本 | 3 天内 |
| 🟡 P1 | XWFilterBankNet LOSO | 半天~1 天 | 1 周内 |

---

## 六、联系方式

如有任何技术问题（如代码框架配置、模型加载、数据路径等），请联系论文第一作者。
