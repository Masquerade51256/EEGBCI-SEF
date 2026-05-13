# XWStroke 任务定义转换计划：从解剖左右到功能侧（健患侧）

> **文档目的**：向另一位开发协作者（KIMIcode）传递当前项目的核心动机、已完成工作、实验发现与下一步方向，以便其继续完善实验代码。
>
> **适用数据集**：XWStroke（玄武卒中数据集，50名卒中患者运动想象EEG）
> **核心问题**：卒中跨被试MI解码中，"左手 vs 右手"标签是否应改为"健侧 vs 患侧"标签？

---

## 1. 项目背景与动机

### 1.1 数据集基本特征

- **数据集**：XWStroke（玄武卒中运动想象EEG数据集）
- **被试**：50名急性/亚急性卒中患者
- **任务**：左手 vs 右手运动想象（原始标签）
- **通道**：32导EEG + 2导EOG（HEOL、HEOR）
- **采样率**：500 Hz，epoch时长4秒
- **被试异质性**：混合左偏瘫（28人）和右偏瘫（22人），NIHSS 1-11，病程1-30月

### 1.2 核心问题：解剖标签 vs 功能标签

卒中患者群体中，**左偏瘫患者**和**右偏瘫患者**的"左手"和"右手"运动想象在神经生理上具有**不同的功能语义**：

| 偏瘫侧 | 左手MI | 右手MI | 功能语义 |
|:---|:---|:---|:---|
| 左偏瘫（如sub-02）| 患侧 | 健侧 | 左手=病理侧，右手=相对正常侧 |
| 右偏瘫（如sub-01）| 健侧 | 患侧 | 左手=相对正常侧，右手=病理侧 |

在跨被试建模（LOSO）中，如果保持"左 vs 右"标签：
- sub-01 的**左手试次**（健侧）和 sub-02 的**左手试次**（患侧）会被归为**同一类**
- 这迫使模型学习"左半球活动模式"，但这两个被试的左半球活动语义完全不同

改为"健侧 vs 患侧"标签后：
- 所有被试的**患侧试次**归为同一类，**健侧试次**归为另一类
- 模型学习的是"与患手运动意图相关的神经模式"，在跨被试层面更具功能一致性

### 1.3 理论依据（文献综述核心结论）

详见 `docs/卒中_MI_任务定义报告_健患侧与左右.md`，核心结论：

1. **卒中后左右对侧模板被破坏**：病灶位置、严重度、是否累及初级运动皮层共同决定运动想象的侧化模式（Park et al. 2016; Fallani et al. 2013）
2. **功能侧标签有助于功能侧对齐**：当跨被试样本混合左/右偏瘫时，健患侧标签可避免"相同功能意图被分到不同类"（Qiu et al. 2017; Lee et al. 2023）
3. **健侧≠正常侧**：卒中后未损伤半球同样发生重组，因此健患侧标签不是万能药，但在混合偏瘫群体中有明确的功能对齐价值（Graziadio et al. 2012）
4. **个体化重组差异大**：比单纯改标签更重要的是配合半球对齐、病灶分层和个体化通道选择（Mansour et al. 2022; Jia et al. 2022）

### 1.4 核心假设

> **H1**：在混合左/右偏瘫的卒中患者跨被试LOSO解码中，将任务定义从"左右手"改为"健患侧手"将提升模型泛化性能。
>
> **H2**：功能侧标签的收益将集中在**低表现患者**和**右偏瘫患者**中，因为这两类群体的解剖左右模板偏离健康状态更严重。

---

## 2. 已完成的工作

### 2.1 代码实现：标签映射机制

已实现**配置驱动的标签映射**，通过 `label_mapping` 参数在数据加载阶段自动转换。

**修改的文件**：
- `core/base/base_dataset.py`：新增 `_remap_labels()` hook，在0-based标签转换后、滑窗前调用
- `data/datasets.py`：`XWStrokeDataset` 和 `XWStrokeEDFDataset` 重载 `_remap_labels()`，自动读取 `participants.tsv` 中的 `ParalysisSide` 进行个体化映射
- `legacy_dataloader/_dataset_XWStroke.py` 和 `legacy_dataloader/dataset_XWStroke.py`：旧框架兼容

**映射逻辑**：
```python
原始标签（0-based）: 0=左手, 1=右手

if label_mapping == "affected":
    # 目标: 0=患侧(affected), 1=健侧(unaffected)
    if 偏瘫侧 == "left":
        # 左(0)=患侧, 右(1)=健侧 → 不变
    else:  # 右偏瘫
        # 左(0)=健侧, 右(1)=患侧 → 翻转
        
elif label_mapping == "unaffected_first":
    # 目标: 0=健侧, 1=患侧
    ...
```

**新增配置**：
- `configs/dataset/XWStroke_affected.yaml`：`label_mapping: "affected"`，`participants_tsv` 路径
- `configs/dataset/XWStroke_subset10_affected.yaml`：10人快速验证子集配置
- `configs/dataset/XWStroke_subset15_affected.yaml`：15人验证子集配置

### 2.2 已完成的实验

#### 实验1：10人快速验证子集（已完成）

**子集选取**：`[4, 6, 19, 24, 28, 36, 39, 44, 47, 49]`，分层抽样（LOSO性能+偏瘫侧+NIHSS）

| 指标 | 任务A (左右手) | 任务B (健患侧) | 判读 |
|:---|:---|:---|:---|
| **Overall Mean** | 46.80% ± 5.79% | **52.50% ± 4.24%** | **+5.70%** |
| **配对 t 检验** | — | — | **t=1.896, p=0.0905**（边缘显著） |
| **Cohen's d** | — | — | **0.600**（中等效应量） |
| **改善人数** | — | — | **7/10 (70%)** |

**关键发现**（详见 `results/comparisons/ANALYSIS_REPORT.md`）：

1. **右偏瘫患者受益更多**（+9.5% vs +3.2%）
2. **轻度NIHSS患者受益最多**（+10.2%）
3. **低表现者全部被抬升**（+14.2%，4/4改善）
4. **高表现者反而略降**（-2.4%）

**生成的图表**（位于 `results/comparisons/`）：
- `fig_scatter_comparison.png`：散点图对比
- `fig_bar_comparison.png`：逐人条形对比
- `fig_histogram_diff.png`：提升分布
- `fig_stratified_paralysis.png`：按偏瘫侧分层箱线图
- `fig_stratified_nihss.png`：按NIHSS分层箱线图

#### 实验2：全50人验证（配置就绪，待执行）

配置已准备：
- `configs/experiment/xwstroke_full_loso_lr.yaml` — 任务A，StreamingLOSOTrainer
- `configs/experiment/xwstroke_full_loso_affected.yaml` — 任务B，StreamingLOSOTrainer
- `scripts/run_full50_loso_comparison.py` — 一键运行脚本

---

## 3. 期待的实验结果与解读方向

### 3.1 对全50人验证的期待

基于10人子集的边缘显著趋势（p=0.0905, d=0.600），期待50人全量验证能：

1. **将 p 值压到 <0.05**：样本量从10→50，统计功效大幅提升
2. **保持或放大效应量**：Cohen's d 维持在 0.5-0.8 区间
3. **确认低表现者被系统性抬升**：在50人中验证"原始LOSO低表现者在功能侧标签下显著改善"的模式
4. **确认方差降低**：被试间标准差在任务B中应持续小于任务A

### 3.2 结果解读框架

无论最终均值是否显著提升，以下观察均有价值：

| 观察模式 | 科学含义 |
|:---|:---|
| 任务B均值 > 任务A | 支持H1：功能侧标签改善跨被试泛化 |
| 任务B std < 任务A | 支持H1：功能侧标签降低被试间异质性 |
| 低表现者系统性抬升 | 功能侧对齐帮助了最难解码群体（核心发现） |
| 右偏瘫 > 左偏瘫受益 | 右偏瘫群体左右模板偏离更严重 |
| 轻度NIHSS > 重度受益 | 保留一定侧化能力的患者更受益于功能侧对齐 |
| 高表现者反而下降 | 功能侧标签不是普适提升，对已有良好侧化的患者无益 |

### 3.3 两种可能的论文结论

**结论A（强结论）**：
> "在急性卒中跨被试MI解码中，将任务定义从解剖学左右手改为功能学健患侧手，显著提升了模型泛化性能（+X%, p<0.05）。"

**结论B（更稳健的 nuanced 结论）**：
> "功能侧重定义不是普适替代，而是针对特定卒中亚群（低表现者、右偏瘫患者、轻中度NIHSS）的条件性优化方案。"

---

## 4. 下一步实验方案（优先级排序）

### P0：全50人验证（正在进行）

- **目标**：在完整数据集上复现10人子集的发现
- **配置**：`configs/experiment/xwstroke_full_loso_*.yaml`
- **执行**：`python scripts/run_full50_loso_comparison.py`
- **预期时间**：8-12小时（StreamingLOSO）

### P1：对照实验设计（与全50人并行规划）

如果全50人验证呈正结果，建议补充以下对照实验以排除混杂变量：

| 对照实验 | 目的 | 方法 |
|:---|:---|:---|
| **通道空间翻转** | 验证"标签+空间对齐"联合效应 | 将患侧半球统一映射到同一伪侧后重新训练 |
| **随机标签置换** | 排除标签重定义本身的统计假象 | 随机打乱患/健映射，验证提升是否消失 |
| **患侧 vs 静息** | 对比第三种任务定义 | 提取基线窗口构建患侧/静息二分类 |
| **多模型验证** | 确认结果不是模型特异的 | 在 EEGNet / XWFilterBankNet 上重复LOSO对照 |

### P2：分层深化分析

- **病灶位置分层**：皮层 vs 皮层下病灶（依据 `StrokeLocation` 字段）
- **病程分层**：急性（≤3月）vs 慢性（>3月）
- **半球对齐分析**：计算 ipsilesional/contralesional 通道的ERD差异，验证功能侧标签是否更好地对齐了这些差异

### P3：扩展到其他数据集

- **LowerStroke数据集**：验证功能侧标签在下肢MI任务中是否同样有效
- **健康人数据集（如BCICIV2a）**：作为阴性对照，健康人不应从健患侧标签中受益

### P4：工程化改进

- **个体化标签**：不是全局改为健/患，而是按患者实际重组模式动态选择标签（ ipsilesional dominant / contralesional dominant / bilateral）
- **自适应通道选择**：在健患侧标签基础上，结合个体化ERD地形图进行通道筛选

---

## 5. 关键文件索引

### 代码修改
| 文件 | 修改内容 |
|:---|:---|
| `core/base/base_dataset.py` | 新增 `_remap_labels()` hook |
| `data/datasets.py` | `XWStrokeDataset` / `XWStrokeEDFDataset` 实现患健侧映射 |
| `trainers/loso_trainer.py` | 修复 `torch.Generator(device='cpu')` bug |
| `legacy_dataloader/_dataset_XWStroke.py` | 旧框架兼容 |
| `legacy_dataloader/dataset_XWStroke.py` | 旧框架兼容 |

### 配置文件
| 文件 | 用途 |
|:---|:---|
| `configs/dataset/XWStroke.yaml` | 标准左右手配置（50人） |
| `configs/dataset/XWStroke_affected.yaml` | 患健侧配置（50人） |
| `configs/dataset/XWStroke_subset10_affected.yaml` | 10人快速验证子集 |
| `configs/dataset/XWStroke_subset15_affected.yaml` | 15人验证子集 |
| `configs/experiment/xwstroke_full_loso_lr.yaml` | 全50人任务A（Streaming LOSO） |
| `configs/experiment/xwstroke_full_loso_affected.yaml` | 全50人任务B（Streaming LOSO） |

### 脚本工具
| 文件 | 用途 |
|:---|:---|
| `scripts/verify_label_mapping.py` | 验证标签映射逻辑正确性 |
| `scripts/analyze_subject_selection.py` | 15人子集选取分析 |
| `scripts/analyze_subject_selection_10.py` | 10人子集选取分析 |
| `scripts/run_full_analysis.py` | **通用分析脚本**：统计检验+可视化+分层分析+报告生成 |
| `scripts/run_full50_loso_comparison.py` | **全50人一键运行**：训练+对比+自动分析 |
| `scripts/run_subset10_loso_comparison.py` | 10人一键运行 |
| `scripts/run_subset15_loso_comparison.py` | 15人一键运行 |

### 文档
| 文件 | 用途 |
|:---|:---|
| `docs/卒中_MI_任务定义报告_健患侧与左右.md` | 完整文献综述与理论分析 |
| `docs/experiment_plan_subset15_loso.md` | 15人LOSO试验方案 |
| `docs/TASK_DEFINITION_TRANSITION_PLAN.md` | **本文档**：传递给协作者的完整计划 |

### 数据与结果
| 文件 | 用途 |
|:---|:---|
| `src/datasets/21679035/participants.tsv` | 被试临床信息（含ParalysisSide） |
| `results/comparisons/subject_subset_10.csv` | 10人子集详细信息 |
| `results/comparisons/subject_subset_15.csv` | 15人子集详细信息 |
| `results/comparisons/ANALYSIS_REPORT.md` | 10人分析完整报告 |
| `results/comparisons/fig_*.png` | 可视化图表 |

---

## 6. 对协作者（KIMIcode）的期待

请基于以上信息，协助完善以下实验代码：

1. **通道空间翻转实验**：在 `data/datasets.py` 中实现 hemisphere alignment（按偏瘫侧镜像翻转通道），配合 `label_mapping` 测试联合效应
2. **随机标签置换对照**：在 `scripts/` 中添加 permutation test 脚本，验证统计显著性
3. **个体化标签实验**：基于 `participants.tsv` 中的临床信息，实现按病灶位置/严重度动态选择标签的策略
4. **多模型LOSO批量运行**：扩展一键运行脚本，支持在 EEGNet / XWFilterBankNet 上自动批量跑LOSO对照
5. **自适应通道选择**：结合个体化ERD地形图，在LOSO训练前为每个被试选择最优通道子集

如有任何关于标签映射逻辑、实验设计或数据结构的疑问，请参考 `data/datasets.py` 中的 `_remap_labels` 实现和 `docs/卒中_MI_任务定义报告_健患侧与左右.md` 中的文献分析。

---

*文档版本：v1.0*
*创建时间：2026-05-08*
*数据集：XWStroke (N=50)*
*当前状态：10人验证完成，50人验证配置就绪待执行*
