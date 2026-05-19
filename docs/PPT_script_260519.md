# PPT 讲稿 — 组会汇报 2026-05-19

**汇报时长：** 15分钟  
**汇报语言：** 建议英文（讲稿提供中英文对照）

---

## Slide 1: Title

**中文：**
各位老师好，今天我汇报的题目是《急性脑卒中患者跨被试MI-EEG解码的临床异质性与自适应策略》，基于XWStroke数据集（50例急性卒中患者）的系统实验验证。

**English:**
Good morning everyone. Today I will present my research on "Clinical Heterogeneity and Adaptive Decoding for Cross-Subject MI-EEG in Acute Stroke," based on systematic experimental validation using the XWStroke dataset of 50 acute stroke patients.

---

## Slide 2: Contents

**中文：**
我的汇报分为四个部分：背景介绍、研究现状与文献对标、初步实验结果、以及未来的研究方向。

**English:**
My presentation is organized into four parts: background, research status and benchmarking, preliminary results, and future directions.

---

## Slide 3: Background (Section Divider)

**中文：**
首先介绍研究背景。

**English:**
Let me start with the background.

---

## Slide 4: Stroke Heterogeneity

**中文：**
脑卒中患者的运动想象EEG解码面临一个根本挑战：患者之间高度异质。病灶位置不同——可能是皮层、皮层下、脑干或小脑；NIHSS评分从轻度到重度不等；瘫痪侧有左有右；病程也有急慢性之分。这些临床差异直接造成了EEG模式的不同。我们提出的核心观点是：异质性不是噪声，而是尚未被利用的结构化临床信息。当前解码器的问题恰恰在于——它们把这些差异当作需要抹除的噪声，而不是可以加以利用的信号。

**English:**
Motor imagery EEG decoding in stroke patients faces a fundamental challenge: extreme heterogeneity across patients. Lesion locations vary — cortical, subcortical, brainstem, or cerebellar. NIHSS severity ranges from mild to severe. Paralysis side differs. Disease duration varies. These clinical differences directly create different EEG patterns. Our core argument is: heterogeneity is NOT noise — it is structured clinical information that current decoders ignore. The problem with existing decoders is that they treat these differences as noise to be erased, rather than signals to be exploited.

---

## Slide 5: XW Stroke Dataset

**中文：**
我们使用的是XWStroke数据集，2024年发表于Scientific Data。包含50例急性卒中患者，发病1到30天，32导EEG加2导EOG，500Hz采样，4秒epoch。其中13例存在较严重的伪迹。我们设计了三个标签体系：Task A是标准的左右手分类；Task B是患侧/健侧，这是临床更有意义的标签；Task C是在Task B基础上做半球对齐，把空间坐标映射到病灶侧。评估采用严格的Streaming LOSO——每个fold训练49人、测试1人，零数据泄漏。

**English:**
We use the XWStroke dataset, published in Scientific Data in 2024. It contains 50 acute stroke patients, 1 to 30 days post-stroke, with 32 EEG plus 2 EOG channels at 500 Hz, 4-second epochs. Thirteen subjects have severe artifacts. We designed three labeling schemes: Task A is standard left/right motor imagery; Task B is affected/unaffected side, which is clinically meaningful; Task C adds hemisphere alignment to remap spatial coordinates to the lesion side. Evaluation uses strict Streaming LOSO — train on 49 subjects, test on 1, zero data leakage.

---

## Slide 6: Research Status (Section Divider)

**中文：**
接下来进入第二部分：研究现状与文献对标。

**English:**
Now I move to Part 2: Research Status and Benchmarking.

---

## Slide 7: Core Method on XW Dataset (6 only)

**中文：**
我们对这个数据集做了严格的文献核验，发现一件令人惊讶的事：声称使用了Liu24数据集的论文很多，但真正经过严格确认的，只有7篇。其中核心方法论文只有4篇。这说明这个数据集的研究生态几乎是一片空白，我们的工作属于早期探索。

**English:**
We conducted a rigorous literature audit and found something surprising: while many papers claim to use the Liu24 dataset, only 7 papers are strictly confirmed to actually use it. Among them, only 4 are core method papers. This means the research ecosystem for this dataset is almost a blank slate — our work is early-stage exploration.

---

## Slide 8: Core Method on XW Dataset (LOSO)

**中文：**
更关键的是协议问题。Bun25报告了97.43%的准确率，但他们的LOSO定义是"50次划分"——这很可能是50次随机划分并池化数据，不是严格的逐被试留出。与此对比，Wan26h使用固定的被试划分LOSO，结果为66.56%，这才是更公平的基准。我们的51.46%是在最严格的Streaming LOSO下获得的，基线自然最低。差距不是算法差距，而是协议差距。

**English:**
More critical is the protocol issue. Bun25 reports 97.43% accuracy, but their "LOSO" is defined as "50 divisions" — likely 50 random splits with pooled data, not strict per-subject holdout. In contrast, Wan26h uses fixed-subject LOSO and reports 66.56%, which is the fair benchmark. Our 51.46% is under the strictest Streaming LOSO, so the baseline is naturally lower. The gap is a protocol gap, not an algorithm gap.

---

## Slide 9: Core Method on XW Dataset

**中文：**
对比这4篇核心方法论文，我们发现一个共同点：没有人引入临床变量。大家都在做分类分数追逐——改进网络结构、换更复杂的模型，但都不回答一个基本问题：这个患者为什么能被正确分类？我们的核心洞察是：当前文献的真正空白不是"再提高5%准确率"，而是"把患者临床特征引入建模过程"。

**English:**
Comparing these 4 core method papers, we find a common pattern: none introduce clinical variables. Everyone is chasing classification scores — improving network architectures, using more complex models — but no one answers the basic question: why was this patient classified correctly? Our key insight is: the real gap in current literature is not "another 5% accuracy," but "bringing patient clinical characteristics into the modeling process."

---

## Slide 10: Preliminary Results (Section Divider)

**中文：**
第三部分：初步实验结果。

**English:**
Part 3: Preliminary Results.

---

## Slide 11: Affected Label Convert: Full-Cohort Baseline

**中文：**
全人群50人的基线结果是：Task A 49.63%，Task B 50.29%，Task C 51.46%。Task C相比A提升了1.83%，p值为0.062，接近显著。Cohen's d是0.270，小至中等效应量。这个提升被高被试间方差掩盖了——但趋势是存在的，方向是积极的。

**English:**
The full-cohort baseline results are: Task A 49.63%, Task B 50.29%, and Task C 51.46%. Task C improves over A by 1.83%, with p = 0.062, trending toward significance. Cohen's d is 0.270, a small-to-medium effect. This improvement is masked by high inter-subject variance — but the trend exists and the direction is positive.

---

## Slide 12: Affected Label Convert: Stratified Analysis

**中文：**
当我们做分层分析时，故事变得清晰了。亚皮层病灶患者从半球对齐中受益最多，提升3.6%；而皮层病灶患者反而受损3.6%。原因是：半球对齐假设运动皮层环路是完整的，但皮层损伤直接破坏了这一假设。慢性期患者提升2.9%，基线表现差的患者提升高达7.1%。这说明标签改进不是对所有人都有效——它只对"损伤较轻、病灶较深"的患者有效。

**English:**
When we perform stratified analysis, the story becomes clear. Subcortical patients benefit most from hemisphere alignment with +3.6% improvement, while cortical patients are harmed by -3.6%. The reason: hemisphere alignment assumes intact motor circuits, but cortical damage directly violates this assumption. Chronic patients improve by 2.9%, and low-baseline patients improve by up to 7.1%. This shows label improvement is not universal — it only works for patients with "less damage, deeper lesions."

---

## Slide 13: Subset Selection

**中文：**
我们做了两种子集选择。CleanC子集14人，筛选条件是：无伪迹、亚皮层病灶、NIHSS不超过7。它意外地排除了全部13个伪迹被试。Task C在这里达到52.29%，提升1.72%。CleanC的作用是机制验证——回答"在理想条件下方法是否有效"。另一种是Subset-20代表性子集，按病灶、NIHSS、瘫痪侧、病程、伪迹分层抽样，比例与全人群几乎一致。它的作用是快速迭代——在保留异质性的前提下把实验时间缩短60%。两者回答的是完全不同的问题。

**English:**
We designed two types of subsets. The CleanC subset has 14 subjects, selected as artifact-free, subcortical, and NIHSS ≤ 7. It accidentally excluded all 13 artifact subjects. Task C reaches 52.29% here, improving by 1.72%. CleanC serves for mechanism validation — answering "does the method work under ideal conditions?" The Subset-20 is a stratified representative sample, matching the full cohort's proportions across lesion, NIHSS, paralysis side, duration, and artifact presence. It serves for fast iteration — cutting experiment time by 60% while preserving heterogeneity. They answer fundamentally different questions.

---

## Slide 14: Domain-Adversarial

**中文：**
我们还尝试了域对抗神经网络DANN，动机是学习跨被试的域不变特征。但在CleanC上的结果是：Baseline 50.57%，DANN dlw=0.3时50.14%——没有提升；dlw=1.0时反而降到48.75%。诊断发现：用stroke_location做域标签在概念上就是错的——病灶位置本身就和MI模式耦合。DANN的域不变目标与临床解码目标根本矛盾。这个阴性结果反而坚定了我们的方向：不是抹除差异，而是利用差异。

**English:**
We also tried Domain-Adversarial Neural Networks, with the motivation of learning domain-invariant features across subjects. But on CleanC, the results are: Baseline 50.57%, DANN with dlw=0.3 at 50.14% — no improvement; and dlw=1.0 drops to 48.75%. The diagnosis: using stroke location as domain label is conceptually wrong — lesion location itself correlates with MI patterns. DANN's domain-invariance objective is fundamentally at odds with clinical decoding. This negative result actually strengthened our direction: don't erase differences, exploit them.

**建议此页放的图：** `fig_dann_results.png`

---

## Slide 15: Future Directions (Section Divider)

**中文：**
最后一部分：未来研究方向。

**English:**
Finally, Part 4: Future Directions.

---

## Slide 16: Clinical Adaptive Framework

**中文：**
基于以上发现，我们提出了临床自适应框架，核心原则是：利用临床分层而不是抹除它。有两个具体方案。方案一：动态源域选择——对每个测试患者，按临床相似度（瘫痪侧、NIHSS分桶、病程）匹配训练被试，在匹配子集上训练个性化模型。方案二：条件特征提取器——把NIHSS、病灶位置、病程等临床元数据作为辅助输入，用它们来门控 backbone 的特征通路。优势在于可解释性和临床可操作性——模型的行为可以追溯到患者特征。

**English:**
Based on these findings, we propose the Clinical Adaptive Framework. Core principle: exploit clinical stratification instead of erasing it. Two concrete approaches. Approach 1: Dynamic Source-Domain Selection — for each test patient, match training subjects by clinical similarity (paralysis side, NIHSS bucket, duration), and train a personalized model on the matched subset. Approach 2: Conditional Feature Extractor — feed clinical metadata such as NIHSS, lesion location, and duration as auxiliary inputs to gate the backbone's feature pathways. The advantage is interpretability and clinical actionability — model behavior can be traced back to patient characteristics.

**此页建议内容（不用框图，用简洁文字+对比）：**

```
Clinical Adaptive Framework
Core Principle: Exploit clinical stratification instead of erasing it

Approach 1: Dynamic Source-Domain Selection
   EEG + Clinical Metadata (NIHSS / Lesion / Duration / ParalysisSide)
                    ↓
   Match test patient to clinically similar source subjects
                    ↓
   Train personalized model on matched subset only

Approach 2: Conditional Feature Extractor
   EEG ──→ [Backbone] ──→ [Classifier] ──→ Decoded Output
              ↑
   Clinical Metadata ──→ [Gating Module]

Advantage: Interpretable + Clinically Actionable
```

---

## Slide 17: External Datasets Validation

**中文：**
外部验证方面，我们筛选了三套可立即使用的公开数据集。Thi25有30例越南卒中患者，临床信息最丰富——含NIHSS、病灶、mRS、Oxford肌力分级，最适合验证临床分层是否可泛化。Liu25是27例恢复期患者，下肢gait MI，结构上与Liu24最接近，适合验证跨任务稳健性。Cho21是BCI挑战赛数据集，10例患者，本身就是为跨被试基准设计的。即使只在Thi25上验证成功，我们的工作在这个领域也是独一无二的——因为此前所有基于Liu24的论文都没有报告过跨数据集验证。

**English:**
For external validation, we identified three publicly available datasets. Thi25 has 30 Vietnamese stroke patients with the richest clinical information — NIHSS, lesion, mRS, and Oxford scale — making it ideal for validating whether clinical stratification generalizes. Liu25 has 27 recovery-phase patients with lower-limb gait MI, structurally closest to Liu24, suitable for cross-task robustness validation. Cho21 is a BCI challenge dataset with 10 patients, designed specifically for cross-subject benchmarking. Success on even one of these — especially Thi25 — would make our work unique in this field, as no prior Liu24-based paper has reported cross-dataset validation.

---

## Slide 18: Work Plans

**中文：**
工作计划如下：全人群基线和文献调研已经完成；Subset-20代表性设计已完成；DANN诊断实验已完成并确认放弃；接下来两周的优先事项是：第一，实现并验证临床自适应框架的Approach 1；第二，在Subset-20上快速迭代；第三，开始Thi25外部验证的数据准备。

**English:**
Our work plan is as follows: full-cohort baseline and literature audit are complete; Subset-20 representative design is complete; DANN diagnostics are complete and abandoned. Priorities for the next two weeks: first, implement and validate Approach 1 of the clinical adaptive framework; second, rapid iteration on Subset-20; third, begin data preparation for external validation on Thi25.

---

## Slide 19: Reference

**中文：**
主要参考文献。

**English:**
Key references.

---

## Slide 20: THANKS!

**中文：**
感谢各位老师的聆听，欢迎批评指正。

**English:**
Thank you for your attention. I welcome your questions and comments.

---

## 关键过渡句（备用）

**Background → Research Status:**
- 中文：了解了背景和数据集之后，我们需要回答一个问题：在这个数据集上，现有文献做了什么？做得怎么样？
- English: Having understood the background and dataset, we need to ask: what has existing literature done on this dataset? And how reliable is it?

**Research Status → Preliminary Results:**
- 中文：文献分析告诉我们，现有方法有两个问题——协议不可靠，方向可能错了。那么我们的实验结果支持这个判断吗？
- English: The literature analysis tells us existing methods have two problems — unreliable protocols and potentially wrong direction. Do our experimental results support this judgment?

**Preliminary Results → Future Directions:**
- 中文：DANN的失败告诉我们，追求域不变特征在卒中MI这个问题上是南辕北辙。那么正确的方向是什么？
- English: DANN's failure tells us that pursuing domain-invariant features is the wrong direction for stroke MI. So what is the right direction?

**Clinical Adaptive → External Validation:**
- 中文：如果这个框架在Liu24内部有效，它在外部数据集上还能成立吗？这是我们必须回答的泛化性问题。
- English: If this framework works within Liu24, can it still hold on external datasets? This is the generalization question we must answer.

---

## DANN 部分用图建议

**推荐图：** `docs/figures/fig_dann_results.png`

**理由：**
1. 极简：3根柱子，10米外都能看懂
2. 有Baseline虚线参考（50.57%），一眼看出DANN没超过基线
3. 最右侧深红色柱子（48.75%）直观传达"更差"
4. 图中有注释 "Domain invariance ≠ Clinical decoding"，省去你解释
5. 底部3个标签清晰对应实验配置

**Slide 14 建议布局：**
- 左半：放 `fig_dann_results.png`
- 右半：3个bullet points
  - Baseline: 50.57%
  - DANN (dlw=0.3): 50.14% → No gain
  - DANN (dlw=1.0): 48.75% → Worse
  - Conclusion: Domain invariance erases clinically-relevant differences

---

## Clinical Adaptive Framework 幻灯片内容建议

如果不用框图，Slide 16 建议这样排：

**标题：** Clinical Adaptive Framework

**核心原则（大字号，居中）：**
> Don't erase differences. Exploit clinical stratification.

**左栏：Approach 1 — Dynamic Source Selection**
```
Step 1: Clinical Similarity Matching
   Match by ParalysisSide + NIHSS_bucket + Duration

Step 2: Personalized Training
   Train on matched subset only
```

**右栏：Approach 2 — Conditional Feature Extractor**
```
Step 1: Metadata Encoding
   Embed NIHSS / Lesion / Duration as vector

Step 2: Gated Feature Extraction
   Clinical variables modulate backbone pathways
```

**底部（加粗，红色）：**
> Advantage: Interpretable + Clinically Actionable
> Model behavior traces back to patient characteristics.

这样的排版不需要复杂框图，纯文字+颜色块即可，清晰且学术。
