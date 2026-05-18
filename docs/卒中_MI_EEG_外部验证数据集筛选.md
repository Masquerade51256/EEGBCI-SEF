# 卒中 MI EEG 外部验证数据集筛选

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [结论](#结论)
- [主表 适合在 Liu24 之外做外部验证的数据集](#主表-适合在-liu24-之外做外部验证的数据集)
- [公开数据集的详细筛选](#公开数据集的详细筛选)
  - [Liu25](#liu25)
  - [Cho21](#cho21)
  - [Thi25](#thi25)
- [不应放进主表但值得保留的临床队列](#不应放进主表但值得保留的临床队列)
  - [Man22](#man22)
  - [Seb20b](#seb20b)
  - [Fro17b](#fro17b)
  - [Par16 与 Shu18b](#par16-与-shu18b)
- [不建议作为独立数据集主候选的文献](#不建议作为独立数据集主候选的文献)
- [临床信息覆盖对比](#临床信息覆盖对比)
- [与 Liu24 的相似度判断](#与-liu24-的相似度判断)
  - [最接近 Liu24 的外部验证集](#最接近-liu24-的外部验证集)
  - [临床分层最强的外部验证集](#临床分层最强的外部验证集)
  - [跨被试迁移最值得优先测试的资源](#跨被试迁移最值得优先测试的资源)
- [筛选后的推荐名单](#筛选后的推荐名单)
  - [公开且可立即用于外部验证](#公开且可立即用于外部验证)
  - [非公开但最值得争取合作或申请的数据队列](#非公开但最值得争取合作或申请的数据队列)
  - [不进入主名单](#不进入主名单)
- [底线判断](#底线判断)
- [References](#references)

## 结论

如果目标是在 \[Liu24\] 之外寻找**可用于外部验证**的卒中患者 MI-EEG 数据集，当前项目里的候选资源可以分成三层。\[Liu24, Liu25, Cho21, Thi25, Man22, Seb20b, Fro17b, Par16, Shu18b\]

**第一层是公开且最适合直接拿来做外部验证的数据集。** 这一层只有少数几套资源真正成立，核心是 [Liu25](#liu25)、[Cho21](#cho21) 和 [Thi25](#thi25)。\[Liu25, Cho21, Thi25\] 其中 [Liu25](#liu25) 与 \[Liu24\] 在“数据集论文 + 患者信息随数据发布”这一点上最接近，但任务从上肢左右手 MI 转到了下肢 gait MI，且临床量表不如 \[Liu24\] 丰富。\[Liu24, Liu25\] [Cho21](#cho21) 更像一个小型挑战赛基准，适合做跨被试迁移 benchmark，但患者临床变量明显更薄。[Cho21](#cho21) [Thi25](#thi25) 则在临床信息这一维度最亮眼，明确给出 NIHSS、病灶侧、病灶位置、mRS 和 Oxford 肌力分级，因而很适合做“模型是否保留临床相关性”的外部验证。[Thi25](#thi25)

**第二层是临床信息很强、但并非真正公开数据集的患者队列。** 这类资源不一定能直接下载，但如果能拿到合作或申请通道，它们对验证算法的临床稳健性很有价值。\[Man22, Seb20b, Fro17b, Par16, Shu18b, Ang10\] [Man22](#man22) 有 136 例受试者和 FMA，规模优势最明显，但数据需申请且更接近机构内队列。[Man22](#man22) [Seb20b](#seb20b)、[Fro17b](#fro17b)、\[Par16\]、\[Shu18b\] 则在病灶位置、患侧和功能评分上更细，适合做病灶分层验证、严重度分层验证或机制相关分析。\[Seb20b, Fro17b, Par16, Shu18b\]

**第三层是与主题相关但不应计入“可用外部验证数据集”主表的记录。** 这类文献要么只是基于既有小队列的方法论文，要么不是卒中患者数据集本身。\[Raz20, Ort23\] \[Raz20\] 用的是作者团队既有的 10 名偏瘫卒中患者数据，并不是一个新的公开 benchmark。\[Raz20\] \[Ort23\] 则不是卒中患者，应直接排除。\[Ort23\]

## 主表 适合在 Liu24 之外做外部验证的数据集

| 数据集或资源 | 是否公开可用 | 受试者与阶段 | MI 任务 | 临床信息强度 | 与 Liu24 相似度 | 是否适合跨被试迁移 | 综合定位 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| [Liu25](#liu25) | 是 | 27 例恢复期卒中，1 到 12 个月 | 下肢 gait MI，对 idle，含多范式与纵向时间点 | 中等，含患侧、病灶位置、卒中类型、病程 | 中高 | 高 | Liu24 之外最值得优先验证的公开数据集 |
| [Cho21](#cho21) | 是 | 10 例偏瘫卒中，平均约 11 个月 | 左右手 grasp attempt，含训练与反馈 session | 较弱，主要有患侧和基本人口学 | 中等 | 很高 | 小样本跨被试迁移 benchmark |
| [Thi25](#thi25) | 是 | 30 例越南 post acute 卒中 | 四分类 MI，左右上肢加左右下肢 | 很强，含 NIHSS、病灶侧、病灶位置、mRS、Oxford | 中等 | 中高 | 临床分层最强的公开验证集 |
| \[Liu24\] | 是 | 50 例急性卒中，1 到 30 天 | 左右手 MI 二分类 | 强，含 NIHSS、患侧、MBI、mRS | 基准本体 | 高 | 参考锚点，不属于外部验证 |

这个主表里的 3 套外部资源各有分工。\[Liu25, Cho21, Thi25\] 若目标是检验模型是否能跨任务型态和跨时间点维持有效性，[Liu25](#liu25) 最合适。[Liu25](#liu25) 若目标是检验跨被试迁移或零校准能力，[Cho21](#cho21) 尽管样本少，但它本身就是为 within subject 与 cross subject benchmark 设计的。[Cho21](#cho21) 若目标是检验模型是否在不同病灶位置、不同 NIHSS 程度和不同肢体任务上仍有解释力，[Thi25](#thi25) 最有价值。[Thi25](#thi25)

## 公开数据集的详细筛选

### Liu25

[Liu25](#liu25) 是当前最像 \[Liu24\]“姊妹资源”的数据集论文。\[Liu24, Liu25\] 它公开发布在 Figshare，并按 BIDS 组织，包含原始 EEG、预处理 EEG 与患者信息。[Liu25](#liu25) 受试者为 27 例恢复期卒中患者，任务是下肢 gait motor imagery 与 idle，对应 5 类实验条件，覆盖 Pre、IES、SES、Post 和 Follow 五个时间点或范式。[Liu25](#liu25)

它最重要的优势有三点。[Liu25](#liu25)

- **第一，公开性强。** 数据以标准结构发布，适合直接接入外部验证流程。[Liu25](#liu25)
- **第二，时间结构强。** 它不是单次采集，而是带纵向与多范式，适合评估跨时点稳健性。[Liu25](#liu25)
- **第三，基础临床信息足够用。** 已明确提供 affected_side、affected_area、diagnosis 和 duration，能做基本分层。[Liu25](#liu25)

它的主要短板是临床量表信息不够强。[Liu25](#liu25) 论文没有明确列出 NIHSS 或 FMA，这意味着它更适合做**泛化验证**和**稳健性验证**，而不如 [Thi25](#thi25) 那样适合做细粒度临床相关性验证。\[Liu25, Thi25\]

### Cho21

[Cho21](#cho21) 对应的是 Clinical BCI Challenge 2020 数据集，是一个公开的小型临床 benchmark。[Cho21](#cho21) 受试者为 10 名偏瘫卒中患者，平均卒中后约 11 个月，任务是左右手 grasp attempt，包含 80 个训练 trial 与 40 个在线反馈 trial。[Cho21](#cho21)

它的关键价值不在临床变量，而在评测结构。[Cho21](#cho21) 这套数据从设计上就把 within subject 与 cross subject 解码都纳入挑战赛，因此非常适合拿来验证跨被试迁移方法是否真正有收益。[Cho21](#cho21) 与 \[Liu24\] 相比，它样本量小很多，任务也更接近 motor attempt 而非纯 MI，临床变量只明确到患侧、年龄和性别，未明确给出 NIHSS、FMA 或病灶位置。\[Liu24, Cho21\]

因此，[Cho21](#cho21) 更适合充当**跨被试迁移的补充 benchmark**，而不适合充当主要的临床分层验证集。[Cho21](#cho21)

### Thi25

[Thi25](#thi25) 是这次搜索里最值得特别关注的新资源。[Thi25](#thi25) 它是 30 例越南卒中患者的公开数据集，任务是四分类 MI，覆盖右臂、左臂、右腿、左腿想象动作，并公开在 GitHub。[Thi25](#thi25) 受试者处于 acute 之后的稳定期，属于较宽口径的 post acute 卒中队列。[Thi25](#thi25)

这套数据最突出的地方是患者信息丰富度。[Thi25](#thi25) 论文明确说明每位患者带有 `Patient's info` JSON 文件，并给出 NIHSS、病灶半球侧、病灶位置描述、mRS、Oxford 肌力分级，以及基础人口学和共病信息。[Thi25](#thi25) 在“外部验证时是否还能保留临床相关结构”这一点上，它比 [Liu25](#liu25) 和 [Cho21](#cho21) 都更强。\[Liu25, Cho21, Thi25\]

它的主要差异在于采集设备和任务定义都与 \[Liu24\] 不同。\[Liu24, Thi25\] [Thi25](#thi25) 使用商用级 Emotiv EPOC Flex，采样率 128 Hz，且任务是四肢四分类，而 \[Liu24\] 是急性期左右手二分类、医疗级设备、500 Hz 采样。\[Liu24, Thi25\] 正因为差异更大，[Thi25](#thi25) 很适合做**外部域偏移验证**。若一个方法在 \[Liu24\] 上成立，却在 [Thi25](#thi25) 上失效，更可能说明方法依赖的是设备条件或任务窄设定，而不是真正稳定的卒中 MI 规律。\[Liu24, Thi25\]

## 不应放进主表但值得保留的临床队列

### Man22

[Man22](#man22) 不是公开数据集，但它可能是这次搜索里**最强的潜在合作验证资源**。[Man22](#man22) 论文汇总了 4 个既往临床试验中的 136 名卒中患者，任务是患侧手臂 MI 对 idle，报告了 FMA、病程分期和较大规模队列的结果。[Man22](#man22) 数据不可直接公开下载，但可在许可下申请。[Man22](#man22)

它最大的优势是样本量和异质性。[Man22](#man22) 用于外部验证时，特别适合回答“模型在大样本患侧 MI 队列上是否仍然成立”以及“性能是否与运动损伤严重度相关”。[Man22](#man22) 但由于缺少直接公开通道，它更适合作为**合作验证目标**，不适合作为立刻就能跑的公开 benchmark。[Man22](#man22)

### Seb20b

[Seb20b](#seb20b) 也不是公开数据集，但临床注释质量很高。[Seb20b](#seb20b) 它包含 36 例卒中患者和 32 名健康对照，患者分成 cortical、subcortical、cortical plus subcortical 三类，并记录 FMA-UE、FMA-LE、BI、MAS、MOCA 等多种量表。[Seb20b](#seb20b) 任务包括 resting EEG 和左右手 wrist dorsiflexion MI，并含 25 次 BCI 治疗 session。[Seb20b](#seb20b)

这使它特别适合做两类验证。[Seb20b](#seb20b)

- **病灶位置分层验证**
- **EEG 指标与功能量表相关性的验证**

它的数据不公开，但可以向作者申请。[Seb20b](#seb20b) 若验证目标不只是“准确率能否迁移”，而是“发现是否与功能状态保持一致”，这类资源价值很高。[Seb20b](#seb20b)

### Fro17b

[Fro17b](#fro17b) 是 74 例患者的多中心随机对照临床试验，任务是患侧手 opening 的 kinesthetic imagery，对应 BCI 手外骨骼反馈训练。[Fro17b](#fro17b) 论文明确给出 FMMA、ARAT、病灶侧向、病灶位置和病程信息。[Fro17b](#fro17b)

它的价值不在公开性，而在**临床终点丰富**。[Fro17b](#fro17b) 如果某种 EEG 表征或分类得分被认为与康复潜力有关，这类数据最适合做“是否与实际功能恢复关联”的外部检验。[Fro17b](#fro17b) 但由于原始数据未公开，它更像高质量临床队列，而不是现成公开数据集。[Fro17b](#fro17b)

### Par16 与 Shu18b

\[Par16\] 和 \[Shu18b\] 样本量都不算大，但都适合做精细分层验证。\[Par16, Shu18b\]

\[Par16\] 有 12 例慢性卒中患者，按病灶位置分成 SM1+、SM1- 和 INF 三组，并提供 affected hand、病程和 FMA-UE 等详细临床资料。\[Par16\] 这非常适合测试一个方法是否对不同病灶解剖位置稳定。\[Par16\]

\[Shu18b\] 有 24 例卒中患者，逐例给出 affected hand、病程、缺血或出血、cortical 或 subcortical、FMA-UE 和 MMSE。\[Shu18b\] 它主要是 BCI inefficiency screening 研究，但其患者表型信息足够完整，适合作为小规模机制验证或分层分析队列。\[Shu18b\]

## 不建议作为独立数据集主候选的文献

| 论文 | 原因 |
|:---|:---|
| \[Raz20\] | 只是对作者团队已有 10 人偏瘫卒中数据做 EEGNet 方法验证，不是新的公开数据集 |
| \[Ang10\] | 54 例患者临床研究很重要，含 FMA，但当前项目无全文，且看起来更像临床试验而非公开数据集 |
| \[Ang11\] | 标题提示是大样本临床研究，但当前项目无全文，无法确认其数据开放性与临床变量全貌 |
| \[Tam11b\] | 多 session 慢性卒中队列可能有价值，但当前项目无全文，且更像内部数据研究 |
| \[Sul23\] | 26 例 sub acute stroke 与 15 次治疗 session 值得关注，但当前项目信息不足，未见公开数据说明 |
| \[Ort23\] | 不是卒中患者，应排除 |

这组文献不应消失，它们说明卒中 MI 队列并不少，但真正做到“公开可复用 + 患者信息清楚 + 可用于外部验证”的资源依旧稀缺。\[Ang10, Ang11, Tam11b, Sul23, Ort23\]

## 临床信息覆盖对比

| 资源 | NIHSS | FMA 或 FMMA | 患侧 | 病灶位置 | 病程 | 备注 |
|:---|:---|:---|:---|:---|:---|:---|
| \[Liu24\] | 有 | 无明确 FMA | 有 | 无明确病灶位置表 | 有 | 另有 MBI 和 mRS |
| [Liu25](#liu25) | 无明确 | 无明确 | 有 | 有 affected_area | 有 | 另有 stroke type |
| [Cho21](#cho21) | 无明确 | 无明确 | 有 | 无明确 | 有 | 更偏 benchmark |
| [Thi25](#thi25) | 有 | 无 FMA，改用 Oxford 与 mRS | 有 | 有 | 未细表但处于 post acute 稳定期 | 临床信息最完整的公开集之一 |
| [Man22](#man22) | 无明确 | 有 FMA | 间接有 | 仅粗略提到 cortical plus subcortical | 有 | 数据需申请 |
| [Seb20b](#seb20b) | 无明确 | 有 FMA UE 与 LE | 有 | 有，三类病灶分组 | 有 | 另有多种临床量表 |
| [Fro17b](#fro17b) | 无明确 | 有 FMMA | 有 | 有，cortical、subcortical、corticosubcortical | 有 | 另有 ARAT |
| \[Par16\] | 无明确 | 有 FMA UE | 有 | 有，SM1 加减与 INF | 有 | 适合病灶分层 |
| \[Shu18b\] | 无明确 | 有 FMA UE | 有 | 有，cortical 或 subcortical | 有 | 逐例表型较完整 |

从临床信息密度看，若只看公开数据集，[Thi25](#thi25) 与 \[Liu24\] 最强，只是它们强调的变量不同。\[Liu24, Thi25\] \[Liu24\] 更像急性期临床基线加二分类 benchmark，[Thi25](#thi25) 更像中后期广义任务和丰富患者描述并重的数据集。\[Liu24, Thi25\] [Liu25](#liu25) 则在病灶位置与病程上已经够用，但量表维度仍偏弱。[Liu25](#liu25)

## 与 Liu24 的相似度判断

### 最接近 Liu24 的外部验证集

若只从“公开可获取、卒中患者、MI 任务、患者信息随数据发布”这四个条件看，[Liu25](#liu25) 最接近 \[Liu24\]。\[Liu24, Liu25\] 两者都是 Scientific Data 风格的数据集论文，都公开发布 EEG 与患者信息，并都面向卒中康复 BCI 的方法开发。\[Liu24, Liu25\] 差异在于 \[Liu24\] 是急性期上肢左右手二分类，而 [Liu25](#liu25) 是恢复期下肢 gait MI、多范式、纵向结构。\[Liu24, Liu25\]

### 临床分层最强的外部验证集

若把重点从“与 Liu24 像不像”转成“是否能检验发现是否跨临床亚型成立”，[Thi25](#thi25) 最强。[Thi25](#thi25) 它公开、任务丰富、临床 JSON 结构明确，且同时有 NIHSS、病灶半球和病灶位置描述。[Thi25](#thi25) 这使它成为最适合做**临床分层外部验证**的公开资源。[Thi25](#thi25)

### 跨被试迁移最值得优先测试的资源

若重点是跨被试迁移，[Cho21](#cho21) 与 [Liu25](#liu25) 的优先级最高，但原因不同。\[Cho21, Liu25\]

- [Cho21](#cho21) 样本小，但本身就是 cross subject benchmark 设计。[Cho21](#cho21)
- [Liu25](#liu25) 样本更多、结构更复杂，适合看模型是否能跨时间点和跨范式保持稳定。[Liu25](#liu25)

这意味着最稳妥的迁移验证顺序不是只找一个替代集，而是把两类验证拆开：先在 [Cho21](#cho21) 看跨被试，再在 [Liu25](#liu25) 看跨时点和跨范式。\[Cho21, Liu25\]

## 筛选后的推荐名单

### 公开且可立即用于外部验证

1.  [Liu25](#liu25)
2.  [Thi25](#thi25)
3.  [Cho21](#cho21)

### 非公开但最值得争取合作或申请的数据队列

1.  [Man22](#man22)
2.  [Seb20b](#seb20b)
3.  [Fro17b](#fro17b)
4.  \[Par16\]
5.  \[Shu18b\]

### 不进入主名单

- \[Raz20\]
- \[Ang10\]
- \[Ang11\]
- \[Tam11b\]
- \[Sul23\]
- \[Ort23\]

## 底线判断

在 \[Liu24\] 之外，真正满足“卒中患者 + MI 任务 + 可作为外部验证数据集”这一要求的公开资源并不多。\[Liu24, Liu25, Cho21, Thi25\] 若只保留最值得立即使用的候选，优先级应是 [Liu25](#liu25)、[Thi25](#thi25)、[Cho21](#cho21)。\[Liu25, Thi25, Cho21\] 其中 [Liu25](#liu25) 最像 \[Liu24\] 的扩展版，[Thi25](#thi25) 的临床信息最强，[Cho21](#cho21) 最适合做跨被试迁移 benchmark。\[Liu25, Thi25, Cho21\]

若验证目标再进一步提升到“模型是否对病灶位置、功能评分和病程阶段保持稳定”，则公开数据已经不够，需要把 [Man22](#man22)、[Seb20b](#seb20b)、[Fro17b](#fro17b)、\[Par16\]、\[Shu18b\] 这类临床队列纳入第二层资源池。\[Man22, Seb20b, Fro17b, Par16, Shu18b\] 这也是当前卒中 MI-EEG 资源版图最清楚的现实：公开 benchmark 很少，临床表型丰富的队列更多存在于单中心或多中心研究内部，而不是标准化开放数据集。\[Liu24, Thi25, Man22, Seb20b, Fro17b\]

---

## References

\[Liu24\] H. Liu *et al.*, “An EEG motor imagery dataset for brain computer interface in acute stroke patients,” *Scientific Data*, vol. 11, Jan. 2024, doi: [10.1038/s41597-023-02787-8](https://doi.org/10.1038/s41597-023-02787-8).

\[Liu25\] Y. Liu *et al.*, “Lower limb motor imagery EEG dataset based on the multi-paradigm and longitudinal-training of stroke patients,” *Scientific Data*, vol. 12, Feb. 2025, doi: [10.1038/s41597-025-04618-4](https://doi.org/10.1038/s41597-025-04618-4).

\[Cho21\] A. Chowdhury and J. Andreu-Perez, “Clinical Brain–Computer Interface Challenge 2020 (CBCIC at WCCI2020): Overview, Methods and Results,” *IEEE Transactions on Medical Robotics and Bionics*, vol. 3, pp. 661–670, Aug. 2021, doi: [10.1109/TMRB.2021.3098108](https://doi.org/10.1109/TMRB.2021.3098108).

\[Thi25\] C. M. Thi *et al.*, “UET175: EEG dataset of motor imagery tasks in Vietnamese stroke patients,” *Frontiers in Neuroscience*, vol. 19, Jun. 2025, doi: [10.3389/fnins.2025.1580931](https://doi.org/10.3389/fnins.2025.1580931).

\[Man22\] S. Mansour, J. Giles, K. Ang, K. Nair, K. Phua, and M. Arvaneh, “Exploring the ability of stroke survivors in using the contralesional hemisphere to control a brain–computer interface,” *Scientific Reports*, vol. 12, Sep. 2022, doi: [10.1038/s41598-022-20345-x](https://doi.org/10.1038/s41598-022-20345-x).

\[Seb20b\] M. Sebastián-Romagosa *et al.*, “EEG Biomarkers Related With the Functional State of Stroke Patients,” *Frontiers in Neuroscience*, vol. 14, Jul. 2020, doi: [10.3389/fnins.2020.00582](https://doi.org/10.3389/fnins.2020.00582).

\[Fro17b\] A. Frolov *et al.*, “Post-stroke Rehabilitation Training with a Motor-Imagery-Based Brain-Computer Interface (BCI)-Controlled Hand Exoskeleton: A Randomized Controlled Multicenter Trial,” *Frontiers in Neuroscience*, vol. 11, Jul. 2017, doi: [10.3389/fnins.2017.00400](https://doi.org/10.3389/fnins.2017.00400).

\[Par16\] W. Park, G. Kwon, Y. Kim, J.-H. Lee, and L. Kim, “EEG response varies with lesion location in patients with chronic stroke,” *Journal of NeuroEngineering and Rehabilitation*, vol. 13, Mar. 2016, doi: [10.1186/s12984-016-0120-2](https://doi.org/10.1186/s12984-016-0120-2).

\[Shu18b\] X. Shu *et al.*, “Fast Recognition of BCI-Inefficient Users Using Physiological Features from EEG Signals: A Screening Study of Stroke Patients,” *Frontiers in Neuroscience*, vol. 12, Feb. 2018, doi: [10.3389/fnins.2018.00093](https://doi.org/10.3389/fnins.2018.00093).

\[Ang10\] K. Ang *et al.*, “Clinical study of neurorehabilitation in stroke using EEG-based motor imagery brain-computer interface with robotic feedback,” *2010 Annual International Conference of the IEEE Engineering in Medicine and Biology*, pp. 5549–5552, Aug. 2010, doi: [10.1109/IEMBS.2010.5626782](https://doi.org/10.1109/IEMBS.2010.5626782).

\[Raz20\] H. Raza, A. Chowdhury, and S. Bhattacharyya, “Deep Learning based Prediction of EEG Motor Imagery of Stroke Patients’ for Neuro-Rehabilitation Application,” *2020 International Joint Conference on Neural Networks (IJCNN)*, pp. 1–8, Mar. 2020, doi: [10.1109/IJCNN48605.2020.9206884](https://doi.org/10.1109/IJCNN48605.2020.9206884).

\[Ort23\] M. Ortíz *et al.*, “An EEG database for the cognitive assessment of motor imagery during walking with a lower-limb exoskeleton,” *Scientific Data*, vol. 10, Jun. 2023, doi: [10.1038/s41597-023-02243-7](https://doi.org/10.1038/s41597-023-02243-7).

\[Ang11\] K. Ang *et al.*, “A Large Clinical Study on the Ability of Stroke Patients to Use an EEG-Based Motor Imagery Brain-Computer Interface,” *Clinical EEG and Neuroscience*, vol. 42, pp. 253–258, Oct. 2011, doi: [10.1177/155005941104200411](https://doi.org/10.1177/155005941104200411).

\[Tam11b\] W. Tam, Z. Ke, and K. Tong, “Performance of common spatial pattern under a smaller set of EEG electrodes in brain-computer interface on chronic stroke patients: A multi-session dataset study,” *2011 Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, pp. 6344–6347, Dec. 2011, doi: [10.1109/IEMBS.2011.6091566](https://doi.org/10.1109/IEMBS.2011.6091566).

\[Sul23\] M. Sultana, C. Reichert, C. Sweeney-Reed, and S. Perdikis, “Towards Calibration-Less BCI-Based Rehabilitation,” *2023 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)*, pp. 11–16, Oct. 2023, doi: [10.1109/MetroXRAINE58569.2023.10405575](https://doi.org/10.1109/MetroXRAINE58569.2023.10405575).
