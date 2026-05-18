# Liu24 数据集严格核验与文献分层

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [结论](#结论)
- [严格纳入标准](#严格纳入标准)
- [已确认直接使用 \[Liu24\] 的论文](#已确认直接使用-liu24-的论文)
- [主报告应保留的核心方法论文](#主报告应保留的核心方法论文)
- [Bun24 与 Bun25 的关系与协议差异](#bun24-与-bun25-的关系与协议差异)
- [已确认使用了 \[Liu24\] 但不应进入主 benchmark 的论文](#已确认使用了-liu24-但不应进入主-benchmark-的论文)
- [当前仍不能被严格确认的候选论文](#当前仍不能被严格确认的候选论文)
- [明确不是 \[Liu24\] 的论文](#明确不是-liu24-的论文)
- [从已确认文献里能读出的真实用法](#从已确认文献里能读出的真实用法)
- [对这个数据集还剩下哪些明确空间](#对这个数据集还剩下哪些明确空间)
  - [协议空白](#协议空白)
  - [问题设定空白](#问题设定空白)
  - [机制与临床空白](#机制与临床空白)
- [最终建议的纳入范围](#最终建议的纳入范围)
- [References](#references)

## 结论

补读新上传的全文后，这份报告的证据边界可以进一步收紧，也更完整。\[Bun24, Len24b, She24, Lin24b\] 当前项目里，能够被严格确认为“直接使用 \[Liu24\] 并报告了对应结果”的论文共有 7 篇，其中 \[Liu24\] 是数据描述与官方 benchmark 论文，真正属于后续复用研究的有 6 篇。\[Liu24, Bun24, Bun25, Kav24b, Wan26h, Liu25c, Lv25\]

如果只问“后续研究者如何真正使用这个 50 人急性卒中 MI 数据集”，当前最值得保留在主报告里的核心方法论文是 4 篇：\[Bun24\]、\[Bun25\]、\[Kav24b\]、\[Wan26h\]。\[Bun24, Bun25, Kav24b, Wan26h\] 它们分别代表四种已经被文献明确走通的路线：完整 50 人二分类高性能建模、同一数据上的更清楚协议版本、任务重定义后的源空间三分类、以及更严格的跨被试适配。\[Bun24, Bun25, Kav24b, Wan26h\]

这次补 PDF 之后，最重要的变化有三点。\[Bun24, Len24b, She24, Lin24b\]

- \[Bun24\] 现在可以从待核池进入“已确认直接使用 \[Liu24\]”集合。\[Bun24\]
- \[Len24b\] 已能明确确认不是 \[Liu24\]，而是 Qilu Hospital 自采 18 名卒中患者加 18 名健康对照数据。\[Len24b\]
- \[She24\] 与 \[Lin24b\] 也都应从候选池移出，前者使用 WCCI 2020 的 Clinical BCI Challenge 数据，后者做的是本地医院自采 post stroke 静息态分类。\[She24, Lin24b\]

## 严格纳入标准

这份报告只把同时满足下列条件的论文视为“已确认直接使用 \[Liu24\]”。\[Liu24, Bun24, Bun25, Kav24b, Wan26h, Liu25c, Lv25\]

- 正文或参考文献中能明确把数据源指向 \[Liu24\]
- 能确认所用数据确实来自该公开数据集，而不是仅仅也是卒中 MI 数据
- 论文报告了该数据上的结果，而不只是讨论任务背景
- 论文的结果可以落到清楚的任务设定中，例如二分类、三分类、LOSO、子集分析、机制验证等

这个标准有一个直接后果：相关但证据不够硬的论文，宁可暂时放进待核池，也不提前写成“已确认使用”。\[Shi25, Shi25b, Thw26, Liu25b, Ma25\]

## 已确认直接使用 \[Liu24\] 的论文

| 论文 | 核验结论 | 数据范围 | 报告结果 | 当前角色 |
|:---|:---|:---|:---|:---|
| \[Liu24\] | 确认 | 完整 50 人急性卒中，左右手 MI 二分类 | TWFB 加 DGFMDRM 72.21%，另有 3 个官方基线 | 数据集定义与官方 benchmark |
| \[Bun24\] | 确认 | 完整 50 人急性卒中，左右手 MI 二分类 | SACAN 94.44% | 早期完整数据集二分类方法论文 |
| \[Bun25\] | 确认 | 完整 50 人急性卒中，左右手 MI 二分类 | 80 比 20 标准评测 99.52%，LOSO 97.43% | 当前协议最清楚的完整数据集二分类强基线 |
| \[Kav24b\] | 确认 | 完整 50 人急性卒中，重组为左 MI 右 MI rest 三分类 | Dipole 91.03%，MNE 89.07%，Beamforming 87.17% | 源空间重建后的三分类路线 |
| \[Wan26h\] | 确认 | 把 \[Liu24\] 作为 XW Stroke 数据集使用，LOSO 跨被试 | XW Stroke 66.56%，另在 2019 Stroke 上为 72.75% | 更严格跨被试适配路线 |
| \[Liu25c\] | 确认 | 公共急性卒中数据中的 30 人筛选子集，用作验证材料 | 各 ROI 解码准确率高于 60%，并做流形与 NIHSS 相关分析 | 机制分析与外部验证，不是主 benchmark |
| \[Lv25\] | 确认 | 从 \[Liu24\] 严格筛选后的 14 人子集 | KNN 75.00% ± 7.94%，LDA 和 SVM 更低 | 子集微状态分析，不是完整 benchmark |

这 7 篇里，\[Liu24\] 提供的是数据定义与官方基线。\[Liu24\] 真正回答“后来研究者怎么用这个数据集”的，是 \[Bun24\]、\[Bun25\]、\[Kav24b\]、\[Wan26h\]、\[Liu25c\] 和 \[Lv25\]。\[Bun24, Bun25, Kav24b, Wan26h, Liu25c, Lv25\]

## 主报告应保留的核心方法论文

如果目标是给后续实验和写作留下一个干净、能落地的主线，主报告最适合保留 4 篇方法论文，再用 \[Liu24\] 作为官方锚点。\[Liu24, Bun24, Bun25, Kav24b, Wan26h\]

| 论文 | 为什么应保留 | 不能忽略的细节 | 应如何解读 |
|:---|:---|:---|:---|
| \[Bun24\] | 明确引用 \[Liu24\]，完整 50 人，任务和样本数都对得上 | 虽写 cross subject classification，但文中暴露的核心评测更接近 80 比 20 划分 | 可视为早期完整数据集高分方法，但不是最严格泛化锚点 |
| \[Bun25\] | 明确引用 \[Liu24\]，完整 50 人，结果和协议都写得清楚 | 同时报告 80 比 20 标准划分与 LOSO，二者必须分开解读 | 当前最强且协议最清楚的完整数据集二分类论文 |
| \[Kav24b\] | 明确引用 \[Liu24\]，完整 50 人，公开数据链接对得上 | 它做的是左 MI 右 MI rest 三分类，不是原始二分类 | 是任务重定义后的源空间路线，不是二分类 SOTA |
| \[Wan26h\] | 明确把 \[Liu24\] 命名为 XW Stroke，并给出 LOSO 表 | 同时用了另一套卒中数据，绝对分数明显低于 \[Bun25\] | 价值在更严格跨被试适配，而不是高随机划分分数 |

这个主表比此前版本更完整，因为 \[Bun24\] 已经可以明确纳入。\[Bun24\] 但它也更能说明一个事实：同样都说自己做卒中 MI 分类，论文之间实际做的任务、划分和难度并不一样。\[Bun24, Bun25, Kav24b, Wan26h\]

## Bun24 与 Bun25 的关系与协议差异

这两篇论文很容易被看成同一路线上的“前后两版高分工作”，这一点基本成立。\[Bun24, Bun25\] 但若把它们直接排成一张总榜，反而会遮住最重要的信息：它们在协议清晰度上并不相同。\[Bun24, Bun25\]

| 维度 | \[Bun24\] | \[Bun25\] | 对解读的影响 |
|:---|:---|:---|:---|
| 数据身份 | 明确引用 \[Liu24\] | 明确引用 \[Liu24\] | 两篇都已可严格确认 |
| 数据规模 | 完整 50 人 | 完整 50 人 | 具有可比基础 |
| 任务 | 左右手 MI 二分类 | 左右手 MI 二分类 | 任务层面一致 |
| 预处理 | 0.5 到 45 Hz，2 秒窗，1 秒重叠，增强，Z score | 0.5 到 40 Hz，2 秒窗，1 秒重叠，增强，Z score | 处理流程相近 |
| 高分结果 | 94.44% | 99.52% 标准评测 | 都可能受 pooled split 难度较低影响 |
| 泛化结果 | 文中强调 cross subject，但未像 \[Bun25\] 那样清楚给出 50 次 LOSO 表 | 明确给出 50 次 LOSO，97.43% | \[Bun25\] 更适合当严格跨被试锚点 |

因此，这两篇最稳妥的读法不是“\[Bun25\] 单纯把 \[Bun24\] 又提了几分”，而是：\[Bun24\] 代表了完整 50 人 \[Liu24\] 二分类高性能建模的早期版本，\[Bun25\] 则把相近路线推进到更高分，并把 LOSO 单独报告出来，使它更适合担任严格 benchmark 的主对手。\[Bun24, Bun25\]

这也意味着，后续若要做对标实验，至少应把下面两种设定分开。\[Bun24, Bun25\]

- 标准随机或 pooled 划分下的二分类性能
- 真正的 LOSO 跨被试泛化性能

若把 \[Bun24\] 的 94.44% 和 \[Bun25\] 的 97.43% LOSO 直接排成同一列，就会混掉协议差异。\[Bun24, Bun25\]

## 已确认使用了 \[Liu24\] 但不应进入主 benchmark 的论文

| 论文 | 为什么不进主榜 | 还能提供什么 insight |
|:---|:---|:---|
| \[Liu25c\] | \[Liu24\] 只是验证材料之一，而且只筛了 30 人，不是完整 benchmark | 说明这个数据集也可被拿来做流形稳定性与临床相关性分析 |
| \[Lv25\] | 只保留了 14 人子集，协议和样本范围都已改变 | 说明微状态特征可作为小样本衍生分析路线 |
| \[Kav24\] | 与 \[Kav24b\] 高度重合，更像会议先行版 | 不宜与期刊版重复计数 |

这组论文很重要，因为它们说明 \[Liu24\] 并不只被当成“谁分数更高”的二分类赛场。\[Liu25c, Lv25, Kav24b\] 文献已经开始把它拿来做三类任务重定义、子集机制分析、以及与临床变量相关的表征研究。\[Kav24b, Liu25c, Lv25\]

## 当前仍不能被严格确认的候选论文

补读新 PDF 后，待核池比之前更短了。\[Shi25, Shi25b, Thw26, Liu25b, Ma25\] 这些论文都与卒中 MI 很近，其中几篇甚至仍有可能用了 \[Liu24\]。但在当前项目证据下，还不够写成“已确认使用”。\[Shi25, Shi25b, Thw26, Liu25b, Ma25\]

| 论文 | 当前状态 | 不能确认的关键原因 | 处理建议 |
|:---|:---|:---|:---|
| \[Shi25\] | 中高相关候选 | 只写 dataset of 50 stroke patients，未明确把来源钉到 \[Liu24\] | 等 PDF 后再判 |
| \[Shi25b\] | 中高相关候选 | 只写 clinical stroke EEG dataset，缺少明确出处 | 等 PDF 后再判 |
| \[Thw26\] | 低到中相关候选 | 只知道面向 acute stroke MI 分类，数据集身份缺失 | 先不纳入 |
| \[Liu25b\] | 低到中相关候选 | 只说在 stroke patient dataset 上验证 | 先不纳入 |
| \[Ma25\] | 低相关 | 摘要缺失，当前无法判断是否真用 \[Liu24\] | 先不纳入 |

现在最值得继续补 PDF 的核心对象，已经从 \[Bun24\]、\[Len24b\] 转移到 \[Shi25\] 和 \[Shi25b\]。\[Shi25, Shi25b\] 它们最有可能继续改变“已确认使用 \[Liu24\]”的主表边界。\[Shi25, Shi25b\]

## 明确不是 \[Liu24\] 的论文

这次补全文后，有三篇原本边界模糊的论文已经可以明确移出主线。\[Len24b, She24, Lin24b\]

| 论文 | 实际数据来源 | 为什么应排除 |
|:---|:---|:---|
| \[Len24b\] | Qilu Hospital 自采数据，18 名卒中患者加 18 名健康对照 | 不是公开的 \[Liu24\] 数据集 |
| \[She24\] | Clinical Brain Computer Interface Challenge WCCI 2020，10 名偏瘫卒中受试者 | 是 CBCIC 系列，不是 \[Liu24\] |
| \[Lin24b\] | 北京清华长庚医院自采 72 名 post stroke 患者静息态数据 | 做的是 EO 对 EC 静息态分类，不是 MI benchmark |
| \[Szt26\] | PhysioNet EEG Motor Movement Imagery Database，109 名健康志愿者中的 25 人子集 | 与卒中患者和 \[Liu24\] 都无关 |
| \[Cho21\] | CBCIC2020，10 名偏瘫卒中受试者，抓握 attempt 范式 | 是另一套竞赛数据 |
| \[Sai23\] | 10 名偏瘫卒中受试者 MI 数据 | 不是 \[Liu24\] |
| \[Ras21\] | 开源偏瘫卒中数据 | 不是 \[Liu24\]，且比 \[Liu24\] 更早 |
| \[Mim26\] | Clinical BCI Challenge WCCI 2020 数据 | 明确是 CBCIC 系列 |
| \[Mia26\] | CBCIC 加自采数据加 BCI IV 2b | 不是 \[Liu24\] |
| \[Liu25\] | 27 人下肢 MI 纵向卒中数据集 | 是另一篇数据集论文 |

这一组论文可以帮助理解更广的卒中 MI 版图，但它们不回答“研究者如何使用 \[Liu24\]”这个问题。\[Len24b, She24, Lin24b, Szt26, Cho21, Sai23, Ras21, Mim26, Mia26, Liu25\] 因此在这份报告里应整体移出主线，只在做跨数据集背景时再回头引用。\[Cho21, Liu25\]

## 从已确认文献里能读出的真实用法

严格筛完以后，\[Liu24\] 的已确认用法主要集中在五条路线。\[Bun24, Bun25, Kav24b, Wan26h, Liu25c, Lv25\]

- 把它当完整 50 人急性卒中二分类 benchmark，直接追求更高准确率。\[Bun24, Bun25\]
- 在相同数据上把 sensor 空间改写成 source 空间，并扩成三分类任务。\[Kav24b\]
- 把它当跨被试适配测试床，专门看更严格的 LOSO 泛化能力。\[Wan26h\]
- 从完整数据中筛选子集，做微状态等小样本表征研究。\[Lv25\]
- 把它当外部验证材料，研究低维流形、跨受试者结构稳定性和 NIHSS 关联。\[Liu25c\]

反过来说，当前已确认文献里，真正缺的并不是“再多一篇 90 多分分类论文”，而是下列几类工作。\[Liu24, Bun24, Bun25, Wan26h, Liu25c\]

- 在统一 LOSO 协议下重跑传统方法、轻量深网和适配方法
- 把患者特征真正引入建模，而不是只把它当纯信号分类问题
- 做更接近临床部署的个体化与时间漂移鲁棒性评测
- 把分类结果与康复机制或临床结局更紧地连起来

## 对这个数据集还剩下哪些明确空间

### 协议空白

目前最缺的不是更高的 pooled split 分数，而是协议更清楚的完整 benchmark。已确认论文里，二分类完整 50 人高分结果已有 \[Bun24\] 和 \[Bun25\]，更严格跨被试适配结果则来自 \[Wan26h\]。\[Bun24, Bun25, Wan26h\] 但这三类结果并不在同一难度层面上，因此未来最有价值的工作，不是再报一个更高但协议含混的百分比，而是在同一数据读取、同一预处理、同一 LOSO 定义下把传统方法、轻量深网、域适配方法统一重跑。\[Liu24, Bun25, Wan26h\]

### 问题设定空白

\[Kav24b\] 已经证明，把 \[Liu24\] 从原始二分类改成左 MI 右 MI rest 三分类是可行的。\[Kav24b\] 但除了这条线，当前已确认文献几乎还没有系统探索病灶侧、健患侧重编码、临床严重度分层、或不同受试者子群的专门建模。\[Liu24, Kav24b, Liu25c, Lv25\] 数据集本身开放了患者特征，但真正把这些信息融入模型和评测的工作仍明显不足。\[Liu24\]

### 机制与临床空白

\[Liu25c\] 和 \[Lv25\] 给出了一个重要信号，即 \[Liu24\] 不只是能做分类，也能拿来研究急性期卒中的脑动力学和表征变化。\[Liu25c, Lv25\] 但这类工作目前仍停在子集分析或验证性分析层面，离“分类结果与康复机制或临床转归真正联动”还有距离。\[Liu25c, Lv25\] 这一空白比继续堆叠网络结构更值得重视。\[Liu25c\]

## 最终建议的纳入范围

如果这份文件的目标是给后续实验和写作提供一个干净证据底座，最合理的纳入范围如下。\[Liu24, Bun24, Bun25, Kav24b, Wan26h, Liu25c, Lv25\]

- 主表保留 \[Liu24\]、\[Bun24\]、\[Bun25\]、\[Kav24b\]、\[Wan26h\]
- 附表保留 \[Liu25c\] 和 \[Lv25\]，明确标注为子集或验证性使用
- \[Kav24\] 作为 \[Kav24b\] 的同线先行版，不单独计数
- \[Shi25\]、\[Shi25b\]、\[Thw26\]、\[Liu25b\]、\[Ma25\] 列入待核池，不进入主结论
- \[Len24b\]、\[She24\]、\[Lin24b\] 以及其他明确非 \[Liu24\] 论文全部移出主线

这样处理后，报告会比最初版本更短，也更硬。当前真正成立的结论不是“很多论文都在 \[Liu24\] 上做到了 90 多分”，而是“已被严格确认的完整数据集方法论文只有少数几篇，而且它们代表的不是同一种比较，而是二分类高分 benchmark、协议更清楚的 LOSO 泛化、源空间三分类、以及更严格跨被试适配这几条不同路线”。\[Bun24, Bun25, Kav24b, Wan26h\]

---

## References

\[Bun24\] C. Bunterngchit, “Improving Motor Imagery Classification for Stroke Rehabilitation Using a Stroke-Aware Capsule Attention Network,” in *2024 International Conference on Decision Aid Sciences and Applications (DASA)*, Dec. 2024, pp. 1–5. doi: [10.1109/DASA63652.2024.10836350](https://doi.org/10.1109/DASA63652.2024.10836350).

\[Len24b\] J. Leng *et al.*, “Time-Frequency-Space EEG Decoding Model Based on Dense Graph Convolutional Network for Stroke,” *IEEE Journal of Biomedical and Health Informatics*, vol. 28, pp. 5214–5226, Jun. 2024, doi: [10.1109/JBHI.2024.3411646](https://doi.org/10.1109/JBHI.2024.3411646).

\[She24\] Y. Shen, J. Liu, Y. Yang, and W. Xie, “Using EEG Data to Classify Hand Motor Imagery in Stroke Patients,” in *2024 4th International Conference on Electronic Information Engineering and Computer (EIECT)*, Nov. 2024, pp. 1102–1108. doi: [10.1109/EIECT64462.2024.10866311](https://doi.org/10.1109/EIECT64462.2024.10866311).

\[Lin24b\] P.-J. Lin *et al.*, “AM-EEGNet: An advanced multi-input deep learning framework for classifying stroke patient EEG task states,” *Neurocomputing*, vol. 585, p. 127622, Jun. 2024, doi: [10.1016/j.neucom.2024.127622](https://doi.org/10.1016/j.neucom.2024.127622).

\[Liu24\] H. Liu *et al.*, “An EEG motor imagery dataset for brain computer interface in acute stroke patients,” *Scientific Data*, vol. 11, Jan. 2024, doi: [10.1038/s41597-023-02787-8](https://doi.org/10.1038/s41597-023-02787-8).

\[Bun25\] C. Bunterngchit *et al.*, “GACL-Net: Hybrid Deep Learning Framework for Accurate Motor Imagery Classification in Stroke Rehabilitation,” *Computers, Materials &amp; Continua*, 2025, doi: [10.32604/cmc.2025.060368](https://doi.org/10.32604/cmc.2025.060368).

\[Kav24b\] S. M. Kaviri and R. Vinjamuri, “Integrating Electroencephalography Source Localization and Residual Convolutional Neural Network for Advanced Stroke Rehabilitation,” *Bioengineering*, vol. 11, Sep. 2024, doi: [10.3390/bioengineering11100967](https://doi.org/10.3390/bioengineering11100967).

\[Wan26h\] X. Wang *et al.*, “PA-TCNet: Pathology-Aware Temporal Calibration with Physiology-Guided Target Refinement for Cross-Subject Motor Imagery EEG Decoding in Stroke Patients,” Apr. 17, 2026.

\[Liu25c\] T. Liu, Z. Wang, S. Shakil, and R. K. Tong, “Uncovering Low-Dimensional Manifolds of Neural Dynamics for Motor-Imagery Based Stroke Rehabilitation: An EEG-Based Brain–Computer Interface Study,” *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 33, pp. 3281–3292, 2025, doi: [10.1109/TNSRE.2025.3600824](https://doi.org/10.1109/TNSRE.2025.3600824).

\[Lv25\] S. Lv *et al.*, “Classification of left and right-hand motor imagery in acute stroke patients using EEG microstate,” *Journal of NeuroEngineering and Rehabilitation*, vol. 22, Jun. 2025, doi: [10.1186/s12984-025-01668-y](https://doi.org/10.1186/s12984-025-01668-y).

\[Shi25\] V. R. Shirodkar, D. Edla, A. Kumari, and R. Dharavath, “Deep feature extraction and swarm-optimized enhanced extreme learning machine for motor imagery recognition in stroke patients.” *Journal of neuroscience methods*, pp. 110565, Sep. 2025, doi: [10.1016/j.jneumeth.2025.110565](https://doi.org/10.1016/j.jneumeth.2025.110565).

\[Shi25b\] V. R. Shirodkar, D. Edla, A. Kumari, and M. M. Afonso, “Multi-domain feature extraction and Sand Cat Swarm Optimized Broad Learning System for EEG-based Motor Imagery decoding in stroke patients,” *Computers in biology and medicine*, vol. 199, pp. 111285, Nov. 2025, doi: [10.1016/j.compbiomed.2025.111285](https://doi.org/10.1016/j.compbiomed.2025.111285).

\[Thw26\] Y. Thwe, D. Maneetham, and P. N. Crisnapati, “Integrating EEG Artifact Removal with Deep Learning for Accurate Motor Imagery Classification in Acute Stroke,” *Engineering, Technology &amp; Applied Science Research*, Feb. 2026, doi: [10.48084/etasr.15300](https://doi.org/10.48084/etasr.15300).

\[Liu25b\] S. Liu *et al.*, “CosELU: An improved activation deep neural network for EEG signal recognition in stroke patients.” *Neural networks : the official journal of the International Neural Network Society*, vol. 191, pp. 107800, Jul. 2025, doi: [10.1016/j.neunet.2025.107800](https://doi.org/10.1016/j.neunet.2025.107800).

\[Ma25\] J. Ma, J. Zhang, Y. Yang, B. Yang, and C. Shan, “Motor Imagery Classification Based on Temporal-Spatial Domain Adaptation for Stroke Patients,” *Cognitive Computation*, vol. 17, Jun. 2025, doi: [10.1007/s12559-025-10477-3](https://doi.org/10.1007/s12559-025-10477-3).

\[Kav24\] S. M. Kaviri and R. Vinjamuri, “Integrating EEG Source Localization and Resnet CNN for Advanced Stroke Rehabilitation,” Sep. 19, 2024. doi: [10.1109/HI-POCT64255.2024.10876201](https://doi.org/10.1109/HI-POCT64255.2024.10876201).

\[Szt26\] B. Sztyler, A. Królak, and P. Strumiłło, “Influence of EEG Signal Augmentation Methods on Classification Accuracy of Motor Imagery Events,” *Sensors (Basel, Switzerland)*, vol. 26, Feb. 2026, doi: [10.3390/s26041258](https://doi.org/10.3390/s26041258).

\[Cho21\] A. Chowdhury and J. Andreu-Perez, “Clinical Brain–Computer Interface Challenge 2020 (CBCIC at WCCI2020): Overview, Methods and Results,” *IEEE Transactions on Medical Robotics and Bionics*, vol. 3, pp. 661–670, Aug. 2021, doi: [10.1109/TMRB.2021.3098108](https://doi.org/10.1109/TMRB.2021.3098108).

\[Sai23\] P. Saideepthi, A. Chowdhury, P. Gaur, and R. B. Pachori, “Sliding Window Along With EEGNet-Based Prediction of EEG Motor Imagery,” *IEEE Sensors Journal*, vol. 23, pp. 17703–17713, Aug. 2023, doi: [10.1109/JSEN.2023.3270281](https://doi.org/10.1109/JSEN.2023.3270281).

\[Ras21\] S. Rasheed and W. Mumtaz, “Classification of Hand-Grasp Movements of Stroke Patients using EEG Data,” *2021 International Conference on Artificial Intelligence (ICAI)*, pp. 86–90, Apr. 2021, doi: [10.1109/ICAI52203.2021.9445231](https://doi.org/10.1109/ICAI52203.2021.9445231).

\[Mim26\] M. H. Mim, D. R. Pranto, and Md. M. Hasan, “FA-EEGNet: Enhancing EEGNet with Frequency-Adaptive Kernels for the classification of stroke rehabilitation movements,” in *2026 5th International Conference on Electrical, Computer & Telecommunication Engineering (ICECTE)*, Jan. 2026, pp. 1–6. doi: [10.1109/ICECTE69292.2026.11429349](https://doi.org/10.1109/ICECTE69292.2026.11429349).

\[Mia26\] M. Miao, W. Fu, H. Zeng, B. Xu, W. Zhang, and W. Hu, “Meta-Learning Enhanced Multi-Source Domain Adaptation for zero-calibration motor imagery EEG decoding.” *Journal of neuroscience methods*, pp. 110742, Mar. 2026, doi: [10.1016/j.jneumeth.2026.110742](https://doi.org/10.1016/j.jneumeth.2026.110742).

\[Liu25\] Y. Liu *et al.*, “Lower limb motor imagery EEG dataset based on the multi-paradigm and longitudinal-training of stroke patients,” *Scientific Data*, vol. 12, Feb. 2025, doi: [10.1038/s41597-025-04618-4](https://doi.org/10.1038/s41597-025-04618-4).
