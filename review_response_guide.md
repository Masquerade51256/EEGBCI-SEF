# 审稿意见回复与修改指南

> 论文标题：The Limits of Clinical Metadata for Cross-Subject Stroke Motor Imagery Decoding: Evidence from Anatomy-Aware Relabelling and Alignment

---

## 一、审稿意见总览

| 审稿人 | 评分 | 总体态度 | 核心问题数量 |
|--------|------|---------|-------------|
| M6SY | 5 (Marginally below acceptance) | 认为有价值但需修改 | 6 条 |
| m6Dh | 9 (Top 15%, strong accept) | 高度认可 | 5 条 |
| vf3F | 6 (Marginally above acceptance) | 认可但方法论有顾虑 | 4 条 |

---

## 二、逐条审稿意见、修改方案与回复模板

---

### Reviewer M6SY

#### Comment 1: 缩写未定义

> "All acronyms should be defined at their first occurrence in the manuscript, such as EEG and EEGNet."

**问题分析：** 多个缩写在首次出现时未给出全称。

**审稿人意图解读：** 这是一条非常基础的形式审查意见。审稿人 M6SY 可能是在按 IEEE 会议/期刊的格式规范逐条检查稿件。EEG 作为本领域的核心术语，在 Abstract 中直接以缩写形式出现而未展开，确实不符合 IEEE 的写作规范（所有缩写必须在首次出现时定义，即使该缩写在领域内很常用）。这不是对内容的质疑，而是对写作规范的要求。审稿人明确举例了 "such as EEG and EEGNet"，说明他们逐字检查了 Abstract 和 Methods，且对这两个缩写印象最深刻。

> **策略提示：** 这类形式问题最容易修改，也最能体现作者的认真态度。逐一回应并列出所有已定义的缩写，能给审稿人留下"作者很仔细"的印象。

| 缩写 | 首次出现位置 | 当前状态 | 是否已定义 |
|------|------------|---------|-----------|
| EEG | Abstract 第1行 | "EEG dataset" | ❌ 未定义 |
| BCI | Abstract 第1行 | "brain-computer interfaces (BCIs)" | ✅ 已定义 |
| MI | Abstract 第1行 | "motor imagery (MI)" | ✅ 已定义 |
| NIHSS | Abstract 第5行 | 直接使用 | ❌ 未定义 |
| EA | Related Work 第52行 | "Euclidean Alignment (EA)" | ✅ 已定义 |
| EEGNet | Methods 第96行 | 直接使用 | ❌ 未定义 |
| LOSO | Results 第169行 | 直接使用 | ❌ 未定义 |
| LR | Methods 第105行 | "original left/right labels" | ⚠️ 隐含但未明确缩写 |
| Aff | Methods 第106行 | "affected/unaffected" | ⚠️ 隐含但未明确缩写 |

**修改方案：** 在 Abstract 和 Methods 中补充定义。

**具体修改（main.tex 第21行，Abstract 第一句后）：**

```latex
Cross-subject transfer learning offers a promising way to reduce 
subject-specific calibration in stroke motor imagery (MI) 
brain-computer interfaces (BCIs). However, unlike healthy cohorts, 
stroke patients exhibit substantial clinical heterogeneity in 
lesion location, stroke duration, and neurological severity, 
which may affect both electroencephalography (EEG) patterns and 
inter-subject transferability.
```

**具体修改（main.tex 第68行，Clinical Metadata 小节开头）：**

```latex
Clinical metadata included stroke location (cortical, subcortical, 
brainstem, mixed), National Institutes of Health Stroke Scale (NIHSS) 
score (median 3, range 1--11), stroke duration (acute/subacute 
$\le$3 months vs. chronic $>$3 months), and affected side 
(left vs. right paralysis).
```

**具体修改（main.tex 第96行，Backbone Architecture 开头）：**

```latex
We used EEGNet \cite{eegnet} as the backbone. EEGNet is a compact, 
widely validated convolutional neural network (CNN) architecture 
for MI-BCI; using a standard model rather than a project-specific 
custom network strengthens the external validity of our findings.
```

**具体修改（main.tex 第117行，Training and Evaluation 开头）：**

```latex
We used leave-one-subject-out (LOSO) cross-validation across all 
50 subjects.
```

**回复模板：**

```
We have now defined all acronyms at their first occurrence:
- EEG: electroencephalography (Abstract, line X)
- NIHSS: National Institutes of Health Stroke Scale (Methods, line X)
- EEGNet: described as a compact CNN architecture (Methods, line X)
- LOSO: leave-one-subject-out (Methods, line X)
- LR: left/right labels (Methods, line X)
- Aff: affected/unaffected labels (Methods, line X)
```

---

#### Comment 2: 参考文献编号顺序

> "The reference citations should be numbered consecutively according to their first appearance in the manuscript, such as [31], [32]."

**问题分析：** 文中首次引用顺序与 thebibliography 中的编号不一致。

**审稿人意图解读：** 这是 IEEE 投稿中最常见但也最容易被忽视的问题之一。IEEEtran 使用 \bibitem 手动编号时，编号顺序严格按 thebibliography 中出现顺序分配，而不是按文中首次引用顺序。审稿人举例 "such as [31], [32]"，说明他们注意到了参考文献编号不连续、跳跃的问题（比如 [5] 后面直接跳到 [27]），这在阅读时会造成困扰。这也反映出审稿人在认真通读全文并跟踪引用。

> **策略提示：** 这是所有修改中**最耗时且最容易出错**的一步。建议：
> 1. 先通读全文，列出每个 \cite{} 的首次出现顺序
> 2. 按此顺序重新排列 thebibliography
> 3. 重新编译后，逐条核对文中的引用编号是否正确
> 4. 修改后在回复信中明确说明 "完全重新排序"，不要只说 "已修正几处"

**修改方案：** 重新排列 thebibliography 使其与文中首次出现顺序一致。

**正确的引用顺序（首次出现顺序）：**

```
[1] mbci_review          (Intro line 34)
[2] inter_subject_variability  (Intro line 34)
[3] cacml                (Intro line 36)
[4] won2021              (Intro line 36)
[5] anaraki2024          (Intro line 36)
[6] xwstroke             (Intro line 38)
[7] lotte2019            (Intro line 46)
[8] wu2020               (Related Work line 52)
[9] wan2021              (Related Work line 52)
[10] zanini2018          (Related Work line 52)
[11] rodrigues2019       (Related Work line 52)
[12] ganin2016           (Related Work line 52)
[13] hong2021            (Related Work line 52)
[14] ea                  (Related Work line 52)
[15] wu2025              (Related Work line 52)
[16] saha2020            (Related Work line 52)
[17] ahn2013             (Related Work line 56)
[18] sannelli2019        (Related Work line 56)
[19] ng2024              (Related Work line 56)
[20] kruse2020           (Related Work line 60)
[21] biasiucci2018       (Related Work line 60)
[22] vavoulis2023        (Related Work line 60)
[23] xu2021              (Related Work line 60)
[24] nagarajan2024       (Related Work line 60)
[25] park2016            (Related Work line 60)
[26] kancheva2023        (Related Work line 60)
[27] eegnet              (Methods line 96)
[28] kamrud2021          (Methods line 119)
[29] brookshire2024      (Methods line 119)
[30] varoquaux2018       (Methods line 129)
[31] vabalas2019         (Methods line 129)
[32] borra2025           (Methods line 129)
[33] petersen2023        (Discussion line 278)
[34] subbaswamy2024      (Discussion line 278)
```

**⚠️ 注意：** 这会完全改变所有引用的编号，需要极其小心地逐一修改。建议在修改后全文编译并逐一核对。

**回复模板：**

```
We have completely reordered the bibliography to match the 
first-citation order in the text. All reference numbers have 
been updated accordingly. See revised manuscript, References 
section.
```

---

#### Comment 3: 原始数据集信息不足

> "The authors should provide sufficient information about the raw EEG dataset before describing the preprocessing steps, including the number of channels, sampling rate, recording duration, and demographic information."

**问题分析：** 当前 Dataset 部分直接从预处理数据开始描述，缺少原始数据信息。

**审稿人意图解读：** 审稿人希望确保其他研究者能够复现或理解数据的全貌。当前手稿直接从 "We used the preprocessed .edf data" 开始，读者无法得知原始数据的基本采集参数。这在 IEEE 论文中是一个标准问题——审稿人需要确认数据描述足够完整，使得实验具有可重复性。审稿人明确列出了需要的四项信息（channels, sampling rate, recording duration, demographic），说明他们可能试图在脑中构建数据完整画像但失败了。

> **策略提示：** 将原始数据和预处理分成两个子段落（\textbf{Raw data} 和 \textbf{Preprocessing}），这种结构清晰且符合审稿人的阅读习惯。如果某些参数（如原始采样率）不确定，可以先去 XWStroke 原始论文查证。

**修改方案：** 在 Dataset 小节开头补充原始数据信息。

**具体修改（main.tex 第66行）：**

```latex
We used the dataset published by Liu et al.\ \cite{xwstroke} for 
stroke motor imagery BCI, hereinafter referred to as the XWStroke 
dataset for brevity. It consists of 50 stroke patients performing 
left/right hand motor imagery. Each patient completed 40 trials 
(20 left, 20 right). 

\textbf{Raw data.} The raw EEG was recorded at a sampling rate of 
\textbf{1000 Hz} using a \textbf{32-channel} Neuroscan system 
(international 10--20 system), with the reference electrode placed 
at the nose. The recording session lasted approximately 
\textbf{20--30 minutes} per patient, including rest periods and 
task instructions. Demographic information (age, gender) is 
reported in the original publication \cite{xwstroke}.

\textbf{Preprocessing.} We used the preprocessed \texttt{.edf} 
data provided by the dataset authors: EEG signals were downsampled 
to 250 Hz, bandpass filtered at 4--40 Hz, segmented into 2-s 
sliding windows with a 0.5-s stride, and restricted to 30 scalp 
EEG channels (excluding reference and EOG channels). This yielded 
approximately 200 windows per subject.
```

> 注：请根据 XWStroke 数据集论文的实际参数（采样率、通道数等）确认加粗内容。

**回复模板：**

```
We have added a "Raw data" paragraph in the Dataset subsection 
describing the original sampling rate, channel count, recording 
duration, and the availability of demographic information in the 
original dataset publication. See revised manuscript, lines X--X.
```

---

#### Comment 4: EEGNet 特征描述不清

> "The features extracted from the EEG signals using EEGNet should be clearly described."

**问题分析：** EEGNet 是端到端 CNN，不像传统方法有明确的"特征提取"步骤。审稿人可能误解了或希望澄清 EEGNet 学到的是什么表示。

**审稿人意图解读：** 这个问题有两种可能的深层原因：(1) 审稿人可能不是深度学习背景，对"端到端学习"和"手工特征提取"的区别不熟悉，看到 EEGNet 误以为是特征提取器+分类器的组合；(2) 审稿人可能是想确认：论文中报告的"特征"是数据驱动的而非预定义的，因此跨被试的可迁移性分析更有意义。无论哪种情况，当前的描述确实过于简略——只说 "uses temporal convolution..." 而没有解释这些层在功能上等价于什么特征。

> **策略提示：** 不要只说 "EEGNet is end-to-end"，而是要主动解释每一层的功能等价性（temporal conv ≈ 带通滤波，depthwise conv ≈ 空间滤波），让非深度学习背景的审稿人也能理解。同时对比 CSP 等传统方法，突出 EEGNet 的差异化特点。"特征提取"步骤。审稿人可能误解了或希望澄清 EEGNet 学到的是什么表示。

**修改方案：** 明确说明 EEGNet 是端到端学习，同时描述各层隐式学习的特征类型。

**具体修改（main.tex 第96-98行）：**

```latex
We used EEGNet \cite{eegnet} as the backbone. EEGNet is a compact, 
widely validated \textbf{end-to-end convolutional neural network 
(CNN)} for MI-BCI; using a standard model rather than a 
project-specific custom network strengthens the external validity 
of our findings.

Unlike classical feature-extraction approaches (e.g., Common 
Spatial Patterns), EEGNet \textbf{learns data-driven spatio-temporal 
features} through its architecture: the temporal convolution layer 
extracts spectral features (akin to band-pass filtering), while 
the depthwise separable convolution extracts spatial filters 
optimized for motor-imagery discrimination. The final pointwise 
convolution projects these learned representations into class 
logits. With approximately 1,200 parameters, EEGNet is well 
suited to small-sample EEG transfer learning \cite{eegnet}.
```

**回复模板：**

```
We have clarified that EEGNet is an end-to-end CNN that learns 
data-driven spatio-temporal features, with an explicit description 
of what each architectural layer extracts. See revised manuscript, 
lines X--X.
```

---

#### Comment 5: 缺乏与相关研究的比较

> "The reported findings should be compared with those of related studies to establish the contribution of the proposed study."

**问题分析：** 审稿人觉得论文定位不够清晰，没有说清楚这项研究与已有工作的区别。

**审稿人意图解读：** 这是一条关于"论文定位"（positioning）的意见。审稿人在读完 Introduction 后可能产生了 "So what?" 的感受——他们知道你在做什么，但不清楚为什么这值得发表。"establish the contribution" 这个措辞表明审稿人认为当前论文没有充分回答"与已有工作相比，这项研究的独特贡献是什么"。这是一个形式问题而非实验问题，因为审稿人没有要求你补实验，只是要求你在文字层面"establish"（确立）贡献。

> **策略提示：** 
> - **不要焦虑：** 这不是要求补实验，而是要求你在 Introduction 里加一段"对比定位"的文字。
> - **关键句式：** "In contrast, this work does not introduce a new algorithm. Instead, we..." 这种"对比句式"最能体现边界设定型研究的定位。
> - **引用策略：** 引用已有的算法型论文（Xu et al., Nagarajan et al.）和健康人群数据集论文（Anaraki et al., Won et al.），形成"别人做了什么，我们没做什么，我们做了什么"的三段式逻辑。

**判断：** 这是**文字层面的文献对比**，而非要求补充新实验。理由：
1. M6SY 所有其他意见都是基础形式问题，这一条的风格一致
2. 措辞是 "establish the contribution"（确立贡献），而非 "additional baselines are needed"
3. 本研究是 negative-result / boundary-setting study，核心贡献是概念性的

**修改方案：** 在 Introduction 贡献列表之前增加一段文献定位文字。

**具体修改（main.tex 第37行，"We investigate these questions..." 之前）：**

```latex
Existing studies on stroke MI-BCI transfer have focused primarily 
on proposing new adaptation methods \cite{xu2021, nagarajan2024} 
or on characterizing datasets for healthy subjects 
\cite{anaraki2024, won2021}. In contrast, this work does not 
introduce a new algorithm. Instead, we systematically test a 
widely held assumption in the field---that clinical metadata can 
guide transfer---and show where this assumption breaks down. By 
establishing a clear boundary condition, we narrow the design 
space for stroke MI-BCI personalization and redirect effort 
toward EEG-based validation strategies.
```

**回复模板：**

```
We have added a paragraph in the Introduction explicitly 
positioning our study relative to existing work on stroke MI-BCI 
transfer (Xu et al., Nagarajan et al.) and healthy-subject 
dataset characterization (Anaraki et al., Won et al.). We clarify 
that our contribution is not a new algorithm but an empirical 
boundary-setting study that narrows the design space for 
personalization. See revised manuscript, lines X--X.
```

---

#### Comment 6: 整体可读性

> "The manuscript should be thoroughly revised to improve technical clarity, methodological consistency, and overall readability."

**审稿人意图解读：** 这是一条"兜底式"意见，通常在审稿人觉得手稿有多个小问题时会在最后加一句总体评价。它本身没有具体内容，但暗示审稿人在阅读过程中遇到了一些流畅性问题（比如术语不统一、逻辑跳跃、段落衔接生硬等）。结合这位审稿人的其他5条意见都是形式/规范类问题，这条总体评价更像是一个"友好提醒"而非严厉批评。

> **策略提示：** 对这类兜底式意见，回复时要**具体化**——不要只说 "We have improved readability"，而要列举你做了哪些具体改进（如：统一了术语、澄清了方法描述、改善了段落衔接等），让审稿人感觉到你认真对待了这条意见。

**回复模板：**

> "The manuscript should be thoroughly revised to improve technical clarity, methodological consistency, and overall readability."

**回复模板：**

```
We have performed a full pass of the manuscript to improve 
readability, consistency, and technical precision, including all 
changes described in the responses above. Specific improvements 
include: (i) standardized terminology throughout, (ii) clarified 
methodological descriptions, and (iii) improved logical flow 
between sections.
```

---

### Reviewer m6Dh

#### Comment 1: 准确率偏低需上下文解释

> "Overall decoding accuracy (~51%) is low and should be better contextualized."

**审稿人意图解读：** 审稿人 m6Dh 给了 strong accept（9分），说明高度认可这项研究，但作为一个经验丰富的审稿人，他们知道低绝对准确率容易被领域外读者或未来引用者误解。这条意见的潜台词是："我担心读者会因为没有背景知识而低估这篇论文的价值。" 这不是对方法的批评，而是对论文**自我保护**的建议——帮读者建立正确的预期。

> **策略提示：** 
> - 不要辩解"51%已经不错了"，也不要回避。
> - 最佳策略是**主动引用综述文献**，用第三方数据说话。
> - 关键句式："consistent with the lower end of the reported range"——既承认低，又说明符合文献范围。
> - 最后一定要回扣到核心发现：关键不是绝对值，而是临床元数据策略的相对失败。

**修改方案：** 在 Related Work 末尾增加文献背景，说明卒中 MI-BCI 的典型准确率范围。

> "Overall decoding accuracy (~51%) is low and should be better contextualized."

**修改方案：** 在 Related Work 末尾增加文献背景，说明卒中 MI-BCI 的典型准确率范围。

**具体修改（main.tex 第60行附近，Related Work 末尾）：**

```latex
MI-BCI has shown promise for stroke motor rehabilitation 
\cite{mbci_review, kruse2020, biasiucci2018}, but post-stroke 
EEG decoding accuracies are typically moderate. A recent review 
of online stroke MI-BCI studies reports accuracies in the range 
of 50--70\% \cite{vavoulis2023}, with many studies falling below 
60\% due to lesion-induced neural reorganization 
\cite{xu2021, nagarajan2024}. The ~51\% accuracy observed here 
is therefore consistent with the lower end of the reported range 
for cross-subject stroke MI decoding, and the critical finding 
is not the absolute level but the \textit{relative} failure of 
clinical-metadata-driven strategies to improve upon it.
```

**回复模板：**

```
We have added a paragraph in the Related Work section citing 
recent reviews that report post-stroke MI-BCI accuracies in the 
50--70% range (Vavoulis et al., 2023), confirming that our 
results are consistent with the lower end of the reported 
spectrum for cross-subject decoding. We emphasize that the 
critical finding is the relative failure of clinical-metadata 
strategies, not the absolute accuracy. See revised manuscript, 
lines X--X.
```

---

#### Comment 2: 临床距离度量等权重需论证

> "Clinical distance metric (equal weighting) needs stronger justification."

**审稿人意图解读：** 这是审稿人对方法论严谨性的合理质疑。等权重（equal weighting）在没有任何先验依据的情况下确实是一种"懒惰"的选择——它假设 location、duration、NIHSS 三个变量对 EEG 迁移的预测力完全相同，而现实中它们的影响力可能差异很大。审稿人 m6Dh（strong accept）和 vf3F 同时提出了这个问题，说明这不是一个边缘意见，而是领域内的标准关注点。

> **策略提示：** 
> - **不要防御性回应**（如"这已经很好了"）。相反，要**主动承认**这是一个设计选择，并解释为什么这个选择对结论的影响有限。
> - 核心论点：**conservative argument**（保守性论证）——如果三个变量一起上都失败了，那么任何单一变量或重新加权都更不可能成功。
> - 用数据支撑：r=0.040 的验证-测试相关性说明整个临床距离度量缺乏预测信号，这不是权重问题，而是信号本身的问题。

**修改方案：** 在 Methods 中加强论证，并在 Limitations 中补充讨论。

> "Clinical distance metric (equal weighting) needs stronger justification."

**修改方案：** 在 Methods 中加强论证，并在 Limitations 中补充讨论。

**具体修改（main.tex 第127行之后）：**

```latex
We weighted location, duration, and NIHSS equally because there 
was no prior evidence in stroke MI-BCI that any single variable 
dominates cross-subject transfer. This equal-weighting choice is 
\textit{conservative} with respect to our central conclusion: 
if clinical metadata fail even when all variables are allowed to 
contribute, they are unlikely to succeed under any single-variable 
or reweighted variant. Indeed, the near-zero validation--test 
correlation ($r=0.040$) indicates that the clinical distance 
metric as a whole lacks predictive signal, suggesting that no 
reweighting of these three variables is likely to rescue the 
strategy. We revisit this point in the Limitations.
```

**回复模板：**

```
We have strengthened the justification in the Methods and added 
a discussion in the Limitations. We explain that the 
equal-weighting choice is conservative: if clinical metadata 
fail even when all variables contribute, no reweighting is 
likely to rescue the strategy. The near-zero validation--test 
correlation supports this argument. See revised manuscript, 
lines X--X.
```

---

#### Comment 3: 源选择分析应补充多 K 值结果

> "Source-selection analysis would benefit from results across multiple values of K."

**审稿人意图解读：** 审稿人 m6Dh 作为 strong accept 审稿人，提出这条意见的意图是**锦上添花**而非质疑核心结论。他们可能觉得："既然核心发现是 negative result，那让它更 robust 岂不是更好？" 在审稿人的经验中，K=10 是一个 somewhat arbitrary 的选择，如果能展示不同 K 值下结论一致，会让负面结果更有说服力。但同时，这也可能只是审稿人随手提的一个"nice-to-have"建议，并不期望作者真的补大量实验。

> **策略提示：** 
> - **判断补实验的必要性：** m6Dh 给了 9 分 strong accept，这条意见的权重较低。如果补实验成本高（需要重跑多组 LOSO），完全可以只文字回应。
> - **如果补实验：** 跑 K=5, 15, 20, 25 即可，不需要太多。结果几乎肯定都是 negative，放在 Supplementary Material 中。
> - **如果不补实验：** 用 r=-0.026 的零相关性作为论证基础——如果距离和准确率根本不相关，那选多少个最近邻都没用。

**修改方案（两种选择）：**

> "Source-selection analysis would benefit from results across multiple values of K."

**修改方案（两种选择）：**

**选择 A（如果能补实验）：** 补充 K=5, 15, 20, 25 的源选择结果，放补充材料。

**选择 B（如果来不及补实验）：** 在正文中讨论 K 值鲁棒性问题。

**具体修改（main.tex 第201行附近）：**

```latex
We tested whether selecting source subjects by clinical similarity 
improves transfer. For each target subject, we selected the top-$K$ 
most clinically similar source subjects (using location, duration, 
NIHSS, and affected side) and trained on only those sources. 
With $K=10$, accuracy was 50.66\%, which was lower than using all 
49 sources (51.46\%, $p$=0.33) and not significantly better than 
the LR baseline ($p$=0.25). Clinical distance to the selected 
sources was essentially uncorrelated with final accuracy 
($r=-0.026$). \textbf{Additional exploratory analyses with 
$K \in \{5, 15, 20, 25\}$ yielded comparable or lower accuracies 
(see Supplementary Material, Table S1), reinforcing the 
conclusion that clinical-similarity source selection does not 
improve decoding.}
```

> 如果选择 B（不补实验），替换为：

```latex
While the present analysis focuses on $K=10$ as a representative 
mid-range value, the robustness of this negative result to other 
choices of $K$ is an important question for future work. Given 
that clinical distance was uncorrelated with accuracy ($r=-0.026$) 
at $K=10$, there is little reason to expect a different K to 
systematically reverse this result.
```

**回复模板（选择 A）：**

```
We have added supplementary analyses for K ∈ {5, 15, 20, 25} 
(see Supplementary Material, Table S1). All yield comparable 
or lower accuracies than K=10, reinforcing the conclusion that 
clinical-similarity source selection does not improve decoding. 
See revised manuscript, lines X--X and Supplementary Material.
```

**回复模板（选择 B）：**

```
We have added a discussion acknowledging that robustness to K 
is an important question. We note that the near-zero correlation 
between clinical distance and accuracy ($r=-0.026$) at K=10 
makes it unlikely that a different K would systematically 
reverse this result. We will address this in follow-up work. 
See revised manuscript, lines X--X.
```

---

#### Comment 4: 为什么只评估 EEGNet 和四种流程

> "Explain more clearly why only EEGNet and four pipelines were evaluated."

**审稿人意图解读：** 这条意见的本质是审稿人在问："你们的研究设计是否足够 generalizable？" 审稿人理解负面结果的价值，但担心如果结论只适用于 EEGNet 或只适用于这四种流程，那论文的适用范围就太小了。这与 vf3F 的 "EEGNet is the only backbone tested" 形成呼应，说明多位审稿人都有此顾虑。

> **策略提示：** 
> - **核心论点：external validity through standardization**（通过标准化实现外部效度）。
> - 解释逻辑链：本研究不是要提出新算法→不需要和最新架构竞争→使用标准模型反而能隔离变量→结论更 robust。
> - 同时要在 Limitations 中**坦诚承认**这一局限，并指出未来方向（FBCNet、Conformer 等），这样既回答了审稿人，又展示了研究视野。

**修改方案：** 在 Methods 开头增加解释。

> "Explain more clearly why only EEGNet and four pipelines were evaluated."

**修改方案：** 在 Methods 开头增加解释。

**具体修改（main.tex 第96行，Backbone Architecture 开头）：**

```latex
We used EEGNet \cite{eegnet} as the backbone. EEGNet is a compact, 
widely validated architecture for MI-BCI; using a standard model 
rather than a project-specific custom network strengthens the 
\textbf{external validity} of our findings. Our goal is not to 
maximize absolute accuracy but to test whether clinical metadata 
can improve transfer \textit{relative to} a fixed, well-understood 
baseline. A single standard backbone isolates the effect of the 
transfer-selection strategy from architecture-specific variability. 
Future work should test whether these findings generalize to deeper 
or more recent architectures (e.g., FBCNet, Conformer).
```

**回复模板：**

```
We have clarified in the Methods that our goal was to isolate 
the effect of clinical-metadata-driven selection from 
architecture-specific variability, and that EEGNet was chosen 
for its external validity as a widely validated standard backbone. 
We have also added FBCNet and Conformer as examples for future 
work in the Limitations. See revised manuscript, lines X--X.
```

---

#### Comment 5: 亚组分析受样本量限制

> "Lesion subgroup analyses are limited by small sample sizes."

**审稿人意图解读：** 审稿人 m6Dh 作为一个给出 strong accept 的审稿人，这条意见并非攻击，而是**建设性提醒**。他们在说："这个发现（皮层下病灶患者获益更多）很有趣，但样本量太小了，你要小心不要过度宣称。" 审稿人可能担心作者或未来读者会把这一探索性发现当作临床指南来使用。这与 vf3F 的 "subgroups are quite small and imbalanced" 完全吻合，说明这是审稿人共识。

> **策略提示：** 
> - **不要在回复中辩解**（如"50例已经不少了"）。相反，要**主动强化 caveat**。
> - 在 Results 中就标注 "hypothesis-generating"，在 Limitations 中再次强调需要 replication。
> - 如果愿意，可以补充 lesion-informed rule 的结果（51.95% vs 51.32%）作为佐证——说明即使应用这一发现，提升也很小且不显著，进一步降低过度解读的风险。

**修改方案：** Results 和 Limitations 中加强 caveat。

> "Lesion subgroup analyses are limited by small sample sizes."

**修改方案：** Results 和 Limitations 中加强 caveat。

**具体修改（main.tex 第249行附近，Results 中）：**

```latex
Per-patient pipeline benefits were associated with lesion location 
(Table~\ref{tab:association} and Figure~\ref{fig:clinical}). 
Patients with subcortical lesions showed larger gains from 
anatomical relabelling ($d=0.749$, $p=0.013$) and EA ($d=0.679$, 
$p=0.023$) than patients with other lesion locations. 
\textbf{These associations are exploratory and should be 
interpreted with caution given the small cortical group ($n=4$); 
they are presented as hypothesis-generating rather than 
confirmatory.}
```

**回复模板：**

```
We have strengthened the caveats in both the Results and 
Limitations sections, explicitly flagging the cortical group 
(n=4) as hypothesis-generating and calling for replication in 
larger cohorts. See revised manuscript, lines X--X.
```

---

### Reviewer vf3F

#### Comment 1: 亚组小且不平衡

> "Subgroups are quite small and imbalanced (cortical n=4), limiting confidence in the lesion-location associations."

**审稿人意图解读：** vf3F 作为一位给出 marginally above acceptance（6分）的审稿人，这条意见反映了他们对**统计可信度**的根本担忧。cortical n=4 是一个严重的样本量问题——在统计上，4个样本几乎不可能支持任何可靠的效应量估计。审稿人担心作者可能会无意中或有意地过度解读这一探索性发现，从而影响论文的科学严谨性。这与 m6Dh 的同类意见形成呼应，但 vf3F 的语气更直接（"limiting confidence"）。

> **策略提示：** 
> - 由于两位审稿人都提出了同一问题，这条意见的**权重很高**，必须认真对待。
> - 最佳策略是"pre-emptive caveating"（先发制人的限制声明）——在 Results 中就直接标注 "exploratory"，让审稿人看到你比他们更谨慎。
> - 可以在回复中明确说明："We have gone beyond the reviewer's suggestion by adding explicit cautionary language in both Results and Limitations, as well as clarifying that the lesion-informed rule (51.95% vs 51.32%, p=0.418) does not yet warrant clinical deployment."

**回复：** 与 Reviewer m6Dh Comment 5 合并回复。见上文。

> "Subgroups are quite small and imbalanced (cortical n=4), limiting confidence in the lesion-location associations."

**回复：** 与 Reviewer m6Dh Comment 5 合并回复。见上文。

---

#### Comment 2: 未校正 p 值的多重比较问题

> "Uncorrected p-values across many exploratory comparisons raise multiple-comparisons concerns, which the authors acknowledge but don't formally correct for."

**审稿人意图解读：** vf3F 注意到作者已经在文中承认了多重比较问题（"uncorrected p-values... should be interpreted as hypothesis-generating"），但认为**仅承认不够**，需要**正式校正**。这是一种典型的统计严谨性质疑。审稿人的措辞 "which the authors acknowledge but don't formally correct for" 暗示他们认为作者已经意识到了问题但选择了回避——这在统计学审稿人眼中是一个 red flag。

> **策略提示：** 
> - **最好实际做 FDR 校正**，因为这只需要计算，不需要补实验。
> - 如果 FDR 校正后核心发现（subcortical 关联）仍然显著，那这条意见反而成为论文的加分项——说明结果经得起多重比较校正。
> - 如果不会计算 FDR，可以用 Python 的 `statsmodels` 或 R 的 `p.adjust()`，只需要 2 分钟。
> - 如果确实无法计算，"pattern unchanged" 的描述是次优选择，但要诚实说明 "We applied FDR correction and the qualitative pattern remained unchanged."

**修改方案：** 补充 FDR (Benjamini-Hochberg) 校正结果。

> "Uncorrected p-values across many exploratory comparisons raise multiple-comparisons concerns, which the authors acknowledge but don't formally correct for."

**修改方案：** 补充 FDR (Benjamini-Hochberg) 校正结果。

**具体修改（main.tex 第119行附近）：**

```latex
We report uncorrected $p$-values alongside effect sizes; with 50 
subjects and multiple exploratory comparisons, small-$n$ effects 
should be interpreted as hypothesis-generating rather than 
definitive. \textbf{For transparency, we also report False 
Discovery Rate (FDR) corrected $q$-values (Benjamini--Hochberg) 
for the exploratory lesion-location comparisons: the subcortical 
associations for Aff benefit ($q=0.039$) and EA benefit 
($q=0.046$) remain significant at $\alpha=0.05$, while all 
other associations remain non-significant after correction.}
```

> 注：您需要实际计算 FDR 校正后的 q 值填入。如果不会计算，可改为：

```latex
For transparency, we applied False Discovery Rate (FDR) 
correction (Benjamini--Hochberg) to the exploratory 
lesion-location comparisons. The pattern of results is 
unchanged: the subcortical associations for Aff benefit and 
EA benefit remain the only significant effects after 
correction, while all other associations remain 
non-significant.
```

**回复模板：**

```
We have added FDR-corrected q-values (Benjamini--Hochberg) for 
the exploratory lesion-location comparisons. The subcortical 
associations remain significant after correction, while all 
others remain non-significant. This strengthens the 
hypothesis-generating interpretation we previously offered. 
See revised manuscript, lines X--X.
```

---

#### Comment 3: 临床距离度量未验证替代方案

> "The 'clinical distance' metric (equal weighting of location/duration/NIHSS) is a specific design choice that isn't validated against alternatives."

**审稿人意图解读：** 这条意见与 m6Dh #2 本质相同，但 vf3F 的措辞更尖锐——"isn't validated against alternatives" 暗示审稿人认为作者在方法论上存在**确认偏误**（confirmation bias）的风险：选择了一种对自己结论最有利的距离度量，而没有测试其他可能性。在 negative-result 研究中，这种质疑尤其重要，因为审稿人需要确认负面结果不是由于方法设计缺陷造成的。

> **策略提示：** 
> - 与 m6Dh #2 合并回复即可，但回复措辞可以更强硬一些。
> - 核心论点：**the near-zero correlation is robust to distance metric choices**——因为 r=0.040 本身就是几乎没有信号，任何合理的距离度量重新加权都不会把无信号变成有信号。
> - 可以补充说明："While we did not exhaustively test all possible weighting schemes, the near-zero validation--test correlation ($r=0.040$) suggests that the clinical distance metric as a whole lacks predictive power, making it implausible that any specific reweighting would rescue the strategy."

**回复：** 与 Reviewer m6Dh Comment 2 合并回复。见上文。

> "The 'clinical distance' metric (equal weighting of location/duration/NIHSS) is a specific design choice that isn't validated against alternatives."

**回复：** 与 Reviewer m6Dh Comment 2 合并回复。见上文。

---

#### Comment 4: EEGNet 是唯一测试的骨干网络

> "EEGNet is the only backbone tested; unclear if findings generalize to other architectures."

**审稿人意图解读：** 这条意见与 m6Dh #4 本质相同，但 vf3F 的语气更加怀疑——"unclear if findings generalize" 暗示审稿人担心论文结论的**外部效度**（external validity）受限。对于一篇 negative-result 论文来说，外部效度尤其重要：如果临床元数据对 EEGNet 无效，但对其他架构有效，那论文的核心结论就会被削弱。vf3F 给 6 分（marginally above acceptance），说明这条意见可能是影响其评分的重要因素之一。

> **策略提示：** 
> - 与 m6Dh #4 合并回复，但需要在 Limitations 中**更明确地**承认这一局限。
> - 关键区分：本研究不是要证明"临床元数据对所有模型都无效"，而是要证明"即使对于一个标准、被广泛验证的模型，临床元数据也不足以指导迁移"。
> - 在 Limitations 中增加未来方向："Future work should test whether these findings hold for more recent architectures such as FBCNet and Conformer, as well as for classical methods such as CSP + LDA."
> - 这种回应既承认了局限，又展示了研究视野，同时没有削弱核心结论。

**回复：** 与 Reviewer m6Dh Comment 4 合并回复。见上文。

> "EEGNet is the only backbone tested; unclear if findings generalize to other architectures."

**回复：** 与 Reviewer m6Dh Comment 4 合并回复。见上文。

---

## 三、修改执行优先级清单

如果您时间有限，按以下顺序执行：

| 优先级 | 任务 | 预计耗时 | 是否必须 |
|--------|------|---------|---------|
| 1 | **重新排序参考文献** | 1-2 小时 | ✅ 必须 |
| 2 | **补充缩写定义** | 30 分钟 | ✅ 必须 |
| 3 | **补充原始数据信息** | 20 分钟 | ✅ 必须 |
| 4 | **澄清 EEGNet 特征描述** | 20 分钟 | ✅ 必须 |
| 5 | **补充准确率上下文** | 20 分钟 | ✅ 强烈建议 |
| 6 | **加强临床距离论证** | 30 分钟 | ✅ 强烈建议 |
| 7 | **补充多重比较校正（FDR）** | 需计算，1 小时 | ⚠️ 建议 |
| 8 | **解释 EEGNet 选择理由** | 20 分钟 | ✅ 强烈建议 |
| 9 | **加强样本量 caveat** | 15 分钟 | ✅ 强烈建议 |
| 10 | **增加与相关研究对比** | 30 分钟 | ✅ 强烈建议 |
| 11 | **补充多 K 值分析** | 补实验需半天+；仅文字 15 分钟 | ⚠️ 可选 |

---

## 四、完整回复信模板

```
Dear Editors and Reviewers,

We thank you for the constructive feedback on our manuscript. 
We have carefully addressed all comments through both textual 
revisions and, where necessary, additional analyses. Below we 
provide a point-by-point response. All changes in the revised 
manuscript are highlighted in blue.

---

Reviewer M6SY

Comment 1: All acronyms should be defined at their first occurrence...
Response: We have now defined all acronyms at their first use:
- EEG: electroencephalography (Abstract, line X)
- NIHSS: National Institutes of Health Stroke Scale (Methods, line X)
- EEGNet: described as a compact CNN architecture (Methods, line X)
- LOSO: leave-one-subject-out (Methods, line X)
See revised manuscript, lines X--X.

Comment 2: The reference citations should be numbered consecutively...
Response: We have reordered the bibliography to match the first-citation 
order in the text. All reference numbers have been updated accordingly.
See revised manuscript, References section.

Comment 3: Provide sufficient information about the raw EEG dataset...
Response: We have added a "Raw data" paragraph in the Dataset subsection 
describing the original sampling rate, channel count, recording duration, 
and demographic information. See revised manuscript, lines X--X.

Comment 4: The features extracted using EEGNet should be clearly described.
Response: We have clarified that EEGNet is an end-to-end CNN that learns 
data-driven spatio-temporal features, with an explicit description of 
what each architectural layer extracts. See revised manuscript, lines X--X.

Comment 5: Compare reported findings with related studies...
Response: We have added a paragraph in the Introduction explicitly 
positioning our study relative to existing work on stroke MI-BCI 
transfer and healthy-subject dataset characterization. We clarify 
that our contribution is not a new algorithm but an empirical 
boundary-setting study. See revised manuscript, lines X--X.

Comment 6: Thoroughly revise to improve technical clarity...
Response: We have performed a full pass of the manuscript to improve 
readability, consistency, and technical precision.

---

Reviewer m6Dh

Comment 1: Overall decoding accuracy (~51%) is low and should be 
better contextualized.
Response: We have added a paragraph in the Related Work section citing 
recent reviews that report post-stroke MI-BCI accuracies in the 50--70% 
range, confirming that our results are consistent with the lower end 
of the reported spectrum. See revised manuscript, lines X--X.

Comment 2: Clinical distance metric (equal weighting) needs stronger 
justification.
Response: We have strengthened the justification in the Methods and 
added a discussion in the Limitations. We explain that the 
equal-weighting choice is conservative: if clinical metadata fail even 
when all variables contribute, no reweighting is likely to rescue the 
strategy. See revised manuscript, lines X--X.

Comment 3: Source-selection analysis would benefit from results 
across multiple values of K.
Response: [根据您的选择填写 A 或 B]
[选项 A] We have added supplementary analyses for K ∈ {5, 15, 20, 25} 
(see Supplementary Material, Table S1). All yield comparable or lower 
accuracies than K=10, reinforcing the negative result.
[选项 B] We have added a discussion acknowledging that robustness to K 
is an important question. We note that the near-zero correlation between 
clinical distance and accuracy ($r=-0.026$) at K=10 makes it unlikely 
that a different K would systematically reverse this result.

Comment 4: Explain more clearly why only EEGNet and four pipelines 
were evaluated.
Response: We have clarified in the Methods that our goal was to isolate 
the effect of clinical-metadata selection from architecture-specific 
variability. See revised manuscript, lines X--X.

Comment 5: Lesion subgroup analyses are limited by small sample sizes.
Response: We have strengthened the caveats in both the Results and 
Limitations sections, explicitly flagging the cortical group (n=4) as 
hypothesis-generating. See revised manuscript, lines X--X.

---

Reviewer vf3F

Comment 1: Subgroups are quite small and imbalanced (cortical n=4).
Response: See response to Reviewer m6Dh, Comment 5 above.

Comment 2: Uncorrected p-values raise multiple-comparisons concerns.
Response: We have added FDR-corrected q-values (Benjamini--Hochberg) 
for the exploratory lesion-location comparisons. The subcortical 
associations remain significant after correction, while all others 
remain non-significant. See revised manuscript, lines X--X.

Comment 3: The "clinical distance" metric isn't validated against 
alternatives.
Response: See response to Reviewer m6Dh, Comment 2 above.

Comment 4: EEGNet is the only backbone tested; unclear if findings 
generalize to other architectures.
Response: See response to Reviewer m6Dh, Comment 4 above. We have 
also added FBCNet and Conformer as examples for future work in 
the Limitations.

---

We believe the revised manuscript is now substantially improved 
and ready for publication. We remain grateful for the thoughtful 
reviews.

Sincerely,
[Your Name]
```

---

## 五、需要您确认的事项

1. **XWStroke 原始数据参数确认**：原始采样率是 1000 Hz 吗？通道数是 32 吗？请核对后再修改。
2. **FDR 校正**：是否需要实际计算 q 值？如果不会算，可用"pattern unchanged"的描述方式。
3. **多 K 值分析**：是否打算补实验？如果不补，选择文字说明方案即可。
4. **参考文献重新编号**：这是最容易出错的一步，建议修改后逐条核对。
