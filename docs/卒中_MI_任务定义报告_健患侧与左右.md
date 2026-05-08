# 卒中 MI 任务定义报告 健患侧与左右

##### [**Undermind**](https://undermind.ai)

---

## 核心判断

卒中患者运动想象 EEG 的跨被试建模里，**左与右手分类**并不是天然最合理的唯一任务定义，但也不应被机械地整体替换成**健侧与患侧分类**。现有证据更支持一个更细的判断：当研究目标是对接公开数据、复现标准 benchmark、或分析左右侧化本身时，左与右仍然有价值 (Liu et al. 2024; Raza et al. 2020; Lv et al. 2025)。当研究目标是康复型 BCI、跨被试泛化、以及尽快稳定检测患侧运动意图时，按功能侧对齐的任务定义往往更接近卒中后的真实神经生理，尤其是**患侧意图对静息** affected hand versus rest 和**患侧意图的个体化半球解码**更有直接支持 (Cruz et al. 2025; Mansour et al. 2022; Bundy et al. 2012, 2017; Li et al. 2019; Jia et al. 2022)。

真正的问题并不是“左与右”还是“健与患”二选一，而是卒中后运动意图的判别轴已经不再稳定地等同于健康人的解剖学左右轴。病灶位置、损伤严重度、是否累及初级运动皮层 primary motor cortex、以及后续的双侧重组 bilateral reorganization，会共同决定运动想象是更偏向同侧优势 ipsilateral dominance、对侧优势 contralateral dominance，还是双侧协同 bihemispheric recruitment (Park et al. 2016; Fallani et al. 2013; Mansour et al. 2022; Jia et al. 2022; Kancheva et al. 2023)。因此，从任务设计角度看，**健侧与患侧比左与右更接近功能逻辑，但它仍然不是最终层面的统一对齐单位**。比单纯改标签更关键的，是把半球对齐 hemisphere alignment、病灶分层 lesion stratification 和个体化通道选择 individualized channel selection 一并纳入。

## 研究现状

卒中运动想象 BCI 里的任务定义，实际上已经分成四条并行路线，而不是单一传统范式。

| 路线 | 中文表述 | 典型目标 | 优点 | 主要问题 | 代表文献 |
|:---|:---|:---|:---|:---|:---|
| 解剖左右任务 | 左手对右手 | 对标公开数据与标准 MI 文献 | 可比性强，最接近传统二分类 MI | 默认健康式侧化，容易掩盖卒中后功能重组 | (Liu et al. 2024; Raza et al. 2020; Xu et al. 2021; Lv et al. 2025) |
| 功能侧任务 | 患侧对健侧 | 对齐偏瘫功能状态 | 比左与右更贴近康复问题 | 健侧并不等于正常侧，组内异质性仍大 | (Qiu et al. 2017; Fallani et al. 2013; Lee et al. 2023; Graziadio et al. 2012) |
| 单侧检测任务 | 患侧对静息 | 直接驱动患手反馈与闭环康复 | 临床目标清楚，常比左右更易解码 | 与标准二分类左右任务不完全可比 | (Cruz et al. 2025; Frolov et al. 2017; Ono et al. 2014; Cantillo-Negrete et al. 2018; Nagarajan et al. 2023) |
| 个体化重组任务 | 依患者实际重组选择 ipsilesional 或 contralesional 或双侧特征 | 尽量贴合重组后脑区 | 生理合理性最强 | 统一 benchmark 难，工程更复杂 | (Mansour et al. 2022; Spüler et al. 2018; Li et al. 2019; Jia et al. 2022; Bundy et al. 2012, 2017) |

从公开数据和算法论文看，**左与右**目前仍是最常见的标准入口。急性卒中公开数据集 (Liu et al. 2024) 就是按左手与右手运动想象采集的，后续算法工作如 (Lv et al. 2025) 与 (Xu et al. 2021) 也沿着这一定义展开。这一路线的优势是比较清晰，方便与健康人 MI 文献和已有深度模型直接对接。

但从康复 BCI 与神经机制研究看，领域早已不断偏离纯粹的解剖左右范式。大量闭环康复工作实际关注的是**能否稳定检测患侧运动意图，并将反馈施加到患手**，因此核心任务常常已经变成了患侧意图对静息，或患侧意图在特定半球信号中的检测 (Frolov et al. 2017; Ono et al. 2014; Cantillo-Negrete et al. 2018; Bundy et al. 2017)。换句话说，很多临床导向研究虽然没有把“左与右改名为健与患”，但它们在操作层面已经把任务轴从左右竞争转成了患侧意图检测。

## 支持功能侧重定义的证据

### 直接比较证据

当前最直接的支持来自 (Cruz et al. 2025)。这篇工作正面比较了传统**左手对右手**和替代性的**患侧运动想象对静息**。其出发点非常明确：卒中后典型的对侧激活 contralateral activation 常常减弱，导致传统左右对分不再稳定。该研究在 FBCSP 和 EEGNet 上都观察到，患侧对静息的表现优于左右对分，由此得出一个很实用的结论：对康复 BCI 而言，先检测患侧意图，再逐步过渡到更复杂范式，可能比一开始坚持左右分类更合适 (Cruz et al. 2025)。

这条证据虽不是“患侧对健侧”与“左对右”的一一正面比较，但它已经说明了关键一点：**当卒中破坏了左右对分所依赖的侧化结构时，按患侧意图重新组织任务，确实可能更优。**

### 神经生理证据

多篇工作指出，患侧运动想象在卒中后并不是健康状态下左右镜像的简单翻版。网络拓扑分析 (Fallani et al. 2013) 显示，患手运动想象 associated hand motor imagery 往往伴随更低的小世界性 small-worldness 与局部效率 local efficiency，并伴随异常增强的半球间连接 interhemispheric connectivity。与健侧手运动想象相比，患侧任务更容易招募额叶与顶叶补偿区，而不是只依赖典型的对侧感觉运动区 contralateral sensorimotor cortex (Fallani et al. 2013)。

病灶位置研究 (Park et al. 2016) 进一步表明，若病灶累及初级运动皮层，运动想象时可能出现**同侧优势反转**，即未损伤半球的反应强于传统意义上的对侧半球。该文明确提出，EEG 神经康复分析应按病灶位置实施，而不能默认所有患者都保留健康式对侧 ERD 模式 (Park et al. 2016)。系统综述 (Kancheva et al. 2023) 也支持这一点，认为皮层病灶与皮层下病灶对 alpha 与 beta 节律、以及 ipsilesional 与 contralesional ERD 的影响并不相同，卒中后的 sensorimotor rhythm 已明显偏离健康左右模板 (Kancheva et al. 2023)。

针对偏瘫侧别的专门分析 (Lee et al. 2023) 发现，患侧上肢运动想象时，受试者常出现更强的同侧 ERD 和更弱的半球差异，提示损伤后功能连接已被重新组织。这类结果共同说明，**患侧意图**在神经表征上更像一个卒中特异状态，而不是“左任务”或“右任务”在健康脑中的对称变体 (Fallani et al. 2013; Lee et al. 2023)。

### 康复闭环与半球替代证据

如果从康复闭环的实际控制逻辑看，支持功能侧重定义的证据更强。早期工作 (Bundy et al. 2012) 已证明，偏瘫卒中患者可以利用**未损伤半球中的同侧运动信号**来控制 BCI。该文的核心含义不是简单说“左脑也能控左手”或“右脑也能控右手”，而是说明在严重偏瘫中，控制信号的来源应按患手相关的功能重组来找，而不是按固定左右定位。

后续家庭化干预研究 (Bundy et al. 2017) 用未损伤半球信号驱动患手外骨骼并观察到功能改善，进一步强化了这一点。大样本分析 (Mansour et al. 2022) 则表明，多数患者可以使用 ipsilesional 或 contralesional 半球来完成患手 MI 控制。更重要的是，重度受损患者常在 contralesional 半球上获得更高的 BCI 准确率，而轻中度患者可能更依赖 ipsilesional 半球 (Mansour et al. 2022)。这意味着跨被试模型真正应对齐的，不一定是解剖左与右，也不一定只是健侧与患侧，而是**患侧意图所依赖的重组通路**。

完全瘫痪患者的设计研究 (Spüler et al. 2018) 也得到相近结论。该文比较 ipsilesional、contralesional 和双侧信号源后发现，双侧信号总体优于单侧，而 ipsilesional 与 contralesional 单独使用时并无显著差别 (Spüler et al. 2018)。这说明单纯按左与右组织特征空间，很可能没有抓住最稳定的判别轴；对这类患者，双侧与半球重组相关的表征更重要。

## 反对简单整体重映射的证据

### 患侧并不自动更容易判别

最直接的保留意见来自 (Qiu et al. 2017)。这篇研究专门比较了患手与健手运动想象活动差异，但其结果并不支持“患手总是更差”或“按健患侧重标后一定更合理”。作者发现，分类表现并未稳定地由哪只手是患侧所决定，一些患者反而表现出明显的右手优势，而且这一现象更可能与惯用手 familiar skilled action 有关，而不是单纯由偏瘫侧决定 (Qiu et al. 2017)。这说明“患侧”虽然更贴近康复语境，但其 EEG 可分性并不比“左与右”天然更整齐。

### 健侧并不等于正常侧

如果把左与右全部重命名为健侧与患侧，最容易忽视的一点是：**健侧不是健康侧。** (Graziadio et al. 2012) 直接把“未受影响侧 unaffected side”称为一种神话。该研究显示，单侧卒中后，非病灶侧 corticospinal system 同样会发生系统性重组，且恢复更依赖两侧系统重新达到平衡，而不是单纯保持一侧正常 (Graziadio et al. 2012)。因此，健侧与患侧在功能上并非健康人与病人的简单二元对立。

这会带来一个重要方法学后果。若把所有 trial 直接改成健侧与患侧两类，模型依然可能面对很大的类内异质性。因为不同患者的“患侧 trial”可能分别由 ipsilesional 聚焦、contralesional 招募、或双侧广泛激活支持；而“健侧 trial”也未必是健康模板 (Graziadio et al. 2012; Jia et al. 2022)。换句话说，**健患侧标签比左右标签更有康复意义，但它不是自动更齐整的机器学习标签。**

### 仍然存在真实的左右差异与 benchmark 价值

急性期数据研究 (Lv et al. 2025) 说明，即便在卒中患者中，左与右手运动想象之间仍可观察到可判别的微状态 microstate 差异。这类研究关注的正是侧化差异本身，因此左与右标签并非完全失效 (Lv et al. 2025)。公开数据 (Liu et al. 2024) 采用左右任务，也说明这一范式在数据构建和算法比较上仍然有现实价值。

对跨研究可比性而言，这一点尤其重要。若一个项目完全放弃左与右主任务，虽然可能得到更贴合临床的结果，但会失去与已有 acute stroke benchmark 的直接可比性 (Liu et al. 2024; Raza et al. 2020; Xu et al. 2021)。因此，从论文策略看，**左与右更像标准评测任务，健侧与患侧更像临床导向任务**。前者用于横向比较，后者用于提出更合理的卒中特异问题。

## 个体化而非全局改名的证据更强

如果把所有证据放在一起看，文献最一致的方向其实不是“把左与右改成健与患”，而是“放弃健康式固定模板，转向个体化重组建模”。

个体化通道研究 (Li et al. 2019) 发现，许多患者在患手任务中的 ERD 并不局限于传统对侧 SM1。作者把患者分成 ERD 扩散、ERD 消失、ERD 局限于 SM1 等不同模式，并证明按个体最强 ERD 通道选取特征，显著优于固定的 contralateral SM1 通道策略 (Li et al. 2019)。这说明即便标签改成患侧，若仍坚持固定左右通道假设，依然会错失关键信号。

个体化康复训练研究 (Jia et al. 2022) 更进一步，把卒中患者的激活模式归纳为三类：双侧广泛激活 bilateral widespread activation、病灶侧聚焦 ipsilesional focusing、和健侧招募 contralesional recruitment。作者据此设计 personalized BMI training，并指出若研究强行固定为某一侧半球主导，反而会被不同患者的重组模式混淆 (Jia et al. 2022)。这篇工作的启示非常直接：**比标签重命名更重要的，是把患者先按重组模式分层。**

从迁移学习角度看，这也有技术上的支持。不同标签集合 domain adaptation with different label sets 的工作 (He and Wu 2019) 说明，BCI 迁移并不要求源域和目标域拥有完全相同的标签空间。也就是说，如果研究最终认为卒中里“患侧对健侧”或“患侧对静息”比“左对右”更合理，方法上并不是走不通。标签空间不同，可以通过 label alignment 等方式处理 (He and Wu 2019)。这减轻了一个常见顾虑：为了和健康数据或现有 benchmark 对接，就必须永远保持左与右标签不变。实际上，**任务定义可以优先服从神经生理合理性，再用迁移方法处理标签差异。**

## 对当前问题的判断框架

### 什么情况下左与右更合适

| 场景 | 原因 | 适用判断 |
|:---|:---|:---|
| 需要与公开 acute stroke 数据直接对标 | (Liu et al. 2024) 等数据按左与右采集，现有基线也沿此设置 | 左与右应保留为主 benchmark |
| 研究目标是分析左右侧化是否仍存在 | 部分急性期研究仍能观测到左右差异 | 左与右是问题本身的一部分 (Lv et al. 2025) |
| 需要兼容健康人 MI 预训练模型 | 健康源域通常也是左与右或手部 MI 任务 | 左与右更利于直接迁移 (Nagarajan et al. 2023) |
| 论文需要与既有算法文献横向比较 | 多数现有模型仍以左右二分类作为报告入口 | 左与右便于对表 (Raza et al. 2020; Xu et al. 2021) |

### 什么情况下健侧与患侧更合适

| 场景 | 原因 | 适用判断 |
|:---|:---|:---|
| 研究目标是康复而非纯解码 | 康复真正关心的是患手意图与患手反馈闭环 | 健患侧更符合任务语义 (Frolov et al. 2017; Ono et al. 2014; Cantillo-Negrete et al. 2018) |
| 跨被试样本混合了左偏瘫与右偏瘫 | 原始左与右标签会把功能相同 trial 分到不同类 | 健患侧有助于做功能侧对齐 |
| 病灶累及 SM1 或典型侧化受损明显 | 左右对侧模板被破坏，患侧意图更依赖重组网络 | 健患侧更有生理逻辑 (Park et al. 2016; Mansour et al. 2022; Fallani et al. 2013) |
| 只需检测患手意图是否出现 | 临床控制上往往只需触发患手外骨骼或 FES | 患侧对静息通常更直接 (Cruz et al. 2025; Bundy et al. 2017) |

### 什么情况下单纯改成健患侧仍然不够

| 情况 | 为什么不够 | 更合适的补充 |
|:---|:---|:---|
| 患者间重组模式差异极大 | 患侧类内仍高度异质 | 按病灶与严重度分层 (Park et al. 2016; Kancheva et al. 2023) |
| 健侧并不正常 | 健侧标签内部也含病理重组 | 用双侧或半球平衡指标辅助 (Graziadio et al. 2012) |
| 特征来源不固定 | 有人依赖 ipsilesional，有人依赖 contralesional | 做半球对齐与个体化通道选择 (Mansour et al. 2022; Li et al. 2019; Jia et al. 2022) |
| 目标是跨域迁移 | 健康源域与卒中目标域标签可能不同 | 用 label alignment 或分阶段迁移 (He and Wu 2019; Nagarajan et al. 2023) |

## 面向实验设计的可行计划

如果目标是为后续组会和实验设计服务，最稳妥的做法不是立刻宣布“以后全部改成健侧与患侧”，而是把任务定义本身做成一个可检验变量。

### 计划一 先做标签重定义的描述性验证

在 (Liu et al. 2024) 这类原始为左与右任务的数据上，先把每名患者的 trial 同时保留两套标签。

- 左手与右手
- 患侧与健侧
- 若数据允许，再派生患侧与静息或基线窗口

第一步先不追模型提分，而是做三类描述分析：

- laterality coefficient 或 ERD laterality，比较左与右标签下和健患侧标签下的组内一致性 (Park et al. 2016)
- 功能连接 functional connectivity 与网络拓扑，检查患侧标签是否比左右标签更集中地反映补偿网络 (Fallani et al. 2013; Lee et al. 2023)
- 病灶侧、严重度、是否累及 SM1 的分层图，判断任务重定义是否只在特定亚组有效 (Park et al. 2016; Mansour et al. 2022; Kancheva et al. 2023)

这一步的目标是先回答一个最根本的问题：**重映射是否真的减少了跨被试异质性。**

### 计划二 做同协议的任务定义对照实验

在完全相同的预处理、完全相同的 leave one subject out 协议下，设置至少三组任务。

| 任务   | 标签空间   | 研究意义               |
|:-------|:-----------|:-----------------------|
| 任务 A | 左手对右手 | benchmark 与文献对照   |
| 任务 B | 患侧对健侧 | 功能侧重定义的直接检验 |
| 任务 C | 患侧对静息 | 康复控制导向检验       |

对每个任务同时报告：

- 平均准确率与平衡准确率 balanced accuracy
- 被试间标准差
- 低表现患者是否被抬升
- lesion side 与 severity 分层结果

若任务 B 平均分未显著高于任务 A，但方差明显更小，或在重度患者中更稳，这同样是有价值的正结果。因为你的核心问题本来就不只是平均分谁更高，而是**哪种定义更合理地组织卒中跨被试问题**。

### 计划三 把标签重定义和半球对齐绑在一起

若只改标签，不改输入空间，功能侧对齐的收益很可能被左右脑空间错位抵消。因此建议在任务 B 上增加一个非常关键的配套实验：

- 原始通道布局
- 按病灶侧镜像翻转后的通道布局，也就是把患侧半球统一映射到同一伪侧
- 双侧输入加个体化通道选择

这一步的逻辑来自 (Mansour et al. 2022)、(Li et al. 2019) 和 (Jia et al. 2022)。既然卒中后的真正差异常体现在 ipsilesional 与 contralesional 的重组，而不是裸左脑与右脑差异，那么标签改成健患侧后，输入空间也应尽量改成**功能半球对齐**。否则“患侧 trial”在不同被试中仍会落在空间相反的位置上。

### 计划四 预设两种可能的论文结论

实验前就应接受两种都合理的结果模式。

第一种是任务 B 或任务 C 明显优于任务 A。这将支持一个较强结论：**卒中跨被试解码应优先按功能侧任务定义。**

第二种是任务 B 总体并未稳定优于任务 A，但在重度患者、SM1 病灶患者、或高异质亚组中更优。这将支持一个更成熟的结论：**健患侧重定义不是普适替代，而是针对特定卒中亚群更合理的条件性方案。**

从现有文献看，第二种结果反而更符合真实证据格局 (Qiu et al. 2017; Park et al. 2016; Mansour et al. 2022; Jia et al. 2022)。

## 最终结论

现有研究并不支持把卒中跨被试实验中的**左与右手分类**一律改写成**健侧与患侧分类**。更准确的结论是：卒中后运动想象的真实判别轴，常常已经从健康状态下的解剖学左右侧化，转向由病灶、严重度和功能重组决定的功能侧表征 (Park et al. 2016; Fallani et al. 2013; Mansour et al. 2022; Jia et al. 2022)。在这个意义上，健患侧比左右更接近卒中康复问题本身，但它仍不足以完全统一患者间差异，因为健侧并不正常，患侧也不对应单一固定网络 (Graziadio et al. 2012; Li et al. 2019)。

如果研究目标是 benchmark 可比性与文献对表，应保留左与右主任务 (Liu et al. 2024; Raza et al. 2020; Lv et al. 2025)。如果研究目标是康复导向、患手意图检测和更合理的跨被试建模，则应优先考虑功能侧任务，尤其是患侧对静息，以及在健患侧标签之上加入半球对齐、病灶分层和个体化通道选择 (Cruz et al. 2025; Mansour et al. 2022; Bundy et al. 2017; Li et al. 2019; Jia et al. 2022)。

因此，对当前项目最合乎逻辑的立场不是简单宣称“以后都用健患侧”，而是把任务定义本身做成研究贡献的一部分：**以左与右作为标准对照，以健侧与患侧作为功能侧重定义，以患侧对静息作为康复控制任务，再检验哪一种在不同卒中亚群中最合理。** 这会比直接选边站更扎实，也更容易写出一篇既有神经生理根据、又有实验设计说服力的论文。

---

## References

Bundy, David T., Lauren Souders, Kelly Baranyai, et al. 2017. “Contralesional Brain–Computer Interface Control of a Powered Exoskeleton for Motor Recovery in Chronic Stroke Survivors.” *Stroke* 48 (June): 1908–15. <https://doi.org/10.1161/STROKEAHA.116.016304>.

Bundy, David T., M. Wronkiewicz, Mohit Sharma, D. Moran, M. Corbetta, and E. Leuthardt. 2012. “Using Ipsilateral Motor Signals in the Unaffected Cerebral Hemisphere as a Signal Platform for Brain–Computer Interfaces in Hemiplegic Stroke Survivors.” *Journal of Neural Engineering* 9 (May). <https://doi.org/10.1088/1741-2560/9/3/036011>.

Cantillo-Negrete, J., R. Carino-Escobar, P. Carrillo-Mora, D. Elías-Viñas, and J. Gutiérrez-Martínez. 2018. “Motor Imagery-Based Brain-Computer Interface Coupled to a Robotic Hand Orthosis Aimed for Neurorehabilitation of Stroke Patients.” *Journal of Healthcare Engineering* 2018 (April). <https://doi.org/10.1155/2018/1624637>.

Cruz, Aniana, Marko Kuzmanoski, and Gabriel Pires. 2025. “Optimizing BCI Rehabilitation Protocols for Stroke: Exploring Task Design and Training Duration.” *2025 IEEE 8th Portuguese Meeting on Bioengineering (ENBENG)*, September 25, 57–60. <https://doi.org/10.1109/ENBENG67130.2025.11199576>.

Fallani, F. D., F. Pichiorri, G. Morone, et al. 2013. “Multiscale Topological Properties of Functional Brain Networks During Motor Imagery After Stroke.” *NeuroImage* 83 (June): 438–49. <https://doi.org/10.1016/j.neuroimage.2013.06.039>.

Frolov, A., O. Mokienko, R. Lyukmanov, et al. 2017. “Post-Stroke Rehabilitation Training with a Motor-Imagery-Based Brain-Computer Interface (BCI)-Controlled Hand Exoskeleton: A Randomized Controlled Multicenter Trial.” *Frontiers in Neuroscience* 11 (July). <https://doi.org/10.3389/fnins.2017.00400>.

Graziadio, Sara, L. Tomasevic, G. Assenza, F. Tecchio, and Janet Eyre. 2012. “The Myth of the ‘Unaffected’ Side After Unilateral Stroke: Is Reorganisation of the Non‐infarcted Corticospinal System to Re-Establish Balance the Price for Recovery?” *Experimental Neurology* 238 (December): 168–75. <https://doi.org/10.1016/j.expneurol.2012.08.031>.

He, He, and Dongrui Wu. 2019. “Different Set Domain Adaptation for Brain-Computer Interfaces: A Label Alignment Approach.” *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 28 (December): 1091–108. <https://doi.org/10.1109/TNSRE.2020.2980299>.

Jia, Tianyu, Chong Li, Linhong Mo, et al. 2022. “Tailoring Brain–Machine Interface Rehabilitation Training Based on Neural Reorganization: Towards Personalized Treatment for Stroke Patients.” *Cerebral Cortex (New York, NY)* 33 (July): 3043–52. <https://doi.org/10.1093/cercor/bhac259>.

Kancheva, I., Sandra M. A. van der Salm, N. Ramsey, and M. Vansteensel. 2023. “Association Between Lesion Location and Sensorimotor Rhythms in Stroke – a Systematic Review with Narrative Synthesis.” *Neurological Sciences* 44 (August): 4263–89. <https://doi.org/10.1007/s10072-023-06982-8>.

Lee, Seho, Hakseung Kim, Jung-Bin Kim, and Dong-Joo Kim. 2023. “Effects of Altered Functional Connectivity on Motor Imagery Brain-Computer Interfaces Based on the Laterality of Paralysis in Hemiplegia Patients.” *Computers in Biology and Medicine* 166 (September): 107435. <https://doi.org/10.1016/j.compbiomed.2023.107435>.

Li, Chong, Tianyu Jia, Quan Xu, Linhong Ji, and Yu Pan. 2019. “Brain-Computer Interface Channel-Selection Strategy Based on Analysis of Event-Related Desynchronization Topography in Stroke Patients.” *Journal of Healthcare Engineering* 2019 (August). <https://doi.org/10.1155/2019/3817124>.

Liu, Haijie, Penghu Wei, Haochong Wang, et al. 2024. “An EEG Motor Imagery Dataset for Brain Computer Interface in Acute Stroke Patients.” *Scientific Data* 11 (January). <https://doi.org/10.1038/s41597-023-02787-8>.

Lv, Shiyang, Xiangying Ran, Mengsheng Xia, et al. 2025. “Classification of Left and Right-Hand Motor Imagery in Acute Stroke Patients Using EEG Microstate.” *Journal of NeuroEngineering and Rehabilitation* 22 (June). <https://doi.org/10.1186/s12984-025-01668-y>.

Mansour, Salem, Joshua Giles, K. Ang, K. Nair, K. Phua, and M. Arvaneh. 2022. “Exploring the Ability of Stroke Survivors in Using the Contralesional Hemisphere to Control a Brain–Computer Interface.” *Scientific Reports* 12 (September). <https://doi.org/10.1038/s41598-022-20345-x>.

Nagarajan, Aarthy, Neethu Robinson, K. Ang, Karen Sui Geok Chua, E. Chew, and Cuntai Guan. 2023. “Transferring a Deep Learning Model from Healthy Subjects to Stroke Patients in a Motor Imagery Brain–Computer Interface.” *Journal of Neural Engineering* 21 (December). <https://doi.org/10.1088/1741-2552/ad152f>.

Ono, T., K. Shindo, Kimiko Kawashima, et al. 2014. “Brain-Computer Interface with Somatosensory Feedback Improves Functional Recovery from Severe Hemiplegia Due to Chronic Stroke.” *Frontiers in Neuroengineering* 7 (July). <https://doi.org/10.3389/fneng.2014.00019>.

Park, Wanjoo, G. Kwon, Y. Kim, Jong-Hwan Lee, and Laehyun Kim. 2016. “EEG Response Varies with Lesion Location in Patients with Chronic Stroke.” *Journal of NeuroEngineering and Rehabilitation* 13 (March). <https://doi.org/10.1186/s12984-016-0120-2>.

Qiu, Zhaoyang, Shugeng Chen, B. Allison, Jie Jia, Xingyu Wang, and Jing Jin. 2017. “Differences in Motor Imagery Activity Between the Paretic and Non-Paretic Hands in Stroke Patients Using an EEG BCI.” *Interacción*, July 9, 378–88. <https://doi.org/10.1007/978-3-319-58625-0_28>.

Raza, H., Anirban Chowdhury, and S. Bhattacharyya. 2020. “Deep Learning Based Prediction of EEG Motor Imagery of Stroke Patients’ for Neuro-Rehabilitation Application.” *2020 International Joint Conference on Neural Networks (IJCNN)*, March 20, 1–8. <https://doi.org/10.1109/IJCNN48605.2020.9206884>.

Spüler, M., E. López-Larraz, and A. Ramos-Murguialday. 2018. “On the Design of EEG-Based Movement Decoders for Completely Paralyzed Stroke Patients.” *Journal of NeuroEngineering and Rehabilitation* 15 (November). <https://doi.org/10.1186/s12984-018-0438-z>.

Xu, Fangzhou, Fenqi Rong, Jiancai Leng, et al. 2021. “Classification of Left-Versus Right-Hand Motor Imagery in Stroke Patients Using Supplementary Data Generated by CycleGAN.” *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 29 (October): 2417–24. <https://doi.org/10.1109/TNSRE.2021.3123969>.
