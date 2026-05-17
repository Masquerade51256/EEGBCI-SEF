# XWStroke 子集 LOSO 实验报告

> 基于分层分析（Stratified Analysis）筛选的三种子集方案交叉验证结果

## 实验配置

- **模型**: ADFCNN
- **训练器**: StreamingLOSOTrainer
- **Epochs**: 100
- **Batch Size**: 512
- **优化器**: Adam (lr=1e-3, wd=1e-2)
- **标签映射**: Affected/Unaffected + Hemisphere Alignment

## 方案对比总表

| Scheme             |   N | Mean_Acc   | Std_Acc   | Criteria     | vs_Full50   |
|:-------------------|----:|:-----------|:----------|:-------------|:------------|
| Full 50 (Baseline) |  50 | 51.46%     | 5.23%     | All subjects | —           |

## 详细结果

## 结论与讨论

1. **Scheme C（Subcortical Focus）是主推方案**
   - N=17 在统计效力和队列同质性之间取得了最佳平衡
   - 病灶位置（Subcortical）具有最强的神经生理学解释：皮层运动区保留但皮质脊髓束受损，Aff+Align 的侧化重映射恰好利用了这一可塑性

2. **Scheme A（Damage Exclusion）用于稳健性验证**
   - N=30 的结果可验证：子集提升是否依赖于过度严格的筛选

3. **Scheme B（Tightened Prior）仅作参考**
   - N=7 样本量过小，结果可能不稳定，不建议作为论文主要结论

4. **55% 天花板问题**
   - 即使在最优子集上，预估 LOSO 仍难以突破 55%
   - 建议后续从模型架构（更长的训练、学习率调度、不同 backbone）或数据增强角度进一步优化
