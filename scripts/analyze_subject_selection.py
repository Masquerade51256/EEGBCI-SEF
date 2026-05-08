"""
XWStroke 15人代表性子集选取分析脚本

综合既往实验性能 + 临床特征，用分层抽样策略选取15人子集。

用法:
    conda activate BCI310
    python scripts/analyze_subject_selection.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pandas as pd
import numpy as np

# ============================================================
# 1. 加载既往实验结果
# ============================================================

def load_exp_results(path, key='avg_val_acc', is_loso=False):
    """加载实验结果JSON，返回 {subject_id: metric} """
    with open(path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for sub in data['subjects']:
        sid = sub.get('subject_id') or sub.get('test_subject_id')
        if is_loso:
            # LOSO uses test_acc
            results[sid] = sub.get('test_acc', sub.get(key, 0))
        else:
            results[sid] = sub.get(key, 0)
    return results, data.get('overall_mean', 0), data.get('overall_std', 0)

exp_paths = {
    'ADFCNN_5fold': ('experiments/XWStroke_ADFCNN_20260416_151632/results/results.json', False),
    'XWFilterBankNet_5fold': ('experiments/XWStroke_XWFilterBankNet_20260409_222122/results/results.json', False),
    'EDF_Baseline_5fold': ('experiments/Ablation_EDF_Baseline_ADFCNN/results/results.json', False),
    'EDF_Sasica_5fold': ('experiments/Ablation_EDF_Sasica_ADFCNN/results/results.json', False),
    'ADFCNN_LOSO': ('experiments/xwstroke_streaming_loso/results/results.json', True),
}

exp_results = {}
for name, (path, is_loso) in exp_paths.items():
    res, mean, std = load_exp_results(path, is_loso=is_loso)
    exp_results[name] = {'data': res, 'mean': mean, 'std': std}
    print(f"Loaded {name}: mean={mean:.3f}, std={std:.3f}, n={len(res)}")

# ============================================================
# 2. 加载临床特征
# ============================================================
participants = pd.read_csv('src/datasets/21679035/participants.tsv', sep='\t')
participants['subject_id'] = participants['Participant_ID'].str.replace('sub-', '').astype(int)
participants = participants.sort_values('subject_id').reset_index(drop=True)

# 添加性能列
for name, info in exp_results.items():
    participants[f'acc_{name}'] = participants['subject_id'].map(info['data'])

# 计算综合性能指标
# 我们重点关心 LOSO 性能，因为它才是真正的跨被试指标
participants['acc_LOSO'] = participants['acc_ADFCNN_LOSO']

# 计算 within-subject 平均（4个实验的平均）
within_cols = ['acc_ADFCNN_5fold', 'acc_XWFilterBankNet_5fold', 'acc_EDF_Baseline_5fold', 'acc_EDF_Sasica_5fold']
participants['acc_within_mean'] = participants[within_cols].mean(axis=1)
participants['acc_within_std'] = participants[within_cols].std(axis=1)

# ============================================================
# 3. 临床特征分层
# ============================================================

# NIHSS 分层
participants['NIHSS_level'] = pd.cut(
    participants['NIHSS'],
    bins=[0, 3, 7, 20],
    labels=['轻度(1-3)', '中度(4-7)', '重度(>=8)'],
    include_lowest=True
)

# MBI 分层
participants['MBI_level'] = pd.cut(
    participants['MBI'],
    bins=[0, 60, 85, 100],
    labels=['重度依赖(0-60)', '中度依赖(61-85)', '轻度依赖(86-100)'],
    include_lowest=True
)

# mRS 分层
participants['mRS_level'] = pd.cut(
    participants['mRS'],
    bins=[-1, 1, 3, 6],
    labels=['良好(0-1)', '中度残疾(2-3)', '重度残疾(4-5)'],
    include_lowest=True
)

# Duration 分层
participants['Duration_level'] = pd.cut(
    participants['Duration'],
    bins=[0, 3, 10, 50],
    labels=['急性/早期(1-3)', '亚急性(4-10)', '慢性(>10)'],
    include_lowest=True
)

# Age 分层
participants['Age_level'] = pd.cut(
    participants['Age'],
    bins=[0, 50, 60, 80],
    labels=['年轻(<50)', '中年(50-60)', '老年(>60)'],
    include_lowest=True
)

# LOSO 性能分层（三分位）
participants['LOSO_perf'] = pd.qcut(
    participants['acc_LOSO'],
    q=3,
    labels=['低', '中', '高']
)

# Within-subject 性能分层
participants['Within_perf'] = pd.qcut(
    participants['acc_within_mean'],
    q=3,
    labels=['低', '中', '高']
)

# ============================================================
# 4. 输出总体统计
# ============================================================
print("\n" + "="*70)
print("总体被试分布统计 (N=50)")
print("="*70)

print(f"\n偏瘫侧: {participants['ParalysisSide'].value_counts().to_dict()}")
print(f"利手: {participants['Handedness'].value_counts().to_dict()}")
print(f"性别: {participants['Gender'].value_counts().to_dict()}")
print(f"NIHSS: {participants['NIHSS_level'].value_counts().to_dict()}")
print(f"MBI: {participants['MBI_level'].value_counts().to_dict()}")
print(f"mRS: {participants['mRS_level'].value_counts().to_dict()}")
print(f"病程: {participants['Duration_level'].value_counts().to_dict()}")
print(f"年龄: {participants['Age_level'].value_counts().to_dict()}")
print(f"LOSO性能: {participants['LOSO_perf'].value_counts().to_dict()}")
print(f"Within-subject性能: {participants['Within_perf'].value_counts().to_dict()}")

# ============================================================
# 5. 选取15人代表性子集（分层抽样策略）
# ============================================================
print("\n" + "="*70)
print("15人代表性子集选取策略")
print("="*70)

# 分层策略:
# - 首要维度: LOSO性能 (高/中/低各占约1/3)
# - 次要考虑: 偏瘫侧 (左/右比例与总体一致, 约 56%/44%)
# - 补充维度: NIHSS严重程度、mRS、病程、性别

# 目标配额:
# LOSO性能: 高5人, 中5人, 低5人
# 偏瘫侧: 左偏瘫约8-9人, 右偏瘫约6-7人

selected_ids = set()

def select_from_stratum(df, n_needed, exclude_ids, prioritize_diversity=True):
    """从给定dataframe中选取n_needed人，尽量保证多样性"""
    df = df[~df['subject_id'].isin(exclude_ids)].copy()
    if len(df) <= n_needed:
        return df['subject_id'].tolist()
    
    # 优先选取在多个维度上具有代表性的
    # 简单策略: 随机但确保不连续选同偏瘫侧/同NIHSS等
    if prioritize_diversity:
        # 按 NIHSS_level, ParalysisSide, Gender 分层后每stratum最多取1人
        df['stratum'] = df['NIHSS_level'].astype(str) + '_' + df['ParalysisSide'] + '_' + df['Gender']
        selected = []
        stratum_counts = {}
        
        # 先shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for _, row in df.iterrows():
            if len(selected) >= n_needed:
                break
            s = row['stratum']
            if stratum_counts.get(s, 0) < 2:  # 每个stratum最多2人
                selected.append(row['subject_id'])
                stratum_counts[s] = stratum_counts.get(s, 0) + 1
        
        # 如果不够，从剩余中补充
        if len(selected) < n_needed:
            remaining = df[~df['subject_id'].isin(selected)]
            selected.extend(remaining['subject_id'].iloc[:n_needed - len(selected)].tolist())
        
        return selected
    else:
        return df['subject_id'].iloc[:n_needed].tolist()

# 按LOSO性能分层选取
target_per_perf = 5

for perf_level in ['高', '中', '低']:
    df_perf = participants[participants['LOSO_perf'] == perf_level]
    n_left = df_perf[df_perf['ParalysisSide'] == 'left'].shape[0]
    n_right = df_perf[df_perf['ParalysisSide'] == 'right'].shape[0]
    print(f"\nLOSO性能[{perf_level}]层: 共{len(df_perf)}人 (左偏瘫{n_left}, 右偏瘫{n_right})")
    
    # 目标: 该层选5人，左/右比例尽量接近总体56/44
    target_left = max(1, round(target_per_perf * 0.56))
    target_right = target_per_perf - target_left
    
    # 从左偏瘫中选
    df_left = df_perf[df_perf['ParalysisSide'] == 'left']
    selected_left = select_from_stratum(df_left, target_left, selected_ids)
    selected_ids.update(selected_left)
    
    # 从右偏瘫中选
    df_right = df_perf[df_perf['ParalysisSide'] == 'right']
    selected_right = select_from_stratum(df_right, target_right, selected_ids)
    selected_ids.update(selected_right)
    
    # 如果不够5人，从该层剩余中补充
    current_in_layer = [sid for sid in selected_ids if sid in df_perf['subject_id'].values]
    if len(current_in_layer) < target_per_perf:
        remaining = df_perf[~df_perf['subject_id'].isin(selected_ids)]
        extra = select_from_stratum(remaining, target_per_perf - len(current_in_layer), selected_ids, prioritize_diversity=False)
        selected_ids.update(extra)
    
    current_in_layer = sorted([sid for sid in selected_ids if sid in df_perf['subject_id'].values])
    print(f"  选取: {current_in_layer} (左{sum(1 for s in current_in_layer if participants[participants.subject_id==s].ParalysisSide.values[0]=='left')}, 右{sum(1 for s in current_in_layer if participants[participants.subject_id==s].ParalysisSide.values[0]=='right')})")

selected_ids = sorted(selected_ids)
print(f"\n初步选取的15人: {selected_ids}")

# 如果超过15人，裁剪；如果不足15人，补充
if len(selected_ids) > 15:
    # 从"高"性能层开始裁剪，保留最具多样性的
    selected_ids = selected_ids[:15]
    print(f"裁剪后15人: {selected_ids}")
elif len(selected_ids) < 15:
    # 从总体中补充，优先选取在现有子集中缺失的特征组合
    remaining = participants[~participants['subject_id'].isin(selected_ids)]
    extra = select_from_stratum(remaining, 15 - len(selected_ids), selected_ids)
    selected_ids.extend(extra)
    selected_ids = sorted(selected_ids)
    print(f"补充后15人: {selected_ids}")

# ============================================================
# 6. 验证子集代表性
# ============================================================
print("\n" + "="*70)
print("子集代表性验证 (N=15 vs N=50)")
print("="*70)

subset = participants[participants['subject_id'].isin(selected_ids)].copy()
full = participants

def compare_dist(col, full_df, sub_df):
    full_vc = full_df[col].value_counts(normalize=True).sort_index() * 100
    sub_vc = sub_df[col].value_counts(normalize=True).sort_index() * 100
    print(f"\n{col}:")
    for idx in full_vc.index:
        f = full_vc.get(idx, 0)
        s = sub_vc.get(idx, 0)
        marker = "OK" if abs(f - s) <= 15 else "!!"
        print(f"  {marker} {idx}: 总体={f:.1f}%, 子集={s:.1f}%")

compare_dist('ParalysisSide', full, subset)
compare_dist('Handedness', full, subset)
compare_dist('Gender', full, subset)
compare_dist('NIHSS_level', full, subset)
compare_dist('mRS_level', full, subset)
compare_dist('Duration_level', full, subset)
compare_dist('Age_level', full, subset)
compare_dist('LOSO_perf', full, subset)
compare_dist('Within_perf', full, subset)

# ============================================================
# 7. 输出子集详细信息
# ============================================================
print("\n" + "="*70)
print("最终15人子集详细信息")
print("="*70)

display_cols = ['subject_id', 'Gender', 'Age', 'Duration', 'ParalysisSide', 'Handedness',
                'NIHSS', 'MBI', 'mRS', 'StrokeLocation',
                'acc_LOSO', 'acc_within_mean',
                'acc_ADFCNN_5fold', 'acc_XWFilterBankNet_5fold']

subset_display = subset[display_cols].copy()
subset_display['acc_LOSO'] = (subset_display['acc_LOSO'] * 100).round(1)
subset_display['acc_within_mean'] = (subset_display['acc_within_mean'] * 100).round(1)
subset_display['acc_ADFCNN_5fold'] = (subset_display['acc_ADFCNN_5fold'] * 100).round(1)
subset_display['acc_XWFilterBankNet_5fold'] = (subset_display['acc_XWFilterBankNet_5fold'] * 100).round(1)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(subset_display.to_string(index=False))

# ============================================================
# 8. 保存结果
# ============================================================
output = {
    'selected_subjects': selected_ids,
    'selection_strategy': 'stratified by LOSO performance + ParalysisSide + NIHSS',
    'subset_stats': {
        'n': len(selected_ids),
        'ParalysisSide': subset['ParalysisSide'].value_counts().to_dict(),
        'NIHSS_mean': subset['NIHSS'].mean(),
        'NIHSS_std': subset['NIHSS'].std(),
        'MBI_mean': subset['MBI'].mean(),
        'MBI_std': subset['MBI'].std(),
        'mRS_mean': subset['mRS'].mean(),
        'mRS_std': subset['mRS'].std(),
        'acc_LOSO_mean': subset['acc_LOSO'].mean(),
        'acc_LOSO_std': subset['acc_LOSO'].std(),
        'acc_within_mean': subset['acc_within_mean'].mean(),
        'acc_within_std': subset['acc_within_mean'].std(),
    },
    'full_stats': {
        'n': len(full),
        'ParalysisSide': full['ParalysisSide'].value_counts().to_dict(),
        'NIHSS_mean': full['NIHSS'].mean(),
        'NIHSS_std': full['NIHSS'].std(),
        'MBI_mean': full['MBI'].mean(),
        'MBI_std': full['MBI'].std(),
        'mRS_mean': full['mRS'].mean(),
        'mRS_std': full['mRS'].std(),
        'acc_LOSO_mean': full['acc_LOSO'].mean(),
        'acc_LOSO_std': full['acc_LOSO'].std(),
        'acc_within_mean': full['acc_within_mean'].mean(),
        'acc_within_std': full['acc_within_mean'].std(),
    }
}

os.makedirs('results/comparisons', exist_ok=True)
with open('results/comparisons/subject_subset_15_selection.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# 保存CSV
display_cols_save = ['subject_id', 'Gender', 'Age', 'Duration', 'ParalysisSide', 'Handedness',
                     'NIHSS', 'MBI', 'mRS', 'StrokeLocation',
                     'acc_LOSO', 'acc_within_mean', 'acc_ADFCNN_5fold', 'acc_XWFilterBankNet_5fold']
subset[display_cols_save].to_csv('results/comparisons/subject_subset_15.csv', index=False, encoding='utf-8-sig')

print(f"\n结果已保存到:")
print(f"  - results/comparisons/subject_subset_15_selection.json")
print(f"  - results/comparisons/subject_subset_15.csv")
print(f"\n最终子集被试ID: {selected_ids}")
