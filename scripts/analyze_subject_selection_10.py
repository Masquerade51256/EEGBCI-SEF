"""
XWStroke 10人代表性子集选取分析脚本（快速验证版）

用法:
    conda activate BCI310
    python scripts/analyze_subject_selection_10.py
"""

import sys, os, json, pandas as pd, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load experiment results
def load_exp(path, is_loso=False):
    with open(path) as f:
        data = json.load(f)
    res = {}
    for sub in data['subjects']:
        sid = sub.get('subject_id') or sub.get('test_subject_id')
        res[sid] = sub.get('test_acc' if is_loso else 'avg_val_acc', 0)
    return res, data.get('overall_mean', 0), data.get('overall_std', 0)

exp_paths = {
    'ADFCNN_5fold': ('experiments/XWStroke_ADFCNN_20260416_151632/results/results.json', False),
    'XWFilterBankNet_5fold': ('experiments/XWStroke_XWFilterBankNet_20260409_222122/results/results.json', False),
    'EDF_Baseline_5fold': ('experiments/Ablation_EDF_Baseline_ADFCNN/results/results.json', False),
    'EDF_Sasica_5fold': ('experiments/Ablation_EDF_Sasica_ADFCNN/results/results.json', False),
    'ADFCNN_LOSO': ('experiments/xwstroke_streaming_loso/results/results.json', True),
}
exp_results = {}
for name, (p, is_l) in exp_paths.items():
    r, m, s = load_exp(p, is_l)
    exp_results[name] = {'data': r, 'mean': m, 'std': s}
    print(f"Loaded {name}: mean={m:.3f}, std={s:.3f}")

# Load participants
participants = pd.read_csv('src/datasets/21679035/participants.tsv', sep='\t')
participants['subject_id'] = participants['Participant_ID'].str.replace('sub-', '').astype(int)
participants = participants.sort_values('subject_id').reset_index(drop=True)

for name, info in exp_results.items():
    participants[f'acc_{name}'] = participants['subject_id'].map(info['data'])

within_cols = ['acc_ADFCNN_5fold', 'acc_XWFilterBankNet_5fold', 'acc_EDF_Baseline_5fold', 'acc_EDF_Sasica_5fold']
participants['acc_within_mean'] = participants[within_cols].mean(axis=1)
participants['acc_LOSO'] = participants['acc_ADFCNN_LOSO']

# Stratification
participants['NIHSS_level'] = pd.cut(participants['NIHSS'], bins=[0,3,7,20], labels=['light','medium','severe'], include_lowest=True)
participants['LOSO_perf'] = pd.qcut(participants['acc_LOSO'], q=3, labels=['low','med','high'])

print("\n" + "="*60)
print("Full cohort stats (N=50)")
print("="*60)
print(f"ParalysisSide: {participants['ParalysisSide'].value_counts().to_dict()}")
print(f"LOSO_perf: {participants['LOSO_perf'].value_counts().to_dict()}")
print(f"NIHSS_level: {participants['NIHSS_level'].value_counts().to_dict()}")
print(f"Gender: {participants['Gender'].value_counts().to_dict()}")

# Select 10 subjects: stratified by LOSO perf (3/4/3) + ParalysisSide
# Target: high=3, med=4, low=3 (or 4/3/3)
selected_ids = set()

def pick(df, n_needed, exclude, seed_offset=0):
    df = df[~df['subject_id'].isin(exclude)].copy()
    if len(df) <= n_needed:
        return df['subject_id'].tolist()
    # Prioritize diversity across ParalysisSide, NIHSS_level, Gender
    df['stratum'] = df['ParalysisSide'].astype(str) + '_' + df['NIHSS_level'].astype(str)
    counts = {}
    picked = []
    df = df.sample(frac=1, random_state=42+seed_offset).reset_index(drop=True)
    for _, row in df.iterrows():
        if len(picked) >= n_needed:
            break
        s = row['stratum']
        if counts.get(s, 0) < 2:
            picked.append(row['subject_id'])
            counts[s] = counts.get(s, 0) + 1
    if len(picked) < n_needed:
        rem = df[~df['subject_id'].isin(picked)]
        picked.extend(rem['subject_id'].iloc[:n_needed-len(picked)].tolist())
    return picked

# Target: 4 high + 3 med + 3 low = 10
# With ~56% left paralysis overall
targets = {'high': 4, 'med': 3, 'low': 3}
for perf, n in targets.items():
    df_perf = participants[participants['LOSO_perf'] == perf]
    n_left = max(1, round(n * 0.56))
    n_right = n - n_left
    sel_left = pick(df_perf[df_perf['ParalysisSide']=='left'], n_left, selected_ids, seed_offset=ord(perf[0]))
    selected_ids.update(sel_left)
    sel_right = pick(df_perf[df_perf['ParalysisSide']=='right'], n_right, selected_ids, seed_offset=ord(perf[0])+10)
    selected_ids.update(sel_right)
    cur = sorted([s for s in selected_ids if s in df_perf['subject_id'].values])
    print(f"LOSO [{perf}] layer: picked {cur}")

selected_ids = sorted(selected_ids)
if len(selected_ids) > 10:
    # Trim from high layer first (usually more candidates)
    selected_ids = selected_ids[:10]
elif len(selected_ids) < 10:
    rem = participants[~participants['subject_id'].isin(selected_ids)]
    extra = pick(rem, 10 - len(selected_ids), selected_ids, seed_offset=99)
    selected_ids.extend(extra)
    selected_ids = sorted(selected_ids)

print(f"\nSelected 10 subjects: {selected_ids}")

# Validation
subset = participants[participants['subject_id'].isin(selected_ids)].copy()
full = participants

print("\n" + "="*60)
print("Validation (N=10 vs N=50)")
print("="*60)

def compare(col):
    f = full[col].value_counts(normalize=True).sort_index()*100
    s = subset[col].value_counts(normalize=True).sort_index()*100
    print(f"\n{col}:")
    for idx in f.index:
        print(f"  {idx}: full={f.get(idx,0):.1f}% subset={s.get(idx,0):.1f}%")

compare('ParalysisSide')
compare('Gender')
compare('NIHSS_level')
compare('LOSO_perf')

print(f"\nLOSO mean: full={full['acc_LOSO'].mean()*100:.1f}% subset={subset['acc_LOSO'].mean()*100:.1f}%")
print(f"Within mean: full={full['acc_within_mean'].mean()*100:.1f}% subset={subset['acc_within_mean'].mean()*100:.1f}%")

# Display details
disp = subset[['subject_id','Gender','Age','Duration','ParalysisSide','Handedness','NIHSS','MBI','mRS','StrokeLocation','acc_LOSO','acc_within_mean','acc_ADFCNN_5fold','acc_XWFilterBankNet_5fold']].copy()
for c in ['acc_LOSO','acc_within_mean','acc_ADFCNN_5fold','acc_XWFilterBankNet_5fold']:
    disp[c] = (disp[c]*100).round(1)
print("\n" + "="*60)
print("Subject details:")
print("="*60)
print(disp.to_string(index=False))

# Save
os.makedirs('results/comparisons', exist_ok=True)
output = {
    'selected_subjects': selected_ids,
    'subset_stats': {
        'n': 10,
        'ParalysisSide': subset['ParalysisSide'].value_counts().to_dict(),
        'acc_LOSO_mean': subset['acc_LOSO'].mean(),
        'acc_LOSO_std': subset['acc_LOSO'].std(),
        'acc_within_mean': subset['acc_within_mean'].mean(),
        'acc_within_std': subset['acc_within_mean'].std(),
    },
    'full_stats': {
        'n': 50,
        'ParalysisSide': full['ParalysisSide'].value_counts().to_dict(),
        'acc_LOSO_mean': full['acc_LOSO'].mean(),
        'acc_LOSO_std': full['acc_LOSO'].std(),
        'acc_within_mean': full['acc_within_mean'].mean(),
        'acc_within_std': full['acc_within_mean'].std(),
    }
}
with open('results/comparisons/subject_subset_10_selection.json','w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
disp.to_csv('results/comparisons/subject_subset_10.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved to results/comparisons/subject_subset_10.csv and .json")
print(f"Final 10-subject IDs: {selected_ids}")
