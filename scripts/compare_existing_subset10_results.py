"""
对比已完成的10人子集LOSO实验结果
（用于一键运行脚本失败后手动对比）
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

# 根据时间顺序推断：先运行的是任务A(lr)，后运行的是任务B(affected)
res_lr = json.load(open('experiments/XWStroke_ADFCNN_20260508_044352/results/results.json'))
res_aff = json.load(open('experiments/XWStroke_ADFCNN_20260508_055459/results/results.json'))

print("\n" + "="*65)
print("XWStroke 10人子集 LOSO 对照实验结果对比")
print("="*65)

rows = []
for r_lr, r_aff in zip(res_lr['subjects'], res_aff['subjects']):
    sid = r_lr['test_subject_id']
    rows.append({
        'Subject': f'sub-{sid:02d}',
        'TaskA_LR': f"{r_lr['test_acc']*100:.1f}%",
        'TaskB_Aff': f"{r_aff['test_acc']*100:.1f}%",
        'Diff': f"{(r_aff['test_acc']-r_lr['test_acc'])*100:+.1f}%",
        'Best_Epoch_A': r_lr['best_epoch'],
        'Best_Epoch_B': r_aff['best_epoch'],
    })

df = pd.DataFrame(rows)
print("\nPer-Subject Accuracy:")
print(df.to_string(index=False))

m_lr = res_lr['overall_mean']*100
s_lr = res_lr['overall_std']*100
m_aff = res_aff['overall_mean']*100
s_aff = res_aff['overall_std']*100

print(f"\n{'='*65}")
print("Overall Summary:")
print(f"  Task A (Left/Right):      {m_lr:.2f}% ± {s_lr:.2f}%")
print(f"  Task B (Affected/Healthy): {m_aff:.2f}% ± {s_aff:.2f}%")
print(f"  Absolute Difference:      {m_aff - m_lr:+.2f}%")
print(f"  Relative Improvement:     {(m_aff - m_lr)/m_lr*100:+.1f}%")
print(f"{'='*65}")

# Count improvements
improved = sum(1 for r in rows if float(r['Diff'].replace('%','').replace('+','')) > 0)
print(f"\n  {improved}/10 subjects improved with affected/unaffected labels")
print(f"  Variance reduced: {s_lr:.2f}% → {s_aff:.2f}%")

# Save
os.makedirs('results/comparisons', exist_ok=True)
df.to_csv('results/comparisons/subset10_loso_comparison.csv', index=False, encoding='utf-8-sig')

summary = {
    'task_a_lr': {'mean': m_lr, 'std': s_lr},
    'task_b_affected': {'mean': m_aff, 'std': s_aff},
    'difference': m_aff - m_lr,
    'relative_improvement_pct': (m_aff - m_lr)/m_lr*100,
    'improved_subjects': improved,
    'total_subjects': 10,
}
with open('results/comparisons/subset10_loso_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to:")
print(f"  - results/comparisons/subset10_loso_comparison.csv")
print(f"  - results/comparisons/subset10_loso_summary.json")
