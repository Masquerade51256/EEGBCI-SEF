"""
一键运行 XWStroke 15人子集 LOSO 对照实验

运行两套LOSO实验:
  任务A (左右手): configs/experiment/xwstroke_subset15_loso_lr.yaml
  任务B (患健侧): configs/experiment/xwstroke_subset15_loso_affected.yaml

用法:
    conda activate BCI310
    python scripts/run_subset15_loso_comparison.py
"""

import sys, os, json, subprocess, pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_exp(config_path, exp_name):
    """Run experiment with fixed name to ensure deterministic output directory."""
    print(f"\n{'='*60}\nRunning: {config_path} (name={exp_name})\n{'='*60}")
    cmd = [sys.executable, "train.py", "--config", config_path, "--name", exp_name]
    result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))
    if result.returncode != 0:
        print(f"[ERROR] Failed: {config_path}")
        return None
    res_file = Path("experiments") / exp_name / "results" / "results.json"
    if not res_file.exists():
        print(f"[ERROR] Results file not found: {res_file}")
        return None
    with open(res_file) as f:
        return json.load(f)

def compare(results_lr, results_aff):
    print(f"\n{'='*60}\nLOSO Comparison (15 subjects)\n{'='*60}")
    rows = []
    for r_lr, r_aff in zip(results_lr['subjects'], results_aff['subjects']):
        sid = r_lr['test_subject_id']
        rows.append({
            'Subject': f'sub-{sid:02d}',
            'TaskA_LR': f"{r_lr['test_acc']*100:.1f}%",
            'TaskB_Aff': f"{r_aff['test_acc']*100:.1f}%",
            'Diff': f"{(r_aff['test_acc']-r_lr['test_acc'])*100:+.1f}%",
        })
    df = pd.DataFrame(rows)
    print("\nPer-Subject:")
    print(df.to_string(index=False))
    m_lr = results_lr['overall_mean']*100
    s_lr = results_lr['overall_std']*100
    m_aff = results_aff['overall_mean']*100
    s_aff = results_aff['overall_std']*100
    print(f"\nOverall:")
    print(f"  Task A (Left/Right):      {m_lr:.2f}% +/- {s_lr:.2f}%")
    print(f"  Task B (Affected/Health): {m_aff:.2f}% +/- {s_aff:.2f}%")
    print(f"  Difference (B - A):       {m_aff - m_lr:+.2f}%")
    print(f"{'='*60}")
    os.makedirs('results/comparisons', exist_ok=True)
    df.to_csv('results/comparisons/subset15_loso_comparison.csv', index=False, encoding='utf-8-sig')
    with open('results/comparisons/subset15_loso_summary.json','w') as f:
        json.dump({'task_a':{'mean':m_lr,'std':s_lr},'task_b':{'mean':m_aff,'std':s_aff},'diff':m_aff-m_lr}, f, indent=2)
    print("Saved to results/comparisons/subset15_loso_*.csv/json")

def main():
    print("\n" + "="*60)
    print("XWStroke 15-subject LOSO Validation")
    print("="*60)
    res_lr = run_exp("configs/experiment/xwstroke_subset15_loso_lr.yaml", "XWStroke_Subset15_LR")
    res_aff = run_exp("configs/experiment/xwstroke_subset15_loso_affected.yaml", "XWStroke_Subset15_Affected")
    if res_lr and res_aff:
        compare(res_lr, res_aff)
    else:
        print("[ERROR] Some experiment failed.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
