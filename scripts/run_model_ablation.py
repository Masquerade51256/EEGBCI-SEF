"""
模型归因对照实验一键运行脚本

运行4组实验：
  1. ADFCNN + BCICIV2a + Streaming LOSO  (健康人基准)
  2. EEGNet  + XWStroke + LOSO (LR)      (换模型，原标签)
  3. EEGNet  + XWStroke + LOSO (Affected+Aligned) (换模型，功能标签+对齐)

实验结束后自动汇总对比，回答核心问题：
  - 是模型不行？还是卒中数据本身太难？

用法:
    source /home/opt/anaconda/etc/profile.d/conda.sh
    conda activate torch
    python scripts/run_model_ablation.py

后台运行:
    nohup python scripts/run_model_ablation.py > model_ablation.log 2>&1 &
"""

import sys, os, json, subprocess, time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_exp(config_path, exp_name):
    """Run experiment with fixed name."""
    print(f"\n{'='*70}")
    print(f"Running: {config_path}")
    print(f"Experiment name: {exp_name}")
    print(f"{'='*70}")

    start_time = time.time()
    cmd = [sys.executable, "train.py", "--config", config_path, "--name", exp_name]
    result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] Failed: {config_path} (elapsed: {timedelta(seconds=int(elapsed))})")
        return None

    res_file = Path("experiments") / exp_name / "results" / "results.json"
    if not res_file.exists():
        print(f"[ERROR] Results file not found: {res_file}")
        return None

    with open(res_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[OK] Completed in {timedelta(seconds=int(elapsed))}")
    print(f"     Overall: {data['overall_mean']*100:.2f}% ± {data['overall_std']*100:.2f}%")
    return data


def summarize(results):
    """Print attribution summary."""
    print(f"\n{'='*70}")
    print("MODEL ABLATION / ATTRIBUTION SUMMARY")
    print(f"{'='*70}")

    # Expected results
    expected = {
        'BCICIV2a_ADFCNN_StreamingLOSO': 'ADFCNN + BCICIV2a (Healthy, 4-class)',
        'XWStroke_Full50_LR': 'ADFCNN + XWStroke LR (Stroke, 2-class)',
        'XWStroke_Full50_EEGNet_LOSO_LR': 'EEGNet + XWStroke LR (Stroke, 2-class)',
        'XWStroke_Full50_EEGNet_LOSO_Affected_Aligned': 'EEGNet + XWStroke Aff+Align (Stroke, 2-class)',
        'XWStroke_Full50_Affected': 'ADFCNN + XWStroke Aff (No Align, legacy)',
        'XWStroke_Full50_LOSO_Affected_Aligned': 'ADFCNN + XWStroke Aff+Align (Stroke, 2-class)',
    }

    rows = []
    for key, label in expected.items():
        if key in results and results[key] is not None:
            r = results[key]
            rows.append({
                'Experiment': label,
                'Mean': f"{r['overall_mean']*100:.2f}%",
                'Std': f"{r['overall_std']*100:.2f}%",
                'N': r.get('streaming_config', {}).get('num_subjects', '-'),
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # Attribution logic
    print(f"\n{'='*70}")
    print("ATTRIBUTION INTERPRETATION")
    print(f"{'='*70}")

    bcic_adf = results.get('BCICIV2a_ADFCNN_StreamingLOSO')
    bcic_eeg = results.get('BCICIV2a_EEGNet_StreamingLOSO')
    xw_adf_lr = results.get('XWStroke_Full50_LR')
    xw_eeg_lr = results.get('XWStroke_Full50_EEGNet_LOSO_LR')
    xw_eeg_aff = results.get('XWStroke_Full50_EEGNet_LOSO_Affected_Aligned')

    # Framework sanity check
    if bcic_eeg:
        bcic_eeg_mean = bcic_eeg['overall_mean'] * 100
        print(f"\n0. FRAMEWORK SANITY CHECK: EEGNet on BCICIV2a (healthy): {bcic_eeg_mean:.1f}%")
        if bcic_eeg_mean >= 55:
            print("   ✅ Framework (StreamingLOSOTrainer + preprocessing) is CAPABLE.")
            print("   → The pipeline itself can achieve reasonable cross-subject performance.")
        else:
            print("   ⚠️  Framework SUSPECT: Even EEGNet on healthy data is poor.")
            print("   → Training hyperparameters, data loading, or preprocessing may be broken.")
            print("   → ALL prior results should be re-evaluated after fixing the pipeline.")

    if bcic_adf:
        bcic_adf_mean = bcic_adf['overall_mean'] * 100
        print(f"\n1. ADFCNN on BCICIV2a (healthy): {bcic_adf_mean:.1f}%")
        if bcic_adf_mean >= 60:
            print("   → ADFCNN itself CAN do cross-subject generalization.")
            print("   → XWStroke's ~50% is NOT primarily a model problem.")
        else:
            print("   → ADFCNN performs poorly even on healthy data.")
            print("   → Model architecture may be a contributing factor.")

    if xw_eeg_lr and xw_adf_lr:
        eeg_mean = xw_eeg_lr['overall_mean'] * 100
        adf_mean = xw_adf_lr['overall_mean'] * 100
        print(f"\n2. EEGNet vs ADFCNN on XWStroke (LR): {eeg_mean:.1f}% vs {adf_mean:.1f}%")
        if abs(eeg_mean - adf_mean) < 3:
            print("   → Two very different models perform similarly.")
            print("   → Strong evidence that the DATA (heterogeneity) is the bottleneck.")
        elif eeg_mean > adf_mean + 5:
            print("   → EEGNet significantly outperforms ADFCNN.")
            print("   → Model choice DOES matter for this dataset.")
        else:
            print("   → Small difference; both models struggle.")

    if xw_eeg_lr and xw_eeg_aff:
        eeg_lr = xw_eeg_lr['overall_mean'] * 100
        eeg_aff = xw_eeg_aff['overall_mean'] * 100
        print(f"\n3. EEGNet LR vs Affected+Aligned on XWStroke: {eeg_lr:.1f}% vs {eeg_aff:.1f}%")
        if eeg_aff > eeg_lr + 1:
            print("   → Functional-side labels + alignment help even with EEGNet.")
            print("   → The combined effect is model-agnostic.")
        else:
            print("   → No clear benefit from functional labels with EEGNet.")
            print("   → May need stronger architectural inductive bias.")

    print(f"\n{'='*70}")

    # Save JSON
    out_dir = Path("results/model_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'experiments': {k: {
            'mean': v['overall_mean'] * 100,
            'std': v['overall_std'] * 100,
        } for k, v in results.items() if v is not None},
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / 'ablation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to: {out_dir / 'ablation_summary.json'}")


def main():
    print("\n" + "="*70)
    print("Model Ablation / Attribution Experiments")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nExperiments to run:")
    print("  1. ADFCNN + BCICIV2a Streaming LOSO   (~30-60 min)")
    print("  2. EEGNet  + BCICIV2a Streaming LOSO  (~30-60 min)  [Framework sanity check]")
    print("  3. EEGNet  + XWStroke LOSO (LR)       (~4-6 hours)")
    print("  4. EEGNet  + XWStroke LOSO (Affected+Aligned) (~4-6 hours)")
    print("\nPress Ctrl+C within 5 seconds to cancel...")

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 0

    total_start = time.time()
    results = {}

    # Experiment 1: ADFCNN on BCICIV2a (healthy baseline)
    results['BCICIV2a_ADFCNN_StreamingLOSO'] = run_exp(
        "configs/experiment/bciciv2a_adfcnn_streaming_loso.yaml",
        "BCICIV2a_ADFCNN_StreamingLOSO"
    )

    # Experiment 2: EEGNet on BCICIV2a (framework sanity check)
    results['BCICIV2a_EEGNet_StreamingLOSO'] = run_exp(
        "configs/experiment/bciciv2a_eegnet_streaming_loso.yaml",
        "BCICIV2a_EEGNet_StreamingLOSO"
    )

    # Experiment 3: EEGNet on XWStroke LR
    results['XWStroke_Full50_EEGNet_LOSO_LR'] = run_exp(
        "configs/experiment/xwstroke_eegnet_full_loso_lr.yaml",
        "XWStroke_Full50_EEGNet_LOSO_LR"
    )

    # Experiment 4: EEGNet on XWStroke Affected + Aligned
    results['XWStroke_Full50_EEGNet_LOSO_Affected_Aligned'] = run_exp(
        "configs/experiment/xwstroke_eegnet_full_loso_affected.yaml",
        "XWStroke_Full50_EEGNet_LOSO_Affected_Aligned"
    )

    # Load existing results for reference
    existing_paths = {
        'XWStroke_Full50_LR': 'experiments/XWStroke_Full50_LR/results/results.json',
        'XWStroke_Full50_Affected': 'experiments/XWStroke_Full50_Affected/results/results.json',
        'XWStroke_Full50_LOSO_Affected_Aligned': 'experiments/XWStroke_Full50_LOSO_Affected_Aligned/results/results.json',
    }
    for key, path in existing_paths.items():
        p = Path(path)
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                results[key] = json.load(f)

    total_elapsed = time.time() - total_start
    print(f"\nTotal elapsed time: {timedelta(seconds=int(total_elapsed))}")

    summarize(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
