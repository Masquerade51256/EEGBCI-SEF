#!/usr/bin/env python3
"""
DANN Full50 Conservative Experiment

Uses conservative hyperparameters based on Clean-C grid search lessons:
- domain_loss_weight: 0.1 (low, to avoid interfering with task learning)
- grl_gamma: 3 (gentler scheduling than 5/10)
- domain_grouping: stroke_location_2class (Subcortical vs Non-subcortical)
- mixed_precision: false (stability)
- num_workers: 0 (avoid multiprocessing issues)
- epochs: 50 (based on best_epoch median analysis)

This serves as a definitive test of whether domain-invariant features
are viable on the full heterogeneous cohort.

Usage:
    python scripts/run_dann_full50_conservative.py

Expected runtime: 4-6 hours (50 subjects x 50 epochs)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results" / "dann_full50"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baselines for comparison
BASELINE_AFF = 0.5146  # Full50 Aff+Align
BASELINE_LR = 0.4963   # Full50 LR


def build_config() -> dict:
    """Build conservative Full50 DANN config."""
    return {
        "experiment": {
            "name": "DANN_Full50_Conservative",
            "seed": 42,
            "description": "Full 50-subject DANN with conservative parameters: dlw=0.1, gamma=3, 2class domain",
        },
        "data": {
            "dataset": "XWStroke",
            "subjects": list(range(1, 51)),
            "info_path": "configs/dataset/XWStroke_affected_aligned.yaml",
        },
        "model": {
            "type": "DANNADFCNN",
            "args": {
                "num_domains": 2,  # stroke_location_2class: Subcortical vs Non-subcortical
                "domain_hidden_dims": [128, 64],  # Smaller than [256,128] for stability
            },
        },
        "training": {
            "device": "cuda",
            "epochs": 50,
            "batch_size": 512,
            "save_checkpoints": False,
            "use_fp16_compression": True,
            "num_workers": 0,
            "pin_memory": False,
            "optimizer": {
                "type": "Adam",
                "lr": 0.001,
                "weight_decay": 0.01,
            },
        },
        "trainer": {
            "type": "DANNStreamingLOSOTrainer",
            "args": {
                "subject_buffer_size": 49,
                "gradient_accumulation_steps": 1,
                "mixed_precision": False,
                "domain_grouping": "stroke_location_2class",
                "grl_gamma": 3.0,  # Gentler than 5 or 10
                "domain_loss_weight": 0.1,  # Very low to avoid task interference
                "use_validation_early_stopping": True,
                "early_stopping_patience": 15,
            },
        },
        "paths": {"root_dir": "./experiments"},
        "logging": {"level": "INFO", "console": True},
    }


def main():
    exp_name = "DANN_Full50_Conservative"
    config_path = EXPERIMENTS_DIR / f"_tmp_{exp_name}.yaml"

    # Write config
    config = build_config()
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("=" * 70)
    print("DANN Full50 Conservative Experiment")
    print("=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"Subjects: 50 (Full cohort)")
    print(f"Domain grouping: stroke_location_2class (Subcortical vs Non-subcortical)")
    print(f"Domain loss weight: 0.1 (conservative)")
    print(f"GRL gamma: 3.0 (gentle scheduling)")
    print(f"Domain hidden dims: [128, 64] (smaller for stability)")
    print(f"Epochs: 50")
    print(f"Early stopping: enabled (patience=15)")
    print("=" * 70)
    print(f"Baseline Aff+Align: {BASELINE_AFF*100:.2f}%")
    print(f"Baseline LR:        {BASELINE_LR*100:.2f}%")
    print(f"Target for success: > {BASELINE_AFF*100+1.5:.2f}% (+1.5% vs baseline)")
    print("=" * 70)
    print("")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "train.py", "--config", str(config_path), "--name", exp_name],
            cwd=PROJECT_ROOT,
            capture_output=False,  # Stream output to console for monitoring
            text=True,
            timeout=8 * 3600,  # 8 hours max
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n❌ EXPERIMENT FAILED (exit code {result.returncode})")
            return

        # Load results
        result_json = EXPERIMENTS_DIR / exp_name / "results" / "results.json"
        if not result_json.exists():
            print(f"\n⚠️  No results.json found at {result_json}")
            return

        with open(result_json) as f:
            data = json.load(f)

        mean_acc = data.get("overall_mean", 0)
        std_acc = data.get("overall_std", 0)

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"DANN Full50:        {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"Baseline Aff+Align: {BASELINE_AFF*100:.2f}% ± 5.23%")
        print(f"Baseline LR:        {BASELINE_LR*100:.2f}% ± 5.77%")
        print(f"Time:               {elapsed/3600:.1f} hours")
        print("")

        diff_vs_aff = (mean_acc - BASELINE_AFF) * 100
        diff_vs_lr = (mean_acc - BASELINE_LR) * 100

        if mean_acc > BASELINE_AFF + 0.015:
            print(f"🎯 SUCCESS: DANN beats Aff+Align by +{diff_vs_aff:.2f} pp")
            print("   Domain-invariant features are viable on this dataset.")
        elif mean_acc > BASELINE_AFF:
            print(f"🤔 MARGINAL: DANN beats Aff+Align by +{diff_vs_aff:.2f} pp")
            print("   Improvement is small but positive.")
        elif mean_acc > BASELINE_LR:
            print(f"⚠️  BELOW Aff+Align: {diff_vs_aff:.2f} pp")
            print(f"   But still above LR by +{diff_vs_lr:.2f} pp")
            print("   Domain adversarial may be interfering with task learning.")
        else:
            print(f"❌ FAILED: Below both baselines")
            print(f"   vs Aff+Align: {diff_vs_aff:.2f} pp")
            print(f"   vs LR:        {diff_vs_lr:.2f} pp")
            print("   Domain-invariant approach is not suitable for this dataset.")

        # Save summary
        summary = {
            "experiment": exp_name,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "baseline_aff": BASELINE_AFF,
            "baseline_lr": BASELINE_LR,
            "diff_vs_aff": diff_vs_aff,
            "diff_vs_lr": diff_vs_lr,
            "elapsed_hours": elapsed / 3600,
            "config": config["trainer"]["args"],
        }
        with open(RESULTS_DIR / "full50_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {RESULTS_DIR / 'full50_summary.json'}")

    except subprocess.TimeoutExpired:
        print(f"\n⏰ TIMEOUT after 8 hours")
    finally:
        if config_path.exists():
            config_path.unlink()


if __name__ == "__main__":
    main()
