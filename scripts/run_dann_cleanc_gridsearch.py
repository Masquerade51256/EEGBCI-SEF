#!/usr/bin/env python3
"""
DANN Grid Search on Clean-C subset.

Searches domain_loss_weight × grl_gamma combinations using NIHSS 3-class
as domain labels (the only viable grouping for the homogeneous Clean-C subset).

Usage:
    python scripts/run_dann_cleanc_gridsearch.py

Outputs:
    - experiments/XWStroke_DANN_CleanC_GridSearch_*/results/results.json
    - results/dann_cleanc_gridsearch/summary.csv
    - results/dann_cleanc_gridsearch/comparison.png
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results" / "dann_cleanc_gridsearch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Search Space
# ------------------------------------------------------------------
GRID = {
    "domain_loss_weight": [0.1, 0.3, 0.5, 1.0],
    "grl_gamma": [5, 10],
}

CLEAN_C_SUBJECTS = [8, 9, 11, 16, 17, 23, 25, 26, 27, 29, 30, 31, 36, 37]

# Baseline for comparison
BASELINE_ACC = 0.5229  # Clean-C Aff+Align
BASELINE_STD = 0.0419


def build_config(domain_loss_weight: float, grl_gamma: float, exp_name: str) -> dict:
    """Build experiment config dict."""
    return {
        "experiment": {
            "name": exp_name,
            "seed": 42,
            "description": f"Clean-C DANN: dlw={domain_loss_weight}, gamma={grl_gamma}",
        },
        "data": {
            "dataset": "XWStroke",
            "subjects": CLEAN_C_SUBJECTS,
            "info_path": "configs/dataset/XWStroke_affected_aligned.yaml",
        },
        "model": {
            "type": "DANNADFCNN",
            "args": {
                "num_domains": 2,  # nihss_3class in Clean-C only has Mild vs Moderate
                "domain_hidden_dims": [256, 128],
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
                "subject_buffer_size": 13,
                "gradient_accumulation_steps": 1,
                "mixed_precision": False,
                "domain_grouping": "nihss_3class",
                "grl_gamma": grl_gamma,
                "domain_loss_weight": domain_loss_weight,
                "use_validation_early_stopping": True,
                "early_stopping_patience": 15,
            },
        },
        "paths": {"root_dir": "./experiments"},
        "logging": {"level": "INFO", "console": True},
    }


def run_single_config(domain_loss_weight: float, grl_gamma: float) -> dict:
    """Run one DANN configuration."""
    exp_name = f"DANN_CleanC_dlw{domain_loss_weight}_g{grl_gamma}"
    config_path = EXPERIMENTS_DIR / f"_tmp_{exp_name}.yaml"

    # Write temp config
    import yaml

    config = build_config(domain_loss_weight, grl_gamma, exp_name)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"  domain_loss_weight={domain_loss_weight}, grl_gamma={grl_gamma}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "train.py", "--config", str(config_path), "--name", exp_name],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per config
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  ❌ FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}")
            return {
                "config": exp_name,
                "domain_loss_weight": domain_loss_weight,
                "grl_gamma": grl_gamma,
                "status": "failed",
                "mean_acc": None,
                "std_acc": None,
                "elapsed_min": elapsed / 60,
            }

        # Load results
        result_json = EXPERIMENTS_DIR / exp_name / "results" / "results.json"
        if not result_json.exists():
            print(f"  ⚠️  No results.json found")
            return {
                "config": exp_name,
                "domain_loss_weight": domain_loss_weight,
                "grl_gamma": grl_gamma,
                "status": "no_results",
                "mean_acc": None,
                "std_acc": None,
                "elapsed_min": elapsed / 60,
            }

        with open(result_json) as f:
            data = json.load(f)

        mean_acc = data.get("overall_mean", 0)
        std_acc = data.get("overall_std", 0)

        print(f"  ✅ DONE in {elapsed/60:.1f} min")
        print(f"  Acc: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

        # Check if beats baseline
        if mean_acc > BASELINE_ACC:
            diff = (mean_acc - BASELINE_ACC) * 100
            print(f"  🎯 BEATS baseline by +{diff:.2f} pp")

        return {
            "config": exp_name,
            "domain_loss_weight": domain_loss_weight,
            "grl_gamma": grl_gamma,
            "status": "success",
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "elapsed_min": elapsed / 60,
        }

    except subprocess.TimeoutExpired:
        print(f"  ⏰ TIMEOUT after 60 min")
        return {
            "config": exp_name,
            "domain_loss_weight": domain_loss_weight,
            "grl_gamma": grl_gamma,
            "status": "timeout",
            "mean_acc": None,
            "std_acc": None,
            "elapsed_min": 60,
        }
    finally:
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()


def main():
    print("=" * 70)
    print("DANN Grid Search on Clean-C")
    print("=" * 70)
    print(f"Baseline: Clean-C Aff+Align = {BASELINE_ACC*100:.2f}% ± {BASELINE_STD*100:.2f}%")
    print(f"Search space: {len(GRID['domain_loss_weight'])} × {len(GRID['grl_gamma'])} = {len(GRID['domain_loss_weight'])*len(GRID['grl_gamma'])} configs")
    print(f"Domain grouping: nihss_3class (Mild vs Moderate in Clean-C)")
    print(f"Epochs: 50, Early stopping: enabled")
    print("=" * 70)

    results = []
    for dlw in GRID["domain_loss_weight"]:
        for gamma in GRID["grl_gamma"]:
            res = run_single_config(dlw, gamma)
            results.append(res)

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"\nSummary saved to: {RESULTS_DIR / 'summary.csv'}")

    # Print ranked results
    print("\n" + "=" * 70)
    print("RANKED RESULTS")
    print("=" * 70)
    success_df = df[df["status"] == "success"].sort_values("mean_acc", ascending=False)
    print(success_df.to_string(index=False))

    # Best config
    if len(success_df) > 0:
        best = success_df.iloc[0]
        print(f"\n🏆 BEST CONFIG:")
        print(f"  domain_loss_weight = {best['domain_loss_weight']}")
        print(f"  grl_gamma = {best['grl_gamma']}")
        print(f"  Accuracy = {best['mean_acc']*100:.2f}% ± {best['std_acc']*100:.2f}%")
        if best["mean_acc"] > BASELINE_ACC:
            print(f"  vs Baseline = +{(best['mean_acc'] - BASELINE_ACC)*100:.2f} pp")
        else:
            print(f"  vs Baseline = {(best['mean_acc'] - BASELINE_ACC)*100:.2f} pp")

    # Visualization
    plot_results(success_df)


def plot_results(df: pd.DataFrame):
    """Create heatmap of grid search results."""
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pivot for heatmap
    pivot_mean = df.pivot(index="grl_gamma", columns="domain_loss_weight", values="mean_acc")
    pivot_std = df.pivot(index="grl_gamma", columns="domain_loss_weight", values="std_acc")

    # Panel A: Mean accuracy heatmap
    ax = axes[0]
    im = ax.imshow(pivot_mean.values, cmap="RdYlGn", aspect="auto", vmin=0.45, vmax=0.60)
    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels(pivot_mean.columns)
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_yticklabels(pivot_mean.index)
    ax.set_xlabel("Domain Loss Weight")
    ax.set_ylabel("GRL Gamma")
    ax.set_title("Mean Accuracy")

    # Annotate
    for i in range(len(pivot_mean.index)):
        for j in range(len(pivot_mean.columns)):
            val = pivot_mean.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.52 else "black"
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy")

    # Panel B: Std heatmap
    ax = axes[1]
    im2 = ax.imshow(pivot_std.values, cmap="RdYlGn_r", aspect="auto", vmin=0.03, vmax=0.08)
    ax.set_xticks(range(len(pivot_std.columns)))
    ax.set_xticklabels(pivot_std.columns)
    ax.set_yticks(range(len(pivot_std.index)))
    ax.set_yticklabels(pivot_std.index)
    ax.set_xlabel("Domain Loss Weight")
    ax.set_ylabel("GRL Gamma")
    ax.set_title("Std (lower = more stable)")

    for i in range(len(pivot_std.index)):
        for j in range(len(pivot_std.columns)):
            val = pivot_std.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.06 else "black"
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    plt.colorbar(im2, ax=ax, label="Std")

    # Baseline reference line
    for ax in axes:
        ax.axhline(y=-0.5, color="blue", linewidth=2, linestyle="--", alpha=0.5)
        ax.text(1.5, -0.8, f"Baseline: {BASELINE_ACC*100:.2f}%", color="blue", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gridsearch_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to: {RESULTS_DIR / 'gridsearch_heatmap.png'}")


if __name__ == "__main__":
    main()
