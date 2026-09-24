#!/usr/bin/env python
"""
Plot per-subject pipeline accuracy comparison for EEGNet baselines.

Outputs:
  - results/comparisons/eegnet_4pipeline_heatmap.png
  - results/comparisons/eegnet_4pipeline_combined.png
  - results/comparisons/eegnet_4pipeline_selection.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PIPELINE_EXPERIMENTS = {
    "LR_noEA": "experiments/XWStroke_Full50_EEGNet_LR_noEA_noAug/results/results.json",
    "LR_EA": "experiments/XWStroke_Full50_EEGNet_LR_EA/results/results.json",
    "Aff_noEA": "experiments/XWStroke_Full50_EEGNet_AffAlign_noEA_noAug/results/results.json",
    "Aff_EA": "experiments/XWStroke_Full50_EEGNet_EA_AffAlign_NoMP/results/results.json",
}


def load_accs(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return {r["test_subject_id"]: r["test_acc"] for r in data["subjects"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/comparisons", type=str)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accs = {name: load_accs(path) for name, path in PIPELINE_EXPERIMENTS.items()}
    subject_ids = sorted(accs["LR_noEA"].keys())
    pipeline_names = list(PIPELINE_EXPERIMENTS.keys())

    # Build matrix: subjects x pipelines
    mat = np.zeros((len(subject_ids), len(pipeline_names)))
    for j, pipe in enumerate(pipeline_names):
        for i, sid in enumerate(subject_ids):
            mat[i, j] = accs[pipe][sid]

    # Best pipeline per subject
    best_pipeline_per_subject = []
    oracle_acc = []
    for i, sid in enumerate(subject_ids):
        best_idx = int(np.argmax(mat[i]))
        best_pipeline_per_subject.append(pipeline_names[best_idx])
        oracle_acc.append(mat[i, best_idx])

    mean_accs = {pipe: float(np.mean(list(accs[pipe].values()))) for pipe in pipeline_names}
    oracle_mean = float(np.mean(oracle_acc))
    oracle_std = float(np.std(oracle_acc))

    print("=== EEGNet 4-pipeline baseline summary ===")
    for pipe in pipeline_names:
        print(f"{pipe:12s}: {mean_accs[pipe]*100:.2f}%")
    print(f"Oracle      : {oracle_mean*100:.2f}% ± {oracle_std*100:.2f}%")

    # Save JSON
    selection = {
        "subject_ids": subject_ids,
        "pipeline_names": pipeline_names,
        "accuracies": {pipe: [accs[pipe][sid] for sid in subject_ids] for pipe in pipeline_names},
        "best_pipeline_per_subject": best_pipeline_per_subject,
        "mean_accuracies": mean_accs,
        "oracle_mean": oracle_mean,
        "oracle_std": oracle_std,
    }
    with open(out_dir / "eegnet_4pipeline_selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    # Plot 1: Combined means + best distribution (for paper)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    x = np.arange(len(pipeline_names))
    bars = ax.bar(x, [mean_accs[p] for p in pipeline_names], color="steelblue", alpha=0.8)
    ax.axhline(oracle_mean, color="darkorange", linestyle="--", linewidth=2,
               label=f"Oracle ({oracle_mean*100:.2f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pipeline_names)
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_ylim([0.45, 0.60])
    ax.set_title("(a) Mean LOSO Accuracy")
    ax.legend(loc="upper left")
    for bar, pipe in zip(bars, pipeline_names):
        height = bar.get_height()
        ax.annotate(f"{height*100:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    counts = pd.Series(best_pipeline_per_subject).value_counts().reindex(pipeline_names, fill_value=0)
    bars2 = ax.bar(counts.index, counts.values, color="seagreen", alpha=0.8)
    ax.set_ylabel("Number of Subjects")
    ax.set_title("(b) Per-Subject Best Pipeline")
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_dir / "eegnet_4pipeline_combined.png", dpi=300)
    plt.savefig(out_dir / "eegnet_4pipeline_combined.pdf", dpi=300)
    plt.close()

    # Plot 2: Heatmap (subjects x pipelines)
    fig, ax = plt.subplots(figsize=(8, 14))
    sns.heatmap(
        mat,
        xticklabels=pipeline_names,
        yticklabels=subject_ids,
        cmap="RdYlGn",
        vmin=0.35,
        vmax=0.75,
        cbar_kws={"label": "Test Accuracy"},
        ax=ax,
        linewidths=0.1,
    )
    ax.set_title("Per-Subject Test Accuracy Across EEGNet Pipelines (N=50)")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Subject ID")
    plt.tight_layout()
    plt.savefig(out_dir / "eegnet_4pipeline_heatmap.png", dpi=300)
    plt.close()

    print(f"Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
