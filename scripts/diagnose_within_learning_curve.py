#!/usr/bin/env python3
"""
Diagnostic 1: within-subject learning curve — is the bottleneck data size or signal SNR?

Train within-subject models with N training trials (N = 4, 8, 16, 24, 32 out of
32 available per fold) and evaluate on the held-out 8 trials, under the clean
protocol (5-fold GroupKFold by trial, fixed epochs, aug train-only).

  - If the curve is still rising at N=32 (not saturated): the within
    performance is data-limited -> more data / augmentation / few-shot can help.
  - If the curve is already flat: the within performance is signal-SNR-limited
    -> no method will help; signal enhancement is the only lever.

Outputs (results/diagnosis_learning_curve/):
  - learning_curve.csv, fig_learning_curve.png, report.md
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.datasets import XWStrokeDataset
from scripts.ablation_multiband_within import build_model
from scripts.ablation_within_backbone_window import train_model, eval_acc

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "diagnosis_learning_curve"
DATASET_YAML = ROOT / "configs" / "dataset" / "XWStroke.yaml"

N_TRIALS_LIST = [4, 8, 16, 24, 32]   # out of 32 train trials per fold
MODEL_NAMES = ["ADFCNN", "EEGNet"]
SEED = 42


def subsample_trials(X, y, groups, n_trials, seed):
    """Pick n_trials unique trials (stratified-ish) from the train set."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    # stratified: keep class balance of selected trials
    trial_label = {g: y[groups == g][0] for g in uniq}
    idx0 = [g for g in uniq if trial_label[g] == 0]
    idx1 = [g for g in uniq if trial_label[g] == 1]
    n0 = min(len(idx0), n_trials // 2)
    n1 = min(len(idx1), n_trials - n0)
    n0 = n_trials - n1  # top up if one class short
    sel = list(rng.choice(idx0, size=min(n0, len(idx0)), replace=False)) + \
          list(rng.choice(idx1, size=min(n1, len(idx1)), replace=False))
    mask = np.isin(groups, sel)
    return X[mask], y[mask]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(DATASET_YAML) as f:
        info = yaml.safe_load(f)
    print(f"device: {device}")

    csv_path = OUT_DIR / "learning_curve.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        rows = []
        for sid in range(1, 51):
            ds = XWStrokeDataset(subject_id=sid, dataset_info=info, mode="test")
            X = np.asarray(ds.data)
            y = np.asarray(ds.labels)
            if set(np.unique(y).tolist()) == {1, 2}:
                y = y - 1
            groups = np.asarray(ds.group_ids)
            _, nb, nch, nt = X.shape
            for mname in MODEL_NAMES:
                t0 = time.time()
                for n_tr in N_TRIALS_LIST:
                    accs = []
                    for fold_i, (tr, te) in enumerate(
                            GroupKFold(5).split(X, y, groups)):
                        Xtr, ytr = subsample_trials(
                            X[tr], y[tr], groups[tr], n_tr, SEED + fold_i)
                        model = build_model(mname, nch, nb, nt)
                        model = train_model(model, Xtr, ytr, device, ds.augmentor)
                        accs.append(eval_acc(model, X[te], y[te], device))
                        del model
                    rows.append({"subject_id": sid, "model": mname,
                                 "n_train_trials": n_tr,
                                 "mean_acc": float(np.mean(accs))})
                print(f"[sub{sid:02d}|{mname}] done  ({time.time()-t0:.0f}s)",
                      flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    summarize(df)


def summarize(df):
    curve = df.groupby(["model", "n_train_trials"])["mean_acc"].agg(["mean", "std"])
    print("\n== Learning curve ==")
    print((curve * 100).round(2).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mname, color in zip(MODEL_NAMES, ["#4472C4", "#8496B0"]):
        sub = curve.loc[mname]
        x = sub.index.values
        ax.errorbar(x, sub["mean"] * 100, yerr=sub["std"] * 100,
                    marker="o", capsize=3, label=mname, color=color)
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.set_xlabel("Training trials per fold (of 32 available)")
    ax.set_ylabel("Within-subject acc (%)")
    ax.set_title("Learning curve: is within-subject performance data-limited? (clean protocol)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_learning_curve.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 诊断 1：学习曲线\n",
        "- 协议: 5-fold GroupKFold by trial, 每 fold 32 train / 8 test trials,",
        "  从 train 中抽取 N trials 训练，fixed epochs, 增强仅训练侧\n",
        "| model | N=4 | N=8 | N=16 | N=24 | N=32 |",
        "|---|---|---|---|---|---|",
    ]
    for mname in MODEL_NAMES:
        sub = curve.loc[mname]
        vals = " | ".join(f"{sub.loc[n, 'mean']*100:.1f}%" for n in N_TRIALS_LIST)
        lines.append(f"| {mname} | {vals} |")
    # saturation estimate: gain from N=16 to N=32
    lines.append("\n## 饱和判断")
    for mname in MODEL_NAMES:
        sub = curve.loc[mname]
        gain = (sub.loc[32, "mean"] - sub.loc[16, "mean"]) * 100
        lines.append(f"- {mname}: N=16→32 增益 {gain:+.1f} 点")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
