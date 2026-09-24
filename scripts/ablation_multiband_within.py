#!/usr/bin/env python3
"""
Within-subject ablation #2: multi-band input + band/channel attention.

The first ablation (ablation_within_backbone_window.py) showed window config and
classic single-band CNN backbones all converge to ~52-55% under the clean
protocol. However, none  of those backbones actually exercised band/channel
attention: they were fed a single wide band (4-40Hz), so ADFCNN's BandAttention
degenerated to num_bands=1. Attribution analysis suggested band-dependence
differences (beta vs theta) between transferable and non-transferable subjects.

This ablation tests the band/channel-adaptive design axis with multi-band input
(8 bands: 4-8, 8-12, 12-16, 16-20, 20-24, 24-30, 30-36, 36-40 Hz):
  - EEGNet (multi-band as input channels; no band attention) — control
  - ADFCNN (BandwiseSpectral + BandAttention, native multi-band)
  - XWFilterBankNet (per-band extractors + cross-band attention)

Protocol (identical clean protocol): 5-fold GroupKFold by trial, fixed 100
epochs, augmentation p=0.3 on train only, Adam lr=1e-3, no best-epoch peeking.

Outputs (results/ablation_multiband/):
  - multiband_results.csv, fig_multiband.png, report.md
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.datasets import XWStrokeDataset
from models.EEGNet import EEGNet
from models.ADFCNN import ADFCNN
from models.XWFilterBankNet import XWFilterBankNet
from scripts.ablation_within_backbone_window import train_model, eval_acc, EPOCHS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "ablation_multiband"
DATASET_YAML = ROOT / "configs" / "dataset" / "XWStroke_multiband.yaml"

MODEL_NAMES = ["EEGNet", "ADFCNN", "XWFilterBankNet"]
SEED = 42


def build_model(name, num_channels, num_bands, input_length):
    if name == "EEGNet":
        return EEGNet(num_classes=2, num_channels=num_channels,
                      num_bands=num_bands, input_length=input_length)
    if name == "ADFCNN":
        return ADFCNN(num_classes=2, num_channels=num_channels,
                      num_bands=num_bands, input_length=input_length,
                      use_band_attention=True)
    if name == "XWFilterBankNet":
        return XWFilterBankNet(num_classes=2, num_channels=num_channels,
                               num_bands=num_bands, input_length=input_length)
    raise ValueError(name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(DATASET_YAML) as f:
        info = yaml.safe_load(f)
    print(f"device: {device}, dataset: {DATASET_YAML.name}")

    csv_path = OUT_DIR / "multiband_results.csv"
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
            _, n_bands, n_ch, n_t = X.shape
            for mname in MODEL_NAMES:
                t0 = time.time()
                accs = []
                for tr, te in GroupKFold(5).split(X, y, groups):
                    model = build_model(mname, n_ch, n_bands, n_t)
                    model = train_model(model, X[tr], y[tr], device, ds.augmentor)
                    accs.append(eval_acc(model, X[te], y[te], device))
                    del model
                rows.append({"subject_id": sid, "model": mname,
                             "mean_acc": float(np.mean(accs)),
                             "std_acc": float(np.std(accs))})
                print(f"[{mname}|sub{sid:02d}] {np.mean(accs)*100:.1f}%  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    summarize(df)


def summarize(df):
    summary = df.groupby("model")["mean_acc"].agg(["mean", "std"])
    print("\n== Multi-band within-subject summary ==")
    print((summary * 100).round(2).to_string())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    models = MODEL_NAMES
    vals = [summary.loc[m, "mean"] * 100 for m in models]
    errs = [summary.loc[m, "std"] * 100 for m in models]
    ax.bar(models, vals, yerr=errs, capsize=4, color=["#8496B0", "#4472C4", "#ED7D31"])
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.set_ylabel("Within-subject 5-fold acc (%)")
    ax.set_title("Multi-band (8-band) input, clean protocol")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_multiband.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 多频带 + 频段注意力消融（干净协议）\n",
        "- 数据: XWStroke_multiband.yaml（8 频带 4-40Hz, 2s/0.5s 滑窗 @250Hz）",
        "- 协议: 5-fold GroupKFold by trial, fixed 100 epochs, 增强仅训练侧, 无 best-epoch\n",
        "| model | mean | std |",
        "|---|---|---|",
    ]
    for m, row in summary.iterrows():
        lines.append(f"| {m} | {row['mean']*100:.2f}% | {row['std']*100:.2f}% |")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
