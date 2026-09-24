#!/usr/bin/env python3
"""
Within-subject ablation #3: artifact removal (SASICA/EMAR) under clean protocol.

The previous ablations showed window config, classic CNN backbones, and
multi-band band-attention all fail to lift within-subject accuracy above
~52-55% (clean protocol). The last remaining within-subject lever that has not
been tested under the clean protocol is preprocessing quality (artifact
removal). Thwe et al. (2026) reported SASICA/EMAR gains, but under a
best-epoch protocol.

Design:
  - Data configs: XWStroke_singleband_sasica.yaml (SASICA),
                  XWStroke_singleband_emar.yaml (EMAR, optional),
                  XWStroke.yaml (baseline, no artifact removal — already run)
  - Models: ADFCNN, EEGNet
  - Protocol: identical clean protocol — 5-fold GroupKFold by trial,
              fixed 100 epochs, augmentation p=0.3 train-only, Adam lr=1e-3,
              no best-epoch peeking.

Outputs (results/ablation_artifact/):
  - artifact_results.csv, fig_artifact.png, report.md
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
from scripts.ablation_within_backbone_window import train_model, eval_acc

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "ablation_artifact"

DATASET_YAMLS = {
    "sasica": ROOT / "configs" / "dataset" / "XWStroke_singleband_sasica.yaml",
    "emar": ROOT / "configs" / "dataset" / "XWStroke_singleband_emar.yaml",
}
MODEL_NAMES = ["ADFCNN", "EEGNet"]
SEED = 42


def build_model(name, num_channels, num_bands, input_length):
    if name == "EEGNet":
        return EEGNet(num_classes=2, num_channels=num_channels,
                      num_bands=num_bands, input_length=input_length)
    if name == "ADFCNN":
        return ADFCNN(num_classes=2, num_channels=num_channels,
                      num_bands=num_bands, input_length=input_length,
                      use_band_attention=True)
    raise ValueError(name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    csv_path = OUT_DIR / "artifact_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        rows = []
        for cfg_name, cfg_path in DATASET_YAMLS.items():
            if not cfg_path.exists():
                print(f"[warn] missing {cfg_path}, skip")
                continue
            with open(cfg_path) as f:
                info = yaml.safe_load(f)
            for sid in range(1, 51):
                t_load = time.time()
                try:
                    ds = XWStrokeDataset(subject_id=sid, dataset_info=info, mode="test")
                except Exception as e:
                    print(f"[{cfg_name}|sub{sid:02d}] LOAD FAILED: {e}", flush=True)
                    continue
                X = np.asarray(ds.data)
                y = np.asarray(ds.labels)
                if set(np.unique(y).tolist()) == {1, 2}:
                    y = y - 1
                groups = np.asarray(ds.group_ids)
                _, n_bands, n_ch, n_t = X.shape
                print(f"[{cfg_name}|sub{sid:02d}] loaded {X.shape} "
                      f"({time.time()-t_load:.0f}s)", flush=True)
                for mname in MODEL_NAMES:
                    t0 = time.time()
                    accs = []
                    for tr, te in GroupKFold(5).split(X, y, groups):
                        model = build_model(mname, n_ch, n_bands, n_t)
                        model = train_model(model, X[tr], y[tr], device, ds.augmentor)
                        accs.append(eval_acc(model, X[te], y[te], device))
                        del model
                    rows.append({"subject_id": sid, "config": cfg_name,
                                 "model": mname, "mean_acc": float(np.mean(accs)),
                                 "std_acc": float(np.std(accs))})
                    print(f"[{cfg_name}|{mname}|sub{sid:02d}] "
                          f"{np.mean(accs)*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
                    if device == "cuda":
                        torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    summarize(df)


def summarize(df):
    summary = df.groupby(["config", "model"])["mean_acc"].agg(["mean", "std"])
    print("\n== Artifact-removal within-subject summary ==")
    print((summary * 100).round(2).to_string())

    configs = [c for c in DATASET_YAMLS if c in df["config"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(MODEL_NAMES))
    width = 0.8 / max(1, len(configs))
    for i, cfg in enumerate(configs):
        vals = [summary.loc[(cfg, m), "mean"] * 100 for m in MODEL_NAMES]
        errs = [summary.loc[(cfg, m), "std"] * 100 for m in MODEL_NAMES]
        ax.bar(x + (i - (len(configs) - 1) / 2) * width, vals, width,
               yerr=errs, capsize=3, label=cfg, alpha=0.85)
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel("Within-subject 5-fold acc (%)")
    ax.set_title("Artifact removal ablation (clean protocol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_artifact.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 伪迹去除消融（干净协议）\n",
        "- 协议: 5-fold GroupKFold by trial, fixed 100 epochs, 增强仅训练侧, 无 best-epoch",
        "- 对照: 无伪迹去除基线（XWStroke.yaml）ADFCNN 54.95% / EEGNet 52.79%\n",
        "| config | model | mean | std |",
        "|---|---|---|---|",
    ]
    for (cfg, m), row in summary.iterrows():
        lines.append(f"| {cfg} | {m} | {row['mean']*100:.2f}% | {row['std']*100:.2f}% |")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
