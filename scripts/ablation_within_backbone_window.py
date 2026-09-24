#!/usr/bin/env python3
"""
Within-subject ablation: data representation x backbone, clean protocol.

Question: is the poor within-subject performance (~55% under the clean
fixed-epoch protocol) driven by the data representation (2s sliding windows
@250Hz vs full 4s trials @500Hz) or by the backbone?

Design (all other factors fixed):
  - Window configs:  W  or),
                     W4 (4s/4s no overlap @500Hz)
  - Backbones:       EEGNet, ADFCNN(band-attention), ShallowConvNet, DeepConvNet
  - Protocol:        5-fold GroupKFold by trial (no sliding-window leakage),
                     fixed 100 epochs, augmentation p=0.3 on train only,
                     Adam lr=1e-3, NO best-epoch selection (final epoch eval).

Outputs (results/ablation_within/):
  - ablation_results.csv   per subject x config x model fold accuracies
  - fig_ablation.png       grouped summary
  - report.md
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert( 0, str(Path(__file__).resolve().parent.parent))
from data.datasets import XWStrokeDataset
from models.EEGNet import EEGNet
from models.ADFCNN import ADFCNN
from models import MODELS, register_all_models  # registry adapters

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "ablation_within"

DATASET_YAMLS = {
    "w2_250hz": ROOT / "configs" / "dataset" / "XWStroke.yaml",
    "w4_500hz": ROOT / "configs" / "dataset" / "XWStroke_ablation_4s.yaml",
}
MODEL_NAMES = ["EEGNet", "ADFCNN", "ShallowConvNet", "DeepConvNet"]

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
SEED = 42


def build_model(name, num_channels, num_bands, input_length):
    if name == "EEGNet":
        return EEGNet(num_classes=2, num_channels=num_channels,
                      num_bands=num_bands, input_length=input_length)
    if name == "ADFCNN":
        return ADFCNN(num_classes=2, num_channels=num_channels,
                      num_bands=num_bands, input_length=input_length,
                      use_band_attention=True)
    if name in ("ShallowConvNet", "DeepConvNet"):
        cls = MODELS.get(name)  # registry adapter (reshape + linear head)
        return cls(num_channels=num_channels, num_classes=2,
                   num_bands=num_bands, input_length=input_length)
    raise ValueError(name)


def train_model(model, Xtr, ytr, device, augmentor):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    ytr_t = torch.from_numpy(np.ascontiguousarray(ytr)).long()
    n = len(Xtr)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx_t = perm[i:i + BATCH_SIZE]
            idx = idx_t.numpy()
            xb_np = Xtr[idx]
            if augmentor is not None:
                xb_np = np.stack([augmentor.process(x) for x in xb_np])
            xb = torch.from_numpy(np.ascontiguousarray(xb_np)).float().to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def eval_acc(model, X, y, device):
    model.eval()
    xb = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
    yb = torch.from_numpy(np.ascontiguousarray(y)).long().to(device)
    return (model(xb).argmax(1) == yb).float().mean().item()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    csv_path = OUT_DIR / "ablation_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        rows = []
        for cfg_name, cfg_path in DATASET_YAMLS.items():
            with open(cfg_path) as f:
                info = yaml.safe_load(f)
            for sid in range(1, 51):
                ds = XWStrokeDataset(subject_id=sid, dataset_info=info, mode="test")
                X = np.asarray(ds.data)
                y = np.asarray(ds.labels)
                if set(np.unique(y).tolist()) == {1, 2}:
                    y = y - 1
                groups = np.asarray(ds.group_ids)
                _, n_ch, n_t = X.shape[1], X.shape[2], X.shape[3]
                for mname in MODEL_NAMES:
                    t0 = time.time()
                    accs = []
                    for tr, te in GroupKFold(5).split(X, y, groups):
                        model = build_model(mname, n_ch, X.shape[1], n_t)
                        model = train_model(model, X[tr], y[tr], device, ds.augmentor)
                        accs.append(eval_acc(model, X[te], y[te], device))
                        del model
                    rows.append({"subject_id": sid, "config": cfg_name,
                                 "model": mname, "mean_acc": float(np.mean(accs)),
                                 "std_acc": float(np.std(accs))})
                    print(f"[{cfg_name}|{mname}|sub{sid:02d}] "
                          f"{np.mean(accs)*100:.1f}%  ({time.time()-t0:.0f}s)",
                          flush=True)
                    if device == "cuda":
                        torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    summarize(df)


def summarize(df):
    pivot = df.pivot_table(index="subject_id", columns=["config", "model"],
                           values="mean_acc")

    summary = df.groupby(["config", "model"])["mean_acc"].agg(["mean", "std"])
    print("\n== Summary (mean within-subject acc) ==")
    print((summary * 100).round( 2).to_string())

    # figure: bar chart config x model
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = MODEL_NAMES
    configs = list(DATASET_YAMLS)
    x = np.arange(len(models))
    width = 0.35
    for i, cfg in enumerate(configs):
        vals = [summary.loc[(cfg, m), "mean"] * 100 for m in models]
        errs = [summary.loc[(cfg, m), "std"] * 100 for m in models]
        ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=3,
               label=cfg, alpha=0.85)
    ax.axhline(50, color="gray", ls="--", lw= 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Within-subject 5-fold acc (%)")
    ax.set_title("Clean-protocol ablation: data representation x backbone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_ablation.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Within-subject 消融：数据表示 × backbone（干净协议）\n",
        "- 协议: 5-fold GroupKFold（按 trial 分组防泄漏）, 固定 100 epochs, "
        "增强仅训练侧, 无 best-epoch 挑选",
        "- W2: 2s/0.5s 滑窗 @250Hz（现行标准）; W4: 4s 整窗 @500Hz\n",
        "## 汇总\n",
        "| config | model | mean | std |",
        "|---|---|---|---|",
    ]
    for (cfg, m), row in summary.iterrows():
        lines.append(f"| {cfg} | {m} | {row['mean']*100:.2f}% | {row['std']*100:.2f}% |")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
