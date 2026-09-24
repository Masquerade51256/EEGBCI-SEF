#!/usr/bin/env python3
"""
Attribution probe: what do within-subject models actually rely on?

Motivation (P0 follow-up): 19/50 subjects have decodable signals (within >= 55%)
but failed cross-subject transfer (quadrant B). Hypothesis: their within-subject
models rely on individual-specific, non-canonical patterns instead of the classic
contralateral C3/C4 mu ERD. This script trains one within-subject ADFCNN per
subject (same data pipeline as the LOSO experiments: _4s.mat -> 250Hz -> 30ch ->
4-40Hz -> 2s/0.5s windows -> zscore), then measures on held-out windows:
  - channel occlusion importance:  Delta CE loss when channel c is zeroed out
  - band occlusion importance:     Delta CE loss when band b is notch-filtered out

Finally compares attribution maps between quadrant A (transfers) and B
(transfer fails) from results/decodability_transferability/master_table.csv.

Outputs (results/within_attribution/):
  - attribution_arrays.npz     channel_imp (50x30), band_imp (50x4), test_acc (50)
  - attribution_table.csv      per-subject importance + quadrant
  - fig_channel_importance_AB.png
  - fig_band_importance_AB.png
  - fig_topo_AB.png
  - report.md
"""

import json
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
from scipy import signal as sps
from scipy import stats
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.datasets import XWStrokeDataset
from models.ADFCNN import ADFCNN

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "within_attribution"
DATASET_YAML = ROOT / "configs" / "dataset" / "XWStroke.yaml"  # p=0.3 augmentation, matches the within-subject reference protocol
MASTER_TABLE = ROOT / "results" / "decodability_transferability" / "master_table.csv"

CHANNELS = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FCZ', 'FC3', 'FC4',
            'FT7', 'FT8', 'CZ', 'C3', 'C4', 'T3', 'T4', 'CPZ', 'CP3', 'CP4',
            'TP7', 'TP8', 'PZ', 'P3', 'P4', 'T5', 'T6', 'OZ', 'O1', 'O2']
BANDS = {"theta": (4, 8), "mu": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
FS = 250
# "classic" sensorimotor channels for the concentration index
MOTOR_SET = ['C3', 'C4', 'CZ', 'CP3', 'CP4', 'FC3', 'FC4']

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
SEED = 42


# ---------------------------------------------------------------------------
def load_subject_xy(subject_id: int, dataset_info: dict):
    ds = XWStrokeDataset(subject_id=subject_id, dataset_info=dataset_info, mode="test")
    X = np.asarray(ds.data)          # (n_windows, n_bands=1, 30, 500)
    y = np.asarray(ds.labels)        # may be {1,2} or {0,1}
    uniq = np.unique(y)
    if set(uniq.tolist()) == {1, 2}:
        y = y - 1
    assert set(np.unique(y).tolist()) <= {0, 1}, f"unexpected labels {uniq}"
    return X, y, ds.augmentor


def train_within_model(Xtr, ytr, device, augmentor=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # ADFCNN with band attention: matches the within-subject reference
    # protocol (WithinSubjectKFoldTrainer, 63.77% mean) used for the
    # quadrant definition, so attribution is measured on the same model
    # family that produced the within-subject accuracies.
    model = ADFCNN(num_classes=2, num_channels=Xtr.shape[2],
                   num_bands=Xtr.shape[1], input_length=Xtr.shape[3],
                   use_band_attention=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr)).float()
    ytr_t = torch.from_numpy(np.ascontiguousarray(ytr)).long()
    n = len(Xtr_t)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE].numpy()
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
def eval_loss_acc(model, X, y, device):
    model.eval()
    xb = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
    yb = torch.from_numpy(np.ascontiguousarray(y)).long().to(device)
    logits = model(xb)
    loss = nn.CrossEntropyLoss()(logits, yb).item()
    acc = (logits.argmax(1) == yb).float().mean().item()
    return loss, acc


def notch_filter(X, lo, hi, fs=FS, order=4):
    b, a = sps.iirfilter(order, [lo / (fs / 2), hi / (fs / 2)], btype="bandstop")
    return sps.filtfilt(b, a, X, axis=-1)


def occlude_channel(X, ch):
    Xo = X.copy()
    Xo[:, :, ch, :] = 0.0
    return Xo


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_YAML) as f:
        dataset_info = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    npz_path = OUT_DIR / "attribution_arrays.npz"
    if npz_path.exists():
        print("loading cached attribution arrays")
        z = np.load(npz_path)
        channel_imp, band_imp, test_accs = z["channel_imp"], z["band_imp"], z["test_acc"]
    else:
        channel_imp = np.zeros((50, len(CHANNELS)))
        band_imp = np.zeros((50, len(BANDS)))
        test_accs = np.zeros(50)
        for i, sid in enumerate(range(1, 51)):
            t0 = time.time()
            X, y, augmentor = load_subject_xy(sid, dataset_info)
            n_win = len(y)
            n_trials = 40
            win_per_trial = n_win // n_trials
            assert n_win % n_trials == 0, f"unexpected window count {n_win}"
            # windows are generated trial-major; verify labels are constant
            # within each block of win_per_trial windows
            yg = y.reshape(n_trials, win_per_trial)
            assert (yg == yg[:, :1]).all(), "windows not trial-major; grouping invalid"
            groups = np.repeat(np.arange(n_trials), win_per_trial)

            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=5)
            ch_folds, bd_folds, acc_folds = [], [], []
            for tr_idx, te_idx in gkf.split(X, y, groups):
                model = train_within_model(X[tr_idx], y[tr_idx], device, augmentor)
                Xte, yte = X[te_idx], y[te_idx]
                base_loss, base_acc = eval_loss_acc(model, Xte, yte, device)
                acc_folds.append(base_acc)
                ch_folds.append([eval_loss_acc(model, occlude_channel(Xte, c),
                                               yte, device)[0] - base_loss
                                 for c in range(len(CHANNELS))])
                bd_folds.append([eval_loss_acc(model, notch_filter(Xte, lo, hi),
                                               yte, device)[0] - base_loss
                                 for lo, hi in BANDS.values()])
                del model
            channel_imp[i] = np.mean(ch_folds, axis=0)
            band_imp[i] = np.mean(bd_folds, axis=0)
            test_accs[i] = np.mean(acc_folds)
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[{sid:02d}/50] within acc={test_accs[i]*100:.1f}%  "
                  f"({time.time()-t0:.1f}s)")
        np.savez(npz_path, channel_imp=channel_imp, band_imp=band_imp,
                 test_acc=test_accs)

    analyze(channel_imp, band_imp, test_accs)


# ---------------------------------------------------------------------------
def normalize_rows(M):
    Mpos = np.maximum(M, 0)
    s = Mpos.sum(axis=1, keepdims=True)
    return np.divide(Mpos, s, out=np.full_like(Mpos, 1.0 / M.shape[1]),
                     where=s > 0)


def analyze(channel_imp, band_imp, test_accs):
    mt = pd.read_csv(MASTER_TABLE).set_index("subject_id")
    quad = mt["quadrant"].str[0]  # 'A'/'B'/'C'/'D'

    df = pd.DataFrame({
        "subject_id": range(1, 51),
        "test_acc": test_accs,
        "quadrant": quad.reindex(range(1, 51)).values,
    })
    ch_norm = normalize_rows(channel_imp)
    band_norm = normalize_rows(band_imp)
    for c, name in enumerate(CHANNELS):
        df[f"ch_{name}"] = ch_norm[:, c]
    for b, name in enumerate(BANDS):
        df[f"band_{name}"] = band_norm[:, b]

    motor_idx = [CHANNELS.index(c) for c in MOTOR_SET]
    df["motor_concentration"] = ch_norm[:, motor_idx].sum(axis=1)
    df.to_csv(OUT_DIR / "attribution_table.csv", index=False)

    ab = df[df["quadrant"].isin(["A", "B"])]
    A = ab[ab["quadrant"] == "A"]
    B = ab[ab["quadrant"] == "B"]
    print(f"\nA: n={len(A)} within(test)={A['test_acc'].mean()*100:.1f}%  "
          f"B: n={len(B)} within(test)={B['test_acc'].mean()*100:.1f}%")

    # --- per-channel & per-band group tests --------------------------------
    rows = []
    for c, name in enumerate(CHANNELS):
        u, p = stats.mannwhitneyu(A[f"ch_{name}"], B[f"ch_{name}"])
        rows.append({"kind": "channel", "name": name,
                     "A_mean": A[f"ch_{name}"].mean(), "B_mean": B[f"ch_{name}"].mean(),
                     "p": p})
    for name in BANDS:
        u, p = stats.mannwhitneyu(A[f"band_{name}"], B[f"band_{name}"])
        rows.append({"kind": "band", "name": name,
                     "A_mean": A[f"band_{name}"].mean(), "B_mean": B[f"band_{name}"].mean(),
                     "p": p})
    u, p = stats.mannwhitneyu(A["motor_concentration"], B["motor_concentration"])
    rows.append({"kind": "index", "name": "motor_concentration",
                 "A_mean": A["motor_concentration"].mean(),
                 "B_mean": B["motor_concentration"].mean(), "p": p})
    cmp_df = pd.DataFrame(rows)
    # BH-FDR within kinds
    cmp_df["p_fdr"] = np.nan
    for kind in cmp_df["kind"].unique():
        m = cmp_df["kind"] == kind
        pv = cmp_df.loc[m, "p"].values
        order = np.argsort(pv)
        adj = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(len(pv))
        out[order] = np.clip(adj, 0, 1)
        cmp_df.loc[m, "p_fdr"] = out
    cmp_df.to_csv(OUT_DIR / "group_comparison.csv", index=False)

    # --- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(CHANNELS))
    ax.bar(x - 0.2, [A[f"ch_{c}"].mean() for c in CHANNELS], 0.4, label=f"A (n={len(A)})")
    ax.bar(x + 0.2, [B[f"ch_{c}"].mean() for c in CHANNELS], 0.4, label=f"B (n={len(B)})")
    for c, name in enumerate(CHANNELS):
        if cmp_df[(cmp_df["kind"] == "channel") & (cmp_df["name"] == name)]["p_fdr"].iloc[0] < 0.05:
            ax.text(c, ax.get_ylim()[1] * 0.95, "*", ha="center", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=90)
    ax.set_ylabel("Normalized occlusion importance")
    ax.set_title("Channel importance: quadrant A vs B (within-subject ADFCNN)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_channel_importance_AB.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    xb = np.arange(len(BANDS))
    ax.bar(xb - 0.2, [A[f"band_{b}"].mean() for b in BANDS], 0.4, label="A")
    ax.bar(xb + 0.2, [B[f"band_{b}"].mean() for b in BANDS], 0.4, label="B")
    ax.set_xticks(xb)
    ax.set_xticklabels(list(BANDS.keys()))
    ax.set_ylabel("Normalized occlusion importance")
    ax.set_title("Band importance: A vs B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_band_importance_AB.png", dpi=160)
    plt.close(fig)

    try:
        _plot_topomaps(A, B, CHANNELS)
    except Exception as e:
        print(f"[warn] topomap skipped: {e}")

    # --- report --------------------------------------------------------------
    lines = [
        "# Attribution probe: A vs B 象限归因对比\n",
        f"- 模型: ADFCNN (band attention) within-subject, 5-fold GroupKFold（按 trial 分组，避免滑窗泄漏）, {EPOCHS} epochs, 数据管线同 LOSO（_4s.mat→250Hz→30ch→4-40Hz→2s/0.5s→zscore）",
        f"- 归因指标: held-out 测试窗上的 CE loss 增量（通道置零 / 频段带阻），5-fold 平均后行归一化",
        f"- sanity check: 50 人 test acc mean = {test_accs.mean()*100:.2f}%（应与 within-subject ~63.8% 同一量级）\n",
        "## 组间对比（Mann-Whitney, BH-FDR within kind）\n",
        "| kind | name | A mean | B mean | p | p_FDR |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in cmp_df.sort_values(["kind", "p"]).iterrows():
        lines.append(f"| {r['kind']} | {r['name']} | {r['A_mean']:.4f} | "
                     f"{r['B_mean']:.4f} | {r['p']:.4f} | {r['p_fdr']:.4f} |")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n== top channel differences (A vs B) ==")
    print(cmp_df[cmp_df["kind"] == "channel"].sort_values("p").head(8).to_string(index=False))
    print("\n== band differences ==")
    print(cmp_df[cmp_df["kind"] == "band"].to_string(index=False))
    print("\n== motor concentration ==")
    print(cmp_df[cmp_df["kind"] == "index"].to_string(index=False))
    print(f"\nDone. Outputs in {OUT_DIR}")


def _plot_topomaps(A, B, channels):
    import mne
    std_names = {'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz',
                 'CPZ': 'CPz', 'PZ': 'Pz', 'OZ': 'Oz', 'T3': 'T7', 'T4': 'T8',
                 'T5': 'P7', 'T6': 'P8'}
    plot_names = [std_names.get(c, c) for c in channels]
    info = mne.create_info(plot_names, sfreq=FS, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    va = np.array([A[f"ch_{c}"].mean() for c in channels])
    vb = np.array([B[f"ch_{c}"].mean() for c in channels])
    for ax, vals, title in zip(axes, [va, vb, va - vb],
                               ["A (transfers)", "B (transfer fails)", "A - B"]):
        mne.viz.plot_topomap(vals, info, axes=ax, show=False, contours=0)
        ax.set_title(title)
    fig.suptitle("Occlusion importance topography")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_topo_AB.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
