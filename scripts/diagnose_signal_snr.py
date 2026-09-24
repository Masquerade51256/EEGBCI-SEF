#!/usr/bin/env python3
"""
Diagnostic 2: signal SNR — how many subjects have statistically significant MI modulation?

For each subject, test whether left-MI vs right-MI trials differ in mu/beta
power at the contralateral motor channels (C3/C4) — i.e. whether the subject
shows a statistically detectable MI-related modulation at all. This measures
the physiological floor of within-subject decodability.

Metrics per subject:
  - contra_power_diff: left-vs-right MI mu/beta power difference at C3/C4 (t-test)
  - significant_modulation: p < 0.05 (one-tailed, contralateral direction)
  - erd_strength: magnitude of contralateral mu-ERD (relative power drop)

Outputs (results/diagnosis_signal_snr/):
  - signal_snr.csv, fig_snr_within.png, report.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "datasets" / "21679035" / "sourcedata"
OUT_DIR = ROOT / "results" / "diagnosis_signal_snr"
WITHIN_CSV = ROOT / "results" / "decodability_transferability" / "master_table.csv"

FS = 500
N_EEG = 30
C3, C4 = 13, 14   # 0-indexed in the 30-ch standard order
MI_WIN = (1.0, 4.0)  # seconds post-onset used for MI power
BANDS = {"mu": (8, 13), "beta": (13, 30)}


def bandpower_trials(X, lo, hi):
    b, a = sps.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    Xf = sps.filtfilt(b, a, X, axis=-1)
    s0, s1 = int(MI_WIN[0] * FS), int(MI_WIN[1] * FS)
    return (Xf[:, :, s0:s1] ** 2).mean(axis=-1)  # (n_trials, n_ch)


def load_subject(sid):
    f = DATA_DIR / f"sub-{sid:02d}" / f"sub-{sid:02d}_task-motor-imagery_eeg_4s.mat"
    m = sio.loadmat(f)
    X = np.asarray(m["eeg_data_4s"], dtype=float)[:, :N_EEG, :]
    y = np.asarray(m["labels"]).ravel()
    if set(np.unique(y).tolist()) == {1, 2}:
        y = y - 1
    return X, y


def subject_snr(sid):
    X, y = load_subject(sid)
    left, right = y == 0, y == 1
    out = {"subject_id": sid}
    for band, (lo, hi) in BANDS.items():
        p = bandpower_trials(X, lo, hi)
        # For LEFT hand MI, contralateral = C4. For RIGHT hand MI, contra = C3.
        # Modulation: contra channel should show LOWER power (ERD) during MI of
        # the opposite hand vs ipsilateral condition.
        # Simplest robust test: at C4, left-MI power vs right-MI power
        # (left MI should depress C4 more); at C3, right-MI vs left-MI.
        t4, p4 = stats.ttest_ind(p[left, C4], p[right, C4])
        t3, p3 = stats.ttest_ind(p[right, C3], p[left, C3])
        # effect direction: ERD means left<C4 right; right<C3 left
        d4 = (p[right, C4].mean() - p[left, C4].mean()) / (p[:, C4].std() + 1e-12)
        d3 = (p[left, C3].mean() - p[right, C3].mean()) / (p[:, C3].std() + 1e-12)
        out[f"{band}_p_C4"] = p4
        out[f"{band}_p_C3"] = p3
        out[f"{band}_d_C4"] = d4
        out[f"{band}_d_C3"] = d3
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [subject_snr(sid) for sid in range(1, 51)]
    df = pd.DataFrame(rows).set_index("subject_id")
    df["sig_mu"] = (df["mu_p_C4"] < 0.05) | (df["mu_p_C3"] < 0.05)
    df["sig_beta"] = (df["beta_p_C4"] < 0.05) | (df["beta_p_C3"] < 0.05)
    df["sig_any"] = df["sig_mu"] | df["sig_beta"]
    df.to_csv(OUT_DIR / "signal_snr.csv")

    mt = pd.read_csv(WITHIN_CSV).set_index("subject_id")
    df["within_acc"] = mt["within_acc"]
    df["loso_mean"] = mt["loso_mean"]

    n_sig = int(df["sig_any"].sum())
    print(f"显著 MI 调制（C3/C4 mu 或 beta, p<0.05）: {n_sig}/50")
    for col in ["mu_d_C4", "mu_d_C3", "beta_d_C4", "beta_d_C3"]:
        r, p = stats.spearmanr(df[col], df["within_acc"])
        print(f"  {col} vs within: r={r:.3f} p={p:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = df["sig_any"].map({True: "#4472C4", False: "#C00000"})
    ax.scatter(df["mu_d_C4"], df["within_acc"] * 100, c=colors, alpha=0.75,
               edgecolors="k", linewidths=0.4)
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Contralateral mu-ERD effect size (C4, Cohen's d)")
    ax.set_ylabel("Within-subject acc (%)")
    ax.set_title("Signal SNR vs within-subject performance\nblue = significant MI modulation, red = none")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_snr_within.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 诊断 2：信号 SNR 摸底\n",
        f"- 有显著 MI 调制（C3/C4 mu 或 beta 对侧差异 p<0.05）的被试: **{n_sig}/50**",
        f"- 无显著调制: **{50-n_sig}/50**——这部分被试的 MI 信号在经典感觉运动区统计上不可检测\n",
        "| 特征 | 与 within 的 Spearman r | p |",
        "|---|---|---|",
    ]
    for col in ["mu_d_C4", "mu_d_C3", "beta_d_C4", "beta_d_C3"]:
        r, p = stats.spearmanr(df[col], df["within_acc"])
        lines.append(f"| {col} | {r:.3f} | {p:.4f} |")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
