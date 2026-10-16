"""E1a 主图：急性 vs 慢性 within-subject 对比（CSP+LDA 与 EEGNet 双方法）。

输出：experiments/e1_acute_vs_chronic/figures/fig_e1a_within.png
布局：1 行 2 列，左 CSP+LDA，右 EEGNet；每图三组 per-subject 分布
（XW 急性 n=50 / HS ses-01 n=14 / HS 全量 n=14），均值±SEM 柱 + 散点。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "experiments/e1_acute_vs_chronic/e1a_within"
OUT = "experiments/e1_acute_vs_chronic/figures"
os.makedirs(OUT, exist_ok=True)

ARMS = [("csp", "CSP + LDA"), ("eegnet", "EEGNet (fixed 100ep)")]
GROUPS = [("xw", "XWStroke acute\n(n=50, 40 trials)", "tab:red"),
          ("hs_ses01", "HSStroke ses-01\n(n=14, 100 trials)", "tab:orange"),
          ("hs_all", "HSStroke all 20 ses\n(n=14, ~1600 trials)", "tab:brown")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (arm, arm_label) in zip(axes, ARMS):
    for i, (name, label, color) in enumerate(GROUPS):
        df = pd.read_csv(f"{BASE}/{name}/{arm}_fold_results.csv")
        v = df.groupby("subject")["acc"].mean().values * 100
        ax.bar(i, v.mean(), yerr=v.std() / np.sqrt(len(v)), color=color, alpha=0.7,
               capsize=5, error_kw={"lw": 1.5})
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.16, 0.16, len(v)), v,
                   color=color, s=20, alpha=0.75, zorder=3)
        ax.text(i, v.mean() + v.std() / np.sqrt(len(v)) + 3, f"{v.mean():.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(50, color="k", ls="--", lw=0.9, alpha=0.6)
    ax.text(2.35, 50.6, "chance", fontsize=8, color="k", alpha=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[1] for g in GROUPS], fontsize=9)
    ax.set_title(arm_label, fontsize=12)
    ax.set_ylim(35, 90)
axes[0].set_ylabel("within-subject accuracy (%)")
fig.suptitle("E1a: within-subject MI decoding — acute vs chronic stroke (7ch aligned, clean protocol)",
             fontsize=12)
fig.tight_layout()
out = f"{OUT}/fig_e1a_within.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
