"""E1c 主图：双向跨阶段迁移 vs within 基线（v2，修布局）。

输出：experiments/e1_acute_vs_chronic/figures/fig_e1c_transfer.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "experiments/e1_acute_vs_chronic"
OUT = f"{BASE}/figures"
os.makedirs(OUT, exist_ok=True)

within = {}
for arm in ["csp", "eegnet"]:
    for name in ["xw", "hs_ses01", "hs_all"]:
        df = pd.read_csv(f"{BASE}/e1a_within/{name}/{arm}_fold_results.csv")
        within[(arm, name)] = df.groupby("subject")["acc"].mean().mean() * 100

transfer = {}
for direction in ["xw2hs", "hs2xw"]:
    for arm in ["csp", "eegnet"]:
        df = pd.read_csv(f"{BASE}/e1c_transfer/{direction}_{arm}.csv")
        transfer[(direction, arm)] = df.groupby("target_subject")["acc"].mean().values * 100

fig, ax = plt.subplots(figsize=(10.5, 5.8))

groups = ["xw2hs", "hs2xw"]
group_labels = ["Acute -> Chronic\n(train XW -> test HS)", "Chronic -> Acute\n(train HS -> test XW)"]
arms = [("csp", "CSP+LDA", "tab:blue", -0.2), ("eegnet", "EEGNet", "tab:purple", 0.2)]

for gi, g in enumerate(groups):
    for arm, label, color, off in arms:
        v = transfer[(g, arm)]
        x = gi + off
        ax.bar(x, v.mean(), width=0.34, color=color, alpha=0.75,
               yerr=v.std() / np.sqrt(len(v)), capsize=4,
               label=label if gi == 0 else None)
        ax.scatter(np.full(len(v), x) + np.random.uniform(-0.07, 0.07, len(v)), v,
                   color=color, s=14, alpha=0.55, zorder=3)
        ax.text(x + (0.09 if off > 0 else -0.09), v.mean() + 1.3, f"{v.mean():.1f}%",
                ha="center", fontsize=9.5, fontweight="bold", color=color)

ax.axhline(50, color="k", ls="--", lw=1.1)
ax.text(-0.52, 50.5, "chance (50%)", fontsize=9)

# within 基线参考线（EEGNet 口径），标签放图右侧
bands = [
    (within[("eegnet", "hs_ses01")], "tab:orange", f"HS within ses-01: {within[('eegnet','hs_ses01')]:.1f}%"),
    (within[("eegnet", "hs_all")], "tab:brown", f"HS within all-ses: {within[('eegnet','hs_all')]:.1f}%"),
    (within[("eegnet", "xw")], "tab:red", f"XW within: {within[('eegnet','xw')]:.1f}%"),
]
for val, color, label in bands:
    ax.axhline(val, color=color, ls=":", lw=1.4, alpha=0.85)
    ax.annotate(label, xy=(1.0, val), xycoords=("axes fraction", "data"),
                xytext=(8, 0), textcoords="offset points", fontsize=9, color=color,
                va="center", ha="left")

ax.set_xticks(range(2))
ax.set_xticklabels(group_labels, fontsize=10)
ax.set_ylabel("target-subject accuracy (%)")
ax.set_ylim(40, 70)
ax.set_xlim(-0.6, 1.6)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.set_title("E1c: bidirectional cross-stage transfer (zero-shot) — both directions at chance\n"
             "dotted lines = within-stage baselines from E1a (EEGNet)", fontsize=11)
fig.tight_layout()
fig.subplots_adjust(right=0.78)
out = f"{OUT}/fig_e1c_transfer.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
