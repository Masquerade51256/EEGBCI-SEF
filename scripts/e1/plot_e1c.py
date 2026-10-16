"""E1c 主图：双向跨阶段迁移 vs within 基线。

输出：experiments/e1_acute_vs_chronic/figures/fig_e1c_transfer.png
设计：两组（急性→慢性 / 慢性→急性）× 两方法（CSP+LDA / EEGNet）柱 + per-target 散点；
水平参考带为 E1a within 基线（XW 51.8%、HS ses-01 62.6%、HS all 55.5%）；chance=50%。
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

# E1a within 基线（EEGNet 3 seeds / CSP）
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

fig, ax = plt.subplots(figsize=(9.5, 5.5))

# within 基线参考带（EEGNet 口径）
bands = [
    (within[("eegnet", "hs_ses01")], "tab:orange", f"HS within ses-01: {within[('eegnet','hs_ses01')]:.1f}%"),
    (within[("eegnet", "hs_all")], "tab:brown", f"HS within all-ses: {within[('eegnet','hs_all')]:.1f}%"),
    (within[("eegnet", "xw")], "tab:red", f"XW within: {within[('eegnet','xw')]:.1f}%"),
]
for val, color, label in bands:
    ax.axhline(val, color=color, ls=":", lw=1.4, alpha=0.85)
    ax.text(3.62, val + 0.4, label, fontsize=8.5, color=color, ha="right")

groups = ["xw2hs", "hs2xw"]
group_labels = ["Acute -> Chronic\n(train XW -> test HS)", "Chronic -> Acute\n(train HS -> test XW)"]
arms = [("csp", "CSP+LDA", "tab:blue", -0.18), ("eegnet", "EEGNet", "tab:purple", 0.18)]

for gi, g in enumerate(groups):
    for arm, label, color, off in arms:
        v = transfer[(g, arm)]
        x = gi + off
        ax.bar(x, v.mean(), width=0.32, color=color, alpha=0.75,
               yerr=v.std() / np.sqrt(len(v)), capsize=4,
               label=label if gi == 0 else None)
        ax.scatter(np.full(len(v), x) + np.random.uniform(-0.06, 0.06, len(v)), v,
                   color=color, s=14, alpha=0.6, zorder=3)
        ax.text(x, v.mean() + 1.2, f"{v.mean():.1f}%", ha="center", fontsize=10, fontweight="bold")

ax.axhline(50, color="k", ls="--", lw=1.1)
ax.text(3.62, 50.5, "chance (50%)", fontsize=8.5, ha="right")
ax.set_xticks(range(2))
ax.set_xticklabels(group_labels, fontsize=10)
ax.set_ylabel("target-subject accuracy (%)")
ax.set_ylim(40, 72)
ax.set_xlim(-0.55, 1.55)
ax.legend(loc="lower left", fontsize=9)
ax.set_title("E1c: bidirectional cross-stage transfer (zero-shot) — both directions at chance\n"
             "dotted lines = within-stage baselines (E1a), 7ch aligned, clean protocol", fontsize=11)
fig.tight_layout()
out = f"{OUT}/fig_e1c_transfer.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
