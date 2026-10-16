"""E1g 主图：N-shot 校准曲线（预训练微调 vs 从零训练，双向）。

输出：experiments/e1_acute_vs_chronic/figures/fig_e1g_fewshot.png
两面板：左 xw2hs（急性→慢性），右 hs2xw（慢性→急性）；
实线=源域预训练+微调，虚线=从零训练；阴影=SEM（across 目标被试×seed）；
水平线：chance 50%、目标域 within 上限（E1a EEGNet：HS all-ses 55.5% / XW 51.8%）。
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

N_LIST = [0, 2, 4, 8, 16, 32]
PANELS = [
    ("xw2hs", "Acute -> Chronic (pretrain XW -> calibrate HS)",
     ("HS within all-ses: 55.5%", 55.48, "tab:brown")),
    ("hs2xw", "Chronic -> Acute (pretrain HS -> calibrate XW)",
     ("XW within: 51.8%", 51.78, "tab:red")),
]
ARMS = [("finetune", "pretrained + fine-tune", "-", "tab:blue"),
        ("scratch", "from scratch (no pretrain)", "--", "tab:gray")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, (direction, title, (wl, wv, wc)) in zip(axes, PANELS):
    xs = np.arange(len(N_LIST))  # 等距分类坐标，避免 log2(0)
    for arm, label, ls, color in ARMS:
        df = pd.read_csv(f"{BASE}/e1_fewshot/{direction}_{arm}.csv")
        # 先按 (subject, seed) 聚合，再跨被试×seed 汇总
        per = df.groupby(["n_shot", "target_subject", "seed"])["acc"].mean().reset_index()
        g = per.groupby("n_shot")["acc"]
        mean_s = g.mean() * 100
        sem_s = (g.std() / np.sqrt(g.count())) * 100
        ns = mean_s.index.values
        xi = np.array([N_LIST.index(int(n)) for n in ns])
        mean = mean_s.values
        sem = sem_s.values
        ax.plot(xi, mean, ls, color=color, marker="o", ms=4, label=label)
        ax.fill_between(xi, mean - sem, mean + sem, color=color, alpha=0.15)
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.axhline(wv, color=wc, ls=":", lw=1.4)
    ax.text(0.05, wv + 0.4, wl, fontsize=8.5, color=wc)
    ax.text(0.05, 50.4, "chance", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in N_LIST])
    ax.set_xlim(-0.3, len(N_LIST) - 0.7)
    ax.set_xlabel("N labeled trials from target subject")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(46, 60)
    ax.legend(fontsize=9, loc="lower right")
axes[0].set_ylabel("target-subject accuracy (%)")
fig.suptitle("E1g: N-shot calibration after cross-stage pretraining — pretraining adds ~nothing",
             fontsize=12)
fig.tight_layout()
out = f"{OUT}/fig_e1g_fewshot.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
