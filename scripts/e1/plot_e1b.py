"""E1b 主图：预训练源对比——患者源 LOSO vs 健康源(E1c+) vs 从零。

输出：experiments/e1_acute_vs_chronic/figures/fig_e1b_source_compare.png
两面板：左 XW 急性目标，右 HS 慢性目标；
蓝实线=健康源预训练+微调（E1c+），绿实线=患者源 LOSO 预训练+微调（E1b），
灰虚线=从零训练；阴影=SEM（across 目标被试×seed）；
水平线：chance 50%、目标域 within 上限（XW 51.8% / HS all-ses 55.5%）。
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
    ("xw", "bcic2xw", "Target: Acute (XWStroke)",
     ("XW within: 51.8%", 51.78, "tab:red")),
    ("hs", "bcic2hs", "Target: Chronic (HS Stroke)",
     ("HS within all-ses: 55.5%", 55.48, "tab:brown")),
]
# (label, ls, color, csv path template)
SERIES = [
    ("healthy-pretrained (BCICIV2a)", "-", "tab:blue",
     lambda ds, d2: f"{BASE}/e1_fewshot/{d2}_finetune.csv"),
    ("patient-pretrained (LOSO pool)", "-", "tab:green",
     lambda ds, d2: f"{BASE}/e1b_loso/{ds}_finetune.csv"),
    ("from scratch (no pretrain)", "--", "tab:gray",
     lambda ds, d2: f"{BASE}/e1b_loso/{ds}_scratch.csv"),
]
# 健康源 vs 患者源配对 t 显著性（见 E1b 统计）
SIG = {"xw": {0: "***", 2: "***", 4: "***", 8: "***", 16: "***"}, "hs": {}}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, (ds, d2, title, (wl, wv, wc)) in zip(axes, PANELS):
    xs = np.arange(len(N_LIST))
    for label, ls, color, get_csv in SERIES:
        df = pd.read_csv(get_csv(ds, d2))
        per = df.groupby(["n_shot", "target_subject", "seed"])["acc"].mean().reset_index()
        g = per.groupby("n_shot")["acc"]
        mean_s, sem_s = g.mean() * 100, (g.std() / np.sqrt(g.count())) * 100
        xi = np.array([N_LIST.index(int(n)) for n in mean_s.index])
        ax.plot(xi, mean_s.values, ls, color=color, marker="o", ms=4, label=label)
        ax.fill_between(xi, (mean_s - sem_s).values, (mean_s + sem_s).values,
                        color=color, alpha=0.13)
    for n, stars in SIG[ds].items():
        i = N_LIST.index(n)
        y = pd.read_csv(f"{BASE}/e1_fewshot/{d2}_finetune.csv")
        ymax = y[y.n_shot == n]["acc"].mean() * 100
        ax.text(i, ymax + 0.9, stars, ha="center", fontsize=11, color="tab:blue")
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
fig.suptitle("E1b vs E1c+: healthy pretraining beats patient-pool pretraining "
             "on acute targets (***), no difference on chronic",
             fontsize=12)
fig.tight_layout()
out = f"{OUT}/fig_e1b_source_compare.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
