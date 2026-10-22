"""E1c+ 主图：健康(BCICIV2a)→患者 N-shot 校准曲线（预训练微调 vs 从零训练）。

输出：experiments/e1_acute_vs_chronic/figures/fig_e1cplus_fewshot.png
两面板：左 bcic2xw（健康→急性），右 bcic2hs（健康→慢性）；
实线=健康预训练+微调，虚线=从零训练；阴影=SEM（across 目标被试×seed）；
水平线：chance 50%、目标域 within 上限（E1a EEGNet：XW 51.8% / HS all-ses 55.5%）。
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
    ("bcic2xw", "Healthy -> Acute (pretrain BCICIV2a -> calibrate XW)",
     ("XW within: 51.8%", 51.78, "tab:red")),
    ("bcic2hs", "Healthy -> Chronic (pretrain BCICIV2a -> calibrate HS)",
     ("HS within all-ses: 55.5%", 55.48, "tab:brown")),
]
ARMS = [("finetune", "healthy-pretrained + fine-tune", "-", "tab:blue"),
        ("scratch", "from scratch (no pretrain)", "--", "tab:gray")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, (direction, title, (wl, wv, wc)) in zip(axes, PANELS):
    xs = np.arange(len(N_LIST))  # 等距分类坐标，避免 log2(0)
    for arm, label, ls, color in ARMS:
        df = pd.read_csv(f"{BASE}/e1_fewshot/{direction}_{arm}.csv")
        per = df.groupby(["n_shot", "target_subject", "seed"])["acc"].mean().reset_index()
        g = per.groupby("n_shot")["acc"]
        mean_s = g.mean() * 100
        sem_s = (g.std() / np.sqrt(g.count())) * 100
        xi = np.array([N_LIST.index(int(n)) for n in mean_s.index])
        ax.plot(xi, mean_s.values, ls, color=color, marker="o", ms=4, label=label)
        ax.fill_between(xi, (mean_s - sem_s).values, (mean_s + sem_s).values,
                        color=color, alpha=0.15)
    # 显著性标注（配对 t，见 E1c+ 统计）
    SIG = {"bcic2xw": {2: "*", 4: "***", 8: "***", 16: "*"}, "bcic2hs": {}}
    for n, stars in SIG[direction].items():
        i = N_LIST.index(n)
        ymax = 0
        for arm, *_ in ARMS:
            df = pd.read_csv(f"{BASE}/e1_fewshot/{direction}_{arm}.csv")
            v = df[df.n_shot == n]["acc"].mean() * 100
            ymax = max(ymax, v)
        ax.text(i, ymax + 0.8, stars, ha="center", fontsize=11, color="tab:blue")
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
fig.suptitle("E1c+: healthy pretraining helps acute-phase calibration (N=4-8, ***/\*), "
             "not chronic",
             fontsize=12)
fig.tight_layout()
out = f"{OUT}/fig_e1cplus_fewshot.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
