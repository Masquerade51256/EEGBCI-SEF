"""E1 sanity check：7 通道对齐后两 loader 的数值与生理方向验证。

检查项：
1. 形状一致 (n_windows, 1, 7, 500)，通道顺序 [FC3,FC4,C3,Cz,C4,CP3,CP4]
2. 标签 0/1 平衡，group_ids == trial 数（GroupKFold 分组依据）
3. 生理方向：左/右手 MI 在 C3(idx2)/C4(idx4) 的 mu(8-12Hz) 功率侧化方向
   预期：左 MI -> C4 功率下降（C3-C4 > 0）；右 MI -> C3 功率下降（C3-C4 < 0）
   两数据集应呈现同向模式（与 XWStroke 审计的侧化分析同口径）

运行：服务器项目根目录下 python scripts/e1/sanity_check_loaders.py
输出：experiments/e1_acute_vs_chronic/sanity/e1_sanity_lateralization.png
"""
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

sys.path.insert(0, ".")
from data.datasets import XWStrokeDataset, HSStrokeDataset

CH_ORDER = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
C3_IDX, C4_IDX = 2, 4
FS = 250
MU = (8, 12)


def load(cfg_path, cls, subject_id, sessions_override=None):
    with open(cfg_path) as f:
        info = yaml.safe_load(f)
    if sessions_override is not None:
        info["dataset"]["sessions"] = sessions_override
    return cls(subject_id=subject_id, dataset_info=info, mode="test")


def mu_power_by_trial(ds):
    """返回 per-trial: mu 功率 (n_trials, 7)（对其所有窗取均值）与标签"""
    data = ds.data[:, 0, :, :]  # (n_win, 7, 500)
    f, Pxx = signal.welch(data, fs=FS, axis=-1, nperseg=250)
    mu_mask = (f >= MU[0]) & (f <= MU[1])
    p = Pxx[:, :, mu_mask].mean(axis=-1)  # (n_win, 7)
    groups = ds.group_ids
    trials, labels = [], []
    for g in np.unique(groups):
        m = groups == g
        trials.append(p[m].mean(axis=0))
        labels.append(ds.labels[m][0])
    return np.asarray(trials), np.asarray(labels)


def cohens_d(a, b):
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


report = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

for ax, (name, cfg, cls, ses) in zip(axes, [
    ("XWStroke sub-01 (acute)", "configs/dataset/XWStroke_7ch_hsalign.yaml", XWStrokeDataset, None),
    ("HSStroke sub-01 ses-01 (chronic)", "configs/dataset/HSStroke_7ch_250hz.yaml", HSStrokeDataset, [1]),
]):
    ds = load(cfg, cls, 1, ses)
    n_win, n_band, n_ch, n_t = ds.data.shape
    lab_uniq, lab_cnt = np.unique(ds.labels, return_counts=True)
    n_trials = len(np.unique(ds.group_ids))
    print(f"[{name}] windows={ds.data.shape} trials={n_trials} labels={dict(zip(lab_uniq.tolist(), lab_cnt.tolist()))}")
    assert (n_band, n_ch, n_t) == (1, 7, 500), f"形状异常: {ds.data.shape}"

    P, y = mu_power_by_trial(ds)
    lat = np.log(P[:, C3_IDX]) - np.log(P[:, C4_IDX])  # log(C3/C4)
    d = cohens_d(lat[y == 0], lat[y == 1])
    report[name] = {"n_trials": int(n_trials), "d_lat": float(d),
                    "lat_left": float(lat[y == 0].mean()), "lat_right": float(lat[y == 1].mean())}
    print(f"  侧化 log(C3/C4): left={lat[y==0].mean():+.4f} (n={int((y==0).sum())}), "
          f"right={lat[y==1].mean():+.4f} (n={int((y==1).sum())}), Cohen's d={d:+.3f}")

    x = np.arange(2)
    w = 0.35
    for i, (cond, c) in enumerate([("left MI", "tab:blue"), ("right MI", "tab:red")]):
        m = y == i
        means = [np.log(P[m, C3_IDX]).mean(), np.log(P[m, C4_IDX]).mean()]
        sems = [np.log(P[m, C3_IDX]).std() / np.sqrt(m.sum()), np.log(P[m, C4_IDX]).std() / np.sqrt(m.sum())]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=sems, label=cond, color=c, alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["C3", "C4"])
    ax.set_title(f"{name}\n侧化 d={d:+.3f}")
    ax.set_ylabel("mean log mu power (8-12Hz)")
    ax.legend(fontsize=8)

fig.suptitle("E1 sanity: 7ch alignment & MI lateralization direction")
fig.tight_layout()
out = "experiments/e1_acute_vs_chronic/sanity/e1_sanity_lateralization.png"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nfigure saved: {out}")
print("生理方向判读: left MI 的 C4 功率应低于 C3（对侧 ERD），两数据集应同向")
