"""E1 sanity（群体级）：三数据集 mu 侧化方向验证 + 复现审计数字。

单被试 mu 侧化噪声太大（健康人单 session 也会反号），群体级才是有效验证：
- SHU（25 健康人 × 5 sessions）：群体 d 必须显著为正（教科书方向）→ 验证计算管线
- XWStroke（50 急性）/ HSStroke（14 慢性）：审计值为 +0.075 / +0.063，
  用 E1 统一管线（7 通道、4s 全段 mu 功率）应复现同向弱侧化

侧化指标：lat = log P_mu(C3) - log P_mu(C4)，d = Cohen's d(lat|left, lat|right)
（左 MI → 右侧 C4 ERD → lat 偏大为正；群体 d > 0 为教科书方向）

输出：
  experiments/e1_acute_vs_chronic/sanity/population_lateralization.csv
  experiments/e1_acute_vs_chronic/sanity/population_lateralization.png
"""
import sys
import os
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
import mne

sys.path.insert(0, ".")
from data.datasets import XWStrokeDataset, HSStrokeDataset

FS = 250
MU = (8, 12)
mne.set_log_level("ERROR")
OUT = "experiments/e1_acute_vs_chronic/sanity"
os.makedirs(OUT, exist_ok=True)


def lat_from_segments(segs_c3, segs_c4):
    """segs: (n, T) -> per-trial lat = logP(C3)-logP(C4)"""
    f, P3 = signal.welch(segs_c3, fs=FS, axis=-1, nperseg=min(250, segs_c3.shape[-1]))
    f, P4 = signal.welch(segs_c4, fs=FS, axis=-1, nperseg=min(250, segs_c4.shape[-1]))
    m = (f >= MU[0]) & (f <= MU[1])
    return np.log(P3[:, m].mean(-1)) - np.log(P4[:, m].mean(-1))


def cohens_d(a, b):
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


rows = []

# ---- SHU（健康人参照，直接读 EDF；C3=12, C4=13）----
shu_subj = []
for sid in range(1, 26):
    lats, ys = [], []
    for ses in range(1, 6):
        edf = f"src/datasets/SHU/edf_raw/edf/sub-{sid:03d}_ses-{ses:02d}_task_motorimagery_eeg.edf"
        evp = f"src/datasets/SHU/events_raw/events/sub-{sid:03d}_ses-{ses:02d}_task_motorimagery_events.tsv"
        if not os.path.exists(edf):
            continue
        raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
        sig = raw.get_data() * 1e6
        ev = pd.read_csv(evp, sep="\t")
        if "sample" not in ev.columns:
            ev = pd.read_csv(evp, sep="\t", header=None,
                             names=["onset", "duration", "trial_type", "response_time", "sample", "value"])
        c3 = np.stack([sig[12, int(r["sample"]):int(r["sample"]) + 1000] for _, r in ev.iterrows()])
        c4 = np.stack([sig[13, int(r["sample"]):int(r["sample"]) + 1000] for _, r in ev.iterrows()])
        lats.append(lat_from_segments(c3, c4))
        ys.append(np.array([0 if str(t).strip() == "left" else 1 for t in ev["trial_type"]]))
    lat = np.concatenate(lats)
    y = np.concatenate(ys)
    d = cohens_d(lat[y == 0], lat[y == 1])
    shu_subj.append(d)
    rows.append(("SHU_healthy", f"sub-{sid:03d}", len(y), d))
    print(f"SHU sub-{sid:03d}: d={d:+.3f} (n={len(y)})", flush=True)

# ---- XWStroke / HSStroke（走 E1 的 7ch dataset 类；C3=idx2, C4=idx4）----
def dataset_subject_lat(cls, cfg_path, sid):
    with open(cfg_path) as f:
        info = yaml.safe_load(f)
    ds = cls(subject_id=sid, dataset_info=info, mode="test")
    data = ds.data[:, 0, :, :]  # (n_win, 7, 500)
    lat_w = lat_from_segments(data[:, 2, :], data[:, 4, :])
    groups, labels = ds.group_ids, ds.labels
    lat_t, y_t = [], []
    for g in np.unique(groups):
        m = groups == g
        lat_t.append(lat_w[m].mean())
        y_t.append(labels[m][0])
    return np.asarray(lat_t), np.asarray(y_t)

for name, cls, cfg, sids in [
    ("XWStroke_acute", XWStrokeDataset, "configs/dataset/XWStroke_7ch_hsalign.yaml", range(1, 51)),
    ("HSStroke_chronic", HSStrokeDataset, "configs/dataset/HSStroke_7ch_250hz.yaml", range(1, 15)),
]:
    ds_list = []
    for sid in sids:
        try:
            lat, y = dataset_subject_lat(cls, cfg, sid)
            d = cohens_d(lat[y == 0], lat[y == 1])
            ds_list.append(d)
            rows.append((name, f"sub-{sid:02d}", len(y), d))
            print(f"{name} sub-{sid:02d}: d={d:+.3f} (n={len(y)})", flush=True)
        except Exception as e:
            rows.append((name, f"sub-{sid:02d}", 0, np.nan))
            print(f"{name} sub-{sid:02d}: ERROR {e}", flush=True)

df = pd.DataFrame(rows, columns=["dataset", "subject", "n_trials", "d_lat"])
df.to_csv(f"{OUT}/population_lateralization.csv", index=False)

# ---- 汇总与图 ----
fig, ax = plt.subplots(figsize=(7, 4.5))
names = ["SHU_healthy", "XWStroke_acute", "HSStroke_chronic"]
colors = ["tab:green", "tab:red", "tab:orange"]
for i, (nm, c) in enumerate(zip(names, colors)):
    v = df[df.dataset == nm]["d_lat"].dropna().values
    ax.bar(i, v.mean(), yerr=v.std() / np.sqrt(len(v)), color=c, alpha=0.75, capsize=5)
    ax.scatter(np.full(len(v), i) + np.random.uniform(-0.15, 0.15, len(v)), v,
               color=c, s=18, alpha=0.7, zorder=3)
    print(f"{nm}: 群体 d = {v.mean():+.3f} ± {v.std():.3f} (n={len(v)}, SEM={v.std()/np.sqrt(len(v)):.3f})")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels(["SHU healthy\n(n=25)", "XWStroke acute\n(n=50)", "HSStroke chronic\n(n=14)"])
ax.set_ylabel("lateralization Cohen's d (mu, C3 vs C4)")
ax.set_title("Population-level MI lateralization (positive = textbook direction)")
fig.tight_layout()
fig.savefig(f"{OUT}/population_lateralization.png", dpi=150, bbox_inches="tight")
print(f"\noutputs: {OUT}/population_lateralization.csv/.png")
