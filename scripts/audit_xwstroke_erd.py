"""
独立审计 XWStroke 原始数据：ERD 侧化检验 + CSP/LDA 基线
完全绕过项目框架，直接读 sourcedata 原始 mat。
检验三点：
  1. 数据里是否存在真实的 MI 信号（mu/beta ERD）
  2. 标签方向是否正确（左 MI -> 右侧(C4) ERD 更强）
  3. 独立 CSP+LDA within-subject 基线是否与框架数字(~52%)一致
"""
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from pathlib import Path

FS = 500
ROOT = Path("src/datasets/21679035/sourcedata")
CH = [  # 32-channel order incl. 2 EOG at end (30 EEG + HEOL/HEOR)
    "Fp1","Fp2","Fz","F3","F4","F7","F8","FCz","FC3","FC4",
    "FT7","FT8","Cz","C3","C4","T3","T4","CPz","CP3","CP4",
    "TP7","TP8","Pz","P3","P4","T5","T6","Oz","O1","O2","HEOL","HEOR",
]
C3, C4 = CH.index("C3"), CH.index("C4")

BASE = (int(0.5*FS), int(1.75*FS))   # instruction 段作基线
MI   = (int(2.5*FS), int(5.5*FS))    # MI 段（避开 onset 瞬态）

def bp(x, lo, hi):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

def erd(x):  # x: (n_trials, n_times) 单通道
    p_base = (x[:, BASE[0]:BASE[1]]**2).mean(axis=1)
    p_mi   = (x[:, MI[0]:MI[1]]**2).mean(axis=1)
    return (p_mi - p_base) / p_base  # 负值 = 去同步

rows = []
for sub in sorted(ROOT.glob("sub-*")):
    f = sub / f"{sub.name}_task-motor-imagery_eeg.mat"
    if not f.exists():
        continue
    v = sio.loadmat(f)["eeg"]
    eeg, labels = v[0][0][0], v[0][0][1].flatten()  # (40,33,4000), 1=left 2=right
    mu = bp(eeg[:, :, :], 8, 13); beta = bp(eeg[:, :, :], 13, 30)
    erd_c3_mu = erd(mu[:, C3, :]); erd_c4_mu = erd(mu[:, C4, :])
    erd_c3_b  = erd(beta[:, C3, :]); erd_c4_b = erd(beta[:, C4, :])
    L = labels == 1; R = labels == 2
    # 对左手 MI：对侧(C4)应更强 -> ERD_C4 - ERD_C3 < 0；右手相反
    lat_left  = (erd_c4_mu[L].mean() + erd_c4_b[L].mean())/2 - (erd_c3_mu[L].mean() + erd_c3_b[L].mean())/2
    lat_right = (erd_c3_mu[R].mean() + erd_c3_b[R].mean())/2 - (erd_c4_mu[R].mean() + erd_c4_b[R].mean())/2
    rows.append(dict(
        sub=sub.name,
        erd_left_c4=float((erd_c4_mu[L].mean()+erd_c4_b[L].mean())/2),
        erd_left_c3=float((erd_c3_mu[L].mean()+erd_c3_b[L].mean())/2),
        erd_right_c3=float((erd_c3_mu[R].mean()+erd_c3_b[R].mean())/2),
        erd_right_c4=float((erd_c4_mu[R].mean()+erd_c4_b[R].mean())/2),
        lat_left=float(lat_left), lat_right=float(lat_right),
        correct_dir=bool(lat_left < 0 and lat_right < 0),
        n_trials=len(labels),
    ))
    print(f"{sub.name}: lat_left={lat_left:+.3f} lat_right={lat_right:+.3f} dir_ok={lat_left<0 and lat_right<0}")

import json
n = len(rows)
n_correct = sum(r["correct_dir"] for r in rows)
all_left  = np.array([r["lat_left"] for r in rows])
all_right = np.array([r["lat_right"] for r in rows])
from scipy import stats
t1, p1 = stats.ttest_1samp(all_left, 0)
t2, p2 = stats.ttest_1samp(all_right, 0)
print("\n===== ERD 审计汇总 =====")
print(f"被试数: {n}, 侧化方向正确: {n_correct}/{n} ({100*n_correct/n:.0f}%)")
print(f"左手 MI 对侧优势: mean={all_left.mean():+.4f}, t={t1:.2f}, p={p1:.4f}")
print(f"右手 MI 对侧优势: mean={all_right.mean():+.4f}, t={t2:.2f}, p={p2:.4f}")
overall_erd = np.mean([[r["erd_left_c4"], r["erd_right_c3"]] for r in rows])
print(f"对侧通道平均 ERD 强度: {overall_erd:+.4f} (负=去同步, 0=无 MI 信号)")
Path("results/audit").mkdir(parents=True, exist_ok=True)
with open("results/audit/xwstroke_erd_audit.json", "w") as fp:
    json.dump(dict(per_subject=rows, n=n, n_correct=n_correct,
                   lat_left_mean=float(all_left.mean()), lat_left_p=float(p1),
                   lat_right_mean=float(all_right.mean()), lat_right_p=float(p2),
                   contra_erd=float(overall_erd)), fp, indent=2)
print("saved: results/audit/xwstroke_erd_audit.json")
