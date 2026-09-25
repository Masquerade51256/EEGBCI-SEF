"""
审计 v2：无基线的侧化检验 + 阳性对照
- 不看相对基线的 ERD（避免视频刺激引起的宽带功率上升污染），
  直接比较 MI 期内 C3/C4 不对称性与标签的对应关系。
- 同一套代码跑 HS Stroke（慢性期，官方验证过信号质量）作阳性对照：
  若对照正常而 XWStroke 不正常 -> 数据集问题坐实；
  若两者都不正常 -> 是我的分析代码有问题。
"""
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from pathlib import Path
from scipy import stats

def bp(x, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

def asymmetry_report(name, eeg, labels, c3, c4, fs, mi_win):
    """eeg: (n_trials, n_ch, n_times); labels: 1=left, 2=right"""
    out = {}
    for band, (lo, hi) in dict(mu=(8,13), beta=(13,30), mubeta=(8,30)).items():
        f = bp(eeg, lo, hi, fs)[:, :, mi_win[0]:mi_win[1]]
        p = (f**2).mean(axis=-1)  # (n_trials, n_ch)
        ai = (p[:, c3] - p[:, c4]) / (p[:, c3] + p[:, c4])  # >0 表示 C4 功率更低(对侧去同步于左手)
        L, R = labels == 1, labels == 2
        t, pv = stats.ttest_ind(ai[L], ai[R])
        d = (ai[L].mean() - ai[R].mean()) / ai.std()
        out[band] = dict(ai_left=float(ai[L].mean()), ai_right=float(ai[R].mean()),
                         p=float(pv), cohen_d=float(d))
    mb = out["mubeta"]
    print(f"[{name}] mu/beta 不对称性: 左手标签 AI={mb['ai_left']:+.4f}, 右手标签 AI={mb['ai_right']:+.4f}, "
          f"t检验 p={mb['p']:.4f}, d={mb['cohen_d']:+.3f} | mu p={out['mu']['p']:.4f} beta p={out['beta']['p']:.4f}")
    return out

FS_XW = 500
ROOT = Path("src/datasets/21679035/sourcedata")
C3, C4 = 13, 14  # XWStroke 通道序
MI_XW = (int(2.5*FS_XW), int(5.5*FS_XW))

print("===== XWStroke（逐被试 MI 期 C3/C4 不对称性 × 标签）=====")
xw_results = []
for sub in sorted(ROOT.glob("sub-*")):
    f = sub / f"{sub.name}_task-motor-imagery_eeg.mat"
    if not f.exists():
        continue
    v = sio.loadmat(f)["eeg"]
    eeg, labels = v[0][0][0][:, :32, :], v[0][0][1].flatten()
    r = asymmetry_report(sub.name, eeg, labels, C3, C4, FS_XW, MI_XW)
    xw_results.append(r["mubeta"]["cohen_d"])

xw_d = np.array(xw_results)
t, p = stats.ttest_1samp(xw_d, 0)
print(f"\nXWStroke 组级: d mean={xw_d.mean():+.3f}, t={t:.2f}, p={p:.4f}, "
      f"d>0 比例={np.mean(xw_d>0)*100:.0f}% (若标签正确, 应显著>0)")

print("\n===== HS Stroke 阳性对照（同一套分析代码）=====")
FS_HS = 256
HS_ROOT = Path("src/datasets/HSStroke")
CH_HS = ["FC3","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP4"]
hs_d = []
for sub in sorted(HS_ROOT.glob("sub-*")):
    mats = sorted(sub.glob("ses-01/eeg/*.mat"))  # 只用 ses-01 快速对照
    if not mats:
        continue
    m = sio.loadmat(mats[0], squeeze_me=True)
    eeg, labels = m["EEGdata"], m["EEGdatalabel"].flatten()
    r = asymmetry_report(sub.name, eeg, labels, CH_HS.index("C3"), CH_HS.index("C4"),
                         FS_HS, (int(0.5*FS_HS), int(3.5*FS_HS)))
    hs_d.append(r["mubeta"]["cohen_d"])

hs_d = np.array(hs_d)
t, p = stats.ttest_1samp(hs_d, 0)
print(f"\nHS Stroke 组级: d mean={hs_d.mean():+.3f}, t={t:.2f}, p={p:.4f}, d>0 比例={np.mean(hs_d>0)*100:.0f}%")
