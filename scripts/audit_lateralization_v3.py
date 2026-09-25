"""审计 v3：用健康人 BCICIV2a 作真阳性对照（mne 读 GDF），并重打印 XWStroke 组级汇总"""
import numpy as np
from scipy.signal import butter, filtfilt
from scipy import stats
import scipy.io as sio
from pathlib import Path

def bp(x, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

def ai_by_label(eeg, labels, c3, c4, fs, mi_win, lo=8, hi=30):
    f = bp(eeg, lo, hi, fs)[:, :, mi_win[0]:mi_win[1]]
    p = (f**2).mean(axis=-1)
    ai = (p[:, c3] - p[:, c4]) / (p[:, c3] + p[:, c4])
    L, R = labels == 1, labels == 2
    t, pv = stats.ttest_ind(ai[L], ai[R])
    d = (ai[L].mean() - ai[R].mean()) / ai.std()
    return float(ai[L].mean()), float(ai[R].mean()), float(pv), float(d)

# ---------- BCICIV2a 健康人阳性对照 ----------
import mne
BC_DIR = Path("src/datasets/BCICIV2a")
# 22 通道顺序: Fz,FC3,FC1,FCz,FC2,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP1,CPz,CP2,CP4,P1,Pz,P2,POz
C3_BC, C4_BC = 7, 11
ds = []
for sid in range(1, 10):
    gdf = BC_DIR / f"A{sid:02d}T.gdf"
    mat = BC_DIR / f"A{sid:02d}T.mat"
    raw = mne.io.read_raw_gdf(str(gdf), preload=True, verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    # train 文件: 769=left, 770=right (subject 4 偏移 -2)
    ev_id = {"left": 7, "right": 8} if sid != 4 else {"left": 5, "right": 6}
    epochs = mne.Epochs(raw, events, ev_id, tmin=0, tmax=4, baseline=None,
                        picks=[C3_BC, C4_BC], preload=True, verbose=False)
    x = epochs.get_data()  # (n, 2, 1001) @250Hz, 单位 V
    lab = epochs.events[:, -1]
    labels = np.where(lab == ev_id["left"], 1, 2)
    a = ai_by_label(x, labels, 0, 1, 250, (int(1.0*250), int(3.5*250)))  # cue 后 1-3.5s
    ds.append(a[3])
    print(f"A0{sid}: AI_left={a[0]:+.4f} AI_right={a[1]:+.4f} p={a[2]:.4f} d={a[3]:+.3f}")

ds = np.array(ds)
t, p = stats.ttest_1samp(ds, 0)
print(f"\nBCICIV2a(健康人) 组级: d mean={ds.mean():+.3f}, t={t:.2f}, p={p:.6f}, d>0 比例={np.mean(ds>0)*100:.0f}%")

# ---------- XWStroke 组级重打印 ----------
ROOT = Path("src/datasets/21679035/sourcedata")
xw = []
for sub in sorted(ROOT.glob("sub-*")):
    f = sub / f"{sub.name}_task-motor-imagery_eeg.mat"
    if not f.exists():
        continue
    v = sio.loadmat(f)["eeg"]
    eeg, labels = v[0][0][0][:, :32, :], v[0][0][1].flatten()
    xw.append(ai_by_label(eeg, labels, 13, 14, 500, (int(2.5*500), int(5.5*500)))[3])
xw = np.array(xw)
t, p = stats.ttest_1samp(xw, 0)
print(f"XWStroke(急性卒中) 组级: d mean={xw.mean():+.3f}, t={t:.2f}, p={p:.4f}, d>0 比例={np.mean(xw>0)*100:.0f}%")
