"""汇总三数据集逐被试侧化效应量（Cohen's d）到 JSON，供本地绘图。"""
import json
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy import stats
from pathlib import Path

def bp(x, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

def ai_d(eeg, labels, c3, c4, fs, mi_win):
    f = bp(eeg, 8, 30, fs)[:, :, mi_win[0]:mi_win[1]]
    p = (f**2).mean(axis=-1)
    ai = (p[:, c3] - p[:, c4]) / (p[:, c3] + p[:, c4])
    L, R = labels == 1, labels == 2
    return float((ai[L].mean() - ai[R].mean()) / ai.std())

out = {}

# XWStroke (50 急性卒中)
ROOT = Path("src/datasets/21679035/sourcedata")
xw = {}
for sub in sorted(ROOT.glob("sub-*")):
    f = sub / f"{sub.name}_task-motor-imagery_eeg.mat"
    if not f.exists():
        continue
    v = sio.loadmat(f)["eeg"]
    xw[sub.name] = ai_d(v[0][0][0][:, :32, :], v[0][0][1].flatten(), 13, 14, 500,
                        (int(2.5*500), int(5.5*500)))
out["XWStroke_acute_stroke"] = xw
print("XWStroke done", len(xw))

# HS Stroke (14 慢性卒中, ses-01)
CH = ["FC3","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP4"]
hs = {}
for sub in sorted(Path("src/datasets/HSStroke").glob("sub-*")):
    mats = sorted(sub.glob("ses-01/eeg/*.mat"))
    if not mats:
        continue
    m = sio.loadmat(mats[0], squeeze_me=True)
    hs[sub.name] = ai_d(m["EEGdata"], m["EEGdatalabel"].flatten(),
                        CH.index("C3"), CH.index("C4"), 256, (int(0.5*256), int(3.5*256)))
out["HSStroke_chronic_stroke"] = hs
print("HSStroke done", len(hs))

# BCICIV2a (9 健康人)
import mne
mne.set_log_level("ERROR")
bc = {}
for sid in range(1, 10):
    raw = mne.io.read_raw_gdf(f"src/datasets/BCICIV2a/A{sid:02d}T.gdf", preload=True, verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    ev_id = {"left": 7, "right": 8} if sid != 4 else {"left": 5, "right": 6}
    ep = mne.Epochs(raw, events, ev_id, tmin=0, tmax=4, baseline=None,
                    picks=[7, 11], preload=True, verbose=False)
    labels = np.where(ep.events[:, -1] == ev_id["left"], 1, 2)
    bc[f"A{sid:02d}"] = ai_d(ep.get_data(), labels, 0, 1, 250, (int(1.0*250), int(3.5*250)))
out["BCICIV2a_healthy"] = bc
print("BCICIV2a done", len(bc))

Path("results/audit").mkdir(parents=True, exist_ok=True)
with open("results/audit/lateralization_d_all.json", "w") as fp:
    json.dump(out, fp, indent=2)
print("saved results/audit/lateralization_d_all.json")
