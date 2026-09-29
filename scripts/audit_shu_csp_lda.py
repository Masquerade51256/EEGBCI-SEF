"""SHU 数据质量独立验证：CSP+LDA within-session 基线（绕过框架）。
官方对照：within-session 五算法平均 60-69%（FBCNet 最高 68.8%，CSP 最低）。
取前 5 名被试的全部 5 个 session 快速验证。"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
import mne

FS = 250
ROOT = Path("src/datasets/SHU")
mne.set_log_level("ERROR")

def bp(x, lo=8, hi=30):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

ev_names = ["onset", "duration", "trial_type", "response_time", "sample", "value"]
session_accs = []
for sid in range(1, 6):
    sub = f"sub-{sid:03d}"
    for ses in range(1, 6):
        edf = ROOT / "edf_raw" / "edf" / f"{sub}_ses-{ses:02d}_task_motorimagery_eeg.edf"
        evp = ROOT / "events_raw" / "events" / f"{sub}_ses-{ses:02d}_task_motorimagery_events.tsv"
        if not edf.exists():
            continue
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
        sig = raw.get_data()
        ev = pd.read_csv(evp, sep="\t")
        if "sample" not in ev.columns:
            ev = pd.read_csv(evp, sep="\t", header=None, names=ev_names)
        trials, labels = [], []
        for _, row in ev.iterrows():
            s = int(row["sample"])
            seg = sig[:, s:s+1000]
            if seg.shape[1] < 1000:
                continue
            trials.append(seg)
            labels.append(0 if str(row["trial_type"]).strip() == "left" else 1)
        X = bp(np.asarray(trials)).astype(np.float64)
        y = np.asarray(labels)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fa = []
        for tr, te in skf.split(X, y):
            clf = Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                            ("lda", LinearDiscriminantAnalysis())])
            clf.fit(X[tr], y[tr])
            fa.append(clf.score(X[te], y[te]))
        session_accs.append(np.mean(fa))
        print(f"{sub} ses-{ses:02d}: {session_accs[-1]*100:.1f}% (n={len(y)})")

a = np.array(session_accs)
print(f"\n===== SHU within-session CSP+LDA（{len(a)} sessions, sub-001~005）=====")
print(f"mean = {a.mean()*100:.2f}% ± {a.std()*100:.2f}%")
print("参考：官方 within-session CSP 最低、FBCNet 最高 68.8% ± 14.6%")
