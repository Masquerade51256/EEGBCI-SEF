"""SHU 全量 25 人 CSP+LDA within-session 基线 × 3 种时间窗（后台批跑版）。
结果写 results/audit/shu_csp_lda_windows.json"""
import json
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

def load(edf, evp, win):
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sig = raw.get_data()
    ev = pd.read_csv(evp, sep="\t")
    if "sample" not in ev.columns:
        ev = pd.read_csv(evp, sep="\t", header=None, names=ev_names)
    T, L = [], []
    for _, row in ev.iterrows():
        s = int(row["sample"]) + win[0]
        seg = sig[:, s:s + (win[1] - win[0])]
        if seg.shape[1] < win[1] - win[0]:
            continue
        T.append(seg)
        L.append(0 if str(row["trial_type"]).strip() == "left" else 1)
    return np.asarray(T), np.asarray(L)

results = {}
for name, win in [("full_4s", (0, 1000)), ("1-3s", (250, 750)), ("1.5-3.5s", (375, 875))]:
    accs = {}
    for sid in range(1, 26):
        for ses in range(1, 6):
            edf = ROOT / "edf_raw" / "edf" / f"sub-{sid:03d}_ses-{ses:02d}_task_motorimagery_eeg.edf"
            evp = ROOT / "events_raw" / "events" / f"sub-{sid:03d}_ses-{ses:02d}_task_motorimagery_events.tsv"
            if not edf.exists():
                continue
            X, y = load(edf, evp, win)
            X = bp(X).astype(np.float64)
            fa = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
                clf = Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                                ("lda", LinearDiscriminantAnalysis())])
                clf.fit(X[tr], y[tr])
                fa.append(clf.score(X[te], y[te]))
            accs[f"sub-{sid:03d}_ses-{ses:02d}"] = float(np.mean(fa))
        print(f"[{name}] sub-{sid:03d} done", flush=True)
    a = np.array(list(accs.values()))
    results[name] = dict(per_session=accs, mean=float(a.mean()), std=float(a.std()),
                         frac_gt60=float(np.mean(a > 0.6)))
    print(f"== {name}: {a.mean()*100:.2f}% ± {a.std()*100:.2f}% ==", flush=True)

Path("results/audit").mkdir(parents=True, exist_ok=True)
with open("results/audit/shu_csp_lda_windows.json", "w") as fp:
    json.dump(results, fp, indent=2)
print("saved results/audit/shu_csp_lda_windows.json")
