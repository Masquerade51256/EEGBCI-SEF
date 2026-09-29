"""N-shot 补臂 C：仅用校准数据训练（源数据不参与），与混合重训对照。
回答"源数据在 N-shot 中到底是资产还是负担"。
协议同 B（跨 session，ses-1 为历史，ses-2~5 为目标），但模型只用目标 session 的
N/类 trial 训练 CSP+LDA。输出 results/audit/shu_nshot_calib_only.json"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
import mne

FS = 250
ROOT = Path("src/datasets/SHU")
N_LIST = [1, 2, 4, 8, 16, 32]
SEEDS = [0, 1, 2, 3, 4]
mne.set_log_level("ERROR")

def bp(x, lo=8, hi=30):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

ev_names = ["onset", "duration", "trial_type", "response_time", "sample", "value"]

def load_session(sid, ses):
    sub = f"sub-{sid:03d}"
    edf = ROOT / "edf_raw" / "edf" / f"{sub}_ses-{ses:02d}_task_motorimagery_eeg.edf"
    evp = ROOT / "events_raw" / "events" / f"{sub}_ses-{ses:02d}_task_motorimagery_events.tsv"
    if not edf.exists():
        return None
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sig = raw.get_data()
    ev = pd.read_csv(evp, sep="\t")
    if "sample" not in ev.columns:
        ev = pd.read_csv(evp, sep="\t", header=None, names=ev_names)
    T, L = [], []
    for _, row in ev.iterrows():
        s = int(row["sample"])
        seg = sig[:, s:s + 1000]
        if seg.shape[1] < 1000:
            continue
        T.append(seg)
        L.append(0 if str(row["trial_type"]).strip() == "left" else 1)
    X = bp(np.asarray(T))[:, :, ::4].astype(np.float64)
    return X, np.asarray(L)

results = {}
for sid in range(1, 26):
    for ses in [2, 3, 4, 5]:
        d = load_session(sid, ses)
        if d is None:
            continue
        X, y = d
        for n in N_LIST:
            accs = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
                if len(i0) < n or len(i1) < n:
                    continue
                calib = np.concatenate([rng.choice(i0, n, replace=False),
                                        rng.choice(i1, n, replace=False)])
                mask = np.ones(len(y), bool)
                mask[calib] = False
                clf = Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                                ("lda", LinearDiscriminantAnalysis())])
                clf.fit(X[calib], y[calib])
                accs.append(float(clf.score(X[mask], y[mask])))
            results.setdefault(f"sub-{sid:03d}_ses-{ses:02d}", {})[str(n)] = accs
    print(f"sub-{sid:03d} done", flush=True)

Path("results/audit").mkdir(parents=True, exist_ok=True)
with open("results/audit/shu_nshot_calib_only.json", "w") as fp:
    json.dump(results, fp)
print("== C_calib_only ==", flush=True)
for n in N_LIST:
    vals = [a for k in results for a in results[k].get(str(n), [])]
    if vals:
        print(f"  N={n:>2}: {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}% (m={len(vals)})", flush=True)
print("saved results/audit/shu_nshot_calib_only.json")
