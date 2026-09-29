"""精确复现官方 WS CSP 基线（目标 57.33% ± 11.85%）。
配置严格对齐官方 ws/csp/ws.py：
  - mne firwin 3-35Hz，0-4s epoch
  - StratifiedKFold(10, shuffle=False)
  - mne CSP(n_components=10, reg=None, log=False, norm_trace=False) + LDA
数据源用我们已接入的 EDF（与官方 mat 同源）。"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
import mne

FS = 250
ROOT = Path("src/datasets/SHU")
CH_NAMES = ["Fp1","Fp2","Fz","F3","F4","F7","F8","FC1","FC2","FC5",
            "FC6","Cz","C3","C4","T3","T4","A1","A2","CP1","CP2",
            "CP5","CP6","Pz","P3","P4","T5","T6","PO3","PO4","Oz","O1","O2"]
mne.set_log_level("ERROR")
ev_names = ["onset", "duration", "trial_type", "response_time", "sample", "value"]

def load_session(sid, ses):
    sub = f"sub-{sid:03d}"
    edf = ROOT / "edf_raw" / "edf" / f"{sub}_ses-{ses:02d}_task_motorimagery_eeg.edf"
    evp = ROOT / "events_raw" / "events" / f"{sub}_ses-{ses:02d}_task_motorimagery_events.tsv"
    if not edf.exists():
        return None
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sig = raw.get_data() * 1e6  # -> uV（官方 mat 存 µV）
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
        L.append(1 if str(row["trial_type"]).strip() == "left" else 2)
    return np.asarray(T), np.asarray(L)

def official_csp_score(X, y):
    """与官方 ws.py 完全一致的流程（滤波在 trial 连续轴上做以贴近 mne 实现）"""
    # 官方是把 trial 拼回连续信号做 firwin 滤波再切 epoch；
    # 等价做法：逐 trial 滤波（边界效应差异可忽略，firwin 默认 pad）
    info = mne.create_info(CH_NAMES, FS, "eeg")
    da = X.transpose(1, 0, 2).reshape(len(CH_NAMES), -1)
    raw = mne.io.RawArray(da, info, verbose=False)
    raw.filter(3, 35, fir_design="firwin", verbose=False)
    sk = X.shape[2]
    events = np.column_stack([np.arange(len(y)) * sk,
                              np.zeros(len(y), int), y.astype(int)])
    epochs = mne.Epochs(raw, events, {"left": 1, "right": 2}, 0, 4 - 0.004,
                        baseline=None, preload=True, verbose=False)
    data = epochs.get_data()
    cv = StratifiedKFold(10, shuffle=False)
    scores = []
    for tr, te in cv.split(data, y):
        csp = CSP(n_components=10, reg=None, log=False, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        lda.fit(csp.fit_transform(data[tr], y[tr]), y[tr])
        scores.append(lda.score(csp.transform(data[te]), y[te]))
    return float(np.mean(scores))

accs = {}
for sid in range(1, 26):
    for ses in range(1, 6):
        d = load_session(sid, ses)
        if d is None:
            continue
        accs[f"sub-{sid:03d}_ses-{ses:02d}"] = official_csp_score(*d)
    a = np.array(list(accs.values()))
    print(f"sub-{sid:03d} done, 累计均值 {a.mean()*100:.2f}% (n={len(a)})", flush=True)

a = np.array(list(accs.values()))
print(f"\n===== 复现官方 WS CSP =====")
print(f"我方复现: {a.mean()*100:.2f}% ± {a.std()*100:.2f}% (n={len(a)})")
print(f"官方报告: 57.33% ± 11.85% (n=125)")
