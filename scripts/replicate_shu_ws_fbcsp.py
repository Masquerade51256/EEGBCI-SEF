"""精确复现官方 WS FBCSP 基线（目标 64.33% ± 14.22%）。
配置对齐官方 ws/fbcsp 工具箱：
  - Chebyshev II 带通，9 个 4Hz 子带 [4-8,...,36-40]，lfilter（因果）
  - 时间窗 0.5-4s
  - 每子带 mne CSP(n_components=8)（首4+尾4分量）→ log 方差特征，9×8=72 维
  - SVR(gamma='auto') 按类回归 + argmax（二分类）
  - StratifiedKFold(10, shuffle=False)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from mne.decoding import CSP
import mne

FS = 250
ROOT = Path("src/datasets/SHU")
mne.set_log_level("ERROR")
ev_names = ["onset", "duration", "trial_type", "response_time", "sample", "value"]

def get_filter_coeff():
    nyq = FS / 2
    coeff = {}
    for i, f_low in enumerate(np.arange(4, 40, 4)):
        f_pass = np.array([f_low, f_low + 4])
        f_stop = np.array([f_low - 2, f_low + 6])
        order, wn = signal.cheb2ord(f_pass / nyq, f_stop / nyq, 3, 30)
        b, a = signal.cheby2(order, 30, wn, btype="bandpass")
        coeff[i] = (b, a)
    return coeff

COEFF = get_filter_coeff()

def load_session(sid, ses):
    sub = f"sub-{sid:03d}"
    edf = ROOT / "edf_raw" / "edf" / f"{sub}_ses-{ses:02d}_task_motorimagery_eeg.edf"
    evp = ROOT / "events_raw" / "events" / f"{sub}_ses-{ses:02d}_task_motorimagery_events.tsv"
    if not edf.exists():
        return None
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sig = raw.get_data() * 1e6
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
    return np.asarray(T), np.asarray(L)

def fbcsp_features(X, csp_per_band=None, y=None):
    """X: (n,32,1000)。训练时返回 (features, csp_list)，测试时只传 features。"""
    t0, t1 = int(0.5 * FS), int(4 * FS)
    feats = []
    csps = []
    for i, (b, a) in COEFF.items():
        Xf = np.asarray([signal.lfilter(b, a, t) for t in X])[:, :, t0:t1]
        if csp_per_band is None:
            csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
            F = csp.fit_transform(Xf, y)
            csps.append(csp)
        else:
            F = csp_per_band[i].transform(Xf)
        feats.append(F)
    F = np.concatenate(feats, axis=1)
    return (F, csps) if csp_per_band is None else F

def fbcsp_score(X, y):
    cv = StratifiedKFold(10, shuffle=False)
    scores = []
    for tr, te in cv.split(X, y):
        Ftr, csps = fbcsp_features(X[tr], y=y[tr])
        Fte = fbcsp_features(X[te], csp_per_band=csps)
        # 二分类：对每个类做 one-vs-rest SVR，取预测值较大者
        preds = []
        for cls in [0, 1]:
            y_bin = (y[tr] != cls).astype(float)  # 与官方一致：本类 0 他类 1
            svr = SVR(gamma="auto")
            svr.fit(Ftr, y_bin)
            preds.append(svr.predict(Fte))
        y_pred = np.argmin(np.vstack(preds).T, axis=1)  # 本类回归目标为 0，取小
        scores.append(np.mean(y_pred == y[te]))
    return float(np.mean(scores))

accs = {}
for sid in range(1, 26):
    for ses in range(1, 6):
        d = load_session(sid, ses)
        if d is None:
            continue
        accs[f"sub-{sid:03d}_ses-{ses:02d}"] = fbcsp_score(*d)
    a = np.array(list(accs.values()))
    print(f"sub-{sid:03d} done, 累计均值 {a.mean()*100:.2f}% (n={len(a)})", flush=True)

a = np.array(list(accs.values()))
print(f"\n===== 复现官方 WS FBCSP =====")
print(f"我方复现: {a.mean()*100:.2f}% ± {a.std()*100:.2f}% (n={len(a)})")
print(f"官方报告: 64.33% ± 14.22% (n=125)")
