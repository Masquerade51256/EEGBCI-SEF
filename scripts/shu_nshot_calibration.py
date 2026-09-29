"""SHU N-shot 校准曲线实验（Q1：新用户最少需要多少校准数据）

两条协议曲线：
  A) 跨被试（LOSO 群体模型）：24 人训练 → 目标被试 session k，
     用目标 session 的 N 个 trial/类做校准（与源数据混合重训），测试其余 trial。
     对应"新患者用群体模型 + 少量校准"。
  B) 跨 session（个人模型）：目标被试 session 1 训练 → 同被试 session k，
     同样 N-shot 校准。
     对应"老用户隔几天再用，需要多少重新校准"。

分类器：8-30Hz + CSP(6) + LDA（CPU 快速，先拿曲线形状；EEGNet 版后续）。
N ∈ {0, 1, 2, 4, 8, 16, 32}（每类 trial 数），5 个采样种子。
输出：results/audit/shu_nshot_curve.json
"""
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
N_LIST = [0, 1, 2, 4, 8, 16, 32]
SEEDS = [0, 1, 2, 3, 4]
mne.set_log_level("ERROR")

def bp(x, lo=8, hi=30):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

ev_names = ["onset", "duration", "trial_type", "response_time", "sample", "value"]

def load_subject(sid):
    """返回 dict: ses_idx -> (X (n,32,1000) float64 filtered, y)"""
    out = {}
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
        T, L = [], []
        for _, row in ev.iterrows():
            s = int(row["sample"])
            seg = sig[:, s:s + 1000]
            if seg.shape[1] < 1000:
                continue
            T.append(seg)
            L.append(0 if str(row["trial_type"]).strip() == "left" else 1)
        out[ses] = (bp(np.asarray(T)).astype(np.float64), np.asarray(L))
    return out

def make_clf():
    return Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                     ("lda", LinearDiscriminantAnalysis())])

def nshot_eval(X_src, y_src, X_tgt, y_tgt, n, seed):
    """源数据 + 目标 session 的 N/类 trial 校准 -> 测试其余。返回 acc 或 None（样本不足）"""
    rng = np.random.RandomState(seed)
    idx0 = np.where(y_tgt == 0)[0]
    idx1 = np.where(y_tgt == 1)[0]
    if n > 0:
        take0 = rng.choice(idx0, size=min(n, len(idx0)), replace=False)
        take1 = rng.choice(idx1, size=min(n, len(idx1)), replace=False)
        if len(take0) < n or len(take1) < n:
            return None
        calib = np.concatenate([take0, take1])
    else:
        calib = np.array([], dtype=int)
    test_mask = np.ones(len(y_tgt), bool)
    test_mask[calib] = False
    X_tr = np.concatenate([X_src, X_tgt[calib]]) if len(calib) else X_src
    y_tr = np.concatenate([y_src, y_tgt[calib]]) if len(calib) else y_src
    clf = make_clf()
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_tgt[test_mask], y_tgt[test_mask]))

print("加载全部被试数据...", flush=True)
DATA = {sid: load_subject(sid) for sid in range(1, 26)}
DATA = {k: v for k, v in DATA.items() if v}
print(f"加载完成: {len(DATA)} 被试", flush=True)

results = {"A_cross_subject": {}, "B_cross_session": {}}

# ---- 协议 A：跨被试 LOSO + N-shot ----
for tgt_sid in sorted(DATA):
    X_src = np.concatenate([DATA[s][ses][0] for s in DATA if s != tgt_sid for ses in DATA[s]])
    y_src = np.concatenate([DATA[s][ses][1] for s in DATA if s != tgt_sid for ses in DATA[s]])
    for ses in DATA[tgt_sid]:
        X_t, y_t = DATA[tgt_sid][ses]
        for n in N_LIST:
            accs = [nshot_eval(X_src, y_src, X_t, y_t, n, seed) for seed in SEEDS]
            accs = [a for a in accs if a is not None]
            key = f"sub-{tgt_sid:03d}_ses-{ses:02d}"
            results["A_cross_subject"][key] = results["A_cross_subject"].get(key, {})
            results["A_cross_subject"][key][str(n)] = accs
    print(f"[A] sub-{tgt_sid:03d} done", flush=True)

# ---- 协议 B：跨 session（session 1 为源）+ N-shot ----
for tgt_sid in sorted(DATA):
    if 1 not in DATA[tgt_sid]:
        continue
    X_src, y_src = DATA[tgt_sid][1]
    for ses in [2, 3, 4, 5]:
        if ses not in DATA[tgt_sid]:
            continue
        X_t, y_t = DATA[tgt_sid][ses]
        for n in N_LIST:
            accs = [nshot_eval(X_src, y_src, X_t, y_t, n, seed) for seed in SEEDS]
            accs = [a for a in accs if a is not None]
            key = f"sub-{tgt_sid:03d}_ses-{ses:02d}"
            results["B_cross_session"][key] = results["B_cross_session"].get(key, {})
            results["B_cross_session"][key][str(n)] = accs
    print(f"[B] sub-{tgt_sid:03d} done", flush=True)

Path("results/audit").mkdir(parents=True, exist_ok=True)
with open("results/audit/shu_nshot_curve.json", "w") as fp:
    json.dump(results, fp)
print("saved results/audit/shu_nshot_curve.json", flush=True)

# 快速汇总
for proto in results:
    print(f"== {proto} ==", flush=True)
    for n in N_LIST:
        vals = [a for key in results[proto] for a in results[proto][key].get(str(n), [])]
        if vals:
            print(f"  N={n:>2}: {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}% (m={len(vals)})", flush=True)
