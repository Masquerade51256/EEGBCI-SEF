"""SHU N-shot 校准曲线 v2（修正语义：校准数据必须参与 CSP 空间滤波器估计）

v1 的教训：CSP 滤波器仅在源数据上拟合时无法跨 session/被试迁移（全 50%），
校准 trial 必须参与 CSP 的协方差估计（混合重训）。

协议：
  A) 跨被试 LOSO：24 人（每折抽样 2000 trial 控制成本）+ N/类校准 → 混合重训 CSP+LDA
  B) 跨 session：目标被试 session 1 + N/类校准 → 混合重训 CSP+LDA

N ∈ {0, 1, 2, 4, 8, 16, 32}（每类），种子：A=3，B=5。
输出：results/audit/shu_nshot_curve_v2.json
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
SEEDS_A, SEEDS_B = [0, 1, 2], [0, 1, 2, 3, 4]
SRC_SUBSAMPLE = 1000          # 源子抽样（协方差估计已足够）
A_SESSIONS = [2, 5]           # 协议 A 只评近/远两个目标 session
TIME_DECIMATE = 4             # 时间轴抽稀 1000->250 点，CSP 协方差计算加速 4 倍
mne.set_log_level("ERROR")

def bp(x, lo=8, hi=30):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

ev_names = ["onset", "duration", "trial_type", "response_time", "sample", "value"]

def load_subject(sid):
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
        out[ses] = (bp(np.asarray(T))[:, :, ::TIME_DECIMATE].astype(np.float64), np.asarray(L))
    return out

def make_clf():
    return Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                     ("lda", LinearDiscriminantAnalysis())])

def split_calib(y_tgt, n, rng):
    idx0 = np.where(y_tgt == 0)[0]
    idx1 = np.where(y_tgt == 1)[0]
    if n == 0:
        return np.array([], dtype=int)
    if len(idx0) < n or len(idx1) < n:
        return None
    return np.concatenate([rng.choice(idx0, n, replace=False),
                           rng.choice(idx1, n, replace=False)])

def nshot_mixed(X_src, y_src, X_tgt, y_tgt, n, seed):
    """混合重训 CSP+LDA：源 + 目标 N/类校准 trial，测试目标其余 trial。"""
    rng = np.random.RandomState(seed)
    calib = split_calib(y_tgt, n, rng)
    if calib is None:
        return None
    test_mask = np.ones(len(y_tgt), bool)
    test_mask[calib] = False
    if len(calib):
        X_tr = np.concatenate([X_src, X_tgt[calib]])
        y_tr = np.concatenate([y_src, y_tgt[calib]])
    else:
        X_tr, y_tr = X_src, y_src
    clf = make_clf()
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_tgt[test_mask], y_tgt[test_mask]))

print("加载全部被试数据...", flush=True)
DATA = {sid: load_subject(sid) for sid in range(1, 26)}
DATA = {k: v for k, v in DATA.items() if v}
print(f"加载完成: {len(DATA)} 被试", flush=True)

results = {"A_cross_subject": {}, "B_cross_session": {}}

# ---- 协议 A：跨被试 LOSO + N-shot（源子抽样 2000 控制成本）----
for tgt_sid in sorted(DATA):
    X_src_all = np.concatenate([DATA[s][ses][0] for s in DATA if s != tgt_sid for ses in DATA[s]])
    y_src_all = np.concatenate([DATA[s][ses][1] for s in DATA if s != tgt_sid for ses in DATA[s]])
    rng_sub = np.random.RandomState(1000 + tgt_sid)
    pick = rng_sub.choice(len(y_src_all), size=min(SRC_SUBSAMPLE, len(y_src_all)), replace=False)
    X_src, y_src = X_src_all[pick], y_src_all[pick]
    for ses in A_SESSIONS:
        if ses not in DATA[tgt_sid]:
            continue
        X_t, y_t = DATA[tgt_sid][ses]
        for n in N_LIST:
            accs = [nshot_mixed(X_src, y_src, X_t, y_t, n, seed) for seed in SEEDS_A]
            accs = [a for a in accs if a is not None]
            key = f"sub-{tgt_sid:03d}_ses-{ses:02d}"
            results["A_cross_subject"].setdefault(key, {})[str(n)] = accs
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
            accs = [nshot_mixed(X_src, y_src, X_t, y_t, n, seed) for seed in SEEDS_B]
            accs = [a for a in accs if a is not None]
            key = f"sub-{tgt_sid:03d}_ses-{ses:02d}"
            results["B_cross_session"].setdefault(key, {})[str(n)] = accs
    print(f"[B] sub-{tgt_sid:03d} done", flush=True)

Path("results/audit").mkdir(parents=True, exist_ok=True)
with open("results/audit/shu_nshot_curve_v2.json", "w") as fp:
    json.dump(results, fp)
print("saved results/audit/shu_nshot_curve_v2.json", flush=True)

for proto in results:
    print(f"== {proto} ==", flush=True)
    for n in N_LIST:
        vals = [a for key in results[proto] for a in results[proto][key].get(str(n), [])]
        if vals:
            print(f"  N={n:>2}: {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}% (m={len(vals)})", flush=True)
