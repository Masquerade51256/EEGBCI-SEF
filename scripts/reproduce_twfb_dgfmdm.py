#!/usr/bin/env python3
"""
Reproduce Liu et al. (Sci Data 2024) TWFB+DGFMDRM within-subject benchmark.

Paper claims 72.21% via "optimal time-window + filter-bank selection +
Discriminative Geodesic Filtering + Minimum Distance to Riemannian Mean",
evaluated with per-subject 10-fold CV. Critically, the paper does NOT state
whether the time-window/filter-bank selection was done inside the training
folds or on all data. We test both:

  - leaky  : select (window, band) on ALL 40 trials, then report 10-fold CV
             accuracy of that combo (reproduces the paper's protocol if their
             selection leaked). Expected ~72% if the paper number is inflated
             by selection leakage.
  - nested : outer 10-fold; within each train fold, select (window, band) by an
             inner 80/20 split; evaluate on the held-out fold. This is the
             honest estimate of TWFB+DGFMDRM.

  - csp_lda: paper's own baseline (55.57%) as an implementation sanity check.

Method mapping: DGFMDRM == pyriemann FgMDM (Fisher-geodesic MDM). Paper grid:
7 time windows (0-1s ... 3-4s, 1s win / 0.5s hop) x 19 bands (4Hz width, 1Hz
hop over 8-30Hz). Data: _4s.mat (40 trials x 30 EEG x 2000 @500Hz).

Outputs (results/twfb_reproduction/):
  - twfb_results.csv, report.md
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps
from pyriemann.estimation import Covariances
from pyriemann.classification import FgMDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, train_test_split
from mne.decoding import CSP

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "datasets" / "21679035" / "sourcedata"
OUT_DIR = ROOT / "results" / "twfb_reproduction"

TIME_WINDOWS = [(i * 0.5, i * 0.5 + 1.0) for i in range(7)]  # 0-1 ... 3-4 s
BANDS = [(8 + i, 12 + i) for i in range(19)]                  # 4Hz width, 1Hz hop
FS = 500
N_EEG_CH = 30
SEED = 42


def load_subject(sid):
    f = DATA_DIR / f"sub-{sid:02d}" / f"sub-{sid:02d}_task-motor-imagery_eeg_4s.mat"
    m = sio.loadmat(f)
    X = np.asarray(m["eeg_data_4s"], dtype=float)[:, :N_EEG_CH, :]  # (40,30,2000)
    y = np.asarray(m["labels"]).ravel()
    if set(np.unique(y).tolist()) == {1, 2}:
        y = y - 1
    return X, y


def bandpass(X, lo, hi, fs=FS, order=4):
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, X, axis=-1)


def covs_for(X, band, tw):
    Xf = bandpass(X, *band)
    s0, s1 = int(tw[0] * FS), int(tw[1] * FS)
    return Covariances(estimator="oas").fit_transform(Xf[:, :, s0:s1])


def precompute_covs(X):
    """Band-pass once per band (19), slice time windows, cache all 133 cov sets.

    Band-passing and covariance estimation are unsupervised (no labels), so
    computing them on all trials does not leak label information; FgMDM class
    means are always fitted on train indices only.
    """
    cov_est = Covariances(estimator="oas")
    cache = {}
    for band in BANDS:
        Xf = bandpass(X, *band)
        for tw in TIME_WINDOWS:
            s0, s1 = int(tw[0] * FS), int(tw[1] * FS)
            cache[(band, tw)] = cov_est.fit_transform(Xf[:, :, s0:s1])
    return cache


def combo_cv_acc(covs, y, n_splits=10):
    """10-fold CV accuracy of FgMDM on precomputed covariances."""
    accs = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(covs, y):
        clf = FgMDM()
        clf.fit(covs[tr], y[tr])
        accs.append(clf.score(covs[te], y[te]))
    return float(np.mean(accs))


def eval_leaky(X, y, cache=None):
    """Select best (band, tw) on all trials, report its 10-fold CV acc."""
    cache = cache or precompute_covs(X)
    best = {"acc": -1}
    for key, c in cache.items():
        acc = combo_cv_acc(c, y)
        if acc > best["acc"]:
            best = {"acc": acc, "band": key[0], "tw": key[1]}
    return best


def eval_nested(X, y, cache=None):
    """Outer 10-fold; select (band, tw) by inner 80/20 split on train only."""
    cache = cache or precompute_covs(X)
    accs = []
    outer = StratifiedKFold(10, shuffle=True, random_state=SEED)
    for tr, te in outer.split(X, y):
        ytr = y[tr]
        itr, ival = train_test_split(np.arange(len(tr)), test_size=0.25,
                                     random_state=SEED, stratify=ytr)
        best, best_score = None, -1
        for key, c in cache.items():
            c_tr = c[tr]
            clf = FgMDM().fit(c_tr[itr], ytr[itr])
            sc = clf.score(c_tr[ival], ytr[ival])
            if sc > best_score:
                best_score, best = sc, key
        c = cache[best]
        clf = FgMDM().fit(c[tr], ytr)
        accs.append(clf.score(c[te], y[te]))
    return float(np.mean(accs))


def eval_csp_lda(X, y):
    accs = []
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y):
        Xf = bandpass(X, 8, 30)
        csp = CSP(n_components=4, log=True)
        lda = LinearDiscriminantAnalysis()
        csp.fit(Xf[tr], y[tr])
        lda.fit(csp.transform(Xf[tr]), y[tr])
        accs.append(lda.score(csp.transform(Xf[te]), y[te]))
    return float(np.mean(accs))


def process_subject(sid):
    t0 = time.time()
    try:
        X, y = load_subject(sid)
    except FileNotFoundError:
        return None
    acc_csp = eval_csp_lda(X, y)
    cache = precompute_covs(X)
    leaky = eval_leaky(X, y, cache)
    acc_nested = eval_nested(X, y, cache)
    row = {"subject_id": sid, "csp_lda": acc_csp,
           "twfb_leaky": leaky["acc"], "leaky_band": str(leaky["band"]),
           "leaky_tw": str(leaky["tw"]), "twfb_nested": acc_nested}
    print(f"[sub{sid:02d}] csp={acc_csp*100:.1f}  "
          f"leaky={leaky['acc']*100:.1f}  nested={acc_nested*100:.1f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_subject)(sid) for sid in range(1, 51))
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "twfb_results.csv", index=False)

    print("\n== Summary ==")
    print(f"CSP+LDA      : {df['csp_lda'].mean()*100:.2f}%  (paper: 55.57%)")
    print(f"TWFB leaky   : {df['twfb_leaky'].mean()*100:.2f}%  (paper TWFB: 72.21%)")
    print(f"TWFB nested  : {df['twfb_nested'].mean()*100:.2f}%  (honest estimate)")

    lines = [
        "# TWFB+DGFMDRM 复现（pyriemann FgMDM 等价）\n",
        "| 协议 | 准确率 | 论文对应值 |",
        "|---|---|---|",
        f"| CSP+LDA 10-fold | {df['csp_lda'].mean()*100:.2f}% | 55.57% |",
        f"| TWFB 时窗频带泄漏版 | {df['twfb_leaky'].mean()*100:.2f}% | 72.21% |",
        f"| TWFB 嵌套选择（严格） | {df['twfb_nested'].mean()*100:.2f}% | — |",
        "",
        "- 泄漏版：在全部 40 trials 上选最优（时窗，频带），再报该组合的 10-fold CV 准确率——若接近论文 72.21%，说明论文数字含特征选择泄漏。",
        "- 嵌套版：外层 10-fold，时窗/频带选择仅用 train fold 内 80/20 划分——TWFB+DGFMDRM 的诚实估计。",
        "- DGFMDRM 用 pyriemann FgMDM 等价实现；时窗 7 个（1s 窗 0.5s 步长），频带 19 个（4Hz 宽 1Hz 步长），同论文 Table 3。",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
