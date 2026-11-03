# -*- coding: utf-8 -*-
"""
Label-blind data-quality metrics for XWStroke 50 subjects (raw 500 Hz 4s epochs).
Metrics per subject (no labels used):
  n_trials, med_std, n_flat, n_noisy, line50_ratio, hf_ratio, kurt_max, amp_max
Run on server from repo root.
"""
import os
import numpy as np
import scipy.io as sio
from scipy import signal as sg
import pandas as pd

XW_DIR = "src/datasets/21679035/sourcedata"
SR = 500

rows = []
for sid in range(1, 51):
    fpath = os.path.join(XW_DIR, f"sub-{sid:02d}", f"sub-{sid:02d}_task-motor-imagery_eeg_4s.mat")
    if not os.path.exists(fpath):
        rows.append(dict(subject=sid)); continue
    mat = sio.loadmat(fpath)
    data = mat["eeg_data_4s"]
    if data.shape[1] > data.shape[2]:
        data = np.transpose(data, (0, 2, 1))
    X = data[:, :30, :].astype(np.float64)  # (trials, ch, time), unit uV presumably

    # per-channel std across all trials/time
    stds = X.reshape(-1, 30, X.shape[2]).std(axis=(0, 2))  # (30,)
    med = np.median(stds)
    n_flat = int((stds < 1.0).sum())
    n_noisy = int((stds > 10 * med).sum()) if med > 0 else 0

    # PSD-based ratios on channel-mean signal
    x = X.mean(axis=1)  # (trials, time) avg across channels
    f, P = sg.welch(x, fs=SR, nperseg=1000, axis=1)
    Pm = P.mean(axis=0)
    def bp(lo, hi):
        m = (f >= lo) & (f <= hi)
        return Pm[m].sum()
    tot = bp(1, 45) + 1e-12
    line50 = bp(49, 51) / tot          # 工频
    hf = bp(38, 45) / tot              # 高频（EMG 代理，避开 50Hz）
    lowband = bp(1, 4) / tot           # 慢波占比（漂移/出汗代理）

    # spike artifacts: kurtosis & max amplitude
    flat = X.reshape(-1)
    from scipy.stats import kurtosis
    kurt = float(kurtosis(flat, fisher=True))
    amp_max = float(np.abs(flat).max())

    rows.append(dict(subject=sid, n_trials=int(X.shape[0]), med_std=float(med),
                     n_flat=n_flat, n_noisy=n_noisy,
                     line50_ratio=float(line50), hf_ratio=float(hf),
                     lowband_ratio=float(lowband),
                     kurt=kurt, amp_max=amp_max))
    print(f"sub-{sid:02d} done", flush=True)

pd.DataFrame(rows).to_csv("results/e3b_quality/xw_quality_metrics.csv", index=False)
print("saved results/e3b_quality/xw_quality_metrics.csv")
