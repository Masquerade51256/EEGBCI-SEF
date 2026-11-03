# -*- coding: utf-8 -*-
"""Healthy (BCIC) SCP polarity check: is the left-vs-right MI slow-wave polarity
consistent across subjects at C3/C4? Contrast with XW aligned C4 polarity spread.
Run from repo root on server."""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import scipy.signal as sg

from scripts.e2.e2e_slowpot import load_bcic_trials

C3, C4 = 7, 11   # BCIC 22-ch order: FC3=1, FC4=5, C3=7, Cz=9, C4=11
OUT = "experiments/e2_hypothesis_families/e2e_slowpot"

rows = []
for sid in range(1, 10):
    X, y = load_bcic_trials(sid)          # (n, 22, 750) uV, 0=left MI, 1=right MI
    b, a = sg.butter(4, 4.0, fs=250, btype="low")
    X = sg.filtfilt(b, a, X, axis=2)
    X = X - X[:, :, :62].mean(axis=2, keepdims=True)   # first 0.25 s post-cue baseline
    w = slice(375, 625)                   # 1.5-2.5 s late window
    L, R = X[y == 0], X[y == 1]
    rec = dict(subject=sid, n_left=len(L), n_right=len(R))
    for ch, nm in [(C3, "C3"), (C4, "C4")]:
        la = L[:, ch, w].mean(axis=1)
        ra = R[:, ch, w].mean(axis=1)
        # d = (left - right) / pooled trial SD; textbook: d(C4)<0, d(C3)>0
        rec[f"d_{nm}"] = (la.mean() - ra.mean()) / np.sqrt((la.var(ddof=1) + ra.var(ddof=1)) / 2)
    rows.append(rec)
    print(f"A{sid:02d} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "bcic_scp_polarity.csv"), index=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
print("\ntextbook consistency: d_C4<0 in", int((df.d_C4 < 0).sum()), "/9;",
      " d_C3>0 in", int((df.d_C3 > 0).sum()), "/9")
