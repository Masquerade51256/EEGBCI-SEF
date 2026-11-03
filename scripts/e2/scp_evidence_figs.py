# -*- coding: utf-8 -*-
"""
SCP signal evidence figures + frontal artifact audit data (XW acute, n=50).
Outputs:
  experiments/e2_hypothesis_families/e2e_slowpot/fig_scp_evidence.png
  experiments/e2_hypothesis_families/e2e_slowpot/scp_evidence_persubj.csv
Run from repo root on server.
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as sg

from scripts.e2.e2e_slowpot import load_xw_trials, XW_SR
from scripts.e2.e2a_spatial import XW_CH_NAMES

OUT = "experiments/e2_hypothesis_families/e2e_slowpot"
os.makedirs(OUT, exist_ok=True)

# SCP-significant vs non-significant (from e2_crossfamily8_subjpmat, binomial p<0.05)
SIG = [2, 3, 4, 8, 13, 17, 20, 22, 30, 32, 34, 35, 40, 42, 45, 50]
NONSIG = [s for s in range(1, 51) if s not in SIG]

def scp_epoch(sid):
    """-> (n_trials, 30, 2000) lowpassed 0-4Hz, baseline-corrected; labels 0=aff,1=unaff"""
    X, y = load_xw_trials(sid)
    b, a = sg.butter(4, 4.0, fs=XW_SR, btype="low")
    X = sg.filtfilt(b, a, X, axis=2)
    base = X[:, :, :250].mean(axis=2, keepdims=True)   # first 0.5 s baseline
    return X - base, y

# collect grand averages
t = np.arange(2000) / XW_SR
grp = {"sig": {"aff": [], "una": []}, "non": {"aff": [], "una": []}}
rows = []
for sid in range(1, 51):
    X, y = scp_epoch(sid)
    ga_aff = X[y == 0].mean(axis=0)   # (30, 2000)
    ga_una = X[y == 1].mean(axis=0)
    g = "sig" if sid in SIG else "non"
    grp[g]["aff"].append(ga_aff)
    grp[g]["una"].append(ga_una)
    # per-subject audit metrics (late window 2-3.5 s)
    w = slice(1000, 1750)
    diff = (ga_aff - ga_una)[:, w].mean(axis=1)      # (30,) topography
    frontal = np.abs(diff[[0, 1]]).mean()            # Fp1/Fp2
    motor = np.abs(diff[[12, 13, 14, 17, 18, 19]]).mean()  # Cz,C3,C4,CPz,CP3,CP4
    rows.append(dict(subject=sid, group=g, frontal_slow_uv=frontal, motor_slow_uv=motor,
                     diffmap=",".join(f"{v:.4f}" for v in diff)))
    print(f"sub-{sid:02d} done", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "scp_evidence_persubj.csv"), index=False)

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
C3 = XW_CH_NAMES.index("C3"); C4 = XW_CH_NAMES.index("C4")
for ax, g, title in [(axes[0], "sig", f"SCP-significant subjects (n={len(SIG)})"),
                     (axes[1], "non", f"non-significant subjects (n={len(NONSIG)})")]:
    aff = np.stack(grp[g]["aff"]); una = np.stack(grp[g]["una"])
    for arr, lab, col in [(aff, "affected-hand MI", "#c1443c"), (una, "unaffected-hand MI", "#3a7ca5")]:
        m = arr[:, C3, :].mean(axis=0)
        se = arr[:, C3, :].std(axis=0) / np.sqrt(len(arr))
        ax.plot(t, m, color=col, lw=1.8, label=lab)
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.25)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title(f"{title}\nchannel C3 (contralesional motor)", fontsize=10)
    ax.set_xlabel("Time in trial (s)"); ax.set_ylabel("Slow potential (uV, baseline-corr.)")
    ax.legend(frameon=False, fontsize=8)

# Panel C: audit scatter frontal vs motor slow amplitude
dff = pd.DataFrame(rows)
ax = axes[2]
for g, col, lab in [("sig", "#5a8f5a", "SCP-significant"), ("non", "#999999", "non-significant")]:
    d = dff[dff.group == g]
    ax.scatter(d.frontal_slow_uv, d.motor_slow_uv, c=col, s=40, label=lab, alpha=0.85)
ax.set_xlabel("Frontal (Fp1/Fp2) slow-wave |diff| (uV)")
ax.set_ylabel("Motor-area slow-wave |diff| (uV)")
ax.set_title("Artifact audit: frontal vs motor slow activity", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.suptitle("XW acute stroke: slow-potential evidence (0-4 Hz, grand average)", y=1.02)
fig.savefig(os.path.join(OUT, "fig_scp_evidence.png"), bbox_inches="tight", dpi=150)
print("saved fig_scp_evidence.png")
