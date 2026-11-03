# -*- coding: utf-8 -*-
"""
SCP signal evidence figures + frontal artifact audit data (XW acute, n=50).

Design note (2026-11-03, v2):
  Group MEAN waveforms proved outlier-dominated (2-3 subjects with |diff| up to
  ~100-160 uV painted fake clean separation in the non-significant group).
  Group MEDIAN waveforms suffer polarity cancellation (sign of the C3 effect
  varies across subjects). Decoding is within-subject, so the honest evidence
  quantity is the within-subject effect size d = mean(aff-una)/pooled trial SD.
  Panels:
    A: 2x2 example SCP-significant subjects (within-subject mean +/- SEM at C3)
    B: group median |d|(t) at C3, sig vs non-sig (unsigned, amplitude-free)
    C: artifact audit scatter (frontal vs motor slow |diff|)
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
C3 = XW_CH_NAMES.index("C3")
W = slice(1000, 1750)  # 2.0-3.5 s late window
EPS = 1e-12

def scp_epoch(sid):
    """-> (n_trials, 30, 2000) lowpassed 0-4Hz, baseline-corrected; labels 0=aff,1=unaff"""
    X, y = load_xw_trials(sid)
    b, a = sg.butter(4, 4.0, fs=XW_SR, btype="low")
    X = sg.filtfilt(b, a, X, axis=2)
    base = X[:, :, :250].mean(axis=2, keepdims=True)   # first 0.5 s baseline
    return X - base, y

t = np.arange(2000) / XW_SR
rows, store = [], {}
for sid in range(1, 51):
    X, y = scp_epoch(sid)
    aff, una = X[y == 0], X[y == 1]
    m_aff, m_una = aff.mean(0), una.mean(0)                       # (30, 2000)
    sd_pool = np.sqrt((aff.var(0, ddof=1) + una.var(0, ddof=1)) / 2)
    d_wave = (m_aff - m_una) / (sd_pool + EPS)                    # signed effect-size wave
    # trial-level late-window means -> per-channel d map
    tw_aff, tw_una = aff[:, :, W].mean(axis=2), una[:, :, W].mean(axis=2)
    d_map = (tw_aff.mean(0) - tw_una.mean(0)) / (
        np.sqrt((tw_aff.var(0, ddof=1) + tw_una.var(0, ddof=1)) / 2) + EPS)
    diff = (m_aff - m_una)[:, W].mean(axis=1)                     # (30,) signed uV topo
    frontal = np.abs(diff[[0, 1]]).mean()                         # Fp1/Fp2
    motor = np.abs(diff[[12, 13, 14, 17, 18, 19]]).mean()         # Cz,C3,C4,CPz,CP3,CP4
    g = "sig" if sid in SIG else "non"
    store[sid] = dict(m_aff=m_aff, m_una=m_una, sd_pool=sd_pool,
                      d_wave=d_wave, d_map=d_map,
                      sem_aff=aff.std(0, ddof=1) / np.sqrt(len(aff)),
                      sem_una=una.std(0, ddof=1) / np.sqrt(len(una)))
    rows.append(dict(subject=sid, group=g, frontal_slow_uv=frontal, motor_slow_uv=motor,
                     c3_d_late=float(d_map[C3]), c3_absd_late=float(abs(d_map[C3])),
                     diffmap=",".join(f"{v:.4f}" for v in diff),
                     dmap=",".join(f"{v:.4f}" for v in d_map)))
    print(f"sub-{sid:02d} done", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "scp_evidence_persubj.csv"), index=False)

# ---------- figure ----------
fig = plt.figure(figsize=(16, 5.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.28, hspace=0.55)

# Panel A: 2x2 example sig subjects (largest within-subject |d| at C3)
ex = sorted(SIG, key=lambda s: -abs(store[s]["d_map"][C3]))[:4]
sub = gs[:, 0].subgridspec(2, 2, hspace=0.6, wspace=0.35)
for i, sid in enumerate(ex):
    ax = fig.add_subplot(sub[i // 2, i % 2])
    s = store[sid]
    ax.plot(t, s["m_aff"][C3], color="#c1443c", lw=1.5, label="affected-hand MI")
    ax.fill_between(t, s["m_aff"][C3] - s["sem_aff"][C3], s["m_aff"][C3] + s["sem_aff"][C3],
                    color="#c1443c", alpha=0.25)
    ax.plot(t, s["m_una"][C3], color="#3a7ca5", lw=1.5, label="unaffected-hand MI")
    ax.fill_between(t, s["m_una"][C3] - s["sem_una"][C3], s["m_una"][C3] + s["sem_una"][C3],
                    color="#3a7ca5", alpha=0.25)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0.5, color="gray", ls=":", lw=0.7)
    ax.set_title(f"sub-{sid:02d}  (C3, within-subj d={s['d_map'][C3]:+.2f})", fontsize=9)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.text(0.06, 0.02, "Panel A: example SCP-significant subjects (top-4 by within-subject |d| at C3); "
                     "band = +/- SEM across trials", fontsize=9)

# Panel B: group median of channel-mean |d|(t) (distributed effect, matches multivariate decoding)
axB = fig.add_subplot(gs[:, 1])
for g, col, lab in [("sig", "#5a8f5a", f"SCP-significant (n={len(SIG)})"),
                    ("non", "#999999", f"non-significant (n={len(NONSIG)})")]:
    sids = SIG if g == "sig" else NONSIG
    absd = np.stack([np.abs(store[s]["d_wave"]).mean(axis=0) for s in sids])  # mean over 30 ch
    med = np.median(absd, axis=0)
    q25, q75 = np.percentile(absd, 25, axis=0), np.percentile(absd, 75, axis=0)
    axB.plot(t, med, color=col, lw=1.8, label=lab)
    axB.fill_between(t, q25, q75, color=col, alpha=0.22)
axB.axvline(0.5, color="gray", ls=":", lw=0.8)
axB.set_xlabel("Time in trial (s)"); axB.set_ylabel("|d| (affected vs unaffected)")
axB.set_title("Panel B: within-subject effect size,\nmean over 30 channels (group median, IQR)", fontsize=10)
axB.legend(frameon=False, fontsize=8)

# Panel C: audit scatter frontal vs motor slow amplitude
dff = pd.DataFrame(rows)
axC = fig.add_subplot(gs[:, 2])
for g, col, lab in [("sig", "#5a8f5a", "SCP-significant"), ("non", "#999999", "non-significant")]:
    d = dff[dff.group == g]
    axC.scatter(d.frontal_slow_uv, d.motor_slow_uv, c=col, s=40, label=lab, alpha=0.85)
axC.set_xlabel("Frontal (Fp1/Fp2) slow-wave |diff| (uV)")
axC.set_ylabel("Motor-area slow-wave |diff| (uV)")
axC.set_title("Panel C: artifact audit\nfrontal vs motor slow activity", fontsize=10)
axC.legend(frameon=False, fontsize=8)

fig.suptitle("XW acute stroke: slow-potential evidence (0-4 Hz)", y=1.0)
fig.savefig(os.path.join(OUT, "fig_scp_evidence.png"), bbox_inches="tight", dpi=150)
print("saved fig_scp_evidence.png")
