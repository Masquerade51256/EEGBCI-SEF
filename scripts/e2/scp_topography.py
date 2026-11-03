# -*- coding: utf-8 -*-
"""Topography of within-subject slow-wave effect size |d| (late window), sig vs non-sig groups.

Design note (2026-11-03, v2): signed uV maps were outlier-dominated (mean) or
polarity-cancelled (median). The honest spatial quantity is the per-subject
unsigned within-subject effect size |d| (mean aff-una / pooled trial SD,
2.0-3.5 s window), computed by scp_evidence_figs.py into the `dmap` column.
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

OUT = "experiments/e2_hypothesis_families/e2e_slowpot"
df = pd.read_csv(os.path.join(OUT, "scp_evidence_persubj.csv"))

XW_CH_NAMES = ["Fp1","Fp2","Fz","F3","F4","F7","F8","FCz","FC3","FC4",
               "FT7","FT8","Cz","C3","C4","T3","T4","CPz","CP3","CP4",
               "TP7","TP8","Pz","P3","P4","T5","T6","Oz","O1","O2"]
ren = {"T3":"T7","T4":"T8","T5":"P7","T6":"P8"}
names = [ren.get(c, c) for c in XW_CH_NAMES]
info = mne.create_info(names, sfreq=250, ch_types="eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1020"))

dmaps = np.stack([np.fromstring(s, sep=",") for s in df["dmap"]])
absd = np.abs(dmaps)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
med = {g: np.median(absd[df.group.values == g], axis=0) for g in ("sig", "non")}
vlim = float(np.concatenate(list(med.values())).max())
for ax, g, ttl in [(axes[0], "sig", f"SCP-significant (n={int((df.group=='sig').sum())})"),
                   (axes[1], "non", f"non-significant (n={int((df.group=='non').sum())})")]:
    im, _ = mne.viz.plot_topomap(med[g], info, axes=ax, cmap="Reds", vlim=(0, vlim),
                                 show=False, contours=4)
    ax.set_title(ttl, fontsize=11)
fig.colorbar(im, ax=axes, label="Within-subject effect size |d|", shrink=0.8)
fig.suptitle("XW acute stroke: scalp topography of MI slow-wave effect size", y=1.0)
fig.savefig(os.path.join(OUT, "fig_scp_topography.png"), bbox_inches="tight", dpi=150)
print("saved fig_scp_topography.png")

# group-level audit stats
from scipy import stats
sig = df.group == "sig"
print("frontal |diff|: sig", df.loc[sig,"frontal_slow_uv"].median(), "vs non", df.loc[~sig,"frontal_slow_uv"].median(),
      "MWU p =", stats.mannwhitneyu(df.loc[sig,"frontal_slow_uv"], df.loc[~sig,"frontal_slow_uv"]).pvalue)
print("motor |diff|:  sig", df.loc[sig,"motor_slow_uv"].median(), "vs non", df.loc[~sig,"motor_slow_uv"].median(),
      "MWU p =", stats.mannwhitneyu(df.loc[sig,"motor_slow_uv"], df.loc[~sig,"motor_slow_uv"]).pvalue)
# motor/frontal ratio: decodable subjects should be motor-dominated
df["mf_ratio"] = df.motor_slow_uv / (df.frontal_slow_uv + 1e-9)
print("motor/frontal ratio: sig", df.loc[sig,"mf_ratio"].median(), "vs non", df.loc[~sig,"mf_ratio"].median(),
      "MWU p =", stats.mannwhitneyu(df.loc[sig,"mf_ratio"], df.loc[~sig,"mf_ratio"]).pvalue)
# effect-size audit: is frontal |d| also elevated in sig group, or is it motor-specific?
print("C3 |d| late: sig", df.loc[sig,"c3_absd_late"].median(), "vs non", df.loc[~sig,"c3_absd_late"].median(),
      "MWU p =", stats.mannwhitneyu(df.loc[sig,"c3_absd_late"], df.loc[~sig,"c3_absd_late"]).pvalue)
df.to_csv(os.path.join(OUT, "scp_evidence_persubj.csv"), index=False)
