# -*- coding: utf-8 -*-
"""Topography of affected-minus-unaffected slow-wave difference (late window), sig vs non-sig groups."""
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
mont = mne.channels.make_standard_montage("standard_1020")
pos = np.array([mont.get_positions()["ch_pos"][n][:2] for n in names])
# normalize to head circle
pos = pos / np.max(np.linalg.norm(pos, axis=1)) * 0.45

diffmaps = np.stack([np.fromstring(s, sep=",") for s in df["diffmap"]])
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
vlim = np.percentile(np.abs(diffmaps), 95)
for ax, g, ttl in [(axes[0], "sig", f"SCP-significant (n={int((df.group=='sig').sum())})"),
                   (axes[1], "non", f"non-significant (n={int((df.group=='non').sum())})")]:
    m = diffmaps[df.group.values == g].mean(axis=0)
    im, _ = mne.viz.plot_topomap(m, pos, axes=ax, cmap="RdBu_r", vlim=(-vlim, vlim),
                                 show=False, contours=4)
    ax.set_title(f"{title}\naffected - unaffected, 2.0-3.5 s", fontsize=10)
fig.colorbar(im, ax=axes, label="Slow-potential difference (uV)", shrink=0.8)
fig.suptitle("XW acute stroke: scalp topography of MI slow-wave difference", y=1.0)
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
df.to_csv(os.path.join(OUT, "scp_evidence_persubj.csv"), index=False)
