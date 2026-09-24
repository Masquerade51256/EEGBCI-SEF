#!/usr/bin/env python3
"""
P0 analysis: subject-level decodability & transferability profiling on XWStroke.

Core idea: place each of the 50 subjects in a 2-D space:
  - X: within-subject decodability (does the subject's own EEG carry MI information?)
  - Y: cross-subject transferability (LOSO accuracy when this subject is held out)
The transfer gap (within - LOSO) isolates subject-specific variance that current
pipelines fail to transfer, and EEG signal features are tested as predictors of
within-acc / LOSO-acc / gap.

Inputs (all already exist in the repo):
  - LOSO per-subject accuracies: experiments/<exp>/results/results.json
  - Within-subject 5-fold CV:    results/adfcnn_edf_within_subject_kfold_summary/
                                 adfcnn_edf_within_subject_kfold.csv
  - EEG signal features:         results/eeg_signal_analysis/subject_feature_table.csv
  - Clinical metadata:           results/stratified/stratified_analysis_detailed.csv

Outputs (results/decodability_transferability/):
  - master_table.csv          per-subject merged table
  - correlations.csv          Spearman r / p / BH-FDR for every feature x target
  - fig_quadrant_scatter.png  within x LOSO quadrant map
  - fig_feature_correlations.png
  - report.md
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "decodability_transferability"

# --- EEGNet LOSO pipelines (AGENTS.md sec.7) -------------------------------
LOSO_EXPERIMENTS = {
    "loso_lr_noea": "XWStroke_Full50_EEGNet_LR_noEA_noAug",
    "loso_lr_ea": "XWStroke_Full50_EEGNet_LR_EA",
    "loso_aff_noea": "XWStroke_Full50_EEGNet_AffAlign_noEA_noAug",
    "loso_aff_ea": "XWStroke_Full50_EEGNet_EA_AffAlign_NoMP",
}
# ADFCNN references (robustness check only)
LOSO_REFERENCE_EXPERIMENTS = {
    "loso_adfcnn_lr": "XWStroke_Full50_LR",
    "loso_adfcnn_affalign": "XWStroke_Full50_LOSO_Affected_Aligned",
}

WITHIN_CSV = ROOT / "results" / "adfcnn_edf_within_subject_kfold_summary" / "adfcnn_edf_within_subject_kfold.csv"
# Clean-protocol within-subject accuracies (ADFCNN, fixed 100 epochs, GroupKFold,
# no best-epoch peeking) produced by analyze_within_attribution.py.
CLEAN_WITHIN_NPZ = ROOT / "results" / "within_attribution" / "attribution_arrays.npz"
FEATURES_CSV = ROOT / "results" / "eeg_signal_analysis" / "subject_feature_table.csv"
CLINICAL_CSV = ROOT / "results" / "stratified" / "stratified_analysis_detailed.csv"

# Quadrant thresholds
WITHIN_THRESHOLD = 0.55   # above-chance within-subject decodability
LOSO_THRESHOLD = 0.50     # chance-level binary LOSO

# Physically implausible magnitudes (division-by-near-zero artifacts) -> NaN
ABSURD_VALUE_CUTOFF = 1e4

FEATURE_EXCLUDE = {"subject_id", "ParalysisSide"}
# Degenerate features: the 4s data are z-score normalized, which removes
# baseline-vs-task power differences, so these two carry no real variance
# (all subjects ~= 1.0002). The current analyze_subject_eeg_differences.py
# already sets them to NaN; the CSV values come from an older version.
INVALID_FEATURES = {"snr_task_baseline", "snr_task_baseline_cv"}
CLINICAL_COLS = ["ParalysisSide", "StrokeLocation_Category", "Duration_Category", "NIHSS"]


# ---------------------------------------------------------------------------
def load_loso_per_subject(exp_name: str) -> pd.Series:
    """Return per-subject test_acc from a StreamingLOSO results.json."""
    path = ROOT / "experiments" / exp_name / "results" / "results.json"
    if not path.exists():
        print(f"  [warn] missing {path}, skipped")
        return pd.Series(dtype=float)
    with open(path) as f:
        data = json.load(f)
    rows = {s["test_subject_id"]: s["test_acc"] for s in data["subjects"]}
    return pd.Series(rows).sort_index()


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted p-values (no statsmodels dependency)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def spearman_with_n(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 8:
        return np.nan, np.nan, n
    r, p = stats.spearmanr(x[mask], y[mask])
    return r, p, n


# ---------------------------------------------------------------------------
def build_master_table() -> pd.DataFrame:
    print("== Loading LOSO per-subject accuracies ==")
    loso_cols = {}
    for col, exp in {**LOSO_EXPERIMENTS, **LOSO_REFERENCE_EXPERIMENTS}.items():
        s = load_loso_per_subject(exp)
        if len(s) > 0:
            loso_cols[col] = s
            print(f"  {col:24s} n={len(s)}  mean={s.mean() * 100:.2f}%")
    df = pd.DataFrame(loso_cols)
    df.index.name = "subject_id"

    eegnet_cols = [c for c in LOSO_EXPERIMENTS if c in df.columns]
    df["loso_mean"] = df[eegnet_cols].mean(axis=1)
    df["loso_best"] = df[eegnet_cols].max(axis=1)

    within = pd.read_csv(WITHIN_CSV).set_index("subject_id")["mean_acc"]
    df["within_ref_acc"] = within  # best-epoch-on-test protocol (optimistic)
    if CLEAN_WITHIN_NPZ.exists():
        z = np.load(CLEAN_WITHIN_NPZ)
        clean = pd.Series(z["test_acc"], index=range(1, len(z["test_acc"]) + 1))
        df["within_acc"] = clean  # primary: fixed-epoch clean protocol
        df.attrs["within_source"] = "clean_fixed_epoch (ADFCNN, GroupKFold)"
        print(f"  within: clean protocol (mean={clean.mean()*100:.2f}%), "
              f"reference protocol kept as within_ref_acc")
    else:
        df["within_acc"] = within
        df.attrs["within_source"] = "reference best-epoch protocol"
        print("  [warn] clean within npz not found; using reference protocol")
    df["gap"] = df["within_acc"] - df["loso_mean"]

    feats = pd.read_csv(FEATURES_CSV)
    feats = feats.set_index("subject_id")
    feats = feats.drop(columns=[c for c in feats.columns
                                if c in FEATURE_EXCLUDE | INVALID_FEATURES],
                       errors="ignore")
    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.mask(feats.abs() > ABSURD_VALUE_CUTOFF)  # kill division artifacts
    df = df.join(feats)

    clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig").set_index("subject_id")
    df = df.join(clin[CLINICAL_COLS])
    return df


def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    print("\n== Spearman correlations (feature x target) ==")
    feature_cols = [c for c in df.columns
                    if c not in set(LOSO_EXPERIMENTS) | set(LOSO_REFERENCE_EXPERIMENTS)
                    | {"loso_mean", "loso_best", "within_acc", "within_ref_acc", "gap"}
                    | set(CLINICAL_COLS)]
    targets = ["within_acc", "loso_mean", "gap"]
    rows = []
    for tgt in targets:
        for feat in feature_cols:
            r, p, n = spearman_with_n(df[feat], df[tgt])
            rows.append({"target": tgt, "feature": feat, "r": r, "p": p, "n": n})
    cor = pd.DataFrame(rows)
    # BH-FDR within each target family
    cor["p_fdr"] = np.nan
    for tgt in targets:
        mask = (cor["target"] == tgt) & cor["p"].notna()
        cor.loc[mask, "p_fdr"] = benjamini_hochberg(cor.loc[mask, "p"].values)
    return cor


def assign_quadrants(df: pd.DataFrame) -> pd.Series:
    def q(row):
        hi_within = row["within_acc"] >= WITHIN_THRESHOLD
        hi_loso = row["loso_mean"] >= LOSO_THRESHOLD
        if hi_within and hi_loso:
            return "A: good signal, transfers"
        if hi_within and not hi_loso:
            return "B: good signal, transfer fails"
        if not hi_within and not hi_loso:
            return "C: undecodable signal"
        return "D: suspicious (check data)"
    return df.apply(q, axis=1)


# ---------------------------------------------------------------------------
def analyze_quadrant_clinical(df_out: pd.DataFrame) -> dict:
    """Compare clinical composition: quadrant A (transfers) vs B (transfer fails).

    Tests whether the transfer-failure group is clinically identifiable --
    if yes, lesion/duration/NIHSS become candidate stratification variables.
    """
    ab = df_out[df_out["quadrant"].str.startswith(("A", "B"))].copy()
    ab["grp"] = ab["quadrant"].str[0]
    res = {"n_A": int((ab["grp"] == "A").sum()), "n_B": int((ab["grp"] == "B").sum())}

    for col in ["StrokeLocation_Category", "Duration_Category", "ParalysisSide"]:
        ct = pd.crosstab(ab["grp"], ab[col].fillna("Unknown"))
        if ct.shape == (2, 2) and ct.min().min() >= 1:
            _, p = stats.fisher_exact(ct)
            test = "Fisher exact"
        else:
            _, p, _, _ = stats.chi2_contingency(ct)
            test = "chi2"
        res[col] = {"table": ct, "p": float(p), "test": test}

    a = ab.loc[ab["grp"] == "A", "NIHSS"].dropna()
    b = ab.loc[ab["grp"] == "B", "NIHSS"].dropna()
    _, p = stats.mannwhitneyu(a, b)
    res["NIHSS"] = {"A_median": float(a.median()), "B_median": float(b.median()),
                    "p": float(p), "test": "Mann-Whitney"}

    for col in ["within_acc", "gap"]:
        a = ab.loc[ab["grp"] == "A", col]
        b = ab.loc[ab["grp"] == "B", col]
        _, p = stats.mannwhitneyu(a, b)
        res[col] = {"A_mean": float(a.mean()), "B_mean": float(b.mean()),
                    "p": float(p), "test": "Mann-Whitney"}
    return res


def plot_quadrant_clinical(df_out: pd.DataFrame):
    """Stacked bar: lesion-location composition of quadrants A vs B."""
    ab = df_out[df_out["quadrant"].str.startswith(("A", "B"))].copy()
    ab["grp"] = ab["quadrant"].str[0]
    ct = pd.crosstab(ab["grp"], ab["StrokeLocation_Category"].fillna("Unknown"),
                     normalize="index")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = {"Subcortical": "#1f77b4", "Brainstem": "#2ca02c",
               "Cortical": "#d62728", "Mixed": "#9467bd", "Unknown": "#7f7f7f"}
    bottom = np.zeros(len(ct))
    for cat in ct.columns:
        ax.bar(ct.index, ct[cat] * 100, bottom=bottom * 100,
               color=palette.get(cat, "#7f7f7f"), label=cat, edgecolor="k", lw=0.4)
        bottom += ct[cat].values
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Lesion-location composition:\nA (transfers) vs B (transfer fails)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_quadrant_clinical.png", dpi=160)
    plt.close(fig)


def plot_quadrant(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cats = df["StrokeLocation_Category"].fillna("Unknown")
    palette = {"Subcortical": "#1f77b4", "Brainstem": "#2ca02c",
               "Cortical": "#d62728", "Mixed": "#9467bd", "Unknown": "#7f7f7f"}
    for cat, sub in df.groupby(cats):
        ax.scatter(sub["within_acc"] * 100, sub["loso_mean"] * 100,
                   s=40 + 12 * sub["NIHSS"].fillna(4),
                   c=palette.get(str(cat), "#7f7f7f"), alpha=0.75,
                   edgecolors="k", linewidths=0.4, label=str(cat))
    ax.axvline(WITHIN_THRESHOLD * 100, color="gray", ls="--", lw=1)
    ax.axhline(LOSO_THRESHOLD * 100, color="gray", ls="--", lw=1)
    lims = [40, 80]
    ax.plot(lims, lims, color="lightgray", ls=":", lw=1)
    ax.set_xlabel("Within-subject 5-fold accuracy (%)  [signal decodability]")
    ax.set_ylabel("Mean LOSO accuracy across 4 EEGNet pipelines (%)")
    ax.set_title("XWStroke subject map: decodability vs transferability\n"
                 "(marker size ~ NIHSS, color = lesion location)")
    ax.legend(title="Lesion", fontsize=9)
    ax.text(76, 44, "B: transfer fails", fontsize=10, color="#8B0000", ha="right")
    ax.text(46, 44, "C: undecodable", fontsize=10, color="#555555")
    ax.text(76, 74, "A: good & transfers", fontsize=10, color="#1a5c1a", ha="right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_quadrant_scatter.png", dpi=160)
    plt.close(fig)


def plot_correlations(cor: pd.DataFrame):
    targets = ["within_acc", "loso_mean", "gap"]
    labels = {"within_acc": "within-subject acc", "loso_mean": "mean LOSO acc", "gap": "transfer gap"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, tgt in zip(axes, targets):
        sub = cor[cor["target"] == tgt].dropna(subset=["r"]).sort_values("r")
        colors = ["#d62728" if (p < 0.05) else "#1f77b4" for p in sub["p_fdr"]]
        ax.barh(sub["feature"], sub["r"], color=colors, alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"vs {labels[tgt]}")
        ax.set_xlabel("Spearman r")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("EEG signal feature correlations (red = FDR<0.05)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_feature_correlations.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_report(df: pd.DataFrame, cor: pd.DataFrame, clin_res: dict):
    quadrants = assign_quadrants(df)
    df_out = df.copy()
    df_out["quadrant"] = quadrants

    lines = []
    lines.append("# P0: XWStroke 被试级可解码性 × 可迁移性图谱\n")
    lines.append(f"- 被试数: {len(df)}")
    lines.append(f"- 阈值: within ≥ {WITHIN_THRESHOLD*100:.0f}% / LOSO ≥ {LOSO_THRESHOLD*100:.0f}%")
    lines.append(f"- **within 数据源: {df.attrs.get('within_source', '?')}**")
    lines.append("  - `within_acc`（主分析）：干净协议——ADFCNN，固定 100 epochs，5-fold GroupKFold（按 trial 分组），无 best-epoch 挑选")
    lines.append("  - `within_ref_acc`（对照）：历史参考协议——best-epoch-on-test（含测试集窥视，系统性偏高 ~5–8 点）")
    lines.append("  - LOSO `loso_mean` 为干净协议（源域选 epoch，目标患者只评一次）\n")

    lines.append("## 总体描述\n")
    lines.append("| 指标 | mean | std | min | max |")
    lines.append("|---|---|---|---|---|")
    for col, name in [("within_acc", "Within-subject（干净协议）"),
                      ("within_ref_acc", "Within-subject（历史 best-epoch 协议）"),
                      ("loso_mean", "LOSO mean (4 EEGNet pipelines)"),
                      ("loso_best", "LOSO best-of-4 (per-subject oracle)"),
                      ("gap", "Transfer gap (within − LOSO)")]:
        s = df[col]
        lines.append(f"| {name} | {s.mean()*100:.2f}% | {s.std()*100:.2f}% | "
                     f"{s.min()*100:.2f}% | {s.max()*100:.2f}% |")

    lines.append("\n## 四象限分型\n")
    lines.append("| 象限 | 含义 | 人数 | 被试 |")
    lines.append("|---|---|---|---|")
    desc = {
        "A: good signal, transfers": "信号好且可迁移",
        "B: good signal, transfer fails": "信号好但迁移失败（新模型的目标人群）",
        "C: undecodable signal": "信号本身不可解码（生理上界）",
        "D: suspicious (check data)": "低 within / 高 LOSO，需核查",
    }
    for qname in ["A: good signal, transfers", "B: good signal, transfer fails",
                  "C: undecodable signal", "D: suspicious (check data)"]:
        subs = df_out[df_out["quadrant"] == qname].index.tolist()
        lines.append(f"| {qname.split(':')[0]} | {desc[qname]} | {len(subs)} | {subs} |")

    # old-protocol quadrant (within_ref) vs new-protocol quadrant comparison
    if "within_ref_acc" in df.columns:
        def qref(row):
            hi_w = row["within_ref_acc"] >= WITHIN_THRESHOLD
            hi_l = row["loso_mean"] >= LOSO_THRESHOLD
            return ("A" if hi_w and hi_l else "B" if hi_w and not hi_l
                    else "C" if not hi_w and not hi_l else "D")
        ref_q = df.apply(qref, axis=1)
        new_q = quadrants.str[0]
        ct = pd.crosstab(ref_q, new_q, rownames=["ref_protocol"], colnames=["clean_protocol"])
        n_moved = int((ref_q.values != new_q.values).sum())
        lines.append(f"\n### 新旧协议分型对照（{n_moved}/50 被试换象限）\n")
        lines.append("| ref \\ clean | " + " | ".join(str(c) for c in ct.columns) + " |")
        lines.append("|" + "---|" * (len(ct.columns) + 1))
        for grp, row in ct.iterrows():
            lines.append(f"| {grp} | " + " | ".join(str(int(v)) for v in row.values) + " |")

    # optimistic ceiling: undecodable subjects stuck at chance, the rest reach within-acc
    ceiling = np.where(df["within_acc"] < WITHIN_THRESHOLD, 0.5, df["within_acc"]).mean()
    lines.append(f"\n## 乐观生理上界估计\n")
    lines.append(f"- 假设 C 象限（within < {WITHIN_THRESHOLD*100:.0f}%）患者 LOSO 恒为 50%（随机），"
                 f"其余患者完美迁移到各自 within 水平：")
    lines.append(f"- **LOSO 乐观上界 ≈ {ceiling*100:.2f}%**"
                 f"（对比：当前最佳固定 pipeline 51.32%，per-subject oracle 55.68%）\n")

    lines.append("\n## A vs B 象限临床构成对比\n")
    lines.append(f"- A 组 {clin_res['n_A']} 人，B 组 {clin_res['n_B']} 人\n")
    lines.append("| 变量 | 检验 | p |")
    lines.append("|---|---|---|")
    for col in ["StrokeLocation_Category", "Duration_Category", "ParalysisSide"]:
        r = clin_res[col]
        lines.append(f"| {col} | {r['test']} | {r['p']:.4f} |")
    r = clin_res["NIHSS"]
    lines.append(f"| NIHSS (A中位 {r['A_median']:.0f} vs B中位 {r['B_median']:.0f}) | {r['test']} | {r['p']:.4f} |")
    r = clin_res["within_acc"]
    lines.append(f"| within_acc (A {r['A_mean']*100:.1f}% vs B {r['B_mean']*100:.1f}%) | {r['test']} | {r['p']:.4f} |")
    r = clin_res["gap"]
    lines.append(f"| gap (A {r['A_mean']*100:.1f}% vs B {r['B_mean']*100:.1f}%) | {r['test']} | {r['p']:.4f} |")
    lines.append("")
    for col in ["StrokeLocation_Category", "Duration_Category", "ParalysisSide"]:
        ct = clin_res[col]["table"]
        lines.append(f"\n### {col}\n")
        lines.append("| group | " + " | ".join(str(c) for c in ct.columns) + " |")
        lines.append("|" + "---|" * (len(ct.columns) + 1))
        for grp, row in ct.iterrows():
            lines.append(f"| {grp} | " + " | ".join(str(int(v)) for v in row.values) + " |")
        lines.append("")

    lines.append("\n## 特征关联（Spearman, BH-FDR within each target）\n")
    for tgt, tname in [("within_acc", "within-subject acc"),
                       ("loso_mean", "mean LOSO acc"),
                       ("gap", "transfer gap")]:
        sub = cor[cor["target"] == tgt].dropna(subset=["r"]).copy()
        sub = sub.sort_values("p")
        lines.append(f"\n### {tname}\n")
        lines.append("| feature | r | p | p_FDR | n |")
        lines.append("|---|---|---|---|---|")
        for _, row in sub.iterrows():
            sig = " **" if row["p_fdr"] < 0.05 else ""
            lines.append(f"| {row['feature']} | {row['r']:.3f} | {row['p']:.4f} | "
                         f"{row['p_fdr']:.4f}{sig} | {int(row['n'])} |")

    lines.append("\n## 备注\n")
    lines.append("- 特征中绝对值 > 1e4 的数值视为除零伪迹置为 NaN（主要影响 alpha_theta_ratio）。")
    lines.append("- snr_task_baseline / snr_task_baseline_cv 已剔除：4s 数据经 z-score 归一化后"
                 " baseline-task 功率差被抹除，CSV 中的残留值（≈1.0002）无实际方差。")
    lines.append("- alpha_peak_freq 缺失较多，相关分析基于 pairwise complete 样本，注意 n 列。")
    lines.append("- ADFCNN LOSO 两列（loso_adfcnn_*）仅作稳健性参考，主分析基于 4 个 EEGNet pipeline。")
    lines.append("- 统计功效：N=50 在 α=0.05/power=0.8 下可检出 |r| ≥ ~0.39 的关联；更弱的关联可能漏检。")

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return df_out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_master_table()
    cor = run_correlations(df)
    df_tmp = df.copy()
    df_tmp["quadrant"] = assign_quadrants(df)
    clin_res = analyze_quadrant_clinical(df_tmp)
    df_out = write_report(df, cor, clin_res)

    df_out.to_csv(OUT_DIR / "master_table.csv")
    cor.to_csv(OUT_DIR / "correlations.csv", index=False)
    plot_quadrant(df)
    plot_correlations(cor)
    plot_quadrant_clinical(df_out)

    # console summary
    print("\n== Quadrant summary ==")
    print(df_out["quadrant"].value_counts().to_string())
    print("\n== A vs B clinical composition ==")
    for col in ["StrokeLocation_Category", "Duration_Category", "ParalysisSide", "NIHSS"]:
        print(f"  {col}: p={clin_res[col]['p']:.4f} ({clin_res[col]['test']})")
    print("\n== Top |r| for gap ==")
    top = cor[cor["target"] == "gap"].dropna(subset=["r"])
    top = top.reindex(top["r"].abs().sort_values(ascending=False).index).head(8)
    print(top[["feature", "r", "p", "p_fdr", "n"]].to_string(index=False))
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
