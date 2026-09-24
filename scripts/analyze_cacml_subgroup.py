#!/usr/bin/env python3
"""
Subgroup analysis for CACML LOSO results.

Tests whether clinical-informed, class-aware, or no-contrastive pretraining
performs differently across patient subgroups (stroke location, duration, NIHSS, LI).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml


def load_loso_results(root_dir):
    """Load per-subject LOSO results for each method and n_shot."""
    root = Path(root_dir)
    results = {}

    for exp_dir in sorted(root.glob('XWStroke_CACML_*')):
        config_file = exp_dir / 'config.yaml'
        results_file = exp_dir / 'results/results.json'
        if not config_file.exists() or not results_file.exists():
            continue

        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        name = cfg['experiment']['name'].lower()
        n_shot = cfg['trainer']['args']['n_shot']

        if 'clinical_informed' in name:
            method = 'Clinical-Informed'
        elif 'class_aware' in name:
            method = 'Class-Aware'
        elif 'no_contrastive' in name:
            method = 'No-Contrastive'
        else:
            continue

        with open(results_file) as f:
            data = json.load(f)

        # Per-subject accuracies
        subject_accs = {r['subject_id']: r['accuracy'] for r in data['subjects']}
        results[(method, n_shot)] = subject_accs

    return results


def load_clinical_features(participants_tsv, stratified_csv):
    """Load clinical features and stratification info for test subjects."""
    participants = pd.read_csv(participants_tsv, sep='\t')
    stratified = pd.read_csv(stratified_csv)

    # Parse subject ID
    participants['subject_id'] = participants['Participant_ID'].str.replace('sub-', '').astype(int)

    # Merge
    clinical = pd.merge(participants, stratified, on='subject_id', how='outer')
    return clinical


def analyze_subgroup(results, clinical, subgroup_col, methods, shot=20):
    """Compare methods within each subgroup level."""
    test_subjects = set()
    for method in methods:
        test_subjects.update(results[(method, shot)].keys())

    # Filter clinical data to test subjects
    sub_df = clinical[clinical['subject_id'].isin(test_subjects)].copy()

    print(f"\n{'='*80}")
    print(f"Subgroup analysis by: {subgroup_col} (n_shot={shot})")
    print(f"{'='*80}")

    table_rows = []
    for level in sorted(sub_df[subgroup_col].dropna().unique()):
        subjects = sub_df[sub_df[subgroup_col] == level]['subject_id'].values
        print(f"\n{subgroup_col} = {level} (n={len(subjects)}):")

        row = {'subgroup': f"{subgroup_col}={level}", 'n': len(subjects)}
        method_values = {}
        for method in methods:
            accs = [results[(method, shot)][sid] for sid in subjects if sid in results[(method, shot)]]
            mean_acc = np.mean(accs) if accs else np.nan
            std_acc = np.std(accs) if accs else np.nan
            method_values[method] = accs
            row[method] = f"{mean_acc*100:.2f} ± {std_acc*100:.2f}"
            print(f"  {method:<20}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

        # Pairwise t-test: Clinical-Informed vs No-Contrastive
        if 'Clinical-Informed' in method_values and 'No-Contrastive' in method_values:
            ci = method_values['Clinical-Informed']
            nc = method_values['No-Contrastive']
            if len(ci) == len(nc) and len(ci) > 2:
                t, p = stats.ttest_rel(ci, nc)
                row['CI vs NC p'] = f"{p:.4f}"
                print(f"  Clinical-Informed vs No-Contrastive: t={t:.3f}, p={p:.4f}")

        table_rows.append(row)

    return pd.DataFrame(table_rows)


def plot_subgroup_bars(results, clinical, subgroup_col, methods, shot=20, out_dir=None):
    """Bar plot comparing methods across subgroup levels."""
    test_subjects = set()
    for method in methods:
        test_subjects.update(results[(method, shot)].keys())
    sub_df = clinical[clinical['subject_id'].isin(test_subjects)].copy()

    levels = sorted(sub_df[subgroup_col].dropna().unique())
    n_levels = len(levels)
    n_methods = len(methods)
    x = np.arange(n_levels)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n_levels * 2), 6))
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for i, method in enumerate(methods):
        means = []
        stds = []
        for level in levels:
            subjects = sub_df[sub_df[subgroup_col] == level]['subject_id'].values
            accs = [results[(method, shot)][sid] for sid in subjects if sid in results[(method, shot)]]
            means.append(np.mean(accs) * 100)
            stds.append(np.std(accs) * 100)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=method, color=colors[i], alpha=0.8, capsize=3)

    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel(subgroup_col)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'LOSO {shot}-shot Accuracy by {subgroup_col}')
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([40, 70])

    plt.tight_layout()
    if out_dir:
        out_path = out_dir / f'subgroup_{subgroup_col.lower().replace(" ", "_")}_{shot}shot.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {out_path}")
    plt.close()


def compute_li(clinical, subject_id):
    """Compute lateralization index from stratified data."""
    row = clinical[clinical['subject_id'] == subject_id]
    if row.empty:
        return np.nan
    li_affected = row['li_affected'].values[0]
    li_unaffected = row['li_unaffected'].values[0]
    # LI as defined in stratified analysis
    return li_affected - li_unaffected


def main():
    loso_root = 'results/cacml_loso_results'
    participants_tsv = 'src/datasets/21679035/participants.tsv'
    stratified_csv = 'results/stratified/stratified_analysis_detailed.csv'
    out_dir = Path('results/cacml_subgroup_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_loso_results(loso_root)
    clinical = load_clinical_features(participants_tsv, stratified_csv)

    methods = ['Clinical-Informed', 'Class-Aware', 'No-Contrastive']
    shot = 20

    # Add LI to clinical dataframe
    clinical['LI'] = clinical['subject_id'].apply(lambda sid: compute_li(clinical, sid))
    clinical['LI_Category'] = clinical['LI'].apply(
        lambda x: 'LI>0' if pd.notna(x) and x > 0 else ('LI<0' if pd.notna(x) else np.nan)
    )

    # NIHSS categories
    clinical['NIHSS_Category'] = clinical['NIHSS_x'].apply(
        lambda x: 'Low (≤5)' if pd.notna(x) and x <= 5 else ('High (>5)' if pd.notna(x) else np.nan)
    )

    print("\nTest subjects clinical summary:")
    test_subjects = set()
    for method in methods:
        test_subjects.update(results[(method, shot)].keys())
    test_clinical = clinical[clinical['subject_id'].isin(test_subjects)]
    print(f"N = {len(test_clinical)}")
    print(test_clinical[['subject_id', 'StrokeLocation_Category', 'Duration_Category',
                          'NIHSS_x', 'LI', 'LI_Category']].to_string(index=False))

    # Subgroup analyses
    tables = {}
    for subgroup_col in ['StrokeLocation_Category', 'Duration_Category', 'NIHSS_Category', 'LI_Category']:
        df = analyze_subgroup(results, clinical, subgroup_col, methods, shot=shot)
        tables[subgroup_col] = df
        df.to_csv(out_dir / f'subgroup_{subgroup_col.lower().replace(" ", "_")}.csv', index=False)
        plot_subgroup_bars(results, clinical, subgroup_col, methods, shot=shot, out_dir=out_dir)

    # Correlation analysis: LI vs accuracy improvement
    print(f"\n{'='*80}")
    print("Correlation: LI vs Clinical-Informed - No-Contrastive accuracy gap")
    print(f"{'='*80}")
    li_values = []
    gaps = []
    for sid in test_subjects:
        li = compute_li(clinical, sid)
        if pd.isna(li):
            continue
        ci_acc = results[('Clinical-Informed', shot)].get(sid, np.nan)
        nc_acc = results[('No-Contrastive', shot)].get(sid, np.nan)
        if pd.isna(ci_acc) or pd.isna(nc_acc):
            continue
        li_values.append(li)
        gaps.append((ci_acc - nc_acc) * 100)

    if len(li_values) > 3:
        r, p = stats.pearsonr(li_values, gaps)
        print(f"Pearson r = {r:.3f}, p = {p:.4f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(li_values, gaps, alpha=0.7, s=100)
        z = np.polyfit(li_values, gaps, 1)
        p_fit = np.poly1d(z)
        plt.plot(sorted(li_values), p_fit(sorted(li_values)), 'r--', alpha=0.8)
        plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
        plt.xlabel('Lateralization Index (LI)')
        plt.ylabel('Accuracy Gain: Clinical-Informed - No-Contrastive (%)')
        plt.title('Clinical-Informed Gain vs Lateralization Index')
        plt.grid(True, alpha=0.3)
        out_path = out_dir / 'li_vs_gain.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved LI correlation plot to {out_path}")

    print(f"\nAll subgroup analysis outputs saved to {out_dir}")


if __name__ == '__main__':
    main()
