#!/usr/bin/env python3
"""
Bootstrap and permutation tests for acute vs chronic CACML findings.

Given the small sample size (acute n=2, chronic n=10), these tests provide
rough confidence intervals and permutation-based p-values, but should be
interpreted as exploratory rather than definitive.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_results(loso_root):
    results = {}
    for exp_dir in sorted(Path(loso_root).glob('XWStroke_CACML_*')):
        cfg_file = exp_dir / 'config.yaml'
        res_file = exp_dir / 'results/results.json'
        if not cfg_file.exists() or not res_file.exists():
            continue
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        name = cfg['experiment']['name'].lower()
        shot = cfg['trainer']['args']['n_shot']
        if 'clinical_informed' in name:
            method = 'CI'
        elif 'class_aware' in name:
            method = 'CA'
        elif 'no_contrastive' in name:
            method = 'NC'
        else:
            continue
        with open(res_file) as f:
            data = json.load(f)
        subject_accs = {r['subject_id']: r['accuracy'] for r in data['subjects']}
        results[(method, shot)] = subject_accs
    return results


def bootstrap_diff(data_a, data_b, n_bootstrap=10000, random_state=42):
    """Bootstrap 95% CI for mean(data_a) - mean(data_b)."""
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(data_a, size=len(data_a), replace=True)
        sample_b = rng.choice(data_b, size=len(data_b), replace=True)
        diffs.append(np.mean(sample_a) - np.mean(sample_b))
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    return ci_low, ci_high, diffs


def permutation_interaction(acute_ci, acute_nc, chronic_ci, chronic_nc, n_perm=10000, random_state=42):
    """
    Permutation test for interaction: (acute_CI - acute_NC) - (chronic_CI - chronic_NC).
    Null hypothesis: duration label is exchangeable.
    """
    rng = np.random.default_rng(random_state)

    # Observed interaction
    observed = (np.mean(acute_ci) - np.mean(acute_nc)) - (np.mean(chronic_ci) - np.mean(chronic_nc))

    # Pool data
    all_ci = np.concatenate([acute_ci, chronic_ci])
    all_nc = np.concatenate([acute_nc, chronic_nc])
    n_acute = len(acute_ci)

    perm_diffs = []
    for _ in range(n_perm):
        # Shuffle which subjects are acute vs chronic
        idx = rng.permutation(len(all_ci))
        perm_acute_ci = all_ci[idx[:n_acute]]
        perm_chronic_ci = all_ci[idx[n_acute:]]
        perm_acute_nc = all_nc[idx[:n_acute]]
        perm_chronic_nc = all_nc[idx[n_acute:]]

        perm_interaction = (np.mean(perm_acute_ci) - np.mean(perm_acute_nc)) - \
                           (np.mean(perm_chronic_ci) - np.mean(perm_chronic_nc))
        perm_diffs.append(perm_interaction)

    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed))
    return observed, p_value, perm_diffs


def main():
    results = load_results('results/cacml_loso_results')
    clinical = pd.read_csv('results/stratified/stratified_analysis_detailed.csv')
    clinical = clinical[clinical['subject_id'].isin([4, 5, 13, 20, 22, 30, 33, 37, 40, 43, 46, 48])]

    shot = 20
    acute_subs = clinical[clinical['Duration_Category'] == 'Acute(≤3mo)']['subject_id'].values
    chronic_subs = clinical[clinical['Duration_Category'] == 'Chronic(>3mo)']['subject_id'].values

    print(f"Acute subjects: {acute_subs} (n={len(acute_subs)})")
    print(f"Chronic subjects: {chronic_subs} (n={len(chronic_subs)})")
    print()

    # Extract accuracies
    acute_ci = np.array([results[('CI', shot)][sid] for sid in acute_subs])
    acute_nc = np.array([results[('NC', shot)][sid] for sid in acute_subs])
    chronic_ci = np.array([results[('CI', shot)][sid] for sid in chronic_subs])
    chronic_nc = np.array([results[('NC', shot)][sid] for sid in chronic_subs])

    print(f"Acute  CI mean: {acute_ci.mean()*100:.2f}%")
    print(f"Acute  NC mean: {acute_nc.mean()*100:.2f}%")
    print(f"Chronic CI mean: {chronic_ci.mean()*100:.2f}%")
    print(f"Chronic NC mean: {chronic_nc.mean()*100:.2f}%")
    print()

    # Bootstrap for acute: CI > NC
    print("=" * 70)
    print("Bootstrap: Acute CI - Acute NC")
    ci_low, ci_high, _ = bootstrap_diff(acute_ci, acute_nc)
    print(f"Observed diff: {(acute_ci.mean() - acute_nc.mean())*100:.2f}%")
    print(f"95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Significant (CI excludes 0): {not (ci_low <= 0 <= ci_high)}")
    print()

    # Bootstrap for chronic: NC > CI
    print("=" * 70)
    print("Bootstrap: Chronic NC - Chronic CI")
    ci_low, ci_high, _ = bootstrap_diff(chronic_nc, chronic_ci)
    print(f"Observed diff: {(chronic_nc.mean() - chronic_ci.mean())*100:.2f}%")
    print(f"95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Significant (CI excludes 0): {not (ci_low <= 0 <= ci_high)}")
    print()

    # Permutation test for interaction
    print("=" * 70)
    print("Permutation test: method × duration interaction")
    observed, p_value, _ = permutation_interaction(acute_ci, acute_nc, chronic_ci, chronic_nc)
    print(f"Observed interaction: {observed*100:.2f}%")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"Significant at α=0.05: {p_value < 0.05}")
    print()

    # Paired t-tests (despite small sample)
    print("=" * 70)
    print("Paired t-tests (exploratory, small samples):")
    if len(acute_ci) >= 2:
        t, p = stats.ttest_rel(acute_ci, acute_nc)
        print(f"Acute CI vs NC: t={t:.3f}, p={p:.4f}")
    if len(chronic_ci) >= 2:
        t, p = stats.ttest_rel(chronic_nc, chronic_ci)
        print(f"Chronic NC vs CI: t={t:.3f}, p={p:.4f}")


if __name__ == '__main__':
    main()
