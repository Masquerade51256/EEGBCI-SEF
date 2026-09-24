#!/usr/bin/env python3
"""Plot acute vs chronic stroke patients across n-shot settings."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


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
            method = 'Clinical-Informed'
        elif 'class_aware' in name:
            method = 'Class-Aware'
        elif 'no_contrastive' in name:
            method = 'No-Contrastive'
        else:
            continue
        with open(res_file) as f:
            data = json.load(f)
        subject_accs = {r['subject_id']: r['accuracy'] for r in data['subjects']}
        results[(method, shot)] = subject_accs
    return results


def main():
    results = load_results('results/cacml_loso_results')
    clinical = pd.read_csv('results/stratified/stratified_analysis_detailed.csv')
    clinical = clinical[clinical['subject_id'].isin([4,5,13,20,22,30,33,37,40,43,46,48])]

    shots = [0, 5, 10, 20]
    methods = ['Clinical-Informed', 'Class-Aware', 'No-Contrastive']
    colors = {'Clinical-Informed': '#e41a1c', 'Class-Aware': '#377eb8', 'No-Contrastive': '#4daf4a'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, (category, label) in enumerate([('Acute(≤3mo)', 'Acute (≤3 months)'),
                                                 ('Chronic(>3mo)', 'Chronic (>3 months)')]):
        ax = axes[ax_idx]
        subs = clinical[clinical['Duration_Category'] == category]['subject_id'].values

        for method in methods:
            means = []
            stds = []
            for shot in shots:
                accs = [results[(method, shot)][sid] for sid in subs]
                means.append(np.mean(accs) * 100)
                stds.append(np.std(accs) * 100)
            ax.errorbar(shots, means, yerr=stds, marker='o', linewidth=2,
                        label=method, color=colors[method], capsize=4, markersize=8)

        ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of Support Trials (n-shot)')
        ax.set_ylabel('LOSO Accuracy (%)')
        ax.set_title(label)
        ax.set_xticks(shots)
        ax.set_ylim([40, 62])
        ax.grid(True, alpha=0.3)
        if ax_idx == 1:
            ax.legend(loc='lower right')

    plt.tight_layout()
    out_dir = Path('results/cacml_subgroup_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'acute_vs_chronic_nshot.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

    # Print table
    print("\nAcute vs Chronic summary:")
    print(f"{'Method':<20} {'0-shot':<12} {'5-shot':<12} {'10-shot':<12} {'20-shot':<12}")
    print("-" * 70)
    for category in ['Acute(≤3mo)', 'Chronic(>3mo)']:
        print(f"\n{category}:")
        subs = clinical[clinical['Duration_Category'] == category]['subject_id'].values
        for method in methods:
            row = [method]
            for shot in shots:
                accs = [results[(method, shot)][sid] for sid in subs]
                row.append(f"{np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}")
            print(f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")


if __name__ == '__main__':
    main()
