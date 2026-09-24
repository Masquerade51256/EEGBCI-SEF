#!/usr/bin/env python3
"""
Subgroup analysis for EEGNet+EA LOSO results.

Tests whether the EEGNet+EA baseline performs differently across
acute (Duration <= 3) vs chronic (Duration > 3) stroke patients.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


PARTICIPANTS_TSV = Path('src/datasets/21679035/participants.tsv')
RESULTS_CSV = Path('results/eegnet_ea_split38_summary/eegnet_ea_per_subject.csv')
OUTPUT_DIR = Path('results/eegnet_ea_split38_summary')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load clinical metadata
    participants = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    participants['subject_id'] = participants['Participant_ID'].str.replace('sub-', '').astype(int)
    participants['group'] = participants['Duration'].apply(
        lambda x: 'Acute' if x <= 3 else 'Chronic'
    )

    # Load EEGNet+EA results
    df = pd.read_csv(RESULTS_CSV)
    df = pd.merge(df, participants[['subject_id', 'Duration', 'group']], on='subject_id', how='left')

    # Summary table: rows = n_shot, cols = group
    table = []
    for n_shot in sorted(df['n_shot'].unique()):
        sub = df[df['n_shot'] == n_shot]
        acute = sub[sub['group'] == 'Acute']['accuracy']
        chronic = sub[sub['group'] == 'Chronic']['accuracy']

        row = {
            'n_shot': int(n_shot),
            'acute_n': len(acute),
            'acute_mean': f"{acute.mean()*100:.2f}",
            'acute_std': f"{acute.std()*100:.2f}",
            'chronic_n': len(chronic),
            'chronic_mean': f"{chronic.mean()*100:.2f}",
            'chronic_std': f"{chronic.std()*100:.2f}",
        }

        if len(acute) > 1 and len(chronic) > 1:
            t, p = stats.ttest_ind(acute, chronic)
            row['t'] = f"{t:.3f}"
            row['p'] = f"{p:.4f}"
        else:
            row['t'] = np.nan
            row['p'] = np.nan

        table.append(row)

    summary_df = pd.DataFrame(table)
    summary_df.to_csv(OUTPUT_DIR / 'eegnet_ea_subgroup_acute_chronic.csv', index=False)
    print(summary_df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    n_shots = summary_df['n_shot'].values.astype(int)
    acute_means = summary_df['acute_mean'].astype(float).values
    acute_stds = summary_df['acute_std'].astype(float).values
    chronic_means = summary_df['chronic_mean'].astype(float).values
    chronic_stds = summary_df['chronic_std'].astype(float).values

    x = np.arange(len(n_shots))
    width = 0.35
    ax.bar(x - width/2, acute_means, width, yerr=acute_stds, label='Acute (n=2)', capsize=3)
    ax.bar(x + width/2, chronic_means, width, yerr=chronic_stds, label='Chronic (n=10)', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}-shot' for n in n_shots])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('EEGNet+EA LOSO: Acute vs Chronic Patients')
    ax.legend()
    ax.set_ylim([0, 80])
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'eegnet_ea_acute_vs_chronic.png', dpi=300)
    plt.close(fig)

    print(f"\nSaved: {OUTPUT_DIR / 'eegnet_ea_subgroup_acute_chronic.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'eegnet_ea_acute_vs_chronic.png'}")


if __name__ == '__main__':
    main()
