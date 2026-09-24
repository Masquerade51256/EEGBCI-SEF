#!/usr/bin/env python3
"""
Channel importance analysis for CACML (no-contrastive pretrained model).

Uses channel occlusion on the 0-shot (pretrained) model to estimate which
EEG channels drive left/right MI classification on the 12 held-out test subjects.
Outputs:
  - per-subject/per-channel importance CSV
  - summary CSV (mean across subjects, split by acute vs chronic)
  - topographic maps (overall, acute vs chronic)
  - a short text report
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import XWStrokeDataset
from models.CACML import CACML
from preprocessing.artifact_removal.base import BaseArtifactRemovalProcessor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRETRAIN_CONFIG = Path('configs/experiment/xwstroke_cacml_split38_no_contrastive_pretrain.yaml')
CHECKPOINT = Path('experiments/cacml_split38_no_contrastive/XWStroke_CACML_20260618_103943/checkpoints/cacml_pretrained_best.pt')
DATASET_CONFIG = Path('configs/dataset/XWStroke.yaml')
PARTICIPANTS_TSV = Path('src/datasets/21679035/participants.tsv')

TEST_SUBJECTS = [4, 5, 13, 20, 22, 30, 33, 37, 40, 43, 46, 48]

OUTPUT_DIR = Path('results/cacml_channel_importance')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CH_NAMES = BaseArtifactRemovalProcessor.DEFAULT_XWSTROKE_CH_NAMES[:30]  # drop EOG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset_info(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # CACML configs are nested under 'data:'; dataset YAML is nested under 'dataset:'
    if 'data' in cfg:
        # experiment config points to dataset info_path
        data_cfg = cfg['data']
        info_path = data_cfg.get('info_path', str(DATASET_CONFIG))
        with open(info_path) as f:
            return yaml.safe_load(f)
    return cfg


def build_model(dataset_info: dict, model_args: dict):
    target_sr = dataset_info['preprocessing']['resample']['target_sr']
    window_sec = dataset_info['preprocessing']['windowing']['window_length_sec']
    input_length = int(target_sr * window_sec)
    channels_selected = dataset_info['preprocessing']['channel_selection']['channels_to_select']
    filter_banks = dataset_info['preprocessing']['filter_bank']['filter_banks']

    model = CACML(
        num_classes=2,
        num_channels=len(channels_selected),
        num_bands=len(filter_banks),
        input_length=input_length,
        embed_dim=model_args.get('embed_dim', 128),
        dropout=model_args.get('dropout', 0.5),
    )
    return model


def load_pretrained_model(checkpoint_path: Path, dataset_info: dict, model_args: dict):
    model = build_model(dataset_info, model_args)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def evaluate(model, loader):
    """Return accuracy on a DataLoader."""
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x, x_clinical=None)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def occlusion_importance(model, loader):
    """
    Zero-out each EEG channel (replace with its mean over the batch) and measure
    accuracy drop relative to baseline.

    Returns:
        baseline_acc: float
        importances: np.ndarray of shape (n_channels,)
    """
    baseline_acc = evaluate(model, loader)

    all_data = []
    all_labels = []
    for x, y in loader:
        all_data.append(x)
        all_labels.append(y)
    data = torch.cat(all_data, dim=0).to(DEVICE)   # [N, bands, C, T]
    labels = torch.cat(all_labels, dim=0).to(DEVICE)

    n_channels = data.size(2)
    importances = np.zeros(n_channels)

    with torch.no_grad():
        for c in range(n_channels):
            occluded = data.clone()
            channel_mean = occluded[:, :, c, :].mean()
            occluded[:, :, c, :] = channel_mean

            logits = model(occluded, x_clinical=None)
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            importances[c] = baseline_acc - acc

    return baseline_acc, importances


def plot_topographic(importance: np.ndarray, title: str, out_path: Path, vmin=None, vmax=None):
    """Plot a 2-D topographic map using MNE."""
    info = mne.create_info(ch_names=CH_NAMES, sfreq=250, ch_types='eeg')
    info.set_montage('standard_1020', match_case=False, on_missing='warn')

    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        importance, info, axes=ax, show=False,
        contours=6, sensors=True, names=CH_NAMES,
        vlim=(vmin, vmax), cmap='RdBu_r'
    )
    ax.set_title(title, fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Accuracy drop (occlusion)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load configs
    with open(PRETRAIN_CONFIG) as f:
        pretrain_cfg = yaml.safe_load(f)
    model_args = pretrain_cfg['model']['args']

    dataset_info = load_dataset_info(PRETRAIN_CONFIG)

    # Clinical metadata
    participants = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    participants['subject_id'] = participants['Participant_ID'].str.replace('sub-', '').astype(int)
    duration_map = dict(zip(participants['subject_id'], participants['Duration']))

    # Load model
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = load_pretrained_model(CHECKPOINT, dataset_info, model_args)

    records = []
    for sid in TEST_SUBJECTS:
        ds = XWStrokeDataset(subject_id=sid, dataset_info=dataset_info, mode='test')
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        baseline, importances = occlusion_importance(model, loader)
        duration = duration_map.get(sid, np.nan)
        group = 'Acute' if duration <= 3 else 'Chronic'

        print(f"Subject {sid:02d} ({group}, duration={duration}): baseline={baseline*100:.2f}%")

        for ch_name, imp in zip(CH_NAMES, importances):
            records.append({
                'subject_id': sid,
                'duration': duration,
                'group': group,
                'channel': ch_name,
                'importance': float(imp),
                'baseline_acc': float(baseline),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / 'channel_importance.csv', index=False)

    # Summary statistics
    summary = df.groupby('channel')['importance'].agg(['mean', 'std', 'sem']).reset_index()
    summary = summary.sort_values('mean', ascending=False)
    summary.to_csv(OUTPUT_DIR / 'channel_importance_summary.csv', index=False)

    # Overall topographic map
    overall = df.groupby('channel')['importance'].mean().reindex(CH_NAMES).values
    plot_topographic(overall, 'CACML (No-Contrastive) - Overall Channel Importance',
                     OUTPUT_DIR / 'topo_overall.png',
                     vmin=-overall.max(), vmax=overall.max())

    # Bar plot of overall mean importance
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CH_NAMES))
    means = summary.set_index('channel').reindex(CH_NAMES)['mean'].values
    colors = ['C0' if m >= 0 else 'C3' for m in means]
    ax.bar(x, means, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Accuracy drop (occlusion)')
    ax.set_title('CACML (No-Contrastive) - Mean Channel Importance across Test Subjects')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'channel_importance_bars.png', dpi=300)
    plt.close(fig)

    # Acute vs Chronic
    if len(df[df['group'] == 'Acute']) > 0 and len(df[df['group'] == 'Chronic']) > 0:
        acute = df[df['group'] == 'Acute'].groupby('channel')['importance'].mean().reindex(CH_NAMES).values
        chronic = df[df['group'] == 'Chronic'].groupby('channel')['importance'].mean().reindex(CH_NAMES).values

        vmax = max(np.abs(acute).max(), np.abs(chronic).max())
        plot_topographic(acute, 'CACML (No-Contrastive) - Acute Patients (n=2)',
                         OUTPUT_DIR / 'topo_acute.png', vmin=-vmax, vmax=vmax)
        plot_topographic(chronic, 'CACML (No-Contrastive) - Chronic Patients (n=10)',
                         OUTPUT_DIR / 'topo_chronic.png', vmin=-vmax, vmax=vmax)
        plot_topographic(acute - chronic, 'Acute - Chronic Importance Difference',
                         OUTPUT_DIR / 'topo_acute_minus_chronic.png',
                         vmin=-vmax, vmax=vmax)

    # Text report
    top5 = summary.head(5)['channel'].tolist()
    bottom5 = summary.tail(5)['channel'].tolist()
    report = []
    report.append("CACML Channel Importance Report (occlusion on no-contrastive pretrained model)")
    report.append("=" * 70)
    report.append(f"Output directory: {OUTPUT_DIR}")
    report.append(f"Mean baseline accuracy across test subjects: {df.groupby('subject_id')['baseline_acc'].first().mean()*100:.2f}%")
    report.append(f"Top 5 channels: {', '.join(top5)}")
    report.append(f"Bottom 5 channels: {', '.join(bottom5)}")
    report.append("")
    report.append("Overall mean importance per channel:")
    for _, row in summary.iterrows():
        report.append(f"  {row['channel']:<6}: {row['mean']:+.4f} ± {row['std']:.4f}")

    report_text = '\n'.join(report)
    (OUTPUT_DIR / 'report.txt').write_text(report_text)
    print('\n' + report_text)

    # JSON for easy downstream use
    json_out = {
        'mean_baseline_acc': float(df.groupby('subject_id')['baseline_acc'].first().mean()),
        'overall': {ch: float(imp) for ch, imp in zip(CH_NAMES, overall)},
        'top5': top5,
        'bottom5': bottom5,
    }
    if len(df[df['group'] == 'Acute']) > 0 and len(df[df['group'] == 'Chronic']) > 0:
        json_out['acute'] = {ch: float(imp) for ch, imp in zip(CH_NAMES, acute)}
        json_out['chronic'] = {ch: float(imp) for ch, imp in zip(CH_NAMES, chronic)}
    (OUTPUT_DIR / 'report.json').write_text(json.dumps(json_out, indent=2))

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
