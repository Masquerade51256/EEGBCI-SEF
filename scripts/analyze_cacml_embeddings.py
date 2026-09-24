#!/usr/bin/env python3
"""
Analyze CACML pretrained embeddings for paper visualizations.

Produces:
1. Training curves (total/cls/contrast loss + accuracy)
2. t-SNE plot of EEG embeddings colored by MI label
3. Cosine similarity matrices (within/across labels)
4. Embedding variance statistics
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import XWStrokeDataset, get_clinical_feature_vector
from models.CACML import CACML


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config, sample_dataset):
    model_args = config.get('model', {}).get('args', {}).copy()
    data_shape = sample_dataset.data.shape
    model_args.setdefault('num_classes', len(set(sample_dataset.labels)))
    model_args.setdefault('num_channels', data_shape[2])
    model_args.setdefault('num_bands', data_shape[1])
    model_args.setdefault('input_length', data_shape[3])
    model = CACML(**model_args)
    return model


def load_pretrained(model, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_embeddings(model, subjects, dataset_info, device='cuda'):
    """Extract EEG embeddings and labels from a list of subjects."""
    all_eeg = []
    all_labels = []
    all_subjects = []
    all_clinical = []

    with torch.no_grad():
        for sid in subjects:
            ds = XWStrokeDataset(subject_id=sid, dataset_info=dataset_info, mode='train')
            if len(ds) == 0:
                continue
            loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
            clinical = get_clinical_feature_vector(sid, dataset_info).unsqueeze(0).to(device)

            for data, labels in loader:
                data = data.to(device)
                B = data.size(0)
                clinical_batch = clinical.repeat(B, 1)
                logits, f_eeg, f_clin = model(data, clinical_batch, return_features=True)
                all_eeg.append(f_eeg.cpu().numpy())
                all_labels.append(labels.numpy())
                all_subjects.append(np.full(B, sid))
                all_clinical.append(f_clin.cpu().numpy())

    return {
        'eeg': np.concatenate(all_eeg, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
        'subjects': np.concatenate(all_subjects, axis=0),
        'clinical': np.concatenate(all_clinical, axis=0),
    }


def parse_training_log(log_path):
    """Parse per-component losses from training.log."""
    pattern = re.compile(
        r'Epoch\s+(\d+)/\d+:\s+L=([\-\d\.]+)\(c=([\-\d\.]+)\s+a=([\-\d\.]+)\s+cls=([\-\d\.]+)\)\s+'
        r'\|\s+Val\s+L=([\-\d\.]+)\(c=([\-\d\.]+)\s+a=([\-\d\.]+)\s+cls=([\-\d\.]+)\)'
    )
    records = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    'epoch': int(m.group(1)),
                    'train_total': float(m.group(2)),
                    'train_contrast': float(m.group(3)),
                    'train_align': float(m.group(4)),
                    'train_cls': float(m.group(5)),
                    'val_total': float(m.group(6)),
                    'val_contrast': float(m.group(7)),
                    'val_align': float(m.group(8)),
                    'val_cls': float(m.group(9)),
                })
    return records


def plot_training_curves(history, log_records, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train')
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Classification loss
    ax = axes[0, 1]
    if log_records:
        ax.plot([r['epoch'] for r in log_records], [r['train_cls'] for r in log_records], label='Train')
        ax.plot([r['epoch'] for r in log_records], [r['val_cls'] for r in log_records], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Loss')
    ax.set_title('Classification Loss (Cross-Entropy)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Contrastive loss
    ax = axes[1, 0]
    if log_records:
        ax.plot([r['epoch'] for r in log_records], [r['train_contrast'] for r in log_records], label='Train')
        ax.plot([r['epoch'] for r in log_records], [r['val_contrast'] for r in log_records], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive Loss')
    ax.set_title('Contrastive Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, history['train_acc'], label='Train')
    if 'val_acc' in history and history['val_acc']:
        ax.plot(epochs, history['val_acc'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'training_curves.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {out_path}")


def plot_tsne(eeg, labels, out_dir, n_samples=1000, perplexity=30):
    rng = np.random.default_rng(42)
    n = len(eeg)
    if n > n_samples:
        idx = rng.choice(n, n_samples, replace=False)
    else:
        idx = np.arange(n)

    eeg_sub = eeg[idx]
    labels_sub = labels[idx]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(eeg_sub) - 1),
                random_state=42, init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(eeg_sub)

    plt.figure(figsize=(8, 6))
    colors = ['#e41a1c', '#377eb8']
    for label in np.unique(labels_sub):
        mask = labels_sub == label
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    c=colors[int(label)], alpha=0.6, s=15,
                    label=f'Class {int(label)}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE of EEG Embeddings (Clinical-Informed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / 'tsne_embeddings.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE to {out_path}")


def plot_similarity_distributions(eeg, labels, out_dir):
    eeg_norm = F.normalize(torch.from_numpy(eeg), p=2, dim=1).numpy()
    sim = np.matmul(eeg_norm, eeg_norm.T)

    # Sample pairs to avoid O(N^2)
    rng = np.random.default_rng(42)
    n = len(eeg)
    n_pairs = min(5000, n * (n - 1) // 2)
    idx_i = rng.integers(0, n, n_pairs)
    idx_j = rng.integers(0, n, n_pairs)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    same_label = labels[idx_i] == labels[idx_j]
    diff_label = ~same_label

    sim_same = sim[idx_i[same_label], idx_j[same_label]]
    sim_diff = sim[idx_i[diff_label], idx_j[diff_label]]

    plt.figure(figsize=(8, 6))
    bins = np.linspace(-1, 1, 50)
    plt.hist(sim_same, bins=bins, alpha=0.6, label='Same MI label', density=True)
    plt.hist(sim_diff, bins=bins, alpha=0.6, label='Different MI label', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of Pairwise Cosine Similarities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / 'similarity_distributions.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity distributions to {out_path}")

    # Print statistics
    print(f"\nCosine similarity statistics:")
    print(f"  Same-label mean: {sim_same.mean():.4f} ± {sim_same.std():.4f}")
    print(f"  Diff-label mean: {sim_diff.mean():.4f} ± {sim_diff.std():.4f}")
    print(f"  Separation (same - diff): {sim_same.mean() - sim_diff.mean():.4f}")


def compute_variance(eeg, labels, out_dir):
    overall_var = np.var(eeg, axis=0).mean()

    class_vars = []
    for label in np.unique(labels):
        class_eeg = eeg[labels == label]
        class_vars.append(np.var(class_eeg, axis=0).mean())

    between_class_var = np.var([eeg[labels == l].mean(axis=0) for l in np.unique(labels)], axis=0).mean()

    print(f"\nEmbedding variance statistics:")
    print(f"  Overall mean variance per dim: {overall_var:.6f}")
    print(f"  Within-class mean variance: {np.mean(class_vars):.6f}")
    print(f"  Between-class variance: {between_class_var:.6f}")

    with open(out_dir / 'variance_stats.txt', 'w') as f:
        f.write(f"Overall mean variance per dim: {overall_var:.6f}\n")
        f.write(f"Within-class mean variance: {np.mean(class_vars):.6f}\n")
        f.write(f"Between-class variance: {between_class_var:.6f}\n")


def main():
    experiment_dir = Path('experiments/cacml_split38_clinical_informed/XWStroke_CACML_20260618_095759')
    config_path = experiment_dir / 'config.yaml'
    checkpoint_path = experiment_dir / 'checkpoints/cacml_pretrained_best.pt'
    results_path = experiment_dir / 'results/results.json'
    # Output to a user-writable directory (experiment dirs may be root-owned)
    out_dir = Path('results/cacml_clinical_informed_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    dataset_info = load_config(config['data']['info_path'])

    # Merge dataset info into config format expected by XWStrokeDataset
    # The config stores preprocessing under data/preprocessing in some configs,
    # but dataset_info YAML may have a different structure. We pass the loaded
    # dataset_info dict directly.
    subjects = config['data']['subjects']

    # Plot training curves from results.json
    with open(results_path) as f:
        results = json.load(f)
    history = results['history']
    # Ensure consistent naming
    history.setdefault('train_cls_loss', history.get('train_cls_loss', history.get('train_loss_cls', [])))
    history.setdefault('val_cls_loss', history.get('val_cls_loss', history.get('val_loss_cls', [])))
    history.setdefault('train_contrast_loss', history.get('train_contrast_loss', history.get('train_loss_contrast', [])))
    history.setdefault('val_contrast_loss', history.get('val_contrast_loss', history.get('val_loss_contrast', [])))
    history.setdefault('train_acc', history.get('train_acc', history.get('train_accuracy', [])))
    history.setdefault('val_acc', history.get('val_acc', history.get('val_accuracy', [])))

    log_records = parse_training_log(experiment_dir / 'logs/training.log')
    plot_training_curves(history, log_records, out_dir)

    # Build model and load weights
    sample_ds = XWStrokeDataset(subject_id=subjects[0], dataset_info=dataset_info, mode='train')
    model = build_model(config, sample_ds)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_pretrained(model, checkpoint_path, device=device)

    # Extract embeddings from training subjects
    print(f"\nExtracting embeddings from {len(subjects)} subjects...")
    data = extract_embeddings(model, subjects, dataset_info, device=device)
    print(f"Extracted {len(data['eeg'])} samples")

    # Plots and statistics
    plot_tsne(data['eeg'], data['labels'], out_dir)
    plot_similarity_distributions(data['eeg'], data['labels'], out_dir)
    compute_variance(data['eeg'], data['labels'], out_dir)

    print(f"\nAll visualizations saved to {out_dir}")


if __name__ == '__main__':
    main()
