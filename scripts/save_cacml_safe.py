#!/usr/bin/env python3
"""
Safely save CACML pretrained weights, filtering out problematic nn.Linear(0,0) layers
from ADFCNNBackbone to avoid segfault during torch.save.

Usage:
    python scripts/save_cacml_safe.py \
        --experiment_dir experiments/cacml_pretrain/XWStroke_CACML_YYYYMMDD_HHMMSS \
        --output_path experiments/cacml_pretrain/cacml_pretrained_safe.pt
"""

import argparse
import sys
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory containing checkpoints/')
    parser.add_argument('--output_path', type=str, default='./experiments/cacml_pretrain/cacml_pretrained_safe.pt',
                        help='Output path for safe checkpoint')
    return parser.parse_args()

def save_safe(experiment_dir: str, output_path: str):
    import glob
    import os

    # Find the latest checkpoint
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: No checkpoints directory found: {checkpoint_dir}")
        sys.exit(1)

    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not checkpoints:
        print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
        sys.exit(1)

    # Load the latest checkpoint
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Loading: {latest}")

    checkpoint = torch.load(latest, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Filter out problematic layers (ADFCNN backbone's initial nn.Linear(0,0))
    safe_state_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if v.numel() == 0:
            skipped.append(k)
            continue
        safe_state_dict[k] = v

    print(f"Total layers: {len(state_dict)}")
    print(f"Skipped (empty): {len(skipped)}")
    if skipped:
        print(f"  Skipped keys: {skipped}")

    # Build safe checkpoint
    safe_checkpoint = {
        'model_state_dict': safe_state_dict,
        'epoch': checkpoint.get('epoch', -1),
        'best_val_acc': checkpoint.get('best_val_acc', -1),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(safe_checkpoint, output_path)
    print(f"Safe checkpoint saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    args = parse_args()
    save_safe(args.experiment_dir, args.output_path)
