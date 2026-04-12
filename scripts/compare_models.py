"""
Compare XWStroke experiment results between ADFCNN and XWFilterBankNet models.

This script extracts ADFCNN results from training.log and compares them with
XWFilterBankNet results from the experiment JSON file.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def extract_adfcnn_results_from_log(log_path: str) -> Dict[int, float]:
    """
    Extract ADFCNN subject accuracies from training.log file.
    
    Returns:
        Dictionary mapping subject_id to average validation accuracy
    """
    subject_results = {}
    current_subject = None
    fold_accuracies = []
    
    # Pattern to match subject lines
    subject_pattern = re.compile(r'Processing Subject: (\d+)')
    # Pattern to match fold best accuracy
    fold_pattern = re.compile(r'Fold (\d) finished\. Best Val Acc: ([\d.]+)')
    
    print(f"Parsing {log_path}...")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check for subject start
            subject_match = subject_pattern.search(line)
            if subject_match:
                # Save previous subject's results if exists
                if current_subject is not None and fold_accuracies:
                    avg_acc = np.mean(fold_accuracies)
                    subject_results[current_subject] = avg_acc
                    print(f"  Subject {current_subject}: {avg_acc:.4f} (from {len(fold_accuracies)} folds)")
                
                current_subject = int(subject_match.group(1))
                fold_accuracies = []
                continue
            
            # Check for fold accuracy
            fold_match = fold_pattern.search(line)
            if fold_match and current_subject is not None:
                fold_acc = float(fold_match.group(2))
                fold_accuracies.append(fold_acc)
    
    # Don't forget the last subject
    if current_subject is not None and fold_accuracies:
        avg_acc = np.mean(fold_accuracies)
        subject_results[current_subject] = avg_acc
        print(f"  Subject {current_subject}: {avg_acc:.4f} (from {len(fold_accuracies)} folds)")
    
    print(f"\nExtracted results for {len(subject_results)} subjects from ADFCNN")
    return subject_results


def load_xwfilterbanknet_results(json_path: str) -> Dict[int, float]:
    """
    Load XWFilterBankNet results from JSON file.
    
    Returns:
        Dictionary mapping subject_id to average validation accuracy
    """
    print(f"\nLoading {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    subject_results = {}
    for subject_data in data.get('subjects', []):
        subject_id = subject_data['subject_id']
        avg_acc = subject_data['avg_val_acc']
        subject_results[subject_id] = avg_acc
    
    print(f"Loaded results for {len(subject_results)} subjects from XWFilterBankNet")
    print(f"Overall mean: {data.get('overall_mean', 0):.4f} ± {data.get('overall_std', 0):.4f}")
    
    return subject_results


def plot_comparison(
    adfcnn_results: Dict[int, float],
    xwfb_results: Dict[int, float],
    output_path: str = None
) -> None:
    """
    Plot comparison bar chart between two models.
    """
    # Get common subjects
    common_subjects = sorted(set(adfcnn_results.keys()) & set(xwfb_results.keys()))
    
    if not common_subjects:
        print("Error: No common subjects found between the two models!")
        return
    
    print(f"\nComparing {len(common_subjects)} common subjects...")
    
    # Prepare data
    adfcnn_accs = [adfcnn_results[s] for s in common_subjects]
    xwfb_accs = [xwfb_results[s] for s in common_subjects]
    
    # Calculate statistics
    adfcnn_mean = np.mean(adfcnn_accs)
    adfcnn_std = np.std(adfcnn_accs)
    xwfb_mean = np.mean(xwfb_accs)
    xwfb_std = np.std(xwfb_accs)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Subplot 1: Individual subject comparison
    x = np.arange(len(common_subjects))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, adfcnn_accs, width, label='ADFCNN', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, xwfb_accs, width, label='XWFilterBankNet', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Subject ID', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('XWStroke Dataset: ADFCNN vs XWFilterBankNet Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_subjects, rotation=90, fontsize=8)
    # ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # Add mean lines
    ax1.axhline(y=adfcnn_mean, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'ADFCNN Mean: {adfcnn_mean:.3f}')
    ax1.axhline(y=xwfb_mean, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7,
                label=f'XWFilterBankNet Mean: {xwfb_mean:.3f}')
    # ax1.legend(loc='upper right', fontsize=9)
    
    # Subplot 2: Distribution comparison (histogram)
    bins = np.linspace(0, 1, 21)
    ax2.hist(adfcnn_accs, bins=bins, alpha=0.6, label=f'ADFCNN (μ={adfcnn_mean:.3f}, σ={adfcnn_std:.3f})', 
             color='#3498db', edgecolor='black')
    ax2.hist(xwfb_accs, bins=bins, alpha=0.6, label=f'XWFilterBankNet (μ={xwfb_mean:.3f}, σ={xwfb_std:.3f})', 
             color='#e74c3c', edgecolor='black')
    
    ax2.set_xlabel('Validation Accuracy', fontsize=12)
    ax2.set_ylabel('Number of Subjects', fontsize=12)
    ax2.set_title('Accuracy Distribution Comparison', fontsize=12, fontweight='bold')
    # ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = 'XWStroke_ADFCNN_vs_XWFilterBankNet_comparison.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED COMPARISON RESULTS")
    print("="*60)
    print(f"\nADFCNN:")
    print(f"  Mean Accuracy: {adfcnn_mean:.4f} ± {adfcnn_std:.4f}")
    print(f"  Min Accuracy:  {min(adfcnn_accs):.4f} (Subject {common_subjects[adfcnn_accs.index(min(adfcnn_accs))]})")
    print(f"  Max Accuracy:  {max(adfcnn_accs):.4f} (Subject {common_subjects[adfcnn_accs.index(max(adfcnn_accs))]})")
    
    print(f"\nXWFilterBankNet:")
    print(f"  Mean Accuracy: {xwfb_mean:.4f} ± {xwfb_std:.4f}")
    print(f"  Min Accuracy:  {min(xwfb_accs):.4f} (Subject {common_subjects[xwfb_accs.index(min(xwfb_accs))]})")
    print(f"  Max Accuracy:  {max(xwfb_accs):.4f} (Subject {common_subjects[xwfb_accs.index(max(xwfb_accs))]})")
    
    improvement = xwfb_mean - adfcnn_mean
    print(f"\nDifference:")
    print(f"  XWFilterBankNet - ADFCNN = {improvement:+.4f} ({improvement/adfcnn_mean*100:+.2f}%)")
    
    # Count wins
    xwfb_wins = sum(1 for a, x in zip(adfcnn_accs, xwfb_accs) if x > a)
    adfcnn_wins = sum(1 for a, x in zip(adfcnn_accs, xwfb_accs) if a > x)
    ties = len(common_subjects) - xwfb_wins - adfcnn_wins
    
    print(f"\nHead-to-head (out of {len(common_subjects)} subjects):")
    print(f"  XWFilterBankNet wins: {xwfb_wins}")
    print(f"  ADFCNN wins:         {adfcnn_wins}")
    print(f"  Ties:                {ties}")
    print("="*60)
    
    plt.show()


def main():
    """Main function."""
    # Define paths
    log_path = 'training.log'
    xwfb_json_path = 'experiments/XWStroke_XWFilterBankNet_20260409_222122/results/results.json'
    output_path = 'XWStroke_ADFCNN_vs_XWFilterBankNet_comparison.png'
    
    # Check if files exist
    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        print("Please ensure training.log is in the current directory")
        return
    
    if not os.path.exists(xwfb_json_path):
        # Try to find any XWFilterBankNet experiment results
        alternative_paths = list(Path('experiments').glob('XWStroke_XWFilterBankNet_*/results/results.json'))
        if alternative_paths:
            xwfb_json_path = str(alternative_paths[0])
            print(f"Using alternative path: {xwfb_json_path}")
        else:
            print(f"Error: XWFilterBankNet results not found at {xwfb_json_path}")
            print("Please check the experiment directory name")
            return
    
    # Extract results
    adfcnn_results = extract_adfcnn_results_from_log(log_path)
    xwfb_results = load_xwfilterbanknet_results(xwfb_json_path)
    
    # Plot comparison
    plot_comparison(adfcnn_results, xwfb_results, output_path)


if __name__ == '__main__':
    main()
