#!/usr/bin/env python
"""
Experiment Results Comparison Script

Compare two experiment results, including:
1. Check if they use the same dataset
2. Check the number of overlapping subjects
3. Plot accuracy comparison bar chart for overlapping subjects
4. Summarize and highlight differences between the two experiments

Usage:
    python scripts/compare_experiments.py <experiment1_name> <experiment2_name> [--output <output_path>]

Example:
    python scripts/compare_experiments.py XWStroke_ADFCNN_20260410_012331 DummyExperiment_XWFilterBankNet_20260410_000001
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch


def load_config(experiment_path: str) -> Dict:
    """Load experiment configuration file"""
    config_path = os.path.join(experiment_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_results(experiment_path: str) -> Dict:
    """Load experiment results file"""
    results_path = os.path.join(experiment_path, "results", "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_experiment_info(experiment_name: str) -> Tuple[Dict, Dict, str]:
    """Get experiment information (config and results)"""
    experiments_dir = "experiments"
    experiment_path = os.path.join(experiments_dir, experiment_name)
    
    if not os.path.exists(experiment_path):
        # Try to find experiments starting with this name
        matching = [d for d in os.listdir(experiments_dir) 
                   if d.startswith(experiment_name) and os.path.isdir(os.path.join(experiments_dir, d))]
        if len(matching) == 1:
            experiment_path = os.path.join(experiments_dir, matching[0])
            experiment_name = matching[0]
        elif len(matching) > 1:
            raise ValueError(f"Multiple experiments match '{experiment_name}': {matching}")
        else:
            raise FileNotFoundError(f"Experiment not found: {experiment_name}")
    
    config = load_config(experiment_path)
    results = load_results(experiment_path)
    
    return config, results, experiment_name


def check_same_dataset(config1: Dict, config2: Dict) -> Tuple[bool, str]:
    """Check if two experiments use the same dataset"""
    dataset1 = config1.get('data', {}).get('dataset', 'Unknown')
    dataset2 = config2.get('data', {}).get('dataset', 'Unknown')
    
    if dataset1 == dataset2:
        return True, dataset1
    else:
        return False, f"{dataset1} vs {dataset2}"


def get_common_subjects(results1: Dict, results2: Dict) -> Tuple[Set[int], Set[int], Set[int]]:
    """Get overlapping subjects"""
    subjects1 = {s['subject_id'] for s in results1.get('subjects', [])}
    subjects2 = {s['subject_id'] for s in results2.get('subjects', [])}
    
    common = subjects1 & subjects2
    only_in_1 = subjects1 - subjects2
    only_in_2 = subjects2 - subjects1
    
    return common, only_in_1, only_in_2


def get_subject_accuracies(results: Dict) -> Dict[int, float]:
    """Get average accuracy for each subject"""
    return {s['subject_id']: s['avg_val_acc'] for s in results.get('subjects', [])}


def compare_configs(config1: Dict, config2: Dict, exp1_name: str, exp2_name: str) -> Dict[str, Any]:
    """Compare two experiment configurations and find differences"""
    differences = {}
    
    # 1. Basic experiment info
    exp1_info = config1.get('experiment', {})
    exp2_info = config2.get('experiment', {})
    
    if exp1_info.get('name') != exp2_info.get('name'):
        differences['experiment_name'] = {
            exp1_name: exp1_info.get('name', 'N/A'),
            exp2_name: exp2_info.get('name', 'N/A')
        }
    
    if exp1_info.get('seed') != exp2_info.get('seed'):
        differences['seed'] = {
            exp1_name: exp1_info.get('seed', 'N/A'),
            exp2_name: exp2_info.get('seed', 'N/A')
        }
    
    # 2. Dataset info
    data1 = config1.get('data', {})
    data2 = config2.get('data', {})
    
    if data1.get('dataset') != data2.get('dataset'):
        differences['dataset'] = {
            exp1_name: data1.get('dataset', 'N/A'),
            exp2_name: data2.get('dataset', 'N/A')
        }
    
    # 3. Model info
    model1 = config1.get('model', {})
    model2 = config2.get('model', {})
    
    if model1.get('type') != model2.get('type'):
        differences['model_type'] = {
            exp1_name: model1.get('type', 'N/A'),
            exp2_name: model2.get('type', 'N/A')
        }
    
    # Compare model arguments
    model_args1 = model1.get('args', {})
    model_args2 = model2.get('args', {})
    model_diff = {}
    all_model_keys = set(model_args1.keys()) | set(model_args2.keys())
    for key in all_model_keys:
        val1 = model_args1.get(key, 'N/A')
        val2 = model_args2.get(key, 'N/A')
        if val1 != val2:
            model_diff[key] = {exp1_name: val1, exp2_name: val2}
    if model_diff:
        differences['model_args'] = model_diff
    
    # 4. Training parameters
    train1 = config1.get('training', {})
    train2 = config2.get('training', {})
    
    training_params = ['epochs', 'batch_size', 'k_folds', 'device', 'deterministic']
    train_diff = {}
    for param in training_params:
        val1 = train1.get(param, 'N/A')
        val2 = train2.get(param, 'N/A')
        if val1 != val2:
            train_diff[param] = {exp1_name: val1, exp2_name: val2}
    
    # Optimizer parameters
    opt1 = train1.get('optimizer', {})
    opt2 = train2.get('optimizer', {})
    opt_params = ['type', 'lr', 'weight_decay']
    opt_diff = {}
    for param in opt_params:
        val1 = opt1.get(param, 'N/A')
        val2 = opt2.get(param, 'N/A')
        if val1 != val2:
            opt_diff[param] = {exp1_name: val1, exp2_name: val2}
    if opt_diff:
        train_diff['optimizer'] = opt_diff
    
    # Learning rate scheduler
    sched1 = train1.get('scheduler', {})
    sched2 = train2.get('scheduler', {})
    if sched1.get('type') != sched2.get('type'):
        train_diff['scheduler'] = {
            exp1_name: sched1.get('type', 'N/A'),
            exp2_name: sched2.get('type', 'N/A')
        }
    
    # Early stopping
    es1 = train1.get('early_stopping', {})
    es2 = train2.get('early_stopping', {})
    if es1.get('enabled') != es2.get('enabled'):
        train_diff['early_stopping'] = {
            exp1_name: f"enabled={es1.get('enabled', False)}",
            exp2_name: f"enabled={es2.get('enabled', False)}"
        }
    
    if train_diff:
        differences['training'] = train_diff
    
    # 5. Trainer type
    trainer1 = config1.get('trainer', {}).get('type', 'N/A')
    trainer2 = config2.get('trainer', {}).get('type', 'N/A')
    if trainer1 != trainer2:
        differences['trainer'] = {exp1_name: trainer1, exp2_name: trainer2}
    
    return differences


def format_config_value(value):
    """Format configuration value for display"""
    if isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return "N/A"
    else:
        return str(value)


def plot_comparison(
    common_subjects: Set[int],
    acc1: Dict[int, float],
    acc2: Dict[int, float],
    exp1_name: str,
    exp2_name: str,
    dataset_name: str,
    differences: Dict,
    output_path: str = None
):
    """Plot comparison bar chart"""
    common_subjects = sorted(list(common_subjects))
    
    # Extract accuracies
    acc1_values = [acc1[s] for s in common_subjects]
    acc2_values = [acc2[s] for s in common_subjects]
    
    # Calculate differences
    diff_values = [acc2[s] - acc1[s] for s in common_subjects]
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.25)
    
    # Main title
    fig.suptitle(
        f'Experiment Comparison: {exp1_name} vs {exp2_name}\nDataset: {dataset_name}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # === Subplot 1: Accuracy Comparison Bar Chart ===
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(common_subjects))
    width = 0.35
    
    # Short names for legend
    short_name1 = exp1_name.split('_')[-1] if '_' in exp1_name else exp1_name[:10]
    short_name2 = exp2_name.split('_')[-1] if '_' in exp2_name else exp2_name[:10]
    
    bars1 = ax1.bar(x - width/2, acc1_values, width, label=short_name1, 
                    color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, acc2_values, width, label=short_name2,
                    color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Annotate values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=45)
    
    ax1.set_xlabel('Subject ID', fontsize=11)
    ax1.set_ylabel('Average Validation Accuracy', fontsize=11)
    ax1.set_title(f'Common Subjects Accuracy Comparison (n={len(common_subjects)})', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Sub-{s}' for s in common_subjects])
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # === Subplot 2: Accuracy Difference Chart ===
    ax2 = fig.add_subplot(gs[1, :])
    
    colors = ['#27ae60' if d > 0 else '#e67e22' for d in diff_values]
    bars3 = ax2.bar(x, diff_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Annotate values
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:+.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Subject ID', fontsize=11)
    ax2.set_ylabel('Accuracy Difference (Exp2 - Exp1)', fontsize=11)
    ax2.set_title('Per-Subject Accuracy Difference', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Sub-{s}' for s in common_subjects])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label=f'{short_name2} Better'),
        Patch(facecolor='#e67e22', edgecolor='black', label=f'{short_name1} Better')
    ]
    ax2.legend(handles=legend_elements, loc='upper left')
    
    # === Subplot 3: Configuration Differences Summary ===
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off')
    
    # Prepare difference text
    diff_text = "[Configuration Differences Summary]\n\n"
    
    if not differences:
        diff_text += "The two experiments have identical configurations!\n"
    else:
        # Model type
        if 'model_type' in differences:
            mt = differences['model_type']
            diff_text += f"[Model Type]\n  {exp1_name}: {mt[exp1_name]}\n  {exp2_name}: {mt[exp2_name]}\n\n"
        
        # Model parameters
        if 'model_args' in differences:
            diff_text += "[Model Arguments Differences]\n"
            for param, values in differences['model_args'].items():
                diff_text += f"  {param}: {format_config_value(values[exp1_name])} -> {format_config_value(values[exp2_name])}\n"
            diff_text += "\n"
        
        # Training parameters
        if 'training' in differences:
            train_diff = differences['training']
            diff_text += "[Training Parameters Differences]\n"
            for param, values in train_diff.items():
                if param == 'optimizer':
                    diff_text += f"  Optimizer:\n"
                    for opt_param, opt_values in values.items():
                        diff_text += f"    {opt_param}: {format_config_value(opt_values[exp1_name])} -> {format_config_value(opt_values[exp2_name])}\n"
                else:
                    diff_text += f"  {param}: {format_config_value(values[exp1_name])} -> {format_config_value(values[exp2_name])}\n"
            diff_text += "\n"
        
        # Other differences
        other_keys = [k for k in differences.keys() if k not in ['model_type', 'model_args', 'training']]
        if other_keys:
            diff_text += "[Other Differences]\n"
            for key in other_keys:
                values = differences[key]
                if isinstance(values, dict) and exp1_name in values:
                    diff_text += f"  {key}: {format_config_value(values[exp1_name])} -> {format_config_value(values[exp2_name])}\n"
    
    # Add statistics
    mean1 = np.mean(acc1_values)
    mean2 = np.mean(acc2_values)
    diff_mean = mean2 - mean1
    
    diff_text += f"\n[Statistical Summary]\n"
    diff_text += f"  {exp1_name} Mean Accuracy: {mean1:.2%}\n"
    diff_text += f"  {exp2_name} Mean Accuracy: {mean2:.2%}\n"
    diff_text += f"  Mean Difference: {diff_mean:+.2%}\n"
    diff_text += f"  Winner: {short_name2 if diff_mean > 0 else short_name1} (better average performance)\n"
    
    ax3.text(0.02, 0.98, diff_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Ensure output directory exists
    if output_path is None:
        output_dir = "results/comparisons"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"comparison_{exp1_name}_vs_{exp2_name}.png")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Comparison plot saved to: {output_path}")
    
    plt.close()
    
    return output_path


def print_summary(
    exp1_name: str,
    exp2_name: str,
    same_dataset: bool,
    dataset_name: str,
    common_subjects: Set[int],
    only_in_1: Set[int],
    only_in_2: Set[int],
    acc1: Dict[int, float],
    acc2: Dict[int, float],
    differences: Dict
):
    """Print comparison summary"""
    print("\n" + "=" * 80)
    print(f"Experiment Comparison Summary: {exp1_name} vs {exp2_name}")
    print("=" * 80)
    
    # Dataset info
    print(f"\n[Dataset Check]")
    if same_dataset:
        print(f"  [OK] Both experiments use the same dataset: {dataset_name}")
    else:
        print(f"  [WARN] Different datasets: {dataset_name}")
    
    # Subject info
    print(f"\n[Subject Information]")
    print(f"  Common subjects count: {len(common_subjects)}")
    print(f"  Common subjects: {sorted(list(common_subjects))}")
    if only_in_1:
        print(f"  Subjects only in Exp 1: {sorted(list(only_in_1))}")
    if only_in_2:
        print(f"  Subjects only in Exp 2: {sorted(list(only_in_2))}")
    
    # Accuracy statistics
    if common_subjects:
        common_list = sorted(list(common_subjects))
        acc1_values = [acc1[s] for s in common_list]
        acc2_values = [acc2[s] for s in common_list]
        
        print(f"\n[Accuracy Statistics (Common Subjects)]")
        print(f"  {exp1_name}:")
        print(f"    Mean Accuracy: {np.mean(acc1_values):.2%}")
        print(f"    Std Dev: {np.std(acc1_values):.2%}")
        print(f"    Max: {np.max(acc1_values):.2%} (Sub-{common_list[np.argmax(acc1_values)]})")
        print(f"    Min: {np.min(acc1_values):.2%} (Sub-{common_list[np.argmin(acc1_values)]})")
        
        print(f"  {exp2_name}:")
        print(f"    Mean Accuracy: {np.mean(acc2_values):.2%}")
        print(f"    Std Dev: {np.std(acc2_values):.2%}")
        print(f"    Max: {np.max(acc2_values):.2%} (Sub-{common_list[np.argmax(acc2_values)]})")
        print(f"    Min: {np.min(acc2_values):.2%} (Sub-{common_list[np.argmin(acc2_values)]})")
        
        diff = np.mean(acc2_values) - np.mean(acc1_values)
        print(f"\n  Mean Accuracy Difference: {diff:+.2%}")
        if diff > 0:
            print(f"  -> {exp2_name} has better average performance")
        elif diff < 0:
            print(f"  -> {exp1_name} has better average performance")
        else:
            print(f"  -> Both experiments have identical average performance")
    
    # Configuration differences
    print(f"\n[Configuration Differences]")
    if not differences:
        print("  The two experiments have identical configurations!")
    else:
        for key, value in differences.items():
            print(f"  {key}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        print(f"    {sub_key}:")
                        for k, v in sub_value.items():
                            print(f"      {k}: {v}")
                    else:
                        print(f"    {sub_key}: {sub_value}")
            else:
                print(f"    {value}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare results of two experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/compare_experiments.py XWStroke_ADFCNN_20260410_012331 DummyExperiment_XWFilterBankNet_20260410_000001
    python scripts/compare_experiments.py XWStroke_ADFCNN DummyExperiment_XWFilterBankNet --output my_comparison.png
        """
    )
    
    parser.add_argument('experiment1', help='Name of the first experiment')
    parser.add_argument('experiment2', help='Name of the second experiment')
    parser.add_argument('--output', '-o', help='Output image path (optional)')
    parser.add_argument('--no-plot', action='store_true', help='Do not display plot')
    
    args = parser.parse_args()
    
    try:
        # Load experiment information
        print(f"\nLoading experiment information...")
        config1, results1, full_name1 = get_experiment_info(args.experiment1)
        config2, results2, full_name2 = get_experiment_info(args.experiment2)
        
        print(f"  [OK] Experiment 1: {full_name1}")
        print(f"  [OK] Experiment 2: {full_name2}")
        
        # Check dataset
        same_dataset, dataset_info = check_same_dataset(config1, config2)
        dataset_name = config1.get('data', {}).get('dataset', 'Unknown')
        
        # Get subject information
        common_subjects, only_in_1, only_in_2 = get_common_subjects(results1, results2)
        
        if not common_subjects:
            print("\n[WARN] No common subjects between the two experiments. Cannot perform comparison.")
            return
        
        # Get accuracy data
        acc1 = get_subject_accuracies(results1)
        acc2 = get_subject_accuracies(results2)
        
        # Compare configurations
        differences = compare_configs(config1, config2, full_name1, full_name2)
        
        # Print summary
        print_summary(
            full_name1, full_name2,
            same_dataset, dataset_name,
            common_subjects, only_in_1, only_in_2,
            acc1, acc2, differences
        )
        
        # Plot comparison
        if not args.no_plot:
            plot_comparison(
                common_subjects, acc1, acc2,
                full_name1, full_name2,
                dataset_name, differences,
                args.output
            )
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
