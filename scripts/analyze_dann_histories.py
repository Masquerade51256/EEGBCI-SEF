"""
Analyze DANN training histories to diagnose failure modes.

Usage:
    python scripts/analyze_dann_histories.py --exp_dir experiments/XWStroke_Subset20_DANN_ParalysisSide_w0.05

Outputs:
    - Console summary of task loss / domain loss trends
    - PNG plot: task_loss vs domain_loss per fold
    - PNG plot: domain classifier accuracy trajectory (if available)
    - JSON report: diagnostic flags (e.g., "domain_loss_dominates", "task_loss_unstable")
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_single_experiment(exp_dir: Path):
    """Analyze one DANN experiment directory."""
    hist_path = exp_dir / "logs" / "training_histories.json"
    results_path = exp_dir / "results" / "results.json"
    
    if not hist_path.exists():
        print(f"[SKIP] History not found: {hist_path}")
        return None
    
    with open(hist_path, "r", encoding="utf-8") as f:
        all_histories = json.load(f)
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    n_folds = len(all_histories)
    report = {
        "experiment": exp_dir.name,
        "n_folds": n_folds,
        "overall_mean": results.get("overall_mean", None),
        "overall_std": results.get("overall_std", None),
        "folds": [],
        "diagnostic_flags": [],
    }
    
    # Per-fold analysis
    for fold in all_histories:
        sid = fold["test_subject_id"]
        h = fold["history"]
        
        task_loss = h.get("task_loss", [])
        domain_loss = h.get("domain_loss", [])
        train_acc = h.get("train_acc", [])
        test_acc = h.get("test_acc", [])
        grl_lambda = h.get("grl_lambda", [])
        
        n_epochs = len(task_loss)
        if n_epochs == 0:
            continue
        
        # Diagnostics
        flags = []
        
        # 1. Domain loss dominates?
        if len(domain_loss) > 5:
            avg_domain = np.mean(domain_loss[5:])
            avg_task = np.mean(task_loss[5:])
            if avg_domain > avg_task:
                flags.append("domain_loss_dominates")
        
        # 2. Task loss increasing? (forgetting)
        if len(task_loss) > 10:
            if np.mean(task_loss[-5:]) > np.mean(task_loss[:5]) * 1.2:
                flags.append("task_loss_increasing")
        
        # 3. Domain loss not decreasing? (adversarial failure)
        if len(domain_loss) > 10:
            if np.mean(domain_loss[-5:]) > np.mean(domain_loss[:5]) * 0.9:
                flags.append("domain_loss_not_decreasing")
        
        # 4. Test accuracy flat or decreasing?
        if len(test_acc) > 10:
            if np.mean(test_acc[-5:]) < np.mean(test_acc[:5]) * 0.95:
                flags.append("test_acc_decreasing")
        
        report["folds"].append({
            "test_subject_id": sid,
            "n_epochs": n_epochs,
            "final_task_loss": task_loss[-1] if task_loss else None,
            "final_domain_loss": domain_loss[-1] if domain_loss else None,
            "final_train_acc": train_acc[-1] if train_acc else None,
            "final_test_acc": test_acc[-1] if test_acc else None,
            "flags": flags,
        })
        
        report["diagnostic_flags"].extend(flags)
    
    # Aggregate flags
    flag_counts = {}
    for f in report["diagnostic_flags"]:
        flag_counts[f] = flag_counts.get(f, 0) + 1
    report["flag_counts"] = flag_counts
    
    return report, all_histories


def plot_histories(all_histories, exp_dir: Path):
    """Generate diagnostic plots."""
    n_folds = len(all_histories)
    if n_folds == 0:
        return
    
    # Plot 1: Task loss vs Domain loss across folds
    fig, axes = plt.subplots(2, min(5, n_folds), figsize=(20, 8), squeeze=False)
    for idx, fold in enumerate(all_histories[:5]):
        sid = fold["test_subject_id"]
        h = fold["history"]
        task_loss = h.get("task_loss", [])
        domain_loss = h.get("domain_loss", [])
        grl_lambda = h.get("grl_lambda", [])
        
        ax1 = axes[0, idx]
        ax1.plot(task_loss, label="Task Loss", color="blue")
        ax1.plot(domain_loss, label="Domain Loss", color="red")
        ax1.set_title(f"Sub-{sid}: Losses")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1, idx]
        ax2.plot(grl_lambda, label="GRL λ", color="green")
        ax2.set_title(f"Sub-{sid}: GRL Lambda")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("λ")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = exp_dir / "visualizations" / "dann_diagnostic_losses.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")
    
    # Plot 2: Summary bar chart of flags
    # (done in main after aggregating multiple experiments)


def print_report(report):
    """Pretty-print diagnostic report."""
    print("\n" + "=" * 60)
    print(f"DANN DIAGNOSTIC REPORT: {report['experiment']}")
    print("=" * 60)
    print(f"Overall Accuracy: {report['overall_mean']:.2%} ± {report['overall_std']:.2%}")
    print(f"Folds Analyzed: {report['n_folds']}")
    print("\nDiagnostic Flags:")
    for flag, count in report["flag_counts"].items():
        print(f"  {flag:30s} : {count} folds")
    
    print("\nPer-Fold Summary:")
    for fold in report["folds"]:
        sid = fold["test_subject_id"]
        flags = ", ".join(fold["flags"]) if fold["flags"] else "OK"
        print(f"  Sub-{sid:2d}: TaskLoss={fold['final_task_loss']:.3f}, "
              f"DomainLoss={fold['final_domain_loss']:.3f}, "
              f"TestAcc={fold['final_test_acc']:.3f}  [{flags}]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze DANN training histories")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--out_json", type=str, default="", help="Output JSON report path")
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    result = analyze_single_experiment(exp_dir)
    if result is None:
        sys.exit(1)
    
    report, histories = result
    print_report(report)
    plot_histories(histories, exp_dir)
    
    # Save JSON report
    out_path = Path(args.out_json) if args.out_json else exp_dir / "dann_diagnostic_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
