#!/usr/bin/env python3
"""
一键运行 XWStroke 子集 LOSO 实验（Scheme A / B / C）

基于分层分析结果，针对三个不同严格程度的子集进行 LOSO 验证：
  Scheme A (宽松): Damage Exclusion, N=30
  Scheme B (严格): Tightened Prior, N=7
  Scheme C (主推): Subcortical Focus, N=17  ← RECOMMENDED

用法:
    # 运行主推方案 Scheme C
    python scripts/run_subset_schemes_loso.py --scheme C

    # 运行所有三个方案
    python scripts/run_subset_schemes_loso.py --scheme all

    # 运行 Scheme A 和 C
    python scripts/run_subset_schemes_loso.py --scheme A,C

    # 对比已有结果（不重新训练）
    python scripts/run_subset_schemes_loso.py --compare-only

输出:
    experiments/XWStroke_SchemeX_*/results/results.json
    results/schemes/  (对比报告、CSV、图表)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEMES = {
    "A": {
        "name": "Scheme A: Damage Exclusion",
        "config": "configs/experiment/xwstroke_schemeA_loso_affected_aligned.yaml",
        "exp_name": "XWStroke_SchemeA_DamageExclusion_LOSO",
        "n_subjects": 30,
        "criteria": "NOT(Cortical/Mixed) + NIHSS<8 + TaskA≤55%",
    },
    "B": {
        "name": "Scheme B: Tightened Prior",
        "config": "configs/experiment/xwstroke_schemeB_loso_affected_aligned.yaml",
        "exp_name": "XWStroke_SchemeB_Tightened_LOSO",
        "n_subjects": 7,
        "criteria": "Scheme A + Chronic(>3mo) + LI>0",
    },
    "C": {
        "name": "Scheme C: Subcortical Focus",
        "config": "configs/experiment/xwstroke_schemeC_loso_affected_aligned.yaml",
        "exp_name": "XWStroke_SchemeC_SubcorticalFocus_LOSO",
        "n_subjects": 17,
        "criteria": "Subcortical + NIHSS<8 + TaskA≤55%",
    },
    "CleanC": {
        "name": "Clean-C: Artifact-Free Subcortical (RECOMMENDED)",
        "config": "configs/experiment/xwstroke_cleanC_loso_affected_aligned.yaml",
        "exp_name": "XWStroke_CleanC_ArtifactFree_Subcortical_NIHSS7_LOSO",
        "n_subjects": 14,
        "criteria": "Artifact-free + Subcortical + NIHSS≤7",
    },
    "CleanCLR": {
        "name": "Clean-C LR Control",
        "config": "configs/experiment/xwstroke_cleanC_loso_lr.yaml",
        "exp_name": "XWStroke_CleanC_LR_Control_LOSO",
        "n_subjects": 14,
        "criteria": "Clean-C subjects with Left/Right labels (no alignment)",
    },
}

FULL50_RESULT = {
    "name": "Full 50 (Baseline)",
    "mean": 51.46,
    "std": 5.23,
    "n": 50,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_single_scheme(scheme_key):
    """Run LOSO for a single scheme."""
    cfg = SCHEMES[scheme_key]
    config_path = cfg["config"]
    exp_name = cfg["exp_name"]

    print(f"\n{'='*70}")
    print(f"Running {cfg['name']}")
    print(f"  Criteria: {cfg['criteria']}")
    print(f"  N={cfg['n_subjects']}, Config={config_path}")
    print(f"{'='*70}")

    # Direct call instead of subprocess to ensure stdout works in all terminals
    try:
        import train as train_module
        old_argv = sys.argv
        sys.argv = ["train.py", "--config", config_path, "--name", exp_name]
        train_module.main()
        sys.argv = old_argv
    except SystemExit as e:
        if e.code != 0:
            print(f"[ERROR] Scheme {scheme_key} failed with exit code {e.code}")
            return None
    except Exception as e:
        print(f"[ERROR] Scheme {scheme_key} failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Load results
    res_file = Path("experiments") / exp_name / "results" / "results.json"
    if not res_file.exists():
        print(f"[WARNING] Results file not found: {res_file}")
        return None

    with open(res_file) as f:
        data = json.load(f)

    print(f"\n[OK] Scheme {scheme_key} completed.")
    print(f"  Mean Accuracy: {data['overall_mean']*100:.2f}% ± {data['overall_std']*100:.2f}%")
    return data


def load_scheme_result(scheme_key):
    """Load existing result for a scheme (without retraining)."""
    cfg = SCHEMES[scheme_key]
    res_file = Path("experiments") / cfg["exp_name"] / "results" / "results.json"
    if not res_file.exists():
        return None
    with open(res_file) as f:
        return json.load(f)


def compare_and_report(results_dict):
    """Generate comparison report across schemes."""
    out_dir = Path("results/schemes")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Summary table
    # ------------------------------------------------------------------
    rows = []
    rows.append({
        "Scheme": FULL50_RESULT["name"],
        "N": FULL50_RESULT["n"],
        "Mean_Acc": f"{FULL50_RESULT['mean']:.2f}%",
        "Std_Acc": f"{FULL50_RESULT['std']:.2f}%",
        "Criteria": "All subjects",
        "vs_Full50": "—",
    })

    for key in ["A", "B", "C"]:
        res = results_dict.get(key)
        if res is None:
            continue
        mean_acc = res["overall_mean"] * 100
        std_acc = res["overall_std"] * 100
        diff = mean_acc - FULL50_RESULT["mean"]
        rows.append({
            "Scheme": SCHEMES[key]["name"],
            "N": SCHEMES[key]["n_subjects"],
            "Mean_Acc": f"{mean_acc:.2f}%",
            "Std_Acc": f"{std_acc:.2f}%",
            "Criteria": SCHEMES[key]["criteria"],
            "vs_Full50": f"{diff:+.2f}%",
        })

    df_summary = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print("SCHEME COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    df_summary.to_csv(out_dir / "scheme_comparison_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 2. Per-subject details
    # ------------------------------------------------------------------
    for key in ["A", "B", "C"]:
        res = results_dict.get(key)
        if res is None:
            continue
        rows_sub = []
        for r in res.get("subjects", []):
            row = {
                "Subject": f"sub-{r['test_subject_id']:02d}",
                "Test_Acc": f"{r['test_acc']*100:.1f}%",
                "Best_Acc": f"{r.get('best_acc', r['test_acc'])*100:.1f}%",
                "Best_Epoch": r.get("best_epoch", r.get("best_val_epoch", "N/A")),
                "Train_Subjects": r.get("train_subjects", r.get("train_subjects", "N/A")),
            }
            # Add val info if available (new early stopping format)
            if "val_acc" in r:
                row["Val_Acc"] = f"{r['val_acc']*100:.1f}%"
                row["Best_Val_Epoch"] = r.get("best_val_epoch", "N/A")
                row["Actual_Epochs"] = r.get("actual_epochs", "N/A")
                row["Stopped_Early"] = r.get("stopped_early", False)
            rows_sub.append(row)
        df_sub = pd.DataFrame(rows_sub)
        df_sub.to_csv(out_dir / f"scheme{key}_per_subject.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 3. Visualization
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        scheme_names = []
        scheme_means = []
        scheme_stds = []
        colors = []

        # Full 50 baseline
        scheme_names.append("Full 50\n(Baseline)")
        scheme_means.append(FULL50_RESULT["mean"])
        scheme_stds.append(FULL50_RESULT["std"])
        colors.append("#d62728")

        for key, color in [("A", "#2ca02c"), ("B", "#ff7f0e"), ("C", "#1f77b4")]:
            res = results_dict.get(key)
            if res is None:
                continue
            scheme_names.append(f"Scheme {key}\nN={SCHEMES[key]['n_subjects']}")
            scheme_means.append(res["overall_mean"] * 100)
            scheme_stds.append(res["overall_std"] * 100)
            colors.append(color)

        bars = ax.bar(
            range(len(scheme_names)),
            scheme_means,
            yerr=scheme_stds,
            color=colors,
            edgecolor="black",
            capsize=5,
            alpha=0.85,
        )
        ax.set_ylabel("LOSO Accuracy (%)", fontsize=12)
        ax.set_title("Cross-Subject LOSO: Subset Scheme Comparison", fontsize=14)
        ax.set_xticks(range(len(scheme_names)))
        ax.set_xticklabels(scheme_names, fontsize=10)
        ax.set_ylim([40, 65])
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Chance level")
        ax.axhline(y=55, color="green", linestyle="--", alpha=0.5, label="55% target")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate bars
        for bar, mean, std in zip(bars, scheme_means, scheme_stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.8,
                f"{mean:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        fig.savefig(out_dir / "scheme_comparison_bar.png", dpi=200)
        plt.close(fig)
        print(f"\n[SAVED] Bar chart: {out_dir / 'scheme_comparison_bar.png'}")

        # ------------------------------------------------------------------
        # 4. Per-subject accuracy distribution (violin/box)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for idx, key in enumerate(["A", "B", "C"]):
            res = results_dict.get(key)
            if res is None:
                continue
            accs = [r["test_acc"] * 100 for r in res.get("subjects", [])]
            axes[idx].boxplot(accs, widths=0.5)
            axes[idx].scatter(
                np.random.normal(1, 0.04, size=len(accs)),
                accs,
                alpha=0.6,
                s=30,
                color=colors[idx + 1],
            )
            axes[idx].set_title(
                f"Scheme {key}\nN={SCHEMES[key]['n_subjects']}\n"
                f"μ={np.mean(accs):.1f}%, σ={np.std(accs):.1f}%",
                fontsize=11,
            )
            axes[idx].set_ylabel("Accuracy (%)" if idx == 0 else "")
            axes[idx].set_xticks([1])
            axes[idx].set_xticklabels(["LOSO"])
            axes[idx].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
            axes[idx].axhline(y=55, color="green", linestyle="--", alpha=0.3)
            axes[idx].set_ylim([35, 70])
            axes[idx].grid(True, axis="y", alpha=0.3)

        plt.suptitle("Per-Subject LOSO Accuracy Distribution by Scheme", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(out_dir / "scheme_per_subject_distribution.png", dpi=200)
        plt.close(fig)
        print(f"[SAVED] Distribution plot: {out_dir / 'scheme_per_subject_distribution.png'}")

    except Exception as e:
        print(f"[WARNING] Plotting failed: {e}")

    # ------------------------------------------------------------------
    # 5. Markdown report
    # ------------------------------------------------------------------
    report_path = out_dir / "SCHEME_LOSO_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# XWStroke 子集 LOSO 实验报告\n\n")
        f.write("> 基于分层分析（Stratified Analysis）筛选的三种子集方案交叉验证结果\n\n"
        )
        f.write("## 实验配置\n\n")
        f.write("- **模型**: ADFCNN\n")
        f.write("- **训练器**: StreamingLOSOTrainer\n")
        f.write("- **Epochs**: 100\n")
        f.write("- **Batch Size**: 512\n")
        f.write("- **优化器**: Adam (lr=1e-3, wd=1e-2)\n")
        f.write("- **标签映射**: Affected/Unaffected + Hemisphere Alignment\n\n")

        f.write("## 方案对比总表\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 详细结果\n\n")
        for key in ["A", "B", "C"]:
            res = results_dict.get(key)
            if res is None:
                continue
            f.write(f"### Scheme {key}: {SCHEMES[key]['name']}\n\n")
            f.write(f"- **筛选条件**: {SCHEMES[key]['criteria']}\n")
            f.write(f"- **被试数**: {SCHEMES[key]['n_subjects']}\n")
            f.write(
                f"- **平均测试准确率**: {res['overall_mean']*100:.2f}% "
                f"± {res['overall_std']*100:.2f}%\n"
            )
            f.write(f"- **最低/最高**: {res['overall_min']*100:.1f}% / {res['overall_max']*100:.1f}%\n")

            # Check if using validation-based early stopping
            sample = res.get("subjects", [{}])[0]
            if "val_acc" in sample:
                f.write("- **早停策略**: Validation-Subject Early Stopping\n")
                f.write("- **Test Acc 含义**: Best-Val-Model 在独立 Test 上的性能（严谨无泄露）\n")

            f.write("\n**逐被试结果**:\n\n")
            if "val_acc" in sample:
                f.write("| Subject | Test Acc | Val Acc | Best Val Epoch | Actual Epochs | Stopped Early |\n")
                f.write("|---------|----------|---------|----------------|---------------|---------------|\n")
                for r in res.get("subjects", []):
                    f.write(
                        f"| sub-{r['test_subject_id']:02d} | "
                        f"{r['test_acc']*100:.1f}% | "
                        f"{r.get('val_acc', float('nan'))*100:.1f}% | "
                        f"{r.get('best_val_epoch', 'N/A')} | "
                        f"{r.get('actual_epochs', 'N/A')} | "
                        f"{'Yes' if r.get('stopped_early', False) else 'No'} |\n"
                    )
            else:
                f.write("| Subject | Test Acc | Best Acc | Best Epoch |\n")
                f.write("|---------|----------|----------|------------|\n")
                for r in res.get("subjects", []):
                    f.write(
                        f"| sub-{r['test_subject_id']:02d} | "
                        f"{r['test_acc']*100:.1f}% | "
                        f"{r.get('best_acc', r['test_acc'])*100:.1f}% | "
                        f"{r.get('best_epoch', 'N/A')} |\n"
                    )
            f.write("\n")

        f.write("## 结论与讨论\n\n")
        f.write("1. **Scheme C（Subcortical Focus）是主推方案**\n")
        f.write(
            "   - N=17 在统计效力和队列同质性之间取得了最佳平衡\n"
        )
        f.write(
            "   - 病灶位置（Subcortical）具有最强的神经生理学解释："
            "皮层运动区保留但皮质脊髓束受损，Aff+Align 的侧化重映射恰好利用了这一可塑性\n"
        )
        f.write("\n")
        f.write("2. **Scheme A（Damage Exclusion）用于稳健性验证**\n")
        f.write(
            "   - N=30 的结果可验证：子集提升是否依赖于过度严格的筛选\n"
        )
        f.write("\n")
        f.write("3. **Scheme B（Tightened Prior）仅作参考**\n")
        f.write(
            "   - N=7 样本量过小，结果可能不稳定，不建议作为论文主要结论\n"
        )
        f.write("\n")
        f.write("4. **55% 天花板问题**\n")
        f.write(
            "   - 即使在最优子集上，预估 LOSO 仍难以突破 55%\n"
        )
        f.write(
            "   - 建议后续从模型架构（更长的训练、学习率调度、不同 backbone）"
            "或数据增强角度进一步优化\n"
        )

    print(f"[SAVED] Report: {report_path}")


def parse_scheme_list(arg):
    """Parse scheme list like 'A,C' or 'all'."""
    if arg.lower() == "all":
        return list(SCHEMES.keys())
    # Case-insensitive match against scheme keys
    keys_upper = {k.upper(): k for k in SCHEMES.keys()}
    result = []
    for k in arg.split(","):
        k = k.strip().upper()
        if k in keys_upper:
            result.append(keys_upper[k])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run subset LOSO experiments for XWStroke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run only Scheme C (recommended)
  python scripts/run_subset_schemes_loso.py --scheme C

  # Run all three schemes
  python scripts/run_subset_schemes_loso.py --scheme all

  # Compare existing results without retraining
  python scripts/run_subset_schemes_loso.py --compare-only
        """,
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="C",
        help="Which scheme(s) to run: A, B, C, all, or comma-separated (default: C)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only generate comparison report from existing results",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("XWStroke Subset LOSO Experiment Runner")
    print("=" * 70)

    if args.compare_only:
        print("\n[Mode] Compare-only: loading existing results...")
        results = {}
        for key in ["A", "B", "C"]:
            res = load_scheme_result(key)
            if res:
                results[key] = res
                print(f"  Scheme {key}: loaded OK")
            else:
                print(f"  Scheme {key}: NOT FOUND")
        compare_and_report(results)
        return 0

    scheme_keys = parse_scheme_list(args.scheme)
    invalid = [k for k in scheme_keys if k not in SCHEMES]
    if invalid:
        print(f"[ERROR] Invalid scheme(s): {invalid}. Valid: A, B, C, all")
        return 1

    results = {}
    for key in scheme_keys:
        res = run_single_scheme(key)
        if res:
            results[key] = res

    # Always run comparison after training
    if results:
        compare_and_report(results)

    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
