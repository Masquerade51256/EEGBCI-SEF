"""
Reproduce paired statistical tests for Full-50 LOSO comparison.

Supports comparing any two task columns from the comparison CSV.

Usage:
    # Default: TaskC_Aligned vs TaskA_LR (Aff+Align effect)
    python scripts/reproduce_full50_statistics.py

    # Compare TaskB (affected, no alignment) vs TaskA
    python scripts/reproduce_full50_statistics.py \
        --csv results/comparisons/full50_loso_comparison.csv \
        --col_a TaskA_LR --col_b TaskB_Aff

Outputs:
    - JSON file with paired t-test, Cohen's d, Wilcoxon, 95% CI, permutation test
    - Console print of key statistics
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def cohens_d_paired(x, y):
    """Compute Cohen's d for paired samples (using std of differences)."""
    diff = np.array(x) - np.array(y)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return mean_diff / std_diff


def paired_t_test_with_ci(x, y, confidence=0.95):
    """Paired t-test with confidence interval for the mean difference (y - x)."""
    diff = np.array(y, dtype=float) - np.array(x, dtype=float)
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se = std_diff / math.sqrt(n)
    df = n - 1

    t_stat, p_val = stats.ttest_rel(y, x)

    t_crit = stats.t.ppf((1 + confidence) / 2.0, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se": se,
        "df": df,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
    }


def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test."""
    stat, p_val = stats.wilcoxon(x, y, zero_method="wilcox")
    return {"statistic": float(stat), "p_value": float(p_val)}


def one_sample_proportion_test(improved_count, total, p0=0.5):
    """Exact binomial test: H1: prop > 0.5."""
    p_val = stats.binom_test(improved_count, total, p0, alternative="greater")
    prop = improved_count / total
    se = math.sqrt(p0 * (1 - p0) / total)
    ci_low = max(0.0, prop - 1.96 * se)
    ci_high = min(1.0, prop + 1.96 * se)
    return {
        "improved_count": improved_count,
        "total": total,
        "proportion": prop,
        "p_value_greater_than_half": float(p_val),
        "ci_95_approx_low": float(ci_low),
        "ci_95_approx_high": float(ci_high),
    }


def permutation_test_paired(x, y, n_permutations=10000, seed=42):
    """Permutation test for paired differences (random sign flip)."""
    rng = np.random.RandomState(seed)
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    observed_mean = np.mean(diff)

    perm_means = []
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_diff = diff * signs
        perm_means.append(np.mean(perm_diff))

    perm_means = np.array(perm_means)
    p_val = np.mean(np.abs(perm_means) >= np.abs(observed_mean))
    return {
        "observed_mean_diff": float(observed_mean),
        "n_permutations": n_permutations,
        "p_value_two_tailed": float(p_val),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Full-50 LOSO statistical tests"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/comparisons_aligned/aligned_vs_lr_comparison.csv",
        help="Path to comparison CSV",
    )
    parser.add_argument(
        "--col_a",
        type=str,
        default="TaskA_LR",
        help="Column name for condition A (baseline)",
    )
    parser.add_argument(
        "--col_b",
        type=str,
        default="TaskC_Aligned",
        help="Column name for condition B (treatment)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (auto-derived from col_b if not set)",
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=10000,
        help="Number of permutations",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    subjects = []
    task_a = []
    task_b = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.col_a not in row or args.col_b not in row:
                print(f"Error: Columns {args.col_a} or {args.col_b} not found in CSV.")
                print(f"Available columns: {list(row.keys())}")
                sys.exit(1)
            subjects.append(row["Subject"])
            # Values in CSV are percentages like 50.0, meaning 50%
            a = float(row[args.col_a]) / 100.0
            b = float(row[args.col_b]) / 100.0
            task_a.append(a)
            task_b.append(b)

    task_a = np.array(task_a)
    task_b = np.array(task_b)
    diffs = task_b - task_a

    n = len(subjects)
    improved_count = int(np.sum(diffs > 0))
    worsened_count = int(np.sum(diffs < 0))
    unchanged_count = int(np.sum(diffs == 0))

    desc = {
        "n": n,
        "task_a_mean": float(np.mean(task_a)),
        "task_a_std": float(np.std(task_a, ddof=1)),
        "task_b_mean": float(np.mean(task_b)),
        "task_b_std": float(np.std(task_b, ddof=1)),
        "diff_mean": float(np.mean(diffs)),
        "diff_std": float(np.std(diffs, ddof=1)),
        "diff_median": float(np.median(diffs)),
        "diff_min": float(np.min(diffs)),
        "diff_max": float(np.max(diffs)),
        "improved_count": improved_count,
        "worsened_count": worsened_count,
        "unchanged_count": unchanged_count,
    }

    ttest = paired_t_test_with_ci(task_a, task_b)
    d = cohens_d_paired(task_b, task_a)
    d_abs = abs(d)
    d_desc = (
        "negligible" if d_abs < 0.2 else
        "small" if d_abs < 0.5 else
        "medium" if d_abs < 0.8 else "large"
    )
    wilcox = wilcoxon_signed_rank(task_b, task_a)
    prop_test = one_sample_proportion_test(improved_count, n, p0=0.5)
    perm = permutation_test_paired(task_b, task_a, n_permutations=args.n_perm)

    report = {
        "comparison": f"{args.col_b} vs {args.col_a}",
        "csv_source": str(csv_path),
        "descriptive": desc,
        "paired_t_test": ttest,
        "cohens_d": {"value": float(d), "abs_value": d_abs, "description": d_desc},
        "wilcoxon_signed_rank": wilcox,
        "proportion_test": prop_test,
        "permutation_test": perm,
    }

    out_path = Path(args.out) if args.out else Path(
        f"results/comparisons/{args.col_b}_vs_{args.col_a}_stats.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"FULL-50 LOSO STATISTICAL REPORT")
    print(f"Comparison: {args.col_b}  vs  {args.col_a}")
    print("=" * 65)
    print(f"N = {n}")
    print(f"{args.col_a}:        {desc['task_a_mean']:.2%} ± {desc['task_a_std']:.2%}")
    print(f"{args.col_b}:        {desc['task_b_mean']:.2%} ± {desc['task_b_std']:.2%}")
    # CI is for (B - A), same direction as diff_mean
    ci_low = min(ttest['ci_95_low'], ttest['ci_95_high'])
    ci_high = max(ttest['ci_95_low'], ttest['ci_95_high'])
    print(f"Mean diff:      {desc['diff_mean']:+.2%} [{ci_low:+.2%}, {ci_high:+.2%}]")
    print()
    print(f"Paired t-test:  t({ttest['df']}) = {ttest['t_statistic']:.3f},  p = {ttest['p_value']:.4f}")
    print(f"Cohen's d:      {d:.3f} ({d_desc})")
    print(f"Wilcoxon:       W = {wilcox['statistic']:.1f},  p = {wilcox['p_value']:.4f}")
    print(f"Permutation:    p = {perm['p_value_two_tailed']:.4f}  ({args.n_perm:,} perm)")
    print()
    print(f"Improved:       {improved_count}/{n} ({improved_count/n:.1%})")
    print(f"Binomial test:  p = {prop_test['p_value_greater_than_half']:.4f}  (H1: prop > 0.5)")
    print()
    print(f"Saved to: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
