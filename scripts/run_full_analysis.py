"""
XWStroke LOSO 对照实验完整分析脚本

功能：统计检验、可视化、分层分析、生成报告

用法（基于已有10人结果）:
    conda activate BCI310
    python scripts/run_full_analysis.py

用法（指定自定义路径）:
    python scripts/run_full_analysis.py \
        --results-a experiments/XWStroke_ADFCNN_20260508_044352/results/results.json \
        --results-b experiments/XWStroke_ADFCNN_20260508_055459/results/results.json \
        --label-a "Left/Right" \
        --label-b "Affected/Unaffected"
"""

import sys, os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Setup Chinese font for Windows
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cohens_d(x, y):
    """Paired Cohen's d."""
    diff = np.array(y) - np.array(x)
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)


def perform_stats(acc_a, acc_b):
    """Perform paired t-test and related statistics."""
    diff = np.array(acc_b) - np.array(acc_a)
    t_stat, p_value = stats.ttest_rel(acc_b, acc_a)
    d = cohens_d(acc_a, acc_b)
    ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    
    return {
        'n': len(diff),
        'mean_a': np.mean(acc_a),
        'std_a': np.std(acc_a, ddof=1),
        'mean_b': np.mean(acc_b),
        'std_b': np.std(acc_b, ddof=1),
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff, ddof=1),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'improved_count': int(np.sum(diff > 0)),
        'unchanged_count': int(np.sum(diff == 0)),
        'worsened_count': int(np.sum(diff < 0)),
    }


def plot_scatter(subjects, acc_a, acc_b, label_a, label_b, output_path):
    """Scatter plot: Task A vs Task B per subject."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Diagonal reference line (y=x)
    min_val = min(min(acc_a), min(acc_b)) - 0.05
    max_val = max(max(acc_a), max(acc_b)) + 0.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='No change')
    
    # Points colored by improvement
    diff = np.array(acc_b) - np.array(acc_a)
    colors = ['#2ecc71' if d > 0 else '#e74c3c' if d < 0 else '#95a5a6' for d in diff]
    
    for i, (sid, a, b, c) in enumerate(zip(subjects, acc_a, acc_b, colors)):
        ax.scatter(a, b, c=c, s=150, edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(f'sub-{sid:02d}', (a, b), textcoords="offset points", 
                    xytext=(5, 5), fontsize=8, alpha=0.7)
    
    ax.set_xlabel(f'{label_a} Accuracy', fontsize=12)
    ax.set_ylabel(f'{label_b} Accuracy', fontsize=12)
    ax.set_title('LOSO Cross-Subject: Task A vs Task B', fontsize=14, fontweight='bold')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text
    mean_diff = np.mean(diff)
    p_val = stats.ttest_rel(acc_b, acc_a)[1]
    textstr = f'Mean Diff: {mean_diff*100:+.2f}%\np-value: {p_val:.4f}\nImproved: {np.sum(diff>0)}/{len(diff)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plot: {output_path}")


def plot_bar_comparison(subjects, acc_a, acc_b, label_a, label_b, output_path):
    """Bar chart comparing per-subject accuracy."""
    x = np.arange(len(subjects))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_a = ax.bar(x - width/2, acc_a, width, label=label_a, color='#3498db', edgecolor='black', alpha=0.8)
    bars_b = ax.bar(x + width/2, acc_b, width, label=label_b, color='#e67e22', edgecolor='black', alpha=0.8)
    
    # Highlight improvement
    diff = np.array(acc_b) - np.array(acc_a)
    for i, d in enumerate(diff):
        if d > 0:
            ax.plot([x[i]-width/2, x[i]+width/2], [acc_b[i]+0.01, acc_b[i]+0.01], 
                   'g-', linewidth=1.5)
            ax.annotate(f'+{d*100:.1f}%', (x[i], max(acc_a[i], acc_b[i])+0.02),
                       ha='center', fontsize=7, color='green', fontweight='bold')
    
    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_ylabel('LOSO Test Accuracy', fontsize=12)
    ax.set_title('Per-Subject LOSO Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'sub-{s:02d}' for s in subjects], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(acc_a), max(acc_b)) + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved bar chart: {output_path}")


def plot_stratified(df, group_col, output_path, title_suffix=""):
    """Box plot grouped by clinical feature."""
    if group_col not in df.columns:
        return
    
    groups = df[group_col].unique()
    if len(groups) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    data_a = [df[df[group_col]==g]['acc_a'].values for g in groups]
    data_b = [df[df[group_col]==g]['acc_b'].values for g in groups]
    
    bp1 = axes[0].boxplot(data_a, labels=groups, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#3498db')
    axes[0].set_title(f'Task A ({title_suffix})', fontsize=12)
    axes[0].set_ylabel('LOSO Accuracy')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    bp2 = axes[1].boxplot(data_b, labels=groups, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#e67e22')
    axes[1].set_title(f'Task B ({title_suffix})', fontsize=12)
    axes[1].set_ylabel('LOSO Accuracy')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    fig.suptitle(f'Stratified by {group_col}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved stratified plot: {output_path}")


def plot_histogram(diff, output_path):
    """Histogram of accuracy differences."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff]
    ax.bar(range(len(diff)), [d*100 for d in diff], color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=np.mean(diff)*100, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(diff)*100:+.2f}%')
    
    ax.set_xlabel('Subject Index', fontsize=12)
    ax.set_ylabel('Accuracy Difference (B - A) %', fontsize=12)
    ax.set_title('Per-Subject Improvement Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {output_path}")


def generate_report(stat_results, df_strata, output_path):
    """Generate Markdown report."""
    s = stat_results
    lines = [
        "# XWStroke LOSO 对照实验分析报告",
        "",
        "## 1. 总体统计",
        "",
        f"- **样本量**: N = {s['n']}",
        f"- **任务A (左右手)**: {s['mean_a']*100:.2f}% ± {s['std_a']*100:.2f}%",
        f"- **任务B (患健侧)**: {s['mean_b']*100:.2f}% ± {s['std_b']*100:.2f}%",
        f"- **平均差异**: {s['mean_diff']*100:+.2f}% (95% CI: [{s['ci_95_low']*100:+.2f}%, {s['ci_95_high']*100:+.2f}%])",
        f"- **配对t检验**: t = {s['t_statistic']:.3f}, p = {s['p_value']:.4f}",
        f"- **Cohen's d**: {s['cohens_d']:.3f} ({'大' if abs(s['cohens_d']) >= 0.8 else '中' if abs(s['cohens_d']) >= 0.5 else '小'}效应)",
        f"- **改善人数**: {s['improved_count']}/{s['n']} ({s['improved_count']/s['n']*100:.0f}%)",
        "",
        "## 2. 分层分析",
        "",
        df_strata.to_markdown(index=False),
        "",
        "## 3. 结论判读",
        "",
    ]
    
    if s['p_value'] < 0.05:
        if s['mean_diff'] > 0:
            lines.append(f"✅ **统计显著**: 任务B (患健侧) 显著优于任务A (左右手), p = {s['p_value']:.4f} < 0.05")
        else:
            lines.append(f"❌ **统计显著**: 任务A 显著优于任务B, p = {s['p_value']:.4f} < 0.05")
    elif s['p_value'] < 0.10:
        lines.append(f"⚠️ **边缘显著**: 任务B 有优于任务A 的趋势, p = {s['p_value']:.4f} < 0.10")
    else:
        lines.append(f"❌ **无显著差异**: p = {s['p_value']:.4f} > 0.10")
    
    if abs(s['cohens_d']) >= 0.8:
        lines.append(f"📊 **效应量大** (Cohen's d = {s['cohens_d']:.3f})，差异具有实际意义。")
    elif abs(s['cohens_d']) >= 0.5:
        lines.append(f"📊 **中等效应量** (Cohen's d = {s['cohens_d']:.3f})，差异有一定实际意义。")
    else:
        lines.append(f"📊 **小效应量** (Cohen's d = {s['cohens_d']:.3f})，差异幅度有限。")
    
    lines.append("")
    lines.append("---")
    lines.append("*Generated by scripts/run_full_analysis.py*")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='XWStroke LOSO Full Analysis')
    parser.add_argument('--results-a', type=str,
                        default='experiments/XWStroke_ADFCNN_20260508_044352/results/results.json',
                        help='Task A results JSON')
    parser.add_argument('--results-b', type=str,
                        default='experiments/XWStroke_ADFCNN_20260508_055459/results/results.json',
                        help='Task B results JSON')
    parser.add_argument('--label-a', type=str, default='Left/Right', help='Task A label')
    parser.add_argument('--label-b', type=str, default='Affected/Unaffected', help='Task B label')
    parser.add_argument('--participants', type=str,
                        default='src/datasets/21679035/participants.tsv',
                        help='Participants TSV')
    parser.add_argument('--output-dir', type=str, default='results/comparisons',
                        help='Output directory')
    args = parser.parse_args()
    
    print("="*65)
    print("XWStroke LOSO Full Analysis")
    print("="*65)
    
    # Load data
    res_a = load_results(args.results_a)
    res_b = load_results(args.results_b)
    participants = pd.read_csv(args.participants, sep='\t')
    participants['subject_id'] = participants['Participant_ID'].str.replace('sub-', '').astype(int)
    
    # Extract accuracy arrays
    subjects = [r['test_subject_id'] for r in res_a['subjects']]
    acc_a = np.array([r['test_acc'] for r in res_a['subjects']])
    acc_b = np.array([r['test_acc'] for r in res_b['subjects']])
    
    # Merge with clinical data
    df = pd.DataFrame({
        'subject_id': subjects,
        'acc_a': acc_a,
        'acc_b': acc_b,
        'diff': acc_b - acc_a,
    })
    df = df.merge(participants, on='subject_id', how='left')
    
    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Statistics
    print("\n[1/5] Statistical Analysis...")
    stat_results = perform_stats(acc_a, acc_b)
    print(f"  Task A: {stat_results['mean_a']*100:.2f}% ± {stat_results['std_a']*100:.2f}%")
    print(f"  Task B: {stat_results['mean_b']*100:.2f}% ± {stat_results['std_b']*100:.2f}%")
    print(f"  Mean Diff: {stat_results['mean_diff']*100:+.2f}%")
    print(f"  Paired t-test: t={stat_results['t_statistic']:.3f}, p={stat_results['p_value']:.4f}")
    print(f"  Cohen's d: {stat_results['cohens_d']:.3f}")
    print(f"  95% CI: [{stat_results['ci_95_low']*100:+.2f}%, {stat_results['ci_95_high']*100:+.2f}%]")
    print(f"  Improved: {stat_results['improved_count']}/{stat_results['n']}")
    
    # 2. Visualizations
    print("\n[2/5] Generating Visualizations...")
    plot_scatter(subjects, acc_a, acc_b, args.label_a, args.label_b,
                 out_dir / 'fig_scatter_comparison.png')
    plot_bar_comparison(subjects, acc_a, acc_b, args.label_a, args.label_b,
                        out_dir / 'fig_bar_comparison.png')
    plot_histogram(acc_b - acc_a, out_dir / 'fig_histogram_diff.png')
    plot_stratified(df, 'ParalysisSide', out_dir / 'fig_stratified_paralysis.png', 'Paralysis Side')
    
    # NIHSS level
    df['NIHSS_level'] = pd.cut(df['NIHSS'], bins=[0,3,7,20], 
                                labels=['Light(1-3)', 'Medium(4-7)', 'Severe(>=8)'],
                                include_lowest=True)
    plot_stratified(df, 'NIHSS_level', out_dir / 'fig_stratified_nihss.png', 'NIHSS')
    
    # 3. Stratified Analysis Table
    print("\n[3/5] Stratified Analysis...")
    
    def stratify_summary(df_sub):
        return pd.Series({
            'N': len(df_sub),
            'TaskA_Mean': f"{df_sub['acc_a'].mean()*100:.1f}%",
            'TaskB_Mean': f"{df_sub['acc_b'].mean()*100:.1f}%",
            'Diff_Mean': f"{df_sub['diff'].mean()*100:+.1f}%",
            'Improved': f"{(df_sub['diff'] > 0).sum()}/{len(df_sub)}",
        })
    
    strata_paralysis = df.groupby('ParalysisSide').apply(stratify_summary).reset_index()
    strata_nihss = df.groupby('NIHSS_level', observed=True).apply(stratify_summary).reset_index()
    strata_perf = df.assign(
        perf_a=pd.qcut(df['acc_a'], q=3, labels=['Low','Med','High'])
    ).groupby('perf_a', observed=True).apply(stratify_summary).reset_index()
    
    print("\n  By ParalysisSide:")
    print(strata_paralysis.to_string(index=False))
    print("\n  By NIHSS:")
    print(strata_nihss.to_string(index=False))
    print("\n  By TaskA Performance Level:")
    print(strata_perf.to_string(index=False))
    
    # Combine for report
    df_strata = pd.concat([
        strata_paralysis.assign(Stratum='ParalysisSide'),
        strata_nihss.assign(Stratum='NIHSS'),
        strata_perf.assign(Stratum='TaskA_Performance'),
    ], ignore_index=True)
    
    # 4. Generate Report
    print("\n[4/5] Generating Report...")
    generate_report(stat_results, df_strata, out_dir / 'ANALYSIS_REPORT.md')
    
    # 5. Save detailed CSV
    print("\n[5/5] Saving detailed data...")
    df[['subject_id','ParalysisSide','Handedness','NIHSS','MBI','mRS','Age','Duration',
        'acc_a','acc_b','diff']].to_csv(
        out_dir / 'analysis_detailed.csv', index=False, encoding='utf-8-sig')
    
    # Save stats JSON
    with open(out_dir / 'analysis_stats.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in stat_results.items()}, f, indent=2)
    
    print("\n" + "="*65)
    print("Analysis Complete!")
    print("="*65)
    print(f"Output files in: {out_dir}")
    print(f"  - fig_scatter_comparison.png")
    print(f"  - fig_bar_comparison.png")
    print(f"  - fig_histogram_diff.png")
    print(f"  - fig_stratified_paralysis.png")
    print(f"  - fig_stratified_nihss.png")
    print(f"  - ANALYSIS_REPORT.md")
    print(f"  - analysis_detailed.csv")
    print(f"  - analysis_stats.json")


if __name__ == '__main__':
    main()
