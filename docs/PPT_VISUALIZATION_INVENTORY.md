# PPT Visualization Inventory

Complete list of all figures suitable for the research progress PPT, with usage recommendations.

---

## 🔴 NEW — Generated for This Report (Recommended Priority)

All figures are in `docs/figures/` with 300 DPI, white background, English text.

| # | Filename | Slide | Description | Recommendation |
|:---|:---|:---|:---|:---|
| 1 | `fig_competitive_landscape.png` | Slide 4 | 2D scatter: Protocol Strictness vs Accuracy, showing Bun24/Bun25/Kav24b/Wan26h/Ours | **Must-use**. This is your defensive weapon against the "97.43%" question. |
| 2 | `fig_stratified_waterfall.png` | Slide 7 | Waterfall bar chart showing Task C vs A differences across 4 stratification dimensions | **Must-use**. Instantly shows "who wins, who loses." |
| 3 | `fig_dann_results.png` | Slide 10 | Bar chart: Baseline vs DANN(dlw=0.3) vs DANN(dlw=1.0) on CleanC | **Must-use**. Turns negative results into a narrative strength ("we tried, we learned, we pivoted"). |
| 4 | `fig_task_comparison.png` | Slide 6 | Bar chart with error bars: Task A/B/C on Full-50 | **Must-use**. Clean, publication-quality baseline comparison. |
| 5 | `fig_action_timeline.png` | Slide 13 | Gantt-style timeline of next 2-4 weeks actions | **Recommended**. Makes plan concrete and visual. |

---

## 🟡 EXISTING — From `results/` Directory (Selective Use)

| # | Path | Slide | Description | Recommendation |
|:---|:---|:---|:---|:---|
| 6 | `results/comparisons/fig_bar_comparison.png` | Backup | Bar comparison of Task A vs B across subjects | **Skip** — use `fig_task_comparison.png` instead (cleaner). |
| 7 | `results/comparisons/fig_histogram_diff.png` | Appendix | Histogram of per-subject differences (Task B − A) | **Optional appendix**. Shows distribution of improvements. |
| 8 | `results/comparisons/fig_scatter_comparison.png` | Backup | Scatter plot: Task A vs Task B per subject | **Skip** — use stratified waterfall instead. |
| 9 | `results/comparisons/fig_stratified_nihss.png` | Backup | NIHSS stratified bar chart | **Skip** — use `fig_stratified_waterfall.png` instead. |
| 10 | `results/comparisons/fig_stratified_paralysis.png` | Backup | Paralysis side stratified bar chart | **Skip** — use `fig_stratified_waterfall.png` instead. |
| 11 | `results/comparisons_aligned/fig_alignment_effect.png` | Slide 8 | Alignment effect visualization (Task C vs A) | **Consider** — if you want a scatter version of the Task C effect. |
| 12 | `results/schemes/scheme_comparison_bar.png` | Backup | Scheme comparison bar chart | **Skip** — Scheme C report is incomplete; only Full-50 baseline shown. |
| 13 | `results/schemes/scheme_per_subject_distribution.png` | Backup | Per-subject accuracy distribution by scheme | **Skip** — same reason as above. |
| 14 | `results/stratified/fig_li_distribution.png` | Appendix | Lateralization Index distribution histogram | **Optional appendix**. Shows neurophysiological heterogeneity. |
| 15 | `results/stratified/fig_li_vs_accuracy.png` | Appendix | Scatter: LI vs LOSO accuracy | **Optional appendix**. Supports the "preserved lateralization → better alignment response" story. |
| 16 | `results/stratified/fig_stratum_duration.png` | Backup | Duration stratum distribution | **Skip** — use waterfall instead. |
| 17 | `results/stratified/fig_stratum_stroke_location.png` | Backup | Stroke location stratum distribution | **Skip** — use waterfall instead. |
| 18 | `results/dann_cleanc_gridsearch/gridsearch_heatmap.png` | Appendix | DANN grid search heatmap (dlw × γ) | **Optional appendix**. Shows most configs failed; only 2 completed. |
| 19 | `results/subject_selection_evidence/subject_improvement_predictors.png` | Appendix | Predictors of subject-level improvement | **Optional appendix**. Supports stratified findings. |

---

## 🔵 EXISTING — From `experiments/` Directory (Mostly Too Granular)

> **General recommendation:** Most experiment-level visualizations (per-subject training history, per-subject LOSO curves) are too granular for a 15-minute PPT. Use only if specifically asked for diagnostic detail.

| # | Path Pattern | Count | Description | Recommendation |
|:---|:---|:---|:---|:---|
| 20 | `experiments/*/visualizations/subject_*/training_history.png` | ~200 | Per-subject training loss/accuracy curves | **Skip for main PPT**. Use only if asked "how does training look?" |
| 21 | `experiments/*/visualizations/subject_*/streaming_loso_history.png` | ~100 | Per-subject LOSO training curves | **Skip for main PPT**. |
| 22 | `experiments/*/visualizations/subject_comparison_*.png` | ~5 | Cross-subject comparison summary | **Skip** — older format, superseded by `results/` figures. |
| 23 | `experiments/XWStroke_CleanC_*/visualizations/streaming_loso_comparison.png` | 2 | CleanC subset LOSO comparison | **Skip** — use new figures instead. |
| 24 | `experiments/xwstroke_streaming_loso/visualizations/streaming_loso_comparison.png` | 1 | Full-50 streaming LOSO comparison | **Skip** — use new figures instead. |

---

## 📋 Suggested Slide-to-Figure Mapping

| Slide | Title | Primary Figure | Backup / Appendix |
|:---|:---|:---|:---|
| 1 | Title | None | — |
| 2 | Clinical Motivation | None (text + maybe one schematic) | — |
| 3 | Dataset & Design | None (text + table) | — |
| 4 | **Competitive Landscape** | `fig_competitive_landscape.png` | — |
| 5 | Benchmarking Table | None (table) | — |
| 6 | **Full-Cohort Baseline** | `fig_task_comparison.png` | — |
| 7 | **Stratified Analysis** | `fig_stratified_waterfall.png` | `fig_li_vs_accuracy.png` (appendix) |
| 8 | CleanC Mechanism | None (table) | `fig_alignment_effect.png` |
| 9 | Subset-20 Design | None (table + maybe stratification matrix) | — |
| 10 | **DANN Results** | `fig_dann_results.png` | `gridsearch_heatmap.png` (appendix) |
| 11 | Route B Proposal | None (schematic/diagram) | — |
| 12 | Defensive Q&A | None (text) | — |
| 13 | **Action Plan** | `fig_action_timeline.png` | — |
| 14 | Risk Assessment | None (table) | — |
| 15 | Conclusion | None (text) | — |
| App. A | Detailed Stats | None (table) | — |
| App. B | DANN Diagnostics | `gridsearch_heatmap.png` | Per-subject loss curves |
| App. C | Literature Audit | None (table) | — |
| App. D | LI Analysis | `fig_li_distribution.png` | `fig_li_vs_accuracy.png` |

---

## 🎨 Design Notes for PPT Assembly

### Color Scheme (Consistent Across All New Figures)
- **Baseline / Reference:** Gray (`#757575`, `#bdbdbd`)
- **Task B (Moderate):** Light blue (`#90caf9`)
- **Task C / Ours:** Red (`#d32f2f`)
- **Benefited:** Green (`#4caf50`)
- **Harmed:** Red (`#f44336`)
- **Lenient protocol (others):** Gray (`#9e9e9e`)
- **Strict protocol (fair baseline):** Blue (`#1976d2`)

### Font
- All new figures use sans-serif (DejaVu Sans / Arial equivalent)
- Font size optimized for projection readability

### Aspect Ratios
- Most figures: ~16:10 (standard widescreen PPT)
- Action timeline: wide panoramic
- All exported at 300 DPI — can be resized without quality loss

---

## 📝 Figure Regeneration

If you need to modify any figure, run:

```bash
conda activate BCI310
python docs/figures/generate_ppt_figures.py
```

All figures will be regenerated in `docs/figures/`.
