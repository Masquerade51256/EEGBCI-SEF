# EEGBCI-SEF Agent Context

> Quick-start context for agents joining this project. Last updated: 2026-05-20.

## 1. Project Identity

- **Name**: EEGBCI-SEF (EEG-BCI Standard Experiment Framework)
- **Goal**: A modular, config-driven Python framework for cross-subject EEG motor-imagery (MI) experiments, plus an IEEE BHI 2026 submission.
- **Language / Stack**: Python 3.10, PyTorch 2.11.0+cu128, MNE-Python 1.7, YAML configs.
- **Hardware note**: Mixed precision (FP16) is disabled on the RTX 5090 because it triggers `SIGILL`.
- **Entry point**: `python train.py --config configs/experiment/<name>.yaml`.

## 2. Active Paper / Submission

- **Venue**: IEEE BHI 2026 (full paper, ~6 pages).
- **Title**: *The Limits of Clinical Metadata for Cross-Subject Stroke Motor Imagery Decoding: Evidence from Anatomy-Aware Relabelling and Alignment*.
- **Thesis**: Clinical metadata (lesion location, stroke duration, NIHSS) are poor direct cues for cross-subject transfer in stroke MI-BCIs. Anatomy-aware relabelling and Euclidean Alignment do help some patients (especially subcortical lesions), but patient-specific gains require EEG-based validation proxies, not clinical labels.
- **Paper source**: `paper_ieee/main.tex` and `paper_ieee/figures/`. These are intentionally **git-ignored**; do not commit them.

## 3. Key Results (verified from scripts)

| Finding | Value |
|---|---|
| Best fixed pipeline | **Aff+EA**: 51.32% ± 4.08% |
| LR+noEA baseline | 49.40% ± 4.97% |
| LR+EA | 50.49% ± 6.26% |
| Aff+noEA | 49.55% ± 4.84% |
| Clinical-nearest validation selection | **49.16%** ± 5.89% (vs fixed: p = 0.030, d = −0.32) |
| Random validation selection | **50.71%** ± 5.05% (vs clinical-nearest: p = 0.067) |
| Oracle per-subject selection | **55.68%** ± 3.81% |
| Validation–test correlation | **r = 0.040, p = 0.574** |
| Subcortical benefit from Aff relabelling | d = 0.749, p = 0.013 |
| Subcortical benefit from EA | d = 0.679, p = 0.023 |
| Lesion-informed rule (subcortical → Aff+EA, others → LR+EA) | 51.95% vs fixed 51.32%, p = 0.418 |

## 4. Dataset

- **Dataset**: XWStroke (local short name) = Liu et al., *Scientific Data*, 2024, "An EEG motor imagery dataset for brain computer interface in acute stroke patients".
- **Size**: 50 stroke patients.
- **Task**: Left/right hand motor imagery, 40 trials per subject (20 left, 20 right).
- **Preprocessing in this work**: resampled to 250 Hz, bandpass 4–40 Hz, 30 scalp EEG channels, 2-s sliding windows with **0.5-s stride** (~200 windows/subject).
- **Metadata file**: `results/stratified/stratified_analysis_detailed.csv`.
- **Clinical characteristics**:
  - Affected side: left 28, right 22.
  - Duration: acute (≤3 mo) 27, chronic (>3 mo) 23.
  - Lesion location: subcortical 20, brainstem 18, mixed 8, cortical 4.
  - NIHSS: median 3, range 1–11.

## 5. Models & Pipelines

- **Backbone**: EEGNet (compact CNN for EEG). ShallowConvNet, DeepConvNet, and ADFCNN are also registered in `models/__init__.py` via adapter wrappers (raw-logit linear head, CrossEntropyLoss-compatible) and have full 4-pipeline LOSO results — see `results/comparisons/three_backbone_4pipeline_summary.csv`.
- **Four candidate pipelines** (crossing label strategy × alignment):
  1. `LR+noEA`: left/right labels, no Euclidean Alignment.
  2. `LR+EA`: left/right labels, with Euclidean Alignment.
  3. `Aff+noEA`: affected/unaffected labels + hemisphere alignment, no EA.
  4. `Aff+EA`: affected/unaffected labels + hemisphere alignment, with EA.
- **Affected/unaffected relabelling**: trials relabelled by paralysis side; right-paralysis subjects’ channels are spatially flipped so the affected hemisphere shares channel indices with left-paralysis subjects.
- **Euclidean Alignment (EA)**: whitens each subject’s data by its mean covariance matrix to align source/target covariance distributions.

## 6. Important Scripts

| Script | Purpose |
|---|---|
| `train.py` | Main entry point. Run with `--config configs/experiment/<name>.yaml`. |
| `scripts/compare_validation_strategies.py` | Produces the main comparison table (clinical-nearest / random / fixed / oracle / lesion-informed). |
| `scripts/plot_validation_test_correlation.py` | Generates Figure 5 (validation vs test accuracy scatter). |
| `scripts/validation_based_pipeline_selection.py` | Runs validation-based pipeline selection. Example: `python scripts/validation_based_pipeline_selection.py --val-strategy clinical_nearest --n-random-seeds 5`. |
| `scripts/analyze_lesion_informed_rule.py` | Tests lesion-location-based pipeline assignment rules. |
| `scripts/clinical_stratification_stats.py` | Subgroup statistics and corrected p-values for Aff vs LR. |
| `scripts/source_selection_pilot.py` | Post-hoc similarity analysis (does **not** produce the 50.66% K=10 accuracy number). |
| `scripts/generate_multik_source_selection_configs.py` | Generates the 8 EEGNet multi-K source-selection LOSO configs (K ∈ {5,10,15,20,25} × LR+noEA/LR+EA). |
| `scripts/run_multik_source_selection.sh` | Batch runner for the multi-K LOSO runs. |
| `scripts/aggregate_multik_source_selection.py` | Aggregates multi-K results (adds K=49 baselines, distance–accuracy correlation) into `results/comparisons/multik_source_selection_*.csv`. |
| `scripts/generate_altbackbone_loso_configs.py` | Generates 4-pipeline LOSO configs for an alternative backbone (`--model ShallowConvNet|ADFCNN|DeepConvNet`). |
| `scripts/run_altbackbone_loso.sh` / `scripts/run_adfcnn_noea_reruns.sh` | Batch runners for ShallowConvNet 4-pipeline LOSO and the ADFCNN noEA protocol-rerun pair. |
| `scripts/aggregate_altbackbone_loso.py` | Per-subject paired comparison of an alternative backbone vs EEGNet (`--model ...`), writes `results/comparisons/altbackbone_<model>_*`. |

## 7. Key Configs

- **Dataset configs**:
  - `configs/dataset/XWStroke_noEA_noaug.yaml`
  - `configs/dataset/XWStroke_EA.yaml`
  - `configs/dataset/XWStroke_affected_aligned_noEA_noaug.yaml`
  - `configs/dataset/XWStroke_EA_affected_aligned.yaml`
- **Base experiment config**:
  - `configs/experiment/XWStroke_EEGNet_Full50.yaml`
- **Per-pipeline LOSO configs**:
  - `configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_noaug.yaml`
  - `configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml`
  - `configs/experiment/xwstroke_eegnet_full_loso_affalign_noEA_noaug.yaml`
  - `configs/experiment/xwstroke_eegnet_full_loso_ea_affalign_no_mp.yaml`

## 8. Results Layout

- Raw experiment outputs: `experiments/<experiment_name>/results/results.json`.
- Aggregated analyses: `results/comparisons/`.
- Clinical-stratification data: `results/stratified/`.
- Generated paper figures (ignored by Git): `paper_ieee/figures/`.

## 9. Known Issues / Open Questions

1. **Source-selection model mismatch — RESOLVED (2026-08-28)**: The paper’s K=10 clinical-similarity source-selection result (50.66% ± 3.87%) came from **ADFCNN + AffAlign**, while the rest of the paper uses **EEGNet**. EEGNet + LR re-runs now exist for K ∈ {5,10,15,20,25} plus the K=49 baselines (see `results/comparisons/multik_source_selection_summary.csv`; e.g. K=10: LR+noEA 49.73%, LR+EA 50.24%). Use the EEGNet numbers in the revision.
2. **Distance–accuracy correlation**: The paper states `r = −0.026` for clinical distance vs. final accuracy, but `scripts/source_selection_pilot.py` reports `r ≈ +0.25` (on Aff benefit). Verify which metric/script produced the −0.026 value.
3. The final compiled PDF is 6 pages; no requirement to stretch to 7 pages.

## 10. Git Policy

- **Do not commit**: `paper_ieee/`, `docs/`, root-level manuscript PNGs, model checkpoints, datasets, or experiment image artifacts.
- **Do commit**: code, YAML configs, analysis scripts, and aggregated JSON/CSV results.
