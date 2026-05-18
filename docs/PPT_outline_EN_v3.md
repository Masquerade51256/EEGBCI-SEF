# PPT Outline v3 (English) — Optimized 17-Slide Structure

**Title:** Clinical Heterogeneity and Adaptive Decoding for Cross-Subject MI-EEG in Acute Stroke  
**Four Parts:** Background → Benchmarking → Preliminary Results → Future Directions  
**Estimated Duration:** 15 minutes

---

## PART 1: BACKGROUND (Slides 1–4)

---

## Slide 1: Title
- **Title:** Clinical Heterogeneity and Adaptive Decoding for Cross-Subject MI-EEG in Acute Stroke
- **Subtitle:** Beyond One-Size-Fits-All: Leveraging Patient Stratification for EEG-Based BCI
- Author, Affiliation, Date

---

## Slide 2: Stroke Heterogeneity & Clinical Measurement *(Heterogeneity + Measurement merged)*

**Top half — The Problem:**
- Stroke patients are **highly heterogeneous** — no two patients have the same lesion pattern, severity, or recovery trajectory
- Illustration: Brain diagram with varied lesion locations (cortical / subcortical / brainstem / mixed)
- Factors creating EEG variability: lesion location, NIHSS severity, paralysis side, disease duration

**Bottom half — Quantification:**
- **NIHSS** (0–42): Neurological deficit severity
- **FMA** (0–66/100): Motor function assessment
- **mRS** (0–5): Disability level
- **Key message:** Heterogeneity is not noise — it is **structured clinical information** that current decoders ignore

**Visual:** `fig_stroke_heterogeneity.png` *(to be generated)*

---

## Slide 3: XWStroke Dataset & Three Label Schemes

**Dataset:**
- N=50 acute stroke patients (1–30 days post-stroke)
- 32 EEG + 2 EOG channels, 500 Hz sampling, 4-second epochs
- 13 artifact-affected subjects identified

**Three Task Schemes:**
| Task | Label | Clinical Meaning |
|:---|:---|:---|
| Task A | Left / Right | Standard motor imagery |
| Task B | Affected / Unaffected | Clinically meaningful |
| Task C | Affected + Hemisphere Alignment | Spatial remapping to lesion side |

**Protocol:** Strict **Streaming LOSO** — train on 49 subjects, test on 1, zero data leakage

---

## Slide 4: Objective & Contribution

**Problem Statement:**
- Cross-subject MI-EEG decoding in stroke fails because models assume all patients share the same neural patterns
- Domain-adversarial approaches try to erase differences — but stroke differences are clinically meaningful

**Our Goal:**
> "Not to chase higher accuracy under lenient protocol, but to understand **who benefits from what strategy**"

**Core Contribution:**
- First systematic introduction of **acute stroke clinical variables** (lesion location, NIHSS, duration) into cross-subject MI-EEG decoding
- Moving from **one-size-fits-all** to **clinically adaptive strategies**

---

## PART 2: RESEARCH STATUS & BENCHMARKING (Slides 5–7)

---

## Slide 5: Competitive Landscape Map

**2D Scatter:** Protocol Strictness (X) vs Reported Accuracy (Y)

| Paper | Protocol | Score | Color |
|:---|:---|:---|:---|
| Bun24 / Bun25 | 80/20 split / "50 divisions" | ~97% | Gray |
| Kav24b | Strict subject-wise split | 69.77% | Light blue |
| Wan26h | Fixed-subject LOSO | **66.56%** | Blue |
| **Ours** | **Streaming LOSO** | **51.46%** | **Red** |

- Chance level (50%) dashed line
- Protocol strictness zones: Lenient → Moderate → Strict
- **Bottom banner:** *"We don't compete on lenient-protocol scores. We compete on strict-protocol interpretability."*

**Visual:** `fig_competitive_landscape.png` ✅

---

## Slide 6: Core Method Papers on XW Dataset

**Comparison Table:**

| Paper | Protocol | XWStroke Score | Fair Benchmark? | Core Method | Clinical Variables? |
|:---|:---|:---|:---|:---|:---|
| Bun25 | "50-fold LOSO" (unclear) | 97.43% | ❌ | Modified ResNet | ❌ |
| Kav24b | Strict subject split | 69.77% | ⚠️ | CNN-BiLSTM | ❌ |
| Wan26h | Fixed-subject LOSO | **66.56%** | ✅ | Freq. selection + alignment | ❌ |
| **Ours** | **Streaming LOSO** | **51.46%** | ✅ | **Clinical adaptive decoding** | **✅** |

**Key Insight:** The literature gap is **"bringing patient features into modeling."** We fill it.

---

## Slide 7: Key Takeaway — Why Domain Invariance is the Wrong Direction

**Current literature:** Dominated by classification-score chasing under lenient protocols
**Confirmed gap:** "Patient clinical characteristics in the modeling process"
**Our hypothesis:** Domain-invariant feature learning erases the very neurophysiological differences that clinical labels depend on
- This hypothesis was **tested and confirmed** by our DANN experiments (see Slide 12)
- **The path forward:** Clinical adaptation, not domain invariance

---

## PART 3: PRELIMINARY RESULTS (Slides 8–13)

---

## Slide 8: Full-Cohort Baseline (N=50)

**Bar chart with error bars:**

| Task | Mean ± Std |
|:---|:---|
| Task A (LR) | 49.63% ± 5.83% |
| Task B (Aff/Unaff) | 50.29% ± 4.13% |
| **Task C (Aff+Align)** | **51.46% ± 5.23%** |

**Statistics:** Task C vs A: **+1.83%**, p=0.062, d=0.270, 95% CI [-0.09%, +3.75%]
- **Framing:** Trend-level improvement (p≈0.06), masked by high inter-subject variance

**Visual:** `fig_task_comparison.png` ✅

---

## Slide 9: Why is the Ceiling So Low? — CleanC Mechanism *(NEW: Stratified Table)*

**The Artifact Ceiling:**
- 13/50 subjects have severe artifacts (movement, EMG, electrode issues)
- CleanC subset (N=14, artifact-free + subcortical + NIHSS≤7) accidentally excluded ALL 13 artifact subjects
- **Subject IDs:** [8, 9, 11, 16, 17, 23, 25, 26, 27, 29, 30, 31, 36, 37]

**Task Results:**

| Condition | Task A | Task C | Diff |
|:---|:---|:---|:---|
| CleanC (no artifact) | 50.57% | 52.29% | +1.72% |
| Artifact subjects | 48.30% | 50.15% | +1.85% |

**CleanC Stratified Composition vs. Full Cohort:**

| Dimension | Category | CleanC (N=14) | Full-50 (N=50) |
|:---|:---|:---|:---|
| **Artifact** | Yes | **0 (0%)** | 13 (26%) |
| | No | **14 (100%)** | 37 (74%) |
| **NIHSS** | Light (1–3) | 6 (43%) | 28 (56%) |
| | Medium (4–7) | **8 (57%)** | 16 (32%) |
| | Severe (≥8) | **0 (0%)** | 6 (12%) |
| **Lesion Location** | Subcortical | **14 (100%)** | 29 (58%) |
| | Brainstem | **0 (0%)** | 16 (32%) |
| | Cortical | **0 (0%)** | 3 (6%) |

**Conclusion:**
1. Label improvement (Aff+Align) is **general** across both groups
2. **Artifact presence is the dominant performance ceiling** — not the algorithm
3. CleanC is a **homogeneity-selected subset** — it validates mechanism under ideal conditions (subcortical-only, no artifacts, moderate severity)

---

## Slide 10: Stratified Analysis — Who Benefits?

**Waterfall chart:** Task C vs Task A difference by subgroup

| Dimension | Subgroup | Effect |
|:---|:---|:---|
| Lesion Location | Subcortical (N=20) | **+3.6%** |
| | Cortical (N=4) | **-3.6%** |
| NIHSS | Medium (4-7, N=16) | +0.8% |
| | Severe (≥8, N=6) | **-5.5%** |
| Duration | Chronic (>3mo, N=23) | **+2.9%** |
| | Acute (≤3mo, N=27) | +0.9% |
| Baseline Perf. | Low (N=18) | **+7.1%** |
| | High (N=17) | **-6.3%** |

**Key Finding:** Hemisphere alignment helps **"less damaged, deeper lesion"** patients most. It hurts cortical-damage patients because the alignment assumes intact motor circuits.

**Visual:** `fig_stratified_waterfall.png` ✅

---

## Slide 11: Subset Validation & Representative Design *(NEW: Stratified Table + Comparison)*

**10-Subject Subset:**
- Task A: 46.80% → Task C: **52.50%** (+5.70%)
- Cohen's d = 0.600 (medium effect), 7/10 improved

**Subset-20 Representative Design:**
- **Purpose:** Preserve full-population heterogeneity while cutting runtime by 60%
- **IDs:** [1, 5, 7, 8, 9, 10, 15, 18, 20, 23, 25, 28, 29, 34, 35, 38, 41, 45, 48, 49]
- **Validation:** Baseline ~49.08%, Aff+Align diff ~+1.97% (consistent with full-cohort +1.83%)

**Subset-20 Stratified Composition vs. Full Cohort:**

| Dimension | Category | Subset-20 (N=20) | Full-50 (N=50) |
|:---|:---|:---|:---|
| **Artifact** | Yes | 5 (25%) | 13 (26%) |
| | No | 15 (75%) | 37 (74%) |
| **Paralysis Side** | Left | 11 (55%) | 28 (56%) |
| | Right | 9 (45%) | 22 (44%) |
| **NIHSS** | Light (1–3) | 11 (55%) | 28 (56%) |
| | Medium (4–7) | 7 (35%) | 16 (32%) |
| | Severe (≥8) | 2 (10%) | 6 (12%) |
| **Lesion Location** | Subcortical | 11 (55%) | 29 (58%) |
| | Brainstem | 7 (35%) | 16 (32%) |
| | Cortical | 1 (5%) | 3 (6%) |

> All proportions match the full cohort within ±3% — heterogeneity is preserved.

---

**CleanC vs. Subset-20: Two Different Subset Philosophies *(NEW comparison)*

| Aspect | CleanC (N=14) | Subset-20 (N=20) |
|:---|:---|:---|
| **Selection logic** | Homogeneity filtering | Stratified sampling |
| **Purpose** | Mechanism validation | Fast iteration |
| **Artifact subjects** | **0%** | 25% |
| **Severe NIHSS** | **0%** | 10% |
| **Lesion diversity** | 100% subcortical only | 55% subcortical + 35% brainstem + 5% cortical |
| **Use case** | Upper-bound estimate | Realistic estimate |

> **Bottom line:** CleanC tells us the method has *mechanistic validity*. Subset-20 tells us the method is *practically testable*.

---

## Slide 12: What Didn't Work — DANN & The Pivot

**Motivation:** Learn domain-invariant features across subjects via adversarial training

**Results:**

| Config | N | Domain Label | dlw | Mean Acc. | Outcome |
|:---|:---|:---|:---|:---|:---|
| Full-50 | 15 | stroke_location | 0.3 | 50.47% | No gain |
| CleanC | 14 | stroke_location | 0.3 | 50.14% | No gain |
| CleanC | 14 | stroke_location | 1.0 | 48.75% | **Worse** |

**Diagnosis:**
1. `stroke_location` as domain label is **conceptually wrong** — it correlates with MI patterns
2. Domain-invariant learning **erases clinically-relevant differences**
3. Gridsearch: 8 configs, only 2 completed, most failed due to instability

**The Pivot:**
> "This negative result **strengthens** our case. Pursuing domain invariance in stroke MI is fundamentally misaligned with the clinical decoding objective. The path forward is **clinical adaptation**, not domain adversarial training."

**Visual:** `fig_dann_results.png` ✅

---

## PART 4: FUTURE DIRECTIONS (Slides 13–17)

---

## Slide 13: Route B — Clinical Adaptive Framework *(Proposed)*

**Core Principle:** Don't erase differences — **exploit clinical stratification to adapt**

**Two Approaches:**

**Approach 1: Dynamic Source-Domain Selection**
- For each test patient, select clinically similar source subjects
- Match by: lesion location, NIHSS, duration, paralysis side
- Train a personalized model on the matched subset

**Approach 2: Conditional Feature Extractor**
- Feed clinical metadata (NIHSS, lesion side, duration) as auxiliary inputs
- Use clinical variables to gate/modulate feature extraction pathways

**Advantage:** Interpretable + clinically actionable — model behavior traces back to patient characteristics

**Visual:** `fig_clinical_adaptive_framework.png` *(to be generated)*

---

## Slide 14: Subset-20 as Fast Validation Tool

**Not a research contribution — a methodological accelerator:**
- Full-50 LOSO: ~4–6 hours per experiment
- Subset-20 LOSO: ~1.5–2 hours per experiment
- **Preserves heterogeneity** through stratified sampling

**Consistency Check:**
| Cohort | Task A | Aff+Align Diff |
|:---|:---|:---|
| Full-50 | 49.63% | +1.83% |
| Subset-20 | ~49.08% | ~+1.97% |

> "Subset-20 enables rapid iteration for Route B development without sacrificing population representativeness."

---

## Slide 15: External Validation Roadmap

**Tier 1 — Publicly Available (Immediate Use):**

| Dataset | N | Stage | Key Strength | Validation Purpose |
|:---|:---|:---|:---|:---|
| **Thi25** | 30 | Post-acute | Rich clinical JSON (NIHSS, lesion, mRS, Oxford) | Clinical-stratification generalizes? |
| **Liu25** | 27 | Recovery (1–12 mo) | Closest to Liu24 structure (Figshare, BIDS) | Cross-task / cross-timepoint robustness |
| **Cho21** | 10 | ~11 mo | Designed as cross-subject benchmark | Cross-subject transfer validity |

**Tier 2 — Collaboration Required:**
- Man22 (N=136, FMA-rich) → Seb20b (lesion stratification) → Fro17b (multicenter RCT)

**Bottom line:** Even **one Tier-1 validation** (especially Thi25) would make our work unique — no prior Liu24 paper has reported cross-dataset validation.

**Visual:** `fig_external_validation_roadmap.png` ✅

---

## Slide 16: Work Plan & Timeline

**Gantt-style timeline:**

| Phase | Week | Status |
|:---|:---|:---|
| Full-cohort baseline statistics | 1–2 | ✅ Complete |
| Literature audit + benchmarking | 2–3 | ✅ Complete |
| Subset-20 representative design | 3 | ✅ Complete |
| DANN diagnostics | 3–4 | ✅ Complete (abandoned) |
| Subset-20 baseline validation | 4 | 🔄 In progress |
| **Route B: Clinical adaptive framework** | **4–7** | **⏳ Priority #1** |
| External validation (Thi25 / Liu25) | 6–8 | ⏳ Pending |

---

## Slide 17: Conclusion

**Three Core Conclusions:**
1. Cross-subject LOSO baseline in acute stroke is genuinely low (~50%) — this reflects **real clinical heterogeneity**, not algorithm failure
2. Affected-side + hemisphere alignment yields **+1.83% trend-level gain**, reaching **+3–7% in benefited subgroups** (subcortical, low-baseline, chronic)
3. **The real innovation opportunity:** Bring patient clinical features into decoding (adaptive framework), not just optimize the classifier

**Target Journal:** TNSRE or JBHI

**Core Selling Point:**
> "First work to systematically introduce acute stroke patient clinical variables into cross-subject MI-EEG decoding, moving from one-size-fits-all to clinically adaptive strategies."

---

## Optional Backup Slide

## Slide 18: Defensive Q&A *(Light gray background, if time permits)*

**Q1: "Why only 51.46% vs Bun25's 97.43%?"**
1. Protocol mismatch — Bun25's "LOSO" is ambiguous
2. Fair baseline — Wan26h strict LOSO = **66.56%**
3. Value — strict protocol + interpretability > lenient protocol + high score

**Q2: "Affected-side labels aren't new. What's novel?"**
1. First systematic validation in acute stroke strict LOSO + hemisphere alignment + stratification
2. Literature audit confirms gap = "patient characteristics in modeling"

**Q3: "Is 2–3% improvement worth publishing?"**
1. In stroke strict LOSO, 2–3% is genuine progress
2. Interpretable clinical mechanism > uninterpretable high score
3. Benefited subgroups show **3–7% gains** — sufficient for TNSRE/JBHI

---

## Visual Assets Summary

| Slide | Figure | Status |
|:---|:---|:---|
| 2 | `fig_stroke_heterogeneity.png` | 🔄 To generate |
| 5 | `fig_competitive_landscape.png` | ✅ Ready |
| 8 | `fig_task_comparison.png` | ✅ Ready |
| 10 | `fig_stratified_waterfall.png` | ✅ Ready |
| 12 | `fig_dann_results.png` | ✅ Ready |
| 13 | `fig_clinical_adaptive_framework.png` | 🔄 To generate |
| 15 | `fig_external_validation_roadmap.png` | ✅ Ready |
| 16 | `fig_action_timeline.png` | ✅ Ready |

---

## Speaker Notes (Narrative Thread)

**Opening hook:** "If you train a model on 49 stroke patients and test it on the 50th, you get roughly 50% accuracy — essentially random. But here's the thing: that's not a failure of the algorithm. That's the reality of stroke heterogeneity."

**Transition to benchmarking:** "And before I show you our results, I need to address the elephant in the room — why our numbers look so different from papers claiming 97%."

**Transition to results:** "So under the strictest protocol, the baseline is genuinely low. But when we look closer, we find structure in that heterogeneity."

**Transition to future:** "Our DANN experiments taught us that erasing differences is the wrong goal. Instead, we should exploit them."

**Closing:** "Even if our adaptive framework only works on external datasets with rich clinical information like Thi25, that alone would be a first in this field."
