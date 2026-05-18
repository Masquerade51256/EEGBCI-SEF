# PPT Outline v2.1 (English) — Research Progress Report

**Title:** Clinical Heterogeneity and Adaptive Decoding for Cross-Subject MI-EEG in Acute Stroke  
**Subtitle:** Systematic Experimental Validation on XWStroke Dataset (N=50)  
**Estimated Duration:** 15 minutes | **Slides:** 15–18

---

## Slide 1: Title
- **Title:** Clinical Heterogeneity and Adaptive Decoding for Cross-Subject MI-EEG in Acute Stroke
- **Subtitle:** Systematic Validation on XWStroke (N=50) under Strict Streaming LOSO
- Author, Affiliation, Date

---

## Slide 2: Clinical Motivation
- **Problem:** Stroke rehabilitation needs MI-BCI, but cross-subject generalization fails
- **Why:** Acute stroke patients are highly heterogeneous (lesion location, NIHSS, paralysis side, duration)
- **Core Question:** Can we leverage clinical stratification for adaptive decoding?

---

## Slide 3: Dataset & Experimental Design
- **XWStroke:** N=50 acute stroke, 32 EEG + 2 EOG, 500 Hz, 4 s epochs
- **Three Label Schemes:**
  - Task A: Left/Right (LR) — standard MI labels
  - Task B: Affected/Unaffected — clinically meaningful
  - Task C: Affected + Hemisphere Alignment — spatial remapping
- **Protocol:** Strict **Streaming LOSO** — each fold trains on 49 subjects, tests on 1, zero data leakage

---

## Slide 4: Competitive Landscape Map *(NEW — Key Visual)*
- **Title:** Literature Benchmarking on Liu24 (XWStroke) — Only 7 Confirmed Papers
- **Left:** Protocol Stratification Pyramid
  - Bottom (Lenient): Bun24 (~97%, 80/20 split) — "Looks good, no generalization"
  - Middle: Kav24b (69.77%, strict subject-wise split) — "Better, but not LOSO"
  - Top (Strict): Wan26h (66.56%, fixed LOSO) — "Closest to real-world application"
  - **Our Position:** Streaming LOSO — "Strictest protocol, naturally lowest baseline"
- **Right:** 2D Scatter Plot (X = Protocol Strictness, Y = Reported Accuracy)
  - Bun25 (97.43%, gray, "50 divisions, LOSO definition unclear")
  - Wan26h (66.56%, blue, "Strict LOSO, current fair baseline")
  - **Ours (51.46%, red, "Streaming LOSO, full heterogeneous cohort")**
- **Bottom Banner:** *"We don't compete on lenient-protocol scores. We compete on strict-protocol interpretability."*

---

## Slide 5: Benchmarking Table
- Clean comparison table:

| Paper | Protocol | XWStroke Score | Fair Benchmark? | Core Method | Clinical Variables? |
|:---|:---|:---|:---|:---|:---|
| Bun25 | "50-fold LOSO" (unclear) | 97.43% | ❌ | Modified ResNet | ❌ |
| Kav24b | Strict subject split | 69.77% | ⚠️ | CNN-BiLSTM | ❌ |
| Wan26h | Fixed-subject LOSO | **66.56%** | ✅ | Freq. selection + alignment | ❌ |
| **Ours** | **Streaming LOSO** | **51.46%** | ✅ | **Clinical adaptive decoding** | **✅** |

- **Key Insight:** The literature gap is "bringing patient features into modeling." We fill it.

---

## Slide 6: Full-Cohort Baseline Results
- Table:

| Task | Label Scheme | Mean ± Std |
|:---|:---|:---|
| Task A | Left/Right | 49.63% ± 5.83% |
| Task B | Affected/Unaffected | 50.29% ± 4.13% |
| **Task C** | **Affected + Aligned** | **51.46% ± 5.23%** |

- Statistics: Task C vs A: **+1.83%**, p = 0.062, d = 0.270, 95% CI [-0.09%, +3.75%]
- **Visual:** Highlight Task C in red, note "trend-level improvement" (not "significant")

---

## Slide 7: Stratified Analysis — Who Benefits?
- **Waterfall / Grouped Bar Chart** (suggested visual)
- Dimensions:
  - **Lesion Location:** Subcortical +3.6% vs Cortical -3.6%
  - **NIHSS:** Medium +0.8% vs Severe -5.5%
  - **Duration:** Chronic +2.9% vs Acute +0.9%
  - **Baseline Perf.:** Low +7.1% vs High -6.3%
- **Key Finding:** Hemisphere alignment helps "less damaged, deeper lesion" patients most.

---

## Slide 8: Mechanism Validation — CleanC Subset (N=14)
- Table:

| Condition | Task A | Task C | Diff |
|:---|:---|:---|:---|
| CleanC (no artifact) | 50.57% | 52.29% | +1.72% |
| Artifact subjects | 48.30% | 50.15% | +1.85% |

- **Critical Realization:** CleanC accidentally excluded ALL 13 artifact subjects.
- Label improvement is general; **artifact presence is the dominant performance ceiling.**

---

## Slide 9: Subset-20 Representative Design *(NEW)*
- **Purpose:** Preserve full-population heterogeneity while cutting runtime by 60%
- **Stratification:** Lesion location, NIHSS, paralysis side, duration, artifact presence
- **IDs:** [1, 5, 7, 8, 9, 10, 15, 18, 20, 23, 25, 28, 29, 34, 35, 38, 41, 45, 48, 49]
- **Validation:** Baseline ~49.08%, Aff+Align diff ~+1.97% (consistent with full-cohort +1.83%)
- **Visual:** Stratification matrix showing representation across all dimensions

---

## Slide 10: Route A — DANN (Domain-Adversarial) *(UPDATED with Negative Results)*
- **Motivation:** Learn domain-invariant features across subjects
- **Implementation:** ADFCNN + GRL, domain labels = stroke_location → paralysis_side
- **Results:**

| Config | N | dlw | Mean Acc. | Outcome |
|:---|:---|:---|:---|:---|
| Full-50 | 15 | 0.3 | 50.47% ± 3.68% | No gain |
| CleanC | 14 | 0.3 | 50.14% ± 2.26% | No gain |
| CleanC | 14 | 1.0 | 48.75% ± 2.69% | **Worse** |

- **Diagnosis:**
  1. `stroke_location` as domain label is conceptually wrong — it correlates with MI patterns
  2. Domain-invariant learning **erases clinically-relevant differences**
  3. **Conclusion: Route A is abandoned.** Pursuing domain invariance is the wrong direction.

---

## Slide 11: Route B — Clinical Adaptive Framework *(Proposed)*
- **Core Principle:** Don't erase differences — **exploit clinical stratification to adapt**
- **Approach 1: Dynamic Source-Domain Selection**
  - For each test patient, select clinically similar source subjects
  - Train personalized model on matched subset
- **Approach 2: Conditional Feature Extractor**
  - Feed NIHSS, lesion side, duration as auxiliary inputs
  - Clinical variables gate feature extraction pathways
- **Advantage:** Interpretable + clinically actionable

---

## Slide 12: Defensive Q&A *(For Supervisor / Reviewer)*

**Q1: "Why only 51.46% vs Bun25's 97.43%?"**
1. Protocol mismatch — Bun25's "LOSO" is ambiguous (likely random splits)
2. Fair baseline — Wan26h strict LOSO = **66.56%**
3. Value — strict protocol + interpretability > lenient protocol + high score

**Q2: "Affected-side labels aren't new. What's novel?"**
1. First systematic validation in acute stroke strict LOSO + hemisphere alignment + stratification
2. Literature audit confirms gap = "patient characteristics in modeling"

**Q3: "Is 2–3% improvement worth publishing?"**
1. In stroke strict LOSO, 2–3% is genuine progress in a hard scenario
2. Interpretable clinical mechanism > uninterpretable high score
3. Benefited subgroups show **3–7% gains** — sufficient for TNSRE/JBHI

---

## Slide 13: Action Plan (Next 2–4 Weeks)
- ✅ Full-cohort baseline statistics
- ✅ Literature audit
- ✅ Subset-20 design
- ✅ DANN diagnostics (**abandoned**)
- 🔄 Subset-20 baseline validation
- ⏳ **Route B: Clinical adaptive framework (Priority #1)**
- ⏳ Cross-dataset validation (LowerStroke)

---

## Slide 14: Risk Assessment

| Risk | Prob. | Impact | Mitigation |
|:---|:---|:---|:---|
| DANN fails (confirmed) | High | Route A abandoned | **Already pivoted to Route B** |
| Route B marginal gains | Medium | Contribution weak | Emphasize clinical insight as core contribution |
| Time shortage | Low | Incomplete comparisons | Subset-20 for fast validation |
| Reviewer miscompares with Bun25 | Medium | Misunderstanding | Defense slide (Slide 12) prepared |

---

## Slide 15: Conclusion & Paper Positioning

**Core Conclusions:**
1. Cross-subject LOSO baseline in acute stroke is genuinely low (~50%) — this reflects **real clinical heterogeneity**
2. Affected-side + hemisphere alignment yields **+1.83% trend-level gain**, reaching **+3–7% in benefited subgroups**
3. **The real innovation opportunity:** Bring patient clinical features into decoding (adaptive framework), not just optimize the classifier

**Target Journal:** TNSRE or JBHI

**Core Selling Point:**
> "First work to systematically introduce acute stroke patient clinical variables into cross-subject MI-EEG decoding, moving from one-size-fits-all to clinically adaptive strategies."

---

## Appendix Slides (Optional / Backup)

- **Appendix A:** Detailed statistical tables (Full-50, CleanC, Subset-10, Subset-20)
- **Appendix B:** DANN training loss curves (diagnostic)
- **Appendix C:** Literature audit raw records (7 papers with protocol analysis)
- **Appendix D:** Lateralization Index distribution and correlation plots

---

## Visual Design Guidelines

### Slide 4: Competitive Landscape Map
- **2D scatter plot:** X = Protocol Strictness, Y = Accuracy
- Color coding: Gray = lenient protocol, Blue = strict LOSO, Red = our method
- Annotations next to each point with paper name + protocol note
- Bottom banner with value proposition in bold

### Slide 7: Stratified Waterfall Chart
- X-axis: Subgroup categories (Lesion / NIHSS / Duration / Baseline)
- Y-axis: Task C vs Task A difference (%)
- Green bars for positive, red bars for negative
- Immediately shows "who wins, who loses"

### Slide 10: DANN Results
- Simple bar chart: Baseline vs DANN variants
- Gray = baseline, Light red = no gain, Dark red = worse
- Annotation: "Domain invariance ≠ Clinical decoding"

### Slide 12: Defensive Q&A
- Three columns, each: Large question text → 3 bullet-point answers
- Light gray background to distinguish from main content
- Label: "Prepared Responses"
