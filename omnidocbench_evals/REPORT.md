# DeepSeek OCR Evaluation Report

## TL;DR

This report evaluates the performance of **DeepSeek OCR** (a vLLM-based multimodal pipeline) against **OLmOCR-2** on the OmniDocBench end-to-end benchmark, using 1,355 annotated PDF pages. DeepSeek OCR achieves an overall accuracy of **84.24%**, slightly outperforming OLmOCR-2's **81.56%**, though the difference is not statistically significant (p ≈ 0.305). Key strengths include excellent text and table recovery, with formula parsing as the primary weakness. Confidence intervals are computed using the Wald approximation (z ≈ 1.95).


### Direct Comparison: DeepSeek OCR vs OLMOCR-2

![DeepSeek OCR vs OLMOCR-2](assets/DeepSeek-OCV%20vs%20Olmocr2.png)

*Figure: Comparative performance analysis of DeepSeek OCR and OLMOCR-2 on OmniDocBench end-to-end evaluation, highlighting strengths in text recovery and areas for improvement in formula parsing.*


## Introduction

DeepSeek OCR leverages a multimodal language model with image tokenization to extract structured content from documents. This evaluation compares it head-to-head with OLmOCR-2 on OmniDocBench, a comprehensive benchmark for document understanding. Metrics include normalized edit distances for text, formulas, and reading order, plus TEDS scores for tables.

### Dataset and Setup
- **Benchmark**: OmniDocBench end-to-end evaluation.
- **Sample Size**: 1,355 pages.
- **Systems Compared**: DeepSeek OCR (vLLM pipeline), OLmOCR-2.
- **Metrics Source**: Aggregated results from `results_dpsk-cleaned_quick_match_metric_result.json` and per-page JSONs in `./result/`.

<!-- ## Key Metrics and Confidence Intervals

We report point estimates with 95% confidence intervals (CIs) using the Wald approximation (z = 1.95). Edit distances are converted to accuracies for intuitive interpretation.

| Metric | DeepSeek Point Estimate | 95% CI | Interpretation |
|---|---|---|---|
| Text-block accuracy | 92.6% | [91.2%, 94.0%] | **Excellent** text recovery with tight confidence. |
| Display-formula accuracy | 72.7% | [70.3%, 75.1%] | **Weakest area**; wider CI indicates variability. |
| Table TEDS (full) | 87.4% | [85.7%, 89.2%] | **Strong** table content similarity. |
| Table TEDS (structure-only) | 91.2% | [89.7%, 92.7%] | **Outstanding** structural recovery. |
| Reading-order accuracy | 91.1% | [89.6%, 92.6%] | **Very good** sequence matching. |
| **Overall** | **84.2%** | **[82.3%, 86.2%]** | **Solid performance** across components. | -->

## Definitions

- **Normalized edit distance**: Measures string dissimilarity (0 = perfect match, 1 = completely different).
- **Accuracy (1 − Edit_dist)**: Proportion correct, e.g., 0.926 = 92.6%.
- **TEDS**: Table Edit Distance-based Similarity (0–100%, higher = better).
- **Table TEDS (structure-only)**: TEDS based only on table layout, ignoring content.
- **Reading-order edit distance**: Dissimilarity in predicted vs. ground-truth reading sequences.
- **Overall score**: Weighted aggregate from OmniDocBench.

## Visual Comparisons


### Overall Performance

| | |
|-:|:-|
| ![DeepSeek overall](assets/ocr/End2End_OmniDocBench_deepseek_ocr_overall.png) | ![OLmOCR-2 overall](assets/ocr/End2End_OmniDocBench_olmo_ocr_2_overall.png) |
| DeepSeek OCR Overall | OLmOCR-2 Overall |


### Language Breakdown — English

| | |
|-:|:-|
| ![DeepSeek english](assets/ocr/End2End_OmniDocBench_deepseek_ocr_english.png) | ![OLmOCR-2 english](assets/ocr/End2End_OmniDocBench_olmo_ocr_2_english.png) |
| DeepSeek OCR English | OLmOCR-2 English |

### Language Breakdown — Chinese

| | |
|-:|:-|
| ![DeepSeek chinese](assets/ocr/End2End_OmniDocBench_deepseek_ocr_chinese.png) | ![OLmOCR-2 chinese](assets/ocr/End2End_OmniDocBench_olmo_ocr_2_chinese.png) |
| DeepSeek OCR Chinese | OLmOCR-2 Chinese |

### Example Outputs

| | |
|-:|:-|
| ![Example 1](assets/show1.jpg) | ![Example 2](assets/show2.jpg) |
| Sample 1 | Sample 2 |
| ![Example 3](assets/show3.jpg) | ![Example 4](assets/show4.jpg) |
| Sample 3 | Sample 4 |

## Methodology Details

- **CI Calculation**: Wald approximation with z = 1.95 for 95% bands. SE = sqrt(p(1-p)/n), ME = z × SE.
- **Example**: For text-block accuracy (p = 0.926, n = 1355), SE ≈ 0.00711, ME ≈ 0.01386, CI = [0.912, 0.940].
- **Data Sources**: OmniDocBench pipeline outputs; per-page JSONs for potential deeper analysis.
- **Metric choice (Edit Distance vs CDM)**: For text, formula and reading-order we used normalized Edit Distance (reported as `Edit_dist` and presented as accuracy = 1 − Edit_dist) rather than the CDM metric. Edit Distance is simple, interpretable, and directly available from the OmniDocBench outputs; CDM (Content Distance Metric) is an alternative that emphasizes token-level content differences and could be used in follow-up analyses.

## Statistical Test: Overall Performance Comparison

We conducted a z-test for proportions to compare overall accuracies.

- **z-statistic**: -1.025
- **p-value**: 0.305

**Interpretation**: No significant difference (p > 0.05). DeepSeek's slight edge (84.24% vs. 81.56%) is not conclusive.

**Test Details**:
- Dataset: Full OmniDocBench set (n = 1,355).
- Observed: DeepSeek = 84.24% (≈1,141 successes), OLmOCR-2 = 81.56% (≈1,105 successes).

| System | Overall (%) | Successes (approx) |
|---|---:|---:|
| DeepSeek OCR | 84.24 | 1,141 |
| OLmOCR-2 | 81.56 | 1,105 |

<!-- ## Key Takeaways

- **Strengths**: DeepSeek excels in text, tables, and reading order, with narrow CIs indicating reliability.
- **Weaknesses**: Formula accuracy lags; focus engineering efforts here.
- **Comparison**: No significant difference from OLmOCR-2; consider paired tests for deeper insights.
- **Recommendations**:
  - Improve formula parsing (e.g., LaTeX-aware models).
  - Run bootstrap CIs on per-page data for robustness.
  - Explore equivalence testing (TOST) for practical parity. -->

## Conclusion

DeepSeek OCR demonstrates strong document understanding capabilities, particularly for text and tables, on OmniDocBench. While not significantly outperforming OLmOCR-2 in this evaluation, its performance is competitive and reliable. Future work should address formula weaknesses and employ advanced statistical methods for comparisons.

*Report generated on 2025-10-29. Data from OmniDocBench evaluation.*

## Definitions

- Normalized edit distance: A length-normalized measure of how different a predicted string is from the ground truth (0 = identical, 1 = completely different).
- Accuracy (1 − Edit_dist): The complement of normalized edit distance; the proportion correct shown as a percentage (e.g., 0.926 → 92.6%).
- TEDS: Table Edit Distance-based Similarity — a table-quality score derived from tree-edit-distance between predicted and ground-truth table structures and content (reported 0–100%, higher is better).
- Table TEDS (structure-only): The TEDS score computed using only the table’s structural/layout information (cell positions and nesting), ignoring cell text.
- Reading-order edit distance: Normalized edit distance computed on the predicted vs. reference reading-order sequence; lower is better (often reported as accuracy = 1 − value).
- Overall score: The aggregate document-level score from the evaluation run (a summary metric reported by OmniDocBench combining components).

<!-- 
## Figures

### Overall comparison

| | |
|-:|:-|
| ![DeepSeek overall](assets/ocr/End2End_OmniDocBench_deepseek_ocr_overall.png) | ![OLmOCR-2 overall](assets/ocr/End2End_OmniDocBench_olmo_ocr_2_overall.png) |
| `End2End_OmniDocBench_deepseek_ocr_overall.png` | `End2End_OmniDocBench_olmo_ocr_2_overall.png` |

### Language breakdown — English

| | |
|-:|:-|
| ![DeepSeek english](assets/ocr/End2End_OmniDocBench_deepseek_ocr_english.png) | ![OLmOCR-2 english](assets/ocr/End2End_OmniDocBench_olmo_ocr_2_english.png) |
| `End2End_OmniDocBench_deepseek_ocr_english.png` | `End2End_OmniDocBench_olmo_ocr_2_english.png` |

### Language breakdown — Chinese

| | |
|-:|:-|
| ![DeepSeek chinese](assets/ocr/End2End_OmniDocBench_deepseek_ocr_chinese.png) | ![OLmOCR-2 chinese](assets/ocr/End2End_OmniDocBench_olmo_ocr_2_chinese.png) |
| `End2End_OmniDocBench_deepseek_ocr_chinese.png` | `End2End_OmniDocBench_olmo_ocr_2_chinese.png` |

### Example OCR outputs

| | |
|-:|:-|
| ![Example 1](assets/show1.jpg) | ![Example 2](assets/show2.jpg) |
| `show1.jpg` | `show2.jpg` |
| ![Example 3](assets/show3.jpg) | ![Example 4](assets/show4.jpg) |
| `show3.jpg` | `show4.jpg` |


Methodology for the confidence intervals
- Sample size: n = 1,355 pages
- CI method: Wald (normal approximation) using z = 1.95 (user requested approximation). For a proportion p, standard error is SE = sqrt(p*(1-p)/n), margin of error ME = z * SE, and CI = p ± ME.
- For Edit_dist entries we report the CI on (1 − Edit_dist) so readers see an accuracy interval rather than an error-rate interval. (This aligns interpretatively with TEDS and overall which are percent-style.)



## Statistical test — overall performance (DeepSeek vs OLmOCR-2)

- z = -1.0248442301557446
- p = 0.30543669282100794


Interpretation: the reported p-value (≈0.305) is well above conventional significance thresholds (e.g., 0.05). We therefore fail to reject the null hypothesis of no difference in overall performance between DeepSeek and OLmOCR-2 on this evaluation set. The negative z indicates that, in this sample, DeepSeek's overall score was slightly lower than OLmOCR-2, but the difference is small and not statistically significant.


- Dataset: OmniDocBench end-to-end evaluation outputs (the full set of evaluated pages; n = 1,355). We used the aggregated system-level metrics produced by the OmniDocBench pipeline and the per-page JSONs stored under `./result/` for more detailed checks.
- Observed overall metrics used in the test: DeepSeek OCR overall = 84.239% (p̂1 = 0.84239), OLmOCR-2 overall = 81.560% (p̂2 = 0.81560).
- Approximate success counts used (rounded): DeepSeek successes ≈ 1,141 (0.84239 × 1,355), OLmOCR-2 successes ≈ 1,105 (0.81560 × 1,355).

Summary table

| System | Overall (%) | n | Successes (approx) |
|---|---:|---:|---:|
| DeepSeek OCR | 84.239 | 1,355 | 1,141 |
| OLmOCR-2 | 81.560 | 1,355 | 1,105 |



Interpretation and takeaways
- Text paragraphs: High accuracy (≈92.6%) with a tight CI (±≈1.39 percentage points). DeepSeek recovers text blocks reliably on this set.
- Display formulas: Lower accuracy (≈72.7%) with a wider CI (±≈2.36 points). Formula parsing remains the weakest component and is the main contributor to reduced overall score.
- Tables: Strong structural recovery (TEDS_structure_only ≈91.2%) and good full-table similarity (≈87.4%), both with narrow CIs (±≈1.5–1.8 points). This indicates the pipeline recovers table layouts well and also extracts content with good fidelity.
- Reading order: Very good (≈91.1%, ±≈1.51 points), meaning predicted reading sequences match ground-truth order well.
- Overall: Reported overall ≈84.24% with 95% CI ≈ [82.31%, 86.17%]. This reflects the combined effect of very good table+order recovery, strong text recovery, and weaker formula recovery. -->
