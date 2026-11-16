# Preliminary Findings

## Status and provenance

The numeric results in this report are preliminary summary values supplied by
the project owner. They have not been independently reproduced. The source data
is preserved in [`preliminary_results.csv`](preliminary_results.csv). Missing
measurements are shown as **N/A**; no per-image observations or unreported
metrics have been inferred from the summaries.

## Methodology and evaluation conditions

The project evaluates an AdvMark-inspired defended Stable Signature VAE
decoder against WAVES-style image attacks and decoder fine-tuning. The four
conditions are:

- **baseline**: original watermark decoder.
- **defended**: decoder trained to retain watermark detection under blur while
  constraining reconstruction quality.
- **baseline_ft**: baseline decoder after the decoder fine-tuning attack.
- **defended_ft**: defended decoder after the same attack.

The shared evaluation schema defines per-image `image_id`, `condition`,
`attack`, and `strength`, with bit accuracy / BER, detection, image quality
(PSNR, SSIM, LPIPS), and generation / attack / detection timings. The metric
utilities in this contribution use PSNR and SSIM from scikit-image and LPIPS
from the project dependency. Inputs must use documented image layouts and
ranges; result analysis accepts schema-compatible JSON or CSV and aggregates
only supplied observations.

The supplied summary contains clean bit accuracy, blurred bit accuracy, and
PSNR by condition only. It does not contain per-image samples, attack
strengths, sample counts, or metric computation details, so the interpretations
below are descriptive and do not establish statistical significance.

## Results

**Preliminary, user-provided summary (not independently reproduced):**

| Condition | Bit accuracy | Blurred bit accuracy | PSNR (dB) | SSIM | LPIPS | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.995 | 0.740 | 30.3 | N/A | N/A | N/A |
| defended | 0.964 | 0.901 | 23.4 | N/A | N/A | N/A |
| baseline_ft | 0.974 | 0.719 | 31.6 | N/A | N/A | N/A |
| defended_ft | 0.958 | 0.839 | 25.8 | N/A | N/A | N/A |

| Blur comparison | Baseline | Defended | Change |
|---|---:|---:|---:|
| Not fine-tuned | 0.740 | 0.901 | +16.1 percentage points |
| Fine-tuned | 0.719 | 0.839 | +12.0 percentage points |

The changes above are arithmetic differences between supplied condition
summaries; they are not significance tests.

## Interpretation

The supplied results show higher blurred bit accuracy for the defended
condition than for its corresponding baseline: +16.1 percentage points before
fine-tuning and +12.0 percentage points after fine-tuning. Clean bit accuracy
is lower for the defended conditions (0.964 versus 0.995, and 0.958 versus
0.974, respectively). The supplied PSNR is also lower for the defended
decoder: 23.4 versus 30.3 dB (a 6.9 dB decrease), and 25.8 versus 31.6 dB
after fine-tuning (a 5.8 dB decrease). This is consistent with a robustness
versus reconstruction-quality trade-off in these preliminary summaries. It
does not quantify perceptual quality beyond PSNR or support causal or
statistical claims.

## Figures and tables

Running the commands in [`README.md`](README.md) with the supplied summary
generates:

1. `figures/01_bit_accuracy_clean_vs_blur.png` — clean and blurred bit accuracy
   for all four conditions.
2. `figures/02_psnr_comparison.png` — PSNR for all four conditions.
3. `figures/03_blur_robustness_improvements.png` — the +16.1 pp and +12.0 pp
   defended-versus-baseline blur changes.
4. `figures/04_quality_vs_blur_robustness.png` — supplied PSNR versus blurred
   bit accuracy by condition.

The detection-versus-image-quality and detection-versus-runtime figures are
not generated from the supplied data. They require at least two observed
per-attack groups with actual detection accuracy and the corresponding
per-attack PSNR or runtime, respectively. The generated comparison table is
`tables/condition_comparison.csv`; analysis also writes
`tables/condition_summary.csv` and `tables/attack_summary.csv`. Unavailable
table cells are explicitly written as `N/A`.

## Limitations and missing measurements

- There are no committed per-image evaluation result files; `src/eval/evaluate.py`
  currently returns an empty frame. The tracked `results/raw/` directory
  contains only its placeholder.
- SSIM, LPIPS, per-image/attack detection outcomes, attack strengths,
  per-condition sample counts, and evaluation runtime are not available in the
  supplied summary. Fine-tuning/training elapsed time, if logged separately,
  is not a substitute for per-attack evaluation runtime.
- The repository snapshot contains no tracked source, attacked, or recovered
  image set. Consequently, no experimental image panel can be produced yet.
- The summary provides no sample-level distributions, repeated runs,
  uncertainty estimates, or statistical tests. No statistical significance is
  claimed.

## Reproducibility

These are metrics and reporting utilities, not a reproduction of the supplied
experiment. See [`README.md`](README.md) for exact commands. To reproduce the
measurements themselves, first run a completed evaluation that writes
per-image rows matching `src/common/schema.py`, preserve the raw result file,
and then pass that file to the analysis and plotting commands. Do not treat
this preliminary CSV as per-image data.
