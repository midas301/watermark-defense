# Metrics and report utilities

Run commands from the repository root, after installing the dependencies in
`requirements.txt`.

## Analyze and compare results

The included [preliminary_results.csv](preliminary_results.csv) contains only
the four user-provided summary rows; it is explicitly preliminary and has not
been independently reproduced. Its `provenance` field is metadata; the four
measurement values are preserved as supplied. Figures from this file carry the
same preliminary/user-provided notice in their titles.

```powershell
python -m src.metrics.analysis --input report\preliminary_results.csv --out-dir report\tables
python -m src.metrics.tables --input report\preliminary_results.csv --output report\tables\condition_comparison.csv
```

The analysis command writes `condition_summary.csv` and
`attack_summary.csv`. The table command writes all four condition rows and
marks unavailable bit accuracy, blurred accuracy, PSNR, SSIM, LPIPS, and
runtime fields as `N/A`. CSV and JSON inputs must include the `condition`
column and at least one recognized metric. Per-image CSV columns follow
`src/common/schema.py`; summary aliases `bit_accuracy` and `psnr_db` are also
accepted. JSON may be a list of row objects, an object containing a
`results`/`rows`/`records`/`data` list, a single result object, or a mapping of
condition names to summary objects. Only the four project conditions and
declared attack names are accepted.

For collected evaluation metrics, use the result file itself:

```powershell
python -m src.metrics.analysis --input results\raw\evaluation.csv --out-dir report\tables
python -m src.metrics.tables --input results\raw\evaluation.csv --output report\tables\condition_comparison.csv
```

A directory can be used as input if it contains only compatible metric
result files. Do not point it at a directory containing training logs or other
JSON files that do not follow a result schema. The analysis functions accept a
list of paths when combining multiple files programmatically.

## Generate figures

```powershell
python -m src.metrics.plots --input report\preliminary_results.csv --out-dir report\figures --dpi 300
```

The supplied summary produces the first four PNGs described in
[findings.md](findings.md). The detection-versus-quality and
detection-versus-runtime figures are omitted with an explanatory message until
there are at least two per-attack groups with the actual required measurements.
No absent metric is replaced with zero. When you collect a compatible
per-image results CSV, pass it as `--input` to enable those plots.
End-to-end runtime is reported only when an explicit `runtime_s` is present,
or all three schema components (`gen_time_s`, `attack_time_s`, and
`detect_time_s`) are present for a row. A partial set of timings is not
presented as total runtime.

## Compare actual images

The panel utility requires four existing files and saves only a visualization
of those files:

```powershell
python -m src.metrics.panels `
  --original data\images\original.png `
  --watermarked data\images\watermarked.png `
  --attacked data\images\attacked.png `
  --recovered data\images\recovered.png `
  --output report\figures\image_comparison.png
```

No example or synthetic images are included as experimental results. Replace
the paths with actual project outputs; the command fails if any input is
missing.

## Image quality API and tests

`src.metrics.quality.compute_quality(ref_img, test_img)` returns `psnr` in dB,
`ssim`, and `lpips`. PIL inputs must be `L` or `RGB`; NumPy uses HWC and Torch
uses CHW. Accepted pixels are `uint8` in `[0, 255]` or floats in `[0, 1]` /
`[-1, 1]`. Float values in `[0, 255]`, alpha channels, mismatched shapes, and
ambiguous channel layouts are rejected. Grayscale is replicated to RGB for
LPIPS. An exact match reports infinite PSNR, SSIM 1, and LPIPS 0.

Run focused, model-free tests from the repository root:

```powershell
python -m unittest src.metrics.test_metrics
```

Non-identical quality tests use a test double for the LPIPS model; they do not
download model weights. A real LPIPS evaluation loads and caches the configured
pretrained model once per process. If torchvision's pretrained AlexNet weights
are not already cached, its first real LPIPS call may fetch them; provision
those weights separately in an approved environment before evaluating. No
pretrained model weights were downloaded for this contribution.
