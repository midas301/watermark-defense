"""Generate publication-ready research figures from observed result metrics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.schema import CONDITIONS
from src.metrics.analysis import (
    aggregate_by_condition_and_attack,
    condition_summary,
    load_results,
)

COLORS = {
    "baseline": "#4C78A8",
    "defended": "#F58518",
    "baseline_ft": "#72B7B2",
    "defended_ft": "#E45756",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    return plt


def _number(value):
    return np.nan if pd.isna(value) else float(value)


def _title_note(results: pd.DataFrame) -> str:
    if "provenance" not in results.columns:
        return ""
    provenance = results["provenance"].dropna().unique()
    if len(provenance) == 1 and provenance[0] == "user-provided preliminary":
        return "Preliminary, user-provided (not independently reproduced)\n"
    return ""


def _save(plt, figure, output_dir: Path, filename: str, dpi: int):
    destination = output_dir / filename
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def generate_research_figures(
    results: pd.DataFrame, output_dir: str = "report\\figures", dpi: int = 300
) -> dict:
    """Write all figures supported by actual measurements; report skipped plots.

    Figures 1–4 use condition summaries. Figures 5–6 require at least two
    observed per-attack groups with actual detection-accuracy and quality/runtime
    measurements. Missing values are left as gaps or omitted, never set to zero.
    """
    if dpi < 1:
        raise ValueError("dpi must be a positive integer")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = condition_summary(results).set_index("condition").reindex(CONDITIONS)
    generated = {}
    skipped = {}
    plt = _pyplot()
    title_note = _title_note(results)

    clean = np.array([_number(value) for value in summary["bit_accuracy"]])
    blurred = np.array([_number(value) for value in summary["blurred_bit_accuracy"]])
    if np.isfinite(clean).any() or np.isfinite(blurred).any():
        figure, axis = plt.subplots(figsize=(9, 5.2))
        positions = np.arange(len(CONDITIONS))
        width = 0.36
        axis.bar(
            positions - width / 2,
            clean * 100,
            width,
            label="Clean",
            color="#4C78A8",
        )
        axis.bar(
            positions + width / 2,
            blurred * 100,
            width,
            label="Blurred",
            color="#F58518",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(CONDITIONS)
        axis.set_ylabel("Bit accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title(f"{title_note}Clean versus blurred watermark bit accuracy")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.25)
        generated["bit_accuracy_clean_vs_blur"] = _save(
            plt, figure, output, "01_bit_accuracy_clean_vs_blur.png", dpi
        )
    else:
        skipped["bit_accuracy_clean_vs_blur"] = "No clean or blurred bit accuracy values."

    psnr = np.array([_number(value) for value in summary["psnr_db"]])
    available_psnr = np.isfinite(psnr)
    if available_psnr.any():
        figure, axis = plt.subplots(figsize=(8, 5))
        labels = np.asarray(CONDITIONS)[available_psnr]
        values = psnr[available_psnr]
        axis.bar(
            labels,
            values,
            color=[COLORS[label] for label in labels],
        )
        axis.set_ylabel("PSNR (dB)")
        axis.set_title(f"{title_note}Reconstruction PSNR by condition")
        axis.grid(axis="y", alpha=0.25)
        generated["psnr_comparison"] = _save(
            plt, figure, output, "02_psnr_comparison.png", dpi
        )
    else:
        skipped["psnr_comparison"] = "No measured PSNR values."

    paired_changes = []
    for label, baseline, defended in (
        ("Non-fine-tuned", "baseline", "defended"),
        ("Fine-tuned", "baseline_ft", "defended_ft"),
    ):
        base_value = _number(summary.loc[baseline, "blurred_bit_accuracy"])
        defended_value = _number(summary.loc[defended, "blurred_bit_accuracy"])
        if np.isfinite(base_value) and np.isfinite(defended_value):
            paired_changes.append((label, (defended_value - base_value) * 100))
    if paired_changes:
        figure, axis = plt.subplots(figsize=(7, 4.8))
        labels, changes = zip(*paired_changes)
        bars = axis.bar(
            labels,
            changes,
            color=["#F58518", "#E45756"][: len(changes)],
        )
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_ylabel("Change in blurred bit accuracy (percentage points)")
        axis.set_title(f"{title_note}Blur robustness: defended minus baseline")
        axis.grid(axis="y", alpha=0.25)
        for bar, change in zip(bars, changes):
            axis.annotate(
                f"{change:+.1f} pp",
                (bar.get_x() + bar.get_width() / 2, change),
                xytext=(0, 5 if change >= 0 else -15),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        generated["blur_robustness_improvements"] = _save(
            plt, figure, output, "03_blur_robustness_improvements.png", dpi
        )
    else:
        skipped["blur_robustness_improvements"] = (
            "Both baseline/defended and baseline_ft/defended_ft blurred accuracies "
            "are needed for their respective comparisons."
        )

    robustness = np.array(
        [_number(value) for value in summary["blurred_bit_accuracy"]]
    )
    reconstruction = np.array([_number(value) for value in summary["psnr_db"]])
    paired = np.isfinite(robustness) & np.isfinite(reconstruction)
    if paired.any():
        figure, axis = plt.subplots(figsize=(7.5, 5.5))
        for index, condition in enumerate(CONDITIONS):
            if paired[index]:
                axis.scatter(
                    reconstruction[index],
                    robustness[index] * 100,
                    color=COLORS[condition],
                    s=75,
                    label=condition,
                    zorder=3,
                )
                axis.annotate(
                    condition,
                    (reconstruction[index], robustness[index] * 100),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=9,
                )
        axis.set_xlabel("Reported PSNR (dB)")
        axis.set_ylabel("Blurred bit accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title(f"{title_note}Reconstruction quality versus blur robustness")
        axis.grid(alpha=0.25)
        generated["quality_vs_blur_robustness"] = _save(
            plt, figure, output, "04_quality_vs_blur_robustness.png", dpi
        )
    else:
        skipped["quality_vs_blur_robustness"] = (
            "No conditions have both measured PSNR and blurred bit accuracy."
        )

    attack_summary = aggregate_by_condition_and_attack(results)
    observed_attacks = attack_summary["attack"].notna()
    detection = attack_summary.loc[observed_attacks].copy()
    if "detection_accuracy" in detection.columns:
        detection = detection.dropna(subset=["detection_accuracy", "psnr_db"])
    else:
        detection = detection.iloc[0:0]
    if len(detection) >= 2:
        figure, axis = plt.subplots(figsize=(7.5, 5.5))
        for _, row in detection.iterrows():
            condition = row["condition"]
            axis.scatter(
                row["psnr_db"],
                row["detection_accuracy"] * 100,
                color=COLORS[condition],
                s=70,
                label=condition,
            )
            axis.annotate(
                f"{condition}/{row['attack']}",
                (row["psnr_db"], row["detection_accuracy"] * 100),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), frameon=False)
        axis.set_xlabel("PSNR (dB)")
        axis.set_ylabel("Detection accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title(
            f"{title_note}Watermark detection accuracy versus image quality"
        )
        axis.grid(alpha=0.25)
        generated["detection_vs_image_quality"] = _save(
            plt, figure, output, "05_detection_vs_image_quality.png", dpi
        )
    else:
        skipped["detection_vs_image_quality"] = (
            "Requires at least two per-attack groups with observed detection "
            "accuracy and PSNR."
        )

    detection = attack_summary.loc[observed_attacks].copy()
    if "detection_accuracy" in detection.columns:
        detection = detection.dropna(subset=["detection_accuracy", "runtime_s"])
    else:
        detection = detection.iloc[0:0]
    if len(detection) >= 2:
        figure, axis = plt.subplots(figsize=(7.5, 5.5))
        for _, row in detection.iterrows():
            condition = row["condition"]
            axis.scatter(
                row["runtime_s"],
                row["detection_accuracy"] * 100,
                color=COLORS[condition],
                s=70,
                label=condition,
            )
            axis.annotate(
                f"{condition}/{row['attack']}",
                (row["runtime_s"], row["detection_accuracy"] * 100),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), frameon=False)
        axis.set_xlabel("Measured runtime (s)")
        axis.set_ylabel("Detection accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title(f"{title_note}Watermark detection accuracy versus runtime")
        axis.grid(alpha=0.25)
        generated["detection_vs_runtime"] = _save(
            plt, figure, output, "06_detection_vs_runtime.png", dpi
        )
    else:
        skipped["detection_vs_runtime"] = (
            "Requires at least two per-attack groups with observed detection "
            "accuracy and runtime."
        )

    return {"generated": generated, "skipped": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/JSON result file or directory")
    parser.add_argument("--out-dir", default="report\\figures", help="PNG output directory")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    args = parser.parse_args(argv)
    result = generate_research_figures(load_results(args.input), args.out_dir, args.dpi)
    for name, path in result["generated"].items():
        print(f"Generated {name}: {path}")
    for name, reason in result["skipped"].items():
        print(f"Skipped {name}: {reason}")


if __name__ == "__main__":
    main()
