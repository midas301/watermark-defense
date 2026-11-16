"""Export condition comparison tables without filling unavailable results."""

import argparse
from pathlib import Path

import pandas as pd

from src.common.schema import CONDITIONS
from src.metrics.analysis import condition_summary, load_results

TABLE_COLUMNS = (
    "bit_accuracy",
    "blurred_bit_accuracy",
    "psnr_db",
    "ssim",
    "lpips",
    "runtime_s",
)


def build_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return all four condition rows and explicit missing values as pandas NA."""
    summary = condition_summary(results).set_index("condition")
    table = summary.reindex(CONDITIONS)[list(TABLE_COLUMNS)].reset_index()
    return table


def export_comparison_table(results: pd.DataFrame, output: str) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_comparison_table(results).to_csv(destination, index=False, na_rep="N/A")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/JSON result file or directory")
    parser.add_argument(
        "--output",
        default="report\\tables\\condition_comparison.csv",
        help="Destination comparison CSV",
    )
    args = parser.parse_args(argv)
    destination = export_comparison_table(load_results(args.input), args.output)
    print(f"Wrote comparison table to {destination}")


if __name__ == "__main__":
    main()
