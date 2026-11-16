"""Load, validate, and aggregate experiment result files."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from src.common.schema import ATTACKS, CONDITIONS, STRENGTHS

METRIC_COLUMNS = (
    "bit_acc",
    "blurred_bit_accuracy",
    "ber",
    "psnr",
    "ssim",
    "lpips",
    "detection_score",
    "detection_accuracy",
    "detected",
    "gen_time_s",
    "attack_time_s",
    "detect_time_s",
    "runtime_s",
)
RUNTIME_COLUMNS = ("gen_time_s", "attack_time_s", "detect_time_s")
ALIASES = {"bit_accuracy": "bit_acc", "psnr_db": "psnr"}
CONDITION_ORDER = list(CONDITIONS)


def _records_from_json(payload, path: Path):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON must contain a list or object of result rows")
    for key in ("results", "rows", "records", "data"):
        if key in payload:
            if not isinstance(payload[key], list):
                raise ValueError(f"{path}: JSON field {key!r} must be a list")
            return payload[key]
    if payload and all(
        condition in CONDITIONS and isinstance(values, dict)
        for condition, values in payload.items()
    ):
        return [{"condition": condition, **values} for condition, values in payload.items()]
    if "condition" in payload:
        return [payload]
    raise ValueError(
        f"{path}: JSON must be a list of rows, contain a results/rows/records/data "
        "list, or map condition names to summary objects"
    )


def validate_results(frame: pd.DataFrame, source: str = "results") -> pd.DataFrame:
    """Return a validated canonical copy; never mutate the caller's raw frame."""
    if frame.empty:
        raise ValueError(f"{source}: result file contains no rows")
    data = frame.copy(deep=True)
    rename = {}
    for alias, canonical in ALIASES.items():
        if alias in data.columns:
            if canonical in data.columns:
                raise ValueError(
                    f"{source}: provide either {alias!r} or {canonical!r}, not both"
                )
            rename[alias] = canonical
    data = data.rename(columns=rename)

    if "condition" not in data.columns:
        raise ValueError(f"{source}: required column 'condition' is missing")
    if not set(METRIC_COLUMNS).intersection(data.columns):
        raise ValueError(
            f"{source}: no recognized metric columns; expected at least one of "
            f"{', '.join(METRIC_COLUMNS)}"
        )
    data["condition"] = data["condition"].astype("string").str.strip()
    if data["condition"].isna().any() or data["condition"].eq("").any():
        raise ValueError(f"{source}: condition values cannot be empty")
    invalid_conditions = sorted(set(data["condition"].dropna()) - set(CONDITIONS))
    if invalid_conditions:
        raise ValueError(
            f"{source}: unsupported condition(s) {invalid_conditions}; "
            f"expected {CONDITIONS}"
        )

    if "attack" not in data.columns:
        data["attack"] = pd.NA
    else:
        data["attack"] = data["attack"].astype("string").str.strip()
        data.loc[data["attack"].eq(""), "attack"] = pd.NA
        invalid_attacks = sorted(set(data["attack"].dropna()) - set(ATTACKS))
        if invalid_attacks:
            raise ValueError(
                f"{source}: unsupported attack(s) {invalid_attacks}; expected {ATTACKS}"
            )
    if "strength" in data.columns:
        data["strength"] = data["strength"].astype("string").str.strip()
        data.loc[data["strength"].eq(""), "strength"] = pd.NA
        invalid_strengths = sorted(set(data["strength"].dropna()) - set(STRENGTHS))
        if invalid_strengths:
            raise ValueError(
                f"{source}: unsupported strength(s) {invalid_strengths}; "
                f"expected {STRENGTHS}"
            )

    for column in METRIC_COLUMNS:
        if column not in data.columns:
            continue
        original = data[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"{source}: metric column {column!r} contains non-numeric values"
            )
        if column == "detected":
            invalid_bool = converted.notna() & ~converted.isin((0, 1))
            if invalid_bool.any():
                raise ValueError(f"{source}: 'detected' values must be boolean or 0/1")
        else:
            non_finite = converted.notna() & ~converted.map(np.isfinite)
            if column == "psnr":
                non_finite = converted.notna() & converted.map(
                    lambda value: not np.isfinite(value) and value != float("inf")
                )
            if non_finite.any():
                raise ValueError(f"{source}: metric column {column!r} must be finite")
        data[column] = converted

    for column in ("bit_acc", "blurred_bit_accuracy", "ber", "detection_accuracy"):
        if column in data.columns:
            invalid = data[column].notna() & ~data[column].between(0.0, 1.0)
            if invalid.any():
                raise ValueError(f"{source}: {column} values must be in [0, 1]")
    if "ssim" in data.columns:
        invalid = data["ssim"].notna() & ~data["ssim"].between(-1.0, 1.0)
        if invalid.any():
            raise ValueError(f"{source}: ssim values must be in [-1, 1]")
    if "lpips" in data.columns:
        invalid = data["lpips"].notna() & (data["lpips"] < 0.0)
        if invalid.any():
            raise ValueError(f"{source}: lpips values must be non-negative")
    for column in (*RUNTIME_COLUMNS, "runtime_s"):
        if column in data.columns:
            invalid = data[column].notna() & (data[column] < 0.0)
            if invalid.any():
                raise ValueError(f"{source}: {column} values must be non-negative")
    return data


def load_results(inputs: Union[str, Path, Iterable[Union[str, Path]]]) -> pd.DataFrame:
    """Load one or more CSV/JSON result files or all such files in a directory."""
    if isinstance(inputs, (str, Path)):
        paths = [Path(inputs)]
    else:
        paths = [Path(item) for item in inputs]
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in (".csv", ".json")
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Result input does not exist: {path}")
    if not files:
        raise FileNotFoundError("No CSV or JSON result files were found in the input")

    frames = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            raw = pd.read_csv(path)
        elif suffix == ".json":
            with path.open(encoding="utf-8") as handle:
                raw = pd.DataFrame(_records_from_json(json.load(handle), path))
        else:
            raise ValueError(f"Unsupported result format: {path.suffix}")
        frames.append(validate_results(raw, str(path)))
    return pd.concat(frames, ignore_index=True, sort=False)


def _mean(values: pd.Series):
    return values.mean() if values.notna().any() else pd.NA


def _runtime_values(group: pd.DataFrame) -> pd.Series:
    runtime = (
        group["runtime_s"].copy()
        if "runtime_s" in group.columns
        else pd.Series(pd.NA, index=group.index, dtype="Float64")
    )
    if all(column in group.columns for column in RUNTIME_COLUMNS) and runtime.isna().any():
        complete = group[list(RUNTIME_COLUMNS)].notna().all(axis=1)
        component_total = group[list(RUNTIME_COLUMNS)].sum(axis=1, min_count=3)
        can_fill = runtime.isna() & complete
        runtime.loc[can_fill] = component_total.loc[can_fill]
    return runtime


def _detection_accuracy(group: pd.DataFrame):
    if "detection_accuracy" in group.columns and group["detection_accuracy"].notna().any():
        return _mean(group["detection_accuracy"])
    if "detected" in group.columns:
        return _mean(group["detected"])
    return pd.NA


def condition_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize clean/blur bit accuracy and measured quality by condition."""
    data = validate_results(results, "results")
    rows = []
    for condition, group in data.groupby("condition", sort=False):
        has_attacks = group["attack"].notna().any()
        clean_rows = group[group["attack"].eq("none")] if has_attacks else group
        quality_rows = clean_rows
        blur_rows = group[group["attack"].eq("blur")]
        explicit_blur = (
            group["blurred_bit_accuracy"]
            if "blurred_bit_accuracy" in group.columns
            else pd.Series(dtype="float64")
        )
        runtime = _runtime_values(group)
        row = {
            "condition": condition,
            "bit_accuracy": (
                _mean(clean_rows["bit_acc"])
                if "bit_acc" in clean_rows.columns
                else pd.NA
            ),
            "blurred_bit_accuracy": (
                _mean(explicit_blur)
                if explicit_blur.notna().any()
                else (
                    _mean(blur_rows["bit_acc"])
                    if "bit_acc" in blur_rows.columns
                    else pd.NA
                )
            ),
            "psnr_db": _mean(quality_rows["psnr"])
            if "psnr" in quality_rows.columns
            else pd.NA,
            "ssim": _mean(quality_rows["ssim"])
            if "ssim" in quality_rows.columns
            else pd.NA,
            "lpips": _mean(quality_rows["lpips"])
            if "lpips" in quality_rows.columns
            else pd.NA,
            "runtime_s": _mean(runtime),
            "detection_accuracy": _detection_accuracy(group),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "condition", key=lambda column: column.map({c: i for i, c in enumerate(CONDITIONS)})
    ).reset_index(drop=True)


def aggregate_by_condition_and_attack(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed metrics for each condition/attack pair."""
    data = validate_results(results, "results")
    rows = []
    for (condition, attack), group in data.groupby(
        ["condition", "attack"], sort=False, dropna=False
    ):
        runtime = _runtime_values(group)
        row = {
            "condition": condition,
            "attack": attack,
            "image_count": (
                group["image_id"].nunique()
                if "image_id" in group.columns
                else pd.NA
            ),
            "bit_accuracy": _mean(group["bit_acc"])
            if "bit_acc" in group.columns
            else pd.NA,
            "blurred_bit_accuracy": _mean(group["blurred_bit_accuracy"])
            if "blurred_bit_accuracy" in group.columns
            else pd.NA,
            "psnr_db": _mean(group["psnr"]) if "psnr" in group.columns else pd.NA,
            "ssim": _mean(group["ssim"]) if "ssim" in group.columns else pd.NA,
            "lpips": _mean(group["lpips"]) if "lpips" in group.columns else pd.NA,
            "detection_accuracy": _detection_accuracy(group),
            "runtime_s": _mean(runtime),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/JSON result file or directory")
    parser.add_argument(
        "--out-dir", default="report\\tables", help="Directory for aggregate CSV outputs"
    )
    args = parser.parse_args(argv)
    results = load_results(args.input)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_summary(results).to_csv(
        output_dir / "condition_summary.csv", index=False, na_rep="N/A"
    )
    aggregate_by_condition_and_attack(results).to_csv(
        output_dir / "attack_summary.csv", index=False, na_rep="N/A"
    )
    print(f"Wrote condition and attack summaries to {output_dir}")


if __name__ == "__main__":
    main()
