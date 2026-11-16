import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.metrics import quality
from src.metrics.analysis import (
    aggregate_by_condition_and_attack,
    condition_summary,
    load_results,
    validate_results,
)
from src.metrics.panels import create_image_comparison_panel
from src.metrics.plots import _title_note, generate_research_figures
from src.metrics.tables import build_comparison_table, export_comparison_table


class _FakeLPIPS:
    def __call__(self, _reference, _test):
        return torch.tensor([0.25])


class QualityMetricTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.image[16:48, 16:48] = 128

    def test_identical_images_across_supported_input_types(self):
        expected = {"psnr": float("inf"), "ssim": 1.0, "lpips": 0.0}
        pil_image = Image.fromarray(self.image)
        tensor_image = torch.from_numpy(self.image).permute(2, 0, 1)
        self.assertEqual(quality.compute_quality(self.image, pil_image), expected)
        self.assertEqual(quality.compute_quality(tensor_image, self.image), expected)

    def test_different_images_compute_bounded_metrics(self):
        changed = self.image.astype(np.float32) / 255.0
        changed[24:40, 24:40] += 0.1
        reference_tensor = torch.from_numpy(
            self.image.astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1)
        with patch.object(
            quality, "_get_lpips_model", return_value=(_FakeLPIPS(), torch.device("cpu"))
        ):
            result = quality.compute_quality(reference_tensor, changed)
        self.assertTrue(np.isfinite(result["psnr"]))
        self.assertGreater(result["psnr"], 0)
        self.assertGreaterEqual(result["ssim"], -1)
        self.assertLessEqual(result["ssim"], 1)
        self.assertEqual(result["lpips"], 0.25)

    def test_grayscale_is_supported(self):
        reference = np.zeros((64, 64), dtype=np.uint8)
        changed = reference.copy()
        changed[20:40, 20:40] = 200
        with patch.object(
            quality, "_get_lpips_model", return_value=(_FakeLPIPS(), torch.device("cpu"))
        ):
            result = quality.compute_quality(reference, changed)
        self.assertLess(result["ssim"], 1.0)

    def test_invalid_shapes_and_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical normalized shapes"):
            quality.compute_quality(self.image, self.image[:, :-1])
        with self.assertRaisesRegex(ValueError, "at least 3 by 3"):
            quality.compute_quality(
                np.zeros((2, 64), dtype=np.uint8),
                np.zeros((2, 64), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "must be in"):
            quality.compute_quality(
                np.full((64, 64, 3), 255.0), np.zeros((64, 64, 3), dtype=np.uint8)
            )
        with self.assertRaisesRegex(ValueError, "PIL image must be"):
            rgba = Image.fromarray(np.zeros((64, 64, 4), dtype=np.uint8))
            quality.compute_quality(rgba, rgba)


class ResultsAndFigureTests(unittest.TestCase):
    def _preliminary_results(self):
        return pd.DataFrame(
            [
                ("baseline", 0.995, 0.740, 30.3),
                ("defended", 0.964, 0.901, 23.4),
                ("baseline_ft", 0.974, 0.719, 31.6),
                ("defended_ft", 0.958, 0.839, 25.8),
            ],
            columns=("condition", "bit_accuracy", "blurred_bit_accuracy", "psnr_db"),
        )

    def _per_attack_results(self):
        rows = []
        for condition, offset in (
            ("baseline", 0.0),
            ("defended", 0.05),
            ("baseline_ft", 0.1),
            ("defended_ft", 0.15),
        ):
            for attack, bit_accuracy in (("none", 0.98 - offset), ("blur", 0.75 + offset)):
                rows.append(
                    {
                        "image_id": f"{condition}-{attack}",
                        "condition": condition,
                        "attack": attack,
                        "detected": True,
                        "bit_acc": bit_accuracy,
                        "psnr": 30.0 - offset,
                        "gen_time_s": 0.2,
                        "attack_time_s": 0.1,
                        "detect_time_s": 0.05,
                    }
                )
        return pd.DataFrame(rows)

    def test_validation_and_aggregation_preserve_input(self):
        raw = self._preliminary_results()
        before = raw.copy(deep=True)
        summary = condition_summary(raw)
        self.assertEqual(raw.columns.tolist(), before.columns.tolist())
        pd.testing.assert_frame_equal(raw, before)
        self.assertAlmostEqual(summary.loc[0, "bit_accuracy"], 0.995)
        self.assertAlmostEqual(summary.loc[1, "blurred_bit_accuracy"], 0.901)
        self.assertTrue(pd.isna(summary.loc[0, "ssim"]))

        per_attack = aggregate_by_condition_and_attack(self._per_attack_results())
        blur = per_attack[per_attack["attack"] == "blur"].set_index("condition")
        self.assertAlmostEqual(blur.loc["defended", "bit_accuracy"], 0.80)
        self.assertAlmostEqual(blur.loc["baseline", "detection_accuracy"], 1.0)
        self.assertAlmostEqual(blur.loc["baseline", "runtime_s"], 0.35)
        summary_rows = aggregate_by_condition_and_attack(raw)
        self.assertTrue(summary_rows["image_count"].isna().all())

    def test_schema_rejects_bad_condition_and_bad_metrics(self):
        with self.assertRaisesRegex(ValueError, "unsupported condition"):
            validate_results(pd.DataFrame([{"condition": "other", "psnr": 1.0}]))
        with self.assertRaisesRegex(ValueError, "bit_acc values"):
            validate_results(pd.DataFrame([{"condition": "baseline", "bit_acc": 1.5}]))
        with self.assertRaisesRegex(ValueError, "unsupported strength"):
            validate_results(
                pd.DataFrame(
                    [
                        {
                            "condition": "baseline",
                            "attack": "blur",
                            "strength": "extreme",
                            "psnr": 1.0,
                        }
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "no recognized metric"):
            validate_results(pd.DataFrame([{"condition": "baseline", "image_id": "x"}]))

    def test_csv_and_json_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "summary.csv"
            json_path = Path(directory) / "rows.json"
            self._preliminary_results().to_csv(csv_path, index=False)
            json_path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "condition": "baseline",
                                "attack": "blur",
                                "bit_acc": 0.74,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            combined = load_results([csv_path, json_path])
            self.assertEqual(len(combined), 5)
            self.assertEqual(combined.loc[0, "bit_acc"], 0.995)

    def test_comparison_table_marks_missing_metrics_unavailable(self):
        table = build_comparison_table(self._preliminary_results())
        self.assertEqual(
            table["condition"].tolist(),
            ["baseline", "defended", "baseline_ft", "defended_ft"],
        )
        self.assertTrue(table["ssim"].isna().all())
        with tempfile.TemporaryDirectory() as directory:
            path = export_comparison_table(
                self._preliminary_results(), str(Path(directory) / "comparison.csv")
            )
            self.assertIn("N/A", path.read_text(encoding="utf-8"))

    def test_summary_plots_generate_four_and_skip_unavailable_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_research_figures(self._preliminary_results(), directory, dpi=50)
            self.assertEqual(len(result["generated"]), 4)
            self.assertIn("detection_vs_runtime", result["skipped"])
            tagged = self._preliminary_results()
            tagged["provenance"] = "user-provided preliminary"
            self.assertEqual(
                _title_note(tagged),
                "Preliminary, user-provided (not independently reproduced)\n",
            )
            self.assertIn(
                "04_quality_vs_blur_robustness.png",
                [path.name for path in Path(directory).glob("*.png")],
            )

    def test_per_attack_data_enables_all_six_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_research_figures(self._per_attack_results(), directory, dpi=50)
            self.assertEqual(len(result["generated"]), 6)
            self.assertTrue((Path(directory) / "06_detection_vs_runtime.png").is_file())

    def test_panels_use_only_supplied_image_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, role in enumerate(
                ("original", "watermarked", "attacked", "recovered")
            ):
                path = Path(directory) / f"{role}.png"
                Image.fromarray(
                    np.full((8, 8, 3), index * 40, dtype=np.uint8)
                ).save(path)
                paths[role] = path
            output = create_image_comparison_panel(
                paths, Path(directory) / "panel.png", dpi=50
            )
            self.assertTrue(output.is_file())
            paths["attacked"] = Path(directory) / "missing.png"
            with self.assertRaises(FileNotFoundError):
                create_image_comparison_panel(paths, Path(directory) / "absent.png", dpi=50)


if __name__ == "__main__":
    unittest.main()
