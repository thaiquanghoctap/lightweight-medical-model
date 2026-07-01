import csv
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.run_all_medical_evaluations import generate_latex, tex_escape


class BatchMedicalEvaluationTest(unittest.TestCase):
    def write_rows(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_latex_generation_uses_real_combined_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = [
                {
                    "config": "mk_mnet__width_0_25",
                    "variant": "raw",
                    "accuracy": "0.8",
                    "macro_f1": "0.7",
                    "macro_auc_ovr": "0.9",
                    "dice_lesion": "0.6",
                    "iou_lesion": "0.5",
                }
            ]
            per_class = [
                {
                    "config": "mk_mnet__width_0_25",
                    "variant": "raw",
                    "class": class_name,
                    "precision": "0.7",
                    "sensitivity": "0.8",
                    "specificity": "0.9",
                    "f1": "0.75",
                    "auc": "0.91",
                    "support": "10",
                }
                for class_name in ("benign", "malignant", "normal")
            ]
            self.write_rows(root / "all_summary.csv", summary)
            self.write_rows(root / "all_per_class_metrics.csv", per_class)
            generate_latex(root)
            generated = (root / "medical_metrics_tables.tex").read_text(encoding="utf-8")
            self.assertIn(r"mk\_mnet\_\_width\_0\_25", generated)
            self.assertIn("0.8000", generated)
            self.assertIn("malignant", generated)

    def test_tex_escape_does_not_reescape_replacement_braces(self):
        self.assertEqual(tex_escape("a_b"), r"a\_b")


if __name__ == "__main__":
    unittest.main()
