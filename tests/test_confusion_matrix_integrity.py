"""
tests/test_confusion_matrix_integrity.py
Strict numerical integrity test for the Thermal Regime Confusion Matrix benchmark.
Guarantees:
1. Every stored metric in results/confusion_matrix.json reproduces exactly from its stored matrix.
2. Sum of each matrix == N (8,017,734).
3. Diagonal accuracy, within ±1 bin, beyond ±1 bin, Cohen's kappa, macro F1, and weighted F1 match to defined precision.
4. Per-class Precision, Recall, F1, and support reproduce exactly.
5. Interactive UI artifact (confusion_matrix.html) is 100% synchronized with confusion_matrix.json.
"""

import os
import re
import json
import unittest
import numpy as np


class TestConfusionMatrixIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.json_path = os.path.join(cls.repo_root, "results", "confusion_matrix.json")
        with open(cls.json_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)
        cls.n_expected = 8017734

    def test_01_json_structure_and_total_samples(self):
        """Verify total evaluated sample count and required models."""
        self.assertEqual(self.data["total_test_samples"], self.n_expected)
        required_models = [
            "B1_Climatology",
            "B2_Ridge",
            "B3_RandomForest",
            "B5_PointwiseMLP",
            "Legacy_MLP_128_64"
        ]
        for m in required_models:
            self.assertIn(m, self.data["models"], f"Missing model: {m}")

    def test_02_all_models_reproduce_metrics_from_matrices(self):
        """Verify that every metric in confusion_matrix.json reproduces from its matrix."""
        for m_key, m_val in self.data["models"].items():
            cm = np.array(m_val["confusion_matrix"], dtype=np.int64)
            n = int(np.sum(cm))
            self.assertEqual(n, self.n_expected, f"Matrix sum for {m_key} != {self.n_expected}")

            diag = int(np.trace(cm))
            acc = round((diag / n) * 100.0, 2)
            self.assertAlmostEqual(acc, m_val["accuracy_pct"], places=2,
                                   msg=f"Accuracy mismatch for {m_key}: {acc} vs {m_val['accuracy_pct']}")

            within1_cnt = 0
            beyond1_cnt = 0
            for i in range(6):
                for j in range(6):
                    if abs(i - j) <= 1:
                        within1_cnt += cm[i, j]
                    else:
                        beyond1_cnt += cm[i, j]

            self.assertEqual(within1_cnt + beyond1_cnt, n, f"Sum of within1 and beyond1 != N for {m_key}")
            within1_pct = round((within1_cnt / n) * 100.0, 2)
            beyond1_pct = round((beyond1_cnt / n) * 100.0, 2)
            self.assertAlmostEqual(within1_pct, m_val["within_1_bin_pct"], places=2,
                                   msg=f"Within ±1 mismatch for {m_key}: {within1_pct} vs {m_val['within_1_bin_pct']}")
            self.assertAlmostEqual(beyond1_pct, m_val["beyond_1_bin_pct"], places=2,
                                   msg=f"Beyond ±1 mismatch for {m_key}: {beyond1_pct} vs {m_val['beyond_1_bin_pct']}")

            # Cohen's kappa
            row_sums = np.sum(cm, axis=1)
            col_sums = np.sum(cm, axis=0)
            po = diag / n
            pe = float(np.sum(row_sums * col_sums)) / (n * n)
            kappa = round((po - pe) / (1.0 - pe), 4)
            self.assertAlmostEqual(kappa, m_val["cohen_kappa"], places=4,
                                   msg=f"Kappa mismatch for {m_key}: {kappa} vs {m_val['cohen_kappa']}")

            # Per-class metrics
            f1_list = []
            for i, pc in enumerate(m_val["per_class"]):
                tp = int(cm[i, i])
                fp = int(col_sums[i] - tp)
                fn = int(row_sums[i] - tp)
                self.assertEqual(pc["support"], int(row_sums[i]), f"Support mismatch for {m_key} class {i}")
                self.assertEqual(pc["true_positives"], tp, f"TP mismatch for {m_key} class {i}")
                self.assertEqual(pc["false_positives"], fp, f"FP mismatch for {m_key} class {i}")
                self.assertEqual(pc["false_negatives"], fn, f"FN mismatch for {m_key} class {i}")

                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                self.assertAlmostEqual(round(p, 4), pc["precision"], places=4,
                                       msg=f"Precision mismatch for {m_key} class {i}")
                self.assertAlmostEqual(round(r, 4), pc["recall"], places=4,
                                       msg=f"Recall mismatch for {m_key} class {i}")
                self.assertAlmostEqual(round(f1, 4), pc["f1"], places=4,
                                       msg=f"F1 mismatch for {m_key} class {i}")
                f1_list.append(round(f1, 4))

            macro_f1 = round(float(np.mean(f1_list)), 4)
            weighted_f1 = round(float(np.sum(np.array(f1_list) * row_sums) / n), 4)
            self.assertAlmostEqual(macro_f1, m_val["macro_f1"], places=4,
                                   msg=f"Macro F1 mismatch for {m_key}: {macro_f1} vs {m_val['macro_f1']}")
            self.assertAlmostEqual(weighted_f1, m_val["weighted_f1"], places=4,
                                   msg=f"Weighted F1 mismatch for {m_key}: {weighted_f1} vs {m_val['weighted_f1']}")

    def test_03_html_artifact_matrix_synchronization(self):
        """Verify that confusion_matrix.html contains the exact same matrices as confusion_matrix.json."""
        html_path = r"C:\Users\ASUS\.gemini\antigravity\brain\801295e5-d465-4ab6-84f6-0e0cc01ce3e8\confusion_matrix.html"
        if not os.path.exists(html_path):
            self.skipTest("confusion_matrix.html artifact not found at expected path")

        with open(html_path, "r", encoding="utf-8") as f:
            html_lines = f.readlines()

        models_map = {
            "b3:": "B3_RandomForest",
            "b5:": "B5_PointwiseMLP",
            "b2:": "B2_Ridge",
            "b1:": "B1_Climatology",
            "leg:": "Legacy_MLP_128_64"
        }

        current_m = None
        cm_lines = {}
        in_cm = False

        for line in html_lines:
            s = line.strip()
            for mk in models_map.keys():
                if s.startswith(mk):
                    current_m = models_map[mk]
                    cm_lines[current_m] = []
                    in_cm = False
            if current_m:
                if "cm: [" in s:
                    in_cm = True
                    continue
                if in_cm:
                    if s.startswith("]"):
                        in_cm = False
                        current_m = None
                    else:
                        row_str = s.rstrip(",")
                        cm_lines[current_m].append(json.loads(row_str))

        for model_name in models_map.values():
            self.assertIn(model_name, cm_lines, f"Missing {model_name} in HTML artifact")
            json_cm = self.data["models"][model_name]["confusion_matrix"]
            diff = np.array(cm_lines[model_name]) - np.array(json_cm)
            self.assertTrue(np.all(diff == 0),
                            f"HTML matrix for {model_name} does not match JSON! Max diff: {np.max(np.abs(diff))}")


if __name__ == "__main__":
    unittest.main()
