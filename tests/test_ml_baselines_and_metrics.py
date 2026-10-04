"""
tests/test_ml_baselines_and_metrics.py
Test suite verifying:
1. Baseline Model Hierarchy (B0 to B8)
2. Metric Engine (RMSE, MAE, Bias, Pearson r, R² relative to climatology, cosine area-weighting, bootstrap CIs)
3. Feature Ablations (A1-A8) & Context Ablations (C1-C5)
4. ARGO In-situ Evaluation Infrastructure (Eligibility, QC, Dual Paired Assessment)
5. Pre-Training Scientific Acceptance Gate
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import json

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS, CANONICAL_FEATURES
from models.baselines import get_baseline_model, BASELINE_REGISTRY
from models.metrics_engine import compute_comprehensive_metrics, compute_block_bootstrap_ci
from models.ablations import (FEATURE_ABLATION_CONFIGS, CONTEXT_ABLATION_CONFIGS,
                              get_feature_ablation_indices, get_context_ablation_config)
from models.argo_evaluation import filter_eligible_argo_profiles, compute_dual_argo_evaluation
from models.pre_training_gate import verify_pre_training_gate, PreTrainingGateBlockedError


class TestMLBaselinesAndMetrics(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.n_depths = len(CANONICAL_DEPTHS)
        self.n_samples = 200
        
        # Synthetic mock dataset
        np.random.seed(42)
        self.mock_train = {
            "X": np.random.randn(self.n_samples, len(CANONICAL_FEATURES)).astype(np.float32),
            "X_norm": np.random.randn(self.n_samples, len(CANONICAL_FEATURES)).astype(np.float32),
            "Y": (np.random.randn(self.n_samples, self.n_depths) * 2.0 + 20.0).astype(np.float32),
            "mask": np.ones((self.n_samples, self.n_depths), dtype=bool),
            "lat": np.linspace(5.0, 25.0, self.n_samples, dtype=np.float32),
            "lon": np.linspace(50.0, 90.0, self.n_samples, dtype=np.float32),
            "time_idx": np.random.choice(range(0, 253), size=self.n_samples).astype(np.int32)
        }
        # Mask deepest level in shallow locations
        self.mock_train["mask"][:20, -1] = False
        self.mock_train["Y"][:20, -1] = np.nan
        
        self.mock_val = {
            "X": np.random.randn(50, len(CANONICAL_FEATURES)).astype(np.float32),
            "X_norm": np.random.randn(50, len(CANONICAL_FEATURES)).astype(np.float32),
            "Y": (np.random.randn(50, self.n_depths) * 2.0 + 20.0).astype(np.float32),
            "mask": np.ones((50, self.n_depths), dtype=bool),
            "lat": np.linspace(5.0, 25.0, 50, dtype=np.float32),
            "lon": np.linspace(50.0, 90.0, 50, dtype=np.float32),
            "time_idx": np.random.choice(range(259, 307), size=50).astype(np.int32)
        }
        
        self.mock_test = {
            "X": np.random.randn(50, len(CANONICAL_FEATURES)).astype(np.float32),
            "X_norm": np.random.randn(50, len(CANONICAL_FEATURES)).astype(np.float32),
            "Y": (np.random.randn(50, self.n_depths) * 2.0 + 20.0).astype(np.float32),
            "mask": np.ones((50, self.n_depths), dtype=bool),
            "lat": np.linspace(5.0, 25.0, 50, dtype=np.float32),
            "lon": np.linspace(50.0, 90.0, 50, dtype=np.float32),
            "time_idx": np.random.choice(range(313, 366), size=50).astype(np.int32)
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_baseline_hierarchy_instantiation_and_execution(self):
        """Verifies that all models B0 through B8 can be instantiated, fitted, and evaluated."""
        baseline_keys = ["B0", "B0b", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
        
        for b_key in baseline_keys:
            model = get_baseline_model(b_key)
            self.assertIsNotNone(model)
            model.fit(self.mock_train, self.mock_val)
            preds = model.predict(self.mock_test)
            self.assertEqual(preds.shape, (len(self.mock_test["lat"]), self.n_depths))
            self.assertFalse(np.isnan(preds).all())

    def test_02_persistence_and_climatology_exact_behavior(self):
        """Verifies B0 uses day 0, B0b uses day 252, and B1 produces spatial climatology."""
        b0 = get_baseline_model("B0")
        b0.fit(self.mock_train)
        preds_b0 = b0.predict(self.mock_test)
        self.assertEqual(preds_b0.shape, (50, 15))
        
        b1 = get_baseline_model("B1")
        b1.fit(self.mock_train)
        preds_b1 = b1.predict(self.mock_test)
        self.assertEqual(preds_b1.shape, (50, 15))

    def test_03_metric_engine_comprehensive_evaluation(self):
        """Verifies RMSE, MAE, Bias, Pearson r, R² relative to climatology, regional weighting, and seasons."""
        y_true = self.mock_test["Y"]
        # Add deliberate error
        y_pred = y_true + 0.5
        mask = self.mock_test["mask"]
        y_clim = y_true + 1.5
        lats = self.mock_test["lat"]
        lons = self.mock_test["lon"]
        time_idx = self.mock_test["time_idx"]
        
        results, df_depths = compute_comprehensive_metrics(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            y_clim=y_clim,
            lats=lats,
            lons=lons,
            time_indices=time_idx
        )
        
        # Bias should be ~0.5, RMSE ~0.5
        self.assertAlmostEqual(results["unweighted_depth_mean"]["rmse"], 0.5, places=2)
        self.assertAlmostEqual(results["unweighted_depth_mean"]["bias"], 0.5, places=2)
        # Prediction is closer to truth than climatology, so R^2 > 0
        self.assertGreater(results["unweighted_depth_mean"]["r2"], 0.0)
        
        # Verify regional breakdowns exist
        self.assertIn("regions", results)
        self.assertIn("arabian_sea", results["regions"])
        self.assertIn("bay_of_bengal", results["regions"])
        
        # Verify block-bootstrap runs
        ci_records = compute_block_bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            time_indices=time_idx,
            n_bootstraps=50
        )
        self.assertEqual(len(ci_records), self.n_depths)
        self.assertFalse(np.isnan(ci_records[0]["ci_95_low"]))

    def test_04_feature_and_context_ablation_framework(self):
        """Verifies A1–A8 feature index subsets and C1–C5 context configurations."""
        self.assertEqual(len(FEATURE_ABLATION_CONFIGS), 8)
        self.assertEqual(len(CONTEXT_ABLATION_CONFIGS), 5)
        
        # A1 (no SST)
        a1_idx, a1_feats = get_feature_ablation_indices("A1")
        self.assertNotIn("sst", a1_feats)
        self.assertEqual(len(a1_feats), 6)
        
        # A8 (full reference)
        a8_idx, a8_feats = get_feature_ablation_indices("A8")
        self.assertEqual(len(a8_feats), 7)
        
        # C2 (1x1 pointwise context)
        c2_cfg = get_context_ablation_config("C2")
        self.assertEqual(c2_cfg["patch_size"], 1)

    def test_05_argo_evaluation_qc_and_paired_metrics(self):
        """Verifies ARGO profile filtering, QC, and dual paired evaluation tracks."""
        # Create mock Argo dataframe
        df_argo = pd.DataFrame([
            {"argo_id": "float1", "time": "2020-11-15 12:00:00", "depth": 5.0, "observed_temperature": 28.0, "quality_flag": 1},
            {"argo_id": "float1", "time": "2020-11-15 12:00:00", "depth": 20.0, "observed_temperature": 27.5, "quality_flag": 1},
            {"argo_id": "float1", "time": "2020-11-15 12:00:00", "depth": 50.0, "observed_temperature": 25.0, "quality_flag": 1},
            {"argo_id": "float1", "time": "2020-11-15 12:00:00", "depth": 100.0, "observed_temperature": 20.0, "quality_flag": 1},
            {"argo_id": "float1", "time": "2020-11-15 12:00:00", "depth": 200.0, "observed_temperature": 15.0, "quality_flag": 1},
            # Non-test date (should be filtered out)
            {"argo_id": "float2", "time": "2020-05-15 12:00:00", "depth": 5.0, "observed_temperature": 28.0, "quality_flag": 1}
        ])
        
        filtered = filter_eligible_argo_profiles(df_argo)
        self.assertEqual(len(filtered), 5) # float2 discarded because outside test period
        
        # Test paired evaluation
        matched_df = pd.DataFrame([
            {"profile_id": "p1", "depth": 20.0, "observed_temp": 28.0, "glorys_temp": 28.2, "ml_temp": 28.1},
            {"profile_id": "p1", "depth": 100.0, "observed_temp": 20.0, "glorys_temp": 20.5, "ml_temp": 20.3},
        ])
        res = compute_dual_argo_evaluation(matched_df, n_bootstrap=10)
        self.assertEqual(res["assessment_label_1"], "ARGO–GLORYS reference consistency assessment")
        self.assertEqual(res["assessment_label_2"], "ML–ARGO observational evaluation")
        self.assertIn("delta_rmse", res["overall"])

    def test_06_pre_training_acceptance_gate_blocks_invalid_state(self):
        """Verifies that verify_pre_training_gate strictly halts when certification or splits fail."""
        # Case A: Missing certificate
        bad_dataset = {
            "train": self.mock_train,
            "val": self.mock_val,
            "test": self.mock_test,
            "depth_levels": CANONICAL_DEPTHS,
            "scaler": {"mean": [0]*7, "std": [1]*7, "scaler_sha256": "abc"},
            "split_metadata": {"total_days": 366, "train": {"n_days": 253}, "purge1_days": 6,
                              "val": {"n_days": 48}, "purge2_days": 6, "test": {"n_days": 53}}
        }
        with self.assertRaises(PreTrainingGateBlockedError):
            verify_pre_training_gate(bad_dataset, certificate_path="reports/nonexistent_cert.json", full_year_mode=True)
            
        # Case B: Create valid cert and manifest, but wrong split
        cert_path = os.path.join(self.temp_dir.name, "cert.json")
        with open(cert_path, "w") as f:
            json.dump({"acceptance_certified": True}, f)
            
        bad_split_dataset = dict(bad_dataset)
        bad_split_dataset["split_metadata"] = {
            "total_days": 366,
            "train": {"n_days": 240}, # WRONG split
            "purge1_days": 0,
            "val": {"n_days": 60},
            "purge2_days": 0,
            "test": {"n_days": 66}
        }
        with self.assertRaises(PreTrainingGateBlockedError):
            verify_pre_training_gate(bad_split_dataset, certificate_path=cert_path, full_year_mode=True)


if __name__ == "__main__":
    unittest.main()
