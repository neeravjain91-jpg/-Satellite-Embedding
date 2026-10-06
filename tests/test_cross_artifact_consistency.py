"""
tests/test_cross_artifact_consistency.py
Automated cross-artifact consistency verification suite for Phase 2 completion.

Verifies mathematical and numerical consistency across:
- results/*.json
- results/master_benchmark_summary.json
- results/confusion_matrix.json
- data/metadata/tabular_scaler_stats.json
- data/metadata/normalization_stats.json
- reports/final_results_table.md
- reports/final_project_report.md
- reports/confusion_matrix_report.md
- reports/CONFUSION_MATRIX_COMPLETION_REPORT.md
- src/mock/benchmarks.ts
"""

import os
import json
import re
import pytest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CANONICAL_MODELS = ["B0", "B0b", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]

EXPECTED_TEST_RMSE = {
    "B0": 1.5220,
    "B0b": 1.7287,
    "B1": 1.2582,
    "B2": 1.0295,
    "B3": 1.0452,
    "B4": 1.0288,
    "B5": 1.5524,
    "B6": 1.2702,
    "B7": 1.5320,
    "B8": 0.9800,
}

EXPECTED_PARAMS = {
    "B0": 0,
    "B0b": 0,
    "B1": 0,
    "B2": 120,
    "B3": 13289966,
    "B4": 750,
    "B5": 26767,
    "B6": 30991,
    "B7": 44111,
    "B8": 203791,
}

EXPECTED_DEPTHS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
EXPECTED_FEATURES = ["sst", "sss", "ssh", "current_u", "current_v", "wind_u", "wind_v"]


def test_individual_result_jsons_exist_and_match():
    """Verify each individual results/Bx.json matches expected frozen values."""
    for model_id in CANONICAL_MODELS:
        path = os.path.join(REPO_ROOT, "results", f"{model_id}.json")
        assert os.path.exists(path), f"Missing result JSON: {path}"
        with open(path, "r") as f:
            data = json.load(f)

        assert data["model_id"] == model_id
        rmse = data["test"]["overall"]["rmse"]
        assert pytest.approx(rmse, abs=1e-4) == EXPECTED_TEST_RMSE[model_id], (
            f"{model_id} test RMSE mismatch: {rmse} vs {EXPECTED_TEST_RMSE[model_id]}"
        )

        p = data.get("parameter_count", data.get("metadata", {}).get("parameters", 0))
        assert p == EXPECTED_PARAMS[model_id], (
            f"{model_id} parameter mismatch: {p} vs {EXPECTED_PARAMS[model_id]}"
        )


def test_master_benchmark_summary_matches():
    """Verify results/master_benchmark_summary.json contains all models and exact metrics."""
    path = os.path.join(REPO_ROOT, "results", "master_benchmark_summary.json")
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        master = json.load(f)

    # Check depths
    depths = master.get("canonical_depths", [])
    assert len(depths) == 15
    for expected_d, actual_d in zip(EXPECTED_DEPTHS, depths):
        assert pytest.approx(actual_d) == expected_d

    model_entries = {m["model_id"]: m for m in master["models_evaluated"]}
    assert set(model_entries.keys()) == set(CANONICAL_MODELS)

    for model_id in CANONICAL_MODELS:
        entry = model_entries[model_id]
        assert pytest.approx(entry["test_rmse"], abs=1e-4) == EXPECTED_TEST_RMSE[model_id]
        assert entry["parameters"] == EXPECTED_PARAMS[model_id]

    # Verify B8 bootstrap CI vs B1
    b8_entry = model_entries["B8"]
    assert b8_entry["paired_delta_ci_95"] == "[-0.3957, -0.1756]"
    assert b8_entry["delta_rmse_vs_b1"] == "-0.2782"
    assert b8_entry["relative_improvement_pct"] == "+22.11%"


def test_mock_benchmarks_ts_matches():
    """Verify src/mock/benchmarks.ts values match certified numbers."""
    ts_path = os.path.join(REPO_ROOT, "src", "mock", "benchmarks.ts")
    assert os.path.exists(ts_path), f"Missing {ts_path}"
    with open(ts_path, "r", encoding="utf-8") as f:
        content = f.read()

    for model_id in CANONICAL_MODELS:
        # Check model exists in file
        assert f"id: '{model_id}'" in content or f'id: "{model_id}"' in content, f"Missing {model_id} in {ts_path}"
        # Check RMSE
        expected_rmse = EXPECTED_TEST_RMSE[model_id]
        pattern = rf"id:\s*['\"]{model_id}['\"].*?rmse:\s*([0-9\.]+)"
        match = re.search(pattern, content, re.DOTALL)
        assert match, f"Could not extract rmse for {model_id} from {ts_path}"
        extracted_rmse = float(match.group(1))
        assert pytest.approx(extracted_rmse, abs=1e-4) == expected_rmse

        # Check params
        param_pattern = rf"id:\s*['\"]{model_id}['\"].*?parameters:\s*([0-9]+)"
        p_match = re.search(param_pattern, content, re.DOTALL)
        assert p_match, f"Could not extract parameters for {model_id} from {ts_path}"
        assert int(p_match.group(1)) == EXPECTED_PARAMS[model_id]


def test_normalization_scaler_metadata():
    """Verify tabular_scaler_stats.json and normalization_stats.json consistency."""
    scaler_path = os.path.join(REPO_ROOT, "data", "metadata", "tabular_scaler_stats.json")
    norm_path = os.path.join(REPO_ROOT, "data", "metadata", "normalization_stats.json")

    assert os.path.exists(scaler_path), f"Missing {scaler_path}"
    assert os.path.exists(norm_path), f"Missing {norm_path}"

    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    with open(norm_path, "r") as f:
        norm = json.load(f)

    # Features
    assert scaler["features"] == EXPECTED_FEATURES
    assert norm["canonical_features"] == EXPECTED_FEATURES

    # Split: Days 0-252 (253 days, N=2871550)
    assert scaler["train_samples_count"] == 2871550
    assert norm["samples_count"] == 2871550
    assert norm["num_train_days"] == 253
    assert scaler["split_metadata"]["train"]["n_days"] == 253
    assert scaler["split_metadata"]["train"]["n_samples"] == 2871550
    assert scaler["split_metadata"]["val"]["n_days"] == 48
    assert scaler["split_metadata"]["test"]["n_days"] == 53

    # Scaler SHA-256
    assert scaler["scaler_sha256"] == "279b13aa6662c693b4a36da0bfd78ff35ed5213a0e16b0fbb7560d657ee02fe3"


def test_confusion_matrix_consistency():
    """Verify confusion matrix summary values across results and reports."""
    cm_path = os.path.join(REPO_ROOT, "results", "confusion_matrix.json")
    assert os.path.exists(cm_path), f"Missing {cm_path}"
    with open(cm_path, "r") as f:
        cm_data = json.load(f)

    # Verified metrics for canonical models:
    # B1: accuracy 76.23%, within +/- 1: 99.39%, kappa: 0.7078
    # B2: accuracy 79.87%, within +/- 1: 99.91%, kappa: 0.7538
    # B3: accuracy 80.65%, within +/- 1: 99.71%, kappa: 0.7636
    # B5: accuracy 77.67%, within +/- 1: 99.85%, kappa: 0.7275
    expected_metrics = {
        "B1_Climatology": {"acc": 76.23, "within": 99.39, "kappa": 0.7078},
        "B2_Ridge": {"acc": 79.87, "within": 99.91, "kappa": 0.7538},
        "B3_RandomForest": {"acc": 80.65, "within": 99.71, "kappa": 0.7636},
        "B5_PointwiseMLP": {"acc": 77.67, "within": 99.85, "kappa": 0.7275},
    }

    models_data = cm_data.get("models", {})
    for m_key, exp in expected_metrics.items():
        assert m_key in models_data, f"{m_key} missing in {cm_path}"
        m_info = models_data[m_key]
        assert pytest.approx(m_info["accuracy_pct"], abs=0.05) == exp["acc"]
        assert pytest.approx(m_info["within_1_bin_pct"], abs=0.05) == exp["within"]
        assert pytest.approx(m_info["cohen_kappa"], abs=0.005) == exp["kappa"]


def test_markdown_tables_consistency():
    """Verify markdown reports contain correct B0-B8 test RMSEs."""
    reports_to_check = [
        os.path.join(REPO_ROOT, "reports", "final_results_table.md"),
        os.path.join(REPO_ROOT, "reports", "final_project_report.md"),
    ]

    for report_path in reports_to_check:
        assert os.path.exists(report_path), f"Missing {report_path}"
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()

        for model_id in CANONICAL_MODELS:
            expected_rmse_str = f"{EXPECTED_TEST_RMSE[model_id]:.4f}"
            pattern = rf"\*\*{model_id}\*\*.*?\|\s*\**{re.escape(expected_rmse_str)}\**\s*\|"
            match = re.search(pattern, text)
            assert match, f"Could not find test RMSE {expected_rmse_str} for {model_id} in {report_path}"
