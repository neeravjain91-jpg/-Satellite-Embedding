"""
scripts/exploratory_b3_study.py
Controlled Exploratory Hyperparameter Study for B3 Random Forest Regressor.

Strict Scientific Constraints:
1. Training strictly on canonical TRAIN partition (Days 0..252).
2. Model selection strictly on VALIDATION partition (Days 259..306).
3. The frozen TEST split (Days 313..365) is NEVER used for hyperparameter tuning.
4. Only the winning configuration is evaluated ONCE on the held-out test split.
5. All 7 canonical surface predictors, train-only normalization, and authoritative
   target-NaN masking semantics are preserved.
6. Results are saved strictly to results/exploratory/; certified production artifacts
   (results/B3.json, master_benchmark_summary.json, confusion_matrix.json) remain untouched.
"""

import os
import sys
import time
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import RandomForestRegressor

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS, CANONICAL_FEATURES
from preprocessing.tabular_dataset import load_tabular_dataset
from models.baselines import B1_Climatology
from models.metrics_engine import compute_comprehensive_metrics
from scripts.evaluate_confusion_matrix import evaluate_regimes


CONFIGS = {
    "B3-A": {
        "config_id": "B3-A",
        "description": "Deeper forest (100 trees), max_depth=15, 100k samples",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "sample_train_size": 100000,
        "random_state": 42
    },
    "B3-B": {
        "config_id": "B3-B",
        "description": "Increased tree depth (100 trees), max_depth=20, 100k samples",
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "sample_train_size": 100000,
        "random_state": 42
    },
    "B3-C": {
        "config_id": "B3-C",
        "description": "Large ensemble (150 trees), max_depth=18, 200k samples",
        "n_estimators": 150,
        "max_depth": 18,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "sample_train_size": 200000,
        "random_state": 42
    }
}


def train_rf_depthwise(cfg, train_data):
    """
    Fits 15 depth-wise Random Forest models strictly on valid targets.
    """
    X_train = train_data["X_norm"]
    Y_train = train_data["Y"]
    M_train = train_data["mask"]

    N_train = len(X_train)
    np.random.seed(cfg["random_state"])
    sample_size = min(cfg["sample_train_size"], N_train)
    sub_idx = np.random.choice(N_train, size=sample_size, replace=False)

    X_sub = X_train[sub_idx]
    Y_sub = Y_train[sub_idx]
    M_sub = M_train[sub_idx]

    depth_models = {}
    n_depths = len(CANONICAL_DEPTHS)

    t0 = time.time()
    for d in range(n_depths):
        valid_d = M_sub[:, d]
        if not np.any(valid_d):
            depth_models[d] = None
            continue

        X_d = X_sub[valid_d]
        y_d = Y_sub[valid_d, d]

        rf = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            max_features=cfg.get("max_features", 1.0),
            n_jobs=-1,
            random_state=cfg["random_state"] + d
        )
        rf.fit(X_d, y_d)
        depth_models[d] = rf
    t_train = time.time() - t0

    # Complexity
    total_nodes = sum(
        sum(tree.tree_.node_count for tree in reg.estimators_)
        for reg in depth_models.values()
        if reg is not None
    )
    total_trees = sum(
        len(reg.estimators_)
        for reg in depth_models.values()
        if reg is not None
    )

    return depth_models, t_train, total_nodes, total_trees


def predict_rf_depthwise(depth_models, eval_data):
    """
    Predicts 15-depth vertical temperature profiles.
    """
    X = eval_data["X_norm"]
    N = len(X)
    n_depths = len(CANONICAL_DEPTHS)
    preds = np.zeros((N, n_depths), dtype=np.float32)

    for d in range(n_depths):
        if depth_models.get(d) is not None:
            preds[:, d] = depth_models[d].predict(X)
        else:
            preds[:, d] = 15.0
    return preds


def main():
    print("=" * 80)
    print("CONTROLLED EXPLORATORY HYPERPARAMETER STUDY FOR B3 RANDOM FOREST")
    print("=" * 80)

    # 1. Load canonical dataset partitions
    print("\n[STEP 1/5] Loading canonical tabular dataset...")
    dataset = load_tabular_dataset(build_context=False)
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]

    print(f"  Train: {len(train_data['X_norm']):,} rows ({train_data['mask'].sum():,} valid targets)")
    print(f"  Val:   {len(val_data['X_norm']):,} rows ({val_data['mask'].sum():,} valid targets)")
    print(f"  Test:  {len(test_data['X_norm']):,} rows ({test_data['mask'].sum():,} valid targets)")

    # 2. Climatology Anchor on Train
    print("\n[STEP 2/5] Fitting Climatology Anchor (B1) on TRAIN...")
    b1 = B1_Climatology().fit(train_data)
    y_clim_val = b1.predict(val_data)
    y_clim_test = b1.predict(test_data)

    # 3. Reference Certified B3 Metrics
    certified_b3_path = os.path.join(repo_root, "results", "B3.json")
    with open(certified_b3_path, "r", encoding="utf-8") as f:
        cert_b3_data = json.load(f)

    cert_cm_path = os.path.join(repo_root, "results", "confusion_matrix.json")
    with open(cert_cm_path, "r", encoding="utf-8") as f:
        cert_cm_data = json.load(f)
    b3_cm = cert_cm_data["models"]["B3_RandomForest"]

    certified_b3 = {
        "model_id": "Certified B3",
        "description": "Certified Production B3 (50 trees, max_depth=15, 100k samples)",
        "parameters": {
            "n_estimators": 50,
            "max_depth": 15,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "sample_train_size": 100000,
            "random_state": 42
        },
        "complexity": {
            "decision_nodes": cert_b3_data["parameter_count"],
            "total_trees": 750
        },
        "validation": {
            "rmse": cert_b3_data["validation"]["overall"]["rmse"],
            "mae": cert_b3_data["validation"]["overall"]["mae"],
            "bias": cert_b3_data["validation"]["overall"]["bias"],
            "r2": cert_b3_data["validation"]["overall"]["r2"],
            "depth_breakdown": {
                round(float(d["depth_m"]), 1): round(float(d["rmse"]), 4)
                for d in cert_b3_data["validation"]["depth_breakdown"]
            }
        },
        "test": {
            "rmse": cert_b3_data["test"]["overall"]["rmse"],
            "mae": cert_b3_data["test"]["overall"]["mae"],
            "bias": cert_b3_data["test"]["overall"]["bias"],
            "r2": cert_b3_data["test"]["overall"]["r2"],
            "thermal_classification": {
                "accuracy_pct": b3_cm["accuracy_pct"],
                "within_1_bin_pct": b3_cm["within_1_bin_pct"],
                "beyond_1_bin_pct": b3_cm["beyond_1_bin_pct"],
                "cohen_kappa": b3_cm["cohen_kappa"],
                "macro_f1": b3_cm["macro_f1"],
                "weighted_f1": b3_cm["weighted_f1"]
            },
            "depth_breakdown": {
                round(float(d["depth_m"]), 1): round(float(d["rmse"]), 4)
                for d in cert_b3_data["test"]["depth_breakdown"]
            }
        }
    }

    # Evaluate Certified B3 on Validation thermal regimes for complete baseline comparison
    print("  Evaluating Certified B3 on Validation thermal classification...")
    cert_rf_models, _, _, _ = train_rf_depthwise(certified_b3["parameters"], train_data)
    cert_preds_val = predict_rf_depthwise(cert_rf_models, val_data)
    M_val = val_data["mask"]
    cert_val_thermal = evaluate_regimes(val_data["Y"][M_val], cert_preds_val[M_val])
    certified_b3["validation"]["thermal_classification"] = {
        "accuracy_pct": cert_val_thermal["accuracy_pct"],
        "within_1_bin_pct": cert_val_thermal["within_1_bin_pct"],
        "beyond_1_bin_pct": cert_val_thermal["beyond_1_bin_pct"],
        "cohen_kappa": cert_val_thermal["cohen_kappa"],
        "macro_f1": cert_val_thermal["macro_f1"],
        "weighted_f1": cert_val_thermal["weighted_f1"]
    }
    print(f"  Certified B3 Val RMSE: {certified_b3['validation']['rmse']} °C | Val Thermal Acc: {certified_b3['validation']['thermal_classification']['accuracy_pct']}%")

    # 4. Train and Evaluate Candidates on VALIDATION ONLY
    print("\n[STEP 3/5] Training candidate configurations & evaluating on VALIDATION...")
    candidate_results = {}
    trained_models = {}

    for cfg_id, cfg in CONFIGS.items():
        print(f"\n--- Training {cfg_id}: {cfg['description']} ---")
        models, t_train, total_nodes, total_trees = train_rf_depthwise(cfg, train_data)
        trained_models[cfg_id] = models

        print(f"  [Trained in {t_train:.1f}s] Complexity: {total_nodes:,} nodes across {total_trees} trees")

        # Predict on validation (timed)
        t_eval_0 = time.time()
        preds_val = predict_rf_depthwise(models, val_data)
        t_eval = time.time() - t_eval_0

        # Continuous metrics
        val_comp, _ = compute_comprehensive_metrics(
            y_true=val_data["Y"],
            y_pred=preds_val,
            mask=val_data["mask"],
            lats=val_data["lat"],
            lons=val_data["lon"],
            time_indices=val_data["time_idx"],
            y_clim=y_clim_val
        )

        # Thermal regime classification
        val_thermal = evaluate_regimes(val_data["Y"][M_val], preds_val[M_val])

        val_rmse = val_comp["unweighted_depth_mean"]["rmse"]
        val_mae = val_comp["unweighted_depth_mean"]["mae"]
        val_r2 = val_comp["unweighted_depth_mean"]["r2"]
        val_bias = val_comp["unweighted_depth_mean"]["bias"]

        print(f"  Val RMSE: {val_rmse:.4f} °C | Val MAE: {val_mae:.4f} °C | Val R²: {val_r2:.4f}")
        print(f"  Val Thermal Accuracy: {val_thermal['accuracy_pct']}% | ±1-bin: {val_thermal['within_1_bin_pct']}% | Kappa: {val_thermal['cohen_kappa']:.4f}")
        print(f"  Inference Time: {t_eval:.2f}s")

        candidate_results[cfg_id] = {
            "config_id": cfg_id,
            "description": cfg["description"],
            "parameters": cfg,
            "training_time_seconds": round(t_train, 2),
            "eval_time_seconds": round(t_eval, 2),
            "complexity": {
                "decision_nodes": total_nodes,
                "total_trees": total_trees
            },
            "validation": {
                "rmse": round(val_rmse, 4),
                "mae": round(val_mae, 4),
                "bias": round(val_bias, 4),
                "r2": round(val_r2, 4),
                "thermal_classification": {
                    "accuracy_pct": val_thermal["accuracy_pct"],
                    "within_1_bin_pct": val_thermal["within_1_bin_pct"],
                    "beyond_1_bin_pct": val_thermal["beyond_1_bin_pct"],
                    "cohen_kappa": val_thermal["cohen_kappa"],
                    "macro_f1": val_thermal["macro_f1"],
                    "weighted_f1": val_thermal["weighted_f1"]
                },
                "depth_breakdown": {
                    round(float(d["depth_m"]), 1): round(float(d["rmse"]), 4)
                    for d in val_comp["depth_breakdown"]
                }
            }
        }

    # 5. Rank configurations primarily by VALIDATION RMSE (primary) and thermal accuracy (secondary)
    print("\n[STEP 4/5] Ranking configurations by VALIDATION RMSE (primary) and thermal accuracy (secondary)...")
    ranked_configs = sorted(
        candidate_results.keys(),
        key=lambda k: (
            candidate_results[k]["validation"]["rmse"],
            -candidate_results[k]["validation"]["thermal_classification"]["accuracy_pct"]
        )
    )

    for rank, cid in enumerate(ranked_configs, 1):
        res = candidate_results[cid]
        print(f"  Rank {rank}: {cid} -> Val RMSE = {res['validation']['rmse']:.4f} °C | Val Thermal Acc = {res['validation']['thermal_classification']['accuracy_pct']}% | Nodes = {res['complexity']['decision_nodes']:,}")

    winner_id = ranked_configs[0]
    winner_cfg = candidate_results[winner_id]
    print(f"\n>>> VALIDATION WINNER: {winner_id} (Lowest Validation RMSE: {winner_cfg['validation']['rmse']:.4f} °C)")

    # 6. Evaluate ONLY the Validation Winner ONCE on Frozen TEST Split
    print(f"\n[STEP 5/5] Evaluating Single Validation Winner ({winner_id}) ONCE on held-out TEST partition...")
    winner_models = trained_models[winner_id]
    t_test_eval_0 = time.time()
    preds_test = predict_rf_depthwise(winner_models, test_data)
    t_test_eval = time.time() - t_test_eval_0

    test_comp, _ = compute_comprehensive_metrics(
        y_true=test_data["Y"],
        y_pred=preds_test,
        mask=test_data["mask"],
        lats=test_data["lat"],
        lons=test_data["lon"],
        time_indices=test_data["time_idx"],
        y_clim=y_clim_test
    )

    M_test = test_data["mask"]
    test_thermal = evaluate_regimes(test_data["Y"][M_test], preds_test[M_test])

    winner_test_rmse = test_comp["unweighted_depth_mean"]["rmse"]
    winner_test_mae = test_comp["unweighted_depth_mean"]["mae"]
    winner_test_bias = test_comp["unweighted_depth_mean"]["bias"]
    winner_test_r2 = test_comp["unweighted_depth_mean"]["r2"]

    winner_test_results = {
        "rmse": round(winner_test_rmse, 4),
        "mae": round(winner_test_mae, 4),
        "bias": round(winner_test_bias, 4),
        "r2": round(winner_test_r2, 4),
        "thermal_classification": {
            "accuracy_pct": test_thermal["accuracy_pct"],
            "within_1_bin_pct": test_thermal["within_1_bin_pct"],
            "beyond_1_bin_pct": test_thermal["beyond_1_bin_pct"],
            "cohen_kappa": test_thermal["cohen_kappa"],
            "macro_f1": test_thermal["macro_f1"],
            "weighted_f1": test_thermal["weighted_f1"]
        },
        "depth_breakdown": {
            round(float(d["depth_m"]), 1): round(float(d["rmse"]), 4)
            for d in test_comp["depth_breakdown"]
        },
        "eval_time_seconds": round(t_test_eval, 2)
    }

    winner_cfg["test"] = winner_test_results

    # Explicit delta calculations against certified B3
    cert_test_rmse = certified_b3["test"]["rmse"]  # 1.0452
    cert_test_thermal_acc = certified_b3["test"]["thermal_classification"]["accuracy_pct"]  # 80.65

    rmse_change = round(winner_test_rmse - cert_test_rmse, 4)
    thermal_accuracy_gain = round(test_thermal["accuracy_pct"] - cert_test_thermal_acc, 2)

    comparison_vs_certified = {
        "certified_B3_test_rmse": cert_test_rmse,
        "winner_test_rmse": round(winner_test_rmse, 4),
        "rmse_change": rmse_change,
        "certified_B3_thermal_accuracy_pct": cert_test_thermal_acc,
        "winner_test_thermal_accuracy_pct": test_thermal["accuracy_pct"],
        "thermal_accuracy_gain": thermal_accuracy_gain,
        "decision_nodes_certified": certified_b3["complexity"]["decision_nodes"],
        "decision_nodes_winner": winner_cfg["complexity"]["decision_nodes"],
        "node_count_change": winner_cfg["complexity"]["decision_nodes"] - certified_b3["complexity"]["decision_nodes"],
        "train_time_seconds": winner_cfg["training_time_seconds"],
        "eval_time_seconds": winner_test_results["eval_time_seconds"]
    }

    print("\n" + "=" * 80)
    print("FINAL COMPARISON: VALIDATION WINNER VS CERTIFIED B3")
    print("=" * 80)
    print(f"Validation Winner:            {winner_id}")
    print(f"Certified B3 Test RMSE:       {cert_test_rmse:.4f} °C")
    print(f"Winner Test RMSE:             {winner_test_rmse:.4f} °C (rmse_change = {rmse_change:+.4f} °C)")
    print(f"Certified B3 Thermal Acc:     {cert_test_thermal_acc:.2f}%")
    print(f"Winner Test Thermal Acc:      {test_thermal['accuracy_pct']:.2f}% (thermal_accuracy_gain = {thermal_accuracy_gain:+.2f}%)")
    print(f"Certified Decision Nodes:     {certified_b3['complexity']['decision_nodes']:,}")
    print(f"Winner Decision Nodes:        {winner_cfg['complexity']['decision_nodes']:,}")
    print(f"Winner Train Time:            {winner_cfg['training_time_seconds']:.1f}s")
    print(f"Winner Test Eval Time:        {t_test_eval:.2f}s")

    # Output JSON bundle
    study_output = {
        "study_metadata": {
            "title": "B3 Random Forest Exploratory Hyperparameter Study",
            "date": "October 2026",
            "protocol": "Controlled Validation Selection with Frozen Single Test Evaluation",
            "leakage_controls": "Strict Chronological Train/Val/Test with 6-day Purge Buffers, Train-Only Normalization, 4-Way Masking",
            "status": "EXPLORATORY • NON-CANONICAL (Certified B3 Preserved)"
        },
        "certified_b3": certified_b3,
        "candidates": candidate_results,
        "validation_ranking": ranked_configs,
        "validation_winner": winner_id,
        "comparison_against_certified": comparison_vs_certified
    }

    out_dir = os.path.join(repo_root, "results", "exploratory")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "B3_hyperparameter_study.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(study_output, f, indent=2)
    print(f"\n[OUTPUT SAVED] {json_path}")

    # Output Markdown report table
    md_content = generate_markdown_report(study_output)
    md_path = os.path.join(out_dir, "B3_hyperparameter_study.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OUTPUT SAVED] {md_path}")


def generate_markdown_report(study):
    cb3 = study["certified_b3"]
    candidates = study["candidates"]
    winner_id = study["validation_winner"]
    winner = candidates[winner_id]
    comp = study["comparison_against_certified"]

    md = []
    md.append("# B3 Random Forest Exploratory Hyperparameter Study")
    md.append("## Controlled Validation Optimization and Frozen-Test Verification")
    md.append("\n---\n")
    md.append("### Protocol Disclosure")
    md.append("- **Purpose**: Controlled exploratory hyperparameter tuning for Baseline B3 (Multi-Depth Random Forest).")
    md.append("- **Leakage Prohibition**: Models trained **strictly** on the canonical TRAIN split (Days 0–252, $N=2,871,550$).")
    md.append("- **Selection Rule**: Configuration selection performed **strictly on the VALIDATION split** (Days 259–306, $N=544,800$).")
    md.append("- **Test Partition Integrity**: The frozen held-out TEST partition (Days 313–365, $N=601,550$ columns, $8,017,734$ valid depth targets) was evaluated **exactly once** for the single validation winner.")
    md.append("- **Certified Results Guarantee**: Certified production baseline results (`results/B3.json`, `1.0452 °C`, `80.65%`) remain **FROZEN and UNCHANGED**.")
    md.append("\n---\n")

    md.append("### 1. Evaluated Configurations")
    md.append("| Model ID | Configuration Parameters | Sample Train Size | Trees per Regressor | Total Trees (15 Depths) |")
    md.append("| :--- | :--- | :---: | :---: | :---: |")
    md.append(f"| **Certified B3** | $n=50$, $\\text{{max\\_depth}}=15$, $\\text{{leaf}}=1$, $\\text{{feat}}=1.0$ | 100,000 | 50 | 750 |")
    for cid in ["B3-A", "B3-B", "B3-C"]:
        c = candidates[cid]["parameters"]
        md.append(f"| **{cid}** | $n={c['n_estimators']}$, $\\text{{max\\_depth}}={c['max_depth']}$, $\\text{{leaf}}={c['min_samples_leaf']}$, $\\text{{feat}}={c['max_features']}$ | {c['sample_train_size']:,} | {c['n_estimators']} | {c['n_estimators']*15} |")

    md.append("\n---\n")
    md.append("### 2. Validation Performance Matrix (Hyperparameter Selection)")
    md.append("| Rank | Model ID | Val RMSE (°C) | Val MAE (°C) | Val R² (vs B1) | Val Bias (°C) | Val Thermal Acc (%) | Val ±1-Bin (%) | Val Cohen's $\\kappa$ | Val Macro F1 | Decision Nodes | Train Time (s) | Eval Time (s) |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| Ref | **Certified B3** | {cb3['validation']['rmse']:.4f} | {cb3['validation']['mae']:.4f} | {cb3['validation']['r2']:.4f} | {cb3['validation']['bias']:+.4f} | {cb3['validation']['thermal_classification']['accuracy_pct']:.2f}% | {cb3['validation']['thermal_classification']['within_1_bin_pct']:.2f}% | {cb3['validation']['thermal_classification']['cohen_kappa']:.4f} | {cb3['validation']['thermal_classification']['macro_f1']:.4f} | {cb3['complexity']['decision_nodes']:,} | ~55s | ~8s |")

    for rank, cid in enumerate(study["validation_ranking"], 1):
        res = candidates[cid]
        v = res["validation"]
        t = v["thermal_classification"]
        is_win = " *(Winner)*" if cid == winner_id else ""
        md.append(f"| {rank} | **{cid}**{is_win} | **{v['rmse']:.4f}** | {v['mae']:.4f} | {v['r2']:.4f} | {v['bias']:+.4f} | {t['accuracy_pct']:.2f}% | {t['within_1_bin_pct']:.2f}% | {t['cohen_kappa']:.4f} | {t['macro_f1']:.4f} | {res['complexity']['decision_nodes']:,} | {res['training_time_seconds']:.1f}s | {res['eval_time_seconds']:.1f}s |")

    md.append(f"\n> **Validation Selection Decision**: **{winner_id}** is selected as the primary winner having achieved the lowest unweighted Validation RMSE ({winner['validation']['rmse']:.4f} °C).")

    md.append("\n---\n")
    md.append(f"### 3. Frozen-Test Evaluation of Validation Winner ({winner_id})")
    md.append("| Model ID | Test Split Role | Test RMSE (°C) | Test MAE (°C) | Test R² (vs B1) | Test Bias (°C) | Test Thermal Acc (%) | Test ±1-Bin (%) | Test Cohen's $\\kappa$ | Test Macro F1 |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **Certified B3** | Locked Baseline | {cb3['test']['rmse']:.4f} | {cb3['test']['mae']:.4f} | {cb3['test']['r2']:.4f} | {cb3['test']['bias']:+.4f} | {cb3['test']['thermal_classification']['accuracy_pct']:.2f}% | {cb3['test']['thermal_classification']['within_1_bin_pct']:.2f}% | {cb3['test']['thermal_classification']['cohen_kappa']:.4f} | {cb3['test']['thermal_classification']['macro_f1']:.4f} |")
    wt = winner["test"]
    wth = wt["thermal_classification"]
    md.append(f"| **{winner_id}** | Exploratory Candidate | **{wt['rmse']:.4f}** | {wt['mae']:.4f} | {wt['r2']:.4f} | {wt['bias']:+.4f} | **{wth['accuracy_pct']:.2f}%** | {wth['within_1_bin_pct']:.2f}% | {wth['cohen_kappa']:.4f} | {wth['macro_f1']:.4f} |")

    md.append("\n---\n")
    md.append("### 4. Direct Quantitative Comparison Against Certified B3")
    md.append("| Metric | Certified B3 | Validation Winner (" + winner_id + ") | Delta / Change | Formula |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    md.append(f"| **Test Column-Avg RMSE** | {comp['certified_B3_test_rmse']:.4f} °C | {comp['winner_test_rmse']:.4f} °C | **{comp['rmse_change']:+.4f} °C** | `winner_test_rmse - 1.0452` |")
    md.append(f"| **Test Thermal Regime Accuracy** | {comp['certified_B3_thermal_accuracy_pct']:.2f}% | {comp['winner_test_thermal_accuracy_pct']:.2f}% | **{comp['thermal_accuracy_gain']:+.2f}%** | `winner_test_thermal_accuracy - 80.65` |")
    md.append(f"| **Decision Tree Nodes** | {comp['decision_nodes_certified']:,} | {comp['decision_nodes_winner']:,} | **{comp['node_count_change']:+,} nodes** | Complexity increment |")

    md.append("\n---\n")
    md.append("### 5. Depth-Wise Test Error Distribution (RMSE in °C)")
    md.append("| Depth (m) | Certified B3 | " + winner_id + " | Delta (°C) | Relative Gain (%) |")
    md.append("| :---: | :---: | :---: | :---: | :---: |")
    for d in CANONICAL_DEPTHS:
        d_key = round(float(d), 1)
        c_rmse = cb3["test"]["depth_breakdown"][d_key]
        w_rmse = wt["depth_breakdown"][d_key]
        delta_d = w_rmse - c_rmse
        rel_gain = (c_rmse - w_rmse) / c_rmse * 100.0
        md.append(f"| **{int(d)} m** | {c_rmse:.4f} | {w_rmse:.4f} | {delta_d:+.4f} | {rel_gain:+.2f}% |")

    md.append("\n---\n")
    md.append("### 6. Summary & Recommendation")
    md.append(f"- **Validation Ranking**: The 3 exploratory configurations were ranked on validation RMSE as follows: `{' > '.join(study['validation_ranking'])}`.")
    md.append(f"- **Test Performance**: On the held-out test split, **{winner_id}** achieved a test RMSE of `{comp['winner_test_rmse']:.4f} °C` (change: `{comp['rmse_change']:+.4f} °C`) and a thermal-regime accuracy of `{comp['winner_test_thermal_accuracy_pct']:.2f}%` (gain: `{comp['thermal_accuracy_gain']:+.2f}%`).")
    md.append("- **Integrity Status**: Under the user-directed release freeze protocol, **Certified B3 remains locked at 1.0452 °C and 80.65%**. These exploratory results are recorded strictly for research documentation without modifying any production benchmark artifacts.")

    return "\n".join(md)


if __name__ == "__main__":
    main()
