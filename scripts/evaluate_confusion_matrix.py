"""
scripts/evaluate_confusion_matrix.py
Evaluates the physical thermal regime diagnostic confusion matrix on the official Test partition:
- 8,017,734 valid test target points across 15 canonical depths and 53 test days.
- Evaluates:
    1. Canonical B1: Spatial-Depth Climatology (Historical Train Mean)
    2. Canonical B2: Multi-Output Ridge Regression (alpha=100,000)
    3. Canonical B3: Multi-Depth Random Forest (50 trees, max depth 15, 13,289,966 nodes)
    4. Canonical B5: Pointwise MLP (PyTorch, [128, 128, 64], 26,767 parameters)
    5. Legacy Exploratory MLP: [128, 64], 10,255 parameters (historical tuning candidate,
       provenance source of the 81.19% accuracy artifact)
- Discretizes continuous ocean potential temperature into 6 temperature-based thermal regimes:
    Regime 1: Deep Water (<10°C, 500–1000m)
    Regime 2: Lower Thermocline (10–15°C, 200–300m)
    Regime 3: Core Thermocline (15–20°C, 100–150m)
    Regime 4: Upper Thermocline (20–25°C, 50–100m)
    Regime 5: Subsurface Mixed Layer (25–28°C, 10–50m)
    Regime 6: Tropical Warm Pool (≥28°C, 0–10m)
- Computes exact Accuracy, Within ±1 Bin %, Beyond ±1 Bin %, Cohen's Kappa, Macro/Weighted F1,
  and per-class Precision/Recall/F1.
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, cohen_kappa_score

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.tabular_dataset import load_tabular_dataset
from models.baselines import PointwiseMLPNet, B1_Climatology, B3_RandomForest
from preprocessing.canonical_grid import CANONICAL_DEPTHS

REGIME_BINS = [-np.inf, 10.0, 15.0, 20.0, 25.0, 28.0, np.inf]
REGIME_LABELS = [
    "Deep Water (<10°C)",
    "Lower Thermocline (10–15°C)",
    "Core Thermocline (15–20°C)",
    "Upper Thermocline (20–25°C)",
    "Mixed Layer (25–28°C)",
    "Tropical Warm Pool (≥28°C)"
]


def evaluate_regimes(y_true, y_pred, labels=REGIME_LABELS, bins=REGIME_BINS):
    y_true_cls = np.digitize(y_true, bins) - 1
    y_pred_cls = np.digitize(y_pred, bins) - 1

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=range(len(labels)))
    total = int(np.sum(cm))
    accuracy = float(np.trace(cm) / total * 100)
    abs_diff = np.abs(y_true_cls - y_pred_cls)
    within1 = float(np.mean(abs_diff <= 1) * 100)
    beyond1 = float(np.mean(abs_diff > 1) * 100)
    kappa = float(cohen_kappa_score(y_true_cls, y_pred_cls))

    per_class = []
    col_totals = np.sum(cm, axis=0)
    row_totals = np.sum(cm, axis=1)

    for i in range(len(labels)):
        tp = int(cm[i, i])
        fp = int(col_totals[i] - tp)
        fn = int(row_totals[i] - tp)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append({
            "regime": labels[i],
            "support": int(row_totals[i]),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        })

    macro_f1 = float(np.mean([c["f1"] for c in per_class]))
    weighted_f1 = float(np.sum([c["f1"] * c["support"] for c in per_class]) / total)

    return {
        "confusion_matrix": cm.tolist(),
        "total_evaluated_samples": total,
        "accuracy_pct": round(accuracy, 2),
        "within_1_bin_pct": round(within1, 2),
        "beyond_1_bin_pct": round(beyond1, 2),
        "cohen_kappa": round(kappa, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class
    }


def main():
    print("=" * 70)
    print("PHYSICAL THERMAL REGIME CONFUSION MATRIX EVALUATION")
    print("=" * 70)
    print("Loading test split for confusion matrix evaluation...")
    dataset = load_tabular_dataset(build_context=False)
    train_data = dataset["train"]
    test_data = dataset["test"]
    X_test = test_data["X_norm"]
    Y_test = test_data["Y"]
    M_test = test_data["mask"]
    y_true_valid = Y_test[M_test]
    n_total = len(y_true_valid)
    print(f"Total valid test targets: {n_total:,}")
    assert n_total == 8017734, f"Expected 8,017,734 test points, got {n_total}"

    # 1. B1 Spatial Climatology
    print("\n1. Evaluating Canonical B1 Spatial-Depth Climatology...")
    b1 = B1_Climatology().fit(train_data)
    y_pred_b1 = b1.predict(test_data)
    res_b1 = evaluate_regimes(y_true_valid, y_pred_b1[M_test])
    print(f"   B1 Accuracy: {res_b1['accuracy_pct']}% | Within ±1: {res_b1['within_1_bin_pct']}% | Kappa: {res_b1['cohen_kappa']}")

    # 2. B2 Multi-Output Ridge
    print("\n2. Evaluating Canonical B2 Multi-Output Ridge Regression (alpha=100,000)...")
    n_depths = len(CANONICAL_DEPTHS)
    b2_models = {}
    for d in range(n_depths):
        d_valid = train_data["mask"][:, d]
        if np.any(d_valid):
            reg = Ridge(alpha=100000.0, random_state=42)
            reg.fit(train_data["X_norm"][d_valid], train_data["Y"][d_valid, d])
            b2_models[d] = reg
    y_pred_b2 = np.zeros_like(Y_test)
    for d in range(n_depths):
        if b2_models.get(d):
            y_pred_b2[:, d] = b2_models[d].predict(X_test)
    res_b2 = evaluate_regimes(y_true_valid, y_pred_b2[M_test])
    print(f"   B2 Accuracy: {res_b2['accuracy_pct']}% | Within ±1: {res_b2['within_1_bin_pct']}% | Kappa: {res_b2['cohen_kappa']}")

    # 3. Canonical B3 Random Forest
    print("\n3. Evaluating Canonical B3 Multi-Depth Random Forest...")
    N_train = len(train_data["X_norm"])
    np.random.seed(42)
    sub_100k = np.random.choice(N_train, size=min(100000, N_train), replace=False)
    train_100k = {k: v[sub_100k] if isinstance(v, np.ndarray) and len(v) == N_train else v for k, v in train_data.items()}
    b3 = B3_RandomForest(n_estimators=50, max_depth=15, sample_train_size=100000)
    b3.fit(train_100k)
    y_pred_b3 = b3.predict(test_data)
    res_b3 = evaluate_regimes(y_true_valid, y_pred_b3[M_test])
    print(f"   B3 Accuracy: {res_b3['accuracy_pct']}% | Within ±1: {res_b3['within_1_bin_pct']}% | Kappa: {res_b3['cohen_kappa']}")

    # 4. Canonical B5 Pointwise MLP (26,767 parameters, [128, 128, 64])
    print("\n4. Evaluating Canonical B5 Pointwise MLP ([128, 128, 64], 26,767 params)...")
    ckpt_b5_path = os.path.join(repo_root, "models", "checkpoints", "pointwise_mlp_best.pt")
    if os.path.exists(ckpt_b5_path):
        ckpt_b5 = torch.load(ckpt_b5_path, map_location="cpu", weights_only=False)
        net_b5 = PointwiseMLPNet(7, [128, 128, 64], 15)
        state_dict = ckpt_b5 if "state_dict" not in ckpt_b5 else ckpt_b5["state_dict"]
        # Handle prefix if present
        cleaned_state = {k.replace("net.", ""): v for k, v in state_dict.items()} if any(k.startswith("net.") for k in state_dict.keys()) else state_dict
        # Or load with class directly matching state dict keys
        try:
            net_b5.load_state_dict(state_dict)
        except Exception:
            # PointwiseMLPNet uses nn.Sequential
            # If checkpoint has net.0.weight, PointwiseMLP in 05_pointwise_mlp.py has self.net
            import importlib.util
            spec = importlib.util.spec_from_file_location("m05", os.path.join(repo_root, "models", "05_pointwise_mlp.py"))
            m05 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m05)
            net_b5 = m05.PointwiseMLP(7, 15, [128, 128, 64])
            net_b5.load_state_dict(state_dict)
        net_b5.eval()
        with torch.no_grad():
            y_pred_b5 = net_b5(torch.tensor(X_test, dtype=torch.float32)).numpy()
        res_b5 = evaluate_regimes(y_true_valid, y_pred_b5[M_test])
        print(f"   B5 Accuracy: {res_b5['accuracy_pct']}% | Within ±1: {res_b5['within_1_bin_pct']}% | Kappa: {res_b5['cohen_kappa']}")
    else:
        res_b5 = None
        print("   [WARNING] Canonical B5 checkpoint pointwise_mlp_best.pt not found!")

    # 5. Legacy Exploratory MLP (10,255 parameters, [128, 64])
    print("\n5. Evaluating Historical Exploratory MLP ([128, 64], 10,255 params)...")
    ckpt_legacy_path = os.path.join(repo_root, "models", "checkpoints", "b3_MLP_128_64_best.pt")
    if os.path.exists(ckpt_legacy_path):
        ckpt_leg = torch.load(ckpt_legacy_path, map_location="cpu", weights_only=False)
        net_leg = PointwiseMLPNet(7, [128, 64], 15)
        net_leg.load_state_dict(ckpt_leg["state_dict"])
        net_leg.eval()
        with torch.no_grad():
            y_pred_leg = net_leg(torch.tensor(X_test, dtype=torch.float32)).numpy()
        res_legacy = evaluate_regimes(y_true_valid, y_pred_leg[M_test])
        print(f"   Legacy MLP Accuracy: {res_legacy['accuracy_pct']}% | Within ±1: {res_legacy['within_1_bin_pct']}% | Kappa: {res_legacy['cohen_kappa']}")
    else:
        res_legacy = None
        print("   [WARNING] Legacy checkpoint b3_MLP_128_64_best.pt not found!")

    results = {
        "evaluation_partition": "Test (Days 313–365, Nov 9 – Dec 31, 2020)",
        "total_test_samples": n_total,
        "diagnostic_definition": "Six temperature-based thermal regimes defined for diagnostic classification of the continuous temperature field.",
        "thermal_regimes": REGIME_LABELS,
        "temperature_boundaries_c": [10.0, 15.0, 20.0, 25.0, 28.0],
        "oceanographic_reference": "GLORYS numerical ocean reanalysis reference (compared against in-situ ARGO float profiles via ARGO–GLORYS Reference Consistency Assessment; note that this comparison assesses the reanalysis reference state and does not constitute independent validation of the ML model)",
        "provenance_notes": {
            "B1_Climatology": "Canonical Spatial-Depth historical train mean climatology (0 parameters)",
            "B2_Ridge": "Canonical Multi-Output Ridge Regression with alpha=100,000 (120 coefficients)",
            "B3_RandomForest": "Canonical Multi-Depth Random Forest (50 trees, max_depth 15, 13,289,966 nodes). B3 classification metrics are reproduced by refitting the canonical B3 training protocol with the locked seed and 100,000-row training subsample, then evaluating on the frozen held-out test partition.",
            "B5_PointwiseMLP": "Canonical Pointwise MLP (26,767 trainable parameters, [128, 128, 64])",
            "Legacy_MLP_128_64": "Historical exploratory tuning candidate (10,255 trainable parameters, [128, 64]). Provenance origin of the 81.19% classification artifact."
        },
        "models": {
            "B1_Climatology": res_b1,
            "B2_Ridge": res_b2,
            "B3_RandomForest": res_b3,
            "B5_PointwiseMLP": res_b5,
            "Legacy_MLP_128_64": res_legacy
        }
    }

    out_path = os.path.join(repo_root, "results", "confusion_matrix.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[RESULTS SAVED] {out_path}")


if __name__ == "__main__":
    main()
