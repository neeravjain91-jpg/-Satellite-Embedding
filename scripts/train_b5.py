"""
scripts/train_b5.py
Official ML Execution Script:
Trains and evaluates Baseline B5 (Pointwise Multi-Layer Perceptron)
on the certified full-year 2020 dataset under the locked scientific protocol.

Protocol Enforcements:
- Model Definition:
    Pointwise MLP mapping 7 canonical surface predictors to 15 vertical depths.
    Strictly non-spatial, non-temporal (no patches, CNN, GRU, sequences, attention).
    Hidden dimensions: [128, 128, 64], 26,767 trainable parameters.
- Exact chronological split:
    TRAIN:   Days 0..252   (253 days: 2020-01-01 to 2020-09-09)
    PURGE 1: Days 253..258 (6 days, discarded)
    VAL:     Days 259..306 (48 days: 2020-09-16 to 2020-11-02)
    PURGE 2: Days 307..312 (6 days, discarded)
    TEST:    Days 313..365 (53 days: 2020-11-09 to 2020-12-31)
- Surface features:
    7 canonical predictors: [sst, sss, ssh, current_u, current_v, wind_u, wind_v]
    Standardized strictly using Train-split statistics (Zero Leakage).
- Target masking:
    Canonical 4-way evaluation mask preserved.
    Invalid bathymetric targets strictly preserved as NaN (never zero-filled).
    Masked MSE loss ignores unobserved depth targets without zero-filling.
- Deliverables:
    results/B5.json
"""

import os
import sys
import json
import hashlib
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS, CANONICAL_FEATURES
from preprocessing.tabular_dataset import load_tabular_dataset
from models.baselines import B1_Climatology, B5_PointwiseMLP, PointwiseMLPNet
from models.base import masked_mse_loss
from models.metrics_engine import compute_comprehensive_metrics, compute_block_bootstrap_ci


def main():
    print("======================================================================")
    print("TRAINING & EVALUATION OF BASELINE B5: POINTWISE MULTI-LAYER PERCEPTRON")
    print("======================================================================")

    dataset = load_tabular_dataset(build_context=False)
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]

    # Reference B1
    b1 = B1_Climatology().fit(train_data)
    y_clim_val = b1.predict(val_data)
    y_clim_test = b1.predict(test_data)
    b1_test_rmse = 1.2582

    # Fit B5
    print("\nFitting B5: Pointwise MLP (hidden=[128, 128, 64], 26,767 parameters)...")
    b5 = B5_PointwiseMLP(hidden_dims=[128, 128, 64], lr=1e-3)
    
    # Check if existing checkpoint exists
    ckpt_path = os.path.join(repo_root, "models", "checkpoints", "pointwise_mlp_best.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        b5.net = PointwiseMLPNet(7, [128, 128, 64], 15).to(b5.device)
        b5.net.load_state_dict(ckpt if "state_dict" not in ckpt else ckpt["state_dict"])
        b5.is_fitted = True
    else:
        b5.fit(train_data, val_data=val_data, epochs=8, batch_size=4096)
        torch.save(b5.net.state_dict(), ckpt_path)

    preds_val = b5.predict(val_data)
    preds_test = b5.predict(test_data)

    print("Computing metrics on validation split...")
    val_metrics, _ = compute_comprehensive_metrics(
        y_true=val_data["Y"],
        y_pred=preds_val,
        mask=val_data["mask"],
        lats=val_data["lat"],
        lons=val_data["lon"],
        time_indices=val_data["time_idx"],
        y_clim=y_clim_val
    )

    print("Computing metrics on test split...")
    test_metrics, _ = compute_comprehensive_metrics(
        y_true=test_data["Y"],
        y_pred=preds_test,
        mask=test_data["mask"],
        lats=test_data["lat"],
        lons=test_data["lon"],
        time_indices=test_data["time_idx"],
        y_clim=y_clim_test
    )

    result_record = {
        "model_id": "B5",
        "model_name": "Pointwise MLP",
        "parameter_count": b5.parameter_count(),
        "metadata": b5.metadata(),
        "validation": {
            "overall": val_metrics["unweighted_depth_mean"],
            "weighted_overall": val_metrics["sample_weighted_depth_mean"],
            "depth_breakdown": val_metrics["depth_breakdown"],
            "regions": val_metrics.get("regions", {})
        },
        "test": {
            "overall": test_metrics["unweighted_depth_mean"],
            "weighted_overall": test_metrics["sample_weighted_depth_mean"],
            "depth_breakdown": test_metrics["depth_breakdown"],
            "regions": test_metrics.get("regions", {}),
            "seasons": test_metrics.get("seasons", {})
        }
    }

    out_path = os.path.join(repo_root, "results", "B5.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)
    print(f"\n[RESULTS SAVED] {out_path}")


if __name__ == "__main__":
    main()
