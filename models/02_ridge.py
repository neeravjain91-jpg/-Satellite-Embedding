"""
models/02_ridge.py
Model 2: Multi-Output Ridge Regression

Protocol:
1. Trains multi-output Ridge regression on Training Split (Days 0-255).
2. Tunes alpha regularizer across candidate values on Validation Split (Days 256-309).
3. Freezes optimal alpha* based strictly on validation performance.
4. Evaluates frozen model on Final Test Split (Days 310-365).
Zero Test Leakage: Test set is never touched during hyperparameter tuning.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.tabular_dataset import load_tabular_dataset
from models.base import compute_masked_metrics, print_evaluation_summary, save_model_evaluation

def train_and_tune_ridge(dataset, alpha_candidates=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]):
    print("=" * 70)
    print("PHASE 1: MODEL 2 - MULTI-OUTPUT RIDGE REGRESSION")
    print("=" * 70)
    
    X_train = dataset["train"]["X_norm"]
    Y_train = dataset["train"]["Y"]
    M_train = dataset["train"]["mask"]
    
    X_val = dataset["val"]["X_norm"]
    Y_val = dataset["val"]["Y"]
    M_val = dataset["val"]["mask"]
    
    X_test = dataset["test"]["X_norm"]
    Y_test = dataset["test"]["Y"]
    M_test = dataset["test"]["mask"]
    
    # In multi-output regression, target values deeper than bathymetry are NaN in raw Y.
    # For fitting scikit-learn Ridge, fill NaNs in Y_train with 0.0 (masked out during loss/metric evaluation)
    # or fit separate per-depth Ridge models where only valid ocean points for that depth are used!
    # Fitting 15 individual depth models is mathematically clean and optimal because it trains strictly on valid ocean depth points!
    print("Training 15 depth-wise Ridge models (one per target depth level)...")
    
    best_overall_rmse = float("inf")
    best_alpha = None
    best_models_dict = None
    
    print("\nTuning Alpha on Validation Split:")
    for alpha in alpha_candidates:
        models_at_alpha = {}
        # Fit 15 depth models
        for d in range(15):
            d_valid = M_train[:, d]
            ridge_d = Ridge(alpha=alpha, random_state=42)
            ridge_d.fit(X_train[d_valid], Y_train[d_valid, d])
            models_at_alpha[d] = ridge_d
            
        # Predict on validation split
        val_pred = np.zeros_like(Y_val)
        for d in range(15):
            val_pred[:, d] = models_at_alpha[d].predict(X_val)
            
        val_summary, _ = compute_masked_metrics(
            y_true=Y_val,
            y_pred=val_pred,
            mask=M_val,
            model_name=f"Ridge (alpha={alpha})",
            split_name="val"
        )
        
        print(f"  alpha={alpha:8.2f} -> Val Masked RMSE: {val_summary['overall_rmse']:.4f} °C | R^2: {val_summary['overall_r2']:.4f}")
        
        if val_summary["overall_rmse"] < best_overall_rmse:
            best_overall_rmse = val_summary["overall_rmse"]
            best_alpha = alpha
            best_models_dict = models_at_alpha

    print(f"\n[SELECTION] Optimal alpha selected from validation: alpha* = {best_alpha} (Val RMSE: {best_overall_rmse:.4f} °C)")
    print("Freezing Ridge model with alpha*...")

    # 1. Validation evaluation with frozen best model
    val_pred = np.zeros_like(Y_val)
    for d in range(15):
        val_pred[:, d] = best_models_dict[d].predict(X_val)
        
    val_summary, df_val_depths = compute_masked_metrics(
        y_true=Y_val,
        y_pred=val_pred,
        mask=M_val,
        model_name="Ridge",
        split_name="val"
    )
    val_summary["best_alpha"] = best_alpha
    print_evaluation_summary(val_summary, df_val_depths)
    save_model_evaluation(val_summary, df_val_depths)

    # 2. Final Test evaluation (Strictly frozen model, no tuning)
    print("\n--- Evaluating Frozen Ridge on Final Test Split (Days 310-365) ---")
    test_pred = np.zeros_like(Y_test)
    for d in range(15):
        test_pred[:, d] = best_models_dict[d].predict(X_test)
        
    test_summary, df_test_depths = compute_masked_metrics(
        y_true=Y_test,
        y_pred=test_pred,
        mask=M_test,
        model_name="Ridge",
        split_name="test"
    )
    test_summary["best_alpha"] = best_alpha
    print_evaluation_summary(test_summary, df_test_depths)
    save_model_evaluation(test_summary, df_test_depths)

    return val_summary, test_summary

if __name__ == "__main__":
    dataset = load_tabular_dataset()
    train_and_tune_ridge(dataset)
