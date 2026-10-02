"""
models/03_random_forest.py
Model 3: Multi-Output Random Forest Regressor

Protocol:
1. Trains Random Forest regressor on Training Split (Days 0-255) using representative subsampling (100k samples).
2. Tunes max_depth and n_estimators strictly on full Validation Split (Days 256-309, N=1,003,374).
3. Selects and freezes optimal hyperparameters based on validation performance.
4. Evaluates frozen model on full Final Test Split (Days 310-365, N=1,040,536).
Zero Test Leakage: Test set is strictly isolated until final freeze.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.tabular_dataset import load_tabular_dataset
from models.base import compute_masked_metrics, print_evaluation_summary, save_model_evaluation

def train_and_tune_random_forest(dataset, sample_train_size=100000):
    print("=" * 70)
    print("PHASE 1: MODEL 3 - RANDOM FOREST REGRESSOR")
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
    
    N_train = len(X_train)
    # Representative random sample from training set for computational efficiency
    np.random.seed(42)
    sample_indices = np.random.choice(N_train, size=min(sample_train_size, N_train), replace=False)
    
    X_train_sub = X_train[sample_indices]
    Y_train_sub = Y_train[sample_indices].copy()
    M_train_sub = M_train[sample_indices]
    
    # Fill target NaNs (seabed) with 0.0 for training scikit-learn multi-output regressor
    # (Evaluation strictly evaluates only valid masked depths via M_val and M_test)
    Y_train_sub = np.nan_to_num(Y_train_sub, nan=0.0)
    
    print(f"Training on {len(X_train_sub):,} representative samples from the 4.75M training split.")
    
    # Candidate hyperparameter grid for validation tuning
    candidates = [
        {"n_estimators": 30, "max_depth": 10},
        {"n_estimators": 30, "max_depth": 15},
        {"n_estimators": 50, "max_depth": 15}
    ]
    
    best_val_rmse = float("inf")
    best_params = None
    best_model = None
    
    print("\nTuning Hyperparameters on Full Validation Split (N=1,003,374):")
    for params in candidates:
        n_est = params["n_estimators"]
        m_depth = params["max_depth"]
        
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=m_depth,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train_sub, Y_train_sub)
        
        val_pred = rf.predict(X_val)
        val_summary, _ = compute_masked_metrics(
            y_true=Y_val,
            y_pred=val_pred,
            mask=M_val,
            model_name=f"RF (n_est={n_est}, depth={m_depth})",
            split_name="val"
        )
        
        print(f"  n_est={n_est:2d}, depth={m_depth:2d} -> Val Masked RMSE: {val_summary['overall_rmse']:.4f} °C | R^2: {val_summary['overall_r2']:.4f}")
        
        if val_summary["overall_rmse"] < best_val_rmse:
            best_val_rmse = val_summary["overall_rmse"]
            best_params = params
            best_model = rf

    print(f"\n[SELECTION] Best hyperparameters selected on Validation: {best_params} (Val RMSE: {best_val_rmse:.4f} °C)")
    print("Freezing Random Forest model...")

    # 1. Validation evaluation with frozen best model
    val_pred = best_model.predict(X_val)
    val_summary, df_val_depths = compute_masked_metrics(
        y_true=Y_val,
        y_pred=val_pred,
        mask=M_val,
        model_name="Random Forest",
        split_name="val"
    )
    val_summary["best_params"] = best_params
    print_evaluation_summary(val_summary, df_val_depths)
    save_model_evaluation(val_summary, df_val_depths)

    # 2. Final Test evaluation (Strictly frozen model, no tuning)
    print("\n--- Evaluating Frozen Random Forest on Final Test Split (Days 310-365, N=1,040,536) ---")
    test_pred = best_model.predict(X_test)
    test_summary, df_test_depths = compute_masked_metrics(
        y_true=Y_test,
        y_pred=test_pred,
        mask=M_test,
        model_name="Random Forest",
        split_name="test"
    )
    test_summary["best_params"] = best_params
    print_evaluation_summary(test_summary, df_test_depths)
    save_model_evaluation(test_summary, df_test_depths)

    return val_summary, test_summary

if __name__ == "__main__":
    dataset = load_tabular_dataset()
    train_and_tune_random_forest(dataset)
