"""
models/04_gradient_boosting.py
Model 4: Pointwise Gradient Boosted Decision Trees (LightGBM with XGBoost fallback)

Protocol:
1. Multi-output subsurface temperature reconstruction across 15 canonical depths using 15 depth-wise GBDT regressors.
2. For each depth, trains strictly on valid training samples where bathymetry allows that depth.
3. Tunes hyperparameters strictly on the full Validation Split (Days 256-309, N=1,003,374).
4. Selects optimal configuration and freezes the model.
5. Evaluates frozen model on the final Test Split (Days 310-365, N=1,040,536).
Zero Test Leakage: Test set is never touched for hyperparameter tuning or selection.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.tabular_dataset import load_tabular_dataset
from models.base import compute_masked_metrics, print_evaluation_summary, save_model_evaluation
from preprocessing.canonical_grid import CANONICAL_DEPTHS

# Try importing LightGBM; fallback to XGBoost if needed
USE_XGBOOST = False
try:
    import lightgbm as lgb
    print(f"Using LightGBM {lgb.__version__}")
except ImportError:
    import xgboost as xgb
    USE_XGBOOST = True
    print(f"LightGBM not found. Falling back to XGBoost {xgb.__version__}")

class MultiDepthGBDT:
    """
    Ensemble of 15 depth-wise GBDT regressors, one per canonical ocean depth.
    Each regressor trains only on points where that depth is bathymetrically valid.
    """
    def __init__(self, use_xgboost=USE_XGBOOST, **model_params):
        self.use_xgboost = use_xgboost
        self.model_params = model_params
        self.depth_models = {}
        self.depth_levels = CANONICAL_DEPTHS

    def fit(self, X_train, Y_train, M_train):
        for d_idx, depth in enumerate(self.depth_levels):
            valid_mask = M_train[:, d_idx]
            if not np.any(valid_mask):
                continue
            
            X_d = X_train[valid_mask]
            y_d = Y_train[valid_mask, d_idx]
            
            if not self.use_xgboost:
                model = lgb.LGBMRegressor(
                    verbosity=-1,
                    n_jobs=-1,
                    random_state=42,
                    **self.model_params
                )
            else:
                model = xgb.XGBRegressor(
                    verbosity=0,
                    n_jobs=-1,
                    random_state=42,
                    **self.model_params
                )
            model.fit(X_d, y_d)
            self.depth_models[d_idx] = model

    def predict(self, X):
        N = len(X)
        Y_pred = np.zeros((N, len(self.depth_levels)), dtype=np.float32)
        for d_idx in range(len(self.depth_levels)):
            if d_idx in self.depth_models:
                Y_pred[:, d_idx] = self.depth_models[d_idx].predict(X)
        return Y_pred

def train_and_tune_gbdt(dataset, sample_train_size=100000):
    print("=" * 70)
    backend_name = "XGBOOST" if USE_XGBOOST else "LIGHTGBM"
    print(f"PHASE 1: MODEL 4 - GRADIENT BOOSTING ({backend_name})")
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
    np.random.seed(42)
    sample_indices = np.random.choice(N_train, size=min(sample_train_size, N_train), replace=False)
    
    X_train_sub = X_train[sample_indices]
    Y_train_sub = Y_train[sample_indices]
    M_train_sub = M_train[sample_indices]
    
    print(f"Training on {len(X_train_sub):,} representative samples across 15 depth-wise regressors.")
    
    # Candidate hyperparameter grid for validation tuning
    if not USE_XGBOOST:
        candidates = [
            {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 50},
            {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 100},
            {"num_leaves": 63, "learning_rate": 0.1, "n_estimators": 50}
        ]
    else:
        candidates = [
            {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 50},
            {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100},
            {"max_depth": 7, "learning_rate": 0.1, "n_estimators": 50}
        ]
        
    best_val_rmse = float("inf")
    best_params = None
    best_model = None
    
    print("\nTuning Hyperparameters on Full Validation Split (N=1,003,374):")
    for params in candidates:
        t0 = time.time()
        gbdt = MultiDepthGBDT(use_xgboost=USE_XGBOOST, **params)
        gbdt.fit(X_train_sub, Y_train_sub, M_train_sub)
        
        val_pred = gbdt.predict(X_val)
        val_summary, _ = compute_masked_metrics(
            y_true=Y_val,
            y_pred=val_pred,
            mask=M_val,
            model_name=f"{backend_name} ({params})",
            split_name="val"
        )
        elapsed = time.time() - t0
        print(f"  Params: {params} -> Val Masked RMSE: {val_summary['overall_rmse']:.4f} °C | R^2: {val_summary['overall_r2']:.4f} ({elapsed:.1f}s)")
        
        if val_summary["overall_rmse"] < best_val_rmse:
            best_val_rmse = val_summary["overall_rmse"]
            best_params = params
            best_model = gbdt

    print(f"\n[SELECTION] Best hyperparameters selected on Validation: {best_params} (Val RMSE: {best_val_rmse:.4f} °C)")
    print(f"Freezing {backend_name} model...")
    
    # Validation Evaluation of Best Model
    val_pred_best = best_model.predict(X_val)
    val_summary, df_val_depths = compute_masked_metrics(
        y_true=Y_val,
        y_pred=val_pred_best,
        mask=M_val,
        model_name="LightGBM" if not USE_XGBOOST else "XGBoost",
        split_name="val"
    )
    val_summary["best_params"] = best_params
    print_evaluation_summary(val_summary, df_val_depths)
    save_model_evaluation(val_summary, df_val_depths)
    
    # Test Evaluation of Frozen Model (Zero Leakage)
    print(f"\n--- Evaluating Frozen {backend_name} on Final Test Split (Days 310-365, N={len(X_test):,}) ---")
    test_pred = best_model.predict(X_test)
    test_summary, df_test_depths = compute_masked_metrics(
        y_true=Y_test,
        y_pred=test_pred,
        mask=M_test,
        model_name="LightGBM" if not USE_XGBOOST else "XGBoost",
        split_name="test"
    )
    test_summary["best_params"] = best_params
    print_evaluation_summary(test_summary, df_test_depths)
    save_model_evaluation(test_summary, df_test_depths)
    
    return val_summary, test_summary

if __name__ == "__main__":
    dataset = load_tabular_dataset()
    train_and_tune_gbdt(dataset)
