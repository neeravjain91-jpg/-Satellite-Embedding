"""
models/03_random_forest.py
Model 3: Multi-Depth Random Forest Regressor

Protocol:
1. Trains 15 depth-wise Random Forest regressors on Training Split (Days 0–252) using representative subsampling.
   Each depth regressor trains strictly on valid ocean depth points (M_train[:, d] == True).
   No fake 0 °C targets: NaN values below seabed are never converted to 0 °C.
2. Tunes max_depth and n_estimators strictly on full Validation Split (Days 259–306).
3. Selects and freezes optimal hyperparameters based on validation performance.
4. Evaluates frozen model on full Final Test Split (Days 313–365).
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
from preprocessing.canonical_grid import CANONICAL_DEPTHS

class MultiDepthRandomForest:
    """
    Ensemble of 15 depth-wise Random Forest regressors, one per canonical ocean depth.
    Each regressor trains strictly on points where that depth is bathymetrically valid,
    eliminating any need to represent seabed/invalid depths as artificial 0 °C targets.
    """
    def __init__(self, n_estimators=30, max_depth=10, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.depth_models = {}
        self.depth_levels = CANONICAL_DEPTHS

    def fit(self, X_train, Y_train, M_train):
        for d_idx, depth in enumerate(self.depth_levels):
            valid_mask = M_train[:, d_idx]
            if not np.any(valid_mask):
                self.depth_models[d_idx] = None
                continue
            
            X_d = X_train[valid_mask]
            y_d = Y_train[valid_mask, d_idx]
            
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            rf.fit(X_d, y_d)
            self.depth_models[d_idx] = rf
        return self

    def predict(self, X):
        N = len(X)
        preds = np.zeros((N, len(self.depth_levels)), dtype=np.float32)
        for d_idx in range(len(self.depth_levels)):
            if self.depth_models.get(d_idx) is not None:
                preds[:, d_idx] = self.depth_models[d_idx].predict(X)
            else:
                preds[:, d_idx] = 15.0
        return preds

def train_and_tune_random_forest(dataset, sample_train_size=100000):
    print("=" * 70)
    print("PHASE 1: MODEL 3 - RANDOM FOREST REGRESSOR (DEPTH-WISE)")
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
    Y_train_sub = Y_train[sample_indices]
    M_train_sub = M_train[sample_indices]
    
    print(f"Training on {len(X_train_sub):,} representative samples across 15 depth-wise regressors.")
    
    # Candidate hyperparameter grid for validation tuning
    candidates = [
        {"n_estimators": 30, "max_depth": 10},
        {"n_estimators": 30, "max_depth": 15},
        {"n_estimators": 50, "max_depth": 15}
    ]
    
    best_val_rmse = float("inf")
    best_params = None
    best_model = None
    
    print("\nTuning Hyperparameters on Full Validation Split (Days 259–306):")
    for params in candidates:
        n_est = params["n_estimators"]
        m_depth = params["max_depth"]
        
        rf = MultiDepthRandomForest(
            n_estimators=n_est,
            max_depth=m_depth,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_sub, Y_train_sub, M_train_sub)
        
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
    print("\n--- Evaluating Frozen Random Forest on Final Test Split (Days 313–365) ---")
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

