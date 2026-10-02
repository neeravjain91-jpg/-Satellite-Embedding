"""
models/05_pointwise_mlp.py
Model 5: Pointwise Multi-Layer Perceptron (PyTorch)

Negative Constraints & Architecture Rules:
- Pure pointwise feed-forward network: Linear -> ReLU -> Linear ... -> Linear(15)
- NO spatial convolution (no Conv2D, no Conv3D)
- NO temporal recurrence (no ConvLSTM, no LSTM, no GRU)
- NO attention mechanisms (no Self-Attention, no Cross-Attention, no Transformers, no ViT)
- NO hybrid model structures

Protocol:
1. Trains on normalized training split features strictly using masked MSE loss.
2. Evaluates after each epoch on the full Validation Split (Days 256-309, N=1,003,374).
3. Early stops and saves the best model checkpoint based strictly on Validation masked RMSE.
4. Freezes optimal checkpoint.
5. Final evaluation on the held-out Test Split (Days 310-365, N=1,040,536).
Zero Test Leakage: Test split is never used for training, hyperparameter tuning, or early stopping.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.tabular_dataset import load_tabular_dataset
from models.base import compute_masked_metrics, print_evaluation_summary, save_model_evaluation
from preprocessing.canonical_grid import CANONICAL_DEPTHS

class PointwiseMLP(nn.Module):
    """
    Pointwise Multi-Layer Perceptron for 1D surface-to-depth column mapping.
    Maps 7 surface variables at a single ocean coordinate to 15 vertical depth temperatures.
    """
    def __init__(self, in_features=7, out_features=15, hidden_dims=[128, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def masked_mse_loss(y_pred, y_true, mask):
    """
    Masked MSE loss respecting variable bathymetric depth cutoffs.
    y_pred: (B, 15)
    y_true: (B, 15)
    mask:   (B, 15) bool tensor
    """
    diff = (y_pred - y_true) * mask.float()
    loss = (diff ** 2).sum() / (mask.float().sum() + 1e-8)
    return loss

def evaluate_torch_model(model, dataloader, device="cpu"):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
    return np.vstack(all_preds)

def train_and_evaluate_mlp(dataset, batch_size=4096, max_epochs=20, patience=4, sample_train_size=500000):
    print("=" * 70)
    print("PHASE 1: MODEL 5 - POINTWISE MULTI-LAYER PERCEPTRON (PyTorch)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    
    X_train_full = dataset["train"]["X_norm"]
    Y_train_full = dataset["train"]["Y"]
    M_train_full = dataset["train"]["mask"]
    
    X_val = dataset["val"]["X_norm"]
    Y_val = dataset["val"]["Y"]
    M_val = dataset["val"]["mask"]
    
    X_test = dataset["test"]["X_norm"]
    Y_test = dataset["test"]["Y"]
    M_test = dataset["test"]["mask"]
    
    # Subsample training data for efficient convergence
    np.random.seed(42)
    N_train = len(X_train_full)
    sub_indices = np.random.choice(N_train, size=min(sample_train_size, N_train), replace=False)
    
    X_train_sub = X_train_full[sub_indices]
    Y_train_sub = np.nan_to_num(Y_train_full[sub_indices], nan=0.0)
    M_train_sub = M_train_full[sub_indices]
    
    print(f"Training on {len(X_train_sub):,} representative samples with batch size {batch_size}.")
    print(f"Evaluating after each epoch on full Validation Split (N={len(X_val):,}).")
    
    # Create PyTorch Datasets
    train_tensor_x = torch.from_numpy(X_train_sub).float()
    train_tensor_y = torch.from_numpy(Y_train_sub).float()
    train_tensor_m = torch.from_numpy(M_train_sub).bool()
    
    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_y, train_tensor_m),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float()),
        batch_size=batch_size * 2,
        shuffle=False
    )
    
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float()),
        batch_size=batch_size * 2,
        shuffle=False
    )
    
    # Model instantiation
    torch.manual_seed(42)
    model = PointwiseMLP(in_features=7, out_features=15, hidden_dims=[128, 128, 64]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    checkpoint_dir = os.path.join(repo_root, "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "pointwise_mlp_best.pt")
    
    best_val_rmse = float("inf")
    patience_counter = 0
    
    print("\n--- Training Loop with Validation-Driven Early Stopping ---")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_accum = 0.0
        n_batches = 0
        
        for bx, by, bm in train_loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = masked_mse_loss(pred, by, bm)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            n_batches += 1
            
        avg_train_loss = train_loss_accum / max(n_batches, 1)
        
        # Validation Evaluation strictly for early stopping
        val_preds = evaluate_torch_model(model, val_loader, device=device)
        val_summary, _ = compute_masked_metrics(
            y_true=Y_val,
            y_pred=val_preds,
            mask=M_val,
            model_name="Pointwise MLP",
            split_name="val"
        )
        val_rmse = val_summary["overall_rmse"]
        scheduler.step(val_rmse)
        epoch_time = time.time() - t0
        
        print(f"Epoch {epoch:2d}/{max_epochs:2d} | Train Loss: {avg_train_loss:.6f} | Val RMSE: {val_rmse:.4f} °C | R^2: {val_summary['overall_r2']:.4f} ({epoch_time:.1f}s)")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val RMSE: {best_val_rmse:.4f} °C)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Validation performance has not improved for {patience} epochs.")
                break
                
    # Load Best Model Checkpoint
    print(f"\nLoading best checkpoint from {best_checkpoint_path}...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    model.eval()
    
    # 1. Full Validation Split Evaluation
    print("\n--- Evaluating Frozen Pointwise MLP on Validation Split (Days 256-309, N=1,003,374) ---")
    val_preds_best = evaluate_torch_model(model, val_loader, device=device)
    val_summary, df_val_depths = compute_masked_metrics(
        y_true=Y_val,
        y_pred=val_preds_best,
        mask=M_val,
        model_name="Pointwise MLP",
        split_name="val"
    )
    print_evaluation_summary(val_summary, df_val_depths)
    save_model_evaluation(val_summary, df_val_depths)
    
    # 2. Final Test Split Evaluation (Held-out, Zero Leakage)
    print(f"\n--- Evaluating Frozen Pointwise MLP on Final Test Split (Days 310-365, N={len(X_test):,}) ---")
    test_preds = evaluate_torch_model(model, test_loader, device=device)
    test_summary, df_test_depths = compute_masked_metrics(
        y_true=Y_test,
        y_pred=test_preds,
        mask=M_test,
        model_name="Pointwise MLP",
        split_name="test"
    )
    print_evaluation_summary(test_summary, df_test_depths)
    save_model_evaluation(test_summary, df_test_depths)
    
    return val_summary, test_summary

if __name__ == "__main__":
    dataset = load_tabular_dataset()
    train_and_evaluate_mlp(dataset)
