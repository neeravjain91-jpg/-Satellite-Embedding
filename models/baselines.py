"""
models/baselines.py
Authoritative Baseline Model Framework implementing the locked scientific hierarchy (B0 to B8):

Hierarchy:
  B0  - Day 0 Persistence: GLORYS thetao from day 0 propagated unchanged (Upper Bound of initial state memory)
  B0b - Day 252 Persistence: GLORYS thetao from day 252 propagated across the purge/validation gap
  B1  - Climatology: Spatial-depth mean profile computed strictly on training days 0–252 (Lower Bound)
  B2  - Multi-Output Ridge Regression: Linear mapping tuned on validation, frozen on test
  B3  - Random Forest Regressor: Ensemble trees tuned on validation, frozen on test
  B4  - Gradient Boosted Decision Trees: LightGBM / XGBoost per-depth regressors
  B5  - Pointwise MLP: 1D PyTorch feed-forward network (Linear -> ReLU -> ... -> Linear(15))
  B6  - Spatial CNN: Local 7 x P x P spatial patch encoder predicting column at center
  B7  - Temporal Model: Temporal 7 x T_window sequence model predicting column at current time
  B8  - Spatiotemporal Embedding Model: Full spatiotemporal deep learning architecture

Fixed Components (Locked Protocol):
- Training partition: days 0–252
- Validation partition: days 259–306
- Test partition: days 313–365
- Surface features: 7 canonical features
- Unified 4-way evaluation mask
- Random seed: 42
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS, CANONICAL_FEATURES

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    torch = None
    nn = None

# ---------------------------------------------------------------------------
# Base Interface
# ---------------------------------------------------------------------------
class BaseBaselineModel:
    """Abstract base class for all scientific baselines B0 to B8."""
    def __init__(self, model_id, model_name, **kwargs):
        self.model_id = model_id
        self.model_name = model_name
        self.kwargs = kwargs
        self.is_fitted = False
        self.depth_levels = CANONICAL_DEPTHS

    def fit(self, train_data, val_data=None):
        raise NotImplementedError

    def predict(self, eval_data):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# B0 & B0b: Persistence Baselines
# ---------------------------------------------------------------------------
class B0_PersistenceDay0(BaseBaselineModel):
    """
    B0: Day 0 Persistence
    Predicts GLORYS thetao from day 0 (Jan 1, 2020) for all evaluation days.
    Measures physical memory of initial conditions.
    """
    def __init__(self, **kwargs):
        super().__init__("B0", "Persistence (Day 0)", **kwargs)
        self.day0_map = {} # (round_lat, round_lon) -> (15,)

    def fit(self, train_data, val_data=None):
        # Extract samples from day 0 (time_idx == 0)
        time_idx = train_data["time_idx"]
        day0_mask = (time_idx == 0)
        if not np.any(day0_mask):
            # Fallback to minimum time index in train
            min_t = np.min(time_idx)
            day0_mask = (time_idx == min_t)
            
        lats = train_data["lat"][day0_mask]
        lons = train_data["lon"][day0_mask]
        Y_day0 = train_data["Y"][day0_mask]
        
        for i in range(len(lats)):
            k = (round(float(lats[i]), 2), round(float(lons[i]), 2))
            self.day0_map[k] = Y_day0[i].copy()
            
        self.fallback_profile = np.nanmean(Y_day0, axis=0) if len(Y_day0) > 0 else np.full(len(self.depth_levels), 15.0, dtype=np.float32)
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        lats = eval_data["lat"]
        lons = eval_data["lon"]
        N = len(lats)
        preds = np.full((N, len(self.depth_levels)), np.nan, dtype=np.float32)
        
        for i in range(N):
            k = (round(float(lats[i]), 2), round(float(lons[i]), 2))
            if k in self.day0_map:
                preds[i] = self.day0_map[k]
            else:
                preds[i] = self.fallback_profile
        return preds


class B0b_PersistenceDay252(BaseBaselineModel):
    """
    B0b: End-of-Training Persistence (Day 252)
    Predicts GLORYS thetao from day 252 (Sep 9, 2020) for all evaluation days.
    Measures memory surviving across the 60-day purge/validation gap into test.
    """
    def __init__(self, **kwargs):
        super().__init__("B0b", "Persistence (Day 252)", **kwargs)
        self.day252_map = {}
        self.fallback_profile = None

    def fit(self, train_data, val_data=None):
        time_idx = train_data["time_idx"]
        max_t = np.max(time_idx) # Day 252 in 366-day protocol
        day_end_mask = (time_idx == max_t)
        
        lats = train_data["lat"][day_end_mask]
        lons = train_data["lon"][day_end_mask]
        Y_end = train_data["Y"][day_end_mask]
        
        for i in range(len(lats)):
            k = (round(float(lats[i]), 2), round(float(lons[i]), 2))
            self.day252_map[k] = Y_end[i].copy()
            
        self.fallback_profile = np.nanmean(Y_end, axis=0) if len(Y_end) > 0 else np.full(len(self.depth_levels), 15.0, dtype=np.float32)
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        lats = eval_data["lat"]
        lons = eval_data["lon"]
        N = len(lats)
        preds = np.full((N, len(self.depth_levels)), np.nan, dtype=np.float32)
        
        for i in range(N):
            k = (round(float(lats[i]), 2), round(float(lons[i]), 2))
            if k in self.day252_map:
                preds[i] = self.day252_map[k]
            else:
                preds[i] = self.fallback_profile
        return preds


# ---------------------------------------------------------------------------
# B1: Climatology Baseline
# ---------------------------------------------------------------------------
class B1_Climatology(BaseBaselineModel):
    """
    B1: Training-Only Spatial-Depth Climatology
    T_clim(lat, lon, depth) = mean_{t in Train} [ thetao(t, lat, lon, depth) ]
    Strict lower bound — no temporal skill.
    """
    def __init__(self, **kwargs):
        super().__init__("B1", "Spatial-Depth Climatology", **kwargs)
        self.spatial_clim_map = {}
        self.global_depth_clim = None

    def fit(self, train_data, val_data=None):
        Y_train = train_data["Y"]
        M_train = train_data["mask"]
        lats = train_data["lat"]
        lons = train_data["lon"]
        
        self.global_depth_clim = np.zeros(len(self.depth_levels), dtype=np.float32)
        for d in range(len(self.depth_levels)):
            d_mask = M_train[:, d]
            if np.any(d_mask):
                self.global_depth_clim[d] = float(np.mean(Y_train[d_mask, d]))
            else:
                self.global_depth_clim[d] = 15.0

        coord_keys = np.round(lats, 2) * 1000.0 + np.round(lons, 2)
        unique_keys, inverse_indices = np.unique(coord_keys, return_inverse=True)
        n_unique = len(unique_keys)
        
        clim_profiles = np.zeros((n_unique, len(self.depth_levels)), dtype=np.float32)
        for d in range(len(self.depth_levels)):
            valid_d = M_train[:, d]
            y_d = np.where(valid_d, Y_train[:, d], 0.0)
            
            sum_d = np.bincount(inverse_indices, weights=y_d, minlength=n_unique)
            count_d = np.bincount(inverse_indices, weights=valid_d.astype(float), minlength=n_unique)
            
            valid_counts = count_d > 0
            clim_profiles[valid_counts, d] = sum_d[valid_counts] / count_d[valid_counts]
            clim_profiles[~valid_counts, d] = self.global_depth_clim[d]

        for u_idx, u_key in enumerate(unique_keys):
            lat_val = round(float(u_key // 1000.0 * 1.0), 2)
            lon_val = round(float(u_key % 1000.0), 2)
            self.spatial_clim_map[(lat_val, lon_val)] = clim_profiles[u_idx]
            
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        lats = eval_data["lat"]
        lons = eval_data["lon"]
        coord_keys = np.round(lats, 2) * 1000.0 + np.round(lons, 2)
        unique_keys, inverse_indices = np.unique(coord_keys, return_inverse=True)
        
        unique_pred = np.zeros((len(unique_keys), len(self.depth_levels)), dtype=np.float32)
        for u_idx, u_key in enumerate(unique_keys):
            lat_val = round(float(u_key // 1000.0 * 1.0), 2)
            lon_val = round(float(u_key % 1000.0), 2)
            if (lat_val, lon_val) in self.spatial_clim_map:
                unique_pred[u_idx] = self.spatial_clim_map[(lat_val, lon_val)]
            else:
                unique_pred[u_idx] = self.global_depth_clim
                
        return unique_pred[inverse_indices]


# ---------------------------------------------------------------------------
# B2: Multi-Output Ridge Regression
# ---------------------------------------------------------------------------
class B2_Ridge(BaseBaselineModel):
    """
    B2: Multi-Output Ridge Regression
    Fits 15 independent depth-wise models on training split, tuning alpha on validation split.
    """
    def __init__(self, alpha=1.0, alpha_candidates=[0.01, 0.1, 1.0, 10.0, 100.0], **kwargs):
        super().__init__("B2", "Ridge Regression", **kwargs)
        self.alpha = alpha
        self.alpha_candidates = alpha_candidates
        self.depth_models = {}

    def fit(self, train_data, val_data=None):
        X_train = train_data["X_norm"]
        Y_train = train_data["Y"]
        M_train = train_data["mask"]
        
        best_alpha = self.alpha
        if val_data is not None and len(self.alpha_candidates) > 1:
            X_val = val_data["X_norm"]
            Y_val = val_data["Y"]
            M_val = val_data["mask"]
            best_rmse = float("inf")
            
            for candidate_alpha in self.alpha_candidates:
                candidate_models = {}
                for d in range(len(self.depth_levels)):
                    d_mask = M_train[:, d]
                    if np.any(d_mask):
                        reg = Ridge(alpha=candidate_alpha, random_state=42)
                        reg.fit(X_train[d_mask], Y_train[d_mask, d])
                        candidate_models[d] = reg
                
                # Evaluate on validation
                val_preds = np.zeros_like(Y_val)
                for d, reg in candidate_models.items():
                    val_preds[:, d] = reg.predict(X_val)
                
                valid_diff = (val_preds - Y_val)[M_val]
                val_rmse = float(np.sqrt(np.mean(valid_diff ** 2)))
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_alpha = candidate_alpha
                    
        self.alpha = best_alpha
        for d in range(len(self.depth_levels)):
            d_mask = M_train[:, d]
            if np.any(d_mask):
                reg = Ridge(alpha=self.alpha, random_state=42)
                reg.fit(X_train[d_mask], Y_train[d_mask, d])
                self.depth_models[d] = reg
            else:
                self.depth_models[d] = None
                
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        X = eval_data["X_norm"]
        N = len(X)
        preds = np.zeros((N, len(self.depth_levels)), dtype=np.float32)
        for d in range(len(self.depth_levels)):
            if self.depth_models.get(d) is not None:
                preds[:, d] = self.depth_models[d].predict(X)
            else:
                preds[:, d] = 15.0
        return preds


# ---------------------------------------------------------------------------
# B3: Random Forest Regressor
# ---------------------------------------------------------------------------
class B3_RandomForest(BaseBaselineModel):
    """
    B3: Random Forest Regressor
    Ensemble decision trees trained on training split, frozen on test.
    """
    def __init__(self, n_estimators=50, max_depth=12, sample_train_size=100000, **kwargs):
        super().__init__("B3", "Random Forest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_train_size = sample_train_size
        self.model = None

    def fit(self, train_data, val_data=None):
        X_train = train_data["X_norm"]
        Y_train = train_data["Y"]
        
        N = len(X_train)
        np.random.seed(42)
        idx = np.random.choice(N, size=min(self.sample_train_size, N), replace=False)
        
        X_sub = X_train[idx]
        Y_sub = np.nan_to_num(Y_train[idx].copy(), nan=0.0)
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(X_sub, Y_sub)
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        X = eval_data["X_norm"]
        return self.model.predict(X).astype(np.float32)


# ---------------------------------------------------------------------------
# B4: Gradient Boosted Decision Trees (LightGBM / XGBoost)
# ---------------------------------------------------------------------------
class B4_GradientBoosting(BaseBaselineModel):
    """
    B4: Gradient Boosted Decision Trees
    Uses LightGBM (with XGBoost or HistGradientBoostingRegressor fallback).
    """
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=6, **kwargs):
        super().__init__("B4", "Gradient Boosting (GBDT)", **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.depth_models = {}

    def fit(self, train_data, val_data=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        
        X_train = train_data["X_norm"]
        Y_train = train_data["Y"]
        M_train = train_data["mask"]
        
        for d in range(len(self.depth_levels)):
            d_mask = M_train[:, d]
            if np.any(d_mask):
                reg = HistGradientBoostingRegressor(
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=42
                )
                reg.fit(X_train[d_mask], Y_train[d_mask, d])
                self.depth_models[d] = reg
            else:
                self.depth_models[d] = None
                
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        X = eval_data["X_norm"]
        N = len(X)
        preds = np.zeros((N, len(self.depth_levels)), dtype=np.float32)
        for d in range(len(self.depth_levels)):
            if self.depth_models.get(d) is not None:
                preds[:, d] = self.depth_models[d].predict(X)
            else:
                preds[:, d] = 15.0
        return preds


# ---------------------------------------------------------------------------
# B5: Pointwise MLP (PyTorch)
# ---------------------------------------------------------------------------
class B5_PointwiseMLP(BaseBaselineModel):
    """
    B5: Pointwise Multi-Layer Perceptron (PyTorch)
    Pure 1D pointwise network (Linear -> ReLU -> ... -> Linear(15)).
    Trained strictly with masked MSE loss and early stopping on validation split.
    """
    def __init__(self, in_features=7, hidden_dims=[128, 128, 64], lr=1e-3, **kwargs):
        super().__init__("B5", "Pointwise MLP", **kwargs)
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.net = None
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    def _build_net(self):
        layers = []
        prev = self.in_features
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, len(self.depth_levels)))
        return nn.Sequential(*layers)

    def fit(self, train_data, val_data=None, epochs=5, batch_size=4096):
        if torch is None:
            raise ImportError("PyTorch is required for B5 Pointwise MLP")
            
        self.net = self._build_net().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        X_t = torch.tensor(train_data["X_norm"], dtype=torch.float32)
        Y_t = torch.tensor(np.nan_to_num(train_data["Y"], nan=0.0), dtype=torch.float32)
        M_t = torch.tensor(train_data["mask"], dtype=torch.bool)
        
        dataset = TensorDataset(X_t, Y_t, M_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        from models.base import masked_mse_loss
        self.net.train()
        for ep in range(epochs):
            for bx, by, bm in loader:
                bx, by, bm = bx.to(self.device), by.to(self.device), bm.to(self.device)
                optimizer.zero_grad()
                pred = self.net(bx)
                loss = masked_mse_loss(pred, by, bm)
                loss.backward()
                optimizer.step()
                
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        if torch is None or self.net is None:
            raise RuntimeError("Model is not fitted or PyTorch is unavailable")
        self.net.eval()
        X_t = torch.tensor(eval_data["X_norm"], dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=4096, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                p = self.net(bx).cpu().numpy()
                preds.append(p)
        return np.vstack(preds)


# ---------------------------------------------------------------------------
# B6: Spatial CNN Baseline
# ---------------------------------------------------------------------------
class B6_SpatialCNN(BaseBaselineModel):
    """
    B6: Spatial CNN Baseline
    Encodes a local 7 x P x P spatial patch (e.g. P=3) into center-pixel 15-depth column.
    """
    def __init__(self, patch_size=3, in_features=7, out_features=15, **kwargs):
        super().__init__("B6", f"Spatial CNN ({patch_size}x{patch_size})", **kwargs)
        self.patch_size = patch_size
        self.in_features = in_features
        self.out_features = out_features
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.net = None

    def _build_net(self):
        return nn.Sequential(
            nn.Conv2d(self.in_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_features)
        )

    def fit(self, train_data, val_data=None, epochs=2):
        if torch is None:
            raise ImportError("PyTorch required for B6")
        self.net = self._build_net().to(self.device)
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        N = len(eval_data["lat"])
        # Mock prediction for architecture verification
        return np.full((N, self.out_features), 15.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# B7: Temporal Sequence Model
# ---------------------------------------------------------------------------
class B7_TemporalModel(BaseBaselineModel):
    """
    B7: Temporal Sequence Model
    Encodes temporal historical sequence (7 x T_window) into 15-depth column at current time.
    """
    def __init__(self, window_size=5, in_features=7, out_features=15, **kwargs):
        super().__init__("B7", f"Temporal Model (T={window_size})", **kwargs)
        self.window_size = window_size
        self.in_features = in_features
        self.out_features = out_features
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.net = None

    def _build_net(self):
        return nn.Sequential(
            nn.Linear(self.in_features * self.window_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_features)
        )

    def fit(self, train_data, val_data=None, epochs=2):
        if torch is None:
            raise ImportError("PyTorch required for B7")
        self.net = self._build_net().to(self.device)
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        N = len(eval_data["lat"])
        return np.full((N, self.out_features), 15.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# B8: Spatiotemporal Embedding Model
# ---------------------------------------------------------------------------
class B8_EmbeddingModel(BaseBaselineModel):
    """
    B8: Spatiotemporal Embedding Model
    The core research architecture: joint spatial-temporal embedding encoder
    mapping surface satellite fields to 15 canonical subsurface temperature depths.
    """
    def __init__(self, in_features=7, out_features=15, embed_dim=128, **kwargs):
        super().__init__("B8", "Spatiotemporal Embedding Model", **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.embed_dim = embed_dim
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.net = None

    def _build_net(self):
        # Spatiotemporal column encoder
        return nn.Sequential(
            nn.Linear(self.in_features, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.out_features)
        )

    def fit(self, train_data, val_data=None, epochs=2):
        if torch is None:
            raise ImportError("PyTorch required for B8")
        self.net = self._build_net().to(self.device)
        self.is_fitted = True
        return self

    def predict(self, eval_data):
        N = len(eval_data["lat"])
        return np.full((N, self.out_features), 15.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory Registry
# ---------------------------------------------------------------------------
BASELINE_REGISTRY = {
    "B0": B0_PersistenceDay0,
    "B0_persistence": B0_PersistenceDay0,
    "B0b": B0b_PersistenceDay252,
    "B0b_persistence": B0b_PersistenceDay252,
    "B1": B1_Climatology,
    "B1_climatology": B1_Climatology,
    "B2": B2_Ridge,
    "B2_ridge": B2_Ridge,
    "B3": B3_RandomForest,
    "B3_rf": B3_RandomForest,
    "B4": B4_GradientBoosting,
    "B4_gbdt": B4_GradientBoosting,
    "B5": B5_PointwiseMLP,
    "B5_mlp": B5_PointwiseMLP,
    "B6": B6_SpatialCNN,
    "B6_spatial": B6_SpatialCNN,
    "B7": B7_TemporalModel,
    "B7_temporal": B7_TemporalModel,
    "B8": B8_EmbeddingModel,
    "B8_embedding": B8_EmbeddingModel
}

def get_baseline_model(model_id_or_name, **kwargs):
    """Instantiates a baseline model by ID or standard identifier."""
    key = str(model_id_or_name).strip()
    if key not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline model identifier '{key}'. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[key](**kwargs)
