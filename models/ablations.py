"""
models/ablations.py
Configurable Ablation Framework implementing Section 8 of the Scientific ML Experiment Protocol:

Feature Ablations (A1–A8):
- A1: Remove SST (no analysed_sst)
- A2: Remove SSH (no sla)
- A3: Remove SSS (no sos)
- A4: Remove Currents (no u_curr, v_curr)
- A5: Remove Wind (no u_wind, v_wind)
- A6: All except SST (analysed_sst only)
- A7: All except SSH (sla only)
- A8: Full feature set (all 7 surface variables) [Reference]

Context Ablations (C1–C5):
- C1: Reduce spatial patch to 3 x 3
- C2: Reduce spatial patch to 1 x 1 (pure pointwise)
- C3: Remove temporal context (predict day t from surface fields at day t only)
- C4: Reduce temporal window to 1 day
- C5: Reduce embedding dimension by 50%
"""

from preprocessing.canonical_grid import CANONICAL_FEATURES

FEATURE_ABLATION_CONFIGS = {
    "A1": {
        "name": "no_sst",
        "description": "Remove sst",
        "features": [f for f in CANONICAL_FEATURES if f != "sst"]
    },
    "A2": {
        "name": "no_ssh",
        "description": "Remove ssh",
        "features": [f for f in CANONICAL_FEATURES if f != "ssh"]
    },
    "A3": {
        "name": "no_sss",
        "description": "Remove sss",
        "features": [f for f in CANONICAL_FEATURES if f != "sss"]
    },
    "A4": {
        "name": "no_currents",
        "description": "Remove current_u and current_v",
        "features": [f for f in CANONICAL_FEATURES if f not in ("current_u", "current_v")]
    },
    "A5": {
        "name": "no_wind",
        "description": "Remove wind_u and wind_v",
        "features": [f for f in CANONICAL_FEATURES if f not in ("wind_u", "wind_v")]
    },
    "A6": {
        "name": "sst_only",
        "description": "All except SST (sst only)",
        "features": ["sst"]
    },
    "A7": {
        "name": "ssh_only",
        "description": "All except SSH (ssh only)",
        "features": ["ssh"]
    },
    "A8": {
        "name": "full_reference",
        "description": "Full canonical feature set (all 7 surface variables)",
        "features": list(CANONICAL_FEATURES)
    }
}

CONTEXT_ABLATION_CONFIGS = {
    "C1": {
        "name": "spatial_patch_3x3",
        "description": "Reduce spatial patch to 3x3",
        "patch_size": 3,
        "temporal_window": 5,
        "embedding_dim": 128
    },
    "C2": {
        "name": "spatial_patch_1x1",
        "description": "Reduce spatial patch to 1x1 (pointwise equivalent)",
        "patch_size": 1,
        "temporal_window": 5,
        "embedding_dim": 128
    },
    "C3": {
        "name": "no_temporal_context",
        "description": "Remove temporal context (predict day t from day t only)",
        "patch_size": 3,
        "temporal_window": 1,
        "embedding_dim": 128
    },
    "C4": {
        "name": "temporal_window_1",
        "description": "Reduce temporal window to 1 day",
        "patch_size": 3,
        "temporal_window": 1,
        "embedding_dim": 128
    },
    "C5": {
        "name": "embedding_dim_half",
        "description": "Reduce embedding dimension by 50%",
        "patch_size": 3,
        "temporal_window": 5,
        "embedding_dim": 64
    }
}

def get_feature_ablation_indices(ablation_id):
    """
    Returns the integer column indices in CANONICAL_FEATURES for an ablation ID (A1..A8).
    """
    cfg = FEATURE_ABLATION_CONFIGS.get(ablation_id.upper())
    if cfg is None:
        raise KeyError(f"Unknown feature ablation '{ablation_id}'. Available: {list(FEATURE_ABLATION_CONFIGS.keys())}")
    
    indices = [CANONICAL_FEATURES.index(f) for f in cfg["features"]]
    return indices, cfg["features"]

def get_context_ablation_config(ablation_id):
    """
    Returns the context configuration dictionary for an ablation ID (C1..C5).
    """
    cfg = CONTEXT_ABLATION_CONFIGS.get(ablation_id.upper())
    if cfg is None:
        raise KeyError(f"Unknown context ablation '{ablation_id}'. Available: {list(CONTEXT_ABLATION_CONFIGS.keys())}")
    return cfg
