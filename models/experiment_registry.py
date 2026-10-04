"""
models/experiment_registry.py
Structured experiment registry and logging engine for scientific ML benchmarks.
Records complete provenance, git commit SHA, configuration, normalization checksums,
and evaluated metrics for every benchmark experiment.
"""

import os
import sys
import json
import time
import subprocess
import hashlib

REGISTRY_DIR = "reports/experiment_registry"

def get_git_commit_sha():
    """Retrieves current git commit SHA."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_GIT_SHA"

def compute_file_sha256(filepath):
    """Computes SHA-256 checksum of any file."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

class ExperimentRegistry:
    """
    Manages experiment runs, recording complete scientific configuration and metrics.
    """
    def __init__(self, registry_dir=REGISTRY_DIR):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def log_experiment_run(self, model_id, model_config, split_metadata,
                           scaler_path, metrics_summary,
                           dataset_version="2020-full-year-v1.0",
                           random_seed=42, optimizer=None, hyperparameters=None,
                           checkpoint_path=None, notes=None):
        """
        Logs a single experiment execution with complete provenance.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{model_id.lower()}_{timestamp}"
        
        git_sha = get_git_commit_sha()
        scaler_checksum = compute_file_sha256(scaler_path) if scaler_path else None
        
        record = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "git_sha": git_sha,
            "dataset_version": dataset_version,
            "model_id": model_id,
            "model_config": model_config,
            "random_seed": random_seed,
            "optimizer": optimizer,
            "hyperparameters": hyperparameters or {},
            "split_metadata": split_metadata,
            "normalization_artifact": {
                "path": scaler_path,
                "sha256": scaler_checksum
            },
            "checkpoint_path": checkpoint_path,
            "metrics": metrics_summary,
            "notes": notes or ""
        }
        
        out_path = os.path.join(self.registry_dir, f"{run_id}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
            
        print(f"[REGISTRY] Experiment run {run_id} recorded at {out_path}")
        return record

    def list_experiments(self):
        """Lists all registered experiment JSONs."""
        files = [os.path.join(self.registry_dir, f) for f in os.listdir(self.registry_dir) if f.endswith(".json")]
        records = []
        for fp in sorted(files):
            with open(fp, "r") as f:
                records.append(json.load(f))
        return records
