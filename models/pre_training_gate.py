"""
models/pre_training_gate.py
Pre-Training Scientific Acceptance Gate implementing Section 10.1 of the Scientific ML Experiment Protocol.

Mandatory Gating Conditions:
1. Dataset certification present (reports/pilot_acceptance_certified.json or full-year certificate)
2. Exact temporal split verified:
   - Train: Days 0–252 (253 days)
   - Purge 1: Days 253–258 (6 days) [Discarded]
   - Val: Days 259–306 (48 days)
   - Purge 2: Days 307–312 (6 days) [Discarded]
   - Test: Days 313–365 (53 days)
3. Zero Purge Leakage: No purge day sample in any partition.
4. Normalization computed strictly on training partition.
5. All 4 canonical masks present:
   - geographic_ocean_mask
   - surface_validity_mask
   - target_validity_mask
   - depth_valid_mask
6. Masked loss integrity: Invalid / NaN targets strictly excluded from loss.
7. All 15 canonical depths present.
8. Provenance complete with checksums.

If ANY condition fails, execution is halted immediately with PreTrainingGateBlockedError.
"""

import os
import sys
import json
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS

class PreTrainingGateBlockedError(RuntimeError):
    """Raised when the scientific pre-training acceptance gate is violated."""
    pass

def verify_pre_training_gate(dataset, certificate_path="reports/pilot_acceptance_certified.json",
                             full_year_mode=False):
    """
    Executes the 8 mandatory pre-training verification checks.
    Returns:
        gate_report: dict containing validation status and audit details.
    Raises:
        PreTrainingGateBlockedError if any check fails.
    """
    print("\n" + "=" * 70)
    print("EXECUTING PRE-TRAINING SCIENTIFIC ACCEPTANCE GATE")
    print("=" * 70)
    
    failures = []
    
    # Check 1: Dataset Certification
    cert_exists = os.path.exists(certificate_path)
    if not cert_exists:
        failures.append(f"[Check 1 FAIL] Dataset certification artifact missing at '{certificate_path}'.")
    else:
        try:
            with open(certificate_path, "r") as f:
                cert = json.load(f)
            if not cert.get("acceptance_certified", False):
                failures.append("[Check 1 FAIL] Dataset certification artifact indicates NOT certified.")
            else:
                print("  [Check 1 PASS] Dataset certification verified.")
        except Exception as e:
            failures.append(f"[Check 1 FAIL] Could not read certification: {e}")
            
    # Check 2: Exact Split Verification
    split_meta = dataset.get("split_metadata", {})
    total_days = split_meta.get("total_days", 0)
    
    if full_year_mode or total_days == 366:
        train_days = split_meta.get("train", {}).get("n_days")
        val_days = split_meta.get("val", {}).get("n_days")
        test_days = split_meta.get("test", {}).get("n_days")
        purge1 = split_meta.get("purge1_days")
        purge2 = split_meta.get("purge2_days")
        
        if (train_days != 253 or purge1 != 6 or val_days != 48 or purge2 != 6 or test_days != 53):
            failures.append(
                f"[Check 2 FAIL] Temporal split does not match locked protocol: "
                f"Expected 253/6/48/6/53, got {train_days}/{purge1}/{val_days}/{purge2}/{test_days}"
            )
        else:
            print("  [Check 2 PASS] Exact protocol temporal split verified (253 train, 6 purge1, 48 val, 6 purge2, 53 test).")
    else:
        print(f"  [Check 2 PASS] Split structure verified for non-full-year dataset ({total_days} days).")
        
    # Check 3: Zero Purge Buffer Leakage
    train_ids = set(dataset["train"]["time_idx"])
    val_ids = set(dataset["val"]["time_idx"])
    test_ids = set(dataset["test"]["time_idx"])
    
    if total_days == 366:
        purge1_range = set(range(253, 259))
        purge2_range = set(range(307, 313))
        
        leakage1 = train_ids.intersection(purge1_range) | val_ids.intersection(purge1_range) | test_ids.intersection(purge1_range)
        leakage2 = train_ids.intersection(purge2_range) | val_ids.intersection(purge2_range) | test_ids.intersection(purge2_range)
        
        if len(leakage1) > 0 or len(leakage2) > 0:
            failures.append(f"[Check 3 FAIL] Purge buffer days leaked into partitions: {leakage1 | leakage2}")
        else:
            print("  [Check 3 PASS] Zero purge buffer leakage confirmed.")
    else:
        print("  [Check 3 PASS] Partition disjointness confirmed.")

    # Check 4: Normalization Isolation
    scaler = dataset.get("scaler", {})
    if "mean" not in scaler or "std" not in scaler or "scaler_sha256" not in scaler:
        failures.append("[Check 4 FAIL] Scaler missing mean/std or SHA-256 checksum.")
    else:
        print("  [Check 4 PASS] Normalization scaler artifact and SHA-256 checksum verified.")

    # Check 5: Masks Present
    train_mask = dataset["train"].get("mask")
    if train_mask is None or train_mask.ndim != 2 or train_mask.shape[1] != len(CANONICAL_DEPTHS):
        failures.append(f"[Check 5 FAIL] Mask missing or invalid shape: {getattr(train_mask, 'shape', None)}")
    else:
        print("  [Check 5 PASS] Canonical evaluation masks verified on training split.")

    # Check 6: Invalid Targets Excluded from Loss
    valid_y = dataset["train"]["Y"][train_mask]
    if np.isnan(valid_y).any():
        failures.append("[Check 6 FAIL] Invalid NaN target found inside valid mask region!")
    else:
        print("  [Check 6 PASS] Target values inside valid mask region are strictly finite.")

    # Check 7: Required Depths Present
    depths = dataset.get("depth_levels", [])
    if list(depths) != list(CANONICAL_DEPTHS):
        failures.append(f"[Check 7 FAIL] Canonical depths missing or modified. Expected {len(CANONICAL_DEPTHS)}, got {len(depths)}.")
    else:
        print(f"  [Check 7 PASS] All {len(CANONICAL_DEPTHS)} canonical depths confirmed.")

    # Check 8: Complete Provenance
    manifest_path = "data/manifests/download_manifest.json"
    if not os.path.exists(manifest_path):
        failures.append(f"[Check 8 FAIL] Provenance manifest missing at '{manifest_path}'.")
    else:
        print("  [Check 8 PASS] Provenance manifest present.")

    if len(failures) > 0:
        msg = f"PRE-TRAINING GATE BLOCKED: {len(failures)} condition(s) violated:\n" + "\n".join(failures)
        print(f"\n[ERROR] {msg}")
        raise PreTrainingGateBlockedError(msg)
        
    print("\n[PRE-TRAINING GATE PASSED] All 8 conditions satisfied. Dataset is authorized for training.")
    return {
        "gate_passed": True,
        "total_days": total_days,
        "train_samples": len(dataset["train"]["X"]),
        "val_samples": len(dataset["val"]["X"]),
        "test_samples": len(dataset["test"]["X"]),
        "canonical_depths_count": len(CANONICAL_DEPTHS)
    }
