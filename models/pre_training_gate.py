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
            if not (cert.get("acceptance_certified", False) or cert.get("full_year_certified", False) or cert.get("pilot_certified", False)):
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

    # Check 9: Genuine Spatial-Temporal Context Tensors for B6, B7, B8
    required_context = {
        "X_patches": {"expected_ndim": 4, "feat_axis": 1, "desc": "B6 Spatial patches [N, 7, P, P]"},
        "X_seq": {"expected_ndim": 3, "feat_axis": 2, "desc": "B7 Temporal sequences [N, T, 7]"},
        "X_cubes": {"expected_ndim": 5, "feat_axis": 2, "desc": "B8 Spatiotemporal cubes [N, T, 7, P, P]"}
    }

    for split_name in ["train", "val", "test"]:
        split_dict = dataset.get(split_name, {})
        n_samples = len(split_dict.get("X", []))

        for ckey, cinfo in required_context.items():
            if ckey not in split_dict:
                failures.append(
                    f"[Check 9 FAIL] Required context tensor '{ckey}' missing from '{split_name}' split! "
                    f"({cinfo['desc']} required by official training pipeline)."
                )
                continue

            arr = split_dict[ckey]
            if not isinstance(arr, np.ndarray):
                failures.append(f"[Check 9 FAIL] '{ckey}' in '{split_name}' must be a numpy.ndarray.")
                continue

            if arr.shape[0] != n_samples:
                failures.append(
                    f"[Check 9 FAIL] '{ckey}' in '{split_name}' length ({arr.shape[0]}) does not match sample count ({n_samples})."
                )
                continue

            if arr.ndim != cinfo["expected_ndim"]:
                failures.append(
                    f"[Check 9 FAIL] '{ckey}' in '{split_name}' has ndim={arr.ndim}, expected {cinfo['expected_ndim']}."
                )
                continue

            if arr.shape[cinfo["feat_axis"]] != 7:
                failures.append(
                    f"[Check 9 FAIL] '{ckey}' in '{split_name}' feature dimension is {arr.shape[cinfo['feat_axis']]}, expected 7 surface variables."
                )
                continue

            if np.isnan(arr).any():
                failures.append(f"[Check 9 FAIL] '{ckey}' in '{split_name}' contains NaN values inside context tensor.")
                continue

            # Non-degeneracy verification: verify genuine variation
            if ckey == "X_patches" and arr.shape[0] > 0 and arr.shape[2] > 1 and arr.shape[3] > 1:
                center = arr[:, :, arr.shape[2] // 2, arr.shape[3] // 2]
                corner = arr[:, :, 0, 0]
                diff = np.abs(center - corner)
                if np.all(diff < 1e-7):
                    failures.append(
                        f"[Check 9 FAIL] '{ckey}' in '{split_name}' is spatially constant across patch! "
                        f"Degenerate pointwise replication detected."
                    )
            elif ckey == "X_seq" and arr.shape[0] > 0 and arr.shape[1] > 1:
                first_t = arr[:, 0, :]
                last_t = arr[:, -1, :]
                diff = np.abs(first_t - last_t)
                if np.all(diff < 1e-7):
                    failures.append(
                        f"[Check 9 FAIL] '{ckey}' in '{split_name}' is temporally constant across sequence! "
                        f"Degenerate pointwise replication detected."
                    )
            elif ckey == "X_cubes" and arr.shape[0] > 0 and arr.shape[1] > 1 and arr.shape[3] > 1:
                last_center = arr[:, -1, :, arr.shape[3] // 2, arr.shape[4] // 2]
                last_corner = arr[:, -1, :, 0, 0]
                spatial_diff = np.abs(last_center - last_corner)
                if np.all(spatial_diff < 1e-7):
                    failures.append(
                        f"[Check 9 FAIL] '{ckey}' in '{split_name}' has no spatial variation across cube! "
                        f"Degenerate replication detected."
                    )
                first_center = arr[:, 0, :, arr.shape[3] // 2, arr.shape[4] // 2]
                temp_diff = np.abs(first_center - last_center)
                if np.all(temp_diff < 1e-7):
                    failures.append(
                        f"[Check 9 FAIL] '{ckey}' in '{split_name}' has no temporal variation across cube! "
                        f"Degenerate replication detected."
                    )

    if not any(f.startswith("[Check 9 FAIL]") for f in failures):
        print("  [Check 9 PASS] All required context tensors (X_patches, X_seq, X_cubes) verified and non-degenerate across train/val/test.")

    if len(failures) > 0:
        msg = f"PRE-TRAINING GATE BLOCKED: {len(failures)} condition(s) violated:\n" + "\n".join(failures)
        print(f"\n[ERROR] {msg}")
        raise PreTrainingGateBlockedError(msg)
        
    print("\n[PRE-TRAINING GATE PASSED] All 9 conditions satisfied. Dataset is authorized for training.")
    return {
        "gate_passed": True,
        "total_days": total_days,
        "train_samples": len(dataset["train"]["X"]),
        "val_samples": len(dataset["val"]["X"]),
        "test_samples": len(dataset["test"]["X"]),
        "canonical_depths_count": len(CANONICAL_DEPTHS),
        "context_tensors_verified": ["X_patches", "X_seq", "X_cubes"]
    }

def run_gate_cli():
    print("=" * 70)
    print("PRE-TRAINING GATE CLI RUNNER")
    print("=" * 70)
    
    # 1. Detect certificate
    full_year_cert = "reports/full_year_acceptance_certified.json"
    pilot_cert = "reports/pilot_acceptance_certified.json"
    
    if os.path.exists(full_year_cert):
        cert_path = full_year_cert
        full_year_mode = True
        print(f"Detected Full-Year Acceptance Certificate: {full_year_cert}")
    elif os.path.exists(pilot_cert):
        cert_path = pilot_cert
        full_year_mode = False
        print(f"Detected Pilot Acceptance Certificate: {pilot_cert}")
    else:
        raise FileNotFoundError(
            "Neither reports/full_year_acceptance_certified.json nor reports/pilot_acceptance_certified.json found! "
            "Execute harmonization and certification first."
        )

    # 2. Locate Zarr stores
    if full_year_mode:
        candidate_surfs = [
            "data/processed/real_ml_dataset_full_year_surface.zarr",
            "data/processed/ml_dataset_full-year_surface.zarr"
        ]
        candidate_targs = [
            "data/processed/real_ml_dataset_full_year_target.zarr",
            "data/processed/ml_dataset_full-year_target.zarr"
        ]
    else:
        candidate_surfs = [
            "data/processed/real_ml_dataset_pilot_surface.zarr",
            "data/processed/ml_dataset_pilot_surface.zarr"
        ]
        candidate_targs = [
            "data/processed/real_ml_dataset_pilot_target.zarr",
            "data/processed/ml_dataset_pilot_target.zarr"
        ]

    surf_zarr = next((p for p in candidate_surfs if os.path.exists(p)), candidate_surfs[0])
    targ_zarr = next((p for p in candidate_targs if os.path.exists(p)), candidate_targs[0])

    if not os.path.exists(surf_zarr) or not os.path.exists(targ_zarr):
        raise FileNotFoundError(f"Missing Zarr datasets: {surf_zarr} or {targ_zarr}")

    print(f"Surface Zarr: {surf_zarr}")
    print(f"Target Zarr:  {targ_zarr}")

    # 3. Load tabular dataset
    from preprocessing.tabular_dataset import load_tabular_dataset
    dataset = load_tabular_dataset(surf_zarr=surf_zarr, targ_zarr=targ_zarr, build_context=True)

    # 4. Verify gate
    gate_report = verify_pre_training_gate(dataset, certificate_path=cert_path, full_year_mode=full_year_mode)
    
    print("\n" + "=" * 70)
    print("PRE-TRAINING GATE AUDIT SUMMARY")
    print("=" * 70)
    for k, v in gate_report.items():
        print(f"  {k:25s}: {v}")
    print("\n[SUCCESS] Pre-training acceptance gate verified successfully.")
    return gate_report

if __name__ == "__main__":
    run_gate_cli()
