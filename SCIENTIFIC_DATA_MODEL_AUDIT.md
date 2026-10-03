# Scientific Data Model and Mask Integrity Audit

## Project
**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature**  
**Domain**: North Indian Ocean (5.00°N–30.00°N, 45.00°E–105.00°E)  
**Date**: October 2026  
**Audit Purpose**: Forensic and architectural inspection of land/ocean, target-validity, bathymetry, and missing-data masking before full-year acquisition.

---

## A. Current Mask Semantics

Prior to this hardening, the codebase utilized two primary masks defined in `preprocessing/ocean_mask.py`:
1. `ocean_mask_2d`:
   - **Current Meaning**: Binary mask intended to represent geographic ocean vs. land.
   - **Current Derivation**: Defined strictly as `~np.isnan(sample.values[0, :, :])` where `sample` is a single day (`time=0`) of GLORYS $\theta_o$ at depth index 0 (0 m).
   - **Flaw**: Conflates geographic coastline definition with the data availability of a single GLORYS daily assimilation time step at 0 m depth.
2. `ocean_mask_3d`:
   - **Current Meaning**: 3D boolean array intended to distinguish water column vs. land/sub-seabed.
   - **Current Derivation**: Evaluated as `~np.isnan(sample.values)` on day 0 of GLORYS $\theta_o$ across 15 depths.
   - **Flaw**: Fails to distinguish between:
     - Permanent bathymetric seafloor limitation (shallow water where seabed is at 50 m vs. 1000 m).
     - Missing observation / numerical missingness on a specific date.
     - True geographic land cells.

Furthermore, there were **no explicit masks** for:
- `target_validity_mask` $(T, D, H, W)$: Stating whether ground truth $\theta_o$ is physically valid at that specific date, depth, and pixel.
- `surface_validity_mask` $(T, H, W, C)$: Stating whether each satellite surface predictor is observed/valid vs. missing due to clouds, sensor track gaps, or land.
- `depth_valid_mask` $(D, H, W)$ or `deepest_valid_depth_m` $(H, W)$: Quantifying the static bathymetric water column depth limit independently of daily observational fluctuations.

---

## B. Current Array Shapes and Dimensions

| Array / Variable | Scope / File | Current Shape | Dimension Names |
| :--- | :--- | :--- | :--- |
| `surface_features` | `build_dataset.py` / Zarr | $(T, 101, 241, 7)$ | `(time, latitude, longitude, feature)` |
| `thetao` | `build_dataset.py` / Zarr | $(T, 15, 101, 241)$ | `(time, depth, latitude, longitude)` |
| `ocean_mask_2d` | `ocean_mask.py` / NetCDF | $(101, 241)$ | `(latitude, longitude)` |
| `ocean_mask_3d` | `ocean_mask.py` / NetCDF | $(15, 101, 241)$ | `(depth, latitude, longitude)` |
| `PyTorch X` | `OceanReconstructionDataset` | $(B, 7, 101, 241)$ | `(batch, feature, latitude, longitude)` |
| `PyTorch Y` | `OceanReconstructionDataset` | $(B, 15, 101, 241)$ | `(batch, depth, latitude, longitude)` |
| `PyTorch mask` | `OceanReconstructionDataset` | $(B, 15, 101, 241)$ | `(batch, depth, latitude, longitude)` |
| `Tabular X` | `tabular_dataset.py` | $(N, 7)$ | `(samples, feature)` |
| `Tabular Y` | `tabular_dataset.py` | $(N, 15)$ | `(samples, depth)` |
| `Tabular mask` | `tabular_dataset.py` | $(N, 15)$ | `(samples, depth)` |

---

## C. Where Geographic Masking is Inferred from GLORYS $\theta_o$

1. **`preprocessing/ocean_mask.py` — `generate_canonical_ocean_mask_from_glorys()`**:
   - Lines 30–35:
     ```python
     mask_3d = ~np.isnan(sample.values)
     mask_2d = mask_3d[0, :, :]
     ```
   - Geographic ocean cells are declared True solely if `sample.values[0, lat, lon]` is not NaN on the first time slice (`time=0`).
2. **`scripts/harmonize_and_validate.py`**:
   - Lines 580–588:
     ```python
     da_target_sample = xr.DataArray(target_array[0], dims=["depth", "latitude", "longitude"], ...)
     mask_ds = generate_canonical_ocean_mask_from_glorys(da_target_sample.to_dataset())
     ```
   - Directly triggers `generate_canonical_ocean_mask_from_glorys()` on `target_array[0]`.
3. **`preprocessing/regrid.py` — `regrid_2d_field()`**:
   - Lines 77–85:
     ```python
     mask_interp = RegularGridInterpolator(..., valid_mask.astype(float), ...)
     dst_mask = mask_interp(query_pts).reshape(...) >= 0.5
     ```
   - Uses the presence of valid numbers in individual source variables to infer a boundary.

---

## D. Where Missing Target Data May Be Interpreted as Land/Ocean

1. **Shallow vs. Deep Ocean Bathymetric Conflation**:
   - In shallow shelf regions (e.g., Persian Gulf, Gulf of Khambhat, Palk Strait, shallow shelf of Bay of Bengal), depths beyond 50 m or 75 m are physically non-existent because the seafloor is reached.
   - In `ocean_mask_3d`, `mask_3d[d, lat, lon]` is False below the seafloor.
   - If a function inspects `mask_3d` at depth 500 m, the cell appears "masked out". If that mask is passed to a routine expecting a geographic land mask, the entire ocean point could be misidentified as land.
2. **Missing Daily Target Data vs. Seafloor**:
   - If a specific GLORYS assimilation grid point is unobserved or masked on day $t$ due to reanalysis boundary artifacts, downstream consumers checking `~np.isnan(thetao[t, d, y, x])` would interpret it as sub-seabed bathymetry rather than an observational missingness gap.
3. **Surface Predictor Missingness vs. Land**:
   - If OSTIA, CCMP, or OSCAR has a missing pixel in open water (e.g., severe squall line, high sea state, or satellite swath boundary), regridding with `~dst_mask` sets that point to NaN.
   - In downstream models or tabular flatteners, any point with NaN can be discarded or misclassified as land unless `geographic_ocean_mask` and `surface_validity_mask` are explicitly decoupled.

---

## E. Potential Scientific Leakage or Incorrect Masking

1. **`preprocessing/build_dataset.py` — Line 138: Conversion of Target NaNs to Zero**:
   - In `OceanReconstructionDataset.__getitem__()`:
     ```python
     x_clean = np.nan_to_num(x_trans, nan=0.0)
     y_clean = np.nan_to_num(y_raw, nan=0.0)  # <-- CRITICAL SCIENTIFIC RISK
     ```
   - Target $\theta_o$ NaNs (representing sub-seabed or unobserved points) were replaced with `0.0`.
   - In oceanography, 0.0 °C is a **real, physically plausible temperature** (e.g., polar deep water or cold thermoclines). If any downstream loss or evaluation failed to strictly apply `mask`, the neural network would be trained to predict 0.0 °C below the seabed!
2. **`models/05_pointwise_mlp.py` — Masked MSE Loss Formulation**:
   - In `masked_mse_loss()`:
     ```python
     diff = (y_pred - y_true) * mask.float()
     loss = (diff ** 2).sum() / (mask.float().sum() + 1e-8)
     ```
   - In IEEE-754 floating-point math: `(y_pred - NaN) * 0.0 = NaN * 0.0 = NaN`.
   - If `y_true` retains NaNs under IEEE-754 without `torch.where` or boolean indexing (`y_pred[mask] - y_true[mask]`), the loss produces `NaN` gradients!
3. **Tabular Feature Leakage / Discarding**:
   - In `preprocessing/tabular_dataset.py`, `spatial_valid = target_mask.any(axis=-1)`.
   - Points are retained if any depth is valid, but surface features with NaNs are normalized via `(X - mean) / std`, yielding `NaN` features that break PyTorch feedforward layers without warning.

---

## F. Exact Files and Functions Requiring Modification

1. **`preprocessing/ocean_mask.py`**:
   - Deprecate ad-hoc single-day inference.
   - Implement `build_explicit_masks()` generating:
     - `geographic_ocean_mask`: $(101, 241)$ boolean (invariant geographic coastline).
     - `depth_valid_mask`: $(15, 101, 241)$ boolean (bathymetric column limit).
     - `deepest_valid_depth_m`: $(101, 241)$ float32 (deepest ocean depth in meters).
   - Update `save_canonical_ocean_mask()` and `load_canonical_ocean_mask()`.

2. **`preprocessing/build_dataset.py`**:
   - Update `assemble_ml_dataset()`:
     - Generate and store `surface_validity_mask`: $(T, 101, 241, 7)$ boolean.
     - Generate and store `target_validity_mask`: $(T, 15, 101, 241)$ boolean.
     - Store `geographic_ocean_mask` and `depth_valid_mask` inside both Zarr stores and NetCDFs.
     - Remove `nan_to_num(..., nan=0.0)` for target $\theta_o$.
   - Update `OceanReconstructionDataset`:
     - Eliminate `y_clean = np.nan_to_num(y_raw, nan=0.0)`.
     - Strictly use `target_validity_mask` for loss computation.

3. **`models/05_pointwise_mlp.py` and `models/base.py`**:
   - Harden `masked_mse_loss()` to use boolean indexing `y_pred[mask] - y_true[mask]` so target NaNs can never contaminate gradients.

4. **`scripts/harmonize_and_validate.py`**:
   - Integrate multi-mask generation and validation.
   - Extend Check 3 and QA reporting to inspect all four masks across entire domain, Arabian Sea, and Bay of Bengal.
   - Update ARGO terminology to "ARGO–GLORYS Reference Consistency Assessment".

5. **`preprocessing/argo_matchup.py`**:
   - Update terminology and reporting from "Independent ARGO Validation" to "ARGO–GLORYS Reference Consistency Assessment".

6. **`tests/test_scientific_masks_and_integrity.py` (New)**:
   - Provide comprehensive regression testing for all 10 invariant properties.
