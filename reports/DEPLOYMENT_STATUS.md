# Vercel Production Deployment Status

**Date**: 2026-10-09T00:04:14+05:30  
**Project**: Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction  
**Environment**: Production (Vercel Cloud)

---

## 1. Deployment Summary

| Parameter | Value | Verification Status |
| :--- | :--- | :--- |
| **Git Commit SHA** | `88f7dbc99cae223a2a66539cddbb5073ca4360c1` | Verified (Latest `main`, supersedes `66652ff` and `4c02fb2`) |
| **Git Branch** | `main` | Verified (`git branch --show-current` = `main`, clean working tree) |
| **Vercel Project Name** | `code` | Verified (Existing project under `neeravjain91-6032s-projects`) |
| **Production URL** | `https://code-gules-three.vercel.app` | Verified (HTTP 200 OK) |
| **Deployment URL** | `https://code-jry5vs2ze-neeravjain91-6032s-projects.vercel.app` | Verified (HTTP 200 OK) |
| **Deployment ID** | `dpl_8rqrbnpzFUPRAJRGCWfBM7BvJjrM` | Verified (`readyState: READY`, `target: production`) |
| **Vercel Inspector URL**| `https://vercel.com/neeravjain91-6032s-projects/code/8rqrbnpzFUPRAJRGCWfBM7BvJjrM` | Verified |
| **HTTP Status** | `200 OK` | Verified (Publicly accessible) |
| **Build Status** | `SUCCESS` | Verified (`tsc && vite build` built in 6.59s, 0 errors) |
| **Frontend JS Bundle** | `dist/assets/index-qmrJetGv.js` (287.25 kB) | Verified (Matches live HTML `<script>` tag) |
| **Frontend CSS Bundle** | `dist/assets/index-DpXea7zc.css` (35.89 kB) | Verified (Matches live HTML `<link>` tag) |

---

## 2. Commit Verification & Provenance

- **Old Commit**: `66652ff` (Built initial upgraded map; bundle was `index-DAryFtT-.js`).
- **Intermediate Submission Commits**:
  - `bb578f3`: repair and harmonize thermal-regime confusion matrix and benchmark hierarchy
  - `66f49bb`: harden scientific claims, remove proof language, and clarify ARGO and B3 provenance
  - `6a13f30`: harmonize B1 confusion matrix table and add strict numerical integrity test
  - `7532359`: complete Phase 2 end-to-end scientific and engineering audit
  - `e51c8e3`: finalize academic project submission package
  - `4c02fb2`: align canonical B6, B7, B8 architecture descriptions and audit statistical claims
- **Deployed Current Head**: `88f7dbc` (`docs(ui): explicitly designate GLORYS numerical ocean reanalysis reference and ARGO-GLORYS Reference Consistency Assessment`).
- **Commit Status**: Latest intended academic submission commit successfully deployed to Vercel production.

---

## 3. UI Factual Consistency & Scientific Data Audit

All frontend displays were audited and confirmed compliant with the authoritative project specification:

### A. Certified Benchmark Hierarchy (Full Test Partition)
- **B0 (Day-0 Persistence)**: `1.5220 °C`
- **B0b (Day-252 Persistence)**: `1.7287 °C`
- **B1 (Spatial-Depth Climatology Reference)**: `1.2582 °C`
- **B2 (Multi-Output Ridge, $\alpha=100{,}000$)**: `1.0295 °C` (120 coefficients)
- **B3 (Multi-Depth Random Forest)**: `1.0452 °C` (13,289,966 decision nodes across 750 Random Forest trees)
- **B4 (Gradient Boosting / LightGBM)**: `1.0288 °C` (750 boosted trees)
- **B5 (Pointwise MLP, 128-128-64)**: `1.5524 °C` (26,767 trainable parameters)
- **B6 (Spatial CNN, 3×3 Patch)**: `1.2702 °C` (30,991 trainable parameters)
- **B7 (Temporal GRU, 5-Day Causal)**: `1.5320 °C` (44,111 trainable parameters)
- **B8 (Spatiotemporal Embedding Model)**: `0.9800 °C` (+22.11% vs B1, 203,791 trainable parameters)

### B. Canonical Architecture Context
- **B6**: Strictly designated as `3×3 spatial patch` (no 5×5 claims).
- **B7**: Strictly designated as `T=5 causal temporal history` (no 7-day claims).
- **B8**: Strictly designated as `T=5 × 3×3 spatiotemporal input` (no 5×5 or 7-day claims).

### C. Reference Terminology & Disclosures
- **SSS Input**: Designated as `Copernicus Multi-Observation SSS`.
- **Target Reference**: Designated as `GLORYS numerical ocean reanalysis reference` (explicitly disclosing that GLORYS is a numerical simulation integrating satellite/in-situ observations via data assimilation, not direct observational ground truth).
- **In-Situ Assessment**: Designated as `ARGO–GLORYS Reference Consistency Assessment` (explicitly disclosing that because operational ARGO float profiles are assimilated into GLORYS, this comparison evaluates reanalysis reference consistency rather than serving as independent validation of the ML model).
- **Prototype Mode**: Badged as `PROTOTYPE • LOCAL SIMULATION` across navigation headers, depth profile views, and exploration cards.
- **Scientific Honesty**: Zero unsubstantiated "State-of-the-Art" or "SOTA" claims across all components.

---

## 4. Route & Feature Verification

The deployed application was verified across all 8 functional views:

1. **Dashboard (`/` / `dashboard`)**: Basin overview, operational KPIs, quick B0–B8 benchmark leaderboard, and interactive target coordinates picker.
2. **Ocean Explorer (`explorer`)**: 7 surface variables (SST, SSS, SSH, CurU, CurV, WindU, WindV) with 7-day sparkline trends, status tags, and spatial coordinates controls.
3. **Reconstruction Flow (`reconstruction`)**: Interactive 8-stage inference execution simulator through the B8 Conv2D → GRU → Latent Bottleneck → Depth Decoder pipeline.
4. **Vertical Depth Profile (`depth-profile`)**: 15 canonical depths (0 m to 1000 m) reconstructed temperature curve vs. GLORYS reference and Climatology with tabular inspection.
5. **Embedding / Latent Space (`embedding`)**: 2D PCA/t-SNE latent manifold visualization of 128-D bottleneck representation vectors clustered by oceanographic regime.
6. **Benchmark Matrix (`benchmarks`)**: Complete B0 through B8 comparative matrix with column-averaged test RMSE, parameter counts, and relative error reductions.
7. **Validation & Error (`validation`)**: B8 exact depth-wise RMSE curve (0.4369 °C at surface to 1.8110 °C at 75 m thermocline), cosine-weighted regional basin metrics, and seasonal partition validation.
8. **Methodology (`methodology`)**: End-to-end mathematical methodology, causal sliding window specification, 6-day purge buffers, train-only normalization, and ARGO–GLORYS reference consistency disclosures.

### North Indian Ocean Map Verification
- **Study Domain**: $5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$ bounding rectangle with corner coordinate badges.
- **Graticule & Mesh**: Major $5^\circ$ grid lines with numeric callouts, and toggleable $0.25^\circ$ canonical research mesh ($24,341$ points).
- **Bathymetry**: Continental shelf ($0–200\text{ m}$) ribbon and subsea ridges (Carlsberg Ridge, Chagos-Laccadive Ridge, Ninety East Ridge, Murray Ridge, Sunda Trench).
- **Surface Overlays**: Interactive toggles for Bathymetry, SST, SSS, SSH, and B8 Subsurface T(z) with dynamic scalar colorbar legend.
- **Interaction**: Mouse wheel and button zoom ($1.0\times$ to $4.0\times$), click-and-drag pan, Focus Target, and Reset View.
- **Target Reticle**: Sonar radar ping animation with crosshairs and coordinate badge.
- **Land Protection**: Point-in-polygon ray casting check flags land clicks without updating ocean targets, displaying informative warning toast.
- **HUD Readout**: Live cursor tracking showing latitude, longitude, estimated bathymetric depth, and basin name.
