# Final Limitations & Scientific Scope Document

**Project**: Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature  
**Status**: Official Scientific Submission Disclosure  

---

## 1. Reference Target vs In-Situ Ground Truth
- **GLORYS12V1 Nature**: The target variable used for model training and evaluation is potential temperature ($\theta_o$) from Copernicus Marine GLORYS12V1 global ocean reanalysis. GLORYS12V1 is a numerical physical state estimate produced by assimilating satellite and in-situ observations into the NEMO physical ocean model under hydrostatic and Boussinesq approximations.
- **Distinction from Raw Ground Truth**: GLORYS is a high-fidelity continuous reference reanalysis field, not raw in-situ ground truth. It inherently carries the structural assumptions, sub-grid parameterizations, and atmospheric forcing uncertainties of the NEMO numerical model.
- **Scientific Implication**: Performance reported in this benchmark reflects the model's capacity to emulate and reconstruct the GLORYS reanalysis physical manifold from surface satellite observations.

---

## 2. In-Situ Argo Profiling Float Validation Tier
- **Role of Argo**: Autonomous Argo profiling floats provide direct in-situ physical measurements of temperature and salinity profiles. However, because GLORYS12V1 assimilates quality-controlled Argo profiles during its operational cycle, Argo profiles within the 2020 domain are not strictly independent of GLORYS.
- **Dedicated Independent Validation**: Establishing a completely independent physical validation requires either evaluating against unassimilated delayed-mode experimental floats or comparing directly against glider/moorings transects. This represents a distinct physical validation tier outside the current benchmark scope.

---

## 3. Depth-Dependent Error Variance (Non-Uniform Performance)
- **Column-Averaged Metric Clarification**: The headline benchmark metric of **0.9800 °C** is an unweighted column-averaged mean across 15 canonical depth levels from 0 m to 1000 m. It must never be interpreted as a uniform error across all depths.
- **Thermocline Error Peak**: Due to steep vertical thermal gradients (up to $0.15\ ^\circ\text{C/m}$) in the main ocean pycnocline/thermocline ($50\text{ m} - 125\text{ m}$), small vertical displacement errors in the predicted isothermal layer result in localized RMSE values of up to **1.8110 °C** at 75 m.
- **Deep Ocean Low-Variance Regime**: In deep waters ($300\text{ m} - 1000\text{ m}$), natural seasonal and synoptic temperature variability is very small ($\sigma < 0.4\ ^\circ\text{C}$). The static climatological mean (B1) acts as an exceptionally strong low-variance baseline in this regime, whereas neural regression exhibits a slight residual offset ($0.59\ ^\circ\text{C}$ vs $0.37\ ^\circ\text{C}$).

---

## 4. Single-Year Benchmark Period (Temporal Scope)
- **2020 Partition**: The benchmark is trained and evaluated exclusively on the full calendar year 2020 (366 days).
- **Interannual Climate Modes**: The Indian Ocean undergoes pronounced interannual and decadal variability driven by the Indian Ocean Dipole (IOD) and El Niño–Southern Oscillation (ENSO) teleconnections. A model trained on a single year cannot capture interannual regime shifts (e.g., extreme positive IOD events as observed in 2019). Multi-year decadal evaluation across 2010–2022 is required before operational forecasting deployment.

---

## 5. Spatial Domain Boundary Effects
- **Geographic Bounds**: The domain is restricted to the North Indian Ocean ($5^\circ\text{N} - 30^\circ\text{N}$, $45^\circ\text{E} - 105^\circ\text{E}$).
- **Cross-Equatorial Dynamic Truncation**: Ocean dynamics south of 5°N (such as the Southern Hemisphere equatorial gyre and deep cross-equatorial overturning cells) are outside the model's spatial boundary, preventing full hemispheric wave tracking.

---

## 6. Frontend Prototype Operational Mode
- **Standalone Local Prototype**: The companion interactive user interface (running on Vercel and local dev server) is a frontend-only interactive prototype designed for system demonstration and scientific presentation.
- **Local Simulation**: All interactive profile renderings, map queries, and coordinate inspections use deterministic local physics-based simulation algorithms. The frontend is not connected to a live multi-GPU Python inference backend.
