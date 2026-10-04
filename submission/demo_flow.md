# Interactive Prototype Demonstration Walkthrough (10-Step Video Guide)

**Project Title**: Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature  
**Domain**: North Indian Ocean (5°N–30°N, 45°E–105°E)  
**Live Production URL**: [https://code-gules-three.vercel.app](https://code-gules-three.vercel.app)  
**Target Video Duration**: 3 – 5 minutes  
**Format**: Screen recording + live voiceover  

---

## Pre-Recording Checklist

1. Open Google Chrome or Firefox in full screen (1920 × 1080 resolution recommended).
2. Navigate to [https://code-gules-three.vercel.app](https://code-gules-three.vercel.app).
3. Ensure browser zoom is set to 100%.
4. Verify microphone input levels and close background notification windows.
5. Keep this guide side-by-side or on a second monitor during recording.

---

## Step-by-Step Demonstration Script

### Step 1: Opening & Title Header (0:00 – 0:25)
- **Visual Action**: Display the top hero header and navigation banner of the dashboard.
- **On-Screen Elements**: 
  - Application title: *"Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature"*.
  - Status badges: `"PROTOTYPE • LOCAL SIMULATION"`, `"2020 PRODUCTION DATASET"`, `"97/97 TESTS PASSING"`.
- **Narration Script**:
  > *"Welcome to the demonstration of our research project: 'Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature'. In this video, we present our interactive research prototype evaluating the inversion of 3D ocean temperature fields across the North Indian Ocean using multi-satellite surface observations."*

---

### Step 2: Scientific Context & Motivation (0:25 – 0:50)
- **Visual Action**: Scroll smoothly down to the Domain & Scientific Overview card.
- **On-Screen Elements**:
  - Bounding box indicator: `5.00°N–30.00°N, 45.00°E–105.00°E`.
  - Canonical 0.25° grid specification ($101 \times 241$ cells = 24,341 points).
  - Target depths: 15 vertical levels (0 to 1000 meters).
- **Narration Script**:
  > *"Because seawater is opaque to electromagnetic radiation, satellites can only measure the surface skin layer. Autonomous Argo floats collect accurate vertical soundings, but they are sparsely distributed with a 300-kilometer average spacing and a 10-day sampling interval. Our goal is to reconstruct daily, synoptic 3D temperature profiles across 15 standard depths from 0 to 1000 meters using deep spatiotemporal representation learning."*

---

### Step 3: Input Predictors Inspection (0:50 – 1:15)
- **Visual Action**: Hover over or click into the **Surface Predictors** panel.
- **On-Screen Elements**:
  - 7 Input variables:
    1. **SST** (OSTIA L4 Reprocessed)
    2. **SSS** (Copernicus Multi-Obs)
    3. **SSH / SLA** (DUACS Altimetry)
    4. **Current U** & 5. **Current V** (OSCAR Surface Currents)
    6. **Wind U** & 7. **Wind V** (RSS CCMP V3.1 10m Winds)
- **Narration Script**:
  > *"Our pipeline harmonizes 7 daily satellite and blended observation streams into a canonical 0.25-degree grid. These predictors capture the essential physics: SST provides thermal boundary conditions, SSS governs upper-layer buoyancy, SSH captures baroclinic thermocline tilt, surface currents govern lateral advection, and vector winds drive Ekman pumping and upwelling."*

---

### Step 4: Leakage Controls & Chronological Split (1:15 – 1:40)
- **Visual Action**: Scroll to the **Experimental Protocol & Data Split** diagram/timeline.
- **On-Screen Elements**:
  - Full-Year 2020 timeline.
  - Train: Days 0–252 (Jan 1 – Sep 9, 253 days).
  - Purge Buffer 1: Days 253–258 (Sep 10 – Sep 15, 6 days).
  - Validation: Days 259–306 (Sep 16 – Nov 2, 48 days).
  - Purge Buffer 2: Days 307–312 (Nov 3 – Nov 8, 6 days).
  - Test: Days 313–365 (Nov 9 – Dec 31, 53 days).
- **Narration Script**:
  > *"A critical priority in this work is scientific rigor and zero data leakage. We enforce a strict chronological partition of the 2020 calendar year. Crucially, we insert 6-day temporal purge buffers between splits. Because our B8 model uses a 5-day causal receptive field, a 6-day buffer mathematically guarantees that no test prediction can access historical or future information from the training or validation sets. Furthermore, all z-score normalizations were fitted exclusively on training days."*

---

### Step 5: Master Benchmark Comparison (1:40 – 2:15)
- **Visual Action**: Click on the **Model Benchmarks** tab or scroll to the Leaderboard Table.
- **On-Screen Elements**:
  - Complete 10-model benchmark hierarchy:
    - B0 (Day-0 Persistence): 1.5220 °C | B0b (Day-252 Persistence): 1.7287 °C | B1 (Climatology): 1.2582 °C
    - B2 (Ridge, alpha=100,000): 1.0295 °C | B3: 1.0452 °C | B4: 1.0288 °C
    - B5: 1.5524 °C | B6: 1.2702 °C | B7: 1.5320 °C
    - **B8**: **0.9800 °C** (Highlighted in emerald green)
  - Label: `"B8 — Best-performing architecture among evaluated internal benchmarks"`.
- **Narration Script**:
  > *"Here is our master benchmark table evaluated strictly on the 53-day test partition. Our proposed B8 Spatiotemporal Embedding Network achieves an overall column-averaged test RMSE of 0.9800 °C. This represents a 22.11% improvement over daily climatology (B1) and a 4.74% improvement over the best tabular machine learning baseline, LightGBM (B4). A 1000-resample 7-day block bootstrap confirms this improvement is statistically significant with p < 0.001."*

---

### Step 6: B8 Architecture & Latent Space (2:15 – 2:40)
- **Visual Action**: Click on or highlight the **Model Architecture** view.
- **On-Screen Elements**:
  - Diagram showing:
    - Input: $5\text{-day} \times 3 \times 3$ patch $\times 7$ channels.
    - Spatial 2D CNN Encoder ($7 \to 32 \to 64$, BatchNorm, ReLU, AdaptiveAvgPool2d, Flatten).
    - 2-Layer Temporal GRU (hidden dimension = 128).
    - Latent Bottleneck: LayerNorm(128) $\to$ 128-D Ocean Latent Embedding.
    - Column Decoder: Linear(128→64) + ReLU $\to$ Linear(64→15) depths.
  - Total parameter count: **203,791 parameters** (~0.81 MB).
- **Narration Script**:
  > *"The B8 architecture consists of three modular components: First, a 2D CNN encoder extracts 64-dimensional spatial features from 3x3 local patches. Second, a 2-layer GRU captures temporal dynamics across the 5-day window. After a 128-dimensional LayerNorm bottleneck, a lightweight decoder projects this embedding into the 15 discrete vertical temperature levels. The model has only 203,791 parameters, representing a lightweight architecture suitable for efficient inference."*

---

### Step 7: Interactive Depth Exploration (2:40 – 3:10)
- **Visual Action**: Move the **Depth Slider** continuously from **0 m down to 1000 m**. Stop briefly at **75 m** and **1000 m**.
- **On-Screen Elements**:
  - Depth slider moving through: 0m, 20m, 50m, 75m, 150m, 300m, 500m, 1000m.
  - Horizontal spatial heatmap updating dynamically.
  - RMSE metric indicator updating with depth-specific certified values:
    - 0 m: `0.4369 °C`
    - 75 m: `1.8110 °C` (Thermocline Peak)
    - 1000 m: `0.5959 °C`
- **Narration Script**:
  > *"Now let's interact with the depth exploration slider. Notice that at the surface (0 to 10 meters), RMSE is approximately 0.44 °C because satellite SST provides a strong thermal constraint. As we descend into the seasonal thermocline at 50 to 125 meters, error peaks at 1.8110 °C at 75 meters. This peak reflects steep vertical gradients and internal wave variability. Below 500 meters, error drops below 0.70 °C as the ocean enters the quiescent, thermally stable mesopelagic zone. This demonstrates that 0.9800 °C is a column average, not a uniform vertical error."*

---

### Step 8: Regional Basin Breakdown (3:10 – 3:35)
- **Visual Action**: Click the **Regional Filters**: Switch from **Full Domain** to **Arabian Sea**, then to **Bay of Bengal**.
- **On-Screen Elements**:
  - Full Domain: `0.9642 °C`
  - Arabian Sea: `1.0907 °C`
  - Bay of Bengal: `0.6775 °C`
- **Narration Script**:
  > *"Using our regional selector, we examine basin-scale hydrodynamic differences. The Full Domain cosine-latitude weighted RMSE is 0.9642 °C. The Arabian Sea has a higher RMSE of 1.0907 °C due to intense winter evaporative cooling, deep convective mixing, and energetic western boundary eddies. Conversely, the Bay of Bengal exhibits a much lower RMSE of 0.6775 °C, stabilized by massive freshwater river discharge forming a buoyant barrier layer that shields upper-layer thermal profiles."*

---

### Step 9: Vertical Sounding CTD Profile (3:35 – 4:00)
- **Visual Action**: Navigate to the **Vertical Profile / Sounding View**.
- **On-Screen Elements**:
  - Interactive sounding plot showing Depth (y-axis, inverted 0 to 1000 m) vs Temperature (°C, x-axis).
  - Blue line: Reference Reanalysis State (GLORYS).
  - Emerald dashed line: B8 Model Prediction.
  - Red dotted line: B1 Climatology Baseline.
- **Narration Script**:
  > *"Here in the vertical sounding view, we compare the reconstructed temperature profile against the GLORYS reference profile. Notice how the B8 model accurately reconstructs the mixed layer depth and the sharp curvature of the thermocline, whereas climatology flattens and blurs these dynamic features. This profile fidelity is crucial for acoustic transmission and mixed-layer heat budget calculations."*

---

### Step 10: Scientific Disclosures & Conclusion (4:00 – 4:30)
- **Visual Action**: Scroll to the footer card / Limitations disclosure section.
- **On-Screen Elements**:
  - Disclosures:
    - GLORYS Reanalysis Reference Target (numerical simulation + data assimilation).
    - Single-year scope (2020) and decadal scaling future work.
    - Verified test suite: 97/97 tests passing.
    - Vercel production deployment and GitHub links.
- **Narration Script**:
  > *"In adherence to scientific integrity, we emphasize three important disclosures: First, our training target is the GLORYS reanalysis, which is a reference assimilation product rather than unassimilated in-situ ground truth. Second, low deep-water RMSE reflects low physical thermal variance at depth. Third, our model was evaluated on the 2020 annual cycle. In conclusion, the B8 Spatiotemporal Embedding Network establishes that joint spatial convolutions and causal temporal memory provide decisive improvements for vertical ocean temperature reconstruction. Thank you for watching!"*

---

## Technical Information for Video Description / Submission Notes

```markdown
**Project Title**: Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature
**Live Application**: https://code-gules-three.vercel.app
**Source Repository**: https://github.com/neeravjain91-jpg/-Satellite-Embedding
**Key Results**:
- Model: B8 Spatiotemporal Embedding Network (203,791 parameters)
- Test RMSE: 0.9800 °C (Column-Averaged, Days 313–365)
- Climatology Gain: +22.11% over B1 (1.2582 °C)
- Tabular ML Gain: +4.74% over B4 LightGBM (1.0288 °C)
- Test Suite: 97/97 passing automated unit and integration tests
- Protocol: 6-day purge buffers, train-only normalization, strict chronological split
```
