# Publication-Quality Architectural & Workflow Diagrams
## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature

---

### Diagram 1: End-to-End Scientific System Pipeline
```mermaid
flowchart TD
    subgraph S1["1. Surface Satellite Remote Sensing"]
        D1["OSTIA SST (°C)"]
        D2["Copernicus SSS (PSU)"]
        D3["DUACS SSH / SLA (m)"]
        D4["OSCAR Surface Current U / V (m/s)"]
        D5["CCMP Vector Wind U / V (m/s)"]
    end

    subgraph S2["2. Data Harmonization & Quality Control"]
        H1["Spatial Subsetting (5°N–30°N, 45°E–105°E)"]
        H2["Normalized Bilinear Regridding (0.25° Grid)"]
        H3["Temporal Harmonization (Full-Year 2020, 366 Days)"]
        H4["Four-Way Masking: Geo ∧ Surf ∧ Target ∧ Bathymetry"]
    end

    subgraph S3["3. Chronological Partitioning & Leakage Controls"]
        P1["Train: Days 0–252 (253d, N=2,871,550)"]
        P2["Purge 1: Days 253–258 (6d, Discarded)"]
        P3["Val: Days 259–306 (48d, N=544,800)"]
        P4["Purge 2: Days 307–312 (6d, Discarded)"]
        P5["Test: Days 313–365 (53d, N=601,550)"]
        P6["Train-Only Z-Score Normalization (Zero Test Contamination)"]
    end

    subgraph S4["4. Machine Learning & Deep Representation Learning"]
        M0["B0 / B0b / B1: Physical & Climatological Baselines"]
        M1["B2: Multi-Output Ridge Regression (alpha=100,000)"]
        M2["B3 / B4: Tree Ensembles (Random Forest / LightGBM)"]
        M3["B5 / B6 / B7: Pointwise / Spatial / Temporal Ablations"]
        M4["B8: Spatiotemporal Embedding Champion Network"]
    end

    subgraph S5["5. Reanalysis Target Reference"]
        T1["GLORYS12V1 Potential Temperature θ_o"]
        T2["15 Canonical Depths: 0, 5, 10, ..., 1000 m (PCHIP Spline)"]
        T3["Strict NaN Seafloor Bathymetric Preservation (Invalid NaNs Never Converted to Zero)"]
    end

    subgraph S6["6. Multi-Tier Evaluation Engine"]
        E1["Continuous Test RMSE (B8 = 0.9800 °C, +22.11% vs B1)"]
        E2["Paired 7-Day Block Bootstrap (95% CI [-0.3957, -0.1756], p<0.001)"]
        E3["Thermal-Regime Diagnostic Classification (6 Regimes, N=8,017,734)"]
        E4["ARGO–GLORYS Reference Consistency Assessment (N=1,482 Matchups)"]
    end

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S6
    S5 --> S6
```

---

### Diagram 2: Data-Source Integration & Preprocessing Flow
```mermaid
flowchart LR
    subgraph RAW["Raw Satellite & Reanalysis Products"]
        R1["UK Met Office OSTIA L4 SST\n(0.05° Native)"]
        R2["Copernicus MULTIOBS SSS\n(0.25° Native)"]
        R3["CMEMS DUACS SSH / SLA\n(0.25° Native)"]
        R4["NOAA OSCAR Currents U/V\n(0.25° Native)"]
        R5["RSS CCMP V3.1 Winds U/V\n(0.25° Native, 6-hr)"]
        R6["CMEMS GLORYS12V1 θ_o Target\n(1/12° Native, 50 Levels)"]
    end

    subgraph REGRID["Regridding & Interpolation"]
        G1["Normalized Bilinear Coastal Regridding\n(No terrestrial zero bleeding)"]
        G2["Daily Aggregation & Time Synchronization\n(00:00:00 UTC)"]
        G3["PCHIP Vertical Interpolation\n(15 Canonical Depths, Monotonic)"]
    end

    subgraph CANON["Canonical Zarr Storage"]
        Z1["Surface Zarr Tensor\nShape: [366, 101, 241, 7]"]
        Z2["Target Zarr Tensor\nShape: [366, 15, 101, 241]"]
        Z3["GLORYS/ORCA12 Bathymetry Mask\n(Preserve NaNs below seabed; Never Converted to Zero)"]
    end

    R1 --> G1
    R2 --> G1
    R3 --> G1
    R4 --> G1
    R5 --> G2
    G1 --> Z1
    G2 --> Z1
    R6 --> G3
    G3 --> Z2
    Z3 -.-> Z2
```

---

### Diagram 3: Chronological Split & Purge Buffer Architecture
```mermaid
flowchart TD
    subgraph CALENDAR["Full-Year 2020 Calendar (366 Days, Leap Year)"]
        D0["Day 0\n2020-01-01"]
        D252["Day 252\n2020-09-09"]
        D253["Day 253\n2020-09-10"]
        D258["Day 258\n2020-09-15"]
        D259["Day 259\n2020-09-16"]
        D306["Day 306\n2020-11-02"]
        D307["Day 307\n2020-11-03"]
        D312["Day 312\n2020-11-08"]
        D313["Day 313\n2020-11-09"]
        D365["Day 365\n2020-12-31"]
    end

    subgraph PARTITIONS["Experimental Partitioning"]
        TR["TRAIN SPLIT\nDays 0–252 (253 Days / 69.1%)\nN = 2,871,550 Columns\nFits All Z-Score Normalization"]
        P1["PURGE BUFFER 1\nDays 253–258 (6 Days)\nCompletely Discarded\nBreaks Ocean Decorrelation Memory"]
        VA["VALIDATION SPLIT\nDays 259–306 (48 Days / 13.1%)\nN = 544,800 Columns\nHyperparameter Tuning & Early Stopping"]
        P2["PURGE BUFFER 2\nDays 307–312 (6 Days)\nCompletely Discarded\nProtects Causal T=5 Windows"]
        TE["TEST SPLIT (FROZEN)\nDays 313–365 (53 Days / 14.5%)\nN = 601,550 Columns\n8,017,734 Valid Depth Evaluations"]
    end

    D0 --- D252 --> TR
    D253 --- D258 --> P1
    D259 --- D306 --> VA
    D307 --- D312 --> P2
    D313 --- D365 --> TE
```

---

### Diagram 4: Ten-Model Benchmark Hierarchy (B0 to B8)
```mermaid
flowchart TD
    subgraph LVL0["Tier 0: Physical & Climatological Reference Baselines"]
        B0["B0: Day-0 Persistence\n(GLORYS Day 0 Frozen; RMSE = 1.5220 °C)"]
        B0b["B0b: Day-252 Persistence\n(Train Boundary Frozen; RMSE = 1.7287 °C)"]
        B1["B1: Spatial-Depth Climatology (Reference Anchor)\n(Train Historical Mean Profile; RMSE = 1.2582 °C)"]
    end

    subgraph LVL1["Tier 1: Tabular Machine Learning (Pointwise 7-Surface)"]
        B2["B2: Multi-Output Ridge Regression\n(alpha = 100,000; 120 Coeffs; RMSE = 1.0295 °C)"]
        B3["B3: Multi-Depth Random Forest\n(750 Trees, 13.3M Nodes; RMSE = 1.0452 °C)"]
        B4["B4: LightGBM Gradient Boosting\n(750 Boosting Trees; RMSE = 1.0288 °C)"]
    end

    subgraph LVL2["Tier 2: Deep Learning Context Ablations"]
        B5["B5: Pointwise MLP\n(Linear 128-128-64; 26,767 Weights; RMSE = 1.5524 °C)"]
        B6["B6: Spatial CNN (3x3 Patch Only)\n(Time-Distributed Conv2D; 30,991 Weights; RMSE = 1.2702 °C)"]
        B7["B7: Temporal GRU (5-Day Causal Only)\n(2-Layer GRU; 44,111 Weights; RMSE = 1.5320 °C)"]
    end

    subgraph LVL3["Tier 3: Spatiotemporal Embedding Champion"]
        B8["B8: Spatiotemporal Embedding Network\n(Conv2D + 2L-GRU + 128D Bottleneck; 203,791 Weights)\nTest RMSE = 0.9800 °C (+22.11% vs B1, p < 0.001)"]
    end

    B0 --> B1
    B0b --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B1 --> B5
    B5 --> B6
    B5 --> B7
    B6 --> B8
    B7 --> B8
```

---

### Diagram 5: Champion B8 Architecture Detailed Topology
```mermaid
flowchart TD
    subgraph IN["Input Context Representation"]
        X["Spatiotemporal Surface Cube\nShape: [B, T=5, C=7, P=3, P=3]\n(~75 km × 75 km area across 5 backward days)"]
    end

    subgraph ENC["Time-Distributed 2D CNN Spatial Encoder"]
        C1["Conv2D(7 → 32, kernel=3, padding=1)\nBatchNorm2d + ReLU"]
        C2["Conv2D(32 → 64, kernel=3, padding=1)\nBatchNorm2d + ReLU"]
        AP["AdaptiveAvgPool2d((1, 1))"]
        TOK["Spatial Token Sequence\nShape: [B, T=5, 64]"]
    end

    subgraph REC["Causal Recurrent Sequence Model"]
        G1["2-Layer Unidirectional GRU\n(input_size=64, hidden_size=128, dropout=0.1)"]
        HT["Terminal Recurrent Hidden State h_T\nShape: [B, 128]"]
    end

    subgraph BOT["Ocean State Latent Bottleneck"]
        LN["LayerNorm(128)\nStabilizes Latent Space Variance"]
        Z["Compressed Ocean State Vector z\nShape: [B, 128]"]
    end

    subgraph DEC["Subsurface Depth Decoder"]
        L1["Linear(128 → 64) + ReLU"]
        L2["Linear(64 → 15)"]
        OUT["Reconstructed Potential Temperature Profile \hat{Y}\nShape: [B, 15] (0 m to 1000 m)"]
    end

    X --> C1 --> C2 --> AP --> TOK
    TOK --> G1 --> HT
    HT --> LN --> Z
    Z --> L1 --> L2 --> OUT
```

---

### Diagram 6: Comprehensive Evaluation & Diagnostic Workflow
```mermaid
flowchart LR
    subgraph PRED["Model Predictions"]
        Y_HAT["Continuous Predictions \hat{Y}\nShape: [N=601,550, 15]"]
    end

    subgraph CONT["1. Primary Continuous Regression Metrics"]
        R1["Column-Averaged RMSE & MAE\n(Across 15 Canonical Depths)"]
        R2["Depth-Stratified RMSE Breakdown\n(0 m, 5 m, ..., 1000 m)"]
        R3["Regional Cosine-Weighted Subsets\n(Arabian Sea, Bay of Bengal, Equatorial)"]
        R4["Paired 7-Day Moving Block Bootstrap\n(B=1,000 Resamples, 95% Confidence Intervals)"]
    end

    subgraph DIAG["2. Secondary Thermal-Regime Diagnostic"]
        D1["Discretize into 6 Thermal Regimes\n(<10°, 10–15°, 15–20°, 20–25°, 25–28°, ≥28°C)"]
        D2["Confusion Matrix Computation\n(N = 8,017,734 Valid Depth Points)"]
        D3["Metrics: Accuracy, Within ±1 Bin, Cohen's κ, Macro F1"]
        D4["Scientific Guardrail: High ±1-bin containment does not\nprove vertical profile monotonicity or rule out gradient inversions"]
    end

    subgraph IN_SITU["3. In-Situ Observational Consistency"]
        A1["N = 1,482 Collocated In-Situ Argo Float Profiles"]
        A2["ARGO–GLORYS Reference Consistency Assessment\n(Evaluates reanalysis reference agreement; not direct ML validation)"]
    end

    Y_HAT --> CONT
    Y_HAT --> DIAG
    PRED -.-> IN_SITU
```
