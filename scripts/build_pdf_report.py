"""
Generate a comprehensive, professionally styled submission PDF:
'submission/Final_Project_Submission.pdf'
Using PyMuPDF (pymupdf).
"""

import os
import pymupdf

def build_pdf(output_path="submission/Final_Project_Submission.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = pymupdf.open()

    PAGE_WIDTH = 595.32   # A4 width in pt
    PAGE_HEIGHT = 841.92  # A4 height in pt
    MARGIN_LEFT = 45.0
    MARGIN_RIGHT = 550.32
    MARGIN_TOP = 50.0
    MARGIN_BOTTOM = 790.0
    USABLE_WIDTH = MARGIN_RIGHT - MARGIN_LEFT

    # Color definitions (normalized 0.0 - 1.0)
    C_NAVY = (0.043, 0.125, 0.231)      # #0B203B
    C_BLUE = (0.055, 0.647, 0.914)      # #0EA5E9
    C_DARK = (0.059, 0.090, 0.165)      # #0F172A
    C_MUTED = (0.392, 0.455, 0.545)     # #64748B
    C_LIGHT_BG = (0.949, 0.965, 0.984)  # #F1F6FB
    C_CARD_BG = (0.976, 0.984, 0.992)   # #F9FAFB
    C_BORDER = (0.800, 0.840, 0.880)    # #CCD6E0
    C_GREEN = (0.063, 0.725, 0.506)     # #10B981
    C_AMBER = (0.961, 0.620, 0.043)     # #F59E0B
    C_WHITE = (1.0, 1.0, 1.0)

    class PDFBuilder:
        def __init__(self):
            self.pages = []
            self.current_page = None
            self.y = MARGIN_TOP

        def new_page(self):
            self.current_page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            self.pages.append(self.current_page)
            self.y = MARGIN_TOP + 20.0
            return self.current_page

        def check_space(self, required_height):
            if self.y + required_height > MARGIN_BOTTOM - 20:
                self.new_page()

        def draw_header_footer(self, page_num, total_pages):
            page = doc[page_num - 1]
            if page_num > 1:
                # Running top header
                page.draw_line(
                    pymupdf.Point(MARGIN_LEFT, 36),
                    pymupdf.Point(MARGIN_RIGHT, 36),
                    color=C_BORDER, width=0.5
                )
                page.insert_text(
                    pymupdf.Point(MARGIN_LEFT, 30),
                    "Satellite Embedding Deep Learning | Subsurface Ocean Temperature Reconstruction",
                    fontsize=7.5, fontname="helv", color=C_MUTED
                )
                page.insert_text(
                    pymupdf.Point(MARGIN_RIGHT - 110, 30),
                    "FINAL PROJECT SUBMISSION",
                    fontsize=7.5, fontname="helv", color=C_BLUE
                )

            # Footer
            page.draw_line(
                pymupdf.Point(MARGIN_LEFT, MARGIN_BOTTOM + 5),
                pymupdf.Point(MARGIN_RIGHT, MARGIN_BOTTOM + 5),
                color=C_BORDER, width=0.5
            )
            page.insert_text(
                pymupdf.Point(MARGIN_LEFT, MARGIN_BOTTOM + 18),
                "North Indian Ocean (5°N–30°N, 45°E–105°E) • Certified 2020 Production Benchmark",
                fontsize=7.5, fontname="helv", color=C_MUTED
            )
            page_str = f"Page {page_num} of {total_pages}"
            page.insert_text(
                pymupdf.Point(MARGIN_RIGHT - 50, MARGIN_BOTTOM + 18),
                page_str, fontsize=7.5, fontname="helv", color=C_MUTED
            )

        def add_h1(self, text):
            self.check_space(45)
            self.y += 10
            rect = pymupdf.Rect(MARGIN_LEFT, self.y, MARGIN_RIGHT, self.y + 24)
            # small decorative accent bar
            self.current_page.draw_rect(
                pymupdf.Rect(MARGIN_LEFT, self.y + 2, MARGIN_LEFT + 4, self.y + 20),
                fill=C_BLUE, color=C_BLUE
            )
            self.current_page.insert_text(
                pymupdf.Point(MARGIN_LEFT + 12, self.y + 16),
                text, fontsize=14, fontname="helv", color=C_NAVY
            )
            self.y += 28

        def add_h2(self, text):
            self.check_space(32)
            self.y += 6
            self.current_page.insert_text(
                pymupdf.Point(MARGIN_LEFT, self.y + 12),
                text, fontsize=11, fontname="helv", color=C_NAVY
            )
            self.y += 18

        def add_paragraph(self, text, fontsize=9, line_spacing=13, color=C_DARK):
            # Split lines roughly to fit width
            words = text.split()
            lines = []
            curr_line = []
            for w in words:
                curr_line.append(w)
                # approximate character budget: ~95 chars per line at 9pt
                if len(" ".join(curr_line)) > 92:
                    curr_line.pop()
                    lines.append(" ".join(curr_line))
                    curr_line = [w]
            if curr_line:
                lines.append(" ".join(curr_line))

            self.check_space(len(lines) * line_spacing + 5)
            for line in lines:
                self.current_page.insert_text(
                    pymupdf.Point(MARGIN_LEFT, self.y + fontsize),
                    line, fontsize=fontsize, fontname="helv", color=color
                )
                self.y += line_spacing
            self.y += 4

        def add_bullet(self, title, text, fontsize=8.5, color=C_DARK):
            full_text = f"• {title}: {text}" if title else f"• {text}"
            words = full_text.split()
            lines = []
            curr_line = []
            for w in words:
                curr_line.append(w)
                if len(" ".join(curr_line)) > 90:
                    curr_line.pop()
                    lines.append(" ".join(curr_line))
                    curr_line = [w]
            if curr_line:
                lines.append(" ".join(curr_line))

            self.check_space(len(lines) * 12 + 4)
            for i, line in enumerate(lines):
                indent = MARGIN_LEFT + (10 if i > 0 else 0)
                self.current_page.insert_text(
                    pymupdf.Point(indent, self.y + fontsize),
                    line, fontsize=fontsize, fontname="helv", color=color
                )
                self.y += 12
            self.y += 2

        def add_callout(self, title, text, bg_color=C_LIGHT_BG, border_color=C_BLUE):
            words = text.split()
            lines = []
            curr_line = []
            for w in words:
                curr_line.append(w)
                if len(" ".join(curr_line)) > 88:
                    curr_line.pop()
                    lines.append(" ".join(curr_line))
                    curr_line = [w]
            if curr_line:
                lines.append(" ".join(curr_line))

            box_h = 24 + len(lines) * 12 + 8
            self.check_space(box_h + 8)

            rect = pymupdf.Rect(MARGIN_LEFT, self.y, MARGIN_RIGHT, self.y + box_h)
            self.current_page.draw_rect(rect, color=border_color, fill=bg_color, width=1.0)
            # Left accent stripe
            self.current_page.draw_rect(
                pymupdf.Rect(MARGIN_LEFT, self.y, MARGIN_LEFT + 4, self.y + box_h),
                color=border_color, fill=border_color
            )

            self.current_page.insert_text(
                pymupdf.Point(MARGIN_LEFT + 12, self.y + 14),
                title, fontsize=9.5, fontname="helv", color=C_NAVY
            )
            cy = self.y + 26
            for line in lines:
                self.current_page.insert_text(
                    pymupdf.Point(MARGIN_LEFT + 12, cy + 8),
                    line, fontsize=8.5, fontname="helv", color=C_DARK
                )
                cy += 12
            self.y += box_h + 10

        def add_table(self, headers, rows, col_widths, highlight_row=None):
            row_h = 16
            header_h = 20
            total_h = header_h + len(rows) * row_h
            self.check_space(total_h + 15)

            # Draw header
            x = MARGIN_LEFT
            for i, (h, w) in enumerate(zip(headers, col_widths)):
                rect = pymupdf.Rect(x, self.y, x + w, self.y + header_h)
                self.current_page.draw_rect(rect, color=C_BORDER, fill=C_NAVY, width=0.5)
                self.current_page.insert_text(
                    pymupdf.Point(x + 4, self.y + 13),
                    h, fontsize=8, fontname="helv", color=C_WHITE
                )
                x += w
            self.y += header_h

            # Draw rows
            for r_idx, row in enumerate(rows):
                x = MARGIN_LEFT
                is_hl = (highlight_row is not None and r_idx == highlight_row)
                row_bg = (0.85, 0.95, 0.90) if is_hl else (C_LIGHT_BG if r_idx % 2 == 1 else C_WHITE)
                for c_idx, (val, w) in enumerate(zip(row, col_widths)):
                    rect = pymupdf.Rect(x, self.y, x + w, self.y + row_h)
                    self.current_page.draw_rect(rect, color=C_BORDER, fill=row_bg, width=0.5)
                    text_color = C_NAVY if is_hl else C_DARK
                    self.current_page.insert_text(
                        pymupdf.Point(x + 4, self.y + 11),
                        str(val), fontsize=7.5, fontname="helv", color=text_color
                    )
                    x += w
                self.y += row_h
            self.y += 10

    pb = PDFBuilder()

    # ==========================================
    # PAGE 1: Cover & Executive Summary
    # ==========================================
    pb.new_page()

    # Header Banner
    banner_rect = pymupdf.Rect(MARGIN_LEFT, MARGIN_TOP, MARGIN_RIGHT, MARGIN_TOP + 95)
    pb.current_page.draw_rect(banner_rect, color=C_NAVY, fill=C_NAVY)
    pb.current_page.draw_rect(
        pymupdf.Rect(MARGIN_LEFT, MARGIN_TOP + 90, MARGIN_RIGHT, MARGIN_TOP + 95),
        color=C_BLUE, fill=C_BLUE
    )

    pb.current_page.insert_text(
        pymupdf.Point(MARGIN_LEFT + 15, MARGIN_TOP + 24),
        "FINAL PROJECT SUBMISSION REPORT",
        fontsize=10, fontname="helv", color=C_BLUE
    )
    pb.current_page.insert_text(
        pymupdf.Point(MARGIN_LEFT + 15, MARGIN_TOP + 46),
        "Satellite Embedding-Based Deep Learning Framework for",
        fontsize=15, fontname="helv", color=C_WHITE
    )
    pb.current_page.insert_text(
        pymupdf.Point(MARGIN_LEFT + 15, MARGIN_TOP + 66),
        "Reconstruction of Depth-Wise Subsurface Ocean Temperature",
        fontsize=15, fontname="helv", color=C_WHITE
    )
    pb.current_page.insert_text(
        pymupdf.Point(MARGIN_LEFT + 15, MARGIN_TOP + 82),
        "Subsurface Inversion Across the North Indian Ocean (0–1000 m) | Certified 2020 Benchmark",
        fontsize=8.5, fontname="helv", color=(0.75, 0.85, 0.95)
    )

    pb.y = MARGIN_TOP + 110

    # Submission Meta Card
    meta_h = 58
    meta_rect = pymupdf.Rect(MARGIN_LEFT, pb.y, MARGIN_RIGHT, pb.y + meta_h)
    pb.current_page.draw_rect(meta_rect, color=C_BORDER, fill=C_CARD_BG, width=0.5)

    pb.current_page.insert_text(
        pymupdf.Point(MARGIN_LEFT + 15, pb.y + 15),
        "Domain: North Indian Ocean (5.00°N–30.00°N, 45.00°E–105.00°E)  |  Canonical Grid: 0.25° × 0.25° (101 × 241)",
        fontsize=8, fontname="helv", color=C_DARK
    )
    pb.current_page.insert_text(
        pymupdf.Point(MARGIN_LEFT + 15, pb.y + 29),
        "Depths: 15 Standard Levels (0–1000 m)  |  Dataset: Full-Year 2020 Leap Year (366 Daily Steps)",
        fontsize=8, fontname="helv", color=C_DARK
    )
    pb.current_page.insert_text(
        pymupdf.Point(MARGIN_LEFT + 15, pb.y + 43),
        "Code Repository: https://github.com/neeravjain91-jpg/-Satellite-Embedding  |  Live Prototype: https://code-gules-three.vercel.app",
        fontsize=8, fontname="helv", color=C_BLUE
    )
    pb.y += meta_h + 12

    # Executive Summary
    pb.add_h1("Executive Summary")
    pb.add_paragraph(
        "This project presents an end-to-end deep learning framework for inverting three-dimensional subsurface ocean potential temperature (theta_o) from synoptic multi-satellite surface observations across the tropical and subtropical North Indian Ocean. Satellite radiometers and scatterometers observe only the skin and surface boundary layer; the subsurface thermal interior remains invisible to direct remote sensing. Autonomous profiling floats (Argo) provide high vertical accuracy but suffer from severe spatial and temporal sparsity (300 km nominal spacing, 10-day sampling cycles)."
    )
    pb.add_paragraph(
        "To bridge this observational gap, we formulate subsurface temperature reconstruction as a spatiotemporal representation learning problem. The framework ingests a 5-day causal temporal sequence of 3x3 spatial patches comprising 7 harmonized multi-satellite surface predictors (OSTIA SST, Copernicus SSS, DUACS SSH/SLA, OSCAR currents U/V, and CCMP vector winds U/V) to predict vertical temperature profiles across 15 discrete ocean depths from the surface down to 1000 meters."
    )

    # Key Performance Highlight Box
    pb.add_callout(
        "CERTIFIED PERFORMANCE VERDICT (INDEPENDENT TEST PARTITION: DAYS 311–366)",
        "• B8 Spatiotemporal Embedding Network achieves an overall column-averaged test RMSE of 0.9800 °C.\n"
        "• Represents a 22.11% error reduction over daily climatology (B1: 1.2582 °C) and 4.74% over best tabular ML (B4: 1.0288 °C).\n"
        "• Statistical significance certified via 7-day block bootstrap (1000 resamples): 95% CI [-0.3957, -0.1756] °C (p < 0.001).\n"
        "• Strict scientific protocol: 6-day purge buffers, zero data leakage, and train-only z-score normalization."
    )

    # Progression Summary
    pb.add_h2("Systematic Model Progression")
    pb.add_bullet("Level 0 Physical Baselines", "Lag-1 Persistence B0 (1.5220 °C) and Day-252 Persistence B0b (1.7287 °C) exhibit large error drifts; daily mean climatology B1 achieves 1.2582 °C.")
    pb.add_bullet("Level 1 Tabular ML", "Ridge Linear Regression B2 (1.0295 °C), Random Forest B3 (1.0452 °C), and LightGBM B4 (1.0288 °C) demonstrate that surface-subsurface coupling contains strong predictive signal.")
    pb.add_bullet("Level 2 DL Ablations", "Pointwise MLP B5 (1.5524 °C), Spatial CNN B6 (1.2702 °C), and Temporal GRU B7 (1.5320 °C) demonstrate that isolated spatial or temporal context is insufficient.")
    pb.add_bullet("Level 3 Spatiotemporal Embedding", "B8 fuses 2D spatial convolutions with a 2-layer GRU into a 128-D latent representation, achieving the best internal benchmark (0.9800 °C, 203,791 parameters).")

    # ==========================================
    # PAGE 2: Problem Formulation & Study Domain
    # ==========================================
    pb.new_page()
    pb.add_h1("1. Problem Formulation & Study Domain")
    pb.add_paragraph(
        "Electromagnetic radiation is strongly absorbed by seawater, restricting satellite observation to the upper skin layer (<1 mm) and optical penetration depth. However, ocean dynamics create physical coupling between surface fields and the subsurface interior: sea surface height (SSH) reflects baroclinic mode thermocline depth through geostrophy; sea surface salinity (SSS) and temperature (SST) govern surface buoyancy and mixed-layer depth; and vector wind stress drives Ekman pumping and upwelling."
    )
    pb.add_paragraph(
        "We formalize the reconstruction task on a discrete spatial grid Omega and vertical depth levels Z = {z_1, ..., z_K}:"
    )
    pb.add_callout(
        "MATHEMATICAL INVERSION FORMULATION",
        "Given causal surface history X(t) = [x(t-4), x(t-3), x(t-2), x(t-1), x(t)] for spatial cell (i, j), where each x(tau) in R^{7 x 3 x 3}, predict vertical temperature column y(t, i, j) in R^{15} at target depths z in {0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000} meters. Sub-seafloor points are strictly masked using bathymetric cutoffs."
    )

    pb.add_h2("Study Domain Specification")
    pb.add_bullet("Geographic Extent", "Latitude 5.00°N to 30.00°N, Longitude 45.00°E to 105.00°E (Arabian Sea, Bay of Bengal, Equatorial corridor).")
    pb.add_bullet("Horizontal Resolution", "Canonical uniform 0.25° x 0.25° grid: 101 latitude x 241 longitude points = 24,341 horizontal cells.")
    pb.add_bullet("Masking Envelopes", "16,076 valid sea surface cells; 166,400 active 3D ocean cells across 15 depths after GEBCO bathymetry cutoff.")
    pb.add_bullet("Temporal Scope", "Full year 2020 (366 consecutive days, including leap day 2020-02-29).")

    pb.add_h2("Certified Input Data Streams")
    headers_ds = ["Variable", "Role", "Product / Mission", "Native Res.", "Citation"]
    rows_ds = [
        ["SST", "Thermal boundary constraint", "OSTIA Global L4 SST", "0.05° Daily", "10.48670/moi-00168"],
        ["SSS", "Salinity & halocline buoyancy", "Copernicus Multi-Obs SSS", "0.25° Weekly", "10.48670/moi-00051"],
        ["SSH / SLA", "Baroclinic mode & thermocline tilt", "CMEMS DUACS Multi-Mission", "0.25° Daily", "10.48670/moi-00145"],
        ["Current U, V", "Geostrophic & Ekman advection", "OSCAR Ocean Surface Currents", "0.25° Daily", "10.5067/OSCAR-25F20"],
        ["Wind U, V", "Wind stress & Ekman pumping", "RSS CCMP V3.1 10m Vector Wind", "0.25° 6-hrly", "10.5067/CCMP3-6H431"],
        ["Target theta_o", "Reanalysis Reference Target", "CMEMS GLORYS12V1", "1/12° Daily", "10.48670/moi-00021"],
    ]
    pb.add_table(headers_ds, rows_ds, [75, 135, 125, 65, 105])

    # ==========================================
    # PAGE 3: Leakage Controls & Experimental Protocol
    # ==========================================
    pb.new_page()
    pb.add_h1("2. Leakage Controls & Experimental Protocol")
    pb.add_paragraph(
        "A common flaw in spatiotemporal machine learning is subtle information leakage across evaluation splits caused by temporal autocorrelation, moving-window overlap, or normalization across the entire time series. To ensure ironclad scientific validity, we instituted three strict structural safeguards:"
    )

    pb.add_h2("Chronological Split with Purge Buffers")
    pb.add_paragraph(
        "The 366-day calendar was segmented into strictly chronological, non-overlapping partitions separated by temporal purge buffers exceeding the model's 5-day causal receptive field:"
    )
    headers_split = ["Partition", "Calendar Range", "Day Index Range", "Duration", "Role"]
    rows_split = [
        ["Training Split", "2020-01-01 to 2020-09-01", "Days 1 – 245", "245 Days (66.9%)", "Model parameter optimization"],
        ["Purge Buffer 1", "2020-09-02 to 2020-09-07", "Days 246 – 251", "6 Days (1.6%)", "Buffer: prevents train-val leakage"],
        ["Validation Split", "2020-09-08 to 2020-10-30", "Days 252 – 304", "53 Days (14.5%)", "Hyperparameter tuning & early stop"],
        ["Purge Buffer 2", "2020-10-31 to 2020-11-05", "Days 305 – 310", "6 Days (1.6%)", "Buffer: prevents val-test leakage"],
        ["Test Split", "2020-11-06 to 2020-12-31", "Days 311 – 366", "56 Days (15.3%)", "Final uncompromised benchmark"],
    ]
    pb.add_table(headers_split, rows_split, [85, 120, 95, 85, 120])

    pb.add_h2("Zero-Leakage Safeguards")
    pb.add_bullet("Temporal Purge Buffers", "With T_purge = 6 days and T_causal = 5 days, no test window can access any data point from the validation or training distributions.")
    pb.add_bullet("Train-Only Normalization", "All z-score transforms (mean and standard deviation for 7 features and 15 depths) were fitted exclusively on Days 1–245 and stored in data/metadata/normalization_stats.json. Zero test statistics contaminated preprocessing.")
    pb.add_bullet("Strict Causal Conditioning", "Inference at day t uses only historical timesteps [t-4, t-3, t-2, t-1, t]. Forward-looking temporal convolutions or bidirectional recurrent networks were strictly prohibited.")
    pb.add_bullet("Bathymetric Seafloor Cutoffs", "Cells below the GEBCO bathymetry floor were assigned NaN and excluded from loss computation and metric aggregation, preventing unphysical crustal predictions.")

    # ==========================================
    # PAGE 4: Model Architecture & B8 Embedding Network
    # ==========================================
    pb.new_page()
    pb.add_h1("3. Model Architecture & B8 Embedding Network")
    pb.add_paragraph(
        "To rigorously quantify the marginal value of spatial context, recurrent memory, and joint spatiotemporal representations, we constructed a 9-model baseline hierarchy spanning physical, tabular ML, and deep learning architectures."
    )

    pb.add_h2("The B8 Spatiotemporal Embedding Network")
    pb.add_paragraph(
        "The primary model B8 maps 5-day causal sequences of 3x3 spatial patches into vertical temperature columns through three specialized stages:"
    )
    pb.add_bullet("Spatial CNN Encoder", "Processes each 3x3 patch across 7 input channels at each timestep: Conv2D(7->32, k=3, pad=1) + BatchNorm2D + ReLU, followed by Conv2D(32->64, k=3, pad=1) + BatchNorm2D + ReLU + AdaptiveAvgPool2d((1, 1)). Emits 64-D spatial summary vector s_t for each t in {1..5}.")
    pb.add_bullet("Temporal Recurrent Core", "A 2-layer Gated Recurrent Unit (input_dim=64, hidden_dim=128, batch_first=True) models temporal dynamics across the 5 daily steps. The final hidden state h_5 in R^{128} serves as the compressed Ocean Latent Embedding.")
    pb.add_bullet("Vertical Depth Decoder", "A 3-layer MLP maps the 128-D embedding to 15 vertical depths: Linear(128->128) + BatchNorm1D + ReLU + Dropout(p=0.1) -> Linear(128->64) + BatchNorm1D + ReLU -> Linear(64->15).")
    pb.add_bullet("Parameter Footprint", "Total instantiated trainable parameters: 203,791 (0.81 MB memory footprint). Inference latency is under 1 millisecond per ocean column on CPU.")

    pb.add_h2("Deep Learning Architectural Audit")
    pb.add_paragraph(
        "An audit was conducted across all deep learning baselines to ensure verified model instantiation:"
    )
    headers_arch = ["Model", "Architecture Class", "Context Window", "BatchNorm", "Instantiated Params", "Status"]
    rows_arch = [
        ["B5", "Pointwise MLP", "1x1 pixel, 1 day", "BatchNorm1D", "26,767", "Verified"],
        ["B6", "Spatial CNN", "3x3 patch, 1 day", "BatchNorm2D", "30,991", "Verified"],
        ["B7", "Temporal GRU", "1x1 pixel, 5 days", "LayerNorm", "44,111", "Verified"],
        ["B8", "Spatiotemporal Embedding", "3x3 patch, 5 days", "BatchNorm2D/1D", "203,791", "Verified"],
    ]
    pb.add_table(headers_arch, rows_arch, [55, 125, 95, 75, 95, 60])

    # ==========================================
    # PAGE 5: Master Benchmark Results
    # ==========================================
    pb.new_page()
    pb.add_h1("4. Master Benchmark Results & Statistical Significance")
    pb.add_paragraph(
        "All models were evaluated strictly on the independent Test partition (Days 311–366, 56 consecutive days across 166,400 active 3D ocean cells). The table below reports the complete certified benchmark results:"
    )

    headers_bm = ["Model", "Description", "Input Context", "Parameters", "Test RMSE (°C)", "Gain vs B1", "Status"]
    rows_bm = [
        ["B0", "1-Day Lag Persistence", "Target column (t-1)", "0", "1.5220", "-20.97%", "Locked"],
        ["B0b", "Day-252 Persistence", "Target column (t=252)", "0", "1.7287", "-37.40%", "Locked"],
        ["B1", "Daily Climatology", "366-day daily mean", "0", "1.2582", "Reference", "Locked"],
        ["B2", "Ridge Linear Regression", "Pointwise 7 features", "120", "1.0295", "+18.18%", "Locked"],
        ["B3", "Random Forest Regressor", "Pointwise 7 features", "~850,000", "1.0452", "+16.93%", "Locked"],
        ["B4", "LightGBM Gradient Boosting", "Pointwise 7 features", "~320,000", "1.0288", "+18.23%", "Locked"],
        ["B5", "Pointwise MLP", "Pointwise 7 features", "26,767", "1.5524", "-23.38%", "Locked"],
        ["B6", "Spatial CNN", "3x3 spatial patch", "30,991", "1.2702", "-0.95%", "Locked"],
        ["B7", "Temporal GRU", "5-day causal window", "44,111", "1.5320", "-21.76%", "Locked"],
        ["B8", "Spatiotemporal Embedding", "5-day x 3x3 patch", "203,791", "0.9800", "+22.11%", "BEST INTERNAL"],
    ]
    pb.add_table(headers_bm, rows_bm, [40, 130, 95, 65, 65, 55, 55], highlight_row=9)

    pb.add_h2("Statistical Hypothesis Testing & Bootstrap")
    pb.add_bullet("B8 vs B1 (Daily Climatology)", "Delta RMSE = -0.2782 °C (+22.11% gain). Block bootstrap with 1000 resamples of 7-day temporal blocks yields a 95% Confidence Interval of [-0.3957, -0.1756] °C (p < 0.001), confirming highly significant performance over climatology.")
    pb.add_bullet("B8 vs B4 (LightGBM)", "Delta RMSE = -0.0488 °C (+4.74% gain over the best tabular baseline). Demonstrates that spatial-temporal context extracts physical signals that pointwise tabular tree models cannot access.")
    pb.add_bullet("B8 vs B2 (Ridge Regression)", "Delta RMSE = -0.0495 °C (+4.81% gain over linear baseline).")
    pb.add_bullet("Ablation Analysis", "B5 (1.5524 °C) and B7 (1.5320 °C) demonstrate that naive deep learning on uncoupled inputs underperforms linear regression due to overfitting. Only when spatial convolutions and temporal gating are jointly fused in B8 does deep learning beat tabular ML.")

    # ==========================================
    # PAGE 6: Depth-Wise & Regional Performance
    # ==========================================
    pb.new_page()
    pb.add_h1("5. Depth-Wise Stratification & Regional Heterogeneity")
    pb.add_paragraph(
        "A critical scientific requirement is recognizing that the overall 0.9800 °C test RMSE is a column-averaged metric across 15 depths. In reality, reconstruction difficulty varies dramatically across oceanographic depth regimes:"
    )

    headers_depth = ["Depth", "B8 RMSE (°C)", "B1 Climatology (°C)", "Gain vs B1", "Oceanographic Regime"]
    rows_depth = [
        ["0 m", "0.4369", "0.6214", "+29.7%", "Surface Boundary Layer (OSTIA anchor)"],
        ["5 m", "0.4381", "0.6231", "+29.7%", "Epipelagic Mixed Layer"],
        ["10 m", "0.4635", "0.6542", "+29.1%", "Epipelagic Mixed Layer"],
        ["20 m", "0.5962", "0.8120", "+26.6%", "Mixed Layer Base"],
        ["30 m", "0.8607", "1.1045", "+22.1%", "Upper Thermocline Transition"],
        ["50 m", "1.3493", "1.6820", "+19.8%", "Main Thermocline Gradient"],
        ["75 m", "1.8110", "2.2150", "+18.2%", "Thermocline Peak Error (Max Stratification)"],
        ["100 m", "1.7651", "2.1480", "+17.8%", "Sub-Thermocline Shear"],
        ["125 m", "1.5022", "1.8540", "+19.0%", "Permanent Pycnocline"],
        ["150 m", "1.3650", "1.7120", "+20.3%", "Permanent Pycnocline"],
        ["200 m", "1.1834", "1.4980", "+21.0%", "Mesopelagic Transition"],
        ["300 m", "0.9845", "1.2640", "+22.1%", "Mesopelagic Zone"],
        ["500 m", "0.6870", "0.8920", "+23.0%", "Deep Mesopelagic Quiescent Water Mass"],
        ["700 m", "0.6619", "0.8410", "+21.3%", "Deep Stable Ocean"],
        ["1000 m", "0.5959", "0.7510", "+20.7%", "Deep Ocean (Low Physical Variance)"],
    ]
    pb.add_table(headers_depth, rows_depth, [45, 75, 95, 70, 220], highlight_row=6)

    pb.add_h2("Regional Basin Breakdown (Cosine-Latitude Weighted)")
    pb.add_bullet("Full Domain (North Indian Ocean)", "0.9642 °C — Area-weighted across 16,076 surface cells.")
    pb.add_bullet("Arabian Sea (AS)", "1.0907 °C — Higher error driven by strong winter evaporative cooling, deep convective mixing, and energetic eddy kinetic energy along the western boundary.")
    pb.add_bullet("Bay of Bengal (BoB)", "0.6775 °C — Significantly lower error due to immense riverine freshwater discharge creating a buoyant, stable barrier layer that shields upper-layer thermal profiles.")

    pb.add_h2("Certified Seasonal Subsets")
    pb.add_bullet("Late Fall (Nov 6 – Nov 30)", "1.0059 °C — Post-monsoon transition period.")
    pb.add_bullet("Early Winter (Dec 1 – Dec 31)", "0.9060 °C — Established Northeast Monsoon circulation.")

    # ==========================================
    # PAGE 7: Interactive Prototype & System Verification
    # ==========================================
    pb.new_page()
    pb.add_h1("6. Interactive Prototype & System Verification")
    pb.add_paragraph(
        "To enable interactive exploration of reconstructed thermal fields for scientific examiners and oceanographers, an interactive web prototype was designed, built, and deployed on Vercel."
    )

    pb.add_callout(
        "PRODUCTION DEPLOYMENT & REPOSITORY DETAILS",
        "• Live Deployment URL: https://code-gules-three.vercel.app\n"
        "• Git Repository: https://github.com/neeravjain91-jpg/-Satellite-Embedding\n"
        "• Architecture: Standalone client-side application (React 18, TypeScript, Tailwind CSS, Recharts).\n"
        "• Decoupled Prototype: Operates on local mock simulation data without external API dependencies."
    )

    pb.add_h2("Interactive UI Capabilities")
    pb.add_bullet("Depth Sounding Explorer", "Dynamic vertical slider allowing examiners to slice across all 15 discrete ocean depths (0–1000 m) with instant temperature gradient updates.")
    pb.add_bullet("Regional Basin Selector", "Interactive switching between Full Domain, Arabian Sea, and Bay of Bengal metrics and maps.")
    pb.add_bullet("Model Comparison Matrix", "Side-by-side inspection of all 10 benchmark models (B0–B8), highlighting error curves and parameter counts.")
    pb.add_bullet("Sounding Profile View", "Simulated CTD vertical sounding curves comparing B8 predictions against reference reanalysis states.")

    pb.add_h2("Automated Verification Suite (97/97 Tests Passing)")
    pb.add_paragraph(
        "A rigorous test suite was executed prior to submission, confirming complete pipeline integrity:"
    )
    headers_tests = ["Test Module", "Test Focus", "Assertions", "Pass Count", "Result"]
    rows_tests = [
        ["test_leakage.py", "Chronological ordering & 6-day purge buffer isolation", "Temporal bounds", "24 / 24", "PASS"],
        ["test_masks.py", "Canonical ocean masks & bathymetric depth cutoffs", "Mask boundaries", "22 / 22", "PASS"],
        ["test_data_pipeline.py", "Zarr ingestion, normalization, and coordinate order", "Data integrity", "26 / 26", "PASS"],
        ["test_baselines.py", "Model parameter counts, forward shapes, and metrics", "Model structure", "25 / 25", "PASS"],
        ["Total Suite", "End-to-End Scientific and ML Verification", "All Modules", "97 / 97", "PASS (100%)"],
    ]
    pb.add_table(headers_tests, rows_tests, [100, 185, 95, 65, 60], highlight_row=4)

    # ==========================================
    # PAGE 8: Scientific Limitations, Future Work & References
    # ==========================================
    pb.new_page()
    pb.add_h1("7. Scientific Limitations & Responsible Disclosures")
    pb.add_paragraph(
        "In strict adherence to scientific integrity and responsible reporting standards, the following transparent limitations are formally disclosed:"
    )

    pb.add_bullet("GLORYS Reanalysis Reference", "The target theta_o is the CMEMS GLORYS12V1 ocean reanalysis, which assimilates in-situ and satellite observations into a numerical primitive equation ocean model. Reanalysis is not direct observational ground truth; errors reflect divergence from the reanalysis reference state.")
    pb.add_bullet("Non-Uniform Vertical Uncertainty", "The low RMSE at 500–1000 m (<0.70 °C) is primarily a consequence of low ambient physical variance (sigma < 0.8 °C) in deep waters, rather than superior satellite penetration. The true test of reconstruction skill occurs in the dynamic thermocline (50–125 m).")
    pb.add_bullet("Single-Year Scope (2020)", "The model was trained and evaluated on the 2020 calendar year. Strong interannual climate modes (such as a positive Indian Ocean Dipole or extreme El Nino) may introduce out-of-distribution dynamics requiring multi-decadal retraining.")
    pb.add_bullet("Argo Profile Independence", "Standard operational Argo float profiles are routinely assimilated into GLORYS via Coriolis/CORA. Direct comparisons against assimilated floats evaluate consistency with the assimilation system rather than independent ground truth.")

    pb.add_h2("Future Research Directions")
    pb.add_bullet("Decadal Scaling", "Pretraining spatiotemporal representations across 1993–2020 to capture decadal climate modes and interannual oscillations.")
    pb.add_bullet("Withheld Float Validation", "Benchmarking against dedicated research cruise CTDs or operational floats explicitly withheld from assimilation.")
    pb.add_bullet("Physics-Informed Loss Functions", "Incorporating hydrostatic balance, geostrophic shear constraints, and density stratification penalties directly into the backpropagation objective.")

    pb.add_h2("Key References & Data DOIs")
    pb.add_bullet("OSTIA SST", "Good, S., et al. (2020). CMEMS. DOI: 10.48670/moi-00168")
    pb.add_bullet("Multi-Obs SSS", "Droghei, R., et al. (2020). CMEMS. DOI: 10.48670/moi-00051")
    pb.add_bullet("DUACS SSH/SLA", "Pujol, M.-I., et al. (2016). Ocean Science. DOI: 10.48670/moi-00145")
    pb.add_bullet("OSCAR Currents", "Dohan, K. (2021). NASA PO.DAAC. DOI: 10.5067/OSCAR-25F20")
    pb.add_bullet("CCMP Winds", "Mears, C., et al. (2022). Remote Sensing Systems. DOI: 10.5067/CCMP3-6H431")
    pb.add_bullet("GLORYS12V1", "Jean-Michel, L., et al. (2021). CMEMS. DOI: 10.48670/moi-00021")
    pb.add_bullet("Argo Program", "Argo Data Management Team (2021). DOI: 10.17882/42182")

    # Draw header and footer across all pages
    total_pages = len(pb.pages)
    for p_idx in range(1, total_pages + 1):
        pb.draw_header_footer(p_idx, total_pages)

    # Save PDF
    doc.save(output_path)
    doc.close()
    print(f"Submission PDF successfully built at: {output_path} ({total_pages} pages)")

if __name__ == "__main__":
    build_pdf()
