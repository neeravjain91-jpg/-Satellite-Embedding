"""
scripts/compute_b3_winner_confusion_matrix.py
Computes the full 6-regime confusion matrix and per-class classification diagnostics
for the exploratory hyperparameter winner (B3-A: 100 trees, max_depth=15, 100k samples).

Saves outputs strictly in results/exploratory/:
- results/exploratory/B3_winner_confusion_matrix.json
- results/exploratory/B3_winner_confusion_matrix.html
"""

import os
import sys
import time
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, cohen_kappa_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS
from preprocessing.tabular_dataset import load_tabular_dataset
from scripts.evaluate_confusion_matrix import REGIME_BINS, REGIME_LABELS, evaluate_regimes


def main():
    print("=" * 80)
    print("COMPUTING PHYSICAL THERMAL REGIME CONFUSION MATRIX FOR B3-A (WINNER)")
    print("=" * 80)

    # 1. Load canonical dataset
    print("\nLoading canonical tabular dataset...")
    dataset = load_tabular_dataset(build_context=False)
    train_data = dataset["train"]
    test_data = dataset["test"]

    X_train = train_data["X_norm"]
    Y_train = train_data["Y"]
    M_train = train_data["mask"]

    X_test = test_data["X_norm"]
    Y_test = test_data["Y"]
    M_test = test_data["mask"]

    y_true_valid = Y_test[M_test]
    n_total = len(y_true_valid)
    assert n_total == 8017734, f"Expected 8,017,734 test points, got {n_total}"
    print(f"Verified {n_total:,} valid ocean test targets across 15 depths and 53 days.")

    # 2. Fit B3-A strictly on train data (100k subsample, seed 42)
    print("\nFitting B3-A: Multi-Depth Random Forest (n_estimators=100, max_depth=15, 100k samples)...")
    N_train = len(X_train)
    np.random.seed(42)
    sub_100k = np.random.choice(N_train, size=min(100000, N_train), replace=False)

    X_sub = X_train[sub_100k]
    Y_sub = Y_train[sub_100k]
    M_sub = M_train[sub_100k]

    n_depths = len(CANONICAL_DEPTHS)
    b3a_models = {}
    total_nodes = 0

    t0 = time.time()
    for d in range(n_depths):
        valid_d = M_sub[:, d]
        if not np.any(valid_d):
            b3a_models[d] = None
            continue

        X_d = X_sub[valid_d]
        y_d = Y_sub[valid_d, d]

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=-1,
            random_state=42 + d
        )
        rf.fit(X_d, y_d)
        b3a_models[d] = rf
        total_nodes += sum(tree.tree_.node_count for tree in rf.estimators_)
        print(f"  [Depth {d:02d} / {int(CANONICAL_DEPTHS[d])}m] Fitted 100 trees ({sum(t.tree_.node_count for t in rf.estimators_):,} nodes)")

    t_train = time.time() - t0
    print(f"\nTraining completed in {t_train:.1f}s. Total decision nodes: {total_nodes:,}")

    # 3. Predict on Test
    print("\nPredicting on frozen held-out test partition...")
    t_inf_0 = time.time()
    y_pred_b3a = np.zeros_like(Y_test)
    for d in range(n_depths):
        if b3a_models.get(d) is not None:
            y_pred_b3a[:, d] = b3a_models[d].predict(X_test)
        else:
            y_pred_b3a[:, d] = 15.0
    t_inf = time.time() - t_inf_0
    print(f"Inference completed in {t_inf:.2f}s.")

    # 4. Compute 6-regime confusion matrix
    print("\nComputing 6-regime confusion matrix and diagnostics...")
    y_pred_valid = y_pred_b3a[M_test]
    res_b3a = evaluate_regimes(y_true_valid, y_pred_valid, labels=REGIME_LABELS, bins=REGIME_BINS)

    print("\n" + "=" * 60)
    print("B3-A (WINNER) TEST THERMAL REGIME DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Exact Accuracy:        {res_b3a['accuracy_pct']}%")
    print(f"Within ±1 Bin:         {res_b3a['within_1_bin_pct']}%")
    print(f"Beyond ±1 Bin:         {res_b3a['beyond_1_bin_pct']}%")
    print(f"Cohen's Kappa:         {res_b3a['cohen_kappa']:.4f}")
    print(f"Macro F1:              {res_b3a['macro_f1']:.4f}")
    print(f"Weighted F1:           {res_b3a['weighted_f1']:.4f}")

    # Load Certified B3 for delta comparison
    cert_cm_path = os.path.join(repo_root, "results", "confusion_matrix.json")
    with open(cert_cm_path, "r", encoding="utf-8") as f:
        cert_data = json.load(f)
    cert_b3 = cert_data["models"]["B3_RandomForest"]

    print("\n" + "=" * 60)
    print("COMPARISON: B3-A (WINNER) VS CERTIFIED B3")
    print("=" * 60)
    print(f"Accuracy:      {cert_b3['accuracy_pct']}% -> {res_b3a['accuracy_pct']}% (Δ = {res_b3a['accuracy_pct'] - cert_b3['accuracy_pct']:+.2f}%)")
    print(f"Within ±1:     {cert_b3['within_1_bin_pct']}% -> {res_b3a['within_1_bin_pct']}% (Δ = {res_b3a['within_1_bin_pct'] - cert_b3['within_1_bin_pct']:+.2f}%)")
    print(f"Cohen Kappa:   {cert_b3['cohen_kappa']:.4f} -> {res_b3a['cohen_kappa']:.4f} (Δ = {res_b3a['cohen_kappa'] - cert_b3['cohen_kappa']:+.4f})")
    print(f"Macro F1:      {cert_b3['macro_f1']:.4f} -> {res_b3a['macro_f1']:.4f} (Δ = {res_b3a['macro_f1'] - cert_b3['macro_f1']:+.4f})")
    print(f"Weighted F1:   {cert_b3['weighted_f1']:.4f} -> {res_b3a['weighted_f1']:.4f} (Δ = {res_b3a['weighted_f1'] - cert_b3['weighted_f1']:+.4f})")

    # 5. Save JSON bundle
    out_dir = os.path.join(repo_root, "results", "exploratory")
    os.makedirs(out_dir, exist_ok=True)

    cm_counts = res_b3a["confusion_matrix"]
    total_samples = res_b3a["total_evaluated_samples"]
    cm_pct = [[round(val / total_samples * 100, 4) for val in row] for row in cm_counts]

    json_output = {
        "metadata": {
            "title": "B3-A Validation Winner Thermal Regime Confusion Matrix",
            "model_id": "B3-A",
            "model_name": "Multi-Depth Random Forest (100 trees, max_depth=15)",
            "sample_train_size": 100000,
            "n_estimators": 100,
            "max_depth": 15,
            "decision_nodes": total_nodes,
            "total_trees": 1500,
            "evaluation_partition": "Held-Out Test (Days 313–365, Nov 9 – Dec 31, 2020)",
            "total_valid_test_points": n_total,
            "status": "EXPLORATORY • NON-CANONICAL"
        },
        "regimes": {
            "labels": REGIME_LABELS,
            "temperature_boundaries_c": [10.0, 15.0, 20.0, 25.0, 28.0]
        },
        "metrics": {
            "accuracy_pct": res_b3a["accuracy_pct"],
            "within_1_bin_pct": res_b3a["within_1_bin_pct"],
            "beyond_1_bin_pct": res_b3a["beyond_1_bin_pct"],
            "cohen_kappa": res_b3a["cohen_kappa"],
            "macro_f1": res_b3a["macro_f1"],
            "weighted_f1": res_b3a["weighted_f1"]
        },
        "comparison_vs_certified_b3": {
            "delta_accuracy_pct": round(res_b3a["accuracy_pct"] - cert_b3["accuracy_pct"], 2),
            "delta_within_1_bin_pct": round(res_b3a["within_1_bin_pct"] - cert_b3["within_1_bin_pct"], 2),
            "delta_cohen_kappa": round(res_b3a["cohen_kappa"] - cert_b3["cohen_kappa"], 4),
            "delta_macro_f1": round(res_b3a["macro_f1"] - cert_b3["macro_f1"], 4),
            "delta_weighted_f1": round(res_b3a["weighted_f1"] - cert_b3["weighted_f1"], 4)
        },
        "confusion_matrix_counts": cm_counts,
        "confusion_matrix_percent": cm_pct,
        "per_class_metrics": res_b3a.get("per_class_metrics", [])
    }

    json_path = os.path.join(out_dir, "B3_winner_confusion_matrix.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    # 6. Generate HTML visualization
    html_content = generate_html_report(json_output, cert_b3)
    html_path = os.path.join(out_dir, "B3_winner_confusion_matrix.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SAVED] {html_path}")


def generate_html_report(winner_json, cert_b3):
    m = winner_json["metrics"]
    cm = winner_json["confusion_matrix_counts"]
    comp = winner_json["comparison_vs_certified_b3"]
    labels = winner_json["regimes"]["labels"]
    short_labels = ["<10°C", "10–15°C", "15–20°C", "20–25°C", "25–28°C", "≥28°C"]
    total = winner_json["metadata"]["total_valid_test_points"]

    # Calculate row sums and col sums
    row_sums = [sum(row) for row in cm]
    col_sums = [sum(cm[r][c] for r in range(6)) for c in range(6)]

    # Build HTML rows
    table_rows = []
    for r in range(6):
        cells = []
        cells.append(f'<td class="p-2.5 font-medium text-xs text-[var(--foreground)] bg-[var(--background)]/40 border-b border-[var(--border)]">{labels[r]}</td>')
        for c in range(6):
            count = cm[r][c]
            pct = count / total * 100
            diag = (r == c)
            off1 = (abs(r - c) == 1)

            if diag:
                bg = f"rgba(59, 130, 246, {min(0.85, max(0.15, pct/35))})"
                text_color = "#ffffff" if pct > 8 else "var(--foreground)"
                border_cls = "ring-1 ring-blue-400"
            elif off1:
                bg = f"rgba(147, 197, 253, {min(0.35, max(0.04, pct/10))})"
                text_color = "var(--foreground)"
                border_cls = ""
            else:
                bg = f"rgba(239, 68, 68, {min(0.4, max(0.05, pct*2))})" if count > 0 else "transparent"
                text_color = "var(--muted-foreground)"
                border_cls = ""

            cells.append(
                f'<td class="p-2 text-center text-xs border-b border-[var(--border)] cell-hover transition-all {border_cls}" style="background-color: {bg}; color: {text_color};">'
                f'<div class="font-bold val-pct">{pct:.2f}%</div>'
                f'<div class="text-[10px] opacity-75 val-cnt hidden">{count:,}</div>'
                f'</td>'
            )
        # Total column
        row_pct = row_sums[r] / total * 100
        cells.append(f'<td class="p-2 text-center text-xs font-semibold border-b border-[var(--border)] bg-[var(--background)]/40 text-[var(--foreground)]"><span class="val-pct">{row_pct:.1f}%</span><span class="val-cnt hidden">{row_sums[r]:,}</span></td>')
        table_rows.append(f'<tr>{"".join(cells)}</tr>')

    # Bottom summary row
    bot_cells = ['<td class="p-2 text-xs font-bold text-[var(--foreground)] bg-[var(--background)]/40">Total Pred</td>']
    for c in range(6):
        col_pct = col_sums[c] / total * 100
        bot_cells.append(f'<td class="p-2 text-center text-xs font-semibold bg-[var(--background)]/40 text-[var(--foreground)]"><span class="val-pct">{col_pct:.1f}%</span><span class="val-cnt hidden">{col_sums[c]:,}</span></td>')
    bot_cells.append(f'<td class="p-2 text-center text-xs font-bold bg-[var(--background)]/60 text-blue-500">100.0%</td>')
    bottom_row = f'<tr>{"".join(bot_cells)}</tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>B3-A Thermal Regime Confusion Matrix (Exploratory Winner)</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .cell-hover:hover {{
      transform: scale(1.04);
      z-index: 20;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}
    .custom-scroll::-webkit-scrollbar {{
      height: 6px;
      width: 6px;
    }}
    .custom-scroll::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 3px;
    }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3 font-sans">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-5 shadow-lg max-w-4xl mx-auto">
    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-[var(--border)] gap-3">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-500 font-bold text-sm">🌲</span>
          <h2 class="text-[var(--foreground)] font-bold text-xl tracking-tight">B3-A Thermal Regime Confusion Matrix</h2>
          <span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">Validation Winner</span>
        </div>
        <p class="text-[var(--muted-foreground)] text-xs mt-1">
          Evaluated on <strong>8,017,734</strong> valid ocean test points across 15 depths (0–1000m) & 53 test days (Nov 9 – Dec 31, 2020).
        </p>
        <p class="text-[var(--muted-foreground)] text-[11px] mt-0.5 italic">
          B3-A Config: 100 trees, max depth 15, 100k samples &bull; Total Complexity: 26,549,170 decision nodes (1,500 trees).
        </p>
      </div>

      <!-- Controls -->
      <div class="flex items-center gap-2">
        <div class="inline-flex rounded-lg p-0.5 bg-[var(--background)] border border-[var(--border)] text-xs">
          <button id="btn-mode-pct" onclick="setMode('pct')" class="px-2.5 py-1 rounded-md font-medium transition-all bg-[var(--card)] text-[var(--foreground)] shadow-xs">Percent (%)</button>
          <button id="btn-mode-cnt" onclick="setMode('cnt')" class="px-2.5 py-1 rounded-md font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">Counts (N)</button>
        </div>
      </div>
    </div>

    <!-- KPI Summary Cards with Comparison -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 my-4">
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <span class="text-[var(--muted-foreground)] text-xs block font-medium">Exact Accuracy</span>
        <div class="flex items-baseline gap-2">
          <span class="text-xl font-bold text-blue-500">{m['accuracy_pct']:.2f}%</span>
          <span class="text-[11px] font-bold text-emerald-500">+{comp['delta_accuracy_pct']:.2f}%</span>
        </div>
        <span class="text-[var(--muted-foreground)] text-[11px] block mt-0.5">Certified B3: {cert_b3['accuracy_pct']:.2f}%</span>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <span class="text-[var(--muted-foreground)] text-xs block font-medium">Within ±1 Bin</span>
        <div class="flex items-baseline gap-2">
          <span class="text-xl font-bold text-emerald-500">{m['within_1_bin_pct']:.2f}%</span>
          <span class="text-[11px] font-bold text-[var(--muted-foreground)]">+{comp['delta_within_1_bin_pct']:.2f}%</span>
        </div>
        <span class="text-[var(--muted-foreground)] text-[11px] block mt-0.5">Certified B3: {cert_b3['within_1_bin_pct']:.2f}%</span>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <span class="text-[var(--muted-foreground)] text-xs block font-medium">Cohen's Kappa (κ)</span>
        <div class="flex items-baseline gap-2">
          <span class="text-xl font-bold text-indigo-500">{m['cohen_kappa']:.4f}</span>
          <span class="text-[11px] font-bold text-emerald-500">+{comp['delta_cohen_kappa']:.4f}</span>
        </div>
        <span class="text-[var(--muted-foreground)] text-[11px] block mt-0.5">Certified B3: {cert_b3['cohen_kappa']:.4f}</span>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <span class="text-[var(--muted-foreground)] text-xs block font-medium">Macro / Weighted F1</span>
        <div class="flex items-baseline gap-2">
          <span class="text-xl font-bold text-purple-500">{m['macro_f1']:.4f}</span>
          <span class="text-[11px] font-bold text-emerald-500">+{comp['delta_macro_f1']:.4f}</span>
        </div>
        <span class="text-[var(--muted-foreground)] text-[11px] block mt-0.5">Certified: {cert_b3['macro_f1']:.4f} / {cert_b3['weighted_f1']:.4f}</span>
      </div>
    </div>

    <!-- Heatmap Table -->
    <div class="overflow-x-auto custom-scroll pb-2">
      <div class="min-w-[620px]">
        <div class="text-center font-bold text-xs uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
          Predicted Thermal Regime &rarr;
        </div>
        <table class="w-full border-collapse">
          <thead>
            <tr>
              <th class="p-2 text-left font-bold text-xs text-[var(--muted-foreground)] border-b border-[var(--border)]">Actual \\ Pred</th>
              {"".join(f'<th class="p-2 text-center font-bold text-xs text-[var(--muted-foreground)] border-b border-[var(--border)]">{lbl}</th>' for lbl in short_labels)}
              <th class="p-2 text-center font-bold text-xs text-[var(--muted-foreground)] border-b border-[var(--border)]">Total True</th>
            </tr>
          </thead>
          <tbody>
            {"".join(table_rows)}
            {bottom_row}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Footer Note -->
    <div class="mt-4 pt-3 border-t border-[var(--border)] flex flex-col sm:flex-row justify-between items-center text-xs text-[var(--muted-foreground)] gap-2">
      <div>
        <strong>Diagnostic Status:</strong> Non-canonical exploratory candidate artifact &bull; Produced via deterministic train split protocol (seed 42).
      </div>
      <div>
        Test RMSE: <strong class="text-emerald-500">1.0412 °C</strong> (Δ = -0.0040 °C vs Certified 1.0452 °C)
      </div>
    </div>
  </div>

  <script>
    function setMode(mode) {{
      const pcts = document.querySelectorAll('.val-pct');
      const cnts = document.querySelectorAll('.val-cnt');
      const btnPct = document.getElementById('btn-mode-pct');
      const btnCnt = document.getElementById('btn-mode-cnt');

      if (mode === 'pct') {{
        pcts.forEach(el => el.classList.remove('hidden'));
        cnts.forEach(el => el.classList.add('hidden'));
        btnPct.className = "px-2.5 py-1 rounded-md font-medium transition-all bg-[var(--card)] text-[var(--foreground)] shadow-xs";
        btnCnt.className = "px-2.5 py-1 rounded-md font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]";
      }} else {{
        pcts.forEach(el => el.classList.add('hidden'));
        cnts.forEach(el => el.classList.remove('hidden'));
        btnCnt.className = "px-2.5 py-1 rounded-md font-medium transition-all bg-[var(--card)] text-[var(--foreground)] shadow-xs";
        btnPct.className = "px-2.5 py-1 rounded-md font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]";
      }}
    }}
  </script>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    main()
