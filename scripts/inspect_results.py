import json
import numpy as np

with open('results/B0.json') as f: b0 = json.load(f)
with open('results/B0b.json') as f: b0b = json.load(f)
with open('results/B1.json') as f: b1 = json.load(f)

print('=== VALIDATION SPLIT ===')
for name, d in [('B0 (Day 0)', b0), ('B0b (Day 252)', b0b), ('B1 (Climatology)', b1)]:
    v = d['validation']
    corrs = [x['corr'] for x in v['depth_breakdown']]
    print(f"{name:20s} | RMSE(unw)={v['overall']['rmse']:.4f} | RMSE(w)={v['weighted_overall']['rmse']:.4f} | MAE={v['overall']['mae']:.4f} | Bias={v['overall']['bias']:+.4f} | Corr={np.mean(corrs):.4f} | R2={v['overall']['r2']:.4f}")

print('\n=== TEST SPLIT ===')
for name, d in [('B0 (Day 0)', b0), ('B0b (Day 252)', b0b), ('B1 (Climatology)', b1)]:
    t = d['test']
    corrs = [x['corr'] for x in t['depth_breakdown']]
    ci = t['bootstrap_ci_95']
    print(f"{name:20s} | RMSE(unw)={t['overall']['rmse']:.4f} [{ci['unweighted_rmse']['ci_95_low']:.4f}, {ci['unweighted_rmse']['ci_95_high']:.4f}] | RMSE(w)={t['weighted_overall']['rmse']:.4f} [{ci['weighted_rmse']['ci_95_low']:.4f}, {ci['weighted_rmse']['ci_95_high']:.4f}] | MAE={t['overall']['mae']:.4f} | Bias={t['overall']['bias']:+.4f} | Corr={np.mean(corrs):.4f} | R2={t['overall']['r2']:.4f}")

print('\n=== DEPTH-WISE TEST RMSE (°C) ===')
header = f"{'Depth':>6s} | {'B0':>8s} | {'B0b':>8s} | {'B1':>8s} | {'B0 Corr':>8s} | {'B0b Corr':>8s} | {'B1 Corr':>8s} | {'B0 R2':>8s} | {'B0b R2':>8s} | {'B1 R2':>8s}"
print(header)
print('-' * len(header))
for i in range(15):
    d_m = b0['test']['depth_breakdown'][i]['depth_m']
    r0 = b0['test']['depth_breakdown'][i]['rmse']
    r0b = b0b['test']['depth_breakdown'][i]['rmse']
    r1 = b1['test']['depth_breakdown'][i]['rmse']
    c0 = b0['test']['depth_breakdown'][i]['corr']
    c0b = b0b['test']['depth_breakdown'][i]['corr']
    c1 = b1['test']['depth_breakdown'][i]['corr']
    r2_0 = b0['test']['depth_breakdown'][i]['r2']
    r2_0b = b0b['test']['depth_breakdown'][i]['r2']
    r2_1 = b1['test']['depth_breakdown'][i]['r2']
    print(f"{d_m:6.0f} | {r0:8.4f} | {r0b:8.4f} | {r1:8.4f} | {c0:8.4f} | {c0b:8.4f} | {c1:8.4f} | {r2_0:8.4f} | {r2_0b:8.4f} | {r2_1:8.4f}")

print('\n=== REGIONAL BREAKDOWN (TEST SPLIT, WEIGHTED RMSE) ===')
for reg in ['full_domain', 'arabian_sea', 'bay_of_bengal']:
    w0 = b0['test']['regions'][reg]['weighted_rmse']
    w0b = b0b['test']['regions'][reg]['weighted_rmse']
    w1 = b1['test']['regions'][reg]['weighted_rmse']
    print(f"{reg:16s} | B0: {w0:.4f}°C | B0b: {w0b:.4f}°C | B1: {w1:.4f}°C")

print('\n=== SEASONAL BREAKDOWN (TEST SPLIT, WEIGHTED RMSE) ===')
for s in ['late_fall_nov', 'early_winter_dec']:
    w0 = b0['test']['seasons'][s]['weighted_rmse']
    w0b = b0b['test']['seasons'][s]['weighted_rmse']
    w1 = b1['test']['seasons'][s]['weighted_rmse']
    print(f"{s:18s} | B0: {w0:.4f}°C | B0b: {w0b:.4f}°C | B1: {w1:.4f}°C")
