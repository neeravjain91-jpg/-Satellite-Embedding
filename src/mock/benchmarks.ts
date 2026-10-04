import { BenchmarkModel } from '../types';

export const BENCHMARK_MODELS: BenchmarkModel[] = [
  {
    id: 'B0',
    name: 'Day 0 Persistence',
    architecture: 'Frozen initial observation state t=0',
    context: 'Initial temporal boundary state',
    family: 'Persistence',
    rmse: 1.5220,
    improvementPct: -20.97,
    parameters: 0,
    status: 'Persistence'
  },
  {
    id: 'B0b',
    name: 'Day 252 Persistence',
    architecture: 'Frozen train boundary observation state',
    context: 'Train-split terminal state',
    family: 'Persistence',
    rmse: 1.7287,
    improvementPct: -37.39,
    parameters: 0,
    status: 'Persistence'
  },
  {
    id: 'B1',
    name: 'Spatial-Depth Climatology',
    architecture: 'Historical domain vertical profile mean',
    context: 'Spatial-Depth Train Climatology',
    family: 'Climatology Reference',
    rmse: 1.2582,
    improvementPct: 0.0,
    parameters: 0,
    status: 'Baseline'
  },
  {
    id: 'B2',
    name: 'Multi-Output Ridge',
    architecture: 'Linear L2-regularized multi-depth regression',
    context: 'Pointwise (7 surface variables)',
    family: 'Linear Regularized',
    rmse: 1.0295,
    improvementPct: 18.18,
    parameters: 120,
    status: 'Baseline'
  },
  {
    id: 'B3',
    name: 'Multi-Depth Random Forest',
    architecture: 'Bagging ensemble of decision trees',
    context: 'Pointwise (7 surface variables)',
    family: 'Bagging Ensemble',
    rmse: 1.0452,
    improvementPct: 16.93,
    parameters: 10255,
    status: 'Baseline'
  },
  {
    id: 'B4',
    name: 'Gradient Boosting (LightGBM)',
    architecture: 'Histogram gradient boosted trees (50 estimators)',
    context: 'Pointwise (7 surface variables)',
    family: 'Boosting Ensemble',
    rmse: 1.0288,
    improvementPct: 18.23,
    parameters: 750,
    status: 'Baseline'
  },
  {
    id: 'B5',
    name: 'Pointwise MLP',
    architecture: 'Feedforward Neural Network (128-128-64)',
    context: 'Pointwise (7 surface variables)',
    family: 'Deep Neural (Pointwise)',
    rmse: 1.5524,
    improvementPct: -23.38,
    parameters: 26767,
    status: 'Ablation'
  },
  {
    id: 'B6',
    name: 'Spatial CNN',
    architecture: 'Time-distributed Conv2D on 3×3 patch',
    context: 'Spatial (3×3 patch, 7 channels)',
    family: 'Spatial Convolutional',
    rmse: 1.2702,
    improvementPct: -0.95,
    parameters: 30991,
    status: 'Ablation'
  },
  {
    id: 'B7',
    name: 'Temporal GRU',
    architecture: '2-layer causal Gated Recurrent Unit',
    context: 'Temporal (5-day causal window)',
    family: 'Sequential Recurrent',
    rmse: 1.5320,
    improvementPct: -21.76,
    parameters: 44111,
    status: 'Ablation'
  },
  {
    id: 'B8',
    name: 'Spatiotemporal Embedding Model',
    architecture: 'Conv2D + 2L-GRU + 128D LayerNorm + Depth Decoder',
    context: 'Spatiotemporal (5-day × 3×3 patch)',
    family: 'Spatiotemporal Deep Learning',
    rmse: 0.9800,
    improvementPct: 22.11,
    parameters: 203791,
    status: 'Champion',
    highlight: true
  }
];

export const REGIONAL_METRICS = [
  { region: 'Full Domain', rmse: 0.9642, sampleCount: 601550, description: '5°N–30°N, 45°E–105°E complete test partition' },
  { region: 'Arabian Sea', rmse: 1.0907, sampleCount: 284120, description: 'Western high-salinity evaporation basin' },
  { region: 'Bay of Bengal', rmse: 0.6775, sampleCount: 221840, description: 'Stratified freshwater river plume regime' },
  { region: 'Equatorial Indian Ocean', rmse: 0.9812, sampleCount: 95590, description: 'Cross-equatorial jet & warm pool' }
];

export const SEASONAL_METRICS = [
  { period: 'Late Fall', dateRange: 'Nov 09 – Nov 30, 2020', rmse: 1.0059, description: 'Transition monsoon wind reversal' },
  { period: 'Early Winter', dateRange: 'Dec 01 – Dec 31, 2020', rmse: 0.9060, description: 'North-east winter cooling & stratification' }
];

export const B8_EXACT_DEPTH_TABLE = [
  { depth: 0, rmse: 0.4369, mae: 0.3213 },
  { depth: 5, rmse: 0.4381, mae: 0.3316 },
  { depth: 10, rmse: 0.4635, mae: 0.3479 },
  { depth: 20, rmse: 0.5962, mae: 0.4277 },
  { depth: 30, rmse: 0.8607, mae: 0.6312 },
  { depth: 50, rmse: 1.3493, mae: 0.9611 },
  { depth: 75, rmse: 1.8110, mae: 1.3867 },
  { depth: 100, rmse: 1.7651, mae: 1.3604 },
  { depth: 125, rmse: 1.5022, mae: 1.1748 },
  { depth: 150, rmse: 1.3650, mae: 1.0873 },
  { depth: 200, rmse: 1.1834, mae: 0.8985 },
  { depth: 300, rmse: 0.9845, mae: 0.7284 },
  { depth: 500, rmse: 0.6870, mae: 0.5115 },
  { depth: 700, rmse: 0.6619, mae: 0.4735 },
  { depth: 1000, rmse: 0.5959, mae: 0.4442 }
];

export function getBenchmarkResults(): BenchmarkModel[] {
  return BENCHMARK_MODELS;
}
