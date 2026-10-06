import { PipelineStage } from '../types';

export const RECONSTRUCTION_STAGES: PipelineStage[] = [
  {
    id: 'stage-surface',
    stepNumber: 1,
    title: 'Surface Observations',
    subtitle: '7 Satellite Variables',
    description: 'Daily multisatellite surface predictors: SST, SSS, SSH, Current U, Current V, Wind U, Wind V.',
    status: 'idle',
    details: {
      'Channels': '7 physical variables',
      'Sources': 'OSTIA, Copernicus SSS, DUACS, OSCAR, CCMP',
      'Resolution': '0.25° spatial grid'
    }
  },
  {
    id: 'stage-temporal',
    stepNumber: 2,
    title: 'Temporal Context',
    subtitle: '5-Day Causal Window [t-4, ..., t]',
    description: 'Constructs chronological sequence strictly backwards in time. Zero future leakage, clamped at partition boundary.',
    status: 'idle',
    details: {
      'Window T': '5 time steps (days)',
      'Causality': 't-4, t-3, t-2, t-1, t',
      'Purge Protection': 'Clamped at partition start'
    }
  },
  {
    id: 'stage-spatial',
    stepNumber: 3,
    title: 'Spatial Context',
    subtitle: '3 × 3 Spatial Patch (P=3)',
    description: 'Extracts 3×3 neighboring ocean grid cells (~75 km × 75 km area) around the center coordinate.',
    status: 'idle',
    details: {
      'Patch Shape': '3 × 3 grid cells',
      'Physical Span': '~75 km horizontal footprint',
      'Tensor Shape': '[B, T=5, C=7, P=3, P=3]'
    }
  },
  {
    id: 'stage-spatial-enc',
    stepNumber: 4,
    title: 'Spatial CNN Encoder',
    subtitle: 'Time-Distributed Conv2D',
    description: 'Applies Conv2D(7→32→64) with BatchNorm and Adaptive Average Pooling across every temporal slice.',
    status: 'idle',
    details: {
      'Conv Block': 'Conv2D(32) → BN → Conv2D(64) → BN',
      'Pooling': 'AdaptiveAvgPool2D((1,1))',
      'Spatial Token': '64-dimensional feature vector per day'
    }
  },
  {
    id: 'stage-temporal-gru',
    stepNumber: 5,
    title: 'Temporal GRU',
    subtitle: '2-Layer Recurrent Sequence Model',
    description: 'Feeds the sequence of 5 spatial tokens into a 2-layer causal GRU to capture baroclinic wave evolution.',
    status: 'idle',
    details: {
      'Layers': '2 stacked GRU layers',
      'Hidden Size': '128 units',
      'Direction': 'Strictly unidirectional forward in time'
    }
  },
  {
    id: 'stage-embedding',
    stepNumber: 6,
    title: '128-D Latent Embedding',
    subtitle: 'LayerNorm Bottleneck',
    description: 'Final recurrent hidden state normalized with LayerNorm(128). Serves as the compressed ocean state vector.',
    status: 'idle',
    details: {
      'Dimension': '128-dimensional dense vector',
      'Normalization': 'LayerNorm(128)',
      'Representation': 'Joint Spatiotemporal Latent Vector'
    }
  },
  {
    id: 'stage-decoder',
    stepNumber: 7,
    title: '15-Depth Decoder',
    subtitle: 'Subsurface Depth Projection',
    description: 'Multi-layer perceptron (Linear 128→64 → ReLU → Linear 64→15) projecting the latent state into ocean depths.',
    status: 'idle',
    details: {
      'Decoder MLP': 'Linear(128, 64) → ReLU → Linear(64, 15)',
      'Output Channels': '15 vertical ocean depths',
      'Loss Mask': 'Masked MSE isolating bathymetric ocean points'
    }
  },
  {
    id: 'stage-profile',
    stepNumber: 8,
    title: 'Temperature Profile',
    subtitle: 'Reconstructed T(z) [0m – 1000m]',
    description: 'Continuous vertical temperature curve reconstructing the upper mixed layer, thermocline, and abyssal water.',
    status: 'idle',
    details: {
      'Depth Range': '0 m to 1000 m (15 canonical levels)',
      'Column-Averaged Test RMSE': '0.9800 °C (-22.11% vs Climatology)',
      'Status': 'Valid Ocean Reconstruction'
    }
  }
];

export const SIMULATION_STEPS = [
  'Ready',
  'Preparing Inputs',
  'Building 5-Day Context',
  'Extracting 3×3 Spatial Patch',
  'Encoding Spatial Features',
  'Processing Temporal Sequence',
  'Generating 128-D Embedding',
  'Reconstructing 15 Depth Levels',
  'Complete'
];
