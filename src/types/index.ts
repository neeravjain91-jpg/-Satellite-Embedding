export type NavigationTab = 
  | 'dashboard'
  | 'explorer'
  | 'reconstruction'
  | 'depth-profile'
  | 'embedding'
  | 'benchmarks'
  | 'validation'
  | 'methodology';

export type CanonicalDepth = 
  | 0 
  | 5 
  | 10 
  | 20 
  | 30 
  | 50 
  | 75 
  | 100 
  | 125 
  | 150 
  | 200 
  | 300 
  | 500 
  | 700 
  | 1000;

export interface SurfaceVariable {
  id: string;
  name: string;
  symbol: string;
  value: number;
  unit: string;
  trend: number[];
  status: 'normal' | 'elevated' | 'depressed';
  description: string;
}

export interface OceanLocation {
  lat: number;
  lon: number;
  date: string;
  depth: CanonicalDepth;
  region: string;
  gridPoint: string;
}

export interface DepthProfilePoint {
  depth: CanonicalDepth;
  reconstructed: number;
  reference: number;
  error: number;
  climatology: number;
  ridge: number;
}

export interface BenchmarkModel {
  id: string;
  name: string;
  architecture: string;
  context: string;
  family: string;
  rmse: number;
  improvementPct: number;
  parameters: number;
  status: 'Baseline' | 'Champion' | 'Ablation' | 'Persistence';
  highlight?: boolean;
}

export interface PipelineStage {
  id: string;
  stepNumber: number;
  title: string;
  subtitle: string;
  description: string;
  status: 'idle' | 'processing' | 'complete';
  details: Record<string, string | number>;
}

export interface EmbeddingPoint {
  id: string;
  x: number;
  y: number;
  cluster: string;
  lat: number;
  lon: number;
  sst: number;
  ssh: number;
  thermoclineDepth: number;
  date: string;
}
