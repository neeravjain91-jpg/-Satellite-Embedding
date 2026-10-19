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

export interface AvailableDateOption {
  date: string;
  label: string;
}

export const PUBLISHED_BENCHMARK_DATES: AvailableDateOption[] = [
  { date: '2020-12-31', label: '31 Dec 2020 (Test End)' },
  { date: '2020-12-15', label: '15 Dec 2020 (Test Mid)' },
  { date: '2020-12-01', label: '01 Dec 2020 (Test Milestone)' },
  { date: '2020-11-22', label: '22 Nov 2020 (Test Sequence)' },
  { date: '2020-11-20', label: '20 Nov 2020 (Late Fall Test)' },
  { date: '2020-11-15', label: '15 Nov 2020 (Test Sequence)' },
  { date: '2020-11-09', label: '09 Nov 2020 (Test Start)' },
  { date: '2020-11-02', label: '02 Nov 2020 (Validation End)' },
  { date: '2020-10-15', label: '15 Oct 2020 (Validation Mid)' },
  { date: '2020-10-01', label: '01 Oct 2020 (Post-Monsoon Val)' },
  { date: '2020-09-16', label: '16 Sep 2020 (Validation Start)' },
  { date: '2020-09-01', label: '01 Sep 2020 (Late Summer Train)' },
  { date: '2020-07-15', label: '15 Jul 2020 (SW Monsoon Peak)' },
  { date: '2020-04-15', label: '15 Apr 2020 (Spring Transition)' },
  { date: '2020-01-15', label: '15 Jan 2020 (Winter NE Monsoon)' },
];
