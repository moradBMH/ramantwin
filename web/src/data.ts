export type Role = "train" | "fleet";
export type ModelKey = "svm" | "naive" | "robust";
export const MODELS: ModelKey[] = ["svm", "naive", "robust"];
export const MODEL_NAMES: Record<ModelKey, string> = {
  svm: "Linear SVM",
  naive: "Naive CNN",
  robust: "Robust CNN",
};

export interface Instrument {
  id: number;
  name: string;
  role: Role;
}

export interface Stage {
  key: string;
  title: string;
  subtitle: string;
  positions: number[];
  silhouette: { instrument: number; class: number };
}

export interface ShiftScene {
  version: number;
  mock?: boolean;
  classes: string[];
  instruments: Instrument[];
  stages: Stage[];
  points: {
    label: number[];
    instrument: number[];
    patient: number[];
    p_naive: number[];
    p_robust: number[];
    ood: number[];
  };
  spectra: { grid: number[]; index: number[]; raw: number[][]; processed: number[][] };
}

export interface Atlas {
  version: number;
  mock?: boolean;
  x: { param: string; label: string; values: number[] };
  y: { param: string; label: string; values: number[] };
  metric: string;
  models: Partial<Record<ModelKey, Array<Array<number | null>>>>;
  train_envelope: Envelope;
  randomization_envelope?: Envelope | null;
}

export interface Envelope {
  x: [number, number];
  y: [number, number];
}

export interface Metric {
  bacc?: number;
  auc?: number;
}

export interface FleetInstrument extends Instrument {
  params: Record<string, number | string>;
  metrics: Partial<Record<ModelKey, Metric>>;
  abstain_rate?: number;
}

export interface Fleet {
  version: number;
  mock?: boolean;
  instruments: FleetInstrument[];
  summary: Partial<Record<ModelKey, { mean_bacc?: number; worst_bacc?: number; mean_bacc_ci95?: [number, number] | null }>>;
  latency_ms: Record<string, number>;
}

export interface Curve {
  x: number[];
  y: number[];
}

export interface QC {
  flag: "ok" | "warn" | "reject";
  ood_score: number;
  threshold: number;
  reasons: string[];
}

export interface SimulateResponse {
  raw: Curve;
  baseline: Curve;
  processed: Curve;
  truth: { label: number; concentrations?: Record<string, number> };
  predictions: Partial<Record<ModelKey, { p_cancer: number }>>;
  qc: QC;
  timing_ms: Record<string, number>;
}

export interface Scenario {
  key: string;
  title: string;
  params: Record<string, number>;
  response: SimulateResponse;
}

export interface TwinlabSamples {
  version: number;
  mock?: boolean;
  scenarios: Scenario[];
}

const cache = new Map<string, Promise<unknown>>();
const mockFiles = new Set<string>();
const listeners: Array<() => void> = [];

export function onMockChange(fn: () => void): void {
  listeners.push(fn);
}

export function usingMock(): boolean {
  return mockFiles.size > 0;
}

export function loadData<T>(name: string): Promise<T> {
  let p = cache.get(name);
  if (!p) {
    p = fetch(`data/${name}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
        return r.json();
      })
      .then((d: { mock?: boolean }) => {
        if (d.mock) {
          mockFiles.add(name);
          listeners.forEach((fn) => fn());
        }
        return d;
      })
      .catch((err: unknown) => {
        cache.delete(name);
        throw err;
      });
    cache.set(name, p);
  }
  return p as Promise<T>;
}

export const loadShift = () => loadData<ShiftScene>("shift_scene.json");
export const loadAtlas = () => loadData<Atlas>("robustness_atlas.json");
export const loadFleet = () => loadData<Fleet>("fleet.json");
export const loadSamples = () => loadData<TwinlabSamples>("twinlab_samples.json");
