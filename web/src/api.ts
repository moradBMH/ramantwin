import type { ModelKey, SimulateResponse } from "./data";

export interface ApiConfig {
  params: Record<string, number | string>;
  ranges: Record<string, [number, number]>;
  grid: { start: number; stop: number; step: number };
  classes: string[];
  models: ModelKey[];
}

export interface SimulateRequest {
  params: Record<string, number>;
  label: number;
  seed: number;
}

async function withTimeout(url: string, init: RequestInit, ms: number): Promise<Response> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

export async function getConfig(): Promise<ApiConfig | null> {
  try {
    const health = await withTimeout("api/health", {}, 1200);
    if (!health.ok) return null;
    const r = await withTimeout("api/config", {}, 3000);
    return r.ok ? ((await r.json()) as ApiConfig) : null;
  } catch {
    return null;
  }
}

export async function simulate(req: SimulateRequest): Promise<SimulateResponse> {
  const r = await withTimeout(
    "api/simulate",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) },
    8000,
  );
  if (!r.ok) throw new Error(`simulate: HTTP ${r.status}`);
  return (await r.json()) as SimulateResponse;
}
