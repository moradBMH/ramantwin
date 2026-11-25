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

export async function checkHealth(): Promise<boolean> {
  try {
    return (await withTimeout("api/health", {}, 1500)).ok;
  } catch {
    return false;
  }
}

export async function getConfig(): Promise<ApiConfig | null> {
  if (!(await checkHealth())) return null;
  try {
    const r = await withTimeout("api/config", {}, 3000);
    return r.ok ? ((await r.json()) as ApiConfig) : null;
  } catch {
    return null;
  }
}

export class SimulateError extends Error {}

export async function simulate(req: SimulateRequest): Promise<SimulateResponse> {
  let r: Response;
  try {
    r = await withTimeout(
      "api/simulate",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) },
      8000,
    );
  } catch (err) {
    throw new SimulateError(err instanceof DOMException && err.name === "AbortError" ? "timeout" : "network error");
  }
  if (!r.ok) throw new SimulateError(`HTTP ${r.status}`);
  return (await r.json()) as SimulateResponse;
}
