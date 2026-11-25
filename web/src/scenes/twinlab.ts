import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { disposeObject, frameFreeArea, type View } from "../gl/engine";
import { CSS } from "../gl/colors";
import { getConfig, simulate } from "../api";
import { MODEL_NAMES, loadSamples, type ModelKey, type Scenario, type SimulateResponse } from "../data";
import { fmt, h, q } from "../ui/dom";
import { LinePlot } from "../ui/plot";
import { buildInstrument } from "./instrument";
import type { ViewContext } from "../main";

interface SliderMeta {
  label: string;
  unit: string;
  digits: number;
}

const META: Record<string, SliderMeta> = {
  laser_drift_nm: { label: "Laser drift", unit: "nm", digits: 2 },
  resolution_cm1: { label: "Resolution", unit: "cm⁻¹", digits: 1 },
  fluorescence_ratio: { label: "Fluorescence / Raman", unit: "×", digits: 1 },
  signal_e_per_s: { label: "Signal", unit: "e⁻/s", digits: 0 },
  exposure_s: { label: "Exposure", unit: "s", digits: 2 },
  irf_tilt: { label: "IRF tilt", unit: "", digits: 2 },
  ambient_light: { label: "Ambient light", unit: "", digits: 2 },
  cosmic_rate: { label: "Cosmic rays", unit: "/frame", digits: 2 },
};

const DEFAULT_RANGES: Record<string, [number, number]> = {
  laser_drift_nm: [-1.5, 1.5],
  resolution_cm1: [4, 14],
  fluorescence_ratio: [0.5, 60],
  signal_e_per_s: [100, 10000],
  exposure_s: [0.1, 5],
  irf_tilt: [-0.5, 0.5],
  ambient_light: [0, 1],
  cosmic_rate: [0, 2],
};

const DEFAULT_PARAMS: Record<string, number> = {
  laser_drift_nm: 0,
  resolution_cm1: 6,
  fluorescence_ratio: 5,
  signal_e_per_s: 2000,
  exposure_s: 1,
  irf_tilt: 0,
  ambient_light: 0,
  cosmic_rate: 0.05,
};

const BANDS = [
  { x: 1003, label: "Phe" },
  { x: 1156, label: "Car" },
  { x: 1302, label: "CH₂" },
  { x: 1445, label: "CH₂" },
  { x: 1520, label: "Car" },
  { x: 1655, label: "Amide I" },
];

const pretty = (key: string) => META[key]?.label ?? key.replace(/_/g, " ");

function gaugeSVG(key: string): string {
  return `<svg viewBox="0 0 120 70">
    <defs><linearGradient id="gg-${key}" x1="0" x2="1">
      <stop offset="0" stop-color="${CSS.normal}"/><stop offset="1" stop-color="${CSS.cancer}"/></linearGradient></defs>
    <path d="M10 60 A50 50 0 0 1 110 60" stroke="url(#gg-${key})" stroke-width="9" fill="none" opacity="0.18" stroke-linecap="round"/>
    <path class="arc" d="M10 60 A50 50 0 0 1 110 60" stroke="url(#gg-${key})" stroke-width="9" fill="none"
      stroke-linecap="round" stroke-dasharray="157.1" stroke-dashoffset="157.1" style="transition:stroke-dashoffset 600ms cubic-bezier(.22,1,.36,1)"/>
    <line x1="60" y1="4" x2="60" y2="15" stroke="#e9edf4" stroke-width="1.5" opacity="0.45"/>
    <g class="needle"><line x1="60" y1="60" x2="60" y2="20" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="60" cy="60" r="4.5" fill="#fff"/></g>
  </svg>`;
}

export async function createTwinLab(ctx: ViewContext): Promise<View> {
  const [samples, config] = await Promise.all([loadSamples().catch(() => null), getConfig()]);
  const scenarios: Scenario[] = samples?.scenarios ?? [];
  let online = config !== null;

  const ranges: Record<string, [number, number]> = config?.ranges ?? DEFAULT_RANGES;
  const keys = Object.keys(ranges);
  const defaults: Record<string, number> = {};
  for (const k of keys) {
    const v = config?.params[k];
    defaults[k] = typeof v === "number" ? v : (DEFAULT_PARAMS[k] ?? (ranges[k][0] + ranges[k][1]) / 2);
  }
  const params: Record<string, number> = { ...defaults };
  let label = 1;
  let seed = 1;
  let scenarioIdx = -1;
  const classes = config?.classes ?? ["normal", "cancer"];
  const models: ModelKey[] = (config?.models ?? ["svm", "naive", "robust"]).filter((m) => m in MODEL_NAMES);

  // ---------------------------------------------------------------- 3D
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0.6, 6.4, 9.2);
  const controls = new OrbitControls(camera, ctx.engine.renderer.domElement);
  controls.target.set(0.2, 0.4, -0.1);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 6;
  controls.maxDistance = 17;
  controls.minPolarAngle = 0.35;
  controls.maxPolarAngle = 1.3;
  controls.minAzimuthAngle = -1.0;
  controls.maxAzimuthAngle = 1.0;

  scene.add(new THREE.HemisphereLight(0x8fa3c8, 0x05070b, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(3, 8, 6);
  scene.add(key);
  const rimLight = new THREE.DirectionalLight(0xb9a5ff, 0.8);
  rimLight.position.set(-6, 3, -6);
  scene.add(rimLight);

  const inst = buildInstrument();
  scene.add(inst.group);
  inst.setSample(label);

  // ---------------------------------------------------------------- UI
  const root = h("div");
  root.innerHTML = `
    <div class="panel tl-left">
      <div class="controls-row" style="justify-content:space-between;margin-bottom:14px">
        <p class="eyebrow" style="margin:0">Virtual instrument</p>
        <span class="chip mode-chip"></span>
      </div>
      <div class="sliders"></div>
      <p class="eyebrow" style="margin-top:6px">Sample</p>
      <div class="controls-row" style="margin-bottom:16px">
        <div class="seg label-seg"></div>
        <button class="btn small reroll" title="New patient spectrum (R)">Re-roll</button>
      </div>
      <p class="eyebrow">Scenarios</p>
      <div class="scenarios"></div>
    </div>
    <div class="panel tl-right">
      <div class="truth"><span class="muted">Ground truth</span><strong class="truth-val">–</strong></div>
      <div class="gauges"></div>
      <div style="border-top:1px solid var(--border);margin:14px 0 12px"></div>
      <p class="eyebrow">Quality control</p>
      <div class="qc"><span class="qc-badge">–</span><span class="qc-reasons"></span></div>
      <div class="ood-bar"><i></i><b style="left:50%"></b></div>
      <div class="gradient-labels"><span>OOD score <span class="mono ood-val"></span></span><span>threshold</span></div>
      <div style="margin-top:14px"><span class="chip timing mono">–</span></div>
    </div>
    <div class="panel tl-bottom">
      <div>
        <div class="plot-title"><span class="key"><i style="background:${CSS.text}"></i>Raw (IRF-corrected)</span>
          <span class="key"><i style="background:${CSS.accent}"></i>BubbleFill baseline</span></div>
        <div class="plot plot-raw"></div>
      </div>
      <div>
        <div class="plot-title"><span class="key"><i class="pk" style="background:${CSS.cancer}"></i>ORPL processed (SNV)</span>
          <span class="muted" style="margin-left:auto">Raman shift (cm⁻¹)</span></div>
        <div class="plot plot-proc"></div>
      </div>
    </div>
    <div class="hint"><kbd>Space</kbd> next scenario · <kbd>N</kbd> normal/cancer · <kbd>R</kbd> re-roll</div>`;
  ctx.ui.appendChild(root);

  const rawPlot = new LinePlot(q(root, ".plot-raw"), { xRange: [450, 1900], xTicks: [600, 1000, 1400, 1800] });
  const procPlot = new LinePlot(q(root, ".plot-proc"), {
    xRange: [600, 1800],
    xTicks: [600, 800, 1000, 1200, 1400, 1600, 1800],
    markers: BANDS,
    zeroLine: true,
  });

  // Sliders.
  const sliderEls = new Map<string, { input: HTMLInputElement; val: HTMLElement }>();
  const isLog = (k: string) => ranges[k][0] > 0 && ranges[k][1] / ranges[k][0] >= 20;
  const toValue = (k: string, pos: number) => {
    const [lo, hi] = ranges[k];
    const u = pos / 1000;
    return isLog(k) ? Math.exp(Math.log(lo) + u * (Math.log(hi) - Math.log(lo))) : lo + u * (hi - lo);
  };
  const toPos = (k: string, v: number) => {
    const [lo, hi] = ranges[k];
    const c = Math.min(hi, Math.max(lo, v));
    const u = isLog(k) ? (Math.log(c) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (c - lo) / (hi - lo);
    return Math.round(u * 1000);
  };
  const sliders = q(root, ".sliders");
  for (const k of keys) {
    const el = h(
      "div",
      { class: "slider" },
      `<div class="slider-head"><span>${pretty(k)}</span><span class="mono"></span></div><input type="range" min="0" max="1000" step="1"/>`,
    );
    const input = q<HTMLInputElement>(el, "input");
    const val = q(el, ".mono");
    input.addEventListener("input", () => {
      params[k] = toValue(k, Number(input.value));
      scenarioIdx = -1;
      renderScenarioButtons();
      syncSlider(k);
      if (k === "laser_drift_nm") inst.setDrift(params[k]);
      schedule();
    });
    sliders.appendChild(el);
    sliderEls.set(k, { input, val });
  }

  function syncSlider(k: string): void {
    const s = sliderEls.get(k);
    if (!s) return;
    const m = META[k];
    const v = params[k];
    const pos = toPos(k, v);
    if (Number(s.input.value) !== pos) s.input.value = String(pos);
    s.input.style.setProperty("--fill", `${pos / 10}%`);
    const digits = m?.digits ?? 2;
    const sign = k === "laser_drift_nm" && v > 0 ? "+" : "";
    s.val.textContent = `${sign}${v.toFixed(digits)}${m?.unit ? ` ${m.unit}` : ""}`;
  }

  // Sample selector.
  const labelSeg = q(root, ".label-seg");
  classes.forEach((c, i) => {
    const b = h("button", { "data-l": String(i) }, c);
    b.addEventListener("click", () => setLabel(i));
    labelSeg.appendChild(b);
  });
  function setLabel(l: number): void {
    label = l;
    inst.setSample(l);
    labelSeg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.classList.toggle("active", Number(b.dataset.l) === l));
    if (online) schedule();
  }
  q(root, ".reroll").addEventListener("click", () => reroll());
  function reroll(): void {
    seed = 1 + Math.floor(Math.random() * 1e6);
    if (online) schedule();
  }

  // Scenarios.
  const scenEl = q(root, ".scenarios");
  function renderScenarioButtons(): void {
    scenEl.querySelectorAll<HTMLButtonElement>("button").forEach((b, i) => b.classList.toggle("active", i === scenarioIdx));
  }
  scenarios.forEach((s, i) => {
    const b = h("button", { class: "btn small" }, s.title);
    b.addEventListener("click", () => applyScenario(i));
    scenEl.appendChild(b);
  });
  if (scenarios.length === 0) scenEl.innerHTML = '<span class="empty">No recorded scenarios.</span>';

  function applyScenario(i: number): void {
    const s = scenarios[i];
    if (!s) return;
    scenarioIdx = i;
    renderScenarioButtons();
    for (const k of keys) params[k] = k in s.params ? s.params[k] : defaults[k];
    keys.forEach(syncSlider);
    inst.setDrift(params.laser_drift_nm ?? 0);
    label = s.response.truth.label;
    inst.setSample(label);
    labelSeg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.classList.toggle("active", Number(b.dataset.l) === label));
    if (online) schedule();
    else render(s.response);
  }

  function setOnline(v: boolean): void {
    online = v;
    const chip = q(root, ".mode-chip");
    chip.className = `chip mode-chip ${v ? "ok" : "warn"}`;
    chip.innerHTML = v ? '<span class="dot"></span>Live · API' : '<span class="dot"></span>Offline · recorded';
    chip.title = v ? "Spectra are simulated live by the Python twin" : "API unreachable: showing recorded scenarios";
    sliderEls.forEach((s) => (s.input.disabled = !v));
    q<HTMLButtonElement>(root, ".reroll").disabled = !v;
    labelSeg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.disabled = !v));
  }

  // Gauges.
  const gauges = q(root, ".gauges");
  for (const m of models) {
    gauges.appendChild(
      h("div", { class: "gauge", "data-m": m }, `${gaugeSVG(m)}<div class="val">–</div><div class="name">${MODEL_NAMES[m]}</div><div class="verdict"></div>`),
    );
  }

  function render(r: SimulateResponse): void {
    const truth = r.truth.label;
    const tv = q(root, ".truth-val");
    tv.textContent = classes[truth] ?? String(truth);
    tv.className = `truth-val ${truth ? "c-cancer" : "c-normal"}`;
    for (const m of models) {
      const g = q(gauges, `[data-m="${m}"]`);
      const p = r.predictions[m]?.p_cancer;
      if (p === undefined) continue;
      q<SVGPathElement>(g, ".arc").style.strokeDashoffset = String(157.1 * (1 - p));
      q<SVGGElement>(g, ".needle").style.transform = `rotate(${-90 + 180 * p}deg)`;
      q(g, ".val").textContent = p.toFixed(2);
      const pred = p >= 0.5 ? 1 : 0;
      const right = pred === truth;
      const v = q(g, ".verdict");
      v.className = `verdict ${right ? "right" : "wrong"}`;
      v.textContent = `${classes[pred] ?? pred} ${right ? "✓" : "✗"}`;
    }

    const qc = r.qc;
    const badge = q(root, ".qc-badge");
    badge.className = `qc-badge ${qc.flag}`;
    badge.textContent = qc.flag.toUpperCase();
    q(root, ".qc-reasons").textContent = qc.reasons.length ? qc.reasons.join(" · ") : "within the validated envelope";
    const frac = Math.min(1, qc.ood_score / (2 * (qc.threshold || 1)));
    const bar = q(root, ".ood-bar i");
    bar.style.width = `${frac * 100}%`;
    bar.style.background = qc.flag === "ok" ? CSS.ok : qc.flag === "warn" ? CSS.warn : CSS.reject;
    q(root, ".ood-val").textContent = `${fmt(qc.ood_score, 1)} / ${fmt(qc.threshold, 1)}`;

    const t = r.timing_ms;
    const parts = [`ORPL ${fmt(t.orpl, 1)} ms`, `inference ${fmt(t.inference, 1)} ms`];
    q(root, ".timing").textContent = parts.join(" · ");

    rawPlot.set([
      { x: r.raw.x, y: r.raw.y, color: CSS.text, width: 1.3, fill: true },
      { x: r.baseline.x, y: r.baseline.y, color: CSS.accent, width: 1.6, dash: [5, 4] },
    ]);
    const pc = truth ? CSS.cancer : CSS.normal;
    q(root, ".pk").style.background = pc;
    procPlot.set([{ x: r.processed.x, y: r.processed.y, color: pc, width: 1.6 }]);
    inst.setSpectrum(r.raw.y);
  }

  // Live simulation: debounced, latest request wins.
  let timer = 0;
  let inflight = false;
  let pending = false;
  function schedule(): void {
    window.clearTimeout(timer);
    timer = window.setTimeout(run, 50);
  }
  async function run(): Promise<void> {
    if (!online) return;
    if (inflight) {
      pending = true;
      return;
    }
    inflight = true;
    try {
      render(await simulate({ params: { ...params }, label, seed }));
    } catch (err) {
      console.warn(err);
      setOnline(false);
      if (scenarios.length) applyScenario(Math.max(0, scenarioIdx));
    } finally {
      inflight = false;
      if (pending && online) {
        pending = false;
        void run();
      }
    }
  }

  keys.forEach(syncSlider);
  setOnline(online);
  setLabel(label);
  inst.setDrift(params.laser_drift_nm ?? 0);
  if (online) schedule();
  else if (scenarios.length) applyScenario(0);

  const stepScenario = (dir: 1 | -1): boolean => {
    const next = scenarioIdx + dir;
    if (next < 0 || next >= scenarios.length) return false;
    applyScenario(next);
    return true;
  };

  return {
    scene,
    camera,
    bloom: { strength: 0.9, radius: 0.5, threshold: 0.55 },
    update(dt, t) {
      controls.update(dt);
      inst.update(dt, t);
    },
    resize(w, hgt) {
      const narrow = w < 1280;
      frameFreeArea(camera, w, hgt, { left: narrow ? 306 : 346, right: narrow ? 346 : 386, top: 70, bottom: 286 }, 1.08);
    },
    next: () => stepScenario(1),
    prev: () => stepScenario(-1),
    onKey(e) {
      if (e.key === "n" && online) {
        setLabel(label ? 0 : 1);
        return true;
      }
      if (e.key === "r") {
        reroll();
        return true;
      }
      return false;
    },
    dispose() {
      window.clearTimeout(timer);
      online = false;
      controls.dispose();
      rawPlot.dispose();
      procPlot.dispose();
      disposeObject(scene);
      root.remove();
    },
  };
}

