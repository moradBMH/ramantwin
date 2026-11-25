// Generates contract-valid mock files in public/data/ (docs/architecture.md section 8).
// They tell the intended story and are overwritten later by the real exports.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "data");

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(20261124);
const uniform = (a, b) => a + (b - a) * rand();
const gauss = () => {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
};
const r3 = (v) => Math.round(v * 1000) / 1000;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const arange = (a, b, step) => {
  const out = [];
  for (let v = a; v <= b + step / 2; v += step) out.push(r3(v));
  return out;
};

// Tissue bands (cm-1, FWHM, height, component). Class trends follow the literature:
// lower carotenoids and lipids, higher nucleic acids and proteins in cancer.
const BANDS = [
  [853, 16, 0.22, "protein"],
  [936, 22, 0.18, "protein"],
  [1003, 8, 0.55, "protein"],
  [1032, 10, 0.14, "protein"],
  [1085, 18, 0.18, "nucleic_acid"],
  [1127, 15, 0.16, "lipid"],
  [1156, 10, 0.3, "carotenoid"],
  [1250, 36, 0.32, "protein"],
  [1300, 24, 0.42, "lipid"],
  [1340, 24, 0.24, "nucleic_acid"],
  [1445, 26, 0.8, "lipid"],
  [1520, 12, 0.36, "carotenoid"],
  [1578, 14, 0.12, "nucleic_acid"],
  [1605, 12, 0.14, "protein"],
  [1655, 30, 0.72, "protein"],
  [1745, 18, 0.16, "lipid"],
];

function concentrations(label) {
  const c = label === 1;
  const v = (mean, sd, lo, hi) => r3(clamp(mean + sd * gauss(), lo, hi));
  return {
    protein: v(c ? 1.18 : 1.0, 0.1, 0.5, 1.6),
    lipid: v(c ? 0.78 : 1.0, 0.12, 0.3, 1.6),
    nucleic_acid: v(c ? 1.45 : 1.0, 0.15, 0.4, 2.2),
    carotenoid: v(c ? 0.45 : 1.0, 0.15, 0.05, 1.8),
    cholesterol: v(c ? 0.9 : 1.0, 0.1, 0.3, 1.6),
    collagen: v(c ? 0.85 : 1.0, 0.1, 0.3, 1.6),
    heme: v(c ? 1.1 : 1.0, 0.1, 0.3, 1.6),
  };
}

function raman(x, conc, shift = 0, broaden = 1) {
  return x.map((xv) => {
    let s = 0;
    for (const [c0, w0, h, comp] of BANDS) {
      const w = (w0 * broaden) / 2;
      const d = xv - (c0 + shift);
      s += (h * conc[comp] * w * w) / (d * d + w * w);
    }
    return s;
  });
}

function fluorescence(x, ratio, tilt = 0) {
  return x.map((xv) => {
    const u = (xv - 450) / 1450;
    return ratio * 0.55 * (0.55 + 0.75 * Math.exp(-1.6 * u) + 0.25 * Math.sin(1.7 * u)) * (1 + tilt * (u - 0.5));
  });
}

function snv(y) {
  const m = y.reduce((a, b) => a + b, 0) / y.length;
  const sd = Math.sqrt(y.reduce((a, b) => a + (b - m) * (b - m), 0) / y.length) || 1;
  return y.map((v) => (v - m) / sd);
}

function smooth(y, half) {
  return y.map((_, i) => {
    let s = 0;
    let n = 0;
    for (let k = Math.max(0, i - half); k <= Math.min(y.length - 1, i + half); k++) {
      s += y[k];
      n++;
    }
    return s / n;
  });
}

// ---------------------------------------------------------------- instruments
const INSTRUMENTS = [
  ["MTL-01", "train"],
  ["MTL-02", "train"],
  ...Array.from({ length: 10 }, (_, i) => [`SN-${1001 + i}`, "fleet"]),
];
const SEVERITY = [0.04, 0.06, 0.28, 0.55, 0.72, 0.4, 0.85, 0.5, 0.95, 0.66, 0.33, 0.78];

const instruments = INSTRUMENTS.map(([name, role], id) => {
  const d = SEVERITY[id];
  const sign = id % 2 === 0 ? 1 : -1;
  const params = {
    name,
    laser_drift_nm: r3(role === "train" ? 0.02 * sign : sign * d * uniform(0.6, 1.3)),
    resolution_cm1: r3(role === "train" ? 6 + 0.2 * gauss() : 6 + d * uniform(1, 5)),
    fluorescence_ratio: r3(role === "train" ? uniform(4, 6) : 5 * Math.exp(d * uniform(-0.8, 1.8))),
    irf_tilt: r3(role === "train" ? 0.02 * gauss() : 0.35 * d * gauss()),
    signal_e_per_s: Math.round(role === "train" ? 2000 : 2000 * Math.exp(-d * uniform(0, 1.2))),
    exposure_s: 1.0,
    ambient_light: r3(role === "train" ? 0 : d > 0.8 ? 0.15 : 0),
    cosmic_rate: 0.05,
  };
  return { id, name, role, severity: d, params };
});

// Directions on a Fibonacci sphere for the raw instrument clusters.
const centers = instruments.map((_, i) => {
  const n = instruments.length;
  const y = 1 - (2 * (i + 0.5)) / n;
  const r = Math.sqrt(1 - y * y);
  const t = i * Math.PI * (3 - Math.sqrt(5));
  return [Math.cos(t) * r * 0.66, y * 0.6, Math.sin(t) * r * 0.66];
});
const driftDirs = instruments.map(() => {
  const a = uniform(0, 2 * Math.PI);
  return [Math.cos(a), Math.sin(a)];
});
const biasDirs = instruments.map((inst) => (inst.role === "train" ? 0 : uniform(-1, 1)));

// ---------------------------------------------------------------- points
const PATIENTS_PER_INSTRUMENT = 12;
const SPECTRA_PER_PATIENT = 25;
const pts = [];
let patientId = 0;
for (const inst of instruments) {
  for (let p = 0; p < PATIENTS_PER_INSTRUMENT; p++) {
    const label = p % 2;
    const re = [gauss(), gauss(), gauss()];
    const conc = concentrations(label);
    const patientLogit = 0.9 * gauss();
    for (let k = 0; k < SPECTRA_PER_PATIENT; k++) {
      pts.push({ inst, label, patient: patientId, re, conc, patientLogit });
    }
    patientId++;
  }
}

const stages = { raw: [], orpl: [], naive: [], robust: [] };
const P = { label: [], instrument: [], patient: [], p_naive: [], p_robust: [], ood: [] };
const pSvm = [];
for (const pt of pts) {
  const { inst, label, re } = pt;
  const s = label === 1 ? 1 : -1;
  const c = centers[inst.id];
  const d = inst.severity;
  const push = (key, v) => stages[key].push(...v.map((q) => r3(clamp(q, -1, 1))));

  push("raw", [0, 1, 2].map((a) => c[a] + (a === 0 ? 0.05 * s : 0) + 0.04 * re[a] + 0.075 * gauss()));
  push("orpl", [0, 1, 2].map((a) => 0.58 * c[a] + (a === 0 ? 0.2 * s : 0) + 0.05 * re[a] + 0.07 * gauss()));

  const train = inst.role === "train";
  const zNaive = s * 2.1 * (train ? 1 : 1 - 0.8 * d) + biasDirs[inst.id] * 3.2 * d + 0.6 * pt.patientLogit + 1.0 * gauss();
  const zRobust = s * 1.95 * (1 - 0.12 * d) + 0.55 * pt.patientLogit + 1.0 * gauss();
  const zSvm = s * 1.85 * (train ? 1 : 1 - 0.5 * d) + biasDirs[inst.id] * 1.8 * d + 0.6 * pt.patientLogit + 1.1 * gauss();
  const [dy, dz] = driftDirs[inst.id];
  push("naive", [
    0.7 * Math.tanh(zNaive / 1.7) + 0.05 * gauss(),
    dy * 0.62 * d + 0.05 * re[1] + 0.07 * gauss(),
    dz * 0.62 * d + 0.05 * re[2] + 0.07 * gauss(),
  ]);
  push("robust", [
    0.7 * Math.tanh(zRobust / 1.7) + 0.05 * gauss(),
    0.08 * c[1] + 0.07 * re[1] + 0.11 * gauss(),
    0.08 * c[2] + 0.07 * re[2] + 0.11 * gauss(),
  ]);

  P.label.push(label);
  P.instrument.push(inst.id);
  P.patient.push(pt.patient);
  P.p_naive.push(r3(sigmoid(zNaive)));
  P.p_robust.push(r3(sigmoid(zRobust)));
  P.ood.push(r3(train ? 1.6 + 0.8 * Math.abs(gauss()) : 1.7 + 4.6 * d + 1.4 * Math.abs(gauss())));
  pSvm.push(sigmoid(zSvm));
}

function silhouette(flat, labels, stride) {
  const idx = [];
  for (let i = 0; i < labels.length; i += stride) idx.push(i);
  const dist = (i, j) =>
    Math.hypot(flat[3 * i] - flat[3 * j], flat[3 * i + 1] - flat[3 * j + 1], flat[3 * i + 2] - flat[3 * j + 2]);
  let total = 0;
  for (const i of idx) {
    const sums = new Map();
    for (const j of idx) {
      if (i === j) continue;
      const e = sums.get(labels[j]) ?? [0, 0];
      e[0] += dist(i, j);
      e[1] += 1;
      sums.set(labels[j], e);
    }
    const own = sums.get(labels[i]);
    const a = own ? own[0] / own[1] : 0;
    let b = Infinity;
    for (const [k, [sum, n]] of sums) if (k !== labels[i]) b = Math.min(b, sum / n);
    total += (b - a) / Math.max(a, b);
  }
  return r3(total / idx.length);
}

const STAGE_TEXT = {
  raw: ["Raw counts", "Every instrument has its own fingerprint: spectra cluster by device, not by diagnosis."],
  orpl: ["ORPL processing", "Calibration, IRF correction and BubbleFill bring devices closer, but not all the way."],
  naive: ["Naive CNN", "Trained on two Montréal instruments: sharp on them, drifting on the rest of the fleet."],
  robust: ["Robust CNN", "Trained through randomised twins: the space is organised by diagnosis on every device."],
};

const SPEC_GRID = arange(600, 1800, 4);
const spectraIdx = [];
for (let i = 0; i < pts.length; i += 9) spectraIdx.push(i);
const spectraRaw = [];
const spectraProc = [];
for (const i of spectraIdx) {
  const { inst, conc } = pts[i];
  const shift = -16.2 * inst.params.laser_drift_nm;
  const broaden = inst.params.resolution_cm1 / 6;
  const ram = raman(SPEC_GRID, conc, shift, broaden);
  const bg = fluorescence(SPEC_GRID, inst.params.fluorescence_ratio, inst.params.irf_tilt);
  const raw = ram.map((v, k) => (v + bg[k]) * (1 + inst.params.irf_tilt * (k / SPEC_GRID.length - 0.5)) + 0.012 * gauss());
  const mx = Math.max(...raw);
  spectraRaw.push(raw.map((v) => r3(v / mx)));
  const resid = 0.03 * inst.severity;
  spectraProc.push(snv(ram.map((v, k) => v + resid * Math.sin(k / 30) + 0.01 * gauss())).map(r3));
}

const shiftScene = {
  version: 1,
  mock: true,
  classes: ["normal", "cancer"],
  instruments: instruments.map(({ id, name, role }) => ({ id, name, role })),
  stages: Object.keys(stages).map((key) => ({
    key,
    title: STAGE_TEXT[key][0],
    subtitle: STAGE_TEXT[key][1],
    positions: stages[key],
    silhouette: {
      instrument: silhouette(stages[key], P.instrument, 6),
      class: silhouette(stages[key], P.label, 6),
    },
  })),
  points: P,
  spectra: { grid: SPEC_GRID, index: spectraIdx, raw: spectraRaw, processed: spectraProc },
};

// ---------------------------------------------------------------- fleet metrics
function bacc(p, y) {
  let tp = 0;
  let tn = 0;
  let np = 0;
  let nn = 0;
  p.forEach((v, i) => {
    if (y[i] === 1) {
      np++;
      if (v >= 0.5) tp++;
    } else {
      nn++;
      if (v < 0.5) tn++;
    }
  });
  return 0.5 * (tp / np + tn / nn);
}
function auc(p, y) {
  const pos = p.filter((_, i) => y[i] === 1);
  const neg = p.filter((_, i) => y[i] === 0);
  let s = 0;
  for (const a of pos) for (const b of neg) s += a > b ? 1 : a === b ? 0.5 : 0;
  return s / (pos.length * neg.length);
}

const OOD_THRESHOLD = 6.0;
const fleetInstruments = instruments.map((inst) => {
  const ids = P.instrument.map((v, i) => (v === inst.id ? i : -1)).filter((i) => i >= 0);
  const y = ids.map((i) => P.label[i]);
  const metric = (p) => ({ bacc: r3(bacc(p, y)), auc: r3(auc(p, y)) });
  return {
    id: inst.id,
    name: inst.name,
    role: inst.role,
    params: inst.params,
    metrics: {
      svm: metric(ids.map((i) => pSvm[i])),
      naive: metric(ids.map((i) => P.p_naive[i])),
      robust: metric(ids.map((i) => P.p_robust[i])),
    },
    abstain_rate: r3(ids.filter((i) => P.ood[i] > OOD_THRESHOLD).length / ids.length),
  };
});
// Patient-level bootstrap of the fleet mean balanced accuracy.
const probs = { svm: pSvm, naive: P.p_naive, robust: P.p_robust };
const fleetPatients = instruments
  .filter((inst) => inst.role === "fleet")
  .map((inst) => {
    const byPatient = new Map();
    P.instrument.forEach((v, i) => {
      if (v !== inst.id) return;
      if (!byPatient.has(P.patient[i])) byPatient.set(P.patient[i], []);
      byPatient.get(P.patient[i]).push(i);
    });
    return [...byPatient.values()];
  });
function bootstrapCI(m, reps = 300) {
  const means = [];
  for (let b = 0; b < reps; b++) {
    let sum = 0;
    let n = 0;
    for (const patients of fleetPatients) {
      const ids = [];
      for (let k = 0; k < patients.length; k++) ids.push(...patients[Math.floor(rand() * patients.length)]);
      const v = bacc(ids.map((i) => probs[m][i]), ids.map((i) => P.label[i]));
      if (Number.isFinite(v)) {
        sum += v;
        n++;
      }
    }
    means.push(sum / n);
  }
  means.sort((a, b) => a - b);
  return [r3(means[Math.floor(0.025 * reps)]), r3(means[Math.ceil(0.975 * reps) - 1])];
}

const summary = {};
for (const m of ["svm", "naive", "robust"]) {
  const v = fleetInstruments.filter((f) => f.role === "fleet").map((f) => f.metrics[m].bacc);
  summary[m] = {
    mean_bacc: r3(v.reduce((a, b) => a + b, 0) / v.length),
    worst_bacc: r3(Math.min(...v)),
    mean_bacc_ci95: bootstrapCI(m),
  };
}
const fleet = {
  version: 1,
  mock: true,
  instruments: fleetInstruments,
  summary,
  latency_ms: { orpl: 2.1, onnx_cnn: 0.3 },
};

// ---------------------------------------------------------------- robustness atlas
const AX = arange(-1.5, 1.5, 0.25);
const AY = [1, 1.5, 2, 3, 5, 8, 12, 20, 30, 50, 80];
const ENV = { x: [-0.2, 0.2], y: [3, 8] };
const ENV_R = { x: [-0.9, 0.9], y: [2, 30] };
const distTo = (env, x, y) => {
  const dx = Math.max(0, Math.abs(x) - env.x[1]) / 0.45;
  const ly = Math.log(y);
  const dy = Math.max(0, Math.log(env.y[0]) - ly, ly - Math.log(env.y[1])) / Math.log(2.6);
  return dx * dx + dy * dy;
};
const dist = (x, y) => distTo(ENV, x, y);
const grid = (f) => AY.map((y) => AX.map((x) => r3(clamp(f(x, y) + 0.006 * gauss(), 0.45, 0.99))));
const atlas = {
  version: 1,
  mock: true,
  x: { param: "laser_drift_nm", label: "Laser drift (nm)", values: AX },
  y: { param: "fluorescence_ratio", label: "Fluorescence / Raman", values: AY },
  metric: "balanced_accuracy",
  models: {
    svm: grid((x, y) => 0.52 + 0.4 * Math.exp(-0.45 * dist(x, y))),
    naive: grid((x, y) => 0.5 + 0.45 * Math.exp(-1.25 * dist(x, y))),
    robust: grid((x, y) => 0.945 - 0.015 * Math.abs(x / 1.5) - 0.12 * (1 - Math.exp(-1.1 * distTo(ENV_R, x, y)))),
  },
  train_envelope: ENV,
  randomization_envelope: ENV_R,
};

// ---------------------------------------------------------------- twin lab scenarios
function simulateResponse({ label, drift = 0, fluo = 5, resolution = 6, signal = 2000, ambient = 0, tilt = 0, noise = 1, preds, qc }) {
  const conc = concentrations(label);
  const px = linspace(452, 1900, 512);
  const shift = -16.2 * drift;
  const ram = raman(px, conc, shift, resolution / 6);
  const bg = fluorescence(px, fluo, tilt);
  const scale = signal * 10;
  const amb = px.map((x) =>
    ambient * ([1114, 1462, 1622].reduce((s, c) => s + Math.exp(-0.5 * ((x - c) / 2.2) ** 2), 0) * 2.2),
  );
  const counts = px.map((_, k) => {
    const clean = (ram[k] + bg[k] + amb[k]) * scale;
    return clean + Math.sqrt(clean) * noise * gauss() * 1.2;
  });
  const top = Math.max(...counts);
  const raw = counts.map((v) => v / top);
  const baseline = smooth(px.map((_, k) => (bg[k] * scale) / top), 6);
  const gx = arange(600, 1800, 2);
  const ramG = raman(gx, conc, shift, resolution / 6);
  const ambG = gx.map((x) => ambient * ([1114, 1462, 1622].reduce((s, c) => s + Math.exp(-0.5 * ((x - c) / 2.2) ** 2), 0) * 2.2));
  const relNoise = (noise * Math.sqrt(fluo + 1)) / Math.sqrt(scale);
  const proc = snv(ramG.map((v, k) => v + ambG[k] + relNoise * 6 * gauss()));
  return {
    raw: { x: px.map(r3), y: raw.map(r3) },
    baseline: { x: px.map(r3), y: baseline.map(r3) },
    processed: { x: gx.map(r3), y: proc.map(r3) },
    truth: { label, concentrations: conc },
    predictions: { svm: { p_cancer: preds[0] }, naive: { p_cancer: preds[1] }, robust: { p_cancer: preds[2] } },
    qc,
    timing_ms: { acquire: 1.3, orpl: r3(2.2 + 0.4 * rand()), inference: r3(0.6 + 0.3 * rand()) },
  };
}

const SCENARIOS = [
  ["nominal", "Nominal instrument", {}, { label: 1, preds: [0.83, 0.94, 0.95], qc: { flag: "ok", ood_score: 2.4, threshold: OOD_THRESHOLD, reasons: [] } }],
  [
    "laser_drift",
    "Laser drift +0.8 nm",
    { laser_drift_nm: 0.8 },
    { label: 1, drift: 0.8, preds: [0.46, 0.17, 0.89], qc: { flag: "warn", ood_score: 7.3, threshold: OOD_THRESHOLD, reasons: ["embedding outside training envelope"] } },
  ],
  [
    "high_fluorescence",
    "High fluorescence",
    { fluorescence_ratio: 40 },
    { label: 1, fluo: 40, noise: 1.4, preds: [0.57, 0.31, 0.86], qc: { flag: "warn", ood_score: 6.6, threshold: OOD_THRESHOLD, reasons: ["fluorescence background above training range"] } },
  ],
  [
    "ambient_light",
    "Room lights on",
    { ambient_light: 0.6 },
    { label: 0, ambient: 0.6, preds: [0.62, 0.78, 0.41], qc: { flag: "reject", ood_score: 11.8, threshold: OOD_THRESHOLD, reasons: ["ambient-light emission lines detected"] } },
  ],
  [
    "low_snr",
    "Short exposure, low signal",
    { signal_e_per_s: 250, exposure_s: 0.2 },
    { label: 0, signal: 50, noise: 1.0, preds: [0.44, 0.58, 0.32], qc: { flag: "warn", ood_score: 5.1, threshold: OOD_THRESHOLD, reasons: ["low signal-to-noise ratio"] } },
  ],
];
const twinlab = {
  version: 1,
  mock: true,
  scenarios: SCENARIOS.map(([key, title, params, spec]) => ({
    key,
    title,
    params: { ...params },
    response: simulateResponse({ ...spec, drift: spec.drift ?? 0 }),
  })),
};

mkdirSync(OUT, { recursive: true });
const write = (name, obj) => {
  writeFileSync(join(OUT, name), JSON.stringify(obj));
  console.log(`${name}`);
};
write("shift_scene.json", shiftScene);
write("robustness_atlas.json", atlas);
write("fleet.json", fleet);
write("twinlab_samples.json", twinlab);
for (const s of shiftScene.stages) console.log(s.key, s.silhouette);
for (const m of Object.keys(summary)) console.log(m, summary[m]);
