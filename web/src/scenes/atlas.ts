import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { damp, disposeObject, easeOutCubic, frameFreeArea, label, type View } from "../gl/engine";
import { CSS, cssGradient, linear, viridis } from "../gl/colors";
import { MODEL_NAMES, finitePair, loadAtlas, loadFleet, validEnvelope, type Atlas, type Fleet, type ModelKey } from "../data";
import { fmt, h, q } from "../ui/dom";
import type { ViewContext } from "../main";

const W = 6.4;
const D = 4.6;
const HGT = 2.8;
const UP = 6;
const ORDER: ModelKey[] = ["naive", "robust", "svm"];
// Fills are always transparent, so every line and overlay must come after them in the transparent queue.
const OVERLAY = 10;

const num = (v: number) => (Number.isInteger(v) ? String(v) : String(Number(v.toFixed(2))));
const minus = (t: string) => t.replace("-", "\u2212");
const sig2 = (v: number) => (Math.abs(v) >= 100 ? String(Math.round(v)) : v === 0 ? "0" : String(Number(v.toPrecision(2))));

/** Replaces missing cells by the mean of their valid neighbours (or the grid minimum) for drawing. */
function fillGrid(g: Array<Array<number | null>>): number[][] {
  const valid = g.flat().filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const fallback = valid.length ? Math.min(...valid) : 0.5;
  return g.map((row, j) =>
    row.map((v, i) => {
      if (typeof v === "number" && Number.isFinite(v)) return v;
      const around = [g[j - 1]?.[i], g[j + 1]?.[i], row[i - 1], row[i + 1]].filter(
        (n): n is number => typeof n === "number" && Number.isFinite(n),
      );
      return around.length ? around.reduce((a, b) => a + b, 0) / around.length : fallback;
    }),
  );
}

function toIndex(values: number[], v: number): number {
  if (values.length < 2) return 0;
  if (v <= values[0]) return 0;
  for (let i = 1; i < values.length; i++) {
    if (v <= values[i]) return i - 1 + (v - values[i - 1]) / (values[i] - values[i - 1] || 1);
  }
  return values.length - 1;
}

function catmull(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

function sampleGrid(g: number[][], gx: number, gy: number): number {
  const ny = g.length;
  const nx = g[0].length;
  const ix = Math.min(nx - 2, Math.max(0, Math.floor(gx)));
  const iy = Math.min(ny - 2, Math.max(0, Math.floor(gy)));
  const tx = gx - ix;
  const ty = gy - iy;
  const at = (y: number, x: number) => g[Math.min(ny - 1, Math.max(0, y))][Math.min(nx - 1, Math.max(0, x))];
  const rows = [-1, 0, 1, 2].map((dy) =>
    catmull(at(iy + dy, ix - 1), at(iy + dy, ix), at(iy + dy, ix + 1), at(iy + dy, ix + 2), nx > 1 ? tx : 0),
  );
  return catmull(rows[0], rows[1], rows[2], rows[3], ny > 1 ? ty : 0);
}

export async function createAtlas(ctx: ViewContext): Promise<View> {
  const [atlas, fleet]: [Atlas, Fleet | null] = await Promise.all([loadAtlas(), loadFleet().catch(() => null)]);
  const xs = atlas.x.values;
  const ys = atlas.y.values;
  const nx = xs.length;
  const ny = ys.length;
  const models = ORDER.filter((m) => atlas.models[m]?.length);
  // Surfaces pass through the raw cell values, so the hover dot and the tooltip agree.
  const display = new Map(models.map((m) => [m, fillGrid(atlas.models[m]!)]));

  let minMetric = 1;
  for (const g of display.values()) for (const row of g) for (const v of row) minMetric = Math.min(minMetric, v);
  const lo = Math.min(0.4, Math.floor(minMetric * 10) / 10);
  const fx = (gx: number) => -W / 2 + (W * gx) / Math.max(1, nx - 1);
  const fz = (gy: number) => D / 2 - (D * gy) / Math.max(1, ny - 1);
  const fy = (m: number) => (HGT * (m - lo)) / (1 - lo);
  const colorT = (m: number) => Math.min(1, Math.max(0, (m - 0.5) / 0.5));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
  camera.position.set(7.4, 5.6, 8.2);
  const controls = new OrbitControls(camera, ctx.engine.renderer.domElement);
  controls.target.set(0, 0.9, 0);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 6;
  controls.maxDistance = 22;
  controls.maxPolarAngle = 1.45;
  controls.autoRotateSpeed = 0.3;

  scene.add(new THREE.HemisphereLight(0xc8d4ff, 0x0a0e15, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(4, 9, 6);
  scene.add(sun);

  // ---------------------------------------------------------------- frame and axes
  const frame = new THREE.Group();
  scene.add(frame);
  const lineMat = (color: number | string, opacity: number) =>
    new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
  const seg = (pts: number[], mat: THREE.LineBasicMaterial) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.LineSegments(g, mat);
  };

  const base: number[] = [];
  for (let i = 0; i < nx; i++) base.push(fx(i), 0, D / 2, fx(i), 0, -D / 2);
  for (let j = 0; j < ny; j++) base.push(-W / 2, 0, fz(j), W / 2, 0, fz(j));
  frame.add(seg(base, lineMat(0x8f9aad, 0.09)));
  const outline = [
    -W / 2, 0, D / 2, W / 2, 0, D / 2, W / 2, 0, D / 2, W / 2, 0, -D / 2,
    W / 2, 0, -D / 2, -W / 2, 0, -D / 2, -W / 2, 0, -D / 2, -W / 2, 0, D / 2,
    -W / 2, 0, -D / 2, -W / 2, HGT, -D / 2, -W / 2, 0, D / 2, -W / 2, HGT, D / 2,
  ];
  frame.add(seg(outline, lineMat(0x8f9aad, 0.35)));

  const metricTicks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].filter((m) => m >= lo);
  const walls: number[] = [];
  for (const m of metricTicks) {
    const y = fy(m);
    walls.push(-W / 2, y, -D / 2, W / 2, y, -D / 2, -W / 2, y, -D / 2, -W / 2, y, D / 2);
    const t = label(m.toFixed(1), "tick3d");
    t.position.set(-W / 2 - 0.32, y, D / 2 + 0.22);
    frame.add(t);
  }
  frame.add(seg(walls, lineMat(0x8f9aad, 0.1)));
  if (lo < 0.5) {
    const y = fy(0.5);
    frame.add(seg([-W / 2, y, -D / 2, W / 2, y, -D / 2, -W / 2, y, -D / 2, -W / 2, y, D / 2], lineMat(0xff5c6c, 0.45)));
    const chance = label("chance", "tick3d");
    chance.element.style.color = CSS.reject;
    chance.center.set(0, 0.5);
    chance.position.set(W / 2 + 0.25, y, -D / 2);
    frame.add(chance);
  }
  const mTitle = label("Balanced accuracy", "axis3d");
  mTitle.position.set(-W / 2 - 0.2, HGT + 0.45, D / 2);
  frame.add(mTitle);

  // At most 6 ticks per axis, and titles anchored on the side away from the ticks.
  const xStep = Math.max(1, Math.ceil(nx / 6));
  for (let i = 0; i < nx; i += xStep) {
    const t = label(minus(num(xs[i])), "tick3d");
    t.position.set(fx(i), 0, D / 2 + 0.38);
    frame.add(t);
  }
  const yStep = Math.max(1, Math.ceil(ny / 6));
  for (let j = 0; j < ny; j += yStep) {
    const t = label(minus(sig2(ys[j])), "tick3d");
    t.position.set(W / 2 + 0.42, 0, fz(j));
    frame.add(t);
  }
  const xTitle = label(atlas.x.label, "axis3d");
  xTitle.center.set(0.5, 0);
  xTitle.position.set(0, 0, D / 2 + 1.0);
  frame.add(xTitle);
  const yTitle = label(atlas.y.label, "axis3d");
  yTitle.center.set(0, 0.5);
  yTitle.position.set(W / 2 + 1.0, 0, 0);
  frame.add(yTitle);

  // ---------------------------------------------------------------- training envelopes
  // Train instruments in neutral white, randomised twins in the robust green, as in the fleet panel.
  // A malformed envelope only skips its own layer.
  const nTrain = fleet?.instruments.filter((i) => i.role === "train").length;
  const envRect = (e: { x: [number, number]; y: [number, number] }) => ({
    x0: fx(toIndex(xs, e.x[0])),
    x1: fx(toIndex(xs, e.x[1])),
    z0: fz(toIndex(ys, e.y[0])),
    z1: fz(toIndex(ys, e.y[1])),
  });
  const loop = (r: { x0: number; x1: number; z0: number; z1: number }, y: number) => [
    r.x0, y, r.z0, r.x1, y, r.z0, r.x1, y, r.z0, r.x1, y, r.z1,
    r.x1, y, r.z1, r.x0, y, r.z1, r.x0, y, r.z1, r.x0, y, r.z0,
  ];
  const envGroup = new THREE.Group();
  scene.add(envGroup);

  const rand = validEnvelope(atlas.randomization_envelope);
  if (rand) {
    const r = envRect(rand);
    const dashed = seg(
      loop(r, 0.006),
      new THREE.LineDashedMaterial({ color: new THREE.Color(CSS.robust).multiplyScalar(1.8), dashSize: 0.14, gapSize: 0.09, toneMapped: false }),
    );
    dashed.computeLineDistances();
    envGroup.add(dashed);
    const fill = new THREE.Mesh(
      new THREE.PlaneGeometry(Math.max(0.02, Math.abs(r.x1 - r.x0)), Math.max(0.02, Math.abs(r.z1 - r.z0))),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(CSS.robust), transparent: true, opacity: 0.1, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    fill.rotation.x = -Math.PI / 2;
    fill.position.set((r.x0 + r.x1) / 2, 0.003, (r.z0 + r.z1) / 2);
    envGroup.add(fill);
    // Back edge, lifted off the plane, so it never meets the x-axis ticks at the front.
    const randLabel = label('Robust training<span class="sub">· randomised twins</span>');
    randLabel.element.style.color = CSS.robust;
    randLabel.position.set((r.x0 + r.x1) / 2, 0.35, Math.min(r.z0, r.z1) - 0.1);
    envGroup.add(randLabel);
  }

  const trainEnv = validEnvelope(atlas.train_envelope);
  let column: THREE.Mesh | null = null;
  if (trainEnv) {
    const e = envRect(trainEnv);
    const white = new THREE.Color(CSS.text);
    const verticals = [e.x0, 0, e.z0, e.x0, HGT, e.z0, e.x1, 0, e.z0, e.x1, HGT, e.z0, e.x1, 0, e.z1, e.x1, HGT, e.z1, e.x0, 0, e.z1, e.x0, HGT, e.z1];
    envGroup.add(seg(loop(e, 0.01), new THREE.LineBasicMaterial({ color: white.clone().multiplyScalar(2.2), toneMapped: false })));
    envGroup.add(seg(verticals, lineMat(CSS.text, 0.22)));
    const envW = Math.max(0.02, Math.abs(e.x1 - e.x0));
    const envD = Math.max(0.02, Math.abs(e.z1 - e.z0));
    column = new THREE.Mesh(
      new THREE.BoxGeometry(envW, HGT, envD),
      new THREE.MeshBasicMaterial({ color: white, transparent: true, opacity: 0.05, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    column.position.set((e.x0 + e.x1) / 2, HGT / 2, (e.z0 + e.z1) / 2);
    envGroup.add(column);
    const floorGlow = new THREE.Mesh(
      new THREE.PlaneGeometry(envW, envD),
      new THREE.MeshBasicMaterial({ color: white, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    floorGlow.rotation.x = -Math.PI / 2;
    floorGlow.position.set((e.x0 + e.x1) / 2, 0.005, (e.z0 + e.z1) / 2);
    envGroup.add(floorGlow);
    const envLabel = label("Train instruments");
    envLabel.position.set((e.x0 + e.x1) / 2, HGT + 0.2, (e.z0 + e.z1) / 2);
    envGroup.add(envLabel);
  }

  frame.traverse((o) => (o.renderOrder = OVERLAY));
  envGroup.traverse((o) => (o.renderOrder = OVERLAY));

  // ---------------------------------------------------------------- surfaces
  // Every model has a colour-mapped fill, a dark grid on it and a neutral ghost wire. A look sets their opacities.
  type Look = "hidden" | "ghost" | "solid" | "compare";
  const LOOKS: Record<Look, { fill: number; grid: number; ghost: number; rise: number }> = {
    hidden: { fill: 0, grid: 0, ghost: 0, rise: 0 },
    ghost: { fill: 0, grid: 0, ghost: 0.14, rise: 1 },
    solid: { fill: 1, grid: 0.45, ghost: 0, rise: 1 },
    compare: { fill: 0.5, grid: 0.35, ghost: 0, rise: 1 },
  };
  interface Surface {
    key: ModelKey;
    group: THREE.Group;
    mesh: THREE.Mesh;
    fillMat: THREE.MeshStandardMaterial;
    gridMat: THREE.LineBasicMaterial;
    ghostMat: THREE.LineBasicMaterial;
    look: Look;
    rise: number;
  }
  const NX = (nx - 1) * UP + 1;
  const NY = (ny - 1) * UP + 1;
  const surfaces = new Map<ModelKey, Surface>();
  const idx: number[] = [];
  for (let j = 0; j < NY - 1; j++) {
    for (let i = 0; i < NX - 1; i++) {
      const a = j * NX + i;
      idx.push(a, a + NX, a + 1, a + 1, a + NX, a + NX + 1);
    }
  }

  models.forEach((key, order) => {
    const grid = display.get(key)!;
    const group = new THREE.Group();
    group.scale.y = 0.001;
    group.visible = false;
    scene.add(group);
    const pos = new Float32Array(NX * NY * 3);
    const col = new Float32Array(NX * NY * 3);
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const gx = i / UP;
        const gy = j / UP;
        const m = Math.min(1, Math.max(lo, sampleGrid(grid, gx, gy)));
        const k = j * NX + i;
        pos.set([fx(gx), fy(m), fz(gy)], k * 3);
        const c = linear(viridis(colorT(m)));
        col.set([c.r, c.g, c.b], k * 3);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    const fillMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.45,
      metalness: 0.05,
      transparent: true,
      opacity: 0,
      emissive: new THREE.Color(0x0a0d14),
    });
    const mesh = new THREE.Mesh(g, fillMat);
    mesh.renderOrder = 1 + order;
    group.add(mesh);

    const wire: number[] = [];
    const wireCol: number[] = [];
    const push = (k: number) => {
      wire.push(pos[3 * k], pos[3 * k + 1] + 0.004, pos[3 * k + 2]);
      wireCol.push(col[3 * k] * 0.45, col[3 * k + 1] * 0.45, col[3 * k + 2] * 0.45);
    };
    for (let j = 0; j < NY; j += UP) for (let i = 0; i < NX - 1; i++) push(j * NX + i), push(j * NX + i + 1);
    for (let i = 0; i < NX; i += UP) for (let j = 0; j < NY - 1; j++) push(j * NX + i), push((j + 1) * NX + i);
    const wg = new THREE.BufferGeometry();
    wg.setAttribute("position", new THREE.Float32BufferAttribute(wire, 3));
    wg.setAttribute("color", new THREE.Float32BufferAttribute(wireCol, 3));
    const gridMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, depthWrite: false });
    // The ghost ignores depth so it stays readable under a solid surface (naive lies below robust almost everywhere).
    const ghostMat = new THREE.LineBasicMaterial({ color: CSS.text, transparent: true, opacity: 0, depthWrite: false, depthTest: false });
    const gridLines = new THREE.LineSegments(wg, gridMat);
    const ghostLines = new THREE.LineSegments(wg, ghostMat);
    gridLines.renderOrder = OVERLAY;
    ghostLines.renderOrder = OVERLAY + 1;
    group.add(gridLines, ghostLines);
    surfaces.set(key, { key, group, mesh, fillMat, gridMat, ghostMat, look: "hidden", rise: 0 });
  });

  // Steps: each model solid in turn with the previous one ghosted, then naive and robust compared.
  type Layout = Partial<Record<ModelKey, Look>>;
  const steps: Layout[] = models.map((m, i) => (i > 0 ? { [m]: "solid", [models[i - 1]]: "ghost" } : { [m]: "solid" }));
  const canCompare = models.includes("naive") && models.includes("robust");
  if (canCompare) steps.push({ naive: "compare", robust: "compare" });
  const visible = (s: Surface) => s.look === "solid" || s.look === "compare";

  // ---------------------------------------------------------------- UI
  const root = h("div");
  root.innerHTML = `
    <div class="panel atlas-left">
      <p class="heading">Models</p>
      <div class="model-toggle"></div>
      <p class="heading" style="margin-top:20px">${sentence(atlas.metric.replace(/_/g, " "))}</p>
      <div class="gradient-bar" style="background:${cssGradient(viridis)}"></div>
      <div class="gradient-labels"><span>0.5 · chance</span><span>1.0</span></div>
      <p class="empty" style="margin-top:14px">Each surface is a model's accuracy over a grid of simulated instrument conditions.</p>
      <div class="env-legend">
        ${trainEnv ? `<div><i class="env-swatch naive"></i>Train instruments: conditions of the ${nTrain ?? ""} Montréal instruments (naive CNN, SVM)</div>` : ""}
        ${rand ? '<div><i class="env-swatch robust"></i>Robust training: range of the randomised twins</div>' : ""}
      </div>
    </div>
    <div class="panel atlas-right fleet"></div>
    <div class="tooltip" style="opacity:0"></div>
    <div class="hint"><kbd>Space</kbd> next model · <kbd>R</kbd> rotate</div>`;
  ctx.ui.appendChild(root);

  const toggles = q(root, ".model-toggle");
  const solidIcon = `background:${cssGradient(viridis, 0.2, 1)}`;
  for (const m of models) {
    const stat = fleet?.summary[m]?.mean_bacc;
    const b = h(
      "button",
      { "data-m": m, title: `Show ${MODEL_NAMES[m]}, ghost the others` },
      `<span class="style-icon" style="${solidIcon}"></span>${MODEL_NAMES[m]}<span class="stat">${typeof stat === "number" ? `fleet ${fmt(stat)}` : ""}</span>`,
    );
    b.addEventListener("click", () => apply(Object.fromEntries(models.map((k) => [k, k === m ? "solid" : "ghost"])), models.indexOf(m)));
    toggles.appendChild(b);
  }
  if (canCompare) {
    const b = h(
      "button",
      { "data-m": "compare", title: "Naive and robust CNN, both semi-transparent" },
      `<span class="style-icon" style="${solidIcon};opacity:.5"></span>Compare naive vs robust`,
    );
    b.addEventListener("click", () => apply(steps[steps.length - 1], steps.length - 1));
    toggles.appendChild(b);
  }
  function renderToggles(): void {
    const compare = [...surfaces.values()].some((s) => s.look === "compare");
    toggles.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
      const m = b.dataset.m;
      b.classList.toggle("on", m === "compare" ? compare : surfaces.get(m as ModelKey)?.look === "solid");
    });
  }

  let step = 0;
  function apply(sc: Layout, stepIndex: number): void {
    step = stepIndex;
    for (const [k, s] of surfaces) s.look = sc[k] ?? "hidden";
    const sorted = [...surfaces.values()].sort((a, b) => Number(visible(a)) - Number(visible(b)));
    sorted.forEach((s, i) => (s.mesh.renderOrder = 1 + i));
    renderToggles();
    dirty = true;
  }

  function setStep(i: number): void {
    apply(steps[i] ?? {}, i);
  }

  if (fleet) renderFleet(q(root, ".fleet"), fleet);
  else q(root, ".fleet").remove();

  // ---------------------------------------------------------------- hover
  const canvas = ctx.engine.renderer.domElement;
  const tooltip = q(root, ".tooltip");
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let pointer: { x: number; y: number } | null = null;
  let dirty = false;
  const probe = new THREE.Group();
  probe.visible = false;
  scene.add(probe);
  const probeLineGeo = new THREE.BufferGeometry();
  probeLineGeo.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, 1, 0], 3));
  const probeLine = new THREE.Line(probeLineGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6, depthTest: false }));
  probeLine.renderOrder = OVERLAY + 2;
  probe.add(probeLine);
  const probeDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.06, 16, 12),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(2, 2, 2), toneMapped: false, transparent: true, depthTest: false }),
  );
  probeDot.renderOrder = OVERLAY + 2;
  probe.add(probeDot);

  const onMove = (e: PointerEvent) => {
    pointer = { x: e.clientX, y: e.clientY };
    dirty = true;
  };
  const onLeave = () => {
    pointer = null;
    dirty = true;
  };
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);

  function hover(): void {
    const shown = [...surfaces.values()].filter(visible);
    if (!pointer || shown.length === 0) {
      tooltip.style.opacity = "0";
      probe.visible = false;
      return;
    }
    ndc.set((pointer.x / ctx.engine.width) * 2 - 1, -(pointer.y / ctx.engine.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObjects(
      shown.map((s) => s.mesh),
      false,
    )[0];
    if (!hit) {
      tooltip.style.opacity = "0";
      probe.visible = false;
      canvas.style.cursor = "";
      return;
    }
    const ix = Math.min(nx - 1, Math.max(0, Math.round(((hit.point.x + W / 2) / W) * (nx - 1))));
    const iy = Math.min(ny - 1, Math.max(0, Math.round(((D / 2 - hit.point.z) / D) * (ny - 1))));
    const rows = models
      .filter((m) => surfaces.get(m)!.look !== "hidden")
      .map((m) => {
        const v = atlas.models[m]?.[iy]?.[ix];
        const ghost = surfaces.get(m)!.look === "ghost" ? " muted" : "";
        if (typeof v !== "number") return `<div class="row${ghost}"><span>${MODEL_NAMES[m]}</span><span class="num">–</span></div>`;
        const [r, g, b] = viridis(colorT(v));
        const color = ghost ? "" : ` style="color:rgb(${r * 255},${g * 255},${b * 255})"`;
        return `<div class="row${ghost}"><span>${MODEL_NAMES[m]}</span><span class="num"${color}>${v.toFixed(3)}</span></div>`;
      })
      .join("");
    tooltip.innerHTML = `<div class="row muted"><span>${atlas.x.label}</span><span class="num">${minus(num(xs[ix]))}</span></div>
      <div class="row muted" style="margin-bottom:6px"><span>${atlas.y.label}</span><span class="num">${minus(num(ys[iy]))}</span></div>${rows}`;
    tooltip.style.left = `${pointer.x}px`;
    tooltip.style.top = `${pointer.y}px`;
    tooltip.style.opacity = "1";
    const top = Math.max(...shown.map((s) => fy(display.get(s.key)![iy][ix])));
    probe.visible = true;
    probe.position.set(fx(ix), 0, fz(iy));
    probeLine.scale.y = Math.max(0.001, top);
    probeDot.position.y = top;
    canvas.style.cursor = "crosshair";
  }

  setStep(0);

  return {
    scene,
    camera,
    bloom: { strength: 0.7, radius: 0.5, threshold: 0.6 },
    update(dt, t) {
      controls.update(dt);
      for (const s of surfaces.values()) {
        const L = LOOKS[s.look];
        s.rise = damp(s.rise, L.rise, 4.5, dt);
        if (Math.abs(s.rise - L.rise) < 0.002) s.rise = L.rise;
        s.fillMat.opacity = damp(s.fillMat.opacity, L.fill, 6, dt);
        s.gridMat.opacity = damp(s.gridMat.opacity, L.grid, 6, dt);
        s.ghostMat.opacity = damp(s.ghostMat.opacity, L.ghost, 6, dt);
        s.fillMat.depthWrite = s.fillMat.opacity > 0.95;
        s.mesh.visible = s.fillMat.opacity > 0.01;
        s.group.visible = s.rise > 0.002;
        s.group.scale.y = Math.max(0.001, easeOutCubic(s.rise));
        if (s.rise !== L.rise) dirty = true;
      }
      if (column) (column.material as THREE.MeshBasicMaterial).opacity = 0.04 + 0.02 * Math.sin(t * 2);
      if (dirty || controls.autoRotate) {
        dirty = false;
        hover();
      }
    },
    resize(w, hgt) {
      const narrow = w < 1280;
      frameFreeArea(camera, w, hgt, { left: narrow ? 306 : 336, right: narrow ? 346 : 406, top: 70, bottom: 30 }, 1.12);
    },
    next() {
      if (step >= steps.length - 1) return false;
      setStep(step + 1);
      return true;
    },
    prev() {
      if (step <= 0) return false;
      setStep(step - 1);
      return true;
    },
    onKey(e) {
      if (e.key === "r") {
        controls.autoRotate = !controls.autoRotate;
        return true;
      }
      return false;
    },
    dispose() {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.style.cursor = "";
      controls.dispose();
      disposeObject(scene);
      root.remove();
    },
  };
}

function sentence(t: string): string {
  return t.charAt(0).toUpperCase() + t.slice(1);
}

function renderFleet(el: HTMLElement, fleet: Fleet): void {
  const s = fleet.summary;
  const ci = (m: "naive" | "robust") => {
    const c = finitePair(s[m]?.mean_bacc_ci95);
    return c ? `<span style="color:${m === "naive" ? CSS.naive : CSS.robust}">${fmt(c[0])}–${fmt(c[1])}</span>` : "";
  };
  const ciParts = [ci("naive"), ci("robust")].filter(Boolean);
  const ciLine = ciParts.length
    ? `<div class="ci">Mean 95% CI ${ciParts.join('<span class="arrow">→</span>')}<br/>bootstrap over instruments and patients</div>`
    : "";
  const stat = (k: string, a?: number, b?: number) =>
    `<div class="stat-block"><div class="k">${k}</div><div class="v"><span style="color:${CSS.naive}">${fmt(a)}</span><span class="arrow">→</span><span style="color:${CSS.robust}">${fmt(b)}</span></div></div>`;
  const rows = [...fleet.instruments].sort((a, b) => (a.role === b.role ? a.id - b.id : a.role === "train" ? -1 : 1));
  const w = (v?: number) => `${v === undefined ? 0 : Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * 100}%`;
  const lat = Object.entries(fleet.latency_ms)
    .map(([k, v]) => `${k.replace(/_/g, " ").toUpperCase()} ${fmt(v, 1)} ms`)
    .join(" · ");
  el.innerHTML = `
    <p class="heading">Fleet, leave-instruments-out</p>
    <div class="stats">${stat("Mean balanced accuracy", s.naive?.mean_bacc, s.robust?.mean_bacc)}${stat("Worst instrument", s.naive?.worst_bacc, s.robust?.worst_bacc)}</div>${ciLine}
    <div class="legend" style="margin-bottom:8px">
      <span><i class="swatch" style="background:${CSS.naive}"></i>Naive CNN</span>
      <span><i class="swatch" style="background:${CSS.robust}"></i>Robust CNN</span>
      <span class="muted" style="margin-left:auto">0.5 → 1.0</span>
    </div>
    <div class="fleet-rows">${rows
      .map(
        (r, i) => `${r.role !== rows[i - 1]?.role ? `<p class="fleet-group">${r.role === "train" ? "Training instruments" : "New instruments"}</p>` : ""}<div class="fleet-row">
        <span class="name">${r.name}</span>
        <div class="fleet-bars">
          <div class="bar"><i data-w="${w(r.metrics.naive?.bacc)}" style="background:${CSS.naive}"></i></div>
          <div class="bar"><i data-w="${w(r.metrics.robust?.bacc)}" style="background:${CSS.robust}"></i></div>
        </div>
        <div class="fleet-vals"><span style="color:${CSS.naive}">${fmt(r.metrics.naive?.bacc)}</span><span class="arrow">→</span><span style="color:${CSS.robust}">${fmt(r.metrics.robust?.bacc)}</span></div>
      </div>`,
      )
      .join("")}</div>
    <p class="heading" style="margin:14px 0 4px">Latency per spectrum</p>
    <p class="latency" style="margin:0">${lat}</p>`;
  requestAnimationFrame(() =>
    requestAnimationFrame(() => el.querySelectorAll<HTMLElement>("[data-w]").forEach((b) => (b.style.width = b.dataset.w!))),
  );
}
