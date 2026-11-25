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

const num = (v: number) => (Number.isInteger(v) ? String(v) : String(Number(v.toFixed(2))));

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
  const grids = new Map(models.map((m) => [m, fillGrid(atlas.models[m]!)]));
  const steps = models.map((_, i) => models.slice(0, i + 1));

  let minMetric = 1;
  for (const g of grids.values()) for (const row of g) for (const v of row) minMetric = Math.min(minMetric, v);
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
    chance.position.set(W / 2 + 0.45, y, -D / 2);
    frame.add(chance);
  }
  const mTitle = label("Balanced accuracy", "axis3d");
  mTitle.position.set(-W / 2 - 0.2, HGT + 0.45, D / 2);
  frame.add(mTitle);

  const xStep = Math.max(1, Math.ceil(nx / 7));
  for (let i = 0; i < nx; i += xStep) {
    const t = label(num(xs[i]), "tick3d");
    t.position.set(fx(i), 0, D / 2 + 0.38);
    frame.add(t);
  }
  const yStep = Math.max(1, Math.ceil(ny / 9));
  for (let j = 0; j < ny; j += yStep) {
    const t = label(num(ys[j]), "tick3d");
    t.position.set(W / 2 + 0.42, 0, fz(j));
    frame.add(t);
  }
  const xTitle = label(atlas.x.label, "axis3d");
  xTitle.position.set(0, -0.05, D / 2 + 0.95);
  frame.add(xTitle);
  const yTitle = label(atlas.y.label, "axis3d");
  yTitle.position.set(W / 2 + 1.25, 0, -0.2);
  frame.add(yTitle);

  // ---------------------------------------------------------------- training envelopes
  // Train instruments in neutral white, randomised twins in the robust violet, as in the fleet panel.
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

  // ---------------------------------------------------------------- surfaces
  interface Surface {
    key: ModelKey;
    group: THREE.Group;
    mesh: THREE.Mesh | null;
    shown: number;
    target: number;
  }
  const NX = (nx - 1) * UP + 1;
  const NY = (ny - 1) * UP + 1;
  const surfaces = new Map<ModelKey, Surface>();

  for (const key of models) {
    const grid = grids.get(key)!;
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
    let mesh: THREE.Mesh | null = null;
    if (key !== "svm") {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
      const idx: number[] = [];
      for (let j = 0; j < NY - 1; j++) {
        for (let i = 0; i < NX - 1; i++) {
          const a = j * NX + i;
          idx.push(a, a + NX, a + 1, a + 1, a + NX, a + NX + 1);
        }
      }
      g.setIndex(idx);
      g.computeVertexNormals();
      const translucent = key === "robust";
      mesh = new THREE.Mesh(
        g,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          roughness: 0.45,
          metalness: 0.05,
          transparent: translucent,
          opacity: translucent ? 0.58 : 1,
          depthWrite: !translucent,
          emissive: new THREE.Color(0x0a0d14),
        }),
      );
      mesh.renderOrder = translucent ? 2 : 1;
      group.add(mesh);
    }
    const wire: number[] = [];
    const wireCol: number[] = [];
    const push = (k: number) => {
      wire.push(pos[3 * k], pos[3 * k + 1] + 0.004, pos[3 * k + 2]);
      const bright = key === "svm" ? 1.25 : key === "robust" ? 0.9 : 0.5;
      wireCol.push(col[3 * k] * bright, col[3 * k + 1] * bright, col[3 * k + 2] * bright);
    };
    for (let j = 0; j < NY; j += UP) for (let i = 0; i < NX - 1; i++) push(j * NX + i), push(j * NX + i + 1);
    for (let i = 0; i < NX; i += UP) for (let j = 0; j < NY - 1; j++) push(j * NX + i), push((j + 1) * NX + i);
    const wg = new THREE.BufferGeometry();
    wg.setAttribute("position", new THREE.Float32BufferAttribute(wire, 3));
    wg.setAttribute("color", new THREE.Float32BufferAttribute(wireCol, 3));
    group.add(new THREE.LineSegments(wg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: key === "naive" ? 0.5 : 0.95 })));
    surfaces.set(key, { key, group, mesh, shown: 0, target: 0 });
  }

  // ---------------------------------------------------------------- UI
  const root = h("div");
  const iconStyle: Record<ModelKey, string> = {
    naive: `background:${cssGradient(viridis, 0, 0.6)}`,
    robust: `background:${cssGradient(viridis, 0.5, 1)};opacity:.6;outline:1px solid ${CSS.accent}`,
    svm: `background:repeating-linear-gradient(90deg, ${CSS.text} 0 1px, transparent 1px 6px), repeating-linear-gradient(0deg, ${CSS.text} 0 1px, transparent 1px 6px);opacity:.7`,
  };
  const styleName: Record<ModelKey, string> = { naive: "solid", robust: "translucent", svm: "wireframe" };
  root.innerHTML = `
    <div class="panel atlas-left">
      <p class="eyebrow">Models</p>
      <div class="model-toggle"></div>
      <p class="eyebrow" style="margin-top:18px">${atlas.metric.replace(/_/g, " ")}</p>
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
    <div class="hint"><kbd>Space</kbd> add model · <kbd>R</kbd> rotate</div>`;
  ctx.ui.appendChild(root);

  const toggles = q(root, ".model-toggle");
  for (const m of models) {
    const stat = fleet?.summary[m]?.mean_bacc;
    const b = h(
      "button",
      { "data-m": m, title: `${MODEL_NAMES[m]} · ${styleName[m]}` },
      `<span class="style-icon" style="${iconStyle[m]}"></span>${MODEL_NAMES[m]}<span class="stat">${stat !== undefined ? `fleet ${fmt(stat)}` : ""}</span>`,
    );
    b.addEventListener("click", () => {
      const s = surfaces.get(m)!;
      s.target = s.target ? 0 : 1;
      renderToggles();
    });
    toggles.appendChild(b);
  }
  function renderToggles(): void {
    toggles.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
      b.classList.toggle("on", surfaces.get(b.dataset.m as ModelKey)!.target === 1);
    });
  }

  let step = 0;
  function setStep(i: number): void {
    step = i;
    for (const [k, s] of surfaces) s.target = steps[i]?.includes(k) ? 1 : 0;
    renderToggles();
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
  const probeLine = new THREE.Line(probeLineGeo, lineMat(0xffffff, 0.6));
  probe.add(probeLine);
  const probeDot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(2, 2, 2), toneMapped: false }));
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
    const meshes = [...surfaces.values()].filter((s) => s.target === 1 && s.mesh).map((s) => s.mesh!);
    if (!pointer || meshes.length === 0) {
      tooltip.style.opacity = "0";
      probe.visible = false;
      return;
    }
    ndc.set((pointer.x / ctx.engine.width) * 2 - 1, -(pointer.y / ctx.engine.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObjects(meshes, false)[0];
    if (!hit) {
      tooltip.style.opacity = "0";
      probe.visible = false;
      canvas.style.cursor = "";
      return;
    }
    const gx = Math.round(((hit.point.x + W / 2) / W) * (nx - 1));
    const gy = Math.round(((D / 2 - hit.point.z) / D) * (ny - 1));
    const ix = Math.min(nx - 1, Math.max(0, gx));
    const iy = Math.min(ny - 1, Math.max(0, gy));
    const rows = models
      .filter((m) => surfaces.get(m)!.target === 1)
      .map((m) => {
        const v = atlas.models[m]?.[iy]?.[ix];
        if (typeof v !== "number") return `<div class="row"><span>${MODEL_NAMES[m]}</span><span class="mono muted">–</span></div>`;
        const [r, g, b] = viridis(colorT(v));
        return `<div class="row"><span>${MODEL_NAMES[m]}</span><span class="mono" style="color:rgb(${r * 255},${g * 255},${b * 255})">${v.toFixed(3)}</span></div>`;
      })
      .join("");
    tooltip.innerHTML = `<div class="row muted"><span>${atlas.x.label}</span><span class="mono">${num(xs[ix])}</span></div>
      <div class="row muted" style="margin-bottom:6px"><span>${atlas.y.label}</span><span class="mono">${num(ys[iy])}</span></div>${rows}`;
    tooltip.style.left = `${pointer.x}px`;
    tooltip.style.top = `${pointer.y}px`;
    tooltip.style.opacity = "1";
    const top = Math.max(...models.filter((m) => surfaces.get(m)!.target === 1).map((m) => fy(grids.get(m)![iy][ix])));
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
        s.shown = damp(s.shown, s.target, 4.5, dt);
        if (Math.abs(s.shown - s.target) < 0.002) s.shown = s.target;
        s.group.visible = s.shown > 0.002;
        s.group.scale.y = Math.max(0.001, easeOutCubic(s.shown));
        if (s.shown !== s.target) dirty = true;
      }
      if (column) (column.material as THREE.MeshBasicMaterial).opacity = 0.04 + 0.02 * Math.sin(t * 2);
      if (dirty || controls.autoRotate) {
        dirty = false;
        hover();
      }
    },
    resize(w, hgt) {
      const narrow = w < 1280;
      frameFreeArea(camera, w, hgt, { left: narrow ? 306 : 336, right: narrow ? 346 : 416, top: 70, bottom: 30 }, 1.12);
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

function renderFleet(el: HTMLElement, fleet: Fleet): void {
  const s = fleet.summary;
  const ci = (m: "naive" | "robust") => {
    const c = finitePair(s[m]?.mean_bacc_ci95);
    return c ? `<span style="color:${m === "naive" ? CSS.naive : CSS.robust}">${fmt(c[0])}–${fmt(c[1])}</span>` : "";
  };
  const ciParts = [ci("naive"), ci("robust")].filter(Boolean);
  const ciLine = ciParts.length
    ? `<div class="ci"><div>${ciParts.join('<span class="arrow">→</span>')}</div><div class="ci-label">95% CI · bootstrap over instruments &amp; patients</div></div>`
    : "";
  const kpi = (k: string, a?: number, b?: number) =>
    `<div class="kpi"><div class="k">${k}</div><div class="v"><span style="color:${CSS.naive}">${fmt(a)}</span><span class="arrow">→</span><span style="color:${CSS.robust}">${fmt(b)}</span></div></div>`;
  const rows = [...fleet.instruments].sort((a, b) => (a.role === b.role ? a.id - b.id : a.role === "train" ? -1 : 1));
  const w = (v?: number) => `${v === undefined ? 0 : Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * 100}%`;
  const lat = Object.entries(fleet.latency_ms)
    .map(([k, v]) => `<span class="chip mono">${k.replace(/_/g, " ").toUpperCase()} ${fmt(v, 1)} ms</span>`)
    .join("");
  el.innerHTML = `
    <p class="eyebrow">Fleet · leave-instruments-out</p>
    <div class="kpis">${kpi("Mean bAcc", s.naive?.mean_bacc, s.robust?.mean_bacc)}${kpi("Worst bAcc", s.naive?.worst_bacc, s.robust?.worst_bacc)}</div>${ciLine}
    <div class="legend" style="margin-bottom:8px">
      <span><i class="swatch" style="background:${CSS.naive}"></i>Naive CNN</span>
      <span><i class="swatch" style="background:${CSS.robust}"></i>Robust CNN</span>
      <span class="muted" style="margin-left:auto">0.5 → 1.0</span>
    </div>
    <div class="fleet-rows">${rows
      .map(
        (r) => `<div class="fleet-row">
        <span class="name ${r.role === "train" ? "train" : ""}">${r.name}${r.role === "train" ? '<span class="tag">TRAIN</span>' : ""}</span>
        <div class="fleet-bars">
          <div class="bar"><i data-w="${w(r.metrics.naive?.bacc)}" style="background:${CSS.naive}"></i></div>
          <div class="bar"><i data-w="${w(r.metrics.robust?.bacc)}" style="background:${CSS.robust}"></i></div>
        </div>
        <div class="fleet-vals"><span style="color:${CSS.naive}">${fmt(r.metrics.naive?.bacc)}</span><span class="arrow">→</span><span style="color:${CSS.robust}">${fmt(r.metrics.robust?.bacc)}</span></div>
      </div>`,
      )
      .join("")}</div>
    <p class="eyebrow" style="margin-top:12px">Latency per spectrum</p>
    <div class="controls-row">${lat}</div>`;
  requestAnimationFrame(() =>
    requestAnimationFrame(() => el.querySelectorAll<HTMLElement>("[data-w]").forEach((b) => (b.style.width = b.dataset.w!))),
  );
}
