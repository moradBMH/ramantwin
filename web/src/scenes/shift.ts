import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { disposeObject, easeInOutCubic, frameFreeArea, type View } from "../gl/engine";
import { CSS, INSTRUMENT_PALETTE, cssGradient, inferno, linear, linearHex } from "../gl/colors";
import { loadShift, type ShiftScene } from "../data";
import { fmt, h, q } from "../ui/dom";
import { LinePlot } from "../ui/plot";
import type { ViewContext } from "../main";

const STAGE_ORDER = ["raw", "orpl", "naive", "robust"];
const STAGE_LABEL: Record<string, string> = { raw: "Raw", orpl: "ORPL", naive: "Naive CNN", robust: "Robust CNN" };
const STORY: Record<string, string> = {
  raw: "Spectra cluster by instrument, not by diagnosis.",
  orpl: "ORPL processing pulls instruments together, but not all the way.",
  naive: "A CNN trained on two instruments separates them well. The rest of the fleet drifts.",
  robust: "Trained through randomised twins, the space is organised by diagnosis on every instrument.",
};
const SCALE = 3;
const STAGGER = 0.35;
const MORPH_S = 1.7;

type Mode = "class" | "instrument" | "error" | "ood";
const MODES: Array<{ key: Mode; label: string }> = [
  { key: "class", label: "Diagnosis" },
  { key: "instrument", label: "Instrument" },
  { key: "error", label: "Naive error" },
  { key: "ood", label: "OOD score" },
];

const VERT = /* glsl */ `
uniform vec4 uFrom;
uniform vec4 uTo;
uniform float uProgress;
uniform float uScale;
uniform float uProj;
attribute vec3 p0;
attribute vec3 p1;
attribute vec3 p2;
attribute vec3 p3;
attribute float seed;
attribute vec3 color;
attribute float size;
varying vec3 vColor;
varying float vDepth;
float easeIO(float t) { return t < 0.5 ? 4.0 * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0; }
void main() {
  float k = easeIO(clamp(uProgress * (1.0 + ${STAGGER.toFixed(2)}) - seed * ${STAGGER.toFixed(2)}, 0.0, 1.0));
  vec4 w = mix(uFrom, uTo, k);
  vec3 pos = (p0 * w.x + p1 * w.y + p2 * w.z + p3 * w.w) * uScale;
  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(1.5, 0.075 * size * uProj / -mv.z);
  vColor = color;
  vDepth = clamp((-mv.z - 6.0) / 8.0, 0.0, 1.0);
}`;

const FRAG = /* glsl */ `
varying vec3 vColor;
varying float vDepth;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c) * 2.0;
  if (d > 1.0) discard;
  float core = smoothstep(0.62, 0.15, d);
  float halo = exp(-d * d * 4.5) * 0.45;
  float a = clamp(core + halo, 0.0, 1.0) * mix(1.0, 0.55, vDepth);
  gl_FragColor = vec4(vColor * (0.75 + 0.6 * core), a);
}`;

export async function createShift(ctx: ViewContext): Promise<View> {
  const data: ShiftScene = await loadShift();
  const n = data.points.label.length;
  const stages = STAGE_ORDER.map((k) => data.stages.find((s) => s.key === k)).filter((s) => s !== undefined);
  if (stages.length === 0) stages.push(...data.stages.slice(0, 4));
  while (stages.length < 4) stages.push(stages[stages.length - 1]);
  const nStages = Math.min(4, data.stages.length);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(6.2, 3.4, 7.4);
  const controls = new OrbitControls(camera, ctx.engine.renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 4;
  controls.maxDistance = 20;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.35;

  // ---------------------------------------------------------------- geometry
  const geo = new THREE.BufferGeometry();
  stages.forEach((s, i) => {
    const arr = new Float32Array(n * 3);
    arr.set(s.positions.slice(0, n * 3));
    geo.setAttribute(`p${i}`, new THREE.BufferAttribute(arr, 3));
  });
  geo.setAttribute("position", geo.getAttribute("p0"));
  const seeds = new Float32Array(n);
  for (let i = 0; i < n; i++) seeds[i] = ((i * 2654435761) % 1000) / 1000;
  geo.setAttribute("seed", new THREE.BufferAttribute(seeds, 1));
  const colorAttr = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
  const sizeAttr = new THREE.BufferAttribute(new Float32Array(n).fill(1), 1);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  sizeAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("color", colorAttr);
  geo.setAttribute("size", sizeAttr);
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), SCALE * 2);

  const uniforms = {
    uFrom: { value: new THREE.Vector4(1, 0, 0, 0) },
    uTo: { value: new THREE.Vector4(1, 0, 0, 0) },
    uProgress: { value: 1 },
    uScale: { value: SCALE },
    uProj: { value: 1000 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
  });
  const points = new THREE.Points(geo, mat);
  scene.add(points);

  const box = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(SCALE * 2, SCALE * 2, SCALE * 2)),
    new THREE.LineBasicMaterial({ color: 0x8f9aad, transparent: true, opacity: 0.08 }),
  );
  scene.add(box);

  // ---------------------------------------------------------------- colours
  const P = data.points;
  const isTrain = new Map(data.instruments.map((ins) => [ins.id, ins.role === "train"]));
  const instName = new Map(data.instruments.map((ins) => [ins.id, ins.name]));
  const palette = data.instruments.map((ins, i) => ({ id: ins.id, rgb: linearHex(INSTRUMENT_PALETTE[i % INSTRUMENT_PALETTE.length]) }));
  const instColor = new Map(palette.map((p) => [p.id, p.rgb]));
  const classRgb = [linearHex(CSS.normal), linearHex(CSS.cancer)];
  const oodSorted = [...P.ood].sort((a, b) => a - b);
  const oodLo = oodSorted[Math.floor(oodSorted.length * 0.02)] ?? 0;
  const oodHi = oodSorted[Math.floor(oodSorted.length * 0.98)] ?? 1;

  function target(mode: Mode): { col: Float32Array; size: Float32Array } {
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let rgb: [number, number, number];
      let s = 1;
      if (mode === "class") rgb = classRgb[P.label[i] ? 1 : 0];
      else if (mode === "instrument") {
        rgb = instColor.get(P.instrument[i]) ?? [1, 1, 1];
        const train = isTrain.get(P.instrument[i]);
        s = train ? 1.55 : 0.9;
        if (!train) rgb = [rgb[0] * 0.55, rgb[1] * 0.55, rgb[2] * 0.55];
      } else if (mode === "error") {
        const err = Math.abs(P.p_naive[i] - P.label[i]);
        const c = linear(inferno(0.12 + 0.88 * err));
        rgb = [c.r, c.g, c.b];
        s = 0.8 + 0.9 * err;
      } else {
        const t = (P.ood[i] - oodLo) / (oodHi - oodLo || 1);
        const c = linear(inferno(0.12 + 0.88 * Math.min(1, Math.max(0, t))));
        rgb = [c.r, c.g, c.b];
        s = 0.8 + 0.7 * Math.min(1, Math.max(0, t));
      }
      col.set(rgb, i * 3);
      size[i] = s;
    }
    return { col, size };
  }

  let mode: Mode = "class";
  let colFrom = target(mode);
  let colTo = colFrom;
  let colT = 1;
  (colorAttr.array as Float32Array).set(colTo.col);
  (sizeAttr.array as Float32Array).set(colTo.size);

  function setMode(m: Mode): void {
    if (m === mode) return;
    mode = m;
    colFrom = { col: (colorAttr.array as Float32Array).slice(), size: (sizeAttr.array as Float32Array).slice() };
    colTo = target(m);
    colT = 0;
    renderModeUI();
  }

  // ---------------------------------------------------------------- stages
  let stage = 0;
  const onehot = (i: number) => new THREE.Vector4(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0, i === 3 ? 1 : 0);

  function setStage(i: number): void {
    i = Math.max(0, Math.min(nStages - 1, i));
    if (i === stage && uniforms.uProgress.value >= 1) return;
    const e = easeInOutCubic(Math.min(1, uniforms.uProgress.value));
    uniforms.uFrom.value.lerp(uniforms.uTo.value, e);
    uniforms.uTo.value.copy(onehot(i));
    uniforms.uProgress.value = 0;
    stage = i;
    renderStageUI();
  }

  // ---------------------------------------------------------------- UI
  const root = h("div");
  root.innerHTML = `
    <div class="story"><div class="fade"><h2></h2><p></p></div></div>
    <div class="stepper"></div>
    <div class="panel side">
      <section>
        <p class="heading">Colour by</p>
        <div class="seg modes"></div>
        <div class="legend-wrap" style="margin-top:12px"></div>
      </section>
      <section>
        <p class="heading">Silhouette score</p>
        <div class="sil-row">
          <div class="sil-head"><span>by instrument</span><span class="num sil-i"></span></div>
          <div class="bar"><i class="bar-i" style="background:${CSS.muted}"></i></div>
        </div>
        <div class="sil-row" style="margin-bottom:0">
          <div class="sil-head"><span>by diagnosis</span><span class="num sil-c"></span></div>
          <div class="bar"><i class="bar-c" style="background:${CSS.text}"></i></div>
        </div>
      </section>
      <section class="inspector"></section>
    </div>
    <div class="hint"><kbd>Space</kbd> next stage · <kbd>C</kbd> colour · <kbd>R</kbd> rotate</div>`;
  ctx.ui.appendChild(root);

  const stepper = q(root, ".stepper");
  stages.slice(0, nStages).forEach((s, i) => {
    if (i > 0) stepper.appendChild(h("div", { class: "step-line" }, "<i></i>"));
    const b = h("button", { class: "step" }, `<span class="node"></span>${STAGE_LABEL[s.key] ?? s.title}`);
    b.addEventListener("click", () => setStage(i));
    stepper.appendChild(b);
  });

  const modesEl = q(root, ".modes");
  for (const m of MODES) {
    const b = h("button", { "data-mode": m.key }, m.label);
    b.addEventListener("click", () => setMode(m.key));
    modesEl.appendChild(b);
  }

  function renderStageUI(): void {
    const s = stages[stage];
    const story = q(root, ".story");
    const caption = s.subtitle?.trim() || STORY[s.key] || "";
    story.innerHTML = `<div class="fade"><h2>${s.title}</h2><p>${caption}</p></div>`;
    stepper.querySelectorAll(".step").forEach((el, i) => {
      el.classList.toggle("active", i === stage);
      el.classList.toggle("done", i < stage);
    });
    stepper.querySelectorAll(".step-line").forEach((el, i) => el.classList.toggle("done", i < stage));
    const si = s.silhouette.instrument;
    const sc = s.silhouette.class;
    q(root, ".sil-i").textContent = fmt(si);
    q(root, ".sil-c").textContent = fmt(sc);
    q(root, ".bar-i").style.width = `${Math.max(0, si) * 100}%`;
    q(root, ".bar-c").style.width = `${Math.max(0, sc) * 100}%`;
  }

  function renderModeUI(): void {
    modesEl.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    const wrap = q(root, ".legend-wrap");
    if (mode === "class") {
      wrap.innerHTML = `<div class="legend">${data.classes
        .map((c, i) => `<span><i class="swatch" style="background:${i ? CSS.cancer : CSS.normal}"></i>${c}</span>`)
        .join("")}</div>`;
    } else if (mode === "instrument") {
      const item = (train: boolean) =>
        data.instruments
          .map((ins, i) => ({ ins, color: INSTRUMENT_PALETTE[i % INSTRUMENT_PALETTE.length] }))
          .filter(({ ins }) => (ins.role === "train") === train)
          .map(({ ins, color }) => `<span class="${train ? "train" : ""}"><i class="swatch" style="background:${color};${train ? "" : "opacity:.6"}"></i>${ins.name}</span>`)
          .join("");
      wrap.innerHTML = `<p class="legend-group">Training instruments</p><div class="legend instruments">${item(true)}</div>
        <p class="legend-group">Fleet</p><div class="legend instruments">${item(false)}</div>`;
    } else {
      const lo = mode === "error" ? "Naive prediction: correct" : "OOD score: low";
      const hi = mode === "error" ? "wrong" : "high";
      wrap.innerHTML = `<div class="gradient-bar" style="background:${cssGradient(inferno, 0.12, 1)}"></div>
        <div class="gradient-labels"><span>${lo}</span><span>${hi}</span></div>`;
    }
  }

  // ---------------------------------------------------------------- inspector
  const inspector = q(root, ".inspector");
  const spectraRow = new Map(data.spectra.index.map((p, r) => [p, r]));
  inspector.innerHTML = `<p class="heading">Selected spectrum</p>
    <div class="insp-body"><p class="empty">Hover a point to inspect it. Click to pin.<br/>Ringed points carry a stored spectrum.</p></div>
    <div class="insp-plots" style="display:none">
      <div class="plot-title"><span class="key"><i style="background:${CSS.muted}"></i>raw, normalised</span></div>
      <div class="plot" style="height:92px"></div>
      <div class="plot-title"><span class="key"><i class="pk"></i>ORPL processed (SNV)</span></div>
      <div class="plot" style="height:92px"></div>
    </div>`;
  const plotEls = inspector.querySelectorAll<HTMLElement>(".plot");
  const grid = data.spectra.grid;
  const xr: [number, number] = [grid[0] ?? 600, grid[grid.length - 1] ?? 1800];
  const ticks = [600, 1000, 1400, 1800].filter((t) => t >= xr[0] && t <= xr[1]);
  const rawPlot = new LinePlot(plotEls[0], { xRange: xr, xTicks: ticks });
  const procPlot = new LinePlot(plotEls[1], { xRange: xr, xTicks: ticks, zeroLine: true });

  let shown = -1;
  function inspect(i: number): void {
    if (i === shown) return;
    shown = i;
    const body = q(inspector, ".insp-body");
    const plots = q(inspector, ".insp-plots");
    if (i < 0) {
      body.innerHTML = `<p class="empty">Hover a point to inspect it. Click to pin.<br/>Ringed points carry a stored spectrum.</p>`;
      plots.style.display = "none";
      return;
    }
    const label = P.label[i];
    const cls = data.classes[label] ?? String(label);
    const train = isTrain.get(P.instrument[i]);
    const pn = P.p_naive[i];
    const pr = P.p_robust[i];
    const ok = (p: number) => ((p >= 0.5 ? 1 : 0) === label ? '<span class="verdict right">✓</span>' : '<span class="verdict wrong">✗</span>');
    body.innerHTML = `<dl class="meta">
      <dt>Instrument</dt><dd>${instName.get(P.instrument[i]) ?? P.instrument[i]}</dd>
      <dt>Role</dt><dd>${train ? "train" : "fleet"}</dd>
      <dt>Patient</dt><dd>#${P.patient[i]}</dd>
      <dt>Diagnosis</dt><dd class="${label ? "c-cancer" : "c-normal"}">${cls}</dd>
      <dt>P(cancer) naive</dt><dd>${fmt(pn)} ${ok(pn)}</dd>
      <dt>P(cancer) robust</dt><dd>${fmt(pr)} ${ok(pr)}</dd>
      <dt>OOD score</dt><dd>${fmt(P.ood[i], 1)}</dd>
    </dl>`;
    const row = spectraRow.get(i);
    if (row === undefined) {
      plots.style.display = "none";
      return;
    }
    plots.style.display = "";
    const color = label ? CSS.cancer : CSS.normal;
    q(inspector, ".pk").style.background = color;
    rawPlot.set([{ x: grid, y: data.spectra.raw[row], color: CSS.muted, width: 1.4, fill: true }]);
    procPlot.set([{ x: grid, y: data.spectra.processed[row], color, width: 1.4 }]);
  }

  // ---------------------------------------------------------------- picking
  const canvas = ctx.engine.renderer.domElement;
  const hasSpectrum = new Uint8Array(n);
  for (const p of data.spectra.index) if (p < n) hasSpectrum[p] = 1;
  const pointer = { x: 0, y: 0, moved: false, inside: false, downX: 0, downY: 0 };
  let hovered = -1;
  let pinned = -1;
  const marker = h("div", { class: "marker" });
  marker.style.opacity = "0";
  root.appendChild(marker);
  const tmp = new THREE.Vector3();

  const stageArrays = [0, 1, 2, 3].map((j) => geo.getAttribute(`p${j}`).array as Float32Array);
  const wFrom = [0, 0, 0, 0];
  const wTo = [0, 0, 0, 0];

  function pointPosition(i: number, out: THREE.Vector3): THREE.Vector3 {
    let k = Math.min(1, Math.max(0, uniforms.uProgress.value * (1 + STAGGER) - seeds[i] * STAGGER));
    k = easeInOutCubic(k);
    out.set(0, 0, 0);
    for (let j = 0; j < 4; j++) {
      const w = wFrom[j] * (1 - k) + wTo[j] * k;
      if (w === 0) continue;
      const a = stageArrays[j];
      out.x += a[3 * i] * w;
      out.y += a[3 * i + 1] * w;
      out.z += a[3 * i + 2] * w;
    }
    return out.multiplyScalar(SCALE);
  }

  const screen = { x: 0, y: 0, z: 0 };
  function toScreen(i: number): typeof screen {
    pointPosition(i, tmp).project(camera);
    screen.x = ((tmp.x + 1) / 2) * ctx.engine.width;
    screen.y = ((1 - tmp.y) / 2) * ctx.engine.height;
    screen.z = tmp.z;
    return screen;
  }

  function pick(): number {
    uniforms.uFrom.value.toArray(wFrom);
    uniforms.uTo.value.toArray(wTo);
    let best = -1;
    let bestD = 18 * 18;
    let bestAny = -1;
    let bestAnyD = 9 * 9;
    for (let i = 0; i < n; i++) {
      const { x, y, z } = toScreen(i);
      if (z > 1) continue;
      const d = (x - pointer.x) ** 2 + (y - pointer.y) ** 2;
      if (hasSpectrum[i] && d < bestD) {
        bestD = d;
        best = i;
      }
      if (d < bestAnyD) {
        bestAnyD = d;
        bestAny = i;
      }
    }
    return best >= 0 ? best : bestAny;
  }

  const onMove = (e: PointerEvent) => {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    pointer.moved = true;
    pointer.inside = true;
  };
  const onLeave = () => {
    pointer.inside = false;
    pointer.moved = true;
  };
  const onDown = (e: PointerEvent) => {
    pointer.downX = e.clientX;
    pointer.downY = e.clientY;
  };
  const onUp = (e: PointerEvent) => {
    if (Math.hypot(e.clientX - pointer.downX, e.clientY - pointer.downY) > 4) return;
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    const i = pick();
    pinned = i === pinned ? -1 : i;
    inspect(pinned >= 0 ? pinned : i);
  };
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointerup", onUp);

  // Rings around points that carry a spectrum, drawn as a second, sparse point layer.
  const ringIdx = data.spectra.index.filter((p) => p < n);
  const ringGeo = new THREE.BufferGeometry();
  for (let j = 0; j < 4; j++) {
    const src = geo.getAttribute(`p${j}`).array as Float32Array;
    const arr = new Float32Array(ringIdx.length * 3);
    ringIdx.forEach((p, r) => arr.set(src.subarray(3 * p, 3 * p + 3), 3 * r));
    ringGeo.setAttribute(`p${j}`, new THREE.BufferAttribute(arr, 3));
  }
  ringGeo.setAttribute("position", ringGeo.getAttribute("p0"));
  ringGeo.setAttribute("seed", new THREE.BufferAttribute(Float32Array.from(ringIdx.map((p) => seeds[p])), 1));
  ringGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(ringIdx.length * 3).fill(0.55), 3));
  ringGeo.setAttribute("size", new THREE.BufferAttribute(new Float32Array(ringIdx.length).fill(2.1), 1));
  ringGeo.boundingSphere = geo.boundingSphere.clone();
  const ringMat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: VERT,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      varying float vDepth;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float ring = smoothstep(0.12, 0.0, abs(d - 0.78));
        if (ring < 0.01) discard;
        gl_FragColor = vec4(vColor, ring * mix(0.7, 0.35, vDepth));
      }`,
    transparent: true,
    depthWrite: false,
  });
  const rings = new THREE.Points(ringGeo, ringMat);
  scene.add(rings);

  renderStageUI();
  renderModeUI();

  return {
    scene,
    camera,
    bloom: { strength: 0.75, radius: 0.45, threshold: 0.32 },
    update(dt) {
      controls.update(dt);
      if (uniforms.uProgress.value < 1) {
        uniforms.uProgress.value = Math.min(1, uniforms.uProgress.value + dt / MORPH_S);
        pointer.moved = true;
      }
      if (colT < 1) {
        colT = Math.min(1, colT + dt / 0.6);
        const e = easeInOutCubic(colT);
        const c = colorAttr.array as Float32Array;
        const s = sizeAttr.array as Float32Array;
        for (let i = 0; i < c.length; i++) c[i] = colFrom.col[i] + (colTo.col[i] - colFrom.col[i]) * e;
        for (let i = 0; i < s.length; i++) s[i] = colFrom.size[i] + (colTo.size[i] - colFrom.size[i]) * e;
        colorAttr.needsUpdate = true;
        sizeAttr.needsUpdate = true;
      }
      if (pointer.moved || controls.autoRotate) {
        pointer.moved = false;
        hovered = pointer.inside ? pick() : -1;
        if (pinned < 0) inspect(hovered);
        canvas.style.cursor = hovered >= 0 ? "pointer" : "";
      }
      const active = pinned >= 0 ? pinned : hovered;
      if (active >= 0) {
        uniforms.uFrom.value.toArray(wFrom);
        uniforms.uTo.value.toArray(wTo);
        const { x, y } = toScreen(active);
        marker.style.transform = `translate(${x}px, ${y}px)`;
        marker.style.left = "0";
        marker.style.top = "0";
        marker.style.opacity = "1";
        marker.style.borderWidth = pinned >= 0 ? "3px" : "1.5px";
      } else marker.style.opacity = "0";
    },
    resize(w, hgt) {
      const narrow = w < 1280 ? 334 : 408;
      frameFreeArea(camera, w, hgt, { left: 24, right: narrow, top: 70, bottom: 190 }, 1.05);
      uniforms.uProj.value = (ctx.engine.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) * camera.zoom;
    },
    next() {
      if (stage >= nStages - 1) return false;
      setStage(stage + 1);
      return true;
    },
    prev() {
      if (stage <= 0) return false;
      setStage(stage - 1);
      return true;
    },
    onKey(e) {
      if (e.key === "c") {
        const i = MODES.findIndex((m) => m.key === mode);
        setMode(MODES[(i + 1) % MODES.length].key);
        return true;
      }
      if (e.key === "r") {
        controls.autoRotate = !controls.autoRotate;
        return true;
      }
      if (e.key === "Escape") {
        pinned = -1;
        inspect(hovered);
        return true;
      }
      return false;
    },
    dispose() {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.style.cursor = "";
      controls.dispose();
      rawPlot.dispose();
      procPlot.dispose();
      disposeObject(scene);
      root.remove();
    },
  };
}
