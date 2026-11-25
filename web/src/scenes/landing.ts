import * as THREE from "three";
import { disposeObject, frameFreeArea, type View } from "../gl/engine";
import { CSS, linearHex } from "../gl/colors";
import { mulberry32, tissueMix, tissueSpectrum } from "../gl/spectra";
import { h } from "../ui/dom";
import type { ViewContext } from "../main";

const N_LINES = 54;
const N_PTS = 360;
const SPACING = 0.34;
const X_HALF = 7.5;
const HEIGHT = 1.9;

export async function createLanding(ctx: ViewContext): Promise<View> {
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x05070b, 7, 21);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 3.4, 10.5);
  camera.lookAt(0, 0.2, -3);

  const group = new THREE.Group();
  group.position.set(1.2, -0.6, 0);
  scene.add(group);

  const rnd = mulberry32(785);
  const blue = linearHex(CSS.normal);
  const orange = linearHex(CSS.cancer);
  const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 });
  const ribbonMat = new THREE.MeshBasicMaterial({
    color: 0x05070b,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });

  interface Ridge {
    line: THREE.Line;
    ribbon: THREE.Mesh;
  }
  const ridges: Ridge[] = [];

  const fill = (r: Ridge) => {
    const cancer = rnd() < 0.5 ? 0 : 1;
    const mix = tissueMix(cancer * (0.6 + 0.4 * rnd()), rnd);
    const offset = (rnd() - 0.5) * 14;
    const fluo = 0.15 + 0.25 * rnd();
    const pos = r.line.geometry.getAttribute("position") as THREE.BufferAttribute;
    const col = r.line.geometry.getAttribute("color") as THREE.BufferAttribute;
    const rib = r.ribbon.geometry.getAttribute("position") as THREE.BufferAttribute;
    const tint = cancer ? orange : blue;
    for (let i = 0; i < N_PTS; i++) {
      const u = i / (N_PTS - 1);
      const shift = 450 + u * 1400;
      const y = (tissueSpectrum(shift, mix, offset) + fluo * (1 - 0.6 * u)) * HEIGHT * 0.62;
      const x = -X_HALF + u * 2 * X_HALF;
      pos.setXYZ(i, x, y, 0);
      const b = 0.12 + 0.9 * Math.min(1, y / HEIGHT);
      col.setXYZ(i, (0.55 * tint[0] + 0.45) * b, (0.55 * tint[1] + 0.45) * b, (0.55 * tint[2] + 0.45) * b);
      rib.setXYZ(2 * i, x, y, 0);
      rib.setXYZ(2 * i + 1, x, -0.4, 0);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    rib.needsUpdate = true;
  };

  const ribbonIndex: number[] = [];
  for (let i = 0; i < N_PTS - 1; i++) {
    const a = 2 * i;
    ribbonIndex.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }

  for (let k = 0; k < N_LINES; k++) {
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N_PTS * 3), 3));
    lg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N_PTS * 3), 3));
    const rg = new THREE.BufferGeometry();
    rg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N_PTS * 6), 3));
    rg.setIndex(ribbonIndex);
    const r: Ridge = { line: new THREE.Line(lg, lineMat), ribbon: new THREE.Mesh(rg, ribbonMat) };
    r.line.frustumCulled = false;
    r.ribbon.frustumCulled = false;
    const z = 3 - k * SPACING;
    r.line.position.z = z;
    r.ribbon.position.z = z;
    fill(r);
    group.add(r.ribbon, r.line);
    ridges.push(r);
  }

  const floor = new THREE.GridHelper(40, 60, 0x1a2233, 0x0e1420);
  floor.position.y = -0.62;
  scene.add(floor);

  const root = h("div");
  root.innerHTML = `
    <div class="landing-scrim"></div>
    <div class="landing">
      <h1>RamanTwin</h1>
      <p class="question">Is an AI model validated on one Raman instrument safe on the next one?</p>
      <p class="lede">A digital twin of the instrument fleet measures the risk, hardens the model, and flags spectra outside the validated envelope.</p>
      <div class="actions">
        <button class="btn primary" data-go="shift">Start the story &rarr;</button>
        <button class="btn" data-go="twin">Twin Lab</button>
        <button class="btn" data-go="atlas">Robustness Atlas</button>
      </div>
    </div>
    <div class="footer">Simulated data · processing by ORPL (Sheehy et al., JBO 2023)</div>
    <div class="hint"><kbd>Space</kbd> next · <kbd>1</kbd>–<kbd>4</kbd> jump · <kbd>F</kbd> fullscreen</div>`;
  root.querySelectorAll<HTMLButtonElement>("[data-go]").forEach((b) => {
    b.addEventListener("click", () => ctx.go(b.dataset.go!));
  });
  ctx.ui.appendChild(root);

  const front = 3.4;
  const depth = N_LINES * SPACING;

  return {
    scene,
    camera,
    bloom: { strength: 1.05, radius: 0.55, threshold: 0.18 },
    update(dt, t) {
      for (const r of ridges) {
        let z = r.line.position.z + dt * 0.55;
        if (z > front) {
          z -= depth;
          fill(r);
        }
        r.line.position.z = z;
        r.ribbon.position.z = z;
      }
      group.rotation.y = Math.sin(t * 0.08) * 0.2 - 0.12;
      camera.position.x = Math.sin(t * 0.05) * 0.6;
      camera.lookAt(0, 0.2, -3);
    },
    resize(w, hgt) {
      frameFreeArea(camera, w, hgt, { left: w * 0.22, right: 0, top: 0, bottom: 0 }, 1.15);
    },
    next() {
      return false;
    },
    prev() {
      return false;
    },
    dispose() {
      disposeObject(scene);
      root.remove();
    },
  };
}
