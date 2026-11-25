import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { label } from "../gl/engine";
import { inferno } from "../gl/colors";

const H = 0.8;
const N_CCD = 1024;
const N_RAYS = 72;
const N_PHOTONS = 260;

export interface Instrument {
  group: THREE.Group;
  setSpectrum(y: number[]): void;
  setDrift(nm: number): void;
  setSample(label: number): void;
  update(dt: number, t: number): void;
}

function beam(from: THREE.Vector3, to: THREE.Vector3, r0: number, r1: number, mat: THREE.Material): THREE.Mesh {
  const dir = new THREE.Vector3().subVectors(to, from);
  const len = dir.length();
  const geo = new THREE.CylinderGeometry(r1, r0, len, 32, 1, true);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(from).addScaledVector(dir, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return mesh;
}

function edges(mesh: THREE.Mesh, opacity = 0.35): THREE.LineSegments {
  const l = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry, 30),
    new THREE.LineBasicMaterial({ color: 0xb9a5ff, transparent: true, opacity }),
  );
  mesh.add(l);
  return l;
}

function gratingTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 64;
  const g = c.getContext("2d")!;
  for (let x = 0; x < 512; x += 2) {
    const hue = 200 + 140 * Math.sin(x / 70);
    g.fillStyle = `hsl(${hue}, 70%, ${x % 4 === 0 ? 38 : 18}%)`;
    g.fillRect(x, 0, 2, 64);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function dotTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.5)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export function buildInstrument(): Instrument {
  const group = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: 0x161c28, metalness: 0.65, roughness: 0.38 });

  // Bench.
  const bench = new THREE.Mesh(
    new RoundedBoxGeometry(13, 0.24, 6.4, 4, 0.1),
    new THREE.MeshStandardMaterial({ color: 0x0a0e15, metalness: 0.3, roughness: 0.75 }),
  );
  bench.position.set(0, -0.12, -0.1);
  group.add(bench);
  const grid = new THREE.GridHelper(12.6, 42, 0x1b2333, 0x121925);
  grid.position.y = 0.002;
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.7;
  grid.scale.z = 6.2 / 12.6;
  group.add(grid);

  // Key points of the optical path.
  const S = new THREE.Vector3(-2.6, 0.38, 1.25);
  const L0 = new THREE.Vector3(-4.55, 1.35, -0.85);
  const C = new THREE.Vector3(-0.85, H, 0.25);
  const SL = new THREE.Vector3(0.65, H, -0.65);
  const G = new THREE.Vector3(2.75, H, -1.75);
  const CCD = new THREE.Vector3(4.35, H, 0.95);

  // Laser.
  const laserDir = new THREE.Vector3().subVectors(S, L0).normalize();
  const laser = new THREE.Mesh(new RoundedBoxGeometry(0.62, 0.62, 1.7, 4, 0.08), metal);
  laser.position.copy(L0).addScaledVector(laserDir, -0.85);
  laser.lookAt(S);
  edges(laser);
  group.add(laser);
  const laserColor = new THREE.Color(0xff1a2e);
  const coreMat = new THREE.MeshBasicMaterial({ color: laserColor.clone().multiplyScalar(4), toneMapped: false });
  const glowMat = new THREE.MeshBasicMaterial({
    color: laserColor.clone().multiplyScalar(1.3),
    transparent: true,
    opacity: 0.22,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  group.add(beam(L0, S, 0.022, 0.022, coreMat), beam(L0, S, 0.075, 0.06, glowMat));
  const aperture = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.025, 12, 32), coreMat);
  aperture.position.copy(L0);
  aperture.lookAt(S);
  group.add(aperture);
  const laserLabel = label('785 nm laser<span class="sub">λ 785.00 nm</span>');
  laserLabel.position.copy(laser.position).add(new THREE.Vector3(0, 0.62, 0));
  group.add(laserLabel);

  // Tissue sample with the laser spot.
  const tissueMat = new THREE.MeshStandardMaterial({ color: 0xd98a94, roughness: 0.55, metalness: 0.0, emissive: 0x2a0a10 });
  const tissue = new THREE.Mesh(new RoundedBoxGeometry(1.4, 0.36, 1.0, 5, 0.14), tissueMat);
  tissue.position.set(S.x, 0.19, S.z);
  group.add(tissue);
  const spot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 16), coreMat);
  spot.position.copy(S);
  group.add(spot);
  const spotLight = new THREE.PointLight(0xff2a3a, 4, 3.5, 1.6);
  spotLight.position.copy(S).add(new THREE.Vector3(0, 0.15, 0));
  group.add(spotLight);
  const sampleLabel = label("Tissue sample");
  sampleLabel.position.set(S.x, 0.62, S.z + 0.55);
  group.add(sampleLabel);

  // Collected Raman light: sample -> lens -> slit -> grating.
  const scatterMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(0xff5a72).multiplyScalar(0.9),
    transparent: true,
    opacity: 0.1,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  group.add(beam(S, C, 0.05, 0.4, scatterMat), beam(C, SL, 0.4, 0.03, scatterMat), beam(SL, G, 0.03, 0.42, scatterMat));

  const lensMat = new THREE.MeshStandardMaterial({
    color: 0x9fd8ff,
    transparent: true,
    opacity: 0.22,
    roughness: 0.05,
    metalness: 0.1,
    emissive: 0x0c2438,
  });
  const lensDir = new THREE.Vector3().subVectors(SL, S).normalize();
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.44, 0.07, 48), lensMat);
  lens.position.copy(C);
  lens.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), lensDir);
  group.add(lens);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.45, 0.025, 12, 64), metal);
  rim.position.copy(C);
  rim.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), lensDir);
  group.add(rim);
  const lensPost = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, H - 0.45, 12), metal);
  lensPost.position.set(C.x, (H - 0.45) / 2, C.z);
  group.add(lensPost);
  const lensLabel = label("Collection optics");
  lensLabel.position.set(C.x, H + 0.7, C.z);
  group.add(lensLabel);

  for (const side of [-1, 1]) {
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.5, 0.04), metal);
    blade.position.copy(SL);
    blade.lookAt(G);
    blade.translateX(side * 0.2);
    group.add(blade);
  }

  // Grating.
  const toSlit = new THREE.Vector3().subVectors(SL, G).normalize();
  const toCcd = new THREE.Vector3().subVectors(CCD, G).normalize();
  const normal = toSlit.clone().add(toCcd).normalize();
  const grating = new THREE.Mesh(
    new THREE.BoxGeometry(1.25, 1.05, 0.07),
    [metal, metal, metal, metal, new THREE.MeshStandardMaterial({ map: gratingTexture(), metalness: 0.9, roughness: 0.25, emissive: 0x111122 }), metal],
  );
  grating.position.copy(G);
  grating.lookAt(G.clone().add(normal));
  edges(grating, 0.25);
  group.add(grating);
  const gLabel = label("Diffraction grating");
  gLabel.position.set(G.x, H + 0.85, G.z);
  group.add(gLabel);

  // CCD with a 1024-pixel heat strip.
  const ccd = new THREE.Group();
  ccd.position.copy(CCD);
  ccd.lookAt(G);
  group.add(ccd);
  const housing = new THREE.Mesh(new RoundedBoxGeometry(2.7, 0.62, 0.4, 4, 0.06), metal);
  housing.position.z = -0.22;
  edges(housing);
  ccd.add(housing);
  const ccdData = new Uint8Array(N_CCD * 4);
  const ccdTex = new THREE.DataTexture(ccdData, N_CCD, 1, THREE.RGBAFormat);
  ccdTex.colorSpace = THREE.SRGBColorSpace;
  ccdTex.magFilter = THREE.LinearFilter;
  ccdTex.needsUpdate = true;
  const strip = new THREE.Mesh(
    new THREE.PlaneGeometry(2.36, 0.16),
    new THREE.MeshBasicMaterial({ map: ccdTex, color: new THREE.Color(1.6, 1.6, 1.6), toneMapped: false }),
  );
  strip.position.z = 0.0;
  ccd.add(strip);
  const ccdLabel = label('CCD<span class="sub">1024 px</span>');
  ccdLabel.position.set(0, 0.62, -0.2);
  ccd.add(ccdLabel);
  ccd.updateMatrixWorld(true);
  const stripA = new THREE.Vector3(-1.18, 0, 0.01).applyMatrix4(ccd.matrixWorld);
  const stripB = new THREE.Vector3(1.18, 0, 0.01).applyMatrix4(ccd.matrixWorld);

  // Dispersed fan from the grating to the CCD.
  const fanPos = new Float32Array(N_RAYS * 6);
  const fanCol = new Float32Array(N_RAYS * 6);
  const rayEnd: THREE.Vector3[] = [];
  for (let k = 0; k < N_RAYS; k++) {
    const end = stripA.clone().lerp(stripB, (k + 0.5) / N_RAYS);
    rayEnd.push(end);
    fanPos.set([G.x, G.y, G.z, end.x, end.y, end.z], k * 6);
  }
  const fanGeo = new THREE.BufferGeometry();
  fanGeo.setAttribute("position", new THREE.BufferAttribute(fanPos, 3));
  const fanColAttr = new THREE.BufferAttribute(fanCol, 3);
  fanGeo.setAttribute("color", fanColAttr);
  group.add(
    new THREE.LineSegments(
      fanGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }),
    ),
  );

  // Photons flowing along the optical path.
  const phPos = new Float32Array(N_PHOTONS * 3);
  const phCol = new Float32Array(N_PHOTONS * 3);
  const phGeo = new THREE.BufferGeometry();
  phGeo.setAttribute("position", new THREE.BufferAttribute(phPos, 3));
  phGeo.setAttribute("color", new THREE.BufferAttribute(phCol, 3));
  const phMat = new THREE.PointsMaterial({
    size: 0.09,
    map: dotTexture(),
    vertexColors: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const photons = new THREE.Points(phGeo, phMat);
  photons.frustumCulled = false;
  group.add(photons);
  interface Photon {
    path: THREE.Vector3[];
    lens: number[];
    total: number;
    s: number;
    speed: number;
    color: THREE.Color;
  }
  const raman = new THREE.Color(0xff4d6a);
  let intensity = new Float32Array(N_RAYS).fill(0.3);
  let cdf = new Float32Array(N_RAYS).map((_, i) => (i + 1) / N_RAYS);
  const spawn = (p: Photon) => {
    const r = Math.random();
    if (r < 0.32) {
      p.path = [L0, S];
      p.color = laserColor.clone().multiplyScalar(2.2);
      p.speed = 1.1;
    } else {
      const u = Math.random();
      let k = 0;
      while (k < N_RAYS - 1 && cdf[k] < u) k++;
      p.path = [S, C, SL, G, rayEnd[k]];
      p.color = raman.clone().multiplyScalar(0.6 + 1.2 * intensity[k]);
      p.speed = 0.38;
    }
    p.lens = [];
    p.total = 0;
    for (let i = 1; i < p.path.length; i++) {
      const l = p.path[i].distanceTo(p.path[i - 1]);
      p.lens.push(l);
      p.total += l;
    }
    p.s = 0;
  };
  const photonsState: Photon[] = Array.from({ length: N_PHOTONS }, () => {
    const p: Photon = { path: [L0, S], lens: [], total: 1, s: 0, speed: 1, color: raman };
    spawn(p);
    p.s = Math.random();
    return p;
  });
  const tmp = new THREE.Vector3();

  function setSpectrum(y: number[]): void {
    if (y.length === 0) return;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of y) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const span = hi - lo || 1;
    const at = (u: number) => {
      const x = u * (y.length - 1);
      const i = Math.min(y.length - 2, Math.floor(x));
      const f = x - i;
      return ((y[i] * (1 - f) + y[i + 1] * f) - lo) / span;
    };
    for (let p = 0; p < N_CCD; p++) {
      const v = at(p / (N_CCD - 1));
      const [r, g, b] = inferno(0.08 + 0.92 * Math.pow(v, 0.85));
      ccdData.set([r * 255, g * 255, b * 255, 255], p * 4);
    }
    ccdTex.needsUpdate = true;
    intensity = new Float32Array(N_RAYS).map((_, k) => at((k + 0.5) / N_RAYS));
    let acc = 0;
    const w = intensity.map((v) => 0.05 + v * v);
    const total = w.reduce((a, b) => a + b, 0);
    cdf = w.map((v) => (acc += v / total));
    for (let k = 0; k < N_RAYS; k++) {
      const v = 0.04 + 0.5 * intensity[k] ** 1.6;
      const c = raman.clone().multiplyScalar(v);
      fanCol.set([c.r * 1.4, c.g * 1.4, c.b * 1.4, c.r, c.g, c.b], k * 6);
    }
    fanColAttr.needsUpdate = true;
  }

  function setDrift(nm: number): void {
    const c = new THREE.Color().setHSL(0.985 - nm * 0.012, 1, 0.5);
    laserColor.copy(c);
    coreMat.color.copy(c).multiplyScalar(4);
    glowMat.color.copy(c).multiplyScalar(1.3);
    spotLight.color.copy(c);
    const el = laserLabel.element.querySelector(".sub");
    if (el) el.textContent = `λ ${(785 + nm).toFixed(2)} nm`;
  }

  function setSample(lbl: number): void {
    tissueMat.color.set(lbl ? 0xb85a78 : 0xd98a94);
  }

  function update(dt: number, t: number): void {
    for (let i = 0; i < N_PHOTONS; i++) {
      const p = photonsState[i];
      p.s += (dt * p.speed * 2.2) / p.total;
      if (p.s >= 1) spawn(p);
      const { lens, total } = p;
      let d = p.s * total;
      let j = 0;
      while (j < lens.length - 1 && d > lens[j]) {
        d -= lens[j];
        j++;
      }
      tmp.copy(p.path[j]).lerp(p.path[j + 1], Math.min(1, d / lens[j]));
      phPos.set([tmp.x, tmp.y, tmp.z], i * 3);
      const fade = Math.min(1, p.s * 12, (1 - p.s) * 12);
      phCol.set([p.color.r * fade, p.color.g * fade, p.color.b * fade], i * 3);
    }
    phGeo.getAttribute("position").needsUpdate = true;
    phGeo.getAttribute("color").needsUpdate = true;
    glowMat.opacity = 0.2 + 0.04 * Math.sin(t * 6);
  }

  return { group, setSpectrum, setDrift, setSample, update };
}
