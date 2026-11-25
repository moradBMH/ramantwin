import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

export interface Bloom {
  strength: number;
  radius: number;
  threshold: number;
}

/** A full-screen 3D scene with its own overlay UI. */
export interface View {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  bloom: Bloom;
  update(dt: number, t: number): void;
  resize?(w: number, h: number): void;
  next(): boolean;
  prev(): boolean;
  onKey?(e: KeyboardEvent): boolean;
  dispose(): void;
}

export class Engine {
  readonly renderer: THREE.WebGLRenderer;
  readonly labels: CSS2DRenderer;
  private readonly composer: EffectComposer;
  private readonly renderPass: RenderPass;
  private readonly bloomPass: UnrealBloomPass;
  private view: View | null = null;
  private last = performance.now();
  private readonly start = performance.now();
  width = window.innerWidth;
  height = window.innerHeight;

  constructor(canvas: HTMLCanvasElement, labelRoot: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x05070b, 1);
    this.renderer.toneMapping = THREE.NoToneMapping;

    this.labels = new CSS2DRenderer({ element: labelRoot });

    const target = new THREE.WebGLRenderTarget(this.width, this.height, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(this.renderer, target);
    this.renderPass = new RenderPass(new THREE.Scene(), new THREE.PerspectiveCamera());
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(this.width, this.height), 0.8, 0.5, 0.2);
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    this.resize();
    window.addEventListener("resize", () => this.resize());
    this.renderer.setAnimationLoop(() => this.frame());
  }

  get pixelRatio(): number {
    return this.renderer.getPixelRatio();
  }

  setView(view: View | null): void {
    if (this.view) {
      this.view.dispose();
      this.labels.domElement.replaceChildren();
    }
    this.view = view;
    if (view) {
      this.renderPass.scene = view.scene;
      this.renderPass.camera = view.camera;
      this.applyCamera();
      view.resize?.(this.width, this.height);
    }
  }

  private applyCamera(): void {
    if (!this.view) return;
    this.view.camera.aspect = this.width / this.height;
    this.view.camera.updateProjectionMatrix();
  }

  private resize(): void {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.width, this.height);
    this.composer.setPixelRatio(dpr);
    this.composer.setSize(this.width, this.height);
    this.labels.setSize(this.width, this.height);
    this.applyCamera();
    this.view?.resize?.(this.width, this.height);
  }

  private frame(): void {
    const now = performance.now();
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const view = this.view;
    if (!view) return;
    view.update(dt, (now - this.start) / 1000);
    this.bloomPass.strength = view.bloom.strength;
    this.bloomPass.radius = view.bloom.radius;
    this.bloomPass.threshold = view.bloom.threshold;
    this.composer.render(dt);
    this.labels.render(view.scene, view.camera);
  }
}

/** Shifts the projection centre so the scene is centred in the free area between panels. */
export function frameFreeArea(
  camera: THREE.PerspectiveCamera,
  w: number,
  h: number,
  free: { left: number; right: number; top: number; bottom: number },
  fill = 1,
): void {
  const cx = (free.left + (w - free.right)) / 2;
  const cy = (free.top + (h - free.bottom)) / 2;
  camera.setViewOffset(w, h, w / 2 - cx, h / 2 - cy, w, h);
  const fw = (w - free.left - free.right) / w;
  const fh = (h - free.top - free.bottom) / h;
  camera.zoom = Math.max(0.35, Math.min(fw, fh * 1.25) * fill);
  camera.updateProjectionMatrix();
}

export function label(html: string, cls = "label3d"): CSS2DObject {
  const el = document.createElement("div");
  el.className = cls;
  el.innerHTML = html;
  return new CSS2DObject(el);
}

export function disposeObject(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    mesh.geometry?.dispose();
    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
    const mats = Array.isArray(mat) ? mat : mat ? [mat] : [];
    for (const m of mats) {
      for (const value of Object.values(m)) {
        if (value instanceof THREE.Texture) value.dispose();
      }
      if (m instanceof THREE.ShaderMaterial) {
        for (const u of Object.values(m.uniforms)) {
          if (u.value instanceof THREE.Texture) u.value.dispose();
        }
      }
      m.dispose();
    }
  });
}

export const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
export const damp = (a: number, b: number, lambda: number, dt: number) => a + (b - a) * (1 - Math.exp(-lambda * dt));
