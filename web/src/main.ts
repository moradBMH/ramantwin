import "./styles.css";
import { Engine, type View } from "./gl/engine";
import { onMockChange, usingMock } from "./data";
import { h } from "./ui/dom";
import { createLanding } from "./scenes/landing";
import { createShift } from "./scenes/shift";

export interface ViewContext {
  engine: Engine;
  ui: HTMLElement;
  go(key: string): void;
}

interface Route {
  key: string;
  label: string;
  create(ctx: ViewContext): Promise<View>;
}

const ROUTES: Route[] = [
  { key: "intro", label: "Intro", create: createLanding },
  { key: "shift", label: "The Shift", create: createShift },
];

const canvas = document.getElementById("gl") as HTMLCanvasElement;
const labelRoot = document.getElementById("labels") as HTMLElement;
const ui = document.getElementById("ui") as HTMLElement;
const tabs = document.getElementById("tabs") as HTMLElement;
const status = document.getElementById("status") as HTMLElement;
const engine = new Engine(canvas, labelRoot);

let current = -1;
let view: View | null = null;
let navToken = 0;

const tabButtons = ROUTES.map((r, i) => {
  const b = h("button", { class: "tab" }, `<kbd>${i + 1}</kbd>${r.label}`);
  b.addEventListener("click", () => go(r.key));
  tabs.appendChild(b);
  return b;
});
document.getElementById("brand")!.addEventListener("click", () => go("intro"));

function renderStatus(): void {
  status.innerHTML = '<span class="chip"><span class="dot"></span>Simulated data</span>';
  if (usingMock()) status.insertAdjacentHTML("afterbegin", '<span class="chip warn" title="Files in public/data carry mock: true">Mock files</span>');
}
renderStatus();
onMockChange(renderStatus);

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

async function go(key: string): Promise<void> {
  const index = Math.max(0, ROUTES.findIndex((r) => r.key === key));
  if (index === current) return;
  const token = ++navToken;
  current = index;
  tabButtons.forEach((b, i) => b.classList.toggle("active", i === index));
  if (location.hash.slice(1) !== ROUTES[index].key) history.replaceState(null, "", `#${ROUTES[index].key}`);

  if (view) {
    document.body.classList.add("switching");
    await sleep(270);
    if (token !== navToken) return;
  }
  engine.setView(null);
  view = null;
  ui.replaceChildren();
  const ctx: ViewContext = { engine, ui, go: (k) => void go(k) };
  try {
    const next = await ROUTES[index].create(ctx);
    if (token !== navToken) {
      next.dispose();
      return;
    }
    view = next;
    engine.setView(next);
  } catch (err) {
    console.error(err);
    ui.replaceChildren(h("div", { class: "panel", style: "left:24px;top:90px" }, `Could not load this view: ${String(err)}`));
  }
  requestAnimationFrame(() => document.body.classList.remove("switching"));
}

function step(dir: 1 | -1): void {
  const handled = view ? (dir === 1 ? view.next() : view.prev()) : false;
  if (handled) return;
  const target = current + dir;
  if (target >= 0 && target < ROUTES.length) void go(ROUTES[target].key);
}

window.addEventListener("keydown", (e) => {
  const target = e.target as HTMLElement;
  if (target instanceof HTMLInputElement && target.type === "range" && e.key.startsWith("Arrow")) return;
  if (target instanceof HTMLInputElement && target.type !== "range") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const n = Number(e.key);
  if (Number.isInteger(n) && n >= 1 && n <= ROUTES.length) {
    void go(ROUTES[n - 1].key);
  } else if (e.key === " " || e.key === "ArrowRight" || e.key === "PageDown") {
    e.preventDefault();
    step(e.shiftKey && e.key === " " ? -1 : 1);
  } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
    e.preventDefault();
    step(-1);
  } else if (e.key === "f") {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void document.documentElement.requestFullscreen();
  } else if (view?.onKey?.(e)) {
    e.preventDefault();
  }
});

// Keep keyboard focus off buttons so Space always means "next step".
document.addEventListener("pointerup", () => {
  const el = document.activeElement;
  if (el instanceof HTMLButtonElement) el.blur();
});

window.addEventListener("hashchange", () => void go(location.hash.slice(1)));
void go(location.hash.slice(1) || "intro");
