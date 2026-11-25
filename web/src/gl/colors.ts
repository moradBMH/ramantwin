import * as THREE from "three";

export const CSS = {
  bg: "#05070b",
  normal: "#4ea8f5",
  cancer: "#f5a03a",
  accent: "#aeb7c6",
  laser: "#ff2a3a",
  text: "#e6e9ee",
  muted: "#8f9aad",
  dim: "#5a6578",
  ok: "#4cc38a",
  warn: "#e8b84a",
  reject: "#e5606b",
  naive: "#8f9aad",
  robust: "#68c3a3",
};

// Categorical palette for instruments, tuned for a dark background.
export const INSTRUMENT_PALETTE = [
  "#ff6b9d",
  "#ffd166",
  "#06d6a0",
  "#4cc9f0",
  "#f78c6b",
  "#9b8cff",
  "#83e377",
  "#f15bb5",
  "#00bbf9",
  "#fee440",
  "#c77dff",
  "#90e0ef",
];

type Stops = Array<[number, number, number]>;

const VIRIDIS: Stops = [
  [0.267, 0.005, 0.329],
  [0.283, 0.141, 0.458],
  [0.254, 0.265, 0.53],
  [0.207, 0.372, 0.553],
  [0.164, 0.471, 0.558],
  [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518],
  [0.267, 0.749, 0.441],
  [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.15],
  [0.993, 0.906, 0.144],
];

const INFERNO: Stops = [
  [0.001, 0.0, 0.014],
  [0.087, 0.045, 0.224],
  [0.258, 0.039, 0.406],
  [0.416, 0.09, 0.433],
  [0.578, 0.148, 0.404],
  [0.735, 0.216, 0.33],
  [0.865, 0.317, 0.226],
  [0.954, 0.469, 0.099],
  [0.988, 0.645, 0.04],
  [0.964, 0.832, 0.251],
  [0.988, 0.998, 0.645],
];

function sample(stops: Stops, t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export const viridis = (t: number) => sample(VIRIDIS, t);
export const inferno = (t: number) => sample(INFERNO, t);

export function cssGradient(fn: (t: number) => [number, number, number], from = 0, to = 1): string {
  const parts: string[] = [];
  for (let i = 0; i <= 10; i++) {
    const [r, g, b] = fn(from + ((to - from) * i) / 10);
    parts.push(`rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${i * 10}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

const tmp = new THREE.Color();

export function linear(rgb: [number, number, number]): THREE.Color {
  return new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
}

export function linearHex(hex: string): [number, number, number] {
  tmp.set(hex);
  return [tmp.r, tmp.g, tmp.b];
}
