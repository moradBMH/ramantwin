// Tissue bands used for decorative spectra (cm-1, FWHM, height, component).
const BANDS: Array<[number, number, number, number]> = [
  [853, 16, 0.22, 0],
  [936, 22, 0.18, 0],
  [1003, 8, 0.55, 0],
  [1085, 18, 0.18, 2],
  [1127, 15, 0.16, 1],
  [1156, 10, 0.3, 3],
  [1250, 36, 0.32, 0],
  [1300, 24, 0.42, 1],
  [1340, 24, 0.24, 2],
  [1445, 26, 0.8, 1],
  [1520, 12, 0.36, 3],
  [1578, 14, 0.12, 2],
  [1655, 30, 0.72, 0],
  [1745, 18, 0.16, 1],
];

/** Weights for protein, lipid, nucleic acid, carotenoid. */
export type Mix = [number, number, number, number];

export function tissueMix(cancer: number, rnd: () => number): Mix {
  const j = () => 1 + 0.12 * (rnd() * 2 - 1);
  return [(1 + 0.18 * cancer) * j(), (1 - 0.22 * cancer) * j(), (1 + 0.45 * cancer) * j(), (1 - 0.55 * cancer) * j()];
}

export function tissueSpectrum(shift: number, mix: Mix, offset = 0): number {
  let s = 0;
  for (const [c, fwhm, h, k] of BANDS) {
    const w = fwhm / 2;
    const d = shift - c - offset;
    s += (h * mix[k] * w * w) / (d * d + w * w);
  }
  return s;
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
