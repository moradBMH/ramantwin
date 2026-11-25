export interface Series {
  x: number[];
  y: number[];
  color: string;
  width?: number;
  dash?: number[];
  fill?: boolean;
}

export interface PlotOptions {
  xRange: [number, number];
  xTicks: number[];
  yLabel?: string;
  markers?: Array<{ x: number; label: string }>;
  zeroLine?: boolean;
}

/** Minimal canvas line plot with tweened updates. */
export class LinePlot {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly ro: ResizeObserver;
  private series: Series[] = [];
  private from: number[][] = [];
  private shown: number[][] = [];
  private range: [number, number] = [0, 1];
  private rangeFrom: [number, number] = [0, 1];
  private rangeTo: [number, number] = [0, 1];
  private t0 = 0;
  private raf = 0;
  private w = 0;
  private h = 0;

  constructor(
    container: HTMLElement,
    private readonly opts: PlotOptions,
  ) {
    this.canvas = document.createElement("canvas");
    container.appendChild(this.canvas);
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("2d context");
    this.ctx = ctx;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
  }

  set(series: Series[], animate = true): void {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series) {
      for (let i = 0; i < s.y.length; i++) {
        const x = s.x[i];
        if (x < this.opts.xRange[0] || x > this.opts.xRange[1]) continue;
        lo = Math.min(lo, s.y[i]);
        hi = Math.max(hi, s.y[i]);
      }
    }
    if (!Number.isFinite(lo)) [lo, hi] = [0, 1];
    const pad = (hi - lo || 1) * 0.08;
    this.rangeTo = [lo - pad, hi + pad];

    const sameShape =
      animate &&
      series.length === this.series.length &&
      series.every((s, i) => s.y.length === this.shown[i]?.length);
    this.from = sameShape ? this.shown.map((a) => a.slice()) : series.map((s) => s.y.slice());
    this.rangeFrom = sameShape ? [...this.range] : [...this.rangeTo];
    this.series = series;
    this.shown = series.map((s) => s.y.slice());
    this.t0 = performance.now();
    cancelAnimationFrame(this.raf);
    this.tick();
  }

  private tick = (): void => {
    const k = Math.min(1, (performance.now() - this.t0) / 320);
    const e = 1 - Math.pow(1 - k, 3);
    this.series.forEach((s, i) => {
      const a = this.from[i];
      const out = this.shown[i];
      for (let j = 0; j < out.length; j++) out[j] = a[j] + (s.y[j] - a[j]) * e;
    });
    this.range = [
      this.rangeFrom[0] + (this.rangeTo[0] - this.rangeFrom[0]) * e,
      this.rangeFrom[1] + (this.rangeTo[1] - this.rangeFrom[1]) * e,
    ];
    this.draw();
    if (k < 1) this.raf = requestAnimationFrame(this.tick);
  };

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = this.canvas.parentElement!.getBoundingClientRect();
    this.w = rect.width;
    this.h = rect.height;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  private draw(): void {
    const { ctx, w, h, opts } = this;
    const L = 8;
    const R = 8;
    const T = opts.markers ? 16 : 6;
    const B = 22;
    ctx.clearRect(0, 0, w, h);
    const [x0, x1] = opts.xRange;
    const [y0, y1] = this.range;
    const px = (x: number) => L + ((x - x0) / (x1 - x0)) * (w - L - R);
    const py = (y: number) => T + (1 - (y - y0) / (y1 - y0 || 1)) * (h - T - B);

    ctx.font = "11px ui-monospace, SF Mono, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of opts.xTicks) {
      const x = px(t);
      ctx.strokeStyle = "rgba(170,190,230,0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, T);
      ctx.lineTo(x, h - B);
      ctx.stroke();
      ctx.fillStyle = "#5a6578";
      const half = ctx.measureText(String(t)).width / 2;
      ctx.fillText(String(t), Math.min(w - half - 1, Math.max(half + 1, x)), h - B + 6);
    }
    ctx.strokeStyle = "rgba(170,190,230,0.22)";
    ctx.beginPath();
    ctx.moveTo(L, h - B + 0.5);
    ctx.lineTo(w - R, h - B + 0.5);
    ctx.stroke();

    if (opts.zeroLine && y0 < 0 && y1 > 0) {
      ctx.strokeStyle = "rgba(170,190,230,0.15)";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(L, py(0));
      ctx.lineTo(w - R, py(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (opts.markers) {
      ctx.textBaseline = "alphabetic";
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      for (const m of opts.markers) {
        const x = px(m.x);
        ctx.strokeStyle = "rgba(185,165,255,0.22)";
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(x, T);
        ctx.lineTo(x, h - B);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(185,165,255,0.75)";
        ctx.fillText(m.label, x, T - 4);
      }
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(L, T - 2, w - L - R, h - T - B + 2);
    ctx.clip();
    this.series.forEach((s, i) => {
      const ys = this.shown[i];
      ctx.beginPath();
      let started = false;
      for (let j = 0; j < ys.length; j++) {
        const x = s.x[j];
        if (x < x0 || x > x1) continue;
        if (!started) {
          ctx.moveTo(px(x), py(ys[j]));
          started = true;
        } else ctx.lineTo(px(x), py(ys[j]));
      }
      if (s.fill && started) {
        const grad = ctx.createLinearGradient(0, T, 0, h - B);
        grad.addColorStop(0, hexAlpha(s.color, 0.22));
        grad.addColorStop(1, hexAlpha(s.color, 0));
        ctx.save();
        ctx.lineTo(px(Math.min(x1, s.x[s.x.length - 1])), h - B);
        ctx.lineTo(px(Math.max(x0, s.x[0])), h - B);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();
        ctx.beginPath();
        started = false;
        for (let j = 0; j < ys.length; j++) {
          const x = s.x[j];
          if (x < x0 || x > x1) continue;
          if (!started) {
            ctx.moveTo(px(x), py(ys[j]));
            started = true;
          } else ctx.lineTo(px(x), py(ys[j]));
        }
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width ?? 1.5;
      ctx.lineJoin = "round";
      ctx.setLineDash(s.dash ?? []);
      ctx.stroke();
      ctx.setLineDash([]);
    });
    ctx.restore();
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.canvas.remove();
  }
}

function hexAlpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
