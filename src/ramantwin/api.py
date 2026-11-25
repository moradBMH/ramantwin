"""FastAPI backend of the Twin Lab: simulate one acquisition, process it with ORPL, classify it.

Run with `ramantwin-demo` (or `uv run ramantwin-demo --artifacts artifacts`).
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ramantwin.constants import CLASSES, GRID, GRID_START, GRID_STEP, GRID_STOP
from ramantwin.ml.models import ModelBundle
from ramantwin.ml.qc import assess, physics_checks

MODELS = ("svm", "naive", "robust")
MAX_POINTS = 600
# Fields that describe the sample or one measurement rather than the hardware. Changing them does
# not trigger a new Tylenol/NIST calibration; laser drift in particular happens after calibration.
MEASUREMENT_FIELDS = frozenset(
    {
        "name",
        "laser_drift_nm",
        "fluorescence_ratio",
        "fluorescence_shape",
        "signal_e_per_s",
        "exposure_s",
        "n_accumulations",
        "cosmic_rate",
        "ambient_light",
    }
)


def curve(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS) -> dict[str, list[float]]:
    """Uniformly decimate a curve to at most `max_points` points, rounded for JSON."""
    step = math.ceil(len(x) / max_points)
    return {
        "x": np.round(x[::step], 2).tolist(),
        "y": np.round(y[::step], 4).tolist(),
    }


class Simulator:
    """Twin + ORPL + models, with calibrations cached per hardware configuration."""

    def __init__(self, models: ModelBundle, cache_size: int = 64):
        self.models = models
        self.cache_size = cache_size
        self._calibrations: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()

    def calibration(self, params) -> tuple[object, bool]:
        """Return (Calibration, cache hit) for the hardware part of `params`."""
        from ramantwin.instrument import VirtualInstrument
        from ramantwin.pipeline import calibrate

        key = tuple(
            sorted((k, v) for k, v in params.to_dict().items() if k not in MEASUREMENT_FIELDS)
        )
        if key in self._calibrations:
            self._calibrations.move_to_end(key)
            return self._calibrations[key], True
        calib = calibrate(VirtualInstrument(params), np.random.default_rng(0))
        self._calibrations[key] = calib
        if len(self._calibrations) > self.cache_size:
            self._calibrations.popitem(last=False)
        return calib, False

    def simulate(self, params: dict, label: int, seed: int) -> dict:
        """Measure one simulated tissue spectrum of class `label` on the instrument `params`."""
        from ramantwin.biochem import raman_spectrum, sample_cohort
        from ramantwin.instrument import InstrumentParams, VirtualInstrument, nominal
        from ramantwin.pipeline import process

        with self._lock:
            p = InstrumentParams.from_dict({**nominal().to_dict(), **params})
            cohort = sample_cohort(1, 1, np.random.default_rng(seed), cancer_fraction=float(label))
            conc = cohort.concentrations[0]

            def spectrum_fn(x: np.ndarray) -> np.ndarray:
                s = raman_spectrum(conc, x)
                return s / s.max()

            t0 = time.perf_counter()
            calib, _ = self.calibration(p)
            t1 = time.perf_counter()
            acq = VirtualInstrument(p).acquire(spectrum_fn, np.random.default_rng([seed, 1]))
            t2 = time.perf_counter()
            proc = process(acq, calib)
            t3 = time.perf_counter()
            out = self.models.outputs(proc.final[None, :].astype(np.float32))
            qc = assess(
                self.models.qc, out["emb_robust"][0], physics_checks(proc, p.resolution_cm1)
            )
            t4 = time.perf_counter()

        scale = float(np.max(proc.corrected))
        return {
            "raw": curve(proc.xaxis, proc.corrected / scale),
            "baseline": curve(proc.xaxis, proc.baseline / scale),
            "processed": {
                "x": GRID[::2].tolist(),
                "y": np.round(proc.final[::2], 4).tolist(),
            },
            "truth": {
                "label": int(cohort.labels[0]),
                "concentrations": {
                    name: round(float(c), 4)
                    for name, c in zip(cohort.component_names, conc, strict=True)
                },
            },
            "predictions": {m: {"p_cancer": round(float(out[m][0]), 4)} for m in MODELS},
            "qc": qc.to_dict(),
            "timing_ms": {
                "calibration": round(1e3 * (t1 - t0), 2),
                "acquire": round(1e3 * (t2 - t1), 2),
                "orpl": round(1e3 * (t3 - t2), 2),
                "inference": round(1e3 * (t4 - t3), 2),
            },
        }


class SimulateRequest(BaseModel):
    params: dict[str, float | int | str] = Field(default_factory=dict)
    label: Literal[0, 1] = 1
    seed: int = 0


def create_app(
    artifacts: Path | None = None, web_dist: Path | None = None, warmup: bool = True
) -> FastAPI:
    """Build the app; artifacts are loaded (and the pipeline warmed up) at startup."""
    artifacts = artifacts or Path(os.environ.get("RAMANTWIN_ARTIFACTS", "artifacts"))
    web_dist = web_dist or Path(os.environ.get("RAMANTWIN_WEB_DIST", "web/dist"))
    state: dict[str, Simulator] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["sim"] = Simulator(ModelBundle.load(artifacts))
        if warmup:
            state["sim"].simulate({}, 1, 0)
        yield

    app = FastAPI(title="RamanTwin", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict:
        from ramantwin.instrument import SLIDER_RANGES, nominal

        return {
            "params": nominal().to_dict(),
            "ranges": {k: list(v) for k, v in SLIDER_RANGES.items()},
            "grid": {"start": GRID_START, "stop": GRID_STOP, "step": GRID_STEP},
            "classes": list(CLASSES),
            "models": list(MODELS),
        }

    @app.post("/api/simulate")
    def simulate(req: SimulateRequest) -> dict:
        from ramantwin.instrument import nominal

        unknown = set(req.params) - set(nominal().to_dict())
        if unknown:
            raise HTTPException(422, f"unknown instrument parameters: {sorted(unknown)}")
        return state["sim"].simulate(req.params, req.label, req.seed)

    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="RamanTwin demo server")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.artifacts), host=args.host, port=args.port)
