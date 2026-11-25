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
from typing import Annotated, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


def sample_tissue(label: int, seed: int):
    """One simulated tissue sample of class `label`: (Cohort of one spectrum, concentrations)."""
    from ramantwin.biochem import sample_cohort

    cohort = sample_cohort(1, 1, np.random.default_rng(seed), cancer_fraction=float(label))
    return cohort, cohort.concentrations[0]


def clip_params(params: dict[str, float]) -> dict[str, float]:
    """Clip slider parameters to their physical range; unknown keys raise ValueError."""
    from ramantwin.instrument import SLIDER_RANGES

    unknown = sorted(set(params) - set(SLIDER_RANGES))
    if unknown:
        raise ValueError(f"unknown instrument parameters: {unknown}")
    out = {k: float(np.clip(v, *SLIDER_RANGES[k])) for k, v in params.items()}
    if "n_accumulations" in out:
        out["n_accumulations"] = int(round(out["n_accumulations"]))
    return out


class Simulator:
    """Twin + ORPL + models, with calibrations cached per hardware configuration."""

    def __init__(self, models: ModelBundle, cache_size: int = 64):
        self.models = models
        self.cache_size = cache_size
        self._calibrations: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()

    def calibration(self, params) -> tuple[object, bool]:
        """Return (Calibration, cache hit) for the hardware part of `params`.

        The Tylenol and NIST standards are measured with the measurement fields at their nominal
        values, so the calibration only depends on the hardware and the cache key is exact.
        """
        from ramantwin.instrument import InstrumentParams, VirtualInstrument, nominal
        from ramantwin.pipeline import calibrate

        base = nominal().to_dict()
        hw = {k: base[k] if k in MEASUREMENT_FIELDS else v for k, v in params.to_dict().items()}
        key = tuple(sorted(hw.items()))
        if key in self._calibrations:
            self._calibrations.move_to_end(key)
            return self._calibrations[key], True
        inst = VirtualInstrument(InstrumentParams.from_dict(hw))
        calib = calibrate(inst, np.random.default_rng(0))
        self._calibrations[key] = calib
        if len(self._calibrations) > self.cache_size:
            self._calibrations.popitem(last=False)
        return calib, False

    def simulate(self, params: dict, label: int, seed: int) -> dict:
        """Measure one simulated tissue spectrum of class `label` on the instrument `params`.

        The spectrum is rendered exactly as in `fleet.acquire_cohort`.
        """
        from ramantwin.instrument import InstrumentParams, VirtualInstrument, nominal
        from ramantwin.ml.acquire import render
        from ramantwin.pipeline import process

        with self._lock:
            p = InstrumentParams.from_dict({**nominal().to_dict(), **params})
            cohort, conc = sample_tissue(label, seed)
            t0 = time.perf_counter()
            calib, _ = self.calibration(p)
            t1 = time.perf_counter()
            inst = VirtualInstrument(p)
            acq = inst.acquire_pixels(render(inst, conc), np.random.default_rng([seed, 1]))
            t2 = time.perf_counter()
            proc = process(acq, calib)
            t3 = time.perf_counter()
            out = self.models.outputs(proc.final[None, :].astype(np.float32))
            qc = assess(
                self.models.qc,
                out["emb_robust"][0],
                physics_checks(proc, p.resolution_cm1, laser_nm=p.laser_nm),
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


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class SimulateRequest(BaseModel):
    params: dict[str, FiniteFloat] = Field(default_factory=dict)
    label: Literal[0, 1] = 1
    seed: int = Field(0, ge=0, lt=2**32)


def create_app(
    artifacts: Path | None = None, web_dist: Path | None = None, warmup: bool = True
) -> FastAPI:
    """Build the app; artifacts are loaded (and the pipeline warmed up) at startup."""
    artifacts = artifacts or Path(os.environ.get("RAMANTWIN_ARTIFACTS", "artifacts"))
    web_dist = web_dist or Path(os.environ.get("RAMANTWIN_WEB_DIST", "web/dist"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # single-spectrum inference is fastest on a few CPU threads
        torch.set_num_threads(2)
        app.state.simulator = Simulator(ModelBundle.load(artifacts))
        if warmup:
            for seed in range(3):
                app.state.simulator.simulate({}, 1, seed)
        yield

    app = FastAPI(title="RamanTwin", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # the offending input is left out: NaN or inf cannot be serialised back to JSON
        detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

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
        try:
            params = clip_params(req.params)
        except ValueError as err:
            raise HTTPException(422, str(err)) from err
        return app.state.simulator.simulate(params, req.label, req.seed)

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
