"""Acquisition helpers on top of the twin: rendering, automatic exposure and checked cohorts.

Spectra are rendered exactly as `ramantwin.fleet.acquire_cohort` does, so that the API, the
robustness grid and the Twin Lab samples see the same data as the training set.
"""

from __future__ import annotations

import numpy as np

from ramantwin.constants import GRID

AUTO_FILL = 0.6  # target max count / full well, as in the ROWS device
CLIP_FRACTION = 0.98


def render(instrument, concentrations: np.ndarray) -> np.ndarray:
    """Noiseless pixel spectra (N_PIX,) or (n, N_PIX) of tissue concentrations (C,) or (n, C)."""
    from ramantwin.biochem import component_matrix

    return np.asarray(concentrations) @ instrument.line_shape(component_matrix)


def auto_exposure(
    params,
    concentrations: np.ndarray,
    rng: np.random.Generator,
    target: float = AUTO_FILL,
    bounds: tuple[float, float] = (0.05, 5.0),
    n_dry: int = 8,
):
    """Return a copy of `params` whose exposure brings the brightest pixel to `target` x full well.

    Dry acquisitions of a typical sample measure the count rate (counts above the bias scale
    linearly with exposure). Tissue fluorescence changes from one acquisition to the next, so the
    brightest of `n_dry` dry runs sets the exposure; if a dry run clips, the exposure is halved
    and the dry runs repeated.
    """
    from ramantwin.instrument import BIAS_E, InstrumentParams, VirtualInstrument

    p = InstrumentParams.from_dict(params.to_dict())
    inst = VirtualInstrument(p)
    pixels = render(inst, concentrations)
    for _ in range(6):
        p.n_accumulations = 1
        dry = [VirtualInstrument(p).acquire_pixels(pixels, rng) for _ in range(n_dry)]
        if max(a.accumulations.max() for a in dry) < 0.95 * p.saturation_e:
            break
        p.exposure_s = max(p.exposure_s / 2, bounds[0])
    peak = max(float((a.accumulations - a.dark).max()) for a in dry)
    rate = max(peak, 1.0) / p.exposure_s
    out = InstrumentParams.from_dict(params.to_dict())
    out.exposure_s = float(np.clip((target * p.saturation_e - BIAS_E) / rate, *bounds))
    return out


def acquire_checked(cohort, params, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Like `fleet.acquire_cohort` but returns (X, max_count_fraction per spectrum)."""
    from ramantwin.instrument import VirtualInstrument
    from ramantwin.pipeline import calibrate, process

    rng = np.random.default_rng(seed)
    inst = VirtualInstrument(params)
    calib = calibrate(inst, rng)
    pixels = render(inst, cohort.concentrations)
    X = np.empty((len(pixels), GRID.size), np.float32)
    fill = np.empty(len(pixels))
    for i, s in enumerate(pixels):
        proc = process(inst.acquire_pixels(s, rng), calib)
        X[i] = proc.final
        fill[i] = proc.max_count_fraction
    return X, fill
