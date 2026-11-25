"""Acquisition helpers on top of the twin: rendering, automatic exposure and checked cohorts.

Spectra are rendered exactly as `ramantwin.fleet.acquire_cohort` does. The training and fleet
datasets use the fixed exposure of each instrument (1 s), whereas the robustness grid and the Twin
Lab scenarios use automatic exposure; on the same cohort the two settings change balanced accuracy
by less than about 0.05, in no systematic direction.
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


def exposure_for(
    params,
    pixels: np.ndarray,
    rng: np.random.Generator,
    target: float = AUTO_FILL,
    bounds: tuple[float, float] = (0.05, 5.0),
    n_dry: int = 8,
) -> float:
    """Exposure (s) that brings the brightest pixel of a sample to `target` x full well.

    Short dry acquisitions of the sample (rendered `pixels`) measure the count rate; counts above
    the bias scale linearly with exposure. The twin draws a new tissue fluorescence at every
    acquisition, so the brightest of `n_dry` dry runs sets the exposure. If a dry run clips, the
    exposure is halved and the dry runs repeated.
    """
    from ramantwin.instrument import BIAS_E, InstrumentParams, VirtualInstrument

    p = InstrumentParams.from_dict({**params.to_dict(), "n_accumulations": 1})
    for _ in range(6):
        inst = VirtualInstrument(p)
        dry = [inst.acquire_pixels(pixels, rng) for _ in range(n_dry)]
        if max(a.accumulations.max() for a in dry) < 0.95 * p.saturation_e:
            break
        p.exposure_s = max(p.exposure_s / 2, bounds[0])
    peak = max(float((a.accumulations - a.dark).max()) for a in dry)
    rate = max(peak, 1.0) / p.exposure_s
    return float(np.clip((target * p.saturation_e - BIAS_E) / rate, *bounds))


def auto_exposure(params, concentrations: np.ndarray, rng: np.random.Generator, **kwargs):
    """Copy of `params` auto-exposed for one sample of tissue `concentrations` (C,)."""
    from ramantwin.instrument import InstrumentParams, VirtualInstrument

    pixels = render(VirtualInstrument(params), concentrations)
    exposure = exposure_for(params, pixels, rng, **kwargs)
    return InstrumentParams.from_dict({**params.to_dict(), "exposure_s": exposure})


def acquire_checked(
    cohort, params, seed: int, auto: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like `fleet.acquire_cohort`, with optional per-acquisition automatic exposure.

    Returns (X, max_count_fraction, exposure_s), one value per spectrum.
    """
    from ramantwin.instrument import InstrumentParams, VirtualInstrument
    from ramantwin.pipeline import calibrate, process

    rng = np.random.default_rng(seed)
    inst = VirtualInstrument(params)
    calib = calibrate(inst, rng)
    pixels = render(inst, cohort.concentrations)
    X = np.empty((len(pixels), GRID.size), np.float32)
    fill = np.empty(len(pixels))
    exposure = np.full(len(pixels), params.exposure_s)
    for i, s in enumerate(pixels):
        if auto:
            exposure[i] = exposure_for(params, s, rng)
            p = InstrumentParams.from_dict({**params.to_dict(), "exposure_s": exposure[i]})
            proc = process(VirtualInstrument(p).acquire_pixels(s, rng), calib)
        else:
            proc = process(inst.acquire_pixels(s, rng), calib)
        X[i] = proc.final
        fill[i] = proc.max_count_fraction
    return X, fill, exposure
