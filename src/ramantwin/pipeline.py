"""ORPL processing in the order of the ROWS study.

dark subtraction -> crfilter_multi -> mean -> Tylenol x-axis -> NIST response correction
-> truncate < 450 cm-1 -> BubbleFill -> interpolation on GRID -> SNV.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from orpl import calibration
from orpl.baseline_removal import bubblefill
from orpl.cosmic_ray import crfilter_multi
from orpl.normalization import snv

import ramantwin  # noqa: F401  (applies the orpl patch)
from ramantwin.constants import (
    GRID,
    GRID_START,
    GRID_STOP,
    TRUNCATE_BELOW_CM1,
    TYLENOL_CAL_PEAKS_CM1,
)
from ramantwin.instrument import Acquisition, VirtualInstrument

# BubbleFill minimum width in pixels (~1.5 cm-1 / px). It must exceed the widest Raman feature,
# here the 1580-1680 cm-1 cluster of amide I, lipid C=C and heme bands.
MIN_BUBBLE_WIDTH = 100
TYLENOL_BUBBLE_WIDTH = 40


@dataclass
class Calibration:
    xaxis: np.ndarray
    irf: np.ndarray
    x_residual: float


@dataclass
class Processed:
    """ORPL outputs; pixel-space arrays are truncated to calibrated shifts >= 450 cm-1."""

    xaxis: np.ndarray
    raw_mean: np.ndarray
    corrected: np.ndarray
    baseline: np.ndarray
    raman: np.ndarray
    final: np.ndarray
    max_count_fraction: float


def clean_mean(acq: Acquisition) -> np.ndarray:
    """Dark subtraction, multi-accumulation cosmic ray filter and averaging."""
    frames = acq.accumulations - acq.dark
    if frames.shape[0] > 1:
        frames = crfilter_multi(frames)
    return frames.mean(axis=0)


def calibrate(instrument: VirtualInstrument, rng: np.random.Generator) -> Calibration:
    """Tylenol x-axis (autogenx) and NIST SRM 2241 response (compute_irf).

    The Tylenol spectrum is first flattened with a NIST response computed on the factory axis,
    so that autogenx sees band heights close to the reference ones even on a strongly tilted
    instrument.
    """
    tylenol = clean_mean(instrument.acquire_tylenol(rng))
    nist = clean_mean(instrument.acquire_nist(rng))

    rough_irf = calibration.compute_irf(nist, instrument.nominal_axis_cm1())
    peaks, _ = bubblefill(tylenol / rough_irf, min_bubble_widths=TYLENOL_BUBBLE_WIDTH)
    xaxis = calibration.autogenx(peaks, preset="tylenol", deg=2)
    residual = min(
        calibration.xaxis_from_peaks(peaks, combo)[1]
        for combo in itertools.combinations(TYLENOL_CAL_PEAKS_CM1, 5)
    )
    irf = calibration.compute_irf(nist, xaxis)
    return Calibration(xaxis=xaxis, irf=irf, x_residual=residual)


def axis_error_cm1(calib: Calibration, instrument: VirtualInstrument) -> float:
    """Twin-only check: RMS error of the calibrated axis over GRID, ignoring laser drift."""
    truth = instrument.pixel_shift_cm1(drift=False)
    in_grid = (truth >= GRID_START) & (truth <= GRID_STOP)
    return float(np.sqrt(np.mean((calib.xaxis - truth)[in_grid] ** 2)))


def process(
    acq: Acquisition, calib: Calibration, min_bubble_width: int = MIN_BUBBLE_WIDTH
) -> Processed:
    """Run the ORPL chain on one acquisition."""
    keep = calib.xaxis >= TRUNCATE_BELOW_CM1
    xaxis = calib.xaxis[keep]
    raw_mean = clean_mean(acq)[keep]
    corrected = raw_mean / calib.irf[keep]
    raman, baseline = bubblefill(corrected, min_bubble_widths=int(min_bubble_width))
    final = snv(np.interp(GRID, xaxis, raman))
    return Processed(
        xaxis=xaxis,
        raw_mean=raw_mean,
        corrected=corrected,
        baseline=baseline,
        raman=raman,
        final=final,
        max_count_fraction=float(acq.accumulations.max() / acq.saturation_e),
    )


def raw_on_grid(acq: Acquisition, instrument: VirtualInstrument) -> np.ndarray:
    """Dark-subtracted mean counts on the factory axis, on GRID, divided by the max."""
    mean = (acq.accumulations - acq.dark).mean(axis=0)
    y = np.interp(GRID, instrument.nominal_axis_cm1(), mean)
    return y / y.max()
