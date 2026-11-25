"""Synthetic spectra without the twin: Lorentzian bands, class-dependent band ratios, shifts.

Used by the tests and experiments of the ml package, which must run before the twin exists.
"""

from dataclasses import dataclass

import numpy as np

from ramantwin.constants import GRID

# center cm-1, FWHM cm-1, height, cancer multiplier
BANDS = (
    (785.0, 14.0, 0.4, 1.5),
    (1004.0, 8.0, 1.0, 1.0),
    (1156.0, 10.0, 0.6, 0.55),
    (1250.0, 30.0, 0.5, 1.1),
    (1300.0, 25.0, 0.6, 0.8),
    (1440.0, 25.0, 0.9, 0.8),
    (1520.0, 12.0, 0.5, 0.55),
    (1655.0, 30.0, 0.8, 1.15),
)


@dataclass
class Synthetic:
    X: np.ndarray
    y: np.ndarray
    patient: np.ndarray
    instrument: np.ndarray


def lorentz(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    return 1.0 / (1.0 + ((x - center) / (fwhm / 2)) ** 2)


def snv(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)


def make_spectra(
    n_patients: int = 40,
    spectra_per_patient: int = 4,
    n_instruments: int = 2,
    seed: int = 0,
    shifts_cm1: tuple[float, ...] | None = None,
    noise: float = 0.03,
) -> Synthetic:
    """Each instrument measures its own patients; an instrument shifts every band by its offset."""
    rng = np.random.default_rng(seed)
    shifts = shifts_cm1 if shifts_cm1 is not None else (0.0,) * n_instruments
    X, y, patient, instrument = [], [], [], []
    pid = 0
    for k in range(n_instruments):
        for i in range(n_patients):
            label = i % 2
            mult = np.array([b[3] if label else 1.0 for b in BANDS])
            w_patient = np.array([b[2] for b in BANDS]) * mult * np.exp(0.25 * rng.normal(size=8))
            for _ in range(spectra_per_patient):
                w = w_patient * np.exp(0.08 * rng.normal(size=8))
                s = sum(
                    wj * lorentz(GRID, c + shifts[k], f)
                    for wj, (c, f, _, _) in zip(w, BANDS, strict=True)
                )
                X.append(s + noise * rng.normal(size=GRID.size))
                y.append(label)
                patient.append(pid)
                instrument.append(k)
            pid += 1
    return Synthetic(
        snv(np.array(X)).astype(np.float32),
        np.array(y, np.int64),
        np.array(patient, np.int32),
        np.array(instrument, np.int16),
    )
