"""Compatibility fixes for orplib.

orpl.calibration.xaxis_from_peaks calls float() on the residual array returned by
np.polyfit(..., full=True). That array has shape (1,), or (0,) for an exact fit, and
recent NumPy releases refuse to convert it to a scalar, so autogenx raises TypeError.
An empty residual (exact or rank-deficient fit) carries no evidence of fit quality, so it is
reported as inf and can never win autogenx's best-combination search.
"""

import numpy as np
from orpl import calibration as _cal


def xaxis_from_peaks(signal: np.ndarray, peaks, deg: int = 2) -> tuple[np.ndarray, float]:
    expy = signal / signal.max()
    exp_pid = _cal.find_npeaks(expy, len(peaks))
    coefs, residual, _, _, _ = np.polyfit(exp_pid, peaks, deg=deg, full=True)
    residual = float(residual[0]) if residual.size else float("inf")
    xaxis = np.polynomial.polynomial.polyval(np.arange(expy.size), coefs[::-1])
    return xaxis, residual


def apply() -> None:
    """Patch orpl in place. Idempotent; called on `import ramantwin`."""
    _cal.xaxis_from_peaks = xaxis_from_peaks
