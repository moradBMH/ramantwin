import numpy as np
from orpl import calibration

import ramantwin  # noqa: F401  (applies the orpl patch)

BANDS = [
    (797, 0.8, 8),
    (858, 0.9, 8),
    (1169, 1.0, 8),
    (1237, 0.7, 9),
    (1324, 0.75, 9),
    (1609, 0.95, 9),
    (1648, 0.6, 9),
    (651, 0.3, 7),
    (1561, 0.3, 8),
    (465, 0.25, 8),
]


def _acetaminophen(xaxis):
    return sum(h / (1 + ((xaxis - c) / (w / 2)) ** 2) for c, h, w in BANDS)


def test_autogenx_recovers_true_axis():
    rng = np.random.default_rng(0)
    pix = np.arange(1024)
    true_x = 400 + 1.5 * pix + 6e-5 * pix**2 + 7.0
    signal = rng.poisson(_acetaminophen(true_x) * 2000 + 300).astype(float)

    estimated = calibration.autogenx(signal, preset="tylenol", deg=2)

    in_range = (true_x > 600) & (true_x < 1800)
    assert np.abs(estimated - true_x)[in_range].max() < 3.0
