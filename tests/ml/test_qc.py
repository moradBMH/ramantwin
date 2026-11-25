from types import SimpleNamespace

import numpy as np

from ramantwin.ml.models import cnn_outputs
from ramantwin.ml.qc import ROOM_LIGHT_NM, MahalanobisOOD, abstain, assess, physics_checks
from ramantwin.ml.synthetic import lorentz, make_spectra


def test_mahalanobis_thresholds_and_far_points():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(2000, 8))
    y = rng.integers(0, 2, 2000)
    emb[y == 1] += 3.0
    ood = MahalanobisOOD.fit(emb, y).calibrate(rng.normal(size=(2000, 8)))
    assert ood.warn < ood.reject
    far = rng.normal(size=(50, 8)) + 15.0
    assert abstain(ood.score(far), ood).all()
    assert assess(ood, far[0]).flag == "reject"
    assert assess(ood, emb[0] * 0).flag == "ok"


def test_ood_flags_far_shifted_spectra(tiny_bundle, synthetic):
    shifted = make_spectra(n_patients=10, n_instruments=1, seed=5, shifts_cm1=(80.0,))
    _, emb_in = cnn_outputs(tiny_bundle.robust, synthetic.X)
    _, emb_far = cnn_outputs(tiny_bundle.robust, shifted.X)
    s_in, s_far = tiny_bundle.qc.score(emb_in), tiny_bundle.qc.score(emb_far)
    assert np.median(s_far) > tiny_bundle.qc.reject
    assert abstain(s_far, tiny_bundle.qc).mean() > abstain(s_in, tiny_bundle.qc).mean()


def processed(raman, xaxis, max_count_fraction=0.3):
    baseline = 5000 + 0 * xaxis
    return SimpleNamespace(
        raman=raman, corrected=raman + baseline, xaxis=xaxis, max_count_fraction=max_count_fraction
    )


def tissue(xaxis, noise, rng):
    bands = ((1004, 10), (1156, 14), (1300, 30), (1450, 30), (1520, 16), (1660, 34))
    signal = sum(1000 * lorentz(xaxis, c, f) for c, f in bands)
    return signal + noise * rng.normal(size=xaxis.size)


def test_physics_checks():
    rng = np.random.default_rng(0)
    x = np.linspace(450, 1900, 900)
    clean = tissue(x, 10.0, rng)
    assert physics_checks(processed(clean, x), resolution_cm1=6.0) == []

    flags = dict(physics_checks(processed(clean, x, 0.995), resolution_cm1=6.0))
    assert "reject" in flags

    line_cm1 = 1e7 / 785.0 - 1e7 / ROOM_LIGHT_NM[0]
    lamp = clean + 800 * np.exp(-0.5 * ((x - line_cm1) / (6.0 / 2.355)) ** 2)
    reasons = [r for _, r in physics_checks(processed(lamp, x), resolution_cm1=6.0)]
    assert any("room light" in r for r in reasons)
    # the same narrow line away from the known argon wavelengths is not room light
    other = clean + 800 * np.exp(-0.5 * ((x - 1230.0) / (6.0 / 2.355)) ** 2)
    assert physics_checks(processed(other, x), resolution_cm1=6.0) == []

    noisy = tissue(x, 400.0, rng)
    reasons = [r for _, r in physics_checks(processed(noisy, x), resolution_cm1=6.0)]
    assert any("signal-to-noise" in r for r in reasons)
