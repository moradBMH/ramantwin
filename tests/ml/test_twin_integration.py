"""ML checks that need the twin: automatic exposure, robustness grid, physics QC rates."""

import numpy as np
import pytest

pytest.importorskip("ramantwin.pipeline")

from ramantwin.biochem import sample_cohort  # noqa: E402
from ramantwin.instrument import (  # noqa: E402
    InstrumentParams,
    VirtualInstrument,
    nominal,
    sample_fleet,
)
from ramantwin.ml.acquire import acquire_checked, auto_exposure, render  # noqa: E402
from ramantwin.ml.evaluate import robustness_grid  # noqa: E402
from ramantwin.ml.qc import physics_checks  # noqa: E402
from ramantwin.pipeline import calibrate, process  # noqa: E402

LOW_SNR = {"signal_e_per_s": 500.0, "exposure_s": 0.5, "n_accumulations": 1}


def measure(params: InstrumentParams, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    inst = VirtualInstrument(params)
    calib = calibrate(inst, rng)
    cohort = sample_cohort(n, 1, rng)
    return [
        process(inst.acquire_pixels(s, rng), calib) for s in render(inst, cohort.concentrations)
    ]


def flagged(procs, params, word: str) -> float:
    hits = [any(word in r for _, r in physics_checks(p, params.resolution_cm1)) for p in procs]
    return float(np.mean(hits))


@pytest.fixture(scope="module")
def instruments() -> list[InstrumentParams]:
    return [nominal(), *sample_fleet(4, np.random.default_rng(5), spread=1.0)]


def with_(params: InstrumentParams, **changes) -> InstrumentParams:
    return InstrumentParams.from_dict({**params.to_dict(), **changes})


def rate(instruments, word: str, n: int = 12, **changes) -> float:
    rates = []
    for i, p in enumerate(instruments):
        q = with_(p, **changes)
        rates.append(flagged(measure(q, n, 300 + i), q, word))
    return float(np.mean(rates))


def test_room_light_detection_and_false_positives(instruments):
    assert rate(instruments, "room-light", ambient_light=0.3) >= 0.9
    assert rate(instruments, "room-light") <= 0.02
    assert rate(instruments, "room-light", **LOW_SNR) <= 0.02


def test_low_snr_is_flagged_only_when_signal_is_weak(instruments):
    assert rate(instruments, "signal-to-noise", **LOW_SNR) >= 0.9
    assert rate(instruments, "signal-to-noise") <= 0.02


def test_auto_exposure_avoids_clipping():
    cohort = sample_cohort(6, 2, np.random.default_rng(0))
    for ratio in (1.0, 80.0):
        params = with_(nominal(), fluorescence_ratio=ratio)
        auto = auto_exposure(params, cohort.concentrations.mean(axis=0), np.random.default_rng(1))
        _, fill = acquire_checked(cohort, auto, 2)
        assert fill.max() < 0.98
        assert fill.max() > 0.25


def test_robustness_grid_shape_and_values():
    predictors = {"svm": lambda X: np.full(len(X), 0.9), "naive": lambda X: X[:, 500] > 0}
    grid = robustness_grid(
        predictors,
        x_values=[0.0, 1.0],
        y_values=[2.0, 60.0],
        n_patients=4,
        spectra_per_patient=1,
        n_jobs=1,
    )
    assert grid["models"]["svm"].shape == (2, 2)
    assert np.allclose(grid["models"]["svm"], 0.5)
    assert np.all(grid["clipped"] <= 0.05)
    assert grid["exposure_s"][1, 0] < grid["exposure_s"][0, 0]
