import numpy as np
import pytest

from ramantwin import biochem
from ramantwin.constants import GRID, N_PIX, TRUNCATE_BELOW_CM1
from ramantwin.instrument import InstrumentParams, VirtualInstrument, sample_fleet
from ramantwin.pipeline import axis_error_cm1, calibrate, clean_mean, process, raw_on_grid


def _band(center, fwhm=10.0):
    return lambda x: 1.0 / (1.0 + ((x - center) / (fwhm / 2)) ** 2)


@pytest.fixture(scope="module")
def nominal_setup():
    inst = VirtualInstrument(InstrumentParams())
    return inst, calibrate(inst, np.random.default_rng(0))


def test_calibration_recovers_axis_without_drift():
    for p in [InstrumentParams(), *sample_fleet(4, np.random.default_rng(5))]:
        inst = VirtualInstrument(p)
        calib = calibrate(inst, np.random.default_rng(1))
        assert calib.xaxis.shape == (N_PIX,)
        assert calib.irf.shape == (N_PIX,)
        assert axis_error_cm1(calib, inst) < 2.0
        assert 0.0 <= calib.x_residual < 10.0


def test_processed_shapes(nominal_setup):
    inst, calib = nominal_setup
    cohort = biochem.sample_cohort(1, 1, np.random.default_rng(2))
    tissue = cohort.concentrations[0]
    acq = inst.acquire(lambda x: biochem.raman_spectrum(tissue, x), np.random.default_rng(3))
    out = process(acq, calib)
    m = out.xaxis.size
    assert m < N_PIX and out.xaxis[0] >= TRUNCATE_BELOW_CM1
    for arr in (out.raw_mean, out.corrected, out.baseline, out.raman):
        assert arr.shape == (m,)
    np.testing.assert_allclose(out.raman, out.corrected - out.baseline)
    assert out.final.shape == GRID.shape
    assert abs(out.final.mean()) < 1e-9 and abs(out.final.std() - 1) < 1e-9
    assert 0 < out.max_count_fraction < 1

    raw = raw_on_grid(acq, inst)
    assert raw.shape == GRID.shape and raw.max() == 1.0


def test_laser_drift_moves_processed_peak(nominal_setup):
    _, calib = nominal_setup  # calibration is not redone after the drift
    peaks = []
    for drift in (0.0, 0.5):
        inst = VirtualInstrument(InstrumentParams(laser_drift_nm=drift))
        out = process(inst.acquire(_band(1100), np.random.default_rng(4)), calib)
        window = (GRID > 1050) & (GRID < 1150)
        peaks.append(GRID[window][out.final[window].argmax()])
    assert peaks[1] - peaks[0] == pytest.approx(8.1, abs=1.5)


def test_irf_correction_flattens_tilt():
    inst = VirtualInstrument(InstrumentParams(irf_tilt=-0.5, irf_curvature=0.2, etalon_amp=0.1))
    calib = calibrate(inst, np.random.default_rng(0))
    acq = inst.acquire(np.ones_like, np.random.default_rng(1), fluorescence=False)
    out = process(acq, calib)
    assert out.raw_mean.std() / out.raw_mean.mean() > 0.3
    assert out.corrected.std() / out.corrected.mean() < 0.02


def test_cosmic_rays_are_removed(nominal_setup):
    _, calib = nominal_setup
    spectrum = _band(1300, 40)
    clean = VirtualInstrument(InstrumentParams(cosmic_rate=0.0))
    hit = VirtualInstrument(InstrumentParams(cosmic_rate=3.0))
    a = clean.acquire(spectrum, np.random.default_rng(7))
    b = hit.acquire(spectrum, np.random.default_rng(7))

    unfiltered = (b.accumulations - a.accumulations).mean(axis=0)
    filtered = clean_mean(b) - clean_mean(a)
    noise = (a.accumulations - a.dark).std(axis=0).mean() / np.sqrt(len(a.accumulations))
    assert unfiltered.max() > 50 * noise
    assert np.abs(filtered).max() < 6 * noise


def test_saturation_is_reported(nominal_setup):
    _, calib = nominal_setup
    inst = VirtualInstrument(InstrumentParams(signal_e_per_s=10000, exposure_s=5.0))
    out = process(inst.acquire(_band(1300), np.random.default_rng(0)), calib)
    assert out.max_count_fraction == 1.0
