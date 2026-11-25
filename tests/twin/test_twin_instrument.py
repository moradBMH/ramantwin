import numpy as np
import pytest
from orpl.calibration import find_npeaks

from ramantwin.constants import N_PIX, TYLENOL_CAL_PEAKS_CM1
from ramantwin.instrument import (
    SLIDER_RANGES,
    InstrumentParams,
    VirtualInstrument,
    acetaminophen_spectrum,
    nominal,
    sample_fleet,
)


def _band(center, fwhm=10.0):
    return lambda x: 1.0 / (1.0 + ((x - center) / (fwhm / 2)) ** 2)


def test_params_roundtrip_and_partial_dict():
    p = sample_fleet(1, np.random.default_rng(0))[0]
    assert InstrumentParams.from_dict(p.to_dict()) == p
    partial = InstrumentParams.from_dict({"laser_drift_nm": 0.4, "n_accumulations": 5.0})
    assert partial.laser_drift_nm == 0.4
    assert partial.n_accumulations == 5
    assert partial.resolution_cm1 == nominal().resolution_cm1


def test_slider_ranges_contain_nominal():
    d = nominal().to_dict()
    for key, (lo, hi) in SLIDER_RANGES.items():
        assert lo <= d[key] <= hi, key


def test_fleet_names_and_coverage():
    train = sample_fleet(2, np.random.default_rng(0), spread=0.15, prefix="MTL")
    fleet = sample_fleet(30, np.random.default_rng(1))
    assert [p.name for p in train] == ["MTL-01", "MTL-02"]
    assert fleet[0].name == "SN-1001"
    for p in train + fleet:
        shift = VirtualInstrument(p).pixel_shift_cm1()
        assert shift[0] < 450 and shift[-1] > 1900


def test_acquisition_shapes_and_determinism():
    p = InstrumentParams(n_accumulations=4, cosmic_rate=0.5)
    inst = VirtualInstrument(p)
    a = inst.acquire(_band(1000), np.random.default_rng(3))
    b = inst.acquire(_band(1000), np.random.default_rng(3))
    assert a.accumulations.shape == (4, N_PIX)
    assert a.dark.shape == (N_PIX,)
    np.testing.assert_array_equal(a.accumulations, b.accumulations)
    np.testing.assert_array_equal(a.dark, b.dark)
    assert a.accumulations.max() <= p.saturation_e


def test_laser_drift_shifts_pixels_by_16_cm1_per_nm():
    inst = VirtualInstrument(InstrumentParams(laser_drift_nm=1.0))
    delta = inst.pixel_shift_cm1(drift=False) - inst.pixel_shift_cm1(drift=True)
    np.testing.assert_allclose(delta, 1e7 / 785 - 1e7 / 786, rtol=1e-12)
    assert 16.0 < delta[0] < 16.5


def test_ambient_light_adds_sharp_lines_fixed_in_wavelength():
    dark_room = VirtualInstrument(InstrumentParams(cosmic_rate=0.0))
    lit_room = VirtualInstrument(InstrumentParams(cosmic_rate=0.0, ambient_light=1.0))
    spectrum = _band(1300, 40)
    a = dark_room.acquire(spectrum, np.random.default_rng(0), fluorescence=False)
    b = lit_room.acquire(spectrum, np.random.default_rng(0), fluorescence=False)
    diff = (b.accumulations - a.accumulations).mean(axis=0)

    line = np.abs(lit_room.pixel_wavelength_nm() - 852.14).argmin()
    noise = diff[line + 40 : line + 80].std()
    assert diff[line] > 30 * noise
    assert abs(diff[line + 20]) < 0.05 * diff[line]


def test_laser_drift_moves_raman_bands_but_not_ambient_lines():
    peaks = []
    for drift in (0.0, 1.0):
        inst = VirtualInstrument(InstrumentParams(laser_drift_nm=drift, ambient_light=1.0))
        acq = inst.acquire(_band(1300), np.random.default_rng(0), fluorescence=False)
        mean = (acq.accumulations - acq.dark).mean(axis=0)
        line = np.abs(inst.pixel_wavelength_nm() - 852.14).argmin()
        band = np.abs(inst.pixel_shift_cm1(drift=False) - 1300).argmin()
        peaks.append(
            (
                line - 10 + mean[line - 10 : line + 11].argmax(),
                band - 30 + mean[band - 30 : band + 31].argmax(),
            )
        )
    axis = VirtualInstrument(InstrumentParams()).pixel_shift_cm1(drift=False)
    (line_a, band_a), (line_b, band_b) = peaks
    assert line_a == line_b
    assert axis[band_b] - axis[band_a] == pytest.approx(16.2, abs=2.0)


def test_tylenol_most_prominent_bands_are_calibration_bands():
    for p in [nominal(), InstrumentParams(x_offset_cm1=-20, x_scale=1.02, resolution_cm1=20)]:
        inst = VirtualInstrument(p)
        shift = inst.pixel_shift_cm1(drift=False)
        found = shift[find_npeaks(inst.line_shape(acetaminophen_spectrum, drift=False), 5)]
        nearest = np.abs(found[:, None] - np.array(TYLENOL_CAL_PEAKS_CM1)).min(axis=1)
        assert nearest.max() < 3.0
