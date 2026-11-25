"""Digital twin of a 785 nm dispersive Raman spectrometer with a 1024-pixel CCD."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields

import numpy as np
from orpl.calibration import NIST_COEFS
from scipy.ndimage import convolve1d

from ramantwin.biochem import Band
from ramantwin.constants import LASER_NM, N_PIX


@dataclass
class InstrumentParams:
    name: str = "nominal"
    laser_nm: float = 785.0
    laser_drift_nm: float = 0.0
    x_offset_cm1: float = 0.0
    x_scale: float = 1.0
    resolution_cm1: float = 6.0
    irf_tilt: float = 0.0
    irf_curvature: float = 0.0
    etalon_amp: float = 0.0
    fluorescence_ratio: float = 5.0
    fluorescence_shape: float = 1.0
    signal_e_per_s: float = 2000.0
    exposure_s: float = 1.0
    n_accumulations: int = 10
    read_noise_e: float = 8.0
    dark_e_per_s: float = 2.0
    cosmic_rate: float = 0.05
    ambient_light: float = 0.0
    saturation_e: float = 65000.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InstrumentParams":
        """Build from a full or partial dict; missing keys take the nominal value."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "n_accumulations" in kwargs:
            kwargs["n_accumulations"] = int(kwargs["n_accumulations"])
        return cls(**kwargs)


SLIDER_RANGES: dict[str, tuple[float, float]] = {
    "laser_drift_nm": (-1.5, 1.5),
    "x_offset_cm1": (-20.0, 20.0),
    "x_scale": (0.98, 1.02),
    "resolution_cm1": (4.0, 20.0),
    "irf_tilt": (-0.5, 0.5),
    "irf_curvature": (-0.3, 0.3),
    "etalon_amp": (0.0, 0.15),
    "fluorescence_ratio": (0.5, 80.0),
    "fluorescence_shape": (0.5, 2.0),
    "signal_e_per_s": (100.0, 10000.0),
    "exposure_s": (0.1, 5.0),
    "n_accumulations": (1, 30),
    "read_noise_e": (2.0, 30.0),
    "dark_e_per_s": (0.0, 50.0),
    "cosmic_rate": (0.0, 1.0),
    "ambient_light": (0.0, 1.0),
}

# Factory dispersion: quadratic in wavelength, about 400-1975 cm-1 at 785 nm.
_SHIFT_FIRST, _SHIFT_LAST = 400.0, 1975.0
_LAM_FIRST = 1e7 / (1e7 / LASER_NM - _SHIFT_FIRST)
_LAM_LAST = 1e7 / (1e7 / LASER_NM - _SHIFT_LAST)
_DISP_B = -5e-6  # nm / px^2
_DISP_A = (_LAM_LAST - _LAM_FIRST - _DISP_B * (N_PIX - 1) ** 2) / (N_PIX - 1)

FINE_STEP_CM1 = 0.5
BIAS_E = 500.0
CAL_ACCUMULATIONS = 10
NIST_ACCUMULATIONS = 20
CAL_FILL = 0.4  # calibration exposures are set to reach this fraction of saturation

# Argon I lines of fluorescent room lighting in the sensor range (nm, relative intensity).
AMBIENT_LINES_NM = (
    (811.53, 1.0),
    (826.45, 0.3),
    (842.46, 0.6),
    (852.14, 0.45),
    (866.79, 0.1),
    (912.30, 1.0),
    (922.45, 0.45),
)
AMBIENT_E_PER_S = 3000.0

ETALON_PERIOD_CM1 = 92.0  # ~15 um back-thinned silicon

ACETAMINOPHEN: tuple[Band, ...] = tuple(
    Band(c, w, h)
    for c, w, h in (
        (390, 10, 0.20),
        (465, 8, 0.20),
        (504, 8, 0.15),
        (651, 7, 0.30),
        (710, 8, 0.10),
        (797, 8, 0.80),
        (858, 8, 0.90),
        (968, 8, 0.10),
        (1105, 8, 0.15),
        (1169, 8, 1.00),
        (1237, 9, 0.70),
        (1278, 9, 0.15),
        (1324, 9, 0.75),
        (1371, 9, 0.20),
        (1561, 8, 0.30),
        (1609, 9, 0.95),
        (1648, 9, 0.60),
    )
)


def acetaminophen_spectrum(xaxis: np.ndarray) -> np.ndarray:
    """Acetaminophen (Tylenol) Raman spectrum on `xaxis` (cm-1), Lorentzian bands."""
    y = np.zeros_like(np.asarray(xaxis, dtype=float))
    for b in ACETAMINOPHEN:
        y += b.height / (1.0 + ((xaxis - b.center) / (b.fwhm / 2.0)) ** 2)
    return y


def nominal() -> InstrumentParams:
    return InstrumentParams()


def _lognormal(rng: np.random.Generator, median: float, sigma: float) -> float:
    return float(median * np.exp(rng.normal(0.0, sigma)))


def sample_fleet(
    n: int, rng: np.random.Generator, spread: float = 1.0, prefix: str = "SN"
) -> list[InstrumentParams]:
    """Sample `n` instruments around nominal; `spread` scales every manufacturing tolerance.

    Names are SN-1001, SN-1002, ... for the default prefix, otherwise PREFIX-01, PREFIX-02, ...
    """
    s = spread
    out = []
    for i in range(n):
        name = f"SN-{1001 + i}" if prefix == "SN" else f"{prefix}-{i + 1:02d}"
        out.append(
            InstrumentParams(
                name=name,
                laser_nm=785.0 + float(rng.normal(0.0, 0.15 * s)),
                laser_drift_nm=float(np.clip(rng.normal(0.0, 0.25 * s), -1.5, 1.5)),
                x_offset_cm1=float(np.clip(rng.normal(0.0, 6.0 * s), -20.0, 20.0)),
                x_scale=float(np.clip(1.0 + rng.normal(0.0, 0.004 * s), 0.98, 1.02)),
                resolution_cm1=float(np.clip(_lognormal(rng, 6.0, 0.3 * s), 4.0, 20.0)),
                irf_tilt=float(np.clip(rng.normal(0.0, 0.2 * s), -0.5, 0.5)),
                irf_curvature=float(np.clip(rng.normal(0.0, 0.1 * s), -0.3, 0.3)),
                etalon_amp=float(min(abs(rng.normal(0.0, 0.03 * s)), 0.15)),
                fluorescence_ratio=float(np.clip(_lognormal(rng, 5.0, 0.6 * s), 0.5, 80.0)),
                fluorescence_shape=float(np.clip(_lognormal(rng, 1.0, 0.25 * s), 0.5, 2.0)),
                signal_e_per_s=float(np.clip(_lognormal(rng, 2000.0, 0.5 * s), 100.0, 10000.0)),
                read_noise_e=float(np.clip(_lognormal(rng, 8.0, 0.3 * s), 2.0, 30.0)),
                dark_e_per_s=float(np.clip(_lognormal(rng, 2.0, 0.5 * s), 0.0, 50.0)),
                cosmic_rate=float(np.clip(_lognormal(rng, 0.05, 0.5 * s), 0.0, 1.0)),
            )
        )
    return out


@dataclass
class Acquisition:
    accumulations: np.ndarray
    dark: np.ndarray
    saturation_e: float = 65000.0


class VirtualInstrument:
    """Pixel-level forward model: line shape, fluorescence, response, noise, artefacts."""

    def __init__(self, params: InstrumentParams):
        self.params = params
        pix = np.arange(N_PIX, dtype=float)
        self._nu_nominal = 1e7 / (_LAM_FIRST + _DISP_A * pix + _DISP_B * pix**2)
        center = self._nu_nominal[N_PIX // 2]
        # absolute wavenumber seen by each pixel, misaligned by offset and dispersion scale
        self._nu = center + params.x_scale * (self._nu_nominal - center) - params.x_offset_cm1
        self._u = np.linspace(-1.0, 1.0, N_PIX)
        self.irf = self._response()

    def pixel_wavelength_nm(self) -> np.ndarray:
        return 1e7 / self._nu

    def pixel_shift_cm1(self, drift: bool = True) -> np.ndarray:
        """True Raman shift of each pixel; `drift` adds the post-calibration laser drift."""
        laser = self.params.laser_nm + (self.params.laser_drift_nm if drift else 0.0)
        return 1e7 / laser - self._nu

    def nominal_axis_cm1(self) -> np.ndarray:
        """Factory axis, i.e. what the instrument believes without calibration."""
        return 1e7 / LASER_NM - self._nu_nominal

    def _response(self) -> np.ndarray:
        p = self.params
        lam = self.pixel_wavelength_nm()
        quantum_eff = 1.0 / (1.0 + np.exp((lam - 940.0) / 40.0))  # deep-depletion CCD roll-off
        shape = 1.0 + p.irf_tilt * self._u + p.irf_curvature * (self._u**2 - 1.0 / 3.0)
        # fringes grow as silicon becomes transparent in the near infrared
        weight = np.clip((lam - 800.0) / 130.0, 0.0, 1.0)
        etalon = 1.0 + p.etalon_amp * weight * np.sin(2.0 * np.pi * self._nu / ETALON_PERIOD_CM1)
        return quantum_eff * shape * etalon

    def line_shape(
        self, spectrum_fn: Callable[[np.ndarray], np.ndarray], drift: bool = True
    ) -> np.ndarray:
        """Noiseless spectrum seen by the pixels: Gaussian slit function then pixel sampling.

        `spectrum_fn` may return (m,) or (k, m) for m shifts; the result is (N_PIX,) or (k, N_PIX).
        """
        shift = self.pixel_shift_cm1(drift)
        fine = np.arange(shift.min() - 60.0, shift.max() + 60.0, FINE_STEP_CM1)
        sigma = self.params.resolution_cm1 / (2.0 * np.sqrt(2.0 * np.log(2.0))) / FINE_STEP_CM1
        half = int(np.ceil(4.0 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        smooth = convolve1d(spectrum_fn(fine), kernel / kernel.sum(), axis=-1, mode="nearest")
        if smooth.ndim == 1:
            return np.interp(shift, fine, smooth)
        return np.stack([np.interp(shift, fine, row) for row in smooth])

    def fluorescence(self, rng: np.random.Generator) -> np.ndarray:
        """Random tissue autofluorescence in wavelength space, max ~ fluorescence_ratio.

        Amplitude (+/- 30 %), emission peak, width and a low-order distortion change from one
        spectrum to the next, independently of the tissue class.
        """
        p = self.params
        lam = self.pixel_wavelength_nm()
        peak = rng.uniform(770.0, 840.0)
        width = 90.0 / p.fluorescence_shape * np.exp(rng.normal(0.0, 0.15))
        tilt, bend = rng.normal(0.0, 0.1, 2)
        bg = np.exp(-0.5 * ((lam - peak) / width) ** 2)
        bg *= 1.0 + tilt * self._u + bend * (self._u**2 - 1.0 / 3.0)
        return p.fluorescence_ratio * np.exp(rng.normal(0.0, 0.3)) * bg / bg.max()

    def ambient_profile(self) -> np.ndarray:
        """Room-light lines on the pixels, fixed in wavelength, strongest line = 1."""
        sigma = self.params.resolution_cm1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        out = np.zeros(N_PIX)
        for lam, rel in AMBIENT_LINES_NM:
            out += rel * np.exp(-0.5 * ((self._nu - 1e7 / lam) / sigma) ** 2)
        return out

    def _expose(
        self,
        photo_e: np.ndarray,
        rng: np.random.Generator,
        exposure_s: float,
        n_acc: int,
    ) -> Acquisition:
        p = self.params
        dark_e = p.dark_e_per_s * exposure_s
        size = (n_acc, N_PIX)
        frames = rng.poisson(np.clip(photo_e * self.irf, 0.0, None), size).astype(float)
        frames += rng.poisson(dark_e, size) + BIAS_E + rng.normal(0.0, p.read_noise_e, size)
        dark = rng.poisson(dark_e, size) + BIAS_E + rng.normal(0.0, p.read_noise_e, size)

        # cosmic rays are drawn last so that they are the only difference when cosmic_rate changes
        hits = rng.poisson(p.cosmic_rate, n_acc)
        for i in range(n_acc):
            for _ in range(hits[i]):
                start = rng.integers(0, N_PIX - 3)
                width = rng.integers(1, 4)
                energy = rng.lognormal(np.log(6000.0), 0.6)
                frames[i, start : start + width] += energy * rng.uniform(0.3, 1.0, width)

        np.minimum(frames, p.saturation_e, out=frames)
        np.minimum(dark, p.saturation_e, out=dark)
        return Acquisition(frames, dark.mean(axis=0), p.saturation_e)

    def acquire_pixels(
        self, raman: np.ndarray, rng: np.random.Generator, fluorescence: bool = True
    ) -> Acquisition:
        """Acquire a sample whose noiseless Raman spectrum is already sampled on the pixels."""
        p = self.params
        signal = raman + (self.fluorescence(rng) * raman.max() if fluorescence else 0.0)
        photo = signal * p.signal_e_per_s * p.exposure_s
        if p.ambient_light > 0:
            room_e = p.ambient_light * AMBIENT_E_PER_S * p.exposure_s
            photo = photo + room_e * self.ambient_profile()
        return self._expose(photo, rng, p.exposure_s, p.n_accumulations)

    def acquire(
        self,
        spectrum_fn: Callable[[np.ndarray], np.ndarray],
        rng: np.random.Generator,
        fluorescence: bool = True,
    ) -> Acquisition:
        """Acquire a sample; `spectrum_fn` maps Raman shift (cm-1) to intensity (max ~ 1)."""
        return self.acquire_pixels(self.line_shape(spectrum_fn), rng, fluorescence)

    def _calibration_exposure(
        self, signal: np.ndarray, rng: np.random.Generator, n_acc: int
    ) -> Acquisition:
        # the operator sets the integration time so that the standard fills ~40 % of the well
        peak = (signal * self.irf).max()
        photo = signal * (CAL_FILL * (self.params.saturation_e - BIAS_E) / peak)
        return self._expose(photo, rng, 1.0, n_acc)

    def acquire_tylenol(self, rng: np.random.Generator) -> Acquisition:
        """Acetaminophen tablet for x-axis calibration, at the calibration laser wavelength."""
        raman = self.line_shape(acetaminophen_spectrum, drift=False)
        return self._calibration_exposure(raman, rng, CAL_ACCUMULATIONS)

    def acquire_nist(self, rng: np.random.Generator) -> Acquisition:
        """NIST SRM 2241 luminescence standard; its certified curve is the ORPL polynomial."""
        luminescence = np.polynomial.polynomial.polyval(
            self.pixel_shift_cm1(drift=False), NIST_COEFS
        )
        return self._calibration_exposure(luminescence, rng, NIST_ACCUMULATIONS)
