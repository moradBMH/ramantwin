"""Tissue model: biochemical components with literature Raman band assignments.

Band positions follow Movasaghi et al., Appl. Spectrosc. Rev. 42 (2007) and Talari et al.,
Appl. Spectrosc. Rev. 50 (2015). Heights are relative within a component (785 nm excitation).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    center: float
    fwhm: float
    height: float


def _bands(*rows: tuple[float, float, float]) -> tuple[Band, ...]:
    return tuple(Band(c, w, h) for c, w, h in rows)


COMPONENTS: dict[str, tuple[Band, ...]] = {
    "protein": _bands(
        (621, 12, 0.10),  # phenylalanine
        (643, 12, 0.12),  # tyrosine
        (760, 12, 0.12),  # tryptophan
        (830, 16, 0.18),  # tyrosine Fermi doublet
        (853, 16, 0.22),
        (935, 25, 0.25),  # C-C backbone (alpha helix)
        (1004, 8, 0.60),  # phenylalanine ring breathing
        (1032, 10, 0.18),  # phenylalanine C-H in-plane
        (1125, 20, 0.18),  # C-N stretch
        (1250, 40, 0.45),  # amide III
        (1340, 25, 0.30),  # C-H deformation
        (1450, 30, 0.80),  # CH2/CH3 deformation
        (1585, 14, 0.12),  # phenylalanine / tyrosine
        (1617, 14, 0.15),  # tyrosine / tryptophan
        (1655, 35, 1.00),  # amide I
    ),
    "lipid": _bands(
        (875, 20, 0.15),
        (970, 20, 0.10),  # =C-H out of plane
        (1063, 20, 0.30),  # C-C stretch (trans)
        (1080, 25, 0.30),  # C-C stretch (gauche)
        (1265, 25, 0.35),  # =C-H in-plane deformation
        (1301, 22, 0.70),  # CH2 twisting
        (1440, 25, 1.00),  # CH2 scissoring
        (1660, 25, 0.70),  # C=C stretch
        (1745, 20, 0.35),  # C=O ester
    ),
    "nucleic_acid": _bands(
        (670, 15, 0.25),  # thymine / guanine
        (728, 15, 0.35),  # adenine
        (785, 18, 1.00),  # O-P-O backbone, pyrimidine ring breathing
        (1095, 25, 0.60),  # PO2- symmetric stretch
        (1250, 20, 0.20),  # cytosine
        (1340, 25, 0.60),  # adenine / guanine
        (1375, 20, 0.35),  # thymine / adenine / guanine
        (1485, 20, 0.45),  # guanine / adenine
        (1578, 20, 0.70),  # guanine / adenine
    ),
    "carotenoid": _bands(
        (960, 14, 0.08),
        (1008, 10, 0.20),  # C-CH3 rocking
        (1156, 10, 0.75),  # C-C stretch
        (1520, 12, 1.00),  # C=C stretch
    ),
    "cholesterol": _bands(
        (548, 12, 0.40),
        (608, 12, 0.30),
        (700, 10, 0.80),  # sterol ring
        (1130, 20, 0.20),
        (1300, 20, 0.35),
        (1440, 25, 1.00),
        (1670, 15, 0.70),  # sterol C=C
    ),
    "collagen": _bands(
        (815, 15, 0.30),  # C-C backbone
        (855, 15, 0.60),  # proline
        (876, 15, 0.50),  # hydroxyproline
        (938, 20, 0.60),  # C-C backbone
        (1004, 8, 0.30),  # phenylalanine
        (1033, 10, 0.15),
        (1245, 25, 0.55),  # amide III
        (1270, 25, 0.45),
        (1450, 30, 0.80),
        (1665, 35, 1.00),  # amide I
    ),
    "heme": _bands(
        (752, 12, 0.60),  # pyrrole breathing (nu15)
        (1128, 15, 0.35),
        (1172, 15, 0.25),
        (1375, 18, 0.50),  # nu4
        (1547, 15, 0.40),
        (1585, 15, 0.70),  # nu37 / nu19
        (1620, 15, 0.60),  # vinyl C=C
        (1640, 15, 0.40),  # nu10
    ),
}

COMPONENT_NAMES: tuple[str, ...] = tuple(COMPONENTS)

# Mean concentrations (relative Raman scattering weights). Cancer follows published trends:
# more nucleic acids and proteins, fewer lipids, carotenoids and cholesterol.
_NORMAL = np.array([0.50, 0.42, 0.10, 0.10, 0.14, 0.20, 0.06])
_CANCER = np.array([0.635, 0.26, 0.18, 0.06, 0.086, 0.227, 0.0735])

PATIENT_SIGMA = 0.20  # log-normal patient random effect per component
SPECTRUM_SIGMA = 0.10  # log-normal intra-patient variability per component
INTENSITY_SIGMA = 0.15  # overall collection efficiency (focus, probe contact)


def component_spectrum(name: str, xaxis: np.ndarray) -> np.ndarray:
    """Noiseless spectrum of one component on `xaxis` (cm-1), Lorentzian bands, max = 1."""
    x = np.asarray(xaxis, dtype=float)
    y = np.zeros_like(x)
    for b in COMPONENTS[name]:
        y += b.height / (1.0 + ((x - b.center) / (b.fwhm / 2.0)) ** 2)
    return y / y.max()


def component_matrix(xaxis: np.ndarray) -> np.ndarray:
    """All component spectra stacked as (C, len(xaxis))."""
    return np.stack([component_spectrum(n, xaxis) for n in COMPONENT_NAMES])


def raman_spectrum(concentrations: np.ndarray, xaxis: np.ndarray) -> np.ndarray:
    """Noiseless tissue Raman spectrum: (C,) -> (len(xaxis),) or (n, C) -> (n, len(xaxis))."""
    return np.asarray(concentrations, dtype=float) @ component_matrix(xaxis)


@dataclass
class Cohort:
    concentrations: np.ndarray
    labels: np.ndarray
    patient_ids: np.ndarray
    component_names: tuple[str, ...]


def sample_cohort(
    n_patients: int,
    spectra_per_patient: int,
    rng: np.random.Generator,
    cancer_fraction: float = 0.5,
    patient_offset: int = 0,
) -> Cohort:
    """Sample patients and their spectra concentrations.

    Cancer patients have a patient-level tumour cell fraction (Beta(5, 2)) that varies from
    spectrum to spectrum, so cancer spectra are mixtures of tumour and normal tissue.
    """
    n_cancer = int(round(n_patients * cancer_fraction))
    patient_labels = rng.permutation(np.r_[np.zeros(n_patients - n_cancer), np.ones(n_cancer)])
    patient_fraction = rng.beta(5.0, 2.0, n_patients) * patient_labels
    patient_effect = rng.normal(0.0, PATIENT_SIGMA, (n_patients, len(COMPONENT_NAMES)))

    labels = np.repeat(patient_labels, spectra_per_patient).astype(int)
    patient_idx = np.repeat(np.arange(n_patients), spectra_per_patient)
    n = labels.size

    fraction = np.clip(patient_fraction[patient_idx] + rng.normal(0.0, 0.1, n), 0.0, 1.0)
    fraction *= labels
    mean = _NORMAL + fraction[:, None] * (_CANCER - _NORMAL)
    log_effect = patient_effect[patient_idx] + rng.normal(0.0, SPECTRUM_SIGMA, mean.shape)
    intensity = np.exp(rng.normal(0.0, INTENSITY_SIGMA, n))
    concentrations = mean * np.exp(log_effect) * intensity[:, None]

    return Cohort(
        concentrations=concentrations,
        labels=labels,
        patient_ids=patient_offset + patient_idx,
        component_names=COMPONENT_NAMES,
    )
