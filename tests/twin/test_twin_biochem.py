import numpy as np

from ramantwin import biochem
from ramantwin.constants import GRID


def test_component_names_are_frozen():
    assert biochem.COMPONENT_NAMES == (
        "protein",
        "lipid",
        "nucleic_acid",
        "carotenoid",
        "cholesterol",
        "collagen",
        "heme",
    )
    assert set(biochem.COMPONENTS) == set(biochem.COMPONENT_NAMES)


def test_component_spectrum_peaks_at_literature_band():
    carotenoid = biochem.component_spectrum("carotenoid", GRID)
    assert carotenoid.max() == 1.0
    assert GRID[carotenoid.argmax()] == 1520
    assert GRID[biochem.component_spectrum("nucleic_acid", GRID).argmax()] == 785


def test_raman_spectrum_shapes():
    c = np.ones(len(biochem.COMPONENT_NAMES))
    assert biochem.raman_spectrum(c, GRID).shape == GRID.shape
    assert biochem.raman_spectrum(np.stack([c, 2 * c, c]), GRID).shape == (3, GRID.size)


def test_sample_cohort_structure():
    cohort = biochem.sample_cohort(10, 4, np.random.default_rng(0), patient_offset=100)
    assert cohort.concentrations.shape == (40, len(biochem.COMPONENT_NAMES))
    assert cohort.labels.sum() == 20
    assert set(cohort.patient_ids) == set(range(100, 110))
    for pid in np.unique(cohort.patient_ids):
        assert np.unique(cohort.labels[cohort.patient_ids == pid]).size == 1
    assert (cohort.concentrations > 0).all()


def test_cancer_follows_published_trends():
    cohort = biochem.sample_cohort(400, 2, np.random.default_rng(1))
    names = list(biochem.COMPONENT_NAMES)
    cancer = cohort.concentrations[cohort.labels == 1].mean(axis=0)
    normal = cohort.concentrations[cohort.labels == 0].mean(axis=0)
    assert cancer[names.index("nucleic_acid")] > normal[names.index("nucleic_acid")]
    assert cancer[names.index("carotenoid")] < normal[names.index("carotenoid")]
    assert cancer[names.index("lipid")] < normal[names.index("lipid")]
