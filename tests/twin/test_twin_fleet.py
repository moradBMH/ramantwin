import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from ramantwin import biochem
from ramantwin.constants import GRID
from ramantwin.fleet import FleetDataset, acquire_cohort, make_dataset, preset_instruments
from ramantwin.instrument import nominal


@pytest.fixture(scope="module")
def tiny():
    instruments, roles = preset_instruments("tiny")
    return make_dataset(instruments, roles, 4, 2, seed=3, n_jobs=1)


def test_dataset_shapes_and_dtypes(tiny):
    n = 3 * 4 * 2
    assert tiny.X.shape == tiny.X_raw.shape == (n, GRID.size)
    assert tiny.X.dtype == tiny.X_raw.dtype == np.float32
    assert tiny.y.dtype == np.int8 and tiny.patient.dtype == np.int32
    assert tiny.instrument.dtype == np.int16
    assert tiny.concentrations.shape == (n, len(biochem.COMPONENT_NAMES))
    assert tiny.roles == ["train", "fleet", "fleet"]
    assert [p.name for p in tiny.instruments] == ["MTL-01", "SN-1001", "SN-1002"]
    assert np.unique(tiny.patient).size == 12
    assert np.isfinite(tiny.X).all()


def test_save_load_roundtrip_and_subset(tiny, tmp_path):
    tiny.save(tmp_path / "fleet")
    loaded = FleetDataset.load(tmp_path / "fleet")
    for key in ("X_raw", "X", "y", "patient", "instrument", "concentrations"):
        np.testing.assert_array_equal(getattr(loaded, key), getattr(tiny, key))
    assert loaded.instruments == tiny.instruments
    assert loaded.roles == tiny.roles

    part = loaded.subset(loaded.instrument == 1)
    assert len(part.y) == 8 and (part.instrument == 1).all()
    assert part.instruments == tiny.instruments


def test_dataset_is_deterministic_and_parallel_safe(tiny):
    instruments, roles = preset_instruments("tiny")
    again = make_dataset(instruments, roles, 4, 2, seed=3, n_jobs=2)
    np.testing.assert_array_equal(again.X, tiny.X)
    np.testing.assert_array_equal(again.X_raw, tiny.X_raw)


def test_patient_offset_gives_new_patients(tiny):
    instruments, roles = preset_instruments("tiny")
    other = make_dataset(instruments[:1], roles[:1], 4, 2, seed=3, n_jobs=1, patient_offset=1000)
    assert other.patient.min() == 1000
    assert not np.isin(other.patient, tiny.patient).any()
    first = tiny.instrument == 0
    assert not np.allclose(other.concentrations, tiny.concentrations[first])


def test_acquire_cohort_seed():
    cohort = biochem.sample_cohort(2, 2, np.random.default_rng(0))
    a = acquire_cohort(cohort, nominal(), seed=11)
    b = acquire_cohort(cohort, nominal(), seed=11)
    c = acquire_cohort(cohort, nominal(), seed=12)
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_in_domain_svm_is_good_but_not_perfect():
    cohort = biochem.sample_cohort(80, 3, np.random.default_rng(0))
    _, X = acquire_cohort(cohort, nominal(), seed=1)
    scores = []
    folds = StratifiedGroupKFold(4, shuffle=True, random_state=0)
    for tr, te in folds.split(X, cohort.labels, cohort.patient_ids):
        model = LinearSVC(C=0.01, max_iter=20000).fit(X[tr], cohort.labels[tr])
        scores.append(balanced_accuracy_score(cohort.labels[te], model.predict(X[te])))
    assert 0.85 < np.mean(scores) < 0.97
