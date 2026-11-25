import numpy as np
import pytest

from ramantwin.ml.augment import AugmentParams
from ramantwin.ml.evaluate import assert_no_patient_leak
from ramantwin.ml.train import build_randomized_training_set, patient_split, train_cnn


def test_patient_split_never_leaks(synthetic):
    for seed in range(5):
        val = patient_split(synthetic.patient, synthetic.y, 0.25, seed)
        assert_no_patient_leak(synthetic.patient[~val], synthetic.patient[val])
        assert set(synthetic.y[val]) == {0, 1}
        assert 0.15 < val.mean() < 0.35


def test_leak_is_detected():
    with pytest.raises(ValueError):
        assert_no_patient_leak(np.array([1, 2, 3]), np.array([3, 4]))


def test_train_cnn_learns_and_calibrates(synthetic):
    s = synthetic
    fit = train_cnn(
        s.X, s.y, s.patient, augment=AugmentParams(), epochs=10, width=8, batch_size=32, seed=0
    )
    assert len(fit.history) >= 1
    assert fit.history[fit.best_epoch]["val_bacc"] > 0.6
    assert_no_patient_leak(s.patient[fit.val_mask], s.patient[~fit.val_mask])


def test_randomized_training_set_reuses_patients():
    pytest.importorskip("ramantwin.fleet")
    from ramantwin.biochem import sample_cohort

    cohort = sample_cohort(4, 2, np.random.default_rng(0))
    X, y, patient, twins = build_randomized_training_set(
        cohort.concentrations, cohort.labels, cohort.patient_ids, 2, 1.0, 0, n_jobs=1
    )
    assert X.shape == (16, 1201)
    assert len(twins) == 2
    np.testing.assert_array_equal(patient[:8], cohort.patient_ids)
    np.testing.assert_array_equal(y[8:], cohort.labels)
