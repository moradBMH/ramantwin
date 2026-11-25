import numpy as np

from ramantwin.ml.evaluate import (
    binary_metrics,
    bootstrap_bacc,
    ece,
    evaluate_fleet,
    risk_coverage,
)


def test_ece_of_calibrated_probabilities_is_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert ece(p, y) < 0.02
    assert ece(np.where(y == 1, 0.99, 0.01), 1 - y) > 0.9


def test_binary_metrics():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(np.array([0.1, 0.6, 0.7, 0.9]), y)
    assert m["bacc"] == 0.75
    assert m["auc"] == 1.0


def test_risk_coverage_prefers_low_scores():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    rc = risk_coverage(p, y, score=np.array([0.0, 1.0, 5.0, 6.0]))
    np.testing.assert_allclose(rc["risk"], [0, 0, 1 / 3, 0.5])
    np.testing.assert_allclose(rc["coverage"], [0.25, 0.5, 0.75, 1.0])


def test_evaluate_fleet_uses_eval_mask_and_abstention():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    instrument = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    good = np.where(y == 1, 0.8, 0.2)
    probs = {"svm": good, "naive": 1 - good, "robust": good}
    reject = np.array([0, 0, 0, 0, 1, 0, 0, 0], bool)
    eval_mask = np.array([0, 0, 1, 1, 1, 1, 1, 1], bool)
    patient = np.arange(8)
    res = evaluate_fleet(
        probs,
        y,
        instrument,
        np.array(["train", "fleet"]),
        patient=patient,
        eval_mask=eval_mask,
        reject=reject,
        n_boot=200,
    )
    train, fleet = res["instruments"]
    assert train["n"] == 2 and fleet["n"] == 4
    assert fleet["abstain_rate"] == 0.25
    assert res["summary"]["svm"]["worst_bacc"] == 1.0
    assert res["summary"]["naive"]["mean_bacc"] == 0.0
    assert list(res["paired_differences"]) == ["robust-naive"]
    diff = res["paired_differences"]["robust-naive"]
    assert diff["mean_bacc_diff"] == 1.0
    assert diff["ci95"] == [1.0, 1.0]


def test_hierarchical_bootstrap_includes_instrument_variability():
    rng = np.random.default_rng(0)
    groups = []
    for acc in (0.6, 0.95):
        y = np.repeat(np.arange(40) % 2, 5)
        patient = np.repeat(np.arange(40), 5)
        p = np.where(rng.random(200) < acc, y, 1 - y).astype(float)
        groups.append((p, y, patient))
    samples = bootstrap_bacc(groups, n_boot=500)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    assert lo < 0.7 and hi > 0.85
    assert np.allclose(samples, bootstrap_bacc(groups, n_boot=500))
