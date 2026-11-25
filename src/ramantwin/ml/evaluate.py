"""Evaluation: leave-instruments-out metrics, risk-coverage and the robustness grid.

All splits are grouped by patient: several spectra of one patient never sit on both sides.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

Predictor = Callable[[np.ndarray], np.ndarray]


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of binary probabilities p = P(y = 1), confidence-binned."""
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    bins = np.minimum((conf - 0.5) / 0.5 * n_bins, n_bins - 1).astype(int)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def binary_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Balanced accuracy at 0.5, ROC AUC (NaN for a single class) and ECE."""
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return {
        "bacc": _bacc(p, y),
        "auc": float(auc),
        "ece": ece(p, y),
    }


def risk_coverage(p: np.ndarray, y: np.ndarray, score: np.ndarray) -> dict:
    """Error rate among the most trusted inputs (lowest OOD score first) at every coverage."""
    order = np.argsort(score, kind="stable")
    errors = ((p[order] >= 0.5) != (y[order] == 1)).astype(float)
    k = np.arange(1, len(p) + 1)
    risk = np.cumsum(errors) / k
    return {"coverage": k / len(p), "risk": risk, "aurc": float(risk.mean())}


def _bacc(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, p >= 0.5))


def assert_no_patient_leak(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    shared = np.intersect1d(train_groups, test_groups)
    if len(shared):
        raise ValueError(f"{len(shared)} patients appear in both train and test")


def evaluate_fleet(
    probs: dict[str, np.ndarray],
    y: np.ndarray,
    instrument: np.ndarray,
    roles: np.ndarray,
    *,
    eval_mask: np.ndarray,
    reject: np.ndarray,
) -> dict:
    """Per-instrument metrics of every model on the rows of `eval_mask`.

    Train instruments are scored on their held-out patients only (in-domain), fleet instruments on
    all their spectra (leave-instruments-out). `reject` marks inputs where the robust model
    abstains.
    """
    instruments = []
    for i, role in enumerate(roles):
        m = eval_mask & (instrument == i)
        keep = m & ~reject
        instruments.append(
            {
                "id": i,
                "role": str(role),
                "n": int(m.sum()),
                "metrics": {k: binary_metrics(p[m], y[m]) for k, p in probs.items()},
                "abstain_rate": float(reject[m].mean()),
                "robust_selective_bacc": _bacc(probs["robust"][keep], y[keep]),
            }
        )
    fleet = [d for d in instruments if d["role"] == "fleet"]
    summary = {
        k: {
            "mean_bacc": float(np.mean([d["metrics"][k]["bacc"] for d in fleet])),
            "worst_bacc": float(np.min([d["metrics"][k]["bacc"] for d in fleet])),
            "mean_auc": float(np.mean([d["metrics"][k]["auc"] for d in fleet])),
            "mean_ece": float(np.mean([d["metrics"][k]["ece"] for d in fleet])),
        }
        for k in probs
    }
    return {"instruments": instruments, "summary": summary}


def grid_values(param: str, n: int = 11) -> np.ndarray:
    """Values of `param` across its Twin Lab slider range (log-spaced for wide positive ranges)."""
    from ramantwin.instrument import SLIDER_RANGES

    lo, hi = SLIDER_RANGES[param]
    if lo > 0 and hi / lo >= 20:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def _acquire_cell(cohort, params_dict: dict, seed: int) -> np.ndarray:
    from ramantwin.fleet import acquire_cohort
    from ramantwin.instrument import InstrumentParams

    return acquire_cohort(cohort, InstrumentParams.from_dict(params_dict), seed)[1]


def robustness_grid(
    predictors: dict[str, Predictor],
    *,
    x_param: str = "laser_drift_nm",
    y_param: str = "fluorescence_ratio",
    x_values: np.ndarray | None = None,
    y_values: np.ndarray | None = None,
    n_patients: int = 40,
    spectra_per_patient: int = 3,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict:
    """Balanced accuracy of every model on one fixed test cohort re-acquired at every grid cell.

    All other instrument parameters stay nominal. Returns values and `models[name][iy][ix]`.
    """
    from joblib import Parallel, delayed

    from ramantwin.biochem import sample_cohort
    from ramantwin.instrument import nominal

    x_values = grid_values(x_param) if x_values is None else np.asarray(x_values)
    y_values = grid_values(y_param) if y_values is None else np.asarray(y_values)
    cohort = sample_cohort(
        n_patients, spectra_per_patient, np.random.default_rng(seed), patient_offset=10_000_000
    )
    base = nominal().to_dict()
    cells = [(iy, ix) for iy in range(len(y_values)) for ix in range(len(x_values))]
    Xs = Parallel(n_jobs=n_jobs)(
        delayed(_acquire_cell)(
            cohort,
            {**base, x_param: float(x_values[ix]), y_param: float(y_values[iy])},
            seed + 1 + iy * len(x_values) + ix,
        )
        for iy, ix in cells
    )
    models = {k: np.zeros((len(y_values), len(x_values))) for k in predictors}
    for (iy, ix), X in zip(cells, Xs, strict=True):
        for k, f in predictors.items():
            models[k][iy, ix] = _bacc(f(X), cohort.labels)
    return {"x_values": x_values, "y_values": y_values, "models": models}
