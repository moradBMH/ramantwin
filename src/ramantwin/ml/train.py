"""Training of the three models: linear SVM, naive CNN and domain-randomised robust CNN.

CLI: `python -m ramantwin.ml.train --data data/demo --out artifacts`
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.pipeline import Pipeline

from ramantwin.ml.augment import AugmentParams
from ramantwin.ml.models import ModelBundle, SpectralCNN, cnn_outputs, make_svm


def patient_split(groups: np.ndarray, y: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
    """Boolean mask of the validation rows; whole patients only, stratified by patient label."""
    rng = np.random.default_rng(seed)
    patients, first = np.unique(groups, return_index=True)
    val_patients = []
    for label in np.unique(y):
        candidates = patients[y[first] == label]
        n_val = max(1, round(val_fraction * len(candidates)))
        val_patients.append(rng.choice(candidates, n_val, replace=False))
    return np.isin(groups, np.concatenate(val_patients))


def train_svm(X: np.ndarray, y: np.ndarray, C: float = 0.01) -> Pipeline:
    """Fit the ROWS-style linear SVM on processed spectra."""
    return make_svm(C=C).fit(X, y)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class CNNFit:
    model: SpectralCNN
    val_mask: np.ndarray
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Temperature scaling (Guo et al. 2017): minimise validation NLL over a single scalar."""
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


@torch.no_grad()
def refresh_batchnorm(model: SpectralCNN, X: torch.Tensor, batch_size: int = 512) -> None:
    """Recompute BatchNorm statistics with the current weights (exact average over X).

    With small training sets the running averages lag behind the weights and the eval-mode
    network drifts away from the train-mode one.
    """
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for start in range(0, len(X), batch_size):
        model.embed(X[start : start + batch_size])
    for m in norms:
        m.momentum = 0.1
    model.eval()


def train_cnn(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    augment: AugmentParams | None = None,
    epochs: int = 40,
    seed: int = 0,
    device: str = "cpu",
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    patience: int = 8,
    val_fraction: float = 0.2,
    width: int = 32,
) -> CNNFit:
    """Train a SpectralCNN with AdamW + cosine schedule and patient-grouped early stopping.

    The best validation-loss weights are kept, then a temperature is fitted on the same
    validation patients so that `softmax(forward(x))` is calibrated.
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    val = patient_split(groups, y, val_fraction, seed)
    Xt = torch.as_tensor(X[~val], dtype=torch.float32, device=device)
    yt = torch.as_tensor(y[~val], dtype=torch.long, device=device)
    Xv = torch.as_tensor(X[val], dtype=torch.float32, device=device)
    yv = torch.as_tensor(y[val], dtype=torch.long, device=device)

    model = SpectralCNN(n_points=X.shape[1], width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = (np.inf, 0, copy.deepcopy(model.state_dict()))
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        total = 0.0
        for start in range(0, len(Xt), batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            xb = Xt[idx] if augment is None else augment(Xt[idx])
            loss = F.cross_entropy(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        refresh_batchnorm(model, Xt)
        with torch.no_grad():
            logits_v = torch.cat([model(Xv[s : s + 2048]) for s in range(0, len(Xv), 2048)])
            val_loss = float(F.cross_entropy(logits_v, yv))
            pred = logits_v.argmax(1)
            recalls = [float((pred[yv == c] == c).float().mean()) for c in (0, 1)]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(Xt),
                "val_loss": val_loss,
                "val_bacc": float(np.mean(recalls)),
            }
        )
        if val_loss < best[0]:
            best = (val_loss, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= patience:
            break

    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        logits_v = torch.cat([model(Xv[s : s + 2048]) for s in range(0, len(Xv), 2048)])
    model.temperature.fill_(fit_temperature(logits_v.cpu(), yv.cpu()))
    model.to("cpu")
    return CNNFit(model, val, history, best[1], time.perf_counter() - t0)


def _acquire_twin(concentrations, labels, patients, params, seed):
    from ramantwin.biochem import COMPONENT_NAMES, Cohort
    from ramantwin.fleet import acquire_cohort

    cohort = Cohort(concentrations, labels, patients, COMPONENT_NAMES)
    return acquire_cohort(cohort, params, seed)[1]


def build_randomized_training_set(
    concentrations: np.ndarray,
    y: np.ndarray,
    patient: np.ndarray,
    n_twins: int,
    spread: float,
    seed: int,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Re-acquire the same training samples through `n_twins` random virtual instruments.

    Returns (X, y, patient, twins) where rows are grouped twin by twin and `twins` is the list of
    sampled InstrumentParams. Patients never change, so patient-grouped splits stay valid.
    """
    from joblib import Parallel, delayed

    from ramantwin.instrument import sample_fleet

    twins = sample_fleet(n_twins, np.random.default_rng(seed), spread=spread, prefix="DR")
    Xs = Parallel(n_jobs=n_jobs)(
        delayed(_acquire_twin)(concentrations, y, patient, p, seed + 1000 + i)
        for i, p in enumerate(twins)
    )
    X = np.concatenate(Xs).astype(np.float32)
    return X, np.tile(y, n_twins), np.tile(patient, n_twins), twins


def train_all(
    data: Path,
    out: Path,
    *,
    epochs: int = 40,
    n_twins: int = 12,
    spread: float = 1.5,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train svm, naive and robust models plus the QC model on a saved FleetDataset."""
    from ramantwin.fleet import FleetDataset
    from ramantwin.ml.evaluate import binary_metrics, evaluate_fleet
    from ramantwin.ml.qc import MahalanobisOOD, abstain

    out.mkdir(parents=True, exist_ok=True)
    ds = FleetDataset.load(data)
    roles = np.asarray(ds.roles)
    is_train = roles[ds.instrument] == "train"
    holdout = np.zeros(len(ds.y), bool)
    holdout[is_train] = patient_split(ds.patient[is_train], ds.y[is_train], 0.2, seed)
    fit = is_train & ~holdout
    y = ds.y.astype(np.int64)

    t0 = time.perf_counter()
    svm = train_svm(ds.X[fit], y[fit])
    svm_seconds = time.perf_counter() - t0

    naive = train_cnn(ds.X[fit], y[fit], ds.patient[fit], epochs=epochs, seed=seed, device=device)

    t0 = time.perf_counter()
    Xr, yr, pr, twins = build_randomized_training_set(
        ds.concentrations[is_train], y[is_train], ds.patient[is_train], n_twins, spread, seed
    )
    twin_seconds = time.perf_counter() - t0
    hold_r = np.tile(holdout[is_train], n_twins)
    X_rob = np.concatenate([ds.X[fit], Xr[~hold_r]])
    y_rob = np.concatenate([y[fit], yr[~hold_r]])
    p_rob = np.concatenate([ds.patient[fit], pr[~hold_r]])
    robust = train_cnn(
        X_rob, y_rob, p_rob, augment=AugmentParams(), epochs=epochs, seed=seed, device=device
    )

    _, emb_fit = cnn_outputs(robust.model, X_rob[~robust.val_mask])
    X_cal = np.concatenate([ds.X[holdout], Xr[hold_r]])
    _, emb_cal = cnn_outputs(robust.model, X_cal)
    qc = MahalanobisOOD.fit(emb_fit, y_rob[~robust.val_mask]).calibrate(emb_cal)

    bundle = ModelBundle(svm, naive.model, robust.model, qc)
    bundle.save(out)
    res = bundle.outputs(ds.X)
    in_domain = {m: binary_metrics(res[m][holdout], y[holdout]) for m in bundle.predictors()}
    fleet = evaluate_fleet(
        {m: res[m] for m in bundle.predictors()},
        y,
        ds.instrument,
        roles,
        eval_mask=~fit,
        reject=abstain(res["ood"], qc),
    )
    metrics = {
        "seed": seed,
        "n_train_spectra": int(fit.sum()),
        "n_holdout_spectra": int(holdout.sum()),
        "holdout_patients": np.unique(ds.patient[holdout]).tolist(),
        "n_robust_spectra": len(y_rob),
        "n_twins": n_twins,
        "twin_spread": spread,
        "seconds": {
            "svm": svm_seconds,
            "naive": naive.seconds,
            "twin_acquisition": twin_seconds,
            "robust": robust.seconds,
        },
        "cnn": {
            name: {
                "best_epoch": f.best_epoch,
                "temperature": float(f.model.temperature),
                "history": f.history,
            }
            for name, f in (("naive", naive), ("robust", robust))
        },
        "qc": {"warn": qc.warn, "reject": qc.reject},
        "in_domain_holdout": in_domain,
        "fleet": fleet,
        "twins": [p.to_dict() for p in twins],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demo"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--twins", type=int, default=12)
    parser.add_argument("--spread", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    m = train_all(
        args.data,
        args.out,
        epochs=args.epochs,
        n_twins=args.twins,
        spread=args.spread,
        seed=args.seed,
        device=resolve_device(args.device),
    )
    print(json.dumps({"in_domain_holdout": m["in_domain_holdout"], **m["fleet"]["summary"]}))


if __name__ == "__main__":
    main()
