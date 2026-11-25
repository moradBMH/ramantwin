"""Export the web data files (docs/architecture.md section 8), the ONNX model and latency numbers.

CLI: `python -m ramantwin.ml.export --data data/demo --artifacts artifacts --out web/public/data`
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ramantwin.constants import CLASSES, GRID
from ramantwin.ml.models import ModelBundle, SpectralCNN
from ramantwin.ml.schema import VALIDATORS

VERSION = 1
MAX_POINTS = 4000
MAX_SPECTRA = 400
SPECTRA_STEP = 4
MAX_BYTES = 5_000_000

STAGES = (
    ("raw", "Raw counts"),
    ("orpl", "ORPL processed"),
    ("naive", "Naive CNN"),
    ("robust", "Robust CNN"),
)
# A large neighbourhood and min_dist give volumetric clusters instead of thin 1D strings for the
# CNN embeddings while keeping the per-instrument islands of the ORPL stage visible (Euclidean
# metric; cosine merged those islands). Chosen by comparing silhouettes and the eigenvalues of the
# 3D positions for n_neighbors 30-50, min_dist 0.3-0.8 and euclidean/cosine/correlation.
UMAP_KW = {"n_neighbors": 50, "min_dist": 0.5, "metric": "euclidean"}
CLUSTERED = 0.25  # silhouette above which a grouping is visible in the scene
# Augmentation shifts in cm-1 expressed as an equivalent laser drift at 785 nm (dv = 1e7 dl / l^2).
CM1_PER_NM = 1e7 / 785.0**2

PARAM_LABELS = {
    "laser_drift_nm": "Laser drift (nm)",
    "fluorescence_ratio": "Fluorescence / Raman",
    "x_offset_cm1": "X-axis offset (cm-1)",
    "resolution_cm1": "Resolution (cm-1)",
    "signal_e_per_s": "Signal (e-/s)",
    "ambient_light": "Ambient light",
}


def clean(obj, digits: int = 3):
    """Round floats, convert NumPy types and map NaN/inf to None so the output is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist(), digits)
    if isinstance(obj, np.integer | np.bool_):
        return obj.item()
    if isinstance(obj, float | np.floating):
        return round(float(obj), digits) if math.isfinite(obj) else None
    return obj


def write_json(path: Path, obj) -> int:
    """Clean, validate against section 8 and write; returns the size in bytes."""
    data = clean(obj)
    VALIDATORS[path.name](data)
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    if len(text) > MAX_BYTES:
        raise ValueError(f"{path.name} is {len(text)} bytes, above {MAX_BYTES}")
    path.write_text(text)
    return len(text)


def stratified_subsample(groups: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    """Sorted indices of at most `n_max` rows, the same number from every group when possible."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    per = n_max // len(uniq)
    idx = [rng.permutation(np.flatnonzero(groups == g))[:per] for g in uniq]
    return np.sort(np.concatenate(idx))


def scene_layout(
    features: list[np.ndarray], label: np.ndarray, instrument: np.ndarray, seed: int = 0
) -> tuple[list[np.ndarray], list[dict]]:
    """3D layout of every stage: PCA(50) -> UMAP(3), Procrustes-aligned to the previous stage.

    Silhouettes are computed on the displayed positions, so they describe what the audience sees.
    """
    import umap

    layouts = []
    for F in features:
        Z = PCA(min(50, F.shape[1], len(F) - 1), random_state=seed).fit_transform(F)
        U = umap.UMAP(
            n_components=3, random_state=seed, n_jobs=1, compatibility_layout=True, **UMAP_KW
        ).fit_transform(Z)
        U = U - U.mean(axis=0)
        U = U / np.sqrt((U**2).sum(axis=1).mean())
        if layouts:
            R, _ = orthogonal_procrustes(U, layouts[-1])
            U = U @ R
        layouts.append(U)
    layouts = [np.round(U / np.abs(U).max(), 3) for U in layouts]
    n = min(len(label), 3000)
    silhouettes = [
        {
            "instrument": silhouette_score(U, instrument, sample_size=n, random_state=seed),
            "class": silhouette_score(U, label, sample_size=n, random_state=seed),
        }
        for U in layouts
    ]
    return layouts, silhouettes


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def stage_subtitles(silhouettes: list[dict], acc: dict[str, dict[str, float]]) -> list[str]:
    """Captions that only claim what the silhouettes and accuracies of the scene support.

    `acc[model]` holds the balanced accuracy on the displayed train-instrument ("own") and fleet
    points.
    """
    raw, orpl, _, robust = silhouettes
    texts = []
    if raw["instrument"] > CLUSTERED:
        texts.append("Raw counts already split by instrument; no diagnostic structure")
    elif raw["class"] < CLUSTERED:
        texts.append("Fluorescence dominates; no diagnostic structure")
    else:
        texts.append("Fluorescence dominates, yet the diagnosis is partly visible")
    if orpl["instrument"] > CLUSTERED:
        texts.append("Raman bands emerge, but each instrument forms its own island")
    else:
        texts.append("Raman bands emerge and instruments overlap")
    own, fleet = acc["naive"]["own"], acc["naive"]["fleet"]
    if fleet < own - 0.05:
        texts.append(
            f"Separates diagnosis on its own instruments ({_pct(own)}); "
            f"fleet spectra spill over ({_pct(fleet)})"
        )
    else:
        texts.append(f"Separates diagnosis on its own instruments and on the fleet ({_pct(fleet)})")
    if robust["class"] > robust["instrument"]:
        texts.append(f"Organised by diagnosis across the fleet ({_pct(acc['robust']['fleet'])})")
    else:
        texts.append(f"Still partly organised by instrument ({_pct(acc['robust']['fleet'])})")
    return texts


def shift_scene(ds, out: dict[str, np.ndarray], eval_mask: np.ndarray, seed: int = 0) -> dict:
    """Build shift_scene.json from test spectra only (rows of `eval_mask`).

    `out` holds the bundle outputs on all the spectra of the FleetDataset `ds`.
    """
    from ramantwin.ml.evaluate import binary_metrics

    rows = np.flatnonzero(eval_mask)
    sel = rows[stratified_subsample(ds.instrument[rows], MAX_POINTS, seed)]
    y = ds.y[sel].astype(int)
    features = [ds.X_raw[sel], ds.X[sel], out["emb_naive"][sel], out["emb_robust"][sel]]
    layouts, silhouettes = scene_layout(features, y, ds.instrument[sel], seed)
    own = np.asarray(ds.roles)[ds.instrument[sel]] == "train"
    acc = {
        m: {
            "own": binary_metrics(out[m][sel][own], y[own])["bacc"],
            "fleet": binary_metrics(out[m][sel][~own], y[~own])["bacc"],
        }
        for m in ("naive", "robust")
    }
    subtitles = stage_subtitles(silhouettes, acc)
    spec = stratified_subsample(ds.instrument[sel], MAX_SPECTRA, seed + 1)
    return {
        "version": VERSION,
        "classes": list(CLASSES),
        "instruments": [
            {"id": i, "name": p.name, "role": r}
            for i, (p, r) in enumerate(zip(ds.instruments, ds.roles, strict=True))
        ],
        "stages": [
            {"key": k, "title": t, "subtitle": sub, "positions": U.ravel(), "silhouette": sil}
            for (k, t), sub, U, sil in zip(STAGES, subtitles, layouts, silhouettes, strict=True)
        ],
        "points": {
            "label": y,
            "instrument": ds.instrument[sel],
            "patient": ds.patient[sel],
            "p_naive": out["naive"][sel],
            "p_robust": out["robust"][sel],
            "ood": out["ood"][sel],
        },
        "spectra": {
            "grid": GRID[::SPECTRA_STEP],
            "index": spec,
            "raw": ds.X_raw[sel][spec, ::SPECTRA_STEP],
            "processed": ds.X[sel][spec, ::SPECTRA_STEP],
        },
    }


def envelope(params: list[dict], x_param: str, y_param: str, x_pad: float = 0.0) -> dict:
    xs = [p[x_param] for p in params]
    ys = [p[y_param] for p in params]
    return {"x": [min(xs) - x_pad, max(xs) + x_pad], "y": [min(ys), max(ys)]}


def robustness_atlas(
    grid: dict,
    x_param: str,
    y_param: str,
    train_params: list[dict],
    twin_params: list[dict] | None = None,
    augment_shift_cm1: float = 0.0,
) -> dict:
    """Build robustness_atlas.json from `evaluate.robustness_grid` output.

    The randomisation envelope spans the twins of the robust training set; on the laser drift
    axis it is widened by the augmentation shift converted to nm.
    """
    atlas = {
        "version": VERSION,
        "x": {
            "param": x_param,
            "label": PARAM_LABELS.get(x_param, x_param),
            "values": grid["x_values"],
        },
        "y": {
            "param": y_param,
            "label": PARAM_LABELS.get(y_param, y_param),
            "values": grid["y_values"],
        },
        "metric": "balanced_accuracy",
        "models": grid["models"],
        "train_envelope": envelope(train_params, x_param, y_param),
    }
    if twin_params:
        pad = augment_shift_cm1 / CM1_PER_NM if x_param == "laser_drift_nm" else 0.0
        atlas["randomization_envelope"] = envelope(twin_params, x_param, y_param, pad)
    return atlas


def fleet_summary(ds, evaluation: dict, latency_ms: dict) -> dict:
    """Build fleet.json from `evaluate.evaluate_fleet` output."""
    instruments = []
    for d, p in zip(evaluation["instruments"], ds.instruments, strict=True):
        instruments.append(
            {
                "id": d["id"],
                "name": p.name,
                "role": d["role"],
                "params": p.to_dict(),
                "metrics": d["metrics"],
                "abstain_rate": d["abstain_rate"],
            }
        )
    keys = ("mean_bacc", "worst_bacc", "mean_bacc_ci95")
    summary = {m: {k: s[k] for k in keys} for m, s in evaluation["summary"].items()}
    return {
        "version": VERSION,
        "instruments": instruments,
        "summary": summary,
        "latency_ms": latency_ms,
    }


LOW_SNR = {"signal_e_per_s": 500.0, "exposure_s": 0.5, "n_accumulations": 1}


def _auto(simulator, params: dict, label: int, seed: int) -> dict:
    """`params` with the exposure set automatically, within the Twin Lab slider range."""
    from ramantwin.api import sample_tissue
    from ramantwin.instrument import SLIDER_RANGES, InstrumentParams, nominal
    from ramantwin.ml.acquire import auto_exposure

    p = InstrumentParams.from_dict({**nominal().to_dict(), **params})
    _, conc = sample_tissue(label, seed)
    p = auto_exposure(p, conc, np.random.default_rng([seed, 2]), bounds=SLIDER_RANGES["exposure_s"])
    return {**params, "exposure_s": round(p.exposure_s, 3)}


def _first(simulator, params: dict, label: int, accept, auto: bool, n_seeds: int = 200):
    """First seed whose response satisfies `accept` (deterministic search)."""
    for seed in range(n_seeds):
        p = _auto(simulator, params, label, seed) if auto else params
        response = simulator.simulate(p, label, seed)
        if accept(response) and not (auto and _reasons(response, "saturated")):
            return seed, p, response
    raise RuntimeError(f"no seed satisfies the scenario {params}")


def _p(r: dict, model: str) -> float:
    return r["predictions"][model]["p_cancer"]


def _reasons(r: dict, word: str) -> bool:
    return any(word in reason for reason in r["qc"]["reasons"])


def scenario_rates(simulator, params: dict, auto: bool, n: int = 100) -> dict:
    """Error rates by model and class, and QC rates, over `n` samples per class at a setting.

    Seeds 10_000.. are used so that the rates do not reuse the displayed sample.
    """
    wrong = {m: {0: 0, 1: 0} for m in ("naive", "robust", "svm")}
    reject = room = low = 0
    for label in (0, 1):
        for seed in range(10_000, 10_000 + n):
            p = _auto(simulator, params, label, seed) if auto else params
            r = simulator.simulate(p, label, seed)
            for m in wrong:
                wrong[m][label] += (_p(r, m) >= 0.5) != (label == 1)
            reject += r["qc"]["flag"] == "reject"
            room += _reasons(r, "room light")
            low += _reasons(r, "signal-to-noise")
    total = 2 * n
    return {
        "wrong": {m: {c: v / n for c, v in w.items()} for m, w in wrong.items()},
        "reject": reject / total,
        "room_light": room / total,
        "low_snr": low / total,
        "n": n,
    }


def scenario_note(rates: dict, setting: str) -> str:
    """One sentence with the measured rates behind a Twin Lab sample.

    The illustrative flag is exported as its own field and the web app prefixes it.
    """
    w = rates["wrong"]
    if w["svm"][1] == 0 and w["svm"][0] == 1:
        svm = "SVM calls every spectrum cancer"
    elif w["svm"][1] == 1 and w["svm"][0] == 0:
        svm = "SVM calls every spectrum normal"
    else:
        svm = f"SVM {_pct(w['svm'][1])} / {_pct(w['svm'][0])}"
    text = (
        f"Over {rates['n']} samples per class {setting}: naive wrong on {_pct(w['naive'][1])} of "
        f"cancer / {_pct(w['naive'][0])} of normal spectra, robust {_pct(w['robust'][1])} / "
        f"{_pct(w['robust'][0])}, {svm}; QC rejects {_pct(rates['reject'])}"
    )
    if rates["room_light"] > 0.05:
        text += f", room light flagged on {_pct(rates['room_light'])}"
    if rates["low_snr"] > 0.05:
        text += f", low SNR flagged on {_pct(rates['low_snr'])}"
    return text + "."


def twinlab_samples(simulator, n_rates: int = 100) -> dict:
    """Pre-computed /api/simulate responses for the static (no backend) mode of the Twin Lab.

    Exposure is automatic (60 % of full well, ROWS device) except in the low-SNR scenario. Every
    scenario carries its seed, label and a note with the rates measured at its setting.
    """
    specs = [
        ("nominal", "Nominal", {}, lambda r: True, True, False, "on the nominal instrument"),
        # Illustrative sample: the seed is searched so that the naive CNN is wrong and the robust
        # CNN right, which happens on a minority of samples; the note gives the measured rates.
        (
            "laser_drift",
            "Laser drift +0.65 nm",
            {"laser_drift_nm": 0.65},
            lambda r: _p(r, "naive") < 0.5 < _p(r, "robust") and r["qc"]["flag"] != "reject",
            True,
            True,
            "at +0.65 nm",
        ),
        (
            "high_fluorescence",
            "Strong fluorescence",
            {"fluorescence_ratio": 20.0},
            lambda r: True,
            True,
            False,
            "at fluorescence 20x Raman",
        ),
        (
            "ambient_light",
            "Room light",
            {"ambient_light": 0.5},
            lambda r: _reasons(r, "room light"),
            True,
            False,
            "with room light 0.5",
        ),
        (
            "low_snr",
            "Low signal",
            LOW_SNR,
            lambda r: _reasons(r, "signal-to-noise") and not _reasons(r, "room light"),
            False,
            False,
            "at 500 e-/s, 0.5 s, one accumulation",
        ),
    ]
    scenarios = []
    for key, title, params, accept, auto, illustrative, setting in specs:
        seed, p, response = _first(simulator, params, 1, accept, auto)
        rates = scenario_rates(simulator, params, auto, n_rates)
        scenarios.append(
            {
                "key": key,
                "title": title,
                "params": p,
                "seed": seed,
                "label": 1,
                "illustrative": illustrative,
                "note": scenario_note(rates, setting),
                "response": response,
            }
        )
    return {"version": VERSION, "scenarios": scenarios}


def export_onnx(model: SpectralCNN, path: Path) -> None:
    """Export the robust CNN (temperature-scaled logits, dynamic batch) with opset 17."""
    torch.onnx.export(
        model.eval(),
        (torch.zeros(1, model.config["n_points"]),),
        str(path),
        input_names=["spectrum"],
        output_names=["logits"],
        dynamic_axes={"spectrum": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def onnx_latency_ms(path: Path, n_points: int = len(GRID), repeats: int = 300) -> float:
    """Median onnxruntime CPU latency for one spectrum (batch of 1)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(1, n_points)).astype(np.float32)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        session.run(None, {"spectrum": x})
        times.append(time.perf_counter() - t)
    return 1e3 * float(np.median(times[repeats // 10 :]))


def orpl_latency_ms(repeats: int = 30, seed: int = 0) -> float:
    """Median ORPL processing time of one nominal acquisition (calibration excluded)."""
    from ramantwin.biochem import sample_cohort
    from ramantwin.instrument import VirtualInstrument, nominal
    from ramantwin.ml.acquire import render
    from ramantwin.pipeline import calibrate, process

    rng = np.random.default_rng(seed)
    inst = VirtualInstrument(nominal())
    calib = calibrate(inst, rng)
    conc = sample_cohort(1, 1, rng).concentrations[0]
    acq = inst.acquire_pixels(render(inst, conc), rng)
    times = []
    for _ in range(repeats + 3):
        t = time.perf_counter()
        process(acq, calib)
        times.append(time.perf_counter() - t)
    return 1e3 * float(np.median(times[3:]))


FILES = ("shift", "fleet", "atlas", "twinlab")


def export_all(
    data: Path,
    artifacts: Path,
    out: Path,
    *,
    x_param: str = "laser_drift_nm",
    y_param: str = "fluorescence_ratio",
    grid_size: int = 11,
    seed: int = 0,
    only: tuple[str, ...] = FILES,
) -> dict[str, int]:
    """Write the web data files listed in `only` (and robust.onnx with fleet); returns sizes."""
    from ramantwin.fleet import FleetDataset
    from ramantwin.ml.evaluate import evaluate_fleet, grid_values, robustness_grid
    from ramantwin.ml.qc import abstain

    out.mkdir(parents=True, exist_ok=True)
    bundle = ModelBundle.load(artifacts)
    ds = FleetDataset.load(data)
    metrics = json.loads((artifacts / "metrics.json").read_text())
    roles = np.asarray(ds.roles)
    is_train = roles[ds.instrument] == "train"
    eval_mask = ~is_train | np.isin(ds.patient, metrics["holdout_patients"])
    res = bundle.outputs(ds.X)
    sizes = {}

    if "shift" in only:
        scene = shift_scene(ds, res, eval_mask, seed)
        sizes["shift_scene.json"] = write_json(out / "shift_scene.json", scene)

    if "fleet" in only:
        export_onnx(bundle.robust, artifacts / "robust.onnx")
        latency = {
            "orpl": orpl_latency_ms(seed=seed),
            "onnx_cnn": onnx_latency_ms(artifacts / "robust.onnx"),
        }
        evaluation = evaluate_fleet(
            {m: res[m] for m in bundle.predictors()},
            ds.y.astype(int),
            ds.instrument,
            roles,
            patient=ds.patient,
            eval_mask=eval_mask,
            reject=abstain(res["ood"], bundle.qc),
        )
        fleet = fleet_summary(ds, evaluation, latency)
        sizes["fleet.json"] = write_json(out / "fleet.json", fleet)

    if "atlas" in only:
        grid = robustness_grid(
            bundle.predictors(),
            x_param=x_param,
            y_param=y_param,
            x_values=grid_values(x_param, grid_size),
            y_values=grid_values(y_param, grid_size),
            seed=seed,
        )
        train_params = [
            p.to_dict() for p, r in zip(ds.instruments, ds.roles, strict=True) if r == "train"
        ]
        atlas = robustness_atlas(
            grid,
            x_param,
            y_param,
            train_params,
            metrics["twins"],
            metrics["augment"]["max_shift_cm1"],
        )
        sizes["robustness_atlas.json"] = write_json(out / "robustness_atlas.json", atlas)

    if "twinlab" in only:
        from ramantwin.api import Simulator

        samples = twinlab_samples(Simulator(bundle))
        sizes["twinlab_samples.json"] = write_json(out / "twinlab_samples.json", samples)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demo"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("web/public/data"))
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=FILES, action="append")
    args = parser.parse_args()
    sizes = export_all(
        args.data,
        args.artifacts,
        args.out,
        grid_size=args.grid_size,
        seed=args.seed,
        only=tuple(args.only or FILES),
    )
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
