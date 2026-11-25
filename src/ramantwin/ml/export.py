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
    ("raw", "Raw counts", "Factory x-axis, no calibration, no baseline removal"),
    ("orpl", "ORPL processed", "Tylenol x-axis, NIST response, BubbleFill baseline, SNV"),
    ("naive", "Naive CNN", "Embedding of a CNN trained on the Montréal instruments only"),
    ("robust", "Robust CNN", "Embedding of a CNN trained on randomised digital twins"),
)

PARAM_LABELS = {
    "laser_drift_nm": "Laser drift (nm)",
    "fluorescence_ratio": "Fluorescence / Raman",
    "x_offset_cm1": "X-axis offset (cm-1)",
    "resolution_cm1": "Resolution (cm-1)",
    "signal_e_per_s": "Signal (e-/s)",
    "ambient_light": "Ambient light",
}

# Twin Lab scenarios: (title, {param: position in its slider range}); log ranges are interpolated
# geometrically. Positions rather than values keep the scenarios valid if the ranges change.
SCENARIOS = {
    "nominal": ("Nominal instrument", {}),
    "laser_drift": ("Laser drift after calibration", {"laser_drift_nm": 0.75}),
    "high_fluorescence": ("Strong tissue autofluorescence", {"fluorescence_ratio": 1.0}),
    "ambient_light": ("Room light leaking into the probe", {"ambient_light": 0.6}),
    "low_snr": ("Short exposure, weak signal", {"exposure_s": 0.0, "signal_e_per_s": 0.1}),
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

    Silhouettes are computed in the PCA space, which is less distorted than the 3D layout.
    """
    import umap

    layouts, silhouettes = [], []
    for F in features:
        Z = PCA(min(50, F.shape[1], len(F) - 1), random_state=seed).fit_transform(F)
        U = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.3,
            random_state=seed,
            n_jobs=1,
            compatibility_layout=True,
        ).fit_transform(Z)
        U = U - U.mean(axis=0)
        U = U / np.sqrt((U**2).sum(axis=1).mean())
        if layouts:
            R, _ = orthogonal_procrustes(U, layouts[-1])
            U = U @ R
        layouts.append(U)
        n = min(len(F), 2000)
        silhouettes.append(
            {
                "instrument": silhouette_score(Z, instrument, sample_size=n, random_state=seed),
                "class": silhouette_score(Z, label, sample_size=n, random_state=seed),
            }
        )
    return [U / np.abs(U).max() for U in layouts], silhouettes


def shift_scene(ds, out: dict[str, np.ndarray], seed: int = 0) -> dict:
    """Build shift_scene.json from a FleetDataset and the bundle outputs on all its spectra."""
    sel = stratified_subsample(ds.instrument, MAX_POINTS, seed)
    features = [ds.X_raw[sel], ds.X[sel], out["emb_naive"][sel], out["emb_robust"][sel]]
    layouts, silhouettes = scene_layout(features, ds.y[sel], ds.instrument[sel], seed)
    spec = stratified_subsample(ds.instrument[sel], MAX_SPECTRA, seed + 1)
    return {
        "version": VERSION,
        "classes": list(CLASSES),
        "instruments": [
            {"id": i, "name": p.name, "role": r}
            for i, (p, r) in enumerate(zip(ds.instruments, ds.roles, strict=True))
        ],
        "stages": [
            {"key": k, "title": t, "subtitle": s, "positions": U.ravel(), "silhouette": sil}
            for (k, t, s), U, sil in zip(STAGES, layouts, silhouettes, strict=True)
        ],
        "points": {
            "label": ds.y[sel],
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


def robustness_atlas(grid: dict, x_param: str, y_param: str, train_params: list[dict]) -> dict:
    """Build robustness_atlas.json from `evaluate.robustness_grid` output."""
    xs = [p[x_param] for p in train_params]
    ys = [p[y_param] for p in train_params]
    return {
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
        "train_envelope": {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]},
    }


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
    summary = {
        m: {"mean_bacc": s["mean_bacc"], "worst_bacc": s["worst_bacc"]}
        for m, s in evaluation["summary"].items()
    }
    return {
        "version": VERSION,
        "instruments": instruments,
        "summary": summary,
        "latency_ms": latency_ms,
    }


def scenario_params(fractions: dict[str, float]) -> dict[str, float]:
    from ramantwin.instrument import SLIDER_RANGES

    params = {}
    for name, f in fractions.items():
        lo, hi = SLIDER_RANGES[name]
        params[name] = lo * (hi / lo) ** f if lo > 0 and hi / lo >= 20 else lo + f * (hi - lo)
    return params


def twinlab_samples(simulator, seed: int = 0) -> dict:
    """Pre-computed /api/simulate responses for the static (no backend) mode of the Twin Lab."""
    scenarios = []
    for key, (title, fractions) in SCENARIOS.items():
        params = scenario_params(fractions)
        response = simulator.simulate(params, 1, seed)
        scenarios.append({"key": key, "title": title, "params": params, "response": response})
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
    from ramantwin.biochem import raman_spectrum, sample_cohort
    from ramantwin.instrument import VirtualInstrument, nominal
    from ramantwin.pipeline import calibrate, process

    rng = np.random.default_rng(seed)
    inst = VirtualInstrument(nominal())
    calib = calibrate(inst, rng)
    conc = sample_cohort(1, 1, rng).concentrations[0]
    acq = inst.acquire(lambda x: raman_spectrum(conc, x) / raman_spectrum(conc, x).max(), rng)
    times = []
    for _ in range(repeats + 3):
        t = time.perf_counter()
        process(acq, calib)
        times.append(time.perf_counter() - t)
    return 1e3 * float(np.median(times[3:]))


def export_all(
    data: Path,
    artifacts: Path,
    out: Path,
    *,
    x_param: str = "laser_drift_nm",
    y_param: str = "fluorescence_ratio",
    grid_size: int = 11,
    seed: int = 0,
) -> dict[str, int]:
    """Write every web data file plus artifacts/robust.onnx; returns the size of each file."""
    from ramantwin.fleet import FleetDataset
    from ramantwin.ml.evaluate import evaluate_fleet, grid_values, robustness_grid
    from ramantwin.ml.qc import abstain

    out.mkdir(parents=True, exist_ok=True)
    bundle = ModelBundle.load(artifacts)
    ds = FleetDataset.load(data)
    holdout_patients = np.array(
        json.loads((artifacts / "metrics.json").read_text())["holdout_patients"]
    )
    roles = np.asarray(ds.roles)
    is_train = roles[ds.instrument] == "train"
    eval_mask = ~is_train | np.isin(ds.patient, holdout_patients)

    res = bundle.outputs(ds.X)
    sizes = {"shift_scene.json": write_json(out / "shift_scene.json", shift_scene(ds, res, seed))}

    export_onnx(bundle.robust, artifacts / "robust.onnx")
    latency = {
        "orpl": orpl_latency_ms(seed=seed),
        "onnx_cnn": onnx_latency_ms(artifacts / "robust.onnx"),
    }
    evaluation = evaluate_fleet(
        {m: res[m] for m in bundle.predictors()},
        ds.y,
        ds.instrument,
        roles,
        eval_mask=eval_mask,
        reject=abstain(res["ood"], bundle.qc),
    )
    sizes["fleet.json"] = write_json(out / "fleet.json", fleet_summary(ds, evaluation, latency))

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
    atlas = robustness_atlas(grid, x_param, y_param, train_params)
    sizes["robustness_atlas.json"] = write_json(out / "robustness_atlas.json", atlas)

    from ramantwin.api import Simulator

    samples = twinlab_samples(Simulator(bundle), seed)
    sizes["twinlab_samples.json"] = write_json(out / "twinlab_samples.json", samples)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demo"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("web/public/data"))
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sizes = export_all(
        args.data, args.artifacts, args.out, grid_size=args.grid_size, seed=args.seed
    )
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
