"""Minimal validators of the web data files (docs/architecture.md section 8).

Each function raises `SchemaError` with the path of the first problem found.
"""

from __future__ import annotations

import math

STAGE_KEYS = ("raw", "orpl", "naive", "robust")
MODELS = ("svm", "naive", "robust")
SCENARIOS = ("nominal", "laser_drift", "high_fluorescence", "ambient_light", "low_snr")
MAX_POINTS = 4000
MAX_SPECTRA = 400


class SchemaError(ValueError):
    pass


def _check(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise SchemaError(f"{where}: {msg}")


def _keys(obj, keys, where: str) -> None:
    _check(isinstance(obj, dict), where, "expected an object")
    missing = [k for k in keys if k not in obj]
    _check(not missing, where, f"missing keys {missing}")


def _numbers(values, where: str, n: int | None = None, allow_null: bool = False) -> None:
    _check(isinstance(values, list), where, "expected a list")
    if n is not None:
        _check(len(values) == n, where, f"expected {n} values, got {len(values)}")
    for v in values:
        if v is None and allow_null:
            continue
        ok = isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
        _check(ok, where, f"not a finite number: {v!r}")


def _decimals(values, where: str, digits: int = 3) -> None:
    for v in values:
        if isinstance(v, float):
            _check(round(v, digits) == v, where, f"more than {digits} decimals: {v}")


def _curve(obj, where: str, max_points: int) -> None:
    _keys(obj, ("x", "y"), where)
    _numbers(obj["x"], f"{where}.x")
    _numbers(obj["y"], f"{where}.y", len(obj["x"]))
    _check(len(obj["x"]) <= max_points, where, f"more than {max_points} points")


def validate_shift_scene(d: dict) -> None:
    _keys(d, ("version", "classes", "instruments", "stages", "points", "spectra"), "shift_scene")
    _check(d["version"] == 1, "version", "expected 1")
    ids = []
    for i, inst in enumerate(d["instruments"]):
        _keys(inst, ("id", "name", "role"), f"instruments[{i}]")
        _check(inst["role"] in ("train", "fleet"), f"instruments[{i}].role", "train or fleet")
        ids.append(inst["id"])
    pts = d["points"]
    _keys(pts, ("label", "instrument", "patient", "p_naive", "p_robust", "ood"), "points")
    n = len(pts["label"])
    _check(0 < n <= MAX_POINTS, "points", f"between 1 and {MAX_POINTS} points")
    for k in ("label", "instrument", "patient", "p_naive", "p_robust", "ood"):
        _numbers(pts[k], f"points.{k}", n)
        _decimals(pts[k], f"points.{k}")
    _check(set(pts["label"]) <= {0, 1}, "points.label", "labels must be 0 or 1")
    _check(set(pts["instrument"]) <= set(ids), "points.instrument", "unknown instrument id")
    _check([s.get("key") for s in d["stages"]] == list(STAGE_KEYS), "stages", "keys out of order")
    for s in d["stages"]:
        where = f"stages.{s['key']}"
        _keys(s, ("title", "subtitle", "positions", "silhouette"), where)
        _numbers(s["positions"], f"{where}.positions", 3 * n)
        _decimals(s["positions"], f"{where}.positions")
        _check(all(-1 <= v <= 1 for v in s["positions"]), where, "positions outside [-1, 1]")
        _keys(s["silhouette"], ("instrument", "class"), f"{where}.silhouette")
    sp = d["spectra"]
    _keys(sp, ("grid", "index", "raw", "processed"), "spectra")
    _numbers(sp["grid"], "spectra.grid")
    _check(len(sp["index"]) <= MAX_SPECTRA, "spectra.index", f"more than {MAX_SPECTRA}")
    _check(all(0 <= i < n for i in sp["index"]), "spectra.index", "index outside points")
    for k in ("raw", "processed"):
        _check(len(sp[k]) == len(sp["index"]), f"spectra.{k}", "one row per index")
        for row in sp[k]:
            _numbers(row, f"spectra.{k}[]", len(sp["grid"]))
            _decimals(row, f"spectra.{k}[]")


def validate_robustness_atlas(d: dict) -> None:
    _keys(d, ("version", "x", "y", "metric", "models", "train_envelope"), "robustness_atlas")
    for axis in ("x", "y"):
        _keys(d[axis], ("param", "label", "values"), axis)
        _numbers(d[axis]["values"], f"{axis}.values")
    nx, ny = len(d["x"]["values"]), len(d["y"]["values"])
    _keys(d["models"], MODELS, "models")
    for m in MODELS:
        _check(len(d["models"][m]) == ny, f"models.{m}", f"expected {ny} rows")
        for row in d["models"][m]:
            _numbers(row, f"models.{m}[]", nx, allow_null=True)
    for env in ("train_envelope", "randomization_envelope"):
        if env in d or env == "train_envelope":
            _keys(d[env], ("x", "y"), env)
            _numbers(d[env]["x"], f"{env}.x", 2)
            _numbers(d[env]["y"], f"{env}.y", 2)


def validate_fleet(d: dict) -> None:
    _keys(d, ("version", "instruments", "summary", "latency_ms"), "fleet")
    for i, inst in enumerate(d["instruments"]):
        where = f"instruments[{i}]"
        _keys(inst, ("id", "name", "role", "params", "metrics", "abstain_rate"), where)
        _keys(inst["metrics"], MODELS, f"{where}.metrics")
        for m in MODELS:
            _keys(inst["metrics"][m], ("bacc", "auc"), f"{where}.metrics.{m}")
        _numbers([inst["abstain_rate"]], f"{where}.abstain_rate")
    _keys(d["summary"], MODELS, "summary")
    for m in MODELS:
        _keys(d["summary"][m], ("mean_bacc", "worst_bacc"), f"summary.{m}")
        if "mean_bacc_ci95" in d["summary"][m]:
            _numbers(d["summary"][m]["mean_bacc_ci95"], f"summary.{m}.mean_bacc_ci95", 2)
    _keys(d["latency_ms"], ("orpl", "onnx_cnn"), "latency_ms")


def validate_simulate_response(d: dict) -> None:
    keys = ("raw", "baseline", "processed", "truth", "predictions", "qc", "timing_ms")
    _keys(d, keys, "simulate")
    for k in ("raw", "baseline", "processed"):
        _curve(d[k], k, 601 if k == "processed" else 600)
    _keys(d["truth"], ("label", "concentrations"), "truth")
    _keys(d["predictions"], MODELS, "predictions")
    for m in MODELS:
        _numbers([d["predictions"][m]["p_cancer"]], f"predictions.{m}.p_cancer")
    _keys(d["qc"], ("flag", "ood_score", "threshold", "reasons"), "qc")
    _check(d["qc"]["flag"] in ("ok", "warn", "reject"), "qc.flag", "ok, warn or reject")
    _keys(d["timing_ms"], ("acquire", "orpl", "inference"), "timing_ms")


def validate_twinlab_samples(d: dict) -> None:
    _keys(d, ("version", "scenarios"), "twinlab_samples")
    _check([s.get("key") for s in d["scenarios"]] == list(SCENARIOS), "scenarios", "wrong keys")
    for s in d["scenarios"]:
        _keys(s, ("key", "title", "params", "response"), f"scenarios.{s['key']}")
        validate_simulate_response(s["response"])


VALIDATORS = {
    "shift_scene.json": validate_shift_scene,
    "robustness_atlas.json": validate_robustness_atlas,
    "fleet.json": validate_fleet,
    "twinlab_samples.json": validate_twinlab_samples,
}
