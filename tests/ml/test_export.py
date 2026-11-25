import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ramantwin.ml.evaluate import evaluate_fleet
from ramantwin.ml.export import (
    clean,
    export_onnx,
    fleet_summary,
    onnx_latency_ms,
    robustness_atlas,
    scenario_note,
    shift_scene,
    stage_subtitles,
    write_json,
)
from ramantwin.ml.schema import SchemaError, validate_shift_scene
from ramantwin.ml.synthetic import make_spectra


class Params(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


@pytest.fixture(scope="module")
def fleet_ds():
    s = make_spectra(
        n_patients=12, spectra_per_patient=3, n_instruments=3, seed=2, shifts_cm1=(0.0, 3.0, 8.0)
    )
    return SimpleNamespace(
        X_raw=s.X + 0.1,
        X=s.X,
        y=s.y,
        patient=s.patient,
        instrument=s.instrument,
        instruments=[
            Params(name=f"SN-{i}", laser_drift_nm=0.1 * i, fluorescence_ratio=5.0 + i)
            for i in range(3)
        ],
        roles=["train", "fleet", "fleet"],
    )


def test_shift_scene_keeps_test_spectra_only(tmp_path, fleet_ds, tiny_bundle):
    out = tiny_bundle.outputs(fleet_ds.X)
    test = fleet_ds.instrument > 0
    scene = clean(shift_scene(fleet_ds, out, test, seed=0))
    validate_shift_scene(scene)
    assert set(scene["points"]["instrument"]) == {1, 2}
    assert set(scene["points"]["patient"]) <= set(fleet_ds.patient[test].tolist())
    n = len(scene["points"]["label"])
    first = np.array(scene["stages"][0]["positions"]).reshape(n, 3)
    assert np.abs(first).max() == pytest.approx(1.0, abs=1e-3)
    assert all(isinstance(s["subtitle"], str) and s["subtitle"] for s in scene["stages"])
    size = write_json(tmp_path / "shift_scene.json", shift_scene(fleet_ds, out, test))
    assert size < 5_000_000
    json.loads((tmp_path / "shift_scene.json").read_text())


def test_subtitles_follow_the_numbers():
    flat = {"instrument": 0.05, "class": 0.0}
    islands = {"instrument": 0.6, "class": 0.0}
    by_class = {"instrument": -0.1, "class": 0.4}
    acc = {"naive": {"own": 0.9, "fleet": 0.7}, "robust": {"own": 0.9, "fleet": 0.88}}
    texts = stage_subtitles([flat, islands, by_class, by_class], acc)
    assert "instrument" not in texts[0]
    assert "island" in texts[1]
    assert "spill" in texts[2]
    assert texts[3].startswith("Organised by diagnosis")
    assert "split by instrument" in stage_subtitles([islands, flat, flat, flat], acc)[0]


def test_validator_rejects_bad_scene(fleet_ds, tiny_bundle):
    test = np.ones(len(fleet_ds.y), bool)
    scene = clean(shift_scene(fleet_ds, tiny_bundle.outputs(fleet_ds.X), test))
    scene["stages"] = scene["stages"][::-1]
    with pytest.raises(SchemaError):
        validate_shift_scene(scene)


def test_fleet_and_atlas_validate(tmp_path, fleet_ds, tiny_bundle):
    res = tiny_bundle.outputs(fleet_ds.X)
    evaluation = evaluate_fleet(
        {m: res[m] for m in ("svm", "naive", "robust")},
        fleet_ds.y,
        fleet_ds.instrument,
        np.array(fleet_ds.roles),
        patient=fleet_ds.patient,
        eval_mask=np.ones(len(fleet_ds.y), bool),
        reject=np.zeros(len(fleet_ds.y), bool),
    )
    fleet = fleet_summary(fleet_ds, evaluation, {"orpl": 2.0, "onnx_cnn": 0.2})
    write_json(tmp_path / "fleet.json", fleet)
    lo, hi = fleet["summary"]["robust"]["mean_bacc_ci95"]
    assert lo <= fleet["summary"]["robust"]["mean_bacc"] <= hi

    grid = {
        "x_values": np.linspace(-1, 1, 3),
        "y_values": np.array([1.0, 10.0]),
        "models": {m: np.full((2, 3), 0.8) for m in ("svm", "naive", "robust")},
    }
    grid["models"]["naive"][1, 2] = np.nan
    params = [p.to_dict() for p in fleet_ds.instruments[:1]]
    twins = [
        {"laser_drift_nm": -0.5, "fluorescence_ratio": 2.0},
        {"laser_drift_nm": 0.4, "fluorescence_ratio": 12.0},
    ]
    atlas = robustness_atlas(grid, "laser_drift_nm", "fluorescence_ratio", params, twins, 16.23)
    write_json(tmp_path / "robustness_atlas.json", atlas)
    loaded = json.loads((tmp_path / "robustness_atlas.json").read_text())
    assert loaded["train_envelope"] == {"x": [0.0, 0.0], "y": [5.0, 5.0]}
    assert loaded["randomization_envelope"] == {"x": [-1.5, 1.4], "y": [2.0, 12.0]}
    assert loaded["models"]["naive"][1][2] is None


def test_clean_maps_nan_to_null():
    assert clean({"a": np.float32(0.12345), "b": float("nan"), "c": np.arange(2)}) == {
        "a": 0.123,
        "b": None,
        "c": [0, 1],
    }


def test_onnx_matches_torch(tmp_path, tiny_bundle):
    import onnxruntime as ort

    path = tmp_path / "robust.onnx"
    export_onnx(tiny_bundle.robust, path)
    x = np.random.default_rng(0).normal(size=(3, 1201)).astype(np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = tiny_bundle.robust(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(session.run(None, {"spectrum": x})[0], expected, atol=1e-4)
    assert onnx_latency_ms(path, repeats=20) > 0


def test_scenario_note_reports_rates():
    rates = {
        "wrong": {
            "naive": {1: 0.3, 0: 0.05},
            "robust": {1: 0.04, 0: 0.03},
            "svm": {1: 0.0, 0: 1.0},
        },
        "reject": 0.02,
        "room_light": 0.0,
        "low_snr": 0.0,
        "n": 100,
    }
    note = scenario_note(rates, "at +0.65 nm", illustrative=True)
    assert note == (
        "Illustrative sample. Over 100 samples per class at +0.65 nm: naive wrong on 30% of cancer"
        " / 5% of normal spectra, robust 4% / 3%, SVM calls every spectrum cancer; QC rejects 2%."
    )


def test_twinlab_optional_fields_are_validated():
    from ramantwin.ml.schema import validate_twinlab_samples

    curve = {"x": [1.0, 2.0], "y": [0.1, 0.2]}
    response = {
        "raw": curve,
        "baseline": curve,
        "processed": curve,
        "truth": {"label": 1, "concentrations": {}},
        "predictions": {m: {"p_cancer": 0.5} for m in ("svm", "naive", "robust")},
        "qc": {"flag": "ok", "ood_score": 1.0, "threshold": 2.0, "reasons": []},
        "timing_ms": {"acquire": 1.0, "orpl": 1.0, "inference": 1.0},
    }
    keys = ("nominal", "laser_drift", "high_fluorescence", "ambient_light", "low_snr")
    doc = {
        "version": 1,
        "scenarios": [
            {
                "key": k,
                "title": k,
                "params": {},
                "seed": 3,
                "label": 1,
                "illustrative": False,
                "note": "n",
                "response": response,
            }
            for k in keys
        ],
    }
    validate_twinlab_samples(doc)
    doc["scenarios"][1]["illustrative"] = "yes"
    with pytest.raises(SchemaError):
        validate_twinlab_samples(doc)
