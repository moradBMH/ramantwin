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
    shift_scene,
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


def test_shift_scene_validates(tmp_path, fleet_ds, tiny_bundle):
    out = tiny_bundle.outputs(fleet_ds.X)
    scene = clean(shift_scene(fleet_ds, out, seed=0))
    validate_shift_scene(scene)
    n = len(scene["points"]["label"])
    first = np.array(scene["stages"][0]["positions"]).reshape(n, 3)
    assert np.abs(first).max() == pytest.approx(1.0, abs=1e-3)
    assert write_json(tmp_path / "shift_scene.json", shift_scene(fleet_ds, out)) < 5_000_000
    json.loads((tmp_path / "shift_scene.json").read_text())


def test_validator_rejects_bad_scene(fleet_ds, tiny_bundle):
    scene = clean(shift_scene(fleet_ds, tiny_bundle.outputs(fleet_ds.X)))
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
        eval_mask=np.ones(len(fleet_ds.y), bool),
        reject=np.zeros(len(fleet_ds.y), bool),
    )
    fleet = fleet_summary(fleet_ds, evaluation, {"orpl": 2.0, "onnx_cnn": 0.2})
    write_json(tmp_path / "fleet.json", fleet)

    grid = {
        "x_values": np.linspace(-1, 1, 3),
        "y_values": np.array([1.0, 10.0]),
        "models": {m: np.full((2, 3), 0.8) for m in ("svm", "naive", "robust")},
    }
    params = [p.to_dict() for p in fleet_ds.instruments[:1]]
    atlas = robustness_atlas(grid, "laser_drift_nm", "fluorescence_ratio", params)
    write_json(tmp_path / "robustness_atlas.json", atlas)
    loaded = json.loads((tmp_path / "robustness_atlas.json").read_text())
    assert loaded["train_envelope"] == {"x": [0.0, 0.0], "y": [5.0, 5.0]}


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
