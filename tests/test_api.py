import pytest
from fastapi.testclient import TestClient

from ramantwin.api import create_app
from ramantwin.ml.models import ModelBundle, cnn_outputs
from ramantwin.ml.qc import MahalanobisOOD
from ramantwin.ml.schema import validate_simulate_response
from ramantwin.ml.synthetic import make_spectra
from ramantwin.ml.train import train_cnn, train_svm


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    s = make_spectra(n_patients=12, spectra_per_patient=3, seed=3)
    naive = train_cnn(s.X, s.y, s.patient, epochs=1, width=8).model
    robust = train_cnn(s.X, s.y, s.patient, epochs=1, width=8, seed=1).model
    emb = cnn_outputs(robust, s.X)[1]
    qc = MahalanobisOOD.fit(emb, s.y).calibrate(emb)
    path = tmp_path_factory.mktemp("artifacts")
    ModelBundle(train_svm(s.X, s.y, s.patient), naive, robust, qc).save(path)
    return path


@pytest.fixture(scope="module")
def web_dist(tmp_path_factory):
    path = tmp_path_factory.mktemp("dist")
    (path / "index.html").write_text("<html>RamanTwin</html>")
    return path


@pytest.fixture(scope="module")
def client(artifacts, web_dist):
    with TestClient(create_app(artifacts, web_dist, warmup=False)) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_cors_allows_vite_dev_server(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_serves_web_dist(client):
    assert "RamanTwin" in client.get("/").text


@pytest.fixture(scope="module")
def twin_client(artifacts):
    pytest.importorskip("ramantwin.pipeline")
    with TestClient(create_app(artifacts, warmup=True)) as c:
        yield c


def test_config_lists_models_and_ranges(twin_client):
    cfg = twin_client.get("/api/config").json()
    assert cfg["models"] == ["svm", "naive", "robust"]
    assert cfg["grid"] == {"start": 600.0, "stop": 1800.0, "step": 1.0}
    assert "laser_drift_nm" in cfg["ranges"]


def test_simulate_contract(twin_client):
    r = twin_client.post("/api/simulate", json={"params": {}, "label": 1, "seed": 4})
    assert r.status_code == 200
    body = r.json()
    validate_simulate_response(body)
    assert len(body["processed"]["x"]) == 601
    assert body["truth"]["label"] == 1
    assert list(body["truth"]["concentrations"])[0] == "protein"


def test_only_hardware_changes_recalibrate(twin_client):
    sim = twin_client.app.state.simulator

    def post(params):
        r = twin_client.post("/api/simulate", json={"params": params, "label": 0, "seed": 1})
        assert r.status_code == 200
        return len(sim._calibrations)

    n = post({})
    measurement = {"laser_drift_nm": 0.8, "cosmic_rate": 0.9, "exposure_s": 2.0}
    assert post(measurement) == n
    assert post({"fluorescence_ratio": 30.0, "ambient_light": 0.5}) == n
    assert post({"x_offset_cm1": 7.0}) == n + 1
    assert post({"x_offset_cm1": 7.0, "laser_drift_nm": -0.5}) == n + 1


def test_simulate_matches_fleet_rendering(twin_client):
    import numpy as np

    from ramantwin.api import sample_tissue
    from ramantwin.biochem import component_matrix
    from ramantwin.instrument import VirtualInstrument, nominal
    from ramantwin.ml.acquire import render

    _, conc = sample_tissue(1, 3)
    inst = VirtualInstrument(nominal())
    np.testing.assert_allclose(render(inst, conc), conc @ inst.line_shape(component_matrix))


@pytest.mark.parametrize(
    "body",
    [
        {"params": {"warp_drive": 1}, "label": 0},
        {"params": {"laser_drift_nm": "far"}, "label": 0},
        {"params": {}, "label": 2},
        {"params": {}, "label": 1, "seed": -1},
    ],
)
def test_simulate_rejects_bad_requests(twin_client, body):
    assert twin_client.post("/api/simulate", json=body).status_code == 422


def test_simulate_rejects_non_finite_values(twin_client):
    body = '{"params": {"laser_drift_nm": NaN}, "label": 1, "seed": 0}'
    r = twin_client.post(
        "/api/simulate", content=body, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_simulate_clips_out_of_range_values(twin_client):
    params = {"laser_drift_nm": 99.0, "n_accumulations": 1000, "exposure_s": 0.0}
    r = twin_client.post("/api/simulate", json={"params": params, "label": 1, "seed": 0})
    assert r.status_code == 200
