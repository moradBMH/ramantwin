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
    ModelBundle(train_svm(s.X, s.y), naive, robust, qc).save(path)
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


def test_simulate_contract_and_drift_does_not_recalibrate(twin_client):
    r = twin_client.post("/api/simulate", json={"params": {}, "label": 1, "seed": 4})
    assert r.status_code == 200
    body = r.json()
    validate_simulate_response(body)
    assert len(body["processed"]["x"]) == 601
    assert body["truth"]["label"] == 1
    drifted = twin_client.post(
        "/api/simulate", json={"params": {"laser_drift_nm": 0.5}, "label": 1, "seed": 4}
    ).json()
    assert drifted["timing_ms"]["calibration"] < 1.0


def test_simulate_rejects_unknown_params(twin_client):
    r = twin_client.post("/api/simulate", json={"params": {"warp_drive": 1}, "label": 0})
    assert r.status_code == 422
