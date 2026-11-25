import pytest

from ramantwin.ml.models import ModelBundle, cnn_outputs
from ramantwin.ml.qc import MahalanobisOOD
from ramantwin.ml.synthetic import Synthetic, make_spectra
from ramantwin.ml.train import train_cnn, train_svm


@pytest.fixture(scope="session")
def synthetic() -> Synthetic:
    return make_spectra(n_patients=30, spectra_per_patient=4, n_instruments=2, seed=0)


@pytest.fixture(scope="session")
def tiny_bundle(synthetic) -> ModelBundle:
    """Small models trained for a few epochs on synthetic spectra."""
    s = synthetic
    svm = train_svm(s.X, s.y, s.patient)
    naive = train_cnn(s.X, s.y, s.patient, epochs=3, width=8, seed=0).model
    robust = train_cnn(s.X, s.y, s.patient, epochs=3, width=8, seed=1).model
    _, emb = cnn_outputs(robust, s.X)
    qc = MahalanobisOOD.fit(emb, s.y).calibrate(emb)
    return ModelBundle(svm, naive, robust, qc)
