import numpy as np
import torch

from ramantwin.constants import GRID
from ramantwin.ml.models import ModelBundle, SpectralCNN, cnn_outputs, make_svm


def test_cnn_shapes():
    model = SpectralCNN(width=8, embed_dim=16).eval()
    x = torch.randn(5, len(GRID))
    assert model(x).shape == (5, 2)
    assert model.embed(x).shape == (5, 16)


def test_temperature_scales_logits():
    model = SpectralCNN(width=8).eval()
    x = torch.randn(3, len(GRID))
    logits = model(x)
    model.temperature.fill_(2.0)
    torch.testing.assert_close(model(x), logits / 2)


def test_cnn_save_load_roundtrip(tmp_path):
    model = SpectralCNN(width=8).eval()
    model.temperature.fill_(1.7)
    model.save(tmp_path / "m.pt")
    loaded = SpectralCNN.load(tmp_path / "m.pt")
    x = np.random.default_rng(0).normal(size=(4, len(GRID))).astype(np.float32)
    np.testing.assert_allclose(cnn_outputs(loaded, x)[0], cnn_outputs(model, x)[0], atol=1e-6)


def test_svm_probabilities(synthetic):
    svm = make_svm().fit(synthetic.X, synthetic.y)
    p = svm.predict_proba(synthetic.X)
    assert p.shape == (len(synthetic.y), 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_bundle_save_load(tmp_path, tiny_bundle, synthetic):
    tiny_bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    a, b = tiny_bundle.outputs(synthetic.X[:10]), loaded.outputs(synthetic.X[:10])
    for k in ("svm", "naive", "robust", "ood"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-5, atol=1e-6)
