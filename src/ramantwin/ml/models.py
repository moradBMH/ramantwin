"""Spectral classifiers: a 1D CNN and the linear SVM baseline of the ROWS study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn

from ramantwin.constants import GRID
from ramantwin.ml.qc import MahalanobisOOD


def _block(c_in: int, c_out: int, kernel: int, pool: bool, stride: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.GELU(),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class SpectralCNN(nn.Module):
    """1D CNN for SNV-normalised spectra on `GRID`.

    A fixed position channel is stacked with the spectrum so that the network knows where each band
    sits after global average pooling: band ratios at given wavenumbers carry the diagnosis.
    `forward` returns temperature-scaled logits; the temperature is fitted after training.
    """

    def __init__(
        self,
        n_points: int = len(GRID),
        embed_dim: int = 64,
        width: int = 32,
        n_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {
            "n_points": n_points,
            "embed_dim": embed_dim,
            "width": width,
            "n_classes": n_classes,
            "dropout": dropout,
        }
        self.register_buffer("position", torch.linspace(-1.0, 1.0, n_points).view(1, 1, -1))
        self.register_buffer("temperature", torch.ones(()))
        self.features = nn.Sequential(
            _block(2, width // 2, 9, pool=True, stride=2),
            _block(width // 2, width, 7, pool=True),
            _block(width, 2 * width, 5, pool=True),
            _block(2 * width, 2 * width, 5, pool=True),
            _block(2 * width, 4 * width, 3, pool=False),
        )
        self.project = nn.Sequential(nn.Linear(4 * width, embed_dim), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, n_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_points) spectra -> (B, embed_dim) embeddings."""
        h = torch.cat([x.unsqueeze(1), self.position.expand(x.shape[0], -1, -1)], dim=1)
        h = self.features(h).mean(dim=-1)
        return self.project(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)) / self.temperature

    def save(self, path: Path) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> SpectralCNN:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()


@torch.no_grad()
def cnn_outputs(
    model: SpectralCNN, X: np.ndarray, batch_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_cancer (N,), embeddings (N, D)) of an eval-mode CNN on CPU arrays."""
    model.eval()
    device = next(model.parameters()).device
    probs, embs = [], []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[start : start + batch_size], dtype=np.float32))
        z = model.embed(xb.to(device))
        logits = model.head(z) / model.temperature
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        embs.append(z.cpu().numpy())
    return np.concatenate(probs), np.concatenate(embs)


def make_svm(C: float = 0.01) -> Pipeline:
    """Linear SVM with Platt-scaled probabilities on ORPL-processed spectra (ROWS approach)."""
    svm = LinearSVC(C=C, dual="auto", max_iter=20_000)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svm", CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False)),
        ]
    )


@dataclass
class ModelBundle:
    """The three trained models and the QC detector, as saved by `ramantwin.ml.train`."""

    svm: Pipeline
    naive: SpectralCNN
    robust: SpectralCNN
    qc: MahalanobisOOD

    def save(self, artifacts: Path) -> None:
        artifacts.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.svm, artifacts / "svm.joblib")
        self.naive.save(artifacts / "naive.pt")
        self.robust.save(artifacts / "robust.pt")
        self.qc.save(artifacts / "qc.npz")

    @classmethod
    def load(cls, artifacts: Path) -> ModelBundle:
        return cls(
            joblib.load(artifacts / "svm.joblib"),
            SpectralCNN.load(artifacts / "naive.pt"),
            SpectralCNN.load(artifacts / "robust.pt"),
            MahalanobisOOD.load(artifacts / "qc.npz"),
        )

    def predictors(self) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
        """p_cancer functions of the three models, keyed by model name."""
        return {
            "svm": lambda X: self.svm.predict_proba(X)[:, 1],
            "naive": lambda X: cnn_outputs(self.naive, X)[0],
            "robust": lambda X: cnn_outputs(self.robust, X)[0],
        }

    def outputs(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Probabilities of every model, CNN embeddings and the OOD score of the robust model."""
        p_naive, emb_naive = cnn_outputs(self.naive, X)
        p_robust, emb_robust = cnn_outputs(self.robust, X)
        return {
            "svm": self.svm.predict_proba(X)[:, 1],
            "naive": p_naive,
            "robust": p_robust,
            "emb_naive": emb_naive,
            "emb_robust": emb_robust,
            "ood": self.qc.score(emb_robust),
        }
