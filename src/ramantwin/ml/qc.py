"""Quality control: Mahalanobis out-of-distribution score on CNN embeddings and physics checks.

The OOD detector follows Lee et al. (NeurIPS 2018): class-conditional Gaussians with a shared
covariance fitted on training embeddings. Thresholds are percentiles of held-out in-domain scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from sklearn.covariance import LedoitWolf

Flag = Literal["ok", "warn", "reject"]
_SEVERITY = {"ok": 0, "warn": 1, "reject": 2}


@dataclass
class QCResult:
    flag: Flag
    ood_score: float
    threshold: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "flag": self.flag,
            "ood_score": round(self.ood_score, 3),
            "threshold": round(self.threshold, 3),
            "reasons": self.reasons,
        }


@dataclass
class MahalanobisOOD:
    means: np.ndarray  # (K, D)
    precision: np.ndarray  # (D, D)
    warn: float = np.inf
    reject: float = np.inf

    @classmethod
    def fit(cls, emb: np.ndarray, y: np.ndarray) -> MahalanobisOOD:
        classes = np.unique(y)
        means = np.stack([emb[y == c].mean(axis=0) for c in classes])
        centred = emb - means[np.searchsorted(classes, y)]
        precision = LedoitWolf(assume_centered=True).fit(centred).precision_
        return cls(means, precision)

    def score(self, emb: np.ndarray) -> np.ndarray:
        """Distance to the closest class mean, in units of the shared standard deviation."""
        d = emb[:, None, :] - self.means[None, :, :]
        m2 = np.einsum("nkd,de,nke->nk", d, self.precision, d)
        return np.sqrt(np.maximum(m2.min(axis=1), 0.0))

    def calibrate(
        self, emb_heldout: np.ndarray, warn_pct: float = 95.0, reject_pct: float = 99.5
    ) -> MahalanobisOOD:
        s = self.score(emb_heldout)
        self.warn, self.reject = (float(v) for v in np.percentile(s, [warn_pct, reject_pct]))
        return self

    def save(self, path: Path) -> None:
        np.savez(
            path, means=self.means, precision=self.precision, warn=self.warn, reject=self.reject
        )

    @classmethod
    def load(cls, path: Path) -> MahalanobisOOD:
        z = np.load(path)
        return cls(z["means"], z["precision"], float(z["warn"]), float(z["reject"]))


def _noise_sigma(signal: np.ndarray) -> float:
    """White-noise level from first differences (robust MAD estimator)."""
    d = np.diff(signal)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) + 1e-12


def physics_checks(
    processed,
    resolution_cm1: float,
    *,
    min_snr: float = 10.0,
    line_width_ratio: float = 1.0,
    line_sigma: float = 8.0,
    edge_cm1: float = 30.0,
) -> list[tuple[Flag, str]]:
    """Model-free checks on a `ramantwin.pipeline.Processed` (or any object with the same fields).

    Uses `max_count_fraction`, `corrected` and `raman` on the calibrated `xaxis`. Tissue Raman
    bands are the instrument line shape convolved with their own width, so lines narrower than
    the instrument resolution are atomic emission lines from room light leaking into the probe.
    """
    issues: list[tuple[Flag, str]] = []
    if processed.max_count_fraction >= 0.98:
        issues.append(
            ("reject", f"detector saturated ({processed.max_count_fraction:.0%} of full well)")
        )

    r = np.asarray(processed.raman, dtype=float)
    # Noise is measured on `corrected`: the baseline is smooth, and `raman` can be exactly zero
    # wherever the baseline touches the signal.
    c = np.asarray(processed.corrected, dtype=float)
    x = np.asarray(processed.xaxis, dtype=float)
    sigma = _noise_sigma(c)
    snr = float((np.percentile(r, 99.5) - np.median(r)) / sigma)
    if snr < min_snr:
        issues.append(("warn", f"low signal-to-noise ratio ({snr:.1f} < {min_snr:.0f})"))

    # Shot noise follows the fluorescence background, so line prominence is judged against the
    # local noise level; single-pixel spikes are noise, not emission lines.
    local = 1.4826 * median_filter(np.abs(np.diff(c, prepend=c[0])), size=51, mode="nearest")
    local = local / np.sqrt(2) + 1e-12
    peaks, props = find_peaks(r, prominence=line_sigma * sigma, width=1.5)
    widths_cm1 = props["widths"] * np.abs(np.gradient(x))[peaks]
    # Bands cut by the 450 cm-1 truncation or the end of the sensor look artificially narrow.
    inside = (x[peaks] > x.min() + edge_cm1) & (x[peaks] < x.max() - edge_cm1)
    strong = props["prominences"] > line_sigma * local[peaks]
    narrow = peaks[inside & strong & (widths_cm1 < line_width_ratio * resolution_cm1)]
    if len(narrow):
        where = ", ".join(f"{x[i]:.0f}" for i in narrow[:3])
        issues.append(("reject", f"sharp lines narrower than the instrument ({where} cm-1)"))
    return issues


def assess(
    ood: MahalanobisOOD, embedding: np.ndarray, issues: list[tuple[Flag, str]] = ()
) -> QCResult:
    """Combine the OOD score of one embedding (D,) with physics issues; worst flag wins."""
    score = float(ood.score(embedding[None, :])[0])
    flags: list[Flag] = [flag for flag, _ in issues]
    reasons = [reason for _, reason in issues]
    if score > ood.reject:
        flags.append("reject")
    elif score > ood.warn:
        flags.append("warn")
    if score > ood.warn:
        reasons.append(f"spectrum unlike training data (score {score:.1f} > {ood.warn:.1f})")
    flag = max(flags, key=_SEVERITY.__getitem__, default="ok")
    return QCResult(flag, score, ood.reject, reasons)


def abstain(scores: np.ndarray, ood: MahalanobisOOD) -> np.ndarray:
    """Selective prediction: abstain when the OOD score is above the reject threshold."""
    return scores > ood.reject
