"""On-the-fly spectral augmentations in processed (SNV) space.

Each transform mimics a residual instrument effect that ORPL does not fully remove: x-axis error
(laser drift), resolution mismatch, residual instrument response, residual baseline and noise.
The batch is re-normalised with SNV at the end so the model always sees SNV spectra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ramantwin.constants import GRID_STEP


def snv(x: torch.Tensor) -> torch.Tensor:
    """Standard normal variate along the last axis."""
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True)
    return (x - mu) / (sd + 1e-8)


def shift(x: torch.Tensor, shift_cm1: torch.Tensor) -> torch.Tensor:
    """Move every spectrum by `shift_cm1` (B,) towards higher wavenumbers, linear interpolation."""
    n = x.shape[-1]
    pos = torch.arange(n, device=x.device, dtype=x.dtype) - (shift_cm1 / GRID_STEP)[:, None]
    pos = pos.clamp(0, n - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    w = pos - lo
    return torch.gather(x, 1, lo) * (1 - w) + torch.gather(x, 1, hi) * w


def broaden(x: torch.Tensor, fwhm_cm1: torch.Tensor, half_width: int = 12) -> torch.Tensor:
    """Convolve each spectrum with a Gaussian of its own FWHM (B,); FWHM 0 leaves it unchanged."""
    b = x.shape[0]
    t = torch.arange(-half_width, half_width + 1, device=x.device, dtype=x.dtype) * GRID_STEP
    sigma = (fwhm_cm1 / (2 * math.sqrt(2 * math.log(2)))).clamp(min=1e-3)[:, None]
    k = torch.exp(-0.5 * (t[None, :] / sigma) ** 2)
    k = (k / k.sum(dim=1, keepdim=True)).unsqueeze(1)
    padded = F.pad(x.unsqueeze(0), (half_width, half_width), mode="replicate")
    return F.conv1d(padded, k, groups=b).squeeze(0)


def _rand(like: torch.Tensor, *shape: int) -> torch.Tensor:
    return torch.rand(*shape, device=like.device, dtype=like.dtype)


def _legendre_basis(n: int, order: int, device, dtype) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    basis = [torch.ones_like(t), t]
    for k in range(1, order):
        basis.append(((2 * k + 1) * t * basis[k] - k * basis[k - 1]) / (k + 1))
    return torch.stack(basis[: order + 1])


@dataclass
class AugmentParams:
    """Augmentation strengths. Every transform is drawn independently per spectrum."""

    max_shift_cm1: float = 6.0
    max_broaden_cm1: float = 6.0
    p_broaden: float = 0.5
    gain_tilt: float = 0.15
    baseline_amp: float = 0.25
    baseline_order: int = 3
    max_noise: float = 0.05

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        x = shift(x, (2 * _rand(x, b) - 1) * self.max_shift_cm1)
        fwhm = _rand(x, b) * self.max_broaden_cm1 * (_rand(x, b) < self.p_broaden)
        x = broaden(x, fwhm)

        basis = _legendre_basis(n, self.baseline_order, x.device, x.dtype)
        # Residual instrument response: smooth multiplicative gain on the positive spectrum.
        gain = 1 + self.gain_tilt * ((2 * _rand(x, b, 3) - 1) @ basis[:3])
        x = snv((x - x.min(dim=1, keepdim=True).values) * gain.clamp(min=0.2))
        coefs = (2 * _rand(x, b, self.baseline_order + 1) - 1) * self.baseline_amp
        x = x + coefs @ basis
        x = x + torch.randn_like(x) * (_rand(x, b, 1) * self.max_noise)
        return snv(x)
