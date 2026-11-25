import torch

from ramantwin.ml.augment import AugmentParams, broaden, shift, snv


def peak(center: int = 600, n: int = 4) -> torch.Tensor:
    x = torch.arange(1201, dtype=torch.float32)
    return snv(torch.exp(-0.5 * ((x - center) / 4.0) ** 2).repeat(n, 1))


def test_augment_keeps_snv():
    x = peak(n=16)
    torch.manual_seed(0)
    out = AugmentParams()(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.mean(dim=1), torch.zeros(16), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.std(dim=1), torch.ones(16), atol=1e-4, rtol=0)


def test_shift_moves_peak():
    x = peak()
    out = shift(x, torch.tensor([5.0, -5.0, 0.0, 12.0]))
    assert out.argmax(dim=1).tolist() == [605, 595, 600, 612]


def test_subpixel_shift_moves_centroid():
    x = peak(n=1) - peak(n=1).min()
    out = shift(x, torch.tensor([0.5]))
    grid = torch.arange(1201, dtype=torch.float32)
    centroid = (out * grid).sum() / out.sum()
    assert abs(float(centroid) - 600.5) < 0.05


def test_broaden_lowers_peak_and_keeps_area():
    x = peak() - peak().min()
    out = broaden(x, torch.tensor([0.0, 4.0, 8.0, 12.0]))
    heights = out.max(dim=1).values
    assert torch.all(heights[1:] < heights[:-1])
    torch.testing.assert_close(out.sum(dim=1), x.sum(dim=1), rtol=1e-3, atol=0)
