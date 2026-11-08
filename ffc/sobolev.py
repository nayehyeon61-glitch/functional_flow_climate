"""Sobolev norms on the sphere through the spherical-harmonic transform.

For f on S^2 with coefficients f_lm (orthonormal basis), the Laplace-Beltrami operator is diagonal,
-Delta Y_lm = l (l + 1) Y_lm, so every Sobolev norm is a weighted sum of the coefficient energies:

    ||f||_{H^s}^2 = sum_{l, m} (1 + l (l + 1))^s |f_lm|^2        (s = 0: L2, s = 1: H1, s = -1: H^-1)

``s = -1`` is the norm of the tangent space of the Wasserstein-2 geometry (Otto calculus: the W2 metric
on densities is the H^-1 norm of the tangent vector, weighted by the density), which is the structure the
pg-flow flow matching is built on; ``s = 1`` asks the rollout to match gradients (fronts, jets) and is what
neuraloperator's ``H1Loss`` does with finite differences on a flat grid. On a latitude-longitude grid of the
sphere this spectral form replaces the flat finite differences, and it is exactly resolution-invariant,
which is the function-space property the operator should be trained with.

The transform is torch-harmonics' ``RealSHT`` (the same one neuraloperator's spherical convolution uses),
so the loss is evaluated in the operator's own basis.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_harmonics import RealSHT
except ImportError as e:  # pragma: no cover
    RealSHT = None
    _IMPORT_ERROR = e


class SphericalSobolev(nn.Module):
    """``norm2(f)``: ||f||_{H^s}^2 of fields (..., H, W) on an equiangular lat-lon grid, by rows of f.

    ``grid``: torch-harmonics quadrature of the latitude nodes. WeatherBench2's 64x32 / 128x64 grids are
    cell centres without the poles ("equiangular" nodes shifted by half a cell); the quadrature then is an
    approximation of the same order as the operator's own transform, and ``parseval_error`` reports how far
    sum |f_lm|^2 is from the area mean of f^2 on this grid (a few percent at 64x32).
    """

    def __init__(self, nlat: int, nlon: int, s: float = 1.0, lmax: Optional[int] = None, grid: str = "equiangular",
                 normalize: bool = True):
        super().__init__()
        if RealSHT is None:  # pragma: no cover
            raise ImportError(f"torch-harmonics is needed for the Sobolev loss: {_IMPORT_ERROR}")
        self.s = float(s)
        lmax = lmax or nlat
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid=grid, norm="ortho")
        l = torch.arange(lmax, dtype=torch.float64)
        w = (1.0 + l * (l + 1.0)) ** self.s
        if normalize:
            # a white spectrum up to lmax has the same H^s and L2 norm: lambda_sob = 1 puts the two terms on
            # the same footing, and the weight only redistributes the error over the degrees l
            w = w / float((w * (2 * l + 1)).sum() / (2 * l + 1).sum())
        self.register_buffer("weight", w)  # (L,)
        # m > 0 coefficients of a real field stand for +-m: counted twice in the energy
        mult = torch.full((lmax,), 2.0, dtype=torch.float64)
        mult[0] = 1.0
        self.register_buffer("mult", mult)  # (M,)

    def coefficients(self, f: torch.Tensor) -> torch.Tensor:
        """(..., H, W) -> (..., L, M) complex spherical-harmonic coefficients (orthonormal basis)."""
        return self.sht(f.to(torch.float32))

    def energy_by_degree(self, f: torch.Tensor) -> torch.Tensor:
        """(..., L): sum_m |f_lm|^2 (orthonormal basis: the integral over the sphere), the power spectrum by
        total wavenumber l."""
        c = self.coefficients(f)
        return ((c.real ** 2 + c.imag ** 2) * self.mult.to(c.real.dtype)).sum(-1)

    def norm2(self, f: torch.Tensor) -> torch.Tensor:
        """(..., H, W) -> (...): sum_l (1 + l(l+1))^s sum_m |f_lm|^2, normalized so that s = 0 is the area
        mean of f^2 over the sphere (the orthonormal basis gives the integral / (4 pi))."""
        e = self.energy_by_degree(f)
        return (e * self.weight.to(e.dtype)).sum(-1) / (4.0 * math.pi)

    def loss(self, a: torch.Tensor, b: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean over the leading dims of ||(a - b) / scale||_{H^s}^2; ``scale`` broadcasts against (..., C, H, W)
        (the 6 h tendency std per channel, so every channel counts alike as in the L2 term)."""
        d = a - b
        if scale is not None:
            d = d / scale
        return self.norm2(d).mean()

    @torch.no_grad()
    def parseval_error(self, f: torch.Tensor, lat_deg) -> float:
        """Relative difference between sum |f_lm|^2 and the cos(lat)-weighted mean of f^2 on this grid."""
        w = torch.cos(torch.deg2rad(torch.as_tensor(lat_deg, dtype=f.dtype, device=f.device)))
        w = (w / w.mean()).view(-1, 1)
        l2 = (f ** 2 * w).mean((-2, -1))
        c = self.energy_by_degree(f).sum(-1) / (4.0 * math.pi)
        return float(((c - l2).abs() / l2.clamp_min(1e-30)).mean())


def sobolev_scale_ratio(s: float, lmax: int) -> float:
    """Rough size of ||f||_{H^s} / ||f||_{L2} for a white spectrum up to lmax, to set the loss weight."""
    l = torch.arange(lmax, dtype=torch.float64)
    return math.sqrt(float(((1.0 + l * (l + 1.0)) ** s * (2 * l + 1)).sum() / (2 * l + 1).sum()))
