"""Step 2: a flow-matching ensemble generator grafted in front of the frozen Step 1 dynamics field.

    x0 (ERA5) --FM generator g_phi(eps | x0)--> scenarios x0^(m) = x0 + kappa * sigma_e * z^(m)
              --frozen SFNO dynamics field G_theta (torchdiffeq, Step 1)--> ensemble forecast

Two-stage training. Stage 1 (Step 1) trains G_theta on dynamics and physics. Stage 2 (this module) keeps
G_theta fixed and learns, with conditional flow matching, the distribution of what the frozen operator gets
wrong from a given state:

    e = x_{6h}^{ERA5} - Phi_theta(x0),     z = e / sigma_e  (sigma_e: per-channel RMS of e over the training years)
    learn p(z | x0, Phi_theta(x0) - x0)

so the scenarios lie along the flow-dependent directions in which the operator's own dynamics is uncertain
(instead of the random 6 h analysis increments of the Step 1 stand-in, ``ffc.ensemble.perturbed_members``).
``kappa`` turns one-step error samples into initial-condition scenarios and is fitted on the validation year
(``scripts/calibrate_ensemble.py``). ``stochastic`` mode also adds a fresh error sample after every 6 h step
(a stochastic dynamics x_{k+1} = Phi(x_k) + kappa_s sigma_e z_k).

The W2 / Sobolev structure of the flow (functional flow matching, Kerrigan et al. 2023, on the sphere):

* source: a Gaussian random field on S^2 instead of white noise per grid point. Its spectrum is either the
  power law (1 + l(l+1))^{-s} (an H^s-regular field) or the per-channel degree spectrum of the targets
  ("matched"), drawn in the spherical-harmonic basis with torch-harmonics' inverse SHT, so the source is a
  function-space measure that does not depend on the grid;
* coupling: minibatch optimal transport (torchcfm's ``OTPlanSampler``, exact EMD) with the squared H^s
  distance between source and target fields as the cost, so the straight paths z_tau = (1 - tau) eps + tau z
  are the W2 displacement interpolation between the two empirical measures of the batch (``indep`` = the
  plain CFM coupling of pg-flow step 2);
* loss: the velocity error in the spherical Sobolev norm of ``ffc.sobolev`` (s = 0: area-weighted L2,
  s = -1: the W2 tangent norm);
* velocity field: neuraloperator's SFNO, conditioned on x0, on the frozen operator's increment and on tau.

Reused: torchcfm (``ConditionalFlowMatcher`` path, ``OTPlanSampler``), pg-flow ``pgcfm.cfm.heun`` (the Heun
sampler of pg-flow step 2) and ``TISR_DYN``, neuraloperator ``SFNO`` (via ``ffc.field.make_backbone``),
torch-harmonics, ``ffc.sobolev.SphericalSobolev``.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from . import deps  # noqa: F401
from .field import N_TIME, FieldConfig, make_backbone
from .sobolev import SphericalSobolev
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN, heun  # noqa: E402
from pgcfm.core import time_features  # noqa: E402

N_DYN = contract.N_DYNAMIC
N_TAU = 8  # sin/cos of tau at 4 frequencies


@dataclass
class FlowConfig:
    coupling: str = "ot"         # ot | indep
    source: str = "matched"      # matched | sobolev | white
    source_s: float = 1.0        # sobolev source: spectrum (1 + l(l+1))^-s
    loss_s: float = 0.0          # Sobolev order of the velocity loss (0 = area-weighted L2, -1 = W2 tangent norm)
    ot_s: float = 0.0            # Sobolev order of the OT cost
    width: int = 64
    n_layers: int = 4
    modes_lat: int = 16
    mlp_expansion: float = 0.5
    sample_steps: int = 4        # Heun steps when sampling (NFE = 2 x steps)


def tau_features(tau: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B,) in [0, 1] -> (B, 8, H, W) constant maps."""
    k = torch.arange(1, N_TAU // 2 + 1, device=tau.device, dtype=tau.dtype)
    a = math.pi * tau[:, None] * k[None]
    f = torch.cat([torch.sin(a), torch.cos(a)], 1)
    return f[:, :, None, None].expand(-1, -1, H, W)


class SphericalGRF(nn.Module):
    """Gaussian random field on the sphere, per channel, drawn in the spherical-harmonic basis.

    ``spectrum`` (C, L): variance of every coefficient of degree l (so the expected energy of degree l is
    (2l + 1) x spectrum). Each channel is scaled to unit point variance (area mean), so the field has the
    same amplitude as white N(0, 1) noise and only its spatial structure differs."""

    def __init__(self, nlat: int, nlon: int, spectrum: torch.Tensor, grid: str = "equiangular"):
        super().__init__()
        from torch_harmonics import InverseRealSHT
        C, L = spectrum.shape
        self.isht = InverseRealSHT(nlat, nlon, lmax=L, mmax=L, grid=grid, norm="ortho")
        # orthonormal basis: area mean of f^2 = sum_lm |f_lm|^2 / (4 pi); E|f_lm|^2 = spectrum_l (complex: m > 0
        # coefficients carry the +-m pair)
        l = torch.arange(L, dtype=torch.float64)
        point_var = (spectrum.double() * (2 * l + 1)).sum(-1) / (4 * math.pi)
        amp = (spectrum.double() / point_var[:, None]).sqrt()
        tri = (torch.arange(L)[:, None] >= torch.arange(L)[None]).double()  # m <= l
        self.register_buffer("amp", (amp[:, :, None] * tri[None]).float())  # (C, L, M)
        self.C = C

    def forward(self, B: int, generator: Optional[torch.Generator] = None, device=None) -> torch.Tensor:
        C, L, M = self.amp.shape
        dev = device or self.amp.device
        re = torch.randn((B, C, L, M), generator=generator, device=dev)
        im = torch.randn((B, C, L, M), generator=generator, device=dev)
        # m = 0 coefficients are real with variance spectrum_l; m > 0 complex with E|c|^2 = spectrum_l / 2 each
        # side (the real field counts them twice)
        im[..., 0] = 0.0
        scale = torch.full((M,), math.sqrt(0.5), device=dev)
        scale[0] = 1.0
        c = torch.complex(re, im) * (self.amp.to(dev) * scale)
        return self.isht(c)

    @staticmethod
    def power_law(C: int, L: int, s: float) -> torch.Tensor:
        l = torch.arange(L, dtype=torch.float64)
        return ((1.0 + l * (l + 1.0)) ** (-s))[None].expand(C, L).float()


class ErrorFlow(nn.Module):
    """Conditional flow p(z | x0): z = (x_6h - Phi_theta(x0)) / sigma_e, the frozen operator's normalized 6 h error.

    ``static`` (S, H, W) position / orography features, ``sigma_e`` (66,) error RMS per channel (normalized KAI
    units; tisr's is 0), ``spectrum`` (66, L) of the targets for the ``matched`` source."""

    def __init__(self, cfg: FlowConfig, static: torch.Tensor, sigma_e: torch.Tensor, spectrum: Optional[torch.Tensor],
                 lat_deg, n_lon: int):
        super().__init__()
        self.cfg = cfg
        H, W = static.shape[-2:]
        self.register_buffer("static", static)
        self.register_buffer("sigma_e", sigma_e.view(1, N_DYN, 1, 1).clone())
        keep = torch.ones(N_DYN)
        keep[TISR_DYN] = 0.0
        self.register_buffer("keep", keep.view(1, N_DYN, 1, 1))
        lmax = min(cfg.modes_lat * 2, H)
        if cfg.source == "matched":
            if spectrum is None:
                raise ValueError("matched source needs the target spectrum")
            spec = spectrum[:, :lmax].clamp_min(1e-12)
        elif cfg.source == "sobolev":
            spec = SphericalGRF.power_law(N_DYN, lmax, cfg.source_s)
        elif cfg.source == "white":
            spec = None
        else:
            raise ValueError(cfg.source)
        self.grf = SphericalGRF(H, W, spec) if spec is not None else None
        self.loss_norm = SphericalSobolev(H, W, s=cfg.loss_s)
        self.ot_norm = SphericalSobolev(H, W, s=cfg.ot_s)
        fcfg = FieldConfig(backbone="sfno", width=cfg.width, n_layers=cfg.n_layers, modes_lat=cfg.modes_lat,
                           mlp_expansion=cfg.mlp_expansion)
        c_in = N_DYN * 3 + static.shape[0] + N_TIME + N_TAU
        self.net = make_backbone(c_in, N_DYN, fcfg, H, n_lon)
        self.ot = None
        if cfg.coupling == "ot":
            from torchcfm.optimal_transport import OTPlanSampler
            self.ot = OTPlanSampler(method="exact")
        elif cfg.coupling != "indep":
            raise ValueError(cfg.coupling)

    # -- source ------------------------------------------------------------------------------------------
    def source(self, B: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        H, W = self.static.shape[-2:]
        if self.grf is None:
            eps = torch.randn((B, N_DYN, H, W), generator=generator, device=self.static.device)
        else:
            eps = self.grf(B, generator, self.static.device)
        return eps * self.keep

    # -- conditioning ------------------------------------------------------------------------------------
    def condition(self, x0: torch.Tensor, inc: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        """x0 (B, 66, H, W) normalized state; inc (B, 66, H, W) the frozen operator's 6 h increment in t6 units."""
        B, _, H, W = x0.shape
        return torch.cat([x0, inc, self.static.expand(B, -1, -1, -1), time_features(hours, H, W)], 1)

    def velocity(self, tau: torch.Tensor, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        H, W = z.shape[-2:]
        return self.net(torch.cat([z, cond, tau_features(tau, H, W)], 1)) * self.keep

    # -- training ----------------------------------------------------------------------------------------
    def couple(self, eps: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Reorder the source draws by the minibatch OT plan under the squared H^s distance; returns eps permuted
        so that (eps[i], z1[i]) are the coupled pairs."""
        if self.ot is None:
            return eps
        with torch.no_grad():
            a = self._embed(eps)
            b = self._embed(z1)
            pi = self.ot.get_map(a, b)
            # exact EMD between two uniform empirical measures of equal size is a permutation
            j_of_i = torch.as_tensor(pi.argmax(1))
        perm = torch.empty_like(j_of_i)
        perm[j_of_i] = torch.arange(len(j_of_i))
        return eps[perm.to(eps.device)]

    def _embed(self, f: torch.Tensor) -> torch.Tensor:
        """Fields -> vectors whose Euclidean distance is the H^s distance (sqrt-weighted SH coefficients)."""
        c = self.ot_norm.coefficients(f)  # (B, C, L, M)
        w = (self.ot_norm.weight.to(c.real.dtype)[:, None] * self.ot_norm.mult.to(c.real.dtype)[None]).sqrt()
        c = c * w
        return torch.cat([c.real, c.imag], -1).flatten(1)

    def loss(self, z1: torch.Tensor, cond: torch.Tensor, generator: Optional[torch.Generator] = None):
        B = z1.shape[0]
        eps = self.couple(self.source(B, generator), z1)
        tau = torch.rand((B,), generator=generator, device=z1.device)
        t = tau.view(B, 1, 1, 1)
        zt = (1 - t) * eps + t * z1
        u = (z1 - eps) * self.keep
        v = self.velocity(tau, zt, cond)
        return self.loss_norm.loss(v, u)

    # -- sampling ----------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, generator: Optional[torch.Generator] = None,
               steps: Optional[int] = None) -> torch.Tensor:
        """Heun integration (pg-flow ``pgcfm.cfm.heun``) of the learned velocity from the source; returns z (B, 66, H, W)."""
        eps = self.source(cond.shape[0], generator)
        return heun(lambda t, s: self.velocity(t, s, cond), eps, steps or self.cfg.sample_steps) * self.keep

    def error(self, z: torch.Tensor) -> torch.Tensor:
        """Normalized flow sample -> error in normalized KAI units."""
        return z * self.sigma_e

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in [k for k in sd if k.endswith("_metadata")]:
            sd.pop(k)
        return sd

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.cfg)


# -- the grafted pipeline: generator -> frozen operator ------------------------------------------------------

@torch.no_grad()
def operator_increment(field, x0: torch.Tensor, hours: torch.Tensor, tisr_next: Optional[torch.Tensor] = None):
    """Phi_theta(x0) and its increment in t6 units (the flow's condition)."""
    x1 = field.step(x0, hours, tisr_next)
    return x1, (x1 - x0) / field.t6.clamp_min(1e-6)


@torch.no_grad()
def flow_members(flow: ErrorFlow, field, x0: torch.Tensor, hours: torch.Tensor, M: int, kappa: float,
                 generator: Optional[torch.Generator] = None, control: bool = True,
                 tisr_next: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, 66, H, W) -> (B, M, 66, H, W): x0 + kappa sigma_e z^(m), z^(m) ~ p(z | x0); member 0 is x0 itself when
    ``control`` (the same convention as the Step 1 stand-in). ``tisr_next`` (B, H, W), the known forcing 6 h later,
    must be given as in training: the condition's tisr channel is the forcing increment."""
    B = x0.shape[0]
    n_gen = M - 1 if control else M
    _, inc = operator_increment(field, x0, hours, tisr_next)
    cond = flow.condition(x0, inc, hours).repeat_interleave(n_gen, 0)
    z = flow.sample(cond, generator).view(B, n_gen, *x0.shape[1:])
    m = x0[:, None] + kappa * flow.error(z.flatten(0, 1)).view_as(z)
    return torch.cat([x0[:, None], m], 1) if control else m


@torch.no_grad()
def rollout_stochastic(flow: ErrorFlow, field, members: torch.Tensor, hour0: torch.Tensor, steps: int,
                       kappa_s: float, tisr: Optional[torch.Tensor] = None,
                       generator: Optional[torch.Generator] = None, control: bool = True) -> torch.Tensor:
    """Members (B, M, 66, H, W) -> (B, M, steps, 66, H, W) with x_{k+1} = Phi(x_k) + kappa_s sigma_e z_k,
    z_k ~ p(z | x_k) drawn afresh per member and step (member 0 stays deterministic when ``control``)."""
    B, M = members.shape[:2]
    x = members.flatten(0, 1)
    hours = hour0.float().repeat_interleave(M, 0)
    det = torch.zeros(B, M, 1, 1, 1, device=x.device)
    if control:
        det[:, 0] = 1.0
    det = det.flatten(0, 1)
    out = []
    for k in range(steps):
        tk = None if tisr is None else tisr[:, k].repeat_interleave(M, 0)
        x1 = field.step(x, hours, tk)
        inc = (x1 - x) / field.t6.clamp_min(1e-6)
        z = flow.sample(flow.condition(x, inc, hours), generator)
        x = x1 + (1 - det) * kappa_s * flow.error(z)
        if field.pm is not None:
            x = field.pm(x)
        hours = hours + 6.0
        out.append(x)
    return torch.stack(out, 1).view(B, M, steps, *members.shape[2:])
