"""The dynamics field: a neural operator G_theta on the sphere that is the tangent vector dx/dt of the state.

    x(t)      the 66 dynamic KAI channels (z, q, t, u, v on 12 levels + surface), normalized, on a lat-lon grid
    G_theta   x -> dx/dt (normalized units per 6 h), a neural operator: the same weights define the map at any
              resolution of the same function space, and its spectral layers act in the spherical-harmonic
              basis in which the Sobolev norms of ``ffc.sobolev`` are diagonal
    step      x(t + 6 h) = ODE solve of dx/ds = G_theta(x, t + s) over 6 h (torchdiffeq, fixed-grid solver),
              then the known forcing tisr is written in from the data and ``PositiveMass`` (pg-flow step 14)
              restores q, tcwv >= 0 with the global mass of every level kept
    rollout   steps chained to any lead; members of an ensemble are separate rows of the batch, so the
              same field maps every flow-matching scenario

Backbones (``FieldConfig.backbone``):
    sfno    neuraloperator ``SFNO``: spherical convolutions (torch-harmonics SHT), dense weights per (l, m)
    fno     neuraloperator ``FNO`` on the lat-lon grid with latitude padding (a planar Fourier basis)
    resnet  ClimODE's ``Climate_ResNet_2D`` through pg-flow's ``pgcfm.core.Net``: the climate-ODE baseline
            (pg-flow step 12/16 DynCore) with the same solver, loss and data, so the comparison isolates the
            operator

The output head of every backbone is zero-initialized: the field starts as persistence (pg-flow step 12).

Physics core (Step 4, ``FieldConfig.core``): with ``core="dinosaur"`` the field is a hybrid
``dx/dt = F_phys(x) + R_theta(x, F_phys(x), t)``: F_phys is alpha * (6 h increment of a spectral primitive-equation
GCM started from x, ``ffc.physcore.SpectralCore``; the SpeedyWeather.jl role), computed once per 6 h step from
the step's start state and held fixed over the step, no gradient; the operator sees it as an extra input and
learns the residual. alpha (66,) is the per-channel least-squares weight of the core increment against ERA5's
6 h change (``calibrate_core``), clipped to [0, 1], so levels where the core has no skill (50 hPa) are switched
off. With the zero-initialized head the hybrid starts as the core alone.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from . import deps  # noqa: F401
from pgcfm import contract  # noqa: E402

from pgcfm.balance import PositiveMass  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402
from pgcfm.core import time_features  # noqa: E402

N_DYN = contract.N_DYNAMIC
HOURS_PER_STEP = 6.0
N_TIME = 4  # day and season phase features


@dataclass
class FieldConfig:
    backbone: str = "sfno"       # sfno | fno | resnet
    width: int = 64              # hidden channels (sfno / fno) or ResNet hidden
    n_layers: int = 4            # operator layers, or ResNet blocks per stage (resnet: (n_layers, n_layers))
    modes_lat: int = 16          # SFNO: lmax (= mmax); FNO: modes along latitude
    modes_lon: int = 32          # FNO: modes along longitude (SFNO ignores it)
    mlp_expansion: float = 0.5   # neuraloperator channel MLP width ratio
    dt_hours: float = 6.0        # solver step inside a 6 h data step
    solver: str = "midpoint"     # torchdiffeq fixed-grid method: euler | midpoint | rk4 | heun3
    positive: bool = True        # PositiveMass after every 6 h step
    lat_padding: float = 0.125   # fno: neuraloperator domain padding in latitude (fraction of H)
    core: str = ""               # "" (pure operator) | "dinosaur" | "speedy" (physics core + operator residual)
    core_cfg: Optional[dict] = None  # ffc.physcore.CoreConfig / ffc.speedycore.SpeedyConfig fields
    history: Optional[list] = None   # Step 6: lags in 6 h steps of the past states the operator sees, e.g. [2, 4]
    #                                  = x(t - 12 h), x(t - 24 h): the field becomes an operator on the sequence


def _zero_last_layer(module: nn.Module) -> None:
    last = None
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            last = m
    if last is None:
        raise ValueError("no linear layer to zero-initialize")
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def make_backbone(c_in: int, c_out: int, cfg: FieldConfig, n_lat: int, n_lon: int) -> nn.Module:
    if cfg.backbone == "sfno":
        from neuralop.models import SFNO
        # neuraloperator's SphericalConv convention: n_modes = (lmax, 2 mmax) with mmax = lmax (|m| <= l)
        lmax = min(cfg.modes_lat, n_lat)
        net = SFNO(n_modes=(lmax, 2 * lmax), in_channels=c_in, out_channels=c_out, hidden_channels=cfg.width,
                   n_layers=cfg.n_layers, positional_embedding=None, channel_mlp_expansion=cfg.mlp_expansion)
    elif cfg.backbone == "fno":
        from neuralop.models import FNO
        pad = [cfg.lat_padding, 0.0] if cfg.lat_padding > 0 else None
        net = FNO(n_modes=(min(cfg.modes_lat, n_lat), min(cfg.modes_lon, n_lon // 2 + 1)), in_channels=c_in,
                  out_channels=c_out, hidden_channels=cfg.width, n_layers=cfg.n_layers, positional_embedding=None,
                  channel_mlp_expansion=cfg.mlp_expansion, domain_padding=pad)
    elif cfg.backbone == "resnet":
        from pgcfm.core import Net
        from pgcfm.seqdct import _NetCfg
        net = Net(c_in, c_out, _NetCfg(cfg.width, (cfg.n_layers, cfg.n_layers)))
    else:
        raise ValueError(cfg.backbone)
    _zero_last_layer(net)
    return net


class DynamicsField(nn.Module):
    """x0 (B, 66, H, W) normalized, hour0 (B,) hours since Jan 1 00 UTC -> states every 6 h.

    ``static`` (S, H, W) position + orography features (``ffc.data.static_features``), ``t6`` (66,) the training
    6 h tendency std per channel (the head's output unit), ``mean`` / ``std`` the 67-channel KAI statistics
    (PositiveMass), ``lat_deg`` the grid latitudes."""

    def __init__(self, cfg: FieldConfig, static: torch.Tensor, t6: torch.Tensor, mean, std, lat_deg, n_lon: int,
                 orog_si=None):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("static", static)
        self.register_buffer("t6", t6.view(1, N_DYN, 1, 1).clone())
        keep = torch.ones(N_DYN)
        keep[TISR_DYN] = 0.0  # tisr is a known forcing, never predicted
        self.register_buffer("keep", keep.view(1, N_DYN, 1, 1))
        self.core = None
        if cfg.core:
            if cfg.core not in ("dinosaur", "speedy"):
                raise ValueError(f"unknown physics core {cfg.core!r}")
            if orog_si is None:
                raise ValueError("a physics core needs the surface geopotential orog_si (H, W)")
            if cfg.core == "dinosaur":
                from .physcore import CoreConfig, SpectralCore
                self.core = SpectralCore(CoreConfig(**(cfg.core_cfg or {})), mean, std, lat_deg, orog_si)
            else:   # Step 5: SpeedyWeather.jl PrimitiveWetModel (with its physics) in a Julia worker process
                from .speedycore import SpeedyConfig, SpeedyCore
                self.core = SpeedyCore(SpeedyConfig(**(cfg.core_cfg or {})), mean, std, lat_deg, orog_si)
            self.register_buffer("core_alpha", torch.ones(1, N_DYN, 1, 1))
        self.lags = sorted(int(k) for k in (cfg.history or []))
        n_in = N_DYN + static.shape[0] + N_TIME + (N_DYN if self.core is not None else 0) + N_DYN * len(self.lags)
        self.net = make_backbone(n_in, N_DYN, cfg, static.shape[-2], n_lon)
        dyn = contract.DYNAMIC_CHANNELS
        self.pm = PositiveMass(mean[dyn], std[dyn], lat_deg) if cfg.positive else None
        self.grad_ckpt = False  # per 6 h step gradient checkpointing for long rollouts (set by the trainer)
        self.cache_core = None  # a list: the core increments of every step are appended (the trainer's PINN term)
        self.cache_hist = None  # a list: the history features of every step (same use)

    # -- the field ---------------------------------------------------------------------------------------
    def _core_raw(self, x: torch.Tensor, hours: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.core == "speedy":   # the physics needs the state's time (solar zenith angle)
            return self.core.increment(x, hours)
        return self.core.increment(x)

    def core_increment(self, x: torch.Tensor, hours: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """F_phys(x): alpha * the physics core's 6 h increment (normalized), no gradient; None without a core.
        ``hours`` (B,) since Jan 1 00 UTC is used by cores with time-dependent physics (SpeedyWeather)."""
        if self.core is None:
            return None
        return self._core_raw(x, hours) * self.core_alpha * self.keep

    @torch.no_grad()
    def calibrate_core(self, x0: torch.Tensor, d: torch.Tensor, lat_w: torch.Tensor,
                       hours: Optional[torch.Tensor] = None) -> torch.Tensor:
        """alpha_c = <D, d>_w / <D, D>_w per channel on ERA5 pairs (x0, d = x(6 h) - x0), clipped to [0, 1]."""
        raw = self._core_raw(x0, hours)
        w = lat_w.view(1, 1, -1, 1)
        alpha = ((raw * d * w).sum((0, 2, 3)) / (raw * raw * w).sum((0, 2, 3)).clamp_min(1e-12)).clamp(0.0, 1.0)
        self.core_alpha.copy_(alpha.view(1, -1, 1, 1))
        return alpha

    @property
    def n_history(self) -> int:
        """Past states (6 h apart, oldest first) a rollout needs before x0: the largest lag, 0 without history."""
        return max(self.lags) if self.lags else 0

    def history_features(self, buf) -> Optional[torch.Tensor]:
        """The sequence x(t - lag_n), ..., x(t - lag_1), x(t) as the operator sees it: the successive differences
        (x(t) - x(t - lag_1)), (x(t - lag_1) - x(t - lag_2)), ... in units of the 6 h tendency std per lag step,
        (B, 66 * n_lags, H, W). ``buf`` is the list of states every 6 h ending with x(t). The differences carry
        the same information as the raw past states (x itself is an input too) at the scale the head works in:
        the observed velocity and acceleration of the flow."""
        if not self.lags:
            return None
        pts = [buf[-1]] + [buf[-1 - k] for k in self.lags]
        steps = [self.lags[0]] + [b - a for a, b in zip(self.lags[:-1], self.lags[1:])]
        return torch.cat([(pts[i] - pts[i + 1]) / (self.t6 * n) for i, n in enumerate(steps)], 1) * 1.0

    def tendency(self, x: torch.Tensor, hours: torch.Tensor, core_inc: Optional[torch.Tensor] = None,
                 hist: Optional[torch.Tensor] = None) -> torch.Tensor:
        """G_theta(x, t): normalized increment per 6 h, (B, 66, H, W). With a physics core it is
        F_phys + R_theta(x, F_phys / t6, t); ``core_inc`` is F_phys if already known (computed otherwise).
        ``hist`` are the ``history_features`` of the step's start (zeros when a history field gets none)."""
        B, _, H, W = x.shape
        parts = [x, self.static.expand(B, -1, -1, -1), time_features(hours, H, W)]
        if self.lags:
            parts.append(hist if hist is not None else x.new_zeros(B, N_DYN * len(self.lags), H, W))
        if self.core is not None:
            if core_inc is None:
                core_inc = self.core_increment(x, hours)
            parts.append(core_inc / self.t6)
        out = self.net(torch.cat(parts, 1)) * self.t6 * self.keep
        return out if core_inc is None else out + core_inc

    def step(self, x: torch.Tensor, hours: torch.Tensor, tisr_next: Optional[torch.Tensor] = None,
             hist: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One 6 h step: torchdiffeq fixed-grid solve of dx/ds = G(x, hours + 6 s), s in [0, 1]. A physics core
        increment is computed once from the step's start state and held fixed over the step."""
        from torchdiffeq import odeint

        ci = self.core_increment(x, hours)
        if ci is not None and self.cache_core is not None:
            self.cache_core.append(ci)

        def func(s, y):
            return self.tendency(y, hours + HOURS_PER_STEP * s, ci, hist)

        n_sub = max(1, int(round(HOURS_PER_STEP / self.cfg.dt_hours)))
        s = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
        x = odeint(func, x, s, method=self.cfg.solver, options={"step_size": 1.0 / n_sub})[-1]
        if tisr_next is not None:
            x = torch.cat([x[:, :TISR_DYN], tisr_next[:, None], x[:, TISR_DYN + 1:]], 1)
        if self.pm is not None:
            x = self.pm(x)
        return x

    def rollout(self, x0: torch.Tensor, hour0: torch.Tensor, steps: int,
                tisr: Optional[torch.Tensor] = None, past: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, steps, 66, H, W): the states 6 h, 12 h, ... after x0; ``tisr`` (B, steps, H, W) from the data.
        ``past`` (B, n_history, 66, H, W): the states before x0 every 6 h, oldest first (history fields); along
        the rollout the sequence slides over the field's own states."""
        x, hours, out = x0, hour0.float(), []
        ckpt = self.grad_ckpt and torch.is_grad_enabled()
        buf = None
        if self.lags:
            if past is None or past.shape[1] < self.n_history:
                raise ValueError(f"a history field needs the {self.n_history} states before x0 (past)")
            buf = [past[:, i] for i in range(past.shape[1])] + [x0]
        for k in range(steps):
            tk = None if tisr is None else tisr[:, k]
            hf = self.history_features(buf) if buf is not None else None
            if hf is not None and self.cache_hist is not None:
                self.cache_hist.append(hf)
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(self.step, x, hours, tk, hf, use_reentrant=False)
            else:
                x = self.step(x, hours, tk, hf)
            hours = hours + HOURS_PER_STEP
            out.append(x)
            if buf is not None:
                buf = buf[1:] + [x]
        if buf is not None and self.cache_hist is not None:
            self.cache_hist.append(self.history_features(buf))   # of the last state (the trainer's PINN term)
        return torch.stack(out, 1)

    def rollout_members(self, members: torch.Tensor, hour0: torch.Tensor, steps: int,
                        tisr: Optional[torch.Tensor] = None, past: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ensemble scenarios (B, M, 66, H, W) -> (B, M, steps, 66, H, W); every member is a row of the batch
        (the field is the same operator for every member; tisr (B, steps, H, W) is shared)."""
        B, M = members.shape[:2]
        rep = lambda t: None if t is None else t.repeat_interleave(M, 0)  # noqa: E731
        y = self.rollout(members.flatten(0, 1), rep(hour0), steps, rep(tisr), rep(past))
        return y.view(B, M, *y.shape[1:])

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # neuraloperator's BaseModel writes its constructor arguments (functions, classes) under "_metadata";
        # a weights-only torch.load refuses them and nn.Module.load_state_dict reports them as unexpected
        sd.pop("_metadata", None)
        return sd

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
