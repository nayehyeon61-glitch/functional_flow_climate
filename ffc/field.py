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

    def __init__(self, cfg: FieldConfig, static: torch.Tensor, t6: torch.Tensor, mean, std, lat_deg, n_lon: int):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("static", static)
        self.register_buffer("t6", t6.view(1, N_DYN, 1, 1).clone())
        keep = torch.ones(N_DYN)
        keep[TISR_DYN] = 0.0  # tisr is a known forcing, never predicted
        self.register_buffer("keep", keep.view(1, N_DYN, 1, 1))
        self.net = make_backbone(N_DYN + static.shape[0] + N_TIME, N_DYN, cfg, static.shape[-2], n_lon)
        dyn = contract.DYNAMIC_CHANNELS
        self.pm = PositiveMass(mean[dyn], std[dyn], lat_deg) if cfg.positive else None
        self.grad_ckpt = False  # per 6 h step gradient checkpointing for long rollouts (set by the trainer)

    # -- the field ---------------------------------------------------------------------------------------
    def tendency(self, x: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        """G_theta(x, t): normalized increment per 6 h, (B, 66, H, W)."""
        B, _, H, W = x.shape
        inp = torch.cat([x, self.static.expand(B, -1, -1, -1), time_features(hours, H, W)], 1)
        return self.net(inp) * self.t6 * self.keep

    def step(self, x: torch.Tensor, hours: torch.Tensor, tisr_next: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One 6 h step: torchdiffeq fixed-grid solve of dx/ds = G(x, hours + 6 s), s in [0, 1]."""
        from torchdiffeq import odeint

        def func(s, y):
            return self.tendency(y, hours + HOURS_PER_STEP * s)

        n_sub = max(1, int(round(HOURS_PER_STEP / self.cfg.dt_hours)))
        s = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
        x = odeint(func, x, s, method=self.cfg.solver, options={"step_size": 1.0 / n_sub})[-1]
        if tisr_next is not None:
            x = torch.cat([x[:, :TISR_DYN], tisr_next[:, None], x[:, TISR_DYN + 1:]], 1)
        if self.pm is not None:
            x = self.pm(x)
        return x

    def rollout(self, x0: torch.Tensor, hour0: torch.Tensor, steps: int,
                tisr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, steps, 66, H, W): the states 6 h, 12 h, ... after x0; ``tisr`` (B, steps, H, W) from the data."""
        x, hours, out = x0, hour0.float(), []
        ckpt = self.grad_ckpt and torch.is_grad_enabled()
        for k in range(steps):
            tk = None if tisr is None else tisr[:, k]
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(self.step, x, hours, tk, use_reentrant=False)
            else:
                x = self.step(x, hours, tk)
            hours = hours + HOURS_PER_STEP
            out.append(x)
        return torch.stack(out, 1)

    def rollout_members(self, members: torch.Tensor, hour0: torch.Tensor, steps: int,
                        tisr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ensemble scenarios (B, M, 66, H, W) -> (B, M, steps, 66, H, W); every member is a row of the batch
        (the field is the same operator for every member; tisr (B, steps, H, W) is shared)."""
        B, M = members.shape[:2]
        rep = lambda t: None if t is None else t.repeat_interleave(M, 0)  # noqa: E731
        y = self.rollout(members.flatten(0, 1), rep(hour0), steps, rep(tisr))
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
