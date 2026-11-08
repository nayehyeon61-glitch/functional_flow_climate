"""PINN term of the dynamics field: strong-form primitive-equation residuals with the operator's own tendency.

Everything physical comes from pg-flow (``pgcfm.physics``): the pressure-coordinate primitive equations
(momentum, thermodynamic, continuity, hydrostatic; paper eqs. 5-8), the spherical finite-difference
operators with the polar filter, the below-ground mask, and ``pgcfm.dyncore.StrongPhysics`` which puts the
network tendency f (normalized units per 6 h) into d_s X of those equations. This module only fixes how the
term enters the loss here:

* scales b_j are the RMS of the same residuals on ERA5 with centred 12 h differences, so 1 = ERA5's own
  level (``estimate_strong_scales``);
* the loss is a hinge at that level, sum_j gamma_j max(m_j - target, 0), because pushing the residual to
  zero on this grid drives the field to a resting atmosphere (pg-flow step 3 / 16: the discrete operator's
  truncation error is as large as the tendency, so "zero residual" means "no motion");
* collocation points are fixed by the observed state at the same lead (surface pressure of the truth), so
  the field cannot remove points by changing sp.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch

from . import deps  # noqa: F401
from pgcfm.coupling import PhysicsConfig, PhysicsGuide  # noqa: E402
from pgcfm.dyncore import StrongPhysics, estimate_strong_scales, finite_difference_tendency  # noqa: E402

DEFAULT_GAMMA = {"u": 1.0, "v": 1.0, "T": 1.0, "mass": 1.0, "hyd": 1.0, "q": 0.0}


@dataclass
class PhysicsLossConfig:
    lambda_p: float = 0.05
    target: float = 1.0                     # hinge target in ERA5 residual units; 0 = plain eq. (10)
    gamma: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GAMMA))
    states_per_sample: int = 2              # rollout states (the IC included) entering the term per window
    pole_cap_deg: float = 85.0
    lat_weight: str = "area"


class PhysicsResidual:
    """Strong residuals of a normalized 66-channel state with a normalized 6 h tendency.

    ``mean``, ``std``: the 67-channel KAI statistics; ``t6``: (66,) tendency std (normalized units / 6 h);
    the grid comes from the registered latitudes (``ffc.data.grid_of``) and ``n_lon``."""

    def __init__(self, n_lat: int, n_lon: int, mean, std, t6: torch.Tensor, cfg: PhysicsLossConfig = PhysicsLossConfig(),
                 dt_hours: float = 6.0, device="cpu"):
        self.cfg = cfg
        guide = PhysicsGuide(n_lat, n_lon, mean, std,
                             PhysicsConfig(dt_seconds=dt_hours * 3600.0, pole_cap_deg=cfg.pole_cap_deg,
                                           lat_weight=cfg.lat_weight))
        for q in guide.parameters():
            q.requires_grad_(False)
        self.guide = guide.to(device)
        self.phys = StrongPhysics(self.guide, t6.to(device), dt_seconds=dt_hours * 3600.0)

    # -- calibration --------------------------------------------------------------------------------------
    def calibrate(self, triples: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        """b_j from ERA5 triples (x_prev, x, x_next, orog) with centred-difference tendencies."""
        self.phys.scales = estimate_strong_scales(self.phys, triples)
        return self.phys.scales

    @torch.no_grad()
    def reference_levels(self, triples) -> Dict[str, Dict[str, float]]:
        """Diagnostics of pg-flow step 16 sec. 4: the rest state (f = 0) and the tendency term alone over
        ERA5's level, i.e. how much of each residual the tendency can change at all on this grid."""
        rest, tend, n = {}, {}, len(triples)
        for x0, x1, x2, orog in triples:
            for k, v in self.phys.per_equation(x1, orog, torch.zeros_like(x1)).items():
                rest[k] = rest.get(k, 0.0) + v ** 2 / n
            R1, m = self.phys.residuals(x1, orog, finite_difference_tendency(x0, x2))
            R0, _ = self.phys.residuals(x1, orog, torch.zeros_like(x1))
            for k, v in self.phys.mean_squares({q: R1[q] - R0[q] for q in ("u", "v", "T")}, m).items():
                tend[k] = tend.get(k, 0.0) + float(v) / n
        return {"rest_over_era5": {k: math.sqrt(v) for k, v in rest.items()},
                "tendency_over_era5": {k: math.sqrt(v) for k, v in tend.items()}}

    # -- loss -----------------------------------------------------------------------------------------------
    def mean_squares(self, x: torch.Tensor, orog: torch.Tensor, f: torch.Tensor, x_obs: torch.Tensor
                     ) -> Dict[str, torch.Tensor]:
        """Weighted mean of (R_j / b_j)^2 per equation at the states x with tendency f (kept in the graph);
        the collocation mask from the observed state ``x_obs`` at the same lead."""
        R, _ = self.phys.residuals(x, orog, f)
        return self.phys.mean_squares(R, self.guide.mask(x_obs, x_obs, orog))

    def loss(self, ms: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.phys.loss(ms, self.cfg.gamma, self.cfg.target)

    @torch.no_grad()
    def per_equation(self, x: torch.Tensor, orog: torch.Tensor, f: torch.Tensor,
                     x_obs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """RMS of R_j / b_j per equation (1 = ERA5's level)."""
        return self.phys.per_equation(x, orog, f, x_obs=x_obs)

    @property
    def scales(self) -> Dict[str, float]:
        return vars(self.phys.scales)
