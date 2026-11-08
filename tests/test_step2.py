"""Step 2 unit tests on a small synthetic grid: spherical GRF source, OT coupling, flow loss / sampling, and the
grafted generator -> frozen field pipeline.

    python -m pytest tests -q
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ffc  # noqa: E402,F401
from ffc import flow as F  # noqa: E402
from ffc.field import DynamicsField, FieldConfig  # noqa: E402
from ffc.sobolev import SphericalSobolev  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402

H, W = 16, 32
LAT = -90 + 180 / H * (np.arange(H) + 0.5)
MEAN, STD = contract.REFERENCE_MEAN, contract.REFERENCE_STD
N = contract.N_DYNAMIC


@pytest.fixture(autouse=True)
def _grid():
    contract.register_latitudes(LAT)
    torch.manual_seed(0)


def _field():
    cfg = FieldConfig(backbone="sfno", width=8, n_layers=1, modes_lat=8, positive=False)
    return DynamicsField(cfg, torch.randn(7, H, W), 0.1 + 0.2 * torch.rand(N), MEAN, STD, LAT, W).eval()


def _flow(coupling="ot", source="matched"):
    cfg = F.FlowConfig(coupling=coupling, source=source, width=8, n_layers=1, modes_lat=4, sample_steps=2)
    spec = F.SphericalGRF.power_law(N, H, 1.5)
    return F.ErrorFlow(cfg, torch.randn(7, H, W), 0.05 + 0.1 * torch.rand(N), spec, LAT, W)


def test_grf_unit_variance_and_spectrum():
    for s in (0.0, 1.0, 2.0):
        grf = F.SphericalGRF(H, W, F.SphericalGRF.power_law(2, H, s))
        f = grf(512, torch.Generator().manual_seed(1))
        w = torch.cos(torch.deg2rad(torch.as_tensor(LAT, dtype=torch.float32))).view(-1, 1)
        var = float((f ** 2 * w / w.mean()).mean())
        assert abs(var - 1.0) < 0.15, (s, var)
    # rougher source (s = 0) has relatively more energy at high degree than a smooth one (s = 2)
    sob = SphericalSobolev(H, W, s=0.0)
    e0 = sob.energy_by_degree(F.SphericalGRF(H, W, F.SphericalGRF.power_law(1, H, 0.0))(64)).mean((0, 1))
    e2 = sob.energy_by_degree(F.SphericalGRF(H, W, F.SphericalGRF.power_law(1, H, 2.0))(64)).mean((0, 1))
    assert (e0[8:].sum() / e0.sum()) > 5 * (e2[8:].sum() / e2.sum())


def test_ot_coupling_is_a_permutation_that_lowers_cost():
    fl = _flow("ot")
    eps = fl.source(8)
    z1 = eps[torch.randperm(8)] + 0.01 * torch.randn_like(eps)
    paired = fl.couple(eps, z1)
    assert torch.allclose(paired.sort(0).values, eps.sort(0).values)  # same draws, reordered
    assert float(((paired - z1) ** 2).mean()) < 1e-3                   # recovers the hidden permutation
    assert torch.equal(_flow("indep").couple(eps, z1), eps)


@pytest.mark.parametrize("source", ["matched", "sobolev", "white"])
def test_flow_loss_backward_and_sample_shapes(source):
    fl = _flow(source=source)
    fld = _field()
    x0 = 0.3 * torch.randn(3, N, H, W)
    hours = torch.tensor([0.0, 6.0, 12.0])
    x1, inc = F.operator_increment(fld, x0, hours)
    cond = fl.condition(x0, inc, hours)
    z1 = torch.randn(3, N, H, W) * fl.keep
    loss = fl.loss(z1, cond)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fl.parameters())
    z = fl.sample(cond)
    assert z.shape == (3, N, H, W)
    assert torch.all(z[:, TISR_DYN] == 0)


def test_flow_members_and_stochastic_rollout():
    fl, fld = _flow(), _field()
    for p in fld.parameters():
        p.requires_grad_(False)
    x0 = 0.3 * torch.randn(2, N, H, W)
    hours = torch.tensor([0.0, 600.0])
    mem = F.flow_members(fl, fld, x0, hours, 4, 2.0, torch.Generator().manual_seed(0))
    assert mem.shape == (2, 4, N, H, W)
    assert torch.equal(mem[:, 0], x0)                       # control member
    assert (mem[:, 1:] - x0[:, None]).abs().sum() > 0
    assert torch.all(mem[:, :, TISR_DYN] == x0[:, None, TISR_DYN])
    y = F.rollout_stochastic(fl, fld, mem, hours, 3, 0.5, torch.full((2, 3, H, W), 0.4),
                             torch.Generator().manual_seed(0))
    assert y.shape == (2, 4, 3, N, H, W)
    det = fld.rollout(x0, hours, 3, torch.full((2, 3, H, W), 0.4))
    assert torch.allclose(y[:, 0], det, atol=1e-5)          # control member stays deterministic
