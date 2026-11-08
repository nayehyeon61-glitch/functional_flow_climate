"""Step 1 unit tests on a small synthetic grid (no data download): field, Sobolev norm, physics term, ensemble I/O.

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
from ffc import ensemble as E  # noqa: E402
from ffc.field import DynamicsField, FieldConfig  # noqa: E402
from ffc.physics import PhysicsLossConfig, PhysicsResidual  # noqa: E402
from ffc.sobolev import SphericalSobolev  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402

H, W = 16, 32
LAT = -90 + 180 / H * (np.arange(H) + 0.5)  # equiangular cell centres without the poles
LON = 360 / W * np.arange(W)
MEAN, STD = contract.REFERENCE_MEAN, contract.REFERENCE_STD
DYN = contract.DYNAMIC_CHANNELS


@pytest.fixture(autouse=True)
def _grid():
    contract.register_latitudes(LAT)
    torch.manual_seed(0)


def _state(n=2):
    """Normalized states with positive humidity (so PositiveMass is the identity)."""
    x = 0.3 * torch.randn(n, contract.N_DYNAMIC, H, W)
    q = [DYN.index(c) for c in range(*contract.level_slice("q").indices(contract.N_CHANNELS))]
    x[:, q] = x[:, q].abs()
    x[:, DYN.index(contract.CHANNEL_INDEX["tcwv"])] = 0.5
    return x


def _model(backbone, positive=True):
    cfg = FieldConfig(backbone=backbone, width=8, n_layers=1, modes_lat=8, modes_lon=8, positive=positive)
    static = torch.randn(7, H, W)
    t6 = 0.1 + 0.2 * torch.rand(contract.N_DYNAMIC)
    return DynamicsField(cfg, static, t6, MEAN, STD, LAT, W)


@pytest.mark.parametrize("backbone", ["sfno", "fno", "resnet"])
def test_field_starts_as_persistence_and_learns(backbone):
    m = _model(backbone)
    x, hours = _state(), torch.tensor([0.0, 1200.0])
    tisr = torch.full((2, 3, H, W), 0.7)
    y = m.rollout(x, hours, 3, tisr)
    assert y.shape == (2, 3, contract.N_DYNAMIC, H, W)
    keep = [i for i in range(contract.N_DYNAMIC) if i != TISR_DYN]
    assert torch.allclose(y[:, :, keep], x[:, None, keep].expand_as(y[:, :, keep]), atol=1e-6)  # zero head
    assert torch.allclose(y[:, :, TISR_DYN], tisr)  # tisr is the data's forcing
    # the field moves after one gradient step and tisr stays untouched
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    loss = ((m.rollout(x, hours, 1) - 0.5) ** 2).mean()
    loss.backward()
    opt.step()
    f = m.tendency(x, hours)
    assert f[:, keep].abs().max() > 0 and f[:, TISR_DYN].abs().max() == 0


def test_members_share_the_operator():
    m = _model("sfno")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    x = _state(2)
    members = torch.stack([x, x + 0.1 * torch.randn_like(x)], 1)  # (B, M, C, H, W)
    hours = torch.tensor([0.0, 6.0])
    y = m.rollout_members(members, hours, 2)
    assert y.shape == (2, 2, 2, contract.N_DYNAMIC, H, W)
    direct = m.rollout(members[:, 1], hours, 2)
    assert torch.allclose(y[:, 1], direct, atol=1e-5)


def test_sobolev_norm_is_parseval_and_orders_scales():
    sob0 = SphericalSobolev(H, W, s=0.0)
    la, lo = np.meshgrid(np.deg2rad(LAT), np.deg2rad(LON), indexing="ij")
    smooth = torch.tensor(np.cos(la) * np.cos(lo), dtype=torch.float32)[None, None]
    assert sob0.parseval_error(smooth, LAT) < 0.05
    rough = torch.tensor(np.cos(la) * np.cos(8 * lo), dtype=torch.float32)[None, None]
    h1 = SphericalSobolev(H, W, s=1.0)
    hm1 = SphericalSobolev(H, W, s=-1.0)
    # same L2 size, but H1 penalizes the rough field more and H^-1 less
    assert h1.norm2(rough) / h1.norm2(smooth) > 5
    assert hm1.norm2(rough) / hm1.norm2(smooth) < 0.5
    # loss is differentiable
    a = torch.randn(1, 3, H, W, requires_grad=True)
    h1.loss(a, torch.zeros_like(a), scale=torch.ones(1, 3, 1, 1)).backward()
    assert a.grad is not None


def test_physics_term_rest_state_and_hinge():
    t6 = 0.1 + 0.2 * torch.rand(contract.N_DYNAMIC)
    phys = PhysicsResidual(H, W, MEAN, STD, t6, PhysicsLossConfig(target=1.0))
    x = _state(3)
    orog = torch.zeros(3, 1, H, W)
    # calibrate on "ERA5" triples of the synthetic data so that the ERA5 level is 1 by construction
    triples = [(x, x + 0.05 * torch.randn_like(x), x + 0.1 * torch.randn_like(x), orog)]
    scales = phys.calibrate(triples)
    assert all(v > 0 for v in vars(scales).values())
    ms = phys.mean_squares(triples[0][1], orog, 0.5 * (triples[0][2] - triples[0][0]), triples[0][1])
    # the calibration pair itself sits at the "ERA5" level: 1 for T / mass / hyd, and for the momentum pair
    # on average (b_mom is the RMS over u and v together; the synoptic floor may lift q)
    for k in ("T", "mass", "hyd"):
        assert abs(float(ms[k]) - 1.0) < 0.3
    assert abs(0.5 * (float(ms["u"]) + float(ms["v"])) - 1.0) < 0.3
    assert float(phys.loss({k: torch.tensor(0.5) for k in ms})) == 0.0  # below the target: no push
    assert float(phys.loss({k: torch.tensor(2.0) for k in ms})) > 0.0
    # gradient reaches the tendency
    f = torch.zeros_like(x, requires_grad=True)
    phys.loss(phys.mean_squares(x, orog, f, x)).backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()


def test_forecast_roundtrip_and_members(tmp_path):
    N, L, M = 2, 3, 2
    fc = np.random.rand(N, L, M, contract.N_CHANNELS, H, W).astype(np.float32) * STD[None, None, None, :, None, None] + MEAN[None, None, None, :, None, None]
    path = str(tmp_path / "ens.nc")
    E.write_forecast(path, fc, [0, 40], [0.0, 6.0, 12.0], LAT, LON)
    back = E.read_forecast(path)
    assert back["has_members"] and back["forecast"].shape == fc.shape
    np.testing.assert_allclose(back["forecast"], fc, rtol=1e-6)
    assert list(back["lead"]) == [0.0, 6.0, 12.0] and list(back["ic_index"]) == [0, 40]
    mem = E.read_members(path, 1, MEAN, STD)
    assert mem["members"].shape == (N, M, contract.N_DYNAMIC, H, W) and mem["lead_hours"] == 6.0
    # deterministic file: no member dim, read back as one member
    E.write_forecast(str(tmp_path / "det.nc"), fc[:, :, 0], [0, 40], [0.0, 6.0, 12.0], LAT, LON)
    det = E.read_members(str(tmp_path / "det.nc"), 0, MEAN, STD)
    assert det["members"].shape[1] == 1
    # perturbed stand-in members: member 0 is the IC, tisr untouched
    data = {2000: torch.randn(10, contract.N_CHANNELS, H, W)}
    x0 = torch.randn(N, contract.N_DYNAMIC, H, W)
    pm = E.perturbed_members(x0, data, [2000], 4, 0.5, np.random.default_rng(0))
    assert pm.shape == (N, 4, contract.N_DYNAMIC, H, W)
    assert torch.equal(pm[:, 0], x0) and torch.equal(pm[:, 1:, TISR_DYN], x0[:, None, TISR_DYN].expand(-1, 3, -1, -1))
