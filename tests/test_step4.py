"""Step 4 tests: the dinosaur physics core (ffc.physcore) and the hybrid field F_phys + R_theta.

The core needs the T21 grid (64 x 32 with the KAI 5.625 deg longitudes); the first call compiles (~30 s on CPU).
    python -m pytest tests/test_step4.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ffc  # noqa: E402,F401
from pgcfm import contract  # noqa: E402

pytest.importorskip("dinosaur")
from ffc.field import DynamicsField, FieldConfig  # noqa: E402
from ffc.physcore import DYN_SLICE, SP_DYN, CoreConfig, SpectralCore  # noqa: E402

H, W = 32, 64
LAT = -90 + 180 / H * (np.arange(H) + 0.5)  # the WB2 5.625 deg latitudes
N = contract.N_DYNAMIC
R_DRY, T0, P0 = 287.0, 250.0, 100000.0
CORE = dict(layers=8, dt_minutes=30.0, dfi_hours=0.0)


@pytest.fixture(autouse=True)
def _grid():
    contract.register_latitudes(LAT)
    torch.manual_seed(0)


def _rest_state(n=1):
    """An isothermal resting atmosphere over flat ground in SI units, hydrostatic on the KAI pressure levels."""
    x = np.zeros((n, N, H, W))
    p = np.asarray(contract.LEVELS_PA)
    x[:, DYN_SLICE["z"]] = (R_DRY * T0 * np.log(P0 / p))[None, :, None, None]
    x[:, DYN_SLICE["t"]] = T0
    x[:, DYN_SLICE["q"]] = 1e-6
    x[:, SP_DYN] = P0
    return torch.tensor(x, dtype=torch.float32)


@pytest.fixture(scope="module")
def core():
    contract.register_latitudes(LAT)
    return SpectralCore(CoreConfig(**CORE), np.zeros(contract.N_CHANNELS), np.ones(contract.N_CHANNELS), LAT,
                        np.zeros((H, W), np.float32))


def test_rest_state_stays_at_rest(core):
    d = core.increment(_rest_state())
    assert d.shape == (1, N, H, W) and torch.isfinite(d).all()
    assert d[:, DYN_SLICE["u"]].abs().max() < 0.05           # m/s in 6 h (ERA5: ~3 m/s)
    assert d[:, DYN_SLICE["t"]].abs().max() < 0.05           # K
    assert d[:, SP_DYN].abs().max() < 20.0                    # Pa (ERA5: ~200 Pa)


def test_a_jet_moves_and_only_core_channels_change(core):
    x = _rest_state()
    phi = torch.tensor(np.deg2rad(LAT), dtype=torch.float32).view(1, 1, H, 1)
    lam = torch.linspace(0, 2 * np.pi, W + 1)[:-1].view(1, 1, 1, W)
    x[:, DYN_SLICE["u"]] = 20.0 * torch.cos(phi) ** 2 * (1 + 0.2 * torch.sin(3 * lam))  # unbalanced wavy jet
    d = core.increment(x)
    assert torch.isfinite(d).all()
    assert d[:, DYN_SLICE["v"]].abs().max() > 0.1             # the Coriolis force turns the jet
    rest = [c for c in range(N) if c != SP_DYN and not any(c in range(s.start, s.stop) for s in DYN_SLICE.values())]
    assert d[:, rest].abs().max() == 0                        # t2m, tisr, tcwv, msl, skt: learned by the operator


def test_hybrid_starts_as_the_core_and_learns():
    cfg = FieldConfig(backbone="sfno", width=8, n_layers=1, modes_lat=8, core="dinosaur", core_cfg=CORE, positive=False)
    m = DynamicsField(cfg, torch.randn(7, H, W), torch.ones(N), np.zeros(contract.N_CHANNELS),
                      np.ones(contract.N_CHANNELS), LAT, W, orog_si=np.zeros((H, W), np.float32))
    x, hours = _rest_state(), torch.zeros(1)
    x[:, DYN_SLICE["u"]] = 10.0
    ci = m.core_increment(x)
    assert torch.allclose(m.tendency(x, hours), ci)          # zero head: the hybrid is the core alone
    loss = (m.rollout(x, hours, 1) ** 2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in m.net.parameters())
