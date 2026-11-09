"""Step 5 tests: the SpeedyWeather.jl physics core (ffc.speedycore) behind the same contract as the dinosaur core.

Skipped without a Julia + SpeedyWeather.jl install (scripts/install_speedyweather.sh); the worker's first call
compiles (~1 min on CPU).
    JULIA_DEPOT_PATH=/opt/jdepot python -m pytest tests/test_step5.py -q
"""

import os
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ffc  # noqa: E402,F401
from pgcfm import contract  # noqa: E402

pytest.importorskip("dinosaur")
JULIA = os.environ.get("FFC_JULIA", "/opt/mamba/envs/jl/bin/julia")
if not (os.path.exists(JULIA) or shutil.which(JULIA)):
    pytest.skip("no Julia / SpeedyWeather.jl (scripts/install_speedyweather.sh)", allow_module_level=True)

from ffc.field import DynamicsField, FieldConfig  # noqa: E402
from ffc.speedycore import DYN_SLICE, SKT_DYN, SP_DYN, SpeedyConfig, SpeedyCore  # noqa: E402

H, W = 32, 64
LAT = -90 + 180 / H * (np.arange(H) + 0.5)
N = contract.N_DYNAMIC
R_DRY, T0, P0 = 287.0, 250.0, 100000.0
CORE = dict(layers=8, dt_minutes=30.0, physics="none")


def _rest_state(n=1):
    """An isothermal resting atmosphere in SI units, hydrostatic on the KAI pressure levels (the core puts its
    surface pressure on its own orography from this z profile)."""
    x = np.zeros((n, N, H, W))
    p = np.asarray(contract.LEVELS_PA)
    x[:, DYN_SLICE["z"]] = (R_DRY * T0 * np.log(P0 / p))[None, :, None, None]
    x[:, DYN_SLICE["t"]] = T0
    x[:, DYN_SLICE["q"]] = 1e-6
    x[:, SP_DYN] = P0
    x[:, SKT_DYN] = T0
    return torch.tensor(x, dtype=torch.float32)


@pytest.fixture(scope="module")
def core():
    contract.register_latitudes(LAT)
    c = SpeedyCore(SpeedyConfig(**CORE), np.zeros(contract.N_CHANNELS), np.ones(contract.N_CHANNELS), LAT)
    yield c
    c.close()


def test_rest_state_stays_near_rest(core):
    d = core.increment(_rest_state(), torch.zeros(1))
    assert d.shape == (1, N, H, W) and torch.isfinite(d).all()
    # sigma-coordinate pressure-gradient error over the T21 terrain only (ERA5 6 h changes: ~3 m/s, ~1 K)
    assert d[:, DYN_SLICE["u"]].abs().max() < 1.0
    assert d[:, DYN_SLICE["t"]].abs().max() < 0.5


def test_a_jet_moves_and_only_core_channels_change(core):
    x = _rest_state()
    phi = torch.tensor(np.deg2rad(LAT), dtype=torch.float32).view(1, 1, H, 1)
    lam = torch.linspace(0, 2 * np.pi, W + 1)[:-1].view(1, 1, 1, W)
    x[:, DYN_SLICE["u"]] = 20.0 * torch.cos(phi) ** 2 * (1 + 0.2 * torch.sin(3 * lam))
    d = core.increment(x, torch.zeros(1))
    assert torch.isfinite(d).all()
    assert d[:, DYN_SLICE["v"]].abs().max() > 0.1             # the Coriolis force turns the jet
    rest = [c for c in range(N) if c != SP_DYN and not any(c in range(s.start, s.stop) for s in DYN_SLICE.values())]
    assert d[:, rest].abs().max() == 0                        # t2m, tisr, tcwv, msl, skt: learned by the operator


def test_hybrid_starts_as_the_core():
    contract.register_latitudes(LAT)
    cfg = FieldConfig(backbone="sfno", width=8, n_layers=1, modes_lat=8, core="speedy", core_cfg=CORE, positive=False)
    m = DynamicsField(cfg, torch.randn(7, H, W), torch.ones(N), np.zeros(contract.N_CHANNELS),
                      np.ones(contract.N_CHANNELS), LAT, W, orog_si=np.zeros((H, W), np.float32))
    x, hours = _rest_state(), torch.zeros(1)
    x[:, DYN_SLICE["u"]] = 10.0
    ci = m.core_increment(x, hours)
    assert torch.allclose(m.tendency(x, hours), ci)          # zero head: the hybrid is the core alone
    m.core.close()
