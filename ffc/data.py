"""ERA5 in the pg-flow (KAI) layout: yearly NetCDF files ``<kai>/<year>.nc`` with ``fields (T, 67, H, W)``
in SI units at 6 h, written by pg-flow's ``scripts/wb2_to_kai.py --all-hours`` from the public
WeatherBench2 zarr (grid 64x32 or 128x64, equiangular without poles).

The loaders come from pg-flow (``pgcfm.runs``: memory-mapped normalized years, tendency std); the
window helpers below are the ones of pg-flow's step 13/16 trainers (``scripts/run_step13_pinn.py``,
``scripts/run_step12_core.py``, ``scripts/run_step11_climode.py``) copied here so that this package
does not import pg-flow's ``scripts`` package (its name collides with ours).
"""

import os
from typing import Dict, List, Sequence, Tuple

import netCDF4 as nc
import numpy as np
import torch

from . import deps  # noqa: F401
from pgcfm import contract  # noqa: E402
from pgcfm.runs import load_years, tendency_std  # noqa: E402,F401

DYN = contract.DYNAMIC_CHANNELS
OROG = contract.STATIC_CHANNELS
N_DYN = contract.N_DYNAMIC
HOURS_PER_STEP = 6.0


def channel_stats(kai: str, years: Sequence[int], out: str) -> Tuple[np.ndarray, np.ndarray]:
    """KAI ``global_{mean,std}.npy`` over the training years (per channel, all times and cells)."""
    os.makedirs(out, exist_ok=True)
    pm, ps = os.path.join(out, "global_mean.npy"), os.path.join(out, "global_std.npy")
    if not os.path.exists(ps):
        s = ss = n = 0.0
        for y in years:
            with nc.Dataset(os.path.join(kai, f"{y}.nc")) as d:
                f = np.asarray(d["fields"][:], dtype=np.float64)
            s, ss, n = s + f.sum((0, 2, 3)), ss + (f ** 2).sum((0, 2, 3)), n + f.shape[0] * f.shape[2] * f.shape[3]
        m = s / n
        np.save(pm, m.astype(np.float32))
        np.save(ps, np.sqrt(np.maximum(ss / n - m ** 2, 0)).astype(np.float32))
    return np.load(pm), np.load(ps)


def grid_of(kai: str, year: int) -> Tuple[np.ndarray, np.ndarray]:
    """(lat, lon) in degrees of a KAI year file; also registers the latitudes with the pg-flow contract so
    every pgcfm grid operator (residuals, area weights) uses them instead of the 1.5 deg default."""
    with nc.Dataset(os.path.join(kai, f"{year}.nc")) as d:
        lat, lon = np.asarray(d["latitude"][:], dtype=np.float64), np.asarray(d["longitude"][:], dtype=np.float64)
    contract.register_latitudes(lat)
    return lat, lon


def times_of(kai: str, year: int) -> np.ndarray:
    with nc.Dataset(os.path.join(kai, f"{year}.nc")) as d:
        t = d["time"]
        return np.asarray(nc.num2date(t[:], units=t.units, calendar=getattr(t, "calendar", "standard"),
                                      only_use_cftime_datetimes=False, only_use_python_datetimes=True),
                          dtype="datetime64[ns]")


def static_features(lat: np.ndarray, lon: np.ndarray, orog_si: np.ndarray, lsm: np.ndarray = None) -> torch.Tensor:
    """(7 or 8, H, W): ClimODE's position features (cos/sin lat, cos/sin lon, sin lat x cos/sin lon) and the
    standardized orography (+ land-sea mask when given); pg-flow ``pgcfm.core.static_features`` without
    the constants file."""
    la, lo = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    oro = orog_si / contract.CONST.gravity
    f = [np.cos(la), np.sin(la), np.cos(lo), np.sin(lo), np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo),
         (oro - oro.mean()) / (oro.std() + 1e-6)]
    if lsm is not None:
        f.append(lsm)
    return torch.from_numpy(np.stack(f).astype(np.float32))


def orography(kai: str, year: int) -> np.ndarray:
    with nc.Dataset(os.path.join(kai, f"{year}.nc")) as d:
        return np.asarray(d["fields"][0, contract.CHANNEL_INDEX["orog"]], dtype=np.float32)


# -- random windows (pg-flow step 13 / 16) ----------------------------------------------------------------

def pick(data: Dict, years: Sequence[int], K: int, rng: np.random.Generator, n: int):
    """n random window starts: (years, indices)."""
    ys = rng.choice(years, n)
    return ys, [int(rng.integers(0, len(data[y]) - K)) for y in ys]


def gather(data: Dict, ys, idx, K: int):
    """Windows -> x0 (n, 66, H, W), d (n, K, 66, H, W) increments, orog (n, 1, H, W), hour0 (n,)."""
    seq = torch.stack([data[y][i:i + K + 1] for y, i in zip(ys, idx)])
    x0 = seq[:, 0, DYN]
    return x0, seq[:, 1:, DYN] - x0[:, None], seq[:, 0, OROG], torch.tensor([HOURS_PER_STEP * i for i in idx])


def pick_hist(data: Dict, years: Sequence[int], K: int, rng: np.random.Generator, n: int, n_hist: int):
    """As ``pick`` with ``n_hist`` states left before every window start."""
    ys = rng.choice(years, n)
    return ys, [int(rng.integers(n_hist, len(data[y]) - K)) for y in ys]


def gather_past(data: Dict, ys, idx, n_hist: int) -> torch.Tensor:
    """(n, n_hist, 66, H, W): the states before each window start every 6 h, oldest first."""
    return torch.stack([data[y][i - n_hist:i] for y, i in zip(ys, idx)])[:, :, DYN]


def sample(data: Dict, years: Sequence[int], K: int, rng: np.random.Generator, n: int):
    return gather(data, *pick(data, years, K, rng, n), K)


def fixed_windows(n_time: int, K: int, n: int, offset: int = 0, hour_index: int = 0) -> List[int]:
    """n evenly spaced window starts at the same UTC hour (index ``hour_index`` of the four 6 h times)."""
    starts = np.linspace(offset, n_time - K - 1, n).astype(int)
    return [int(s - (s % 4) + hour_index) for s in starts]


def wmse(a: torch.Tensor, b, w: torch.Tensor, t6: torch.Tensor) -> torch.Tensor:
    """Latitude-weighted MSE in units of the 6 h tendency variance per channel."""
    return (((a - b) / t6) ** 2 * w).mean()


def lat_weights(lat: np.ndarray, device=None) -> torch.Tensor:
    w = torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32))
    return (w / w.mean()).to(device)
