"""pg-flow ensemble I/O: the operator takes flow-matching scenarios in and writes forecasts in the same file.

pg-flow (``scripts/infer_pgcfm.py``, ``pgcfm/eval/wb2.py from_kai_forecast_nc``) stores an ensemble as the
KAI ``forecast.nc`` layout with a member dimension:

    forecast (initial_condition, lead_time, member, channel, latitude, longitude)   physical units, 67 channels
    lead 0 = the initial condition, ``initial_condition`` = index into the test-year KAI file

A deterministic file has no ``member`` dimension. ``read_members`` turns such a file into the operator's
input (normalized members at one lead), ``write_forecast`` writes the operator's rollout back in the same
layout, so pg-flow's evaluation and this package's WeatherBench2 scoring read both alike.

When no flow-matching ensemble is at hand (this cloud environment has no pg-flow checkpoint),
``perturbed_members`` makes scenarios in the same format from the data: x0 + alpha x (a random 6 h analysis
increment of the training years), which has the spatial covariance of real tendencies. It is a stand-in
for the flow-matching members, and is labelled as such in every result that uses it.
"""

import os
from typing import Dict, Optional, Sequence

import netCDF4 as nc
import numpy as np
import torch

from . import deps  # noqa: F401
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402
from pgcfm.eval.wb2 import from_kai_forecast_nc  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS


def write_forecast(path: str, forecast: np.ndarray, ic_index: Sequence[int], lead_hours: Sequence[float],
                   lat: np.ndarray, lon: np.ndarray, init_times: Optional[np.ndarray] = None,
                   title: str = "functional_flow_climate dynamics field") -> str:
    """forecast (N, L, [M], 67, H, W) in physical units -> pg-flow / KAI ``forecast.nc``."""
    has_m = forecast.ndim == 6
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with nc.Dataset(path + ".tmp", "w", format="NETCDF4") as ds:
        ds.createDimension("initial_condition", forecast.shape[0])
        ds.createDimension("lead_time", forecast.shape[1])
        if has_m:
            ds.createDimension("member", forecast.shape[2])
        ds.createDimension("channel", contract.N_CHANNELS)
        ds.createDimension("latitude", forecast.shape[-2])
        ds.createDimension("longitude", forecast.shape[-1])
        dims = ("initial_condition", "lead_time") + (("member",) if has_m else ()) + ("channel", "latitude", "longitude")
        v = ds.createVariable("forecast", "f4", dims, zlib=True, complevel=1, shuffle=True)
        v[:] = forecast.astype(np.float32)
        v.units = "physical (pgcfm.contract.UNITS per channel)"
        ds.createVariable("initial_condition", "i8", ("initial_condition",))[:] = np.asarray(ic_index, dtype=np.int64)
        lt = ds.createVariable("lead_time", "f8", ("lead_time",))
        lt[:] = np.asarray(lead_hours, dtype=np.float64)
        lt.units = "hours"
        if has_m:
            ds.createVariable("member", "i4", ("member",))[:] = np.arange(forecast.shape[2])
        ch = ds.createVariable("channel", str, ("channel",))
        for i, n in enumerate(contract.CHANNEL_NAMES):
            ch[i] = n
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        if init_times is not None:
            it = ds.createVariable("init_time", "i8", ("initial_condition",))
            it[:] = (np.asarray(init_times, dtype="datetime64[ns]") - np.datetime64("1970-01-01")) // np.timedelta64(1, "h")
            it.units = "hours since 1970-01-01"
        ds.title = title
    os.replace(path + ".tmp", path)
    return path


def read_forecast(path: str) -> Dict[str, np.ndarray]:
    """pg-flow / KAI forecast file -> ``forecast`` (N, L, [M], 67, H, W) physical, ``ic_index``, ``lead``,
    ``has_members`` (pgcfm.eval.wb2.from_kai_forecast_nc), plus ``lat``, ``lon``, ``init_time`` when present."""
    out = from_kai_forecast_nc(path)
    with nc.Dataset(path) as ds:
        for k in ("latitude", "longitude", "init_time"):
            if k in ds.variables:
                out[{"latitude": "lat", "longitude": "lon"}.get(k, k)] = np.asarray(ds[k][:])
        if "lead_time" in ds.variables and getattr(ds["lead_time"], "units", "hours").startswith("day"):
            out["lead"] = out["lead"] * 24.0
    return out


def read_members(path: str, lead_index: int, mean: np.ndarray, std: np.ndarray) -> Dict[str, torch.Tensor]:
    """The scenarios of a pg-flow file at one lead as operator input.

    Returns ``members`` (N, M, 66, H, W) normalized (M = 1 for a deterministic file), ``orog`` (N, 1, H, W)
    normalized, ``ic_index``, ``lead_hours``."""
    f = read_forecast(path)
    x = f["forecast"][:, lead_index]
    if not f["has_members"]:
        x = x[:, None]
    xn = contract.normalize(x, mean, std, channel_axis=-3).astype(np.float32)
    return {"members": torch.from_numpy(xn[:, :, DYN]), "orog": torch.from_numpy(xn[:, 0][:, contract.STATIC_CHANNELS]),
            "ic_index": f["ic_index"], "lead_hours": float(f["lead"][lead_index])}


def perturbed_members(x0: torch.Tensor, data: Dict, years: Sequence[int], M: int, alpha: float,
                      rng: np.random.Generator, dyn=DYN) -> torch.Tensor:
    """(B, 66, H, W) -> (B, M, 66, H, W): member 0 is x0 itself, members 1..M-1 are x0 + alpha x a random 6 h
    analysis increment of the training years (tisr untouched). A stand-in for flow-matching scenarios."""
    B = x0.shape[0]
    out = [x0]
    for _ in range(M - 1):
        incs = []
        for _ in range(B):
            y = rng.choice(years)
            i = int(rng.integers(0, len(data[y]) - 1))
            pair = data[y][i:i + 2][:, dyn]
            incs.append(pair[1] - pair[0])
        d = torch.stack(incs).to(x0)
        d[:, TISR_DYN] = 0.0
        out.append(x0 + alpha * d)
    return torch.stack(out, 1)
