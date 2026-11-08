"""WeatherBench2 scoring of pg-flow / ffc forecast files against the ERA5 truth.

The metric code is WeatherBench2's own (vendored in pg-flow, ``pgcfm/eval/wb2.py`` and ``scores.py``: RMSE,
bias, ACC, CRPS, spread, spread-skill ratio with WB2's area weighting and averaging order). This module
builds the xarray datasets WB2 expects from the KAI channel layout on the grid at hand (6 h leads, latitudes
without poles) and a day-of-year / hour climatology for the ACC from the training years.
"""

import os
from typing import Dict, List, Optional, Sequence

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

from . import deps  # noqa: F401
from pgcfm import contract  # noqa: E402
from pgcfm.eval import scores as wb2_scores  # noqa: E402
from pgcfm.eval.wb2 import HEADLINE, LEAD, REALIZATION, WB2_NAMES, headline_label, select_headline  # noqa: E402

HEADLINE_LABELS = [headline_label(v, l) for v, l in HEADLINE]


def to_dataset(x: np.ndarray, init_times: Sequence[np.datetime64], lead_hours: Sequence[float], lat: np.ndarray,
               lon: np.ndarray, member_axis: bool, variables: Optional[Sequence[str]] = None) -> xr.Dataset:
    """(N, L, [M], 67, H, W) physical -> WB2 dataset (init_time, prediction_timedelta, [realization], ...).
    pg-flow's ``wb2.to_dataset`` with hourly leads and the grid's own latitudes."""
    variables = list(variables) if variables is not None else contract.UPPER_AIR_VARS + contract.SURFACE_VARS
    init = np.asarray(init_times, dtype="datetime64[ns]")
    lead = (np.asarray(lead_hours, dtype=np.float64) * 3600e9).astype("timedelta64[ns]")
    dims = ["init_time", LEAD] + ([REALIZATION] if member_axis else [])
    coords = {"init_time": init, LEAD: lead, "latitude": np.asarray(lat), "longitude": np.asarray(lon)}
    if member_axis:
        coords[REALIZATION] = np.arange(x.shape[2])
    c_axis = len(dims)
    data_vars = {}
    for v in variables:
        if v in contract.UPPER_AIR_VARS:
            sl = contract.level_slice(v)
            data_vars[WB2_NAMES[v]] = (dims + ["level", "latitude", "longitude"],
                                       np.take(x, range(sl.start, sl.stop), axis=c_axis))
        else:
            data_vars[WB2_NAMES[v]] = (dims + ["latitude", "longitude"], np.take(x, contract.CHANNEL_INDEX[v], axis=c_axis))
    if any(v in contract.UPPER_AIR_VARS for v in variables):
        coords["level"] = np.array(contract.LEVELS_HPA)
    ds = xr.Dataset(data_vars, coords=coords)
    return ds.assign_coords(valid_time=ds.init_time + ds[LEAD])


def truth_from_kai(kai: str, year: int, ic_index: Sequence[int], lead_hours: Sequence[float], lat, lon,
                   step_hours: float = 6.0) -> xr.Dataset:
    """ERA5 states at init + lead from the KAI year file, as a WB2 truth dataset (no members)."""
    with nc.Dataset(os.path.join(kai, f"{year}.nc")) as d:
        fields = d["fields"]
        t = d["time"]
        times = np.asarray(nc.num2date(t[:], units=t.units, calendar=getattr(t, "calendar", "standard"),
                                       only_use_cftime_datetimes=False, only_use_python_datetimes=True),
                           dtype="datetime64[ns]")
        steps = [int(round(h / step_hours)) for h in lead_hours]
        # One pass over the needed time slab in 64-step reads, then gather in memory: reading the 1800 single time
        # steps of a 72 IC x 25 lead truth one by one decompresses a chunk per read and took longer than scoring.
        lo = int(min(ic_index)) + min(steps)
        hi = int(max(ic_index)) + max(steps) + 1
        buf = np.empty((hi - lo,) + fields.shape[1:], dtype=np.float32)
        for i0 in range(lo, hi, 64):
            i1 = min(i0 + 64, hi)
            buf[i0 - lo:i1 - lo] = fields[i0:i1]
        out = buf[np.array([[int(i) + s - lo for s in steps] for i in ic_index])]
        del buf
    return to_dataset(out, times[list(ic_index)], lead_hours, lat, lon, member_axis=False)


def init_times_of(kai: str, year: int, ic_index: Sequence[int]) -> np.ndarray:
    with nc.Dataset(os.path.join(kai, f"{year}.nc")) as d:
        t = d["time"]
        times = np.asarray(nc.num2date(t[:], units=t.units, calendar=getattr(t, "calendar", "standard"),
                                       only_use_cftime_datetimes=False, only_use_python_datetimes=True),
                           dtype="datetime64[ns]")
    return times[list(ic_index)]


def climatology_from_kai(kai: str, years: Sequence[int], lat, lon, window_days: int = 15) -> xr.Dataset:
    """(dayofyear, hour, ...) mean of the training years, smoothed over +-window_days (WB2's climatology
    layout for ``metrics.ACC``: it selects by valid_time.dayofyear and valid_time.hour)."""
    # Accumulated in float32 straight into the (dayofyear, hour) array, year file read in chunks, smoothing per
    # channel: the whole thing stays near one copy of the climatology in memory (the six 64x32 years in float64
    # with a dict of sums went past the memory of the cloud box).
    C, H, W = contract.N_CHANNELS, len(lat), len(lon)
    arr = cnt = hours = None
    for y in years:
        with nc.Dataset(os.path.join(kai, f"{y}.nc")) as d:
            t = d["time"]
            times = pd.DatetimeIndex(nc.num2date(t[:], units=t.units, calendar=getattr(t, "calendar", "standard"),
                                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True))
            doy, hour = times.dayofyear.values, times.hour.values
            if hours is None:
                hours = sorted(set(hour.tolist()))
                arr = np.zeros((366, len(hours), C, H, W), np.float32)
                cnt = np.zeros((366, len(hours)), np.int32)
            hidx = np.array([hours.index(h) for h in hour])
            fields = d["fields"]
            for i0 in range(0, len(times), 64):
                f = np.asarray(fields[i0:i0 + 64], dtype=np.float32)
                for j in range(f.shape[0]):
                    arr[doy[i0 + j] - 1, hidx[i0 + j]] += f[j]
                    cnt[doy[i0 + j] - 1, hidx[i0 + j]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        arr /= np.where(cnt > 0, cnt, np.nan).astype(np.float32)[:, :, None, None, None]
    # circular nan-aware running mean over the day of year, one channel at a time
    if window_days > 0:
        w, L = window_days, 2 * window_days + 1
        valid = np.isfinite(arr)
        for c in range(C):
            a, v = arr[:, :, c], valid[:, :, c]
            pad = np.concatenate([a[-w:], a, a[:w]], 0)
            padv = np.concatenate([v[-w:], v, v[:w]], 0)
            num = np.cumsum(np.concatenate([np.zeros_like(pad[:1]), np.nan_to_num(pad)], 0), 0, dtype=np.float64)
            den = np.cumsum(np.concatenate([np.zeros_like(padv[:1]), padv], 0), 0, dtype=np.float64)
            s, n = num[L:] - num[:-L], den[L:] - den[:-L]
            arr[:, :, c] = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
    days = np.arange(1, 367)
    coords = {"dayofyear": days, "hour": np.array(hours), "latitude": np.asarray(lat), "longitude": np.asarray(lon),
              "level": np.array(contract.LEVELS_HPA)}
    data_vars = {}
    for v in contract.UPPER_AIR_VARS + contract.SURFACE_VARS:
        if v in contract.UPPER_AIR_VARS:
            sl = contract.level_slice(v)
            data_vars[WB2_NAMES[v]] = (["dayofyear", "hour", "level", "latitude", "longitude"], arr[:, :, sl.start:sl.stop])
        else:
            data_vars[WB2_NAMES[v]] = (["dayofyear", "hour", "latitude", "longitude"], arr[:, :, contract.CHANNEL_INDEX[v]])
    return xr.Dataset(data_vars, coords=coords)


def score_files(forecasts: Dict[str, xr.Dataset], truth: xr.Dataset, climatology: Optional[xr.Dataset] = None,
                regions: Sequence[str] = ("global",)) -> pd.DataFrame:
    """{model: WB2 forecast dataset} -> long table (model, region, metric, variable, lead_hours, value) of the
    headline variables (z500, t850, q700, u850, v850, t2m, msl) with pg-flow's ``score_forecast``."""
    truth_h = select_headline(truth)
    clim_h = select_headline(climatology) if climatology is not None else None
    out = {}
    for name, ds in forecasts.items():
        out[name] = wb2_scores.score_forecast(select_headline(ds), truth_h, climatology=clim_h,
                                              region_names=list(regions))
    return _to_frame(out)


def _to_frame(scores: Dict[str, xr.Dataset]) -> pd.DataFrame:
    rows = []
    for model, ds in scores.items():
        df = ds.to_dataframe().reset_index()
        keep = ["region", "metric", LEAD] + list(ds.data_vars)
        df = df[keep].melt(id_vars=["region", "metric", LEAD], var_name="variable", value_name="value")
        df["model"] = model
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["lead_hours"] = (df.pop(LEAD) / np.timedelta64(1, "h")).astype(float)
    return df[["model", "region", "metric", "variable", "lead_hours", "value"]]


def table(df: pd.DataFrame, metric: str, variables: Sequence[str] = ("z500", "t850", "t2m"),
          leads: Sequence[float] = (24, 72, 144), region: str = "global") -> pd.DataFrame:
    """Wide table model x (variable, lead) of one metric."""
    d = df[(df.metric == metric) & (df.region == region) & df.variable.isin(variables) & df.lead_hours.isin(leads)]
    piv = d.pivot_table(index="model", columns=["variable", "lead_hours"], values="value")
    return piv.reindex(columns=[(v, float(l)) for v in variables for l in leads if (v, float(l)) in piv.columns])


def markdown(df: pd.DataFrame, metric: str, fmt: str = "{:.3g}", **kw) -> str:
    piv = table(df, metric, **kw)
    cols = [f"{v} {int(l)}h" for v, l in piv.columns]
    lines = ["| model | " + " | ".join(cols) + " |", "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for m, row in piv.iterrows():
        lines.append(f"| {m} | " + " | ".join(fmt.format(v) if np.isfinite(v) else "-" for v in row.values) + " |")
    return "\n".join(lines)
