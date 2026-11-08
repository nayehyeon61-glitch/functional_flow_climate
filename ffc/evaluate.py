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
        out = np.stack([np.stack([np.asarray(fields[i + s], dtype=np.float32) for s in steps]) for i in ic_index])
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
    sums, counts = {}, {}
    for y in years:
        with nc.Dataset(os.path.join(kai, f"{y}.nc")) as d:
            t = d["time"]
            times = pd.DatetimeIndex(nc.num2date(t[:], units=t.units, calendar=getattr(t, "calendar", "standard"),
                                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True))
            f = np.asarray(d["fields"][:], dtype=np.float64)
        doy, hour = times.dayofyear.values, times.hour.values
        for dd, hh in {(a, b) for a, b in zip(doy, hour)}:
            m = (doy == dd) & (hour == hh)
            sums[(dd, hh)] = sums.get((dd, hh), 0.0) + f[m].sum(0)
            counts[(dd, hh)] = counts.get((dd, hh), 0) + int(m.sum())
    days = sorted({k[0] for k in sums})
    hours = sorted({k[1] for k in sums})
    arr = np.full((len(days), len(hours), contract.N_CHANNELS, len(lat), len(lon)), np.nan)
    for i, dd in enumerate(days):
        for j, hh in enumerate(hours):
            if (dd, hh) in sums:
                arr[i, j] = sums[(dd, hh)] / counts[(dd, hh)]
    # circular smoothing over the day of year
    if window_days > 0:
        k = np.ones(2 * window_days + 1) / (2 * window_days + 1)
        pad = np.concatenate([arr[-window_days:], arr, arr[:window_days]], 0)
        sm = np.empty_like(arr)
        for i in range(len(days)):
            sm[i] = np.nanmean(pad[i:i + 2 * window_days + 1], 0)
        arr = sm
    coords = {"dayofyear": np.array(days), "hour": np.array(hours), "latitude": np.asarray(lat), "longitude": np.asarray(lon),
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


def headline_channels():
    """[(label, KAI channel index)] of the headline variables (z500, t850, q700, u850, v850, t2m, msl) present in
    the 67-channel contract."""
    out = []
    for var, level in HEADLINE:
        short = {v: k for k, v in WB2_NAMES.items()}[var]
        if level is None:
            if short in contract.CHANNEL_INDEX:
                out.append((headline_label(var, level), contract.CHANNEL_INDEX[short]))
        elif level in contract.LEVELS_HPA:
            out.append((headline_label(var, level), contract.level_slice(short).start + contract.LEVELS_HPA.index(level)))
    return out


def headline_dataset(x: np.ndarray, init_times, lead_hours, lat, lon, member_axis: bool) -> xr.Dataset:
    """(N, L, [M], 67, H, W) -> flat WB2 dataset of the headline variables only (what ``select_headline`` yields),
    built from a channel subset so a large ensemble file never exists as a 67-channel xarray."""
    init = np.asarray(init_times, dtype="datetime64[ns]")
    lead = (np.asarray(lead_hours, dtype=np.float64) * 3600e9).astype("timedelta64[ns]")
    dims = ["init_time", LEAD] + ([REALIZATION] if member_axis else []) + ["latitude", "longitude"]
    coords = {"init_time": init, LEAD: lead, "latitude": np.asarray(lat), "longitude": np.asarray(lon)}
    if member_axis:
        coords[REALIZATION] = np.arange(x.shape[2])
    c_axis = 3 if member_axis else 2
    ds = xr.Dataset({lab: (dims, np.ascontiguousarray(np.take(x, c, axis=c_axis))) for lab, c in headline_channels()},
                    coords=coords)
    return ds.assign_coords(valid_time=ds.init_time + ds[LEAD])


def score_one(name: str, ds_headline: xr.Dataset, truth: xr.Dataset, climatology: Optional[xr.Dataset] = None,
              regions: Sequence[str] = ("global",)) -> pd.DataFrame:
    """Score one model given as a headline dataset (``headline_dataset``) with pg-flow's ``score_forecast``."""
    clim_h = select_headline(climatology) if climatology is not None else None
    sc = wb2_scores.score_forecast(ds_headline, select_headline(truth), climatology=clim_h, region_names=list(regions))
    return _to_frame({name: sc})


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
