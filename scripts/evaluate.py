#!/usr/bin/env python
"""Score pg-flow / ffc forecast files against ERA5 with WeatherBench2 metrics (through pg-flow's wrappers).

    python scripts/evaluate.py --kai data/era5_5p625/kai6h --year 2017 \
        --forecast SFNO=runs/step1/sfno_2017.nc --forecast "climate ODE"=runs/step1/resnet_2017.nc \
        --clim-years 2010-2015 --out runs/step1/eval_2017

Writes ``scores.csv`` (model, region, metric, variable, lead_hours, value) and ``tables.md`` (RMSE, ACC, CRPS,
spread-skill of the WB2 / NeuralGCM headline variables at 24 / 72 / 144 h), and adds a persistence baseline.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ffc  # noqa: E402,F401
from ffc import data as D  # noqa: E402
from ffc import ensemble as E  # noqa: E402
from ffc import evaluate as EV  # noqa: E402


def _years(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kai", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--forecast", action="append", required=True, help="name=path (repeat)")
    p.add_argument("--clim-years", default=None, help="training years for the ACC climatology, e.g. 2010-2015")
    p.add_argument("--no-persistence", action="store_true")
    p.add_argument("--leads", default="24,72,144", help="lead hours shown in the tables")
    p.add_argument("--variables", default="z500,t850,t2m,q700,u850,msl")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lat, lon = D.grid_of(a.kai, a.year)
    # read the IC list of the first file for the truth, then score one file at a time (a 73 x 25 x 8-member
    # file is 2 GB at 64x32: only its headline channels are kept, and it is freed before the next one)
    ref = E.read_forecast(a.forecast[0].split("=", 1)[1])
    ref = {"ic_index": ref["ic_index"], "lead": ref["lead"]}
    truth = EV.truth_from_kai(a.kai, a.year, ref["ic_index"], ref["lead"], lat, lon)
    clim = None
    if a.clim_years:
        clim_path = os.path.join(a.out, f"climatology_{a.clim_years}.nc")
        if os.path.exists(clim_path):
            clim = xr.open_dataset(clim_path).load()
        else:
            clim = EV.climatology_from_kai(a.kai, _years(a.clim_years), lat, lon)
            clim.to_netcdf(clim_path)
    frames = []
    for spec in a.forecast:
        name, path = spec.split("=", 1)
        f = E.read_forecast(path)
        if not (np.array_equal(f["ic_index"], ref["ic_index"]) and np.allclose(f["lead"], ref["lead"])):
            raise ValueError(f"{path}: initial conditions / leads differ from {a.forecast[0]}")
        init = EV.init_times_of(a.kai, a.year, f["ic_index"])
        ds = EV.headline_dataset(f["forecast"], init, f["lead"], lat, lon, member_axis=f["has_members"])
        del f
        frames.append(EV.score_one(name, ds, truth, clim))
        del ds
        print(f"scored {name}", flush=True)
    if not a.no_persistence:
        pers = truth.isel(prediction_timedelta=0, drop=True).expand_dims(prediction_timedelta=truth.prediction_timedelta).transpose(
            "init_time", "prediction_timedelta", ...)
        pers = pers.assign_coords(valid_time=truth.valid_time)
        frames.append(EV.score_one("persistence", EV.select_headline(pers), truth, clim))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(a.out, "scores.csv"), index=False)
    leads = [float(x) for x in a.leads.split(",")]
    variables = a.variables.split(",")
    lines = [f"# WeatherBench2 scores, {a.year}, {len(ref['ic_index'])} ICs, grid {len(lat)}x{len(lon)}", ""]
    for metric, title in (("rmse", "RMSE"), ("acc", "ACC"), ("crps", "CRPS (ensemble) / MAE (deterministic)"),
                          ("ssr", "spread / skill (ensembles)"), ("bias", "bias")):
        if metric not in set(df.metric):
            continue
        lines += [f"## {title}", "", EV.markdown(df, metric, variables=variables, leads=leads), ""]
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
