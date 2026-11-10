#!/usr/bin/env python
"""How good is a physics core's 6 h increment on its own? (Step 4 / Step 5 core check.)

For ``--n`` ERA5 states of ``--year`` (evenly spaced, all four UTC hours), the core's increment D(x0) is compared
with ERA5's own 6 h change d = x(6 h) - x0: latitude-weighted RMSE of x0 + D against x(6 h) (persistence = RMS
of d) and the correlation of D with d, per channel. With ``--alpha`` the per-channel least-squares weight
alpha = <D, d> / <D, D> (clipped to [0, 1], as ``DynamicsField.calibrate_core``) is reported too.

    python scripts/validate_core.py --kai data/era5_5p625/kai6h --year 2016 --core speedy --n 16
    python scripts/validate_core.py --core dinosaur --n 16          # the Step 4 backend, same states
"""

import argparse
import json
import os
import sys
import time

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ffc import data as D  # noqa: E402
from pgcfm import contract  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS
NAMES = [contract.CHANNEL_NAMES[c] for c in DYN]
REPORT = ["z500", "z250", "z850", "t850", "t500", "u850", "u250", "v500", "q700", "q850", "sp"]


def build(a, mean, std, lat, orog):
    if a.core == "dinosaur":
        from ffc.physcore import CoreConfig, SpectralCore
        return SpectralCore(CoreConfig(truncation=21, layers=a.layers, dt_minutes=a.dt_min, dfi_hours=a.dfi_h),
                            mean, std, lat, orog)
    from ffc.speedycore import SpeedyConfig, SpeedyCore
    return SpeedyCore(SpeedyConfig(layers=a.layers, dt_minutes=a.dt_min, physics=a.physics), mean, std, lat, orog)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kai", default="data/era5_5p625/kai6h")
    p.add_argument("--stats", default="runs/step1/stats")
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--core", default="speedy", choices=["speedy", "dinosaur"])
    p.add_argument("--layers", type=int, default=18)
    p.add_argument("--dt-min", type=float, default=30.0)
    p.add_argument("--dfi-h", type=float, default=6.0, help="dinosaur only")
    p.add_argument("--physics", default="all", help="speedy only: all | none | comma list of parameterizations off")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    mean, std = np.load(os.path.join(a.stats, "global_mean.npy")), np.load(os.path.join(a.stats, "global_std.npy"))
    lat, _ = D.grid_of(a.kai, a.year)
    orog = D.orography(a.kai, a.year)
    with nc.Dataset(os.path.join(a.kai, f"{a.year}.nc")) as d:
        T = d["fields"].shape[0]
        idx = np.linspace(0, T - 2, a.n).astype(int)
        x0 = np.stack([np.asarray(d["fields"][i], np.float64) for i in idx])[:, DYN]
        x1 = np.stack([np.asarray(d["fields"][i + 1], np.float64) for i in idx])[:, DYN]
    core = build(a, mean, std, lat, orog)
    core.increment_si(x0[:1], *(([0.0],) if a.core == "speedy" else ()))   # warm-up (Julia / XLA compilation)
    t0 = time.time()
    if a.core == "speedy":
        inc = core.increment_si(x0, 6.0 * idx)
    else:
        inc = core.increment_si(x0)
    sec = (time.time() - t0) / a.n
    if hasattr(core, "close"):
        core.close()
    dtrue = x1 - x0
    w = np.cos(np.deg2rad(lat))[None, :, None]
    w = w / w.mean()
    res = {"core": a.core, "args": vars(a), "seconds_per_state": sec, "channels": {}}
    for c, name in enumerate(NAMES):
        e = inc[:, c] - dtrue[:, c]
        rmse = float(np.sqrt((e ** 2 * w).mean()))
        pers = float(np.sqrt((dtrue[:, c] ** 2 * w).mean()))
        dd = (inc[:, c] * dtrue[:, c] * w).mean()
        nn = (inc[:, c] ** 2 * w).mean()
        corr = float(dd / np.sqrt(nn * (dtrue[:, c] ** 2 * w).mean() + 1e-30))
        alpha = float(np.clip(dd / max(nn, 1e-30), 0, 1))
        res["channels"][name] = {"rmse": rmse, "persistence": pers, "corr": corr, "alpha": alpha}
    print(f"{a.core} {a.physics if a.core == 'speedy' else ''} {sec:.3f} s/state")
    print(f"{'channel':8s} {'rmse':>10s} {'persist':>10s} {'corr':>6s} {'alpha':>6s}")
    for k in REPORT:
        r = res["channels"][k]
        print(f"{k:8s} {r['rmse']:10.3f} {r['persistence']:10.3f} {r['corr']:6.2f} {r['alpha']:6.2f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
