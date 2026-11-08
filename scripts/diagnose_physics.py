#!/usr/bin/env python
"""Physics diagnostics of trained fields on validation rollouts (pg-flow step 16 gates), for every arm alike:

    residual   RMS of the strong primitive-equation residual R_j / b_j (1 = ERA5's level, b_j from ERA5 with
               centred 12 h differences) with the field's own tendency, at the IC and along the rollout
    energy     kinetic energy of the rollout state / of the truth (step 3 amplitude damping: < 0.95 fails)
    amplitude  RMS 6 h increment of the rollout / of the truth
    spectrum   power of the z500 field by total wavenumber l (rollout / truth) in three bands

    python scripts/diagnose_physics.py --kai data/era5_5p625/kai6h --run runs/step1 --tags climate_ode sfno_l2 sfno_h1_pinn sfno_w2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ffc  # noqa: E402,F401
from ffc import data as D  # noqa: E402
from ffc.physics import PhysicsLossConfig, PhysicsResidual  # noqa: E402
from ffc.sobolev import SphericalSobolev  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402
from pgcfm.coupling import LEVEL_DYN  # noqa: E402
from pgcfm.dyncore import kinetic_energy  # noqa: E402
from pgcfm.runs import device_from  # noqa: E402

sys.path.insert(0, HERE)
from predict import load_model  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS
Z500 = LEVEL_DYN["z"][contract.LEVELS_HPA.index(500)]
BANDS = {"l1-4": (1, 5), "l5-12": (5, 13), "l13-31": (13, 32)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kai", required=True)
    p.add_argument("--run", default="runs/step1")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--leads", type=int, nargs="+", default=[4, 12, 24], help="rollout steps reported (6 h each)")
    p.add_argument("--cache", default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = device_from("cpu")
    rng = np.random.default_rng(a.seed)
    stats = os.path.join(a.run, "stats")
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    lat, lon = D.grid_of(a.kai, a.val_year)
    H, W = len(lat), len(lon)
    info0 = json.load(open(os.path.join(a.run, f"{a.tags[0]}.json")))
    train_years = info0["args"]["train_years"]
    data = D.load_years(a.kai, train_years + [a.val_year], mean, std, cache=a.cache)
    t6v = torch.tensor(info0["t6"])
    w = D.lat_weights(lat).view(1, 1, -1, 1)
    phys = PhysicsResidual(H, W, mean, std, t6v, PhysicsLossConfig(), dt_hours=6.0)
    triples = []
    for _ in range(4):
        x0, d, orog, _ = D.sample(data, train_years, 2, rng, 8)
        triples.append((x0, x0 + d[:, 0], x0 + d[:, 1], orog))
    phys.calibrate(triples)
    ref = phys.reference_levels(triples)
    sob = SphericalSobolev(H, W, s=0.0)
    idx = D.fixed_windows(len(data[a.val_year]), a.K, a.n_val)
    vx0, vd, vorog, vh = D.gather(data, [a.val_year] * len(idx), idx, a.K)
    truth = vx0[:, None] + vd
    tisr = vx0[:, None, TISR_DYN] + vd[:, :, TISR_DYN]
    sd, mn = torch.as_tensor(std[DYN], dtype=torch.float32), torch.as_tensor(mean[DYN], dtype=torch.float32)
    results = {"scales": phys.scales, **ref, "leads_hours": [6 * k for k in a.leads], "arms": {}}

    def ke(x):  # (N, 66, H, W) -> mean KE
        return float(kinetic_energy(x, sd, mn, w[0]).mean())

    def band_power(z):  # (N, H, W) physical z500 -> {band: mean power}
        e = sob.energy_by_degree(z)  # (N, L)
        return {b: float(e[:, lo:hi].sum(-1).mean()) for b, (lo, hi) in BANDS.items()}

    truth_ke = {k: ke(truth[:, k - 1]) for k in a.leads}
    truth_amp = {k: float(((truth[:, k - 1] - (vx0 if k == 1 else truth[:, k - 2])) / t6v.view(1, -1, 1, 1)).square().mean().sqrt()) for k in a.leads}
    truth_pw = {k: band_power(truth[:, k - 1, Z500] * sd[Z500] + mn[Z500]) for k in a.leads}
    for tag in a.tags:
        model, info, *_ = load_model(a.run, tag, a.kai, a.val_year, dev)
        with torch.no_grad():
            y = torch.cat([model.rollout(vx0[i:i + 2], vh[i:i + 2], a.K, tisr[i:i + 2]) for i in range(0, len(vx0), 2)])
            r = {"n_params": info["n_params"], "residual_over_era5": {}, "ke_ratio": {}, "amplitude_ratio": {},
                 "z500_power_ratio": {}, "z500_rmse": {}}
            f0 = model.tendency(vx0, vh)
            r["residual_over_era5"]["ic"] = phys.per_equation(vx0, vorog, f0)
            for k in a.leads:
                xs, xo = y[:, k - 1], truth[:, k - 1]
                f = model.tendency(xs, vh + 6.0 * k)
                r["residual_over_era5"][f"{6 * k}h"] = phys.per_equation(xs, vorog, f, x_obs=xo)
                r["ke_ratio"][f"{6 * k}h"] = ke(xs) / truth_ke[k]
                prev = vx0 if k == 1 else y[:, k - 2]
                r["amplitude_ratio"][f"{6 * k}h"] = float(((xs - prev) / t6v.view(1, -1, 1, 1)).square().mean().sqrt()) / truth_amp[k]
                pw = band_power(xs[:, Z500] * sd[Z500] + mn[Z500])
                r["z500_power_ratio"][f"{6 * k}h"] = {b: pw[b] / truth_pw[k][b] for b in BANDS}
                r["z500_rmse"][f"{6 * k}h"] = float((((xs - xo)[:, Z500] * sd[Z500]) ** 2 * w[0]).mean().sqrt())
        results["arms"][tag] = r
        print(tag, json.dumps({q: r[q] for q in ("ke_ratio", "z500_rmse")}), flush=True)
        print("   residual", {k: {q: round(v, 2) for q, v in d.items()} for k, d in r["residual_over_era5"].items()}, flush=True)
    out = a.out or os.path.join(a.run, "diagnose_physics.json")
    json.dump(results, open(out, "w"), indent=1, default=float)
    # markdown
    lines = ["| arm | " + " | ".join(f"R_{q}/ERA5 IC" for q in ("u", "v", "T", "hyd")) + " | "
             + " | ".join(f"R_{q}/ERA5 {6 * a.leads[-1]}h" for q in ("u", "v", "T", "hyd"))
             + " | KE ratio " + " / ".join(f"{6 * k}h" for k in a.leads) + " | amp ratio " + " / ".join(f"{6 * k}h" for k in a.leads)
             + " | z500 power l1-4 / l5-12 / l13-31 at " + f"{6 * a.leads[-1]}h |",
             "| --- | " + " | ".join("---" for _ in range(11)) + " |"]
    for tag, r in results["arms"].items():
        last = f"{6 * a.leads[-1]}h"
        lines.append(f"| {tag} | " + " | ".join(f"{r['residual_over_era5']['ic'][q]:.2f}" for q in ("u", "v", "T", "hyd")) + " | "
                     + " | ".join(f"{r['residual_over_era5'][last][q]:.2f}" for q in ("u", "v", "T", "hyd")) + " | "
                     + " / ".join(f"{r['ke_ratio'][f'{6 * k}h']:.3f}" for k in a.leads) + " | "
                     + " / ".join(f"{r['amplitude_ratio'][f'{6 * k}h']:.2f}" for k in a.leads) + " | "
                     + " / ".join(f"{r['z500_power_ratio'][last][b]:.2f}" for b in BANDS) + " |")
    md = "\n".join(lines)
    open(out.replace(".json", ".md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
