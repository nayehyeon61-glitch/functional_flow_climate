#!/usr/bin/env python
"""Fit the scenario size of an ensemble generator on the validation year (never the test year).

For every candidate size the members are rolled through the frozen dynamics field from fixed 00 UTC initial
conditions of the validation year, and the fair ensemble CRPS (area-weighted, in units of the channel std) of
z500, t850, t2m and u850 is averaged over the leads 24, 72 and 144 h. The size with the lowest CRPS is chosen
(a proper score: it trades spread against the error of the members, unlike matching spread/skill alone);
spread/skill is reported next to it.

    stand-in (Step 1)   --generator standin --sizes 0.25 0.5 1 2
    FM (Step 2)         --generator flow --flow-run runs/step2 --flow-tag fm_ot_matched --sizes 0.5 1 2 4
    FM stochastic       --generator flow ... --kappa 2 --stochastic-sizes 0.25 0.5 1

The flow samples z do not depend on the size, so they are drawn once and only rescaled.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import ffc  # noqa: E402,F401
from ffc import data as D  # noqa: E402
from ffc import ensemble as E  # noqa: E402
from ffc import flow as F  # noqa: E402
from predict import load_flow, load_model  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.coupling import LEVEL_DYN  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS
CH = {"z500": LEVEL_DYN["z"][contract.LEVELS_HPA.index(500)], "t850": LEVEL_DYN["t"][contract.LEVELS_HPA.index(850)],
      "u850": LEVEL_DYN["u"][contract.LEVELS_HPA.index(850)], "t2m": DYN.index(contract.CHANNEL_INDEX["t2m"])}
LEADS = (24, 72, 144)


def ens_scores(y: torch.Tensor, truth: torch.Tensor, w: torch.Tensor):
    """y (B, M, K, 66, H, W), truth (B, K, 66, H, W) normalized -> {var: {crps, rmse, spread} per lead index}."""
    out = {}
    M = y.shape[1]
    for name, c in CH.items():
        ye, tr = y[:, :, :, c], truth[:, :, c]                    # (B, M, K, H, W), (B, K, H, W)
        mae = (ye - tr[:, None]).abs().mean(1)
        pair = (ye[:, :, None] - ye[:, None]).abs().sum((1, 2)) / (2 * M * (M - 1))  # fair CRPS
        crps = ((mae - pair) * w).mean((0, 2, 3))
        mean = ye.mean(1)
        rmse = (((mean - tr) ** 2 * w).mean((0, 2, 3))).sqrt()
        spread = ((ye.var(1, unbiased=True) * w).mean((0, 2, 3))).sqrt()
        out[name] = {"crps": crps.tolist(), "rmse": rmse.tolist(), "spread": spread.tolist()}
    return out


def summarize(sc, lead_idx):
    crps = float(np.mean([sc[v]["crps"][k] for v in CH for k in lead_idx]))
    ssr = {v: [sc[v]["spread"][k] / sc[v]["rmse"][k] * np.sqrt((8 + 1) / 8) for k in lead_idx] for v in CH}
    return crps, ssr


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kai", required=True)
    p.add_argument("--run", required=True, help="Step 1 run folder (frozen field)")
    p.add_argument("--tag", default="sfno_h1_pinn")
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--generator", default="flow", choices=["flow", "standin"])
    p.add_argument("--flow-run", default=None)
    p.add_argument("--flow-tag", default=None)
    p.add_argument("--sizes", type=float, nargs="+", default=[0.5, 1, 2, 4])
    p.add_argument("--kappa", type=float, default=None, help="fixed IC size when calibrating --stochastic-sizes")
    p.add_argument("--stochastic-sizes", type=float, nargs="*", default=[])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--n-ic", type=int, default=16)
    p.add_argument("--chunk", type=int, default=2)
    p.add_argument("--out", required=True, help="json with the scores of every size and the chosen one")
    p.add_argument("--cache", default=None, help="memory-mapped normalized years (pgcfm.runs.load_years)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")
    field, info, mean, std, lat, lon = load_model(a.run, a.tag, a.kai, a.year, dev)
    train_years = info["args"]["train_years"]
    data = D.load_years(a.kai, [a.year] + (train_years if a.generator == "standin" else []), mean, std, cache=a.cache)
    xy = data[a.year]
    steps = max(LEADS) // 6
    lead_idx = [h // 6 - 1 for h in LEADS]
    times = D.times_of(a.kai, a.year)
    cands = [i for i in range(0, len(xy) - steps - 1) if int(times[i].astype("datetime64[h]").astype(int) % 24) == 0]
    ics = [cands[int(j)] for j in np.linspace(3, len(cands) - 4, a.n_ic).astype(int)]  # offset from the test-like grid
    w = D.lat_weights(lat).view(1, -1, 1)
    flow = load_flow(a.flow_run, a.flow_tag, field, dev)[0] if a.generator == "flow" else None
    stoch = bool(a.stochastic_sizes)
    sizes = a.stochastic_sizes if stoch else a.sizes
    acc = {s: [] for s in sizes}
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        for c in range(0, len(ics), a.chunk):
            idx = ics[c:c + a.chunk]
            seq = torch.stack([xy[i:i + steps + 1] for i in idx])
            x0, truth = seq[:, 0, DYN], seq[:, 1:, DYN]
            tisr = seq[:, 1:, contract.CHANNEL_INDEX["tisr"]]
            hour0 = torch.tensor([6.0 * i for i in idx])
            if a.generator == "flow":
                base = F.flow_members(flow, field, x0, hour0, a.members, 1.0, gen)  # kappa 1: x0 + sigma_e z
                dz = base - x0[:, None]
            else:
                base = E.perturbed_members(x0, data, train_years, a.members, 1.0, rng)
                dz = base - x0[:, None]
            for s in sizes:
                if stoch:
                    mem = x0[:, None] + a.kappa * dz
                    gs = torch.Generator().manual_seed(a.seed + 1 + c)
                    y = F.rollout_stochastic(flow, field, mem, hour0, steps, s, tisr, gs)
                else:
                    y = field.rollout_members(x0[:, None] + s * dz, hour0, steps, tisr)
                acc[s].append((y[:, :, lead_idx].clone(), truth[:, lead_idx].clone()))
            print(f"  ICs {c + len(idx)}/{len(ics)}", flush=True)
    res = {"generator": a.generator, "flow_tag": a.flow_tag, "field": a.tag, "year": a.year, "ics": ics,
           "stochastic": stoch, "kappa": a.kappa, "leads": LEADS, "scores": {}}
    best = None
    for s in sizes:
        y = torch.cat([q[0] for q in acc[s]])
        tr = torch.cat([q[1] for q in acc[s]])
        sc = ens_scores(y, tr, w)
        crps, ssr = summarize(sc, range(len(LEADS)))
        res["scores"][str(s)] = {"crps_mean": crps, "ssr": ssr, "detail": sc}
        print(f"size {s:g}: mean CRPS {crps:.4f}  SSR z500 " + " ".join(f"{v:.2f}" for v in ssr["z500"])
              + "  t850 " + " ".join(f"{v:.2f}" for v in ssr["t850"]), flush=True)
        if best is None or crps < best[1]:
            best = (s, crps)
    res["best"] = best[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"chosen size {best[0]:g} -> {a.out}")


if __name__ == "__main__":
    main()
