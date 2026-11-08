#!/usr/bin/env python
"""Roll a trained dynamics field out from ERA5 initial conditions, or from pg-flow ensemble scenarios, and write
a pg-flow / KAI ``forecast.nc`` (initial_condition, lead_time, [member], channel, lat, lon), physical units.

    deterministic   python scripts/predict.py --kai data/era5_5p625/kai6h --run runs/step1 --tag sfno --year 2017
    scenarios       --members 8 --alpha 1.0      perturbed-IC stand-in members (ffc.ensemble.perturbed_members)
    pg-flow input   --pgflow-ens path/to/ensemble.nc --pgflow-lead 0   members of a pg-flow file as the ICs

Initial conditions: every ``--stride`` 6 h steps of the year at 00 UTC, ``--n-ic`` of them at most; leads 6 h ...
``--max-lead-hours``; lead 0 (the IC) is written first as pg-flow does. tisr along the rollout is the known
forcing, taken from the truth file.
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
from ffc import ensemble as E  # noqa: E402
from ffc.field import DynamicsField, FieldConfig  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402
from pgcfm.data import assemble_full_state  # noqa: E402
from pgcfm.runs import device_from  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS


def load_model(run: str, tag: str, kai: str, year: int, dev):
    info = json.load(open(os.path.join(run, f"{tag}.json")))
    stats = os.path.join(run, "stats")
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    lat, lon = D.grid_of(kai, year)
    static = D.static_features(lat, lon, D.orography(kai, year))
    cfg = FieldConfig(**info["cfg"])
    model = DynamicsField(cfg, static, torch.tensor(info["t6"]), mean, std, lat, len(lon)).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, f"{tag}.pt"), map_location=dev))
    model.eval()
    return model, info, mean, std, lat, lon


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kai", required=True)
    p.add_argument("--run", default="runs/step1")
    p.add_argument("--tag", required=True)
    p.add_argument("--year", type=int, default=2017)
    p.add_argument("--stride", type=int, default=20, help="6 h steps between ICs (20 = 5 days)")
    p.add_argument("--n-ic", type=int, default=0, help="0 = all")
    p.add_argument("--max-lead-hours", type=float, default=144.0)
    p.add_argument("--members", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.0, help="perturbation size in 6 h increments")
    p.add_argument("--pgflow-ens", default=None, help="pg-flow ensemble forecast.nc whose members are the ICs")
    p.add_argument("--pgflow-lead", type=int, default=0, help="lead index of the pg-flow file to start from")
    p.add_argument("--out", default=None, help="output forecast.nc (default <run>/<tag>_<year>[_mM].nc)")
    p.add_argument("--chunk", type=int, default=4, help="ICs per rollout batch")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = device_from(a.device)
    rng = np.random.default_rng(a.seed)
    model, info, mean, std, lat, lon = load_model(a.run, a.tag, a.kai, a.year, dev)
    train_years = info["args"]["train_years"]
    data = D.load_years(a.kai, [a.year] + (train_years if a.members > 1 and a.pgflow_ens is None else []), mean, std)
    x_year = data[a.year]
    steps = int(round(a.max_lead_hours / 6.0))
    lead_hours = [0.0] + [6.0 * (k + 1) for k in range(steps)]
    times = D.times_of(a.kai, a.year)

    if a.pgflow_ens is not None:
        src = E.read_members(a.pgflow_ens, a.pgflow_lead, mean, std)
        ics = [int(i) for i in src["ic_index"]]
        members0 = src["members"].to(dev)  # (N, M, 66, H, W)
        label = f"pg-flow members (lead {src['lead_hours']:g} h)"
    else:
        ics = [i for i in range(0, len(x_year) - steps, a.stride) if int(times[i].astype("datetime64[h]").astype(int) % 24) == 0]
        if a.n_ic:
            ics = ics[: a.n_ic]
        x0 = torch.stack([x_year[i][DYN] for i in ics]).to(dev)
        members0 = E.perturbed_members(x0, data, train_years, a.members, a.alpha, rng) if a.members > 1 else x0[:, None]
        label = "ERA5 analysis" if a.members == 1 else f"perturbed-IC stand-in members (alpha {a.alpha:g})"
    N, M = members0.shape[:2]
    out = np.empty((N, len(lead_hours), M, contract.N_CHANNELS, len(lat), len(lon)), dtype=np.float32)
    mean_t, std_t = (torch.as_tensor(q, dtype=torch.float32, device=dev).view(1, 1, -1, 1, 1) for q in (mean, std))
    with torch.no_grad():
        for c in range(0, N, a.chunk):
            sl = slice(c, min(N, c + a.chunk))
            idx = ics[sl]
            x0 = members0[sl]
            seq = torch.stack([x_year[i:i + steps + 1] for i in idx]).to(dev)  # (b, steps+1, 67, H, W) truth
            tisr = seq[:, 1:, contract.CHANNEL_INDEX["tisr"]]
            orog = seq[:, 0, contract.STATIC_CHANNELS]
            hour0 = torch.tensor([6.0 * i for i in idx], device=dev)
            y = model.rollout_members(x0, hour0, steps, tisr)  # (b, M, steps, 66, H, W)
            full0 = assemble_full_state(x0, orog[:, None].expand(-1, M, -1, -1, -1))
            fully = assemble_full_state(y, orog[:, None, None].expand(-1, M, steps, -1, -1, -1))
            allx = torch.cat([full0[:, :, None], fully], 2)  # (b, M, L, 67, H, W)
            phys = allx * (std_t + contract.KAI_STD_EPS) + mean_t
            out[sl] = phys.permute(0, 2, 1, 3, 4, 5).cpu().numpy()
            print(f"  ICs {sl.start}-{sl.stop} / {N}", flush=True)
    if M == 1:
        out = out[:, :, 0]
    path = a.out or os.path.join(a.run, f"{a.tag}_{a.year}" + (f"_m{M}" if M > 1 else "") + ".nc")
    E.write_forecast(path, out, ics, lead_hours, lat, lon, init_times=times[ics],
                     title=f"ffc dynamics field {a.tag}; ICs: {label}")
    print(f"wrote {path}: {N} ICs x {len(lead_hours)} leads" + (f" x {M} members" if M > 1 else ""))


if __name__ == "__main__":
    main()
