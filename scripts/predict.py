#!/usr/bin/env python
"""Roll a trained dynamics field out from ERA5 initial conditions, or from pg-flow ensemble scenarios, and write
a pg-flow / KAI ``forecast.nc`` (initial_condition, lead_time, [member], channel, lat, lon), physical units.

    deterministic   python scripts/predict.py --kai data/era5_5p625/kai6h --run runs/step1 --tag sfno --year 2017
    scenarios       --members 8 --alpha 1.0      perturbed-IC stand-in members (ffc.ensemble.perturbed_members)
    pg-flow input   --pgflow-ens path/to/ensemble.nc --pgflow-lead 0   members of a pg-flow file as the ICs
    Step 2 FM       --members 8 --flow-run runs/step2 --flow-tag fm_ot_matched --kappa 2.0 [--kappa-s 0.5]
                    scenarios from the flow-matching generator (ffc.flow) in front of the frozen field;
                    --kappa-s > 0 also adds a fresh flow error sample after every 6 h step (stochastic dynamics)

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
from ffc import flow as F  # noqa: E402
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
    orog = D.orography(kai, year)
    static = D.static_features(lat, lon, orog)
    cfg = FieldConfig(**info["cfg"])
    model = DynamicsField(cfg, static, torch.tensor(info["t6"]), mean, std, lat, len(lon), orog_si=orog).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, f"{tag}.pt"), map_location=dev))
    model.eval()
    return model, info, mean, std, lat, lon


def load_flow(run: str, tag: str, field, dev):
    """Step 2 generator trained by scripts/train_flow.py on the errors of ``field``."""
    info = json.load(open(os.path.join(run, f"{tag}.json")))
    st = torch.load(os.path.join(run, f"error_stats_{info['field_tag']}.pt"))
    lat = None  # unused by ErrorFlow beyond the grid shape
    flow = F.ErrorFlow(F.FlowConfig(**info["cfg"]), field.static.cpu(), st["sigma_e"], st["spectrum"], lat,
                       field.static.shape[-1]).to(dev)
    flow.load_state_dict(torch.load(os.path.join(run, f"{tag}.pt"), map_location=dev))
    flow.eval()
    return flow, info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kai", required=True)
    p.add_argument("--run", default="runs/step1")
    p.add_argument("--tag", required=True)
    p.add_argument("--year", type=int, default=2017)
    p.add_argument("--stride", type=int, default=20, help="6 h steps between ICs (20 = 5 days)")
    p.add_argument("--n-ic", type=int, default=0, help="0 = all")
    p.add_argument("--max-lead-hours", type=float, default=144.0)
    p.add_argument("--lead-every-hours", type=float, default=6.0, help="leads kept in the file (rollout stays 6 h)")
    p.add_argument("--members", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.0, help="perturbation size in 6 h increments")
    p.add_argument("--pgflow-ens", default=None, help="pg-flow ensemble forecast.nc whose members are the ICs")
    p.add_argument("--pgflow-lead", type=int, default=0, help="lead index of the pg-flow file to start from")
    p.add_argument("--flow-run", default=None, help="Step 2 run folder of the flow-matching generator")
    p.add_argument("--flow-tag", default=None)
    p.add_argument("--kappa", type=float, default=1.0, help="scenario size in flow error samples")
    p.add_argument("--kappa-s", type=float, default=0.0, help="stochastic flow error added after every 6 h step")
    p.add_argument("--out", default=None, help="output forecast.nc (default <run>/<tag>_<year>[_mM].nc)")
    p.add_argument("--chunk", type=int, default=4, help="ICs per rollout batch")
    p.add_argument("--cache", default=None, help="memory-mapped normalized years of the training run (pgcfm.runs.load_years)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = device_from(a.device)
    rng = np.random.default_rng(a.seed)
    model, info, mean, std, lat, lon = load_model(a.run, a.tag, a.kai, a.year, dev)
    train_years = info["args"]["train_years"]
    flow = None
    if a.flow_tag:
        flow, _ = load_flow(a.flow_run, a.flow_tag, model, dev)
    standin = a.members > 1 and a.pgflow_ens is None and flow is None
    NH = model.n_history   # Step 6 sequence fields: the states before each IC (from the previous year in January)
    prev = [a.year - 1] if NH and os.path.exists(os.path.join(a.kai, f"{a.year - 1}.nc")) else []
    data = D.load_years(a.kai, [a.year] + prev + [y for y in (train_years if standin else []) if y not in prev],
                        mean, std, cache=a.cache)
    x_year = data[a.year]

    def past_of(i):
        if i >= NH:
            return x_year[i - NH:i][:, DYN]
        if not prev:
            raise ValueError(f"IC {i} needs {NH} earlier states: add {a.year - 1}.nc or skip it")
        return torch.cat([data[a.year - 1][len(data[a.year - 1]) - (NH - i):], x_year[:i]])[:, DYN]
    steps = int(round(a.max_lead_hours / 6.0))
    keep = [k for k in range(steps) if (6.0 * (k + 1)) % a.lead_every_hours == 0]  # rollout steps written
    lead_hours = [0.0] + [6.0 * (k + 1) for k in keep]
    times = D.times_of(a.kai, a.year)

    if a.pgflow_ens is not None:
        src = E.read_members(a.pgflow_ens, a.pgflow_lead, mean, std)
        # the members at lead L are valid L hours after the file's ICs: the rollout, its tisr forcing, the time
        # features and the written initial_condition / init_time all start at that valid time
        shift = int(round(src["lead_hours"] / 6.0))
        ics = [int(i) + shift for i in src["ic_index"]]
        members0 = src["members"].to(dev)  # (N, M, 66, H, W)
        if steps > 0:
            ok = [k for k, i in enumerate(ics) if i + steps < len(x_year)]
            if len(ok) < len(ics):
                print(f"dropping {len(ics) - len(ok)} ICs whose {a.max_lead_hours:g} h rollout runs past the year", flush=True)
                ics, members0 = [ics[k] for k in ok], members0[ok]
        label = f"pg-flow members at lead {src['lead_hours']:g} h (ICs shifted by {shift} steps to that valid time)"
    else:
        ics = [i for i in range(0, len(x_year) - steps, a.stride) if int(times[i].astype("datetime64[h]").astype(int) % 24) == 0]
        if a.n_ic:
            ics = ics[: a.n_ic]
        x0 = torch.stack([x_year[i][DYN] for i in ics]).to(dev)
        if flow is not None:
            members0 = None  # drawn per chunk (needs the operator increment)
            label = f"flow-matching scenarios {a.flow_tag} (kappa {a.kappa:g}, kappa_s {a.kappa_s:g})"
        else:
            members0 = E.perturbed_members(x0, data, train_years, a.members, a.alpha, rng) if a.members > 1 else x0[:, None]
            label = "ERA5 analysis" if a.members == 1 else f"perturbed-IC stand-in members (alpha {a.alpha:g})"
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    if members0 is None:
        N, M = len(ics), a.members
    else:
        N, M = members0.shape[:2]
    path = a.out or os.path.join(a.run, f"{a.tag}_{a.year}" + (f"_m{M}" if M > 1 else "") + ".nc")
    writer = E.ForecastWriter(path, N, lead_hours, M if M > 1 else None, lat, lon,
                              title=f"ffc dynamics field {a.tag}; ICs: {label}")
    mean_t, std_t = (torch.as_tensor(q, dtype=torch.float32, device=dev).view(1, 1, -1, 1, 1) for q in (mean, std))
    keep_t = torch.tensor(keep, device=dev)
    with torch.no_grad():
        for c in range(0, N, a.chunk):
            sl = slice(c, min(N, c + a.chunk))
            idx = ics[sl]
            seq = torch.stack([x_year[i:i + steps + 1] for i in idx]).to(dev)  # (b, steps+1, 67, H, W) truth
            tisr = seq[:, 1:, contract.CHANNEL_INDEX["tisr"]]
            orog = seq[:, 0, contract.STATIC_CHANNELS]
            hour0 = torch.tensor([6.0 * i for i in idx], device=dev)
            if members0 is None:
                x0 = F.flow_members(flow, model, seq[:, 0, DYN], hour0, M, a.kappa, gen, tisr_next=tisr[:, 0])
            else:
                x0 = members0[sl]
            if flow is not None and a.kappa_s > 0:
                y = F.rollout_stochastic(flow, model, x0, hour0, steps, a.kappa_s, tisr, gen)[:, :, keep_t]
            else:
                past = torch.stack([past_of(i) for i in idx]).to(dev) if NH else None
                y = model.rollout_members(x0, hour0, steps, tisr, past)[:, :, keep_t]  # (b, M, L-1, 66, H, W)
            full0 = assemble_full_state(x0, orog[:, None].expand(-1, M, -1, -1, -1))
            fully = assemble_full_state(y, orog[:, None, None].expand(-1, M, len(keep), -1, -1, -1))
            allx = torch.cat([full0[:, :, None], fully], 2)  # (b, M, L, 67, H, W)
            phys = (allx * (std_t + contract.KAI_STD_EPS) + mean_t).permute(0, 2, 1, 3, 4, 5).cpu().numpy()
            writer.write(sl, phys if M > 1 else phys[:, :, 0], idx, times[idx])
            del y, fully, allx, phys
            print(f"  ICs {sl.start}-{sl.stop} / {N}", flush=True)
    writer.close()
    print(f"wrote {path}: {N} ICs x {len(lead_hours)} leads" + (f" x {M} members" if M > 1 else ""))


if __name__ == "__main__":
    main()
