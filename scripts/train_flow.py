#!/usr/bin/env python
"""Stage 2: train the flow-matching ensemble generator on the errors of the frozen Step 1 dynamics field.

    target  z = (x_{6h}^{ERA5} - Phi_theta(x0)) / sigma_e     Phi_theta: one 6 h step of the frozen field (Step 1)
    model   p(z | x0, Phi_theta(x0) - x0, time) by conditional flow matching (ffc.flow.ErrorFlow): SFNO velocity,
            Gaussian-random-field source on S^2, minibatch OT coupling in H^s, Sobolev velocity loss

    python scripts/train_flow.py --kai data/era5_5p625/kai6h --field-run runs/step1 --field-tag sfno_h1_pinn \
        --out runs/step2 --coupling ot --source matched

The field's weights are never updated (requires_grad off, eval mode). sigma_e and the target degree spectrum are
measured once on random training windows and stored in <tag>.json. Resumable (pgcfm.runs.fit).
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ffc  # noqa: E402,F401
from ffc import data as D  # noqa: E402
from ffc.flow import ErrorFlow, FlowConfig, operator_increment  # noqa: E402
from ffc.sobolev import SphericalSobolev  # noqa: E402
from predict import load_model  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402
from pgcfm.coupling import LEVEL_DYN  # noqa: E402
from pgcfm.runs import EXIT_BUDGET, FitConfig, Prefetch, device_from, fit, run_record  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kai", required=True)
    p.add_argument("--field-run", required=True, help="Step 1 run folder with <tag>.pt, <tag>.json, stats/")
    p.add_argument("--field-tag", default="sfno_h1_pinn")
    p.add_argument("--out", default="runs/step2")
    p.add_argument("--tag", default=None)
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--coupling", default="ot", choices=["ot", "indep"])
    p.add_argument("--source", default="matched", choices=["matched", "sobolev", "white"])
    p.add_argument("--source-s", type=float, default=1.0)
    p.add_argument("--loss-s", type=float, default=0.0)
    p.add_argument("--ot-s", type=float, default=0.0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--modes-lat", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=4)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--n-val", type=int, default=32)
    p.add_argument("--n-stats", type=int, default=256, help="training windows for sigma_e and the spectrum")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--cache", default=None)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.iters, a.batch, a.width, a.layers, a.modes_lat, a.n_val, a.val_every, a.n_stats = 4, 4, 8, 1, 8, 4, 2, 8
    return a


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = device_from(a.device)
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"fm_{a.coupling}_{a.source}" + ("_quick" if a.quick else "")
    log = lambda s: print(s, flush=True)  # noqa: E731

    field, finfo, mean, std, lat, lon = load_model(a.field_run, a.field_tag, a.kai, a.val_year, dev)
    for q in field.parameters():
        q.requires_grad_(False)
    train_years = finfo["args"]["train_years"]
    if a.quick:
        train_years = train_years[:1]
    H, W = len(lat), len(lon)
    data = D.load_years(a.kai, train_years + [a.val_year], mean, std, cache=a.cache)
    static = field.static.cpu()

    def errors(x0, d, hour0):
        """frozen operator: x0 -> Phi(x0); returns the condition increment and the 6 h error (normalized units)."""
        tisr1 = x0[:, TISR_DYN] + d[:, 0, TISR_DYN]
        x1, inc = operator_increment(field, x0, hour0, tisr1)
        return inc, (x0 + d[:, 0]) - x1

    # -- error statistics of the frozen operator ----------------------------------------------------------
    stats_path = os.path.join(a.out, f"error_stats_{a.field_tag}.pt")
    if os.path.exists(stats_path):
        st = torch.load(stats_path)
    else:
        srng = np.random.default_rng(12345)
        sq, spec, n = 0.0, 0.0, 0
        sob = SphericalSobolev(H, W, s=0.0).to(dev)
        w = D.lat_weights(lat, dev).view(1, 1, -1, 1)
        for _ in range(max(1, a.n_stats // 16)):
            x0, d, _, hour0 = (q.to(dev) for q in D.sample(data, train_years, 1, srng, 16))
            _, e = errors(x0, d, hour0)
            sq = sq + (e ** 2 * w).mean((2, 3)).sum(0)
            n += len(e)
        sigma = (sq / n).sqrt()
        sigma[TISR_DYN] = 0.0
        for _ in range(max(1, a.n_stats // 16)):
            x0, d, _, hour0 = (q.to(dev) for q in D.sample(data, train_years, 1, srng, 16))
            _, e = errors(x0, d, hour0)
            z = e / sigma.clamp_min(1e-8).view(1, -1, 1, 1)
            z[:, TISR_DYN] = 0.0
            L = sob.energy_by_degree(z)  # (B, C, L): sum_m |z_lm|^2
            spec = spec + L.sum(0)
        spec = spec / (n * (2 * torch.arange(spec.shape[-1], device=dev) + 1))  # per-coefficient variance
        spec[TISR_DYN] = spec.mean(0)  # unused channel: any valid spectrum
        st = {"sigma_e": sigma.cpu(), "spectrum": spec.cpu(), "n": n}
        torch.save(st, stats_path)
    sigma_e, spectrum = st["sigma_e"], st["spectrum"]
    # power-law fit of the mean target spectrum: E_l ~ (1 + l(l+1))^-s_fit (the Sobolev regularity of the errors)
    l = torch.arange(1, spectrum.shape[-1], dtype=torch.float64)
    m = spectrum[[i for i in range(contract.N_DYNAMIC) if i != TISR_DYN]].double().mean(0)[1:]
    s_fit = float(-np.polyfit(np.log(1 + l * (l + 1)).numpy(), np.log(m).numpy(), 1)[0])
    z500 = LEVEL_DYN["z"][contract.LEVELS_HPA.index(500)]
    log(f"[stats] sigma_e / t6: z500 {float(sigma_e[z500] / field.t6.flatten()[z500].cpu()):.3f}; "
        f"target spectrum ~ (1 + l(l+1))^-{s_fit:.2f}")

    cfg = FlowConfig(coupling=a.coupling, source=a.source, source_s=a.source_s, loss_s=a.loss_s, ot_s=a.ot_s,
                     width=a.width, n_layers=a.layers, modes_lat=a.modes_lat, sample_steps=a.sample_steps)
    flow = ErrorFlow(cfg, static, sigma_e, spectrum, lat, W).to(dev)
    res = {"args": vars(a), "cfg": dataclasses.asdict(cfg), "tag": tag, "field_tag": a.field_tag, "grid": [H, W],
           "n_params": flow.n_params(), "sigma_e": sigma_e.tolist(), "spectrum_s_fit": s_fit, "run": run_record(dev)}
    log(f"[step 2 {tag}] flow {res['n_params']} params, frozen field {a.field_tag}, coupling {a.coupling}, "
        f"source {a.source}, loss H^{a.loss_s:g}, train {train_years}")

    def batch_of(x0, d, hour0):
        x0, d, hour0 = x0.to(dev), d.to(dev), hour0.to(dev)
        inc, e = errors(x0, d, hour0)
        z1 = e / flow.sigma_e.clamp_min(1e-8) * flow.keep
        return z1, flow.condition(x0, inc, hour0)

    ck = os.path.join(a.out, f"{tag}.pt")
    gen = torch.Generator(device=dev).manual_seed(a.seed + 7)
    # fixed validation windows of the validation year, with their own noise
    n_val_time = len(data[a.val_year])
    val_idx = D.fixed_windows(n_val_time, 1, a.n_val)
    vx0, vd, _, vh = D.gather(data, [a.val_year] * len(val_idx), val_idx, 1)
    vz, vc = batch_of(vx0, vd, vh)

    def val_fn():
        g = torch.Generator(device=dev).manual_seed(999)
        with torch.no_grad():
            ls = [flow.loss(vz[i:i + a.batch], vc[i:i + a.batch], g).item() for i in range(0, len(vz), a.batch)]
        return {"loss": float(np.mean(ls))}

    if os.path.exists(ck):
        flow.load_state_dict(torch.load(ck, map_location=dev))
        log("  trained weights found: skipping training")
    else:
        opt = torch.optim.AdamW(flow.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)

        def loss_of(batch):
            x0, d, _, hour0 = batch
            z1, cond = batch_of(x0, d, hour0)
            return flow.loss(z1, cond, gen)

        batches = Prefetch(lambda r: D.pick(data, train_years, 1, r, a.batch), lambda p: D.gather(data, *p, 1), rng,
                           depth=a.prefetch)
        fcfg = FitConfig(iters=a.iters, ckpt_every=a.ckpt_every, log_every=max(1, min(50, a.iters // 10)),
                         val_every=a.val_every, select="best", max_hours=a.max_hours, clip=a.clip)
        t0 = time.time()
        hist = fit(flow, flow.parameters(), opt, sched, batches, loss_of, fcfg, ck, val_fn=val_fn,
                   generators={"fm": gen}, log=log)
        res["fit"] = {k: v for k, v in hist.items() if k != "hist"}
        res["train_seconds"] = time.time() - t0
        res["train_curve"] = [h for h in hist["hist"][:: max(1, len(hist["hist"]) // 200)]]
        if hist["stopped"] == "budget":
            json.dump(res, open(os.path.join(a.out, f"{tag}.partial.json"), "w"), indent=1, default=float)
            sys.exit(EXIT_BUDGET)
        torch.save(flow.state_dict(), ck)
    res["val_final"] = val_fn()
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1, default=float)
    log(f"[done {tag}] val FM loss {res['val_final']['loss']:.4f} -> {ck}")


if __name__ == "__main__":
    main()
