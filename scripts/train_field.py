#!/usr/bin/env python
"""Train the dynamics field G_theta (or the climate-ODE baseline) on 6 h ERA5 in the pg-flow KAI layout.

    model   dx/dt = G_theta(x, static, time); x_{k+1} = ODE(G_theta, x_k, 6 h) (torchdiffeq); tisr from the data
    loss    rollout:  |x_k^theta - x_k|^2, latitude-weighted, in 6 h tendency units, k = 1..K_cur (curriculum)
            sobolev:  lambda_sob x ||x_k^theta - x_k||_{H^s}^2 in the spherical-harmonic basis (ffc.sobolev)
            physics:  lambda_p x sum_j gamma_j max(m_j - target, 0), m_j = weighted mean (R_j / b_j)^2 of the
                      strong primitive-equation residuals with the field's own tendency at random rollout states
                      (pg-flow pgcfm.physics; b_j = ERA5's level, target 1 = ERA5's level)
    arms    --backbone sfno | fno (neural operators)  --backbone resnet (ClimODE ResNet: the climate-ODE baseline)

    python scripts/train_field.py --kai data/era5_5p625/kai6h --out runs/step1 --backbone sfno --lambda-sob 1 --lambda-p 0.05

Resumable (pg-flow ``pgcfm.runs.fit``): rerun the same command after an interruption or a --max-hours budget.
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
from ffc.field import DynamicsField, FieldConfig  # noqa: E402
from ffc.physics import PhysicsLossConfig, PhysicsResidual  # noqa: E402
from ffc.sobolev import SphericalSobolev  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN  # noqa: E402
from pgcfm.coupling import LEVEL_DYN  # noqa: E402
from pgcfm.runs import EXIT_BUDGET, FitConfig, Prefetch, autocast, device_from, fit, run_record  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS
Z500 = LEVEL_DYN["z"][contract.LEVELS_HPA.index(500)]
T850 = LEVEL_DYN["t"][contract.LEVELS_HPA.index(850)]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kai", required=True, help="folder of <year>.nc KAI files at 6 h (wb2_to_kai.py --all-hours)")
    p.add_argument("--out", default="runs/step1")
    p.add_argument("--tag", default=None)
    p.add_argument("--train-years", type=int, nargs="+", default=[2010, 2011, 2012, 2013, 2014, 2015])
    p.add_argument("--val-year", type=int, default=2016)
    # model
    p.add_argument("--backbone", default="sfno", choices=["sfno", "fno", "resnet"])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--modes-lat", type=int, default=16)
    p.add_argument("--modes-lon", type=int, default=32)
    p.add_argument("--dt-hours", type=float, default=6.0)
    p.add_argument("--solver", default="midpoint", choices=["euler", "midpoint", "rk4", "heun3"])
    p.add_argument("--no-positive", action="store_true")
    # loss
    p.add_argument("--K", type=int, default=8, help="training window (6 h steps)")
    p.add_argument("--curriculum", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--lambda-sob", type=float, default=0.0, help="weight of the H^s term (1 = same footing as L2)")
    p.add_argument("--sob-s", type=float, default=1.0, help="Sobolev order s (1: H1 gradients; -1: W2 tangent norm)")
    p.add_argument("--sob-lmax", type=int, default=0, help="0 = the grid's nlat")
    p.add_argument("--lambda-p", type=float, default=0.0)
    p.add_argument("--phys-target", type=float, default=1.0)
    p.add_argument("--phys-states", type=int, default=2)
    p.add_argument("--gamma-q", type=float, default=0.0)
    # optimisation
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--n-val", type=int, default=8, help="fixed validation windows")
    p.add_argument("--val-K", type=int, default=8)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--select", default="best", choices=["best", "last"])
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--cache", default=None, help="memory-mapped normalized years (pgcfm.runs.load_years)")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters, a.K, a.curriculum = a.train_years[:1], 6, 2, [1, 2]
        a.width, a.layers, a.modes_lat, a.n_val, a.val_K, a.val_every, a.batch = 8, 1, 8, 2, 2, 3, 2
    return a


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = device_from(a.device)
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or (f"{a.backbone}" + (f"_sob{a.sob_s:g}x{a.lambda_sob:g}" if a.lambda_sob > 0 else "")
                    + (f"_lp{a.lambda_p:g}" if a.lambda_p > 0 else "") + ("_quick" if a.quick else ""))
    log = lambda s: print(s, flush=True)  # noqa: E731

    # -- data --------------------------------------------------------------------------------------------
    mean, std = D.channel_stats(a.kai, a.train_years, os.path.join(a.out, "stats"))
    lat, lon = D.grid_of(a.kai, a.val_year)
    H, W = len(lat), len(lon)
    data = D.load_years(a.kai, a.train_years + [a.val_year], mean, std, cache=a.cache)
    t6v = D.tendency_std(data, a.train_years, DYN).clamp_min(1e-6)
    t6 = t6v.view(1, 1, -1, 1, 1).to(dev)
    w = D.lat_weights(lat, dev).view(1, 1, 1, -1, 1)
    static = D.static_features(lat, lon, D.orography(a.kai, a.val_year))

    # -- model -------------------------------------------------------------------------------------------
    cfg = FieldConfig(backbone=a.backbone, width=a.width, n_layers=a.layers, modes_lat=a.modes_lat,
                      modes_lon=a.modes_lon, dt_hours=a.dt_hours, solver=a.solver, positive=not a.no_positive)
    model = DynamicsField(cfg, static, t6v, mean, std, lat, W).to(dev)
    model.grad_ckpt = a.grad_ckpt
    res = {"args": vars(a), "cfg": dataclasses.asdict(cfg), "tag": tag, "grid": [H, W], "n_params": model.n_params(),
           "t6": t6v.tolist(), "run": run_record(dev)}

    # -- physics -----------------------------------------------------------------------------------------
    phys = None
    if a.lambda_p > 0:
        pcfg = PhysicsLossConfig(lambda_p=a.lambda_p, target=a.phys_target, states_per_sample=a.phys_states)
        pcfg.gamma["q"] = a.gamma_q
        phys = PhysicsResidual(H, W, mean, std, t6v, pcfg, dt_hours=6.0, device=dev)
        triples = []
        for _ in range(4):
            x0, d, orog, _ = (q.to(dev) for q in D.sample(data, a.train_years, 2, rng, 8))
            triples.append((x0, x0 + d[:, 0], x0 + d[:, 1], orog))
        phys.calibrate(triples)
        res["physics"] = {"scales": phys.scales, **phys.reference_levels(triples), "cfg": dataclasses.asdict(pcfg)}
        log(f"[physics] b_j {phys.scales}; rest state / ERA5 " + " ".join(f"{k} {v:.2f}" for k, v in res["physics"]["rest_over_era5"].items()))
    sob = None
    if a.lambda_sob > 0:
        sob = SphericalSobolev(H, W, s=a.sob_s, lmax=a.sob_lmax or None).to(dev)
        x0, d, _, _ = D.sample(data, a.train_years, 1, rng, 4)
        res["sobolev"] = {"s": a.sob_s, "lmax": a.sob_lmax or H, "parseval_error_on_increments": sob.parseval_error(d[:, 0].to(dev), lat)}
    log(f"[step 1 {tag}] {a.backbone} {res['n_params']} params on {H}x{W}, train {a.train_years}, val {a.val_year}, "
        f"K {a.K} curriculum {a.curriculum}, lambda_sob {a.lambda_sob} (s {a.sob_s}), lambda_p {a.lambda_p}, {res['run']['device_name']}")

    def tisr_of(x0, d, K):
        return (x0[:, None, TISR_DYN] + d[:, :K, TISR_DYN])

    # -- training ----------------------------------------------------------------------------------------
    ck = os.path.join(a.out, f"{tag}.pt")
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location=dev))
        res["resumed_final"] = True
        log("  trained weights found: skipping training")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        gen = torch.Generator().manual_seed(a.seed + 1)
        step = {"it": 0}
        if os.path.exists(ck + ".train"):
            step["it"] = torch.load(ck + ".train", weights_only=False)["it"]

        def cur_K():
            # clamped to the sampled window a.K: batches carry a.K future states, so a curriculum stage above it
            # (e.g. --K 4 with the default 1 2 4 8) would index past the gathered tisr frames
            return min(a.K, a.curriculum[min(len(a.curriculum) - 1, step["it"] * len(a.curriculum) // a.iters)])

        def after_step():
            step["it"] += 1

        def loss_of(batch):
            x0, d, orog, hour0 = (q.to(dev) for q in batch)
            K = cur_K()
            y = model.rollout(x0, hour0, K, tisr_of(x0, d, K))
            err = y - x0[:, None]
            l2 = D.wmse(err, d[:, :K], w, t6)
            loss, parts = l2, {"l2": l2.item(), "K": float(K)}
            if sob is not None:
                hs = sob.loss(err, d[:, :K], scale=t6)
                loss = loss + a.lambda_sob * hs
                parts["sob"] = hs.item()
            if phys is not None:
                ks = torch.randint(0, K + 1, (a.phys_states,), generator=gen).tolist()
                ms = None
                for k in ks:
                    xs = x0 if k == 0 else y[:, k - 1]
                    xo = x0 if k == 0 else x0 + d[:, k - 1]
                    f = model.tendency(xs, hour0 + 6.0 * k)
                    m = phys.mean_squares(xs, orog, f, xo)
                    ms = m if ms is None else {q: ms[q] + m[q] for q in m}
                ms = {q: v / len(ks) for q, v in ms.items()}
                lp = phys.loss(ms)
                loss = loss + a.lambda_p * lp.to(loss.dtype)
                parts["phys"] = lp.detach().item()
                parts.update({f"m_{q}": float(v) for q, v in ms.items()})
            return loss, parts

        n_val_time = len(data[a.val_year])
        val_idx = D.fixed_windows(n_val_time, a.val_K, a.n_val)
        vx0, vd, vorog, vh = D.gather(data, [a.val_year] * len(val_idx), val_idx, a.val_K)
        vx0, vd, vorog, vh = (q.to(dev) for q in (vx0, vd, vorog, vh))
        sd = torch.as_tensor(std[DYN], dtype=torch.float32, device=dev)

        def val_fn():
            out, chunk = {}, max(1, 8 // max(1, a.val_K // 4))
            l2s, z24, t24, z72 = [], [], [], []
            for i in range(0, len(vx0), chunk):
                sl = slice(i, i + chunk)
                y = model.rollout(vx0[sl], vh[sl], a.val_K, tisr_of(vx0[sl], vd[sl], a.val_K))
                e = y - vx0[sl][:, None] - vd[sl][:, :a.val_K]
                l2s.append(D.wmse(y - vx0[sl][:, None], vd[sl][:, :a.val_K], w, t6).item() * (sl.stop - sl.start))
                ww = w.view(1, -1, 1)
                for store, k, c in ((z24, 3, Z500), (t24, 3, T850), (z72, 11, Z500)):
                    if k < a.val_K:
                        store.append(float(((e[:, k, c] * sd[c]) ** 2 * ww).mean().sqrt()) * (sl.stop - sl.start))
            n = len(vx0)
            out["loss"] = sum(l2s) / n
            if z24:
                out["z500_24h"], out["t850_24h"] = sum(z24) / n, sum(t24) / n
            if z72:
                out["z500_72h"] = sum(z72) / n
            return out

        batches = Prefetch(lambda r: D.pick(data, a.train_years, a.K, r, a.batch),
                           lambda p: D.gather(data, *p, a.K), rng, depth=a.prefetch)
        fcfg = FitConfig(iters=a.iters, ckpt_every=a.ckpt_every, log_every=max(1, min(50, a.iters // 10)),
                         val_every=a.val_every, patience=a.patience, select=a.select, max_hours=a.max_hours, clip=a.clip)
        t0 = time.time()
        hist = fit(model, model.parameters(), opt, sched, batches, loss_of, fcfg, ck, val_fn=val_fn,
                   generators={"phys": gen}, after_step=after_step, amp=autocast(dev, a.amp), log=log)
        res["fit"] = {k: v for k, v in hist.items() if k != "hist"}
        res["train_seconds"] = time.time() - t0
        res["train_curve"] = [{k: h[k] for k in h if k != "K"} for h in hist["hist"][:: max(1, len(hist["hist"]) // 200)]]
        if hist["stopped"] == "budget":
            json.dump(res, open(os.path.join(a.out, f"{tag}.partial.json"), "w"), indent=1)
            sys.exit(EXIT_BUDGET)
        torch.save(model.state_dict(), ck)

    # -- final diagnostics on the validation windows -----------------------------------------------------
    model.eval()
    with torch.no_grad():
        n_val_time = len(data[a.val_year])
        val_idx = D.fixed_windows(n_val_time, a.val_K, a.n_val)
        vx0, vd, vorog, vh = (q.to(dev) for q in D.gather(data, [a.val_year] * len(val_idx), val_idx, a.val_K))
        y = torch.cat([model.rollout(vx0[i:i + 2], vh[i:i + 2], a.val_K, tisr_of(vx0[i:i + 2], vd[i:i + 2], a.val_K))
                       for i in range(0, len(vx0), 2)])
        truth = vx0[:, None] + vd[:, :a.val_K]
        sd = torch.as_tensor(std[DYN], dtype=torch.float32, device=dev)
        ww = w.view(1, -1, 1)
        res["val_rmse_by_lead"] = {
            "lead_hours": [6.0 * (k + 1) for k in range(a.val_K)],
            "z500": [float((((y - truth)[:, k, Z500] * sd[Z500]) ** 2 * ww).mean().sqrt()) for k in range(a.val_K)],
            "t850": [float((((y - truth)[:, k, T850] * sd[T850]) ** 2 * ww).mean().sqrt()) for k in range(a.val_K)],
            "z500_persistence": [float((((vx0[:, None] - truth)[:, k, Z500] * sd[Z500]) ** 2 * ww).mean().sqrt()) for k in range(a.val_K)],
        }
        if phys is not None:
            f_roll = model.tendency(y[:, -1], vh + 6.0 * a.val_K)
            f_ic = model.tendency(vx0, vh)
            res["val_strong_residual_over_era5"] = {
                "ic": phys.per_equation(vx0, vorog, f_ic),
                f"lead_{6 * a.val_K}h": phys.per_equation(y[:, -1], vorog, f_roll, x_obs=truth[:, -1]),
            }
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1, default=float)
    log(f"[done {tag}] val z500 RMSE 24 h {res['val_rmse_by_lead']['z500'][min(3, a.val_K - 1)]:.1f} "
        f"(persistence {res['val_rmse_by_lead']['z500_persistence'][min(3, a.val_K - 1)]:.1f}) -> {a.out}/{tag}.json")


if __name__ == "__main__":
    main()
