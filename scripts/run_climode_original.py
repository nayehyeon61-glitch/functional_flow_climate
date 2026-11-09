#!/usr/bin/env python
"""The original ClimODE (Verma, Heinonen, Garg, ICLR 2024; published global checkpoint, authors' code) on our
2017 initial conditions, scored with the same WeatherBench2 code as the other arms (docs/step2_retrain_results.md).

Everything model-side is upstream's, vendored unchanged in pg-flow (``third_party/pgflow/third_party/climode``):
data loading and min-max normalization (2006-2018), the per-IC velocity fit (``Optim_velocity`` + Gaussian
kernel), the ODE integration (Euler, 1 h), the Gaussian emission head and the evaluation protocol of
``evaluation_global.py`` as pg-flow's ``scripts/run_climode_eval.py`` runs it: the 2017 and 2018 states of the
same calendar time share one batch of two (upstream never calls ``model.eval()``, so its batch norm uses the
statistics of that batch; ``--eval-mode`` is the alternative). Differences to upstream's test loop: our ICs
(00 UTC every 5 days, 72 of them, like ``scripts/predict.py``) instead of 12 UTC every 2 days, the past frames of
the first ICs of a year come from the previous year, and leads run to 144 h (upstream evaluates 6-42 h).

Outputs (``--out``): ``pred_<part>.npz`` (de-normalized mean / std of the 2017 row at 6 h leads 0..144 h), then
with ``--score``: ``scores_<name>.csv`` (``--members`` samples of the Gaussian emission, fair CRPS / spread /
SSR like the other ensembles), ``scores_<name>_det.csv`` (the mean) and ``crps_gauss_<name>.json`` (the
closed-form Gaussian CRPS upstream reports, area-weighted, physical units), written to ``--eval-dir`` and
merged into its ``scores.csv`` / ``tables.md``.

    python scripts/run_climode_original.py --data data/era5_5p625/era5_data --out runs/step3/climode_orig \
        --part 0/2 --threads 2
    python scripts/run_climode_original.py --data ... --out ... --score --kai data/era5_5p625/kai6h \
        --eval-dir runs/step3/eval_2017
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ffc  # noqa: E402,F401
from ffc import data as D  # noqa: E402
from ffc import deps  # noqa: E402
from ffc import evaluate as EV  # noqa: E402
from pgcfm.eval import scores as wb2_scores  # noqa: E402

CLIMODE = os.path.join(deps.pgflow_root(), "third_party", "climode")
CKPT_URL = ("https://raw.githubusercontent.com/Aalto-QuML/ClimODE/e729d23e8799ce0e075699e76d60227d848d8d0c/"
            "checkpoints/ClimODE_global.pt")
LEVELS = ["z", "t", "t2m", "u10", "v10"]
FOLDERS = ["geopotential_500", "temperature_850", "2m_temperature", "10m_u_component_of_wind",
           "10m_v_component_of_wind"]
WB2_VARS = {"z": "z500", "t": "t850", "t2m": "t2m"}  # ClimODE variables that are headline channels of the KAI contract
STEPS_PER_YEAR = 1460


def load_upstream(data_root):
    """Upstream's loader for the five variables: states of 2016 / 2017 / 2018 (upstream's val + test tensors) side
    by side, (1460, 3, 5, H, W) in min-max units, plus the scales and the constants."""
    import utils as cu  # noqa: E402  (upstream, on sys.path)
    years, max_lev, min_lev = [], [], []
    for lev, folder in zip(LEVELS, FOLDERS):
        _, va, te, _, lat, lon, mx, mn, _ = cu.get_train_test_data_without_scales_batched(
            os.path.join(data_root, folder, "*.nc"), slice("2006", "2016"), slice("2016", "2016"),
            slice("2017", "2018"), lev, 0)
        years.append(torch.cat([va, te], dim=1))  # (1460, 3 years, 1, H, W)
        max_lev.append(mx)
        min_lev.append(mn)
    years = torch.cat(years, dim=2).float()  # (1460, 3, 5, H, W)
    const, lat_map, lon_map = cu.add_constant_info([os.path.join(data_root, "constants", "constants_5.625deg.nc")])
    return years, np.array(max_lev), np.array(min_lev), lat, lon, const, lat_map, lon_map


def predict(a):
    sys.path.insert(0, CLIMODE)
    import model_function  # noqa: F401  (classes of the pickled checkpoint)
    from model_function import Optim_velocity
    import utils as cu

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = False  # pg-flow step 11: 2x faster for these 128-channel 32x64 convs on CPU
    cu.set_seed(42)
    os.makedirs(a.out, exist_ok=True)
    ckpt = a.ckpt or os.path.join(a.out, "ClimODE_global.pt")
    if not os.path.exists(ckpt):
        import urllib.request
        urllib.request.urlretrieve(CKPT_URL, ckpt)
    years, max_lev, min_lev, lat, lon, const, lat_map, lon_map = load_upstream(a.data)
    H, W = years.shape[-2], years.shape[-1]
    cwd = os.getcwd()
    os.chdir(a.out)  # upstream writes / reads kernel.npy in the working directory
    try:
        if not os.path.exists("kernel.npy"):
            t0 = time.time()
            cu.get_gauss_kernel((H, W), lat, lon)
            print(f"kernel {time.time() - t0:.0f}s", flush=True)
        kernel = torch.from_numpy(np.load("kernel.npy"))
    finally:
        os.chdir(cwd)
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.method = a.solver
    if a.eval_mode:
        model.eval()

    L = a.lead_steps
    ics_all = list(range(0, STEPS_PER_YEAR - L, a.stride))  # scripts/predict.py: 00 UTC every --stride steps
    if a.n_ic:
        ics_all = ics_all[: a.n_ic]
    k, n = (int(s) for s in a.part.split("/"))
    ics = ics_all[k::n]
    cols = [1, 2] if a.pair else [1]  # batch rows: 2017 and (upstream protocol) 2018 at the same calendar time
    B = len(cols)
    mean_out = np.zeros((len(ics), L + 1, 5, H, W), np.float32)
    std_out = np.zeros_like(mean_out)
    scale = (max_lev - min_lev)[:, None, None]  # upstream: x = x_mm * (max - min) + min
    t0 = time.time()
    for e, i in enumerate(ics):
        # past frames i-2, i-1, i of each batch row; the first steps of a year take them from the previous year
        past = []
        for c in cols:
            frames = []
            for j in (i - 2, i - 1, i):
                frames.append(years[j, c] if j >= 0 else years[STEPS_PER_YEAR + j, c - 1])
            past.append(torch.stack(frames))
        past = torch.stack(past)  # (B, 3, 5, H, W)
        past_time = torch.tensor([i - 2, i - 1, i], dtype=torch.float32)
        delta_u = cu.get_delta_u(past, past_time)
        data = years[i, cols].view(B, 1, 5, H, W)
        torch.manual_seed(42 + i)  # Optim_velocity starts from randn: fixed per IC, independent of the partition
        tv = time.time()
        v_x, v_y, _, _ = cu.optimize_vel(B, data, delta_u, Optim_velocity, kernel, H, W)
        vel = torch.cat([v_x, v_y], dim=1).detach().view(B, 2 * 5, H, W)
        model.update_param([vel, const, lat_map, lon_map])
        T = torch.arange(i, i + L + 1, dtype=torch.float32)
        torch.manual_seed(1000 + i)  # upstream's dropout stays on in its evaluation
        tm = time.time()
        with torch.no_grad():
            mean, std, _ = model(T, data)  # (L+1, B, 5, H, W) in min-max units
        mean_out[e] = mean[:, 0].numpy() * scale + min_lev[:, None, None]
        std_out[e] = std[:, 0].numpy() * scale
        print(f"ic {e + 1}/{len(ics)} (index {i}): velocity {tm - tv:.0f}s, ode {time.time() - tm:.0f}s, "
              f"total {time.time() - t0:.0f}s", flush=True)
    tag = f"part{k}of{n}" + ("_eval" if a.eval_mode else "")
    np.savez_compressed(os.path.join(a.out, f"pred_{tag}.npz"), mean=mean_out, std=std_out, ics=np.array(ics),
                        lead_hours=np.arange(L + 1) * 6.0, max_lev=max_lev, min_lev=min_lev, lat=lat, lon=lon)
    print(f"wrote pred_{tag}.npz", flush=True)


def _gauss_crps(truth, mean, std):
    """Closed-form CRPS of N(mean, std) in the truth's units (upstream's ``evaluation_crps_mm`` uses the same
    formula in min-max units through properscoring)."""
    from scipy.special import erf
    z = (truth - mean) / std
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
    return std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def score(a):
    pattern = os.path.join(a.out, "pred_part*of*" + ("_eval" if a.eval_mode else "") + ".npz")
    parts = [np.load(p) for p in sorted(glob.glob(pattern)) if (("_eval" in p) == a.eval_mode)]
    if not parts:
        raise SystemExit(f"no predictions at {pattern}")
    ics = np.concatenate([p["ics"] for p in parts])
    order = np.argsort(ics)
    ics = ics[order]
    mean = np.concatenate([p["mean"] for p in parts])[order]
    std = np.concatenate([p["std"] for p in parts])[order]
    lead = parts[0]["lead_hours"]
    lat, lon = D.grid_of(a.kai, a.year)
    assert np.allclose(lat, parts[0]["lat"]) and np.allclose(lon, parts[0]["lon"]), "ClimODE grid != KAI grid"
    N, Lp1 = mean.shape[:2]
    init = EV.init_times_of(a.kai, a.year, ics)
    truth = EV.select_headline(EV.truth_from_kai(a.kai, a.year, ics, lead, lat, lon))[list(WB2_VARS.values())]
    clim = None
    clim_path = os.path.join(a.eval_dir, f"climatology_{a.clim_years}.nc")
    if os.path.exists(clim_path):
        clim = EV.select_headline(xr.open_dataset(clim_path).load())[list(WB2_VARS.values())]
    else:
        print(f"no climatology at {clim_path}: ACC skipped", flush=True)
    lead_td = (lead * 3600e9).astype("timedelta64[ns]")
    coords = {"init_time": init, EV.LEAD: lead_td, "latitude": lat, "longitude": lon}
    dims = ["init_time", EV.LEAD, "latitude", "longitude"]
    sel = [LEVELS.index(k) for k in WB2_VARS]
    det = xr.Dataset({v: (dims, mean[:, :, c]) for v, c in zip(WB2_VARS.values(), sel)}, coords=coords)
    det = det.assign_coords(valid_time=det.init_time + det[EV.LEAD])
    g = torch.Generator().manual_seed(a.seed)
    eps = torch.randn((N, Lp1, a.members, len(sel), len(lat), len(lon)), generator=g).numpy()
    ens = mean[:, :, None, sel] + std[:, :, None, sel] * eps
    dims_m = ["init_time", EV.LEAD, EV.REALIZATION, "latitude", "longitude"]
    ensd = xr.Dataset({v: (dims_m, ens[:, :, :, j]) for j, v in enumerate(WB2_VARS.values())},
                      coords=dict(coords, **{EV.REALIZATION: np.arange(a.members)}))
    ensd = ensd.assign_coords(valid_time=ensd.init_time + ensd[EV.LEAD])
    name = a.name + ("_eval" if a.eval_mode else "")
    frames = []
    for nm, ds in ((name, ensd), (name + "_det", det)):
        sc = wb2_scores.score_forecast(ds, truth, climatology=clim, region_names=["global"])
        df = EV._to_frame({nm: sc})
        df.to_csv(os.path.join(a.eval_dir, f"scores_{nm}.csv"), index=False)
        frames.append(df)
        print(f"scored {nm}", flush=True)
    # closed-form Gaussian CRPS (what upstream reports), area-weighted like WB2
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    w = w / w.sum()
    out = {}
    for j, (lev, v) in enumerate(WB2_VARS.items()):
        tr = truth[v].transpose("init_time", EV.LEAD, "latitude", "longitude").values
        c = _gauss_crps(tr, mean[:, :, sel[j]], std[:, :, sel[j]])
        out[v] = {str(int(h)): float((c[:, k] * w).sum(axis=(-2, -1)).mean()) for k, h in enumerate(lead)}
    json.dump({"model": name, "n_ic": int(N), "ics": ics.tolist(), "crps_gauss": out},
              open(os.path.join(a.eval_dir, f"crps_gauss_{name}.json"), "w"), indent=1)
    # merge into the evaluation folder's combined table (same layout as scripts/evaluate.py)
    allf = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(a.eval_dir, "scores_*.csv")))]
    df = pd.concat(allf, ignore_index=True)
    df.to_csv(os.path.join(a.eval_dir, "scores.csv"), index=False)
    leads = [24.0, 72.0, 144.0]
    lines = [f"# WeatherBench2 scores, {a.year}, {N} ICs, grid {len(lat)}x{len(lon)}", ""]
    for metric, title in (("rmse", "RMSE"), ("acc", "ACC"), ("crps", "CRPS (ensemble) / MAE (deterministic)"),
                          ("ssr", "spread / skill (ensembles)"), ("bias", "bias")):
        if metric in set(df.metric):
            lines += [f"## {title}", "", EV.markdown(df, metric, variables=["z500", "t850", "t2m", "q700", "u850", "msl"],
                                                    leads=leads), ""]
    open(os.path.join(a.eval_dir, "tables.md"), "w").write("\n".join(lines))
    for m in ("rmse", "crps", "ssr"):
        print(m, EV.markdown(df[df.model.str.startswith(name) | df.model.isin(["climateODE_det", "E_standin_climateODE"])],
                             m, variables=["z500", "t850", "t2m"], leads=leads), sep="\n")
    print("gauss crps", {v: {h: round(x, 3) for h, x in d.items() if h in ("24", "72", "144")} for v, d in out.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="WB1-style era5_data folder (pg-flow scripts/wb2_to_kai.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--year", type=int, default=2017)
    p.add_argument("--stride", type=int, default=20, help="6 h steps between ICs (scripts/predict.py)")
    p.add_argument("--n-ic", type=int, default=None)
    p.add_argument("--lead-steps", type=int, default=24)
    p.add_argument("--part", default="0/1", help="k/n: every n-th IC starting at k (parallel processes)")
    p.add_argument("--no-pair", dest="pair", action="store_false",
                   help="batch of one (2017 only) instead of upstream's 2017 + 2018 pair")
    p.add_argument("--eval-mode", action="store_true", help="model.eval(): running batch-norm statistics, no dropout")
    p.add_argument("--solver", default="euler")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--score", action="store_true", help="score the merged predictions instead of predicting")
    p.add_argument("--kai", default="data/era5_5p625/kai6h")
    p.add_argument("--eval-dir", default="runs/step3/eval_2017")
    p.add_argument("--clim-years", default="2010-2015")
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="F_ClimODE_orig")
    a = p.parse_args(argv)
    (score if a.score else predict)(a)


if __name__ == "__main__":
    main()
