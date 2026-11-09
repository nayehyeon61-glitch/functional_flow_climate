#!/usr/bin/env python
"""Step 4 figure: WB2 2017 RMSE and ACC by lead of the hybrid (physics core + SFNO) against Step 1 arms.

    python scripts/plot_step4.py runs/step4/eval_2017/scores.csv docs/step4/fig_rmse_acc.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODELS = [("hybrid_sfno_h1_pinn", "core + SFNO (Step 4)", "C3", "-"), ("sfno_h1_pinn", "SFNO H1+PINN (Step 1)", "C0", "-"),
          ("climate_ode", "climate ODE (Step 1)", "C2", "-"), ("core_only", "dinosaur core alone", "C7", "--"),
          ("persistence", "persistence", "k", ":")]


def main(src, out):
    df = pd.read_csv(src)
    df = df[df.region == "global"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, var in enumerate(["z500", "t850", "u850"]):
        for i, metric in enumerate(["rmse", "acc"]):
            ax = axes[i, j]
            for m, label, c, ls in MODELS:
                s = df[(df.model == m) & (df.metric == metric) & (df.variable == var)].sort_values("lead_hours")
                s = s[s.lead_hours > 0]
                if len(s):
                    ax.plot(s.lead_hours, s.value, color=c, ls=ls, label=label)
            ax.set_title(f"{var} {metric.upper()}")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("lead [h]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("WeatherBench2 ERA5 5.625 deg, 2017, 72 ICs: same 3000-iteration budget")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
