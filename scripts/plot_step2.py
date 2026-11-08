#!/usr/bin/env python
"""Step 2 figure: ensemble CRPS and spread/skill by lead for z500, t850, t2m (WB2 2017 scores.csv).

    python scripts/plot_step2.py --scores runs/step2/eval_2017/scores.csv --out docs/step2/fig_crps_ssr.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# fixed categorical order (one hue per model, never cycled)
MODELS = [
    ("A_standin_SFNO", "A  stand-in → SFNO", "#2a78d6"),
    ("B_FM_OT_GRF", "B  FM (OT, GRF) → SFNO", "#eb6834"),
    ("C_FM_OT_GRF_stoch", "C  B + per-step FM error", "#1baf7a"),
    ("D_FM_indep_white", "D  FM (indep, white) → SFNO", "#eda100"),
    ("E_standin_climateODE", "E  stand-in → climate ODE", "#6250d6"),
]
VARS = [("z500", "z500 [m²/s²]"), ("t850", "t850 [K]"), ("t2m", "t2m [K]")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = pd.read_csv(a.scores)
    df = df[(df.region == "global") & (df.lead_hours > 0)]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9a9a95", "axes.labelcolor": "#3d3d3a", "xtick.color": "#5f5f5a",
                         "ytick.color": "#5f5f5a"})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for j, (v, lab) in enumerate(VARS):
        for i, metric in enumerate(("crps", "ssr")):
            ax = axes[i, j]
            ax.grid(axis="y", color="#e6e5e0", lw=0.8)
            for key, name, col in MODELS:
                d = df[(df.model == key) & (df.metric == metric) & (df.variable == v)].sort_values("lead_hours")
                ax.plot(d.lead_hours, d.value, color=col, lw=2, label=name)
            if metric == "ssr":
                ax.axhline(1.0, color="#5f5f5a", lw=1, ls="--")
                ax.set_xlabel("lead [h]")
            ax.set_title(f"{lab} — {'CRPS' if metric == 'crps' else 'spread / skill'}", fontsize=9, color="#1f1f1d",
                         loc="left")
            ax.set_xticks([24, 48, 72, 96, 120, 144])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
