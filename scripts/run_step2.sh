#!/usr/bin/env bash
# Step 2 end to end on WB2 ERA5 64x32 (CPU-sized): flow-matching ensemble generator -> frozen Step 1 SFNO.
#   KAI=data/era5_5p625/kai6h STEP1=runs/step1 OUT=runs/step2 bash scripts/run_step2.sh
# STEP1 must hold the trained fields of Step 1 (<tag>.pt, <tag>.json, stats/).
set -euo pipefail
cd "$(dirname "$0")/.."
KAI=${KAI:-data/era5_5p625/kai6h}; STEP1=${STEP1:-runs/step1}; OUT=${OUT:-runs/step2}
FIELD=${FIELD:-sfno_h1_pinn}; CLIMODE=${CLIMODE:-climate_ode}; M=${M:-8}; T=${THREADS:-2}; CACHE=${CACHE:-$OUT/cache}
mkdir -p "$OUT"
# memory-mapped normalized years, built once before parallel jobs read them (pgcfm.runs.load_years)
python -c "import sys,numpy as np; sys.path.insert(0,'.'); import ffc; from ffc import data as D; \
m=np.load('$STEP1/stats/global_mean.npy'); s=np.load('$STEP1/stats/global_std.npy'); \
D.load_years('$KAI', list(range(2010, 2018)), m, s, cache='$CACHE')"

# 1. stage 2 training (the field stays frozen): proposed W2/Sobolev flow and the plain-CFM control
python scripts/train_flow.py --kai "$KAI" --field-run "$STEP1" --field-tag "$FIELD" --out "$OUT" \
    --coupling ot --source matched --iters 4000 --batch 16 --threads "$T" --cache "$CACHE" &
python scripts/train_flow.py --kai "$KAI" --field-run "$STEP1" --field-tag "$FIELD" --out "$OUT" \
    --coupling indep --source white --iters 4000 --batch 16 --threads "$T" --cache "$CACHE" &
wait

# 2. scenario size on the validation year 2016 (fair CRPS of z500, t850, t2m, u850 at 24/72/144 h)
cal() { python scripts/calibrate_ensemble.py --kai "$KAI" --run "$STEP1" --year 2016 --members "$M" --threads "$T" --cache "$CACHE" "$@"; }
best() { python -c "import json,sys; print(json.load(open(sys.argv[1]))['best'])" "$1"; }
# two calibrations at a time (each uses T threads)
cal --tag "$FIELD" --generator standin --sizes 0.5 1 2 4 8 --out "$OUT/cal_standin_$FIELD.json" &
cal --tag "$CLIMODE" --generator standin --sizes 0.5 1 2 4 --out "$OUT/cal_standin_$CLIMODE.json" &
wait
for f in fm_ot_matched fm_indep_white; do
    cal --tag "$FIELD" --generator flow --flow-run "$OUT" --flow-tag $f --sizes 1 2 4 8 16 --out "$OUT/cal_$f.json" &
done
wait
K=$(best "$OUT/cal_fm_ot_matched.json")
# stochastic arm: IC size kappa in {kappa_IC, kappa_IC / 2} x per-step size kappa_s (run: kappa 2, kappa_s 0.5 won)
KH=$(python -c "print($K / 2)")
for k in "$K" "$KH"; do
    cal --tag "$FIELD" --generator flow --flow-run "$OUT" --flow-tag fm_ot_matched --kappa "$k" \
        --stochastic-sizes 0.25 0.5 1 2 --out "$OUT/cal_fm_ot_matched_stoch_k$k.json" &
done
wait
read KS KS_S < <(python -c "import json,sys; r=[json.load(open(f)) for f in sys.argv[1:]]; \
b=min(r, key=lambda q: q['scores'][str(q['best'])]['crps_mean']); print(b['kappa'], b['best'])" "$OUT"/cal_fm_ot_matched_stoch_k*.json)

# 3-4. test year 2017: 72 ICs (00 UTC, every 5 days), 6 .. 144 h, M members, scored with WeatherBench2 one file at a
# time and deleted after scoring (an 8-member file is 5.9 GB at 64x32); scores_<name>.csv, scores.csv, tables.md
P="--kai $KAI --run $STEP1 --year 2017 --stride 20 --max-lead-hours 144 --threads 4 --cache $CACHE"
EV="python scripts/evaluate.py --kai $KAI --year 2017 --clim-years 2010-2015 --leads 24,72,144 --out $OUT/eval_2017"
first=1
run() { local name=$1 file=$OUT/$2 keep=$3; shift 3
    python scripts/predict.py $P "$@" --out "$file"
    if [ $first = 1 ]; then $EV --forecast "$name=$file"; first=0; else $EV --no-persistence --forecast "$name=$file"; fi
    [ "$keep" = keep ] || rm -f "$file"
}
run SFNO_det sfno_det.nc keep --tag "$FIELD"
run climateODE_det climode_det.nc keep --tag "$CLIMODE"
run A_standin_SFNO A.nc del --members "$M" --tag "$FIELD" --alpha "$(best "$OUT/cal_standin_$FIELD.json")"
run B_FM_OT_GRF B.nc del --members "$M" --tag "$FIELD" --flow-run "$OUT" --flow-tag fm_ot_matched --kappa "$K"
run C_FM_OT_GRF_stoch C.nc del --members "$M" --tag "$FIELD" --flow-run "$OUT" --flow-tag fm_ot_matched --kappa "$KS" --kappa-s "$KS_S"
run D_FM_indep_white D.nc del --members "$M" --tag "$FIELD" --flow-run "$OUT" --flow-tag fm_indep_white \
    --kappa "$(best "$OUT/cal_fm_indep_white.json")"
run E_standin_climateODE E.nc del --members "$M" --tag "$CLIMODE" --alpha "$(best "$OUT/cal_standin_$CLIMODE.json")"
python scripts/evaluate.py --kai "$KAI" --year 2017 --clim-years 2010-2015 --leads 24,72,144 --out "$OUT/eval_2017" \
    --forecast "SFNO_det=$OUT/sfno_det.nc" --forecast "climateODE_det=$OUT/climode_det.nc" \
    $(for n in A_standin_SFNO B_FM_OT_GRF C_FM_OT_GRF_stoch D_FM_indep_white E_standin_climateODE; do echo "--forecast $n=$OUT/$n.nc"; done)
python scripts/plot_step2.py --scores "$OUT/eval_2017/scores.csv" --out "$OUT/fig_crps_ssr.png"
