#!/usr/bin/env bash
# Step 2 end to end on WB2 ERA5 64x32 (CPU-sized): flow-matching ensemble generator -> frozen Step 1 SFNO.
#   KAI=data/era5_5p625/kai6h STEP1=runs/step1 OUT=runs/step2 bash scripts/run_step2.sh
# STEP1 must hold the trained fields of Step 1 (<tag>.pt, <tag>.json, stats/).
set -euo pipefail
cd "$(dirname "$0")/.."
KAI=${KAI:-data/era5_5p625/kai6h}; STEP1=${STEP1:-runs/step1}; OUT=${OUT:-runs/step2}
FIELD=${FIELD:-sfno_h1_pinn}; M=${M:-8}; T=${THREADS:-2}
mkdir -p "$OUT"

# 1. stage 2 training (the field stays frozen): proposed W2/Sobolev flow and the plain-CFM control
python scripts/train_flow.py --kai "$KAI" --field-run "$STEP1" --field-tag "$FIELD" --out "$OUT" \
    --coupling ot --source matched --iters 4000 --batch 16 --threads "$T" &
python scripts/train_flow.py --kai "$KAI" --field-run "$STEP1" --field-tag "$FIELD" --out "$OUT" \
    --coupling indep --source white --iters 4000 --batch 16 --threads "$T" &
wait

# 2. scenario size on the validation year 2016 (fair CRPS of z500, t850, t2m, u850 at 24/72/144 h)
cal() { python scripts/calibrate_ensemble.py --kai "$KAI" --run "$STEP1" --year 2016 --members "$M" --threads "$T" "$@"; }
best() { python -c "import json,sys; print(json.load(open(sys.argv[1]))['best'])" "$1"; }
cal --tag "$FIELD" --generator standin --sizes 0.5 1 2 4 8 --out "$OUT/cal_standin_$FIELD.json"
cal --tag climate_ode --generator standin --sizes 0.5 1 2 4 --out "$OUT/cal_standin_climate_ode.json"
for f in fm_ot_matched fm_indep_white; do
    cal --tag "$FIELD" --generator flow --flow-run "$OUT" --flow-tag $f --sizes 1 2 4 8 16 --out "$OUT/cal_$f.json"
done
K=$(best "$OUT/cal_fm_ot_matched.json")
cal --tag "$FIELD" --generator flow --flow-run "$OUT" --flow-tag fm_ot_matched --kappa "$K" \
    --stochastic-sizes 0.25 0.5 1 2 --out "$OUT/cal_fm_ot_matched_stoch.json"

# 3. test year 2017: 73 ICs (00 UTC, every 5 days), 6 .. 144 h, M members
P="--kai $KAI --run $STEP1 --year 2017 --stride 20 --max-lead-hours 144 --members $M --threads 4"
python scripts/predict.py $P --tag "$FIELD" --alpha "$(best "$OUT/cal_standin_$FIELD.json")" --out "$OUT/A_standin_sfno.nc"
python scripts/predict.py $P --tag "$FIELD" --flow-run "$OUT" --flow-tag fm_ot_matched --kappa "$K" --out "$OUT/B_fm_ot.nc"
python scripts/predict.py $P --tag "$FIELD" --flow-run "$OUT" --flow-tag fm_ot_matched --kappa "$K" \
    --kappa-s "$(best "$OUT/cal_fm_ot_matched_stoch.json")" --out "$OUT/C_fm_ot_stoch.nc"
python scripts/predict.py $P --tag "$FIELD" --flow-run "$OUT" --flow-tag fm_indep_white \
    --kappa "$(best "$OUT/cal_fm_indep_white.json")" --out "$OUT/D_fm_indep.nc"
python scripts/predict.py $P --tag climate_ode --alpha "$(best "$OUT/cal_standin_climate_ode.json")" --out "$OUT/E_standin_climode.nc"
python scripts/predict.py --kai "$KAI" --run "$STEP1" --year 2017 --stride 20 --max-lead-hours 144 --tag climate_ode --out "$OUT/E_climode_det.nc"
python scripts/predict.py --kai "$KAI" --run "$STEP1" --year 2017 --stride 20 --max-lead-hours 144 --tag "$FIELD" --out "$OUT/sfno_det.nc"

# 4. WeatherBench2 scores (+ persistence)
python scripts/evaluate.py --kai "$KAI" --year 2017 --clim-years 2010-2015 --leads 24,72,144 --out "$OUT/eval_2017" \
    --forecast "A stand-in -> SFNO=$OUT/A_standin_sfno.nc" --forecast "B FM(OT,GRF) -> SFNO=$OUT/B_fm_ot.nc" \
    --forecast "C FM(OT,GRF) -> SFNO + stoch=$OUT/C_fm_ot_stoch.nc" --forecast "D FM(indep,white) -> SFNO=$OUT/D_fm_indep.nc" \
    --forecast "E stand-in -> climate ODE=$OUT/E_standin_climode.nc" --forecast "climate ODE (det)=$OUT/E_climode_det.nc" \
    --forecast "SFNO (det)=$OUT/sfno_det.nc"
