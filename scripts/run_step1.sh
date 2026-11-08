#!/usr/bin/env bash
# Step 1 end to end on the WB2 64x32 data (configs/step1_64x32.yaml): the four training arms, the 2017 rollouts
# (deterministic and the 8-member perturbed-IC stand-in ensemble) and the WeatherBench2 tables.
#   KAI=data/era5_5p625/kai6h OUT=runs/step1 THREADS=4 bash scripts/run_step1.sh [train|predict|evaluate|all]
set -euo pipefail
cd "$(dirname "$0")/.."
KAI=${KAI:-data/era5_5p625/kai6h}; OUT=${OUT:-runs/step1}; CACHE=${CACHE:-runs/cache}; THREADS=${THREADS:-4}
YEAR=${YEAR:-2017}; MEMBERS=${MEMBERS:-8}; ITERS=${ITERS:-3000}
STAGE=${1:-all}
COMMON="--kai $KAI --out $OUT --cache $CACHE --iters $ITERS --batch 4 --val-every 250 --n-val 8 --val-K 8 --threads $THREADS"
declare -A ARMS=(
  [climate_ode]="--backbone resnet --width 48 --layers 2"
  [sfno_l2]="--backbone sfno --width 64 --layers 4 --modes-lat 16"
  [sfno_h1_pinn]="--backbone sfno --width 64 --layers 4 --modes-lat 16 --lambda-sob 1 --sob-s 1 --lambda-p 0.05 --phys-target 1"
  [sfno_w2]="--backbone sfno --width 64 --layers 4 --modes-lat 16 --lambda-sob 1 --sob-s -1"
)
ORDER="climate_ode sfno_l2 sfno_h1_pinn sfno_w2"
if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
    for t in $ORDER; do python scripts/train_field.py $COMMON ${ARMS[$t]} --tag "$t" 2>&1 | tee "$OUT/$t.log"; done
fi
if [ "$STAGE" = predict ] || [ "$STAGE" = all ]; then
    for t in $ORDER; do
        python scripts/predict.py --kai "$KAI" --run "$OUT" --tag "$t" --year "$YEAR" --stride 20 --max-lead-hours 144 --threads "$THREADS"
        python scripts/predict.py --kai "$KAI" --run "$OUT" --tag "$t" --year "$YEAR" --max-lead-hours 144 --threads "$THREADS" \
            --members "$MEMBERS" --alpha 1.0 --stride 40 --lead-every-hours 24 --cache "$CACHE"
    done
fi
if [ "$STAGE" = evaluate ] || [ "$STAGE" = all ]; then
    F=""; for t in $ORDER; do F="$F --forecast $t=$OUT/${t}_$YEAR.nc --forecast ${t}_ens$MEMBERS=$OUT/${t}_${YEAR}_m$MEMBERS.nc"; done
    python scripts/evaluate.py --kai "$KAI" --year "$YEAR" $F --clim-years 2010-2015 --leads 24,72,144 --out "$OUT/eval_$YEAR"
fi
