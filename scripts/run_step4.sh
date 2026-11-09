#!/usr/bin/env bash
# Step 4 on WB2 ERA5 64x32 (CPU-sized): a spectral primitive-equation core under the neural operator,
#   dx/dt = alpha * F_phys(x) + R_theta(x, F_phys(x), t)     (ffc.physcore, FieldConfig.core = "dinosaur")
# with the Step 1 sfno_h1_pinn recipe and budget (3000 it, H1 + PINN), the core alone (R_theta = 0) as a
# reference, and the same 2017 WeatherBench2 protocol as Step 1 (72 ICs, 6 .. 144 h; 8-member stand-in ensemble).
#   KAI=data/era5_5p625/kai6h STEP1=runs/step1 OUT=runs/step4 bash scripts/run_step4.sh [train|predict|evaluate|all]
# STEP1/stats (the Step 1 normalization) is copied so every arm shares it; STEP1 forecasts are rescored if present.
set -euo pipefail
cd "$(dirname "$0")/.."
KAI=${KAI:-data/era5_5p625/kai6h}; STEP1=${STEP1:-runs/step1}; OUT=${OUT:-runs/step4}; CACHE=${CACHE:-runs/cache}
T=${THREADS:-4}; YEAR=${YEAR:-2017}; M=${MEMBERS:-8}; ITERS=${ITERS:-3000}
STAGE=${1:-all}
mkdir -p "$OUT"; [ -d "$OUT/stats" ] || cp -r "$STEP1/stats" "$OUT/stats"
COMMON="--kai $KAI --out $OUT --cache $CACHE --batch 4 --val-every 250 --n-val 8 --val-K 8 --threads $T"
SFNO="--backbone sfno --width 64 --layers 4 --modes-lat 16 --lambda-sob 1 --sob-s 1 --lambda-p 0.05 --phys-target 1"
CORE="--core dinosaur --core-trunc 21 --core-layers 18 --core-dt-min 30 --core-dfi-h 6"
if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
    python scripts/train_field.py $COMMON $SFNO $CORE --iters "$ITERS" --tag hybrid_sfno_h1_pinn 2>&1 | tee "$OUT/hybrid_sfno_h1_pinn.log"
    # the core alone: the same model with its zero-initialized head left untouched (one step at lr 0)
    python scripts/train_field.py $COMMON $SFNO $CORE --iters 1 --lr 0 --wd 0 --tag core_only 2>&1 | tee "$OUT/core_only.log"
fi
if [ "$STAGE" = predict ] || [ "$STAGE" = all ]; then
    for t in core_only hybrid_sfno_h1_pinn; do
        python scripts/predict.py --kai "$KAI" --run "$OUT" --tag "$t" --year "$YEAR" --stride 20 --max-lead-hours 144 --threads "$T"
    done
    python scripts/predict.py --kai "$KAI" --run "$OUT" --tag hybrid_sfno_h1_pinn --year "$YEAR" --max-lead-hours 144 --threads "$T" \
        --members "$M" --alpha 1.0 --stride 40 --lead-every-hours 24 --cache "$CACHE"
fi
if [ "$STAGE" = evaluate ] || [ "$STAGE" = all ]; then
    F="--forecast core_only=$OUT/core_only_$YEAR.nc --forecast hybrid_sfno_h1_pinn=$OUT/hybrid_sfno_h1_pinn_$YEAR.nc"
    F="$F --forecast hybrid_sfno_h1_pinn_ens$M=$OUT/hybrid_sfno_h1_pinn_${YEAR}_m$M.nc"
    # Step 1 arms enter the tables through their cached WB2 scores (same ICs, leads, climatology and scorer)
    for n in sfno_h1_pinn sfno_h1_pinn_ens$M climate_ode climate_ode_ens$M; do
        if [ -f "$STEP1/eval_$YEAR/scores_$n.csv" ]; then
            cp "$STEP1/eval_$YEAR/scores_$n.csv" "$OUT/eval_$YEAR/" 2>/dev/null || { mkdir -p "$OUT/eval_$YEAR"; cp "$STEP1/eval_$YEAR/scores_$n.csv" "$OUT/eval_$YEAR/"; }
            F="$F --forecast $n=cached"
        fi
    done
    python scripts/evaluate.py --kai "$KAI" --year "$YEAR" $F --clim-years 2010-2015 --leads 24,72,144 --out "$OUT/eval_$YEAR"
fi
