#!/usr/bin/env bash
# Step 5: SpeedyWeather.jl as the physics core under the neural operator (the backend swap Step 4 could not do),
#   dx/dt = alpha * F_phys(x) + R_theta(x, F_phys(x), t)     (ffc.speedycore, FieldConfig.core = "speedy")
# with exactly the Step 4 recipe and budget (sfno_h1_pinn, 3000 it, 2010-15 / 2016 / 2017), two arms:
#   hybrid_speedy_full  SpeedyWeather PrimitiveWetModel with all its parameterizations (radiation, convection,
#                       condensation, boundary layer, surface fluxes over SST / land from ERA5 skt)
#   hybrid_speedy_dry   the same model with every parameterization off (the adiabatic core: like dinosaur)
# and the cores alone (head left at zero) as references. The Step 4 dinosaur hybrid and Step 1 arms enter the
# tables through their cached WB2 scores (same ICs, leads, climatology and scorer).
#   KAI=data/era5_5p625/kai6h STEP4=runs/step4 OUT=runs/step5 bash scripts/run_step5.sh [train|predict|evaluate|all] [arm ...]
# Needs Julia + SpeedyWeather.jl: scripts/install_speedyweather.sh
set -euo pipefail
cd "$(dirname "$0")/.."
KAI=${KAI:-data/era5_5p625/kai6h}; STEP4=${STEP4:-runs/step4}; OUT=${OUT:-runs/step5}; CACHE=${CACHE:-runs/cache}
T=${THREADS:-4}; YEAR=${YEAR:-2017}; M=${MEMBERS:-8}; ITERS=${ITERS:-3000}
STAGE=${1:-all}; shift || true
ARMS=${*:-"full dry"}
mkdir -p "$OUT"; [ -d "$OUT/stats" ] || cp -r "$STEP4/stats" "$OUT/stats"
COMMON="--kai $KAI --out $OUT --cache $CACHE --batch 4 --val-every 250 --n-val 8 --val-K 8 --threads $T"
SFNO="--backbone sfno --width 64 --layers 4 --modes-lat 16 --lambda-sob 1 --sob-s 1 --lambda-p 0.05 --phys-target 1"
core_of() { case $1 in full) echo "--core speedy --core-layers 18 --core-dt-min 30 --core-physics all";;
                        dry)  echo "--core speedy --core-layers 18 --core-dt-min 30 --core-physics none";; esac; }
if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
    for arm in $ARMS; do
        python scripts/train_field.py $COMMON $SFNO $(core_of $arm) --iters "$ITERS" --tag hybrid_speedy_$arm 2>&1 | tee "$OUT/hybrid_speedy_$arm.log"
        python scripts/train_field.py $COMMON $SFNO $(core_of $arm) --iters 1 --lr 0 --wd 0 --tag core_speedy_$arm 2>&1 | tee "$OUT/core_speedy_$arm.log"
    done
fi
if [ "$STAGE" = predict ] || [ "$STAGE" = all ]; then
    for arm in $ARMS; do
        for t in core_speedy_$arm hybrid_speedy_$arm; do
            python scripts/predict.py --kai "$KAI" --run "$OUT" --tag "$t" --year "$YEAR" --stride 20 --max-lead-hours 144 --threads "$T"
        done
        python scripts/predict.py --kai "$KAI" --run "$OUT" --tag hybrid_speedy_$arm --year "$YEAR" --max-lead-hours 144 --threads "$T" \
            --members "$M" --alpha 1.0 --stride 40 --lead-every-hours 24 --cache "$CACHE"
    done
fi
if [ "$STAGE" = evaluate ] || [ "$STAGE" = all ]; then
    mkdir -p "$OUT/eval_$YEAR"; F=""
    for arm in full dry; do
        for t in core_speedy_$arm hybrid_speedy_$arm; do
            [ -f "$OUT/${t}_$YEAR.nc" ] && F="$F --forecast $t=$OUT/${t}_$YEAR.nc"
        done
        [ -f "$OUT/hybrid_speedy_${arm}_${YEAR}_m$M.nc" ] && F="$F --forecast hybrid_speedy_${arm}_ens$M=$OUT/hybrid_speedy_${arm}_${YEAR}_m$M.nc"
    done
    # Step 4 (dinosaur hybrid, its core alone) and Step 1 arms: cached scores
    for n in hybrid_sfno_h1_pinn hybrid_sfno_h1_pinn_ens$M core_only sfno_h1_pinn climate_ode climate_ode_ens$M; do
        if [ -f "$STEP4/eval_$YEAR/scores_$n.csv" ]; then
            cp "$STEP4/eval_$YEAR/scores_$n.csv" "$OUT/eval_$YEAR/"
            F="$F --forecast $n=cached"
        fi
    done
    cp "$STEP4/eval_$YEAR/scores_persistence.csv" "$OUT/eval_$YEAR/" 2>/dev/null || true
    python scripts/evaluate.py --kai "$KAI" --year "$YEAR" $F --clim-years 2010-2015 --leads 24,72,144 --out "$OUT/eval_$YEAR"
fi
