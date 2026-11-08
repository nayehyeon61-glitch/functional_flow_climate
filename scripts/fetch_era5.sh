#!/usr/bin/env bash
# ERA5 from the public WeatherBench2 zarr into the pg-flow KAI layout (<out>/kai6h/<year>.nc, 67 channels, 6 h),
# with pg-flow's own converter (third_party/pgflow/scripts/wb2_to_kai.py). Grid 64x32 (5.625 deg) by default;
# GRID=128x64 for 2.8125 deg. About 0.6 GB a year at 64x32.
#   OUT=data/era5_5p625 YEARS="2010 2011 2012 2013 2014 2015 2016 2017" bash scripts/fetch_era5.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-data/era5_5p625}; GRID=${GRID:-64x32}; YEARS=${YEARS:-"2010 2011 2012 2013 2014 2015 2016 2017"}
PGFLOW=${PGFLOW_ROOT:-third_party/pgflow}
[ -f "$PGFLOW/scripts/wb2_to_kai.py" ] || { echo "pg-flow checkout not found: git submodule update --init third_party/pgflow"; exit 1; }
mkdir -p "$OUT"
for y in $YEARS; do
    if [ -f "$OUT/kai6h/$y.nc" ]; then echo "$y exists"; continue; fi
    python "$PGFLOW/scripts/wb2_to_kai.py" --out "$OUT" --all-hours --grid "$GRID" --years "$y" --no-climode
done
ls -la "$OUT/kai6h"
