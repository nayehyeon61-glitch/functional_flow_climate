#!/usr/bin/env bash
# Julia + SpeedyWeather.jl for ffc.speedycore (Step 5) without the Julia package server.
# The cloud container blocks julialang-s3 / pkg.julialang.org / install.julialang.org and github.com HTTP
# downloads, but conda-forge and git over github.com work, so:
#   1. julia from conda-forge (through a micromamba taken from conda-forge itself),
#   2. JULIA_PKG_SERVER="" -> Pkg clones the General registry and the packages with git,
#   3. SpeedyWeather's boundary-condition assets (orography, land-sea mask, SST, ...) from a git clone of
#      SpeedyWeatherAssets, registered as artifacts through a file:// assets URL.
# Defaults match ffc.speedycore: FFC_JULIA=/opt/mamba/envs/jl/bin/julia, FFC_JULIA_PROJECT=/opt/jsw.
# Export JULIA_DEPOT_PATH (default here /opt/jdepot) for every later run as well.
set -euo pipefail
PREFIX=${PREFIX:-/opt}; JV=${JULIA_VERSION:-1.12}
export MAMBA_ROOT_PREFIX=$PREFIX/mamba JULIA_PKG_SERVER="" JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$PREFIX/jdepot}
if [ ! -x "$PREFIX/mamba/envs/jl/bin/julia" ]; then
    mkdir -p "$PREFIX/mm"
    curl -sSL https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2 | tar xj -C "$PREFIX/mm"
    "$PREFIX/mm/bin/micromamba" create -y -q -n jl -c conda-forge "julia=$JV"
fi
JL=$PREFIX/mamba/envs/jl/bin/julia
[ -d "$PREFIX/swassets" ] || git clone -q --depth 1 --branch v1.0.0 https://github.com/SpeedyWeather/SpeedyWeatherAssets "$PREFIX/swassets"
mkdir -p "$PREFIX/swrefs/tags"; ln -sfn "$PREFIX/swassets" "$PREFIX/swrefs/tags/v1.0.0"
"$JL" -e "using Pkg; isdir(joinpath(DEPOT_PATH[1], \"registries\", \"General\")) || Pkg.Registry.add(url=\"https://github.com/JuliaRegistries/General\");
          Pkg.activate(\"$PREFIX/jsw\"); Pkg.add([\"SpeedyWeather\", \"NCDatasets\"])"
"$JL" --project="$PREFIX/jsw" -e "using SpeedyWeather, NCDatasets; import SpeedyWeather.RingGrids as RG
    for (f, v) in [(\"orography.nc\", \"orog\"), (\"land-sea_mask.nc\", \"\"), (\"sea_surface_temperature.nc\", \"\"),
                   (\"sea_ice.nc\", \"\"), (\"land_surface_temperature.nc\", \"\"), (\"snow.nc\", \"\"), (\"albedo.nc\", \"\"),
                   (\"soil_moisture.nc\", \"swl1\"), (\"soil_type.nc\", \"slt\"), (\"vegetation.nc\", \"vegh\")]
        RG.get_asset(\"data/boundary_conditions/\" * f; assets_url=\"file://$PREFIX/swrefs\", FileFormat=NCDataset, name=v)
    end
    sim = initialize!(PrimitiveWetModel(SpectralGrid(FullGaussianGrid(16); nlayers=8))); run!(sim, period=Day(1))
    println(\"SpeedyWeather \", pkgversion(SpeedyWeather), \" ok\")"
