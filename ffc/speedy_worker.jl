# SpeedyWeather.jl worker for ffc.speedycore (Step 5): one PrimitiveWetModel on the T21 full Gaussian grid,
# driven over a line protocol on stdin/stdout. Python does the KAI <-> sigma mapping; this process only
# integrates the model.
#
#   start:  julia --project=<env> speedy_worker.jl <nlayers> <dt_minutes> <physics all|none|off-list>
#   ->      "@@ready <nlat> <nlon> <nlayers>" after writing <tmp>/geometry.bin (see `write_geometry`)
#   <-      "run <in.bin> <out.bin> <hours> <n> <time_1> ... <time_n>"     (times as yyyy-mm-ddTHH:MM)
#   ->      "@@done <seconds>"
#
# in.bin, per state, Float32 in (nlev, nlat N->S, nlon) C order = Julia (nlon, nlat, nlev):
#   u, v [m/s], T [K], q [kg/kg] on the model's sigma layers; ln(ps [Pa]); skin temperature [K]
# out.bin, per state: the same u, v, T, q, ln ps twice, at t0 (the spectrally truncated initial state as the
# model sees it) and after <hours> hours of integration.
# Every state starts from the same boundary state (a copy taken at initialization), with the ocean's SST and
# the land temperature set from the state's own skin temperature, so calls are independent of each other.

using SpeedyWeather, Dates

const NLAYERS = parse(Int, ARGS[1])
const DT_MIN = parse(Float64, ARGS[2])
# physics: "all", "none" (adiabatic dynamical core), or a comma list of parameterizations to switch off
const PHYS_ALL = (:large_scale_condensation, :convection, :radiation, :vertical_diffusion, :boundary_layer,
                  :surface_heat_flux, :surface_humidity_flux, :surface_momentum_flux)
const OFF = ARGS[3] == "all" || ARGS[3] == "1" ? () :
            ARGS[3] == "none" || ARGS[3] == "0" ? PHYS_ALL : Tuple(Symbol.(split(ARGS[3], ",")))
const TMP = length(ARGS) >= 4 ? ARGS[4] : tempdir()

sg = SpectralGrid(FullGaussianGrid(16); nlayers = NLAYERS)    # T21 (dealiasing 2), 32 x 64 = the KAI longitudes
kw = (; ocean = PrescribedOcean(), sea_ice = nothing,
      land = LandModel(sg; temperature = ConstantLandTemperature(sg)), feedback = nothing)
# "none" = the dynamical core only (the dinosaur arm's equivalent), adiabatic
kw = (; kw..., NamedTuple{OFF}(ntuple(_ -> nothing, length(OFF)))...)
model = PrimitiveWetModel(sg; kw...)
set!(model, Δt = Minute(DT_MIN))
sim = initialize!(model; time = DateTime(2016, 1, 1))
vars = sim.variables
const NLAT, NLON = 32, 64
const NPT = NLAT * NLON
land0 = deepcopy(vars.prognostic.land)
mask = model.land_sea_mask.land_fraction   # land fraction in [0, 1]

function write_geometry(path)
    open(path, "w") do io
        write(io, Float32.(model.geometry.latd))                      # (nlat,) N -> S
        write(io, Float32.(model.geometry.σ_levels_half))             # (nlev + 1,) top -> surface
        write(io, Float32.(model.geometry.σ_levels_full))             # (nlev,)
        write(io, Float32.(model.orography.orography.data))           # (nlat * nlon,) surface height [m]
        write(io, Float32.(mask.data))                                # land fraction
    end
end

grid3(a) = reshape(Float32.(a.data), NPT, NLAYERS)

function snapshot!(out, off)
    g = vars.grid
    for f in (g.u, g.v, g.temperature, g.humidity)
        d = Float32.(f.data)                 # (npt, nlev) current step
        d = ndims(d) == 3 ? d[:, :, 1] : d
        out[off+1:off+NPT*NLAYERS] .= vec(d)
        off += NPT * NLAYERS
    end
    p = Float32.(g.pressure.data)
    p = ndims(p) == 2 ? p[:, 1] : p
    out[off+1:off+NPT] .= p
    return off + NPT
end

function run_states(inpath, outpath, hours, times)
    n = length(times)
    per_in = 4 * NPT * NLAYERS + 2 * NPT
    per_out = 2 * (4 * NPT * NLAYERS + NPT)
    x = Vector{Float32}(undef, per_in * n)
    read!(inpath, x)
    out = zeros(Float32, per_out * n)
    for i in 1:n
        o = (i - 1) * per_in
        lay(k) = reshape(view(x, o + (k - 1) * NPT * NLAYERS + 1:o + k * NPT * NLAYERS), NPT, NLAYERS)
        u, v, t, q = lay(1), lay(2), lay(3), lay(4)
        lnp = view(x, o + 4 * NPT * NLAYERS + 1:o + 4 * NPT * NLAYERS + NPT)
        skt = view(x, o + 4 * NPT * NLAYERS + NPT + 1:o + 4 * NPT * NLAYERS + 2 * NPT)
        # stateless: the boundary state of initialization, surface temperatures from this state's skt
        for (dst, src) in zip(values(vars.prognostic.land), values(land0))
            copyto!(dst, src)
        end
        sst = vars.prognostic.ocean.sea_surface_temperature
        sst.data .= reshape(skt, size(sst.data))
        st = vars.prognostic.land.soil_temperature
        for k in 1:size(st.data, 2)
            st.data[:, k] .= skt
        end
        mk(a) = Field(Matrix{Float32}(a), sg.grid)
        set!(vars, model.geometry; u = mk(u), v = mk(v), temperature = mk(t), humidity = mk(q),
             pressure = Field(Vector{Float32}(lnp), sg.grid), spectral_transform = model.spectral_transform)
        set!(vars.prognostic.clock, time = times[i], start = times[i])
        run!(sim, steps = 0)                       # spectral -> grid: the truncated t0 state
        off = snapshot!(out, (i - 1) * per_out)
        run!(sim, period = Minute(round(Int, hours * 60)))
        if haskey(ENV, "FFC_SPEEDY_DEBUG")
            pz = vars.parameterizations
            ex(a) = (b = filter(isfinite, vec(Float32.(a))); isempty(b) ? (NaN, NaN, length(a)) : (extrema(b)..., count(!isfinite, a)))
            for k in (:sensible_heat_flux, :surface_air_temperature, :surface_wind_speed, :boundary_layer_drag)
                haskey(pz, k) && println(stderr, "dbg ", k, " ", ex(getproperty(pz, k).data))
            end
            for g in (:ocean, :land), k in keys(getproperty(pz, g))
                println(stderr, "dbg ", g, ".", k, " ", ex(getproperty(getproperty(pz, g), k).data))
            end
            println(stderr, "dbg sst ", ex(sst.data), " soil ", ex(st.data), " skt ", ex(skt))
        end
        snapshot!(out, off)
    end
    write(outpath, out)
    return nothing
end

write_geometry(joinpath(TMP, "geometry.bin"))
println("@@ready $NLAT $NLON $NLAYERS")
flush(stdout)
for line in eachline(stdin)
    parts = split(strip(line))
    isempty(parts) && continue
    parts[1] == "quit" && break
    if parts[1] == "run"
        t0 = time()
        try
            hours = parse(Float64, parts[4])
            n = parse(Int, parts[5])
            times = [DateTime(s, dateformat"yyyy-mm-ddTHH:MM") for s in parts[6:5+n]]
            run_states(parts[2], parts[3], hours, times)
            println("@@done $(time() - t0)")
        catch e
            println("@@error ", replace(sprint(showerror, e), '\n' => ' ')[1:min(end, 2000)])
        end
        flush(stdout)
    end
end
