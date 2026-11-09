"""The physics core F_phys: a spectral primitive-equation dynamical core that gives the 6 h increment of a state.

Step 4 (SpeedyWeather-style auxiliary structure): the neural operator does not learn the whole tendency, it
learns the residual of a global spectral GCM,

    dx/dt = F_phys(x) + R_theta(x, F_phys(x), t)

where F_phys is what a dynamical core on the sphere (spherical harmonics, sigma coordinates, semi-implicit
time stepping, hyperdiffusion) does to the ERA5 state x in 6 h. The core is the spatial "teacher": it knows
advection, the Coriolis force, the pressure-gradient force, hydrostatic balance and gravity-wave
propagation on the sphere exactly (to its truncation), so the operator only has to learn what the core
misses (subgrid physics, radiation, the surface, the core's own truncation error).

Backend. SpeedyWeather.jl (Julia) is the intended core; it cannot be installed in the cloud container
(julialang-s3 / pkg.julialang.org are blocked), so the backend here is **dinosaur** (NeuralGCM's JAX
spectral core, the same model class: T-truncated spherical harmonics, sigma levels, IMEX RK, moist
primitive equations). ``SpectralCore`` is the only place that knows the backend: anything that returns the
same (B, 66, H, W) normalized 6 h increment can replace it (a SpeedyWeather.jl run through juliacall,
or tendencies exported from a local SpeedyWeather run).

Mapping. KAI pressure levels (12, 50-1000 hPa) on the 5.625 deg equiangular grid ->
linear in latitude to the T21 Gaussian grid (same longitudes) -> pressure to sigma (dinosaur
``vertical_interpolation``, with the state's own surface pressure) -> modal State -> n IMEX RK3 steps ->
nodal u, v, T, q, geopotential (with moisture), sp -> sigma to pressure -> back to the KAI grid.
The increment is P(core_6h(s0)) - P(s0), with s0 the spectral state, so the interpolation round trip
(and the T21 truncation of x itself) cancels to first order: only what the dynamics changed survives.
Channels the core does not carry (t2m, tisr, tcwv, msl, skt) get a zero increment; the operator learns them.

The core runs without gradients (a fixed forcing for the operator): torch -> numpy -> jax -> torch.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from . import deps  # noqa: F401
from pgcfm import contract  # noqa: E402

N_DYN = contract.N_DYNAMIC
LEV = contract.N_LEVELS
# positions in the 66 dynamic channels (orog, channel 64 of 67, is the one static channel left out)
DYN_SLICE = {v: slice(k * LEV, (k + 1) * LEV) for k, v in enumerate(contract.UPPER_AIR_VARS)}
SP_DYN = 5 * LEV + contract.SURFACE_VARS.index("sp")


@dataclass
class CoreConfig:
    truncation: int = 21          # T21 Gaussian grid: 64 x 32, the 5.625 deg KAI grid's longitudes
    layers: int = 12              # equidistant sigma levels
    dt_minutes: float = 20.0      # IMEX RK3 step
    hours: float = 6.0            # increment window
    ref_temp: float = 250.0       # K, reference temperature of the semi-implicit scheme
    moist: bool = True            # q as a tracer with virtual-temperature effects
    diffusion_tau_h: float = 0.0  # 0 = dinosaur's resolution rule 8.6 h / 2.4^log2(nlat/128)
    orog_filter_order: int = 2
    dfi_hours: float = 0.0        # >0: digital filter initialization (removes the initial gravity-wave shock)


def _lat_interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(len(dst), len(src)) linear interpolation in latitude, linear extrapolation past the ends."""
    M = np.zeros((len(dst), len(src)))
    for i, y in enumerate(dst):
        j = int(np.clip(np.searchsorted(src, y) - 1, 0, len(src) - 2))
        w = (y - src[j]) / (src[j + 1] - src[j])
        M[i, j], M[i, j + 1] = 1.0 - w, w
    return M


class SpectralCore:
    """6 h increments of the dinosaur moist primitive-equation core for normalized KAI states.

    ``mean``, ``std`` are the 67-channel KAI statistics, ``lat_deg`` (S -> N) the KAI latitudes,
    ``orog_si`` (H, W) the surface geopotential [m2 s-2]. Call ``increment(x)`` with x (B, 66, H, W)
    normalized; it returns the normalized increment (B, 66, H, W), no gradient."""

    def __init__(self, cfg: CoreConfig, mean, std, lat_deg, orog_si, chunk: int = 16):
        import jax
        import jax.numpy as jnp
        from dinosaur import (coordinate_systems, filtering, primitive_equations, scales, sigma_coordinates,
                              spherical_harmonic, time_integration, vertical_interpolation)

        self.cfg, self.chunk = cfg, chunk
        units = scales.units
        dyn = contract.DYNAMIC_CHANNELS
        self.mean = np.asarray(mean, np.float64)[dyn]
        self.std = np.asarray(std, np.float64)[dyn]
        grid = getattr(spherical_harmonic.Grid, f"T{cfg.truncation}")()
        coords = coordinate_systems.CoordinateSystem(grid, sigma_coordinates.SigmaCoordinates.equidistant(cfg.layers))
        H = len(lat_deg)
        glat = np.degrees(np.arcsin(grid.nodal_axes[1]))
        glon = np.degrees(grid.nodal_axes[0])
        if grid.nodal_shape != (orog_si.shape[1], len(glat)):
            raise ValueError(f"T{cfg.truncation} grid {grid.nodal_shape} does not match the KAI grid {orog_si.shape}")
        self.to_g = jnp.asarray(_lat_interp_matrix(np.asarray(lat_deg, np.float64), glat), jnp.float32)
        self.to_k = jnp.asarray(_lat_interp_matrix(glat, np.asarray(lat_deg, np.float64)), jnp.float32)
        self.glat, self.glon = glat, glon
        specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
        nd = lambda q: float(specs.nondimensionalize(q))  # noqa: E731
        f_wind, f_T, f_len = nd(1 * units.m / units.s), nd(1 * units.degK), nd(1 * units.m)
        f_gp = nd(1 * units.m ** 2 / units.s ** 2)
        f_pa = nd(1 * units.Pa)
        pcoords = vertical_interpolation.PressureCoordinates(np.asarray(contract.LEVELS_HPA, np.float64))
        sig = coords.vertical
        ref = specs.nondimensionalize(cfg.ref_temp * units.degK) * np.ones((cfg.layers,), np.float32)

        to_g, to_k = self.to_g, self.to_k
        # below-ground pressure levels (p > sp) and the levels above the top sigma centre: unlimited linear
        # extrapolation in sigma (dinosaur's default returns NaN past one cell); the increment differences it
        lin_extrap = vertical_interpolation.vectorize_vertical_interpolation(
            vertical_interpolation.linear_interp_with_linear_extrap)

        def kai_to_g(a):   # (..., H, W) -> (..., W, Hg): dinosaur's (lon, lat) layout on Gaussian latitudes
            return jnp.einsum("gh,...hw->...wg", to_g, a)

        def g_to_kai(a):   # (..., W, Hg) -> (..., H, W)
            return jnp.einsum("hg,...wg->...hw", to_k, a)

        orog_nodal = kai_to_g(jnp.asarray(orog_si, jnp.float32))[None] / 9.80665 * f_len  # (1, W, Hg) height
        orog_modal = filtering.exponential_filter(grid, order=cfg.orog_filter_order)(grid.to_modal(orog_nodal))
        orog_nodal_f = grid.to_nodal(orog_modal)
        orog_m = orog_nodal_f / f_len
        eq_cls = primitive_equations.MoistPrimitiveEquations if cfg.moist else primitive_equations.PrimitiveEquations
        eq = eq_cls(ref, orog_modal, coords, specs)
        dt = specs.nondimensionalize(cfg.dt_minutes * units.minute)
        tau_h = cfg.diffusion_tau_h or 8.6 / (2.4 ** np.log2(len(glat) / 128))
        hyper = time_integration.horizontal_diffusion_step_filter(grid, dt=dt, tau=specs.nondimensionalize(tau_h * units.hour), order=2)
        n_steps = int(round(cfg.hours * 60 / cfg.dt_minutes))
        integrate = time_integration.repeated(time_integration.step_with_filters(time_integration.imex_rk_sil3(eq, dt), [hyper]), n_steps)
        self.n_steps = n_steps
        dfi = None
        if cfg.dfi_hours > 0:
            span = specs.nondimensionalize(cfg.dfi_hours * units.hour)
            dfi = time_integration.digital_filter_initialization(
                equation=eq, ode_solver=time_integration.imex_rk_sil3, filters=[hyper],
                time_span=span, cutoff_period=span, dt=dt)

        def to_pressure(state):
            """Spectral state -> KAI-grid (u, v, t, q, z) on pressure levels in SI, and sp [Pa]."""
            u, v = spherical_harmonic.vor_div_to_uv_nodal(grid, state.vorticity, state.divergence)
            t = grid.to_nodal(state.temperature_variation) + ref[:, None, None]
            q = grid.to_nodal(state.tracers["specific_humidity"]) if cfg.moist else None
            z = primitive_equations.get_geopotential_on_sigma(
                t, q, nodal_orography=orog_nodal_f, sigma=sig, gravity_acceleration=specs.g,
                ideal_gas_constant=specs.R, water_vapor_gas_constant=specs.R_vapor if cfg.moist else None)
            sp = jnp.exp(grid.to_nodal(state.log_surface_pressure))  # nondimensional
            sp_hpa = sp / f_pa / 100.0
            fields = {"u": u / f_wind, "v": v / f_wind, "t": t / f_T, "z": z / f_gp}
            if cfg.moist:
                fields["q"] = q
            fp = vertical_interpolation.interp_sigma_to_pressure(fields, pcoords, sig, sp_hpa, lin_extrap)
            out = {k: g_to_kai(a) for k, a in fp.items()}
            out["sp"] = g_to_kai(sp[0] / f_pa)
            return out

        def states(x):
            """x (66, H, W) SI -> the spectral state s0 and the core's state 6 h later."""
            # surface pressure on the core's (filtered T21) orography from ERA5 geopotential on pressure levels
            # (dinosaur get_surface_pressure); ERA5's own sp sits on the real terrain and the mismatch launches
            # a ~25 hPa / 6 h adjustment over mountains
            z_g = kai_to_g(x[DYN_SLICE["z"]])
            sp_g = 100.0 * vertical_interpolation.get_surface_pressure(pcoords, z_g, orog_m, 9.80665)  # (1, W, Hg) Pa
            p = {k: kai_to_g(x[DYN_SLICE[k]]) for k in ("u", "v", "t", "q")}
            s = vertical_interpolation.interp_pressure_to_sigma(p, pcoords, sig, sp_g / 100.0)
            vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(grid, s["u"] * f_wind, s["v"] * f_wind)
            tracers = {"specific_humidity": grid.to_modal(jnp.clip(s["q"], 0.0, None))} if cfg.moist else {}
            s0 = primitive_equations.State(
                vorticity=vor, divergence=div,
                temperature_variation=grid.to_modal(s["t"] * f_T - ref[:, None, None]),
                log_surface_pressure=grid.to_modal(jnp.log(sp_g * f_pa)), tracers=tracers)
            if dfi is not None:
                s0 = dfi(s0)
            return s0, integrate(s0)

        def one(x):
            s0, s1 = states(x)
            a, b = to_pressure(s0), to_pressure(s1)
            inc = jnp.zeros_like(x)
            for k in ("z", "t", "u", "v") + (("q",) if cfg.moist else ()):
                inc = inc.at[DYN_SLICE[k]].set(b[k] - a[k])
            return inc.at[SP_DYN].set(b["sp"] - a["sp"])

        self._fn = jax.jit(jax.vmap(one))
        self._diag = jax.jit(jax.vmap(lambda x: tuple(to_pressure(s) for s in states(x))))
        self._jnp = jnp

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def increment_si(self, x_si: np.ndarray) -> np.ndarray:
        """(B, 66, H, W) SI -> (B, 66, H, W) SI increment over ``cfg.hours``."""
        out = [np.asarray(self._fn(self._jnp.asarray(x_si[i:i + self.chunk], self._jnp.float32)))
               for i in range(0, len(x_si), self.chunk)]
        return np.concatenate(out, 0)

    @torch.no_grad()
    def increment(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized (B, 66, H, W) -> normalized 6 h increment (B, 66, H, W); finite (NaN -> 0)."""
        m = self.mean.reshape(1, -1, 1, 1)
        s = self.std.reshape(1, -1, 1, 1)
        x_si = x.detach().double().cpu().numpy() * s + m
        d = self.increment_si(x_si) / s
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(d, dtype=x.dtype, device=x.device)
