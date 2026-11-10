"""SpeedyWeather.jl as the physics core F_phys (Step 5): the backend swap that Step 4 could not do.

Same contract as ``ffc.physcore.SpectralCore`` (the dinosaur backend): ``increment(x, hours)`` takes normalized
KAI states (B, 66, H, W) and returns the normalized 6 h increment P(core_6h(s0)) - P(s0), no gradient. What
changes is the model behind it: SpeedyWeather.jl's ``PrimitiveWetModel`` (spectral T21 on the 32 x 64 full
Gaussian grid, sigma levels, leapfrog + semi-implicit, hyperdiffusion) **with its physics**: large-scale
condensation, Betts-Miller convection, one-band short/longwave radiation with the true solar zenith angle
of the state's time, a bulk Richardson boundary layer and surface fluxes of momentum, heat and moisture over
ocean (SST) and land (soil temperature), both set from the state's ERA5 skin temperature. ``physics="none"``
switches all parameterizations off (the adiabatic core, the dinosaur arm's equivalent).

Julia runs in a separate process (``ffc/speedy_worker.jl``), started once, fed through files in /dev/shm and
a line protocol, so Julia's runtime never shares a process with torch. The KAI <-> sigma mapping stays in
Python and reuses dinosaur's vertical interpolation, so the two backends see exactly the same mapping:

KAI 12 pressure levels on the 5.625 deg grid -> linear in latitude to the 32 Gaussian latitudes (same
longitudes) -> surface pressure on SpeedyWeather's own T21 orography from the ERA5 geopotential profile
(dinosaur ``get_surface_pressure``) -> pressure to sigma (dinosaur ``interp_pressure_to_sigma``) -> Julia:
set u, v, T, q, ln ps, integrate ``hours`` -> u, v, T, q, ln ps at t0 (spectrally truncated) and at t0 + 6 h
-> geopotential on sigma (hydrostatic, virtual temperature, from the model orography) -> sigma to pressure
(linear extrapolation below ground) -> KAI grid. Channels the core does not carry (t2m, tisr, tcwv, msl, skt)
get a zero increment.

Install (no Julia package server needed, only github.com over git and conda-forge):
``scripts/install_speedyweather.sh``. The Julia binary and project are found through ``FFC_JULIA`` /
``FFC_JULIA_PROJECT`` (defaults /opt/mamba/envs/jl/bin/julia, /opt/jsw).
"""

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import deps  # noqa: F401
from .physcore import DYN_SLICE, SP_DYN, _lat_interp_matrix
from pgcfm import contract  # noqa: E402

N_DYN = contract.N_DYNAMIC
SKT_DYN = contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["skt"])
G = 9.80665
RD, RV = 287.0, 461.5
EPS_V = RV / RD - 1.0   # virtual temperature: Tv = T (1 + 0.608 q)
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speedy_worker.jl")


@dataclass
class SpeedyConfig:
    layers: int = 18              # equidistant sigma levels (SpeedyWeather default spacing)
    dt_minutes: float = 30.0      # leapfrog step
    hours: float = 6.0            # increment window
    physics: str = "all"          # "all" | "none" (adiabatic) | comma list of parameterizations to switch off
    year: int = 2015              # nominal year for the solar zenith angle (hours are since Jan 1 00 UTC)


class SpeedyCore:
    """6 h increments of SpeedyWeather.jl's PrimitiveWetModel for normalized KAI states.

    ``mean``, ``std`` are the 67-channel KAI statistics, ``lat_deg`` (S -> N) the KAI latitudes. ``orog_si`` is
    accepted for interface parity with ``SpectralCore`` and unused: the core runs on its own orography and the
    surface pressure is recomputed on it."""

    def __init__(self, cfg: SpeedyConfig, mean, std, lat_deg, orog_si=None, chunk: int = 64):
        from dinosaur import sigma_coordinates, vertical_interpolation as vi

        self.cfg, self.chunk = cfg, chunk
        dyn = contract.DYNAMIC_CHANNELS
        self.mean = np.asarray(mean, np.float64)[dyn]
        self.std = np.asarray(std, np.float64)[dyn]
        # exchange files: /dev/shm when it can hold a full chunk (in + out, 2x margin), else the disk temp dir
        # (Docker's default /dev/shm is 64 MiB; a 64-state chunk at 18 layers needs ~110 MiB)
        need = 2 * chunk * 4 * (6 * cfg.layers + 6) * 32 * 64 * 2
        shm = "/dev/shm" if os.path.isdir("/dev/shm") and shutil.disk_usage("/dev/shm").free > need else None
        self.tmp = tempfile.mkdtemp(prefix="speedy_", dir=shm)
        julia = os.environ.get("FFC_JULIA", "/opt/mamba/envs/jl/bin/julia")
        project = os.environ.get("FFC_JULIA_PROJECT", "/opt/jsw")
        env = dict(os.environ)
        env.setdefault("JULIA_NUM_THREADS", "1")
        env.setdefault("JULIA_PKG_SERVER", "")
        self.proc = subprocess.Popen(
            [julia, f"--project={project}", "--startup-file=no", WORKER, str(cfg.layers), str(cfg.dt_minutes),
             str(cfg.physics), self.tmp],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
        line = self._reply()
        _, nlat, nlon, nlev = line.split()
        self.nlat, self.nlon, self.nlev = int(nlat), int(nlon), int(nlev)
        geo = np.fromfile(os.path.join(self.tmp, "geometry.bin"), np.float32)
        o = 0
        glat_ns = geo[o:o + self.nlat].astype(np.float64); o += self.nlat
        self.sig_half = geo[o:o + self.nlev + 1].astype(np.float64); o += self.nlev + 1
        self.sig_full = geo[o:o + self.nlev].astype(np.float64); o += self.nlev
        npt = self.nlat * self.nlon
        orog_ns = geo[o:o + npt].reshape(self.nlat, self.nlon); o += npt
        self.land_ns = geo[o:o + npt].reshape(self.nlat, self.nlon)
        lat = np.asarray(lat_deg, np.float64)
        if len(lat) != self.nlat or (orog_si is not None and np.shape(orog_si)[1] != self.nlon):
            raise ValueError(f"SpeedyWeather grid {self.nlat} x {self.nlon} does not match the KAI grid")
        # Python side works S -> N (KAI order); Julia rings are N -> S
        self.glat = glat_ns[::-1].copy()
        self.to_g = _lat_interp_matrix(lat, self.glat)   # (Hg, H)
        self.to_k = _lat_interp_matrix(self.glat, lat)   # (H, Hg)
        self.orog_m = orog_ns[::-1].astype(np.float64)   # (Hg, W) metres, S -> N
        self.vi = vi
        self.pcoords = vi.PressureCoordinates(np.asarray(contract.LEVELS_HPA, np.float64))
        self.sig = sigma_coordinates.SigmaCoordinates(self.sig_half)
        import jax.numpy as jnp
        self.lin_extrap = vi.vectorize_vertical_interpolation(vi.linear_interp_with_linear_extrap)
        # u, v, T, q below ground: the lowest model level's value (no extrapolation). SpeedyWeather's surface
        # fluxes reshape the lowest layers' lapse rate within 6 h, and extrapolating that below the terrain
        # turns into 5-10 K "increments" at 850-1000 hPa under mountains and ice sheets (2016 check: t850
        # increment RMSE 5.3 K, 1.3 K above ground); z keeps the hydrostatic linear extrapolation.
        self.below_const = vi.vectorize_vertical_interpolation(
            lambda x, xp, fp: jnp.where(x > xp[-1], fp[-1], vi.linear_interp_with_linear_extrap(x, xp, fp)))
        self.seconds = 0.0

    # -- process -------------------------------------------------------------------------------------------
    def _reply(self) -> str:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("SpeedyWeather worker exited")
            if line.startswith("@@error"):
                raise RuntimeError("SpeedyWeather worker: " + line[8:].strip())
            if line.startswith("@@"):
                return line[2:].strip()

    def close(self):
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(10)
            except Exception:
                self.proc.kill()
        self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def config_dict(self) -> dict:
        return {"backend": "speedyweather", **asdict(self.cfg)}

    # -- mapping -------------------------------------------------------------------------------------------
    def _to_g(self, a):   # (..., H, W) -> (..., Hg, W)
        return np.einsum("gh,...hw->...gw", self.to_g, a)

    def _to_k(self, a):   # (..., Hg, W) -> (..., H, W)
        return np.einsum("hg,...gw->...hw", self.to_k, a)

    def _p_to_sigma(self, x):
        """One SI state (66, H, W) -> sigma-level u, v, T, q (L, Hg, W), ln ps (Hg, W), skt (Hg, W)."""
        vi = self.vi
        # dinosaur's vertical helpers are (level, lon, lat); transpose the trailing axes in and out
        z = self._to_g(x[DYN_SLICE["z"]]).transpose(0, 2, 1)
        sp = 100.0 * np.asarray(vi.get_surface_pressure(self.pcoords, z, self.orog_m.T[None], G))  # (1, W, Hg) Pa
        p = {k: self._to_g(x[DYN_SLICE[k]]).transpose(0, 2, 1) for k in ("u", "v", "t", "q")}
        s = vi.interp_pressure_to_sigma(p, self.pcoords, self.sig, sp / 100.0)
        out = {k: np.asarray(a).transpose(0, 2, 1) for k, a in s.items()}
        out["q"] = np.clip(out["q"], 0.0, None)
        return out, np.log(sp[0].T), self._to_g(x[SKT_DYN])

    def _geopotential(self, t, q):
        """Hydrostatic geopotential on sigma full levels (L, Hg, W) from T, q and the model orography."""
        tv = t * (1.0 + EPS_V * q)
        phi_half = G * self.orog_m                              # surface
        phi = np.empty_like(t)
        sh, sf = self.sig_half, self.sig_full
        for k in range(self.nlev - 1, -1, -1):                  # bottom -> top
            phi[k] = phi_half + RD * tv[k] * np.log(sh[k + 1] / sf[k])
            phi_half = phi_half + RD * tv[k] * np.log(sh[k + 1] / max(sh[k], 1e-6)) if k > 0 else phi_half
        return phi

    def _sigma_to_kai(self, u, v, t, q, lnps):
        """Sigma fields (L, Hg, W) + ln ps -> KAI-grid dict of pressure-level fields and sp [Pa]."""
        z = self._geopotential(t, q)
        ps_hpa = np.exp(lnps) / 100.0
        tr = lambda a: a.transpose(0, 2, 1)  # noqa: E731
        fp = self.vi.interp_sigma_to_pressure({"u": tr(u), "v": tr(v), "t": tr(t), "q": tr(q)},
                                              self.pcoords, self.sig, ps_hpa.T[None], self.below_const)
        fp["z"] = self.vi.interp_sigma_to_pressure({"z": tr(z)}, self.pcoords, self.sig, ps_hpa.T[None],
                                                   self.lin_extrap)["z"]
        out = {k: self._to_k(np.asarray(a).transpose(0, 2, 1)) for k, a in fp.items()}
        out["sp"] = self._to_k(np.exp(lnps))
        return out

    # -- increments ----------------------------------------------------------------------------------------
    def _times(self, hours):
        base = np.datetime64(f"{self.cfg.year}-01-01T00:00")
        hy = 8760.0
        return [str(base + np.timedelta64(int(round(h % hy * 60)), "m"))[:16] for h in hours]

    def increment_si(self, x_si: np.ndarray, hours=None) -> np.ndarray:
        """(B, 66, H, W) SI (+ hours since Jan 1 00 UTC) -> (B, 66, H, W) SI increment over ``cfg.hours``."""
        hours = np.zeros(len(x_si)) if hours is None else np.asarray(hours, np.float64).reshape(-1)
        return np.concatenate([self._run(x_si[i:i + self.chunk], hours[i:i + self.chunk])
                               for i in range(0, len(x_si), self.chunk)], 0)

    def _run(self, xs, hours):
        L, Hg, W = self.nlev, self.nlat, self.nlon
        buf = []
        for x in xs:
            s, lnps, skt = self._p_to_sigma(x)
            for k in ("u", "v", "t", "q"):
                buf.append(s[k][:, ::-1].astype(np.float32).ravel())     # Julia rings N -> S
            buf.append(lnps[::-1].astype(np.float32).ravel())
            buf.append(skt[::-1].astype(np.float32).ravel())
        fin, fout = os.path.join(self.tmp, "in.bin"), os.path.join(self.tmp, "out.bin")
        np.concatenate(buf).tofile(fin)
        t0 = time.time()
        self.proc.stdin.write(f"run {fin} {fout} {self.cfg.hours} {len(xs)} {' '.join(self._times(hours))}\n")
        self.proc.stdin.flush()
        self._reply()
        self.seconds += time.time() - t0
        y = np.fromfile(fout, np.float32).astype(np.float64)
        per = 4 * L * Hg * W + Hg * W
        y = y.reshape(len(xs), 2, per)
        incs = np.zeros(xs.shape, np.float64)
        for i in range(len(xs)):
            st = []
            for j in range(2):
                a = y[i, j]
                f = [a[n * L * Hg * W:(n + 1) * L * Hg * W].reshape(L, Hg, W)[:, ::-1] for n in range(4)]
                lnps = a[4 * L * Hg * W:].reshape(Hg, W)[::-1]
                st.append(self._sigma_to_kai(*f, lnps))
            a0, a1 = st
            for k in ("z", "t", "u", "v", "q"):
                incs[i, DYN_SLICE[k]] = a1[k] - a0[k]
            incs[i, SP_DYN] = a1["sp"] - a0["sp"]
        return incs

    @torch.no_grad()
    def increment(self, x: torch.Tensor, hours=None) -> torch.Tensor:
        """Normalized (B, 66, H, W) -> normalized 6 h increment (B, 66, H, W); finite (NaN -> 0)."""
        m = self.mean.reshape(1, -1, 1, 1)
        s = self.std.reshape(1, -1, 1, 1)
        x_si = x.detach().double().cpu().numpy() * s + m
        h = None if hours is None else torch.as_tensor(hours).detach().double().cpu().numpy()
        d = self.increment_si(x_si, h) / s
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(d, dtype=x.dtype, device=x.device)
