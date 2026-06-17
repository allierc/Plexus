"""well_engine: the generic interpreter for The Well PDE specs.

Mirrors the water prototype's engine (build a Hierarchy from a validated spec, run
the schedule) but for the *field* half: it builds MultiFields and/or a particle
`Level`, applies declared initial conditions, then iterates the schedule. It holds
NO scenario-specific logic -- every Well dataset is a different spec, never an
engine edit (the spec-is-the-API claim of plexus.tex Part II).

Integration model (the framework contract): a FIELD self-update operator owns its
sub-dt and mutates its grid in place (returns {}); a SET operator returns a
per-level delta which the engine sums and integrates once per tick (1st-order:
x += dt*delta). Field histories and particle states are recorded for rendering.
"""

from __future__ import annotations

import os
import math
import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
from well_fields import MultiField


# --------------------------------------------------------------------------- #
#  Initial-condition builders (declared in spec `init:`)
# --------------------------------------------------------------------------- #
def _grid_xy(nx, ny, dx, device):
    x = (torch.arange(nx, device=device).float() + 0.5) * dx
    y = (torch.arange(ny, device=device).float() + 0.5) * dx
    return x[:, None].expand(nx, ny), y[None, :].expand(nx, ny)


def _seed_clusters(shape, n, radius, g, device, val=1.0):
    """Sum of n Gaussian bumps at random centres -> field in [0, val]."""
    nx, ny = shape
    out = torch.zeros(nx, ny, device=device)
    cx = torch.randint(0, nx, (n,), generator=g, device=device)
    cy = torch.randint(0, ny, (n,), generator=g, device=device)
    X, Y = _grid_xy(nx, ny, 1.0, device)
    for i in range(n):
        out = out + torch.exp(-(((X - cx[i].float()) ** 2 + (Y - cy[i].float()) ** 2) / (2 * radius ** 2)))
    return (out.clamp(0, 1) * val)


def _random_fourier(shape, kmax, g, device):
    """Smooth random field: low-wavenumber Fourier series, normalized to [0,1]."""
    nx, ny = shape
    X, Y = _grid_xy(nx, ny, 1.0 / max(nx, ny), device)
    out = torch.zeros(nx, ny, device=device)
    for _ in range(8):
        kx = torch.randint(1, kmax + 1, (1,), generator=g, device=device).item()
        ky = torch.randint(1, kmax + 1, (1,), generator=g, device=device).item()
        ph = torch.rand(1, generator=g, device=device).item() * 2 * math.pi
        amp = torch.rand(1, generator=g, device=device).item()
        out = out + amp * torch.sin(2 * math.pi * (kx * X + ky * Y) + ph)
    out = out - out.min()
    return out / out.max().clamp(min=1e-6)


def _rho_inclusions(shape, dx, g, device, n_incl=8, base=1.0):
    """Heterogeneous density: smooth background + ellipsoidal inclusions
    (the_well/acoustic_scattering_inclusions). Returns rho>0 [nx,ny]."""
    nx, ny = shape
    X, Y = _grid_xy(nx, ny, dx, device)
    rho = torch.full((nx, ny), base, device=device)
    # smooth background gradient (two random gaussian bumps)
    for _ in range(2):
        cx = torch.rand(1, generator=g, device=device).item()
        cy = torch.rand(1, generator=g, device=device).item()
        sig = 0.2 + 0.4 * torch.rand(1, generator=g, device=device).item()
        peak = 1.0 + 4.0 * torch.rand(1, generator=g, device=device).item()
        rho = rho + peak * torch.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2)))
    # ellipsoidal inclusions of very different density
    for _ in range(n_incl):
        cx = torch.rand(1, generator=g, device=device).item()
        cy = torch.rand(1, generator=g, device=device).item()
        a = 0.03 + 0.10 * torch.rand(1, generator=g, device=device).item()
        b = 0.03 + 0.10 * torch.rand(1, generator=g, device=device).item()
        ang = (torch.rand(1, generator=g, device=device).item() - 0.5) * math.pi / 2
        ca, sa = math.cos(ang), math.sin(ang)
        xr = (X - cx) * ca + (Y - cy) * sa
        yr = -(X - cx) * sa + (Y - cy) * ca
        inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        lr = -1.0 + 11.0 * torch.rand(1, generator=g, device=device).item()   # ln(rho) ~ U(-1,10) (the_well)
        rho = torch.where(inside, torch.full_like(rho, math.exp(min(lr, 3.0))), rho)
    return rho.clamp(min=0.2, max=25.0)


def _pressure_rings(shape, dx, g, device, n_rings=3):
    """Initial pressure: a few high-pressure rings (the_well acoustic IC)."""
    nx, ny = shape
    X, Y = _grid_xy(nx, ny, dx, device)
    p = torch.zeros(nx, ny, device=device)
    W = nx * dx
    for _ in range(n_rings):
        cx = (0.2 + 0.6 * torch.rand(1, generator=g, device=device).item()) * W
        cy = 0.2 + 0.6 * torch.rand(1, generator=g, device=device).item()
        r0 = 0.06 + 0.09 * torch.rand(1, generator=g, device=device).item()
        inten = 0.5 + 1.5 * torch.rand(1, generator=g, device=device).item()
        rr = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        p = p + inten * torch.exp(-((rr - r0) ** 2) / (2 * (0.02) ** 2))
    return p


def _raster(shapes, ny, width, device):
    """Rasterize rectangles [x0,y0,x1,y1] / circles [cx,cy,r] to a [nx,ny] mask."""
    dx = 1.0 / ny
    nx = int(round(width * ny))
    X, Y = _grid_xy(nx, ny, dx, device)
    m = torch.zeros(nx, ny, dtype=torch.bool, device=device)
    for r in shapes:
        if len(r) == 3:
            m |= (X - r[0]) ** 2 + (Y - r[1]) ** 2 <= r[2] ** 2
        else:
            m |= (X >= r[0]) & (X <= r[2]) & (Y >= r[1]) & (Y <= r[3])
    return m


# --------------------------------------------------------------------------- #
#  Build
# --------------------------------------------------------------------------- #
_BC = {"periodic": "periodic", "wall": "neumann", "neumann": "neumann", "open": "open"}


def build(sc, device="cpu"):
    g = torch.Generator(device=device).manual_seed(sc.seed)
    H = Hierarchy()
    H.config = sc
    H.rng = torch.Generator(device=device).manual_seed(sc.seed + 777)
    H.world_width = sc.world

    # --- fields ---
    for fname, fs in sc.fields.items():
        bcx = _BC[fs.get("bc_x", fs.get("bc", "periodic"))]
        bcy = _BC[fs.get("bc_y", fs.get("bc", "periodic"))]
        obstacles = fs.get("obstacles", [])
        ny = int(fs["res"])
        walls = _raster(obstacles, ny, sc.world, device) if obstacles else None
        fld = MultiField(fname, fs.get("couples_to", "none"), channels=int(fs["channels"]),
                         ny=ny, width=sc.world, device=device, bc=(bcx, bcy), walls=walls)
        H.add_field(fld)

    # --- particle set (active matter) ---
    for sname, ss in sc.sets.items():
        N = int(ss["n"])
        pos = torch.rand(N, 2, generator=g, device=device)
        theta = torch.rand(N, generator=g, device=device) * 2 * math.pi
        # state = [x, y, vx, vy, theta]
        state = torch.zeros(N, 5, device=device)
        state[:, :2] = pos
        state[:, 4] = theta
        lvl = Level(sname, level=0, state=state)
        # species label (for multi-species slime): even split, deterministic
        K = int(ss.get("species", 1))
        node_type = torch.arange(N, device=device) % K
        lvl.register_buffer("node_type", node_type)
        lvl.n_species = K
        H.add_level(lvl)

    # --- initial conditions (declared) ---
    for fname, spec in sc.init.items():
        if fname in H.fields:
            _init_field(H.fields[fname], spec, g, device, sc)
        # sets are randomized at build; spec may override below

    return H


def _init_field(fld, spec, g, device, sc):
    nx, ny = fld.nx, fld.ny
    # static coefficient map (acoustic rho)
    rho_spec = spec.get("rho")
    if rho_spec is not None:
        kind = rho_spec.get("kind", "inclusions")
        if kind == "inclusions":
            fld.coeffs["rho"] = _rho_inclusions((nx, ny), fld.dx, g, device,
                                                n_incl=int(rho_spec.get("n", 8)),
                                                base=float(rho_spec.get("base", 1.0)))
        elif kind == "split":                       # discontinuous interface: two halves
            rho = torch.ones(nx, ny, device=device)
            lo = float(rho_spec.get("lo", 1.0)); hi = float(rho_spec.get("hi", 5.0))
            rho[: nx // 2] = lo; rho[nx // 2:] = hi
            fld.coeffs["rho"] = rho
        elif kind == "uniform":
            fld.coeffs["rho"] = torch.full((nx, ny), float(rho_spec.get("value", 1.0)), device=device)
        elif kind == "lens":                        # converging acoustic lens: a slow disc
            X, Y = _grid_xy(nx, ny, fld.dx, device)
            cx = float(rho_spec.get("cx", 0.5 * sc.world)); cy = float(rho_spec.get("cy", 0.5))
            r = float(rho_spec.get("r", 0.18)); hi = float(rho_spec.get("rho", 6.0))
            inside = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
            fld.coeffs["rho"] = torch.where(inside, torch.full((nx, ny), hi, device=device),
                                            torch.ones(nx, ny, device=device))
        elif kind == "gradient":                    # smooth vertical density gradient (refracting layer)
            X, Y = _grid_xy(nx, ny, fld.dx, device)
            lo = float(rho_spec.get("lo", 1.0)); hi = float(rho_spec.get("hi", 6.0))
            fld.coeffs["rho"] = lo + (hi - lo) * Y
    # channel inits
    for ch_name, ch_spec in spec.items():
        if ch_name in ("rho",):
            continue
        ci = {"A": 0, "B": 1, "p": 0, "u": 1, "v": 2}.get(ch_name)
        if ci is None:
            continue
        if isinstance(ch_spec, (int, float)):
            fld.grid[ci] = float(ch_spec)
        elif isinstance(ch_spec, dict):
            seed = ch_spec.get("seed")
            if seed == "clusters":
                fld.grid[ci] = _seed_clusters((nx, ny), int(ch_spec.get("n", 25)),
                                              float(ch_spec.get("radius", 3.0)), g, device,
                                              float(ch_spec.get("value", 1.0)))
            elif seed == "fourier":
                f = _random_fourier((nx, ny), int(ch_spec.get("kmax", 6)), g, device)
                fld.grid[ci] = (f > float(ch_spec.get("thresh", 0.6))).float() * float(ch_spec.get("value", 1.0))
            elif seed == "square":
                s = ch_spec.get("box", [0.45, 0.45, 0.55, 0.55])
                m = _raster([s], ny, sc.world, device)
                fld.grid[ci] = torch.where(m, torch.full_like(fld.grid[ci], float(ch_spec.get("value", 1.0))),
                                           fld.grid[ci])
            elif seed == "rings":
                fld.grid[ci] = _pressure_rings((nx, ny), fld.dx, g, device, int(ch_spec.get("n", 3)))
            elif seed == "point":                  # deterministic gaussian pressure source
                X, Y = _grid_xy(nx, ny, fld.dx, device)
                cx = float(ch_spec.get("cx", 0.5 * sc.world)); cy = float(ch_spec.get("cy", 0.5))
                r = float(ch_spec.get("r", 0.02)); inten = float(ch_spec.get("inten", 2.0))
                fld.grid[ci] = inten * torch.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r ** 2)))
    # walls hold nothing
    if fld.walls.any():
        fld.grid = torch.where(fld.walls[None].expand_as(fld.grid),
                               torch.zeros_like(fld.grid), fld.grid)


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
def run(sc, device="cpu"):
    H = build(sc, device)
    inst = [(o.op, get_operator(o.op)({**o.params, "field": o.params.get("field", o.on)}, device), o.on)
            for o in sc.operators]

    re = sc.record_every
    n_rec = sc.n_frames // re + 1
    field_hist = {fn: np.zeros((n_rec, fl.grid.shape[0], fl.nx, fl.ny), np.float32)
                  for fn, fl in H.fields.items()}
    has_set = len(H.levels) > 0
    if has_set:
        lvl0 = next(iter(H.levels.values()))
        set_hist = np.zeros((n_rec, lvl0.n, 3), np.float32)   # x, y, theta
    rec = 0
    dt = 1.0
    for frame in range(sc.n_frames + 1):
        # zero per-level delta accumulators
        deltas = {name: torch.zeros_like(l.state[:, :2]) for name, l in H.levels.items()}
        for step in sc.schedule:
            for tok in (step if isinstance(step, list) else [step]):
                for nm, ob, on in inst:
                    if nm != tok:
                        continue
                    out = ob(H, None)
                    for lvl_name, d in out.items():
                        deltas[lvl_name] = deltas[lvl_name] + d
        # integrate sets (1st order, periodic torus)
        for name, l in H.levels.items():
            l.state[:, :2] = torch.remainder(l.state[:, :2] + dt * deltas[name], 1.0)
        if frame % re == 0:
            for fn, fl in H.fields.items():
                field_hist[fn][rec] = fl.grid.detach().cpu().numpy()
            if has_set:
                set_hist[rec, :, :2] = lvl0.state[:, :2].detach().cpu().numpy()
                set_hist[rec, :, 2] = lvl0.state[:, 4].detach().cpu().numpy()
            rec += 1

    out = dict(name=sc.name, fields={fn: h[:rec] for fn, h in field_hist.items()},
               coeffs={fn: {k: v.detach().cpu().numpy() for k, v in fl.coeffs.items() if not k.startswith("_")}
                       for fn, fl in H.fields.items()},
               walls={fn: (fl.walls.detach().cpu().numpy() if fl.walls.any() else None)
                      for fn, fl in H.fields.items()},
               world=sc.world)
    if has_set:
        out["set"] = set_hist[:rec]
        out["species"] = (lvl0.node_type.detach().cpu().numpy()
                          if hasattr(lvl0, "node_type") else np.zeros(lvl0.n, int))
    return out
