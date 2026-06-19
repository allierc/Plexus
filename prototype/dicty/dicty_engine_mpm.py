"""dicty_engine_mpm.py -- the SOFT-MPM cell-body arm of the dicty model.

Same chemistry/operators as the point-cell engine, but each cell is a small deformable body of
`per_parent` (default 8) MLS-MPM particles instead of a point. The registered `mpm` operator does
the soft mechanics (true volume exclusion + deformation through the grid), so the point-cell
`spring` op is dropped. Chemotaxis still drives motion: `sense` -> H.cell_accel -> `mpm` broadcasts
it to each cell's particles. `inflow_mpm` is the SOURCE (wake a dormant cell = activate its
particle disc at an independent position). cAMP is the same diffuse+decay GridField.

run() returns the SAME hist dict structure as dicty_engine (pos = active CELL centroids, count,
field, pos_full, active), so eval_sweeps/opt_dicty metrics apply unchanged -> directly comparable
to the point-cell arm. Select it with eval's DICTY_ENGINE=dicty_engine_mpm.

Reuses engine2's MPM build idioms + grow_engine_mpm's dormant-buffer scheme.
"""
from __future__ import annotations

import os, math
import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
import mpm as _mpm                       # registers `mpm`
import ops as _ops                       # registers secrete/sense/random_walk
import dicty_ops_mpm as _dmpm            # registers `inflow_mpm`
from grid_field import GridField

EPS = 1e-6
NU = 0.2


def _lame(E):
    return E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))


def build(sc, device="cpu"):
    g = torch.Generator(device=device).manual_seed(sc.seed)
    H = Hierarchy(); H.config = sc
    H.rng = torch.Generator(device=device).manual_seed(sc.seed + 12345)
    H.world_width = float(getattr(sc, "world", 1.0))
    H.periodic = (getattr(sc, "boundary", "wall") == "periodic")
    H.walls_mpm = None

    cs = sc.sets["cell"]; ps = sc.sets["particle"]
    N0 = int(cs["n"]); Nmax = int(cs.get("buffer", N0))
    ppc = int(ps.get("per_parent", 8)); rad = float(ps.get("radius", 0.012)); rho = float(ps.get("density", 1.0))
    youngs = float(next(iter(cs.get("types", {"dicty": {"youngs": 60.0}}).values())).get("youngs", 60.0))
    H.ppc = ppc; H.rad = rad

    # --- cell level: fixed buffer, N0 active ---
    cell = Level("cell", level=1, state=torch.zeros(Nmax, 4, device=device))
    cell.type_names = list(cs.get("types", {"dicty": {}}).keys()) or ["dicty"]
    cell.register_buffer("node_type", torch.zeros(Nmax, dtype=torch.long, device=device))
    H.add_level(cell)
    H.c_active = torch.zeros(Nmax, dtype=torch.bool, device=device); H.c_active[:N0] = True

    # cell centres: seed active from real frame-0 positions (init_npz), else uniform
    init_npz = cs.get("init_npz")
    centers = torch.rand(Nmax, 2, generator=g, device=device); centers[:, 0] *= H.world_width
    if init_npz is not None:
        path = init_npz if os.path.isabs(init_npz) else os.path.join(os.path.dirname(__file__), init_npz)
        ip = torch.tensor(np.load(path)["init_pos"], dtype=torch.float32, device=device)
        M = ip.shape[0]
        sel = (torch.randperm(M, generator=g, device=device)[:N0] if N0 <= M
               else torch.randint(M, (N0,), generator=g, device=device))
        centers[:N0] = ip[sel]
    cell.state[:, :2] = centers

    # --- particle level: ppc per cell; dormant cells carry mass 0 (parked) ---
    Np = Nmax * ppc
    parent = torch.arange(Nmax, device=device).repeat_interleave(ppc)
    r = torch.sqrt(torch.rand(Np, generator=g, device=device)) * rad
    th = torch.rand(Np, generator=g, device=device) * 2 * math.pi
    ppos = centers[parent] + torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    active_p = H.c_active[parent]
    ppos = torch.where(active_p[:, None], ppos, torch.full_like(ppos, -1.0))   # park dormant off-domain
    part = Level("particle", level=0, state=torch.cat([ppos, torch.zeros(Np, 2, device=device)], 1))
    part.register_buffer("parent", parent)
    part.register_buffer("C", torch.zeros(Np, 2, 2, device=device))
    part.register_buffer("F", torch.eye(2, device=device).expand(Np, 2, 2).contiguous())
    mu, la = _lame(youngs)
    part.register_buffer("mu", torch.full((Np,), mu, device=device))
    part.register_buffer("la", torch.full((Np,), la, device=device))
    p_vol = math.pi * rad * rad / ppc
    H.p_mass0 = p_vol * rho; H.p_vol0 = p_vol; H.mu0 = mu; H.la0 = la
    part.p_vol = p_vol
    part.register_buffer("mass", torch.where(active_p, torch.full((Np,), p_vol * rho, device=device),
                                             torch.zeros(Np, device=device)))
    H.add_level(part)

    for fname, f in sc.fields.items():
        dt_f = min(0.2 / max(f["diffusion"], 1e-6), 0.2)
        H.add_field(GridField(fname, f.get("couples_to", "cell"), res=int(f["res"]),
                              diffusion=f["diffusion"], decay=f.get("decay", 0.0), dt=dt_f,
                              device=device, periodic=H.periodic, width=H.world_width))
    H.cell_accel = torch.zeros(Nmax, 2, device=device)
    return H


def _aggregate(H):
    """Active-cell centroid = mass-weighted mean of its particles (circular on the torus)."""
    cell, part = H.level("cell"), H.level("particle")
    Nc = cell.n; dev = part.state.device
    m = torch.zeros(Nc, 1, device=dev).index_add_(0, part.parent, part.mass[:, None]).clamp(min=EPS)
    if H.periodic:
        ang = part.state[:, :2] * (2 * math.pi)
        s = torch.zeros(Nc, 2, device=dev).index_add_(0, part.parent, part.mass[:, None] * torch.sin(ang))
        c = torch.zeros(Nc, 2, device=dev).index_add_(0, part.parent, part.mass[:, None] * torch.cos(ang))
        pos = torch.remainder(torch.atan2(s, c) / (2 * math.pi), 1.0)
    else:
        pos = torch.zeros(Nc, 2, device=dev).index_add_(0, part.parent, part.mass[:, None] * part.state[:, :2]) / m
    cell.state[:, :2] = torch.where(H.c_active[:, None], pos, cell.state[:, :2])


def run(sc, device="cpu"):
    H = build(sc, device)
    inst = [(o.op, get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), o.on)
            for o in sc.operators]
    cell, part = H.level("cell"), H.level("particle")
    re = sc.record_every; hist = []
    with torch.no_grad():
        for frame in range(sc.n_frames + 1):
            H.cell_accel = torch.zeros(cell.n, 2, device=device)
            for step in sc.schedule:
                for tok in (step if isinstance(step, list) else [step]):
                    if tok == "aggregate":
                        _aggregate(H)
                    elif tok.endswith(".diffuse"):
                        H.fields[tok[:-len(".diffuse")]].step()
                    else:
                        for nm, ob, sel in inst:
                            if nm != tok:
                                continue
                            mask = H.c_active if sel.set == "cell" else None
                            for lvl, d in ob(H, mask).items():
                                if lvl == "cell":
                                    H.cell_accel = H.cell_accel + d
            if frame % re == 0:
                a = H.c_active.cpu().numpy()
                fld = next(iter(H.fields.values())).grid.cpu().numpy()
                cpos = cell.state[:, :2].cpu().numpy()
                pa = (part.mass > 0).cpu().numpy()                 # active MPM particles (the soft bodies)
                hist.append(dict(pos=cpos[a], count=int(H.c_active.sum()), field=fld,
                                 pos_full=cpos.copy(), active=a.copy(),
                                 ppos=part.state[:, :2].cpu().numpy()[pa]))
    return H, hist
