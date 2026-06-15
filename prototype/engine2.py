"""Two-level engine: particle (L0) inside cell (L1), MPM mechanics.

Generic runner for the hierarchical scenario. Builds both levels (cells as discs
of MPM particles, with a parent map), then runs the schedule:
  aggregate (up) -> boids/sense (cell accel) -> secrete/diffuse (field) -> mpm (down).
"""

from __future__ import annotations

import os
import math
import numpy as np
import torch
import zarr

# bit-reproducible GPU runs: deterministic scatter/index_add (else atomics differ)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from tissue_graph.models.base import Hierarchy, Level
from tissue_graph.models.registry import get_operator

import ops as _ops             # noqa: F401  boids/secrete/sense
import mpm as _mpm             # noqa: F401  mpm operator
from grid_field import GridField
from scenario_schema import load

NU = 0.2                       # Poisson ratio (shared)


def _lame(E):
    mu = E / (2 * (1 + NU))
    la = E * NU / ((1 + NU) * (1 - 2 * NU))
    return mu, la


def _geodesic_potential(walls_np, source_np):
    """Navigation potential = distance-to-source through OPEN corridors (BFS),
    mapped so the field is high at the source and decreases with maze distance.
    This is the steady state of source-diffusion-with-walls; gradient ascent on it
    routes around obstacles. Unreachable/wall cells are 0."""
    from collections import deque
    res = walls_np.shape[0]
    INF = 1 << 30
    dist = np.full((res, res), INF, np.int64)
    dq = deque()
    src = np.argwhere(source_np & ~walls_np)
    for i, j in src:
        dist[i, j] = 0; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < res and 0 <= b < res and not walls_np[a, b] and dist[a, b] == INF:
                dist[a, b] = dist[i, j] + 1; dq.append((a, b))
    reach = dist < INF
    pot = np.zeros((res, res), np.float32)
    if reach.any():
        dmax = dist[reach].max()
        pot[reach] = (dmax - dist[reach]).astype(np.float32) / max(dmax, 1)
    pot[walls_np] = 0.0
    return pot


def _raster(rects, res, device):
    """Rasterize a list of [x0,y0,x1,y1] rectangles to a boolean [res,res] grid.
    grid[i,j] center is at ((i+0.5)/res, (j+0.5)/res) -- matches GridField/MPM indexing."""
    c = (torch.arange(res, device=device).float() + 0.5) / res
    X = c[:, None].expand(res, res)
    Y = c[None, :].expand(res, res)
    m = torch.zeros(res, res, dtype=torch.bool, device=device)
    for r in rects:
        m |= (X >= r[0]) & (X <= r[2]) & (Y >= r[1]) & (Y <= r[3])
    return m


def build(sc, device="cpu"):
    g = torch.Generator(device=device).manual_seed(sc.seed)
    H = Hierarchy()
    H.config = sc

    # --- cell level (L1) ---
    cs = sc.sets["cell"]
    Nc = int(cs["n"])
    types = cs["types"]
    cell = Level("cell", level=1, state=torch.zeros(Nc, 4, device=device))
    cell.type_names = list(types.keys())
    node_type = torch.zeros(Nc, dtype=torch.long, device=device)
    youngs = torch.zeros(Nc, device=device)
    perm = torch.randperm(Nc, generator=g, device=device)
    start = 0
    for tid, (tname, t) in enumerate(types.items()):
        k = int(round(t["fraction"] * Nc))
        idx = perm[start:start + k]; start += k
        node_type[idx] = tid
        youngs[idx] = float(t["youngs"])
    cell.register_buffer("node_type", node_type)
    cell.loaded = torch.zeros(Nc, dtype=torch.bool, device=device)   # forage state (unused otherwise)
    H.add_level(cell)

    ps = sc.sets["particle"]
    ppc = int(ps["per_parent"]); rad = float(ps["radius"]); rho = float(ps.get("density", 1.0))
    start_rect = cs.get("start")
    if start_rect is not None:
        # scatter cell centers uniformly in a start region (e.g. home)
        x0, y0, x1, y1 = start_rect
        centers = torch.rand(Nc, 2, generator=g, device=device)
        centers[:, 0] = x0 + centers[:, 0] * (x1 - x0)
        centers[:, 1] = y0 + centers[:, 1] * (y1 - y0)
    else:
        # disc centers on a jittered lattice (deterministic, non-overlapping)
        side = math.ceil(math.sqrt(Nc)); spacing = 1.0 / side
        rows = (torch.arange(Nc, device=device) // side).float()
        cols = (torch.arange(Nc, device=device) % side).float()
        jit = (torch.rand(Nc, 2, generator=g, device=device) - 0.5) * spacing * 0.2
        centers = torch.stack([(cols + 0.5) * spacing, (rows + 0.5) * spacing], 1) + jit

    # --- particle level (L0): uniform disc around each cell center ---
    Np = Nc * ppc
    parent = torch.arange(Nc, device=device).repeat_interleave(ppc)
    r = torch.sqrt(torch.rand(Np, generator=g, device=device)) * rad
    th = torch.rand(Np, generator=g, device=device) * 2 * math.pi
    ppos = centers[parent] + torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    part = Level("particle", level=0,
                 state=torch.cat([ppos, torch.zeros(Np, 2, device=device)], 1))
    part.register_buffer("parent", parent)
    part.register_buffer("C", torch.zeros(Np, 2, 2, device=device))
    part.register_buffer("F", torch.eye(2, device=device).expand(Np, 2, 2).contiguous())
    mu, la = _lame(youngs[parent])
    part.register_buffer("mu", mu)
    part.register_buffer("la", la)
    # per-particle volume = actual disc area / particles  (consistent with packing)
    p_vol = math.pi * rad * rad / ppc
    part.p_vol = p_vol
    part.register_buffer("mass", torch.full((Np,), p_vol * rho, device=device))
    H.add_level(part)

    # --- obstacles (maze): one mask, reused as BC by both fields and MPM ---
    obstacles = getattr(sc, "obstacles", [])
    n_grid = next((int(o.params.get("n_grid", 128)) for o in sc.operators if o.op == "mpm"), 128)
    H.walls_mpm = (_raster(obstacles, n_grid, device).view(-1) if obstacles
                   else torch.zeros(n_grid * n_grid, dtype=torch.bool, device=device))

    H.periodic = (getattr(sc, "boundary", "wall") == "periodic")
    for fname, f in sc.fields.items():
        dt_f = min(0.2 / max(f["diffusion"], 1e-6), 0.2)
        res = int(f["res"])
        walls = _raster(obstacles, res, device) if obstacles else None
        src = _raster([f["source"]], res, device) if f.get("source") else None
        fld = GridField(fname, f["couples_to"], res=res, diffusion=f["diffusion"],
                        decay=f.get("decay", 0.0), dt=dt_f, device=device,
                        walls=walls, source=src, source_rate=f.get("source_rate", 0.0),
                        periodic=H.periodic)
        if f.get("source"):  # static navigation potential = geodesic distance to source
            wnp = (walls.cpu().numpy() if walls is not None else np.zeros((res, res), bool))
            pot = _geodesic_potential(wnp, src.cpu().numpy())
            fld.grid = torch.from_numpy(pot).to(device)
        H.add_field(fld)
    H.cell_accel = torch.zeros(Nc, 2, device=device)
    H.rng = torch.Generator(device=device).manual_seed(sc.seed + 12345)   # for motility
    return H


def _mask(H, sel):
    """Resolve a selector to a boolean mask -- recomputed every frame so that
    state-dependent selectors (e.g. cell[loaded=1]) track the live state."""
    lvl = H.level(sel.set)
    if sel.attr is None:
        return torch.ones(lvl.n, dtype=torch.bool, device=lvl.state.device)
    if sel.attr == "type":
        return lvl.node_type == lvl.type_names.index(sel.val)
    if sel.attr == "loaded":
        return lvl.loaded == bool(int(sel.val))
    raise ValueError(f"unknown selector attribute {sel.attr!r}")


def _aggregate_up(H):
    """cell position/velocity = mean over its particles (Aggregate up).
    On the torus, position uses a circular mean so a cell straddling the wrap is
    not placed at the wrong midpoint."""
    cell, part = H.level("cell"), H.level("particle")
    Nc = cell.n; dev = part.state.device
    cnt = torch.zeros(Nc, 1, device=dev)
    cnt.index_add_(0, part.parent, torch.ones(part.n, 1, device=dev))
    cnt = cnt.clamp(min=1)
    vel = torch.zeros(Nc, 2, device=dev); vel.index_add_(0, part.parent, part.state[:, 2:4])
    vel = vel / cnt
    if getattr(H, "periodic", False):
        ang = part.state[:, :2] * (2 * math.pi)
        sc_ = torch.zeros(Nc, 2, device=dev); cc = torch.zeros(Nc, 2, device=dev)
        sc_.index_add_(0, part.parent, torch.sin(ang)); cc.index_add_(0, part.parent, torch.cos(ang))
        pos = torch.remainder(torch.atan2(sc_, cc) / (2 * math.pi), 1.0)
    else:
        pos = torch.zeros(Nc, 2, device=dev); pos.index_add_(0, part.parent, part.state[:, :2]); pos = pos / cnt
    cell.state = torch.cat([pos, vel], dim=1)


def run(sc, out_path=None, device="cpu", compile_mpm=False):
    H = build(sc, device)
    # instances as a list (op_name, object, selector) -> multiple ops of the same
    # type are allowed; masks are resolved per-frame (dynamic selectors).
    inst = [(o.op, get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), o.on)
            for o in sc.operators]
    if compile_mpm:
        for nm, ob, _ in inst:
            if nm == "mpm":
                ob.compiled = torch.compile(_mpm.mlsmpm_substep, dynamic=False)

    cell, part = H.level("cell"), H.level("particle")
    fld_name = next(iter(sc.fields))
    re = sc.record_every
    n_rec = sc.n_frames // re + 1
    cen = np.zeros((n_rec, cell.n, 2), np.float32)
    ppos = np.zeros((n_rec, part.n, 2), np.float32)
    loaded = np.zeros((n_rec, cell.n), bool)
    fhist = np.zeros((n_rec, H.fields[fld_name].res, H.fields[fld_name].res), np.float32)
    delivered_t = np.zeros(n_rec, np.int32)

    rec = 0
    for frame in range(sc.n_frames + 1):
        H.cell_accel = torch.zeros(cell.n, 2, device=device)
        for step in sc.schedule:
            for tok in (step if isinstance(step, list) else [step]):
                if tok == "aggregate":
                    _aggregate_up(H)
                elif tok.endswith(".diffuse"):
                    H.fields[tok[:-len(".diffuse")]].step()
                else:                                  # run every operator of this name
                    for nm, ob, sel in inst:
                        if nm != tok:
                            continue
                        for _, d in ob(H, _mask(H, sel)).items():
                            H.cell_accel = H.cell_accel + d
        if frame % re == 0:
            cen[rec] = cell.state[:, :2].cpu().numpy()
            ppos[rec] = part.state[:, :2].cpu().numpy()
            loaded[rec] = cell.loaded.cpu().numpy()
            fhist[rec] = H.fields[fld_name].grid.cpu().numpy()
            delivered_t[rec] = getattr(H, "food_delivered", 0)
            rec += 1

    out = dict(cell_pos=cen[:rec], cell_type=cell.node_type.cpu().numpy(),
               particle_pos=ppos[:rec], parent=part.parent.cpu().numpy(),
               loaded=loaded[:rec], field=fhist[:rec], type_names=cell.type_names,
               food_delivered=int(getattr(H, "food_delivered", 0)),
               delivered_t=delivered_t[:rec],
               walls=(_raster(getattr(sc, "obstacles", []), H.fields[fld_name].res, device).cpu().numpy()
                      if getattr(sc, "obstacles", []) else None))
    if out_path is not None:
        root = zarr.open_group(out_path, mode="w")
        for k in ("cell_pos", "cell_type", "particle_pos", "parent", "loaded", "delivered_t"):
            root.create_dataset(k, data=out[k])
        root.create_dataset(f"field_{fld_name}", data=out["field"])
        if out["walls"] is not None:
            root.create_dataset("walls", data=out["walls"])
        root.attrs.update(dict(name=sc.name, seed=sc.seed, type_names=cell.type_names,
                               record_every=re, food_delivered=out["food_delivered"]))
    return out_path, out
