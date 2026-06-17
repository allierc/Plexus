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

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

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
    nx, ny = walls_np.shape
    INF = 1 << 30
    dist = np.full((nx, ny), INF, np.int64)
    dq = deque()
    src = np.argwhere(source_np & ~walls_np)
    for i, j in src:
        dist[i, j] = 0; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and not walls_np[a, b] and dist[a, b] == INF:
                dist[a, b] = dist[i, j] + 1; dq.append((a, b))
    reach = dist < INF
    pot = np.zeros((nx, ny), np.float32)
    if reach.any():
        dmax = dist[reach].max()
        pot[reach] = (dmax - dist[reach]).astype(np.float32) / max(dmax, 1)
    pot[walls_np] = 0.0
    return pot


def _raster(shapes, ny, width, device):
    """Rasterize a list of obstacles to a boolean [nx,ny] grid over the world
    [0,width]x[0,1] with square cells dx=1/ny (nx=round(width*ny)). Each obstacle is
    a rectangle [x0,y0,x1,y1] (len 4) or a circle [cx,cy,radius] (len 3). Cell [i,j]
    center is at ((i+0.5)*dx, (j+0.5)*dx) -- matches GridField/MPM indexing."""
    dx = 1.0 / ny
    nx = int(round(width * ny))
    cx = (torch.arange(nx, device=device).float() + 0.5) * dx
    cy = (torch.arange(ny, device=device).float() + 0.5) * dx
    X = cx[:, None].expand(nx, ny)
    Y = cy[None, :].expand(nx, ny)
    m = torch.zeros(nx, ny, dtype=torch.bool, device=device)
    for r in shapes:
        if len(r) == 3:                                   # circle: (cx, cy, radius)
            m |= (X - r[0]) ** 2 + (Y - r[1]) ** 2 <= r[2] ** 2
        else:                                             # rectangle: (x0, y0, x1, y1)
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
    core_youngs = torch.zeros(Nc, device=device)        # 0 -> homogeneous (no core)
    core_frac = torch.zeros(Nc, device=device)          # core radius as a fraction of the disc
    perm = torch.randperm(Nc, generator=g, device=device)
    start = 0
    type_layers = {}                                    # tid -> [(outer_frac, youngs), ...] inner->outer
    for tid, (tname, t) in enumerate(types.items()):
        k = int(round(t["fraction"] * Nc))
        idx = perm[start:start + k]; start += k
        node_type[idx] = tid
        youngs[idx] = float(t["youngs"])
        core = t.get("core")                            # optional {youngs, frac}: a stiffer inner core
        if core is not None:
            core_youngs[idx] = float(core["youngs"])
            core_frac[idx] = float(core.get("frac", 0.5))
        layers = t.get("layers")                        # optional N-layer ball (inner->outer concentric shells)
        if layers is not None:                          # each layer: (outer_frac, youngs, material)
            type_layers[tid] = [(float(L["frac"]), float(L["youngs"]), L.get("material", "elastic"))
                                for L in layers]
    cell.register_buffer("node_type", node_type)
    cell.loaded = torch.zeros(Nc, dtype=torch.bool, device=device)   # forage state (unused otherwise)
    H.add_level(cell)

    ps = sc.sets["particle"]
    ppc = int(ps["per_parent"]); rad = float(ps["radius"]); rho = float(ps.get("density", 1.0))
    start_rect = cs.get("start")
    if start_rect is not None and len(start_rect) and isinstance(start_rect[0], (list, tuple)):
        # explicit per-cell centers: start: [[x,y], ...] (deterministic placement, tiled if fewer than Nc)
        pts = torch.tensor([[float(p[0]), float(p[1])] for p in start_rect], device=device)
        centers = pts[torch.arange(Nc, device=device) % pts.shape[0]]
    elif start_rect is not None:
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
    # a type may instead FILL a rectangle (a block/pool of material, e.g. a cube of water)
    for tid, (tname, t) in enumerate(types.items()):
        blk = t.get("block")
        if blk is not None:
            bm = node_type[parent] == tid
            nb = int(bm.sum())
            x0, y0, x1, y1 = blk
            u = torch.rand(nb, 2, generator=g, device=device)
            ppos[bm] = torch.stack([x0 + u[:, 0] * (x1 - x0), y0 + u[:, 1] * (y1 - y0)], 1)
    part = Level("particle", level=0,
                 state=torch.cat([ppos, torch.zeros(Np, 2, device=device)], 1))
    part.register_buffer("parent", parent)
    part.register_buffer("C", torch.zeros(Np, 2, 2, device=device))
    part.register_buffer("F", torch.eye(2, device=device).expand(Np, 2, 2).contiguous())
    # per-particle stiffness. Two ways to make a multi-material ball:
    #   core:   one stiffer inner disc (r < frac*rad)         -> is_core flag (legacy, 2 materials)
    #   layers: N concentric shells inner->outer by frac      -> layer_id 0..N-1 (general)
    # default: homogeneous (the cell's shell youngs).
    is_core = (core_youngs[parent] > 0) & (r < core_frac[parent] * rad)
    p_youngs = torch.where(is_core, core_youngs[parent], youngs[parent])
    layer_id = torch.zeros(Np, dtype=torch.long, device=device)        # 0 if not layered
    is_liquid = torch.zeros(Np, dtype=torch.bool, device=device)       # liquid material (mu=0, F reset)
    is_snow = torch.zeros(Np, dtype=torch.bool, device=device)         # snow/plastic (SVD clamp + hardening)
    if type_layers:
        rnorm = r / max(rad, 1e-9)                                     # particle radius in [0,1]
        ntype = node_type[parent]
        for tid, lyrs in type_layers.items():
            sel = ntype == tid
            assigned = torch.zeros_like(sel)
            for li, (frac, yng, mat) in enumerate(lyrs):              # inner -> outer; first band that contains it
                band = sel & (~assigned) & (rnorm <= frac)
                p_youngs = torch.where(band, torch.full_like(p_youngs, yng), p_youngs)
                layer_id = torch.where(band, torch.full_like(layer_id, li), layer_id)
                if mat == "liquid":
                    is_liquid = is_liquid | band
                elif mat == "snow":
                    is_snow = is_snow | band
                assigned = assigned | band
            rem = sel & (~assigned)                                    # rounding slop -> outermost layer
            p_youngs = torch.where(rem, torch.full_like(p_youngs, lyrs[-1][1]), p_youngs)
            layer_id = torch.where(rem, torch.full_like(layer_id, len(lyrs) - 1), layer_id)
            if lyrs[-1][2] == "liquid":
                is_liquid = is_liquid | rem
            elif lyrs[-1][2] == "snow":
                is_snow = is_snow | rem
    mu, la = _lame(p_youngs)
    mu = torch.where(is_liquid, torch.zeros_like(mu), mu)              # liquid: no shear modulus -> pressure only
    part.register_buffer("is_core", is_core)
    part.register_buffer("layer_id", layer_id)
    part.register_buffer("is_liquid", is_liquid)
    part.register_buffer("is_snow", is_snow)
    part.register_buffer("Jp", torch.ones(Np, device=device))         # plastic volume ratio (snow hardening)
    part.register_buffer("mu", mu)
    part.register_buffer("la", la)
    # per-particle volume = cell footprint area / particles. Disc cells: pi*rad^2/ppc;
    # block cells (a pool/cube): block rectangle area / ppc -> correct density for a mix
    # of a big water block and a small dropped ball in one scene.
    p_vol = torch.full((Np,), math.pi * rad * rad / ppc, device=device)
    for tid, (tname, t) in enumerate(types.items()):
        blk = t.get("block")
        if blk is not None:
            x0, y0, x1, y1 = blk
            area = abs((x1 - x0) * (y1 - y0))
            p_vol = torch.where(node_type[parent] == tid,
                                torch.full_like(p_vol, area / ppc), p_vol)
    part.p_vol = p_vol
    part.register_buffer("mass", p_vol * rho)
    H.add_level(part)

    # --- world: rectangle [0,width]x[0,1] (width>1 -> longitudinal); square grid cells ---
    width = float(getattr(sc, "world", 1.0))
    H.world_width = width

    # --- obstacles (maze): one mask, reused as BC by both fields and MPM ---
    obstacles = getattr(sc, "obstacles", [])
    n_grid = next((int(o.params.get("n_grid", 128)) for o in sc.operators if o.op == "mpm"), 128)
    nx_mpm = int(round(width * n_grid))
    H.walls_mpm = (_raster(obstacles, n_grid, width, device).view(-1) if obstacles
                   else torch.zeros(nx_mpm * n_grid, dtype=torch.bool, device=device))

    H.periodic = (getattr(sc, "boundary", "wall") == "periodic")
    for fname, f in sc.fields.items():
        dt_f = min(0.2 / max(f["diffusion"], 1e-6), 0.2)
        res = int(f["res"])
        nx = int(round(width * res))
        walls = _raster(obstacles, res, width, device) if obstacles else None
        src = _raster([f["source"]], res, width, device) if f.get("source") else None
        fld = GridField(fname, f["couples_to"], res=res, diffusion=f["diffusion"],
                        decay=f.get("decay", 0.0), dt=dt_f, device=device,
                        walls=walls, source=src, source_rate=f.get("source_rate", 0.0),
                        periodic=H.periodic, width=width)
        # initial fill (declared): uniform value across the open domain
        if f.get("init") is not None:
            fld.grid = torch.full((nx, res), float(f["init"]), device=device)
            if walls is not None:
                fld.grid = torch.where(walls, torch.zeros_like(fld.grid), fld.grid)
        # navigation mode (declared, generic -- the engine does not branch on scenario name):
        #   geodesic  -> precomputed distance-to-source potential (an EXTERNAL gradient)
        #   dynamic   -> field evolves at run time (self-generated gradient: consume + diffuse)
        # default: geodesic when a source is given (back-compat), else dynamic.
        nav = f.get("navigation", "geodesic" if f.get("source") else "dynamic")
        if nav == "geodesic" and src is not None:
            wnp = (walls.cpu().numpy() if walls is not None else np.zeros((nx, res), bool))
            pot = _geodesic_potential(wnp, src.cpu().numpy())
            fld.grid = torch.from_numpy(pot).to(device)
        H.add_field(fld)
    H.cell_accel = torch.zeros(Nc, 2, device=device)
    H.harvested = 0.0                                                     # graze objective
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
    if sel.attr == "done":
        done = getattr(lvl, "done", None)
        if done is None:
            done = torch.zeros(lvl.n, dtype=torch.bool, device=lvl.state.device)
        return done == bool(int(sel.val))
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
    done = np.zeros((n_rec, cell.n), bool)
    _fg = H.fields[fld_name].grid
    fhist = np.zeros((n_rec, _fg.shape[0], _fg.shape[1]), np.float32)
    # record EVERY field too (generic; multi-field sims like the ant colony need all of them)
    fhist_all = {nm: np.zeros((n_rec, fl.grid.shape[0], fl.grid.shape[1]), np.float32)
                 for nm, fl in H.fields.items()}
    delivered_t = np.zeros(n_rec, np.int32)
    finished_t = np.zeros(n_rec, np.int32)

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
            done[rec] = (cell.done.cpu().numpy() if hasattr(cell, "done")
                         else np.zeros(cell.n, bool))
            fhist[rec] = H.fields[fld_name].grid.cpu().numpy()
            for nm, fl in H.fields.items():
                fhist_all[nm][rec] = fl.grid.cpu().numpy()
            delivered_t[rec] = getattr(H, "food_delivered", 0)
            finished_t[rec] = getattr(H, "finished", 0)
            rec += 1

    out = dict(cell_pos=cen[:rec], cell_type=cell.node_type.cpu().numpy(),
               particle_pos=ppos[:rec], parent=part.parent.cpu().numpy(),
               is_core=part.is_core.cpu().numpy() if hasattr(part, "is_core") else None,
               layer_id=part.layer_id.cpu().numpy() if hasattr(part, "layer_id") else None,
               is_liquid=part.is_liquid.cpu().numpy() if hasattr(part, "is_liquid") else None,
               is_snow=part.is_snow.cpu().numpy() if hasattr(part, "is_snow") else None,
               loaded=loaded[:rec], done=done[:rec], field=fhist[:rec],
               fields={nm: h[:rec] for nm, h in fhist_all.items()}, type_names=cell.type_names,
               food_delivered=int(getattr(H, "food_delivered", 0)),
               harvested=float(getattr(H, "harvested", 0.0)),
               finished=int(getattr(H, "finished", 0)),
               delivered_t=delivered_t[:rec], finished_t=finished_t[:rec],
               walls=(_raster(getattr(sc, "obstacles", []), H.fields[fld_name].res,
                              getattr(H, "world_width", 1.0), device).cpu().numpy()
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
