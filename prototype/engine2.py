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
    H.add_level(cell)

    # disc centers on a jittered lattice (deterministic, non-overlapping)
    ps = sc.sets["particle"]
    ppc = int(ps["per_parent"]); rad = float(ps["radius"]); rho = float(ps.get("density", 1.0))
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

    for fname, f in sc.fields.items():
        # field dt chosen for diffusion stability (dt*D <= 0.25)
        dt_f = min(0.2 / max(f["diffusion"], 1e-6), 0.2)
        H.add_field(GridField(fname, f["couples_to"], res=f["res"], diffusion=f["diffusion"],
                              decay=f.get("decay", 0.0), dt=dt_f, device=device))
    H.cell_accel = torch.zeros(Nc, 2, device=device)
    return H


def _mask(H, sel):
    lvl = H.level(sel.set)
    n = lvl.n
    if sel.attr is None:
        return torch.ones(n, dtype=torch.bool, device=lvl.state.device)
    tid = lvl.type_names.index(sel.val)
    return lvl.node_type == tid


def _aggregate_up(H):
    """cell position/velocity = mean over its particles (Aggregate up)."""
    cell, part = H.level("cell"), H.level("particle")
    Nc = cell.n
    sums = torch.zeros(Nc, 4, device=part.state.device)
    sums.index_add_(0, part.parent, part.state)
    cnt = torch.zeros(Nc, 1, device=part.state.device)
    cnt.index_add_(0, part.parent, torch.ones(part.n, 1, device=part.state.device))
    cell.state = sums / cnt.clamp(min=1)


def run(sc, out_path, device="cpu", compile_mpm=False):
    H = build(sc, device)
    inst = {o.op: (get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), _mask(H, o.on))
            for o in sc.operators}
    if compile_mpm:
        inst["mpm"][0].compiled = torch.compile(_mpm.mlsmpm_substep, dynamic=False)

    cell, part = H.level("cell"), H.level("particle")
    fld_name = next(iter(sc.fields))
    re = sc.record_every
    n_rec = sc.n_frames // re + 1
    cen = np.zeros((n_rec, cell.n, 2), np.float32)
    ppos = np.zeros((n_rec, part.n, 2), np.float32)
    fhist = np.zeros((n_rec, H.fields[fld_name].res, H.fields[fld_name].res), np.float32)

    rec = 0
    for frame in range(sc.n_frames + 1):
        H.cell_accel = torch.zeros(cell.n, 2, device=device)
        for step in sc.schedule:
            for tok in (step if isinstance(step, list) else [step]):
                if tok == "aggregate":
                    _aggregate_up(H)
                elif tok.endswith(".diffuse"):
                    H.fields[tok[:-len(".diffuse")]].step()
                elif tok == "mpm":
                    inst["mpm"][0](H, None)
                else:
                    op, m = inst[tok]
                    for _, d in op(H, m).items():
                        H.cell_accel = H.cell_accel + d
        if frame % re == 0:
            cen[rec] = cell.state[:, :2].cpu().numpy()
            ppos[rec] = part.state[:, :2].cpu().numpy()
            fhist[rec] = H.fields[fld_name].grid.cpu().numpy()
            rec += 1

    root = zarr.open_group(out_path, mode="w")
    root.create_dataset("cell_pos", data=cen[:rec])
    root.create_dataset("cell_type", data=cell.node_type.cpu().numpy())
    root.create_dataset("particle_pos", data=ppos[:rec])
    root.create_dataset("particle_parent", data=part.parent.cpu().numpy())
    root.create_dataset(f"field_{fld_name}", data=fhist[:rec])
    root.attrs.update(dict(name=sc.name, seed=sc.seed, type_names=cell.type_names,
                           record_every=re, ppc=int(sc.sets["particle"]["per_parent"])))
    return out_path, dict(cell_pos=cen[:rec], cell_type=cell.node_type.cpu().numpy(),
                          particle_pos=ppos[:rec], parent=part.parent.cpu().numpy(),
                          field=fhist[:rec], type_names=cell.type_names)
