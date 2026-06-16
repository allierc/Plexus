"""grow_engine_mpm.py -- growth/division engine with REAL MPM mechanics.

Same contract and cardinality-change support as grow_engine.py, but the soft
cell is a disc of MLS-MPM particles: the registered `mpm` Exchange operator does
the elastic mechanics (moving the particles itself), a light `cohere` supplies
per-cell identity (via the per-particle `H.part_accel` hook in mpm), and
`duplicate`/`divide` grow and split the sets on a fixed buffer.

(grow_engine.py and grow_engine_mpm.py share most of build/run and should fold
into one generic engine; kept explicit here while MPM support is young.)

  PYTHONPATH=../src python grow_engine_mpm.py
"""
from __future__ import annotations

import os, math
import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
from scenario_schema import load
import ops_grow as _og          # noqa: F401  registers cohere/duplicate/divide
import mpm as _mpm              # noqa: F401  registers mpm

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
EPS = 1e-6
NU = 0.2


def _lame(E):
    return E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))


def build(sc, device=DEV):
    H = Hierarchy(); H.config = sc
    H.rng = torch.Generator(device=device).manual_seed(sc.seed)
    H.world_width = 1.0; H.periodic = False
    H.walls_mpm = None                                              # no obstacles (mpm fills zeros)

    cs = sc.sets["cell"]; ps = sc.sets["particle"]
    Nc0 = int(cs["n"]); Nc_max = int(cs.get("buffer", Nc0))
    ppc = int(ps["per_parent"]); Np_max = int(ps.get("buffer", Nc0 * ppc))
    rad = float(ps["radius"]); rho = float(ps.get("density", 1.0))
    youngs = float(next(iter(cs["types"].values()))["youngs"])      # single soft type

    cell = Level("cell", level=1, state=torch.zeros(Nc_max, 4, device=device))
    cell.type_names = list(cs["types"].keys())
    cell.register_buffer("node_type", torch.zeros(Nc_max, dtype=torch.long, device=device))
    H.add_level(cell)
    H.c_active = torch.zeros(Nc_max, dtype=torch.bool, device=device); H.c_active[:Nc0] = True

    part = Level("particle", level=0, state=torch.zeros(Np_max, 4, device=device))
    parent = torch.zeros(Np_max, dtype=torch.long, device=device)
    w = torch.zeros(Np_max, device=device)
    n0 = Nc0 * ppc
    parent[:n0] = torch.arange(Nc0, device=device).repeat_interleave(ppc)
    r = torch.sqrt(torch.rand(n0, generator=H.rng, device=device)) * rad
    th = torch.rand(n0, generator=H.rng, device=device) * 2 * math.pi
    part.state[:n0, :2] = torch.tensor([0.5, 0.5], device=device) \
        + torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    w[:n0] = 1.0
    part.register_buffer("parent", parent)
    # MPM per-particle state (full buffer; dormant carry mass 0 so they don't scatter)
    p_vol = math.pi * rad * rad / ppc
    H.p_mass0 = p_vol * rho
    mu, la = _lame(youngs)
    part.register_buffer("C", torch.zeros(Np_max, 2, 2, device=device))
    part.register_buffer("F", torch.eye(2, device=device).expand(Np_max, 2, 2).contiguous())
    part.register_buffer("mu", torch.full((Np_max,), mu, device=device))
    part.register_buffer("la", torch.full((Np_max,), la, device=device))
    part.register_buffer("mass", torch.where(w > EPS, torch.full((Np_max,), p_vol * rho, device=device),
                                             torch.zeros(Np_max, device=device)))
    part.p_vol = p_vol
    H.add_level(part); H.p_w = w
    H.cell_birth = [0.0] * Nc_max; H.cell_birth[0] = float(ppc)
    return H


def _aggregate(H):
    part, cell = H.level("particle"), H.level("cell")
    w = H.p_w; pos = part.state[:, :2]; Nc = cell.n; dev = w.device
    m = torch.zeros(Nc, device=dev).index_add(0, part.parent, w)
    sp = torch.zeros(Nc, 2, device=dev).index_add(0, part.parent, w[:, None] * pos)
    cell.state[:, :2] = sp / m.clamp_min(EPS)[:, None]


def run(sc, device=DEV):
    H = build(sc, device)
    inst = [(o.op, get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), o.on)
            for o in sc.operators]
    part, cell = H.level("particle"), H.level("cell")
    re = sc.record_every; hist = []
    with torch.no_grad():
        for frame in range(sc.n_frames + 1):
            H.cell_accel = torch.zeros(cell.n, 2, device=device)   # no cell-level drivers
            H.part_accel = torch.zeros(part.n, 2, device=device)   # filled by cohere, read by mpm
            for step in sc.schedule:
                for tok in (step if isinstance(step, list) else [step]):
                    if tok == "aggregate":
                        _aggregate(H)
                    else:
                        for nm, ob, sel in inst:
                            if nm != tok:
                                continue
                            for lvl, d in ob(H, None).items():
                                if lvl == "particle":
                                    H.part_accel = H.part_accel + d
                                elif lvl == "cell":
                                    H.cell_accel = H.cell_accel + d
            if frame % re == 0:
                a = (H.p_w > EPS).cpu().numpy()
                hist.append((part.state[:, :2].cpu().numpy()[a],
                             H.p_w.cpu().numpy()[a],
                             part.parent.cpu().numpy()[a],
                             int(H.c_active.sum())))
    return H, hist


if __name__ == "__main__":
    import sys, warnings; warnings.filterwarnings("ignore")
    from divide_cell import render

    spec = sys.argv[1] if len(sys.argv) > 1 else "divide_cell_mpm"
    sc = load(os.path.join(HERE, "scenarios", spec + ".yaml"))
    print(f"[1/2] running '{sc.name}' (MPM mechanics) through the engine ...", flush=True)
    H, hist = run(sc)
    last = hist[-1]
    print(f"      final: {len(last[0])} particles, {last[3]} cells over {len(hist)} frames", flush=True)
    out = os.path.join(HERE, sc.name + ".gif")
    render(hist, out, title=sc.name)
    print(f"[2/2] wrote {out}", flush=True)
