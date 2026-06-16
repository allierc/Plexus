"""grow_engine.py -- a generic engine that supports CARDINALITY CHANGE.

Same contract as engine2 (load a validated spec, build a `base.Hierarchy` of
`base.Level`s, resolve operator names from the registry, run the `Schedule`,
integrate per-level deltas), but generalised so sets can grow and split:

  * each level is a FIXED BUFFER of `buffer` slots with an occupancy mass
    (`H.p_w` for particles, `H.c_active` for cells); `per_parent`/`n` are the
    initially-active counts;
  * the `integrate` builtin (already in the schedule grammar, unused by engine2)
    moves a level overdamped by its accumulated acceleration;
  * structural operators (`duplicate`, `divide`) flip slots dormant<->active.

`engine2` and this engine should eventually merge into one. Kept separate here
because engine2 is wired to fixed-size MPM.

  PYTHONPATH=../src python grow_engine.py            # run the spec + render
"""
from __future__ import annotations

import os, math
import numpy as np
import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
from scenario_schema import load
import ops_grow as _ops_grow          # noqa: F401  registers cohere/repulse/duplicate/divide

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
EPS = 1e-6
FMAX = 1.2                            # per-particle accel clamp (stability)


def build(sc, device=DEV):
    H = Hierarchy(); H.config = sc
    H.rng = torch.Generator(device=device).manual_seed(sc.seed)

    cs = sc.sets["cell"]; ps = sc.sets["particle"]
    Nc0 = int(cs["n"]);  Nc_max = int(cs.get("buffer", Nc0))
    ppc = int(ps["per_parent"]); Np_max = int(ps.get("buffer", Nc0 * ppc))
    rad = float(ps["radius"])

    # cell level: fixed buffer of Nc_max slots, Nc0 active
    cell = Level("cell", level=1, state=torch.zeros(Nc_max, 4, device=device))
    H.add_level(cell)
    H.c_active = torch.zeros(Nc_max, dtype=torch.bool, device=device); H.c_active[:Nc0] = True

    # particle level: fixed buffer of Np_max slots, Nc0*ppc active in a disc per cell
    part = Level("particle", level=0, state=torch.zeros(Np_max, 4, device=device))
    parent = torch.zeros(Np_max, dtype=torch.long, device=device)
    w = torch.zeros(Np_max, device=device)
    n0 = Nc0 * ppc
    parent[:n0] = torch.arange(Nc0, device=device).repeat_interleave(ppc)
    r = torch.sqrt(torch.rand(n0, generator=H.rng, device=device)) * rad
    th = torch.rand(n0, generator=H.rng, device=device) * 2 * math.pi
    centre = torch.tensor([0.5, 0.5], device=device)
    part.state[:n0, :2] = centre + torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    w[:n0] = 1.0
    part.register_buffer("parent", parent)
    H.add_level(part); H.p_w = w

    # optional particle roles by radius: innermost -> nucleus, outermost -> membrane.
    H.nuc_id = H.mem_id = None
    ptypes = ps.get("types")
    node_type = torch.zeros(Np_max, dtype=torch.long, device=device)
    if ptypes:
        names = list(ptypes.keys()); part.type_names = names
        rr = (part.state[:n0, :2] - centre).norm(dim=1); order = rr.argsort()
        if "nucleus" in names:
            H.nuc_id = names.index("nucleus")
            kn = int(float(ptypes["nucleus"]["fraction"]) * n0)
            node_type[order[:kn]] = H.nuc_id                   # innermost
        if "membrane" in names:
            H.mem_id = names.index("membrane")
            km = int(float(ptypes["membrane"]["fraction"]) * n0)
            node_type[order[n0 - km:]] = H.mem_id              # outermost
    part.register_buffer("node_type", node_type)

    H.cell_birth = [0.0] * Nc_max; H.cell_birth[0] = float(ppc)
    return H


def _aggregate(H):
    """cell centroid = mass-weighted mean of its (active) particles."""
    part, cell = H.level("particle"), H.level("cell")
    w = H.p_w; pos = part.state[:, :2]; Nc = cell.n; dev = w.device
    m = torch.zeros(Nc, device=dev).index_add(0, part.parent, w)
    sp = torch.zeros(Nc, 2, device=dev).index_add(0, part.parent, w[:, None] * pos)
    cell.state[:, :2] = sp / m.clamp_min(EPS)[:, None]


def _integrate(H, accel, dt):
    """overdamped move: pos += dt * clamp(accel). Particle-level deltas apply
    directly; cell-level deltas (e.g. inter-cell `tissue` adhesion) are broadcast
    down to each cell's particles."""
    part = H.level("particle")
    a = accel.get("particle")
    ac = accel.get("cell")
    if ac is not None:
        b = ac[part.parent]
        a = b if a is None else a + b
    if a is None:
        return
    fn = a.norm(dim=1, keepdim=True).clamp_min(EPS)
    a = a * (fn.clamp(max=FMAX) / fn)
    part.state[:, :2] = part.state[:, :2] + dt * a * H.p_w[:, None]


def run(sc, device=DEV):
    H = build(sc, device)
    inst = [(o.op, get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), o.on)
            for o in sc.operators]
    part, cell = H.level("particle"), H.level("cell")
    re = sc.record_every
    hist = []
    with torch.no_grad():
        for frame in range(sc.n_frames + 1):
            accel = {}
            for step in sc.schedule:
                for tok in (step if isinstance(step, list) else [step]):
                    if tok == "aggregate":
                        _aggregate(H)
                    elif tok == "integrate":
                        _integrate(H, accel, sc.dt); accel = {}
                    else:
                        for nm, ob, sel in inst:
                            if nm != tok:
                                continue
                            for lvl, d in ob(H, None).items():
                                accel[lvl] = accel.get(lvl, 0) + d
            if frame % re == 0:
                a = (H.p_w > EPS).cpu().numpy()
                role = (part.node_type.cpu().numpy()[a]
                        if hasattr(part, "node_type") else None)
                hist.append((part.state[:, :2].cpu().numpy()[a],
                             H.p_w.cpu().numpy()[a],
                             part.parent.cpu().numpy()[a],
                             int(H.c_active.sum()), role))
    return H, hist


if __name__ == "__main__":
    import sys, warnings; warnings.filterwarnings("ignore")
    from divide_cell import render             # reuse the follow-cam + membrane render

    spec = sys.argv[1] if len(sys.argv) > 1 else "divide_cell"
    sc = load(os.path.join(HERE, "scenarios", spec + ".yaml"))
    print(f"[1/2] running spec '{sc.name}' through the framework engine ...", flush=True)
    H, hist = run(sc)
    last = hist[-1]
    print(f"      final: {len(last[0])} particles, {last[3]} cells over {len(hist)} frames", flush=True)
    out = os.path.join(HERE, sc.name + ".gif")
    render(hist, out, title=sc.name, nuc_id=getattr(H, "nuc_id", None),
           mem_id=getattr(H, "mem_id", None))
    print(f"[2/2] wrote {out}", flush=True)
