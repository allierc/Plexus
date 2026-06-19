"""Slime engine: a single agent set (S_0 = cell) coupled to one multi-channel
trail field. No containment, no MPM -- the slime model is a flat agent system, so
this is the lightest possible Plexus engine.

Generic over the spec: it builds the `cell` Level (broadcasting each `type`'s
scalar properties to per-agent buffers, exactly as Unity hands each agent its
`SpeciesSettings`), builds the `TrailField` with one channel per type, then
iterates the schedule (`slime`, `trail.diffuse`). It contains NO slime-specific
branching and no scenario names -- every behaviour lives behind the registry
(slime_ops.py) and is selected by the spec, per the framework contract.

Validation is delegated to the shared `scenario_schema.load` (the contract
gatekeeper): import slime_ops first so the `slime` operator and `trail` field are
registered before a spec is checked.

    PYTHONPATH=../../src python -c "import slime_ops, slime_engine; ..."
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

import slime_ops  # noqa: F401  registers TrailField + slime

# default per-species display colours (matches the spirit of Lague's RGBA channels)
DEFAULT_COLORS = [
    (0.20, 0.95, 0.65),   # green-cyan
    (1.00, 0.35, 0.45),   # red
    (0.40, 0.65, 1.00),   # blue
    (1.00, 0.85, 0.30),   # yellow
]


def _spawn(mode, N, W, radius, g, device):
    """Initial positions + headings. Modes mirror Lague's SpawnMode enum."""
    cx, cy = W / 2.0, 0.5
    if mode == "random":
        pos = torch.rand(N, 2, generator=g, device=device)
        pos[:, 0] *= W
        head = torch.rand(N, generator=g, device=device) * 2 * math.pi
    elif mode in ("point", "center"):
        pos = torch.stack([torch.full((N,), cx, device=device),
                           torch.full((N,), cy, device=device)], 1)
        pos = pos + (torch.rand(N, 2, generator=g, device=device) - 0.5) * 1e-3
        head = torch.rand(N, generator=g, device=device) * 2 * math.pi
    elif mode == "disc":
        r = torch.sqrt(torch.rand(N, generator=g, device=device)) * radius
        a = torch.rand(N, generator=g, device=device) * 2 * math.pi
        pos = torch.stack([cx + r * torch.cos(a), cy + r * torch.sin(a)], 1)
        head = torch.rand(N, generator=g, device=device) * 2 * math.pi
    elif mode in ("ring_in", "ring_out"):
        a = torch.rand(N, generator=g, device=device) * 2 * math.pi
        pos = torch.stack([cx + radius * torch.cos(a), cy + radius * torch.sin(a)], 1)
        inward = a + math.pi                              # point back toward centre
        head = inward if mode == "ring_in" else a
    else:
        raise ValueError(f"unknown spawn mode {mode!r}")
    pos[:, 0] = pos[:, 0].clamp(0, W - 1e-6)
    pos[:, 1] = pos[:, 1].clamp(0, 1 - 1e-6)
    return pos, head


def build(sc, device="cpu"):
    g = torch.Generator(device=device).manual_seed(sc.seed)
    H = Hierarchy()
    H.config = sc

    cs = sc.sets["cell"]
    Nc = int(cs["n"])
    types = cs["types"]
    cell = Level("cell", level=0, state=torch.zeros(Nc, 4, device=device))
    cell.type_names = list(types.keys())

    # which scalar per-type props to broadcast to per-agent buffers (union over types)
    propnames = set()
    for t in types.values():
        for k, v in t.items():
            if k in ("fraction", "color"):
                continue
            if isinstance(v, (int, float)):
                propnames.add(k)
    bufs = {p: torch.zeros(Nc, device=device) for p in propnames}
    colors = torch.zeros(Nc, 3, device=device)
    node_type = torch.zeros(Nc, dtype=torch.long, device=device)

    # colours come from the `plotting:` block (Style category), NOT from the set
    pcolors = dict(getattr(sc, "plotting", {}).get("colors", {}))
    def color_of(tname, tid):
        return pcolors.get(tname, DEFAULT_COLORS[tid % len(DEFAULT_COLORS)])

    perm = torch.randperm(Nc, generator=g, device=device)
    tlist = list(types.items())
    start = 0
    for tid, (tname, t) in enumerate(tlist):
        k = int(round(t["fraction"] * Nc))
        idx = perm[start:start + k]
        start += k
        node_type[idx] = tid
        for p in propnames:
            bufs[p][idx] = float(t.get(p, 0.0))
        col = color_of(tname, tid)
        colors[idx] = torch.tensor([float(c) for c in col[:3]], device=device)
    if start < Nc:                                        # rounding slop -> last type
        idx = perm[start:]
        tid = len(tlist) - 1
        node_type[idx] = tid
        for p in propnames:
            bufs[p][idx] = float(tlist[tid][1].get(p, 0.0))
        col = color_of(tlist[tid][0], tid)
        colors[idx] = torch.tensor([float(c) for c in col[:3]], device=device)

    cell.register_buffer("node_type", node_type)
    for p, b in bufs.items():
        cell.register_buffer(p, b)
    cell.register_buffer("color_rgb", colors)

    W = float(getattr(sc, "world", 1.0))
    H.world_width = W
    pos, head = _spawn(cs.get("spawn", "random"), Nc, W, float(cs.get("spawn_radius", 0.3)), g, device)
    cell.state[:, 0] = pos[:, 0]
    cell.state[:, 1] = pos[:, 1]
    cell.heading = head
    H.add_level(cell)

    H.periodic = (getattr(sc, "boundary", "wall") == "periodic")

    fname, f = next(iter(sc.fields.items()))
    fld = slime_ops.TrailField(
        fname, f["couples_to"], n_channels=len(types), res=int(f["res"]),
        deposit=f.get("deposit", 5.0), decay=f.get("decay", 0.2),
        diffuse=f.get("diffuse", 3.0), dt=float(getattr(sc, "dt", 1.0)),
        device=device, width=W)
    H.add_field(fld)

    H.rng = torch.Generator(device=device).manual_seed(sc.seed + 12345)
    return H


def _mask(H, sel):
    """Resolve a selector to a per-frame boolean mask (categorical separation)."""
    lvl = H.level(sel.set)
    if sel.attr is None:
        return torch.ones(lvl.n, dtype=torch.bool, device=lvl.state.device)
    if sel.attr == "type":
        return lvl.node_type == lvl.type_names.index(sel.val)
    raise ValueError(f"unknown selector attribute {sel.attr!r}")


def run(sc, device="cpu"):
    H = build(sc, device)
    inst = [(o.op, get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), o.on)
            for o in sc.operators]
    cell = H.level("cell")
    fname = next(iter(sc.fields))
    fld = H.fields[fname]
    dt = float(getattr(sc, "dt", 1.0))

    re = sc.record_every
    n_rec = sc.n_frames // re + 1
    fhist = np.zeros((n_rec, fld.C, fld.nx, fld.ny), np.float32)
    cpos = np.zeros((n_rec, cell.n, 2), np.float32)
    chead = np.zeros((n_rec, cell.n), np.float32)

    graph_hist = []          # per recorded frame: (nodes [M,2], edges [2,E]) if trail_graph runs
    graph_chan_hist = []     # per recorded frame: [(nodes, edges, channel), ...] if per_species
    rec = 0
    for frame in range(sc.n_frames + 1):
        for step in sc.schedule:
            for tok in (step if isinstance(step, list) else [step]):
                if tok.endswith(".diffuse"):
                    H.fields[tok[:-len(".diffuse")]].step()
                else:
                    for nm, ob, sel in inst:
                        if nm != tok:
                            continue
                        d = ob(H, _mask(H, sel))
                        for _, delta in d.items():        # generic first-order integrate (slime self-moves -> none)
                            cell.state[:, :2] = cell.state[:, :2] + dt * delta
        if frame % re == 0:
            fhist[rec] = fld.grid.cpu().numpy()
            cpos[rec] = cell.state[:, :2].cpu().numpy()
            if hasattr(cell, "heading"):
                chead[rec] = cell.heading.cpu().numpy()
            if hasattr(H, "graph_nodes"):                 # trail_graph (Rewire) ran -> record the scaffold
                graph_hist.append((H.graph_nodes.cpu().numpy(), H.graph_edges.cpu().numpy()))
                gc = getattr(H, "graph_channels", None)
                graph_chan_hist.append([(n.cpu().numpy(), e.cpu().numpy(), c) for (n, e, c) in gc]
                                       if gc else None)
            rec += 1

    # per-channel display colour = colour of the first agent of that type
    nt = cell.node_type.cpu().numpy()
    cols = cell.color_rgb.cpu().numpy()
    chan_color = np.zeros((fld.C, 3), np.float32)
    for c in range(fld.C):
        m = nt == c
        chan_color[c] = cols[m][0] if m.any() else slime_engine_default(c)

    return dict(field=fhist[:rec], cell_pos=cpos[:rec], cell_head=chead[:rec], node_type=nt,
                chan_color=chan_color, type_names=cell.type_names,
                C=fld.C, W=H.world_width, n_agents=cell.n,
                graph=graph_hist or None, graph_channels=graph_chan_hist or None)


def slime_engine_default(c):
    return np.array(DEFAULT_COLORS[c % len(DEFAULT_COLORS)], np.float32)
