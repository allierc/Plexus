"""dicty_engine.py -- a flat, single-level engine for Dictyostelium aggregation.

A dicty amoeba is a POINT chemotactic crawler, so (per the user's steer) this drops
MPM entirely: one `cell` Level of point nodes over a fixed buffer of `N_max` slots,
an occupancy mass `H.c_w` + active mask `H.c_active` (the framework's dormant-slot
scheme), and ONE diffusing/decaying chemical field (cAMP). Cells

    secrete  cAMP into the field         (registered ops/chemotaxis.py)
    sense    its gradient -> chemotaxis  (registered ops/chemotaxis.py)
    interact short-range repulsion + cohesion   (dicty_ops.py)
    random_walk  undirected motility           (registered ops/random_walk.py)
    divide   proliferate on the dormant buffer  (dicty_ops.py)

Self-generated gradient (secrete + diffuse + decay) drives the scattered cells into
streams and compact mounds -- the morphology of the real movie -- while `interact`
gives the mound a finite size instead of a Keller-Segel point collapse.

Integration is overdamped first-order: pos += dt * sum(cell velocities), clamped.

  PYTHONPATH=../../src python -c "from dicty_engine import run; ..."
"""
from __future__ import annotations

import os
import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import ops as _ops              # noqa: F401  registers secrete/sense/random_walk/motility
import dicty_ops as _dops       # noqa: F401  registers interact/divide
from grid_field import GridField

EPS = 1e-6
FMAX = 0.06                     # per-cell speed cap (sub-cell displacement/frame); override via spec `vmax`


def build(sc, device="cpu"):
    g = torch.Generator(device=device).manual_seed(sc.seed)
    H = Hierarchy(); H.config = sc
    H.rng = torch.Generator(device=device).manual_seed(sc.seed + 12345)
    H.world_width = float(getattr(sc, "world", 1.0))
    H.periodic = (getattr(sc, "boundary", "wall") == "periodic")

    cs = sc.sets["cell"]
    N0 = int(cs["n"]); Nmax = int(cs.get("buffer", N0))
    cell = Level("cell", level=0, state=torch.zeros(Nmax, 4, device=device))
    cell.type_names = list(cs.get("types", {"dicty": {}}).keys())
    cell.register_buffer("node_type", torch.zeros(Nmax, dtype=torch.long, device=device))
    H.add_level(cell)
    H.c_active = torch.zeros(Nmax, dtype=torch.bool, device=device); H.c_active[:N0] = True
    H.c_w = torch.zeros(Nmax, device=device); H.c_w[:N0] = 1.0

    # initial condition: seed from the REAL movie frame-0 cell positions when given
    # (cell.init_npz -> 'init_pos' [M,2] in [0,1]^2), else a uniform scatter. If N0
    # differs from the M detected cells we subsample (N0<M) or tile+jitter (N0>M), so
    # the initial spatial distribution always matches the data's.
    init_npz = cs.get("init_npz")
    start = cs.get("start")
    if init_npz is not None:
        path = init_npz if os.path.isabs(init_npz) else os.path.join(os.path.dirname(__file__), init_npz)
        ip = torch.tensor(np.load(path)["init_pos"], dtype=torch.float32, device=device)
        M = ip.shape[0]
        if N0 <= M:
            sel = torch.randperm(M, generator=g, device=device)[:N0]
            pos = ip[sel]
        else:
            reps = torch.randint(M, (N0,), generator=g, device=device)
            jit = (torch.rand(N0, 2, generator=g, device=device) - 0.5) * 0.01
            pos = (ip[reps] + jit).clamp(0, 1)
        pos[:, 0] *= H.world_width
    else:
        pos = torch.rand(N0, 2, generator=g, device=device)
        if start is not None:                          # optional [x0,y0,x1,y1] seeding box
            x0, y0, x1, y1 = start
            pos[:, 0] = x0 + pos[:, 0] * (x1 - x0)
            pos[:, 1] = y0 + pos[:, 1] * (y1 - y0)
        else:
            pos[:, 0] *= H.world_width
    cell.state[:N0, :2] = pos

    # fields (cAMP): diffuse + decay, dynamic self-generated gradient
    for fname, f in sc.fields.items():
        dt_f = min(0.2 / max(f["diffusion"], 1e-6), 0.2)
        fld = GridField(fname, f.get("couples_to", "cell"), res=int(f["res"]),
                        diffusion=f["diffusion"], decay=f.get("decay", 0.0), dt=dt_f,
                        device=device, periodic=H.periodic, width=H.world_width)
        H.add_field(fld)

    H.cell_accel = torch.zeros(Nmax, 2, device=device)
    return H


def _mask(H, sel):
    """Active cells only (dormant slots never secrete / sense / move). One cell type
    here, so the type sub-selector is a no-op; `active` is implicit."""
    return H.c_active


def run(sc, device="cpu"):
    H = build(sc, device)
    inst = [(o.op, get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), o.on)
            for o in sc.operators]
    cell = H.level("cell")
    dt = float(getattr(sc, "dt", 0.05))
    fmax = float(getattr(sc, "vmax", FMAX))
    re = sc.record_every
    hist = []
    accel = torch.zeros(cell.n, 2, device=device)

    def integrate():
        nonlocal accel
        fn = accel.norm(dim=1, keepdim=True).clamp_min(EPS)
        a = accel * (fn.clamp(max=fmax) / fn)
        new = cell.state[:, :2] + dt * a * H.c_active.float()[:, None]
        if H.periodic:
            new = torch.stack([torch.remainder(new[:, 0], H.world_width),
                               torch.remainder(new[:, 1], 1.0)], 1)
        else:
            new = torch.stack([new[:, 0].clamp(0, H.world_width - 1e-6),
                               new[:, 1].clamp(0, 1 - 1e-6)], 1)
        cell.state[:, :2] = new
        accel = torch.zeros(cell.n, 2, device=device)

    with torch.no_grad():
        for frame in range(sc.n_frames + 1):
            for step in sc.schedule:
                for tok in (step if isinstance(step, list) else [step]):
                    if tok == "integrate":
                        integrate()
                    elif tok.endswith(".diffuse"):
                        H.fields[tok[:-len(".diffuse")]].step()
                    else:
                        for nm, ob, sel in inst:
                            if nm != tok:
                                continue
                            for lvl, d in ob(H, _mask(H, sel)).items():
                                if lvl == "cell":
                                    accel = accel + d
            if frame % re == 0:
                a = H.c_active.cpu().numpy()
                fld = next(iter(H.fields.values())).grid.cpu().numpy()
                hist.append(dict(pos=cell.state[:, :2].cpu().numpy()[a],
                                 count=int(H.c_active.sum()), field=fld,
                                 pos_full=cell.state[:, :2].cpu().numpy().copy(),  # slot-consistent (for flow)
                                 active=a.copy()))
    return H, hist
