#!/usr/bin/env python
"""Decompose ONE cell division vertex-by-vertex: pick a ready-to-divide cell on a still-smooth mesh,
print its ring (all vertex positions), centroid, Newell normal, area, and deviation from its neighbours
BEFORE; apply the exact Hertwig long-axis edge-midpoint septum (divide_face_3d, as divide_3d does);
then print the TWO daughters' rings (marking the NEW septum midpoints), normals, areas, and deviation
from the (unchanged) neighbours AFTER. Shows precisely what geometry the cut injects."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
np.set_printoptions(precision=3, suppress=True)
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from tyssue_topology_ops3d import rings_from_flat_3d, divide_face_3d
import torch

RADIUS, JITTER, SEED = 5.0, 0.16, 0
STOP = 78   # cells reach ~2x volume ~frame 77 (synchronised), still pre-buckling


def build():
    verts, es, et, ef, nF = build_sphere_mesh(150, RADIUS, JITTER, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": RADIUS, "jitter": JITTER,
            "p0": 3.72, "seed": SEED, "before_frame": 1},
           {"op": "vesicle_growth", "at": "vertex", "rate": 0.003, "every": 1},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.5,
            "Gamma": 0.1, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 20}]
    sched = ["seed_mesh_3d", "vesicle_growth", "shape_energy_3d", "reconnect_t1_3d"]   # NO divide -> clean mesh
    cfg = {"general": {"name": "dcmp", "seed": SEED, "n_frames": STOP, "dt": 1.0, "record_cap": 300,
                       "boundary": "free", "dim": 3, "world": [6 * RADIUS] * 3},
           "sets": {"vertex": {"n": int(Nv * 4)}}, "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, (es, et, ef, nF, Nv)


def normal_area(P):
    c = P.mean(0); N = np.zeros(3)
    for i in range(len(P)):
        N += np.cross(P[i] - c, P[(i + 1) % len(P)] - c)
    return c, N / (np.linalg.norm(N) + 1e-12), 0.5 * np.linalg.norm(N)


sim, (es, et, ef, nF, Nv) = build()
Hf, out = engine_run(sim, device="cpu")
m = Hf.level("vertex")._mesh
es, et, ef, nF = np.asarray(m["E_srce"]), np.asarray(m["E_trgt"]), np.asarray(m["E_face"]), int(m["nF"])
pos = out["sets"]["vertex"]["pos"][-1][:int(m["Nv"])].astype(np.float64)
rings = rings_from_flat_3d(es, et, ef, nF)
vf = face_geometry_3d(torch.as_tensor(pos), torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)[3].numpy()
Vbirth = m["Vbirth"].cpu().numpy()

# neighbour map (share an edge)
from collections import defaultdict
byedge = defaultdict(list)
for k in range(len(ef)):
    byedge[(min(int(es[k]), int(et[k])), max(int(es[k]), int(et[k])))].append(int(ef[k]))
nbr = defaultdict(set)
for fs in byedge.values():
    for a in fs:
        for b in fs:
            if a != b:
                nbr[a].add(b)


def dev_of(P, neigh_normals):
    _, n, _ = normal_area(P); mn = np.mean(neigh_normals, 0); mn /= (np.linalg.norm(mn) + 1e-12)
    return np.degrees(np.arccos(np.clip(n @ mn, -1, 1)))


# pick a ready-to-divide 5-7-gon with all neighbours present
cand = [f for f in range(nF) if rings[f] is not None and 5 <= len(rings[f]) <= 7
        and vf[f] >= 1.5 * Vbirth[f] and len(nbr[f]) >= 4]
f = cand[0]
r = rings[f]; P = pos[r]
neigh_normals = [normal_area(pos[rings[g]])[1] for g in nbr[f] if rings[g] is not None]
c, n0, a0 = normal_area(P)
print(f"\n===== MOTHER cell f={f}  ({len(r)}-gon)  vf={vf[f]:.3f}  Vbirth={Vbirth[f]:.3f} =====")
print(f"ring vertex ids: {list(r)}")
for i, v in enumerate(r):
    print(f"  v{i} id={int(v):4d}  pos={pos[v]}  |r|={np.linalg.norm(pos[v]):.3f}")
print(f"centroid={c}  |c|={np.linalg.norm(c):.3f}  normal={n0}  area={a0:.3f}")
print(f"deviation from {len(neigh_normals)} neighbours: {dev_of(P, neigh_normals):.1f} deg")

# apply the SAME Hertwig long-axis septum as divide_3d
cc = P.mean(0); _, _, vh = np.linalg.svd(P - cc, full_matrices=False); u = vh[0]
nrm = cc / (np.linalg.norm(cc) + 1e-9); w = np.cross(nrm, u); w /= (np.linalg.norm(w) + 1e-9)
mids = 0.5 * (P + np.roll(P, -1, 0)); proj = (mids - cc) @ w
ea, eb = int(np.argmax(proj)), int(np.argmin(proj))
print(f"\nlong axis u={u}  cut edges ea={ea} (v{ea}-v{(ea+1)%len(r)})  eb={eb} (v{eb}-v{(eb+1)%len(r)})")

rings2 = [(list(rr) if rr is not None else None) for rr in rings]   # divide_face_3d wants a LIST (mutates+appends)
pos_list = [p.copy() for p in pos]
res = divide_face_3d(rings2, pos_list, f, ea=ea, eb=eb)
print(f"divide_face_3d -> {'OK' if res is not None else 'FAILED'}")
if res is not None:
    idxB, m1, m2 = res
    pos2 = np.array(pos_list)
    print(f"new septum midpoint vertex ids: m1={m1} pos={pos2[m1]} |r|={np.linalg.norm(pos2[m1]):.3f}   "
          f"m2={m2} pos={pos2[m2]} |r|={np.linalg.norm(pos2[m2]):.3f}")
    for tag, fi in [("daughter A", f), ("daughter B", idxB)]:
        rr = rings2[fi]; Q = pos2[rr]; cc2, n2, a2 = normal_area(Q)
        newmark = ["*NEW*" if v in (m1, m2) else "" for v in rr]
        print(f"\n----- {tag} f={fi}  ({len(rr)}-gon)  area={a2:.3f} (mother {a0:.3f})  |c|={np.linalg.norm(cc2):.3f} -----")
        for i, v in enumerate(rr):
            print(f"  id={int(v):4d} {newmark[i]:5s} pos={pos2[v]}  |r|={np.linalg.norm(pos2[v]):.3f}")
        print(f"  normal={n2}   dot(mother_normal)={n2@n0:+.3f}   dev vs mother's neighbours: {dev_of(Q, neigh_normals):.1f} deg")
