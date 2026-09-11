#!/usr/bin/env python
"""The size-control test of Ginzberg, Kafri & Kirschner (Science 2015), run on an archived tissue.

ONE ROW PER SPEC, FOUR NUMBERS THAT DECIDE THE RULE. For every cell that was born AND divided
inside the run, the volume it was born with, `V_b`, and the volume it added before dividing,
`dV = V_d - V_b`, both in units of `v_ref` (the seed-time median cell volume):

    slope of dV on V_b     -1  a sizer: division volume independent of birth volume
                            0  an adder: a constant increment, deviations halve per generation
                           +1  a timer under exponential growth: the deviation is passed on
    r(cycle length, V_b)   negative when small-born cells are held longer -- the G1 coupling
    CV(V_d)                the spread of division volumes; equals the drawn jitter for a sizer
    median V drift         median volume over the last two cycles against the seed: a rule that
                           "controls size" while the population's size runs away controls nothing.
                           A sizer can remove x2 per division, so exp(3 rate (t_s+t_g2+t_m)) < 2
                           is the condition for a checkpoint to act at all.

WHERE BIRTH IS READ. Twelve frames after the septum, not at it: at the septum the two daughters
are the geometric halves of a ring cut across its short axis, which are not volume halves (CV 0.27
at the cut, 0.16 twelve frames later on the withdrawn cvd2_adder_tension), and the mechanics
pulls them to their targets over the next few relax passes. Reading at the cut regresses on noise
the rule never saw. `--birth-lag` moves it.

LINEAGE. Read from `age` (reset to 0 at a division; daughter A keeps the mother's index, daughter
B is appended) -- sound while nothing dies, which is every spec this was written for. R4 of
`notes/size_cycle/SIZE_CYCLE_PLAN.md` records a `cell_id` block so this argument is not needed.

    PYTHONPATH=src python tools/size_report.py size_sizer size_adder size_timer
    PYTHONPATH=src python tools/size_report.py --group tissue --glob 'size_*' --glob 'cycle_*'
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _traj(name, group):
    from plexus.paths import graphs_data_path
    p = os.path.join(graphs_data_path(), group, name, "trajectory.npz")
    return p if os.path.exists(p) else None


def _volumes(z, t):
    """Polyhedron volumes when the run carries a separation, wedge volumes otherwise."""
    import torch
    from plexus.operators.vertex_ops import apicobasal_geometry_3d, face_geometry_3d
    off = z["vertex__mesh_offsets"]
    nF, Nv = int(z["vertex__mesh_nF"][t]), int(z["vertex__mesh_Nv"][t])
    a, b = int(off[t]), int(off[t + 1])
    es = torch.as_tensor(z["vertex__mesh_E_srce"][a:b].astype(np.int64))
    et = torch.as_tensor(z["vertex__mesh_E_trgt"][a:b].astype(np.int64))
    ef = torch.as_tensor(z["vertex__mesh_E_face"][a:b].astype(np.int64))
    P = torch.as_tensor(z["vertex__pos"][t][:Nv]).float()
    if "vertex__sep" in z.files:
        S = torch.as_tensor(z["vertex__sep"][t][:Nv]).float()
        v, _, _, _ = apicobasal_geometry_3d(P, S, es, et, ef, nF)
    else:
        _, _, _, v = face_geometry_3d(P, es, et, ef, nF)
    return v.numpy().astype(np.float64)


def _age(z, t):
    n = int(z["vertex__mesh_nF"][t])
    if "cell__age" in z.files:
        return np.asarray(z["cell__age"][t])[:n, 0]
    foff = z["vertex__mesh_face_offsets"]
    return z["vertex__mesh_age"][int(foff[t]):int(foff[t]) + n]


def row(name, group="tissue", birth_lag=12):
    p = _traj(name, group)
    if p is None:
        return None
    z = np.load(p)
    T = len(z["vertex__mesh_nF"])
    V = [_volumes(z, t) for t in range(T)]
    ages = [_age(z, t) for t in range(T)]
    births = []                                   # (index, frame) of every birth after frame 0
    for t in range(1, T):
        n0, a0, a1 = len(V[t - 1]), ages[t - 1], ages[t]
        for i in range(len(V[t])):
            if i >= n0 or a1[i] < a0[i]:
                births.append((i, t))
    resets = {}
    for i, t in births:
        resets.setdefault(i, []).append(t)
    v_ref = float(np.median(V[0]))
    cyc = []
    for i, tb in births:
        later = [t for t in resets[i] if t > tb]
        if not later:
            continue
        td = later[0]
        tr = min(tb + birth_lag, td - 1)
        cyc.append((V[tr][i] / v_ref, V[td - 1][i] / v_ref, td - tb))
    out = dict(name=name, T=T, n0=len(V[0]), nT=len(V[-1]), cycles=len(cyc))
    med = np.array([np.median(v) for v in V]) / v_ref
    out["med_end"] = float(med[-1])
    out["cv_end"] = float(V[-1].std() / V[-1].mean())
    if len(cyc) >= 10:
        c = np.array(cyc)
        Vb, Vd, L = c[:, 0], c[:, 1], c[:, 2]
        out["slope"] = float(np.polyfit(Vb, Vd - Vb, 1)[0])
        out["r_L_Vb"] = float(np.corrcoef(Vb, L)[0, 1])
        out["cv_Vd"] = float(Vd.std() / Vd.mean())
        out["mean_L"] = float(L.mean())
        # drift of the median volume over the last two mean cycle lengths, as a fraction
        w = int(min(T - 1, 2 * L.mean()))
        out["drift"] = float(med[-1] / med[-1 - w] - 1.0) if w > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--glob", action="append", default=[],
                    help="spec-name glob under the group's data dir; repeatable")
    ap.add_argument("--birth-lag", type=int, default=12,
                    help="frames after the septum at which the birth volume is read")
    a = ap.parse_args()
    names = list(a.names)
    if a.glob:
        from plexus.paths import graphs_data_path
        for g in a.glob:
            names += sorted(os.path.basename(p) for p in
                            glob.glob(os.path.join(graphs_data_path(), a.group, g))
                            if os.path.isdir(p))
    print(f"{'spec':<20} {'cells':>11} {'cycles':>6} {'slope':>6} {'r(L,Vb)':>8} {'CV(Vd)':>7} "
          f"{'L':>6} {'med V/vref':>10} {'drift':>6} {'CV(V)':>6}")
    for n in dict.fromkeys(names):
        r = row(n, a.group, a.birth_lag)
        if r is None:
            print(f"{n:<20} -- no trajectory on disk"); continue
        f = lambda k, w, d=2: (f"{r[k]:{w}.{d}f}" if k in r else " " * (w - 1) + "-")   # noqa: E731
        print(f"{r['name']:<20} {r['n0']:5d}->{r['nT']:<5d} {r['cycles']:6d} {f('slope', 6)} "
              f"{f('r_L_Vb', 8)} {f('cv_Vd', 7)} {f('mean_L', 6, 0)} {r['med_end']:10.2f} "
              f"{f('drift', 6)} {r['cv_end']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
