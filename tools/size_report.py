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
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _traj(name, group):
    from plexus.paths import graphs_data_path
    p = os.path.join(graphs_data_path(), group, name, "trajectory.npz")
    return p if os.path.exists(p) else None


def _convention(traj):
    """The convention the run's size rules read (`vertex_ops.cell_size`): the polyhedron only when
    the seed declared `v0_from: polyhedron`, the wedge otherwise. Read from the spec archived beside
    the trajectory, so the report regresses the quantity the rule compared and not another one."""
    import yaml
    sp = os.path.join(os.path.dirname(traj), "spec.yaml")
    if os.path.exists(sp):
        for op in (yaml.safe_load(open(sp)).get("seed") or []):
            if op.get("op") in ("seed_mesh", "mesh_seed") and str(op.get("v0_from", "wedge")).lower() == "polyhedron":
                return "polyhedron"
    return "wedge"


def _volumes(z, t, convention="wedge"):
    """Cell volumes in the run's own convention (see `_convention`)."""
    import torch
    from plexus.operators.vertex_ops import apicobasal_geometry_3d, face_geometry_3d
    off = z["vertex__mesh_offsets"]
    nF, Nv = int(z["vertex__mesh_nF"][t]), int(z["vertex__mesh_Nv"][t])
    a, b = int(off[t]), int(off[t + 1])
    es = torch.as_tensor(z["vertex__mesh_E_srce"][a:b].astype(np.int64))
    et = torch.as_tensor(z["vertex__mesh_E_trgt"][a:b].astype(np.int64))
    ef = torch.as_tensor(z["vertex__mesh_E_face"][a:b].astype(np.int64))
    P = torch.as_tensor(z["vertex__pos"][t][:Nv]).float()
    if convention == "polyhedron" and "vertex__sep" in z.files:
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


def row(name, group="tissue", birth_lag=12, until=None):
    """`until`: score only the frames before it -- the first frame `tools/spheroid_gauge.py`
    refuses, so a table never reads a crumpled mesh."""
    p = _traj(name, group)
    if p is None:
        return None
    z = np.load(p)
    nFs = z["vertex__mesh_nF"]; T = len(nFs)
    if until is not None:
        T = max(2, min(T, int(until)))
    # AGES FIRST, VOLUMES ONLY WHERE THEY ARE READ. Lineage comes from the recorded `age`, which is
    # cheap; the polyhedron geometry is not, and a run of 800 frames and 2,000 cells took minutes
    # per spec when every frame was rebuilt. The volumes are needed at each birth's read frame,
    # at the frame before each division, and on a stride for the population medians.
    ages = [_age(z, t) for t in range(T)]
    births = []                                   # (index, frame) of every birth after frame 0
    for t in range(1, T):
        n0, a0, a1 = len(ages[t - 1]), ages[t - 1], ages[t]
        for i in range(len(ages[t])):
            if i >= n0 or a1[i] < a0[i]:
                births.append((i, t))
    resets = {}
    for i, t in births:
        resets.setdefault(i, []).append(t)
    pairs = []
    for i, tb in births:
        later = [t for t in resets[i] if t > tb]
        if later:
            td = later[0]
            pairs.append((i, tb, min(tb + birth_lag, td - 1), td))
    # the settle window: the reference is the median once the size rules start reading
    ref_frame = int(z["vertex__mesh_scalar_ref_frame"][0]) if "vertex__mesh_scalar_ref_frame" in z.files else 0
    stride = max(1, T // 80)
    need = {0, T - 1, min(ref_frame, T - 1)} | set(range(0, T, stride)) \
        | {tr for _, _, tr, _ in pairs} | {td - 1 for _, _, _, td in pairs}
    conv = _convention(p)
    V = {t: _volumes(z, t, conv) for t in sorted(need)}
    v_ref = float(np.median(V[min(ref_frame, T - 1)]))
    cyc = [(V[tr][i] / v_ref, V[td - 1][i] / v_ref, td - tb) for i, tb, tr, td in pairs]
    out = dict(name=name, T=T, n0=int(nFs[0]), nT=int(nFs[T - 1]), cycles=len(cyc))
    med_t = sorted(t for t in V if t % stride == 0 or t == T - 1)
    med = np.array([np.median(V[t]) for t in med_t]) / v_ref
    out["med_end"] = float(med[-1])
    out["cv_end"] = float(V[T - 1].std() / V[T - 1].mean())
    if len(cyc) >= 10:
        c = np.array(cyc)
        Vb, Vd, L = c[:, 0], c[:, 1], c[:, 2]
        out["slope"] = float(np.polyfit(Vb, Vd - Vb, 1)[0])
        out["r_L_Vb"] = float(np.corrcoef(Vb, L)[0, 1])
        out["cv_Vd"] = float(Vd.std() / Vd.mean())
        out["mean_L"] = float(L.mean())
        # drift of the median volume over the last two mean cycle lengths, as a fraction
        w = int(min(T - 1, 2 * L.mean()))
        t0 = min(med_t, key=lambda t: abs(t - (T - 1 - w)))
        out["drift"] = float(med[-1] / med[med_t.index(t0)] - 1.0) if w > 0 else float("nan")
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
    ap.add_argument("--until", type=int, default=None,
                    help="score only frames before this one (the gauge's first refused frame)")
    ap.add_argument("--gauge", action="store_true",
                    help="run tools/spheroid_gauge.py first and score up to its first refused frame")
    a = ap.parse_args()
    names = list(a.names)
    if a.glob:
        from plexus.paths import graphs_data_path
        for g in a.glob:
            names += sorted(os.path.basename(p) for p in
                            glob.glob(os.path.join(graphs_data_path(), a.group, g))
                            if os.path.isdir(p))
    print(f"{'spec':<26} {'cells':>11} {'cycles':>6} {'slope':>6} {'r(L,Vb)':>8} {'CV(Vd)':>7} "
          f"{'L':>6} {'med V/vref':>10} {'drift':>6} {'CV(V)':>6}")
    for n in dict.fromkeys(names):
        until = a.until
        if a.gauge:
            import spheroid_gauge as G
            # THE SHELL BANDS CUT THE WINDOW. The prism bands (trapezoid, shear, tilt, in-cell
            # thickness) are layer 0's objective and trip on the first divisions of every arm
            # while that layer is open; a window cut there would hold no cycles at all.
            fb, _ = G.gauge(_traj(n, a.group), every=50, verbose=False, which="shell")
            until = fb[0] if fb is not None else None
        r = row(n, a.group, a.birth_lag, until)
        if r is not None:
            r["name"] = f"{n}<{until}" if until else n
        if r is None:
            print(f"{n:<20} -- no trajectory on disk"); continue
        f = lambda k, w, d=2: (f"{r[k]:{w}.{d}f}" if k in r else " " * (w - 1) + "-")   # noqa: E731
        print(f"{r['name']:<26} {r['n0']:5d}->{r['nT']:<5d} {r['cycles']:6d} {f('slope', 6)} "
              f"{f('r_L_Vb', 8)} {f('cv_Vd', 7)} {f('mean_L', 6, 0)} {r['med_end']:10.2f} "
              f"{f('drift', 6)} {r['cv_end']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
