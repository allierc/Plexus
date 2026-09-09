#!/usr/bin/env python
"""Cell-volume dispersion over a run, and where it comes from -- one row per spec.

WHAT IT SEPARATES, because the two are different problems with different fixes:

    CV(V)     the ACTUAL polyhedron volumes. What you see.
    CV(V0f)   the TARGET volumes the cells are asking for.
    lag       CV(V / V0f) -- how far the mechanics is from delivering those targets.

RAISING `k_v` COMPRESSES ONLY THE LAG. The energy makes each cell track ITS OWN target harder, so
in the stiff limit `CV(V) -> CV(V0f)` and not to zero. Measured on `divide_growing_ball`, the lag
sits at 0.12 for an entire 800-frame run while `CV(V0f)` climbs 0.017 -> 0.331: the growing term is
the one the mechanics cannot touch, and a sweep over `k_v` that reported only `CV(V)` would have
looked like it was doing something.

THE CONTROL THAT MAKES IT READABLE is `seed_mesh v0_uniform: true` -- every cell asks for the same
volume. With per-cell targets no arm can beat the target spread, so the sweep measures the seed
rather than the energy; with identical targets the shell settles to CV 0.006, which is what the
mechanics is actually capable of.

    PYTHONPATH=src python tools/cv_report.py cv_target_uniform cv_kv_quad ...
    PYTHONPATH=src python tools/cv_report.py --group tissue --glob 'cv_*'
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


def row(name, group="tissue"):
    import torch
    from plexus.operators.vertex_ops import apicobasal_geometry_3d
    p = _traj(name, group)
    if p is None:
        return None
    z = np.load(p)
    off, foff = z["vertex__mesh_offsets"], z["vertex__mesh_face_offsets"]
    nF, Nv = z["vertex__mesh_nF"], z["vertex__mesh_Nv"]
    pos, sep = z["vertex__pos"], z["vertex__sep"]
    # `V0f` MOVED TO THE CELL SET. It was `vertex__mesh_V0f`, a ragged per-face column cut by the
    # face offsets; it is `cell__V0f` now, a dense [T, buffer, 1] block cut by the frame's `nF`.
    # Both spellings are read, because trajectories from before the move are still on disk and a
    # report that silently dropped its target column would have looked like a run with no targets.
    V0f = z["vertex__mesh_V0f"] if "vertex__mesh_V0f" in z.files else None
    V0c = z["cell__V0f"] if "cell__V0f" in z.files else None
    T = len(nF)

    def at(t):
        a, b = int(off[t]), int(off[t + 1]); n = int(Nv[t]); f = int(nF[t])
        es = torch.as_tensor(z["vertex__mesh_E_srce"][a:b].astype(np.int64))
        et = torch.as_tensor(z["vertex__mesh_E_trgt"][a:b].astype(np.int64))
        ef = torch.as_tensor(z["vertex__mesh_E_face"][a:b].astype(np.int64))
        P = torch.as_tensor(pos[t][:n]).float(); S = torch.as_tensor(sep[t][:n]).float()
        vp, _, _, _ = apicobasal_geometry_3d(P, S, es, et, ef, f)
        v = vp.numpy()
        v0 = (V0f[int(foff[t]):int(foff[t] + f)] if V0f is not None
              else (np.asarray(V0c[t])[:f, 0] if V0c is not None else None))
        h = float(S.norm(dim=1).median()) * 2.0
        return v, v0, h, f

    v0_, t0_, h0_, f0_ = at(0)
    vN, tN, hN, fN = at(T - 1)
    cv0 = v0_.std() / v0_.mean()
    cvN = vN.std() / vN.mean()
    cvt = (tN.std() / tN.mean()) if tN is not None and tN.mean() > 0 else float("nan")
    lag = float("nan")
    if tN is not None:
        r = vN / np.maximum(tN, 1e-12)
        lag = r.std() / r.mean()
    return dict(name=name, T=T, nF0=f0_, nFN=fN, cv0=cv0, cvN=cvN, cvt=cvt, lag=lag,
                h0=h0_, hN=hN, meanN=vN.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--glob", default=None, help="spec-name glob under the group's data dir")
    a = ap.parse_args()
    names = list(a.names)
    if a.glob:
        from plexus.paths import graphs_data_path
        names += sorted(os.path.basename(p) for p in
                        glob.glob(os.path.join(graphs_data_path(), a.group, a.glob))
                        if os.path.isdir(p))
    print(f"{'spec':<22} {'frames':>6} {'cells':>11} {'CV(V) 0':>8} {'CV(V) end':>10} "
          f"{'CV(V0f)':>8} {'lag':>7} {'h end':>6} {'mean V':>7}")
    for n in dict.fromkeys(names):
        r = row(n, a.group)
        if r is None:
            print(f"{n:<22} -- no trajectory on disk"); continue
        print(f"{r['name']:<22} {r['T']:6d} {r['nF0']:5d}->{r['nFN']:<5d} {r['cv0']:8.4f} "
              f"{r['cvN']:10.4f} {r['cvt']:8.4f} {r['lag']:7.4f} {r['hN']:6.3f} {r['meanN']:7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
