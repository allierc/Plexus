#!/usr/bin/env python
"""Every death in a run, joined to what the cell was when it died -- the apoptosis rig's report.

A cell dies the frame after its `cell_id` last appears with no child carrying it as `parent_id`
(a division ends an id too, but leaves two children). For each death the report takes, at the
cell's last frame: its polyhedron volume over `v_ref`, its volume at birth (read `--birth-lag`
frames after it appeared), its age in frames, its cycle phase if the run records one, its
number of neighbours (ring edges), and how long it carried `apop_flag` before it was removed --
the shrink-shed-extrude latency. One row per spec summarises them, so a `cell_die` model can be
read against a size rule, a cycle model or a topology (T1 on / off) in one table.

    PYTHONPATH=src python tools/death_report.py --glob 'rig_apop_*'
    GNN_OUTPUT_ROOT=log/size_cycle/R4rig PYTHONPATH=src python tools/death_report.py rig_apop_small_sizer --csv deaths.csv
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def deaths(traj, birth_lag=12):
    import size_report as S
    z = np.load(traj)
    nFs = z["vertex__mesh_nF"]; T = len(nFs)
    if "cell__cell_id" not in z.files:
        raise SystemExit(f"{traj}: no cell_id block recorded; declare cell_id and parent_id on the cell set")
    ids = [np.rint(np.asarray(z["cell__cell_id"][t])[:int(nFs[t]), 0]).astype(np.int64) for t in range(T)]
    pids = [np.rint(np.asarray(z["cell__parent_id"][t])[:int(nFs[t]), 0]).astype(np.int64) for t in range(T)]
    flag = ([np.asarray(z["cell__apop_flag"][t])[:int(nFs[t]), 0] for t in range(T)]
            if "cell__apop_flag" in z.files else None)
    phase = ([np.asarray(z["cell__phase"][t])[:int(nFs[t]), 0] for t in range(T)]
             if "cell__phase" in z.files else None)
    first, last, where, marked = {}, {}, {}, {}
    for t in range(T):
        for j, c in enumerate(ids[t]):
            c = int(c)
            first.setdefault(c, t); last[c] = t; where[(c, t)] = j
            if flag is not None and flag[t][j] > 0 and c not in marked:
                marked[c] = t
    parents = set(int(p) for t in range(T) for p in pids[t])
    dead = [c for c, tl in last.items() if tl < T - 1 and c not in parents]
    conv = S._convention(traj)
    need = {0, 60} | {last[c] for c in dead} | {min(first[c] + birth_lag, last[c]) for c in dead}
    V = {t: S._volumes(z, t, conv) for t in sorted(need)}
    v_ref = float(np.median(V[min(60, T - 1)]))
    off = z["vertex__mesh_offsets"]
    rows = []
    for c in dead:
        tl, tb = last[c], first[c]; j = where[(c, tl)]
        a, b = int(off[tl]), int(off[tl + 1])
        ef = z["vertex__mesh_E_face"][a:b]
        nb = int((ef == j).sum())                              # ring edges = neighbours
        tr = min(tb + birth_lag, tl)
        rows.append(dict(id=c, t_death=tl + 1, age=tl + 1 - tb, v_death=float(V[tl][j] / v_ref),
                         v_birth=float(V[tr][where[(c, tr)]] / v_ref), neighbours=nb,
                         phase=(int(round(phase[tl][j])) if phase is not None else -1),
                         latency=(tl + 1 - marked[c]) if c in marked else -1,
                         seeded=int(tb == 0)))
    n_end = int(nFs[-1]); cell_frames = float(np.sum(nFs))
    return rows, dict(T=T, n0=int(nFs[0]), nT=n_end, cell_frames=cell_frames, v_ref=v_ref)


def summary(name, rows, meta):
    d = {"spec": name, "cells": f"{meta['n0']}->{meta['nT']}", "deaths": len(rows)}
    d["per_100cf"] = 100.0 * len(rows) / max(meta["cell_frames"], 1.0)      # deaths per 100 cell-frames
    if rows:
        vd = np.array([r["v_death"] for r in rows]); vb = np.array([r["v_birth"] for r in rows])
        age = np.array([r["age"] for r in rows]); nb = np.array([r["neighbours"] for r in rows])
        lat = np.array([r["latency"] for r in rows if r["latency"] >= 0])
        ph = np.array([r["phase"] for r in rows if r["phase"] >= 0])
        d.update(v_death=float(np.median(vd)), v_birth=float(np.median(vb)), age=float(np.median(age)),
                 nb=float(np.median(nb)), nb_le4=float(np.mean(nb <= 4)),
                 latency=(float(np.median(lat)) if lat.size else float("nan")),
                 g1_frac=(float(np.mean(ph == 0)) if ph.size else float("nan")),
                 t_first=int(min(r["t_death"] for r in rows)))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*"); ap.add_argument("--group", default="tissue")
    ap.add_argument("--glob", action="append", default=[]); ap.add_argument("--birth-lag", type=int, default=12)
    ap.add_argument("--csv", default=None, help="write every death of every spec to this CSV")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path
    names = list(a.names)
    for g in a.glob:
        names += sorted(os.path.basename(p) for p in glob.glob(os.path.join(graphs_data_path(), a.group, g)) if os.path.isdir(p))
    print(f"{'spec':<30} {'cells':>11} {'deaths':>6} {'/100cf':>7} {'V_death':>7} {'V_birth':>7} {'age':>5} "
          f"{'nb':>4} {'nb<=4':>5} {'latency':>7} {'G1':>5} {'first':>5}")
    allrows = []
    for n in dict.fromkeys(names):
        p = os.path.join(graphs_data_path(), a.group, n, "trajectory.npz")
        if not os.path.exists(p):
            print(f"{n:<30} -- no trajectory on disk"); continue
        rows, meta = deaths(p, a.birth_lag); d = summary(n, rows, meta)
        for r in rows:
            r["spec"] = n; allrows.append(r)
        f = lambda k, w, dd=2: (f"{d[k]:{w}.{dd}f}" if k in d and isinstance(d[k], float) else (f"{d[k]:{w}d}" if k in d else " " * (w - 1) + "-"))  # noqa: E731
        print(f"{n:<30} {d['cells']:>11} {d['deaths']:6d} {d['per_100cf']:7.3f} {f('v_death', 7)} {f('v_birth', 7)} "
              f"{f('age', 5, 0)} {f('nb', 4, 0)} {f('nb_le4', 5)} {f('latency', 7, 0)} {f('g1_frac', 5)} {f('t_first', 5)}")
    if a.csv and allrows:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
        print(f"wrote {len(allrows)} deaths -> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
