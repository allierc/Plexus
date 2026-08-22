#!/usr/bin/env python
"""Turn a REGENERATED core trajectory into the surface archive `mesh_contact` reads.

    "do not use replay, always regenerate data"                        -- Cedric, 22 August

WHAT THAT RULE MEANS HERE, AND WHAT IT CANNOT MEAN. Gate 04 couples a growing epithelium to an MPM
matrix, and it does so through a REPLAY: the tissue is not solved in pass 2, its surface is
prescribed frame by frame from a cache. That is not laziness, it is the only thing the two clocks
allow. The tissue runs at `dt = 1.0` for 401 frames -- 66.8 hours at 600 s a frame -- while the
matrix runs at `dt = 3.2e-3` for 199 frames, which is 0.64 seconds. Solving both in one schedule
would need about 10^5 matrix frames per tissue frame. The one-way kinematic coupling is a modelling
decision the prototype made on purpose, and undoing it is a research question, not a promotion.

SO THE RULE APPLIES TO THE INPUT, NOT TO THE COUPLING. What must never be depended on is a 32.7 MB
npz somebody built in August from a spec that no longer exists on disk -- and that is exactly what
gate 04 reads today: `cellfix_B_new_f401_x4_c4a5698982.npz`, whose parent
`tissue.CELL_SPEC` now resolves to a DIFFERENT MODEL (`rate: 0.03` against the 0.003457 that built
it, reaching 1,451 cells by frame 60 against the cache's 227). A gate that reads it is gating
against a file, not against a model.

This makes the replay's input an artefact of a gate we run: pass 1 is a declared spec in
`config/gates/`, regenerated from source with its own threshold table, and this writes its recorded
surface into the layout pass 2 consumes.

THE ENGINE ALREADY RECORDS EVERYTHING THE CACHE CARRIED, which is the other half of why this is now
possible. The okuda cache's per-mesh keys are `pos`, `E_srce`, `E_trgt`, `E_face`, `nF`, `Nv`,
`age`, `ndiv`, `myo_med`, `myo`, `myo_amount`. `MeshTable.FACE_RECORD` covers the per-face ones and
`EDGE_RECORD` the two per-half-edge ones, so the core trajectory is a superset -- it also has the
operators' own counters, which the cache did not.

    python tools/export_tissue.py --gate gate_04_tissue --out log/gates/_tissue/<name>.npz
    python tools/export_tissue.py --data <dir with trajectory.npz> --out <file> --keep 200
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import gate_measures as GM                                            # noqa: E402


def export(data_dir, out, keep=200, scale_to=None):
    """Write the surface archive. Returns a dict of what went in.

    `keep` MESHES OUT OF EVERY RECORDED ROW, chosen exactly the way the okuda cache chose them --
    `np.linspace(0, T-1, keep)`, first and last always in. Not because 200 is special but because
    `mesh_contact`'s `mesh_stride` arithmetic is written against a fixed count, and a cache with a
    different one silently changes how far the surface advances per pass-2 frame.
    """
    T = GM.open_traj(data_dir)
    n = T.n_rows()
    idx = np.unique(np.linspace(0, n - 1, min(keep, n)).astype(int))
    z, cells, rad, cen = {}, [], [], []
    for j, t in enumerate(idx):
        p = T.pos(int(t))
        c = p.mean(0)
        # CENTRED AND OPTIONALLY RESCALED. `mesh_contact` multiplies by its own `scale:` and adds
        # its own `centre:`, so what belongs in the file is the tissue in ITS OWN units about its
        # own centroid -- which is what the okuda cache holds. Rescaling here as well would apply
        # the factor twice, and the result would look like a contact that starts too early.
        z[f"m{j}_pos"] = np.asarray(p - c, np.float32)
        es, et, ef = T.half_edges(int(t))
        z[f"m{j}_E_srce"] = np.asarray(es, np.int32)
        z[f"m{j}_E_trgt"] = np.asarray(et, np.int32)
        z[f"m{j}_E_face"] = np.asarray(ef, np.int32)
        z[f"m{j}_nF"] = np.int32(T.nF(int(t)))
        z[f"m{j}_Nv"] = np.int32(T.nV(int(t)))
        for nm in ("age", "ndiv", "myo_med"):
            v = T.face_col(nm, int(t))
            if v is not None:
                z[f"m{j}_{nm}"] = np.asarray(v, np.float32)
        for nm in ("myo", "myo_amount"):
            v = T.edge_col(nm, int(t))
            if v is not None:
                z[f"m{j}_{nm}"] = np.asarray(v, np.float32)
        r = np.linalg.norm(p - c, axis=1)
        cells.append(T.nF(int(t))); rad.append(float(np.median(r))); cen.append(c)
    z["mesh_frames"] = np.asarray(idx, np.int32)
    z["n_cells"] = np.asarray(cells, np.int32)
    z["r_apical"] = np.asarray(rad, np.float32)
    z["r_med"] = np.asarray(rad, np.float32)
    z["centroid"] = np.asarray(cen, np.float32)
    # THE CAMERA BOX, the `run_box` convention: the widest the tissue ever gets, plus 12%.
    z["Lbox"] = np.float32(max(float(np.abs(np.asarray(GM.open_traj(data_dir).pos(int(t))) ).max())
                               for t in idx) * 1.12)
    z["r_eq"] = z["r_apical"]; z["r_ax"] = z["r_apical"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **z)
    return dict(meshes=len(idx), rows=n, cells_start=cells[0], cells_end=cells[-1],
                r_start=rad[0], r_end=rad[-1], Lbox=float(z["Lbox"]),
                mb=os.path.getsize(out) / 1e6, out=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=None, help="a spec name under config/gates/")
    ap.add_argument("--data", default=None, help="a directory holding trajectory.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", type=int, default=200)
    a = ap.parse_args()
    if a.gate:
        from plexus.paths import graphs_data_path
        d = graphs_data_path("gates", a.gate)
        out = a.out or os.path.join(ROOT, "log", "gates", "_tissue", f"{a.gate}.npz")
    else:
        d, out = a.data, a.out
    if not d or not os.path.isdir(d):
        print(f"  no trajectory directory: {d}")
        return 3
    r = export(d, out, keep=a.keep)
    print(f"  {r['meshes']} meshes from {r['rows']} recorded rows -> "
          f"{os.path.relpath(r['out'], ROOT)} ({r['mb']:.1f} MB)")
    print(f"  {r['cells_start']} -> {r['cells_end']} cells, "
          f"median radius {r['r_start']:.4f} -> {r['r_end']:.4f}, Lbox {r['Lbox']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
