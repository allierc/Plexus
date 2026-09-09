#!/usr/bin/env python
"""Why an apico-basal cell that was sentenced to death is still alive at the last frame.

THE OBSERVATION THIS EXISTS FOR. `cell_die` kills by shrinking a cell's target volume until the
cell contracts, its neighbours shed it through `edge_flip` one short edge at a time, and
`face_collapse_3d` extrudes the surviving triangle to a point. On the mid-surface sheet that
pathway completes: `apop2_sheet_half` ends with exactly one cell mid-extrusion, its measured volume
already at -0.0002. On the apico-basal twin `apop2_ab_half` it does not -- 59 cells end the run with
`V0f` shrunk all the way to 0, an actual polyhedron volume of 0.150 against a healthy cell's 0.264,
and a ring valence of 6 where extrusion needs 3. They are under sentence for ever, they hold a
permanent yellow band across the tissue, and they are why the AB spheroid ends up pear-shaped where
the sheet ends up symmetric.

WHAT THE TWO COLUMNS SEPARATE, because "it did not die" has two possible causes and they need
different fixes:

    volume ratio   the cell's ACTUAL volume over the population's median target. If a sentenced
                   cell still reads near 1.0, the energy declined to follow the target down -- the
                   surface term (`kappa_s`) is holding the cell open against the volume term
                   (`k_v`), and the fix is in `cell_mechanics`.
    ring valence   how many edges the cell's mid-surface ring still has. If the volume DID collapse
                   but the valence is still 6, the cell shrank and nobody shed it -- `edge_flip`'s
                   `l_th_frac` never found an edge short enough, and the fix is in the topology.

A sweep over `kappa_s` moves the first and not the second; a sweep over `l_th_frac` moves the
second and not the first. Running both and reading this table is what tells them apart, which is
why the control arm exists rather than a single sweep.

    PYTHONPATH=src python tools/apop_stall.py --glob 'apop2_ks*' apop2_flip060 apop2_ab_half
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def row(name, group="tissue"):
    import torch
    from plexus.operators.vertex_ops import apicobasal_geometry_3d, face_geometry_3d
    from plexus.paths import graphs_data_path
    p = os.path.join(graphs_data_path(), group, name, "trajectory.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    nFs = np.asarray(z["vertex__mesh_nF"])
    t = len(nFs) - 1
    off = z["vertex__mesh_offsets"]
    nF, Nv = int(nFs[t]), int(z["vertex__mesh_Nv"][t])
    a, b = int(off[t]), int(off[t + 1])
    es = torch.as_tensor(z["vertex__mesh_E_srce"][a:b].astype(np.int64))
    et = torch.as_tensor(z["vertex__mesh_E_trgt"][a:b].astype(np.int64))
    ef = torch.as_tensor(z["vertex__mesh_E_face"][a:b].astype(np.int64))
    P = torch.as_tensor(z["vertex__pos"][t][:Nv]).double()
    # APICO-BASAL RUNS ARE POLYHEDRA AND MID-SURFACE RUNS ARE WEDGES, and the cell's volume has to
    # be measured in the convention its own energy uses or the ratio below is comparing two solids.
    if "vertex__sep" in z.files:
        S = torch.as_tensor(z["vertex__sep"][t][:Nv]).double()
        v, _, _, _ = apicobasal_geometry_3d(P, S, es, et, ef, nF)
    else:
        _, _, _, v = face_geometry_3d(P, es, et, ef, nF)
    v = v.numpy()
    flag = np.asarray(z["cell__apop_flag"][t])[:nF, 0]
    V0f = np.asarray(z["cell__V0f"][t])[:nF, 0]
    val = np.bincount(ef.numpy(), minlength=nF)          # edges per cell = ring valence
    mk = np.where(flag > 0)[0]
    live = np.ones(nF, bool); live[mk] = False
    v_target = float(np.median(V0f[live])) if live.any() else float("nan")
    R = float(np.linalg.norm(P.numpy() - P.numpy().mean(0), axis=1).mean())
    return dict(name=name, T=len(nFs), nF0=int(nFs[0]), nFN=nF, stalled=len(mk),
                killed=int(nFs[0]) - nF, R=R,
                vratio=(float(np.median(v[mk])) / v_target) if len(mk) and v_target else float("nan"),
                val=(float(np.median(val[mk])) if len(mk) else float("nan")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--glob", default=None)
    a = ap.parse_args()
    names = list(a.names)
    if a.glob:
        names += sorted(os.path.basename(p)[:-5] for p in
                        glob.glob(os.path.join(ROOT, "config", a.group, a.glob + ".yaml")))
    print(f"{'run':<20} {'cells 0->end':>13} {'killed':>7} {'STALLED':>8} "
          f"{'vol/target':>10} {'valence':>8} {'radius':>7}")
    for n in dict.fromkeys(names):
        r = row(n, a.group)
        if r is None:
            print(f"{n:<20} -- no trajectory on disk"); continue
        print(f"{r['name']:<20} {r['nF0']:5d}->{r['nFN']:<6d} {r['killed']:7d} {r['stalled']:8d} "
              f"{r['vratio']:10.3f} {r['val']:8.1f} {r['R']:7.3f}")
    print("\n  vol/target ~1  = the energy never shrank the cell   -> the surface term (kappa_s)")
    print("  vol/target ~0 but valence > 3 = it shrank and nobody shed it -> edge_flip l_th_frac")
    return 0


if __name__ == "__main__":
    sys.exit(main())
