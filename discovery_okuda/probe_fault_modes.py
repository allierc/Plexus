#!/usr/bin/env python
"""probe_fault_modes -- settle Phase 0 item 0B-11: is the mesh-damage count measuring damage?

THE QUESTION. On archived run r01_03_5e3159_3 the legacy blended "hollow" count correlates with
the number of TIP CELLS at r = +0.971, while the tube grows monotonically (length 2.96 -> 19.00),
narrows smoothly, and never discontinuously fails. Cedric watched the movie: the tube goes
straight and does not explode. So the blend appears to be counting the tube rather than damage,
which would mean the metric penalises exactly the phenotype the campaign is looking for.

A FIRST HYPOTHESIS, TESTED AND REFUTED. I proposed that the "folded" test conflates CURVATURE with
folding, since it thresholds the angle between a face normal and its neighbours' mean normal, and
a narrow tube is strongly curved. Direct test: perfect spheres, 150 cells, radius 5.0 down to 0.7.
Every radius gives mean deviation 1.8 degrees, max 5.6, ZERO cells flagged -- because scaling a
sphere scales its cells too, and averaging the neighbours cancels smooth curvature to first order.
The test measures NON-SMOOTHNESS, not curvature. Hypothesis dead.

WHAT THIS PROBE DOES INSTEAD. Grows a real tube and, at every recorded frame, evaluates the three
failure modes SEPARATELY (folded / sliver / broken) against the tip-cell count. The blend cannot
tell us which term drives the correlation; the split can. The remaining suspicion is the SLIVER
term: division concentrates at the activator-rich growing tip, and a just-divided daughter is a
sliver for a frame or two, so a longer tube would mechanically produce more slivers without
anything being wrong.

WHAT WOULD SETTLE IT
  * `broken` tracks the tip     -> the tube really is damaging the mesh; the horizon is honest.
  * `sliver` tracks the tip     -> the blend was counting cell division. Benign. The horizon must
                                   key on `broken` alone, which is what curve_shape now demands.
  * `folded` tracks the tip     -> the tube is genuinely warping caps; worth understanding before
                                   the horizon can use it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "prototype", "Tyssue"))


def main(frames=300, every=10):
    import turing_vertex_study as S
    from plexus.engine import run as engine_run
    from tyssue_diag import mesh_faults
    from tube_analysis import cell_census, protrusion_ratio, _cell_centroids, tube_diameter

    k = {**S.BASE, "name": "_probe_faults", "conserve_amount": 0, "rho": 0.05, "a_sw": 0.30}
    sim, cfg, mesh0 = S.build(k, frames)
    Hf, out = engine_run(sim, device="cpu")
    hist = Hf.level("vertex")._mesh.get("hist")
    posf = out["sets"]["vertex"]["pos"]
    chemf = out["sets"]["cell"]["state"]["chem"]
    T = posf.shape[0]

    rows = []
    for t in range(0, T, every):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        pt = posf[t][:mt["Nv"]].astype(np.float64)
        act = chemf[t][:mt["nF"], 0]
        f = mesh_faults(pt, mt)
        cen, rad, live = _cell_centroids(pt, mt)
        cc = cell_census(pt, mt, act)
        td = tube_diameter(pt, mt)
        rows.append(dict(frame=t, cells=int(mt["nF"]),
                         folded=int(f["folded"].sum()), sliver=int(f["sliver"].sum()),
                         broken=int(f["broken"].sum()), hollow=int(f["hollow"].sum()),
                         n_tip=int(cc["n_tip"]),
                         elong=protrusion_ratio(rad[live]),
                         tube_len=float(td["tube_len"])))

    print(f"\n{'frame':>6}{'cells':>7}{'n_tip':>7}{'tube_len':>9}{'elong':>7}"
          f"{'folded':>8}{'sliver':>8}{'broken':>8}{'blend':>7}")
    for r in rows:
        print(f"{r['frame']:6d}{r['cells']:7d}{r['n_tip']:7d}{r['tube_len']:9.2f}{r['elong']:7.2f}"
              f"{r['folded']:8d}{r['sliver']:8d}{r['broken']:8d}{r['hollow']:7d}")

    tip = np.array([r["n_tip"] for r in rows], float)
    print("\ncorrelation of each failure mode with the TIP-CELL COUNT:")
    for m in ("folded", "sliver", "broken", "hollow"):
        y = np.array([r[m] for r in rows], float)
        c = np.corrcoef(y, tip)[0, 1] if y.std() > 0 else float("nan")
        tot = int(y[-1])
        print(f"  {m:8} corr {c:+.3f}   final count {tot:5d}"
              + ("   <-- drives the blend" if m != "hollow" and c > 0.85 else "")
              + ("   (all zero -- no real damage)" if y.max() == 0 else ""))
    return rows


if __name__ == "__main__":
    main(frames=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
