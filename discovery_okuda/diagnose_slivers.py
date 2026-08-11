#!/usr/bin/env python
"""Are the degenerate cells transient division churn, or the same cells stuck?

Cedric, 11 August, on \\op{r013_05}: *"there are some defect cells that appear late... by eye there
are persistent and very thin, elongated."*

WHY PER-CELL AND NOT PER-FRAME. The recorded series says `sliver_n` climbs 0 -> 123 and
`folded_n` 0 -> 121 over the run, and that number is the same whether 123 cells are permanently
degenerate or 123 different cells are briefly thin on the frame after they divided. Those are
opposite findings: the first is a mesh that is failing, the second is what division looks like.
`sliver_n`'s own definition says so --- "usually one that just divided, and not on its own a fault".

The distinction is only visible per CELL, and it is measurable here for a specific reason: this run
has `n_apop` 0. Death renumbers faces through `keep`, so on a run with deaths a cell index is not a
cell identity; with no deaths, `divide_3d` only ever APPENDS, so index i is the same cell for the
whole run and persistence can simply be counted.

WHAT IS MEASURED, per recorded frame, per cell:

    shape index   P / sqrt(A), the dimensionless measure the mechanics itself minimises towards
                  `p0`. A regular hexagon is 3.72; a sliver runs away upwards.
    aspect        the cell ring's longest axis over its shortest, by the eigenvalues of its own
                  covariance. This is the "very thin, elongated" the eye reports, stated as a number.
    radius        distance of the cell centroid from the body centre, normalised by the median ---
                  so a value near 1 is on the body and a large value is out on a tube.

    python diagnose_slivers.py [run]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "log", "okuda")
# 3.72 is the regular hexagon; the campaign's own P8 premise floors the shape index at 3.545 (the
# circle). Anything past 5 is not a cell shape the vertex model is meant to hold.
SLIVER_SI = 5.0
THIN_ASPECT = 4.0


def rings_of(mt):
    from tyssue_topology_ops3d import rings_from_flat_3d
    return rings_from_flat_3d(np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]),
                              np.asarray(mt["E_face"]), int(mt["nF"]))


def per_cell(pos, mt):
    """Shape index, aspect and normalised radius for every live cell of one frame."""
    rings = rings_of(mt)
    nF = int(mt["nF"])
    si = np.full(nF, np.nan); asp = np.full(nF, np.nan); rad = np.full(nF, np.nan)
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        p = pos[np.asarray(r, int)]
        c = p.mean(0)
        e = np.roll(p, -1, axis=0) - p
        per = float(np.linalg.norm(e, axis=1).sum())
        # polygon area in 3D by the fan from the centroid
        a = 0.5 * float(np.linalg.norm(np.cross(p - c, np.roll(p, -1, axis=0) - c), axis=1).sum())
        if a > 1e-12:
            si[f] = per / np.sqrt(a)
        w = np.linalg.eigvalsh(np.cov((p - c).T) + 1e-15 * np.eye(3))[::-1]
        asp[f] = float(np.sqrt(max(w[0], 0)) / (np.sqrt(max(w[1], 0)) + 1e-12))
        rad[f] = float(np.linalg.norm(c))
    med = np.nanmedian(rad)
    return si, asp, (rad / med if med > 1e-12 else rad)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "r013_05"
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "prototype", "Tyssue"))
    d = os.path.join(LOG, run)
    z = np.load(os.path.join(d, "traj.npz"), allow_pickle=True)
    nfr = len([k for k in z.files if k.startswith("pos_")])
    if json.load(open(os.path.join(d, "diag.json")))["summary"].get("n_apop"):
        print("WARNING: this run has deaths, so a cell index is not a cell identity -- "
              "persistence below is not trustworthy")

    frames = list(range(max(0, nfr - 12), nfr))          # the last quarter, where they appear
    seen = {}                                            # cell -> frames it was degenerate
    last = None
    for t in frames:
        pos = np.asarray(z[f"pos_{t}"], float)
        mt = z[f"mesh_{t}"].item() if hasattr(z[f"mesh_{t}"], "item") else z[f"mesh_{t}"]
        si, asp, rad = per_cell(pos, mt)
        bad = np.where((si > SLIVER_SI) | (asp > THIN_ASPECT))[0]
        for f in bad:
            seen.setdefault(int(f), []).append(t)
        last = (si, asp, rad, bad, int(mt["nF"]))
    si, asp, rad, bad, nF = last

    print(f"{run}: {nF} cells, last {len(frames)} recorded frames\n")
    print(f"  degenerate at the LAST frame (shape index > {SLIVER_SI} or aspect > {THIN_ASPECT}): "
          f"{len(bad)}  ({100*len(bad)/max(nF,1):.2f}% of cells)")
    if len(bad):
        print(f"    shape index  median {np.nanmedian(si[bad]):.2f}   max {np.nanmax(si[bad]):.2f}"
              f"      (healthy median {np.nanmedian(si):.2f})")
        print(f"    aspect       median {np.nanmedian(asp[bad]):.2f}   max {np.nanmax(asp[bad]):.2f}"
              f"      (healthy median {np.nanmedian(asp):.2f})")
        print(f"    radius/med   median {np.nanmedian(rad[bad]):.2f}"
              f"      (all cells {np.nanmedian(rad):.2f}) -- >1 means out on a tube")
    # PERSISTENCE is the whole question: a cell degenerate on one frame is division churn, a cell
    # degenerate on every frame it is measured is stuck.
    runs_len = {f: len(v) for f, v in seen.items()}
    if runs_len:
        import collections
        h = collections.Counter(runs_len.values())
        print(f"\n  of {len(seen)} cells degenerate at any point in those {len(frames)} frames:")
        for k in sorted(h):
            print(f"    degenerate on {k:>2} of {len(frames)} frames : {h[k]:>4} cells")
        persistent = sum(v for k, v in h.items() if k >= len(frames) - 1)
        print(f"\n  PERSISTENT (degenerate on all or all-but-one frame): {persistent} of {len(seen)}"
              f"  -> {'STUCK, not division churn' if persistent > len(seen) * 0.3 else 'mostly transient'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
