#!/usr/bin/env python
"""How elongated a tissue is, and along which axis -- from its trajectory alone. Experiment 5's ruler.

    PYTHONPATH=src python tools/elongation.py tissue/exp05_corset_a2_s1 [...] [--json out.jsonl]

Per sampled frame, over the live vertices:

    principal axes   eigenvectors of the vertex scatter (the gyration tensor) about their centroid
    AR (extent)      the tissue's LENGTH along the longest principal axis over its mean WIDTH along
                     the other two (max - min of the projections): a sphere reads 1.0, a body twice
                     as long as it is wide reads 2.0 -- the paper's "aspect ratio" (Crest et al.
                     2017, length over width of the follicle)
    angle            degrees between the longest principal axis and the axis the spec's
                     `junction_myosin` declares (0 = elongated along it, 90 = across it); `null`
                     when the spec declares none

Reported: the series every `--every` frames, the last frame's values, and `AR_rise` = AR at the end
minus AR at the settle frame, so a shape the seed already had is not counted as elongation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _axis_of(run_dir):
    try:
        import yaml
        s = yaml.safe_load(open(os.path.join(run_dir, "spec.yaml")))
        for o in s.get("operators") or []:
            if o.get("op") == "junction_myosin" and "axis" in o:
                a = np.asarray(o["axis"], float)
                return a / max(np.linalg.norm(a), 1e-12)
    except Exception:                                            # noqa: BLE001
        pass
    return None


def shape(P):
    """(AR_extent, AR_gyration, longest-axis unit vector) of a point cloud."""
    c = P.mean(0)
    X = P - c
    w, V = np.linalg.eigh(X.T @ X / len(X))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    ext = [np.ptp(X @ V[:, k]) for k in range(3)]
    ar_ext = ext[0] / max(0.5 * (ext[1] + ext[2]), 1e-12)
    ar_gyr = float(np.sqrt(w[0] / max(0.5 * (w[1] + w[2]), 1e-12)))
    return float(ar_ext), ar_gyr, V[:, 0]


def measure(spec, every=50):
    from plexus.paths import graphs_data_path
    from size_report import LazyNpz
    group, name = spec.split("/", 1)
    d = graphs_data_path(group, name)
    z = LazyNpz(os.path.join(d, "trajectory.npz"))
    pos, occ, Nv = z["vertex__pos"], (z["vertex__occ"] if "vertex__occ" in z.files else None), z["vertex__mesh_Nv"]
    T = len(Nv)
    ref = int(z["vertex__mesh_scalar_ref_frame"][0]) if "vertex__mesh_scalar_ref_frame" in z.files else 0
    ax = _axis_of(d)
    rows = []
    for t in sorted(set(list(range(min(ref, T - 1), T, every)) + [T - 1])):
        n = int(Nv[t])
        P = np.asarray(pos[t], float)[:n]
        if occ is not None:
            P = P[np.asarray(occ[t])[:n] > 0]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 10:
            continue
        ar, arg, e1 = shape(P)
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(e1 @ ax)))))) if ax is not None else None
        cells = int((np.asarray(z["cell__occ"][t]) > 0).sum()) if "cell__occ" in z.files else None
        rows.append(dict(t=int(t), AR=round(ar, 4), AR_gyr=round(arg, 4),
                         angle_deg=(round(ang, 2) if ang is not None else None), cells=cells))
    first, last = rows[0], rows[-1]
    # THE LATE WINDOW, not the last frame: a division burst in the last 50 frames moved the end AR by
    # 0.04-0.07 and the axis by up to 45 deg on some runs (exp 5 round 4, the judge's check), so the
    # scored shape is the mean over the samples in the last 10 % of the run.
    late = [r for r in rows if r["t"] >= 0.9 * (T - 1)] or [last]
    ang_late = [r["angle_deg"] for r in late if r["angle_deg"] is not None]
    lateAR = float(np.mean([r["AR"] for r in late]))
    return dict(spec=spec, frames=T, settle=ref, AR_late=round(lateAR, 4),
                AR_late_rise=round(lateAR - first["AR"], 4),
                angle_late_deg=(round(float(np.mean(ang_late)), 2) if ang_late else None),
                AR_end=last["AR"], AR_start=first["AR"],
                AR_rise=round(last["AR"] - first["AR"], 4), angle_end_deg=last["angle_deg"],
                cells=[first["cells"], last["cells"]],
                axis=(ax.tolist() if ax is not None else None), series=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    for sp in a.specs:
        r = measure(sp, a.every)
        print(f"{sp:<36} AR {r['AR_start']:.3f} -> {r['AR_end']:.3f} (rise {r['AR_rise']:+.3f})  "
              f"axis angle {r['angle_end_deg']}  cells {r['cells']}")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
