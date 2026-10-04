"""exp20: WHERE THE PAPER'S STATIONS SIT IN EACH FISH, inferred from its own gut-responsive cells (no atlas registration
exists for the deposit): the pathway DVC/AP -> PBN -> forebrain the flow maps are read against (Cedric, 2026-10-04: "put the
acronyms on the plot in yellow").

    from exp20_landmarks import landmarks; landmarks("gutbrain_glucose_f1")    # {label: [(x, y) um, ...]} in pos_view

The gut-responsive cells (tools/gutbrain_baselines.py) are clustered (DBSCAN, 15 um, >= 25 cells); with u the position
along the brain (head 0, tail 1; pos_view, head drawn left) and v the lateral offset from the midline (in brain widths):
    DVC/AP     the largest cluster with u > 0.70 and |v| < 0.12          (the paper's caudal midline: DVC, AP, medial idMO)
    PBN        clusters with 0.35 < u < 0.65 and 0.08 < |v| < 0.25       (one per side)
    nodose     clusters with u > 0.60 and |v| >= 0.25                     (lateral, outside the hindbrain; one per side)
    forebrain  u = 0.07 on the midline                                    (the head end; no gut cluster needed)
A station whose cluster is not found is left out. Cached in data/landmarks_<rec>.json. INFERRED, never measured: the
labels say so.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
DATA = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")


def landmarks(rec_name):
    base = rec_name.replace("_nosw", "")
    cache = os.path.join(DATA, f"landmarks_{base}.json")
    if os.path.exists(cache):
        return {k: [tuple(p) for p in v] for k, v in json.load(open(cache)).items()}
    from sklearn.cluster import DBSCAN
    from plexus.tasks import trace_recording as TR
    P = np.asarray(TR.load(base)["pos_view"], np.float64)
    resp = np.load(os.path.join(DATA, f"baselines_{base}_cells.npz"))["responsive"]
    out = {}
    x0, xs = P[:, 0].min(), np.ptp(P[:, 0])
    ymid, ys = np.median(P[:, 1]), np.ptp(P[:, 1])
    out["forebrain"] = [(x0 + 0.07 * xs, ymid)]
    if resp.sum() >= 25:
        Q = P[resp]
        lab = DBSCAN(eps=15, min_samples=25).fit_predict(Q)
        cl = []
        for k in set(lab) - {-1}:
            m = lab == k
            c = Q[m].mean(0)
            cl.append((int(m.sum()), (c[0] - x0) / xs, (c[1] - ymid) / ys, c))
        dvc = [c for c in cl if c[1] > 0.70 and abs(c[2]) < 0.12]
        if dvc:
            out["DVC/AP"] = [tuple(max(dvc)[3][:2])]
        for name, sel in (("PBN", lambda c: 0.35 < c[1] < 0.65 and 0.08 < abs(c[2]) < 0.25),
                          ("nodose", lambda c: c[1] > 0.60 and abs(c[2]) >= 0.25)):
            pts = []
            for side in (-1, 1):
                cs = [c for c in cl if sel(c) and np.sign(c[2]) == side]
                if cs:
                    pts.append(tuple(max(cs)[3][:2]))
            if pts:
                out[name] = pts
    json.dump({k: [list(map(float, p)) for p in v] for k, v in out.items()}, open(cache, "w"), indent=1)
    return out


def annotate(ax, lm, to_px, fontsize=10):
    """Yellow labels at the stations, `to_px` mapping (x, y) um to the axes' data coordinates."""
    for name, pts in lm.items():
        for (x, y) in pts:
            px, py = to_px(x, y)
            ax.plot(px, py, marker="o", ms=4, mfc="none", mec="#ffeb3b", mew=1.2)
            ax.annotate(name, (px, py), xytext=(6, 6), textcoords="offset points", color="#ffeb3b", fontsize=fontsize,
                        fontweight="bold")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(r, landmarks(r))
