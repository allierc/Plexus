#!/usr/bin/env python
"""Does a plaque STAY, and where -- turnover and tangential drift, per frame.

    python plaque_identity.py 07h_bind_cull

WHY THIS EXISTS. In `movie_vtk.mp4` the red links flicker: they seem to appear and vanish rather than
to sit on the tissue and be carried by it. Three different things produce that appearance and only
one of them is the model:

  1  THE RENDERER SUBSAMPLES.  `vtk_ecm` draws at most 800 links, at a stride of
     `ceil(n / 800)`. The set grows from 2,389 to 72,606 over the run, so the stride runs 3 -> 91 and
     the DRAWN SUBSET is a different one-in-N sample at every frame. Nothing has to move for that to
     flicker, and it is measured here as `drawn_stride`, not as adhesion.
  2  THE SET TURNS OVER.       plaques are seeded when a cell divides and culled when they let go.
     Measured here as `new` and `lost`, as a fraction of the set.
  3  THE PLAQUES MOVE.         and this is expected: a plaque is anchored to a cell, and the cells
     both grow outward and rearrange tangentially. Radial motion is growth and says nothing about
     adhesion, so it is divided out: every attachment point is projected onto the unit sphere about
     the tissue centre and the ANGLE it turns through between kept frames is what is reported. A
     plaque that is merely carried outward by a growing spheroid has an angle of zero.

IDENTITY BY INDEX, and it is checked rather than assumed. The contact arrays are append-only except
when a cull compacts them, so entry i is the same plaque from one frame to the next only while no
cull has fired in between. The cell each plaque belongs to is stable (G76), so the overlap's
`ct_face` is compared frame to frame and the fraction that agrees is REPORTED: at 1.0 the indexing
is sound and the drift numbers mean what they say; below it, a cull reshuffled the array and the
frame is marked, because a drift measured across a reshuffle is a permutation, not a motion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))
DRAW_CAP = 800.0                   # vtk_ecm's own cap on how many links it draws


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30)


def measure(run):
    d = os.path.join(LOG, run)
    z = np.load(os.path.join(d, "bm_frames.npz"))
    n_kept = int(z["n_kept"])
    c, sc = np.asarray(z["centre"], float), float(z["scale"])
    sp = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    um = float(sp["sets"]["epithelium"]["box_scale"]) * float(sp["general"]["units"]["length_um"])
    S = {k: [] for k in ("t", "n", "persist", "new", "lost", "same_node", "ang_med", "ang_p95",
                         "ang_cum_med", "drift_um_med", "drawn_stride", "id_ok")}
    prev = None
    born = {}                       # index -> the unit direction it was first seen at
    for i in range(n_kept):
        t = int(z[f"t{i}"])
        p = (np.asarray(z[f"p{i}"], float) - c) / sc     # the store is in box units
        nd = np.asarray(z[f"n{i}"])
        cf = np.asarray(z[f"cf{i}"])
        u = unit(p)
        n = len(nd)
        S["t"].append(t); S["n"].append(n)
        S["drawn_stride"].append(int(np.ceil(max(n, 1) / DRAW_CAP)))
        if prev is None:
            S["persist"].append(float("nan")); S["new"].append(float("nan"))
            S["lost"].append(float("nan")); S["same_node"].append(float("nan"))
            for k in ("ang_med", "ang_p95", "ang_cum_med", "drift_um_med"):
                S[k].append(float("nan"))
            S["id_ok"].append(1.0)
        else:
            pu, pnd, pcf, pn = prev
            k = min(n, pn)
            ok = float((cf[:k] == pcf[:k]).mean()) if k else 1.0
            S["id_ok"].append(ok)
            S["persist"].append(k / max(n, 1))
            S["new"].append(max(n - pn, 0) / max(n, 1))
            S["lost"].append(max(pn - n, 0) / max(pn, 1))
            S["same_node"].append(float((nd[:k] == pnd[:k]).mean()) if k else float("nan"))
            cosang = np.clip((u[:k] * pu[:k]).sum(1), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            S["ang_med"].append(float(np.median(ang)))
            S["ang_p95"].append(float(np.percentile(ang, 95)))
            # AND THE SAME THING IN MICRONS on the tissue's own surface, which is what a reader of
            # the movie is actually judging: the arc the plaque swept at the radius it sits at.
            r_um = np.linalg.norm(p[:k], axis=1) * um
            S["drift_um_med"].append(float(np.median(np.radians(ang) * r_um)))
            cu = np.array([born.get(j, u[j]) for j in range(k)])
            cc = np.clip((u[:k] * cu).sum(1), -1.0, 1.0)
            S["ang_cum_med"].append(float(np.median(np.degrees(np.arccos(cc)))))
        for j in range(len(u)):
            born.setdefault(j, u[j])
        prev = (u, nd, cf, n)
    return d, S


def report(run):
    d, S = measure(run)
    a = {k: np.asarray(v, float) for k, v in S.items()}
    fin = np.isfinite(a["ang_med"])
    res = dict(
        run=run, frames=len(S["t"]), n_first=int(a["n"][0]), n_last=int(a["n"][-1]),
        identity_by_index_ok=float(np.nanmin(a["id_ok"])),
        turnover=dict(
            name="what fraction of the set at each frame was there at the previous kept frame",
            persist_mean=float(np.nanmean(a["persist"])), persist_min=float(np.nanmin(a["persist"])),
            new_mean=float(np.nanmean(a["new"])), new_max=float(np.nanmax(a["new"])),
            lost_max=float(np.nanmax(a["lost"]))),
        anchoring=dict(
            name="of the plaques present in both frames, the fraction still on the SAME sheet node",
            same_node_mean=float(np.nanmean(a["same_node"])),
            same_node_min=float(np.nanmin(a["same_node"]))),
        drift=dict(
            name="tangential motion with growth divided out: the angle a plaque's attachment point "
                 "turns through about the tissue centre between kept frames",
            deg_per_interval_median=float(np.nanmedian(a["ang_med"][fin])),
            deg_per_interval_p95=float(np.nanmax(a["ang_p95"][fin])),
            um_per_interval_median=float(np.nanmedian(a["drift_um_med"][fin])),
            deg_from_birth_final=float(a["ang_cum_med"][-1])),
        rendering=dict(
            name="vtk_ecm draws at most 800 links; the stride is the reason the drawn subset "
                 "changes composition every frame",
            stride_first=int(a["drawn_stride"][0]), stride_last=int(a["drawn_stride"][-1])),
        series={k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in a.items()})
    json.dump(res, open(os.path.join(d, "plaque_identity.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    t = a["t"]
    ax[0].plot(t, 100 * a["persist"], color="0.15", label="present at the previous frame")
    ax[0].plot(t, 100 * a["new"], color="crimson", label="newly seeded")
    ax[0].plot(t, 100 * a["lost"], color="tab:blue", label="culled")
    ax[0].set_ylabel("% of the set"); ax[0].set_ylim(-2, 102); ax[0].legend(fontsize=7, loc="center right")
    ax[1].plot(t, a["ang_med"], color="0.15", label="median")
    ax[1].plot(t, a["ang_p95"], color="crimson", lw=0.8, label="p95")
    ax[1].plot(t, a["ang_cum_med"], color="tab:green", label="median, from birth")
    ax[1].set_ylabel("tangential drift (deg about the centre)"); ax[1].legend(fontsize=7)
    ax[2].plot(t, 100 * a["same_node"], color="0.15")
    ax[2].set_ylabel("% still on the same sheet node")
    for k, x in enumerate(ax):
        x.set_xlabel("frame")
        x.text(0.02, 0.95, "abc"[k], transform=x.transAxes, fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "plaque_identity.png"), dpi=140)
    r, g, h = res["turnover"], res["drift"], res["rendering"]
    print(f"[id] {run}: identity by index sound on {100*res['identity_by_index_ok']:.1f}% of the "
          f"overlap at the worst frame", flush=True)
    print(f"[id] {run}: {100*r['persist_mean']:.1f}% of the set was there the previous kept frame "
          f"(worst {100*r['persist_min']:.1f}%), {100*r['new_mean']:.1f}% newly seeded "
          f"(max {100*r['new_max']:.1f}%), at most {100*r['lost_max']:.1f}% culled in one interval",
          flush=True)
    print(f"[id] {run}: they DO move -- median {g['deg_per_interval_median']:.2f} deg per interval "
          f"({g['um_per_interval_median']:.2f} um of arc), {g['deg_from_birth_final']:.1f} deg from "
          f"birth by the end; {100*res['anchoring']['same_node_mean']:.1f}% keep the same sheet node",
          flush=True)
    print(f"[id] {run}: the renderer's stride went {h['stride_first']} -> {h['stride_last']}, so the "
          f"drawn 800 are a different sample every frame -> {d}/plaque_identity.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    for r in ap.parse_args().runs:
        report(r)


if __name__ == "__main__":
    main()
