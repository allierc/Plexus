#!/usr/bin/env python
"""Shared tube-quality analysis: per-FRAME metrics (single end-frame numbers hid a transient hollowing the
movie clearly showed). For each frame: hollow-cell COUNT + fraction, cell-size CV (area & volume), and the
average TUBE DIAMETER. Tubes are found by clustering protruding cells (radius > 1.3x body median) by
direction; a tube's diameter = 2x median perpendicular distance of its cells to the cluster axis
(Okuda: diameter is the control target, ~chi^1/4). Emits metrics.json (series) + metrics.png (vs frame)."""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from tyssue_ops3d import face_geometry_3d
from tyssue_diag import hollow_flags
from tyssue_topology_ops3d import rings_from_flat_3d

_COS = np.cos(np.deg2rad(24.0))   # tubes merge cells within 24 deg of the cluster axis


def _cell_centroids(pt, mt):
    rings = rings_from_flat_3d(np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"])
    cen = np.array([pt[r].mean(0) if (r is not None and len(r)) else [0.0, 0.0, 0.0] for r in rings])
    return cen, np.linalg.norm(cen, axis=1)


def tube_diameter(pt, mt, prot_frac=1.3):
    """Average tube diameter + tube count. Cluster protruding cells (r>1.3x body median) by direction,
    diameter = 2x median perpendicular distance to the tube axis; also return protrusion + tube length."""
    cen, rad = _cell_centroids(pt, mt); good = rad > 1e-9
    if good.sum() < 8:
        return dict(tube_diam=0.0, n_tubes=0, tube_len=0.0)
    rbody = float(np.median(rad[good]))
    prot = np.where(good & (rad > prot_frac * rbody))[0]
    if prot.size < 5:
        return dict(tube_diam=0.0, n_tubes=0, tube_len=0.0)
    clusters = []                                            # greedy angular clustering into tubes
    for i in prot:
        di = cen[i] / rad[i]
        for c in clusters:
            if float(di @ c["dir"]) > _COS:
                c["idx"].append(i); m = cen[c["idx"]].mean(0); c["dir"] = m / (np.linalg.norm(m) + 1e-12); break
        else:
            clusters.append({"dir": di.copy(), "idx": [i]})
    diams, lens = [], []
    for c in clusters:
        if len(c["idx"]) < 4:
            continue
        ax = c["dir"]; P = cen[c["idx"]]
        perp = P - np.outer(P @ ax, ax)                      # component perpendicular to the tube axis
        diams.append(2.0 * float(np.median(np.linalg.norm(perp, axis=1))))
        lens.append(float((P @ ax).max() - rbody))
    return dict(tube_diam=round(float(np.median(diams)), 3) if diams else 0.0,
                n_tubes=len(diams), tube_len=round(float(np.median(lens)), 3) if lens else 0.0)


def frame_metrics(pt, mt, act=None):
    """All per-frame tube-quality metrics in one dict. `act` (per-cell activator) adds red_frac -- the
    fraction of ACTIVATED cells (top half of the activator range): LOW == localised spots (distinct
    tubes), HIGH == the activator has spread over the shell (one fat lumpy lobe, not tubes)."""
    es, et, ef, nF = (np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"])
    area, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
    area, vf = area.numpy(), vf.numpy()
    a = area[area > 1e-9]; v = np.abs(vf[np.abs(vf) > 1e-9])
    hollow, _, hst = hollow_flags(pt, mt)
    m = dict(cells=int(nF), hollow_n=int(hst["n"]), hollow_frac=round(float(hst["frac"]), 4),
             area_cv=round(float(a.std() / (a.mean() + 1e-9)), 3) if a.size else 0.0,
             vol_cv=round(float(v.std() / (v.mean() + 1e-9)), 3) if v.size else 0.0)
    _, rad = _cell_centroids(pt, mt); rad = rad[rad > 1e-9]
    m["protr"] = round(float(np.percentile(rad, 95) / (np.median(rad) + 1e-9)), 3) if rad.size else 1.0
    if act is not None and len(act):
        act = np.asarray(act, float); thr = act.min() + 0.5 * (act.max() - act.min())
        m["red_frac"] = round(float((act > thr).mean()), 3)
    m.update(tube_diameter(pt, mt))
    return m


def analyze(frames, OUT):
    """frames = list of (frame_index, pt, mt); compute the series, save metrics.json + metrics.png, and
    return a summary with peak hollow / peak size-CV / final tube diameter."""
    series = []
    for fr in frames:
        (t, pt, mt), act = fr[:3], (fr[3] if len(fr) > 3 else None)
        r = frame_metrics(pt, mt, act); r["frame"] = int(t); series.append(r)
    def col(k): return np.array([s[k] for s in series], float)
    fr = col("frame")
    summ = dict(hollow_n_peak=int(col("hollow_n").max()), hollow_frac_peak=round(float(col("hollow_frac").max()), 4),
                hollow_n_final=int(series[-1]["hollow_n"]), area_cv_peak=round(float(col("area_cv").max()), 3),
                area_cv_final=series[-1]["area_cv"], vol_cv_final=series[-1]["vol_cv"],
                tube_diam_final=series[-1]["tube_diam"], n_tubes_final=series[-1]["n_tubes"],
                tube_len_final=series[-1]["tube_len"], protr_final=series[-1]["protr"],
                red_frac_final=series[-1].get("red_frac", 0.0))
    json.dump({"summary": summ, "series": series}, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.4)); fig.patch.set_facecolor("white")
    ax[0].plot(fr, col("hollow_n"), "-", color="crimson"); ax[0].set_title("hollow cell count"); ax[0].set_xlabel("frame")
    ax[1].plot(fr, col("area_cv"), "-", color="C0", label="area"); ax[1].plot(fr, col("vol_cv"), "-", color="C2", label="vol")
    ax[1].set_title("cell-size CV"); ax[1].set_xlabel("frame"); ax[1].legend(fontsize=8)
    ax[2].plot(fr, col("tube_diam"), "-", color="darkorange"); ax[2].set_title("avg tube diameter"); ax[2].set_xlabel("frame")
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "metrics.png"), dpi=110); plt.close(fig)
    return summ
