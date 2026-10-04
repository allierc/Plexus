#!/usr/bin/env python
"""exp20: THE PAPER'S FIG. 3d ON OUR RECORDINGS -- where gut-responsive cells also follow the visual stimulus or the swim.

    PYTHONPATH=src:tools python tools/exp20_integration.py [--recordings gutbrain_glucose_f1 ...] [--bins 5]

Chen 2026 Fig. 3d: for the gut-responsive cells of each region, the percent gain in a cell's model correlation when
motor (GM), visual (GV) or both (GMV) regressors are added to the gut-only model (G); hindbrain regions gain from motor
and little from visual (GV/GM 0.26-0.35 in PBN, medial idMO, DVC), the midbrain (OT) from both (GV/GM 1.18).

Here, on each fish's ORIGINAL recording (swim kept), its gut-responsive cells (tools/gutbrain_baselines.py), the
paper's smoothness-regularised distributed-lag regression (the same closed form, gutbrain_baselines.fused_ridge;
kernels 55 s gut and all-UV, 10 s motor, 5 s visual; fitted on every frame, as the paper's Fig. 3) for four models:
    G    gut-pulse train + all-UV train
    GM   G + the swim power (left; the live channel)
    GV   G + the grating speed
    GMV  G + both
improvement_X = (r_X - r_G) / r_G x 100 per cell, then the mean over the cells of each BIN along the head-to-tail axis
(pos_view x, head = 0, tail = 1; no atlas registration exists, so the axis stands in for the regions: the paper's OT
lies rostral of its PBN, which lies rostral of its medial idMO / DVC). A fish with no live swim channel gets no GM.
Writes experiments/exp20_gutbrain_graphcast/data/integration.json and presentation/figs/integration.png.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
DATA = os.path.join(EXP, "data")
CTRL_SITE = 1


def one(rec_name, bins, device):
    import gutbrain_baselines as B
    from plexus.tasks import trace_recording as TR
    rec = TR.load(rec_name)
    prov = json.load(open(os.path.join(os.environ["GNN_OUTPUT_ROOT"], "graphs_data", "zebrafish",
                                       f"{rec_name}_recording.json")))
    cz = np.load(os.path.join(DATA, f"baselines_{rec_name}_cells.npz"))
    resp = np.where(cz["responsive"])[0]
    if len(resp) < 20:
        return {"recording": rec_name, "cells": int(len(resp)), "skipped": "fewer than 20 gut-responsive cells"}
    X = torch.as_tensor(rec["dff"][:, resp], device=device)
    S, tr = rec["stimulus"], rec["trials"]
    T = X.shape[0]
    dt = float(np.median(np.diff(rec["t_s"])))
    g = np.zeros(T); a = np.zeros(T)
    for f, site in zip(tr[:, 0].astype(int), tr[:, 2].astype(int)):
        a[f] = 1.0
        if site != CTRL_SITE:
            g[f] = 1.0
    K = {k: int(round(v / dt)) for k, v in B.KERNEL_S.items()}
    fit = np.ones(T, bool)
    rows = torch.as_tensor(np.arange(T), device=device)
    swim_live = bool(prov.get("swim_live", [False])[0])
    models = {"G": ([g, a], [K["gut"], K["ctrl"]]), "GV": ([g, a, S[:, 3]], [K["gut"], K["ctrl"], K["visual"]])}
    if swim_live:
        models["GM"] = ([g, a, S[:, 4]], [K["gut"], K["ctrl"], K["motor"]])
        models["GMV"] = ([g, a, S[:, 3], S[:, 4]], [K["gut"], K["ctrl"], K["visual"], K["motor"]])
    r = {}
    for k, (regs, orders) in models.items():
        p, _ = B.fused_ridge(X, regs, orders, fit, device)
        r[k] = B.corr_cols(p, X, rows).cpu().numpy()
    Pv = np.asarray(rec["pos_view"])[resp]
    allx = np.asarray(rec["pos_view"])[:, 0]
    u = (Pv[:, 0] - allx.min()) / np.ptp(allx)                      # 0 = head, 1 = tail
    edges = np.linspace(0, 1, bins + 1)
    ok = r["G"] > 0.05                                               # the improvement is a ratio over r_G
    out = {"recording": rec_name, "cells": int(len(resp)), "cells_used": int(ok.sum()), "swim_live": swim_live,
           "bins": edges.tolist(), "per_bin": []}
    for i in range(bins):
        m = ok & (u >= edges[i]) & (u < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        row = {"bin": i, "n": int(m.sum())}
        for k in ("GM", "GV", "GMV"):
            if k in r and m.sum() >= 10:
                row[k] = float(np.mean((r[k][m] - r["G"][m]) / r["G"][m] * 100))
        if "GM" in row and "GV" in row and row["GM"] > 0:
            row["GV_over_GM"] = row["GV"] / row["GM"]
        out["per_bin"].append(row)
    for k in ("GM", "GV", "GMV"):
        if k in r:
            out[f"all_{k}"] = float(np.mean((r[k][ok] - r["G"][ok]) / r["G"][ok] * 100))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", nargs="*", default=[f"gutbrain_glucose_f{k}" for k in range(1, 7)]
                    + [f"gutbrain_glutamate_f{k}" for k in range(1, 6)])
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    res = []
    for rn in a.recordings:
        try:
            o = one(rn, a.bins, a.device)
        except FileNotFoundError as e:
            o = {"recording": rn, "skipped": str(e)[:120]}
        res.append(o)
        line = ", ".join(f"bin {b['bin']}: GM {b.get('GM', float('nan')):.0f} GV {b.get('GV', float('nan')):.0f}"
                         for b in o.get("per_bin", []))
        print(f"[integration] {rn}: {o.get('cells_used', 0)} cells; {line or o.get('skipped')}", flush=True)
    json.dump({"what": __doc__.split("\n")[0], "bins": a.bins, "fish": res}, open(os.path.join(DATA, "integration.json"), "w"),
              indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="black")
    mids = 0.5 * (np.array(res[0]["bins"][:-1]) + np.array(res[0]["bins"][1:])) if res and "bins" in res[0] else None
    for a_, k, lab in zip(ax, ("GM", "GV", "GV_over_GM"), ("% gain from the swim (GM)", "% gain from the grating (GV)",
                                                            "GV / GM")):
        a_.set_facecolor("black")
        for sp in a_.spines.values():
            sp.set_color("0.6")
        a_.tick_params(colors="0.8")
        Y = []
        for o in res:
            if "per_bin" not in o:
                continue
            y = [b.get(k, np.nan) for b in o["per_bin"]]
            col = "#ff8a65" if "glucose" in o["recording"] else "#4fc3f7"
            a_.plot(mids, y, color=col, lw=0.8, alpha=0.6)
            Y.append(y)
        if Y:
            a_.plot(mids, np.nanmedian(np.array(Y, float), 0), color="white", lw=2.2, label="median over fish")
        a_.set_xlabel("position along the brain: head (0) to tail (1)", color="0.9")
        a_.set_title(lab, color="white")
        if k == "GV_over_GM":
            a_.axhline(1.18, color="#ffd54f", ls="--", lw=1, label="paper OT (midbrain) 1.18")
            a_.axhline(0.30, color="#e57373", ls="--", lw=1, label="paper PBN / idMO / DVC ~0.30")
            a_.legend(frameon=False, labelcolor="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "presentation", "figs", "integration.png"), dpi=170, facecolor="black")
    print("[integration] -> data/integration.json, figs/integration.png")


if __name__ == "__main__":
    main()
