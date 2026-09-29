"""Membrane holes over time: count, sizes, open and detached fractions -- `exp11.holes` on one or more runs.

    python tools/hole_track.py tissue/exp11_w2h_hole100_s1 [more runs] [--every 5] [--cover 1.5] [--png out.png]

Prints one row per sampled frame and writes a three-panel figure (white, labels above the panels):
(a) the number of holes, (b) the open fraction of the basal surface and the largest hole's area, (c) every
hole's area at every sampled row -- the size distribution over time, log scale. Time in hours from the
spec's `general.units.time_s` (else frames). The measure itself is `tools/exp_measures/exp11.py::holes`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_measures.common import open_run          # noqa: E402
from exp_measures import exp11 as M               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--cover", type=float, default=1.5)
    ap.add_argument("--band", type=float, default=0.3)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    res = []
    for r in a.runs:
        T = open_run(r if r.startswith("tissue/") else "tissue/" + r)
        o = M.holes(T, every=a.every, cover=a.cover, band=a.band)
        if not o.get("available"):
            print(r, o.get("why")); continue
        ts = (((T.spec.get("general") or {}).get("units") or {}).get("time_s"))
        x = np.asarray(o["row"], float) * (float(ts) / 3600.0 if ts else 1.0)
        res.append((os.path.basename(r), x, "h" if ts else "frame", o))
        print(f"{r}  ({o['unit']}; spacing = median nearest-node distance, um)")
        print(f"  {'row':>5} {'t':>6} {'holes':>6} {'small':>6} {'open':>7} {'detach':>7} {'max':>9} {'median':>8} {'total':>9} {'spacing':>8}")
        for i, row in enumerate(o["row"]):
            print(f"  {row:>5} {x[i]:>6.1f} {o['n_holes'][i]:>6} {o['n_small'][i]:>6} {o['frac_open'][i]:>7.4f} {o['frac_detached'][i]:>7.4f} "
                  f"{o['area_max'][i]:>9.1f} {o['area_median'][i]:>8.1f} {o['area_total'][i]:>9.1f} {o['spacing'][i]:>8.2f}")
        tr = o.get("tracks") or []
        life = [t["last"] - t["first"] for t in tr]
        print(f"  tracks {len(tr)}; lifetime (rows) median {np.median(life) if life else 0:.0f}, max {max(life) if life else 0}")
    if not res:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    cols = ["#e63946", "#1d6fb8", "#2a9d8f", "#6c757d", "#f4a261"]
    for k, (name, x, xu, o) in enumerate(res):
        c = cols[k % len(cols)] if len(res) > 1 else "#333333"
        ax[0].plot(x, o["n_holes"], color=c, lw=1.6, label=name)
        ax[0].plot(x, o["n_small"], color=c, lw=1.0, ls=":")
        ax[1].plot(x, np.asarray(o["frac_open"]) * 100, color=c, lw=1.6)
        ax[1].plot(x, np.asarray(o["frac_detached"]) * 100, color=c, lw=1.0, ls="--")
        for xi, A in zip(x, o["areas"]):
            if A:
                ax[2].scatter(np.full(len(A), xi), A, s=6, color=c, alpha=0.5, lw=0)
    unit = res[0][3]["unit"].replace("um^2", "µm²")
    lab = ["holes >= 1 node patch (solid), smaller gaps (dotted), count", "basal surface open (solid) / detached (dashed), %", f"each hole's area, {unit}"]
    for i, t in enumerate(lab):
        ax[i].set_title(t, loc="left", fontsize=10, fontweight="normal")
        ax[i].set_xlabel(f"time ({res[0][2]})")
        for sp in ("top", "right"):
            ax[i].spines[sp].set_visible(False)
    ax[2].set_yscale("log")
    if len(res) > 1:
        ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = a.png or os.path.join("/tmp", f"holes_{res[0][0]}.png")
    fig.savefig(out, dpi=130, facecolor="white")
    print("figure:", out)


if __name__ == "__main__":
    main()
