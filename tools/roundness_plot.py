#!/usr/bin/env python
"""Cell roundness and mean cell area over time, for several vertex-model runs on one figure.

    PYTHONPATH=src python tools/roundness_plot.py cell/exp02_v5 cell/exp02_v6 ... \
        --out experiments/figs/exp02_roundness.png

Two panels, one quantity each (never two scales on one axis):

  a  the mean shape index q = perimeter / sqrt(area) over the live cells, per frame, with its
     cell-to-cell SD as a band. Dimensionless; lower is rounder. Reference lines at the regular
     hexagon (3.722) and the regular pentagon (3.812).
  b  the mean cell area in um^2 (from the spec's `general.units.length_um`), per frame.

Perimeters are not a recorded block, so they are rebuilt from the mesh table of every frame -- the
same computation `tools/judge.py --measure area_smooth` uses for its last frame. The numbers behind
the figure are written beside it as a CSV, so the plot has a table view.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Categorical slots 1-4 of the dataviz reference palette, in fixed order; validated on white
# (adjacent CVD dE >= 9.1). Slots 3 and 4 sit under 3:1 contrast, so every line is labelled directly.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#222222", "#8a8a8a"


def series(run_dir: str) -> dict:
    """{frame, q_mean, q_sd, area_mean} per frame, over the live cells."""
    import yaml
    z = np.load(os.path.join(run_dir, "trajectory.npz"))
    L = 1.0
    try:
        u = (yaml.safe_load(open(os.path.join(run_dir, "spec.yaml"))).get("general") or {}).get("units") or {}
        L = float(u.get("length_um") or 1.0)
    except Exception:                                            # noqa: BLE001
        pass
    occ = np.asarray(z["cell__occ"]) > 0
    area = np.asarray(z["cell__area"], float)[:, :, 0] * L * L
    off = np.asarray(z["vertex__mesh_offsets"])
    es, et, ef = z["vertex__mesh_E_srce"], z["vertex__mesh_E_trgt"], z["vertex__mesh_E_face"]
    pos = z["vertex__pos"]
    T, N = occ.shape
    out = {"frame": np.arange(T), "q_mean": np.full(T, np.nan), "q_sd": np.full(T, np.nan),
           "area_mean": np.full(T, np.nan)}
    for t in range(T):
        s_, e_, f_ = (np.asarray(v[off[t]:off[t + 1]]) for v in (es, et, ef))
        x = np.asarray(pos[t], float)
        ok = f_ >= 0
        per = np.bincount(f_[ok], weights=np.linalg.norm(x[e_[ok]] - x[s_[ok]], axis=1),
                          minlength=N)[:N] * L
        live = occ[t] & (area[t] > 0)
        q = per[live] / np.sqrt(area[t][live])
        out["q_mean"][t], out["q_sd"][t] = q.mean(), q.std()
        out["area_mean"][t] = area[t][live].mean()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="<group>/<name> of each run, as Plexus_Main takes it")
    ap.add_argument("--labels", nargs="*", default=None, help="one label per run (default: the name)")
    ap.add_argument("--area_band", nargs=2, type=float, default=None,
                    help="accepted mean-area range in um^2, drawn as a band in panel b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.runs) > len(COLOURS):
        raise SystemExit(f"at most {len(COLOURS)} runs on one figure; facet the rest")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path

    labels = a.labels or [r.split("/", 1)[1] for r in a.runs]
    data = [series(graphs_data_path(*r.split("/", 1))) for r in a.runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for ax, tag in ((ax1, "a"), (ax2, "b")):
        ax.set_facecolor("white")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=10)
        ax.text(0.0, 1.06, tag, transform=ax.transAxes, fontsize=14, color=INK, va="bottom")

    for ref, name in ((3.722, "regular hexagon 3.722"), (3.812, "regular pentagon 3.812")):
        ax1.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax1.text(1.0, ref, name, transform=ax1.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=9, color=MUTED)
    if a.area_band:
        ax2.axhspan(a.area_band[0], a.area_band[1], color="#eeeeee", zorder=0)
        ax2.text(1.0, a.area_band[1], f"accepted {a.area_band[0]:g}-{a.area_band[1]:g}",
                 transform=ax2.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color=MUTED)

    for d, lab, col in zip(data, labels, COLOURS):
        f = d["frame"]
        ax1.fill_between(f, d["q_mean"] - d["q_sd"], d["q_mean"] + d["q_sd"], color=col, alpha=0.07, lw=0)
        ax1.plot(f, d["q_mean"], color=col, lw=2, label=lab)
        ax2.plot(f, d["area_mean"], color=col, lw=2, label=lab)

    def _end_labels(ax, key, fmt):
        """Direct labels at the line ends, nudged apart so none overlaps: sorted by value, each
        pushed up to at least 6% of the axis span above the one below it."""
        ends = sorted(((d[key][-1], lab, d["frame"][-1]) for d, lab in zip(data, labels)))
        lo, hi = ax.get_ylim()
        gap, placed = 0.06 * (hi - lo), []
        for y, lab, x in ends:
            yl = max(y, placed[-1] + gap) if placed else y
            placed.append(yl)
            ax.annotate(f"{lab}  {fmt(y)}", (x, y), xytext=(x + 0.02 * x, yl), textcoords="data",
                        va="center", fontsize=9, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6) if abs(yl - y) > 1e-12 else None)
    _end_labels(ax1, "q_mean", lambda v: f"{v:.3f}")
    _end_labels(ax2, "area_mean", lambda v: f"{v:.2f}")

    ax1.set_xlabel("frame", color=INK)
    ax1.set_ylabel("shape index  perimeter / sqrt(area)", color=INK)
    ax2.set_xlabel("frame", color=INK)
    ax2.set_ylabel("mean cell area (um$^2$)", color=INK)
    for ax in (ax1, ax2):
        ax.set_xlim(0, max(len(d["frame"]) for d in data) - 1)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    fig.tight_layout(w_pad=6)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "frame", "shape_index_mean", "shape_index_sd", "area_mean_um2"])
        for d, lab in zip(data, labels):
            for i in range(len(d["frame"])):
                w.writerow([lab, int(d["frame"][i]), f"{d['q_mean'][i]:.5f}", f"{d['q_sd'][i]:.5f}",
                            f"{d['area_mean'][i]:.4f}"])
    print(f"[roundness] {a.out}")


if __name__ == "__main__":
    main()
