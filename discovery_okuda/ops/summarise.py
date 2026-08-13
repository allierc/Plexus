#!/usr/bin/env python
"""summarise -- every finished ECM run as one table and one figure.

    python summarise.py                       # all 24-38
    python summarise.py --runs 29,30,31

WHY A FIGURE AND NOT JUST A TABLE. `strained_frac_end` is one number off the end of a curve, and two
runs can land on the same end value by entirely different routes -- one strains early and plateaus,
the other stays quiet and then runs away. The propagation is a CURVE, the runs are a sweep, and a
sweep of curves is a picture. The table is still written, because a number you can read off a plot
to two digits is not a number you can compare.

Panels: `strained_frac(t)`, the fraction of matrix above the colour floor; `front_r95(t)`, how far
out the strained material reaches; and the two of them against each other, which separates "more
material strained" from "strained further away".

BATCHES ARE DRAWN SEPARATELY AND SAID SO. 24-28 used stress_scale 0.05 and the reference reservoir
(3,170 cells); 29-38 used 0.08 and the x4 reservoir (5,968 cells). `strained_frac` is defined
relative to that scale, so a curve from one batch and a curve from the other are not the same
measurement -- overlaying them would invent a comparison neither run supports.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOG = os.path.join(ROOT, "log", "okuda_ECM")

BATCH = {"epi_": "batch 1  (reference reservoir, 3170 cells, stress_scale 0.05)",
         "epi2": "batch 2  (reservoir x4, 5968 cells, stress_scale 0.08)"}


def load(runs=None):
    out = []
    for d in sorted(glob.glob(os.path.join(LOG, "[23][0-9]_*"))):
        name = os.path.basename(d)
        if runs and name.split("_")[0] not in runs:
            continue
        mp = os.path.join(d, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        # ONLY RUNS MEASURED BY THE CURRENT CODE. `strained_frac` as a per-frame SERIES is the marker:
        # runs 20-23 have a `metrics.json` from the version that recorded one end-of-run number, no
        # front position and a `contact_frame` of 0 that came from a cavity thinner than the tissue.
        # Tabling them beside 24-38 would put four differently-defined numbers in the same column,
        # and cropping their strips into the montage would put the OLD dot renderer in a figure
        # about the new one. They are still on disk; they are not comparable.
        if not isinstance(m.get("strained_frac"), list):
            print(f"[summarise] {name}: metrics predate the per-frame series -- skipped "
                  f"(re-run `run_ecm.remeasure` on it if you want it in the table)")
            continue
        pp = os.path.join(d, "pass1.json")
        if os.path.exists(pp):
            m["pass1"] = json.load(open(pp))
        m["name"] = name
        out.append(m)
    return out


def table(rows):
    hdr = (f"{'run':28}{'contact':>8}{'strained_end':>13}{'front_r95':>10}{'cheb95':>8}"
           f"{'wall@':>7}{'max_disp':>9}{'expl':>6}{'wall_s':>8}  varied")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['name'][:27]:28}{'ERROR':>8}  {r['error'][:60]}")
            continue
        v = r.get("varied") or (r.get("pass1", {}) or {}).get("varied") or {}
        lines.append(
            f"{r['name'][:27]:28}{str(r.get('contact_frame')):>8}"
            f"{(r.get('strained_frac_end') or 0):>13.3f}{(r.get('front_r95_end') or 0):>10.3f}"
            f"{(r.get('front_cheb95_end') or 0):>8.3f}{str(r.get('front_reaches_wall')):>7}"
            f"{(r.get('max_disp') or 0):>9.3f}{str(r.get('exploded'))[0]:>6}"
            f"{(r.get('wall_s') or 0):>8.0f}  "
            + (", ".join(f"{k}={v[k]}" for k in v) or "baseline"))
    return "\n".join(lines)


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {}
    for r in rows:
        if "error" in r or not r.get("strained_frac"):
            continue
        groups.setdefault("epi2" if "epi2" in r["name"] else "epi_", []).append(r)
    if not groups:
        print("[summarise] no runs with a strain series -- no figure written")
        return
    nb = len(groups)
    fig, axes = plt.subplots(nb, 3, figsize=(15, 4.4 * nb), facecolor="black", squeeze=False)
    # BATCH 1 FIRST. Sorting the keys puts "epi2" before "epi_" ('2' < '_' in ASCII),
    # which reads as though the second batch came first.
    order = [k for k in ("epi_", "epi2") if k in groups]
    for bi, key in enumerate(order):
        rs = groups[key]
        # A PERCEPTUAL RAMP, not a qualitative palette: these runs are an ORDERED sweep, and
        # categorical colours would hide the ordering the sweep exists to show.
        cols = plt.cm.viridis(np.linspace(0.12, 0.95, len(rs)))
        for (ax, xk, yk, xl, yl) in [
                (axes[bi][0], None, "strained_frac", "frame", "strained fraction"),
                (axes[bi][1], None, "front_r95", "frame", "front r95  (box units)"),
                (axes[bi][2], "strained_frac", "front_r95", "strained fraction",
                 "front r95  (box units)")]:
            ax.set_facecolor("black")
            for r, c in zip(rs, cols):
                y = np.asarray(r[yk], float)
                x = np.arange(len(y)) if xk is None else np.asarray(r[xk], float)
                ax.plot(x, y, color=c, lw=1.4, label=r["name"][3:])
                cf = r.get("contact_frame")
                if xk is None and cf is not None and cf < len(y):
                    ax.plot([cf], [y[cf]], "o", color=c, ms=4)   # first contact, on the curve
            for s in ax.spines.values():
                s.set_color("#666")
            ax.tick_params(colors="#aaa", labelsize=8)
            ax.set_xlabel(xl, color="#aaa", fontsize=9)
            ax.set_ylabel(yl, color="#aaa", fontsize=9)
        axes[bi][0].text(0.02, 0.97, BATCH.get(key, key), transform=axes[bi][0].transAxes,
                         color="white", fontsize=11, va="top")
        axes[bi][2].legend(fontsize=7, labelcolor="#ddd", facecolor="black", edgecolor="#444",
                           loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    print(f"[summarise] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    a = ap.parse_args()
    runs = None if a.runs is None else {s.strip() for s in a.runs.split(",")}
    rows = load(runs)
    t = table(rows)
    print(t)
    open(os.path.join(LOG, "summary.txt"), "w").write(t + "\n")
    figure(rows, os.path.join(LOG, "summary.png"))
    montage(rows, os.path.join(LOG, "summary_montage.png"))
    json.dump([{k: v for k, v in r.items()
                if k not in ("strained_frac", "front_r95", "front_cheb95")} for r in rows],
              open(os.path.join(LOG, "summary.json"), "w"), indent=1)


# --------------------------------------------------------------------------- the sweep as one image
def montage(rows, path, col=-1, row=0, ncol=5):
    """One panel per run, cropped from the strips that already exist.

    CROPPED RATHER THAN RE-RENDERED, deliberately: every panel then comes from the same routine with
    the same fixed camera and the same palette, so the montage cannot disagree with the strip it was
    cut from. A second renderer built "just for the comparison figure" is a second thing that can be
    wrong, and the one that gets looked at.
    """
    import PIL.Image as I
    import PIL.ImageDraw as D
    tiles = []
    for r in rows:
        p = os.path.join(LOG, r["name"], "strip.png")
        if "error" in r or not os.path.exists(p):
            continue
        im = I.open(p)
        w, h = im.size[0] // 8, im.size[1] // 4          # the strip is 8 columns x 4 rows
        x = (im.size[0] - w) if col == -1 else col * w
        t = im.crop((x, row * h, x + w, (row + 1) * h)).convert("RGB")
        # AT THE BOTTOM, because the cropped panel already carries the renderer's own top-left
        # stamp (frame / cell count / strained %). Two labels in the same corner is one label.
        d = D.Draw(t)
        # `varied` lives in pass1.json for these runs: the sweep sets it on the dict AFTER `run`
        # has already written metrics.json, so metrics has the numbers and pass1 has the knob.
        v = r.get("varied") or (r.get("pass1", {}) or {}).get("varied") or {}
        d.text((10, t.size[1] - 34), r["name"], fill="white")
        d.text((10, t.size[1] - 20),
               ", ".join(f"{k}={v[k]}" for k in v) or "baseline (one edit per run)", fill="#999")
        tiles.append(t)
    if not tiles:
        print("[summarise] no strips to montage")
        return
    w, h = tiles[0].size
    nrow = (len(tiles) + ncol - 1) // ncol
    out = I.new("RGB", (w * ncol, h * nrow), "black")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % ncol) * w, (i // ncol) * h))
    out.save(path)
    print(f"[summarise] {path}  ({len(tiles)} runs, final frame, 3D side view)")


if __name__ == "__main__":
    main()
