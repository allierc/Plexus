#!/usr/bin/env python
"""op_figure -- one panel per parameter: how the run responds as that knob is swept.

Cedric, 6 August: *"a template for a panel plot for each operator parameter dependence, to be saved
in a given folder"*.

WHAT THIS IS FOR, AND WHY THE BATTERY IS NOT ENOUGH. `op_probe --all` answers LIVE or DEAD from two
points, which is the right question to ask before spending a GPU on a knob. It cannot tell a knob
that responds smoothly from one that is railed, saturated, or non-monotonic -- and those are
exactly the failures that cost this campaign its rounds:

  * `cell_chem_from_shape.beta` reads LIVE and is pinned to the F clamp, so -2 and -4 emit the SAME field.
    Two points cannot see that. Five can: the curve is flat.
  * ten runs at exactly protr 1.022 was a rail nobody saw for two rounds.

A sweep with a plotted curve makes both visible at a glance, which is the same argument as
`figure_geometries.py`: an assertion cannot catch a defect nobody thought to bound, and a picture
beside the number can.

COST. Each point is one warm 50-frame run (~75 s at 2.9k cells), so a 5-point sweep of a 9-parameter
operator is ~45 runs. Sweep one operator at a time, and use `--values` to keep it small.

TOPOLOGY IS FROZEN BY DEFAULT. With `cell_divide` live, a 0.1% perturbation and a 100% perturbation
both saturate at rel ~0.29 -- measured -- because the cell count diverges and the distance stops
being about the parameter. A dependence plot drawn on that would be a picture of chaos.

RUN:  python op_figure.py <spec.yaml> --op cell_chem_from_shape
      python op_figure.py <spec.yaml> --op cell_chem_diffuse --values 7 --out figs/operators
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.abspath(os.path.join(HERE, "..", "src"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import op_probe as P                                                 # noqa: E402

OUT = os.path.join(HERE, "figs", "operators")
GOLD = "#f2c14e"        # the house "this one fires" colour, from figure_geometries.py
GREY = "0.55"


def sweep_values(v, n):
    """`n` values spanning a decade around the spec's OWN value, log-spaced for a scale parameter.

    Around the parent's value, not across the declared range: every pool parent sits outside its
    declared box on at least one parameter, so a range midpoint would be a different composition.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return []
    if v == 0:
        return list(np.linspace(-1.0, 1.0, n))
    lo, hi = abs(v) * 0.25, abs(v) * 4.0
    g = np.geomspace(lo, hi, n) * (1 if v > 0 else -1)
    if isinstance(v, int):
        return sorted({max(1, int(round(x))) for x in g})
    # PLAIN FLOATS, not numpy scalars. The spec is round-tripped through yaml.safe_dump on every
    # probe, and a np.float64 raises RepresenterError -- which surfaced as "nothing measurable",
    # i.e. a sweep that silently produced no points and a figure that reported the parameter dead.
    return [float(x) for x in g]


def measure(spec, ckpt, op_name, frames=50, n=5, device="cpu", freeze=True):
    """{param: (values, rel_changes)} -- one warm run per point, one shared baseline."""
    warm = P.warm_spec(spec, ckpt, frames=frames, freeze_topology=freeze)
    if not any(o["op"] == op_name for o in warm["operators"]):
        raise SystemExit(f"{op_name} is not in this spec"
                         + (" (frozen out -- pass --no-freeze)" if not freeze else ""))
    base = P.run_spec(warm, frames, device)
    declared = P._params_of(warm, op_name)
    missing, _ = P.unread_params(op_name, declared, P._impl_of(warm, op_name))

    curves = {}
    for key, val in sorted(declared.items()):
        if key in (missing or []):
            curves[key] = ([], [], val, "UNREAD")
            print(f"  {op_name}.{key}: UNREAD, skipped", flush=True)
            continue
        vals = sweep_values(val, n)
        if not vals:
            continue
        xs, ys = [], []
        for v in vals:
            d = copy.deepcopy(warm)
            P._set_param(d, op_name, key, v)
            try:
                rel, _ = P._distance(P.run_spec(d, frames, device), base)
            except Exception as e:
                print(f"    {key}={v}: {type(e).__name__}", flush=True)
                continue
            xs.append(float(v))
            ys.append(float(rel))
            print(f"    {key}={v:<12.6g} rel {rel:.4e}", flush=True)
        curves[key] = (xs, ys, val, "measured")
    return curves


def draw(curves, op_name, out_dir=OUT, note=""):
    """One panel per parameter. Black ground, bold letter top-left, no titles -- house style."""
    keys = [k for k, v in curves.items() if v[3] == "measured" and len(v[0]) > 1]
    dead = [k for k, v in curves.items() if v[3] != "measured" or len(v[0]) <= 1]
    if not keys:
        raise SystemExit(f"nothing measurable for {op_name}")
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), facecolor="black",
                             squeeze=False)

    for i, k in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        xs, ys, parent, _ = curves[k]
        ax.set_facecolor("black")
        for s in ax.spines.values():
            s.set_color("0.35")
        ax.tick_params(colors="0.7", labelsize=8)

        # FLAT MEANS SATURATED, and that is the thing the plot exists to show. A knob whose
        # response varies by less than 10% across a 16x sweep is not a strength, it is a switch --
        # which is exactly what cell_chem_from_shape.beta turned out to be against the F clamp.
        span = (max(ys) - min(ys)) / max(abs(np.mean(ys)), 1e-30) if ys else 0.0
        col = GOLD if span > 0.10 else GREY
        ax.plot(xs, ys, "o-", color=col, lw=1.4, ms=4)
        ax.axvline(parent, color="white", lw=0.8, ls="--", alpha=0.55)   # the spec's own value
        if min(x for x in xs) > 0:
            ax.set_xscale("log")
        ax.set_yscale("log")

        ax.text(0.03, 0.95, chr(97 + i), transform=ax.transAxes, color="white",
                fontsize=13, fontweight="bold", va="top")
        ax.text(0.03, 0.83, k, transform=ax.transAxes, color="white", fontsize=10, va="top")
        ax.text(0.03, 0.72, f"span {span * 100:.0f}%" + ("  SATURATED" if span <= 0.10 else ""),
                transform=ax.transAxes, color=col, fontsize=8.5, va="top")

    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.text(0.005, 0.995, op_name, color="white", fontsize=11, va="top", fontweight="bold")
    tail = f"response = relative L2 on the trajectory; dashed = the spec's own value.  {note}"
    if dead:
        tail += f"   not plotted: {', '.join(dead)}"
    fig.text(0.005, 0.012, tail, color="0.6", fontsize=8, va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{op_name}.{ext}")
        fig.savefig(p, facecolor="black", dpi=170)
        paths.append(p)
        print(f"  wrote {p}")
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--op", required=True)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "fixtures", "coral_gate_div_f400.npz"))
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--values", type=int, default=5, help="points per parameter")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-freeze", action="store_true",
                    help="keep divide/T1 -- the response then saturates, see the module docstring")
    a = ap.parse_args()

    import yaml
    spec = yaml.safe_load(open(a.spec))
    curves = measure(spec, a.ckpt, a.op, a.frames, a.values, a.device, freeze=not a.no_freeze)
    draw(curves, a.op, a.out,
         note=f"{a.frames} frames, warm from {os.path.basename(a.ckpt)}"
              + (", topology frozen" if not a.no_freeze else ", FULL composition"))
    with open(os.path.join(a.out, f"{a.op}.json"), "w") as fh:
        json.dump({k: {"values": v[0], "rel": v[1], "parent": v[2], "status": v[3]}
                   for k, v in curves.items()}, fh, indent=1)


if __name__ == "__main__":
    main()
