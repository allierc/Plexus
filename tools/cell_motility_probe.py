"""Does the cycling cell actually EXTEND AND RETRACT, and does it get anywhere?

A movie of a crawling cell answers neither question. A cell that is translating because a steady
rearward traction is dragging it looks, at 30 frames a second, exactly like a cell that is
translating because it reached forward and pulled itself up -- and the two differ in the only place
that matters, which is whether the shape changes at all. So this measures the shape, per frame,
along the cell's OWN polarity direction:

    L_par   the cell's extent ALONG its polarity n, in um -- the length that a protrusion
            increases and a retraction decreases
    L_perp  its extent ACROSS n, in the horizontal plane, in um -- which must go the OTHER WAY
            if the active stress is doing what a deviatoric (constant-volume) stress does
    height  its extent along the up axis, in um
    x_par   the centroid's displacement along n since frame 0, in um -- the distance travelled
    phase   the cell's own phase phi, in radians, so an oscillation in L_par can be matched to
            the cycle that is supposed to be driving it rather than to noise

and reports the OSCILLATION AMPLITUDE of L_par as a percentage of its own mean, which is the
number the design is about: a 1% cell is not protruding, whatever its centroid is doing.

    PYTHONPATH=src python tools/cell_motility_probe.py --spec cell/adh_stress_f25 --frames 600

The run here is throwaway -- `save_data: false`, no plotting -- because the output wanted is the
table and the PNG, not another movie of the thing already being rendered on the cluster.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="cell/adh_stress_f25")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--every", type=int, default=5, help="frames between measurements")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=None, help="PNG; default graphs_data/cell/<name>_probe.png")
    args = ap.parse_args()

    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine
    from plexus.paths import resolve_config

    yaml_file, _pre, _name = resolve_config(args.spec)
    raw = yaml.safe_load(open(yaml_file))
    tag = raw["general"]["name"]
    raw["general"].update(n_frames=args.frames, name=f"{tag}_probe", save_data=False, record_cap=3)
    raw["plotting"] = {}
    um = float(raw["general"].get("units", {}).get("length_um", 100.0))
    tmp = os.path.join(tempfile.mkdtemp(prefix="motprobe_"), "spec.yaml")
    yaml.safe_dump(raw, open(tmp, "w"), sort_keys=False)

    rows = []

    def on_frame(H, tick):
        if tick % args.every:
            return
        import torch
        cl = H.level("cell")
        b0, b1 = cl.state_schema["polarity"]
        n = cl.state[0, b0:b1].detach().cpu().numpy()
        n[1] = 0.0                                   # the crawl is in the plane of the substrate
        n = n / max(np.linalg.norm(n), 1e-12)
        q0, q1 = cl.state_schema["phase"]
        ph = float(cl.state[0, q0])
        # LIVE POINTS ONLY. A run with a dormant monomer pool carries points that are not part of
        # the cell yet -- they contribute no mass and no stress, and counting them in an extent
        # measures the pool's parking spot rather than the cell.
        P = torch.cat([l.get("pos").detach()[l.occ > 0] for k, l in H.levels.items()
                       if k.endswith("_node")], 0).cpu().numpy()
        c = P.mean(0)
        t = np.array([-n[2], 0.0, n[0]])             # the horizontal direction across the polarity
        par = P @ n
        per = P @ t
        rows.append(dict(tick=tick, phase=ph,
                         L_par=float(par.max() - par.min()) * um,
                         L_perp=float(per.max() - per.min()) * um,
                         height=float(P[:, 1].max() - P[:, 1].min()) * um,
                         cx=float(c @ n) * um, cy=float(c[1]) * um))

    H, _ = engine.run(load(tmp), device=args.device, on_frame=on_frame, progress=False)

    tk = np.array([r["tick"] for r in rows], float)
    Lp = np.array([r["L_par"] for r in rows])
    Lq = np.array([r["L_perp"] for r in rows])
    Hh = np.array([r["height"] for r in rows])
    cx = np.array([r["cx"] for r in rows]) - rows[0]["cx"]
    ph = np.unwrap(np.array([r["phase"] for r in rows]))

    # THE SETTLED HALF ONLY. The first frames are the load's transient -- the material starts at
    # rest and unstressed, so the cell relaxes into the floor once, and that one-off relaxation is
    # bigger than anything the cycle does. An amplitude measured over it would be the transient's.
    h = len(Lp) // 2
    amp = 0.5 * (Lp[h:].max() - Lp[h:].min())
    print(f"\n  spec {tag}   {args.frames} frames, measured every {args.every}")
    print(f"  {'tick':>6} {'phase':>7} {'L_par um':>9} {'L_perp um':>10} {'height um':>10} "
          f"{'travel um':>10}")
    for r, x in zip(rows[:: max(1, len(rows) // 24)], cx[:: max(1, len(rows) // 24)]):
        print(f"  {r['tick']:6d} {r['phase'] % 6.2832:7.2f} {r['L_par']:9.2f} {r['L_perp']:10.2f} "
              f"{r['height']:10.2f} {x:10.2f}")
    print(f"\n  L_par over the settled half: mean {Lp[h:].mean():.2f} um, "
          f"swing +-{amp:.2f} um = {100 * amp / max(Lp[h:].mean(), 1e-9):.1f}% of its own length")
    print(f"  travelled {cx[-1]:.2f} um along the polarity in "
          f"{args.frames * float(raw['general']['dt']):.2f} s "
          f"({ph[-1] / 6.2832:.1f} cycles)\n", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True, facecolor="black")
    for a in ax:
        a.set_facecolor("black")
        for s in a.spines.values():
            s.set_color("0.6")
        a.tick_params(colors="0.8")
    ax[0].plot(tk, Lp, color="#ff5555", lw=1.6, label="along polarity")
    ax[0].plot(tk, Lq, color="#5599ff", lw=1.6, label="across polarity")
    ax[0].plot(tk, Hh, color="0.7", lw=1.2, label="height")
    ax[0].set_ylabel("cell extent (µm)", color="0.9")
    ax[0].legend(facecolor="black", labelcolor="0.9", edgecolor="0.4", fontsize=9)
    ax[1].plot(tk, cx, color="#66dd88", lw=1.8)
    ax[1].plot(tk, 2.0 * np.cos(ph) + cx.mean(), color="0.45", lw=1.0,
               label="cycle (arbitrary scale)")
    ax[1].set_ylabel("travel along polarity (µm)", color="0.9")
    ax[1].set_xlabel("frame", color="0.9")
    ax[1].legend(facecolor="black", labelcolor="0.9", edgecolor="0.4", fontsize=9)
    out = args.out or os.path.join(ROOT, "graphs_data", "cell", f"{tag}_probe.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.suptitle("", color="w")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="black")
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
