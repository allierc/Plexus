"""Did the polymerised material actually deform the CELL, or only draw itself?

The picture cannot answer this. New material is rendered in its own colour, so a fibre bundle
reaching past the membrane looks exactly like a membrane that has been pushed out -- and the cell
is ALSO still relaxing from the configuration it was loaded from, so its front edge advances for
reasons that have nothing to do with polymerisation. Two things are therefore measured, per frame,
and they are the two the claim rests on:

    front / rear   the PLASMA MEMBRANE's reach along the cell's polarity, measured from the cell's
                   own centroid, in um. This is the cell's shape. The new fibres are excluded --
                   they are what is being tested, not evidence.
    new           how many pool points are live and what volume they are worth, in um^3 and as a
                   percentage of the cell's own volume, so a shape change can be checked against
                   the volume that was supposed to cause it.

and the run is done TWICE: once as specified, once with `rate: 0`, which is the same cell relaxing
with no polymerisation at all. The difference between the two front reaches is the effect. The
front reach on its own is not.

    PYTHONPATH=src python tools/cell_poly_probe.py --spec cell/adh_poly_shape --frames 300
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

NEW = "cytoskeleton_new_node"
MEMB = "plasma_membrane_node"


def run_one(raw, frames, every, device, rate, um):
    import torch
    from plexus.schema import load
    from plexus import engine

    r = yaml.safe_load(yaml.safe_dump(raw))
    r["general"].update(n_frames=frames, save_data=False, record_cap=3,
                        name=f"{raw['general']['name']}_probe{'' if rate else '_ctl'}")
    r["plotting"] = {}
    for o in r["operators"]:
        if o["op"] == "polymerize_tips":
            o["rate"] = rate
    tmp = os.path.join(tempfile.mkdtemp(prefix="polyprobe_"), "spec.yaml")
    yaml.safe_dump(r, open(tmp, "w"), sort_keys=False)

    rows = []

    def on_frame(H, tick):
        if tick % every:
            return
        cl = H.level("cell")
        b0, b1 = cl.state_schema["polarity"]
        n = cl.state[0, b0:b1].detach().clone()
        n[1] = 0.0
        n = n / n.norm().clamp_min(1e-12)
        # THE CENTROID OF THE CELL BODY, not of everything drawn: the pool is excluded here too, so
        # a bundle of fibres cannot move the origin the reach is measured from.
        body = [k for k, l in H.levels.items() if k.endswith("_node") and k != NEW]
        P = torch.cat([H.level(k).get("pos")[H.level(k).occ > 0] for k in body], 0)
        c = P.mean(0)
        m = H.level(MEMB)
        Xm = m.get("pos")[m.occ > 0]
        pm = ((Xm - c) * n).sum(1)
        w = ((Xm - c) * torch.tensor([-n[2], 0.0, n[0]], device=n.device)).sum(1)
        nl = H.level(NEW)
        live = nl.occ > 0
        # THE CELL'S ACTUAL VOLUME, by counting the 1 um voxels its material occupies. Extents are
        # what a shape LOOKS like; volume is what "the added material had to go somewhere" is a
        # statement about, and it is the only way to tell an expanding cell from a compressing one.
        V = torch.unique((P * (um / 1.0)).floor().to(torch.int64), dim=0).shape[0] * 1.0
        # ABSOLUTE POSITIONS AS WELL AS CENTROID-RELATIVE ONES. Removing material from the back
        # moves the centroid forward, so a centroid-relative front reach FALLS while the cell is
        # advancing -- measured, front 14.5 -> 14.2 um on a cell that was moving. `length` and
        # `travel` are the two independent facts: one is the shape, the other is the motion.
        pa = (Xm * n).sum(1)
        # PERCENTILES, NOT THE EXTREMES. A max-minus-min extent is decided by ONE point, so a single
        # membrane node flung to the wall reads as a cell three times its own length -- measured,
        # 28.0 -> 86.5 um with the width unchanged, which no elongation of a fixed volume can do.
        # The 0.5th and 99.5th percentiles describe the body; `stray` counts what is outside them by
        # more than a cell radius, which is the number that says whether the run is sane.
        q = torch.quantile(pm, torch.tensor([0.005, 0.995], device=pm.device, dtype=pm.dtype))
        rows.append(dict(tick=tick, travel=float((c * n).sum()) * um,
                         length=float(q[1] - q[0]) * um,
                         raw_len=float(pm.max() - pm.min()) * um,
                         stray=int(((pm > q[1] + 0.10) | (pm < q[0] - 0.10)).sum()),
                         abs_front=float(pa.max()) * um, abs_rear=float(pa.min()) * um,
                         front=float(pm.max()) * um, rear=float(-pm.min()) * um,
                         width=float(w.max() - w.min()) * um,
                         height=float(Xm[:, 1].max() - Xm[:, 1].min()) * um,
                         vol=float(V), n_new=int(live.sum()),
                         v_new=float(nl.p_vol[live].sum()) * um ** 3))

    H, _ = engine.run(load(tmp), device=device, on_frame=on_frame, progress=False)
    # the cell's own volume, from the material it is made of, for the percentage to mean something
    vol = sum(float(H.level(k).p_vol[H.level(k).occ > 0].sum())
              for k, l in H.levels.items() if k.endswith("_node") and k != NEW) * um ** 3
    return rows, vol


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="cell/adh_poly_shape")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--ctl-device", default=None, help="device for the rate-0 control")
    ap.add_argument("--no-control", action="store_true")
    args = ap.parse_args()

    import plexus.operators  # noqa: F401
    from plexus.paths import resolve_config

    yaml_file, _p, _n = resolve_config(args.spec)
    raw = yaml.safe_load(open(yaml_file))
    um = float(raw["general"].get("units", {}).get("length_um", 100.0))
    tag = raw["general"]["name"]

    on, vol = run_one(raw, args.frames, args.every, args.device,
                      next(o["rate"] for o in raw["operators"] if o["op"] == "polymerize_tips"), um)
    off = None if args.no_control else run_one(raw, args.frames, args.every,
                                               args.ctl_device or args.device, 0.0, um)[0]

    print(f"\n  {tag}: {args.frames} frames, cell body worth {vol:,.0f} um^3\n")
    print(f"  {'tick':>6} {'length um':>10} {'width um':>9} {'high um':>8} {'tip um':>8} "
          f"{'travel um':>10} {'vol um3':>9} {'raw len':>8} {'stray':>6} {'new pts':>9} {'new um^3':>9} {'% cell':>7}" +
          ("" if off is None else f" | {'len ctl':>8} {'DIFF um':>8}"))
    for i, r in enumerate(on):
        line = (f"  {r['tick']:6d} {r['length']:10.2f} {r['width']:9.2f} {r['height']:8.2f} "
                f"{r['abs_front'] - on[0]['abs_front']:8.2f} "
                f"{r['travel'] - on[0]['travel']:10.2f} "
                f"{r['vol']:9.0f} {r['raw_len']:8.1f} {r['stray']:6d} "
                f"{r['n_new']:9,d} {r['v_new']:9.1f} {100 * r['v_new'] / max(vol, 1e-9):7.1f}")
        if off is not None and i < len(off):
            line += f" | {off[i]['length']:8.2f} {r['length'] - off[i]['length']:8.2f}"
        print(line)
    d0 = on[0]["length"]
    print(f"\n  cell VOLUME {on[0]['vol']:,.0f} -> {on[-1]['vol']:,.0f} um^3 "
          f"({on[-1]['vol'] - on[0]['vol']:+,.0f}), for {on[-1]['v_new']:,.0f} um^3 added "
          f"-- {100 * (on[-1]['vol'] - on[0]['vol']) / max(on[-1]['v_new'], 1e-9):.0f}% of it showed up")
    print(f"  length {d0:.2f} -> {on[-1]['length']:.2f} um "
          f"({on[-1]['length'] - d0:+.2f} um, x{on[-1]['length'] / d0:.2f}), width "
          f"{on[0]['width']:.2f} -> {on[-1]['width']:.2f} um, tip advanced "
          f"{on[-1]['abs_front'] - on[0]['abs_front']:+.2f} um, cell travelled "
          f"{on[-1]['travel'] - on[0]['travel']:+.2f} um")
    if off is not None:
        c0 = off[0]["length"]
        print(f"  length {c0:.2f} -> {off[-1]['length']:.2f} um "
              f"({off[-1]['length'] - c0:+.2f} um) with NO polymerisation")
        eff = (on[-1]['length'] - d0) - (off[-1]['length'] - c0)
        print(f"  EFFECT OF POLYMERISATION: {eff:+.2f} um of LENGTH, for "
              f"{on[-1]['v_new']:,.0f} um^3 of material added "
              f"({100 * on[-1]['v_new'] / max(vol, 1e-9):.1f}% of the cell)\n", flush=True)


if __name__ == "__main__":
    main()
