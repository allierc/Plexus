#!/usr/bin/env python
"""p1a_ceiling.py -- PROBE A, the ceiling. What is the BEST single cell worth, in certified steps?

The six cells p1a_percell.py scored over a full beat were chosen for SPREAD (interior, edge, wall
band, corner), which is the right choice for showing the range and the wrong one for settling the
question. "Per-cell gain is invisible" is a claim about the best case, not the median: if the most
favourably placed cell in the sheet is worth less than one distinguishable step, no cell is
recoverable, and no amount of averaging over cells rescues a parameter that is per-cell by
definition.

So: take the top cells by the one-frame sensitivity the ladder already measured, and score a full
150-frame beat for each, for BOTH parameters, at both ticks. Same rollout, same reading surface,
same floors as p1a_percell.py -- only the choice of cell changes.

usage:
  PYTHONPATH=/workspace/Plexus/src python p1a_ceiling.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/workspace/Plexus/src", ALG, HERE, DISC):
    sys.path.insert(0, p)

import crash_test as CT                                               # noqa: E402
import accept as AC                                                   # noqa: E402
import metrics as MET                                                 # noqa: E402
from p1a_percell import clock_of, disp_stats, run, theta_vectors      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=180)
    ap.add_argument("--pulse-tick", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--beat-frames", type=int, default=150)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--ladder", default=os.path.join(HERE, "p1a_percell.json"))
    ap.add_argument("--tag", default="p1a")
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    d = json.load(open(args.ladder))
    per = d["ladder"]["pulse"]["per_cell"]
    C = d["C"]
    cells = list(range(1, C + 1))
    rank = {}
    for p in ("E", "gain"):
        v = np.array([per[str(c)][p]["frame"]["max_px"] for c in cells])
        rank[p] = [cells[i] for i in np.argsort(-v)[:args.top]]
    log(f"[top {args.top} by one-frame sensitivity at the pulse tick]  "
        f"E: {rank['E']}   gain: {rank['gain']}")

    floors = AC.working_floors()
    R = {"argv": vars(args), "top_cells": rank, "rows": {}}
    G = args.beat_frames
    pe = MET.REGISTRY["peak_excursion"]

    with torch.no_grad():
        for btag, W in (("pulse", args.pulse_tick), ("campaign", args.warmup)):
            a2 = argparse.Namespace(**vars(args))
            a2.warmup = W
            sy, _ = CT.plant_and_warm(a2, lambda *a: None)
            tracers = {m: CT.tracer_indices(sy.x0, CT.probe_points(m))
                       for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
            x_base, tr_base = run(sy, sy.E_true, sy.gain_true, W, G, tracers=tracers)
            real = tr_base[MET.MARGIN_SAFE].cpu().numpy()
            log(f"\n{'=' * 100}\n  CEILING [{btag}] tick {W} (clock {clock_of(W):.4f}), {G} frames"
                f"; reference peak_excursion {float(np.median(pe.reading(real))):.6g}\n{'=' * 100}")
            log(f"  {'candidate':<18s} {'max|dx| px':>11s} {'rms px':>9s} "
                + "".join(f"{n[:9]:>11s}" for n in AC.CERTIFIED) + f"{'WORST':>9s}")
            for param in ("E", "gain"):
                for c in rank[param]:
                    E, g = theta_vectors(sy, cell=c, param=param, frac=args.frac)
                    x_p, tr_p = run(sy, E, g, W, G, tracers=tracers)
                    sim = tr_p[MET.MARGIN_SAFE].cpu().numpy()
                    one = AC.score_one(sim, real, floors)
                    st = {n: one[n]["steps"] for n in AC.CERTIFIED}
                    live = [s for s in st.values() if s is not None]
                    worst = max(live) if live else None
                    rec = {"tick": W, "cell": c, "param": param,
                           "final_frame": disp_stats(x_p, x_base),
                           "peak_excursion_sim": float(np.median(pe.reading(sim))),
                           "steps": st, "worst_steps": worst}
                    R["rows"][f"{btag}|cell{c}_{param}"] = rec
                    log(f"  cell{c:<4d} {param:<8s} {rec['final_frame']['max_px']:>11.4f} "
                        f"{rec['final_frame']['rms_px']:>9.4f} "
                        + "".join(f"{st[n]:>11.4f}" if st[n] is not None else f"{'n/a':>11s}"
                                  for n in AC.CERTIFIED)
                        + f"{worst:>9.4f}")
            for param in ("E", "gain"):
                w = [R["rows"][f"{btag}|cell{c}_{param}"]["worst_steps"] for c in rank[param]]
                R[f"best_{btag}_{param}"] = float(max(w))
                log(f"  BEST SINGLE CELL, {param:<5s} [{btag}]: {max(w):.4f} steps "
                    f"({'ABOVE' if max(w) >= 1 else 'below'} one distinguishable step)")
            del sy
            torch.cuda.empty_cache()

    out = os.path.join(HERE, f"{args.tag}_ceiling.json")
    json.dump(R, open(out, "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{args.tag}_ceiling.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
