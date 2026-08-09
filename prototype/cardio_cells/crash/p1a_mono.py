#!/usr/bin/env python
"""p1a_mono.py -- PROBE A, the last control. Is the best cell's signal a SIGNAL, or divergence?

p1a_ceiling.py found that +10% on the single most favourable cell's gain moves the acceptance
statistic 2.56 steps over a 150-frame free rollout, against 0.88 for the most favourable cell's E.
A number that large after 150 frames of free integration has two possible parents and they mean
opposite things:

  A RESPONSE.  The amplitude channel reads a monotone function of the parameter, so an observed
               amplitude names a value and the parameter is recoverable in principle.
  DIVERGENCE.  Two trajectories that started 1e-6 apart have walked apart by 150 frames. The
               statistic is large, and it is large for -10% as well as +10%, and for +3% as well
               as +30%. It carries no information about WHICH WAY the parameter is wrong, so no
               estimator can descend it.

The discriminator is the sign and the ladder: sweep the fraction through zero and read the raw
`peak_excursion` property (not the paired distance, which is an absolute value and therefore blind
to sign) at each. A response is monotone through zero. Divergence is a V.

usage:
  PYTHONPATH=/workspace/Plexus/src python p1a_mono.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--beat-frames", type=int, default=150)
    ap.add_argument("--fracs", default="-0.3,-0.1,-0.03,0.03,0.1,0.3")
    ap.add_argument("--cases", default="66:gain,6:gain,7:E,51:E",
                    help="cell:param pairs, the ceiling's winners")
    ap.add_argument("--tag", default="p1a")
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    fracs = [float(x) for x in args.fracs.split(",")]
    cases = [(int(c.split(":")[0]), c.split(":")[1]) for c in args.cases.split(",")]
    cases += [(None, "gain"), (None, "E")]                 # the uniform control
    floors = AC.working_floors()
    pe = MET.REGISTRY["peak_excursion"]
    G, W = args.beat_frames, args.warmup
    R = {"argv": vars(args), "cases": {}}

    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, lambda *a: None)
        tracers = {m: CT.tracer_indices(sy.x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
        x_base, tr_base = run(sy, sy.E_true, sy.gain_true, W, G, tracers=tracers)
        real = tr_base[MET.MARGIN_SAFE].cpu().numpy()
        amp0 = float(np.median(pe.reading(real)))
        log(f"[base] tick {W} (clock {clock_of(W):.4f}), {G} frames, peak_excursion {amp0:.6g}")

        for c, param in cases:
            name = f"cell{c}_{param}" if c else f"UNIFORM_{param}"
            log(f"\n  {name:<16s} {'frac':>7s} {'amplitude':>12s} {'d_amp/amp0':>11s} "
                f"{'max|dx| px':>11s} {'STEPS':>8s}  limiting")
            rows = []
            for f in fracs:
                if c:
                    E, g = theta_vectors(sy, cell=c, param=param, frac=f)
                else:
                    E, g = theta_vectors(sy, frac=f, uniform=param)
                x_p, tr_p = run(sy, E, g, W, G, tracers=tracers)
                sim = tr_p[MET.MARGIN_SAFE].cpu().numpy()
                one = AC.score_one(sim, real, floors)
                st = {n: one[n]["steps"] for n in AC.CERTIFIED}
                live = [n for n in st if st[n] is not None]
                worst = max([st[n] for n in live]) if live else None
                lim = max(live, key=lambda n: st[n]) if live else None
                amp = float(np.median(pe.reading(sim)))
                rows.append({"frac": f, "amplitude": amp, "rel_amp": (amp - amp0) / amp0,
                             "max_px": disp_stats(x_p, x_base)["max_px"],
                             "steps": st, "worst_steps": worst, "limiting": lim})
                log(f"  {'':<16s} {f:>+7.2f} {amp:>12.6g} {(amp-amp0)/amp0:>+11.4f} "
                    f"{rows[-1]['max_px']:>11.4f} "
                    f"{worst:>8.4f}  {lim}")
            a = np.array([r["amplitude"] for r in rows])
            df = np.diff(a)
            mono = bool(np.all(df > 0) or np.all(df < 0))
            sgn = float(np.sign(rows[-1]["rel_amp"]) * np.sign(rows[0]["rel_amp"]))
            R["cases"][name] = {"cell": c, "param": param, "rows": rows,
                                "amplitude_monotone_in_frac": mono,
                                "n_turning_points": int((np.diff(np.sign(df)) != 0).sum()),
                                "amp0": amp0,
                                "opposite_sign_at_the_two_ends": sgn < 0,
                                "span_rel_amp": float(a.max() - a.min()) / amp0}
            v = R["cases"][name]
            log(f"  {'':<16s} -> amplitude monotone in the perturbation: {mono}   "
                f"turning points {v['n_turning_points']}   "
                f"ends have opposite sign: {v['opposite_sign_at_the_two_ends']}   "
                f"span {100*v['span_rel_amp']:.2f}% of the reference amplitude")

    out = os.path.join(HERE, f"{args.tag}_mono.json")
    json.dump(R, open(out, "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{args.tag}_mono.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
