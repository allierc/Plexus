#!/usr/bin/env python
"""p1v_long_impostor -- ADVERSARIAL VERIFICATION of probe C, second decisive test.

Probe C's ATTACK 4 established, on its own numbers, that ONE beat understates a candidate's error:
uniform E = 800 reads 4.36 steps on beat 1 and 33.41 through accept() over three beats. Every
number that carries the claim's second half -- "a single uniform modulus imitates the whole planted
field to 0.73-2.66 steps" -- was read on ONE beat.

So: rescan the uniform-E impostor over THREE beats, scored through accept() exactly as attack 4
did (three ticks, worst tick, worst instrument). If the best impostor stays under a step, the claim
holds under its own strongest protocol. If it climbs to several steps, the 0.73 is a one-beat
artefact and the per-cell field IS separable from every uniform sheet.

Same reading surface (10x10, margin 20), same floors, same reference (theta_true rollout).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p1c_lib as L                                                        # noqa: E402
import accept as ACC                                                       # noqa: E402

T0 = time.time()
INS = L.INSTRUMENTS


def log(s=""):
    print(f"[{time.time() - T0:7.1f}s] {s}" if s else "")
    sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--beats", type=int, default=3)
    ap.add_argument("--out", default="p1v_long_impostor.json")
    a = ap.parse_args()

    per, beats = 150, a.beats
    args = L.default_args(device=a.device, warmup=per + 30, window=per * beats)
    rig = L.Rig(args)
    ref = rig.roll(rig.theta_true, G=per * beats)
    log(f"planted E [{float(rig.E_true.min()):.1f}, {float(rig.E_true.max()):.1f}] "
        f"median {float(rig.E_true.median()):.1f}; {beats} beats of {per} frames")

    cands = {}
    for E in (66.2, 78.3, 86.2, 92.6, 100.0, 109.5, 128.4):
        cands[f"uniform E={E}"] = rig.theta(E=E)
    cands["theta_true (control)"] = rig.theta_true
    cands["gain x1.05, E true (control)"] = rig.theta(gain=(rig.gain_true * 1.05).cpu().numpy())

    out, rows = {}, []
    log(f"  {'candidate':<30s}{'beat1 STAT':>12s}{'accept 3-tick':>15s}{'limiting':>20s}"
        f"{'per-beat':>34s}")
    for name, th in cands.items():
        sim = rig.roll(th, G=per * beats)
        one = L.steps_row(sim[:per], ref[:per])
        pairs = [(sim[k * per:(k + 1) * per], ref[k * per:(k + 1) * per]) for k in range(beats)]
        acc = ACC.accept(pairs, L.floors())
        pb = [float(max(v for v in (L.steps_row(p[0], p[1])[n] for n in INS)
                        if isinstance(v, float))) for p in pairs]
        out[name] = {"one_beat_STAT": one["STAT"], "accept_statistic": acc["statistic"],
                     "limiting": acc["limiting_instrument"], "per_beat_STAT": pb,
                     "beats_null": acc["beats_null"], "informative": acc["informative"]}
        rows.append((name, one["STAT"], acc["statistic"]))
        log(f"  {name:<30s}{one['STAT']:12.2f}{acc['statistic']:15.2f}"
            f"{acc['limiting_instrument']:>20s}{str([round(x, 2) for x in pb]):>34s}")

    best1 = min((r for r in rows if r[0].startswith("uniform")), key=lambda r: r[1])
    best3 = min((r for r in rows if r[0].startswith("uniform")), key=lambda r: r[2])
    log(f"\n  best uniform impostor on ONE beat : {best1[0]} at {best1[1]:.2f} steps")
    log(f"  best uniform impostor on THREE beats: {best3[0]} at {best3[2]:.2f} steps "
        f"(accept, 3 ticks)")
    log(f"  the null (knowing nothing): {min(L.null_row().values()):.2f} steps")
    out["_summary"] = {"best_one_beat": {"candidate": best1[0], "steps": best1[1]},
                       "best_three_beats": {"candidate": best3[0], "steps": best3[2]},
                       "null_min": float(min(L.null_row().values()))}
    rig.free()
    json.dump(out, open(os.path.join(HERE, a.out), "w"), indent=1, default=float)
    log(f"  -> {a.out}")
