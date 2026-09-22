#!/usr/bin/env python
"""One line per run: is the cilium beating, is it upright, and is it deformed?

    python tools/rod_row.py cil_s7_beat cil_s6_measured ...

The full `rod_probe` prints twenty lines and is right for reading one run carefully. Iterating
needs the opposite: every run on one line so the previous three are still on the screen, and every
column a thing that can be WRONG rather than a thing that is interesting.

    base     the beat's amplitude at the BASE joint, degrees peak to peak. Without this column
             a dead tip is ambiguous: a base that is not moving means the drive never got past
             whatever holds it, while a base that swings and a tip that does not is the stroke
             dying on its way up the rod, which is a sperm-number problem. Two different fixes.
    tip      the beat's amplitude at the tip, degrees peak to peak. The target.
    t.mean   the angle the beat is CENTRED on. 0 is upright; a rod can sweep beautifully about a
             rest direction that has flopped over, and amplitude alone calls that a beat.
    b.mean   the same at the BASE. This is the one that moves first when the drive overwhelms
             whatever is holding the base, and it reached +107 degrees once while the tip still
             looked calm.
    len%     end to end, against rest. Away from 100 means stretched OR curled.
    strain   the worst SEGMENT, which is what separates those two: a curled rod holds its segments
             and a pulled-apart one does not.
    bend     the worst joint angle, degrees.
    drift    how far the pinned base node has slid from its anchor.
    period   measured against commanded. Disagreement means the rod is not following its drive.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))


def row(nm):
    import yaml
    from rod_probe import load, angles
    tr, sp = load(nm)
    d = yaml.safe_load(open(sp))
    rs = next(o for o in d["seed"] if o["op"] == "rod_seed")
    rb = next(o for o in d["operators"] if o["op"] == "rod_base")
    d0 = np.asarray(rs["direction"], float); d0 /= np.linalg.norm(d0)
    n = np.asarray(rs["beat_axis"], float); n = n - (n @ d0) * d0; n /= np.linalg.norm(n)
    P = np.asarray(tr["rod_node__pos"]); T, N, _ = P.shape
    A = angles(P, d0, n); s = slice(int(T / 3), None)
    L0 = float(rs["length"]); rest = L0 / (N - 1)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    e = np.diff(P, axis=1); e = e / np.maximum(np.linalg.norm(e, axis=2, keepdims=True), 1e-12)
    bend = np.degrees(np.arccos(np.clip((e[:, :-1] * e[:, 1:]).sum(2), -1, 1)))
    stride = max(1, d["general"]["n_frames"] // max(T - 1, 1))
    dt = float(d["general"]["dt"]) * stride
    y = A[s, -1] - A[s, -1].mean()
    per = float("nan")
    if np.ptp(y) > 1e-9:
        f = np.fft.rfftfreq(y.size, dt)
        k = 1 + int(np.argmax(np.abs(np.fft.rfft(y))[1:]))
        per = 1.0 / f[k] if f[k] > 0 else float("nan")
    return dict(name=nm, tip=np.ptp(A[s, -1]), base=np.ptp(A[s, 1]),
                tmean=A[s, -1].mean(), bmean=A[s, 1].mean(),
                length=100 * np.linalg.norm(P[-1, -1] - P[-1, 0]) / L0,
                strain=100 * np.abs((seg[s] - rest) / rest).max(), bend=bend[s].max(),
                drift=np.linalg.norm(P[:, 0, :] - P[0, 0, :], axis=1).max(),
                per=per, cmd=2 * math.pi / float(rb["omega"]), moment=float(rb["moment"]),
                clamp=float(rb.get("clamp", 0.0)))


if __name__ == "__main__":
    print(f"  {'run':22s} {'base':>7s} {'tip':>7s} {'t.mean':>8s} {'b.mean':>8s} {'len%':>6s} "
          f"{'strain':>7s} {'bend':>6s} {'drift':>9s} {'period':>7s} {'cmd':>6s}")
    for nm in sys.argv[1:]:
        try:
            r = row(nm)
        except SystemExit:
            print(f"  {nm:22s}  (no run yet)"); continue
        except Exception as e:                                        # noqa: BLE001
            print(f"  {nm:22s}  {type(e).__name__}: {str(e)[:60]}"); continue
        print(f"  {r['name']:22s} {r['base']:7.2f} {r['tip']:7.2f} {r['tmean']:+8.2f} {r['bmean']:+8.2f} "
              f"{r['length']:6.1f} {r['strain']:7.2f} {r['bend']:6.1f} {r['drift']:9.2e} "
              f"{r['per']:7.3f} {r['cmd']:6.3f}")
    print(f"\n  want: tip near the target, t.mean and b.mean near 0, len% near 100, strain < 1,")
    print(f"        drift tiny, period == cmd.")
