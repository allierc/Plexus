"""run_synergies -- drive an eye's SYNERGIES in turn and ask where the eye actually goes.

    python run_synergies.py --model G --device cuda:0

Four phases, each held past settling and released:

    LR + SO   the two the colours call blue and violet   -> expected UP
    IR + IO   green and orange                           -> expected DOWN
    LR        blue alone                                 -> expected RIGHT
    MR        yellow alone                               -> expected LEFT

No single extraocular muscle moves the eye along a cardinal axis, so a per-muscle tour
cannot answer "does this geometry work". A synergy can: the expected direction is
written down BEFORE the run, and each phase either produces it or does not.

The elevator pairing is not the mammalian one. On the traced fish plant SO elevates and
IO depresses -- both obliques pull from the ROSTRAL orbit with no trochlea to reverse
the superior one -- so LR+SO is the elevating pair here and IR+IO the depressing one.
The run tests exactly that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops
import eye_anatomy as EA
import run_eye
import render_eye_vtk
from plexus.schema import load as load_spec
from run_staircase import base_spec, settled

try:                                # eye_G's geometry comes from the Blender model
    import blend_mpm_ops           # noqa: F401
except Exception as e:             # a plant that does not need it still runs
    print(f"[synergies] blend_mpm_ops not loaded ({type(e).__name__}); "
          f"only specs that do not use blend_* will build", flush=True)

ARCHIVE = os.path.join(HERE, "archive")
K = {k: i for i, k in enumerate(EA.MUSCLE_KEYS)}
# (name, muscles, the axis and sign expected, in the (h, v, t) readout)
PHASES = [("up", ["LR", "SO"], (1, +1)),
          ("down", ["IR", "IO"], (1, -1)),
          ("right", ["LR"], (0, +1)),
          ("left", ["MR"], (0, -1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G")
    ap.add_argument("--hold", type=int, default=500)
    ap.add_argument("--rest", type=int, default=300)
    ap.add_argument("--lead", type=int, default=100)
    ap.add_argument("--tail", type=int, default=250)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    d = os.path.join(ARCHIVE, f"eye_{a.model}")
    spec, src = base_spec(a.model)
    dt = float(spec["general"]["dt"])
    tonic = float(next((o for o in spec["operators"]
                        if o["op"] in ("oculomotor_drive", "muscle_probe")), {}).get("tonic", 0.14))
    groups = [[K[m] for m in ms] for _, ms, _ in PHASES]
    s = probe_ops.groups_spec(spec, groups, hold=a.hold, rest=a.rest, lead=a.lead,
                              tail=a.tail, tonic=tonic)
    path = os.path.join(d, f"{a.model}_synergies_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# {a.model}: synergies in turn -- "
                 + ", ".join(f"{n}({'+'.join(ms)})" for n, ms, _ in PHASES) + "\n")
        yaml.safe_dump(s, fh, sort_keys=False, width=100)

    prb = probe_ops.MuscleProbeGroups({"groups": groups, "hold": a.hold, "rest": a.rest,
                                       "lead": a.lead, "tail": a.tail, "tonic": tonic})
    print(f"[{a.model}] {len(PHASES)} phases, {prb.n_frames()} frames "
          f"({prb.n_frames() * dt:.1f} s)", flush=True)
    t0 = time.time()
    sim = load_spec(path)
    _, cap = run_eye.capture_run(sim, a.device, stride=a.stride)
    t = np.asarray(cap["frame"]) * dt
    g = np.asarray(cap["gaze"])
    np.savez_compressed(os.path.join(d, f"{a.model}_synergies_curves.npz"),
                        frame=np.asarray(cap["frame"]), t=t.astype(np.float32),
                        gaze=g.astype(np.float32), act=np.asarray(cap["act"], np.float32),
                        length=np.asarray(cap["length"], np.float32),
                        rest_length=np.asarray(cap["rest_length"], np.float32),
                        centre=np.asarray(cap["centre"], np.float32),
                        muscles=np.array(EA.MUSCLE_KEYS))

    rows, base = [], None
    print("\n%-6s %-9s %-26s %-9s %s" % ("phase", "muscles", "settled (h,v,t) deg",
                                         "expected", "verdict"))
    for slot, (name, ms, (ax, sgn)) in enumerate(PHASES):
        t_on, t_off = prb.window(slot)
        pre = g[t <= t_on * dt]
        base = pre[-1] if len(pre) else g[0]
        mean, sd, ptp = settled(t, g, t_on * dt, t_off * dt)
        dg = mean - base
        got = float(dg[ax])
        ok = (got * sgn) > 0.5
        rows.append(dict(phase=name, muscles=ms, expected_axis="hvt"[ax],
                         expected_sign=sgn, gaze_deg=[round(float(v), 3) for v in dg],
                         settled=bool(max(ptp) <= 0.05), correct=bool(ok)))
        print("%-6s %-9s %-26s %-9s %s" % (name, "+".join(ms), str(rows[-1]["gaze_deg"]),
                                           f'{"hvt"[ax]}{"+" if sgn > 0 else "-"}',
                                           "YES" if ok else "no"))
    meta = dict(model=a.model, built_from=os.path.basename(src), dt=dt,
                hold_s=round(a.hold * dt, 3), n_frames=prb.n_frames(),
                seconds=round(time.time() - t0, 1),
                n_correct=sum(r["correct"] for r in rows), phases=rows)
    with open(os.path.join(d, f"{a.model}_synergies.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"\n[{a.model}] {meta['n_correct']}/{len(PHASES)} phases moved the eye the "
          f"way the anatomy predicts  [{meta['seconds']}s]", flush=True)
    if not a.no_movie:
        render_eye_vtk.render(cap, dt, os.path.join(d, f"{a.model}_synergies.mp4"),
                              os.path.join(d, f"{a.model}_synergies_strip.png"))


if __name__ == "__main__":
    main()
