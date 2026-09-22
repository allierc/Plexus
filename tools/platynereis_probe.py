#!/usr/bin/env python
"""Vary one number of a spec across several short runs, and report what each one did.

    python tools/platynereis_probe.py plat_r17_anchored torque 0.02 0.05 0.2 0.5
    python tools/platynereis_probe.py plat_r17_anchored duty 0.15 0.30 0.50 --frames 200

WHY A PROBE AND NOT A GUESS. The blade has a handful of numbers that decide whether it beats and
what the beat costs -- the torque, the beat rate, the power/recovery asymmetry, the anchor, the
blade's own shape -- and they trade against each other THROUGH THE FLUID, which is not something
to reason out from a formula when a 120-frame run takes forty seconds. Each variant is a spec on
disk, run through the ordinary pipeline, and measured by the same code that measures every rung.

THE COLUMN THAT DECIDES IT IS `water/drive`, not `water`. Turning the torque up moves more water
and that proves nothing -- it is supposed to. The objective is motion per unit of drive, so the
water momentum is divided by tau * omega * n_blades * t, which is proportional to the work the
command made available. Two variants at different torques are then comparable, even though the
constant in front of that product is not known.

AND EVERY ROW CARRIES THE INVARIANTS. A variant that moves beautifully while the body inflates or
the momentum stops cancelling has not found a better cilium, it has found a worse bug, and a table
without those columns would rank it first.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))
OUT = os.path.join(REPO, "config", "platynereis")
UM = 195.9

# where each knob lives, so the probe edits one place rather than a scattered set
KNOBS = {
    "torque":  ("op", "cilium_pose_map", "sweep_deg"),
    "omega":   ("op", "cilium_pose_map", "omega"),
    "duty":    ("op", "cilium_pose_map", "duty"),
    "anchor":  ("op", "cilium_anchor", "omega_n"),
    "n_react": ("op", "cilium_torque", "n_react"),
    "youngs":  ("type", "cilium_point", "youngs"),
    "eta":     ("op", "mpm_viscosity", "eta"),
}


def variant(base: dict, knob: str, v: float, name: str, frames: int) -> dict:
    d = copy.deepcopy(base)
    d["general"]["name"] = name
    d["general"]["n_frames"] = frames
    d["plotting"]["max_frames"] = frames
    kind, target, key = KNOBS[knob]
    if kind == "op":
        for o in d["operators"]:
            if o.get("op") == target:
                o[key] = v
    else:
        for t in d["sets"][target]["types"].values():
            t[key] = v
    return d


def score(name: str, n_across: int, dt: float) -> dict:
    from platynereis_momentum import load, masses, measure
    tr, sp = load(name)
    d = yaml.safe_load(open(sp))
    ops = {o.get("op"): o for o in d["operators"]}
    tau = float(ops.get("cilium_pose_map", {}).get("sweep_deg", 0.0))
    om = float(ops.get("cilium_pose_map", {}).get("omega", 0.0))

    P = np.asarray(tr["cilium_point__pos"])
    T, N, _ = P.shape
    n_c = int(np.asarray(tr["cilium_point__parent"]).max()) + 1
    per = N // n_c
    Q = P.reshape(T, n_c, per // n_across, n_across, 3)
    root, tip = Q[:, :, 0, :, :].mean(2), Q[:, :, -1, :, :].mean(2)
    B = np.asarray(tr["body_point__pos"])
    k = np.linalg.norm(B[0][None] - root[0][:, None], axis=2).argmin(1)
    sep = np.linalg.norm(root - B[:, k, :], axis=2) * UM
    t0 = int(0.25 * T)
    et = np.linalg.norm(tip[t0:].max(0) - tip[t0:].min(0), axis=1) * UM
    er = np.linalg.norm(root[t0:].max(0) - root[t0:].min(0), axis=1) * UM

    m_of = masses(sp)
    st = measure(tr, m_of, dt)
    W = np.asarray(tr["water_particle__pos"])
    occ = np.asarray(tr["water_particle__occ"])[0].astype(bool)
    vw = np.diff(W[:, occ], axis=0) / dt
    p_w = float(np.median(np.linalg.norm(m_of["water_particle"] * vw.sum(1), axis=1)))
    com = B.mean(1)
    swim = float(np.linalg.norm(com[-1] - com[0]) * UM / (T * dt))
    drive = max(abs(tau) * abs(om) * n_c * (T * dt), 1e-30)
    return {
        "lost": int((sep[-1] > 25).sum()), "ratio": float(np.median(et / np.maximum(er, 1e-9))),
        "water": float(np.median(np.linalg.norm(vw, axis=2).mean(0)) * UM),
        "p_w_per_drive": p_w / drive, "swim": swim, "swim_per_drive": swim / drive,
        "body": float(100 * st["radius"][-1] / st["radius"][0]),
        "angle": float(np.median(st["angle"][T // 8:])),
        "cancel": float(np.median(np.linalg.norm(st["total"], axis=1)
                                  / np.maximum(np.linalg.norm(st["body_p"], axis=1)
                                               + np.linalg.norm(st["water_p"], axis=1), 1e-30))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("knob", choices=sorted(KNOBS))
    ap.add_argument("values", nargs="+", type=float)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--n-across", type=int, default=7)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    base = yaml.safe_load(open(os.path.join(OUT, a.base + ".yaml")))
    rows = []
    for v in a.values:
        nm = f"pb_{a.knob}_{str(v).replace('.', 'p').replace('-', 'm')}"
        with open(os.path.join(OUT, nm + ".yaml"), "w") as fh:
            fh.write(f"# probe: {a.knob} = {v} on {a.base}, {a.frames} frames. "
                     f"Written by tools/platynereis_probe.py\n")
            yaml.safe_dump(variant(base, a.knob, v, nm, a.frames), fh,
                           sort_keys=False, width=110)
        print(f"  running {a.knob} = {v} ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "Plexus_Main.py"),
                            "-o", "generate", f"platynereis/{nm}",
                            "--device", a.device, "--no-describe", "--no-viz"],
                           capture_output=True, text=True, cwd=REPO, timeout=7200)
        try:
            rows.append((v, score(nm, a.n_across, a.dt)))
        except Exception as e:                                       # noqa: BLE001
            print(f"    FAILED {type(e).__name__}: {str(e)[:150]}\n    {(r.stderr or '')[-300:]}")

    print(f"\n  {a.knob:>9s} {'lost':>5s} {'tip/root':>9s} {'water':>7s} {'water/drive':>12s} "
          f"{'swim':>7s} {'swim/drive':>11s} {'body%':>6s} {'angle':>6s} {'cancel':>7s}")
    for v, m in rows:
        print(f"  {v:9g} {m['lost']:5d} {m['ratio']:9.2f} {m['water']:7.3f} "
              f"{m['p_w_per_drive']:12.3e} {m['swim']:7.2f} {m['swim_per_drive']:11.3e} "
              f"{m['body']:6.0f} {m['angle']:6.1f} {m['cancel']:7.2f}")
    print(f"\n  lost = blades ending >25 um from their anchor (of 74); tip/root ~1 means carried, "
          f"not beating\n  water um/s, swim um/s; /drive divides by tau*omega*n*t, the work the "
          f"command made available\n  body% is 100 if the animal is still one body; angle wants "
          f"180; cancel wants 0 (1 = the scene drifts as one)")
    if rows:
        b = max(rows, key=lambda kv: kv[1]["swim_per_drive"])
        print(f"\n  best motion per unit drive: {a.knob} = {b[0]:g} "
              f"({b[1]['swim_per_drive']:.3e} um/s per unit, {b[1]['lost']} blades lost, "
              f"body {b[1]['body']:.0f}%)")


if __name__ == "__main__":
    main()
