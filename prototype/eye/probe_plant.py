#!/usr/bin/env python
"""probe_plant -- Phase 2: measure the oculomotor plant by open-loop step injection.

    python probe_plant.py archive/t03_c_a --muscles 0 1 2 --device cuda:0
    python probe_plant.py archive/t03_c_a --fit          # fit + verdict from finished probes

THE QUESTION. Is the standing gaze error a CONTROL problem or a MECHANICAL one? Phase 1a asserted
an answer from a closed-loop fit whose surrogate failed its own fidelity check. This measures it.

THE METHOD. Open the loop (`muscle_probe`), step one muscle at a time, and read the STATIC GAIN off
the plateau:

    G[k][m] = ( theta_k during the hold  -  theta_k before the step ) / ( a_hi - tonic )

in degrees per unit activation. Six runs give the full 3x6 matrix. Extrapolated to a = 1 this is the
most that muscle can turn the eye, on its own, with the antagonists at rest -- which settles the
question directly, because the commands the eye is failing to reach are known.

THE PREDICTION, registered in eye_note.pdf before running:
    LR static gain on t03_c_a lands in 8-16 deg, i.e. BELOW the 26 deg it is commanded to reach
    => mechanically incapable, and no PID retune can close it.
    Falsifier: >= 26 deg => the plant has the authority, the loop is throwing it away, and
    Phase 1a's conclusion was wrong.
    Secondary: LR/MR nearly pure direct-axis (off-axis < 15%); SR/IR/SO/IO substantially coupled.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops
import eye_anatomy as EA
from plexus.schema import load as load_spec
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive", "phase2_stepresponse")


# --------------------------------------------------------------------------- #
#  run one probe
# --------------------------------------------------------------------------- #
def run_probe(base_spec, muscle, device, a_hi, tonic, t_on, t_off, n_frames, outdir):
    spec_d = probe_ops.probe_spec(base_spec, muscle, a_hi=a_hi, tonic=tonic,
                                  t_on=t_on, t_off=t_off, n_frames=n_frames)
    key = EA.MUSCLE_KEYS[muscle] if muscle >= 0 else "null"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"spec_{key}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(spec_d, f, sort_keys=False, width=100)
    sim = load_spec(path)

    rec = {"frame": [], "gaze": [], "act": [], "length": [], "centre": [], "cmd": []}

    def hook(H, frame):
        m, eye = H.levels["muscle"], H.levels["eye"]
        f32 = lambda t: t.detach().cpu().numpy().astype(np.float32)
        rec["frame"].append(frame)
        rec["gaze"].append(f32(eye.get("gaze")[0]))
        rec["act"].append(f32(m.get("act")[:, 0]))
        rec["length"].append(f32(m.get("length")[:, 0]))
        rec["centre"].append(f32(eye.get("pos")[0]))
        rec["cmd"].append(np.float32(prb.level(frame)) if muscle >= 0 else np.float32(tonic))

    prb = probe_ops.MuscleProbe({"muscle": muscle, "a_hi": a_hi, "tonic": tonic,
                                 "t_on": t_on, "t_off": t_off})
    print(f"[probe] {key}: step {tonic}->{a_hi} at frame {t_on}, release {t_off}, "
          f"{n_frames} frames", flush=True)
    H, _ = engine_run(sim, out_path=None, device=device, on_frame=hook, progress=False)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["rest_length"] = H.levels["muscle"].rest_length.detach().cpu().numpy()
    np.savez_compressed(os.path.join(outdir, f"probe_{key}.npz"), **out)
    g = out["gaze"]
    print(f"[probe] {key}: gaze range h/v/t = "
          f"{np.round(np.ptp(g, axis=0), 2)}  final {np.round(g[-1], 2)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
#  fit: static gain, rise time, overshoot
# --------------------------------------------------------------------------- #
def fit_probe(d, t_on, t_off, dt):
    """Static gain (deg per unit activation) + first-order dynamics, from one step response."""
    fr, g, a = d["frame"], d["gaze"], d["act"]
    pre = (fr >= t_on - 30) & (fr < t_on)                      # baseline before the step
    hold = (fr >= t_on + 0.65 * (t_off - t_on)) & (fr < t_off)  # settled tail of the hold
    if pre.sum() < 3 or hold.sum() < 3:
        return None
    theta0 = g[pre].mean(0)
    theta1 = g[hold].mean(0)
    d_theta = theta1 - theta0
    # the achieved activation change, measured -- not the commanded one, since the first-order
    # activation dynamics may not have fully settled
    m = int(d.get("_muscle", 0))
    d_act = float(a[hold, m].mean() - a[pre, m].mean())
    gain = d_theta / max(abs(d_act), 1e-6)

    # dominant axis: rise time to 63% and peak overshoot
    k = int(np.argmax(np.abs(d_theta)))
    seg = (fr >= t_on) & (fr < t_off)
    tr = fr[seg].astype(float)
    yr = g[seg, k] - theta0[k]
    target = d_theta[k]
    t63 = np.nan
    if abs(target) > 1e-9:
        idx = np.nonzero(np.abs(yr) >= 0.632 * abs(target))[0]
        if idx.size:
            t63 = float((tr[idx[0]] - t_on) * dt)
    peak = float(yr[np.argmax(np.abs(yr))]) if yr.size else 0.0
    overshoot = float(100.0 * (abs(peak) - abs(target)) / max(abs(target), 1e-9))
    return {"d_act": d_act, "gain_deg_per_act": [float(x) for x in gain],
            "delta_theta_deg": [float(x) for x in d_theta],
            "dominant_axis": ["horizontal", "vertical", "torsion"][k],
            "t63_s": t63, "overshoot_pct": round(overshoot, 1),
            "offaxis_frac": float(np.linalg.norm(np.delete(gain, k)) / max(abs(gain[k]), 1e-9))}


def report(outdir, t_on, t_off, dt, command_deg=26.0):
    rows, G = {}, np.zeros((3, EA.N_MUSCLE))
    for m, key in enumerate(EA.MUSCLE_KEYS):
        f = os.path.join(outdir, f"probe_{key}.npz")
        if not os.path.exists(f):
            continue
        d = dict(np.load(f))
        d["_muscle"] = m
        r = fit_probe(d, t_on, t_off, dt)
        if r is None:
            continue
        rows[key] = r
        G[:, m] = r["gain_deg_per_act"]

    print("\n" + "=" * 92)
    print("STATIC GAIN MATRIX  (degrees of gaze per unit activation, open loop)")
    print(f"{'':12}" + "".join(f"{k:>12}" for k in EA.MUSCLE_KEYS))
    for k, nm in enumerate(("horizontal", "vertical", "torsion")):
        print(f"{nm:12}" + "".join(f"{G[k, m]:12.2f}" for m in range(EA.N_MUSCLE)))
    print("-" * 92)
    print(f"{'dominant':12}" + "".join(f"{rows.get(k, {}).get('dominant_axis', '-')[:11]:>12}"
                                      for k in EA.MUSCLE_KEYS))
    print(f"{'off-axis':12}" + "".join(f"{rows.get(k, {}).get('offaxis_frac', float('nan')):12.2f}"
                                      for k in EA.MUSCLE_KEYS))
    print(f"{'t63 (s)':12}" + "".join(f"{rows.get(k, {}).get('t63_s', float('nan')):12.3f}"
                                     for k in EA.MUSCLE_KEYS))
    print(f"{'overshoot%':12}" + "".join(f"{rows.get(k, {}).get('overshoot_pct', float('nan')):12.1f}"
                                        for k in EA.MUSCLE_KEYS))
    print("=" * 92)

    verdict = None
    if "LR" in rows:
        # extrapolate the measured gain to full activation, from tonic
        lr = rows["LR"]
        reach = abs(lr["gain_deg_per_act"][0]) * (1.0 - 0.14)
        pred_lo, pred_hi = 8.0, 16.0
        inside = pred_lo <= reach <= pred_hi
        control_limited = reach >= command_deg
        verdict = {"lr_reach_deg_at_full_activation": round(float(reach), 2),
                   "command_deg": command_deg,
                   "prediction_band_deg": [pred_lo, pred_hi],
                   "prediction_held": bool(inside),
                   "conclusion": "control-limited" if control_limited else "mechanics-limited"}
        print(f"\nLR at full activation reaches {reach:.1f} deg against a {command_deg:.0f} deg "
              f"command.")
        print(f"  registered prediction was {pred_lo:.0f}-{pred_hi:.0f} deg: "
              f"{'HELD' if inside else 'BROKEN'}")
        if control_limited:
            print("  VERDICT: CONTROL-limited. The plant has the authority and the loop is "
                  "throwing it away.\n           Phase 1a's conclusion was wrong; retune the PID.")
        else:
            print("  VERDICT: MECHANICS-limited. This configuration cannot reach the command at "
                  "any gain.\n           The fix is the plant (the A/E pulley ceiling), not the "
                  "controller.")

    out = {"gain_matrix_deg_per_act": G.tolist(), "per_muscle": rows, "verdict": verdict,
           "muscles": EA.MUSCLE_KEYS}
    with open(os.path.join(outdir, "plant.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[phase2] -> {os.path.join(outdir, 'plant.json')}", flush=True)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="archived run whose spec.yaml is the baseline (e.g. archive/t03_c_a)")
    ap.add_argument("--muscles", type=int, nargs="*", default=list(range(EA.N_MUSCLE)))
    ap.add_argument("--a_hi", type=float, default=1.0)
    ap.add_argument("--tonic", type=float, default=None, help="default: the baseline's own tonic")
    ap.add_argument("--t_on", type=int, default=60)
    ap.add_argument("--t_off", type=int, default=240)
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--fit", action="store_true", help="skip the runs, fit what is already there")
    a = ap.parse_args()

    base = yaml.safe_load(open(os.path.join(a.base, "spec.yaml")))
    drv = next(o for o in base["operators"] if o["op"] == "oculomotor_drive")
    tonic = a.tonic if a.tonic is not None else float(drv.get("tonic", 0.14))
    dt = float(base["general"]["dt"])

    if not a.fit:
        for m in a.muscles:
            run_probe(base, m, a.device, a.a_hi, tonic, a.t_on, a.t_off, a.frames, a.outdir)
    report(a.outdir, a.t_on, a.t_off, dt)


if __name__ == "__main__":
    main()
