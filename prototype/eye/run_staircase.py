"""run_staircase -- long-hold amplitude sweeps for plant identification.

    python run_staircase.py --model F --muscles LR MR SR IR --device cuda:0

One run per muscle: a descending staircase 1.00 / 0.75 / 0.50 / 0.25 / tonic, each
level held 2.0 s, with no return to rest between levels.

WHY 2.0 s. The fitted plant is omega_n = 9.8 rad/s, zeta = 0.26, so settling to 2%
takes 4/(zeta.omega_n) = 1.28 s. Every hold in this archive is shorter than that --
1.27 s at best, 0.54 s in t36-t43, 0.10 s in the tour -- so no steady state has ever
been measured and every static gain rests on an endpoint that was still moving. 2.0 s
is about 1.5 settling times.

WHY A SWEEP. Full-on steps pin the activation-to-torque map at u = +-1 and leave it
free in between, which is precisely where the circuit spends its time. The curvature
that separates a real muscle from a linear gain is only visible at intermediate
amplitudes.

WHY DESCENDING, IN ONE RUN. Each transition is also a step DOWN, and those transients
carry omega_n and zeta -- so a single run feeds both halves of the joint fit and
nothing is paid for twice.

WHAT IS WRITTEN, per muscle, into the model's archive:

    <M>_<MUSCLE>_stair_curves.npz   frame, t, cmd [n,6], cmd_probed [n], act [n,6],
                                    gaze [n,3] (horizontal, vertical, TORSION), length,
                                    centre, rest_length
    <M>_<MUSCLE>_stair.json         the level windows and, per level, the SETTLED gaze
                                    -- the mean over the last quarter of the hold, with
                                    its spread, which is the static-gain point Phi wants
                                    and the number no previous run could supply

Torsion is recorded because it is what tests the two-independent-axes assumption; if
the obliques carry vertical gain it will show up there.
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
from plexus.schema import load as load_spec

ARCHIVE = os.path.join(HERE, "archive")
ESSENTIAL = ["LR", "MR", "SR", "IR"]           # what the circuit drives
OPTIONAL = ["SO", "IO"]                        # complete the 3x6 gain matrix


def base_spec(model):
    d = os.path.join(ARCHIVE, f"eye_{model}")
    p = os.path.join(d, "baseline_spec.yaml")
    if not os.path.exists(p):
        import glob
        c = sorted(glob.glob(os.path.join(d, "*_spec.yaml")))
        if not c:
            raise FileNotFoundError(f"no spec in {d}")
        p = c[0]
    with open(p) as fh:
        return yaml.safe_load(fh), p


def settled(t, y, t_on, t_off, frac=0.25):
    """Mean and spread of `y` over the LAST `frac` of the hold [t_on, t_off].

    Averaging the tail rather than taking the endpoint is what makes this a static
    measurement: the endpoint of a hold that is still ringing is not a steady state,
    and the spread reported alongside says whether this one had settled.
    """
    lo = t_off - frac * (t_off - t_on)
    m = (t >= lo) & (t <= t_off)
    if m.sum() < 2:
        m = t >= lo
    return y[m].mean(0), y[m].std(0), y[m].max(0) - y[m].min(0)


def stair(model, muscle_key, device="cuda:0", levels=(1.0, 0.75, 0.50, 0.25),
          hold_s=2.0, lead_s=0.5, tail_s=2.0, stride=6):
    d = os.path.join(ARCHIVE, f"eye_{model}")
    spec, src = base_spec(model)
    dt = float(spec["general"]["dt"])
    mi = EA.MUSCLE_KEYS.index(muscle_key)
    drv = next((o for o in spec["operators"]
                if o["op"] in ("oculomotor_drive", "muscle_probe")), {})
    tonic = float(drv.get("tonic", 0.14))
    hold, lead, tail = (int(round(s / dt)) for s in (hold_s, lead_s, tail_s))
    s_spec = probe_ops.staircase_spec(spec, mi, levels=levels, hold=hold, lead=lead,
                                      tail=tail, tonic=tonic)
    path = os.path.join(d, f"{model}_{muscle_key}_stair_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# model {model} -- descending staircase on {muscle_key}, built from "
                 f"{os.path.basename(src)}\n"
                 f"# levels {list(levels)} held {hold_s} s each "
                 f"({hold} frames at dt={dt}); settling time of the fitted plant is 1.28 s\n")
        yaml.safe_dump(s_spec, fh, sort_keys=False, width=100)

    sim = load_spec(path)
    prb = probe_ops.MuscleProbeStaircase({"muscle": mi, "levels": list(levels), "hold": hold,
                                          "lead": lead, "tail": tail, "tonic": tonic})
    n = prb.n_frames()
    print(f"[{model}/{muscle_key}] staircase {list(levels)} x {hold_s}s -> {n} frames "
          f"({n * dt:.1f} s sim), stride {stride} ({1e3 * stride * dt:.0f} ms)", flush=True)
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, device, stride=stride)
    frames = np.asarray(cap["frame"])
    t = frames * dt
    out = dict(frame=frames, t=t.astype(np.float32),
               gaze=np.asarray(cap["gaze"], np.float32),
               act=np.asarray(cap["act"], np.float32),
               length=np.asarray(cap["length"], np.float32),
               centre=np.asarray(cap["centre"], np.float32),
               rest_length=np.asarray(cap["rest_length"], np.float32),
               cmd=np.stack([prb.levels_all(f) for f in frames]).astype(np.float32),
               cmd_probed=np.array([prb.level(f) for f in frames], np.float32))
    np.savez_compressed(os.path.join(d, f"{model}_{muscle_key}_stair_curves.npz"), **out)

    g = out["gaze"]
    base_mean, _, _ = settled(t, g, 0.0, lead * dt, frac=0.5)      # the pre-step baseline
    per = []
    for lv, f_on, f_off in prb.windows():
        mean, sd, ptp = settled(t, g, f_on * dt, f_off * dt)
        act_mean, _, _ = settled(t, out["act"][:, mi][:, None], f_on * dt, f_off * dt)
        per.append(dict(level=lv, window_s=[round(f_on * dt, 3), round(f_off * dt, 3)],
                        act_settled=round(float(act_mean[0]), 4),
                        gaze_settled_deg=[round(float(v), 4) for v in mean],
                        gaze_minus_baseline_deg=[round(float(v), 4) for v in (mean - base_mean)],
                        sd_deg=[round(float(v), 4) for v in sd],
                        peak_to_peak_deg=[round(float(v), 4) for v in ptp]))
    meta = dict(model=model, muscle=muscle_key, muscle_index=mi, built_from=os.path.basename(src),
                dt=dt, stride=stride, sample_ms=round(1e3 * stride * dt, 1),
                levels=list(levels), hold_s=hold_s, lead_s=lead_s, tail_s=tail_s,
                n_frames=n, tonic=tonic, seconds=round(time.time() - t0, 1),
                baseline_gaze_deg=[round(float(v), 4) for v in base_mean],
                settling_time_s=1.28, per_level=per)
    with open(os.path.join(d, f"{model}_{muscle_key}_stair.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{model}/{muscle_key}] done in {meta['seconds']}s", flush=True)
    print("    %-6s %8s  %-28s %s" % ("level", "act", "settled gaze - baseline (h,v,t)",
                                      "p-p during the tail (h,v,t)"))
    for r in per:
        print("    %-6.2f %8.3f  %-28s %s" % (r["level"], r["act_settled"],
                                              r["gaze_minus_baseline_deg"],
                                              r["peak_to_peak_deg"]), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="F")
    ap.add_argument("--muscles", nargs="*", default=ESSENTIAL)
    ap.add_argument("--levels", type=float, nargs="*", default=[1.0, 0.75, 0.50, 0.25])
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--lead-s", type=float, default=0.5)
    ap.add_argument("--tail-s", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    done = []
    for k in a.muscles:
        try:
            done.append(stair(a.model, k, a.device, tuple(a.levels), a.hold_s, a.lead_s,
                              a.tail_s, a.stride))
        except Exception as e:
            print(f"[{a.model}/{k}] FAILED: {type(e).__name__}: {e}", flush=True)
    if done:
        # MERGE, never overwrite: muscles are run in separate invocations (a failure, a
        # machine to share), and a summary that silently dropped the muscles this call
        # did not touch would be worse than no summary.
        p = os.path.join(ARCHIVE, f"eye_{a.model}", f"{a.model}_staircase_summary.json")
        prev = []
        if os.path.exists(p):
            try:
                prev = json.load(open(p))
            except Exception:
                prev = []
        by = {m["muscle"]: m for m in prev if isinstance(m, dict) and "muscle" in m}
        by.update({m["muscle"]: m for m in done})
        merged = [by[k] for k in EA.MUSCLE_KEYS if k in by]
        with open(p, "w") as fh:
            json.dump(merged, fh, indent=2)
        print("summary ->", p, flush=True)


if __name__ == "__main__":
    main()
