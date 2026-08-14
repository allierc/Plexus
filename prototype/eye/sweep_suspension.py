"""sweep_suspension -- what is holding eye F's globe still?

    python sweep_suspension.py --tonic 0.02 --device cuda:0

LR shortens 31% and turns the eye 6 degrees. It is not failing to contract, and its
moment arm is not the problem either -- F's arms are LARGER than the mammalian plant's
(LR 107 um against 69). So the contraction is being absorbed rather than converted, and
there are three places it can go:

  tonic     the five muscles that are not being driven still sit at `tonic` = 0.14.
            Five straps pulling back is the most direct way to lose travel, and it is
            the only one of the three that a real animal also has.
  k_fat     the orbital-fat suspension. The prototype's claim is that a UNIFORM
            restoring body force exerts no torque about the centroid, so it recentres
            the eye without resisting gaze -- an argument that deserves a measurement
            rather than a citation of itself.
  k_socket  the penalty contact with the bony cup. The cup clears the globe by 0.007,
            so it should only engage on translation -- unless the globe is riding on it.

One at a time from the baseline, because the question is WHICH, not how they combine;
one combined cell at the end says whether the effects add. Every cell is LR at full
drive held past settling, and reports the settled pose, whether it settled, and the
globe's radius drift as an integrity guard.
"""
from __future__ import annotations
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators, eye_ops, muscle_ops, probe_ops   # noqa: F401
import eye_anatomy as EA, run_eye, render_eye_vtk
from plexus.schema import load as load_spec
from run_staircase import base_spec, settled

OUT = os.path.join(HERE, "archive", "suspension")


def run(model, tonic, k_fat, k_socket, c_fat=None, muscle="LR", device="cuda:0",
        hold_s=2.0, lead_s=0.3, tail_s=0.2, stride=6, substep=2.0e-4, movie=True, tag=None):
    os.makedirs(OUT, exist_ok=True)
    spec, src = base_spec(model)
    dt = float(spec["general"]["dt"]); mi = EA.MUSCLE_KEYS.index(muscle)
    hold, lead, tail = (int(round(s / dt)) for s in (hold_s, lead_s, tail_s))
    s = probe_ops.staircase_spec(spec, mi, levels=(1.0,), hold=hold, lead=lead, tail=tail,
                                 tonic=float(tonic))
    for o in s["operators"]:
        if o["op"] == "orbit_socket":
            o["k"] = float(k_socket); o["k_fat"] = float(k_fat)
            if c_fat is not None:
                o["c_fat"] = float(c_fat)
    # the substep validated by derisk_tests: 2.0e-4 moves the settled pose by 0.007 deg
    # against a 0.05 deg tolerance and costs 2.6x less
    for step in s["schedule"]:
        if isinstance(step, dict) and "substep_dt" in step:
            step["substep_dt"] = float(substep)
    tag = tag or f"tonic{tonic:g}_fat{k_fat:g}_socket{k_socket:g}"
    name = f"{model}_{muscle}_{tag}"
    path = os.path.join(OUT, f"{name}_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# suspension sweep: tonic {tonic}, k_fat {k_fat}, k_socket {k_socket}\n")
        yaml.safe_dump(s, fh, sort_keys=False, width=100)
    sim = load_spec(path)
    print(f"[{tag}] {muscle} u=1, hold {hold_s}s", flush=True)
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, device, stride=stride)
    t = np.asarray(cap["frame"]) * dt; g = np.asarray(cap["gaze"])
    base, _, _ = settled(t, g, 0.0, lead * dt, frac=0.5)
    mean, sd, ptp = settled(t, g, lead * dt, (lead + hold) * dt)
    ln = np.asarray(cap["length"])[:, mi]
    rest = np.asarray(cap["rest_length"]); rest = float(rest[0][mi] if np.ndim(rest) > 1 else rest[mi])
    rad = np.asarray(cap["radius"]) if "radius" in cap else np.zeros(len(t))
    res = dict(model=model, muscle=muscle, tonic=float(tonic), k_fat=float(k_fat),
               k_socket=float(k_socket), tag=tag,
               pose_deg=[round(float(v), 3) for v in (mean - base)],
               settle_ptp_deg=[round(float(v), 4) for v in ptp],
               settled=bool(max(ptp) <= 0.05),
               shortening_pct=round(100.0 * (1.0 - float(np.median(ln[-max(len(ln)//5,1):])) / rest), 2),
               radius_drift_pct=round(100.0 * float(rad[-1] / max(rad[0], 1e-9) - 1.0), 3),
               seconds=round(time.time() - t0, 1))
    with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    if movie:
        render_eye_vtk.render(cap, dt, os.path.join(OUT, f"{name}.mp4"),
                              os.path.join(OUT, f"{name}_strip.png"))
    print(f"[{tag}] pose {res['pose_deg']}  settled={res['settled']}  "
          f"shortening {res['shortening_pct']}%  radius {res['radius_drift_pct']}%  "
          f"[{res['seconds']}s]", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="F"); ap.add_argument("--muscle", default="LR")
    ap.add_argument("--tonic", type=float, default=0.14)
    ap.add_argument("--k_fat", type=float, default=4000.0)
    ap.add_argument("--k_socket", type=float, default=5000.0)
    ap.add_argument("--c_fat", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()
    run(a.model, a.tonic, a.k_fat, a.k_socket, a.c_fat, a.muscle, a.device,
        movie=not a.no_movie, tag=a.tag)
