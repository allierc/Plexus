"""sweep_gap_span -- does the sclera STAND-OFF explain eye F's missing span?

    python sweep_gap_span.py --model F --gap 0.042 --device cuda:0

The muscle and the globe are separate bodies that couple only through the shared
MLS-MPM grid. A strap lying hard against the sclera is welded to it along its whole
arc of contact, so shortening drags the surface locally instead of winding the globe
round; it has to ride clear and grip at the tendon. `gap` is that clearance, and it
is the parameter that took eye B to eye C: 0.020 -> 0.042, and travel 3.4 -> 15.0 deg.

Eye F sits at 0.0161 -- tighter than B -- because the measured fish straps are a third
the thickness of the mammalian ones and a 0.042 stand-off left them floating in
mid-orbit. This runs one value, LR alone at full drive, held past settling, and
reports the settled pose. The point is to replace an inference with a measurement.
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

OUT = os.path.join(HERE, "archive", "gap_sweep")


def run(model, gap, muscle="LR", device="cuda:0", hold_s=2.0, lead_s=0.3, tail_s=0.2,
        stride=6, movie=True):
    os.makedirs(OUT, exist_ok=True)
    spec, src = base_spec(model)
    dt = float(spec["general"]["dt"]); mi = EA.MUSCLE_KEYS.index(muscle)
    tonic = float(next((o for o in spec["operators"]
                        if o["op"] in ("oculomotor_drive", "muscle_probe")), {}).get("tonic", 0.14))
    hold, lead, tail = (int(round(s / dt)) for s in (hold_s, lead_s, tail_s))
    s = probe_ops.staircase_spec(spec, mi, levels=(1.0,), hold=hold, lead=lead, tail=tail,
                                 tonic=tonic)
    for o in s["operators"]:
        if o["op"] == "muscle_morphogenesis":
            o["gap"] = float(gap)
    tag = f"{model}_{muscle}_gap{gap:g}"
    path = os.path.join(OUT, f"{tag}_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# stand-off sweep: gap {gap} (F ships 0.0161; C, which spans 30 deg, is 0.042)\n")
        yaml.safe_dump(s, fh, sort_keys=False, width=100)
    sim = load_spec(path)
    print(f"[gap {gap:g}] {muscle} at u=1, hold {hold_s}s", flush=True)
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, device, stride=stride)
    t = np.asarray(cap["frame"]) * dt; g = np.asarray(cap["gaze"])
    base, _, _ = settled(t, g, 0.0, lead * dt, frac=0.5)
    mean, sd, ptp = settled(t, g, lead * dt, (lead + hold) * dt)
    ln = np.asarray(cap["length"])[:, mi]
    rest = np.asarray(cap["rest_length"]); rest = float(rest[0][mi] if np.ndim(rest) > 1 else rest[mi])
    rad = np.asarray(cap["radius"]) if "radius" in cap else np.zeros(len(t))
    res = dict(model=model, muscle=muscle, gap=gap,
               pose_deg=[round(float(v), 3) for v in (mean - base)],
               settle_ptp_deg=[round(float(v), 4) for v in ptp],
               settled=bool(max(ptp) <= 0.05),
               shortening_pct=round(100.0 * (1.0 - float(np.median(ln[-max(len(ln)//5,1):])) / rest), 2),
               radius_drift_pct=round(100.0 * float(rad[-1] / max(rad[0], 1e-9) - 1.0), 3),
               seconds=round(time.time() - t0, 1))
    with open(os.path.join(OUT, f"{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    if movie:
        render_eye_vtk.render(cap, dt, os.path.join(OUT, f"{tag}.mp4"),
                              os.path.join(OUT, f"{tag}_strip.png"))
    print(f"[gap {gap:g}] pose {res['pose_deg']} deg  settled={res['settled']}  "
          f"shortening {res['shortening_pct']}%  radius drift {res['radius_drift_pct']}%  "
          f"[{res['seconds']}s]", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="F"); ap.add_argument("--muscle", default="LR")
    ap.add_argument("--gap", type=float, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()
    run(a.model, a.gap, a.muscle, a.device, movie=not a.no_movie)
