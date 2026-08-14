"""sweep_buckle -- the strap absorbs 82% of its own contraction. Can that be stopped?

    python sweep_buckle.py --k_sleeve 2500 --device cuda:0
    python sweep_buckle.py --width-scale 2.0 --device cuda:0

LR shortens 31% of its rest length and delivers a sixth of it to the globe. The globe
is not translating (0.5% of a semi-axis, twenty times less than the insertion moves),
the suspension is not resisting, and the moment arm is larger than the mammalian
plant's. What is left is the muscle itself: the measured fish straps are 10-20 um wide
where the mammalian model used 34, and a slender strap under axial active stress
BUCKLES -- it folds instead of transmitting, and a folded centreline is shorter without
its endpoints having moved.

Two ways to stop that, and they are different claims about the animal:

  k_sleeve      a connective-tissue sleeve holding the strap against the globe -- the
                `muscle_sleeve` operator that eyes D and E used and F has switched off.
                The claim is that the fish has such a sheath and the model omitted it.
  width/thick   a fatter strap. Bending stiffness goes as thickness^3, so this attacks
                buckling directly -- but it also raises the force, so any span it buys
                has to be checked against the globe staying intact.

Both are measured the same way: LR at full drive, held past settling, reporting the
settled pose and -- the number that matters here -- how much of the shortening reaches
the insertion.
"""
from __future__ import annotations
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators, eye_ops, muscle_ops, probe_ops   # noqa: F401
import eye_anatomy as EA, run_eye, render_eye_vtk, fish_anatomy as FA
from plexus.schema import load as load_spec
from run_staircase import base_spec, settled

OUT = os.path.join(HERE, "archive", "buckle")
ARM = 0.0986        # LR moment arm, world units (107 um)


def run(model, k_sleeve, width_scale, thick_scale, muscle="LR", device="cuda:0", k_bone=None,
        hold_s=2.0, lead_s=0.3, tail_s=0.2, stride=6, substep=2.0e-4, movie=True, tag=None):
    os.makedirs(OUT, exist_ok=True)
    spec, src = base_spec(model)
    dt = float(spec["general"]["dt"]); mi = EA.MUSCLE_KEYS.index(muscle)
    tonic = float(next((o for o in spec["operators"]
                        if o["op"] in ("oculomotor_drive", "muscle_probe")), {}).get("tonic", 0.14))
    hold, lead, tail = (int(round(s / dt)) for s in (hold_s, lead_s, tail_s))
    s = probe_ops.staircase_spec(spec, mi, levels=(1.0,), hold=hold, lead=lead, tail=tail,
                                 tonic=tonic)
    for o in s["operators"]:
        if o["op"] == "muscle_sleeve":
            o["k"] = float(k_sleeve)
        elif o["op"] == "bone_anchor" and k_bone is not None:
            o["k"] = float(k_bone)
        elif o["op"] == "muscle_morphogenesis":
            w = o.get("width")
            o["width"] = ([float(v) * width_scale for v in w] if isinstance(w, list)
                          else float(w) * width_scale)
            o["thickness"] = float(o["thickness"]) * thick_scale
    for step in s["schedule"]:
        if isinstance(step, dict) and "substep_dt" in step:
            step["substep_dt"] = float(substep)
    tag = tag or f"sleeve{k_sleeve:g}_w{width_scale:g}_t{thick_scale:g}"
    name = f"{model}_{muscle}_{tag}"
    path = os.path.join(OUT, f"{name}_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# buckling sweep: k_sleeve {k_sleeve}, width x{width_scale}, "
                 f"thickness x{thick_scale}\n")
        yaml.safe_dump(s, fh, sort_keys=False, width=100)
    sim = load_spec(path)
    print(f"[{tag}] {muscle} u=1, hold {hold_s}s", flush=True)
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, device, stride=stride)
    t = np.asarray(cap["frame"]) * dt; g = np.asarray(cap["gaze"])
    base, _, _ = settled(t, g, 0.0, lead * dt, frac=0.5)
    mean, sd, ptp = settled(t, g, lead * dt, (lead + hold) * dt)
    # WHERE THE SHORTENING WENT. The muscle's two end caps are its origin (s < 0.08,
    # held by `bone_anchor`) and its tendon (s > 0.92, embedded in the sclera). If the
    # path shortens but the ends do not approach, the tissue is compressing internally;
    # if the ends approach but the INSERTION does not move, the origin is being dragged
    # off its bone. Those are different failures with different fixes, so both are
    # measured rather than argued about.
    Y = np.asarray(cap["mus_pos"]); par = np.asarray(cap["mus_parent"])
    sv = np.asarray(cap["mus_s"]); sel = par == mi
    org_i, ten_i = sel & (sv < 0.08), sel & (sv > 0.92)
    org = np.array([Y[k][org_i].mean(0) for k in range(len(Y))])
    ten = np.array([Y[k][ten_i].mean(0) for k in range(len(Y))])
    end_to_end = np.linalg.norm(ten - org, axis=1)
    ln = np.asarray(cap["length"])[:, mi]
    rest = np.asarray(cap["rest_length"]); rest = float(rest[0][mi] if np.ndim(rest) > 1 else rest[mi])
    short = rest - float(np.median(ln[-max(len(ln) // 5, 1):]))
    delivered = ARM * np.radians(abs(float(mean[0] - base[0])))
    rad = np.asarray(cap["radius"]) if "radius" in cap else np.zeros(len(t))
    res = dict(model=model, muscle=muscle, k_sleeve=float(k_sleeve),
               width_scale=float(width_scale), thick_scale=float(thick_scale), tag=tag,
               pose_deg=[round(float(v), 3) for v in (mean - base)],
               settle_ptp_deg=[round(float(v), 4) for v in ptp],
               settled=bool(max(ptp) <= 0.05),
               shortening_pct=round(100.0 * short / rest, 2),
               # THE NUMBER THIS SWEEP EXISTS FOR: of everything the muscle shortened by,
               # how much actually moved the insertion round the globe
               delivered_pct=round(100.0 * delivered / max(short, 1e-9), 1),
               end_to_end_shortening_pct=round(float(100.0 * (1 - end_to_end[-1] / end_to_end[0])), 2),
               origin_cap_moved=round(float(np.linalg.norm(org[-1] - org[0])), 5),
               tendon_cap_moved=round(float(np.linalg.norm(ten[-1] - ten[0])), 5),
               radius_drift_pct=round(100.0 * float(rad[-1] / max(rad[0], 1e-9) - 1.0), 3),
               seconds=round(time.time() - t0, 1))
    with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)   # never lose a run to a numpy scalar
    if movie:
        render_eye_vtk.render(cap, dt, os.path.join(OUT, f"{name}.mp4"),
                              os.path.join(OUT, f"{name}_strip.png"))
    print(f"[{tag}] pose {res['pose_deg']}  shortened {res['shortening_pct']}%  "
          f"DELIVERED {res['delivered_pct']}%  origin cap moved {res['origin_cap_moved']}  "
          f"tendon cap moved {res['tendon_cap_moved']}  "
          f"[{res['seconds']}s]", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="F"); ap.add_argument("--muscle", default="LR")
    ap.add_argument("--k_bone", type=float, default=None)
    ap.add_argument("--k_sleeve", type=float, default=0.0)
    ap.add_argument("--width-scale", type=float, default=1.0)
    ap.add_argument("--thick-scale", type=float, default=1.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()
    run(a.model, a.k_sleeve, a.width_scale, a.thick_scale, a.muscle, a.device,
        k_bone=a.k_bone, movie=not a.no_movie, tag=a.tag)
