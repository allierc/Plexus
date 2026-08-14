"""run_07 -- my best version of eye_G, built from what runs 01-06 measured.

    python run_07.py --device cuda:0

Starts from eye_G's own baseline spec -- the scanned Blender anatomy, its layered globe
and its socket, all untouched -- and changes only the three things the rig identified as
broken, each with a measurement behind it:

  1  THE ORIGINS GET A BONE.  eye_G holds them with `bone_anchor`, a penalty spring, and
     nothing behind it. run_01 measured that spring letting the origin slide 0.063 world
     off the bone: 99% of the contraction, with 0.0007 reaching the load. run_04 embedded
     the origin in a pinned body instead and the slip fell to 28% with delivery tripled.
     So: a bone nodule per origin, pinned, with the anchored cap inside it, and no spring.

  2  THE TENDONS GET SEATED.  The audit found every tendon 6-17% of a radius OFF the
     globe -- about 1.3 grid cells -- while the bellies of SR, IR and IO penetrate it by
     15-16%: strongest grip where the muscle should slide, weakest where it should
     attach. `relieve_overlap` fixes the belly but can only push outward, so nothing
     brings a tendon in. `seat_attachments` does both halves.

  3  IT RUNS AT THE VALIDATED SUBSTEP.  2.0e-4 rather than 1.2e-4: derisk measured the
     settled pose moving 0.007 deg against a 0.05 deg tolerance, for 2.6x less compute.

Then it drives the synergies rather than single muscles, because no extraocular muscle
moves the eye along a cardinal axis alone, and the expectation is written down first:

    LR + SO  (blue + violet)  -> up        IR + IO  (green + orange) -> down
    LR       (blue)           -> right     MR       (yellow)         -> left

The elevator pairing is the fish one, not the mammalian: on the traced plant SO elevates
and IO depresses, because both obliques pull from the rostral orbit with no trochlea.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EYE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(EYE, "..", "..", "src"))
sys.path.insert(0, EYE)
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops
import blend_mpm_ops               # noqa: F401  the scanned geometry
# bench_ops FIRST: both modules define the `bone_particle` entity and the second
# registration raises rather than merging, so the one with the guard has to come last
import bench_ops                   # noqa: F401  pin_region[clamp], from the rig
import run07_ops                   # noqa: F401  the two fixes
import eye_anatomy as EA
import run_eye
from plexus.schema import load as load_spec
from run_staircase import settled

ARCHIVE = os.path.join(EYE, "archive")
SRC = os.path.join(ARCHIVE, "eye_G", "baseline_spec.yaml")     # READ ONLY
K = {k: i for i, k in enumerate(EA.MUSCLE_KEYS)}
PHASES = [("up", ["LR", "SO"], (1, +1)), ("down", ["IR", "IO"], (1, -1)),
          ("right", ["LR"], (0, +1)), ("left", ["MR"], (0, -1))]
# The devcontainer and the partition mount the SAME export at DIFFERENT paths, and
# neither path exists on the other host: the container has /workspace and no
# /groups/.../Graph, the partition the reverse. So the geometry paths are rewritten for
# whichever host is building the spec -- not translated unconditionally, which is what
# sent the loader hunting for a cache that was not there and made it re-cut the .blend.
MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")


def host_path(p):
    if not isinstance(p, str):
        return p
    for a, b in (MAP, MAP[::-1]):
        if p.startswith(a) and not os.path.exists(a) and os.path.exists(b):
            return b + p[len(a):]
    return p


def build(hold=500, rest=300, lead=100, tail=250, embed=0.012, standoff=0.010,
          n_bone=18000, substep=2.0e-4, tonic=None):
    spec = copy.deepcopy(yaml.safe_load(open(SRC)))
    centre = list(spec["sets"]["eye"]["start"][0])
    if tonic is None:
        tonic = float(next((o for o in spec["operators"]
                            if o["op"] in ("oculomotor_drive", "muscle_probe")),
                           {}).get("tonic", 0.14))

    # 1. a bone body for the origins, and the spring removed
    spec["sets"]["bone"] = {"n": 1, "start": [centre],
                            "types": {"bone": {"fraction": 1.0, "youngs": 1600.0}}}
    spec["sets"]["bone_particle"] = {"parent": "bone", "per_parent": int(n_bone),
                                     "radius": 0.05, "density": 2.0}

    ops, sched = [], []
    for o in spec["operators"]:
        if o["op"] == "bone_anchor":
            continue                                   # replaced by a body, not tuned
        for k in ("blend", "parts"):                   # valid on the host that runs it
            if k in o:
                o[k] = host_path(o[k])
        ops.append(o)
        if o["op"] == "blend_muscles":
            ops.append({"op": "seat_attachments", "at": "muscle_particle",
                        "before_frame": 1, "globe": "mpm_particle", "center": centre,
                        "embed": float(embed), "standoff": float(standoff), "cap": 0.12})
            ops.append({"op": "bone_from_origins", "at": "bone_particle",
                        "before_frame": 1, "muscles": "muscle_particle",
                        "youngs": 1600.0, "pad": 1.45})
            ops.append({"op": "pin_region", "implementation": "clamp",
                        "at": "bone_particle", "axis": 0, "beyond": -1e9})
    # the bone joins the shared grid: that is how the muscle feels it
    for o in list(ops):
        if o["op"] == "mpm_strain" and o.get("at") == "muscle_particle":
            ops.append({"op": "mpm_strain", "at": "bone_particle"})
        if o["op"] == "mpm_scatter" and o.get("at") == "muscle_particle":
            ops.append({"op": "mpm_scatter", "at": "bone_particle", "to": "mpm_grid",
                        "implementation": "accumulate", "drag": 0.0, "a_max": 200,
                        "polar": "higham"})
        if o["op"] == "mpm_gather" and o.get("at") == "muscle_particle":
            ops.append({"op": "mpm_gather", "at": "bone_particle", "from": "mpm_grid",
                        "wall_damp": 1.0, "wall_contact": 0.02, "vmax": 1e9})
        if o["op"] == "mpm_scatter":
            o["polar"] = "higham"
    spec["operators"] = ops

    sched = []
    for t in spec["schedule"]:
        if isinstance(t, dict):
            t = dict(t)
            t["substep_dt"] = float(substep)             # the validated substep
            sched.append(t)
            continue
        if t == "bone_anchor":
            continue
        sched.append(t)
        if t == "blend_muscles":
            sched += ["seat_attachments", "bone_from_origins", "pin_region"]
    spec["schedule"] = sched

    groups = [[K[m] for m in ms] for _, ms, _ in PHASES]
    spec = probe_ops.groups_spec(spec, groups, hold=hold, rest=rest, lead=lead,
                                 tail=tail, tonic=tonic)
    spec["general"]["name"] = "eye_G_run07"
    return spec, groups, tonic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run_07")
    ap.add_argument("--embed", type=float, default=0.012)
    ap.add_argument("--standoff", type=float, default=0.010)
    ap.add_argument("--hold", type=int, default=500)
    ap.add_argument("--rest", type=int, default=300)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    out = os.path.join(ARCHIVE, a.tag)
    os.makedirs(out, exist_ok=True)
    spec, groups, tonic = build(hold=a.hold, rest=a.rest, embed=a.embed,
                                standoff=a.standoff)
    path = os.path.join(out, f"{a.tag}_spec.yaml")
    with open(path, "w") as fh:
        fh.write("# run_07 -- eye_G's scanned anatomy with the three fixes runs 01-06\n"
                 "# measured: origins embedded in pinned bone (not a spring), tendons\n"
                 "# seated into the sclera and bellies held clear, validated substep.\n")
        yaml.safe_dump(spec, fh, sort_keys=False, width=100)

    dt = float(spec["general"]["dt"])
    prb = probe_ops.MuscleProbeGroups({"groups": groups, "hold": a.hold, "rest": a.rest,
                                       "lead": 100, "tail": 250, "tonic": tonic})
    print(f"[{a.tag}] {spec['general']['n_frames']} frames "
          f"({spec['general']['n_frames'] * dt:.1f} s), 4 synergies", flush=True)
    t0 = time.time()
    sim = load_spec(path)
    _, cap = run_eye.capture_run(sim, a.device, stride=a.stride)
    t = np.asarray(cap["frame"]) * dt
    g = np.asarray(cap["gaze"])
    np.savez_compressed(os.path.join(out, f"{a.tag}_curves.npz"),
                        frame=np.asarray(cap["frame"]), t=t.astype(np.float32),
                        gaze=g.astype(np.float32), act=np.asarray(cap["act"], np.float32),
                        length=np.asarray(cap["length"], np.float32),
                        rest_length=np.asarray(cap["rest_length"], np.float32),
                        centre=np.asarray(cap["centre"], np.float32),
                        muscles=np.array(EA.MUSCLE_KEYS))

    rows = []
    print("\n%-6s %-9s %-26s %-8s %s" % ("phase", "muscles", "settled (h,v,t) deg",
                                         "wanted", "verdict"))
    for slot, (name, ms, (ax, sgn)) in enumerate(PHASES):
        t_on, t_off = prb.window(slot)
        pre = g[t <= t_on * dt]
        base = pre[-1] if len(pre) else g[0]
        mean, sd, ptp = settled(t, g, t_on * dt, t_off * dt)
        dg = mean - base
        ok = float(dg[ax]) * sgn > 0.5
        rows.append(dict(phase=name, muscles=ms, gaze_deg=[round(float(v), 3) for v in dg],
                         settled=bool(max(ptp) <= 0.05), correct=bool(ok)))
        print("%-6s %-9s %-26s %-8s %s" % (name, "+".join(ms), str(rows[-1]["gaze_deg"]),
                                           f'{"hvt"[ax]}{"+" if sgn > 0 else "-"}',
                                           "YES" if ok else "no"))
    span_h = float(g[:, 0].max() - g[:, 0].min())
    span_v = float(g[:, 1].max() - g[:, 1].min())
    meta = dict(tag=a.tag, embed=a.embed, standoff=a.standoff, dt=dt,
                span_h=round(span_h, 2), span_v=round(span_v, 2),
                gate_pass=bool(span_h >= 25.0 and span_v >= 10.0),
                n_correct=sum(r["correct"] for r in rows),
                seconds=round(time.time() - t0, 1), phases=rows)
    with open(os.path.join(out, f"{a.tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"\n[{a.tag}] span {span_h:.1f} h / {span_v:.1f} v deg "
          f"(eye_G measured 5.9 / 6.0), {meta['n_correct']}/4 synergies correct, "
          f"gate {'PASS' if meta['gate_pass'] else 'FAIL'}  [{meta['seconds']}s]",
          flush=True)
    if not a.no_movie:
        import render_eye_vtk
        render_eye_vtk.render(cap, dt, os.path.join(out, f"{a.tag}.mp4"),
                              os.path.join(out, f"{a.tag}_strip.png"))


if __name__ == "__main__":
    main()
