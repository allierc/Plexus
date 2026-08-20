#!/usr/bin/env python
"""run_eye_G -- the oculomotor plant with its geometry taken from the Blender model.

    python run_eye_G.py --spec-only            # just write archive/eye_G/baseline_spec.yaml
    python run_eye_G.py                        # build, run, render the orbiting movie
    python run_eye_G.py --preset atlas --particles 110000 --label full

Model G is model F's mechanics on SCANNED ANATOMY. Everything about the plant that was
tuned over models A-F -- stiffnesses, drive gains, socket, drag, substep -- is unchanged;
what changes is where the tissue is. Two operators are swapped:

    eye_anatomy            -> blend_globe      (seed)
    muscle_morphogenesis   -> blend_muscles    (seed)

so instead of squashing a seeded ball into an ovoid and generating six straps from
`strap_path`, the material points are placed inside the artist's meshes for the retina,
cornea, lens and the six extraocular muscles of `260802_s2_EYE_MUSCLES_MODEL.blend`
(see `blend_mpm_ops`). ONE EYE ONLY: the blend is mirror-symmetric and the plant's frame
is per-eye, so a run is one side, `--side R` by default (the right eye needs no reflection
to reach the canonical frame; `--side L` is its enantiomorph, and only torsion changes sign).

The other numbers the blend now supplies rather than `fish_anatomy`:

    muscle `start`      each strap's centroid, so the engine seeds its points near it
    orbit `radius`      the measured globe's equatorial semi-axis plus a clearance
    rest lengths        measured from the strap itself, by `blend_muscles`

Outputs, all in `archive/eye_G/`:

    baseline_spec.yaml  the spec -- the deliverable
    movie.mp4           ONE 3-D scene, translucent globe, camera flying once round it
    strip.png           five frames of that movie
    curves.npz          the captured traces
    diag.json           the metrics `run_eye.diagnose` computes for every model
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import yaml

import plexus.operators             # noqa: F401  stock operator library
import eye_ops                      # noqa: F401  eye operators (prototype-local)
import muscle_ops                   # noqa: F401  muscle-as-tissue operators
import blend_mpm_ops as BM          # the two seeds that read the .blend
import probe_groups as PG           # the open-loop synergy probe (muscle_probe [groups])
import eye_anatomy as EA
import eye_spec as ES
import run_eye
import render_orbit_vtk
import render_surface_vtk
from plexus.schema import load as load_spec

OUT_DIR = os.path.join(HERE, "archive", "eye_G")

# The G lineage's mechanics, as they stood at the end of the F/G campaign. These are NOT
# re-tuned for the scanned geometry -- keeping them fixed is what makes the run a test of
# the anatomy rather than of a new set of numbers.
G_MECHANICS = dict(
    n_muscle_particles=2200, n_grid=112, dt=0.003, substep_dt=0.00012,
    drag=5.0, muscle_drag=6.0, contract=67.0, stretch_activation=0.0,
    kp=0.10, ki=0.0, kd=0.010, tonic=0.14, gain=1.2, tau=0.020,
    k_socket=5000.0, k_fat=4000.0, c_fat=90.0, k_bone=9000.0, c_bone=60.0,
    k_sleeve=0.0, c_sleeve=30.0, sleeve_free=(0.70, 0.88),
    sclera_youngs=420.0, vitreous_youngs=45.0, choroid_youngs=130.0, muscle_youngs=240.0,
)


# --------------------------------------------------------------------------- #
#  the spec
# --------------------------------------------------------------------------- #
def build_spec(preset="probe", n_particles=45000, side="R", blend=BM.DEFAULT_BLEND,
               parts=BM.DEFAULT_PARTS, name="eye_G_blend", inflate=1.0, standoff=0.008,
               embed=-0.006, bone=False, n_bone=18000, smooth_iters=0, smooth_lambda=0.5,
               taper=True, strength=None, **kw):
    """Model F's spec with the two anatomy operators replaced by the blend seeds."""
    params = dict(G_MECHANICS)
    params.update(kw)
    spec = ES.build_spec(name=name, preset=preset, n_particles=n_particles,
                         plant="fish_larva", **params)
    pl = BM.plant(blend=blend, parts_dir=parts, side=side, a_eq=EA.A_EQ,
                  center=EA.GLOBE_CENTER, keys=EA.MUSCLE_KEYS, inflate=inflate)

    # RELATIVE to the prototype dir, so the same spec runs here and on the cluster
    seed_common = dict(blend=os.path.relpath(os.path.abspath(blend), HERE),
                       parts=os.path.relpath(os.path.abspath(parts), HERE), side=side,
                       a_eq=float(EA.A_EQ), center=[float(x) for x in EA.GLOBE_CENTER],
                       inflate=float(inflate))
    globe_seed = dict(op="blend_globe", at="mpm_particle", before_frame=1, **seed_common,
                      lens_youngs=float(EA.LENS_YOUNGS), cornea_youngs=320.0)
    muscle_seed = dict(op="blend_muscles", at="muscle_particle", before_frame=1, **seed_common,
                       keys=list(EA.MUSCLE_KEYS), cap=0.10,
                       standoff=float(standoff), embed=float(embed),
                       youngs=float(params["muscle_youngs"]),
                       smooth_iters=int(smooth_iters), smooth_lambda=float(smooth_lambda))

    ops = []
    for o in spec["operators"]:
        if o["op"] == "eye_anatomy":
            ops.append(globe_seed)
        elif o["op"] == "muscle_morphogenesis":
            ops.append(muscle_seed)
        elif o["op"] == "muscle_contract":
            o = dict(o, taper=bool(taper))
            if strength is not None:
                o["strength"] = [float(v) for v in strength]
            ops.append(o)
        else:
            ops.append(o)
    spec["operators"] = ops
    spec["schedule"] = [{"eye_anatomy": "blend_globe",
                         "muscle_morphogenesis": "blend_muscles"}.get(s, s)
                        if isinstance(s, str) else s
                        for s in spec["schedule"]]

    # the geometry the SETS carry: where each strap starts, and how wide the socket is
    spec["sets"]["muscle"]["start"] = [[round(float(v), 5) for v in s] for s in pl["starts"]]
    for o in spec["operators"]:
        if o["op"] == "orbit_socket":
            o["radius"] = round(float(pl["semi"][:2].mean() + 0.007), 5)
    if bone:
        spec = add_bone(spec, n_bone=n_bone)
    return spec, pl


def add_bone(spec, n_bone=18000, youngs=1600.0, pad=1.45, density=2.0):
    """Replace the origin SPRING with an origin BODY, pinned -- model I.

    `bone_anchor` is a penalty: it yields under exactly the load it exists to resist.
    Measured on the minimal rig in `archive/run_bench.py` (bone -> muscle -> ball, nothing
    else to absorb the pull), the anchored cap slid 0.063 world off its bone while the load
    moved 0.0007 -- 99% of the contraction lost -- and stiffening does not fix it: at
    k = 300 000 the run destabilised before the slip stopped.

    So the origin gets a BODY and the body gets pinned: one bone nodule per muscle,
    swallowing its anchored cap (`bone_from_origins`), held as a kinematic constraint
    rather than a spring (`pin_region [clamp]`: pos <- rest, vel <- 0 every step), and
    joined to the SAME MLS-MPM grid the globe and the muscles share. The muscle is then
    held the way its tendon is -- by material overlap through the grid -- and the spring is
    deleted rather than tuned. On the rig: slip 99% -> 28%, delivery 0.119 -> 0.275.

    The rigidity comes from the PINNING, not from the modulus: `youngs` is capped near 2000
    by the substep, and a stiffer nodule buys nothing.

    This is the ONLY change from model G. Seating the tendons deeper is a separate
    experiment (it made the span worse, 5.9 -> 3.8 deg, in the run that tried both), and
    `archive/run07_ops.py` keeps that operator for when it is measured on its own.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(HERE, "archive"))
    import bench_ops   # noqa: F401  pin_region[clamp]; MUST come first -- both modules
    import run07_ops   # noqa: F401  bone_from_origins; define `bone_particle`

    centre = list(spec["sets"]["eye"]["start"][0])
    spec["sets"]["bone"] = {"n": 1, "start": [centre],
                            "types": {"bone": {"fraction": 1.0, "youngs": float(youngs)}}}
    spec["sets"]["bone_particle"] = {"parent": "bone", "per_parent": int(n_bone),
                                     "radius": 0.05, "density": float(density)}
    ops = []
    for o in spec["operators"]:
        if o["op"] == "bone_anchor":
            continue                                   # replaced by a body, not tuned
        ops.append(o)
        if o["op"] == "blend_muscles":
            ops.append({"op": "bone_from_origins", "at": "bone_particle", "before_frame": 1,
                        "muscles": "muscle_particle", "youngs": float(youngs), "pad": float(pad)})
            ops.append({"op": "pin_region", "implementation": "clamp", "at": "bone_particle",
                        "axis": 0, "beyond": -1e9})
    for o in list(ops):                                # the bone joins the shared grid
        if o["op"] == "mpm_strain" and o.get("at") == "muscle_particle":
            ops.append({"op": "mpm_strain", "at": "bone_particle"})
        if o["op"] == "mpm_scatter" and o.get("at") == "muscle_particle":
            ops.append({"op": "mpm_scatter", "at": "bone_particle", "to": "mpm_grid",
                        "implementation": "accumulate", "drag": 0.0, "a_max": 200})
        if o["op"] == "mpm_gather" and o.get("at") == "muscle_particle":
            ops.append({"op": "mpm_gather", "at": "bone_particle", "from": "mpm_grid",
                        "wall_damp": 1.0, "wall_contact": 0.02, "vmax": 1e9})
    spec["operators"] = ops
    sched = []
    for t in spec["schedule"]:
        if isinstance(t, dict) or t != "bone_anchor":
            sched.append(t)
        if t == "blend_muscles":
            sched += ["bone_from_origins", "pin_region"]
    spec["schedule"] = sched
    return spec


# --------------------------------------------------------------------------- #
#  the run
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="probe", choices=sorted(ES.PRESETS))
    ap.add_argument("--program", default="gaze", choices=("gaze", "pairs"),
                    help="gaze = the closed-loop tour; pairs = open-loop synergies "
                         "(SR+SO up, IR+IO down, LR temporal, MR nasal)")
    ap.add_argument("--groups", default=None,
                    help="override the synergies, e.g. 'SR,SO|IR,IO|LR|MR'")
    ap.add_argument("--hold", type=int, default=70, help="frames each synergy is contracted")
    ap.add_argument("--rest", type=int, default=45, help="frames between synergies")
    ap.add_argument("--a-hi", type=float, default=1.0, help="activation of a driven muscle")
    ap.add_argument("--contract", type=float, default=G_MECHANICS["contract"],
                    help="peak active stress in muscle_contract (the traction; G uses 67)")
    ap.add_argument("--inflate", type=float, default=1.0,
                    help="grow the GLOBE by this factor about its centre, leaving the straps "
                         "where they are (1.2 buries the tendon tips inside the sclera)")
    ap.add_argument("--k-bone", type=float, default=G_MECHANICS["k_bone"],
                    help="bone_anchor SPRING stiffness (9000 in eye G); no effect with --bone, "
                         "which deletes the spring for a pinned rigid body instead")
    ap.add_argument("--bone", action="store_true",
                    help="model I: pin the origins with a bone BODY instead of bone_anchor")
    ap.add_argument("--n-bone", type=int, default=18000, help="bone particles, all six nodules")
    ap.add_argument("--standoff", type=float, default=0.008,
                    help="clearance the muscle belly keeps from the globe surface")
    ap.add_argument("--embed", type=float, default=-0.006,
                    help="how deep the tendon cap sits INSIDE the surface (negative)")
    ap.add_argument("--eye-youngs", type=float, default=None,
                    help="set sclera/choroid/vitreous ALL to this one value (a uniform globe; "
                         "overrides the three flags below)")
    ap.add_argument("--sclera-youngs", type=float, default=G_MECHANICS["sclera_youngs"])
    ap.add_argument("--vitreous-youngs", type=float, default=G_MECHANICS["vitreous_youngs"])
    ap.add_argument("--choroid-youngs", type=float, default=G_MECHANICS["choroid_youngs"])
    ap.add_argument("--muscle-youngs", type=float, default=G_MECHANICS["muscle_youngs"],
                    help="muscle passive stiffness E (240 in eye G; A/E = 67/240 = 0.28)")
    ap.add_argument("--smooth-iters", type=int, default=0,
                    help="Laplacian-smooth the fibre coordinate s BEFORE differentiating it "
                         "into f=grad(s)/|grad(s)| -- s->s~->f~, not f smoothed after the "
                         "fact; k-NN graph, s=0/s=1 endpoints preserved by rescale. 0 = off")
    ap.add_argument("--smooth-lambda", type=float, default=0.5,
                    help="smoothing step size per iteration, s <- (1-lam)s + lam*mean(nbrs)")
    ap.add_argument("--k-sleeve", type=float, default=G_MECHANICS["k_sleeve"],
                    help="orbital-pulley transverse constraint (0 = off, the G default). "
                         "Penalizes displacement PERPENDICULAR to the fibre only, leaving "
                         "axial shortening free -- this is the anti-buckling mechanism: "
                         "muscle_ops.MuscleSleeve, built after the obliques folded to "
                         "55-70% of their length under load with nothing holding their path")
    ap.add_argument("--c-sleeve", type=float, default=G_MECHANICS["c_sleeve"])
    ap.add_argument("--sleeve-free", type=float, nargs=2, default=[0.70, 0.88],
                    metavar=("FROM", "TO"),
                    help="fibre-coordinate range over which the sleeve tapers to fully free "
                         "(the tendon end must stay free to follow the globe's rotation)")
    ap.add_argument("--taper", default="on", choices=("on", "off"),
                    help="muscle_contract's T(s)=sin(pi s)^0.5 gate, zero at both caps, "
                         "peak over the belly (on, the default); 'off' activates the "
                         "whole muscle uniformly including the caps")
    ap.add_argument("--lr-strength", type=float, default=1.0,
                    help="per-muscle strength MULTIPLIER on LR only (the other five stay "
                         "at fish_anatomy's measured-cross-section weight of 1.0); the "
                         "geometry's own volumes already carry relative strength, so this "
                         "is a deliberate departure from it, not a correction")
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--side", default="R", choices=("L", "R"))
    ap.add_argument("--blend", default=BM.DEFAULT_BLEND)
    ap.add_argument("--parts", default=None,
                    help="where the .blend's cut (parts.npz/parts.json) is cached; defaults "
                         "to <out>/blend_parts, so a second --blend never overwrites the "
                         "first eye's cut just because they both fell back to the same "
                         "global default")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stride", type=int, default=3, help="capture every Nth frame")
    ap.add_argument("--turns", type=float, default=1.0,
                    help="camera orbits over the movie; 0 locks the camera entirely")
    ap.add_argument("--renderer", default="surface", choices=("surface", "points"),
                    help="surface = the blend meshes skinned to the particles; "
                         "points = the material points themselves")
    ap.add_argument("--az", type=float, default=25.0,
                    help="camera azimuth in degrees (0 = straight at the pupil)")
    ap.add_argument("--frames", type=int, default=None, help="override n_frames (a quick look)")
    ap.add_argument("--instrument", action="store_true",
                    help="mechanism-discrimination diagnostics per muscle (axial/transverse "
                         "stress, insertion torque decomposition, bone reaction) -- see "
                         "run_eye._instrument_frame. Off by default: it costs an extra pass "
                         "over the muscle particles every captured frame.")
    ap.add_argument("--spec-only", action="store_true")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--rerender", default=None, metavar="CURVES.NPZ",
                    help="re-draw the movie from a saved capture, without running anything")
    args = ap.parse_args()

    if args.rerender:                        # the movie is a view of the capture, not of the run
        cap = {k: v for k, v in np.load(args.rerender).items()}
        stem = args.rerender.replace("_curves.npz", "")
        parts_dir = args.parts or os.path.join(args.out, "blend_parts")
        mp4 = stem + ".mp4"
        R = render_surface_vtk if args.renderer == "surface" else render_orbit_vtk
        R.render(cap, 0.003, mp4, stem + ".png", turns=args.turns, az0=args.az,
                 side=args.side, blend=args.blend, parts=parts_dir)
        print(f"[eye_G] {mp4}")
        return

    os.makedirs(args.out, exist_ok=True)
    sclera = vitreous = choroid = None
    if args.eye_youngs is not None:
        sclera = vitreous = choroid = args.eye_youngs
    parts_dir = args.parts or os.path.join(args.out, "blend_parts")
    spec, pl = build_spec(preset=args.preset, n_particles=args.particles, side=args.side,
                          blend=args.blend, parts=parts_dir,
                          contract=args.contract, inflate=args.inflate,
                          standoff=args.standoff, embed=args.embed,
                          bone=args.bone, n_bone=args.n_bone,
                          sclera_youngs=sclera if sclera is not None else args.sclera_youngs,
                          vitreous_youngs=vitreous if vitreous is not None else args.vitreous_youngs,
                          choroid_youngs=choroid if choroid is not None else args.choroid_youngs,
                          muscle_youngs=args.muscle_youngs,
                          smooth_iters=args.smooth_iters, smooth_lambda=args.smooth_lambda,
                          k_bone=args.k_bone, taper=(args.taper == "on"),
                          strength=([args.lr_strength, 1.0, 1.0, 1.0, 1.0, 1.0]
                                    if args.lr_strength != 1.0 else None),
                          k_sleeve=args.k_sleeve, c_sleeve=args.c_sleeve,
                          sleeve_free=tuple(args.sleeve_free))
    probe = None
    if args.program == "pairs":
        groups = PG.PAIRS if not args.groups else [
            [EA.MUSCLE_KEYS.index(k.strip()) for k in g.split(",")]
            for g in args.groups.split("|")]
        labels = (PG.PAIR_LABELS[:len(groups)] if not args.groups
                  else ["+".join(EA.MUSCLE_KEYS[i] for i in g) for g in groups])
        drv = next(o for o in spec["operators"] if o["op"] == "oculomotor_drive")
        tonic = float(drv.get("tonic", 0.14))
        pk = dict(groups=groups, a_hi=args.a_hi, tonic=tonic, hold=args.hold,
                  rest=args.rest, lead=40, tail=60)
        spec = PG.groups_spec(spec, **pk)
        probe = PG.MuscleProbeGroups(pk)
        if args.label == "baseline":
            args.label = "pairs"
    if args.frames:
        spec["general"]["n_frames"] = int(args.frames)
    spec_path = os.path.join(args.out, f"{args.label}_spec.yaml"
                             if args.label != "baseline" else "baseline_spec.yaml")
    with open(spec_path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, default_flow_style=False)
    print(f"[eye_G] {spec_path}")
    print(f"[eye_G] {pl['frame'].describe()}")
    print("[eye_G] muscle volumes (sim units^3): "
          + "  ".join(f"{k}={v:.2e}" for k, v in pl["volumes"].items()))
    if args.spec_only:
        return

    sim = load_spec(spec_path)
    H, cap = run_eye.capture_run(sim, device=args.device, stride=args.stride,
                                 instrument=args.instrument)
    np.savez_compressed(os.path.join(args.out, f"{args.label}_curves.npz"),
                        **{k: v for k, v in cap.items() if k not in ("gpos", "gvel")})
    if probe is not None:
        # custom groups are not the four cardinal synergies, so the direction each one is
        # SUPPOSED to move is not known in advance: fall back to reporting the dominant axis
        # rather than scoring against a prior that does not apply.
        per = PG.report(cap, probe, labels=labels,
                        expect=None if not args.groups else [])
        diag = dict(n_frames=int(sim.n_frames), program="pairs", synergies=per,
                    radius_worst_pct=round(float(100.0 * np.max(np.abs(cap["radius"] - 1.0))), 2),
                    strain_p99=round(float(np.percentile(cap["cut_strain"], 99)), 4),
                    peak_shortening_pct=round(float(np.max(
                        100.0 * (1.0 - cap["length"] / cap["rest_length"][None, :]))), 2),
                    max_shortening_pct={EA.MUSCLE_KEYS[j]: round(float(np.max(
                        100.0 * (1.0 - cap["length"][:, j] / cap["rest_length"][j]))), 2)
                        for j in range(EA.N_MUSCLE)})
        print("\n  what each synergy did (open loop, so this is the geometry talking):")
        for k, v in per.items():
            print(f"    {k:6s} {v['label']:42s} gaze {v['gaze_excursion_deg']}  "
                  f"expected {v['expected']:12s} {'OK' if v['ok'] else 'NO'}")
        h = [v["gaze_excursion_deg"][0] for v in per.values()] + [0.0]
        v_ = [v["gaze_excursion_deg"][1] for v in per.values()] + [0.0]
        span_h, span_v = max(h) - min(h), max(v_) - min(v_)
        GREEN, BOLD, RESET = "\033[32m", "\033[1m", "\033[0m"
        print(f"\n  {BOLD}{GREEN}workspace: {span_h:.1f} deg horizontal x "
              f"{span_v:.1f} deg vertical{RESET}\n")

        if args.instrument:
            # each muscle is judged in ITS OWN driven window -- the settled 40% of the hold
            # where the group containing it was actually pulling, `run_eye.diagnose`'s
            # convention -- not averaged over the whole run, most of which it is at tonic.
            fr = cap["frame"]
            table = {}
            for mi, key in enumerate(EA.MUSCLE_KEYS):
                slot = next((s for s, g in enumerate(probe.groups) if mi in g), None)
                if slot is None:
                    continue
                t_on, t_off = probe.window(slot)
                sel = (fr >= t_on + 0.6 * (t_off - t_on)) & (fr <= t_off)
                if sel.sum() < 2:
                    continue
                L0, Lt = cap["rest_length"][mi], cap["length"][sel, mi]
                axial = float(cap["mus_axial"][sel, mi].mean())
                trans = float(cap["mus_transverse"][sel, mi].mean())
                table[key] = dict(
                    path_shortening_pct=round(float(100.0 * (1.0 - Lt.min() / L0)), 2),
                    axial_stress=round(axial, 4),
                    transverse_stress=round(trans, 4),
                    transverse_over_axial=round(trans / max(axial, 1e-9), 3),
                    tau_desired=round(float(np.abs(cap["tau_desired"][sel, mi]).mean()), 6),
                    tau_orth=round(float(cap["tau_orth"][sel, mi].mean()), 6),
                    eta_torque=round(float(cap["eta_torque"][sel, mi].mean()), 4),
                    bone_reaction=round(float(np.nanmean(cap["bone_reaction"][sel, mi])), 6))
            table["_whole_eye"] = dict(span_h_deg=round(span_h, 2), span_v_deg=round(span_v, 2))
            with open(os.path.join(args.out, f"{args.label}_instrument_table.json"), "w") as fh:
                json.dump(table, fh, indent=2)
            print("  mechanism table (each muscle, its own driven window):")
            print(f"    {'muscle':6s}{'short%':>8s}{'axial':>8s}{'trans':>8s}{'t/a':>6s}"
                  f"{'tau_des':>10s}{'tau_orth':>10s}{'eta':>6s}{'F_bone':>10s}")
            for key, row in table.items():
                if key == "_whole_eye":
                    continue
                print(f"    {key:6s}{row['path_shortening_pct']:8.1f}{row['axial_stress']:8.3f}"
                      f"{row['transverse_stress']:8.3f}{row['transverse_over_axial']:6.2f}"
                      f"{row['tau_desired']:10.2e}{row['tau_orth']:10.2e}"
                      f"{row['eta_torque']:6.2f}{row['bone_reaction']:10.2e}")
            print()
    else:
        diag = run_eye.diagnose(cap, sim)
    diag["geometry"] = dict(source=os.path.basename(args.blend), side=args.side,
                            scale=round(pl["scale"], 5),
                            axial_ratio=round(pl["axial_ratio"], 4),
                            semi_axes=[round(float(x), 4) for x in pl["semi"]],
                            rest_lengths={k: round(float(v), 4)
                                          for k, v in zip(EA.MUSCLE_KEYS, pl["rest_lengths"])},
                            muscle_volumes={k: float(f"{v:.6g}") for k, v in pl["volumes"].items()})
    with open(os.path.join(args.out, f"{args.label}_diag.json"), "w") as fh:
        json.dump(diag, fh, indent=2)
    print(json.dumps({k: v for k, v in diag.items() if k != "holds"}, indent=2)[:2400])

    if not args.no_movie:
        mp4 = os.path.join(args.out, f"{args.label}.mp4" if args.label != "baseline" else "movie.mp4")
        png = os.path.splitext(mp4)[0].replace("movie", "strip") + ".png"
        R = render_surface_vtk if args.renderer == "surface" else render_orbit_vtk
        R.render(cap, sim.dt, mp4, png, turns=args.turns, az0=args.az, side=args.side,
                 inflate=args.inflate, blend=args.blend, parts=parts_dir)
        print(f"[eye_G] {mp4}\n[eye_G] {png}")


if __name__ == "__main__":
    main()

# python run_eye_G.py --program pairs --eye-youngs 240 --label E240 --hold 200 --rest 160 --turns 0 --az 25

# # muscle passive stiffness E (default 240)
# python run_eye_G.py --program pairs --muscle-youngs 180 --label E_mus180 ...

# # peak active stress A (default 67) — already existed as --contract
# python run_eye_G.py --program pairs --contract 90 --label A90 ...

# # both together, e.g. to hold A/E fixed while scaling
# python run_eye_G.py --program pairs --contract 50 --muscle-youngs 178 --label ratio_hold ...
