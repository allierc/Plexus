"""bench_spec -- the minimal muscle-transmission rig, as a Plexus2 spec.

    bone (pinned)  ====[ muscle tube ]====  ( deformable sphere, far side pinned )

One muscle, one ball, one question: OF THE LENGTH THE MUSCLE LOSES, HOW MUCH REACHES
THE LOAD? On the full eye that question is answered against six muscles, a socket,
orbital fat, a lens and five antagonists at tonic, any of which could be the sink. Here
there is nowhere else for the contraction to go, so the answer is unambiguous.

Everything mechanical is the eye's: `muscle_contract`, `bone_anchor`, the shared
`mpm_grid` that couples the two bodies, the same material law and the same substep. Only
the geometry is simpler, which is the point -- a gate has to fail for one reason.
"""
from __future__ import annotations

import numpy as np

# the rig, in world units. The sphere sits at the centre of the box; the muscle runs
# horizontally to it from a bone plate on the left.
SPHERE_C = (0.62, 0.50, 0.50)
SPHERE_R = 0.105
BONE_FACE = 0.24                      # the bone's right face
BONE_LO = (0.10, 0.40, 0.40)          # the block itself
BONE_HI = (BONE_FACE, 0.60, 0.60)
MUSCLE_R = 0.022
EMBED = 0.012          # how far the tendon cap sinks into the sphere
BONE_BITE = 0.030      # how far the muscle's origin sinks INTO the bone (>= 3 grid cells)


def build(name="bench_muscle", n_frames=900, dt=0.003, substep_dt=2.0e-4,
          n_sphere=40000, n_muscle=9000, n_bone=16000, bone_youngs=1600.0, n_grid=112,
          muscle_youngs=240.0, sphere_youngs=420.0, vitreous_youngs=45.0,
          choroid_youngs=130.0, pin_frac=0.75, contract=67.0, pair=False, lever=0.62,
          k_socket=5000.0, k_fat=4000.0, c_fat=90.0, tendon_gap=None,
          muscle_radius=MUSCLE_R, k_bone=9000.0, c_bone=60.0, k_pin=60000.0,
          tonic=0.02, a_hi=1.0, lead=100, hold=600, rest=250, tail=200, seed=0,
          anchor="bone",
          muscle_length=None):
    cx, cy, cz = SPHERE_C
    # MUSCLE LENGTH IS SET BY MOVING THE LOAD, not by moving the attachment: the bone
    # bite and the tendon embed stay exactly as they were, so a length sweep changes
    # length and nothing else. Halving it should halve the absolute shortening at a
    # given strain, and the rig says whether the delivered fraction follows.
    if muscle_length is not None:
        cx = float((BONE_FACE - BONE_BITE) + float(muscle_length) + SPHERE_R - EMBED)

    # A PAIR, PULLING TANGENTIALLY, so the ball can TURN rather than be squashed. One
    # muscle on the axis can only push the load along it; the eye's horizontal pair
    # attaches off-axis on opposite sides, and the offset IS the moment arm. `lever` is
    # that offset as a fraction of the radius, and the two muscles are mirror images, so
    # alternating them rocks the ball one way and then the other.
    # WHERE THE TENDON SITS RELATIVE TO THE GLOBE. `EMBED` bites into the surface;
    # `tendon_gap` overrides it with a signed clearance, so the same rig can be run with
    # the tendon embedded, touching, or floating clear. The scanned eye's tendons sit
    # about 1.3 grid cells OFF the globe while their bellies penetrate it, and this is
    # what that costs, measured on a rig whose only other variable is which muscle pulls.
    bite = EMBED if tendon_gap is None else -float(tendon_gap)
    n_musc = 2 if pair else 1
    if pair:
        d = float(lever) * SPHERE_R
        # plain floats: numpy scalars are not YAML-representable and the spec is the
        # deliverable, so they must not leak into it
        xs = float(cx - np.sqrt(max(SPHERE_R ** 2 - d ** 2, 1e-9)) + bite)
        sphere_ends = [(xs, float(cy + d), cz), (xs, float(cy - d), cz)]
        bone_ends = [(float(BONE_FACE - BONE_BITE), float(cy + d), cz),
                     (float(BONE_FACE - BONE_BITE), float(cy - d), cz)]
    else:
        sphere_ends = [sphere_end]
        bone_ends = [bone_end]
    # THE MUSCLE IS EMBEDDED AT BOTH ENDS: its origin sinks BONE_BITE into the bone
    # block and its tendon EMBED into the sphere, and in both places the shared grid
    # carries the force. The same mechanism at both ends, so they are comparable --
    # neither is a spring.
    bone_end = (BONE_FACE - BONE_BITE, cy, cz)
    sphere_end = (cx - SPHERE_R + bite, cy, cz)      # + bites in, - stands clear

    return {
        "general": {"name": name, "seed": int(seed), "n_frames": int(n_frames),
                    "dt": float(dt), "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0],
                    "record_cap": 4000, "field_record_cap": 4000},
        "sets": {
            # the load: a deformable ball, held on its far side so it cannot simply be
            # dragged bodily toward the bone
            # THE BALL IS A SHELL, not a blob. run_02 gave it one modulus throughout and
            # it lost 18% of its radius and 40x its shape scatter under load -- a soft
            # sphere pinned on one side and pulled on the other collapses into the pin,
            # and the transmission measured against it was measuring the collapse. The
            # eye specs have always layered it: a soft vitreous core inside a stiff
            # sclera, which is what lets a globe hold its shape while it is pulled.
            "eye": {"n": 1, "start": [[cx, cy, cz]],
                    "state": {"pos": {"width": 3, "integration": "second_order_coordinate",
                                      "boundary": "world"},
                              "vel": {"width": 3, "integration": "second_order_rate",
                                      "boundary": "free", "record": False},
                              "gaze": {"width": 3, "integration": "none", "boundary": "free"}},
                    "types": {"globe": {
                        "fraction": 1.0, "youngs": float(sphere_youngs),
                        "layers": [{"frac": 0.60, "youngs": float(vitreous_youngs)},
                                   {"frac": 0.88, "youngs": float(choroid_youngs)},
                                   {"frac": 1.00, "youngs": float(sphere_youngs)}]}}},
            "mpm_particle": {"parent": "eye", "per_parent": int(n_sphere),
                             "radius": SPHERE_R, "density": 1.0},
            # the muscle: ONE, so nothing else can pull
            "muscle": {"n": n_musc,
                       "start": [[float(0.5 * (b[0] + t[0])), float(0.5 * (b[1] + t[1])),
                                  float(cz)] for b, t in zip(bone_ends, sphere_ends)],
                       "state": {"pos": {"width": 3, "integration": "second_order_coordinate",
                                         "boundary": "world"},
                                 "vel": {"width": 3, "integration": "second_order_rate",
                                         "boundary": "free", "record": False},
                                 "act": {"width": 1, "integration": "first_order",
                                         "boundary": "free"},
                                 "tension": {"width": 1, "integration": "none",
                                             "boundary": "free"},
                                 "length": {"width": 1, "integration": "none",
                                            "boundary": "free"}},
                       "types": {"muscle": {"fraction": 1.0, "youngs": float(muscle_youngs)}}},
            "muscle_particle": {"parent": "muscle", "per_parent": int(n_muscle),
                                "radius": float(muscle_radius), "density": 1.0},
            # the bone: a THIRD body on the same grid, pinned so it cannot move
            "bone": {"n": 1, "start": [[0.5 * (BONE_LO[0] + BONE_HI[0]), cy, cz]],
                     "types": {"bone": {"fraction": 1.0, "youngs": float(bone_youngs)}}},
            "bone_particle": {"parent": "bone", "per_parent": int(n_bone),
                              "radius": 0.10, "density": 2.0},
            "orbit": {"n": 1, "start": [[cx, cy, cz]]},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},
        "operators": [
            # a PLAIN SPHERE: axial_ratio 1 (no ovoid flattening) and no lens. The
            # operator is here because it is what labels the tissue and stores the rest
            # frame that `eye_pose` and the capture read -- not for its optics.
            {"op": "eye_anatomy", "at": "mpm_particle", "before_frame": 1,
             "center": [cx, cy, cz], "a_eq": SPHERE_R, "axial_ratio": 1.0,
             "lens_radius": 0.0, "lens_center": [0.0, 0.0, 0.0],
             "lens_youngs": float(sphere_youngs), "cornea_youngs": float(sphere_youngs),
             "pupil_deg": 0.0, "iris_deg": 0.0},
            {"op": "bone_block", "at": "bone_particle", "before_frame": 1,
             "lo": list(BONE_LO), "hi": list(BONE_HI), "youngs": float(bone_youngs)},
            {"op": "muscle_morphogenesis", "implementation": "tube", "at": "muscle_particle",
             "before_frame": 1, "bone_end": list(bone_end), "sphere_end": list(sphere_end),
             "radius": float(muscle_radius), "youngs": float(muscle_youngs), "cap": 0.10,
             "bone_ends": [[float(x) for x in v] for v in bone_ends],
             "sphere_ends": [[float(x) for x in v] for v in sphere_ends]},
            {"op": "eye_pose", "at": "eye", "child": "mpm_particle", "shell_min": 0.86},
            {"op": "muscle_geometry", "at": "muscle", "child": "muscle_particle", "eye": "eye"},
            # a plain step on the one muscle, ramped, held past settling, released
            # ALTERNATING, not together: two antagonists driven at once fight each other
            # and the ball goes nowhere. The `tour` probe steps each in turn with a rest
            # between, which is what makes the ball rock one way and then the other.
            (dict(op="muscle_probe", implementation="tour", at="muscle",
                  order=list(range(n_musc)), a_hi=float(a_hi), tonic=float(tonic),
                  hold=int(hold), rest=int(rest), lead=int(lead), tau=0.02,
                  step_frames=60)
             if pair else
             dict(op="muscle_probe", implementation="hold_vector", at="muscle",
                  muscles=list(range(n_musc)), levels_vec=[float(a_hi)] * n_musc,
                  tonic=float(tonic), hold=int(hold), lead=int(lead), tail=int(tail),
                  tau=0.02, step_frames=60)),
            {"op": "muscle_contract", "at": "muscle_particle", "muscles": "muscle",
             "amplitude": float(contract), "stretch_activation": 0.0,
             "strength": [1.0] * n_musc},
            # THE MUSCLE IS NOT ANCHORED BY AN OPERATOR AT ALL when `anchor="bone"`:
            # it is held because its origin cap is inside a pinned body. `spring` and
            # `clamp` keep the old boundary-condition versions for comparison.
            *([] if anchor == "bone" else
              [dict({"op": "bone_anchor", "at": "muscle_particle", "k": float(k_bone),
                     "c": float(c_bone)},
                    **({"implementation": "clamp"} if anchor == "clamp" else {}))]),
            {"op": "pin_region", "implementation": "clamp", "at": "bone_particle",
             "axis": 0, "beyond": -1e9},                  # the whole block, immovable
            # a FAR CAP, not half the ball: pinning half of it is what let the free half
            # be dragged into the fixed half. This holds the last quarter, enough to stop
            # the ball being towed away and little enough to leave it a body.
            # PINNING A BALL STOPS IT TURNING. For the pair the load is held the way the
            # eye holds it -- a bony cup it cannot leave, and orbital fat whose restoring
            # force is uniform and so exerts no torque about the centroid. It recentres
            # without resisting rotation, which is the whole point.
            *([{"op": "orbit_socket", "at": "mpm_particle", "orbit": "orbit",
                "k": float(k_socket), "damp": 12.0, "radius": SPHERE_R + 0.007,
                "aperture": 62.0, "k_fat": float(k_fat), "c_fat": float(c_fat)}]
              if pair else
              [{"op": "pin_region", "at": "mpm_particle", "k": float(k_pin), "c": 200.0,
                "axis": 0, "beyond": cx + float(pin_frac) * SPHERE_R}]),
            {"op": "drag", "at": "mpm_particle", "k": 5.0, "emit": "mpm_acceleration"},
            {"op": "drag", "at": "muscle_particle", "k": 6.0, "emit": "mpm_acceleration"},
            {"op": "mpm_strain", "at": "mpm_particle"},
            {"op": "mpm_strain", "at": "muscle_particle"},
            {"op": "mpm_strain", "at": "bone_particle"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200, "polar": "higham"},
            {"op": "mpm_scatter", "at": "muscle_particle", "to": "mpm_grid",
             "implementation": "accumulate", "drag": 0.0, "a_max": 200, "polar": "higham"},
            {"op": "mpm_scatter", "at": "bone_particle", "to": "mpm_grid",
             "implementation": "accumulate", "drag": 0.0, "a_max": 200, "polar": "higham"},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 1.0},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "wall_contact": 0.02, "vmax": 1e9},
            {"op": "mpm_gather", "at": "muscle_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "wall_contact": 0.02, "vmax": 1e9},
            {"op": "mpm_gather", "at": "bone_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "wall_contact": 0.02, "vmax": 1e9},
        ],
        "schedule": ["eye_anatomy", "bone_block", "muscle_morphogenesis", "eye_pose",
                     "muscle_geometry", "muscle_probe", "muscle_contract",
                     *([] if anchor == "bone" else ["bone_anchor"]),
                     *(["orbit_socket"] if pair else []), "pin_region", "drag",
                     {"substep_dt": float(substep_dt),
                      "steps": ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]}],
        "plotting": {"background": "black", "fps": 30},
    }
