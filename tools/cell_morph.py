"""Transfer a learned deformation-gradient control from the cheap single-material proxy
(tools/morph_gallery.py) to the 15-compartment atlas cell (config/cell/adh_poly_press.yaml).

THE TRANSFER, AND WHY IT NEEDS NO RE-OPTIMISATION. `deform_control`'s `field:` mode reads a
K^3 x 6 rate cube at each particle's own MATERIAL coordinate (its position at t=0). That cube was
fit against a 5,000-50,000-point single-material ball; handed to a 2M-point, 15-material cell it is
applied unchanged PROVIDED `extent:` is set to where the atlas cell actually sits, so a particle at
material coordinate x maps to the same trilinear cell in the control cube the proxy's own particles
did at the equivalent FRACTION of their own ball's radius. The rate values themselves need no
rescaling -- they are a rate on F, dimensionless per unit time, and the same time units are used on
both sides because `deform_control` reads `dt` off the HOST spec, never off the field.

THE 15 SETS, one node set per organelle, numbers copied verbatim from config/cell/adh_poly_press.yaml
(the canonical ~2M-point atlas cell this repo already runs): piece counts, per-piece node counts,
youngs, density, material, tau. `--divide` cuts every per-piece node count for a cheap smoke test
before committing cluster time to the full budget, exactly as tools/make_cell_atlas_sets.py's own
`--divide` does, but the piece counts (421 membrane patches, 138 nuclear-envelope patches, ...) are
never touched -- those are the atlas's geometry, not its resolution.

THE NUCLEUS IS THE ONE COMPARTMENT DELIBERATELY LEFT OUT of the control: `nuclear_envelope_node`,
`chromatin_node` and `nucleolus_node` -- the whole karyoplasm, envelope and contents together --
never receive `deform_control`. Every other node set does. Whether the nucleus is carried and
squeezed by the cytoplasm reshaping around it, or the cell tears at its boundary, is the question
`measure` below answers from the recorded trajectory -- not asserted from the movie.

WHAT IS DELIBERATELY OMITTED, relative to adh_poly_press.yaml itself, and why: no gravity, no
polymerize_tips, no mpm_density_pressure, no `set_material` overrides to elastic. Those exist there
to make an ADHERENT, SPREADING cell with a growing cytoskeleton; none of them are needed to ask
whether a rate field transfers, and mpm_density_pressure in particular resists local density change
-- exactly what an imposed rest-shape change produces -- so leaving it out keeps this comparison
against the SAME physics family morph_gallery.py optimised against (deform_control + mpm_strain
+ mpm_scatter + mpm_gather, nothing else). The organelles keep their ORIGINAL viscoelastic materials
(only the nuclear envelope is elastic), because a viscoelastic cytoplasm is what lets an imposed
rest-shape change show up as actual shape rather than being fought to a standstill as elastic
stress -- see `polar_growth`'s own docstring in cell_ops.py.

TIMING. morph_gallery.py's training rollout runs ticks 0..frames-1; its own callback special-cases
tick 0 as "reset to X0, apply no G" and applies G at ticks 1..frames-1 -- frames-1 applications, not
frames. `deform_control`'s `over:` gates on `H.frame < over`, ticks 0..over-1, no such special case.
Setting `over = frames - 1` reproduces the SAME NUMBER of G-applications (frames-1), hence the same
total accumulated deformation G^(frames-1) = expm(-A * (frames-1) * dt), independent of which
specific ticks it lands on since A and dt are constant across the run. `general.dt` here is set
equal to the dt morph_gallery.py trained against (0.002) so no rescaling of `over` is needed either.

    PYTHONPATH=src python tools/cell_morph.py build --target teapot \
        --control graphs_data/cell_morph_control/morph_teapot/morph.npz --divide 1
    PYTHONPATH=src python tools/cell_morph.py measure --target teapot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RESULTS_DIR = os.path.join(ROOT, "graphs_data", "cell", "cell_morph")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")

# THE SAME NFS EXPORT, TWO PREFIXES. `/workspace/Plexus` here is `/groups/saalfeld/home/allierc/
# Graph/Plexus` on the cluster -- discovery_okuda/cluster.py's own MAP. The spec YAML this module
# writes is READ ON THE CLUSTER (jobs/cell_morph.sh's `run` mode, never locally), so an absolute
# `field:` path baked in with the devcontainer prefix is a `FileNotFoundError` there even though
# the bytes are the same file -- caught once already: the first `run` submission failed in 15s on
# exactly this. `_cluster_path` is `cluster.cpath` duplicated rather than imported, so this module
# still works if discovery_okuda/ is ever unavailable on the PYTHONPATH the cluster job sets.
_MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")


def _cluster_path(p):
    ap = os.path.abspath(p)
    return _MAP[1] + ap[len(_MAP[0]):] if ap.startswith(_MAP[0]) else ap

# name, pieces, nodes_full, youngs, density, material, tau, atlas-type-name (the key `types:`
# carries, which `seed_cell_atlas._row` looks up in cell_ops.ATLAS -- verbatim from
# config/cell/adh_poly_press.yaml, the canonical ~2M-point atlas cell.
ORGANELLES = [
    ("plasma_membrane", 421, 300, 20.0, 1.0, "viscoelastic", 0.05, "plasma_membrane"),
    ("cytoskeleton", 400, 600, 75.0, 1.0, "viscoelastic", 0.05, "cytoskeleton"),
    ("nuclear_envelope", 138, 5000, 1200.0, 1.1, "elastic", 0.0, "nucleus"),
    ("chromatin", 31, 1250, 150.0, 1.2, "viscoelastic", 0.05, "chromatin"),
    ("nucleolus", 2, 1250, 300.0, 1.3, "viscoelastic", 0.05, "nucleolus"),
    ("rough_er", 11, 12500, 30.0, 1.0, "viscoelastic", 0.05, "rough_er"),
    ("smooth_er", 3, 12500, 25.0, 1.0, "viscoelastic", 0.05, "smooth_er"),
    ("golgi", 14, 1250, 27.5, 1.0, "viscoelastic", 0.05, "golgi"),
    ("mitochondria", 77, 1250, 45.0, 1.1, "viscoelastic", 0.05, "mitochondria"),
    ("centriole", 2, 1250, 200.0, 1.1, "viscoelastic", 0.05, "centriole"),
    ("protein_a", 1, 125000, 5.0, 0.6, "viscoelastic", 0.02, "protein_a"),
    ("protein_b", 1, 125000, 5.0, 0.6, "viscoelastic", 0.02, "protein_b"),
    ("protein_c", 1, 125000, 5.0, 0.6, "viscoelastic", 0.02, "protein_c"),
    ("protein_d", 1, 125000, 5.0, 0.6, "viscoelastic", 0.02, "protein_d"),
    ("protein_e", 1, 125000, 5.0, 0.6, "viscoelastic", 0.02, "protein_e"),
]
# THE NUCLEUS, AS ONE UNIT: envelope + its contents. Left rigid together -- deliberately never
# given `deform_control` -- so the interesting question is answerable at all: a control applied to
# chromatin/nucleolus but not their own envelope would tear the karyoplasm from the INSIDE, which
# is a different (and uninteresting) failure from the one this experiment asks about.
RIGID_NUCLEUS = ["nuclear_envelope_node", "chromatin_node", "nucleolus_node"]

COLORS = {
    "plasma_membrane": [0.62, 0.82, 0.96], "cytoskeleton": [0.30, 0.85, 0.40],
    "nuclear_envelope": [0.74, 0.62, 0.88], "chromatin": [0.52, 0.36, 0.74],
    "nucleolus": [0.92, 0.38, 0.66], "rough_er": [0.16, 0.52, 0.56],
    "smooth_er": [0.36, 0.78, 0.72], "golgi": [0.96, 0.62, 0.20],
    "mitochondria": [0.90, 0.16, 0.12], "centriole": [0.85, 1.00, 0.15],
    "protein_a": [0.95, 0.78, 0.25], "protein_b": [0.45, 0.78, 0.98],
    "protein_c": [0.98, 0.45, 0.62], "protein_d": [0.66, 0.48, 0.96],
    "protein_e": [0.55, 0.95, 0.82],
}
MEMBRANE_MAT = dict(ambient=0.12, diffuse=0.62, specular=0.55, specular_power=60,
                    backface_culling=True)


def build_spec(target, control_npz=None, rate=None, over=None, divide=1, n_grid=320,
              substep_dt=3.174603e-05, dt=0.002, frames_active=20, frames_relax=11,
              cell_radius=0.1, cell_centre=(0.5, 0.5, 0.5), seed=1, rigid_sets=None):
    """The cell_morph spec, as a dict -- see the module docstring for what is and is not in it.

    TWO CONTROL FORMS, mutually exclusive. `control_npz` is the K^3 x 6 field (deform_control's
    `field:` mode); `rate` is six numbers applied identically to every controlled particle
    (`rate:` mode) -- the right choice for a target with no spatial structure to it (a disc or a
    column is the SAME affine map everywhere), and the one that needs no `extent:` at all, since
    there is no cube to map material coordinates into.
    """
    if (control_npz is None) == (rate is None):
        raise ValueError("build_spec: give exactly one of `control_npz` (field mode) or `rate` "
                         "(global six-number mode), not both and not neither")
    if over is None:
        over = frames_active - 1              # see TIMING above -- the field-mode default
    n_frames = frames_active + frames_relax
    extent = [cell_centre[0], cell_centre[1], cell_centre[2], cell_radius]
    # WHICH SETS ARE LEFT OUT OF THE CONTROL. Defaults to the whole karyoplasm (RIGID_NUCLEUS);
    # `rigid_sets=[]` frees it -- every one of the 15 gets `deform_control` -- to test whether an
    # undeformable inclusion, not the control itself, is what caps how far the cell can follow it.
    rigid = RIGID_NUCLEUS if rigid_sets is None else list(rigid_sets)

    sets = {"cell": {"n": 1, "start": [list(cell_centre)],
                     "state": {
                         "pos": {"width": 3, "role": "coordinate",
                                 "integration": "second_order_coordinate", "boundary": "world"},
                         "vel": {"width": 3, "role": "rate",
                                 "integration": "second_order_rate", "record": False}}}}
    seeds, strain, scat, gath, aggr, deform = [], [], [], [], [], []
    total = 0
    for oname, pieces, nodes_full, youngs, dens, mat, tau, atlas_name in ORGANELLES:
        nname = f"{oname}_node"
        nodes = max(3, nodes_full // divide)
        type_block = {"fraction": 1.0, "youngs": youngs, "density": dens, "material": mat}
        if tau:
            type_block["tau"] = tau
        oset = {"parent": "cell", "per_parent": pieces, "radius": cell_radius,
               "types": {atlas_name: type_block}}
        if oname == "mitochondria":
            oset["state"] = {
                "pos": {"width": 3, "role": "geometry", "integration": "none", "boundary": "world"},
                "vel": {"width": 3, "role": "rate", "integration": "none", "record": False},
                "voltage": {"width": 1, "role": "coordinate", "integration": "first_order"},
            }
        sets[oname] = oset
        sets[nname] = {"parent": oname, "per_parent": nodes, "radius": cell_radius,
                       "density": dens, "entity": "mpm_particle"}
        total += pieces * nodes

        geom = {"geometry": {"plasma_membrane": {"thickness": 0.1}}} if oname == "plasma_membrane" else {}
        seeds.append({"op": "seed_cell_atlas", "at": oname, "particles": nname,
                     "cell_radius": cell_radius, "vel_init": 0.0, **geom})
        strain.append({"op": "mpm_strain", "at": nname, "implementation": "warp"})
        scat.append({"op": "mpm_scatter", "at": nname, "to": "mpm_grid", "drag": 2.0,
                    "a_max": 200.0, "implementation": "warp", "polar": "higham"})
        gath.append({"op": "mpm_gather", "at": nname, "from": "mpm_grid", "wall_damp": 1.0,
                    "vmax": 1.0, "implementation": "warp"})
        aggr.append({"op": "aggregate_centroid", "at": oname, "child": nname})
        if nname not in rigid:
            if control_npz is not None:
                # `rate: [0]*6` IS DEAD WEIGHT, NEVER READ: `DeformControl.forward` branches to
                # `_field_forward` and never touches `self.r` whenever `field:` is set. It is here
                # only because `schema.py::resolve_op_line` checks `REQUIRES_PARAMS` against the RAW
                # operator line's keys before the operator is constructed, so a field-mode line
                # missing `rate:` is refused before `DeformControl.__init__` runs its own (correct)
                # `field is not None` exemption. See the report for the one-line schema fix this wants.
                deform.append({"op": "deform_control", "at": nname, "field": control_npz,
                              "extent": extent, "over": over, "rate": [0.0] * 6})
            else:
                deform.append({"op": "deform_control", "at": nname,
                              "rate": [round(float(v), 6) for v in rate], "over": over})

    seeds.append({"op": "seed_state_random", "at": "mitochondria", "block": "voltage",
                 "lo": 0.0, "hi": 1.0, "seed": seed})

    operators = deform + strain + scat + \
               [{"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 1.0}] + gath + aggr
    schedule = (["deform_control"] * len(deform)
               + [{"substep_dt": substep_dt,
                   "steps": ["mpm_strain"] * len(strain) + ["mpm_scatter"] * len(scat)
                            + ["mpm_grid_update"] + ["mpm_gather"] * len(gath)}]
               + ["aggregate_centroid"] * len(aggr))

    node_names = [f"{o[0]}_node" for o in ORGANELLES]
    colors = {o[0]: COLORS[o[0]] for o in ORGANELLES}
    colors.update({f"{n}_node": c for n, c in list(colors.items())})
    surface = {n: {"render": "dots", "point_size": 0.9} for n in node_names}
    surface["plasma_membrane_node"] = dict(spacing=round(cell_radius / 21.0, 5), blur=1.4,
                                           smooth=40, **MEMBRANE_MAT)
    surface["nuclear_envelope_node"] = dict(spacing=round(cell_radius / 28.0, 5), blur=1.2, smooth=30)
    opacity = {n: 0.196 for n in node_names}
    opacity["plasma_membrane_node"] = 0.38
    opacity["nuclear_envelope_node"] = 0.28
    opacity["cytoskeleton_node"] = 0.28
    for k in "abcde":
        opacity[f"protein_{k}_node"] = 0.08

    spec = {
        "general": {"name": f"cell_morph_{target}", "seed": seed, "n_frames": n_frames, "dt": dt,
                   "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": n_frames,
                   "save_data": None, "units": {"length_um": 100.0}},
        "sets": sets,
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": n_grid}},
        "seed": seeds,
        "operators": operators,
        "schedule": schedule,
        "plotting": {
            "renderer": "vtk_points", "background": "black", "up_axis": 1, "box_frame": True,
            "camera_elev": 1.18, "camera_turns": 0.0, "camera_zoom": 0.0,
            "render_3d": "compartments", "dot_size": 0.9, "fps": 30, "slow_motion": 2,
            "surface_sample": 60000, "surface_max_cells": 80000000,
            "hide_sets": ["cell"] + [o[0] for o in ORGANELLES],
            "compartment_sets": node_names,
            "surface": surface, "opacity": opacity, "colors": colors,
        },
    }
    return spec, total, over


def local_bbox_check(spec):
    """Seed the spec ON CPU -- no physics, no GPU, no cluster -- and report the bounding boxes
    that matter before a single cluster minute is spent: the cell's own extent (from the plasma
    membrane, the organelle that actually defines the cell's outer boundary) and the nucleus's
    (envelope + chromatin + nucleolus together), against the control cube's `extent:`. A mismatch
    here is silent at runtime -- `deform_control` clamps out-of-range material coordinates to the
    cube's edge rather than raising -- so this is the only place it gets caught.
    """
    import torch
    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine
    import tempfile

    tmp = os.path.join(tempfile.mkdtemp(prefix="cell_morph_check_"), "spec.yaml")
    with open(tmp, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    sim = load(tmp)
    H = engine.build(sim, "cpu")
    engine.seed(H, sim, "cpu")

    def bbox(names):
        pts = torch.cat([H.level(n).get("pos").detach() for n in names], dim=0)
        return pts.min(0).values.numpy(), pts.max(0).values.numpy(), int(pts.shape[0])

    cell_lo, cell_hi, cell_n = bbox(["plasma_membrane_node"])
    nuc_lo, nuc_hi, nuc_n = bbox(RIGID_NUCLEUS)
    all_lo, all_hi, all_n = bbox([f"{o[0]}_node" for o in ORGANELLES])

    report = {
        "cell_bbox_lo_t0": cell_lo.tolist(), "cell_bbox_hi_t0": cell_hi.tolist(),
        "cell_n_membrane_points": cell_n,
        "nucleus_bbox_lo_t0": nuc_lo.tolist(), "nucleus_bbox_hi_t0": nuc_hi.tolist(),
        "nucleus_n_points": nuc_n,
        "whole_cell_bbox_lo_t0": all_lo.tolist(), "whole_cell_bbox_hi_t0": all_hi.tolist(),
        "whole_cell_n_points": all_n,
    }
    print(f"[cell_morph] t=0 bboxes:")
    print(f"  cell (plasma_membrane_node)     lo={cell_lo} hi={cell_hi}  ({cell_n:,} pts)")
    print(f"  nucleus (envelope+chromatin+nucleolus) lo={nuc_lo} hi={nuc_hi}  ({nuc_n:,} pts)")
    print(f"  whole cell (all 15 organelles)  lo={all_lo} hi={all_hi}  ({all_n:,} pts)")

    # THE EXTENT CHECK ONLY APPLIES TO FIELD MODE -- a `rate:`-mode line has no `extent:` to be
    # wrong about, since the same six numbers reach every controlled particle regardless of where
    # it started.
    first_deform = next((o for o in spec["operators"] if o["op"] == "deform_control"), None)
    if first_deform is not None and "extent" in first_deform:
        cx, cy, cz, half = first_deform["extent"]
        centre = np.array([cx, cy, cz])
        pts_all = torch.cat([H.level(f"{o[0]}_node").get("pos").detach()
                             for o in ORGANELLES], dim=0).numpy()
        r_all = np.linalg.norm(pts_all - centre, axis=1)
        frac_outside = float((np.abs(pts_all - centre).max(axis=1) > half).mean())
        report.update({"control_extent": [cx, cy, cz, half],
                       "max_radial_extent_t0": float(r_all.max()),
                       "fraction_points_outside_extent_cube": frac_outside})
        print(f"  control extent [cx,cy,cz,half] = {[cx, cy, cz, half]}; "
              f"max radial extent at t=0 = {r_all.max():.5f}; "
              f"{100*frac_outside:.3f}% of points fall outside the control cube (clamped to its edge)")
    elif first_deform is not None:
        report["control_rate"] = first_deform["rate"]
        report["control_over"] = first_deform["over"]
        print(f"  control: global rate {first_deform['rate']}, over {first_deform['over']} frames "
              f"(rate mode -- no extent to check, applies identically everywhere)")
    return report


def _append_results(entry):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {}
    if os.path.exists(RESULTS_JSON):
        try:
            data = json.load(open(RESULTS_JSON))
        except Exception:
            data = {}
    data.setdefault("runs", {})
    run = data["runs"].setdefault(entry["target"], {})
    for k, v in entry.items():
        # ONE LEVEL OF MERGE, not overwrite: `build` writes `commands.build`/`commands.run` and a
        # later `measure` call writes `commands.measure` -- a plain `dict.update` would replace the
        # whole `commands` dict and lose the earlier two, which is exactly the kind of thing
        # `results.json` exists to survive an interruption WITHOUT losing.
        if isinstance(v, dict) and isinstance(run.get(k), dict):
            run[k].update(v)
        else:
            run[k] = v
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[cell_morph] results -> {RESULTS_JSON}")


def cmd_build(a):
    if bool(a.control) == bool(a.rate):
        raise ValueError("cell_morph build: give exactly one of --control (field npz) or "
                         "--rate (six numbers, rate mode)")
    extra = {}
    if a.control:
        control_npz = os.path.abspath(a.control)
        if not os.path.exists(control_npz):
            raise FileNotFoundError(
                f"{control_npz} does not exist -- train it first with jobs/cell_morph.sh train "
                f"{a.target} (wraps tools/morph_gallery.py --control grid)")
        z = np.load(control_npz)
        if "control" not in z.files:
            raise ValueError(
                f"{control_npz} has no `control` key (files: {z.files}) -- deform_control's "
                f"`field:` mode needs the K^3 x 6 rate cube tools/morph_gallery.py writes under "
                f"that key. This npz looks like it was written by a different/older tool.")
        K = round(float(z["control"].shape[0]) ** (1.0 / 3.0))
        control_npz_cluster = _cluster_path(control_npz)
        print(f"[cell_morph] control field: {control_npz}  ({z['control'].shape[0]} = {K}^3 "
              f"nodes, 6 rates each)")
        if control_npz_cluster != control_npz:
            print(f"[cell_morph] spec will reference the CLUSTER path: {control_npz_cluster}")
        meta = None
        over = a.over
        if "meta" in z.files:
            meta = json.loads(str(z["meta"]))
            print(f"[cell_morph] npz meta: {meta}")
            if meta.get("ctrl_K") and int(meta["ctrl_K"]) != K:
                raise ValueError(f"meta says ctrl_K={meta['ctrl_K']} but control.shape implies "
                                 f"K={K} -- the npz is internally inconsistent")
            # `opt_frames - 1`: the SAME tick-0 exclusion as the TIMING note above, now read from
            # the file's own record instead of assumed from a CLI convention that has to match.
            if over is None and "opt_frames" in meta:
                over = int(meta["opt_frames"]) - 1
                print(f"[cell_morph] `over` defaulted from meta.opt_frames={meta['opt_frames']} "
                      f"- 1 = {over}")
        kw = dict(control_npz=control_npz_cluster, over=over)
        extra = {"control_npz_local": control_npz, "control_npz_cluster": control_npz_cluster,
                "control_K": K, "control_meta": meta}
        run_cmd_extra = ""
    else:
        rate = [float(v) for v in a.rate.split(",")]
        if len(rate) != 6:
            raise ValueError(f"--rate needs six comma-separated numbers, got {len(rate)}")
        if a.over is None:
            raise ValueError("--rate mode needs an explicit --over (the number of frames this "
                             "rate was fitted/reported to act for in its source spec)")
        print(f"[cell_morph] global rate {rate}, over {a.over} frames "
              f"(rate x{a.rate_scale:g} on top before writing)" if a.rate_scale != 1.0 else
              f"[cell_morph] global rate {rate}, over {a.over} frames")
        over_eff = a.over
        rate_eff = rate
        if a.rate_scale != 1.0:
            # SCALE `over`, NOT THE RATE. G^n = expm(-A n dt) for a CONSTANT A, so extending the
            # number of frames the rate acts for accumulates exactly the same total deformation a
            # `rate_scale`x stronger A would over the original `over` -- and it reuses the fitted
            # numbers unchanged, which is one fewer thing that can be gotten wrong transcribing them.
            over_eff = max(1, round(a.over * a.rate_scale))
        kw = dict(rate=rate_eff, over=over_eff)
        extra = {"control_rate_source": rate, "rate_scale": a.rate_scale,
                "over_source": a.over, "over_effective": over_eff}
        run_cmd_extra = ""

    rigid_sets = [] if a.free_nucleus else RIGID_NUCLEUS
    spec, total, over = build_spec(a.target, divide=a.divide, n_grid=a.n_grid,
                                   substep_dt=a.substep_dt, dt=a.dt,
                                   frames_active=a.frames_active, frames_relax=a.frames_relax,
                                   cell_radius=a.cell_radius, seed=a.seed, rigid_sets=rigid_sets,
                                   **kw)
    out = os.path.join(ROOT, "config", "cell", f"cell_morph_{a.target}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
    n_sets = len(spec["sets"])
    print(f"[cell_morph] wrote {out}")
    print(f"  {total:,} material points (divide={a.divide}), {n_sets} sets, "
          f"{len(spec['operators'])} operators, n_frames={spec['general']['n_frames']}, "
          f"over={over}")
    print(f"  rigid (no deform_control): {rigid_sets}")
    print(f"  controlled ({len([o for o in ORGANELLES if f'{o[0]}_node' not in rigid_sets])} "
          f"sets): {[f'{o[0]}_node' for o in ORGANELLES if f'{o[0]}_node' not in rigid_sets]}")

    bboxes = local_bbox_check(spec)
    _append_results({
        "target": a.target, "spec": os.path.relpath(out, ROOT),
        "divide": a.divide, "n_grid": a.n_grid, "substep_dt": a.substep_dt,
        "dt": a.dt, "n_frames": spec["general"]["n_frames"], "over": over,
        "total_points": total, "cell_radius": a.cell_radius,
        "rigid_sets": rigid_sets,
        "controlled_sets": [f"{o[0]}_node" for o in ORGANELLES
                           if f"{o[0]}_node" not in rigid_sets],
        "t0_bboxes": bboxes,
        "commands": {
            "build": " ".join(sys.argv),
            "run": f"conda run -n connectome-gnn python Plexus_Main.py -o generate "
                  f"cell/cell_morph_{a.target} --device cuda:0 --force --keep-stills "
                  f"--render-stills 8",
        },
        **extra,
    })


def cmd_measure(a):
    npz = os.path.join(ROOT, "graphs_data", "cell", f"cell_morph_{a.target}", "trajectory.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} not found -- run the spec first (jobs/cell_morph.sh run "
                                f"{a.target})")
    z = np.load(npz)

    def bbox_series(names):
        """[T,3] lo, [T,3] hi across the union of `names`, respecting `occ` if present."""
        los, his = [], []
        T = z[f"{names[0]}__pos"].shape[0]
        for t in range(T):
            pts = []
            for n in names:
                p = z[f"{n}__pos"][t]
                occ = z.get(f"{n}__occ")
                if occ is not None:
                    p = p[occ[t].astype(bool)]
                pts.append(p)
            pts = np.concatenate(pts, axis=0)
            los.append(pts.min(0)); his.append(pts.max(0))
        return np.stack(los), np.stack(his)

    cell_lo, cell_hi = bbox_series(["plasma_membrane_node"])
    nuc_lo, nuc_hi = bbox_series(RIGID_NUCLEUS)
    T = cell_lo.shape[0]

    def extent(lo, hi):
        return hi - lo

    cell_ext = extent(cell_lo, cell_hi)
    nuc_ext = extent(nuc_lo, nuc_hi)

    # THE TEAR METRIC: nearest-neighbour gap from every nuclear-envelope point to the nearest
    # point of the CONTROLLED cytoplasm (cytoskeleton + rough/smooth ER, the organelles that sit
    # immediately around the nuclear envelope in the atlas geometry -- ATLAS r_in for both ER rows
    # starts at 0.52-0.54 R, just outside the envelope's own 0.42 R). Grown from t=0 (contact) to
    # the final frame, this is a MEASURED separation, not an assertion of tearing.
    cyto_names = ["cytoskeleton_node", "rough_er_node", "smooth_er_node"]

    def min_gap(t):
        env = z["nuclear_envelope_node__pos"][t]
        occ_e = z.get("nuclear_envelope_node__occ")
        if occ_e is not None:
            env = env[occ_e[t].astype(bool)]
        cyto = np.concatenate([
            (lambda p, o: p[o[t].astype(bool)] if o is not None else p)(
                z[f"{n}__pos"][t], z.get(f"{n}__occ"))
            for n in cyto_names], axis=0)
        # subsample for tractability: 4,000 envelope points against the cytoplasm cloud is enough
        # to estimate a MINIMUM gap without an O(N^2) distance matrix on the full run.
        rng = np.random.default_rng(0)
        if env.shape[0] > 4000:
            env = env[rng.choice(env.shape[0], 4000, replace=False)]
        if cyto.shape[0] > 50000:
            cyto = cyto[rng.choice(cyto.shape[0], 50000, replace=False)]
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(cyto).query(env, k=1)
        except Exception:
            d = np.min(np.linalg.norm(env[:, None, :] - cyto[None, :, :], axis=-1), axis=1)
        return float(d.mean()), float(d.max())

    gap_t0_mean, gap_t0_max = min_gap(0)
    gap_tN_mean, gap_tN_max = min_gap(T - 1)

    # MEMBRANE CONTAINMENT: does the nucleus ever poke outside the cell's own bounding box, per
    # axis? A necessary (not sufficient) condition for the membrane having torn open around it.
    poked_out = bool(np.any(nuc_lo[-1] < cell_lo[-1]) or np.any(nuc_hi[-1] > cell_hi[-1]))

    report = {
        "cell_extent_t0": cell_ext[0].tolist(), "cell_extent_tN": cell_ext[-1].tolist(),
        "nucleus_extent_t0": nuc_ext[0].tolist(), "nucleus_extent_tN": nuc_ext[-1].tolist(),
        "cell_bbox_lo_tN": cell_lo[-1].tolist(), "cell_bbox_hi_tN": cell_hi[-1].tolist(),
        "nucleus_bbox_lo_tN": nuc_lo[-1].tolist(), "nucleus_bbox_hi_tN": nuc_hi[-1].tolist(),
        "n_frames_recorded": T,
        "nucleus_cytoplasm_gap_mean_t0": gap_t0_mean, "nucleus_cytoplasm_gap_max_t0": gap_t0_max,
        "nucleus_cytoplasm_gap_mean_tN": gap_tN_mean, "nucleus_cytoplasm_gap_max_tN": gap_tN_max,
        "nucleus_poked_outside_cell_bbox_tN": poked_out,
        "cell_extent_ratio_tN_over_t0": (cell_ext[-1] / cell_ext[0]).tolist(),
        "nucleus_extent_ratio_tN_over_t0": (nuc_ext[-1] / nuc_ext[0]).tolist(),
    }
    print(f"[cell_morph] {a.target}: measured over {T} recorded frames")
    print(f"  cell extent   t0 {cell_ext[0]} -> tN {cell_ext[-1]}  "
          f"(ratio {report['cell_extent_ratio_tN_over_t0']})")
    print(f"  nucleus extent t0 {nuc_ext[0]} -> tN {nuc_ext[-1]}  "
          f"(ratio {report['nucleus_extent_ratio_tN_over_t0']})")
    print(f"  nucleus-cytoplasm gap: mean {gap_t0_mean:.5f}->{gap_tN_mean:.5f}, "
          f"max {gap_t0_max:.5f}->{gap_tN_max:.5f}")
    print(f"  nucleus poked outside cell bbox at final frame: {poked_out}")
    tore = bool(poked_out) or (gap_tN_mean > 3.0 * max(gap_t0_mean, 1e-9))
    _append_results({"target": a.target, "measured": report, "tore": tore,
                     "commands": {"measure": " ".join(sys.argv)}})
    print(f"  VERDICT (measured, not asserted): tore={tore} "
          f"(poked_outside_cell_bbox={poked_out}, gap grew "
          f"{gap_tN_mean / max(gap_t0_mean, 1e-9):.2f}x)")


# --------------------------------------------------------------------------- the tear diagnostics
def _n_components(sub_pts, radius):
    """Connected components of `sub_pts` under an edge at every pair closer than `radius`.

    Takes an ALREADY-SUBSAMPLED cloud (see `cmd_tear`), because `cKDTree.query_pairs` on 126,300
    membrane points at a radius that has to resolve real gaps returns tens of millions of pairs. A
    macroscopic break -- a membrane torn into several lobes -- is visible in a 15,000-point
    subsample exactly as it is in the full cloud; a single dropped particle is not the kind of
    tear this is checking for. The RADIUS must be calibrated to THIS subsample's own spacing, not
    the full cloud's -- passing a radius sized for 126,300 points to a 15,000-point subsample
    fragments an intact sheet purely from under-sampling, which is a bug this function had until
    `cmd_tear` was changed to subsample once and calibrate on that exact subsample.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
    pairs = cKDTree(sub_pts).query_pairs(r=radius, output_type="ndarray")
    n = sub_pts.shape[0]
    if len(pairs) == 0:
        return n
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, _ = _cc(graph, directed=False)
    return n_comp


def _median_nn_dist(pts, sample=4000, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], min(sample, pts.shape[0]), replace=False)
    d, _ = cKDTree(pts).query(pts[idx], k=2)
    return float(np.median(d[:, 1]))


def _local_j_proxy(pts0, ptsN, k=8, sample=8000, seed=0):
    """A PROXY for the deformation-gradient determinant J, from positions alone.

    F itself is a buffer (`REQUIRES_BUFFERS = ["F"]` on the MPM particle), never written into
    `state_schema`, so it is not in `trajectory.npz` -- recording pulls from recorded STATE, and F
    is not state. What IS recoverable from two position clouds sharing the same particle indexing
    is local packing: mass is conserved per particle, so local density ~ 1 / (local neighbourhood
    volume), and J = V_now / V_reference. The k-th nearest-neighbour distance is a one-number
    estimate of that local neighbourhood's radius; cubing it estimates the volume up to a constant
    that cancels in the ratio.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    n = min(pts0.shape[0], ptsN.shape[0])
    idx = rng.choice(n, min(sample, n), replace=False)
    d0, _ = cKDTree(pts0).query(pts0[idx], k=k + 1)
    dN, _ = cKDTree(ptsN).query(ptsN[idx], k=k + 1)
    r0, rN = d0[:, -1], dN[:, -1]
    return (rN / np.clip(r0, 1e-12, None)) ** 3     # J proxy per sampled particle


def _frac_outside_envelope(cyto_pts, membrane_pts, centroid):
    """Fraction of `cyto_pts` FARTHER from `centroid` than the membrane point NEAREST to them.

    A global convex hull would call a concave (or lobed, or torn) membrane's own reentrant
    material "outside" -- wrong for exactly the shape a tear produces. Comparing each point only
    against its own nearest bit of membrane is what stays correct when the membrane is not convex.
    """
    from scipy.spatial import cKDTree
    d, idx = cKDTree(membrane_pts).query(cyto_pts, k=1)
    r_cyto = np.linalg.norm(cyto_pts - centroid, axis=1)
    r_memb = np.linalg.norm(membrane_pts[idx] - centroid, axis=1)
    return float((r_cyto > r_memb).mean())


def cmd_tear(a):
    npz = os.path.join(ROOT, "graphs_data", "cell", f"cell_morph_{a.target}", "trajectory.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} not found -- run the spec first")
    z = np.load(npz)

    def occ_pos(name, t):
        p = z[f"{name}__pos"][t]
        occ = z.get(f"{name}__occ")
        return p[occ[t].astype(bool)] if occ is not None else p

    T = z["plasma_membrane_node__pos"].shape[0]
    memb0, membN = occ_pos("plasma_membrane_node", 0), occ_pos("plasma_membrane_node", T - 1)
    cyto0, cytoN = occ_pos("cytoskeleton_node", 0), occ_pos("cytoskeleton_node", T - 1)

    # ONE subsample, by INDEX, shared between t0 and tN -- particles keep their row across the
    # run, so the same indices are the same physical points at both times. The radius is
    # calibrated on THIS subsample's own t0 spacing, not the full cloud's -- see `_n_components`.
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(memb0.shape[0], min(15000, memb0.shape[0]), replace=False)
    memb0_sub, membN_sub = memb0[sub_idx], membN[sub_idx]
    d0 = _median_nn_dist(memb0_sub, sample=min(4000, memb0_sub.shape[0]))
    radius = 2.5 * d0
    n_comp0 = _n_components(memb0_sub, radius)
    n_compN = _n_components(membN_sub, radius)

    # LOCAL J PROXY over THE CYTOPLASM PROPER: protein_a..e_node, the five bulk-filling "crowd"
    # sets cell_ops.py's own ATLAS table calls the reason a cell is not hollow. Restricted to
    # these and not every controlled set, because the k-NN local-volume estimate assumes a roughly
    # ISOTROPIC 3D packing -- true for a protein ball, false for a thin membrane patch (2D) or a
    # cytoskeletal filament (locally 1D), where the same estimator returned max|J-1| > 50 from
    # geometry alone, before any real strain. That is a property of the ESTIMATOR on those shapes,
    # not a finding about them.
    cytoplasm_sets = [f"protein_{k}_node" for k in "abcde"]
    jN_all = []
    for name in cytoplasm_sets:
        p0, pN = occ_pos(name, 0), occ_pos(name, T - 1)
        if min(p0.shape[0], pN.shape[0]) < 20:
            continue
        j = _local_j_proxy(p0, pN, sample=min(2000, p0.shape[0]))
        jN_all.append(j)
    jN_all = np.concatenate(jN_all) if jN_all else np.array([1.0])
    absJ = np.abs(jN_all - 1.0)

    centroidN = membN.mean(0)
    frac_out_t0 = _frac_outside_envelope(cyto0, memb0, memb0.mean(0))
    frac_out_tN = _frac_outside_envelope(cytoN, membN, centroidN)

    cell_ext0 = memb0.max(0) - memb0.min(0)
    cell_extN = membN.max(0) - membN.min(0)

    report = {
        "membrane_n_components_t0": int(n_comp0), "membrane_n_components_tN": int(n_compN),
        "membrane_component_radius": radius, "membrane_component_sample": 15000,
        "local_J_proxy_max_abs_minus1": float(absJ.max()),
        "local_J_proxy_p99_abs_minus1": float(np.percentile(absJ, 99)),
        "local_J_proxy_n_samples": int(absJ.size),
        "cytoskeleton_fraction_outside_membrane_t0": frac_out_t0,
        "cytoskeleton_fraction_outside_membrane_tN": frac_out_tN,
        "cell_extent_ratio_tN_over_t0_from_membrane": (cell_extN / cell_ext0).tolist(),
    }
    print(f"[cell_morph] {a.target}: TEAR diagnostics over {T} recorded frames")
    print(f"  membrane connected components (radius {radius:.5f}, from t0 packing): "
          f"{n_comp0} -> {n_compN}")
    print(f"  local J proxy over {absJ.size} controlled-cytoplasm samples: "
          f"max|J-1|={absJ.max():.3f}  p99|J-1|={np.percentile(absJ, 99):.3f}")
    print(f"  cytoskeleton points outside membrane envelope: {100*frac_out_t0:.2f}% (t0) -> "
          f"{100*frac_out_tN:.2f}% (tN)")
    print(f"  cell extent ratio (from membrane bbox): {report['cell_extent_ratio_tN_over_t0_from_membrane']}")
    _append_results({"target": a.target, "measured": report})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="write config/cell/cell_morph_<target>.yaml and check bboxes")
    b.add_argument("--target", required=True)
    b.add_argument("--control", default=None, help="path to a morph.npz with a `control` key "
                  "(field mode). Give this OR --rate, not both.")
    b.add_argument("--rate", default=None, help="six comma-separated numbers (rate mode): the "
                  "SAME global rate applied to every controlled particle, no field/extent -- the "
                  "right form for a target with no spatial structure (a disc, a column).")
    b.add_argument("--over", type=int, default=None, help="frames the control acts for. Field "
                  "mode defaults to --frames-active - 1; rate mode has no default and must be "
                  "given (the number of frames the source rate was fitted/reported against).")
    b.add_argument("--rate-scale", type=float, default=1.0, help="rate mode only: extend `over` "
                  "by this factor (same total accumulated deformation as scaling the rate itself, "
                  "for a constant A) -- how much harder to drive the SAME control.")
    b.add_argument("--divide", type=int, default=1, help="cut every organelle's node budget by this")
    b.add_argument("--n-grid", type=int, default=320)
    b.add_argument("--substep-dt", type=float, default=3.174603e-05)
    b.add_argument("--dt", type=float, default=0.002)
    b.add_argument("--frames-active", type=int, default=20,
                  help="must match --frames used to TRAIN the control")
    b.add_argument("--frames-relax", type=int, default=11)
    b.add_argument("--cell-radius", type=float, default=0.1)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--free-nucleus", action="store_true", help="give nuclear_envelope_node, "
                  "chromatin_node and nucleolus_node the SAME control as everything else, instead "
                  "of leaving them rigid -- isolates whether the undeformable karyoplasm is what "
                  "caps how far the cell follows the control.")
    b.set_defaults(func=cmd_build)

    m = sub.add_parser("measure", help="read a finished run's trajectory.npz, append to results.json")
    m.add_argument("--target", required=True)
    m.set_defaults(func=cmd_measure)

    t = sub.add_parser("tear", help="connected components, local J proxy, envelope-escape "
                       "fraction -- the tear diagnostics, appended under runs/<target>/measured")
    t.add_argument("--target", required=True)
    t.set_defaults(func=cmd_tear)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
