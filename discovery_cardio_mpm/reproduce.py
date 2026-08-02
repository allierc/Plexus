#!/usr/bin/env python
"""reproduce -- make the corpus runnable again, and check the library still does what it did.

THE SITUATION
================================================================================================
Seventy-one MPM runs sit finished in `graphs_data/material`, and **seven of the ten that drive
active traction can no longer be run at all**: their specs name operators the library no longer
has. Nothing is corrupt --- the trajectories are outputs and remain perfectly good evidence --- but
a result you cannot regenerate is a result you cannot interrogate, and this corpus is about to be
used to certify the instruments the whole campaign will rank on.

The cause is the exact defect that destroyed six batches of the previous campaign: operators were
renamed and merged in the library, and the recipes that named them were not carried along.

    pulse_stimulus          --> activation_pulse   (no delay map)
    phase_delay_pulse       --> activation_pulse   (with delay_from)   [commit 0acb0de, "M3"]
    pulse_to_active_stress  --> active_stress
    mpm_drag                --> drag               (+ emit: mpm_acceleration)
    p2g                     --> mpm_scatter
    g2p                     --> mpm_gather

WHY RE-RUNNING IS WORTH THE COMPUTE, RATHER THAN JUST FIXING THE NAMES
------------------------------------------------------------------------------------------------
Renaming a token in a file proves nothing. Running the migrated recipe and comparing the result
against the archived trajectory asks a question nobody has asked: **was that merge actually
behaviour-preserving?** Two operators were folded into one and the library moved on. If the new
trajectory matches the old one, the corpus is restored AND the merge is validated. If it does not,
we have found a silent change in the forward model -- which would matter far more, and which no
amount of reading the diff would have told us.

    python reproduce.py --check          # migrate every stale spec in memory; do they load now?
    python reproduce.py --write          # write the migrated specs into _repro/config/
    python reproduce.py --run NAME       # regenerate ONE run into the scratch root
    python reproduce.py --compare NAME   # archived vs regenerated trajectory

THE ARCHIVE IS NEVER TOUCHED. Everything is written under `_repro/`, so a regeneration that goes
wrong cannot destroy the thing it was meant to reproduce. The previous campaign lost two batches to
exactly that kind of accident.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "src")
DATA = os.path.join(REPO, "graphs_data", "material")
REPRO = os.path.join(HERE, "_repro")                      # scratch root; the archive is read-only
PY = sys.executable

sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# The rename map, taken from the library's own history rather than guessed. `activation_pulse`
# accepts every parameter both predecessors carried (period, duration, max_delay, delay_from,
# clock, center, radius, profile), so no parameter has to be translated -- only the name.
RENAME = {
    "pulse_stimulus": "activation_pulse",
    "phase_delay_pulse": "activation_pulse",
    "pulse_to_active_stress": "active_stress",
    "mpm_drag": "drag",
    "p2g": "mpm_scatter",
    "g2p": "mpm_gather",
    "agent_to_mpm": "mpm_scatter",
    "mpm_to_agent": "mpm_gather",
}

# Ordered by what the campaign actually needs. The four phase-delayed runs come first: they are the
# only simulations in the corpus that carry COORDINATION structure, and coordination is the axis
# the inherited objective is blind to -- so they are the test set for the measure Phase 2 has to
# invent. Nothing else in the corpus can play that role.
PRIORITY = [
    "material_active_phase_radial",       # a radial travelling wave
    "material_active_phase_swirl",        # a swirling one
    "material_active_phase_quadrants",    # four regions out of phase
    "material_active_phase_horizontal",   # a plane wave along x
    "material_active_swirl",              # rotating axis, no phase delay -- the contrast
    "material_active_radial_in",          # near-radial: the low-enclosure end of the ordering
    "material_active_horizontal",         # fixed axis
    "material_active_vertical",           # fixed axis, orthogonal -- orientation must separate these
]


def _shorten(yml, n_frames):
    """Cut the run to `n_frames`. The merge question does not need 250 of them: if the two operator
    sets behave identically, frame five already agrees; if they do not, the trajectories separate
    immediately. A short run is the RIGHT experiment here, not a compromise -- and the full run
    costs over two hours per recipe, which would put the answer out of reach entirely."""
    with open(yml) as f:
        spec = yaml.safe_load(f)
    spec.setdefault("general", {})["n_frames"] = int(n_frames)
    with open(yml, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return yml


def load_spec_raw(name):
    p = os.path.join(DATA, name, "spec.yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p) as f:
        return yaml.safe_load(f)


def migrate(spec):
    """Rename retired operators and modernise the substep block. Returns (spec, changes)."""
    s = copy.deepcopy(spec)
    changes = []

    for o in s.get("operators", []):
        if isinstance(o, dict) and o.get("op") in RENAME:
            old = o["op"]
            o["op"] = RENAME[old]
            changes.append(f"operator {old} -> {o['op']}")
            # `drag` emits an acceleration the MPM transfer reads; the old `mpm_drag` did it
            # implicitly. Made explicit here because the current cardio spec does so.
            if o["op"] == "drag":
                o.setdefault("emit", "mpm_acceleration")
                changes.append("drag: emit=mpm_acceleration (was implicit)")

    sched = s.get("schedule", [])
    for i, step in enumerate(sched):
        if isinstance(step, str) and step in RENAME:
            changes.append(f"schedule {step} -> {RENAME[step]}")
            sched[i] = RENAME[step]
        elif isinstance(step, dict):
            if "steps" in step:
                step["steps"] = [RENAME.get(x, x) for x in step["steps"]]
            # the substep block was {dt, steps, substep}; it is now {substep_dt, steps}
            if "dt" in step and "substep_dt" not in step:
                step["substep_dt"] = step.pop("dt")
                changes.append("schedule substep: dt -> substep_dt")
            step.pop("substep", None)

    # `activation_pulse` may now appear twice under one name where two operators used to differ;
    # that is legal (they are distinguished by `delay_from`) but worth surfacing.
    ops = [o.get("op") for o in s.get("operators", []) if isinstance(o, dict)]
    if ops.count("activation_pulse") > 1:
        changes.append("NOTE: two activation_pulse operators after the merge")
    return s, changes


def check(names=None, verbose=True):
    """Migrate in memory and ask the library whether it can load the result."""
    import plexus.operators                                     # registers them
    from plexus.models.registry import get_operator
    from plexus.schema import load

    rows = []
    for name in (names or PRIORITY):
        row = {"name": name}
        try:
            raw = load_spec_raw(name)
        except FileNotFoundError:
            row.update(status="NO SPEC"); rows.append(row); continue
        mig, changes = migrate(raw)
        row["changes"] = changes
        tmp = os.path.join(REPRO, "config", "material", f"{name}.yaml")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w") as f:
            yaml.safe_dump(mig, f, sort_keys=False)
        try:
            spec = load(tmp)
            missing = []
            for o in sorted({op.op for op in spec.operators}):
                try:
                    get_operator(o)
                except Exception:
                    missing.append(o)
            row["missing"] = missing
            row["status"] = "LOADS" if not missing else "MISSING OPS"
        except Exception as e:
            row["status"] = f"{type(e).__name__}: {str(e)[:90]}"
        rows.append(row)

    if verbose:
        print(f"\n{'=' * 100}\n  MIGRATING THE STALE RECIPES\n{'=' * 100}")
        for r in rows:
            ok = r.get("status") == "LOADS"
            print(f"  [{'  ok  ' if ok else ' FAIL '}] {r['name']:<36s} {r['status']}"
                  f"{'  MISSING ' + str(r['missing']) if r.get('missing') else ''}")
            for c in r.get("changes", [])[:3]:
                print(f"            {c}")
            if len(r.get("changes", [])) > 3:
                print(f"            ... and {len(r['changes']) - 3} more")
        n = sum(1 for r in rows if r.get("status") == "LOADS")
        print(f"\n  {n}/{len(rows)} migrated recipes load against today's library.")
        print(f"  written to {os.path.join(REPRO, 'config', 'material')}")
    json.dump(rows, open(os.path.join(HERE, "_metrology", "migration.json"), "w"), indent=1)
    return rows


def run(name, device="cuda:0", timeout=7200, frames=None):
    """Regenerate ONE run into the scratch root. The archive is not written to."""
    os.makedirs(REPRO, exist_ok=True)
    # The recipes read their field maps (fibre directions, delay maps, stiffness) as .tif files
    # from the data root. Symlink them into the scratch root rather than copying or, worse,
    # writing the run back into the archive: the inputs stay exactly the ones the archived run
    # used, and nothing can overwrite them.
    src_dir, dst_dir = DATA, os.path.join(REPRO, "graphs_data", "material")
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.endswith(".tif") and not os.path.exists(os.path.join(dst_dir, f)):
            os.symlink(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    # resolve_config takes an absolute .yaml path, so the migrated recipe is handed over
    # directly. No env var, no search order -- the same rule as data.py.
    spec_yaml = os.path.join(REPRO, "config", "material", f"{name}.yaml")
    if not os.path.exists(spec_yaml):
        raise FileNotFoundError(f"migrated spec not written yet: {spec_yaml} (run --check first)")
    if frames:
        _shorten(spec_yaml, frames)
    cmd = [PY, os.path.join(REPO, "Plexus_Main.py"), "-o", "generate", spec_yaml,
           "--force", "--output_root", REPRO, "--device", device, "--no-describe"]
    print(f"[reproduce] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout or "").strip().splitlines()[-4:] + (r.stderr or "").strip().splitlines()[-4:]
    for l in tail:
        print("   ", l[:150])
    return r.returncode


def compare(name):
    """Archived versus regenerated. The question is whether the merge preserved behaviour."""
    a = os.path.join(DATA, name, "trajectory.npz")
    b = os.path.join(REPRO, "graphs_data", "material", name, "trajectory.npz")
    if not os.path.exists(b):
        print(f"[reproduce] nothing regenerated yet at {b}")
        return 1
    with np.load(a) as za, np.load(b) as zb:
        pa = za["mpm_particle__pos"].astype(np.float64)
        pb = zb["mpm_particle__pos"].astype(np.float64)
    n = min(pa.shape[0], pb.shape[0])           # a shortened rerun compares over its own length
    if pa.shape[1:] != pb.shape[1:]:
        print(f"[reproduce] SHAPE CHANGED  archived {pa.shape}  regenerated {pb.shape}")
        return 1
    d = np.abs(pa[:n] - pb[:n])
    scale = np.abs(pa[:n] - pa[0]).max()
    rec = {"name": name, "frames_compared": int(n),
           "archived_frames": int(pa.shape[0]), "regenerated_frames": int(pb.shape[0]),
           "max_abs": float(d.max()), "motion_scale": float(scale),
           "relative": float(d.max() / scale) if scale else float("nan"),
           "bit_identical": bool(np.array_equal(pa[:n], pb[:n]))}
    print(f"  {name}: frames {rec['archived_frames']} vs {rec['regenerated_frames']}, "
          f"max|d| {rec['max_abs']:.3e}, relative to the motion {rec['relative']:.3e}, "
          f"bit-identical {rec['bit_identical']}")
    p = os.path.join(HERE, "_metrology", "reproduce_compare.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev[name] = rec
    json.dump(prev, open(p, "w"), indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--run", metavar="NAME", default=None)
    ap.add_argument("--compare", metavar="NAME", default=None)
    ap.add_argument("--all", action="store_true", help="run every priority spec, in order")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=None,
                    help="cut the rerun to N frames. The merge question is answered in the first "
                         "few: identical operators agree at frame 5, different ones separate at once")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    if a.compare:
        return compare(a.compare)
    if a.run:
        check([a.run], verbose=False)
        rc = run(a.run, a.device, frames=a.frames)
        return compare(a.run) if rc == 0 else rc
    if a.all:
        check(verbose=True)
        for n in PRIORITY:
            if run(n, a.device, frames=a.frames) == 0:
                compare(n)
        return 0
    return 0 if all(r.get("status") == "LOADS" for r in check()) else 1


if __name__ == "__main__":
    sys.exit(main())
