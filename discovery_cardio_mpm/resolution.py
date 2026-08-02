#!/usr/bin/env python
"""resolution -- is the answer physics, or is it the grid?

THE QUESTION
================================================================================================
The real motion is **smaller than one cell of the simulation grid** -- two thirds of one cell at
the peak of the beat, one tenth in RMS. The particle-to-grid transfer smooths over a three-cell
stencil, roughly 24 image pixels, which is coarser than the thing being measured. So the numerical
method is operating at a scale where it could plausibly be doing the modelling.

Three settings decide that scale, and **all three were inherited and never once varied** across
sixty batches: `n_grid = 128`, `per_parent = 16384` particles, and `--substeps 10`. They appear in
`HYPOTHESES.md` as untested defaults, because a default is a belief in disguise.

WHAT IS BEING ASKED, AND WHAT IS NOT
------------------------------------------------------------------------------------------------
This is a pre-flight test, not a study. It does not ask which resolution is right, and it does not
tune anything. It asks one question:

    if the discretisation changes, does the measured trajectory change by MORE than the noise?

If it does not, the numbers are converged at the inherited settings and the settings can be frozen
with that stated. If it does, then the discretisation is a lever like any other -- it must be
declared, frozen, and never varied inside a comparison, and any claim made at one resolution is a
claim about that resolution.

There is no forward-model tuning here and no fitting: every variant runs the SAME configuration,
from the same seed, for one iteration, and the simulated beat is compared with the Track B
descriptors -- the same instrument Track B will be judged by, so the answer is in the units the
campaign already uses.

    python resolution.py --check          # the ladder, from the inherited point outward
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "src")
SPEC = os.path.join(REPO, "config", "material", "material_aniso_cardio.yaml")
WORK = os.path.join(HERE, "_resolution")
PY = sys.executable

sys.path.insert(0, HERE)

# The inherited operating point, and the ladder around it. One knob moves at a time -- the same
# causal discipline the loop itself is held to.
BASE = {"n_grid": 128, "per_parent": 16384, "substeps": 10, "dt_sub": 2e-4}

# CAUGHT WHILE RUNNING THIS: `--substeps` is NOT a pure refinement knob. The trainer computes the
# duration of one recorded frame as `substeps * dt_sub` (train.py:156-162, 576), so halving the
# substeps halves how much simulated time one frame of the recording corresponds to. Varying it
# alone changes the PHYSICS, not the numerics -- and the inherited `--substeps 10` is therefore a
# statement about the timescale of the tissue, never identified as one in sixty batches.
#
# So the ladder carries both: `sub_*_naive` reproduces what varying it alone does (kept because it
# is what the campaign would have done), and `sub_*` holds `substeps * dt_sub` FIXED, which is the
# actual convergence test.
LADDER = [
    {"label": "inherited",    **BASE},
    {"label": "grid_96",      **{**BASE, "n_grid": 96}},
    {"label": "grid_192",     **{**BASE, "n_grid": 192}},
    {"label": "part_8k",      **{**BASE, "per_parent": 8192}},
    {"label": "part_32k",     **{**BASE, "per_parent": 32768}},
    {"label": "sub_5_naive",  **{**BASE, "substeps": 5}},
    {"label": "sub_20_naive", **{**BASE, "substeps": 20}},
    {"label": "sub_5",        **{**BASE, "substeps": 5, "dt_sub": 4e-4}},
    {"label": "sub_20",       **{**BASE, "substeps": 20, "dt_sub": 1e-4}},
]


def variant_spec(n_grid, per_parent, out_yaml, dt_sub=None):
    with open(SPEC) as f:
        s = yaml.safe_load(f)
    s = copy.deepcopy(s)
    s["fields"]["mpm_grid"]["n_grid"] = int(n_grid)
    s["sets"]["mpm_particle"]["per_parent"] = int(per_parent)
    if dt_sub is not None:
        # the trainer reads dt_sub off mpm_scatter (train.py:576); frame dt = substeps * dt_sub
        for o in s.get("operators", []):
            if isinstance(o, dict) and o.get("op") == "mpm_scatter":
                o["dt_sub"] = float(dt_sub)
    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(s, f, sort_keys=False)
    return out_yaml


def run_variant(v, seed=11, device="cuda:0", timeout=3600):
    """One forward + one step at this discretisation, dumping the simulated beat."""
    os.makedirs(WORK, exist_ok=True)
    yml = variant_spec(v["n_grid"], v["per_parent"],
                       os.path.join(WORK, "config", "material", f"res_{v['label']}.yaml"),
                       v.get("dt_sub"))
    out = os.path.join(WORK, v["label"])
    dump = os.path.join(WORK, f"{v['label']}.npz")
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    # --eval_dump is FORWARD ONLY and exits: no training, no checkpoint needed. The learnable
    # parameters sit at their seeded initial values, identical across the ladder, so the only
    # thing that differs between variants is the discretisation.
    cmd = [PY, os.path.join(HERE, "train.py"), yml,
           "--substeps", str(v["substeps"]), "--seed", str(seed), "--device", device,
           "--outdir", out, "--eval_dump", dump, "--allow_nondeterministic_ops", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if not os.path.exists(dump):
        return None, ((r.stderr or r.stdout or "").strip().splitlines() or ["no output"])[-1][:160]
    return dump, None


def compare(dumps, verbose=True):
    """Every variant against the inherited point, in Track B's own units."""
    import descriptors as DS
    ref = np.load(dumps["inherited"])
    rs, rr = ref["sim_d"], ref["real_d"]
    rows = []
    for label, path in dumps.items():
        z = np.load(path)
        s = z["sim_d"]
        # Particle counts differ across the ladder, so a per-node comparison is not defined.
        # Compare each variant against the SAME real beat instead -- that is what the campaign
        # actually reads, and it is well defined at every resolution.
        r = DS.loop_residual(s, z["real_d"])
        rows.append({"label": label, "particles": int(s.shape[1]), "frames": int(s.shape[0]),
                     "magnitude_peak": r["magnitude_peak"]["ratio"],
                     "opening_area": r["opening_area"]["ratio"],
                     "opening_loopiness": r["opening_loopiness"]["ratio"],
                     "direction_chirality": r["direction_chirality"]["ratio"],
                     "orientation_rad": r["orientation_error_rad"]["median_rad_covariance"],
                     "shape_minor": r["shape_minor_fraction"]["ratio"]})
    base = next(x for x in rows if x["label"] == "inherited")
    for x in rows:
        x["delta_vs_inherited"] = {k: float(x[k] - base[k]) for k in
                                   ("magnitude_peak", "opening_area", "opening_loopiness",
                                    "direction_chirality", "orientation_rad", "shape_minor")}
    if verbose:
        print(f"\n{'=' * 104}\n  RESOLUTION -- the same configuration at different discretisations\n{'=' * 104}")
        print(f"  {'variant':<12s} {'parts':>7s} {'magnitude':>10s} {'opening':>9s} "
              f"{'loopy':>8s} {'direction':>10s} {'orient rad':>11s}")
        for x in rows:
            print(f"  {x['label']:<12s} {x['particles']:>7d} {x['magnitude_peak']:>10.4f} "
                  f"{x['opening_area']:>9.4f} {x['opening_loopiness']:>8.4f} "
                  f"{x['direction_chirality']:>10.4f} {x['orientation_rad']:>11.4f}")
        print("\n  Read as DELTA from the inherited point. A delta larger than the seed-noise")
        print("  floor (Phase 2) means the discretisation is a lever and must be declared.")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", default=None, help="comma-separated labels")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    ladder = [v for v in LADDER if not a.only or v["label"] in a.only.split(",")]
    dumps, errors = {}, {}
    for v in ladder:
        print(f"[resolution] {v['label']}: n_grid={v['n_grid']} particles={v['per_parent']} "
              f"substeps={v['substeps']} dt_sub={v.get('dt_sub')} "
              f"(frame dt = {v['substeps'] * v.get('dt_sub', 2e-4):.1e})", flush=True)
        d, err = run_variant(v, a.seed, a.device)
        if d:
            dumps[v["label"]] = d
        else:
            errors[v["label"]] = err
            print(f"    FAILED: {err}", flush=True)

    out = {"ladder": ladder, "errors": errors}
    if "inherited" in dumps and len(dumps) > 1:
        out["rows"] = compare(dumps)
    else:
        print("[resolution] not enough variants completed to compare")
    json.dump(out, open(os.path.join(HERE, "_metrology", "resolution.json"), "w"), indent=1)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
