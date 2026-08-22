#!/usr/bin/env python
"""The promotion gate: run a spec on BOTH sides, fresh, in parallel on the L4 node, compare the bytes.

    THE PROTOCOL, and it is the whole point of this file
    ----------------------------------------------------
    Run the spec in okuda FIRST, and run the twin spec in the codebase. DO NOT USE PREVIOUS OKUDA
    RESULTS -- they can be updated. Run both on the L4 cluster node, IN PARALLEL.

Nothing already sitting in `log/okuda` or `log/okuda_ECM` is ever the reference. Those artefacts
were produced by whatever okuda was on the day they were written, and okuda is still moving:
`round.py` runs campaigns into that tree and `staged.py` writes into it by hand. A promotion checked
against a stale file proves that the file has not changed, which is not the question. So every check
regenerates both sides and submits them together.

    WHAT COUNTS AS IDENTICAL
    ------------------------
The two writers differ -- okuda's `RunArchive` writes `traj.npz` (`pos_i`, `mesh_i`, `act_i`), the
core writes `trajectory.npz` (`<set>__pos`, `<set>__occ`) -- so a file hash cannot match and is not
the test. What is compared is ARRAY BY ARRAY, BIT FOR BIT (`a.tobytes() == b.tobytes()`) over every
recorded frame, reduced to one sha1 per side so a run records as a single number.

    THE REPEATABILITY FLOOR IS ZERO, AND IT WAS MEASURED
    ----------------------------------------------------
Byte-identity is only the right criterion if the platform can deliver it, and over 1,800 frames --
with division, T1 and chemistry in the loop, growing 2,000 cells to 12,272 -- that is not obvious:
one differing mantissa bit can move an `edge_flip` decision (`l < l_th_frac * mean_l`) and from then
on the two runs have DIFFERENT TOPOLOGY, at which point "max |delta|" is not small, it is meaningless.
So it was measured rather than assumed. `--phase R` runs the identical spec, the identical code and
the identical commit TWICE, as two separate cluster jobs:

    b_star        1800 frames, 2000 -> 12,272 cells   beb6fb24fe86dcde == beb6fb24fe86dcde
    b_null_plain  1800 frames                          14aab859c575fe25 == 14aab859c575fe25

    max |delta| = 0.0 exactly, first differing bit: none, over 120 arrays per run.

The floor is ZERO, so `tol = 0` is a real criterion at full length and not an aspiration, and every
row keeps it. A tolerance would have to be justified by a floor above zero; there isn't one. Re-run
`--phase R` if the queue, the driver or the GPU model changes -- that is what would move it.

    DETERMINISM IS ASSERTED, NOT ASSUMED
    ------------------------------------
`plexus/engine.py` sets `torch.use_deterministic_algorithms(True, warn_only=True)` -- "bit-
reproducible runs: deterministic scatter/index_add (else GPU atomics differ)". `warn_only` silently
downgrades any kernel with no deterministic implementation, and a downgraded kernel is exactly where
two runs of one spec stop matching. Each side therefore runs with `PLEXUS_STRICT_DETERMINISM=1`,
which turns the downgrade into an exception, and a side that dies of it FAILS rather than passing on
a comparison it never earned.

    THE SIDES
    ---------
    okuda            run_one.py, from the working tree
    okuda@<ref>      run_one.py, from a git worktree at <ref> -- how "before an engine change" is
                     regenerated rather than remembered
    core             Plexus_Main.py -o generate, the promoted registry

    python tools/promotion_identical.py --phase 0        the pairs for one phase
    python tools/promotion_identical.py --all            everything in PAIRS
    python tools/promotion_identical.py --compare-only   skip submission, compare what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OKUDA = os.path.join(ROOT, "discovery_okuda")
CFG_OKUDA = os.path.join(ROOT, "config", "okuda")
LOG_OKUDA = os.path.join(ROOT, "log", "okuda")
OUT = os.path.join(ROOT, "log", "promotion")
sys.path.insert(0, OKUDA)

# =============================================================================================
# THE SPEC SET. One pair per mechanism group; a phase is not done until its rows are green.
#
# SPECS ARE REUSED, NOT WRITTEN. `config/okuda/` holds 458 of them and they are what produced the
# runs this promotion has to reproduce; inventing a fresh spec for the comparison would be
# comparing the promotion against a case neither side has ever run.
# =============================================================================================
# `frames=None` means RUN THE SPEC AT ITS OWN LENGTH. The short rows are regression rows -- the
# same 100 frames every time, so the digest is a number that can be quoted and re-checked. The
# full-length rows are the SCENES: four of them are the minisite's own death figures, and they are
# run to the end because a gate whose archive stops at frame 100 cannot show the deformation it
# claims to be testing.
PAIRS = [
    # phase   spec               frames tol   side A         side B     what it exercises
    #
    # `frames=None` MEANS THE SPEC'S OWN LENGTH, and that is now the default for every row. 100
    # frames was too short to be a test of anything structural: `b_star`'s topology keeps changing
    # past frame 800, so a 100-frame gate green-lights a promotion that has not yet run a single one
    # of the divisions and flips it is supposed to preserve. These specs are 600 and 1,800 frames
    # long because that is how long the mechanisms take.
    #
    # `tol` IS AN ABSOLUTE TOLERANCE ON POSITION in the spec's own units (world 80.0, cells of
    # radius ~5); 0 means byte-identical. It is set from the MEASURED repeatability floor -- the
    # `R` rows below -- and not from a number that looked reasonable.
    (0,  "b_null_plain",         None, 0.0, "okuda@HEAD",  "okuda",  "mesh_seed, cell_mechanics, edge_flip, topo_record"),
    # A DEATH SPEC, because `b_null_plain` never calls `cell_die`. Removal is the first operation in
    # this engine that MOVES A ROW, so a renumber change gated only on that spec passes without
    # executing a line of what changed.
    (0,  "apop_one",             None, 0.0, "okuda@HEAD",  "okuda",  "+ cell_die -> H.renumber_set: ONE row moves"),
    # ---- the death scenes. One row moved is the correctness case and it is not the hard case.
    # `renumber_set` is a GATHER through `keep`, and these four ask progressively more of it: 200
    # rows out of order, then a patch, then nine bands, then most of the sheet. Each is an existing
    # okuda spec and each is a figure on the minisite, so `log/promotion/<spec>/` is both the gate's
    # evidence and the scene itself.
    (0,  "apop_many",            None, 0.0, "okuda@HEAD",  "okuda",  "200 SCATTERED deaths -- an interleaved `keep`, not a truncation"),
    (0,  "apop_patch_big",       None, 0.0, "okuda@HEAD",  "okuda",  "293 of 2000 in a 45-deg cap -> the surface is drawn INWARD (invagination 0.0369)"),
    (0,  "apop_rings9",          None, 0.0, "okuda@HEAD",  "okuda",  "895 of 2000 in nine bands -> closes over every gap, stays a sphere (red_vol 0.9813)"),
    (0,  "apopgeo_half",         None, 0.0, "okuda@HEAD",  "okuda",  "285 of 400 above the equator -> topology survives, sphere -> ellipsoid (gyr_prolate 1.869)"),
    (0.5, "ecm_block",           None, 0.0, "okuda@HEAD",  "okuda",  "mpm_scatter/gather/grid_update/strain, ecm_seed, ecm_stress"),
    # ---- Phase B: nine okuda operator files became two modules in `src/plexus/operators/`. Side A
    # is okuda BEFORE the move, side B okuda AFTER it, so what is under test is the MOVE, through
    # the runner that has always driven these operators. That is a necessary step and NOT the claim
    # the promotion is making -- `B-core` below is.
    ("B", "b_gs_plain_soft_lo",  None, 0.0, "okuda@HEAD",  "okuda",  "+ seed_cell_chem/diffuse/react, cell_geometry, cell_neighbours"),
    ("B", "b_star",              None, 0.0, "okuda@HEAD",  "okuda",  "+ cell_grow, cell_divide, interface_tension, cell_chem_from_shape"),
    # ---- B-core: THE ACTUAL CLAIM. `Plexus_Main.py -o generate` against `run_one.py`. Different
    # runner, different recorder, different writer -- the same numbers.
    ("B-core", "b_gs_plain_soft_lo", None, 0.0, "okuda",   "core",   "the same run, from the core registry, no okuda import"),
    ("B-core", "b_star",             None, 0.0, "okuda",   "core",   "the same run, from the core registry, no okuda import"),
    # ---- R: THE REPEATABILITY FLOOR. Same spec, same code, same commit, same queue -- twice. Any
    # tolerance above must be set from what this measures, because a gate tighter than the platform's
    # own noise fails on runs that are correct, and a gate looser than it passes runs that are not.
    ("R", "b_star",              None, 0.0, "okuda",       "okuda",  "1800 f, 2000->12272 cells, twice -- MEASURED FLOOR 0.0"),
    ("R", "b_null_plain",        None, 0.0, "okuda",       "okuda",  "1800 f, twice -- MEASURED FLOOR 0.0"),
    ("C", "01c_tissue",          None, 0.0, "okuda",       "core",   "junction_myosin (both pools), junction_sync, cytokinetic_ring"),
    ("D", "04_spheroid_ecm_pass2", None, 0.0, "okuda",     "core",   "mesh_contact, mesh_inside, ecm_*, bm_*"),
]

# The default for a row that asks for neither: short enough to sit in a per-phase loop, long enough
# that the mechanisms have acted (a 100-frame vertex run has divided and flipped).
FRAMES = int(os.environ.get("PROMO_FRAMES", "100"))


# ---------------------------------------------------------------------------------- the sides
def _worktree(ref):
    """A git worktree at `ref`, so "before the change" is RUN rather than remembered.

    The alternative -- keeping the previous run's output and diffing against it -- is exactly the
    stale reference this file exists to refuse. A worktree costs a checkout of the tracked files
    (`log/` is gitignored, so none of the 15 GB comes with it) and gives a real, runnable okuda at
    the old commit, which can then be submitted beside the new one.
    """
    d = os.path.join(OUT, "_worktrees", ref.replace("/", "_"))
    if os.path.isdir(os.path.join(d, ".git")) or os.path.isfile(os.path.join(d, ".git")):
        return d
    os.makedirs(os.path.dirname(d), exist_ok=True)
    subprocess.run(["git", "-C", ROOT, "worktree", "add", "--detach", d, ref],
                   check=True, capture_output=True, timeout=300)
    return d


def _side_paths(side):
    """(home, config dir, log dir) for a side.

    A WORKTREE IS A WHOLE TREE, not just a copy of the code: it has its own `config/okuda/` from the
    commit and writes its runs into its own `log/okuda/`. The first version of this wrote the spec
    into the main tree's config and the worktree job died on a missing file, which is the honest
    failure -- the two sides must not share anything the comparison could smuggle state through.
    """
    if side == "okuda":
        home = OKUDA
    elif side.startswith("okuda@"):
        home = os.path.join(_worktree(side.split("@", 1)[1]), "discovery_okuda")
    else:                                                     # core
        home = ROOT
    root = ROOT if side == "core" else os.path.dirname(home)
    return home, os.path.join(root, "config", "okuda"), os.path.join(root, "log", "okuda")


def _spec_copy(spec, run_name, frames, cfg_dir=None):
    """`config/okuda/<run_name>.yaml`: the pair's spec at the comparison length, under its own name.

    A DISTINCT NAME PER SIDE, because okuda writes to `log/okuda/<name>/` and two sides of one
    comparison would otherwise write into the same directory and the second would overwrite the
    first -- a comparison that always passes. `general.name` is a label, not an input: the RNG is
    seeded from `general.seed` (`engine.build`: `torch.Generator(...).manual_seed(sim.seed)`), so
    renaming cannot move a byte. The first row of the table is the check on that claim.
    """
    cfg_dir = cfg_dir or CFG_OKUDA
    src = os.path.join(CFG_OKUDA, f"{spec}.yaml")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} -- the comparison reuses okuda specs; this one is missing")
    cfg = yaml.safe_load(open(src))
    cfg["general"] = dict(cfg.get("general") or {})
    cfg["general"]["name"] = run_name
    if frames is not None:                      # None = the spec's own length (the scene rows)
        cfg["general"]["n_frames"] = int(frames)
        cfg["general"]["record_cap"] = int(frames) + 2
    os.makedirs(cfg_dir, exist_ok=True)
    dst = os.path.join(cfg_dir, f"{run_name}.yaml")
    yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
    return dst


def _bsub_lines(pair_dir, spec, side, run_name, frames):
    """The bsub command for one side. Both sides go into ONE remote script, so they are submitted
    together and the scheduler runs them at the same time -- 'in parallel' is a property of the
    submission, not a hope about the queue."""
    import cluster as C
    home, _cfg, _log = _side_paths(side)
    if side.startswith("okuda"):
        script = os.path.join(pair_dir, f"{run_name}.sh")
        with open(script, "w") as f:
            f.write("\n".join([
                "#!/bin/bash -l",
                f"cd {C.cpath(home)}",
                f"export PYTHONPATH={C.cpath(os.path.join(os.path.dirname(home), 'src'))}:"
                f"{C.cpath(os.path.join(home, 'ops'))}:{C.cpath(home)}",
                "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
                "export MPLBACKEND=Agg",
                # THE DETERMINISM ASSERTION travels with the job, not with the submitter: the run
                # happens on another machine and a flag set here would not reach it.
                "export PLEXUS_STRICT_DETERMINISM=1",
                f"conda run -n {C.ENV} python run_one.py {run_name}"
                + (f" --frames {frames}" if frames is not None else "")
                + " --device cuda:0 --campaign promotion",
            ]) + "\n")
    else:
        script = os.path.join(pair_dir, f"{run_name}.sh")
        with open(script, "w") as f:
            f.write("\n".join([
                "#!/bin/bash -l",
                f"cd {C.cpath(ROOT)}",
                f"export PYTHONPATH={C.cpath(os.path.join(ROOT, 'src'))}",
                "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
                "export MPLBACKEND=Agg",
                "export PLEXUS_STRICT_DETERMINISM=1",
                f"conda run -n {C.ENV} python Plexus_Main.py -o generate promotion/{run_name} "
                f"--output_root {C.cpath(pair_dir)} --device cuda:0 --force",
            ]) + "\n")
    os.chmod(script, 0o755)
    out = C.cpath(os.path.join(pair_dir, f"{run_name}.out"))
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    excl = "".join(f'-R "hname!={h}" ' for h in C.EXCLUDE_HOSTS if h)
    return (f"cd {C.cpath(ROOT)} && bsub -n {C.NCPUS} {gpu}{excl}-q {C.QUEUE} -W {C.WALL} "
            f"-J promo_{run_name} -o {out} -e {out[:-4]}.err bash -l {C.cpath(script)}")


# ---------------------------------------------------------------------------------- comparison
def _arrays(d, ticks=None):
    """Every recorded array of a run, in a fixed order, whichever writer produced it.

    Returns (labelled_arrays, ticks). THE TWO WRITERS RECORD THE SAME ROWS, which is what makes an
    okuda run and a core run comparable at all rather than merely similar:

        run_one.py:593    posf  = out["sets"]["vertex"]["pos"]
                          chemf = out["sets"]["cell"]["state"]["chem"]

    -- okuda's own frames come straight out of `engine._assemble`, the same structure
    `graph_data_generator` writes to `trajectory.npz`. So okuda's `pos_i` IS core's
    `vertex__pos[t]`, cropped to the live prefix, and `act_i` IS `cell__chem[t][:nF, 0]`.

    WHAT DIFFERS IS ONLY THE SUBSAMPLE. okuda keeps ~60 frames (the ones the movie draws) and stores
    their row indices in `ticks`; core keeps every recorded row. So the okuda side is read first, its
    `ticks` are handed to the core side, and the same rows are compared. Comparing "all of A against
    all of B" would report 60 arrays against 1801 and call a correct promotion a failure.

    THE LIVE CROP IS A PREFIX, not a scatter: `mesh_seed` sets `occ[:Nv] = 1`, `cell_divide` appends,
    and `cell_die` renumbers the CELL set only -- the vertex set is never holed. So `occ.sum()` is
    the prefix length okuda sliced with, and `[:n]` reproduces it exactly.
    """
    out = []
    p_ok = os.path.join(d, "traj.npz")
    p_core = None
    for root, _dirs, files in os.walk(d):
        if "trajectory.npz" in files:
            p_core = os.path.join(root, "trajectory.npz")
            break
    if os.path.exists(p_ok):                                   # okuda: pos_i / act_i per kept frame
        z = np.load(p_ok, allow_pickle=True)
        n = sum(1 for k in z.files if k.startswith("pos_"))
        for i in range(n):
            out.append((f"pos_{i}", np.asarray(z[f"pos_{i}"])))
            if f"act_{i}" in z.files:
                out.append((f"act_{i}", np.asarray(z[f"act_{i}"])))
        return out, (np.asarray(z["ticks"]).tolist() if "ticks" in z.files else None)
    if p_core:                                                 # core: <set>__pos / __occ / __<block>
        z = np.load(p_core, allow_pickle=True)
        vp, vo = z.get("vertex__pos"), z.get("vertex__occ")
        ch, co = z.get("cell__chem"), z.get("cell__occ")
        if vp is not None:
            rows = ticks if ticks is not None else range(vp.shape[0])
            for i, t in enumerate(rows):
                nv = int(np.asarray(vo[t]).sum()) if vo is not None else vp.shape[1]
                out.append((f"pos_{i}", np.asarray(vp[t][:nv], np.float32)))
                if ch is not None:
                    nf = int(np.asarray(co[t]).sum()) if co is not None else ch.shape[1]
                    out.append((f"act_{i}", np.asarray(ch[t][:nf, 0], np.float32)))
            return out, list(rows)
        for k in sorted(z.files):                              # a non-mesh core run: whatever it wrote
            if k.endswith(("__pos", "__occ", "__state")):
                out.append((k, np.asarray(z[k])))
    return out, None


def _digest(arrs):
    """One sha1 over every array's raw bytes, in order. Bit-for-bit by construction: `tobytes()` is
    the buffer, so a float that differs in its last mantissa bit changes the digest."""
    h = hashlib.sha1()
    for label, a in arrs:
        h.update(label.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def tb_is_core(d):
    """A core run writes `trajectory.npz` and no `traj.npz`."""
    if os.path.exists(os.path.join(d, "traj.npz")):
        return False
    return any("trajectory.npz" in f for _r, _d, f in os.walk(d))


def compare(dir_a, dir_b, tol=0.0):
    """(ok, digest_a, digest_b, report). `report` is a dict, always -- not only on failure.

    BYTE-IDENTITY IS THE DEFAULT AND IT IS NOT ALWAYS THE RIGHT QUESTION. Over 100 frames two runs of
    one spec on one GPU agree to the last mantissa bit, so `tol=0` is a real test. Over 1,800 frames
    with division and T1 in the loop, a single differing bit can move a flip decision -- `edge_flip`
    thresholds on `l < l_th_frac * mean_l` -- and from then on the two runs have DIFFERENT TOPOLOGY,
    at which point "max |delta|" is not small, it is meaningless: the arrays no longer describe the
    same cells. So the report says WHERE the two part company (the first differing frame) as well as
    how far apart they end, and a run's tolerance has to be set against a MEASURED repeatability
    floor -- two runs of the identical spec on the identical code -- not against a number somebody
    liked. `--phase R` measures that floor.

    `tol` is an ABSOLUTE tolerance on position, in the spec's own length units (`world: 80.0` here,
    cells of radius ~5). `tol=0` means byte-identical.
    """
    A, ta = _arrays(dir_a)
    B, tb = _arrays(dir_b, ticks=ta if (ta is not None and tb_is_core(dir_b)) else None)
    rep = dict(n_arrays_a=len(A), n_arrays_b=len(B), tol=tol, ticks=len(ta or []))
    if not A or not B:
        rep["why"] = f"no arrays on the {'A' if not A else 'B'} side"
        return False, _digest(A), _digest(B), rep
    if len(A) != len(B):
        rep["why"] = f"{len(A)} arrays vs {len(B)} -- the runs are not the same length"
        return False, _digest(A), _digest(B), rep

    first_bit, first_over_tol, worst, worst_at = None, None, 0.0, ""
    n_shape = 0
    for i, ((la, a), (lb, b)) in enumerate(zip(A, B)):
        if a.shape != b.shape:
            n_shape += 1
            if first_bit is None:
                first_bit = la
            if first_over_tol is None:
                first_over_tol = f"{la}: shape {a.shape} vs {b.shape}"
            continue
        if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            if first_bit is None:
                first_bit = la
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            m = float(d.max())
            if m > worst:
                worst, worst_at = m, la
            if m > tol and first_over_tol is None:
                first_over_tol = (f"{la}: {int((d > 0).sum())} of {d.size} differ, "
                                  f"max |delta| {m:.3g} > tol {tol:g}")
    rep.update(first_differing_bit=first_bit, max_abs_delta=worst, max_at=worst_at,
               n_shape_mismatches=n_shape)
    ok = (first_bit is None) if tol == 0 else (first_over_tol is None)
    rep["why"] = "" if ok else (first_over_tol or f"first bit differs at {first_bit}")
    if first_bit is not None and ok:
        rep["note"] = (f"NOT bit-identical (first at {first_bit}) but within tol: "
                       f"max |delta| {worst:.3g} at {worst_at}")
    return ok, _digest(A), _digest(B), rep


def _side_dir(pair_dir, side):
    return os.path.join(pair_dir, "A" if side == "a" else "B")


# ---------------------------------------------------------------------------------- the run
def _ptag(phase):
    """A phase as a filesystem-safe prefix: `0` -> "0", `0.5` -> "0p5", `B-core` -> "B_core"."""
    return str(phase).replace(".", "p").replace("-", "_")


def run_pair(phase, spec, side_a, side_b, what, frames, submit=True):
    """THE PHASE IS PART OF THE NAME, and it has to be. `--phase R` runs `b_star` with BOTH sides on
    the working tree, while `--phase B` runs the same spec with side A on a worktree: the two would
    write into `log/okuda/promo_b_star_B` and into `log/promotion/b_star/` together, and the second
    to finish would silently be compared against the first's leftovers. A gate that can overwrite
    its own reference is not a gate."""
    import cluster as C
    pair_dir = os.path.join(OUT, f"{_ptag(phase)}_{spec}")
    os.makedirs(pair_dir, exist_ok=True)
    names = {}
    for tag, side in (("A", side_a), ("B", side_b)):
        run_name = f"promo_{_ptag(phase)}_{spec}_{tag}"
        names[tag] = (side, run_name)
        if submit:
            # THE CORE SIDE NEEDS A SPEC TOO, and in its own folder: `Plexus_Main.py -o generate
            # promotion/<name>` resolves to `config/promotion/<name>.yaml`. Writing only the okuda
            # side's copy is how side A once died on a missing file while side B passed.
            _spec_copy(spec, run_name, frames, cfg_dir=(
                _side_paths(side)[1] if side.startswith("okuda")
                else os.path.join(ROOT, "config", "promotion")))
    if submit:
        lines = [_bsub_lines(pair_dir, spec, side, rn, frames) for side, rn in names.values()]
        runner = os.path.join(pair_dir, "_submit.sh")
        with open(runner, "w") as f:
            f.write("#!/bin/bash -l\n" + "\n".join(lines) + "\n")
        os.chmod(runner, 0o755)
        C._ssh(f"nohup bash {C.cpath(runner)} > {C.cpath(runner)}.log 2>&1 < /dev/null &",
               timeout=30)
        print(f"  [{spec}] submitted BOTH sides together: "
              f"{names['A'][1]} ({side_a}) | {names['B'][1]} ({side_b})", flush=True)
    return names, pair_dir


def _landed(side, run_name, pair_dir):
    """Has this side written the file the comparison reads?"""
    return os.path.exists(_out_path(side, run_name, pair_dir))


def _out_path(side, run_name, pair_dir):
    if side.startswith("okuda"):
        return os.path.join(_side_paths(side)[2], run_name, "traj.npz")
    return os.path.join(pair_dir, "graphs_data", "promotion", run_name, "trajectory.npz")


def _collect_live(names, pair_dir):
    """Mirror every side that has landed, now, without waiting for its partner."""
    for tag, (side, run_name) in names.items():
        if not _landed(side, run_name, pair_dir):
            continue
        src = os.path.dirname(_out_path(side, run_name, pair_dir))
        dst = os.path.join(pair_dir, tag)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as e:                    # a file being written under us is not a failure
            print(f"    [{os.path.basename(pair_dir)}/{tag}] partial mirror ({type(e).__name__})")


def collect(spec, names, pair_dir):
    """Copy each side's output into the pair directory, so the comparison reads from one place and
    the okuda tree is left alone."""
    for tag, (side, run_name) in names.items():
        dst = os.path.join(pair_dir, tag)
        src = (os.path.join(_side_paths(side)[2], run_name) if side.startswith("okuda")
               # the core runner writes under `--output_root/graphs_data/<pre_folder>/<name>/`
               else os.path.join(pair_dir, "graphs_data", "promotion", run_name))
        if os.path.isdir(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
    return {t: os.path.join(pair_dir, t) for t in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None, help="only the pairs of one phase (0, 0.5, B, C, D)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--frames", type=int, default=None,
                help="override every row's length; default = the row's own")
    ap.add_argument("--compare-only", action="store_true",
                    help="do not submit; compare what is already in log/promotion/")
    ap.add_argument("--wait-min", type=float, default=90.0)
    ap.add_argument("--tol", type=float, default=None,
                    help="override every row's absolute position tolerance "
                         "(0 = byte-identical, which is every row's default)")
    a = ap.parse_args()

    pairs = [p for p in PAIRS if a.all or (a.phase is not None and str(p[0]) == str(a.phase))]
    if not pairs:
        print("  no pair selected -- use --phase or --all"); return 2

    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for phase, spec, nfr, tol, sa, sb, what in pairs:
        # --frames on the command line overrides every row; otherwise the row decides, and
        # `None` on the row means the spec's own `general.n_frames`.
        frames = a.frames if a.frames is not None else nfr
        try:
            names, pd = run_pair(phase, spec, sa, sb, what, frames, submit=not a.compare_only)
        except FileNotFoundError as e:
            print(f"  [{spec}] SKIPPED: {e}")
            continue
        jobs.append((phase, spec, tol, sa, sb, what, names, pd))

    if not a.compare_only and jobs:
        # WAIT ON THE ARTEFACTS, NOT ONLY ON THE QUEUE. `bjobs` empties when the SIMULATION ends and
        # the run then spends minutes writing metrics, the strip and two mp4s; a compare fired at
        # that moment reads a directory with no `traj.npz` in it and reports DIFFER. Worse, if the
        # wrapper is interrupted before this loop returns, `collect` never runs and the outputs stay
        # in each side's own tree -- which is how `log/promotion/apop_one/` ended up holding six
        # scripts and no mp4 while both runs had finished perfectly.
        import cluster as C
        print(f"  waiting for {2 * len(jobs)} job(s); each side is MIRRORED INTO "
              f"log/promotion/ AS SOON AS IT LANDS...", flush=True)
        t0 = time.time()
        while time.time() - t0 < a.wait_min * 60:
            # MIRROR WHAT IS THERE, EVERY POLL. Collecting only after the LAST job finished meant
            # `log/promotion/R_b_star/` and `B_core_b_star/` sat empty for half an hour while four of
            # the six sides had already written their movies -- there was nothing to look at, and no
            # way to tell a slow run from a dead one. `_collect_live` copies each side the moment its
            # `traj.npz`/`trajectory.npz` exists, and it is cheap to repeat: `shutil.copytree` onto a
            # side that has not changed is one stat per file.
            for _p, _sp, _t, _sa, _sb, _w, names, pd in jobs:
                _collect_live(names, pd)
            # `_ssh` RETURNS A `CompletedProcess`, NOT THE OUTPUT. `str()` of one begins
            # "CompletedProcess(args=[...", never "0", so the queue test was false on every poll and
            # the loop always ran to `--wait-min`. An ssh timeout returns None, and an unreachable
            # login node is NOT an empty queue, so that reads as "still waiting".
            st = C._ssh("bjobs -w 2>/dev/null | grep -c promo_ || true", timeout=30)
            queue_empty = st is not None and (st.stdout or "").strip().startswith("0")
            landed = all(_landed(side, rn, pd)
                         for *_r, names, pd in jobs for side, rn in names.values())
            if queue_empty and landed:
                break
            time.sleep(60)

    rows, bad = [], 0
    for phase, spec, tol, sa, sb, what, names, pd in jobs:
        dirs = collect(spec, names, pd)
        # THE MP4 IS PART OF THE EVIDENCE. A digest says two runs differ; the movie says HOW, and a
        # gate whose only output is a hex string sends you back to the cluster to find out.
        for t, d in dirs.items():
            mp4 = [f for f in sorted(os.listdir(d)) if f.endswith(".mp4")] if os.path.isdir(d) else []
            if not mp4:
                print(f"  [{spec}] {t}: no mp4 collected -- the run may not have finished rendering")
        ok, da, db, rep = compare(dirs["A"], dirs["B"], tol=a.tol if a.tol is not None else tol)
        bad += 0 if ok else 1
        rows.append(dict(phase=str(phase), spec=spec, a=sa, b=sb, ok=bool(ok), tol=rep["tol"],
                         digest_a=da, digest_b=db, why=rep["why"], report=rep, what=what))
    print(f"\n  {'phase':6s} {'spec':26s} {'A':12s} {'B':8s} {'A digest':18s} {'B digest':18s} result")
    for r in rows:
        print(f"  {r['phase']:6s} {r['spec']:26s} {r['a']:12s} {r['b']:8s} "
              f"{r['digest_a']:18s} {r['digest_b']:18s} "
              + ("IDENTICAL" if r["ok"] and not r["report"].get("first_differing_bit")
                 else ("WITHIN TOL -- " + r["report"]["note"]) if r["ok"]
                 else f"DIFFER -- {r['why']}"))
    # ONE FILE PER PHASE, PLUS A MERGED ONE. A single `promotion_identical.json` was overwritten by
    # whichever phase ran last, so the archive held one phase's rows and read as if it held all of
    # them -- the same silent-truncation failure the mp4 collection had.
    tag = str(a.phase) if a.phase is not None else "all"
    json.dump(rows, open(os.path.join(OUT, f"promotion_identical_{tag}.json"), "w"), indent=1)
    merged = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("promotion_identical_") and f.endswith(".json"):
            for r in json.load(open(os.path.join(OUT, f))):
                merged[(r["phase"], r["spec"])] = r
    allrows = [merged[k] for k in sorted(merged)]
    json.dump(allrows, open(os.path.join(OUT, "promotion_identical.json"), "w"), indent=1)
    n_ok = sum(1 for r in allrows if r["ok"])
    print(f"\n  {len(rows) - bad}/{len(rows)} identical this run; "
          f"{n_ok}/{len(allrows)} across every phase on record -> "
          f"{os.path.relpath(os.path.join(OUT, 'promotion_identical.json'), ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
