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
    # phase   spec               frames  side A         side B     what it exercises
    (0,  "b_null_plain",           100, "okuda@HEAD",  "okuda",  "mesh_seed, cell_mechanics, edge_flip, topo_record"),
    # A DEATH SPEC, because `b_null_plain` never calls `cell_die` -- its ledger is cell_divide 26,
    # cell_geometry 83, cell_mechanics 101, edge_flip 27, mesh_seed 1, topo_record 101 and no death
    # at all. Removal is the first operation in this engine that MOVES A ROW, so a renumber change
    # gated only on that spec passes without executing a line of what changed.
    (0,  "apop_one",               100, "okuda@HEAD",  "okuda",  "+ cell_die -> H.renumber_set: ONE row moves"),
    # ---- the death scenes. One row moved is the correctness case and it is not the hard case.
    # `renumber_set` is a GATHER through `keep`, and the four rows below ask progressively more of
    # it: 200 rows out of order, then a patch, then nine bands, then most of the sheet. Each is an
    # existing okuda spec run at its own length, and each is a figure on the minisite -- so the
    # archive under `log/promotion/<spec>/` is both the gate's evidence and the scene itself.
    (0,  "apop_many",             None, "okuda@HEAD",  "okuda",  "200 SCATTERED deaths -- an interleaved `keep`, not a truncation"),
    (0,  "apop_patch_big",        None, "okuda@HEAD",  "okuda",  "293 of 2000 in a 45-deg cap -> the surface is drawn INWARD (invagination 0.0369)"),
    (0,  "apop_rings9",           None, "okuda@HEAD",  "okuda",  "895 of 2000 in nine bands -> closes over every gap, stays a sphere (red_vol 0.9813)"),
    (0,  "apopgeo_half",          None, "okuda@HEAD",  "okuda",  "285 of 400 above the equator -> topology survives, sphere -> ellipsoid (gyr_prolate 1.869)"),
    (0.5, "ecm_block",            None, "okuda@HEAD",  "okuda",  "mpm_scatter/gather/grid_update/strain, ecm_seed, ecm_stress"),
    ("B", "b_gs_plain_soft_lo",    100, "okuda",       "core",   "+ cell_chem_seed/diffuse/react, cell_geometry, cell_neighbours"),
    ("B", "b_star",                100, "okuda",       "core",   "+ cell_grow, cell_divide, interface_tension, cell_chem_from_shape"),
    ("C", "01c_tissue",            100, "okuda",       "core",   "junction_myosin (both pools), junction_sync, cytokinetic_ring"),
    ("D", "04_spheroid_ecm_pass2", 100, "okuda",       "core",   "mesh_contact, mesh_inside, ecm_*, bm_*"),
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
def _arrays(d):
    """Every recorded array of a run, in a fixed order, whichever writer produced it.

    Returns [(label, ndarray)]. Reading BOTH layouts here is what lets an okuda run and a core run
    be compared at all: the file names and the key names differ, the numbers must not.
    """
    out = []
    p_ok = os.path.join(d, "traj.npz")
    p_core = None
    for root, _dirs, files in os.walk(d):
        if "trajectory.npz" in files:
            p_core = os.path.join(root, "trajectory.npz")
            break
    if os.path.exists(p_ok):                                   # okuda: pos_i / act_i per frame
        z = np.load(p_ok, allow_pickle=True)
        n = sum(1 for k in z.files if k.startswith("pos_"))
        for i in range(n):
            out.append((f"pos_{i}", np.asarray(z[f"pos_{i}"])))
            if f"act_{i}" in z.files:
                out.append((f"act_{i}", np.asarray(z[f"act_{i}"])))
    elif p_core:                                               # core: <set>__pos / <set>__occ
        z = np.load(p_core, allow_pickle=True)
        for k in sorted(z.files):
            if k.endswith(("__pos", "__occ", "__state")):
                out.append((k, np.asarray(z[k])))
    return out


def _digest(arrs):
    """One sha1 over every array's raw bytes, in order. Bit-for-bit by construction: `tobytes()` is
    the buffer, so a float that differs in its last mantissa bit changes the digest."""
    h = hashlib.sha1()
    for label, a in arrs:
        h.update(label.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def compare(dir_a, dir_b):
    """(ok, digest_a, digest_b, first_difference). The first difference is NAMED, because "they
    differ" sends you to read two 60-frame trajectories and "pos_37 differs in 12 of 4002 rows,
    max |delta| 3.1e-07" sends you to one operator."""
    A, B = _arrays(dir_a), _arrays(dir_b)
    if not A or not B:
        return False, _digest(A), _digest(B), f"no arrays on {'A' if not A else 'B'} side"
    if len(A) != len(B):
        return False, _digest(A), _digest(B), f"{len(A)} arrays vs {len(B)}"
    for (la, a), (lb, b) in zip(A, B):
        if a.shape != b.shape:
            return False, _digest(A), _digest(B), f"{la}: shape {a.shape} vs {b.shape}"
        if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            return (False, _digest(A), _digest(B),
                    f"{la}: {int((d > 0).sum())} of {d.size} differ, max |delta| {d.max():.3g}")
    return True, _digest(A), _digest(B), ""


def _side_dir(pair_dir, side):
    return os.path.join(pair_dir, "A" if side == "a" else "B")


# ---------------------------------------------------------------------------------- the run
def run_pair(phase, spec, side_a, side_b, what, frames, submit=True):
    import cluster as C
    pair_dir = os.path.join(OUT, spec)
    os.makedirs(pair_dir, exist_ok=True)
    names = {}
    for tag, side in (("A", side_a), ("B", side_b)):
        run_name = f"promo_{spec}_{tag}"
        names[tag] = (side, run_name)
        if submit and side.startswith("okuda"):
            _spec_copy(spec, run_name, frames, cfg_dir=_side_paths(side)[1])
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


def collect(spec, names, pair_dir):
    """Copy each side's output into the pair directory, so the comparison reads from one place and
    the okuda tree is left alone."""
    for tag, (side, run_name) in names.items():
        dst = os.path.join(pair_dir, tag)
        if side.startswith("okuda"):
            src = os.path.join(_side_paths(side)[2], run_name)
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
    a = ap.parse_args()

    pairs = [p for p in PAIRS if a.all or (a.phase is not None and str(p[0]) == str(a.phase))]
    if not pairs:
        print("  no pair selected -- use --phase or --all"); return 2

    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for phase, spec, nfr, sa, sb, what in pairs:
        # --frames on the command line overrides every row; otherwise the row decides, and
        # `None` on the row means the spec's own `general.n_frames`.
        frames = a.frames if a.frames is not None else nfr
        try:
            names, pd = run_pair(phase, spec, sa, sb, what, frames, submit=not a.compare_only)
        except FileNotFoundError as e:
            print(f"  [{spec}] SKIPPED: {e}")
            continue
        jobs.append((phase, spec, sa, sb, what, names, pd))

    if not a.compare_only and jobs:
        # WAIT ON THE ARTEFACTS, NOT ONLY ON THE QUEUE. `bjobs` empties when the SIMULATION ends and
        # the run then spends minutes writing metrics, the strip and two mp4s; a compare fired at
        # that moment reads a directory with no `traj.npz` in it and reports DIFFER. Worse, if the
        # wrapper is interrupted before this loop returns, `collect` never runs and the outputs stay
        # in each side's own tree -- which is how `log/promotion/apop_one/` ended up holding six
        # scripts and no mp4 while both runs had finished perfectly.
        import cluster as C
        print(f"  waiting for {2 * len(jobs)} job(s)...", flush=True)
        t0 = time.time()
        while time.time() - t0 < a.wait_min * 60:
            # `_ssh` RETURNS A `CompletedProcess`, NOT THE OUTPUT. `str()` of one begins
            # "CompletedProcess(args=[...", never "0", so the queue test was false on every poll and
            # the loop always ran to `--wait-min` -- which is the real reason `collect` never ran and
            # `log/promotion/<spec>/` held scripts and no mp4. An ssh timeout returns None, and an
            # unreachable login node is NOT an empty queue, so that reads as "still waiting".
            st = C._ssh("bjobs -w 2>/dev/null | grep -c promo_ || true", timeout=30)
            queue_empty = st is not None and (st.stdout or "").strip().startswith("0")
            landed = all(os.path.exists(os.path.join(_side_paths(side)[2], rn, "traj.npz"))
                         if side.startswith("okuda") else True
                         for *_r, names, _pd in jobs for side, rn in names.values())
            if queue_empty and landed:
                break
            time.sleep(60)

    rows, bad = [], 0
    for phase, spec, sa, sb, what, names, pd in jobs:
        dirs = collect(spec, names, pd)
        # THE MP4 IS PART OF THE EVIDENCE. A digest says two runs differ; the movie says HOW, and a
        # gate whose only output is a hex string sends you back to the cluster to find out.
        for t, d in dirs.items():
            mp4 = [f for f in sorted(os.listdir(d)) if f.endswith(".mp4")] if os.path.isdir(d) else []
            if not mp4:
                print(f"  [{spec}] {t}: no mp4 collected -- the run may not have finished rendering")
        ok, da, db, why = compare(dirs["A"], dirs["B"])
        bad += 0 if ok else 1
        rows.append(dict(phase=str(phase), spec=spec, a=sa, b=sb, ok=bool(ok),
                         digest_a=da, digest_b=db, why=why, what=what))
    print(f"\n  {'phase':6s} {'spec':26s} {'A':12s} {'B':8s} {'A digest':18s} {'B digest':18s} result")
    for r in rows:
        print(f"  {r['phase']:6s} {r['spec']:26s} {r['a']:12s} {r['b']:8s} "
              f"{r['digest_a']:18s} {r['digest_b']:18s} "
              + ("IDENTICAL" if r["ok"] else f"DIFFER -- {r['why']}"))
    json.dump(rows, open(os.path.join(OUT, "promotion_identical.json"), "w"), indent=1)
    print(f"\n  {len(rows) - bad}/{len(rows)} identical -> "
          f"{os.path.relpath(os.path.join(OUT, 'promotion_identical.json'), ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
