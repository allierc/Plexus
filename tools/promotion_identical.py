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
`plexus/engine.py` leaves CUDA reductions on their fast atomic default, because forcing
`use_deterministic_algorithms(True)` on every run cost a measured 4.4x on si_waterfall and no spec
in the corpus asked for it. Determinism is therefore the PROPERTY OF THIS TOOL, not of the engine:
each side runs with `PLEXUS_STRICT_DETERMINISM=1`, which turns it on with `warn_only=False`, so a
kernel with no deterministic implementation raises instead of quietly downgrading -- and a
downgraded kernel is exactly where two runs of one spec stop matching. A side that dies of it FAILS
rather than passing on a comparison it never earned.

    THE SIDES
    ---------
    okuda            run_one.py, from the working tree
    okuda@<ref>      run_one.py, from a git worktree at <ref> -- how "before an engine change" is
                     regenerated rather than remembered
    core             Plexus_Main.py -o generate, the promoted registry

    SIDE A IS `okuda@0da57dd0` ON EVERY ROW THAT IS NOT MEASURING THE FLOOR
    ----------------------------------------------------------------------
`discovery_okuda/ops` is now 315 lines of re-export shims: its `cell_die`, `edge_flip` and the rest
ARE the core's classes. So a row reading `okuda` vs `core` -- or `okuda@HEAD` vs `okuda` -- runs the
SAME CODE ON BOTH SIDES and can only ever detect a divergence between two copies of one bug. It
cannot detect a regression, because a regression moves both sides together.

That is not hypothetical. `Hierarchy.renumber_set` guarded on `hasattr(self.levels, "get")`, and
`levels` is an `nn.ModuleDict` WITH NO `.get`, so the method returned False having touched nothing on
every call for the whole promotion; the cell chemistry of every run with a death in it was scrambled
from the first extrusion, and NINETEEN ROWS WERE GREEN THROUGHOUT. The only row that caught it was
the one pinned to a commit.

`0da57dd0` is the last commit before the promotion touched an operator -- the code the archive in
`log/okuda` was actually produced by. It is the reference. The `R` rows are the deliberate exception:
they run one tree against itself because they measure the repeatability floor, and same-tree is the
whole point of them.

    python tools/promotion_identical.py --phase 0        the pairs for one phase
    python tools/promotion_identical.py --all            everything in PAIRS
    python tools/promotion_identical.py --compare-only   skip submission, compare what is on disk
"""
from __future__ import annotations

import argparse
import glob
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
# TWIN OUTPUT IS GENERATED DATA, so it lives with the generated data. `graphs_data/` is the symlink
# onto the GraphData filer that every other run in this repo writes through; `log/` is for the
# harness's own bookkeeping. Moved here on 4 September -- a twin run is two full simulations, which
# is the same kind of object as anything under `graphs_data/<campaign>/`, and putting it under
# `log/` was what made a 393 GB tree look like a log directory and invite a `rm -rf`.
#
#
# `_worktrees/` DOES NOT MOVE WITH IT, and that separation is the whole lesson of 4 September. The
# worktrees are CODE -- side A executes `run_one.py` from inside one -- and they used to sit in the
# output tree. Deleting the output therefore deleted the code out from under fifteen running jobs,
# every okuda side lost its `cd` target mid-flight, and the wrapper then died on
# `git worktree add ... exit 128` because git still held the registration for a directory that was
# gone. Output is deletable at any time by design; the thing a job is running must not be inside it.
OUT = os.path.join(ROOT, "graphs_data", "promotion")
WORKTREES = os.path.join(ROOT, "log", "_worktrees")

# UNSET `DISPLAY` BEFORE ANYTHING CAN LOOK AT IT. A VS Code remote session exports a DISPLAY whose
# X server this container cannot authenticate to, and VTK does not fall back quietly from that: one
# path aborts the PROCESS with `BadValue ... X_GLXCreateContext`, which killed this wrapper after it
# had submitted and was waiting on jobs. `render_vtk.offscreen()` does exactly this, but only once a
# render function is called, which is too late. With no DISPLAY, VTK goes to EGL/OSMesa directly,
# and nothing here ever wants an on-screen window.
os.environ.pop("DISPLAY", None)
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
    (0,  "b_null_plain",         None, 0.0, "okuda@0da57dd0",  "okuda",  "mesh_seed, cell_mechanics, edge_flip, topo_record"),
    # A DEATH SPEC, because `b_null_plain` never calls `cell_die`. Removal is the first operation in
    # this engine that MOVES A ROW, so a renumber change gated only on that spec passes without
    # executing a line of what changed.
    (0,  "apop_one",             None, 0.0, "okuda@0da57dd0",  "okuda",  "+ cell_die -> H.renumber_set: ONE row moves"),
    # ---- the death scenes. One row moved is the correctness case and it is not the hard case.
    # `renumber_set` is a GATHER through `keep`, and these four ask progressively more of it: 200
    # rows out of order, then a patch, then nine bands, then most of the sheet. Each is an existing
    # okuda spec and each is a figure on the minisite, so `log/promotion/<spec>/` is both the gate's
    # evidence and the scene itself.
    (0,  "apop_many",            None, 0.0, "okuda@0da57dd0",  "okuda",  "200 SCATTERED deaths -- an interleaved `keep`, not a truncation"),
    (0,  "apop_patch_big",       None, 0.0, "okuda@0da57dd0",  "okuda",  "293 of 2000 in a 45-deg cap -> the surface is drawn INWARD (invagination 0.0369)"),
    (0,  "apop_rings9",          None, 0.0, "okuda@0da57dd0",  "okuda",  "895 of 2000 in nine bands -> closes over every gap, stays a sphere (red_vol 0.9813)"),
    (0,  "apopgeo_half",         None, 0.0, "okuda@0da57dd0",  "okuda",  "285 of 400 above the equator -> topology survives, sphere -> ellipsoid (gyr_prolate 1.869)"),
    (0.5, "ecm_block",           None, 0.0, "okuda@0da57dd0",  "okuda",  "mpm_scatter/gather/grid_update/strain, ecm_seed, ecm_stress"),
    # ---- Phase B: nine okuda operator files became two modules in `src/plexus/operators/`. Side A
    # is okuda BEFORE the move, side B okuda AFTER it, so what is under test is the MOVE, through
    # the runner that has always driven these operators. That is a necessary step and NOT the claim
    # the promotion is making -- `B-core` below is.
    ("B", "b_gs_plain_soft_lo",  None, 0.0, "okuda@0da57dd0",  "okuda",  "+ seed_cell_chem/diffuse/react, cell_geometry, cell_neighbours"),
    ("B", "b_star",              None, 0.0, "okuda@0da57dd0",  "okuda",  "+ cell_grow, cell_divide, interface_tension, cell_chem_from_shape"),
    # ---- B-core: THE ACTUAL CLAIM. `Plexus_Main.py -o generate` against `run_one.py`. Different
    # runner, different recorder, different writer -- the same numbers.
    ("B-core", "b_gs_plain_soft_lo", None, 0.0, "okuda@0da57dd0",   "core",   "the same run, from the core registry, no okuda import"),
    ("B-core", "b_star",             None, 0.0, "okuda@0da57dd0",   "core",   "the same run, from the core registry, no okuda import"),
    # ---- R: THE REPEATABILITY FLOOR. Same spec, same code, same commit, same queue -- twice. Any
    # tolerance above must be set from what this measures, because a gate tighter than the platform's
    # own noise fails on runs that are correct, and a gate looser than it passes runs that are not.
    ("R", "b_star",              None, 0.0, "okuda",       "okuda",  "1800 f, 2000->12272 cells, twice -- MEASURED FLOOR 0.0"),
    ("R", "b_null_plain",        None, 0.0, "okuda",       "okuda",  "1800 f, twice -- MEASURED FLOOR 0.0"),
    # ---- G: EVERY LIFTED GATE, AS A TWIN RUN. The gates grade the CORE against thresholds; that
    # says the core is right about the numbers a human wrote down, and it does NOT say the core and
    # okuda agree. These rows are the missing half: the same gate spec through `run_one.py` and
    # through `Plexus_Main.py`, both fresh, both on gpu_l4, compared array by array. A gate is only
    # promoted when both are true.
    ("G", "gates/gate_00_spheroid",     None, 0.0, "okuda@0da57dd0", "core", "the growth line: seed, geometry, grow, belt, mechanics, T1, divide, sync"),
    ("G", "gates/gate_01_nosync",       None, 0.0, "okuda@0da57dd0", "core", "gate 01's own arm: the belt WITHOUT the re-keying operator"),
    ("G", "gates/gate_01_nomyosin",     None, 0.0, "okuda@0da57dd0", "core", "gate 01's contrast arm: the same tissue with no belt"),
    # 01b IS NOT IN THIS SUITE, and the gap is declared rather than left as an absence. Its spec was
    # a RECONSTRUCTION -- `log/okuda_ECM/01b_myosin_pools/spec.yaml` is one of the 43 PROSE records
    # (`what` and `operators_exercised`, no `sets`, no `schedule`), so there was never anything to
    # replay -- and it was deleted on 2026-09-04 because it carried NO `_gate:` BLOCK: `run_gates`
    # filters on that key, so for its whole life it was a spec in the gates folder that could not be
    # run and could not fail, and this row was twinning it anyway. The two operators it was meant to
    # cover, `medioapical_myosin` and `junction_myosin[two_pool]`, are therefore UNTWINNED TODAY.
    # Restoring it means writing the thresholds first and the row second, in that order.
    # ---- THE TWO MPM GATES HAVE NO OKUDA TWIN, and that is a fact about okuda's runner rather
    # than a gap in the promotion. `run_one.py` reads `H.level("vertex")` in three places -- the
    # heartbeat, the live snapshot and the cell ceiling -- so a spec with no mesh set dies with
    # `KeyError: 'vertex'` before frame 0. okuda is a MESH HARNESS; gates 02 and 04-pass-2 are pure
    # MPM. So the comparison available for them is core against core across a commit, which is a
    # real regression check, and it is labelled as that instead of being dressed up as agreement
    # with okuda. `cc52f512` is the commit at which every operator had landed in core.
    ("G", "gates/gate_02_ecm_block",    None, 0.0, "core@cc52f512", "core", "MLS-MPM: ecm_seed, the four-step cycle, ecm_stress, gravity (no okuda twin: mesh-free)"),
    # ---- REP: REPLICATION, on specs the CAMPAIGN wrote rather than specs a human did.
    # Every row above tests a spec somebody chose for the promotion; these three were emitted by
    # `round.py` from its own search, carry 14-15 operators apiece, and are the compositions the
    # okuda work actually produced. A promotion that reproduces its own test set and not the corpus
    # it was built from has reproduced the tests. They also exercise the two RENAMED seed operators
    # (`seed_mesh`, `seed_cell_chem`) on the corpus that uses them, which is 324 of the 461 specs.
    ("REP", "r010_00_ctrl",      None, 0.0, "okuda@0da57dd0",       "core",   "a campaign control: 14 operators, 1800 frames"),
    ("REP", "r020_00_ctrl",      None, 0.0, "okuda@0da57dd0",       "core",   "round 20's control -- the best composition the search produced"),
    ("REP", "r023_07",           None, 0.0, "okuda@0da57dd0",       "core",   "15 operators; the run whose rerun reproduced it exactly (n_tubes 12, protr 1.765)"),
    # ---- BISECT: r023_07 against okuda BEFORE the promotion. The REP row compared current-okuda
    # to current-core and they agreed -- but the archived run of the same spec is healthy (12,608
    # cells, no extinction) and both of mine are NaN from frame 889 (2,995 cells). A twin where both
    # sides are current cannot see a change that moved BOTH sides. This is the row that can.
    # THE VERIFICATION ROW for the renumber_set fix: the pristine pre-promotion tree against the
    # current one, on the spec that exposed it. This must be IDENTICAL, and it is the only row that
    # can say so -- every other comparison has both sides on the current tree, which is exactly how
    # a change that moved BOTH sides stayed invisible for nineteen green rows.
    ("BISECT", "r023_07",        None, 0.0, "okuda@0da57dd0", "core", "pristine baseline vs the fixed core"),
    ("C", "01c_tissue",          None, 0.0, "okuda@0da57dd0",       "core",   "junction_myosin (both pools), junction_sync, cytokinetic_ring"),
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

    IT LIVES IN `log/_worktrees/`, NOT IN THE OUTPUT TREE. See the note on `OUT`: a worktree is the
    code a running job is executing, and the output tree is meant to be deletable at any moment.
    """
    d = os.path.join(WORKTREES, ref.replace("/", "_"))
    if os.path.isdir(os.path.join(d, ".git")) or os.path.isfile(os.path.join(d, ".git")):
        return d
    os.makedirs(os.path.dirname(d), exist_ok=True)
    subprocess.run(["git", "-C", ROOT, "worktree", "add", "--detach", d, ref],
                   check=True, capture_output=True, timeout=300)
    return d


def _side_root(side):
    """The repository root a side runs from: this working tree, or a git worktree at `<ref>`.

    ONE FUNCTION FOR BOTH SIDE FAMILIES, so `core@<ref>` gets what `okuda@<ref>` already had. It is
    needed because `run_one.py` CANNOT RUN A MESH-FREE SPEC: it reads `H.level("vertex")` in three
    places and dies with `KeyError: 'vertex'` on a pure-MPM gate. So for gate 02 there is no
    okuda-versus-core twin to be had -- okuda's runner is a mesh harness -- and the comparison that
    IS available is core against core across a commit, which is a real regression check and is
    stated as that rather than dressed up as agreement with okuda.
    """
    return _worktree(side.split("@", 1)[1]) if "@" in side else ROOT


def _side_paths(side):
    """(home, config dir, log dir) for a side.

    A WORKTREE IS A WHOLE TREE, not just a copy of the code: it has its own `config/okuda/` from the
    commit and writes its runs into its own `log/okuda/`. The first version of this wrote the spec
    into the main tree's config and the worktree job died on a missing file, which is the honest
    failure -- the two sides must not share anything the comparison could smuggle state through.
    """
    root = _side_root(side)
    home = os.path.join(root, "discovery_okuda") if side.startswith("okuda") else root
    return home, os.path.join(root, "config", "okuda"), os.path.join(root, "log", "okuda")


# =============================================================================================
#  SUITES: rows generated from an ARCHIVE rather than typed. `--suite ecm` compares every runnable
#  spec under `log/okuda_ECM/`, `--suite base` a ladder through `log/okuda/`. Generated because
#  there are 13 and 408 of them and a hand-typed list of either would be out of date by Friday.
# =============================================================================================
def _loadable(path):
    try:
        import plexus.operators                                       # noqa: F401
        import plexus.schema as S
        S.load(path)
        return True
    except Exception:
        return False


def suite_ecm():
    """Every `log/okuda_ECM/<run>/spec.yaml` that is a SPEC and loads.

    THERE ARE 96 DIRECTORIES AND 13 SPECS. The rest are prose records: 43 carry only
    `what`/`measures`/`operators_exercised`, and 37 more have an `operators:` list whose entries are
    SENTENCES ("bm_elastic (StVK on the rest metric)"). Those rigs were driven from Python and never
    wrote a machine-readable spec, so they cannot be replayed -- only RECONSTRUCTED, which is what
    `config/gates/` is. The 13 that can be replayed are replayed.
    """
    import yaml as _y
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "log", "okuda_ECM", "*", "spec.yaml"))):
        n = os.path.basename(os.path.dirname(d))
        if n.startswith("_") or not _loadable(d):
            continue
        # THE SIDES ARE CHOSEN BY THE SPEC, not by hand. `run_one.py` reads `H.level("vertex")` in
        # three places, so okuda cannot run a mesh-free spec at all -- and all thirteen of these are
        # `cell` + `mpm_particle`, the pure-MPM half of the prototype. Asking okuda anyway would
        # burn thirteen GPU slots to collect thirteen identical KeyErrors. They compare core against
        # core across the commit where every operator had landed, which is a real regression check
        # and is labelled as one.
        try:
            has_mesh = "vertex" in (_y.safe_load(open(d)).get("sets") or {})
        except Exception:
            has_mesh = False
        A = "okuda" if has_mesh else "core@cc52f512"
        out.append(("ECM", f"ecm/{n}", None, 0.0, A, "core",
                    f"archived rig {n}" + ("" if has_mesh else " (mesh-free: no okuda twin)")))
    return out


def suite_base(n_want=20):
    """A ladder through `log/okuda/`, simple to complex, by operator count then frame count.

    408 of the 408 archived `spec_run.yaml` load, so the constraint is GPU rather than
    availability. Twenty spread across the range says more than twenty of the same size: the
    cheapest rows catch a broken import in minutes and the dearest exercise fifteen operators over
    1,800 frames.
    """
    import yaml as _y
    cand = []
    for d in sorted(glob.glob(os.path.join(ROOT, "log", "okuda", "*", "spec_run.yaml"))):
        n = os.path.basename(os.path.dirname(d))
        if n.startswith("promo_"):
            continue
        try:
            c = _y.safe_load(open(d))
            cand.append((len(c.get("operators") or []), int(c["general"]["n_frames"]), n))
        except Exception:
            continue
    cand.sort()
    if not cand:
        return []
    step = max(1, len(cand) // n_want)
    pick = cand[::step][:n_want]
    # SIDE A IS THE PRISTINE BASELINE, NOT THE LIVE OKUDA TREE. `discovery_okuda/ops` is now 315
    # lines of re-export shims: its `cell_die` and `edge_flip` ARE the core's, so a row comparing
    # `okuda` against `core` runs the same code twice and can only detect a divergence between two
    # copies of one bug. That is exactly how the broken `renumber_set` stayed green for nineteen
    # rows. `0da57dd0` is the last commit before the promotion touched an operator, and it is the
    # reference the archive was produced by -- it is the only side A that can catch a regression.
    return [("BASE", f"base/{n}", None, 0.0, "okuda@0da57dd0", "core",
             f"{k} operators, {f} frames") for k, f, n in pick]


# =============================================================================================
# THE MINISITE'S OWN SCENES, grouped by the four headings they appear under on the front page.
#
# WHY THESE AND NOT A SAMPLE. `suite_base` picks twenty rows spread across the archive by operator
# and frame count -- good coverage, but nobody has ever LOOKED at most of them. These are the clips
# on the front page: if the promotion changed one of them, it changed the thing a reader judges the
# project by, and it would be found by a visitor rather than by a gate.
#
# THE 2D TURING SCENES ARE NOT HERE, and that is a gap, not an oversight. The three clips under
# "Turing -- Gray-Scott reaction-diffusion" (`turing2d_coral`, `turing2d_chi_small`,
# `turing2d_two_species`) exist ONLY as text inlined in `index.qmd`'s hover panels -- there is no
# yaml for them anywhere in the tree -- and they are `dim: 2` with no mesh, so `run_one.py` cannot
# run them at all (it reads `H.level("vertex")` in three places). They have no pristine twin that
# could exist. `atlas/turing_coral` is the mesh-backed Gray-Scott scene and stands in for the
# section's MECHANISM; it is not the same clip and is not labelled as one.
_MINISITE = [
    # section on the front page                     spec
    ("Turing -- Gray-Scott",                        "atlas/turing_coral"),
    ("Epithelial mechanics -- a vertex model",      "atlas/vertex_spheroid"),
    # THE SCENE IS THE BUILT TISSUE, NOT ITS PARENT SPEC. The front page's "grow & divide" clip is
    # what `tissue.build(frames=401, buffer_x=4, myosin=1.0)` produced from `cellfix_B_new.yaml`,
    # and `build` makes three edits: both reservoirs x4, `junction_myosin` before `cell_mechanics`,
    # `junction_sync` after the topology operators. Running the bare parent instead is a different
    # run and it shows -- at buffer_x=1 the vertex array is pinned at 6,396, divisions are refused
    # for want of room, and the clip is a reservoir, not a tissue. `gates/gate_00_spheroid` IS that
    # built spec, lifted to a file, and it already runs pristine-identical.
    ("Epithelial mechanics -- a vertex model",      "gates/gate_00_spheroid"),
    ("Cell death -- sculpting an epithelial surface", "apop_patch_big"),
    ("Cell death -- sculpting an epithelial surface", "apop_rings9"),
    ("Cell death -- sculpting an epithelial surface", "apopgeo_half"),
    ("Growing tissue + Turing pattern",             "atlas/turing_grow_divide"),
    ("Growing tissue + Turing pattern",             "grow_divide"),
    ("Growing tissue + Turing pattern",             "r021_12"),
    # THE FOLDER IS NAMED because two different models carry this name. `_superseded_r001-r029`'s
    # copy is "uniform growth (ungated, rho baseline only)" with `connections: []` and `impl: spot`;
    # this one is the gated route -- `impl: cone` plus `cell_chem_react -> cell_grow.gate` -- which
    # is what the minisite's own inlined spec for the "morphogen + growth" clip shows. An ungated
    # run under that caption would be a different experiment wearing the right label.
    ("Growing tissue + Turing pattern",             "okuda/_superseded_pre_basis/r005_06"),
    ("Growing tissue + Turing pattern",             "r013_05"),
    ("Growing tissue + Turing pattern",             "r016_01"),
    ("Growing tissue + Turing pattern",             "r017_00_ctrl"),
    ("Growing tissue + Turing pattern",             "r020_01"),
    ("Growing tissue + Turing pattern",             "r020_00_ctrl"),
    ("Growing tissue + Turing pattern",             "r021_06"),
    ("Growing tissue + Turing pattern",             "tsd_max"),
    ("Growing tissue + Turing pattern",             "sc_antiphase"),
]


def suite_minisite():
    """Every front-page scene that HAS a pristine twin, at its own length, pristine against core."""
    return [("MINISITE", spec, None, 0.0, "okuda@0da57dd0", "core", section)
            for section, spec in _MINISITE]


def suite_turing():
    """The two-species Gray-Scott exploration -- CORE ONLY, and not a gate.

    These four are not promotion rows: there is nothing to compare them against, because the second
    species does not exist in the archive. Side A is named only because a row is a pair; run them
    with `--sides B` and side A is never submitted. They live here rather than in a one-off script
    so they archive into `log/promotion/` like everything else and can be re-run by name.
    """
    return [("TURING", f"atlas/{n}", None, 0.0, "core", "core", what) for n, what in (
        ("turing_coral",      "the one-species baseline: chem width 2, F 0.046 / kk 0.062"),
        ("turing_two_scale",  "two species, SAME reaction, activator range 32x apart"),
        ("turing_two_regime", "two species, SAME diffusion, coral point vs self-replicating spots"),
        ("turing_two_both",   "two species differing in scale AND regime"),
        # ---- FLAT, which is what the minisite's Turing section actually is. Its three clips are
        # `dim: 2` and are written against operators that were never promoted (`aggregate_seed`
        # mode disc, `graph_diffuse`, `react`), so the section had no runnable core spec at all.
        # `turing2d_disc` rebuilds it from promoted parts: `spawn: sunflower` for the placement,
        # `radius_graph` for the adjacency, `cell_chem_seed` for the chemistry.
        ("turing2d_disc",      "flat disc, 4000 cells, chi 1.3"),
        ("turing2d_disc_chi4", "the same disc at chi 4.0 -- the right value on THIS graph is not "
                               "yet measured, so both are run rather than one guessed"),
        ("turing2d_two",       "two species on the flat disc: the 2D twin of turing_two_both"),
        # ---- the two new reaction MODELS. Both need a span that is not a pair, which is why the
        # `chan`-must-be-even rule had to become "a multiple of this model's own width" first.
        ("turing2d_rps",       "May-Leonard cyclic competition: THREE species, travelling domains"),
        ("turing2d_coupled",   "FOUR species: two Gray-Scott systems suppressing each other. "
                               "At gamma 0 it reduces to turing2d_two bit for bit."),
        # ---- THE PARAMETER SWEEPS. Six two-species points from Pearson's own (F, k) diagram,
        # which transfers because d_h/d_a = 2 is his Du/Dv; three three-species points sweeping
        # the cyclic suppression. Each is DERIVED from the control that already patterns, so one
        # parameter differs and the motif is attributable to it.
        ("turing2d_gs_delta",  "GS Pearson delta: self-replicating spots"),
        ("turing2d_gs_beta",   "GS Pearson beta: spatiotemporal chaos, worms that never settle"),
        ("turing2d_gs_eta",    "GS Pearson eta: isolated stationary spots"),
        ("turing2d_gs_kappa",  "GS Pearson kappa: holes rather than spots"),
        ("turing2d_gs_lambda", "GS Pearson lambda: stationary labyrinth"),
        ("turing2d_gs_theta",  "GS Pearson theta: worms and loops with spots"),
        ("turing2d_rps_a0p2",  "RPS a=0.2: weak dominance, broad slow domains"),
        ("turing2d_rps_a1p2",  "RPS a=1.2: strong dominance, tight spirals"),
        ("turing2d_rps_a2p0",  "RPS a=2.0: invasion fronts outrun the spiral cores"),
        # ---- the 3D twin: the same May-Leonard working point on a CLOSED surface, where a wave
        # has no boundary to die against and a spiral core has no edge to anchor to.
        ("turing_rps",         "May-Leonard on the closed vertex mesh: the 3D twin of turing2d_rps"),
        # ---- the same six Pearson points WITH the cross term, so gs_X / two_X /
        # coupled_X is one reaction point at three degrees of company: alone,
        # sharing the cells, and chemically suppressed by a neighbour.
        ("turing2d_gs_coupled_delta",  "coupled twin of turing2d_two_delta: same B point, cross term on"),
        ("turing2d_gs_coupled_beta",   "coupled twin of turing2d_two_beta: same B point, cross term on"),
        ("turing2d_gs_coupled_eta",    "coupled twin of turing2d_two_eta: same B point, cross term on"),
        ("turing2d_gs_coupled_kappa",  "coupled twin of turing2d_two_kappa: same B point, cross term on"),
        ("turing2d_gs_coupled_lambda", "coupled twin of turing2d_two_lambda: same B point, cross term on"),
        ("turing2d_gs_coupled_theta",  "coupled twin of turing2d_two_theta: same B point, cross term on"),
        ("turing_rps_growth",  "r013_05 x turing_rps: the RED species gates cell_grow, so the "
                               "growth front travels with the rotating wave"),
        ("turing_eta_growth",  "r013_05 x turing2d_gs_eta: growth switched on inside isolated "
                               "STATIONARY spots rather than along a maze"),
        ("turing_rps_growth_30k", "the same, twice as long, growth stopped at 30k cells so the "
                                  "wave flows over a frozen population"),
        # ---- THE CUTAWAY SERIES (an EPITHELIUM seen from inside; `bulk_*` is the filled spheroid). One control and four single-variable additions, all rendered with
        # a quadrant cut away so the interior is visible. The two uncoupled chemistry members are
        # the controls for the two coupled ones.
        ("cutaway_growth",          "control: growth and division, no chemistry"),
        ("cutaway_growth_turing",   "+ Gray-Scott, running but NOT gating growth"),
        ("cutaway_growth_rps",      "+ May-Leonard, running but NOT gating growth"),
        ("cutaway_growth_turing_g", "+ Gray-Scott, activator GATES growth"),
        ("cutaway_growth_rps_gd",   "+ May-Leonard, RED gates growth and BLUE gates death"),
        # ---- the TWO-SYSTEM twin of the Pearson sweep: system A fixed, system B walking the
        # diagram, so each motif is asked whether it survives being laid over another on the same
        # cells. Independent spans, no cross term -- what differs from the solo run is competition
        # for space, not chemistry.
        ("turing2d_two_delta",   "two systems, B at Pearson delta: the twin of turing2d_gs_delta"),
        ("turing2d_two_beta",    "two systems, B at Pearson beta: the twin of turing2d_gs_beta"),
        ("turing2d_two_eta",     "two systems, B at Pearson eta: the twin of turing2d_gs_eta"),
        ("turing2d_two_kappa",   "two systems, B at Pearson kappa: the twin of turing2d_gs_kappa"),
        ("turing2d_two_lambda",  "two systems, B at Pearson lambda: the twin of turing2d_gs_lambda"),
        ("turing2d_two_theta",   "two systems, B at Pearson theta: the twin of turing2d_gs_theta"),)]


SUITES = {"ECM": suite_ecm, "BASE": suite_base, "MINISITE": suite_minisite,
          "TURING": suite_turing}


def _spec_src(spec):
    """The source yaml for a PAIRS row, and the stem its run names are built from.

    `folder/name` reads `config/<folder>/<name>.yaml`, which is how a row names a spec outside the
    okuda corpus -- the four lifted gates live in `config/gates/`. A bare name is an okuda spec,
    which is what every row was until the gates arrived, so nothing existing changes.
    """
    if spec.startswith("ecm/"):
        n = spec.split("/", 1)[1]
        return os.path.join(ROOT, "log", "okuda_ECM", n, "spec.yaml"), n
    if spec.startswith("base/"):
        n = spec.split("/", 1)[1]
        return os.path.join(ROOT, "log", "okuda", n, "spec_run.yaml"), n
    if "/" in spec:
        folder, name = spec.split("/", 1)
        return os.path.join(ROOT, "config", folder, f"{name}.yaml"), name
    flat = os.path.join(CFG_OKUDA, f"{spec}.yaml")
    if os.path.exists(flat):
        return flat, spec
    # THE CORPUS HAS SUBFOLDERS, and three front-page scenes live in them: `cellfix_B_new` and
    # `r005_06` under `_superseded_pre_basis/`, `r021_06` under `_superseded_r001-r029/`. A bare
    # name is searched one level down rather than hard-coded here, because hard-coding a path is
    # how a row silently starts running a different file when the corpus is reorganised.
    #
    # IT REFUSES TO GUESS. Two files of one name are two DIFFERENT MODELS, not two copies: gate 00's
    # header records exactly this trap -- `cellfix_B_new` has a second candidate parent that says
    # `rate: 0.03` against the real one's 0.003457 and reaches 1451 cells by frame 60 against 227.
    # Picking the alphabetically-first would have gated a correct run against the wrong tissue. So
    # an ambiguous name raises and the caller names the folder explicitly with `folder/name`.
    hits = sorted(glob.glob(os.path.join(CFG_OKUDA, "*", f"{spec}.yaml")))
    if len(hits) == 1:
        return hits[0], spec
    if len(hits) > 1:
        rel = [os.path.relpath(h, ROOT) for h in hits]
        raise SystemExit(
            f"  spec {spec!r} is AMBIGUOUS -- {len(hits)} files carry that name and they are not\n"
            f"  copies of one model:\n    " + "\n    ".join(rel) +
            f"\n  Name the folder explicitly in the row, e.g. "
            f"{os.path.basename(os.path.dirname(hits[0]))}/{spec}.")
    return flat, spec


def _stag(spec):
    """A spec as a filesystem-safe stem: `gates/gate_00_spheroid` -> `gates_gate_00_spheroid`."""
    return spec.replace("/", "_")


def _abspath_operator_files(cfg):
    """Rewrite an operator's file-valued parameters to absolute paths.

    A GATE SPEC WRITES THEM RELATIVE (`log/gates/_tissue/gate_04_tissue.npz`) so the file reads the
    same on any checkout, and the core runner resolves them from the repo root. The okuda side runs
    with `cd discovery_okuda`, so the same relative path points one directory too deep and
    `mesh_contact` dies on a missing file -- AFTER the scheduler has granted a GPU.

    AND THE ABSOLUTE PATH MUST BE THE CLUSTER'S, NOT THIS CONTAINER'S. The first version wrote
    `/workspace/Plexus/log/gates/_tissue/gate_04_tissue.npz`, which does not exist on a compute node:
    the same tree is `/groups/saalfeld/home/allierc/Graph/Plexus/...` there. `cluster.cpath` is the
    one translation, and it is the same one every `bsub` line already goes through -- so the two
    were disagreeing about where the repository is, which is a thing that can only fail on the
    cluster and always after the GPU has been granted.
    """
    import cluster as C
    for o in cfg.get("operators", []) or []:
        for k in ("tissue", "load", "gate_npz", "ckpt", "map", "surface"):
            v = o.get(k)
            if isinstance(v, str) and v and not os.path.isabs(v):
                cand = os.path.join(ROOT, v)
                if os.path.exists(cand):
                    o[k] = C.cpath(cand)
    return cfg


def _spec_copy(spec, run_name, frames, cfg_dir=None):
    """`config/okuda/<run_name>.yaml`: the pair's spec at the comparison length, under its own name.

    A DISTINCT NAME PER SIDE, because okuda writes to `log/okuda/<name>/` and two sides of one
    comparison would otherwise write into the same directory and the second would overwrite the
    first -- a comparison that always passes. `general.name` is a label, not an input: the RNG is
    seeded from `general.seed` (`engine.build`: `torch.Generator(...).manual_seed(sim.seed)`), so
    renaming cannot move a byte. The first row of the table is the check on that claim.
    """
    cfg_dir = cfg_dir or CFG_OKUDA
    src, _stem = _spec_src(spec)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} -- this row's spec is missing")
    cfg = _abspath_operator_files(yaml.safe_load(open(src)))
    # THE SEED OPERATORS GET THEIR ORIGINAL SPELLING, on both sides. `config/okuda/*.yaml` carries
    # `seed_mesh` / `seed_cell_chem` / `seed_ecm` since the seed refactor; a PRE-PROMOTION worktree
    # registers only `mesh_seed` / `cell_chem_seed` / `ecm_seed` and dies at load with
    # `operator 'seed_mesh' not in registry`. The current tree accepts both -- the new name is an
    # alias for the same class -- so writing the old one gives both sides the identical operator and
    # costs the current side nothing. Without this, every comparison against a pristine baseline
    # fails for a reason that has nothing to do with what is being compared.
    _BACK = {"seed_mesh": "mesh_seed", "seed_cell_chem": "cell_chem_seed", "seed_ecm": "ecm_seed"}
    for _o in cfg.get("operators", []) or []:
        if _o.get("op") in _BACK:
            _o["op"] = _BACK[_o["op"]]
    cfg["schedule"] = [_BACK.get(x, x) if isinstance(x, str) else x
                       for x in (cfg.get("schedule") or [])]
    # `_gate:` IS THE GATE'S THRESHOLD TABLE, not part of the model. It is dropped from the copy so
    # the twin run compares the SPEC and nothing else -- and so a grader cannot mistake a promotion
    # run for a graded gate run.
    cfg.pop("_gate", None)
    _check_record_clocks(cfg, spec)
    cfg["general"] = dict(cfg.get("general") or {})
    cfg["general"]["name"] = run_name
    if frames is not None:                      # None = the spec's own length (the scene rows)
        cfg["general"]["n_frames"] = int(frames)
        cfg["general"]["record_cap"] = int(frames) + 2
    os.makedirs(cfg_dir, exist_ok=True)
    dst = os.path.join(cfg_dir, f"{run_name}.yaml")
    yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
    return dst


# WHICH OKUDA MODULE REGISTERS AN OPERATOR `run_one.py` HAS NEVER IMPORTED.
#
# `run_one.py` imports a HARDCODED list -- `mesh_ops, chem_ops, t1_ops, monolayer_ops, ckpt,
# shape_chem_ops, shape_probe_ops` -- and that list has never contained `junction_ops`, at the
# pristine baseline or at HEAD. In okuda the junction operators are registered in exactly one place:
# `ops/tissue.py:221` does `import junction_ops` at the moment it INSERTS `junction_myosin` into a
# spec it is building. So okuda's junction specs have only ever run through `tissue.build()`, never
# through `run_one.py` -- which is why the archived gate-00 rig LOADS a cached tissue instead of
# simulating one, and why gate 00's okuda side died with `operator 'junction_myosin' not in
# registry` the first time it was asked to run for real.
#
# The registration is the same class under the same name either way, so importing the module is not
# a behaviour change; it is the same line `tissue.py` runs. It is done from a launcher in the PAIR
# DIRECTORY rather than by editing `run_one.py`, because side A is a pristine worktree and the whole
# value of it is that nothing edits it.
#
# ONLY THE MODULES A SPEC ACTUALLY NEEDS ARE IMPORTED. An unconditional import would put extra
# registrations into every run in the suite, including the phase-0 rows that are already green, and
# "it should not matter" is the reasoning that produced the last defect.
_OKUDA_OP_MODULE = {
    "junction_myosin": "junction_ops", "junction_sync": "junction_ops",
    "medioapical_myosin": "medioapical_ops", "cytokinetic_ring": "medioapical_ops",
}


def _okuda_entry(out_dir, cfg_dir, run_name):
    """The okuda side's entry point: `run_one.py`, or a launcher that registers what it omits.

    Returns a path to run instead of `run_one.py`. When the spec needs nothing extra the answer IS
    `run_one.py` and no launcher is written, so the overwhelming majority of rows keep running the
    exact command they always ran.
    """
    try:
        cfg = yaml.safe_load(open(os.path.join(cfg_dir, f"{run_name}.yaml")))
    except Exception:
        return "run_one.py"
    names = {o.get("op") for o in (cfg.get("operators") or []) if isinstance(o, dict)}
    need = sorted({_OKUDA_OP_MODULE[n] for n in names if n in _OKUDA_OP_MODULE})
    if not need:
        return "run_one.py"
    path = os.path.join(out_dir, f"_launch_{run_name}.py")
    with open(path, "w") as f:
        f.write('"""Register the okuda op modules `run_one.py` does not import, then run it.\n\n'
                'Written by tools/promotion_identical.py. `run_one.py` is executed under\n'
                '`__name__ == "__main__"` with sys.argv untouched, so it parses its own arguments\n'
                'exactly as it does when invoked directly.\n"""\n'
                "import runpy\n\n"
                f"for _m in {need!r}:\n"
                "    try:\n"
                "        __import__(_m)\n"
                "    except Exception as _e:\n"
                # LOUD, because a launcher that silently failed to register would hand the run
                # straight to `operator '...' not in registry` several seconds later, and the
                # traceback would name the spec rather than the import that was supposed to fix it.
                "        raise SystemExit(f'[launch] {_m} did not import, so the operators it \\\n"
                "registers are absent and this run cannot be a twin of anything: {_e!r}')\n\n"
                'runpy.run_path("run_one.py", run_name="__main__")\n')
    return path


def _check_record_clocks(cfg, spec):
    """Refuse a spec whose TOPOLOGY and POSITIONS record on different clocks.

    `run_one.py`'s D3 check already catches this -- and catches it AFTER the run, which is the
    problem. `atlas/turing_coral` burned a 6,000-frame job to reach

        [D3] recording misalignment: positions=601 frames but topology=6001. do() silently clamped
        this with hist[min(t, len(hist)-1)], which pairs each frame's coordinates with ANOTHER
        frame's connectivity and fabricates inverted cells.

    and `atlas/turing_grow_divide` burned an 1,800-frame one for the same reason. The engine derives
    the position stride from `record_cap` (`ceil(n_frames / (record_cap - 1))`) while `topo_record`
    carries its own `every`, so the two are set in different places by different people and nothing
    compared them until the run was over. Two seconds here saves the job.

    IT IS A WARNING, NOT A REFUSAL, because the core side tolerates the mismatch -- it records the
    mesh through the engine's recorder rather than through `topo_record`'s `hist` -- so a core-only
    row is legitimately unaffected, and turning this into an exception would block rows that work.
    """
    g = cfg.get("general") or {}
    nf, cap = g.get("n_frames"), g.get("record_cap")
    if not nf or not cap or cap > nf + 1:                 # cap above the frame count => stride 1
        return
    stride = -(-int(nf) // max(1, int(cap) - 1))          # what engine._setup_recording will pick
    for o in cfg.get("operators") or []:
        if isinstance(o, dict) and o.get("op") == "topo_record":
            every = int(o.get("every", 1) or 1)
            if every != stride:
                print(f"  [{spec}] WARNING: topo_record every={every} but the engine will record "
                      f"positions every {stride} ticks ({nf} frames, record_cap {cap}). The okuda "
                      f"side will record {nf // every + 1} topology rows against {nf // stride + 1} "
                      f"position rows and die on the D3 alignment check. Set `every: {stride}`, or "
                      f"raise record_cap above {nf + 1} to put both clocks on 1.", flush=True)


def _bsub_lines(out_dir, spec, side, run_name, frames):
    """The bsub command for one side, writing EVERYTHING into that side's own directory.

    `out_dir` is `log/promotion/<phase>_<spec>_<A|B>` -- the launcher, the job script, its stdout and
    stderr, and (for the core side) `--output_root`. There is no pair directory above it: a twin is
    two sibling folders whose names differ only in the last character, and the comparison reads one
    against the other."""
    import cluster as C
    home, _cfg, _log = _side_paths(side)
    if side.startswith("okuda"):
        entry = _okuda_entry(out_dir, _cfg, run_name)
        script = os.path.join(out_dir, f"{run_name}.sh")
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
                # `cpath` TRANSLATES A DEVCONTAINER PATH TO ITS CLUSTER PATH -- it is not a
                # general "make absolute". `_okuda_entry` returns the BARE NAME `run_one.py` when
                # the spec needs no launcher, and the script has already `cd`-ed into the okuda
                # directory, so that name must stay relative; passing it through `cpath` rewrote it
                # to `<repo>/run_one.py`, which does not exist, and seventeen of eighteen A sides
                # died in five seconds with `can't open file`. The one that worked was the one that
                # NEEDED a launcher -- an absolute path, where `cpath` is right. The fix took out
                # every row it was not needed for.
                f"conda run -n {C.ENV} python "
                f"{C.cpath(entry) if os.path.isabs(entry) else entry} {run_name}"
                + (f" --frames {frames}" if frames is not None else "")
                + " --device cuda:0 --campaign promotion",
            ]) + "\n")
    else:
        script = os.path.join(out_dir, f"{run_name}.sh")
        with open(script, "w") as f:
            f.write("\n".join([
                "#!/bin/bash -l",
                f"cd {C.cpath(home)}",
                f"export PYTHONPATH={C.cpath(os.path.join(home, 'src'))}",
                "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
                "export MPLBACKEND=Agg",
                "export PLEXUS_STRICT_DETERMINISM=1",
                f"conda run -n {C.ENV} python Plexus_Main.py -o generate promotion/{run_name} "
                f"--output_root {C.cpath(out_dir)} --device cuda:0 --force",
            ]) + "\n")
    os.chmod(script, 0o755)
    out = C.cpath(os.path.join(out_dir, f"{run_name}.out"))
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
        vp = z["vertex__pos"] if "vertex__pos" in z.files else None
        ch = z["cell__chem"] if "cell__chem" in z.files else None
        # THE CROP IS THE MESH'S `nF`/`Nv`, NOT `occ.sum()`, and the difference is not pedantic.
        # okuda's own frame builder slices `posf[t][:mt["Nv"]]` and `chemf[t][:mt["nF"], 0]`, so a
        # comparison that crops by occupancy is comparing a different number of cells and reports a
        # DIFFER on two runs that agree. It did: `r023_07` came back
        # "act_20: shape (2529,) vs (2530,)" while the two sides' `nF` was 2529 at every single
        # recorded tick.
        #
        # THE ONE-CELL GAP IS REAL, THOUGH, and is now a gate row rather than a mystery: on the core
        # side `cell__occ.sum()` leads `nF` by exactly one on 23 of 1801 rows, always at a death, and
        # never by more. `vertex__occ.sum()` matches `Nv` on all 1801. So the cell set's occupancy
        # and the mesh's face count are updated in a different order on the tick a cell is extruded.
        mnf = z["vertex__mesh_nF"] if "vertex__mesh_nF" in z.files else None
        mnv = z["vertex__mesh_Nv"] if "vertex__mesh_Nv" in z.files else None
        if vp is not None:
            rows = ticks if ticks is not None else range(vp.shape[0])
            for i, t in enumerate(rows):
                nv = int(mnv[t]) if mnv is not None else vp.shape[1]
                out.append((f"pos_{i}", np.asarray(vp[t][:nv], np.float32)))
                if ch is not None:
                    nf = int(mnf[t]) if mnf is not None else ch.shape[1]
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


def _pair_tag(phase, spec):
    """The pair's NAME, which is a prefix and not a directory: `G_gates_gate_01_nomyosin`."""
    return f"{_ptag(phase)}_{_stag(spec)}"


def _side_dir(phase, spec, tag):
    """A side's directory: `log/promotion/<phase>_<spec>_<A|B>`.

    FLAT, AND THAT IS THE POINT. The old layout put a pair directory above the two sides and then
    kept each run TWICE inside it -- once where the runner natively wrote it
    (`<pair>/graphs_data/promotion/<name>/` for the core, the worktree's `log/okuda/<name>/` for
    okuda) and once in the `A/`/`B/` mirror the comparison read. Measured on one pair, that was the
    same 7,507,620,532-byte `trajectory.npz` at two different inodes. Worse than the disk: after a
    killed run the mirror held one attempt and the native tree another, with nothing in the names to
    say which was which. One directory per side, and the run writes into it."""
    return os.path.join(OUT, f"{_pair_tag(phase, spec)}_{tag}")


# ---------------------------------------------------------------------------------- the run
def _ptag(phase):
    """A phase as a filesystem-safe prefix: `0` -> "0", `0.5` -> "0p5", `B-core` -> "B_core"."""
    return str(phase).replace(".", "p").replace("-", "_")


def run_pair(phase, spec, side_a, side_b, what, frames, submit=True, sides=None):
    """THE PHASE IS PART OF THE NAME, and it has to be. `--phase R` runs `b_star` with BOTH sides on
    the working tree, while `--phase B` runs the same spec with side A on a worktree: the two would
    write into `log/okuda/promo_b_star_B` and into `log/promotion/b_star/` together, and the second
    to finish would silently be compared against the first's leftovers. A gate that can overwrite
    its own reference is not a gate."""
    import cluster as C
    dirs = {t: _side_dir(phase, spec, t) for t in ("A", "B")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    names = {}
    for tag, side in (("A", side_a), ("B", side_b)):
        run_name = f"promo_{_pair_tag(phase, spec)}_{tag}"
        names[tag] = (side, run_name)
        if submit:
            # THE CORE SIDE NEEDS A SPEC TOO, and in its own folder: `Plexus_Main.py -o generate
            # promotion/<name>` resolves to `config/promotion/<name>.yaml`. Writing only the okuda
            # side's copy is how side A once died on a missing file while side B passed.
            _spec_copy(spec, run_name, frames, cfg_dir=(
                _side_paths(side)[1] if side.startswith("okuda")
                else os.path.join(_side_root(side), "config", "promotion")))
    if submit:
        # NORMALLY BOTH SIDES, TOGETHER -- that is the protocol and `sides` defaults to both.
        # `--sides A` exists for ONE situation: a bug that killed one side of a phase while the
        # other side's runs are still legitimately in flight. Resubmitting the pair would collide
        # with them in the same output directory and throw away hours of correct compute, and
        # killing them to keep the submission tidy is a worse trade. The runs are still both fresh
        # -- what the protocol forbids is reusing a STALE ARCHIVE, not a sibling submitted an hour
        # ago from the same code.
        want = [t for t in ("A", "B") if t in (sides or "AB")]
        lines = [_bsub_lines(dirs[t], spec, names[t][0], names[t][1], frames) for t in want]
        # THE SUBMITTER IS ONE SCRIPT FOR BOTH SIDES, so it cannot live in either side's directory
        # without implying it belongs to that side. It goes beside them, named for the pair.
        runner = os.path.join(OUT, f"_submit_{_pair_tag(phase, spec)}"
                                   f"{'' if len(want) == 2 else '_' + ''.join(want)}.sh")
        with open(runner, "w") as f:
            f.write("#!/bin/bash -l\n" + "\n".join(lines) + "\n")
        os.chmod(runner, 0o755)
        C._ssh(f"nohup bash {C.cpath(runner)} > {C.cpath(runner)}.log 2>&1 < /dev/null &",
               timeout=30)
        if len(want) == 2:
            print(f"  [{spec}] submitted BOTH sides together: "
                  f"{names['A'][1]} ({side_a}) | {names['B'][1]} ({side_b})", flush=True)
        else:
            t = want[0]
            print(f"  [{spec}] submitted SIDE {t} ONLY: {names[t][1]} ({names[t][0]})", flush=True)
    return names, dirs


def _landed(side, run_name, side_dir):
    """Has this side written the file the comparison reads -- natively, or after promotion?"""
    return os.path.exists(_out_path(side, run_name, side_dir)) or os.path.exists(
        os.path.join(side_dir, "traj.npz" if side.startswith("okuda") else "trajectory.npz"))


def _out_path(side, run_name, side_dir):
    """Where the RUNNER writes, which is not yet where the comparison reads.

    Neither runner can be told to write straight into `side_dir`: okuda's `run_one.py` writes under
    its own tree's `log/okuda/<name>/`, and `Plexus_Main.py --output_root X` always lands in
    `X/graphs_data/<campaign>/<name>/`. `promote_side` moves the contents up afterwards and deletes
    the shell, so the duplicate exists only while the job is in flight."""
    if side.startswith("okuda"):
        return os.path.join(_side_paths(side)[2], run_name, "traj.npz")
    return os.path.join(side_dir, "graphs_data", "promotion", run_name, "trajectory.npz")


# The two names a runner gives its trajectory, and the zarr beside it. These are the files worth
# GB, and the only ones `_collect_live` refuses to copy: a live mirror exists so a human can watch a
# run, and nobody watches a 7.5 GB npz. `promote_side` moves them once, at the end.
_HEAVY = ("trajectory.npz", "traj.npz", "simulation.zarr")


def _collect_live(names, dirs):
    """Show what has landed, WITHOUT copying the heavy files.

    The previous version copied the whole native tree on every poll, so a finished side existed
    twice for the rest of the run -- and if the wrapper was then killed, the two copies were a
    finished attempt and a live one with nothing to tell them apart. Here the poll copies only the
    small artefacts (stills, movies, json), and `promote_side` MOVES the rest when the side is done.
    """
    for tag, (side, run_name) in names.items():
        d = dirs[tag]
        if not _landed(side, run_name, d):
            continue
        src = os.path.dirname(_out_path(side, run_name, d))
        if not os.path.isdir(src) or os.path.abspath(src) == os.path.abspath(d):
            continue
        try:
            for f in os.listdir(src):
                if f in _HEAVY:
                    continue
                sp, dp = os.path.join(src, f), os.path.join(d, f)
                if os.path.isdir(sp):
                    shutil.copytree(sp, dp, dirs_exist_ok=True)
                elif not os.path.exists(dp) or os.path.getmtime(sp) > os.path.getmtime(dp):
                    shutil.copy2(sp, dp)
        except Exception as e:                    # a file being written under us is not a failure
            print(f"    [{os.path.basename(d)}] partial mirror ({type(e).__name__})")


def promote_side(side, run_name, side_dir):
    """MOVE the runner's native output up into `side_dir`, then delete the shell it wrote into.

    A move, not a copy, and on purpose: source and destination are on the same filesystem, so this
    is a rename and the trajectory is stored ONCE. The old `collect` copied, which is how one pair
    came to hold the same 7.5 GB npz at two inodes."""
    src = os.path.dirname(_out_path(side, run_name, side_dir))
    if not os.path.isdir(src) or os.path.abspath(src) == os.path.abspath(side_dir):
        return
    for f in os.listdir(src):
        sp, dp = os.path.join(src, f), os.path.join(side_dir, f)
        if os.path.exists(dp):
            shutil.rmtree(dp, ignore_errors=True) if os.path.isdir(dp) else os.remove(dp)
        shutil.move(sp, dp)
    shutil.rmtree(src, ignore_errors=True)
    # and the empty scaffolding the runner insisted on: `<side>/graphs_data/promotion/`
    for up in (os.path.dirname(src), os.path.dirname(os.path.dirname(src))):
        if os.path.abspath(up).startswith(os.path.abspath(side_dir)) and os.path.isdir(up) \
                and not os.listdir(up):
            os.rmdir(up)


def collect(spec, names, dirs):
    """Move each side's output into its own directory. Returns {tag: dir}, which is what compares."""
    for tag, (side, run_name) in names.items():
        promote_side(side, run_name, dirs[tag])
    return dict(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None, help="only the pairs of one phase (0, 0.5, B, C, D)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--frames", type=int, default=None,
                help="override every row's length; default = the row's own")
    ap.add_argument("--compare-only", action="store_true",
                    help="do not submit; compare what is already in log/promotion/")
    ap.add_argument("--wait-min", type=float, default=90.0)
    ap.add_argument("--batch", type=int, default=0,
                    help="submit at most this many PAIRS at once (2 jobs each) and wait for a "
                         "slot before the next; 0 = all at once")
    ap.add_argument("--n-base", type=int, default=20, help="how many rows `--phase BASE` picks")
    ap.add_argument("--sides", default=None,
                    help="submit only side A or only side B (default: both, which is the protocol). "
                         "For repairing one dead side while the other is still in flight.")
    ap.add_argument("--skip", default=None,
                    help="drop rows whose spec name contains any of these comma-separated "
                         "substrings (the inverse of --only)")
    ap.add_argument("--only", default=None,
                    help="keep only rows whose spec name contains this substring, so one row can be "
                         "re-run without re-running its whole phase")
    ap.add_argument("--no-compare-render", action="store_true",
                    help="skip the side-by-side compare.png / compare.mp4")
    ap.add_argument("--tol", type=float, default=None,
                    help="override every row's absolute position tolerance "
                         "(0 = byte-identical, which is every row's default)")
    a = ap.parse_args()

    if a.phase in SUITES:
        pairs = SUITES[a.phase]() if a.phase != "BASE" else SUITES[a.phase](a.n_base)
        print(f"  suite {a.phase}: {len(pairs)} pair(s) generated from the archive")
    else:
        pairs = [p for p in PAIRS if a.all or (a.phase is not None and str(p[0]) == str(a.phase))]
    # ONE ROW, BY NAME. A phase is the unit the promotion advances in, but it is not the unit
    # DEBUGGING works in: re-running six gate rows to look at one of them costs twelve jobs on a
    # shared queue and buries the row that was asked about. `--only` filters whatever `--phase` or
    # `--all` selected, by substring against the spec name, so the row keeps its own tolerance,
    # frame count and pinned side A rather than being retyped as a one-off.
    if a.only:
        pairs = [p for p in pairs if a.only in str(p[1])]
        print(f"  --only {a.only!r}: {len(pairs)} row(s) kept")
    # THE INVERSE, for the case `--only` cannot express: re-verifying a whole phase while a couple
    # of its rows are already in flight. Resubmitting those would collide in the same output
    # directory and discard the runs under way, and the rows left over share no substring.
    if a.skip:
        _drop = [x.strip() for x in a.skip.split(",") if x.strip()]
        before = len(pairs)
        pairs = [p for p in pairs if not any(d in str(p[1]) for d in _drop)]
        print(f"  --skip {a.skip!r}: {before - len(pairs)} row(s) dropped, {len(pairs)} kept")
    if not pairs:
        print("  no pair selected -- use --phase or --all"); return 2

    os.makedirs(OUT, exist_ok=True)
    # THE RUN LOG GOES IN THE ARCHIVE. Cedric: "you must archive in log/promotion". The harness's own
    # stdout was going to /tmp, so the one record of WHICH rows were submitted, when, and what the
    # table said lived outside the directory that is supposed to hold the evidence -- and /tmp does
    # not survive a container restart.
    _tag = str(a.phase) if a.phase is not None else "all"
    _log = open(os.path.join(OUT, f"run_{_ptag(_tag)}.log"), "a")

    class _Tee:
        def write(self, x):
            sys.__stdout__.write(x); _log.write(x); _log.flush()

        def flush(self):
            sys.__stdout__.flush(); _log.flush()

    sys.stdout = _Tee()
    print(f"\n=== phase {_tag}: {len(pairs)} pair(s)", flush=True)

    def _running():
        """How many of THIS run's jobs are still in the queue."""
        import cluster as C
        st = C._ssh("bjobs -w 2>/dev/null | grep -c promo_ || true", timeout=30)
        try:
            return int((st.stdout or "0").strip().split()[0])
        except Exception:
            return 99                      # an unreachable login node is not an empty queue

    jobs = []
    for phase, spec, nfr, tol, sa, sb, what in pairs:
        # BATCHED, BECAUSE THE QUEUE IS NOT INFINITE. Cedric: "you can run batch of 8 on l4". Forty
        # pairs is eighty jobs; submitted at once they queue behind each other anyway and the first
        # failure is invisible until the last one lands. A slot is a PAIR (two jobs), so `--batch 8`
        # is sixteen jobs in flight -- and both sides of a pair still go in together, which is the
        # protocol: the two sides must not be separated by an hour of drift in the tree they read.
        if a.batch and not a.compare_only:
            while _running() >= 2 * a.batch:
                time.sleep(45)
            # SETTLE BEFORE THE NEXT POLL. `bsub` returns as soon as the scheduler accepts the job,
            # and `bjobs` does not list it for a few seconds -- so a tight loop reads a stale count
            # and submits past the cap. The first ECM run overshot to twelve pairs against a limit
            # of eight for exactly this reason. Two seconds is longer than the lag and shorter than
            # anything it delays.
            time.sleep(3)
        frames = a.frames if a.frames is not None else nfr
        try:
            names, pd = run_pair(phase, spec, sa, sb, what, frames,
                                 submit=not a.compare_only, sides=a.sides)
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
            # COUNT ONLY THIS RUN'S JOBS. `grep -c promo_` counts every promotion job in the
            # queue, so a `--phase 0` waiter whose twelve jobs had all finished sat blocked for
            # twenty minutes on two `--phase B-core` jobs it has nothing to do with. Two phases in
            # flight at once is the normal case, not the exception.
            mine = "|".join(rn for *_r, names, _pd in jobs for _side, rn in names.values())
            st = C._ssh(f"bjobs -w 2>/dev/null | grep -cE '{mine}' || true", timeout=30)
            queue_empty = st is not None and (st.stdout or "").strip().startswith("0")
            landed = all(_landed(side, rn, pd[t])
                         for *_r, names, pd in jobs for t, (side, rn) in names.items())
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
        # THE COMPARISON ITSELF, as one picture and one clip. Two directories each holding a movie
        # is not a comparison: it is two movies, and telling them apart means opening both and
        # holding one in your head. The digest says WHETHER they agree; this says where and how when
        # they do not, which is the only reason to keep the pixels.
        if not a.no_compare_render:
            try:
                sys.path.insert(0, os.path.join(ROOT, "src"))
                import plexus.render_vtk as _R
                if _R.available():
                    ttl = (f"{spec}   A={sa}  B={sb}   {da} vs {db}   "
                           + ("IDENTICAL" if ok else f"DIFFER: {rep['why'][:60]}"))
                    lab = (f"A  {sa}", f"B  {sb}")
                    # BESIDE the two sides, named for the pair -- a comparison belongs to neither
                    # of them, and a third DIRECTORY would put a folder back above the twins.
                    _cmp = os.path.join(OUT, _pair_tag(phase, spec))
                    _R.compare_still(dirs["A"], dirs["B"], _cmp + "_compare.png",
                                     labels=lab, title=ttl)
                    _R.compare(dirs["A"], dirs["B"], _cmp + "_compare.mp4",
                               labels=lab, title=ttl)
            except Exception as e:
                print(f"  [{spec}] comparison render skipped ({type(e).__name__}: {str(e)[:60]})")
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
