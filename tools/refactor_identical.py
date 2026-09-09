#!/usr/bin/env python
"""Byte-identity across a REFACTOR: the same spec, this tree against a git ref, every array.

WHY NOT `promotion_identical.py`. That tool answers a different question -- okuda against core, two
DIFFERENT writers -- so it digests the two arrays the two writers have in common, `vertex__pos` and
`cell__chem`, and crops them to the mesh's own `nF`/`Nv`. For a refactor both sides are this
codebase, so every recorded array is comparable, and the ones it omits are exactly the ones a
restructuring moves: `vertex__sep`, every `vertex__mesh_*` face column, every `cell__*` block. A
harness that compared only `pos` would pass a commit that silently dropped `phase`, which is a
defect this repo has already shipped once.

WHAT COUNTS AS IDENTICAL. Array by array, `a.tobytes() == b.tobytes()`, over every key present in
both trajectories, reduced to one sha1 per side so a run records as a single number. The floor is
zero and it is measured, not assumed: `promotion_identical`'s own header records byte-identity
holding over 1,800 frames with division, T1 and chemistry live, where one differing mantissa bit
would move an `edge_flip` decision and diverge from then on. `PLEXUS_STRICT_DETERMINISM=1` is set on
both sides for the same reason.

KEYS THAT APPEAR OR VANISH ARE REPORTED, NOT IGNORED, and that is the point of the tool. A rung that
moves `A0` from the mesh table to the cell set changes `vertex__mesh_A0` -> `cell__A0`; the run is
then NOT identical and must not be reported as such. The spec is listed under `--opt-in`, its key
change is printed in full, and every other spec must still be bit-equal.

BOTH SIDES ARE RUN FRESH, from a git worktree at `--ref` and from this tree, in the same
invocation. Nothing stored from a previous session is ever the reference -- the reason is
`promotion_identical`'s: an artefact proves only that the artefact has not changed.

    PYTHONPATH=src python tools/refactor_identical.py --ref 20eb3d06
    PYTHONPATH=src python tools/refactor_identical.py --ref 20eb3d06 --specs cycle_phases mech_hexprism
    PYTHONPATH=src python tools/refactor_identical.py --ref HEAD~1 --opt-in cycle_phases

Exit 0 identical (or differing only in declared opt-ins), 1 a real difference, 2 a run failed.
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# THREE SPECS IN `config/tissue` ARE NOT PART OF THIS CAMPAIGN AND COST MORE THAN ALL THE REST
# TOGETHER. `b_star`, `r010_00_ctrl` and `r020_00_ctrl` predate the apico-basal work, run 1,800
# frames of 2,000 cells each, and exercise no operator the other eighteen do not. Left in the
# default set they turn a per-rung check into an overnight one, and a check that is not run after
# every commit is not a check. Skipped by default and reachable with `--all`.
#
# It is a judgement about COST, not about coverage: the covering set is what makes a byte-identity
# claim mean something, and these three add nothing to it.
SLOW = ("b_star", "r010_00_ctrl", "r020_00_ctrl")
WORKTREES = os.path.join(ROOT, "log", "_worktrees")
PY = sys.executable


def _worktree(ref):
    """A git worktree at `ref`, so "before the change" is RUN rather than remembered.

    Reused verbatim from `promotion_identical._worktree`, including where it lives: `log/_worktrees`
    and not the output tree, because a worktree is CODE and an output root may be a different
    filesystem.
    """
    d = os.path.join(WORKTREES, ref.replace("/", "_"))
    if os.path.isdir(os.path.join(d, ".git")) or os.path.isfile(os.path.join(d, ".git")):
        return d
    os.makedirs(WORKTREES, exist_ok=True)
    subprocess.run(["git", "-C", ROOT, "worktree", "prune"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-C", ROOT, "worktree", "add", "--detach", d, ref], check=True)
    return d


def _spawn(root, group, name, out_root, device):
    """Start one spec from `root`, into `out_root`. Returns (Popen, trajectory_path, label).

    THE TWO SIDES RUN AT THE SAME TIME, ON THE TWO GPUs. They were sequential and that doubled the
    wall clock of every rung for nothing: the sides are independent by construction -- different
    trees, different output roots, different devices -- and the comparison happens after both are
    written. On a suite whose longest specs are 600 frames of 2,000 cells, that is the difference
    between a rung checked after every commit and a rung checked when someone remembers.

    `--output_root` must EXIST before the run: `Plexus_Main` asserts on it rather than creating it,
    which is the right call for a flag that decides where gigabytes land.
    """
    os.makedirs(out_root, exist_ok=True)
    env = dict(os.environ,
               PYTHONPATH=os.path.join(root, "src"),
               PLEXUS_STRICT_DETERMINISM="1",     # the floor is only zero if the run is pinned
               MPLBACKEND="Agg")
    pr = subprocess.Popen(
        [PY, "Plexus_Main.py", "-o", "generate", f"{group}/{name}",
         "--output_root", out_root, "--device", device, "--force", "--no-describe", "--no-viz"],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # `--output_root` IS THE ROOT, NOT THE DATA DIRECTORY. `graphs_data_path` appends
    # `graphs_data/<group>/<name>` under it, so the trajectory is one level deeper than the flag.
    return pr, os.path.join(out_root, "graphs_data", group, name, "trajectory.npz"), \
        ("this tree" if root == ROOT else "ref")


def _reap(pr, path, label):
    out, _ = pr.communicate()
    if pr.returncode != 0 or not os.path.exists(path):
        tail = (out or "").strip().splitlines()[-3:]
        print(f"    RUN FAILED ({label}): " + " | ".join(tail))
        return None
    return path


def _digest(path):
    """{key: sha1} for every array in a trajectory, plus one sha1 over all of them in key order."""
    z = np.load(path, allow_pickle=True)
    per, h = {}, hashlib.sha1()
    for k in sorted(z.files):
        a = np.asarray(z[k])
        d = hashlib.sha1(a.tobytes()).hexdigest()
        per[k] = (d, a.shape, str(a.dtype))
        h.update(k.encode()); h.update(d.encode())
    return per, h.hexdigest()


def _close(pa, pb, tol):
    """Compare two runs to a RELATIVE TOLERANCE instead of bit for bit.

    WHY A TOLERANCE MODE AT ALL. Byte-identity is the right gate while a rung claims to move
    nothing, and it carried S1, S2a, S2b and S2c-1. It stops being the right gate the moment a
    rung reassociates a floating-point sum, because this model is chaotic in `edge_flip`: a
    perturbation at the 24th bit of a float32 coordinate decides one reconnection differently and
    the trajectories part. Measured on `gate_00_spheroid`, 4.5e-06 at frame 33 becomes 6,914 cells
    against 6,749 by frame 401 -- a 2.4% divergence from a change that moved no physics.

    WHAT IS COMPARED, AND WHY NOT ELEMENTWISE. The ragged mesh arrays are concatenations whose
    length is the run's own history -- 204,162 entries against 202,658 -- so they cannot be lined
    up elementwise at all once the cell counts differ by one. What CAN be compared is the run's
    trajectory of summary quantities: the live cell count frame by frame, and the mean of every
    recorded per-face column over the live faces. Those are what a reader of the movie sees, and a
    refactor that leaves them within a tolerance has not changed the model even where it has
    changed the last bits.

    Returns (ok, "<worst array> <relative deviation>").
    """
    za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    worst, wname = 0.0, "-"

    def rel(u, v):
        d = float(np.max(np.abs(np.asarray(u, np.float64) - np.asarray(v, np.float64))))
        s = max(float(np.max(np.abs(np.asarray(u, np.float64)))), 1e-12)
        return d / s

    for k in sorted(set(za.files) & set(zb.files)):
        A, B = np.asarray(za[k]), np.asarray(zb[k])
        if k.endswith(("_offsets",)) or A.dtype.kind not in "fiu":
            continue
        if k.endswith("__mesh_nF") or k.endswith("__mesh_Nv"):
            r = rel(A, B) if A.shape == B.shape else 1.0
        elif "__mesh_" in k and A.ndim == 1 and A.shape != B.shape:
            # a ragged per-face column: compare the PER-FRAME MEAN, which is defined either way
            s = k.split("__mesh_")[0]
            fa, fb = np.asarray(za[f"{s}__mesh_face_offsets"]), np.asarray(zb[f"{s}__mesh_face_offsets"])
            na, nb = np.asarray(za[f"{s}__mesh_nF"]), np.asarray(zb[f"{s}__mesh_nF"])
            T = min(len(na), len(nb))
            ma = np.array([A[int(fa[t]):int(fa[t] + na[t])].mean() for t in range(T)])
            mb = np.array([B[int(fb[t]):int(fb[t] + nb[t])].mean() for t in range(T)])
            r = rel(ma, mb)
        elif A.shape != B.shape:
            r = 1.0
        else:
            r = rel(A, B)
        if r > worst:
            worst, wname = r, k
    return worst <= tol, f"{wname} {worst:.3%}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="git ref for the reference side")
    ap.add_argument("--group", default="tissue", help="config/<group>/ (default: tissue)")
    ap.add_argument("--specs", nargs="*", default=None, help="spec names; default every one in the group")
    ap.add_argument("--opt-in", nargs="*", default=[],
                    help="specs ALLOWED to differ, by name -- listed, printed, not counted as failures")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-ref", default="cuda:1", help="the reference side runs here, in parallel")
    ap.add_argument("--out", default="/tmp/plexus_refactor")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="accept a RELATIVE difference this large instead of bit-for-bit "
                         "(e.g. 0.01); compares per-frame summaries, see `_close`")
    ap.add_argument("--all", action="store_true",
                    help=f"include the slow pre-campaign specs ({', '.join(SLOW)})")
    a = ap.parse_args()

    wt = _worktree(a.ref)
    names = a.specs or sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(wt, "config", a.group, "*.yaml")))
    # ONLY SPECS THAT EXIST ON BOTH SIDES. A spec added by the branch has nothing to compare against
    # and a spec deleted by it cannot be run; either is a legitimate change and neither is evidence.
    here = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ROOT, "config", a.group, "*.yaml"))}
    skipped = [n for n in names if n not in here]
    names = [n for n in names if n in here]
    if not a.all and not a.specs:
        slow = [n for n in names if n in SLOW]
        names = [n for n in names if n not in SLOW]
        if slow:
            print(f"skipping {len(slow)} slow pre-campaign spec(s): {', '.join(slow)}  (--all to include)")

    print(f"reference {a.ref} in {wt}")
    print(f"{len(names)} spec(s) in config/{a.group}/"
          + (f"; {len(skipped)} not on both sides: {', '.join(skipped)}" if skipped else ""))
    if a.opt_in:
        print(f"opt-in (allowed to differ): {', '.join(a.opt_in)}")
    print()

    bad, optd, failed = [], [], []
    for n in names:
        ja = _spawn(wt, a.group, n, os.path.join(a.out, "ref"), a.device_ref)
        jb = _spawn(ROOT, a.group, n, os.path.join(a.out, "now"), a.device)
        pa, pb = _reap(*ja), _reap(*jb)
        if pa is None or pb is None:
            failed.append(n); print(f"  {n:<28} RUN FAILED"); continue
        if a.tol > 0:
            ok, worst = _close(pa, pb, a.tol)
            if ok:
                print(f"  {n:<28} within {a.tol:.1%}   worst {worst}")
                continue
            print(f"  {n:<28} EXCEEDS {a.tol:.1%}   worst {worst}")
            bad.append(n)
            continue
        da, ha = _digest(pa)
        db, hb = _digest(pb)
        only_a, only_b = sorted(set(da) - set(db)), sorted(set(db) - set(da))
        diff = [k for k in sorted(set(da) & set(db)) if da[k][0] != db[k][0]]
        if not diff and not only_a and not only_b:
            print(f"  {n:<28} identical   {ha[:12]}")
            continue
        tag = "OPT-IN" if n in a.opt_in else "DIFFER"
        (optd if n in a.opt_in else bad).append(n)
        print(f"  {n:<28} {tag}")
        for k in only_a[:6]:
            print(f"      removed  {k}  {da[k][1]} {da[k][2]}")
        for k in only_b[:6]:
            print(f"      added    {k}  {db[k][1]} {db[k][2]}")
        for k in diff[:6]:
            print(f"      differs  {k}  ref{da[k][1]} vs now{db[k][1]}")
        if len(diff) > 6:
            print(f"      ... and {len(diff) - 6} more arrays")

    print()
    print(f"{len(names) - len(bad) - len(optd) - len(failed)} identical, {len(optd)} opt-in, "
          f"{len(bad)} DIFFER, {len(failed)} failed")
    if bad:
        print("  differing: " + ", ".join(bad))
    return 2 if failed else (1 if bad else 0)


if __name__ == "__main__":
    sys.exit(main())
