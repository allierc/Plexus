#!/usr/bin/env python
"""Where a run's state LIVES and whether it adds up -- the check that replaces byte-identity.

WHY THIS EXISTS AND WHY IT REPLACES THE OTHER INSTRUMENT. `tools/refactor_identical.py` answers
"did anything move", which is the right question while a refactor claims to move nothing. It is the
wrong question the moment a rung has to CHANGE something -- and the alignment campaign reached that
point: `cell_divide` splits a mother's target volume between two daughters and copies other
quantities to both, and any rung that touches that code will legitimately move numbers. A gate that
refuses every such rung refuses the campaign.

So the gate becomes: after the run, is the state where the paper says it goes, is all of it
accounted for, and do the quantities that must add up add up. Four checks, all measured on a real
run rather than read off the source, because the defects this campaign has actually found -- a
daughter inheriting a stranger's phase, a cap area with a null space, a replay reading a literal
column list -- were none of them visible in the code.

    CENSUS      per-cell state still living on the mesh table instead of the cell set.
                plexus2 says a face of the mesh IS a cell and per-cell state belongs to the cell
                SET, where `Hierarchy.renumber_set` already carries it. Every name this reports is
                a column the topology operators have to know by name. It is the campaign's own
                progress bar and it should end at zero.

    DEAD        declared blocks that never move off their initial value for the whole run. The
                mirror image of the census: state the spec declares and no operator ever writes,
                recorded every frame as a column of zeros. Seven `config/tissue` specs declare
                `chem` with no chemistry operator anywhere in them.

    LENGTHS     every per-face array exactly `nF` long, every cell block's live rows exactly `nF`,
                occupancy summing to `nF`. This is the clamp defect's home: `reindex_faces` clamps
                an out-of-range index rather than raising, so a short array does not fail, it
                silently hands out the last cell's value.

    SUMS        the extensive per-cell targets across a division. `cell_divide` SPLITS `A0` and
                `V0f` between the two daughters, so the population total is continuous across a
                division; a copy would double the mother's contribution. Measured as the jump in
                the live-cell total at frames where `nF` grows, against the total itself.

    PYTHONPATH=src python tools/consistency.py --specs cycle_phases divide_growing_ball
    PYTHONPATH=src python tools/consistency.py --group tissue --frames 40
    PYTHONPATH=src python tools/consistency.py --traj log/.../trajectory.npz     # DEAD/LENGTHS/SUMS only

Exit 0 all clean, 1 a check failed, 2 a run failed.
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# NAMES ON THE MESH TABLE THAT ARE NOT PER-CELL STATE, so the census does not report them as debt.
# The first five ARE the mesh -- the half-edge table and its two counts -- and the rest are the
# operators' own cumulative counters, which are one number per run and not one per cell (`MeshTable`
# lists them as `SCALAR_RECORD` for the same reason). `face_carry` and `vertex_carry` are the name
# SETS the topology operators walk, not arrays.
MESH_OWN = {"E_srce", "E_trgt", "E_face", "nF", "Nv",
            "n_t1", "n_apop", "n_div", "div_blocked", "apop_spill", "renumber_failed",
            "mono_h", "mono_k", "mono_delta", "buf_full", "v_ref", "v_ref_poly", "R0",
            "face_carry", "vertex_carry", "mech", "centroid_np", "apop_marked_once"}


def _live(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x) if not isinstance(x, (set, dict, str, bool)) else None


def census(H, vset, cset):
    """[(name, len)] for every per-cell array still on the mesh table, plus the cell set's blocks.

    A "per-cell array" is one whose length is `nF`. That is the test, rather than a name list,
    because the point of the check is to catch a column NOBODY REMEMBERED TO LIST -- which is the
    exact defect `face_carry` was introduced for and the one the replay's literal four-column list
    reproduced one level up.
    """
    lvl = H.level(vset)
    m = getattr(lvl, "mesh", None) or getattr(lvl, "_mesh", None)
    if m is None:
        return [], []
    nF = int(m["nF"])
    on_mesh = []
    for k in sorted(getattr(m, "keys", lambda: [])()):
        if k in MESH_OWN:
            continue
        a = _live(m.get(k))
        if a is None or a.ndim == 0:
            continue
        if a.shape[0] == nF:
            on_mesh.append((k, int(a.shape[0])))
    clvl = H.level(cset) if cset and cset in getattr(H, "levels", {}) else None
    on_cell = [] if clvl is None else [(b.name, b.width, bool(b.record))
                                       for b in clvl.state_schema.blocks]
    return on_mesh, on_cell


def dead_blocks(z, sets):
    """Recorded blocks that never leave their initial value. `sets` is {set: [block, ...]}."""
    out = []
    for s, blocks in sets.items():
        for b in blocks:
            k = f"{s}__{b}"
            if k not in z.files:
                continue
            a = np.asarray(z[k])
            if a.ndim < 2 or a.shape[0] < 2:
                continue
            # ONLY THE LIVE ROWS, and that matters: a cell set is allocated to a capacity and its
            # tail is stale for the whole run, so a block that IS written looks half-dead if the
            # dead slots are counted. `occ` says which rows are there.
            occ = np.asarray(z[f"{s}__occ"]) if f"{s}__occ" in z.files else None
            n = int(occ[-1].sum()) if occ is not None else a.shape[1]
            n = max(1, min(n, a.shape[1]))
            v = a[:, :n]
            if np.all(v == v[0]):
                out.append((s, b, float(np.abs(v).max())))
    return out


def lengths(z, vset):
    """Frames where a per-face column is not exactly `nF` long, or occupancy disagrees with `nF`."""
    bad = []
    kf = f"{vset}__mesh_nF"
    if kf not in z.files:
        return bad
    nF = np.asarray(z[kf])
    fo = np.asarray(z[f"{vset}__mesh_face_offsets"])
    for k in z.files:
        p = f"{vset}__mesh_"
        if not k.startswith(p) or k.endswith(("offsets", "nF", "Nv")) or "scalar" in k:
            continue
        c = k[len(p):]
        if c in ("E_srce", "E_trgt", "E_face"):
            continue
        a = np.asarray(z[k])
        if a.ndim != 1 or a.shape[0] != int(fo[-1]):
            bad.append((c, f"total {a.shape} against face offsets ending {int(fo[-1])}"))
            continue
        for t in range(len(nF)):
            if int(fo[t + 1] - fo[t]) != int(nF[t]):
                bad.append((c, f"frame {t}: {int(fo[t+1]-fo[t])} entries for nF={int(nF[t])}"))
                break
    return bad


def sums(z, vset, name="V0f"):
    """The live-cell total of an EXTENSIVE per-cell target, frame by frame, across divisions.

    WHAT A FAILURE LOOKS LIKE. `cell_divide` splits the mother's `V0f` between the two daughters
    (`v0d = V0f[f] * p`, `v0e = V0f[f] * (1 - p)`), so a division adds a cell and adds NOTHING to
    the population total: the total moves only by what `cell_grow` put in. If a rung ever replaced
    that split with a copy -- the trap this whole campaign nearly walked into once already -- the
    total would jump by one mother's worth at every division, and on a run that goes 200 -> 6,000
    cells that is a target volume thirty times too large, reached smoothly enough to look like
    growth.

    TWO LEGITIMATE REGIMES, AND THE CHECK HAS TO TELL THEM APART FROM THE DEFECT. With
    `g1_ramp: false` the split is exact and the population total is conserved across a division to
    floating point. With `g1_ramp: true` -- "birth at target" -- the daughters' targets are set from
    their ACTUAL birth volumes (`v0d = half`, where `half = vf[f] * p`) rather than by dividing the
    mother's target, so the total legitimately steps by the mother's volume ERROR, `vf[f] - V0f[f]`,
    which is a fraction of one cell. A COPY is a different number entirely: it adds a whole mother's
    target per division, so the jump is about ONE cell-equivalent PER CELL BORN. That ratio is what
    is reported and thresholded, and it separates the three cases cleanly -- 0.00 exact split, ~0.27
    measured on `cycle_phases` with the ramp on, ~1.00 a copy.
    """
    kn, kv = f"{vset}__mesh_nF", f"{vset}__mesh_{name}"
    if kn not in z.files or kv not in z.files:
        return None
    nF = np.asarray(z[kn])
    fo = np.asarray(z[f"{vset}__mesh_face_offsets"])
    v = np.asarray(z[kv])
    tot = np.array([v[int(fo[t]):int(fo[t] + nF[t])].sum() for t in range(len(nF))], np.float64)
    per = np.array([tot[t] / max(int(nF[t]), 1) for t in range(len(nF))], np.float64)
    born = np.diff(nF.astype(np.int64))
    d = np.diff(tot)
    ev = np.where(born > 0)[0]
    if ev.size == 0:
        return dict(divisions=0, worst=0.0, frame=-1)
    # a division frame's jump, in units of ONE CELL's current target
    j = np.abs(d[ev]) / np.maximum(per[ev], 1e-12) / np.maximum(born[ev], 1)
    i = int(np.argmax(j))
    return dict(divisions=int(born[ev].sum()), worst=float(j[i]), frame=int(ev[i]),
                born_there=int(born[ev][i]), total0=float(tot[0]), total1=float(tot[-1]))


def one(group, name, frames, device, traj_dir=None):
    """(census rows, trajectory, declared blocks) for one spec.

    TWO SOURCES ON PURPOSE. The CENSUS needs the live `Hierarchy` -- most per-cell arrays are
    unrecorded (`Vbirth`, `divjit`, `alive`, `mg_scale`), so a trajectory cannot say where they
    live, and "where does state live" is the question this whole instrument exists for. The other
    three checks need the SAVED trajectory, because `engine.run`'s in-memory return is nested per
    set and only the writer flattens it to `set__block`; re-implementing that flattening here would
    be a second copy of a format, which is how a reader and a writer drift apart.

    So: a short fresh run for the census, and the run's own `trajectory.npz` for the rest. If there
    is no trajectory on disk the three are skipped and said to be skipped, never quietly passed.
    """
    from plexus import schema
    from plexus.engine import run as engine_run
    path = os.path.join(ROOT, "config", group, f"{name}.yaml")
    sim = schema.load(path)
    vsets = {s: d for s, d in (sim.sets or {}).items()
             if isinstance(d, dict) and d.get("mesh")}
    if frames:
        sim.n_frames = int(frames)
    H, _traj = engine_run(sim, out_path=None, device=device)
    rows = []
    for vs, d in vsets.items():
        cs = d.get("cell_set")
        on_mesh, on_cell = census(H, vs, cs)
        rows.append((vs, cs, on_mesh, on_cell))
    declared = {s: list((d.get("state") or {}).keys())
                for s, d in (sim.sets or {}).items() if isinstance(d, dict)}
    z = None
    for cand in ([traj_dir] if traj_dir else []) + [_data_dir(group, name)]:
        if cand and os.path.exists(os.path.join(cand, "trajectory.npz")):
            z = np.load(os.path.join(cand, "trajectory.npz"), allow_pickle=True)
            break
    return rows, z, declared


def _data_dir(group, name):
    try:
        from plexus.paths import graphs_data_path
        return os.path.join(graphs_data_path(), group, name)
    except Exception:                                            # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--specs", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=40,
                    help="override n_frames; 0 keeps the spec's own (default 40)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--traj-dir", default=None,
                    help="directory holding trajectory.npz; default the run's own output dir")
    ap.add_argument("--quiet-dead", action="store_true", help="do not fail on DEAD, only report")
    a = ap.parse_args()

    names = a.specs or sorted(os.path.basename(p)[:-5]
                              for p in glob.glob(os.path.join(ROOT, "config", a.group, "*.yaml")))
    bad = failed = 0
    tot_mesh = 0
    for n in names:
        print(f"\n=== {n}")
        try:
            rows, z, declared = one(a.group, n, a.frames, a.device, a.traj_dir)
        except Exception as e:                                   # noqa: BLE001
            print(f"    RUN FAILED: {type(e).__name__}: {e}")
            failed += 1
            continue
        for vs, cs, on_mesh, on_cell in rows:
            tot_mesh += len(on_mesh)
            print(f"  CENSUS  {vs} -> cell set {cs!r}")
            print(f"    on the MESH TABLE ({len(on_mesh)}): "
                  + (", ".join(k for k, _ in on_mesh) or "-- none, this set is aligned"))
            print(f"    on the CELL SET   ({len(on_cell)}): "
                  + (", ".join(f"{k}{'' if rec else '*'}" for k, _w, rec in on_cell) or "--")
                  + "        (* = record: false)")
            if on_mesh:
                bad += 1
            if z is None:
                print("  (no trajectory.npz on disk -- DEAD / LENGTHS / SUMS skipped, "
                      "run `-o generate` first)")
                continue
            d = dead_blocks(z, {cs: declared.get(cs, [])} if cs else {})
            if d:
                print("  DEAD    declared and never written: "
                      + ", ".join(f"{s}.{b} (|max| {m:g})" for s, b, m in d))
                if not a.quiet_dead:
                    bad += 1
            L = lengths(z, vs)
            if L:
                print("  LENGTHS " + "; ".join(f"{c}: {w}" for c, w in L[:4]))
                bad += 1
            s = sums(z, vs)
            if s and s["divisions"]:
                verdict = ("exact split" if s["worst"] < 1e-6 else
                           "ok (birth-at-target ramp)" if s["worst"] < 0.8 else
                           "SUSPECT -- a copy where a split belongs?")
                print(f"  SUMS    V0f over {s['divisions']} division(s): worst jump "
                      f"{s['worst']:.3f} cell-equivalents PER CELL BORN, at frame {s['frame']} "
                      f"({s['born_there']} born) -- {verdict}")
                if s["worst"] >= 0.8:
                    bad += 1

    print(f"\n{len(names) - failed} spec(s) analysed, {failed} failed to run")
    print(f"CENSUS TOTAL: {tot_mesh} per-cell array(s) still on a mesh table across the group")
    return 2 if failed else (1 if bad else 0)


if __name__ == "__main__":
    sys.exit(main())
