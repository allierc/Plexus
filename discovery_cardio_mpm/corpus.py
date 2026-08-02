#!/usr/bin/env python
"""corpus -- the simulations we already own, and what they are good for.

WHY THIS MATTERS MORE THAN IT LOOKS
================================================================================================
`graphs_data/material` holds seventy-one MPM runs produced by this codebase, each with its spec,
its full particle trajectory and its movies. Eight of them drive tissue with active traction of a
pattern we CHOSE -- swirl, radial-in, horizontal, vertical, four quadrants, and the cardio recipe
itself. Those are loops whose ground truth is their own recipe.

That is exactly what an instrument has to be certified against. The rule the previous campaign
paid for the hard way is that a metric must reproduce a known ordering on cases whose answer we
already know, BEFORE it is allowed to judge a recording whose answer we do not. Until now the only
known-answer cases available were ellipses drawn by hand. These are the real forward model, at the
real particle count, with a known driver.

They also cost nothing. They are already computed.

    python corpus.py                 # inventory: what exists, and does it still run today?
    python corpus.py --descriptors   # read every active run with the Track B descriptors
    python corpus.py --figure OUT    # the visual instrument, rebuilt

THE PROVENANCE QUESTION, ANSWERED RATHER THAN ASSUMED
------------------------------------------------------------------------------------------------
"These were run with this codebase" is a claim, and the inventory checks it: every operator each
spec names is resolved against the live registry, and any that no longer exists is reported. That
is the same check that would have caught the two operator renames which destroyed four batches of
the previous campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "src")
DATA = os.path.join(REPO, "graphs_data", "material")
LOGDIR = os.path.join(REPO, "log", "material")

sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# The runs whose driver is an ACTIVE traction pattern we chose. These are the certification set;
# the rest of the corpus is water, snow, bouncing balls and other material tests.
ACTIVE = ["material_active_horizontal", "material_active_vertical", "material_active_swirl",
          "material_active_radial_in", "material_active_phase_horizontal",
          "material_active_phase_quadrants", "material_active_phase_radial",
          "material_active_phase_swirl", "material_central_contraction", "material_aniso_cardio"]

# What each active run's own recipe says its motion should be. Written from the SPEC, not from
# looking at the answer -- this is the ordering the descriptors have to reproduce, and it must not
# be edited to make a metric pass.
EXPECTED = {
    "material_active_radial_in":   "radial: contraction toward a centre; paths should be close to "
                                   "straight lines, so LITTLE enclosed area",
    "material_active_horizontal":  "one fixed contraction axis; paths near-linear along it",
    "material_active_vertical":    "one fixed contraction axis, orthogonal to horizontal -- the "
                                   "orientation axis must separate these two and nothing else should",
    "material_active_swirl":       "a rotating direction field; paths should enclose MORE area than "
                                   "the fixed-axis runs",
    "material_active_phase_radial":    "a radial phase DELAY: neighbouring points contract at "
                                       "different times -- coordination structure, invisible to LoopScore",
    "material_active_phase_swirl":     "a swirling phase delay -- ditto",
    "material_active_phase_quadrants": "four regions out of phase -- ditto",
    "material_active_phase_horizontal": "a travelling phase delay along x -- ditto",
    "material_central_contraction": "a central contractile spot",
    "material_aniso_cardio":       "the cardio recipe: anisotropic active stress on a fibre field",
}


def _spec_path(name):
    for p in (os.path.join(DATA, name, "spec.yaml"), os.path.join(LOGDIR, name, "config.yaml")):
        if os.path.exists(p):
            return p
    return None


def inventory(names=None, check_ops=True):
    """What exists, what shape it is, and whether its spec still resolves against today's library."""
    import plexus.operators                                    # importing is what REGISTERS them
    from plexus.models.registry import get_operator
    from plexus.schema import load

    rows = []
    for name in sorted(names or os.listdir(DATA)):
        d = os.path.join(DATA, name)
        if not os.path.isdir(d):
            continue
        traj = os.path.join(d, "trajectory.npz")
        row = {"name": name, "dir": d, "has_trajectory": os.path.exists(traj),
               "spec": _spec_path(name), "movies": sorted(
                   f for f in os.listdir(d) if f.endswith(".mp4"))[:4]}
        if row["has_trajectory"]:
            with np.load(traj) as z:
                key = "mpm_particle__pos"
                row["frames"], row["particles"] = (list(z[key].shape)[:2] if key in z.files
                                                   else [None, None])
                row["arrays"] = len(z.files)
        if check_ops and row["spec"]:
            try:
                spec = load(row["spec"])
                ops = sorted({o.op for o in spec.operators})
                missing = []
                for o in ops:
                    try:
                        get_operator(o)
                    except Exception:
                        missing.append(o)
                row["operators"] = ops
                row["operators_missing"] = missing
                row["runs_today"] = not missing
            except Exception as e:
                row["spec_error"] = f"{type(e).__name__}: {e}"
                row["runs_today"] = False
        rows.append(row)
    return rows


def beat_window(P, min_gap=8):
    """One beat of a periodic run: (onset, length). Falls back to the first 60 frames."""
    from scipy.signal import find_peaks
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    pk, _ = find_peaks(spd, height=spd.mean(), distance=min_gap)
    if len(pk) >= 2:
        gaps = np.diff(pk)
        g = int(np.median(gaps))
        return int(pk[0]), g, [int(x) for x in pk]
    return 0, min(60, P.shape[0]), [int(x) for x in pk]


def load_beat(name, moving_frac=0.2):
    """The per-particle path over one beat, plus a moving-node mask. [G,N,2], [N] bool."""
    with np.load(os.path.join(DATA, name, "trajectory.npz")) as z:
        P = z["mpm_particle__pos"].astype(np.float64)
    o, g, peaks = beat_window(P)
    path = P[o:o + g] - P[o]
    amp = np.linalg.norm(path, axis=-1).max(axis=0)
    mov = amp > moving_frac * amp.max()
    return path, mov, {"onset": o, "length": g, "peaks": peaks,
                       "frames": int(P.shape[0]), "particles": int(P.shape[1])}


def read_descriptors(names=None):
    """Read every active run with the Track B descriptors, and with the inherited LoopScore.

    Two things are being asked at once:
      * do the descriptors reproduce the ordering the SPECS predict (swirl encloses more than
        radial, horizontal and vertical differ only in orientation)?
      * does the inherited objective see the coordination the `phase_*` runs were built to carry?
    """
    import descriptors as DS
    out = []
    for name in (names or ACTIVE):
        if not os.path.exists(os.path.join(DATA, name, "trajectory.npz")):
            continue
        path, mov, meta = load_beat(name)
        res = DS.loop_residual(path, path, mov)                 # self: the reference reading
        rec = {"name": name, "expected": EXPECTED.get(name, ""), **meta,
               "nodes_scored": res["n_nodes"],
               "area": res["opening_area"]["real"],
               "loopiness": res["opening_loopiness"]["real"],
               "minor": res["shape_minor_fraction"]["real"],
               "peak": res["magnitude_peak"]["real"]}

        # absolute orientation of the run, so horizontal and vertical can be told apart
        ang = DS._major_axis_angle(path[:, mov])
        rec["axis_angle_median_rad"] = float(np.median(ang))

        # the coordination probe: scramble each node's timing and ask both instruments
        scr = DS.random_phase_scramble(path[:, mov], np.random.default_rng(0))
        d_scr = DS.loop_residual(scr, path[:, mov])
        rec["scramble_moves_descriptors"] = {
            "magnitude": d_scr["magnitude_peak"]["ratio"],
            "opening": d_scr["opening_area"]["ratio"],
            "direction": d_scr["direction_chirality"]["ratio"]}
        try:
            import torch
            import harmonic_inherited as HARM
            t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
            # harmonic_score returns a float, not a tuple -- harmonic_stats returns (mean, sd).
            rec["loopscore_self"] = float(HARM.harmonic_score(t(path[:, mov]), t(path[:, mov]), None))
            rec["loopscore_scrambled"] = float(HARM.harmonic_score(t(scr), t(path[:, mov]), None))
            # and the null the previous campaign never measured: predict no motion at all
            rec["loopscore_zero_motion"] = float(
                HARM.harmonic_score(torch.zeros_like(t(path[:, mov])), t(path[:, mov]), None))
        except Exception as e:
            rec["loopscore_error"] = f"{type(e).__name__}: {e}"
        out.append(rec)
    return out


def figure(out_png, names=None, n=6):
    """The visual instrument, rebuilt.

    The inherited montage (`harmonic_montage.py`) and the metric-sensitivity figure
    (`make_loopscore_sensitivity.py`) BOTH crash today on a module deleted with a sibling
    directory, so the campaign's primary picture cannot be drawn at all. This is the replacement,
    and it depends on nothing outside this folder.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [x for x in (names or ACTIVE)
             if os.path.exists(os.path.join(DATA, x, "trajectory.npz"))]
    fig, axes = plt.subplots(2, len(names), figsize=(2.3 * len(names), 5.2), squeeze=False)
    rng = np.random.default_rng(0)
    for j, name in enumerate(names):
        path, mov, meta = load_beat(name)
        idx = np.flatnonzero(mov)
        pick = rng.choice(idx, size=min(n * n, idx.size), replace=False)
        # top: the loops themselves, centred, on a common scale
        ax = axes[0][j]
        p = path[:, pick]
        p = p - p.mean(axis=0, keepdims=True)
        sc = np.abs(p).max() or 1.0
        for k in range(p.shape[1]):
            ax.plot(p[:, k, 0] / sc, p[:, k, 1] / sc, lw=0.5, alpha=0.55, color="#2B4C7E")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_title(name.replace("material_", "").replace("_", " "), fontsize=7)
        if j == 0:
            ax.set_ylabel("one beat, per particle\n(centred, own scale)", fontsize=7)
        # bottom: the same loops with each particle's timing scrambled
        ax = axes[1][j]
        import descriptors as DS
        q = DS.random_phase_scramble(p, np.random.default_rng(1))
        for k in range(q.shape[1]):
            ax.plot(q[:, k, 0] / sc, q[:, k, 1] / sc, lw=0.5, alpha=0.55, color="#B3261E")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        if j == 0:
            ax.set_ylabel("same, timing scrambled\nper particle", fontsize=7)
    fig.suptitle("The corpus we already own: one beat per run, and the same beat with coordination destroyed.\n"
                 "The two rows are indistinguishable to the objective the previous campaign ranked on.",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptors", action="store_true")
    ap.add_argument("--figure", metavar="OUT", default=None)
    ap.add_argument("--all", action="store_true", help="inventory the whole corpus, not just the active runs")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    if a.figure:
        print(f"[corpus] wrote {figure(a.figure)}")
        return 0

    if a.descriptors:
        rows = read_descriptors()
        print(f"\n{'=' * 108}\n  THE CORPUS READ WITH THE TRACK B DESCRIPTORS "
              f"(each run against itself: these are its own properties)\n{'=' * 108}")
        print(f"  {'run':<34s} {'nodes':>6s} {'beat':>5s} {'|area|':>10s} {'loopy':>6s} "
              f"{'minor':>6s} {'axis':>6s} {'LS self':>8s} {'LS scrambled':>12s} {'LS zero':>9s}")
        for r in rows:
            ls = f"{r.get('loopscore_self', float('nan')):8.4f}"
            lsc = f"{r.get('loopscore_scrambled', float('nan')):12.4f}"
            lsz = f"{r.get('loopscore_zero_motion', float('nan')):9.4f}"
            print(f"  {r['name']:<34s} {r['nodes_scored']:>6d} {r['length']:>5d} "
                  f"{r['area']:>10.3e} {r['loopiness']:>6.3f} {r['minor']:>6.3f} "
                  f"{r['axis_angle_median_rad']:>6.3f} {ls} {lsc} {lsz}")
        print("\n  'LS scrambled' is the inherited objective scoring a tissue whose particles were")
        print("  given independent random timing. If it is not far below 'LS self', the objective")
        print("  cannot see coordination -- on real forward-model output, not on a toy.")
        json.dump(rows, open(os.path.join(HERE, "_metrology", "corpus_descriptors.json"), "w"),
                  indent=1, default=float)
        return 0

    rows = inventory(None if a.all else ACTIVE)
    print(f"\n{'=' * 108}\n  THE CORPUS -- MPM runs already computed by this codebase\n{'=' * 108}")
    print(f"  {'run':<36s} {'frames':>7s} {'particles':>10s} {'ops':>4s} {'runs today':>11s}  movies")
    for r in rows:
        print(f"  {r['name']:<36s} {str(r.get('frames', '-')):>7s} {str(r.get('particles', '-')):>10s} "
              f"{len(r.get('operators', [])):>4d} {str(r.get('runs_today', '?')):>11s}  "
              f"{len(r.get('movies', []))} mp4")
        if r.get("operators_missing"):
            print(f"      MISSING OPERATORS: {r['operators_missing']}")
    n_ok = sum(1 for r in rows if r.get("runs_today"))
    print(f"\n  {n_ok}/{len(rows)} specs resolve against today's operator registry.")
    print(f"  Trajectories present: {sum(1 for r in rows if r['has_trajectory'])}/{len(rows)}")
    json.dump(rows, open(os.path.join(HERE, "_metrology", "corpus.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
