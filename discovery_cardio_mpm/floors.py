#!/usr/bin/env python
"""floors -- where is zero, and how big is nothing.

PHASE 2, ITEM 1. THE MOST IMPORTANT NUMBER THIS PROJECT WILL PRODUCE
================================================================================================
Every score the campaign will ever quote is meaningless until we know what a model that knows
NOTHING scores. The previous campaign never asked. Its documentation asserted that a model
recovering no loop scores about zero on the objective; the measured answer is **+0.075 with a
spread of +-0.117**, so every headline number was read against the wrong origin and the spread of
the null was larger than most of the differences that were called findings.

So: score the trivial models, on the frozen split, on the frozen mask, and report everything
afterwards as a **difference from the null** -- never as a bare number and never as a ratio, both
of which can be made to look like anything once the origin is not at zero.

THE BANK
------------------------------------------------------------------------------------------------
  N0  zero motion         predict nothing. The origin everything else is measured from.
  N1  mean translation    the whole sheet slides rigidly by the field average. Knows the beat's
                          timing and amplitude and nothing about its shape.
  N2  replay              predict this beat by copying the previous one. NO physics, no fitting,
                          no parameters -- and on the evidence already on disk it is the bar the
                          previous campaign never cleared.
  N3  boundary            interpolate inward from the pinned band, harmonically, no physics. What
                          the anchoring alone is worth.
  N4  passive             the real model with the muscle switched off.
  N5  sham fields         the real model with its fields present but never trained. Because
                          *freezing a field is not removing it*, and reporting "fields off" when
                          it means "fields not fitted" is a silent confound.

N4 and N5 need a forward run; N0-N3 are arithmetic on the recording and cost seconds.

WHAT THIS DOES NOT DO
------------------------------------------------------------------------------------------------
It does not fit anything, so it cannot say whether the active model beats these. That is the
STOP, and it needs the noise floor (item 2) beside it: a model beats a null when it beats it by
more than the spread of doing the same thing twice. This file measures the left-hand side.

    python floors.py --nulls          # N0-N3, arithmetic only
    python floors.py --nulls --model  # + N4, N5 (needs the engine)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
PY = sys.executable
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import data as D                                                    # noqa: E402
import descriptors as DS                                            # noqa: E402
import split as SP                                                  # noqa: E402

FIT_SPEC = os.path.join(HERE, "config", "material", "material_aniso_cardio_fit.yaml")


# ---------------------------------------------------------------------------------------------
# GEOMETRY. Everything is scored on ONE support and ONE mask, both frozen in Phase 1.
# ---------------------------------------------------------------------------------------------
def reference_dump(device="cuda:0", timeout=7200):
    """One forward pass whose artefact carries the geometry everything else is scored on.

    NOT a rebuild. The first version of this file constructed the particle layout itself with
    `engine.build(spec, "cpu")` and scored a GPU run against it -- and the two layouts are not the
    same, so it produced a dramatic false finding (that the pinned band drives a third of the
    sheet) before the mapping was checked against the run's own `rest`.

    The cause is worth knowing on its own: **the initial particle layout depends on the DEVICE.**
    Same seed, CPU versus GPU, and the positions differ by the width of the whole sheet; same
    device and a different seed changes nothing, because the layout comes from the spec's own
    generator. So a fit on the processor and a fit on the card do not start from the same
    material, and nothing may be compared across the two. Phase 0's determinism check compared
    CPU with CPU and GPU with GPU, so it could not have seen this.

    The rule that follows: geometry is READ FROM THE ARTEFACT, never rebuilt beside it.
    """
    cache = os.path.join(HERE, "_metrology", f"_geom_{device.replace(':', '')}.npz")
    if os.path.exists(cache):
        return cache
    d = tempfile.mkdtemp(prefix="floor_geom_")
    dump = os.path.join(d, "dump.npz")
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([PY, os.path.join(HERE, "train.py"), FIT_SPEC, "--seed", "11",
                        "--device", device, "--outdir", d, "--eval_dump", dump,
                        "--amplitude", "0", "--allow_nondeterministic_ops", "1"],
                       capture_output=True, text=True, env=env, timeout=timeout)
    if not os.path.exists(dump):
        raise RuntimeError(((r.stderr or "").strip().splitlines() or ["no output"])[-1][:200])
    z = np.load(dump)
    np.savez(cache, rest=z["rest"], bnd=z["bnd"], sim_passive=z["sim_d"], real_fit=z["real_d"],
             trainer_mov=z["mov"])
    return cache


def geometry(device="cuda:0", bwidth=0.06):
    """(rest, real_disp, bnd, mask, split) on the RUN's support, taken from its own artefact."""
    from scipy.spatial import cKDTree
    g = np.load(reference_dump(device))
    rest, bnd = g["rest"].astype(np.float64), g["bnd"].astype(bool)

    z = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)
    P = z["pos"].astype(np.float64)
    Pm = D.DOM_LO + D.DOM * P
    node = cKDTree(Pm[0]).query(rest)[1]
    real_disp = (Pm[:, node] - Pm[0, node])

    node_mask = np.load(os.path.join(HERE, "_data", "eval_mask.npy"))
    mask = node_mask[node] & ~bnd
    return rest, real_disp, bnd, mask, SP.load()


def check_geometry(device="cuda:0"):
    """The check that would have caught the false finding: does our mapping reproduce the run's
    own `real_d`? If it does not, nothing computed here is about the same particles."""
    g = np.load(reference_dump(device))
    rest, real_disp, bnd, mask, sp = geometry(device)
    G = sp["beats"]["common_length"]; o = sp["fit"]["span"][0]
    mine = real_disp[o:o + G] - real_disp[o]
    theirs = g["real_fit"].astype(np.float64)
    n = min(mine.shape[0], theirs.shape[0])
    d = float(np.abs(mine[:n] - theirs[:n]).max())
    scale = float(np.abs(theirs[:n]).max())
    return d / scale if scale else float("nan"), d, scale


# ---------------------------------------------------------------------------------------------
# THE NULLS
# ---------------------------------------------------------------------------------------------
def _laplace_infill(frame, bnd, rest, iters=400):
    """Harmonic interpolation of a [N,2] field from the band into the interior. No physics."""
    side = int(round(np.sqrt(rest.shape[0])))
    if side * side != rest.shape[0]:
        return None
    order = np.lexsort((rest[:, 1], rest[:, 0]))                     # particles sit on a lattice
    inv = np.argsort(order)
    g = frame[order].reshape(side, side, 2).copy()
    m = bnd[order].reshape(side, side)
    g[~m] = 0.0
    for _ in range(iters):
        nb = np.zeros_like(g)
        nb[1:-1, 1:-1] = 0.25 * (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        g = np.where(m[..., None], g, nb)
    return g.reshape(-1, 2)[inv]


def nulls(with_model=False, device="cuda:0"):
    rel, d, scale = check_geometry(device)
    if not (rel < 0.01):
        raise RuntimeError(f"geometry mismatch: our mapping differs from the run's own real_d by "
                           f"{rel:.3f} of the signal. Nothing below would be about the same "
                           f"particles. (This check exists because it once failed.)")
    rest, real_disp, bnd, mask, sp = geometry(device)
    G = sp["beats"]["common_length"]
    fit_span = sp["fit"]["span"]
    held = sp["heldout_beats"]["spans"]

    def beat(onset):
        return real_disp[onset:onset + G] - real_disp[onset]

    windows = {"fit": beat(fit_span[0])}
    for i, h in enumerate(held):
        windows[f"heldout{i}"] = beat(h[0])

    out = {"geometry_check": {"relative_mismatch": rel, "abs": d, "signal": scale},
           "support": {"particles": int(rest.shape[0]), "scored": int(mask.sum()),
                       "band": int(bnd.sum()), "beat_frames": int(G)},
           "split_sha": open(SP.SPLIT + ".sha256").read().strip(), "models": {}}

    def register(name, fn, note):
        rec = {"note": note, "windows": {}}
        for wname, real in windows.items():
            sim = fn(wname, real)
            if sim is None:
                continue
            rec["windows"][wname] = score(sim, real, mask)
        out["models"][name] = rec
        return rec

    # N0 -- predict nothing
    register("N0_zero", lambda w, r: np.zeros_like(r),
             "predict no motion at all. THE ORIGIN: every score below is a difference from this.")

    # N1 -- the sheet slides rigidly by the field mean
    register("N1_mean_translation",
             lambda w, r: np.repeat(r.mean(axis=1, keepdims=True), r.shape[1], axis=1),
             "the whole sheet slides by the field average: right timing, right amplitude, no shape")

    # N2 -- copy the previous beat
    order = ["fit"] + [f"heldout{i}" for i in range(len(held))]
    onsets = [fit_span[0]] + [h[0] for h in held]
    prev = {}
    for nm, o in zip(order, onsets):
        j = sorted(x[0] for x in [fit_span] + held)
        k = j.index(o)
        prev[nm] = beat(j[k - 1]) if k > 0 else None
    register("N2_replay", lambda w, r: prev.get(w),
             "predict this beat by copying the previous one. No physics, no parameters, no fit.")

    # N3 -- harmonic infill from the pinned band
    register("N3_boundary",
             lambda w, r: np.stack([_laplace_infill(f, bnd, rest) for f in r]),
             "interpolate inward from the pinned band, harmonically. What the anchoring is worth.")

    if with_model:
        for name, args_, note in (
            ("N4_passive", ["--amplitude", "0"],
             "the real model with the muscle switched off"),
            ("N5_sham_fields", ["--stiff_src", "siren", "--siren_fibre", "1", "--learn", "dur"],
             "the real model with its fields PRESENT but never trained -- freezing a field is not "
             "removing it")):
            d = tempfile.mkdtemp(prefix=f"floor_{name}_")
            dump = os.path.join(d, "dump.npz")
            env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
            r = subprocess.run([PY, os.path.join(HERE, "train.py"), FIT_SPEC, "--seed", "11",
                                "--device", device, "--outdir", d, "--eval_dump", dump,
                                "--allow_nondeterministic_ops", "1", *args_],
                               capture_output=True, text=True, env=env, timeout=7200)
            if not os.path.exists(dump):
                out["models"][name] = {"note": note, "error":
                                       ((r.stderr or "").strip().splitlines() or ["?"])[-1][:140]}
                continue
            z = np.load(dump)
            sim, real = z["sim_d"].astype(np.float64), z["real_d"].astype(np.float64)
            n = min(sim.shape[0], real.shape[0], G)
            out["models"][name] = {"note": note,
                                   "windows": {"fit": score(sim[:n], real[:n], mask)}}
    return out


def score(sim, real, mask):
    """Both rulers, on the frozen mask. The inherited one for comparability with the old record,
    the Track B axes because they are what the campaign is judged on."""
    rec = {}
    r = DS.loop_residual(sim, real, mask)
    for k in ("magnitude_peak", "opening_area", "opening_loopiness", "direction_chirality",
              "shape_minor_fraction"):
        rec[k] = r[k]["ratio"]
    rec["orientation_rad"] = r["orientation_error_rad"]["median_rad_covariance"]
    try:
        import torch
        import harmonic_inherited as HARM
        t = lambda a: torch.tensor(np.ascontiguousarray(a[:, mask]), dtype=torch.float32)
        m, sd = HARM.harmonic_stats(t(sim), t(real), None)
        rec["loopscore"], rec["loopscore_sd"] = float(m), float(sd)
        rec["interior_r2"] = float(HARM.interior_r2(t(sim), t(real), None)) \
            if hasattr(HARM, "interior_r2") else None
    except Exception as e:
        rec["loopscore_error"] = f"{type(e).__name__}: {e}"
    return rec


def report(out):
    print(f"\n{'=' * 104}\n  PHASE 2 ITEM 1 -- WHERE IS ZERO\n{'=' * 104}")
    s = out["support"]
    print(f"  {s['scored']} scored particles of {s['particles']} ({s['band']} in the pinned band), "
          f"{s['beat_frames']}-frame beats, frozen split {out['split_sha'][:16]}")
    n0 = out["models"].get("N0_zero", {}).get("windows", {}).get("fit", {})
    base = n0.get("loopscore")
    print(f"\n  {'model':<22s} {'window':<10s} {'LoopScore':>10s} {'vs N0':>8s} "
          f"{'peak':>7s} {'opening':>8s} {'chir':>6s}")
    for name, rec in out["models"].items():
        if "error" in rec:
            print(f"  {name:<22s} FAILED: {rec['error'][:60]}")
            continue
        for w, sc in rec["windows"].items():
            ls = sc.get("loopscore", float("nan"))
            dv = (ls - base) if base is not None else float("nan")
            print(f"  {name:<22s} {w:<10s} {ls:>10.4f} {dv:>+8.4f} "
                  f"{sc.get('magnitude_peak', float('nan')):>7.3f} "
                  f"{sc.get('opening_area', float('nan')):>8.3f} "
                  f"{sc.get('direction_chirality', float('nan')):>6.3f}")
    print(f"\n  Everything afterwards is reported as the 'vs N0' column, never as the bare score.")
    print(f"  The active model has to beat the BEST of these by more than the noise (item 2).")
    print("=" * 104)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulls", action="store_true")
    ap.add_argument("--model", action="store_true", help="also run N4 and N5 (needs the engine)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)
    out = nulls(with_model=a.model, device=a.device)
    report(out)
    p = os.path.join(HERE, "_metrology", "floors.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev.update(out)
    json.dump(prev, open(p, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
