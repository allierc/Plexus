#!/usr/bin/env python
"""feedback -- close the loop: let the FIBRES do what the rigid plates did.

    python feedback.py --device cuda:0

THE QUESTION. `plate_confine` produces an ovoid by fiat -- the plates are rigid, so the aspect ratio
is the tissue's mechanics answering a boundary that was decided before the run started. The question
worth asking is whether the MATRIX can do it: can a fibrous solid, resisting only by deforming, hold a
growing epithelium back enough to change its shape? That needs the reaction force, and a replayed
tissue has nowhere to put one. So:

    STEP 0   an ANISOTROPIC matrix -- a disc-shaped cavity, thin above and below the tissue and roomy
             around its equator -- loaded by the unconfined tissue. `ecm_from_cell` records the contact
             pressure it develops, by direction, per frame. Thin matrix resists sooner, so the map is
             polar-heavy: this is the anisotropy the fibres actually generate, measured rather than
             imposed.
    STEP 1   CALIBRATE THE GAIN. The pressure is in MPM units over a unit box and the vertex model is
             in AVM units over a 50-unit world; the conversion is the dimensional calibration the
             two-pass structure exists to avoid, so the coupling constant is chosen and not derived.
             Three tissue passes at different gains cost 80 seconds each and turn "I picked a number"
             into a curve of aspect ratio against coupling strength, which is the honest form of the
             claim.
    STEP 2   the tissue re-run WITH that load, and then the matrix re-run against the tissue that
             comes out. One full staggered iteration. `ecm_load`'s docstring states what a single
             iteration does and does not establish.

WHAT A RESULT HERE MEANS. If the fibres produce an ovoid at a gain in the range where the tissue is
otherwise healthy, then matrix resistance is a sufficient mechanism for the shape and the plates were
a stand-in for something real. If it takes a gain so large that the tissue collapses instead of
elongating, that is also an answer: this matrix, at this stiffness, cannot do it, and the plates are
modelling something the fibres are not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "discovery_okuda", "ops"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

BUFFER_X = 4

# THE MATRIX IS THE ANISOTROPIC ONE. A spherical cavity loads the tissue evenly from every direction,
# so its reaction can only make the tissue SMALLER, never a different shape -- and "the matrix
# suppressed growth" is a weaker claim than the one being tested. The disc is thin along z and roomy
# in the equatorial plane, so the fibres resist the poles first.
BASE = dict(n_particles=140000, n_grid=48, youngs=15.0, k_contact=900.0,
            cavity_r=0.200, cavity_h=0.095, axis=2, n_fibres=6000, fibre_len=0.16,
            align=0.0, substep_dt=2.0e-4, stress_scale=0.08)


def aspect_of(npz):
    z = np.load(npz)
    eq, ax = np.asarray(z["r_eq"], float), np.asarray(z["r_ax"], float)
    return float(eq[-1] / max(ax[-1], 1e-9)), float(eq[-1]), float(ax[-1]), int(z["n_cells"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gains", default="0.004,0.012,0.035")
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--movie-frames", type=int, default=90)
    ap.add_argument("--step0", default="44_fb_i0_disc")
    ap.add_argument("--step2", default="45_fb_i1_disc")
    ap.add_argument("--skip-step0", action="store_true",
                    help="reuse an existing load.npz from --step0 instead of re-running the matrix")
    a = ap.parse_args()

    import combine as C
    import run_ecm as R
    import tissue as TIS

    report = {"gains": {}, "base": BASE}
    t0 = time.time()

    # ---- STEP 0: the anisotropic matrix, loaded by the unconfined tissue -----------------------
    npz0 = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X)
    load = os.path.join(R.LOG, a.step0, "load.npz")
    if not (a.skip_step0 and os.path.exists(load)):
        spec, info = C.build(a.step0, npz0, **dict(BASE))
        out_dir = os.path.join(R.LOG, a.step0)
        os.makedirs(out_dir, exist_ok=True)
        info["varied"] = {"iteration": 0, "cavity": "disc"}
        json.dump(info, open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
        print(f"\n[feedback] STEP 0: {a.step0} -- disc cavity, unconfined tissue", flush=True)
        m0 = R.run(a.step0, spec, device=a.device, movie=True,
                   render_kw={"movie_frames": a.movie_frames})
        report["step0"] = {k: m0[k] for k in ("contact_frame", "strained_frac_end", "max_disp",
                                             "exploded")}
    if not os.path.exists(load):
        raise RuntimeError(f"no load.npz at {load} -- step 0 recorded no contact pressure, so there "
                           f"is nothing for the tissue to feel")
    P = np.asarray(np.load(load)["pmap"], float)
    # THE ANISOTROPY, AS A NUMBER. If the polar rows are not carrying more pressure than the equatorial
    # ones then the disc cavity did not do what it was chosen for, and any elongation downstream came
    # from somewhere else.
    nth = P.shape[1]
    pol = P[:, [0, 1, 2, nth - 3, nth - 2, nth - 1], :].mean()
    equ = P[:, nth // 2 - 3:nth // 2 + 3, :].mean()
    print(f"[feedback] recorded load: {P.shape[0]} frames, peak {P.max():.4g}; "
          f"polar mean {pol:.4g} vs equatorial {equ:.4g} = anisotropy {pol / max(equ, 1e-12):.2f}x",
          flush=True)
    report["load"] = {"peak": float(P.max()), "polar_mean": float(pol),
                      "equatorial_mean": float(equ), "anisotropy": float(pol / max(equ, 1e-12))}

    # ---- STEP 1: calibrate the gain. 80 seconds each, so this is a curve and not a guess ------
    print(f"\n[feedback] STEP 1: gain calibration", flush=True)
    a0, eq0, ax0, nc0 = aspect_of(npz0)
    print(f"[feedback]   gain 0 (no load): aspect {a0:.2f}  r_eq {eq0:.2f}  r_ax {ax0:.2f}  "
          f"{nc0} cells", flush=True)
    report["gains"]["0"] = {"aspect": a0, "r_eq": eq0, "r_ax": ax0, "n_cells": nc0}
    best = None
    for g in [float(x) for x in a.gains.split(",")]:
        npz = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X,
                                load_npz=load, load_gain=g,
                                tag_extra=f"_load{g:g}".replace(".", "p"))
        asp, eq, ax_, nc = aspect_of(npz)
        print(f"[feedback]   gain {g:g}: aspect {asp:.2f}  r_eq {eq:.2f}  r_ax {ax_:.2f}  "
              f"{nc} cells", flush=True)
        report["gains"][f"{g:g}"] = {"aspect": asp, "r_eq": eq, "r_ax": ax_, "n_cells": nc,
                                     "npz": npz}
        # THE LARGEST GAIN THAT ELONGATES WITHOUT CRUSHING. `r_eq` must not collapse: a tissue that
        # got smaller in BOTH axes was squeezed, not shaped, and its aspect ratio is an artefact of
        # two shrinkages at different rates.
        if asp > (best[1] if best else a0 + 0.02) and eq > 0.85 * eq0:
            best = (g, asp, npz)
    if best is None:
        print("[feedback] no gain in the sweep elongated the tissue without shrinking it. "
              "That is the result: at this matrix stiffness the fibres cannot do what the plates do.",
              flush=True)
        json.dump(report, open(os.path.join(R.LOG, "feedback.json"), "w"), indent=1)
        return
    gain, asp, npz1 = best
    print(f"\n[feedback] chose gain {gain:g} (aspect {asp:.2f} vs {a0:.2f} unloaded)", flush=True)
    report["chosen_gain"] = gain

    # ---- STEP 2: the matrix again, against the tissue the matrix shaped -----------------------
    spec, info = C.build(a.step2, npz1, **dict(BASE))
    out_dir = os.path.join(R.LOG, a.step2)
    os.makedirs(out_dir, exist_ok=True)
    info["varied"] = {"iteration": 1, "cavity": "disc", "load_gain": gain}
    json.dump(info, open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
    print(f"\n[feedback] STEP 2: {a.step2} -- matrix vs the load-shaped tissue "
          f"(aspect {info['aspect_end']:.2f})", flush=True)
    m1 = R.run(a.step2, spec, device=a.device, movie=True,
               render_kw={"movie_frames": a.movie_frames})
    report["step2"] = {k: m1[k] for k in ("contact_frame", "strained_frac_end", "max_disp",
                                          "exploded")}
    report["step2"]["aspect_end"] = info["aspect_end"]

    # CONVERGENCE, OR THE LACK OF IT. One iteration is not a fixed point; the change in the recorded
    # load between iterations is the only thing that says how far from one this is.
    l2 = os.path.join(R.LOG, a.step2, "load.npz")
    if os.path.exists(l2):
        Q = np.asarray(np.load(l2)["pmap"], float)
        n = min(P.shape[0], Q.shape[0])
        rel = float(np.abs(Q[:n] - P[:n]).mean() / max(np.abs(P[:n]).mean(), 1e-12))
        print(f"[feedback] load changed {100 * rel:.0f}% between iteration 0 and 1 -- "
              f"{'near a fixed point' if rel < 0.15 else 'NOT converged; a further iteration would move it again'}",
              flush=True)
        report["load_change_iter0_to_1"] = rel
    json.dump(report, open(os.path.join(R.LOG, "feedback.json"), "w"), indent=1)
    print(f"\n[feedback] done in {(time.time() - t0) / 60:.0f} min -> "
          f"{os.path.relpath(os.path.join(R.LOG, 'feedback.json'), ROOT)}", flush=True)


if __name__ == "__main__":
    main()
