#!/usr/bin/env python
"""aniso -- fibre anisotropy -> stress anisotropy -> GROWTH anisotropy -> an anisotropic spheroid.

    python aniso.py --device cuda:0

THE CHAIN THIS TESTS, one link at a time, with a number on each:

    1  the fibres are laid down with a preferred direction (`ecm_seed align`), so the matrix is
       DENSER along one axis than the others
    2  a growing tissue therefore meets more material in that direction and the matrix pushes back
       harder there -- `cell_to_ecm` records the contact pressure by direction, so this is measured,
       not assumed
    3  `ecm_growth_gate_3d` slows the CELL CYCLE where that pressure is high: a cell whose target
       volume grows more slowly reaches `divide_3d`'s doubling threshold later and divides less often
    4  so the spheroid adds cells preferentially in the compliant directions and comes out ANISOTROPIC
       -- and the anisotropy of its semi-axes should line up with the anisotropy of the pressure map

EVERY LINK CAN FAIL, AND THE FAILURES LOOK DIFFERENT. If (2) is null the fibre alignment made no
mechanical difference and the honest report is that this matrix cannot transmit its architecture --
which `ecm_ops` already warns is possible, because MPM interpolates every particle onto a continuum
grid and a fibrous ARRANGEMENT of an isotropic material still responds isotropically. Any pressure
anisotropy here is where the material happens to be DENSE, not fibre reinforcement, and that is a
weaker claim that has to be made in those words. If (2) holds and (4) is null the gate is too weak or
too saturated, which is `p_half` and shows up in the gate range this prints. The point of measuring
each link is that the end result cannot be reported without saying which link carried it.

WHY GROWTH RATE AND NOT FORCE. A force fights the growth every frame and is bounded by how hard you
can push before cells invert. A rate difference COMPOUNDS over 400 frames, so a few percent of stress
anisotropy becomes a visible shape anisotropy -- which is also the biology, since load suppresses
proliferation rather than merely deforming a tissue.
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
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

BUFFER_X = 4

# ALIGNED FIBRES, SPHERICAL CAVITY. The cavity has to be isotropic or it would supply an anisotropy of
# its own and the experiment could not attribute the result to the fibres. `align 0.85` about x is
# strong: a weak alignment is the more interesting experiment and the wrong FIRST one, because a null
# at 0.85 says the mechanism cannot work here at all.
BASE = dict(n_particles=140000, n_grid=48, youngs=15.0, k_contact=1200.0, a_max=300.0,
            cavity_r=0.095, cavity_h=0.095, axis=2, n_fibres=6000, fibre_len=0.16,
            align=0.85, align_dir=(1.0, 0.0, 0.0), substep_dt=2.0e-4,
            # A DENSER MATRIX AT THE POLES, on top of the alignment. Alignment alone gave a 1.50x
            # directional pressure difference -- real, and small, because MPM averages a fibrous
            # ARRANGEMENT of an isotropic material into a nearly isotropic continuum. Redistributing
            # the fibres so there are 3x as many within 40 degrees of z changes the mass and stiffness
            # the grid actually sees, which is not something the continuum can average away. Dense at
            # the poles means growth should be suppressed along z: an OBLATE spheroid, produced by
            # mechanosensitive proliferation rather than by a plate -- which is the comparison worth
            # having, since `42`/`43` make the same shape the other way.
            # 55 DEGREES, NOT 40, AND THE NUMBER IS MEASURED. `gate_geom.py` sweeps the suppressed
            # solid angle against a synthetic polar pressure map at 90 seconds a point and finds a
            # non-monotonic optimum, with both ends failing:
            #
            #     suppressed frac of 4pi   flat   0.23   0.43   0.66   0.91
            #     oblateness eq/z          1.015  1.435  1.535  1.374  1.063
            #
            # Two caps of half-angle 55 deg cover 0.43 of 4.pi. Narrower and only a patch lags -- a
            # dimple; wider and nearly every direction is suppressed, so the tissue grows smaller
            # ISOTROPICALLY and the aspect returns to 1. The tissue needs a free equatorial belt to
            # grow INTO, which is why more suppression is not more shape.
            dense_axis=2, dense_cone_deg=55.0, dense_boost=3.0,
            stress_scale=0.008, stress_measure="vonmises")


def pmap_anisotropy(P):
    """Mean pressure in the +/-x, +/-y and +/-z direction cones of an equirectangular map.

    Cones rather than single bins: one bin at the pole is a sliver and one on the equator is not, so
    comparing bins would compare solid angles. Each cone is the set of directions within ~30 degrees of
    an axis, weighted by the bin's own solid angle.
    """
    T, nth, nph = P.shape
    th = (np.arange(nth) + 0.5) / nth * np.pi
    ph = (np.arange(nph) + 0.5) / nph * 2 * np.pi
    st, ct = np.sin(th)[:, None], np.cos(th)[:, None]
    ux = st * np.cos(ph)[None, :]
    uy = st * np.sin(ph)[None, :]
    uz = np.broadcast_to(ct, (nth, nph))
    w = np.broadcast_to(np.sin(th)[:, None], (nth, nph))
    out = {}
    for name, u in (("x", ux), ("y", uy), ("z", uz)):
        m = np.abs(u) > np.cos(np.deg2rad(30.0))
        ww = (w * m)[None]
        out[name] = float((P * ww).sum() / max(ww.sum() * T, 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--movie-frames", type=int, default=90)
    ap.add_argument("--p-half", type=float, default=0.10)
    ap.add_argument("--floor", type=float, default=0.25)
    ap.add_argument("--step0", default="49_aniso_i0_fibres")
    ap.add_argument("--step2", default="50_aniso_growth")
    ap.add_argument("--skip-step0", action="store_true")
    a = ap.parse_args()

    import combine as C
    import run_ecm as R
    import tissue as TIS

    rep, t0 = {}, time.time()

    # ---- 1-2: the aligned matrix, loaded by the ungated tissue -> pressure by direction -----------
    npz0 = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X)
    # THE CONTROL HAS TO CARRY THE SAME METRICS AS THE TEST. The ungated cache predates `r_xyz`, and a
    # comparison that cannot read the control's semi-axes is not a comparison -- 90 seconds to rebuild.
    if "r_xyz" not in np.load(npz0).files:
        print("[aniso] control cache predates per-axis semi-axes -- rebuilding it", flush=True)
        npz0 = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X,
                                 rebuild=True)
    load = os.path.join(R.LOG, a.step0, "load.npz")
    if not (a.skip_step0 and os.path.exists(load)):
        spec, info = C.build(a.step0, npz0, **dict(BASE))
        d = os.path.join(R.LOG, a.step0); os.makedirs(d, exist_ok=True)
        info["varied"] = {"align": BASE["align"], "align_dir": list(BASE["align_dir"]),
                          "dense_boost": BASE["dense_boost"],
                          "dense_cone_deg": BASE["dense_cone_deg"], "iteration": 0}
        json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
        print(f"\n[aniso] STEP 1-2: {a.step0} -- fibres aligned along x, isotropic cavity", flush=True)
        m0 = R.run(a.step0, spec, device=a.device, movie=True,
                   render_kw={"movie_frames": a.movie_frames})
        rep["step0"] = {k: m0.get(k) for k in ("contact_frame", "strained_frac_end", "max_disp",
                                              "exploded")}
    if not os.path.exists(load):
        raise RuntimeError(f"no {load}: step 0 recorded no contact pressure")
    P = np.asarray(np.load(load)["pmap"], float)
    an = pmap_anisotropy(P)
    lo = min(an.values()); hi = max(an.values())
    rep["pressure_by_axis"] = an
    rep["pressure_anisotropy"] = hi / max(lo, 1e-12)
    print(f"[aniso] LINK 2 -- mean contact pressure by axis: "
          + "  ".join(f"{k} {v:.4g}" for k, v in an.items())
          + f"   max/min = {hi / max(lo, 1e-12):.3f}", flush=True)
    if hi / max(lo, 1e-12) < 1.05:
        print("[aniso] LINK 2 IS NULL: the aligned fibres produced no directional pressure "
              "difference, so nothing downstream can be attributed to them. Reporting it rather "
              "than gating on it -- the gate would still make a shape, from noise.", flush=True)

    # ---- 3-4: the tissue, with its cell cycle gated by that pressure ------------------------------
    print(f"\n[aniso] STEP 3-4: tissue with the growth gate (p_half {a.p_half}, hill {a.hill}, "
          f"floor {a.floor}, smoothing {a.smooth_frames} frames / {a.smooth_phi:g} deg)", flush=True)
    npz1 = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X,
                             gate_npz=load, gate_p_half=a.p_half, gate_hill=a.hill,
                             gate_floor=a.floor, gate_smooth_frames=a.smooth_frames,
                             gate_smooth_phi=a.smooth_phi,
                             tag_extra=f"_g{a.p_half}h{a.hill:g}f{a.floor:g}".replace(".", "p"))
    z0, z1 = np.load(npz0), np.load(npz1)
    for tag, z in (("ungated", z0), ("gated", z1)):
        x, y, zz = np.asarray(z["r_xyz"])[-1]
        eq = float(np.sqrt(max(x * x + y * y, 1e-12) / 2.0))
        rep[tag] = {"r_x": float(x), "r_y": float(y), "r_z": float(zz),
                    "n_cells": int(z["n_cells"][-1]), "r_eq_rms": eq,
                    "oblateness_eq_over_z": float(eq / max(zz, 1e-9)),
                    "xy": float(x / max(y, 1e-9)), "xz": float(x / max(zz, 1e-9))}
        print(f"[aniso]   {tag:8} semi-axes x/y/z = {x:.2f} / {y:.2f} / {zz:.2f}   "
              f"x:y = {x / max(y, 1e-9):.3f}   x:z = {x / max(zz, 1e-9):.3f}   "
              f"{int(z['n_cells'][-1])} cells", flush=True)
    rep["oblateness_gain"] = (rep["gated"]["oblateness_eq_over_z"]
                              / max(rep["ungated"]["oblateness_eq_over_z"], 1e-12))
    print(f"[aniso] LINK 4 -- OBLATENESS eq/z {rep['ungated']['oblateness_eq_over_z']:.3f} "
          f"-> {rep['gated']['oblateness_eq_over_z']:.3f} "
          f"(x{rep['oblateness_gain']:.3f}), cells {rep['ungated']['n_cells']} -> "
          f"{rep['gated']['n_cells']}", flush=True)

    # ---- the artefact: the matrix against the tissue its own stress shaped ------------------------
    spec, info = C.build(a.step2, npz1, **dict(BASE))
    d = os.path.join(R.LOG, a.step2); os.makedirs(d, exist_ok=True)
    info["varied"] = {"align": BASE["align"], "gate_p_half": a.p_half, "gate_floor": a.floor,
                      "iteration": 1}
    info["aniso"] = rep
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    print(f"\n[aniso] ARTEFACT: {a.step2}", flush=True)
    m1 = R.run(a.step2, spec, device=a.device, movie=True,
               render_kw={"movie_frames": a.movie_frames})
    rep["step2"] = {k: m1.get(k) for k in ("contact_frame", "strained_frac_end", "max_disp",
                                          "exploded")}
    json.dump(rep, open(os.path.join(R.LOG, "aniso.json"), "w"), indent=1)
    print(f"\n[aniso] done in {(time.time() - t0) / 60:.0f} min -> log/okuda_ECM/aniso.json")


if __name__ == "__main__":
    main()
