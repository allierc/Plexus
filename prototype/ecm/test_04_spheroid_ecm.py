#!/usr/bin/env python
"""test_04 -- the spheroid pushing the matrix, through the interface 03 licensed.

    python test_04_spheroid_ecm.py [--device cuda:0] [--frames 199]   ->  log/okuda_ECM/04_spheroid_ecm/

WHAT THIS COMBINES, AND WHY EACH PIECE IS THE ONE IT IS.

  * the tissue is `01c`'s: two-pool myosin with a cytokinetic ring at `ring = 1`, the configuration
    whose newborn junction sits at 0.92 of the tissue mean instead of 0.63 and whose starting
    density no longer drifts with the frame index. It is loaded from the cache, not re-run: pass 1
    is finished before pass 2 begins.
  * the matrix is `02h`'s: 20 particles per strand rather than 60, eight substeps rather than
    sixteen, `drag = 8` so it does not ring, `store_stress` + `measure: vonmises` so the readout is
    the stress and not the volume change. Those four numbers are 5.9x cheaper than the control they
    reproduce to 0.14%.
  * the contact is `03`'s, generalised to a curved moving surface in `mesh_contact_ops.py`, and
    placed INSIDE the substep block -- at frame level a penalty's stability ceiling is
    (dt_frame/dt_sub)^2 = 64 times lower here, and the flat rig measured what happens when that is
    ignored.

WHAT IT REPLACES. `cell_to_ecm[replay]` pushes along the radius with a force from a SMOOTHED,
32x64 radius map, and `cell_exclude_3d` then projects out whatever the penalty failed to keep out.
Both go. The first cannot return its reaction to anything (a bin of a map has no vertices) and
cannot generate a shear (a radial force has no tangential component); the second makes the question
"does the contact hold the matrix out" unanswerable, because the answer is zero by construction.
Here the reaction is computed on named vertices and the non-penetration count is a MEASUREMENT.

THE FIVE THINGS IT MEASURES, each of which can come back wrong:
  1. momentum      sum of the contact forces on the particles and on the tissue's vertices, over
                   their own magnitude. 03 got float32 machine precision on a flat patch; a curved
                   mesh with a virtual centroid per face is where that bookkeeping can break.
  2. penetration   how many particles are behind the surface at the end of a frame and how deep,
                   with NO backstop operator to hide it.
  3. the far field the radial displacement profile u(r) and its exponent. An incompressible linear
                   continuum gives u ~ r^-2; a fibrous matrix gives r^-0.5 to r^-1 (Wang et al.
                   2014), and that single exponent is what the whole spheroid-matrix coupling rests
                   on. This substrate should give the CONTINUUM answer -- the strands are an
                   arrangement of the mass, as note_fibre measured -- so r^-2 is the prediction and
                   anything else is a defect in this rig, not a fibre effect.
  4. compaction    the matrix's own density against radius: a spheroid tripling its radius has to
                   put its volume somewhere.
  5. the reaction  the pressure the tissue would have felt, by direction, which is what a two-way
                   pass feeds to `ecm_growth_gate_3d`.

WHAT IT IS NOT. The coupling is still one-way: the tissue is a replay and does not feel the
reaction, so this run shows how a growing epithelium loads a matrix and not how the matrix shapes
the epithelium. And the matrix is a SHELL with a free outer surface at 2.7 final radii, because
filling the box at a usable particle density costs a million particles; the far-field exponent is
therefore fitted inside that shell and compared against the same geometry's closed form, not
against the infinite-medium one.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.animation import FFMpegWriter                        # noqa: E402
from matplotlib.collections import LineCollection                    # noqa: E402
from matplotlib.colors import ListedColormap                         # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import ecm_ops                                                       # noqa: E402,F401
import ecm_spec as ES                                                # noqa: E402
import mesh_contact_ops as MC                                        # noqa: E402
import tissue as TIS                                                 # noqa: E402

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
CMAP = ListedColormap(ES.STRESS_COLORS)
TISSUE_C = "#e8dcc0"

# THE SPHEROID'S FINAL RADIUS IN THE BOX, and everything geometric follows from it. Larger and the
# matrix shell has no far field to fit an exponent in; smaller and the tissue is a handful of grid
# cells across. 0.15 puts the final surface at 9.6 cells of radius with 2.7 more radii of matrix
# outside it.
R_FINAL_BOX = 0.15
CENTRE = [0.5, 0.5, 0.5]


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


# ------------------------------------------------------------------ the spec
def build(name, frames, tissue_npz, scale, n_particles=200000, n_fibres=10000, fibre_len=0.12,
          n_grid=64, dt=3.2e-3, sub=4.0e-4, youngs=15.0, drag=8.0, shell_r=0.40,
          cavity_r=0.045, k_frac=0.15, mu=0.4, stress_scale=2.0, seed=0, mesh_stride=1):
    """Sets, one field, operators, schedule -- as a plain dict, the same shape `test_02` uses.

    `youngs = 15` is the stroma of the specification note_fibre section 9 proposes, and under
    one-way coupling it is very nearly a free parameter: the tissue never feels the matrix, so a
    softer stroma changes the substep it needs and not the shape that comes out. The moment an
    operator writes back to the tissue this stops being true and the number has to be argued for.
    """
    types = {f"s{i}": {"fraction": 1.0 / len(ES.STRESS_COLORS), "youngs": float(youngs)}
             for i in range(len(ES.STRESS_COLORS))}
    return {
        "general": {"name": name, "seed": int(seed), "n_frames": int(frames), "dt": float(dt),
                    "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0]},
        "sets": {
            "cell": {"n": 1, "start": [list(CENTRE)], "types": types},
            "mpm_particle": {"parent": "cell", "per_parent": int(n_particles), "radius": 0.48,
                             "density": 1.0, "types": types},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},
        "operators": [
            {"op": "aggregate", "at": "cell"},
            # THE CAVITY IS THE SPHEROID AT FRAME 0, plus a skin. A strand seeded where the tissue
            # already is would be in contact before the clock starts, and "the moment of first
            # contact" is an event this run reports.
            {"op": "seed_ecm", "at": "mpm_particle", "centre": list(CENTRE),
             "cavity_r": float(cavity_r), "cavity_h": float(cavity_r), "cavity_sphere": True,
             "axis": 1, "shell_r": float(shell_r), "margin": 0.02,
             "n_fibres": int(n_fibres), "fibre_len": float(fibre_len), "align": 0.0,
             "align_dir": [1.0, 0.0, 0.0], "seed": int(seed)},
            {"op": "ecm_stress", "at": "mpm_particle", "scale": float(stress_scale),
             "bands": len(ES.STRESS_COLORS), "measure": "vonmises"},
            {"op": "mesh_inside_count", "at": "mpm_particle", "n_grid": int(n_grid)},
            # `mesh_stride` IS PASS-2 FRAMES PER KEPT MESH. The cache holds 200 meshes for 402
            # tissue frames, so a 400-frame pass 2 runs at stride 2 and the operator interpolates
            # the surface between kept meshes -- the tissue's clock is the cache's either way, and
            # what the stride buys is temporal resolution on the CONTACT, not on the tissue.
            {"op": "mesh_contact", "at": "mpm_particle", "tissue": str(tissue_npz),
             "centre": list(CENTRE), "scale": float(scale), "k_frac": float(k_frac),
             "mu": float(mu), "dt": float(dt), "n_grid": int(n_grid), "a_max": 3.0e4,
             "mesh_stride": int(mesh_stride)},
            {"op": "mpm_strain", "at": "mpm_particle"},
            # `a_max` ABOVE THE CONTACT'S OWN CLAMP. The scatter's clamp acts after `mesh_contact`
            # has already recorded its reaction, so a contact clipped here would be recorded at a
            # value that never acted and the momentum test would pass on a fiction.
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": float(drag),
             "a_max": 1.0e5, "store_stress": True},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.7},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 0.7,
             "wall_contact": 0.04, "vmax": 1.0e9},
        ],
        # `mesh_contact` INSIDE THE BLOCK; the two diagnostics AFTER it. Inside, because at frame
        # level a penalty's stability ceiling is (dt/dt_sub)^2 = 64 times lower here. After, because
        # both read what the frame produced: `ecm_stress` reads the Cauchy stress the last substep
        # cached, and `mesh_inside_count` has to test this frame's positions against THIS frame's
        # mesh -- scheduled before the block it would test them against the previous frame's, which
        # is a non-penetration count for a surface that has since moved.
        "schedule": ["aggregate", "seed_ecm",
                     {"substep_dt": float(sub),
                      "steps": ["mesh_contact", "mpm_strain", "mpm_scatter", "mpm_grid_update",
                                "mpm_gather"]},
                     "ecm_stress", "mesh_inside_count"],
        "plotting": {"background": "black", "up_axis": 1, "box_frame": True},
    }


# ------------------------------------------------------------------ what it measures
def _lame(r, b, nu=0.2):
    """The radial displacement of an isotropic linear-elastic SPHERICAL SHELL whose inner surface is
    driven and whose outer surface at `b` is free: u(r) = C1 r + C2/r^2, with the free-surface
    condition sigma_rr(b) = 0 fixing C1/C2 = 2(1-2nu)/((1+nu) b^3).

    THIS IS THE CONTROL, AND IT IS NOT r^-2. The infinite-medium answer for an incompressible
    continuum is u ~ r^-2, and a fibrous matrix gives r^-0.5 to r^-1 (Wang et al. 2014) -- but this
    rig's matrix is a SHELL with a free outer surface at 2.7 tissue radii, and a free surface adds
    the C1 r term, which bends the profile up and makes the local slope shallower than -2 with no
    fibre mechanics involved whatsoever. Fitting the two-parameter form and reporting how well it
    fits is therefore a much stronger test than an exponent: it asks whether the matrix is an
    isotropic elastic shell, which -- the strands being an arrangement of the mass rather than a
    mechanism -- is exactly what it should be.
    """
    c1_over_c2 = 2 * (1 - 2 * nu) / ((1 + nu) * b ** 3)
    return c1_over_c2 * r, 1.0 / r ** 2             # the two basis functions, C2 factored out


def _lame_fit(r, u, b, nu=0.2):
    """Least squares of u against the shell's own two-term form. Returns (C2, R^2, local slope)."""
    g1, g2 = _lame(r, b, nu)
    A = (g1 + g2)[:, None]                          # one free amplitude: C1 is tied to C2 above
    c, *_ = np.linalg.lstsq(A, u, rcond=None)
    pred = (A @ c).reshape(-1)
    ss = 1.0 - np.sum((u - pred) ** 2) / max(np.sum((u - u.mean()) ** 2), 1e-30)
    du = c[0] * (_lame(r, b, nu)[0] / r - 2.0 / r ** 3)
    return float(c[0]), float(ss), du * r / np.clip(pred, 1e-30, None)


def measure(P, vm, per, scale, r_tissue, out_dir, dx):
    """Every number the note quotes, from the trajectory alone."""
    T, N, D = P.shape
    c = np.asarray(CENTRE, np.float32)
    r0 = np.linalg.norm(P[0] - c, axis=1)
    m = dict(frames=int(T), particles=int(N), per_strand=int(per), dx=float(dx),
             r_tissue=[float(v) for v in r_tissue])

    # ---- displacement against radius, and its exponent ----------------------------------
    # Binned by the particle's OWN INITIAL radius, so a bin is a material shell and not a region of
    # space: with the matrix moving outward, a spatial bin at late time holds material that started
    # somewhere else and its mean displacement is a mixture.
    edges = np.linspace(0.05, 0.40, 22)
    ib = np.clip(np.digitize(r0, edges) - 1, 0, len(edges) - 2)
    rb = 0.5 * (edges[1:] + edges[:-1])
    prof, noise, expo, nfit, lam_r2, lam_slope, snr = [], [], [], [], [], [], []
    d0 = P[0] - c
    u_hat = d0 / np.clip(r0, 1e-9, None)[:, None]
    for t in (0, T // 4, T // 2, 3 * T // 4, T - 1):
        dv = P[t] - P[0]
        ur = (dv * u_hat).sum(1)
        # THE FLOOR IS MEASURED, NOT ASSUMED, AND IT IS NOT ALL NOISE. The loading is spherically
        # symmetric, so a bin's NON-RADIAL displacement is everything the symmetry does not
        # predict -- float32 and seeding disorder in the far field, and the tissue's own bumpiness
        # near it, which is real. Either way it is the right floor for the radial number in the same
        # bin: without it the fit runs out into bins where the signal is 2e-5 box units and returns
        # an exponent that is a fit to rounding (measured: -5.2 on the 40-frame smoke run).
        ut = np.linalg.norm(dv - ur[:, None] * u_hat, axis=1)
        pr = np.array([ur[ib == k].mean() if (ib == k).sum() > 30 else np.nan
                       for k in range(len(rb))])
        nz = np.array([np.sqrt((ut[ib == k] ** 2).mean()) if (ib == k).sum() > 30 else np.nan
                       for k in range(len(rb))])
        prof.append([float(v) for v in pr]); noise.append([float(v) for v in nz])
        # THE FIT IS OVER THE FAR FIELD ONLY -- outside 1.5 tissue radii, inside the outer surface
        # whose own free boundary bends the profile up, and above three times the local noise. All
        # three cuts are stated, and the count of bins that survived them is reported beside the
        # exponent so a fit over four points cannot be read as a fit over twenty.
        # THE WINDOW IS GEOMETRIC AND THE NOISE IS REPORTED BESIDE IT, rather than used to select
        # the bins. Cutting at `pr > 3*nz` worked at 200 frames and left one bin at 400: the same
        # tissue growth spread over twice the matrix's own clock gives the far field twice as long
        # to rearrange, so the non-radial displacement grows while the radial signal does not, and a
        # noise-gated fit silently became a fit to two points. The gate is now a NUMBER --
        # `disp_snr`, the median of radial over non-radial in the window -- so a fit with little
        # signal is reported as one instead of disappearing.
        lo, hi = 1.5 * r_tissue[t], 0.34
        ok = np.isfinite(pr) & np.isfinite(nz) & (rb > lo) & (rb < hi) & (pr > 0)
        nfit.append(int(ok.sum()))
        snr.append(float(np.median(pr[ok] / np.clip(nz[ok], 1e-12, None))) if ok.any()
                   else float("nan"))
        if ok.sum() > 4:
            expo.append(float(np.polyfit(np.log(rb[ok]), np.log(pr[ok]), 1)[0]))
            _, r2, sl = _lame_fit(rb[ok], pr[ok], 0.40)
            lam_r2.append(r2)
            lam_slope.append(float(np.mean(sl)))
        else:
            expo.append(float("nan")); lam_r2.append(float("nan"))
            lam_slope.append(float("nan"))
    m["disp_r"] = [float(v) for v in rb]
    m["disp_profile"] = prof
    m["disp_noise"] = noise
    m["disp_frames"] = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    m["disp_exponent"] = expo
    m["disp_fit_bins"] = nfit
    m["disp_snr"] = snr
    m["lame_r2"] = lam_r2
    m["lame_slope"] = lam_slope
    m["lame_exponent"] = m["lame_slope"][-1]

    # ---- compaction: density against radius ---------------------------------------------
    dens = []
    for t in (0, T - 1):
        rr = np.linalg.norm(P[t] - c, axis=1)
        h, _ = np.histogram(rr, bins=edges)
        shell = (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        dens.append([float(v) for v in h / shell])
    m["density_r"] = dens
    m["density_ratio"] = [float(a / b) if b > 0 else float("nan")
                          for a, b in zip(dens[1], dens[0])]
    # THE PEAK IS TAKEN OUTSIDE THE TISSUE AND INSIDE THE FREE SURFACE. A bin the tissue now
    # occupies has no matrix left in it and reads 0; the outermost bin gains material that moved
    # outward into what was empty and reads 2.8. Neither is compaction, and quoting either as the
    # headline would be quoting the geometry of the shell.
    okd = (rb > r_tissue[-1]) & (rb < 0.34)
    m["density_peak"] = float(np.nanmax(np.asarray(m["density_ratio"])[okd]))
    m["density_peak_r"] = float(rb[okd][np.nanargmax(np.asarray(m["density_ratio"])[okd])])

    # ---- the fibres: stretch, bow, and how they lie -------------------------------------
    nf = N // per
    Fb = P[:, : nf * per].reshape(T, nf, per, D)
    e2e_v = Fb[:, :, -1] - Fb[:, :, 0]
    e2e = np.linalg.norm(e2e_v, axis=-1)
    lam = e2e / np.clip(e2e[0], 1e-9, None)
    u_ax = e2e_v / np.clip(e2e, 1e-9, None)[..., None]
    rel = Fb - Fb[:, :, :1]
    perp = rel - (rel * u_ax[:, :, None, :]).sum(-1)[..., None] * u_ax[:, :, None, :]
    bow = np.sqrt((perp ** 2).sum(-1).mean(-1))
    # AND WHETHER THEY TURN. cos^2 between a strand's own axis and the direction from the tissue
    # centre to its midpoint: 1 is radial, 0 is circumferential, 1/3 is isotropic.
    mid = Fb.mean(2) - c
    u_rad = mid / np.clip(np.linalg.norm(mid, axis=-1), 1e-9, None)[..., None]
    cos2 = ((u_ax * u_rad).sum(-1)) ** 2
    r_mid = np.linalg.norm(mid, axis=-1)
    inner = r_mid[0] < 2.0 * r_tissue[-1]
    # THE CONTROL FOR IT, AND WITHOUT THIS THE NUMBER MEANS NOTHING. A matrix pushed outward turns
    # its own material lines circumferential whether or not it is fibrous -- the map r -> r + u(r)
    # stretches the two tangential directions and compresses the radial one, and a line embedded in
    # it rotates accordingly. So the frame-0 strands are carried through the run's OWN measured
    # radial displacement profile, spherically symmetric by construction, and their cos^2 is
    # computed the same way. What is left over after that is the only part a fibre mechanism could
    # claim -- and this material's law reads the strand direction nowhere, so the prediction is that
    # nothing is left over.
    rb_i = np.interp(np.linalg.norm(P[0] - c, axis=1), rb[np.isfinite(prof[-1])],
                     np.asarray(prof[-1])[np.isfinite(prof[-1])], left=np.nan, right=0.0)
    Paff = (P[0] - c) * (1.0 + rb_i[:, None] / np.clip(r0, 1e-9, None)[:, None]) + c
    Ab = Paff[: nf * per].reshape(nf, per, D)
    av = Ab[:, -1] - Ab[:, 0]
    au = av / np.clip(np.linalg.norm(av, axis=-1), 1e-9, None)[..., None]
    amid = Ab.mean(1) - c
    aur = amid / np.clip(np.linalg.norm(amid, axis=-1), 1e-9, None)[..., None]
    cos2_affine = ((au * aur).sum(-1)) ** 2
    good = np.isfinite(cos2_affine)
    # AND IT IS READ AGAINST ITS OWN FRAME 0, NOT AGAINST 1/3. `seed_ecm` pushes any particle that
    # lands in the cavity out through its rim, which leaves the strands nearest the tissue biased
    # circumferential before anything has moved: measured, the inner shell starts at 0.276 and not
    # at the 0.333 of an isotropic seeding. Quoting 1/3 as the baseline would read that bias as an
    # effect of the loading.
    m.update(lam_mean=lam.mean(1).tolist(), lam_p95=np.percentile(lam, 95, axis=1).tolist(),
             bow_mean=bow.mean(1).tolist(), bow0=float(bow[0].mean()),
             e2e0=float(e2e[0].mean()), n_fibres=int(nf),
             cos2_all=cos2.mean(1).tolist(),
             cos2_inner=cos2[:, inner].mean(1).tolist(),
             cos2_all_0=float(cos2[0].mean()), cos2_inner_0=float(cos2[0, inner].mean()),
             cos2_affine=float(cos2_affine[good].mean()),
             cos2_affine_inner=float(cos2_affine[good & inner].mean()),
             cos2_measured_matched=float(cos2[-1][good].mean()),
             cos2_measured_matched_inner=float(cos2[-1][good & inner].mean()))
    # AND EVERY FIBRE NUMBER AGAINST THE RADIUS IT STARTED AT. The far field holds most of the
    # strands (volume goes as r^3), so a tissue-wide mean is the far field's answer wearing the
    # whole matrix's name: measured, the mean end-to-end stretch is 1.013 while the shell nearest
    # the tissue is doing something several times larger.
    be = np.array([0.045, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22, 0.28, 0.36])
    kb = np.clip(np.digitize(r_mid[0], be) - 1, 0, len(be) - 2)
    m["fibre_r"] = [float(v) for v in 0.5 * (be[1:] + be[:-1])]
    for key, arr in (("lam", lam[-1]), ("bow", bow[-1] / np.clip(bow[0], 1e-12, None)),
                     ("cos2", cos2[-1]), ("cos2_0", cos2[0]), ("cos2_aff", cos2_affine)):
        m[f"fibre_{key}_r"] = [float(np.nanmean(arr[(kb == q) & np.isfinite(arr)]))
                               if ((kb == q) & np.isfinite(arr)).sum() > 20 else float("nan")
                               for q in range(len(be) - 1)]

    # ---- the stress, against radius and time --------------------------------------------
    if vm is not None:
        m["vm_mean"] = [float(np.mean(a)) for a in vm]
        m["vm_p99"] = [float(np.percentile(a, 99)) for a in vm]
        prof = []
        for t in (0, T // 4, T // 2, 3 * T // 4, T - 1):
            rr = np.linalg.norm(P[t] - c, axis=1)
            v = np.asarray(vm[t], np.float32)
            prof.append([float(np.mean(v[(rr >= a) & (rr < b)]))
                         if ((rr >= a) & (rr < b)).sum() > 30 else float("nan")
                         for a, b in zip(edges[:-1], edges[1:])])
        m["vm_profile"] = prof

    # ---- the interface's own record ------------------------------------------------------
    if MC.CONTACT_HISTORY:
        h = MC.CONTACT_HISTORY
        m["contact"] = dict(
            momentum_residual_max=float(max(x["momentum_residual"] for x in h)),
            momentum_residual_med=float(np.median([x["momentum_residual"] for x in h])),
            penetration_max_cells=float(max(x["depth_max"] for x in h) / dx),
            contacts_max=int(max(x["n_contact"] for x in h)),
            a_max_seen=float(max(x["a_max"] for x in h)),
            slip_mean=float(np.mean([x["slip"] for x in h if x["n_contact"] > 0] or [0.0])),
            first_contact=int(next((x["frame"] for x in h if x["n_contact"] > 0), -1)),
            series={k: [float(x[k]) for x in h]
                    for k in ("n_contact", "depth_max", "momentum_residual", "slip", "n_tri")})
    if MC.INSIDE_HISTORY:
        h = MC.INSIDE_HISTORY
        m["inside"] = dict(max_count=int(max(x["n_inside"] for x in h)),
                           last_count=int(h[-1]["n_inside"]),
                           max_depth_cells=float(max(x["depth_max_cells"] for x in h)),
                           series=[int(x["n_inside"]) for x in h])
        # AND THE ONE COMPARISON THAT SAYS WHAT THOSE PARTICLES ARE. A penalty holds a particle at
        # the depth where its push balances the surface's advance, so a contact layer one
        # penetration-depth thick is not a leak and its count should track the number of particles
        # IN CONTACT, not grow relative to it. If this ratio drifts upward the matrix is being
        # swallowed; if it sits near 1 the "particles behind the surface" are the contact itself.
        if "contact" in m:
            n_c = m["contact"]["series"]["n_contact"]
            k = min(len(n_c), len(h))
            m["inside"]["over_contact"] = [float(h[i]["n_inside"] / max(n_c[i], 1))
                                           for i in range(k)]
            m["inside"]["over_contact_last"] = m["inside"]["over_contact"][-1]
    if MC.PRESSURE_MAP:
        Pm = np.stack([p for p in MC.PRESSURE_MAP if p is not None])
        m["pressure"] = dict(mean=[float(np.mean(p)) for p in Pm],
                             p95=[float(np.percentile(p, 95)) for p in Pm],
                             anisotropy=[float(np.percentile(p, 95) / max(np.mean(p), 1e-12))
                                         for p in Pm])
        np.savez_compressed(os.path.join(out_dir, "pressure.npz"), pressure=Pm.astype(np.float32))
    return m



def _panel(ax, letter):
    """A bold letter top-left and no title. The numbers a title used to carry go into the note's
    caption, where they can be read against the gate they belong to; a title repeats them in a place
    the figure cannot explain them."""
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def plot(m, out):
    fig, ax = plt.subplots(2, 4, figsize=(17.0, 6.6), facecolor="white")
    c = m.get("contact", {})
    s = c.get("series", {})
    if s:
        ax[0, 0].semilogy(np.maximum(s["momentum_residual"], 1e-18), color="#2b6cb0", lw=1.1)
        ax[0, 0].set_ylabel(r"$|\sum f_{\rm part}+\sum f_{\rm vert}|\,/\,\sum|f|$")
        _panel(ax[0, 0], "a")
        ax[0, 1].plot(np.asarray(s["depth_max"]) / m["dx"], color="#e0452b", lw=1.1)
        a1 = ax[0, 1].twinx()
        a1.plot(s["n_contact"], color="#e08a2e", lw=1.0)
        a1.set_ylabel("particles in contact", color="#e08a2e")
        ax[0, 1].set_ylabel("max penetration (grid cells)", color="#e0452b")
        _panel(ax[0, 1], "b")
    if "inside" in m:
        ax[0, 2].plot(m["inside"]["series"], color="#1f8a5c", lw=1.2, label="behind the surface")
        if s:
            ax[0, 2].plot(s["n_contact"], color="#e08a2e", lw=1.0, ls="--", label="in contact")
        ax[0, 2].set_ylabel("particles")
        ax[0, 2].legend(fontsize=7, frameon=False)
        _panel(ax[0, 2], "c")
    rb = np.asarray(m["disp_r"])
    if "vm_profile" in m:
        for k, (fr, pr) in enumerate(zip(m["disp_frames"], m["vm_profile"])):
            ax[0, 3].loglog(rb, np.maximum(pr, 1e-6), lw=1.2,
                            color=plt.cm.inferno(0.15 + 0.7 * k
                                                 / max(len(m["disp_frames"]) - 1, 1)),
                            label=f"frame {fr}")
        ax[0, 3].set_xlabel("radius (box units)")
        ax[0, 3].set_ylabel("von Mises stress")
        _panel(ax[0, 3], "d")
        ax[0, 3].legend(fontsize=6.5, frameon=False)

    for k, (fr, pr) in enumerate(zip(m["disp_frames"], m["disp_profile"])):
        ax[1, 0].loglog(rb, np.maximum(pr, 1e-9), lw=1.2,
                        color=plt.cm.viridis(k / max(len(m["disp_frames"]) - 1, 1)),
                        label=f"frame {fr}")
    ax[1, 0].loglog(rb, np.maximum(m["disp_noise"][-1], 1e-12), lw=0.9, ls=":", color="#999",
                    label="non-radial")
    ax[1, 0].set_xlabel("initial radius (box units)")
    ax[1, 0].set_ylabel("radial displacement")
    _panel(ax[1, 0], "e")
    ax[1, 0].legend(fontsize=6.5, frameon=False)
    ax[1, 1].plot(rb, m["density_ratio"], color="#7b4fb5", lw=1.4)
    ax[1, 1].axhline(1.0, color="#999", ls="--", lw=0.8)
    ax[1, 1].axvline(m["r_tissue"][-1], color="#c8a94e", lw=1.0)
    ax[1, 1].set_xlabel("radius (box units)")
    ax[1, 1].set_ylabel("density, last frame / first")
    _panel(ax[1, 1], "f")
    fr = np.asarray(m["fibre_r"])
    ax[1, 2].plot(fr, m["fibre_lam_r"], color="#2b6cb0", lw=1.4, marker="o", ms=3,
                  label="end-to-end stretch")
    ax[1, 2].plot(fr, m["fibre_bow_r"], color="#e0452b", lw=1.4, marker="s", ms=3,
                  label="bow, over its own frame 0")
    ax[1, 2].axhline(1.0, color="#999", ls="--", lw=0.8)
    ax[1, 2].axvline(m["r_tissue"][-1], color="#c8a94e", lw=1.0)
    ax[1, 2].set_xlabel("the radius the strand started at")
    _panel(ax[1, 2], "g")
    ax[1, 2].legend(fontsize=7, frameon=False)
    ax[1, 3].plot(fr, m["fibre_cos2_0_r"], color="#999", lw=1.2, ls="--", label="as seeded")
    ax[1, 3].plot(fr, m["fibre_cos2_aff_r"], color="#1f8a5c", lw=1.6, ls=":",
                  label="affine control")
    ax[1, 3].plot(fr, m["fibre_cos2_r"], color="#e0452b", lw=1.4, marker="o", ms=3,
                  label="measured")
    ax[1, 3].axvline(m["r_tissue"][-1], color="#c8a94e", lw=1.0)
    ax[1, 3].set_xlabel("the radius the strand started at")
    ax[1, 3].set_ylabel(r"$\langle\cos^2\theta\rangle$ to the radius")
    _panel(ax[1, 3], "h")
    ax[1, 3].legend(fontsize=7, frameon=False)
    for a in ax.reshape(-1):
        a.set_xlabel(a.get_xlabel() or "frame")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ the movie
def _strands(P_t, band_t, per, keep, max_lines=1200):
    """Fibre segments as a LineCollection-ready list, one polyline per strand."""
    nf = min(len(keep), max_lines)
    seg, col = [], []
    for k in keep[:nf]:
        s = slice(k * per, (k + 1) * per)
        seg.append(P_t[s])
        col.append(int(np.median(band_t[s])))
    return seg, np.asarray(col)


def render(P, band, tissue_npz, scale, d, name, per, frames_drawn=200, fps=20, slab=0.02):
    """Two panels, the same convention as every other artefact here: the cut-away in 3D and the
    section. The tissue is drawn from its OWN mesh at that frame, not from the radius map, because
    the radius map is precisely what this run stopped using."""
    z = np.load(tissue_npz)
    nmesh = len(z["mesh_frames"])
    T, N, _ = P.shape
    ts = np.unique(np.linspace(0, T - 1, min(frames_drawn, T)).astype(int))
    c = np.asarray(CENTRE, np.float32)
    nf = N // per
    rng = np.random.default_rng(0)
    keep = rng.permutation(nf)
    # the cut-away keeps the strands on one side of the box, so the spheroid is not buried
    mid0 = P[0][: nf * per].reshape(nf, per, 3).mean(1)
    keep3d = keep[mid0[keep, 1] < CENTRE[1]]
    # THE SLAB IS THE AXIS THE SECTION DOES NOT DRAW. The first version cut in y and then plotted
    # x against y, so the "section" was a slab seen edge-on -- a flat bar of matrix with the tissue
    # drawn as a circle on top of it, which is a picture of the bug and not of the matrix.
    keep2d = keep[np.abs(mid0[keep, 2] - CENTRE[2]) < slab]

    fig = plt.figure(figsize=(11.6, 5.8), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    strip_at = set(np.round(np.linspace(0, T - 1, 8)).astype(int).tolist())
    strip = []
    lim = 0.46
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for t in ts:
            j = min(nmesh - 1, t)
            V = z[f"m{j}_pos"] * scale + c
            es, et = z[f"m{j}_E_srce"], z[f"m{j}_E_trgt"]
            fig.clf()
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black",
                                 computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            seg, col = _strands(P[t], band[t], per, keep3d)
            for s, cc in zip(seg, col):
                ax.plot(s[:, 0], s[:, 2], s[:, 1], "-", lw=0.6,
                        color=ES.STRESS_COLORS[int(cc) % len(ES.STRESS_COLORS)], alpha=0.85)
            ax.scatter(V[:, 0], V[:, 2], V[:, 1], s=1.2, c=TISSUE_C, marker=".", linewidths=0,
                       depthshade=False, alpha=0.9)
            ax.set_xlim(0.5 - lim, 0.5 + lim); ax.set_ylim(0.5 - lim, 0.5 + lim)
            ax.set_zlim(0.5 - lim, 0.5 + lim)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-60)
            ax.text2D(0.02, 0.97, f"{name}   frame {t}", transform=ax.transAxes, color="white",
                      fontsize=11, va="top")
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            seg2, col2 = _strands(P[t], band[t], per, keep2d, max_lines=4000)
            if seg2:
                a2.add_collection(LineCollection(
                    [s[:, [0, 1]] for s in seg2], linewidths=0.8,
                    colors=[ES.STRESS_COLORS[int(k) % len(ES.STRESS_COLORS)] for k in col2],
                    alpha=0.9))
            mseg = 0.5 * (V[es] + V[et])
            sl = np.abs(mseg[:, 2] - CENTRE[2]) < slab
            a2.add_collection(LineCollection(
                [np.stack([V[a][[0, 1]], V[b][[0, 1]]]) for a, b in zip(es[sl], et[sl])],
                linewidths=0.5, colors=TISSUE_C, alpha=0.8))
            a2.set_xlim(0.5 - lim, 0.5 + lim); a2.set_ylim(0.5 - lim, 0.5 + lim)
            a2.set_aspect("equal"); a2.axis("off")
            a2.text(0.02, 0.98, "section, coloured by von Mises stress", transform=a2.transAxes,
                    color="white", fontsize=11, va="top")
            wri.grab_frame()
            if t in strip_at:
                strip.append((t, [s[:, [0, 1]] for s in seg2], col2.copy(),
                              [np.stack([V[a][[0, 1]], V[b][[0, 1]]])
                               for a, b in zip(es[sl], et[sl])]))
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)

    figs = plt.figure(figsize=(3.0 * len(strip), 3.2), facecolor="black")
    for i, (t, seg2, col2, tseg) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        if seg2:
            a.add_collection(LineCollection(
                seg2, linewidths=0.55,
                colors=[ES.STRESS_COLORS[int(k) % len(ES.STRESS_COLORS)] for k in col2], alpha=0.9))
        a.add_collection(LineCollection(tseg, linewidths=0.4, colors=TISSUE_C, alpha=0.8))
        a.set_xlim(0.5 - lim, 0.5 + lim); a.set_ylim(0.5 - lim, 0.5 + lim)
        a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.96, f"frame {t}", transform=a.transAxes, color="white", fontsize=11,
               va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=110, facecolor="black")
    plt.close(figs)


# ------------------------------------------------------------------ the run
def main():
    import plexus.operators                                          # noqa: F401
    from plexus import schema
    from plexus.engine import run as engine_run

    dev = arg("--device", "cuda:0", str)
    name = arg("--name", "04_spheroid_ecm", str)
    n_part = arg("--particles", 200000, int)
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    # PASS 1, FROM THE CACHE: 01c's tissue, ring = 1, two-pool myosin. Built if missing, which costs
    # the vertex model and nothing else.
    tissue_npz = TIS.load_or_build(
        frames=401, device=dev, buffer_x=4, myosin=1.0, myo_tau=20.0, myo_new=1.0,
        myo_model="two_pool", myo_k_on=0.219, myo_tau_med=20.0, myo_k_ex=0.05, myo_beta_T=0.0,
        myo_ring=1.0, myo_new_rel=True)
    z = np.load(tissue_npz)
    r_ap = z["r_apical"]
    nmesh = len(z["mesh_frames"])
    scale = R_FINAL_BOX / float(r_ap[-1])
    frames = arg("--frames", nmesh - 1, int)
    # PASS-2 FRAMES PER KEPT MESH, derived from the frame count unless it is given: 200 meshes over
    # 400 frames is stride 2. Derived rather than defaulted, so asking for more frames cannot
    # silently run the tissue off the end of its own cache at frame 200.
    stride = arg("--stride", max(1, round((frames + 1) / nmesh)), int)
    # THE TISSUE'S RADIUS PER PASS-2 FRAME, in box units -- every radial measurement below is
    # against this and not against a nominal, and it follows the same frame -> mesh map the
    # operator uses.
    r_tissue = [float(np.median(np.linalg.norm(
        z[f"m{min(f // stride, nmesh - 1)}_pos"], axis=1)) * scale) for f in range(frames + 1)]
    print(f"[04] tissue {os.path.basename(tissue_npz)}: {nmesh} meshes, apical radius "
          f"{r_ap[0]:.2f} -> {r_ap[-1]:.2f} tissue units, scale {scale:.6f} -> "
          f"{r_tissue[0]:.4f} -> {r_tissue[-1]:.4f} box units "
          f"({r_tissue[-1] * 64:.1f} grid cells at n_grid 64); {frames} pass-2 frames at "
          f"{stride} per kept mesh", flush=True)

    spec = build(name, frames, tissue_npz, scale, n_particles=n_part,
                 n_fibres=arg("--fibres", n_part // 20, int),
                 sub=arg("--sub", 4.0e-4, float), drag=arg("--drag", 8.0, float),
                 k_frac=arg("--kfrac", 0.15, float), mu=arg("--mu", 0.4, float),
                 mesh_stride=stride)
    path = os.path.join(d, "spec.yaml")
    yaml.safe_dump(spec, open(path, "w"), sort_keys=False)
    per = max(1, n_part // spec["operators"][1]["n_fibres"])

    ecm_ops.STRESS_HISTORY.clear(); ecm_ops.STRESS_RAW.clear(); MC.reset()
    t0 = time.time()
    H, out = engine_run(schema.load(path), device=dev)
    solve_s = time.time() - t0
    print(f"[{name}] SOLVE {solve_s:.1f} s for {frames} frames", flush=True)

    P = np.asarray(out["sets"]["mpm_particle"]["pos"], np.float32)
    vm = [np.asarray(v, np.float32) for v in ecm_ops.STRESS_RAW] or None
    n = min(len(P), len(vm) if vm else len(P))
    P = P[:n]; vm = vm[:n] if vm else None
    import test_02_ecm_block as T2
    if vm:
        band, sc = T2.bands_from_vm(vm)
        print(f"[{name}] stress colour full-scale {sc:.4g} (p99 over the run)", flush=True)
    else:
        band = np.zeros((n, P.shape[1]), np.uint8); sc = 1.0
    np.savez_compressed(os.path.join(d, "traj.npz"), pos=P, stress=np.asarray(band, np.uint8),
                        vm=np.asarray(vm, np.float16) if vm else np.zeros((0,), np.float16))

    m = measure(P, vm, per, scale, r_tissue[:n], d, 1.0 / 64)
    m["solve_s"] = solve_s
    m["scale"] = scale
    m["stress_full_scale"] = float(sc)
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    plot(m, os.path.join(d, "metrics.png"))
    # THE PROTOTYPE'S OWN RENDERER, ON REQUEST, AND IT IS THE ONE TO PREFER. `render` below draws the
    # epithelium as a cloud of vertex dots, which cannot show a cell, a division or a junction --
    # `run_ecm.render` draws the entities, in the 2x2 the reference runs use, with the matrix as the
    # strands it was seeded as. It is not duplicated here: this hands it the arrays.
    if "--panels" in sys.argv:
        import run_ecm
        yaml.safe_dump(spec, open(os.path.join(d, "spec_run.yaml"), "w"), sort_keys=False)
        ecm_ops.STRESS_HISTORY[:] = list(np.asarray(band))
        ecm_ops.STRESS_RAW[:] = list(vm) if vm else []
        run_ecm.render(name, {"sets": {"mpm_particle": {"pos": P}}}, spec, d,
                       movie_frames=arg("--movie-frames", 200, int), fps=arg("--fps", 20, int))
    else:
        render(P, band, tissue_npz, scale, d, name, per=per)

    c = m.get("contact", {})
    print(f"[{name}] momentum residual max {c.get('momentum_residual_max', float('nan')):.2e}, "
          f"penetration max {c.get('penetration_max_cells', float('nan')):.2f} cells, "
          f"{m.get('inside', {}).get('max_count', -1)} particles ever behind the surface, "
          f"displacement exponent {m['disp_exponent'][-1]:.2f} against the shell's own "
          f"{m['lame_exponent']:.2f} -> {d}", flush=True)

    spec_note = dict(
        what="a replayed epithelial spheroid loading a fibrous matrix through particle-to-surface "
             "contact",
        combines={"tissue": "01c (two-pool myosin, cytokinetic ring x1)",
                  "matrix": "02h lean (20 per strand, 8 substeps, drag 8, von Mises)",
                  "interface": "03 (ICFEMP), generalised to a curved moving surface"},
        replaces=["cell_to_ecm[replay] -- a radial force from a smoothed radius map, with no "
                  "reaction and no shear",
                  "cell_exclude_3d -- a projection backstop, which makes non-penetration "
                  "unmeasurable"],
        lookup="the (theta, phi) bin of the star-shaped surface, re-sized each frame from the "
               "largest triangle; the two polar caps are single buckets",
        one_way="the tissue is a replay: the reaction is computed, conserved and recorded, and "
                "nothing gives it back to the epithelium",
        measures=["momentum residual", "penetration in grid cells", "particles behind the surface",
                  "radial displacement exponent against the shell's closed form",
                  "compaction against radius", "strand cos^2 to the radius", "reaction pressure map"],
        plexus2=dict(mesh_contact=dict(kind="Lateral", acts_on="contact (edge set: mpm_particle -> "
                                                               "vertex)",
                                       returns="a delta to both endpoints in one call")))
    yaml.safe_dump(spec_note, open(os.path.join(d, "what.yaml"), "w"), sort_keys=False)


if __name__ == "__main__":
    main()
