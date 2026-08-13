#!/usr/bin/env python
"""test_03f_shear_breach -- the other two things a surface does to a matrix: it DRAGS it, and it
BREACHES it.

    python test_03f_shear_breach.py --which shear   ->  log/okuda_ECM/03f_mesh_shear/
    python test_03f_shear_breach.py --which breach  ->  log/okuda_ECM/03g_mesh_breach/

WHY THESE TWO AND NOT MORE OF THE PRESS. `03c` already shows the normal half of the contact law: a
patch pressed into a block against a rigid floor, coloured by |J-1|, with the reaction returning to
the mesh. The two halves it does NOT show are the ones the interface was built for.

  shear   The tangential law -- regularised Coulomb, saturating at mu*f_n -- appears in `03c` only as
          a number in `metrics.png` (mean slip with friction against without). Here it is the whole
          loading: the pad presses 6 grid cells in and then TRANSLATES 0.13 box units at that depth,
          and what the matrix does with that is dragged, piled ahead of the pad and sheared against
          a pinned base. Coloured by von Mises rather than |J-1|, because a drag is deviatoric: it
          changes shape at constant volume, and a volumetric colour is blank where the physics is.

  breach  What a hole is FOR. `03c` has one and nothing goes through it, because the block is free to
          squeeze out sideways from under the patch -- the path of least resistance is the rim, not
          the hole. Close the box (grid-velocity walls on x and y, the same boundary condition as
          the floor) and the only way out is the breach: the matrix extrudes through it as a plug
          while the surface around it keeps descending. That is the geometry a proteolytic breach
          makes, and the reason to have a hole in the mesh at all.

BOTH RUNS ARE THE SAME RIG AS 03c -- `MeshOnMatrix`, the same particle-to-surface penalty written as
a fraction of the explicit stability ceiling, the same dt -- with the rim's prescribed velocity
pointed a different way and, for the breach, four more grid-velocity clamps. What each run changes
beyond that is named where it is set and why: a no-slip floor and a softer, sticker interface for the
shear; near-incompressibility, closed sides and a heavy plate for the breach. Nothing about the
CONTACT SCHEME changes in either, which is what makes them evidence about it rather than three
unrelated movies.

THE SCHEME: ICFEMP, Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015) Comput. Methods Appl. Mech. Engrg.
293:1-19, doi:10.1016/j.cma.2015.04.005.

NO LABELS ON THE FRAMES. These are drawn for the minisite, where the page writes the caption and a
run name burnt into the top-left corner survives cropping and contradicts it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.animation import FFMpegWriter                              # noqa: E402
from matplotlib.colors import ListedColormap                               # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import ecm_spec as _ES                                                     # noqa: E402
from test_03_mesh_contact import MeshOnMatrix, LOG, MESH_C                 # noqa: E402

CMAP = ListedColormap(_ES.STRESS_COLORS)
FLOOR_C = "#9aa0a6"


# ---------------------------------------------------------------------------------------------
# the two loadings
# ---------------------------------------------------------------------------------------------
def make(which, dev):
    """The rig, the per-frame boundary condition, and the view -- one dict per demo."""
    if which == "shear":
        # FOUR THINGS DIFFER FROM 03c, AND EACH ONE HAD TO. What is being asked for is a strain
        # GRADIENT under the interface, and every one of these is a way the first attempts failed
        # to produce one:
        #   floor_stick   03c's floor only stops the DOWNWARD component of the grid velocity, so
        #                 under a tangential load the whole column TRANSLATES with the pad -- one
        #                 rigid slab moving, no gradient anywhere. No-slip pins the base.
        #   mu 0.9, E 150 (03c: 0.4 and 400)  the traction the contact can carry is mu*f_n, and
        #                 what it buys in strain is that over the shear modulus. A sticky pad on a
        #                 matrix 2.7x softer is a drag you can see rather than one you can measure.
        #   m_vert 3e-3 (03c: 2e-4)  the reaction enters the mesh as fv/m_v while the springs scale
        #                 WITH m_v, so a heavier lattice is a flatter one at identical spring
        #                 dynamics. A light pad bows away from the load and presses with its rim.
        #   patch 0.30 on a block 0.56 wide  the pad must finish its 0.13-box-unit travel still
        #                 standing on matrix; at 03c's 0.42 on a 0.44 block it drags itself off the
        #                 edge and the second half of the clip is a mesh hanging over nothing.
        # k_frac 0.25 rather than 03c's 0.15: the fraction is SQUARED into the stiffness, so this is
        # a contact 2.8x stiffer, and at this press it has to be -- the softer one let the matrix
        # through the surface by 1.7 grid cells, which in a clip about a surface DRAGGING a matrix
        # reads as the matrix going through the surface.
        rig = MeshOnMatrix(dev=dev, floor=0.18, floor_stick=True, mu=0.9, E=150.0, m_vert=3.0e-3,
                           patch=0.30, k_frac=0.25,
                           block=(0.22, 0.30, 0.18, 0.78, 0.70, 0.58), track_vm=True)
        press_frac, v_press, v_drag = 0.30, 2.5, 2.0
        # PRESS FIRST, THEN DRAG AT CONSTANT DEPTH. Dragging while still descending mixes the two
        # loadings and the band that appears cannot be attributed to either; holding z fixed makes
        # the normal force a constant the tangential law is measured against.
        def drive(t, frames):
            return ((0.0, 0.0, -v_press) if t < press_frac * frames
                    else (v_drag, 0.0, 0.0))
        view = dict(xlim=(0.20, 0.80), ylim=(0.28, 0.72), zlim=(0.13, 0.62),
                    colour="vm", elev=14, azim=-62,
                    total_drag=v_drag * (1 - press_frac))
        return rig, drive, view
    if which == "breach":
        # patch 0.46 against walls at 0.28/0.72: the mesh overhangs the confined column on every
        # side, so there is no annulus between the patch's rim and the wall for the material to
        # escape through. A patch narrower than the box is a press with a leak.
        # NEARLY INCOMPRESSIBLE (nu = 0.46, against 03c's 0.2) and a plate 50x the mass. Both decide
        # whether anything comes through the hole at all, and both were found by watching it not:
        # at nu = 0.2 the matrix answered a closed-box press by getting 7% shorter and extruded
        # nothing, and at 03c's vertex mass the lattice answered it by DOMING upwards off the
        # material -- a surface bowing away from the load, with the mound appearing over the whole
        # patch rather than over the hole. Held flat against material that cannot shrink, the
        # displaced volume has one way out, and the mound is the hole's doing.
        # k_frac 0.4 -- a contact SEVEN TIMES stiffer than 03c's, the fraction being squared --
        # because this run's whole claim is that the material leaves through the hole. At 0.15 it
        # left 1.3 grid cells of itself standing above the mesh AWAY from the hole, and a plug that
        # cannot be told from leakage is not evidence of a breach. It still leaks 1.9 cells at the
        # deepest press, which `penetration_max_cells` reports and the page quotes.
        rig = MeshOnMatrix(dev=dev, floor=0.18, floor_stick=True, mu=0.4, patch=0.46, hole_r=0.09,
                           nu=0.46, m_vert=1.0e-2, k_frac=0.4, walls=(0.28, 0.72), track_vm=True)
        v_press = 0.8
        def drive(t, frames):
            return (0.0, 0.0, -v_press)
        # VON MISES HERE TOO, not |J-1|. The plug is the point of the run and it RELAXES as it comes
        # out -- once through the hole it is no longer compressed -- so a volumetric colour draws it
        # in the palette's darkest band, on black. The throat is where the material is sheared most,
        # so the deviatoric measure lights up exactly the stream going through.
        view = dict(xlim=(0.25, 0.75), ylim=(0.25, 0.75), zlim=(0.13, 0.72),
                    colour="vm", elev=16, azim=-62, total_drag=0.0)
        return rig, drive, view
    sys.exit(f"unknown demo {which!r} -- shear or breach")


# ---------------------------------------------------------------------------------------------
# the movie: 3D left, the same instant in section right -- the layout `03c` uses, so the three
# cards on the page are one series
# ---------------------------------------------------------------------------------------------
def render(rig, drive, view, frames, d, every=6, fps=15, n_draw=13000):
    """Step once, keep the drawn frames, then draw them against ONE colour scale.

    The scale cannot be known before the run -- it is the p99 of the stress over every kept frame --
    and a scale chosen per frame turns "the stress front arrives" into a movie in which nothing
    happens, because the brightest thing on screen is always the same brightness.
    """
    st = max(1, rig.N // n_draw)
    kept = []
    for t in range(frames):
        rig.drive = drive(t, frames)
        rig.step()
        if t % every:
            continue
        X = rig.x.detach().cpu().numpy()
        V = rig.V.detach().cpu().numpy()
        S = (rig.vm.detach().cpu().numpy() if view["colour"] == "vm"
             else np.abs(rig.J.detach().cpu().numpy() - 1.0))
        kept.append((t, X, V, S))
    allS = np.concatenate([k[3][::7] for k in kept])
    s_hi = float(np.percentile(allS[np.isfinite(allS)], 99)) or 1.0
    name = os.path.basename(d)
    print(f"[{name}] colour full-scale ({view['colour']}) = {s_hi:.4g} (p99 over the run), "
          f"{len(kept)} frames", flush=True)

    hole = rig.hole.detach().cpu().numpy() if rig.hole_r > 0 else None
    xr = view["xlim"][1] - view["xlim"][0]
    yr = view["ylim"][1] - view["ylim"][0]
    zr = view["zlim"][1] - view["zlim"][0]
    fig = plt.figure(figsize=(11.2, 5.6), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for (t, X, V, S) in kept:
            col = np.clip(S / s_hi, 0, 1)
            zf = V[:, 2].reshape(rig.nx, rig.nx)
            if hole is not None:                    # a hole is drawn by not drawing it
                zf = zf.copy()
                zf.reshape(-1)[hole] = np.nan
            fig.clf()
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black",
                                 computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            ax.scatter(X[::st, 0], X[::st, 1], X[::st, 2], s=3.5, c=col[::st], cmap=CMAP,
                       vmin=0, vmax=1, marker=".", linewidths=0, alpha=0.85, depthshade=False)
            ax.plot_wireframe(V[:, 0].reshape(rig.nx, rig.nx), V[:, 1].reshape(rig.nx, rig.nx),
                              zf, color=MESH_C, lw=0.7)
            ax.set_xlim(*view["xlim"]); ax.set_ylim(*view["ylim"]); ax.set_zlim(*view["zlim"])
            # THE BOX ASPECT FOLLOWS THE LIMITS, so a box unit is the same length on all three axes
            # (up to the 0.9 flattening 03c uses). A fixed (1,1,0.9) with a wider x range squashes
            # the patch's travel back out of the picture.
            ax.set_box_aspect((xr, yr, 0.9 * zr))
            ax.view_init(elev=view["elev"], azim=view["azim"])

            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            sl = np.abs(X[:, 1] - 0.5) < 0.02
            a2.scatter(X[sl][:, 0], X[sl][:, 2], s=7, c=col[sl], cmap=CMAP, vmin=0, vmax=1,
                       marker=".", linewidths=0)
            mid = rig.nx // 2
            a2.plot(V[:, 0].reshape(rig.nx, rig.nx)[:, mid], zf[:, mid], "-", color=MESH_C, lw=1.6)
            a2.plot(view["xlim"], [rig.floor, rig.floor], "-", color=FLOOR_C, lw=2.5)
            if rig.walls is not None:
                for w in rig.walls:
                    a2.plot([w, w], [rig.floor, view["zlim"][1]], "-", color=FLOOR_C, lw=1.6,
                            alpha=0.55)
            a2.set_xlim(*view["xlim"]); a2.set_ylim(*view["zlim"])
            a2.set_aspect("equal"); a2.axis("off")
            wri.grab_frame()
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)
    return s_hi


def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    which = sys.argv[sys.argv.index("--which") + 1] if "--which" in sys.argv else "shear"
    frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 900
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 6
    d = os.path.join(LOG, {"shear": "03f_mesh_shear", "breach": "03g_mesh_breach"}[which])
    os.makedirs(d, exist_ok=True)
    rig, drive, view = make(which, dev)
    print(f"[{which}] penalty k = {rig.k_pen:.3g} = {100*(rig.k_pen/rig.k_ceiling)**0.5:.0f}% of "
          f"the explicit ceiling in sqrt, {rig.N} particles, dt {rig.dt}", flush=True)
    s_hi = render(rig, drive, view, frames, d, every=every)

    res = rig.res
    out = dict(demo=which, frames=frames, particles=rig.N, k_penalty=rig.k_pen, mu=rig.mu,
               dt=rig.dt, colour=view["colour"], colour_full_scale=s_hi,
               momentum_residual_max=float(max(res["momentum"])),
               penetration_max_cells=float(max(res["depth"]) * rig.n_grid),
               contacts_max=int(max(res["n_pen"])),
               contact_frames=int(sum(1 for n in res["n_pen"] if n > 0)),
               slip_mean=float(np.mean([v for v, n in zip(res["slip"], res["n_pen"])
                                        if n > 0] or [0.0])),
               height_start=float(res["height"][0]), height_min=float(min(res["height"])),
               compression=float(1 - min(res["height"]) / max(res["height"][0], 1e-9)),
               top_max=float(rig.x[:, 2].max()),
               # THE TWO NUMBERS THE PAGE QUOTES, in the units a reader can picture: how far the
               # surface went in, in grid cells, and how far the material came back OUT past it.
               # Both are derived here rather than in the site generator, so the caption cannot
               # drift from the run that produced it.
               indent_cells=float((res["z_mesh"][0] - min(res["z_mesh"])) * rig.n_grid),
               drag_box_units=float(view["total_drag"] * frames * rig.dt),
               plug_above_surface_cells=float((float(rig.x[:, 2].max()) - min(res["z_mesh"]))
                                              * rig.n_grid),
               series={k: [float(x) for x in v[::4]] for k, v in res.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)

    import yaml
    what = ("a triangulated patch pressed into an MPM block and then DRAGGED across it: the "
            "tangential half of the contact law, as a loading rather than as a number"
            if which == "shear" else
            "a triangulated patch with a hole pressed into a laterally CONFINED MPM block: with no "
            "rim to escape through, the matrix extrudes as a plug through the breach")
    yaml.safe_dump(dict(
        what=what,
        scheme="ICFEMP-style particle-to-surface contact (Chen et al. 2015 CMAME 293:1), the same "
               "interface as 03c; only the rim's prescribed velocity and the grid boundary "
               "conditions differ",
        loading=("press 2.6 grid cells, then translate 0.12 box units at constant depth"
                 if which == "shear" else
                 "press at 0.5 box units per unit time throughout, box closed on x and y"),
        colour=("von Mises of the Kirchhoff stress -- a drag is deviatoric, so |J-1| is blank "
                "where the shear band is" if which == "shear" else
                "|J-1|, the volumetric strain: under a confined press the material cannot leave "
                "except through the hole"),
        rig=dict(mesh_cell_over_dx=rig.cell_size / rig.dx, n_grid=rig.n_grid, particles=rig.N,
                 k_penalty=float(rig.k_pen), mu=rig.mu, dt=rig.dt, floor=rig.floor,
                 hole_r=rig.hole_r, walls=list(rig.walls) if rig.walls else None),
        measured=dict(momentum_residual_max=out["momentum_residual_max"],
                      penetration_max_cells=out["penetration_max_cells"],
                      compression=out["compression"])),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{which}] momentum residual max {out['momentum_residual_max']:.2e}, penetration max "
          f"{out['penetration_max_cells']:.2f} cells, block compressed "
          f"{100*out['compression']:.1f}%, highest particle {out['top_max']:.3f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
