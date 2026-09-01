"""VTK views of the two-scale toy, 2-D and 3-D, built on the promoted renderer.

WHY VTK AND NOT MATPLOTLIB, for the 2-D case as much as the 3-D one. The 3-D case has no choice:
`mpl_toolkits.mplot3d` has no depth buffer, and a 256^3 field is not a surface anyway -- it needs a
ray cast. The 2-D case *could* be an `imshow`, and was. It is here instead so that the two
dimensions are the SAME PICTURE at different D: one colour map, one background, one caption line,
one movie container, one anti-aliaser. A figure whose 2-D and 3-D panels came out of two different
renderers cannot be compared panel to panel, and this prototype's whole claim is that the 2-D and
3-D toys pose the same problem.

Everything that draws is `plexus.render_vtk`'s: `_plotter` (off-screen, black, MSAA x8),
`open_movie` (fragmented mp4, readable while it is written), `_grid` (ImageData spanning the UNIT
BOX, which is where every Plexus field lives), `_range` (percentile clim fixed for the whole clip).
The 3-D path is `render_vtk.evolve_volume` itself, reached by writing the `trajectory.npz` layout
it reads -- not a reimplementation of it.

THE ONE PLACE THE TWO DIMENSIONS GENUINELY DIFFER, stated rather than hidden. A volume render shows
only what its opacity transfer function lets through, so `evolve_volume` ray-casts |field| under
`sigmoid_5`: the near-zero bulk stays transparent and what is visible is where the field is large.
A plane has no such integral, so the 2-D view shows the SIGNED field on a symmetric range. So in
3-D a Kuramoto domain wall (v ~ 0) reads as a transparent gap between two bright sheets, and in 2-D
it reads as the mid-colour line between a yellow band and a blue one. Same wall, two renderings,
because one is seen through and the other is not.
"""

from __future__ import annotations

import os

import numpy as np

from plexus.render_vtk import _grid, _plotter, _range, evolve_volume, open_movie

SIZE_ = 448        # per panel; a pair movie is twice this wide
CMAP = "viridis"    # the one colour map, matching viz.CMAP
FPS = 24


def plane_movie(a, out, label, cmap=CMAP, fps=FPS, clim=None):
    """A 2-D field as a flat VTK plane, one frame per time step. `a` is [T, nx, ny].

    The slab is one cell thick in z so that `pv.ImageData` -- a volumetric type -- can carry a
    plane; `view_xy` then looks straight down it and the thickness never appears. Doing it this way
    rather than with `pv.Plane` keeps `_grid`'s unit-box convention, so a 2-D panel and a 3-D panel
    put the same world coordinate in the same place.
    """
    import pyvista as pv

    a = np.asarray(a, np.float32)
    lo, hi = clim if clim is not None else _range(a, 0.5, 99.5)
    lim = max(abs(lo), abs(hi))                       # symmetric: the sign is the signal
    p = _plotter()
    open_movie(p, out, fps)
    actor = txt = None
    for t in range(a.shape[0]):
        nx, ny = a[t].shape
        g = pv.ImageData(dimensions=(nx + 1, ny + 1, 2))
        g.spacing = (1.0 / nx, 1.0 / ny, 1.0)
        g.origin = (0.0, 0.0, 0.0)
        g.cell_data["a"] = np.ascontiguousarray(a[t]).ravel(order="F")
        if actor is not None:
            p.remove_actor(actor)
        if txt is not None:
            p.remove_actor(txt)
        actor = p.add_mesh(g, scalars="a", cmap=cmap, clim=[-lim, lim],
                           lighting=False, show_scalar_bar=False)
        txt = p.add_text(f"{label}   {nx}x{ny}   frame {t + 1}/{a.shape[0]}   "
                         f"clim [{-lim:.2f}, {lim:.2f}]",
                         position="upper_left", font_size=11, color="white")
        if t == 0:
            p.view_xy()
            p.reset_camera()
            p.camera.zoom(1.25)
        p.write_frame()
    p.close()
    return f"{a.shape[0]} frames, {a.shape[1]}x{a.shape[2]}, clim [{-lim:.3f}, {lim:.3f}]"


def _opacity_for(a, empty_tol=1e-6, empty_frac=0.5):
    """Choose the transfer function FROM THE DATA, because the wrong one hides the measurement.

    `sigmoid_5` is `evolve_volume`'s default and is right for a field that fills the box: it keeps
    the low-magnitude bulk faint so the strong structure is what reads. It is wrong for a MASKED
    field, and both ways at once. Measured on the 3-D fine rule, which is zero outside four tubes
    and so is 90% empty:

      * the sigmoid's alpha at |v| = 0 is small but not zero, and a ray crosses ~256 empty voxels,
        so the vacuum integrated to a visible purple fog -- viridis(0) is dark purple -- and the
        cube looked full of something it does not contain;
      * inside a tube |v| is near 1 almost everywhere except at the phase domain walls, so the
        tube wall saturated opaque and the pattern was legible only where a tube met a FACE.

    A ramp pinned to zero at the bottom fixes both with one change. Empty space contributes exactly
    nothing, the domain walls (|v| ~ 0) open as gaps, and the low peak alpha lets a ray reach the
    far side of a tube, so the pattern is read through the volume instead of off its skin.

    The choice is made by measuring the zero fraction rather than by naming the toy, so it holds for
    any masked field and violates no part of G2.
    """
    frac = float((np.abs(a) < empty_tol).mean())
    if frac < empty_frac:
        return "sigmoid_5", frac                      # fills the box: the promoted default
    return [0.0, 0.0, 0.04, 0.12, 0.30], frac         # masked: vacuum transparent, walls open


def volume_movie(a, out, label, fps=FPS, cmap=CMAP, opacity=None, fill=0.95):
    """A 3-D field as a ray-cast cube, through `render_vtk.evolve_volume`.

    `evolve_volume` reads a run DIRECTORY containing `trajectory.npz`, so the array is written into
    that layout in a scratch directory beside the movie and handed over. Going through the promoted
    function rather than around it is the point: the cube in this prototype's figures is drawn by
    the same code as every other cube in the repo, and a change to the renderer reaches both.
    """
    a = np.asarray(a, np.float32)
    if opacity is None:
        opacity, frac = _opacity_for(a)
        print(f"[vtk_toy] {os.path.basename(out)}: {frac:.1%} of voxels empty -> "
              f"opacity {opacity}", flush=True)
    scratch = os.path.join(os.path.dirname(out) or ".", "_vtk_" + os.path.basename(out)[:-4])
    os.makedirs(scratch, exist_ok=True)
    npz = os.path.join(scratch, "trajectory.npz")
    np.savez(npz, **{"f__grid": a[:, None]})          # [T, C, nx, ny, nz], the layout it reads
    try:
        return evolve_volume(scratch, out, field="f", cmap=cmap, fill=fill,
                             label=label, fps=fps, opacity=opacity)
    finally:
        os.remove(npz)
        os.rmdir(scratch)


def movie(a, out, label, **kw):
    """Dispatch on the field's own rank, so a caller does not branch on the toy's dimension."""
    a = np.asarray(a)
    if a.ndim == 3:
        return plane_movie(a, out, label, **kw)
    if a.ndim == 4:
        return volume_movie(a, out, label, **kw)
    raise ValueError(f"expected [T, nx, ny] or [T, nx, ny, nz], got {a.shape}")


def pair_movie(a, b, out, labels=("ground truth", "inferred"), cmap=CMAP, fps=FPS,
               clim=None, per_step=None):
    """TWO PANELS, ONE FRAME, ONE COLOUR SCALE: what was recorded beside what the model produced.

    THE CLIM IS SHARED AND FIXED FOR THE WHOLE CLIP, and that is the only thing that makes the
    comparison honest. Per-panel normalisation would rescale a decaying rollout back to full
    contrast every frame, so a model whose amplitude was collapsing would look identical to one
    that was right; per-frame normalisation would do the same in time. Both panels are drawn
    against the range of the GROUND TRUTH, so a prediction that leaves that range saturates -- which
    is the correct rendering of a prediction that has left the data.

    `per_step` IS PER STEP, AND THAT IS THE POINT OF IT. The first version stamped the rollout's
    MEAN R^2 on every frame, which is a summary of the whole clip printed over one moment of it --
    so a frame at step 21 whose own agreement was R^2 0.513 carried the label "-0.123", the mean
    dragged down by a tail decaying to -0.91 by step 100. The two panels looked alike and the
    caption said they did not, and the caption was what was wrong. A rollout figure has to report
    the frame it is showing.

    Dispatches on rank exactly as `movie` does, so a 2-D toy and a 3-D toy give the same figure at
    different D.
    """
    import pyvista as pv

    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"pair_movie shapes differ: {a.shape} vs {b.shape}")
    lo, hi = clim if clim is not None else _range(a, 0.5, 99.5)
    lim = max(abs(lo), abs(hi))
    vol = a.ndim == 4
    opacity = _opacity_for(a)[0] if vol else None

    p = _plotter()
    p.close()
    p = pv.Plotter(off_screen=True, window_size=(2 * (SIZE_ or 448), SIZE_ or 448), shape=(1, 2))
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    open_movie(p, out, fps)
    actors = [None, None]
    txts = [None, None]
    for t in range(a.shape[0]):
        for k, (arr, lab) in enumerate(zip((a, b), labels)):
            p.subplot(0, k)
            if actors[k] is not None:
                p.remove_actor(actors[k])
            if txts[k] is not None:
                p.remove_actor(txts[k])
            if vol:
                actors[k] = p.add_volume(_grid(np.abs(arr[t])), scalars="a", cmap=cmap,
                                         opacity=opacity, clim=[0.0, lim], show_scalar_bar=False)
            else:
                nx, ny = arr[t].shape
                g = pv.ImageData(dimensions=(nx + 1, ny + 1, 2))
                g.spacing = (1.0 / nx, 1.0 / ny, 1.0)
                g.cell_data["a"] = np.ascontiguousarray(arr[t]).ravel(order="F")
                actors[k] = p.add_mesh(g, scalars="a", cmap=cmap, clim=[-lim, lim],
                                       lighting=False, show_scalar_bar=False)
            note = ""
            if per_step is not None and k == 1 and t < len(per_step):
                r2, rr = per_step[t]
                note = f"   R2 {r2:.3f}  r {rr:.3f}"
            txts[k] = p.add_text(f"{lab}   step {t + 1}/{a.shape[0]}{note}",
                                 position="upper_left", font_size=10, color="white")
            if t == 0:
                if vol:
                    p.camera_position = "iso"
                else:
                    p.view_xy()
                p.reset_camera()
                p.camera.zoom(1.15)
        p.write_frame()
    p.close()
    return f"{a.shape[0]} frames, {'x'.join(str(s) for s in a.shape[1:])}, clim +-{lim:.3f}"
