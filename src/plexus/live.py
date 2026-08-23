"""The live snapshot: `3d.png`, rewritten every 10% of the run, while the run is still going.

WHY IT EXISTS. A generation of a 1,800-frame vertex model takes 20-40 minutes and, until it finishes,
writes nothing anyone can look at -- `progress.json` says `frame 1430, n_cells 7974` and that is the
whole of it. Two runs of a promotion pair are then indistinguishable from two runs of the WRONG THING
for the entire time they are running, and the first look at a mistake comes half an hour after it was
made. This writes a picture of the current state ten times over the run, in place, so `3d.png` in
either side's directory is always a recent view of what that side is doing.

    "can you write 3d.png every 10% of frames (print frame out of top left) so that I can check live
     the progress in A and B that should also be live? the live 3d.png should be default when
     generating"                                                            -- Cedric, 22 August

IT IS WRITTEN ATOMICALLY. The file is opened while a viewer may be reading it, so the render goes to
`3d.png.tmp` and is renamed over `3d.png` -- an `os.replace` on one filesystem is atomic, so a reader
sees the old picture or the new one and never half a PNG.

IT IS OVERWRITTEN BY THE FINAL RENDER, deliberately. The end-of-run `3d.png` is the better picture
(the movie's own code path, the run's final camera); this is the same file, refreshed early. Nothing
downstream has to know which one it is looking at.

THE MATPLOTLIB IMPORT IS INSIDE THE FUNCTION. `plexus.generators.graph_data_generator` states in its
own docstring that it never imports matplotlib -- visualization is a separate concern and a plot
switch in the generator is the ParticleGraph anti-pattern it is avoiding. A lazy import inside the
callback keeps that true at module level while still letting `-o generate` draw.
"""
from __future__ import annotations

import os

import numpy as np


def every_n(n_frames: int, frac: float = 0.1) -> int:
    """The stride that gives `1/frac` snapshots over the run, at least 1."""
    return max(1, int(round(n_frames * float(frac))))


def _live_pos(H, sname):
    """(positions of the live elements, the mesh table or None) for the set to draw."""
    lvl = H.level(sname)
    if "pos" not in lvl.state_schema:
        return None, None
    m = getattr(lvl, "mesh", None)
    pos = lvl.get("pos").detach().cpu().numpy()
    # A MESH SET IS DRAWN TO `Nv`, NOT TO `occ`. `cell_die` rewrites the half-edge table and never
    # touches `Nv`, so vertices orphaned by a death are still inside every `state[:Nv]` slice --
    # that is the model's behaviour, and a picture that quietly drew a different subset from the one
    # the mechanics acts on would be the wrong picture.
    if m and int(m.get("Nv", 0)) > 0:
        return pos[:int(m["Nv"])], m
    return pos[lvl.active.detach().cpu().numpy()], None


def dot_area_pt2(pos, span_data, fig_px, dpi, fill=0.9, cap=(1.0, 400.0)):
    """Scatter `s` (an AREA in points^2) such that a dot spans `fill` of the LOCAL SPACING.

    THE PROBLEM THIS REPLACES. `point_size` was a constant in the spec, so the same number that
    made a 4,000-cell disc read as a sheet made a 1,000-cell one read as dinner plates and a
    40,000-cell one as dust. Dot size is not a property of the picture, it is a property of the
    picture RELATIVE TO THE SPACING, and the spacing changes whenever the population does -- which
    for a growing tissue is every frame.

    IT IS MEASURED, NOT ESTIMATED FROM DENSITY. `sqrt(area / n)` needs a hull, is wrong for any
    non-convex or non-uniform layout, and is biased at the boundary. The MEDIAN NEAREST-NEIGHBOUR
    DISTANCE is the quantity "nearly touching" actually refers to, it needs no hull, and the median
    shrugs off the edge cells that have fewer neighbours than the interior.

    THE CONVERSION IS EXACT, so the result is invariant to figure size and dpi as well as to n:
    data -> px is `fig_px / span_data`, px -> pt is `72 / dpi`, and `s` is an area, hence squared.
    A caller that changes the canvas from 6.2in at 110 dpi to 8in at 200 dpi therefore gets the
    same apparent dot, which is what makes the live snapshot and the movie agree.

    `fill` = 1.0 is exactly touching. 0.9 leaves a hairline so the population reads as flat while
    the individual dots stay countable, which is the look these plots want.
    """
    q = np.asarray(pos, float)[:, :2]
    if len(q) < 2 or span_data <= 0:
        return 12.0
    try:
        from scipy.spatial import cKDTree
        # k=2 because the nearest neighbour of a point is itself at distance 0
        nn = cKDTree(q).query(q, k=2)[0][:, 1]
    except Exception:
        i = np.random.default_rng(0).choice(len(q), size=min(len(q), 1500), replace=False)
        dd = np.linalg.norm(q[i][:, None, :] - q[i][None, :, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        nn = dd.min(1)
    sp = float(np.median(nn[np.isfinite(nn)])) if np.isfinite(nn).any() else 0.0
    if sp <= 0:
        return 12.0
    pt = sp * (float(fig_px) / float(span_data)) * (72.0 / float(dpi))
    return float(np.clip((fill * pt) ** 2, *cap))


def chem_rgb(chem, nF=None):
    """(rgb [n,3], [max per species]) from a `chem` array -- THE ONE COLOUR LAW for RD output.

    Lives here, and is imported by `plexus.plot`, because the alternative is two copies that drift:
    the live snapshot and the movie of the same run showed DIFFERENT PICTURES of the same numbers
    for exactly that reason -- `movie_cell.mp4` never read `chem` at all and drew every cell in
    matplotlib's default tab10 blue, so a two-species Turing run rendered as a uniform blue disc.
    This function takes a numpy array and no matplotlib, so either caller can use it.

    WHITE IS QUIET, COLOUR IS ACTIVITY -- subtractive, not additive. Building up from black made an
    inactive cell near-black on a black canvas: the disc read as red specks in a void, the tissue
    was invisible, and there was no way to see WHERE the pattern was not.

    TWO SPECIES, TWO CHANNELS. Columns 0,1 are the first Gray-Scott pair and 2,3 the second (see
    `_chan`), so A takes green and blue away leaving red, B takes red and green leaving blue. A
    cell carrying both loses everything and would go pure black, indistinguishable from the canvas,
    so the floor stops at 0.14 -- "both species here" is the most interesting cell on the plot and
    it stays a dark dot rather than a hole.
    """
    a = np.asarray(chem, float)
    if a.ndim != 2 or a.shape[1] < 2:
        return None, []
    n = a.shape[0] if nF is None else int(nF)
    a = a[:n]
    hi, band = [], []
    for ch in (0, 2):
        if a.shape[1] > ch:
            v = np.where(np.isfinite(a[:, ch]), a[:, ch], 0.0)
            top = float(v.max())
            hi.append(top)
            band.append(np.clip(v / top, 0, 1) if top > 1e-9 else np.zeros(n))
        else:
            band.append(np.zeros(n))
    A, B = band
    f = 0.86
    cols = np.stack([1.0 - f * B, 1.0 - f * np.maximum(A, B), 1.0 - f * A], axis=1)
    return np.clip(cols, 0, 1), hi


def _face_colours(H, nF):
    """`chem_rgb` for the first set that carries a `chem` block, or (None, []) when none does."""
    for _s, l in H.levels.items():
        if "chem" not in getattr(l, "state_schema", {}):
            continue
        c = l.get("chem")
        if c is None or c.shape[0] < nF:
            continue
        return chem_rgb(c.detach().cpu().numpy(), nF)
    return None, []


def snapshot(H, tick, n_frames, out_dir, name="", sname=None):
    """Draw the current state to `<out_dir>/3d.png`. Never raises: a failed picture must not take
    down a run that is otherwise fine, which is exactly what a 3D-axes `text` signature once did to
    sixteen runs at once."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if sname is None:                        # the biggest spatial set is the interesting one
            cand = [(int(getattr(l, "n", 0)), s) for s, l in H.levels.items()
                    if "pos" in l.state_schema]
            if not cand:
                return None
            sname = max(cand)[1]
        pos, m = _live_pos(H, sname)
        _clim = []
        if pos is None or len(pos) == 0:
            return None
        D = pos.shape[1]
        os.makedirs(out_dir, exist_ok=True)
        fig = plt.figure(figsize=(6.2, 6.2), facecolor="black")
        if D >= 3:
            ax = fig.add_subplot(111, projection="3d", facecolor="black")
            # FRAMED ON THE CLOUD, NOT ON THE ORIGIN. A vertex mesh is centred on 0 and
            # `abs(pos).max()` frames it correctly; an MPM domain is [0,1]^3 and the same rule puts
            # the body in a corner at a third of the frame -- which is what gate 02's first live
            # picture looked like. Centring on the cloud's own centroid works for both.
            c3 = pos[:, :3].mean(0)
            r = float(np.abs(pos[:, :3] - c3).max()) * 1.08 or 1.0
            if m is not None:
                # THE SURFACE, not a point cloud: the ring of each face, drawn as a polygon. A cloud
                # of 20,000 vertices reads as a fuzzy ball whatever the tissue is doing, which is
                # the one thing a live view must not do.
                from plexus.models.topology import rings_from_flat_3d
                es, et, ef = (np.asarray(_np(m[k])) for k in ("E_srce", "E_trgt", "E_face"))
                rings = rings_from_flat_3d(es, et, ef, int(m["nF"]))
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                keep = [i for i, rg in enumerate(rings) if len(rg) >= 3]
                polys = [pos[np.asarray(rings[i], int), :3] for i in keep]
                fc, _clim = _face_colours(H, int(m["nF"]))
                pc = Poly3DCollection(
                    polys, facecolor=(fc[keep] if fc is not None else "#dcdcdc"),
                    edgecolor="#5a5a5a", linewidths=0.15, alpha=1.0)
                ax.add_collection3d(pc)
            else:
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1.0, c="#dcdcdc", depthshade=False)
            ax.set_xlim(c3[0] - r, c3[0] + r); ax.set_ylim(c3[1] - r, c3[1] + r)
            ax.set_zlim(c3[2] - r, c3[2] + r)
            ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
            put = ax.text2D
        else:
            ax = fig.add_subplot(111, facecolor="black")
            # THE 2D BRANCH CARRIES THE CHEMISTRY TOO. It hardcoded "#dcdcdc" for the same reason
            # the 3D branch did, and a flat RD run -- which is what the whole minisite Turing
            # section is -- has NOTHING to show but its chemistry: the cells never move, so a grey
            # scatter is the identical picture at frame 1 and frame 6000.
            fc, _clim = _face_colours(H, len(pos))
            # SIZED FROM THE MEASURED SPACING, not from the count. `22 * 4000/n` assumed the
            # points were spread over a fixed area, so it was right for this disc and wrong for
            # any other layout -- and wrong every frame of a growing tissue. `dot_area_pt2`
            # measures the median nearest-neighbour distance and converts it exactly, so the dots
            # stay "nearly touching" whatever n, whatever the world box, whatever the canvas.
            _span = 2.0 * float(np.abs(pos[:, :2] - pos[:, :2].mean(0)).max()) * 1.08 or 1.0
            ax.scatter(pos[:, 0], pos[:, 1],
                       s=dot_area_pt2(pos, _span, 6.2 * 110, 110),
                       c=(fc if fc is not None else "#dcdcdc"), linewidths=0)
            ax.set_aspect("equal"); ax.set_axis_off()
            put = ax.text
        n_live = (int(m["nF"]) if m is not None else len(pos))
        unit = "cells" if m is not None else "elements"
        put(0.02, 0.97, f"{name}  frame {tick} / {n_frames}   {n_live} {unit}   [live]",
            transform=ax.transAxes, color="white", fontsize=11, va="top")
        if _clim:
            # THE SCALE, STATED. The colours are normalised per frame (see `_face_colours`), so the
            # picture is meaningless without the number it was divided by.
            put(0.02, 0.94, "  ".join(
                f"{'AB'[i]} max {v:.4g}" for i, v in enumerate(_clim)),
                transform=ax.transAxes, color="#b0b0b0", fontsize=8, va="top")
        fig.subplots_adjust(0, 0, 1, 1)
        # `.tmp` LAST WOULD PICK THE FORMAT FROM THE EXTENSION and matplotlib refuses "tmp"; the
        # temporary name has to keep the `.png` suffix.
        # NAMED FOR THE DIMENSION IT DRAWS. A 2D run wrote a file called `3d.png` showing a flat
        # scatter, which is a small lie that costs a reader real time when four runs sit in one
        # directory and only some of them are flat.
        stem = "3d" if D >= 3 else "2d"
        tmp = os.path.join(out_dir, f".{stem}.partial.png")
        fig.savefig(tmp, dpi=110, facecolor="black")
        plt.close(fig)
        os.replace(tmp, os.path.join(out_dir, f"{stem}.png"))   # atomic: never a half-written PNG
        return os.path.join(out_dir, f"{stem}.png")
    except Exception as e:
        print(f"[live] snapshot at frame {tick} skipped ({type(e).__name__}: {str(e)[:70]})",
              flush=True)
        return None


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
