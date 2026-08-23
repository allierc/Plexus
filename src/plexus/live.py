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


def _face_colours(H, nF):
    """Per-face RGB from the cell set's activator, or None when there is no chemistry to show.

    WHY THE LIVE PICTURE NEEDS THIS. The faces were drawn `facecolor="#dcdcdc"`, a constant, so the
    snapshot showed GEOMETRY ONLY. On a reaction-diffusion run the geometry barely moves -- a coral
    spheroid at frame 600 looks like a coral spheroid at frame 6000 -- and the one question worth
    asking during the twenty-five minutes it takes is whether the pattern is forming. The answer
    was not in the picture: every `3d.png` of every Turing run came out a plain white ball, and the
    only way to find out was to wait for the movie.

    TWO SPECIES GET TWO CHANNELS. A four-wide `chem` is two independent Gray-Scott pairs (columns
    0,1 and 2,3 -- see `_chan`), so species A drives RED and species B drives BLUE and a cell where
    both are present goes magenta. With one pair it is red alone. That makes the two-species runs
    legible at a glance, which is the whole reason they exist.

    PER-FRAME NORMALISATION, DELIBERATELY, and it is the opposite of what the movie does. `evolve`
    fixes its range over the whole run so a strengthening pattern does not look static. A live
    snapshot has no whole run yet -- it cannot know the final maximum -- and a fixed guess would
    render the first snapshots black. So each frame is scaled to its own range and the range is
    printed in the caption, which says what the colour means instead of implying a constant scale.
    """
    for s, l in H.levels.items():
        if "chem" not in getattr(l, "state_schema", {}):
            continue
        c = l.get("chem")
        if c is None or c.shape[0] < nF:
            continue
        a = c[:nF].detach().cpu().numpy()
        cols = np.zeros((nF, 3), float)
        hi = []
        for ch, band in ((0, 0), (2, 2)):                 # species A -> red, species B -> blue
            if a.shape[1] > ch:
                v = a[:, ch]
                v = np.where(np.isfinite(v), v, 0.0)
                top = float(v.max())
                hi.append(top)
                if top > 1e-9:
                    cols[:, band] = np.clip(v / top, 0, 1)
        cols = 0.16 + 0.84 * cols                          # a floor, so an inactive cell is still a cell
        return np.clip(cols, 0, 1), hi
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
            ax.scatter(pos[:, 0], pos[:, 1], s=6.0,
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
