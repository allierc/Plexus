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
        if pos is None or len(pos) == 0:
            return None
        D = pos.shape[1]
        os.makedirs(out_dir, exist_ok=True)
        fig = plt.figure(figsize=(6.2, 6.2), facecolor="black")
        if D >= 3:
            ax = fig.add_subplot(111, projection="3d", facecolor="black")
            r = float(np.abs(pos[:, :3]).max()) * 1.05 or 1.0
            if m is not None:
                # THE SURFACE, not a point cloud: the ring of each face, drawn as a polygon. A cloud
                # of 20,000 vertices reads as a fuzzy ball whatever the tissue is doing, which is
                # the one thing a live view must not do.
                from plexus.models.topology import rings_from_flat_3d
                es, et, ef = (np.asarray(_np(m[k])) for k in ("E_srce", "E_trgt", "E_face"))
                rings = rings_from_flat_3d(es, et, ef, int(m["nF"]))
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                polys = [pos[np.asarray(rg, int), :3] for rg in rings if len(rg) >= 3]
                pc = Poly3DCollection(polys, facecolor="#dcdcdc", edgecolor="#5a5a5a",
                                      linewidths=0.15, alpha=1.0)
                ax.add_collection3d(pc)
            else:
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1.0, c="#dcdcdc", depthshade=False)
            ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
            ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
            put = ax.text2D
        else:
            ax = fig.add_subplot(111, facecolor="black")
            ax.scatter(pos[:, 0], pos[:, 1], s=1.0, c="#dcdcdc")
            ax.set_aspect("equal"); ax.set_axis_off()
            put = ax.text
        n_live = (int(m["nF"]) if m is not None else len(pos))
        unit = "cells" if m is not None else "elements"
        put(0.02, 0.97, f"{name}  frame {tick} / {n_frames}   {n_live} {unit}   [live]",
            transform=ax.transAxes, color="white", fontsize=11, va="top")
        fig.subplots_adjust(0, 0, 1, 1)
        # `.tmp` LAST WOULD PICK THE FORMAT FROM THE EXTENSION and matplotlib refuses "tmp"; the
        # temporary name has to keep the `.png` suffix.
        tmp = os.path.join(out_dir, ".3d.partial.png")
        fig.savefig(tmp, dpi=110, facecolor="black")
        plt.close(fig)
        os.replace(tmp, os.path.join(out_dir, "3d.png"))     # atomic: never a half-written PNG
        return os.path.join(out_dir, "3d.png")
    except Exception as e:
        print(f"[live] snapshot at frame {tick} skipped ({type(e).__name__}: {str(e)[:70]})",
              flush=True)
        return None


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
