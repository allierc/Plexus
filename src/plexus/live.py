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


def cutaway_mask(pos, cutaway):
    """Which elements to KEEP -- True everywhere unless `cutaway` names a wedge to drop.

    THE PARTICLE TWIN OF THE MESH CUTAWAY in `render_vtk.mesh_of`, and it exists because a FILLED
    body has no faces to drop. An epithelium is a closed surface: removing a wedge of its faces
    opens a window onto the far wall. A solid spheroid of cells is a point cloud, and the only way
    to see its interior is to remove the points that are in front of it.

    `cutaway: ["x", "y"]` drops the quadrant where both coordinates exceed the cloud's own
    CENTROID -- its own, not the world origin, because a growing aggregate drifts and a cut taken
    against a fixed origin would slide off the body. One axis halves it, two take a quadrant,
    three an octant. A leading "-" flips that axis.
    """
    q = np.asarray(pos, float)
    if not cutaway or q.ndim != 2 or q.shape[1] < 2:
        return None
    ax = {"x": 0, "y": 1, "z": 2}
    sel = [(ax[a.lower().lstrip("+-")], a.startswith("-"))
           for a in cutaway if a.lower().lstrip("+-") in ax and ax[a.lower().lstrip("+-")] < q.shape[1]]
    if not sel:
        return None
    c = q.mean(0)
    inside = np.ones(len(q), bool)
    for i, neg in sel:
        inside &= (q[:, i] < c[i]) if neg else (q[:, i] > c[i])
    return ~inside

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


def chem_rgb(chem, nF=None, lut=None, blend=None, background="black"):
    """(rgb [n,3], [max per drawn column]) from a `chem` array -- THE ONE COLOUR LAW for RD output.

    Lives here and is imported by `plexus.plot`, because the alternative is two copies that drift:
    the live snapshot and the movie of one run showed DIFFERENT PICTURES of the same numbers for
    exactly that reason -- `movie_cell.mp4` never read `chem` at all and drew every cell in tab10
    blue. Pure numpy, no matplotlib, so either caller can use it.

    `lut` IS THE COLUMN-TO-COLOUR TABLE AND IT BELONGS IN THE SPEC. Which columns are drawn, and in
    what colour, is a property of the MODEL, not of the renderer: a Gray-Scott pair wants its
    activator drawn and its substrate usually not, while May-Leonard's three species are a
    partition and want red/green/blue. Hardcoding "columns 0 and 2 are activators" made the second
    correct only by accident and the third impossible. `lut` is positional -- one entry per chem
    column, `None` meaning "declared and deliberately not drawn", which is a statement rather than
    the silent consequence of a loop that only visits even columns.

    `blend` MUST BE DECLARED because the right one depends on what the columns mean.
      subtractive  -- white is quiet and each species takes light away. Right for Gray-Scott: the
                      activator is a sparse figure on a quiet ground, and starting from black made
                      an inactive cell invisible on a black canvas.
      additive     -- black is quiet and each species adds its colour. Right for a PARTITION like
                      May-Leonard, where u+v+w is conserved and the barycentric mix is the honest
                      picture -- it is what ParticleGraph's RPS renders.
    Default follows `background`: subtractive on a dark canvas is the Gray-Scott look this started
    with, so an absent `blend` keeps every existing plot unchanged.

    Each drawn column is normalised by its OWN maximum -- see the caller for why per-frame.
    """
    a = np.asarray(chem, float)
    if a.ndim != 2 or a.shape[1] < 1:
        return None, []
    n = a.shape[0] if nF is None else int(nF)
    a = a[:n]
    ncol = a.shape[1]

    if lut is None:                        # the historical default: activators of up to two pairs
        lut = [None] * ncol
        for c, col in ((0, "#ff2b2b"), (2, "#2b6bff")):
            if ncol > c:
                lut[c] = col
    lut = list(lut) + [None] * max(0, ncol - len(lut))
    blend = (blend or "subtractive").lower()

    def _rgb(h):
        h = str(h).lstrip("#")
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

    drawn = [(k, _rgb(lut[k])) for k in range(ncol) if lut[k] is not None]
    if not drawn:
        return None, []
    hi, norm = [], []
    for k, _c in drawn:
        v = np.where(np.isfinite(a[:, k]), a[:, k], 0.0)
        top = float(v.max())
        hi.append(top)
        norm.append(np.clip(v / top, 0, 1) if top > 1e-9 else np.zeros(n))

    if blend == "additive":
        cols = np.zeros((n, 3))
        for (k, c), w in zip(drawn, norm):
            cols += w[:, None] * c[None, :]
    else:
        # subtractive: start white, each species removes the light its own colour lacks. The
        # floor stops short of 0 so a cell carrying every species stays a dark dot rather than a
        # hole in the canvas -- and that cell is the most interesting one on the plot.
        f = 0.86
        cols = np.ones((n, 3))
        for (k, c), w in zip(drawn, norm):
            cols = cols - f * w[:, None] * (1.0 - c)[None, :]
    return np.clip(cols, 0, 1), hi


def _face_colours(H, nF, style=None):
    """`chem_rgb` for the first set that carries a `chem` block, using the spec's own LUT."""
    style = style or {}
    for _s, l in H.levels.items():
        if "chem" not in getattr(l, "state_schema", {}):
            continue
        c = l.get("chem")
        if c is None or c.shape[0] < nF:
            continue
        return chem_rgb(c.detach().cpu().numpy(), nF,
                        lut=style.get("species"), blend=style.get("blend"),
                        background=style.get("background", "black"))
    return None, []


def snapshot(H, tick, n_frames, out_dir, name="", sname=None, style=None):
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
                fc, _clim = _face_colours(H, int(m["nF"]), style)
                pc = Poly3DCollection(
                    polys, facecolor=(fc[keep] if fc is not None else "#dcdcdc"),
                    edgecolor="#5a5a5a", linewidths=0.15, alpha=1.0)
                ax.add_collection3d(pc)
            else:
                _k = cutaway_mask(pos[:, :3], (style or {}).get("cutaway"))
                _p = pos if _k is None else pos[_k]
                _fc, _clim = _face_colours(H, len(pos), style)
                if _fc is not None and _k is not None:
                    _fc = _fc[_k]
                ax.scatter(_p[:, 0], _p[:, 1], _p[:, 2], depthshade=False,
                           s=dot_area_pt2(_p[:, :3], 2.0 * r or 1.0, 6.2 * 110, 110),
                           c=(_fc if _fc is not None else "#dcdcdc"), linewidths=0)
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
            fc, _clim = _face_colours(H, len(pos), style)
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
                # NOT `'AB'[i]`: that was written when two species was the only case and it
                # raised IndexError on the first three-species run, which `snapshot`'s catch-all
                # turned into no picture at all.
                f"{chr(65 + i) if i < 26 else i} max {v:.4g}" for i, v in enumerate(_clim)),
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
