#!/usr/bin/env python
"""Two panels for a composed cell: instance segmentation in 3D, and the same in cross section.

WHY A CROSS SECTION IS NOT OPTIONAL. A filled body is opaque. Rendered from outside, a cell whose
interior is doing something interesting is a featureless ball, and the entire claim of the
composition -- that several distinct substrates coexist inside one parent -- is invisible. The 3D
panel says the cell is there; the section says what it is made of.

INSTANCE SEGMENTATION MEANS ONE COLOUR PER SET, and the colours come from the SPEC. Which object is
drawn in which colour is a property of the model, not of the renderer -- the same argument that put
the reaction-diffusion colour table into `plotting.species`. A renderer that picks its own colours
makes two runs of different models look comparable when they are not.

    plotting:
      colors: {light: [0.25, 0.69, 1.0], heavy: [0.12, 0.31, 0.66], n: [1.0, 0.82, 0.29]}

    python tools/cell_panels.py <run_dir> [--frame -1] [--axis z] [--thick 0.06]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

AX = {"x": 0, "y": 1, "z": 2}


def _load(run_dir):
    """(spec, trajectory) for a generated run, or a clear failure saying which is missing."""
    import yaml
    npz = os.path.join(run_dir, "trajectory.npz")
    spec = os.path.join(run_dir, "spec.yaml")
    for f in (npz, spec):
        if not os.path.exists(f):
            raise SystemExit(f"  missing {f} -- run `-o generate` first")
    return yaml.safe_load(open(spec)), np.load(npz, allow_pickle=True)


def _sets(spec, z):
    """The child sets that have recorded positions, with their per-particle colours.

    A set is drawn only if BOTH the spec declares it and the trajectory recorded it. A set declared
    and not recorded is reported rather than skipped: silently drawing three sets when the spec has
    four is exactly the failure a segmentation panel exists to catch.
    """
    # THE EXISTING KEY, `plotting.colors`, indexed by TYPE name -- the one `_typed_palette`
    # already reads. There was no need to invent a per-SET table: the cytosol's two species and the
    # nucleus have distinct type names, so one flat mapping colours every set in the composition.
    lut = ((spec.get("plotting") or {}).get("colors")) or {}
    out, missing = [], []
    for name in (spec.get("sets") or {}):
        key = f"{name}__pos"
        types = list(((spec["sets"][name] or {}).get("types") or {}).keys())
        if key not in z.files:
            if any(t in lut for t in types):
                missing.append(name)
            continue
        pos = np.asarray(z[key])
        if pos.ndim != 3 or pos.shape[2] < 3:
            continue
        nt = np.asarray(z[f"{name}__node_type"]) if f"{name}__node_type" in z.files else None
        if types and nt is not None:
            per = [_rgb(lut.get(t, "#888888")) for t in types]
            rgb = np.array([per[min(int(k), len(per) - 1)] for k in nt])
            label = f"{name}: " + ", ".join(types)
        else:
            rgb = np.tile(_rgb(lut.get(name, "#cccccc")), (pos.shape[1], 1))
            label = name
        # THE PARENT IS NOT DRAWN. `cell` is one point at the centre of the composition -- it has
        # no extent, it is hidden inside everything else, and a legend entry for it labels a mark
        # nobody can see. Its children are what the picture is of.
        if pos.shape[1] <= 1:
            continue
        out.append((name, pos, rgb, label))
    return out, missing


def _rgb(c):
    """A colour as [r,g,b] in 0..1, from either an RGB list or a hex string.

    `plotting.colors` in the material specs holds lists; the reaction-diffusion LUT holds hex. Both
    spellings are already in the corpus, so both are read rather than one being declared correct.
    """
    if isinstance(c, (list, tuple)):
        return np.asarray([float(x) for x in c[:3]])
    h = str(c).lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _box(spec):
    """The world box as (lo, hi), or None when the spec declares no world."""
    w = (spec.get("general") or {}).get("world")
    if not w:
        return None
    hi = np.asarray([float(x) for x in (w if isinstance(w, (list, tuple)) else [w])], float)
    if hi.size < 3:
        hi = np.concatenate([hi, np.ones(3 - hi.size)])
    return np.zeros(3), hi[:3]


def _draw_box3d(ax, lo, hi, c="#4a4a4a", lw=0.7):
    """The twelve edges of the domain.

    WHY IT IS DRAWN AT ALL. Without it a point cloud in an empty frame has no orientation and no
    floor: a body falling straight down the y axis was read as drifting diagonally, and the surface
    it landed on was read as a diagonal wall, because nothing in the picture said which way was
    down. The trajectory was correct throughout -- dx and dz were zero to four decimals. A frame of
    reference is not decoration when the thing being judged is a direction.
    """
    import itertools
    for a, b in itertools.combinations(list(itertools.product(*zip(lo, hi))), 2):
        if sum(1 for i in range(3) if a[i] != b[i]) == 1:          # an edge: differs on one axis
            ax.plot(*zip(a, b), color=c, lw=lw)

def panels(run_dir, frame=-1, axis="z", thick=0.06, out=None, fill3d=1.9, fill2d=1.1,
           frame_to="world", quiet=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.live import dot_area_pt2

    spec, z = _load(run_dir)
    sets, missing = _sets(spec, z)
    if not sets:
        raise SystemExit("  no set has recorded positions")
    k = AX[axis.lower()]

    # ONE FRAME, ONE CAMERA BOX, SHARED BY BOTH PANELS. Framing each panel to its own contents would
    # make the section look like a different, larger object than the body it was cut from.
    allp = np.concatenate([p[frame] for _n, p, _c, _l in sets])
    box = _box(spec)
    # THE TWO PANELS FRAME DIFFERENTLY, ON PURPOSE, because they answer different questions.
    #
    # LEFT is the domain: where the cell IS. A body framed to itself looks identical at the top of
    # its fall and resting on the floor, and the box is what makes "down" and "which wall" legible
    # at all -- without it a straight drop was read as a diagonal drift for two rungs running.
    #
    # RIGHT is the body: what the cell is MADE of. Framed to the domain the section shrinks to a
    # speck in a corner and the compartments it exists to show are a few pixels across. Giving both
    # panels one framing, which is what the first version did, means one of them is always wrong.
    wc = 0.5 * (box[0] + box[1]) if box is not None else allp.mean(0)
    wr = float((box[1] - box[0]).max()) * 0.52 if box is not None else 1.0
    bc = allp.mean(0)
    br = float(np.abs(allp - bc).max()) * 1.10 or 1.0
    if frame_to == "body":                       # both panels on the body, for a structural rung
        wc, wr = bc, br
    c3, r = wc, wr

    fig = plt.figure(figsize=(12.6, 6.4), facecolor="black")
    ax1 = fig.add_subplot(121, projection="3d", facecolor="black")
    ax2 = fig.add_subplot(122, facecolor="black")

    # THE SLAB FOLLOWS THE BODY, not the domain: a section taken about the middle of the box
    # misses a cell that has fallen to the floor entirely.
    lo, hi = bc[k] - thick * br, bc[k] + thick * br
    o1, o2 = [i for i in range(3) if i != k]

    for _name, pos, rgb, label in sets:
        p = pos[frame]
        # A PROJECTED VOLUME NEEDS FATTER DOTS THAN A FLAT SHEET, and `fill` is where that goes.
        # `dot_area_pt2` measures the median NEAREST-NEIGHBOUR distance, which in a dense 3D cloud
        # is the spacing to the neighbour in FRONT of or BEHIND a particle as often as beside it.
        # Sized at that spacing the body renders as a haze of specks: you only ever see its front
        # layer, and that layer is sparse in projection. `fill` near 2 makes the front layer close
        # over, so the cell reads as a solid object -- which is what it is. A mplot3d axes also
        # draws its data into well under the full axes width, which the exact px conversion cannot
        # know, and the same factor absorbs it.
        ax1.scatter(p[:, 0], p[:, 1], p[:, 2], c=rgb, depthshade=False, linewidths=0,
                    s=dot_area_pt2(p, 2.0 * r, 6.3 * 110, 110, fill=fill3d), label=label)
        m = (p[:, k] >= lo) & (p[:, k] <= hi)                 # the slab
        if m.any():
            q = p[m]
            # THE SECTION IS NEARLY A SHEET, so it wants close to the flat-disc value -- the slab
            # is thin and the spacing measured in it is the spacing you actually see.
            ax2.scatter(q[:, o1], q[:, o2], c=rgb[m], linewidths=0,
                        s=dot_area_pt2(q[:, [o1, o2]], 2.0 * br, 6.3 * 110, 110, fill=fill2d))

    if box is not None:
        _draw_box3d(ax1, box[0], box[1])
        # the domain's own cross section at this slab: the rectangle the section is cut from
        from matplotlib.patches import Rectangle
        # the domain edge, drawn only where it falls inside the zoomed section -- when the cell
        # is resting on the floor its lower wall is in view and is worth seeing.
        ax2.add_patch(Rectangle((box[0][o1], box[0][o2]),
                                box[1][o1] - box[0][o1], box[1][o2] - box[0][o2],
                                fill=False, edgecolor="#4a4a4a", lw=0.7))
    for a in (ax1,):
        a.set_xlim(c3[0] - r, c3[0] + r); a.set_ylim(c3[1] - r, c3[1] + r)
        a.set_zlim(c3[2] - r, c3[2] + r); a.set_axis_off(); a.set_box_aspect((1, 1, 1))
    ax2.set_xlim(bc[o1] - br, bc[o1] + br); ax2.set_ylim(bc[o2] - br, bc[o2] + br)
    ax2.set_aspect("equal"); ax2.set_axis_off()

    name = (spec.get("general") or {}).get("name", os.path.basename(run_dir))
    n_rows = sets[0][1].shape[0]
    fr = frame if frame >= 0 else n_rows + frame
    ax1.text2D(0.02, 0.97, f"{name}   frame {fr + 1}/{n_rows}   domain",
               transform=ax1.transAxes, color="white", fontsize=11, va="top")
    ax2.text(0.02, 0.97, f"cell, zoomed   cross section {axis} = body centre +/- {thick:.2f} R",
             transform=ax2.transAxes, color="white", fontsize=11, va="top")
    counts = "   ".join(f"{n}: {p.shape[1]}" for n, p, _c, _l in sets)
    ax2.text(0.02, 0.93, counts, transform=ax2.transAxes, color="#b0b0b0", fontsize=8, va="top")
    if missing:
        # LOUD, because a set that is declared and not recorded is the one thing this panel is
        # supposed to reveal, and a silently absent colour looks identical to a set that is simply
        # hidden behind another.
        ax2.text(0.02, 0.05, f"DECLARED BUT NOT RECORDED: {', '.join(missing)}",
                 transform=ax2.transAxes, color="#ff5555", fontsize=9)
    # NO FRAME AND NO FILL. A boxed legend on a black canvas draws a bright rectangle over the
    # body it is labelling, and the labels are legible against black without one.
    leg = ax1.legend(loc="lower left", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color("white")

    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.02)
    out = out or os.path.join(run_dir, "panels.png")
    fig.savefig(out, dpi=120, facecolor="black")
    plt.close(fig)
    if not quiet:
        print(f"  {out}")
        for n, p, _c, _l in sets:
            print(f"    {n:<10} {p.shape[1]:>6} elements")
    if missing and not quiet:
        print(f"    DECLARED BUT NOT RECORDED: {missing}")
    return out


def panels_movie(run_dir, axis="z", thick=0.10, out=None, max_frames=150, fps=20,
                 fill3d=1.9, fill2d=1.1):
    """The two-panel view over TIME -- the one movie that replaces the per-set pile.

    A STILL OF THE LAST FRAME ANSWERS NOTHING ABOUT A RUN THAT MOVES. `cell_02` is a cell dropped
    from a height: its final frame is a ball resting on the floor, which is equally consistent with
    falling, bouncing, splashing or never having moved. The whole reason for the ladder is what the
    compartments DO to each other on impact, and that is a sequence.

    SUBSAMPLED, because 501 recorded rows at two matplotlib panels each is minutes of rendering for
    a clip nobody watches frame by frame. `max_frames` evenly spaced rows is enough to read a
    bounce, and the frame counter in the corner says which row each panel is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import tempfile

    spec, z = _load(run_dir)
    sets, _missing = _sets(spec, z)
    if not sets:
        raise SystemExit("  no set has recorded positions")
    n_rows = sets[0][1].shape[0]
    idx = np.unique(np.linspace(0, n_rows - 1, min(max_frames, n_rows)).astype(int))
    out = out or os.path.join(run_dir, "panels.mp4")
    tmp = tempfile.mkdtemp()
    frames = []
    for k, t in enumerate(idx):
        f = os.path.join(tmp, f"{k:05d}.png")
        panels(run_dir, frame=int(t), axis=axis, thick=thick, out=f,
               fill3d=fill3d, fill2d=fill2d, quiet=True)
        frames.append(f)
    with imageio.get_writer(out, fps=fps, quality=8) as w:
        for f in frames:
            w.append_data(imageio.imread(f))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {out}   ({len(idx)} of {n_rows} rows @ {fps} fps)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--axis", default="z", help="the axis the section is taken across")
    ap.add_argument("--thick", type=float, default=0.06, help="slab half-thickness, as a fraction "
                                                              "of the body radius")
    ap.add_argument("--fill3d", type=float, default=1.9,
                    help="dot size as a multiple of the measured spacing, 3D panel")
    ap.add_argument("--fill2d", type=float, default=1.1, help="the same, section panel")
    ap.add_argument("--frame-to", dest="frame_to", default="world",
                    choices=("world", "body"),
                    help="frame on the domain (default) or zoom to the contents")
    ap.add_argument("--movie", action="store_true", help="render the two-panel view over time")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    (panels_movie if a.movie else panels)(
        a.run_dir, axis=a.axis, thick=a.thick, out=a.out,
        fill3d=a.fill3d, fill2d=a.fill2d,
        **({} if a.movie else dict(frame=a.frame, frame_to=a.frame_to)))
