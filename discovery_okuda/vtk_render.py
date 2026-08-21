#!/usr/bin/env python
"""The VTK renderer: a z-buffered, GPU, lit picture of the vesicle -- stills, rotations and movies.

Cedric, 12 August: *"ok c2 and c3 are gorgeous, we go VTK. Could you make two kburns and two mp4
evolve with c2 and c3 in b_star."*

WHY VTK REPLACES `_draw`, stated as a defect and not a preference. `mpl_toolkits.mplot3d` has no
depth buffer: it sorts polygons by mean z and paints back to front, which is only exact when no two
polygons overlap in depth order -- and a closed cellular body is the worst case for it. Measured on
b_star's end frame, 6,124 of 12,272 apical faces point away from the camera at azimuth 310 and are
drawn anyway, so which far-side face wins a tie changes with the angle, and one surface is drawn
two different ways at 0:12 and 0:14 of a single rotation. VTK discards a fragment behind another
per pixel, so the question cannot arise. It is also 29x faster on this mesh: 0.32 s a frame against
9.33 (log/okuda/b_star/render_compare.png).

SHADING IS ALWAYS SMOOTH. Cedric, 12 August: *"I do not want flat at all."* Flat shading gives
every cell one normal, so a curved arm reads as a faceted cone and the surface's own curvature is
lost; the only thing it buys is a stronger sense of the mesh, and the mesh has its own switch:

    mesh     smooth-shaded surface WITH the cell outlines drawn on it. Read this when the question
             is about cells -- who divided, how big, how many across a tube.
    nomesh   the same surface with no outlines. Read this when the question is about SHAPE: an arm
             reads as a round tube with light running down its length, and nothing competes with
             the silhouette.

SEQUENCES, so a caller asks for a job and not for four commands:

    0   kburns, no mesh -- ONE clip, and what the loop writes for every run
    1   evolve, with mesh
    2   evolve + kburns, with mesh
    3   evolve + kburns without mesh, then evolve + kburns with mesh -- all four

THE COLOUR SEMANTICS carried over from `_draw`, and what each one means:

    white -> red   the activator. This is the measurement.
    magenta        non-finite: not a cell any more.
    green          a cell that has just divided.
    blue           where the SECOND field acted -- marked a cell to die, or switched its growth
                   off. Not how much of that field there is: `traj.npz` carries one activator
                   channel, so this is an action and a card claiming a concentration would be
                   claiming data we do not have.

See `_marks` for the two rules that are not obvious: division needs `ndiv > 0` as well as a young
`age`, or an untouched tissue is green in its opening frames; and inhibition is drawn only where
the activator is LOW, because it covers 4,222 of 4,232 cells in `sc_inh_soft` and painting it
outright erases the pattern the picture is about.

    python vtk_render.py b_star                  sequence 0: the turning nomesh clip, what the
                                                 loop writes for every run
    python vtk_render.py b_star --seq 2          evolve + kburns with the mesh, for cell questions
    python vtk_render.py b_star --seq 3          all four, for comparing styles
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "log", "okuda")
for _p in (HERE, os.path.join(ROOT, "discovery_okuda", "ops"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# EGL BEFORE VTK IS IMPORTED, or VTK opens an X window, finds the devcontainer's stub display and
# silently falls back to a software rasteriser. The picture is identical and the speed is not.
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

CAM = dict(elev=18.0, azim=30.0)
# 896 = 56 x 16, NOT 900. ffmpeg's macro_block_size is 16 and imageio silently resamples anything
# else -- 900 was being stretched to 912, so every frame of a renderer chosen for PRECISION was
# resampled on the way out. The warning said so and is easy to read past.
SIZE = 896
FPS = 25
KB_SECONDS = 18.0          # one revolution; see kburns_render.SECONDS for why the length IS the speed
KB_ZOOM = 0.55
EV_FPS = 12                # the archive holds ~60 recorded frames; 12 fps makes that a 5 s clip

# THE NAMED JOBS. A caller asks for a sequence, not for four commands, so what a run produces is one
# number in one place and the loop and the command line cannot drift apart.
SEQUENCES = {
    0: [("kburns", "nomesh")],
    1: [("evolve", "mesh")],
    2: [("evolve", "mesh"), ("kburns", "mesh")],
    3: [("evolve", "nomesh"), ("kburns", "nomesh"),
        ("evolve", "mesh"), ("kburns", "mesh")],
    # THE ONE THE SURROGATE WANTS. Cedric, 13 August, looking at a strip: *"we should just use first
    # row... why not using vtk_evolve_nomesh.mp4."* An embedding of the tissue wants the SHAPE
    # THROUGH TIME and nothing else -- no mesh strokes (they smear to a dark field at 20k cells), no
    # turn-on-the-spot, and none of the strip's other three rows, of which one is a second viewpoint,
    # one is a per-frame contrast stretch of radius, and one is a cross-section with a hardcoded
    # hollow centre. Seq 1 is the nearest existing entry and draws the mesh; this is seq 1 without it.
    4: [("evolve", "nomesh")],
}
# 0, NOT 3. Cedric, 12 August: *"do not write four vtk mp4 in each folder but just the
# vtk_kburns_nomesh.mp4."* Sequence 3 was chosen when the question was WHICH style to use; that is
# settled, and four clips a run is 15 a round and several hundred a campaign, of which three are
# never opened. The turning nomesh clip is the one that answers "what shape is this", which is what
# a run is looked at for.
#
# The other sequences stay, because they are still the right answer by hand -- `--seq 2` when the
# question is about CELLS, `--seq 3` when two styles need comparing again -- and any of them
# rebuilds from `traj.npz` in seconds without re-simulating.
LOOP_SEQ = 0               # what run_one asks for


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wr", ["white", "#d62728"])


def frames_of(run, traj=None):
    """Every recorded (pos, mesh, act) of a finished run, from the archive `movie.mp4` uses.

    `traj` overrides the file, so a MID-RUN snapshot -- one frame written in the same format by
    `run_one._live_snapshot` -- renders through exactly this path and needs no special case.
    """
    p = traj or os.path.join(LOG, run, "traj.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    n = sum(1 for k in z.files if k.startswith("pos_"))
    out = []
    for t in range(n):
        mt = z[f"mesh_{t}"]
        mt = mt.item() if hasattr(mt, "item") else mt
        act = z[f"act_{t}"] if f"act_{t}" in z.files else None
        out.append((np.asarray(z[f"pos_{t}"], float), mt, act))
    return out


def box_of(run, fr):
    """The run's own fixed camera box -- shared with every other picture of it."""
    dj = os.path.join(LOG, run, "diag.json")
    if os.path.exists(dj):
        try:
            L = (json.load(open(dj)).get("summary") or {}).get("camera_lbox")
            if L:
                return float(L)
        except Exception:
            pass
    from run_one import run_box
    return float(run_box([(p, m, a, None) for p, m, a in fr]))


# THE TWO MARKS, and what they are read from. Neither is a second colour SCALE -- they are states
# the archive already carries per cell, and they overwrite the activator colour where they are set:
#   green   a cell that has just divided: `age` <= DIVIDED (reset to 0 by cell_divide) AND `ndiv` > 0,
#           because `age` alone starts at 0 for every seeded cell and paints an untouched tissue
#           green in the opening frames.
#   blue    the second morphogen's ACTION on the cell -- `apop` where it marks the cell to die,
#           `inhib` where it switches the cell's growth off. The field's concentration is not in
#           `traj.npz` (one activator channel is), so this is where the second field acted and not
#           how much of it there is; a card that says otherwise would be claiming data we do not have.
DIVIDED = 4
GREEN = (44, 160, 44)
GREEN_A = 0.5              # how much of the cell's own colour the division mark keeps
BLUE = (31, 119, 180)


def _marks(mt, idx, nF):
    """(divided, kills, suppresses) masks over the drawn faces, or None where unrecorded.

    THE SECOND FIELD IS TWO DIFFERENT MARKS and they cannot share a rule. `apop` is rare and is the
    event -- a cell sentenced to die -- so it takes the colour outright. `inhib` is nearly the whole
    tissue (4,222 of 4,232 cells in `sc_inh_soft`), so painting it outright erases the activator
    pattern the card is about: the run's own caption says "red spots inside teal that does not
    grow", and a solid blue ball says nothing at all. It is drawn only where the activator is LOW,
    which is where "does not grow" means something.
    """
    def col(k):
        """One per-cell state over the drawn faces, PADDED rather than dropped when it is short.

        THE FLICKER THIS FIXES. A state is recorded before the frame's divisions are applied, so on
        the frames where cells divided it is a few entries shorter than `nF` -- 3,867 against 3,884,
        4,222 against 4,232. Dropping the whole mask on those frames, which is what returning None
        did, made the blue vanish and come back every second or third frame of `sc_inh_soft`: not
        the simulation changing its mind, an index guard throwing away 99.6% of a mask because
        0.4% of it was missing. The cells the state does not cover are the ones just created, and
        an unmarked new cell is the truthful default.
        """
        v = mt.get(k)
        if v is None:
            return None
        a = np.asarray(v).ravel()
        if a.size < nF:
            a = np.concatenate([a, np.zeros(nF - a.size, a.dtype)])
        return a[:nF][idx]
    age, ndiv = col("age"), col("ndiv")
    div = None if age is None or ndiv is None else ((age <= DIVIDED) & (ndiv > 0))
    kills, sup = col("apop"), col("inhib")
    return div, (None if kills is None else kills > 0), (None if sup is None else sup > 0)


def mesh_of(pos, mt, act, lo=None, hi=None, show_div=True):
    """The apical shell as PolyData with per-cell RGB. Rebuilt per frame: cells divide."""
    import pyvista as pv
    from topology_ops import rings_from_flat_3d
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    rings = rings_from_flat_3d(es[live], et[live], ef[live], nF)
    faces, idx = [], []
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        faces.append(len(r)); faces.extend(int(v) for v in r); idx.append(f)
    if not idx:
        return None
    m = pv.PolyData(pos, faces=np.asarray(faces, np.int64))
    if act is None:
        rgb = np.full((len(idx), 3), 235, np.uint8)
    if act is not None:
        a = np.asarray(act, float)[:nF][idx]
        ok = np.isfinite(a)
        # THE RANGE IS THE RUN'S, NOT THE FRAME'S, on a movie -- otherwise every frame renormalises
        # and a pattern that is strengthening looks static, which is the same defect as the camera
        # autofit that hid growth.
        _lo = float(np.nanmin(a)) if lo is None else lo
        _hi = float(np.nanmax(a)) if hi is None else hi
        x = np.clip((a - _lo) / (_hi - _lo + 1e-9), 0, 1)
        rgb = (np.asarray(_cmap()(x))[:, :3] * 255).astype(np.uint8)
        rgb[~ok] = (255, 26, 217)                 # magenta: not a cell any more
    div, kills, sup = _marks(mt, idx, nF)
    # ORDER MATTERS: suppression is the background, death is the event, division is the rarest and
    # shortest-lived (four calls), so each later mark may overwrite the one before it.
    if sup is not None and sup.any():
        # A BLEND, NOT A THRESHOLD. `x < 0.5` on a smooth activator field toggles every cell that
        # sits near the line, so whole regions flipped blue and back between frames -- the same
        # flicker as the dropped mask, from the other direction. Weighting by how LOW the activator
        # is makes a cell that is halfway halfway blue, and nothing jumps.
        w = (sup.astype(float) if act is None else sup * (1.0 - x)) [:, None]
        rgb = ((1.0 - w) * rgb.astype(float) + w * np.asarray(BLUE, float)).astype(np.uint8)
    # DEATH FOLLOWS DIVISION'S RULE, and for its reason: both name ONE CELL, so both need the cell
    # to be visible. On the smooth surface a blue patch with no outlines marks a cell nobody can
    # see. Suppression is not in this class -- it tints a whole REGION and reads with or without
    # the outlines -- so it stays in both styles.
    if show_div and kills is not None and kills.any():
        rgb[kills] = BLUE
    # GREEN ONLY WHERE BOTH ITS AXES EXIST. It marks ONE CELL that divided in the last four calls,
    # so it needs a before and an after -- which `kburns` does not have, being one frame held still
    # while the camera moves -- and it needs the cell to be visible, which `nomesh` does not give:
    # on a smooth surface with no outlines a green patch names a cell nobody can see. So it is drawn
    # in `evolve` + `mesh` and nowhere else. Blue is not subject to either: it marks a REGION, and a
    # region reads on a smooth surface and in a single frame.
    if show_div and div is not None and div.any():
        # A TINT, NOT A REPAINT. Solid green throws away what the cell was -- its activator level,
        # or the blue that says the second field is acting on it -- to say one bit: it just divided.
        # Blended at GREEN_A the cell keeps its own colour and is legibly green over it.
        rgb[div] = ((1.0 - GREEN_A) * rgb[div].astype(float)
                    + GREEN_A * np.asarray(GREEN, float)).astype(np.uint8)
    m.cell_data["rgb"] = rgb
    return m


def add(p, m, style):
    """Add the shell and RETURN THE ACTOR, which is the whole reason this is not `p.clear()`.

    `Plotter.clear()` removes every actor AND every light: measured 5 lights before, 0 after. So
    `evolve` rendered its first frame lit and all fifty-nine others with no light source at all --
    mean brightness 195.6 -> 238.9 and shading contrast 47.5 -> 33.5, a washed-out white body.
    Cedric spotted it as "the last frame is different from the first, the first is better with
    shading", which is exactly the frame-1-only-is-lit signature. Removing the one actor we added
    leaves the lighting rig alone.
    """
    return p.add_mesh(m, scalars="rgb", rgb=True, lighting=True,
                      # `flat` ADDED 16 AUGUST, and it is a reversal of the note at the top of this
                      # file, asked for by name. One normal per cell, so each cell is a facet and
                      # the mesh reads through the shading with no stroke drawn: an outline is a
                      # constant 0.4 px whatever the cell's size, which at montage scale merges into
                      # a grey wash, while a facet scales with the cell it belongs to.
                      # The cost stands as written: a curved arm reads as a faceted cone. `nomesh`
                      # remains the default and the loop's own choice for the shape question.
                      smooth_shading=(style != "flat"),
                      show_edges=(style == "mesh"), edge_color="black", line_width=0.4,
                      ambient=0.35, diffuse=0.75, specular=0.12, specular_power=18)


def aim(p, L, azim=None, elev=None, fill=1.0):
    """`fill` < 1 pulls the camera back: the run's box then occupies that fraction of the frame.

    WHY IT IS NEEDED AT ALL, given the box is the run's own. `L` is set by the widest thing the run
    ever reaches, so a star's box is its ARMS and its body looks small in it, while a run whose body
    IS the widest thing -- a sphere that only grows, a vesicle that only shrinks -- fills the card
    edge to edge and reads as a close-up. Same rule, different subject; the fraction is what makes
    two such cards the same size on a page.
    """
    e = np.radians(CAM["elev"] if elev is None else elev)
    a = np.radians(CAM["azim"] if azim is None else azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    p.camera.position = tuple(d * L * 3.2)
    p.camera.focal_point = (0, 0, 0)
    p.camera.up = (0, 0, 1)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = L * 1.05 / max(fill, 1e-3)


def _ease(u):
    return u * u * (3.0 - 2.0 * u)


def _plotter():
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(SIZE, SIZE))
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    return p


def kburns(run, style, out, fill=1.0):
    """The finished specimen, turned once and zoomed in. Geometry fixed, camera moving."""
    fr = frames_of(run)
    if not fr:
        return "no traj.npz"
    L0 = box_of(run, fr)
    pos, mt, act = fr[-1]
    m = mesh_of(pos, mt, act, show_div=False)
    n = int(KB_SECONDS * FPS)
    p = _plotter(); add(p, m, style)
    p.add_text(f"{run}  {style}", position="upper_left", font_size=11, color="white")
    p.open_movie(out, framerate=FPS, quality=8)
    for i in range(n):
        u = i / (n - 1)
        aim(p, L0 * (1.0 - (1.0 - KB_ZOOM) * _ease(u)),
            azim=CAM["azim"] + 360.0 * u,                 # a FULL turn, so the clip loops
            elev=CAM["elev"] + 12.0 * np.sin(np.pi * u), fill=fill)
        p.write_frame()
    p.close()
    return f"{n} frames"


def evolve(run, style, out, fill=1.0):
    """The run through time, camera nailed down -- the successor to movie.mp4."""
    fr = frames_of(run)
    if not fr:
        return "no traj.npz"
    L = box_of(run, fr)
    # ONE ACTIVATOR RANGE FOR THE WHOLE CLIP, taken over every recorded frame. Per-frame
    # normalisation would make a strengthening pattern look constant.
    vals = [np.asarray(a, float) for _p, _m, a in fr if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0
    p = _plotter()
    p.open_movie(out, framerate=EV_FPS, quality=8)
    actor = txt = None
    for t, (pos, mt, act) in enumerate(fr):
        m = mesh_of(pos, mt, act, lo, hi, show_div=(style == "mesh"))
        if m is None:
            continue
        if actor is not None:
            p.remove_actor(actor)                  # NOT p.clear(): that removes the lights too
        if txt is not None:
            p.remove_actor(txt)
        actor = add(p, m, style)
        txt = p.add_text(f"{run}  {style}   frame {t + 1}/{len(fr)}   {int(mt['nF'])} cells",
                         position="upper_left", font_size=11, color="white")
        aim(p, L, fill=fill)
        p.write_frame()
    p.close()
    return f"{len(fr)} frames"


def still(run, style="flat", out=None, fill=1.0, frame=-1, label=True, traj=None):
    """The last frame as ONE image -- the VTK successor to `3d.png`.

    `3d.png` IS THE MOST-READ PICTURE IN THIS PROJECT and it was the only one still drawn by
    matplotlib. Every montage tiles it, the forecast graph puts it in each node, and `Read` opens it
    -- while the movies moved to VTK months ago for a reason this still inherits: matplotlib sorts
    faces by depth (painter's algorithm) and draws back-facing ones anyway, so on a star's end frame
    thousands of hidden faces are painted and which one wins a tie depends on the angle. A z-buffer
    cannot have that argument. The same mesh renders about 29x faster and lights the frame properly:
    28.9% of pixels lit against 4.5%.

    FLAT BY DEFAULT HERE, and only here. `nomesh` stays the loop's choice for the movies, where the
    question is the silhouette and a faceted cone would lose the curvature. A still that is going to
    be tiled at ~190 px is a different question: at that size a 0.4 px outline is a grey wash and
    smooth shading is a smooth blob, while facets scale with the cells they belong to, so the mesh
    is legible in a thumbnail without drawing a single line.
    """
    fr = frames_of(run, traj)
    if not fr:
        return "no traj.npz"
    L = box_of(run, fr)
    pos, mt, act = fr[frame][:3]
    m = mesh_of(pos, mt, act, show_div=(style == "mesh"))
    p = _plotter()
    add(p, m, style)
    if label:
        p.add_text(f"{run}  frame {len(fr) - 1 if frame == -1 else frame}",
                   position="upper_left", font_size=11, color="white")
    aim(p, L, fill=fill)
    p.screenshot(out or os.path.join(LOG, run, "3d.png"))
    p.close()
    return f"{len(fr)} frames, drew {frame}"


def render_all(run, seq=LOOP_SEQ, size=None, quiet=False, fill=1.0):
    """Run one named sequence for one finished run. Returns {filename: seconds}, {} if it cannot.

    CALLED BY THE LOOP AS WELL AS BY HAND, which is why it is a function and not just a `main`.
    Sequence 3 -- four clips, 1,020 frames -- measured at about 20 s on b_star's 12,272 cells,
    against a run that costs 30-40 MINUTES of GPU. That is the budget test Cedric set ("3 in the
    loop if it is less than one minute") and it passes with room to spare, so the loop asks for the
    full set rather than a reduced one.
    """
    global SIZE
    if size:
        SIZE = size
    d = os.path.join(LOG, run)
    if not os.path.exists(os.path.join(d, "traj.npz")):
        return {}
    took = {}
    for kind, st in SEQUENCES[int(seq)]:
        out = os.path.join(d, f"vtk_{kind}_{st}.mp4")
        t0 = time.perf_counter()
        msg = (kburns if kind == "kburns" else evolve)(run, st, out, fill=fill)
        dt = time.perf_counter() - t0
        if msg == "no traj.npz":
            return {}
        took[os.path.basename(out)] = dt
        if not quiet:
            print(f"  {kind:7s} {st:7s} {msg:12s} {dt:7.1f} s -> {os.path.relpath(out, ROOT)}")
    return took


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seq", type=int, default=LOOP_SEQ, choices=sorted(SEQUENCES),
                    help="0 = the turning nomesh clip (default, what the loop writes); "
                         "1 = evolve+mesh; 2 = +kburns; 3 = all four")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--fill", type=float, default=1.0,
                    help="fraction of the frame the run's own box fills; <1 pulls the camera back")
    a = ap.parse_args()
    t0 = time.perf_counter()
    took = render_all(a.run, a.seq, a.size, fill=a.fill)
    if not took:
        print(f"{a.run}: no traj.npz -- nothing to render"); return 1
    print(f"\n  sequence {a.seq}: {len(took)} clips in {time.perf_counter() - t0:.1f} s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
