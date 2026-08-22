"""The VTK renderer, promoted: a z-buffered, GPU, lit picture of a mesh set.

WHY VTK IS THE ENGINE'S RENDERER, stated as a defect and not a preference. `mpl_toolkits.mplot3d`
has no depth buffer: it sorts polygons by mean z and paints back to front, which is only exact when
no two polygons overlap in depth order -- and a closed cellular body is the worst case for it.
Measured on b_star's end frame, 6,124 of 12,272 apical faces point away from the camera at azimuth
310 and are drawn anyway, so which far-side face wins a tie changes with the angle, and one surface
is drawn two different ways at 0:12 and 0:14 of a single rotation. VTK discards a fragment behind
another per pixel, so the question cannot arise. It is also ~29x faster on this mesh -- 0.32 s a
frame against 9.33 -- and lights the frame properly: 28.9% of pixels lit against 4.5%.

WHAT CHANGED IN THE PROMOTION, and it is only the reading. Every function that makes a picture --
`mesh_of`, `_marks`, `add`, `aim`, `_plotter` -- is carried across character for character from
`discovery_okuda/vtk_render.py`, because a renderer that draws a slightly different picture is a
renderer whose two versions cannot be compared. What is new is that `frames_of` reads EITHER
trajectory layout:

    okuda   traj.npz         pos_i, mesh_i (a pickled dict), act_i, ticks
    core    trajectory.npz   vertex__pos, vertex__occ, vertex__mesh_* (ragged, no pickle),
                             cell__chem

and that a run is named by its DIRECTORY rather than by a name under `log/okuda`, because the core
has no such tree.

SHADING IS ALWAYS SMOOTH unless asked otherwise. Flat shading gives every cell one normal, so a
curved arm reads as a faceted cone and the surface's own curvature is lost; the only thing it buys
is a stronger sense of the mesh, and the mesh has its own switch:

    mesh     smooth-shaded surface WITH the cell outlines. Read this when the question is about
             cells -- who divided, how big, how many across a tube.
    nomesh   the same surface, no outlines. Read this when the question is about SHAPE: an arm
             reads as a round tube with light running down it and nothing competes with the
             silhouette.
    flat     one normal per cell, no stroke. For a STILL that will be tiled at ~190 px, where a
             0.4 px outline is a grey wash and smooth shading is a smooth blob.

THE COLOUR SEMANTICS, and each one is a claim about what the data contains:

    white -> red   the activator. This is the measurement.
    magenta        non-finite: not a cell any more.
    green          a cell that has just divided (`age` <= 4 AND `ndiv` > 0 -- `age` alone starts at
                   0 for every seeded cell and would paint an untouched tissue green).
    blue           where the SECOND field ACTED -- marked a cell to die, or switched its growth
                   off. Not how much of that field there is: the trajectory carries one activator
                   channel, so a card claiming a concentration would be claiming data we do not
                   have.

WHAT THE ENGINE HAD TO RECORD FOR THIS TO WORK ON THE CORE SIDE. Three of those four colours read
per-face state that lives on the mesh and nowhere else, and `MeshTable.snapshot` was topology-only
-- so a core-side movie was a plain white surface. `MeshTable.FACE_RECORD` is the list it keeps now;
every name in it is optional and a run without an inhibitor simply has no `inhib` column.
"""
from __future__ import annotations

import os
import time

import numpy as np

CAM = dict(elev=18.0, azim=30.0)
SIZE = 896
FPS = 25
KB_SECONDS = 18.0          # one revolution; the length IS the speed
KB_ZOOM = 0.55
EV_FPS = 12                # ~60 recorded frames -> a 5 s clip

# (kind, style) per sequence, so a caller asks for a JOB and not for four commands.
SEQUENCES = {
    0: [("kburns", "nomesh")],
    1: [("evolve", "mesh")],
    2: [("evolve", "mesh"), ("kburns", "mesh")],
    3: [("evolve", "nomesh"), ("kburns", "nomesh"), ("evolve", "mesh"), ("kburns", "mesh")],
}
LOOP_SEQ = 0

DIVIDED = 4
GREEN = (44, 160, 44)
GREEN_A = 0.5              # how much of the cell's own colour the division mark keeps
BLUE = (31, 119, 180)

# THE DIVISION PAIR, AS TWO COLOURS RATHER THAN ONE. A just-divided cell used to be tinted green --
# which says "a division happened here" and cannot say WHICH TWO CELLS it produced. On a sheet where
# several cells divide in the same four-frame window the green patches merge and the pairing is
# unreadable, and the pairing is the thing: a division that hands its daughter the wrong share of
# the mother's area, or puts the septum in the wrong place, looks exactly like a healthy one until
# you can see the two halves apart.
#
# MOTHER BLUE, DAUGHTER RED, and the assignment is not arbitrary. `cell_divide` keeps the mother at
# her own face index and APPENDS the daughter at `nF`, so a cell whose index is at or beyond the
# PREVIOUS frame's face count is new. That is the only frame-local way to tell them apart, and it is
# why `evolve` threads `prev_nF` through and `kburns`/`still` do not draw the pair at all -- one held
# frame has no previous frame to be new with respect to.
MOTHER = (31, 119, 180)    # blue
DAUGHTER = (214, 39, 40)   # red
PAIR_A = 0.62              # how much of the pair colour is laid over the cell's own


# --------------------------------------------------------------------------- reading a run
def _okuda_frames(path):
    """okuda's `traj.npz`: pos_i / mesh_i / act_i, the mesh a pickled dict."""
    z = np.load(path, allow_pickle=True)
    out = []
    for t in range(sum(1 for k in z.files if k.startswith("pos_"))):
        mt = z[f"mesh_{t}"]
        mt = mt.item() if hasattr(mt, "item") else mt
        act = z[f"act_{t}"] if f"act_{t}" in z.files else None
        out.append((np.asarray(z[f"pos_{t}"], float), mt, act))
    return out


def _core_frames(path, set_name=None, cell_set=None, chan=0):
    """The core's `trajectory.npz`, unpacked into the same (pos, mesh, act) triples.

    TWO OFFSET ARRAYS, and mixing them is the failure this reader has to avoid: the half-edge
    columns are indexed by HALF-EDGE (`mesh_offsets`) and the per-face state by FACE
    (`mesh_face_offsets`), and there are roughly six half-edges per face. One offsets array for
    both would slice the wrong rows out of the myosin and the picture would look plausible.
    """
    z = np.load(path)
    if set_name is None:
        cand = [k[:-len("__mesh_nF")] for k in z.files if k.endswith("__mesh_nF")]
        if not cand:
            return None
        set_name = cand[0]
    if cell_set is None:
        cc = [k[:-len("__chem")] for k in z.files if k.endswith("__chem")]
        cell_set = cc[0] if cc else None
    pos = z[f"{set_name}__pos"]
    nF, Nv = z[f"{set_name}__mesh_nF"], z[f"{set_name}__mesh_Nv"]
    off, foff = z[f"{set_name}__mesh_offsets"], z[f"{set_name}__mesh_face_offsets"]
    face_cols = [k.split("__mesh_")[1] for k in z.files
                 if "__mesh_" in k and k.startswith(set_name)
                 and k.split("__mesh_")[1] not in ("E_srce", "E_trgt", "E_face", "nF", "Nv",
                                                   "offsets", "face_offsets")]
    chem = z[f"{cell_set}__chem"] if cell_set and f"{cell_set}__chem" in z.files else None
    out = []
    for t in range(len(nF)):
        a, b = int(off[t]), int(off[t + 1])
        fa, fb = int(foff[t]), int(foff[t + 1])
        mt = {"E_srce": z[f"{set_name}__mesh_E_srce"][a:b],
              "E_trgt": z[f"{set_name}__mesh_E_trgt"][a:b],
              "E_face": z[f"{set_name}__mesh_E_face"][a:b],
              "nF": int(nF[t]), "Nv": int(Nv[t])}
        for c in face_cols:
            mt[c] = z[f"{set_name}__mesh_{c}"][fa:fb]
        act = None if chem is None else np.asarray(chem[t][:int(nF[t]), chan], float)
        out.append((np.asarray(pos[t][:int(Nv[t])], float), mt, act))
    return out


def frames_of(run_dir, traj=None):
    """Every recorded (pos, mesh, act) of a run, from whichever writer produced it.

    `traj` overrides the file, so a MID-RUN snapshot written in the same format renders through
    exactly this path and needs no special case.
    """
    if traj:
        return (_okuda_frames(traj) if os.path.basename(traj) != "trajectory.npz"
                else _core_frames(traj))
    p = os.path.join(run_dir, "traj.npz")
    if os.path.exists(p):
        return _okuda_frames(p)
    for root, _d, files in os.walk(run_dir):
        if "trajectory.npz" in files:
            return _core_frames(os.path.join(root, "trajectory.npz"))
    return None


def box_of(run_dir, fr):
    """The run's own fixed camera box -- shared with every other picture of it.

    FIXED, NOT PER-FRAME. A camera that refits each frame makes a growing body look static, which
    is the same defect as renormalising the colour scale every frame: the thing the movie is about
    becomes the one thing it cannot show.
    """
    import json
    dj = os.path.join(run_dir, "diag.json")
    if os.path.exists(dj):
        try:
            L = (json.load(open(dj)).get("summary") or {}).get("camera_lbox")
            if L:
                return float(L)
        except Exception:
            pass
    return float(max(np.abs(np.asarray(p)).max() for p, _m, _a in fr)) * 1.12


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wr", ["white", "#d62728"])


def _marks(mt, idx, nF, prev_nF=None):
    """(mother, daughter, kills, suppresses) masks over the drawn faces, None where unrecorded.

    `mother`/`daughter` SPLIT WHAT WAS ONE `divided` MASK, using the previous frame's face count:
    every face at or beyond `prev_nF` was appended this frame and is therefore a daughter, and the
    rest of the just-divided set are the mothers they came from. With `prev_nF=None` -- a single
    held frame, which is all `kburns` and `still` have -- both come back None and nothing is drawn,
    because a pair cannot be identified without a before.

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
    mother = daughter = None
    if prev_nF is not None:
        # A DAUGHTER IS IDENTIFIED BY ITS INDEX ALONE, and it has to be. The first version asked for
        # `div & new` -- a just-divided cell that is also newly appended -- and drew ZERO daughters
        # on every frame. The reason is the same one-frame staleness that makes `col()` pad: `age`
        # and `ndiv` are written by the operators BEFORE the frame's divisions are applied, so on a
        # dividing frame the arrays are `nF_prev` long, the new faces are padded with zeros, and
        # `ndiv > 0` is False for exactly the cells that were just created. The `div` set is
        # therefore the MOTHERS and can never contain a daughter.
        #
        # So the daughter test is `index >= prev_nF`, which is what `cell_divide` guarantees by
        # appending, and it needs no per-cell state at all. Over a trajectory recorded at stride 1
        # that is "born this tick"; over a decimated archive it is "born since the previous drawn
        # frame", which is the honest reading of a picture that skips frames.
        new = np.asarray(idx, int) >= int(prev_nF)
        daughter = new
        mother = (div & ~new) if div is not None else None
    kills, sup = col("apop"), col("inhib")
    return (mother, daughter,
            (None if kills is None else kills > 0), (None if sup is None else sup > 0))


def mesh_of(pos, mt, act, lo=None, hi=None, show_div=True, prev_nF=None):
    """The apical shell as PolyData with per-cell RGB. Rebuilt per frame: cells divide."""
    import pyvista as pv
    from plexus.models.topology import rings_from_flat_3d
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
    mother, daughter, kills, sup = _marks(mt, idx, nF, prev_nF)
    # ORDER MATTERS, and it changed with the division pair. Suppression is the background; the
    # DIVISION PAIR comes next; DEATH is last and wins outright. Death last because a cell sentenced
    # to die is the rarer and more consequential event, and because blue is also the mother's colour
    # -- if a dying cell had just divided, the two marks would otherwise be the same blue and the
    # picture would say "mother" where it means "about to be extruded".
    if sup is not None and sup.any():
        # A BLEND, NOT A THRESHOLD. `x < 0.5` on a smooth activator field toggles every cell that
        # sits near the line, so whole regions flipped blue and back between frames -- the same
        # flicker as the dropped mask, from the other direction. Weighting by how LOW the activator
        # is makes a cell that is halfway halfway blue, and nothing jumps.
        w = (sup.astype(float) if act is None else sup * (1.0 - x)) [:, None]
        rgb = ((1.0 - w) * rgb.astype(float) + w * np.asarray(BLUE, float)).astype(np.uint8)
    # THE PAIR, BEFORE DEATH. Blended rather than painted for the reason the green tint was: solid
    # colour throws away what the cell WAS -- its activator level, or the blue that says the second
    # field is acting on it -- to say one bit. At PAIR_A the cell keeps a readable share of its own
    # colour and the two halves are still unmistakably a red one and a blue one.
    for mask, col_ in ((mother, MOTHER), (daughter, DAUGHTER)):
        if show_div and mask is not None and mask.any():
            rgb[mask] = ((1.0 - PAIR_A) * rgb[mask].astype(float)
                         + PAIR_A * np.asarray(col_, float)).astype(np.uint8)
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
    # (the single-colour green tint this replaced is kept in `GREEN`/`GREEN_A` for a caller that
    # wants "a division happened" without the pairing -- nothing in the core asks for it now.)
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


# --------------------------------------------------------------------------- the three products
def kburns(run_dir, style, out, fill=1.0, label=None):
    """The finished specimen, turned once and zoomed in. Geometry fixed, camera moving."""
    fr = frames_of(run_dir)
    if not fr:
        return "no trajectory"
    L0 = box_of(run_dir, fr)
    name = label or os.path.basename(run_dir.rstrip("/"))
    pos, mt, act = fr[-1]
    m = mesh_of(pos, mt, act, show_div=False)
    n = int(KB_SECONDS * FPS)
    p = _plotter(); add(p, m, style)
    p.add_text(f"{name}  {style}", position="upper_left", font_size=11, color="white")
    p.open_movie(out, framerate=FPS, quality=8)
    for i in range(n):
        u = i / (n - 1)
        aim(p, L0 * (1.0 - (1.0 - KB_ZOOM) * _ease(u)),
            azim=CAM["azim"] + 360.0 * u,                 # a FULL turn, so the clip loops
            elev=CAM["elev"] + 12.0 * np.sin(np.pi * u), fill=fill)
        p.write_frame()
    p.close()
    return f"{n} frames"


def evolve(run_dir, style, out, fill=1.0, label=None):
    """The run through time, camera nailed down."""
    fr = frames_of(run_dir)
    if not fr:
        return "no trajectory"
    L = box_of(run_dir, fr)
    name = label or os.path.basename(run_dir.rstrip("/"))
    # ONE ACTIVATOR RANGE FOR THE WHOLE CLIP, taken over every recorded frame. Per-frame
    # normalisation would make a strengthening pattern look constant.
    vals = [np.asarray(a, float) for _p, _m, a in fr if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0
    p = _plotter()
    p.open_movie(out, framerate=EV_FPS, quality=8)
    actor = txt = None
    prev_nF = None
    for t, (pos, mt, act) in enumerate(fr):
        m = mesh_of(pos, mt, act, lo, hi, show_div=(style == "mesh"), prev_nF=prev_nF)
        prev_nF = int(mt["nF"])
        if m is None:
            continue
        if actor is not None:
            p.remove_actor(actor)                  # NOT p.clear(): that removes the lights too
        if txt is not None:
            p.remove_actor(txt)
        actor = add(p, m, style)
        txt = p.add_text(f"{name}  {style}   frame {t + 1}/{len(fr)}   {int(mt['nF'])} cells",
                         position="upper_left", font_size=11, color="white")
        aim(p, L, fill=fill)
        p.write_frame()
    p.close()
    return f"{len(fr)} frames"


def still(run_dir, style="flat", out=None, fill=1.0, frame=-1, label=True, traj=None, name=None):
    """One frame as an image -- the VTK successor to `3d.png`.

    FLAT BY DEFAULT HERE, AND ONLY HERE. `nomesh` stays the choice for the movies, where the
    question is the silhouette and a faceted cone would lose the curvature. A still that is going
    to be tiled at ~190 px is a different question: at that size a 0.4 px outline is a grey wash
    and smooth shading is a smooth blob, while facets scale with the cells they belong to, so the
    mesh is legible in a thumbnail without drawing a single line.
    """
    fr = frames_of(run_dir, traj)
    if not fr:
        return "no trajectory"
    L = box_of(run_dir, fr)
    nm = name or os.path.basename(run_dir.rstrip("/"))
    pos, mt, act = fr[frame][:3]
    m = mesh_of(pos, mt, act, show_div=(style == "mesh"))
    p = _plotter()
    add(p, m, style)
    if label:
        p.add_text(f"{nm}  frame {len(fr) - 1 if frame == -1 else frame}",
                   position="upper_left", font_size=11, color="white")
    aim(p, L, fill=fill)
    p.screenshot(out or os.path.join(run_dir, "3d.png"))
    p.close()
    return f"{len(fr)} frames, drew {frame}"


def render_all(run_dir, seq=LOOP_SEQ, size=None, quiet=False, fill=1.0, name=None):
    """Run one named sequence for one finished run directory. {} if there is nothing to render.

    Sequence 3 -- four clips, 1,020 frames -- measured at about 20 s on b_star's 12,272 cells,
    against a run that costs 30-40 MINUTES of GPU. That is what makes it affordable to ask for the
    full set from inside a generation rather than as a separate pass.
    """
    global SIZE
    if size:
        SIZE = size
    if frames_of(run_dir) is None:
        return {}
    nm = name or os.path.basename(run_dir.rstrip("/"))
    took = {}
    for kind, st in SEQUENCES[int(seq)]:
        out = os.path.join(run_dir, f"vtk_{kind}_{st}.mp4")
        t0 = time.perf_counter()
        msg = (kburns if kind == "kburns" else evolve)(run_dir, st, out, fill=fill, label=nm)
        dt = time.perf_counter() - t0
        if msg.startswith("no "):
            return {}
        took[os.path.basename(out)] = dt
        if not quiet:
            print(f"  {kind:7s} {st:7s} {msg:12s} {dt:7.1f} s -> {out}", flush=True)
    return took


def available():
    """Is a VTK render possible here? `plot.py` falls back to matplotlib when it is not -- a
    headless node with no GL is a real configuration and it must not take a generation down."""
    try:
        import pyvista                                    # noqa: F401
        return True
    except Exception:
        return False


# ==========================================================================================
#  POINT CLOUDS AND VOLUMES -- the neural products.
#
#  The three products above (kburns / evolve / still) all draw a MESH: a closed cellular
#  surface with per-face colour. A neural run has neither. It has a cloud of somas carrying a
#  scalar, and a regular grid rendered from them, and the two want different renderers -- a
#  point splat and a volume ray-cast. What they SHARE with the mesh products is everything
#  around the picture: `_plotter` (off-screen, black, 8x MSAA), `aim` (the camera framing),
#  `p.open_movie` (the mp4 writer), `EV_FPS`. Those are called, not copied.
#
#  VIRIDIS, not the white->red activator ramp of `mesh_of`. The activator is a concentration
#  with a floor at zero, so a ramp from white reads correctly; a membrane potential is signed
#  and centred, and a perceptually-uniform map is what lets a reader compare two frames.
# ==========================================================================================
def _traj(run_dir):
    """The core `trajectory.npz` of a run directory."""
    for root, _d, files in os.walk(run_dir):
        if "trajectory.npz" in files:
            return np.load(os.path.join(root, "trajectory.npz"), allow_pickle=True)
    raise FileNotFoundError(f"no trajectory.npz under {run_dir}")


def _range(a, lo_q=0.5, hi_q=99.5):
    """One colour range for the WHOLE clip, from percentiles rather than min/max.

    Per-frame normalisation would make a strengthening pattern look constant -- the same
    reason `evolve` fixes its activator range. Percentiles rather than extrema because a
    single runaway neuron would otherwise compress every other value into one colour."""
    f = np.asarray(a, np.float64).ravel()
    f = f[np.isfinite(f)]
    return float(np.percentile(f, lo_q)), float(np.percentile(f, hi_q))


def evolve_points(run_dir, out, set_name="neuron", block="voltage", cmap="viridis",
                  point_size=9.0, fill=0.95, label=None, fps=None):
    """The neurons themselves: one sphere per soma at its real position, coloured by `block`.

    This is the picture that says whether the SEED did its job -- the cloud has the shape of
    the region it was cropped from, or it does not."""
    import pyvista as pv
    from matplotlib import colormaps
    z = _traj(run_dir)
    pos = np.asarray(z[f"{set_name}__pos"], np.float64)          # [T, N, D]
    val = np.asarray(z[f"{set_name}__{block}"], np.float64)[..., 0]   # [T, N]
    occ = np.asarray(z[f"{set_name}__occ"])
    lo, hi = _range(val)
    cm = colormaps[cmap]
    name = label or os.path.basename(run_dir.rstrip("/"))
    L = float(np.nanmax(np.abs(pos))) * 1.05
    p = _plotter()
    p.open_movie(out, framerate=fps or EV_FPS, quality=8)
    actor = txt = None
    for t in range(pos.shape[0]):
        live = occ[t] > 0
        cloud = pv.PolyData(np.ascontiguousarray(pos[t][live]))
        x = np.clip((val[t][live] - lo) / max(hi - lo, 1e-12), 0, 1)
        cloud["rgb"] = (np.asarray(cm(x))[:, :3] * 255).astype(np.uint8)
        if actor is not None:
            p.remove_actor(actor)                                # NOT p.clear(): keeps the lights
        if txt is not None:
            p.remove_actor(txt)
        actor = p.add_points(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True,
                             point_size=point_size, lighting=True)
        txt = p.add_text(f"{name}  {block}  frame {t + 1}/{pos.shape[0]}   "
                         f"{int(live.sum())} neurons   [{lo:.2f}, {hi:.2f}]",
                         position="upper_left", font_size=11, color="white")
        # `aim` frames a box CENTRED ON THE ORIGIN, which is what every mesh run is. A neural
        # region is mapped into the UNIT BOX [0,1]^3, so aiming at a half-width of max|pos|
        # put the cloud in the top corner of the frame. `reset_camera` frames the actual
        # bounds instead, which is correct for either convention.
        p.camera_position = "iso"
        p.reset_camera()
        p.camera.zoom(fill)
        p.write_frame()
    p.close()
    return f"{pos.shape[0]} frames, {pos.shape[1]} neurons, range [{lo:.3f}, {hi:.3f}]"


def evolve_volume(run_dir, out, field="neural_activity", cmap="viridis", fill=0.95,
                  label=None, fps=None, opacity="sigmoid_5",
                  skeletons=None, skel_neurons=1000, skel_opacity=0.16, skel_width=1.0):
    """The 128^3 rendered activity, ray-cast as a volume -- the cube a transformer would see.

    THE OPACITY TRANSFER FUNCTION IS PART OF THE PICTURE AND NOT A STYLE CHOICE. A volume
    render shows what the transfer function lets through; a linear ramp on a field whose mass
    sits near zero shows fog. `sigmoid_5` keeps the low-magnitude bulk transparent so the
    active structure is what is visible, and it is named in the frame so the reader knows
    which map produced the picture they are looking at."""
    z = _traj(run_dir)
    key = f"{field}__grid"
    if key not in z.files:
        raise KeyError(f"{key} not in the trajectory (fields: "
                       f"{[k for k in z.files if k.endswith('__grid')]})")
    g = np.asarray(z[key], np.float32)                           # [T, C, nx, ny, nz]
    if g.ndim == 5:
        g = g[:, 0]
    lo, hi = _range(np.abs(g), 0.0, 99.9)
    name = label or os.path.basename(run_dir.rstrip("/"))
    import pyvista as pv
    p = _plotter()
    # THE ANATOMY GOES IN ONCE, BEFORE THE LOOP, because it does not move. Rebuilding half a
    # million line segments per frame would dominate the render for a picture that is
    # identical in every frame; the volume actor is the only thing swapped.
    skel_note = ""
    if skeletons:
        poly, meta = skeleton_lines(skeletons, max_neurons=skel_neurons)
        if poly is not None:
            p.add_mesh(poly, color="white", opacity=skel_opacity, line_width=skel_width,
                       lighting=False)
            frac = meta["segments"] / max(meta["segments_before_clip"], 1)
            skel_note = (f"   skeletons {meta['neurons']}n {meta['segments'] // 1000}k seg "
                         f"({frac:.0%} in cube)")
            print(f"[render] skeletons: {meta}", flush=True)
    p.open_movie(out, framerate=fps or EV_FPS, quality=8)
    actor = txt = None
    for t in range(g.shape[0]):
        vol = pv.ImageData(dimensions=np.asarray(g[t].shape) + 1)
        vol.cell_data["a"] = np.abs(g[t]).ravel(order="F")
        if actor is not None:
            p.remove_actor(actor)
        if txt is not None:
            p.remove_actor(txt)
        actor = p.add_volume(vol, scalars="a", cmap=cmap, opacity=opacity,
                             clim=[lo, hi], show_scalar_bar=False)
        txt = p.add_text(f"{name}  |{field}|  {'x'.join(str(s) for s in g[t].shape)}   "
                         f"frame {t + 1}/{g.shape[0]}   opacity={opacity}   "
                         f"clim [{lo:.2f}, {hi:.2f}]{skel_note}",
                         position="upper_left", font_size=10, color="white")
        if t == 0:
            p.camera_position = "iso"
            p.reset_camera()
            p.camera.zoom(fill)
        p.write_frame()
    p.close()
    return f"{g.shape[0]} frames, {'x'.join(str(s) for s in g[0].shape)}, range [{lo:.3f}, {hi:.3f}]"


def skeleton_lines(region, max_neurons=200, stride=1, seed=0):
    """SWC skeletons of a frozen region, as ONE `pv.PolyData` of line segments in the unit box.

    THREE THINGS HAVE TO HAPPEN AND ALL THREE ARE EASY TO GET WRONG.

    1. THE TRANSFORM. `fetch_skeleton` returns DATASET VOXELS, while the region's cube is
       defined in nanometres, so a segment must go through the same `scale_nm`/`offset_nm` the
       importer recorded before it can be compared with a soma. Checked on this region: the
       first node of body 100003774 maps to within 300 nm of its own recorded soma.

    2. THE CLIP. A neuron's arbour is far larger than the cube its SOMA fell in -- body
       100003774 spans 10,700 voxels in x against the cube's 2,837. Drawing the arbours
       unclipped would put the anatomy an order of magnitude outside the volume and shrink the
       volume to nothing. Segments with either endpoint outside the unit box are dropped.

    3. THE BUDGET. 1,000 skeletons at a median 13,460 nodes is 13.5M segments. Rather than
       thin every neuron into a dashed line, a deterministic SUBSET is drawn at full
       resolution: a partial arbour still reads as an arbour, a dashed one reads as noise.
       `max_neurons` is what was drawn and is reported, so the picture never implies more
       morphology than it shows.
    """
    import glob
    import json
    import pyvista as pv
    from plexus.io.neuprint import region_path
    root = region_path(region)
    man = json.load(open(os.path.join(root, "manifest.json")))
    r, vx = man["region"], man["source"]["voxel_to_nm"]
    sc, of = np.asarray(vx["scale_nm"], float), np.asarray(vx["offset_nm"], float)
    lo, side = np.asarray(r["bounds_lo_nm"], float), float(r["side_nm"])
    files = sorted(glob.glob(os.path.join(root, "skeletons", "*.swc")))
    if max_neurons and len(files) > max_neurons:                 # deterministic subset
        idx = np.random.default_rng(seed).choice(len(files), max_neurons, replace=False)
        files = [files[i] for i in sorted(idx)]
    pts, lines, kept, total = [], [], 0, 0
    off = 0
    for f in files:
        a = np.loadtxt(f, comments="#", ndmin=2)
        if a.size == 0:
            continue
        xyz = (a[:, 2:5] * sc + of - lo) / side                  # voxels -> nm -> unit box
        rid = a[:, 0].astype(np.int64)
        par = a[:, 6].astype(np.int64)
        pos_of = {int(k): i for i, k in enumerate(rid)}
        seg = np.array([[pos_of[int(p)], i] for i, p in enumerate(par)
                        if int(p) in pos_of], dtype=np.int64)
        total += len(seg)
        if stride > 1:
            seg = seg[::stride]
        if len(seg) == 0:
            continue
        inside = np.all((xyz >= 0) & (xyz <= 1), axis=1)
        seg = seg[inside[seg[:, 0]] & inside[seg[:, 1]]]         # clip to the cube
        if len(seg) == 0:
            continue
        # COMPACT TO THE POINTS THE SURVIVING SEGMENTS ACTUALLY USE. Clipping the segments is
        # not enough: a PolyData carrying the un-clipped points still has the un-clipped
        # BOUNDS, and 96% of these nodes are outside the cube (a soma is in the region, its
        # arbour crosses the brain). `reset_camera` then frames the whole arbour field and the
        # in-cube anatomy renders as a small blob beside the volume -- which is exactly what
        # the first 4x4 montage showed.
        used, seg = np.unique(seg, return_inverse=True)
        seg = seg.reshape(-1, 2)
        xyz = xyz[used]
        pts.append(xyz)
        lines.append(np.column_stack([np.full(len(seg), 2), seg + off]))
        off += len(xyz)
        kept += len(seg)
    if not pts:
        return None, {"neurons": 0, "segments": 0}
    poly = pv.PolyData(np.ascontiguousarray(np.concatenate(pts)))
    poly.lines = np.concatenate(lines).ravel()
    return poly, {"neurons": len(files), "segments": kept, "segments_before_clip": total}


def evolve_skeleton_activity(run_dir, out, region, field="neural_activity", n_arbours=100,
                             cmap="viridis", opacity=0.95, line_width=1.6, fill=0.92,
                             label=None, fps=None):
    """The morphology, carrying the activity: SWC neurites coloured by the field at each node.

    WHY NOT SKELETONS *AND* A VOLUME IN ONE SCENE. Measured: they do not composite. VTK's GPU
    ray-cast volume mapper draws after and over translucent polygonal geometry, so half a
    million faintly-drawn line segments vanish entirely behind the volume -- at every opacity
    tried, and with 8-layer depth peeling enabled (which applies to translucent polygons, not
    to the volume). The 4x4 montage in `log/promotion/` is that measurement. So the two are
    separate products: `evolve_volume` for the field a continuum model consumes, and this for
    the anatomy a reader recognises.

    THIS IS THE CONVENTION OF THE REFERENCE FIGURE, not a new one. connectome-gnn's
    `figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py` draws the skeleton segments
    and puts the activity ON them (a faint base pass, then a coloured overlay on the lit
    segments) rather than beside them. The same idea in 3D: one line set, coloured per node.

    `n_arbours` IS A REAL LIMIT AND IS DRAWN IN THE FRAME. All 1,000 arbours of this region
    fill the cube solid -- 514,949 in-cube segments render as an opaque mass with no
    morphology visible at all. 100 is dense but legible; 25 shows individual neurites.
    """
    from matplotlib import colormaps
    import pyvista as pv
    z = _traj(run_dir)
    g = np.asarray(z[f"{field}__grid"], np.float32)
    if g.ndim == 5:
        g = g[:, 0]
    lo, hi = _range(np.abs(g), 0.0, 99.9)
    poly, meta = skeleton_lines(region, max_neurons=n_arbours)
    if poly is None:
        return "no skeletons"
    P = np.asarray(poly.points)
    R = g.shape[1]
    idx = np.clip((P * R).astype(int), 0, R - 1)                 # nearest voxel per node
    cm = colormaps[cmap]
    name = label or os.path.basename(run_dir.rstrip("/"))
    p = _plotter()
    p.open_movie(out, framerate=fps or EV_FPS, quality=8)
    actor = txt = None
    for t in range(g.shape[0]):
        a = np.abs(g[t])[idx[:, 0], idx[:, 1], idx[:, 2]]
        x = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
        poly["rgb"] = (np.asarray(cm(x))[:, :3] * 255).astype(np.uint8)
        if actor is not None:
            p.remove_actor(actor)
        if txt is not None:
            p.remove_actor(txt)
        actor = p.add_mesh(poly, scalars="rgb", rgb=True, opacity=opacity,
                           line_width=line_width, lighting=False)
        txt = p.add_text(f"{name}  skeletons coloured by |{field}|   frame {t + 1}/{g.shape[0]}"
                         f"   {meta['neurons']} of 1000 arbours, {meta['segments'] // 1000}k "
                         f"segments in cube   clim [{lo:.2f}, {hi:.2f}]",
                         position="upper_left", font_size=10, color="white")
        if t == 0:
            p.camera_position = "iso"
            p.reset_camera()
            p.camera.zoom(fill)
        p.write_frame()
    p.close()
    return f"{g.shape[0]} frames, {meta['neurons']} arbours, {meta['segments']} segments"
