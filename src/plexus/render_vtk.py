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

# THE NEURAL CLIPS RUN AT THEIR OWN RATE, and it is 8x `EV_FPS` rather than a tweak to it. A
# mesh `evolve` draws ~60 subsampled frames of a morphogenesis run, where 12 fps is a slideshow
# of distinct shapes and slower is more readable. A neural clip draws EVERY recorded frame of a
# 2,000-step run of a circuit whose membrane time constant is a handful of steps, so at 12 fps
# it is a 3-minute crawl through dynamics that look static frame to frame. 96 fps puts 2,001
# frames in 21 s, which is the timescale the activity actually moves on.
NEURAL_FPS = 96

# HOW MANY FRAMES AN `evolve` CLIP DRAWS, AND WHY IT IS CAPPED. okuda's archive keeps ~60 of a run's
# rows; the core's trajectory keeps every one -- 1,801 on `r023_07`. Uncapped, the two sides of a
# promotion pair produce a 5-second clip and a 2.5-minute one of the same run, which cannot be
# compared frame for frame by the eye they are for, and the core side takes twenty minutes to draw.
# Capped, both sides subsample the same way (`linspace`, first and last always in) and land on the
# same count, so `A/vtk_evolve_mesh.mp4` and `B/vtk_evolve_mesh.mp4` step through the run together.
MAX_EVOLVE_FRAMES = 60

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

# THE DYING CELL GETS ITS OWN COLOUR, and not the mother's blue. `BLUE` is both the suppression
# tint and, at (31,119,180), exactly `MOTHER` -- so on a sheet where cells divide AND die the same
# blue meant two opposite things, and `mesh_of` had to paint death last purely to break the tie.
# Electric blue is far enough from the mother's mid-blue to be read at a glance and far enough from
# the grey body to survive the shading.
#
# IT IS PAINTED OUTRIGHT, NOT BLENDED. A sentenced cell is a discrete event on a handful of faces;
# blending it at PAIR_A over a white body gave a pale wash that vanished at the silhouette, which
# is where most of the dying patch is in `apop_patch_big`.
KILL = (0, 153, 255)       # electric blue: a cell sentenced to die

# THE BODY IS GREY WHEN THERE IS NO CHEMISTRY TO SHOW, not colormap-white. An all-zero activator is
# not missing -- `cell__chem` exists and is identically 0 on every apoptosis scene -- so `act` is
# not None, the colormap runs, and every cell takes `cm(0)`, which is WHITE. A white body says
# "activator at the bottom of its range" when the truth is "there is no activator here", and it
# leaves the death mark to carry the entire picture against the brightest possible background.
# Grey says the same thing honestly and gives the electric blue something to read against.
BODY_GREY = (176, 176, 176)

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
# DIVIDED (4) division calls x `cell_divide.every` (4) ticks = the window a mother stays marked.
PAIR_TICKS = 16


class _frames_ticks:
    """Set by `frames_of` to the recorded tick numbers, when the writer kept them.

    A module cell rather than a return value because `frames_of` has three callers and two of them
    do not want it; the ticks are only needed to convert the pair window from TICKS into rows, and
    a trajectory recorded at stride 1 does not need the conversion at all.
    """
    value = None


def _pair_reference(t, nFs, ticks, window_ticks):
    """The face count `window_ticks` ago, as a daughter threshold for row `t`.

    Returns None at the start of a run, where there is no `before` and nothing should be drawn.
    """
    if t == 0:
        return None
    if ticks is None:                       # stride 1: a row IS a tick
        return nFs[max(0, t - window_ticks)]
    cut = ticks[t] - window_ticks
    j = 0
    for k in range(t, -1, -1):
        if ticks[k] <= cut:
            j = k
            break
    return nFs[j]


# --------------------------------------------------------------------------- reading a run
def _okuda_frames(path):
    """okuda's `traj.npz`: pos_i / mesh_i / act_i, the mesh a pickled dict."""
    z = np.load(path, allow_pickle=True)
    out = []
    for t in range(sum(1 for k in z.files if k.startswith("pos_"))):
        mt = z[f"mesh_{t}"]
        mt = mt.item() if hasattr(mt, "item") else mt
        act = z[f"act_{t}"] if f"act_{t}" in z.files else None
        # okuda's traj.npz stores the activator column only, so there is no LUT colouring to do
        # from it: the fourth slot is None and `mesh_of` falls back to white -> red, unchanged.
        out.append((np.asarray(z[f"pos_{t}"], float), mt, act, None))
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
    # EVERY ARRAY IS READ ONCE. `np.load` returns an NpzFile whose `z[key]` DECOMPRESSES THE WHOLE
    # ARRAY on every access, so reading one row inside a loop over 1,801 rows re-reads the entire
    # column 1,801 times. On `r023_07`'s 6.7 GB trajectory that is terabytes of decompression and the
    # render never finishes -- it is the same defect `gate_measures._Lazy` exists for, and it was
    # here too.
    pos = z[f"{set_name}__pos"]
    nF, Nv = z[f"{set_name}__mesh_nF"], z[f"{set_name}__mesh_Nv"]
    off, foff = z[f"{set_name}__mesh_offsets"], z[f"{set_name}__mesh_face_offsets"]
    E = {c: z[f"{set_name}__mesh_E_{c}"] for c in ("srce", "trgt", "face")}
    face_cols = {k.split("__mesh_")[1]: z[k] for k in z.files
                 if k.startswith(set_name + "__mesh_")
                 and k.split("__mesh_")[1] not in ("E_srce", "E_trgt", "E_face", "nF", "Nv",
                                                   "offsets", "face_offsets")
                 and not k.split("__mesh_")[1].startswith(("scalar_", "e_"))}
    # THE PER-HALF-EDGE COLUMNS, sliced by the HALF-EDGE offsets. `e_myo` is recorded every frame and
    # was dropped here, so nothing downstream could colour a junction by the myosin on it -- the one
    # quantity `junction_myosin` exists to produce. It is excluded from `face_cols` for a good reason
    # (its rows are half-edges, not faces, and `foff` would slice the wrong ones); it needs `off`.
    edge_cols = {k.split("__mesh_")[1]: (z[k], z[f"{k}_offsets"] if f"{k}_offsets" in z.files else off)
                 for k in z.files
                 if k.startswith(set_name + "__mesh_e_") and not k.endswith("_offsets")}
    chem = z[f"{cell_set}__chem"] if cell_set and f"{cell_set}__chem" in z.files else None
    out = []
    for t in range(len(nF)):
        a, b = int(off[t]), int(off[t + 1])
        fa, fb = int(foff[t]), int(foff[t + 1])
        mt = {"E_srce": E["srce"][a:b], "E_trgt": E["trgt"][a:b], "E_face": E["face"][a:b],
              "nF": int(nF[t]), "Nv": int(Nv[t])}
        for c, arr in face_cols.items():
            mt[c] = arr[fa:fb]
        for c, (arr, eoff) in edge_cols.items():       # each column's OWN offsets -- see live_movie
            ea, eb = int(eoff[t]), int(eoff[t + 1])
            if eb > ea:
                mt[c] = arr[ea:eb]
        act = None if chem is None else np.asarray(chem[t][:int(nF[t]), chan], float)
        # THE WHOLE CHEM ROW TRAVELS WITH THE FRAME, not only the column `chan` names. A
        # three-species run (May-Leonard u,v,w) has no single activator: colouring it from one
        # column is why a rock-paper-scissors mesh rendered white-to-red and threw two species
        # away. `act` stays what it was -- the scalar the range and the metrics are taken from --
        # and the extra element is what `mesh_of` colours from when the spec declares a LUT.
        rows = None if chem is None else np.asarray(chem[t][:int(nF[t])], float)
        out.append((np.asarray(pos[t][:int(Nv[t])], float), mt, act, rows))
    return out


def style_of(run_dir):
    """`(lut, blend)` from the run's own `spec.yaml` -- the colour table is part of the MODEL.

    Returns `(lut, blend, cutaway)`; all three are None when the spec declares no `plotting`
    block, which is every Gray-Scott run -- those keep the white -> red activator map they were
    built with and draw the whole closed surface.
    """
    for nm in ("spec.yaml", "spec_run.yaml"):
        f = os.path.join(run_dir, nm)
        if not os.path.exists(f):
            continue
        try:
            import yaml
            pl = (yaml.safe_load(open(f)) or {}).get("plotting") or {}
        except Exception:
            return None, None
        lut = pl.get("species")
        return (list(lut) if lut else None), pl.get("blend"), pl.get("cutaway")
    return None, None, None


_PLOT_OVERRIDE: dict | None = None


def plot_style(run_dir) -> dict:
    """The `plotting:` block for the render flags that are not colour tables.

    THE LIVE CONFIG WINS OVER THE RUN'S FROZEN COPY. `spec.yaml` in the data directory is written at
    GENERATE time, so reading it here meant that editing a render flag and re-running `-o plot`
    silently used the old value -- change `edge_lut` to viridis, re-plot, get inferno, and nothing
    says why. `plot_dataset` has the freshly loaded spec and now hands it down; the run's copy stays
    the fallback for anything calling this directly (tools, comparisons) with no spec in hand.

    `style_of` returns the three keys the colourmap needs and predates the rest; this returns the
    block so a new flag does not mean a new element on a tuple every caller has to unpack.
    """
    if _PLOT_OVERRIDE is not None:
        return dict(_PLOT_OVERRIDE)
    for nm in ("spec.yaml", "spec_run.yaml"):
        f = os.path.join(run_dir, nm)
        if os.path.exists(f):
            try:
                import yaml
                return dict((yaml.safe_load(open(f)) or {}).get("plotting") or {})
            except Exception:
                return {}
    return {}


def edges_of(pos, mt, mode, ntype=None, rng=None, colors=None, lut="inferno"):
    """The junctions as a line mesh with per-edge RGB -- `plotting.edge_color`.

    WHY A SECOND ACTOR AND NOT A FACE COLOUR. Myosin lives on the JUNCTION, and a face colour can
    only say something about a cell; painting a cell by the mean of its edges is a different claim
    and it hides exactly what a contractile belt does, which is to differ from edge to edge around
    one cell. Drawing the edges themselves is the only honest picture of a per-junction quantity.

      `myosin`  the value `junction_myosin` wrote, through a fixed range so a frame's colour means
                the same number as every other frame's. `plotting.edge_lut` names the colour table
                (any matplotlib colormap); the default is `inferno`, dark-to-bright, which reads as
                "how much" on a black background where a diverging or cyclic map would not.
      `type`    the owning cell's node_type, as flat categorical colours
    """
    import pyvista as pv
    from matplotlib import colormaps
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    if not live.any():
        return None
    i, j, f = es[live].astype(int), et[live].astype(int), ef[live].astype(int)
    if mode == "type":
        if ntype is None:
            return None
        k = np.asarray(ntype)[np.clip(f, 0, len(ntype) - 1)].astype(int)
        pal = np.asarray(colors if colors is not None else
                         [(0.35, 0.60, 1.00), (1.00, 0.35, 0.25), (0.45, 0.95, 0.55),
                          (1.00, 0.85, 0.30)], float)
        rgb = (np.clip(pal[k % len(pal)], 0, 1) * 255).astype(np.uint8)
    else:
        v = mt.get("e_myo")
        if v is None:
            return None
        v = np.asarray(v, float)[live]
        lo, hi = (rng if rng else (float(np.nanmin(v)), float(np.nanmax(v))))
        x = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = (np.asarray(colormaps[lut](x))[:, :3] * 255).astype(np.uint8)
    lines = np.empty((len(i), 3), np.int64)
    lines[:, 0] = 2; lines[:, 1] = i; lines[:, 2] = j
    m = pv.PolyData(np.asarray(pos, float), lines=lines.ravel())
    m["rgb"] = rgb
    return m


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
        z = np.load(p, allow_pickle=True)
        _frames_ticks.value = (np.asarray(z["ticks"]).tolist() if "ticks" in z.files else None)
        return _okuda_frames(p)
    _frames_ticks.value = None                    # a core trajectory records every tick
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
    return float(max(np.abs(np.asarray(p)).max() for p, _m, _a, _c in fr)) * 1.12


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wr", ["white", "#d62728"])


def _marks(mt, idx, nF, prev_nF=None):
    """(mother, daughter, kills, suppresses) masks over the drawn faces, None where unrecorded.

    `mother`/`daughter` SPLIT WHAT WAS ONE `divided` MASK. A face at or beyond `prev_nF` was
    appended since that reference and is a daughter; the rest of the just-divided set are the
    mothers. With `prev_nF=None` -- a single held frame, which is all `kburns` and `still` have --
    both come back None and nothing is drawn, because a pair cannot be identified without a before.

    `prev_nF` IS NOT THE PREVIOUS FRAME'S COUNT, and using it that way is the bug this argument
    replaced. The two masks are on DIFFERENT CLOCKS: `age <= DIVIDED` counts division CALLS, and
    `cell_divide` runs every 4 ticks, so a mother stays blue for about 16 ticks -- while "appended
    since the previous frame" is one row. On a stride-1 trajectory divisions fire on one row in
    four, so three rows out of four drew 372 mothers and ZERO daughters and the pair was almost
    never shown together. The caller passes the face count from far enough back to cover the SAME
    window the mothers are drawn over.

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


def mesh_of(pos, mt, act, lo=None, hi=None, show_div=True, prev_nF=None, chem=None, lut=None,
            cutaway=None,
            blend=None):
    """The apical shell as PolyData with per-cell RGB. Rebuilt per frame: cells divide."""
    import pyvista as pv
    from plexus.models.topology import rings_from_flat_3d
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    rings = rings_from_flat_3d(es[live], et[live], ef[live], nF)
    # THE CUTAWAY, and it is a WEDGE OF FACES DROPPED, not a clipping plane. A vtkClipPolyData
    # cuts through faces and leaves a torn edge whose cells are half-drawn and half-coloured; the
    # point of looking inside a growing tissue is to see whole cells on the far wall, so this drops
    # each face ENTIRELY, by the side its centroid falls on. The surface stays a set of complete
    # cells and the opening is jagged along cell boundaries, which is what a real dissection looks
    # like anyway.
    #
    # `cutaway: [ax, ax]` names the axes whose POSITIVE octant is removed -- ["x","y"] takes the
    # quarter where both x and y exceed the body's centroid, which on a spheroid opens a quadrant
    # and shows the lumen and the inner wall. A single axis halves it; three take one octant.
    _keep_face = None
    if cutaway:
        _ax = {"x": 0, "y": 1, "z": 2}
        _sel = [_ax[a.lower().lstrip("+-")] for a in cutaway if a.lower().lstrip("+-") in _ax]
        _neg = [a.startswith("-") for a in cutaway]
        if _sel:
            _c = np.asarray(pos, float).mean(0)
            _cen = np.array([np.asarray(pos, float)[np.asarray(r, int)].mean(0)
                             if (r is not None and len(r) >= 3) else _c for r in rings])
            _in = np.ones(len(rings), bool)
            for _k, _ax_i in enumerate(_sel):
                _side = _cen[:, _ax_i] < _c[_ax_i] if (_k < len(_neg) and _neg[_k]) \
                    else _cen[:, _ax_i] > _c[_ax_i]
                _in &= _side
            _keep_face = ~_in                       # drop the wedge, keep everything else
    faces, idx = [], []
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        if _keep_face is not None and not _keep_face[f]:
            continue
        faces.append(len(r)); faces.extend(int(v) for v in r); idx.append(f)
    if not idx:
        return None
    m = pv.PolyData(pos, faces=np.asarray(faces, np.int64))
    # A FLAT ACTIVATOR IS NOT AN ACTIVATOR. `act` is non-None on every mesh run because the `cell`
    # set always carries a `chem` block, so the apoptosis scenes -- which have no chemistry at all
    # and whose `cell__chem` is identically 0 over 601 frames -- were being colour-mapped to a
    # single value. See `BODY_GREY`.
    #
    # AN ALL-NaN FRAME IS NOT FLAT. It has to keep reaching the colormap branch, because that is
    # where `rgb[~ok] = magenta` marks "not a cell any more" -- and a run going non-finite is
    # exactly what that mark exists to make visible (`r023_07` was half NaN from frame 889). So
    # "flat" requires that finite values EXIST and are all zero, not merely that none is nonzero.
    _a = None if act is None else np.asarray(act, float)[:nF][idx]
    _fin = None if _a is None else np.isfinite(_a)
    flat = _a is not None and bool(_fin.any()) and not bool(np.any(_a[_fin] != 0.0))
    # A DECLARED LUT WINS. `plotting.species` is a column-to-colour table and it is a property of
    # the MODEL: Gray-Scott draws its activator and hides its substrate, May-Leonard's three
    # species are a partition and want red/green/blue. The 2D path has honoured it since
    # `live.chem_rgb` existed; the mesh path did not, so a spec that already declared
    # `species: [red, green, blue]` still got one column through a white -> red colormap.
    rgb = x = None
    if chem is not None and lut and not flat:
        from plexus.live import chem_rgb
        _cols, _ = chem_rgb(np.asarray(chem, float)[:nF][idx], lut=lut, blend=blend)
        if _cols is not None:
            rgb = (np.clip(_cols, 0, 1) * 255).astype(np.uint8)
            # `x` is what the suppression tint weighs itself by, so it still has to be the
            # activator's own 0..1 position and not something derived from the colour.
            _lo = float(np.nanmin(_a)) if lo is None else lo
            _hi = float(np.nanmax(_a)) if hi is None else hi
            x = np.clip((_a - _lo) / (_hi - _lo + 1e-9), 0, 1)
    if rgb is None and (act is None or flat):
        rgb = np.full((len(idx), 3), BODY_GREY, np.uint8)
        x = np.zeros(len(idx))
    if rgb is None and act is not None and not flat:
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
        rgb[kills] = KILL
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


def offscreen():
    """Make VTK stop trying to talk to an X server, and stop it printing when it fails.

    THE NOISE IS Xlib's, NOT VTK's, which is why filtering python warnings never silenced it.
    A VS Code remote session exports `DISPLAY=:11` with no Xauthority the container can read, so
    every render window opened here printed

        Authorization required, but no authorization protocol specified
        vtkXOpenGLRenderWindow: bad X server connection. DISPLAY=:11

    -- twice per plotter, straight to fd 2 from C, before falling back to the off-screen path it
    was going to use anyway. Unsetting DISPLAY means the attempt is never made: with no DISPLAY,
    VTK goes to EGL/OSMesa directly. Nothing here ever wants an on-screen window (`OFF_SCREEN` is
    set unconditionally), so there is nothing to lose by removing it.

    Idempotent, and safe to call before `import pyvista` or after.
    """
    os.environ.pop("DISPLAY", None)
    try:
        import vtkmodules.vtkCommonCore as _vcc          # keep VTK's own warnings off the terminal
        _vcc.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass



def open_movie(p, out, framerate, quality=8):
    """Open an mp4 that is READABLE WHILE IT IS BEING WRITTEN, and after a run is killed.

    A plain mp4 puts its index -- the `moov` atom -- at the END, because the muxer only knows the
    frame offsets once it has written them. So a run that dies, is killed, or runs out of memory
    before the writer closes leaves `ftyp` + `mdat`: tens or hundreds of MB of perfectly good H.264
    that no player will open. That is not hypothetical -- si_fill lost an 86 MB movie of a
    2M-particle run exactly this way, and si_bench_100m_fast would have lost 190 MB.

    `frag_keyframe+empty_moov` writes a self-contained index in front and one per fragment, so the
    file is valid from the first frame on. `-flush_packets 1` is the part that is easy to miss and
    without which the rest is decorative: ffmpeg buffers, and a fragmented file that is not flushed
    still sits at 36 bytes after 40 frames. Measured on a 400-frame run: readable at 232 frames
    while the writer was at frame 280.

    `-g 1` makes every frame a keyframe, which costs size and buys a file that can be cut anywhere.
    """
    p.open_movie(out, framerate=max(1, int(round(framerate))), quality=quality,
                 output_params=["-movflags", "frag_keyframe+empty_moov+default_base_moof",
                                "-g", "1", "-flush_packets", "1"])


def _plotter(size=None):
    offscreen()
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(size or SIZE, size or SIZE))
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    return p


# --------------------------------------------------------------------------- the three products
def kburns(run_dir, style, out, fill=1.0, label=None):
    """The finished specimen, turned once and zoomed in. Geometry fixed, camera moving."""
    fr = frames_of(run_dir)
    _lut, _blend, _cut = style_of(run_dir)
    _pl = plot_style(run_dir)
    _ec = str(_pl.get("edge_color", "") or "").lower()          # myosin | type | ""
    # DIVISION MARKS OFF WHEN SOMETHING ELSE IS BEING SHOWN. `show_div` paints mothers and daughters
    # red/blue over the face colour, which is the right default for watching a tissue grow and
    # exactly wrong while the picture is about myosin: two unrelated quantities on one surface, and
    # the reader cannot tell which is which. Declared, so a spec says what its picture is about.
    _div = bool(_pl.get("show_division", True))
    if not fr:
        return "no trajectory"
    L0 = box_of(run_dir, fr)
    name = label or os.path.basename(run_dir.rstrip("/"))
    pos, mt, act, _chem = fr[-1]
    m = mesh_of(pos, mt, act, show_div=False, chem=_chem, lut=_lut, blend=_blend,
                cutaway=_cut)
    n = int(KB_SECONDS * FPS)
    p = _plotter(); add(p, m, style)
    p.add_text(f"{name}  {style}", position="upper_left", font_size=11, color="white")
    open_movie(p, out, FPS)
    for i in range(n):
        u = i / (n - 1)
        aim(p, L0 * (1.0 - (1.0 - KB_ZOOM) * _ease(u)),
            azim=CAM["azim"] + 360.0 * u,                 # a FULL turn, so the clip loops
            elev=CAM["elev"] + 12.0 * np.sin(np.pi * u), fill=fill)
        p.write_frame()
    p.close()
    return f"{n} frames"


def evolve(run_dir, style, out, fill=1.0, label=None, max_frames=None):
    """The run through time, camera nailed down."""
    fr = frames_of(run_dir)
    _lut, _blend, _cut = style_of(run_dir)
    _pl = plot_style(run_dir)
    _ec = str(_pl.get("edge_color", "") or "").lower()          # myosin | type | ""
    # DIVISION MARKS OFF WHEN SOMETHING ELSE IS BEING SHOWN. `show_div` paints mothers and daughters
    # red/blue over the face colour, which is the right default for watching a tissue grow and
    # exactly wrong while the picture is about myosin: two unrelated quantities on one surface, and
    # the reader cannot tell which is which. Declared, so a spec says what its picture is about.
    _div = bool(_pl.get("show_division", True))
    if not fr:
        return "no trajectory"
    ticks_all = _frames_ticks.value
    cap = MAX_EVOLVE_FRAMES if max_frames is None else int(max_frames)
    if cap and len(fr) > cap:
        keep = np.unique(np.linspace(0, len(fr) - 1, cap).astype(int))
        fr = [fr[i] for i in keep]
        _frames_ticks.value = ([ticks_all[i] for i in keep] if ticks_all is not None
                               else keep.tolist())
    L = box_of(run_dir, fr)
    name = label or os.path.basename(run_dir.rstrip("/"))
    # ONE ACTIVATOR RANGE FOR THE WHOLE CLIP, taken over every recorded frame. Per-frame
    # normalisation would make a strengthening pattern look constant.
    vals = [np.asarray(a, float) for _p, _m, a, _c in fr if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0
    p = _plotter()
    open_movie(p, out, EV_FPS)
    # THE PAIR WINDOW, IN TICKS, matched to how long a mother stays marked: `age <= DIVIDED` is four
    # division CALLS and `cell_divide` runs every four ticks, so a mother is blue for about sixteen
    # ticks. The daughter threshold is therefore the face count from sixteen ticks ago, not from the
    # previous frame -- with the previous frame, three rows in four of a stride-1 trajectory drew
    # hundreds of mothers and no daughters at all.
    ticks = getattr(_frames_ticks, "value", None)
    nFs = [int(m_["nF"]) for _p, m_, _a, _c in fr]
    actor = txt = eactor = None
    # ONE RANGE FOR THE WHOLE CLIP, as for the activator: a per-frame myosin range would renormalise
    # every frame and a belt that is tightening would look constant.
    _erng = None
    if _ec == "myosin":
        _vals = [np.asarray(m_["e_myo"], float)[np.asarray(m_["E_face"]) < int(m_["nF"])]
                 for _p, m_, _a, _c in fr if m_.get("e_myo") is not None]
        if _vals:
            _erng = (float(min(np.nanmin(v) for v in _vals)),
                     float(max(np.nanmax(v) for v in _vals)))
    for t, (pos, mt, act, _chem) in enumerate(fr):
        back = _pair_reference(t, nFs, ticks, PAIR_TICKS)
        m = mesh_of(pos, mt, act, lo, hi, show_div=(style == "mesh" and _div), prev_nF=back,
                    chem=_chem, lut=_lut, blend=_blend, cutaway=_cut)
        if m is None:
            continue
        if actor is not None:
            p.remove_actor(actor)                  # NOT p.clear(): that removes the lights too
        if txt is not None:
            p.remove_actor(txt)
        if eactor is not None:
            p.remove_actor(eactor)
        actor = add(p, m, style)
        if _ec:
            em = edges_of(pos, mt, _ec, ntype=None, rng=_erng,
                          colors=list((_pl.get("colors") or {}).values()) or None,
                          lut=str(_pl.get("edge_lut", "inferno")))
            eactor = None if em is None else p.add_mesh(
                em, scalars="rgb", rgb=True, line_width=_pl.get("edge_width", 3.0),
                lighting=False, render_lines_as_tubes=True)
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
    pos, mt, act, _chem = fr[frame][:4]
    m = mesh_of(pos, mt, act, show_div=(style == "mesh"), chem=_chem,
                lut=style_of(run_dir)[0], blend=style_of(run_dir)[1],
                cutaway=style_of(run_dir)[2])
    p = _plotter()
    add(p, m, style)
    if label:
        p.add_text(f"{nm}  frame {len(fr) - 1 if frame == -1 else frame}",
                   position="upper_left", font_size=11, color="white")
    aim(p, L, fill=fill)
    p.screenshot(out or os.path.join(run_dir, "3d.png"))
    p.close()
    return f"{len(fr)} frames, drew {frame}"


def compare(dir_a, dir_b, out, style="mesh", fill=1.0, labels=("A", "B"), title="",
            max_frames=None):
    """A and B SIDE BY SIDE in one clip, stepped together. This is the comparison artefact.

    WHY IT IS NEEDED AT ALL, given both sides already have their own movie. Cedric: "I do not see
    the comparison folder". Two directories each holding a movie is not a comparison -- it is two
    movies, and telling them apart means opening both, scrubbing to the same moment, and holding one
    in your head. A digest says whether they agree; this says WHERE and HOW they differ when they do,
    which is the question a digest cannot answer and the only reason to keep the pixels at all.

    BOTH PANELS DRAW THE SAME FRAME INDEX, subsampled identically, on a SHARED camera box -- the max
    of the two -- because a per-side box would silently rescale one panel and make a size difference
    invisible. If the two sides record different numbers of rows (okuda keeps ~60, the core keeps
    every tick) each is subsampled to the same count first, so panel k of the left is panel k of the
    right in RUN time, not in row index.
    """
    import pyvista as pv
    fa, ta = frames_of(dir_a), _frames_ticks.value
    fb, tb = frames_of(dir_b), _frames_ticks.value
    if not fa or not fb:
        return "no trajectory on " + ("A" if not fa else "B")
    n = min(MAX_EVOLVE_FRAMES if max_frames is None else int(max_frames), len(fa), len(fb))
    ia = np.unique(np.linspace(0, len(fa) - 1, n).astype(int))
    ib = np.unique(np.linspace(0, len(fb) - 1, n).astype(int))
    n = min(len(ia), len(ib))
    L = max(box_of(dir_a, fa), box_of(dir_b, fb))
    # one activator range across BOTH sides, for the same reason one camera box: a per-side range
    # would normalise away exactly the difference the picture exists to show.
    vals = [np.asarray(a, float) for f in (fa, fb) for _p, _m, a in f if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(2 * SIZE, SIZE), shape=(1, 2), border=False)
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    open_movie(p, out, EV_FPS)
    # ONE LABEL FOR BOTH PANELS, but each side keeps ITS OWN ticks for `_pair_reference`. The two are
    # different questions: "what time is this frame" is shared, because the panels are aligned; "what
    # was nF a few ticks ago on THIS side" is per-side, and feeding it the other side's tick array
    # would silently mis-date the just-divided highlight.
    _lab_tk, _lab_idx = (ta, ia) if _has_ticks(ta) else (tb, ib)
    nFa = [int(m["nF"]) for _q, m, _a in fa]
    nFb = [int(m["nF"]) for _q, m, _a in fb]
    keep = [None, None]
    for k in range(n):
        when = _when(_lab_tk, int(_lab_idx[k]), max(len(fa), len(fb)))
        for col, (fr, idx, nFs, tk, lab) in enumerate(
                ((fa, ia, nFa, ta, labels[0]), (fb, ib, nFb, tb, labels[1]))):
            t = int(idx[k])
            p.subplot(0, col)
            if keep[col] is not None:
                p.remove_actor(keep[col])
            pos, mt, act, _chem = fr[t]
            m = mesh_of(pos, mt, act, lo, hi, show_div=(style == "mesh"),
                        prev_nF=_pair_reference(t, nFs, tk, PAIR_TICKS))
            if m is None:
                continue
            keep[col] = add(p, m, style)
            p.add_text(f"{lab}   {when}   {int(mt['nF'])} cells",
                       position="upper_left", font_size=10, color="white", name=f"t{col}")
            aim(p, L, fill=fill)
        if title:
            p.subplot(0, 0)
            p.add_text(title, position="lower_left", font_size=9, color="#9a9a9a", name="ttl")
        p.write_frame()
    p.close()
    return f"{n} paired frames"


def _when(tk, t, n_rows):
    """WHEN this panel is, in SIMULATION TIME rather than in row index.

    THE TWO SIDES DO NOT RECORD AT THE SAME RATE. okuda's `RunArchive` keeps about sixty rows --
    gate 00 recorded ticks [0, 6, 13, ... 394, 401] -- while the core keeps every tick, 402 of them.
    The comparison has always aligned them correctly, by okuda's own `ticks`; but the caption printed
    the ROW INDEX, so the two panels of a byte-identical run read "frame 60" beside "frame 402" and
    invited the one conclusion the picture exists to rule out. Both were tick 401. A label that makes
    a correct comparison look misaligned costs more than no label.

    ONE SIDE'S TICKS LABEL BOTH PANELS. okuda writes a `ticks` array; the core's `trajectory.npz`
    does not, so read literally the core panel can only say "row 402/402" -- which restates the very
    row index that caused the confusion. The panels are aligned BY CONSTRUCTION (the comparison
    selects core rows through okuda's own `ticks`), so the tick either side reports is the tick both
    are at, and `compare`/`compare_still` pass the one that exists for both. Falls back to the row
    index, and says so, only when NEITHER side recorded ticks.
    """
    try:
        return f"tick {int(np.asarray(tk).ravel()[t])}"
    except Exception:
        return f"row {t + 1}/{n_rows} (no ticks)"


def _has_ticks(t):
    """Whether a side recorded a tick array at all -- see `_when`."""
    try:
        return bool(np.asarray(t).ravel().size)
    except Exception:
        return False


def compare_still(dir_a, dir_b, out, style="flat", fill=1.0, labels=("A", "B"), title=""):
    """The last frame of both sides, side by side. The still the movie is scrubbed from."""
    import pyvista as pv
    fa, ta = frames_of(dir_a), _frames_ticks.value
    fb, tb = frames_of(dir_b), _frames_ticks.value
    if not fa or not fb:
        return "no trajectory on " + ("A" if not fa else "B")
    L = max(box_of(dir_a, fa), box_of(dir_b, fb))
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(2 * SIZE, SIZE), shape=(1, 2), border=False)
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    _lab_tk, _lab_n = (ta, len(fa)) if _has_ticks(ta) else (tb, len(fb))
    when = _when(_lab_tk, _lab_n - 1, _lab_n)
    for col, (fr, lab) in enumerate(((fa, labels[0]), (fb, labels[1]))):
        p.subplot(0, col)
        pos, mt, act, _chem = fr[-1]
        add(p, mesh_of(pos, mt, act, show_div=False), style)
        p.add_text(f"{lab}   {when}   {int(mt['nF'])} cells",
                   position="upper_left", font_size=10, color="white")
        aim(p, L, fill=fill)
    if title:
        p.subplot(0, 0)
        p.add_text(title, position="lower_left", font_size=9, color="#9a9a9a")
    p.screenshot(out)
    p.close()
    return "ok"


def use_plotting(pl: dict | None) -> None:
    """Install the live spec's `plotting:` block for this process -- see `plot_style`."""
    global _PLOT_OVERRIDE
    _PLOT_OVERRIDE = None if pl is None else dict(pl)


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
        # THE DEFAULT PAIR IS `movie.mp4` + `movie_kburns.mp4`, the same names the point renderer
        # writes, because a reader opening a run directory should not have to know which renderer
        # made it. They were `vtk_evolve_mesh.mp4` / `vtk_kburns_mesh.mp4`, so a mesh run's folder
        # looked nothing like every other run's and its `movie.mp4` was whatever the point renderer
        # had managed to draw of a mesh set -- which is a few hundred vertices in the corner of the
        # world box. The style suffix survives only where it distinguishes something: sequence 3
        # writes both the `mesh` and `nomesh` cuts and those must not collide.
        out = os.path.join(run_dir, {("evolve", "mesh"): "movie.mp4",
                                     ("kburns", "mesh"): "movie_kburns.mp4"}
                           .get((kind, st), f"movie_{kind}_{st}.mp4"))
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


# DARK GREY -> GREEN, the palette of the reference anatomy figure
# (connectome-gnn/figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py: a faint grey base
# pass with a green overlay whose alpha follows the per-neuron z-score). Here it is ONE
# per-vertex RGBA rather than two draw passes, which is the same idea and one actor.
#
# WHY ALPHA CARRIES THE ACTIVITY TOO, and not colour alone. In a cube where the tissue is
# nearly space-filling, a colour-only map paints every voxel of a 500,000-segment felt and the
# active neurites are lost inside it. Making quiet tissue TRANSPARENT as well as dim removes it
# from the frame instead of merely darkening it, and what is left is the structure that is
# active. `alpha_gamma > 1` biases that further: it keeps the mid-range faint so only the top
# of the distribution reads as solid.
GREY_GREEN = ("#2a2d2e", "#3f6b52", "#39ff7a")

# elongation-confidence bins (on `neurite_elong`). Four is enough to read as a gradient and
# cheap enough that the extra actors do not show in the frame time.
_BINS = ((0.00, 0.72), (0.72, 0.82), (0.82, 0.90), (0.90, 1.01))


def _rgba(x, lo, hi, colors=GREY_GREEN, a_lo=0.03, a_hi=0.95, gamma=1.6, knee=None):
    """[N, 4] uint8 RGBA: colour AND opacity both driven by |x| against one clim.

    TWO OPACITY CURVES, and the power law is the wrong one for cell bodies.

    `gamma` alone is `u**gamma`, which has no threshold: it is small everywhere below 1 and only
    approaches opaque as u approaches the very top of the clim. Measured on these four runs at
    gamma = 5, the median soma sits at alpha 0.0002 and the 90th percentile at 0.05 -- so a soma
    at the 90th percentile of activity is 95% transparent, which is why the ACTIVE ones looked
    too faint. Raising `a_hi` does not help; the curve is flat where the somas actually are.

    `knee=(t0, t1)` is a SMOOTHSTEP instead: fully transparent below t0, fully opaque above t1,
    and a smooth Hermite ramp between. That is the shape the eye wants for "off below a value,
    increasingly on above it" -- a real threshold with no hard edge, and the opaque end is
    actually reached rather than approached.
    """
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("gg", list(colors))
    u = np.clip((np.abs(np.asarray(x)) - lo) / max(hi - lo, 1e-12), 0, 1)
    rgb = (np.asarray(cm(u))[:, :3] * 255).astype(np.uint8)
    if knee is not None:
        t0, t1 = float(knee[0]), float(knee[1])
        w = np.clip((u - t0) / max(t1 - t0, 1e-12), 0.0, 1.0)
        f = w * w * (3.0 - 2.0 * w)                       # smoothstep
        a = a_lo + (a_hi - a_lo) * f
    else:
        a = a_lo + (a_hi - a_lo) * u ** gamma
    return np.concatenate([rgb, (a * 255).astype(np.uint8)[:, None]], axis=1)


def _tetra(radius, elong=1.0):
    """A tetrahedron with ONE VERTEX ON +X, optionally STRETCHED along that axis.

    `vtkGlyph3D` aligns a glyph's +X axis with the per-point vector, so the glyph has to have a
    well-defined nose along +X or "aligned with the direction" means nothing. `pv.Tetrahedron`
    is built in an arbitrary orientation; this one is apex-first by construction: the apex sits
    at (r, 0, 0) and the opposite face is the equilateral triangle at x = -r/3, circumradius
    2*sqrt(2)*r/3. A tetrahedron is also the cheapest solid that is CHIRAL ENOUGH to read as
    pointing -- a sphere shows no direction and a cylinder shows an axis but not a sense.
    """
    import pyvista as pv
    r, e = float(radius), float(elong)
    rb = 2.0 * np.sqrt(2.0) / 3.0 * r
    ang = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    # `elong` scales the X extent ONLY, so the glyph becomes a dart pointing along the neuron's
    # own axis. A regular tetrahedron shows a direction; an elongated one shows it at a glance,
    # because the eye reads aspect ratio far faster than it reads which vertex is nearest.
    pts = np.vstack([[r * e, 0.0, 0.0],
                     np.column_stack([np.full(3, -r * e / 3.0), rb * np.cos(ang), rb * np.sin(ang)])])
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 2, 3], [3, 0, 3, 1], [3, 1, 3, 2]])
    return pv.PolyData(pts, faces)


def _downsample(im, k):
    """Box-average an image by an integer factor -- the second half of supersampling.

    WHY SUPERSAMPLE AT ALL. A neurite is drawn as a LINE, and OpenGL clamps line width to a
    1-pixel minimum: at 896x896 every `line_width` below 1.0 renders identically, which is why
    an earlier 0.15 / 0.25 / 0.35 sweep produced three indistinguishable images. The only way
    to make a line thinner than a pixel is to make the pixel smaller -- render at k times the
    size and average down, so a 1-pixel line becomes a 1/k-pixel line with the coverage carried
    as intensity instead of width. It also antialiases half a million overlapping segments,
    which MSAA alone does not do well at this density.
    """
    h, w = im.shape[:2]
    return im.reshape(h // k, k, w // k, k, im.shape[2]).mean(axis=(1, 3)).astype(np.uint8)


def _grid(a):
    """A scalar volume as `pv.ImageData` SPANNING THE UNIT BOX, which is where the sets live.

    `pv.ImageData(dimensions=...)` defaults to spacing 1.0, so a 128^3 field would occupy world
    coordinates 0..128 while every Plexus set sits in 0..1. Mixed in one scene the geometry is
    then 128x too small and off in a corner -- which is exactly what happened, and what was
    misdiagnosed as "the volume mapper does not composite with geometry". It composites fine;
    the two were never in the same space. Setting the spacing is the whole fix.
    """
    import pyvista as pv
    n = np.asarray(a.shape)
    g = pv.ImageData(dimensions=n + 1)
    g.spacing = tuple(1.0 / n)
    g.origin = (0.0, 0.0, 0.0)
    g.cell_data["a"] = np.ascontiguousarray(a).ravel(order="F")
    return g


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
    open_movie(p, out, fps or EV_FPS)
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
    open_movie(p, out, fps or EV_FPS)
    actor = txt = None
    for t in range(g.shape[0]):
        vol = _grid(np.abs(g[t]))
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


def _set_text(actor, s):
    """Update a caption in place. `add_text(position="upper_left")` hands back a
    `vtkCornerAnnotation`, which takes `SetText(corner, s)` with corner 2 = upper left; a plain
    `vtkTextActor` takes `SetInput(s)`. Which one comes back depends on the position argument,
    so the caller cannot assume either."""
    if hasattr(actor, "SetInput"):
        actor.SetInput(s)
    else:
        actor.SetText(2, s)


def skeleton_lines(region, max_neurons=200, stride=1, seed=0):
    """SWC skeletons of a frozen region, as ONE `pv.PolyData` of line segments in the unit box.

    NEURITES, NOT DENDRITES. A NeuPrint skeleton is the WHOLE arbour of a body -- dendrite and
    axon together, with no compartment labels to separate them (fish2's SWC `type` column has
    zero `type == 1` soma rows, so nothing in the file distinguishes the two). Calling the layer
    "dendrites" would claim a split the data does not contain.

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
    pts, lines, owner, kept, total = [], [], [], 0, 0
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
        owner.append(np.full(len(xyz), len(pts) - 1, dtype=np.int64))
        lines.append(np.column_stack([np.full(len(seg), 2), seg + off]))
        off += len(xyz)
        kept += len(seg)
    if not pts:
        return None, {"neurons": 0, "segments": 0}
    poly = pv.PolyData(np.ascontiguousarray(np.concatenate(pts)))
    poly.lines = np.concatenate(lines).ravel()
    # WHICH NEURON EACH NODE BELONGS TO. Without it the only colour available for a neurite is
    # the voxelized FIELD sampled at that point -- which is the smoothed sum over every nearby
    # neuron, not this neuron's own state. An arbour coloured by the field and a soma coloured
    # by voltage are two different quantities on two different scales wearing one colour map.
    poly["owner"] = np.concatenate(owner)
    # THE BODY IDS ARE RETURNED, not just the count. A caller drawing somas beside these
    # arbours has to draw the SAME neurons, and reconstructing the subset from `max_neurons`
    # and a seed at the call site is how two layers of one picture come to disagree.
    body_ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
    body_ids = [body_ids[i] for i in range(len(files))]
    return poly, {"neurons": len(files), "segments": kept, "segments_before_clip": total,
                  "body_ids": body_ids}


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
    open_movie(p, out, fps or EV_FPS)
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


def evolve_neural(run_dir, out, region, field="neural_activity", n_arbours=None,
                  soma=True, volume=True, cmap="viridis", line_width=1.0,
                  neurites=True, neurite_opacity=0.30, neurite_stride=8,
                  # None MEANS "USE THE MEASURED PER-NEURON RADIUS". fish2 records no soma size
                  # (`somaRadius` is 0 on all 177,513 bodies), so this used to be one literature
                  # constant drawn on every cell alike; `compute_soma_radii` now sizes each
                  # soma from the ball around it that no other neuron's neurite enters, and the
                  # renderer scales each glyph by its own value. Pass a float to force a
                  # constant instead -- which is what a size sweep wants, and the fallback when
                  # a region predates the measurement.
                  soma_radius_um=None, soma_elong=2.5,
                  supersample=1, palette="grey_green", soma_glyph="sphere",
                  # a_lo = 0 and a HARD gamma: a quiet soma is fully transparent rather than
                  # faintly drawn. ~200 spheres at alpha 0.06 accumulate along a ray into
                  # exactly the haze that made solid spheres look Gaussian; at alpha 0 they
                  # contribute nothing and what is left in the frame is spheres.
                  # NEURITE CEILING QUARTERED, 0.85 -> 0.2125, FLOOR HELD AT 0.02. The alpha ramp
                  # is a_lo + (a_hi - a_lo) * u**gamma, so lowering only a_hi divides the
                  # brightest neurites by ~4 while a quiet one keeps the 0.02 it had -- the
                  # arbours stay as context and stop out-shouting the somas. Raising a_lo
                  # instead, or scaling both, would have faded the faint neurites to nothing,
                  # which is the one thing this change must not do: the point is to rebalance
                  # the two layers, not to delete one.
                  neurite_alpha=(0.02, 0.2125, 1.8), soma_alpha=(0.0, 1.0, 5.0),
                  # HARDCODED SOMA KNEE. u = 0.35 is the 75th percentile of |voltage| on these
                  # runs and u = 0.80 is about the 98th, so three quarters of the cell bodies
                  # are invisible and the top few per cent are solid -- which is the contrast
                  # the power law could not produce (it left the 90th percentile at alpha 0.05).
                  soma_knee=(0.35, 0.80),
                  # HOLD THEN FADE, in frames: draw the neurites at a constant colour for the
                  # first `hold` frames, ramp their opacity to zero over the next `fade`, then
                  # take them out of the scene entirely. The arbours are context -- they say
                  # where the tissue is -- and once that is established the movie is about the
                  # somas. It is also what makes large N affordable: the neurite layer is the
                  # expensive one, and this pays for it in 200 frames of 2,001 rather than all
                  # of them. None keeps the old behaviour, activity-coloured arbours throughout.
                  neurite_fade=(100, 100), neurite_const_q=0.90,
                  fill=0.92, label=None, fps=None, n_frames_max=None,
                  vol_opacity=(0, 0.006, 0.018, 0.045, 0.10)):
    """Soma, neurites and the rendered volume, in one clip -- by COMPOSITING TWO PASSES.

    ONE SCENE, and the story of why that took three tries is worth keeping. The skeletons at
    first did not appear at all, and the conclusion drawn was that VTK's ray-cast volume mapper
    cannot composite with polygonal geometry. That was WRONG. `pv.ImageData` defaults to
    spacing 1.0, so the 128^3 volume occupied world 0..128 while the sets occupy 0..1: the
    anatomy was 128x too small and outside the frame, not occluded. `_grid` sets the spacing,
    and volume, lines and points then depth-composite correctly in a single render.

    THE COLOUR IS ONE QUANTITY ON ONE SCALE. Soma and neurite are both coloured by the
    NEURON'S OWN |voltage|, with one clim. Colouring the arbour by the voxelized field instead
    -- which is what the first version did -- puts two different quantities on two different
    scales under one colour map: the field at a neurite is the smoothed sum over every nearby
    neuron, not that neuron's state. `skeleton_lines` returns a per-node `owner` so each
    arbour can carry its own neuron's value.

    LAYER ORDER IS SOMA FIRST, DENDRITES SECOND, both in the geometry pass -- the cell bodies
    are the entities the model actually integrates, and the arbours are context.

    `n_arbours=None` draws every neuron in the region. All 1,000 of them at `line_width=0.35`
    is dense but legible; at the 1.4 this started from it was an opaque felt.
    """
    import json
    import imageio_ffmpeg
    import pyvista as pv
    from matplotlib import colormaps
    from plexus.io.neuprint import region_path

    z = _traj(run_dir)
    # THE FRAME COUNT COMES FROM WHAT IS DRAWN. A field frame is a 128^3 grid -- 8.4 MB -- so
    # the engine strides it hard (`field_record_cap`), while the per-neuron voltage is 4 kB a
    # frame and is kept in full. Driving an anatomy-only clip off the field rows therefore threw
    # away 95% of the time resolution of data it never reads: a 2,000-frame run rendered as 125
    # frames. The field is loaded only when the volume is actually drawn.
    sv_all = np.asarray(z["neuron__voltage"], np.float32)[..., 0]              # [T_set, N]
    if volume:
        g = np.asarray(z[f"{field}__grid"], np.float32)
        if g.ndim == 5:
            g = g[:, 0]
        n_frames = g.shape[0]
        R = g.shape[1]
        lo, hi = _range(np.abs(g), 0.0, 99.9)
    else:
        g, R, lo, hi = None, None, 0.0, 1.0
        n_frames = sv_all.shape[0]
    cm = colormaps[cmap]

    def _rgb(v, a, b):
        return (np.asarray(cm(np.clip((v - a) / max(b - a, 1e-12), 0, 1)))[:, :3] * 255).astype(np.uint8)

    poly, meta = skeleton_lines(region, max_neurons=n_arbours, stride=neurite_stride)
    own = np.asarray(poly["owner"])                         # which neuron each node belongs to

    root = region_path(region)
    man = json.load(open(os.path.join(root, "manifest.json")))["region"]
    nz = np.load(os.path.join(root, "neurons.npz"), allow_pickle=True)
    spos = (np.asarray(nz["xyz_nm"], float) - np.asarray(man["bounds_lo_nm"])) / man["side_nm"]
    bid = np.asarray(nz["body_id"])
    sv = sv_all
    tv = (np.linspace(0, sv.shape[0] - 1, n_frames).round().astype(int)
          if volume else np.arange(n_frames))                                  # row -> set row
    # A LOOK CHECK IS NOT A CLIP. Deciding a glyph or an alpha ramp needs a handful of frames,
    # and rendering 2,001 to look at one is 20 minutes to answer a question that takes seconds.
    # SPREAD ACROSS THE RUN, NOT TRUNCATED TO ITS START: these networks begin near their initial
    # condition and the activity develops over the first few hundred steps, so the first six
    # frames are six pictures of a resting network -- which is what a truncating cap showed, and
    # it said nothing about the glyph it was rendered to check. The clim is still computed from
    # the whole trajectory, so the check is directly comparable to the full clip.
    if n_frames_max and n_frames_max < n_frames:
        tv = tv[np.linspace(0, n_frames - 1, int(n_frames_max)).round().astype(int)]
        n_frames = len(tv)
    pos_of = {int(b): i for i, b in enumerate(bid)}
    arbour_row = np.array([pos_of[int(b)] for b in np.asarray(meta["body_ids"])])
    if n_arbours:
        keep = np.isin(bid, np.asarray(meta["body_ids"]))
        spos, sv_s = spos[keep], sv[:, keep]
    else:
        sv_s = sv
    vlo, vhi = _range(np.abs(sv), 0.0, 99.5)                # ONE clim, both layers
    # SOMAS ARE WORLD-SIZED SPHERES, not screen-space points: a point of "size 4" is 4 pixels
    # whatever the zoom, which is not a cell body. `soma_radius_um` is a STATED CONSTANT here --
    # fish2 populates `somaRadius` on 0 bodies and the SWC radius is unusable (its per-neuron
    # max implies a 54 um diameter, larger than the cube). hemibrain, by contrast, measures it
    # on all 23,008 bodies at a 2.38 um median. At 2.5 um these fill 70% of the cube, which is
    # what the tissue actually looks like: median soma spacing here is 3.97 um.
    side_um = float(man["side_um"])
    _cloud = pv.PolyData(np.ascontiguousarray(spos))
    # 24 SEGMENTS, NOT 10. At 10 the silhouette is a visible decagon once a soma covers more
    # than a few pixels, and a faceted ball under a low alpha reads as a soft blob rather than
    # a sphere -- which is what "a mix of sphere and gaussian" was.
    # THE SOMA GLYPH IS A SPHERE AGAIN, and the tetra remains available behind `soma_glyph`.
    # A tetrahedron oriented by `neurite_dir` shows real per-cell information a ball throws
    # away -- the arbour's principal axis, which on these regions carries a median 82-91% of
    # each neuron's variance. But it spends the glyph's SHAPE on anatomy that is static, while
    # the thing the movie is about, the activity, has only colour and opacity left. A ball
    # reads as one cell body at any orientation, so nothing in the frame competes with the
    # thing that changes. The direction is still in the neuron state and still plotted, in
    # `fabric_directions_4x2.png`, where it can be read quantitatively rather than guessed at
    # from a cloud of darts.
    _el = np.asarray(nz["neurite_elong"], np.float32) if "neurite_elong" in nz.files else None
    if _el is not None and n_arbours:
        _el = _el[keep]
    # PER-NEURON RADIUS WHEN THE REGION MEASURED ONE. `plexus.io.neuprint.compute_soma_radii`
    # sizes each cell body from the ball around it that no other neuron's neurite enters, so
    # the somas can be drawn at their own sizes instead of one constant on all of them. A
    # `soma_radius_um=<float>` argument still forces a constant, which is what a size sweep
    # wants; None means "use the measurement if the region has one".
    _srad = np.asarray(nz["soma_radius_um"], np.float64) if "soma_radius_um" in nz.files else None
    if _srad is not None and n_arbours:
        _srad = _srad[keep]
    measured = soma_radius_um is None and _srad is not None
    r_um = soma_radius_um if soma_radius_um is not None else 1.2       # fallback constant
    if measured:
        # unit geometry scaled per point: vtkGlyph3D scales isotropically off a scalar array,
        # which is exactly a radius. The array is in UNIT-BOX units, the state is in um.
        _cloud["r"] = np.ascontiguousarray(_srad / side_um)
        _ball = (_tetra(1.0, elong=soma_elong) if soma_glyph == "tetra"
                 else pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24))
    elif soma_glyph == "tetra":
        _ball = _tetra(r_um / side_um, elong=soma_elong)
    else:
        _ball = pv.Sphere(radius=r_um / side_um, theta_resolution=24, phi_resolution=24)
    _ndir = np.asarray(nz["neurite_dir"], np.float32) if "neurite_dir" in nz.files else None
    if _ndir is not None and n_arbours:
        _ndir = _ndir[keep]
    if _ndir is not None:
        _cloud["neurite_dir"] = np.ascontiguousarray(_ndir[:, :3])

    name = label or os.path.basename(run_dir.rstrip("/"))
    ss = max(1, int(supersample))
    W = SIZE
    writer = imageio_ffmpeg.write_frames(out, (W, W), fps=fps or NEURAL_FPS, quality=8)
    writer.send(None)

    def _soma_mesh():
        """The glyphed somas AND, per glyph vertex, which neuron it belongs to.

        The index is what lets a persistent scene recolour: `vtkGlyph3D` copies the geometry
        once per point IN POINT ORDER, so vertex i belongs to neuron i // (points per glyph).
        Returned alongside the mesh because the tetra path glyphs in elongation BINS and the
        merged output is therefore not in neuron order -- deriving the mapping at the call site
        would silently be wrong for exactly the case that needs it most.
        """
        scale = "r" if measured else False
        if soma_glyph == "tetra" and _ndir is not None and _el is not None:
            out_m, out_i = None, []
            for lo_e, hi_e in _BINS:
                m = np.flatnonzero((_el >= lo_e) & (_el < hi_e))
                if not len(m):
                    continue
                w = np.clip((0.5 * (lo_e + hi_e) - 0.70) / (0.92 - 0.70), 0.0, 1.0)
                w = w * w * (3.0 - 2.0 * w)                       # smoothstep on confidence
                sub = _cloud.extract_points(m, adjacent_cells=False)
                gl = sub.glyph(geom=_tetra(1.0 if measured else r_um / side_um,
                                           elong=1.0 + (soma_elong - 1.0) * w),
                               scale=scale, orient="neurite_dir")
                out_i.append(np.repeat(m, gl.n_points // max(len(m), 1))[:gl.n_points])
                out_m = gl if out_m is None else out_m.merge(gl)
            return out_m, np.concatenate(out_i)
        orient = "neurite_dir" if (soma_glyph == "tetra" and _ndir is not None) else False
        gl = _cloud.glyph(geom=_ball, scale=scale, orient=orient)
        n_pt = _cloud.n_points
        return gl, np.repeat(np.arange(n_pt), gl.n_points // max(n_pt, 1))[:gl.n_points]

    gg_pal = palette == "grey_green"
    if not volume:
        # ---------------------------------------------------------------- persistent scene
        # THE GEOMETRY NEVER CHANGES; ONLY THE COLOURS DO. Rebuilding the scene per frame
        # re-glyphs 8.5M soma cells and re-uploads a 9.4M-segment mesh 2,001 times over --
        # measured at 3.12 s/frame on zf_Forebrain_8192 against 1.27 s when the scene is built
        # once and only the scalar arrays are rewritten. The colours are also evaluated PER
        # NEURON and gathered, not evaluated per vertex: the LUT is a pure function of the
        # value, so `_rgba(v)[own]` is bit-identical to `_rgba(v[own])` and ~1,000x less work.
        #
        # `p.render()` IS NOT OPTIONAL. Without it `screenshot()` returns the PREVIOUS image:
        # the timing looks even better (0.85 s/frame) and every frame of the movie is
        # identical. Measured frame-to-frame delta was exactly 0.0 -- a frozen clip that costs
        # less. The check below is what stops that shipping silently.
        balls, soma_of = _soma_mesh()
        p = _plotter(SIZE * ss)
        p.enable_depth_peeling(12)
        key = "rgba" if gg_pal else "rgb"
        n_actor = None
        if neurites:
            if neurite_fade:
                # WITHOUT INTENSITY: one constant colour for the whole arbour field, so the
                # neurites read as anatomy rather than as a second activity signal. The fade is
                # then a single actor opacity per frame, not 9.4M rewritten alphas.
                #
                # THE CONSTANT IS A QUANTILE OF |VOLTAGE|, and the choice is bounded on both
                # sides. At the top of the clim every segment draws full green at the alpha
                # ceiling and 9.4M of them overlapping turn the cube into a solid sheet with the
                # somas invisible behind it. At the median the arbours nearly vanish, because
                # the ramp is u**1.8 and the median u is small. `neurite_const_q` picks between
                # them; it is a legibility knob and says nothing about the dynamics, which is
                # why the intro carries no activity at all.
                const = np.full(poly.n_points,
                                float(np.quantile(np.abs(sv), neurite_const_q)), np.float32)
                poly["rgba"] = (_rgba(const, vlo, vhi, a_lo=neurite_alpha[0],
                                      a_hi=neurite_alpha[1], gamma=neurite_alpha[2])
                                if gg_pal else _rgb(const, vlo, vhi))
            else:
                v0 = sv[tv[0]][arbour_row]
                poly[key] = (_rgba(v0, vlo, vhi, a_lo=neurite_alpha[0], a_hi=neurite_alpha[1],
                                   gamma=neurite_alpha[2])[own] if gg_pal
                             else _rgb(v0, vlo, vhi)[own])
            n_actor = p.add_mesh(poly, scalars=key if not neurite_fade else "rgba", rgb=True,
                                 line_width=line_width, lighting=False,
                                 opacity=None if gg_pal else neurite_opacity)
        if soma:
            s0 = sv_s[tv[0]]
            balls[key] = (_rgba(s0, vlo, vhi, a_lo=soma_alpha[0], a_hi=soma_alpha[1],
                                gamma=soma_alpha[2], knee=soma_knee)[soma_of] if gg_pal
                          else _rgb(s0, vlo, vhi)[soma_of])
            lit = soma_glyph == "tetra"
            p.add_mesh(balls, scalars=key, rgb=True, lighting=lit, smooth_shading=not lit,
                       ambient=0.20, diffuse=0.80, specular=0.0, culling="back")
        layers = " + ".join([x for x, on in (("soma", soma), ("neurites", neurites)) if on])
        seg_txt = (f"{meta['segments'] // 1000}k segments (1/{neurite_stride})   "
                   if neurites else "")
        soma_txt = ((f"soma r={np.median(_srad):.2f} um median (measured)   "
                     if measured else f"soma r={r_um} um   ") if soma else "")
        txt = p.add_text("", position="upper_left", font_size=10 * ss, color="white")
        p.camera_position = "iso"
        p.reset_camera()
        p.camera.zoom(fill)
        hold, fadelen = (neurite_fade if neurite_fade else (n_frames, 0))
        prev, n_static, dropped = None, 0, None
        import time as _time
        _t0 = _time.time()
        for t in range(n_frames):
            if soma:
                s = sv_s[tv[t]]
                balls[key] = (_rgba(s, vlo, vhi, a_lo=soma_alpha[0], a_hi=soma_alpha[1],
                                    gamma=soma_alpha[2], knee=soma_knee)[soma_of] if gg_pal
                              else _rgb(s, vlo, vhi)[soma_of])
            if n_actor is not None:
                if neurite_fade:
                    o = 1.0 if t < hold else max(0.0, 1.0 - (t - hold) / max(fadelen, 1))
                    if o <= 0.0:
                        # REMOVED, not drawn at zero opacity. A transparent actor still
                        # rasterises and still costs its depth-peeling passes; taking it out of
                        # the scene is the entire point of the fade, and it is what makes the
                        # remaining ~90% of the clip cost soma-only.
                        p.remove_actor(n_actor)
                        n_actor, dropped = None, t
                    else:
                        n_actor.GetProperty().SetOpacity(float(o))
                else:
                    vv = sv[tv[t]][arbour_row]
                    poly[key] = (_rgba(vv, vlo, vhi, a_lo=neurite_alpha[0], a_hi=neurite_alpha[1],
                                       gamma=neurite_alpha[2])[own] if gg_pal
                                 else _rgb(vv, vlo, vhi)[own])
            _set_text(txt, f"{name}  {layers}   frame {t + 1}/{n_frames}   "
                           f"{meta['neurons']} arbours   {seg_txt}{soma_txt}"
                           + (f"ss{ss}x   " if ss > 1 else "")
                           + f"|voltage| clim [{vlo:.2f}, {vhi:.2f}]")
            p.render()                                    # see the note above -- NOT optional
            frame = np.asarray(p.screenshot(return_img=True))[..., :3]
            if ss > 1:
                frame = _downsample(frame, ss)
            if prev is not None and np.array_equal(frame, prev):
                n_static += 1
            prev = frame
            writer.send(np.ascontiguousarray(frame))
            # PROGRESS, because THE OUTPUT FILE IS NOT A PROGRESS INDICATOR. ffmpeg buffers the
            # mp4, so a long render sits at 0 bytes for hours and looks hung when it is not --
            # that reading is what got a batch of correct cluster jobs killed at 64 minutes.
            if t % 50 == 0 or t == n_frames - 1:
                print(f"[evolve_neural] {name}: frame {t + 1}/{n_frames}"
                      f"  {(_time.time() - _t0) / max(t + 1, 1):.1f} s/frame"
                      + ("  (neurites dropped)" if dropped is not None else ""), flush=True)
        p.close()
        writer.close()
        if n_static > n_frames // 2:
            raise RuntimeError(
                f"{name}: {n_static}/{n_frames - 1} consecutive frames were pixel-identical -- "
                f"the scene is not being re-rendered. This is the stale-screenshot failure; "
                f"check that p.render() runs before every screenshot.")
        return (f"{n_frames} frames, {meta['neurons']} arbours, {meta['segments']} segments, "
                f"soma={soma} neurites={neurites} volume=False"
                + (f", neurites dropped at frame {dropped}" if dropped else "")
                + (f", {n_static} static frame pairs" if n_static else ""))

    cam = None
    for t in range(n_frames):
        v = sv[tv[t]]
        p = _plotter(SIZE * ss)
        # translucent geometry must sort by depth against itself, not by draw order
        p.enable_depth_peeling(12)
        gg = palette == "grey_green"
        # neurites first (context), then somas on top (the entities the model integrates)
        if neurites:
            if gg:
                poly["rgba"] = _rgba(v[arbour_row][own], vlo, vhi, a_lo=neurite_alpha[0],
                                     a_hi=neurite_alpha[1], gamma=neurite_alpha[2])
                p.add_mesh(poly, scalars="rgba", rgb=True, line_width=line_width, lighting=False)
            else:
                poly["rgb"] = _rgb(v[arbour_row][own], vlo, vhi)
                p.add_mesh(poly, scalars="rgb", rgb=True, opacity=neurite_opacity,
                           line_width=line_width, lighting=False)
        if soma:
            key = "rgba" if gg else "rgb"
            _cloud[key] = (_rgba(sv_s[tv[t]], vlo, vhi, a_lo=soma_alpha[0], a_hi=soma_alpha[1],
                                 gamma=soma_alpha[2], knee=soma_knee)
                           if gg else _rgb(sv_s[tv[t]], vlo, vhi))
            # `orient="neurite_dir"` rotates each glyph so its +X nose follows that neuron's
            # own axis. A neuron with no skeleton has a zero vector and vtk leaves it unrotated,
            # which is the right default -- unoriented, not hidden.
            if soma_glyph == "tetra" and _ndir is not None and _el is not None:
                balls = None
                for lo_e, hi_e in _BINS:
                    m = (_el >= lo_e) & (_el < hi_e)
                    if not m.any():
                        continue
                    w = np.clip((0.5 * (lo_e + hi_e) - 0.70) / (0.92 - 0.70), 0.0, 1.0)
                    w = w * w * (3.0 - 2.0 * w)                   # smoothstep on confidence
                    sub = _cloud.extract_points(np.flatnonzero(m), adjacent_cells=False)
                    g = sub.glyph(geom=_tetra(1.0 if measured else r_um / side_um,
                                              elong=1.0 + (soma_elong - 1.0) * w),
                                  scale="r" if measured else False, orient="neurite_dir")
                    balls = g if balls is None else balls.merge(g)
            elif soma_glyph == "tetra" and _ndir is not None:
                balls = _cloud.glyph(geom=_ball, scale="r" if measured else False,
                                     orient="neurite_dir")
            else:
                balls = _cloud.glyph(geom=_ball, scale="r" if measured else False, orient=False)
            # FLAT FOR A BALL, LIT AND FACETED FOR A DART. With `lighting=False` a sphere draws
            # as a uniformly coloured disc: the silhouette is still a circle at the soma's true
            # world size, but nothing inside it varies, so every cell body carries EXACTLY the
            # colour its voltage maps to. Shading is the competing signal here -- a lit ball
            # spans a range of brightnesses from its own curvature, and that range is the same
            # visual channel the activity uses, so two cells at one voltage read as two values.
            # A dart is the opposite case: it needs per-face normals or its four faces take one
            # brightness and it collapses to a flat triangle, which is what it did.
            # BACK-FACE CULLING STILL MATTERS WITH LIGHTING OFF, because the glyphs are
            # alpha-blended -- the far hemisphere would otherwise composite under the near one
            # and every soma would draw at roughly twice its intended opacity.
            lit = soma_glyph == "tetra"
            p.add_mesh(balls, scalars=key, rgb=True, lighting=lit, smooth_shading=not lit,
                       ambient=0.20, diffuse=0.80, specular=0.0, culling="back")
        if volume:
            p.add_volume(_grid(np.abs(g[t])), scalars="a", cmap=cmap,
                         opacity=list(vol_opacity), clim=[lo, hi], show_scalar_bar=False)
        layers = " + ".join([x for x, on in (("soma", soma), ("neurites", neurites),
                                             ("128^3 volume", volume)) if on])
        seg_txt = (f"{meta['segments'] // 1000}k segments (1/{neurite_stride})   "
                   if neurites else "")
        soma_txt = ((f"soma r={np.median(_srad):.2f} um median (measured)   "
                     if measured else f"soma r={r_um} um   ") if soma else "")
        p.add_text(f"{name}  {layers}   frame {t + 1}/{n_frames}   "
                   f"{meta['neurons']} arbours   {seg_txt}{soma_txt}"
                   + (f"ss{ss}x   " if ss > 1 else "")
                   + f"|voltage| clim [{vlo:.2f}, {vhi:.2f}]",
                   position="upper_left", font_size=10 * ss, color="white")
        if cam is None:
            p.camera_position = "iso"
            p.reset_camera()
            p.camera.zoom(fill)
            cam = p.camera_position
        else:
            p.camera_position = cam
        frame = np.asarray(p.screenshot(return_img=True))[..., :3]
        if ss > 1:
            frame = _downsample(frame, ss)
        writer.send(np.ascontiguousarray(frame))
        p.close()
    writer.close()
    return (f"{n_frames} frames, {meta['neurons']} arbours, {meta['segments']} segments, "
            f"soma={soma} neurites={neurites} volume={volume}")
