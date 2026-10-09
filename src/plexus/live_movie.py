"""Write an mp4 WHILE the run proceeds, straight off the GPU, without reading the trajectory.

WHY THIS IS NOT `plexus.plot`. The normal route is: generate writes the trajectory, then
`plot_dataset` reads it back and renders. That works and stays the default for everything it can
handle. It stops working when the trajectory is the problem rather than the renderer: at 100 million
particles one recorded frame is $100\\times10^6 \\times 3 \\times 4$ B = 1.2 GB, so a 60-frame clip
would need 72 GB on disk before a renderer saw a pixel. Here the picture is taken while the state is
still in device memory and only the picture is kept.

It is wired in as an `on_frame` hook, which is the extension point `plexus.live.snapshot` already
uses, so `graph_data_generator` gains a renderer without gaining a plotting import: the pyvista
import lives inside `LiveMovie.__init__`, below the call that decides whether to build one.

TWO SUBSAMPLINGS, AND THEY ARE DIFFERENT THINGS.
  * `render_n` bounds how many particles are DRAWN. The simulation still runs all of them; this only
    bounds what VTK is asked to hold, because a point cloud of 100 M vertices is ~2.4 GB of VTK
    memory and tens of seconds a frame. A uniform random subset of a uniform-density fluid looks
    like the fluid: it is a sampling of the picture, not of the physics.
  * `max_frames` bounds how many frames are RENDERED, by striding. A 6000-frame Turing run must not
    pay a render 6000 times for a clip nobody will watch at that length.
Both are printed on the movie, because a subsampled render that does not say so is a lie about how
many particles ran.

THE `ms/frame` STAMPED ON THE MOVIE IS NOT THE THROUGHPUT FIGURE. It includes the device->host copy
and the VTK render and is slower by construction. `tools/mpm_bench.py` and `tools/mpm_warp_gate.py`
render nothing; quote those for throughput, and use `--no-viz` to run a generate with no renderer at
all.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

FLAT = dict(render_points_as_spheres=True, lighting=False, ambient=1.0, diffuse=0.0, specular=0.0)
_CS_AXIS = {"x": 0, "y": 1, "z": 2}

# THE GREEK LETTER, AND ONE FORMATTER FOR EVERY LENGTH THE OVERLAY PRINTS. The header and the
# scale bar each had their own, so a frame could carry `dx 0.000625 mm` above a bar reading
# `25 um` -- two units, two conventions and an ASCII stand-in for micro, for two numbers that are
# the same kind of thing. VTK's text renderer takes the real character.
def _si_length(metres: float) -> str:
    """`metres` as a value and an SI prefix, chosen so the number always reads AS A NUMBER.

    `%g` was the obvious formatter and it is the wrong one: at two significant figures 160 um
    comes out `1.6e+02 um`, an exponent on an overlay that is read at a glance. The prefix is
    picked so the value lands in [1, 1000) and the decimals follow its magnitude, so nothing ever
    renders in scientific notation and a trailing zero never survives.

    No centimetres: for a scene spanning micrometres, `200 cm` for 2 m is a worse answer than `2 m`.
    """
    unit, v = "m", metres
    for scale, u in ((1e-6, "\u00b5m"), (1e-3, "mm"), (1.0, "m"), (1e3, "km")):
        unit, v = u, metres / scale
        if v < 1000.0:
            break
    dec = 0 if v >= 100 else 1 if v >= 10 else 2 if v >= 1 else 3
    t = f"{v:.{dec}f}"
    # ONLY STRIP INSIDE A DECIMAL. `"160".rstrip("0")` is `"16"` -- the guard is not pedantry,
    # it silently turned a 160 um box into a 16 um one on the overlay.
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return f"{t} {unit}"
# EVERY `plotting.color_field` THE RENDERER KNOWS. Named in one place so the check that
# refuses an unknown one can quote the list, the way `plotting.renderer` already does.
_FIELDS = ("vorticity", "speed", "radial", "pressure", "deformation", "strain", "volume")


def _biggest_particle_set(H):
    """The set the movie is about: the largest Level that carries positions.

    Not hardcoded to `mpm_particle`. A composed cell run names its sets `nucleus`/`cytosol`, and a
    hook that looked for one name would silently render nothing on every spec that did not use it.
    """
    best, bn = None, -1
    for name, lvl in H.levels.items():
        try:
            pos = lvl.get("pos")
        except Exception:
            continue
        if pos is None or pos.ndim != 2:
            continue
        # BY LIVE COUNT, NOT BY BUFFER SIZE. `lvl.n` is the RESERVOIR, and a vertex set's reservoir
        # is sized for the tissue it will grow into: raising it to 524,288 for a long run made it
        # "bigger" than 500,000 material points while holding 396 live vertices, so the renderer
        # picked the mesh as the particle cloud, found no deformation gradient on it, and disabled
        # the movie for the whole run. A reservoir is a promise about the future; `occ` is the
        # present, and the present is what is being drawn.
        occ = getattr(lvl, "occ", None)
        k = int(occ.sum()) if occ is not None else int(lvl.n)
        # A MESH IS DRAWN AS A MESH, NOT AS THE CLOUD. `_add_meshes` draws every level that carries
        # a half-edge table whatever the subject is, so choosing the vertex set as the subject only
        # costs the picture its point sets: a tissue of 396 vertices holding 200 nuclei chose the
        # vertices and drew no nucleus. A mesh level is the subject only when nothing else has a
        # position: then the dots are its corners, as before.
        _m = getattr(lvl, "mesh", None)
        if _m is not None and int(_m.get("nF", 0) or 0):
            k = -0.5                                        # below any live count, above "none"
        if k <= bn:
            continue
        best, bn = name, k
    return best


def _no_strays(P, k=3.0):
    """The bound points, with any that have left the body pulled back onto its edge.

    A skinned surface RIDES the particles it was bound to at build time, so one particle that
    escapes later drags its surface vertices with it: at frame 267 of adh_poly_press a single
    membrane point near a box corner turned the cell into a long thin triangle reaching into that
    corner, while frame 135 of the same run was clean. The cell had not changed shape -- one point
    had left it, and the renderer drew the convex hull of a lie.

    Escapees are found ROBUSTLY, by distance from the median point rather than from the mean, and
    against the 99th percentile of that distance rather than its maximum: a genuinely elongated cell
    has a large 99th percentile and survives, while a point at `k` times it cannot be part of the
    same body. They are clamped to the edge rather than dropped, because the skin's vertex-to-point
    binding is fixed at build time and a missing row would misalign every vertex after it.
    """
    import numpy as np
    c = np.median(P, axis=0)
    d = np.linalg.norm(P - c, axis=1)
    lim = k * float(np.quantile(d, 0.99))
    if lim <= 0 or not np.isfinite(lim):
        return P
    bad = d > lim
    if not bad.any():
        return P
    Q = P.copy()
    Q[bad] = c + (P[bad] - c) * (lim / d[bad])[:, None]
    return Q


# THE ONLY CONVERSION PATH IN THE RENDERER. Imported under short names because the curve code
# below calls them per panel, and because naming them here makes it greppable that the plot is the
# ONE place in the tree where a simulation number becomes a physical one -- `plexus/units.py` says
# so and this is the surface it means.
from plexus.units import to_physical as _units_to_physical      # noqa: E402
from plexus.units import unit_label as _units_label             # noqa: E402



def body_contours(X, labels, dx, sigma=0.9, iso_frac=0.35, smooth_iter=20, min_points=8):
    """ONE CLOSED SURFACE PER BODY, each contoured on its own small grid: (merged PolyData with a per-point
    `body` array holding the label, number of bodies drawn), or (None, 0).

    X [n, 3] positions, labels [n] integer body of each point, dx the voxel side (world). Each body's points are
    counted into a grid that covers them with a 3-voxel margin, blurred by a gaussian of `sigma` voxels and
    contoured at `iso_frac` x the mean density of its occupied voxels (morph.reconstruct_contour's rule, per
    body), then smoothed. WHY LOCAL: a world-sized grid per body (`contour_by_type`'s path) costs the whole box
    for every one of a few hundred cells, and one grid for all of them fuses touching cells into one skin."""
    from scipy.ndimage import gaussian_filter
    import pyvista as pv
    X = np.asarray(X, np.float64); labels = np.asarray(labels).astype(np.int64)
    order = np.argsort(labels, kind="stable")
    ls, Xs = labels[order], X[order]
    cut = np.flatnonzero(np.diff(ls)) + 1
    parts = []
    for grp, lab in zip(np.split(Xs, cut), np.split(ls, cut)):
        if grp.shape[0] < min_points:
            continue
        lo = grp.min(0) - 3.0 * dx
        dims = np.ceil((grp.max(0) - lo) / dx).astype(int) + 4
        ijk = np.clip(((grp - lo) / dx).astype(int), 0, dims - 1)
        dens = np.zeros(tuple(dims), np.float32)
        np.add.at(dens, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
        dens = gaussian_filter(dens, sigma=sigma)
        g = pv.ImageData(dimensions=tuple(int(d) + 1 for d in dims), spacing=(dx, dx, dx), origin=tuple(lo))
        g["v"] = np.pad(dens, ((0, 1), (0, 1), (0, 1))).flatten(order="F")
        iso = max(iso_frac * float(dens[dens > 0].mean()), 1e-6)
        surf = g.contour([iso], scalars="v")
        if surf.n_points == 0:
            continue
        if smooth_iter:
            surf = surf.smooth(n_iter=int(smooth_iter), relaxation_factor=0.15)
        surf = surf.extract_surface() if not isinstance(surf, pv.PolyData) else surf
        surf.point_data.clear(); surf.cell_data.clear()
        surf["body"] = np.full(surf.n_points, int(lab[0]), np.int64)
        parts.append(surf)
    if not parts:
        return None, 0
    out = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
    out = out.extract_surface() if not isinstance(out, pv.PolyData) else out
    out = out.compute_normals(auto_orient_normals=False, consistent_normals=True, split_vertices=False)
    return out, len(parts)


def body_partition(X, labels, dx, reach, sigma=0.8, iso=0.55, smooth_iter=15, min_points=8, clip=None,
                   smooth_method="laplace", pass_band=0.05):
    """ONE SURFACE PER BODY FROM A PARTITION OF SPACE, so two bodies in contact share one face and never overlap:
    (merged PolyData with a per-point `body` array, number of bodies drawn), or (None, 0).

    Every voxel (side `dx`) is given to the body owning the nearest point, if that point is within `reach`
    (world), else to nobody; each body's voxels are blurred by a gaussian of `sigma` voxels and contoured at `iso`
    (0.5 is the partition's own boundary; above it leaves a thin seam between neighbours). `clip: (lo, hi)` gives no
    voxel outside that box to anyone, so no surface is drawn past a wall the run declares. WHY NOT A DENSITY
    CONTOUR (`body_contours`): the blur that makes a density smooth pushes every surface past its own material,
    so neighbouring cells overlap on screen and the flattened face where two cells press -- the contact
    deformation the picture is for -- cannot show (the human, exp 21, 2026-10-08)."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    import pyvista as pv
    X = np.asarray(X, np.float64); labels = np.asarray(labels).astype(np.int64)
    if X.shape[0] < min_points:
        return None, 0
    lo = X.min(0) - 2.0 * reach
    dims = np.ceil((X.max(0) + 2.0 * reach - lo) / dx).astype(int) + 1
    axes = [lo[k] + dx * (np.arange(dims[k]) + 0.5) for k in range(3)]
    V = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, j = cKDTree(X).query(V, k=1, distance_upper_bound=reach)
    lab = np.full(V.shape[0], -1, np.int64)
    ok = np.isfinite(d)
    if clip is not None:                       # `clip: (lo, hi)` -- no voxel outside the box belongs to a body
        clo, chi = np.asarray(clip[0], float), np.asarray(clip[1], float)
        ok &= np.all((V >= clo) & (V <= chi), axis=1)
    lab[ok] = labels[j[ok]]
    lab = lab.reshape(tuple(dims))
    parts = []
    order = np.argsort(labels, kind="stable")
    ls, Xs = labels[order], X[order]
    cut = np.flatnonzero(np.diff(ls)) + 1
    for grp, lb in zip(np.split(Xs, cut), np.split(ls, cut)):
        if grp.shape[0] < min_points:
            continue
        c = int(lb[0])
        i0 = np.clip(np.floor((grp.min(0) - reach - lo) / dx).astype(int) - 2, 0, dims - 1)
        i1 = np.clip(np.ceil((grp.max(0) + reach - lo) / dx).astype(int) + 3, 1, dims)
        sub = (lab[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] == c).astype(np.float32)
        if sub.sum() < 4:
            continue
        sub = np.pad(sub, 1)
        if sigma:
            sub = gaussian_filter(sub, sigma=sigma)
        org = lo + dx * (i0 - 1) + 0.5 * dx
        g = pv.ImageData(dimensions=tuple(int(n) for n in sub.shape), spacing=(dx, dx, dx), origin=tuple(org))
        g["v"] = sub.flatten(order="F")
        surf = g.contour([iso], scalars="v")
        if surf.n_points == 0:
            continue
        if smooth_iter:
            # `taubin` (windowed sinc) smooths WITHOUT SHRINKING: Laplacian smoothing strong enough to remove the
            # point-scale bumps of a 160-point cell also pulls its faces in, opening gaps between neighbours
            if smooth_method == "taubin":
                surf = surf.smooth_taubin(n_iter=int(smooth_iter), pass_band=float(pass_band),
                                          boundary_smoothing=False, normalize_coordinates=True)
            else:
                surf = surf.smooth(n_iter=int(smooth_iter), relaxation_factor=0.1, boundary_smoothing=False)
        surf = surf.extract_surface() if not isinstance(surf, pv.PolyData) else surf
        surf.point_data.clear(); surf.cell_data.clear()
        surf["body"] = np.full(surf.n_points, c, np.int64)
        parts.append(surf)
    if not parts:
        return None, 0
    out = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
    out = out.extract_surface() if not isinstance(out, pv.PolyData) else out
    out = out.compute_normals(auto_orient_normals=False, consistent_normals=True, split_vertices=False)
    return out, len(parts)


def _snap_range(lo, hi, ticks):
    """Round ends with a round step: the largest of {1, 2, 2.5, 5} x 10^k no bigger than
    range/(ticks-1), and the top is the first multiple of it above hi*1.02. Returns
    (lo, hi, tick_count). One implementation, used at setup and again when a live curve
    leaves its range."""
    raw = max(hi * 1.02 - lo, 1e-12) / max(int(ticks) - 1, 1)
    e = 10.0 ** np.floor(np.log10(raw))
    step = max([f * e for f in (1.0, 2.0, 2.5, 5.0) if f * e <= raw] or [e])
    n = int(np.ceil((hi * 1.02 - lo) / step))
    return float(lo), float(lo + step * max(n, 1)), max(n, 1) + 1


def _round1(v, up):
    """Snap to ONE significant figure, away from zero when `up` matches the sign."""
    if v == 0 or not np.isfinite(v):
        return 0.0
    e = 10.0 ** np.floor(np.log10(abs(v)))
    f = np.ceil(abs(v) / e) if (v > 0) == up else np.floor(abs(v) / e)
    return float(np.sign(v) * max(f, 1.0) * e)

GLYPH_MAX = 60_000                 # spheres per type before a type falls back to point sprites


class LiveMovie:
    """An `on_frame(H, tick)` hook that writes one mp4 for the whole run.

    Build it before `engine.run`, pass `__call__` as `on_frame`, and call `close()` afterwards.
    Every failure is swallowed and reported: a renderer that kills a twelve-minute simulation on the
    last frame is worse than no renderer.
    """

    def __init__(self, out, world, n_frames, up=2, render_n=400_000, max_frames=300,
                 fps=20, px=1280, dot=None, fill=0.9, elev=18.0, azim=-58.0, name="", seed=0,
                 sim=None, style=None, stills=10, keep_stills=False,
                 dt=None, time_s=None, real_time=True, length_um=None, centred=False, can_curve=False,
                 force_nN=None):
        # THE SPEC'S `plotting.fps` WAS DECORATIVE. `style` carries it, `fps` was a separate
        # keyword defaulting to 20, and nothing connected them -- so every movie was written at 20
        # regardless of what the spec asked for. It matters twice over now: `fps` sets the mp4's
        # framerate AND the render stride that makes playback real time, and the two must agree or
        # the clock in the overlay is a claim about a file that does not keep it. si_gate at
        # fps 20 took stride 60 and held 30 frames for 1.5 s of world -- arithmetically real time,
        # but so heavily aliased that it reads as several times too fast. At 60 it is stride 20 and
        # 90 frames, the same 1.5 s, smooth.
        fps = float((style or {}).get("fps", fps))
        # THE CANVAS, BECAUSE A DOT CANNOT BE SMALLER THAN A PIXEL. `dot_size` below ~1.0 buys
        # nothing -- VTK draws a point at one pixel minimum, so 0.2 and 0.7 are the same picture.
        # What makes a dot smaller RELATIVE TO THE SCENE is more pixels to put it in: at 100 M
        # particles in a 1280 frame there are 61 particles per pixel and the galaxy is a white
        # blob, while the same run at 2560 has 15 and the arms separate. This is the knob that
        # matters at high N, and it was not reachable from a spec.
        px = int((style or {}).get("render_px", px))
        from plexus.render_vtk import offscreen
        offscreen()                                   # kill the Xlib chatter before VTK loads
        import pyvista as pv
        pv.OFF_SCREEN = True
        # AN ACTOR THAT STARTS EMPTY IS NOT AN ERROR HERE. The graph overlay creates its line
        # meshes up front and fills them only in the closing frames, and pyvista refuses to add a
        # mesh with zero points -- which took the whole movie down at frame 0 rather than the one
        # actor. The alternative, adding the actors lazily mid-run, is worse: it changes the
        # renderer's actor list partway through a clip.
        pv.global_theme.allow_empty_mesh = True

        self.pv = pv
        self.sim, self.style = sim, dict(style or {})
        # `plotting.camera: {elev, azim}`, IN DEGREES -- the key `render_vtk` already reads for the
        # mesh renderer. This renderer took elev/azim only as constructor defaults (18, -58) and no
        # caller passed them, so a spec could not ask for a top view: a flat sheet was always seen
        # from 18 degrees above its plane. elev 90 looks straight down the up axis.
        _cam = self.style.get("camera") or {}
        if isinstance(_cam, dict):
            elev = float(_cam.get("elev", elev))
            azim = float(_cam.get("azim", azim))
        self.out, self.name, self.n_frames = out, name, int(n_frames)
        self.render_n, self.seed = int(render_n), int(seed)
        # DOT SIZE IS NOT A PROPERTY OF THE PICTURE, it is a property of the picture RELATIVE TO THE
        # SPACING -- `plexus.live.dot_area_pt2` says so and this class ignored it, with a fixed
        # 1.4 px. The consequence was visible immediately: a 100 M run drawn at 1 particle in 250
        # rendered as a thin spray while a 94.5 k run drawn in full rendered as a solid slab, and
        # the two hold THE SAME 1.41% of the box as material. What differs is only how densely the
        # renderer samples it, and a dot sized to the DRAWN spacing puts that right: the picture
        # then shows the material at the resolution actually drawn, instead of showing 250x fewer
        # dots at the size that suited 250x more of them.
        # THE SPEC ALREADY SAYS HOW BIG A DOT IS. `plotting.dot_size` is declared in essentially
        # every material spec and `plot.py` honours it; this renderer did not, so a config that
        # said 1.2 got whatever the CLI defaulted to. Precedence: an explicit argument (the CLI)
        # beats the spec, the spec beats "auto".
        if dot is None:
            dot = self.style.get("dot_size", "auto")
        self.dot, self.fill = dot, float(fill)
        # A CROSS SECTION, AS AN OVERLAY. `plotting.cross_section` selects a slab normal to one axis
        # and scatters what is inside it in the plane of the other two -- for a jet falling in -y an
        # xz slice shows the stream's footprint, so a round column and a broken-up turbulent one look
        # different at a glance where the 3D view shows only its silhouette.
        #
        # It is a vtkChartXY OVERLAY on the same renderer, not a second render pass: the frame still
        # costs one `write_frame()`, and the stills keep coming out of that same image.
        _cs = (style or {}).get("cross_section")
        self.cs = None
        if _cs:
            _cs = {} if _cs is True else dict(_cs)
            self.cs_axis = _CS_AXIS[str(_cs.get("axis", "y")).lower()]
            self.cs_at = float(_cs.get("at", 0.35))          # fraction of the box along that axis
            # THICKNESS IS THE FULL SLAB, IN CELLS, and halved here. It read as a half-width
            # before, so `thickness: 4` cut a slab 8 cells (8.3 mm) thick -- two diameters of a 4 mm
            # sphere seen at once, which is why the column looked solid rather than a sheet.
            self.cs_cells = float(_cs.get("thickness", 4.0)) / 2.0
            self.cs_max = int(_cs.get("max_points", 6000))
            # `only: true` REPLACES the 3D view rather than sitting in its corner. A slice IS the
            # better picture for a jet: the 3D column is an opaque silhouette that hides its own
            # interior, and the wake behind an obstacle is exactly the thing a silhouette cannot
            # show. As an inset it is legible but small; as the whole frame it is the movie.
            self.cs_only = bool(_cs.get("only", False))
            self.cs_cfg = _cs
        self.px_used = None
        self._curve_labels = []         # one on-panel value readout per declared curve
        self._rate_of = "compute"       # `replay` sets "render" or "compute": see below
        self._fixed_ms = None           # replay: the run's own median ms/tick, from the trajectory
        self.up = int(up)
        # (reset to 1 for 2D below, once the world tells us the run is planar)
        self.stride = max(1, int(np.ceil(self.n_frames / max(1, int(max_frames)))))
        # REAL TIME, WHICH IS ONLY MEANINGFUL ONCE A RUN HAS UNITS. One simulated frame lasts
        # `dt * time_s` SECONDS, so playing at `fps` shows real time exactly when the render stride
        # is 1/(fps * dt * time_s). Without a `units:` block `time_s` is 1.0 by default and the run
        # is dimensionless, so this computes a stride for a second that means nothing -- hence it
        # only engages when the units were DECLARED, and otherwise the movie is what it always was.
        #
        # It cannot always be reached: a run whose frames are further apart than 1/fps of real time
        # is already faster than real when every frame is drawn, and one with a huge frame count
        # would need a stride so large the motion aliases. Both are reported rather than silently
        # accepted, because "this movie is real time" is a claim.
        # REAL TIME BY DERIVING THE FRAMERATE, not by thinning the movie.
        #
        # The movie is capped at `max_frames` (300) because that is what bounds the file, and the
        # stride follows from it: ceil(n_frames / max_frames). What makes playback real time is then
        # the FRAMERATE, which is not a free choice at all --
        #
        #     fps = frames_rendered / (n_frames * dt * time_s)
        #
        # -- because the video must last exactly as long as the world it shows. For a 1.5 s run cut
        # to 300 frames that is 200 fps. High, and legal in H.264; a player that cannot honour it
        # shows the movie SLOWER than real, never faster, which is the safe direction to fail in.
        #
        # This replaces a `playback` speed knob, which was a way of asking for slow motion and is
        # not what a movie with units should do: it should show the world at the rate the world
        # ran, and if that is too fast to watch, the answer is a longer run, not a slower film.
        # THE SPEC MAY OVERRIDE THE REAL-TIME CLOCK, and some must. `real_time` picks the framerate
        # so the movie runs at the world's own pace, which is what a 1.28 s MPM run wants and is
        # nonsense for a tissue at 600 s a frame: 402 frames of that is 2.8 DAYS, so honest playback
        # asks for 0.0002 fps, every player clamps it, and the file then lies about its duration.
        # It was reachable only as a CLI flag (`--no-real-time`), i.e. not from the artefact that
        # describes the run.
        self.dt, self.time_s = dt, time_s
        self.real_time = bool((self.style or {}).get("real_time", real_time))
        if float((self.style or {}).get("duration_s", 0.0) or 0.0) > 0.0:
            self.real_time = False                  # a declared length outranks the world clock
        self.length_um = length_um
        # THE RUN'S DECLARATION, REBUILT FOR THE ONE FUNCTION ALLOWED TO CONVERT. The renderer is
        # handed the two scales as bare floats by its caller; `plexus.units.to_physical` wants the
        # `Units` those floats came from, and `declared` is the whole difference between a number
        # that may carry a unit and one that may not -- so a run with no `units:` block gets
        # `declared=False` here and every panel prints bare, which is what it should always have
        # done. `force_nN` is absent because no curve quantity needs it.
        from plexus.units import Units as _Units
        # THE FORCE SCALE TOO, or nothing built on force can be quoted here: an energy, a stress, a
        # voltage (energy per elementary charge). The renderer had length and time only, so a
        # curve of the membrane potential declared `unit: voltage` converted to None and drew
        # nothing (exp_02 colour test, 2026-09-23).
        self._units = _Units(length_um=float(length_um or 1.0),
                             time_s=float(time_s if time_s is not None else 1.0),
                             force_nN=(float(force_nN) if force_nN is not None else None),
                             declared=bool(length_um))
        # THE GRID, BESIDE THE PARTICLE COUNT. The two together are what actually determines
        # whether a run resolves anything: 100M particles on a 96^3 grid is 8,176 per cell and a
        # picture of nothing in particular, while the same 100M on 330^3 is the MPM convention of 8.
        # The particle count alone has been the headline on every movie in this corpus and it is the
        # half that flatters.
        self._grid_label = ""
        _fl = getattr(sim, "fields", None) or {}
        _ng = next((int(fc["n_grid"]) for fc in _fl.values()
                    if isinstance(fc, dict) and "n_grid" in fc), None)
        if _ng:
            _d = len(world) if world is not None else 3
            # NO CELL COUNT. The MPM background grid is made of "cells" and so is the biology, and
            # a header reading "4.1M cells" over a picture of six of them is a collision of two
            # meanings in the one place a reader looks first. The RESOLUTION is what the number was
            # for, and `grid 160^3` already says it.
            _fc0 = next((fc for fc in _fl.values() if isinstance(fc, dict) and "n_grid" in fc), {})
            # THE GRID'S OWN BOX when it declares one (`box`, or `cell_box` for a grid per body), not the
            # world's: exp 11's 4.7-unit per-cell blocks read "dx 200 um" off the 200-unit world.
            _gb = (float(_fc0["cell_box"]) if _fc0.get("cell_box") is not None else
                   float(_fc0["box"][1]) if _fc0.get("box") is not None else
                   float(world[1] if len(world) > 1 else world[0]))
            self._grid_label = (f"   grid {_ng}^{_d} per {_fc0['per_parent']}" if _fc0.get("per_parent")
                                else f"   grid {_ng}^{_d}")
            if length_um:                       # with units, the spacing has a SIZE worth quoting
                _dx = _gb / _ng * float(length_um) / 1.0e6
                # IN THE SAME UNIT THE SCALE BAR PICKS, and at two significant figures. `dx
                # 0.000625 mm` is the same length as `0.62 um` written to be unreadable, and it
                # disagreed with a scale bar standing right below it saying `25 um`.
                self._grid_label += f", dx {_si_length(_dx)}"
        self._box_label = ""
        if length_um and time_s is not None:
            _m = float(length_um) / 1.0e6
            _w = [float(x) * _m for x in world]
            # THREE SIGNIFICANT FIGURES. `:g` on a measured box printed "1.17233 mm cube" -- six
            # digits of a number the reader is being given for scale, where the point is the order
            # of magnitude and the leading figure. (The scale BAR is a chosen round number and is
            # exact; this is a measurement and is rounded.)
            # ONE FORMATTER FOR EVERY LENGTH ON THE OVERLAY. The box had its own ladder,
            # so a 160 um scene was announced as "0.16 mm cube" above a bar reading "25 um".
            self._box_label = ("   box " + " x ".join(_si_length(v) for v in _w)
                               if len(set(_w)) > 1 else f"   box {_si_length(_w[0])} cube")
        self.speed = None
        self.fps = float(fps)          # the declared rate; both branches below refine it
        # SLOW MOTION APPLIES EITHER WAY. It lived entirely inside the real-time branch, so a spec
        # that turned the clock off -- which a tissue at 600 s a frame must -- silently lost the one
        # knob that says how fast to play the result. "Play it twice as slowly" is a statement about
        # the FILE, and it is true whether the base framerate came from the world clock or from
        # `plotting.fps`. Off the clock it simply halves the declared rate; nothing is dropped or
        # resampled, the same frames take longer.
        self.slow_motion = float((self.style or {}).get("slow_motion", 1.0))
        if self.slow_motion <= 0:
            raise ValueError(f"plotting.slow_motion must be > 0, got {self.slow_motion}")
        if not self.real_time:
            # A MOVIE OF A DECLARED LENGTH. `plotting.duration_s` (video seconds) sets the framerate
            # from the frames actually rendered, fps = n_rendered / duration_s, so a 600-frame run
            # cut to 200 movie frames plays as long as a 300-frame run cut to 300: the builder's
            # record wants every movie 8 s, and a fixed `fps` could only promise that for one frame
            # count. Slow motion then multiplies the length as it does everywhere else.
            _len = float((self.style or {}).get("duration_s", 0.0) or 0.0)
            if _len > 0.0:
                # AN INTEGER FRAMERATE, so the container's timescale is exact: 37.5 fps wrote files a
                # browser timed at 7.89 s of 8 and, on the human's screen, at minutes (2026-09-24).
                # 200 frames in 10 s is 20 fps, 300 is 30 fps; a 1-frame still plays 1 s.
                _nr = max(1, self.n_frames // self.stride)
                fps = float(max(1, int(round(_nr / _len))))
                print(f"[live-movie] duration_s {_len:g}: {_nr} movie frames at {fps:g} fps "
                      f"= {_nr / fps:.3g} s of video", flush=True)
            self.fps = fps = float(fps) / self.slow_motion
            if abs(self.slow_motion - 1.0) > 1e-9:
                print(f"[live-movie] {fps * self.slow_motion:g} fps declared / "
                      f"{self.slow_motion:g}x slow = {fps:g} fps", flush=True)
        if self.real_time and dt and time_s:
            frame_s = float(dt) * float(time_s)
            self.duration_s = self.n_frames * frame_s
            # SLOW MOTION AS A DECLARED FACTOR, not as a thinner movie. The frame COUNT is fixed
            # by max_frames and the stride follows from it; slowing the film down is then purely a
            # matter of the framerate, `fps = frames / (duration * slow_motion)`. Nothing is
            # dropped, nothing is resampled -- the same 300 frames simply take 4x longer to play,
            # which is what slow motion is. `slow_motion: 1` is real time.
            _sm = float((style or {}).get("slow_motion", 1.0))
            if _sm <= 0:
                raise ValueError(f"plotting.slow_motion must be > 0, got {_sm}")
            n_rendered = max(1, self.n_frames // self.stride)
            # A ZERO-LENGTH RUN HAS NO DURATION TO DIVIDE BY. `n_frames: 0` is a legitimate request
            # -- it renders the seeded scene and nothing else, which is what the studio's preview
            # wants -- but `duration_s` is then 0 and this was a ZeroDivisionError inside the movie
            # writer, so the run died after building the whole hierarchy. A single frame has no
            # playback rate to get right; 1 fps is as true as any other.
            _dur = self.duration_s * _sm
            fps = (n_rendered / _dur) if _dur > 0 else 1.0
            self.slow_motion = _sm
            self.speed = float(fps) * self.stride * frame_s      # world-seconds per video-second
            self.fps = fps                                       # <- what open_movie must use
            _how = "real time" if abs(_sm - 1.0) < 1e-9 else f"{_sm:g}x slow motion"
            print(f"[live-movie] {self.n_frames} frames of {self.duration_s:.4g} s -> stride "
                  f"{self.stride}, {n_rendered} movie frames at {fps:.4g} fps = {_how} "
                  f"({n_rendered / fps:.4g} s of video)"
                  + ("" if 5.0 <= fps <= 120.0 else
                     f"  (NOTE: {fps:.4g} fps is outside the 5-120 most players honour; if it is "
                     f"clamped the movie runs SLOW, not fast)"), flush=True)
        # STILLS COME OUT OF THE MOVIE'S OWN RENDER, not a second one. `Plotter.image` is the frame
        # `write_frame()` just rasterised, so a PNG costs a file write and nothing else -- no extra
        # render pass, and no second copy of the camera/palette/dot-size code to drift out of sync
        # with this one. They are therefore chosen from the RENDERED ticks: a still on a tick the
        # movie skipped would have no image to copy and would have to re-render, which is the thing
        # being avoided.
        _rendered = list(range(self.stride, self.n_frames + 1, self.stride)) or [self.n_frames]
        n_st = max(0, int(stills))
        self.still_ticks = (set(np.unique(np.linspace(0, len(_rendered) - 1, n_st).astype(int))
                                .tolist()) if n_st else set())
        self.still_ticks = {_rendered[i] for i in self.still_ticks}
        self.still_dir = os.path.dirname(out) or "."
        self.stills_written = 0
        # THE NUMBERED STILLS ARE A LIVE-PROGRESS ARTEFACT, NOT AN OUTPUT. Their job is to let a
        # run be watched while it is running; once the mp4 exists they are 10 redundant copies of
        # frames the movie already holds, and across a spec library they add up -- 851 files and
        # 0.21 GB from one night's material runs alone. `3d.png` is kept, because that is the one
        # a file browser is pointed at, and it is the final frame.
        self.keep_stills = bool(keep_stills)
        self._still_paths = []
        self.cloud = self.idx = None
        # INITIALISED HERE, NOT ONLY IN `_skin_build`. That builder runs only for
        # `render_3d: surface`, so every OTHER spec reached `_skin_update` with the attribute
        # never created and died on `AttributeError: 'LiveMovie' object has no attribute
        # '_skin'` at frame 2 -- which the class swallows and turns into "movie DISABLED",
        # so a run still finished, still wrote its trajectory, and silently had no movie.
        self._skin = self._surf = self._skin_sub = None
        self._skins = []                      # `render_3d: compartments`: one bound surface per type
        self._graphs = []                     # `plotting.graph_overlay`: a relation drawn as lines
        self._graph_ticks = {}
        self._meshes = []
        self._mesh_is_subject = False
        self._curves = []
        self._cs_rng = None
        self.drawn = self.n = self.rendered = 0
        self.t0 = None
        self.failed = None
        self.colour_by = "?"
        self.n_obstacles = 0

        px = int(px) // 16 * 16                       # ffmpeg's macro_block_size; see cell_panels
        # WIDER ONLY WHEN THERE IS SOMETHING TO PUT THERE. The frame is square because a scene is,
        # and a curve panel then has nowhere to go but on top of it -- over the box, and over the
        # tissue on the frames where it has grown. Declaring `curve` adds a COLUMN to the right and
        # the camera is shifted left by the same amount below, so the panels sit beside the scene
        # rather than in front of it. A spec with no curves is unchanged, pixel for pixel.
        # WIDEN ONLY IF THE PANELS WILL BE DRAWN. The curve's axes are fixed over the whole clip, so
        # only the REPLAY can build them -- and the live path was still reserving the column, giving
        # a frame a third wider than it needed with a band of black down the right. `can_curve` is
        # the caller saying which path this is, not a guess from the style.
        # ...OR WHEN EVERY PANEL DECLARES ITS RANGE. `_curves_setup` draws live exactly then, so the
        # column has to be reserved then too, or the chart is placed at x = 1.005 and pyvista
        # refuses it -- which disabled the whole movie after frame 0 the first time this ran.
        _cv = (style or {}).get("curve")
        _cvl = [_cv] if isinstance(_cv, dict) else list(_cv or [])
        if not can_curve and not (_cvl and all(("ymin" in c and "ymax" in c) for c in _cvl)):
            _cv = None
        _ncv = 0 if not _cv else (1 if isinstance(_cv, dict) else len(_cv))
        # THE COLUMN IS THE PANEL PLUS A MARGIN, and the ASPECT is what makes the scene fit beside
        # it -- 1 + column is not enough. Parallel projection fits the box to the frame's HEIGHT, so
        # the box is half-width 0.5/aspect of the width; to sit inside a scene column of (1 - c) it
        # needs 0.5/aspect <= (1 - c)/2, i.e. aspect >= 1/(1 - c). At c = 0.32 that is 1.47, and
        # 1 + c = 1.32 left the box hanging off the left edge with the scale bar cut in half.
        # A SQUARE PANEL, WHICH IS A CONSTRAINT ON THE ASPECT AND NOT ON THE PANEL. `size` is in
        # WINDOW fractions and the window is not square, so (0.30, 0.26) drew a panel 614 x 333 px.
        # Squareness ties the two: a panel `h` of the height is h*px tall, so it must be h*px wide,
        # which is h/aspect of the width. The aspect then has to leave room for the box beside it --
        # 0.5/aspect <= (1 - c)/2 with c = h/aspect + margin -- which solves to
        # aspect >= (1 + h)/(1 - margin), plus 9% so the box is not flush against the edge.
        _ch = float((style or {}).get("curve_height", 0.26))
        _mg = float((style or {}).get("curve_margin", 0.03))
        _asp = float((style or {}).get("movie_aspect",
                                       1.0 if not _ncv else
                                       round((1.0 + _ch) / max(1.0 - _mg, 0.2) * 1.09, 3)))
        self._curve_size = (_ch / _asp, _ch)
        self._curve_col = 0.0 if not _ncv else self._curve_size[0] + _mg
        pxw = max(16, int(px * _asp) // 16 * 16)
        self.aspect = pxw / float(px)
        self.p = pv.Plotter(off_screen=True, window_size=(pxw, px), border=False)
        self._set_lights()
        # `surface_env: true` -- IMAGE-BASED LIGHTING, so a reflective surface has something to
        # reflect. A PBR material with `metallic`/`roughness` and no environment map reads as dark
        # plastic with a few white flecks: there is nothing in the scene for it to pick up except
        # the point lights. `plexus.morph` already builds a synthetic sky (there is no HDRI on a
        # cluster node) and a key/fill/rim/back rig for exactly this look; both are reused here
        # rather than copied, so the movie and the morphing script cannot drift apart.
        if bool((self.style or {}).get("surface_env", False)):
            try:
                from plexus.morph import build_lights, default_settings, env_texture
                _es = default_settings()
                _es["env_bright"] = float((self.style or {}).get("surface_env_bright",
                                                                 _es["env_bright"]))
                self.p.set_environment_texture(self._env_texture(_es["env_bright"]))
                build_lights(self.p, _es)
                print(f"[live-movie] environment map + 4-light rig (bright {_es['env_bright']:g})",
                      flush=True)
            except Exception as _e:                                  # noqa: BLE001
                print(f"[live-movie] surface_env unavailable ({type(_e).__name__}: {_e})",
                      flush=True)
        # `plotting.background`: black unless the spec says otherwise. The structural-biology
        # figure is drawn on WHITE (every cryo-EM panel of Drobnic et al. 2025, and the ChimeraX
        # default), and on white every overlay -- scale bar, its label, the frame header, the
        # graph pens -- has to turn dark or vanish; `self._fg` is that one colour, read wherever
        # the overlays used to say "white". Added for builder/exp_02_bacterium step 0013.
        _bg = str(self.style.get("background", "black") or "black")
        self.p.set_background(_bg)
        try:
            from matplotlib.colors import to_rgb as _to_rgb
            self._fg = "black" if sum(_to_rgb(_bg)) > 1.5 else "white"
        except Exception:                                            # noqa: BLE001
            self._fg = "white"
        if _bg != "black":
            print(f"[live-movie] background {_bg}; overlays drawn in {self._fg}", flush=True)
        self.p.enable_anti_aliasing("msaa", multi_samples=8)

        # 2D IS DETECTED FROM THE WORLD ITSELF, BEFORE PADDING. This used to pad `world` to three
        # entries, replace the zero third span with 1.0 so the camera maths would not divide by it,
        # and then test `span[2] <= 1e-6` to decide whether the run was 2D -- a test that could
        # never fire, because the line above had just overwritten the thing it tested. Every 2D run
        # was therefore drawn as an angled 3D cube with the particles lying on its floor.
        w = [float(x) for x in world]
        self.world = w                 # the per-axis box, kept for the cross-section slab
        # THE SLAB'S HALF-WIDTH IS IN CELLS, so it needs the cell size -- read from the spec's own
        # n_grid rather than assumed, since `thickness: 4` must mean four of the grid's cells and
        # not four of some default's.
        _ng = next((int(fc["n_grid"]) for fc in ((getattr(sim, "fields", None) or {}) or {}).values()
                    if isinstance(fc, dict) and "n_grid" in fc), 96)
        self._cs_dx = (w[1] if len(w) > 1 else w[0]) / float(_ng)
        self.is2d = len(w) < 3
        if self.is2d:
            self.up = 1
        while len(w) < 3:
            w.append(0.0)
        self.lo, self.hi = np.zeros(3), np.array(w)
        span = np.array([x if x > 0 else 1.0 for x in w])
        # WHERE THE CONTENT IS, AND IT IS THE CALLER THAT KNOWS. This renderer frames on [0, world],
        # which is right for a walled run and wrong for a `boundary: free` one: the okuda vesicle is
        # built about the ORIGIN and grows outward, so in a [0, 50] world it renders in a corner and
        # eventually half outside it.
        #
        # KEYED ON A KWARG, NOT ON `sim.boundary`, and that distinction cost a wrong picture.
        # `replay` ALREADY solves this its own way -- it shifts the recorded positions so the cloud
        # sits inside a box built from the data bounds -- so a renderer that ALSO re-centred on the
        # origin whenever the spec said `free` fought its own caller and framed the corner of an
        # already-corrected scene. The live path passes `centred=True`; the replay does not, because
        # by the time it calls, the content is at [0, box] by construction.
        # ...AND `centred` IS A HINT, NOT A FACT. `boundary: free` says nothing about WHERE the
        # content is: a vesicle is built about the origin, an MPM block is seeded inside [0, world],
        # and a spec with both has one of each. Framed on +-world/2 the gel drew outside the box it
        # was supposedly in. The hint decides the DEFAULT; the first drawn frame corrects it from
        # the content's own bounds, which is what `replay` has always done by shifting instead.
        self.free = bool(centred)
        self._reframe = bool(centred)
        if self.free:
            self.lo, self.hi = -np.array(w) / 2.0, np.array(w) / 2.0
        # `plotting.zoom` -- FRAME A SUB-BOX, AND DRAW THAT BOX. `camera_zoom` has been in specs for
        # a long time and this renderer never read it, so asking for a closer view did nothing.
        #
        # ZOOMING IS NOT MOVING THE CAMERA IN. Everything downstream -- the wireframe box, the
        # camera's parallel scale, the scale bar, the cross-section's fixed range -- is derived from
        # `lo`/`hi`, so shrinking THOSE about the scene's centre zooms all of them together and the
        # box still frames the picture instead of falling outside it. `zoom: 2` frames the central
        # half of each axis; the scale bar re-picks its round number for the new extent, so it stays
        # honest rather than becoming a bar that no longer fits.
        self.zoom = float((style or {}).get("zoom", 1.0) or 1.0)
        if self.zoom <= 0:
            raise ValueError(f"plotting.zoom must be > 0, got {self.zoom}")
        if abs(self.zoom - 1.0) > 1e-9:
            _c = 0.5 * (np.asarray(self.lo, float) + np.asarray(self.hi, float))
            _hf = 0.5 * (np.asarray(self.hi, float) - np.asarray(self.lo, float)) / self.zoom
            self.lo, self.hi = _c - _hf, _c + _hf
            print(f"[live-movie] zoom x{self.zoom:g}: framing "
                  f"{np.round(self.lo, 4).tolist()}..{np.round(self.hi, 4).tolist()}", flush=True)

        if self.is2d:
            # A RECTANGLE, NOT A BOX, and seen square-on. A wireframe cube around a plane of
            # particles says the run has a depth it does not have.
            r = pv.Rectangle([[0.0, 0.0, 0.0], [span[0], 0.0, 0.0], [span[0], span[1], 0.0]])
            self.p.add_mesh(r.extract_all_edges(), color="#4a4a4a", line_width=1.0, lighting=False)
            centre = np.array([span[0] / 2, span[1] / 2, 0.0])
            self.p.camera.position = tuple(centre + np.array([0.0, 0.0, 1.0]) * span.max() * 4.0)
            self.p.camera.focal_point = tuple(centre)
            self.p.camera.up = (0.0, 1.0, 0.0)                 # +y is up on screen
            self.p.camera.parallel_projection = True
            self.p.camera.parallel_scale = float(max(span[0], span[1])) * 0.55
        else:
            # AND A ZOOMED VIEW HAS NO BOX. The frame is a sub-box of the world, so its edges are
            # not walls and not the domain -- drawing them puts a rectangle around an arbitrary crop
            # and invites it to be read as the boundary. The SCALE BAR stays, because that is the
            # thing a crop still needs: it re-picks its round number for the framed extent, so it
            # says 0.25 mm, 0.1 mm or 50 um as the view closes in.
            #
            # THE BOX IS A SCENE REFERENCE, NOT A CLAIM ABOUT A WALL. I dropped it for a free
            # boundary on the argument that drawing one asserts a wall the model does not have --
            # but the spec asks for it with `box_frame`, the replay path draws it, and without it a
            # sphere alone on black has no scale, no orientation and no sense of where the camera
            # is. Dropping it also made the two entry points disagree AGAIN, which is the whole
            # thing this renderer was unified to stop. Drawn at `lo..hi`, which is [0, world] for a
            # walled run and centred on the origin for a free one.
            _b = np.asarray(self.lo), np.asarray(self.hi)
            # A ZOOMED VIEW DROPS THE BOX, AND IT HAS TO SAY SO. Drawing it at the shrunken
            # `lo..hi` would label a 27 um crop as the 50 um world, so suppressing it is right --
            # but it was silent, and a spec that asked for `box_frame: true` got no box and no
            # reason. Measured on cil_s19_slowlocal: with `zoom: 1.8` the water filled the frame
            # edge to edge with nothing around it, which reads as the fluid having escaped the
            # domain rather than as the camera being inside it.
            if abs(self.zoom - 1.0) > 1e-9 and bool((self.style or {}).get("box_frame", True)) \
                    and not getattr(self, "_zoom_box_said", False):
                self._zoom_box_said = True
                print(f"[live-movie] `zoom: {self.zoom:g}` frames the central {1 / self.zoom:.0%} "
                      f"of each axis, so the world box is OUTSIDE the view and `box_frame` is not "
                      f"drawn -- the scene will fill the frame with no border. Set `zoom: 1` to "
                      f"get the box back.", flush=True)
            if abs(self.zoom - 1.0) < 1e-9 and bool((self.style or {}).get("box_frame", True)):
                self.p.add_mesh(pv.Box((_b[0][0], _b[1][0], _b[0][1], _b[1][1],
                                        _b[0][2], _b[1][2])).extract_all_edges(),
                                color="#4a4a4a", line_width=1.0, lighting=False)
            # A FLOOR, `plotting.floor: <colour>`: the bottom face of the box as a lit plane, so the
            # bodies have something to land on in the picture as they do in the model.
            _fl = (self.style or {}).get("floor")
            if _fl:
                _lo, _hi = _b[0].astype(float), _b[1].astype(float)
                _c = 0.5 * (_lo + _hi); _c[self.up] = _lo[self.up]
                _ax = [i for i in range(3) if i != self.up]
                _dir = np.zeros(3); _dir[self.up] = 1.0
                self.p.add_mesh(pv.Plane(center=tuple(_c), direction=tuple(_dir),
                                         i_size=float(_hi[_ax[0]] - _lo[_ax[0]]), j_size=float(_hi[_ax[1]] - _lo[_ax[1]])),
                                color=str(_fl), lighting=True, ambient=0.35, diffuse=0.65, specular=0.0,
                                opacity=float((self.style or {}).get("floor_opacity", 1.0)),
                                show_scalar_bar=False, name="floor")
            # A SCALE BAR, AND ONLY WHERE THERE IS A SCALE. Without `general.units` the box is
            # a number of nothing and a bar labelled "20" would be a lie. The length is the largest
            # round number (1, 2 or 5 times a power of ten) fitting in a third of the box, so it
            # sizes itself: 20 m for a 100 m box, 2 cm for a 0.1 m one, with nothing to set.
            #
            # A PLAIN SEGMENT: no end ticks, and the label in the same font and size as the
            # top-left print, so it reads as one annotation rather than two competing ones.
            if self.time_s is not None and getattr(self, "length_um", None) \
                    and bool((self.style or {}).get("scale_bar", True)):   # the bio page draws its own, view-fixed bar
                _m = float(self.length_um) / 1.0e6            # metres per simulation length unit
                _ax0 = [i for i in range(3) if i != self.up][0]
                # THE ROUND NUMBER IS CHOSEN IN THE PHYSICAL UNIT, NOT IN BOX UNITS. It used to
                # be the other way round -- 1/2/5 times a power of ten of `span/3` in SIMULATION
                # units -- and the label then quoted whatever that happened to convert to: a tidy
                # 0.2 of the box printed as "0.234467 mm". A scale bar's whole job is to be a number
                # a reader can carry, so the number is picked first and the bar is drawn to fit it.
                # THE FRAMED EXTENT, so a zoomed view gets a bar that fits it.
                _tgt_m = float((np.asarray(self.hi) - np.asarray(self.lo))[_ax0]) / 3.0 * _m
                _p10 = 10.0 ** np.floor(np.log10(max(_tgt_m, 1e-30)))
                _len_m = max([f * _p10 for f in (1.0, 2.0, 2.5, 5.0) if f * _p10 <= _tgt_m]
                             or [_p10])
                # `plotting.scale_bar_um` -- A DECLARED LENGTH WINS, for a series of slides that
                # must carry the same bar whatever each one's framing picks.
                if (self.style or {}).get("scale_bar_um"):
                    _len_m = float(self.style["scale_bar_um"]) * 1.0e-6
                _len = _len_m / _m                            # back to box units for the geometry
                _other = [i for i in range(3) if i not in (self.up, _ax0)][0]
                _a = np.zeros(3); _b = np.zeros(3)
                # PLACED AGAINST THE SCENE'S OWN CORNER for the same reason the camera is: with a
                # free boundary the box's origin is in the middle of the tissue, and the bar was
                # drawn straight through it.
                # `scale_bar_corner: left | right | right_face` -- WHICH EDGE OF THE BOX, AND WHICH
                # END OF IT. `left` is where the bar has always been, along the first horizontal
                # axis at the near-low corner, and stays the default so no existing run moves.
                # `right` is the far end of that same edge. `right_face` runs the bar along the
                # OTHER horizontal axis on the box's right-hand face, which is what a scene wants
                # when the bottom edge is already busy -- the cross-section panel sits in the
                # bottom-left corner of every apico-basal spec, and a bar on the front-bottom edge
                # reads as part of it.
                #
                # `scale_bar_lift` -- HOW FAR OFF THE FLOOR, as a fraction of the vertical span.
                # The bar sat exactly on `lo[up]`, which is the bottom edge of the wireframe box,
                # so at any camera elevation it renders within a few pixels of the frame's lower
                # border and its label can fall off it. 0 is the old position.
                # THE RIGHT FACE BY DEFAULT. On the bottom-left edge the bar lies along the
                # box's near-bottom rail, where it is foreshortened by the camera and its label
                # competes with the player's own chrome; on the right face it stands clear of both.
                _corner = str((self.style or {}).get("scale_bar_corner", "right_face")).lower()
                if _corner == "right_face":
                    _run, _off = _other, _ax0            # along the other horizontal, on the far face
                    _a[_run] = float(self.hi[_run]) - _len; _b[_run] = float(self.hi[_run])
                    _a[_off] = _b[_off] = float(self.hi[_off]) + 0.04 * float(span[_off])
                else:
                    _run, _off = _ax0, _other
                    if _corner == "right":
                        _b[_run] = float(self.hi[_run]); _a[_run] = float(self.hi[_run]) - _len
                    else:
                        _a[_run] = float(self.lo[_run]); _b[_run] = float(self.lo[_run]) + _len
                    _a[_off] = _b[_off] = float(self.lo[_off]) - 0.04 * float(span[_off])
                _lift = float((self.style or {}).get("scale_bar_lift", 0.0)) * float(span[self.up])
                _a[self.up] = _b[self.up] = float(self.lo[self.up]) + _lift
                self.p.add_mesh(pv.Line(_a, _b), color=self._fg, line_width=4.0, lighting=False)
                _lab = _si_length(_len_m)
                # THE LABEL IS 3D TEXT LYING ALONG THE BAR, not a screen-aligned point label.
                #
                # `add_point_labels` billboards: the glyphs always face the camera, so a bar that
                # the projection tilts -- and it tilts in every one of these scenes -- carries a
                # horizontal caption at an angle to it, sitting further away than it looks because
                # the gap is measured in world units along a foreshortened axis. VTK cannot rotate
                # a point label; the only way to align text with a line in the scene is to put the
                # text IN the scene.
                #
                # Built in its own plane and mapped onto the frame (bar direction, in-plane up,
                # normal), so it reads left-to-right along the bar and upright, and is
                # foreshortened by exactly as much as the bar is. Height is a fraction of the
                # bar's LENGTH, so the two stay in proportion at any zoom.
                _d = _b - _a
                _L = float(np.linalg.norm(_d))
                _d = _d / max(_L, 1e-12)
                _camup = np.zeros(3); _camup[self.up] = 1.0
                # FROM ABOVE, THE LABEL LIES FLAT. Standing in the vertical plane it is seen edge-on
                # by a top-view camera and vanishes; lying in the ground plane, reading up-screen,
                # it faces that camera.
                if abs(float(elev)) > 80.0:
                    _camup = np.zeros(3); _camup[[i for i in range(3) if i != self.up][1]] = 1.0
                # THE IN-PLANE UP IS WORLD UP, so the glyphs stand upright rather than following
                # whatever handedness a cross product happened to give.
                _upv = _camup - _d * float(np.dot(_camup, _d))
                if np.linalg.norm(_upv) < 1e-6:            # bar parallel to up: any up will do
                    _upv = np.cross(_d, np.array([1.0, 0.0, 0.0]))
                _upv = _upv / np.linalg.norm(_upv)
                _nrm = np.cross(_d, _upv)
                # AND THE TEXT MUST FACE THE CAMERA. Built from an arbitrary normal it is a 50%
                # chance of being seen from BEHIND -- which renders as a mirror image, legible
                # enough to look like a font bug and not like a facing one. The camera direction is
                # not set yet at this point in the frame, so it is derived from the same elev/azim
                # the camera block below uses; reversing the READING direction (and with it the
                # normal) keeps the glyphs upright, where flipping the normal alone would invert
                # them.
                _e, _az = np.radians(elev), np.radians(azim)
                _axh = [i for i in range(3) if i != self.up]
                _view = np.zeros(3)
                _view[_axh[0]] = np.cos(_e) * np.cos(_az)
                _view[_axh[1]] = np.cos(_e) * np.sin(_az)
                _view[self.up] = np.sin(_e)
                if float(np.dot(_nrm, _view)) < 0.0:
                    _d = -_d
                    _nrm = np.cross(_d, _upv)
                # WITH A FLOOR SET BY THE FRAME, NOT THE BAR. Tied to the bar alone, a short bar got
                # an unreadable label: `scale_bar_um: 20` on a 150 um view printed "20 um" in
                # glyphs 10 px tall on a 1280 px frame, half the height of the header's. The floor,
                # `scale_bar_text_min` of the framed width (default 0.045, ~20 px glyphs), keeps
                # the label at header size whatever length the bar is.
                _ext = float((np.asarray(self.hi) - np.asarray(self.lo))[_ax0])
                _h = max(float((self.style or {}).get("scale_bar_text", 0.17)) * _L,
                         float((self.style or {}).get("scale_bar_text_min", 0.045)) * _ext)
                # THE LABEL IS A TEXTURED QUAD, NOT VECTOR TEXT.
                #
                # `pv.Text3D` wraps vtkVectorText, which is ASCII ONLY: it rendered "50 um" as
                # "50 m", silently dropping the micro sign this overlay was changed to use in the
                # first place. A wrong unit that looks like a font quirk is worse than no label.
                # Rasterising the string with matplotlib keeps every glyph -- micro signs,
                # superscripts, anything the header can print -- and mapping it onto a quad in the
                # bar's own frame keeps the alignment that vector text was chosen for.
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as _plt
                    _fig = _plt.figure(figsize=(6, 1.4), dpi=200)
                    _fig.patch.set_alpha(0.0)
                    _tx = _fig.text(0.5, 0.5, _lab, ha="center", va="center",
                                    color=self._fg, fontsize=44)
                    _fig.canvas.draw()
                    _bb = _tx.get_window_extent(_fig.canvas.get_renderer())
                    _img = np.asarray(_fig.canvas.buffer_rgba())
                    _H = _img.shape[0]
                    _pad = 6
                    _crop = _img[max(0, int(_H - _bb.y1) - _pad): int(_H - _bb.y0) + _pad,
                                 max(0, int(_bb.x0) - _pad): int(_bb.x1) + _pad].copy()
                    _plt.close(_fig)
                    _asp = _crop.shape[1] / max(_crop.shape[0], 1)
                    _w = _h * _asp
                    # CLOSER THAN THE OLD 9% OF THE SPAN: two thirds of the label's own height
                    # below the bar, so the gap scales with the label and not with the scene.
                    _c = 0.5 * (_a + _b) - _upv * (0.66 * _h + 0.5 * _h)
                    _hw, _hh = 0.5 * _w, 0.5 * _h
                    _q = pv.PolyData(
                        np.array([_c - _d * _hw - _upv * _hh, _c + _d * _hw - _upv * _hh,
                                  _c + _d * _hw + _upv * _hh, _c - _d * _hw + _upv * _hh]),
                        faces=np.array([4, 0, 1, 2, 3]))
                    _q.active_texture_coordinates = np.array(
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)
                    self.p.add_mesh(_q, texture=pv.Texture(_crop), lighting=False,
                                    show_scalar_bar=False)
                except Exception as e:                     # noqa: BLE001 -- never lose the movie
                    print(f"[live-movie] 3D scale label unavailable ({type(e).__name__}: {e}); "
                          f"using a flat one", flush=True)
                    _mid = 0.5 * (_a + _b); _mid[self.up] -= 0.09 * float(span[self.up])
                    self.p.add_point_labels([_mid], [_lab], font_size=26, text_color=self._fg,
                                            bold=False, shape=None, show_points=False,
                                            always_visible=True,
                                            justification_horizontal="center",
                                            justification_vertical="top")
            # AIMED AT THE SCENE, NOT AT [0, world]. `0.5 * span` is the middle of the world box,
            # which is where the content is only when a wall puts it there. With `boundary: free`
            # nothing does: the okuda vesicle is built about the ORIGIN, so the camera looked at
            # [25, 25, 25] while the tissue sat at [0, 0, 0] and the frame was empty but for a
            # corner of it, seen from inside. `lo`/`hi` already carry the framing decision made
            # above -- [0, world] for a walled run, +-world/2 for a free one -- so the camera reads
            # them instead of re-deriving a box.
            centre = 0.5 * (np.asarray(self.lo) + np.asarray(self.hi))
            radius = float(np.max(np.asarray(self.hi) - np.asarray(self.lo))) * 0.55
            e, az = np.radians(elev), np.radians(azim)
            ax_h = [i for i in range(3) if i != self.up]
            d = np.zeros(3)
            d[ax_h[0]], d[ax_h[1]], d[self.up] = (np.cos(e) * np.cos(az), np.cos(e) * np.sin(az),
                                                  np.sin(e))
            self.p.camera.position = tuple(centre + d * radius * 6.0)
            self.p.camera.focal_point = tuple(centre)
            # KEPT, so `cutaway: {axis: view}` can cut along the direction the camera looks. A cut
            # plane that is not perpendicular to the view is seen as a bowl at an angle, and the
            # shell then reads as a crescent -- thick where it is seen edge-on through its whole
            # depth, and vanishing on the opposite side where the cut circle meets the silhouette.
            # That asymmetry is entirely projective and looks exactly like a seeding defect.
            self._view_dir = d.copy()
            u = np.zeros(3); u[self.up] = 1.0
            # LOOKING DOWN THE UP AXIS, SCREEN-UP IS THE SECOND HORIZONTAL AXIS. With the view
            # parallel to `up` the camera's up vector is degenerate: VTK picked its own roll, and the
            # curve-column shift below, cross(-d, u), came out zero -- a top-view sheet sat off
            # centre with the curves drawn over its side of the frame.
            if abs(float(np.dot(d, u))) > 0.99:
                u = np.zeros(3); u[[i for i in range(3) if i != self.up][1]] = 1.0
            self.p.camera.up = tuple(u)
            # `plotting.camera_roll`, IN DEGREES: WHICH END OF THE SUBJECT IS UP.
            #
            # The up vector is +`up_axis` and there is no way to ask for the other end, which
            # matters for any dataset whose long axis is anatomical rather than vertical: the
            # Platynereis region runs head (low z) to tail (high z), so a z-up movie shows the
            # larva upside down against every figure in its own paper. The GUI's own camera grew
            # the same knob for the same reason, and without it here the stills and the movie of
            # ONE run disagreed -- the local VLM watched the rolled stills and the unrolled movie
            # and reported the animal "slowly migrating upward" as it fell.
            #
            # A roll and not a flipped `up_axis`: the up axis also decides where the floor is
            # drawn and which way the scale bar is lifted.
            _roll = float((self.style or {}).get("camera_roll", 0.0) or 0.0)
            if _roll:
                self.p.camera.roll = self.p.camera.roll + _roll
            self.p.camera.parallel_projection = True
            self.p.camera.parallel_scale = radius * 1.45
            # AND SHIFTED OUT FROM UNDER THE PANELS. Widening the frame alone does not clear the
            # scene: parallel projection fits the VERTICAL extent, so the extra width is slack on
            # BOTH sides and the box still reaches into the right-hand column. Translating the
            # camera along its own horizontal screen axis moves the scene left by exactly the
            # column's width -- `parallel_scale` is the half-height in world units, so the window is
            # 2*scale*aspect wide and a column of `f` of it is f*2*scale*aspect.
            if self._curve_col > 0:
                # `d` POINTS FROM THE SCENE TO THE CAMERA, so the view direction is -d and
                # screen-right is cross(-d, u), not cross(d, u). With the sign the other way
                # the camera moved left and the scene slid RIGHT, straight under the panels
                # the shift exists to clear.
                _h = np.cross(-d, u)
                _n = float(np.linalg.norm(_h))
                if _n > 1e-12:
                    _sh = (_h / _n) * (0.5 * self._curve_col * 2.0
                                       * self.p.camera.parallel_scale * self.aspect)
                    self.p.camera.position = tuple(np.asarray(self.p.camera.position) + _sh)
                    self.p.camera.focal_point = tuple(np.asarray(self.p.camera.focal_point) + _sh)

        self._draw_obstacles(span)
        # A FRAGMENTED MP4, SO THE FILE IS READABLE WHILE IT IS STILL BEING WRITTEN.
        # A plain mp4 keeps its index -- the moov atom -- at the END, so nothing before close()
        # plays and copying the file mid-run copies an unreadable prefix. That is why killing a run
        # used to lose the movie, and why SIGINT (which lets close() run) saved a 10-hour job's
        # 98.9 MB where SIGKILL would have left 52 MB of rubble.
        #
        # `frag_keyframe+empty_moov` writes a self-contained fragment per keyframe and `-g 1` makes
        # every frame a keyframe, but neither is enough on its own: MEASURED, both leave the file at
        # 36 bytes after 40 frames because ffmpeg buffers. `-flush_packets 1` is the one that
        # matters -- with it the same file reads 30 frames at frame 40, and is still a valid
        # complete movie after close.
        #
        # It costs a little size (no global index, a fragment header per frame) and it means a
        # long run can be WATCHED from the file at any moment, with no duplicate and no second
        # writer, which is what `3d.png` was standing in for.
        # `only` DRAWS THE SLICE IN THE 3D VIEW ITSELF -- no chart at all. Keeping the ordinary
        # renderer means the slice arrives with the box, the obstacles, the camera, the colours and
        # the scale bar already correct, and the sphere the jet is hitting is simply THERE, drawn as
        # the 3D actor it is. A 2D chart had to reproduce every one of those and reproduced none:
        # it showed no obstacle, needed its own axes, disabled point sprites by existing, and froze
        # unless its series was rebuilt each frame.
        if self.cs is None and getattr(self, "cs_cfg", None) is not None and not self.cs_only:
            ax = self.cs_axis
            lat = [k for k in range(3) if k != ax][:2]
            # SQUARE ON SCREEN, WHICH IS A CONSTRAINT ON THE SIZE AND NOT ON THE RANGE. `size` is in
            # WINDOW fractions and the window need not be square (a curve column widens it), so
            # (0.26, 0.26) is 0.26*W by 0.26*H -- and equal DATA ranges then still render stretched:
            # a spherical shell sliced through its middle came out as a tall ellipse beside a 3D
            # view showing a sphere. A panel `h` of the height must be h/aspect of the width.
            _csh = float((style or {}).get("cross_section_height", 0.26))
            # `cross_section.loc` -- WHERE THE PANEL SITS, in window fractions from the BOTTOM left,
            # because the default corner is not always free. The run's own title block is drawn in
            # the top left and a tall panel meets it; a taller `cross_section_height` makes that
            # worse, since the panel grows upward from `loc`. Declared rather than nudged in code so
            # a spec that needs the panel elsewhere says so, and every spec that does not is
            # unchanged at the (0.015, 0.645) this has always used.
            _csl = (style or {}).get("cross_section", {}).get("loc") or (0.015, 0.645)
            ch = pv.Chart2D(size=(_csh / self.aspect, _csh), loc=(float(_csl[0]), float(_csl[1])))
            ch.background_color = (0, 0, 0, 0.55)
            ch.border_color = "#9a9a9a"
            names = "xyz"
            ch.title = (f"{names[lat[0]]}{names[lat[1]]} slice at "
                        f"{names[ax]} = {self.cs_at:.2f} of the box")
            # FIXED RANGES, NOT AUTOSCALED. An autoscaling axis rescales to whatever is in the slab,
            # so a jet that thins to a thread would fill the panel exactly as a full one does and the
            # thing the panel exists to show would be the one thing it hides.
            ch.x_axis.range = [0.0, float(self.world[lat[0]])]
            ch.y_axis.range = [0.0, float(self.world[lat[1]])]
            # LABELS AND TICKS ON, EXPLICITLY. The first build drew a bare rectangle: `.label` is
            # set here but a chart at this size hides its decorations unless asked, so the panel had
            # no axes, no ticks and no title -- an empty box that could equally have meant "no data"
            # or "not working".
            ch.x_axis.label = f"{names[lat[0]]} (m)"
            ch.y_axis.label = f"{names[lat[1]]} (m)"
            for _a in (ch.x_axis, ch.y_axis):
                _a.label_visible = True
                _a.ticks_visible = True
                _a.tick_labels_visible = True
                _a.grid = False
            ch.legend_visible = False
            _c = list((style or {}).get("colors", {}).values())
            self._cs_size = 6 if self.cs_only else 3
            self._cs_colour = tuple(_c[0]) if _c else (0.3, 0.62, 1.0)
            self._cs_series = ch.scatter([0.0], [0.0], size=self._cs_size, style="o",
                                         color=self._cs_colour)
            self.p.add_chart(ch)
            self.cs = ch
            self._cs_lat = lat
        # `-g` IS THE KEYFRAME INTERVAL AND IT WAS 1, which means EVERY frame was an I-frame and
        # the movie carried no inter-frame compression at all. That is why the record's files run
        # to 229 MB for a few hundred frames of a scene that barely changes: an all-intra encode
        # pays full price for every frame whether anything moved or not. The flag was there for a
        # real reason -- `frag_keyframe` starts a new fragment at each keyframe, and with
        # `empty_moov` that makes the file playable WHILE it is still being written, which is what
        # lets the watcher show a run in progress. But a fragment per frame is a very expensive way
        # to buy that: every 15 frames is under a second of video at these framerates, so the live
        # view is no less live and the file is a fraction of the size.
        _g = int((style or {}).get("movie_keyframe_every", 15) or 15)
        self.p.open_movie(out, framerate=max(1, int(round(getattr(self, "fps", fps)))), quality=8,
                          output_params=["-movflags", "frag_keyframe+empty_moov+default_base_moof",
                                         "-g", str(max(1, _g)), "-flush_packets", "1"])

    def _draw_obstacles(self, span):
        """The world's solid geometry, which the simulation sees and this renderer did not.

        `general.obstacles` is what `mpm_grid_update` rasterises into its no-slip mask, so a movie
        that omits it shows fluid parting around nothing. `plot.py` has drawn them since the
        beginning (`_draw_obstacles`); this renderer simply never did, so every obstacle spec --
        genA/genB/genC and the four material_3d_obstacle_* -- rendered a hole in the flow with no
        cause visible.

        The length disambiguates, exactly as plot.py reads it: 3 = 2D disc [cx,cy,r], 4 = 2D
        rectangle [x0,y0,x1,y1] in a planar run and a 3D SPHERE [cx,cy,cz,r] otherwise, 6 = 3D box.
        """
        obs = list(getattr(self.sim, "obstacles", []) or []) if self.sim is not None else []
        pv = self.pv
        for r in obs:
            v = [float(x) for x in r]
            try:
                if self.is2d and len(v) == 3:
                    m = pv.Polygon(center=(v[0], v[1], 0.0), radius=v[2], n_sides=64)
                elif self.is2d and len(v) == 4:
                    m = pv.Rectangle([[v[0], v[1], 0.0], [v[2], v[1], 0.0], [v[2], v[3], 0.0]])
                elif len(v) == 4:
                    m = pv.Sphere(radius=v[3], center=(v[0], v[1], v[2]),
                                  theta_resolution=48, phi_resolution=48)
                elif len(v) == 6:
                    m = pv.Box((v[0], v[3], v[1], v[4], v[2], v[5]))
                else:
                    continue
            except Exception as e:
                print(f"[live-movie] obstacle {v} not drawn: {type(e).__name__}: {e}", flush=True)
                continue
            # OPAQUE AND LIT, unlike the particles. The dots are flat and unshaded so density reads
            # as brightness; an obstacle drawn the same way would be a featureless silhouette, and
            # in 3D you could not tell a sphere from a disc.
            #
            # MATTE, NOT GLOSSY, AND FLAT-SHADED. `specular=0.2` with `smooth_shading=True` put a
            # moving highlight on every box and rounded the edges of shapes that are exactly
            # axis-aligned boxes -- the stair in si_avalanche read as polished metal and its steps
            # had soft corners they do not have. `specular=0` removes the sheen; flat shading gives
            # each face one constant tone, so a box looks like a box and the geometry is legible
            # from its face brightnesses alone. Depth then comes from the shadows below, not from
            # a highlight sliding across the surface.
            self.p.add_mesh(m, color="#9a9a9a", opacity=1.0, lighting=not self.is2d,
                            specular=0.0, specular_power=1.0, ambient=0.28, diffuse=0.85,
                            smooth_shading=False)
        self.n_obstacles = len(obs)
        # REAL SHADOWS, once, and only when there is something to cast them. VTK's shadow pass
        # costs a second render of the scene per light, which is why it is not on by default; with
        # obstacles present it is what separates a body resting ON a step from one floating above
        # it, and that ambiguity is exactly what a flat-shaded scene cannot resolve on its own.
        # SHADOWS ARE OPT-IN, `plotting.shadows: true`, and default OFF. They make obstacle
        # geometry legible -- which is why they were added -- but VTK's shadow pass RE-LIGHTS every
        # actor, including the particle cloud that explicitly asked for `lighting=False`. Measured on
        # si_jet_sphere_wide: a cloud whose colour array is uniformly [76, 158, 255] renders at
        # [134, 135, 137] with shadows on and [98, 104, 112] with them off. The blue water came out
        # grey, and nothing about the colour pipeline was wrong -- the lighting was.
        if obs and not self.is2d and bool(self.style.get("shadows", False)):
            try:
                self.p.enable_shadows(); self._shadows_on = True
            except Exception as e:                      # not fatal: the movie is still readable
                print(f"[live-movie] shadows unavailable ({type(e).__name__}: {e}); "
                      f"obstacles are flat-shaded without them", flush=True)

    def _xyz(self, lvl):
        # float32, NOT float64. VTK stores points in whatever dtype it is handed; float64 doubles
        # both the host copy and VTK's resident buffer (10 M points: 240 MB against 120 MB) to carry
        # digits that never survive the projection to a 1280 px frame.
        _p = lvl.get("pos")[self.idx].detach()
        # DORMANT SLOTS ARE NOT DRAWN. A set with a reserve parks its unused slots far off-domain
        # (-1e6 for integrins); a cloud that carried them stretched the camera's depth range by a
        # million and VTK culled the whole scene -- blank frames exactly while the reserve was not
        # yet used up, a picture that appeared at frame 350 when the buffer filled. A dormant slot
        # is drawn ON TOP of the first live one: same point, same colour, invisible.
        _o = getattr(lvl, "occ", None)
        if _o is not None:
            _ol = torch.as_tensor(_o)[self.idx].to(_p.device) > 0
            if bool(_ol.any()) and not bool(_ol.all()):
                _p = torch.where(_ol[:, None], _p, _p[_ol][:1].expand_as(_p))
        if getattr(self, "cs_only", False):
            # OUTSIDE THE SLAB IS PARKED, NOT REMOVED. The drawn cloud has a fixed length -- its
            # colours were bound to it at t=0 and `cloud.points = ...` replaces an array of the same
            # size -- so a particle is hidden by being put where the camera is not, exactly as the
            # dormant pool is. Filtering the array instead would change its length every frame and
            # detach it from its colours.
            _ax, _w = self.cs_axis, float(self.world[self.cs_axis])
            _sel = (_p[:, _ax] - self.cs_at * _w).abs() < self.cs_cells * self._cs_dx
            if not getattr(self, "_cs_said", False):
                self._cs_said = True
                print(f"[live-movie] cross section: slab {2 * self.cs_cells * self._cs_dx * 1000:.2f}"
                      f" mm thick ({2 * self.cs_cells:.2f} cells) at "
                      f"{'xyz'[_ax]} = {self.cs_at * _w * 1000:.1f} mm, "
                      f"{int(_sel.sum()):,} of {_sel.numel():,} drawn", flush=True)
            # PARKED JUST OUTSIDE, NOT FAR OUTSIDE. VTK sizes `render_points_as_spheres` sprites
            # from the ACTOR'S BOUNDS, so hiding particles at -9 m beside a 0.1 m box made the
            # bounding box 90x the domain and shrank every drawn dot to nothing: 17,338 lit pixels,
            # none of them coloured. Two centimetres outside the wall is just as invisible -- the
            # gather clamps the domain at 2*dx -- and leaves the bounds essentially the box's own.
            _p = torch.where(_sel[:, None], _p, torch.full_like(_p, -0.02 * _w))
        pos = _p.cpu().numpy().astype(np.float32)
        if pos.shape[1] == 2:                         # pad a 2D run into the z=0 plane
            pos = np.concatenate([pos, np.zeros((pos.shape[0], 1))], 1)
        # `near_side` ON THE DRAWN CLOUD. Its colours are bound to a fixed-length array, so a point
        # is hidden the way the cross-section hides one: put where the camera is not, not removed.
        if (self.style or {}).get("near_side") is not None and (self.style or {}).get("near_side") is not False \
                and len(pos) and getattr(self, "p", None) is not None:
            f = (self.style or {}).get("near_side")
            c = np.asarray(self.p.camera.position, float) - np.asarray(self.p.camera.focal_point, float)
            n = c / max(float(np.linalg.norm(c)), 1e-12)
            ctr = pos.mean(0)
            d = (pos - ctr) @ n
            if isinstance(f, str) and f.lower() == "far":             # the cut-away: hide the near half
                far = d > 0.0
            else:
                cut = 0.0 if f is True else float(f) * float(np.abs(d).max() or 1.0)
                far = d < cut
            if far.any() and not far.all():
                pos = np.where(far[:, None], pos[~far][:1], pos)
            self._xyz_cut = far
        return pos

    def __call__(self, H, tick):
        if self.failed:
            return
        try:
            self._frame(H, tick)
        except Exception as e:                        # never take the run down for a picture
            self.failed = f"{type(e).__name__}: {e}"
            print(f"[live-movie] DISABLED after frame {tick}: {self.failed}", flush=True)

    def _body_shade(self, lvl, pts, base):
        """`dot_shading: body` -- light each dot by its body's outward normal, per frame.

        WHY SPRITE LIGHTING WAS NOT ENOUGH. `dot_shading: true` lights each sprite as a sphere, and
        at one or two pixels a body of 40,000 of them has no macroscopic normal: every sprite shows
        the same lit hemisphere and across the body they average to a flat disc. Measured on the
        cows at 1.2 px and again at 2.5 px -- no lit side, no dark side, silhouettes.

        THE CHEAPEST NORMAL A BODY HAS is the direction from its own centroid to the dot,
        which is exact for a sphere and a fair reading for anything convex-ish. The shade is
        Lambert on that normal against one fixed light, floored at `dot_ambient`, and it is
        recomputed every frame from the current positions, so a body that tumbles turns its lit
        side. No surface is reconstructed and the dots stay the reference picture; this only
        modulates their brightness. Bodies come from `lvl.parent`, the per-child parent index the
        hierarchy already carries.
        """
        par = getattr(lvl, "parent", None)
        if par is None or base is None:
            return base
        par = np.asarray(par.detach().cpu().numpy() if hasattr(par, "detach") else par).astype(int)
        n = min(len(par), len(pts), len(base))
        par, P, B = par[:n], np.asarray(pts[:n], np.float64), np.asarray(base[:n], np.float64)
        k = int(par.max()) + 1 if n else 0
        if k == 0:
            return base
        cnt = np.bincount(par, minlength=k).astype(np.float64)
        cen = np.zeros((k, 3))
        for d in range(3):
            cen[:, d] = np.bincount(par, weights=P[:, d], minlength=k) / np.maximum(cnt, 1.0)
        nrm = P - cen[par]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        st = self.style or {}
        L = np.asarray(st.get("dot_light", [-0.45, 0.8, 0.4]), np.float64)
        L /= max(np.linalg.norm(L), 1e-12)
        amb = float(st.get("dot_ambient", 0.35))
        shade = amb + (1.0 - amb) * np.clip(nrm @ L, 0.0, 1.0)
        # SPECULAR, ADDED TO THE HUE RATHER THAN SCALING IT: a highlight is the light's colour, not
        # the body's, so it is white pushed on top. Blinn-Phong on the same body normal, with the
        # half-vector between the light and a fixed view direction -- the camera does not move in
        # these clips, so a constant `dot_view` is the honest approximation and a per-frame camera
        # query would buy nothing. `dot_specular` is the strength and `dot_specular_power` the
        # tightness, the same two keys sprite lighting reads.
        spec = float(st.get("dot_specular", 0.0))
        if spec > 0.0:
            V = np.asarray(st.get("dot_view", [0.0, 0.25, 1.0]), np.float64)
            Hv = L + V / max(np.linalg.norm(V), 1e-12)
            Hv /= max(np.linalg.norm(Hv), 1e-12)
            hl = spec * np.clip(nrm @ Hv, 0.0, 1.0) ** float(st.get("dot_specular_power", 24))
        else:
            hl = np.zeros(n)
        out = np.asarray(base).copy()
        out[:n] = np.clip(B * shade[:, None] + 255.0 * hl[:, None], 0, 255).astype(out.dtype)
        return out

    def _frame(self, H, tick):
        import torch
        self._H_now = H                        # the hierarchy the builders may read (block colours)
        self._cb_tick = int(tick)
        # `plotting.subject: <set>` NAMES THE SET THE MOVIE IS ABOUT; otherwise the largest set
        # carrying positions is it. A metabolic network has more reactions than metabolites and
        # both sit somewhere, and the concentrations are the picture.
        sname = (self.style or {}).get("subject") or _biggest_particle_set(H)
        if sname is not None and sname not in H.levels:
            self.failed = f"plotting.subject: no set {sname!r}"
            return
        if sname is None:
            self.failed = "no set carries positions"
            return
        self._sname = sname                           # `_rgb` needs it to look up the type palette
        lvl = H.level(sname)
        # THE LIVE COUNT, RE-READ EVERY FRAME. `lvl.n` is the RESERVOIR -- the table a growing set
        # was allocated into -- and it never changes, so a tissue that divided from 396 vertices to
        # 4,440 was labelled "25,584 nodes" from the first frame to the last: a true statement about
        # the allocation and a wrong one about the tissue. `occ` is the present.
        _occ = getattr(lvl, "occ", None)
        self.n_live = int(_occ.sum()) if _occ is not None else int(lvl.n)
        if self.cloud is None:
            self.n = int(lvl.n)
            # SEEDED, AND DRAWN ONCE. A subset re-drawn each frame makes the fluid boil: every dot
            # would be a different particle, so nothing would appear to move. Fixing the subset is
            # what makes this a movie of the material rather than of noise.
            k = min(self.render_n, self.n)
            if k >= self.n:
                # DRAWING EVERYTHING: index with a CONTIGUOUS range, not a permutation. A full
                # randperm is a no-op as a sample -- it selects every particle either way -- but it
                # makes each frame's gather a random scatter across the whole set instead of a
                # coalesced sequential read, and it costs 8 B per particle to store the permutation
                # (0.8 GB at 100 M). All cost, no effect.
                self.idx = torch.arange(self.n, device=lvl.state.device)
            else:
                g = torch.Generator(device="cpu").manual_seed(self.seed)
                self.idx = torch.randperm(self.n, generator=g)[:k].to(lvl.state.device)
            # `plotting.cutaway` -- OPEN THE BODY AND KEEP THE COLOURS. A dense 3D cloud is an
            # opaque silhouette: 813,000 points arranged as a cell show a beige sphere, and every
            # organelle inside it is hidden behind the membrane that encloses them. The existing
            # `cross_section` answers this with a 2D chart overlay, which is the right picture for a
            # jet's footprint and the wrong one here -- it is flat, it is capped at a few thousand
            # points, and it carries `color_field` bands rather than the per-type palette that says
            # which organelle a dot belongs to.
            #
            # This instead DISCARDS one half-space, so the remaining points are drawn by the
            # ordinary path: same renderer, same dots, same colours, and the cell is simply cut
            # open. It is the CUT AWAY of the atlas viewer the model is built from.
            #
            # SELECTED ONCE, FROM THE FIRST FRAME, and this is the same argument as the fixed
            # subset above. Re-testing the plane every frame would let a point cross it and vanish
            # or appear, so the cut surface would boil while the body deformed and a bounce would
            # read as material being created. Cutting the material once and then following THAT
            # material is what makes the section a section rather than a stencil.
            _cut = (self.style or {}).get("cutaway")
            if _cut:
                _cut = {} if _cut is True else dict(_cut)
                _axn = str(_cut.get("axis", "view")).lower()
                _P = lvl.get("pos").detach()[self.idx]
                if _axn == "view":
                    # THE DEFAULT, AND THE ONLY ONE THAT LOOKS RIGHT WITHOUT TUNING. Cutting along
                    # the VIEW direction puts the camera on the plane's normal, so the exposed face
                    # is seen square-on and the remaining shell is an even ring instead of a
                    # crescent. A named axis is still allowed for a fixed anatomical section, but
                    # then the camera has to be aimed at it by hand.
                    _n = torch.as_tensor(getattr(self, "_view_dir", np.array([0.0, 0.0, 1.0])),
                                         dtype=_P.dtype, device=_P.device)
                    _n = _n / _n.norm().clamp_min(1e-12)
                    _c = _P @ _n
                    # `at` is a fraction of the DRAWN BODY's extent along the normal, not of the
                    # world box: the plane has to pass through the cell wherever the cell is, and
                    # a world fraction would miss it the moment the body moved or was not centred.
                    _lo, _hi = float(_c.min()), float(_c.max())
                    _at = _lo + float(_cut.get("at", 0.5)) * (_hi - _lo)
                    # DISCARD THE HALF NEAREST THE CAMERA (`_view_dir` points scene -> camera), so
                    # the cut face is what you look into.
                    _m = _c <= _at
                    _keep = "far side of the view plane"
                else:
                    _ax = _CS_AXIS[_axn]
                    _at = float(_cut.get("at", 0.5)) * float(self.world[_ax])
                    # `thickness` TURNS THE HALF-SPACE INTO A SLAB, and for a dense cloud that is
                    # the difference between a section and a fog. Keeping a half-space still leaves
                    # every particle BEHIND the cut plane in the picture, so a 155,000-particle box
                    # draws 78,000 of them stacked along the line of sight and the near field the
                    # section was opened to expose is buried under all the quiet fluid behind it.
                    # A slab of `thickness` (a FRACTION of the world box along this axis, like
                    # `at`) keeps only what lies within half that of the plane, so the picture is a
                    # true 2D section of the flow. Absent, the behaviour is the old half-space.
                    _th = _cut.get("thickness")
                    _c = _P[:, _ax]
                    if _th is not None:
                        _h = 0.5 * float(_th) * float(self.world[_ax])
                        _m = (_c >= _at - _h) & (_c <= _at + _h)
                        _keep = (f"slab of axis {_ax} within {_h:.3f} of the plane "
                                 f"(thickness {float(_th):.3g} of the box)")
                    else:
                        _below = str(_cut.get("keep", "below")).lower() in ("below", "low", "minus")
                        _m = (_c <= _at) if _below else (_c >= _at)
                        _keep = f"{'lower' if _below else 'upper'} half-space of axis {_ax}"
                if bool(_m.any()):
                    self.idx = self.idx[_m]
                    print(f"[live-movie] cutaway: keeping the {_keep} at {_at:.3f} -- "
                          f"{int(self.idx.numel()):,} of {k:,} drawn", flush=True)
                else:
                    print(f"[live-movie] cutaway at {_at:.3f} keeps NO particles; drawing the "
                          f"whole set instead", flush=True)
                k = int(self.idx.numel())
            self.drawn = k
            pos = self._xyz(lvl)
            self.cloud = self.pv.PolyData(pos)
            # RESOLVED BEFORE IT IS ASSIGNED. Writing `cloud["rgb"] = None` and testing afterwards
            # cannot work: pyvista raises "Empty array unable to be added" ON THE ASSIGNMENT, so the
            # fallback line was unreachable and every spec WITHOUT `color_field` lost its movie --
            # silently, because LiveMovie swallows its own errors to avoid killing a long run. Three
            # A100 jobs rendered nothing before this was noticed.
            # THE CHEMISTRY WINS OVER THE HEIGHT RAMP, AND `color_field` WINS OVER BOTH -- an
            # explicit request beats a default, a default beats a fallback. Without the middle term
            # a reaction-diffusion run rendered as its own GEOMETRY: the minisite's Turing disc came
            # out pixel-for-pixel identical to the same 4,000 cells carrying no chemistry at all
            # (exp_03 steps 0008, 0009), on a run whose entire subject is the pattern.
            _rgbv = self._rgb_field(H, lvl)
            if _rgbv is None:
                _rgbv = self._rgb_chem(H, lvl)
            if _rgbv is None and str((self.style or {}).get("color_by", "") or "").lower() == "chem":
                _rgbv = self._rgb_parent_chem(H, lvl)     # opt-in: see `_rgb_parent_chem`
            self.cloud["rgb"] = _rgbv if _rgbv is not None else self._rgb(H, lvl, pos)
            self._base_rgb = None
            if str((self.style or {}).get("dot_shading", "")).lower() == "body":
                self._base_rgb = np.asarray(self.cloud["rgb"]).copy()
                self.cloud["rgb"] = self._body_shade(lvl, np.asarray(self.cloud.points),
                                                     self._base_rgb)
            # POINT SPRITES AND THE CHART OVERLAY DO NOT COEXIST. `add_chart` inserts a
            # vtkContextActor, and with one present `render_points_as_spheres=True` draws nothing at
            # all: measured on this scene, 3,701 blue pixels with the panel and 61,331 without, from
            # an identical simulation whose colour array was uniformly [76, 158, 255] either way.
            # Plain GL points render correctly alongside the chart and are visually identical at the
            # 1-2 px these dots are drawn at, so the panel costs the sprite, not the picture.
            _flat = dict(FLAT)
            # `dot_shading: true` -- LIT SPHERE SPRITES INSTEAD OF FLAT DISCS. FLAT switches
            # lighting off so a dot is one colour edge to edge, which is right for a cloud read as
            # density and wrong for a cloud read as a BODY: thirty bouncing cows drawn flat are
            # thirty silhouettes. With lighting on, `render_points_as_spheres` shades each sprite
            # as the sphere it claims to be, so a body gets a lit side and a dark side from the
            # scene's lights without any surface reconstruction -- which is the point: the dots are
            # the reference picture and this only lights them.
            if (self.style or {}).get("dot_shading", False) is True:
                _st = self.style or {}
                _flat.update(lighting=True,
                             ambient=float(_st.get("dot_ambient", 0.3)),
                             diffuse=float(_st.get("dot_diffuse", 0.7)),
                             specular=float(_st.get("dot_specular", 0.3)),
                             specular_power=float(_st.get("dot_specular_power", 20)))
            if self.cs is not None:
                _flat["render_points_as_spheres"] = False
            # WHEN THE SUBJECT IS A MESH, THE DOTS ARE ITS OWN VERTICES and drawing them is drawing
            # the corners of the thing rather than the thing. One renderer covers both cases: a
            # material run draws its cloud (or a skinned surface of it) with any mesh set over the
            # top, and a mesh-only run draws the mesh. Nothing about the spec has to say which.
            _m = getattr(lvl, "mesh", None)
            self._mesh_is_subject = bool(_m is not None and int(_m.get("nF", 0) or 0))
            _r3d = str((self.style or {}).get("render_3d", "dots")).lower()
            if self._mesh_is_subject:
                pass
            elif _r3d == "compartments":
                # ONE TRANSLUCENT SURFACE PER COMPARTMENT. Falls back to the dot cloud if the
                # reconstruction cannot be built, like the single-surface path does.
                if not self._skins_build(H, lvl, pos):
                    self.p.add_mesh(self.cloud, scalars="rgb", rgb=True, **_flat,
                                    point_size=self._dot_px(pos))
            elif _r3d == "contour":
                # A SURFACE RE-CONTOURED EVERY FRAME, for a body whose topology changes -- a syrup
                # rope that coils, folds and merges with the pool it lands in. The bound `Skin` of
                # `render_3d: surface` is built once and advected, which is right for a body that
                # deforms and wrong for one that reconnects; `compartments` is per type and static
                # in topology for the same reason. This path pays for a fresh density contour each
                # frame and draws it as morph.py's dielectric, which is the armadillo look.
                if not self._contour_build(H, lvl, pos):
                    self.p.add_mesh(self.cloud, scalars="rgb", rgb=True, **_flat,
                                    point_size=self._dot_px(pos))
            elif (self.style or {}).get("dot_radius") is not None and (
                    self._glyph_cover_subject(H, lvl)
                    or bool((self.style or {}).get("glyphs_only", False))):
                # SPHERES OF WORLD RADIUS PER TYPE (below), and with `glyphs_only` NOTHING ELSE.
                #
                # `dot_radius` names the types to draw and `_glyph_types` skips any type without
                # an entry, which is what makes a layered build possible: a class absent from the
                # mapping is not drawn small or drawn dark, it is not drawn. But the fall-through
                # undid it -- an EMPTY mapping is falsy, so `dot_radius: {}` (draw no cells at
                # all, the rung that shows only the body's outline) skipped this branch entirely
                # and the plain cloud drew all 4,117 somata in the default blue. `is not None`
                # distinguishes "no mapping declared" from "a mapping that selects nothing", and
                # `glyphs_only` says the fall-back is unwanted even when the mapping selects
                # nothing, which is exactly the empty case.
                pass
            elif _r3d != "surface" or not self._skin_build(H, lvl, pos):
                self.p.add_mesh(self.cloud, scalars="rgb", rgb=True, **_flat,
                                point_size=self._dot_px(pos))
            # EYE-DOME LIGHTING, `plotting.edl: true`: a screen-space depth cue (Boucheny & Ribes
            # 2011) that darkens a pixel where its neighbours' depth jumps, so a flat cloud of dots
            # reads as bodies with an outline -- one render pass, no normals, no geometry.
            if bool((self.style or {}).get("edl", False)):
                try:
                    self.p.enable_eye_dome_lighting()
                except Exception as e:                        # noqa: BLE001
                    print(f"[live-movie] edl unavailable: {e}", flush=True)
            # AMBIENT OCCLUSION, `plotting.ssao: true`: the soft shadow a body casts into its own
            # crevices and onto what it stands on, from the depth buffer (VTK's SSAO pass). The
            # contour renderer has always had it with its own keys; this is the same pass for any
            # 3D scene -- spheres, compartment dots, surfaces -- with `ssao_radius_frac` of the box
            # (0.02) and `ssao_kernel` (128) as its two knobs. And `plotting.shadows: true` without
            # an obstacle: the obstacle block above only enables VTK's shadow pass when there is an
            # obstacle to cast one, but a scene of spheres and lit dots casts them too.
            if not self.is2d and bool((self.style or {}).get("ssao", False)):
                try:
                    _box = float(np.max(self.world))
                    self.p.enable_ssao(radius=_box * float((self.style or {}).get("ssao_radius_frac", 0.02)),
                                       bias=_box * float((self.style or {}).get("ssao_bias_frac", 0.001)),
                                       kernel_size=int((self.style or {}).get("ssao_kernel", 128)))
                except Exception as e:                        # noqa: BLE001
                    print(f"[live-movie] ssao unavailable: {e}", flush=True)
            if not self.is2d and bool((self.style or {}).get("shadows", False)) and not getattr(self, "_shadows_on", False):
                try:
                    self.p.enable_shadows(); self._shadows_on = True
                except Exception as e:                        # noqa: BLE001
                    print(f"[live-movie] shadows unavailable: {e}", flush=True)
            self._static_mesh(H)
            self._chain_build(H)
            self._spheres_build(H)
            self._also_build(H)                           # `plotting.also_sets` -- opt-in
            self._field_slice_update(H, first=True)
            self._field_iso_update(H, first=True)
            self._add_meshes(H)
            for _n, _l, _m in self._mesh_levels(H):
                self._edge_actor(H, _l, _m, first=True)
                break
            self._curves_setup(H, lvl)
            self._graph_setup(H)
            self._graph_update(H, tick)                   # drawn from frame 0 when `always`, else hidden
            self.t0 = time.perf_counter()
            # THE SETUP TICK IS ALSO A FRAME. It used to `return` here, having built every actor and
            # written nothing, so the first frame of every movie was tick `stride` -- 7 on a
            # 2,000-frame run, 20 on a 6,000-frame one -- and no live movie in the repository ever
            # showed the state its run started from. That is invisible until a slide has to begin
            # where the previous one ended: exp_03's diffusion slide opened on a disc whose seeded
            # dots had already blurred to 24 percent of their spread, directly after a slide that
            # showed them sharp. Tick 0 falls through to the write below (0 % stride is 0), so the
            # movie now opens on the initial condition -- which is what engine.py's own note on row 0
            # asks for ("THE INITIAL-CONDITION ROW IS STILL WANTED").
        if tick % self.stride:
            return
        self.cloud.points = self._xyz(lvl)
        self._chain_update(H)
        self._spheres_update(H)
        self._also_update(H)
        self._protein_insets_update(H)
        self._field_slice_update(H)
        self._field_iso_update(H)
        self._glyph_update(lvl)
        self._skin_update(H, lvl, self.cloud.points)
        self._contour_update(H, lvl, self.cloud.points)
        self._graph_update(H, tick)
        self._update_meshes(H)
        self._curves_update(tick, H)
        if getattr(self, "_base_rgb", None) is not None:
            self.cloud["rgb"] = self._body_shade(lvl, np.asarray(self.cloud.points), self._base_rgb)
        # A FIELD COLOUR IS A PROPERTY OF NOW, so unlike the body hue it is recomputed each frame.
        # A CHEMISTRY COLOUR IS TOO, and that is the whole reason it cannot ride on `_rgb`, which is
        # fixed at t = 0 and carried with the particle: the cells of a Turing disc never move, so a
        # colour frozen at t = 0 would render 300 identical frames of the initial condition.
        if str(self.style.get("color_field", "") or ""):
            _c = self._rgb_field(H, lvl)
            if _c is not None:
                self.cloud["rgb"] = _c
        elif getattr(self, "_chem_live", False):
            _c = (self._rgb_parent_chem(H, lvl) if getattr(self, "_chem_parent", False)
                  else self._rgb_chem(H, lvl))
            if _c is not None:
                self.cloud["rgb"] = _c
        el = time.perf_counter() - self.t0
        sub = f", {self.drawn:,} drawn" if self.drawn < self.n else ""
        # THE CLOCK, WHEN THERE IS ONE. With units declared the overlay carries the world's own
        # time and how fast the movie is running against it, so nobody has to ask.
        # THE WORLD'S OWN CLOCK, and nothing else. The playback rate is a property of the FILE,
        # reported once when the movie opens; repeating it on every frame said the same thing 300
        # times and crowded out the number that changes. `ms/frame compute` stays because it is the
        # machine's speed and it is genuinely useful while a run is in flight.
        clk = ""
        if self.speed is not None:
            clk = f"\nt = {tick * float(self.dt) * float(self.time_s):.4g} s"
            if abs(getattr(self, "slow_motion", 1.0) - 1.0) > 1e-9:
                clk += f"   {self.slow_motion:g}x slow"
        # WHAT THE COLOURS MEAN, ON THE PICTURE. The closing print names the LUT, but a movie is
        # read frame by frame and by someone who did not run it: a field colouring with no legend is
        # a colourful picture of an unnamed quantity, and the range matters as much as the name
        # since it is FIXED for the whole movie by design.
        _lut = ""
        if self.colour_by and self.colour_by != "?" and str(self.style.get("color_field", "") or ""):
            # `colour_by` ALREADY CARRIES THE RANGE AND THE MAP -- `_rgb_field` builds it as
            # "<label> <range> (<cmap>)" -- so appending them here printed each of them twice.
            _lut = f"\ncolour = {self.colour_by}"
        # `mesh_chem_label`: what the layer's face colour means, as a line of this header (the human, 2026-09-29)
        if (self.style or {}).get("mesh_chem_label") and bool((self.style or {}).get("mesh_chem", True)):
            _lut += f"\n{self.style['mesh_chem_label']}"
        # `nodes`, NOT `particles`. What is drawn is the members of a SET -- vertices of a mesh,
        # cells, neurons -- and only one substrate in this tree calls them particles.
        # "4,440 of 25,584 nodes" while a set is growing into its reservoir, and plainly
        # "25,584 nodes" once it is full or was never a growing set -- so the number that moves is
        # the one being watched, and the allocation is still visible behind it.
        _live = getattr(self, "n_live", self.n)
        _cnt = f"{_live:,}" if _live == self.n else f"{_live:,} of {self.n:,}"
        self.p.add_text(f"{self.name}{self._box_label}\n"
                        f"{_cnt} nodes{sub}{self._grid_label}\n"
                        f"frame {tick}/{self.n_frames}   "
                        f"{(self._fixed_ms if self._fixed_ms is not None else el / max(tick, 1) * 1000):.0f}"
                        f" ms/frame {self._rate_of}{clk}{_lut}",
                        position="upper_left", font_size=11, color=self._fg, name="hdr")
        if self.cs is not None:
            self._update_cross_section(H)
        # THE CLIPPING RANGE FOLLOWS WHAT IS DRAWN. `reset_camera` fits the near and far planes to
        # the actors present when it runs -- the subject cloud, a ring at the motor's base -- and
        # everything built after it that stands nearer the camera (the hook, the bushing, 60 nm
        # above the ring) was sliced by the near plane: a camera-facing window through the hook
        # and the bushing that did not turn with them (exp_02 steps 0034-0049, the "holes"). Reset
        # against every actor before each frame is written; VTK's own automatic reset does not
        # run under the passes (SSAO, EDL, depth peeling) this renderer uses.
        try:
            self.p.renderer.ResetCameraClippingRange()
        except Exception:                                            # noqa: BLE001
            pass
        self.p.write_frame()
        self.rendered += 1
        if tick in self.still_ticks:
            self._still(tick)

    # ---- the inset curves ------------------------------------------------------------------
    #
    # `plotting.curve` PORTED FROM `render_vtk._curve_setup`, and generalised in two ways it needed:
    # it names a QUANTITY rather than being wired to `e_myo`, and it may be a LIST, because "how many
    # cells" and "how big are they" are two questions and answering them in one panel would need two
    # y axes. The chart styling -- no box, no title, white axis text through the VTK accessors,
    # 3-significant-figure ticks -- is that function's, decision for decision; see it for the why.
    #
    #     quantity: cells      the live face count, one line, no band
    #     quantity: area       mean +- SD of the live faces' areas
    #
    # THE AXES ARE FIXED OVER THE WHOLE CLIP, which is the part that makes the panel readable and the
    # part that constrains where this can run. An autoscaled y renormalises every frame, so a
    # population that doubles looks exactly like one that does not move. Fixing it needs the whole
    # series up front, so the panel is built on the REPLAY path, which has the trajectory; a live
    # generate has only the frame it is on and says so rather than drawing a rescaling plot.
    # `radius` IS THE ONE THAT SPEAKS TO A CONTACT. A shell's mean radius says how big it is; the
    # SPREAD says whether it is still a shell. `mesh_contact` is star-shaped and its whole premise
    # is that a ray from `centre` meets the surface once, so a rising sd is the surface going
    # off-sphere -- which is what collapses the direction-bin grid, and it does so long before the
    # picture looks wrong.
    #     quantity: myosin    mean +- SD of the junctional myosin, per type -- the only one of
    #                         these that lives on an EDGE rather than on a face, so it is grouped by
    #                         the type of the cell each half-edge belongs to.
    #     quantity: cycle_progress   mean +- SD of the cells' continuous position through the
    #                         cycle, 0 at birth and 1 at the end of M. S4's state, and the one the
    #                         `phase` panel cannot show: four fractions say how the population is
    #                         SPLIT between phases, this says how far through it is, so a tissue
    #                         cycling in waves and one cycling steadily are told apart by the SD.
    from plexus.measures import CURVE_QUANTITIES as _CURVE_Q   # the registry names them

    @staticmethod
    def _curve_dim(q):
        """The dimension a curve quantity converts through; `count:<set>` is a count."""
        from plexus.measures import CURVE_DIMS
        return "count" if str(q).startswith(("count:", "species:")) else (None if str(q).startswith(("block:", "total:", "jacobian:", "strain:", "rate:")) else CURVE_DIMS.get(q))

    # WHAT EACH CURVE IS, SO THE PANEL CAN CONVERT IT. Until this existed the volume panel appended
    # `µm³` to a raw simulation number and was wrong by `length_um ** 3` -- a factor of 1000 on every
    # tissue run in the tree. The unit is now DERIVED from the same declaration that converts the
    # number, so a label can no longer appear over a value nobody scaled. `myosin` is deliberately
    # absent: it is a per-junction activity with no declared dimension, and UNKNOWN prints bare.
    from plexus.measures import CURVE_DIMS as _CURVE_UNITS   # one declaration, with the measures

    def _species_n(self, H, q):
        """How many lines `species:<set>` draws: the width of the set's `chem` block, else the number of
        `plotting.species` colours, else one."""
        try:
            a, b = H.level(str(q)[len("species:"):]).state_schema["chem"]
            return max(1, int(b) - int(a))
        except Exception:                                # noqa: BLE001
            return max(1, len((self.style or {}).get("species") or []))

    def _curve_block_ok(self, H, lvl, q):
        """Does `block:<set>:<block>` / `total:<set>:<block>` name something this run has? A set that is a
        level answers through its schema; the mesh's cell set, which on a replay is not a level, through the
        cell columns the mesh carries (width-1 blocks only)."""
        if q.count(":") != 2:
            return False
        _, name, block = q.split(":", 2)
        blocks = [b for b in block.split("@") if b]                  # `total:<set>:<block>@<weight>`
        if name in getattr(H, "levels", {}):
            sch = getattr(H.level(name), "state_schema", None)
            try:
                return sch is None or all(b in sch for b in blocks)
            except TypeError:
                return True
        m = getattr(lvl, "mesh", None)
        nF = int(m.get("nF", 0) or 0) if m is not None else 0
        cc = self._cell_cols(H, lvl, nF) if nF else {}
        return bool(nF) and all(b in cc for b in blocks)

    def _curve_series(self, H, lvl, q, ntype):
        """[T, ntype, 2] of (mean, sd) for `q` over every recorded frame. Replay only.

        PARTITIONED BY TYPE WHEN THERE IS ONE. A face of the mesh IS a cell, so the cell set's
        `node_type` indexes the faces directly and a per-type split costs one mask. With no types
        declared it is a single series -- which is the honest picture and not a degenerate case of
        the other, because "the mean over one population" and "the mean over each of several" are
        different claims.
        """
        T = int(getattr(lvl, "_pos").shape[0])
        nt = 1 if ntype is None else int(np.max(ntype)) + 1
        if q == "phase":
            nt = 4                                   # G1, S, G2, M -- the partition IS the phase
        elif str(q).startswith("species:"):
            nt = self._species_n(H, q)               # one line per strain -- the partition IS the strain
        out = np.full((T, nt, 2), np.nan)
        for t in range(T):
            lvl.t = t
            out[t] = self._curve_row(H, lvl, q, ntype, nt)
        lvl.t = 0
        return out

    def _clone_series(self, H, lvl, q, top):
        """(S, ids, nsurv) for `clones:<block>` over every recorded frame -- `measures.muller_bands` on
        the block's value for each frame's live cells, read through the same cell-column overlay the
        face colouring uses. Replay only."""
        from plexus.measures import muller_bands
        block = str(q).split(":", 1)[1]
        T = int(getattr(lvl, "_pos").shape[0])
        frames = []
        for t in range(T):
            lvl.t = t
            m = getattr(lvl, "mesh", None)
            nF = int(m.get("nF", 0) or 0) if m is not None else 0
            v = self._cell_cols(H, lvl, nF).get(block) if nF else None
            frames.append(None if v is None else
                          np.rint(np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                                             float)[:nF]).astype(np.int64))
        lvl.t = 0
        return muller_bands(frames, top)

    def _curve_row(self, H, lvl, q, ntype, nt):
        """[nt, 2] of (mean, sd) for `q` at the level's CURRENT frame -- one row of `_curve_series`.
        The body lives in `plexus.measures.curve_row`, one entry point with the gates and the
        fingerprints; this passes the renderer's cell-column reader and nothing else."""
        from plexus.measures import curve_row
        return curve_row(H, lvl, q, ntype, nt, self._cell_cols)

    def _curve_types(self, H, lvl):
        """The per-cell type ids, or None. `mesh_cell_set` names which set a face belongs to."""
        _nts = getattr(H, "node_types", None) or {}
        for nm in (getattr(lvl, "mesh_cell_set", None), "cell"):
            v = _nts.get(nm)
            if v is not None and int(np.max(v)) > 0:
                return np.asarray(v).astype(int)
        for nm in (getattr(lvl, "mesh_cell_set", None), "cell"):
            if not nm:
                continue
            try:
                v = getattr(H.level(nm), "node_type", None)
            except Exception:                            # noqa: BLE001
                continue
            if v is None:
                continue
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            if v.ndim > 1:
                v = v[0]
            if int(np.max(v)) > 0:                       # one type is no partition at all
                return v.astype(int)
        return None

    def _curves_setup(self, H, lvl):
        self._curves = []
        cfgs = (self.style or {}).get("curve")
        if not cfgs:
            return
        cfgs = [cfgs] if isinstance(cfgs, dict) else list(cfgs)
        # THREE PANELS, STACKED. More than that and each is a hundred pixels tall on a 1280 frame --
        # too short to read a spread off, which is the whole reason the band is drawn. `loc` still
        # wins where a spec gives one; without it they stack down the left, which is the arrangement
        # "one on top of another" means and saves every spec from computing three offsets.
        if len(cfgs) > 3:
            raise ValueError(f"plotting.curve: {len(cfgs)} panels asked for, at most 3 fit "
                             f"legibly -- each is 26% of the frame's height")
        # THE CURVE READS THE SET THAT CARRIES THE MESH, not the set being drawn. `cells`, `area`
        # and `radius` are properties of a SURFACE, and the drawn set is whichever positional set is
        # largest -- which in a coupled run is the 200,000 material points, not the 25,000-vertex
        # vesicle. Pointed at the particles they came back all-NaN and the panels were skipped
        # without a word, so a spec that asked for three curves silently got none.
        # `live` IS DECIDED BEFORE THE MESH SET IS CHOSEN, because the choice keys on `_pos` -- a
        # replay-only attribute -- and a live pass has it on no level. Keyed after, the live pass
        # kept the SUBJECT, which in a coupled run is the material points; a particle set has no
        # mesh, every row came back NaN, and the panels drew as `0` and `nan +- nan`.
        live = not hasattr(lvl, "_pos")
        lq = lvl
        for _nm, _lv in H.levels.items():
            _m = getattr(_lv, "mesh", None)
            if _m is not None and int(_m.get("nF", 0) or 0) and (live or hasattr(_lv, "_pos")):
                lq = _lv
                if _nm != getattr(self, "_sname", None):
                    print(f"[live-movie] curves read set {_nm!r} (it carries the mesh), not "
                          f"{getattr(self, '_sname', '?')!r}", flush=True)
                break
        lvl = lq
        self._curve_set = next((nm for nm, lv in H.levels.items() if lv is lvl), None)
        ntype = self._curve_types(H, lvl)
        # LIVE WHEN THE AXES ARE DECLARED. The replay has the whole clip and fixes each panel's
        # range from it; a live generate has only the current frame, so it can draw the same
        # panels only if the spec says the range -- `ymin`/`ymax` on every curve -- and the series
        # is then filled one row a frame in `_curves_update`. Without them it declines, as before.
        if live and not all(("ymin" in c and "ymax" in c) for c in cfgs):
            print("[live-movie] plotting.curve needs the whole clip to fix its axes and a live "
                  "generate has only the current frame -- declare `ymin`/`ymax` on every curve to "
                  "draw it live, or re-render with `-o plot`", flush=True)
            return
        T_live = int(self.n_frames) + 1
        for _i, cfg in enumerate(cfgs):
            cfg = dict(cfg)
            # THE KIND IS CASE-FREE, A BLOCK'S NAME IS NOT: lowercasing all of `block:cell:nK_in` looked up `nk_in`,
            # which no set has, and the panel drew nothing -- the ion-flow runs' "K+ inside" and "Cl- inside"
            # panels read "nan +- nan" through batch 10 while the trajectory held the counts.
            _q = str(cfg.get("quantity", "cells"))
            q = _q if _q.lower().startswith(("block:", "total:", "count:", "jacobian:", "strain:", "rate:", "species:", "clones:")) else _q.lower()
            if ":" in q:
                q = q.split(":", 1)[0].lower() + ":" + q.split(":", 1)[1]
            if q not in self._CURVE_Q and not any(str(q).startswith(k) for k in ("count:", "block:", "total:", "jacobian:", "strain:", "rate:", "species:", "clones:")):
                raise ValueError(f"plotting.curve.quantity: {q!r} is not one of "
                                 f"{', '.join(self._CURVE_Q)}, count:<set>, block:<set>:<block>, total:<set>:<block>, jacobian:<set>, strain:<set> or rate:<set>")
            if str(q).startswith(("block:", "total:")) and not self._curve_block_ok(H, lvl, q):
                raise ValueError(f"plotting.curve.quantity: {q!r} must be block:<set>:<block> or total:<set>:<block> with a set and block this run has")
            if str(q).startswith("count:") and q[len("count:"):].split(":", 1)[0] not in getattr(H, "levels", {}):
                raise ValueError(f"plotting.curve.quantity: {q!r} names a set this run does not have")
            if str(q).startswith("species:") and q[len("species:"):] not in getattr(H, "levels", {}):
                raise ValueError(f"plotting.curve.quantity: {q!r} must be species:<set> with a set this run has")
            _clones = None
            if str(q).startswith("clones:"):
                # A MULLER PLOT, AND ONLY ON THE REPLAY: which labels get their own band is decided
                # from their PEAK share over the whole clip, which a live pass does not have yet.
                if live:
                    print(f"[live-movie] curve {q}: a Muller plot needs the whole clip -- re-render with "
                          f"`-o plot` (or tools/exp14_twin_render.py)", flush=True)
                    continue
                _clones = self._clone_series(H, lvl, q, int(cfg.get("top", 12)))
            if _clones is not None:
                S = _clones[0]
            elif live:
                nt0 = (4 if q == "phase" else self._species_n(H, q) if str(q).startswith("species:")
                       else (1 if ntype is None else int(np.max(ntype)) + 1))
                S = np.full((T_live, nt0, 2), np.nan)
            else:
                S = self._curve_series(H, lvl, q, ntype)
                # `with: <quantity>` -- A SECOND QUANTITY ON THE SAME PANEL, its own line (exp 11, 2026-09-27: the
                # surface and the interior cell counts on one curve). `colors: [a, b]` colours the two. Replay only:
                # a live pass keeps one series per panel.
                # `with: [q2, q3, ...]` -- AS MANY AS THE PANEL NEEDS (exp 11, 2026-09-29: the membrane's node count
                # and every protein's total on one panel, `total:<set>:<block>`), one colour each from `colors:`,
                # named by `labels:` in a legend. A named set or block the run does not have is an error here,
                # not a flat line: a protein panel on a run without the proteins must say so.
                _w = cfg.get("with")
                if _w:
                    _ws = [_w] if isinstance(_w, str) else list(_w)
                    _parts = [S[:, :1, :]]
                    for _q2 in _ws:
                        _q2 = str(_q2)
                        _q2 = _q2 if _q2.lower().startswith(("block:", "total:", "count:", "jacobian:", "strain:", "rate:",
                                                             "species:", "clones:")) else _q2.lower()
                        if _q2.startswith("count:") and _q2.split(":")[1] not in getattr(H, "levels", {}):
                            raise ValueError(f"plotting.curve.with: {_q2!r} names a set this run does not have")
                        if _q2.startswith(("block:", "total:")) and not self._curve_block_ok(H, lvl, _q2):
                            raise ValueError(f"plotting.curve.with: {_q2!r} -- no such set and block in this run")
                        _parts.append(self._curve_series(H, lvl, _q2, ntype)[:, :1, :])
                    n = min(p_.shape[0] for p_ in _parts)
                    S = np.concatenate([p_[:n] for p_ in _parts], axis=1)
                # `log: true` -- A LOG Y AXIS, for series decades apart (a membrane of 27,000 nodes beside a
                # protease total of a few). Nothing <= 0 has a place on it, so those points are left out.
                if cfg.get("log"):
                    S = S.copy(); _nz = ~(S[..., 0] > 0)
                    S[..., 0][_nz] = np.nan; S[..., 1] = 0.0
                if not np.isfinite(S[..., 0]).any():
                    continue
            # CONVERT ONCE, HERE, AND EVERY SURFACE FOLLOWS. The y-axis range, the +-SD band, the
            # lines and the on-panel `mean +- SD` readout are all derived from `S`, so scaling it
            # at the source is what keeps them from disagreeing -- which is exactly how the label
            # and the number came to disagree in the first place. `to_physical` returns None when
            # the run declared no scale, and then nothing is scaled and nothing is labelled.
            # `unit:` ON THE CURVE names the dimension of a block the quantity table cannot know
            # (`block:cell:psi` is a voltage only the spec can say), converted and labelled through
            # the run's declaration like every registered quantity; `factor:` remains for a bare
            # number.
            _qdim = cfg.get("unit") or self._curve_dim(q)
            _scale = _units_to_physical(1.0, _qdim, self._units)
            # `factor:` ON THE CURVE, for a block the units table cannot convert: a membrane
            # potential kept as e psi in sim energy is drawn in mV by the factor the spec states.
            if cfg.get("factor") is not None:
                _scale = (float(_scale) if _scale is not None else 1.0) * float(cfg["factor"])
            if _scale is not None and _scale != 1.0:
                S = S * float(_scale)
            # `smooth: <frames>` ON A REPLAY: the whole series is known, so the exponential running
            # mean is taken here, one recorded frame at a time, and the axis below is framed from
            # the SMOOTHED series -- framed from the raw spikes, a 11 fA current sat flat on a
            # -141..159 fA axis (step 0073). The live pass smooths in `_curves_update` instead.
            _sm = float(cfg.get("smooth", 0.0) or 0.0)
            if not live and _sm > 0.0 and S.shape[0] > 1:
                _smw = 1.0 - math.exp(-1.0 / _sm)
                _smu = S[..., 0].copy()
                for _smk in range(1, _smu.shape[0]):          # its own name: `_i` is the panel index
                    _smp, _smc = _smu[_smk - 1], _smu[_smk]
                    _smok = np.isfinite(_smp) & np.isfinite(_smc)
                    _smu[_smk] = np.where(_smok, _smp + _smw * (_smc - _smp), _smc)
                S[..., 0] = _smu
            lo, hi = ((float(cfg["ymin"]), float(cfg["ymax"])) if live else
                      (float(np.nanmin(S[..., 0] - S[..., 1])), float(np.nanmax(S[..., 0] + S[..., 1]))))
            # THE ENDS ARE ROUND NUMBERS, AND THE FLOOR IS ZERO WHERE ZERO IS THE FLOOR. An 8% pad
            # below the minimum put the `cells` axis at -317, i.e. a tick labelled with a negative
            # count of cells, and left the top at 7181 -- five digits of a bound that is a padding
            # artefact, not a measurement. Both ends are snapped to ONE significant figure (7181 ->
            # 8000, 0.6494 -> 0.7) so the first and last ticks are numbers a reader can hold, and a
            # quantity that cannot be negative starts at 0. `ymin`/`ymax` still override.
            def _r1(v, up):
                if v == 0 or not np.isfinite(v):
                    return 0.0
                e = 10.0 ** np.floor(np.log10(abs(v)))
                f = np.ceil(abs(v) / e) if (v > 0) == up else np.floor(abs(v) / e)
                return float(np.sign(v) * max(f, 1.0) * e)
            lo = 0.0 if (lo >= 0 or q in ("cells", "area", "volume", "myosin") or str(q).startswith("count:")) else _r1(lo, False)
            # A ROUND STEP, NOT A ROUND TOP. Snapping only the top to one significant figure still
            # left the ticks between the ends to be whatever the count divided into: 0..8000 over
            # `ticks: 4` printed 0, 2667, 5333, 8000, and the two in the middle are the artefact of
            # a division, not numbers anyone chose. The STEP is snapped instead -- to the largest of
            # {1, 2, 2.5, 5} x 10^k that is no bigger than range/(ticks-1) -- and the top is then the
            # first multiple of it above the data. Every tick is round by construction, and the tick
            # COUNT follows from the step rather than forcing it.
            _ticks_req = int(cfg.get("ticks", 4))
            _floor0 = bool(lo == 0.0)
            lo, hi, cfg["ticks"] = _snap_range(lo, hi, _ticks_req)
            pad = 0.0
            # THE PANEL'S WIDTH IS A FRACTION OF THE WINDOW, USED AS GIVEN. Dividing it by the
            # aspect kept its SHAPE constant and left the column it was supposed to fill only
            # two thirds occupied -- a band of blank down the right edge. The column was sized from
            # this number in the first place, so the two agree by construction.
            _sz = tuple(cfg.get("size", self._curve_size))
            # DOWN THE RIGHT. The left is where the header prints -- the name, the box, the frame
            # counter, the LUT -- so a panel there sits under four lines of text on the first row
            # and the stack has to start below them.
            ch = self.pv.Chart2D(size=_sz,
                                 loc=tuple(cfg.get(
                                     "loc", (1.0 - self._curve_col + 0.005,
                                             0.71 - _i * (_sz[1] + 0.05)))))
            ch.background_color = (0, 0, 0, 0.0)
            ch.border_style = None
            ch.title = str(cfg.get("title", ""))
            # `plotting.curve_time: {per_frame_s, unit}` -- THE X AXIS IN PHYSICAL TIME, not frames: a
            # 400,000-frame run printed its ticks as "79999159999239998", and the target asks for curves
            # "against time in physical units" (exp04, 2026-09-26). Absent: frames, as before.
            _ct = (self.style or {}).get("curve_time") or {}
            self._curve_xs = (float(_ct["per_frame_s"]) * {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}[str(_ct.get("unit", "ns"))]
                              * float(getattr(self, "_frames_per_row", 1.0)) if _ct.get("per_frame_s") else 1.0)
            ch.x_axis.range = [0.0, float(S.shape[0] - 1) * self._curve_xs]
            # A DECLARED RANGE IS WHERE THE AXIS STARTS, NEVER A CLIP. Replaying off the trajectory,
            # the whole series is known: if it climbs past the declared top the top moves so the
            # maximum sits at 60% of the axis (the same rule `_curve_follow` applies frame by frame
            # in a live generate); a declared bottom that is not the zero floor moves the same way.
            # Before this, `ymax: 120` on a 200-cell run drew an empty panel with "200" above it.
            y0 = float(cfg.get("ymin", lo - pad)); y1 = float(cfg.get("ymax", hi + pad))
            # ON A REPLAY THE AXIS COMES FROM THE DATA (the human's rule: a range is measured, not
            # guessed). `ymin`/`ymax` are what the live pass needs, having only the current frame;
            # the replay has the whole series and frames it with 5% of air, unless the spec says
            # `fixed_range: true` to keep runs comparable on one axis.
            # AND FROM ZERO WHEN THE DATA NEVER GOES BELOW IT. Framed from the data alone, a count
            # holding at 372 cells was drawn on a 372..380 axis and a mean area of 18.9 um^2 on
            # 16..24: a flat line looked like a curve and a 2% band looked like a factor. Zero is
            # the floor of every quantity that has one; only a series that goes negative (a
            # voltage, a signed current) is framed from its own minimum.
            if not live and not bool(cfg.get("fixed_range", False)):
                _mu = S[..., 0]; _sd = np.nan_to_num(S[..., 1])
                _d0, _d1 = float(np.nanmin(_mu - _sd)), float(np.nanmax(_mu + _sd))
                if np.isfinite(_d0) and np.isfinite(_d1):
                    _air = 0.05 * max(_d1 - _d0, 1e-12)
                    _lo0 = 0.0 if _d0 >= 0.0 else _d0 - _air
                    y0, y1, cfg["ticks"] = _snap_range(_lo0, _d1 + _air, _ticks_req)
            if not live:
                _dmax = float(np.nanmax(S[..., 0] + S[..., 1])); _dmin = float(np.nanmin(S[..., 0] - S[..., 1]))
                # A RELATIVE TOLERANCE, because a series that ENDS at the declared top -- a Muller plot's
                # last band edge at 100 %, summed in floating point -- reads 100.00000000000001 and
                # otherwise widened a `fixed_range` 0..100 axis to 0..166.
                if np.isfinite(_dmax) and _dmax > y1 + 1e-9 * max(abs(y1), 1.0):
                    y0, y1, cfg["ticks"] = _snap_range(y0, y0 + (_dmax - y0) / 0.6, _ticks_req)
                if np.isfinite(_dmin) and _dmin < y0 and y0 != 0.0:
                    y0 = _round1(_dmin - 0.4 * (y1 - _dmin), False)
                    y0, y1, cfg["ticks"] = _snap_range(y0, y1, _ticks_req)
            if cfg.get("log") and not live:
                _pv = S[..., 0][np.isfinite(S[..., 0]) & (S[..., 0] > 0)]
                if _pv.size:
                    y0 = float(10.0 ** np.floor(np.log10(_pv.min())))
                    y1 = float(10.0 ** np.ceil(np.log10(_pv.max() * 1.0001)))
                    if y1 <= y0:
                        y1 = y0 * 10.0
                    try:
                        ch.y_axis.log_scale = True
                    except Exception:                        # noqa: BLE001
                        pass
            ch.y_axis.range = [y0, y1]
            ch.x_axis.label = str(cfg.get("xlabel", (f"time ({_ct.get('unit', 'ns')})" if _ct.get("per_frame_s") else "frame")))
            # UNITS ON THE Y AXIS, DERIVED FROM THE DECLARED SCALE AND NOT TYPED IN.
            # `general.units: length_um` is the one thing that turns a length in this model into a
            # length in the world, so a panel may say um, um^2 or um^3 exactly when that key is
            # present -- and must say nothing when it is absent, because an undeclared run is
            # dimensionless and a unit on it would be a claim the spec did not make. A count and a
            # percentage carry no length unit either way.
            # THE REAL CHARACTERS. VTK's text renderer takes them, so `um^3` -- an ASCII
            # transliteration of micro and a caret standing in for an exponent -- was a choice, not
            # a limitation, and it sat two lines under a scale bar already saying `\u00b5m`.
            _u = _units_label(cfg.get("unit") or self._curve_dim(q), self._units)
            _u = {"phase": "%"}.get(q, _u)
            ch.y_axis.label = str(cfg.get("ylabel", q)) + (f"  ({_u})" if _u else "")
            # TWO SIZES, NOT ONE MINUS TWO. The axis TITLE ("cells", "frame") and the TICK NUMBERS
            # are read at different distances -- the title once, the ticks repeatedly while
            # following a line -- so they get their own keys instead of one being derived from the
            # other. Both default larger than they were: at 9 and 7 on a 1280 px frame the ticks
            # were about eight pixels tall.
            # SIZED FOR THE FRAME THE PANEL ENDS UP IN. These are absolute point sizes, and the
            # frame grew from 1280 square to 2048x1280 when the curve column was added -- so a size
            # that was small at 1280 is smaller still as a fraction of the wider frame. 18/15 on a
            # panel ~600 px across is readable at the size these clips are actually watched.
            _fs = int(cfg.get("font_size", 18))
            _tfs = int(cfg.get("tick_font_size", max(6, _fs - 3)))
            for _a in (ch.x_axis, ch.y_axis):
                _a.label_visible = _a.ticks_visible = _a.tick_labels_visible = True
                _a.grid = False
                _a.label_size = _fs
                _a.tick_label_size = _tfs
                _a.tick_count = int(cfg.get("ticks", 4))
                try:
                    # ENOUGH DIGITS THAT ADJACENT TICKS DIFFER, derived from the tick SPACING and
                    # not from the axis's top: a range ending at 8000 needs no decimals and one
                    # ending at 0.7 needs one, and hard-coding "%.2f" printed 0.20, 0.40, 0.60.
                    #
                    # KEYING ON THE TOP WAS WRONG FOR EVERY AXIS BETWEEN 1 AND 10, which is most of
                    # them: an area axis running 0 to 2 has log10(2) = 0.3, floor 0, so it asked for
                    # zero decimals and printed its five ticks as "0, 0, 1, 2, 2" -- two pairs of
                    # duplicate labels and no way to read the middle. The spacing (2/4 = 0.5) is
                    # what has to be resolvable, and it asks for the one digit the comment claimed.
                    _lo, _hi = ((float(ch.y_axis.range[0]), float(ch.y_axis.range[1]))
                                if _a is ch.y_axis else (0.0, float(len(S) - 1) * getattr(self, "_curve_xs", 1.0)))
                    _n = max(1, int(cfg.get("ticks", 4)))
                    _sp = abs(_hi - _lo) / _n
                    _dec = int(min(3, max(0, -np.floor(np.log10(max(_sp, 1e-12))))))
                    _a.SetNotation(_a.PRINTF_NOTATION)
                    _fmt = str(cfg.get("tick_format", f"%.{_dec}f"))
                    _a.SetLabelFormat(_fmt)
                    # AND THE FIRST AND LAST TICK, which vtkAxis formats through a SEPARATE
                    # `RangeLabelFormat`. Leaving it default is why an axis whose middle ticks read
                    # 0.5, 1.0, 1.5 still capped itself with a bare "0" and "2".
                    if hasattr(_a, "SetRangeLabelFormat"):
                        _a.SetRangeLabelFormat(_fmt)
                except Exception:                        # noqa: BLE001
                    pass
                try:
                    _a.pen.color = self._fg
                    # NOT BOLD. vtkAxis renders both its title and its tick labels bold by default,
                    # which at this size reads as emphasis the panel is not making -- and bold white
                    # on black blooms, so the strokes close up and a 3 becomes an 8. `SetColor` was
                    # already being reached through these accessors; the weight is on the same
                    # objects and was simply never set.
                    for _tp in (_a.GetLabelProperties(), _a.GetTitleProperties()):
                        _tp.SetColor(1.0, 1.0, 1.0)
                        _tp.SetBold(0)
                    _a.GetTitleProperties().SetFontSize(_fs)
                    _a.GetLabelProperties().SetFontSize(_tfs)
                except Exception:                        # noqa: BLE001
                    pass
            ch.legend_visible = False
            # COLOURED BY TYPE WHERE THERE ARE TYPES, WHITE WHERE THERE ARE NOT. A single population
            # drawn in the first slot of a categorical palette invites the reader to ask what the
            # other colours would have been; white says there is one thing being measured.
            nt = S.shape[1]
            _pal = [tuple(v) for v in ((self.style or {}).get("colors") or {}).values()]
            _pal = _pal or [(0.35, 0.60, 1.00), (1.00, 0.35, 0.25),
                            (0.45, 0.95, 0.55), (1.00, 0.85, 0.30)]
            # THE PHASE CURVE TAKES THE PHASE COLOURS, so the panel and the tissue say the same
            # thing with the same four colours and neither needs a legend.
            if q == "phase":
                import matplotlib.colors as _mc2
                _pc = (self.style or {}).get("phase_colors") or ["#3b57c0", "#2e9e4f",
                                                                 "#e8b024", "#e03b2f"]
                cols = [tuple(_mc2.to_rgb(c)) for c in _pc[:nt]]
            elif _clones is not None:
                # EACH BAND IN ITS LABEL'S OWN COLOUR (`measures.label_rgb`, the rule the faces use),
                # the other labels together in grey.
                from plexus.measures import label_rgb
                _lp = (self.style or {}).get("label_colors") or []
                cols = [tuple(label_rgb(int(i), _lp)) for i in _clones[1]] + [(0.35, 0.35, 0.35)]
            elif str(q).startswith("species:"):
                # THE SPECIES CURVE TAKES THE SPECIES COLOURS, as the phase curve takes the phase
                # colours: the strain drawn red on the lattice is the red line.
                import matplotlib.colors as _mc3
                _sc = (self.style or {}).get("species") or []
                cols = [tuple(_mc3.to_rgb(_sc[j])) if j < len(_sc) else _pal[j % len(_pal)] for j in range(nt)]
            elif cfg.get("colors"):
                import matplotlib.colors as _mc4
                cols = [tuple(_mc4.to_rgb(c)) for c in cfg["colors"]][:nt]
                cols += [_pal[j % len(_pal)] for j in range(len(cols), nt)]
            else:
                cols = ([tuple(cfg["color"])] if "color" in cfg and nt == 1
                        else [(1.0, 1.0, 1.0)] if nt == 1
                        else [_pal[j % len(_pal)] for j in range(nt)])
            bands, lines = [], []
            for j in range(nt):
                c = cols[j]
                # A MULLER BAND IS THE WHOLE STATEMENT, so it is opaque and has no centre line.
                bands.append(ch.area([0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     color=(*c, 1.0 if _clones is not None else 0.28)))
                lines.append(ch.line([0.0, 0.0], [0.0, 0.0],
                                     color=(*c, 0.0 if _clones is not None else 1.0), width=2.0,
                                     label=str((cfg.get("labels") or [])[j]) if j < len(cfg.get("labels") or []) else ""))
            # `labels: [...]` -- A LEGEND inside the plot, white on clear, one entry per line, when the panel
            # carries more series than its y label can name in words; `legend_loc: top_left` (default),
            # `top_right`, `bottom_left` or `bottom_right` moves it off the curves.
            if cfg.get("labels"):
                ch.legend_visible = True
                try:
                    _lg = ch.GetLegend()
                    _v, _h = str(cfg.get("legend_loc", "top_left")).split("_", 1)
                    _lg.SetHorizontalAlignment({"left": _lg.LEFT, "right": _lg.RIGHT}[_h])
                    _lg.SetVerticalAlignment({"top": _lg.TOP, "bottom": _lg.BOTTOM}[_v])
                    _lg.SetLabelSize(int(cfg.get("legend_font_size", max(8, int(cfg.get("font_size", 18)) - 6))))
                    _lg.GetLabelProperties().SetColor(1.0, 1.0, 1.0)
                    _lg.GetBrush().SetColorF(0.0, 0.0, 0.0); _lg.GetBrush().SetOpacityF(0.0)
                    _lg.GetPen().SetOpacityF(0.0)
                    _lg.SetInline(True)
                except Exception:                            # noqa: BLE001
                    pass
            self.p.add_chart(ch)
            # WHERE THE VALUE READOUT GOES, from the same `loc` and `size` the chart got rather than
            # a second guess at them -- see `_curves_update` for what it prints.
            _loc = tuple(cfg.get("loc", (1.0 - self._curve_col + 0.005,
                                         0.71 - _i * (_sz[1] + 0.05))))
            # THE READOUT SITS OVER THE PLOT AREA, NOT OVER THE AXIS LABEL. `+ 0.010` put it hard
            # against the panel's left edge, above the y-axis title rather than above the curve it
            # reports; `+ 0.055` centres it on the data. `fs` is the HEADER's size, because the two
            # are the same kind of statement -- a number the reader is meant to take away -- and
            # the tick size made this one look like an axis annotation.
            _nw = (0 if not cfg.get("with") else 1 if isinstance(cfg.get("with"), str) else len(cfg.get("with")))
            self._curve_labels.append(dict(name=f"curveval{_i}", q=q, unit=_u, **({"with": True} if _nw == 1 else {}),
                                           **({"multi": True} if _nw > 1 else {}),
                                           x=float(_loc[0]) + 0.055,
                                           y=float(_loc[1]) + float(_sz[1]) + 0.004,
                                           fs=int(cfg.get("value_font_size", 11))))
            self._curves.append({"S": S, "bands": bands, "lines": lines, "nt": nt,
                                 # `legend_frac` (default 0.03): the legend shows over the first 3 % of the run's
                                 # frames only -- it is readable while the panel is still empty, and the curves
                                 # it names then run under it (the human, 2026-09-29). 1.0 keeps it throughout.
                                 "legend": bool(cfg.get("labels")), "legend_frac": float(cfg.get("legend_frac", 0.03)),
                                 "sd": bool(cfg.get("sd", q not in ("cells", "phase") and not str(q).startswith("species:")))
                                       or _clones is not None,
                                 "nsurv": None if _clones is None else _clones[2],
                                 "live": live, "lvl": lvl, "q": q, "ntype": ntype,
                                 "ch": ch, "ticks": _ticks_req, "floor0": _floor0,
                                 "declared": ("ymin" in cfg, "ymax" in cfg),
                                 "scale": (float(_scale) if (_scale is not None and _scale != 1.0) else None),
                                 # `smooth: <frames>`: an exponential running mean over that many frames,
                                 # applied to the series as it is revealed (live and replay alike). For a
                                 # current made of 10 us strokes on a 10 us frame the raw sum is spikes
                                 # between 0 and a few tens of fA; the mean is the current.
                                 "smooth": float(cfg.get("smooth", 0.0) or 0.0)})
            print(f"[live-movie] curve {q}: {S.shape[0]} frames, {nt} "
                  f"{'series by type' if nt > 1 else 'series'}, "
                  f"y [{ch.y_axis.range[0]:.4g}, {ch.y_axis.range[1]:.4g}]", flush=True)

    def _curve_follow(self, cv, row):
        """A live curve that leaves its declared range gets a WIDER range, once, not a new one
        every frame. The declared `ymin`/`ymax` are where the axis starts; when the value climbs
        past the top, the top moves so the value sits at 60% of the new range (it has to grow
        1.67x before the axis moves again), snapped to the same round step as at setup; when it
        drops below a bottom that is not the zero floor, the bottom moves the same way. An axis
        that renormalised every frame is unreadable; one that never moves loses the curve.
        """
        ch = cv.get("ch")
        if ch is None:
            return
        lo, hi = float(ch.y_axis.range[0]), float(ch.y_axis.range[1])
        mu = np.asarray(row[:, 0], dtype=float)
        sd = np.asarray(row[:, 1], dtype=float) if cv.get("sd") else np.zeros_like(mu)
        ok = np.isfinite(mu)
        if not ok.any():
            return
        vmax = float(np.nanmax((mu + sd)[ok])); vmin = float(np.nanmin((mu - sd)[ok]))
        changed = False
        if vmax > hi:
            lo, hi, n = _snap_range(lo, lo + (vmax - lo) / 0.6, cv.get("ticks", 4))
            changed = True
        if vmin < lo and not cv.get("floor0", False):
            lo = _round1(vmin - 0.4 * (hi - vmin), False)
            lo, hi, n = _snap_range(lo, hi, cv.get("ticks", 4))
            changed = True
        if changed:
            ch.y_axis.range = [lo, hi]
            ch.y_axis.tick_count = int(n)
            print(f"[live-movie] curve {cv.get('q')}: value left the axis, range -> [{lo:.4g}, {hi:.4g}]", flush=True)

    def _curves_update(self, tick, H=None):
        """Reveal each series up to the current RECORDED row -- the band is mean-SD .. mean+SD."""
        for i, cv in enumerate(getattr(self, "_curves", []) or []):
            S = cv["S"]
            # THE PANEL GROWS WITH THE RUN IT IS DRAWING, because the run can be longer than the
            # clip the panel was built for. The page builds its view with `n_frames: 1` -- it only
            # ever draws the seed -- and then sets `n_frames` to the run's length when RUN is
            # pressed; the series had already been allocated with two rows, so `t` clamped at 1 and
            # every one of 101 frames wrote the SAME row while the x axis stayed at [0, 1]. A run
            # of any length then drew as a two-point line pinned to frame 1. Extend to whichever is
            # longer, the tick or the declared length, and move the axis with it.
            if cv.get("live") and int(tick) >= S.shape[0]:
                _want = max(int(tick), int(self.n_frames)) + 1
                S = cv["S"] = np.concatenate(
                    [S, np.full((_want - S.shape[0],) + S.shape[1:], np.nan)], 0)
                cv["ch"].x_axis.range = [0.0, float(S.shape[0] - 1) * getattr(self, "_curve_xs", 1.0)]
            t = min(int(tick), S.shape[0] - 1)
            if cv.get("legend"):
                _on = int(tick) <= max(1.0, cv["legend_frac"] * float(S.shape[0] - 1))
                if bool(cv["ch"].legend_visible) != _on:
                    cv["ch"].legend_visible = _on
            if cv.get("live") and H is not None:
                # THE ROW FOR THIS FRAME, from the live level, in the units the axis was declared in.
                # BY NAME, NOT BY OBJECT: `engine.run` builds and seeds its OWN hierarchy, so a panel
                # holding the level of the seeded one read the same 200 cells for the whole run --
                # the curve sat flat while the tissue divided in the picture beside it.
                _lv = cv["lvl"]
                _nm = getattr(self, "_curve_set", None)
                if _nm and _nm in getattr(H, "levels", {}):
                    _lv = H.level(_nm)
                r = self._curve_row(H, _lv, cv["q"], cv["ntype"], cv["nt"])
                S[t] = r * cv["scale"] if cv["scale"] else r
                if cv.get("smooth", 0.0) > 0.0 and t >= 1 and np.all(np.isfinite(S[t - 1, :, 0])):
                    _w = 1.0 - math.exp(-float(self.stride) / cv["smooth"])   # per RENDERED frame
                    S[t, :, 0] = S[t - 1, :, 0] + _w * (S[t, :, 0] - S[t - 1, :, 0])
                if int(tick) in (48, 50, 498, 500) and str(cv["q"]).split(":")[0] in ("rate", "jacobian", "strain", "block"):
                    # WHAT THE LIVE PANEL HOLDS, once early and once late, so a panel that draws a flat
                    # line can be told apart from a quantity that is flat (exp_02 step 0038's live
                    # movie showed 0.6 where the after-run still showed 274 Hz).
                    print(f"[live-movie] curve {cv['q']} at tick {tick} (writer {id(self) % 10000}, out {os.path.basename(str(getattr(self, 'out', '?')))}, stride {self.stride}): "
                          f"row {np.round(r[:, 0], 5).tolist()} x scale {cv['scale']} -> {np.round(S[t, :, 0], 3).tolist()} (nt {cv['nt']})", flush=True)
                self._curve_follow(cv, S[t])
            if t < 1:
                continue
            x = np.arange(t + 1, dtype=float) * getattr(self, "_curve_xs", 1.0)
            for j in range(cv["nt"]):
                mu, sd = S[: t + 1, j, 0], S[: t + 1, j, 1]
                ok = np.isfinite(mu)
                if ok.sum() < 2:
                    continue
                if cv["sd"]:
                    cv["bands"][j].update(x[ok], (mu - sd)[ok], (mu + sd)[ok])
                if cv.get("nsurv") is None:                   # a Muller band draws no centre line
                    cv["lines"][j].update(x[ok], mu[ok])
            # THE NUMBER, ON THE PANEL. A curve gives the shape and leaves the reader squinting at
            # an axis for the value, which is the one thing a frame of a movie should hand over
            # directly. `cells` is a count and prints as an integer; a distribution prints as
            # `mean +- SD` to one decimal, because a second decimal on a quantity whose SD is shown
            # beside it is precision the picture does not have.
            lab = (self._curve_labels[i] if i < len(getattr(self, "_curve_labels", []) or [])
                   else None)
            if lab is not None:
                m0, s0 = S[t, :, 0], S[t, :, 1]
                if lab.get("multi"):                          # many series: the number is the panel's own quantity
                    m0, s0 = m0[:1], s0[:1]
                if cv.get("nsurv") is not None:              # a Muller plot: how many labels survive
                    txt = f"{int(cv['nsurv'][t])} clones"
                elif lab.get("with"):                         # `with:` -- two quantities, two numbers, never their sum
                    txt = "  |  ".join(f"{v:,.0f}" for v in m0)
                elif lab["q"] == "cells" or str(lab["q"]).startswith("count:"):   # a count prints as an integer
                    txt = f"{np.nansum(m0):,.0f}"
                elif lab["q"] == "phase":
                    txt = "  ".join(f"{v:.0f}" for v in m0) + " %"
                elif str(lab["q"]).startswith("species:"):     # one count per strain, as phase prints one % per stage
                    txt = "  ".join(f"{v:,.0f}" for v in m0)
                elif lab["q"] == "shape_index":
                    # THREE DECIMALS, because the whole scale of this quantity is narrow: a regular
                    # hexagon is 3.722 and a regular pentagon 3.812, so one decimal prints "3.8"
                    # for both and every change the panel exists to show vanishes.
                    txt = f"{np.nanmean(m0):.3f} \u00b1 {np.nanmean(s0):.3f}"
                else:
                    txt = f"{np.nanmean(m0):.1f} \u00b1 {np.nanmean(s0):.1f}"
                    if lab["unit"]:
                        txt += f" {lab['unit']}"
                self.p.add_text(txt, position=(lab["x"], lab["y"]), viewport=True,
                                font_size=lab["fs"], color=self._fg, name=lab["name"])

    # ---- the cloud AS a surface -----------------------------------------------------------
    #
    # THE METHOD IS `prototype/eye/render_surface_vtk.py`'s, and its docstring is the argument for
    # it: "render_orbit_vtk draws the material points themselves, which is honest and unreadable:
    # 45 000 dots make a speckled ball, and the six straps lose the shape the model gave them."
    #
    # WHAT COULD NOT BE IMPORTED, AND WHY IT IS REBUILT RATHER THAN CALLED. The eye binds an
    # AUTHORED Blender mesh -- a globe and six muscle straps an artist drew -- to the particles
    # seeded inside it. A slab of gel has no such mesh, and `render_surface_vtk` also imports
    # `eye_anatomy`, `blend_mpm_ops` and `render_eye` at module scope, so calling it would drag the
    # prototype into `src/plexus/`. `Skin` below is that class, kept line for line.
    #
    # THE REST SURFACE IS RECONSTRUCTED INSTEAD OF AUTHORED -- VTK's SurfaceReconstructionFilter
    # over a subsample of the frame-0 cloud -- and that is the ONLY substitution. Everything after
    # is the eye's: the mesh is built ONCE, at rest, and thereafter RIDDEN by the particles,
    #
    #     x_v(t) = sum_i w_i x_i(t),   sum_i w_i = 1,   w_i ~ 1/d_i^2 to the k nearest at rest,
    #
    # rather than re-extracted every frame. Re-extracting is what loses the crispness (the eye's
    # own stated reason for rejecting marching cubes), and skinning inherits the simulation's
    # rotations and stretches for free because the particles carry them. What it cannot show is
    # deformation FINER than the particle spacing -- the same limit the simulation has.
    #
    # The colour scalar rides the SAME weights, so a `color_field` appears ON the surface.
    class Skin:
        """A mesh bound to a set of moving particles: `deform(X)` returns its vertices."""

        def __init__(self, verts, rest_pts, k=8):
            from scipy.spatial import cKDTree
            k = int(min(k, len(rest_pts)))
            d, idx = cKDTree(rest_pts).query(np.asarray(verts, float), k=k)
            d = np.atleast_2d(d.T).T if k > 1 else d[:, None]
            idx = np.atleast_2d(idx.T).T if k > 1 else idx[:, None]
            w = 1.0 / np.maximum(d, 1e-9) ** 2
            self.w = (w / w.sum(axis=1, keepdims=True)).astype(np.float64)
            self.idx = idx.astype(np.int64)
            # the bind pose is where the particles put the vertex, so t = 0 renders the surface
            # exactly as it was reconstructed and every later frame is a pure displacement
            self.offset = np.asarray(verts, float) - self.deform(rest_pts)
            # THE OFFSET IS A VECTOR IN THE BODY, NOT IN THE WORLD. A vertex sits `offset` away
            # from the weighted centre of its k particles, and that vector was kept in world
            # axes: under a rigid rotation of the body it did not turn, so a rotating rotor's
            # surface breathed once per turn by up to twice the offset (~1-2 nm on a 5.5 nm rod
            # -- the "shape change" the human saw in builder/exp_02_bacterium step 0030 while
            # the particles themselves moved non-rigidly by 0.3 nm, step 0033). The k neighbours'
            # rest positions about their centre are kept here, and each frame the rotation that
            # best carries them onto their current positions (Kabsch, weighted, batched over the
            # vertices) turns the offset with the body. Exact for a rigid motion; for a deforming
            # tissue it is the local frame the old code assumed was the world's.
            R0 = np.asarray(rest_pts, float)[self.idx]                        # [V, k, 3]
            self.rest_local = R0 - np.einsum("vk,vkj->vj", self.w, R0)[:, None, :]

        def _rotations(self, X):
            """[V, 3, 3]: per vertex, the rotation carrying its neighbours' rest offsets onto now."""
            P = np.asarray(X, float)[self.idx]                                 # [V, k, 3]
            Q = P - np.einsum("vk,vkj->vj", self.w, P)[:, None, :]
            Hm = np.einsum("vk,vki,vkj->vij", self.w, self.rest_local, Q)     # weighted cross-covariance
            U, _, Vt = np.linalg.svd(Hm)
            R = np.einsum("vji,vkj->vik", Vt, U)                               # V U^T
            flip = np.linalg.det(R) < 0                                        # a reflection: negate V's last row
            if flip.any():
                Vt2 = Vt.copy(); Vt2[flip, 2, :] *= -1.0
                R = np.einsum("vji,vkj->vik", Vt2, U)
            return R

        def deform(self, X):
            return np.einsum("vk,vkj->vj", self.w, np.asarray(X, float)[self.idx])

        def __call__(self, X):
            if self.idx.shape[1] < 3:
                return self.deform(X) + self.offset
            return self.deform(X) + np.einsum("vij,vj->vi", self._rotations(X), self.offset)

        def scalar(self, values):
            """Per-vertex interpolation of a per-particle scalar, same inverse-square weights."""
            return np.einsum("vk,vk->v", self.w, np.asarray(values, float)[self.idx])

    def _env_texture(self, bright):
        """The environment map: morph.py's navy-teal sky, or `surface_env_color`'s.

        WHY THE SKY IS A KNOB. A translucent dielectric takes its reflected colour from the
        environment, so the amber honey and the orange droplet both read BROWN under the
        armadillo's blue sky -- the sky was chosen to make a blue glass sing and it tints
        everything else. `surface_env_color: white` (or any colour name / RGB triplet) builds the
        same equirectangular gradient -- darker at the horizon, the colour at the zenith, one soft
        bright sun patch -- in that hue instead, so a body's own colour comes back off it.
        Unset means morph's sky, unchanged, so the armadillo look is exactly what it was.
        """
        from plexus.morph import env_texture
        st = self.style or {}
        col = st.get("surface_env_color")
        if not col:
            return env_texture(bright)
        import pyvista as pv
        from matplotlib.colors import to_rgb
        c = np.asarray(to_rgb(tuple(col) if isinstance(col, (list, tuple)) else col), np.float64)
        Hh, W = 128, 256
        yy = np.linspace(0, 1, Hh)[:, None] * np.ones((1, W))
        xx = np.linspace(0, 1, W)[None, :] * np.ones((Hh, 1))
        sky = (0.35 + 0.65 * yy)[..., None] * c[None, None, :] * bright
        sun = np.exp(-(((xx - 0.7) * W) ** 2 + ((yy - 0.15) * Hh) ** 2) / (2 * 22 ** 2))
        sky = sky + sun[..., None] * np.array([0.6, 0.6, 0.6]) * bright
        tex = pv.Texture((np.clip(sky, 0, 1) * 255).astype(np.uint8))
        tex.SetMipmap(True); tex.SetInterpolate(True)
        return tex

    def _set_lights(self):
        """`plotting.light` -- which lights the scene has, when the spec says (`surface_env` brings
        its own four-light rig and wins):

            default    VTK's light kit (a key, a fill, a back light, all following the camera)
            headlight  one light at the camera; every face towards the viewer is lit the same
            sun        one directional light from `dot_light` (default upper left), hard shadows
            studio     key + fill + rim from fixed directions, the photographic three-point rig
            flat       no light at all: ambient only, every colour as declared
        """
        mode = str((self.style or {}).get("light", "") or "").lower()
        if not mode or mode == "default" or bool((self.style or {}).get("surface_env", False)):
            return
        import pyvista as pv
        self.p.remove_all_lights()
        if mode == "flat":
            return
        if mode == "headlight":
            self.p.add_light(pv.Light(light_type="headlight", intensity=1.0))
        elif mode == "sun":
            d = np.asarray((self.style or {}).get("dot_light", [-0.45, 0.8, 0.4]), np.float64)
            self.p.add_light(pv.Light(position=tuple(d * 10.0), focal_point=(0, 0, 0), light_type="scene light",
                                      intensity=1.1, positional=False))
        elif mode == "studio":
            for pos, inten in (((-1.0, 1.2, 1.0), 1.0), ((1.2, 0.3, 0.8), 0.45), ((0.2, 0.8, -1.2), 0.6)):
                self.p.add_light(pv.Light(position=tuple(np.asarray(pos) * 10.0), focal_point=(0, 0, 0),
                                          light_type="scene light", intensity=inten, positional=False))
        else:
            print(f"[live-movie] plotting.light: {mode!r} is not one of default, headlight, sun, studio, flat",
                  flush=True)
            return
        print(f"[live-movie] light: {mode}", flush=True)

    def _contour_settings(self):
        """morph.py's dielectric, with the spec allowed to override the few knobs that are a look.

        THE SETTINGS ARE THE ARMADILLO'S BY DEFAULT and are imported, not copied, so this render
        and the morphing script cannot drift: `default_settings()` is the blue glass -- opacity
        0.5, roughness 0.07, an index of refraction and a clear coat, a bright synthetic sky. A
        spec changes the COLOUR (`surface_color`) and may move opacity, roughness, metallic and the
        sky's brightness; the coat, IOR and lights are the recipe and stay.
        """
        from plexus.morph import default_settings
        s = default_settings()
        st = self.style or {}
        s["color"] = str(st.get("surface_color", s["color"]))
        s["opacity"] = float(st.get("surface_opacity", s["opacity"]))
        s["roughness"] = float(st.get("surface_roughness", s["roughness"]))
        s["metallic"] = float(st.get("surface_metallic", s["metallic"]))
        s["env_bright"] = float(st.get("surface_env_bright", s["env_bright"]))
        # THE TWO THAT MAKE A WARM COLOUR READ AS ITSELF. The armadillo's diffuse 0.42 / ambient
        # 0.06 is a dark saturated glass; an orange droplet under them at 60% opacity over black
        # comes out brown. A spec that wants its declared colour back raises these.
        s["diffuse"] = float(st.get("surface_diffuse", s["diffuse"]))
        s["ambient"] = float(st.get("surface_ambient", s["ambient"]))
        s["ngrid"] = int(st.get("contour_ngrid", s["ngrid"]))
        s["sigma"] = float(st.get("contour_sigma", s["sigma"]))
        s["iso_frac"] = float(st.get("contour_iso_frac", s["iso_frac"]))
        s["smooth_iter"] = int(st.get("contour_smooth", s["smooth_iter"]))
        return s

    def _contour_live(self, lvl):
        """Which DRAWN particles are alive: `lvl.occ` over the drawn index, or None if untracked.

        THE DORMANT POOL IS PARKED, NOT REMOVED. `_xyz` hands back every node the set was allocated
        -- an emitter's 11.3 M pool with 173 k live -- and hides the rest by position, at one
        corner, because the dot cloud's colours are bound to a fixed-length array. A density contour
        does not know that: 11 M particles in one voxel set the iso-level a thousand times above the
        6 mm thread, and the only surface it found was the parking spot. Measured on si_honey: the
        jet invisible, one speck in the corner. So the contour is built from live particles only.
        """
        occ = getattr(lvl, "occ", None)
        if occ is None or getattr(self, "idx", None) is None:
            return None
        try:
            m = np.asarray((occ[self.idx] > 0).detach().cpu().numpy() if hasattr(occ, "detach")
                           else np.asarray(occ)[np.asarray(self.idx)] > 0)
            return m if m.shape[0] == int(self.idx.numel()) else None
        except Exception:                        # noqa: BLE001
            return None

    def _contour_surface(self, pts):
        """The density contour of the live cloud, or None when there is nothing to contour."""
        from plexus.morph import reconstruct_contour
        s = self._contour_s
        X = np.asarray(pts, np.float64)
        if X.shape[0] < 8:
            return None
        box = float(np.max(self.world)) if getattr(self, "world", None) is not None \
            else float(np.max(X))
        return reconstruct_contour(X, box, ngrid=s["ngrid"], sigma=s["sigma"],
                                   iso_frac=s["iso_frac"], smooth_iter=s["smooth_iter"])

    def _contour_partition(self, H, lvl):
        """`contour_by_type: true` -- (names, type id per drawn particle, colour per name), or None.

        THE PARTITION IS THE ONE THE MODEL DECLARES, exactly as `render_3d: compartments` reads
        it: a particle's type is its own `node_type` if the set is typed, else its PARENT's. Two
        droplets of two types become two contours in two hues, and a droplet that splashes into
        the other stays its own colour through the splash, which is the picture a bicolour run is
        for. Computed once: a particle never changes parent.
        """
        st = self.style or {}
        if not bool(st.get("contour_by_type", False)):
            return None
        # A SURFACE PER BODY, WHEN THE SPEC SAYS WHICH BODY. Partitioned by TYPE, ten copies of one
        # type reconstructed as ONE surface in one colour -- the ring came out a single doughnut and
        # the ten rabbits one blob. When the scene colours by a scalar of its own (`color_field:
        # copy`), that scalar IS the partition, and the colours come from its colormap.
        _cf = str(st.get("color_field", "") or "")
        _sch = getattr(lvl, "state_schema", None)
        if _cf and _sch is not None and _cf in _sch and (_sch[_cf][1] - _sch[_cf][0]) == 1:
            import matplotlib.pyplot as _plt
            from matplotlib.colors import to_hex
            a, b = _sch[_cf]
            v = lvl.state[self.idx, a].detach().cpu().numpy()
            tid = np.rint(v).astype(int)
            ids = sorted(set(int(x) for x in np.unique(tid)))
            rng = st.get("color_range") or [min(ids), max(ids)]
            lo, hi = float(rng[0]), float(rng[1])
            cm = _plt.get_cmap(st.get("field_cmap", "turbo"))
            names = [f"{_cf} {i}" for i in ids]
            remap = {v_: k for k, v_ in enumerate(ids)}
            tid = np.vectorize(remap.get)(tid)
            cols = {f"{_cf} {i}": to_hex(cm((i - lo) / max(hi - lo, 1e-9))[:3]) for i in ids}
            return names, tid, cols
        own = getattr(lvl, "node_type", None); par = getattr(lvl, "parent", None)
        if own is not None:
            names = list(getattr(lvl, "type_names", []) or [])
            tid = np.asarray(own[self.idx].detach().cpu().numpy())
        elif getattr(lvl, "parent_name", None) and par is not None:
            plv = H.level(lvl.parent_name)
            names = list(getattr(plv, "type_names", []) or [])
            pnt = getattr(plv, "node_type", None)
            if pnt is None:
                return None
            tid = np.asarray(pnt.detach().cpu().numpy())[np.asarray(par[self.idx].detach().cpu().numpy())]
        else:
            return None
        if not names:
            return None
        from matplotlib.colors import to_rgb, to_hex
        pal = st.get("colors") or {}
        cols = {nm: (to_hex(to_rgb(tuple(pal[nm]) if isinstance(pal.get(nm), (list, tuple)) else pal[nm]))
                     if nm in pal else self._contour_s["color"]) for nm in names}
        return names, tid, cols

    def _contour_build(self, H, lvl, pos):
        """First frame: the material, the sky, the lights and the screen-space occlusion, once."""
        self._contour_s = None
        self._contour_part = None
        try:
            import pyvista as pv
            from plexus.morph import build_lights, env_texture
            s = self._contour_settings()
            self._contour_s = s
            self._contour_part = self._contour_partition(H, lvl)
            # `contour_bodies:` -- ONE GLASS BODY PER CELL (exp 21's sheet of MPM cells), keyed by an integer block
            # of the points (the cell id), each contoured on its own small grid, with an opaque nucleus inside.
            self._contour_bodies = (self.style or {}).get("contour_bodies") or None
            if self._contour_bodies:
                self._contour_part = None
            _live = self._contour_live(lvl)
            pos = np.asarray(pos)[_live] if _live is not None else np.asarray(pos)
            if self._contour_bodies:
                if not getattr(self, "_contour_lit", False):
                    self.p.set_environment_texture(self._env_texture(s["env_bright"]))
                    if not bool((self.style or {}).get("surface_env", False)):
                        build_lights(self.p, s)
                    try:
                        self.p.enable_depth_peeling(number_of_peels=int(s["peels"]))
                        box = float(np.max(self.world))
                        self.p.enable_ssao(radius=box * s["ssao_radius_frac"], bias=box * s["ssao_bias_frac"],
                                           kernel_size=int(s["ssao_kernel"]))
                    except Exception as _e:                          # noqa: BLE001
                        print(f"[live-movie] contour: ssao/peeling unavailable ({_e})", flush=True)
                    self._contour_lit = True
                n_b = self._contour_draw_bodies(H, lvl, pos, _live)
                print(f"[live-movie] contour bodies: {n_b} cells, each its own glass", flush=True)
                if n_b == 0:
                    self._contour_s = None
                    return False
                return True
            if self._contour_part is None:
                surf = self._contour_surface(pos)
                if surf is None or surf.n_points == 0:
                    print("[live-movie] contour: nothing to contour on the first frame; drawing dots",
                          flush=True)
                    self._contour_s = None
                    return False
            # THE SKY AND THE RIG, set once. `surface_env` may already have set them at plotter
            # construction; setting the texture twice is harmless, adding the lights twice is not.
            if not getattr(self, "_contour_lit", False):
                self.p.set_environment_texture(self._env_texture(s["env_bright"]))
                if not bool((self.style or {}).get("surface_env", False)):
                    build_lights(self.p, s)
                try:
                    self.p.enable_depth_peeling(number_of_peels=int(s["peels"]))
                    box = float(np.max(self.world))
                    self.p.enable_ssao(radius=box * s["ssao_radius_frac"],
                                       bias=box * s["ssao_bias_frac"],
                                       kernel_size=int(s["ssao_kernel"]))
                except Exception as _e:                              # noqa: BLE001
                    print(f"[live-movie] contour: ssao/peeling unavailable ({_e})", flush=True)
                self._contour_lit = True
            if self._contour_part is None:
                self._contour_draw(surf)
                print(f"[live-movie] contour: {surf.n_points:,} pts / {surf.n_cells:,} faces at "
                      f"{s['ngrid']}^3, colour {s['color']}, opacity {s['opacity']:g}", flush=True)
            else:
                n_ok = self._contour_draw_types(np.asarray(pos), _live)
                names = self._contour_part[0]
                print(f"[live-movie] contour by type: {n_ok} of {len(names)} surfaces "
                      f"({', '.join(names)}) at {s['ngrid']}^3, opacity {s['opacity']:g}", flush=True)
                if n_ok == 0:
                    self._contour_s = None
                    return False
            return True

        except Exception as e:                                       # noqa: BLE001
            print(f"[live-movie] contour unavailable ({type(e).__name__}: {e}); drawing dots",
                  flush=True)
            self._contour_s = None
            return False

    def _contour_draw_types(self, pts, live=None):
        """One contour per type, each in its own hue; returns how many had a surface.

        `pts` is already the LIVE subset when `live` is given; `tid` is over the full drawn index,
        so it is masked the same way here to stay aligned."""
        names, tid, cols = self._contour_part
        if live is not None:
            tid = tid[live]
        n_ok = 0
        for j, nm in enumerate(names):
            sel = tid == j
            if int(sel.sum()) < 8:
                continue
            surf = self._contour_surface(pts[sel])
            if surf is not None and surf.n_points > 0:
                self._contour_draw(surf, name=f"contour_{nm}", color=cols[nm]); n_ok += 1
        return n_ok

    def _contour_draw_bodies(self, H, lvl, pts, live=None):
        """`plotting.contour_bodies` -- every cell its own glass, the nucleus an opaque body inside it:

            contour_bodies:
              block: pid              the integer point block naming each point's cell (required)
              colors: [...]           the glass palette, one colour per cell by its id (default: seven blues)
              dx: 0.08                voxel side, world (default: 0.6 x the drawn points' median spacing)
              sigma: 0.9, iso_frac: 0.35, smooth: 20     the per-body contour (`body_contours`)
              nucleus: nuc            a width-1 point block > 0.5 on the nucleus's points (optional)
              color_by: {block: chem, channel: 0, cmap: Blues, range: [0, 1], floor: 0.25}
                                      each cell coloured by a value of its own parent-set block (a wave)
              nucleus_color: '#ffb347', nucleus_opacity: 1.0, nucleus_sigma: 1.2

        The glass is morph's dielectric (`_contour_settings`: opacity, roughness, coat, sky), coloured per cell."""
        cfg = self._contour_bodies if isinstance(self._contour_bodies, dict) else {"block": str(self._contour_bodies)}
        # THE CELL OF EACH POINT IS THE CONTAINMENT MAP (`parent`), which the engine's Level and the replay's level
        # both carry; a named integer block (`block:`) is read only when the set has no parent. The nucleus block is
        # read through `get`, which both serve -- `state_schema` is the live Level's alone, and a replay reading it
        # fell back to dots (smoke_exp21_sheet_base, 2026-10-08).
        idx = np.asarray(self.idx.detach().cpu().numpy() if hasattr(self.idx, "detach") else self.idx)

        def _block(name):
            try:
                v = lvl.get(name)
            except Exception:                                            # noqa: BLE001
                v = None
            if v is None:
                return None
            v = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v)
            return v.reshape(v.shape[0], -1)[idx, 0] if v.shape[0] > idx.max() else None
        par = getattr(lvl, "parent", None)
        if par is not None:
            lab = np.asarray(par.detach().cpu().numpy() if hasattr(par, "detach") else par).astype(np.int64)[idx]
        else:
            lab = _block(str(cfg.get("block", "pid")))
            if lab is None:
                if not getattr(self, "_bodies_warned", False):
                    self._bodies_warned = True
                    print(f"[live-movie] contour_bodies: the set has no parent and no block "
                          f"{cfg.get('block', 'pid')!r}", flush=True)
                return 0
            lab = np.rint(lab).astype(np.int64)
        nuc = None
        nb = cfg.get("nucleus")
        if nb:
            v = _block(str(nb))
            nuc = None if v is None else (v > 0.5)
        if live is not None:
            lab = lab[live]
            nuc = nuc[live] if nuc is not None else None
        pts = np.asarray(pts, np.float64)
        mode = str(cfg.get("mode", "partition"))
        if getattr(self, "_bodies_dx", None) is None:
            from scipy.spatial import cKDTree
            sub = pts[:: max(1, pts.shape[0] // 20000)]
            spacing = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))
            dx = cfg.get("dx")
            if dx is None:
                dx = (0.35 if mode == "partition" else 0.6) * spacing
            self._bodies_dx = float(dx)
            # 1.2 x THE SPACING: a voxel inside a body is never farther than ~0.7 spacings from a point, so a reach
            # under one spacing leaves holes (0.8 x read as foam, exp21_w_still_s1)
            self._bodies_reach = float(cfg.get("reach", 1.2 * spacing))
            print(f"[live-movie] contour_bodies: mode {mode}, voxel {self._bodies_dx:.3g}, reach "
                  f"{self._bodies_reach:.3g} (points {spacing:.3g} apart)", flush=True)
        s = self._contour_s
        # STATIC BODIES ARE CONTOURED ONCE: a frame whose points have not moved reuses the last surface and only its
        # colours change (the wave arm freezes the mechanics on its seconds-long clock)
        _same = getattr(self, "_bodies_last_pts", None)
        if _same is not None and _same.shape == pts.shape and np.array_equal(_same, pts) \
                and getattr(self, "_bodies_last_surf", None) is not None:
            surf, n_b = self._bodies_last_surf.copy(), self._bodies_last_n
        elif mode == "partition":
            # `clip_to_box: true` -- THE BODIES STAY INSIDE THE DRAWN BOX: the box frame is the run's walls, and a body
            # drawn `reach` past its outermost points crossed the frame lines and read as a cell outside its corral
            # (exp 21, steps 28 and 30, 2026-10-08)
            _clip = None
            if cfg.get("clip_to_box") and getattr(self, "world", None) is not None:
                _clip = (np.zeros(3), np.asarray(self.world, float)[:3])
            surf, n_b = body_partition(pts, lab, self._bodies_dx, self._bodies_reach,
                                       sigma=float(cfg.get("sigma", 0.8)), iso=float(cfg.get("iso", 0.55)),
                                       smooth_iter=int(cfg.get("smooth", 15)), clip=_clip,
                                       smooth_method=str(cfg.get("smooth_method", "laplace")),
                                       pass_band=float(cfg.get("pass_band", 0.05)))
        else:
            surf, n_b = body_contours(pts, lab, self._bodies_dx, sigma=float(cfg.get("sigma", 0.9)),
                                      iso_frac=float(cfg.get("iso_frac", 0.35)), smooth_iter=int(cfg.get("smooth", 20)))
        if surf is None:
            return 0
        self._bodies_last_pts, self._bodies_last_surf, self._bodies_last_n = pts.copy(), surf.copy(), n_b
        from matplotlib.colors import to_rgb
        pal = cfg.get("colors") or ["#1a4fc0", "#2a6fdb", "#3b8beb", "#1e5aa8", "#4a7fd0", "#2b9be0", "#3d5fc4"]
        pal = np.array([to_rgb(c) for c in pal])
        body = np.asarray(surf["body"])
        # NEIGHBOURS NEVER SHARE A COLOUR: a cell is given, once, the palette entry least used among the cells
        # touching it (centres within 1.5 x the median centre spacing), so every contact face separates two hues.
        # A hash gave half the cells a same-coloured neighbour, whose shared face then disappeared.
        cmap = getattr(self, "_body_colour", None)
        if cmap is None:
            cmap = self._body_colour = {}
        new_ids = [int(b) for b in np.unique(lab) if int(b) not in cmap]
        if new_ids:
            from scipy.spatial import cKDTree
            ids = np.unique(lab)
            cen = np.stack([pts[lab == b].mean(0) for b in ids])
            tree = cKDTree(cen)
            d0 = float(np.median(tree.query(cen, k=2)[0][:, 1])) if len(ids) > 1 else 1.0
            pos_of = {int(b): i for i, b in enumerate(ids)}
            for b in new_ids:
                nb = tree.query_ball_point(cen[pos_of[b]], 1.5 * d0)
                used = np.bincount([cmap[int(ids[j])] for j in nb if int(ids[j]) in cmap], minlength=len(pal))
                tot = np.bincount(list(cmap.values()), minlength=len(pal)) if cmap else np.zeros(len(pal), int)
                # the least used among its neighbours, ties to the least used overall: a hexagonal sheet needs only
                # three colours, and first-index ties drew it in the palette's three palest blues
                cmap[b] = int(np.argmin(used * 100000 + tot))
        h = np.array([cmap.get(int(b), 0) for b in body])
        surf["rgb"] = (pal[h] * 255).astype(np.uint8)
        # `color_by: {block, channel, cmap, range}` -- EACH CELL LIT BY A VALUE OF ITS OWN (its excitation, a
        # morphogen): the parent set's block, mapped through `cmap` over `range`, replaces the palette colour, so a
        # wave reads as glass lighting up cell by cell (exp 21's gap-junction wave)
        cb = cfg.get("color_by")
        if cb:
            try:
                plv = H.level(getattr(lvl, "parent_name", None) or cfg.get("cell_set", "cell"))
                v = plv.get(str(cb.get("block", "chem")))
                v = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, float)
                v = v.reshape(v.shape[0], -1)[:, int(cb.get("channel", 0))]
                lo, hi = (cb.get("range") or [float(np.nanmin(v)), float(np.nanmax(v)) + 1e-12])
                import matplotlib.pyplot as _plt
                cm = _plt.get_cmap(str(cb.get("cmap", "Blues")))
                x = np.clip((v[np.clip(body, 0, v.shape[0] - 1)] - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
                lo_c = float(cb.get("floor", 0.25))                      # the resting cell stays a visible glass
                surf["rgb"] = (np.asarray(cm(lo_c + (1.0 - lo_c) * x))[:, :3] * 255).astype(np.uint8)
            except Exception as _e:                                      # noqa: BLE001
                if not getattr(self, "_cb_warned", False):
                    self._cb_warned = True
                    print(f"[live-movie] contour_bodies.color_by unavailable ({type(_e).__name__}: {_e})", flush=True)
        # `coat: 0` -- NO CLEAR COAT: the dielectric's second, sharp highlight read as streaks and lines across a
        # sheet of opaque cells (the human, exp 21 run 30); with `surface_opacity: 1` and a satin roughness the cells
        # are soft opaque bodies
        coat = float(cfg.get("coat", s["coat_strength"]))
        act = self.p.add_mesh(surf, name="contour_bodies", scalars="rgb", rgb=True, pbr=True,
                              metallic=s["metallic"], roughness=s["roughness"], opacity=s["opacity"],
                              diffuse=s["diffuse"], ambient=s["ambient"], smooth_shading=True, show_scalar_bar=False)
        try:
            prop = act.GetProperty()
            prop.SetBaseIOR(s["ior"]); prop.SetCoatStrength(coat)
            prop.SetCoatRoughness(s["coat_roughness"]); prop.SetCoatIOR(s["coat_ior"])
        except Exception:                                                # noqa: BLE001
            pass
        if nuc is not None and nuc.any():
            ns, _n = body_contours(pts[nuc], lab[nuc], self._bodies_dx, sigma=float(cfg.get("nucleus_sigma", 1.2)),
                                   iso_frac=float(cfg.get("iso_frac", 0.35)), smooth_iter=int(cfg.get("smooth", 20)),
                                   min_points=4)
            if ns is not None:
                self.p.add_mesh(ns, name="contour_nuclei", color=str(cfg.get("nucleus_color", "#ffb347")),
                                pbr=True, metallic=0.0, roughness=0.35, opacity=float(cfg.get("nucleus_opacity", 1.0)),
                                diffuse=0.8, ambient=0.15, smooth_shading=True, show_scalar_bar=False)
        return n_b

    def _contour_draw(self, surf, name="contour", color=None):
        """Replace the contour actor -- same name, so pyvista swaps rather than stacks."""
        s = self._contour_s
        # `surface_pbr: false` -- A MATTE SURFACE: plain Lambert shading, no specular, no coat, no
        # index of refraction. The dielectric below always keeps a highlight however rough it is
        # set (the coat is a second, sharp layer), so "flat" needs the plain material.
        if (self.style or {}).get("surface_pbr", True) is False:
            # diffuse 0.55 + ambient 0.2 under VTK's three-light kit: at diffuse 1 the pastel
            # colours of a tab20 palette blew out to white.
            self.p.add_mesh(surf, name=name, color=(color or s["color"]), pbr=False, specular=0.0,
                            diffuse=0.55, ambient=0.2, opacity=s["opacity"], smooth_shading=True,
                            show_scalar_bar=False)
            return
        act = self.p.add_mesh(surf, name=name, color=(color or s["color"]), pbr=True,
                              metallic=s["metallic"], roughness=s["roughness"],
                              opacity=s["opacity"], diffuse=s["diffuse"], ambient=s["ambient"],
                              smooth_shading=True, show_scalar_bar=False)
        # THE DIELECTRIC ITSELF, as morph.draw_scene sets it: an index of refraction and a thin,
        # sharp clear coat over the base layer -- the glassy double highlight rather than the
        # single soft plastic one.
        try:
            prop = act.GetProperty()
            prop.SetBaseIOR(s["ior"]); prop.SetCoatStrength(s["coat_strength"])
            prop.SetCoatRoughness(s["coat_roughness"]); prop.SetCoatIOR(s["coat_ior"])
        except Exception:                                            # noqa: BLE001
            pass

    def _contour_update(self, H, lvl, pts):
        if getattr(self, "_contour_s", None) is None:
            return
        try:
            _live = self._contour_live(lvl)
            pts = np.asarray(pts)[_live] if _live is not None else np.asarray(pts)
            if getattr(self, "_contour_bodies", None):
                self._contour_draw_bodies(H, lvl, pts, _live)
            elif getattr(self, "_contour_part", None) is not None:
                self._contour_draw_types(pts, _live)
            else:
                surf = self._contour_surface(pts)
                if surf is not None and surf.n_points > 0:
                    self._contour_draw(surf)
        except Exception as e:                                       # noqa: BLE001
            if not getattr(self, "_contour_warned", False):
                self._contour_warned = True
                print(f"[live-movie] contour update failed ({type(e).__name__}: {e})", flush=True)

    def _skin_build(self, H, lvl, pos):
        """Reconstruct the rest surface and bind it. Returns True when the surface is live."""
        self._skin = self._surf = self._skin_sub = None
        try:
            st = self.style or {}
            X = np.asarray(pos, dtype=np.float64)
            # A SUBSAMPLE FOR THE RECONSTRUCTION, THE FULL CLOUD FOR THE SKIN. The filter is
            # O(N log N) with a large constant and 500,000 points is minutes; 60,000 resolves a
            # feature the grid can resolve anyway, since `sample_spacing` defaults to the grid's
            # own dx and the simulation cannot represent anything finer.
            nsub = int(st.get("surface_sample", 60_000))
            step = max(1, X.shape[0] // max(nsub, 1))
            sub = np.arange(0, X.shape[0], step)
            self._skin_sub = sub
            # THE REST SURFACE IS AN ISOSURFACE OF THE PARTICLE DENSITY, and this is the one
            # place the eye's recipe had to be replaced rather than copied.
            #
            # `reconstruct_surface` (VTK's SurfaceReconstructionFilter) fits an implicit function
            # from LOCAL TANGENT PLANES, which is what a laser scan of a SURFACE gives it. An MPM
            # cloud is a SOLID: 500,000 points fill the interior, where there is no tangent plane
            # and the signed distance it estimates is noise. Measured on this slab it returned a
            # lace of spikes and holes at every spacing tried -- 110,981 faces of foam around a box.
            #
            # Counting the particles into cells and contouring at half the bulk density is the
            # standard answer for a filled cloud, and the eye's objection to marching cubes does not
            # reach it: that objection is to RE-EXTRACTING every frame, which loses crispness and
            # temporal coherence. This runs ONCE, at rest, and the skinning below carries it -- so
            # the surface is still ridden by the particles, not rebuilt from them.
            #
            # AT THE SIMULATION'S OWN RESOLUTION. The cell is the MPM grid's dx, so the surface can
            # show exactly what the solver can represent and no more -- and at ~10 particles per
            # occupied cell the count is a density rather than a speckle.
            from scipy.ndimage import gaussian_filter
            h = float(st.get("surface_spacing", 0.0)) or self._cs_dx
            # THE PAD COVERS THE BLUR'S HALO. Three cells of padding with a blur of 1.5 cells left the
            # smoothed density above the contour level ON THE GRID'S FACES, so marching cubes cut the
            # shell open there: a window through the hook and the bushing on the -x/-y side, fixed in
            # the world while the parts turned (exp_02 steps 0034-0049; measured: 240 open edges on the
            # hook's surface, density 0.51 on the faces against a level of 0.22). Pad by three cells
            # plus three sigma of the blur and the shell closes.
            _pad = (3.0 + 3.0 * float(st.get('surface_blur', 1.0))) * h
            lo = X.min(0) - _pad
            dim = np.maximum(np.ceil((X.max(0) + _pad - lo) / h).astype(int) + 1, 2)
            ijk = np.clip(((X - lo) / h).astype(np.int64), 0, dim - 1)
            D = np.zeros(tuple(dim), np.float32)
            np.add.at(D, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
            _held = D > 0                                   # the cells that HOLD particles
            D = gaussian_filter(D, sigma=float(st.get("surface_blur", 1.0)))
            # THE BULK IS THE DENSITY WHERE THE PARTICLES ARE, NOT WHERE THE BLUR REACHES. The median
            # used to be taken over every cell the blur had touched (`D > 0` AFTER the filter), and
            # with a 1.5-cell blur those halo cells outnumber the body's own: the "bulk" came out at
            # 0.01-0.03 particles per cell against a true 4-7, the level 0.3 x that sat far out in
            # the halo's tail, and EVERY skin was drawn 5-6 nm (5 cells) fatter than its particles
            # in each direction -- a 5.5 nm rod at 12 nm, FliG's plate swallowing the membrane
            # sheet 2 nm above it (exp_02 step 0073, measured offline on its trajectory). Over the
            # cells that held particles the median is the bulk, and the same level leaves a skin
            # one cell (1-2 nm) outside its particles, which is the blur's own width.
            occ = D[_held]
            # `surface_iso_frac`: the contour as a FRACTION of the median occupied cell (0.5 unless
            # said). Lower closes the pinholes a sparse patch of points leaves in the shell -- the
            # dark spots on the scaffold's PDZ blobs and the window in the bushing of exp_02 steps
            # 0034-0043 -- at the price of a slightly fatter surface; `surface_iso` still sets an
            # absolute value when given.
            iso = float(st.get("surface_iso", 0.0)) or float(st.get("surface_iso_frac", 0.5)) * float(np.median(occ))
            g = self.pv.ImageData(dimensions=tuple(int(v) for v in dim),
                                  spacing=(h, h, h), origin=tuple(float(v) for v in lo))
            g.point_data["d"] = D.ravel(order="F")
            surf = g.contour([iso], scalars="d")
            # TAUBIN, NOT LAPLACIAN. Laplacian smoothing shrinks a closed surface toward its
            # centroid, and the thickness of this slab is the measurement. Taubin alternates a
            # shrink and an expand and holds the volume.
            surf = surf.extract_largest().smooth_taubin(
                n_iter=int(st.get("surface_smooth", 30)), pass_band=0.08)
            self._skin = self.Skin(surf.points, X[sub], k=int(st.get("surface_k", 8)))
            self._surf = surf
            # THE SCALAR MUST EXIST BEFORE `add_mesh`, not after the first update: pyvista
            # resolves `scalars=` at add time and raises "Data array (f) not present in this
            # dataset". The whole surface then fell back to dots -- with the reason printed, which
            # is the only thing that made it a five-minute bug instead of a silent one.
            fld = str(st.get("color_field", "") or "")
            clim = None
            if fld:
                val = self._field(H, lvl)[0]
                if val is not None:
                    v = val.detach().cpu().numpy().astype(np.float64)[sub]
                    surf["f"] = self._skin.scalar(v)
                    rng = st.get("color_range")
                    clim = ([float(rng[0]), float(rng[1])] if rng and len(rng) == 2
                            else [float(np.percentile(v, 2)), float(np.percentile(v, 98))])
                else:
                    fld = ""
            self.p.add_mesh(surf, scalars=("f" if fld else None), clim=clim,
                            cmap=st.get("field_cmap", "turbo"),
                            color=(None if fld else st.get("surface_color", "#cfd8e3")),
                            opacity=float(st.get("surface_opacity", 1.0)),
                            smooth_shading=True, specular=0.25, specular_power=18,
                            ambient=0.25, diffuse=0.75, show_scalar_bar=False)
            _past = float(max((surf.points.max(0) - X.max(0)).max(), (X.min(0) - surf.points.min(0)).max()) / h)
            print(f"[live-movie] surface: density isosurface of {X.shape[0]:,} points at cell "
                  f"{h:.4g} ({'x'.join(str(int(v)) for v in dim)}), iso {iso:.3g} of a bulk "
                  f"{float(np.median(occ)):.3g} -> {surf.n_points:,} vertices, "
                  f"{surf.n_faces_strict:,} faces, skinned to {self._skin.idx.shape[1]} "
                  f"particles each over a {len(sub):,}-point bind set; the skin reaches "
                  f"{_past:.1f} cells past its particles", flush=True)
            return True
        except Exception as e:                       # noqa: BLE001 -- fall back to dots, never die
            self._skin = self._surf = None
            print(f"[live-movie] surface reconstruction failed ({type(e).__name__}: {e}); "
                  f"drawing the point cloud instead", flush=True)
            return False

    # ---- ONE SURFACE PER COMPARTMENT, WITH TRANSPARENCY -----------------------------------
    #
    # `render_3d: compartments`. The single-surface path above skins the WHOLE cloud, which for a
    # composed cell reconstructs one blob: the atlas's nine organelles share a set, so the density
    # isosurface of all of them together is the outside of the plasma membrane and nothing else.
    # What the reference picture actually shows is nine SEPARATE surfaces, the outer ones
    # translucent so the inner ones read through them -- which is why a cut-away was needed at all
    # to see inside a dot cloud, and is not needed here.
    #
    # THE PARTITION IS THE ONE THE MODEL ALREADY DECLARES: the particles' parent's `node_type`, so
    # the surfaces are the compartments of the specification and not a clustering of the point
    # cloud. Each gets `plotting.colors[<type>]` for its hue and `plotting.opacity[<type>]` for its
    # transparency, defaulting to `surface_opacity`.
    #
    # DRAWN BACK TO FRONT IS NOT NEEDED. VTK's depth peeling resolves the ordering, and it is
    # enabled here rather than left to the caller because without it a translucent membrane in
    # front of a translucent nucleus composites in draw order and the nucleus disappears at some
    # camera angles and not others -- an intermittent picture, which is the worst kind.
    def _skins_build(self, H, lvl, pos):
        """Build one bound surface per parent type. Returns True when at least one is live."""
        self._skins = []
        try:
            import numpy as np
            st = self.style or {}
            # A COMPARTMENT MAY BE A SET, NOT A TYPE, and once each organelle owns its own
            # particle set that is the only reading that works. This renderer draws ONE level --
            # the largest that carries positions -- so a model split into fifteen node sets came
            # out as a picture of whichever of them happened to be biggest, with the other
            # fourteen organelles simply absent and nothing saying so (measured: "compartment
            # surfaces: 1 of 1 reconstructed", on a cell with fifteen).
            #
            # `plotting.compartment_sets` names them, and each group then carries its OWN set
            # rather than a slice of one. Everything after this branch is shared: the same
            # isosurface, the same skinning, the same per-compartment material.
            csets = [n for n in (st.get("compartment_sets") or []) if n in H.levels]
            if csets:
                names, tid = csets, None
            else:
                pname = getattr(lvl, "parent_name", None)
                own = getattr(lvl, "node_type", None)
                par = getattr(lvl, "parent", None)
                if own is not None:
                    names = list(getattr(lvl, "type_names", []) or [])
                    tid = own[self.idx].detach().cpu().numpy()
                elif pname and par is not None:
                    plv = H.level(pname)
                    names = list(getattr(plv, "type_names", []) or [])
                    pnt = getattr(plv, "node_type", None)
                    if pnt is None:
                        raise ValueError(f"{pname!r} carries no node_type to partition by")
                    tid = pnt.detach().cpu().numpy()[par[self.idx].detach().cpu().numpy()]
                else:
                    raise ValueError("no type partition: no types and no typed parent")
                if not names:
                    raise ValueError("the typed set declares no `types:`; no compartments")
            from matplotlib.colors import to_rgb
            pal = st.get("colors") or {}
            opa = st.get("opacity") or {}
            _dflt_op = float(st.get("surface_opacity", 1.0))
            X = np.asarray(pos, dtype=np.float64)
            nsub = int(st.get("surface_sample", 60_000))
            self.p.enable_depth_peeling(number_of_peels=int(st.get("depth_peels", 12)),
                                        occlusion_ratio=0.0)
            _per = st.get("surface") or {}
            for j, nm in enumerate(names):
                if tid is None:                       # a SET per compartment
                    _lv = H.level(nm)
                    Xg = np.asarray(_lv.get("pos").detach().cpu().numpy(), np.float64)
                    # LIVE POINTS ONLY. A dormant point -- a monomer pool waiting to be
                    # polymerised -- has no mass, no stress and no part in the cell, but it has a
                    # position, and drawing it puts a cloud of the pool's own colour wherever it
                    # was parked: measured, a green haze 40 um above a cell whose cytoskeleton is
                    # green. `occ` is what says which rows are the compartment.
                    _oc = getattr(_lv, "occ", None)
                    sel = (np.nonzero(np.asarray(_oc.detach().cpu().numpy()) > 0)[0]
                           if _oc is not None else np.arange(Xg.shape[0]))
                else:                                  # a TYPE per compartment, sliced out of one
                    Xg = X
                    sel = np.nonzero(tid == j)[0]
                # A COMPARTMENT TOO SPARSE TO CONTOUR IS DRAWN AS DOTS, NOT DROPPED. A 50-point
                # membrane patch has no density field worth contouring at the grid's dx, and a
                # missing plasma membrane is a picture of a cell that has none.
                #
                # SO IS ONE TOO THIN TO CONTOUR, and that is a separate case with the same answer.
                # A cytoskeletal filament is 60 nm across in a 20 um cell; resolving it as an
                # isosurface needs a voxel of ~30 nm, and the filaments span the whole cytoplasm,
                # so the grid would be 750^3 = 422 M cells FOR ONE COMPARTMENT. Contoured at any
                # affordable spacing it comes back as a chain of beads -- which is exactly what the
                # first render showed, and reads as a defect in the model rather than in the
                # renderer. `render: dots` on that compartment draws its points instead: a hairline
                # is what a thin filament looks like, and it costs no grid at all.
                if str((_per.get(nm) or {}).get("render", "")).lower() == "dots" \
                        or sel.size < int(st.get("surface_min_points", 400)):
                    self._skins.append(self._dots_for(Xg, sel, nm, pal, opa, _dflt_op,
                                                      (_per.get(nm) or {}).get("point_size"),
                                                      every=int((_per.get(nm) or {}).get("every", 1)),
                                                      set_name=(nm if tid is None else None)))
                    continue
                step = max(1, sel.size // max(nsub, 1))
                sub = sel[::step]
                surf = self._isosurface(Xg[sel], st, per_compartment=nm)
                if surf is None or surf.n_points == 0:
                    self._skins.append(self._dots_for(Xg, sel, nm, pal, opa, _dflt_op,
                                                      set_name=(nm if tid is None else None)))
                    continue
                # THE SOUNDNESS TEST OF A SHELL, printed, not eyeballed: a compartment's surface must be
                # CLOSED (no boundary edges). An open shell is a window the viewer sees through and
                # takes for a hole in the material (exp_02 steps 0034-0049: the hook and the bushing
                # cut open where the density touched the contouring grid's faces). Counted here on
                # every build; a non-zero count is said out loud with the set's name.
                try:
                    _open = int(surf.n_open_edges)
                except Exception:                                    # noqa: BLE001
                    _open = -1
                self._open_edges = getattr(self, "_open_edges", {}); self._open_edges[nm] = _open
                # HOW FAR THE SKIN REACHES PAST ITS PARTICLES, the other soundness test: a shell
                # drawn fatter than the body it wraps swallows whatever sits next to it (the membrane
                # sheet 2 nm above FliG, exp_02 step 0073). World units; summarised below.
                try:
                    _Xs = Xg[sel]
                    _reach = float(max((surf.points.max(0) - _Xs.max(0)).max(), (_Xs.min(0) - surf.points.min(0)).max()))
                except Exception:                                    # noqa: BLE001
                    _reach = float("nan")
                self._skin_reach = getattr(self, "_skin_reach", {}); self._skin_reach[nm] = _reach
                if _open > 0:
                    print(f"[live-movie] compartment {nm!r}: surface NOT closed, {_open} open edges -- a window, "
                          f"not a hole in the material (raise surface padding/blur)", flush=True)
                skin = self.Skin(surf.points, Xg[sub], k=int(st.get("surface_k", 8)))
                col = to_rgb(tuple(pal[nm])) if nm in pal else (0.8, 0.8, 0.8)
                # THE MATERIAL, NOT JUST THE ALPHA -- and it is the material that decides whether a
                # membrane reads as GLASS or as frosted plastic. These four numbers were fixed at
                # `ambient 0.22, diffuse 0.78, specular 0.3`, which is a matte plastic: ambient
                # light is emitted regardless of angle, so a closed shell contributes 0.22 of its
                # colour at EVERY pixel and does it twice (the near wall and the far one). At
                # alpha 0.04 that is still a visible milky wash, which is why dropping the opacity
                # alone stopped helping -- measured on the first contact sheet, where the 0.04 row
                # was barely clearer than the 0.08 one.
                #
                # Water is the opposite: almost no ambient, little diffuse, a hard specular
                # highlight. `backface_culling` completes it by drawing only the wall facing the
                # camera, which halves the accumulated alpha AND is what makes the inside visible
                # -- you look through the near surface at the organelles, not at the far surface
                # through the near one.
                _m = (_per.get(nm) or {})
                # `surface_pbr: true` -- A PHYSICALLY-BASED MATERIAL: `metallic` and `roughness`
                # instead of the Phong specular/ambient/diffuse below, which VTK ignores under PBR.
                # Pair it with `surface_env`, or the metal has nothing to reflect.
                _pbr = {}
                if bool(_m.get("pbr", st.get("surface_pbr", False))):
                    _pbr = dict(pbr=True,
                                metallic=float(_m.get("metallic", st.get("surface_metallic", 0.3))),
                                roughness=float(_m.get("roughness", st.get("surface_roughness", 0.25))))
                _cb = self._block_cfg(nm) if tid is None else None
                _col_kw = {"color": col, "show_scalar_bar": False}
                if _cb is not None and getattr(self, "_H_now", None) is not None:
                    _v, _lo, _hi, _cmap, _label = self._block_values(self._H_now, nm, _cb)
                    surf["v"] = skin.scalar(_v[sub]).astype(np.float32)
                    _leg = self._legend_on(_cb)
                    _col_kw = {"scalars": "v", "cmap": _cmap, "clim": [_lo, _hi], "show_scalar_bar": _leg,
                               **({"scalar_bar_args": self._legend_args(_label)} if _leg else {})}
                _act = self.p.add_mesh(surf, **_col_kw,
                                opacity=float(opa.get(nm, _dflt_op)),
                                smooth_shading=True, **_pbr,
                                specular=float(_m.get("specular", st.get("surface_specular", 0.3))),
                                # CLAMPED, because VTK's range is (0, 128] and pyvista RAISES
                                # outside it -- which this class swallows, so a spec asking for a
                                # harder highlight silently lost its whole surface and fell back to
                                # a point cloud. Measured: `specular_power: 160` on the membrane
                                # dropped all six reconstructed compartments in one panel.
                                specular_power=min(128.0, max(1e-3, float(
                                    _m.get("specular_power",
                                           st.get("surface_specular_power", 24))))),
                                ambient=float(_m.get("ambient", st.get("surface_ambient", 0.22))),
                                diffuse=float(_m.get("diffuse", st.get("surface_diffuse", 0.78))),
                                **self._silhouette())
                # BACKFACE CULLING IS AN ACTOR PROPERTY, not an `add_mesh` keyword -- pyvista's
                # signature has no such argument in this version, so passing it raised and the
                # whole compartment fell back to dots. Set on the actor, where VTK keeps it.
                if bool(_m.get("backface_culling",
                               st.get("surface_backface_culling", False))):
                    try:
                        _act.prop.backface_culling = True
                    except Exception:                            # noqa: BLE001
                        _act.GetProperty().BackfaceCullingOn()
                # `surface.<set>.near_side` -- ONE SURFACE CUT, THE REST WHOLE: a membrane slab opened in
                # front of the channel standing in it, while the protein keeps its closed skins.
                _ns = _m.get("near_side", (self.style or {}).get("near_side"))
                if _ns and isinstance(_ns, str) and _ns.lower() == "far" and bool(_m.get("recontour", False)):
                    surf.copy_from(self._clip_far(surf))
                elif _ns:
                    self._near_side_faces(surf, _ns)
                self._skins.append({"kind": "surface", "surf": surf, "skin": skin, "sub": sub, "cb": _cb, "set": nm if tid is None else None, "actor": _act,
                                    "near_side": _ns,
                                    "name": nm, "n": int(sel.size), "set": (nm if tid is None
                                                                            else None),
                                    "faces": int(surf.n_faces_strict),
                                    # `surface.<set>.recontour: true` -- see _skins_update
                                    "recontour": bool(_m.get("recontour", False)), "sel": sel})
            _ns = sum(1 for s in self._skins if s and s["kind"] == "surface")
            _oe = getattr(self, "_open_edges", {}) or {}
            _closed = sum(1 for v in _oe.values() if v == 0)
            print(f"[live-movie] shells closed: {_closed} of {len(_oe)}"
                  + (f" (open: {', '.join(f'{k} {v}' for k, v in _oe.items() if v)})" if any(_oe.values()) else ""), flush=True)
            _rc = getattr(self, "_skin_reach", {}) or {}
            if _rc:
                _um = float(getattr(self, "length_um", 0.0) or 0.0)
                _fmt = (lambda v: f"{v * _um * 1e3:.1f} nm") if _um > 0 else (lambda v: f"{v:.4f}")
                _worst = max(_rc, key=lambda k: _rc[k])
                print(f"[live-movie] skins reach past their particles by at most {_fmt(_rc[_worst])} ({_worst}); "
                      + ", ".join(f"{k} {_fmt(v)}" for k, v in _rc.items()), flush=True)
            print(f"[live-movie] compartment surfaces: {_ns} of {len(names)} reconstructed "
                  f"(" + ", ".join(f"{s['name']} {s.get('faces', 0):,}f" for s in self._skins
                                   if s and s['kind'] == 'surface') + "); the rest drawn as dots",
                  flush=True)
            return bool(self._skins)
        except Exception as e:                       # noqa: BLE001 -- fall back to dots, never die
            self._skins = []
            print(f"[live-movie] compartment surfaces failed ({type(e).__name__}: {e}); "
                  f"drawing the point cloud instead", flush=True)
            return False

    def _dots_for(self, X, sel, nm, pal, opa, dflt_op, point_size=None, set_name=None, every=1):
        """A compartment kept as points: its own PolyData, its own hue, its own opacity.

        `every: k` DRAWS ONE POINT IN k, and it is what makes CO-LOCATED species legible. Five
        protein species are five balls of the cell's own radius -- a species is a concentration,
        not a place -- so drawn translucent they BLEND (yellow over blue over pink over violet over
        green averages to grey: measured, zero saturated pixels) and drawn opaque the last one
        simply covers the other four (measured, one hue). Neither is a palette bug and neither is
        fixed by choosing better colours. Thinning each species is what lets all five show through,
        and it changes no number in the model -- only how many of its points are drawn.
        """
        import numpy as np
        from matplotlib.colors import to_rgb
        if sel.size == 0:
            return None
        if int(every) > 1:
            sel = sel[:: int(every)]
        pd = self.pv.PolyData(np.asarray(X[sel], np.float32))
        col = to_rgb(tuple(pal[nm])) if nm in pal else (0.8, 0.8, 0.8)
        _ps = float(point_size if point_size is not None
                    else (self.style or {}).get("dot_size", 2.0))
        # `dot_shading: true` LIGHTS THESE SPRITES TOO. The subject cloud already honours it (see
        # the FLAT block); a compartment's dots were always unlit, so a scene made of six
        # compartment sets -- the six proteins of the C. jejuni scaffold, PDB 9HMF, one point per
        # alpha carbon (builder/exp_02_bacterium step 0012) -- came out as flat confetti while the
        # spheres beside it were shaded. Same keys, same defaults, except `dot_specular` defaults
        # to 0 here: a protein drawn from a cryo-EM model is matte in every figure it comes from.
        cb = self._block_cfg(set_name) if set_name else None
        if cb is not None and getattr(self, "_H_now", None) is not None:
            v, lo, hi, cmap, label = self._block_values(self._H_now, set_name, cb)
            pd["v"] = v[sel].astype(np.float32)
            _leg = self._legend_on(cb)
            _cb_actor = self.p.add_mesh(pd, scalars="v", cmap=cmap, clim=[lo, hi], opacity=float(opa.get(nm, dflt_op)),
                            render_points_as_spheres=True, point_size=_ps, show_scalar_bar=_leg,
                            **({"scalar_bar_args": self._legend_args(label)} if _leg else {}),
                            **self._dot_light())
        else:
            _cb_actor = None
            cb = None
            self.p.add_mesh(pd, color=col, opacity=float(opa.get(nm, dflt_op)),
                            render_points_as_spheres=True, point_size=_ps,
                            show_scalar_bar=False, **self._dot_light())
        return {"kind": "dots", "surf": pd, "sub": sel, "name": nm, "n": int(sel.size),
                "set": set_name, "cb": cb, "actor": _cb_actor}

    # ---- `plotting.color_block`: COLOUR A SET BY ONE OF ITS STATE BLOCKS ------------------------
    #
    #     color_block:
    #       membrane: {block: psi, unit: voltage, range: [120, 160], cmap: viridis}
    #       stator_unit: {block: x, range: [0, 0.25], cmap: plasma}
    #
    # A compartment drawn as dots or as a surface, and a `spheres` set, take a per-element scalar
    # from the named block, converted through the run's declaration (`unit:`) so the range and the
    # legend read in mV or in radians, and drawn through a colormap. Written for the motor's ion
    # economy (builder/exp_02_bacterium, 2026-09-23): the membrane's potential as a distribution
    # over its material, a stator's stretch on its ball. `color_field` colours the SUBJECT cloud by
    # a solver field; this colours ANY set by its own state, which `color_field` never could.
    def _legend_on(self, cfg):
        """Whether a block colouring draws its colour bar: `color_block.<set>.legend`, else
        `plotting.legend`, else yes. The human asked for the bar gone from the motor's movies
        (2026-09-24): the range is in the record's why and the picture reads without it."""
        st = self.style or {}
        return bool((cfg or {}).get("legend", st.get("legend", True)))

    def _legend_args(self, label):
        """The colour legend of a block: SMALL, vertical, bottom-left, so it never competes with the
        picture (the first one spanned the frame's width under the motor and the human said so)."""
        return {"title": label, "color": self._fg, "n_labels": 3, "fmt": "%.3g", "vertical": True,
                "width": 0.035, "height": 0.22, "position_x": 0.015, "position_y": 0.06,
                "title_font_size": 11, "label_font_size": 10, "shadow": False}

    def _block_cfg(self, set_name):
        cb = (self.style or {}).get("color_block") or {}
        c = cb.get(set_name) if isinstance(cb, dict) else None
        return dict(c) if isinstance(c, dict) and c.get("block") else None

    def _block_values(self, H, set_name, cfg):
        """[n] physical values of `cfg['block']` on `set_name` (column `col`, default 0), and
        (lo, hi, cmap, label). The range settles on the first frame's 2nd-98th percentile when
        the spec gives none, and is then FIXED, so a colour means one number for the whole movie."""
        import numpy as np
        lv = H.level(set_name)
        col = int(cfg.get("col", 0))
        if hasattr(lv, "state_schema"):                                   # a live Level
            a, b = lv.state_schema[str(cfg["block"])]
            v = lv.state[..., a + col].detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:                                                             # a replay level: the block at its frame
            g = lv.get(str(cfg["block"]))
            g = g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)
            v = np.asarray(g, np.float64).reshape(g.shape[0], -1)[:, col]
        unit = cfg.get("unit")
        f = _units_to_physical(1.0, unit, self._units) if unit else None
        if f is not None:
            v = v * float(f)
        if bool(cfg.get("abs", False)):                                   # the magnitude: a current that
            v = np.abs(v)                                                 # flickers in sign still lights its spot
        key = f"_cb_range_{set_name}"
        rng = cfg.get("range")
        if rng and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        else:
            # THE RANGE COMES FROM THE DATA, NEVER FROM A GUESS (the human's rule). On a replay the
            # whole series is in hand: the range is the 1st and 99th percentile of every recorded
            # value of the block, the zeros left out when the block is a sparse current so that
            # seventeen spots on a sheet of zeros are not squeezed into the top bin. Live there
            # is only the run so far: the range widens with the values seen over the first fifth
            # of the run, then holds, so a colour means one number for the rest of the movie.
            series = getattr(lv, "_blocks", {}).get(str(cfg["block"])) if not hasattr(lv, "state_schema") else None
            if series is not None and getattr(self, key, None) is None:
                allv = np.asarray(series, np.float64).reshape(series.shape[0], series.shape[1], -1)[:, :, col].ravel()
                if f is not None:
                    allv = allv * float(f)
                if bool(cfg.get("abs", False)):
                    allv = np.abs(allv)
                nz = allv[allv != 0.0]
                pool = nz if nz.size > max(10, allv.size // 1000) else allv
                lo, hi = float(np.nanpercentile(pool, 1)), float(np.nanpercentile(pool, 99))
                if (allv == 0.0).any() and lo > 0.0:
                    lo = 0.0                                         # a current that is zero elsewhere starts at zero
                if not hi > lo:
                    hi = lo + 1.0
                setattr(self, key, (lo, hi))
            elif getattr(self, key, None) is not None:
                lo, hi = getattr(self, key)
                _settling = int(getattr(self, "_cb_tick", 0)) < max(2, int(self.n_frames) // 5)
                if _settling and v.size:
                    lo, hi = min(lo, float(np.nanmin(v))), max(hi, float(np.nanpercentile(v, 99)))
                    setattr(self, key, (lo, hi))
            else:
                lo, hi = (float(np.nanmin(v)), float(np.nanpercentile(v, 99))) if v.size else (0.0, 1.0)
                if not hi > lo:
                    hi = lo + 1e-9
                setattr(self, key, (lo, hi))
        lab = _units_label(unit, self._units) if unit else None
        label = ("|" if bool(cfg.get("abs", False)) else "") + f"{set_name}.{cfg['block']}" + ("|" if bool(cfg.get("abs", False)) else "") + (f" ({lab})" if lab else "")
        return v, lo, hi, str(cfg.get("cmap", "viridis")), label

    def _dot_light(self):
        """The lighting keywords of a dot sprite: unlit unless `plotting.dot_shading: true`."""
        st = self.style or {}
        if st.get("dot_shading", False) is not True:
            return dict(lighting=False)
        return dict(lighting=True, ambient=float(st.get("dot_ambient", 0.3)),
                    diffuse=float(st.get("dot_diffuse", 0.7)), specular=float(st.get("dot_specular", 0.0)),
                    specular_power=float(st.get("dot_specular_power", 20)))

    def _silhouette(self):
        """`plotting.silhouette: {color, width}` -- THE OUTLINE OF THE STRUCTURAL-BIOLOGY FIGURE.
        ChimeraX draws every surface with a dark line where its depth jumps against what is behind
        it (`graphics silhouettes true`), and that line is what makes a segmented cryo-EM map read
        as bodies in front of bodies instead of one coloured blob; every panel of Drobnic et al.
        2025 is drawn so. VTK's silhouette filter is that line for a mesh (pyvista `silhouette=`),
        so surfaces and spheres take it here; point sprites cannot carry one, and for them `edl`
        is the same cue computed from the depth buffer instead."""
        sil = (self.style or {}).get("silhouette")
        if not sil:
            return {}
        sil = sil if isinstance(sil, dict) else {}
        # NO `feature_angle` UNLESS A NUMBER IS GIVEN: pyvista turns any non-None value into
        # SetEnableFeatureAngle(True) with that value as the angle, and False is the angle 0, at
        # which EVERY edge is a feature -- measured as 17 wireframe globes where 17 outlined balls
        # were asked for.
        out = dict(color=str(sil.get("color", "black")), line_width=float(sil.get("width", 2.0)))
        if isinstance(sil.get("feature_angle"), (int, float)) and not isinstance(sil.get("feature_angle"), bool):
            out["feature_angle"] = float(sil["feature_angle"])
        return dict(silhouette=out)

    def _isosurface(self, X, st, per_compartment=""):
        """The density isosurface of one point cloud -- the same recipe `_skin_build` uses.

        PER-COMPARTMENT RESOLUTION, via `plotting.surface: {<type>: {spacing, blur, smooth, iso}}`.
        One spacing cannot serve the whole atlas and the default (the MPM grid's own dx) serves
        almost none of it: at dx = 0.39 um a cytoskeletal filament of radius 0.18 um is thinner
        than one voxel, so its density field is a chain of isolated blobs and the reconstruction
        draws a string of beads; a plasma membrane assembled from 421 separate patches needs the
        OPPOSITE -- a coarser cell and a heavier blur, so the patches merge into one shell instead
        of 421 lumps. The simulation's resolution is a property of the solver; the resolution a
        structure has to be DRAWN at is a property of the structure.
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        _ov = ((st.get("surface") or {}).get(per_compartment) or {}) if per_compartment else {}
        st = {**st, **{f"surface_{k}": v for k, v in _ov.items()}}
        h = float(st.get("surface_spacing", 0.0)) or self._cs_dx
        # THE PAD COVERS THE BLUR'S HALO. Three cells of padding with a blur of 1.5 cells left the
        # smoothed density above the contour level ON THE GRID'S FACES, so marching cubes cut the
        # shell open there: a window through the hook and the bushing on the -x/-y side, fixed in
        # the world while the parts turned (exp_02 steps 0034-0049; measured: 240 open edges on the
        # hook's surface, density 0.51 on the faces against a level of 0.22). Pad by three cells
        # plus three sigma of the blur and the shell closes.
        _pad = (3.0 + 3.0 * float(st.get('surface_blur', 1.0))) * h
        lo = X.min(0) - _pad
        dim = np.maximum(np.ceil((X.max(0) + _pad - lo) / h).astype(int) + 1, 2)
        # A CAP ON THE GRID, because this now runs once PER COMPARTMENT and a thin shell spanning
        # the whole cell has a bounding box as large as the cell: nine full-cell grids at the MPM
        # dx is nine times the memory of one, and a smooth-ER tubule network wandering 0.9 R has
        # the biggest box of all while holding the fewest points.
        while int(np.prod(dim)) > int(st.get("surface_max_cells", 24_000_000)):
            h *= 1.3
            dim = np.maximum(np.ceil((X.max(0) + _pad - lo) / h).astype(int) + 1, 2)
        ijk = np.clip(((X - lo) / h).astype(np.int64), 0, dim - 1)
        D = np.zeros(tuple(dim), np.float32)
        np.add.at(D, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
        _held = D > 0                                       # the cells that HOLD particles
        D = gaussian_filter(D, sigma=float(st.get("surface_blur", 1.0)))
        # THE BULK OVER THE CELLS THAT HELD PARTICLES, not over the blur's halo -- see _skin_build:
        # the halo's median is 0.01-0.03 against a true 4-7, and the skins came out 5 cells fat.
        occ = D[_held]
        if occ.size == 0:
            return None
        # `surface_iso_frac`, the contour as a fraction of the median occupied cell (0.5 unless said):
        # lower closes the pinholes a sparse patch leaves in a compartment's shell (exp_02 steps 0034-0043).
        iso = float(st.get("surface_iso", 0.0)) or float(st.get("surface_iso_frac", 0.5)) * float(np.median(occ))
        g = self.pv.ImageData(dimensions=tuple(int(v) for v in dim),
                              spacing=(h, h, h), origin=tuple(float(v) for v in lo))
        g.point_data["d"] = D.ravel(order="F")
        surf = g.contour([iso], scalars="d")
        if surf.n_points == 0:
            return None
        # NOT `extract_largest`, which is right for one body and wrong for a COMPARTMENT: there are
        # 77 mitochondria and 421 membrane patches, and keeping only the biggest connected piece
        # would draw one of them.
        return surf.smooth_taubin(n_iter=int(st.get("surface_smooth", 20)), pass_band=0.08)

    @staticmethod
    def _refresh_normals(surf):
        """THE NORMALS TURN WITH THE BODY. `smooth_shading` computes a `Normals` array once, at
        `add_mesh`; writing new `points` moves the vertices and leaves that array as it was, so a
        body that has turned is lit as if it had not -- after half a turn its lit side is the one
        the light no longer reaches, and the hook of exp_02 step 0073 went from orange to brown
        over the movie while the anchored LP ring beside it kept its colour (the human,
        2026-09-24). FROM THE FACES, IN NUMPY, because `compute_normals` handed back the normals
        the mesh already carried rather than new ones (measured: a hook turned rigidly through
        82 degrees came back with normals at cos 82 to the true ones) and an in-place result
        never reached the GPU. Area-weighted face normals summed at the vertices, oriented like
        the array they replace, assigned as a NEW array and the mesh marked modified: a rigid turn
        then renders at constant brightness (63.0 +- 0.1 over 160 degrees, SSAO off)."""
        try:
            if surf is None or not surf.n_points or "Normals" not in surf.point_data:
                return
            F = getattr(surf, "regular_faces", None)
            if F is None or len(F) == 0:
                fc = np.asarray(surf.faces).reshape(-1, 4)
                F = fc[:, 1:4]
            F = np.asarray(F, np.int64)
            V = np.asarray(surf.points, np.float64)
            fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])     # area-weighted
            N = np.zeros_like(V)
            for j in range(3):
                np.add.at(N, F[:, j], fn)
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-30)
            old = np.asarray(surf.point_data["Normals"], np.float64)
            if old.shape == N.shape and float(np.mean(np.sum(N * old, axis=1))) < 0.0:
                N = -N                                       # keep the orientation the shading was built on
            surf.point_data["Normals"] = N.astype(np.float32)
            surf.point_data.active_normals_name = "Normals"
            surf.Modified()
        except Exception:                                    # noqa: BLE001 -- a stale shade, never a lost frame
            pass

    def _skins_update(self, pos, H=None):
        """Ride each compartment's surface on the points it was bound to.

        A group whose `set` is None is a slice of the DRAWN level and reads `pos`; a group that
        names a set reads THAT set's own positions, because with one particle set per organelle
        there is no single array they all live in.
        """
        import numpy as np
        X = np.asarray(pos, dtype=np.float64)
        for s in getattr(self, "_skins", None) or []:
            if not s:
                continue
            src = X
            if s.get("set") and H is not None:
                src = np.asarray(H.level(s["set"]).get("pos").detach().cpu().numpy(), np.float64)
            if s.get("cb") is not None and s.get("set") and H is not None:
                _v, _lo, _hi, _, _ = self._block_values(H, s["set"], s["cb"])
                s["surf"]["v"] = (s["skin"].scalar(_v[s["sub"]]) if s["kind"] == "surface" else _v[s["sub"]]).astype(np.float32)
                try:                                                     # the legend follows the settled range
                    _act = s.get("actor")
                    if _act is not None:
                        _act.mapper.scalar_range = (_lo, _hi)
                except Exception:                                        # noqa: BLE001
                    pass
            if s["kind"] == "surface" and s.get("recontour"):
                # `surface.<set>.recontour: true` -- CONTOURED AGAIN EVERY RENDERED FRAME, for a FLUID.
                # A skin is built once and then rides on the particles it was bound to, which is right
                # for a body (a rotor, a protein chain: its particles keep their neighbours) and wrong
                # for a lipid bilayer, whose particles diffuse past each other: a skin bound to the
                # neighbours of frame 0 crumples as they drift apart. Same recipe as the first build
                # (`_isosurface`, the set's own spacing/blur/iso), same actor, new geometry
                # (experiments/exp04_membrane_channels, 2026-09-25; session A's file, noted there).
                try:
                    _new = self._isosurface(src[s["sel"]], self.style or {}, per_compartment=s["name"])
                    if _new is not None and _new.n_points:
                        _new = _new.compute_normals(auto_orient_normals=False)
                        _nsk = self._skin_cut(s)
                        _cut_far = isinstance(_nsk, str) and _nsk.lower() == "far"
                        if _cut_far:
                            # A PLANE CLIP, NOT A FACE FILTER, for a surface re-made every frame anyway: dropping
                            # whole faces by their centroids left the slab's cut rim a sawtooth (round 5-6 judges)
                            _new = self._clip_far(_new)
                        s["surf"].copy_from(_new)
                        if _nsk and not _cut_far:
                            s["surf"]._full_faces = None               # new geometry: re-cut from ITS faces
                            self._near_side_faces(s["surf"], _nsk)
                        s["surf"].Modified()
                except Exception as _e:                              # noqa: BLE001 -- keep the last shell
                    if not getattr(self, "_recontour_warned", False):
                        print(f"[live-movie] recontour of {s['name']!r} failed ({type(_e).__name__}: {_e}); "
                              f"keeping the previous shell", flush=True)
                        self._recontour_warned = True
            elif s["kind"] == "surface":
                s["surf"].points = s["skin"](_no_strays(src[s["sub"]])).astype(np.float32)
                # `near_side` CUTS A COMPARTMENT SURFACE TOO. The mesh-set path has always been cut
                # by `_near_side_faces`; a compartment skin was not, so a motor drawn as thirteen
                # closed surfaces could not be looked into (exp_02, 2026-09-23). Same plane, same
                # rule, re-applied after the skin moves the vertices.
                _nsk = self._skin_cut(s)
                if _nsk:
                    self._near_side_faces(s["surf"], _nsk)
                elif getattr(s["surf"], "_full_faces", None) is not None:
                    s["surf"].faces = s["surf"]._full_faces               # the slice switched off: whole again
                self._refresh_normals(s["surf"])
            else:
                s["surf"].points = np.asarray(src[s["sub"]], np.float32)

    # ---- the relation, drawn ---------------------------------------------------------------
    #
    # `plotting.graph_overlay: {sets: [...]}` draws each named set's `edge_index` as line segments
    # between the positions of the elements it relates, in that set's own colour.
    #
    # IT RUNS AT THE END, AND ONE SET AT A TIME. A relation over a moving body is unreadable while
    # the body is moving -- and it is not what the movie is about until the movie is over. So the
    # last len(sets)+1 RENDERED frames are given to it: one frame per relation, then a final frame
    # with all of them at once. The mechanics is the film; the graphs are the closing statement.
    #
    # THE POSITIONS COME FROM THE ORGANELLE SET, NOT FROM THE MATERIAL POINTS, which is only
    # meaningful because `aggregate_centroid` has been keeping that set's `pos` current -- the
    # relation is between mitochondria, and a mitochondrion's position is the centroid of the
    # points it owns. Without that aggregate the overlay would draw a graph over frozen seed
    # positions and look plausible while being a picture of frame 0.
    def _graph_setup(self, H):
        self._graphs = []
        cfg = (self.style or {}).get("graph_overlay")
        if not cfg:
            return
        cfg = {"sets": list(cfg)} if isinstance(cfg, (list, tuple)) else dict(cfg)
        names = [n for n in (cfg.get("sets") or []) if n in H.levels]
        if not names:
            print("[live-movie] graph_overlay: none of its sets exist in this model", flush=True)
            return
        from matplotlib.colors import to_rgb
        pal = (self.style or {}).get("colors") or {}
        self._graph_cfg = cfg
        rendered = list(range(self.stride, self.n_frames + 1, self.stride)) or [self.n_frames]
        k = len(names) + 1
        self._graph_ticks = {t: (names[:i + 1] if i < len(names) else names)
                             for i, t in enumerate(rendered[-k:])}
        # the LAST rendered tick shows every relation at once
        self._graph_ticks[rendered[-1]] = names
        for i, t in enumerate(rendered[-k:-1]):
            self._graph_ticks[t] = [names[i]]
        for n in names:
            c = pal.get(n)
            col = to_rgb(tuple(c)) if isinstance(c, (list, tuple)) else (0.9, 0.9, 0.9)
            pd = self.pv.PolyData()
            act = self.p.add_mesh(pd, color=col, line_width=float(cfg.get("line_width", 1.6)),
                                  opacity=float(cfg.get("opacity", 0.95)), lighting=False,
                                  show_scalar_bar=False)
            act.SetVisibility(False)
            self._graphs.append({"name": n, "pd": pd, "actor": act})
        print(f"[live-movie] graph_overlay: {', '.join(names)} -- one per frame over the last "
              f"{k} rendered frames, all together on the last", flush=True)

    def _graph_update(self, H, tick):
        gs = getattr(self, "_graphs", None)
        if not gs:
            return
        show = getattr(self, "_graph_ticks", {}).get(tick)
        # `graph_overlay.always: true` draws the relation on EVERY frame -- a circuit's synapses or
        # a metabolic network's stoichiometry are the picture, not its closing statement.
        if (getattr(self, "_graph_cfg", None) or {}).get("always"):
            show = [g["name"] for g in gs]
        if show is None:
            for g in gs:
                g["actor"].SetVisibility(False)
            return
        import numpy as np
        cap = int((getattr(self, "_graph_cfg", None) or {}).get("max_edges", 40000))
        for g in gs:
            on = g["name"] in show
            g["actor"].SetVisibility(on)
            if not on:
                continue
            lvl = H.level(g["name"])
            ei = getattr(lvl, "edge_index", None)
            # AN EDGE-SET IS A RELATION TOO, drawn between the sets it joins: a synapse set has
            # `pre`/`post` into the neuron set and no positions of its own, so the segments run
            # from the pre endpoint's `pos` to the post endpoint's. Directed, so every edge is
            # drawn (the symmetric filter below is for a proximity graph, which stores each pair
            # twice). A bipartite relation (stoichiometry: metabolite -> reaction) draws only
            # when both endpoint sets carry a `pos`.
            _pre = getattr(lvl, "pre", None)
            if (ei is None or ei.numel() == 0) and _pre is not None and getattr(lvl, "post", None) is not None:
                try:
                    Pa = H.level(lvl.pre_name).get("pos")
                    Pb = H.level(lvl.post_name).get("pos")
                except Exception:                    # noqa: BLE001 -- an endpoint with no position
                    g["actor"].SetVisibility(False)
                    continue
                a = _pre.detach().cpu().numpy(); b = lvl.post.detach().cpu().numpy()
                if a.shape[0] > cap:
                    a = a[:: a.shape[0] // cap + 1]; b = b[:: b.shape[0] // cap + 1]
                Pa = np.asarray(Pa.detach().cpu().numpy(), np.float32)
                Pb = np.asarray(Pb.detach().cpu().numpy(), np.float32)
                if Pa.shape[1] == 2:
                    Pa = np.concatenate([Pa, np.zeros((len(Pa), 1), np.float32)], 1)
                    Pb = np.concatenate([Pb, np.zeros((len(Pb), 1), np.float32)], 1)
                pts = np.concatenate([Pa[a], Pb[b]], axis=0)
                m = a.shape[0]
                lines = np.column_stack([np.full(m, 2, np.int64),
                                         np.arange(m, dtype=np.int64),
                                         np.arange(m, 2 * m, dtype=np.int64)]).ravel()
                g["pd"].points = pts
                g["pd"].lines = lines
                g["pd"].Modified()
                continue
            if ei is None or ei.numel() == 0:
                g["actor"].SetVisibility(False)
                continue
            e = ei.detach().cpu().numpy()
            keep = e[0] < e[1]                       # the relation is symmetric: draw each pair once
            e = e[:, keep]
            if e.shape[1] > cap:                     # a proximity graph is O(N * density); cap it
                e = e[:, :: e.shape[1] // cap + 1]
            P = np.asarray(lvl.get("pos").detach().cpu().numpy(), np.float32)
            src, dst = e[0], e[1]
            pts = np.concatenate([P[src], P[dst]], axis=0)
            m = src.shape[0]
            lines = np.column_stack([np.full(m, 2, np.int64),
                                     np.arange(m, dtype=np.int64),
                                     np.arange(m, 2 * m, dtype=np.int64)]).ravel()
            g["pd"].points = pts
            g["pd"].lines = lines
            g["pd"].Modified()

    def _skin_update(self, H, lvl, pos):
        if getattr(self, "_skins", None):
            self._skins_update(pos, H)
            return
        if self._skin is None or self._surf is None:
            return
        X = np.asarray(pos, dtype=np.float64)[self._skin_sub]
        self._surf.points = self._skin(X).astype(np.float32)
        self._refresh_normals(self._surf)
        if str((self.style or {}).get("color_field", "") or ""):
            val = self._field(H, lvl)[0]
            if val is not None:
                v = val.detach().cpu().numpy().astype(np.float64)[self._skin_sub]
                self._surf["f"] = self._skin.scalar(v)

    # ---- the surface, drawn over the cloud ------------------------------------------------
    #
    # A MESH SET IN THE SAME SCENE, NOT A SECOND RENDERER. This renderer drew exactly one thing --
    # the largest Level carrying positions -- so a spec that couples a triangulated surface to a
    # continuum rendered the continuum and left the surface out, and the one picture nobody could
    # get was the picture of the coupling. `discovery_okuda/ops/test_03_mesh_contact.py` solved it
    # with a matplotlib wireframe over a scatter; that is the right IMAGE and the wrong renderer
    # (mpl has no depth buffer, so the far half of a surface is drawn over the near half whenever
    # the painter's-algorithm tie goes the wrong way). Here the surface is a second VTK actor,
    # z-buffered against the dots by construction, and it costs nothing per frame but a point
    # array swap.
    def _mesh_map(self, name):
        """(scale, centre) for a mesh set, read off the `mesh_contact` that consumes it.

        A SURFACE NEED NOT BE IN BOX COORDINATES. `mesh_contact` maps it at the point of USE --
        `scale` and `centre`, gate 04's own device -- so a vesicle can live at the ORIGIN in its own
        units, which is where `cell_mechanics`'s radial term requires it. The renderer drew it where
        it truly is: radius 2.3 -> 9.3 about the origin, entirely outside a [0, 1] box, so the movie
        showed the gel and an empty space where the experiment was.
        READ OFF THE OPERATOR, NOT DECLARED AGAIN. A second copy of `scale` in `plotting` is a
        second chance to disagree, and a picture drawn at a different scale from the physics is
        worse than no picture.
        """
        for o in (getattr(self.sim, "operators", None) or []):
            pr = getattr(o, "params", None) or {}
            if getattr(o, "op", "") == "mesh_contact" and pr.get("surface") == name:
                c = pr.get("centre", [0.0, 0.0, 0.0])
                c = [0.0, 0.0, 0.0] if isinstance(c, str) else [float(v) for v in c]
                return float(pr.get("scale", 1.0)), np.asarray(c, np.float32)
        return 1.0, None

    def _mesh_xyz(self, lvl, nv, sc, ct):
        P = lvl.get("pos")[:nv].detach()
        # `plotting.mesh_surface: apical | basal | mid` -- WHICH SURFACE OF AN APICO-BASAL TISSUE
        # THE MESH IS DRAWN AT. The mesh table lives on the mid-surface, which is a coordinate and
        # not a membrane; integrins ride the basal cap at `pos - sep`, and drawn against the
        # mid-surface they floated half a cell thickness off the tissue. Default `mid`, so nothing
        # archived moves. A tissue without a `sep` block ignores the key.
        _surf = str((self.style or {}).get("mesh_surface", "mid")).lower()
        if _surf in ("apical", "basal"):
            try:
                _sep = lvl.get("sep")
            except Exception:                                # noqa: BLE001
                _sep = None
            if _sep is not None and int(_sep.shape[0]) >= nv:
                _sep = _sep[:nv]                             # a tensor live, a numpy array on replay
                _sep = (_sep.detach().to(P.device, P.dtype) if hasattr(_sep, "detach")
                        else torch.as_tensor(np.asarray(_sep), device=P.device, dtype=P.dtype))
                P = P + _sep if _surf == "apical" else P - _sep
        P = P.cpu().numpy().astype(np.float32)
        return P if (sc == 1.0 or ct is None) else (P - P.mean(0)) * sc + ct

    def _mesh_levels(self, H):
        """Every Level carrying a non-empty half-edge table, minus `plotting.hide_sets`.

        THE CELL SET'S BLOCKS ARE OVERLAID ON THE MESH TABLE, and that is what lets one renderer
        keep working while per-cell state moves off the mesh. A face of the mesh IS a cell -- the
        spec declares the pairing as `cell_set:` and the engine records it as `Level.mesh_cell_set`
        -- so a width-1 block on the cell set is a per-face quantity by construction, and the two
        readers that want one (`_mesh_face_rgb`'s `mesh_color_by`, the `phase` curve) go on asking
        `m.get(name)`. `phase` is the first to arrive this way; `A0`, `age` and the rest follow
        without touching this file again.

        A `_MeshView` RATHER THAN A WRITE INTO `m`. Putting the cell's block back onto the mesh
        table would restore the two homes this move exists to remove, and it would do it in the
        renderer, where nothing renumbers. The view is read-only and the mesh table wins any name
        clash, so an overlay can never shadow `nF`, `E_srce` or a genuine face column.
        """
        hide = set((self.style or {}).get("hide_sets", []) or [])
        out = []
        for name, lvl in H.levels.items():
            # THE DRAWN SET IS EXCLUDED ONLY WHEN ITS DOTS ARE ACTUALLY ON SCREEN. For a spec whose
            # ONLY positional set is a vertex mesh -- a vertex-model run with no material -- that set
            # is both `_sname` and the surface, so this skipped the one thing there was to draw and
            # the movie was a few hundred dots in a corner of the world box.
            if name in hide or (name == getattr(self, "_sname", None) and not self._mesh_is_subject):
                continue
            m = getattr(lvl, "mesh", None)
            if m is None or not int(m.get("nF", 0) or 0):
                continue
            out.append((name, lvl, _MeshView(m, self._cell_cols(H, lvl, int(m["nF"])))))
        return out

    @staticmethod
    def _cell_cols(H, lvl, nF):
        """{block: array[nF]} for every width-1 recorded block on this mesh's cell set.

        TWO SOURCES, ONE ANSWER, AND NEITHER GUESSES THE SET. Live, the name is
        `Level.mesh_cell_set`, written by the engine from the spec's `cell_set:`; on replay the
        level has already loaded the same set's recorded blocks, because `replay` read the pairing
        out of the spec and handed it down. A renderer that scanned the trajectory for a key ending
        in `__phase` would be inferring a declared relation instead of reading it.
        """
        pre = getattr(lvl, "cell_cols", None)          # replay: loaded in _ReplayLevel.__init__
        if pre is not None:
            return pre(nF)
        cs = getattr(lvl, "mesh_cell_set", None)
        clvl = H.level(cs) if cs and cs in getattr(H, "levels", {}) else None
        if clvl is None or getattr(clvl, "state", None) is None:
            return {}
        out = {}
        for b in getattr(clvl.state_schema, "blocks", ()):
            if not getattr(b, "record", True) or b.width != 1:
                continue
            out[b.name] = clvl.get(b.name)[:nF, 0]
        return out

    @staticmethod
    def _mesh_faces(m):
        """The half-edge table as VTK's flat face array: [n, i0..i(n-1), n, i0.., ...].

        THE RING ORDER IS ALREADY THERE. `E_face` is grouped by face and `E_srce` walks each face's
        vertices in order, so the polygon is read straight off the table -- no triangulation, which
        means a quad plate and a polygonal epithelium go through the same three lines.
        """
        ef = m["E_face"].detach().cpu().numpy()
        es = m["E_srce"].detach().cpu().numpy()
        nF = int(m["nF"])
        cnt = np.bincount(ef, minlength=nF)
        offs = np.concatenate([[0], np.cumsum(cnt)])
        faces = np.empty(len(es) + nF, np.int64)
        faces[offs[:-1] + np.arange(nF)] = cnt
        faces[np.arange(len(es)) + np.repeat(np.arange(nF), cnt) + 1] = es
        return faces

    def _add_meshes(self, H):
        """One actor per mesh set, built once. Wireframe by default: a filled surface over a point
        cloud hides the material the surface is acting on, which is the half of the picture the
        contact is about."""
        self._meshes = []
        try:
            st = self.style or {}
            colr = st.get("mesh_color", "#e6dcc0")
            lw = float(st.get("mesh_line_width", 0.8))
            # A WIREFRAME OVER A CLOUD, A LIT SOLID WHEN IT IS THE PICTURE. Over 500,000 dots a solid
            # surface hides the material it is acting on, which is half of what a contact run is
            # about; with nothing behind it a wireframe is a tangle of edges with no shape.
            style = str(st.get("mesh_style", "surface" if self._mesh_is_subject else "wireframe"))
            # TRANSLUCENT ONLY AS AN OVERLAY. 0.55 is right for a surface sitting ON a point cloud --
            # it is there so the material underneath is not hidden by the thing acting on it -- and
            # wrong when the surface is the whole picture: a 6,000-cell epithelium at 0.55 shows its
            # own far side through its near side, so every cell is read through another cell.
            opac = float(st.get("mesh_opacity",
                                1.0 if (style == "wireframe" or self._mesh_is_subject) else 0.55))
            # A FULLY TRANSPARENT SURFACE IS NOT DRAWN AT ALL. `mesh_opacity: 0` reads as "leave the
            # surface out", and adding the actor anyway is not the same thing: with depth peeling on,
            # a transparent occluder in front of the material blanked the entire scene -- measured,
            # two specs differing ONLY in mesh_opacity (0.0 against 0.22) gave an empty box and a
            # correct picture of 690,000 nodes.
            if opac <= 0.0:
                self._meshes = []
                return
            for name, lvl, m in self._mesh_levels(H):
                nv = int(m["Nv"])
                sc, ct = self._mesh_map(name)
                pd = self.pv.PolyData(self._mesh_xyz(lvl, nv, sc, ct), self._mesh_faces(m))
                if sc != 1.0:
                    print(f"[live-movie] surface {name!r} drawn through mesh_contact's own mapping: "
                          f"x{sc:g} about its centroid, placed at {list(np.round(ct, 4))}", flush=True)
                # THE CELL BOUNDARIES ARE THE SUBJECT WHEN THE MESH IS. A shaded surface with no
                # edges renders a 6,000-cell epithelium as a smooth grey ball -- the tessellation,
                # which is the entire reason the model has faces, is invisible. `render_vtk` draws
                # one polygon per cell with its outline, and this matches it. Off by default when
                # the mesh is an OVERLAY: 2,304 plate quads of edge over a cloud is a moire.
                _edges = bool(st.get("mesh_edges", self._mesh_is_subject and style == "surface"))
                # FLAT WHEN THE MESH IS THE SUBJECT, which is `render_vtk`'s own default style for
                # this picture and is not a preference. A lit shaded ball reads its own curvature
                # as brightness, so the darkening toward the limb competes with the per-cell colour
                # the marks are carrying; unlit, a cell's colour means only what it was set to.
                _flat_m = (self._mesh_is_subject and style == "surface"
                           and bool(st.get("mesh_flat", True)))
                _rgb = self._mesh_face_rgb(m, pd, H, lvl) if self._mesh_is_subject else None
                # A CONCAVE CELL IS FILLED FROM A TRIANGULATION, NOT AS ONE POLYGON. VTK draws a
                # polygon cell as a fan from its first vertex, which is right only when the cell is
                # convex: a concave one paints over its neighbours. Measured on a C-shaped test
                # polygon, the notch came out at brightness 148 of 255 where it is empty, and on
                # exp01_v3 -- 86 of 169 left-half cells concave -- the sheet rendered as a tangle
                # of overlapping shards over a trajectory that was flat, with no cell inverted.
                # `vtkTriangleFilter` ear-cuts each polygon (and carries its `rgb` to every piece),
                # and it is wired to `pd` as a PIPELINE, so every per-frame change to the points,
                # the rings, the colours or the near-side cut re-triangulates on the next render.
                # The outlines are then drawn from the polygons themselves, as a wireframe of
                # `pd`: `show_edges` on the triangulation would draw every ear-cut diagonal too.
                _tri = style == "surface"
                _src = pd
                if _tri:
                    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
                    _src = vtkTriangleFilter()
                    _src.SetInputData(pd); _src.PassVertsOff(); _src.PassLinesOff(); _src.Update()
                _ecol = st.get("mesh_edge_color", "#2b2b2b")
                _eop = float(st.get("mesh_edge_opacity", 1.0))
                _fill = self.p.add_mesh(_src, color=(None if _rgb is not None else colr),
                                scalars=("rgb" if _rgb is not None else None), rgb=(_rgb is not None),
                                style=style, line_width=lw, opacity=opac,
                                lighting=(style != "wireframe" and not _flat_m),
                                ambient=(1.0 if _flat_m else 0.3),
                                diffuse=(0.0 if _flat_m else 0.7), specular=0.0,
                                render_lines_as_tubes=False,
                                show_edges=(_edges and not _tri), edge_color=_ecol,
                                edge_opacity=_eop)
                if _tri:
                    self._mesh_fill = getattr(self, "_mesh_fill", {})
                    self._mesh_fill[id(pd)] = _fill
                    if _edges:
                        self.p.add_mesh(pd, style="wireframe", color=_ecol, line_width=lw,
                                        opacity=_eop * opac, lighting=False,
                                        render_lines_as_tubes=False)
                self._near_side_faces(pd)
                self._meshes.append((name, nv, pd, sc, ct))
                print(f"[live-movie] surface {name!r}: {int(m['nF']):,} faces, {nv:,} vertices, "
                      f"drawn as {style}", flush=True)
        except Exception as e:                       # noqa: BLE001 -- never kill a run for a picture
            self._meshes = []
            print(f"[live-movie] mesh overlay unavailable ({type(e).__name__}: {e})", flush=True)

    def _mesh_face_chem(self, H, nF):
        """Per-face colour from the FACE set's own `chem` block, or None when it carries none.

        The same law as every other reader of `chem` -- `plexus.live.chem_rgb`, with the spec's
        `species`, `blend` and `chem_max` -- so a morphogen on a closed shell is drawn the way the
        same morphogen on a flat sheet or a point disc is. Read through `lvl.get`, not the state
        schema, because a replayed level serves blocks without carrying a schema.
        """
        if H is None:
            return None
        st = self.style or {}
        # `mesh_chem: false` (default true): the layer keeps `mesh_color` even though its cells carry `chem` -- exp 11,
        # 2026-09-29: `bm_sense` writes a membrane deficit into chem channel 0, and it painted those cells orange on
        # the main view, a colour nobody had asked for and no label named
        if not bool(st.get("mesh_chem", True)):
            return None
        v = None
        try:
            v = H.level(str(st.get("mesh_chem_set", "cell"))).get("chem")
        except Exception:                                                # noqa: BLE001
            # A REPLAYED CELL SET IS NOT A LEVEL -- it has no `pos` -- and its blocks ride on the
            # mesh level it was paired with (`_ReplayLevel._cell_blocks`, see `cell_cols`).
            for _l in (getattr(H, "levels", {}) or {}).values():
                _b = (getattr(_l, "_cell_blocks", None) or {}).get("chem")
                if _b is not None:
                    v = np.asarray(_b[_l.t])
                    break
        if v is None or int(v.shape[-1]) < 1:
            return None
        from plexus.live import chem_rgb
        c = (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))[:nF]
        if c.shape[0] < nF:
            c = np.concatenate([c, np.zeros((nF - c.shape[0],) + c.shape[1:], c.dtype)])
        rgb, _ = chem_rgb(np.asarray(c, float), lut=st.get("species"),
                          blend=st.get("blend"), background=st.get("background", "black"),
                          vmax=st.get("chem_max"))
        if rgb is None:
            return None
        return (np.clip(np.asarray(rgb, float), 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def _face_node_type(H, lvl, nF):
        """[nF] type of each face of the mesh `lvl`, THIS frame: the paired cell set's `node_type`.

        Live, `Level.mesh_cell_set` names the set and its `node_type` is current (an operator that
        retypes a cell writes it there). On replay the level serves the recorded column itself --
        the per-row `<cell>__node_type_t` when the types changed, the final column otherwise
        (`_ReplayLevel.cell_node_type`). None when neither exists.
        """
        if lvl is None:
            return None
        pre = getattr(lvl, "cell_node_type", None)       # replay
        if pre is not None:
            return pre(nF)
        cs = getattr(lvl, "mesh_cell_set", None)
        clvl = H.level(cs) if (H is not None and cs and cs in getattr(H, "levels", {})) else None
        nt = getattr(clvl, "node_type", None)
        return None if nt is None else nt[:nF]

    def _mesh_face_rgb(self, m, pd, H=None, lvl=None):
        """One colour per cell: the base, with the division pair marked -- `render_vtk`'s own rule.

        REUSED, NOT REIMPLEMENTED. `render_vtk._marks` is where the mother/daughter split lives, and
        it is subtle enough to be worth importing rather than restating: the two masks are on
        DIFFERENT CLOCKS (`age <= DIVIDED` counts division CALLS, "appended since" counts rows), so
        a naive "compare with the previous frame" draws mothers on every frame and daughters on
        almost none. It takes a face count from far enough back to cover the same window.
        """
        import matplotlib.colors as _mc
        st = self.style or {}
        nF = int(m["nF"])
        base = np.tile((np.asarray(_mc.to_rgb(st.get("mesh_color", "#e6dcc0"))) * 255)
                       .astype(np.uint8), (nF, 1))
        _chem = self._mesh_face_chem(H, nF)
        if _chem is not None and _chem.shape == base.shape:
            base = _chem

        # `mesh_color_by: phase` -- PAINT THE CELL CYCLE, AND DO IT BEFORE ANYTHING ELSE CAN FAIL.
        # Placed AFTER the division marks inside one `try`, this block draws nothing on any run
        # that divides: `_marks` reads a face count that has just changed, and
        # anything it raises aborts the whole block, leaving the PREVIOUS frame's colours on the
        # actor. A six-frame run with no divisions painted phases correctly and a forty-frame run
        # with them painted mother-blue, from the same code and the same spec. Two independent
        # encodings must not share a failure.
        #
        # THE COLOURS ARE THE TEXTBOOK ONES -- G1 blue, S green, G2 amber, M red. Every cell-biology
        # figure uses them, so a reader already has the mapping; a private ramp would ask them to
        # learn a second one, and a phase is a name rather than a position on a scale.
        if str(st.get("mesh_color_by", "") or "").lower() == "phase":
            ph = m.get("phase")
            if ph is None:
                if not getattr(self, "_phase_warned", False):
                    self._phase_warned = True
                    print("[live-movie] mesh_color_by: phase, but the mesh carries no `phase` "
                          "column -- is `cell_cycle` in the schedule?", flush=True)
            else:
                p = (ph.detach().cpu().numpy() if hasattr(ph, "detach") else np.asarray(ph))
                p = np.asarray(p, np.float64).ravel()
                # PADDED, NOT TRUNCATED-AND-HOPED. A boolean mask shorter than the array it indexes
                # raises, and this function's outer guard turns that into a silently uncoloured
                # frame -- so the one length that must never surprise it is made safe here.
                p = np.resize(p, nF) if p.size != nF else p
                pal = st.get("phase_colors") or ["#3b57c0", "#2e9e4f", "#e8b024", "#e03b2f"]
                q = np.clip(np.rint(p).astype(int), 0, len(pal) - 1)
                for k, c in enumerate(pal):
                    sel = q == k
                    if sel.any():
                        base[sel] = (np.asarray(_mc.to_rgb(c)) * 255).astype(np.uint8)
                pd.cell_data["rgb"] = base
                return base

        # `mesh_color_by: cycle_progress` -- THE CYCLE AS A POSITION, for a rule with no phases.
        # A whole-cycle sizer, adder or timer runs its entire cycle in G1 (S = G2 = M = 0), so the
        # phase colouring above paints every cell blue and a dividing one red for a frame: the
        # picture says nothing (exp 3, 2026-09-25). `cycle_progress` is the cell's own position
        # between birth (0) and division (1); it is drawn through the SAME four colours, blended, so
        # a newborn reads blue and a cell about to divide reads red on either kind of run.
        if str(st.get("mesh_color_by", "") or "").lower() == "cycle_progress":
            cp = m.get("cycle_progress")
            if cp is not None:
                p = (cp.detach().cpu().numpy() if hasattr(cp, "detach") else np.asarray(cp))
                p = np.resize(np.asarray(p, np.float64).ravel(), nF)
                # `cycle_colors` -- THE GRADIENT'S OWN STOPS. A run with no S or G2 (a whole-cycle
                # rule) is drawn blue -> red only: green and amber would claim phases it does not
                # have (user, 2026-09-25). Default: the four phase colours.
                pal = (st.get("cycle_colors") or st.get("phase_colors")
                       or ["#3b57c0", "#2e9e4f", "#e8b024", "#e03b2f"])
                # `cycle_bins` -- DISCRETE, AT THE PHASE BOUNDARIES OF A PHASED RUN. Given the
                # cumulative fractions (e.g. [0.458, 0.792, 0.958] for 183.3/133.3/66.7/16.7 of
                # 400 frames), a cell takes the colour of the phase it WOULD be in on a phased run
                # at the same position, so the two kinds of run read on one discrete scale side by
                # side (user, 2026-09-25: graded here, discrete on the phased steps). Without it the
                # colour is graded through the same four.
                bins = st.get("cycle_bins")
                if bins:
                    q = np.searchsorted(np.asarray(bins, float), np.clip(p, 0.0, 1.0), side="right")
                    q = np.clip(q, 0, len(pal) - 1)
                    base = (np.asarray([_mc.to_rgb(c) for c in pal])[q] * 255).astype(np.uint8)
                else:
                    from matplotlib.colors import LinearSegmentedColormap as _LSC
                    cm = _LSC.from_list("cycle", [_mc.to_rgb(c) for c in pal])
                    base = (np.asarray(cm(np.clip(p, 0.0, 1.0)))[:, :3] * 255).astype(np.uint8)
                pd.cell_data["rgb"] = base
                return base

        # `mesh_color_by: <label block>` -- A CATEGORY PER CELL, for any integer-valued block that is
        # not one of the two above: `clone` (the founder each cell descends from, exp 14) paints the
        # mosaic of lineages, `mutant` the mutant and wild-type populations. `label_colors`, when
        # given, colours values 0, 1, ... in order; any value beyond it takes a hue hashed from its
        # number (splitmix64: the seed's Fibonacci sphere puts golden-ratio steps side by side, and any
        # multiplicative hash is one), so neighbours differ and a clone's colour is the same in every frame and run.
        _lab = str(st.get("mesh_color_by", "") or "")
        if _lab and _lab.lower() not in ("phase", "cycle_progress"):
            lv = m.get(_lab)
            # `node_type` (alias `type`) IS THE CELL SET'S TYPE COLUMN, which is not a state block and
            # so is not on the view: read per frame off the face set (live) or its recorded per-row
            # column (replay), so a cell retyped mid-run is recoloured (exp 9 Part B).
            if lv is None and _lab.lower() in ("node_type", "type"):
                lv = self._face_node_type(H, lvl, nF)
            # A COLORMAP INSTEAD OF CATEGORIES is opt-in, `mesh_color_scale: continuous | auto` --
            # `render_vtk.mesh_scale_continuous`. The range widens over the frames drawn (never
            # narrows) unless `mesh_color_range` fixes it. Absent, the categorical law below stands.
            if lv is not None:
                from plexus.render_vtk import mesh_block_continuous, mesh_scale_continuous
                _v = np.resize(np.asarray(lv.detach().cpu().numpy() if hasattr(lv, "detach")
                                          else lv, np.float64).ravel(), nF)
                if getattr(self, "_mcb_cont", False) or mesh_scale_continuous(_v, st):
                    self._mcb_cont = True
                    _f = np.isfinite(_v)
                    _r = getattr(self, "_mcb_rng", None)
                    if _f.any():
                        _lo, _hi = float(_v[_f].min()), float(_v[_f].max())
                        _r = (_lo, _hi) if _r is None else (min(_r[0], _lo), max(_r[1], _hi))
                        self._mcb_rng = _r
                    base = mesh_block_continuous(_v, st, _r)
                    pd.cell_data["rgb"] = base
                    return base
            if lv is not None:
                from plexus.measures import label_rgb
                q = np.rint(np.resize(np.asarray(lv.detach().cpu().numpy() if hasattr(lv, "detach")
                                                 else lv, np.float64).ravel(), nF)).astype(np.int64)
                pal = st.get("label_colors") or []
                rgb = np.empty((nF, 3))
                for k in np.unique(q):
                    rgb[q == k] = label_rgb(int(k), pal)
                base = (rgb * 255).astype(np.uint8)
                pd.cell_data["rgb"] = base
                return base

        # `mesh_mark_division: false` -- DRAW NO MOTHER/DAUGHTER MARKS. It used to return None here,
        # which is not "no marks" but "no colouring at all": the caller then drew the tissue in one
        # flat colour and any other per-cell encoding was lost with it. Now it returns the base, so
        # turning the marks off leaves everything else standing.
        if not bool(st.get("mesh_mark_division", True)):
            pd.cell_data["rgb"] = base
            return base
        try:
            from plexus.render_vtk import _marks
            self._nF_hist = (getattr(self, "_nF_hist", []) + [nF])[-8:]
            prev = self._nF_hist[0] if len(self._nF_hist) > 1 else None
            # NUMPY, BECAUSE `_marks` IS NUMPY. On the REPLAY path the columns come out of an npz
            # and already are; live from the engine they are CUDA tensors, so the call raised
            # "can't convert cuda:0 device type tensor to numpy" -- and the guard below turned that
            # into a one-line warning and a uniformly grey tissue. The two paths were drawing
            # different pictures again, which is the thing this renderer was unified to stop.
            # ASKED FOR BY NAME, NOT FILTERED OUT OF `items()`. `age` and `ndiv` are blocks on the
            # CELL SET now, so they are served by the mesh view rather than stored in its dict --
            # `items()` walks the dict and would silently drop both, and the division marks are
            # exactly the thing that then draws nothing. Same lesson as the replay's literal
            # four-column list, one level up.
            mt = {}
            # THE BLOCK NAME FIRST, THE RETIRED ALIAS SECOND. `apop_flag` and `inhib_frac` are
            # declared blocks now and are recorded under their own names; a trajectory written
            # before the move still carries `apop`/`inhib`, and a reader that knew only one of the
            # two spellings would draw nothing on half the runs on disk.
            for k, old in (("age", None), ("ndiv", None),
                           ("apop_flag", "apop"), ("inhib_frac", "inhib")):
                v = m.get(k)
                if v is None and old is not None:
                    v = m.get(old)
                if v is not None:
                    mt[old or k] = v.detach().cpu().numpy() if hasattr(v, "detach") else v
            mt["nF"] = nF
            mother, daughter, kills, _sup = _marks(mt, np.arange(nF), nF, prev_nF=prev)
            rgb = base
            for msk, key, dflt in ((mother, "mesh_mother_color", "#4a86c8"),
                                   (daughter, "mesh_daughter_color", "#d9534f"),
                                   (kills, "mesh_apop_color", "#e8c33a")):
                if msk is not None and np.any(msk):
                    rgb[np.asarray(msk, bool)] = (np.asarray(_mc.to_rgb(st.get(key, dflt))) * 255
                                                  ).astype(np.uint8)
            pd.cell_data["rgb"] = rgb
            return rgb
        except Exception as e:                       # noqa: BLE001 -- a colouring is not the run
            if not getattr(self, "_mark_warned", False):
                self._mark_warned = True
                print(f"[live-movie] division marks unavailable ({type(e).__name__}: {e})",
                      flush=True)
            return None

    def _edge_actor(self, H, lvl, m, first):
        """`plotting.edge_color: myosin | type` -- the junctions as their own coloured line mesh.

        REUSED, NOT RESTATED. `render_vtk.edges_of` builds it, the same way `_marks` is imported
        rather than copied: a face colour can only say something about a cell, and myosin lives on
        the junction, so the edges have to be drawn as edges. The range is taken ONCE, on the first
        frame, for the reason every other range here is fixed -- a per-frame one would renormalise
        and a belt that is tightening would look constant.
        """
        mode = str((self.style or {}).get("edge_color", "") or "").lower()
        if not mode:
            return
        from plexus.render_vtk import edges_of
        nt = self._curve_types(H, lvl)
        if first and mode == "myosin":
            v = m.get("e_myo")
            if v is not None:
                ef = np.asarray(m["E_face"]); live = ef < int(m["nF"])
                vv = np.asarray(v, float)[live]
                self._erng = (float(np.nanmin(vv)), float(np.nanmax(vv)))
        pos = np.asarray(lvl.get("pos")[: int(m["Nv"])], np.float32)
        em = edges_of(pos, {k: np.asarray(m[k]) for k in ("E_srce", "E_trgt", "E_face")}
                      | {"nF": int(m["nF"]), "e_myo": m.get("e_myo")},
                      mode, ntype=nt, rng=getattr(self, "_erng", None),
                      colors=list((self.style or {}).get("colors", {}).values()) or None,
                      lut=str((self.style or {}).get("edge_lut", "inferno")))
        if em is None:
            return
        if getattr(self, "_eactor", None) is not None:
            self.p.remove_actor(self._eactor)
        self._eactor = self.p.add_mesh(em, scalars="rgb", rgb=True, lighting=False,
                                       line_width=float((self.style or {}).get("edge_width", 3.0)),
                                       render_lines_as_tubes=True)

    @staticmethod
    def _conn_sig(m):
        """Order-sensitive signature of a half-edge table: changes on any rewiring, not only on a
        change of counts. Torch on the live path, numpy on replay -- both index the same way."""
        es = m["E_srce"]
        if hasattr(es, "detach"):
            import torch
            es = es.detach().to(torch.int64)
            w = torch.arange(1, es.numel() + 1, device=es.device, dtype=torch.int64)
            return int((es * w).sum())
        es = np.asarray(es, np.int64)
        return int((es * np.arange(1, es.size + 1, dtype=np.int64)).sum())

    def _static_mesh(self, H):
        """`plotting.static_mesh` -- a fixed surface loaded from a FILE, drawn once behind the scene.

        THE GAP THIS FILLS. Every other mesh the renderer draws belongs to a SET: `_mesh_levels`
        finds the half-edge mesh a vertex model builds on `lvl.mesh`, and a point set never has
        one. So a body outline that exists only as an `.obj` on disk -- an organism's own surface,
        segmented once and never simulated -- had no way into the picture at all. A spec could
        name one and nothing happened, silently, because unknown `plotting` keys are accepted.

        It is deliberately STATIC: loaded at the first frame, never updated, no scalars, no
        picking. It is scenery, and treating it as anything more would mean pretending a fixed
        surface is part of a simulation that never touches it.

        The file is resolved through the same roots as every other data path, so a spec names
        `neural_regions/<region>/body.obj` and not an absolute path. Coordinates are taken in the
        run's own world units; a mesh in nanometres against a unit box is invisible, so
        `to_world` gives the affine that maps it in -- the region's `bounds_lo` and `side`.

            plotting:
              static_mesh:
                file: neural_regions/platynereis_larva_4117/body_outline.obj
                color: "#8a93a0"
                opacity: 0.10
                to_world: {origin: [x, y, z], scale: 1.0e-5}
        """
        cfg = (self.style or {}).get("static_mesh")
        if not cfg:
            return
        cfgs = cfg if isinstance(cfg, (list, tuple)) else [cfg]
        for c in cfgs:
            try:
                path = str(c.get("file") or "")
                if not os.path.isabs(path):
                    from plexus.paths import graphs_data_path
                    path = os.path.join(graphs_data_path(), path)
                if not os.path.exists(path):
                    print(f"[live-movie] static_mesh: no file {path}", flush=True)
                    continue
                m = self.pv.read(path)
                # SCALARS OFF. `add_mesh(color=...)` is IGNORED whenever the mesh carries a data
                # array -- pyvista colours by the array instead and the surface came out in the
                # default cyan, opaque-looking, with the requested grey silently dropped. An
                # imported .obj often carries normals or a texture coordinate, so clear them.
                try:
                    m.clear_data()
                except Exception:                                    # noqa: BLE001
                    pass
                tw = c.get("to_world") or {}
                o = np.asarray(tw.get("origin", [0.0, 0.0, 0.0]), float)
                sc = float(tw.get("scale", 1.0))
                if sc != 1.0 or o.any():
                    m.points = (np.asarray(m.points, float) - o) * sc
                _op = float(c.get("opacity", 0.12))
                if _op < 1.0:
                    # DEPTH PEELING, or a translucent surface is not translucent. Without it VTK
                    # blends in draw order, so whatever is drawn after the shell hides behind it
                    # and the cells inside the body simply do not appear.
                    try:
                        self.p.enable_depth_peeling(number_of_peels=8)
                    except Exception:                                # noqa: BLE001
                        pass
                self.p.add_mesh(m, color=str(c.get("color", "#8a93a0")),
                                opacity=_op, scalars=None,
                                smooth_shading=True, specular=0.15, diffuse=0.6, ambient=0.25,
                                lighting=True, show_scalar_bar=False, name=f"static_{os.path.basename(path)}")
                print(f"[live-movie] static_mesh: {os.path.basename(path)} "
                      f"({m.n_points:,} points, {m.n_cells:,} faces) at opacity "
                      f"{float(c.get('opacity', 0.12)):g}", flush=True)
            except Exception as e:                                   # noqa: BLE001
                print(f"[live-movie] static_mesh failed: {type(e).__name__}: {e}", flush=True)

    def _update_meshes(self, H):
        """POINTS ONLY, unless the CONNECTIVITY moved. Swapping the face array every frame on a
        12,000-cell surface would cost more than the rest of the frame, so it is rebound only on
        the events that actually change what a cell's ring contains.

        COUNTING FACES IS NOT ENOUGH, and the operator this misses is the one it most needed to
        catch. `edge_flip` (`ReconnectT1_3D`) is documented as keeping V, E and F fixed -- a T1
        only REWIRES, so `nF` and `Nv` are both unchanged and the old guard passed. The renderer
        then kept frame 0's rings and drew them on the current positions, so every rewired cell
        came out as a self-intersecting star: the zigzag spikes across the face of a spheroid that
        the geometry itself never had (planarity 0.998, zero folded faces, measured on the same
        run). On a 200-cell spheroid a single sweep rewired 916 of 1,188 half-edge slots, so most
        of the tissue was drawn with the wrong polygon.

        So the guard is an ORDER-SENSITIVE CHECKSUM of `E_srce` rather than a per-operator counter:
        `edge_flip` does maintain `n_t1`, but the replay level rebuilds its table from the npz and
        carries no counter at all, so keying on one would have left `-o plot` broken while the live
        path was fixed -- the two-renderers-in-one-name failure again. The weights make it sensitive
        to a permutation, which a plain sum is not, and 72k half-edges of a 12,000-cell mesh peak
        near 1.3e14, well inside int64. One reduction per frame against ~50 ms of VTK.
        """
        for name, nv, pd, sc, ct in getattr(self, "_meshes", []) or []:
            try:
                lvl = H.level(name)
                m = getattr(lvl, "mesh", None)
                if m is None:
                    continue
                # THE SAME OVERLAY THE FIRST FRAME GOT. This is the PER-FRAME path -- the first
                # frame is built in `_mesh_actors` through `_mesh_levels`, every frame after it
                # comes through here -- and it re-fetched the bare mesh table. So a per-cell block
                # that lives on the cell set was visible on frame 0 and gone from frame 1, which
                # renders as the cycle colours appearing once and then freezing.
                m = _MeshView(m, self._cell_cols(H, lvl, int(m["nF"])))
                n_now = int(m["Nv"])
                sig = self._conn_sig(m)
                seen = getattr(self, "_mesh_conn", {})
                if n_now != nv or pd.n_faces_strict != int(m["nF"]) or seen.get(name) != sig:
                    seen[name] = sig
                    self._mesh_conn = seen
                    pd.points = self._mesh_xyz(lvl, n_now, sc, ct)
                    pd.faces = self._mesh_faces(m)
                else:
                    pd.points = self._mesh_xyz(lvl, nv, sc, ct)
                self._edge_actor(H, lvl, m, first=False)
                if self._mesh_is_subject:
                    self._mesh_face_rgb(m, pd, H, lvl)
                self._near_side_faces(pd)
                # THE TRIANGULATED FILL READS `pd` THROUGH A PIPELINE, which re-executes only when
                # `pd` says it changed; a colour array replaced in place does not say so.
                pd.Modified()
            except Exception:                        # noqa: BLE001
                pass

    def _drawn_centre(self, H):
        """The centroid of what this frame actually draws, or None if that cannot be read.

        Used to place the cross-section window. Prefers a mesh set, because on a spec that has both
        a tissue and a gel the section is there for the tissue; falls back to the drawn particles.
        """
        try:
            for _nm, _nv, _pd, _sc, _ct in getattr(self, "_meshes", []) or []:
                q = np.asarray(_pd.points)
                if len(q):
                    return q.mean(0)
            lvl = H.level(self.set_name) if getattr(self, "set_name", None) else None
            if lvl is not None:
                q = np.asarray(lvl.get("pos").detach().cpu().numpy())
                occ = getattr(lvl, "occ", None)
                if occ is not None:
                    q = q[np.asarray(occ.detach().cpu().numpy()) > 0]
                if len(q):
                    return q.mean(0)
        except Exception:                                # noqa: BLE001
            pass
        return None

    def _mono_shell_frame(self, H, name, pd, sc):
        """`(points, vertex normals, vertex thickness, E_srce, E_trgt)` for a monolayer mesh, in the
        SAME coordinates as `pd`. None for every other model, which is the gate.

        The ingredients rather than the two surfaces, because the section needs to offset the
        CROSSING POINTS of the cut plane -- not the vertices -- and that means interpolating the
        normal and the thickness along the crossing edge. Slicing two separately-offset surfaces
        and pairing the results by nearest neighbour is what this replaced; it drew a scribble,
        because `slice` returns points in the order it meets them and the k-th point of one surface
        is not above the k-th of the other.

        Taken from `pd.points` -- the mid-surface actor's own array, already in render coordinates.
        `_mesh_xyz` returns those too, and applying the placement offset to them a second time put
        the shells tens of units outside the window, which is indistinguishable from not drawing.
        """
        try:
            lvl = H.level(name)
            m = getattr(lvl, "mesh", None) or getattr(lvl, "_mesh", None)
            if m is None:
                return None
            # EITHER SPELLING: the operator writes `mono_h` on the live table, and the recorder
            # stores it as a SCALAR, so the replay gets it back as `scalar_mono_h`.
            _h = m.get("mono_h", m.get("scalar_mono_h"))
            # ASK FOR THE BLOCK, NOT THE SCHEMA. A replayed level (`_ReplayLevel`) has no
            # `state_schema` but serves `sep` from the trajectory, so the schema test refused every
            # replayed apico-basal run with no `cell_mechanics` -- no `mono_h` either -- and its
            # section came out empty (demo f1_sheet, a seeded shell with nothing acting on it).
            try:
                _has_sep = lvl.get("sep") is not None
            except Exception:                                            # noqa: BLE001
                _has_sep = False
            # AN APICO-BASAL MESH CARRIES ITS THICKNESS IN `sep` FROM THE SEED; `scalar_mono_h` is
            # written by cell_mechanics from frame 1. Refusing on a missing scalar left the section
            # of a freshly seeded tissue without its rings (the bio page's first picture).
            if _h is None and not _has_sep:
                return None
            import torch as _t
            from plexus.operators.vertex_ops import monolayer_shells
            _np_ = lambda v: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            es, et, ef = _np_(m["E_srce"]), _np_(m["E_trgt"]), _np_(m["E_face"])
            nF = int(m["nF"])
            P = np.asarray(pd.points, np.float64)
            _as = lambda v: _t.as_tensor(np.asarray(v), dtype=_t.long)                # noqa: E731

            # THE APICO-BASAL RUN DRAWS ITS OWN SEPARATION, AND WITHOUT THIS BRANCH IT CANNOT.
            # `monolayer_shells` below is the mid-surface model's KINEMATIC IDENTITY,
            # a, b = x +/- (h/2) n, with ONE scalar thickness for the whole tissue -- which is
            # precisely what `cell_mechanics[model: apicobasal]` exists to remove. The operator
            # still publishes `mono_h`, but as a MEAN over vertices, so a section built from it
            # would draw a uniform shell over a tissue whose whole claim is that the shell is not
            # uniform: a wedged cell, a bottle cell and a flat one would all look identical, and
            # the run would be unfalsifiable by eye. The comment further down says exactly this
            # about the monolayer against a mid-surface run; the same trap returns one level up.
            #
            # `sep` IS A VECTOR AND NOT A THICKNESS, so it is handed back as the pair the caller
            # already interpolates -- a unit direction and a length -- with n = sep/|sep| and
            # hv = 2|sep|. That reproduces apical = pos + sep and basal = pos - sep exactly, per
            # vertex, rather than approximating them along the mid-surface normal. Vertices with no
            # span (an orphan left by an extrusion) fall back to the vertex normal so the section
            # stays drawable; they are the ones `apicobasal_span_zero_fraction` grades.
            _sep = None
            for _b in ("sep",):
                try:
                    _v = lvl.get(_b)
                except Exception:                                        # noqa: BLE001
                    _v = None
                if _v is not None and int(_v.shape[0]) >= P.shape[0]:
                    _sep = _np_(_v)[:P.shape[0]].astype(np.float64)
                    break
            if _sep is not None and np.isfinite(_sep).all() and np.abs(_sep).max() > 0.0:
                # `pd.points` IS THE DRAWN CAP, NOT THE MID-SURFACE. With `mesh_surface: basal` the
                # actor's points are `pos - sep`, and offsetting THEM by +-sep put the section's two
                # rings around the basal cap: "apical" one thickness outside the real basal cap,
                # nuclei that sit between the true caps drawn hugging the inner ring. Undo the
                # drawn cap's offset first so the rings are the caps themselves, pos +- sep.
                _k = {"apical": 1.0, "basal": -1.0, "mid": 0.0}.get(
                    str((self.style or {}).get("mesh_surface", "mid")).lower(), 0.0)
                if _k:
                    P = P - _k * _sep
                _, _, nmid, _hv0 = monolayer_shells(_t.as_tensor(P, dtype=_t.float32),
                                                    _as(es), _as(et), _as(ef), nF,
                                                    _t.ones(nF, dtype=_t.float32))
                nmid = nmid.numpy().astype(np.float64)
                L = np.linalg.norm(_sep, axis=1)
                ok = L > 1e-12
                n = np.where(ok[:, None], _sep / np.maximum(L, 1e-12)[:, None], nmid)
                hv = 2.0 * L * float(sc)
                return (P, n, hv, es.astype(np.int64), et.astype(np.int64))

            if _h is None:
                return None
            h = _t.full((nF,), float(np.asarray(_h).ravel()[0]) * float(sc), dtype=_t.float32)
            _, _, n, hv = monolayer_shells(_t.as_tensor(P, dtype=_t.float32),
                                           _as(es), _as(et), _as(ef), nF, h)
            return (P, n.numpy().astype(np.float64), hv.numpy().astype(np.float64),
                    es.astype(np.int64), et.astype(np.int64))
        except Exception:                                # noqa: BLE001 -- a shell is not the run
            return None

    def _update_cross_section(self, H):
        """Scatter whatever is inside the slab, in the plane of the other two axes."""
        try:
            lvl = H.level(self.set_name) if getattr(self, "set_name", None) else None
            # THE SET `self.idx` INDEXES, which is the one the 3D view draws (`_sname`), before the
            # biggest one: they were the same set until a spec carried a point set larger than its
            # drawn mesh (exp 11's 21,760 interior MPM points beside a 396-vertex layer), and then a
            # vertex subset indexed the MPM set -- "index 24600 is out of bounds for size 21760".
            if lvl is None and getattr(self, "_sname", None):
                lvl = H.level(self._sname)
            if lvl is None:
                from plexus.live_movie import _biggest_particle_set
                lvl = H.level(_biggest_particle_set(H))
            # THE SECTION READS THE SAME PARTICLES THE 3D VIEW DRAWS. `_field` is computed over
            # `self.idx` -- the drawn subset -- so a section built from every particle in the level
            # could not be coloured by it: the two arrays are different lengths, and lining them up
            # by slicing twice in different orders is how a colouring ends up on the wrong points.
            # One index space, used for the positions and the values alike.
            import torch as _t
            I = self.idx if self.idx is not None else _t.arange(int(lvl.n),
                                                                device=lvl.state.device)
            X = lvl.get("pos").detach()[I]
            occ = getattr(lvl, "occ", None)
            live = (occ[I] > 0) if occ is not None else None
            if live is not None:
                X = X[live]
            if X.shape[0] == 0:
                return
            ax, (a, b) = self.cs_axis, self._cs_lat
            dx = float(self.world[ax]) / 96.0
            for fc in getattr(H, "fields", {}).values():
                if hasattr(fc, "dx"):
                    dx = float(fc.dx)
                    break
            # THE PLANE IS A FRACTION OF THE DRAWN BOX, WHICH IS NOT ALWAYS [0, world]. A free run
            # is centred on the origin (`self.lo/hi` = +-world/2), and `cs_at * world` put the plane
            # at +world/2, the far wall: the live section of every free-boundary tissue was empty
            # and only the replay (which shifts positions into [0, world]) ever showed one.
            y0 = float(self.lo[ax]) + self.cs_at * float(self.hi[ax] - self.lo[ax])
            keep = _t.nonzero((X[:, ax] - y0).abs() < self.cs_cells * dx).squeeze(1)
            if keep.numel() > self.cs_max:          # a slab of a big jet is tens of thousands
                keep = keep[:: keep.numel() // self.cs_max + 1]
            P = X[keep]
            # THE SERIES IS REPLACED, NOT UPDATED. `update()` swaps the arrays and the rendered
            # panel keeps whatever it drew first: 81,583 particles spanning the full column were
            # handed over every frame while the picture still showed the inlet sheet from frame 0.
            # `Modified()` on the plot and the chart did not shift it either. Removing the plot and
            # adding a fresh one does, and at <= max_points it is a few thousand values a frame.
            #
            # Counting the array said "live" the whole time, which is why tools/viz_smoke.py
            # compares PIXELS between frames rather than state.
            xs = P[:, a].cpu().numpy()
            ys = P[:, b].cpu().numpy()
            for _s in ([self._cs_series] if not isinstance(self._cs_series, list)
                       else self._cs_series):
                try:
                    self.cs.remove_plot(_s)
                except Exception:                        # noqa: BLE001
                    pass
            # THE SECTION CARRIES THE SAME FIELD AS THE 3D VIEW, IN BANDS. `Chart2D.scatter` takes
            # ONE colour for a whole series, so a per-point colouring is not available -- and a
            # section drawn in flat blue next to a 3D view drawn in `deformation` invites the reader
            # to compare two pictures of different quantities. One series per band of the SAME fixed
            # range is the same LUT, quantised: 12 steps is finer than the eye reads off a colour
            # bar anyway, and it costs twelve chart plots a frame instead of one.
            # EQUAL ASPECT, OR THE SECTION LIES ABOUT SHAPE. `Chart2D` scales its two axes
            # independently to the data, and the slab is 0.70 wide by 0.40 tall -- so a ROUND shell
            # sliced through its middle drew as a flat ellipse while the 3D view beside it showed a
            # sphere. Two pictures of one frame disagreeing about the geometry is the worst kind of
            # artefact: it reads exactly like a physics bug. The ranges are set to a common span
            # about the content's centre instead.
            # CENTRED ON EVERYTHING IN THE PANEL. The range was taken from the PARTICLES alone, so
            # a surface sitting above them -- the spheroid over the gel, which is the whole point of
            # the section -- fell outside it and the content sat low and off-centre in its own box.
            # THE RANGE IS COMPUTED ONCE AND HELD, and getting this wrong is the same mistake the
            # curve panels carry a paragraph about. Taken from the CURRENT frame's content it grows
            # with the spheroid -- 0.039 to 0.150 of the box over the run -- so the panel rescales
            # every frame and a gel that never moves appears to shrink and drift across it. Nothing
            # in a section should move except what is actually moving.
            #
            # THE BOX IS THE FIXED CHOICE, not the first frame's content: it is the same coordinates
            # for every frame by construction, needs no guess about how far the run will get, and is
            # what the 3D view beside it is already drawn in. Equal span on both axes, so a circle
            # renders as a circle.
            #
            # SET ON `x_axis.range`, NOT ON `Chart2D.x_range`: the latter is a convenience the chart
            # RE-DERIVES from its plots whenever one is added, and this panel removes and re-adds
            # every series every frame, so it was overwritten the moment it was set.
            if getattr(self, "_cs_rng", None) is None:
                _l, _h = np.asarray(self.lo, float), np.asarray(self.hi, float)
                _sp = float(max(_h[a] - _l[a], _h[b] - _l[b], 1e-9))
                # `cross_section.span` -- THE WINDOW IN WORLD UNITS, when the box is the wrong size.
                # The box is the right default for a walled run, whose material fills it. It is the
                # wrong one for a `free` run: a vesicle of radius 11 in a 50-unit box is a fifth of
                # the panel, and the section -- the whole point of which is to show a thickness of
                # about 1 -- becomes unreadable. Framing on the CONTENT instead cannot be done live
                # (the tissue grows, and a window that tracks it makes the scale jump every frame),
                # so the spec declares the window once and it is fixed for the clip like the rest.
                _decl = (self.style or {}).get("cross_section", {}).get("span")
                if _decl:
                    _sp = float(_decl)
                _cx, _cy = 0.5 * (_l[a] + _h[a]), 0.5 * (_l[b] + _h[b])
                # CENTRED ON THE CONTENT, NOT ON THE BOX, and measured ONCE so the window is still
                # fixed for the clip. The box centre is right for a walled run whose material fills
                # it, and wrong for a `free` one: this renderer draws the box at [0, w] on the
                # replay path and a vesicle is built about the ORIGIN, so the section framed
                # x 11..39 around a tissue living at x -11..11 -- a slice of empty space beside the
                # cells. Reading the centre off what is actually drawn fixes it for both paths and,
                # unlike moving the drawing box, leaves the 3D camera alone.
                # `cross_section.offset` -- MOVE THE WINDOW OFF THE CONTENT CENTRE, in world units.
                # A section that frames the whole object proves its SHAPE and cannot show its wall:
                # an epithelium of thickness 0.35 on a shell of radius 18.7 is 1.9% of the frame,
                # about one pixel, so the apical and basal curves land on each other and the thing
                # the section exists to show is the thing it cannot resolve. Offsetting lets a spec
                # put a small window on the wall instead of a large one on the whole shell.
                _off = (self.style or {}).get("cross_section", {}).get("offset") or (0.0, 0.0)
                _ctr = self._drawn_centre(H)
                if _ctr is None:
                    # NOT CACHED YET. The window is computed ONCE and kept, so computing it on a
                    # frame where the mesh actor does not exist yet would pin the box centre for the
                    # whole clip -- which is what happened on the replay path: the live pass framed
                    # x -14..14 and the replay, which writes the movie that is kept, framed x 11..39
                    # around a tissue living at x -11..11. Leaving `_cs_rng` unset retries next frame.
                    return
                _cx = float(_ctr[a]) + float(_off[0]); _cy = float(_ctr[b]) + float(_off[1])
                self._cs_rng = ([_cx - _sp / 2, _cx + _sp / 2], [_cy - _sp / 2, _cy + _sp / 2])
                print(f"[live-movie] cross section: axes fixed to "
                      f"{'the declared span' if _decl else 'the box'}, "
                      f"{'xyz'[a]} {self._cs_rng[0][0]:.3g}..{self._cs_rng[0][1]:.3g}  "
                      f"{'xyz'[b]} {self._cs_rng[1][0]:.3g}..{self._cs_rng[1][1]:.3g}", flush=True)
            # "fixed" IS A STRING HERE, not an enum -- pyvista validates against {"auto","fixed"}
            # and the enum I reached for raised, which the panel's own guard turned into
            # "cross section unavailable" and no section at all for the whole run. `behavior` is
            # what stops the chart re-deriving the range when a series is removed and re-added,
            # which this panel does on every frame.
            for _a in (self.cs.x_axis, self.cs.y_axis):
                _a.behavior = "fixed"
            self._cs_series = []
            # `cross_section.style: outlines` (exp 11, 2026-09-28; default = the scatter/walls drawing below): the section
            # drawn as a histological one -- the outlines of the apico-basal cells where the plane cuts their prisms and
            # of each interior cell (`cross_section.interior: <set with a pid block>`), from the density of ALL its
            # particles, in white; the membrane's nodes in the slab (`cross_section.membrane: <set>`) as orange dots.
            # `plexus.section_render` does the drawing; nothing else in the panel runs.
            _csx = (self.style or {}).get("cross_section", {}) or {}
            if str(_csx.get("style", "")).lower() == "outlines":
                for _s in getattr(self, "_cs_mesh_series", []) or []:
                    try:
                        self.cs.remove_plot(_s)
                    except Exception:                    # noqa: BLE001
                        pass
                self._cs_mesh_series = []
                from plexus import section_render as _SR
                from plexus.models.topology import rings_from_flat_3d as _rf
                _vl = H.level(self.set_name) if getattr(self, "set_name", None) else lvl
                _m = getattr(_vl, "_mesh", None)
                if _m is None:
                    _mm = getattr(_vl, "mesh", None)     # the replay level serves its frame's mesh (a property)
                    _m = _mm() if callable(_mm) else _mm
                try:
                    _sep = _vl.get("sep")
                except Exception:                        # noqa: BLE001
                    _sep = None
                if _m is not None and _sep is not None:
                    _nF = int(_m["nF"]); _Nv = int(_m["Nv"])
                    _tn = lambda v: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))  # noqa: E731
                    _Pv = _tn(_vl.get("pos"))[:_Nv].astype(np.float64)
                    _Sv = _tn(_sep)[:_Nv].astype(np.float64)
                    _rings = _rf(_tn(_m["E_srce"]), _tn(_m["E_trgt"]), _tn(_m["E_face"]), _nF)
                    _pts, _lab = _SR.prism_cloud(_Pv, _Sv, _rings, _nF)
                else:
                    _pts, _lab = np.zeros((0, 3)), np.zeros(0, int)
                _Q = np.zeros((0, 3)); _qc = np.zeros(0, int)
                _inn = _csx.get("interior")
                if _inn:
                    try:
                        _il = H.level(_inn); _n2 = lambda v: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))  # noqa: E731
                        _o = _n2(_il.occ).reshape(-1) > 0.5
                        _Q = _n2(_il.get("pos"))[_o].astype(np.float64)
                        _qc = np.rint(_n2(_il.get("pid")).reshape(-1)[_o]).astype(int)
                    except Exception:                    # noqa: BLE001
                        pass
                _xr, _yr = self._cs_rng
                if os.environ.get("PLEXUS_CS_DEBUG") and not getattr(self, "_cs_dbg", False):
                    self._cs_dbg = True
                    print(f"[cs-outlines] set_name {getattr(self, 'set_name', None)} _sname {getattr(self, '_sname', None)} "
                          f"vl {type(_vl).__name__} m {None if _m is None else list(_m)[:6]} sep {None if _sep is None else getattr(_sep, 'shape', None)}",
                          flush=True)
                    print(f"[cs-outlines] y0 {y0:.3g} ax {ax} a {a} b {b} xr {_xr} yr {_yr} pts {len(_pts)} "
                          f"P range {(_pts.min(0), _pts.max(0)) if len(_pts) else None} Q {len(_Q)} "
                          f"Q range {(_Q.min(0), _Q.max(0)) if len(_Q) else None} qc {len(np.unique(_qc))} cells", flush=True)
                _npx = int(_csx.get("resolution", 360))
                _L = _SR.section_labels(_pts, _lab, _Q, _qc, ax, a, b, y0, _xr, _yr, _npx)
                _bd = _SR.borders(_L)
                _px = (_xr[1] - _xr[0]) / _npx
                _iy, _ix = np.nonzero(_bd)
                if os.environ.get("PLEXUS_CS_DEBUG"):
                    print(f"[cs-outlines] labels {(int((_L >= 0).sum()))} border px {len(_ix)}", flush=True)
                if len(_ix):
                    try:
                        self._cs_series.append(self.cs.scatter(_xr[0] + (_ix + 0.5) * _px, _yr[0] + (_iy + 0.5) * _px,
                                                               size=float(_csx.get("line_px", 2.5)), style="o",
                                                               color=(235, 235, 235, 255)))
                    except Exception as _e:              # noqa: BLE001
                        print(f"[cs-outlines] scatter failed: {type(_e).__name__}: {_e}", flush=True)
                _mem = _csx.get("membrane")
                if _mem:
                    try:
                        _ml = H.level(_mem); _n3 = lambda v: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))  # noqa: E731
                        _o = _n3(_ml.occ).reshape(-1) > 0.5
                        _B = _n3(_ml.get("pos"))[_o]
                        _sl = np.abs(_B[:, ax] - y0) < 0.30
                        if _sl.any():
                            self._cs_series.append(self.cs.scatter(_B[_sl, a], _B[_sl, b], size=3.0, style="o",
                                                                   color=(255, 159, 28, 255)))
                    except Exception:                    # noqa: BLE001
                        pass
                # the axes last, as the default path does (a series added after the range resets it)
                for _a2, _r2 in ((self.cs.x_axis, self._cs_rng[0]), (self.cs.y_axis, self._cs_rng[1])):
                    _a2.label_visible = _a2.ticks_visible = _a2.tick_labels_visible = False
                    _a2.grid = False; _a2.range = _r2; _a2.behavior = "fixed"
                return
            fld = str(self.style.get("color_field", "") or "")
            val = self._field(H, lvl)[0] if fld else None
            # `cross_section.points: false` -- DO NOT SCATTER THE DRAWN SET'S PARTICLES IN THE
            # SECTION. On a MESH run those particles are the mesh's own vertices, which on this
            # promotion are the MID-SURFACE -- so a section that already draws apical, basal and
            # the walls gets the mid-surface back a second time, as a dotted line down the middle
            # of the band, after `cross_section.mid: false` removed the curve -- first a parallel
            # white ring, then blue dashes behind it.
            #
            # It is a key rather than a rule because on a PARTICLE run the scatter is the section:
            # an MPM slab has nothing else in it, and every existing cross_section spec is one of
            # those. Default true, so none of them moves.
            # `cross_section.spheres: true` -- THE SECTION CUTS THE SPHERES. A piece drawn as a
            # sphere of world radius r (`plotting.dot_radius`) that sits at distance d from the
            # section plane shows in the section as a DISC of radius sqrt(r^2 - d^2), in its own
            # type's colour; a piece farther than r from the plane is not cut and not drawn. That
            # is what a histological section of a nucleus looks like, and it is the only way the
            # inset can say whether the nucleus fits inside the cell's two caps. Discs are binned
            # by radius into a few scatter series per type (a chart scatter has one size).
            _cs_sty = (self.style or {}).get("cross_section", {}) or {}
            _rad = (self.style or {}).get("dot_radius") or {}
            if _cs_sty.get("spheres") and _rad and getattr(lvl, "node_type", None) is not None:
                from matplotlib.colors import to_rgb
                names = list(getattr(lvl, "type_names", []) or [])
                pal = (self.style or {}).get("colors") or {}
                ntI = _t.as_tensor(lvl.node_type)[I].to(X.device)
                ntI = ntI[live] if live is not None else ntI
                Xall = lvl.get("pos").detach()[I]
                Xall = Xall[live] if live is not None else Xall
                dplane = (Xall[:, ax] - y0).abs()
                _span = float(self._cs_rng[0][1] - self._cs_rng[0][0])
                _px_per_unit = 0.85 * float(self.p.window_size[1]) * float(
                    (self.style or {}).get("cross_section_height", 0.24)) / max(_span, 1e-9)
                for tid, nm in enumerate(names):
                    r = _rad.get(nm)
                    if r is None:
                        continue
                    cut = (ntI == tid) & (dplane < float(r))
                    if not bool(cut.any()):
                        continue
                    disc = (float(r) ** 2 - dplane[cut] ** 2).clamp_min(0.0).sqrt().cpu().numpy()
                    Pc = Xall[cut]
                    xc, yc = Pc[:, a].cpu().numpy(), Pc[:, b].cpu().numpy()
                    col = to_rgb(tuple(pal[nm]) if isinstance(pal.get(nm), (list, tuple)) else pal[nm]) if nm in pal else (0.9, 0.9, 0.9)
                    col = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255), 255)
                    nb = 6
                    q = np.clip((disc / float(r) * nb).astype(int), 0, nb - 1)
                    for k in range(nb):
                        m_ = q == k
                        if not m_.any():
                            continue
                        pxs = max(1.0, 2.0 * (k + 0.5) / nb * float(r) * _px_per_unit)
                        self._cs_series.append(self.cs.scatter(xc[m_], yc[m_], size=pxs, style="o", color=col))
            # `cross_section.also_sets: [set, ...]` -- OTHER POINT SETS IN THE SLICE (exp 11, 2026-09-27: the interior's
            # cells inside a stratified bud). Each set's live points within the slab are scattered in its `also_colors`
            # colour (`cross_section.also_size` px, default 4), drawn before the subject's own scatter. Opt-in.
            for _nm in (_cs_sty.get("also_sets") or []):
                try:
                    from matplotlib.colors import to_rgb as _to_rgb
                    _lv = H.level(str(_nm))
                    _P = _lv.get("pos").detach()
                    _oc = getattr(_lv, "occ", None)
                    if _oc is not None:
                        _P = _P[_t.as_tensor(_oc).to(_P.device).reshape(-1)[:_P.shape[0]] > 0.5]
                    # `cross_section.also_halfwidth` -- THE SLAB'S HALF-WIDTH FOR THESE SETS, in world units (default
                    # the subject's `thickness * dx`). At thickness 6 in a 200-unit world that is 12.5 units, the whole
                    # interior ball: every point was projected, not a slice (the human, 2026-09-27, "too many for cells").
                    _hw = float(_cs_sty.get("also_halfwidth", self.cs_cells * dx))
                    _in = (_P[:, ax] - y0).abs() < _hw
                    # `cross_section.also_color_by: {set: <block>}` -- A COLOUR PER VALUE OF THAT BLOCK (a cell
                    # label, 0..19, `tab20`), one scatter series per value, so the interior's cells read as
                    # cells in the section rather than as one colour of dots (exp 11, 2026-09-27). Opt-in.
                    _lab = None
                    _cb = (_cs_sty.get("also_color_by") or {}).get(str(_nm))
                    if _cb:
                        _Lr = _lv.get(str(_cb))
                        _L = (_Lr.detach() if hasattr(_Lr, "detach") else _t.as_tensor(np.asarray(_Lr))).reshape(-1)
                        _L = _L.to(_P.device)[: _lv.get("pos").shape[0]]
                        if _oc is not None:
                            _L = _L[_t.as_tensor(_oc).to(_L.device).reshape(-1)[:_L.shape[0]] > 0.5]
                        _lab = _L[_in].round().long().cpu().numpy()
                    _P = _P[_in]
                    if _P.shape[0] and _lab is not None:
                        import matplotlib as _mpl
                        _cm = _mpl.colormaps["tab20"]
                        _xa, _xb = _P[:, a].cpu().numpy(), _P[:, b].cpu().numpy()
                        for _v in sorted(set(int(v) for v in _lab)):
                            _sel = _lab == _v
                            _c = _cm((_v % 20) / 19.0)
                            self._cs_series.append(self.cs.scatter(_xa[_sel], _xb[_sel],
                                                                   size=float(_cs_sty.get("also_size", 4)), style="o",
                                                                   color=(int(_c[0] * 255), int(_c[1] * 255),
                                                                          int(_c[2] * 255), 255)))
                    elif _P.shape[0]:
                        _c = ((self.style or {}).get("also_colors") or {}).get(str(_nm), "#ffffff")
                        _c = _to_rgb(_c)
                        self._cs_series.append(self.cs.scatter(_P[:, a].cpu().numpy(), _P[:, b].cpu().numpy(),
                                                               size=float(_cs_sty.get("also_size", 4)), style="o",
                                                               color=(int(_c[0] * 255), int(_c[1] * 255),
                                                                      int(_c[2] * 255), 255)))
                except Exception as _e:                              # noqa: BLE001 -- not the movie
                    print(f"[live-movie] cross_section.also_sets: {_nm!r} not drawn ({type(_e).__name__}: {_e})",
                          flush=True)
            if _cs_sty.get("points", True) is False:
                pass                                  # neither branch: no scatter at all
            elif val is not None:
                import matplotlib.pyplot as _plt
                v = val.detach()
                v = (v[live] if live is not None else v)[keep].float().cpu().numpy()
                rng = getattr(self, "_frng", None) or self.style.get("color_range")
                lo, hi = ((float(rng[0]), float(rng[1])) if rng and len(rng) == 2
                          else (float(np.nanmin(v)), float(np.nanmax(v))))
                nb = int(self.style.get("cross_section_bands", 12))
                cm = _plt.get_cmap(self.style.get("field_cmap", "turbo"))
                q = np.clip(((v - lo) / max(hi - lo, 1e-12) * nb).astype(int), 0, nb - 1)
                for k in range(nb):
                    m_ = q == k
                    if not m_.any():
                        continue
                    c = cm((k + 0.5) / nb)
                    self._cs_series.append(self.cs.scatter(
                        xs[m_], ys[m_], size=self._cs_size, style="o",
                        color=(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255)))
            else:
                self._cs_series.append(self.cs.scatter(xs, ys, size=self._cs_size, style="o",
                                                       color=self._cs_colour))
            # AND THE SURFACE'S PROFILE THROUGH THE SAME SLAB. The section exists because a
            # compressed slab is an opaque silhouette from outside; leaving the indenter out of it
            # would show the dimple with nothing making it. Vertices inside the slab, ordered along
            # the in-plane axis -- for a plate that is its own cross-section, exactly.
            for _s in getattr(self, "_cs_mesh_series", []) or []:
                try:
                    self.cs.remove_plot(_s)
                except Exception:                        # noqa: BLE001
                    pass
            self._cs_mesh_series = []
            _mc = (self.style or {}).get("mesh_color", "#e6dcc0")
            # A TRUE PLANE SLICE OF THE SURFACE, NOT ITS VERTICES SORTED BY x.
            #
            # Taking the vertices inside the slab and joining them in order of one in-plane
            # coordinate is right for a PLATE, whose slab-band is a single row, and wrong for
            # anything closed: a sphere's band is a whole belt of vertices, so x-ordering zigzags
            # back and forth across it and fills the disc in solid -- which is what a 400-face
            # sphere drew. `slice` intersects the polygons with the plane and returns the curve
            # itself; `strip` then joins the segments into ordered polylines, so one series per
            # closed loop and no ordering to invent.
            _nrm = [0.0, 0.0, 0.0]; _nrm[ax] = 1.0
            _org = [0.0, 0.0, 0.0]; _org[ax] = y0
            for _nm, _nv, _pd, _sc, _ct in getattr(self, "_meshes", []) or []:
                try:
                    _sl = _pd.slice(normal=_nrm, origin=_org)
                    if _sl.n_points < 3:
                        continue
                    # ONE CLOSED CURVE, ORDERED BY ANGLE. `slice` returns the intersection as
                    # hundreds of unordered SEGMENTS -- 7,000 faces cut by a plane -- and `strip`
                    # joins only those that already share endpoints, so a shell came out as dozens
                    # of polylines drawn as dozens of series: a scribble where the outline should
                    # be, and dozens of chart plots a frame. A section of a star-shaped shell is a
                    # closed curve about its own centre, so sorting the points by angle recovers it
                    # exactly, in one series, and closes it by repeating the first point.
                    _P = np.asarray(_sl.points)
                    _u, _v = _P[:, a], _P[:, b]
                    _th = np.arctan2(_v - _v.mean(), _u - _u.mean())
                    _o = np.argsort(_th)
                    _o = np.append(_o, _o[0])
                    # `cross_section.mid: false` -- DO NOT DRAW THE MID-SURFACE AT ALL, and on an
                    # apico-basal run that is the honest default to reach for. `pos` is not a
                    # membrane: the design defines apical = pos + sep and basal = pos - sep, so
                    # `pos` is IDENTICALLY the midpoint of the two caps (measured on
                    # gate_ab_curved: max |pos - (apical+basal)/2| = 0.000e+00 over every vertex).
                    # It can never be anywhere else and it carries no information the other two
                    # curves do not already have -- that redundancy IS the change of variables.
                    # Drawn between them it reads as a third surface the cell does not have, which
                    # is the first thing a viewer questions.
                    #
                    # IT IS NOT ALWAYS MERELY BOOKKEEPING, which is why this is a key and not a
                    # deletion: `gamma` and `Lambda` act on the mid-surface RING and `K_R` on
                    # |pos|, so on a spec that sets any of them the curve is where a real force
                    # lives and belongs in the picture. On a spec with all three at zero it is a
                    # parameterisation and nothing more.
                    #
                    # `cross_section.mid_color` dims it instead of removing it; the default is
                    # `mesh_color`, so every section that has no shells is unchanged.
                    if (self.style or {}).get("cross_section", {}).get("mid", True) is False:
                        continue
                    _midc = ((self.style or {}).get("cross_section", {}).get("mid_color") or _mc)
                    self._cs_mesh_series.append(
                        self.cs.line(_u[_o], _v[_o], color=_midc, width=2.0))
                except Exception:                        # noqa: BLE001 -- the section is not the run
                    pass
            # THE THICKNESS, WHERE THE THICKNESS IS THE POINT.
            #
            # `cell_mechanics[model: monolayer]` gives every cell a 3D volume with apical, basal and
            # lateral surfaces, but the mesh it stores and the renderer draws is the MID-surface --
            # so a monolayer run and a mid-surface run produce the SAME picture, and the one thing
            # the thick model adds is the one thing that cannot be seen. The shells are rebuilt here
            # through `monolayer_shells`, the function the energy itself is written on, so the
            # drawing cannot drift from the model, and sliced by the same plane as the mid-surface.
            # Red outside, blue inside: two distinct surfaces, not a measurement against a truth.
            for _nm, _nv, _pd, _sc, _ct in getattr(self, "_meshes", []) or []:
                # THE WALL, BUILT THE WAY THE REFERENCE SECTION BUILDS IT -- and it took reading
                # `discovery_okuda/ops/run_tyssue_round.py::_cross_screen` to get right, after three
                # attempts at inferring it from the picture.
                #
                # That function finds the mesh edges that CROSS the plane, keeps the crossing point,
                # and draws a quad from each crossing point to the next, spanning outward surface to
                # inward surface. Its two surfaces are the crossing point X and `X * inner` with
                # `inner = 0.82` -- so the wall in every okuda_ECM section is 18% of the radius
                # because a renderer constant says so, with no basal surface and no thickness in the
                # model at all. That is why a MID-SURFACE run shows a thick banded ring, and why a
                # faithful apical/basal drawing of a real monolayer looks thin beside it: h0/R here
                # is 2-4%, not 18%.
                #
                # Same construction, real thickness. The crossing points are offset along the
                # INTERPOLATED VERTEX NORMAL by the model's own h/2 -- `monolayer_shells` supplies
                # both -- so the ticks are radial by construction rather than by nearest-neighbour
                # matching between two separately sliced surfaces, which is what made the earlier
                # attempt a scribble.
                try:
                    _sh = self._mono_shell_frame(H, _nm, _pd, _sc)
                    # `cross_section.inner` -- THE REFERENCE'S COSMETIC BAND, declared as cosmetic.
                    # `_cross_screen` draws its inner surface as `X * inner` with `inner = 0.82`, so
                    # the wall in every okuda_ECM section is 18% of the radius whatever the model
                    # says, and a MID-SURFACE run -- which has no thickness at all -- produces the
                    # thick banded ring. Offered here so those figures can be reproduced, and named
                    # so nobody reads the band as a measurement: a spec that sets `inner` is asking
                    # for a drawing, and one that runs a monolayer gets the model's own h instead.
                    _inner = (self.style or {}).get("cross_section", {}).get("inner")
                    if _sh is None and _inner and getattr(self, "_meshes", None):
                        _P = np.asarray(_pd.points, np.float64)
                        _mm = getattr(H.level(_nm), "mesh", None) or getattr(H.level(_nm), "_mesh", None)
                        if _mm is not None:
                            _np_ = lambda v: (v.detach().cpu().numpy() if hasattr(v, "detach")
                                              else np.asarray(v))
                            _e0, _e1 = _np_(_mm["E_srce"]).astype(np.int64), _np_(_mm["E_trgt"]).astype(np.int64)
                            _c0 = _P.mean(0)
                            _rad = _P - _c0
                            _hv = (1.0 - float(_inner)) * np.linalg.norm(_rad, axis=1)
                            _nn = _rad / (np.linalg.norm(_rad, axis=1, keepdims=True) + 1e-12)
                            _sh = (_P, _nn, _hv, _e0, _e1)
                    if _sh is not None:
                        _P, _nrmv, _hv, _es0, _es1 = _sh
                        _pr = _P[:, ax] - y0
                        _c = np.flatnonzero(_pr[_es0] * _pr[_es1] < 0)
                        if len(_c) > 3:
                            _s0, _s1 = _es0[_c], _es1[_c]
                            _f = (-_pr[_s0] / (_pr[_s1] - _pr[_s0]))[:, None]
                            _X = _P[_s0] + _f * (_P[_s1] - _P[_s0])
                            _N = _nrmv[_s0] + _f * (_nrmv[_s1] - _nrmv[_s0])
                            _N = _N / (np.linalg.norm(_N, axis=1, keepdims=True) + 1e-12)
                            _h = (_hv[_s0] + _f[:, 0] * (_hv[_s1] - _hv[_s0]))[:, None]
                            _ap = _X + 0.5 * _h * _N
                            _ba = _X - 0.5 * _h * _N
                            _o = np.argsort(np.arctan2(_X[:, b] - _X[:, b].mean(),
                                                       _X[:, a] - _X[:, a].mean()))
                            _ap, _ba = _ap[_o], _ba[_o]
                            _cl = np.append(np.arange(len(_o)), 0)
                            # RED/BLUE ONLY WHEN THEY ARE TWO REAL SURFACES. Apical and basal are
                            # two distinct sources and get the two-source colours; a cosmetic
                            # `inner` band is ONE surface drawn twice, so colouring it as two would
                            # claim a measurement the run does not have.
                            _ca, _cb = ("#d9534f", "#4a86c8") if not _inner else (
                                (self.style or {}).get("mesh_color", "#e6dcc0"),) * 2
                            self._cs_mesh_series.append(self.cs.line(
                                _ap[_cl, a], _ap[_cl, b], color=_ca, width=2.0))
                            self._cs_mesh_series.append(self.cs.line(
                                _ba[_cl, a], _ba[_cl, b], color=_cb, width=2.0))
                            _n = int((self.style or {}).get("cross_section", {}).get("walls", 0))
                            if _n > 0:
                                # THE WALLS GET THEIR OWN COLOUR, BECAUSE THE SECTION DRAWS THREE
                                # OBJECTS AND USED TO HAVE TWO COLOURS FOR THEM. The mid-surface
                                # slice above is drawn in `mesh_color` and the walls were too, so a
                                # reader saw a red curve, a blue curve, a white curve BETWEEN them
                                # and white ticks ACROSS them, with the ring and the ticks claiming
                                # to be the same thing. They are not: the ring is `pos`, the
                                # incumbent's only surface, and the ticks are `2|sep|`, which is
                                # what this promotion added. Coloured alike, the parallel white ring
                                # reads as a bug -- it is not, it is the mid-surface, and the
                                # colouring is what makes that a question.
                                #
                                # Grey rather than another hue: red and blue are already spoken for
                                # by apical and basal, which are two distinct SOURCES, and a third
                                # saturated colour would imply a third surface of the same kind.
                                _wc = ((self.style or {}).get("cross_section", {}).get("wall_color")
                                       or "#8a8a8a")
                                for _k in np.unique(np.linspace(0, len(_o) - 1,
                                                                min(_n, len(_o))).astype(int)):
                                    self._cs_mesh_series.append(self.cs.line(
                                        np.array([_ba[_k, a], _ap[_k, a]]),
                                        np.array([_ba[_k, b], _ap[_k, b]]),
                                        color=_wc, width=1.0))
                except Exception:                        # noqa: BLE001 -- the section is not the run
                    pass
            # LAST, AFTER EVERY SERIES IS BACK. Adding a plot re-derives the chart's range from its
            # data, so a range set before the series were added was overwritten by the last `line`
            # call and the panel autoscaled to the data's own extent -- wider than tall, which drew
            # a spherical shell as an ellipse next to a 3D view showing a sphere.
            #
            # AND THE AXES ARE OFF. Their labels and ticks take space on the left and bottom, so the
            # PLOT AREA is not square even when the panel is, and equal ranges still render
            # stretched. A section is a picture of a shape; the numbers on it are box coordinates
            # nobody reads.
            for _a, _r in ((self.cs.x_axis, self._cs_rng[0]), (self.cs.y_axis, self._cs_rng[1])):
                _a.label_visible = _a.ticks_visible = _a.tick_labels_visible = False
                _a.grid = False
                _a.range = _r
                _a.behavior = "fixed"
        except Exception as e:                       # noqa: BLE001 -- a panel must never kill a run
            if not getattr(self, "_cs_warned", False):
                self._cs_warned = True
                print(f"[live-movie] cross section unavailable ({type(e).__name__}: {e})", flush=True)

    def _field(self, H, lvl):
        """A per-particle SCALAR to colour by, recomputed every frame. None when not asked for.

        WHY VORTICITY IS THE DEFAULT AND NOT PRESSURE. `mpm_particle.C` is the affine velocity
        GRADIENT the MLS transfer already carries, so its antisymmetric part is curl(v) exactly --
        no extra state, no extra pass. Vortex cores and shear layers are what |omega| lights up, and
        those are what "turbulence" means to look at. Pressure and deformation are the SAME field
        for an MPM liquid (mu = 0, so stress is isotropic and p = K(1-J)), and in a tall column they
        are dominated by the hydrostatic ramp -- 0.88% top to bottom on si_ball_wake -- so a wake
        worth 1e-4 of J sits invisible on top of it unless rho*g*h is subtracted first.

        THE RANGE IS FIXED, NOT PER-FRAME. Auto-scaling each frame makes a colour mean a different
        number in every frame, so a brightening wake could be the wake growing or the rest of the
        field calming down, and the movie cannot tell you which. `plotting.color_range: [lo, hi]`.
        """
        want = str(self.style.get("color_field", "") or "").lower()
        if not want:
            return None, ""
        # AN UNKNOWN NAME RAISES. It used to return None, which is the same answer as "no
        # `color_field` was asked for" -- so `color_field: deformation` was accepted, ignored, and
        # the cloud fell back to `_rgb`'s HEIGHT RAMP, which is fixed at t = 0 and carried with the
        # particle. The movie then showed a smooth red-to-blue gradient that never changed and
        # looked exactly like a field, on a run whose whole subject is a deformation. A colour that
        # looks like data and is not is worse than no colour, and this is the one line that decides
        # which of the two a typo produces.
        import torch
        idx = self.idx
        # A DECLARED SCALAR BLOCK IS A FIELD TOO. A neuron's `voltage`, a metabolite's `conc`: the
        # quantity the run is about is a column of the set's own state, declared in its schema,
        # and colouring by it needs no solver buffer -- so it is available on replay as well.
        # Looked up on the schema BEFORE the derived names below, and only for width-1 blocks,
        # because a vector block has no single colour.
        _sch = getattr(lvl, "state_schema", None)
        if _sch is not None and want in _sch and want not in _FIELDS:
            a, b = _sch[want]
            if b - a == 1:
                return lvl.state[idx, a], want
        if want not in _FIELDS:
            raise ValueError(f"plotting.color_field: {want!r} is not one of "
                             f"{', '.join(sorted(_FIELDS))} nor a scalar block of {lvl.name!r}")
        # A REPLAY HAS NO DEFORMATION GRADIENT. `trajectory.npz` stores positions, occupancy and the
        # mesh; `F` and `C` are solver state and are not recorded (9 floats per particle per frame
        # would be 1.8 GB on this run alone). Returning None here reads to the caller as "no colour
        # was asked for", and the answer to that is the HEIGHT RAMP fixed at t=0 -- so `-o plot`
        # quietly replaced a strain-coloured movie with a static gradient and said so in one line.
        if want in ("deformation", "strain", "volume", "pressure", "vorticity") \
                and getattr(lvl, "F", None) is None and getattr(lvl, "C", None) is None:
            raise ValueError(
                f"plotting.color_field: {want!r} needs the per-particle deformation gradient, which "
                f"a trajectory does not store -- `-o plot` cannot draw it and must not overwrite a "
                f"correct movie with a height ramp. Render from `-o generate`, or use `speed`, "
                f"which is computed from the recorded velocities.")
        if want == "speed":
            # THE UNIT IS WORLD UNITS PER SECOND AND THE LABEL USED TO SAY `m/s`, which is wrong by
            # the length unit -- 5e-5 on a run whose world unit is 50 um. That is not cosmetic: the
            # legend is the only place a movie states its scale, so `color_range: [0, 8.66e-06]`
            # was read as 8.66 um/s when it is really 0.00043 um/s, and a field whose median is
            # 0.00056 um/s came out uniformly saturated while the number in the caption said there
            # was headroom to spare. The VALUE is left in world units so the 87 specs that already
            # declare a `color_range` keep meaning what they meant; only the label is corrected,
            # with the physical equivalent appended when the run declares a length unit.
            v = lvl.get("vel")[idx]
            _lu = getattr(self, "length_um", None)
            return v.norm(dim=1), ("|v| (world/s" + (f", x{_lu:g} = um/s)" if _lu else ")"))
        if want == "radial":
            # THE SIGNED OUTWARD COMPONENT, so a diverging colour map means something. `speed` is a
            # magnitude and can only ever be drawn on a one-sided ramp, which cannot answer the
            # question actually being asked of these runs -- is the cilium PUMPING fluid away, or
            # just shaking it back and forth in place? This is v . rhat about the scene centre,
            # where the cilium stands: positive is outward, negative inward, and zero is white on a
            # blue-white-red map. A reciprocal stroke averages to zero here however fast |v| is,
            # which is exactly the distinction Purcell's scallop theorem turns on.
            v = lvl.get("vel")[idx]
            x = lvl.get("pos")[idx]
            c = 0.5 * (torch.as_tensor(np.asarray(self.lo), device=v.device, dtype=v.dtype)
                       + torch.as_tensor(np.asarray(self.hi), device=v.device, dtype=v.dtype))
            r = x - c[None, :]
            rn = r.norm(dim=1, keepdim=True).clamp_min(1e-12)
            _lu = getattr(self, "length_um", None)
            return (v * r / rn).sum(1), ("outward v (world/s"
                                         + (f", x{_lu:g} = um/s)" if _lu else ")"))
        if want == "vorticity":
            C = lvl.C[idx]                                   # [N,3,3] velocity gradient
            w = torch.stack([C[:, 2, 1] - C[:, 1, 2],
                             C[:, 0, 2] - C[:, 2, 0],
                             C[:, 1, 0] - C[:, 0, 1]], 1)    # curl(v) = 2 * antisym(C)
            return w.norm(dim=1), "|curl v| (1/s)"
        F = lvl.F[idx].float()
        J = torch.linalg.det(F)                              # volume ratio
        if want in ("deformation", "strain"):
            # THE SAME SCALAR `ecm_stress[measure: dev]` BANDS, and deliberately the same lines:
            # the volume-normalised left Cauchy-Green tensor's deviator, i.e. SHAPE change with the
            # volume change divided out. Two readings of one F that disagreed would be worse than
            # one, and this is what a colour called "deformation" has to mean if the run is also
            # allowed to quote `ecm_stress`.
            #
            # WHY NOT |J-1| UNDER THIS NAME. A fixed-corotated MPM solid resists volume change
            # stiffly, so an indented gel is SHEARED far more than it is compressed: |J-1| stays
            # near zero while the material is visibly flowing around the plate. That reading is
            # available as `volume`, named for what it is.
            B = F @ F.transpose(-1, -2)
            Bb = B / J.abs().clamp_min(1e-9).pow(2.0 / 3.0)[:, None, None]
            tr = Bb.diagonal(dim1=-2, dim2=-1).sum(-1)
            eye = torch.eye(Bb.shape[-1], device=Bb.device, dtype=Bb.dtype)
            dev = Bb - (tr / 3.0)[:, None, None] * eye
            return torch.sqrt((1.5 * (dev * dev).sum((-1, -2))).clamp_min(0.0)), "equiv. dev. strain"
        if want == "volume":
            return (J - 1.0).abs(), "|J - 1|"
        K = float(self.style.get("pressure_K", 3.0e6))
        return K * (1.0 - J), "p = K(1-J) (Pa)"

    @staticmethod
    def _block_unit(lvl, name):
        """The `unit:` a set declared for one of its state blocks, or None.

        THROUGH THE SCHEMA, NOT A NAME TABLE. U1 put the declaration on `Block.unit`, so this reads
        the same object the checker and the engine read; a renderer-local mapping from block name to
        unit would be a second declaration and therefore a second chance to disagree.
        """
        try:
            sch = getattr(lvl, "state_schema", None)
            return getattr(sch.block(name), "unit", None) if sch and name in sch else None
        except Exception:                        # noqa: BLE001 -- a legend is not the run
            return None

    def _rgb_field(self, H, lvl):
        """Map `_field` through a colormap with a FIXED range -> uint8 RGB, or None."""
        val, label = self._field(H, lvl)
        if val is None:
            return None
        import math
        import matplotlib.pyplot as plt
        import torch
        want = str(self.style.get("color_field", "") or "").lower()
        rng = self.style.get("color_range")
        if rng and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        else:                                                # settled ONCE, on the first frame THAT HAS A FIELD
            # Frame 0 of a run at rest has no strain, no pressure and no speed anywhere, and a range
            # settled there is [0, 0] -- every later frame then clamps to the top colour. The range
            # is taken from the first frame whose 2nd and 98th percentiles differ, and kept from
            # then on (a seeded view coloured by a field is that case exactly).
            # AND IT KEEPS WIDENING OVER THE FIRST THIRD OF THE RUN, rather than locking onto the
            # first frame that happens to have a spread. That frame is the STARTUP, where the flow
            # a cilium drives has not developed yet: measured on the coupling ladder, the range
            # settled at plus or minus 0.214 world per second while the developed field reaches
            # 1.42 -- saturated by seven times, the same fault as the stale hardcoded range it was
            # brought in to replace, arrived at from the other direction. Taking the widest range
            # seen over the first third costs nothing (the frames are drawn anyway) and ends up
            # representative of the flow rather than of its first millisecond.
            _fr = getattr(self, "_frng", None)
            _settling = int(getattr(self, "_fr_n", 0)) < max(2, self.n_frames // (3 * self.stride))
            self._fr_n = int(getattr(self, "_fr_n", 0)) + 1
            if _fr is None or not (_fr[1] > _fr[0]) or _settling:
                q = torch.quantile(val.float()[:: max(1, val.numel() // 200_000)],
                                   torch.tensor([0.02, 0.98], device=val.device))
                _lo, _hi = float(q[0]), float(q[1])
                # A SIGNED FIELD GETS A SYMMETRIC RANGE, or the colour map lies about where zero is.
                # The 2nd and 98th percentiles of a signed quantity are not mirror images -- on the
                # cilium's outward-velocity field they came out -3.46e-02 and +4.57e-02 -- and on a
                # diverging map like bwr that puts WHITE at +5.6e-03 instead of at 0, so still water
                # renders faintly red and a viewer reads outward flow that is not there. Symmetric
                # about zero, so white means zero and the two directions share one scale.
                if _lo < 0.0 < _hi:
                    _m = max(abs(_lo), abs(_hi))
                    _lo, _hi = -_m, _m
                if _fr is not None and _fr[1] > _fr[0]:          # widen, never narrow
                    _lo, _hi = min(_lo, _fr[0]), max(_hi, _fr[1])
                self._frng = (_lo, _hi)
            lo, hi = self._frng
        # LOG BY DEFAULT FOR VORTICITY, because the quantity spans decades and a linear ramp shows
        # one of them. Measured on si_ball_wake, water |curl v|: median 0.6 -> 163 1/s over the run
        # and 0.6 -> 1305 within the last frame. On a linear [0, 2000] the median sits at 8% of the
        # map, so the column reads as one flat colour and the wake -- which IS there -- is the same
        # dark purple as the still fluid. log10 puts three decades across the ramp instead of one.
        # A LOG RAMP NEEDS A POSITIVE TOP, AND AN AUTO RANGE DOES NOT ALWAYS HAVE ONE. When
        # `color_range` is absent the range is settled from the first frame that HAS a field, and
        # until one arrives it is [0, 0]; `eps = max(lo, hi*1e-4)` is then 0 and `log10(0)` raises
        # `ValueError: math domain error` from inside the renderer, which does not fall back -- it
        # prints "DISABLED after frame 0" and the run finishes with no movie at all. Measured on
        # cil_s17_amp25, whose water starts at rest by construction: a spec that asked for a log
        # ramp lost every frame because its first one was quiet. Linear until there is a decade to
        # show, log from then on.
        if bool(self.style.get("field_log", want == "vorticity")) and hi > 0 and max(lo, hi * 1e-4) > 0:
            eps = max(lo, hi * 1e-4)
            t = ((torch.log10(val.clamp(min=eps)) - math.log10(eps))
                 / max(math.log10(hi) - math.log10(eps), 1e-12)).clamp(0, 1).detach().cpu().numpy()
            self._lut = f"log [{eps:.3g}, {hi:.3g}]"
        else:
            t = ((val - lo) / max(hi - lo, 1e-12)).clamp(0, 1).detach().cpu().numpy()
            self._lut = f"[{lo:.3g}, {hi:.3g}]"
        cm = plt.get_cmap(self.style.get("field_cmap", "turbo"))
        # THE LUT LEGEND CARRIES ITS UNIT NOW, when the block it colours by declared one. The
        # legend is the only surface in the frame that states a RANGE, and a range is exactly the
        # kind of number a reader takes away -- "[0, 0.26]" of what? The unit comes from the
        # block's own `unit:` on the set that owns it (U1), so it is the same declaration the
        # curve panels and the checker read, and the range is left BARE when there is none rather
        # than guessed. The numbers themselves are not rescaled here: `lo`/`hi` are the colour
        # map's endpoints and rescaling them would desynchronise the legend from the colours,
        # which are computed from the simulation values above.
        _bu = self._block_unit(lvl, str(self.style.get("color_field", "") or ""))
        _bl = _units_label(_bu, self._units) if _bu else None
        self.colour_by = (f"{label} {self._lut}{' ' + _bl if _bl else ''} "
                          f"({self.style.get('field_cmap','turbo')})")
        return (cm(t)[:, :3] * 255).astype(np.uint8)

    def _rgb_chem(self, H, lvl):
        """Per-particle colour from the set's own `chem` block -- THE SAME LAW the still and the
        `-o plot` movie use, `plexus.live.chem_rgb`, and not a second one.

        WHY THIS IS A CALL AND NOT AN IMPLEMENTATION. `chem_rgb`'s own docstring says what happens
        otherwise: "the alternative is two copies that drift: the live snapshot and the movie of one
        run showed DIFFERENT PICTURES of the same numbers for exactly that reason". This renderer
        was the third reader of `chem` and had no copy at all -- it fell through to the height ramp,
        so the live movie of a reaction-diffusion run drew the disc's SHAPE while `2d.png` beside it
        drew the pattern.

        The law needs no new spec key. `plotting.species` is the positional column -> colour table
        (`None` = declared and deliberately not drawn) and `plotting.blend` is `subtractive` (white
        quiet, each species takes light away -- the Gray-Scott look) or `additive` (black quiet,
        each species adds its colour -- right for a conserved three-species partition). Absent, the
        law's own default draws the activator of each pair: column 0 red, column 2 blue.
        """
        sch = getattr(lvl, "state_schema", None)
        if sch is None or "chem" not in sch:
            # THE REPLAY LEVEL HAS NO `state_schema`, BUT IT SERVES THE RECORDED `chem` THROUGH `get`,
            # as the curves read it (`measures.curve_row`, `species:` / `block:`). Returning None here
            # sent every replay of a chemistry run to the height ramp -- and a spec with a curve panel
            # is re-rendered on the replay, so exp 15's colony movies came out grey-lilac while the
            # live pass had drawn the strains red and green (2026-09-27). The live path is unchanged.
            try:
                v = lvl.get("chem")
            except Exception:                                          # noqa: BLE001
                return None
            if v is None:
                return None
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            if v.ndim == 3:                                            # [T, n, w]: the replay's frame
                v = v[int(getattr(lvl, "t", 0))]
            if v.ndim != 2 or v.shape[1] == 0:
                return None
            _ix = self.idx.detach().cpu().numpy() if hasattr(self.idx, "detach") else np.asarray(self.idx)
            v = v[_ix]
            a, b = 0, v.shape[1]
        else:
            a, b = sch["chem"]
            if b <= a:
                return None
            v = None
        try:
            from plexus.live import chem_rgb
            if v is None:
                v = lvl.state[self.idx, a:b]
                v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            cols, _hi = chem_rgb(np.asarray(v, float).reshape(v.shape[0], -1),
                                 lut=(self.style or {}).get("species"),
                                 blend=(self.style or {}).get("blend"),
                                 background=(self.style or {}).get("background", "black"),
                                 vmax=(self.style or {}).get("chem_max"))
        except Exception as e:                                       # noqa: BLE001
            print(f"[live-movie] chem colouring unavailable ({type(e).__name__}: {e})", flush=True)
            return None
        if cols is None:
            return None
        self._chem_live = True
        self.colour_by = f"chem, {b - a} column(s), via plexus.live.chem_rgb"
        return (np.clip(cols, 0, 1) * 255).astype(np.uint8)

    def _rgb_parent_chem(self, H, lvl):
        """`plotting.color_by: chem` -- each particle in its PARENT's `chem` colour, per frame.

        WHY. A segmentation-seeded sheet (exp 6: `seed_from_segmentation` puts `mpm_particle`s in each
        `cell`) has its chemistry on the cells and its positions on the material points, and the
        points are the larger set, so they are the subject. They carry no `chem`, so `_rgb_chem`
        returned None and `_rgb` painted them by parent BODY -- one hue per cell id, fixed at t = 0:
        a static confetti square while the excitation wave crossed the tissue (14 exp 6 movies).
        Opt-in because a parent-body hue is the right picture for every MPM run whose cells carry
        a chemistry that is not the point (drops, balls). The law is `plexus.live.chem_rgb` over the
        PARENT's rows -- `species`, `blend`, `chem_max` -- then gathered by `lvl.parent`, so a
        particle is exactly its cell's colour. None when the subject has no chem-carrying parent.
        """
        pname = getattr(lvl, "parent_name", None)
        par = getattr(lvl, "parent", None)
        if not pname or par is None or pname not in getattr(H, "levels", {}):
            return None
        plv = H.level(pname)
        try:
            sch = getattr(plv, "state_schema", None)
            if sch is not None and "chem" in sch:
                a, b = sch["chem"]
                v = plv.state[:, a:b]
            else:
                v = plv.get("chem")
            if v is None:
                return None
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            if v.ndim == 3:                                            # [T, n, w]: the replay's frame
                v = v[int(getattr(plv, "t", 0))]
            if v.ndim != 2 or v.shape[1] == 0:
                return None
            from plexus.live import chem_rgb
            st = self.style or {}
            cols, _ = chem_rgb(np.asarray(v, float), lut=st.get("species"), blend=st.get("blend"),
                               background=st.get("background", "black"), vmax=st.get("chem_max"))
        except Exception as e:                                       # noqa: BLE001
            print(f"[live-movie] color_by: chem unavailable ({type(e).__name__}: {e})", flush=True)
            return None
        if cols is None:
            return None
        pi = par.detach().cpu().numpy() if hasattr(par, "detach") else np.asarray(par)
        ix = self.idx.detach().cpu().numpy() if hasattr(self.idx, "detach") else np.asarray(self.idx)
        pi = np.clip(np.asarray(pi, np.int64)[ix], 0, len(cols) - 1)
        self._chem_live = True
        self._chem_parent = True
        self.colour_by = f"parent {pname!r} chem, {v.shape[1]} column(s), via plexus.live.chem_rgb"
        return (np.clip(cols[pi], 0, 1) * 255).astype(np.uint8)

    def _rgb(self, H, lvl, pos):
        """Per-particle colour, FIXED AT t=0 and carried with the particle.

        THE SPEC ALREADY SAYS WHAT THE COLOURS ARE. `plotting.colors` in these specs is a 27-entry
        rainbow, `w00`..`w26`, one hue per parent body -- exactly what `plot.py:279` paints, and
        exactly what makes mixing legible: a dot's hue says which blob it started in, so two blobs
        interpenetrating is visible as two hues interleaving. This class originally invented its own
        height ramp instead, which threw that away and rendered 27 distinct bodies as one red-to-blue
        gradient. Colouring by height also encodes a quantity that CHANGES, so a colour recomputed
        per frame would hide the very motion the movie exists to show.

        Falls back to height only when the spec declares no palette and the set has no parent.
        """
        # `plotting.dot_color: <colour>` -- ONE FIXED COLOUR FOR EVERY DOT (exp 11, 2026-09-27: white basement-membrane
        # nodes over a white mesh). Opt-in; absent, the palette / height logic below is unchanged.
        _dc = (self.style or {}).get("dot_color")
        if _dc:
            from matplotlib.colors import to_rgb
            self.colour_by = f"fixed {_dc}"
            return (np.tile(np.asarray(to_rgb(_dc), float), (len(pos), 1)) * 255).astype(np.uint8)
        n = pos.shape[0]
        try:
            from plexus.plot import _typed_palette
            # THE SET'S OWN TYPE FIRST, when it has one. This branch did not exist: colour came from
            # the PARENT body or from height, which is right for MPM (particles inherit their cell's
            # material) and wrong for every set that is typed directly. A galaxy's stars carry
            # `node_type` 0/1 for the two discs and no parent at all, so `plotting.colors:
            # {red: ..., blue: ...}` -- the whole point of the picture, since the colours ARE the two
            # galaxies -- fell through to the height ramp and the merger rendered as one gradient.
            own = getattr(lvl, "node_type", None)
            if own is not None and getattr(self, "_sname", None):
                pal, _ = _typed_palette(self.sim, self._sname, self.style)
                if pal is not None:
                    tid = own[self.idx].detach().cpu().numpy() % len(pal)
                    self.colour_by = f"own node_type ({len(pal)} hues from plotting.colors)"
                    return (np.clip(pal[tid], 0, 1) * 255).astype(np.uint8)
                # A TYPED SET FALLS BACK TO A COLORMAP, NEVER TO HEIGHT. `_typed_palette` returns
                # None unless the spec lists `plotting.colors` BY TYPE NAME, so a spec that declares
                # sixteen types and a `colormap:` -- which is how ParticleGraph's boids specs are
                # written, and how anyone would reasonably ask for sixteen hues -- got the height
                # ramp instead: a red-to-blue gradient across the box, on a picture whose entire
                # subject is which type a cell is. The key was read by `plot.py` and ignored here,
                # which is the same shape of defect as the chemistry colour and the apico-basal
                # surface: the replay path honoured a spec key the live path did not.
                _names = list(getattr(lvl, "type_names", []) or [])
                if len(_names) > 1:
                    import matplotlib.pyplot as plt
                    _cm = plt.get_cmap(self.style.get("colormap", "tab20"))
                    _n = max(len(_names), 2)
                    # SAMPLED ACROSS THE MAP, not by integer index. `tab20` holds exactly 20 colours
                    # and indexing it 0..15 takes the first sixteen, which are eight hues in
                    # light/dark pairs -- adjacent types then read as the same colour. Spreading the
                    # samples over the whole map uses its full range whatever its length.
                    _cols = np.array([_cm(k / max(_n - 1, 1))[:3] for k in range(_n)], np.float32)
                    tid = own[self.idx].detach().cpu().numpy() % _n
                    self.colour_by = (f"own node_type ({_n} hues sampled from "
                                      f"{self.style.get('colormap', 'tab20')})")
                    return (np.clip(_cols[tid], 0, 1) * 255).astype(np.uint8)
            pname = getattr(lvl, "parent_name", None)
            par = getattr(lvl, "parent", None)
            if pname and par is not None:
                pal, _ = _typed_palette(self.sim, pname, self.style)
                pnt = getattr(H.level(pname), "node_type", None)
                if pal is not None and pnt is not None:
                    idx = par[self.idx].detach().cpu().numpy()
                    tid = pnt.detach().cpu().numpy()[idx] % len(pal)
                    self.colour_by = f"parent body ({len(pal)} hues from plotting.colors)"
                    return (np.clip(pal[tid], 0, 1) * 255).astype(np.uint8)
                if par is not None:                       # no declared palette: a distinct hue each
                    import matplotlib.pyplot as plt
                    cm = plt.get_cmap(self.style.get("colormap", "tab10"))
                    idx = par[self.idx].detach().cpu().numpy()
                    self.colour_by = f"parent body ({self.style.get('colormap', 'tab10')})"
                    return (np.array([cm(int(i) % cm.N)[:3] for i in idx]) * 255).astype(np.uint8)
        except Exception as e:
            print(f"[live-movie] palette unavailable ({type(e).__name__}: {e}); "
                  f"colouring by height", flush=True)
        h = (pos[:, self.up] - self.lo[self.up]) / max(self.hi[self.up] - self.lo[self.up], 1e-9)
        self.colour_by = "height at t=0"
        return (np.stack([np.clip(1.4 - 1.6 * h, 0, 1), np.clip(0.35 + 0.5 * h, 0, 1),
                          np.clip(0.25 + 1.1 * h, 0, 1)], 1) * 255).astype(np.uint8)

    # ------------------------------------------------------------------ spheres of a real radius
    # `plotting.dot_radius: {nucleus: 0.3, mitochondria: 0.06}` draws each type of the subject set as
    # lit spheres of that WORLD radius, one glyph actor per type, instead of pixel dots. A dot has no
    # size in the world; an organelle does, and "does the nucleus fit inside the cell" is only
    # visible if the picture draws it at its radius. Rebuilt each frame (a few thousand spheres).
    def _glyph_types(self, lvl):
        names = list(getattr(lvl, "type_names", []) or [])
        rad = (self.style or {}).get("dot_radius") or {}
        own = getattr(lvl, "node_type", None)
        if not names or own is None:
            return None
        from matplotlib.colors import to_rgb
        pal = (self.style or {}).get("colors") or {}
        out = []
        for tid, nm in enumerate(names):
            r = rad.get(nm)
            if r is None:
                continue
            col = to_rgb(tuple(pal[nm]) if isinstance(pal.get(nm), (list, tuple)) else pal[nm]) if nm in pal else (0.9, 0.9, 0.9)
            out.append((tid, nm, float(r), col))
        return out or None

    def _glyph_opacity(self, key):
        """`plotting.dot_opacity: {body: 0.12, yolk_mass: 0.5}` -- per TYPE, 1 when unstated.

        A SET OF 33,000 SPHERES IS A WALL, and that is the problem this solves. The bodies in a
        swimming scene are drawn as glyphs precisely so they read as objects rather than dust,
        and the consequence is that they also hide everything behind them: an animal drawn solid
        occludes the water it is supposed to be moving, and the run's whole subject is what the
        water does. Opacity is the knob that lets a body be present without being a screen.

        Keyed by type, like `dot_radius` and `colors`, so a scene can make its animal a ghost and
        leave its yolk legible inside it. `key` is `<set>_<type>`, which is how `_glyphs` names
        its actors, so the type is what follows the set's name.
        """
        opa = (self.style or {}).get("dot_opacity") or {}
        if not opa:
            return 1.0
        for nm, v in opa.items():
            if key.endswith("_" + str(nm)):
                return float(v)
        return 1.0

    def _glyph_points(self, lv, tid, subject):
        """Live positions of one type: the drawn subset for the subject, every live row otherwise."""
        import torch
        own = getattr(lv, "node_type", None)
        occ = getattr(lv, "occ", None)
        if subject and not (self.style or {}).get("near_side"):
            nt = torch.as_tensor(own)[self.idx]
            sel = nt == tid
            if occ is not None:
                sel = sel & (torch.as_tensor(occ)[self.idx].to(nt.device) > 0)
            return np.asarray(self.cloud.points)[sel.cpu().numpy()]
        # WITH `near_side` ON, THE CLOUD IS ALREADY PARKED. `_xyz` hides a far point by moving it
        # onto a near one (its colours are bound to a fixed-length array), so reading the glyphs
        # off the cloud drew every far nucleus stacked on one near position -- the cut looked
        # like no cut at all. The glyph actor has its own geometry, so it reads the real state.
        nt = torch.as_tensor(own)
        sel = nt == tid
        if occ is not None:
            sel = sel & (torch.as_tensor(occ).to(nt.device) > 0)
        P = lv.get("pos").detach()[sel.to(lv.get("pos").device)]
        P = P.cpu().numpy().astype(np.float32)
        if P.shape[1] == 2:
            P = np.concatenate([P, np.zeros((P.shape[0], 1), np.float32)], 1)
        return P

    def _chain_build(self, H):
        """Draw a set's points as a CONNECTED FILAMENT, `plotting.chain`.

        A rod, a flagellum, a fibre and a chain of crosslinked beads are all ordered sequences of
        points, and a point renderer draws them as loose dots. On the first cilium rig that was
        the difference between a picture of a cilium and a picture of eighteen blue dots: the rod
        was bending correctly and the render could not say so, because nothing joined the nodes.

            plotting:
              chain: {set: rod_node, per: 24, width: 5, color: "#ffffff"}

        `per` is the nodes per filament, so one set can hold many and the lines do not jump from
        one filament's tip to the next one's base. Left out, the whole set is one chain.

        `per` IS THE FALLBACK NOW, NOT THE MECHANISM. A rod set carries its own SEGMENT TABLE --
        `E_srce`/`E_trgt` in the same `MeshTable` an epithelium uses, with `face` absent and `nF`
        0, which is what a 1D mesh is -- and this reads it when it is there. The difference is not
        tidiness: `per` can only describe N filaments of EQUAL length laid in contiguous blocks,
        so it cannot draw a ring, filaments of unequal length, or nine cross-linked doublets, and
        the last of those is what an axoneme is.
        """
        cfg = (self.style or {}).get("chain")
        if not cfg:
            return
        cfg = dict(cfg)
        # `set:` MAY BE A LIST. Two cells with their own rod sets are two filament families, and a
        # renderer that draws one of them puts half the physics outside the picture -- measured on
        # p3s_twocells, where cell_b's five cilia were in the run and not in the movie. One actor
        # per set, all moved each frame.
        names = cfg.get("set", "")
        names = [str(n) for n in (names if isinstance(names, (list, tuple)) else [names])]
        self._chains = []
        for name in names:
            self._chain_build_one(H, name, cfg)
        self._chain = self._chains[0][0] if self._chains else None

    def _chain_build_one(self, H, name, cfg):
        if name not in H.levels:
            print(f"[live-movie] chain: no set {name!r}; nothing drawn", flush=True)
            return
        lvl = H.level(name)
        pos = self._xyz(lvl, all_rows=True) if hasattr(self, "_xyz_all") else None
        pos = np.asarray(lvl.get("pos").detach().cpu().numpy(), np.float32) if pos is None else pos
        n = pos.shape[0]
        # THE SEGMENT TABLE IS THE AUTHORITY WHEN THERE IS ONE, and `per` is the fallback. This
        # docstring used to say the connectivity "is not stored anywhere because it does not need
        # to be: consecutive rows ARE the filament" -- which stopped being true when `rod_seed`
        # began writing `E_srce`/`E_trgt` into the set's MeshTable. Reading the table matters for
        # more than tidiness: `per` can only describe N filaments of EQUAL length laid in
        # contiguous blocks, so it cannot draw a ring, filaments of different lengths, or nine
        # cross-linked doublets -- and the last of those is what an axoneme is.
        m = getattr(lvl, "mesh", None)
        tbl = None
        if m is not None and "E_srce" in m and m["E_srce"].numel():
            _i = m["E_srce"].detach().cpu().numpy().astype(np.int64)
            _j = m["E_trgt"].detach().cpu().numpy().astype(np.int64)
            tbl = (_i, _j)
        if tbl is not None:
            _i, _j = tbl
            seg = np.stack([np.full(_i.size, 2, np.int64), _i, _j], 1).reshape(-1)
            n_ch, per = int(getattr(lvl, "n_rod", 1)), n // max(int(getattr(lvl, "n_rod", 1)), 1)
            _src = f"{_i.size} segments from the table"
        else:
            per = int(cfg.get("per", n) or n)
            per = max(min(per, n), 2)
            n_ch = n // per
            # two-point line cells, consecutive WITHIN a filament: [2, i, i+1] per segment
            seg = np.stack([np.full(n_ch * (per - 1), 2, np.int64),
                            np.concatenate([np.arange(r * per, r * per + per - 1)
                                            for r in range(n_ch)]),
                            np.concatenate([np.arange(r * per + 1, r * per + per)
                                            for r in range(n_ch)])], 1).reshape(-1)
            _src = f"row order, per={per}"
        mesh = self.pv.PolyData(pos)
        mesh.lines = seg
        self.p.add_mesh(mesh, color=str(cfg.get("color", "#ffffff")), lighting=False,
                        line_width=float(cfg.get("width", 5.0)), render_lines_as_tubes=True)
        self._chains.append((mesh, name))
        print(f"[live-movie] chain: {n_ch} filament(s) of {per} nodes from {name!r} "
              f"({_src}), drawn as tubes", flush=True)

    def _chain_update(self, H):
        """Move the filament's points. The connectivity never changes, so only the points do."""
        for mesh, name in getattr(self, "_chains", []) or []:
            mesh.points = np.asarray(H.level(name).get("pos").detach().cpu().numpy(), np.float32)

    def _field_slice_update(self, H, first=False):
        """`plotting.field_slice: {field, axis, at, channel, range, cmap, opacity}` -- ONE PLANE OF
        A FIELD, drawn as a textured quad where it sits in the box: the slice of the field's grid
        normal to `axis` at the fraction `at` of the box, its `channel` through a colormap over
        `range` (settled once from the first frame when absent). A field is a continuum on its own
        grid, and this is the one picture of it the renderer had no way to give -- the proton pool
        around the motor, a morphogen -- without a cloud of samples. Built for the motor's ion
        economy (builder/exp_02_bacterium, 2026-09-23); a field without a `.grid` (the MPM grid)
        is skipped with a line."""
        cfg = (self.style or {}).get("field_slice")
        if not cfg:
            return
        import numpy as np
        cfg = dict(cfg)
        name = str(cfg.get("field", ""))
        fields = getattr(H, "fields", None)
        fld = fields[name] if (fields is not None and name in fields) else None
        g = getattr(fld, "grid", None) if fld is not None else None
        if g is None:
            if first:
                print(f"[live-movie] field_slice: field {name!r} has no `.grid` to slice; nothing drawn", flush=True)
            return
        arr = g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)
        ch = int(cfg.get("channel", 0))
        arr = arr[ch] if arr.ndim == 4 else arr
        if arr.ndim != 3:
            if first:
                print(f"[live-movie] field_slice: {name!r} is not a 3-D grid; nothing drawn", flush=True)
            return
        ax = {"x": 0, "y": 1, "z": 2}[str(cfg.get("axis", "z")).lower()]
        n = arr.shape[ax]
        k = int(min(n - 1, max(0, round(float(cfg.get("at", 0.5)) * (n - 1)))))
        sl = np.take(arr, k, axis=ax).astype(np.float64)
        # `extent: [[a0, a1], [b0, b1]]` -- DRAW ONLY A WINDOW OF THE PLANE, in world units along its
        # two in-plane axes (in axis order, the normal left out). A channel's current lives within a
        # few nm of its pore; the rest of the plane is zero and, drawn at all, cut a visible band
        # through the translucent membrane behind it (exp04 render smoke test, 2026-09-25).
        _ext = cfg.get("extent")
        _other = [i for i in range(3) if i != ax]
        _w = np.asarray(self.world, float)
        _win = None
        if _ext is not None and len(_ext) == 2:
            _win = []
            for _d, (_a0, _a1) in zip(_other, _ext):
                _n = sl.shape[_other.index(_d)]
                _i0 = int(max(0, np.floor(float(_a0) / _w[_d] * _n)))
                _i1 = int(min(_n, np.ceil(float(_a1) / _w[_d] * _n)))
                _win.append((_i0, max(_i1, _i0 + 1)))
            sl = sl[_win[0][0]:_win[0][1], _win[1][0]:_win[1][1]]
        rng = cfg.get("range")
        if rng and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        elif getattr(self, "_fs_range", None) is not None:
            lo, hi = self._fs_range
        else:
            lo, hi = float(np.nanpercentile(sl, 2)), float(np.nanpercentile(sl, 98))
            if not hi > lo:
                hi = lo + 1.0
            self._fs_range = (lo, hi)
        import matplotlib.pyplot as plt
        _u = (np.clip(sl, lo, hi) - lo) / max(hi - lo, 1e-12)
        rgba = plt.get_cmap(str(cfg.get("cmap", "magma")))(_u)
        # `fade: true` -- EACH PIXEL AS OPAQUE AS ITS VALUE IS STRONG (alpha = u^0.5, so a weak current
        # still shows). A plane of the electrolyte's current is zero almost everywhere, and drawn at
        # one opacity its zero half painted a black band across the membrane behind it (exp04 render
        # smoke test, 2026-09-25); faded, only the current itself is drawn. Off by default.
        if bool(cfg.get("fade", False)):
            rgba[..., 3] = np.sqrt(_u)
            img = (rgba * 255).astype(np.uint8)
        else:
            img = (rgba[..., :3] * 255).astype(np.uint8)
        tex = self.pv.Texture(np.ascontiguousarray(np.transpose(img, (1, 0, 2))))
        if first or getattr(self, "_fs_actor", None) is None:
            w = np.asarray(self.world, float)
            at = float(cfg.get("at", 0.5)) * w[ax]
            centre = [0.5 * w[0], 0.5 * w[1], 0.5 * w[2]]; centre[ax] = at
            direction = [0.0, 0.0, 0.0]; direction[ax] = 1.0
            other = [i for i in range(3) if i != ax]
            if _win is not None:
                _n0, _n1 = arr.shape[other[0]], arr.shape[other[1]]
                _c0 = 0.5 * (_win[0][0] + _win[0][1]) / _n0 * w[other[0]]
                _c1 = 0.5 * (_win[1][0] + _win[1][1]) / _n1 * w[other[1]]
                centre[other[0]], centre[other[1]] = _c0, _c1
                plane = self.pv.Plane(center=centre, direction=direction,
                                      i_size=float((_win[0][1] - _win[0][0]) / _n0 * w[other[0]]),
                                      j_size=float((_win[1][1] - _win[1][0]) / _n1 * w[other[1]]))
            else:
                plane = self.pv.Plane(center=centre, direction=direction, i_size=float(w[other[0]]), j_size=float(w[other[1]]))
            plane.texture_map_to_plane(inplace=True)
            self._fs_actor = self.p.add_mesh(plane, texture=tex, opacity=float(cfg.get("opacity", 0.85)),
                                             lighting=False, name="field_slice")
            print(f"[live-movie] field_slice: {name!r} channel {ch}, {'xyz'[ax]} = {at:.3f}, range [{lo:.4g}, {hi:.4g}]", flush=True)
        else:
            try:
                self._fs_actor.SetTexture(tex)
            except Exception:                                        # noqa: BLE001
                pass

    def _field_iso_update(self, H, first=False):
        """`plotting.field_iso: {field, channel, levels: [..], colors: [..], opacity: [..], smooth}` --
        A GRID FIELD AS NESTED ISOSURFACES, re-contoured every rendered frame: the density-map way of
        showing a continuum, and the cryo-EM figure's own idiom. Written for the electrolyte current
        through a channel (experiments/exp04, 2026-09-25): a textured `field_slice` through a
        translucent membrane either painted its zero region across the bilayer or, faded by alpha,
        was drawn as a grey window VTK would not make transparent; a surface at a current density
        has no plane to mis-blend. `levels` are in the field's own units (pA/nm^2 for the current),
        each drawn in its colour at its opacity, matte, unlit by the studio light's specular."""
        cfg = (self.style or {}).get("field_iso")
        if not cfg:
            return
        import numpy as np
        cfg = dict(cfg)
        name = str(cfg.get("field", ""))
        fields = getattr(H, "fields", None)
        fld = fields[name] if (fields is not None and name in fields) else None
        g = getattr(fld, "grid", None) if fld is not None else None
        if g is None:
            if first:
                print(f"[live-movie] field_iso: field {name!r} has no `.grid`; nothing drawn", flush=True)
            return
        arr = g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)
        arr = arr[int(cfg.get("channel", 0))] if arr.ndim == 4 else arr
        if arr.ndim != 3:
            return
        w = np.asarray(self.world, float)
        sp = tuple(float(w[i]) / arr.shape[i] for i in range(3))
        img = self.pv.ImageData(dimensions=arr.shape, spacing=sp, origin=tuple(0.5 * s for s in sp))
        img.point_data["f"] = np.ascontiguousarray(arr, np.float32).ravel(order="F")
        levels = [float(v) for v in (cfg.get("levels") or [])]
        cols = list(cfg.get("colors") or ["#ffd24a"] * len(levels))
        ops = list(cfg.get("opacity") or [0.5] * len(levels))
        if first or getattr(self, "_fi_actors", None) is None:
            self._fi_actors, self._fi_meshes = [], []
            for k, lev in enumerate(levels):
                m = img.contour([lev], scalars="f")
                if m.n_points == 0:
                    m = self._fi_empty()
                elif int(cfg.get("smooth", 0)):
                    m = m.smooth_taubin(n_iter=int(cfg["smooth"]), pass_band=0.1)
                self._fi_meshes.append(m)
                self._fi_actors.append(self.p.add_mesh(m, color=cols[k % len(cols)], opacity=float(ops[k % len(ops)]),
                                                       smooth_shading=True, specular=0.0, ambient=0.6, diffuse=0.4,
                                                       name=f"field_iso_{k}"))
            print(f"[live-movie] field_iso: {name!r} at {levels} (field units), re-contoured every frame", flush=True)
            return
        for k, lev in enumerate(levels):
            m = img.contour([lev], scalars="f")
            if m.n_points == 0:
                m = self._fi_empty()
            elif int(cfg.get("smooth", 0)):
                m = m.smooth_taubin(n_iter=int(cfg["smooth"]), pass_band=0.1)
            try:
                self._fi_meshes[k].copy_from(m)
                self._fi_meshes[k].Modified()
            except Exception:                                    # noqa: BLE001 -- keep the last surface
                pass

    # ---- `plotting.also_sets`: MORE POINT SETS OVER THE SUBJECT ---------------------------------
    #
    #     plotting:
    #       subject: cell
    #       also_sets: [host]                  # drawn over the subject, every frame
    #       also_colors: {host: "#ffffff"}     # optional: a fixed colour per set
    #       also_dot_size: 6                   # optional: px; default the subject's dot size
    #
    # The movie draws ONE subject set, and a model with two populations -- exp 15's community on
    # its lattice and the host cells it feeds -- lost the second: the talk movie needed its own
    # script. Each named set is its own point actor, moved each frame. Its colour, in order: the
    # fixed `also_colors` entry, else its own `chem` through `plexus.live.chem_rgb` (the spec's
    # `species`, `blend`, `chem_max`; recomputed every frame), else its type palette
    # (`plotting.colors` by type name), else white. Dormant slots are parked on a live point.
    # Absent, nothing is built and no pixel changes.
    def _also_names(self, H):
        names = (self.style or {}).get("also_sets") or []
        names = [names] if isinstance(names, str) else list(names)
        out = []
        for nm in names:
            nm = str(nm)
            if nm not in getattr(H, "levels", {}):
                print(f"[live-movie] also_sets: no set {nm!r}; not drawn", flush=True)
            elif nm != getattr(self, "_sname", None):
                out.append(nm)
        return out

    def _also_xyz(self, lv):
        P = lv.get("pos")
        if P is None:
            return None, None
        P = np.asarray(P.detach().cpu().numpy() if hasattr(P, "detach") else P, np.float32)
        occ = getattr(lv, "occ", None)
        live = (np.ones(P.shape[0], bool) if occ is None else
                np.asarray(occ.detach().cpu().numpy() if hasattr(occ, "detach") else occ).astype(bool)
                .reshape(-1)[:P.shape[0]])
        if live.any() and not live.all():
            P = np.where(live[:, None], P, P[live][:1])
        if P.shape[1] == 2:
            # ON TOP, NOT IN THE SAME PLANE: a 2D run is drawn at z = 0 and seen from +z, and two
            # point sets at one depth z-fight. A lift far below a pixel settles which is in front.
            P = np.concatenate([P, np.full((P.shape[0], 1), 1e-3 * float(np.max(self.world)),
                                           np.float32)], 1)
        return P, live

    def _also_rgb(self, H, nm, lv, n, live):
        from matplotlib.colors import to_rgb
        st = self.style or {}
        fixed = (st.get("also_colors") or {}).get(nm) if isinstance(st.get("also_colors"), dict) else None
        cols = None
        # `color_block.<set>` ON AN `also_sets` SET (exp 11, 2026-09-29: the membrane's mass or bonds, `bm_M`, `bond_P`):
        # the set's dots -- and its `also_mesh` wireframe, which takes these colours per node -- through the block's
        # colormap at its fixed range, instead of one flat `also_colors` hue. Absent, nothing changes.
        _cb = self._block_cfg(nm)
        if _cb is not None:
            try:
                from matplotlib import colormaps
                _v, _lo, _hi, _cmap, _ = self._block_values(H, nm, _cb)
                _v = np.asarray(_v, float)[:n]
                _t = np.clip((_v - _lo) / max(_hi - _lo, 1e-12), 0.0, 1.0)
                _t = np.where(np.isfinite(_t), _t, 0.0)
                cols = colormaps[str(_cmap or "magma")](_t)[:, :3]
            except Exception as e:                                   # noqa: BLE001 -- the flat hue, loudly
                if not getattr(self, f"_cb_warn_{nm}", False):
                    setattr(self, f"_cb_warn_{nm}", True)
                    print(f"[live-movie] color_block.{nm} on also_sets not drawn ({type(e).__name__}: {e})", flush=True)
                cols = None
        if cols is not None:
            pass
        elif fixed is not None:
            cols = np.tile(np.asarray(to_rgb(fixed), float), (n, 1))
        else:
            try:
                sch = getattr(lv, "state_schema", None)
                if sch is not None and "chem" in sch:
                    a, b = sch["chem"]
                    v = lv.state[:n, a:b]
                else:
                    v = lv.get("chem")
                if v is not None:
                    v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                    if v.ndim == 2 and v.shape[1] > 0:
                        from plexus.live import chem_rgb
                        cols = chem_rgb(np.asarray(v, float)[:n], lut=st.get("species"),
                                        blend=st.get("blend"), background=st.get("background", "black"),
                                        vmax=st.get("chem_max"))[0]
            except Exception:                                        # noqa: BLE001
                cols = None
            if cols is None:
                own = getattr(lv, "node_type", None)
                pal = None
                if own is not None:
                    from plexus.plot import _typed_palette
                    pal, _ = _typed_palette(self.sim, nm, st)
                if pal is not None:
                    tid = np.asarray(own.detach().cpu().numpy() if hasattr(own, "detach") else own,
                                     np.int64)[:n] % len(pal)
                    cols = np.asarray(pal, float)[tid]
                else:
                    cols = np.ones((n, 3))
        rgb = (np.clip(np.asarray(cols, float), 0, 1) * 255).astype(np.uint8)
        if live is not None and live.any() and not live.all():
            rgb[~live] = rgb[live][0]
        return rgb

    def _also_build(self, H):
        self._also = []
        if not (self.style or {}).get("also_sets"):
            return
        _ps = (self.style or {}).get("also_dot_size")
        _ps = float(_ps) if _ps is not None else float(getattr(self, "px_used", None) or 3.0)
        for nm in self._also_names(H):
            try:
                lv = H.level(nm)
                P, live = self._also_xyz(lv)
                if P is None or not len(P):
                    continue
                pd = self.pv.PolyData(P)
                pd["rgb"] = self._also_rgb(H, nm, lv, P.shape[0], live)
                self.p.add_mesh(pd, scalars="rgb", rgb=True, **FLAT, point_size=_ps)
            except Exception as e:                                   # noqa: BLE001 -- not the movie
                print(f"[live-movie] also_sets: {nm!r} not drawn ({type(e).__name__}: {e})", flush=True)
                continue
            self._also.append((nm, pd))
            print(f"[live-movie] also_sets: {nm!r} drawn over the subject, {P.shape[0]:,} nodes, "
                  f"{_ps:g} px", flush=True)
        # `plotting.protein_bars` (exp 11, 2026-09-29): COLOUR BARS for what the picture is painted with -- the
        # layer's `mesh_color_by` (when `mesh_color_scale: continuous` and `mesh_color_range` are given; its title
        # `mesh_color_label`) and every `color_block` set among `also_sets` (its `label`) -- side by side at the
        # bottom right of the scene, clear of the section inset at the bottom left. Opt-in: absent, no bar.
        if (self.style or {}).get("protein_bars"):
            try:
                self._protein_bars(H)
            except Exception as e:                                   # noqa: BLE001 -- not the movie
                print(f"[live-movie] protein_bars not drawn ({type(e).__name__}: {e})", flush=True)
        # `plotting.also_mesh: [set, ...]` -- THE SET DRAWN AS A SHEET, not only as dots (exp 11, 2026-09-27: the
        # basement membrane). Each frame its LIVE nodes are triangulated by the convex hull of their directions from
        # their own centroid -- a spherical Delaunay, right for a star-shaped sheet such as a membrane around a
        # spheroid -- and drawn as a wireframe in the set's `also_colors` colour (`also_mesh_line_width`, default 1).
        # A picture of how the nodes neighbour each other, NOT the model's spring bonds (the trajectory does not
        # record them). Opt-in; absent, nothing is built.
        self._also_mesh = []
        am = (self.style or {}).get("also_mesh") or []
        am = [am] if isinstance(am, str) else list(am)
        for nm in am:
            try:
                lv = H.level(str(nm))
                P, live = self._also_xyz(lv)
                faces = self._hull_faces(P, live)
                if faces is None:
                    continue
                mpd = self.pv.PolyData(P, faces)
                from matplotlib.colors import to_rgb
                col = ((self.style or {}).get("also_colors") or {}).get(str(nm), "#ffffff")
                _lw = float((self.style or {}).get("also_mesh_line_width", 1.0))
                _op = float((self.style or {}).get("also_mesh_opacity", 1.0))
                if self._block_cfg(str(nm)) is not None:        # the wireframe in the block's colours, per node
                    mpd.point_data["rgb"] = self._also_rgb(H, str(nm), lv, P.shape[0], live)
                    self.p.add_mesh(mpd, style="wireframe", scalars="rgb", rgb=True, lighting=False,
                                    line_width=_lw, opacity=_op, render_lines_as_tubes=False)
                else:
                    self.p.add_mesh(mpd, style="wireframe", color=to_rgb(col), lighting=False,
                                    line_width=_lw, opacity=_op, render_lines_as_tubes=False)
                self._also_mesh.append((str(nm), mpd))
                print(f"[live-movie] also_mesh: {nm!r} drawn as a hull-triangulated wireframe", flush=True)
            except Exception as e:                                   # noqa: BLE001 -- not the movie
                print(f"[live-movie] also_mesh: {nm!r} not drawn ({type(e).__name__}: {e})", flush=True)
        # `plotting.iso_cells: {set, label, color, spacing, blur, iso_frac, smooth}` -- EVERY CELL OF A SET OF
        # MATERIAL POINTS DRAWN AS ITS OWN SURFACE (exp 11, 2026-09-27: the gland's interior MPM cells, the human's
        # "use iso surface for the rendering"). The points of each value of `label` (a per-point cell id, e.g.
        # `pid`) are splatted on a small grid of `spacing` (world units) around that cell, blurred by `blur`
        # voxels, and contoured at `iso_frac` of the median occupied density -- the recipe `_isosurface` uses for
        # a whole cloud, one cell at a time -- then coloured by `color` (a 0..19 block, `tab20`). With `near_side:
        # far` the cells in front of the plane through the body's centre are left out whole, so a cut-open gland
        # shows its interior cells behind the cut. Opt-in; absent, nothing is built.
        self._iso = None
        ic = (self.style or {}).get("iso_cells")
        if ic:
            try:
                surf = self._iso_cells_poly(H, dict(ic))
                if surf is not None:
                    self.p.add_mesh(surf, scalars="c", cmap="tab20", clim=[0, 19], smooth_shading=True,
                                    specular=0.25, show_scalar_bar=False)
                    self._iso = (dict(ic), surf)
                    print(f"[live-movie] iso_cells: {ic.get('set')!r} drawn as one isosurface per cell "
                          f"({surf.n_points:,} vertices)", flush=True)
            except Exception as e:                                   # noqa: BLE001 -- not the movie
                print(f"[live-movie] iso_cells not drawn ({type(e).__name__}: {e})", flush=True)

    def _iso_cells_poly(self, H, cfg):
        """The merged per-cell isosurfaces of `cfg['set']` (see `plotting.iso_cells`), scalars `c` = colour label."""
        from scipy.ndimage import gaussian_filter
        lv = H.level(str(cfg["set"]))
        P, live = self._also_xyz(lv)
        if P is None or not len(P):
            return None
        def _blk(name):
            v = lv.get(str(name))
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            return np.asarray(v, float).reshape(len(v), -1)[:, 0][: len(P)]
        lab = _blk(cfg.get("label", "pid")).round().astype(np.int64)
        col = _blk(cfg["color"]) if cfg.get("color") else (lab % 20).astype(float)
        if live is not None:
            P, lab, col = P[live], lab[live], col[live]
        h = float(cfg.get("spacing", 0.1)); blur = float(cfg.get("blur", 1.0))
        frac = float(cfg.get("iso_frac", 0.5)); n_sm = int(cfg.get("smooth", 10))
        far = str((self.style or {}).get("near_side", "") or "").lower() == "far"
        if far:
            n = np.asarray(self.p.camera.position, float) - np.asarray(self.p.camera.focal_point, float)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            _nc = (self.style or {}).get("near_side_centre")
            ctr = np.asarray(_nc, float) if _nc is not None else P.mean(0)
        order = np.argsort(lab, kind="stable")
        lab_s, P_s, col_s = lab[order], P[order], col[order]
        cuts = np.flatnonzero(np.diff(lab_s)) + 1
        parts = []
        pad = (2.0 + 3.0 * blur) * h
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(lab_s)]):
            X = P_s[a:b]
            if len(X) < 4:
                continue
            if far and float((X.mean(0) - ctr) @ n) > 0.0:
                continue
            lo = X.min(0) - pad
            dim = np.maximum(np.ceil((X.max(0) + pad - lo) / h).astype(int) + 1, 2)
            if int(np.prod(dim)) > 200_000:                          # a torn cell: do not draw a room-sized grid
                continue
            ijk = np.clip(((X - lo) / h).astype(np.int64), 0, dim - 1)
            D = np.zeros(tuple(dim), np.float32)
            np.add.at(D, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
            held = D > 0
            D = gaussian_filter(D, sigma=blur)
            iso = frac * float(np.median(D[held]))
            g = self.pv.ImageData(dimensions=tuple(int(v) for v in dim), spacing=(h, h, h),
                                  origin=tuple(float(v) for v in lo))
            g.point_data["d"] = D.ravel(order="F")
            srf = g.contour([iso], scalars="d")
            if srf.n_points == 0:
                continue
            if n_sm > 0:
                srf = srf.smooth_taubin(n_iter=n_sm, pass_band=0.1)
            # ONLY THE COLOUR ARRAY: the contour keeps its density `d`, constant on the surface, and after a
            # `copy_from` it became the active scalar -- every cell one colour
            for _k in list(srf.point_data.keys()):
                if _k != "Normals":
                    srf.point_data.remove(_k)
            srf.point_data["c"] = np.full(srf.n_points, float(col_s[a]) % 20.0)
            parts.append(srf)
        if not parts:
            return None
        out = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
        return out.extract_surface() if hasattr(out, "extract_surface") and not isinstance(out, self.pv.PolyData) else out

    @staticmethod
    def _hull_faces(P, live):
        """VTK face array of the convex hull of the live points' unit directions about their centroid, or None."""
        from scipy.spatial import ConvexHull
        idx = np.flatnonzero(live) if live is not None else np.arange(len(P))
        if idx.size < 4:
            return None
        Q = P[idx].astype(np.float64)
        u = Q - Q.mean(0)
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        tri = idx[ConvexHull(u).simplices]
        # A HOLE STAYS A HOLE: the hull closes every gap, so triangles whose longest edge is above 3x the median edge
        # (spanning a membrane hole rather than joining neighbours) are dropped.
        e = np.stack([np.linalg.norm(P[tri[:, a]] - P[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1)
        tri = tri[e.max(1) <= 3.0 * float(np.median(e))]
        return np.hstack([np.full((len(tri), 1), 3), tri]).astype(np.int64).ravel()

    # ---- `plotting.protein_insets`: SMALL 3D MAPS OF ONE BLOCK EACH, beside the section inset ------------------
    #
    #     protein_insets:
    #       - {what: mesh, block: itg_B, cmap: viridis, label: bound integrin}     # the layer's cells
    #       - {what: bm_node, block: bm_M, cmap: magma, label: BM mass}            # a set's elements, as points
    #     protein_inset_size: 0.215     # each inset's height, a fraction of the window's
    #
    # exp 11, 2026-09-29, the human: the proteins painted on the main view saturated (a fixed range) and the
    # membrane's mass could not be read on the spheroid, "two inlets ... to see separately bound integrin and BM
    # mass 3D maps, no colorbar". Each inset is its own renderer on the MAIN CAMERA (it turns with the scene), a
    # square viewport in a row just right of the section inset, a label inside its top edge and no bar. `range: auto`
    # (the default) is the block's 2nd-98th percentile over EVERY recorded row on a replay (positive values; the
    # set's live elements for a set), else the first frame's, then fixed: one colour, one number, all movie. A
    # quantity with an absolute meaning declares `range: [lo, hi]` instead, so its colour compares ACROSS runs: the
    # membrane's mass, 1 at rest, is [0, 2] in exp 11 (a per-run range had nothing to spread where it never moves).
    def _protein_insets_update(self, H):
        cfgs = (self.style or {}).get("protein_insets") or []
        if not cfgs:
            return
        try:
            if getattr(self, "_pins", None) is None:
                self._pins = []
                s = float((self.style or {}).get("protein_inset_size", 0.17))
                csl = ((self.style or {}).get("cross_section") or {}).get("loc") or (0.015, 0.03)
                csh = float((self.style or {}).get("cross_section_height", 0.26))
                x_first = float(csl[0]) + csh / self.aspect + 0.008
                for k, cfg in enumerate(cfgs):
                    # IN A ROW along the section inset's bottom edge, not a column (the human, 2026-09-29): side
                    # by side the two maps are read together, at the same height as the section
                    x0 = x_first + k * (s / self.aspect + 0.008)
                    y0 = float(csl[1])
                    ren = self.pv.Renderer(self.p, border=True, border_color="#9a9a9a", border_width=1.0)
                    ren.SetViewport(x0, y0, x0 + s / self.aspect, y0 + s)
                    ren.set_background("black")
                    # ITS OWN CAMERA (`_pin_camera`): the main view's direction, centred on the inset's object
                    from vtkmodules.vtkRenderingCore import vtkCamera
                    ren.SetActiveCamera(vtkCamera())
                    self.p.ren_win.AddRenderer(ren)
                    # THE TITLE ABOVE THE INSET, on the main scene: inside it, the inset's own renderer painted over it
                    _t = str(cfg.get("label", cfg.get("block", "")))
                    if len(_t) > 18 and " " in _t:            # two lines, split at the space nearest the middle
                        _sp = [i for i, ch in enumerate(_t) if ch == " "]
                        _i = min(_sp, key=lambda i: abs(i - len(_t) / 2))
                        _t = _t[:_i] + "\n" + _t[_i + 1:]
                    self.p.add_text(_t, position=(x0, y0 + s + 0.006),
                                    viewport=True, font_size=int((self.style or {}).get("protein_inset_font_size", 11)),
                                    color=self._fg, name=f"pin_label_{k}")
                    self._pins.append({"cfg": dict(cfg), "ren": ren, "pd": None, "rng": None})
            for pin in self._pins:
                self._pin_frame(H, pin)
                self._pin_camera(pin)
        except Exception as e:                                       # noqa: BLE001 -- not the movie
            if not getattr(self, "_pins_warned", False):
                self._pins_warned = True
                print(f"[live-movie] protein_insets not drawn ({type(e).__name__}: {e})", flush=True)

    def _pin_camera(self, pin):
        """THE MAIN VIEW'S DIRECTION, THE INSET'S OWN FRAMING (the human, 2026-09-29: "the spheroid is not well
        centred in the small inlet"): looking along the main camera, centred on the inset's own object and fitted
        to its bounds every frame, since it grows. A copy of the main camera kept the object where it sits in the
        wide main view, off the side of a square inset. `protein_inset_zoom` (default 1.15) tightens the fit."""
        pd = pin.get("pd")
        if pd is None or pd.n_points == 0:
            return
        mc = self.p.renderer.GetActiveCamera()
        cam = pin["ren"].GetActiveCamera()
        b = pd.bounds
        c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        d = np.asarray(mc.GetPosition(), float) - np.asarray(mc.GetFocalPoint(), float)
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + d))
        cam.SetViewUp(*mc.GetViewUp())
        cam.SetViewAngle(mc.GetViewAngle())
        pin["ren"].ResetCamera(b)
        cam.Zoom(float((self.style or {}).get("protein_inset_zoom", 1.15)))
        pin["ren"].ResetCameraClippingRange()

    @staticmethod
    def _pin_actor(ren, pd, cells, point_size=3.0):
        """A plain VTK actor on the inset renderer (pyvista's Renderer has no add_mesh): the `pin_rgb` array
        drawn as colours directly, unlit, per cell or per point."""
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        mp = vtkPolyDataMapper()
        mp.SetInputData(pd)
        mp.SetColorModeToDirectScalars()
        if cells:
            mp.SetScalarModeToUseCellFieldData()
        else:
            mp.SetScalarModeToUsePointFieldData()
        mp.SelectColorArray("pin_rgb")
        mp.ScalarVisibilityOn()
        a = vtkActor()
        a.SetMapper(mp)
        a.GetProperty().LightingOff()
        a.GetProperty().SetPointSize(float(point_size))
        ren.AddActor(a)
        return a

    def _pin_range(self, H, cfg, now):
        r = cfg.get("range", "auto")
        if isinstance(r, (list, tuple)) and len(r) == 2:
            return float(r[0]), float(r[1])
        allv = None
        what, b = str(cfg.get("what", "mesh")), str(cfg["block"])
        try:
            if what == "mesh":
                for name, _nv, _pd, _sc, _ct in getattr(self, "_meshes", []) or []:
                    cb = getattr(H.level(name), "_cell_blocks", {}) or {}
                    if b in cb:
                        allv = np.asarray(cb[b], np.float64).reshape(-1)
                        break
            else:
                lv = H.level(what)
                ser = (getattr(lv, "_blocks", {}) or {}).get(b)
                if ser is not None:
                    v = np.asarray(ser, np.float64).reshape(ser.shape[0], ser.shape[1], -1)[:, :, 0]
                    oc = getattr(lv, "_occ", None)
                    if oc is not None:
                        oc = oc.detach().cpu().numpy() if hasattr(oc, "detach") else np.asarray(oc)
                        v = v[np.asarray(oc).reshape(v.shape) > 0.5]
                    allv = v.reshape(-1)
        except Exception:                                            # noqa: BLE001 -- the frame's own range
            allv = None
        pool = allv if allv is not None else np.asarray(now, np.float64)
        pool = pool[np.isfinite(pool) & (pool > 0)]
        if not pool.size:
            return 0.0, 1.0
        lo, hi = float(np.percentile(pool, 2)), float(np.percentile(pool, 98))
        # A BLOCK THAT DOES NOT VARY (the membrane's mass at its rest point, 1 everywhere) sat at one end of the map
        # -- black on black for magma. It is drawn mid-map instead, on a range from 0 to twice its value.
        if hi - lo < 0.05 * max(abs(hi), 1e-12):
            mid = 0.5 * (lo + hi)
            lo, hi = 0.0, (2.0 * mid if mid > 0 else 1.0)
        return lo, hi

    def _pin_frame(self, H, pin):
        from matplotlib import colormaps
        cfg, ren = pin["cfg"], pin["ren"]
        what, b = str(cfg.get("what", "mesh")), str(cfg["block"])
        cmap = colormaps[str(cfg.get("cmap", "viridis"))]
        if what == "mesh":
            got = None
            for name, _nv, mpd, _sc, _ct in getattr(self, "_meshes", []) or []:
                lvl = H.level(name)
                m = getattr(lvl, "mesh", None)
                if m is None:
                    continue
                cc = self._cell_cols(H, lvl, int(m["nF"]))
                if b in cc:
                    _v = cc[b]
                    _v = _v.detach().cpu().numpy() if hasattr(_v, "detach") else _v   # live: a CUDA tensor
                    got = (mpd, np.asarray(_v, np.float64))
                    break
            if got is None:
                return
            mpd, v = got
            if pin["rng"] is None:
                pin["rng"] = self._pin_range(H, cfg, v)
                print(f"[live-movie] protein_insets: {cfg.get('label', b)} ({b}) coloured over [{pin['rng'][0]:.4g}, "
                      f"{pin['rng'][1]:.4g}] ({'declared' if isinstance(cfg.get('range'), (list, tuple)) else 'the whole run, 2nd-98th percentile'})", flush=True)
            lo, hi = pin["rng"]
            if pin["pd"] is None:
                pin["pd"] = self.pv.PolyData()
            pd = pin["pd"]
            pd.copy_from(mpd)
            t = np.clip((v[: pd.n_cells] - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
            rgb = (cmap(np.where(np.isfinite(t), t, 0.0))[:, :3] * 255).astype(np.uint8)
            if rgb.shape[0] < pd.n_cells:
                rgb = np.vstack([rgb, np.zeros((pd.n_cells - rgb.shape[0], 3), np.uint8)])
            pd.cell_data["pin_rgb"] = rgb
            pd.Modified()
            if not pin.get("actor"):
                pin["actor"] = self._pin_actor(ren, pd, cells=True)
        else:
            lv = H.level(what)
            P, live = self._also_xyz(lv)
            if P is None:
                return
            vals = lv.get(b)
            vals = np.asarray(vals.detach().cpu().numpy() if hasattr(vals, "detach") else vals, np.float64).reshape(P.shape[0], -1)[:, 0]
            keep = live if live is not None else np.ones(P.shape[0], bool)
            keep = keep & np.isfinite(P).all(1)
            if cfg.get("skip_zero"):                  # e.g. a membrane node laid this frame, its mass not yet set (bm_mass
                keep = keep & (vals != 0)             # runs first next frame): drawn as a black speck, it is not a value
            if pin["rng"] is None:
                pin["rng"] = self._pin_range(H, cfg, vals[keep])
                print(f"[live-movie] protein_insets: {cfg.get('label', b)} ({b}) coloured over [{pin['rng'][0]:.4g}, "
                      f"{pin['rng'][1]:.4g}] ({'declared' if isinstance(cfg.get('range'), (list, tuple)) else 'the whole run, 2nd-98th percentile'})", flush=True)
            lo, hi = pin["rng"]
            t = np.clip((vals[keep] - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
            rgb = (cmap(t)[:, :3] * 255).astype(np.uint8)
            pts = self.pv.PolyData(P[keep].astype(np.float32))
            pts.point_data["pin_rgb"] = rgb
            if pin["pd"] is None:
                pin["pd"] = self.pv.PolyData()
                pin["pd"].copy_from(pts)
                pin["actor"] = self._pin_actor(ren, pin["pd"], cells=False, point_size=float(cfg.get("point_size", 3.0)))
            else:
                pin["pd"].copy_from(pts)
            pin["pd"].Modified()

    def _protein_bars(self, H):
        st = self.style or {}
        bars = []
        if str(st.get("mesh_color_scale", "")).lower() == "continuous" and st.get("mesh_color_range"):
            lo, hi = (float(x) for x in st["mesh_color_range"])
            bars.append((str(st.get("mesh_color_label", st.get("mesh_color_by", ""))), str(st.get("mesh_cmap", "viridis")), lo, hi))
        for nm in self._also_names(H):
            cb = self._block_cfg(nm)
            if cb is not None and cb.get("range"):
                lo, hi = (float(x) for x in cb["range"])
                bars.append((str(cb.get("label", cb["block"])), str(cb.get("cmap", "magma")), lo, hi))
        x0 = float(st.get("protein_bars_x", 0.60))
        for k, (title, cmap, lo, hi) in enumerate(bars):
            dummy = self.pv.PolyData(np.zeros((2, 3)))
            dummy["v"] = np.array([lo, hi], np.float32)
            args = {**self._legend_args(title), "position_x": x0 + 0.075 * k, "position_y": 0.06, "height": 0.25,
                    "title_font_size": int(st.get("protein_bars_font_size", 16)), "label_font_size": int(st.get("protein_bars_font_size", 16)) - 2}
            self.p.add_mesh(dummy, scalars="v", cmap=cmap, clim=[lo, hi], opacity=0.0, show_scalar_bar=True,
                            scalar_bar_args=args)
        if bars:
            print(f"[live-movie] protein_bars: " + ", ".join(f"{t} [{lo:g}, {hi:g}] {c}" for t, c, lo, hi in bars), flush=True)

    def _also_update(self, H):
        for nm, mpd in getattr(self, "_also_mesh", []) or []:
            try:
                P, live = self._also_xyz(H.level(nm))
                faces = self._hull_faces(P, live)
                if faces is not None:
                    # `near_side: far` CUTS THE SHEET TOO: the membrane's front half would hide a cut-open body
                    if str((self.style or {}).get("near_side", "") or "").lower() == "far":
                        tri = faces.reshape(-1, 4)[:, 1:]
                        n = np.asarray(self.p.camera.position, float) - np.asarray(self.p.camera.focal_point, float)
                        n = n / max(float(np.linalg.norm(n)), 1e-12)
                        _nc = (self.style or {}).get("near_side_centre")
                        ctr = np.asarray(_nc, float) if _nc is not None else P[live].mean(0) if live is not None else P.mean(0)
                        keep = ((P[tri].mean(1) - ctr) @ n) <= 0.0
                        faces = np.hstack([np.full((int(keep.sum()), 1), 3), tri[keep]]).astype(np.int64).ravel()
                    mpd.copy_from(self.pv.PolyData(P, faces))
                    if self._block_cfg(nm) is not None:
                        mpd.point_data["rgb"] = self._also_rgb(H, nm, H.level(nm), P.shape[0], live)
            except Exception as e:                                   # noqa: BLE001
                print(f"[live-movie] also_mesh: {nm!r} not updated ({type(e).__name__}: {e})", flush=True)
        if getattr(self, "_iso", None) is not None:
            try:
                cfg, surf = self._iso
                new = self._iso_cells_poly(H, cfg)
                if new is not None:
                    surf.copy_from(new)
                    surf.set_active_scalars("c")
            except Exception as e:                                   # noqa: BLE001
                print(f"[live-movie] iso_cells not updated ({type(e).__name__}: {e})", flush=True)
        for nm, pd in getattr(self, "_also", []) or []:
            try:
                lv = H.level(nm)
                P, live = self._also_xyz(lv)
                if P is None or P.shape[0] != pd.n_points:
                    continue
                pd.points = P
                pd["rgb"] = self._also_rgb(H, nm, lv, P.shape[0], live)
            except Exception as e:                                   # noqa: BLE001
                print(f"[live-movie] also_sets: {nm!r} not updated ({type(e).__name__}: {e})",
                      flush=True)

    def _spheres_build(self, H):
        """Draw a set's elements as SPHERES OF A WORLD RADIUS, `plotting.spheres`.

            plotting:
              spheres: {set: cell, radius: 0.04, color: "#f2d24e"}

        WHY NOT A BIG DOT. `render_points_as_spheres` is a screen-space sprite sized in PIXELS,
        so a cell drawn that way is the same size at every zoom and never the size the model says
        it is. A cilium 20 um long standing on a cell 4 um across has to be drawn on a cell 4 um
        across, or the picture states a different anatomy from the spec. This is a real mesh at
        the declared radius, one per element, moved with the element each frame.
        """
        cfg = (self.style or {}).get("spheres")
        if not cfg:
            return
        cfg = dict(cfg)
        names = cfg.get("set", "")
        names = [str(n) for n in (names if isinstance(names, (list, tuple)) else [names])]
        r = float(cfg.get("radius", 0.02))
        unit = self.pv.Sphere(radius=r, theta_resolution=24, phi_resolution=24)
        self._sph_unit = np.asarray(unit.points, np.float32).copy()
        self._sph_meshes = []                        # (mesh, set name, element index)
        for name in names:
            if name not in H.levels:
                print(f"[live-movie] spheres: no set {name!r}; nothing drawn", flush=True)
                continue
            pos = np.asarray(H.level(name).get("pos").detach().cpu().numpy(), np.float32)
            _cb = self._block_cfg(name)
            _cols = None
            if _cb is not None and getattr(self, "_H_now", None) is not None:
                import matplotlib.pyplot as _plt
                _v, _lo, _hi, _cmap, _label = self._block_values(self._H_now, name, _cb)
                _cols = _plt.get_cmap(_cmap)((np.clip(_v, _lo, _hi) - _lo) / max(_hi - _lo, 1e-12))[:, :3]
                print(f"[live-movie] spheres of {name!r} coloured by {_label}, range [{_lo:.4g}, {_hi:.4g}]", flush=True)
            for i in range(pos.shape[0]):
                m = unit.copy()
                m.points = self._sph_unit + pos[i][None, :]
                # MATERIAL FROM THE SPEC, glossy plastic by default as it always was: `specular`,
                # `ambient`, `diffuse`, `smooth` on the `spheres` block. `specular: 0` is the matte
                # ball a stator complex is drawn as in a cryo-EM figure (exp_02 step 0013).
                _a = self.p.add_mesh(m, color=(tuple(_cols[i]) if _cols is not None else str(cfg.get("color", "#f2d24e"))),
                                smooth_shading=bool(cfg.get("smooth", True)),
                                specular=float(cfg.get("specular", 0.3)),
                                ambient=float(cfg.get("ambient", 0.0)),
                                diffuse=float(cfg.get("diffuse", 1.0)),
                                opacity=float(cfg.get("opacity", 1.0)), **self._silhouette())
                self._sph_meshes.append((m, name, i, _a))
            print(f"[live-movie] spheres: {pos.shape[0]} sphere(s) of radius {r:g} from {name!r}",
                  flush=True)

    def _spheres_visible(self, pos):
        """`spheres.within: {centre: [x, y, z], radius: r, half_height: h}` (world) -- ONLY THE ELEMENTS
        NEAR THE SUBJECT ARE DRAWN: a cylinder about the axis through `centre`. A potassium channel in
        a bath of 255 ions is a starfield with a protein somewhere in it; the ions in and at the pore
        are the story, and the rest are simulated but not drawn (exp04, 2026-09-26). With
        `near_side: far` the elements on the camera's side of the cut are hidden too, as the
        surfaces are. Absent: every element drawn, as before."""
        cfg = (self.style or {}).get("spheres") or {}
        w = cfg.get("within")
        vis = np.ones(pos.shape[0], bool)
        if w:
            c = np.asarray(w.get("centre", [0.5, 0.5, 0.5]), float)
            r = np.hypot(pos[:, 0] - c[0], pos[:, 1] - c[1])
            vis &= r <= float(w.get("radius", 1e9))
            vis &= np.abs(pos[:, 2] - c[2]) <= float(w.get("half_height", 1e9))
        ns = (self.style or {}).get("near_side")
        if isinstance(ns, str) and ns.lower() == "far" and getattr(self, "p", None) is not None:
            n = np.asarray(self.p.camera.position, float) - np.asarray(self.p.camera.focal_point, float)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            ctr = np.asarray((self.style or {}).get("near_side_centre", pos.mean(0)), float)
            vis &= ((pos - ctr) @ n) <= 0.0
        return vis

    def _spheres_update(self, H):
        _cache = {}
        _vis = {}
        for m, name, i, _a in getattr(self, "_sph_meshes", []) or []:
            pos = np.asarray(H.level(name).get("pos").detach().cpu().numpy(), np.float32)
            m.points = self._sph_unit + pos[i][None, :]
            if (((self.style or {}).get("spheres") or {}).get("within")
                    or isinstance((self.style or {}).get("near_side"), str)):
                if name not in _vis:
                    _vis[name] = self._spheres_visible(pos)
                if not bool(_vis[name][i]):
                    # COLLAPSED, NOT HIDDEN: the silhouette is its own actor, fed by this mesh, so an
                    # actor switched off left its outline circle behind (Kv1.2 check, 2026-09-26);
                    # a sphere of radius zero draws neither
                    m.points = np.repeat(pos[i][None, :], self._sph_unit.shape[0], 0)
            _cb = self._block_cfg(name)
            if _cb is not None:
                if name not in _cache:
                    import matplotlib.pyplot as _plt
                    _v, _lo, _hi, _cmap, _ = self._block_values(H, name, _cb)
                    _cache[name] = _plt.get_cmap(_cmap)((np.clip(_v, _lo, _hi) - _lo) / max(_hi - _lo, 1e-12))[:, :3]
                try:
                    _a.prop.color = tuple(float(c) for c in _cache[name][i])
                except Exception:                                    # noqa: BLE001
                    pass

    def _glyph_cover_subject(self, H, lvl):
        """Build every set's glyphs, and say whether THE SUBJECT is among them.

        WHY THE TWO QUESTIONS ARE DIFFERENT. `_glyph_build` draws glyphs for every typed set that
        carries a `dot_radius`, not only the subject, and returns whether it drew ANY -- and the
        caller used that to decide whether to draw the subject's point CLOUD. So naming a radius
        for one small set silently deleted the cloud of a different, larger one: a scene of 74
        cilia and 2,000 yolk points over 140,000 water particles rendered the first two as spheres
        and the water not at all, because the water had no radius of its own and something else
        did. A set is drawn as glyphs or as a cloud; which of the two is a property of THAT set.
        """
        built = self._glyph_build(H, lvl)
        return built and bool(self._glyph_types(lvl))

    def _glyph_build(self, H, lvl):
        """EVERY typed point set with a `dot_radius`, not only the subject. A tissue holds its
        proteins in one set and its organelles in another; the subject is the larger of them and
        the other was not drawn at all. One glyph actor per (set, type), named `glyph_<set>_<type>`."""
        self._glyphs = {}
        self._glyph_geom = getattr(self, "_glyph_geom", {})
        said = []
        for lname, lv in H.levels.items():
            try:
                if lv.get("pos") is None:
                    continue
            except Exception:                                    # noqa: BLE001
                continue
            types = self._glyph_types(lv)
            if not types:
                continue
            for tid, nm, r, col in types:
                key = f"{lname}_{nm}"
                self._glyph_geom[key] = self.pv.Sphere(radius=r, theta_resolution=14, phi_resolution=10)
                self._glyphs[key] = (lname, tid, col, None)
                said.append(f"{lname}.{nm} r={r:g}")
        if not self._glyphs:
            return False
        self._glyph_update_all(H)
        print(f"[live-movie] dot_radius: {', '.join(said)} drawn as spheres", flush=True)
        return True

    def _glyph_update(self, lvl):
        """Per-frame refresh; `lvl` is the subject (its points are the cloud's)."""
        H = getattr(self, "_glyph_H", None)
        if H is not None:
            self._glyph_update_all(H)

    def _fi_empty(self):
        """An EMPTY contour, drawn as nothing: a single placeholder POINT at the origin was drawn as a dot
        in the box's corner in every frame without current (the 'orange speck' of rounds 5-6)."""
        c = 0.5 * np.asarray(self.world, np.float32)
        return self.pv.PolyData(np.stack([c, c, c]).astype(np.float32), faces=np.array([3, 0, 1, 2]))

    def _clip_far(self, pd):
        """The half of `pd` behind the plane through `near_side_centre` normal to the view, cut by a
        plane (a clean edge), for `near_side: far` on a re-contoured surface."""
        try:
            n = np.asarray(self.p.camera.position, float) - np.asarray(self.p.camera.focal_point, float)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            _nc = (self.style or {}).get("near_side_centre")
            ctr = np.asarray(_nc, float) if _nc is not None else np.asarray(pd.points, float).mean(0)
            out = pd.clip(normal=tuple(n), origin=tuple(ctr), invert=True)
            out = out.extract_surface() if hasattr(out, "extract_surface") else out
            return out.compute_normals(auto_orient_normals=False) if out.n_points else pd
        except Exception:                                            # noqa: BLE001 -- keep the uncut shell
            return pd

    def _near_side_faces(self, pd, f=None):
        """`near_side` on a SURFACE: keep the polygons whose centroid is in front of the plane
        through the body's centre, normal to the view. A hollow shell drawn whole hides its own
        interior and shows every far-side piece through it; cut, you look INTO the tissue.
        `f` is the mode for this one surface (`surface.<set>.near_side`); None reads the style's."""
        if f is None:
            f = (self.style or {}).get("near_side")
        if not f or pd is None or pd.n_points == 0:
            return
        full = getattr(pd, "_full_faces", None)
        if full is None:
            full = np.asarray(pd.faces).copy()
            pd._full_faces = full
        P = np.asarray(pd.points, float)
        c = np.asarray(self.p.camera.position, float) - np.asarray(self.p.camera.focal_point, float)
        n = c / max(float(np.linalg.norm(c)), 1e-12)
        # ONE PLANE FOR EVERY SURFACE when `near_side_centre` names a point: a motor of thirteen
        # parts cut each at its own centre is thirteen different sections (the hook's plane is
        # not the C-ring's); through a declared point on the axis it is one section of one machine.
        _nc = (self.style or {}).get("near_side_centre")
        ctr = np.asarray(_nc, float) if _nc is not None else P.mean(0)
        out, i, keep_n = [], 0, 0
        span = float(np.abs((P - ctr) @ n).max() or 1.0)
        # `near_side: far` -- THE CUT-AWAY: keep the half BEHIND the plane, so the camera looks into
        # the body through the section -- a channel's lumen and whatever flows in it, as a figure
        # opens a map by clipping its front half. `true` (or a fraction) keeps the near half, which
        # for a translucent shell hides the far-side clutter and for an opaque one hides the inside.
        far = isinstance(f, str) and f.lower() == "far"
        cut = 0.0 if (f is True or far) else float(f) * span
        while i < len(full):
            k = int(full[i]); idx = full[i + 1:i + 1 + k]
            d_ = float(((P[idx].mean(0) - ctr) @ n))
            if (d_ <= -cut) if far else (d_ >= cut):
                out.append(full[i:i + 1 + k]); keep_n += 1
            i += 1 + k
        pd.faces = np.concatenate(out) if out else np.zeros(0, np.int64)

    def _light_inside(self, on: bool):
        """A CUT SHELL SHOWS ITS INSIDE, whose normals point away from the camera: lit from the
        front only, the interior came out almost black. Two-sided lighting and a lifted ambient
        make the inner wall of the tissue read as the same material seen from within."""
        for _n, _nv, pd, _sc, _ct in getattr(self, "_meshes", []) or []:
            # a triangulated fill's mapper reads the filter's output, not `pd` -- see `_add_meshes`
            act = (getattr(self, "_mesh_fill", {}) or {}).get(id(pd))
            for a in ([] if act is not None else self.p.renderer.actors.values()):
                try:
                    if a.GetMapper() is not None and a.GetMapper().GetInput() is pd:
                        act = a
                        break
                except Exception:                                # noqa: BLE001
                    continue
            if act is None:
                continue
            p = act.GetProperty()
            if on:
                self._mesh_light = getattr(self, "_mesh_light", {})
                self._mesh_light.setdefault(id(pd), (p.GetAmbient(), p.GetDiffuse(), p.GetOpacity()))
                p.SetAmbient(0.55); p.SetDiffuse(0.55); p.SetBackfaceCulling(False)
                p.SetOpacity(min(1.0, float(p.GetOpacity()) * 2.2))
            else:
                a0 = (getattr(self, "_mesh_light", {}) or {}).get(id(pd))
                if a0:
                    p.SetAmbient(a0[0]); p.SetDiffuse(a0[1]); p.SetOpacity(a0[2])

    def _skin_cut(self, s):
        """The cut a compartment surface takes NOW: the watcher's slice when it is on (`_slice_all`, every surface
        cut away through one plane), else the surface's own `near_side`, else the style's. A surface built with no
        cut of its own carries None and follows the style as it is now -- it read the style once, at build, so a cut
        switched on later never reached the protein chains (the watcher's slice button, 2026-09-26)."""
        if getattr(self, "_slice_all", False):
            return "far"
        ns = s.get("near_side")
        return ns if ns is not None else (self.style or {}).get("near_side")

    def near_side_refresh(self):
        """Re-cut everything the view hides: the surfaces and the sphere glyphs. Called when the
        switch is thrown and whenever the camera turns -- the cut is defined BY the camera."""
        for s in getattr(self, "_skins", None) or []:
            if not s or s.get("kind") != "surface" or s.get("recontour"):
                continue                                          # a re-contoured skin is cut at its next frame
            try:
                ns_ = self._skin_cut(s)
                if ns_:
                    self._near_side_faces(s["surf"], ns_)
                elif getattr(s["surf"], "_full_faces", None) is not None:
                    s["surf"].faces = s["surf"]._full_faces
            except Exception:                                    # noqa: BLE001
                pass
        for _n, _nv, pd, _sc, _ct in getattr(self, "_meshes", []) or []:
            try:
                if not (self.style or {}).get("near_side"):
                    full = getattr(pd, "_full_faces", None)
                    if full is not None:
                        pd.faces = full
                else:
                    self._near_side_faces(pd)
            except Exception:                                    # noqa: BLE001
                pass
        self._light_inside(bool((self.style or {}).get("near_side")))
        if getattr(self, "_glyphs", None) and getattr(self, "_glyph_H", None) is not None:
            self._glyph_update_all(self._glyph_H)

    def _near_side(self, pts):
        """`plotting.near_side` -- KEEP ONLY WHAT FACES THE CAMERA. A shell of cells shows every
        nucleus at once, the near ones and the far ones through the surface, and 120 spheres over
        one another read as a cloud rather than as one per cell. The cut is the plane through the
        drawn body's centre, normal to the view: a point behind it is on the side you are looking
        at the back of. `near_side` may be true (the centre) or a fraction of the body's radius,
        so 0.3 keeps a shallower cap and -0.2 keeps a little past the equator."""
        f = (self.style or {}).get("near_side")
        if not f or not len(pts):
            return pts
        c = np.asarray(self.p.camera.position, float) - np.asarray(self.p.camera.focal_point, float)
        n = c / max(float(np.linalg.norm(c)), 1e-12)
        ctr = np.asarray(getattr(self, "_glyph_centre", None) if getattr(self, "_glyph_centre", None) is not None
                         else np.asarray(pts, float).mean(0), float)
        d = (np.asarray(pts, float) - ctr) @ n
        if isinstance(f, str) and f.lower() == "far":                 # the cut-away: keep the far half
            keep = d <= 0.0
        else:
            cut = 0.0 if f is True else float(f) * float(np.abs(d).max() or 1.0)
            keep = d >= cut
        return np.asarray(pts)[keep] if keep.any() else np.asarray(pts)[:0]

    def _glyph_update_all(self, H):
        g = getattr(self, "_glyphs", None)
        if not g:
            return
        self._glyph_H = H
        subject = getattr(self, "_sname", None)
        # THE CENTRE THE CUT IS MEASURED FROM is the whole drawn body's, not each type's: the Golgi
        # of one hemisphere must not be judged against the Golgi's own centre.
        if (self.style or {}).get("near_side"):
            _all = [self._glyph_points(H.level(ln), ti, ln == subject) for ln, ti, _c, _a in g.values()]
            _all = [p for p in _all if len(p)]
            self._glyph_centre = np.concatenate(_all, 0).mean(0) if _all else None
        for key, (lname, tid, col, actor) in list(g.items()):
            lv = H.level(lname)
            pts = self._near_side(self._glyph_points(lv, tid, lname == subject))
            if actor is not None:
                self.p.remove_actor(actor, render=False)
                actor = None
            if len(pts) > GLYPH_MAX:
                # A SPHERE PER POINT HAS A CEILING. 140 triangles a sphere, 300k spheres is 42 M
                # triangles and VTK dies allocating them (`std::bad_array_new_length`, seen when a
                # prompt asked for "a lot of integrins"). Past the ceiling a type is drawn as lit
                # point sprites at the pixel size its radius has on screen -- the same picture at
                # a distance, and it costs nothing.
                r = float(self._glyph_geom[key].bounds[1] - self._glyph_geom[key].bounds[0]) / 2.0
                ps = float(self.p.camera.parallel_scale) if self.p.camera.parallel_projection else None
                px = max(1.0, r / ps * self.p.window_size[1] / 2.0) if ps else 3.0
                actor = self.p.add_mesh(self.pv.PolyData(pts), color=col, render_points_as_spheres=True,
                                        point_size=px, opacity=self._glyph_opacity(key),
                                        name=f"glyph_{key}")
            elif len(pts):
                pd = self.pv.PolyData(pts).glyph(geom=self._glyph_geom[key], scale=False, orient=False)
                actor = self.p.add_mesh(pd, color=col, smooth_shading=True, lighting=True, ambient=0.35,
                                        diffuse=0.7, specular=0.2,
                                        opacity=self._glyph_opacity(key), name=f"glyph_{key}")
            g[key] = (lname, tid, col, actor)

    def _dot_px(self, pos):
        """Dot diameter in pixels such that a dot spans `fill` of the local spacing of the DRAWN
        subset -- measured, as the median nearest-neighbour distance, not estimated from density.

        `sqrt(volume / n)` needs a hull, is wrong for any non-convex or non-uniform layout and is
        biased at the boundary; the median nearest-neighbour distance is the quantity "nearly
        touching" refers to and the median shrugs off the boundary points. Measured once, at t=0,
        on a sample: it is a property of how the material was seeded, and re-measuring it every
        frame would make the dots breathe as the fluid compresses.
        """
        # A SLICE THINS THE CLOUD, SO THE DOTS MUST GROW. `cross_section.only` keeps a slab a few
        # cells thick and parks the rest outside the box, so of the particles that remain VISIBLE
        # only a few percent survive -- 2 of 41 z-cells on si_ball_wake. The dot size was measured
        # against the spacing of the DRAWN set, which still counts the parked ones, so the slab
        # renders as sparse specks with gaps between them and reads as a much emptier fluid than it
        # is. Doubling the diameter restores roughly the coverage the full cloud had.
        _slab = bool(self.cs is not None or (self.style or {}).get("cross_section"))
        _k = 2.0 if _slab else 1.0
        if self.dot != "auto":
            self.px_used = float(self.dot) * _k
            return self.px_used
        q = pos[:, :3]
        if len(q) > 20000:                          # the median converges long before the full set
            q = q[np.random.default_rng(0).choice(len(q), 20000, replace=False)]
        try:
            from scipy.spatial import cKDTree
            nn = cKDTree(q).query(q, k=2)[0][:, 1]  # k=2: a point's nearest neighbour is itself at 0
            # A SET OF ONE HAS NO SPACING, AND THE MEDIAN OF NOTHING IS NaN. `query(k=2)` on a
            # single point returns `inf` for the neighbour that does not exist, the finite filter
            # then empties the array, and `np.median([])` is NaN -- which passes the `sp <= 0` test
            # below, survives `np.clip` unchanged, and reaches VTK as a point size of NaN. Nothing
            # raises and nothing is drawn: the demo's opening rung, one cell alone in a 4x4 world,
            # rendered 200 frames of an empty box while its trajectory said the cell was exactly
            # where it was seeded (displacement 0.0 over 201 rows). An invisible entity reads as a
            # model that did nothing, which is the most expensive kind of wrong picture here.
            _fin = nn[np.isfinite(nn)]
            sp = float(np.median(_fin)) if _fin.size else 0.0
        except Exception:
            sp = 0.0
        span = float((self.hi - self.lo).max()) or 1.0
        if sp <= 0:
            # 1.5 px is a LAST RESORT, not a size anyone chose: with no spacing to measure there is
            # no scale in the drawn set at all. Say so, because a one-and-a-half-pixel dot on a
            # 1280 px frame is easy to mistake for nothing being drawn -- which is exactly the
            # confusion this branch now exists to end. `plotting.dot_size: <px>` overrides it.
            print(f"[live-movie] no spacing to measure from {len(pos):,} drawn point(s) -- dot "
                  f"falls back to 1.5 px. Set `plotting.dot_size` to choose one.", flush=True)
            self.px_used = 1.5 * _k
        else:
            # world -> px through the parallel projection: the camera frames `parallel_scale`
            # half-heights over the window's half-height.
            world_per_px = (2.0 * self.p.camera.parallel_scale) / max(self.p.window_size[1], 1)
            self.px_used = float(np.clip(_k * self.fill * sp / max(world_per_px, 1e-12), 0.7, 24.0))
        print(f"[live-movie] dot {self.px_used:.2f} px  (median spacing of the {len(pos):,} drawn "
              f"= {sp:.3e} world, box {span:.3g})", flush=True)
        return self.px_used

    def _still(self, tick):
        """Write the frame just rendered as a PNG. `3d.png` is always the newest, so a long run can
        be watched from the file browser; the numbered copies survive so the run leaves a strip."""
        try:
            import imageio.v3 as iio
            img = self.p.image                      # the frame write_frame() just rasterised
            n = f"{self.stills_written:02d}"
            _p = os.path.join(self.still_dir, f"still_{n}_f{tick:05d}.png")
            iio.imwrite(_p, img)
            iio.imwrite(os.path.join(self.still_dir, "3d.png"), img)
            self._still_paths.append(_p)
            self.stills_written += 1
        except Exception as e:                      # a missing PNG must never end a 20-minute run
            print(f"[live-movie] still at frame {tick} failed: {type(e).__name__}: {e}", flush=True)

    def close(self):
        try:
            self.p.close()
        except Exception:
            pass
        if not self.keep_stills:
            _n = 0
            for _p in self._still_paths:
                try:
                    os.remove(_p); _n += 1
                except OSError:
                    pass
            if _n:
                self.stills_written = 0
                self._removed_stills = _n
        if self.failed or not self.rendered:
            print(f"[live-movie] wrote nothing ({self.failed or 'no frames rendered'})", flush=True)
            return None
        sub = f", {self.drawn:,} of them drawn" if self.drawn < self.n else ""
        print(f"[live-movie] {self.out}   {self.n:,} nodes{sub}, {self.rendered} frames"
              f"{'' if self.stride == 1 else f' (every {self.stride}th)'}, "
              # `px_used` IS None WHENEVER THE DOTS WERE NOT DRAWN -- `render_3d: surface` never
              # calls `_dot_px` -- and `{None:.2f}` raises. It raised in `close()`, i.e. AFTER every
              # frame was written and before the writer was closed, so the run "succeeded", the
              # summary never printed, and the mp4 was left unfinalised. A reporting line must not
              # be able to cost the artefact it is reporting on.
              f"coloured by {self.colour_by}"
              + (f", dot {self.px_used:.2f} px" if self.px_used is not None
                 else f", surface ({self._surf.n_faces_strict:,} faces)"
                 if getattr(self, "_surf", None) is not None else "")
              + (f", {self.n_obstacles} obstacle(s)" if self.n_obstacles else "")
              + (f", {self.stills_written} stills + 3d.png" if self.stills_written
                 else (f", {getattr(self, '_removed_stills', 0)} stills removed, 3d.png kept"
                       if getattr(self, "_removed_stills", 0) else "")),
              flush=True)
        return self.out


# ==========================================================================================================
#  REPLAY -- the same renderer, driven from a saved trajectory instead of a running engine
# ==========================================================================================================
# WHY THIS EXISTS RATHER THAN A SECOND POINT RENDERER. `plexus.plot` had its own 3D path -- a numpy
# gaussian splat -- so a 3D point set was drawn one way DURING generation (this class, VTK, real
# dots, obstacles, a box, a scale bar) and a different way afterwards (`-o plot`, soft blobs, no
# obstacles, its own camera keys). Two renderers for one kind of data is two sets of bugs and two
# looks, and nothing in the spec said which one a run would get.
#
# The engine hands this class an `H`: a state object it reads six things from. Replaying a saved run
# only needs those six to come from an npz instead, so the renderer itself is untouched -- which is
# the point. Anything fixed for the live path is fixed for the replay for free.
#
# WHAT REPLAY CANNOT DO, and says so: `color_field` (vorticity / pressure) needs `C` and `F`, the
# per-particle affine and deformation tensors, and a trajectory stores neither. Those colours are a
# live-only feature; the replay falls back to the type palette and prints that it did.
from plexus.measures import _MeshView  # noqa: E402  -- the read-only mesh view lives with the measures now

class _ReplayLevel:
    """One set of a trajectory.npz, shaped like the Level the renderer reads off the engine."""

    def __init__(self, z, name, dev, cell_set=None):
        import torch
        self._pos = torch.as_tensor(np.asarray(z[f"{name}__pos"], np.float32), device=dev)
        _occ = z[f"{name}__occ"] if f"{name}__occ" in z.files else None
        self._occ = None if _occ is None else torch.as_tensor(np.asarray(_occ), device=dev)
        self.n = int(self._pos.shape[1])
        self.state = self._pos[0]                     # only `.state.device` is ever read
        self.t = 0
        for k in ("node_type", "parent"):
            v = z[f"{name}__{k}"] if f"{name}__{k}" in z.files else None
            setattr(self, "_node_type" if k == "node_type" else k,
                    None if v is None else torch.as_tensor(np.asarray(v), device=dev))
        pn = z[f"{name}__parent_name"] if f"{name}__parent_name" in z.files else None
        self.parent_name = None if pn is None else str(pn)
        tn = z[f"{name}__type_names"] if f"{name}__type_names" in z.files else None
        self.type_names = None if tn is None else [str(x) for x in np.asarray(tn).tolist()]
        # a type column that changed during the run is recorded per row; `node_type` follows `t`
        nt_t = z[f"{name}__node_type_t"] if f"{name}__node_type_t" in z.files else None
        self._node_type_t = None if nt_t is None else torch.as_tensor(np.asarray(nt_t), device=dev)
        # THE PER-CHILD PARENT INDEX, WHICH THE TRAJECTORY HAS ALWAYS RECORDED AND THIS CLASS NEVER
        # READ. `dot_shading: body` groups dots by the body they belong to, so on the replay path --
        # every `-o plot` re-render -- it silently did nothing while the live path shaded. Same
        # attribute name as the engine's `Level.parent`, so one renderer reads both.
        self.parent = (torch.as_tensor(np.asarray(z[f"{name}__parent"]).astype(np.int64))
                       if f"{name}__parent" in z.files else None)
        self.C = self.F = None                        # not stored in a trajectory -- see above
        # EVERY RECORDED STATE BLOCK OF THIS SET, so `get` can serve them -- see `get`.
        _skip = ("pos", "occ", "node_type", "parent", "parent_name")
        self._blocks = {k[len(name) + 2:]: np.asarray(z[k]) for k in z.files
                        if k.startswith(f"{name}__") and "__mesh_" not in k
                        and k[len(name) + 2:] not in _skip}
        # THE CELL SET'S RECORDED BLOCKS, for a set that has a mesh. `cell_set` is the name the
        # spec declared and `replay` passed down; the structural arrays are excluded by name
        # because they are not per-cell quantities. See `cell_cols`.
        self._cell_blocks = {}
        if cell_set:
            skip = ("occ", "node_type", "parent", "parent_name", "pos")
            pre = f"{cell_set}__"
            for k in z.files:
                if not k.startswith(pre) or "__mesh_" in k:
                    continue
                b = k[len(pre):]
                if b in skip:
                    continue
                self._cell_blocks[b] = np.asarray(z[k])
        # AND ITS TYPE COLUMN, which is not a block: per row when a type changed during the run
        # (`__node_type_t`), the final column otherwise. Read only by `mesh_color_by: node_type`.
        self._cell_nt = None
        if cell_set:
            for _k in (f"{cell_set}__node_type_t", f"{cell_set}__node_type"):
                if _k in z.files:
                    self._cell_nt = np.asarray(z[_k])
                    break
        # THE HALF-EDGE TABLE IS IN THE TRAJECTORY AND WAS NOT BEING READ, so a replay of a
        # vertex-model run drew the mesh's VERTICES as dots while the same renderer, driven live
        # from the engine, drew the surface. One renderer that produces two different pictures of
        # one run depending on which entry point called it is two renderers wearing one name.
        #
        # RAGGED, HENCE THE OFFSETS. `nF` changes every frame under division, so the recorder
        # concatenates the per-frame half-edge arrays and stores `mesh_offsets` to cut them apart
        # again; `mesh_face_offsets` does the same for the per-face columns and is a DIFFERENT
        # array -- reading E_face with the face offsets gives one entry per face and a mesh that
        # renders as confetti.
        self._mo = np.asarray(z[f"{name}__mesh_offsets"]) if f"{name}__mesh_offsets" in z.files \
            else None
        if self._mo is not None:
            self._mesh_cols = {k: np.asarray(z[f"{name}__mesh_{k}"])
                               for k in ("E_srce", "E_trgt", "E_face")}
            self._mesh_nF = np.asarray(z[f"{name}__mesh_nF"])
            self._mesh_Nv = np.asarray(z[f"{name}__mesh_Nv"])
            # PER-FACE COLUMNS ON THEIR OWN OFFSETS. `age` and `ndiv` are what the division marks
            # are computed from, and they are cut by `mesh_face_offsets`, not by `mesh_offsets`:
            # one is a per-face array and the other per-half-edge, and reading either with the
            # other's offsets gives a mask that is the right dtype and the wrong length.
            self._fo = np.asarray(z[f"{name}__mesh_face_offsets"])
            # EVERY RECORDED FACE COLUMN, NOT A LITERAL FOUR. This was
            # `("age", "ndiv", "apop", "inhib")`, so a per-face quantity added to `FACE_RECORD`
            # later was written to the trajectory and never read back: the LIVE frames carried it
            # and the REPLAY -- which is what writes movie.mp4 and 3d.png -- did not. `cell_cycle`'s
            # `phase` hit exactly that, and the symptom is the confusing one: the run coloured
            # itself correctly frame by frame and the finished movie came out uniform, because two
            # passes were reading different meshes.
            #
            # It is the same defect this file already records one level down -- "the core recorded
            # NEITHER the dying-cell flag NOR the growth inhibitor, ever" -- and it recurred because
            # the fix there added names to a list instead of removing the list. `FACE_RECORD` is the
            # one place that says what a face column is; asking it cannot drift.
            from plexus.models.mesh import MeshTable as _MT
            self._face_cols = {k: np.asarray(z[f"{name}__mesh_{k}"])
                               for k in _MT.FACE_RECORD
                               if f"{name}__mesh_{k}" in z.files}
            # AND THE PER-ROW SCALARS, which are one number a frame rather than a column, so they
            # need no offsets at all. Omitting them is how the monolayer's thickness reached the
            # trajectory and still did not reach the picture: `scalar_mono_h` was recorded on every
            # frame, and the replay -- the pass that writes the movie that is kept -- handed the
            # renderer a mesh dict of six keys that did not include it, so the cross section drew the
            # mid-surface alone and a thick epithelium looked exactly like a thin one.
            self._scalar_cols = {k[len(name) + len("__mesh_"):]: np.asarray(z[k])
                                 for k in z.files
                                 if k.startswith(f"{name}__mesh_scalar_")}
            # AND THE PER-HALF-EDGE COLUMNS, on `mesh_offsets` like E_srce/E_trgt/E_face. `e_myo` is
            # the only quantity in this model that lives on a JUNCTION, so without it neither the
            # myosin curve nor a myosin edge colour can be drawn from a trajectory at all.
            # A THIRD OFFSETS ARRAY, and it is per COLUMN. `e_myo` is written only on the ticks its
            # operator ran, so it is NOT in step with `mesh_offsets` even though both are indexed by
            # half-edge: with frame 0 now the initial condition, junction_myosin has not run and
            # `e_myo_offsets` reads [0, 0, 1188, ...] -- an EMPTY row 0 -- while `mesh_offsets` reads
            # [0, 1188, 2376, ...]. Cutting one with the other shifts every frame by one and makes
            # the initial condition look like a frame of physics, which is exactly the symptom this
            # was chased for. Each column carries its own offsets; use them.
            self._edge_cols = {}
            for k in z.files:
                if not k.startswith(f"{name}__mesh_e_") or k.endswith("_offsets"):
                    continue
                c = k[len(name) + len("__mesh_"):]
                o = f"{k}_offsets"
                self._edge_cols[c] = (np.asarray(z[k]),
                                      np.asarray(z[o]) if o in z.files else self._mo)

    def cell_cols(self, nF):
        """{block: array[nF]} for this frame, from the CELL SET the spec paired with this mesh.

        THE REPLAY HALF OF `_mesh_levels._cell_cols`, and it exists because a cell set has no
        `pos`: it never becomes a level here, so the renderer cannot reach it the way it reaches
        the live one. The blocks are loaded once in `__init__` from the name `replay` read out of
        the spec -- not from a scan of the trajectory for a key that looks right.

        CUT TO `nF`, NOT TO THE BUFFER. The cell set is allocated to a capacity (12,800 slots for a
        run that ends near 5,000 cells) and the tail is stale, so a colour map over the whole array
        would paint thousands of dead slots. `nF` is the frame's live face count and a face IS a
        cell.
        """
        return {k: np.asarray(v[self.t])[:nF, 0] for k, v in self._cell_blocks.items()
                if v.ndim == 3 and v.shape[2] == 1}

    def cell_node_type(self, nF):
        """[nF] type of each face at this frame, from the paired cell set -- see `_cell_nt`."""
        a = getattr(self, "_cell_nt", None)
        if a is None:
            return None
        return np.asarray(a[min(int(self.t), len(a) - 1)] if a.ndim == 2 else a)[:nF]

    @property
    def mesh(self):
        """The frame's half-edge table, in the shape `_mesh_live` and `_mesh_faces` expect."""
        if self._mo is None:
            return None
        import torch
        a, b = int(self._mo[self.t]), int(self._mo[self.t + 1])
        d = {k: torch.as_tensor(v[a:b].astype(np.int64)) for k, v in self._mesh_cols.items()}
        d["nF"] = int(self._mesh_nF[self.t]); d["Nv"] = int(self._mesh_Nv[self.t])
        fa, fb = int(self._fo[self.t]), int(self._fo[self.t + 1])
        for k, v in self._face_cols.items():
            d[k] = v[fa:fb]
        for k, (v, o) in getattr(self, "_edge_cols", {}).items():
            ea, eb = int(o[self.t]), int(o[self.t + 1])  # this COLUMN's offsets, not the mesh's
            if eb > ea:
                d[k] = v[ea:eb]
        for k, v in getattr(self, "_scalar_cols", {}).items():
            d[k] = v[self.t]                             # one number a frame; no offsets to cut
        return d

    @property
    def node_type(self):
        t = getattr(self, "_node_type_t", None)
        return t[self.t] if t is not None else getattr(self, "_node_type", None)

    @property
    def occ(self):
        """Occupancy AS AN ATTRIBUTE, because that is how every consumer asks for it.

        `getattr(lvl, "occ", None)` is the idiom throughout the renderer, and on this class it
        returned None -- the value was reachable only through `get("occ")`. So on the REPLAY path,
        the pass that writes the movie that is kept, nothing masked the reservoir: a 25,584-slot
        vertex set with 5,052 live cells drew all 20,532 dead slots too. They are zero-initialised,
        so they sit at exactly (0,0,0) and land on top of each other -- one stray dot at the origin
        in the cross section of every mesh run, which reads as a particle and is an artefact.
        """
        return None if self._occ is None else self._occ[self.t]

    def get(self, key):
        """Any RECORDED state block of this set at the current frame, not just `pos` and `occ`.

        THIS RETURNED None FOR EVERYTHING ELSE, AND A CURVE FELL BACK WITHOUT SAYING SO. The
        `volume` curve asks for `sep` to compute the polyhedron volume -- the one the energy
        defends -- and falls back to the origin-referenced WEDGE volume when a run carries no
        separation. On the replay path `sep` is in the trajectory but this method did not serve it,
        so every saved movie and every `3d.png` plotted the wedge under a panel labelled "cell
        volume". The wedge is a cone from the world origin to the cell's ring, so its spread is
        dominated by WHERE a cell sits on the shell rather than by how big it is: on
        `cv_target_uniform` the panel showed a band of about +/-4 on a mean of 1.4, while the cells'
        actual volumes were 1.3725 +/- 0.0080 -- a tissue that is uniform to half a percent drawn
        as one that is not uniform at all.

        The live pass reads the real `Level` and was always right, so the two passes disagreed and
        only the wrong one was kept. That is the same live-versus-replay split that hid the
        cell-cycle phase colours, and the same fix: ask the file what it has instead of listing
        names.
        """
        if key == "pos":
            return self._pos[self.t]
        if key == "occ" and self._occ is not None:
            return self._occ[self.t]
        a = self._blocks.get(key)
        if a is None and key == "vel" and self._pos.shape[0] > 1:
            # VELOCITY IS NOT RECORDED, AND IT DOES NOT NEED TO BE. `vel` is declared
            # `record: False` in every spatial set, so a trajectory holds positions only -- which
            # is why `plotting.color_field: speed` was thrown away on every replay and the movie
            # came back coloured by type. But the positions ARE the velocity's integral: a forward
            # difference between consecutive RECORDED frames reconstructs it to first order, and
            # that is the same quantity the live pass plots.
            #
            # `_rec_dt` is the time between recorded FRAMES, not the tick: a strided trajectory
            # (`record_cap` below `n_frames`) skips ticks, and dividing by the tick would report a
            # speed too large by exactly the stride. Set by `replay`, which knows both numbers.
            t = min(int(self.t), self._pos.shape[0] - 2)
            return (self._pos[t + 1] - self._pos[t]) / float(getattr(self, "_rec_dt", 1.0) or 1.0)
        return None if a is None else a[self.t]


class _ReplayState:
    """The `H` the renderer expects: a name -> level mapping and nothing else."""

    def __init__(self, z, dev, cell_sets=None):
        # EVERY SET'S TYPES, not only the sets that carry positions. A vertex model's `cell` set has
        # `centroid`/`area`/`node_type` and NO `pos`, so it never becomes a level here -- and the curve's
        # per-type split, which indexes faces by the cell set's node_type, silently collapsed to one
        # series. The types are in the file either way.
        self.node_types = {k[: -len("__node_type")]: np.asarray(z[k])
                           for k in z.files if k.endswith("__node_type")}
        names = sorted({k[: -len("__pos")] for k in z.files if k.endswith("__pos")})
        cs = cell_sets or {}
        self.levels = {n: _ReplayLevel(z, n, dev, cell_set=cs.get(n)) for n in names}
        self.fields = {}
        self.dim = int(next(iter(self.levels.values()))._pos.shape[2]) if self.levels else 3

    def level(self, name):
        return self.levels[name]

    def seek(self, t):
        for lvl in self.levels.values():
            lvl.t = t


def replay(data_dir, sim, out=None, *, max_frames=300, render_n=500_000_000, stills=0,
           keep_stills=False, name=None, fps=None, traj=None):
    """Render `data_dir/trajectory.npz` with the live VTK point renderer. Returns the mp4 path.

    THE BOX IS TAKEN FROM THE DATA WHEN THE RUN LEFT ITS OWN. This renderer draws a wireframe box
    at [0, world] and frames the camera on it, which is right for a walled run and useless for a
    `boundary: free` one -- a galaxy encounter throws stars to several times the world size, so a
    12-unit box would be a small cube in the middle of a cloud that had left it. `frame_percentile`
    (the key `plot.py` already reads) frames on the central p% instead, so the ~20% of stars thrown
    out by a passage cannot set the scale for the 80% worth looking at.
    """
    import torch
    z = traj if traj is not None else np.load(os.path.join(data_dir, "trajectory.npz"))
    style = dict((sim.plotting or {}) if sim is not None else {})
    dev = torch.device("cpu")
    # THE MESH -> CELL PAIRING COMES FROM THE SPEC, WHICH IS WHERE IT WAS DECLARED. `sets.<s>.mesh`
    # with `sets.<s>.cell_set` is the same pair the engine puts on `Level.mesh_cell_set` for the
    # live path; reading it here is what lets the replay draw a per-cell block that no longer sits
    # on the mesh table -- `phase` today, `A0` and `age` later -- without scanning the trajectory
    # for a plausible-looking key.
    # THE PAIRING MOVED WITH S6 AND THIS READER DID NOT, which is why a replayed run drew no
    # per-cell block at all. `sets.<s>.cell_set` was retired: `mesh:` now names a declared HALF-EDGE
    # SET, and that set's `maps.face` names the cells -- the same chain `engine._link_mesh_maps`
    # walks to bind `Level.mesh_cell_set`. Reading the dead key left `_cs` empty, `_cell_blocks`
    # empty, and every per-cell quantity invisible on the replay path: `phase` fell back to height
    # colouring and its curve panel drew nothing, and `A0`, `age` and `V0f` would have too. The live
    # path was unaffected, so the two entry points drew different pictures of the same run.
    _cs = {}
    _sets = (sim.sets or {}) if sim is not None else {}
    for n, d in _sets.items():
        if not isinstance(d, dict) or not d.get("mesh"):
            continue
        h = _sets.get(d["mesh"])
        face = (h or {}).get("maps", {}).get("face") if isinstance(h, dict) else None
        # `cell_set:` SECOND, not first: specs predating S6 are still on disk beside their
        # trajectories, and a replay of one should keep working.
        face = face or d.get("cell_set")
        if face:
            _cs[n] = face
    H = _ReplayState(z, dev, cell_sets=_cs)
    if not H.levels:
        raise ValueError(f"{data_dir}: no set carries positions, nothing to render")
    sname = ((sim.plotting or {}).get("subject") if sim is not None else None) or _biggest_particle_set(H)
    lvl = H.levels[sname]
    P = lvl._pos                                       # [T, N, D]
    T, D = int(P.shape[0]), int(P.shape[2])

    # A ONE-FRAME TRAJECTORY CANNOT REPLACE A MOVIE, and until now it did. `save_data: false` tells
    # the engine to record a single frame as a stub, because the run's picture is the LIVE movie
    # written frame by frame while it computes. `-o generate` then captions by default, and the
    # captioner needs the mp4, so `plot_dataset` re-renders -- off the stub. Measured on
    # cell_atlas_bounce: a 200-frame live movie was overwritten by a 1-frame replay, and the
    # caption that followed described a still image as if it were a run ("the cluster expands until
    # the particles reach the boundaries", of a cell that fell and bounced). Every `save_data:
    # false` spec in the corpus -- the whole si_material family -- has been losing its movie this
    # way. The stub is still a legitimate thing to render if there is nothing to lose.
    # `< 3`, NOT `< 2`: a `save_data: false` stub records the FIRST and LAST tick, so it is two
    # frames, not one -- and two frames of a 400-frame run is a still with a flicker, not a movie.
    _out = out or os.path.join(data_dir, "movie.mp4")
    if T < 3 and os.path.exists(_out):
        print(f"[live-movie] replay skipped: trajectory.npz holds {T} frame(s) (the spec sets "
              f"`save_data: false`, so it is a stub), and {os.path.basename(_out)} already exists "
              f"-- keeping the movie the run wrote live rather than replacing it with a still. "
              f"Set `save_data: true` or a `record_cap` to make `-o plot` reproduce it.",
              flush=True)
        return _out

    # --- the box, and the shift that puts the cloud inside it ---
    ws = np.asarray(z["world_size"], np.float64) if "world_size" in z.files else None
    if ws is None or len(ws) != D:
        w = float(z["world"]) if "world" in z.files else float(getattr(sim, "world", 1.0))
        ws = np.full(D, w, np.float64)
    # THE BOX IS MEASURED OVER LIVE ROWS ONLY. A set with a dormant reserve parks its unused slots
    # far off-domain (integrins at -1e6), and a box that read them framed a 50-unit tissue in a
    # 10-metre cube. `occ` is what says which rows are content.
    _occ = getattr(lvl, "_occ", None)
    if _occ is not None and tuple(_occ.shape[:2]) == tuple(P.shape[:2]) and bool(_occ.any()):
        flat = P[_occ.bool()].reshape(-1, D).numpy()
    else:
        flat = P.reshape(-1, D).numpy()
    fp = style.get("frame_percentile")
    shift = None                                            # added to EVERY level's positions
    if fp is not None:
        q = 0.5 * (100.0 - float(fp))
        lo = np.percentile(flat, q, axis=0)
        hi = np.percentile(flat, 100.0 - q, axis=0)
        box = (hi - lo) * 1.06
        shift = -(0.5 * (lo + hi) - 0.5 * box)
    else:
        lo, hi = flat.min(0), flat.max(0)
        # A HAIR BELOW ZERO IS STILL [0, world]. The test was lo < -1e-6 x world, so one point a soft wall let
        # press 0.02 past x = 0 made the replay read the run as centred on the origin and shift every set by half
        # a box (T-76: exp 21's corral drawn half a box off its frame). A run centred on the origin reaches about
        # -world/2; 2 % of the box separates the two with room on both sides.
        if not bool(((hi - lo) > ws * 1.02).any()) and not bool((lo < -0.02 * ws).any()):
            box = ws                                            # already in [0, world]
        elif not bool(((hi - lo) > ws * 1.02).any()):
            box = ws                                            # fits [-world/2, world/2]
            shift = 0.5 * box
        else:
            box = (hi - lo) * 1.06
            shift = -(0.5 * (lo + hi) - 0.5 * box)
    # THE SHIFT MOVES EVERY SET, NOT ONLY THE ONE THE BOX WAS MEASURED ON. Shifting the subject
    # alone drew a tissue's mesh 25 units from the particles riding on it (integrins on a
    # free-boundary spheroid: the dots framed, the surface out of the frustum).
    if shift is not None:
        for _lv in H.levels.values():
            _p = getattr(_lv, "_pos", None)
            if _p is not None:
                _lv._pos = _p + torch.as_tensor(np.asarray(shift, np.float64), dtype=_p.dtype, device=_p.device)
    out = out or os.path.join(data_dir, "movie.mp4")
    # UNITS ONLY WHEN THEY WERE DECLARED. `Units` defaults to length_um 1.0 / time_s 1.0 with
    # `declared: False`, and handing those to the renderer would put a scale bar and a wall clock on
    # a run that has neither -- a bar reading "2 m" across a galaxy 12 dimensionless units wide.
    u = getattr(sim, "units", None)
    dec = bool(getattr(u, "declared", False))
    lm = LiveMovie(out=out, world=list(np.asarray(box, np.float64)), n_frames=T,
                   force_nN=(float(u.force_nN) if (dec and getattr(u, "force_nN", None) is not None) else None),
                   can_curve=True,      # a replay holds every frame; the live path does not
                   up=int(style.get("up_axis", 2)), render_n=render_n, max_frames=max_frames,
                   name=name or getattr(sim, "name", ""), sim=sim, style=style,
                   stills=stills, keep_stills=keep_stills,
                   dt=getattr(sim, "dt", None),
                   time_s=(float(u.time_s) if dec else None),
                   length_um=(float(u.length_um) if dec else None),
                   **({"fps": float(fps)} if fps else {}))
    # A REPLAY'S ROW IS A RECORD, NOT A FRAME: `curve_time.per_frame_s` is the time of one simulation frame, and
    # a 300,000-frame run kept 3,001 records -- its panels' time axis read 0-0.47 ns for 47 ns (exp04 step 0050,
    # 2026-09-26). The series' rows are scaled by the frames each record stands for.
    _nf = getattr(sim, "n_frames", None)
    if _nf and T > 1:
        lm._frames_per_row = float(_nf) / float(T - 1)
    # THE STAMP REPORTS WHAT THE RUN COST, NOT WHAT THE PICTURE COST, when the trajectory carries it.
    #
    # `_rate_of` was always "render" here, because a replay can only time itself -- and a figure
    # labelled `ms/frame` on a movie of a simulation reads as the simulation's speed. `engine.run`
    # now records one wall-clock reading per simulated tick and the writer stores it, so the number
    # stamped is the median of THOSE. The median and not the mean: tick 0 builds every run-constant
    # cache and a `torch.compile` or CUDA-graph capture lands on tick 1, so a mean over a short clip
    # is dominated by warm-up that no later frame pays.
    _fms = np.asarray(z["frame_ms"], np.float64) if "frame_ms" in z.files else None
    if _fms is not None and _fms.size > 2:
        lm._rate_of = "compute"
        lm._fixed_ms = float(np.median(_fms[1:]))
    else:
        lm._rate_of = "render"
    # SPEED SURVIVES A REPLAY; THE TENSOR FIELDS DO NOT, and lumping them together cost the better
    # artefact every time. `F` and `C` are solver state and are not recorded -- 9 floats per
    # particle per frame is 1.8 GB on a run this size -- so `deformation`, `strain`, `volume`,
    # `pressure` and `vorticity` genuinely cannot be drawn from a file. `speed` can: it is the
    # forward difference of the recorded positions, which `_ReplayLevel.get('vel')` now returns,
    # and it is the same quantity the live pass plots.
    #
    # WHY IT MATTERED. `-o generate` renders live (with the field) and then the caption pass
    # replays and OVERWRITES movie.mp4 at the same path. Dropping every field here meant a
    # speed-coloured film of the flow was replaced, every single run, by one coloured by which set
    # each dot belongs to -- and the only notice was one line of log. `_colour_of` already says as
    # much at the other end ("use `speed`, which is computed from the recorded velocities"); this
    # end had not been told.
    _cf = str(style.get("color_field") or "")
    if _cf and _cf != "speed":
        print(f"[replay] plotting.color_field: {_cf!r} needs the per-particle C/F tensors, which a "
              f"trajectory does not store -- colouring by type instead. Use the live renderer, or "
              f"`speed`, which a replay reconstructs from the recorded positions.", flush=True)
        lm.style.pop("color_field", None)
    # THE TIME BETWEEN RECORDED FRAMES IS A FACT ABOUT THE TRAJECTORY, set on every level whether or
    # not anything is coloured by speed. It was set only under `color_field: speed`, so a replay
    # without that key served velocities as bare per-frame displacements and the `rate:` curve
    # read 1/dt too small -- 0.16 "Hz" for a rotor turning at 79 Hz (exp_02 step 0073, dt 0.002).
    _st = max(1.0, float(getattr(sim, "n_frames", T - 1) or (T - 1)) / max(T - 1, 1))
    for _lv in H.levels.values():
        _lv._rec_dt = float(getattr(sim, "dt", 1.0) or 1.0) * _st
    if _cf:
        print(f"[replay] plotting.color_field: 'speed' reconstructed from the recorded positions "
              f"over {_st:g} tick(s) a frame -- the movie keeps its field.", flush=True)
    for t in range(T):
        H.seek(t)
        lm(H, t)
    return lm.close()
