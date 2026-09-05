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

import os
import time

import numpy as np
import torch

FLAT = dict(render_points_as_spheres=True, lighting=False, ambient=1.0, diffuse=0.0, specular=0.0)
_CS_AXIS = {"x": 0, "y": 1, "z": 2}
# EVERY `plotting.color_field` THE RENDERER KNOWS. Named in one place so the check that
# refuses an unknown one can quote the list, the way `plotting.renderer` already does.
_FIELDS = ("vorticity", "speed", "pressure", "deformation", "strain", "volume")


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
        if k <= bn:
            continue
        best, bn = name, k
    return best


class LiveMovie:
    """An `on_frame(H, tick)` hook that writes one mp4 for the whole run.

    Build it before `engine.run`, pass `__call__` as `on_frame`, and call `close()` afterwards.
    Every failure is swallowed and reported: a renderer that kills a twelve-minute simulation on the
    last frame is worse than no renderer.
    """

    def __init__(self, out, world, n_frames, up=2, render_n=400_000, max_frames=300,
                 fps=20, px=1280, dot=None, fill=0.9, elev=18.0, azim=-58.0, name="", seed=0,
                 sim=None, style=None, stills=10, keep_stills=False,
                 dt=None, time_s=None, real_time=True, length_um=None, centred=False, can_curve=False):
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

        self.pv = pv
        self.sim, self.style = sim, dict(style or {})
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
        self._rate_of = "compute"       # `replay` sets "render": see below
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
        self.length_um = length_um
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
            _cells = _ng ** _d
            _c = (f"{_cells / 1e6:.1f}M" if _cells >= 1e6 else f"{_cells / 1e3:.0f}k")
            self._grid_label = (f"   grid {_ng}^{_d} = {_c} cells")
            if length_um:                       # with units, the cell has a SIZE worth quoting
                _dx = float(world[1] if len(world) > 1 else world[0]) / _ng \
                    * float(length_um) / 1.0e6
                self._grid_label += (f", dx {_dx * 1e3:.3g} mm" if _dx < 1.0
                                     else f", dx {_dx:.3g} m")
        self._box_label = ""
        if length_um and time_s is not None:
            _m = float(length_um) / 1.0e6
            _w = [float(x) * _m for x in world]
            # THREE SIGNIFICANT FIGURES. `:g` on a measured box printed "1.17233 mm cube" -- six
            # digits of a number the reader is being given for scale, where the point is the order
            # of magnitude and the leading figure. (The scale BAR is a chosen round number and is
            # exact; this is a measurement and is rounded.)
            _f = (lambda v: f"{v * 1e3:.3g} mm" if v < 0.01 else f"{v * 100:.3g} cm"
                  if v < 1.0 else f"{v:.3g} m" if v < 1000.0 else f"{v / 1000:.3g} km")
            self._box_label = ("   box " + " x ".join(_f(v) for v in _w)
                               if len(set(_w)) > 1 else f"   box {_f(_w[0])} cube")
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
        _cv = (style or {}).get("curve") if can_curve else None
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
        self.p.set_background("black")
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
            if abs(self.zoom - 1.0) < 1e-9:
                self.p.add_mesh(pv.Box((_b[0][0], _b[1][0], _b[0][1], _b[1][1],
                                        _b[0][2], _b[1][2])).extract_all_edges(),
                                color="#4a4a4a", line_width=1.0, lighting=False)
            # A SCALE BAR, AND ONLY WHERE THERE IS A SCALE. Without `general.units` the box is
            # a number of nothing and a bar labelled "20" would be a lie. The length is the largest
            # round number (1, 2 or 5 times a power of ten) fitting in a third of the box, so it
            # sizes itself: 20 m for a 100 m box, 2 cm for a 0.1 m one, with nothing to set.
            #
            # A PLAIN SEGMENT: no end ticks, and the label in the same font and size as the
            # top-left print, so it reads as one annotation rather than two competing ones.
            if self.time_s is not None and getattr(self, "length_um", None):
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
                _len = _len_m / _m                            # back to box units for the geometry
                _other = [i for i in range(3) if i not in (self.up, _ax0)][0]
                _a = np.zeros(3); _b = np.zeros(3)
                # PLACED AGAINST THE SCENE'S OWN CORNER for the same reason the camera is: with a
                # free boundary the box's origin is in the middle of the tissue, and the bar was
                # drawn straight through it.
                _a[_ax0] = float(self.lo[_ax0]); _b[_ax0] = float(self.lo[_ax0]) + _len
                _a[_other] = _b[_other] = float(self.lo[_other]) - 0.04 * float(span[_other])
                _a[self.up] = _b[self.up] = float(self.lo[self.up])
                self.p.add_mesh(pv.Line(_a, _b), color="white", line_width=4.0, lighting=False)
                _v = _len_m
                _lab = (f"{_v * 1e6:g} um" if _v < 1e-4 else f"{_v * 1e3:g} mm" if _v < 0.01
                        else f"{_v * 100:g} cm" if _v < 1.0
                        else f"{_v:g} m" if _v < 1000.0 else f"{_v / 1000:g} km")
                _mid = 0.5 * (_a + _b); _mid[self.up] -= 0.05 * float(span[self.up])  # noqa
                # TWICE THE HEADER'S NUMBER TO GET THE SAME HEIGHT. `add_text` and
                # `add_point_labels` do not interpret `font_size` the same way -- both set to 11 and
                # the label renders about half the cap height of the top-left print. 22 matches it,
                # and at that size the label spans roughly two thirds of the bar, which is what
                # makes the two read as one annotation.
                self.p.add_point_labels([_mid], [_lab], font_size=22, text_color="white",
                                        shape=None, show_points=False, always_visible=True,
                                        justification_horizontal="center")
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
            u = np.zeros(3); u[self.up] = 1.0
            self.p.camera.up = tuple(u)
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
        self.p.open_movie(out, framerate=max(1, int(round(getattr(self, "fps", fps)))), quality=8,
                          output_params=["-movflags", "frag_keyframe+empty_moov+default_base_moof",
                                         "-g", "1", "-flush_packets", "1"])

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
                self.p.enable_shadows()
            except Exception as e:                      # not fatal: the movie is still readable
                print(f"[live-movie] shadows unavailable ({type(e).__name__}: {e}); "
                      f"obstacles are flat-shaded without them", flush=True)

    def _xyz(self, lvl):
        # float32, NOT float64. VTK stores points in whatever dtype it is handed; float64 doubles
        # both the host copy and VTK's resident buffer (10 M points: 240 MB against 120 MB) to carry
        # digits that never survive the projection to a 1280 px frame.
        _p = lvl.get("pos")[self.idx].detach()
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
        return pos

    def __call__(self, H, tick):
        if self.failed:
            return
        try:
            self._frame(H, tick)
        except Exception as e:                        # never take the run down for a picture
            self.failed = f"{type(e).__name__}: {e}"
            print(f"[live-movie] DISABLED after frame {tick}: {self.failed}", flush=True)

    def _frame(self, H, tick):
        import torch
        sname = _biggest_particle_set(H)
        if sname is None:
            self.failed = "no set carries positions"
            return
        self._sname = sname                           # `_rgb` needs it to look up the type palette
        lvl = H.level(sname)
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
            self.drawn = k
            pos = self._xyz(lvl)
            self.cloud = self.pv.PolyData(pos)
            # RESOLVED BEFORE IT IS ASSIGNED. Writing `cloud["rgb"] = None` and testing afterwards
            # cannot work: pyvista raises "Empty array unable to be added" ON THE ASSIGNMENT, so the
            # fallback line was unreachable and every spec WITHOUT `color_field` lost its movie --
            # silently, because LiveMovie swallows its own errors to avoid killing a long run. Three
            # A100 jobs rendered nothing before this was noticed.
            _rgbv = self._rgb_field(H, lvl)
            self.cloud["rgb"] = _rgbv if _rgbv is not None else self._rgb(H, lvl, pos)
            # POINT SPRITES AND THE CHART OVERLAY DO NOT COEXIST. `add_chart` inserts a
            # vtkContextActor, and with one present `render_points_as_spheres=True` draws nothing at
            # all: measured on this scene, 3,701 blue pixels with the panel and 61,331 without, from
            # an identical simulation whose colour array was uniformly [76, 158, 255] either way.
            # Plain GL points render correctly alongside the chart and are visually identical at the
            # 1-2 px these dots are drawn at, so the panel costs the sprite, not the picture.
            _flat = dict(FLAT)
            if self.cs is not None:
                _flat["render_points_as_spheres"] = False
            # WHEN THE SUBJECT IS A MESH, THE DOTS ARE ITS OWN VERTICES and drawing them is drawing
            # the corners of the thing rather than the thing. One renderer covers both cases: a
            # material run draws its cloud (or a skinned surface of it) with any mesh set over the
            # top, and a mesh-only run draws the mesh. Nothing about the spec has to say which.
            _m = getattr(lvl, "mesh", None)
            self._mesh_is_subject = bool(_m is not None and int(_m.get("nF", 0) or 0))
            if self._mesh_is_subject:
                pass
            elif str((self.style or {}).get("render_3d", "dots")).lower() != "surface" \
                    or not self._skin_build(H, lvl, pos):
                self.p.add_mesh(self.cloud, scalars="rgb", rgb=True, **_flat,
                                point_size=self._dot_px(pos))
            self._add_meshes(H)
            for _n, _l, _m in self._mesh_levels(H):
                self._edge_actor(H, _l, _m, first=True)
                break
            self._curves_setup(H, lvl)
            self.t0 = time.perf_counter()
            return
        if tick % self.stride:
            return
        self.cloud.points = self._xyz(lvl)
        self._skin_update(H, lvl, self.cloud.points)
        self._update_meshes(H)
        self._curves_update(tick)
        # A FIELD COLOUR IS A PROPERTY OF NOW, so unlike the body hue it is recomputed each frame.
        if str(self.style.get("color_field", "") or ""):
            _c = self._rgb_field(H, lvl)
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
        self.p.add_text(f"{self.name}{self._box_label}\n"
                        f"{self.n:,} particles{sub}{self._grid_label}\n"
                        f"frame {tick}/{self.n_frames}   "
                        f"{el / max(tick, 1) * 1000:.0f} ms/frame {self._rate_of}{clk}{_lut}",
                        position="upper_left", font_size=11, color="white", name="hdr")
        if self.cs is not None:
            self._update_cross_section(H)
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
    _CURVE_Q = ("cells", "area", "radius", "myosin")

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
        out = np.full((T, nt, 2), np.nan)
        for t in range(T):
            lvl.t = t
            m = getattr(lvl, "mesh", None)
            if m is None or not int(m.get("nF", 0) or 0):
                continue
            nF = int(m["nF"])
            k = (np.zeros(nF, int) if ntype is None
                 else np.asarray(ntype)[np.clip(np.arange(nF), 0, len(ntype) - 1)].astype(int))
            if q == "cells":
                for j in range(nt):
                    out[t, j] = (float((k == j).sum()), 0.0)
                continue
            if q == "myosin":
                v = m.get("e_myo")
                if v is None:
                    continue
                ef = np.asarray(m["E_face"]); live = ef < nF
                vv = np.asarray(v, float)[live]
                ke = k[np.clip(ef[live].astype(int), 0, nF - 1)]
                for j in range(nt):
                    sel = ke == j
                    if sel.any():
                        out[t, j] = (float(np.nanmean(vv[sel])), float(np.nanstd(vv[sel])))
                continue
            import torch
            if q == "radius":
                nv = int(m["Nv"])
                P = np.asarray(lvl.get("pos")[:nv])
                r = np.linalg.norm(P - P.mean(0), axis=1)
                for j in range(nt):                   # per VERTEX, so the type split is by face
                    out[t, j] = (float(np.mean(r)), float(np.std(r)))
                continue
            from plexus.operators.vertex_ops import face_geometry_3d
            nv = int(m["Nv"])
            pos = torch.as_tensor(lvl.get("pos")[:nv], dtype=torch.float64)
            a, _p, _c, _v = face_geometry_3d(pos, m["E_srce"], m["E_trgt"], m["E_face"], nF)
            a = a.numpy()
            for j in range(nt):
                sel = k == j
                if sel.any():
                    out[t, j] = (float(np.nanmean(a[sel])), float(np.nanstd(a[sel])))
        lvl.t = 0
        return out

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
        lq = lvl
        for _nm, _lv in H.levels.items():
            _m = getattr(_lv, "mesh", None)
            if _m is not None and int(_m.get("nF", 0) or 0) and hasattr(_lv, "_pos"):
                lq = _lv
                if _nm != getattr(self, "_sname", None):
                    print(f"[live-movie] curves read set {_nm!r} (it carries the mesh), not "
                          f"{getattr(self, '_sname', '?')!r}", flush=True)
                break
        lvl = lq
        ntype = self._curve_types(H, lvl)
        if not hasattr(lvl, "_pos"):
            print("[live-movie] plotting.curve needs the whole clip to fix its axes and a live "
                  "generate has only the current frame -- re-render with `-o plot`", flush=True)
            return
        for _i, cfg in enumerate(cfgs):
            cfg = dict(cfg)
            q = str(cfg.get("quantity", "cells")).lower()
            if q not in self._CURVE_Q:
                raise ValueError(f"plotting.curve.quantity: {q!r} is not one of "
                                 f"{', '.join(self._CURVE_Q)}")
            S = self._curve_series(H, lvl, q, ntype)
            if not np.isfinite(S[..., 0]).any():
                continue
            lo = float(np.nanmin(S[..., 0] - S[..., 1])); hi = float(np.nanmax(S[..., 0] + S[..., 1]))
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
            lo = 0.0 if (lo >= 0 or q in ("cells", "area", "myosin")) else _r1(lo, False)
            # A ROUND STEP, NOT A ROUND TOP. Snapping only the top to one significant figure still
            # left the ticks between the ends to be whatever the count divided into: 0..8000 over
            # `ticks: 4` printed 0, 2667, 5333, 8000, and the two in the middle are the artefact of
            # a division, not numbers anyone chose. The STEP is snapped instead -- to the largest of
            # {1, 2, 2.5, 5} x 10^k that is no bigger than range/(ticks-1) -- and the top is then the
            # first multiple of it above the data. Every tick is round by construction, and the tick
            # COUNT follows from the step rather than forcing it.
            _raw = max(hi * 1.02 - lo, 1e-12) / max(int(cfg.get("ticks", 4)) - 1, 1)
            _e = 10.0 ** np.floor(np.log10(_raw))
            _step = max([f * _e for f in (1.0, 2.0, 2.5, 5.0) if f * _e <= _raw] or [_e])
            _n = int(np.ceil((hi * 1.02 - lo) / _step))
            hi = lo + _step * max(_n, 1)
            cfg["ticks"] = max(_n, 1) + 1
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
            ch.x_axis.range = [0.0, float(S.shape[0] - 1)]
            ch.y_axis.range = [float(cfg.get("ymin", lo - pad)), float(cfg.get("ymax", hi + pad))]
            ch.x_axis.label = str(cfg.get("xlabel", "frame"))
            ch.y_axis.label = str(cfg.get("ylabel", q))
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
                                if _a is ch.y_axis else (0.0, float(len(S) - 1)))
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
                    _a.pen.color = "white"
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
            cols = ([tuple(cfg["color"])] if "color" in cfg and nt == 1
                    else [(1.0, 1.0, 1.0)] if nt == 1 else [_pal[j % len(_pal)] for j in range(nt)])
            bands, lines = [], []
            for j in range(nt):
                c = cols[j]
                bands.append(ch.area([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], color=(*c, 0.28)))
                lines.append(ch.line([0.0, 0.0], [0.0, 0.0], color=(*c, 1.0), width=2.0))
            self.p.add_chart(ch)
            self._curves.append({"S": S, "bands": bands, "lines": lines, "nt": nt,
                                 "sd": bool(cfg.get("sd", q != "cells"))})
            print(f"[live-movie] curve {q}: {S.shape[0]} frames, {nt} "
                  f"{'series by type' if nt > 1 else 'series'}, "
                  f"y [{ch.y_axis.range[0]:.4g}, {ch.y_axis.range[1]:.4g}]", flush=True)

    def _curves_update(self, tick):
        """Reveal each series up to the current RECORDED row -- the band is mean-SD .. mean+SD."""
        for cv in getattr(self, "_curves", []) or []:
            S = cv["S"]
            t = min(int(tick), S.shape[0] - 1)
            if t < 1:
                continue
            x = np.arange(t + 1, dtype=float)
            for j in range(cv["nt"]):
                mu, sd = S[: t + 1, j, 0], S[: t + 1, j, 1]
                ok = np.isfinite(mu)
                if ok.sum() < 2:
                    continue
                if cv["sd"]:
                    cv["bands"][j].update(x[ok], (mu - sd)[ok], (mu + sd)[ok])
                cv["lines"][j].update(x[ok], mu[ok])

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

        def deform(self, X):
            return np.einsum("vk,vkj->vj", self.w, np.asarray(X, float)[self.idx])

        def __call__(self, X):
            return self.deform(X) + self.offset

        def scalar(self, values):
            """Per-vertex interpolation of a per-particle scalar, same inverse-square weights."""
            return np.einsum("vk,vk->v", self.w, np.asarray(values, float)[self.idx])

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
            lo = X.min(0) - 3.0 * h
            dim = np.maximum(np.ceil((X.max(0) + 3.0 * h - lo) / h).astype(int) + 1, 2)
            ijk = np.clip(((X - lo) / h).astype(np.int64), 0, dim - 1)
            D = np.zeros(tuple(dim), np.float32)
            np.add.at(D, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
            D = gaussian_filter(D, sigma=float(st.get("surface_blur", 1.0)))
            occ = D[D > 0]
            iso = float(st.get("surface_iso", 0.0)) or 0.5 * float(np.median(occ))
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
            print(f"[live-movie] surface: density isosurface of {X.shape[0]:,} points at cell "
                  f"{h:.4g} ({'x'.join(str(int(v)) for v in dim)}), iso {iso:.3g} of a bulk "
                  f"{float(np.median(occ)):.3g} -> {surf.n_points:,} vertices, "
                  f"{surf.n_faces_strict:,} faces, skinned to {self._skin.idx.shape[1]} "
                  f"particles each over a {len(sub):,}-point bind set", flush=True)
            return True
        except Exception as e:                       # noqa: BLE001 -- fall back to dots, never die
            self._skin = self._surf = None
            print(f"[live-movie] surface reconstruction failed ({type(e).__name__}: {e}); "
                  f"drawing the point cloud instead", flush=True)
            return False

    def _skin_update(self, H, lvl, pos):
        if self._skin is None or self._surf is None:
            return
        X = np.asarray(pos, dtype=np.float64)[self._skin_sub]
        self._surf.points = self._skin(X).astype(np.float32)
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
        P = lvl.get("pos")[:nv].detach().cpu().numpy().astype(np.float32)
        return P if (sc == 1.0 or ct is None) else (P - P.mean(0)) * sc + ct

    def _mesh_levels(self, H):
        """Every Level carrying a non-empty half-edge table, minus `plotting.hide_sets`."""
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
            out.append((name, lvl, m))
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
                _rgb = self._mesh_face_rgb(m, pd) if self._mesh_is_subject else None
                self.p.add_mesh(pd, color=(None if _rgb is not None else colr),
                                scalars=("rgb" if _rgb is not None else None), rgb=(_rgb is not None),
                                style=style, line_width=lw, opacity=opac,
                                lighting=(style != "wireframe" and not _flat_m),
                                ambient=(1.0 if _flat_m else 0.3),
                                diffuse=(0.0 if _flat_m else 0.7), specular=0.0,
                                render_lines_as_tubes=False,
                                show_edges=_edges, edge_color=st.get("mesh_edge_color", "#2b2b2b"),
                                edge_opacity=float(st.get("mesh_edge_opacity", 1.0)))
                self._meshes.append((name, nv, pd, sc, ct))
                print(f"[live-movie] surface {name!r}: {int(m['nF']):,} faces, {nv:,} vertices, "
                      f"drawn as {style}", flush=True)
        except Exception as e:                       # noqa: BLE001 -- never kill a run for a picture
            self._meshes = []
            print(f"[live-movie] mesh overlay unavailable ({type(e).__name__}: {e})", flush=True)

    def _mesh_face_rgb(self, m, pd):
        """One colour per cell: the base, with the division pair marked -- `render_vtk`'s own rule.

        REUSED, NOT REIMPLEMENTED. `render_vtk._marks` is where the mother/daughter split lives, and
        it is subtle enough to be worth importing rather than restating: the two masks are on
        DIFFERENT CLOCKS (`age <= DIVIDED` counts division CALLS, "appended since" counts rows), so
        a naive "compare with the previous frame" draws mothers on every frame and daughters on
        almost none. It takes a face count from far enough back to cover the same window.
        """
        if not bool((self.style or {}).get("mesh_mark_division", True)):
            return None
        try:
            from plexus.render_vtk import _marks
            import matplotlib.colors as _mc
            nF = int(m["nF"])
            self._nF_hist = (getattr(self, "_nF_hist", []) + [nF])[-8:]
            prev = self._nF_hist[0] if len(self._nF_hist) > 1 else None
            # NUMPY, BECAUSE `_marks` IS NUMPY. On the REPLAY path the columns come out of an npz
            # and already are; live from the engine they are CUDA tensors, so the call raised
            # "can't convert cuda:0 device type tensor to numpy" -- and the guard below turned that
            # into a one-line warning and a uniformly grey tissue. The two paths were drawing
            # different pictures again, which is the thing this renderer was unified to stop.
            mt = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
                  for k, v in m.items() if k in ("age", "ndiv", "apop", "inhib")}
            mt["nF"] = nF
            mother, daughter, kills, _sup = _marks(mt, np.arange(nF), nF, prev_nF=prev)
            st = self.style or {}
            rgb = np.tile((np.asarray(_mc.to_rgb(st.get("mesh_color", "#e6dcc0"))) * 255)
                          .astype(np.uint8), (nF, 1))
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
                    self._mesh_face_rgb(m, pd)
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
            if _h is None:
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
                _, _, nmid, _hv0 = monolayer_shells(_t.as_tensor(P, dtype=_t.float32),
                                                    _as(es), _as(et), _as(ef), nF,
                                                    _t.ones(nF, dtype=_t.float32))
                nmid = nmid.numpy().astype(np.float64)
                L = np.linalg.norm(_sep, axis=1)
                ok = L > 1e-12
                n = np.where(ok[:, None], _sep / np.maximum(L, 1e-12)[:, None], nmid)
                hv = 2.0 * L * float(sc)
                return (P, n, hv, es.astype(np.int64), et.astype(np.int64))

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
            y0 = self.cs_at * float(self.world[ax])
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
            fld = str(self.style.get("color_field", "") or "")
            val = self._field(H, lvl)[0] if fld else None
            # `cross_section.points: false` -- DO NOT SCATTER THE DRAWN SET'S PARTICLES IN THE
            # SECTION. On a MESH run those particles are the mesh's own vertices, which on this
            # promotion are the MID-SURFACE -- so a section that already draws apical, basal and
            # the walls gets the mid-surface back a second time, as a dotted line down the middle
            # of the band, after `cross_section.mid: false` removed the curve. Cedric found it in
            # exactly that order: first the parallel white ring, then the blue dashes behind it.
            #
            # It is a key rather than a rule because on a PARTICLE run the scatter is the section:
            # an MPM slab has nothing else in it, and every existing cross_section spec is one of
            # those. Default true, so none of them moves.
            if (self.style or {}).get("cross_section", {}).get("points", True) is False:
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
                    # is what Cedric queried on sight.
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
                                # what this promotion added. Cedric read the picture and asked
                                # whether the parallel white ring was a bug -- it is not, it is the
                                # mid-surface, and the colouring is what made that a question.
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
        if want not in _FIELDS:
            raise ValueError(f"plotting.color_field: {want!r} is not one of "
                             f"{', '.join(sorted(_FIELDS))}")
        import torch
        idx = self.idx
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
            v = lvl.get("vel")[idx]
            return v.norm(dim=1), "|v| (m/s)"
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
        else:                                                # settled ONCE, on the first frame
            if getattr(self, "_frng", None) is None:
                q = torch.quantile(val.float()[:: max(1, val.numel() // 200_000)],
                                   torch.tensor([0.02, 0.98], device=val.device))
                self._frng = (float(q[0]), float(q[1]))
            lo, hi = self._frng
        # LOG BY DEFAULT FOR VORTICITY, because the quantity spans decades and a linear ramp shows
        # one of them. Measured on si_ball_wake, water |curl v|: median 0.6 -> 163 1/s over the run
        # and 0.6 -> 1305 within the last frame. On a linear [0, 2000] the median sits at 8% of the
        # map, so the column reads as one flat colour and the wake -- which IS there -- is the same
        # dark purple as the still fluid. log10 puts three decades across the ramp instead of one.
        if bool(self.style.get("field_log", want == "vorticity")):
            eps = max(lo, hi * 1e-4)
            t = ((torch.log10(val.clamp(min=eps)) - math.log10(eps))
                 / max(math.log10(hi) - math.log10(eps), 1e-12)).clamp(0, 1).detach().cpu().numpy()
            self._lut = f"log [{eps:.3g}, {hi:.3g}]"
        else:
            t = ((val - lo) / max(hi - lo, 1e-12)).clamp(0, 1).detach().cpu().numpy()
            self._lut = f"[{lo:.3g}, {hi:.3g}]"
        cm = plt.get_cmap(self.style.get("field_cmap", "turbo"))
        self.colour_by = f"{label} {self._lut} ({self.style.get('field_cmap','turbo')})"
        return (cm(t)[:, :3] * 255).astype(np.uint8)

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
            sp = float(np.median(nn[np.isfinite(nn)]))
        except Exception:
            sp = 0.0
        span = float((self.hi - self.lo).max()) or 1.0
        if sp <= 0:
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
        print(f"[live-movie] {self.out}   {self.n:,} particles{sub}, {self.rendered} frames"
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
class _ReplayLevel:
    """One set of a trajectory.npz, shaped like the Level the renderer reads off the engine."""

    def __init__(self, z, name, dev):
        import torch
        self._pos = torch.as_tensor(np.asarray(z[f"{name}__pos"], np.float32), device=dev)
        _occ = z[f"{name}__occ"] if f"{name}__occ" in z.files else None
        self._occ = None if _occ is None else torch.as_tensor(np.asarray(_occ), device=dev)
        self.n = int(self._pos.shape[1])
        self.state = self._pos[0]                     # only `.state.device` is ever read
        self.t = 0
        for k in ("node_type", "parent"):
            v = z[f"{name}__{k}"] if f"{name}__{k}" in z.files else None
            setattr(self, k, None if v is None else torch.as_tensor(np.asarray(v), device=dev))
        pn = z[f"{name}__parent_name"] if f"{name}__parent_name" in z.files else None
        self.parent_name = None if pn is None else str(pn)
        self.C = self.F = None                        # not stored in a trajectory -- see above
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
            self._face_cols = {k: np.asarray(z[f"{name}__mesh_{k}"])
                               for k in ("age", "ndiv", "apop", "inhib")
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
        if key == "pos":
            return self._pos[self.t]
        if key == "occ" and self._occ is not None:
            return self._occ[self.t]
        return None


class _ReplayState:
    """The `H` the renderer expects: a name -> level mapping and nothing else."""

    def __init__(self, z, dev):
        # EVERY SET'S TYPES, not only the sets that carry positions. A vertex model's `cell` set has
        # `cen`/`area`/`node_type` and NO `pos`, so it never becomes a level here -- and the curve's
        # per-type split, which indexes faces by the cell set's node_type, silently collapsed to one
        # series. The types are in the file either way.
        self.node_types = {k[: -len("__node_type")]: np.asarray(z[k])
                           for k in z.files if k.endswith("__node_type")}
        names = sorted({k[: -len("__pos")] for k in z.files if k.endswith("__pos")})
        self.levels = {n: _ReplayLevel(z, n, dev) for n in names}
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
    H = _ReplayState(z, dev)
    if not H.levels:
        raise ValueError(f"{data_dir}: no set carries positions, nothing to render")
    sname = _biggest_particle_set(H)
    lvl = H.levels[sname]
    P = lvl._pos                                       # [T, N, D]
    T, D = int(P.shape[0]), int(P.shape[2])

    # --- the box, and the shift that puts the cloud inside it ---
    ws = np.asarray(z["world_size"], np.float64) if "world_size" in z.files else None
    if ws is None or len(ws) != D:
        w = float(z["world"]) if "world" in z.files else float(getattr(sim, "world", 1.0))
        ws = np.full(D, w, np.float64)
    flat = P.reshape(-1, D).numpy()
    fp = style.get("frame_percentile")
    if fp is not None:
        q = 0.5 * (100.0 - float(fp))
        lo = np.percentile(flat, q, axis=0)
        hi = np.percentile(flat, 100.0 - q, axis=0)
        box = (hi - lo) * 1.06
        lvl._pos = P - torch.as_tensor((0.5 * (lo + hi) - 0.5 * box), dtype=P.dtype)
    else:
        lo, hi = flat.min(0), flat.max(0)
        # A CENTRED RUN STILL HAS A DECLARED BOX, and falling back to the data bounds threw it away.
        # The test was "does the content sit in [0, world]" -- true for a walled run and false for
        # every `boundary: free` one, whose content is about the ORIGIN -- so a free run was always
        # framed on its own extent. Two runs of one model at different sizes then filled the frame
        # identically and the size difference, which is the entire experiment, was invisible. If the
        # content fits in [-world/2, +world/2] the declared box is used and the cloud is shifted into
        # it; only content that fits NEITHER convention falls back to its own bounds.
        if not bool(((hi - lo) > ws * 1.02).any()) and not bool((lo < -1e-6 * ws.max()).any()):
            box = ws                                            # already in [0, world]
        elif not bool(((hi - lo) > ws * 1.02).any()):
            box = ws                                            # fits [-world/2, world/2]
            lvl._pos = P + torch.as_tensor(0.5 * box, dtype=P.dtype)
        else:
            box = (hi - lo) * 1.06
            lvl._pos = P - torch.as_tensor((0.5 * (lo + hi) - 0.5 * box), dtype=P.dtype)
    # `movie.mp4`, NOT `movie_<set>.mp4`. Every other path in this codebase writes `movie.mp4`, so a
    # re-render under `-o plot` left the folder holding both and neither obviously the current one.
    out = out or os.path.join(data_dir, "movie.mp4")
    # UNITS ONLY WHEN THEY WERE DECLARED. `Units` defaults to length_um 1.0 / time_s 1.0 with
    # `declared: False`, and handing those to the renderer would put a scale bar and a wall clock on
    # a run that has neither -- a bar reading "2 m" across a galaxy 12 dimensionless units wide.
    u = getattr(sim, "units", None)
    dec = bool(getattr(u, "declared", False))
    lm = LiveMovie(out=out, world=list(np.asarray(box, np.float64)), n_frames=T,
                   can_curve=True,      # a replay holds every frame; the live path does not
                   up=int(style.get("up_axis", 2)), render_n=render_n, max_frames=max_frames,
                   name=name or getattr(sim, "name", ""), sim=sim, style=style,
                   stills=stills, keep_stills=keep_stills,
                   dt=getattr(sim, "dt", None),
                   time_s=(float(u.time_s) if dec else None),
                   length_um=(float(u.length_um) if dec else None),
                   **({"fps": float(fps)} if fps else {}))
    lm._rate_of = "render"
    if style.get("color_field"):
        print("[replay] plotting.color_field needs the per-particle C/F tensors, which a trajectory "
              "does not store -- colouring by type instead. Use the live renderer for a field.",
              flush=True)
        lm.style.pop("color_field", None)
    for t in range(T):
        H.seek(t)
        lm(H, t)
    return lm.close()
