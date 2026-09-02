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
        if pos is None or pos.ndim != 2 or int(lvl.n) <= bn:
            continue
        best, bn = name, int(lvl.n)
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
                 dt=None, time_s=None, real_time=True, length_um=None):
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
        self.dt, self.time_s, self.real_time = dt, time_s, bool(real_time)
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
            _f = (lambda v: f"{v * 1e3:g} mm" if v < 0.01 else f"{v * 100:g} cm"
                  if v < 1.0 else f"{v:g} m" if v < 1000.0 else f"{v / 1000:g} km")
            self._box_label = ("   box " + " x ".join(_f(v) for v in _w)
                               if len(set(_w)) > 1 else f"   box {_f(_w[0])} cube")
        self.speed = None
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
        self.drawn = self.n = self.rendered = 0
        self.t0 = None
        self.failed = None
        self.colour_by = "?"
        self.n_obstacles = 0

        px = int(px) // 16 * 16                       # ffmpeg's macro_block_size; see cell_panels
        self.p = pv.Plotter(off_screen=True, window_size=(px, px), border=False)
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
            self.p.add_mesh(pv.Box((0, span[0], 0, span[1], 0, span[2])).extract_all_edges(),
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
                _tgt = float(span[_ax0]) / 3.0
                _p10 = 10.0 ** np.floor(np.log10(max(_tgt, 1e-30)))
                _len = max([f * _p10 for f in (1.0, 2.0, 5.0) if f * _p10 <= _tgt] or [_p10])
                _other = [i for i in range(3) if i not in (self.up, _ax0)][0]
                _a = np.zeros(3); _b = np.zeros(3)
                _a[_ax0] = 0.0; _b[_ax0] = _len
                _a[_other] = _b[_other] = -0.04 * float(span[_other])
                self.p.add_mesh(pv.Line(_a, _b), color="white", line_width=4.0, lighting=False)
                _v = _len * _m
                _lab = (f"{_v * 1e3:g} mm" if _v < 0.01 else f"{_v * 100:g} cm" if _v < 1.0
                        else f"{_v:g} m" if _v < 1000.0 else f"{_v / 1000:g} km")
                _mid = 0.5 * (_a + _b); _mid[self.up] -= 0.05 * float(span[self.up])
                # TWICE THE HEADER'S NUMBER TO GET THE SAME HEIGHT. `add_text` and
                # `add_point_labels` do not interpret `font_size` the same way -- both set to 11 and
                # the label renders about half the cap height of the top-left print. 22 matches it,
                # and at that size the label spans roughly two thirds of the bar, which is what
                # makes the two read as one annotation.
                self.p.add_point_labels([_mid], [_lab], font_size=22, text_color="white",
                                        shape=None, show_points=False, always_visible=True,
                                        justification_horizontal="center")
            centre, radius = 0.5 * span, float(span.max()) * 0.55
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
            ch = pv.Chart2D(size=(0.26, 0.26), loc=(0.015, 0.645))
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
            self.p.add_mesh(self.cloud, scalars="rgb", rgb=True, **_flat,
                            point_size=self._dot_px(pos))
            self.t0 = time.perf_counter()
            return
        if tick % self.stride:
            return
        self.cloud.points = self._xyz(lvl)
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
        self.p.add_text(f"{self.name}{self._box_label}\n"
                        f"{self.n:,} particles{sub}{self._grid_label}\n"
                        f"frame {tick}/{self.n_frames}   "
                        f"{el / max(tick, 1) * 1000:.0f} ms/frame {self._rate_of}{clk}",
                        position="upper_left", font_size=11, color="white", name="hdr")
        if self.cs is not None:
            self._update_cross_section(H)
        self.p.write_frame()
        self.rendered += 1
        if tick in self.still_ticks:
            self._still(tick)

    def _update_cross_section(self, H):
        """Scatter whatever is inside the slab, in the plane of the other two axes."""
        try:
            lvl = H.level(self.set_name) if getattr(self, "set_name", None) else None
            if lvl is None:
                from plexus.live_movie import _biggest_particle_set
                lvl = H.level(_biggest_particle_set(H))
            X = lvl.get("pos").detach()
            occ = getattr(lvl, "occ", None)
            if occ is not None:
                X = X[occ > 0]
            if X.shape[0] == 0:
                return
            ax, (a, b) = self.cs_axis, self._cs_lat
            dx = float(self.world[ax]) / 96.0
            for fc in getattr(H, "fields", {}).values():
                if hasattr(fc, "dx"):
                    dx = float(fc.dx)
                    break
            y0 = self.cs_at * float(self.world[ax])
            sel = (X[:, ax] - y0).abs() < self.cs_cells * dx
            P = X[sel]
            if P.shape[0] > self.cs_max:            # a slab of a big jet is tens of thousands
                step = P.shape[0] // self.cs_max + 1
                P = P[::step]
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
            try:
                self.cs.remove_plot(self._cs_series)
            except Exception:                            # noqa: BLE001
                pass
            self._cs_series = self.cs.scatter(xs, ys, size=self._cs_size, style="o",
                                              color=self._cs_colour)
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
        if want not in ("vorticity", "speed", "pressure"):
            return None, ""
        import torch
        idx = self.idx
        if want == "speed":
            v = lvl.get("vel")[idx]
            return v.norm(dim=1), "|v| (m/s)"
        if want == "vorticity":
            C = lvl.C[idx]                                   # [N,3,3] velocity gradient
            w = torch.stack([C[:, 2, 1] - C[:, 1, 2],
                             C[:, 0, 2] - C[:, 2, 0],
                             C[:, 1, 0] - C[:, 0, 1]], 1)    # curl(v) = 2 * antisym(C)
            return w.norm(dim=1), "|curl v| (1/s)"
        J = torch.linalg.det(lvl.F[idx].float())             # volume ratio
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
              f"coloured by {self.colour_by}, dot {self.px_used:.2f} px"
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

    def get(self, key):
        if key == "pos":
            return self._pos[self.t]
        if key == "occ" and self._occ is not None:
            return self._occ[self.t]
        return None


class _ReplayState:
    """The `H` the renderer expects: a name -> level mapping and nothing else."""

    def __init__(self, z, dev):
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
        if bool(((hi - lo) > ws * 1.02).any()) or bool((lo < -1e-6 * ws.max()).any()):
            box = (hi - lo) * 1.06
            lvl._pos = P - torch.as_tensor((0.5 * (lo + hi) - 0.5 * box), dtype=P.dtype)
        else:
            box = ws
    out = out or os.path.join(data_dir, f"movie_{sname}.mp4")
    # UNITS ONLY WHEN THEY WERE DECLARED. `Units` defaults to length_um 1.0 / time_s 1.0 with
    # `declared: False`, and handing those to the renderer would put a scale bar and a wall clock on
    # a run that has neither -- a bar reading "2 m" across a galaxy 12 dimensionless units wide.
    u = getattr(sim, "units", None)
    dec = bool(getattr(u, "declared", False))
    lm = LiveMovie(out=out, world=list(np.asarray(box, np.float64)), n_frames=T,
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
