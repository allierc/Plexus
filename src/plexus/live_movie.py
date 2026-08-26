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

FLAT = dict(render_points_as_spheres=True, lighting=False, ambient=1.0, diffuse=0.0, specular=0.0)


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
        self.px_used = None
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
            fps = n_rendered / (self.duration_s * _sm)
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
                self.p.add_point_labels([_mid], [_lab], font_size=11, text_color="white",
                                        shape=None, show_points=False, always_visible=True)
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
        self.p.open_movie(out, framerate=max(1, int(round(getattr(self, "fps", fps)))), quality=8)

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
            self.p.add_mesh(m, color="#9a9a9a", opacity=1.0, lighting=not self.is2d,
                            specular=0.2, smooth_shading=True)
        self.n_obstacles = len(obs)

    def _xyz(self, lvl):
        # float32, NOT float64. VTK stores points in whatever dtype it is handed; float64 doubles
        # both the host copy and VTK's resident buffer (10 M points: 240 MB against 120 MB) to carry
        # digits that never survive the projection to a 1280 px frame.
        pos = lvl.get("pos")[self.idx].detach().cpu().numpy().astype(np.float32)
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
            self.cloud["rgb"] = self._rgb(H, lvl, pos)
            self.p.add_mesh(self.cloud, scalars="rgb", rgb=True, **FLAT,
                            point_size=self._dot_px(pos))
            self.t0 = time.perf_counter()
            return
        if tick % self.stride:
            return
        self.cloud.points = self._xyz(lvl)
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
        self.p.add_text(f"{self.name}{self._box_label}\n{self.n:,} particles{sub}\n"
                        f"frame {tick}/{self.n_frames}   "
                        f"{el / max(tick, 1) * 1000:.0f} ms/frame compute{clk}",
                        position="upper_left", font_size=11, color="white", name="hdr")
        self.p.write_frame()
        self.rendered += 1
        if tick in self.still_ticks:
            self._still(tick)

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
        if self.dot != "auto":
            self.px_used = float(self.dot)
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
            self.px_used = 1.5
        else:
            # world -> px through the parallel projection: the camera frames `parallel_scale`
            # half-heights over the window's half-height.
            world_per_px = (2.0 * self.p.camera.parallel_scale) / max(self.p.window_size[1], 1)
            self.px_used = float(np.clip(self.fill * sp / max(world_per_px, 1e-12), 0.7, 24.0))
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
