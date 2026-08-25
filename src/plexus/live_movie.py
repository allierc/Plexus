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
                 fps=20, px=1280, dot="auto", fill=0.9, elev=18.0, azim=-58.0, name="", seed=0,
                 sim=None, style=None):
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
        self.dot, self.fill = dot, float(fill)
        self.px_used = None
        self.up = int(up)
        self.stride = max(1, int(np.ceil(self.n_frames / max(1, int(max_frames)))))
        self.cloud = self.idx = None
        self.drawn = self.n = self.rendered = 0
        self.t0 = None
        self.failed = None
        self.colour_by = "?"

        px = int(px) // 16 * 16                       # ffmpeg's macro_block_size; see cell_panels
        self.p = pv.Plotter(off_screen=True, window_size=(px, px), border=False)
        self.p.set_background("black")
        self.p.enable_anti_aliasing("msaa", multi_samples=8)

        w = [float(x) for x in world]
        while len(w) < 3:                             # a 2D run is drawn in the z=0 plane
            w.append(0.0)
        self.lo, self.hi = np.zeros(3), np.array(w)
        span = np.where(self.hi > self.lo, self.hi, self.lo + 1.0)
        self.p.add_mesh(pv.Box((0, span[0], 0, span[1], 0, max(span[2], 1e-6))
                               ).extract_all_edges(),
                        color="#4a4a4a", line_width=1.0, lighting=False)
        centre, radius = 0.5 * span, float(span.max()) * 0.55
        e, az = np.radians(elev), np.radians(azim)
        ax_h = [i for i in range(3) if i != self.up]
        d = np.zeros(3)
        d[ax_h[0]], d[ax_h[1]], d[self.up] = np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)
        if span[2] <= 1e-6:                           # 2D: look straight down the empty axis
            d = np.array([0.0, 0.0, 1.0])
        self.p.camera.position = tuple(centre + d * radius * 6.0)
        self.p.camera.focal_point = tuple(centre)
        u = np.zeros(3); u[self.up if span[2] > 1e-6 else 1] = 1.0
        self.p.camera.up = tuple(u)
        self.p.camera.parallel_projection = True
        self.p.camera.parallel_scale = radius * 1.45
        self.p.open_movie(out, framerate=int(fps), quality=8)

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
        self.p.add_text(f"{self.name}\n{self.n:,} particles{sub}\n"
                        f"frame {tick}/{self.n_frames}   {el / max(tick, 1) * 1000:.0f} ms/frame",
                        position="upper_left", font_size=11, color="white", name="hdr")
        self.p.write_frame()
        self.rendered += 1

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

    def close(self):
        try:
            self.p.close()
        except Exception:
            pass
        if self.failed or not self.rendered:
            print(f"[live-movie] wrote nothing ({self.failed or 'no frames rendered'})", flush=True)
            return None
        sub = f", {self.drawn:,} of them drawn" if self.drawn < self.n else ""
        print(f"[live-movie] {self.out}   {self.n:,} particles{sub}, {self.rendered} frames"
              f"{'' if self.stride == 1 else f' (every {self.stride}th)'}, "
              f"coloured by {self.colour_by}, dot {self.px_used:.2f} px", flush=True)
        return self.out
