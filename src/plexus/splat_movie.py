"""`plotting.renderer: gpu_splat` -- a live movie that can actually draw 100 M particles.

    plotting:
      renderer: gpu_splat
      splat_res: 1280

WHY THE EXISTING RENDERERS CANNOT. There are two, and each fails at a different scale for a
different reason:

  * `LiveMovie` (pyvista/VTK) copies positions to the HOST every rendered frame and hands VTK a
    point cloud. At 100 M that is a 1.2 GB PCIe transfer per frame plus 100 M GL points, which is
    why `render_n` defaults to 400,000 -- the simulation runs everything and the picture shows
    0.4% of it.
  * `plot.gaussian_splat_3d_tight` is sharper, and it composites with a PYTHON LOOP over
    depth-sorted points -- `for idx in np.argsort(-depth)`. That is 100 M python iterations per
    frame. Its sibling `_splat_accumulate` uses `np.add.at`, whose own docstring says "fast enough
    for 8k points".

WHAT THIS DOES INSTEAD. Everything stays on the GPU, where the particles already are, and every
step is O(N) vectorised:

  1. rotate the centred world points into the camera frame (the same yaw-then-pitch as
     `plot._rot_camera`, so the viewpoint matches the offline renderer),
  2. one `scatter_reduce(amin)` pass for a DEPTH BUFFER, so near material occludes far,
  3. one weighted `bincount` per channel for colour and one for weight, over the points that
     survive the depth slab,
  4. a separable gaussian blur on the ACCUMULATOR (res^2, not N), and the same
     `1 - exp(-w/ref)` opacity curve `plot._composite_splat` uses.

No per-point sprite: at 100 M points into 1280^2 the average pixel receives ~61 points, so a sprite
would be stamping detail finer than the sampling. A single-pixel splat plus a 2 px blur is both
faster and a better estimator of the same density field.

CHUNKED, because the simulation owns most of the card. The per-point temporaries are the flat pixel
index (int64, 8 B) and two weight vectors; at 100 M that is 1.6 GB in one shot and 130 MB at the
8 M default chunk. The result is identical -- `bincount` accumulates.

DEPTH SLAB, NOT A SORT. `zbuf` holds the nearest depth per pixel; a point contributes only if it
lies within `splat_slab` of that pixel's front surface. That is what makes a dense fluid look
solid instead of like a cloud of everything at once, and it costs one extra pass rather than an
N log N sort.
"""
from __future__ import annotations

import os

import numpy as np


class SplatMovie:
    """An `on_frame(H, tick)` hook that writes one mp4, rendering every particle on the GPU."""

    def __init__(self, out, world, n_frames, up=2, render_n=0, max_frames=300, fps=20,
                 res=1280, name="", sim=None, style=None, dt=None, time_s=None,
                 chunk=8_000_000, stills=10, **_ignored):
        self.out = out
        self.world = [float(w) for w in world]
        self.n_frames = int(n_frames)
        self.up = int(up)
        self.name = name
        self.sim = sim
        self.style = dict(style or {})
        self.dt, self.time_s = dt, time_s
        self.res = int(self.style.get("splat_res", res))
        self.fps = int(self.style.get("fps", fps))
        self.chunk = int(chunk)
        self.stride = max(1, int(np.ceil(self.n_frames / max(1, int(max_frames)))))
        # THE SAME VIEWPOINT EVERY OTHER MOVIE IN THE REPO USES. `LiveMovie` does not read
        # `camera_elev`/`camera_azim` at all -- it is hard-wired to elev 18 deg, azim -58 deg -- so
        # a splat that honoured those spec keys produced a picture that could not be compared with
        # any existing render of the same scene. `splat_elev_deg` / `splat_azim_deg` override.
        # 122 = -58 + 180 BECAUSE THE TWO ANGLES MEAN OPPOSITE THINGS: LiveMovie's is the direction
        # the camera SITS in, this one's is the direction it LOOKS. At -58 the bench's blocks, which
        # occupy x in [0.024, 0.496] of a unit box, rendered on the left of the frame where every
        # existing movie of that scene puts them on the right.
        self.azim = np.radians(float(self.style.get("splat_azim_deg", 122.0)))
        # NEGATED, AND THE PICTURE IS THE REASON. `plot._rot_camera` computes
        # `y2 = cos(e)*y1 - sin(e)*z`, so a LARGER world up-coordinate lands LOWER on screen once
        # the image is flipped for display -- si_waterfall's slab starts at y in [0.44, 0.49] of a
        # 0.5 box, i.e. at the top before it falls, and rendered at the bottom. Negating the
        # elevation is the same rotation viewed from the other side of the horizontal plane, and it
        # puts up on screen where up is in the world.
        self.elev = -np.radians(float(self.style.get("splat_elev_deg", 18.0)))
        self.zoom = float(self.style.get("camera_zoom", 0.0)) or 1.0
        self.fog = float(self.style.get("splat_fog", 0.55))
        self.gamma = float(self.style.get("splat_gamma", 0.8))
        self.density = float(self.style.get("splat_density", 0.02))
        self.blur = float(self.style.get("splat_blur", 1.6))
        self.slab = float(self.style.get("splat_slab", 0.06))
        self.box_frame = bool(self.style.get("box_frame", True))
        self.box_grey = float(self.style.get("box_grey", 0.42))
        self.margin = float(self.style.get("splat_margin", 1.10))
        self.bg = np.asarray(_to_rgb(self.style.get("background", "black")), np.float32)
        # STILLS, FOR THE SAME REASON `LiveMovie` WRITES THEM. An mp4 is only readable once
        # ffmpeg flushes at close, so a 1-hour run shows a 48-byte container header for an hour and
        # nobody can tell a working renderer from a broken one. `3d.png` is always the newest frame.
        self.stills = int(stills)
        self.still_dir = os.path.dirname(out) or "."
        self.stills_written = 0
        self._still_at = None
        self.rendered, self.n, self.drawn = 0, 0, 0
        self.failed = None
        self._w = None
        self._rgb = None
        self._t0 = None

    # ---------------------------------------------------------------- writer
    def _writer(self):
        if self._w is None:
            import imageio
            os.makedirs(os.path.dirname(self.out) or ".", exist_ok=True)
            self._w = imageio.get_writer(self.out, fps=max(1, self.fps), codec="libx264",
                                         quality=8, macro_block_size=None)
        return self._w

    # ---------------------------------------------------------------- colour
    def _colours(self, H, lvl, dev):
        """A PALETTE AND AN INDEX, never an [N,3] colour array.

        Materialising per-particle colour costs 12 B/particle: 1.2 GB at 100 M and 12 GB at 1 B.
        The 1 B run fits a B300 with about 9 GB spare, so that one array alone would take the run
        from "fits" to "OOM at the first rendered frame". The palette is [n_types, 3] and the type
        index already exists on the level, so the chunk loop looks up `tab[ti[chunk]]` and the only
        colour that is ever resident is the chunk's.
        """
        import torch
        if self._rgb is not None:
            return self._rgb
        tab = torch.tensor([[0.30, 0.62, 1.0]], device=dev, dtype=torch.float32)
        par = pnt = None
        # COLOUR IS THE PARENT BODY'S, exactly as `LiveMovie._rgb` resolves it: the palette is
        # indexed by the PARENT's `node_type`, and each particle carries its parent's slot. Both of
        # those already exist -- `parent` is [N] int64 the level owns, `node_type` is [n_cells] --
        # so nothing per-particle is allocated here.
        try:
            from plexus.plot import _typed_palette
            pname = getattr(lvl, "parent_name", None)
            _par = getattr(lvl, "parent", None)
            if pname and _par is not None:
                pal, _ = _typed_palette(self.sim, pname, self.style)
                _pnt = getattr(H.level(pname), "node_type", None)
                if pal is not None and _pnt is not None:
                    tab = torch.tensor(np.asarray(pal, np.float32)[:, :3], device=dev)
                    par, pnt = _par, _pnt.long()
        except Exception:
            pass
        self._rgb = (tab, par, pnt)
        return self._rgb

    # ---------------------------------------------------------------- render
    def _image(self, pos, palette):
        import torch
        tab, par, pnt = palette
        R = self.res
        dev = pos.device
        box = torch.tensor(self.world, device=dev, dtype=pos.dtype)
        # UP-AXIS FIRST. The camera treats world z as the vertical, so a scene whose gravity is on
        # y (MPM's convention) has to be permuted or it renders lying on its side.
        if self.up == 1:
            perm = [0, 2, 1]
        elif self.up == 0:
            perm = [1, 2, 0]
        else:
            perm = [0, 1, 2]
        # NO [N,3] COPY. `pos[:, perm] - 0.5*box` is 12 B/particle -- 12 GB at 1 B, against the
        # ~9 GB of headroom the 1 B run leaves on a B300. The permutation and the centring are done
        # per chunk inside `project` instead, where the temporary is the chunk's.
        cen = 0.5 * box[perm]
        ca, sa = float(np.cos(self.azim)), float(np.sin(self.azim))
        ce, se = float(np.cos(self.elev)), float(np.sin(self.elev))
        # FIT THE BOX, DO NOT ASSUME THE HALF-DIAGONAL. Using |box|/2 as the half-width is only
        # right when the view looks down a body diagonal; at any other angle the projected cube is
        # smaller, and at THIS angle it filled the frame corner to corner with no margin at all --
        # so the material appeared to run off the edge and there was nothing to judge it against.
        # Projecting the eight corners and fitting them is exact for any azim/elev.
        _c = torch.tensor([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)],
                          device=dev, dtype=box.dtype) * box[perm]
        _cx = ca * _c[:, 0] - sa * _c[:, 1]
        _cy = ce * (sa * _c[:, 0] + ca * _c[:, 1]) - se * _c[:, 2]
        span = float(torch.maximum(_cx.abs().max(), _cy.abs().max())) * self.margin \
            / max(self.zoom, 1e-6) + 1e-9
        acc_w = torch.zeros(R * R, device=dev)
        acc_c = torch.zeros(3, R * R, device=dev)
        zbuf = torch.full((R * R,), float("inf"), device=dev)

        def project(sl):
            chunk_p = pos[sl][:, perm] - cen
            x1 = ca * chunk_p[:, 0] - sa * chunk_p[:, 1]
            y1 = sa * chunk_p[:, 0] + ca * chunk_p[:, 1]
            y2 = ce * y1 - se * chunk_p[:, 2]
            depth = se * y1 + ce * chunk_p[:, 2]
            sx = ((x1 / (2 * span) + 0.5) * R).round().long().clamp_(0, R - 1)
            sy = ((y2 / (2 * span) + 0.5) * R).round().long().clamp_(0, R - 1)
            return sy * R + sx, depth

        n = pos.shape[0]
        for a in range(0, n, self.chunk):                       # pass 1: depth buffer
            flat, depth = project(slice(a, a + self.chunk))
            zbuf.scatter_reduce_(0, flat, depth, reduce="amin", include_self=True)
        zmin = float(zbuf[torch.isfinite(zbuf)].min()) if bool(torch.isfinite(zbuf).any()) else 0.0
        zmax = float(zbuf[torch.isfinite(zbuf)].max()) if bool(torch.isfinite(zbuf).any()) else 1.0
        dz = max(zmax - zmin, 1e-9)
        for a in range(0, n, self.chunk):                       # pass 2: colour + weight
            sl = slice(a, a + self.chunk)
            flat, depth = project(sl)
            keep = (depth <= zbuf[flat] + self.slab * dz).to(pos.dtype)
            dn = 1.0 - (depth - zmin) / dz                      # 1 = near, 0 = far
            w = keep * ((1.0 - self.fog) + self.fog * dn.clamp(0.0, 1.0))
            acc_w.index_add_(0, flat, w)
            # THE CHUNK'S COLOUR ONLY -- see `_colours`. `tab` is [n_types, 3] and `ti` indexes it.
            rgb = (tab[pnt[par[sl]] % tab.shape[0]] if par is not None
                   else tab[0].expand(flat.shape[0], 3))
            for ch in range(3):
                acc_c[ch].index_add_(0, flat, w * rgb[:, ch])

        acc_w = _blur(acc_w.view(R, R), self.blur)
        acc_c = torch.stack([_blur(acc_c[ch].view(R, R), self.blur) for ch in range(3)], -1)
        ref = float(acc_w.max()) * self.density + 1e-9
        col = acc_c / acc_w.clamp(min=1e-8)[..., None]
        alpha = (1.0 - torch.exp(-acc_w / ref)).clamp(0, 1)[..., None] ** self.gamma
        bg = torch.tensor(self.bg, device=dev)[None, None, :]
        img = (col * alpha + bg * (1 - alpha)).clamp(0, 1)
        if self.box_frame:
            # THE TWELVE EDGES, SAMPLED AND STAMPED. A wireframe is what tells a viewer that the
            # material is inside the domain rather than off the side of it -- the pyvista renderer
            # draws one and its absence here was read, reasonably, as particles leaving the scene.
            e = []
            for i in range(8):
                for d in range(3):
                    j = i ^ (1 << d)
                    if j > i:
                        e.append((i, j))
            t = torch.linspace(0, 1, 2 * R, device=dev, dtype=box.dtype)[:, None]
            for i, j in e:
                q = _c[i][None] * (1 - t) + _c[j][None] * t
                qx = ca * q[:, 0] - sa * q[:, 1]
                qy = ce * (sa * q[:, 0] + ca * q[:, 1]) - se * q[:, 2]
                px = ((qx / (2 * span) + 0.5) * R).round().long().clamp_(0, R - 1)
                py = ((qy / (2 * span) + 0.5) * R).round().long().clamp_(0, R - 1)
                img[py, px] = torch.maximum(img[py, px],
                                            torch.full((3,), self.box_grey, device=dev))
        return (img.flip(0) * 255).to(torch.uint8).cpu().numpy()

    # ---------------------------------------------------------------- hook
    def __call__(self, H, tick):
        import time

        import torch
        try:
            from plexus.live_movie import _biggest_particle_set
            sname = _biggest_particle_set(H)
            if sname is None:
                self.failed = "no set carries positions"
                return
            lvl = H.level(sname)
            if self._t0 is None:
                self._t0 = time.perf_counter()
                self.n = self.drawn = int(lvl.n)
            if tick % self.stride:
                return
            pos = lvl.get("pos")
            if pos.shape[1] != 3:
                self.failed = "gpu_splat is 3D only"
                return
            img = self._image(pos, self._colours(H, lvl, pos.device))
            el = time.perf_counter() - self._t0
            _fr = _stamp(
                img, f"{self.name}   {self.n:,} particles, all drawn   "
                     f"frame {tick}/{self.n_frames}   "
                     f"{el / max(tick, 1) * 1000:.0f} ms/frame compute")
            self._writer().append_data(np.ascontiguousarray(np.asarray(_fr, dtype=np.uint8)))
            self.rendered += 1
            # EVENLY SPACED OVER THE RENDERED FRAMES, computed once from how many there will be --
            # a modulo on a counter gave 16 stills for a requested 4.
            if self.stills > 0:
                if self._still_at is None:
                    R = max(1, -(-self.n_frames // self.stride))
                    k = max(1, min(self.stills, R))
                    self._still_at = {int(round(i * (R - 1) / max(k - 1, 1))) + 1
                                      for i in range(k)}
                if self.rendered in self._still_at:
                    self._write_still(_fr, tick)
        except Exception as e:                        # a renderer must never kill a long run
            # SAID AT ONCE, not at close(). Reporting only on close meant a 1-hour run could fail on
            # its first frame and say so 60 minutes later, which is the same as not saying it.
            if self.failed is None:
                print(f"[splat] renderer failed at frame {tick}: {type(e).__name__}: {e}",
                      flush=True)
            self.failed = f"{type(e).__name__}: {e}"

    def _write_still(self, img, tick):
        try:
            import imageio.v3 as iio
            iio.imwrite(os.path.join(self.still_dir,
                                     f"still_{self.stills_written:02d}_f{tick:05d}.png"), img)
            iio.imwrite(os.path.join(self.still_dir, "3d.png"), img)
            self.stills_written += 1
        except Exception as e:
            print(f"[splat] still at frame {tick} failed: {type(e).__name__}: {e}", flush=True)

    def close(self):
        if self._w is not None:
            self._w.close()
            self._w = None
        if self.failed:
            print(f"[splat] renderer failed: {self.failed}", flush=True)


def _parent_of(sim, name):
    s = (sim.sets.get(name) or {}) if getattr(sim, "sets", None) else {}
    return s.get("parent", name)


def _to_rgb(c):
    from matplotlib.colors import to_rgb
    return to_rgb(c)


def _blur(a, sigma: float):
    """Separable gaussian on an [R,R] accumulator. On the IMAGE, not the points -- res^2 work
    whatever N is, which is the whole reason the sprite loop is not needed."""
    import torch
    import torch.nn.functional as F
    if sigma <= 0:
        return a
    r = max(1, int(round(2 * sigma)))
    x = torch.arange(-r, r + 1, device=a.device, dtype=a.dtype)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / k.sum()
    t = a[None, None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return t[0, 0]


def _stamp(img, text):
    """The overlay, drawn with PIL so the renderer needs no VTK at all. Silently skipped if PIL is
    missing -- a missing caption is not a reason to lose the frame."""
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        for i, line in enumerate(text.split("\n")):
            d.text((12, 10 + 14 * i), line, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return img
