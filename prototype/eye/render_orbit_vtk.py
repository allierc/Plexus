"""render_orbit_vtk -- one 3-D scene of the running eye, with the camera flying round it.

    from render_orbit_vtk import render
    render(cap, dt, "movie.mp4", "strip.png")

`render_eye_vtk` draws six fixed panels; this draws ONE, and moves the camera instead. It
is the view the Blender model is easiest to check against: the globe is TRANSLUCENT, so the
six muscles, the lens and the far wall of the sclera are all visible through it, and one
orbit of the camera shows every insertion in turn without a single cutaway.

WHAT IS DRAWN

    globe      the shell points, coloured by tissue (`render_eye.PALETTE`) at `globe_alpha`
               opacity -- the transparency is the point of this renderer
    lens       drawn opaque inside the translucent globe, because a zebrafish lens is a
               hard ball optic and it is the clearest single marker of where the eye points
    muscles    the muscle points, opaque, in `render_eye.MUS_RGB`, BRIGHTENING WITH
               ACTIVATION -- so which muscle is driving a given rotation is visible without
               reading a trace
    hud        frame, simulated time, the commanded and the achieved (h, v, t)

Depth peeling is on: without it VTK composites the translucent shell in draw order and the
muscles behind it come out wrong (and flicker as the camera turns, since draw order changes
with it).

The camera orbits in the plant's frame, where +z is the OPTIC AXIS: azimuth sweeps `turns`
full turns about the dorsal axis over the whole movie while the elevation holds, so the
scene is seen from in front of the cornea, from the caudal side, from behind and back. The
sweep is gated on the drive (see `azimuth_schedule`), and `turns=0` LOCKS the camera at
`az0` for the whole movie -- the only thing that then moves on screen is the eye.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

import eye_anatomy as EA
from render_eye import PALETTE, MUS_RGB


def _rgb_globe(tissue):
    return np.clip(PALETTE[tissue], 0, 1).astype(np.float32)


def _rgb_muscle(parent, act):
    """Muscle colour, brightened by that muscle's activation (0.55 -> 1.30 of base)."""
    base = MUS_RGB[parent]
    lit = 0.55 + 0.75 * np.clip(act[parent], 0.0, 1.0)
    return np.clip(base * lit[:, None], 0, 1).astype(np.float32)


# tissue ids, in TISSUE_NAMES order (eye_anatomy.EyeAnatomy / blend_mpm_ops.BlendGlobe)
PUPIL, CORNEA, LENS = 6, 3, 7


def gaze_marker(tissue):
    """Which labelled tissue to take the optic axis from, and its point mask.

    MEASURED, not reconstructed. The eye's `gaze` block is a clinical (h, v, t) triple, and
    turning it back into a direction means re-implementing the Euler convention and hoping
    the two agree. The pupil is a disc centred on the optic axis by construction, so the
    direction from the globe's centroid to the pupil's centroid IS the gaze -- read off the
    same deforming tissue everything else is read from. Cornea then lens are the fallbacks.
    """
    for tid in (PUPIL, CORNEA, LENS):
        sel = tissue == tid
        if sel.sum() >= 20:
            return sel
    return None


class OrbitScene:
    """One live scene whose clouds are UPDATED per frame; only the camera and the colours move."""

    def __init__(self, cap, size=(1600, 1200), globe_alpha=0.30, lens_alpha=0.85,
                 point_globe=7.0, point_muscle=9.0, span=None):
        self.cap = cap
        self.tissue = np.asarray(cap["tissue"])
        self.mus_parent = np.asarray(cap["mus_parent"])
        lens_id = 7                                    # EyeAnatomy/BlendGlobe TISSUE_NAMES order
        self.is_lens = self.tissue == lens_id
        self.centre0 = np.asarray(cap["centre"][0], float)
        # frame on what is actually there -- globe plus the muscles' reach, which is wider
        # than the globe by however far the origins sit back in the orbit
        all0 = np.concatenate([np.asarray(cap["shell"][0], float),
                               np.asarray(cap["mus_pos"][0], float)])
        reach = float(np.abs(all0 - self.centre0[None, :]).max())
        self.span = float(span if span is not None else 1.15 * reach)

        pv.OFF_SCREEN = True
        self.p = pv.Plotter(off_screen=True, window_size=size, border=False)
        self.p.set_background("black")
        self.p.enable_depth_peeling(10)                # translucency has to be order-independent

        g = np.asarray(cap["shell"][0], float)
        self.globe = pv.PolyData(g[~self.is_lens])
        self.globe["rgb"] = _rgb_globe(self.tissue[~self.is_lens])
        self.lens = pv.PolyData(g[self.is_lens]) if self.is_lens.any() else None
        m = np.asarray(cap["mus_pos"][0], float)
        self.mus = pv.PolyData(m)
        self.mus["rgb"] = _rgb_muscle(self.mus_parent, np.asarray(cap["act"][0]))

        self.p.add_mesh(self.globe, scalars="rgb", rgb=True, point_size=point_globe,
                        render_points_as_spheres=True, opacity=globe_alpha,
                        show_scalar_bar=False)
        if self.lens is not None:
            self.lens["rgb"] = _rgb_globe(self.tissue[self.is_lens])
            self.p.add_mesh(self.lens, scalars="rgb", rgb=True, point_size=point_globe,
                            render_points_as_spheres=True, opacity=lens_alpha,
                            show_scalar_bar=False)
        self.p.add_mesh(self.mus, scalars="rgb", rgb=True, point_size=point_muscle,
                        render_points_as_spheres=True, show_scalar_bar=False)

        # the gaze arrow, and a dim one frozen at the primary position to deviate from
        self.gaze_sel = gaze_marker(self.tissue)
        self.arrow_len = 1.1 * float(np.abs(g - self.centre0[None, :]).max())
        t0, d0 = self._gaze(0)
        if d0 is not None:
            self.p.add_mesh(pv.Arrow(start=t0, direction=d0, tip_length=0.26,
                                     tip_radius=0.075, shaft_radius=0.024,
                                     scale=self.arrow_len),
                            color="#7a7a7a", opacity=0.5, name="gaze_rest")   # primary position
        self._text("", "")

    def _gaze(self, k):
        """(start, unit direction) of the optic axis at capture frame `k`.

        The arrow LEAVES THE PUPIL rather than starting at the globe's centre: an arrow
        drawn from the centre lies inside the translucent shell for most of its length, and
        when the camera swings round to look down the optic axis it foreshortens into a blob
        on top of the eye. Starting on the surface, it reads as a beam out of the pupil from
        every angle.
        """
        if self.gaze_sel is None:
            return None, None
        g = np.asarray(self.cap["shell"][k], float)
        tip = g[self.gaze_sel].mean(axis=0)
        v = tip - np.asarray(self.cap["centre"][k], float)
        n = np.linalg.norm(v)
        return (tip, v / n) if n > 1e-9 else (None, None)

    def _text(self, hud, legend):
        """Re-add the two overlays under fixed names -- pyvista replaces an actor of the same
        name, and a corner annotation has no in-place setter that survives a screenshot."""
        self.p.add_text(hud, position="upper_left", font_size=11, color="white", name="hud")
        self.p.add_text(legend, position="lower_left", font_size=10, color="white", name="legend")

    def camera(self, az_deg, el_deg=18.0):
        c = self.centre0
        a, e = np.radians(az_deg), np.radians(el_deg)
        # +z is the optic axis, +y dorsal: orbit about the DORSAL axis, starting in front
        # of the cornea (az = 0 looks along -z, i.e. straight at the pupil).
        d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        self.p.camera_position = (tuple(c + d * 10.0), tuple(c), (0.0, 1.0, 0.0))
        self.p.camera.parallel_projection = True
        self.p.camera.parallel_scale = self.span

    def frame(self, k, az_deg, dt):
        cap = self.cap
        g = np.asarray(cap["shell"][k], float)
        self.globe.points = g[~self.is_lens]
        if self.lens is not None:
            self.lens.points = g[self.is_lens]
        self.mus.points = np.asarray(cap["mus_pos"][k], float)
        self.mus["rgb"] = _rgb_muscle(self.mus_parent, np.asarray(cap["act"][k]))
        tip, d = self._gaze(k)
        if d is not None:
            self.p.add_mesh(pv.Arrow(start=tip, direction=d, tip_length=0.26,
                                     tip_radius=0.075, shaft_radius=0.024,
                                     scale=self.arrow_len),
                            color="#ffe066", name="gaze")     # same name -> replaces last frame's
        h, v, t = np.asarray(cap["gaze"][k], float)
        th, tv, tt = np.asarray(cap["target"][k], float)
        fr = int(cap["frame"][k])
        a = np.asarray(cap["act"][k], float)
        self._text(f"frame {fr:4d}   t = {fr * dt:5.2f} s\n"
                   f"command  h {th:+6.1f}  v {tv:+6.1f}  t {tt:+6.1f}\n"
                   f"gaze     h {h:+6.1f}  v {v:+6.1f}  t {t:+6.1f}",
                   "activation   " + "   ".join(f"{k_} {a[j]:.2f}"
                                                for j, k_ in enumerate(EA.MUSCLE_KEYS)))
        self.camera(az_deg)
        return self.p.screenshot(return_img=True)

    def close(self):
        self.p.close()


def azimuth_schedule(cap, turns=1.0, still_margin=0.03, still_above=None, az0=0.0):
    """[n] camera azimuth, which ADVANCES ONLY WHILE THE MUSCLES ARE QUIET.

    A camera that keeps turning through a contraction is unreadable: the globe swings
    because a muscle pulled it and it also swings because the viewpoint moved, and the two
    cannot be told apart on screen. So the orbit is gated on the drive -- the camera holds
    perfectly still on every frame where any muscle is activated above its resting level,
    and takes up the remaining rotation over the quiet frames, so a full `turns` is still
    completed by the end.

    The resting level is read from the run itself, as the 20th percentile of the per-frame
    peak activation -- i.e. a typical QUIET frame. The global minimum will not do: it is the
    opening frame, before activation has even risen to its tonic level, so a threshold set
    from it counts tonic innervation as "pulling" and the camera never moves at all (which
    is exactly what the first pass did). Pass `still_above` to set the level explicitly when
    the spec's `tonic` is known.
    """
    act = np.asarray(cap["act"], float)                    # [n, 6]
    rest = float(still_above if still_above is not None
                 else np.percentile(act.max(axis=1), 20.0))
    moving = act.max(axis=1) > rest + still_margin         # frames where a muscle is pulling
    if abs(turns) < 1e-9:                                  # turns = 0: a LOCKED camera
        return np.full(len(act), float(az0)), moving
    step = np.where(moving, 0.0, 1.0)
    if step.sum() < 1e-9:                                  # everything is active: turn anyway
        step = np.ones_like(step)
    return az0 + 360.0 * turns * np.concatenate([[0.0], np.cumsum(step)[:-1]]) / step.sum(), moving


def render(cap, dt, out_mp4, out_strip=None, fps=30, size=(1600, 1200), turns=1.0,
           quality=8, globe_alpha=0.30, strip_n=5, still_margin=0.03, still_above=None,
           az0=0.0):
    """The movie: every captured frame, the camera orbiting only between contractions."""
    import imageio.v2 as iio

    n = len(cap["frame"])
    scene = OrbitScene(cap, size=size, globe_alpha=globe_alpha)
    az, moving = azimuth_schedule(cap, turns=turns, still_margin=still_margin,
                                  still_above=still_above, az0=az0)
    print(f"[orbit] camera turns on {int((~moving).sum())} of {n} frames; "
          f"held still on {int(moving.sum())} while a muscle contracts", flush=True)
    strip_at = set(np.linspace(0, n - 1, strip_n).round().astype(int).tolist())
    strip = []
    with iio.get_writer(out_mp4, fps=fps, quality=quality, macro_block_size=None) as w:
        for k in range(n):
            img = scene.frame(k, float(az[k]), dt)
            w.append_data(img)
            if k in strip_at:
                strip.append(img)
    scene.close()
    if out_strip and strip:
        import imageio.v2 as iio2
        iio2.imwrite(out_strip, np.concatenate(strip, axis=1))
    return out_mp4
