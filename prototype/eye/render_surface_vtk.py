"""render_surface_vtk -- the MPM run drawn as SURFACES, not as a point cloud.

    from render_surface_vtk import render
    render(cap, dt, "movie.mp4", "strip.png")

`render_orbit_vtk` draws the material points themselves, which is honest and unreadable:
45 000 dots make a speckled ball, and the six straps lose the shape the model gave them.
This draws the SAME RUN through the Blender geometry -- the smooth shaded globe and the six
solid straps of `archive/run_03/view_muscles_vtk.py` -- with the simulation moving them.

HOW THE SURFACE FOLLOWS THE SIMULATION. The meshes are not re-extracted from the particles
(marching cubes on a point cloud loses exactly the crispness this is for). Each mesh is
SKINNED to the particles that were seeded inside it:

    every vertex is bound, once, to its `k` nearest captured particles AT REST, with
    inverse-square weights; thereafter the vertex simply rides them,
        x_v(t) = sum_i w_i x_i(t) ,   sum_i w_i = 1 .

Linear blend skinning, in other words, with the MPM particles as the bones. It costs one
sparse matrix multiply per frame, it reproduces the rest shape exactly at t = 0 (the
weights sum to one and every particle is at its rest position), and it inherits the
simulation's rotations and stretches because the particles do. What it cannot represent is
deformation FINER than the particle spacing, which is the same limit the simulation has.

Colour is transferred the same way: each globe vertex takes the tissue of its nearest
captured particle, so the pupil, the iris ring and the gold iridophore flecks -- which is
what makes torsion visible -- appear ON the surface rather than as loose dots. Muscles keep
one flat colour each and brighten with activation.

The camera schedule, the gaze arrow and the HUD are `render_orbit_vtk`'s, imported rather
than copied: the same run should be readable the same way whichever renderer draws it.
"""
from __future__ import annotations

import os

from tqdm import tqdm
import numpy as np
import pyvista as pv

import eye_anatomy as EA
import blend_mpm_ops as BM
from render_eye import PALETTE, MUS_RGB
from render_orbit_vtk import azimuth_schedule, gaze_marker

K_BIND = 8                      # particles a vertex is bound to
GLOBE_ALPHA = 0.20              # the eyeball is translucent -- the muscles run behind it


def _poly(v, f):
    return pv.PolyData(np.asarray(v, float),
                       np.hstack([np.full((len(f), 1), 3, np.int64),
                                  np.asarray(f, np.int64)]).ravel())


class Skin:
    """A mesh bound to a set of moving particles: `deform(X)` returns its vertices."""

    def __init__(self, verts, rest_pts, k=K_BIND):
        from scipy.spatial import cKDTree
        k = int(min(k, len(rest_pts)))
        d, idx = cKDTree(rest_pts).query(np.asarray(verts, float), k=k)
        d = np.atleast_2d(d.T).T if k > 1 else d[:, None]
        idx = np.atleast_2d(idx.T).T if k > 1 else idx[:, None]
        w = 1.0 / np.maximum(d, 1e-9) ** 2
        self.w = (w / w.sum(axis=1, keepdims=True)).astype(np.float64)
        self.idx = idx.astype(np.int64)
        # the bind pose is where the particles put the vertex, so t = 0 renders the mesh
        # exactly as the artist drew it and every later frame is a pure displacement
        self.offset = np.asarray(verts, float) - self.deform(rest_pts)

    def deform(self, X):
        return np.einsum('vk,vkj->vj', self.w, np.asarray(X, float)[self.idx])

    def __call__(self, X):
        return self.deform(X) + self.offset

    def nearest(self, values):
        """Per-vertex value taken from the closest bound particle (for the tissue colours)."""
        return np.asarray(values)[self.idx[:, 0]]


class SurfaceScene:
    """The blend's meshes, skinned to the run's particles, in one translucent-globe scene."""

    def __init__(self, cap, side="R", blend=None, parts=None, size=(1600, 1200),
                 globe_alpha=GLOBE_ALPHA, span=None, inflate=1.0):
        self.cap = cap
        self.tissue = np.asarray(cap["tissue"])
        self.mus_parent = np.asarray(cap["mus_parent"])
        self.centre0 = np.asarray(cap["centre"][0], float)

        d, man = BM.load_cut(blend or BM.DEFAULT_BLEND, parts or BM.DEFAULT_PARTS)
        fr = BM.BlendFrame(man, d, side, EA.A_EQ, EA.GLOBE_CENTER, inflate)
        shell0 = np.asarray(cap["shell"][0], float)
        mus0 = np.asarray(cap["mus_pos"][0], float)

        pv.OFF_SCREEN = True
        self.p = pv.Plotter(off_screen=True, window_size=size, border=False)
        self.p.set_background("black")
        self.p.enable_depth_peeling(10)

        # --- the globe: retina + cornea translucent, lens solid -------------------
        self.globe = []
        for part, alpha, spec in (("retina", globe_alpha, 0.30), ("cornea", 0.26, 0.65),
                                  ("lens", 0.85, 0.85)):
            key = f"{side}_{part}"
            if f"{key}__v" not in d:
                continue
            V = fr.globe(d[f"{key}__v"])           # the globe is drawn as BUILT (inflated)
            mesh = _poly(V, d[f"{key}__f"])
            skin = Skin(V, shell0)
            mesh["rgb"] = np.clip(PALETTE[skin.nearest(self.tissue)], 0, 1).astype(np.float32)
            self.p.add_mesh(mesh, scalars="rgb", rgb=True, opacity=alpha, smooth_shading=True,
                            specular=spec, specular_power=24, show_scalar_bar=False)
            self.globe.append((mesh, skin))

        # --- the six straps, one skin each, bound only to their OWN particles ------
        self.muscles = []
        for mi, key in enumerate(EA.MUSCLE_KEYS):
            name = f"{side}_{key}"
            if f"{name}__v" not in d:
                continue
            own = self.mus_parent == mi
            if own.sum() < K_BIND:
                continue
            V = fr(d[f"{name}__v"])
            mesh = _poly(V, d[f"{name}__f"])
            self.p.add_mesh(mesh, color=EA.MUSCLES[mi]["color"], smooth_shading=True,
                            specular=0.35, specular_power=22, show_scalar_bar=False,
                            name=f"mus{mi}")
            self.muscles.append((mi, mesh, Skin(V, mus0[own]), own))

        all0 = np.concatenate([shell0, mus0])
        reach = float(np.abs(all0 - self.centre0[None, :]).max())
        self.span = float(span if span is not None else 1.15 * reach)
        self.gaze_sel = gaze_marker(self.tissue)
        self.arrow_len = 1.1 * float(np.abs(shell0 - self.centre0[None, :]).max())
        t0, d0 = self._gaze(0)
        if d0 is not None:
            self.p.add_mesh(pv.Arrow(start=t0, direction=d0, tip_length=0.26, tip_radius=0.075,
                                     shaft_radius=0.024, scale=self.arrow_len),
                            color="#7a7a7a", opacity=0.5, name="gaze_rest")
        self._text("", "")

    # --- the pieces shared with the point renderer ---------------------------
    def _gaze(self, k):
        if self.gaze_sel is None:
            return None, None
        g = np.asarray(self.cap["shell"][k], float)
        tip = g[self.gaze_sel].mean(axis=0)
        v = tip - np.asarray(self.cap["centre"][k], float)
        n = np.linalg.norm(v)
        return (tip, v / n) if n > 1e-9 else (None, None)

    def _text(self, hud, legend):
        self.p.add_text(hud, position="upper_left", font_size=11, color="white", name="hud")
        self.p.add_text(legend, position="lower_left", font_size=10, color="white", name="legend")

    def camera(self, az_deg, el_deg=18.0):
        c = self.centre0
        a, e = np.radians(az_deg), np.radians(el_deg)
        d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        self.p.camera_position = (tuple(c + d * 10.0), tuple(c), (0.0, 1.0, 0.0))
        self.p.camera.parallel_projection = True
        self.p.camera.parallel_scale = self.span

    def frame(self, k, az_deg, dt):
        cap = self.cap
        X = np.asarray(cap["shell"][k], float)
        for mesh, skin in self.globe:
            mesh.points = skin(X)
        Y = np.asarray(cap["mus_pos"][k], float)
        act = np.asarray(cap["act"][k], float)
        for mi, mesh, skin, own in self.muscles:
            mesh.points = skin(Y[own])
            # brighten with activation, as the point renderer does
            # a muscle at rest must still read as ITS colour: the point renderer could dim
            # to 0.55 of base and stay legible because a dot is its own light source, but a
            # shaded surface seen THROUGH the translucent globe cannot.
            base = MUS_RGB[mi]
            lit = np.clip(base * (0.78 + 0.42 * float(np.clip(act[mi], 0, 1))), 0, 1)
            self.p.renderer.actors[f"mus{mi}"].prop.color = tuple(float(c) for c in lit)
        tip, d = self._gaze(k)
        if d is not None:
            self.p.add_mesh(pv.Arrow(start=tip, direction=d, tip_length=0.26, tip_radius=0.075,
                                     shaft_radius=0.024, scale=self.arrow_len),
                            color="#ffe066", name="gaze")
        h, v, t = np.asarray(cap["gaze"][k], float)
        th, tv, tt = np.asarray(cap["target"][k], float)
        fr = int(cap["frame"][k])
        self._text(f"frame {fr:4d}   t = {fr * dt:5.2f} s\n"
                   f"command  h {th:+6.1f}  v {tv:+6.1f}  t {tt:+6.1f}\n"
                   f"gaze     h {h:+6.1f}  v {v:+6.1f}  t {t:+6.1f}",
                   "activation   " + "   ".join(f"{k_} {act[j]:.2f}"
                                                for j, k_ in enumerate(EA.MUSCLE_KEYS)))
        self.camera(az_deg)
        return self.p.screenshot(return_img=True)

    def close(self):
        self.p.close()


def render(cap, dt, out_mp4, out_strip=None, fps=30, size=(1600, 1200), turns=1.0,
           quality=8, globe_alpha=GLOBE_ALPHA, strip_n=5, still_margin=0.03,
           still_above=None, az0=0.0, side="R", blend=None, parts=None, inflate=1.0):
    """Same signature as `render_orbit_vtk.render`, so the two are interchangeable."""
    import imageio.v2 as iio

    n = len(cap["frame"])
    scene = SurfaceScene(cap, side=side, blend=blend, parts=parts, size=size,
                         globe_alpha=globe_alpha, inflate=inflate)
    az, moving = azimuth_schedule(cap, turns=turns, still_margin=still_margin,
                                  still_above=still_above, az0=az0)
    print(f"[surface] camera turns on {int((~moving).sum())} of {n} frames; "
          f"held still on {int(moving.sum())} while a muscle contracts", flush=True)
    strip_at = set(np.linspace(0, n - 1, strip_n).round().astype(int).tolist())
    strip = []
    with iio.get_writer(out_mp4, fps=fps, quality=quality, macro_block_size=None) as w:
        for k in tqdm(range(n), desc="[render]", unit="frame", dynamic_ncols=True, ncols=140, leave=False):
            img = scene.frame(k, float(az[k]), dt)
            w.append_data(img)
            if k in strip_at:
                strip.append(img)
    scene.close()
    if out_strip and strip:
        iio.imwrite(out_strip, np.concatenate(strip, axis=1))
    return out_mp4
