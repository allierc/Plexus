"""render_stress_pair_vtk -- the eye moving, next to the muscle stress that motion costs.

    python render_stress_pair_vtk.py --curves archive/eye_G/pairs_long_curves.npz

Two panels, one scene, same motion: LEFT is the plant AS IS -- `render_surface_vtk`'s own
scene, muscles in their plant colours. RIGHT is the SAME frame with the muscles coloured by
their von Mises stress instead of their identity, and drawn more transparent so a muscle's
OWN internal stress gradient -- belly against tendon, near side against far -- is visible
through the tissue rather than only on its silhouette. Nothing is re-simulated: von Mises
stress (`mus_vm`) is captured by every run already (`run_eye._scalars`, elastic + active
Cauchy stress, no `--instrument` needed), so this is a second view of a run you already have.

WHY A SEPARATE SCRIPT rather than a flag on `render_surface_vtk`. The two panels need
independent colouring of the SAME geometry at the SAME frame, which means two `Skin`
instances (one per panel) sharing one clock -- close enough to `SurfaceScene` that copying
its skinning and camera code was simpler than threading a second colour mode through it.

THE COLOUR SCALE IS FIXED FOR THE WHOLE MOVIE, not autoscaled per frame: the 1st-99th
percentile of `mus_vm` over every captured frame, so a color means the same stress on frame
10 as on frame 400. Autoscaling per frame would make a quiet hold look exactly as "red" as a
peak contraction, which defeats the purpose of drawing stress at all.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyvista as pv
from tqdm import tqdm

import eye_anatomy as EA
import blend_mpm_ops as BM
from render_eye import PALETTE, MUS_RGB
from render_surface_vtk import Skin, _poly, K_BIND, GLOBE_ALPHA
from render_orbit_vtk import azimuth_schedule, gaze_marker

STRESS_CMAP = "inferno"                 # this project's convention for a stress panel


def _stress_clim(mus_vm):
    lo, hi = np.percentile(mus_vm, [1.0, 99.0])
    return float(lo), float(max(hi, lo + 1e-6))


class StressPairScene:
    """One 2-panel scene: left the plant as drawn elsewhere, right muscle stress."""

    def __init__(self, cap, side="R", blend=None, parts=None, size=(2000, 1100),
                 globe_alpha=GLOBE_ALPHA, stress_alpha=0.72, span=None, inflate=1.0):
        self.cap = cap
        self.tissue = np.asarray(cap["tissue"])
        self.mus_parent = np.asarray(cap["mus_parent"])
        self.centre0 = np.asarray(cap["centre"][0], float)

        d, man = BM.load_cut(blend or BM.DEFAULT_BLEND, parts or BM.DEFAULT_PARTS)
        fr = BM.BlendFrame(man, d, side, EA.A_EQ, EA.GLOBE_CENTER, inflate)
        shell0 = np.asarray(cap["shell"][0], float)
        mus0 = np.asarray(cap["mus_pos"][0], float)

        pv.OFF_SCREEN = True
        self.p = pv.Plotter(off_screen=True, window_size=size, shape=(1, 2), border=False)
        self.clim = _stress_clim(np.asarray(cap["mus_vm"]))

        self.globe = {0: [], 1: []}       # panel -> [(mesh, skin), ...]
        self.muscles = {0: [], 1: []}     # panel -> [(mi, mesh, skin, own), ...]
        for panel in (0, 1):
            self.p.subplot(0, panel)
            self.p.set_background("black")
            self.p.enable_depth_peeling(10)
            self._build_globe(panel, d, side, fr, shell0, globe_alpha)
            self._build_muscles(panel, d, side, fr, mus0,
                               stress_alpha if panel == 1 else 1.0)
            self.p.add_text("a)  the plant, as driven" if panel == 0
                            else "b)  muscle stress (von Mises, fixed scale)",
                            position="upper_edge", font_size=13, color="white")
        self.p.subplot(0, 1)
        self.p.add_scalar_bar(title="von Mises stress", color="white", vertical=True,
                              position_x=0.90, position_y=0.08, width=0.05, height=0.5,
                              n_labels=3, fmt="%.2f")

        all0 = np.concatenate([shell0, mus0])
        reach = float(np.abs(all0 - self.centre0[None, :]).max())
        self.span = float(span if span is not None else 1.15 * reach)
        self.gaze_sel = gaze_marker(self.tissue)
        self.arrow_len = 1.1 * float(np.abs(shell0 - self.centre0[None, :]).max())
        for panel in (0, 1):
            self.p.subplot(0, panel)
            t0, d0 = self._gaze(0)
            if d0 is not None:
                self.p.add_mesh(pv.Arrow(start=t0, direction=d0, tip_length=0.26,
                                         tip_radius=0.075, shaft_radius=0.024,
                                         scale=self.arrow_len),
                                color="#7a7a7a", opacity=0.5, name="gaze_rest")
        self._text("")

    def _build_globe(self, panel, d, side, fr, shell0, globe_alpha):
        for part, alpha, spec in (("retina", globe_alpha, 0.30), ("cornea", 0.26, 0.65),
                                  ("lens", 0.85, 0.85)):
            key = f"{side}_{part}"
            if f"{key}__v" not in d:
                continue
            V = fr.globe(d[f"{key}__v"])
            mesh = _poly(V, d[f"{key}__f"])
            skin = Skin(V, shell0)
            mesh["rgb"] = np.clip(PALETTE[skin.nearest(self.tissue)], 0, 1).astype(np.float32)
            self.p.add_mesh(mesh, scalars="rgb", rgb=True, opacity=alpha, smooth_shading=True,
                            specular=spec, specular_power=24, show_scalar_bar=False)
            self.globe[panel].append((mesh, skin))

    def _build_muscles(self, panel, d, side, fr, mus0, opacity):
        for mi, key in enumerate(EA.MUSCLE_KEYS):
            name = f"{side}_{key}"
            if f"{name}__v" not in d:
                continue
            own = self.mus_parent == mi
            if own.sum() < K_BIND:
                continue
            V = fr(d[f"{name}__v"])
            mesh = _poly(V, d[f"{name}__f"])
            skin = Skin(V, mus0[own])
            if panel == 0:
                mesh["rgb"] = np.tile(np.array(_hex_rgb(EA.MUSCLES[mi]["color"]), np.float32),
                                      (mesh.n_points, 1))
                self.p.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True,
                                specular=0.35, specular_power=22, show_scalar_bar=False,
                                name=f"mus{panel}_{mi}")
            else:
                mesh["stress"] = np.zeros(mesh.n_points, np.float32)
                self.p.add_mesh(mesh, scalars="stress", cmap=STRESS_CMAP, clim=self.clim,
                                opacity=opacity, smooth_shading=True, specular=0.2,
                                show_scalar_bar=False, name=f"mus{panel}_{mi}")
            self.muscles[panel].append((mi, mesh, skin, own))

    def _gaze(self, k):
        if self.gaze_sel is None:
            return None, None
        g = np.asarray(self.cap["shell"][k], float)
        tip = g[self.gaze_sel].mean(axis=0)
        v = tip - np.asarray(self.cap["centre"][k], float)
        n = np.linalg.norm(v)
        return (tip, v / n) if n > 1e-9 else (None, None)

    def _text(self, hud):
        self.p.subplot(0, 0)
        self.p.add_text(hud, position="lower_left", font_size=11, color="white", name="hud")

    def camera(self, az_deg, el_deg=18.0):
        c = self.centre0
        a, e = np.radians(az_deg), np.radians(el_deg)
        d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        for panel in (0, 1):
            self.p.subplot(0, panel)
            self.p.camera_position = (tuple(c + d * 10.0), tuple(c), (0.0, 1.0, 0.0))
            self.p.camera.parallel_projection = True
            self.p.camera.parallel_scale = self.span

    def frame(self, k, az_deg, dt):
        cap = self.cap
        X = np.asarray(cap["shell"][k], float)
        for panel in (0, 1):
            for mesh, skin in self.globe[panel]:
                mesh.points = skin(X)
        Y = np.asarray(cap["mus_pos"][k], float)
        vm = np.asarray(cap["mus_vm"][k], float)
        act = np.asarray(cap["act"][k], float)
        for panel in (0, 1):
            for mi, mesh, skin, own in self.muscles[panel]:
                mesh.points = skin(Y[own])
                if panel == 0:
                    base = MUS_RGB[mi]
                    lit = np.clip(base * (0.78 + 0.42 * float(np.clip(act[mi], 0, 1))), 0, 1)
                    mesh["rgb"] = np.tile(lit.astype(np.float32), (mesh.n_points, 1))
                else:
                    mesh["stress"] = skin.scalar(vm[own])
        h, v, t = np.asarray(cap["gaze"][k], float)
        fr_i = int(cap["frame"][k])
        length = np.asarray(cap["length"][k], float)
        rest = np.asarray(cap["rest_length"], float)
        shorten = 100.0 * (1.0 - length / rest)          # + shorter than rest, - stretched
        len_line = "  ".join(f"{key} {s:+4.1f}%" for key, s in zip(EA.MUSCLE_KEYS, shorten))
        self._text(f"frame {fr_i:4d}   t = {fr_i * dt:5.2f} s   "
                  f"gaze h {h:+5.1f}  v {v:+5.1f}  t {t:+5.1f}\n"
                  f"muscle length, % shorter than rest (- = stretched)   {len_line}")
        self.camera(az_deg)
        return self.p.screenshot(return_img=True)

    def close(self):
        self.p.close()


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def render(cap, dt, out_mp4, out_strip=None, fps=30, size=(2000, 1100), turns=1.0,
           quality=8, globe_alpha=GLOBE_ALPHA, stress_alpha=0.72, strip_n=5,
           still_margin=0.03, still_above=None, az0=0.0, side="R", blend=None, parts=None):
    import imageio.v2 as iio

    n = len(cap["frame"])
    scene = StressPairScene(cap, side=side, blend=blend, parts=parts, size=size,
                            globe_alpha=globe_alpha, stress_alpha=stress_alpha)
    az, moving = azimuth_schedule(cap, turns=turns, still_margin=still_margin,
                                  still_above=still_above, az0=az0)
    print(f"[stress-pair] camera turns on {int((~moving).sum())} of {n} frames; "
          f"held still on {int(moving.sum())} while a muscle contracts; "
          f"stress scale fixed to [{scene.clim[0]:.3f}, {scene.clim[1]:.3f}]", flush=True)
    strip_at = set(np.linspace(0, n - 1, strip_n).round().astype(int).tolist())
    strip = []
    with iio.get_writer(out_mp4, fps=fps, quality=quality, macro_block_size=None) as w:
        for k in tqdm(range(n), desc="[render]", unit="frame", dynamic_ncols=True,
                     ncols=140, leave=False):
            img = scene.frame(k, float(az[k]), dt)
            w.append_data(img)
            if k in strip_at:
                strip.append(img)
    scene.close()
    if out_strip and strip:
        iio.imwrite(out_strip, np.concatenate(strip, axis=1))
    return out_mp4


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--curves", required=True, help="a *_curves.npz from run_eye_G.py")
    ap.add_argument("--out", default=None, help="defaults next to --curves")
    ap.add_argument("--side", default="R", choices=("L", "R"))
    ap.add_argument("--blend", default=None)
    ap.add_argument("--parts", default=None)
    ap.add_argument("--turns", type=float, default=1.0)
    ap.add_argument("--az", type=float, default=25.0)
    ap.add_argument("--dt", type=float, default=0.003)
    ap.add_argument("--stress-alpha", type=float, default=0.72)
    args = ap.parse_args()

    cap = {k: v for k, v in np.load(args.curves).items()}
    stem = args.out or args.curves.replace("_curves.npz", "_stress_pair")
    mp4 = stem + ".mp4"
    render(cap, args.dt, mp4, stem + ".png", turns=args.turns, az0=args.az, side=args.side,
          blend=args.blend, parts=args.parts, stress_alpha=args.stress_alpha)
    print(f"[stress-pair] {mp4}")


if __name__ == "__main__":
    main()
