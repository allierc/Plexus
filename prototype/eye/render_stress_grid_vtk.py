"""render_stress_grid_vtk -- the stress-pair view, expanded into a 7-panel mechanism board.

    python render_stress_grid_vtk.py --curves archive/eye_H/duction_h_L_m20_k30000_curves.npz

`render_stress_pair_vtk` answers ONE question (where is the stress). This answers the
follow-on questions that came out of reading that movie frame by frame: is the stress axial
(fibre tension) or transverse (shear/bending)? how far has the tissue actually MOVED from
rest, not just how hard is it pushing back? and does the routing itself explain any of it,
independent of stress?

    a) the plant, as driven         -- unchanged, `render_stress_pair_vtk`'s own left panel,
                                        BIG (spans the 2x2 the other four panels occupy)
    b) total stress                 -- von Mises, `render_stress_pair_vtk`'s own field
    c) axial stress                 -- |sigma_par| = |f^T sigma f|, fibre-aligned normal
                                        stress; sign (tension vs compression) is dropped from
                                        the COLOUR here so b, c, d share one LUT and one fixed
                                        scale -- a shade means the same magnitude in any of the
                                        three -- but the signed value is still in the capture
                                        (`mus_axial_p`) for anything that wants it
    d) transverse stress            -- sigma_perp = sqrt(||sigma||_F^2 - sigma_par^2), the
                                        part of the traction NOT along the fibre (shear +
                                        off-axis normal stress); same scale as b, c
    e) muscle length, % vs rest     -- one flat colour per muscle, diverging at 0: stretched
                                        (blue) vs shortened (red) -- NOT a stress, kept on its
                                        own diverging scale, the sign that separates an
                                        antagonist being dragged from a muscle merely sheared
    f) muscle routing, side view    -- g) muscle routing, top view -- the skeleton (bone
                                        points, static), a translucent globe, and the six fibre
                                        centrelines RETRACED LIVE from mus_pos each frame (not
                                        the rest-pose PNG), so routing reads against the same
                                        clock as a-e. Camera axes come off the bone cloud's own
                                        extent (whichever of X/Z it is most elongated along is
                                        "head-tail"; PCA's fitted direction was tried and
                                        dropped -- close enough for a side view, visibly tilted
                                        looking straight down it). The OTHER eye (this run only
                                        ever drives one side) is drawn too, static, dimmer, and
                                        offset to the skull's own mirror-symmetric position --
                                        not its true position, which nothing here recovers
                                        cheaply, but not overlapping the live one either.

b, c, d need `mus_axial_p` / `mus_trans_p` in the capture (`run_eye.capture_run`, unconditional
since this session -- NOT the same as `--instrument`'s `mus_axial`/`mus_transverse`, which are
six PER-MUSCLE numbers for the mechanism-discrimination report, not a per-particle field a mesh
can be coloured by). A `*_curves.npz` from before that change will KeyError here; re-run it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv
from tqdm import tqdm

import eye_anatomy as EA
import blend_mpm_ops as BM
from render_eye import PALETTE, MUS_RGB
from render_surface_vtk import Skin, _poly, K_BIND, GLOBE_ALPHA
from render_orbit_vtk import azimuth_schedule, gaze_marker

DIVERGING = "coolwarm"           # signed fields (axial stress, length%): blue-white-red
SEQUENTIAL = "inferno"           # unsigned fields (transverse stress, displacement)
N_TRACE_BINS = 14                # matches blend_mpm_ops's own `self.bins` (muscle_geometry)


def _clim(a, sym=False):
    lo, hi = np.percentile(a, [1.0, 99.0])
    if sym:
        m = max(abs(lo), abs(hi), 1e-6)
        return -m, m
    return float(lo), float(max(hi, lo + 1e-6))


def _traced_centreline(pos, s, n_bins=N_TRACE_BINS):
    """Bin-centroid polyline through `pos`, ordered by fibre coordinate `s` -- the LIVE
    analogue of `read_blend.figure`'s `__centreline` trace, retraced from the sim each frame
    instead of read once from the rest-pose mesh."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        if m.any():
            pts.append(pos[m].mean(axis=0))
    return np.asarray(pts, dtype=float)


class StressGridScene:
    """a) plant (big) | b) axial  c) displacement / d) transverse  e) length% | f,g) routing."""

    def __init__(self, cap, side="R", blend=None, parts=None, size=(2000, 1500),
                 globe_alpha=GLOBE_ALPHA, mesh_alpha=0.72, inflate=1.0):
        self.cap = cap
        self.tissue = np.asarray(cap["tissue"])
        self.mus_parent = np.asarray(cap["mus_parent"])
        self.mus_s = np.asarray(cap["mus_s"])
        self.centre0 = np.asarray(cap["centre"][0], float)

        if "mus_axial_p" not in cap:
            raise KeyError("this capture has no mus_axial_p/mus_trans_p -- it predates the "
                           "per-particle fibre-stress capture in run_eye.py; re-run it")

        d, man = BM.load_cut(blend or BM.DEFAULT_BLEND, parts or BM.DEFAULT_PARTS)
        self.d, self.man = d, man
        fr = BM.BlendFrame(man, d, side, EA.A_EQ, EA.GLOBE_CENTER, inflate)
        self.fr = fr
        self.side = side
        shell0 = np.asarray(cap["shell"][0], float)
        mus0 = np.asarray(cap["mus_pos"][0], float)
        self.shell0, self.mus0 = shell0, mus0

        pv.OFF_SCREEN = True
        self.p = pv.Plotter(off_screen=True, window_size=size, border=False,
                            shape=(3, 4), groups=[(np.s_[0:2], np.s_[0:2]),
                                                  (2, np.s_[0:2]), (2, np.s_[2:4])])

        # b, c, d are ALL "stress" (per feedback: one LUT, one scale, so a colour means the
        # same magnitude in any of the three) -- axial trades its sign for that; the sign
        # (tension vs compression) still lives in the raw mus_axial_p capture, just not drawn
        # here anymore. length% keeps its own diverging scale: it isn't a stress at all.
        self.clim_stress = _clim(np.concatenate([
            np.asarray(cap["mus_vm"]).ravel(),
            np.abs(np.asarray(cap["mus_axial_p"])).ravel(),
            np.asarray(cap["mus_trans_p"]).ravel(),
        ]))
        shorten_all = 100.0 * (1.0 - np.asarray(cap["length"]) / np.asarray(cap["rest_length"]))
        self.clim_len = _clim(shorten_all, sym=True)

        # panel key -> (mode, title); mode picks which scalar/colouring frame() writes
        self.panels = {
            (0, 0): ("plant", "a)  the plant, as driven"),
            (0, 2): ("vm", "b)  total stress"),
            (0, 3): ("axial", "c)  axial stress"),
            (1, 2): ("trans", "d)  transverse stress"),
            (1, 3): ("length", "e)  length % (signed)"),
        }
        self.globe = {}      # panel -> [(mesh, skin), ...]
        self.muscles = {}    # panel -> [(mi, mesh, skin, own), ...]
        for panel, (mode, title) in self.panels.items():
            self.p.subplot(*panel)
            self.p.set_background("black")
            self.p.enable_depth_peeling(10)
            self._build_globe(panel, mode, globe_alpha)
            self._build_muscles(panel, mode, mesh_alpha if mode != "plant" else 1.0)
            self.p.add_text(title, position="upper_left", font_size=9, color="white")

        self._build_routing((2, 0), "f)  muscle routing -- side view", kind="lateral")
        self._build_routing((2, 2), "g)  muscle routing -- top view", kind="top")

        all0 = np.concatenate([shell0, mus0])
        reach = float(np.abs(all0 - self.centre0[None, :]).max())
        self.span = 1.15 * reach
        self.gaze_sel = gaze_marker(self.tissue)
        self.arrow_len = 1.1 * float(np.abs(shell0 - self.centre0[None, :]).max())
        for panel in self.panels:
            self.p.subplot(*panel)
            t0, d0 = self._gaze(0)
            if d0 is not None:
                self.p.add_mesh(pv.Arrow(start=t0, direction=d0, tip_length=0.26,
                                         tip_radius=0.075, shaft_radius=0.024,
                                         scale=self.arrow_len),
                                color="#7a7a7a", opacity=0.5, name="gaze_rest")
        self._text("")

    # --- panels a-e: skinned globe + muscle meshes, one Skin instance each -----------
    def _build_globe(self, panel, mode, globe_alpha):
        if mode not in ("plant", "vm", "axial", "trans", "length"):
            return
        lst = self.globe.setdefault(panel, [])
        for part, alpha, spec in (("retina", globe_alpha, 0.30), ("cornea", 0.26, 0.65),
                                  ("lens", 0.85, 0.85)):
            key = f"{self.side}_{part}"
            if f"{key}__v" not in self.d:
                continue
            V = self.fr.globe(self.d[f"{key}__v"])
            mesh = _poly(V, self.d[f"{key}__f"])
            skin = Skin(V, self.shell0)
            mesh["rgb"] = np.clip(PALETTE[skin.nearest(self.tissue)], 0, 1).astype(np.float32)
            self.p.add_mesh(mesh, scalars="rgb", rgb=True, opacity=alpha, smooth_shading=True,
                            specular=spec, specular_power=24, show_scalar_bar=False)
            lst.append((mesh, skin))

    def _build_muscles(self, panel, mode, opacity):
        lst = self.muscles.setdefault(panel, [])
        for mi, key in enumerate(EA.MUSCLE_KEYS):
            name = f"{self.side}_{key}"
            if f"{name}__v" not in self.d:
                continue
            own = self.mus_parent == mi
            if own.sum() < K_BIND:
                continue
            V = self.fr(self.d[f"{name}__v"])
            mesh = _poly(V, self.d[f"{name}__f"])
            skin = Skin(V, self.mus0[own])
            if mode == "plant":
                mesh["rgb"] = np.tile(np.array(_hex_rgb(EA.MUSCLES[mi]["color"]), np.float32),
                                      (mesh.n_points, 1))
                self.p.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True,
                                specular=0.35, specular_power=22, show_scalar_bar=False,
                                name=f"mus_{panel}_{mi}")
            else:
                cmap, clim = {"vm": (SEQUENTIAL, self.clim_stress),
                             "axial": (SEQUENTIAL, self.clim_stress),
                             "trans": (SEQUENTIAL, self.clim_stress),
                             "length": (DIVERGING, self.clim_len)}[mode]
                mesh["v"] = np.zeros(mesh.n_points, np.float32)
                self.p.add_mesh(mesh, scalars="v", cmap=cmap, clim=clim, opacity=opacity,
                                smooth_shading=True, specular=0.2, show_scalar_bar=False,
                                name=f"mus_{panel}_{mi}")
            lst.append((mi, mesh, skin, own))

    def _body_axis(self, bone):
        """The head-tail direction: SNAPPED to whichever world axis (X or Z; never Y, that's
        up everywhere else in this file) the bone cloud is most elongated along. PCA's raw
        direction was tried first and rejected -- for this cut it read [0.97, 0, -0.24], a
        genuine ~14deg tilt from pure X caused by asymmetric bone mass (more material on one
        side), not a real tilt of the fish. That's invisible in the LATERAL view (an error in
        VIEW_UP a human barely notices) but glaring in the TOP view (the same error in the
        VIEWING DIRECTION itself, slicing the head at an angle). The extent ratio here is
        stark enough (X five to ten times Z, typically) that snapping beats fitting."""
        if len(bone) > 10:
            extent = bone.max(axis=0) - bone.min(axis=0)
            extent[1] = -1.0                    # never pick Y
            axis = np.zeros(3)
            axis[int(np.argmax(extent))] = 1.0
        else:
            axis = np.array([0.0, 0.0, 1.0])
        return axis

    # --- panels f, g: raw bone points + a translucent globe + live-traced centrelines,
    # PLUS a static mirror of the untouched eye (this run only ever drives one side) -------
    def _build_routing(self, panel, title, kind):
        self.p.subplot(*panel)
        self.p.set_background("black")
        bone_pts = []
        for rec in self.man["parts"]:
            if rec["group"] != "bone":
                continue
            key = rec["part"]
            if f"{key}__v" in self.d:
                bone_pts.append(self.fr(self.d[f"{key}__v"]))
        bone = np.concatenate(bone_pts, axis=0) if bone_pts else np.zeros((0, 3))
        if len(bone):
            cloud = pv.PolyData(bone)
            self.p.add_mesh(cloud, color="#8e8494", opacity=0.35, point_size=2.0,
                            render_points_as_spheres=True, show_scalar_bar=False)

        c = self.centre0
        up = np.array([0.0, 1.0, 0.0])
        eye_reach = float(np.abs(np.concatenate([self.shell0, self.mus0]) - c[None, :]).max())
        # PCA over the WHOLE skeleton is noisy -- ribs/vertebrae far down the body have their
        # own spread and can tilt the estimate. Restrict to bone within a few eye-reaches of
        # the orbit, which is dominated by the (elongated, head-tail) skull, not the trunk.
        near = (np.linalg.norm(bone - c[None, :], axis=1) < 4.0 * eye_reach) if len(bone) \
            else np.zeros(0, bool)
        body = self._body_axis(bone[near] if near.any() else bone)
        lateral = np.cross(body, up)
        lateral /= np.linalg.norm(lateral).clip(1e-9)

        # the OTHER eye was drawn here too at one point -- a true mirror reflection of the
        # simulated one across the fish's own sagittal plane, geometrically correct, but
        # dropped on request: one eye, read cleanly, beat two eyes at a still-approximate
        # relative scale.
        for part, alpha in (("retina", GLOBE_ALPHA), ("cornea", 0.20), ("lens", 0.25)):
            key = f"{self.side}_{part}"
            if f"{key}__v" not in self.d:
                continue
            V = self.fr.globe(self.d[f"{key}__v"])
            mesh = _poly(V, self.d[f"{key}__f"])
            self.p.add_mesh(mesh, color="#cfcfd6", opacity=alpha, smooth_shading=True,
                            specular=0.3, specular_power=20, show_scalar_bar=False)
            self.globe.setdefault(panel, []).append((mesh, Skin(V, self.shell0)))

        self._routing_lines = getattr(self, "_routing_lines", {})
        lines = {}
        for mi, key in enumerate(EA.MUSCLE_KEYS):
            own = self.mus_parent == mi
            if own.sum() < 3:
                continue
            pts0 = _traced_centreline(self.mus0[own], self.mus_s[own])
            if len(pts0) < 2:
                continue
            line = pv.lines_from_points(pts0)
            self.p.add_mesh(line, color=EA.MUSCLES[mi]["color"], line_width=4,
                            show_scalar_bar=False, name=f"trace_{panel}_{mi}")
            self.p.add_mesh(pv.PolyData(pts0[:1]), color=EA.MUSCLES[mi]["color"],
                            point_size=10, render_points_as_spheres=True,
                            name=f"tip_{panel}_{mi}")
            lines[mi] = line
        self._routing_lines[panel] = lines

        if kind == "lateral":
            dirv, view_up = lateral, up            # from the flank, body axis horizontal
        else:
            dirv = up                               # straight down
            # mirrored L/R from the lateral view's own, rolled 15deg (picked by eye off a
            # 3x3 montage of candidate angles -- see git history of _montage_g.py)
            roll = np.radians(15.0)
            lr = -lateral
            view_up = (lr * np.cos(roll) + np.cross(dirv, lr) * np.sin(roll)
                      + dirv * np.dot(dirv, lr) * (1.0 - np.cos(roll)))
        # framed on muscle+eye extent, same as `read_blend.figure`'s own `orbit` box -- bone is
        # drawn but does not set the zoom, or the skeleton (reaching well past the orbit, down
        # the whole body) shrinks the eye to a speck.
        span = 1.3 * eye_reach
        self.p.camera_position = (tuple(c + dirv * 10.0), tuple(c), tuple(view_up))
        self.p.camera.parallel_projection = True
        self.p.camera.parallel_scale = span
        self.p.add_text(title, position="upper_left", font_size=9, color="white")
        self._routing_cam = getattr(self, "_routing_cam", {})
        self._routing_cam[panel] = (c, dirv, view_up, span)

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
        self.p.add_text(hud, position=(0.02, 0.85), font_size=9, color="white", name="hud",
                        viewport=True)

    def camera(self, az_deg, el_deg=18.0):
        c = self.centre0
        a, e = np.radians(az_deg), np.radians(el_deg)
        dv = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        for panel in self.panels:
            self.p.subplot(*panel)
            self.p.camera_position = (tuple(c + dv * 10.0), tuple(c), (0.0, 1.0, 0.0))
            self.p.camera.parallel_projection = True
            self.p.camera.parallel_scale = self.span

    def frame(self, k, az_deg, dt):
        cap = self.cap
        X = np.asarray(cap["shell"][k], float)
        for panel in self.globe:                       # a-e AND the f, g routing panels
            for mesh, skin in self.globe[panel]:
                mesh.points = skin(X)
        Y = np.asarray(cap["mus_pos"][k], float)
        vm = np.asarray(cap["mus_vm"][k], float)
        ax = np.asarray(cap["mus_axial_p"][k], float)
        tr = np.asarray(cap["mus_trans_p"][k], float)
        length = np.asarray(cap["length"][k], float)
        rest_l = np.asarray(cap["rest_length"], float)
        shorten = 100.0 * (1.0 - length / rest_l)
        act = np.asarray(cap["act"][k], float)
        for panel, (mode, _) in self.panels.items():
            for mi, mesh, skin, own in self.muscles.get(panel, []):
                mesh.points = skin(Y[own])
                if mode == "plant":
                    base = MUS_RGB[mi]
                    lit = np.clip(base * (0.78 + 0.42 * float(np.clip(act[mi], 0, 1))), 0, 1)
                    mesh["rgb"] = np.tile(lit.astype(np.float32), (mesh.n_points, 1))
                elif mode == "vm":
                    mesh["v"] = skin.scalar(vm[own])
                elif mode == "axial":
                    mesh["v"] = skin.scalar(np.abs(ax[own]))
                elif mode == "trans":
                    mesh["v"] = skin.scalar(tr[own])
                elif mode == "length":
                    mesh["v"] = np.full(mesh.n_points, shorten[mi], np.float32)
        for panel, lines in self._routing_lines.items():
            for mi, line in lines.items():
                own = self.mus_parent == mi
                pts = _traced_centreline(Y[own], self.mus_s[own])
                if len(pts) == line.n_points:
                    line.points = pts
        h, v, t = np.asarray(cap["gaze"][k], float)
        fr_i = int(cap["frame"][k])
        len_line = "  ".join(f"{key} {s:+4.1f}%" for key, s in zip(EA.MUSCLE_KEYS, shorten))
        self._text(f"frame {fr_i:4d}   t = {fr_i * dt:5.2f} s   "
                  f"gaze h {h:+5.1f}  v {v:+5.1f}  t {t:+5.1f}\n"
                  f"muscle length, % shorter than rest (- = stretched)   {len_line}")
        self.camera(az_deg)
        for panel, (c, dirv, view_up, span) in self._routing_cam.items():
            self.p.subplot(*panel)
            self.p.camera_position = (tuple(c + dirv * 10.0), tuple(c), tuple(view_up))
            self.p.camera.parallel_projection = True
            self.p.camera.parallel_scale = span
        return self.p.screenshot(return_img=True)

    def close(self):
        self.p.close()


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def render(cap, dt, out_mp4, out_strip=None, fps=30, size=(2000, 1500), turns=1.0,
           quality=8, strip_n=5, still_margin=0.03, still_above=None, az0=0.0,
           side="R", blend=None, parts=None):
    import imageio.v2 as iio

    n = len(cap["frame"])
    scene = StressGridScene(cap, side=side, blend=blend, parts=parts, size=size)
    az, moving = azimuth_schedule(cap, turns=turns, still_margin=still_margin,
                                  still_above=still_above, az0=az0)
    print(f"[stress-grid] camera turns on {int((~moving).sum())} of {n} frames; "
          f"clim stress (b,c,d, shared)={scene.clim_stress}  length%={scene.clim_len}",
          flush=True)
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
    ap.add_argument("--curves", required=True, help="a *_curves.npz WITH mus_axial_p/mus_trans_p")
    ap.add_argument("--out", default=None, help="defaults next to --curves")
    ap.add_argument("--side", default="R", choices=("L", "R"))
    ap.add_argument("--blend", default=None)
    ap.add_argument("--parts", default=None)
    ap.add_argument("--turns", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=0.003)
    args = ap.parse_args()

    cap = {k: v for k, v in np.load(args.curves).items()}
    stem = args.out or args.curves.replace("_curves.npz", "_stress_grid")
    mp4 = stem + ".mp4"
    render(cap, args.dt, mp4, stem + ".png", turns=args.turns, az0=args.az, side=args.side,
          blend=args.blend, parts=args.parts)
    print(f"[stress-grid] {mp4}")


if __name__ == "__main__":
    main()
