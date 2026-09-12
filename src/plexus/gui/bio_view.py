"""The bio page's eye is the movie renderer.

One `View` per session: the spec built and seeded exactly as a run would (`engine.build`,
`engine.seed`), then `plexus.live_movie.LiveMovie` draws frame 0 off-screen -- the same mesh,
the same sphere glyphs of world radius (`plotting.dot_radius`), the same palette a generated
movie has. The page shows that plotter's screenshot and sends camera moves, picks and
visibility toggles here; nothing is drawn twice in two codebases, so what the page shows is what
the movie will show.

Picking is a projection, not a VTK picker: every live object (a cluster, a piece, a cell
centroid, a vertex) is projected through the plotter's own camera to the screen and the nearest
to the click within a few pixels wins, nearer-to-camera on a tie. That works for glyph actors
(whose point ids are the sphere's, not the piece's) and costs one matrix product.

VTK is not thread-safe and its off-screen OpenGL context belongs to the thread that made it; the
server handles each request on its own thread, and a second view built on a different thread
than the first died in VTK with `std::bad_array_new_length`. So EVERY call that touches the
plotter runs on one dedicated thread (`_vtk`), whichever request thread asks.
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import time

import numpy as np

from concurrent.futures import ThreadPoolExecutor

LOCK = threading.RLock()
CURRENT: dict = {"view": None}
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plexus-vtk")
_VTK_THREAD: dict = {"ident": None}
# REQUESTS FOR THE VTK THREAD WAIT HERE, NOT ONLY IN THE EXECUTOR. While a run occupies the VTK
# thread (the pipeline is called ON it, see `View.run`, because a second off-screen plotter on a
# second thread dies inside VTK), the executor's own queue is stuck behind the run; the run's
# per-frame hook drains THIS queue instead, so a camera move from the page is answered between two
# frames rather than after the last one. Outside a run the executor drains it as before.
import queue as _queue
_PENDING: "_queue.Queue" = _queue.Queue()


def _drain() -> int:
    """Answer every waiting request, on the VTK thread. Returns how many were served."""
    n = 0
    while True:
        try:
            fn, args, kwargs, fut = _PENDING.get_nowait()
        except _queue.Empty:
            return n
        if fut.set_running_or_notify_cancel():
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as e:                               # noqa: BLE001
                fut.set_exception(e)
        n += 1


def _vtk(fn, *args, **kwargs):
    """Run `fn` on the one VTK thread and return its result (a call from that thread runs inline)."""
    if _VTK_THREAD["ident"] == threading.get_ident():
        return fn(*args, **kwargs)
    from concurrent.futures import Future
    fut: Future = Future()
    _PENDING.put((fn, args, kwargs, fut))

    def _serve():
        _VTK_THREAD["ident"] = threading.get_ident()
        _drain()
    _EXEC.submit(_serve)
    return fut.result()


class View:
    def __init__(self, spec_path: str, device: str = "cpu", px: int = 1000):
        from plexus import engine, schema
        from plexus.gui import bio
        from plexus.live_movie import LiveMovie
        t0 = time.time()
        sim = schema.load(spec_path)
        H = engine.build(sim, device)
        engine.seed(H, sim, device)
        self.sim, self.H, self.spec_path = sim, H, spec_path
        self.scene = bio.scene_from(H, sim, spec_path)
        style = dict(sim.plotting or {})
        for k in ("curve", "stills"):                       # the curves need a clip; the section inset stays
            style.pop(k, None)
        style["real_time"] = False
        # a walled box IS the scene (the balls bounce on its floor): keep its frame; a free tissue has
        # no wall, and its box was only a camera hint that shrank the cyst to a tenth of the picture
        style["box_frame"] = str(getattr(sim, "boundary", "") or "").lower() == "wall"
        style["scale_bar"] = False                               # the movie's bar sits at the box edge; ours follows the view
        style["cross_section_height"] = min(float(style.get("cross_section_height", 0.24) or 0.24), 0.2)
        u = getattr(sim, "units", None)
        dec = bool(getattr(u, "declared", False))                # the scale bar needs declared units, as the movie does
        self._tmp = tempfile.mkdtemp(prefix="plexus_bio_")
        free = str(getattr(sim, "boundary", "") or "").lower() == "free"
        # THE CIRCUIT PANEL IS NOT A CAMERA PICTURE. `plotting.renderer: neural_panel` draws the
        # message on the connectivity matrix, the vectors and the kinograph (plexus/neural_panel.py);
        # the page shows that figure, orbit and pick do nothing, PLAY redraws any captured frame.
        self.panel = None
        if str(style.get("renderer", "") or "").lower() == "neural_panel":
            from plexus.neural_panel import NeuralPanel
            self.panel = NeuralPanel(out=os.path.join(self._tmp, "view.mp4"), n_frames=max(1, int(sim.n_frames)),
                                     sim=sim, style=style, name=sim.name, stills=0, keep_stills=False,
                                     dt=getattr(sim, "dt", None), time_s=(float(u.time_s) if dec else None))
            self.panel.capture(H, 0)
            self.lm = None; self.p = None
            self.focal = np.zeros(3); self.dist0 = 1.0; self.scale0 = 1.0
            self.azim, self.elev, self.zoom = 0.0, 0.0, 1.0
            self.up_axis = 2
            self.pick = None
            self.hidden: set = set()
            self.RUN = {"running": False, "frame": 0, "n_frames": 0, "seconds": 0.0, "error": None, "stop": False, "counts": {}, "frames_kept": 0}
            self.snaps: list = []
            self.frame_shown = None
            self.seconds = round(time.time() - t0, 2)
            return
        self.lm = LiveMovie(out=os.path.join(self._tmp, "view.mp4"), world=list(sim.world_size), n_frames=1,
                            up=int(style.get("up_axis", 2)), name=sim.name, sim=sim, style=style,
                            centred=free and not any(k in (sim.sets or {}) for k in ("mpm_particle", "cytosol", "nucleus")),
                            can_curve=False, render_n=400_000, stills=0, px=int(px),
                            dt=getattr(sim, "dt", None),
                            time_s=(float(u.time_s) if dec else None),
                            length_um=(float(u.length_um) if dec else None))
        self.lm(H, 0)                                            # builds every actor, writes one frame
        if self.lm.failed:
            raise RuntimeError(f"renderer: {self.lm.failed}")
        if getattr(self.lm, "cs", None) is not None:             # the first frame builds the inset but fills it from frame 1
            self.lm._update_cross_section(H)
        self.p = self.lm.p
        cam = self.p.camera
        # FRAME THE OBJECTS, NOT THE BOX. The movie frames the world box (a 50-unit box around a
        # 5-unit cyst leaves the cyst a tenth of the frame); the page is about the objects, so the
        # camera looks at their centre from a distance where their bounding sphere fills the view.
        self.focal = np.asarray(cam.focal_point, float)
        self.dist0 = float(np.linalg.norm(np.asarray(cam.position, float) - self.focal))
        P = np.concatenate([np.asarray(st["pos"], float) for st in self.scene["sets"].values() if st.get("pos")], 0) \
            if any(st.get("pos") for st in self.scene["sets"].values()) else None
        if P is not None and len(P):
            P = P[np.abs(P).max(1) < 1e5]                        # parked slots sit at -1e6
            if P.shape[1] == 2:                                  # a 2-D world sits in the z = 0 plane
                P = np.concatenate([P, np.zeros((len(P), 1))], 1)
            lo, hi = P.min(0), P.max(0)
            self.focal = 0.5 * (lo + hi) + np.asarray(getattr(self.lm, "_shift", 0.0) or 0.0, float)
            R = 0.5 * float(np.linalg.norm(hi - lo))
            self.dist0 = 1.15 * R / np.tan(np.radians(float(cam.view_angle) / 2.0)) if R > 0 else self.dist0
            self.scale0 = 1.15 * R if R > 0 else float(cam.parallel_scale)
        else:
            self.scale0 = float(cam.parallel_scale)
        self.azim, self.elev, self.zoom = 30.0, 20.0, 1.0
        self.up_axis = int(style.get("up_axis", 2))              # a material box is y-up, a tissue z-up
        if int(getattr(sim, "dim", 3)) == 2:
            # A 2-D WORLD IS LOOKED AT FROM +z WITH y UP: the orbit's up axis is y and the camera
            # starts on the +z side (azim 90 around y at elevation 0), so the plane fills the view.
            self.up_axis = 1
            self.azim, self.elev = 90.0, 0.0
        self.pick = None
        self.hidden: set = set()
        self.RUN = {"running": False, "frame": 0, "n_frames": 0, "seconds": 0.0, "error": None, "stop": False, "counts": {}, "frames_kept": 0}
        self.snaps: list = []                                    # the run's frames as level states, for PLAY at any camera
        self.seconds = round(time.time() - t0, 2)
        self.set_camera(self.azim, self.elev, self.zoom)

    # ------------------------------------------------------------------ camera and picture
    def set_camera(self, azim=None, elev=None, zoom=None):
        return _vtk(self._set_camera, azim, elev, zoom)

    def _set_camera(self, azim=None, elev=None, zoom=None):
        if self.panel is not None:
            return
        with LOCK:
            if azim is not None: self.azim = float(azim)
            if elev is not None: self.elev = max(-89.0, min(89.0, float(elev)))
            if zoom is not None: self.zoom = max(0.05, min(60.0, float(zoom)))
            a, e = np.radians(self.azim), np.radians(self.elev)
            # the orbit is about the spec's `up_axis`: elevation climbs along it, azimuth turns around it
            up = self.up_axis
            h = [i for i in range(3) if i != up]
            d = np.zeros(3); d[h[0]] = np.cos(e) * np.cos(a); d[h[1]] = np.cos(e) * np.sin(a); d[up] = np.sin(e)
            U = np.zeros(3); U[up] = 1.0
            cam = self.p.camera
            cam.focal_point = tuple(self.focal)
            # under a parallel projection the distance does not change the picture, so the camera
            # stays far away and only the parallel scale zooms: the near plane never cuts the scene
            par = bool(cam.parallel_projection)
            cam.position = tuple(self.focal + d * (self.dist0 if par else self.dist0 / self.zoom))
            cam.up = tuple(U)
            # THE MOVIE'S CAMERA IS ORTHOGRAPHIC (`parallel_projection`, live_movie.py:547), so
            # zoom is the parallel scale (half the view height in world units), not the distance.
            if bool(cam.parallel_projection):
                cam.parallel_scale = self.scale0 / self.zoom
            self.p.renderer.ResetCameraClippingRange()
            self._scale_bar(cam, d)
            self._section_zoom()

    def _section_zoom(self):
        """The section inset follows the view's zoom: its window is the declared span over the zoom,
        centred where it was, and the section is redrawn (discs re-sized) when the zoom changed."""
        lm = self.lm
        if getattr(lm, "cs", None) is None or getattr(lm, "_cs_rng", None) is None:
            return
        if not hasattr(self, "_cs0"):
            (x0, x1), (y0, y1) = lm._cs_rng
            self._cs0 = (0.5 * (x0 + x1), 0.5 * (y0 + y1), float(x1 - x0), float(y1 - y0))
            self._cs_zoom = 1.0
        if abs(self.zoom - self._cs_zoom) < 1e-9:
            return
        cx, cy, sx, sy = self._cs0
        lm._cs_rng = ([cx - 0.5 * sx / self.zoom, cx + 0.5 * sx / self.zoom],
                      [cy - 0.5 * sy / self.zoom, cy + 0.5 * sy / self.zoom])
        self._cs_zoom = self.zoom
        lm._update_cross_section(self.H)

    def _scale_bar(self, cam, view_dir):
        """A bar of a round length, fixed at the bottom-left of the VIEW whatever the camera does.

        The movie's bar is a line at the world box's edge, sized to a third of the box; the page
        frames the objects and zooms, so that bar is off-screen or the wrong length here. This one
        is a world-space line rebuilt on every camera move: it lies in the screen plane at the
        bottom of the view, its length the largest 1-2-2.5-5 x 10^n metres under a third of the
        view width, labelled in SI. Needs `general.units` (length_um), as the movie's bar does."""
        Lum = getattr(self.lm, "length_um", None)
        if not Lum:
            return
        from plexus.live_movie import _si_length
        import pyvista as pv
        up = np.zeros(3); up[self.up_axis] = 1.0
        right = np.cross(up, view_dir); right /= max(np.linalg.norm(right), 1e-12)
        vup = np.cross(view_dir, right); vup /= max(np.linalg.norm(vup), 1e-12)
        half_h = float(cam.parallel_scale) if cam.parallel_projection else float(self.dist0 / self.zoom) * np.tan(np.radians(float(cam.view_angle) / 2))
        W, Hh = self.p.window_size
        half_w = half_h * float(W) / float(Hh)
        m_per_unit = float(Lum) / 1.0e6
        tgt_m = (2.0 * half_w / 3.0) * m_per_unit
        p10 = 10.0 ** np.floor(np.log10(max(tgt_m, 1e-30)))
        len_m = max([f * p10 for f in (1.0, 2.0, 2.5, 5.0) if f * p10 <= tgt_m] or [p10])
        L = len_m / m_per_unit
        # bottom-RIGHT: the section inset owns the bottom-left corner
        # just in front of the camera, so nothing in the scene can occlude it at any zoom
        near = 0.5 * (self.dist0 if cam.parallel_projection else self.dist0 / self.zoom)
        b = self.focal + right * (half_w * 0.92) - vup * (half_h * 0.9) + view_dir * near
        a = b - right * L
        for nm in ("scale_bar", "scale_label"):
            try:
                self.p.remove_actor(nm, render=False)
            except Exception:                                    # noqa: BLE001
                pass
        self.p.add_mesh(pv.Line(a, b), color="white", line_width=4.0, lighting=False, name="scale_bar")
        self.p.add_text(_si_length(len_m), position=(0.80, 0.065), viewport=True, font_size=11, color="white", name="scale_label")

    def png(self) -> bytes:
        if self.panel is not None:
            def _panel_png():
                with LOCK:
                    i = self.frame_shown if self.frame_shown is not None else len(self.panel.hist) - 1
                    return self.panel.frame_at(i)
            img = _vtk(_panel_png)
            import imageio.v3 as iio
            buf = io.BytesIO()
            iio.imwrite(buf, np.asarray(img), extension=".png")
            return buf.getvalue()
        img = _vtk(self._grab)
        import imageio.v3 as iio
        buf = io.BytesIO()
        iio.imwrite(buf, np.asarray(img), extension=".png")
        return buf.getvalue()

    SNAPS_MAX = 300                                              # frames kept per run by default: live_movie's max_frames
    LIVE_PICS = 20                                               # pictures shown while the run goes
    SNAP_BUDGET = 1_500_000_000                                  # bytes of kept positions per run
    SCENE_INTERVAL = 3.0                                         # seconds between pick-scene rebuilds

    def _snapshot(self, H):
        """What the renderer reads, per level, copied to the CPU: the position (and separation)
        columns, occupancy, type, parent, and the half-edge tables of a mesh level. Enough to put
        the picture back at that frame later from ANY camera; not the whole state."""
        import torch
        snap = {}
        for name, lv in H.levels.items():
            try:
                sch = lv.state_schema
            except Exception:                                    # noqa: BLE001
                continue
            if "pos" not in sch:
                continue
            e = {}
            # `vel` AND THE SOLVER BUFFERS TOO (F, C, Jp), so a frame replayed under a field colour
            # (`speed`, `deformation`, `pressure`) is that frame's field and not the last one's.
            for key in ("pos", "sep", "vel"):
                if key in sch:
                    a, b = sch[key]
                    e[key] = lv.state[:, a:b].detach().cpu().clone()
            for key in ("occ", "node_type", "parent", "F", "C", "Jp"):
                v = getattr(lv, key, None)
                if v is not None and torch.is_tensor(v):
                    e[key] = v.detach().cpu().clone()
            m = getattr(lv, "_mesh", None)
            if m:
                e["mesh"] = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in m.items()}
            snap[name] = e
        self.snaps.append(snap)
        self.RUN["frames_kept"] = len(self.snaps)

    def show_frame(self, i: int):
        """Put the picture at kept frame `i` (on the VTK thread): the levels of the view's own
        hierarchy take the snapshot's columns, then the movie renderer redraws them as it would
        have during the run. The camera is whatever the page set, so a replay can be orbited."""
        return _vtk(self._show_frame, i)

    def _show_frame(self, i: int):
        if self.panel is not None:
            self.frame_shown = max(0, min(int(i), len(self.panel.hist) - 1)) if self.panel.hist else None
            return
        if not self.snaps:
            return
        i = max(0, min(int(i), len(self.snaps) - 1))
        snap = self.snaps[i]
        with LOCK:
            H = self.H
            for name, e in snap.items():
                if name not in H.levels:
                    continue
                lv = H.level(name)
                dev = lv.state.device
                for key in ("pos", "sep", "vel"):
                    if key in e and key in lv.state_schema:
                        a, b = lv.state_schema[key]
                        lv.state[:, a:b] = e[key].to(dev)
                for key in ("occ", "node_type", "parent", "F", "C", "Jp"):
                    v = getattr(lv, key, None)
                    if key in e and v is not None and v.shape == e[key].shape:
                        v.copy_(e[key].to(dev))
                if "mesh" in e and getattr(lv, "_mesh", None) is not None:
                    for k, v in e["mesh"].items():
                        lv._mesh[k] = v.to(dev) if hasattr(v, "to") else v
            self.lm(H, i * self._keep_every)
            if getattr(self.lm, "cs", None) is not None:
                self.lm._update_cross_section(H)
            self.frame_shown = i

    def _grab(self):
        with LOCK:
            self.p.render()                                      # screenshot() alone returns the stale frame
            return np.asarray(self.p.screenshot(return_img=True)).copy()

    # ------------------------------------------------------------------ picking by projection
    def _candidates(self):
        """(positions [n,3], ids ['set:idx', ...]) of everything a click may mean."""
        P, ids = [], []
        T = self.scene.get("tissue")
        for name, st in self.scene["sets"].items():
            if not st.get("pos"):
                continue
            if T and name == T["set"]:
                V = np.asarray(T["caps"]["mid"]["verts"], float)
                P.append(V[: T["Nv"]]); ids += [f"{name}:{i}" for i in range(T["Nv"])]
                P.append(V[T["Nv"]:]); ids += [f"cell:{f}" for f in range(len(V) - T["Nv"])]
                continue
            sp_names = st.get("type_names") or []
            nt = st.get("node_type") or [0] * len(st["pos"])
            keep = [i for i in range(len(st["pos"])) if not sp_names or sp_names[nt[i]] not in self.hidden]
            if keep:
                P.append(np.asarray(st["pos"], float)[keep]); ids += [f"{name}:{i}" for i in keep]
        if not P:
            return np.zeros((0, 3)), []
        return np.concatenate(P, 0), ids

    def pick_at(self, fx: float, fy: float, tol: float = 0.012) -> str | None:
        """The object under the click at screen fractions (fx from the left, fy from the top)."""
        if self.panel is not None:
            return None
        def _proj():
            W, Hh = self.p.window_size
            M = self.p.camera.GetCompositeProjectionTransformMatrix(float(W) / float(Hh), -1.0, 1.0)
            return W, Hh, np.array([[M.GetElement(i, j) for j in range(4)] for i in range(4)], float)
        W, Hh, A = _vtk(_proj)
        P, ids = self._candidates()
        if not ids:
            return None
        X = np.concatenate([P, np.ones((len(P), 1))], 1) @ A.T
        w = np.where(np.abs(X[:, 3]) < 1e-12, 1e-12, X[:, 3])
        ndc = X[:, :3] / w[:, None]
        sx, sy = (ndc[:, 0] + 1) / 2, (1 - ndc[:, 1]) / 2
        d2 = (sx - fx) ** 2 + ((sy - fy) * Hh / W) ** 2
        ok = (w > 0) & (d2 < tol ** 2)
        if not ok.any():
            return None
        score = np.where(ok, d2 + 0.02 * tol ** 2 * ndc[:, 2], np.inf)     # nearer to the camera on a tie
        return ids[int(np.argmin(score))]

    def highlight(self, pick: str | None):
        return _vtk(self._highlight, pick)

    def _highlight(self, pick: str | None):
        """PAINT THE PICKED BODY YELLOW -- the body itself, not a marker around it: its particles in
        the point cloud (their colours are put back when the pick moves on), or its surface actor
        when the scene is contoured per body."""
        self.pick = pick
        if self.panel is not None or self.lm is None:
            return
        with LOCK:
            lm = self.lm
            # put back what the last pick painted
            _prev = getattr(self, "_paint", None)
            if _prev is not None:
                sel, rgb = _prev
                try:
                    cur = np.asarray(lm.cloud["rgb"]).copy(); cur[sel] = rgb; lm.cloud["rgb"] = cur
                    if getattr(lm, "_base_rgb", None) is not None:
                        lm._base_rgb[sel] = rgb
                except Exception:                                # noqa: BLE001
                    pass
                self._paint = None
            for nm, act in list(getattr(lm.p, "renderer", None).actors.items() if lm.p is not None else []):
                if nm.startswith("contour_") and getattr(self, "_paint_actor", None) == nm:
                    try:
                        act.GetProperty().SetColor(*self._paint_color)
                    except Exception:                            # noqa: BLE001
                        pass
                    self._paint_actor = None
            if not pick or not pick.startswith("cell:"):
                return
            try:
                k = int(pick.split(":", 1)[1])
            except ValueError:
                return
            lvl = H_lvl = None
            try:
                H_lvl = self.H.level(getattr(lm, "_sname", None))
            except Exception:                                    # noqa: BLE001
                H_lvl = None
            if H_lvl is None:
                return
            # the contour per body: the actor is named by the body's type
            names = list(getattr(self.H.level("cell"), "type_names", []) or []) if "cell" in self.H.levels else []
            nt = getattr(self.H.level("cell"), "node_type", None) if "cell" in self.H.levels else None
            if names and nt is not None and k < int(nt.numel()):
                tname = names[int(nt[k])]
                act = lm.p.renderer.actors.get(f"contour_{tname}") if lm.p is not None else None
                if act is not None:
                    self._paint_color = act.GetProperty().GetColor()
                    act.GetProperty().SetColor(1.0, 0.93, 0.2)
                    self._paint_actor = f"contour_{tname}"
                    return
            par = getattr(H_lvl, "parent", None)
            idx = getattr(lm, "idx", None)
            if par is None or idx is None or getattr(lm, "cloud", None) is None:
                return
            sel = (par.detach().cpu().numpy()[np.asarray(idx)] == k)
            if not sel.any():
                return
            try:
                cur = np.asarray(lm.cloud["rgb"]).copy()
                self._paint = (sel, cur[sel].copy())
                cur[sel] = np.array([255, 238, 51], np.uint8)
                lm.cloud["rgb"] = cur
                if getattr(lm, "_base_rgb", None) is not None:
                    lm._base_rgb[sel] = np.array([255, 238, 51], np.uint8)
            except Exception:                                    # noqa: BLE001
                self._paint = None

    def _thickness(self) -> float:
        """The tissue's cell thickness (2|sep| averaged), or 1 without a tissue."""
        T = self.scene.get("tissue")
        if not T:
            return 1.0
        A = np.asarray(T["caps"]["apical"]["verts"][: T["Nv"]], float)
        B = np.asarray(T["caps"]["basal"]["verts"][: T["Nv"]], float)
        h = float(np.linalg.norm(A - B, axis=1).mean())
        return h if h > 0 else 1.0

    # ------------------------------------------------------------------ visibility by species
    def set_visible(self, species: str, on: bool):
        return _vtk(self._set_visible, species, on)

    def _set_visible(self, species: str, on: bool):
        if self.panel is not None:
            return False
        with LOCK:
            acts = [a for n, a in self.p.renderer.actors.items() if n.startswith("glyph_") and n.endswith("_" + species)]
            if not acts:
                return False
            for act in acts:
                act.SetVisibility(bool(on))
            (self.hidden.discard if on else self.hidden.add)(species)
            return True

    def species(self) -> list:
        names = []
        for st in self.scene["sets"].values():
            for n in st.get("type_names") or []:
                if n not in names:
                    names.append(n)
        return names

    # ------------------------------------------------------------------ running the engine
    def run(self, device: str | None = None, **_ignored) -> dict:
        """Generate the spec THROUGH THE PIPELINE -- `plexus.pipeline.generate`, the body of
        `Plexus_Main.py -o generate` -- on the VTK thread, with its per-frame hook feeding this
        view.

        WHY ON THE VTK THREAD. The pipeline builds its own `LiveMovie` (the movie.mp4 and the stills
        in graphs_data/studio/<name>/); VTK gives one off-screen context per thread, and a second
        plotter on a second thread died with `std::bad_array_new_length`. So the run is submitted to
        the one VTK thread, both plotters live there, and the hook drains `_PENDING` -- the page's
        camera moves, picks and screenshots -- between frames. That is what makes orbit and zoom
        work DURING generation: every request is answered on the thread that owns the context, at
        most one frame late.

        What the hook does per tick: counts (cheap), a snapshot every movie stride (so PLAY can
        replay the run at any camera, the movie's own frames), and a redraw of this view's plotter
        whenever a request is waiting or a still is due, so the picture the page asks for is the
        frame being computed."""
        import torch
        if self.RUN.get("running"):
            return {"error": "already running; STOP it first"}
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        n = int(self.sim.n_frames)
        self.RUN.update(running=True, frame=0, n_frames=n, seconds=0.0, error=None, stop=False, device=dev,
                        started=time.time(), frames_kept=0, out_dir=None, stopped=False, ms_per_frame=None)
        self.snaps = []
        if self.panel is not None:
            self.panel.hist = []; self.panel.n_frames = n; self.frame_shown = None
        else:
            # THE OVERLAY'S DENOMINATOR IS THIS RUN'S LENGTH, not the 1 the renderer was built with to draw the seed.
            self.lm.n_frames = n
            self.lm.t0 = time.perf_counter()
        # THE MOVIE'S OWN STRIDES. Kept frames for PLAY are the frames the pipeline's movie keeps
        # (`plotting.max_frames`, live_movie.py:256), widened only by a memory budget on the kept
        # positions; the picture is refreshed at the pipeline's stills (`plotting.stills`) and
        # whenever the page asks.
        pl = self.sim.plotting or {}
        per_frame = 0
        for lv in self.H.levels.values():
            sch = getattr(lv, "state_schema", None)
            if sch is not None and "pos" in sch:
                for key in ("pos", "sep", "vel"):
                    if key in sch:
                        a, b = sch[key]
                        per_frame += int(lv.state.shape[0]) * (b - a) * 4
                for key in ("F", "C", "Jp"):                    # the solver buffers ride along
                    v = getattr(lv, key, None)
                    if v is not None and torch.is_tensor(v):
                        per_frame += int(v.numel()) * 4
        by_count = max(1, -(-n // max(1, int(pl.get("max_frames", 300)))))
        by_mem = max(1, -(-((n + 1) * max(per_frame, 1)) // self.SNAP_BUDGET))
        self._keep_every = max(by_count, by_mem)
        self._draw_every = max(1, -(-n // max(1, int(pl.get("stills", 10)))))
        self.RUN["keep_every"] = self._keep_every
        self.RUN["draw_every"] = self._draw_every
        self._last_scene = 0.0
        from plexus.pipeline import StopRun

        def _on_frame(H, tick):
            if self.RUN.get("stop"):
                raise StopRun()
            now = time.time()
            with LOCK:
                self.H = H
                if self.panel is not None:
                    if tick % self._keep_every == 0 or tick == n:
                        self.panel.capture(H, tick)
                        self.RUN["frames_kept"] = len(self.panel.hist)
                else:
                    due = tick == n or tick % self._draw_every == 0
                    if due or not _PENDING.empty():
                        self.lm(H, tick)
                        if getattr(self.lm, "cs", None) is not None and tick == 0:
                            self.lm._update_cross_section(H)
                    if tick % self._keep_every == 0 or tick == n:
                        self._snapshot(H)
                if tick == n or (now - self._last_scene) >= self.SCENE_INTERVAL:
                    from plexus.gui import bio
                    self.scene = bio.scene_from(H, self.sim, self.spec_path)
                    if self.pick:
                        self._highlight(self.pick)
                    self._last_scene = time.time()
            self.RUN["frame"] = int(tick)
            self.RUN["seconds"] = round(time.time() - self.RUN["started"], 1)
            self.RUN["counts"] = self.counts_live(H)
            _drain()                                                 # the page's camera, between frames

        def _go():
            _VTK_THREAD["ident"] = threading.get_ident()
            from plexus import pipeline
            try:
                r = pipeline.generate(self.spec_path, device=dev, force=True, describe=False, on_frame=_on_frame)
                self.RUN["out_dir"] = r.get("data_dir")
                self.RUN["stopped"] = bool(r.get("stopped"))
                fm = r.get("frame_ms")
                if fm is not None and len(fm) > 1:
                    self.RUN["ms_per_frame"] = float(np.mean(np.asarray(fm)[1:]))
            except Exception as e:                                   # noqa: BLE001
                self.RUN["error"] = f"{type(e).__name__}: {e}"[:600]
            finally:
                self.RUN["running"] = False
                self.RUN["seconds"] = round(time.time() - self.RUN["started"], 1)
                _drain()
        _EXEC.submit(_go)
        return {"started": True, "frames": n, "device": dev}

    def stop(self) -> dict:
        self.RUN["stop"] = True
        return {"stopping": True}

    def counts_live(self, H=None) -> dict:
        """Live count per set and per species, straight off the hierarchy (cheap, every frame)."""
        import torch
        H = H or self.H
        out = {"sets": {}, "species": {}}
        for name, lv in H.levels.items():
            occ = getattr(lv, "active", None)
            if occ is None:
                occ = getattr(lv, "occ", None)
            if occ is None:
                continue
            live = torch.as_tensor(occ) > 0
            out["sets"][name] = int(live.sum())
            names = list(getattr(lv, "type_names", []) or [])
            nt = getattr(lv, "node_type", None)
            if names and nt is not None:
                for i, nm in enumerate(names):
                    out["species"][nm] = int((live & (torch.as_tensor(nt) == i)).sum())
        return out

    RUN: dict = {"running": False, "frame": 0, "n_frames": 0, "seconds": 0.0, "error": None, "stop": False, "counts": {}}

    def close(self):
        self.RUN["stop"] = True

        def _c():
            with LOCK:
                try:
                    (self.panel.close() if self.panel is not None else self.lm.close())
                except Exception:                                # noqa: BLE001
                    pass
        _vtk(_c)


def open_view(spec_path: str, device: str = "cpu", carry: bool = False) -> View:
    """Replace the session's view with a fresh one for `spec_path`, built on the VTK thread.

    `carry=True` KEEPS THE LAST RUN'S FRAMES on the new view when it is the same scene -- same
    spec name, every set the same buffer shape -- so a change of render (dot size, surface,
    light) re-opens the renderer and PLAY replays the frames already in memory through it,
    rather than asking for the run again. The frames are level states, not pictures, which is
    what makes them re-drawable at all."""
    def _open():
        with LOCK:
            old = CURRENT.get("view")
            keep = None
            if old is not None and carry and os.path.basename(old.spec_path) == os.path.basename(spec_path):
                keep = old
            if old is not None:
                old.close()
            CURRENT["view"] = None
            v = View(spec_path, device)
            if keep is not None and keep.snaps and getattr(v, "panel", None) is None:
                same = all(n in v.H.levels and tuple(v.H.level(n).state.shape) == tuple(keep.H.level(n).state.shape)
                           for n in keep.H.levels)
                if same:
                    v.snaps = keep.snaps
                    v._keep_every = getattr(keep, "_keep_every", 1)
                    v.RUN = dict(keep.RUN); v.RUN["running"] = False
                    v.lm.n_frames = int(v.RUN.get("n_frames") or 1)   # the overlay's denominator is the run's
                    print(f"[view] {len(v.snaps)} frames of the last run carried to the new render", flush=True)
            elif keep is not None and getattr(v, "panel", None) is not None and getattr(keep, "panel", None) is not None:
                v.panel.hist = keep.panel.hist
                v.RUN = dict(keep.RUN); v.RUN["running"] = False
            CURRENT["view"] = v
            return v
    return _vtk(_open)


def current() -> View | None:
    return CURRENT.get("view")
