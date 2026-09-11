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

VTK is not thread-safe and the server is threaded, so every call holds one lock.
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import time

import numpy as np

LOCK = threading.RLock()
CURRENT: dict = {"view": None}


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
        self._tmp = tempfile.mkdtemp(prefix="plexus_bio_")
        free = str(getattr(sim, "boundary", "") or "").lower() == "free"
        self.lm = LiveMovie(out=os.path.join(self._tmp, "view.mp4"), world=list(sim.world_size), n_frames=1,
                            up=int(style.get("up_axis", 2)), name=sim.name, sim=sim, style=style,
                            centred=free and not any(k in (sim.sets or {}) for k in ("mpm_particle", "cytosol", "nucleus")),
                            can_curve=False, render_n=400_000, stills=0, px=int(px))
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
            lo, hi = P.min(0), P.max(0)
            self.focal = 0.5 * (lo + hi) + np.asarray(getattr(self.lm, "_shift", 0.0) or 0.0, float)
            R = 0.5 * float(np.linalg.norm(hi - lo))
            self.dist0 = 1.15 * R / np.tan(np.radians(float(cam.view_angle) / 2.0)) if R > 0 else self.dist0
            self.scale0 = 1.15 * R if R > 0 else float(cam.parallel_scale)
        else:
            self.scale0 = float(cam.parallel_scale)
        self.azim, self.elev, self.zoom = 30.0, 20.0, 1.0
        self.pick = None
        self.hidden: set = set()
        self.RUN = {"running": False, "frame": 0, "n_frames": 0, "seconds": 0.0, "error": None, "stop": False, "counts": {}}
        self.seconds = round(time.time() - t0, 2)
        self.set_camera(self.azim, self.elev, self.zoom)

    # ------------------------------------------------------------------ camera and picture
    def set_camera(self, azim=None, elev=None, zoom=None):
        with LOCK:
            if azim is not None: self.azim = float(azim)
            if elev is not None: self.elev = max(-89.0, min(89.0, float(elev)))
            if zoom is not None: self.zoom = max(0.15, min(8.0, float(zoom)))
            a, e = np.radians(self.azim), np.radians(self.elev)
            d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            cam = self.p.camera
            cam.focal_point = tuple(self.focal)
            cam.position = tuple(self.focal + d * (self.dist0 / self.zoom))
            cam.up = (0.0, 0.0, 1.0)
            # THE MOVIE'S CAMERA IS ORTHOGRAPHIC (`parallel_projection`, live_movie.py:547), so
            # zoom is the parallel scale (half the view height in world units), not the distance.
            if bool(cam.parallel_projection):
                cam.parallel_scale = self.scale0 / self.zoom
            self.p.renderer.ResetCameraClippingRange()

    def png(self) -> bytes:
        with LOCK:
            self.p.render()                                      # screenshot() alone returns the stale frame
            img = self.p.screenshot(return_img=True)
        import imageio.v3 as iio
        buf = io.BytesIO()
        iio.imwrite(buf, np.asarray(img), extension=".png")
        return buf.getvalue()

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
        with LOCK:
            W, Hh = self.p.window_size
            cam = self.p.camera
            M = cam.GetCompositeProjectionTransformMatrix(float(W) / float(Hh), -1.0, 1.0)
            A = np.array([[M.GetElement(i, j) for j in range(4)] for i in range(4)], float)
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
        """A yellow dot on a cluster or vertex; the cell's rings and lateral edges for a cell."""
        import pyvista as pv
        from plexus.gui import bio
        with LOCK:
            for nm in ("pick_dot", "pick_cell"):
                try:
                    self.p.remove_actor(nm, render=False)
                except Exception:                                # noqa: BLE001
                    pass
            self.pick = pick
            if not pick:
                return
            info = bio.resolve_pick(self.scene, pick)
            if not info:
                return
            T = self.scene.get("tissue")
            f = info.get("cell") if info.get("kind") == "cell" else (info.get("parent_cell"))
            if info.get("kind") != "cell" and info.get("position") is not None:
                r = 0.25
                self.p.add_mesh(pv.Sphere(radius=r, center=info["position"]), color="#ffee33", name="pick_dot")
            if T and f is not None:
                seg = []
                ring = set()
                for cap in ("apical", "basal"):
                    c = T["caps"][cap]
                    for k, t in enumerate(c["tri"]):
                        if c["face"][k] == f:
                            seg.append((c["verts"][t[1]], c["verts"][t[2]])); ring.add(t[1])
                for i in ring:
                    seg.append((T["caps"]["apical"]["verts"][i], T["caps"]["basal"]["verts"][i]))
                if seg:
                    pts = np.asarray([p for s in seg for p in s], float)
                    lines = np.concatenate([[2, 2 * i, 2 * i + 1] for i in range(len(seg))])
                    pd = pv.PolyData(pts); pd.lines = lines
                    self.p.add_mesh(pd, color="#ffee33", line_width=4, name="pick_cell")

    # ------------------------------------------------------------------ visibility by species
    def set_visible(self, species: str, on: bool):
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
    def run(self, frames: int | None = None, device: str | None = None) -> dict:
        """Simulate the spec forward in a thread, the picture following every frame.

        `engine.run` builds and seeds its own hierarchy and loops internally, so a run always
        starts from the seed (frame 0) and there is no "continue from here" yet; `frames` caps the
        number of frames, `stop()` aborts. Each frame the movie renderer is fed the run's
        hierarchy exactly as a generate does (`lm(H, tick)`), so what the page shows during the run
        is the movie's frame; the scene dict used for picking is refreshed every 10 frames."""
        import torch
        if self.RUN.get("running"):
            return {"error": "already running; STOP it first"}
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        n = int(frames or self.sim.n_frames)
        self.RUN.update(running=True, frame=0, n_frames=n, seconds=0.0, error=None, stop=False, device=dev, started=time.time())

        class _Stop(Exception):
            pass

        def _on_frame(H, tick):
            if self.RUN.get("stop") or tick > n:
                raise _Stop()
            with LOCK:
                self.H = H
                self.lm(H, tick)
                if tick % 10 == 0 or tick == n:
                    from plexus.gui import bio
                    self.scene = bio.scene_from(H, self.sim, self.spec_path)
                    if self.pick:
                        self.highlight(self.pick)
            self.RUN["frame"] = int(tick)
            self.RUN["seconds"] = round(time.time() - self.RUN["started"], 1)
            self.RUN["counts"] = self.counts_live(H)

        def _go():
            from plexus import engine
            try:
                engine.run(self.sim, out_path=None, device=dev, on_frame=_on_frame)
            except _Stop:
                pass
            except Exception as e:                                   # noqa: BLE001
                self.RUN["error"] = f"{type(e).__name__}: {e}"[:600]
            finally:
                self.RUN["running"] = False
                self.RUN["seconds"] = round(time.time() - self.RUN["started"], 1)
        threading.Thread(target=_go, daemon=True).start()
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
        with LOCK:
            try:
                self.lm.close()
            except Exception:                                    # noqa: BLE001
                pass


def open_view(spec_path: str, device: str = "cpu") -> View:
    """Replace the session's view with a fresh one for `spec_path`."""
    with LOCK:
        old = CURRENT.get("view")
        if old is not None:
            old.close()
        CURRENT["view"] = None
        v = View(spec_path, device)
        CURRENT["view"] = v
        return v


def current() -> View | None:
    return CURRENT.get("view")
