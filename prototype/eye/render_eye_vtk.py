"""render_eye_vtk -- the same movie, drawn by VTK instead of matplotlib.

    from render_eye_vtk import render
    render(cap, dt, "movie.mp4", "strip.png")        # drop-in for render_eye.render

WHY. `render_eye` re-rasterises about 140 000 scatter points across five panels on the
CPU every frame, and re-creates all eight axes to do it -- a second or so per frame,
which is the same order as the MLS-MPM step that produced the frame. The points are
already a point cloud in 3-D; VTK draws them on the GPU, where 140 000 points is
nothing. Same panels, same colours, same conventions, one to two orders of magnitude
less wall clock.

WHAT IS DRAWN (and it is drawn as geometry, not as a picture of geometry):

    a  anterior view    globe points coloured by tissue, muscles by muscle, brightening
                        with activation -- and now genuinely 3-D, so the depth sorting
                        and the far-hemisphere cull that `render_eye` had to compute by
                        hand are done by the depth buffer
    b  lateral view     the same scene from the side
    c  strain           ||E|| on the cut through globe and muscles
    d  von Mises        stress on the same cut
    e  grid momentum    |v| on the shared MLS-MPM grid -- the coupling itself
    f  traces           activation, shortening and gaze, with a playhead

The trace panel is the one thing VTK is bad at, so it is drawn ONCE by matplotlib for
the whole run and composited under the render each frame with a moving playhead: a
column blend, a few milliseconds, instead of three line plots per frame.
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv

import eye_anatomy as EA
from render_eye import PALETTE, MUS_RGB, camera

BG = (0.0, 0.0, 0.0)
PANEL_FONT = 15                 # panel titles
AXIS_FONT = 13                  # axis labels in the trace panel
TICK_FONT = 11
DOT_GLOBE = 6.0
DOT_MUSCLE = 5.0
DOT_FIELD = 3.4
DOT_GRID = 3.0

# (label, kind) -- laid out so that reading down a column is reading one thing.
# Top row is WHERE THINGS ARE (two views of the plant) and the velocity field that
# moves them; the second row is WHAT THE TISSUE CARRIES, the two stress-like fields
# side by side where they can be compared, with the traces beside them.
PANELS = [("a)  anterior view -- globe and the six muscles", "scene_anterior"),
          ("b)  lateral view -- the ovoid in its cup", "scene_lateral"),
          ("c)  mls-mpm grid momentum -- the coupling", "grid"),
          ("d)  green-lagrange strain |E|", "strain"),
          ("e)  von mises stress", "vm")]


def gaze_axis(h_deg, v_deg):
    """Unit optic axis for a gaze of (horizontal, vertical), in the eye's own frame.

    Same convention `eye_ops` uses to BUILD the pose, so the arrow points where the model
    says the eye points rather than where the drawing thinks it does.
    """
    h, v = np.radians(h_deg), np.radians(v_deg)
    return np.array([np.sin(h) * np.cos(v), np.sin(v), np.cos(h) * np.cos(v)])


def _cam(view, centre, span):
    """Camera position / focus / up for one of render_eye's named views, so the two
    renderers frame the plant identically."""
    r, u, f = camera(view)
    pos = np.asarray(centre) - 4.0 * np.asarray(f) * span
    return [tuple(pos), tuple(centre), tuple(u)]


def _scene_colours(cap, k):
    """Per-point RGB for the globe and the muscles at frame `k`.

    The tissue palette and the activation glow are `render_eye`'s, so the two movies
    are the same movie: sclera white, iris silver, flecks gold (they are what makes
    torsion visible), each muscle its own colour, lit from 0.34 to 1.0 by its own
    activation.
    """
    tis = np.clip(PALETTE[cap["tissue"]], 0, 1)
    act = np.clip(cap["act"][k], 0, 1)
    par = cap["mus_parent"]
    mus = np.clip(MUS_RGB[par] * (0.34 + 0.66 * act[par])[:, None], 0, 1)
    return (tis * 255).astype(np.uint8), (mus * 255).astype(np.uint8)


def _traces_image(cap, dt, width, height, dpi=100):
    """The three trace panels, rendered ONCE for the whole run.

    Returns (image, x_of_frame) so the playhead can be drawn by blending a column --
    which is the only per-frame cost this panel then has.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(cap["frame"], float) * dt
    fig, axs = plt.subplots(3, 1, figsize=(width / dpi, height / dpi), dpi=dpi,
                            facecolor="black", sharex=True)
    act = np.asarray(cap["act"])
    ln = np.asarray(cap["length"])
    rest = np.asarray(cap["rest_length"])
    rest = rest[0] if rest.ndim > 1 else rest
    gaze = np.asarray(cap["gaze"])
    cols = [m["color"] for m in EA.MUSCLES]
    keys = EA.MUSCLE_KEYS
    for i, k in enumerate(keys):
        axs[0].plot(t, act[:, i], color=cols[i], lw=1.3, label=k)
        axs[1].plot(t, 100.0 * ln[:, i] / max(float(rest[i]), 1e-9), color=cols[i], lw=1.3)
    for j, (lab, c) in enumerate(zip(["horizontal", "vertical", "torsion"],
                                     ["#4da3ff", "#7ee081", "#ff9c42"])):
        axs[2].plot(t, gaze[:, j], color=c, lw=1.5, label=lab)
    for ax, ylab in zip(axs, ["activation", "length  % of rest", "gaze  deg"]):
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=TICK_FONT)
        for s in ax.spines.values():
            s.set_color("#555555")
        ax.set_ylabel(ylab, color="white", fontsize=AXIS_FONT)
        ax.grid(alpha=0.15, color="white", lw=0.4)
    axs[0].legend(ncol=6, fontsize=TICK_FONT, frameon=False, loc="upper right",
                  labelcolor="linecolor", handlelength=1.0, columnspacing=0.9)
    axs[2].legend(ncol=3, fontsize=TICK_FONT, frameon=False, loc="upper right",
                  labelcolor="linecolor", handlelength=1.0, columnspacing=0.9)
    axs[2].set_xlabel("time  s", color="white", fontsize=AXIS_FONT)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.985, bottom=0.135, hspace=0.12)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    # where each recorded frame sits, in pixels, so the playhead can be placed
    x0, x1 = axs[2].get_position().x0, axs[2].get_position().x1
    xs = (x0 + (x1 - x0) * (t - t[0]) / max(t[-1] - t[0], 1e-9)) * img.shape[1]
    plt.close(fig)
    return img, xs.astype(int)


class EyeVTK:
    """A live VTK scene whose point clouds are UPDATED per frame, never rebuilt.

    Rebuilding the geometry every frame is what makes a naive VTK port no faster than
    matplotlib. Here each panel owns one `PolyData`; a frame writes new coordinates and
    new colours into it and asks for one screenshot.
    """

    def __init__(self, cap, size=(1920, 1080), span=0.245):
        self.cap = cap
        self.span = span
        n_panel_rows, n_panel_cols = 2, 3
        pv.OFF_SCREEN = True
        self.p = pv.Plotter(off_screen=True, window_size=size,
                            shape=(n_panel_rows, n_panel_cols), border=False)
        self.clouds = {}
        self.arrows = {}
        tis_rgb, mus_rgb = _scene_colours(cap, 0)
        centre = cap["centre"][0]

        for idx, (label, kind) in enumerate(PANELS):
            self.p.subplot(idx // n_panel_cols, idx % n_panel_cols)
            self.p.set_background("black")
            if kind.startswith("scene"):
                view = "anterior" if kind.endswith("anterior") else "lateral"
                g = pv.PolyData(np.asarray(cap["shell"][0], float))
                g["rgb"] = tis_rgb
                m = pv.PolyData(np.asarray(cap["mus_pos"][0], float))
                m["rgb"] = mus_rgb
                self.p.add_mesh(g, scalars="rgb", rgb=True, point_size=DOT_GLOBE,
                                render_points_as_spheres=True, show_scalar_bar=False)
                self.p.add_mesh(m, scalars="rgb", rgb=True, point_size=DOT_MUSCLE,
                                render_points_as_spheres=True, show_scalar_bar=False)
                arrow = self._add_gaze_arrow(centre, span)
                self.p.add_mesh(arrow, color="#00e5ff", opacity=0.9,
                                show_scalar_bar=False)
                self.arrows[kind] = (arrow, np.asarray(centre, float), span)
                self.clouds[kind] = (g, m)
                self.p.camera_position = _cam(view, centre, span)
                self.p.camera.parallel_projection = True
                self.p.camera.parallel_scale = span
            elif kind in ("strain", "vm"):
                X = np.concatenate([cap["cut_pos"][0], cap["mus_pos"][0]])
                v = np.concatenate([cap["cut_" + kind][0], cap["mus_" + kind][0]])
                c = pv.PolyData(np.asarray(X, float))
                c["v"] = np.asarray(v, float)
                hi = float(np.percentile(np.concatenate(
                    [np.concatenate(cap["cut_" + kind]).ravel(),
                     np.concatenate(cap["mus_" + kind]).ravel()]), 99.5))
                self.p.add_mesh(c, scalars="v", cmap="magma" if kind == "strain" else "inferno",
                                clim=(0.0, hi), point_size=DOT_FIELD,
                                render_points_as_spheres=True, show_scalar_bar=False)
                self.clouds[kind] = c
                self.p.camera_position = _cam("oblique", centre, 0.185)
                self.p.camera.parallel_projection = True
                self.p.camera.parallel_scale = 0.185
            else:
                P = cap["gpos"][0] if len(cap["gpos"]) else np.zeros((1, 3))
                V = cap["gvel"][0] if len(cap["gvel"]) else np.zeros(1)
                c = pv.PolyData(np.asarray(P, float))
                c["v"] = np.asarray(V, float).ravel()
                gv = np.concatenate([g for g in cap["gvel"] if np.size(g)]) \
                    if len(cap["gvel"]) else np.array([1.0])
                hi = float(np.percentile(gv, 99.0)) if gv.size else 1.0
                self.p.add_mesh(c, scalars="v", cmap="viridis", clim=(0.0, hi),
                                point_size=DOT_GRID, render_points_as_spheres=True,
                                show_scalar_bar=False)
                self.clouds[kind] = c
                self.p.camera_position = _cam("oblique", centre, 0.215)
                self.p.camera.parallel_projection = True
                self.p.camera.parallel_scale = 0.215
            self.p.add_text(label, position="upper_left", font_size=PANEL_FONT, color="white")

        # the sixth cell is the trace panel; it is filled by compositing, so keep it black
        self.p.subplot(1, 2)
        self.p.set_background("black")
        self.trace_cell = (1, 2)
        self.size = size
        self.shape = (n_panel_rows, n_panel_cols)

    def _add_gaze_arrow(self, centre, span):
        """The arrow that says WHERE THE EYE IS LOOKING.

        Without it the only cue is the pupil, and the pupil is the thing the muscles are
        deforming -- so during a contraction there is nothing in the frame that reports
        gaze. The arrow is rebuilt each frame from the pose readout, so it is the model's
        own answer, not the renderer's guess. The camera stays FIXED for the same reason:
        turning the scene while the muscles turn the eye makes the two motions
        indistinguishable.
        """
        a = pv.Arrow(start=tuple(centre), direction=(0, 0, 1),
                     tip_length=0.28, tip_radius=0.075, shaft_radius=0.028,
                     scale=1.9 * span)
        return a

    def frame(self, k):
        cap = self.cap
        tis_rgb, mus_rgb = _scene_colours(cap, k)
        for kind in ("scene_anterior", "scene_lateral"):
            g, m = self.clouds[kind]
            g.points = np.asarray(cap["shell"][k], float)
            g["rgb"] = tis_rgb
            m.points = np.asarray(cap["mus_pos"][k], float)
            m["rgb"] = mus_rgb
        gz = np.asarray(cap["gaze"])[k] if "gaze" in cap else np.zeros(3)
        for kind, (arrow, c0, span) in self.arrows.items():
            d = gaze_axis(gz[0], gz[1])
            centre = np.asarray(cap["centre"][k], float) if "centre" in cap else c0
            new = pv.Arrow(start=tuple(centre), direction=tuple(d), tip_length=0.28,
                           tip_radius=0.075, shaft_radius=0.028, scale=1.9 * span)
            arrow.points = new.points
        for kind in ("strain", "vm"):
            c = self.clouds[kind]
            c.points = np.concatenate([cap["cut_pos"][k], cap["mus_pos"][k]]).astype(float)
            c["v"] = np.concatenate([cap["cut_" + kind][k], cap["mus_" + kind][k]]).astype(float)
        if len(cap["gpos"]) > k and np.size(cap["gpos"][k]):
            c = self.clouds["grid"]
            c.points = np.asarray(cap["gpos"][k], float)
            c["v"] = np.asarray(cap["gvel"][k], float).ravel()
        # WITHOUT THIS THE MOVIE IS A STILL. `screenshot()` grabs whatever is in the
        # render window; updating a mesh's points marks the dataset modified but does not
        # itself trigger a re-render, so every frame came back byte-identical to the first
        # and only the trace playhead moved. Cost is the render this panel needs anyway.
        self.p.render()
        return self.p.screenshot(None, return_img=True)


def render(cap, dt, out_mp4, out_strip, fps=30, size=(1920, 1080), quality=8):
    """Drop-in replacement for `render_eye.render`, drawn by VTK."""
    import imageio.v2 as iio

    n = len(cap["frame"])
    scene = EyeVTK(cap, size=size)
    rows, cols = scene.shape
    ph, pw = size[1] // rows, size[0] // cols
    traces, xs = _traces_image(cap, dt, pw, ph)
    if traces.shape[:2] != (ph, pw):
        traces = np.asarray(
            iio.imread(iio.imwrite("<bytes>", traces, format="png")))[:ph, :pw]

    y0, x0 = (rows - 1) * ph, (cols - 1) * pw
    writer = iio.get_writer(out_mp4, fps=fps, quality=quality,
                            macro_block_size=None)
    strip_at = np.linspace(0, n - 1, 5).astype(int)
    strip = []
    for k in range(n):
        img = scene.frame(k)
        img[y0:y0 + traces.shape[0], x0:x0 + traces.shape[1]] = traces
        # the playhead: one blended column, which is all this panel costs per frame
        px = x0 + int(np.clip(xs[k] * traces.shape[1] / max(traces.shape[1], 1), 0,
                              traces.shape[1] - 1))
        img[y0:y0 + traces.shape[0], max(px - 1, x0):px + 2] = (255, 60, 60)
        writer.append_data(img)
        if k in strip_at:
            strip.append(img[:, :pw])
        if k % 40 == 0:
            print(f"    [vtk] {k}/{n}", flush=True)
    writer.close()
    scene.p.close()
    if strip:
        iio.imwrite(out_strip, np.concatenate(strip, axis=1))
    return out_mp4
