"""run_bench -- run the minimal rig and render the 2x2 movie.

    python run_bench.py --tag base --device cuda:0
    python run_bench.py --tag stiff --muscle-youngs 960 --device cuda:0

THE MEASUREMENT. Every frame records two lengths of the same muscle:

    path        the centreline, summed along the tissue -- what `muscle_geometry`
                reports and what "the muscle shortened by 31%" has always meant here
    end-to-end  the straight distance between the anchored cap and the tendon cap --
                what the LOAD actually feels

and their ratio, the TRANSMISSION. A muscle that pulls has transmission 1: every
millimetre it loses, its ends approach by a millimetre. A muscle that squashes along
its own axis has transmission 0: it shortens and its ends do not move. Eye F's lateral
rectus measures 0.08. That number is the whole reason this rig exists, so it is the
fourth panel rather than a line in a log.

    a  the rig in 3-D          bone, muscle, sphere -- muscle lit by activation
    b  von Mises stress        where the active stress is actually going
    c  the two lengths         path and end-to-end on one axis, so the gap is visible
    d  transmission            (1 - end-to-end/rest) / (1 - path/rest), against 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EYE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(EYE, "..", "..", "src"))
sys.path.insert(0, EYE)
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops                   # noqa: F401
import bench_ops                   # noqa: F401
import bench_spec
import run_eye
from plexus.schema import load as load_spec

# one folder per run: the spec, the curves, the json and the movie of a given
# configuration belong together, and a flat `runs/` makes an iteration hard to hand over
def rundir(tag):
    d = os.path.join(HERE, tag)
    os.makedirs(d, exist_ok=True)
    return d


def measure(cap, dt):
    """path length, end-to-end length and transmission, per recorded frame."""
    Y = np.asarray(cap["mus_pos"])
    s = np.asarray(cap["mus_s"])
    par = np.asarray(cap["mus_parent"])
    first = par == par.min()          # muscle 0; with a pair the second is its mirror
    org_i, ten_i = first & (s < 0.10), first & (s > 0.90)
    org = np.array([Y[k][org_i].mean(0) for k in range(len(Y))])
    ten = np.array([Y[k][ten_i].mean(0) for k in range(len(Y))])
    e2e = np.linalg.norm(ten - org, axis=1)
    # WHAT THE LOAD RECEIVES. "The ends approached" counts an anchor sliding off its
    # bone as if it were useful work -- on this rig that was 99% of it. Delivered work
    # is the TENDON's displacement alone; the anchor's is reported beside it as slip.
    d_ten = np.linalg.norm(ten - ten[0], axis=1)
    d_org = np.linalg.norm(org - org[0], axis=1)
    path = np.asarray(cap["length"])[:, 0]
    rest = np.asarray(cap["rest_length"])
    rest = float(rest[0][0] if np.ndim(rest) > 1 else rest[0])
    d_path = 1.0 - path / rest
    d_e2e = 1.0 - e2e / max(e2e[0], 1e-12)
    short = rest * d_path
    ok = np.abs(short) > 1e-5
    trans = np.where(ok, d_ten / np.where(ok, short, 1), np.nan)
    slip = np.where(ok, d_org / np.where(ok, short, 1), np.nan)
    return dict(t=np.asarray(cap["frame"]) * dt, path=path, e2e=e2e, rest=rest,
                gaze=np.asarray(cap["gaze"]) if "gaze" in cap else None,
                act_all=np.asarray(cap["act"]),
                d_path=d_path, d_e2e=d_e2e, transmission=trans, anchor_slip=slip,
                d_ten=d_ten, d_org=d_org,
                act=np.asarray(cap["act"])[:, 0], ten_pos=ten, org_pos=org)


def render(cap, M, dt, out_mp4, out_strip, fps=30, size=(1600, 1200)):
    """One VTK render pass for all four panels.

    The plots are VTK charts, not matplotlib composited underneath: it keeps one set of
    fonts and one colour pipeline across the frame, and it drops the per-frame figure
    that was costing more than the 3-D itself.

    A LUT PER BODY. One shared colour map makes the quiet body black, which reads as
    absent -- the first version of this panel lost the whole rig that way. Each body
    keeps its own colour family and its own range, so a muscle is always red and a ball
    always pale, and the stress modulates within that identity rather than replacing it.
    """
    import pyvista as pv
    import imageio.v2 as iio
    from matplotlib.colors import LinearSegmentedColormap

    # A LUT PER BODY, EACH STARTING AT THAT BODY'S OWN COLOUR. `hot` and `bone` both map
    # zero to black, so at rest -- every frame before the step -- the muscle and the ball
    # were black on black and the rig looked empty. A body at rest must still look like
    # itself; the stress brightens it from there.
    MUSCLE_LUT = LinearSegmentedColormap.from_list(
        "muscle", ["#8c3b3b", "#ff5c5c", "#ff9c42", "#ffd24d", "#fff3c4"])
    TISSUE_LUT = LinearSegmentedColormap.from_list(
        "tissue", ["#9aa4b0", "#c9cfd6", "#8fd3ff", "#d8f0ff", "#ffffff"])

    n = len(cap["frame"])
    n_cut = len(cap["cut_pos"][0])
    hi_s = float(np.percentile(np.concatenate(cap["cut_vm"]).ravel(), 99.5))
    hi_m = float(np.percentile(np.concatenate(cap["mus_vm"]).ravel(), 99.5))

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, shape=(2, 2), border=False)

    sph = pv.PolyData(np.asarray(cap["cut_pos"][0], float))
    sph["vm"] = np.asarray(cap["cut_vm"][0], float)
    mus = pv.PolyData(np.asarray(cap["mus_pos"][0], float))
    mus["vm"] = np.asarray(cap["mus_vm"][0], float)
    shell = pv.PolyData(np.asarray(cap["shell"][0], float))
    bone = pv.Cube(center=tuple(0.5 * (np.array(bench_spec.BONE_LO)
                                       + np.array(bench_spec.BONE_HI))),
                   x_length=bench_spec.BONE_HI[0] - bench_spec.BONE_LO[0],
                   y_length=bench_spec.BONE_HI[1] - bench_spec.BONE_LO[1],
                   z_length=bench_spec.BONE_HI[2] - bench_spec.BONE_LO[2])

    # frame whatever rig this is: from the bone's left face to the far side of the ball
    x0 = bench_spec.BONE_LO[0]
    x1 = float(np.asarray(cap["centre"][0])[0]) + bench_spec.SPHERE_R
    fx, scale = 0.5 * (x0 + x1), max(0.62 * (x1 - x0), 0.13)
    VIEWS = [((fx, -1.4, 0.52), (fx, 0.5, 0.5), (0, 0, 1), "a   3-D + stress, side"),
             ((fx, 0.5, 1.6), (fx, 0.5, 0.5), (-1, 0, 0), "b   the same, from above")]
    for i, (pos, foc, up, label) in enumerate(VIEWS):
        p.subplot(0, i)
        p.set_background("black")
        p.add_mesh(bone, color="#d8d0bb", opacity=0.95, specular=0.2)
        p.add_mesh(shell, color="#b6bec9", opacity=0.30, point_size=2.8,
                   render_points_as_spheres=True, show_scalar_bar=False)
        p.add_mesh(sph, scalars="vm", cmap=TISSUE_LUT, clim=(0, hi_s), point_size=4.0,
                   render_points_as_spheres=True, show_scalar_bar=False)
        p.add_mesh(mus, scalars="vm", cmap=MUSCLE_LUT, clim=(0, hi_m), point_size=5.0,
                   render_points_as_spheres=True, show_scalar_bar=False)
        p.add_text(label, position="upper_left", font_size=13, color="white")
        p.camera_position = [pos, foc, up]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = scale
        p.enable_depth_peeling(8)

    def _style(ch, xl, yl):
        ch.background_color = (0, 0, 0, 1)
        ch.border_color = "#555555"
        ch.x_label, ch.y_label = xl, yl
        for ax in (ch.x_axis, ch.y_axis):
            ax.label_size = 20
            ax.tick_label_size = 16
            ax.label_color = "white"
            ax.tick_label_color = "white"
            ax.color = "#888888"
            # no grid: two curves against a red playhead read better without one, and
            # the grid was the brightest thing in the frame
            try:
                ax.grid = False
            except Exception:
                ax.grid_color = (0, 0, 0, 0)
        ch.legend_visible = True

    t = M["t"]
    y0 = float(np.nanmin(1e3 * M["e2e"])) - 3
    y1 = float(np.nanmax(1e3 * M["path"])) + 3

    p.subplot(1, 0)
    p.set_background("black")
    ch_len = pv.Chart2D()
    ch_len.line(t, 1e3 * M["path"], color="#ffd24d", width=3.0, label="path (centreline)")
    ch_len.line(t, 1e3 * M["e2e"], color="#4da3ff", width=3.0, label="end to end")
    head_len = ch_len.line([t[0], t[0]], [y0, y1], color="#ff5c5c", width=2.5)
    _style(ch_len, "time  s", "length  x10^-3 world")
    p.add_chart(ch_len)
    p.add_text("c   what the muscle loses, and what the load feels",
               position="upper_left", font_size=13, color="white")

    p.subplot(1, 1)
    p.set_background("black")
    ch_tr = pv.Chart2D()
    if M.get("gaze") is not None and float(np.abs(M["gaze"]).max()) > 0.2:
        # a pair on a socket-held ball TURNS it, and the angle is the readout the eye
        # cares about -- transmission is a single-muscle question
        g = M["gaze"]
        for j, (lab, col) in enumerate(zip(["horizontal", "vertical", "torsion"],
                                           ["#4da3ff", "#7ee081", "#ff9c42"])):
            ch_tr.line(t, g[:, j], color=col, width=3.0, label=lab)
        lo, hiy = float(g.min()) - 1.0, float(g.max()) + 1.0
        head_tr = ch_tr.line([t[0], t[0]], [lo, hiy], color="#ff5c5c", width=2.5)
        _style(ch_tr, "time  s", "rotation  deg")
    else:
        ch_tr.line([t[0], t[-1]], [1.0, 1.0], color="#7ee081", width=2.0, style="--",
                   label="perfect")
        ch_tr.line(t, np.nan_to_num(M["transmission"]), color="#ff9c42", width=3.0,
                   label="delivered to the load")
        ch_tr.line(t, np.nan_to_num(M["anchor_slip"]), color="#ff5c5c", width=2.5,
                   style="--", label="lost to anchor slip")
        head_tr = ch_tr.line([t[0], t[0]], [-0.1, 1.25], color="#ff5c5c", width=2.5)
        _style(ch_tr, "time  s", "fraction of the shortening")
    p.add_chart(ch_tr)
    p.add_text("d   what the ball does", position="upper_left",
               font_size=13, color="white")

    writer = iio.get_writer(out_mp4, fps=fps, quality=8, macro_block_size=None)
    strip_at = set(np.linspace(0, n - 1, 5).astype(int))
    strip = []
    for k in range(n):
        sph.points = np.asarray(cap["cut_pos"][k], float)
        sph["vm"] = np.asarray(cap["cut_vm"][k], float)
        mus.points = np.asarray(cap["mus_pos"][k], float)
        mus["vm"] = np.asarray(cap["mus_vm"][k], float)
        shell.points = np.asarray(cap["shell"][k], float)
        head_len.update([t[k], t[k]], [y0, y1])
        head_tr.update([t[k], t[k]], [-0.1, 1.25])
        p.render()
        img = p.screenshot(None, return_img=True)
        writer.append_data(img)
        if k in strip_at:
            strip.append(img)
        if k % 40 == 0:
            print(f"    [render] {k}/{n}", flush=True)
    writer.close()
    p.close()
    if strip:
        iio.imwrite(out_strip, np.concatenate(strip, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--muscle-youngs", type=float, default=240.0)
    ap.add_argument("--contract", type=float, default=67.0)
    ap.add_argument("--muscle-radius", type=float, default=bench_spec.MUSCLE_R)
    ap.add_argument("--k_bone", type=float, default=9000.0)
    ap.add_argument("--pair", action="store_true",
                    help="two antagonist muscles pulling tangentially on a ball held in a "
                         "socket, alternately -- the minimal LR/MR pair")
    ap.add_argument("--tendon-gap", type=float, default=None,
                    help="signed clearance of the tendon from the globe, world units; "
                         "positive floats it clear, default embeds it by 0.012")
    ap.add_argument("--lever", type=float, default=0.62,
                    help="insertion offset as a fraction of the radius: the moment arm")
    ap.add_argument("--muscle-length", type=float, default=None,
                    help="world units; the sphere moves to suit, the attachments do not")
    ap.add_argument("--anchor", default="bone", choices=["bone", "clamp", "spring"],
                    help="bone = embedded in a pinned MPM block (the muscle is attached "
                         "to a body, as at the tendon end); clamp = kinematic on the cap; "
                         "spring = the stock penalty, which slips 0.063 world")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--hold", type=int, default=600)
    ap.add_argument("--rest", type=int, default=250)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--style", default="panels", choices=["panels", "pairs"],
                    help="pairs = the single fixed view of eye_G/pairs_fixed.mp4, for a "
                         "direct comparison between the two implementations")
    ap.add_argument("--render-only", action="store_true",
                    help="re-draw from the cached capture; the physics is unchanged")
    a = ap.parse_args()

    OUT = rundir(a.tag)
    spec = bench_spec.build(name=f"bench_{a.tag}", n_frames=a.frames,
                            muscle_youngs=a.muscle_youngs, contract=a.contract,
                            muscle_radius=a.muscle_radius, k_bone=a.k_bone,
                            anchor=a.anchor, muscle_length=a.muscle_length,
                            pair=a.pair, lever=a.lever, hold=a.hold, rest=a.rest,
                            tendon_gap=a.tendon_gap)
    path = os.path.join(OUT, f"{a.tag}_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# minimal transmission rig -- {a.tag}: E {a.muscle_youngs}, A {a.contract}, "
                 f"radius {a.muscle_radius}, k_bone {a.k_bone}\n")
        yaml.safe_dump(spec, fh, sort_keys=False, width=100)

    sim = load_spec(path)
    dt = float(sim.dt)
    print(f"[bench {a.tag}] {a.frames} frames, E {a.muscle_youngs}, A {a.contract}", flush=True)
    t0 = time.time()
    cache = os.path.join(OUT, f"{a.tag}_cap.npz")
    if a.render_only and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        cap = {k: (list(z[k]) if z[k].dtype == object else z[k]) for k in z.files}
        print(f"[bench {a.tag}] re-rendering from {cache}", flush=True)
    else:
        _, cap = run_eye.capture_run(sim, a.device, stride=a.stride)
        np.savez_compressed(cache, **{k: np.array(v, dtype=object) if isinstance(v, list)
                                      else v for k, v in cap.items()})
    M = measure(cap, dt)
    hold = slice(int(0.6 * len(M["t"])), int(0.85 * len(M["t"])))
    res = dict(tag=a.tag, muscle_youngs=a.muscle_youngs, contract=a.contract,
               A_over_E=round(a.contract / a.muscle_youngs, 4),
               muscle_radius=a.muscle_radius, k_bone=a.k_bone,
               tendon_gap=(-0.012 if a.tendon_gap is None else a.tendon_gap),
               rest_length=round(float(M["rest"]), 5),
               path_shortening_pct=round(float(100 * np.nanmax(M["d_path"])), 2),
               end_to_end_shortening_pct=round(float(100 * np.nanmax(M["d_e2e"])), 2),
               transmission=round(float(np.nanmedian(M["transmission"][hold])), 4),
               anchor_slip=round(float(np.nanmedian(M["anchor_slip"][hold])), 4),
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(os.path.join(OUT, f"{a.tag}_curves.npz"),
                        **{k: v for k, v in M.items()})
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"[bench {a.tag}] path -{res['path_shortening_pct']}%  "
          f"end-to-end -{res['end_to_end_shortening_pct']}%  "
          f"DELIVERED {res['transmission']}  anchor slip {res['anchor_slip']}  "
          f"[{res['seconds']}s]", flush=True)
    if not a.no_movie:
        if a.style == "pairs":
            render_pairs_style(cap, M, dt, os.path.join(OUT, f"{a.tag}_pairs.mp4"),
                               os.path.join(OUT, f"{a.tag}_pairs_strip.png"))
        else:
            render(cap, M, dt, os.path.join(OUT, f"{a.tag}.mp4"),
                   os.path.join(OUT, f"{a.tag}_strip.png"))
        print(f"[bench {a.tag}] -> {os.path.join(OUT, a.tag)}.mp4", flush=True)


if __name__ == "__main__":
    main()


def render_pairs_style(cap, M, dt, out_mp4, out_strip, fps=30, size=(1600, 1200),
                       cmd=None):
    """The rig, drawn the way `eye_G/pairs_fixed.mp4` draws the scanned eye.

    Same layout, same conventions, so the two implementations can be put side by side and
    compared without translating between two figures: one fixed camera on the whole scene,
    the globe as a pale point cloud, each muscle in its key colour, a gaze arrow, and the
    state as text -- frame and time and command and gaze at the top, per-muscle activation
    at the bottom. No panels and no plots, because the question this view answers is
    "where is the eye looking and which muscle is pulling", and a plot answers a different
    one.

    The camera does not move. A scene that turns while the muscles turn the eye makes the
    two rotations indistinguishable, which is the whole reason the arrow is here.
    """
    import pyvista as pv
    import imageio.v2 as iio
    from matplotlib.colors import to_rgb
    import eye_anatomy as EA

    n = len(cap["frame"])
    par = np.asarray(cap["mus_parent"])
    keys = [EA.MUSCLE_KEYS[i % len(EA.MUSCLE_KEYS)] for i in range(int(par.max()) + 1)]
    cols = [EA.MUSCLES[i % len(EA.MUSCLES)]["color"] for i in range(len(keys))]

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, border=False)
    p.set_background("black")
    shell = pv.PolyData(np.asarray(cap["shell"][0], float))
    p.add_mesh(shell, color="#d5d8dc", point_size=3.0, render_points_as_spheres=True,
               show_scalar_bar=False)
    muscles = []
    for i in range(len(keys)):
        sel = par == i
        m = pv.PolyData(np.asarray(cap["mus_pos"][0], float)[sel])
        p.add_mesh(m, color=cols[i], point_size=4.6, render_points_as_spheres=True,
                   show_scalar_bar=False)
        muscles.append((m, sel))
    bone = pv.Cube(center=tuple(0.5 * (np.array(bench_spec.BONE_LO)
                                       + np.array(bench_spec.BONE_HI))),
                   x_length=bench_spec.BONE_HI[0] - bench_spec.BONE_LO[0],
                   y_length=bench_spec.BONE_HI[1] - bench_spec.BONE_LO[1],
                   z_length=bench_spec.BONE_HI[2] - bench_spec.BONE_LO[2])
    p.add_mesh(bone, color="#cfc7b4", opacity=0.85)

    centre0 = np.asarray(cap["centre"][0], float)
    R = float(np.linalg.norm(np.asarray(cap["shell"][0], float) - centre0, axis=1).mean())
    arrow = pv.Arrow(start=tuple(centre0), direction=(0, 0, 1), tip_length=0.26,
                     tip_radius=0.07, shaft_radius=0.025, scale=2.1 * R)
    p.add_mesh(arrow, color="#ffd24d")

    x1 = centre0[0] + R
    fx = 0.5 * (bench_spec.BONE_LO[0] + x1)
    p.camera_position = [(fx, -1.4, 0.52), (fx, 0.5, 0.5), (0, 0, 1)]
    p.camera.parallel_projection = True
    p.camera.parallel_scale = max(0.70 * (x1 - bench_spec.BONE_LO[0]), 0.16)
    t_top = p.add_text("", position="upper_left", font_size=13, color="white")
    t_bot = p.add_text("", position="lower_left", font_size=12, color="white")

    g = M.get("gaze")
    act = M["act_all"]
    writer = iio.get_writer(out_mp4, fps=fps, quality=8, macro_block_size=None)
    strip_at = set(np.linspace(0, n - 1, 5).astype(int))
    strip = []
    for k in range(n):
        shell.points = np.asarray(cap["shell"][k], float)
        Y = np.asarray(cap["mus_pos"][k], float)
        for m, sel in muscles:
            m.points = Y[sel]
        centre = np.asarray(cap["centre"][k], float)
        gz = g[k] if g is not None else np.zeros(3)
        import render_eye_vtk as RV
        d = RV.gaze_axis(gz[0], gz[1])
        arrow.points = pv.Arrow(start=tuple(centre), direction=tuple(d), tip_length=0.26,
                                tip_radius=0.07, shaft_radius=0.025, scale=2.1 * R).points
        cm = cmd[k] if cmd is not None else None
        t_top.SetText(0, "frame %4d  t = %.2f s\n%sgaze     h %+5.1f  v %+5.1f  t %+5.1f"
                      % (int(cap["frame"][k]), M["t"][k],
                         ("command  " + "  ".join(f"{ky} {cm[i]:.2f}"
                                                  for i, ky in enumerate(keys)) + "\n")
                         if cm is not None else "",
                         gz[0], gz[1], gz[2]))
        t_bot.SetText(1, "activation   " + "   ".join(f"{ky} {act[k, i]:.2f}"
                                                      for i, ky in enumerate(keys)))
        p.render()
        img = p.screenshot(None, return_img=True)
        writer.append_data(img)
        if k in strip_at:
            strip.append(img)
        if k % 40 == 0:
            print(f"    [pairs] {k}/{n}", flush=True)
    writer.close()
    p.close()
    if strip:
        iio.imwrite(out_strip, np.concatenate(strip, axis=1))
