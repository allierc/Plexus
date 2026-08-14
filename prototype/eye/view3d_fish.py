"""view3d_fish -- render the MEASURED plant in 3-D, so the geometry can be checked
against the drawing it came from before any of it is simulated.

Everything drawn here is what the spec will build: the globe at its measured
flattening, the lens at its measured size and position, and each muscle as the
actual MPM points `fish_anatomy.seed_points` will seed, in the shape traced off
Fig. 12.1A. Nothing is a schematic.

    python view3d_fish.py                 -> archive/eye_F/fig_plant3d.png
    python view3d_fish.py --plant adult   -> the Kasprick adult insertions

WHICH EYE THIS IS. Fig. 12.1A draws the RIGHT eye. Its coordinates, used directly
as a right-handed (x = caudal, y = dorsal, z = lateral) frame -- the frame every
operator here already uses -- describe the LEFT eye, because (caudal, dorsal,
lateral) is a left-handed triad on the right side and a right-handed one on the
left. The modelled plant is therefore the left eye, the mirror twin of the drawn
one, and the ventral panel below is the mirror of the figure: medial to the LEFT.
Nothing else changes -- a mirror does not alter any muscle's length, section or
line of action, only the sign convention for torsion.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyvista as pv

import fish_anatomy as FA

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "eye_F")

# globe-local units throughout: 1.0 = the equatorial semi-axis (125 um)
BG = "black"
GLOBE_COL = "#2b2f36"
SCLERA_COL = "#e8e2d4"
CORNEA_COL = "#7fd8ff"
LENS_COL = "#cfe8ff"

# (label, camera direction, view-up) -- in (caudal, dorsal, lateral)
VIEWS = [
    ("ventral  -- rostral up, medial LEFT (the mirror of Fig. 12.1A: this is the left eye)",
     (0, -1, 0), (-1, 0, 0)),
    ("lateral  -- down the optic axis, the cornea facing you; rostral left",
     (0, 0, 1), (0, 1, 0)),
    ("dorsal   -- from above; SO and SR (dashed in the figure) run over this face",
     (0, 1, 0), (1, 0, 0)),
    ("rostro-ventro-lateral  -- three-quarter", (-0.62, -0.52, 0.62), (0, 1, 0)),
]

# everything is contained in a box about this big; a PARALLEL projection at a fixed
# scale keeps all four panels at the same magnification, so a muscle that looks
# longer in one panel IS longer.
SCENE_CENTRE = (0.0, 0.0, -0.15)
SCENE_SCALE = 1.55


def globe_mesh(ratio, lens_c, lens_r, n=180):
    """The globe as an ellipsoid, scalar-coloured by tissue.

    The corneal window is not a decoration: it is where the lens reaches the
    surface, so it is computed from the lens, not chosen. Everything the lens does
    not reach is sclera -- and the sclera is where a muscle is allowed to insert.
    """
    s = pv.Sphere(radius=1.0, theta_resolution=n, phi_resolution=n)
    p = s.points.copy()
    p[:, 2] *= ratio
    s.points = p
    lc = np.array([0.0, 0.0, lens_c * ratio])
    d = np.linalg.norm(s.points - lc, axis=1)
    # the limbus is where the lens rim lands on the shell. On the measured globe the
    # lens reaches the surface and this is a real window; on the guessed one it sits
    # further back, so the window is what that geometry implies -- possibly none.
    s["tissue"] = np.where(d < lens_r * 1.15, 1.0, 0.0)
    return s, lc, lens_r


def strap_cloud(key, ins_dir, origin, ratio, width, thickness, n_pts,
                arc_deg=None, gap=0.14, embed=-0.113, frac=1.0):
    """The muscle's material points, built by the SAME code the operator runs.

    `muscle_ops.strap_path` is imported rather than re-implemented, so this figure
    cannot drift from the simulation: what is drawn here is what gets seeded.
    Distances are in equatorial semi-axes (the globe is the unit sphere squashed
    to `ratio` along the optic axis).
    """
    from muscle_ops import strap_path, resample, _taper, _radical_inverse

    path, binorm = strap_path(np.zeros(3), ins_dir, origin, 1.0, ratio,
                              arc_deg, gap, embed, frac)
    pts, tan, L = resample(path)
    j = np.arange(n_pts)
    sv = (j + 0.5) / n_pts
    rr = np.sqrt(_radical_inverse(j, 2))
    th = 2 * np.pi * _radical_inverse(j, 3)
    k = np.clip((sv * (len(pts) - 1)).astype(int), 0, len(pts) - 1)
    t_hat = tan[k]
    b_hat = np.tile(binorm, (n_pts, 1))
    b_hat -= (b_hat * t_hat).sum(1, keepdims=True) * t_hat
    b_hat /= np.linalg.norm(b_hat, axis=1, keepdims=True).clip(1e-12)
    r_hat = np.cross(t_hat, b_hat)
    tap = _taper(sv)
    x = pts[k] + (0.5 * width * tap * rr * np.cos(th))[:, None] * b_hat \
        + (0.5 * thickness * tap * rr * np.sin(th))[:, None] * r_hat
    cloud = pv.PolyData(x)
    cloud["s"] = sv
    return cloud, L


def plant_config(plant):
    """(insertion directions, origins, widths, thickness, globe ratio, strap frac).

    `mammal` is the guess this prototype started from -- `eye_anatomy`'s annulus of
    Zinn, trochlea and 0.82 globe -- kept so the two can be drawn side by side.
    """
    if plant == "mammal":
        import eye_anatomy as EA
        org = (EA.origins_world() - np.asarray(EA.GLOBE_CENTER)[None, :]) / EA.A_EQ
        return dict(ins=EA.insertion_dirs(), org=org, width=np.full(6, 0.034 / EA.A_EQ),
                    thickness=0.021 / EA.A_EQ, ratio=EA.AXIAL_RATIO, frac=0.88,
                    lens_c=EA.LENS_CENTER[2], lens_r=EA.LENS_RADIUS)
    L = FA.lens()
    return dict(ins=FA.insertion_dirs(plant), org=FA.origins(plant),
                width=FA.strap_widths() / FA.A_EQ, thickness=FA.strap_thickness() / FA.A_EQ,
                ratio=FA.axial_ratio(), frac=1.0, arc=None, gap=0.14, embed=-0.113,
                lens_c=L["center_axial"], lens_r=L["radius"])


def build(plant="larva", n_pts=9000, seed=0):
    C = plant_config(plant)
    ins, org, ratio = C["ins"], C["org"], C["ratio"]
    pv.global_theme.allow_empty_mesh = True
    globe, lc, lr = globe_mesh(ratio, C["lens_c"], C["lens_r"])
    sclera = globe.threshold(0.5, scalars="tissue", invert=True)
    cornea = globe.threshold(0.5, scalars="tissue")
    lens = pv.Sphere(radius=lr, center=lc, theta_resolution=90, phi_resolution=90)
    muscles, lengths = {}, {}
    for i, k in enumerate(FA.MUSCLE_KEYS):
        muscles[k], lengths[k] = strap_cloud(k, ins[i], org[i], ratio, C["width"][i],
                                             C["thickness"], n_pts, frac=C["frac"],
                                             arc_deg=C["arc"], gap=C["gap"],
                                             embed=C["embed"])
    return dict(sclera=sclera, cornea=cornea, lens=lens, muscles=muscles, ratio=ratio,
                lengths=lengths, ins=ins, org=org)


# Opacity of the globe. High enough that a muscle passing BEHIND it is visibly
# dimmed -- which is the check the drawing asks for, since SO and SR are dashed in
# Fig. 12.1A precisely because they run on the far (dorsal) side -- and low enough
# that it can still be followed there.
SCLERA_ALPHA = 0.60
CORNEA_ALPHA = 0.45
LENS_ALPHA = 0.34


def _add_globe(p, B):
    p.add_mesh(B["sclera"], color=SCLERA_COL, opacity=SCLERA_ALPHA, specular=0.35,
               smooth_shading=True, show_scalar_bar=False)
    p.add_mesh(B["cornea"], color=CORNEA_COL, opacity=CORNEA_ALPHA, specular=0.9,
               smooth_shading=True, show_scalar_bar=False)
    p.add_mesh(B["lens"], color=LENS_COL, opacity=LENS_ALPHA, specular=1.0,
               smooth_shading=True, show_scalar_bar=False)
    # translucency needs ordered compositing, or a muscle behind the globe is drawn
    # in front of it depending only on the order the actors happened to be added
    try:
        p.enable_depth_peeling(12)
    except Exception:
        pass


def render(plant="larva", n_pts=9000, out=None, size=(2100, 1900)):
    B = build(plant, n_pts)
    out = out or os.path.join(OUT, f"fig_plant3d_{plant}.png")
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, shape=(2, 2), border=False)
    ins_all = B["ins"] * (1.0 / np.linalg.norm(B["ins"], axis=1, keepdims=True))
    org_all = B["org"]
    for i, (label, direction, up) in enumerate(VIEWS):
        p.subplot(i // 2, i % 2)
        p.set_background(BG)
        _add_globe(p, B)
        for k, cloud in B["muscles"].items():
            p.add_mesh(cloud, color=FA.COLOR[k], point_size=3.4,
                       render_points_as_spheres=True, show_scalar_bar=False)
        # each muscle's two attachments: a ring on the globe, a cube at the skull
        for j, k in enumerate(FA.MUSCLE_KEYS):
            p.add_mesh(pv.Sphere(radius=0.045, center=ins_all[j]), color="white")
            p.add_mesh(pv.Cube(center=org_all[j], x_length=0.07, y_length=0.07,
                               z_length=0.07), color=FA.COLOR[k])
        p.add_mesh(pv.Line((0, 0, -0.2), (0, 0, 1.75)), color="#00e5ff", line_width=2)
        for j, k in enumerate(FA.MUSCLE_KEYS):
            p.add_point_labels(np.array([ins_all[j] * 1.30]), [k], text_color=FA.COLOR[k],
                               font_size=20, bold=True, shape=None, show_points=False,
                               always_visible=True)
        p.add_text(label, position="upper_left", font_size=10, color="white")
        p.camera_position = [tuple(np.asarray(SCENE_CENTRE) + 6.0 * np.asarray(direction)),
                             SCENE_CENTRE, up]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = SCENE_SCALE
    p.screenshot(out)
    p.close()
    return out


def render_single(plant="larva", view=0, out=None, size=(960, 1250), parallel_scale=1.122,
                  mirror=False, labels=True, axis=True):
    """One view, at a chosen scale -- for laying the model beside the drawing.

    `parallel_scale` is the half-height of the frame in equatorial semi-axes, so it can
    be matched to a figure panel exactly: crop the drawing to +-N a_eq and pass N here
    and the two globes come out the same size on the page.

    `mirror` flips the image left-right, which is precisely the left-eye/right-eye map:
    the model is the left eye and Fig. 12.1A draws the right one, so mirroring is what
    puts them in the same orientation. It is a reflection, not a cheat -- no length or
    angle changes, only the sign of torsion.
    """
    B = build(plant)
    label, direction, up = VIEWS[view]
    out = out or os.path.join(OUT, f"fig_plant3d_{plant}_v{view}.png")
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, border=False)
    p.set_background(BG)
    _add_globe(p, B)
    for k, cloud in B["muscles"].items():
        p.add_mesh(cloud, color=FA.COLOR[k], point_size=3.0,
                   render_points_as_spheres=True, show_scalar_bar=False)
    ins = B["ins"] / np.linalg.norm(B["ins"], axis=1, keepdims=True)
    for j, k in enumerate(FA.MUSCLE_KEYS):
        p.add_mesh(pv.Sphere(radius=0.04, center=ins[j]), color="white")
        p.add_mesh(pv.Cube(center=B["org"][j], x_length=0.06, y_length=0.06, z_length=0.06),
                   color=FA.COLOR[k])
    if axis:
        p.add_mesh(pv.Line((0, 0, -0.15), (0, 0, 1.7)), color="#00e5ff", line_width=2)
    if labels:
        for j, k in enumerate(FA.MUSCLE_KEYS):
            p.add_point_labels(np.array([ins[j] * 1.26]), [k], text_color=FA.COLOR[k],
                               font_size=19, bold=True, shape=None, show_points=False,
                               always_visible=True)
    p.camera_position = [tuple(6.0 * np.asarray(direction)), (0, 0, 0), up]
    p.camera.parallel_projection = True
    p.camera.parallel_scale = parallel_scale
    img = p.screenshot(None, return_img=True)
    p.close()
    if mirror:
        img = img[:, ::-1]
    import imageio.v2 as iio
    iio.imwrite(out, img)
    return out


def movie(plant="larva", out=None, n_frames=240, fps=30, size=(1280, 1120), n_pts=11000):
    """A turntable of the plant, so the third dimension is actually visible.

    The camera swings once around the dorso-ventral axis and then tips over the top,
    which is the pair of moves that separates the two things a still cannot show: which
    muscles run on the far side of the globe, and how far each one wraps before it
    leaves for its bone. The globe is drawn solid enough to occlude, so a band that
    dims is a band that has gone behind.
    """
    B = build(plant, n_pts)
    out = out or os.path.join(OUT, "movie.mp4")
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, border=False)
    p.set_background(BG)
    _add_globe(p, B)
    for k, cloud in B["muscles"].items():
        p.add_mesh(cloud, color=FA.COLOR[k], point_size=3.2,
                   render_points_as_spheres=True, show_scalar_bar=False)
    ins = B["ins"] / np.linalg.norm(B["ins"], axis=1, keepdims=True)
    for j, k in enumerate(FA.MUSCLE_KEYS):
        p.add_mesh(pv.Sphere(radius=0.04, center=ins[j]), color="white")
        p.add_mesh(pv.Cube(center=B["org"][j], x_length=0.06, y_length=0.06, z_length=0.06),
                   color=FA.COLOR[k])
    p.add_mesh(pv.Line((0, 0, -0.15), (0, 0, 1.8)), color="#00e5ff", line_width=2)
    for j, k in enumerate(FA.MUSCLE_KEYS):
        p.add_point_labels(np.array([ins[j] * 1.28]), [k], text_color=FA.COLOR[k],
                           font_size=18, bold=True, shape=None, show_points=False,
                           always_visible=True)
    txt = p.add_text("", position="upper_left", font_size=14, color="white")
    p.camera.parallel_projection = True
    p.camera.parallel_scale = 1.98
    p.open_movie(out, framerate=fps, quality=8)
    half = n_frames // 2
    for i in range(n_frames):
        if i < half:                       # spin about the dorso-ventral axis
            th = 2 * np.pi * i / half
            pos = (np.sin(th) * 6.0, 0.0, np.cos(th) * 6.0)
            up = (0, 1, 0)
            lab = "yaw %3.0f deg" % np.degrees(th)
        else:                              # then tip over the top: ventral -> lateral -> dorsal
            th = 2 * np.pi * (i - half) / (n_frames - half)
            pos = (0.0, -np.cos(th) * 6.0, np.sin(th) * 6.0)
            up = (-1, 0, 0)
            lab = "pitch %3.0f deg" % np.degrees(th)
        p.camera_position = [pos, (0, 0, -0.1), up]
        p.camera.parallel_scale = 1.98
        txt.SetText(2, "zebrafish larval oculomotor plant, measured off Fig. 12.1A   |   " + lab)
        p.write_frame()
    p.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default="larva", choices=list(FA.PLANTS) + ["mammal"])
    ap.add_argument("--points", type=int, default=9000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--movie", action="store_true", help="turntable mp4 instead of the panels")
    a = ap.parse_args()
    print(FA.summary())
    if a.movie:
        print("wrote", movie(a.plant, a.out))
    else:
        print("wrote", render(a.plant, a.points, a.out))
