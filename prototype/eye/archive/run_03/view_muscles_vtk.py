"""view_muscles_vtk -- the twelve extraocular muscles of the blend, in 3-D.

Reads what `read_blend.py` cut out of `260802_s2_EYE_MUSCLES_MODEL.blend`
(`blend_parts/parts.npz` + `parts.json`) and draws the MUSCLES AS SURFACES, one
colour per muscle in `fish_anatomy`'s scheme, both eyes at once. Nothing here
re-opens the blend: this is a viewer for the cut, so it runs in the ordinary
project env (pyvista/VTK 9.5), not under `bpy`.

    python view_muscles_vtk.py                       # four views -> muscles_3d.png
    python view_muscles_vtk.py --turntable 72        # + muscles_turntable.mp4
    python view_muscles_vtk.py --side L --globe none # left set alone, no eyeball
    python view_muscles_vtk.py --interactive         # a window, if you have a display

WHAT IS DRAWN, and why each thing is optional:

    muscles     the twelve strap surfaces, opaque, in the LR/SR/MR/IR/SO/IO colours
    globe       the two retina shells, translucent grey -- the muscles wrap them, so
                without it the straps float; `--globe cornea` adds the corneal cap so
                you can see WHICH WAY EACH EYE LOOKS (laterally, out of the head)
    centreline  the measured polyline of each muscle with a ball at the INSERTION
                end -- this is the geometry `read_blend` names the muscles from, so
                drawing it is how you check the naming with your eyes
    bones       the cartilage the origins sit on, very faint, off by default: 38
                plates at 45 k vertices each is most of the render time and it hides
                the muscles

Views are PARALLEL projections along the head axes (the blend's +x = animal's right,
+y = caudal, +z = dorsal), so lengths in a panel are comparable and the muscles do
not fan out with perspective.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS_DIR = os.path.join(HERE, "blend_parts")

MUSCLE_KEYS = ["LR", "SR", "MR", "IR", "SO", "IO"]
LONG_NAME = {"LR": "lateral rectus", "SR": "superior rectus", "MR": "medial rectus",
             "IR": "inferior rectus", "SO": "superior oblique", "IO": "inferior oblique"}
COLOR = {"LR": "#4da3ff", "SR": "#ff5c5c", "MR": "#ffd24d",
         "IR": "#7ee081", "SO": "#c58cff", "IO": "#ff9c42"}

# camera direction and up-vector per view, in head axes
VIEWS = {
    "frontal":  (( -1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # looking from the animal's left
    "dorsal":   ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),    # from above, rostral up
    "rostral":  ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),    # from in front of the snout
    "oblique":  ((-0.75, -0.55, 0.36), (0.0, 0.0, 1.0)),
}


def load_parts(parts_dir=PARTS_DIR):
    npz_path = os.path.join(parts_dir, "parts.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} not found -- run read_blend.py first")
    d = np.load(npz_path)
    manifest = json.load(open(os.path.join(parts_dir, "parts.json")))
    return d, manifest


def poly(d, part):
    """One part of the cut as a pyvista surface."""
    v = np.asarray(d[part + "__v"], dtype=float)
    f = np.asarray(d[part + "__f"], dtype=np.int64)
    faces = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    return pv.PolyData(v, faces)


def add_scene(p, d, manifest, sides=("L", "R"), globe="retina", centrelines=True,
              bones=False, label=None, legend=False):
    """Everything one panel shows. Returns the bounds of the muscles alone."""
    box = []
    for side in sides:
        for k in MUSCLE_KEYS:
            part = f"{side}_{k}"
            if part + "__v" not in d:
                continue
            m = poly(d, part)
            box.append(m.points)
            p.add_mesh(m, color=COLOR[k], smooth_shading=True, specular=0.25,
                       specular_power=18, show_scalar_bar=False)
            if centrelines:
                cl = np.asarray(d[part + "__centreline__v"], dtype=float)
                p.add_mesh(pv.Spline(cl, 120).tube(radius=0.008), color="white", opacity=0.55)
                p.add_mesh(pv.Sphere(radius=0.022, center=cl[0]), color="white", opacity=0.9)
        if globe in ("retina", "cornea") and f"{side}_retina__v" in d:
            p.add_mesh(poly(d, f"{side}_retina"), color="#b9b2bd", opacity=0.22,
                       smooth_shading=True, show_scalar_bar=False)
        if globe == "cornea" and f"{side}_cornea__v" in d:
            p.add_mesh(poly(d, f"{side}_cornea"), color="#9fd8ff", opacity=0.35,
                       smooth_shading=True, show_scalar_bar=False)
            p.add_mesh(poly(d, f"{side}_lens"), color="white", opacity=0.5,
                       smooth_shading=True, show_scalar_bar=False)
    if bones:
        for r in manifest["parts"]:
            if r["group"] == "bone" and r["side"] in sides:
                p.add_mesh(poly(d, r["part"]), color="#8e8494", opacity=0.10,
                           smooth_shading=True, show_scalar_bar=False)
    if label:
        p.add_text(label, position="upper_left", font_size=11, color="white")
    if legend:
        p.add_legend(labels=[[f"{k}  {LONG_NAME[k]}", COLOR[k]] for k in MUSCLE_KEYS],
                     bcolor=None, border=False, face="rectangle",
                     loc="lower left", size=(0.30, 0.24))
    return np.concatenate(box) if box else np.zeros((1, 3))


def set_view(p, name, centre, span):
    d, up = VIEWS[name]
    d = np.asarray(d, float)
    p.camera_position = (tuple(centre + d * 10.0), tuple(centre), up)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = span


def render_panels(d, manifest, out_png, sides, globe, centrelines, bones,
                  size=(1900, 1400)):
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, shape=(2, 2), border=False)
    order = ["frontal", "dorsal", "rostral", "oblique"]
    labels = {"frontal": "A   frontal (from the animal's left)",
              "dorsal": "B   dorsal (rostral up)",
              "rostral": "C   rostral (from the snout)",
              "oblique": "D   oblique"}
    for idx, view in enumerate(order):
        p.subplot(idx // 2, idx % 2)
        p.set_background("black")
        pts = add_scene(p, d, manifest, sides=sides, globe=globe,
                        centrelines=centrelines, bones=bones, label=labels[view],
                        legend=(view == "frontal"))
        centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        span = float((pts.max(axis=0) - pts.min(axis=0)).max()) * 0.62
        set_view(p, view, centre, span)
    p.screenshot(out_png)
    p.close()
    return out_png


def render_turntable(d, manifest, out_mp4, sides, globe, centrelines, bones,
                     n_frames=72, size=(1280, 960), fps=24):
    import imageio.v2 as iio

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, border=False)
    p.set_background("black")
    pts = add_scene(p, d, manifest, sides=sides, globe=globe,
                    centrelines=centrelines, bones=bones,
                    label="six muscles per eye, 96 hpf zebrafish")
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    span = float((pts.max(axis=0) - pts.min(axis=0)).max()) * 0.62

    with iio.get_writer(out_mp4, fps=fps, quality=8, macro_block_size=None) as w:
        for k in range(n_frames):
            th = 2.0 * np.pi * k / n_frames
            # orbit in the horizontal plane, tilted 20 deg above it
            direction = np.array([np.cos(th), np.sin(th), 0.36])
            direction /= np.linalg.norm(direction)
            p.camera_position = (tuple(centre + direction * 10.0), tuple(centre), (0, 0, 1))
            p.camera.parallel_projection = True
            p.camera.parallel_scale = span
            w.append_data(p.screenshot(return_img=True))
    p.close()
    return out_mp4


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default=PARTS_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "muscles_3d.png"))
    ap.add_argument("--side", default="both", choices=("L", "R", "both"))
    ap.add_argument("--globe", default="cornea", choices=("retina", "cornea", "none"))
    ap.add_argument("--no-centrelines", action="store_true")
    ap.add_argument("--bones", action="store_true", help="add the cartilage, faintly")
    ap.add_argument("--turntable", type=int, default=0, metavar="N",
                    help="also write an N-frame turntable mp4")
    ap.add_argument("--interactive", action="store_true", help="open a window instead")
    args = ap.parse_args()

    d, manifest = load_parts(args.parts)
    sides = ("L", "R") if args.side == "both" else (args.side,)
    kw = dict(sides=sides, globe=args.globe, centrelines=not args.no_centrelines,
              bones=args.bones)

    if args.interactive:
        p = pv.Plotter(window_size=(1400, 1000))
        p.set_background("black")
        pts = add_scene(p, d, manifest, label="extraocular muscles", **kw)
        centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        set_view(p, "oblique", centre, float((pts.max(axis=0) - pts.min(axis=0)).max()) * 0.62)
        p.show()
        return

    png = render_panels(d, manifest, args.out, **kw)
    print(f"wrote {png}")
    if args.turntable:
        mp4 = os.path.splitext(args.out)[0] + "_turntable.mp4"
        print(f"wrote {render_turntable(d, manifest, mp4, n_frames=args.turntable, **kw)}")

    for r in manifest["parts"]:
        if r["group"] == "muscle" and r["side"] in sides:
            print(f"  {r['part']:6s} {r['long_name']:18s} length {r['length']:.3f}  "
                  f"insertion (caud,dors,lat) {r['insertion_eye']}  "
                  f"rotation axis {r['rot_axis_eye']}")


if __name__ == "__main__":
    main()
