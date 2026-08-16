#!/usr/bin/env python
"""fig_eyeG_anterior -- eye G at rest, drawn by the surface renderer, with labels.

    python fig_eyeG_anterior.py

Panel (a) of the note's symbol figure used to be a schematic: a circle with six
straps placed from `eye_anatomy.MUSCLES`, the analytic table the earlier eyes were
generated from. Eye G is not generated, it is scanned, and its muscles do not sit
where that table puts them -- the obliques in particular run from the rostral orbit
with no trochlea, which is why SO elevates here and IO depresses, the reverse of the
mammalian arrangement the schematic implied.

So this draws the real thing: `render_surface_vtk.SurfaceScene`, the same skinned
Blender meshes the movies use, viewed down the optic axis at rest, with each strap
labelled at its own insertion. The label positions are projected from the muscle
particles rather than placed by hand, so they cannot drift from the geometry.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MUS = ["LR", "SR", "MR", "IR", "SO", "IO"]
# what each muscle does on this plant, measured in stage 0 rather than assumed
ACTION = {"LR": "temporal", "MR": "nasal", "SR": "up", "IR": "down",
          "SO": "torsion +", "IO": "torsion -"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye", default=os.path.join(HERE, "archive", "eye_G"))
    p.add_argument("--az", type=float, default=16.0, help="0 = down the optic axis")
    p.add_argument("--el", type=float, default=10.0)
    p.add_argument("--size", type=int, nargs=2, default=[1100, 1100])
    p.add_argument("--out", default=os.path.join(HERE, "fig_eyeG_anterior.png"))
    a = p.parse_args()

    import pyvista as pv
    import render_surface_vtk as RS
    pv.OFF_SCREEN = True

    z = np.load(os.path.join(a.eye, "baseline_curves.npz"), allow_pickle=True)
    n = 1
    cap = {k: z[k][:n] if z[k].ndim and k in
           ("shell", "mus_pos", "centre", "act", "gaze", "target", "frame")
           else z[k] for k in z.files}
    cap["frame"] = np.array([0])
    scene = RS.SurfaceScene(cap, side="R", size=tuple(a.size), globe_alpha=0.16)
    scene.frame(0, a.az, 0.003)
    scene.span *= 1.18          # room for the labels outside the straps

    # labels at each strap's own tip, taken from the particles
    mp = np.asarray(z["mus_pos"][0], float)
    par = np.asarray(z["mus_parent"], int)
    c = np.asarray(z["centre"][0], float)
    # A label placed radially in 3-D can land behind the globe, because "outward"
    # in space is not outward on screen. Push each one out in the IMAGE PLANE
    # instead: strip the component along the view direction and grow what is left.
    e_, a_ = np.radians(a.el), np.radians(a.az)
    view = np.array([np.sin(a_) * np.cos(e_), np.sin(e_), np.cos(a_) * np.cos(e_)])
    R = float(np.linalg.norm(np.asarray(z["shell"][0], float) - c, axis=1).max())
    pts, lab = [], []
    for k, m in enumerate(MUS):
        q = mp[par == k]
        if not len(q):
            continue
        # the most peripheral point ON SCREEN, not in space: eye G's straps all
        # converge at the rostral orbit, so "furthest in 3-D" picks the same place
        # for several of them and the labels pile up
        vv = q - c
        pp = vv - np.outer(vv @ view, view)
        far = q[np.argmax(np.linalg.norm(pp, axis=1))]
        v = far - c
        perp = v - np.dot(v, view) * view
        n = np.linalg.norm(perp)
        pts.append(far + (0.30 * R / n) * perp if n > 1e-9 else far)
        lab.append(f"{m}   {ACTION[m]}")
    # VTK culls labels that collide, which silently dropped three of the six. The
    # positions are projected here and written out instead, so the caller places
    # them in matplotlib where nothing is hidden.
    fwd = -view
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    xy = {}
    for q, m in zip(pts, MUS):
        v = np.asarray(q) - c
        xy[m] = [float(0.5 + 0.5 * np.dot(v, right) / scene.span),
                 float(0.5 - 0.5 * np.dot(v, up) / scene.span)]
    json.dump({"labels": xy, "action": ACTION}, open(a.out.replace(".png", ".json"), "w"),
              indent=2)
    # the renderer's own HUD is for the movies; a still in a figure carries its
    # information in the caption instead
    scene._text("", "")
    scene.camera(a.az, a.el)
    scene.p.render()
    img = np.asarray(scene.p.screenshot(return_img=True))
    import imageio.v2 as iio
    iio.imwrite(a.out, img)
    scene.close()
    print(f"{len(xy)} label positions; wrote {a.out}  {img.shape}")
    for m, (x, y) in xy.items(): print(f"   {m}  ({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
