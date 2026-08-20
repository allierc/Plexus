"""One-off: g) muscle-routing-top-view at 9 roll angles, side by side, so a rotation can be
picked by eye instead of by guessing signs one render at a time. Not part of the pipeline."""
import os
import sys

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import eye_anatomy as EA
import blend_mpm_ops as BM
from render_surface_vtk import Skin, _poly, GLOBE_ALPHA

BLEND = "260802_s2_EYE_MUSCLES_MODEL 2.blend"
PARTS = "archive/eye_H/blend_parts"
SIDE = "L"
N_BINS = 14


def _traced(pos, s, n_bins=N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        if m.any():
            pts.append(pos[m].mean(axis=0))
    return np.asarray(pts, float)


cap = {k: v for k, v in np.load("archive/eye_H/_smoke_axial_curves.npz").items()}
d, man = BM.load_cut(BLEND, PARTS)
fr = BM.BlendFrame(man, d, SIDE, EA.A_EQ, EA.GLOBE_CENTER, 1.0)
mus_parent = np.asarray(cap["mus_parent"])
mus_s = np.asarray(cap["mus_s"])
shell0 = np.asarray(cap["shell"][0], float)
mus0 = np.asarray(cap["mus_pos"][0], float)
c = np.asarray(cap["centre"][0], float)

bone_pts = [fr(d[f'{r["part"]}__v']) for r in man["parts"]
            if r["group"] == "bone" and f'{r["part"]}__v' in d]
bone = np.concatenate(bone_pts, axis=0)

up = np.array([0.0, 1.0, 0.0])
eye_reach = float(np.abs(np.concatenate([shell0, mus0]) - c[None, :]).max())
near = bone[np.linalg.norm(bone - c[None, :], axis=1) < 4.0 * eye_reach]
extent = (near if len(near) else bone).max(0) - (near if len(near) else bone).min(0)
extent[1] = -1.0
body = np.zeros(3)
body[int(np.argmax(extent))] = 1.0
lateral = np.cross(body, up)
lateral /= np.linalg.norm(lateral)
span = 1.3 * eye_reach                              # tighter than the mirrored version (1.45x
                                                     # + mirror_dist), still clears the origins

ANGLES = [0, 5, 10, 15, 20, 25, 30, 35, 40]

pv.OFF_SCREEN = True
p = pv.Plotter(off_screen=True, window_size=(2100, 2100), shape=(3, 3), border=True,
              border_color="white")

for i, ang in enumerate(ANGLES):
    r, cix = divmod(i, 3)
    p.subplot(r, cix)
    p.set_background("black")
    p.add_mesh(pv.PolyData(bone), color="#8e8494", opacity=0.35, point_size=2.0,
              render_points_as_spheres=True, show_scalar_bar=False)
    for part, alpha in (("retina", GLOBE_ALPHA), ("cornea", 0.20), ("lens", 0.25)):
        key = f"{SIDE}_{part}"
        if f"{key}__v" not in d:
            continue
        V = fr.globe(d[f"{key}__v"])
        p.add_mesh(_poly(V, d[f"{key}__f"]), color="#cfcfd6", opacity=alpha,
                  smooth_shading=True, show_scalar_bar=False)
    for mi, key in enumerate(EA.MUSCLE_KEYS):
        own = mus_parent == mi
        if own.sum() < 3:
            continue
        pts0 = _traced(mus0[own], mus_s[own])
        if len(pts0) < 2:
            continue
        p.add_mesh(pv.lines_from_points(pts0), color=EA.MUSCLES[mi]["color"], line_width=4,
                  show_scalar_bar=False)

    roll = np.radians(float(ang))                   # other direction: sign flipped from before
    lr = -lateral
    view_up = (lr * np.cos(roll) + np.cross(up, lr) * np.sin(roll)
              + up * np.dot(up, lr) * (1.0 - np.cos(roll)))
    p.camera_position = (tuple(c + up * 10.0), tuple(c), tuple(view_up))
    p.camera.parallel_projection = True
    p.camera.parallel_scale = span
    p.add_text(f"{ang} deg", position="upper_left", font_size=14, color="yellow")

out = "archive/eye_H/_montage_g.png"
p.screenshot(out)
p.close()
print(out)
