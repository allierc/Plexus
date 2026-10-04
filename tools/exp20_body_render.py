"""exp20: the larval zebrafish body (WHOLISTIC whole-body model, data/anatomy/wholebody.npz from tools/exp20_body_export.py)
with the UV TARGET REGIONS the paper names, for the deck's anatomy slide (Cedric, 2026-10-03: "pinpoint the sites on a
fish anatomy"). Lateral and dorsal views, head LEFT (exp17's convention), black, off-screen pyvista.

    env -u DISPLAY PYVISTA_OFF_SCREEN=true PYTHONPATH=src python tools/exp20_body_render.py

THE MARKERS ARE THE PAPER'S REGIONS, NOT MEASURED BEAM POSITIONS: off the fish (control; Results p.2: "outside the
fish, on the tail, or on the swim bladder"), foregut and midgut (Fig. 1b, 2e), the hepatic portal system (Fig. 5a:
"the vascular bed draining the gut near the liver"). The deposit's site labels (ch_gpos) are not calibrated to the body
(the same site sits at different galvo voltages fish to fish; different sites share one), so which label is which region
is an inference the slide states as such. Each marker sits on the model's own organ: foregut / midgut = the gut's centre
in the anterior / middle third of its intestinal length (behind the oesophagus), portal = the liver's point nearest the gut.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
NPZ = os.path.join(EXP, "data", "anatomy", "wholebody.npz")
COL = {"skin": ("#9e9e9e", 0.10), "gut": ("#ff8a65", 0.85), "liver": ("#8d4a3a", 0.85), "pancreas": ("#d7ccc8", 0.6),
       "swim_bladder": ("#e0e0e0", 0.35), "heart": ("#e53935", 0.9), "eye": ("#424242", 0.9),
       "kidney": ("#7e57c2", 0.35), "notochord": ("#bdbdbd", 0.25), "brain": ("#4fc3f7", 0.8)}
MARK = {"off the fish (control)": "#bdbdbd", "foregut": "#ff4040", "midgut": "#ffd54f",
        "hepatic portal system": "#66bb6a"}


def markers(z):
    g = z["gut__v"]
    gi = g[g[:, 0] > 4.4]                                  # the intestine, behind the oesophagus and the liver's front
    lo, hi = gi[:, 0].min(), gi[:, 0].max()
    seg = lambda a, b: gi[(gi[:, 0] >= lo + a * (hi - lo)) & (gi[:, 0] <= lo + b * (hi - lo))].mean(0)
    L = z["liver__v"]
    d = np.linalg.norm(L[:, None, :] - gi[None, ::50, :], axis=2).min(1)
    portal = L[np.argsort(d)[:200]].mean(0)
    sk = z["skin__v"]
    trunk = sk[(sk[:, 0] > 9) & (sk[:, 0] < 10)]
    off = np.array([9.5, 0.0, trunk[:, 2].max() + 0.9])   # above the trunk, outside the skin
    br = z["brain__v"]
    return {"foregut": seg(0.05, 0.30), "midgut": seg(0.40, 0.65), "hepatic portal system": portal,
            "off the fish (control)": off}


def main():
    import pyvista as pv
    pv.OFF_SCREEN = True
    z = np.load(NPZ)
    M = markers(z)
    meshes = {}
    for k in COL:
        if f"{k}__v" in z:
            f = z[f"{k}__f"]
            meshes[k] = pv.PolyData(z[f"{k}__v"], np.c_[np.full(len(f), 3), f].ravel())
    out = []
    for view in ("lateral", "dorsal"):
        p = pv.Plotter(off_screen=True, window_size=(2400, 900))
        p.set_background("black")
        for k, m in meshes.items():
            c, a = COL[k]
            p.add_mesh(m, color=c, opacity=a, smooth_shading=True, specular=0.2)
        for k, x in M.items():                              # drawn in FRONT of the organs (toward the camera)
            x = np.asarray(x) + (np.array([0, -1.6, 0]) if view == "lateral" else np.array([0, 0, 1.8]))
            p.add_mesh(pv.Sphere(radius=0.24, center=x), color=MARK[k], opacity=1.0)
        # ONE CAMERA FRAME FOR BOTH VIEWS (Cedric, 2026-10-03: aligned, head not cut): the same centre and the same
        # parallel scale, the whole fish (x -0.5..21.3) plus a margin across the window's width
        cx = np.array([10.4, 0.0, 0.0])
        if view == "lateral":                              # camera on -y looking +y: head (x ~ 0) LEFT, dorsal up
            p.camera_position = [(cx[0], -45.0, 0.0), cx, (0, 0, 1)]
        else:                                              # from above, head left
            p.camera_position = [(cx[0], 0.0, 45.0), cx, (0, 1, 0)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 23.0 / 2 / (2400 / 900)
        png = os.path.join(EXP, "data", "anatomy", f"body_{view}.png")
        p.screenshot(png)
        p.close()
        out.append(png)
        print("->", png)
    np.save(os.path.join(EXP, "data", "anatomy", "markers.npy"), {k: v.tolist() for k, v in M.items()}, allow_pickle=True)


SITE_KINDS = {"off": "off the fish (control)", "gutA": "foregut", "gutB": "midgut",
              "vessel": "hepatic portal system"}              # blood glucose (batch 4): the paper's Fig. 5a spot


def render_sites(kinds=None):
    """One SMALL lateral diagram per kind of site, only its marker drawn (the deck's per-site slides): off the fish,
    the gut spot of site 2 (and 3), the other gut spot of fish 4's site 5. Region names are the paper's; which label
    went where is inferred (tools/exp20_slides.py SITE_REGION)."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    z = np.load(NPZ)
    M = markers(z)
    for kind, mk in SITE_KINDS.items():
        if kinds and kind not in kinds:
            continue
        p = pv.Plotter(off_screen=True, window_size=(1200, 420))
        p.set_background("black")
        for k in ("skin", "gut", "liver", "swim_bladder", "brain", "eye", "heart"):
            if f"{k}__v" in z:
                f = z[f"{k}__f"]
                c, a = COL[k]
                p.add_mesh(pv.PolyData(z[f"{k}__v"], np.c_[np.full(len(f), 3), f].ravel()), color=c, opacity=a,
                           smooth_shading=True)
        p.add_mesh(pv.Sphere(radius=0.45, center=np.asarray(M[mk]) + np.array([0, -1.6, 0])), color="#ffeb3b")
        cx = np.array([10.4, 0.0, 0.0])
        p.camera_position = [(cx[0], -45.0, 0.0), cx, (0, 0, 1)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 23.0 / 2 / (1200 / 420)
        p.screenshot(os.path.join(EXP, "data", "anatomy", f"site_{kind}.png"))
        p.close()
        print("-> site", kind)


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "sites":
        render_sites(sys.argv[2:] or None)
    else:
        main()
        render_sites()
