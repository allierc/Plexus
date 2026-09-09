"""A 4x4 contact sheet of the cell atlas at sixteen transparency settings -- pick one, then put
its two numbers in the spec.

WHY A SWEEP AND NOT A GUESS. "The cells are not transparent enough" is a judgement about a
PICTURE, and the two knobs that make it -- how much of the plasma membrane you see through, and
how much the protein crowd fogs the interior -- interact: a nearly clear membrane over a dense
protein haze looks the same as an opaque membrane over a clear one, and neither is what anyone
wants. Sixteen renders of the SAME simulation state, varying only those two, is the cheapest way
to make the choice by looking rather than by argument.

    presets  four complete looks, laid out in a square grid


ONE BUILD, SIXTEEN RENDERS. The state is seeded once and the renderer is rebuilt per panel, so
every panel is the same cells in the same places and the only difference is the material. It is
rendered at t = 0 on purpose: the cells are spheres there, so nothing about the shape distracts
from the thing being judged.

    python tools/cell_opacity_montage.py                       # 25 cells, the default sweep
    python tools/cell_opacity_montage.py --spec cell/cell_atlas_surface --divide 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import plexus.operators  # noqa: F401,E402  self-register the operator library
from plexus.schema import load  # noqa: E402
from plexus.engine import build, seed  # noqa: E402
from plexus.paths import resolve_config  # noqa: E402

# A 2x2, AND THE SWEEP RUNS THE OTHER WAY NOW.
#
# The earlier sheets pushed the membrane toward invisibility and held the organelles at full
# opacity, on the reading that the shell was in the way. The ask has inverted: the MEMBRANE is the
# thing to see, and the organelles should sit behind it. So the two axes are
#
#     rows     how much membrane there is  -- alpha and material together
#     columns  how far everything inside recedes
#
# `lead` is the nucleus and the cytoskeleton, `second` the ER / Golgi / mitochondria / chromatin /
# nucleolus / centriole at 0.7 of it, so the interior thins as a whole while keeping the ordering
# that made the nucleus and the cage readable. Backface culling stays on throughout: it is what
# stops the far wall of a shell doubling the wash, and it is worth more than any alpha choice.
_SHEER = dict(ambient=0.06, diffuse=0.45, specular=0.70, specular_power=90, backface_culling=True)
_SOLID = dict(ambient=0.12, diffuse=0.62, specular=0.55, specular_power=60, backface_culling=True)

MEMBRANE = (("sheer 0.20", 0.20, _SHEER), ("solid 0.38", 0.38, _SOLID))
INTERIOR = (("organelles 0.50", 0.50), ("organelles 0.28", 0.28))
PROTEIN_FIXED = 0.08
SECOND_SCALE = 0.7
PRESETS = tuple(
    (f"{ml}   {il}",
     dict(membrane=("surface", ma, mm), nucleus=lead, cyto=lead,
          second=round(lead * SECOND_SCALE, 3), protein=PROTEIN_FIXED))
    for ml, ma, mm in MEMBRANE
    for il, lead in INTERIOR
)
SECOND_KEYS = ("rough_er", "smooth_er", "golgi", "mitochondria", "chromatin",
               "nucleolus", "centriole")


def apply_preset(style, cfg):
    """Return `style` with one preset's opacities and materials written into it."""
    st = dict(style)
    op = dict(st.get("opacity") or {})
    surf = {k: dict(v) for k, v in (st.get("surface") or {}).items()}
    kind, alpha, mat = cfg["membrane"]
    op["plasma_membrane"] = alpha
    m = surf.setdefault("plasma_membrane", {})
    if kind == "dots":
        # A MEMBRANE THAT IS NOT A SURFACE. Its own points, at their own size -- so the boundary is
        # a shell of specks you see straight through, and no alpha is accumulated at all.
        m.clear()
        m.update(render="dots", point_size=1.3)
    else:
        m.pop("render", None)
        m.pop("point_size", None)
        m.update(mat)
    op["nucleus"] = cfg["nucleus"]
    op["cytoskeleton"] = cfg["cyto"]
    for k in SECOND_KEYS:
        op[k] = cfg["second"]
    for k in ("a", "b", "c", "d", "e"):
        op[f"protein_{k}"] = cfg["protein"]
    st["opacity"] = op
    st["surface"] = surf
    st.pop("cutaway", None)                 # these looks earn their legibility without cutting
    return st


def panel(H, sim, style, out_dir, px):
    """Render one still of `H` with `style`. Returns the image, or None if the renderer failed."""
    from plexus.live_movie import LiveMovie
    os.makedirs(out_dir, exist_ok=True)
    mov = LiveMovie(out=os.path.join(out_dir, "m.mp4"),
                    world=list(sim.world_size), n_frames=1,
                    up=int((sim.plotting or {}).get("up_axis", 2)),
                    name="", sim=sim, style=style, stills=1, keep_stills=False,
                    render_n=500_000_000, max_frames=1,
                    dt=sim.dt, length_um=(sim.units.length_um if sim.units.declared else None),
                    time_s=(sim.units.time_s if sim.units.declared else None), real_time=False)
    try:
        # TICK 0 BUILDS THE SCENE AND RETURNS WITHOUT RASTERISING -- `_frame` sets the cloud, the
        # surfaces and the camera on its first call and exits, and the still writer only fires on a
        # tick that the movie actually RENDERED. So the panel needs a second call: one to build,
        # one to draw. Calling only the first is why every panel came back "wrote nothing".
        mov(H, 0)
        mov.stride = 1
        mov.still_ticks = {1}
        mov(H, 1)
    finally:
        try:
            mov.close()
        except Exception:                                  # noqa: BLE001
            pass
    p = os.path.join(out_dir, "3d.png")
    if not os.path.exists(p):
        return None
    import imageio.v3 as iio
    img = iio.imread(p)
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def label(img, text, size=34):
    """A white label, top-left, on the panel itself -- no titles, no captions underneath."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(np.asarray(img, np.uint8))
    d = ImageDraw.Draw(im)
    font = None
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(cand):
            font = ImageFont.truetype(cand, size)
            break
    d.text((18, 96), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="cell/cell_atlas_surface")
    ap.add_argument("--out", default=os.path.join(ROOT, "graphs_data", "cell",
                                                  "cell_opacity_montage.png"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--divide", type=int, default=8,
                    help="cut every per_parent budget by this, so the sweep is minutes not hours; "
                         "transparency does not depend on the point count once a surface exists")
    ap.add_argument("--n-grid", type=int, default=160)
    ap.add_argument("--px", type=int, default=900)
    args = ap.parse_args()

    yaml_file, _pre, _name = resolve_config(args.spec)
    raw = yaml.safe_load(open(yaml_file))
    if args.divide > 1:
        pp = raw["sets"]["mpm_particle"]["per_parent"]
        if isinstance(pp, dict):
            raw["sets"]["mpm_particle"]["per_parent"] = {k: max(3, v // args.divide)
                                                         for k, v in pp.items()}
        else:
            raw["sets"]["mpm_particle"]["per_parent"] = max(3, int(pp) // args.divide)
        for f in raw["fields"].values():
            if "n_grid" in f:
                f["n_grid"] = args.n_grid
    raw["general"]["name"] = raw["general"]["name"] + "_opacity"
    raw["plotting"]["render_px"] = args.px
    tmp_spec = os.path.join(tempfile.mkdtemp(prefix="opacity_"), "spec.yaml")
    yaml.safe_dump(raw, open(tmp_spec, "w"), sort_keys=False)

    sim = load(tmp_spec)
    H = build(sim, device=args.device)
    seed(H, sim, device=args.device)

    work = tempfile.mkdtemp(prefix="opacity_panels_")
    cols = int(np.ceil(np.sqrt(len(PRESETS))))
    panels = []
    for n, (lab, cfg) in enumerate(PRESETS):
        st = apply_preset(dict(sim.plotting or {}), cfg)
        img = panel(H, sim, st, os.path.join(work, f"p{n}"), args.px)
        if img is None:
            raise RuntimeError(f"preset {lab!r} produced no image")
        panels.append(label(img, lab))
        print(f"[montage] {lab}", flush=True)
    rows = [np.concatenate(panels[r * cols:(r + 1) * cols], axis=1)
            for r in range(int(np.ceil(len(panels) / cols)))]
    sheet = np.concatenate(rows, axis=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(args.out, sheet)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[montage] {sheet.shape[1]}x{sheet.shape[0]} -> {args.out}", flush=True)
    for lab, _ in PRESETS:
        print(f"[montage]   {lab}", flush=True)

if __name__ == "__main__":
    main()
