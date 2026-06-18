"""Mix the RPS field with every other operator family, one token at a time.

    python rps_mix_suite.py [name ...]

Each config is the RPS field (react+diffuse) plus a different coupling, to show the
operator algebra composes: advection (field<-field), chemotaxis & flocking
(set<-field, +rewire), and grazing (set->field).
"""
import os
import sys
import numpy as np
import yaml
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rps_mix_engine
import rps_mix_render
import rps_render

BASE = dict(grid=256, species=3, a=0.6, D=0.5, dt=0.3, bc="periodic",
            steps=4200, record_every=30, init="random", seed=0,
            flow="none", flow_strength=0.0, particles=0)

CONFIGS = [
    # --- field <- field : stir the species fields with a frozen flow ---
    ("rpsmix_swirl",     {"flow": "swirl",     "flow_strength": 5.0}),
    ("rpsmix_shear",     {"flow": "shear",     "flow_strength": 4.0}),
    ("rpsmix_turbulent", {"flow": "turbulent", "flow_strength": 4.0}),
    # --- set <- field : particles read the species field ---
    ("rpsmix_taxis",     {"particles": 1400, "particle": {"channel": 0, "chemotaxis": 14.0, "follow_flow": 0.0}}),
    ("rpsmix_flock",     {"flow": "swirl", "flow_strength": 5.0, "particles": 1400,
                          "particle": {"align": 0.85, "speed": 0.012, "radius": 0.05, "follow_flow": 1.0}}),
    ("rpsmix_drift",     {"flow": "turbulent", "flow_strength": 4.0, "particles": 1600,
                          "particle": {"follow_flow": 1.0}}),
    # --- set <-> field : particles chemotax to a species AND graze it down (feedback) ---
    ("rpsmix_graze",     {"particles": 1500,
                          "particle": {"channel": 0, "chemotaxis": 16.0, "graze": 0.9, "follow_flow": 0.0}}),
    # --- everything at once ---
    ("rpsmix_all",       {"flow": "turbulent", "flow_strength": 4.0, "particles": 1500,
                          "particle": {"channel": 0, "chemotaxis": 10.0, "align": 0.7, "speed": 0.008,
                                       "radius": 0.05, "graze": 0.5, "follow_flow": 0.8}}),
]


def main(only=None):
    sdir = os.path.join(HERE, "scenarios"); os.makedirs(sdir, exist_ok=True)
    tiles = []
    for name, ov in CONFIGS:
        if only and name not in only:
            continue
        spec = {**BASE, **ov}
        with open(os.path.join(sdir, name + ".yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
        out = rps_mix_engine.run(spec, device="cuda")
        rps_mix_render.render_mix(name, out, fps=25, out_dir=HERE)
        f = out["frames"]; tiles.append((name, rps_render.to_rgb(f[int(len(f) * 0.85)])))
        print(f"[done] {name:18s} flow={spec['flow']:9s} P={spec['particles']}", flush=True)
    if tiles and not only:
        _gallery(tiles, os.path.join(HERE, "gallery_rpsmix.png"))


def _gallery(tiles, path, cols=4, cell=200):
    rows = (len(tiles) + cols - 1) // cols
    c = Image.new("RGB", (cols * cell, rows * cell), "black"); d = ImageDraw.Draw(c)
    for i, (name, rgb) in enumerate(tiles):
        im = Image.fromarray((rgb * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST)
        x, y = (i % cols) * cell, (i // cols) * cell
        c.paste(im, (x, y)); d.text((x + 4, y + 4), name.replace("rpsmix_", ""), fill=(255, 255, 255))
    c.save(path); print(f"[gallery] {path}  ({len(tiles)} sims)")


if __name__ == "__main__":
    main(only=set(sys.argv[1:]) or None)
