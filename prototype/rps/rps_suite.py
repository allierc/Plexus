"""Generate a large spread of RPS reaction-diffusion specs + gifs to chart how the
one `reaction_diffusion` operator behaves across its parameter space.

    python rps_suite.py            # write every scenarios/<name>.yaml, run, render gif+montage
    python rps_suite.py D_080 ...  # only the named configs

Each config is BASE + a few overrides -> one spec.yaml -> one gif. Sweeps cover
initial condition, diffusion D (spiral wavelength), competition a, species count,
boundary, and grid scale. A gallery_rps.png tiles a late frame of every run.
"""
import os
import sys
import numpy as np
import yaml
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rps_engine
import rps_render

BASE = dict(grid=256, species=3, a=0.6, D=0.5, dt=0.3, bc="periodic",
            steps=5000, record_every=33, init="coexist", seed=0)

# group -> [(name, overrides)]; dt=0.3 keeps D*dt < 0.25 (Euler-stable) for D <= 0.8.
GROUPS = {
    "initial condition (D=0.5, a=0.6, S=3)": [
        ("rps_random",     {"init": "random"}),
        ("rps_coexist",    {"init": "coexist"}),
        ("rps_blob",       {"init": "blob"}),
        ("rps_two_blobs",  {"init": "two_blobs"}),
        ("rps_pinwheel",   {"init": "pinwheel"}),
        ("rps_spots",      {"init": "spots"}),
        ("rps_stripes",    {"init": "stripes"}),
        ("rps_corners",    {"init": "corners"}),
        ("rps_ring",       {"init": "ring"}),
        ("rps_half",       {"init": "half"}),
    ],
    "diffusion D -> spiral wavelength (init=coexist)": [
        ("rps_D_005", {"D": 0.05}),
        ("rps_D_010", {"D": 0.10}),
        ("rps_D_020", {"D": 0.20}),
        ("rps_D_040", {"D": 0.40}),
        ("rps_D_080", {"D": 0.80}),
    ],
    "competition a -> coexistence vs turbulence (init=coexist)": [
        ("rps_a_020", {"a": 0.2}),
        ("rps_a_040", {"a": 0.4}),
        ("rps_a_060", {"a": 0.6}),
        ("rps_a_080", {"a": 0.8}),
        ("rps_a_100", {"a": 1.0}),
        ("rps_a_120", {"a": 1.2}),
    ],
    "species count (cyclic N, init=pinwheel)": [
        ("rps_species_3", {"species": 3, "init": "pinwheel"}),
        ("rps_species_4", {"species": 4, "init": "pinwheel"}),
        ("rps_species_5", {"species": 5, "init": "pinwheel"}),
        ("rps_species_6", {"species": 6, "init": "pinwheel"}),
        ("rps_species_5_turb", {"species": 5, "init": "random", "a": 0.8}),
    ],
    "boundary": [
        ("rps_wall_blob",   {"bc": "wall", "init": "blob"}),
        ("rps_wall_random", {"bc": "wall", "init": "random"}),
    ],
    "grid scale (number of spiral cores)": [
        ("rps_grid_128", {"grid": 128, "init": "random"}),
        ("rps_grid_384", {"grid": 384, "init": "random", "record_every": 40, "steps": 6000}),
    ],
    "showcase": [
        ("rps_single_spiral", {"init": "pinwheel", "D": 0.7, "steps": 4000}),
        ("rps_turbulence",    {"grid": 384, "D": 0.18, "a": 0.9, "init": "random",
                               "steps": 6000, "record_every": 40}),
        ("rps_coarsen",       {"a": 0.3, "init": "random", "steps": 6000, "record_every": 40}),
    ],
}


def all_configs():
    out = []
    for group, items in GROUPS.items():
        for name, ov in items:
            out.append((group, name, {**BASE, **ov}))
    return out


def main(only=None):
    sdir = os.path.join(HERE, "scenarios"); os.makedirs(sdir, exist_ok=True)
    tiles = []
    for group, name, spec in all_configs():
        if only and name not in only:
            continue
        with open(os.path.join(sdir, name + ".yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
        out = rps_engine.run(spec, device="cuda")
        frames = out["frames"]
        rps_render.render_frames(name, frames, fps=25, out_dir=HERE)
        tiles.append((name, rps_render.to_rgb(frames[int(frames.shape[0] * 0.85)])))
        print(f"[done] {name:22s} S={spec['species']} D={spec['D']} a={spec['a']} "
              f"init={spec['init']} bc={spec['bc']}  T={frames.shape[0]}", flush=True)
    if tiles and not only:
        _gallery(tiles, os.path.join(HERE, "gallery_rps.png"))


def _gallery(tiles, path, cols=6, cell=180):
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "black")
    d = ImageDraw.Draw(canvas)
    for i, (name, rgb) in enumerate(tiles):
        im = Image.fromarray((rgb * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST)
        x, y = (i % cols) * cell, (i // cols) * cell
        canvas.paste(im, (x, y))
        d.text((x + 4, y + 4), name.replace("rps_", ""), fill=(255, 255, 255))
    canvas.save(path)
    print(f"[gallery] {path}  ({len(tiles)} sims)")


if __name__ == "__main__":
    main(only=set(sys.argv[1:]) or None)
