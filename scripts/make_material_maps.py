"""Create 6 test stiffness-map TIFFs + 6 specs that apply each via `apply_material_map`.

TIFFs (128x128, float [0,1]) -> $GraphData/graphs_data/material/map_<name>.tif
Specs  -> config/material/material_map_<name>.yaml  (central contraction + a stiffness map)

Run:  python scripts/make_material_maps.py
"""
import os
import numpy as np
import tifffile
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material"
CFG = os.path.join(REPO, "config", "material")
N = 128

# image arrays are [row=y (top->bottom), col=x]; the `image` field flips vertical so
# image-top maps to domain-top. Values are normalised by the field, but we keep [0,1].
maps = {}
rng = np.random.default_rng(0)
maps["random"] = rng.random((N, N)).astype("float32")

two = np.full((N, N), 0.15, "float32"); two[:, N // 2:] = 1.0          # left soft | right stiff
maps["two_regions"] = two

quad = np.zeros((N, N), "float32")                                     # four stiffness quadrants
quad[:N // 2, :N // 2] = 0.20; quad[:N // 2, N // 2:] = 0.50           # TL, TR
quad[N // 2:, :N // 2] = 0.75; quad[N // 2:, N // 2:] = 1.00           # BL, BR
maps["four_quadrants"] = quad

grad = np.tile(np.linspace(0, 1, N, dtype="float32")[None, :], (N, 1))  # soft left -> stiff right
maps["gradient"] = grad

yy, xx = np.mgrid[0:N, 0:N]
r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2) / (N / 2)
maps["circle"] = np.clip(1.0 - r, 0.05, 1.0).astype("float32")        # stiff disc centre -> soft edge

stripes = np.where(((xx // 16) % 2) == 0, 1.0, 0.15).astype("float32")  # vertical stiff/soft stripes
maps["stripes"] = stripes

os.makedirs(MAT, exist_ok=True)
os.makedirs(CFG, exist_ok=True)
for name, arr in maps.items():
    tifffile.imwrite(os.path.join(MAT, f"map_{name}.tif"), arr)
    print(f"[tiff] map_{name}.tif  range [{arr.min():.2f}, {arr.max():.2f}]")

# --- 6 specs: central contraction + a per-particle stiffness map -------------- #
base = yaml.safe_load(open(os.path.join(CFG, "material_central_contraction.yaml")))
for name in maps:
    s = yaml.safe_load(yaml.safe_dump(base))           # deep copy
    s["general"]["name"] = f"material_map_{name}"
    s["general"]["n_frames"] = 250
    s["fields"]["stiffness_map"] = {"frame": "image", "source": f"material/map_{name}.tif"}
    # insert apply_material_map right after aggregate, before the pulse/mechanics ops
    ops = s["operators"]
    amm = {"op": "apply_material_map", "at": "mpm_particle", "from": "stiffness_map",
           "target": "youngs", "min": 20, "max": 200}
    ops.insert(1, amm)
    s["schedule"].insert(1, "apply_material_map")
    out = os.path.join(CFG, f"material_map_{name}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(s, f, sort_keys=False, default_flow_style=None)
    print(f"[spec] {os.path.relpath(out, REPO)}")
