"""Direction (active-stress-orientation) maps + specs that use a UNIFORM stimulus and
a `vector_grid` direction map to drive a DIRECTED contraction.

Direction TIFFs (128x128x2, dx/dy in [-1,1]) -> $GraphData/.../material/dir_<name>.tif
Specs -> config/material/material_directional_<name>.yaml
  - pulse_stimulus profile: uniform   (activation a(x,t)=p(t), global)
  - pulse_to_contraction mode: directional, direction_from: direction
  - one combined cardio spec also carries a stiffness map (K + D, the two latent fields)

Stored [iy,ix] with iy=y(up); saved flipped so the `vector_grid` field's vertical flip
round-trips to world (dx,dy) at world (x,y).  Run:  python scripts/make_direction_maps.py
"""
import os
import numpy as np
import tifffile
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material"
CFG = os.path.join(REPO, "config", "material")
N = 128

ix = (np.arange(N) + 0.5) / N
iy = (np.arange(N) + 0.5) / N
X, Y = np.meshgrid(ix, iy, indexing="xy")          # X[iy,ix]=x, Y[iy,ix]=y (y up)

dirs = {}
dirs["radial_in"] = np.stack([0.5 - X, 0.5 - Y], -1)                  # contract toward centre
dirs["horizontal"] = np.stack([np.sign(0.5 - X), np.zeros_like(X)], -1)  # contract along x
dirs["vertical"] = np.stack([np.zeros_like(X), np.sign(0.5 - Y)], -1)    # contract along y
dirs["swirl"] = np.stack([-(Y - 0.5), (X - 0.5)], -1)                 # tangential (rotation/shear)

os.makedirs(MAT, exist_ok=True)
for name, d in dirs.items():
    d = d.astype("float32")
    tifffile.imwrite(os.path.join(MAT, f"dir_{name}.tif"), d[::-1].copy())   # flip so field round-trips
    print(f"[dir ] dir_{name}.tif  shape={d.shape}")

base = yaml.safe_load(open(os.path.join(CFG, "material_central_contraction.yaml")))


def make_spec(name, direction, stiffness=None):
    s = yaml.safe_load(yaml.safe_dump(base))
    s["general"]["name"] = name
    s["general"]["n_frames"] = 250
    s["fields"]["direction"] = {"frame": "vector_grid", "source": f"material/dir_{direction}.tif"}
    for o in s["operators"]:
        if o["op"] == "pulse_stimulus":
            o["profile"] = "uniform"
        if o["op"] == "pulse_to_contraction":
            o["mode"] = "directional"; o["direction_from"] = "direction"; o["amplitude"] = 25.0
    if stiffness is not None:                       # the combined K + D cardio setup
        s["fields"]["stiffness_map"] = {"frame": "image", "source": f"material/map_{stiffness}.tif"}
        s["operators"].insert(1, {"op": "apply_material_map", "at": "mpm_particle",
                                  "from": "stiffness_map", "target": "youngs", "min": 20, "max": 200})
        s["schedule"].insert(1, "apply_material_map")
    out = os.path.join(CFG, f"{name}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(s, f, sort_keys=False, default_flow_style=None)
    print(f"[spec] {os.path.relpath(out, REPO)}")


for nm in dirs:
    make_spec(f"material_directional_{nm}", nm)
make_spec("material_directional_cardio", "horizontal", stiffness="two_regions")   # K + D combined
