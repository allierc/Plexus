"""Phase-delay maps + 4 specs that combine an ACTIVE-STRESS tensor with a PHASE DELAY.

Delay TIFFs (128x128, float [0,1]) -> $GraphData/graphs_data/material/delay_<name>.tif
Specs -> config/material/material_active_phase_<name>.yaml

Each spec is the matching `material_active_<variant>` (active-stress sigma_a = A a nn^T with a
`vector_grid` contraction-axis map) with its (pacemaker + pulse_stimulus) pair REPLACED by a
single `phase_delay_pulse`: the activation field becomes a(x,y,t)=pulse(t - max_delay*delay(x,y)),
a travelling wave whose timing gradient the active stress converts into curved / rotating motion.

Delay maps are authored in image space `[row=y(top->bottom), col=x]`, normalised [0,1]; the
`image` field flips vertical so image-top -> domain-top (same convention as the stiffness maps).
The 4th spec reuses the existing `map_four_quadrants.tif` as a delay map -- a delay map is just a
TIFF, exactly like a stiffness map.  Run:  python scripts/make_delay_maps.py
"""
import os
import numpy as np
import tifffile
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material"
CFG = os.path.join(REPO, "config", "material")
N = 128

yy, xx = np.mgrid[0:N, 0:N]                    # yy=row (y, top->bottom), xx=col (x)
cx = cy = (N - 1) / 2.0

maps = {}
maps["x"] = np.tile(np.linspace(0.0, 1.0, N, dtype="float32")[None, :], (N, 1))   # tau ~ x  (left fires first)
r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
maps["radial"] = (r / r.max()).astype("float32")                                  # tau ~ r  (centre first, rim lags)
th = np.arctan2(yy - cy, xx - cx)
maps["swirl"] = ((th + np.pi) / (2 * np.pi)).astype("float32")                    # tau ~ angle -> rotating wave

os.makedirs(MAT, exist_ok=True)
for name, arr in maps.items():
    tifffile.imwrite(os.path.join(MAT, f"delay_{name}.tif"), arr.astype("float32"))
    print(f"[tiff] delay_{name}.tif  range [{arr.min():.2f}, {arr.max():.2f}]")

# --- 4 specs: active-stress base (variant) + a phase-delay activation ----------------------- #
# (base variant, delay source, max_delay ticks)
SPECS = [
    ("horizontal", "material/delay_x.tif",              60.0),   # contract along x, wave sweeps L->R
    ("radial",     "material/delay_radial.tif",         60.0),   # base radial_in, wave centre->rim
    ("swirl",      "material/delay_swirl.tif",         150.0),   # tangential axis + angular wave -> rotary
    ("quadrants",  "material/map_four_quadrants.tif",   60.0),   # delay-from-a-TIFF, like stiffness
]
BASE = {"horizontal": "material_active_horizontal", "radial": "material_active_radial_in",
        "swirl": "material_active_swirl", "quadrants": "material_active_swirl"}


def build(name, delay_src, max_delay):
    base = yaml.safe_load(open(os.path.join(CFG, f"{BASE[name]}.yaml")))
    s = yaml.safe_load(yaml.safe_dump(base))
    s["general"]["name"] = f"material_active_phase_{name}"
    s["fields"]["delay_map"] = {"frame": "image", "source": delay_src}
    # read period/duration off the pacemaker so the wave keeps the base beat, then drop the
    # (pacemaker + pulse_stimulus) pair and paint the activation with phase_delay_pulse instead.
    pm = next((o for o in s["operators"] if o.get("op") == "pacemaker"), {})
    period, duration = float(pm.get("period", 150)), float(pm.get("duration", 30))
    s["operators"] = [o for o in s["operators"] if o.get("op") not in ("pacemaker", "pulse_stimulus")]
    pdp = {"op": "phase_delay_pulse", "at": "activation", "delay_from": "delay_map",
           "max_delay": max_delay, "period": period, "duration": duration}
    idx = next(i for i, o in enumerate(s["operators"]) if o.get("op") == "pulse_to_active_stress")
    s["operators"].insert(idx, pdp)
    s["schedule"] = ["phase_delay_pulse" if x in ("pacemaker", "pulse_stimulus") else x
                     for x in s["schedule"] if x != "pulse_stimulus"]
    out = os.path.join(CFG, f"material_active_phase_{name}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(s, f, sort_keys=False, default_flow_style=None)
    print(f"[spec] {os.path.relpath(out, REPO)}  (delay={delay_src}, max_delay={max_delay})")


for name, src, md in SPECS:
    build(name, src, md)
