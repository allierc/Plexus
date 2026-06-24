"""Quantitative test that the active-stress operator (M1) does what it should:
a uniform contraction-AXIS field makes the sheet SHORTEN along that axis with little/no net drift,
whereas the body-force counterpart TRANSLATES the sheet (centroid drifts). Runs each spec a few
frames into the first beat and reports centroid drift + per-axis extent change %."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from plexus.schema import load
from plexus.paths import resolve_config
import plexus.engine as E

NF = 70                       # frames into the first beat (pulse peaks ~frame 30, period 150)
DEV = "cuda:0"

def measure(cfg):
    spec = load(resolve_config(cfg)[0]); spec.n_frames = NF
    _, out = E.run(spec, device=DEV)
    P = out["sets"]["mpm_particle"]["pos"]            # [NF+1, N, 2]
    p0 = P[0]; c0 = p0.mean(0)
    drift = np.linalg.norm(P.mean(1) - c0[None], axis=1).max()
    sx0, sy0 = p0[:, 0].std(), p0[:, 1].std()
    sx, sy = P[:, :, 0].std(1), P[:, :, 1].std(1)
    return drift, (sx.min()-sx0)/sx0*100, (sy.min()-sy0)/sy0*100

pairs = [("material_active_horizontal", "material_directional_horizontal", "x-axis"),
         ("material_active_vertical",   "material_directional_vertical",   "y-axis"),
         ("material_active_radial_in",  "material_directional_radial_in",  "radial"),
         ("material_active_swirl",      "material_directional_swirl",      "swirl")]

print(f"{'spec':<34}{'centroid drift':>15}{'x-extent %':>13}{'y-extent %':>13}")
print("-"*75)
for act, force, axis in pairs:
    for cfg, tag in [(act, "STRESS"), (force, "force ")]:
        try:
            d, dx, dy = measure(cfg)
            print(f"{cfg:<34}{d:>15.4f}{dx:>13.2f}{dy:>13.2f}   [{tag} {axis}]")
        except Exception as e:
            print(f"{cfg:<34}  ERR {e}")
    print()
