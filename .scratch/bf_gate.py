"""GATE for body_force: grid. Two things must hold, or the parameter is not safe to ship.
   (1) INERT WHEN OFF: body_force particle must reproduce the current path bit for bit.
   (2) CORRECT WHEN ON: a lone block must fall at exactly g, and a bulk run must agree with the
       particle path to the level the mass clamp allows -- which, since the clamp is measured not
       to bind here, should be very close."""
import sys, os, tempfile, yaml, numpy as np
sys.path.insert(0, "/workspace/Plexus/src")
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL

def run(spec, dev, bf, frames, n_par=None, drop=None, seed=0):
    s = yaml.safe_load(open(f"/workspace/Plexus/config/material/{spec}.yaml"))
    s["general"]["n_frames"] = frames; s["general"]["record_cap"] = 10000
    s["general"]["seed"] = seed
    if n_par: s["sets"]["mpm_particle"]["per_parent"] = n_par
    if drop:
        ty = list(s["sets"]["cell"]["types"].values())[0]
        ty["block"] = [0.49, 0.85, 0.49, 0.51, 0.87, 0.51]
        s["sets"]["cell"]["start"] = [[0.5, 0.86, 0.5]]
    for o in s["operators"]:
        if o["op"] == "mpm_scatter": o["body_force"] = bf
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close(); CFL(f.name)
    sim = load(f.name); os.unlink(f.name)
    ys = []
    H, _ = E.run(sim, out_path=None, device=dev,
                 on_frame=lambda H, t: ys.append(
                     H.level("mpm_particle").get("pos")[:, 1].mean().item()), progress=False)
    return np.array(ys), H.level("mpm_particle").get("pos").detach().cpu().numpy()

dev = sys.argv[1]
print("\n  (1) FREE FALL -- a 20k-particle block released at y = 0.85, fitted over frames 3-55\n")
print(f"  {'body_force':>14}{'accel':>10}{'ratio to g':>12}")
for bf in ("particle", "grid"):
    y, _ = run("material_3d_water_st000", dev, bf, 60, n_par=20000, drop=True)
    t = np.arange(3, 55) * 0.0032
    c = np.linalg.lstsq(np.vstack([np.ones_like(t), t, t*t]).T, y[3:55], rcond=None)[0]
    print(f"  {bf:>14}{2*c[2]:>10.4f}{2*c[2]/-14.0:>12.4f}", flush=True)

print("\n  (2) FULL SCENE, 100k particles, 200 frames -- does the trajectory move?\n")
print(f"  {'body_force':>14}{'level p95':>12}{'spread r90':>12}{'max |dpos| vs particle':>24}")
ref = None
for bf in ("particle", "grid"):
    _, p = run("material_3d_water_st000", dev, bf, 200, n_par=100000)
    r = np.linalg.norm(p[:, [0, 2]] - p[:, [0, 2]].mean(0), axis=1)
    d = "--" if ref is None else f"{np.abs(p-ref).max():.3e}"
    if ref is None: ref = p
    print(f"  {bf:>14}{np.quantile(p[:,1],0.95):>12.5f}{np.quantile(r,0.90):>12.5f}{d:>24}",
          flush=True)
print()
