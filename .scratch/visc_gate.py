"""GATE for mpm_viscosity. Three claims, three measurements.
   (1) INERT AT eta=0 / absent.
   (2) IT IS NOT DRAG: a body in rigid translation has C=0, so free fall must be UNCHANGED --
       this is the whole distinction from `drag`, which slows it.
   (3) IT DISSIPATES: kinetic energy must decay monotonically faster with eta."""
import sys, os, tempfile, yaml, numpy as np
sys.path.insert(0, "/workspace/Plexus/src")
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL

def run(spec, dev, eta, frames, n_par=None, drop=None, impl="warp"):
    s = yaml.safe_load(open(f"/workspace/Plexus/config/material/{spec}.yaml"))
    s["general"]["n_frames"] = frames; s["general"]["record_cap"] = 10000; s["general"]["seed"] = 0
    if n_par: s["sets"]["mpm_particle"]["per_parent"] = n_par
    if drop:
        ty = list(s["sets"]["cell"]["types"].values())[0]
        ty["block"] = [0.49, 0.85, 0.49, 0.51, 0.87, 0.51]
        s["sets"]["cell"]["start"] = [[0.5, 0.86, 0.5]]
    for o in s["operators"]:
        if o["op"] in ("mpm_strain","mpm_scatter","mpm_gather"): o["implementation"] = impl
    if eta is not None:
        s["operators"].append({"op": "mpm_viscosity", "at": "mpm_particle", "eta": float(eta)})
        for blk in s["schedule"]:
            if isinstance(blk, dict) and "steps" in blk:
                blk["steps"] = ["mpm_strain", "mpm_viscosity"] + [x for x in blk["steps"]
                                                                 if x != "mpm_strain"]
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close(); CFL(f.name)
    sim = load(f.name); os.unlink(f.name)
    ys, ke = [], []
    def cb(H, t):
        L = H.level("mpm_particle")
        ys.append(L.get("pos")[:, 1].mean().item())
        ke.append((L.get("vel") ** 2).sum(1).mean().item())
    H, _ = E.run(sim, out_path=None, device=dev, on_frame=cb, progress=False)
    p = H.level("mpm_particle").get("pos").detach().cpu().numpy()
    return np.array(ys), np.array(ke), p

dev = sys.argv[1]
print("\n  (2) FREE FALL -- 20k block, fitted frames 3-55. A VISCOUS STRESS MUST NOT CHANGE IT.\n")
print(f"  {'eta':>10}{'accel':>10}{'ratio to g':>12}")
for eta in (None, 0.0, 3e-4, 1e-2):
    y, _, _ = run("material_3d_water_st000", dev, eta, 60, n_par=20000, drop=True)
    t = np.arange(3, 55) * 0.0032
    c = np.linalg.lstsq(np.vstack([np.ones_like(t), t, t*t]).T, y[3:55], rcond=None)[0]
    print(f"  {str(eta):>10}{2*c[2]:>10.4f}{2*c[2]/-14.0:>12.4f}", flush=True)

print("\n  (1)+(3) SPLASH -- 100k particles, 300 frames. eta=0 must match absent; KE must fall.\n")
print(f"  {'eta':>10}{'Re @impact':>12}{'Re @end':>10}{'KE f150':>11}{'KE f299':>11}"
      f"{'KE/KE(eta=0)':>14}{'r90':>9}{'haze':>7}")
ref = None
for eta in (None, 0.0, 1e-4, 3e-4, 1e-3, 3e-3):
    _, ke, p = run("material_3d_water_st000", dev, eta, 300, n_par=100000)
    r = np.linalg.norm(p[:, [0, 2]] - p[:, [0, 2]].mean(0), axis=1)
    yv = p[:, 1]; hz = int((yv > np.quantile(yv, 0.95) + 0.05).sum())
    e = eta or 1e-30
    if ref is None: ref = ke[-1]
    print(f"  {str(eta):>10}{3.9*0.2/e:>12.0f}{0.05*0.2/e:>10.1f}{ke[150]:>11.5f}{ke[-1]:>11.5f}"
          f"{ke[-1]/ref:>14.4f}{np.quantile(r,0.90):>9.5f}{hz:>7}", flush=True)
print()
