"""Does a lone MPM particle fall at g? The simplest possible probe."""
import sys, os, tempfile, yaml, numpy as np
sys.path.insert(0, "/workspace/Plexus/src")
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load

def run(n_par, dev, impl="warp", drag=0.1, n_grid=96, block=0.02, y0=0.85, frames=60):
    s = yaml.safe_load(open("/workspace/Plexus/config/material/material_3d_water_st000.yaml"))
    s["general"]["n_frames"] = frames
    s["general"]["record_cap"] = 10000
    ty = list(s["sets"]["cell"]["types"].values())[0]
    ty["block"] = [0.5 - block/2, y0, 0.5 - block/2, 0.5 + block/2, y0 + block, 0.5 + block/2]
    s["sets"]["mpm_particle"]["per_parent"] = n_par
    s["sets"]["cell"]["start"] = [[0.5, y0 + block/2, 0.5]]
    for fc in s["fields"].values():
        fc["n_grid"] = n_grid
    for o in s["operators"]:
        if o["op"] in ("mpm_strain", "mpm_scatter", "mpm_gather"):
            if impl is None: o.pop("implementation", None)
            else: o["implementation"] = impl
        if o["op"] == "mpm_scatter":
            o["drag"] = drag
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    sim = load(f.name); os.unlink(f.name)
    ys = []
    E.run(sim, out_path=None, device=dev,
          on_frame=lambda H, t: ys.append(float(H.level("mpm_particle").get("pos")[:, 1].mean())),
          progress=False)
    y = np.array(ys); dt = 0.0032
    # FIT ONLY WHERE IT IS IN FLIGHT. From y0 = 0.85 free fall reaches the floor at
    # t = sqrt(2*0.85/14) = 0.348 s = 109 frames, so a 120-frame window would put the IMPACT
    # inside the parabola and read back an acceleration that is 36% too small -- which is exactly
    # what the first version of this probe reported before the window was cut.
    lo, hi = 3, 55
    y = y[lo:hi]; t = np.arange(lo, hi) * dt
    A = np.vstack([np.ones_like(t), t, t*t]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = float(np.sqrt(((y - A@c)**2).mean()))
    return 2*c[2], y[0], resid

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
print(f"\n  A BLOCK IN FREE FALL, no floor contact. Expect acceleration = -14.0 exactly.\n")
print(f"  {'n particles':>13}{'impl':>9}{'drag':>7}{'n_grid':>8}{'accel':>10}{'ratio to g':>12}{'fit resid':>12}")
print("  " + "-" * 60)
for n, impl, drag, ng in [(1,"warp",0.1,96), (8,"warp",0.1,96), (64,"warp",0.1,96),
                          (512,"warp",0.1,96), (4096,"warp",0.1,96),
                          (1,"warp",0.0,96), (8,"warp",0.0,96),
                          (1,None,0.1,96), (8,None,0.1,96),
                          (1,"warp",0.1,32), (8,"warp",0.1,32)]:
    a, y0, rs = run(n, dev, impl=impl, drag=drag, n_grid=ng)
    print(f"  {n:>13}{str(impl or 'default'):>9}{drag:>7}{ng:>8}{a:>10.3f}{a/-14.0:>12.4f}"
          f"{rs:>12.2e}", flush=True)
print()
