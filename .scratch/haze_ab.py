"""Direct counterfactual on the spec where the haze is: does mass_floor cause it?"""
import sys, os, tempfile, yaml, numpy as np
sys.path.insert(0, "/workspace/Plexus/src")
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition

def run(spec, frames, dev, **over):
    s = yaml.safe_load(open(f"/workspace/Plexus/config/material/{spec}.yaml"))
    s["general"]["n_frames"] = frames; s["general"]["record_cap"] = 2
    for o in s["operators"]:
        for k, v in over.items():
            if o["op"] == "mpm_grid_update" and k in ("mass_floor", "csf_mass_floor"):
                o[k] = v
            if o["op"] == "mpm_scatter" and k in ("drag", "a_max"):
                o[k] = v
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close(); Courant_Friedrichs_Lewy_condition(f.name)
    sim = load(f.name); os.unlink(f.name)
    H, _ = E.run(sim, out_path=None, device=dev, progress=False)
    p = H.level("mpm_particle").get("pos").detach().cpu().numpy()
    y = p[:, 1]; lvl = np.quantile(y, 0.95); m = y > lvl + 0.05
    return int(m.sum()), float(m.mean() * 100), float(y.max()), float(lvl)

spec, dev = sys.argv[1], sys.argv[2]
ARMS = [("shipped", {}),
        ("mass_floor 1e-14", dict(mass_floor=1e-14)),
        ("both floors 1e-14", dict(mass_floor=1e-14, csf_mass_floor=1e-14)),
        ("drag 0", dict(drag=0.0)),
        ("a_max 1e9", dict(a_max=1e9))]
print(f"\n  {spec}, 720 frames -- haze = above bulk_p95 + 0.05\n")
print(f"  {'arm':>20}{'haze n':>9}{'haze %':>9}{'y max':>9}{'bulk p95':>10}")
for tag, kw in ARMS:
    n, pc, ym, lvl = run(spec, 720, dev, **kw)
    print(f"  {tag:>20}{n:>9}{pc:>8.3f}%{ym:>9.4f}{lvl:>10.4f}", flush=True)
print()
