"""Does warp track default on SNOW over enough frames for Jp to move?

The gate ran 20 frames, over which Jp barely leaves 1, so a missing hardening term was invisible.
Snow plasticity needs hundreds of frames of sustained compaction to diverge. Reported here is the
thing that actually went wrong on screen: the height of the snow body.
"""
import sys, os, yaml, tempfile
sys.path.insert(0, "src")
import torch
torch.cuda.set_device(1)
import plexus.operators, plexus.operators.mpm_warp
from plexus.schema import load
from plexus import engine as E

def run(impl, frames):
    s = yaml.safe_load(open("config/material/material_3d_multimaterial.yaml"))
    for o in s["operators"]:
        if o.get("op", "").startswith("mpm_"):
            o.pop("implementation", None)
            if impl != "default" and o["op"] != "mpm_grid_update":
                o["implementation"] = impl
    s["general"]["n_frames"] = frames; s["general"]["record_cap"] = 2
    for st in s["schedule"]:
        if isinstance(st, dict) and "substep_dt" in st: st["capture"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False); yaml.safe_dump(s, f); f.close()
    H, _ = E.run(load(f.name), out_path=None, device="cuda:1", progress=False); os.unlink(f.name)
    p = H.level("mpm_particle")
    snow = p.is_snow
    pos = p.get("pos")[snow]
    return (float(p.Jp[snow].mean()), float(p.Jp[snow].min()),
            float(pos[:, 1].max() - pos[:, 1].min()), float(pos[:, 1].mean()))

for fr in (20, 200):
    d = run("default", fr); w = run("warp", fr)
    print(f"RESULT {fr:>4} frames | default Jp mean {d[0]:.4f} min {d[1]:.4f}  snow height {d[2]:.4f}", flush=True)
    print(f"RESULT {fr:>4} frames | warp    Jp mean {w[0]:.4f} min {w[1]:.4f}  snow height {w[2]:.4f}"
          f"   height diff {abs(d[2]-w[2])/max(d[2],1e-9)*100:.2f}%", flush=True)
