"""Byte-identity of recorded arrays between two checkouts. tobytes(), not allclose()."""
import sys, os, tempfile, yaml, numpy as np
ROOT = sys.argv[1]; SPEC = sys.argv[2]; DEV = sys.argv[3]; OUT = sys.argv[4]
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch, plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
s = yaml.safe_load(open(os.path.join(ROOT, "config", "material", SPEC + ".yaml")))
IMPL = sys.argv[5] if len(sys.argv) > 5 else None
if IMPL:
    for o in s["operators"]:
        if o["op"] in ("mpm_strain", "mpm_scatter", "mpm_gather"):
            if IMPL == "default": o.pop("implementation", None)
            else: o["implementation"] = IMPL
s["general"]["n_frames"] = 120; s["general"]["record_cap"] = 10000; s["general"]["seed"] = 0
f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(s, f); f.close()
sim = load(f.name); os.unlink(f.name)
snaps = []
E.run(sim, out_path=None, device=DEV, progress=False,
      on_frame=lambda H, t: snaps.append(
          np.concatenate([H.level("mpm_particle").get("pos").detach().cpu().numpy().ravel(),
                          H.level("mpm_particle").get("vel").detach().cpu().numpy().ravel()])
      ) if t % 20 == 0 else None)
np.save(OUT, np.stack([x for x in snaps if x is not None]))
g = sim  # report the grid geometry too
print(f"  {SPEC}: {len([x for x in snaps if x is not None])} snapshots -> {OUT}")
