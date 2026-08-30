"""Throughput at a FIXED grid, capture off: isolate the grid's cost from everything else."""
import sys, os, tempfile, yaml, torch, time
sys.path.insert(0, 'src')
import plexus.operators, plexus.operators.mpm_warp
from plexus import engine as E
from plexus.schema import load
from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL

N, dev, NG = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
s = yaml.safe_load(open("config/si_material/si_bench_100m.yaml"))
s["general"]["n_frames"] = 8
s["sets"]["mpm_particle"]["per_parent"] = N // 27
for fc in s["fields"].values():
    fc["n_grid"] = NG
for blk in s["schedule"]:
    if isinstance(blk, dict) and "substep_dt" in blk:
        blk["capture"] = False
        sub = blk["substep_dt"]
f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(s, f); f.close()
CFL(f.name)
nsub = round(float(s["general"]["dt"]) / float(sub))
sim = load(f.name); os.unlink(f.name)
torch.cuda.set_device(int(dev.split(":")[-1])); torch.cuda.init()
torch.cuda.reset_peak_memory_stats()
ts = []
E.run(sim, out_path=None, device=dev, progress=False,
      on_frame=lambda H, t: ts.append(time.perf_counter()))
d = [(ts[i + 1] - ts[i]) * 1000 for i in range(len(ts) - 1)]
w = d[len(d) // 2:] or d
ms = sum(w) / len(w)
n = 27 * (N // 27)
print(f"\n  {n / 1e6:.0f}M  n_grid {NG} ({NG ** 3 / 1e6:.1f}M cells)  {nsub} substeps  capture OFF")
print(f"    peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB   {ms:.0f} ms/frame"
      f"   {ms * 1e6 / (n * nsub):.1f} ns per particle-substep")
print(f"    600 frames -> {600 * ms / 1000 / 3600:.2f} h\n")
