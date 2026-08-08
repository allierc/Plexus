import numpy as np
import beat as B, validate as V, seeded as SD, affine_test as AT

seeds, gi, gj = SD.nuclei_on_grid()
uv = B.load()
fitb = V.half_beats(uv, [0, 2])          # partitions are derived from these
evalb = V.half_beats(uv, [1, 3])         # and scored on these
K = int(seeds.max())

parts = {}
parts["voronoi (nuclei, no motion)"] = SD.voronoi(seeds)
parts["motion (nuclei-seeded)"] = SD.seeded_watershed(SD.bnd_from(fitb), seeds)
parts["cheating (from the eval beats)"] = SD.seeded_watershed(SD.bnd_from(evalb), seeds)
rng = np.random.default_rng(0)
sh = AT.shifted_seeds(seeds, 23, 31)
parts["voronoi of DISPLACED nuclei"] = SD.voronoi(sh)
parts["one region (no partition)"] = np.ones_like(seeds)

print(f"  K = {K} regions where applicable; scored on beats the partition never saw\n")
print(f"  {'partition':<34s} {'regions':>8s} {'FVU held-out':>13s}   {'vs voronoi':>10s}")
base = None
rows = {}
for nm, lab in parts.items():
    f = AT.fvu(lab, evalb)
    rows[nm] = f
    if "voronoi (nuclei" in nm:
        base = f
    d = "" if base is None else f"{(base - f) / base:+.2%}"
    print(f"  {nm:<34s} {len(np.unique(lab)):>8d} {f:>13.5f}   {d:>10s}", flush=True)

print(f"\n  A partition that follows cell boundaries explains MORE of the held-out motion than the")
print(f"  same nuclei tessellated by geometry alone. Negative means the motion-derived borders are")
print(f"  worse than simply cutting halfway between nuclei.")
