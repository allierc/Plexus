"""Gate mpm_grid_update[nosync] on the 2D specs: BYTE-IDENTICAL to default, and how much faster.

Byte-identity is the right bar here, unlike for the warp implementations: this is the same
arithmetic in the same order on the same reads, only expressed without boolean-mask indexing. If it
is not bit-identical, the rewrite changed the boundary condition and the speed is irrelevant.
"""
import sys, os, yaml, tempfile, time, hashlib
sys.path.insert(0, "src")
import torch
torch.cuda.set_device(1)
import plexus.operators
from plexus.schema import load
from plexus import engine as E
torch.cuda.init(); torch.zeros(1, device="cuda:1")
DEV = "cuda:1"
SPECS = ["material_dam_break", "material_dam_viscous", "material_slosh", "material_funnel",
         "material_hydrostatic", "material_crown_splash", "material_snow_pile",
         "material_snow_funnel", "material_bowl_1", "material_steps_1", "material_vessel_1",
         "material_zigzag", "material_coalesce", "material_two_drops_st", "material_active_swirl"]
FRAMES, WARM = 20, 3

def run(nm, impl):
    s = yaml.safe_load(open(f"config/material/{nm}.yaml"))
    for o in s["operators"]:
        if o.get("op") == "mpm_grid_update":
            o.pop("implementation", None)
            if impl != "default":
                o["implementation"] = impl
    s["general"]["n_frames"] = FRAMES + WARM
    s["general"]["record_cap"] = 2
    s["general"]["seed"] = 0
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    sim = load(f.name); os.unlink(f.name)
    m = {}
    def on_frame(H, t):
        if t == WARM: torch.cuda.synchronize(DEV); m["a"] = time.perf_counter()
        elif t == WARM + FRAMES: torch.cuda.synchronize(DEV); m["b"] = time.perf_counter()
    H, _ = E.run(sim, out_path=None, device=DEV, on_frame=on_frame, progress=False)
    ms = (m["b"] - m["a"]) / FRAMES * 1000
    h = hashlib.sha256()
    for name in sorted(H.levels):
        lvl = H.levels[name]
        if not hasattr(lvl, "F"): continue
        for t in (lvl.get("pos"), lvl.get("vel"), lvl.F, lvl.C):
            h.update(t.detach().cpu().numpy().tobytes())
    return ms, h.hexdigest()[:16], sum(int(l.n) for l in H.levels.values() if hasattr(l, "F"))

print(f"\n  {'spec':<26}{'N':>8}{'default':>10}{'nosync':>9}{'speedup':>9}  byte-identical")
print("  " + "-" * 76)
ok = bad = 0; sp = []
for nm in SPECS:
    try:
        d_ms, d_h, n = run(nm, "default")
        n_ms, n_h, _ = run(nm, "nosync")
    except Exception as e:
        print(f"  {nm:<26}  ERROR {type(e).__name__}: {str(e).splitlines()[0][:40]}", flush=True); continue
    same = d_h == n_h
    ok += same; bad += (not same); sp.append(d_ms / n_ms)
    print(f"  {nm:<26}{n:>8,}{d_ms:>10.1f}{n_ms:>9.1f}{d_ms/n_ms:>8.2f}x  "
          f"{'YES' if same else 'NO  ' + d_h + ' vs ' + n_h}", flush=True)
print("  " + "-" * 76)
print(f"  {ok} byte-identical, {bad} NOT.  speedup min {min(sp):.2f}x  median "
      f"{sorted(sp)[len(sp)//2]:.2f}x  max {max(sp):.2f}x\n")
