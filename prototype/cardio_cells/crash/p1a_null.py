"""The divergence null, in this harness: theta EXACTLY true, a random kick at t0 instead."""
import argparse, json, os, sys
import numpy as np, torch
for p in ("/workspace/Plexus/src","/workspace/Plexus/prototype/cardio_cells/algebraic",
          "/workspace/Plexus/prototype/cardio_cells/crash","/workspace/Plexus/discovery_cardio_mpm"):
    sys.path.insert(0,p)
import crash_test as CT, accept as AC, metrics as MET
from p1a_percell import run, theta_vectors, disp_stats

a = argparse.Namespace(device="cuda:0", cells=100, per_parent=100, n_grid=128, warmup=180,
                       window=150, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                       g_lo=0.5, g_hi=1.5)
G, W = 150, 180
fl = AC.working_floors(); pe = MET.REGISTRY["peak_excursion"]
out = {}
with torch.no_grad():
    sy,_ = CT.plant_and_warm(a, lambda *x: None)
    tr = {m: CT.tracer_indices(sy.x0, CT.probe_points(m)) for m in (20,10)}
    xb, trb = run(sy, sy.E_true, sy.gain_true, W, G, tracers=tr)
    real = trb[20].cpu().numpy()
    print(f"[base] peak_excursion {float(np.median(pe.reading(real))):.6g}")
    print(f"  {'null':<22s} {'kick px':>9s} {'final max px':>13s} {'final rms px':>13s} {'STEPS':>8s}  limiting")
    for j in (1e-6, 1e-5, 1e-4, 7.8125e-5, 7.8125e-4):
        t, _, _ = CT.rollout(sy, sy.theta_true, W, G, tr, jitter=j)
        sim = t[20].cpu().numpy()
        xf = sy.p.get("pos").clone()
        d = disp_stats(xf, xb)
        one = AC.score_one(sim, real, fl)
        st = {n: one[n]["steps"] for n in AC.CERTIFIED}
        live=[n for n in st if st[n] is not None]
        w = max(st[n] for n in live); lim=max(live,key=lambda n: st[n])
        out[f"jitter_{j:g}"] = {"kick_px": j*1024, "final": d, "steps": st, "worst": w}
        print(f"  jitter {j:<15g} {j*1024:>9.4f} {d['max_px']:>13.4f} {d['rms_px']:>13.4f} {w:>8.4f}  {lim}")
json.dump(out, open("/workspace/Plexus/prototype/cardio_cells/crash/p1a_divergence_null.json","w"), indent=1, default=str)
print("wrote p1a_divergence_null.json")
