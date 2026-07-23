#!/usr/bin/env python
"""Profile the first N frames of a recent spec: split engine (physics) vs rendering, and cProfile the hot
functions, so we optimise the real bottleneck (user: rendering must not be the limiting computation)."""
import sys, time, cProfile, pstats, io
import numpy as np
import run_tyssue_round as R
from run_tyssue_round import engine_run, _draw, _cross_screen

preset = sys.argv[1] if len(sys.argv) > 1 else "round_34_700"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
p = dict(R.PRESETS[preset], frames=N)
print(f"=== profiling {preset}, {N} frames ===", flush=True)

sim, cfg = R.make(p)
t0 = time.perf_counter()
pr = cProfile.Profile(); pr.enable()
Hf, out = engine_run(sim, device="cpu")
pr.disable()
te = time.perf_counter() - t0
print(f"\nENGINE: {te:.1f}s total = {te/N*1000:.0f} ms/frame", flush=True)
s = io.StringIO(); st = pstats.Stats(pr, stream=s).sort_stats("tottime"); st.print_stats(18)
print("\n".join(s.getvalue().splitlines()[:26]))

# --- correctness: the T1 optimisation must keep the mesh CLOSED (Euler=2) ---
from tyssue_topology_ops3d import rings_from_flat_3d
from tyssue_t1_ops3d import _check_closed
m = Hf.level("vertex")._mesh
rings = rings_from_flat_3d(m["E_srce"].cpu().numpy(), m["E_trgt"].cpu().numpy(), m["E_face"].cpu().numpy(), int(m["nF"]))
ok, V, E, F, eu = _check_closed(rings)
print(f"CORRECTNESS: closed={ok} Euler={eu} (must be 2)  V={V} E={E} F={F}  cells={int(m['nF'])}")

# --- rendering cost: one 3D panel + one cross-section, timed ---
emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
mt = hist[-1]; pt = posf[T-1][:mt["nF"]*0 + mt["Nv"]].astype(np.float64); a = chemf[T-1][:mt["nF"], 0]
col = np.clip(a, 0, 1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5, 5)); ax3 = fig.add_subplot(1, 2, 1, projection="3d"); ax2 = fig.add_subplot(1, 2, 2)
K = 10
t0 = time.perf_counter()
for _ in range(K):
    _draw(ax3, pt, mt, 3.90, azim=30, act=col, Lbox=20.0)
t_draw = (time.perf_counter() - t0) / K
t0 = time.perf_counter()
for _ in range(K):
    _cross_screen(ax2, pt, mt, col, seed_dir=p.get("seed_dir"), Lbox=20)
t_cross = (time.perf_counter() - t0) / K
plt.close(fig)
print(f"\nRENDER (final frame, {mt['nF']} cells): 3D panel {t_draw*1000:.0f} ms  |  cross-section {t_cross*1000:.0f} ms")
print(f"  movie (~110 frames x (3D+cross)) ~= {110*(t_draw+t_cross):.1f}s ; strip (8x) ~= {8*(t_draw+t_cross):.1f}s")
print(f"\nSUMMARY: engine {te:.0f}s ({te/N*1000:.0f} ms/f)  vs  movie-render ~{110*(t_draw+t_cross):.0f}s  -> "
      f"{'RENDER-bound' if 110*(t_draw+t_cross) > te else 'ENGINE-bound'}")
