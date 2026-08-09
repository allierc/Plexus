"""finject_thresh.py -- how accurate must the measured F be?

finject_noise.py showed that the recording's own F error (sigma_F = 0.0039 temporal, 0.033 between
two estimates of the same quantity) destroys the F-injection recovery: med|dE/E| 0.008 -> 0.24 ->
0.97. This finds the threshold, so the requirement on a better PIV/derivative estimator is a number
and not an adjective.

usage: PYTHONPATH=/workspace/Plexus/src python finject_thresh.py --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)
from recover import Solver, score                               # noqa: E402
import crash_test as CT                                         # noqa: E402
from finject import assemble_inj, record_substeps, lerp         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128, warmup=165,
                       window=150, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                       g_lo=0.5, g_hi=1.5)
lines = []
def log(s):
    print(s, flush=True); lines.append(str(s))

R = {"config": vars(args)}
t0 = time.time()
torch.manual_seed(0)
with torch.no_grad():
    sy, _ = CT.plant_and_warm(args, log)
    C, n = sy.C, sy.n_sub_per_frame
    th = sy.theta_true.double()
    x0, cid = sy.x0.clone(), sy.cid
    Fs, Cs, Xs = record_substeps(sy, n)
    x_next = Xs[-1].clone()
    Fl = lerp(sy.F0.clone(), Fs[-1].clone(), n)
    gen = torch.Generator(device=sy.device).manual_seed(777)
    R["rows"] = {}
    log(f"\n[threshold] sigma_F sweep, F_lerp injection, no position noise, {a.reps} reps")
    log(f"    {'sigma_F':>10s} {'mode':>6s} {'medE':>8s} {'p90E':>8s} {'medg':>8s} {'negE':>6s}")
    for mode in ("indep", "cell"):
        for sF in (0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3.9e-3):
            sc = []
            neg = []
            for _ in range(a.reps if sF > 0 else 1):
                if sF <= 0:
                    iF = Fl
                elif mode == "indep":
                    iF = Fl + (sF / 2.0) * torch.randn(Fl.shape, generator=gen, device=Fl.device,
                                                       dtype=Fl.dtype)
                else:
                    ec = torch.randn((Fl.shape[0], C + 1, 2, 2), generator=gen, device=Fl.device,
                                     dtype=Fl.dtype)
                    iF = Fl + (sF / 2.0) * ec[:, cid]
                A, y0, _ = assemble_inj(sy, n, iF, None)
                S = Solver(A, C)
                t_hat = S((x_next - x0).reshape(-1) - y0)["ridge0"]
                sc.append(score(t_hat, th, C)); neg.append(int((t_hat[:C] < 0).sum()))
                S.free(); del A, S
                torch.cuda.empty_cache()
            k = f"{mode}_sF{sF:g}"
            R["rows"][k] = {"mode": mode, "sigma_F": sF,
                            "med_E": float(np.mean([s["med_E"] for s in sc])),
                            "p90_E": float(np.mean([s["p90_E"] for s in sc])),
                            "med_gain": float(np.mean([s["med_gain"] for s in sc])),
                            "n_negative_E": float(np.mean(neg)),
                            "per_rep_med_E": [s["med_E"] for s in sc]}
            v = R["rows"][k]
            log(f"    {sF:>10.2e} {mode:>6s} {v['med_E']:>8.4f} {v['p90_E']:>8.4f} "
                f"{v['med_gain']:>8.4f} {v['n_negative_E']:>6.1f}")
R["wall_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "finject_thresh.json"), "w"), indent=1, default=str)
open(os.path.join(HERE, "finject_thresh.log"), "w").write("\n".join(lines) + "\n")
log(f"\nwrote finject_thresh.json [{R['wall_seconds']:.0f} s]")
