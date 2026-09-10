"""step_test -- how fast does the sheet load and unload when the clock is a step? (PLAN_R2 step 2a)

gamma: 0 until frame 5, 1 from frame 5 to 25, 0 after. Every cell gets its fitted g and phi; the
mean shortening over interior cells is recorded per frame; the rise and decay time constants are
the frames to reach 63% of the step (loading) and to fall to 37% of the plateau (unloading). One
knob is swept at a time from the reference setting (kappa 1e4, drag 30, E 80, grid 128, 120/cell,
substep 2e-4). The tissue unloads in ~4 frames; the model in ~10 -- this finds what sets that.

    python step_test.py --device cuda:1
"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model as M, recording as R

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--frames", type=int, default=45)
ap.add_argument("--params", default=os.path.join(HERE, "out", "fits", "s4_live_r6_Eshrink0.3", "params.npz"))
args = ap.parse_args(); dev = args.device
rec = R.load(device=dev); C = rec["n_cells"]; z = np.load(args.params); inter = torch.as_tensor(z["interior"], device=dev)
REF = dict(anchor=1e4, drag=30.0, youngs=80.0, n_grid=128, ppc=120, sub=2e-4, density=1.0)
# INERTIA. With unit density the substrate mode is under-damped: damping ratio k / 2 sqrt(kappa rho)
# = 0.15, so after unloading the sheet RINGS (period 2 pi / sqrt(kappa/rho) = 31 frames) -- what the
# step test's "left at end" measures. A tissue on a gel has no inertia to speak of. Lowering the
# density raises the damping ratio (rho 0.02 -> 1.06, critical) at the price of a faster elastic wave
# (CFL: substep 1e-4). Time constant of the return then ~ k / kappa = 1.5 frames.
SWEEP = [("anchor", [1e3, 3e4]), ("drag", [10.0, 100.0]), ("youngs", [40.0, 160.0]), ("n_grid", [96, 160]),
         ("ppc", [50, 200]), ("sub", [1e-4]), ("density", [0.1, 0.05]), ("density+sub", [(0.02, 1e-4), (0.01, 5e-5)])]


class StepParams(M.Params):
    def gamma_cells(self, t):
        v = 1.0 if 5 <= t < 25 else 0.0
        return torch.full_like(self.delay, v)


def run(cfg):
    P = StepParams(C, dev); P.load({k: z[k] for k in ("g", "phi")}); 
    with torch.no_grad(): P.logE.fill_(np.log(cfg["youngs"]))
    kw = dict(n_grid=cfg["n_grid"], per_parent=cfg["ppc"], n_frames=args.frames, n_cells=C, anchor_k=cfg["anchor"],
              drag_k=cfg["drag"], youngs=cfg["youngs"], sub=cfg["sub"], density=cfg["density"], label_tif=M.LABEL_TIF,
              differentiable=False, name="step")
    t0 = time.time()
    with torch.no_grad():
        out = M.rollout(M.load_sim(M.build_spec(**kw)), P, dev, C, grad=False)
    # SIGNED strain along each cell's own fibre (negative = shortening): an overshoot into expansion
    # shows as a sign change, which -lambda_min (always >= 0) would have counted as more shortening
    f = torch.stack([torch.cos(P.phi), torch.sin(P.phi)], 1)
    E = out["A"] - torch.eye(2, device=dev); Sy = 0.5 * (E + E.transpose(-1, -2))
    along = torch.einsum("ci,tcij,cj->tc", f, Sy, f)[:, inter]
    s = (-along.mean(1)).detach().cpu().numpy()                 # positive = mean shortening along the fibre
    plateau = s[20:25].mean()
    up = s[5:25]; rise = int(np.argmax(up >= 0.63 * plateau)) if (up >= 0.63 * plateau).any() else -1
    down = s[25:]; decay = int(np.argmax(down <= 0.37 * plateau)) if (down <= 0.37 * plateau).any() else -1
    tail = s[25:] / plateau
    return dict(cfg=cfg, plateau=float(plateau), rise_frames=rise, decay_frames=decay, residual_end=float(s[-1] / plateau),
                overshoot=float(tail.min()), curve=[round(float(v), 5) for v in s], seconds=time.time() - t0)


rows = [run(REF)]
print(f"  reference {REF}: plateau {rows[0]['plateau']:.4f}, rise {rows[0]['rise_frames']} fr, decay {rows[0]['decay_frames']} fr, "
      f"left at end {rows[0]['residual_end']:+.2f}, min after unloading {rows[0]['overshoot']:+.2f}  ({rows[0]['seconds']:.0f} s)", flush=True)
for k, vals in SWEEP:
    for v in vals:
        cfg = dict(REF)
        if k == "density+sub":
            cfg["density"], cfg["sub"] = v
        else:
            cfg[k] = v
        r = run(cfg); rows.append(r)
        print(f"  {k}={v}: plateau {r['plateau']:.4f}, rise {r['rise_frames']} fr, decay {r['decay_frames']} fr, "
              f"left at end {r['residual_end']:+.2f}, min after unloading {r['overshoot']:+.2f}  ({r['seconds']:.0f} s)", flush=True)
json.dump(rows, open(os.path.join(HERE, "out", "step_test.json"), "w"), indent=1)
