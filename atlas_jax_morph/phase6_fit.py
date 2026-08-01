"""phase6_fit -- inverse design through the engine, and the variance question it raises.

Phase 6 is the paper's Figure 5: optimise parameters until a growing cluster self-organises into a
specified shape. `grad_fit.py` showed the loop closes on ONE parameter and ONE seed. This script
exists to answer the objection that note has carried since the first gradient probe, and that no
amount of further engineering makes go away on its own:

    the division draw is DISCRETE. A straight-through gradient taken through one sampled rollout
    is a gradient through that sample's LUCK. Optimising against it may be optimising noise.

That is a measurable claim, not a philosophical one. Fit the same target K times from the same
start, averaging the loss over K independent seeds per Adam step, and look at the spread of the
answer across independent replicates. If K=1 replicates scatter and K=8 replicates agree, the
estimator is noisy but consistent and the fix is simply more seeds. If they scatter at every K,
the pathwise estimator is biased and the reference's trace/replay/score contract is not a nicety.

Each Adam step costs K rollouts, and the replicates are completely independent -- which is why
this belongs on the cluster rather than on one machine.

    python phase6_fit.py --seeds 4 --replicate 0 --frames 24 --steps 20 --device cuda:0
    python phase6_fit.py --params max_radius epsilon --out runs/fit_k4_r0.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))
sys.path.insert(0, HERE)

SPEC = os.path.join(PLEXUS, "config", "atlas", "jax_morph_proliferation.yaml")

# knob -> (operator it belongs to, start value, floor). The start is deliberately AWAY from the
# value that hits the target, so a fit that "succeeds" cannot be the initial condition.
KNOBS = {
    "max_radius": ("grow_radius", 0.60, 0.05),
    "epsilon":    ("relax",       1.00, 0.05),
    "mobility":   ("relax",       1.00, 0.05),
}


def rollout(theta, seed, frames, device):
    """One differentiable rollout of the real engine at one seed. Returns the live-cell state."""
    from plexus import engine
    from plexus.schema import load

    sim = load(SPEC)
    sim.n_frames = frames
    sim.seed = int(seed)
    for o in sim.operators:
        for name, t in theta.items():
            if o.op == KNOBS[name][0]:
                o.params[name] = t
    with contextlib.redirect_stdout(io.StringIO()):          # the engine's per-run banner
        H, _ = engine.run(sim, device=device, grad=True)
    lvl = H.level("cell")
    live = lvl.occ > 0
    a, b = lvl.state_schema["radius"]
    r = lvl.state[:, a:b].reshape(-1)[live]
    p = lvl.state[:, 0:2][live]
    gyr = ((p - p.mean(0)) ** 2).sum(1).mean().sqrt()
    return r.mean(), gyr, int(live.sum())


def loss_at(theta, seeds, frames, device, target):
    """Mean loss over K independent seeds -- the K-sample estimator whose variance we are after."""
    terms, obs = [], []
    for s in seeds:
        r_mean, gyr, n = rollout(theta, s, frames, device)
        terms.append((r_mean - target["radius"]) ** 2 + (gyr - target["gyration"]) ** 2)
        obs.append((float(r_mean), float(gyr), n))
    return torch.stack(terms).mean(), obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs="+", default=["max_radius"], choices=sorted(KNOBS))
    ap.add_argument("--seeds", type=int, default=1, help="K: rollouts averaged per Adam step")
    ap.add_argument("--replicate", type=int, default=0, help="independent repeat; picks the seeds")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--target-radius", type=float, default=0.42)
    ap.add_argument("--target-gyration", type=float, default=2.60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import plexus.operators  # noqa: F401
    import run_spec
    run_spec.load_atlas_candidates()

    target = {"radius": a.target_radius, "gyration": a.target_gyration}
    theta = {k: torch.tensor(KNOBS[k][1], device=a.device, requires_grad=True) for k in a.params}
    opt = torch.optim.Adam(list(theta.values()), lr=a.lr)
    # Replicates must not share seeds, or "independent repeat" is a lie and the spread this
    # script measures would be an artefact of reusing the same luck.
    base = 1000 * (a.replicate + 1)

    hist = []
    print(f"[phase6] params={a.params} K={a.seeds} replicate={a.replicate} frames={a.frames} "
          f"steps={a.steps} device={a.device} target={target}", flush=True)
    t0 = time.time()
    for it in range(a.steps):
        seeds = [base + it * a.seeds + j for j in range(a.seeds)]   # fresh seeds every step
        opt.zero_grad()
        loss, obs = loss_at(theta, seeds, a.frames, a.device, target)
        loss.backward()
        grads = {k: (float(t.grad) if t.grad is not None else float("nan"))
                 for k, t in theta.items()}
        vals = {k: float(t) for k, t in theta.items()}
        hist.append({"step": it, "loss": float(loss), "params": vals, "grads": grads,
                     "obs": obs, "seeds": seeds})
        print(f"  {it:>3}  loss {float(loss):.5e}  " +
              "  ".join(f"{k}={v:.4f}(g={grads[k]:+.3e})" for k, v in vals.items()) +
              f"  n={obs[0][2]}", flush=True)
        opt.step()
        with torch.no_grad():
            for k, t in theta.items():
                t.clamp_(min=KNOBS[k][2])

    final = {k: float(t) for k, t in theta.items()}
    out = {"params": a.params, "K": a.seeds, "replicate": a.replicate, "frames": a.frames,
           "steps": a.steps, "lr": a.lr, "target": target, "device": a.device,
           "final": final, "final_loss": hist[-1]["loss"] if hist else None,
           "wall_s": round(time.time() - t0, 1), "history": hist}
    print(f"[phase6] done in {out['wall_s']}s -> {final}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[phase6] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
