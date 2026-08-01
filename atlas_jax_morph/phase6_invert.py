"""phase6_invert -- Figure 5 in miniature: fit a gene circuit to a target morphology.

`regulated_growth.yaml` is the FORWARD half: a hand-written circuit whose activator (W_in > 0)
makes the crowded interior grow fastest, producing a centre-hot radial gradient. This is the
INVERSE half, and it asks for the opposite morphology:

    make the RIM grow fastest.

That target is chosen because its answer is known qualitatively and is a sign flip, not a
tweak: a cell that grows *less* where the signal is *higher* needs an INHIBITOR, W_in < 0. If
gradient descent through the engine discovers that on its own -- starting from the hand-written
activator -- then the inverse half of Plexus works on a gene circuit, and Figure 5 is the same
thing with more genes.

WHAT IS BEING DIFFERENTIATED. A loss on the final state of a 24-frame rollout of the real engine,
back through growth, adhesion, the morphogen solve, the intracellular ODE, and the discrete
division draw, to two numbers of the circuit: `W_in` and `b`.

The loss asks for the response, not for a picture: each cell's growth rate should fall linearly
with the concentration it sensed. Mean-centring makes the target adaptive rather than a fixed
number the optimiser could reach by shifting everything down.

TWO SETTINGS ARE NOT ARBITRARY, both from the variance sweep (S12): average K >= 2 seeds per step
(a single-seed fit is a draw, not an answer), and run past Adam's overshoot rather than to a
round step count.

    python phase6_invert.py --steps 40 --seeds 2 --device cuda:0
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

SPEC = os.path.join(PLEXUS, "config", "atlas", "regulated_growth.yaml")


def rollout(W_in, b, seed, frames, device):
    """One differentiable rollout; returns (sensed chemical, decided growth rate) for live cells."""
    from plexus import engine
    from plexus.schema import load

    sim = load(SPEC)
    sim.n_frames = frames
    sim.seed = int(seed)
    for o in sim.operators:
        if o.op == "regulate":
            o.params["W_in"] = W_in
            o.params["b"] = b
    with contextlib.redirect_stdout(io.StringIO()):
        H, _ = engine.run(sim, device=device, grad=True)
    lvl = H.level("cell")
    live = lvl.occ > 0
    c0, c1 = lvl.state_schema["chemical"]
    g0, g1 = lvl.state_schema["growth_rate"]
    return (lvl.state[:, c0:c1].reshape(-1)[live],
            lvl.state[:, g0:g1].reshape(-1)[live], int(live.sum()))


def loss_one(W_in, b, seed, frames, device, k0, gain, w_slope=20.0):
    """Ask for an INVERTED RESPONSE, and say so in the loss rather than hoping.

    A pointwise target `k_i = k0 - gain*(c_i - mean c)` looks like it asks for a negative slope,
    but it does not: the mean term dominates it. Measured on the first attempt -- the fit happily
    RAISED W_in (an activator, the wrong sign) because that moved the population mean toward k0
    while the slope, contributing a few percent of the loss, was ignored. The loss got better and
    the morphology got further from the target.

    So the slope is regressed explicitly -- d(growth)/d(sensed), the number the target is about --
    and the level is left to a separate, weaker term that `b` can satisfy on its own.
    """
    c, k, n = rollout(W_in, b, seed, frames, device)
    cd = c - c.mean()
    denom = (cd ** 2).sum()
    slope = (cd * (k - k.mean())).sum() / denom.clamp(min=1e-9)
    return w_slope * (slope + gain) ** 2 + (k.mean() - k0) ** 2, c, k, n


def response(c, k):
    """Slope and correlation of decided-vs-sensed -- the number the target is really about."""
    cd, kd = c - c.mean(), k - k.mean()
    denom = float((cd ** 2).sum())
    slope = float((cd * kd).sum() / denom) if denom > 1e-12 else float("nan")
    sd = float(cd.std()) * float(kd.std())
    corr = float((cd * kd).mean() / sd) if sd > 1e-12 else float("nan")
    return slope, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--k0", type=float, default=0.75, help="target mean growth rate")
    ap.add_argument("--gain", type=float, default=0.15,
                    help="target slope magnitude; the target response is k0 - gain*(c - mean c)")
    ap.add_argument("--w-slope", type=float, default=20.0,
                    help="weight on the response-slope term relative to the level term")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "_state", "phase6_invert.json"))
    a = ap.parse_args()

    import plexus.operators  # noqa: F401
    import run_spec
    run_spec.load_atlas_candidates()

    # start from the hand-written ACTIVATOR -- the fit has to cross zero to succeed
    W = torch.tensor([[0.35]], device=a.device, requires_grad=True)
    b = torch.tensor([-0.6], device=a.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=a.lr)

    print(f"[invert] target: growth FALLS with sensed signal (slope -{a.gain}, mean {a.k0})")
    print(f"[invert] start:  W_in {float(W):+.4f} (activator), b {float(b):+.4f}\n")
    print(f"  {'step':>4} {'loss':>10} {'W_in':>9} {'b':>9} {'slope':>9} {'corr':>8} {'n':>5}")
    hist, t0 = [], time.time()
    for it in range(a.steps):
        opt.zero_grad()
        terms, obs = [], []
        for j in range(a.seeds):
            L, c, k, n = loss_one(W, b, 4000 + it * a.seeds + j, a.frames, a.device,
                                  a.k0, a.gain, a.w_slope)
            terms.append(L)
            obs.append((*response(c, k), n))
        loss = torch.stack(terms).mean()
        loss.backward()
        slope, corr, n = obs[0]
        hist.append({"step": it, "loss": float(loss), "W_in": float(W), "b": float(b),
                     "slope": slope, "corr": corr, "n": n})
        print(f"  {it:>4} {float(loss):>10.5f} {float(W):>+9.4f} {float(b):>+9.4f} "
              f"{slope:>+9.4f} {corr:>+8.3f} {n:>5}", flush=True)
        opt.step()

    out = {"steps": a.steps, "seeds": a.seeds, "frames": a.frames, "lr": a.lr,
           "target": {"k0": a.k0, "gain": a.gain}, "device": a.device,
           "start": {"W_in": 0.35, "b": -0.6},
           "final": {"W_in": float(W), "b": float(b)},
           "wall_s": round(time.time() - t0, 1), "history": hist}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    flipped = float(W) < 0
    print(f"\n[invert] W_in {0.35:+.4f} -> {float(W):+.4f}   "
          f"{'SIGN FLIPPED: the activator became an inhibitor' if flipped else 'no sign change'}")
    print(f"[invert] {out['wall_s']}s -> {os.path.relpath(a.out, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
