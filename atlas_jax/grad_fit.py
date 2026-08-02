"""grad_fit -- the smallest honest version of Figure 5: optimise a parameter THROUGH the engine.

Figure 5 of the paper trains a gene network until a growing cluster self-organises into a
specified structure. That is a big object. This is its irreducible core, and it is what has to
work before the big one is worth attempting:

    a target shape  ->  a loss on the final state  ->  a gradient through N frames of real
    Plexus physics (growth, adhesion, and division events)  ->  Adam  ->  a better parameter.

If this converges, Figure 5 is an engineering problem. If it does not, no amount of gene network
will help.

WHAT IS DELIBERATELY NOT DONE HERE. The discrete division draw is not differentiated -- the
straight-through surrogate carries a gradient to the RATE, but the number of divisions in a given
rollout is a sample, and optimising against one sample optimises against luck. The reference's
answer is its stochastic trace/replay/score contract (a score-function estimator over recorded
histories). Plexus does not have that contract; it is the single largest thing the atlas has found
that the language is missing, and Figure 5 proper needs it.

So this fits a CONTINUOUS parameter of a smooth operator, and reports the discrete part honestly.

    python grad_fit.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))
sys.path.insert(0, HERE)

import grad_probe  # noqa: E402


def rollout_with(max_radius, n_frames, spec_path, device="cpu"):
    """One differentiable rollout with `max_radius` (a tensor) driving cell growth."""
    import plexus.operators  # noqa: F401
    import run_spec
    run_spec.load_atlas_candidates()
    from plexus.models.registry import get_operator
    from plexus.schema import load

    with grad_probe.grad_enabled_engine() as engine:
        sim = load(spec_path)
        sim.n_frames = n_frames
        H = engine.build(sim, device)
        H.emit_order = engine._resolve_emit(sim, H)
        inst = []
        for o in sim.operators:
            params = {**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}
            if o.op == "grow_radius":
                params["max_radius"] = max_radius          # the tensor, on the tape
            inst.append((o.op, get_operator(o.op, o.impl)(params, device), o.on,
                         (int(o.params.get("after_frame", 0)),
                          int(o.params.get("before_frame", 1 << 30)))))
        for tick in range(n_frames + 1):
            H.frame = tick
            H.zero_delta()
            for step in sim.schedule:
                for token in (step if isinstance(step, list) else [step]):
                    for nm, ob, sel, (a, b) in inst:
                        if nm != token or not (a <= tick < b):
                            continue
                        for lname, d in (ob(H, engine._selector_mask(H, sel)) or {}).items():
                            H.add_delta(lname, d, getattr(ob, "INTEGRAND", None))
            engine._integrate(H, sim.dt)
        return H


def observables(H):
    lvl = H.level("cell")
    live = lvl.occ > 0
    a, b = lvl.state_schema["radius"]
    r = lvl.state[:, a:b].reshape(-1)[live]
    q = lvl.state[:, 0:2][live]
    gyr = ((q - q.mean(0)) ** 2).sum(1).mean().sqrt()
    return r.mean(), gyr, int(live.sum())


def main():
    spec = os.path.join(PLEXUS, "config", "atlas_jax", "jax_morph_proliferation.yaml")
    n_frames = 12
    target = 0.42                       # a target mean cell radius, chosen away from the default

    theta = torch.tensor(0.60, requires_grad=True)         # max_radius, the parameter we fit
    opt = torch.optim.Adam([theta], lr=0.05)

    print(f"target mean radius {target}   ({n_frames} frames, real engine, divisions on)\n")
    print(f"  {'step':>4}  {'max_radius':>10}  {'mean r':>8}  {'loss':>10}  {'|grad|':>9}  cells")
    for it in range(12):
        opt.zero_grad()
        H = rollout_with(theta, n_frames, spec)
        r_mean, gyr, n = observables(H)
        loss = (r_mean - target) ** 2
        loss.backward()
        g = float(theta.grad) if theta.grad is not None else float("nan")
        print(f"  {it:>4}  {float(theta):>10.4f}  {float(r_mean):>8.4f}  {float(loss):>10.3e}  "
              f"{abs(g):>9.3e}  {n}")
        opt.step()
        with torch.no_grad():
            theta.clamp_(min=0.05)

    H = rollout_with(theta, n_frames, spec)
    r_mean, gyr, n = observables(H)
    print(f"\n  converged: max_radius {float(theta):.4f} gives mean radius "
          f"{float(r_mean):.4f} against a target of {target}")
    print("  The gradient came through 12 frames of growth, adhesion and real division events.")
    print("  What it did NOT come through is the division DRAW -- see the module docstring.")


if __name__ == "__main__":
    main()
