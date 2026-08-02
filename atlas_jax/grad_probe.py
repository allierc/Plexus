"""grad_probe -- can a gradient survive a Plexus rollout at all?

Phase 6 is the paper's Figure 5: optimise a gene network until a growing cluster self-organises
into a specified shape. That is not a plotting exercise, it is the inverse half of Plexus, and it
rests on one precondition nobody has tested: **is the forward engine differentiable?**

It is not, today. `plexus.engine.run` wraps its entire tick loop in `torch.no_grad()`, so every
gradient is discarded at the first frame -- by construction, silently, and for a good reason
(forward generation has no use for a tape and pays for it in memory).

`prototype/inverse_slime/` worked around this by re-implementing the field tick as a separate
differentiable function, validated against the engine to ~1e-7. That is a sound pattern for four
parameters and an expensive one for a whole model: two implementations of the same physics, kept
in step by hand.

This probe measures the alternative -- run the SAME engine with the guard lifted -- and answers
three questions, in order, because each only matters if the previous one passed:

  1. Does a gradient reach the initial state through N frames of the real engine?
  2. Does it reach an operator PARAMETER, if that parameter is stored as a tensor?
  3. What does a discrete event (division) do to it -- vanish, or merely go non-smooth?

    python grad_probe.py
"""
from __future__ import annotations

import contextlib
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))
sys.path.insert(0, HERE)


@contextlib.contextmanager
def grad_enabled_engine():
    """Lift the engine's `no_grad` for the duration of one run.

    A patch, not a fix: the honest version is an opt-in `grad=True` argument on `engine.run`, so
    generation keeps its memory back and the inverse path asks for the tape explicitly. This
    probe exists to find out whether that argument is worth adding.
    """
    from plexus import engine
    real = torch.no_grad
    torch.no_grad = torch.enable_grad          # what the engine calls, for this run only
    try:
        yield engine
    finally:
        torch.no_grad = real


def rollout(spec_path, n_frames, seed_pos_grad=True, param_grad=None, device="cpu",
            only=None):
    import importlib

    import plexus.operators  # noqa: F401
    import run_spec
    run_spec.load_atlas_candidates()
    importlib.invalidate_caches()

    from plexus.schema import load
    with grad_enabled_engine() as engine:
        sim = load(spec_path)
        sim.n_frames = n_frames
        H = engine.build(sim, device)
        lvl = H.level("cell")
        if seed_pos_grad:
            lvl.state = lvl.state.detach().clone().requires_grad_(True)
        leaf = lvl.state

        # the engine's own loop, minus the recording -- so this measures the ENGINE, not a
        # re-implementation of it (which is the thing inverse_slime had to accept)
        H.emit_order = engine._resolve_emit(sim, H)
        from plexus.models.registry import get_operator
        inst = [(o.op,
                 get_operator(o.op, o.impl)({**o.params, "to": o.to, "from": o.frm,
                                             "_at": o.on.set}, device),
                 o.on,
                 (int(o.params.get("after_frame", 0)),
                  int(o.params.get("before_frame", 1 << 30))))
                for o in sim.operators]
        if param_grad:
            name, attr = param_grad
            for nm, ob, _, _ in inst:
                if nm == name:
                    setattr(ob, attr, torch.tensor(float(getattr(ob, attr)), device=device,
                                                   requires_grad=True))
                    leaf = getattr(ob, attr)

        if only is not None:
            inst = [x for x in inst if x[0] in only]
            sim.schedule = [t for t in sim.schedule if t in only]
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
        return H, leaf


def report(name, loss, leaf):
    g = torch.autograd.grad(loss, leaf, retain_graph=False, allow_unused=True)[0]
    if g is None:
        print(f"  {name:<44} NO GRADIENT (the tape does not reach it)")
        return False
    n = float(g.abs().sum())
    print(f"  {name:<44} |grad| = {n:.4g}   {'flows' if n > 0 else 'ZERO'}")
    return n > 0


def gyration(H):
    lvl = H.level("cell")
    q = lvl.state[:, 0:2][lvl.occ > 0]
    return ((q - q.mean(0)) ** 2).sum(1).mean()


def main():
    spec = os.path.join(PLEXUS, "config", "atlas_jax", "jax_morph_proliferation.yaml")

    # Which operators can a gradient pass through? Add them one at a time: an in-place write
    # anywhere in the schedule poisons the whole tape, so the only way to attribute the damage is
    # to grow the schedule until it breaks.
    print("0. one operator at a time, 4 frames, gradient to the initial state")
    for subset in (["seed_state"], ["relax"], ["grow_radius"], ["cell_divide"],
                   ["seed_state", "relax"], ["seed_state", "relax", "grow_radius"],
                   ["seed_state", "relax", "grow_radius", "cell_divide"]):
        try:
            H, leaf = rollout(spec, 4, only=subset)
            loss = gyration(H)
            g = torch.autograd.grad(loss, leaf, allow_unused=True)[0]
            n = 0.0 if g is None else float(g.abs().sum())
            print(f"  {'+'.join(subset):<46} |grad| = {n:.4g}"
                  f"{'   NO GRADIENT' if g is None else ''}")
        except RuntimeError as e:
            print(f"  {'+'.join(subset):<46} BREAKS THE TAPE: {str(e)[:80]}")

    print("\n1. gradient from a final-state loss back to the INITIAL STATE, 6 frames")
    H, leaf = rollout(spec, 6)
    pos = H.level("cell").state[:, 0:2]
    live = H.level("cell").occ > 0
    q = pos[live]
    loss = ((q - q.mean(0)) ** 2).sum(1).mean()            # radius of gyration, squared
    ok1 = report("d(gyration^2) / d(initial state)", loss, leaf)

    print("\n2. gradient to an OPERATOR PARAMETER (adhesion strength), 6 frames")
    try:
        H, leaf = rollout(spec, 6, seed_pos_grad=False, param_grad=("relax", "epsilon"))
        pos = H.level("cell").state[:, 0:2]
        live = H.level("cell").occ > 0
        q = pos[live]
        loss = ((q - q.mean(0)) ** 2).sum(1).mean()
        report("d(gyration^2) / d(epsilon)", loss, leaf)
    except Exception as e:
        print(f"  parameter probe failed: {type(e).__name__}: {e}")

    print("\n3. does the gradient survive a division event? 20 frames (divisions do occur)")
    H, leaf = rollout(spec, 20)
    lvl = H.level("cell")
    live = lvl.occ > 0
    print(f"  live cells after 20 frames: {int(live.sum())} (started at 4)")
    pos = lvl.state[:, 0:2]
    q = pos[live]
    loss = ((q - q.mean(0)) ** 2).sum(1).mean()
    report("d(gyration^2) / d(initial state), post-division", loss, leaf)

    print("\nWhat this means for Phase 6 is in the note; the short version is that the engine's "
          "\n`no_grad` is the only thing in the way of a differentiable rollout, and it is one "
          "\nopt-in argument, not a re-implementation.")


if __name__ == "__main__":
    main()
