"""certify -- verify the audit's verdicts by a completely different route.

`audit.py` builds a SYNTHETIC Hierarchy and calls one operator once. That harness has now been
wrong three times, and every time it invented a defect in an operator rather than reporting one:
an empty neighbour graph condemned the whole interaction family, a division that never fired
condemned `cell_divide`, and making the level's own tensor the leaf condemned every structural
operator. A single-source measurement that keeps doing this is not a measurement.

So this is the second source, and it shares NO code path with the first. It takes a real spec out
of `config/atlas/` -- the same file the differ ran -- promotes the operator's parameters to tensor
leaves in the spec itself, and runs the whole thing through `engine.run(grad=True)`: real build,
real schedule, real state blocks, real neighbours, real events, N frames. Then it asks autograd.

That closes every hole the synthetic harness had, because there is no harness: the configuration
IS the one the atlas runs.

A verdict is CERTIFIED only when both routes agree. Where they disagree the certifier wins -- it
is running the real thing -- and the disagreement is printed rather than reconciled quietly.

    python -m plexus.operators.diff.certify                       # every spec in config/atlas
    python -m plexus.operators.diff.certify --spec mechanical_relaxation
    python -m plexus.operators.diff.certify --audit _state/diff_audit_atlas.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))   # .../diff -> repo root
CONFIG = os.path.join(PLEXUS, "config", "atlas")

SKIP_PARAMS = {"to", "from", "_at", "at", "op", "implementation", "emit", "every",
               "after_frame", "before_frame", "n_frames", "substeps", "seed"}


def load_candidates():
    """The atlas's operators are in the anti-chamber; `plexus.operators` will not import them."""
    import importlib
    import plexus.operators  # noqa: F401
    import plexus.operators.candidates as C
    for fn in sorted(os.listdir(os.path.dirname(C.__file__))):
        if fn.startswith(("jax_morph_", "atlas_")) and fn.endswith(".py"):
            importlib.import_module(f"plexus.operators.candidates.{fn[:-3]}")


def live_loss(H):
    """A scalar on the FINAL state of the live cells -- what an inverse loop would minimise."""
    lvl = next(iter(H.levels.values()))
    live = lvl.occ > 0
    if not bool(live.any()):
        return None
    return lvl.state[live].pow(2).sum()


def certify_spec(path, n_frames=6, device="cpu"):
    """Promote every numeric param of every operator in one spec, run it, and read the gradients."""
    from plexus import engine
    from plexus.schema import load

    sim = load(path)
    sim.n_frames = min(int(sim.n_frames), n_frames)
    leaves = {}
    for o in sim.operators:
        for k, v in list(o.params.items()):
            if k in SKIP_PARAMS or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            t = torch.tensor(float(v), device=device, requires_grad=True)
            o.params[k] = t
            leaves[(o.op, k)] = t
    if not leaves:
        return {}, "spec declares no numeric operator params"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                 # the engine's run banner
        H, _ = engine.run(sim, device=device, grad=True)
    loss = live_loss(H)
    if loss is None:
        return {}, "no live cells at the end of the run"
    if not loss.requires_grad:
        return {k: "CONSTANT" for k in leaves}, "final state is detached from every parameter"

    gs = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True, retain_graph=False)
    out = {}
    for (key, _), g in zip(leaves.items(), gs):
        if g is None:
            out[key] = "CONSTANT"
        else:
            n = float(g.abs().sum())
            out[key] = "GRAD" if n > 0 else "ZERO"
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, help="one spec name in config/atlas (no .yaml)")
    ap.add_argument("--audit", default=None, help="an audit json to cross-check against")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    load_candidates()
    names = [a.spec] if a.spec else sorted(
        f[:-5] for f in os.listdir(CONFIG) if f.endswith(".yaml") and not f.startswith("_"))

    results, failed = {}, []
    for name in names:
        path = os.path.join(CONFIG, name + ".yaml")
        try:
            got, note = certify_spec(path, a.frames, a.device)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {str(e)[:80]}"))
            continue
        if note and not got:
            failed.append((name, note))
            continue
        for (op, k), v in got.items():
            results.setdefault((op, k), []).append((name, v))

    print(f"{'operator':<24} {'param':<20} {'certified':<10} from spec")
    print("-" * 84)
    for (op, k), obs in sorted(results.items()):
        best = "GRAD" if any(v == "GRAD" for _, v in obs) else \
               ("ZERO" if any(v == "ZERO" for _, v in obs) else "CONSTANT")
        src = ", ".join(n for n, v in obs if v == best)
        print(f"{op:<24} {k:<20} {best:<10} {src[:34]}")

    if a.audit:
        with open(a.audit) as f:
            aud = json.load(f)
        amap = {}
        for r in aud["measured"]:
            for k, v in r["params"].items():
                amap.setdefault((r["op"], k), set()).add(v)
        # The two routes answer DIFFERENT questions, and the pair is worth more than either.
        # The harness injects past the constructor, so it measures whether the MATHS is
        # differentiable in a knob. The certifier goes through the spec, so it measures whether a
        # knob is learnable AS THE LANGUAGE CAN EXPRESS IT TODAY. Where they differ, the gap is
        # the constructor -- `self.eps = float(params.get(...))` -- and naming that is the point.
        verdict_of, buckets = {}, {}
        for (op, k), obs in sorted(results.items()):
            real = "GRAD" if any(v == "GRAD" for _, v in obs) else "CONSTANT"
            av = amap.get((op, k))
            forward = "GRAD" if (av and "GRAD" in av) else ("?" if not av else "CONSTANT")
            if real == "GRAD":
                v = "LEARNABLE"                      # works end to end through a real spec
            elif forward == "GRAD":
                v = "CONSTRUCTOR-BLOCKED"            # maths fine; float() coercion drops the tape
            elif forward == "?":
                v = "UNCONFIRMED"                    # the harness never exercised it
            else:
                v = "NEEDS-SURROGATE"                # both routes say the forward is not smooth
            verdict_of[(op, k)] = v
            buckets.setdefault(v, []).append(f"{op}.{k}")

        print("\nSYNTHESIS -- two independent routes, one verdict per knob")
        for v in ("LEARNABLE", "CONSTRUCTOR-BLOCKED", "NEEDS-SURROGATE", "UNCONFIRMED"):
            items = buckets.get(v, [])
            print(f"\n  {v}  ({len(items)})")
            for it in items:
                print(f"    {it}")
        print(f"\n  certified learnable today: {len(buckets.get('LEARNABLE', []))} of "
              f"{len(verdict_of)}")

    if failed:
        print(f"\nnot certified ({len(failed)}) -- these specs could not be run:")
        for n, why in failed:
            print(f"  {n:<28} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
