"""audit -- which operators actually pass a gradient, measured rather than assumed.

`Operator.DIFFERENTIABLE` defaults to True and is inherited by every operator in the library.
Nothing has ever checked it. Before writing a differentiable twin for each operator it is worth
knowing which ones need a *surrogate* (the physics is genuinely non-smooth: a hard branch, a
discrete draw, an argmax) and which need only a tensor parameter -- because those are different
objects, and guessing which is which is how a "differentiable" library ends up with silent zero
gradients.

THE QUESTION IS ABOUT PARAMETERS, NOT ONLY STATE. An inverse loop fits an operator's constants,
so the load-bearing question is whether a gradient reaches the PARAMETER. Asking only about the
input state gives the wrong answer twice over: `gravity` returns a constant acceleration, so
d(delta)/d(state) is legitimately zero while d(delta)/d(g) is not, and an operator can pass state
gradients while baking its parameter in as a Python float that autograd never sees.

Per operator, on one synthetic Hierarchy: mark the state as a leaf, promote each declared tunable
to a tensor leaf, run the operator ONCE, reduce what it produced to a scalar, and ask autograd
about each leaf separately.

    GRAD       a non-zero gradient reached this leaf.
    ZERO       the tape reaches it and the gradient is identically zero -- real for a sensor whose
               output does not depend on the leaf, a bug otherwise.
    CONSTANT   what the operator produced does not depend on this leaf AT ALL (`requires_grad` is
               False downstream). Correct for `gravity` w.r.t. state; damning for a parameter.
    n/a        the harness could not exercise it -- a missing buffer, a field it needs, a second
               set. NOT a verdict: counted and listed separately, because an unexercised operator
               is unknown, not fine. A harness that scored its own failures as passes would report
               a fully differentiable library, which is the claim being checked.

    python -m plexus.operators.diff.audit
    python -m plexus.operators.diff.audit --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from plexus.models.base import Field, Hierarchy, Level
from plexus.models.registry import _OP_CONTRACTS

# state blocks the library's operators read, so one synthetic set exercises most of them
BLOCKS = {"pos": 2, "vel": 2, "radius": 1, "growth_rate": 1, "division_rate": 1,
          "gene": 3, "drive": 2, "chemical": 1, "secretion_rate": 1, "stress": 1,
          "heading": 1, "speed": 1, "s": 1, "age": 1, "mass": 1, "chem": 1, "a0": 1}

# per-node buffers operators read off the set besides `state` (REQUIRES_BUFFERS and friends)
BUFFERS = {"heading": 1, "speed": 1, "cohesion": 1, "separation": 1, "div_rate": 1,
           "mass": 1, "mu": 1, "la": 1, "p_vol": 1, "omega": 1, "youngs": 1,
           "growth_rate": 1, "remodel": 1, "alignment": 1}

# plausible stand-ins for required params the harness cannot infer. Values are arbitrary but
# non-degenerate -- a zero would make a real operator look inert.
PARAM_DEFAULTS = {"sigma": 1.0, "v0": 0.5, "k": 1.0, "tau": 1.0, "omega": 0.5, "g": 9.8,
                  "rate": 0.5, "amount": 1.0, "strength": 1.0, "radius": 1.0, "gain": 1.0,
                  "cutoff": 2.0, "eps": 1.0, "epsilon": 1.0, "alpha": 1.0, "beta": 1.0,
                  "system": "lorenz", "target": 1.0, "max_radius": 2.0, "width": 1.0,
                  "d": 1.0, "D": 1.0, "decay": 0.1, "speed": 0.5, "turn": 0.5}

# Attributes that are floats but are not physics: cadence, bookkeeping, the engine's own clock.
NOT_TUNABLE = {"device", "dt", "dt_sub", "substeps", "n_grid", "every", "after_frame",
               "before_frame", "compile", "maxnb", "seed"}


def float_attrs(op):
    """The operator's own Python-float attributes -- its tunables, as it actually stores them.

    Read off the INSTANCE rather than from `PARAM_ROLES`, because that dict also names string
    knobs (`mode`, `law`, `gate`) that are dispatch, not physics, and passing a tensor for one of
    those just raises. What an inverse loop can fit is exactly the set of numbers the forward
    multiplies by.
    """
    return sorted(k for k, v in vars(op).items()
                  if isinstance(v, float) and not k.startswith("_") and k not in NOT_TUNABLE)


def build_harness(n=12, buffer=48, dim=2, device="cpu", dt=0.1, rate=0.0):
    """One Hierarchy rich enough that most operators find what they read.

    Deliberately generic: not any model's state but a superset of the blocks and buffers the
    library declares, so an operator fails here for its OWN reasons rather than for something the
    harness forgot to provide.
    """
    schema, off = {}, 0
    for name, w in BLOCKS.items():
        schema[name] = (off, off + w)
        off += w
    g = torch.Generator(device=device).manual_seed(0)

    state = torch.rand(buffer, off, generator=g, device=device) * 0.5 + 0.25
    state[:, schema["pos"][0]:schema["pos"][1]] *= 20.0        # spread the cloud over the world
    occ = torch.zeros(buffer, device=device)
    occ[:n] = 1.0
    # A LATERAL operator with no relation has no pairs to act on, so it returns zeros and the
    # audit would report "CONSTANT" for the whole interaction family -- a verdict about the
    # harness, not about the operator. Give the set a real neighbour graph.
    p = state[:n, schema["pos"][0]:schema["pos"][1]]
    d = torch.cdist(p, p)
    d.fill_diagonal_(float("inf"))
    src, dst = (d < 8.0).nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst]) if src.numel() else torch.empty(2, 0, dtype=torch.long)

    lvl = Level("cell", state=state, occ=occ, state_schema=schema, edge_index=edge_index)
    for bname, w in BUFFERS.items():
        if hasattr(lvl, bname):
            continue
        v = torch.rand(buffer, generator=g, device=device) if w == 1 else \
            torch.rand(buffer, w, generator=g, device=device)
        lvl.register_buffer(bname, v)
    lvl.register_buffer("node_type", torch.zeros(buffer, dtype=torch.long, device=device))
    lvl.register_buffer("move_speed", torch.rand(buffer, generator=g, device=device))
    lvl.register_buffer("turn_speed", torch.rand(buffer, generator=g, device=device))
    lvl.register_buffer("F", torch.eye(dim, device=device).repeat(buffer, 1, 1))
    lvl.register_buffer("C", torch.zeros(buffer, dim, dim, device=device))
    lvl.type_names = ["a"]

    H = Hierarchy()
    H.dim = dim
    H.world_width = 40.0
    H.world_size = torch.tensor([40.0] * dim, device=device)
    H.add_level(lvl)

    nx = ny = 32
    fld = Field("chem")
    fld.grid = torch.rand(1, nx, ny, generator=g, device=device)
    fld.nx, fld.ny, fld.n = nx, ny, nx
    fld.width = 40.0
    fld.dx = 40.0 / nx
    fld.inv_dx = nx / 40.0
    H.add_field(fld)

    H.frame = 0
    H.rng = torch.Generator(device=device).manual_seed(0)

    class _Cfg:
        pass
    _Cfg.dt = dt
    H.config = _Cfg()
    H.zero_delta(dim)
    return H


def _event_driven(cls):
    """Operators whose work is gated on a random draw or a discrete event."""
    return getattr(cls, "KIND", None) in ("structural", "rewire")


def _acted(out, H, occ_before):
    """Did this call do anything at all? Same question `run_spec.py`'s acted ledger asks --
    a delta that moved something, or a structural operator that woke or retired a slot."""
    if any(torch.is_tensor(d) and d.numel() and float(d.abs().max()) > 0
           for d in (out or {}).values()):
        return True
    return float(H.level("cell").occ.sum()) != occ_before


def _fingerprint(out, H):
    """A detached signature of everything one forward produced, for a value comparison."""
    xs = [d.detach().reshape(-1) for d in (out or {}).values()
          if torch.is_tensor(d) and d.numel()]
    xs.append(H.level("cell").state.detach().reshape(-1))
    for f in H.fields.values():
        if torch.is_tensor(f.grid):
            xs.append(f.grid.detach().reshape(-1))
    return torch.cat(xs)


def _reads_param(cls, device, key, at, to, required):
    """Does the forward READ this knob at all, in this configuration?

    A parameter can be CONSTANT for two completely different reasons, and treating them alike is
    how a harness invents defects. `grow_radius.rate` is the uniform fallback used only when the
    set has no per-cell `growth_rate` block -- give the harness that block and the knob is never
    read, which says nothing about whether it is differentiable. Run it twice with two values and
    compare: identical output means the branch was not taken.
    """
    outs = []
    for scale in (1.0, 1.7):
        H = build_harness(device=device)
        params = {"_at": at, "to": to, "from": to, **required}
        op = cls(params, device)
        base = float(getattr(op, key))
        setattr(op, key, base * scale + 0.13)
        with torch.no_grad():
            outs.append(_fingerprint(op(H, None), H))
    if outs[0].shape != outs[1].shape:
        return True
    return not torch.equal(outs[0], outs[1])


def _verdict(loss, leaf):
    """One leaf, one answer. CONSTANT and ZERO are different findings and are kept apart."""
    if not loss.requires_grad:
        return "CONSTANT", 0.0
    g = torch.autograd.grad(loss, leaf, allow_unused=True, retain_graph=True)[0]
    if g is None:
        return "CONSTANT", 0.0
    n = float(g.abs().sum())
    return ("GRAD" if n > 0 else "ZERO"), n


def probe(cls, device="cpu"):
    """Run one operator on a fresh harness; report the gradient to state and to each tunable.

    The tunables are promoted to tensors AFTER construction. Every operator in the library writes
    `self.rate = float(params.get("rate", 0.0))`, so a tensor passed through the spec is coerced
    to a Python float before `forward` ever sees it -- the constructor, not the maths, is what
    stops the gradient. Injecting past the constructor separates the two questions: *is the
    forward differentiable in this knob* (here) from *can the spec express a learnable knob*
    (it cannot, today -- and that is the language change, reported separately).
    """
    last, rejects_tensor = None, False
    for promote in (True, False):
        for at, to in (("cell", "chem"), ("chem", "chem"), ("cell", "cell")):
            H = build_harness(device=device)
            lvl = H.level("cell")
            # The leaf is held SEPARATELY and the level gets a computed copy of it. Mid-rollout
            # `lvl.state` is never a leaf -- it is the output of the previous tick -- so a
            # structural operator's in-place write is legal there and merely recorded. Making the
            # level's own tensor the leaf turns every such write into a RuntimeError and would
            # report the whole structural family as unexercised.
            state_leaf = lvl.state.detach().clone().requires_grad_(True)
            lvl.state = state_leaf * 1.0
            params_required = {k: PARAM_DEFAULTS.get(k, 1.0)
                               for k in getattr(cls, "REQUIRES_PARAMS", []) or []}
            params = {"_at": at, "to": to, "from": to, **params_required}
            try:
                op = cls(params, device)
                leaves = {}
                if promote:
                    for k in float_attrs(op):       # past the constructor, into the forward
                        t = torch.tensor(float(getattr(op, k)), device=device,
                                         requires_grad=True)
                        setattr(op, k, t)
                        leaves[k] = t
                occ_before = float(H.level("cell").occ.sum())
                out = op(H, None)
                if _acted(out, H, occ_before) or not _event_driven(cls):
                    break
                # INERT: a stochastic/structural operator whose event never fired this frame
                # reports every knob as unread, which is a fact about the draw, not the operator.
                # Retry at a macro-step long enough that the hazard actually fires.
                H = build_harness(device=device, dt=5.0)
                lvl = H.level("cell")
                state_leaf = lvl.state.detach().clone().requires_grad_(True)
                lvl.state = state_leaf * 1.0
                op = cls(params, device)
                leaves = {}
                if promote:
                    for k in float_attrs(op):
                        t = torch.tensor(float(getattr(op, k)), device=device,
                                         requires_grad=True)
                        setattr(op, k, t)
                        leaves[k] = t
                out = op(H, None)
                break
            except Exception as e:                  # try the next binding before giving up
                last = e
                H = None
        if H is not None:
            break
        # It ran with plain floats and RAISED with tensor knobs -- e.g. `torch.full(shape, eps)`,
        # which will not take a tensor fill value. That is not an unknown: it is the operator
        # refusing a learnable parameter, and it names exactly what a twin has to change.
        rejects_tensor = True
    if H is None:
        raise last
    if rejects_tensor:
        return {"state": ("n/a", 0.0), "params": {}, "rejects_tensor": True}

    # a scalar touching everything the operator produced: a returned delta, or -- for a
    # structural / field / exchange operator that returns {} -- whatever it wrote
    terms = []
    for d in (out or {}).values():
        if torch.is_tensor(d) and d.numel():
            terms.append(d.float().pow(2).sum())
    if not terms:
        st = H.level("cell").state
        if torch.is_tensor(st):
            terms.append(st.float().pow(2).sum())
        for f in H.fields.values():
            if torch.is_tensor(f.grid):
                terms.append(f.grid.float().pow(2).sum())
    if not terms:
        return {"state": ("CONSTANT", 0.0), "params": {}}
    loss = sum(terms)
    pv = {}
    for k, t in leaves.items():
        v, mag = _verdict(loss, t)
        if v == "CONSTANT":
            # CONSTANT means one of two very different things. Separate them before reporting.
            try:
                v = "CONSTANT" if _reads_param(cls, device, k, at, to, params_required) \
                    else "NOT-READ"
            except Exception:
                pass
        pv[k] = (v, mag)
    return {"state": _verdict(loss, state_leaf), "params": pv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--candidates", action="store_true",
                    help="also audit the anti-chamber (candidates/jax_morph_*, atlas_*), which "
                         "`plexus.operators` deliberately does not auto-import")
    ap.add_argument("--only-impl", default=None,
                    help="restrict to implementations whose name contains this substring")
    a = ap.parse_args()

    import plexus.operators  # noqa: F401   self-registers the library
    if a.candidates:
        # The atlas's specs run on THESE, not on the promoted defaults: `grow_radius` resolves to
        # jax_morph_saturating_cell_growth, `cell_divide` to implementation `volume_conserving`.
        # Auditing only `plexus.operators` measures a set the atlas never runs.
        import importlib
        import os as _os
        import plexus.operators.candidates as C
        for fn in sorted(_os.listdir(_os.path.dirname(C.__file__))):
            if fn.startswith(("jax_morph_", "atlas_")) and fn.endswith(".py"):
                importlib.import_module(f"plexus.operators.candidates.{fn[:-3]}")

    rows, na = [], []
    for name in sorted(_OP_CONTRACTS):
        for impl, cls in sorted(_OP_CONTRACTS[name].implementations.items()):
            if a.only_impl and a.only_impl not in impl:
                continue
            try:
                r = probe(cls, a.device)
            except Exception as e:
                na.append({"op": name, "impl": impl,
                           "why": f"{type(e).__name__}: {str(e)[:88]}"})
                continue
            if r.get("rejects_tensor"):
                rows.append({"op": name, "impl": impl, "kind": getattr(cls, "KIND", None),
                             "declared": bool(cls.DIFFERENTIABLE), "state": "REJECTS-TENSOR",
                             "state_grad": 0.0, "params": {}, "fittable": False})
                continue
            pv = r["params"]
            # the operator is usable by an inverse loop if SOME tunable takes a gradient;
            # an operator with no declared tunable is judged on its state gradient alone
            fit = ("GRAD" in [v[0] for v in pv.values()]) if pv else (r["state"][0] == "GRAD")
            rows.append({"op": name, "impl": impl, "kind": getattr(cls, "KIND", None),
                         "declared": bool(cls.DIFFERENTIABLE),
                         "state": r["state"][0], "state_grad": r["state"][1],
                         "params": {k: v[0] for k, v in pv.items()},
                         "fittable": fit})

    print(f"{'operator':<24} {'impl':<16} {'kind':<11} {'state':<9} {'tunables':<34} fit")
    print("-" * 104)
    for r in rows:
        ps = ", ".join(f"{k}:{v}" for k, v in r["params"].items()) or "-- none declared --"
        print(f"{r['op']:<24} {r['impl']:<16} {str(r['kind']):<11} {r['state']:<9} "
              f"{ps[:34]:<34} {'yes' if r['fittable'] else 'NO'}")

    fit = [r for r in rows if r["fittable"]]
    unfit = [r for r in rows if not r["fittable"]]
    print(f"\nexercised {len(rows)} of {sum(len(c.implementations) for c in _OP_CONTRACTS.values())}"
          f":  fittable {len(fit)} · NOT fittable {len(unfit)}")
    for r in unfit:
        print(f"  NOT FITTABLE  {r['op']}/{r['impl']}  state={r['state']}  params={r['params']}")
    print(f"\nNOT EXERCISED (unknown, not fine): {len(na)}")
    for r in na:
        print(f"  {r['op']:<24} {r['impl']:<16} {r['why']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"measured": rows, "not_exercised": na}, f, indent=2)
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
