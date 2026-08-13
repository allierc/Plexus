"""instrument -- D4: make every operator report whether it ACTUALLY DID ANYTHING.

The most dangerous bug in an automatic mechanism search is an operator that silently no-ops.
It still returns metrics, the run still finishes, and the loop records

    "this mechanism cannot produce tubes"

when the mechanism never ran. Under hand-written recipes this was harmless, because a human wrote
the prerequisites in. Under a search that generates combinations no preset ever ran, it
manufactures false IMPOSSIBILITY claims -- the strongest and least recoverable kind of error the
campaign can make.

--------------------------------------------------------------------------------------------
HOW "DID IT ACT?" IS DECIDED
--------------------------------------------------------------------------------------------
Explicitly asking each operator to self-report would mean editing every operator and trusting
each edit. Instead we decide it the way an experimentalist would: **did the state change?**

Around every operator call we take a cheap fingerprint of the whole hierarchy -- per-level state
sum, occupancy sum, relation size, and the half-edge mesh's (Nv, nF, |E|). If the fingerprint
moved, or the operator returned a non-empty delta, it acted. Otherwise it was inert on that tick.

This is generic (no operator edits), honest (it measures effect, not intent), and cheap (a few
tensor reductions per operator per tick).

Two consequences the campaign relies on:
  * an operator whose precondition is unmet scores 0 acts -> the run is NOT evidence;
  * an operator that acts on only a handful of ticks is visible, which is how a mis-set
    `after_frame` or an exhausted reservoir shows up instead of hiding.
"""
from __future__ import annotations

import torch

_INSTALLED = False


def _fp_level(lvl):
    """A cheap, order-insensitive fingerprint of one Level."""
    try:
        v = float(lvl.state.sum().item())
        o = float(lvl.occ.sum().item()) if getattr(lvl, "occ", None) is not None else 0.0
        e = int(lvl.edge_index.shape[1]) if getattr(lvl, "edge_index", None) is not None else 0
        m = getattr(lvl, "_mesh", None)
        if isinstance(m, dict):
            # The mesh dict holds the per-cell mechanical TARGETS (A0, P0, V0f, v_eq ...).
            # Growth operators write those and nothing else, so a fingerprint that covers only
            # (Nv, nF, |E|) reports them INERT. That false positive would invalidate perfectly
            # good runs -- caught by running the detector on a known-good composition before
            # trusting it. Verify the instrument before trusting the measurement.
            nums = []
            for k in sorted(m):
                if k in ("hist", "E_srce", "E_trgt", "E_face", "Nv", "nF"):
                    continue
                val = m[k]
                try:
                    if hasattr(val, "sum"):
                        nums.append((k, round(float(val.sum()), 6)))
                except Exception:
                    pass
            mm = (int(m.get("Nv", 0)), int(m.get("nF", 0)),
                  int(len(m["E_srce"])) if m.get("E_srce") is not None else 0,
                  len(m.get("hist", [])), tuple(nums))
        else:
            mm = ()
        return (round(v, 6), round(o, 6), e, mm)
    except Exception:
        return ()


def fingerprint(H):
    try:
        return tuple(_fp_level(H.level(n)) for n in sorted(H.levels))
    except Exception:
        return ()


def install():
    """Wrap every registered operator class so each call records whether it acted.

    Idempotent. Call once, after the operator modules are imported and before the engine runs.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from plexus.models import registry as R

    seen = set()
    for name, cls in list(R._OPERATOR_REGISTRY.items()):
        for impl_cls in list(getattr(R._OP_CONTRACTS.get(name, None), "implementations",
                                     {}).values()) or [cls]:
            if id(impl_cls) in seen or not hasattr(impl_cls, "forward"):
                continue
            seen.add(id(impl_cls))
            _wrap(impl_cls)
    _INSTALLED = True


def _wrap(cls):
    orig = cls.forward
    if getattr(orig, "_acted_wrapped", False):
        return

    def forward(self, H, mask=None):
        before = fingerprint(H)
        out = orig(self, H, mask)
        after = fingerprint(H)
        # A ZERO DELTA IS NOT AN ACTION, and `bool(out)` said it was. An operator returning a
        # full-size tensor of zeros -- which is what EVERY lateral operator does when its parameter
        # has put it outside the regime where it does anything -- scored `acted = True`, so the
        # ledger reported it running on every frame of every run while it changed nothing.
        #
        # THIS IS THE GENERAL FORM OF A DEFECT THIS PROJECT HAS PAID FOR FIVE TIMES.
        # `cell_chem_from_shape` was edited 25 times across 13 rounds, 8 of them same-seed, and scored
        # 100% acted while not changing the run by a single bit. `rd_interface_tension` ran 800
        # frames at `a_sw = 1.0` -- cells strictly above the maximum, the empty set -- and was
        # written off as inert twice without ever having fired. Species B's chemistry was extinct
        # (act_max 0.0) in every two-species run, so its react operator returned ~0 forever and
        # three downstream results were read as being about the mechanism.
        #
        # The distinction that matters is INERT versus REFUTED. "This operator changed nothing" is
        # evidence about the OPERATING POINT; "this mechanism does not produce the effect" is
        # evidence about the MECHANISM. Scoring the first as the second is how a campaign spends
        # rounds refuting an operator that never ran, and the ledger is the only place that
        # distinction can be made cheaply -- so it has to be made here rather than trusted to a
        # reader downstream.
        def _nonzero(o):
            if not o:
                return False
            for v in (o.values() if isinstance(o, dict) else [o]):
                try:
                    if v is not None and bool((v != 0).any()):
                        return True
                except Exception:
                    return True                    # not a tensor: assume it means something
            return False

        acted = _nonzero(out) or (before != after)
        led = getattr(H, "_acted", None)
        if led is None:
            led = {}
            try:
                object.__setattr__(H, "_acted", led)
            except Exception:
                setattr(H, "_acted", led)
        names = getattr(self, "REGISTERED_NAMES", None) or [cls.__name__]
        key = names[0]
        rec = led.setdefault(key, {"acts": 0, "calls": 0})
        rec["calls"] += 1
        rec["acts"] += int(acted)
        return out

    forward._acted_wrapped = True
    cls.forward = forward


def report(H, scheduled):
    """{op: n_acts} for the scheduled operators, plus the inert list.

    An operator that appears in the schedule and never acted invalidates the run as evidence.
    """
    led = getattr(H, "_acted", {}) or {}
    acts = {op: int(led.get(op, {}).get("acts", 0)) for op in scheduled}
    inert = sorted([op for op, n in acts.items() if n == 0])
    return acts, inert


if __name__ == "__main__":
    import os
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "discovery_okuda", "ops"), HERE]
    import plexus.operators                                          # noqa: F401
    import mesh_ops, chem_ops, t1_ops, monolayer_ops, ckpt  # noqa: F401
    install()
    from plexus.models import registry as R
    n = sum(1 for _, c in R._OPERATOR_REGISTRY.items()
            if getattr(getattr(c, "forward", None), "_acted_wrapped", False))
    print(f"instrumented {n} registered operator classes (idempotent: install() again is a no-op)")
    install()
    print("D4 instrumentation OK")
