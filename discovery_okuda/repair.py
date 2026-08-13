#!/usr/bin/env python
"""repair -- when a child breaks a premise its parent holds, diff them and name the suspects.

CEDRIC, 5 AUGUST: *"it has a parent that is ok so it can look at difference and test one of them?"*

That is the move the loop did not have. Round 2 posed twelve slots off `coral_gate`; all twelve
broke P1, P7 and P11, and the parent had been recorded in round 1 as *"specimen valid; every
applicable premise holds"*. Both halves were on disk and nothing joined them. The loop refused
twelve runs, rolled back, and stopped -- which is honest, and is not learning.

A refusal is not a diagnosis. The Biologist already produces the diagnosis, and it is a good one:
"volume went 522.1 -> 312.9", "the top 5% of cells reach shape index 5.83", "the surface folds
through itself at frame 50". What was missing is the step after: the child differs from a WORKING
parent in a small, enumerable set of ways, so each difference is a hypothesis for the failure and
reverting exactly one of them is a legal one-edit experiment.

WHY THE DIFF IS ARITHMETIC AND NOT JUDGEMENT. Two emitted specs, compared key by key. No agent is
needed to compute it, and no agent should: a diff computed by a model is a diff that can be wrong
in a way nobody checks. What needs judgement is only which difference to test FIRST, and the
ranking below hands that decision most of the way.

THE RANKING, AND WHY IT WOULD HAVE FOUND THIS ONE. A value outside its own declared range is the
first suspect, because the space itself says it should not be there. On r002c_00_9d40a8:

    edge_flip.l_th_frac   parent 0.28   child 1.96   declared range (0.01, 0.12)

`l_th_frac` is "threshold as a fraction of the mean edge length". At 1.96 the flip threshold is
nearly TWICE the mean edge, so every junction in the tissue is eligible to flip, every fourth
frame -- the mesh rearranges continuously, cannot hold a shape, drains volume, thins its cells and
folds through itself. All three premise failures are that one number seen from three angles, and
the chemistry was healthy throughout (act_cv_peak 3.97, act_alive_frac 1.0). Sixteen times its own
declared ceiling, and it sorts to the top of this list without anyone knowing what T1 flipping is.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _ops_by_id(spec):
    """Emitted spec -> {op_name: params}. Emitted specs carry no ids, so the op name is the key."""
    out = {}
    for o in (spec.get("operators") or []):
        out[o.get("op")] = {k: v for k, v in o.items() if k not in ("op", "at")}
    return out


def spec_diff(parent_spec, child_spec):
    """Every way two emitted specs differ, as [(op, key, parent_value, child_value)].

    `None` on either side means the operator or key exists on only one of them -- which is itself a
    difference worth testing, and the shape a missing `cell_chem_seed` or a dropped parameter takes.
    """
    pa, ch = _ops_by_id(parent_spec), _ops_by_id(child_spec)
    rows = []
    for op in sorted(set(pa) | set(ch)):
        if op not in pa:
            rows.append((op, "<operator>", None, "present")); continue
        if op not in ch:
            rows.append((op, "<operator>", "present", None)); continue
        a, b = pa[op], ch[op]
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                rows.append((op, k, a.get(k), b.get(k)))
    return rows


def _declared_range(op, key):
    """(lo, hi, default) for a parameter, from the search space -- or None if it declares none."""
    try:
        from composition_space import OPERATORS
        tri = (OPERATORS.get(op, {}).get("params") or {}).get(key)
        if isinstance(tri, (tuple, list)) and len(tri) == 3:
            return tuple(float(x) for x in tri)
    except Exception:
        pass
    return None


def rank_suspects(rows):
    """Order the differences by how likely each is to be the cause. Deterministic.

    Four tiers, and the top one is the whole point:

      OUT OF DECLARED RANGE   the space itself says the value should not be there. This is the
                              only tier that carries its own evidence, and it is where
                              l_th_frac = 1.96 against a ceiling of 0.12 lands.
      OPERATOR PRESENT/ABSENT  a structural difference, which changes what the tissue can do at
                              all -- bigger than any dial.
      LARGE RELATIVE CHANGE   ordered by |log ratio|, so a 7x is ranked above a 10%.
      EVERYTHING ELSE         strings, flags, and changes too small to be a first suspect.
    """
    scored = []
    for op, key, pv, cv in rows:
        tier, why, mag = 3, "", 0.0
        if key == "<operator>":
            tier, why = 1, ("the operator is ABSENT from the child" if cv is None
                            else "the operator is ABSENT from the parent")
        else:
            rng = _declared_range(op, key)
            if rng and isinstance(cv, (int, float)) and not isinstance(cv, bool):
                lo, hi = rng[0], rng[1]
                if cv < lo or cv > hi:
                    tier = 0
                    over = (cv / hi) if (hi > 0 and cv > hi) else (lo / cv if cv else float("inf"))
                    why = (f"OUTSIDE its declared range ({lo:g}, {hi:g}) by {over:.0f}x -- the "
                           f"search space itself says this value should not be there")
                    mag = over
            if tier == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                 for v in (pv, cv)):
                if pv and cv and pv * cv > 0:
                    import math
                    mag = abs(math.log(abs(cv) / abs(pv)))
                    if mag > 0.25:                      # more than ~1.3x either way
                        tier, why = 2, f"changed {abs(cv)/abs(pv):.2f}x"
        scored.append((tier, -mag, op, key, pv, cv, why))
    scored.sort()
    return [(op, key, pv, cv, why) for _t, _m, op, key, pv, cv, why in scored]


def repair_leads(parent_spec, child_spec, premises_broken=(), max_leads=5):
    """The suspects, as one-edit REVERTS the loop can pose next round.

    Each lead is a legal `set_param` back to the parent's value -- so testing it is an experiment
    with a prediction ("the premise will pass"), not a patch. That is the difference between
    repairing a specimen and learning why it broke: the revert is scored like anything else, and if
    the premise still fails the suspect is cleared.
    """
    rows = rank_suspects(spec_diff(parent_spec, child_spec))
    leads = []
    for op, key, pv, cv, why in rows[:max_leads]:
        if key == "<operator>":
            edit = ("add_op", op) if cv is None else ("remove_op", op)
            desc = f"restore the parent's structure: {edit[0]} {op}"
        else:
            edit = ("set_param", f"{op}.{key}", pv)
            desc = f"revert {op}.{key} from {cv!r} to the parent's {pv!r}"
        leads.append({"edit": edit, "why": why or "differs from a parent whose specimen is valid",
                      "op": op, "key": key, "parent": pv, "child": cv,
                      "predict": (f"the specimen passes again "
                                  f"({', '.join(str(p) for p in premises_broken) or 'the broken premises'})"),
                      "describe": desc})
    return leads


def brief(parent_name, child_name, leads, premises_broken=()):
    """The block handed to the Proposer. Short, ranked, and every line carries its number."""
    if not leads:
        return ""
    L = [f"A SPECIMEN BROKE THAT ITS PARENT HOLDS. `{child_name}` broke "
         f"{', '.join(premises_broken) or 'a premise'}; its parent `{parent_name}` was recorded "
         f"valid on every applicable premise. They differ in a small, enumerable set of ways, so "
         f"each difference is a HYPOTHESIS for the failure and reverting exactly one is a legal "
         f"one-edit experiment. Ranked, most suspect first:"]
    for i, d in enumerate(leads, 1):
        L.append(f"  {i}. {d['describe']}")
        L.append(f"     {d['why']}")
    L.append("A revert is a real experiment, not a patch: pose it with the prediction that the "
             "premise passes. If it still fails, that suspect is CLEARED and the next one is "
             "worth a slot -- which is knowledge either way.")
    return "\n".join(L)


if __name__ == "__main__":
    import json
    import yaml
    LOG = os.path.join(os.path.dirname(HERE), "log", "okuda")
    pa = sys.argv[1] if len(sys.argv) > 1 else "coral_gate"
    ch = sys.argv[2] if len(sys.argv) > 2 else "r002c_00_9d40a8"
    ps = yaml.safe_load(open(os.path.join(LOG, pa, "spec_run.yaml")))
    cs = yaml.safe_load(open(os.path.join(LOG, ch, "spec_run.yaml")))
    try:
        broken = json.load(open(os.path.join(LOG, ch, "diag.json"))).get("premises_broken") or []
    except Exception:
        broken = []
    print(brief(pa, ch, repair_leads(ps, cs, broken), broken))
