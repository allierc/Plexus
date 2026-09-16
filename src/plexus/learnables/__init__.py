"""Learnables: function approximators that stand in for an operator, and are fitted.

An operator says what a mechanism IS. A learnable says the mechanism is unknown and will be
recovered from data. They are the third variant axis of the registry, and what distinguishes the
three is WHO DETERMINES THE LAW:

    model            the modeller      a different biological hypothesis at this slot
    implementation   the numericist    the same law, computed differently
    learnable        THE DATA          the law is not stated; it is fitted

The forward spec does not change. A `learnable:` block names operators to substitute; everything
else about the model stays as written, so one file describes the analytic model and the fitted one
and the two can be run against each other. That is the point: a residual measured against a
stated law is interpretable, and a residual against nothing is not.

TWO SHAPES, BECAUSE THERE ARE TWO KINDS OF OPERATOR. What a learnable must do to stand in for an
operator is decided by the relations that operator traverses -- which `H.measured_maps()` reports,
having watched it run:

    intrinsic     the operator reads its own set's blocks and nothing else (`neuron_update`,
                  `organ_mechanics`). The learnable is a pointwise map over those blocks.
    relational    the operator gathers along an edge set and scatters back (`neuron_signal`,
                  `project`, `readout`). The learnable is a message function over the SAME edge
                  set: it sees the pre-state, the post-state and the edge weight, and its output
                  is aggregated along `post` exactly as the operator's was.

THE SECOND SHAPE IS THE WHOLE REASON THE CHECK EXISTS. A pointwise net standing in for
`neuron_signal` satisfies every declared field of it -- same sets, same blocks, same EMIT -- while
silently dropping the connectome. It would train, score well, and mean nothing. So a learnable
declares which shape it is, and substituting one whose shape does not match the operator's
measured relations is refused rather than allowed to produce a plausible number.

WHAT IS NOT NEGOTIABLE, checked in `check_substitution`, in order of how quietly each fails:

    EMIT, INTEGRAND   exactly. The engine resolves a whole SET's integration order from EMIT, so
                      a mismatch changes how that set moves in time with nothing in the output to
                      show for it.
    WRITES            exactly.
    READS             may NARROW but never widen. A learnable reading a block the operator does
                      not is a different model wearing its name.
    the relations     the measured ones, exactly.

Registered families are in `nets.py`. Add one by decorating it; a learnable that does not declare
`SHAPE` and the blocks it consumes cannot be checked, so it cannot be registered.
"""
from __future__ import annotations

from typing import Callable

_LEARNABLES: dict[str, Callable] = {}

INTRINSIC = "intrinsic"
RELATIONAL = "relational"
SHAPES = (INTRINSIC, RELATIONAL)


def register_learnable(name, **meta):
    """Register a learnable family. `meta` carries `family` and anything the catalog shows."""
    def wrap(cls):
        if name in _LEARNABLES:
            raise ValueError(
                f"learnable {name!r} is already registered by "
                f"{getattr(_LEARNABLES[name], '__module__', '?')}. Two approximators under one "
                f"name is the defect a registry exists to prevent.")
        shape = getattr(cls, "SHAPE", None)
        if shape not in SHAPES:
            raise ValueError(
                f"learnable {name!r} declares SHAPE={shape!r}; it must be one of {SHAPES}. "
                f"Without it there is no way to tell whether it may stand in for an operator "
                f"that traverses a relation, and a pointwise net replacing `neuron_signal` "
                f"would silently drop the connectome.")
        cls.learnable_name = name
        cls.learnable_meta = dict(meta)
        _LEARNABLES[name] = cls
        return cls
    return wrap


def get_learnable(name):
    if name not in _LEARNABLES:
        raise KeyError(
            f"unknown learnable {name!r}. Registered: {sorted(_LEARNABLES)}.")
    return _LEARNABLES[name]


def learnables() -> list:
    return sorted(_LEARNABLES)


class SubstitutionError(ValueError):
    """A learnable that cannot stand in for the operator it names.

    Its own class because these are the refusals the whole design turns on, and a caller
    (a search agent, a sweep) wants to tell "this substitution is ill-typed" apart from "this
    spec is malformed"."""


def check_substitution(op_cls, learn_cls, measured_maps, *, op_name="?", learn_name="?"):
    """Refuse a learnable that cannot stand in for `op_cls`. Returns the reason list (empty = ok).

    `measured_maps` is `H.measured_maps()[op_name]` -- the relations the operator ACTUALLY walked,
    watched by the engine, not a field anybody filled in. A declared one was tried and removed for
    being unmaintained on 23 of ~150 operators, and this is its first real consumer.
    """
    bad = []
    o_emit = getattr(op_cls, "EMIT", None)
    l_emit = getattr(learn_cls, "EMIT", "inherit")
    if l_emit != "inherit" and l_emit != o_emit:
        bad.append(f"EMIT: operator emits {o_emit!r}, learnable emits {l_emit!r}. The engine "
                   f"resolves the set's integration ORDER from this, so the substitution would "
                   f"change how the set moves in time.")
    o_integ = getattr(op_cls, "INTEGRAND", None)
    l_integ = getattr(learn_cls, "INTEGRAND", "inherit")
    if l_integ != "inherit" and l_integ != o_integ:
        bad.append(f"INTEGRAND: operator integrates into {o_integ!r}, learnable into {l_integ!r}")

    o_w, l_w = set(getattr(op_cls, "WRITES", [])), set(getattr(learn_cls, "WRITES", []) or [])
    if l_w and l_w != o_w:
        bad.append(f"WRITES: operator writes {sorted(o_w)}, learnable writes {sorted(l_w)}")

    o_r, l_r = set(getattr(op_cls, "READS", [])), set(getattr(learn_cls, "READS", []) or [])
    extra = l_r - o_r
    if extra:
        bad.append(f"READS: the learnable reads {sorted(extra)}, which {op_name!r} does not. "
                   f"Narrowing is allowed; widening is a different model wearing its name.")

    # THE RELATION. Everything above can match while this does not, and that is the case the
    # check exists for.
    walks = bool(measured_maps)
    shape = getattr(learn_cls, "SHAPE", None)
    if walks and shape != RELATIONAL:
        rels = ", ".join(f"{e}:{r}" for e, r in measured_maps)
        bad.append(
            f"RELATION: {op_name!r} traverses [{rels}] but {learn_name!r} is {shape!r}, which "
            f"reads only its own set's blocks. It would satisfy every declared field of the "
            f"operator and silently drop the structure the operator acts through -- and train "
            f"perfectly well doing it.")
    if not walks and shape == RELATIONAL:
        bad.append(
            f"RELATION: {op_name!r} traverses nothing, but {learn_name!r} is relational and "
            f"needs an edge set to gather along.")
    return bad


from plexus.learnables import nets          # noqa: E402,F401  self-registers the families

__all__ = ["register_learnable", "get_learnable", "learnables", "check_substitution",
           "SubstitutionError", "INTRINSIC", "RELATIONAL", "SHAPES", "nets"]
