"""The build-time units check: read the declaration, say what disagrees, and never stop the run.

WHEN IT RUNS. Once, inside `schema.load`, after the spec is parsed and every operator resolved,
before the hierarchy is built and before frame 0. That is the only moment at which the whole
declaration is visible at once -- the run's base scales, every set's state blocks, and every
scheduled operator's parameters -- and it is not inside anything that runs per frame, so it costs
one pass and cannot slow a simulation down.

WARNING ONLY. IT NEVER RAISES. A spec that fails every row still loads and still runs to
completion. This is not timidity: the first defect a checker of this kind finds is usually in the
checker, and a units annotation is the least load-bearing thing in any spec. A typo in one must
never cost a 600-frame run. Everything here is therefore wrapped -- a bug in this module prints one
line and the spec proceeds.

WHAT IT CAN AND CANNOT SEE. A spec supplies NUMBERS, not units, so there is nothing to compare a
parameter's value against. What is comparable is DECLARATION AGAINST DECLARATION:

  * the run's base scales against what the declared quantities need -- a stress cannot be quoted
    without `force_nN`, which is `units.py`'s own existing rule;
  * a state block's `unit:` against what an operator says it reads or writes there (`BLOCK_UNITS`),
    which is the check that catches wedge-against-polyhedron;
  * two operators disagreeing about the same block, which is the same check without a spec in the
    middle;
  * an annotation that does not parse, in either place -- a typo, not an absence;
  * a `1/T` parameter in a run whose `dt` is not 1, which is the shape of the `cell_grow.rate`
    defect: per frame and per unit time coincide at `dt: 1.0` and nowhere else.

WHAT IT DELIBERATELY DOES NOT DO is warn about absence. An undeclared block or parameter is
`UNKNOWN` and silent -- 145 of the tree's 177 operator classes are in that state by design, and a
checker that named them all would be switched off on its first run and would then catch nothing at
all. `tools/units_coverage.py` is where the fraction declared is reported, on purpose, so that
coverage is something you go and look at rather than something that shouts.
"""
from __future__ import annotations

from plexus.units import UNKNOWN, quantity, why_incompatible


def _fmt(units):
    """One line naming the run's scale, or the paper's own sentence for a dimensionless run."""
    if not getattr(units, "declared", False):
        return ("NONE DECLARED -- this run is dimensionless and no result from it may be quoted "
                "with a unit")
    f = "force NOT DECLARED (ratios only)" if units.force_nN is None else f"1 force unit = {units.force_nN:g} nN"
    return (f"1 length unit = {units.length_um:g} µm, 1 time unit = {units.time_s:g} s, {f}")


def check(sim, classes_by_op, emit=print):
    """Report on `sim`. `classes_by_op` is `{operator name -> class}` for everything scheduled.

    Returns the list of warning strings, so a caller (a test) can assert on them instead of
    scraping stdout. The printing is the point for a human; the return value is for the gate.
    """
    warn: list[str] = []
    # THE WHOLE BODY IS INSIDE THE GUARD, INCLUDING THE HEADER LINE. A first version put the
    # `emit` of the scale line outside it, on the reasoning that reading `sim.units` cannot fail --
    # and a spec object without that attribute raised straight through, which is the warn-only
    # guarantee broken by the one statement that looked too simple to break it. Caught by
    # `test_the_checker_never_raises_and_never_stops_a_load`, which exists for exactly this.
    try:
        warn = _check(sim, classes_by_op, warn)
        emit(f"[units] {_fmt(sim.units)}")
        for w in warn:
            emit(f"[units] {w}")
    except Exception as e:                       # noqa: BLE001 -- a checker must not break a load
        emit(f"[units] check skipped ({type(e).__name__}: {e}) -- the run is unaffected")
    return warn


def _check(sim, classes_by_op, warn):
    dt = float(getattr(sim, "dt", 1.0) or 1.0)

    # ---- 1. annotations that do not parse. A TYPO, NOT AN ABSENCE, and the only reason this is
    # reported at all: `parse_dim` swallows a malformed string into UNKNOWN so that a run is never
    # stopped, which means a misspelt unit would otherwise be indistinguishable from a missing one.
    for sname, s in (getattr(sim, "sets", None) or {}).items():
        for bname, decl in ((s.get("state") or {}) if isinstance(s, dict) else {}).items():
            u = decl.get("unit") if isinstance(decl, dict) else None
            if u is not None and quantity(u).dim is UNKNOWN:
                warn.append(f"sets.{sname}.{bname}.unit = {u!r} does not parse -- treated as unknown")

    # ---- 2. what each operator says about a BLOCK, against the spec and against each other.
    # `BLOCK_UNITS` is `{block name -> quantity}`: what this operator believes it is reading or
    # writing there. This is the check that catches a wedge volume compared with a polyhedron one,
    # and it is the reason the vocabulary carries a convention tag at all.
    claims: dict[str, list[tuple[str, str]]] = {}     # block -> [(operator, quantity string)]
    for opname, cls in (classes_by_op or {}).items():
        for bname, u in (getattr(cls, "BLOCK_UNITS", None) or {}).items():
            if quantity(u).dim is UNKNOWN and u is not None:
                warn.append(f"{opname}.BLOCK_UNITS[{bname!r}] = {u!r} does not parse")
            claims.setdefault(bname, []).append((opname, u))

    declared_blocks = {}
    for sname, s in (getattr(sim, "sets", None) or {}).items():
        for bname, decl in ((s.get("state") or {}) if isinstance(s, dict) else {}).items():
            u = decl.get("unit") if isinstance(decl, dict) else None
            if u:
                declared_blocks[bname] = (sname, u)

    for bname, entries in sorted(claims.items()):
        spec_set, spec_u = declared_blocks.get(bname, (None, None))
        if spec_u:
            for opname, u in entries:
                whyn = why_incompatible(spec_u, u)
                if whyn:
                    warn.append(f"{opname} reads {bname} as {u}, but sets.{spec_set}.{bname} "
                                f"declares {spec_u} -- {whyn}")
        # OPERATOR AGAINST OPERATOR, with no spec in the middle. Two mechanisms that disagree about
        # what a shared block holds is the same defect whether or not the spec ever said.
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                whyn = why_incompatible(entries[i][1], entries[j][1])
                if whyn:
                    warn.append(f"{entries[i][0]} and {entries[j][0]} disagree about {bname}: "
                                f"{entries[i][1]} vs {entries[j][1]} -- {whyn}")

    # ---- 3. a per-unit-time parameter in a run whose dt is not 1. THE SHAPE OF THE `cell_grow.rate`
    # DEFECT: the law applied `rate` once per call, so it meant "per frame", and per frame and per
    # unit time are the same number at `dt: 1.0` and at no other dt. This is informational rather
    # than an accusation -- a `1/T` parameter at `dt != 1` is CORRECT now, and the line is here so
    # that a reader of a `dt: 0.0032` spec sees which numbers the step size is scaling.
    if abs(dt - 1.0) > 1e-12:
        rated = []
        for o in (getattr(sim, "operators", None) or []):
            cls = (classes_by_op or {}).get(o.op)
            pu = getattr(cls, "PARAM_UNITS", None) or {}
            for k in (o.params or {}):
                if k in pu and quantity(pu[k]).dim == quantity("rate").dim:
                    rated.append(f"{o.op}.{k}")
        if rated:
            warn.append(f"dt = {dt:g}, so these per-unit-time parameters are scaled by it each "
                        f"frame: {', '.join(sorted(set(rated)))}")

    return warn
