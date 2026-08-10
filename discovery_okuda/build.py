#!/usr/bin/env python
"""build -- rebuild a finished run as a composition, and read its numbers safely.

The 252 lines of `round.py` (2,504) that the new loop actually reached. `graph_from_run` recovers a
run's wiring from its spec -- the only code that can, since a spec records operators but not
connections. `read_diag_summary` keeps the Q quarantine, which I nearly deleted after finding no Q
keys in `coral_gate`: 34 of 48 runs carry them, including every campaign run. Measured, not assumed.
"""
from __future__ import annotations

import json
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import critic as C                                             # noqa: E402
from composition_space import CompositionGraph, OPERATORS      # noqa: E402,F401

LOG = os.path.join(os.path.dirname(HERE), "log", "okuda")

# THE TWO CONSTANTS I FAILED TO PORT, and the whole quarantine crashed without them. `read_diag_summary`
# raised NameError on any summary carrying a Q key -- 34 of the 48 runs on disk -- and `measure()` does
# not catch it, so the round's metrics node would have returned None and every prediction scored
# inconclusive. It did not fire yet only because none of the six pool parents has a Q key; the moment a
# run is launched with the relax probe, it would have taken the round out. Found by checking the stopped
# run rather than by anything failing.
SEED_SPHERE_Q = 1.014          # protr of the relaxed SEED SPHERE -- what the broken test measured
SEED_SPHERE_Q_TOL = 5e-4       # run_one rounds Q to 3 dp, so this is exact-match with slack
Q_KEY = "Q_protr_after_relax"

Q_DERIVED = ("Q_drop",)

STALE_SUFFIX = "__STALE"
# THE THIRD CONSTANT THE PORT DROPPED, found by the test written for the first two. `_quarantine_log`
# appends here every time a poisoned Q is intercepted, so the interceptions are countable rather than
# inferred. Deliberately in campaign/ and NOT in _archive/: the archive is the immutable research
# record, and this is a log of what the reader refused.
Q_QUARANTINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "campaign", "q_quarantine.jsonl")

def stale_q_reason(summary):
    """Why this summary's Q must not be scored, or None if it may be.

    The test is on the VALUE, not on provenance, because provenance is not recoverable: the
    broken and the fixed `quasi_static_Q` both write `metric_version="metric_v1"`, so no record
    can say which code produced it. A Q sitting on the seed-sphere constant is therefore
    INDISTINGUISHABLE from the artefact -- and indistinguishable resolves to STALE, never to
    trusted. Refusing a good value costs one re-measurement; accepting a poisoned one costs a
    conclusion, which is a bill this campaign has already paid.

    The corroborating evidence is reported with the reason so a human re-scoring the archive can
    see how strong each case is: a run that ended at protr_final 2.805 and "relaxed" to 1.014 is
    the seed sphere beyond argument, whereas one that ended at 1.024 is merely unprovable.
    """
    if not isinstance(summary, dict):
        return None
    q = summary.get(Q_KEY)
    if q is None:
        return None
    try:
        qf = float(q)
    except (TypeError, ValueError):
        return f"{Q_KEY}={q!r} is not numeric -- cannot be scored"
    if abs(qf - SEED_SPHERE_Q) > SEED_SPHERE_Q_TOL:
        return None
    fin = summary.get("protr_final")
    corr = ""
    try:
        if fin is not None and abs(float(fin) - qf) > 0.15:
            corr = (f"; the run ended at protr_final={float(fin):.3f}, so the relaxation did not "
                    f"start where the run finished")
    except (TypeError, ValueError):
        pass
    return (f"STALE {Q_KEY}={qf:.3f}: this is the relaxed-seed-sphere constant produced by the "
            f"pre-fix quasi_static_Q, which rebuilt the simulation from the seed instead of "
            f"continuing from the end state{corr}. Quarantined -- recompute before scoring.")

def _quarantine_log(entry, ledger_path=None):
    """Append one line to the quarantine ledger. Never raises into the caller.

    Append-only, and outside _archive/ and log/okuda/, because an interception is a NEW fact
    about a record -- not a licence to edit the record.
    """
    p = ledger_path or Q_QUARANTINE
    try:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")
        return p
    except Exception as e:                       # a failed audit line must not lose the scrub
        print(f"  [Q-stale] could not append to the quarantine ledger {p}: "
              f"{type(e).__name__}: {e}")
        return None

def scrub_stale_q(summary, source="", ledger_path=None, quiet=False):
    """Return a COPY of `summary` with any poisoned Q moved OUT OF SCORING REACH.

    The value is not destroyed -- it moves to `Q_protr_after_relax__STALE` (and `Q_drop__STALE`)
    and the reason rides alongside as `Q_stale_reason`. Downstream this means:
      * control.score_run     sees no Q and takes its documented `else fin` fallback, instead of
                              adding the sphere constant with weight 1.0
      * control.meets_success likewise falls back rather than testing 1.014 >= 2.0
      * predict.score         reports `Q_drop not measured` -> inconclusive, rather than scoring
                              a prediction against an artefact (that clause is already dead for
                              an unrelated reason -- see the case-fold note above -- so this is
                              the path being held shut, not one being reopened)
    Never mutates the caller's dict; never writes to the file the summary came from.
    """
    reason = stale_q_reason(summary)
    out = dict(summary or {})
    if not reason:
        return out
    moved = {}
    for k in (Q_KEY,) + Q_DERIVED:
        if k in out:
            moved[k] = out.pop(k)
            out[k + STALE_SUFFIX] = moved[k]
    out["Q_stale"] = True
    out["Q_stale_reason"] = reason
    if source:
        out["Q_stale_source"] = source
    if not quiet:
        print(f"  [Q-stale] {source or 'summary'}: {reason}")
    _quarantine_log({"source": source, "quarantined": moved, "reason": reason,
                     "protr_final": summary.get("protr_final"),
                     "protr_peak": summary.get("protr_peak")}, ledger_path)
    return out

def read_diag_summary(path, source=None, quiet=False):
    """Read a run's diag.json `summary` with the stale-Q quarantine already applied."""
    d = json.load(open(path)).get("summary", {})
    return scrub_stale_q(d, source or path, quiet=quiet)

def _graph_from_run(name):
    """Rebuild the composition a finished run was launched with, from its own spec on disk."""
    import composition_space as CS
    p = os.path.join(LOG, name, "composition.json")
    if os.path.exists(p):
        try:
            r = json.load(open(p))
            return CS.CompositionGraph(ops=r["ops"], conns=r["conns"], params=r["params"])
        except Exception:
            pass
    # SECOND CHANCE: REBUILD IT FROM THE SPEC. A recon slot copies its spec VERBATIM and so never
    # gets a composition sidecar -- `write_config` is the only writer of one. The consequence was
    # measured on 3 August: twelve replays finished cleanly, the Archivist correctly named the
    # three best starting points on disk, could reach NONE of them, and round 2 began from the
    # reference recipes as though round 1 had never run.
    #
    # The spec does carry the operator list with its implementations and every parameter. What it
    # does not carry is `conns`, so the rebuild is HONEST ABOUT BEING PARTIAL: a graph with no
    # explicit wiring, which is right for these pipelines (the routing is implicit in each
    # operator's `at` / `cell_set` / `vertex_set`) and would be wrong for one that branches. A
    # partial parent that can be edited beats a perfect parent that does not exist.
    # AND FALL BACK TO THE CONFIG THE RUN WAS BUILT FROM. `spec_run.yaml` is the run's own record
    # and is the right source, but until 10 August it was written only when `--frames` was passed
    # -- which the round's job script does not do -- so sixteen finished runs existed with no
    # recoverable spec at all and the round that should have inherited from them launched nothing.
    # The config in config/okuda/<name>.yaml is what the run was LAUNCHED with; it differs from
    # spec_run.yaml only by a command-line frame override, and a parent rebuilt from it is the
    # composition the run actually had. Preferring spec_run.yaml keeps the record authoritative
    # where it exists.
    sp = os.path.join(LOG, name, "spec_run.yaml")
    if not os.path.exists(sp):
        _cfg = os.path.join(os.path.dirname(HERE), "config", "okuda", f"{name}.yaml")
        if os.path.exists(_cfg):
            sp = _cfg
    if os.path.exists(sp):
        try:
            import yaml as _y
            import composition_space as _CS
            spec = _y.safe_load(open(sp)) or {}
            ops, seen_op, params, skipped, clock_dropped = [], {}, {}, [], []
            for o in (spec.get("operators") or []):
                nm = o.get("op")
                if not nm:
                    continue
                # ONLY WHAT THE SEARCH SPACE KNOWS. A spec also carries INSTRUMENTATION --
                # `topo_snapshot_3d` and friends -- which record the run and are not moves anyone
                # can make. Carrying them into a graph raises KeyError the moment legal_edits()
                # asks for their role, and more importantly it would offer the Proposer edits on
                # apparatus rather than on biology.
                if nm not in _CS.OPERATORS:
                    skipped.append(nm)
                    continue
                idx = seen_op.get(nm, 0)
                seen_op[nm] = idx + 1
                oid = f"{nm}{idx}"
                # "default" IS NOT AN IMPLEMENTATION. A spec omits `implementation` when the
                # operator has only one, so the name has to come from the space, not from a
                # placeholder -- `seed_mesh_3d:default` compiles to nothing and refuses the parent.
                _impls = (_CS.OPERATORS[nm].get("impls") or ["default"])
                _im = o.get("model") or o.get("implementation") or o.get("impl")
                ops.append({"id": oid, "op": nm,
                            "impl": _im if _im in _impls else _impls[0]})
                for k, v in o.items():
                    if k in ("op", "implementation", "impl", "at"):
                        continue
                    # THE CLOCK-COUPLED PARAMETERS ARE NOT PORTABLE, and carrying them is what
                    # blew up round 2. The archived specs run `every: 4`; a rebuilt graph runs
                    # `every: 1`. `max_div_frac = 0.03` therefore means 0.03 PER CALL there and
                    # 0.03 PER FRAME here -- FOUR TIMES the proliferation. Measured: 2000 cells
                    # became 48,459 by frame 214, and the projection 2000 x 1.03^800 = 3.7e13
                    # against a 65,004 reservoir, so the array filled at frame 118 of 800 and
                    # every job spent 85% of its wall time pinned against it.
                    #
                    # composition_space.CLOCK_COUPLED names them and states the conversion, and
                    # this function ignored it. Rather than apply a factor that is only exact for
                    # the archived period, drop them: the vocabulary defaults ARE the re-anchored
                    # working point (max_div_frac 0.0075 per frame, not 0.03 per call).
                    if k in _CS.CLOCK_COUPLED or k == "every":
                        clock_dropped.append(f"{oid}.{k}")
                        continue
                    params[f"{oid}.{k}"] = v
            if ops:
                g = _CS.CompositionGraph(ops=ops, conns=[], params=params)
                # WIRE WHAT IS UNAMBIGUOUS. A spec records operators, not connections, so every
                # rebuilt graph arrived with its required slots dangling and was refused -- which
                # is the last thing standing between the frontier and our own measured runs.
                #
                # For a dangling slot, ask which operators in THIS graph could fill it. Exactly
                # one candidate is not a guess; it is the only wiring the composition admits, and
                # it is what the spec must have meant. Two or more IS a guess, and is left
                # dangling so the Critic says so -- a silently mis-wired parent would breed a
                # whole branch of experiments about a composition nobody chose.
                for _ in range(6):
                    dangling = g.unrouted_slots()
                    if not dangling:
                        break
                    made = False
                    for nid, _op, slot in dangling:
                        cands = [(a, b, sl) for a, b, sl in g._candidate_links()
                                 if b == nid and sl == slot]
                        if len(cands) > 1:
                            # NOT A GUESS -- THE PIPELINE'S OWN ORDER. seed_cell_rd and cell_react
                            # both "produce" morphogen because they both write the same shared
                            # field; there is no choice between them in the engine, only a
                            # sequence. The effective source for a consumer is the LAST producer
                            # ahead of it in the spec, which is exactly what running the operators
                            # in order does. Ordering by spec position recovers that.
                            _pos = {o["id"]: k for k, o in enumerate(g.ops)}
                            cands = [max((c for c in cands if _pos[c[0]] < _pos[nid]),
                                         key=lambda c: _pos[c[0]], default=cands[0])]
                        if len(cands) == 1:
                            g, _ = g.apply(("connect",) + cands[0])
                            made = True
                    if not made:
                        break
                # AND WIRE THE GATE THE SPEC SAYS IS THERE. `grow_3d.gate` is an OPTIONAL slot,
                # so `unrouted_slots()` never reports it and the loop above never considers it --
                # a rebuilt parent therefore always came back with the gate unwired, whatever the
                # run actually did. The physics was unaffected (grow_3d reads cell.chem directly,
                # so the Hill term fires either way), but `name_region()` decides gated-versus-
                # ungated from `conns`, so EVERY rebuilt parent was labelled "uniform growth
                # (ungated, rho baseline only)" -- including fifteen of sixteen slots in a round
                # whose specs all carry a_sw = 0.35. That label is what the Proposer and the
                # Analyst are handed as the description of the composition, and it was false.
                #
                # The spec carries the evidence: `a_sw > 0` means the Hill switch selects on the
                # activator, which IS the gate. `a_sw = 0` means the switch is open and growth
                # really is ungated. So the rebuild reads the parameter instead of guessing from
                # the wiring it does not have -- and takes the last producer ahead of grow_3d, the
                # same spec-order rule the ambiguous case above uses.
                _grow = next((o for o in g.ops if o["op"] == "grow_3d"), None)
                if _grow is not None and float(g.params.get(f"{_grow['id']}.a_sw", 0) or 0) > 0:
                    if not any(c["dst"] == _grow["id"] and c["slot"] == "gate" for c in g.conns):
                        _pos = {o["id"]: k for k, o in enumerate(g.ops)}
                        _src = [a for a, b, sl in g._candidate_links()
                                if b == _grow["id"] and sl == "gate" and _pos[a] < _pos[_grow["id"]]]
                        if _src:
                            g, _ = g.apply(("connect", max(_src, key=lambda a: _pos[a]),
                                            _grow["id"], "gate"))

                # THE SPEC'S NUMBERS MAY BE OUTSIDE THE DECLARED SPACE. cfl_c000p080_d002p000 ran
                # with cell_diffuse0.d_h=2.0 against a Critic ceiling of 0.346 and produced valid
                # evidence -- so a faithful rebuild is refused as a parent by R5, and the whole
                # frontier with it. Which of the two is wrong (the range or the run) is a real
                # question and not this function's to settle.
                #
                # What it CAN do is be explicit: keep every parameter the declared space allows,
                # drop the ones it does not, and name them. The parent is then the measured run as
                # closely as the space permits, and the discrepancy is on the record rather than
                # silently deciding whether the campaign has a frontier at all.
                import re as _re
                # DROP UNTIL IT STOPS COMPLAINING, because the Critic reports ONE offender at a
                # time: reconnect's engine_clock, then divide's, then the next. Two passes was an
                # arbitrary number and it left every rebuilt run refused for the third.
                dropped = []
                for _ in range(12):
                    adm, rej = C.admit(g, ())
                    if adm:
                        break
                    fresh = [m.group(1) for m in
                             _re.finditer(r"R5_PARAM_OUT_OF_RANGE:\s*([\w.]+)=", str(rej))]
                    for m in _re.finditer(r"(\w+):\w+ was given ([^.]+?), and its emitter", str(rej)):
                        oid_ = next((o["id"] for o in ops if o["op"] == m.group(1)), None)
                        if oid_:
                            fresh += [f"{oid_}.{w.strip()}" for w in
                                      m.group(2).replace(" and ", ", ").split(",") if w.strip()]
                    if not fresh:
                        break
                    dropped += fresh
                    params = {k: v for k, v in params.items() if k not in dropped}
                    # KEEP THE WIRING. Rebuilding with conns=[] here silently threw away the
                    # links inferred above, so every graph arrived at the Critic dangling again
                    # and the auto-wiring looked like it had never run.
                    g = _CS.CompositionGraph(ops=ops, conns=list(g.conns), params=params)
                # SAY WHERE IT STANDS. The rebuild recovers the operators; whether the result is
                # ADMISSIBLE is the Critic's to say, and on the current log it usually is not --
                # the measured runs sit outside the declared parameter ranges in several places.
                # That is a real finding about the search space, not a defect in this function:
                # `from_preset` warns of exactly it -- "if a recipe we already trust cannot be
                # built from legal one-edit moves, the campaign is searching a space that does not
                # contain our own evidence." It is reported here, once per run, and the caller
                # falls back rather than pretending.
                # ONE SHORT LINE. This used to spend three wrapped lines per parent saying the
                # rebuild worked -- "rebuilt from its spec (10 ops, 1 instrument op(s) left out, 1
                # clock-coupled param(s) reset to the re-anchored defaults); dropped 2 param(s) the
                # space disallows" -- and with one line per parent that is still nine lines of screen
                # for no decision. The counts are what a reader needs; the prose was for whoever
                # wrote it. A REFUSAL still gets its full reason, because that one is actionable.
                bits = [f"{len(ops)} ops"]
                if skipped:
                    bits.append(f"-{len(skipped)} instrument")
                if clock_dropped:
                    bits.append(f"{len(clock_dropped)} clock re-anchored")
                if dropped:
                    bits.append(f"-{len(dropped)} out-of-space")
                _adm, _why = C.admit(g, ())
                # THE TAG WAS `[archivist]`, A ROLE THAT NO LONGER EXISTS. Phase 12 deleted it, and a
                # line labelled with a dead role is worse than an unlabelled one: it tells the reader
                # someone is speaking who is not there.
                print(f"[build] {name}: {', '.join(bits)}"
                      + ("" if _adm else f"  STILL REFUSED: {_why}"))
                return g
        except Exception as e:
            print(f"[build] {name}: spec rebuild FAILED ({type(e).__name__}: {e})")
    print(f"[build] {name}: no composition.json and no usable spec -- cannot be a parent")
    return None


# THE PUBLIC NAME HAS NO UNDERSCORE. It was `_graph_from_run` in round.py because only that file used
# it; it is now the interface between the engine and every run on disk.
graph_from_run = _graph_from_run
