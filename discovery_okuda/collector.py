#!/usr/bin/env python
"""collector -- the round record, built from the files on disk. NOT an agent.

WHY THIS IS CODE. The campaign's defining failure was not that an agent judged badly; it was that
correct judgements never reached anybody. The Biologist broke five premises on `r002c_00` -- the
activator had decayed to NaN, the chemistry was extinct -- and wrote them to a terminal and to a
field of `diag.json` that no prompt mentioned. An Analyst then read that run's numbers and named a
phenotype. The Critic's refusals went nowhere. The Supervisor's steer went to a terminal.

Every one of those is a collection failure, and collection is a `for` loop. Making it a role that
*reasons* means it can forget, and its forgetting is silent -- there is no error, just an absent
paragraph. Built from disk instead, a missing input is a HOLE with a name: the record says
`analyst: MISSING` and the round says so out loud.

WHAT IT REFUSES TO DO. It does not summarise, weigh, or explain. Every field is either

    MEASURED    read from diag.json / metrics / the premise verdict, or
    QUOTED      the exact words an agent said at the time, attributed

and nothing else. The moment a collector starts writing prose it has become an unaccountable
second opinion, and the record stops being checkable against the run.

WHO OWNS WHAT (ROLES.md):
    analysis.md   THIS MODULE. Append-only, one entry per round, fixed fields. It was written by
                  the PROPOSER -- the agent under evaluation writing its own record, which is how
                  "parent 2 is fully PROPOSED" became coverage: territory counted because it had
                  been proposed, never because anything was measured.
    memory.md     the Meta-review. A state document is a judgement about what matters later.

    python collector.py <round_id>        # rebuild a round's record from disk and print it
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
CAMP = os.path.join(HERE, "campaign")
ANALYSIS = os.path.join(CAMP, "analysis.md")
RECORDS = os.path.join(CAMP, "round_records.jsonl")

MISSING = "MISSING"          # never None, never "" -- an absent input must read as absent


def _read_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def collect_run(name, hyp=None, summary=None):
    """Everything every role said about ONE run. Absent inputs are named, not skipped."""
    d = os.path.join(LOG, name)
    diag = _read_json(os.path.join(d, "diag.json")) or {}
    s = summary if summary is not None else (diag.get("summary") or {})

    prem = diag.get("premises") or []
    broken = diag.get("premises_broken") or []
    rec = {
        "run": name,
        "exists": os.path.isdir(d),
        # --- MEASURED -------------------------------------------------------------------
        "comp_hash": diag.get("comp_hash", MISSING),
        "region": diag.get("region", MISSING),
        "metrics": {k: s.get(k) for k in ("protr_peak", "protr_final", "ta_n_tubes_final",
                                          "mech_p_ratio") if k in s},
        # --- the BIOLOGIST, categorical --------------------------------------------------
        "specimen": _specimen(prem, broken),
        "premises_broken": broken,
        # --- the ANALYSTS: the label, and how much they agreed on it ---------------------
        "analyst_consensus": s.get("analyst_consensus", MISSING),
        "analyst_agreement": s.get("analyst_agreement", MISSING),
        "analyst_specimen": s.get("analyst_specimen", MISSING),
        # --- the EYE-CHECK: an observation, never a score --------------------------------
        "eye_verdict": s.get("watcher_verdict", MISSING),
        "eye_disagrees": bool(s.get("watcher_blocks")),
        "eye_why": (s.get("watcher_why") or "")[:200] or MISSING,
        # --- the CRITIC ------------------------------------------------------------------
        "acted": diag.get("acted", MISSING),
        # --- the ENGINE, which is an instrument and not an agent --------------------------
        # Its output is measurement, so it is collected exactly as the Biologist's verdicts and
        # the Critic's refusals are. It reached nobody before this: divide_3d counted the
        # divisions it refused and flagged a full array, run_one recorded them, and the round
        # record -- the one thing every downstream role reads -- did not carry them. A run that
        # stopped at 98.5% of its buffer looked, to every agent, exactly like a run that stopped.
        "reservoir": {
            "cells_final": s.get("n_cells_final"),
            "buf_full": bool(s.get("buf_full")),
            "div_blocked": s.get("div_blocked") or 0,
            "first_refused_frame": s.get("div_blocked_first_frame"),
            "cap_cells": _cap_of(name),
            **_resize_of(name),
        },
    }
    if hyp is not None:
        rec.update({                      # QUOTED: what was believed BEFORE the run
            "hid": getattr(hyp, "hid", MISSING),
            "edit": getattr(hyp, "edit", MISSING),
            "intent": getattr(hyp, "intent", MISSING),
            "track": getattr(hyp, "track", MISSING),
            "predicted": getattr(hyp, "predicted", MISSING),
            "rationale": (getattr(hyp, "rationale", "") or "")[:300] or MISSING,
        })
    return rec


def _resize_of(name):
    """What the launcher DID to this run's array before it ran, from campaign/reservoir.jsonl.

    Distinct from what the engine observed: this is the enlargement decision. It lived as a local
    variable and a printed warning, so a composition on its third enlargement looked identical to
    one on its first.
    """
    p = os.path.join(CAMP, "reservoir.jsonl")
    if not os.path.exists(p):
        return {}
    hit = None
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("run") == name or r.get("slot") == name:
            hit = r
    if not hit:
        return {}
    return {"resized_from": hit.get("from"), "clamped_from_cells": hit.get("clamped_from_cells"),
            "times_censored": hit.get("times_censored") or 0}


def _reservoir_line(rv):
    """What the ENGINE said about the array, in the block every downstream role reads.

    A run stopped by its buffer is a CENSORED measurement -- a lower bound, not a destination --
    and the difference is invisible in `n_cells_final` alone. Written as a line rather than a
    field because the Interpreter, Meta-review, Supervisor and Archivist all read the rendered
    record, and a number nobody can see is a number nobody uses.
    """
    rv = rv or {}
    cap, fin = rv.get("cap_cells"), rv.get("cells_final")
    if not cap or fin is None:
        return "Reservoir: " + MISSING
    frac = fin / cap
    if rv.get("buf_full") or rv.get("div_blocked") or frac >= 0.97:
        where = (f", first refused division at frame {rv['first_refused_frame']}"
                 if rv.get("first_refused_frame") is not None else "")
        again = ""
        if rv.get("times_censored"):
            again = (f" This composition has been censored {rv['times_censored']}x before and its "
                     f"array was already enlarged")
            if rv.get("clamped_from_cells"):
                again += (f", then CLAMPED by the memory budget — no buffer will fix it, so its "
                          f"growth is unbounded and the composition is what must change")
            again += "."
        return (f"Reservoir: **CAPPED** — {fin} of {cap} cells ({frac:.0%}), "
                f"{rv.get('div_blocked') or 0} divisions refused{where}. This growth is a LOWER "
                f"BOUND: the array stopped it, not the biology. Do not read the final cell count "
                f"or any metric that depends on it as the composition's outcome." + again)
    return f"Reservoir: {fin} of {cap} cells ({frac:.0%}) — not limiting."


def _cap_of(name):
    """How many cells this run's array could hold, from its own spec. Euler: V = 2F - 4."""
    try:
        import yaml
        c = yaml.safe_load(open(os.path.join(LOG, name, "spec_run.yaml")))
        v = ((c.get("sets") or {}).get("vertex") or {}).get("n")
        return (int(v) + 4) // 2 if v else None
    except Exception:
        return None


def _specimen(premises, broken):
    """The Biologist's verdict, categorical. The grade in PREMISES.md decides it, not this file."""
    if not premises:
        return "unchecked"
    try:
        import biologist as B
        return B.specimen_verdict(premises)
    except Exception:
        return "invalid" if broken else "valid"


def collect_round(rid, mode, rows, refused=(), posed=(), steer=None, aborted=False):
    """The whole round. `rows` are the runs that produced evidence; `refused` those that did not.

    Nothing here is optional. A round that ran eight simulations and admitted one records the
    seven refusals WITH THEIR REASONS, because "0 runs, coverage 0%" was what the Proposer was
    handed, and from an insane input it drew the only sane conclusion available: that the ledger
    was broken.
    """
    runs = []
    for nm, _g, s, _sc, oc, h in rows:
        r = collect_run(nm, hyp=h, summary=s)
        # THE HYPOTHESIS OUTCOME, scored by predict.py -- arithmetic, not an agent's opinion.
        # `inconclusive` is a THIRD outcome and not a soft refutation: it drops out of the
        # surprise denominator, because a prediction that could not fail must not dilute the
        # rate that steers the next batch.
        r["outcome"] = oc
        r["surprise"] = bool(getattr(h, "is_surprise", False))
        runs.append(r)
    rec = {
        "round": rid,
        "mode": mode,
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "aborted": bool(aborted),
        "n_posed": len(posed) or len(rows) + len(refused),
        "n_evidence": len(runs),
        "n_refused": len(refused),
        "refused": [dict(run=r[0], why=r[1]) for r in refused],
        "runs": runs,
        "steer": steer or MISSING,
        # the two things a later reader most needs and can least re-derive
        "specimens": {v: sum(1 for r in runs if r["specimen"] == v)
                      for v in {r["specimen"] for r in runs}},
        # THE SURPRISE RATE, over predictions that could actually be checked. A round of eight
        # confirmations has bought coverage and no knowledge, and the entry must say so.
        "outcomes": {v: sum(1 for r in runs if r.get("outcome") == v)
                     for v in {r.get("outcome") for r in runs} if v},
        "n_checkable": sum(1 for r in runs if r.get("outcome") in ("confirmed", "refuted")),
        "n_surprise": sum(1 for r in runs if r.get("surprise")),
        "eye_disagreements": [r["run"] for r in runs if r["eye_disagrees"]],
        # Runs the ARRAY stopped, not the biology. A reader of the record should not have to
        # infer this from a cell count that happens to sit near a buffer size.
        "reservoir_capped": [r["run"] for r in runs
                             if (r.get("reservoir") or {}).get("buf_full")
                             or ((r.get("reservoir") or {}).get("cells_final") or 0) >=
                             0.97 * ((r.get("reservoir") or {}).get("cap_cells") or 1e18)],
    }
    return rec


def write(rec, analysis=ANALYSIS, records=RECORDS):
    """Append the machine record and the human entry. Both append-only, both from `rec`."""
    os.makedirs(os.path.dirname(records), exist_ok=True)
    with open(records, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    entry = render(rec)
    with open(analysis, "a") as fh:
        fh.write("\n" + entry)
    return entry


def render(rec):
    """The analysis.md entry, on TEMPLATE_analysis.md's fields. Measured or quoted, never prose.

    One block per slot plus a round summary, following the connectome-gnn-cx exploration log:
    Node/parent carries the ANCESTRY, Mutation the DIFF, `Hypothesis tested` is quoted verbatim
    as it was posed, and the Verdict must cite the number that settled it.
    """
    R, runs = rec, rec["runs"]
    L = []
    for i, r in enumerate(runs):
        L += [f"## Round {R['round']} — slot {i}: {r.get('outcome', 'inconclusive')}", "",
              f"Node: id={r.get('comp_hash', MISSING)}, parent={r.get('parent', 'none')}",
              f"Track: {r.get('track', '-')}",
              f'Hypothesis tested: "{r.get("predicted", MISSING)}"',
              f"Config: {r['run']}"
              + (f", {', '.join(f'{k}={v}' for k, v in (r.get('config') or {}).items())}"
                 if r.get("config") else ""),
              f"Measured: " + (", ".join(f"{k}={v:.4g}" for k, v in (r["metrics"] or {}).items()
                                         if isinstance(v, (int, float))) or "no admitted metric"),
              _reservoir_line(r.get("reservoir")),
              f"Specimen: {r['specimen']}" + (f" — {', '.join(r['premises_broken'])}"
                                              if r["premises_broken"] else " — all hold"),
              f"Reader: phenotype={r['analyst_consensus']}, "
              f"specimen={r.get('analyst_specimen', MISSING)}",
              f"Eye-check: {'DISAGREES — ' + r['eye_why'] if r['eye_disagrees'] else r['eye_verdict']}",
              f"Mutation: {_cell(r.get('edit')) if r.get('intent') != 'control' else 'none (control)'}",
              f"Verdict: {_verdict(r)}",
              f"Next: parent={r.get('comp_hash', MISSING)}", ""]

    spec = R["specimens"]
    nc, ns = R.get("n_checkable", 0), R.get("n_surprise", 0)
    L += [f"## Round {R['round']} — summary", "",
          f"Posed: {R['n_posed']}   Evidence: {R['n_evidence']}   Refused: {R['n_refused']}"
          + (f" ({'; '.join(x['why'][:40] for x in R['refused'])})" if R["refused"] else ""),
          f"Surprise: {ns}/{nc}" + (" — a round that only confirms has bought coverage and no "
                                    "knowledge" if nc and not ns else ""),
          f"Tracks: {sum(1 for r in runs if r.get('track') == 'A')} Track A, "
          f"{sum(1 for r in runs if r.get('track') == 'B')} Track B",
          f"Specimens: {', '.join(f'{v} {n}' for v, n in spec.items()) or 'none'}",
          f"Frontier after: {', '.join(sorted({r.get('comp_hash', '?') for r in runs}))}",
          f"Diagnosis: " + (f"{R['diagnosis'].get('cause')} — guard: "
                            f"{R['diagnosis'].get('guard_to_add')}"
                            if R.get("diagnosis") else "not called"),
          f"Steer: {R['steer']}", ""]
    if R["aborted"]:
        L.insert(0, f"## Round {R['round']} — ABORTED, no admissible evidence\n")
    return "\n".join(L)


def _verdict(r):
    """supported | falsified | partial | inconclusive, WITH the number that settled it."""
    oc = r.get("outcome") or "inconclusive"
    word = {"confirmed": "supported", "refuted": "falsified"}.get(oc, oc)
    m = r["metrics"] or {}
    num = ", ".join(f"{k}={v:.3g}" for k, v in m.items() if isinstance(v, (int, float)))
    if r["specimen"] in ("invalid", "ambiguous"):
        return (f"inconclusive — specimen {r['specimen']}, so {num or 'the numbers'} describe the "
                f"configuration and not a tissue")
    if oc == "inconclusive":
        return (f"inconclusive — the prediction could not be checked; it leaves the surprise "
                f"denominator and buys nothing. Measured {num or 'nothing admitted'}")
    return f"{word} — measured {num or 'nothing admitted'} against \"{r.get('predicted', '?')}\""


def _headline(runs):
    if not runs:
        return "NO RUNS"
    best = max(runs, key=lambda r: (r["metrics"] or {}).get("protr_peak") or -1)
    ph = {r["analyst_consensus"] for r in runs} - {MISSING}
    return (f"{len(runs)} RUNS, PHENOTYPES {'/'.join(sorted(ph)) or 'UNREAD'}, "
            f"BEST protr_peak {(best['metrics'] or {}).get('protr_peak', float('nan')):.2f}")


def _cell(v):
    return str(v).replace("|", "/")[:60] if v not in (None, "") else MISSING


def holes(rec):
    """What the record is MISSING. This is the point of building it from disk."""
    out = []
    for r in rec["runs"]:
        for k in ("analyst_consensus", "eye_verdict", "comp_hash"):
            if r.get(k) == MISSING:
                out.append(f"{r['run']}: no {k}")
        if r["specimen"] == "unchecked":
            out.append(f"{r['run']}: the Biologist never ran on it")
    if rec["steer"] == MISSING:
        out.append("no steer recorded for the next round")
    return out


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    found = [json.loads(l) for l in open(RECORDS)] if os.path.exists(RECORDS) else []
    if not found:
        raise SystemExit(f"no records in {RECORDS}")
    rec = ([r for r in found if r["round"] == rid] or found)[-1]
    print(render(rec))
    for h in holes(rec):
        print(f"  [hole] {h}")
