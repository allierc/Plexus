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
    """The analysis.md entry, on TEMPLATE_analysis.md's fields. Measured or quoted, never prose."""
    R, runs = rec, rec["runs"]
    head = (f"ABORTED — NO ADMISSIBLE EVIDENCE" if R["aborted"] else
            _headline(runs))
    L = [f"## Round {R['round']} ({R['mode']}, {R['when']}) — {head}", ""]

    parents = sorted({r.get("comp_hash", MISSING) for r in runs}) or [MISSING]
    intents = [r.get("intent", MISSING) for r in runs]
    n_adv = sum(1 for i in intents if i == "adversarial")
    L += [f"**Parent**: {', '.join(parents[:3])}",
          f"**Edits**: {R['n_posed']} posed, {R['n_evidence']} produced evidence, "
          f"{R['n_refused']} refused — {len(intents) - n_adv} confirmatory / {n_adv} adversarial",
          ""]

    L += ["| Slot | Run | Edit | Track | Intent | Prediction | Specimen | Phenotype | Outcome |",
          "| ---: | --- | ---- | ----- | ------ | ---------- | -------- | --------- | ------- |"]
    for i, r in enumerate(runs):
        L.append(f"| {i} | `{r['run']}` | {_cell(r.get('edit'))} | {r.get('track', '-')} "
                 f"| {r.get('intent', MISSING)} "
                 f"| {_cell(r.get('predicted'))} | **{r['specimen']}** "
                 f"| {r['analyst_consensus']} | {r.get('outcome', '—')} |")
    L.append("")

    # QUOTED, not composed: the Proposer's own stated reason, as written at propose time.
    why = [r.get("rationale") for r in runs if r.get("rationale") not in (None, MISSING)]
    L += [f"**Why these** (quoted from the Proposer): "
          f"{why[0][:220] if why else MISSING}", ""]

    L.append("**Result**:")
    for r in runs:
        m = ", ".join(f"{k} {v:.2f}" for k, v in (r["metrics"] or {}).items()
                      if isinstance(v, (int, float)))
        L.append(f"- `{r['run']}` {m or 'no admitted metric'} — specimen **{r['specimen']}**"
                 + (f", premises broken: {', '.join(r['premises_broken'])}"
                    if r["premises_broken"] else ""))
    L.append("")

    L += [f"**Refused**: " + ("; ".join(f"`{x['run']}` {x['why']}" for x in R["refused"])
                             if R["refused"] else "none"), ""]

    spec = R["specimens"]
    bad = spec.get("invalid", 0) + spec.get("ambiguous", 0)
    L += [f"**Verdict**: " + ("NO EVIDENCE — the round produced nothing admissible, which is a "
                              "result about the apparatus and not a failed round."
                              if R["aborted"] else
                              f"{R['n_evidence']} run(s) admitted; "
                              f"{bad} of them on a specimen that was not sound "
                              f"({', '.join(f'{v}: {n}' for v, n in spec.items())})"), ""]

    surprises = [r["run"] for r in runs if r.get("surprise")]
    nc, ns = R.get("n_checkable", 0), R.get("n_surprise", 0)
    rate = f"{ns}/{nc}" if nc else "0/0"
    incon = R.get("outcomes", {}).get("inconclusive", 0)
    L += [f"**Surprise**: {rate} of the checkable predictions were wrong"
          + (f" ({', '.join(surprises)})" if surprises else
             " — a round that only confirms has bought coverage and no knowledge")
          + (f". {incon} prediction(s) were INCONCLUSIVE and are excluded from that rate: "
             f"a prediction that could not fail must not dilute it." if incon else "."), ""]
    if R["eye_disagreements"]:
        L += [f"**Eye-check disagreed** (recorded, never scored): "
              f"{', '.join(R['eye_disagreements'])}", ""]
    L += [f"**Next**: {R['steer']}", ""]
    return "\n".join(L)


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
