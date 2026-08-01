#!/usr/bin/env python
"""archivist -- the role that reads the WHOLE history and decides where to search from.

THE HOLE IT FILLS. Every other role sees one batch. Nothing in the roster ever looked across
rounds, and two failures follow directly from that:

  THE SEARCH COULD DRIFT with nothing able to say so. A line of descent that stopped paying was
  indistinguishable, from inside a round, from one that was about to.

  THE PROPOSER COUNTED WHAT IT PROPOSED. Its own words, round 4: "those two batches proposed ALL
  FOUR valid single-op edits on parent 2, so parent 2 is fully PROPOSED and cannot be built on
  further." A family is explored when its edits produced EVIDENCE, never when they were merely
  posed -- and only the history knows which. Nobody was holding the history.

WHAT IT DECIDES -- three answers, and it must give exactly one:

    continue                  the current line is still paying
    roll_back <comp_hash>     a better branch exists behind us; restart the frontier there
    stop                      no branch left is worth compute

IT ADVISES; THE SUPERVISOR DECIDES. Two controllers is the failure this rebuild removed, so this
returns a recommendation with its evidence, and the Supervisor is free to override it -- in which
case the override is recorded, because a recommendation nobody can refuse is a command.

WHEN IT RUNS. Between rounds, and ON EVERY ABORT. The abort case is why it exists at all: a round
that produced no evidence routes back to Act 1, and re-proposing inside the envelope the Critic
just refused is how "route back to Act 1" becomes a week-long loop. The Archivist is what makes
that path actionable -- it can move the frontier instead of retrying it.

WHAT IS ARITHMETIC AND WHAT IS JUDGEMENT. The history is assembled by code (below): every branch,
what it measured, how many of its runs were admissible, whether its specimens were sound. Only the
CHOICE is put to a model, and it is given the table rather than the raw logs -- an agent that has
to re-derive the arithmetic will re-derive it differently every time.

    python archivist.py            # the branch table, as code sees it
    python archivist.py --decide   # ... and ask for the decision
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

CAMP = os.path.join(HERE, "campaign")
RECORDS = os.path.join(CAMP, "round_records.jsonl")
DECISIONS = os.path.join(CAMP, "archivist.jsonl")

DECISIONS_ALLOWED = ("continue", "roll_back", "stop")


def history(records=RECORDS):
    """Every round ever recorded, oldest first. The Collector's output is the only source."""
    if not os.path.exists(records):
        return []
    out = []
    for line in open(records):
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def branches(rounds=None):
    """Per composition: what it measured, and whether its evidence was worth anything.

    ARITHMETIC ONLY. `n_evidence` counts runs that produced an admissible number -- not runs
    posed, not runs submitted. That distinction is the one the Proposer got wrong, so it is
    computed here and never asked of a model.
    """
    rounds = history() if rounds is None else rounds
    b = {}
    for r in rounds:
        for run in r.get("runs", []):
            h = run.get("comp_hash") or "?"
            e = b.setdefault(h, {"comp": h, "region": run.get("region", "?"), "rounds": set(),
                                 "n_evidence": 0, "n_sound": 0, "best": None, "best_run": None,
                                 "phenotypes": set(), "last_round": 0})
            e["rounds"].add(r["round"])
            e["last_round"] = max(e["last_round"], r["round"])
            e["n_evidence"] += 1
            if run.get("specimen") == "valid":
                e["n_sound"] += 1
            ph = run.get("analyst_consensus")
            if ph and ph != "MISSING":
                e["phenotypes"].add(ph)
            v = (run.get("metrics") or {}).get("protr_peak")
            if isinstance(v, (int, float)) and (e["best"] is None or v > e["best"]):
                e["best"], e["best_run"] = v, run.get("run")
    for e in b.values():
        e["rounds"] = sorted(e["rounds"])
        e["phenotypes"] = sorted(e["phenotypes"])
    return b


def drift(rounds=None, window=3):
    """Has the last `window` rounds improved on the best seen before them? Arithmetic, not opinion."""
    rounds = history() if rounds is None else rounds
    live = [r for r in rounds if not r.get("aborted")]
    if len(live) < window + 1:
        return {"enough_history": False, "n_rounds": len(live)}

    def best_of(rs):
        vals = [(run.get("metrics") or {}).get("protr_peak")
                for r in rs for run in r.get("runs", [])]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return max(vals) if vals else None

    before, recent = best_of(live[:-window]), best_of(live[-window:])
    return {"enough_history": True, "n_rounds": len(live), "window": window,
            "best_before": before, "best_recent": recent,
            "improved": (before is None or (recent is not None and recent > before)),
            "aborts_in_a_row": _aborts_in_a_row(rounds)}


def _aborts_in_a_row(rounds):
    n = 0
    for r in reversed(rounds):
        if r.get("aborted"):
            n += 1
        else:
            break
    return n


def table(rounds=None):
    """The history, as the model will be shown it. One line per branch, newest activity first."""
    b = branches(rounds)
    d = drift(rounds)
    if not b:
        return "no history yet -- this is the first round."
    L = [f"{len(b)} branch(es) across {d.get('n_rounds', 0)} completed round(s).", "",
         f"{'composition':12}{'region':16}{'rounds':10}{'evidence':>9}{'sound':>7}"
         f"{'best':>8}  phenotypes"]
    for e in sorted(b.values(), key=lambda e: (-e["last_round"], -(e["best"] or -1))):
        L.append(f"{e['comp'][:11]:12}{e['region'][:15]:16}"
                 f"{str(e['rounds'])[:9]:10}{e['n_evidence']:>9}{e['n_sound']:>7}"
                 f"{_fmt(e['best']):>8}  {', '.join(e['phenotypes']) or '—'}")
    if d.get("enough_history"):
        L += ["", f"Best protr_peak before the last {d['window']} rounds: "
                  f"{_fmt(d['best_before'])}; within them: {_fmt(d['best_recent'])} "
                  f"-> {'IMPROVING' if d['improved'] else 'NOT IMPROVING'}"]
    if d.get("aborts_in_a_row"):
        L.append(f"Aborted rounds in a row: {d['aborts_in_a_row']}")
    return "\n".join(L)


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def decide(reason="between rounds", ledger=None, timeout_min=6, rounds=None):
    """Ask for the decision, having done all the arithmetic first. Returns a dict, always.

    The fallback is `continue`, not `stop`: an Archivist that cannot be reached must not be able
    to end a campaign, and must not silently move the frontier either.
    """
    rounds = history() if rounds is None else rounds
    tab = table(rounds)
    b = branches(rounds)
    d = drift(rounds)

    if not b:
        return _record({"decision": "continue", "why": "no history yet", "target": None,
                        "reason": reason, "asked": False})

    import llm_agents  # noqa: F401  -- ensures the agents package path is set up
    from llm import run_agent, budget_note

    prompt = f"""ARCHIVIST. You read the WHOLE history of this campaign, not one batch.
{budget_note(timeout_min, "1) the JSON decision")}
You are being called because: {reason}

Everything below is MEASURED. Do not re-derive it; use it.
{tab}

A branch is EXPLORED when its edits produced evidence -- never when they were merely proposed.
`evidence` is the count of runs that returned an admissible number; `sound` is how many of those
were on a specimen the Biologist passed. A branch with evidence 4 / sound 0 has been measured
four times and has told us nothing about tissue.

Decide ONE of:
  continue   the current line is still paying
  roll_back  a better branch exists behind us -- give its composition hash as `target`
  stop       no branch left is worth compute

Reply with ONLY:
{{"decision": "continue|roll_back|stop",
  "target": "<comp hash, or null>",
  "why": "<<=40 words, citing the numbers above>",
  "confidence": 0.0-1.0}}"""
    ok, out = run_agent("archivist", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read"], quiet=True)
    dec = _first_json(out) or {}
    if dec.get("decision") not in DECISIONS_ALLOWED:
        dec = {"decision": "continue", "why": f"archivist unreachable or unparsable ({ok})",
               "target": None, "confidence": 0.0}
    if dec["decision"] == "roll_back" and dec.get("target") not in b:
        dec = {"decision": "continue", "target": None, "confidence": 0.0,
               "why": f"refused a roll_back to {dec.get('target')!r}: not a branch in the history"}
    dec.update({"reason": reason, "asked": True, "n_branches": len(b),
                "aborts_in_a_row": d.get("aborts_in_a_row", 0)})
    return _record(dec)


def _first_json(text):
    import re
    for m in re.finditer(r"\{.*?\}", text or "", re.S):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def _record(dec):
    os.makedirs(CAMP, exist_ok=True)
    with open(DECISIONS, "a") as fh:
        fh.write(json.dumps(dec) + "\n")
    return dec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", action="store_true")
    a = ap.parse_args()
    print(table())
    if a.decide:
        print("\n" + json.dumps(decide(), indent=1))


# ============================================================ COLD START -- once, at the beginning
# The campaign has always begun from a seed sphere plus the hand-written reference recipes, and
# never from THE BEST THING IT EVER ACTUALLY RAN. Sixty-odd runs sit in log/okuda with their specs
# and their measured results, and until now nothing read them at the start of a campaign: every
# new campaign threw that away and re-derived it, badly. That is the "something we messed up
# prior" -- not a bug in a round, a bug in where rounds begin.
#
# This reads the LOG, not the campaign state: spec_run.yaml for the composition, diag.json for
# what it measured and whether the Biologist passed it. It runs ONCE, when there is no frontier.

def survey_log(log_dir=None, min_protr=None):
    """Every finished run on disk: what it was, what it measured, whether it was a tissue.

    ARITHMETIC ONLY. A run counts as a candidate starting point when it has a composition, an
    admitted number, and a specimen the Biologist did not call `invalid` -- three facts, all
    already recorded. Nothing here is a judgement.
    """
    import yaml
    log_dir = log_dir or os.path.join(os.path.dirname(HERE), "log", "okuda")
    out = []
    for name in sorted(os.listdir(log_dir)):
        d = os.path.join(log_dir, name)
        dj, sy = os.path.join(d, "diag.json"), os.path.join(d, "spec_run.yaml")
        if not (os.path.isfile(dj) and os.path.isfile(sy)):
            continue
        try:
            j = json.load(open(dj))
        except Exception:
            continue
        s = j.get("summary") or {}
        peak = s.get("protr_peak")
        if not isinstance(peak, (int, float)):
            continue
        try:
            import biologist as B
            spec = B.specimen_verdict(j.get("premises") or [])
        except Exception:
            spec = "invalid" if j.get("premises_broken") else "unchecked"
        out.append({
            "run": name, "comp": j.get("comp_hash", "?"), "region": j.get("region", "?"),
            "protr_peak": round(float(peak), 3),
            "protr_final": _num(s.get("protr_final")),
            "morphology": s.get("morphology", "?"),
            "n_cells_final": _num(s.get("n_cells_final")),
            "specimen": spec,
            "broken": j.get("premises_broken") or [],
            "horizon": s.get("horizon_frame"),
            "spec_path": sy,
            **_mechanics(d, s),
        })
    out.sort(key=lambda r: -r["protr_peak"])
    if min_protr is not None:
        out = [r for r in out if r["protr_peak"] >= min_protr]
    return out


def _num(v):
    return round(float(v), 3) if isinstance(v, (int, float)) else None


def _mechanics(run_dir, summary):
    """Read mechanics.npz, not just the endpoint the summary kept.

    WHY THE NPZ AND NOT THE SCALAR. `mech_p_ratio` in the summary is one number at one moment,
    and a starting point chosen on it can be a run that was never mechanically settled -- the
    force was still falling when the frames ran out, so its shape is a transient and breeding
    from it breeds from a snapshot of something in motion. The SERIES says which: a residual
    force still descending at the last frame is a run that had not finished relaxing, whatever
    its final elongation was.

        p_ratio      ~3 = FORCED protrusion (pushed), ~1 = growth-driven equilibrium (grown)
        relaxed      did the residual force settle, or was it still falling at the end?
    """
    out = {"p_ratio": _num(summary.get("mech_p_ratio")), "relaxed": None,
           "f_end": None, "n_protruding_max": None}
    p = os.path.join(run_dir, "mechanics.npz")
    if not os.path.exists(p):
        return out
    try:
        import numpy as np
        z = np.load(p, allow_pickle=True)
        if "force_mean" not in z.files:
            return out
        f = np.asarray(z["force_mean"], dtype=float).ravel()
        f = f[np.isfinite(f)]
        if f.size < 6:
            return out
        k = max(2, f.size // 5)
        tail, mid = f[-k:], f[f.size // 2 - k // 2: f.size // 2 + k // 2 + 1]
        out["f_end"] = round(float(tail.mean()), 3)
        # SETTLED means the residual force stopped moving, not that it is small: a run whose
        # force is still falling at the last frame has not finished, whatever its elongation was.
        span = float(np.nanmax(f) - np.nanmin(f)) or 1.0
        out["relaxed"] = bool(abs(float(tail.mean()) - float(mid.mean())) / span < 0.10)
        # protruding cells, if the run ever had any -- a tube that never formed is worth knowing
        if "n_protruding" in z.files:
            npr = np.asarray(z["n_protruding"], dtype=float).ravel()
            out["n_protruding_max"] = int(np.nanmax(npr)) if npr.size else 0
    except Exception:
        pass
    return out


def log_table(rows=None, top=18):
    rows = survey_log() if rows is None else rows
    if not rows:
        return "log/okuda holds no finished run with a composition and an admitted number."
    L = [f"{len(rows)} finished run(s) on disk with a composition and a measured protr_peak.",
         "Sorted by protr_peak. `specimen` is the Biologist's verdict on the run that produced it.",
         "",
         f"{'run':26}{'comp':10}{'protr_peak':>11}{'cells':>7}{'f_end':>8}{'relaxed':>9}"
         f"{'protrud':>8}  specimen"]
    for r in rows[:top]:
        L.append(f"{r['run'][:25]:26}{str(r['comp'])[:9]:10}"
                 f"{r['protr_peak']:>11.2f}{(r['n_cells_final'] or 0):>7.0f}"
                 f"{_fmt(r.get('f_end')):>8}"
                 f"{('yes' if r.get('relaxed') else ('no' if r.get('relaxed') is not None else '—')):>9}"
                 f"{(r.get('n_protruding_max') if r.get('n_protruding_max') is not None else '—'):>8}"
                 f"  {r['specimen']}"
                 + (f"  [{', '.join(r['broken'][:3])}]" if r["broken"] else ""))
    sound = [r for r in rows if r["specimen"] in ("valid", "valid (declared)")]
    L += ["", f"{len(sound)} of {len(rows)} were run on a specimen the Biologist passed.",
          "From mechanics.npz: `f_end` is the mean residual force over the last fifth of the "
          "run, `relaxed` says whether it had STOPPED MOVING by then (a run still relaxing is a "
          "snapshot of something in motion -- breeding from it breeds from a transient), and "
          "`protrud` is the most cells that were ever protruding. A run with protrud 0 never "
          "made a protrusion at all, whatever its elongation number says."]
    if sound:
        L.append(f"Best SOUND run: {sound[0]['run']} at protr_peak {sound[0]['protr_peak']:.2f} "
                 f"({sound[0]['morphology']}).")
    return "\n".join(L)


def cold_start(ledger=None, timeout_min=6, log_dir=None):
    """Choose where a NEW campaign begins, from what is already on disk. Runs once.

    Returns {"start": [run names], "why": ...}. The caller turns those into the frontier.

    The default when this cannot be answered is the EMPTY list, not a guess: an unreachable
    Archivist must not be able to silently pick a starting point, because a bad start is the one
    error a campaign cannot measure its way out of -- every subsequent diff is taken against it.
    """
    rows = survey_log(log_dir)
    if not rows:
        return _record({"stage": "cold_start", "start": [], "why": "no finished runs on disk",
                        "asked": False})
    sound = [r for r in rows if r["specimen"] in ("valid", "valid (declared)")]

    # THE QUESTION THAT MUST BE ASKED BEFORE THE MODEL IS. Measured 2026-08-01 over all 62 runs
    # on disk: NOT ONE ever had a protruding cell, and not one exceeded protr_peak 1.20. The
    # scalar summary said "best sound run 1.11", which reads like a result and is a sphere with
    # noise on it -- `n_protruding` in mechanics.npz is what says so, and nothing had read it.
    #
    # So there is a real possibility that the honest answer is THERE IS NOTHING HERE TO BREED
    # FROM, and an Archivist asked to pick the best of sixty failures will pick one. Arithmetic
    # decides this, not the model: if nothing ever protruded, no run on disk is a starting point
    # for a campaign whose objective is a tube.
    protruded = [r for r in rows if (r.get("n_protruding_max") or 0) > 0]
    if not protruded:
        return _record({
            "stage": "cold_start", "start": [], "asked": False,
            "n_on_disk": len(rows), "n_sound": len(sound), "n_protruded": 0,
            "why": f"NOTHING ON DISK EVER PROTRUDED. Across all {len(rows)} finished runs, "
                   f"n_protruding never exceeded 0 and protr_peak never exceeded "
                   f"{max(r['protr_peak'] for r in rows):.2f}. Every one of them is a sphere. "
                   f"There is no measured starting point for a campaign whose objective is a "
                   f"tube, so the frontier falls back to the reference recipes -- and that is a "
                   f"CHOICE recorded here, not a default nobody noticed."})
    tab = log_table(rows)

    from llm import run_agent, budget_note
    prompt = f"""ARCHIVIST, COLD START. A new campaign is beginning. Choose where it starts.

{budget_note(timeout_min, "1) the JSON choice")}
The campaign has always begun from a seed sphere and hand-written reference recipes, throwing away
everything already measured. Below is what is ACTUALLY ON DISK. All of it is measured; do not
re-derive it.

{tab}

The objective is Okuda's coupling experiment: a hollow ball of ~2000 cells whose surface chemistry
drives tubulation, branching and undulation. `protr_peak` is elongation -- higher means more
protrusion -- but a high number on an `invalid` specimen measured a configuration error, not a
tissue, and is worth less than a lower number on a sound one.

Choose 1-3 runs whose compositions the campaign should breed from. Prefer SOUND specimens. Say
what each one gives you that the others do not -- three variations on one composition is one
starting point, not three.

Reply with ONLY:
{{"start": ["<run name>", ...],
  "why": "<<=50 words, citing the numbers above>",
  "avoid": "<any composition family the table says is a dead end, or null>"}}"""
    ok, out = run_agent("archivist", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read"], quiet=True)
    dec = _first_json(out) or {}
    known = {r["run"] for r in rows}
    start = [r for r in (dec.get("start") or []) if r in known]
    if not start:
        # Not a guess: the best SOUND run, named as a fallback so the record says it was a
        # fallback. A silent default here would be indistinguishable from a decision.
        start = [sound[0]["run"]] if sound else []
        dec["why"] = (f"archivist unusable ({ok}); fell back to the best sound run on disk"
                      if start else "no sound run on disk to fall back to")
    return _record({"stage": "cold_start", "start": start, "why": dec.get("why", ""),
                    "avoid": dec.get("avoid"), "asked": True, "n_on_disk": len(rows),
                    "n_sound": len(sound)})
