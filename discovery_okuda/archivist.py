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
    stop                      no branch left is worth compute -- and this is a TRACK B verdict,
                            never a Track A one. Track A never runs out: the operator landscape
                            always has more of itself to understand, and a composition that
                            patterns without protruding is a point on the map, not a failure.
                            A search stops for Track A only when the space itself is exhausted,
                            which is escalation.py's judgement, not this one's.

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
    _w = max(12, max(len(e["comp"]) for e in b.values()) + 1)
    L = [f"{len(b)} branch(es) across {d.get('n_rounds', 0)} completed round(s).", "",
         f"{'composition':{_w}}{'region':16}{'rounds':10}{'evidence':>9}{'sound':>7}"
         f"{'best':>8}  phenotypes"]
    # THE IDENTIFIER IS NOT TRUNCATED. `comp[:11]` fitted the column and broke the table: hashes on
    # disk run 12 to 28 characters, and cutting them to eleven makes DISTINCT compositions
    # identical here -- measured, it collides on the current log. The Archivist reads this table
    # and names a branch back to the Supervisor, so a collision is not a cosmetic problem: it
    # rolls the campaign back to the wrong composition, or to one that does not exist. Same
    # disease as `run[:14]`, which cost six of twelve recon slots.
    #
    # The region is display and may be cut. The hash is identity and may not.
    for e in sorted(b.values(), key=lambda e: (-e["last_round"], -(e["best"] or -1))):
        L.append(f"{e['comp']:{_w}}{e['region'][:15]:16}"
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

`serves` says which track a branch can support. A = it runs, it patterns, the Biologist passes it;
that is a point on the operator map and it is worth having whether or not anything protruded.
B = it also protruded, so it can carry the reproduction. protr_peak and protrud are TRACK B
measures and say nothing about a Track A branch.

Decide ONE of:
  continue   the current line is still paying
  roll_back  a better branch exists behind us -- give its composition hash as `target`
  stop       no branch left is worth compute. THIS IS A TRACK B VERDICT ONLY. Track A never
             runs out -- there is always more of the operator landscape to understand -- so do
             not answer `stop` because the picture has not appeared.

Reply with ONLY:
{{"decision": "continue|roll_back|stop",
  "target": "<comp hash, or null>",
  "why": "<<=40 words, citing the numbers above>",
  "confidence": 0.0-1.0,
  "headline": "<at most 90 characters: the ONE thing a person watching the terminal should know>"}}"""
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
        _d = decide()
        print(f"\n[archivist] {_d.get('decision', '?')} -- {str(_d.get('why', ''))[:160]}")


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
            "capped": _capped(d),
            **_chemistry(d),
            **_mechanics(d, s),
        })
    # PATTERNED FIRST, then sound, then elongation. Ranking on elongation alone is what left the
    # cold start with nothing to say: every run on disk sits at protr_peak ~1.1, and the ones that
    # differ meaningfully differ in whether their chemistry is alive.
    out.sort(key=lambda r: (-(r.get("patterned") or 0.0), not r.get("chem_ok"),
                            r["specimen"] != "valid", -r["protr_peak"]))
    if min_protr is not None:
        out = [r for r in out if r["protr_peak"] >= min_protr]
    return out


def _num(v):
    return round(float(v), 3) if isinstance(v, (int, float)) else None


def _capped(run_dir):
    """Did this run hit its vertex reservoir? A capped run cannot be replayed usefully.

    The Archivist kept choosing the `wk_*` specs -- they pattern, they are sound, they rank well
    -- and every one was refused P2_BUFFER_SATURATED afterwards, because their buffers were sized
    for a 150-cell start and they grow to the 1778-cell ceiling. Ranking on pattern and soundness
    alone cannot see that; the run looks excellent right up to the moment it is thrown away.
    """
    import json as _j
    try:
        j = _j.load(open(os.path.join(run_dir, "diag.json")))
    except Exception:
        return None
    s = j.get("summary") or {}
    if s.get("buf_full"):
        return True
    n = s.get("n_cells_final")
    ser = None
    try:
        ser = _j.load(open(os.path.join(run_dir, "metrics.json"))).get("series")
    except Exception:
        pass
    if not (ser and isinstance(n, (int, float))):
        return None
    import numpy as np
    c = np.array([e.get("cells") if isinstance(e.get("cells"), (int, float)) else np.nan
                  for e in ser], float)
    if not np.isfinite(c).any():
        return None
    # a flat ceiling held for the last quarter of the run, after having grown
    i = int(np.nanargmax(c))
    # >= 0.15, not > 0.25. wk_pressure_pos_s0 plateaus for EXACTLY a quarter of its frames and
    # was missed by a hair -- 150 -> 1778 cells, flat for the last ten of forty, and reported as
    # a fine starting point. A threshold that a real case sits exactly on is a threshold chosen
    # without looking at the data.
    return bool(c[-1] > c[0] * 1.05 and (len(c) - i) / len(c) >= 0.15
                and np.allclose(c[i:], np.nanmax(c), atol=0.5))


def _chemistry(run_dir):
    """DID THE CHEMISTRY WORK? The question the cold start was not asking, and the only one that
    separates a usable starting point from a dead one.

    It ranked on protr_peak and n_protruding, which are zero for every run on disk, so it
    correctly reported that nothing had ever protruded and then had nothing left to rank by. It
    fell back to the reference recipes -- and okuda_route carries gierer_meinhardt, whose
    activator runs away uniformly to 1.4e6 and takes the run with it. Hours were spent there.

    A run with a LIVE TURING PATTERN and no protrusion is a far better place to start than a
    recipe whose chemistry explodes: the pattern is the hard part, and the protrusion is what the
    campaign is for.

        patterned   max spatial spread of the activator. A Turing pattern REQUIRES this > 0;
                    a uniform field is a well-mixed ODE and can never make a shape.
        chem_ok     did it stay finite for the whole run
    """
    out = {"patterned": None, "chem_ok": None, "act_peak": None}
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return out
    try:
        import numpy as np
        s = json.load(open(p)).get("series") or []
        if not s:
            return out
        g = lambda e, k: (e.get(k) if isinstance(e.get(k), (int, float)) else float("nan"))
        mx = np.array([g(e, "act_max") for e in s])
        mn = np.array([g(e, "act_min") for e in s])
        me = np.array([g(e, "act_mean") for e in s])
        fin = np.isfinite(mx) & np.isfinite(mn)
        out["chem_ok"] = bool(np.isfinite(me).all())
        # A SPREAD MEASURED ON A DIVERGING FIELD IS NOT A PATTERN. cfl_c000p050_d010p000 reports
        # 7.2e20 because it exploded, and sorting on that put the worst run on disk at the top of
        # the list of good starting points. The spread only counts where the chemistry stayed
        # finite for the whole run.
        out["patterned"] = (float(np.nanmax(mx[fin] - mn[fin]))
                            if (fin.any() and out["chem_ok"]) else 0.0)
        out["act_peak"] = float(np.nanmax(me[np.isfinite(me)])) if np.isfinite(me).any() else None
    except Exception:
        pass
    return out


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


def _tag_tracks(rows):
    """Which track each run can serve. Track B needs a protrusion; Track A needs it to WORK."""
    for r in rows:
        pat = (r.get("patterned") or 0) > 0.01 and r.get("chem_ok")
        snd = r["specimen"] in ("valid", "valid (declared)")
        # A CAPPED RUN SERVES NEITHER TRACK. It will be refused P2_BUFFER_SATURATED the moment
        # it is replayed, so choosing it spends a slot to be told something already known.
        if r.get("capped"):
            r["serves"] = "capped"
            continue
        r["serves"] = ("B" if (pat and snd and (r.get("n_protruding_max") or 0) > 0)
                       else "A" if (pat and snd) else "-")
    return rows


def log_table(rows=None, top=18):
    rows = _tag_tracks(survey_log() if rows is None else rows)
    if not rows:
        return "log/okuda holds no finished run with a composition and an admitted number."
    L = [f"{len(rows)} finished run(s) on disk with a composition and a measured protr_peak.",
         "Sorted by PATTERN first, then soundness, then elongation.",
         "",
         f"{'run':26}{'serves':>7}{'pattern':>9}{'chem':>6}{'protr_peak':>11}{'cells':>7}"
         f"{'protrud':>8}  specimen"]
    for r in rows[:top]:
        L.append(f"{r['run'][:25]:26}{str(r.get('serves', '?')):>7}{_fmt(r.get('patterned')):>9}"
                 f"{('ok' if r.get('chem_ok') else 'NaN'):>6}"
                 f"{r['protr_peak']:>11.2f}{(r['n_cells_final'] or 0):>7.0f}"
                 f"{(r.get('n_protruding_max') if r.get('n_protruding_max') is not None else '—'):>8}"
                 f"  {r['specimen']}"
                 + (f"  [{', '.join(r['broken'][:3])}]" if r["broken"] else ""))
    sound = [r for r in rows if r["specimen"] in ("valid", "valid (declared)")]
    npat = sum(1 for r in rows if r["serves"] in ("A", "B"))
    nb = sum(1 for r in rows if r["serves"] == "B")
    L += ["", f"`serves` A = usable for TRACK A (understand the operator landscape): it runs, it "
              f"patterns, the Biologist passes it. B = also usable for TRACK B (reproduce "
              f"Okuda's figures): it additionally protruded.",
          f"{npat} of {len(rows)} serve Track A. {nb} serve Track B.",
          f"protr_peak and protrud are TRACK B measures and say nothing about a Track A starting "
          f"point -- a composition that patterns without protruding is a point on the map.",
          f"{len(sound)} of {len(rows)} were run on a specimen the Biologist passed.",
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
    # WHICH TRACK CAN THIS RUN SERVE? The cold start used to rank every candidate on protr_peak
    # and n_protruding and then give up when they came back zero across the whole log. Those are
    # TRACK B measures -- they say whether we made Okuda's picture. Track A asks a different
    # question, about the landscape of operators, and a composition that RUNS, PATTERNS and holds
    # a SOUND specimen is good Track A evidence whether or not anything ever protruded.
    #
    # Giving up on a Track B criterion is giving up on the very work you do when you cannot yet
    # make the picture. It is also how a fallback to okuda_route -- whose chemistry explodes --
    # came to be the starting point three times over.
    _tag_tracks(rows)
    patterned = [r for r in rows if r["serves"] in ("A", "B")]
    if not patterned:
        return _record({
            "stage": "cold_start", "start": [], "asked": False,
            "n_on_disk": len(rows), "n_sound": len(sound), "n_protruded": 0,
            "why": f"NOT ONE of the {len(rows)} finished runs on disk both patterns and passes "
                   f"the Biologist, so none is usable even for Track A. The frontier falls back "
                   f"to the reference recipes -- a CHOICE recorded here, not a default nobody "
                   f"noticed."})
    tab = log_table(rows)
    rows = patterned + [r for r in rows if r not in patterned]

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
