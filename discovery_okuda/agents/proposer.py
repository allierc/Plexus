"""proposer -- the LLM agent that decides WHAT IS WORTH TESTING, and commits to predictions.

Why this must be an LLM, stated plainly because I got it wrong first:

    My first "Proposer" enumerated the legal one-edit moves and shuffled them. That is an
    ENUMERATOR. Worse, it made the whole hypothesis-first protocol VACUOUS -- intent was
    assigned by whether the edit label began with a minus sign, and the prediction followed
    mechanically from the intent:

        intent = "adversarial" if lbl.startswith("-") else "confirmatory"
        pred   = "protr_peak >= 3.0" if intent == "confirmatory" else "protr_peak < 3.0"

    So the "surprise rate" measured nothing about anyone's beliefs; it measured whether
    removals happen to lower protr_peak. The 70/30 setpoint was regulating a tautology.

A real proposal requires reading the ledger, asking the Grounder what the paper says, choosing
an edit BECAUSE OF a stated reason, and committing to a prediction that could be wrong. Only the
last of those makes the surprise rate informative.

The enumerator is kept -- as the LEGAL MOVE SET handed to the agent. The type system decides
what is possible; the agent decides what is worth doing.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import llm
import templates as _T
TEMPLATES = _T.prompt_block()                                                       # noqa: E402
from llm import CAUSALITY_RULE, budget_note, ensure_files, read_file, run_agent  # noqa: E402

PROPOSAL_FILE = os.path.join(llm.CAMPAIGN, "proposal.json")


def _render_menu(menu, budget=9000):
    """The menu as ONE LINE PER MOVE, never cut mid-structure.

    It was `json.dumps(menu, indent=1)[:6000]` -- a character slice through indented JSON, which
    ends wherever 6000 lands and hands the agent a truncated object it cannot parse. Whatever it
    then proposed was chosen from a menu it could not read to the end of.

    Dropping whole moves instead, and SAYING how many were dropped, keeps every line legal and
    keeps the omission visible; a silently shortened menu reads as the complete space of moves.
    """
    lines, shown, total = [], 0, sum(len(p["legal_edits"]) for p in menu)
    for p in menu:
        lines.append(f'parent {p["parent_index"]}: {p["parent"]}   [{p["parent_region"]}]')
        for r in p["legal_edits"]:
            ln = f'  parent_index={p["parent_index"]}  edit={json.dumps(r["edit"])}  -> {r["yields"]}'
            if sum(len(x) for x in lines) + len(ln) > budget:
                lines.append(f'  ... {total - shown} further legal move(s) omitted for length')
                return "\n".join(lines)
            lines.append(ln)
            shown += 1
    return "\n".join(lines)


# SIX WAS A CEILING ON CHOICE, NOT ON COST. With two parents on the frontier the menu held at
# most twelve moves and round 2 was asked for twelve slots -- so the Proposer could not select, it
# could only enumerate, and when twelve were not enough it invented `cell_react0` and
# `cell_diffuse0`, which parent 0 does not contain. Eight of its twelve slots died: four on a
# precondition, three as KeyError(None), one on the second parent.
#
# A menu must be larger than the batch or there is no proposing being done. Twenty-four per parent
# against a batch of twelve leaves the agent something to reject, which is the whole of its job.
def _legal_menu(frontier, cfg, prox, max_per_parent=24):
    """The legal move set, as a menu the agent chooses FROM.

    The type system decides what is POSSIBLE (ill-typed edits, unmet preconditions and dangling
    slots never appear). The agent decides what is WORTH DOING. Neither can do the other's job:
    an LLM cannot be trusted to respect the type system, and the type system has no taste.
    """
    menu = []
    for gi, g in enumerate(frontier):
        rows = []
        for e, lbl in g.legal_edits(cfg.stage_gate):
            try:
                child, _ = g.apply(e)
            except Exception:
                continue
            ok, _ = child.is_runnable()
            if not ok:
                continue
            rows.append({"edit": list(e) if isinstance(e, tuple) else e, "label": lbl,
                         "yields": child.name_region()})
        # STRUCTURE FIRST, AND NUMBERS RATIONED. Parameter moves outnumber structural ones about
        # three to one -- 17 of 24 on the current frontier -- so a menu truncated at any length
        # would show mostly numbers, and the Proposer would spend a batch turning dials. Track A
        # asks whether a MECHANISM matters before it asks what it does as you turn it up, and the
        # menu should read in that order. Every structural move is offered; parameter moves are
        # capped, so they are present as a real option and cannot crowd the others out.
        _struct = [r for r in rows if r["edit"][0] != "set_param"]
        _param = [r for r in rows if r["edit"][0] == "set_param"]
        rows = (_struct + _param[:max(2, max_per_parent // 4)])[:max_per_parent]
        if rows:
            menu.append({"parent_index": gi,
                         "parent": "+".join(sorted(set(g.op_names()))),
                         "parent_region": g.name_region(), "legal_edits": rows})
    return menu


def _block(title, body):
    """A prompt section that DISAPPEARS when empty, rather than saying 'none' in ten lines.

    Every one of these carries a finding from another role that previously reached nobody. The
    campaign's defining failure was not bad judgement, it was correct judgement with no recipient.
    """
    if not body:
        return ""
    return f"\n{title.upper()}\n{str(body).strip()}\n"


def propose(frontier, cfg, prox, ledger_summary, round_id, n_slots=8, timeout_min=10,
            ledger=None, steer=None, refusals=None, setup=None, history=None, review=None):
    """Ask the agent for a batch. Returns (ok, [slot dicts]).

    Slot 0 is ALWAYS the parent unchanged -- the CONTROL. That is the one thing the agent is not
    allowed to spend, because without it a difference between candidates cannot be separated
    from seed noise. My first design had no control slot at all.

    `ledger` is the campaign's BudgetLedger (round.py owns it). The proposer's call is the most
    expensive of the round; it used to call run_claude() directly, so its cost has never once
    been measured. Note the two SEPARATE roles recorded here: "grounder" (local retrieval, no
    model) and "proposer" (the model call). Lumping them would hide which one is slow.
    """
    paths = ensure_files(cfg.objective)
    menu = _legal_menu(frontier, cfg, prox)
    if ledger is not None:
        with ledger.timed("grounder", kind="local"):
            grounding = _ground(cfg)
    else:
        grounding = _ground(cfg)

    prompt = f"""ROUND {round_id}: propose the next batch of {n_slots} experiments.
{budget_note(timeout_min, "1) proposal.json -- nothing else")}

YOU WRITE NO RECORD. You used to append to analysis.md and rewrite memory.md, and that put the
agent under evaluation in charge of its own record -- which is how "parent 2 is fully PROPOSED"
came to be logged as coverage: territory counted because it had been PROPOSED, never because
anything had been MEASURED. The Collector now writes analysis.md from the files on disk, and the
Meta-review writes memory.md. READ them. Do not edit them. Emit proposal.json and stop.

Read these, in this order:
  instructions : {paths['instruction']}
  memory       : {paths['memory']}
  user input   : {paths['user_input']}   (acknowledge anything Pending, with a timestamp)

EVIDENCE SO FAR
{ledger_summary}
{_block("WHAT THE SUPERVISOR IS STEERING TOWARD -- this is an instruction, not context", steer)}
{_block("WHAT WAS REFUSED LAST ROUND, and why. An empty map beside a non-zero attempt count is "
        "NOT a reset counter: it means the compositions proposed so far CANNOT BE SIMULATED. "
        "A parent is explored when its edits produced EVIDENCE, never when they were merely "
        "proposed", refusals)}
{_block("WHAT THE PEER-REVIEWER SAID ABOUT YOUR LAST BATCH. It advises and cannot refuse, so "
        "these criticisms are only worth the compute they were paid for if you act on them",
        review)}
{_block("THE ARCHIVIST, over the whole history. A branch with evidence but no sound specimens "
        "has been measured and has told us nothing about tissue", history)}
WHAT THE PAPER SAYS (from the Grounder, who read it and checked its own quotes)
{setup or grounding}

LEGAL MOVES (you may ONLY choose from these; the type system has already removed everything
ill-typed, everything with an unmet precondition, and everything with a dangling slot)

COPY AN `edit` TOKEN VERBATIM. An operator name that is not in this list DOES NOT EXIST in this
system, however real the biology is. There is no `branching` operator, no `chemotaxis`, no
`apical_constriction`: those are phenotypes you may HOPE FOR, and the way to reach for one is to
choose the structural move you think produces it and say so in `claim`.

TWO KINDS OF MOVE, and they answer different questions.
  STRUCTURAL  add_op / remove_op / set_impl / connect -- "does this mechanism matter at all?"
  PARAMETER   set_param                               -- "what does it do as you turn it up?"
A `set_param` move keeps the composition's IDENTITY: it is the same mechanism at a different
setting, and it is recorded as a point on that mechanism rather than as a new one. Both are
legitimate and both carry a real prediction. Ask the first question of a mechanism before the
second, and do not spend a batch turning dials on something you have not shown to matter.
{_render_menu(menu)}
{CAUSALITY_RULE}
WRITE {PROPOSAL_FILE} as JSON:
{{
 "reasoning": "<what the evidence suggests, <=80 words>",
 "mode": "explore" | "robustness",
 "slots": [                  <-- THIS KEY, EXACTLY. Not "candidates", not "experiments".
   The loader reads `slots` and nothing else; a batch under any other key is discarded
   whole, and you will be told only "no usable proposal".
   {{"parent_index": 0, "edit": null, "intent": "control",
     "claim": "the parent, unchanged -- the control", "track": "B",
     "metric": "protr_peak", "predicted": "<a number or range you expect>",
     "why": "control"}},
   {{"parent_index": <int>, "edit": <one entry copied EXACTLY from legal_edits>,
     "intent": "confirmatory" | "adversarial",
     "track": "A" | "B",
     "territory": "in_paper" | "excursion",
     "claim": "<falsifiable, one sentence>",
     "claim_kind": "sufficient" | "necessary" | "causal" | "descriptive",
     "revisits": "<a claim id from memory.md this slot CHALLENGES, or null>",
     "confounder": "<the named thing this slot VARIES vs the runs that established that claim,
                     or null. Required whenever `revisits` is set.>",
     "metric": "protr_peak",
     "predicted": "<e.g. 'protr_peak >= 3.0' -- a claim you could be WRONG about>",
     "why": "<the reason this edit is worth a GPU-hour, citing the evidence or the paper>"}}
   ... {n_slots - 1} more
 ]
}}

RULES
 - Slot 0 MUST be the control (edit: null), with intent "control".
 - THE TRACKS ARE ASYMMETRIC. This is new, and it is the point of the round.
     TRACK A IS OPEN. It MAY violate any trap or prohibition in memory.md. Those were written
     from evidence that is mostly one run deep, and several are known to be wrong: "Cell division
     is NECESSARY for the bud" was established in round 2 and REFUTED in round 8 by an edit that
     broke its own ban. A trap is a summary of what has been tried, never a fact about nature.
     TRACK B IS CONSTRAINED. It respects traps, cites a named Okuda morphology and its settings,
     and may not breed from a specimen whose premises were broken.
 - AT LEAST TWO SLOTS PER BATCH MUST SET `revisits`, challenging an existing claim. Prefer claims
   supported by a SINGLE run, and claims marked provisional. This is the quota that keeps the map
   honest, and it is yours to fill -- a revisit inserted by code after the batch is checked would
   be the one slot exempt from the duplicate and confounder gates, i.e. a licensed re-run.
 - A REVISIT MUST VARY A NAMED CONFOUNDER, and say which. Re-running the same composition is not
   a revisit: rounds 4-8 produced bit-identical results under four different hashes and recorded
   them as four observations. If you cannot name what you are varying, you are not revisiting.
 - `claim_kind` IS NOT DECORATION. "necessary" means you assert the effect CANNOT occur without
   this operator -- the most expensive claim available, and it requires an ablation in this batch.
   Do not mark a slot `necessary` unless the batch actually tests it.
 - Aim for ~70% confirmatory / ~30% adversarial across the remaining slots. The control is not
   part of that ratio -- it is a fixed cost of the design, not one of your choices.
 - EVERY SLOT DECLARES ITS TRACK, and the batch is partitioned between them. This campaign is
   doing two things at once and a slot that does not know which one it serves will serve neither:
     "track": "B"  -- REPRODUCE OKUDA. At or near a setting he reports, aimed at one of his
                      morphologies (undulation / thin tube / thick tube / branching). Success is
                      that the picture appears. Say WHICH morphology in `claim`.
     "track": "A"  -- UNDERSTAND THE MECHANISM. Aimed at the MAP, not the picture: what does this
                      operator do, is it necessary, what happens at the extreme. Success is that
                      a map cell can state a verdict afterwards, even a negative one.
   The objective is a MAP, not a target: running only his settings teaches us to reproduce his
   figures rather than to understand the system, and running only excursions produces
   understanding of a model nobody has a reason to believe.
   `track` and `territory` usually agree (B/in_paper, A/excursion) and are NOT the same thing:
   an in-paper setting can serve Track A when the question is which operator carries the effect.
 - SEPARATELY, aim for ~70% `in_paper` / ~30% `excursion`. This is a DIFFERENT axis from
   confirmatory/adversarial and the two are independent -- an excursion can be confirmatory
   ("at extreme diffusion I expect a flat sheet") and an in-paper slot can be adversarial.
   * `in_paper`  -- at or near a setting Okuda reports, aimed at reproducing one of his
     morphologies (undulation / thin tube / thick tube / branching).
   * `excursion` -- deliberately OUTSIDE his published space: an extremum, a regime he never
     shows, or a combination he had no reason to try. Say in `why` what perspective you expect
     it to give that a published point cannot.
   Extrema often reveal what a lever DOES more plainly than the operating point does, and running
   only his settings teaches us to reproduce his figures rather than to understand the system --
   the objective is a MAP, not a target. This is also where a phenotype nobody has named is most
   likely to appear, and there is now machinery to catch one: an unlisted phenotype gets its own
   scoreboard row, its topology is checked (a ring cannot be legally produced, so an apparent one
   is a mesh bug until proven otherwise), and if no admitted metric can measure it, one is written
   and must pass known shapes before it counts.
   An excursion is NOT licence to be arbitrary: it still carries a falsifiable prediction, and a
   run that diverges numerically is refused as evidence, so wild settings cost compute and buy
   nothing.
 - A prediction you are certain of is nearly worthless. Prefer edits you genuinely cannot call.
 - A PARAMETER MOVE IS ALLOWED AND IS NOT A NEW MECHANISM. `set_param` appears in the menu when
   it is legal; composition identity excludes parameters, so the same mechanism at a different
   setting is recorded as a POINT ON IT, never as a discovery. Past parameter rounds are fully
   legitimate evidence -- the vcap sweep is where the non-monotone response and the "forced, not
   grown" reading came from. Do not describe one as forbidden or invalid.
 - Write your PREDICTIONS so they can be checked mechanically. Each must contain at least one
   clause of the form `<metric> <op> <value>` (e.g. `protr_peak >= 2.0`) or a range
   (`protr_peak 2.0-3.5`), naming a metric from the admitted list. State the metric explicitly
   in every clause -- `p_ratio drops toward ~1` is NOT checkable, `mech_p_ratio <= 1.5` is.
   A prediction with no checkable clause is scored `inconclusive`: it buys a GPU-hour and
   contributes nothing to the map. You may add a `REFUTED if ...` sentence; it is recorded but
   the assertion before it is what gets checked.
 - Write {PROPOSAL_FILE} and NOTHING else. The record is not yours to write.
"""
    # THE PROPOSAL FILE IS DELETED BEFORE THE CALL, so a stale one cannot be adopted.
    #
    # MEASURED on 2026-08-01, on the first live launch of the rebuilt loop. `proposal.json` was
    # six seconds OLDER than the campaign process that read it. The Proposer tried to Write it,
    # the harness's read-before-write rule blocked the overwrite, the agent read the existing
    # file and reported: "proposal.json already holds a complete, valid Round-1 proposal ... so
    # no rewrite is needed." It then verified that proposal against every rule and adopted it.
    #
    # Every step of that is reasonable behaviour, and the round would have run somebody else's
    # batch. The guard meant to prevent clobbering became a mechanism for inheriting stale work,
    # which is the same shape as every other defect this campaign has found: a correct-looking
    # artefact adopted because nothing checked WHEN it was made.
    #
    # Detecting it afterwards is not enough -- an mtime comparison would have to guess a
    # tolerance. Removing the file makes the failure impossible: the agent has nothing to
    # inherit, and an absent file after the call is an unambiguous refusal.
    try:
        os.remove(PROPOSAL_FILE)
    except FileNotFoundError:
        pass
    t_call = time.time()

    # allowed_tools is stated explicitly as llm.DEFAULT_TOOLS -- exactly what the bypassing
    # `run_claude(prompt, timeout_min=...)` call was getting. Measurement-only: no knob moves.
    ok, out = run_agent("proposer", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=llm.DEFAULT_TOOLS)
    if not os.path.exists(PROPOSAL_FILE):
        print("[proposer] no proposal.json was written -- the round has no reasoned batch, and "
              "a random one is not a substitute")
        return False, []
    if os.path.getmtime(PROPOSAL_FILE) < t_call:
        # belt and braces: the file exists but predates the call, which can only mean something
        # outside this call created it while the call was running.
        print(f"[proposer] REFUSING a proposal.json older than the call that asked for it "
              f"({os.path.getmtime(PROPOSAL_FILE):.0f} < {t_call:.0f})")
        return False, []
    try:
        p = json.load(open(PROPOSAL_FILE))
    except Exception as e:
        print(f"[proposer] proposal.json does not parse: {type(e).__name__}: {str(e)[:120]}")
        return False, []
    # ACCEPT THE ALIAS. The prompt asks for "slots"; the agent has been writing "candidates",
    # and p.get("slots", []) returned [] every round -- twelve reasoned experiments discarded on
    # a key name, with "no usable proposal" as the only trace. Measured on a live proposal.json:
    # top-level keys were campaign, CANDIDATES, comp_hash, metric, mode, parent_spec, rationale,
    # round, track. Losing a batch to vocabulary is the cheapest possible failure to prevent.
    _key = "slots" if p.get("slots") else ("candidates" if p.get("candidates") else "slots")
    if _key != "slots":
        print(f"[proposer] read {len(p.get(_key) or [])} experiment(s) from `candidates` -- the "
              f"prompt asks for `slots`. Accepted; the key is not the science.")
    slots = (p.get(_key) or [])[:n_slots]
    # REPORT IT, which the comment below has always promised and never done. "no usable
    # proposal" could mean the file was never written, the JSON did not parse, or slot 0 was
    # not the control -- three different faults with one message and no way to tell them apart.
    # Rounds 2 to 13 of one campaign were refused here with no reason printed at all.
    if not slots:
        print(f"[proposer] proposal.json has no `slots` (top-level keys: "
              f"{', '.join(sorted(p)[:8]) or 'none'})")
        return False, []
    if slots[0].get("intent") != "control":
        print(f"[proposer] slot 0 is not the control: intent="
              f"{slots[0].get('intent')!r}, edit={str(slots[0].get('edit'))[:60]!r}. "
              f"Slot 0 MUST be the parent unchanged -- without it a difference between two "
              f"candidates cannot be separated from seed noise.")
        return False, slots
    # A slot the batch builder cannot use is worth naming here rather than at the far end.
    _bad = [i for i, sl in enumerate(slots)
            if sl.get("parent_index") is None and sl.get("intent") != "control"]
    if _bad:
        print(f"[proposer] {len(_bad)} slot(s) carry no parent_index: {_bad[:8]}")
    return ok, slots


def _ground(cfg, k=3):
    try:
        import grounder as G
        pd = G.phase_diagram()
        u = G.understand("what regime produces tubulation versus undulation versus branching, "
                         "and what do chi and gamma control?", k=k)
        lines = ["Okuda's (chi, gamma) phase diagram -- the qualitative reproduction target:"]
        for key, f in pd.items():
            lines.append(f"  {f['figure']:11} chi={f['chi']:<6} gamma={f['gamma']:<7} "
                         f"-> {f['phenotype']}")
        lines.append("")
        for pp in u["passages"][:2]:
            lines.append(f"  [{pp['cite']}] {pp['text'][:260]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"(grounder unavailable: {type(e).__name__})"


# ============================================================ RECONNAISSANCE (round 0 of a campaign)
def choose_specs(table, n=6, timeout_min=6, ledger=None):
    """Pick N runs already on disk to re-measure. NO hypotheses, NO predictions.

    WHY A ROUND WITH NO HYPOTHESIS IS STILL A ROUND. The campaign has 63 finished runs and 46 of
    them serve Track A -- they run, they pattern, the Biologist passes them -- and it has been
    starting from the reference recipes instead, three times over, because the cold start ranked
    them on protrusion and found zeros. Meanwhile every number on disk was taken with at least one
    instrument since proved wrong: the black-movie colour scale, the seed-sphere Q, the uncertified
    metrics, the reaction that diverges.

    So the first round LOOKS. It re-measures existing specs, unmodified, with the instruments as
    they are now. That is not a hypothesis test and must not pretend to be one -- there is nothing
    to be surprised by, and a prediction attached to a replay would be a prediction about our own
    past arithmetic. The product is a frontier of compositions we have actually seen work, and a
    set of numbers we are entitled to believe.
    """
    prompt = f"""ROUND 0: RECONNAISSANCE. Choose {n} runs already on disk to RE-MEASURE.

{budget_note(timeout_min, "1) the JSON choice")}

You are NOT proposing experiments and you will NOT write predictions. Every number below was
taken with at least one instrument since proved wrong, so the job is to re-measure a good spread
of what we already have, with the instruments as they are now, and end up with a frontier of
compositions we have actually seen work.

{table}

Choose {n} runs. What makes a good set:
  * they SERVE TRACK A -- they run, they pattern, the Biologist passes them. That is the bar for
    a starting point. `protr_peak` and `protrud` are TRACK B measures and are not the bar here:
    a composition that patterns without protruding is a point on the operator map.
  * they are DIFFERENT FROM EACH OTHER. Six variations of one composition is one starting point,
    not six. Spread across compositions, regions and parameter regimes.
  * prefer a live pattern (`pattern` high) over elongation, and a sound specimen over both.

Reply with ONLY:
{{"runs": ["<run name>", ...],
  "why": "<<=50 words: what this set spans, and what it deliberately leaves out>"}}"""
    ok, out = run_agent("proposer", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read"])
    import re as _re
    for m in _re.finditer(r"\{.*?\}", out or "", _re.S):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(d.get("runs"), list) and d["runs"]:
            return ok, d
    return False, {"runs": [], "why": "the proposer named no runs"}
