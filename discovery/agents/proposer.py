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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import llm                                                       # noqa: E402
from llm import CAUSALITY_RULE, budget_note, ensure_files, read_file, run_claude  # noqa: E402

PROPOSAL_FILE = os.path.join(llm.CAMPAIGN, "proposal.json")


def _legal_menu(frontier, cfg, prox, max_per_parent=6):
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
            if len(rows) >= max_per_parent:
                break
        if rows:
            menu.append({"parent_index": gi,
                         "parent": "+".join(sorted(set(g.op_names()))),
                         "parent_region": g.name_region(), "legal_edits": rows})
    return menu


def propose(frontier, cfg, prox, ledger_summary, round_id, n_slots=8, timeout_min=10):
    """Ask the agent for a batch. Returns (ok, [slot dicts]).

    Slot 0 is ALWAYS the parent unchanged -- the CONTROL. That is the one thing the agent is not
    allowed to spend, because without it a difference between candidates cannot be separated
    from seed noise. My first design had no control slot at all.
    """
    paths = ensure_files(cfg.objective)
    menu = _legal_menu(frontier, cfg, prox)
    grounding = _ground(cfg)

    prompt = f"""ROUND {round_id}: propose the next batch of {n_slots} experiments.
{budget_note(timeout_min, "1) proposal.json  2) an entry appended to analysis.md  3) memory.md")}
Read these, in this order:
  instructions : {paths['instruction']}
  memory       : {paths['memory']}
  user input   : {paths['user_input']}   (acknowledge anything Pending, with a timestamp)

EVIDENCE SO FAR
{ledger_summary}

WHAT THE REFERENCE MODEL SAYS (from the Grounder)
{grounding}

LEGAL MOVES (you may ONLY choose from these; the type system has already removed everything
ill-typed, everything with an unmet precondition, and everything with a dangling slot)
{json.dumps(menu, indent=1)[:6000]}
{CAUSALITY_RULE}
WRITE {PROPOSAL_FILE} as JSON:
{{
 "reasoning": "<what the evidence suggests, in 3-6 sentences>",
 "mode": "explore" | "robustness",
 "slots": [
   {{"parent_index": 0, "edit": null, "intent": "control",
     "claim": "the parent, unchanged -- the control",
     "metric": "protr_peak", "predicted": "<a number or range you expect>",
     "why": "control"}},
   {{"parent_index": <int>, "edit": <one entry copied EXACTLY from legal_edits>,
     "intent": "confirmatory" | "adversarial",
     "claim": "<falsifiable, one sentence>",
     "metric": "protr_peak",
     "predicted": "<e.g. 'protr_peak >= 3.0' -- a claim you could be WRONG about>",
     "why": "<the reason this edit is worth a GPU-hour, citing the evidence or the paper>"}}
   ... {n_slots - 1} more
 ]
}}

RULES
 - Slot 0 MUST be the control (edit: null).
 - Aim for ~70% confirmatory / ~30% adversarial across the remaining slots.
 - A prediction you are certain of is nearly worthless. Prefer edits you genuinely cannot call.
 - Do NOT propose a parameter change. Composition identity excludes parameters; a retune is
   Loop II's job and CANNOT count as a new hypothesis here.
 - Then append one dated entry to {paths['analysis']} and revise {paths['memory']}.
"""
    ok, out = run_claude(prompt, timeout_min=timeout_min)
    if not os.path.exists(PROPOSAL_FILE):
        return False, []
    try:
        p = json.load(open(PROPOSAL_FILE))
    except Exception as e:
        return False, []
    slots = p.get("slots", [])[:n_slots]
    if not slots or slots[0].get("intent") != "control":
        # do not silently repair the agent's design error -- report it
        return False, slots
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
