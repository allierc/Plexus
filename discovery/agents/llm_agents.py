"""llm_agents -- the remaining LLM agents, following Co-Scientist and Robin.

WHAT THE PAPERS DO
------------------
Co-Scientist runs SIX specialized LLM agents under a mechanical Supervisor:
    Generation   propose hypotheses (literature exploration, simulated scientific debate)
    Reflection   peer review WITH SEARCH -- their ablation showed the search tool "effectively
                 prevented the hallucination of seemingly novel but implausible hypotheses"
    Ranking      Elo tournament by simulated scientific DEBATE; the debate prompt "substantially
                 improved the ranking and reduced positional bias"
    Proximity    a proximity graph for clustering and de-duplication
    Evolution    REFINES the top-ranked (synthesis, analogy, literature grounding, simplification)
    Meta-review  distils recurring patterns from all reviews and APPENDS them to the other
                 agents' prompts -- "feedback propagation without back-propagation", no fine-tuning

Robin uses Crow/Falcon (literature) and Finch (data analysis, writing code in a notebook), ranks
with an LLM JUDGE over pairwise comparisons aggregated by Bradley-Terry-Luce, and -- decisively --
runs EIGHT INDEPENDENT Finch trajectories on the same data, reconciled into a consensus, because
"the analysis results can vary between runs, even when given identical prompts and data".

WHERE WE COMPLY, AND WHERE WE UPGRADE
-------------------------------------
Comply: every agent that GENERATES or JUDGES is an LLM. Reflection and Evolution are added --
both papers have them and I had neither (I only type-checked, and I only truncated, never
refined).

Upgrade 1 -- RANKING IS BOTH. They rank by debate because a wet-lab assay is scarce. Ours is a
simulation scored by code, so measurement is available and an opinion about available ground
truth is a step backwards. BUT the instrument gate just measured a metric scoring 9.30 on a BUD.
So: the metric bank ranks (primary), an LLM/VLM judge ranks independently from the MOVIE
(secondary), and THEIR DISAGREEMENT IS THE SIGNAL -- it is the eye/number divergence detector,
which neither paper needs because their measurement is external to their system.

Upgrade 2 -- THE SCIENTIFIC METHOD. Neither paper pre-registers. Their hypotheses are generated,
reviewed and ranked, but nothing records what was BELIEVED before the evidence arrived, so a
prediction cannot be distinguished from a post-hoc rationalisation. We require a falsifiable
prediction BEFORE the run, classify every outcome in a confirmatory/adversarial x
confirmed/refuted 2x2, and drive the batch mixture from the resulting SURPRISE RATE. The
knowledge document is append-only and ordered as the science happened.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm                                                              # noqa: E402
from llm import budget_note, ensure_files, read_file, run_claude        # noqa: E402

CAMP = llm.CAMPAIGN


# ============================================================================ 7. ANALYST x N
def analyse(run_name, out_dir, n=3, timeout_min=6):
    """N INDEPENDENT readings of the SAME run, reconciled by consensus (Robin's 8x Finch).

    The point is not redundancy, it is that a single LLM reading is not reproducible: Robin
    observed their analysis "can vary between runs, even when given identical prompts and data",
    and used that diversity deliberately. Disagreement between trajectories is recorded, not
    hidden -- it is the calibration signal for how much any one reading is worth.
    """
    diag = os.path.join(out_dir, "diag.json")
    desc = os.path.join(out_dir, "description.txt")
    mech = os.path.join(out_dir, "mechanics.png")
    reads = []
    for i in range(n):
        prompt = f"""ANALYST {i + 1} of {n}. Read ONE simulation run and report what happened.

Work independently. Do not try to agree with anyone.
{budget_note(timeout_min, "1) your JSON verdict  2) nothing else")}
  metrics    : {diag}
  VLM caption: {desc}
  mechanics  : {mech}   (force | pressure body-vs-protruding | tension | migration)
  strip      : {os.path.join(out_dir, 'strip.png')}

Only these metrics are admissible (the others were MEASURED to lie and are excluded):
  ADMITTED  protr_peak, ta_n_tubes_final, protr_final
  REJECTED  ta_aspect_len_over_diam, ta_tube_len_final, retention
Informative but not scored: mech_p_ratio (~3 = FORCED protrusion, ~1 = growth-driven equilibrium).

Reply with ONLY this JSON, nothing else:
{{"phenotype": "sphere|bud|spike|tube|branching|undulation|exploded|degenerate",
  "confidence": 0.0-1.0,
  "forced_or_grown": "forced|grown|unclear",
  "evidence": "<2 sentences citing SPECIFIC numbers or what the caption says>",
  "eye_vs_number": "agree|disagree",
  "concern": "<anything that looks like an artefact rather than physics, or empty>"}}"""
        ok, out = run_claude(prompt, timeout_min=timeout_min, allowed_tools=["Read"], quiet=True)
        reads.append(_first_json(out) or {"phenotype": "unreadable", "confidence": 0.0})

    phen = [r.get("phenotype") for r in reads if r.get("phenotype")]
    consensus = max(set(phen), key=phen.count) if phen else "unreadable"
    agree = phen.count(consensus) / max(1, len(phen))
    return {"analyst_consensus": consensus,
            "analyst_agreement": round(agree, 2),
            "analyst_forced_or_grown": _majority([r.get("forced_or_grown") for r in reads]),
            "analyst_disagreement": agree < 0.67,        # recorded, not hidden
            "analyst_concerns": [r.get("concern") for r in reads if r.get("concern")],
            "analyst_reads": reads}


# ============================================================================ 8b. WATCHER veto
def watch(run_name, out_dir, expected_phenotype, timeout_min=5):
    """Compare what the movie SHOWS against what the composition CLAIMED, and gate promotion.

    The Watcher's eye already exists (the VLM captions every run). This is the missing half: the
    VETO. Without it the caption is produced and ignored, and the loop can be fooled exactly as I
    was -- by a metric scoring a tube on a bud.
    """
    desc = read_file(os.path.join(out_dir, "description.txt"), 4000)
    if not desc or desc.startswith("UNAVAILABLE"):
        return {"watcher_verdict": "no_caption", "watcher_blocks": False,
                "watcher_why": "no caption was produced -- cannot gate, recorded as such"}
    prompt = f"""WATCHER. A run CLAIMED to produce: {expected_phenotype!r}

This is what a vision model saw in the rendered movie:
---
{desc[:2500]}
---
{budget_note(timeout_min, "1) the JSON verdict")}
Does the description support the claim? Be strict: 'a small bud' does NOT support 'a tube', and
'a chaotic cluster of shards' does NOT support any clean morphology.

Reply with ONLY: {{"supports": true|false, "seen": "<what it actually shows, 6 words>",
                  "why": "<one sentence>"}}"""
    ok, out = run_claude(prompt, timeout_min=timeout_min, allowed_tools=[], quiet=True)
    j = _first_json(out) or {}
    supports = bool(j.get("supports", True))
    return {"watcher_verdict": "supports" if supports else "CONTRADICTS",
            "watcher_seen": j.get("seen", ""), "watcher_why": j.get("why", ""),
            "watcher_blocks": not supports}


# ============================================================================ 9. INTERPRETER
def interpret(comp_hash, region, edit, summary, analyst, out_path, timeout_min=8):
    """The causal description: THIS SPEC GIVES THIS PHENOMENON, BY THIS ROUTE.

    This is what makes the operator library reusable rather than merely catalogued, and it is
    what a reader of the paper actually wants. Co-Scientist's Meta-review synthesises a research
    overview at the END; we write the causal account PER COMPOSITION, CONTINUOUSLY, so it is
    evidence rather than a summary written afterwards.
    """
    prompt = f"""INTERPRETER. Write the causal description for one composition.
{budget_note(timeout_min, "1) append the entry to the file  2) nothing else")}
composition : {comp_hash}   ({region})
edit tested : {edit}
measured    : {json.dumps({k: v for k, v in summary.items() if not k.startswith('analyst_')})}
analysts    : {json.dumps(analyst, default=str)[:900]}

APPEND to {out_path} a markdown entry of exactly this shape:

### {comp_hash} — {region}
- **phenomenon**: <what it does, one sentence, in the language a biologist would use>
- **route**: <the MECHANISM: which operator does what to which, in order, so the phenomenon
  follows. Not a restatement of the operator list -- an account of WHY these operators produce
  THIS shape.>
- **evidence**: <the specific numbers, named>
- **forced or grown**: <and how you can tell>
- **what would falsify this**: <the single edit that should destroy the phenomenon if the route
  above is correct>

Be honest about uncertainty. If the route is not determined by the evidence, say which
additional measurement would determine it."""
    ok, out = run_claude(prompt, timeout_min=timeout_min,
                         allowed_tools=["Read", "Edit", "Write"], quiet=True)
    return ok


# ============================================================================ 10. META-REVIEW
def meta_review(round_id, timeout_min=10, max_chars=4000):
    """Distil recurring patterns and APPEND them to the Proposer's instructions.

    Co-Scientist: "feedback applicable to all agents, which is simply appended to their prompts
    in the next iteration -- feedback propagation and learning without back-propagation".

    The bounded-growth rule is ours: over a multi-week run an append-only prompt grows without
    limit and eventually crowds out the task. The distilled section is CAPPED and rewritten in
    place each time, so it stays a summary rather than becoming a transcript.
    """
    paths = ensure_files()
    prompt = f"""META-REVIEW after round {round_id}.
{budget_note(timeout_min, "1) the LEARNED PATTERNS section in instruction.md  2) memory.md")}
Read:
  analysis log : {paths['analysis']}
  memory       : {paths['memory']}
  knowledge    : {os.path.join(CAMP, 'knowledge.md')}

Find the patterns that RECUR across rounds -- not a summary of what happened, but what a
proposer should know before choosing the next edit. Especially:
  * edits that repeatedly fail, and the reason they fail (so they are never proposed again);
  * predictions that were repeatedly WRONG, and in which direction (the map is miscalibrated);
  * metrics or artefacts that keep misleading us;
  * any composition family that looks exhausted.

Then REWRITE IN PLACE the section of {paths['instruction']} that begins with the marker
'<!-- LEARNED PATTERNS -->' (create it at the end of the file if absent). Keep it under
{max_chars} characters -- it is a distillation, not a transcript. Over a multi-week campaign an
ever-growing prompt eventually crowds out the task itself, so old patterns that no longer earn
their place must be DROPPED, not accumulated.

Also append a dated round summary to {paths['memory']}."""
    ok, out = run_claude(prompt, timeout_min=timeout_min,
                         allowed_tools=["Read", "Edit", "Write"], quiet=True)
    return ok


# ============================================================================ REFLECTION (new)
def reflect(slots, timeout_min=8):
    """Peer review BEFORE the batch costs GPU time. Co-Scientist has this; I did not.

    My "Critic" only type-checks. Co-Scientist's Reflection agent is a scientific peer reviewer,
    and their ablation found that giving it SEARCH "effectively prevented the hallucination of
    seemingly novel but implausible hypotheses". A type-correct proposal can still be a bad
    experiment, and this is the only agent whose job is to say so before it runs.
    """
    prompt = f"""REFLECTION. Peer-review a proposed batch BEFORE it costs cluster time.
{budget_note(timeout_min, "1) the JSON review")}
proposed slots:
{json.dumps(slots, indent=1)[:5000]}

For each slot judge, as a reviewer would:
  * is the prediction FALSIFIABLE, and is it one the proposer could plausibly be wrong about?
    (A prediction that cannot fail teaches nothing.)
  * is the claim already SETTLED by the evidence or the reference model? (then it wastes a slot)
  * is the stated reason a real mechanism, or a restatement of the edit?
  * does the batch actually contain a control, and roughly 70/30 confirmatory/adversarial?

Reply with ONLY:
{{"batch_ok": true|false,
  "issues": [{{"slot": <int>, "problem": "<...>", "severity": "minor|serious"}}],
  "verdict": "<2 sentences on whether this batch is worth running as proposed>"}}"""
    ok, out = run_claude(prompt, timeout_min=timeout_min, allowed_tools=["Read"], quiet=True)
    return _first_json(out) or {"batch_ok": True, "issues": [],
                                "verdict": "reflection unavailable"}


# ============================================================================ EVOLUTION (new)
def evolve(winner_desc, ledger_summary, timeout_min=8):
    """REFINE the top-ranked rather than only truncating. Co-Scientist has this; I did not.

    Batch-and-truncate (Robin) prevents depth-first drift, but truncation ALONE means a winner
    is never improved -- the loop can only ever pick among what the enumerator happened to offer.
    Co-Scientist's Evolution agent refines the best by synthesis, analogy, literature grounding
    and simplification. The two are complementary: truncate ACROSS the batch, evolve WITHIN the
    winner.
    """
    prompt = f"""EVOLUTION. Refine the current best composition.
{budget_note(timeout_min, "1) the JSON proposal")}
current best:
{winner_desc}

evidence so far:
{ledger_summary}

Propose up to THREE refinements, each ONE legal edit, in the spirit of:
  * SIMPLIFY   -- can an operator be REMOVED with the phenomenon surviving? A simpler
                  sufficient mechanism is a stronger result than a complicated one.
  * SYNTHESISE -- combine with a mechanism that worked in a DIFFERENT cluster.
  * GROUND     -- make it closer to what the reference model actually does.

Reply with ONLY:
{{"refinements": [{{"edit_label": "<...>", "kind": "simplify|synthesise|ground",
                   "claim": "<falsifiable>", "predicted": "<...>", "why": "<...>"}}]}}"""
    ok, out = run_claude(prompt, timeout_min=timeout_min, allowed_tools=["Read"], quiet=True)
    return (_first_json(out) or {}).get("refinements", [])


# ============================================================================ JUDGE (2nd opinion)
def judge_pair(a, b, timeout_min=4):
    """An LLM judge over a PAIR, from the pictures -- Robin's tournament shape.

    Robin ranks by LLM judge because their outcome is not measurable. Ours is, so the METRIC
    BANK ranks first. This runs alongside it: where the judge and the metric DISAGREE, that
    disagreement is the eye/number divergence detector -- the failure that has cost this project
    the most, and the one neither paper faces because their measurement is external.
    """
    prompt = f"""JUDGE. Two runs. Which is closer to a clean, sustained TUBE (Okuda Fig. 5)?
{budget_note(timeout_min, "1) the JSON verdict")}
A ({a['name']}): caption = {a.get('caption', '')[:700]}
   metrics = {json.dumps(a.get('metrics', {}))[:400]}

B ({b['name']}): caption = {b.get('caption', '')[:700]}
   metrics = {json.dumps(b.get('metrics', {}))[:400]}

Judge from the DESCRIPTIONS first; use the numbers only to break a tie. A metric in this project
has already scored 9.30 on a small bud, so a number that contradicts the picture is the number
that is wrong.

Reply with ONLY: {{"winner": "A"|"B"|"tie", "why": "<one sentence>"}}"""
    ok, out = run_claude(prompt, timeout_min=timeout_min, allowed_tools=[], quiet=True)
    j = _first_json(out) or {}
    w = j.get("winner", "tie")
    return (1.0 if w == "A" else 0.0 if w == "B" else 0.5), j.get("why", "")


# ============================================================================ helpers
def _first_json(text):
    if not text:
        return None
    for opener, closer in (("{", "}"),):
        i = text.find(opener)
        while i >= 0:
            depth, j = 0, i
            while j < len(text):
                if text[j] == opener:
                    depth += 1
                elif text[j] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except Exception:
                            break
                j += 1
            i = text.find(opener, i + 1)
    return None


def _majority(vals):
    vals = [v for v in vals if v]
    return max(set(vals), key=vals.count) if vals else "unclear"


# ============================================================================ ESCALATION (new)
def request_operator(ledger_summary, map_summary, frontier_desc, exhausted_why, round_id,
                     timeout_min=8):
    """Ask what mechanism the language cannot express. The escalation path's only LLM call.

    This is invoked when the search has run out of moves that could teach it anything: every
    stage gate open, every proximity cluster frozen, or literally no legal edit left. At that
    point more compute cannot help -- the OPERATOR SET is the binding constraint, and the only
    useful output is a precise statement of what is missing.

    The load-bearing field is `why_inexpressible`. "I want a better growth operator" is a wish and
    is refused by `escalation.OperatorRequest`; "no operator EMITs a per-edge tension, and the
    cell set carries no edge-indexed state block to route into one" is a request -- it names the
    limit, so it can be acted on and it is evidence about the LANGUAGE, which is the thesis.
    """
    prompt = f"""OPERATOR REQUEST. The mechanism search has run out of moves.

{budget_note(timeout_min, "1) the JSON request")}
WHY WE ARE STUCK
{exhausted_why}

THE CAUSAL MAP SO FAR
{map_summary}

EVIDENCE LEDGER
{ledger_summary}

THE COMPOSITIONS ON THE FRONTIER
{frontier_desc}

You are NOT being asked for another composition -- there are none left worth running. You are
being asked what the OPERATOR LANGUAGE cannot say. Name ONE mechanism you would test next if the
type system allowed it, and be specific about the limit that blocks it.

A request is only useful if it names the LIMIT. Compare:
  WISH    "a better growth operator that makes tubes"
  REQUEST "no operator EMITs a per-EDGE tension; shape_energy_3d reads one scalar Lambda for the
           whole mesh and the cell set has no edge-indexed state block to route into it"

Reply with ONLY:
{{"mechanism": "<the biology, one sentence>",
  "why_inexpressible": "<WHICH type-system limit blocks it: a missing EMIT kind, a missing state
                        block, a set/kind mismatch, an absent contract, a routing slot that does
                        not exist. Name the operators and attributes involved.>",
  "wanted_for": "<the question it would answer / which cell of the map it would fill>",
  "proposed_contract": {{"contract": "<name>", "set": "vertex|cell|edge|field",
                        "kind": "structural|lateral|aggregate|rewire|field",
                        "family": "growth|mechanics|fields|topology|hierarchy",
                        "EMIT": "velocity|force|none",
                        "params": {{"<param>": "<role>"}}}},
  "acceptance_test": "<a concrete test on a SIMPLE geometry that would show the new operator
                       works, with a number in it>",
  "confidence": "high|medium|low"}}"""
    ok, out = run_claude(prompt, timeout_min=timeout_min, allowed_tools=["Read"], quiet=True)
    return _first_json(out)
