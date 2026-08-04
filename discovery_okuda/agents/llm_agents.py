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
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm                                                              # noqa: E402
import term as _T                                                       # noqa: E402
from llm import budget_note, ensure_files, read_file, run_agent         # noqa: E402

CAMP = llm.CAMPAIGN

# HOW MANY READERS PER RUN. One. See analyse() for why the x8 argument does not transfer, and
# ROLES.md for what is given up: with a single reader, nothing records that a label was a close
# call. Raising this is the whole change.
N_READERS = 1

# EVERY agent call in this file goes through `run_agent(role, ..., ledger=ledger)`, never
# through `run_claude` directly. That was the defect: run_agent() consulted the BudgetLedger and
# stopwatch-ed the call, but no call site used it, so a round of ~25 model calls reported
# `[llm] {'calls': 0, 'round_min': 0.0, 'total_min': 0.0}` and the declared 25-min per-round
# ceiling had never constrained anything. `ledger` is threaded down from round.py; passing None
# still works (ad-hoc use) but then the call is simply unaccounted, which run_claude reports.
#
# The timeouts and tool sets below are UNCHANGED from the bypassing versions on purpose: this
# phase is measurement only, and moving a number would destroy the baseline being measured.


# ============================================================================ 7. ANALYST x N

# ============================================================================ BREVITY
# WALL CLOCK IS GENERATION. Measured 2026-08-01 across every call the ledger holds: the token
# RATE is near-identical for every agent (64-77 tok/s), so an agent is slow exactly in proportion
# to how much it writes. A proposer turn emits ~2100 output tokens and takes 28 s; an analyst
# turn emits ~470 and takes 7. There is nothing else to reclaim -- the API is 95-99% of wall time,
# so no queueing or tool overhead is hiding in there.
#
# The prompts already said "two sentences" in the places that had a limit. What had none were the
# LISTS: the reviewer emitted six issues of 38-64 words each. So the limits below are on the
# things that grow without bound, and they are limits on PROSE, not on content -- a flag with its
# reason in twenty words is worth the same as the same flag in sixty, and arrives four times
# sooner. What must never be shortened is a NUMBER: a threshold, a metric name, a citation.
def _admitted():
    """The admissible metrics, from the registry. Never a hand-written list in a prompt."""
    try:
        from predict import admitted_block
        return admitted_block(new_since=NEW_INSTRUMENTS)
    except Exception:
        return "  (metric registry unavailable)"


# Announced to every role that reads numbers, until the campaign after they land.
NEW_INSTRUMENTS = ("n_spots_final", "wavelength_cells_final", "spot_spacing_cells_final",
                   "spot_frac_final")

from llm import BREVITY  # noqa: F401  (defined in llm.py so run_agent can apply it)


def analyse(run_name, out_dir, n=N_READERS, timeout_min=6, ledger=None, parallel=True):
    """Read ONE run and LABEL it. N is one number, and it is 1 (ROLES.md).

    THIS IS NOT ROBIN'S x8, and the earlier version of this docstring claimed it was. Robin's
    Finch WRITES THE ANALYSIS CODE -- it chooses the flow-cytometry gating and the RNA-seq
    filters, so eight trajectories genuinely produce different NUMBERS and the consensus is over
    measurements. Nothing here measures: by the time this runs, diag.json, metrics.npz, the curve
    shapes and the strip were computed by instruments the Metrologist certifies against known
    answers, and every reader would see IDENTICAL numbers.

    What a reader produces is a LABEL -- phenotype, forced_or_grown, eye_vs_number, a concern --
    and those are judgements over images and a caption. Running N of them would measure PHENOTYPE
    AMBIGUITY, which is a real quantity but a much smaller prize than the one the x8 argument was
    imported for. Settled at one; raising N is this one number.
    """
    diag = os.path.join(out_dir, "diag.json")
    desc = os.path.join(out_dir, "description.txt")
    mech = os.path.join(out_dir, "mechanics.png")
    # THE TRAJECTORIES. Until now an Analyst was handed ENDPOINTS (diag.json, labelled "metrics")
    # and never learned the per-frame curves existed -- `metrics.png` was referenced nowhere in the
    # codebase, drawn for every run since the beginning and read by nobody. A scalar cannot tell a
    # plateau at 2.7 from a spike to 2.7 at the moment the mesh tears, and the campaign ranks on
    # exactly that number. The SHAPE of each curve is classified deterministically (arithmetic, not
    # judgement) and handed over as text, so the reading starts from what actually happened over
    # time rather than from where the run happened to stop.
    shapes = ""
    try:
        import curve_shape as _CS
        _rep = _CS.report(out_dir, write=True)
        shapes = _CS.summarise(_rep) or ""
    except Exception as e:
        shapes = (f"(trajectory shapes unavailable: {type(e).__name__}: {str(e)[:70]} -- read "
                  f"metrics.png yourself and say so in your verdict)")
    # WHAT THE BIOLOGIST FOUND. It has been running on every run since it was written, and its
    # verdict has been going to a terminal and to a field of diag.json that no prompt mentioned.
    # On round 2 it broke five premises on r002c_00 -- the activator had decayed to NaN, the
    # chemistry was extinct -- and an Analyst read that run's numbers and named a phenotype.
    # Handing an analyst a summary without telling it the specimen is broken is asking for a
    # confident reading of a configuration error.
    premises = ""
    try:
        import biologist as _B
        _d = json.load(open(os.path.join(out_dir, "diag.json")))
        _res = [_B.R(p["id"], p["tier"], p["premise"], p["status"], p["detail"], p.get("measured"))
                for p in _d.get("premises") or []]
        premises = _B.brief(_res, run_name) if _res else ""
    except Exception as e:
        premises = f"(premise verdict unavailable: {type(e).__name__}: {str(e)[:70]})"
    # INDEPENDENT BY DESIGN, SO RUN THEM TOGETHER. Three readings of one run exist precisely
    # because a single LLM reading is not reproducible -- they must not influence one another,
    # and no analyst's prompt depends on another's answer. They were sequential only because
    # this was written as a `for` loop, which on a five-run round spends an hour of allowance
    # waiting. Concurrency changes the wall-clock and not one thing about the experiment.
    prompts = []
    for i in range(n):
        prompt = f"""READER {i + 1} of {n}. Read ONE simulation run and report what happened.

Work independently. Do not try to agree with anyone.
{budget_note(timeout_min, "1) your JSON verdict  2) nothing else")}
  final numbers : {diag}      (ENDPOINTS ONLY -- see the trajectories below before trusting them)
  VLM caption   : {desc}
  mechanics     : {mech}   (force | pressure body-vs-protruding | tension | migration)
  strip         : {os.path.join(out_dir, 'strip.png')}
  curves        : {os.path.join(out_dir, 'metrics.png')}   (every metric vs frame)

IS THIS EVEN A TISSUE? -- the Biologist ran the premises against this run before you read it:
{premises or "  (no premise check on this run)"}

HOW EACH MEASUREMENT BEHAVED OVER TIME -- classified automatically, not by anyone's judgement:
{shapes or "  (no trajectory recorded)"}

ALSO LOOK AT THE CELLS THEMSELVES, not only the tissue's outline. The cross-section row of the
strip shows individual cells; say whether they look uniform or whether some are visibly stretched,
thin, compressed or otherwise distorted, and WHERE (in the protrusion? at its base? scattered?).
`shape_idx_*` in the numbers is the measurement: the cells' preferred value here is about 3.5, and
above 3.81 a tissue stops behaving like a solid and starts to flow. A tissue that is flowing can
be pushed into a tube but cannot hold one, so this bears directly on forced-versus-grown.
Nobody looked at cell shape for months because no number recorded it; do not let that continue.

Read those shapes BEFORE the final numbers, and say in your verdict when the two disagree:
  peaked   the best moment was mid-run; the final value reports the DECAY, not the phenomenon
  rising   still climbing when the run stopped -- the run was cut short, not finished
  pinned   held against a hard limit (a buffer), not converged. Not a result about biology
  exploded a late blow-up -- suspect the mesh, not the tissue
A run whose headline number came from an `exploded` or `pinned` curve is not the run its summary
claims it is. Saying so is more useful than a phenotype label.

{_admitted()}
Informative but not scored: mech_p_ratio (~3 = FORCED protrusion, ~1 = growth-driven equilibrium).

Reply with ONLY this JSON, nothing else:
{{"phenotype": "sphere|bud|spike|tube|branching|undulation|exploded|degenerate",
  "confidence": 0.0-1.0,
  "forced_or_grown": "forced|grown|unclear",
  "evidence": "<2 sentences citing SPECIFIC numbers or what the caption says>",
  "eye_vs_number": "agree|disagree",
  "specimen": "sound|compromised",
  "concern": "<anything that looks like an artefact rather than physics, or empty>"}}

`specimen` is `compromised` whenever a premise above is broken in a way that touches what you
just claimed -- an extinct chemistry beneath a "pattern" reading, a stretched sheet beneath a
"tube". A field forces the question to be answered rather than passed over."""
        prompts.append(prompt)

    def _one(prompt):
        ok, out = run_agent("reader", prompt, ledger=ledger, timeout_min=timeout_min,
                            quiet=True)
        return _first_json(out) or {"phenotype": "unreadable", "confidence": 0.0}

    if parallel and n > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n) as ex:
            reads = list(ex.map(_one, prompts))
    else:
        reads = [_one(p) for p in prompts]

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
# A CLAIM THE PICTURE CANNOT ADJUDICATE. A recon slot claims "re-measure X under the current
# instruments" -- procedural, not morphological -- and asking a vision model whether the movie
# supports it can only produce a false CONTRADICTS: no image shows that a run is a
# re-measurement. Three of eight runs were flagged that way in one round, each with a reason
# like "no visual evidence this is a re-measurement" or "does not reference the measurement
# parameter cfl". The eye-check was right and the question was wrong.
_UNGATEABLE = re.compile(r"^\s*(re-?measure|replay|control|baseline|unstated)\b", re.I)


def _watch_describe_only(run_name, desc, timeout_min, ledger):
    """Describe, do not adjudicate. For a claim no picture can settle.

    The description is still the most useful thing the eye-check produces -- it is the only
    role that looks at SHAPE -- so it is still asked for. What it is not asked for is a verdict
    on a proposition that is not about the image.
    """
    prompt = f"""WATCHER. Describe what this movie shows. There is NO claim to check: this run is
a replay or a control, and whether it is one cannot be seen in a picture.

{budget_note(timeout_min, "1) the JSON")}
This is what a vision model saw:
---
{desc[:2500]}
---
Reply with ONLY: {{"seen": "<what it shows, 6 words>",
                  "describe": "<FOUR SENTENCES: the shape and how it changes, whether anything
                   protrudes, what the colour is doing, and anything that looks like an artefact
                   rather than tissue>",
                  "headline": "<at most 90 characters: the ONE thing worth knowing>"}}"""
    ok, out = run_agent("watcher", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=[], quiet=True)
    j = _first_json(out) or {}
    return {"watcher_verdict": "described", "watcher_blocks": False,
            "watcher_seen": j.get("seen", ""), "watcher_why": "",
            "watcher_describe": j.get("describe", ""), "watcher_headline": j.get("headline", "")}


def watch(run_name, out_dir, expected_phenotype, timeout_min=5, ledger=None, gate=None):
    """Compare what the movie SHOWS against what the composition CLAIMED, and gate promotion.

    The Watcher's eye already exists (the VLM captions every run). This is the missing half: the
    VETO. Without it the caption is produced and ignored, and the loop can be fooled exactly as I
    was -- by a metric scoring a tube on a bud.
    """
    desc = read_file(os.path.join(out_dir, "description.txt"), 4000)
    if not desc or desc.startswith("UNAVAILABLE"):
        return {"watcher_verdict": "no_caption", "watcher_blocks": False,
                "watcher_why": "no caption was produced -- cannot gate, recorded as such"}
    # gate=None means decide from the claim; an explicit gate= from the call site wins.
    if gate is None:
        gate = not _UNGATEABLE.match(str(expected_phenotype or ""))
    if not gate:
        return _watch_describe_only(run_name, desc, timeout_min, ledger)
    prompt = f"""WATCHER. A run CLAIMED to produce: {expected_phenotype!r}

This is what a vision model saw in the rendered movie:
---
{desc[:2500]}
---
{budget_note(timeout_min, "1) the JSON verdict")}
Does the description support the claim? Be strict: 'a small bud' does NOT support 'a tube', and
'a chaotic cluster of shards' does NOT support any clean morphology.

Reply with ONLY: {{"supports": true|false, "seen": "<what it actually shows, 6 words>",
                  "why": "<one sentence>",
                  "describe": "<FOUR SENTENCES on what the movie actually shows: the shape and
                   how it changes, whether anything protrudes, what the colour is doing, and
                   anything that looks like an artefact rather than tissue>",
                  "headline": "<at most 90 characters: the ONE thing a person watching the
                   terminal should know about this run>"}}"""
    ok, out = run_agent("watcher", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=[], quiet=True)
    j = _first_json(out) or {}
    # SILENCE IS NOT AGREEMENT. `j.get("supports", True)` made an unparsed reply -- a caption the
    # model never produced, a JSON object that never appeared, a timeout -- read as the eye
    # AGREEING with the numbers. That is the one direction this role must never fail in: it is the
    # only channel in the loop that looks at SHAPE rather than at number, so a silent yes removes
    # the only thing that could contradict the arithmetic, and removes it invisibly.
    #
    # Three states, not two. `supports` absent is UNKNOWN: it does not support and it does not
    # block, and it is recorded as unknown so the record shows the eye was not heard from.
    _raw = j.get("supports", None)
    heard = _raw is not None
    supports = bool(_raw) if heard else False
    return {"watcher_verdict": ("supports" if supports else "CONTRADICTS") if heard
            else "NOT SEEN -- the eye returned nothing parseable",
            "watcher_seen": j.get("seen", ""), "watcher_why": j.get("why", ""),
            "watcher_describe": j.get("describe", ""),
            "watcher_headline": j.get("headline", ""),
            "watcher_heard": heard,
            # An unheard eye must not veto either: it blocks only when it actually said so.
            "watcher_blocks": heard and not supports}


# ============================================================================ 9. INTERPRETER
def interpret(comp_hash, region, edit, summary, analyst, out_path, timeout_min=8, ledger=None):
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
    prompt += _T.HEADLINE_ASK
    ok, out = run_agent("interpreter", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read", "Edit", "Write"], quiet=True)
    # RETURN WHAT IT SAID, not merely whether it ran. `return ok` meant the causal record -- the
    # one sentence explaining why a composition did what it did -- reached a file and never the
    # terminal, so nobody watching a round could tell whether it was reasoning or reciting.
    return ok, _T.headline(out)


# ============================================================================ 10. META-REVIEW
def _templates():
    """The memory.md shape, read from the template file so there is ONE definition of it."""
    try:
        import templates as T
        return T.prompt_block()
    except Exception as e:
        return f"(templates unavailable: {type(e).__name__})"


def meta_review(round_id, timeout_min=10, max_chars=4000, ledger=None, runs=()):
    """Distil recurring patterns and APPEND them to the Proposer's instructions.

    Co-Scientist: "feedback applicable to all agents, which is simply appended to their prompts
    in the next iteration -- feedback propagation and learning without back-propagation".

    The bounded-growth rule is ours: over a multi-week run an append-only prompt grows without
    limit and eventually crowds out the task. The distilled section is CAPPED and rewritten in
    place each time, so it stays a summary rather than becoming a transcript.
    """
    paths = ensure_files()
    # THE BIOLOGIST'S ROUND TALLY. One run breaking a premise is that run's problem; the same
    # premise breaking across most of a round is the campaign proposing a family of compositions
    # that cannot hold a tissue, and THAT is a pattern to carry forward -- which is this agent's
    # entire job. The per-run verdicts already existed and were read by nobody.
    tally = ""
    try:
        import biologist as _B
        tally = _B.round_tally(list(runs)) if runs else ""
    except Exception as e:
        tally = f"(premise tally unavailable: {type(e).__name__}: {str(e)[:70]})"
    prompt = f"""META-REVIEW after round {round_id}.
{budget_note(timeout_min, "1) the LEARNED PATTERNS section in instruction.md  2) memory.md")}
Read:
  analysis log : {paths['analysis']}
  memory       : {paths['memory']}
  knowledge    : {os.path.join(CAMP, 'knowledge.md')}

WHAT THE BIOLOGIST FOUND ACROSS THIS ROUND -- premises are the things we take as known about
tissue, written in PREMISES.md. A premise broken in most of a round is the strongest pattern
available to you, and it names the edit family to stop proposing:
{tally or "  (no premise tally for this round)"}

Find the patterns that RECUR across rounds -- not a summary of what happened, but what a
proposer should know before choosing the next edit. Especially:
  * edits that repeatedly fail, and the reason they fail (so they are never proposed again);
  * predictions that were repeatedly WRONG, and in which direction (the map is miscalibrated);
  * metrics or artefacts that keep misleading us;
  * any composition family that looks exhausted.

Then do TWO things, and nothing else.

1) REWRITE IN PLACE the section of {paths['instruction']} that begins with the marker
'<!-- LEARNED PATTERNS -->' (create it at the end of the file if absent). Keep it under
{max_chars} characters -- it is a distillation, not a transcript. Over a multi-week campaign an
ever-growing prompt eventually crowds out the task itself, so old patterns that no longer earn
their place must be DROPPED, not accumulated. This is the prompt write-back, and it is the
mechanism by which this loop learns at all.

2) REWRITE {paths['memory']} IN PLACE. It is YOURS now -- the Proposer used to write it, which
put the agent under evaluation in charge of its own memory. It is a STATE DOCUMENT, not a log:
the named sections below and no others, each corrected rather than appended to. The history is
in analysis.md and hypotheses.jsonl, which are append-only and are NOT yours to touch. A line
earns its place here only if a LATER round needs it and could not re-derive it. memory.md has
been used as a log and reached 1904 words across six appended blocks; that is the failure this
shape prevents.

{_templates()}"""
    prompt += _T.HEADLINE_ASK
    ok, out = run_agent("meta_review", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read", "Edit", "Write"], quiet=True)
    return ok, _T.headline(out)


# ============================================================================ REFLECTION (new)
def reflect(slots, timeout_min=8, ledger=None):
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
  "issues": [{{"slot": <int>, "problem": "<the flaw, <=25 words>", "severity": "minor|serious"}}],
  "verdict": "<<=40 words on whether this batch is worth running as proposed>"}}

AT MOST THREE ISSUES, the three that would most change what is learned. A fourth costs a minute
of wall clock and changes nothing -- if everything is flagged, nothing is."""
    ok, out = run_agent("reflection", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read"], quiet=True)
    return _first_json(out) or {"batch_ok": True, "issues": [],
                                "verdict": "reflection unavailable"}


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
                     timeout_min=8, ledger=None):
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
    ok, out = run_agent("operator_request", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read"], quiet=True)
    return _first_json(out)
