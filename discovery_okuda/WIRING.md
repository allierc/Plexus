# WIRING.md — every hand-off the loop makes, and who is on the other end

The fourth source of truth, beside `ROLES.md` (who exists), `PREMISES.md` (what a specimen must
satisfy) and `LOGIC.md` (what may be concluded from what). Each is a document a checker enforces.

**Why this one exists.** Every defect closed on 3 August was an *interface* defect, not a logic
error — components individually correct, disagreeing at the seam:

| the defect | the shape |
|---|---|
| `composition.json` written by `write_config`, never by recon | producer with no consumer |
| `run[:14]` read back as an identifier | a rendering used as identity |
| `batch_ok=None` while the verdict said REJECT | a decision with no effect |
| "the type system has already removed every unmet precondition" | a contract asserted in prose, enforced nowhere |
| `prompt.split("BREVITY")[0]` | 331 characters of a 16,093-character prompt reached the agent |

`roles.py --check` verifies that a declared **Sends to:** lands on a role that *exists*. It cannot
verify that the hand-off *happens*, because a hand-off is a file. That is this document's job.

## The rule

**An artifact with no reader is a defect.** Either wire a reader or stop writing it. Work the loop
does for nobody is worse than work it does not do: it looks like diligence in the record and costs
real time every round.

A new artifact must be added here, **with a reader**, or `wiring.py --check` fails. That is the
whole mechanism — the same one that keeps `ROLES.md` honest.

## Writers that are not code

- `(hand)` — written by a person, read by the loop. `PREMISES.md`, `okuda_corpus.md`, `ROLES.md`,
  `LOGIC.md`, this file.
- `(agent)` — written by an agent through its own tools, not by a `json.dump` anywhere.

## The campaign's own artifacts

| artifact | writer | readers |
|---|---|---|
| `proposal.json` | `(agent)` Proposer | `proposer.py` |
| `frontier.json` | `round.py` | `round.py`, `campaign_loop.py` |
| `state.json` | `supervisor.py` | `round.py`, `campaign_loop.py` |
| `hypotheses.jsonl` | `escalation.py` | `round.py`, `archivist.py` |
| `round_records.jsonl` | `round.py` | `archivist.py`, `campaign_loop.py` |
| `batch_refusals.jsonl` | `round.py` | `round.py` |
| `batch_attrition.jsonl` | `round.py` | `round.py` |
| `peer_review.jsonl` | `round.py` | `round.py` |
| `claims.jsonl` | `logic.py` | `logic.py`, `round.py` |
| `logic_report.jsonl` | `logic.py` | `round.py` |
| `lever_map.jsonl` | `lever_map.py` | `round.py`, `reel.py` |
| `supervisor.jsonl` | `supervisor.py` | `campaign_loop.py` |
| `archivist.jsonl` | `archivist.py` | `round.py` |
| `diagnoses.jsonl` | `diagnostician.py` | `round.py` |
| `holes.jsonl` | `collector.py` | `round.py` |
| `operator_requests.jsonl` | `metrologist.py` | `round.py` |
| `llm_timing.jsonl` | `llm.py` | `campaign_loop.py` |
| `campaign_loop.jsonl` | `campaign_loop.py` | `campaign_loop.py` |
| `memory.md` | `(agent)` Meta-review | `llm.py`, `proposer.py` |
| `analysis.md` | `(agent)` several | `llm.py` |
| `knowledge.md` | `(agent)` Interpreter | `grounder.py`, `proposer.py` |
| `causal_descriptions.md` | `(agent)` Interpreter | `round.py` |
| `operator_backlog.md` | `metrologist.py` | `round.py` |
| `instruction.md` | `campaign_loop.py` | `llm.py` |
| `user_input.md` | `(hand)` | `llm.py` |
| `EXTERNAL_AUDIT.md` | `(hand)` | `(hand)` |
| `EXTERNAL_AUDIT_1.md` | `(hand)` | `(hand)` |
| `campaign.log` | `campaign_loop.py` | `(hand)` |
| `trace.log` | `round.py` | `(hand)` |
| `records.jsonl` | `collector.py` | `round.py`, `archivist.py`, `campaign_loop.py` |
| `reservoir.jsonl` | `collector.py` | `round.py`, `campaign_loop.py` |
| `map.jsonl` | `round.py` | `lever_map.py`, `reel.py`, `campaign_loop.py` |
| `lever_map.md` | `round.py` | `lever_map.py`, `campaign_loop.py` |
| `q_quarantine.jsonl` | `round.py` | `round.py`, `campaign_loop.py` |
| `llm_usage.jsonl` | `llm.py` | `llm.py`, `campaign_loop.py` |
| `campaign.json` | `control.py` | `campaign_loop.py` |
| `_submit.log` | `cluster.py` | `cluster.py` |

## Per-run artifacts

| artifact | writer | readers |
|---|---|---|
| `diag.json` | `run_one.py` | `round.py`, `archivist.py`, `biologist.py` |
| `metrics.json` | `run_one.py` | `round.py`, `diagnostician.py` |
| `progress.json` | `run_one.py` | `cluster.py` |
| `spec_run.yaml` | `run_one.py` | `round.py`, `translate.py` |
| `composition.json` | `translate.py` | `round.py`, `logic.py` |
| `description.txt` | `caption_wave.py` | `llm_agents.py` |

## Known open edges

These are declared **so they are visible**, not because they are acceptable. Each is a Phase 8 item.

- `composition.json` is written by `write_config` only. A **recon** run copies its spec verbatim
  and so carries none, which is why twelve replays could not seed the frontier and round 2 began
  from reference recipes as though round 1 had never happened.
- `batch_attrition.jsonl` was added on 3 August and nothing reads it yet. The Supervisor should:
  a batch that repeatedly under-delivers is evidence about the Proposer, not about the biology.
