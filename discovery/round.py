#!/usr/bin/env python
"""round -- ONE round of the agentic discovery loop, end to end.

    propose  ->  critic  ->  configs  ->  hypotheses  ->  L4  ->  score  ->  rank
             ->  truncate ->  supervisor  ->  knowledge.md

Deterministic control flow (Robin de-agentified their orchestrator for the same reason).
Language models are called only where judgement is required. Ranking is by MEASURING, using
only the metrics the instrument gate admitted.

    python round.py --frames 900 --batch 8            # run one round
    python round.py --status                          # supervisor state
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

import cluster                                                          # noqa: E402
import translate as T                                                   # noqa: E402
from composition_space import reference_recipes, seed                   # noqa: E402
from control import CampaignConfig, Supervisor, propose_batch, score_run, truncate, meets_success  # noqa: E402
from hypothesis import Hypothesis                                       # noqa: E402
from metrologist import Certification                                   # noqa: E402
from run_record import comp_hash                                        # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
CAMPAIGN = os.path.join(HERE, "campaign")
FRONTIER = os.path.join(CAMPAIGN, "frontier.json")


# --------------------------------------------------------------------------- frontier
def load_frontier():
    """Graphs to expand from. Round 1 seeds from the reference recipes + the bare substrate."""
    if os.path.exists(FRONTIER):
        import composition_space as CS
        raw = json.load(open(FRONTIER))
        out = []
        for r in raw:
            g = CS.CompositionGraph(ops=r["ops"], conns=r["conns"], params=r["params"])
            out.append(g)
        if out:
            return out
    return [seed("substrate")] + list(reference_recipes().values())


def save_frontier(graphs):
    os.makedirs(CAMPAIGN, exist_ok=True)
    json.dump([{"ops": g.ops, "conns": g.conns, "params": g.params} for g in graphs],
              open(FRONTIER, "w"), indent=1)


# --------------------------------------------------------------------------- the round
def run_round(frames=900, batch=8, dry=False):
    cert = Certification(os.path.join(HERE, "_metrology"))
    ok, why = cert.may_admit()
    if not ok:
        print(f"[round] ADMISSION GATE CLOSED -- refusing to run.\n  {why}")
        return 2

    cfg = CampaignConfig(batch=batch, keep_truncate=max(2, batch // 3))
    sup = Supervisor(cfg, CAMPAIGN)
    rng = np.random.default_rng(sup.round + 1)

    print("=" * 92)
    print(f"ROUND {sup.round + 1}   campaign={cfg.name}")
    print("=" * 92)
    print(f"objective: {cfg.objective}\n")

    # ---------------------------------------------------------------- propose + critic
    frontier = load_frontier()
    cands = propose_batch(frontier, cfg, sup.prox, rng)
    print(f"[propose] {len(cands)} candidates survive the critic "
          f"(ill-typed / unmet precondition / duplicate are rejected free)")
    if not cands:
        print("[round] no legal candidates -- escalating")
        print(json.dumps(sup.escalate(), indent=1))
        return 1

    # ---------------------------------------------------------------- hypotheses FIRST
    # A candidate is not run until its prediction is recorded.
    names, posed = [], []
    for i, (child, lbl, parent) in enumerate(cands):
        nm = f"r{sup.round + 1:02d}_{i:02d}_{comp_hash(child)[1:7]}"
        T.write_config(child, nm, frames=frames)
        names.append(nm)
        has_ex = "extrude" in child.op_names()
        # the intent split: adding a mechanism is confirmatory, removing one is adversarial
        intent = "adversarial" if lbl.startswith("-") else "confirmatory"
        pred = ("protr_peak >= 3.0" if intent == "confirmatory" else "protr_peak < 3.0")
        h = Hypothesis(
            hid=f"R{sup.round + 1}.{i}.{comp_hash(child)[1:7]}", comp_hash=comp_hash(child),
            parent_hash=comp_hash(parent), edit=lbl, intent=intent,
            claim=f"edit `{lbl}` on {parent.name_region()} yields {child.name_region()}",
            metric="protr_peak", predicted=pred,
            rationale=("gate-admissible metrics only: protr_peak tau=+1.00, n_tubes tau=+1.00. "
                       + ("no extrude node -- tests the growth-driven route" if not has_ex
                          else "carries the extrude forcing node")),
            grounding=["okuda.pdf p.4 (activator-inhibitor drives growth regions)"],
            round_id=sup.round + 1)
        if not dry:                      # a dry run must not consume hypothesis ids
            sup.reg.pose(h)
        posed.append((nm, child, h))
        print(f"  {nm}  {lbl:34} {intent:13} predict {pred}")

    if dry:
        print("\n[round] --dry: configs + hypotheses written, nothing submitted")
        return 0

    # ---------------------------------------------------------------- run on L4
    print(f"\n[run] submitting {len(names)} to the L4 partition")
    ids = cluster.submit(names, frames=frames, do_q=True, campaign=f"round{sup.round + 1}")
    if not ids:
        print("[round] submission did not land -- aborting rather than scoring nothing")
        return 1
    cluster.wait_for_ids(ids, poll=60)

    # ---------------------------------------------------------------- score
    results, rows = [], []
    for nm, child, h in posed:
        d = os.path.join(LOG, nm, "diag.json")
        if not os.path.exists(d):
            sup.reg.resolve(h.hid, {}, "inconclusive", note="run produced no diag.json")
            continue
        summ = json.load(open(d)).get("summary", {})
        if not summ.get("valid_evidence", True):
            sup.reg.resolve(h.hid, summ, "inconclusive",
                            note=f"NOT EVIDENCE: inert={summ.get('inert_operators')} "
                                 f"saturated={summ.get('saturated')}")
            continue
        sc = score_run(summ, cfg)
        got = float(summ.get("protr_peak", 0.0))
        outcome = ("confirmed" if (got >= 3.0) == (h.predicted.startswith("protr_peak >="))
                   else "refuted")
        sup.reg.resolve(h.hid, summ, outcome, run_ids=[nm])
        results.append((child, summ, h.hid))
        rows.append((nm, child, summ, sc, outcome, h))

    if not results:
        print("[round] no admissible evidence this round")
        print(json.dumps(sup.observe([]), indent=1))
        return 1

    # ---------------------------------------------------------------- rank + truncate
    rows.sort(key=lambda r: -r[3])
    kept, dropped = truncate(rows, cfg.keep_truncate)
    print(f"\n[rank] by MEASURED score (gate-admissible metrics only)")
    for nm, g, s, sc, oc, h in rows:
        star = "KEEP" if (nm, g, s, sc, oc, h) in kept else "drop"
        print(f"  [{star}] {nm}  score {sc:6.2f}  protr_peak {s.get('protr_peak', 0):5.2f} "
              f"final {s.get('protr_final', 0):5.2f} Q {s.get('Q_protr_after_relax', '-')}"
              f"  n_tubes {s.get('ta_n_tubes_final', 0):.0f}  [{oc}]"
              + ("  🔥 SURPRISE" if h.is_surprise else ""))

    # ---------------------------------------------------------------- supervisor + ledger
    rep = sup.observe(results)
    print(f"\n[supervisor] {json.dumps({k: v for k, v in rep.items() if k != 'mix_why'})}")
    print(f"  mix: {rep['mix_why']}")
    sup.reg.render_knowledge(os.path.join(CAMPAIGN, "knowledge.md"),
                             ledger={"kept": len(kept), "dropped": len(dropped)},
                             round_id=sup.round)
    save_frontier([g for _, g, _, _, _, _ in kept])

    winners = [nm for nm, g, s, _, _, _ in kept
               if meets_success(s, cfg, "extrude" in g.op_names())]
    if winners:
        print(f"\n  🎯 OBJECTIVE MET by: {winners}")
    if rep["stop"]:
        print(f"\n[round] TERMINAL: {rep['reason']}")
    elif "ESCALATE" in rep["reason"]:
        print(f"\n[round] {rep['reason']}")
        print(json.dumps(sup.escalate(), indent=1))
    print(f"\n  knowledge -> {os.path.relpath(os.path.join(CAMPAIGN, 'knowledge.md'), ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        sup = Supervisor(CampaignConfig(), CAMPAIGN)
        print(json.dumps({"round": sup.round, "spent": sup.spent,
                          "clusters": sup.prox.summary()}, indent=1, default=str))
        sys.exit(0)
    sys.exit(run_round(frames=a.frames, batch=a.batch, dry=a.dry))
