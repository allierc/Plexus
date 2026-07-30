#!/usr/bin/env python
"""round -- ONE round of the agentic discovery loop, with the real agents.

  Grounder -> Proposer(LLM) -> Critic -> Reflection(LLM) -> hypotheses -> L4
           -> Analyst xN(LLM) -> Watcher veto(LLM) -> Referee + Judge(LLM)
           -> truncate -> Interpreter(LLM) -> LeverMap -> Supervisor -> Meta-review(LLM)

TWO MODES, because the campaign needs both and I had only built one:

  --mode composition   LOOP I. Search over mechanism structure. comp_hash EXCLUDES theta, so a
                       retune provably cannot pose as a new hypothesis.
  --mode theta         LOOP II. Sweep ONE parameter of ONE fixed composition. The composition
                       hash is CONSTANT across the sweep by construction -- that is the point,
                       not a defect. Verdicts land in the map as parameter-sensitivity, never as
                       a new mechanism.

Why the second mode had to exist: the first question we wanted to ask the loop was "does vcap
explain why the clock-re-anchored tube is lost?" (Metrologist D1d). That is a PARAMETER question.
The composition search structurally cannot ask it -- every vcap value is the same hypothesis --
so a loop with only Loop I would have answered it by accident or not at all.

    python round.py --mode theta --param divide_3d0.vcap --base round_40_mc8 --values 0.0,1.5,3.0
    python round.py --mode composition --batch 8
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

import cluster                                                            # noqa: E402
import critic as C                                                        # noqa: E402
import translate as T                                                     # noqa: E402
from composition_space import reference_recipes, seed                     # noqa: E402
from control import CampaignConfig, Supervisor, score_run, truncate       # noqa: E402
from hypothesis import Hypothesis                                         # noqa: E402
from lever_map import LeverMap                                            # noqa: E402
from run_record import comp_hash                                          # noqa: E402

import llm                                                                # noqa: E402
import llm_agents as A                                                    # noqa: E402
import proposer as P                                                      # noqa: E402
from metrologist import Certification                                     # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
CAMP = os.path.join(HERE, "campaign")
FRONTIER = os.path.join(CAMP, "frontier.json")
MAP = os.path.join(CAMP, "lever_map.jsonl")


# --------------------------------------------------------------------------- frontier
def load_frontier():
    if os.path.exists(FRONTIER):
        import composition_space as CS
        raw = json.load(open(FRONTIER))
        out = [CS.CompositionGraph(ops=r["ops"], conns=r["conns"], params=r["params"])
               for r in raw]
        if out:
            return out
    return [seed("substrate")] + list(reference_recipes().values())


def save_frontier(graphs):
    os.makedirs(CAMP, exist_ok=True)
    json.dump([{"ops": g.ops, "conns": g.conns, "params": g.params} for g in graphs],
              open(FRONTIER, "w"), indent=1)


def seen_hashes(sup):
    return {o["comp"] for o in LeverMap(MAP).obs} | set(sup.prox.clusters and
                                                        [h for c in sup.prox.clusters.values()
                                                         for h in c["members"]] or [])


# --------------------------------------------------------------------------- LOOP I batch
def build_composition_batch(sup, cfg, n_slots, ledger):
    """Proposer(LLM) -> Critic -> Reflection(LLM). Returns [(graph, label, hyp_fields)]."""
    frontier = load_frontier()
    seen = seen_hashes(sup)
    lm = LeverMap(MAP)
    ledger_summary = _ledger_summary(sup, lm)

    ok, slots = P.propose(frontier, cfg, sup.prox, ledger_summary, sup.round + 1,
                          n_slots=n_slots)
    if not slots:
        print("[round] the Proposer produced no usable proposal -- NOT falling back to random. "
              "A round with no reasoned proposal is a failed round, not a random one.")
        return []

    out, rejected = [], []
    for i, sl in enumerate(slots):
        pi = int(sl.get("parent_index", 0))
        if not (0 <= pi < len(frontier)):
            rejected.append((i, f"parent_index {pi} out of range"))
            continue
        parent = frontier[pi]
        if sl.get("intent") == "control" or sl.get("edit") in (None, "null"):
            g, lbl = parent, "control (parent unchanged)"
        else:
            e = sl["edit"]
            e = tuple(e["edit"]) if isinstance(e, dict) and "edit" in e else tuple(e)
            try:
                g, _ = parent.apply(e)
            except Exception as ex:
                rejected.append((i, f"edit not applicable: {ex}"))
                continue
            lbl = sl.get("label") or str(e)
        adm, rej = C.admit(g, seen if sl.get("intent") != "control" else ())
        if not adm:
            rejected.append((i, f"CRITIC: {rej}"))
            continue
        out.append((g, lbl, sl))
    for i, why in rejected:
        print(f"  [critic] slot {i} rejected -- {why}")

    review = A.reflect([{k: v for k, v in s.items() if k != "edit"} for _, _, s in out])
    print(f"  [reflection] batch_ok={review.get('batch_ok')} :: {review.get('verdict','')[:200]}")
    for iss in review.get("issues", []):
        print(f"     slot {iss.get('slot')}: [{iss.get('severity')}] {iss.get('problem')}")
    if review.get("batch_ok") is False and any(
            i.get("severity") == "serious" for i in review.get("issues", [])):
        print("  [reflection] SERIOUS issues -- the batch is not run as proposed.")
        return []
    return out


# --------------------------------------------------------------------------- LOOP II batch
def build_theta_batch(base_name, param, values, n_slots, predictions=None):
    """A parameter sweep of ONE composition. The hash is constant BY CONSTRUCTION.

    CORRECTION (Cedric): I had conflated two different things. Composition IDENTITY excludes
    theta, so a retune cannot count as a new MECHANISM -- that rule is right and stays. But I
    wrongly inferred that a parameter cannot carry a HYPOTHESIS. It obviously can:

        "if I raise vcap to 3.0, tip cells divide sooner, so the tube shortens"

    is falsifiable, predictive, and exactly the kind of claim one can be wrong about. Stamping
    these points `unknown -- sensitivity sweep` and excluding them from the surprise rate threw
    away the most informative part of the round. A parameter hypothesis now carries a real
    prediction and COUNTS toward surprise, like any other.
    """
    presets = T.load_presets()
    if base_name in presets:
        base = T.from_preset(presets[base_name])
    else:
        base = reference_recipes().get(base_name) or seed("substrate")
    h0 = comp_hash(base)
    preds = predictions or {}
    out = []
    base_v = base.params.get(param)
    for v in values[:n_slots]:
        g = base.with_params({**base.params, param: v})
        assert comp_hash(g) == h0, "theta must not change composition identity"
        # a real, falsifiable prediction per point -- supplied, or derived from the mechanism
        pred = preds.get(str(v))
        if pred is None:
            pred = _theta_prediction(param, v, base_v)
        intent = "confirmatory" if (base_v is None or abs(v - base_v) <= 1e-9
                                    or pred.startswith("protr_peak >=")) else "adversarial"
        out.append((g, f"theta {param}={v}",
                    {"intent": intent, "metric": "protr_peak", "predicted": pred,
                     "claim": f"{param}={v} on {base_name}: {pred}",
                     "why": "Loop II: a parameter hypothesis. Composition identity is unchanged "
                            "by construction, so this cannot pose as a new mechanism -- but it "
                            "is a prediction that can be wrong, and it counts as one."}))
    print(f"  [theta] {len(out)} points of `{param}` on {base_name} (comp {h0}, constant)")
    return out


# --------------------------------------------------------------------------- the round
def run_round(mode="composition", frames=900, batch=8, base=None, param=None, values=None,
              dry=False):
    cert = Certification(os.path.join(HERE, "_metrology"))
    ok, why = cert.may_admit()
    if not ok:
        print(f"[round] ADMISSION GATE CLOSED -- refusing to run.\n  {why}")
        return 2

    cfg = CampaignConfig(batch=batch, keep_truncate=max(2, batch // 3))
    sup = Supervisor(cfg, CAMP)
    ledger = llm.BudgetLedger()
    ledger.new_round()
    llm.ensure_files(cfg.objective)
    lm = LeverMap(MAP)
    rid = sup.round + 1

    print("=" * 96)
    print(f"ROUND {rid}   mode={mode}   campaign={cfg.name}")
    print("=" * 96)

    if mode == "theta":
        cands = build_theta_batch(base, param, values, batch)
    else:
        cands = build_composition_batch(sup, cfg, batch, ledger)
    if not cands:
        print("[round] no candidates -- escalating")
        print(json.dumps(sup.escalate(), indent=1))
        return 1

    # ------------------------------------------------ hypotheses FIRST, then configs
    posed = []
    for i, (g, lbl, sl) in enumerate(cands):
        nm = f"r{rid:02d}_{i:02d}_{comp_hash(g)[1:7]}" + (f"_{i}" if mode == "theta" else "")
        T.write_config(g, nm, frames=frames)
        h = Hypothesis(hid=f"R{rid}.{i}.{comp_hash(g)[1:7]}", comp_hash=comp_hash(g),
                       parent_hash=None, edit=lbl,
                       intent=sl.get("intent", "confirmatory"),
                       claim=sl.get("claim", lbl),
                       metric=sl.get("metric", "protr_peak"),
                       predicted=sl.get("predicted", "unstated"),
                       rationale=sl.get("why", ""), round_id=rid)
        if not dry:
            sup.reg.pose(h)
        posed.append((nm, g, h))
        print(f"  {nm}  {lbl[:44]:44} {h.intent:13} predict {h.predicted[:40]}")

    if dry:
        print("\n[round] --dry: configs + hypotheses written, nothing submitted")
        return 0

    # ------------------------------------------------ run
    names = [n for n, _, _ in posed]
    ids = cluster.submit(names, frames=frames, do_q=True, campaign=f"round{rid}")
    if not ids:
        print("[round] submission did not land -- aborting rather than scoring nothing")
        return 1
    cluster.wait_for_ids(ids, poll=60)

    # ------------------------------------------------ analyse + watch + score
    rows = []
    for nm, g, h in posed:
        d = os.path.join(LOG, nm, "diag.json")
        if not os.path.exists(d):
            sup.reg.resolve(h.hid, {}, "inconclusive", note="no diag.json")
            continue
        summ = json.load(open(d)).get("summary", {})
        post = C.check_posthoc(summ)
        if post:
            sup.reg.resolve(h.hid, summ, "inconclusive", note=f"NOT EVIDENCE: {post}")
            print(f"  [critic] {nm} is not evidence: {post}")
            continue
        out_dir = os.path.join(LOG, nm)
        an = A.analyse(nm, out_dir, n=3)
        wa = A.watch(nm, out_dir, an["analyst_consensus"])
        summ.update({k: v for k, v in an.items() if k != "analyst_reads"})
        summ.update(wa)
        if wa.get("watcher_blocks"):
            print(f"  [watcher] {nm} VETOED -- {wa.get('watcher_why','')[:110]}")
        sc = score_run(summ, cfg) if not wa.get("watcher_blocks") else -np.inf
        got = float(summ.get("protr_peak", 0.0))
        outcome = "confirmed" if _pred_holds(h.predicted, got) else "refuted"
        sup.reg.resolve(h.hid, summ, outcome, run_ids=[nm])
        lm.add(comp_hash(g), g, an["analyst_consensus"], sc if np.isfinite(sc) else -1.0, summ, nm)
        rows.append((nm, g, summ, sc, outcome, h))

    if not rows:
        print("[round] no admissible evidence")
        print(json.dumps(sup.observe([]), indent=1))
        return 1

    # ------------------------------------------------ rank (measure) + judge (2nd opinion)
    rows.sort(key=lambda r: -r[3])
    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        w, why_j = A.judge_pair(
            {"name": a[0], "caption": _cap(a[0]), "metrics": _m(a[2])},
            {"name": b[0], "caption": _cap(b[0]), "metrics": _m(b[2])})
        agree = (w >= 0.5)
        print(f"  [judge] top pair: {'agrees with' if agree else 'DISAGREES with'} the metric "
              f"ranking -- {why_j[:130]}")
        if not agree:
            print("  [judge] eye/number divergence recorded -- this is the signal, not noise")

    kept, dropped = truncate(rows, cfg.keep_truncate)
    print(f"\n[rank] kept {len(kept)}, dropped {len(dropped)} (never refined)")
    for nm, g, s, sc, oc, h in rows:
        tag = "KEEP" if (nm, g, s, sc, oc, h) in kept else "drop"
        print(f"  [{tag}] {nm}  score {sc:6.2f}  protr_peak {s.get('protr_peak',0):5.2f} "
              f"phen={s.get('analyst_consensus','?'):9} watcher={s.get('watcher_verdict','?'):10}"
              f" [{oc}]" + ("  SURPRISE" if h.is_surprise else ""))

    # ------------------------------------------------ interpret + ledger + meta-review
    for nm, g, s, sc, oc, h in kept:
        A.interpret(comp_hash(g), g.name_region(), h.edit, s,
                    {k: s.get(k) for k in ("analyst_consensus", "analyst_agreement")},
                    os.path.join(CAMP, "causal_descriptions.md"))

    rep = sup.observe([(g, s, h.hid) for _, g, s, _, _, h in rows])
    sup.reg.render_knowledge(os.path.join(CAMP, "knowledge.md"),
                             ledger={"kept": len(kept), "dropped": len(dropped)}, round_id=rid)
    lm.render(os.path.join(CAMP, "lever_map.md"))
    A.meta_review(rid)
    save_frontier([g for _, g, _, _, _, _ in kept] or load_frontier())

    cov = lm.coverage()["overall"]
    print(f"\n[supervisor] {json.dumps({k: v for k, v in rep.items() if k != 'mix_why'})}")
    print(f"  mix: {rep['mix_why']}")
    print(f"[map] coverage {cov['frac']:.0%} ({cov['covered']}/{cov['total']} cells, "
          f"{cov['n_runs']} runs)")
    print(f"[llm] {ledger.summary()}")
    return 0


# --------------------------------------------------------------------------- helpers
def _theta_prediction(param, v, base_v):
    """A default falsifiable prediction when none was supplied.

    For vcap specifically the mechanism is on record: vcap force-divides oversized cells while
    BYPASSING the throttle, and the Tyssue notes attribute the tube tip to cells BACKLOGGING
    behind that throttle and continuing to ramp. So a LOWER cap should split tip cells sooner,
    remove the backlog, and SHORTEN the tube; a higher cap should restore it. That is the claim
    under test, and it is one we can be wrong about.
    """
    if param.endswith("vcap"):
        if v <= 0.5:
            return "protr_peak < 1.5"      # no cap == no forced split of oversized tip cells
        if base_v is not None and v > base_v:
            return "protr_peak >= 2.0"     # a higher cap restores the backlog -> longer tube
        return "protr_peak < 2.0"
    return "protr_peak >= 2.0"


def _pred_holds(pred, got):
    import re
    m = re.search(r"(>=|<=|>|<)\s*([0-9.]+)", str(pred))
    if not m:
        return True                     # only a genuinely UNSTATED prediction is uncounted
    op, v = m.group(1), float(m.group(2))
    return {">=": got >= v, "<=": got <= v, ">": got > v, "<": got < v}[op]


def _cap(nm):
    p = os.path.join(LOG, nm, "description.txt")
    return open(p, errors="ignore").read()[:900] if os.path.exists(p) else ""


def _m(s):
    return {k: s.get(k) for k in ("protr_peak", "protr_final", "ta_n_tubes_final",
                                  "mech_p_ratio")}


def _ledger_summary(sup, lm):
    cov = lm.coverage()
    solo = lm.solo()
    lines = [f"round {sup.round}, {cov['overall']['n_runs']} runs, "
             f"map coverage {cov['overall']['frac']:.0%}",
             f"phenotypes so far: {lm.phenotypes()}", "", "solo effects (Δscore, verdict):"]
    for op, v in sorted(solo.items(), key=lambda kv: -(kv[1].get('delta') or -99))[:10]:
        lines.append(f"  {op:24} {v.get('delta','—')}  {v['verdict']}")
    inter = [(k, v) for k, v in lm.pairs().items()
             if v.get("verdict") in ("SYNERGY", "ANTAGONISM")]
    if inter:
        lines += ["", "interactions found:"]
        lines += [f"  {k}: {v['verdict']} ({v['interaction']:+})" for k, v in inter[:6]]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["composition", "theta"], default="composition")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", default="round_40_mc8")
    ap.add_argument("--param", default=None)
    ap.add_argument("--values", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    vals = [float(x) for x in a.values.split(",")] if a.values else None
    sys.exit(run_round(mode=a.mode, frames=a.frames, batch=a.batch, base=a.base,
                       param=a.param, values=vals, dry=a.dry))
