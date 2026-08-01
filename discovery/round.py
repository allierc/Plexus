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
import escalation as ESC                                                  # noqa: E402
import predict as PR                                                      # noqa: E402
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
# Per-round agent timing, one JSON line per round, so LLM cost can be tracked ACROSS rounds
# instead of scrolling past once in a terminal.
LLM_TIMING = os.path.join(CAMP, "llm_timing.jsonl")


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


# --------------------------------------------------------------------- stale-Q quarantine (D2)
# THE DEFECT BEING PREVENTED. `run_one.quasi_static_Q` used to delete the growth/driver operators
# and then `S.load()` the edited spec -- which rebuilds the simulation FROM THE SEED SPHERE. It
# therefore relaxed a fresh seed sphere for 60 frames and reported ITS elongation, whatever the
# run had actually done. Q came back 1.014 in 14 of the 16 runs that computed one, over real
# `protr_final` spanning 1.02-2.81. That constant reaches the science in three places:
#     control.score_run        LIVE. Q carries weight 1.0 in the campaign's scalar objective.
#     control.meets_success    LIVE. Gates the success criterion at Q >= 2.0, hence unreachable.
#     predict.score            LATENT. `Q_drop` (= protr_final - Q) is in predict.KNOWN_METRICS,
#                              so a prediction naming it is MEANT to be checked -- but
#                              `predict.parse` lowercases the captured metric name, and Q_drop is
#                              the only KNOWN_METRIC carrying a capital, so every Q_drop clause
#                              currently resolves `inconclusive` on "q_drop not measured".
#                              MEASURED, not assumed:
#                                PR.score("Q_drop >= 1.5", {"Q_drop": 1.792}) -> inconclusive
#                                PR.score("Q_drop >= 1.5", {"q_drop": 1.792}) -> confirmed
#                              That case-fold bug is in predict.py, not here; the day it is fixed
#                              the path goes live, and the quarantine below already closes it.
# The generator was fixed (run_one.quasi_static_Q now checkpoints the end state and guards frame
# 0), but the RECORDS on disk still hold the old value.
#
# Those records are IMMUTABLE: discovery/_archive/analyses.jsonl and log/okuda/*/diag.json are
# the research record and are never deleted, and never rewritten in place. The idiom is
# hypothesis.HypothesisRegister.amend(): annotate, append, leave every prior line standing. So
# the poison is intercepted AT READ TIME instead -- the value is moved aside under a `__STALE`
# key where nothing scores it, the reason travels with it into the hypothesis record, and an
# append-only quarantine ledger records each interception. The archive is opened read-only here.
SEED_SPHERE_Q = 1.014          # protr of the relaxed SEED SPHERE -- what the broken test measured
SEED_SPHERE_Q_TOL = 5e-4       # run_one rounds Q to 3 dp, so this is exact-match with slack
Q_KEY = "Q_protr_after_relax"
Q_DERIVED = ("Q_drop",)        # protr_final - Q: computed FROM the poison, so equally poisoned
STALE_SUFFIX = "__STALE"
Q_QUARANTINE = os.path.join(CAMP, "q_quarantine.jsonl")   # append-only; NOT in _archive/


class StaleQ(ValueError):
    """A poisoned Q was found where the caller demanded a trustworthy one."""


def stale_q_reason(summary):
    """Why this summary's Q must not be scored, or None if it may be.

    The test is on the VALUE, not on provenance, because provenance is not recoverable: the
    broken and the fixed `quasi_static_Q` both write `metric_version="metric_v1"`, so no record
    can say which code produced it. A Q sitting on the seed-sphere constant is therefore
    INDISTINGUISHABLE from the artefact -- and indistinguishable resolves to STALE, never to
    trusted. Refusing a good value costs one re-measurement; accepting a poisoned one costs a
    conclusion, which is a bill this campaign has already paid.

    The corroborating evidence is reported with the reason so a human re-scoring the archive can
    see how strong each case is: a run that ended at protr_final 2.805 and "relaxed" to 1.014 is
    the seed sphere beyond argument, whereas one that ended at 1.024 is merely unprovable.
    """
    if not isinstance(summary, dict):
        return None
    q = summary.get(Q_KEY)
    if q is None:
        return None
    try:
        qf = float(q)
    except (TypeError, ValueError):
        return f"{Q_KEY}={q!r} is not numeric -- cannot be scored"
    if abs(qf - SEED_SPHERE_Q) > SEED_SPHERE_Q_TOL:
        return None
    fin = summary.get("protr_final")
    corr = ""
    try:
        if fin is not None and abs(float(fin) - qf) > 0.15:
            corr = (f"; the run ended at protr_final={float(fin):.3f}, so the relaxation did not "
                    f"start where the run finished")
    except (TypeError, ValueError):
        pass
    return (f"STALE {Q_KEY}={qf:.3f}: this is the relaxed-seed-sphere constant produced by the "
            f"pre-fix quasi_static_Q, which rebuilt the simulation from the seed instead of "
            f"continuing from the end state{corr}. Quarantined -- recompute before scoring.")


def _quarantine_log(entry, ledger_path=None):
    """Append one line to the quarantine ledger. Never raises into the caller.

    Append-only, and outside _archive/ and log/okuda/, because an interception is a NEW fact
    about a record -- not a licence to edit the record.
    """
    p = ledger_path or Q_QUARANTINE
    try:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")
        return p
    except Exception as e:                       # a failed audit line must not lose the scrub
        print(f"  [Q-stale] could not append to the quarantine ledger {p}: "
              f"{type(e).__name__}: {e}")
        return None


def scrub_stale_q(summary, source="", ledger_path=None, quiet=False):
    """Return a COPY of `summary` with any poisoned Q moved OUT OF SCORING REACH.

    The value is not destroyed -- it moves to `Q_protr_after_relax__STALE` (and `Q_drop__STALE`)
    and the reason rides alongside as `Q_stale_reason`. Downstream this means:
      * control.score_run     sees no Q and takes its documented `else fin` fallback, instead of
                              adding the sphere constant with weight 1.0
      * control.meets_success likewise falls back rather than testing 1.014 >= 2.0
      * predict.score         reports `Q_drop not measured` -> inconclusive, rather than scoring
                              a prediction against an artefact (that clause is already dead for
                              an unrelated reason -- see the case-fold note above -- so this is
                              the path being held shut, not one being reopened)
    Never mutates the caller's dict; never writes to the file the summary came from.
    """
    reason = stale_q_reason(summary)
    out = dict(summary or {})
    if not reason:
        return out
    moved = {}
    for k in (Q_KEY,) + Q_DERIVED:
        if k in out:
            moved[k] = out.pop(k)
            out[k + STALE_SUFFIX] = moved[k]
    out["Q_stale"] = True
    out["Q_stale_reason"] = reason
    if source:
        out["Q_stale_source"] = source
    if not quiet:
        print(f"  [Q-stale] {source or 'summary'}: {reason}")
    _quarantine_log({"source": source, "quarantined": moved, "reason": reason,
                     "protr_final": summary.get("protr_final"),
                     "protr_peak": summary.get("protr_peak")}, ledger_path)
    return out


def refuse_stale_q(summary, source=""):
    """Hard refusal, for a re-scorer that must not proceed at all. Raises rather than scores."""
    reason = stale_q_reason(summary)
    if reason:
        _quarantine_log({"source": source, "reason": reason, "action": "REFUSED"}, None)
        raise StaleQ(f"{source or 'summary'}: {reason}")
    return summary


def read_diag_summary(path, source=None, quiet=False):
    """Read a run's diag.json `summary` with the stale-Q quarantine already applied."""
    d = json.load(open(path)).get("summary", {})
    return scrub_stale_q(d, source or path, quiet=quiet)


def read_archive_analyses(path=None, quiet=True):
    """Yield (run_id, metric_version, scrubbed_result) from the IMMUTABLE analyses.jsonl.

    THE ENTRY POINT FOR ANYTHING THAT RE-SCORES FROM THE ARCHIVE. Opened 'r' only -- this
    function must never gain a write path.
    """
    p = path or os.path.join(HERE, "_archive", "analyses.jsonl")
    if not os.path.exists(p):
        return
    for i, line in enumerate(open(p)):
        if not line.strip():
            continue
        e = json.loads(line)
        res = e.get("result")
        if isinstance(res, dict):
            res = scrub_stale_q(res, f"analyses.jsonl:{i + 1}:{e.get('run_id')}", quiet=quiet)
        yield e.get("run_id"), e.get("metric_version"), res


def quarantine_scan(archive=None, log_root=None, ledger_path=None, verbose=True):
    """Sweep every Q on disk, report which are stale, and record the interceptions.

    READS ONLY. It exists so the poison count is a measured number rather than a remembered one,
    and so the append-only ledger names every record a re-scorer has to skip.
    """
    n_q = n_stale = 0
    rows = []
    for rid, _mv, res in read_archive_analyses(archive, quiet=True):
        if not isinstance(res, dict):
            continue
        if res.get("Q_stale"):
            n_q += 1
            n_stale += 1
            rows.append(("analyses.jsonl", rid, res.get(Q_KEY + STALE_SUFFIX), True))
        elif res.get(Q_KEY) is not None:
            n_q += 1
            rows.append(("analyses.jsonl", rid, res.get(Q_KEY), False))
    import glob
    for d in sorted(glob.glob(os.path.join(log_root or LOG, "*", "diag.json"))):
        try:
            s = read_diag_summary(d, source=os.path.basename(os.path.dirname(d)), quiet=True)
        except Exception as e:
            print(f"  [Q-scan] unreadable {d}: {type(e).__name__}: {e}")
            continue
        if s.get("Q_stale"):
            n_q += 1
            n_stale += 1
            rows.append(("diag.json", os.path.basename(os.path.dirname(d)),
                         s.get(Q_KEY + STALE_SUFFIX), True))
        elif s.get(Q_KEY) is not None:
            n_q += 1
            rows.append(("diag.json", os.path.basename(os.path.dirname(d)), s.get(Q_KEY), False))
    if verbose:
        print(f"[Q-scan] {n_stale} STALE of {n_q} recorded Q values "
              f"(seed-sphere constant {SEED_SPHERE_Q} +- {SEED_SPHERE_Q_TOL})")
        for where, who, val, bad in rows:
            print(f"   {'STALE' if bad else '  ok ':>5}  {where:15} {str(who)[:26]:26} Q={val}")
        print(f"[Q-scan] the records are UNCHANGED; interceptions appended to "
              f"{ledger_path or Q_QUARANTINE}")
    return {"n_q": n_q, "n_stale": n_stale, "rows": rows}


# --------------------------------------------------------------------------- LOOP I batch

def _ground_starting_conditions(g, sl):
    """Set this slot's starting cell count from the PAPER rather than from a config default.

    All three morphology figures -- tubulation, branching, undulation -- are ONE coupling
    experiment at ONE starting size (2,000 cells); only chi and gamma separate them. An earlier
    version of this function chose between 200 and 2,000 by reading the slot's claim text, on the
    belief that tubulation started at 200. It does not: that number belongs to two control
    experiments the paper runs first, one on arrested tissue and one with growth decoupled from
    the morphogen. The quote attached to it was accurate and the mapping was wrong, which is the
    more dangerous of the two -- it made a tenfold error look verified.

    So there is no case to infer. A slot that declares an explicit `okuda_case` gets it; anything
    else gets the coupling case, because that is the experiment this campaign exists to reproduce.

    Returns the graph unchanged if the composition has no seeding node -- a checkpoint start
    carries its own count and must not be overwritten.
    """
    from agents.grounder import SETUP, setup
    case = sl.get("okuda_case", "coupled")
    if case not in SETUP:
        print(f"  [grounder] unknown okuda_case {case!r} -- refusing to guess a starting size")
        return g
    spec = setup(case)
    seeder = next((o for o in g.ops if o["op"] == "seed_mesh_3d"), None)
    if seeder is None:
        return g
    key = f"{seeder['id']}.n_cells"
    if g.params.get(key) == spec["n_cells"]:
        return g
    sl["grounded"] = (f"n_cells {spec['n_cells']} ({case}) -- Okuda: "
                      f"\u201c{spec['quote'][:80]}\u2026\u201d")
    return g.with_params({**g.params, key: spec["n_cells"]})



def _read_one(nm, out_dir, ledger):
    """Three analysts and the eye-check for one run. The analysts run CONCURRENTLY.

    PHASE 3(c). Twenty of a round's twenty-six agent calls are per-run, and they are completely
    independent of one another -- three analysts reading the same run cannot influence each
    other, that is the entire reason there are three. They ran one after another because the
    code was written as a loop, not for any reason to do with the science, and on a five-run
    round that is sixty minutes of allowance spent serially.

    The eye-check still runs AFTER, because it is not independent: it is handed the analysts'
    consensus and asked whether the movie supports it. Parallelising a dependency would not be
    an optimisation, it would be a different experiment.
    """
    from concurrent.futures import ThreadPoolExecutor
    an = A.analyse(nm, out_dir, n=3, ledger=ledger, parallel=True)
    wa = A.watch(nm, out_dir, an["analyst_consensus"], ledger=ledger)
    return an, wa


def _read_batch(posed_rows, ledger):
    """Every run's readers, across the whole batch, at once. Returns {name: (analysis, watch)}.

    The batch dimension is independent too: run A's analysts have nothing to say to run B's. So
    the whole block costs the time of its slowest single call rather than the sum of twenty.
    """
    from concurrent.futures import ThreadPoolExecutor
    out = {}
    if not posed_rows:
        return out
    with ThreadPoolExecutor(max_workers=min(8, len(posed_rows))) as ex:
        futs = {ex.submit(_read_one, nm, d, ledger): nm for nm, d in posed_rows}
        for f in futs:
            nm = futs[f]
            try:
                out[nm] = f.result()
            except Exception as e:
                print(f"  [read] {nm} FAILED: {type(e).__name__}: {str(e)[:90]}")
                out[nm] = None
    return out



def _referee_rank(rows, cfg):
    """Rank a batch by pairwise tournament (Bradley-Terry) instead of sorting on one score.

    PHASE 3(b). `control.rank_btl` has existed and been self-tested since the loop was built and
    has never ranked a real batch: the live round sorted on `score_run`, which is a total order
    imposed by whichever metric the instrument gate happened to admit. A tournament asks each
    comparison on its own terms and aggregates, so no single number is the ranking.

    The comparator stays ARITHMETIC -- a veto beats everything, then the scalar. Making it an
    agent would add twenty-odd calls a round to re-decide something already measured; the value
    here is the aggregation, not another opinion.
    """
    from control import rank_btl
    if len(rows) < 3:
        return sorted(rows, key=lambda r: -r[3])

    def compare(a, b):
        if bool(a[2].get("watcher_blocks")) != bool(b[2].get("watcher_blocks")):
            return 0.0 if a[2].get("watcher_blocks") else 1.0
        sa, sb = a[3], b[3]
        if not np.isfinite(sa) and not np.isfinite(sb):
            return 0.5
        if not np.isfinite(sa):
            return 0.0
        if not np.isfinite(sb):
            return 1.0
        return 1.0 if sa > sb else (0.0 if sa < sb else 0.5)

    strength = rank_btl(rows, compare)
    order = sorted(range(len(rows)), key=lambda i: -strength.get(i, 0.0))
    ranked = [rows[i] for i in order]
    naive = sorted(range(len(rows)), key=lambda i: -rows[i][3])
    if order != naive:
        print("  [referee] the tournament disagrees with sorting on the scalar -- "
              "recorded, and the tournament is what the round uses")
    return ranked


def build_composition_batch(sup, cfg, n_slots, ledger):
    """Proposer(LLM) -> Critic -> Reflection(LLM). Returns [(graph, label, hyp_fields)]."""
    frontier = load_frontier()
    seen = seen_hashes(sup)
    lm = LeverMap(MAP)
    ledger_summary = _ledger_summary(sup, lm)

    ok, slots = P.propose(frontier, cfg, sup.prox, ledger_summary, sup.round + 1,
                          n_slots=n_slots, ledger=ledger)
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
        # THE GROUNDER SPEAKS HERE, and until now it never did. It is the only agent that reads
        # Okuda's paper, it had no call site anywhere in the pipeline, and the consequence was
        # measured rather than argued: a 27-run batch was launched at 150 cells against a paper
        # that says 200, and every run stopped dead on the mesh reservoir.
        #
        # It ADVISES, it does not gate. The faithful share of a batch inherits his starting
        # conditions; the exploratory share is free to leave them, which is the 70/30 split --
        # a search that may only ever stand where the paper stands cannot discover that the
        # paper is not the only place to stand. What is NOT optional is the reservoir, and that
        # lives in the translator, where it applies to both shares alike.
        fid = sl.get("fidelity", "okuda" if i < max(1, int(round(0.7 * len(slots)))) else "free")
        if fid == "okuda":
            g = _ground_starting_conditions(g, sl)
        sl["fidelity"] = fid
        adm, rej = C.admit(g, seen if sl.get("intent") != "control" else ())
        if not adm:
            rejected.append((i, f"CRITIC: {rej}"))
            continue
        out.append((g, lbl, sl))
    for i, why in rejected:
        print(f"  [critic] slot {i} rejected -- {why}")

    review = A.reflect([{k: v for k, v in s.items() if k != "edit"} for _, _, s in out],
                       ledger=ledger)
    print(f"  [reflection] batch_ok={review.get('batch_ok')} :: {review.get('verdict','')[:200]}")
    for iss in review.get("issues", []):
        print(f"     slot {iss.get('slot')}: [{iss.get('severity')}] {iss.get('problem')}")
    if review.get("batch_ok") is False and any(
            i.get("severity") == "serious" for i in review.get("issues", [])):
        print("  [reflection] SERIOUS issues -- the batch is not run as proposed.")
        return []
    return out


# --------------------------------------------------------------------------- LOOP II batch
def build_theta_batch(base_name, param, values, n_slots, predictions=None, intents=None):
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
        # INTENT MUST NOT BE READ OFF THE SHAPE OF THE PREDICTION STRING.
        # This line used to be:
        #     intent = "confirmatory" if (... or pred.startswith("protr_peak >=")) else "adversarial"
        # i.e. "predicts an increase" => confirmatory, "predicts a decrease" => adversarial. That
        # is the SAME vacuity proposer.py was written to remove from Loop I (`intent = adversarial
        # if lbl.startswith("-")`), reintroduced in Loop II. Under it the surprise rate stops
        # measuring belief and starts measuring the SIGN OF THE EFFECT: a predicted decrease can
        # only ever be "adversarial", so it can never be a surprise when it fails.
        #
        # A theta point is CONFIRMATORY when it is the standing mechanism story's own forecast,
        # and ADVERSARIAL only when it is chosen to break that story. `_theta_prediction` derives
        # every point from the recorded vcap mechanism, so these are its forecasts: confirmatory
        # unless the caller says otherwise.
        intent = (intents or {}).get(str(v), "confirmatory")
        out.append((g, f"theta {param}={v}",
                    {"intent": intent, "metric": "protr_peak", "predicted": pred,
                     "claim": f"{param}={v} on {base_name}: {pred}",
                     "why": "Loop II: a parameter hypothesis. Composition identity is unchanged "
                            "by construction, so this cannot pose as a new mechanism -- but it "
                            "is a prediction that can be wrong, and it counts as one."}))
    print(f"  [theta] {len(out)} points of `{param}` on {base_name} (comp {h0}, constant)")
    return out


# --------------------------------------------------------------------------- the round
def _finish(ledger, rid, mode, status, code):
    """Print the per-role agent-time breakdown and persist it. EVERY exit path calls this.

    Reporting only on the happy path is how a round that died early, or one that blew the LLM
    ceiling, comes to look indistinguishable from a clean one in the log.

    Call it through `_RoundBookkeeping.finish`, never directly: that is what puts the CRASH path
    on the same footing as the seven `return`s.
    """
    print()
    print(ledger.report("round"))
    p = ledger.persist(LLM_TIMING, mode=mode, status=status, exit_code=code)
    print(f"[llm] {ledger.summary()}")
    if ledger.overruns:
        print(f"[llm] ROUND EXCEEDED THE {llm.ROUND_LLM_BUDGET_MIN} min LLM CEILING "
              f"({len(ledger.overruns)} breach(es)) -- this round is NOT a clean round.")
    if ledger.unmetered:
        print(f"[llm] {len(ledger.unmetered)} UNMETERED call(s) bypassed run_agent() -- "
              f"their cost is attributed but not budgeted; fix the call site.")
    if p:
        print(f"[llm] timing appended to {p}")
    return code


class _RoundBookkeeping:
    """Guarantees the round's timing is written EXACTLY ONCE, on EVERY way out of the round.

    THE DEFECT BEING PREVENTED. `run_round` had seven `return`s, all correctly routed through
    `_finish`, and ZERO try blocks. Hand-routing covers only the exits somebody remembered. An
    uncaught exception -- and this campaign has spent weeks finding defects MID-ROUND, so a crash
    is the LIKELY exit, not the exotic one -- unwound straight past all seven and the round's LLM
    cost vanished from llm_timing.jsonl entirely. A crashed round then looked, in the cost log,
    exactly like a round that never ran.

    `crash()` records and then the caller RE-RAISES. Nothing here swallows an exception: the
    project's standing rule is never to swallow an exception around an artefact, and a crash must
    stay loud. The bookkeeping is also not allowed to become the thing that hides the real
    failure, so every path through `finish()` catches its own errors and prints them instead.
    """

    def __init__(self, ledger, mode):
        self.ledger, self.mode = ledger, mode
        # `_finish` takes rid but does not use it -- the round id that lands in llm_timing.jsonl
        # comes from `ledger.round_id`, set by `ledger.new_round(rid)` in the body. So a crash
        # BEFORE that call honestly records round=null, and one after it records the round. This
        # is tracked here only so the crash path passes the same argument the seven returns do.
        self.rid = None
        self.done = False

    def finish(self, rid=None, status="complete", code=0, warn=False):
        if self.done:
            return code
        self.done = True
        if rid is not None:
            self.rid = rid
        if warn:
            print(f"\n[round] BOOKKEEPING FALLBACK: run_round returned without calling _finish "
                  f"(status={status}). The timing is recorded anyway; the exit path that skipped "
                  f"it is the bug to fix.")
        try:
            return _finish(self.ledger, self.rid, self.mode, status, code)
        except Exception as e:
            print(f"[round] TIMING BOOKKEEPING FAILED ({type(e).__name__}: {e}) -- round "
                  f"{self.rid} status={status} is NOT in {LLM_TIMING}")
            return code

    def crash(self, exc):
        print(f"\n[round] UNCAUGHT {type(exc).__name__} mid-round -- recording this round's "
              f"timing BEFORE the exception propagates: {str(exc)[:160]}")
        return self.finish(status=f"crashed:{type(exc).__name__}", code=3)


def run_round(mode="composition", frames=900, batch=8, base=None, param=None, values=None,
              dry=False):
    """Thin wrapper: the round's body, with the timing bookkeeping made crash-proof."""
    # The ledger is created FIRST and every `return` in the body goes through `bk.finish`, so
    # even a round that aborts at the admission gate reports what it spent getting there.
    ledger = llm.BudgetLedger(path=LLM_TIMING)
    bk = _RoundBookkeeping(ledger, mode)
    try:
        return _run_round(bk, ledger, mode, frames, batch, base, param, values, dry)
    except BaseException as exc:
        # BaseException, not Exception: a Ctrl-C or a SystemExit mid-round costs the same LLM
        # minutes as a TypeError and must be accounted for the same way.
        bk.crash(exc)
        raise                      # NEVER swallowed -- the traceback is the point
    finally:
        # Belt and braces for an EIGHTH return path added later that forgets to route through
        # `finish`. Hand-routing is the fragile part, so the guarantee lives here rather than in
        # each individual `return`. No-op once `finish` has already run.
        bk.finish(status="unfinished_no_exit_status", code=4, warn=True)


def _run_round(bk, ledger, mode, frames, batch, base, param, values, dry):
    cert = Certification(os.path.join(HERE, "_metrology"))
    ok, why = cert.may_admit()
    if not ok:
        print(f"[round] ADMISSION GATE CLOSED -- refusing to run.\n  {why}")
        return bk.finish(None, "admission_gate_closed", 2)

    cfg = CampaignConfig(batch=batch, keep_truncate=max(2, batch // 3))
    sup = Supervisor(cfg, CAMP)
    lm = LeverMap(MAP)
    rid = sup.round + 1
    ledger.new_round(rid)
    llm.ensure_files(cfg.objective)

    print("=" * 96)
    print(f"ROUND {rid}   mode={mode}   campaign={cfg.name}")
    print("=" * 96)

    _purge_round_configs(rid, mode)    # F18: stale configs must not shadow this round's
    if mode == "theta":
        cands = build_theta_batch(base, param, values, batch)
    else:
        cands = build_composition_batch(sup, cfg, batch, ledger)
    if not cands:
        print("[round] no candidates -- escalating")
        print(json.dumps(run_escalation(cfg, sup, lm, rid, "the batch builder produced no "
                                        "runnable candidate", ledger=ledger), indent=1))
        return bk.finish(rid, "no_candidates", 1)

    # ------------------------------------------------ hypotheses FIRST, then configs
    # CLAIM THE ROUND NUMBER BEFORE POSING ANYTHING.
    # `pose()` refuses to overwrite a hypothesis id, and ids are f"R{rid}.{slot}.{hash}". The
    # round counter only advanced at the END, in `sup.observe`, so a round that died after
    # posing left the counter untouched -- and every retry rebuilt the SAME ids and raised
    # `hypothesis R3.0.xxxxxx already posed`, forever. One crash on day two would have cost the
    # rest of the week, silently, and nothing in a week-long run would have been watching.
    # A spent round is spent: claim the number first, so a retry gets a fresh one.
    sup.round = rid
    sup._save()

    posed = []
    for i, (g, lbl, sl) in enumerate(cands):
        # F18: names carry round AND mode AND slot. A previous round's config with a matching
        # name could otherwise be picked up by a job -- silently running the wrong experiment.
        nm = f"r{rid:03d}{mode[0]}_{i:02d}_{comp_hash(g)[1:7]}"
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
        return bk.finish(rid, "dry", 0)

    # ------------------------------------------------ run
    names = [n for n, _, _ in posed]
    # THE SPLIT: local = intelligence, cluster = jobs. Certify the job env BEFORE committing a
    # round to it -- an import that exists only locally would degrade every cluster run silently
    # while the campaign still reported completed jobs (this is exactly how the Watcher went
    # blind for a whole wave).
    if cluster.preflight(verbose=True) is False:
        print("[round] preflight FAILED -- not submitting. Fix the job environment first.")
        return bk.finish(rid, "preflight_failed", 1)
    ids = cluster.submit(names, frames=frames, do_q=True, campaign=f"round{rid}")
    if not ids:
        print("[round] submission did not land -- aborting rather than scoring nothing")
        return bk.finish(rid, "submit_failed", 1)
    # The return value used to be discarded, so "all six finished" and "we waited 24 h and gave
    # up" were indistinguishable. A killed straggler is recorded and its hypothesis is resolved
    # `inconclusive` below (no diag.json), which keeps a degenerate slot out of the surprise rate
    # instead of scoring it as evidence.
    wait = cluster.wait_for_ids(ids, poll=60)
    if not wait["ok"]:
        print(f"[round] batch did not complete cleanly: exit={wait['exit']} "
              f"killed={wait['killed']} timed_out={wait['timed_out']} -- scoring what landed")

    # ------------------------------------------------ caption the wave (one model load)
    # Must happen BEFORE the Analysts and the Watcher: both read description.txt, and a blind
    # Watcher cannot veto. This is where the cluster's missing `transformers` is worked around.
    from caption_wave import caption_wave
    caption_wave(names)

    # ------------------------------------------------ analyse + watch + score
    rows = []
    for nm, g, h in posed:
        d = os.path.join(LOG, nm, "diag.json")
        if not os.path.exists(d):
            sup.reg.resolve(h.hid, {}, "inconclusive", note="no diag.json")
            continue
        # THE ONLY DOOR the poisoned Q can come through in a round: a run's own diag.json. A
        # re-run of round N re-reads log/okuda/rNNNc_*/diag.json, and 14 of those on disk hold
        # the seed-sphere constant. Scrub at the read, so score_run / meets_success /
        # predict.score cannot reach it. The diag.json itself is untouched.
        summ = read_diag_summary(d, source=nm)
        post = C.check_posthoc(summ)
        if post:
            sup.reg.resolve(h.hid, summ, "inconclusive", note=f"NOT EVIDENCE: {post}")
            print(f"  [critic] {nm} is not evidence: {post}")
            continue
        out_dir = os.path.join(LOG, nm)
        an, wa = _read_one(nm, out_dir, ledger)
        summ.update({k: v for k, v in an.items() if k != "analyst_reads"})
        summ.update(wa)
        if wa.get("watcher_blocks"):
            print(f"  [watcher] {nm} VETOED -- {wa.get('watcher_why','')[:110]}")
        sc = score_run(summ, cfg) if not wa.get("watcher_blocks") else -np.inf
        # `predict.score` refuses to guess: a prediction it cannot check resolves `inconclusive`
        # and drops out of the surprise denominator, rather than being recorded as `confirmed`
        # (which is what the old first-match regex did -- see predict.py P1/P2/P3).
        outcome, why = PR.score(h.predicted, summ, primary_metric=h.metric)
        if outcome == "inconclusive":
            print(f"  [predict] {nm} NOT CHECKABLE -- {why[:150]}")
        # A quarantined Q must be visible in the SCIENTIFIC record, not only in the terminal:
        # `why` is what lands in hypotheses.jsonl, so anyone reading the resolution later sees
        # that the survival number was withheld rather than measured.
        if summ.get("Q_stale"):
            why = f"{why} | {summ['Q_stale_reason']}"
        sup.reg.resolve(h.hid, summ, outcome, run_ids=[nm], note=why)
        lm.add(comp_hash(g), g, an["analyst_consensus"], sc if np.isfinite(sc) else -1.0, summ, nm)
        rows.append((nm, g, summ, sc, outcome, h))

    if not rows:
        print("[round] no admissible evidence")
        print(json.dumps(sup.observe([]), indent=1))
        return bk.finish(rid, "no_evidence", 1)

    # ------------------------------------------------ rank (measure) + judge (2nd opinion)
    # THE REFEREE, which until now existed only in its own self-test. Sorting on one scalar is
    # a total order imposed by whichever metric happens to be admitted; a tournament asks the
    # comparison directly and aggregates, so no single number gets to be the ranking. Cheap:
    # the comparator is arithmetic, not an agent, and rank_btl was already written and tested.
    rows = _referee_rank(rows, cfg)
    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        w, why_j = A.judge_pair(
            {"name": a[0], "caption": _cap(a[0]), "metrics": _m(a[2])},
            {"name": b[0], "caption": _cap(b[0]), "metrics": _m(b[2])}, ledger=ledger)
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
                    os.path.join(CAMP, "causal_descriptions.md"), ledger=ledger)

    # EVOLUTION. Truncation alone means a winner is never improved -- the loop can only ever
    # PICK from what the enumerator happened to offer. Refining the best is the other half, and
    # it is the last agent that was written and never called.
    if kept:
        try:
            best_nm, best_g, best_s = kept[0][0], kept[0][1], kept[0][2]
            ev = A.evolve(f"{best_g.name_region()} :: {best_nm} :: "
                          f"{ {k: best_s.get(k) for k in ('protr_peak', 'analyst_consensus')} }",
                          _ledger_summary(sup, lm), ledger=ledger)
            if ev:
                os.makedirs(CAMP, exist_ok=True)
                with open(os.path.join(CAMP, "evolution.jsonl"), "a") as fh:
                    fh.write(json.dumps({"round": rid, "on": best_nm, "proposal": ev}) + "\n")
                print(f"  [evolution] refinement proposed on {best_nm} -- "
                      f"carried into the next round's frontier")
        except Exception as e:
            print(f"  [evolution] FAILED: {type(e).__name__}: {str(e)[:90]}")

    sup.round = rid - 1          # observe() increments; the claim above already moved it
    rep = sup.observe([(g, s, h.hid) for _, g, s, _, _, h in rows])
    sup.reg.render_knowledge(os.path.join(CAMP, "knowledge.md"),
                             ledger={"kept": len(kept), "dropped": len(dropped)}, round_id=rid)
    lm.render(os.path.join(CAMP, "lever_map.md"))
    A.meta_review(rid, ledger=ledger)
    # THE CONTROL IS ALWAYS RETAINED, whatever it scored.
    # `kept` is a RANKING product, and a Watcher veto sets the score to -inf -- so in round 2 the
    # control (the parent, unchanged, protr_peak 4.03) was vetoed, fell out of `kept`, and the
    # frontier became exactly its two ablations (-extrude 1.39, -morphogen_growth_3d 1.03). The
    # search was then breeding from knockouts of a composition it no longer carried, and every
    # subsequent diff would be measured against a parent that is not in the pool. A veto is a
    # statement about what the MOVIE SHOWS, not about whether the composition is a useful parent;
    # it must not silently delete the reference the round's own diffs were taken against.
    front = [g for _, g, _, _, _, _ in kept]
    ctrl = [g for _, g, _, _, _, h in rows if h.intent == "control"]
    for g in ctrl:
        if not any(comp_hash(g) == comp_hash(x) for x in front):
            front.append(g)
            print(f"  [frontier] control {comp_hash(g)} retained despite score/veto -- the "
                  f"round's diffs are measured against it")
    save_frontier(front or load_frontier())

    # ------------------------------------------------ escalate if the space is spent
    # `terminal()` has always been able to return "ESCALATE: ...", and nothing ever read it. So
    # the escalation branch could only be reached by the batch builder returning nothing -- i.e.
    # by a crash-like condition, never by the ordinary course of a campaign exhausting its
    # reachable space, which over weeks is the NORMAL way a mechanism search ends.
    if str(rep.get("reason", "")).startswith("ESCALATE"):
        print(json.dumps(run_escalation(cfg, sup, lm, rid, rep["reason"], ledger=ledger),
                         indent=1))

    cov = lm.coverage()["overall"]
    print(f"\n[supervisor] {json.dumps({k: v for k, v in rep.items() if k != 'mix_why'})}")
    print(f"  mix: {rep['mix_why']}")
    print(f"[map] coverage {cov['frac']:.0%} ({cov['covered']}/{cov['total']} cells, "
          f"{cov['n_runs']} runs)")
    return bk.finish(rid, "complete", 0)


# --------------------------------------------------------------------------- escalation
def run_escalation(cfg, sup, lm, rid, why, ledger=None):
    """Execute the escalation decision. The branch a human took by hand last time.

    Three actions, cheapest first (see escalation.py): open a stage gate, file an operator
    request, or declare the region exhausted. Only the middle one costs an LLM call, and only
    when the stage gates are already spent -- so this cannot become a per-round expense.
    """
    backlog = ESC.Backlog(os.path.join(CAMP, "operator_requests.jsonl"))
    frontier = load_frontier()
    n_edits = sum(len(g.legal_edits(cfg.stage_gate)) for g in frontier)
    action, detail = ESC.decide(cfg, sup, backlog, n_edits)
    print(f"[escalate] {action}: {detail}")

    if action == "open_stage_gate":
        rec = sup.escalate()                       # advances cfg.stage_gate and checkpoints it
    elif action == "request_operator":
        req = A.request_operator(
            _ledger_summary(sup, lm), _map_summary(lm),
            "\n".join(f"  {comp_hash(g)}  {g.name_region()}" for g in frontier[:8]),
            f"{why}\n{detail}", rid, ledger=ledger)
        if not req or not req.get("why_inexpressible"):
            print("[escalate] the agent produced no usable request -- recording the fact rather "
                  "than inventing one")
            rec = sup.escalate()
        else:
            dup = backlog.duplicate_of(req["mechanism"])
            if dup:
                print(f"[escalate] already filed as {dup.rid} -- not duplicating")
                rec = {"action": "operator_request_duplicate", "rid": dup.rid}
            else:
                r = backlog.file(ESC.OperatorRequest(
                    rid=backlog.next_rid(), round_id=rid,
                    mechanism=req["mechanism"], why_inexpressible=req["why_inexpressible"],
                    wanted_for=req.get("wanted_for", ""),
                    proposed_contract=req.get("proposed_contract", {}),
                    acceptance_test=req.get("acceptance_test", ""),
                    evidence=[comp_hash(g) for g in frontier[:4]]))
                print(f"[escalate] filed {r.rid}: {r.mechanism}")
                print(f"           limit: {r.why_inexpressible[:150]}")
                rec = sup.escalate(operator_request=r.to_dict())
    else:
        rec = {"action": "exhausted", "detail": detail,
               "open_requests": [r.rid for r in backlog.open_requests()]}
        sup._log(rec)

    backlog.render(os.path.join(CAMP, "operator_backlog.md"))
    return rec


def _map_summary(lm):
    cov = lm.coverage()["overall"]
    solo = lm.solo()
    lines = [f"coverage {cov['frac']:.0%} ({cov['covered']}/{cov['total']} cells, "
             f"{cov['n_runs']} runs)", "solo effects:"]
    for op, v in sorted(solo.items(), key=lambda kv: -(kv[1].get('delta') or -99))[:12]:
        lines.append(f"  {op:24} {v.get('delta','—')}  {v['verdict']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- helpers
def _purge_round_configs(rid, mode):
    """Remove any config already carrying THIS round's AND THIS MODE's prefix.

    A stale config from an aborted run of the same round number would otherwise shadow a fresh
    one, and a cluster job would silently execute the wrong experiment. Found when a dry-run
    leftover (r01_01_4af688.yaml) shadowed a theta slot.

    THE GLOB MUST CARRY THE MODE LETTER. Configs are namespaced `r{rid:03d}{mode[0]}_...` -- that
    was F18's fix -- but the purge globbed `r{rid:03d}*`, which matches BOTH modes. So running a
    composition round deleted the theta round's configs of the same number, and vice versa. On
    the first attended composition run this really did delete two committed `r001t_*` files
    (recovered with `git restore`). The purge is meant to protect this round from its own
    leftovers, never to reach into the other loop's evidence.
    """
    import glob
    n = 0
    for f in glob.glob(os.path.join(ROOT, "config", "okuda", f"r{rid:03d}{mode[0]}_*.yaml")):
        os.remove(f); n += 1
    if n:
        print(f"  [configs] purged {n} stale {mode} config(s) for round {rid}")



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


# `_pred_holds` lived here. It was a first-match regex that scored an UNPARSEABLE prediction as
# `confirmed`, applied whatever threshold it found first to `protr_peak` regardless of the metric
# named, and read `REFUTED if ...` as the assertion. All three biased the surprise rate DOWNWARD,
# toward "nothing was learned". Replaced by `predict.score()`, which parses every clause with its
# metric and returns `inconclusive` rather than guessing. See predict.py for the three defects.


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
    # Read-only sweep of every recorded Q. Exits non-zero while poison remains, so a script that
    # re-scores from the archive can gate on it.
    ap.add_argument("--quarantine-scan", action="store_true",
                    help="report every STALE Q in _archive/ and log/okuda/ (reads only)")
    a = ap.parse_args()
    if a.quarantine_scan:
        sys.exit(1 if quarantine_scan()["n_stale"] else 0)
    vals = [float(x) for x in a.values.split(",")] if a.values else None
    sys.exit(run_round(mode=a.mode, frames=a.frames, batch=a.batch, base=a.base,
                       param=a.param, values=vals, dry=a.dry))
