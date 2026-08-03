#!/usr/bin/env python
"""round -- ONE round of the agentic discovery loop, with the real agents.

THE ROSTER IS ROLES.md, and `roles.py --check` compares it against this file in both directions.

  ACT 1 propose   Grounder -> Proposer(LLM) -> Peer-review(LLM) -> Critic -> Biologist
  ACT 2 measure   the batch runs -> Metrologist -> Analyst xN(LLM) -> Eye-check(LLM) -> Collector
  ACT 3 decide    Interpreter(LLM) -> Meta-review(LLM) -> Supervisor -> next round
  cross-run       Archivist: reads the whole history, may roll the search back to a better branch

Judge, Referee and Evolution were removed on 1 August. The first two were called ZERO times in
the live run -- we rank by a certified number, and a tournament of opinions about a measurement
we already hold is a step backwards. Evolution asked "what should change next?" and so did the
Meta-review; the surviving split is by SCOPE, Meta-review over this batch and the Archivist over
the whole history.

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
import glob
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
import archivist as ARCH                                                  # noqa: E402
import collector as COL                                                   # noqa: E402
import term as T_
T_.install_line_colour()   # [role] prefixes get their voice, wherever they are printed                                                         # noqa: E402
import diagnostician as DIAG                                              # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
CAMP = os.path.join(HERE, "campaign")
FRONTIER = os.path.join(CAMP, "frontier.json")
MAP = os.path.join(CAMP, "lever_map.jsonl")
# Per-round agent timing, one JSON line per round, so LLM cost can be tracked ACROSS rounds
# instead of scrolling past once in a terminal.
LLM_TIMING = os.path.join(CAMP, "llm_timing.jsonl")


# --------------------------------------------------------------------------- progress
# WHY BANNERS. A round emits a few hundred lines over twenty minutes and a reader watching the
# terminal cannot tell "thinking" from "wedged". These print at every act boundary and at every
# step that takes minutes, with a wall clock, so `tail -f` reads as progress rather than as noise.
_T0 = [None]


TRACE = os.path.join(CAMP, "trace.log")


def trace(where):
    """A breadcrumb that survives the process dying without a traceback.

    The recon round vanished after captioning with no exception and no further output, so the
    only thing known was "somewhere after the last print". stdout was not enough: it is buffered
    through nohup, and a SIGKILL takes whatever has not been flushed. This writes and FSYNCS a
    line per step, so the last line in trace.log is the last thing that actually ran.
    """
    line = f"{time.strftime('%H:%M:%S')}  {where}"
    print(f"  [trace] {where}", flush=True)
    try:
        os.makedirs(CAMP, exist_ok=True)
        with open(TRACE, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass


def act(title, detail=""):
    if _T0[0] is None:
        _T0[0] = time.time()
    el = time.time() - _T0[0]
    print(T_.act(title, detail, f"[+{int(el // 60):02d}:{int(el % 60):02d}]"), flush=True)


def step(msg):
    el = 0 if _T0[0] is None else time.time() - _T0[0]
    print(T_.step(msg, f"[+{int(el // 60):02d}:{int(el % 60):02d}]"), flush=True)


# --------------------------------------------------------------------------- frontier
def load_frontier(ledger=None):
    """Where the search breeds from. On a COLD campaign, the Archivist chooses it from the log.

    THE BUG THIS CLOSES IS NOT IN A ROUND, IT IS IN WHERE ROUNDS BEGIN. Every campaign started
    from a seed sphere plus the hand-written reference recipes and threw away sixty-odd finished
    runs sitting in log/okuda with their specs and their measured results. The best thing the
    project ever ran was never the thing the next campaign started from -- it was re-derived,
    badly, every time.

    So: an existing frontier is used as-is; an ABSENT one is not defaulted, it is CHOSEN, by the
    one role whose job is the whole history. If the Archivist cannot name a starting run, the old
    seed-plus-recipes fallback stands -- and the record says it was a fallback.
    """
    if os.path.exists(FRONTIER):
        import composition_space as CS
        raw = json.load(open(FRONTIER))
        out = [CS.CompositionGraph(ops=r["ops"], conns=r["conns"], params=r["params"])
               for r in raw]
        if out:
            return out

    print("[frontier] no frontier -- COLD START. Asking the Archivist what is already on disk.")
    try:
        pick = ARCH.cold_start(ledger=ledger)
        _st = pick.get("start") or []
        print(f"[archivist] start from {len(_st)}: {', '.join(_st) or 'nothing usable'}")
        if pick.get("why"):
            print(T_.quiet(T_.wrap_names([pick["why"][:200]])))
        graphs = [g for g in (_graph_from_run(nm) for nm in pick.get("start", [])) if g]
        if graphs:
            save_frontier(graphs)
            return graphs
        print("  [archivist] no usable run -- using reference recipes")
    except Exception as e:
        print(f"[archivist] cold start failed ({type(e).__name__}) -- using reference recipes")
    return [seed("substrate")] + list(reference_recipes().values())


def _graph_from_run(name):
    """Rebuild the composition a finished run was launched with, from its own spec on disk."""
    import composition_space as CS
    p = os.path.join(LOG, name, "composition.json")
    if os.path.exists(p):
        try:
            r = json.load(open(p))
            return CS.CompositionGraph(ops=r["ops"], conns=r["conns"], params=r["params"])
        except Exception:
            pass
    print(T_.quiet(f"  [archivist] {name}: no composition.json -- cannot be rebuilt as a graph"))
    return None


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


def _read_json_quiet(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


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

REVIEWS = os.path.join(CAMP, "peer_review.jsonl")


def _save_review(rid, review):
    os.makedirs(CAMP, exist_ok=True)
    with open(REVIEWS, "a") as fh:
        fh.write(json.dumps({"round": rid, **(review or {})}) + "\n")


def _last_review():
    """What Peer-review said about the LAST batch, for the Proposer that writes the next one."""
    if not os.path.exists(REVIEWS):
        return None
    rows = []
    for line in open(REVIEWS):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        return None
    r = rows[-1]
    iss = r.get("issues") or []
    if not iss and not r.get("verdict"):
        return None
    out = [f"On round {r.get('round')}'s batch, the reviewer said: "
           f"{(r.get('verdict') or '').strip()}"]
    for i in iss:
        out.append(f"  slot {i.get('slot')} [{i.get('severity')}]: {i.get('problem')}")
    out.append("These are the SAME KIND of mistake if you make them again. Do not repeat them.")
    return "\n".join(out)


# Per-run ceiling on the recorded trajectory. The instantaneous state is a rounding error; this
# is the array that scales with frames, and twelve slots share one GPU.
TRAJECTORY_BUDGET_GB = 1.5

# A HARD CEILING ON CELLS, above the memory clamp and independent of it.
#
# The memory budget alone gave 69,446 cells, and a run at that size takes over an hour per 900
# frames: five wk_curvature replays sat at 43k-50k cells and frame ~750 while eleven other slots
# had finished, and the round waited on them. A 50k body is not more informative than a 20k one
# for a map that is looking for a BUD -- Okuda's figures are hundreds of cells, not tens of
# thousands -- and the cost is superlinear while the information is not.
#
# This is a BUDGET decision, not a claim about biology. A composition that saturates it has told
# us its growth is unbounded on this recipe, which is a finding; it is not a reason to buy it a
# bigger array.
MAX_CELLS = int(os.environ.get("OKUDA_MAX_CELLS", 50_000))


def _times_censored(run, log_dir=None):
    """How often THIS composition has been stopped by its array, across every replay of it.

    Sizing that ignores its own history repeats it. The wk_* family was capped at 1778, resized,
    and capped again at 7686 -- and the second estimate was built from the same seed as the first,
    so it could not have known better. Counting the censorings makes the next estimate strictly
    larger than the one that just failed.
    """
    import glob
    log_dir = log_dir or LOG
    n = 0
    tail = run.split("_", 2)[-1][:12]
    for d in glob.glob(os.path.join(log_dir, f"*{tail}*")):
        try:
            j = json.load(open(os.path.join(d, "diag.json")))
            s = j.get("summary") or {}
            if s.get("buf_full") or s.get("div_blocked"):
                n += 1
                continue
            # ... and the plateau, because the engine's own flags are absent from every run made
            # before they existed, and were written from the LAST FRAME by every run made between
            # then and now. Counting only the flags read 0 censorings for a composition that has
            # visibly been stopped twice, and handed it the same ceiling that had just failed.
            import archivist as _AR
            if _AR._capped(d):
                n += 1
        except Exception:
            continue
    return min(n, 4)          # 8 * 2^4 = 128x; beyond that the composition, not the array, is wrong


def _reached_before(run):
    """(cells the run reached, whether it was CAPPED there). Measurement, not estimate."""
    try:
        j = json.load(open(os.path.join(LOG, run, "diag.json")))
        s = j.get("summary") or {}
        import archivist as _AR
        return s.get("n_cells_final"), bool(_AR._capped(os.path.join(LOG, run)))
    except Exception:
        return None, True          # unknown: treat as censored and be generous


def _resize_reservoir(spec_path, name, run=None, frames=None):
    """Give a replayed spec a buffer sized for where it is going. NOT a change to the experiment.

    THE VERTEX BUFFER IS AN ARRAY SIZE, NOT A MODEL PARAMETER. It sets how many cells the mesh
    CAN hold, and nothing about what the tissue does. The old specs were sized from a 150-cell
    start, so they cap at 1778 -- and 37 of the 75 runs on disk hit that ceiling and grew no
    further. Replaying one verbatim reproduces the ceiling, the Critic refuses it
    P2_BUFFER_SATURATED, and the slot is spent confirming an array is full.

    So the buffer is resized and everything else is copied untouched. This is the same experiment
    without the artefact -- and it is the one edit a replay may make, precisely because it is not
    part of the model. The specs on disk are NOT rewritten: they are the research record.
    """
    try:
        import yaml
        c = yaml.safe_load(open(spec_path))
        seed = next((o for o in c.get("operators", []) if o["op"] == "seed_mesh_3d"), None)
        if not seed:
            return
        # SIZE IT FROM WHAT THE RUN ACTUALLY REACHED, not from seed x 40.
        #
        # The blind headroom put 208,004 vertices behind the cfl_* specs -- a 104,004-cell ceiling
        # for runs that go 2000 -> 2000 and never divide at all. That is ~2.25 GB of recorded
        # trajectory each, for tissue that does not grow.
        #
        # We have measured every one of these runs, so the destination is known rather than
        # guessed: take what it reached last time and give it 4x room. A run that stopped AT its
        # ceiling reached an unknown destination, so that one keeps the generous estimate -- it is
        # the only case where the measurement is censored.
        clamped = None
        n_seed = int(seed.get("n_cells") or 0)
        reached, was_capped = _reached_before(run or name)
        if reached and not was_capped:
            # OBSERVED DESTINATION. It stopped on its own, so we know where it was going.
            want_v, want_c, target = T._reservoirs(max(int(reached / 2), n_seed), 0,
                                                   growth_headroom=8.0)
        elif reached:
            # CENSORED: it was STOPPED at `reached`, so that is a lower bound on where it was
            # going, not the destination. Sizing from the SEED is what failed -- wk_tension_pos
            # was capped at 1778, an estimate from its 150-cell seed gave a 7804 ceiling, and it
            # grew to 7686 and hit that too. Two stops, and the second estimate knew nothing the
            # first did not, because both were built from the same seed.
            #
            # So size from where it was stopped, and escalate every time the same composition is
            # censored again. That makes the sizing self-correcting instead of a fresh guess each
            # round: the next ceiling is always strictly above the one that just failed. Being
            # too generous costs memory, which is a fraction of a GB; being too tight costs the
            # entire run.
            n_cens = _times_censored(run or name)
            want_v, want_c, target = T._reservoirs(int(reached), 0,
                                                   growth_headroom=8.0 * (2 ** n_cens))
        else:
            want_v, want_c, target = T._reservoirs(n_seed, 0)
        # BOUNDED BY MEMORY, and hitting the bound is itself a finding. Escalation alone reached
        # 295,863 cells and 6.4 GB of trajectory per run -- 45 GB across a batch of seven, on a
        # 49 GB card. A composition that still saturates at this size is not going to be
        # understood by making the array bigger; the growth itself is the thing to question,
        # which is a Proposer's problem and not a buffer's.
        # THE FRAMES THE ROUND WILL RUN, not the ones the spec was written with. --frames
        # overrides the spec, so budgeting from the spec's count under-counts the trajectory: a
        # 400-frame spec run at 900 frames costs 2.25x what the budget was told.
        frames_est = int(frames or (c.get("general") or {}).get("n_frames") or 900)
        max_v = int(TRAJECTORY_BUDGET_GB * 1e9 / (max(frames_est, 1) * 3 * 4))
        # whichever binds first -- the memory budget or the cell ceiling
        max_v = min(max_v, 2 * MAX_CELLS + 4)
        if want_v > max_v:
            clamped = ((want_v + 4) // 2, (max_v + 4) // 2)
            want_v, want_c = max_v, min(want_c or max_v, max_v)
        have = ((c.get("sets") or {}).get("vertex") or {}).get("n") or 0
        if want_v <= have:
            return
        c.setdefault("sets", {}).setdefault("vertex", {})["n"] = want_v
        if want_c:
            c["sets"].setdefault("cell", {})["n"] = want_c
        yaml.safe_dump(c, open(spec_path, "w"), sort_keys=False)
        # ONE LINE PER DECISION. A clamp printed a second warning beside the resize, so a single
        # choice read as two problems -- and a warning that fires twice for one thing is how a
        # reader learns to skim warnings.
        # SHORT, AND THE CAVEAT ON ITS OWN LINE. This was one 230-character sentence per slot,
        # twelve of them per recon round, and the number a reader wants -- the new cell cap --
        # was buried in the middle. The reasoning is still worth having, so it goes underneath
        # rather than away, and only when the clamp actually fired.
        print(T_.warn(f"[recon] {name}: cap {(have + 4) // 2} -> {(want_v + 4) // 2} cells"
                      + (" (CLAMPED)" if clamped else "")))
        if clamped:
            print(T_.quiet(f"clamped from {clamped[0]} by the {TRAJECTORY_BUDGET_GB} GB budget"))
        # RECORDED, so an agent can read it. Until now the resize existed only as a printed line
        # and a local variable, which means the clamp's own conclusion -- "if it saturates again
        # the composition's growth is the problem, not the array" -- was a finding addressed to
        # nobody. That is the exact defect this campaign spent days removing everywhere else.
        try:
            os.makedirs(CAMP, exist_ok=True)
            with open(os.path.join(CAMP, "reservoir.jsonl"), "a") as fh:
                fh.write(json.dumps({"run": run or name, "slot": name, "from": have, "to": want_v,
                                     "cap_cells": (want_v + 4) // 2,
                                     "clamped_from_cells": clamped[0] if clamped else None,
                                     "times_censored": _times_censored(run or name)}) + "\n")
        except Exception:
            pass
    except Exception as e:
        print(T_.warn(f"[recon] {name}: resize failed ({type(e).__name__}), replaying verbatim"))


def _partial_evidence(nm):
    """How many frames a run managed before it was cut. 0 if it produced nothing."""
    try:
        import json as _j
        return len(_j.load(open(os.path.join(LOG, nm, "metrics.json"))).get("series") or [])
    except Exception:
        return 0


def _tail_of(path, chars=600):
    """The last paragraph of an append-only file -- what the agent just added to it."""
    try:
        txt = open(path).read().rstrip()
    except Exception:
        return ""
    blocks = [b.strip() for b in txt.split("\n\n") if b.strip() and not b.strip().startswith("#")]
    return " ".join(blocks[-1].split())[:chars] if blocks else ""


def _abstract_of(path):
    """memory.md's Abstract section: three sentences, rewritten every round by the Meta-review."""
    try:
        txt = open(path).read()
    except Exception:
        return ""
    import re as _re
    m = _re.search(r"^##\s+Abstract\s*$(.*?)(?=^##|\Z)", txt, _re.M | _re.S)
    if not m:
        return ""
    body = _re.sub(r"<!--.*?-->", "", m.group(1), flags=_re.S)
    return " ".join(body.split())[:600]


def _steer_for(sup):
    """The Supervisor's own words about what the next batch should be, or None.

    It has always produced this -- "surprise 0.00: the batch is confirming what we already
    believe, near-zero information. Push adversarial" -- and it has always gone to a terminal.
    """
    try:
        st = json.load(open(os.path.join(CAMP, "state.json")))
    except Exception:
        return None
    bits = [st.get("mix_why"), st.get("reason")]
    return " | ".join(b for b in bits if b) or None


def _paper_setup():
    """What the Grounder says the starting conditions are, in the paper's own words."""
    try:
        from agents.grounder import SETUP, setup
        s = setup("coupled")
        return (f"Okuda's coupling experiment (Figs 5-7, one experiment, {s['n_cells']} cells; "
                f"chi and gamma are what separate tubulation / branching / undulation):\n"
                f"  \u201c{s['quote'][:220]}\u2026\u201d\n"
                f"  Other cases available by naming `okuda_case`: "
                f"{', '.join(k for k in SETUP if k != 'coupled')}")
    except Exception as e:
        return f"(grounder unavailable: {type(e).__name__}: {str(e)[:80]})"


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



def _short(nm):
    """A run name a person can tell apart, for the terminal.

    This was nm[-12:] -- the LAST twelve characters -- which throws away the slot id and keeps
    the shared suffix, so r001n_02_cfl_c001p300_d and r001n_03_cfl_c001p300_d both printed as
    "l_c001p300_d". Every cfl replay in a batch looked like the same run. The slot id is the part
    that distinguishes them, so lead with it.
    """
    p = str(nm).split("_")
    return f"{p[0]}_{p[1]}" if len(p) > 2 and p[0].startswith("r") else str(nm)[:14]


def _read_one(nm, out_dir, ledger, claim=None):
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
    an = A.analyse(nm, out_dir, ledger=ledger, parallel=True)
    # AGAINST THE REGISTERED PREDICTION, not against the Reader's label.
    #
    # It was handed `analyst_consensus` -- a label derived from the same caption it then reads --
    # so it could catch a summariser drifting from a caption and could NOT catch the numbers
    # disagreeing with reality, which is the thing it has been credited with. It was checking a
    # reading against itself.
    #
    # Given the hypothesis's claim instead, it compares a picture with what was committed to
    # BEFORE the run, by a different agent, from different evidence. That is an independent
    # check, and it is the only one in the loop that looks at shape rather than at number.
    wa = A.watch(nm, out_dir, claim or an["analyst_consensus"], ledger=ledger)
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



def build_composition_batch(sup, cfg, n_slots, ledger):
    """Proposer(LLM) -> Critic -> Reflection(LLM). Returns [(graph, label, hyp_fields)]."""
    frontier = load_frontier(ledger=ledger)
    seen = seen_hashes(sup)
    lm = LeverMap(MAP)
    ledger_summary = _ledger_summary(sup, lm)

    # THE THREE RETURN PATHS, which is what Act 1 was missing. Every one of these was produced
    # correctly by some role and reached nobody: the Supervisor's steer went to a terminal, the
    # Critic's refusals went nowhere, and the Grounder wrote into a config the Proposer never
    # reads. Handed over here, in the words their authors used.
    steer = _steer_for(sup)
    refusals = "\n\n".join(x for x in (_refusal_summary(sup), _batch_refusals(),
                                        _reservoir_note()) if x)
    setup = _paper_setup()
    hist = ARCH.table()
    prior_review = _last_review()
    ok, slots = P.propose(frontier, cfg, sup.prox, ledger_summary, sup.round + 1,
                          n_slots=n_slots, ledger=ledger, steer=steer, refusals=refusals,
                          setup=setup, history=hist, review=prior_review)
    if not slots:
        print(T_.no(f"[round] no usable proposal ({len(slots or [])} slot(s) returned) -- "
                    f"not falling back to random. The reason is on the [proposer] line above."))
        return []

    out, rejected, in_batch = [], [], {}
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
        # THE STARTING CELL COUNT IS NOT AN AXIS OF THE SEARCH, so it is grounded on every slot.
        #
        # This line had two defects, both found by reading round 1's own configs. It split the
        # batch BY SLOT INDEX -- `i < 0.7 * len(slots)` -- and so ignored the `territory` the
        # Proposer had actually declared: slot 5 said `in_paper` and was denied Okuda's starting
        # conditions purely for being sixth. And what it varied was `n_cells`, so slots 4 and 5
        # ran 500 cells against a control at 2000. A difference between them then confounds the
        # EDIT with a fourfold change in specimen size, which destroys the one property the batch
        # was designed for: every slot a single edit off one shared control.
        #
        # An excursion is free to leave the paper's PARAMETER REGIME -- chi, gamma, the
        # diffusivities -- which is where "the paper is not the only place to stand" actually
        # lives. It is not free to change how much tissue it starts with, unless cell count is
        # itself the variable under test, and then it must be declared as the edit.
        sl["territory"] = sl.get("territory", "in_paper" if sl.get("intent") == "control"
                                 else "excursion")
        sl["fidelity"] = "okuda" if sl["territory"] == "in_paper" else "free"
        g = _ground_starting_conditions(g, sl)
        adm, rej = C.admit(g, seen if sl.get("intent") != "control" else ())
        if not adm:
            rejected.append((i, f"CRITIC: {rej}"))
            continue
        # THE SAME COMPOSITION TWICE IN ONE BATCH IS ONE EXPERIMENT, NOT TWO.
        #
        # `seen` holds compositions from PREVIOUS rounds, so a slot duplicated inside this batch
        # passes every gate: it is type-legal, its preconditions are met, and it is genuinely
        # unseen. Measured on round 2 of the batch-12 campaign -- only SEVEN distinct edits across
        # twelve slots, with `remove_op divide_3d0` proposed three times carrying MUTUALLY
        # CONTRADICTORY predictions on one composition:
        #
        #     slot 5  protr_peak 1.0-1.4      slot 7  protr_peak >= 1.3      slot 10  >= 1.5
        #
        # Five of twelve GPU runs would have bought nothing, and worse: two of those three resolve
        # `refuted` purely because the Proposer contradicted itself, which feeds a FALSE surprise
        # signal into the mixture the Supervisor sets from it. A replicate is a legitimate
        # experiment, but it is one the proposer must ASK for -- and it would carry one prediction.
        h = comp_hash(g)
        if h in in_batch:
            rejected.append((i, f"DUPLICATE_IN_BATCH: identical to slot {in_batch[h]} "
                                f"({h}). Same composition, so at most one prediction about it "
                                f"can be right; the others are the same experiment re-run under "
                                f"a different guess. Vary the EDIT, not the number."))
            continue
        in_batch[h] = i
        out.append((g, lbl, sl))
    # GROUPED BY REASON. Eleven consecutive lines each saying "identical to slot 0" is one
    # finding printed eleven times, and it buries the one line that mattered. The full list still
    # goes to batch_refusals.jsonl and into the next Proposer's prompt.
    _by_reason = {}
    for i, why in rejected:
        _by_reason.setdefault(str(why).split(":")[0].split("(")[0].strip(), []).append((i, why))
    for _reason, _items in _by_reason.items():
        _slots = ", ".join(str(i) for i, _ in _items)
        if len(_items) == 1:
            print(T_.no(f"[critic] slot {_slots} rejected -- {_items[0][1]}"))
        else:
            print(T_.no(f"[critic] {len(_items)} slots rejected ({_slots}) -- {_reason}"))
            print(T_.quiet(f"          {str(_items[0][1])[:150]}"))
    # PERSISTED, so the next Proposer is told. A refusal that reaches only the terminal is the
    # exact defect this campaign spent a day removing: the Proposer repeats the mistake because
    # nothing carried the reason back. These are the PRE-COMPUTE refusals -- the ones that cost
    # nothing to make and everything to repeat.
    if rejected:
        os.makedirs(CAMP, exist_ok=True)
        with open(os.path.join(CAMP, "batch_refusals.jsonl"), "a") as fh:
            fh.write(json.dumps({"round": sup.round + 1,
                                 "refused": [{"slot": i, "why": w} for i, w in rejected]}) + "\n")

    review = A.reflect([{k: v for k, v in s.items() if k != "edit"} for _, _, s in out],
                       ledger=ledger)
    print(T_.say("peer-review", f"batch_ok={review.get('batch_ok')}. {review.get('verdict','')}",
                 sentences=2))
    for iss in review.get("issues", []):
        _sev = iss.get("severity", "minor")
        _line = f"     slot {iss.get('slot')}: [{_sev}] {iss.get('problem')}"
        print(T_.yellow(_line) if _sev == "serious" else T_.dim(_line))
    # THE REVIEW IS KEPT, so the next round's Proposer can be handed it. It was printed and
    # dropped, and the cost of that was measured across the first two batches of the rebuilt
    # loop: Peer-review raised the SAME serious issue both times -- a confirmatory floor sitting
    # inside the control's own predicted band, so a positive cannot be told from the control --
    # and the Proposer repeated the design error because nothing carried the criticism back.
    # A reviewer whose reviews reach nobody is a reviewer measuring its own patience.
    _save_review(sup.round + 1, review)   # the round id is claimed after this returns
    # PEER-REVIEW ADVISES; IT CANNOT REFUSE. ROLES.md has said so since the rebuild, and this
    # `return []` said otherwise: an advisory role was returning "no candidates", which the round
    # reads as nothing runnable, which escalates, which exits 5 -- and two of those in a row
    # stopped the campaign. It was stopped by an opinion, not by a finding.
    #
    # The distinction is the whole reason there are two roles here. The CRITIC decides whether a
    # batch CAN be run and its refusals are enumerable, coded and arguable-with-never. The
    # reviewer decides whether a batch is WORTH running, which is a judgement, and a judgement
    # that can silently cost twelve GPU-hours by returning an empty list is a veto wearing an
    # adviser's name.
    #
    # Its issues are already saved and handed to the next round's Proposer, which is the
    # mechanism by which a review is supposed to change anything. That is slower than a veto and
    # it is accountable, which is the trade this design makes everywhere else.
    if review.get("batch_ok") is False:
        n_ser = sum(1 for i in review.get("issues", []) if i.get("severity") == "serious")
        print(T_.warn(f"[reflection] batch_ok=False with {n_ser} serious issue(s) -- RUNNING IT "
                      f"ANYWAY. Peer-review advises; only the Critic refuses. The issues are "
                      f"carried to the next Proposer."))
    return out


# --------------------------------------------------------------------------- LOOP II batch
def build_recon_batch(sup, cfg, n_slots, ledger):
    """ROUND 0: re-measure specs already on disk. No edits, no hypotheses, no predictions.

    The specs are copied VERBATIM into this round's config names. Nothing is grounded, nothing is
    swapped: the point is to learn what these compositions do under the instruments as they are
    now, and a spec altered on the way in would answer a different question.

    Every slot is posed with intent `recon` and `predicted` unstated, so `predict.score` resolves
    it `inconclusive` and it drops out of the surprise denominator -- which is correct. There is
    nothing to be surprised by in a replay, and a prediction attached to one would be a prediction
    about our own past arithmetic.
    """
    import shutil
    import composition_space as CS
    tab = ARCH.log_table(top=24)
    ok, choice = P.choose_specs(tab, n=n_slots, ledger=ledger)
    names = choice.get("runs") or []
    print(f"[recon] the Proposer chose {len(names)}:")
    print(T_.wrap_names(names))
    print(T_.say("proposer", choice.get("why", ""), sentences=2))

    out = []
    for i, run in enumerate(names[:n_slots]):
        src = os.path.join(LOG, run, "spec_run.yaml")
        if not os.path.exists(src):
            print(f"  [recon] {run}: no spec on disk -- skipped")
            continue
        out.append((run, src, {"intent": "recon", "track": "A",
                               "claim": f"re-measure {run} under the current instruments",
                               "metric": "protr_peak", "predicted": "unstated",
                               "why": choice.get("why", ""), "territory": "in_paper",
                               "source_run": run}))
    # THE FLOOR. A recon round exists to put SOMETHING measurable on the cluster; it has no
    # hypothesis to protect and nothing to be clever about. When the Proposer names nothing --
    # or names runs whose specs are not on disk -- the honest fallback is not an empty round, it
    # is to replay what IS there. On 3 August this returned empty and the campaign spent thirteen
    # rounds and 35 agent-minutes without launching a single job, while 63 runs with a spec and
    # 96 configs sat on disk unused.
    #
    # Deterministic: no model is consulted, and the pick is the Archivist's own ranking where it
    # exists, alphabetical after that, so the same disk gives the same batch.
    if len(out) < n_slots:
        have = {r for r, _, _ in out}
        ranked = [r.get("run") for r in (tab or []) if isinstance(r, dict) and r.get("run")]
        pool = [r for r in ranked if r not in have]
        # A RUN THAT EXECUTED FIRST, then a spec that merely exists. log/okuda/<run>/spec_run.yaml
        # is a composition we have already measured, so replaying it is a re-measurement with a
        # known prior; config/okuda/<name>.yaml is flat (no subfolders) and may never have run,
        # which is still far better than an empty round.
        pool += sorted(os.path.basename(os.path.dirname(p))
                       for p in glob.glob(os.path.join(LOG, "*", "spec_run.yaml"))
                       # a leading underscore is a diagnostic or quarantine directory, not a run
                       if not os.path.basename(os.path.dirname(p)).startswith("_")
                       and os.path.basename(os.path.dirname(p)) not in have
                       and os.path.basename(os.path.dirname(p)) not in ranked)
        pool += sorted(os.path.basename(p)[:-5]
                       for p in glob.glob(os.path.join(os.path.join(ROOT, "config", "okuda"), "*.yaml"))
                       if not os.path.basename(p).startswith(("_", "r0"))
                       and os.path.basename(p)[:-5] not in have)
        added = 0
        seen_pool = set()
        for run in pool:
            if len(out) >= n_slots:
                break
            if run in seen_pool:
                continue
            seen_pool.add(run)
            src = os.path.join(LOG, run, "spec_run.yaml")
            if not os.path.exists(src):
                src = os.path.join(os.path.join(ROOT, "config", "okuda"), f"{run}.yaml")
            if not os.path.exists(src):
                continue
            out.append((run, src, {"intent": "recon", "track": "A",
                                   "claim": f"re-measure {run} under the current instruments",
                                   "metric": "protr_peak", "predicted": "unstated",
                                   "why": "seeded from disk: the Proposer named too few usable "
                                          "specs, and a recon round with nothing to replay is a "
                                          "round that teaches nothing",
                                   "territory": "in_paper", "source_run": run}))
            added += 1
        if added:
            print(T_.warn(f"[recon] seeded {added} slot(s) from disk "
                          f"({len(out) - added} of the Proposer's {len(names)} were usable)"))
    if not out:
        print(T_.no("  [recon] nothing on disk carries a spec_run.yaml -- there is genuinely "
                    "nothing to replay"))
    return out


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

    _T0[0] = time.time()
    act("ACT 1 - PROPOSE")
    _purge_round_configs(rid, mode)    # F18: stale configs must not shadow this round's
    if mode == "theta":
        cands = build_theta_batch(base, param, values, batch)
    elif mode == "recon":
        cands = build_recon_batch(sup, cfg, batch, ledger)
    else:
        cands = build_composition_batch(sup, cfg, batch, ledger)
    if not cands:
        print("[round] no candidates -- escalating")
        _esc = run_escalation(cfg, sup, lm, rid, "the batch builder produced no "
                              "runnable candidate", ledger=ledger)
        print(T_.warn(f"[escalate] {_esc.get('action', 'no action')}"))
        return bk.finish(rid, "no_candidates", 5)

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
    if mode == "recon":
        # THE SPEC IS COPIED VERBATIM. No grounding, no translate: a spec altered on the way in
        # answers a different question than the one this round is asking.
        import shutil
        for i, (run, src, sl) in enumerate(cands):
            nm = f"r{rid:03d}n_{i:02d}_{run[:14]}"
            dst = os.path.join(ROOT, "config", "okuda", f"{nm}.yaml")
            shutil.copyfile(src, dst)
            _resize_reservoir(dst, nm, run=run, frames=frames)
            h = Hypothesis(hid=f"R{rid}.{i}.recon", comp_hash=f"RECON_{run[:20]}",
                           parent_hash=None, edit=f"replay {run}",
                           # A replay IS a control in the strict sense -- the composition
                           # unmodified -- and `intent` is a closed set for good reason: it is
                           # what the confirmatory/adversarial mixture is computed from, and a
                           # fourth value would silently leave that arithmetic.
                           intent="control", track=sl.get("track", "A"),
                           claim=sl["claim"], metric=sl["metric"], predicted="unstated",
                           rationale=sl.get("why", ""), round_id=rid)
            if not dry:
                sup.reg.pose(h)
            posed.append((nm, None, h))
            print(f"  {nm}  replay {run:24} track A  control       no prediction")
        if dry:
            print("\n[round] --dry: recon configs written, nothing submitted")
            return bk.finish(rid, "dry", 0)
    for i, (g, lbl, sl) in enumerate([] if mode == "recon" else cands):
        # F18: names carry round AND mode AND slot. A previous round's config with a matching
        # name could otherwise be picked up by a job -- silently running the wrong experiment.
        nm = f"r{rid:03d}{mode[0]}_{i:02d}_{comp_hash(g)[1:7]}"
        T.write_config(g, nm, frames=frames)
        # Defaulted BEFORE the hypothesis is built, because it is passed to it. It used to be
        # assigned to the slot dict two lines AFTER the constructor ran, purely to feed a print.
        sl["track"] = sl.get("track") or ("B" if sl.get("territory") == "in_paper" else "A")
        h = Hypothesis(hid=f"R{rid}.{i}.{comp_hash(g)[1:7]}", comp_hash=comp_hash(g),
                       parent_hash=None, edit=lbl,
                       intent=sl.get("intent", "confirmatory"),
                       track=sl["track"],
                       # CARRIED THROUGH AT LAST. claim_kind has existed in hypothesis.py since
                       # the loop was built, with a validator, and all 170 hypotheses in the
                       # ledger read "descriptive" because nothing ever passed it. A closed set
                       # nobody populates is not a type, it is a comment.
                       claim_kind=sl.get("claim_kind", "descriptive"),
                       revisits=sl.get("revisits") or "",
                       confounder=sl.get("confounder") or "",
                       claim=sl.get("claim", lbl),
                       metric=sl.get("metric", "protr_peak"),
                       predicted=sl.get("predicted", "unstated"),
                       rationale=sl.get("why", ""), round_id=rid)
        if not dry:
            sup.reg.pose(h)
        posed.append((nm, g, h))
        print(f"  {nm}  {lbl[:40]:40} track {sl['track']}  {h.intent:13} "
              f"predict {h.predicted[:34]}")

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
    # A1_NO_ABLATION, LIVE AT LAST. critic.check_batch has existed since the loop was built and
    # was never called -- so a necessity claim has never once been required to test itself. It
    # also had a latent false-refusal bug that only surfaced when it was finally exercised:
    # `add_op` names the OPERATOR (divide_3d) and `remove_op` names the NODE (divide_3d0), so the
    # two directions of the same experiment never matched and a claim whose ablation sat in the
    # same batch was refused anyway.
    _batch_bad = C.check_batch([h for _, _, h in posed])
    for _r in _batch_bad:
        print(T_.no(f"[critic] {_r.code}: {_r.detail[:150]}"))
    if _batch_bad:
        print(T_.warn("[critic] the batch makes a claim it does not test. Recorded, not refused: "
                      "A1 is live for the first time and its false-refusal behaviour is unproven "
                      "on real batches. It refuses from the round after its first clean pass."))

    act("ACT 2 - MEASURE", f"submitting {len(names)} simulation(s) -- the only expensive step")
    ids = cluster.submit(names, frames=frames, do_q=True, campaign=f"round{rid}")
    if not ids:
        # EXIT 6, NOT 1. A submission that did not land is a fact about the CLUSTER, not about
        # this code, and exit 1 is what Python gives an uncaught exception -- so the driver
        # printed "round 1 CRASHED (uncaught exception). That is a bug in the CODE, not a
        # finding about the batch" for an ssh/queue problem, and counted it toward MAX_CRASHES.
        # Twelve jobs were left running with nobody watching them.
        print(T_.no("[round] submission did not land -- aborting rather than scoring nothing. "
                    "This is a CLUSTER fact, not a crash: check ssh and the queue before "
                    "relaunching, and check `bjobs` for orphans from this attempt."))
        return bk.finish(rid, "submit_failed", 6)
    # The return value used to be discarded, so "all six finished" and "we waited 24 h and gave
    # up" were indistinguishable. A killed straggler is recorded and its hypothesis is resolved
    # `inconclusive` below (no diag.json), which keeps a degenerate slot out of the surprise rate
    # instead of scoring it as evidence.
    step(f"{len(ids)} job(s) on the cluster; waiting. Check with `bjobs`.")
    wait = cluster.wait_for_ids(ids, poll=60)
    step("batch finished; captioning the wave (one model load for all runs)")
    if not wait["ok"]:
        print(f"[round] batch did not complete cleanly: exit={wait['exit']} "
              f"killed={wait['killed']} timed_out={wait['timed_out']} -- scoring what landed")

    # ------------------------------------------------ caption the wave (one model load)
    # Must happen BEFORE the Analysts and the Watcher: both read description.txt, and a blind
    # Watcher cannot veto. This is where the cluster's missing `transformers` is worked around.
    # CAPTIONING MUST NOT BE ABLE TO KILL A ROUND -- and on the recon batch it did NOT, which is
    # worth writing down because the first diagnosis said it did.
    #
    # What actually happened: all six runs completed, caption_wave loaded its weights, and ALL SIX
    # description.txt files were written with real captions. Then the round process vanished with
    # no traceback and no further output -- the next `step()` line never printed. The captioner
    # was the last thing in the log, which made it look like the culprit; it had in fact finished
    # its job. CAUSE UNKNOWN. It is not memory: the devcontainer had 474 GB free and two 49 GB
    # A6000s for a 23 GB model.
    #
    # The guard below stays, because it is correct on its own terms -- a caption is a convenience
    # and must never be able to cost a round -- but it is NOT the fix for that failure, and
    # believing it was would leave the real one in place. This is the exact condition the
    # Diagnostician exists for and it should be pointed at the next occurrence.
    #
    # A caption is a CONVENIENCE: the Reader also has the numbers, the curve shapes and the strip.
    # Losing it costs the Eye-check its input and costs the Reader one of several sources. Losing
    # the round costs everything. So this is caught, said out loud, and the round continues
    # without captions -- which is what `description.txt` being absent already means downstream.
    trace("about to import caption_wave")
    try:
        from caption_wave import caption_wave
        trace("caption_wave imported; calling it")
        caption_wave(names)
        trace("caption_wave RETURNED")
    except BaseException as e:                       # BaseException: a MemoryError or a SIGKILL
        step(f"CAPTIONING FAILED ({type(e).__name__}: {str(e)[:80]}) -- continuing WITHOUT "
             f"captions. The Eye-check has nothing to read and will say so; the Reader still has "
             f"the numbers, the curve shapes and the strip.")

    # ---------------------------------------------------------------- ACT 2: measure, then read
    # `refused` is the other half of the round and used to exist only as terminal output. A round
    # that posed eight and admitted one must say so WITH THE REASONS, because "0 runs, coverage
    # 0%" is what the Proposer was handed, and from it drew the only sane conclusion available:
    # that the ledger was broken.
    trace("past captioning; entering the reading loop")
    step("reading each run: Biologist -> Metrologist -> Reader -> Eye-check")
    # WHICH HYPOTHESES HAVE BEEN RESOLVED THIS ROUND. The register is append-only and refuses to
    # resolve one twice -- correctly, since a rewritten verdict is not a record. Round 1 of the
    # 2 August campaign crashed on exactly that: `R1.11.recon already resolved as inconclusive`,
    # after the run had been refused by the Critic AND then read. The path that let it reach the
    # second resolve is not obvious from the code, so rather than guess, the round tracks what it
    # has already settled and skips it -- and says so, because a hypothesis reaching this twice
    # means a control-flow fault that should not be silent.
    resolved = set()
    rows, refused = [], []
    for nm, g, h in posed:
        d = os.path.join(LOG, nm, "diag.json")
        if not os.path.exists(d):
            if h.hid not in resolved:
                sup.reg.resolve(h.hid, {}, "inconclusive", note="no diag.json")
                resolved.add(h.hid)
            # A KILLED RUN IS TRUNCATED, NOT EMPTY. It has no diag.json because run_one writes
            # that at the end, but every frame it did produce is on disk in metrics.json -- and
            # those frames are evidence up to the cut, exactly as the frames before a reservoir
            # cap are. Five runs were killed as stragglers on 2 August and every frame they had
            # produced was discarded with them, which is the more expensive of the two mistakes.
            _partial = _partial_evidence(nm)
            if _partial:
                refused.append((nm, f"KILLED mid-run, but {_partial} frames survive on disk and "
                                    f"are evidence up to that point -- read metrics.json"))
            else:
                refused.append((nm, "no diag.json -- the run produced no record at all"))
            continue
        # THE ONLY DOOR the poisoned Q can come through in a round: a run's own diag.json. A
        # re-run of round N re-reads log/okuda/rNNNc_*/diag.json, and 14 of those on disk hold
        # the seed-sphere constant. Scrub at the read, so score_run / meets_success /
        # predict.score cannot reach it. The diag.json itself is untouched.
        summ = read_diag_summary(d, source=nm)
        # THE VERDICT TRAVELS WITH THE NUMBERS. `premises_broken` lives at the TOP LEVEL of
        # diag.json, and every consumer here reads the `summary` sub-dict -- so the rank key
        # below asked summ for it, got None, and sorted an extinct-chemistry run as if its
        # premises held. That is how the search came to breed from a dead field: the runs whose
        # activator had blown up to 1.4e6 and gone negative had the highest protr_peak, nothing
        # demoted them, and they became the frontier five rounds running.
        _dg = _read_json_quiet(d) or {}
        summ.setdefault("premises_broken", _dg.get("premises_broken") or [])
        summ.setdefault("premises", _dg.get("premises") or [])
        post = C.check_posthoc(summ)
        if post:
            if h.hid not in resolved:
                sup.reg.resolve(h.hid, summ, "inconclusive", note=f"NOT EVIDENCE: {post}")
                resolved.add(h.hid)
            print(T_.no(f"[critic] {nm} is not evidence: {post}"))
            refused.append((nm, f"critic post-hoc: {post}"))
            continue
        out_dir = os.path.join(LOG, nm)
        trace(f"reading {nm}")
        an, wa = _read_one(nm, out_dir, ledger,
                           claim=f"{h.claim} (predicted: {h.predicted})" if h.claim else None)
        try:
            import biologist as _B
            _pv = summ.get("premises") or []
            print(T_.say(f"biologist {_short(nm)}",
                         f"Specimen {_B.specimen_verdict(_pv)}"
                         + (f"; broken: {', '.join(summ.get('premises_broken') or [])}."
                            if summ.get("premises_broken") else "; every applicable premise holds."),
                         sentences=1))
        except Exception:
            pass
        trace(f"read {nm} OK")
        summ.update({k: v for k, v in an.items() if k != "analyst_reads"})
        summ.update(wa)
        # THE EYE-CHECK NO LONGER BLOCKS. Its verdict is recorded and carried into the record,
        # and it stops deciding whether a run may be ranked. Two reasons, and Cedric's call:
        #
        #   it is circular as built. `watch()` is handed `analyst_consensus` -- a label derived
        #   from the same caption it then reads -- so it can catch a summariser drifting from a
        #   caption and cannot catch numbers disagreeing with reality, which is what it has been
        #   credited with.
        #
        #   the camera was broken for the whole period the veto was trusted: one fixed viewpoint,
        #   and a zoom that re-fitted to the tissue every frame, so growth was drawn as shrinkage
        #   and a tube could sit behind the body. A blocker reading a broken instrument is worse
        #   than no blocker.
        #
        # A disagreement between the picture and the numbers is still worth having -- it is the
        # only thing in the loop that looks at SHAPE rather than at numbers. It is now an
        # observation, not a verdict.
        if wa.get("watcher_headline"):
            print(T_.say(f"eye-check {_short(nm)}", wa["watcher_headline"], sentences=1))
        if an.get("analyst_consensus"):
            print(T_.say(f"reader {_short(nm)}",
                         f"Phenotype {an['analyst_consensus']}"
                         + (f", {an.get('forced_or_grown')}" if an.get("forced_or_grown") else "")
                         + (f". {an.get('concern')}" if an.get("concern") else "."), sentences=2))
            # NAMED. This printed with an empty `who`, so the description arrived as a bare
            # "◉ :" -- four sentences of the most detailed observation in the round, attributed
            # to nobody and looking like a formatting fault.
            print(T_.say("eye-check", wa.get("watcher_describe") or wa.get("watcher_why"),
                         sentences=4))
        # THE RECORD DOES NOT DEPEND ON THE READER HAVING SPOKEN. Everything below
        # scores, resolves and RECORDS the run; it sat inside the `if` above, so a run
        # whose reader returned no label was dropped without a word.
        sc = score_run(summ, cfg)
        if wa.get("watcher_blocks"):
            print(T_.warn(f"[eye] {nm} DISAGREES -- recorded, not vetoed"))
            print(T_.quiet(T_.wrap_names([str(wa.get("watcher_why") or "")])))
        # `predict.score` refuses to guess: a prediction it cannot check resolves `inconclusive`
        # and drops out of the surprise denominator, rather than being recorded as `confirmed`
        # (which is what the old first-match regex did -- see predict.py P1/P2/P3).
        outcome, why = PR.score(h.predicted, summ, primary_metric=h.metric)
        if outcome == "inconclusive":
            print(f"[predict] {nm}: not checkable")
            print(T_.quiet(T_.wrap_names([str(why)[:180]])))
        # A quarantined Q must be visible in the SCIENTIFIC record, not only in the terminal:
        # `why` is what lands in hypotheses.jsonl, so anyone reading the resolution later sees
        # that the survival number was withheld rather than measured.
        if summ.get("Q_stale"):
            why = f"{why} | {summ['Q_stale_reason']}"
        if h.hid in resolved:
            print(T_.warn(f"[round] {h.hid} reached the resolver twice -- keeping the FIRST "
                          f"verdict. The record is append-only and this is a control-flow fault, "
                          f"not a re-judgement."))
        else:
            sup.reg.resolve(h.hid, summ, outcome, run_ids=[nm], note=why)
            resolved.add(h.hid)
        if g is not None:
            lm.add(comp_hash(g), g, an.get("analyst_consensus") or "unlabelled",
                   sc if np.isfinite(sc) else -1.0,
                   summ, nm)
        rows.append((nm, g, summ, sc, outcome, h))

    if not rows:
        return _abort(bk, sup, rid, mode, refused, posed, ledger)

    # ------------------------------------------------------------------------- rank (measure)
    # RANKED BY THE ADMITTED NUMBER. The Referee's tournament and the Judge's second opinion were
    # both removed on 1 August: they were called ZERO times in the live run, and that was not an
    # accident. Co-Scientist ranks by tournament because it performs no experiments and can only
    # debate its proposals; we measure, and where a certified number exists a tournament is a
    # worse ranker than the number. The Judge existed only to settle the Eye-check against that
    # number, and the Eye-check is now an observation that does not score. See ROLES.md.
    rows = sorted(rows, key=lambda r: (bool(r[2].get("premises_broken")), -r[3]))
    kept, dropped = truncate(rows, cfg.keep_truncate)
    print(T_.ok(f"[rank] kept {len(kept)}, dropped {len(dropped)} (never refined)"))
    for nm, g, s, sc, oc, h in rows:
        tag = "KEEP" if (nm, g, s, sc, oc, h) in kept else "drop"
        print(f"  [{tag}] {nm}  score {sc:6.2f}  protr_peak {s.get('protr_peak',0):5.2f} "
              f"phen={s.get('analyst_consensus','?'):9} watcher={s.get('watcher_verdict','?'):10}"
              f" [{oc}]" + ("  SURPRISE" if h.is_surprise else ""))

    # ---------------------------------------------------------------- ACT 3: COLLECT, then decide
    # THE COLLECTOR RUNS FIRST, and it is code. Everything downstream reads the RECORD rather than
    # rummaging for its own inputs -- which is the whole repair: a finding that is not collected
    # into a visible record disappears, and its disappearance is silent.
    # THE DIAGNOSTICIAN, before anything is interpreted. A round whose runs diverged has produced
    # evidence about an integrator, not about biology, and interpreting it is worse than useless:
    # the Interpreter would write a causal story about a configuration error. It also has standing
    # to STOP the campaign, because six more rounds cannot re-measure a broken apparatus into a
    # working one.
    trace("reading loop done; entering Act 3")
    _names = [nm for nm, _, _ in posed]
    _bad = [f["run"] for f in (DIAG.failure(n) for n in _names) if f["broke_at"] is not None]
    _sick = [nm for nm, _g, s, _sc, _oc, _h in rows
             if (s.get("premises_broken") or [])]
    if _bad or len(_sick) >= max(1, len(rows)) // 2 + (len(rows) % 2):
        step(f"Diagnostician: {len(_bad)} run(s) diverged, {len(_sick)} specimen(s) broken")
        dg = DIAG.diagnose(_names, ledger=ledger,
                           reason=f"round {rid}: {len(_bad)} diverged, {len(_sick)} unsound")
        print(T_.say("diagnostician", dg.get("headline") or dg.get("cause", "?"), sentences=1))
        print(f"      evidence : {dg.get('evidence','')}")
        print(f"      guard    : {dg.get('guard_to_add','')}")
        print(f"      action   : {dg.get('action')}")
        if dg.get("action") == "stop":
            rec = COL.collect_round(rid, mode, rows, refused=refused, posed=posed, aborted=True)
            rec["diagnosis"] = {k: v for k, v in dg.items() if k != "table"}
            rec["steer"] = f"APPARATUS FAULT: {dg.get('cause')} -- guard: {dg.get('guard_to_add')}"
            COL.write(rec)
            print("\n[round] STOPPING: the apparatus is at fault, not the biology. Add the guard.")
            # EXIT 4, NOT 3. Exit 3 means "two aborted rounds, the refusals are not actionable" --
            # a fact about the SEARCH. An apparatus fault is a fact about the CODE, it names
            # the guard to add, and it is fixed in one edit. Sharing a code made the driver
            # print "the refusal reasons are not actionable" for a diagnosis that was
            # precise and actionable, which is the same collision that once made a NameError
            # read as "a problem with the batch or the instruments".
            return bk.finish(rid, "apparatus_fault", 4)

    act("ACT 3 - DECIDE", f"{len(rows)} run(s) admitted, {len(refused)} refused")
    step("Collector: building the round record from the files on disk")
    trace("Collector: building the record")
    record = COL.collect_round(rid, mode, rows, refused=refused, posed=posed)
    trace("Collector done")
    # The holes are reported at WRITE time, not here. Checked here, the record has not yet been
    # given the round's steer -- that is set after the Supervisor observes, further down -- so
    # every round reported `HOLE: no steer recorded` by construction. A check that always fires
    # is a check nobody reads, and it would have buried a real hole among the noise.

    # THE INTERPRETER HAS NOTHING TO SAY ABOUT A REPLAY. A recon slot carries no graph -- it is a
    # spec re-run verbatim -- so there is no edit, no parent and no causal claim to write. It
    # crashed here on `g.name_region()` with g None, which is the honest signature of asking a
    # role a question its input cannot answer.
    # THE INTERPRETER RUNS BEFORE THE CHEAP ROLES CAN SPEND THE CEILING. It is the most
    # expensive role per call (round 10: 4 calls, 13.0 min, 39% of the spend) and it ran LAST,
    # so it was the role the budget dropped -- the loop paid for its deepest analysis and then
    # discarded it, five times in one round. Reserving its budget first means an overrun now
    # costs a summary rather than the causal record.
    interpretable = [r for r in kept if r[1] is not None]
    if interpretable and ledger is not None:
        try:
            ledger.reserve("interpreter", len(interpretable))
        except Exception:
            pass
    if interpretable:
        step(f"Interpreter: writing the causal record for {len(interpretable)} kept run(s)")
    else:
        step("Interpreter: skipped -- a replay has no edit and no causal story to tell")
    for nm, g, s, sc, oc, h in interpretable:
        _iok, _isaid = A.interpret(comp_hash(g), g.name_region(), h.edit, s,
                    {k: s.get(k) for k in ("analyst_consensus", "analyst_agreement")},
                    os.path.join(CAMP, "causal_descriptions.md"), ledger=ledger)
        # READ WHAT IT WROTE, not what it replied. An agent with Write tools puts its substance
        # in the file and returns a receipt -- "Wrote the entry" -- so printing the return value
        # shows that it ran and nothing of what it thought.
        print(T_.say(f"interpreter {_short(nm)}",
                     _isaid or _tail_of(os.path.join(CAMP, "causal_descriptions.md")),
                     sentences=1))

    # EVOLUTION WAS REMOVED on 1 August. It was asked "what should change next?" and so was the
    # Meta-review, and two agents answering the same question is not redundancy that costs a call
    # -- it is a roster nobody can reason about. The split that survives is by SCOPE, not by
    # phrasing: the Meta-review writes the lesson of THIS batch into the next round's prompts,
    # and the ARCHIVIST (to build) reasons over the WHOLE run history and may roll the search
    # back to a better branch. Local refinement of a winner inside the current branch was the
    # weakest of the three jobs and is the one dropped. See ROLES.md.

    sup.round = rid - 1          # observe() increments; the claim above already moved it
    # EVERY DOWNSTREAM CONSUMER OF A ROW WANTS ITS GRAPH, and a recon row has none -- it is a
    # spec replayed verbatim, with no composition object behind it. Rather than guard each call
    # site as it crashes (this is the second), the graphless rows are separated ONCE, here, and
    # every graph-consuming step below reads `graph_rows`.
    #
    # What the Supervisor, the lever map and the frontier all do with a row is place its
    # COMPOSITION in the search space -- cluster it, map it, breed from it. A replay has nothing
    # to place: its composition is already in the space, which is why it was chosen. Excluding it
    # is not losing evidence, it is declining to file the same composition twice.
    graph_rows = [r for r in rows if r[1] is not None]
    if len(graph_rows) != len(rows):
        print(f"  [round] {len(rows) - len(graph_rows)} replayed run(s) carry no composition "
              f"graph -- recorded as evidence, excluded from clustering and the frontier")
    rep = sup.observe([(g, s, h.hid) for _, g, s, _, _, h in graph_rows])
    sup.reg.render_knowledge(os.path.join(CAMP, "knowledge.md"),
                             ledger={"kept": len(kept), "dropped": len(dropped)}, round_id=rid)
    lm.render(os.path.join(CAMP, "lever_map.md"))
    step("Meta-review: prompt write-back + memory.md")
    _mok, _msaid = A.meta_review(rid, ledger=ledger,
                                 runs=[nm for nm, _, _, _, _, _ in rows])
    # THE RETURN VALUE IS READ NOW. `_mok` was captured and discarded, so a Meta-review that
    # failed or timed out left LAST round's memory in place and nothing said so -- the Proposer
    # then opened the next round on a document describing a round that never happened.
    if not _mok:
        print(T_.warn("[meta-review] FAILED or timed out -- memory.md still describes the "
                      "PREVIOUS round. The next Proposer will read a stale document."))
        COL.note_hole(rid, "meta_review_failed")

    # ------------------------------------------------------------------ LOGIC.md, enforced
    # Nothing in this loop has ever parsed memory.md: it reaches the Proposer as a path inside a
    # prompt, so every rule about what may be concluded was suasion. Two checks that already
    # existed and had never once fired now run here -- templates.check_memory (defined, called
    # only under __main__, with memory.md at 1186 words against a 900 budget) and logic.py's
    # modality/support/refuter gate. They REPORT rather than refuse: the claim register is not
    # yet the source memory.md is rendered from, so refusing here would stall a round over a
    # document the Meta-review cannot yet emit in the required form. Reporting makes the gap
    # visible every round instead of once a fortnight when somebody reads the file.
    try:
        import logic as LG
        import templates as TPL
        _mem = os.path.join(CAMP, "memory.md")
        if os.path.exists(_mem):
            _rows = LG.check_file(_mem)
            _neg = [r for r in _rows if r[1] in (LG.CANNOT_BE, LG.CANNOT_NOT_BE)]
            _bad = [r for r in _neg if not r[2]]
            _cb = [r for r in _rows if r[1] == LG.COULD_BE]
            print(T_.say("logic", f"{len(_rows)} claim(s): {len(_neg)} negative, "
                         f"{len(_bad)} unearned, {len(_cb)} could-be", sentences=1))
            for _n, _m, _ok, _rf, _nr, _tx, _why in _bad[:4]:
                print(T_.quiet(f"      unearned [{_m}] {_why}: {_tx[:64]}"))
            _w = len(open(_mem, errors="replace").read().split())
            if _w > TPL.MAX_MEMORY_WORDS:
                print(T_.warn(f"  [logic] memory.md {_w} words against a "
                              f"{TPL.MAX_MEMORY_WORDS} budget"))
            LG.write_report(rid, _rows, os.path.join(CAMP, "logic_report.jsonl"))

            # ------------------------------------------ the Metrologist, TRIGGERED not scheduled
            # A conclusion about a property nothing measures is not a softer verdict, it is a
            # REQUEST FOR AN INSTRUMENT. `morphology=sphere` was recorded for the run carrying the
            # finest Turing pattern in the campaign, because the shape was measured and the
            # pattern was not -- and no metric for wavelength, domain count or contrast exists, so
            # the variable that actually governs budding could not be represented, let alone
            # falsified.
            #
            # It files into the Backlog that ALREADY EXISTS (escalation.Backlog, duplicate_of()
            # included) rather than waking a model. The Metrologist is expensive and this is not
            # per-round work: it wakes only when the same gap has been asked for repeatedly, on
            # its own budget line, outside the round ceiling. Expect it to fire perhaps twice in
            # twenty rounds; certified once, the instrument is code that runs free forever.
            _gaps = {}
            for _row in _rows:
                for _p in LG.unmeasured_properties(_row[5]):
                    _gaps[_p] = _gaps.get(_p, 0) + 1
            if _gaps:
                _bl = ESC.Backlog(os.path.join(CAMP, "operator_requests.jsonl"))
                for _p, _n in sorted(_gaps.items(), key=lambda kv: -kv[1])[:3]:
                    _mech = f"an instrument reporting `{_p}`"
                    if _bl.duplicate_of(_mech):
                        continue
                    try:
                        _bl.file(ESC.OperatorRequest(
                            rid=_bl.next_rid(), round_id=rid, mechanism=_mech,
                            why_inexpressible=(
                                f"round {rid} drew {_n} conclusion(s) naming `{_p}`, and no "
                                f"admitted instrument reports it. The honest record is NOT "
                                f"MEASURED; concluding anything about it is the error that "
                                f"recorded the campaign's best Turing pattern as a null sphere."),
                            wanted_for=f"claims mentioning {_p} cannot be checked or falsified"))
                        print(T_.say("metrologist", f"instrument requested: {_p} -- {_n} "
                                 f"conclusion(s) rest on it and nothing measures it",
                                 sentences=1))
                    except Exception:
                        pass
    except Exception as _e:
        print(T_.warn(f"  [logic] check did not run: {type(_e).__name__}: {_e}"))
    # The Meta-review's product is memory.md, whose HEAD is a three-sentence abstract written
    # for exactly this: the campaign's position, stated so it can be read in one line.
    print(T_.say("meta-review", _msaid or _abstract_of(os.path.join(CAMP, "memory.md")),
                 sentences=1))
    step("Archivist: reading the whole history")

    # THE ARCHIVIST, over the whole history rather than this batch. It advises; the Supervisor
    # decides. Its recommendation is recorded either way, so an override is visible.
    arch = ARCH.decide(reason=f"end of round {rid}", ledger=ledger)
    print(f"  {T_.I['think']} [archivist] {T_.verdict(arch['decision'])}"
          + (f" -> {arch.get('target')}" if arch.get("target") else ""))
    print(T_.say("archivist", arch.get("headline") or arch.get("why", ""), sentences=1))
    record["archivist"] = arch
    record["steer"] = rep.get("mix_why", COL.MISSING)
    for hole in COL.holes(record):
        print(T_.warn(f"[collector] HOLE: {hole}"))
    COL.write(record)
    # THE CONTROL IS ALWAYS RETAINED, whatever it scored.
    # `kept` is a RANKING product, and a Watcher veto sets the score to -inf -- so in round 2 the
    # control (the parent, unchanged, protr_peak 4.03) was vetoed, fell out of `kept`, and the
    # frontier became exactly its two ablations (-extrude 1.39, -morphogen_growth_3d 1.03). The
    # search was then breeding from knockouts of a composition it no longer carried, and every
    # subsequent diff would be measured against a parent that is not in the pool. A veto is a
    # statement about what the MOVIE SHOWS, not about whether the composition is a useful parent;
    # it must not silently delete the reference the round's own diffs were taken against.
    # AN INVALID SPECIMEN IS NOT A PARENT. Its numbers describe the configuration and not a
    # tissue (ROLES.md), so breeding from it propagates a measurement of a dead field. Round 5
    # spent six of eight runs climbing exactly that gradient -- 1.317/5 tubes -> 1.340/10 ->
    # 1.529/30 -- while the activator was extinct in every one of them.
    front, barred = [], []
    for nm_k, g, s_k, _, _, _ in kept:
        if g is None:
            continue
        if (s_k.get("premises_broken") or []):
            barred.append((nm_k, s_k["premises_broken"]))
            continue
        front.append(g)
    for nm_k, br in barred:
        print(T_.warn(f"[frontier] {nm_k} is NOT a parent: specimen invalid ({', '.join(br)}). "
                      f"A high score on a broken specimen is a measurement of the configuration, "
                      f"not of a tissue."))
    ctrl = [g for _, g, _, _, _, h in graph_rows if h.intent == "control"]
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
        _esc = run_escalation(cfg, sup, lm, rid, rep["reason"], ledger=ledger)
        print(T_.warn(f"[escalate] {_esc.get('action', 'no action')}"))

    cov = lm.coverage()["overall"]
    # A DICT DUMPED TO A TERMINAL IS NOT A REPORT: the record is in supervisor.jsonl, and what a
    # person reading the round needs is the decision and the numbers behind it.
    #
    # EVERY FIELD IS `or 0` BEFORE IT IS FORMATTED. `rep.get("surprise", 0)` returns the DEFAULT
    # only when the key is ABSENT -- a key present and None returns None, and f"{None:.2f}" raises.
    # It did, at 43 minutes into a round whose measurement, reading and decision were all already
    # finished, and the driver recorded the whole round as a code crash. A terminal line must not
    # be able to destroy a round.
    _sup_bits = [str(rep.get("reason") or "continue")]
    if rep.get("next_confirmatory_frac") is not None:
        _sup_bits.append(f"next batch {int(100 * rep['next_confirmatory_frac'])}% confirmatory")
    if rep.get("surprise") is not None:
        _sup_bits.append(f"surprise {float(rep['surprise']):.2f}")
    if rep.get("best") is not None:
        _sup_bits.append(f"best {float(rep['best']):.2f}")
    print(T_.say("supervisor", "; ".join(_sup_bits), sentences=1))
    print(f"[map] coverage {cov['frac']:.0%} ({cov['covered']}/{cov['total']} cells, "
          f"{cov['n_runs']} runs)")
    return bk.finish(rid, "complete", 0)


# --------------------------------------------------------------------------- the abort path
def _abort(bk, sup, rid, mode, refused, posed, ledger):
    """A round that produced no evidence. It does NOT advance to Act 3 (ROLES.md).

    THREE THINGS, and each of them was got wrong before:

      IT IS NOT A ROUND. The counter does not advance and no coverage denominator grows. Counting
      it as one is exactly how a log came to tell the Proposer `coverage 0%` for a round in which
      eight simulations had died of diverged chemistry -- and the Proposer, handed an insane
      input, correctly concluded that the ledger was broken and fell back on its own prose memory.

      IT ROUTES BACK THROUGH THE ARCHIVIST, not straight back to the Proposer. Re-proposing inside
      the envelope the Critic just refused is how "route back to Act 1" becomes a week-long loop.
      The Archivist can move the frontier instead of retrying it.

      TWO IN A ROW STOPS THE CAMPAIGN. A second abort means the refusal reasons were not
      actionable, and that is a fact about US rather than about the search. Nothing downstream can
      discover that on its own, so the loop must say it and stop.
    """
    print(f"\n[round {rid}] ABORTED -- no admissible evidence from {len(posed)} posed run(s).")
    for nm, why in refused:
        print(f"    refused  {nm}: {why}")

    rec = COL.collect_round(rid, mode, [], refused=refused, posed=posed, aborted=True)
    arch = ARCH.decide(reason=f"round {rid} produced NO evidence -- "
                              f"{len(refused)} refusal(s): "
                              + "; ".join(w for _, w in refused[:4]), ledger=ledger)
    rec["archivist"] = arch
    rec["steer"] = (f"ABORT. Archivist says {arch['decision']}"
                    + (f" to {arch.get('target')}" if arch.get("target") else "")
                    + f": {arch.get('why','')}")
    COL.write(rec)
    print(f"  [archivist] {arch['decision']} -- {arch.get('why','')[:140]}")

    # The counter is rolled back: a round that produced nothing did not happen, scientifically.
    # The COMPUTE is still spent and still recorded -- that is what round_records.jsonl is for.
    sup.round = rid - 1
    sup._save()

    n_row = ARCH._aborts_in_a_row(ARCH.history())
    if n_row >= 2:
        print(f"\n[round] {n_row} ABORTED ROUNDS IN A ROW -- STOPPING THE CAMPAIGN.")
        print("  The refusal reasons are not actionable. That is a fact about the design of the")
        print("  search space or the envelope, not about the biology, and no agent in this loop")
        print("  can discover it. Read campaign/round_records.jsonl and decide.")
        return bk.finish(rid, "aborted_twice_stopped", 3)
    return bk.finish(rid, "aborted_no_evidence", 5)


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
    print(T_.say("supervisor", f"escalating: {action} -- {detail}", sentences=1))

    if action == "open_stage_gate":
        rec = sup.escalate()                       # advances cfg.stage_gate and checkpoints it
    elif action == "request_operator":
        req = A.request_operator(
            _ledger_summary(sup, lm), _map_summary(lm),
            "\n".join(f"  {comp_hash(g)}  {g.name_region()}" for g in frontier[:8]),
            f"{why}\n{detail}", rid, ledger=ledger)
        if not req or not req.get("why_inexpressible"):
            print(T_.warn("[escalate] no usable request -- recording the fact, not inventing one"))
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


def _reservoir_note(n=6):
    """What was done to the reservoirs, and what it means, for the Proposer.

    A composition CLAMPED at the memory budget has been given every array we can afford. If it
    saturates at that size the growth is unbounded, and no buffer will fix it -- that is a
    proposal problem, and the Proposer is the only one who can act on it.
    """
    p = os.path.join(CAMP, "reservoir.jsonl")
    if not os.path.exists(p):
        return ""
    rows, seen = [], set()
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["run"] not in seen:
            seen.add(r["run"])
            rows.append(r)
    rows = [r for r in rows if r.get("times_censored")][-n:]
    if not rows:
        return ""
    out = ["RESERVOIRS. The vertex buffer is an array size, not a model parameter, so a replay "
           "may enlarge it -- that removes an artefact rather than changing the experiment. "
           "These compositions have been stopped by their array before:"]
    for r in rows:
        line = (f"  {r['run']}: censored {r['times_censored']}x, buffer raised to "
                f"{r['cap_cells']} cells")
        if r.get("clamped_from_cells"):
            line += (f" -- CLAMPED from {r['clamped_from_cells']} by the memory budget. If this "
                     f"one saturates AGAIN, its growth is unbounded and no buffer will fix it: "
                     f"that is a composition to change, not an array")
        out.append(line)
    return "\n".join(out)


def _batch_refusals(n=8):
    """What the Critic refused BEFORE any compute, last round. Cheap mistakes, expensive to repeat."""
    p = os.path.join(CAMP, "batch_refusals.jsonl")
    if not os.path.exists(p):
        return ""
    rows = []
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        return ""
    last = rows[-1]
    out = [f"Last round the Critic refused {len(last['refused'])} slot(s) BEFORE they cost "
           f"anything. These are free to avoid and expensive to repeat:"]
    for r in last["refused"][:n]:
        out.append(f"  slot {r['slot']}: {r['why']}")
    return "\n".join(out)


def _refusal_summary(sup, n=12):
    """What was RUN and REFUSED, and why. The half of the evidence the Proposer never saw.

    THE DEFECT THIS CLOSES, diagnosed from the Proposer's own words on 2026-08-01. The lever map
    is filled AFTER the Critic, so a refused run contributes nothing to it -- and the ledger
    summary was built from the lever map alone. Eight simulations died (chemistry diverged, cell
    buffer saturated) and the Proposer was told only "0 runs, coverage 0%".

    It drew the sane conclusion from an insane input. Its own reasoning, verbatim:

        "Header shows round 3, 0 runs, coverage 0% ... another counter-reset artifact;
         real record = memory.md"
        "those two batches proposed ALL FOUR valid single-op edits on parent 2, so parent 2
         is fully PROPOSED and ... cannot be built on further"

    Two failures follow from one omission. It decided the LEDGER was broken and trusted its own
    prose memory instead -- which is the only evidence it had that anything had happened. And it
    counted territory as covered by what it PROPOSED rather than what it MEASURED, so it moved to
    a fresh parent believing the previous one was explored when nothing had been learned there.

    A loop that is told only about its successes cannot avoid repeating its failures. Refusals
    are evidence: "this composition diverges" is a fact about the space, and an expensive one.
    """
    try:
        rows = [h for h in sup.reg.all() if getattr(h, "outcome", None) == "inconclusive"]
    except Exception:
        return ""
    rows = rows[-n:]
    if not rows:
        return ""
    import collections
    why = collections.Counter()
    for h in rows:
        note = str(getattr(h, "note", "") or "")
        code = "no diag.json (run died or was killed)"
        for c in ("P3_CHEMISTRY_DIVERGED", "P2_BUFFER_SATURATED", "P1_INERT_OPERATOR"):
            if c in note:
                code = c
                break
        why[code] += 1
    out = ["", f"REFUSED RUNS -- {len(rows)} recent attempts produced NO admissible evidence.",
           "These are not missing results. They ran, and the Critic rejected them:"]
    for code, k in why.most_common():
        out.append(f"  {k:>3} x  {code}")
    out += ["",
            "READ THIS CORRECTLY: an empty map beside a non-zero attempt count is NOT a reset",
            "counter and NOT a lost record. It means the compositions proposed so far cannot be",
            "simulated. A parent is EXPLORED when its edits produced evidence, never when they",
            "were merely proposed -- so a family whose whole batch was refused is still unmapped,",
            "and reproposing near it will be refused for the same reason.",
            "The refusal codes above say what to change: DIVERGED means the chemistry settings",
            "leave the integrable region; SATURATED means the run outgrew its cell reservoir."]
    return "\n".join(out)


def _ledger_summary(sup, lm):
    cov = lm.coverage()
    solo = lm.solo()
    lines = [f"round {sup.round}, {cov['overall']['n_runs']} runs WITH ADMISSIBLE EVIDENCE, "
             f"map coverage {cov['overall']['frac']:.0%}",
             f"phenotypes so far: {lm.phenotypes()}", "", "solo effects (Δscore, verdict):"]
    for op, v in sorted(solo.items(), key=lambda kv: -(kv[1].get('delta') or -99))[:10]:
        lines.append(f"  {op:24} {v.get('delta','—')}  {v['verdict']}")
    inter = [(k, v) for k, v in lm.pairs().items()
             if v.get("verdict") in ("SYNERGY", "ANTAGONISM")]
    if inter:
        lines += ["", "interactions found:"]
        lines += [f"  {k}: {v['verdict']} ({v['interaction']:+})" for k, v in inter[:6]]
    ref = _refusal_summary(sup)
    if ref:
        lines.append(ref)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["composition", "theta", "recon"], default="composition")
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
