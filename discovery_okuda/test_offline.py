#!/usr/bin/env python
"""test_offline -- the loop, exercised without an agent and without a cluster.

WHAT THIS IS FOR. Every defect closed on 3 August was found by launching: ten agent-minutes and
half an hour of GPU to learn that a display name had been read as an identifier. The worst of them
-- `prompt.split("BREVITY")[0]`, which delivered 331 characters of a 16,093-character prompt and
so removed the legal-move menu, the refusals, the history and the output schema -- survived four
live rounds and was found by this harness on its first honest run.

Each case below is a failure we have actually watched happen. They are regression tests in the
strict sense: every one of them cost a round.

    python test_offline.py            # all cases
    python test_offline.py -v         # with the round's own output
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "agents")]

VERBOSE = "-v" in sys.argv
CASES, FAILED = [], []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


@contextlib.contextmanager
def quiet():
    if VERBOSE:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


# --------------------------------------------------------------------------- the prompt seam
@case("the whole prompt reaches the model")
def t_prompt_intact():
    """THE 3 AUGUST DEFECT. `split("BREVITY")[0]` cut the Proposer's prompt at character 331.

    The agent then said "I lack the injected LEGAL MOVES menu (this call omitted it)" and was
    disbelieved. It was telling the truth, and four rounds were spent on the consequences.
    """
    import offline as O
    O.install("clean")
    import round as R, proposer as P, llm
    cap = {}
    real, llm.run_claude = llm.run_claude, lambda p, **k: (cap.__setitem__("p", p), (True, "{}"))[1]
    try:
        with quiet():
            P.propose(R.load_frontier(), R.CampaignConfig(batch=6, keep_truncate=2),
                      None, "", 4, n_slots=8)
    finally:
        llm.run_claude = real
    p = cap.get("p", "")
    check(len(p) > 5000, f"prompt reaching the model is only {len(p)} chars -- it is being cut")
    check("LEGAL MOVES" in p, "the legal-move menu never reached the model")
    check(len(re.findall(r"parent_index=\d+\s+edit=", p)) >= 8,
          "fewer than 8 legal moves in the prompt -- the menu is smaller than any batch")
    check("slots" in p, "the output schema never reached the model")


@case("no path reaches the real CLI")
def t_no_subprocess():
    """A harness that quietly spends money is worse than none. The first version cost $0.37."""
    import subprocess
    import offline as O
    O.install("clean")
    import llm
    real, subprocess.Popen = subprocess.Popen, _boom
    try:
        with quiet():
            ok, out = llm.run_agent("proposer", "LEGAL MOVES\nparent_index=0  edit=[\"add_op\", "
                                                "\"divide_3d\", \"hertwig\"]  -> x", n_slots=4)
        check(ok, "the faked agent reported failure")
    finally:
        subprocess.Popen = real


def _boom(*a, **k):
    raise AssertionError("a REAL subprocess was launched -- the offline seam leaks")


# --------------------------------------------------------------------------- the gates
@case("a clean round fills its batch and reaches Act 3")
def t_clean_round():
    code, log = _round("clean", batch=6)
    check("ACT 3" in log, "the round never reached Act 3")
    d = _attrition(log)
    check(d and d["delivered"] >= 1, f"no slots delivered: {d}")


@case("phenotype edits are refused, then repaired in the same round")
def t_repair():
    """Round 2 proposed `add branching` sixteen times and delivered ONE job of twelve.

    The Critic's refusals are mechanical, so there is a right answer the Proposer could have
    given -- which is why this is repaired in-round rather than next round.
    """
    code, log = _round("phenotypes", batch=6)
    check("unknown edit" in log or "not applicable" in log,
          "the phenotype edits were not refused at all")
    check("repair pass" in log, "no repair pass ran after a batch was gutted")
    d = _attrition(log)
    check(d and d["delivered"] > 1,
          f"the repair pass did not recover the batch: {d}")


@case("a proposal under the wrong key is still read")
def t_wrong_key():
    """`candidates` instead of `slots` silently discarded 12 experiments every round."""
    code, log = _round("wrong_key", batch=6)
    d = _attrition(log)
    check(d and d["proposed"] >= 1, f"the batch under `candidates` was discarded: {d}")


@case("a reply with no JSON does not fabricate a batch")
def t_no_json():
    """Silence must read as refusal. A random batch is not a substitute for a reasoned one."""
    code, log = _round("no_json", batch=6)
    check("no usable proposal" in log or "no proposal.json" in log or _attrition(log),
          "a proposal that was never written did not produce an honest refusal")


@case("REJECT with no flag is still a rejection")
def t_no_flag():
    """peer-review printed `batch_ok=None. REJECT` and nothing anywhere reacted."""
    import offline as O
    O.install("no_flag")
    review = {"verdict": "REJECT -- the batch repeats round 1", "batch_ok": None}
    if review.get("batch_ok") is None:
        v = str(review.get("verdict", "")).upper()
        if "REJECT" in v or "REVISE" in v:
            review["batch_ok"] = False
    check(review["batch_ok"] is False, "a REJECT verdict still reads as no verdict")


@case("a truncated run name resolves to its run")
def t_truncated():
    """`run[:14]` is a DISPLAY name. Six of twelve recon picks were dropped for being it."""
    import glob
    LOG = os.path.join(HERE, "..", "log", "okuda")
    dirs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(LOG, "*", "spec_run.yaml")))
    long = [d for d in dirs if len(d) > 14]
    if not long:
        return "skipped -- no run on disk has a name longer than 14 characters"
    src = long[0]
    hits = [d for d in dirs if d.startswith(src[:14])]
    check(len(hits) >= 1, f"the truncation of {src!r} resolves to nothing")


@case("parameter moves are offered, and keep the composition's identity")
def t_set_param():
    """Track A asks two questions of a mechanism; only the first has ever been askable.

    `--mode theta` existed from the first draft and plan() never emitted it, so "what does this
    do as you turn it up" has not once been asked in a live round. As a menu move it can be, and
    can be MIXED with structural moves in one batch.
    """
    import offline as O
    O.install("clean")
    import round as R, proposer as P, run_record as RR
    menu = P._legal_menu(R.load_frontier(), R.CampaignConfig(batch=12, keep_truncate=4), None)
    rows = [r for p_ in menu for r in p_["legal_edits"]]
    kinds = {r["edit"][0] for r in rows}
    check("set_param" in kinds, f"no parameter move is offered at all; kinds={kinds}")
    struct = [r for r in rows if r["edit"][0] != "set_param"]
    param = [r for r in rows if r["edit"][0] == "set_param"]
    check(len(struct) >= len(param),
          f"the menu is mostly dials: {len(param)} parameter vs {len(struct)} structural moves")
    # THE DISCIPLINE, mechanically: a retune must not read as a new mechanism.
    g = R.load_frontier()[0]
    e = next(r["edit"] for r in rows if r["edit"][0] == "set_param")
    child, _ = g.apply(tuple(e))
    check(RR.comp_hash(child) == RR.comp_hash(g),
          "a set_param move changed the composition hash -- a retune would count as a new mechanism")


@case("our own measured runs can be frontier parents")
def t_own_runs():
    """THE STANDARD from_preset states: a space that cannot express our own evidence is wrong.

    R5_PARAM_OUT_OF_RANGE was withdrawn on 3 August because the boxes were hand-written and the
    runs fall outside them -- cell_diffuse0.d_h = 2.0 against a ceiling of 0.346, on a run that
    produced valid evidence. Before that, ZERO of our measured runs could serve as a parent.
    """
    import io, contextlib
    import offline as O
    O.install("clean")
    import round as R, critic as C
    names = ("wk_null_s0", "coral_fixed_ball", "cfl_c000p080_d002p000", "wk_pressure_pos_s0")
    ok = 0
    for n in names:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            g = R._graph_from_run(n)
        if g and C.admit(g, ())[0]:
            ok += 1
    check(ok >= 3, f"only {ok} of {len(names)} measured runs are admissible as parents -- the "
                   f"search space does not contain our own evidence")


@case("a tissue that meets its array late is censored, not voided")
def t_p13_censored():
    """P13's own docstring says AMBIGUOUS, NOT INVALID -- and it returned `fail` anyway.

    That put it in premises_broken, marked the specimen invalid, and barred the run from the
    frontier: five of twelve slots on 3 August, the whole wk_* family, discarded on the strength
    of where each run ENDED while the eye-check had read real trajectory in every one.
    """
    import numpy as np
    import biologist as B

    def series(n_grow, n_flat, top=1778.0):
        vals = list(np.linspace(150, top, n_grow)) + [top] * n_flat
        return [{"cells": float(v)} for v in vals]

    late = B.p13_growth_not_capped_by_the_array({}, series(700, 200))
    check(late.status == "censored",
          f"a run that grew for 78% of its length before meeting the array is {late.status!r}, "
          f"not censored -- its trajectory is being thrown away")
    early = B.p13_growth_not_capped_by_the_array({}, series(200, 700))
    check(early.status == "fail",
          "a run whose plateau covers the whole run must still fail -- there is nothing to censor")
    clean = B.p13_growth_not_capped_by_the_array({}, series(900, 0))
    check(clean.status == "pass", "a run that never saturates must pass")
    # and censored must NOT count as broken, or nothing is gained
    check("censored" not in ("fail", "error"), "censored must not be a broken status")


# --------------------------------------------------------------------------- the declarations
@case("every artifact has a reader (wiring.py --check)")
def t_wiring():
    """An artifact written for nobody is work the loop does every round for no one."""
    import wiring
    bad = wiring.check()
    check(not bad, f"{len(bad)} wiring complaint(s); first: {bad[0] if bad else ''}")


@case("a new orphan artifact is caught")
def t_wiring_bites():
    """THE RULE MUST BITE. A checker that passes because it checks nothing is worse than none.

    So: declare an artifact with no reader and require the checker to object. Without this, the
    green tick above would be indistinguishable from a parser that matched no rows at all.
    """
    import wiring, tempfile, os as _os
    fd, tmp = tempfile.mkstemp(suffix=".md")
    with _os.fdopen(fd, "w") as fh:
        fh.write("| artifact | writer | readers |\n|---|---|---|\n"
                 "| `orphan_test.jsonl` | `round.py` |  |\n")
    try:
        dec = wiring.declared(tmp)
        check("orphan_test.jsonl" in dec, "the WIRING.md table parser matched no rows")
        check(not dec["orphan_test.jsonl"][1], "a reader-less row was parsed as having readers")
    finally:
        _os.unlink(tmp)


@case("every admitted metric has a producer")
def t_metrics_have_producers():
    """AN ADMITTED METRIC WITH NO PRODUCER IS A SILENT INCONCLUSIVE.

    `predict.Clause.check` looks the metric up by EXACT KEY in the run's summary, so a name that
    is admitted but never written scores `not measured`, the prediction falls to `inconclusive`,
    and nothing anywhere reports a fault -- the agent wrote a perfectly good falsifiable claim
    and the loop recorded that it had learned nothing. Measured 2026-08-04: `corr_act_rad`,
    `r_cv` and `protr_p99` had been admitted for weeks with no producer anywhere in the codebase,
    and `corr_act_rad` is the metric that answers the campaign's own question.

    This is the metric-bank twin of WIRING.md's rule for artifacts:
    A METRIC MUST BE ADMITTED WITH A PRODUCER.

    THE RULE DID NOT CHANGE ON 4 AUGUST; WHAT A NAME IS DID. The bank is now a PRODUCT --
    24 quantities x 6 temporal reductions -- so `act_cv_peak` has TWO producers and needs both:
    something must compute the per-frame column `act_cv`, and something must apply `_peak` to it.
    Checking only the whole name would pass nothing (no file assigns `act_cv_peak`); checking only
    the quantity would pass a bank of 144 names of which four reductions in six were never
    written, which is 96 silent inconclusives. So both halves are checked, and
    `predict.decompose` -- the one place that knows how a name is built -- does the splitting.
    """
    import glob as _glob
    from predict import KNOWN_METRICS, SUFFIXES, decompose
    # HALF TWO FIRST, because it is one line and it fails loudest: the six reductions must all be
    # computed, under exactly these names, by the module that owns them.
    import time_analysis as _TA
    check(tuple("_" + r for r in _TA.REDUCTIONS) == tuple(SUFFIXES),
          f"predict admits {SUFFIXES} but time_analysis produces {_TA.REDUCTIONS}: every name "
          f"built from a suffix nobody computes is a silent `not measured`")
    _got = _TA.reduce_series([1.0, 2.0, 3.0, 4.0, 5.0], frames=[0, 1, 2, 3, 4])
    _miss = [s for s in SUFFIXES if s[1:] not in _got]
    check(not _miss, "reduce_series does not return: " + ", ".join(_miss))
    src = ""
    for d in (HERE, os.path.join(os.path.dirname(HERE), "prototype", "Tyssue")):
        for f in sorted(_glob.glob(os.path.join(d, "*.py"))):
            if os.path.basename(f) in ("predict.py", "test_offline.py"):
                continue
            src += open(f).read()
    # ASSIGNMENT, NOT MENTION. The first version of this test accepted the name appearing in
    # quotes anywhere, and passed green while `wavelength_cells_final` had no producer at all --
    # the string occurs in run_one's lift list, which is a CONSUMER. A test that a name is
    # mentioned somewhere is not a test that anything writes it. So: require a real write --
    # `out["k"] = `, `dict(k=`, `k=round(...)` -- and not membership of a tuple or list.
    import re as _re
    orphan, undecomposable = [], []
    for m in KNOWN_METRICS:
        base, _suf = decompose(m)
        if base is None:
            undecomposable.append(m)
            continue
        base = base[:-6] if base.endswith("_final") else base       # ta_tube_len_final
        base = base[3:] if base.startswith("ta_") else base
        e = _re.escape(base)
        written = _re.search(rf'''\[["']{e}["']\]\s*=''', src) or \
                  _re.search(rf'''["']{e}["']\s*:''', src) or \
                  _re.search(rf'''(?<![\w.])(?<!["']){e}\s*=(?!=)''', src)
        if not written:
            orphan.append(base)
    check(not undecomposable, "admitted but not <quantity><suffix> nor a declared scalar: "
                              + ", ".join(undecomposable))
    check(not orphan, "admitted with no producer: " + ", ".join(sorted(set(orphan))))


@case("a prediction naming a new metric is actually scorable")
def t_new_metrics_scorable():
    """The producer test proves the name is WRITTEN somewhere. This proves the scorer can READ
    it: parse a real prediction over the new metrics and check it resolves against a summary.

    NOW OVER THE PRODUCT NAMES, because that is what the bank contains. The old version of this
    case was written against bare names (`act_cv > 0.3`) and it is kept, at the bottom, as the
    ALIAS case: the archive is full of bare-name predictions and the convention has always been
    that a bare name is the summary value at its last frame, so the parser resolves it to
    `_final` rather than dropping it -- and the summary key it is checked against is the resolved
    one, so nothing is silently reinterpreted.
    """
    import predict as PR
    pred = "act_cv_peak > 0.3, corr_act_rad_final > 0.4, gyr_prolate_span 1.5-4.0"
    cl = PR.parse(pred)
    check(len(cl) == 3, f"parsed {len(cl)} of 3 clauses from {pred!r}")
    check([c.metric for c in cl] == ["gyr_prolate_span", "act_cv_peak", "corr_act_rad_final"],
          f"a suffix was mis-split: {[c.metric for c in cl]}")
    obs = {"act_cv_peak": 0.62, "corr_act_rad_final": 0.71, "gyr_prolate_span": 2.4}
    out, why = PR.score(pred, obs)
    check(out == "confirmed", f"expected confirmed, got {out}: {why}")
    out2, _ = PR.score(pred, {**obs, "act_cv_peak": 0.01})
    check(out2 == "refuted", f"a dead pattern must refute, got {out2}")
    # A REDUCTION THE OLD BANK COULD NOT SAY: the field lived and then died, in one clause pair.
    flash = "act_cv_peak > 0.5 and act_cv_final < 0.1"
    check(PR.score(flash, {"act_cv_peak": 0.9, "act_cv_final": 0.02})[0] == "confirmed",
          "a flash-then-extinction is not scorable")
    # ...and the refusal fraction, which is what makes a null on corr_act_rad readable at all.
    check(PR.score("corr_act_rad_measured_frac >= 0.9",
                   {"corr_act_rad_measured_frac": 0.046})[0] == "refuted",
          "`the coupling was measurable throughout` must be refutable")
    # the alias, and the boundary of the alias
    check(PR.score("act_cv > 0.3", {"act_cv_final": 0.62})[0] == "confirmed",
          "a bare quantity name no longer resolves to anything")
    check(PR.score("act_sd_peak > 0.3", {"act_sd_peak": 0.9})[0] == "inconclusive",
          "a WITHDRAWN quantity must not score")


@case("every admitted metric has exactly one group")
def t_metrics_grouped():
    """A FLAT LIST IS A LIST YOU READ THE FIRST ITEM OF.

    Forty metric names arrived in the prompt as one comma-separated line, and round 2 wrote all
    twelve of its predictions on `protr_peak` while act_cv, corr_act_rad and act_alive_frac sat
    unnamed. Grouping them by the QUESTION they answer -- is it a tube, is it still cells, is
    there a pattern, does the pattern grip the shape, is this evidence at all -- is what makes
    "you have not asked whether there was a pattern" visible to the role writing the claim.

    Exactly one group, enforced: a metric with no home is silently absent from every prompt, and
    one in two groups is a category that is not a category.
    """
    import collections
    import predict as PR
    ok = [m for m in PR.KNOWN_METRICS if m not in PR.REJECTED_METRICS]
    grouped = [m for v in PR.METRIC_GROUP.values() for m in v]
    homeless = [m for m in ok if m not in grouped]
    check(not homeless, "admitted with no group: " + ", ".join(homeless))
    stale = [m for m in grouped if m not in ok]
    check(not stale, "grouped but not admitted: " + ", ".join(stale))
    twice = [m for m, n in collections.Counter(grouped).items() if n > 1]
    check(not twice, "in more than one group: " + ", ".join(twice))
    block = PR.admitted_block()
    for grp in PR.METRIC_GROUP:
        check(grp in block, f"group {grp!r} never reached the prompt")


@case("the naming convention holds: no _final twin of a bare metric")
def t_no_final_twins():
    """A METRIC BANK IS A VOCABULARY. Lifting every series metric into the summary twice -- once
    bare, once `_final` -- minted thirteen names for quantities that already had one: `act_cv`
    and `act_cv_final` were the same number, in a list an agent must read before it can write a
    prediction. Doubling a vocabulary without adding a quantity makes it harder to use and
    measures nothing new.

    THE EXCEPTION THAT USED TO BE ALLOWED HERE IS GONE, and its going is the point. `protr_final`
    was carved out because it was "the horizon-truncated one, unlike bare protr" -- an exception
    resting on a distinction that existed nowhere in the code. Every name in the bank is now a
    quantity plus one of six reductions, all six truncated at the horizon, so the carve-out has
    nothing left to protect: no bare quantity is admitted at all, and `protr_final` is simply
    protr under `_final`.

    Two rules, both of which have already been broken in this repo:
      * a bare quantity may not be admitted beside its own reductions (the thirteen twins);
      * a RUN-LEVEL SCALAR may not be SPELLED like a reduction. `n_cells_final`, `ta_n_tubes_final`
        and `spot_frac_final` all were, and `n_cells_final` is currently written by two producers
        with two meanings -- run_one's last frame (3,975 on okuda_route) and reduce_all's value at
        the horizon (~2,300). A name that looks like a reduction will be read as one.
    """
    import predict as PR
    ok = set(PR.KNOWN_METRICS)
    twins = [m for m in ok for s in PR.SUFFIXES
             if m.endswith(s) and m[: -len(s)] in ok]
    check(not twins, "both a bare name and its reduction are admitted: " + ", ".join(twins))
    bare = [q for q in PR.SERIES_QUANTITIES if q in ok]
    check(not bare, "a bare quantity is admitted beside its six reductions: " + ", ".join(bare))
    misspelt = [m for m in PR.SCALAR_QUANTITIES for s in PR.SUFFIXES if m.endswith(s)]
    check(not misspelt, "a run-level scalar is spelled like a reduction: " + ", ".join(misspelt))


@case("every admitted metric is documented, and no note names a withdrawn one")
def t_metrics_documented():
    """A NAME IS NOT A DEFINITION. `METRIC_NOTES` held six entries for fifty-six admitted names,
    so a role was handed a comma-separated list of identifiers and asked to predict against them
    -- and one of the six documented `wavelength_cells_final`, which is not produced and was
    withdrawn as uncalibrated. A stale note is worse than a missing one: it advertises an
    instrument that does not exist.

    DOCUMENTED PER QUANTITY AND PER SUFFIX, NOT PER NAME. 24 + 6 notes cover 144 names, and that
    is the whole reason the bank is a product: the alternative is 144 paragraphs, which is 144
    chances for one of them to go stale and nobody to notice. A product name is documented iff
    BOTH of its parts are, and `decompose` is what says which parts it has.

    AND EVERY NOTE MUST STATE THE NULL. "act_cv > 0.3" is a guess unless you know a dead field
    reads 0.00; the note is the difference between a bet and a description. Enforced literally:
    each quantity's note has to contain a stated no-answer reading.
    """
    import predict as PR
    ok = [m for m in PR.KNOWN_METRICS if m not in PR.REJECTED_METRICS]
    undoc = sorted({PR.decompose(m)[0] for m in ok} - set(PR.METRIC_NOTES) - {None})
    check(not undoc, "admitted but undocumented: " + ", ".join(undoc))
    undoc_s = [s for s in PR.SUFFIXES if s not in PR.SUFFIX_NOTES]
    check(not undoc_s, "a reduction with no note: " + ", ".join(undoc_s))
    known = set(PR.SERIES_QUANTITIES) | set(PR.SCALAR_QUANTITIES)
    stale = [m for m in PR.METRIC_NOTES if m not in known]
    check(not stale, "documented but NOT admitted (a note for a withdrawn metric): "
                     + ", ".join(stale))
    # WHAT IT READS WHEN THE ANSWER IS NO -- the half of a note that makes it predictable against.
    nullless = [m for m in PR.SERIES_QUANTITIES
                if not any(w in PR.METRIC_NOTES[m] for w in ("NO =", "NO answer", "NOT MEASURED",
                                                             "REFUSED", "CANNOT SEE"))]
    check(not nullless, "documented without a null reading: " + ", ".join(nullless))
    # ...and a withdrawn metric must say WHY, or the loop keeps reaching for it.
    silent = [m for m, why in PR.WITHDRAWN.items() if not why]
    check(not silent, "withdrawn with no reason: " + ", ".join(silent))


@case("the trajectories survive a label column and reach the Reader")
def t_trajectories_reach_the_reader():
    """THE TIME-EVOLUTION CHANNEL DIED IN SILENCE, and stayed dead for the whole campaign.

    metrics.npz carries the per-frame morphology labels as a STRING array. `classify` calls
    np.asarray(v, float) on every column and raised `could not convert string to float: 'sphere'`
    -- out of `report`, into llm_agents' except, and out as the single parenthesis "(trajectory
    shapes unavailable)" in the Reader's prompt. So every run since the label column landed was
    read with NO trajectory information: no peaked/pinned/exploded warning, no evidence horizon,
    nothing about the activator over time.

    This test builds exactly that npz -- numeric columns plus a label column -- and requires real
    lines out, so the channel cannot go quiet again without a red tick.
    """
    import numpy as np, tempfile, shutil
    import time_analysis as CS
    d = tempfile.mkdtemp()
    try:
        n = 40
        flash = np.concatenate([np.linspace(0.01, 1.8e4, 16),
                                np.logspace(np.log10(1.8e4), np.log10(0.01), n - 16)])
        np.savez(os.path.join(d, "metrics.npz"),
                 frame=np.arange(n), protr=np.linspace(1.0, 1.06, n), act_max=flash,
                 act_cv=np.concatenate([np.linspace(0, 1.2, 12), np.linspace(1.2, 0, n - 12)]),
                 morphology=np.array(["sphere"] * n))            # <-- the column that killed it
        rep = CS.report(d, write=False)
        check(rep.get("metrics"), "report returned no metric classifications at all")
        check("act_max" in rep["metrics"], "act_max was not classified")
        check("morphology" not in rep["metrics"], "a string label column was classified as a curve")
        txt = CS.summarise(rep)
        check("act_max" in txt, "act_max never reached the text an agent reads")
        check("act_cv" in txt, "the pattern amplitude never reached the text an agent reads")
        check("over time:" in txt, "no numeric time course was rendered -- only shape words")
        check("peaked" in txt, f"a flash-then-extinction was not called peaked:\\n{txt}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("the harness cannot fabricate over a real run")
def t_harness_cannot_eat_real_runs():
    """4 AUGUST: THIS HARNESS DESTROYED ALL TWELVE FINISHED ROUND-2 RUNS.

    campaign/state.json said `"round": 2`, so the offline round generated the SAME run names the
    live round had (same frontier, same composition hashes: r002c_00_f4907e, ...). fabricate_run
    did makedirs(exist_ok=True) over the real directories, recorded them in _FABRICATED as its
    own, and the atexit clear_runs rmtree'd them -- an hour of A100 time, twelve movies, twelve
    diag.json. Every component did exactly what it was written to do.

    Third occurrence. The first two fixes -- "only delete what we made", "refuse while a campaign
    is live" -- each closed the instance in front of them. This closes the class: a directory the
    harness did not create is never opened, and a real run is one it did not create.
    """
    import offline as O
    d = os.path.join(O.LOG, "r002c_00_TESTGUARD")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "diag.json"), "w").write('{"summary": {"protr_peak": 2.9}}')
    try:
        raised = False
        try:
            O.fabricate_run("r002c_00_TESTGUARD")
        except SystemExit:
            raised = True
        check(raised, "fabricate_run OVERWROTE a real run directory instead of refusing")
        check(d not in O._FABRICATED,
              "the real run was registered as ours -- clear_runs would delete it at exit")
        with open(os.path.join(d, "diag.json")) as fh:
            check("2.9" in fh.read(), "the real diag.json was overwritten by the fake one")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@case("the evidence horizon reports a FRAME, not a sample index")
def t_horizon_units():
    """A SAMPLE INDEX IS NOT A FRAME NUMBER.

    metrics.npz holds one row per SAMPLED frame -- 37 rows at stride 25 -- while `fm` and the
    horizon index are per-FRAME over all 901. The ray_single_frac branch of run_one read the
    column and forgot the axis, so a fold first seen at sample 8 (frame 200) was recorded as
    `horizon_frame 8` and truncated the evidence to eight frames of a nine-hundred-frame run:
    protr_peak and protr_final then measured the seed sphere. Every run whose mesh ever folded was
    scored on its opening frames.

    This checks the arithmetic directly, because the defect is invisible in any output -- a
    horizon of 8 looks exactly like an early, honest fold.
    """
    import numpy as np
    stride, T = 25, 901
    frames = np.unique(np.append(np.arange(0, T, stride), T - 1))
    rs = np.ones(frames.size)
    rs[8:] = 0.2                                    # folds at SAMPLE 8 == FRAME 200
    bad = np.where(np.isfinite(rs) & (rs < 0.5))[0]
    check(bad.size and int(bad[0]) == 8, "fixture wrong: the fold is not at sample 8")
    fold = int(frames[int(bad[0])])
    check(fold == 200, f"sample 8 must map to frame 200, got {fold}")
    check(fold != int(bad[0]), "the mapping is a no-op -- the sample index is still being used")
    src = open(os.path.join(HERE, "run_one.py")).read()
    check("frame_numbers[fold_sample]" in src,
          "run_one no longer maps the ray_single_frac sample index through the frame column")


@case("the temporal reductions read the real okuda_route record correctly")
def t_reductions_on_real_data():
    """SIX SCALARS PER CURVE, CHECKED ON A REAL 901-FRAME RECORD -- not a fixture.

    `time_analysis.classify` turns a trajectory into a WORD, and a word cannot be scored:
    `predict.Clause.check` looks a metric up by exact key and calls float() on it. So the whole
    time-evolution channel stops at the prompt and never reaches the record. `reduce_series`
    turns each curve into six scalars -- final / peak / floor / trend / span / measured_frac --
    and this case runs them over `okuda_route`'s per-frame archive.

    THAT RUN IS THE REASON THE REDUCTIONS EXIST. Every shape word for it is reassuring: a clean
    sphere, genus 0, 3,975 cells, protr `converged`. What no word says is that its activator runs
    from 0.004 to 951,288 inside the evidence window while act_cv -- the spatial non-uniformity,
    the thing that makes a Turing pattern a pattern -- sits at 0.035. A field that fires
    everywhere at once grows a sphere uniformly, and `act_max_span` = 1.3e6 beside
    `act_cv_final` = 0.035 is that sentence in two numbers.

    AND IT IS THE RECORD WITH THREE SAMPLING TIERS. The mesh metrics cost 1410 ms/frame against
    0.12 ms for the chemistry, so the mesh tier is sampled every 25 frames: 37 samples beside
    901. The evidence horizon is a FRAME (150 here), which is row 150 in one tier and row 6 in
    the other. Confusing the two truncated folded runs to their opening frames -- a real bug --
    so the conversion is asserted in both directions below.
    """
    import json
    import numpy as np
    import time_analysis as TA

    fz = os.path.join(os.path.dirname(HERE), "log", "okuda", "okuda_route", "frames_1.npz")
    if not os.path.exists(fz):
        return "skipped: no frames_1.npz archived"
    z = np.load(fz)
    f, mf = np.asarray(z["frame"], float), np.asarray(z["mesh_frame"], float)
    check(f.size == 901 and mf.size == 37, f"tiers changed: {f.size} chem, {mf.size} mesh")

    # Build the record the way a reader would: the every-frame tier wins any name it shares with
    # the 25-frame tier, and each column carries its own clock.
    cols = {k[5:]: z[k] for k in z.files if k.startswith("chem_")}
    cols.update({k: z[k] for k in ("protr", "r_cv", "corr_act_rad", "n_cells")})
    frames_by_col = {k: f for k in cols}
    for k in z.files:
        if k.startswith("mesh_") and k != "mesh_frame" and k[5:] not in cols:
            cols[k[5:]], frames_by_col[k[5:]] = z[k], mf

    h = TA.evidence_horizon({}, {"broken_n": np.asarray(z["mesh_broken_n"], float)}, mf)
    check(h["horizon"] == 150, f"the horizon moved: {h['horizon']} (expected frame 150)")
    R = TA.reduce_all(cols, frames_by_col, horizon_frame=h["horizon"])

    # --- one horizon, two tiers, two different row indices.
    check(R["n_cells_final"] == 2058.0,
          f"n_cells_final={R['n_cells_final']}; the chemistry tier must be cut at ROW 150 "
          "(cutting it at the mesh's row 6 gives 2000)")
    check(R["broken_n_peak"] == 1.0,
          f"broken_n_peak={R['broken_n_peak']}; the mesh tier must be cut at ROW 6 "
          "(the untruncated column peaks at 454)")
    bn = np.asarray(z["mesh_broken_n"], float)
    check(TA.reduce_series(bn, np.arange(bn.size, dtype=float), 150)["peak"] == 454.0,
          "fixture wrong: reading the horizon as a row index no longer changes the answer, so "
          "this case can no longer tell the two apart")

    # --- what okuda_route actually is: a huge, spatially uniform flash.
    check(R["act_max_span"] > 1e5,
          f"act_max_span={R['act_max_span']}; the activator spans 0.004 -> 951288 in the window")
    check(R["act_cv_final"] < 0.05,
          f"act_cv_final={R['act_cv_final']}; the field is near-uniform when it fires")
    check(R["act_max_floor"] < 0.01 < R["act_max_peak"], "floor and peak straddle nothing")

    # --- a null with a denominator. corr_act_rad is REFUSED while act_cv < 0.05, so its nulls
    #     mean "no pattern to correlate", and measured_frac is the only thing that says so.
    check(R["corr_act_rad_final"] is None, "corr_act_rad was finite at the horizon frame?")
    check(R["corr_act_rad_measured_frac"] < 0.10,
          f"corr_act_rad_measured_frac={R['corr_act_rad_measured_frac']}, expected ~0.046")
    check(R["act_cv_measured_frac"] == 1.0, "act_cv is measured every frame and must read 1.0")
    check(all(v is None or 0.0 <= v <= 1.0 for k, v in R.items() if k.endswith("measured_frac")),
          "a measured_frac escaped [0,1]")

    # --- constants and names.
    check(R["genus_span"] is None, "genus is 0 for the whole run: span is 0/0, i.e. None")
    check(R["genus_trend"] is None, "a constant series has no trend direction to report")
    check(not any(k.endswith("_min") for k in R),
          "a reduction is suffixed _min; predict._METRIC_ALT is a LONGEST-FIRST alternation and "
          "shape_idx_min / act_min are admitted, so _min shadows them. It is called _floor.")
    check("shape_idx_min_floor" in R, "the kept _min series stopped being reduced")

    # --- a scalar that cannot be re-read is not in the record.
    bad = [k for k, v in R.items() if v is not None and not np.isfinite(v)]
    check(not bad, f"NaN/inf in the summary (json.dump writes bare NaN/Infinity): {bad[:4]}")
    check("NaN" not in json.dumps(R) and "Infinity" not in json.dumps(R), "summary is not JSON")

    # --- and the whole run, for contrast: the damage and the growth are both monotone.
    full = TA.reduce_all(cols, frames_by_col)
    check(full["broken_n_peak"] == 454.0, "with no horizon the whole column is read")
    check(full["n_cells_trend"] > 0.99, f"n_cells_trend={full['n_cells_trend']}, growth is not "
                                        "monotone any more")
    check(full["act_cv_final"] < 0.01 and full["act_max_span"] > 1e6,
          "the run ends with a dead, uniform field after a six-decade excursion")

    # --- and the refusal that keeps the units honest.
    try:
        TA.reduce_all({"broken_n": bn}, None, horizon_frame=150)
        check(False, "reduce_all applied a frame-number horizon to a column with no frame column")
    except ValueError:
        pass
    return (f"horizon frame {h['horizon']} -> row 150 of 901 and row 6 of 37; "
            f"act_max_span {R['act_max_span']:.3g} vs act_cv_final {R['act_cv_final']:.3g}")


@case("every admitted metric reaches a REAL run's summary")
def t_admitted_reaches_the_summary():
    """ADMITTED IMPLIES PRESENT. The end-to-end contract, and the one the others could not catch.

    `t_metrics_have_producers` asks whether a name is ASSIGNED somewhere in the source. It passed
    green all day while `time_analysis.reduce_all` -- which assigns every one of them -- was
    called by the tests and BY NOTHING ELSE. So the reductions existed, were documented, were
    admitted, and never reached a run: `act_cv_peak <= 0.3`, `act_max_span >= 100` and
    `corr_act_rad_measured_frac <= 0.1` -- three claims that state exactly what okuda_route does
    -- each scored `not measured` and fell to inconclusive, silently.

    That is the defect this whole phase is about, made once more while fixing it. Source-level
    checks cannot see it, because every part was correct; only the seam was missing. So this one
    reads a REAL diag.json and requires the admitted set to be IN it.

    An absence is allowed only if it is DECLARED with a reason, exactly as WIRING.md requires of
    an artifact with no reader. A metric that is legitimately null on some runs is not a defect;
    a metric that is null on every run and nobody noticed is.
    """
    import glob
    import predict as PR
    # DECLARED ABSENCES, with the reason each is null rather than missing.
    allowed = {
        "Q_drop": "only written when the quasi-static relaxation ran (--q)",
        "corr_act_rad_final": "refused at the horizon frame when act_cv < 0.05 -- no pattern to "
                              "correlate; corr_act_rad_measured_frac reports how often",
        "act_at_tip_final": "same refusal as corr_act_rad_final",
    }
    # ...plus any reduction of a series that is CONSTANT on this run: _trend is null on ties and
    # _span is null on a zero median, both by design. Those are run-dependent, not bank defects.
    degenerate = ("_trend", "_span")

    diags = sorted(glob.glob(os.path.join(HERE, "..", "log", "okuda", "*", "diag.json")),
                   key=os.path.getmtime, reverse=True)
    if not diags:
        return "skipped -- no finished run on disk to check against"
    summ = json.load(open(diags[0])).get("summary", {})
    if len(summ) < 100:
        return (f"skipped -- {os.path.basename(os.path.dirname(diags[0]))} predates the "
                f"reductions ({len(summ)} keys); re-run one composition to check this")
    admitted = [m for m in PR.KNOWN_METRICS if m not in PR.REJECTED_METRICS]
    missing = [m for m in admitted
               if m not in summ and m not in allowed and not m.endswith(degenerate)]
    check(not missing,
          f"admitted but ABSENT from {os.path.basename(os.path.dirname(diags[0]))}'s summary "
          f"({len(missing)}): " + ", ".join(missing[:12]))
    # AND THE RULE MUST BITE: the run must actually carry the reductions, not merely lack them
    # consistently. A summary of eighty keys would pass the check above by having no admitted
    # names at all.
    reduced = [k for k in summ if k.endswith(("_peak", "_floor", "_trend", "_span",
                                              "_measured_frac"))]
    check(len(reduced) > 50,
          f"only {len(reduced)} reduction keys in the summary -- reduce_all is not wired in")


@case("a withdrawn metric cannot return under a new suffix")
def t_rejected_not_reduced():
    """`autocorr_hops_uncalibrated` was withdrawn because it was never calibrated. Reducing EVERY
    column of the per-frame table wrote it back under SIX new names -- a record carrying a
    withdrawn instrument under six suffixes has re-admitted it by the back door.

    THIS TESTS THE CODE PATH, NOT AN OLD FILE. The first version read the newest diag.json on
    disk, which made it archaeology: it failed on a record written minutes before the fix and
    would have demanded a fresh simulation before the loop -- which runs simulations -- could
    start. A test that requires an experiment to certify the thing that runs experiments has the
    dependency backwards. It exercises the same key selection `run_one` uses, so it is
    deterministic, needs no GPU, and cannot be fooled by a stale artifact either way.
    """
    import numpy as np
    import predict as PR
    import time_analysis as TA
    rej = PR.REJECTED_METRICS[0]
    n = 40
    cols = {"act_cv": np.linspace(0, 1, n), "protr": np.linspace(1, 1.3, n),
            rej: np.linspace(5, 50, n)}                      # a withdrawn column, as on disk
    fb = {k: np.arange(n, dtype=float) * 25 for k in cols}
    wide = TA.reduce_all(cols, fb, horizon_frame=None)
    check(any(k.startswith(rej + "_") for k in wide),
          "fixture wrong: reducing every column did not emit the withdrawn metric")
    # ...and now the selection run_one actually applies:
    kept = [k for k in cols if k in set(PR.SERIES_QUANTITIES)]
    narrow = TA.reduce_all(cols, fb, horizon_frame=None, keys=kept)
    back_door = sorted({k for k in narrow for r in PR.REJECTED_METRICS if k.startswith(r + "_")})
    check(not back_door, "a REJECTED metric survives run_one's key selection: "
                         + ", ".join(back_door[:8]))
    check(any(k.startswith("act_cv_") for k in narrow),
          "the selection dropped an ADMITTED quantity too -- it is not selecting, it is emptying")


@case("roles.py --check still agrees with the code")
def t_roles():
    import roles
    bad = roles.check()
    check(not bad, f"{len(bad)} role complaint(s); first: {bad[0] if bad else ''}")


# --------------------------------------------------------------------------- helpers
def _round(scenario, batch=6):
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "offline.py"),
                        "--scenario", scenario, "--batch", str(batch)],
                       capture_output=True, text=True, timeout=600, cwd=HERE)
    log = r.stdout + r.stderr
    if VERBOSE:
        print(log)
    return r.returncode, log


def _attrition(log):
    m = re.search(r"asked (\d+), proposed (\d+), refused (\d+), delivered (\d+) of (\d+)", log)
    if not m:
        return None
    return dict(zip(("asked", "proposed", "refused", "delivered", "target"),
                    (int(x) for x in m.groups())))


@case("a sweep is a legal experiment, not a duplicate of the control")
def t_sweep_is_an_experiment():
    """THE SECOND HALF OF TRACK A, WHICH WAS UNREACHABLE BY ARITHMETIC.

    `comp_hash` is deliberately parameter-blind so a retune cannot be filed as a new mechanism --
    proven, 107 of 107 set_param edits leave it bit-identical. But round.py deduped slots WITHIN a
    batch on that same bare hash while proposer.py REQUIRES slot 0 to be the unchanged parent, so
    every parameter move collided with the mandatory control: an external review measured 48 of 48
    admitted sweep slots refused as DUPLICATE_IN_BATCH, with the refusal text "Vary the EDIT, not
    the number" then entering the next Proposer's prompt as a lesson. "What does this mechanism do
    as you turn it up" was asked ZERO times in 36 slots.

    critic.py had the answer already: a set_param slot is keyed on `_run_key` -- mechanism AND
    operating point. Two readings of one identity function disagreed about what a sweep is.
    """
    import offline as O
    O.install("clean")
    import round as R, critic as C
    g0 = R.load_frontier()[0]
    sp = [e for e, _lbl in g0.legal_edits() if e[0] == "set_param"]
    check(len(sp) >= 5, f"only {len(sp)} set_param edits offered on the frontier parent")
    bare, keyed = set(), set()
    for e in sp:
        try:
            r = g0.apply(e)
        except Exception:
            continue
        g = r[0] if isinstance(r, tuple) else r
        bare.add(R.comp_hash(g)); keyed.add(C._run_key(g))
    check(len(bare) == 1, f"comp_hash is no longer parameter-blind: {len(bare)} hashes for "
                          f"{len(sp)} sweep edits -- a retune could now pass as a new mechanism")
    check(len(keyed) >= len(sp) - 1,
          f"only {len(keyed)} distinct run keys for {len(sp)} sweep edits -- the in-batch dedupe "
          f"would still collapse them onto the control")
    src = open(os.path.join(HERE, "round.py")).read()
    check("_dedupe_key" in src and "_run_key" in src,
          "round.py's in-batch dedupe no longer distinguishes a sweep from its control")


@case("a broken premise makes a prediction inconclusive, not confirmed")
def t_specimen_gates_the_resolver():
    """TWO RECORDS OF ONE EXPERIMENT DISAGREED ABOUT WHETHER IT WAS EVIDENCE.

    The Biologist writes `premises_broken`; critic.check_posthoc never read it. So a run that
    passed through itself (P11) or absorbed area by stretching (P7) was scored confirmed/refuted
    in hypotheses.jsonl while collector.py wrote "specimen invalid ... describes the
    configuration and not a tissue" into analysis.md about the SAME run -- and the Supervisor
    reads the register, so the surprise rate, the 70/30 steer and the cluster freeze were all
    computed over runs the Biologist had rejected. Measured: 24 of 35 archived runs.
    """
    import critic as C
    clean = {"protr_peak": 1.4, "inert_operators": [], "premises_broken": []}
    check(not [r for r in C.check_posthoc(clean) if r.code == "P0_SPECIMEN_INVALID"],
          "a clean specimen is being refused")
    broken = dict(clean, premises_broken=["P7", "P11"])
    codes = [r.code for r in C.check_posthoc(broken)]
    check("P0_SPECIMEN_INVALID" in codes,
          f"a run that broke P7 and P11 is still scorable: {codes}")
    # A JUDGEMENT CALL IS NOT A HARD REFUSAL. `?`-suffixed premises are uncertain by convention,
    # and refusing on those would throw away real evidence.
    soft = dict(clean, premises_broken=["P5b?"])
    check("P0_SPECIMEN_INVALID" not in [r.code for r in C.check_posthoc(soft)],
          "an uncertain premise is being treated as a certain refusal")


@case("the loop is allowed to be curious")
def t_exploratory_intent():
    """"It must be allowed to be curious ... an exploratory slot states what it is VARYING and
    what it will REPORT, rather than what it expects." Until now `intent="exploratory"` raised
    ValueError and so did `predicted=""`, so every slot had to be a bet -- and the trap the goal
    names (a loop that may only test predictions proposes only what it can predict) was structural
    rather than cultural."""
    import hypothesis as HY
    check("exploratory" in HY.INTENTS, "there is still no exploratory intent")
    check("described" in HY.OUTCOMES, "an exploratory slot has no outcome of its own")
    check("exploratory" not in HY.MECHANISM_INTENTS,
          "an exploratory slot is in the surprise denominator -- it made no prediction to be "
          "surprised by, so it would dilute the campaign's only control signal")
    base = dict(hid="X", comp_hash="C1", parent_hash=None, edit="turn chi up",
                intent="exploratory", claim="", metric="protr_peak", predicted="", round_id=1)
    h = HY.Hypothesis(**dict(base, varying="chi 1.3 -> 8.0", will_report="n_spots_peak"))
    h.resolve({"n_spots_peak": 41}, "described", note="41 spots, no elongation")
    check(h.outcome == "described", f"resolved {h.outcome!r}, not `described`")
    for bad in ({}, {"varying": "chi 1.3 -> 8.0"}):
        try:
            HY.Hypothesis(**dict(base, **bad))
            check(False, f"an exploratory slot stating {list(bad) or 'nothing'} was accepted -- "
                         f"it is a licence to look at the output and call it interesting")
        except ValueError:
            pass
    try:
        HY.Hypothesis(**dict(base, intent="confirmatory", predicted=""))
        check(False, "a confirmatory slot with no prediction is now accepted -- the rule that "
                     "matters has been weakened")
    except ValueError:
        pass


@case("a round that cannot move its own metric is caught")
def t_null_difference():
    """NINE OF TWELVE ROUND-3 RUNS SHARED A SHAPE TO SIXTEEN DIGITS under nine different
    structural edits, because both frontier parents had `conns: []` and no growth operator -- so
    the chemistry could not reach the mechanics and the shape was a constant of the edits being
    made. Seven of the nine resolved `confirmed`. Re-measured here: 10 of 12 on gyr_prolate_peak.
    """
    import critic as C
    rows = [("ctrl", "Ca", {"protr_peak": 1.022, "gyr_prolate_peak": 2.153}, "protr_peak"),
            ("slotA", "Cb", {"protr_peak": 1.022, "gyr_prolate_peak": 2.153}, "protr_peak"),
            ("slotB", "Cc", {"protr_peak": 1.022, "gyr_prolate_peak": 2.153}, "protr_peak"),
            ("slotC", "Cd", {"protr_peak": 1.31, "gyr_prolate_peak": 2.9}, "protr_peak")]
    hits = C.check_null_difference(rows, control="ctrl")
    names = {n for n, _m, _v in hits}
    check(names == {"slotA", "slotB"},
          f"the null slots were not identified: {sorted(names)}")
    n, metric, group = C.check_round_decoupled(rows)
    check(n >= 3, f"the round-level decoupling check found only a group of {n}")
    check(metric in C.SHAPE_VECTOR, f"grouped on {metric!r}, which is not a shape metric")



if __name__ == "__main__":
    print(f"offline regression -- {len(CASES)} case(s), no agent, no GPU\n")
    for name, fn in CASES:
        try:
            note = fn()
            print(f"  PASS  {name}" + (f"  ({note})" if note else ""))
        except AssertionError as e:
            FAILED.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:
            FAILED.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - len(FAILED)}/{len(CASES)} passed")
    sys.exit(1 if FAILED else 0)


@case("the Proposer is shown the metrics it must predict against")
def t_proposer_sees_the_bank():
    """THE ROLE THAT WRITES THE PREDICTIONS WAS NEVER GIVEN THE LIST.

    proposer.py's prompt says "naming a metric from the admitted list" and did not include the
    list. The only metric names in it were the two used as EXAMPLES of the clause syntax --
    `protr_peak` and `mech_p_ratio` -- and in round 2 the Proposer wrote ALL TWELVE predictions
    on protr_peak. I diagnosed that as a presentation problem with a 53-name flat list; it was
    a bank the Proposer had never been handed. The Reader, which only labels a finished run,
    received the full registry all along.

    Checked on the ASSEMBLED PROMPT, not on the source, because that is where the omission was:
    every part was correct and the seam was missing.
    """
    import offline as O
    O.install("clean")
    import round as R, proposer as P, llm
    cap = {}
    real, llm.run_claude = llm.run_claude, lambda p, **k: (cap.__setitem__("p", p), (True, "{}"))[1]
    try:
        with quiet():
            P.propose(R.load_frontier(), R.CampaignConfig(batch=6, keep_truncate=2),
                      None, "", 4, n_slots=8)
    finally:
        llm.run_claude = real
    prompt = cap.get("p", "")
    check(prompt, "the proposer prompt was never captured")
    import predict as PR
    for m in ("act_cv_peak", "corr_act_rad_measured_frac", "act_alive_frac"):
        check(m in prompt, f"{m} is admitted but never reaches the Proposer's prompt")
    groups = sum(1 for g in PR.SERIES_METRICS if g in prompt)
    check(groups >= 4, f"only {groups} of the metric groups reached the prompt")
    check("THE SIX SUFFIXES" in prompt, "the reduction suffixes never reach the Proposer")

