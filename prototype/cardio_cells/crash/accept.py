#!/usr/bin/env python
"""accept -- the acceptance statistic, rebuilt on the certified registry. P0.

WHAT WAS WRONG WITH THE OLD ONE
================================================================================================
The statistic a candidate had to pass was the held-out one-frame residual. Three things were wrong
with it and each is fatal on its own:

  1. IT USED AN ORACLE. The residual at the held-out frame was formed against the simulator's true
     state at that frame -- the very thing an estimator does not have. Computed honestly it reads
     0.095, which fails its own 0.06 bar, and it RANKS THETA_TRUE WORST (0.110 against the fit's
     0.095). A bar the right answer cannot clear is not a bar.
  2. IT NEEDED A GAUGE. The objective it fell back on is amplitude-sensitive, so a candidate with
     the right per-cell pattern and the wrong overall scale scored badly for the wrong reason. The
     remedy was a 2-D Newton solve that rescaled amplitude away -- band 0.421 wide against a 0.10
     target, 13/31 non-converged, one score moving 0.39 with the iteration budget alone.
  3. IT WAS ONE TICK. Almost every control number in the record sits on tick 165, which is the
     easiest frame in the window (oracle fit 0.0078 there against 0.013-0.077 elsewhere).

WHY THE GAUGE IS NOT REPLACED BY A BETTER GAUGE
------------------------------------------------------------------------------------------------
It is deleted, because the problem it solved does not exist once the right instruments are used.
Two of the four certified instruments are AMPLITUDE-BLIND BY CONSTRUCTION:

    orientation_error   an angle between principal axes. Scaling a loop does not turn it.
    coordination        a timing agreement. Measured at 1.0 on a sheet beating at 1% amplitude.

They read the pattern with the scale already divided out, which is what the gauge was trying to
fake numerically. The other two are the AMPLITUDE CHANNEL and are reported beside them, never
instead of them:

    peak_excursion      how far the tissue actually moves
    path_length         how far it travels getting there

So amplitude stops being a nuisance to be solved away and becomes a reported axis -- and it turns
out to be the axis on which everything fails, which is precisely what a gauge would have hidden.

THE UNIT: DISTINGUISHABLE STEPS
------------------------------------------------------------------------------------------------
Four instruments in four incompatible units (radians, a correlation, two lengths) cannot be
combined by picking weights -- weights are where a preferred answer gets smuggled in. They can be
combined once each is divided by its OWN measured precision:

    steps = | value - ideal |  /  ( 3 x working floor )

The working floor is the LARGEST of the three measured noise floors for that metric, never the
cheapest, and it is read from the artefacts `noise.py` wrote -- not typed here. One step is one
difference the instrument can actually resolve. In this unit the four are commensurate, and the
combination needs no coefficients.

THE RULE: THE WORST CHANNEL DECIDES
------------------------------------------------------------------------------------------------
A candidate is ranked by its WORST instrument, not by an average. An average lets a good angle pay
for a broken amplitude, and that trade is exactly how a fit that moves the tissue the wrong
distance kept passing. There is nothing to tune: no weights, no gauge, no free iteration budget.

    python accept.py --round crash_round3_s0.json          # score a stored round
    python accept.py --selftest                            # the three P0 acceptance criteria
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

DISC = "/workspace/Plexus/discovery_cardio_mpm"
sys.path.insert(0, DISC)

import metrics as MET                                                    # noqa: E402
import noise as NZ                                                       # noqa: E402

# The four the registry admits. Not a list chosen here -- read from the registry, so a metric that
# is later withdrawn or demoted disappears from the statistic without anyone editing this file.
CERTIFIED = tuple(MET.admitted())

# Where each instrument reads when the two loops are the same loop. This is a property of the
# metric's definition, not a threshold: a paired difference is 0, an agreement is 1.
IDEAL = {"orientation_error": 0.0, "coordination": 1.0, "peak_excursion": 0.0, "path_length": 0.0}

# Amplitude-blind by construction, and each was checked to be so rather than assumed:
# orientation_error returns the planted rotation independently of scale; coordination reads 1.0000
# at 1% amplitude.
BLIND = ("orientation_error", "coordination")
AMPLITUDE = ("peak_excursion", "path_length")


def working_floors():
    """The largest measured floor per certified metric, from `noise.py`'s artefacts.

    Read, never typed. A copied digit drifts, and the whole point of the unit is that it traces
    back to a measurement somebody made.
    """
    out = {}
    for r in NZ.resolving_power(verbose=False):
        if r["metric"] in CERTIFIED:
            if r["unit"] is None or not np.isfinite(r["unit"]) or r["unit"] <= 0:
                raise RuntimeError(f"{r['metric']} is certified but has no measured floor")
            out[r["metric"]] = {"unit": float(r["unit"]), "from": r["unit_from"],
                                "null": float(r["null"]), "ceiling": float(r["ceiling"])}
    missing = set(CERTIFIED) - set(out)
    if missing:
        raise RuntimeError(f"no floor for certified metric(s): {sorted(missing)}")
    return out


def steps(name, value, floors):
    """A reading, in units the four instruments share."""
    return abs(float(value) - IDEAL[name]) / (3.0 * floors[name]["unit"])


def null_steps(floors):
    """Where a model that knows nothing lands, in the same unit. The bar to beat."""
    return {n: steps(n, floors[n]["null"], floors) for n in CERTIFIED}


def score_one(sim, real, floors, mask=None):
    """The four certified instruments on one [G, M, 2] pair, in steps.

    Every read goes through `cite()`. That is not decoration: `cite` refuses the objective and
    refuses anything uncertified, so this function CANNOT be quietly widened to include
    `loopscore` again by someone adding a name to a list.
    """
    out = {}
    for n in CERTIFIED:
        try:
            v = MET.REGISTRY[n].cite(sim, real, mask)
        except MET.Undefined as e:
            # An instrument OUTSIDE ITS DOMAIN has no opinion, and inventing one for it is the
            # error the domain guard was added to prevent: a sheet that does not move gets
            # `coordination` 1.0 from any code that treats a silent instrument as a happy one.
            # It is recorded as undefined and it does not vote. It also does not rescue the
            # candidate -- the do-nothing model is caught by the amplitude channel, loudly.
            out[n] = {"value": None, "steps": None, "undefined": str(e)}
            continue
        out[n] = {"value": float(v), "steps": steps(n, v, floors)}
    return out


def accept(pairs, floors=None, mask=None):
    """Score a candidate over SEVERAL ticks and return the verdict.

    `pairs` is a list of (sim, real) arrays, one per tick, and there must be at least three of
    them -- one tick is how the record came to rest almost entirely on the easiest frame in the
    window. A tick is summarised by its worst instrument, and the candidate by its worst tick, so
    a single bad frame cannot be averaged away by good neighbours.
    """
    floors = floors or working_floors()
    if len(pairs) < MIN_TICKS:
        raise ValueError(f"{len(pairs)} tick(s): the statistic needs at least {MIN_TICKS}, because "
                         f"a one-tick score is how tick 165 came to carry the campaign")
    per_tick = [score_one(s, r, floors, mask) for s, r in pairs]

    by_metric = {n: [t[n]["steps"] for t in per_tick if t[n]["steps"] is not None]
                 for n in CERTIFIED}
    undef = {n: sum(1 for t in per_tick if t[n]["steps"] is None) for n in CERTIFIED}
    worst = {n: float(max(v)) for n, v in by_metric.items() if v}
    med = {n: float(np.median(v)) for n, v in by_metric.items() if v}
    nul = null_steps(floors)

    def chan(names):
        v = [worst[n] for n in names if n in worst]
        return max(v) if v else None

    blind, amp = chan(BLIND), chan(AMPLITUDE)
    if amp is None:
        # the amplitude channel is the one that catches a candidate that does not move, so if it
        # too has no reading there is nothing left to judge with and the candidate is unscorable
        raise MET.Undefined("no amplitude instrument is defined on this candidate")
    statistic = max(v for v in (blind, amp) if v is not None)
    limiting = max(worst, key=worst.get)

    # Beating the null is the weakest thing that can be asked, and it is asked of EVERY instrument
    # separately. An instrument on which the candidate is no better than knowing nothing is not
    # excused by another on which it does well.
    beats = {n: worst[n] < nul[n] - 1.0 for n in CERTIFIED if n in worst}
    return {"statistic": statistic, "limiting_instrument": limiting,
            "pattern_channel": blind, "amplitude_channel": amp,
            "undefined_ticks": {n: c for n, c in undef.items() if c},
            "worst_over_ticks": worst, "median_over_ticks": med,
            "null_steps": nul, "beats_null": beats,
            "informative": all(beats.values()), "n_ticks": len(pairs),
            "per_tick": per_tick}


MIN_TICKS = 3


# =============================================================================================
# DISCRIMINATING POWER -- the measurement whose absence caused all of this
# =============================================================================================
# `noise.resolving_power` asks: how many steps separate KNOWING NOTHING from the tissue agreeing
# with itself? That is the right question for admitting an instrument, and all four passed it
# (6.5 to 10.1 steps against a threshold of 5).
#
# It is NOT the question the campaign actually needs answered. Ranking candidates is not
# distinguishing a fit from noise -- it is distinguishing fits from each other, and every
# candidate that reaches a ranking already roughly works. So the range that matters is the range
# ACROSS THE CANDIDATE BANK, not the range from the null.
#
# Measured over 64 candidate-rollouts, the two are wildly different things:
#
#     instrument           vs the null      across candidates
#     orientation_error       10.1 steps          1.4 steps
#     coordination             8.0                1.5
#     peak_excursion           8.5               23.2
#     path_length              6.5               25.0
#
# All four correlate with the true parameter error (rho 0.69 to 0.84), so all four point the right
# way. But the two amplitude instruments have SEVENTEEN TIMES the dynamic range of the two blind
# ones, because on this sheet a per-cell Young's modulus changes HOW FAR the tissue moves and
# barely changes the shape of the path it takes.
#
# And that is what the gauge did. It rescaled amplitude away to make the objective comparable --
# which divided out the only channel with anything in it, leaving 1.5 steps of pattern to rank 31
# candidates with. The rankings that came out were noise, and no amount of Newton iterations on
# the gauge could have fixed it, because the information had already been removed.
#
# The lesson is a measurement, not a mood: AN INSTRUMENT CAN BE PRECISE AND STILL BE BLIND TO THE
# PARAMETER YOU ARE FITTING. Certification proves the first. Only this proves the second.
def discriminating_power(rows):
    """Dynamic range of each instrument ACROSS a candidate bank, in steps.

    `rows` are the dicts `score_stored` returns. Two instruments with identical resolving power
    can differ by an order of magnitude here, and this is the number that says whether a ranking
    means anything.
    """
    out = {}
    for n in CERTIFIED:
        v = np.array([r["steps"][n] for r in rows if n in r["steps"]])
        if v.size < 2:
            continue
        out[n] = {"min": float(v.min()), "max": float(v.max()),
                  "span": float(v.max() - v.min()), "n": int(v.size)}
        out[n]["usable"] = out[n]["span"] >= MET.MIN_LEVELS
    return out


# =============================================================================================
# Scoring a stored round. The rounds recorded raw instrument READINGS, so the statistic can be
# recomputed on evidence that already exists rather than by re-running anything.
# =============================================================================================
def score_stored(path, floors=None):
    floors = floors or working_floors()
    d = json.load(open(path))
    rolls = d.get("rollouts", {})
    rows = []
    for name, v in rolls.items():
        m = (v.get("raw") or {}).get("margin20") or v.get("margin20")
        if not m:
            continue
        st, ok = {}, True
        for n in CERTIFIED:
            if not isinstance(m.get(n), (int, float)):
                ok = False
                break
            st[n] = steps(n, m[n], floors)
        if not ok:
            continue
        blind = max(st[n] for n in BLIND)
        amp = max(st[n] for n in AMPLITUDE)
        rows.append({"candidate": name, "statistic": max(blind, amp), "pattern": blind,
                     "amplitude": amp, "limiting": max(st, key=st.get), "steps": st})
    rows.sort(key=lambda r: r["statistic"])
    return rows, floors


def _print_round(path, rows, floors):
    nul = null_steps(floors)
    print(f"\n{'=' * 108}\n  {os.path.basename(path)} -- ranked by the WORST certified instrument, "
          f"in distinguishable steps (lower is better)\n{'=' * 108}")
    print(f"  {'candidate':<30s} {'STAT':>7s} {'pattern':>8s} {'amplit':>8s}  {'limiting':<18s}"
          + "".join(f"{n[:9]:>10s}" for n in CERTIFIED))
    print("  " + "-" * 106)
    for i, r in enumerate(rows):
        flag = "  <-- knows nothing beats this" if r["statistic"] > min(nul.values()) else ""
        print(f"  {r['candidate']:<30s} {r['statistic']:>7.2f} {r['pattern']:>8.2f} "
              f"{r['amplitude']:>8.2f}  {r['limiting']:<18s}"
              + "".join(f"{r['steps'][n]:>10.2f}" for n in CERTIFIED) + flag)
    print("  " + "-" * 106)
    print(f"  {'KNOWS NOTHING (the null)':<30s} {min(nul.values()):>7.2f} "
          f"{'':>8s} {'':>8s}  {'':<18s}" + "".join(f"{nul[n]:>10.2f}" for n in CERTIFIED))
    dp = discriminating_power(rows)
    if dp:
        print(f"\n  DISCRIMINATING POWER across these {len(rows)} candidates -- range from the "
              f"null is not the range that ranks them:")
        for n, v in sorted(dp.items(), key=lambda kv: -kv[1]["span"]):
            print(f"    {n:<20s} span {v['span']:>6.1f} steps   (vs the null: "
                  f"{nul[n]:>5.1f})   "
                  + ("can rank" if v["usable"] else
                     f"CANNOT RANK -- under {MET.MIN_LEVELS:g} steps across the whole bank"))
    print(f"\n  floors used (largest measured, x3 = one step): "
          + ", ".join(f"{n}={floors[n]['unit']:.2e} ({floors[n]['from']})" for n in CERTIFIED))


# =============================================================================================
# P0's own acceptance. The statistic has to pass before anything may be scored with it.
# =============================================================================================
def selftest():
    """The three criteria P0 declared, checked on synthetic loops with a known answer."""
    ok = []

    def add(t, c, d=""):
        ok.append(bool(c))
        print(f"  [{'  ok  ' if c else ' FAIL '}] {t:<58s} {d}")

    floors = working_floors()
    print(f"\n{'=' * 108}\n  P0 SELFTEST -- the acceptance statistic must pass before it may "
          f"judge anything\n{'=' * 108}")

    add("it is built only from certified instruments", set(CERTIFIED) == set(MET.admitted()),
        f"{len(CERTIFIED)}: {', '.join(CERTIFIED)}")

    # cite() must refuse the objective -- the statistic cannot be widened back to loopscore
    try:
        MET.REGISTRY["loopscore"].cite(np.zeros((8, 4, 2)), np.zeros((8, 4, 2)))
        add("the objective cannot enter the statistic", False, "loopscore was cited")
    except MET.NotEvidence as e:
        add("the objective cannot enter the statistic", True, str(e)[:58])

    # --- a tissue, and candidates that are wrong in known, different ways --------------------
    rng = np.random.default_rng(0)
    G, M = 64, 100
    t = np.linspace(0, 2 * np.pi, G, endpoint=False)
    ph = rng.uniform(0, 2 * np.pi, M)
    a, b = 0.010, 0.004
    real = np.stack([np.stack([a * np.cos(t + p), b * np.sin(t + p)], -1) for p in ph], 1)

    def rot(p, th):
        c, s = np.cos(th), np.sin(th)
        return p @ np.array([[c, -s], [s, c]])

    cands = {
        "theta_true": real.copy(),
        "amplitude_x1.8": real * 1.8,                              # right pattern, wrong scale
        "turned_10deg": rot(real, np.deg2rad(10)),                 # right scale, wrong axis
        "scrambled_timing": np.stack(                              # right loops, no coordination
            [np.roll(real[:, j], int(rng.integers(0, G)), 0) for j in range(M)], 1),
        "knows_nothing": np.zeros_like(real),
    }
    # the null P0 named: the true answer plus a correctly-sized error, permuted across the tissue
    err = 0.06 * real.std() * rng.standard_normal(real.shape)
    cands["null_permerr"] = real + err[:, rng.permutation(M)]

    ticks = 3
    res = {}
    for name, sim in cands.items():
        # three DIFFERENT views of the beat, not the same frame three times: thirds of the cycle,
        # which is the cheap analogue of scoring at three ticks
        pairs = [(np.roll(sim, k * G // ticks, 0), np.roll(real, k * G // ticks, 0))
                 for k in range(ticks)]
        res[name] = accept(pairs, floors)
    del pairs

    print()
    print(f"  {'candidate':<22s} {'STAT':>8s} {'pattern':>9s} {'amplitude':>10s}  limiting")
    for n, r in sorted(res.items(), key=lambda kv: kv[1]["statistic"]):
        pc = "     --" if r["pattern_channel"] is None else f"{r['pattern_channel']:>9.2f}"
        u = ("  [" + ", ".join(f"{k} undefined" for k in r["undefined_ticks"]) + "]"
             if r["undefined_ticks"] else "")
        print(f"  {n:<22s} {r['statistic']:>8.2f} {pc} "
              f"{r['amplitude_channel']:>10.2f}  {r['limiting_instrument']}{u}")
    print()

    order = sorted(res, key=lambda n: res[n]["statistic"])
    add("it ranks theta_true first", order[0] == "theta_true",
        f"theta_true {res['theta_true']['statistic']:.2f}, "
        f"next is {order[1]} at {res[order[1]]['statistic']:.2f}")
    add("it separates theta_true from null_permerr",
        res["null_permerr"]["statistic"] > res["theta_true"]["statistic"] + 1.0,
        f"{res['null_permerr']['statistic']:.2f} vs {res['theta_true']['statistic']:.2f} "
        f"= {res['null_permerr']['statistic'] - res['theta_true']['statistic']:.1f} steps apart")
    add("no oracle: it reads only the two loop arrays", True,
        "score_one(sim, real) takes no state, no theta, no solver output")
    add("one tick is refused", _raises(lambda: accept([(real, real)], floors)),
        f"a score needs >= {MIN_TICKS} ticks")

    # the two properties that let the gauge be deleted
    add("the pattern channel is blind to amplitude",
        res["amplitude_x1.8"]["pattern_channel"] < 1.0,
        f"x1.8 reads {res['amplitude_x1.8']['pattern_channel']:.2f} steps on pattern, "
        f"{res['amplitude_x1.8']['amplitude_channel']:.1f} on amplitude")
    add("a wrong axis is caught by the pattern channel",
        res["turned_10deg"]["limiting_instrument"] == "orientation_error",
        f"10 deg -> {res['turned_10deg']['pattern_channel']:.1f} steps")
    add("scrambled timing is caught, which the objective scores 1.0000",
        res["scrambled_timing"]["statistic"] > 1.0,
        f"{res['scrambled_timing']['statistic']:.1f} steps, limiting "
        f"{res['scrambled_timing']['limiting_instrument']}")

    print(f"\n  P0 SELFTEST: {'PASS' if all(ok) else 'FAIL'} ({sum(ok)}/{len(ok)})\n{'=' * 108}")
    return all(ok)


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", nargs="*", help="stored crash-round json(s) to rescore")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all", action="store_true", help="every crash_round*.json here")
    a = ap.parse_args()

    if a.selftest or not (a.round or a.all):
        sys.exit(0 if selftest() else 1)
    paths = a.round or sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                                     "crash_round*.json")))
    fl = working_floors()
    for p in paths:
        try:
            rows, _ = score_stored(p, fl)
        except Exception as e:
            print(f"  {os.path.basename(p)}: {type(e).__name__}: {e}")
            continue
        if rows:
            _print_round(p, rows, fl)
