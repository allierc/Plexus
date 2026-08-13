#!/usr/bin/env python
"""foresight -- how far ahead the accumulated knowledge can see. One number per round.

CEDRIC, 13 AUGUST: *"the discrepancies between prediction and eye should not govern the loop,
because the eye might be wrong, and the knowledge too limited for the forecaster. I see it more like
a score for the knowledge building."*

SO THIS IS A THERMOMETER AND NOT A THERMOSTAT, and the distinction is the whole design. Nothing
downstream consumes what this returns: it does not choose parents, rank runs, gate an experiment or
reach any role's prompt. It is written to the record and printed. That restraint is not modesty --
it is the only defensible use of a signal whose two halves are both weak. The Eye can misread a
frame; `knowledge.md` can be thin or wrong in the regime a spec lands in. A search steered by the
disagreement between two unreliable instruments would chase its own measurement error, which is
precisely the failure the campaign has already documented in its metrics (`protrusion_aspect_max`
reading 0.0 on an eleven-armed star) and has no reason to repeat in its epistemics.

WHAT IT ACTUALLY MEASURES. The Forecaster fills `crew/description.md`'s six-slot form from the spec
and the knowledge, before the GPU runs. The Eye fills the same form from the frames, after, having
seen neither the forecast nor the spec nor the metrics. Neither can converge on the other except by
both being right about the tissue. The fraction of slots they agree on is therefore an estimate of
how much of what the campaign believes is true enough to act on -- the property `knowledge.md` has
never had a number for in twenty-two rounds of growing.

WHY THE SCORING IS MECHANICAL AND NOT A JUDGE. A third model asked "do these two descriptions
agree?" would be a third opinion in a measurement that already has two, it would cost a call per
run, and it would not be reproducible between rounds -- so a rise in the score could not be
distinguished from a drift in the judge. Slots compare as words and numbers. The cost is bluntness,
and the answer to bluntness is that every per-slot verdict is printed: a score nobody can audit is
worth as little as no score.

THE VOCABULARIES ARE PARSED OUT OF `crew/description.md`. They are not repeated here. The two roles
are shown that file and this scorer reads the same words out of it, so editing the schema changes
what is written AND what is scored, together. Five separate defects in this codebase have come from
one declaration living in two places; this is the one place it would be least visible, because a
scorer that has quietly drifted from its schema still returns a plausible number every round.

    python foresight.py --round 3        score one round from what is on disk
    python foresight.py --self-test      the scorer against hand-written pairs
"""
from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_MD = os.path.join(HERE, "crew", "description.md")

# THE SLOTS, IN ORDER, AND WHICH ARE SCORED. `free` is carried and never compared -- description.md
# says why at length: a scored free slot is a slot the writer games toward whatever the other role
# is likely to say, and it is the one channel in this loop that has ever produced "like a flower".
SLOTS = ["form", "topology", "count", "surface", "chem", "time", "free"]
SCORED = ["form", "topology", "count", "surface", "chem", "time"]

# Words that carry no shape information. Deliberately tiny: a long stoplist starts making judgements
# about which words matter, which is the scorer deciding what a description means.
_STOP = {"a", "an", "the", "of", "at", "in", "on", "to", "and", "or", "with", "is", "are", "it",
         "its", "then", "from", "into", "that", "this", "some", "very", "quite", "still"}


def vocab():
    """{slot: {suggested words}} -- read from description.md, never declared here.

    Each slot's `###` section lists its starting words in backticks. A slot with no backticked words
    (`count`, whose values are numbers; `free`, which is unscored) simply gets an empty set and
    falls through to token overlap, which is the right behaviour rather than a special case.
    """
    try:
        with open(SCHEMA_MD) as f:
            md = f.read()
    except OSError:
        print(f"[foresight] {SCHEMA_MD} is missing -- scoring with no vocabularies, on token "
              f"overlap alone")
        return {s: set() for s in SLOTS}
    out = {}
    for s in SLOTS:
        m = re.search(rf"^### {s}\s*$(.*?)(?=^### |\Z)", md, re.S | re.M)
        body = m.group(1) if m else ""
        # THE `anchors:` LINE ONLY, not every backtick in the section. Reading the whole section
        # scooped up any word the PROSE quoted: `form`, named in the topology section's explanation
        # of what it took off `form`, became an anchor of `topology` -- so any answer containing the
        # word "form" matched. Caught by printing the vocabulary sizes after adding a slot, which is
        # the only reason it was caught at all: 28 anchors where 27 were written is not visible in
        # a score.
        #
        # SINGULARISED WITH `_tokens`, or the vocabulary never matches the text: `tips` in the
        # schema against "at the tip of each arm" in a reply, where the reply is stemmed and the
        # schema was not, is the plural problem again on the other side of the comparison.
        am = re.search(r"^\*\*anchors:\*\*(.*)$", body, re.M)
        out[s] = ({t for w in re.findall(r"`([a-z][a-z-]+)`", am.group(1)) for t in _tokens(w)}
                  if am else set())
    return out


def parse(text):
    """One role's reply -> {slot: value}. Tolerant, because a role that adds a stray line is not a
    failed measurement.

    Only the six known keys are taken, and the LAST occurrence of each wins: a model that writes a
    preamble and then the form has given its answer in the form. A missing slot comes back absent
    rather than empty, so `score` can tell "did not answer" from "answered nothing" -- the two mean
    opposite things about a forecast.

    JSON IS ACCEPTED, AND FINDING THAT OUT IS WHY THE BASIS TEST EXISTS. Its first run parsed 2 of
    12 replies. The other ten were not bad answers -- they were the SAME six slots, correct and
    well-filled, wrapped in a ```json fence, because every other role in this loop answers in JSON
    and the models carried the house habit over. A schema that only accepted `key: value` would have
    scored the campaign's foresight at 0.371 over two runs while ten good forecasts sat on disk
    unread, and the number would have looked entirely plausible. Instrument blindness, in the
    epistemic layer, on its first day.

    The line form stays primary because it is what `description.md` shows both roles. This just
    stops the parser from being the thing that decides whether a correct answer counts.
    """
    raw = str(text or "")
    out = {}
    # JSON FIRST, then lines over the same text -- so a reply that fences its JSON and then adds a
    # stray `count: 9` line still ends up with the line's value, and the LAST-wins rule holds
    # across both forms rather than only within one.
    for blob in re.findall(r"\{[^{}]*\}", raw, re.S):
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                if str(k).lower() in SLOTS and v not in (None, ""):
                    out[str(k).lower()] = str(v).strip()
    for line in raw.splitlines():
        m = re.match(r"^\s*[-*]?\s*\**\s*[\"']?(form|topology|count|surface|chem|time|free)[\"']?\s*\**\s*"
                     r":\s*(.+?)\s*,?\s*$", line, re.I)
        if m:
            v = m.group(2).strip().strip("`*\"',").strip()
            # A JSON LINE IS NOT A FORM LINE. Inside a fenced object every slot is also a
            # `"form": "sphere",` line, and letting the line rule re-read it would be harmless --
            # except that it would also re-read `"count": 0,` as the string "0" after the JSON pass
            # had it as an int. Same value, two types, and `_count` is the only consumer that cares.
            # Cheaper to skip a line the JSON pass already claimed than to normalise types twice.
            if v and not (m.group(1).lower() in out and line.lstrip().startswith(('"', "'"))):
                out[m.group(1).lower()] = v
    return out


def _tokens(v):
    """Content words, crudely singularised.

    THE PLURAL COST A WHOLE SLOT. Without this, `at the tips` and `at the tip of each arm` share no
    token at all once the stopwords are gone -- {tips} against {tip, each, arm} -- and score a clean
    zero for saying the same thing. The campaign's own vocabulary is full of the pair (arm/arms,
    lobe/lobes, tip/tips), so it is not an edge case, it is the common case for the two slots that
    describe repeated features. Trailing -s only, and only on words long enough that it is not the
    word: no stemmer, because a stemmer would start making decisions about meaning.
    """
    out = []
    for w in re.findall(r"[a-z][a-z-]*", str(v).lower()):
        if w in _STOP:
            continue
        out.append(w[:-1] if (len(w) > 3 and w.endswith("s") and not w.endswith("ss")) else w)
    return out


def _count(v):
    """A count slot -> (lo, hi), or None if it is not a number at all.

    A RANGE IS NOT A HEDGE and description.md says so: eleven arms of which three are stubby is
    honestly 8-11. So a range is parsed as an interval and compared as one.
    """
    s = str(v).lower().replace("–", "-").replace("~", "")
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = re.search(r"\d+", s)
    return (int(m.group(0)), int(m.group(0))) if m else None


def _score_count(f, o):
    """Overlapping intervals agree. Otherwise the gap is scored against the observed magnitude.

    NOT A BINARY. Forecasting 9 arms and seeing 11 is a different quality of knowledge from
    forecasting 0 and seeing 11, and a binary would record them identically -- which would let the
    campaign's foresight look flat while it was in fact getting steadily closer.
    """
    a, b = _count(f), _count(o)
    if a is None or b is None:
        return None
    if a[1] >= b[0] and b[1] >= a[0]:
        return 1.0
    gap = b[0] - a[1] if a[1] < b[0] else a[0] - b[1]
    scale = max(1, b[1])
    return max(0.0, 1.0 - gap / scale)


def _score_words(f, o, allowed):
    """A phrase slot.

    Three cases, in order, and the order is what makes it defensible:

      both name a vocabulary word   the OVERLAP of the two word-sets, not whether they intersect.
                                    `sphere` vs `star` is a clean miss and must score 0 even though
                                    both sentences share "the" and "body" -- token overlap alone
                                    would give that pair a respectable number and the whole score
                                    would drift upward.

                                    A SET, BECAUSE ANY-INTERSECTION REWARDS HEDGING, and the basis
                                    test caught it doing so on the first try: `form` read 1.00 on
                                    all six pairs. "lobed sphere, nine bulges" names BOTH `lobed`
                                    and `sphere`, so it matched an eye that saw a plain sphere AND
                                    would have matched one that saw a lobed body -- a forecast that
                                    cannot lose, scored as a forecast that was right. Divided by
                                    the larger set, naming two bodies where the eye named one is
                                    worth 0.5 of naming the right one alone.
      one names one, the other not  0.5 at most, via overlap: a partial answer, scored partially.
      neither names one            token overlap. This is the case description.md expects for a
                                    body the campaign has never seen, where forcing a word would be
                                    the morphology classifier's mistake again.
    """
    tf, to = set(_tokens(f)), set(_tokens(o))
    if not tf or not to:
        return None
    vf, vo = tf & allowed, to & allowed
    if vf and vo:
        return len(vf & vo) / max(len(vf), len(vo))
    inter = len(tf & to)
    jac = inter / len(tf | to) if (tf | to) else 0.0
    return min(0.5, jac) if (vf or vo) else jac


def score(forecast_text, observed_text, vocabularies=None):
    """One run -> {slot: score or None}, plus the mean over the slots that could be compared.

    A slot only one side filled scores None and is EXCLUDED from the mean rather than counted as
    zero. A forecast that did not answer is a hole in the measurement, not evidence that the
    knowledge is wrong, and scoring it as a miss would make a broken parse look like a failed
    prediction -- the single easiest way for this number to lie.
    """
    V = vocabularies or vocab()
    f, o = parse(forecast_text), parse(observed_text)
    per = {}
    for s in SCORED:
        if s not in f or s not in o:
            per[s] = None
            continue
        per[s] = _score_count(f[s], o[s]) if s == "count" else _score_words(f[s], o[s], V.get(s, set()))
    got = [v for v in per.values() if v is not None]
    return {
        "per_slot": per,
        "compared": len(got),
        "foresight": round(sum(got) / len(got), 3) if got else None,
        "forecast": f,
        "observed": o,
    }


def split(blob, node):
    """The engine's fan-out join -> {run: text}.

    `run_round` collapses a fanned-out node into one string, `"[eye] r003_04: ..."` per item joined
    by blank lines, because that is what a downstream ROLE should read. This node is not a role and
    needs the items back. Parsing the join is the cost of not changing the engine's contract for the
    six nodes that are happy with it.
    """
    out, cur = {}, None
    for line in str(blob or "").splitlines():
        m = re.match(rf"^\[{re.escape(node)}\]\s+(\S+?):\s?(.*)$", line)
        if m:
            cur = m.group(1)
            out[cur] = m.group(2) + "\n"
        elif cur is not None:
            out[cur] += line + "\n"
    return out


def round_score(forecast_blob, observed_blob):
    """The round -> per-run scores and the round's mean. What `round.foresight` files."""
    F, O = split(forecast_blob, "forecaster"), split(observed_blob, "eye")
    V = vocab()
    runs, means = {}, []
    for name in sorted(set(F) & set(O)):
        r = score(F[name], O[name], V)
        runs[name] = r
        if r["foresight"] is not None:
            means.append(r["foresight"])
    return {
        "runs": runs,
        "foresight": round(sum(means) / len(means), 3) if means else None,
        "scored_runs": len(means),
        # SAID OUT LOUD, EVERY ROUND. A run the Forecaster saw and the Eye did not, or the reverse,
        # is a hole in the measurement -- and a mean over the survivors reads identically whether it
        # was taken over eleven runs or two. The campaign has already been burned once by a
        # truncation that was invisible in its own output.
        "forecast_only": sorted(set(F) - set(O)),
        "observed_only": sorted(set(O) - set(F)),
    }


def render(rs):
    """The per-slot table. A score nobody can audit is worth as little as no score."""
    if not rs.get("runs"):
        return "no run had both a forecast and an observation -- foresight not measured this round"
    w = max(len(n) for n in rs["runs"])
    head = f"{'run':<{w}}  " + "  ".join(f"{s:^7}" for s in SCORED) + "   mean"
    lines = [head, "-" * len(head)]
    for n, r in rs["runs"].items():
        cells = "  ".join(("   .   " if r["per_slot"][s] is None else f"{r['per_slot'][s]:^7.2f}")
                          for s in SCORED)
        m = "  .  " if r["foresight"] is None else f"{r['foresight']:5.2f}"
        lines.append(f"{n:<{w}}  {cells}   {m}")
    lines.append("-" * len(head))
    lines.append(f"{'FORESIGHT':<{w}}  " + " " * (9 * len(SCORED) - 2)
                 + f"   {rs['foresight'] if rs['foresight'] is not None else '.'}"
                 + f"   over {rs['scored_runs']} run(s)")
    for k, label in (("forecast_only", "forecast but never observed"),
                     ("observed_only", "observed but never forecast")):
        if rs.get(k):
            lines.append(f"  NOT SCORED, {label}: {', '.join(rs[k])}")
    lines.append("  a dot is a slot one side did not fill -- excluded from the mean, not counted "
                 "as a miss")
    return "\n".join(lines)


# ---------------------------------------------------------------- checking the scorer
_CASES = [
    # (forecast, observed, expected mean, why this pair is in the test)
    ("form: sphere\ncount: 0\nsurface: smooth\nchem: uniform\ntime: grows throughout",
     "form: sphere\ncount: 0\nsurface: smooth\nchem: uniform\ntime: grows throughout",
     1.0, "identical forms must score 1.0 or nothing below is interpretable"),
    ("form: a smooth sphere\ncount: 0\nsurface: smooth\nchem: uniform\ntime: grows throughout",
     "form: a star with arms\ncount: 11\nsurface: smooth\nchem: at the tips\ntime: appears halfway",
     None, "the campaign's worst case: forecast the average run, meet r013_05. Must score LOW"),
    # 9 vs 11 is a gap of 2 against an observed magnitude of 11: 1 - 2/11 = 0.818. Written out
    # because the first version of this case asserted 1.0, which would have made the count slot
    # binary in everything but name and hidden a campaign getting steadily closer.
    ("count: 9", "count: 11", 0.818, "9 vs 11 -- graded, not binary"),
    ("count: 8-11", "count: 11", 1.0, "an overlapping range agrees"),
    ("count: 0", "count: 11", 0.0, "0 vs 11 is the miss a binary would score the same as 9 vs 11"),
    # THE PLURAL. Both say the activator is at the ends of the arms; before `_tokens` singularised,
    # they shared no token and scored 0.0 -- a slot agreeing perfectly and reading as a total miss.
    ("chem: at the tips", "chem: at the tip of each arm", (0.2, 1.0),
     "singular/plural must not read as disagreement"),
]


def _self_test():
    V = vocab()
    print(f"vocabularies from crew/description.md: "
          + ", ".join(f"{s}={len(V[s])}" for s in SLOTS) + "\n")
    bad = 0
    for f, o, want, why in _CASES:
        r = score(f, o, V)
        got = r["foresight"]
        # `want` is a number for an exact score, a (lo, hi) for a bound, or None for "must be LOW"
        # -- and LOW is stated as a number here rather than left to the reader, because a threshold
        # in a comment stops describing the code the first time either changes.
        if want is None:
            ok = got is not None and got < 0.34
        elif isinstance(want, tuple):
            ok = got is not None and want[0] <= got <= want[1]
        else:
            ok = got is not None and abs(got - want) < 5e-4
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {got}   {why}")
        if not ok or want is None or isinstance(want, tuple):
            print(f"        per slot: {r['per_slot']}")
    print(f"\n{'PASS' if not bad else 'FAIL'}: {bad} case(s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.round is None:
        ap.error("give --round N or --self-test")
    p = os.path.join(HERE, "campaign", f"round_{a.round:03d}.json")
    if not os.path.exists(p):
        print(f"{p} does not exist")
        return 1
    d = json.load(open(p))
    print(render(round_score(d.get("forecast"), d.get("observed"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
