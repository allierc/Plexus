#!/usr/bin/env python
"""vocab -- one name per measured thing, and a way back from the old ones.

PHASE 0, ITEM 20. Three of the campaign's four headline names misled, each differently:

    protr           reads as a LENGTH and is a RATIO. 1.0 is a sphere. Reading 1.62 as "1.62
                    of protrusion" is a mistake the name actively invites, and it was made.
    protr_final     became ambiguous the moment the evidence horizon existed: "final" is now
                    the last VALID frame, not the last frame, and the name says neither.
    Q               said nothing at all. It is the forced-versus-grown test -- the whole reason
                    the campaign exists -- and it was called a letter.

WHY BOTH NAMES SURVIVE. A rename across 92 references in one step leaves a window where half
the code writes a key the other half has stopped reading, and the failure is silent: a metric
that is absent reads as "not measured", which the loop is careful to treat as inconclusive
rather than zero -- so the bug would present as a round that quietly learns nothing. Writers
emit both, readers resolve through `canonical()`, and the archive on disk is never rewritten.
"""
from __future__ import annotations

CANONICAL = {
    "protr":               "elongation",
    "protr_peak":          "elongation_peak",
    "protr_final":         "elongation_at_end",
    "ta_protr_final":      "ta_elongation_at_end",
    "Q_protr_after_relax": "elongation_unforced",
    "Q_drop":              "elongation_lost_when_unforced",
}
MEANING = {
    "elongation": "95th-percentile cell radius / median, about the TISSUE centroid. 1.0 = sphere.",
    "elongation_peak": "the most elongated it ever got, over VALID frames only.",
    "elongation_at_end": "at the last VALID frame -- not simply the last frame.",
    "elongation_unforced": "what survives once growth and pushing stop. THE forced-vs-grown test.",
    "elongation_lost_when_unforced": "how much did not survive.",
}


def canonical(key):
    """The current name for a key, old or new. Unknown keys pass through untouched."""
    return CANONICAL.get(key, key)


def resolve(summary, key):
    """Read `key` from a summary written under EITHER vocabulary. None if truly absent.

    Deliberately returns None rather than 0.0 for a missing metric: the loop treats absent as
    inconclusive and zero as measured, and conflating them is how a broken reader becomes a
    scientific claim.
    """
    new = canonical(key)
    if new in summary:
        return summary[new]
    for old, cur in CANONICAL.items():
        if cur == new and old in summary:
            return summary[old]
    return summary.get(key)


if __name__ == "__main__":
    old = {"protr_peak": 4.03, "protr_final": 1.70}
    new = {"elongation_peak": 4.03, "elongation_at_end": 1.70}
    for d, label in ((old, "an archived record"), (new, "a record written today")):
        got = (resolve(d, "elongation_peak"), resolve(d, "protr_peak"))
        assert got == (4.03, 4.03), got
        print(f"  {label:26} -> both names resolve to {got[0]}")
    assert resolve({}, "elongation_peak") is None
    print("  a genuinely absent metric  -> None, never 0.0")
    print("\nvocab OK")
