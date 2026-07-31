#!/usr/bin/env python
"""scoreboard -- what the campaign is actually for, and how far along it is.

WHY THIS REPLACES A PASS/FAIL THRESHOLD
------------------------------------------------------------------------------------------------
`CampaignConfig.objective` says, in the same dataclass:

    "BUILD THE CAUSAL LEVER-MAP ... The product is a MAP, NOT A WINNER. Specific questions --
     which composition makes a sustained tube, which reproduces Okuda's (chi,gamma) phase
     diagram, which mechanism is necessary for branching -- are QUERIES AGAINST that map."

and then, fourteen lines further down, `meets_success` is four TUBE metrics ANDed together. The
stated objective is coverage; the coded criterion is a tube detector. It cannot express
undulation, cannot express branching, cannot express budding, and cannot express "we reproduced
the phase diagram" -- even though the `queries` field names all four morphologies explicitly.

Okuda's paper is titled for THREE behaviours -- undulation, tubulation and branching -- and the
tubulation figure has two distinct regimes (thin and thick). A criterion that measures how far
one protrusion sticks out can score at most one of the four, and scores the other three as
failures. Worse, it scores UNDULATION -- a successful reproduction of Figure 7 -- as the least
successful thing the campaign can do, because undulation is by definition many shallow bumps
rather than one deep protrusion.

So success is not a number to clear. It is two questions:

    1. WHAT DO WE KNOW?          how much of the mechanism map can state a verdict
    2. WHAT CAN WE REPRODUCE?    which of the paper's morphologies do we get, qualitatively

The derived tube threshold is not discarded -- it is DEMOTED to what it always was: the test for
whether a shape is a tube, i.e. the criterion for ONE cell of the scoreboard.

WHAT THIS DISSOLVES
------------------------------------------------------------------------------------------------
The 33 archived records carrying a broken survival measurement were about to be a dilemma: quarantine
them and a fallback silently promotes them to "success"; leave them and the ledger is poisoned.
Under a scoreboard neither happens. Success is not a boolean a single run can trip, so those runs
are simply evidence whose forced-versus-grown status is UNKNOWN -- an honest cell, neither pass
nor fail.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

# The paper's morphologies, with the (chi, gamma) Okuda reports and what would count as having
# reproduced each one. `criterion` is deliberately QUALITATIVE: the target is "the figure looks
# like his figure", not "a number cleared a bar".
OKUDA_TARGETS = {
    "undulation": dict(figure="Fig 7", chi=0.1, gamma=100.0,
                       criterion="many shallow bumps, not one deep protrusion: high spot count, "
                                 "low elongation",
                       measures=("spot_count", "elongation")),
    "thin_tube":  dict(figure="Fig 5a", chi=0.01, gamma=100.0,
                       criterion="ONE high-aspect protrusion of small diameter, activator at "
                                 "the tip",
                       measures=("elongation", "tube_diameter", "tip_activator")),
    "thick_tube": dict(figure="Fig 5b", chi=0.1, gamma=1.0,
                       criterion="one protrusion, LARGER diameter than the thin case at the "
                                 "same length",
                       measures=("elongation", "tube_diameter")),
    "branching":  dict(figure="Fig 6", chi=0.01, gamma=0.01,
                       criterion="a protrusion that BIFURCATES -- the tube count rises over time",
                       measures=("branch_count_over_time",)),
}

# Not an Okuda figure, but a phenotype this substrate produces and the campaign keeps meeting.
# Tracked so it stops being invisible: it has no cell in any current criterion.
OTHER_PHENOTYPES = {
    "budding": dict(figure="—", criterion="multiple localised outward buds from the shell "
                                          "(the 'coral' regime), no single dominant axis",
                    measures=("spot_count", "elongation")),
}

STATES = ("not_attempted", "attempted_failed", "partial", "reproduced")


def _blank():
    rows = {}
    for k, v in {**OKUDA_TARGETS, **OTHER_PHENOTYPES}.items():
        rows[k] = {"state": "not_attempted", "evidence": [], "note": "",
                   "figure": v["figure"], "criterion": v["criterion"]}
    return rows


class Scoreboard:
    """Append-only record of what has been reproduced, and of what is known."""

    def __init__(self, path):
        self.path = path
        self.rows = _blank()
        if os.path.exists(path):
            saved = json.load(open(path)).get("morphologies", {})
            for k, v in saved.items():
                if k in self.rows:
                    self.rows[k].update(v)

    def set(self, morphology, state, evidence=(), note=""):
        if morphology not in self.rows:
            raise KeyError(f"{morphology!r} not in {sorted(self.rows)}")
        if state not in STATES:
            raise ValueError(f"state {state!r} not in {STATES}")
        r = self.rows[morphology]
        r["state"] = state
        r["evidence"] = sorted(set(list(r["evidence"]) + list(evidence)))
        r["note"] = note or r["note"]
        return r

    # ---------------------------------------------------------------- the two questions
    def knowledge(self, lever_map=None, register=None):
        """QUESTION 1: what do we know? Every number here already existed; none was collected."""
        out = {}
        if lever_map is not None:
            cov = lever_map.coverage()
            out["map_coverage"] = cov["overall"]
            out["by_kind"] = {k: v for k, v in cov.items() if k != "overall"}
        if register is not None:
            hs = register.all()
            resolved = [h for h in hs if h.outcome in ("confirmed", "refuted")]
            out["claims_resolved"] = len(resolved)
            out["claims_open"] = len(hs) - len(resolved)
            s = register.surprise_rate()
            out["surprise_rate"] = None if s is None else round(s, 3)
        return out

    def reproduction(self):
        """QUESTION 2: what can we reproduce? Okuda's morphologies, counted honestly."""
        ok = [k for k in OKUDA_TARGETS if self.rows[k]["state"] == "reproduced"]
        part = [k for k in OKUDA_TARGETS if self.rows[k]["state"] == "partial"]
        return {"reproduced": sorted(ok), "partial": sorted(part),
                "of_okuda_targets": len(OKUDA_TARGETS),
                "score": f"{len(ok)}/{len(OKUDA_TARGETS)}"}

    def save(self, lever_map=None, register=None):
        json.dump({"morphologies": self.rows,
                   "knowledge": self.knowledge(lever_map, register),
                   "reproduction": self.reproduction()},
                  open(self.path, "w"), indent=1, default=str)
        return self.path

    def render(self, path, lever_map=None, register=None):
        k = self.knowledge(lever_map, register)
        rep = self.reproduction()
        mark = {"reproduced": "YES", "partial": "partial", "attempted_failed": "no",
                "not_attempted": "-"}
        L = ["# Campaign scoreboard", "",
             "_Success is not a threshold a run clears. It is two questions._", "",
             "## 1. What can we reproduce?", "",
             f"**{rep['score']} of Okuda's morphologies.**", "",
             "| morphology | figure | state | what would count | evidence |",
             "|---|---|---|---|---|"]
        for name, v in {**OKUDA_TARGETS, **OTHER_PHENOTYPES}.items():
            r = self.rows[name]
            L.append(f"| **{name}** | {r['figure']} | {mark[r['state']]} | {r['criterion']} | "
                     f"{', '.join(r['evidence']) or '—'} |")
        L += ["", "## 2. What do we know?", ""]
        if not k:
            # Never render a silent blank. An empty section reads as "nothing is known", when the
            # truth may be "nobody passed the ledger in". Say which it is.
            L.append("_Not available — this scoreboard was rendered without the map or the "
                     "hypothesis register, so knowledge could not be counted. This is a missing "
                     "input, NOT a finding of zero knowledge._")
        if k.get("map_coverage"):
            c = k["map_coverage"]
            L.append(f"- **map coverage {c['frac']:.0%}** — {c['covered']} of {c['total']} cells "
                     f"can state a verdict, over {c['n_runs']} runs")
            for kind, v in (k.get("by_kind") or {}).items():
                L.append(f"  - {kind}: {v['covered']}/{v['total']}")
        if "claims_resolved" in k:
            L.append(f"- **{k['claims_resolved']} claims resolved**, {k['claims_open']} open")
            if k.get("surprise_rate") is not None:
                L.append(f"- surprise rate {k['surprise_rate']} "
                         f"(the fraction of predictions that were wrong — the information rate)")
        L += ["", "## What this replaces", "",
              "A single pass/fail over four tube metrics. It could score at most one of the four",
              "morphologies, and scored **undulation — a successful reproduction of Figure 7 — as",
              "the least successful thing the campaign could do**, because undulation is by",
              "definition many shallow bumps rather than one deep protrusion. The derived tube",
              "threshold is not discarded; it decides the two tube rows above and nothing else.", ""]
        open(path, "w").write("\n".join(L))
        return path


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sb = Scoreboard(os.path.join(d, "scoreboard.json"))
        assert sb.reproduction()["score"] == "0/4"
        assert set(sb.rows) == {"undulation", "thin_tube", "thick_tube", "branching", "budding"}
        print("blank board:", sb.reproduction())

        sb.set("undulation", "reproduced", evidence=["r007c_02_abc123"],
               note="7 shallow bumps at chi=0.1 gamma=100, elongation 1.06")
        sb.set("thin_tube", "partial", evidence=["round_40_mc8"],
               note="elongation 1.74 but forced -- survival never validly measured")
        sb.set("branching", "attempted_failed", evidence=["round_44_base"],
               note="tube count never rose; the shape-to-chemistry arrow is missing")
        print("after 3 updates:", sb.reproduction())
        assert sb.reproduction()["score"] == "1/4"

        sb2 = Scoreboard(sb.save())                      # must survive a reload
        assert sb2.rows["undulation"]["state"] == "reproduced"
        assert sb2.rows["branching"]["state"] == "attempted_failed"
        print("reload OK")

        try:
            sb2.set("tubulation", "reproduced")
            raise AssertionError("unknown morphology must be refused")
        except KeyError as e:
            print("refused unknown morphology:", str(e)[:60])
        try:
            sb2.set("branching", "probably")
            raise AssertionError("unknown state must be refused")
        except ValueError as e:
            print("refused unknown state:", str(e)[:60])

        # against the real campaign, if it is present
        try:
            from lever_map import LeverMap
            from hypothesis import HypothesisRegister
            lm = LeverMap(os.path.join(HERE, "campaign", "lever_map.jsonl"))
            reg = HypothesisRegister(os.path.join(HERE, "campaign", "hypotheses.jsonl"))
            print("\n--- against the live campaign ---")
            print(json.dumps(sb2.knowledge(lm, reg), indent=1))
        except Exception as e:
            print(f"(live campaign unavailable: {type(e).__name__})")

        p = sb2.render(os.path.join(d, "scoreboard.md"))
        print("\n" + open(p).read()[:900])
        print("scoreboard OK")
