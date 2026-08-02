"""lever_map -- the campaign's PRODUCT: a causal map of the mechanism space, and its coverage.

    "Understanding is earned by interrogating the decomposition as a causal system -- learning
     what each operator and parameter does on its own and, the harder part, what it does in
     COMBINATION, since mixtures rarely surrender their causal structure to inspection. The
     loop's product is a causal lever-map of the model."          -- plexus2, Loop I

WHY A MAP AND NOT A TARGET
--------------------------
The campaign was first specified around one question -- "produce a sustained tube as a
growth-driven equilibrium". That framing has a structural defect which the throughput arithmetic
exposed: with a single target the search CONVERGES, freezes every cluster within days, and
stops, having learned one fact. It also makes most runs worthless: a composition that does not
make a tube teaches nothing about tubes.

With COVERAGE as the objective every run is informative, because every run fills a cell. "Which
composition makes a tube" becomes a QUERY against the map, alongside "which operators are
necessary for branching", "does the (chi,gamma) phase diagram reproduce", and "which families
provably cannot". The campaign then runs until the levers are characterised, which is a
weeks-scale programme rather than a days-scale search.

THE CELLS
---------
  SOLO         for each operator (and each structural implementation): what does adding it to a
               base composition do? Its effect ALONE.
  PAIR         for each pair present together: is the joint effect what the two solo effects
               predict (ADDITIVE), or not (INTERACTION)? Interactions are the expensive
               knowledge -- they are exactly what cannot be read off the code.
  ROUTING      for each legal connection: morphogen -> growth.gate vs -> divide.axis vs
               -> extrude.site. Same operators, different mechanism.
  NECESSITY    for each observed phenotype: which operators, when ablated, destroy it.
  PHENOTYPE    which regions of the space produce which observed morphology (labels are
               DISCOVERED by the analysts, never declared in advance).

Coverage is the fraction of cells carrying enough evidence to state a verdict. A campaign
reports its progress as coverage, and its LEARNING RATE as the surprise rate; when the latter
collapses the map has stopped growing even while the GPUs are still busy.
"""
from __future__ import annotations

import itertools
import json
import os
from collections import defaultdict

from run_record import wilson


class LeverMap:
    """Accumulates evidence into cells and reports coverage. Append-only; never overwritten."""

    MIN_SAMPLES = 6                  # per cell, before a verdict may be stated

    def __init__(self, path):
        self.path = path
        self.obs = []                # [{comp, ops, impls, conns, phenotype, score, metrics}]
        if os.path.exists(path):
            self.obs = [json.loads(l) for l in open(path) if l.strip()]

    # ---------------------------------------------------------------- record
    def add(self, comp_hash, graph, phenotype, score, metrics, run_id=None):
        rec = {"comp": comp_hash,
               "ops": sorted(set(graph.op_names())),
               "impls": {o["op"]: graph.impl_of(o) for o in graph.ops},
               "conns": sorted(f"{graph._op_of(c['src'])}->{graph._op_of(c['dst'])}.{c['slot']}"
                               for c in graph.conns),
               "phenotype": phenotype, "score": float(score),
               "metrics": {k: v for k, v in (metrics or {}).items()
                           if isinstance(v, (int, float))},
               "run_id": run_id}
        self.obs.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    # ---------------------------------------------------------------- cells
    def _with(self, op):
        return [o for o in self.obs if op in o["ops"]]

    def _without(self, op):
        return [o for o in self.obs if op not in o["ops"]]

    def solo(self):
        """Effect of each operator ALONE: mean score with it minus mean score without."""
        out = {}
        ops = sorted({op for o in self.obs for op in o["ops"]})
        for op in ops:
            a, b = self._with(op), self._without(op)
            if len(a) < 2 or len(b) < 2:
                out[op] = {"n_with": len(a), "n_without": len(b), "verdict": "insufficient"}
                continue
            ma = sum(x["score"] for x in a) / len(a)
            mb = sum(x["score"] for x in b) / len(b)
            enough = min(len(a), len(b)) >= self.MIN_SAMPLES
            out[op] = {"n_with": len(a), "n_without": len(b),
                       "delta": round(ma - mb, 3),
                       "phenotypes_with": _counts(a), "verdict":
                       ("raises" if ma - mb > 0.25 else "lowers" if ma - mb < -0.25 else "neutral")
                       if enough else "insufficient"}
        return out

    def pairs(self):
        """INTERACTION: is the joint effect the sum of the solo effects?

        This is the expensive half of the map, and the half that cannot be read off the code.
        A pair whose joint effect exceeds the additive prediction is a genuine interaction and
        the most valuable single entry the campaign can produce.
        """
        solo = self.solo()
        ops = [op for op, v in solo.items() if v.get("verdict") not in (None, "insufficient")]
        base = (sum(o["score"] for o in self.obs) / len(self.obs)) if self.obs else 0.0
        out = {}
        for a, b in itertools.combinations(sorted(ops), 2):
            both = [o for o in self.obs if a in o["ops"] and b in o["ops"]]
            if len(both) < 2:
                out[f"{a}+{b}"] = {"n": len(both), "verdict": "insufficient"}
                continue
            m = sum(o["score"] for o in both) / len(both)
            predicted = base + solo[a]["delta"] + solo[b]["delta"]
            resid = m - predicted
            out[f"{a}+{b}"] = {
                "n": len(both), "observed": round(m, 3), "additive_prediction": round(predicted, 3),
                "interaction": round(resid, 3),
                "verdict": ("insufficient" if len(both) < self.MIN_SAMPLES else
                            "SYNERGY" if resid > 0.5 else
                            "ANTAGONISM" if resid < -0.5 else "additive")}
        return out

    def routing(self):
        out = defaultdict(list)
        for o in self.obs:
            for c in o["conns"]:
                out[c].append(o["score"])
        return {k: {"n": len(v), "mean": round(sum(v) / len(v), 3),
                    "verdict": "characterised" if len(v) >= self.MIN_SAMPLES else "insufficient"}
                for k, v in out.items()}

    def phenotypes(self):
        return _counts(self.obs)

    # ---------------------------------------------------------------- coverage
    def coverage(self):
        """The campaign's progress measure. Fraction of cells that can state a verdict."""
        parts = {}
        for name, cells in (("solo", self.solo()), ("pair", self.pairs()),
                            ("routing", self.routing())):
            done = sum(1 for v in cells.values() if v.get("verdict") not in
                       (None, "insufficient"))
            parts[name] = {"covered": done, "total": len(cells),
                           "frac": round(done / max(1, len(cells)), 3)}
        tot = sum(p["total"] for p in parts.values())
        cov = sum(p["covered"] for p in parts.values())
        parts["overall"] = {"covered": cov, "total": tot,
                            "frac": round(cov / max(1, tot), 3), "n_runs": len(self.obs)}
        return parts

    # ---------------------------------------------------------------- report
    def render(self, path):
        cov, solo, pair = self.coverage(), self.solo(), self.pairs()
        L = ["# Causal lever-map", "",
             f"_{cov['overall']['n_runs']} runs · coverage "
             f"**{cov['overall']['frac']:.0%}** ({cov['overall']['covered']}/"
             f"{cov['overall']['total']} cells)_", "",
             "The campaign's product. Specific questions are queries against this table.", "",
             "## Coverage", "", "| block | covered | total | |", "|---|---|---|---|"]
        for k in ("solo", "pair", "routing"):
            L.append(f"| {k} | {cov[k]['covered']} | {cov[k]['total']} | {cov[k]['frac']:.0%} |")
        L += ["", "## Solo effects — what each operator does ALONE", "",
              "| operator | n(with) | n(without) | Δscore | verdict | phenotypes seen |",
              "|---|---|---|---|---|---|"]
        for op, v in sorted(solo.items(), key=lambda kv: -(kv[1].get("delta") or -99)):
            L.append(f"| `{op}` | {v.get('n_with',0)} | {v.get('n_without',0)} | "
                     f"{v.get('delta','—')} | {v['verdict']} | "
                     f"{_fmt(v.get('phenotypes_with', {}))} |")
        L += ["", "## Interactions — where the joint effect is NOT the sum", "",
              "_The expensive half of the map: what cannot be read off the code._", "",
              "| pair | n | observed | additive prediction | interaction | verdict |",
              "|---|---|---|---|---|---|"]
        inter = [(k, v) for k, v in pair.items() if v.get("verdict") in ("SYNERGY", "ANTAGONISM")]
        for k, v in sorted(inter, key=lambda kv: -abs(kv[1]["interaction"]))[:20]:
            L.append(f"| `{k}` | {v['n']} | {v['observed']} | {v['additive_prediction']} | "
                     f"**{v['interaction']:+}** | {v['verdict']} |")
        if not inter:
            L.append("| _(none established yet)_ | | | | | |")
        L += ["", "## Phenotypes observed", "", _fmt(self.phenotypes()), ""]
        open(path, "w").write("\n".join(L) + "\n")
        return path


def _counts(rows):
    c = defaultdict(int)
    for r in rows:
        c[r.get("phenotype") or "unlabelled"] += 1
    return dict(c)


def _fmt(d):
    return ", ".join(f"{k}×{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "—"


if __name__ == "__main__":
    import tempfile
    import numpy as np
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from composition_space import reference_recipes, seed
    from run_record import comp_hash

    with tempfile.TemporaryDirectory() as d:
        m = LeverMap(os.path.join(d, "map.jsonl"))
        rng = np.random.default_rng(0)
        recipes = list(reference_recipes().values()) + [seed("substrate")]
        phen = ["sphere", "bud", "spike", "tube", "exploded"]
        for i in range(60):                       # synthetic evidence, to exercise the cells
            g = recipes[i % len(recipes)]
            m.add(comp_hash(g), g, phen[i % len(phen)], float(rng.normal(2, 1)),
                  {"protr_peak": float(rng.normal(3, 1))})
        cov = m.coverage()
        print("coverage:", json.dumps(cov, indent=1))
        p = m.render(os.path.join(d, "lever_map.md"))
        print(open(p).read()[:1200])
        print("lever_map OK")
