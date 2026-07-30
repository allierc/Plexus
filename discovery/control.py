"""control -- the anti-rabbit-hole control law, as a DETERMINISTIC script.

Robin's authors observed that their agentic orchestrator "almost always called tools in the same
order", and replaced it with a streamlined notebook. We follow that: the loop below is ordinary
control flow. Language models are called only where judgement is genuinely required -- proposing
an edit, watching a movie, writing a causal description. Ranking is done by MEASURING.

--------------------------------------------------------------------------------------------
ONE ROUND
--------------------------------------------------------------------------------------------
    propose B          one legal edit each, drawn across DISTINCT clusters
    critic             free rejection: ill-typed, unmet precondition, already evaluated
    tier 1             cheap smoke (small mesh, short horizon) -- a GATE, never evidence
    tier 2             full run on the partition, full metric bank, per-frame
    rank               by measured metrics; BTL over pairwise where the order is ill-defined
    truncate           keep the top K; LOSERS ARE DROPPED, NOT REFINED
    starve / freeze    within-cluster competition; a stalled cluster is frozen and its budget
                       reallocated the same round
    terminate          objective met / all clusters frozen / dry / budget spent

The five pathologies this answers, from our own record:
  1 depth-first drift        -> batch and truncate; never re-expand a single best node
  2 no terminal-state rule   -> the Supervisor owns it, computed from statistics
  3 near-duplicates          -> proximity clusters compete internally and starve together
  4 eye/number divergence    -> the Watcher gates promotion
  5 goal drift               -> the campaign config is held by the Supervisor, not the workers
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from composition_space import CompositionGraph, seed
from hypothesis import HypothesisRegister
from run_record import Claim, RunArchive, comp_hash, wilson


# ============================================================================ campaign config
@dataclass
class CampaignConfig:
    """Held by the Supervisor. Workers cannot amend it; amending it is a logged act.

    This is countermeasure #5. An agent elaborates whatever is in front of it and cannot detect
    its own drift; every course correction in the hand-run record came from outside the loop.
    """
    name: str = "okuda_growth_driven"
    objective: str = ("find a composition that produces a sustained tube as a GROWTH-DRIVEN "
                      "quasi-static equilibrium -- surviving relaxation, sharing load between "
                      "tube and body, requiring no explicit extrusion force -- or establish "
                      "that no composition in the searched space can, and say which capability "
                      "is missing")
    # success criteria, authored BEFORE the search (falsifiable, not an impression)
    success: dict = field(default_factory=lambda: {
        "aspect_final": ">= 3.0",       # there is a tube at the end, not only a transient
        "retention": ">= 0.6",          # it is not a peak that collapses
        "Q": ">= 0.5",                  # it survives driver-off relaxation
        "no_extrude": True,             # achieved WITHOUT the forcing node
    })
    batch: int = 24                 # B: candidates proposed per round
    keep_tier1: int = 8             # survive the cheap gate
    keep_truncate: int = 3          # survive ranking; the rest are DROPPED, never refined
    freeze_after: int = 6           # cluster evaluations with no best-score gain -> frozen
    dry_rounds_to_escalate: int = 2
    max_rounds: int = 500
    budget_runs: int = 5000
    min_samples_promote: int = 12   # no promotion below this (D9)
    stage_gate: int = 2             # opened by escalation
    thresholds: dict = field(default_factory=lambda: {
        "established_rate": 0.60, "structural_rate": 0.10, "necessity_drop": 0.50})

    def to_dict(self):
        return asdict(self)


# ============================================================================ clusters
class ProximityIndex:
    """Groups compositions by structural distance so NEAR-DUPLICATES COMPETE WITHIN A CLUSTER.

    Pathology #3: thirty rounds explored perhaps four genuinely distinct ideas. Without a notion
    of distance, twenty variants of one idea consume twenty times the budget of one idea -- and
    their collective failure is then misread as strong evidence against the idea rather than as
    one observation repeated.
    """

    def __init__(self, radius=2.0):
        self.radius = radius
        self.centroids = []          # [(encoding, cluster_id)]
        self.clusters = {}           # cid -> dict(members, evals, best, frozen, why)

    def assign(self, g: CompositionGraph):
        e = g.encode()
        for c, cid in self.centroids:
            if float(np.abs(e - c).sum()) <= self.radius:
                self.clusters[cid]["members"].add(comp_hash(g))
                return cid
        cid = f"K{len(self.centroids):03d}"
        self.centroids.append((e, cid))
        self.clusters[cid] = dict(members={comp_hash(g)}, evals=0, best=-np.inf,
                                  frozen=False, why="")
        return cid

    def record(self, cid, score):
        c = self.clusters[cid]
        c["evals"] += 1
        if score > c["best"] + 1e-9:
            c["best"] = score
            c["stale"] = 0
        else:
            c["stale"] = c.get("stale", 0) + 1

    def freeze_stalled(self, freeze_after):
        """A cluster with >K evaluations and no best-score gain is frozen; budget reallocates."""
        newly = []
        for cid, c in self.clusters.items():
            if c["frozen"]:
                continue
            if c.get("stale", 0) >= freeze_after:
                c["frozen"] = True
                c["why"] = (f"{c['evals']} evaluations, no best-score gain in the last "
                            f"{c['stale']} -- budget reallocated")
                newly.append(cid)
        return newly

    def active(self):
        return [cid for cid, c in self.clusters.items() if not c["frozen"]]

    def summary(self):
        return {cid: {k: (list(v) if isinstance(v, set) else v) for k, v in c.items()}
                for cid, c in self.clusters.items()}


# ============================================================================ ranking
def rank_btl(items, compare, n_pairs=None, iters=200):
    """Bradley-Terry-Luce strengths from pairwise comparisons.

    Robin ranks <=25 hypotheses by full round robin and >25 by 300 random pairs, aggregating with
    BTL rather than win/loss tallies -- because a distributed tournament plus BTL is their fix for
    POSITION BIAS, which simple tallies are highly susceptible to.

    `compare(a, b) -> 1.0 if a wins, 0.0 if b wins, 0.5 for a tie`. Here the comparator is the
    METRIC BANK, not a debate: our experiments are cheap, so where ground truth is available an
    opinion about it is a step backwards.
    """
    n = len(items)
    if n < 2:
        return {0: 1.0} if n == 1 else {}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if n > 25 and n_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=min(n_pairs, len(pairs)), replace=False)
        pairs = [pairs[k] for k in idx]
    w = np.zeros((n, n))
    for i, j in pairs:
        r = compare(items[i], items[j])
        w[i, j] += r
        w[j, i] += 1.0 - r
    p = np.ones(n)
    for _ in range(iters):                       # standard MM update
        new = np.zeros(n)
        for i in range(n):
            num = w[i].sum()
            den = sum((w[i, j] + w[j, i]) / (p[i] + p[j]) for j in range(n) if j != i)
            new[i] = num / den if den > 1e-12 else p[i]
        s = new.sum()
        p = new / s * n if s > 1e-12 else p
    return {i: float(p[i]) for i in range(n)}


# ============================================================================ scoring
def score_run(summary, cfg: CampaignConfig):
    """The campaign's scalar objective. Deliberately explicit and auditable.

    A run that is not valid evidence scores -inf: an inert operator or a saturated buffer must
    never win a tournament.
    """
    if summary.get("inert_operators") or summary.get("saturated"):
        return -math.inf
    a = float(summary.get("aspect_final", 0.0))
    r = float(summary.get("retention", 0.0))
    q = summary.get("Q")
    q = float(q) if q is not None else r        # retention stands in for Q when Q is not run
    # a tube that survives is worth far more than a transient one
    return a * (0.25 + 0.75 * min(1.0, q))


def meets_success(summary, cfg: CampaignConfig, has_extrude: bool):
    s = cfg.success
    ok = (float(summary.get("aspect_final", 0)) >= 3.0
          and float(summary.get("retention", 0)) >= 0.6
          and float(summary.get("Q", summary.get("retention", 0))) >= 0.5)
    if s.get("no_extrude") and has_extrude:
        ok = False
    return ok


# ============================================================================ supervisor
class Supervisor:
    """Holds the objective, the budget and the terminal-state decision. Never does the work."""

    def __init__(self, cfg: CampaignConfig, root):
        self.cfg = cfg
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.prox = ProximityIndex()
        self.reg = HypothesisRegister(os.path.join(root, "hypotheses.jsonl"))
        self.arch = RunArchive(os.path.join(root, "_archive"))
        self.round = 0
        self.spent = 0
        self.dry = 0
        self.best = -math.inf
        self.log_path = os.path.join(root, "supervisor.jsonl")
        cfgp = os.path.join(root, "campaign.json")
        if not os.path.exists(cfgp):
            json.dump(cfg.to_dict(), open(cfgp, "w"), indent=1)

    # ---------------------------------------------------------------- terminal state
    def terminal(self):
        """(stop, reason). Computed from statistics, not from anyone's patience."""
        c = self.cfg
        if self.spent >= c.budget_runs:
            return True, f"budget spent ({self.spent}/{c.budget_runs} runs)"
        if self.round >= c.max_rounds:
            return True, f"round cap ({c.max_rounds})"
        if self.prox.clusters and not self.prox.active():
            return False, "ESCALATE: all clusters frozen"
        if self.dry >= c.dry_rounds_to_escalate:
            return False, f"ESCALATE: {self.dry} dry rounds (no new cluster, no best-score gain)"
        return False, "continue"

    def escalate(self, operator_request=None):
        """The branch a human took by hand last time.

        Either open the next stage gate, or record an OPERATOR REQUEST -- a mechanism the
        proposer wanted and the language could not express. The request is a deliverable: it is
        this campaign's contribution to the operator atlas.
        """
        rec = {"round": self.round, "t": time.time()}
        if self.cfg.stage_gate < 3:
            self.cfg.stage_gate += 1
            rec["action"] = f"opened stage gate -> {self.cfg.stage_gate}"
        elif operator_request:
            rec["action"] = "operator_request"
            rec["request"] = operator_request
        else:
            rec["action"] = "exhausted: no further stage, no operator request filed"
        self.dry = 0
        self._log(rec)
        return rec

    def _log(self, obj):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    # ---------------------------------------------------------------- one round's bookkeeping
    def observe(self, results):
        """results: [(graph, summary, hypothesis_id)]. Returns the round report."""
        self.round += 1
        gained = False
        new_clusters = 0
        for g, summ, hid in results:
            cid_before = len(self.prox.clusters)
            cid = self.prox.assign(g)
            if len(self.prox.clusters) > cid_before:
                new_clusters += 1
            sc = score_run(summ, self.cfg)
            self.prox.record(cid, sc)
            self.spent += 1
            if sc > self.best + 1e-9:
                self.best = sc
                gained = True
        frozen = self.prox.freeze_stalled(self.cfg.freeze_after)
        self.dry = 0 if (gained or new_clusters) else self.dry + 1
        surprise = self.reg.surprise_rate(self.round)
        mix_frac, mix_why = self.reg.advise_mix(self.round)
        stop, reason = self.terminal()
        rep = {"round": self.round, "spent": self.spent, "best": None if self.best == -math.inf
               else round(self.best, 3), "new_clusters": new_clusters,
               "frozen": frozen, "active_clusters": len(self.prox.active()),
               "dry": self.dry, "surprise": surprise,
               "next_confirmatory_frac": mix_frac, "mix_why": mix_why,
               "stop": stop, "reason": reason}
        self._log(rep)
        return rep


# ============================================================================ the round
def propose_batch(frontier, cfg: CampaignConfig, prox: ProximityIndex, rng):
    """B candidates, ONE legal edit each, drawn across DISTINCT clusters.

    Countermeasure #1: never expand a single best node repeatedly. The batch is spread over
    clusters, so a good idea cannot monopolise the round and a bad one cannot be lovingly
    refined.
    """
    out, seen = [], set()
    per_cluster = {}
    order = sorted(frontier, key=lambda g: rng.random())
    for g in order:
        cid = prox.assign(g) if prox.centroids else "K000"
        if prox.clusters.get(cid, {}).get("frozen"):
            continue
        if per_cluster.get(cid, 0) >= max(2, cfg.batch // 4):
            continue
        edits = g.legal_edits(cfg.stage_gate)
        rng.shuffle(edits)
        for e, lbl in edits:
            try:
                child, _ = g.apply(e)
            except Exception:
                continue
            h = comp_hash(child)
            if h in seen:
                continue
            ok, _why = child.is_runnable()
            if not ok:
                continue                       # the CRITIC: free rejection, no cluster time
            seen.add(h)
            per_cluster[cid] = per_cluster.get(cid, 0) + 1
            out.append((child, lbl, g))
            break
        if len(out) >= cfg.batch:
            break
    return out


def truncate(ranked, keep):
    """Keep the top `keep`. THE LOSERS ARE DROPPED, NOT REFINED.

    They remain in the archive as evaluated evidence and are never re-expanded. This is Robin's
    batch-and-truncate, and it is the single most direct answer to depth-first drift.
    """
    return ranked[:keep], ranked[keep:]


if __name__ == "__main__":
    import tempfile

    cfg = CampaignConfig()
    print("=" * 78)
    print(f"CAMPAIGN  {cfg.name}")
    print("=" * 78)
    print(f"objective: {cfg.objective}\n")
    print(f"success  : {json.dumps(cfg.success)}")
    print(f"batch B={cfg.batch}  tier1 keep={cfg.keep_tier1}  truncate keep={cfg.keep_truncate}  "
          f"freeze_after={cfg.freeze_after}\n")

    with tempfile.TemporaryDirectory() as d:
        sup = Supervisor(cfg, d)
        rng = np.random.default_rng(0)

        # one simulated round, to exercise the control law end to end
        from composition_space import reference_recipes
        frontier = [seed("substrate")] + list(reference_recipes().values())
        batch = propose_batch(frontier, cfg, sup.prox, rng)
        print(f"[propose] {len(batch)} candidates, one legal edit each, across clusters")
        for child, lbl, parent in batch[:6]:
            print(f"    {lbl:34} -> {comp_hash(child)}  {child.name_region()}")

        # fake summaries so the ranking + truncation + freezing are exercised
        results = []
        for i, (child, lbl, parent) in enumerate(batch):
            summ = {"aspect_final": 1.0 + 2.5 * rng.random(),
                    "retention": rng.random(), "inert_operators": [], "saturated": False}
            results.append((child, summ, f"R1.h{i}"))

        scored = sorted(results, key=lambda r: -score_run(r[1], cfg))
        kept, dropped = truncate(scored, cfg.keep_truncate)
        print(f"\n[rank+truncate] kept {len(kept)}, DROPPED {len(dropped)} (never refined)")
        for g, s, _ in kept:
            print(f"    keep {comp_hash(g)}  score {score_run(s, cfg):.2f}  "
                  f"aspect {s['aspect_final']:.2f} retention {s['retention']:.2f}")

        rep = sup.observe(results)
        print(f"\n[supervisor] {json.dumps({k: v for k, v in rep.items() if k != 'mix_why'})}")
        print(f"    mix: {rep['mix_why']}")

        # BTL sanity: a strict ordering must be recovered
        vals = [3.0, 1.0, 2.0, 0.5]
        st = rank_btl(vals, lambda a, b: 1.0 if a > b else 0.0)
        order = [vals[i] for i in sorted(st, key=lambda k: -st[k])]
        assert order == sorted(vals, reverse=True), order
        print(f"\n[BTL] recovered strict order {order} from pairwise comparisons")
        print("\ncontrol law OK")
