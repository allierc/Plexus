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
    objective: str = (
        "BUILD THE CAUSAL LEVER-MAP of the Okuda mechanism space: for every operator, every "
        "implementation and every routing, what does it do ALONE, and what does it do IN "
        "COMBINATION -- since mixtures rarely surrender their causal structure to inspection. "
        "The product is a map, not a winner. Specific questions -- which composition makes a "
        "sustained tube, which reproduces Okuda's (chi,gamma) phase diagram, which mechanism is "
        "necessary for branching -- are QUERIES AGAINST that map, and each is answered as a "
        "by-product of covering it.")
    # Why a map and not a target: with a single objective the search converges, freezes every
    # cluster within days, and stops. With coverage as the objective the campaign runs until
    # every lever is characterised alone AND in interaction -- which is a weeks-scale programme
    # and is what plexus2 Loop I actually asks for. It also makes every run informative: a
    # composition that fails to make a tube still fills a cell of the map.
    # success criteria, authored BEFORE the search (falsifiable, not an impression)
    # QUERIES against the map. None of these is "the" objective; each is a question the map
    # answers once the relevant cells are covered. They are recorded so the campaign can report
    # progress on them, not so it can stop at one.
    queries: dict = field(default_factory=lambda: {
        "sustained_tube": "a tube that survives relaxation, WITHOUT the extrude forcing node",
        "phase_diagram": "Okuda's four (chi,gamma) morphologies from ONE composition: "
                         "thin tube / thick tube / undulation / branching",
        "necessity": "for each observed phenotype, which operators are necessary",
        "impossibility": "which composition families provably CANNOT reach a phenotype",
    })
    success: dict = field(default_factory=lambda: {
        # Authored before the search, and REVISED twice. (M4, the instrument gate): the first
        # version used `retention` and a RATIO-valued Q, both of which the gate showed reward
        # STASIS -- a sphere that never moved scored 1.0. Only gate-admissible metrics appear
        # here now (protr_peak tau=+1.00, n_tubes tau=+1.00, protr_final tau=+0.67).
        #
        # SECOND REVISION 2026-07-31 -- THE 1.37 IS DERIVED, NOT TUNED. Do not "round it up".
        # ---------------------------------------------------------------------------------
        # The defect: 3.0 / 2.0 / 2.0 were authored as an IMPRESSION of "clearly a tube". Nobody
        # ever asked what the metric reads on a shape of known aspect ratio. The question went
        # unasked for months because `Q_protr_after_relax` was broken (it returned the constant
        # 1.014 -- run_one.quasi_static_Q), so the criterion was unreachable BY ACCIDENT and the
        # accident hid the fact that the numbers were never justified.
        #
        # `calibrate_success.py` derives them. protrusion = percentile(r,95)/median(r) is measured
        # on analytic capsules of KNOWN aspect ratio (no simulator in the loop), against the same
        # tube_analysis.protrusion_ratio the campaign uses:
        #       sphere (aspect 1, THE NULL) 1.000    aspect 1.5  1.204    aspect 2  1.378
        #       aspect 3  1.608     aspect 4  1.721     aspect -> inf  1.900 (the CEILING)
        # A capsule of aspect A has a parallel-sided barrel (A-1) diameters long; A = 2 is the
        # first aspect at which a straight tube segment as long as it is wide exists at all, i.e.
        # where a bump becomes a tube. protrusion(aspect 2) = 1.3784, FLOORED to 1.37 -- floored,
        # not rounded, because 1.38 sits above 1.3784 and would reject a shape of exactly the
        # aspect ratio the bar was derived to admit. The acceptance test caught that when the bar
        # was first written as 1.38; any rounding of a >= bar must go toward the calibration shape.
        #
        # The old numbers were not merely strict. For a long cylinder r ~ |z| with z uniform, so
        # the metric tends to 0.95/0.50 = 1.90: NO CAPSULE OF ANY ASPECT RATIO CAN REACH 2.0. The
        # only shapes that cleared 2.0 and 3.0 were thin spikes on a body, which over-read the
        # ceiling -- the runs that scored 5.2 and 8.3 and that every analyst called "spike". So
        # the old criterion did not ask for a tube, it asked for the artefact.
        #
        # One derived number at all three times, not a peak bar above the final bar: a peak bar
        # above the final bar pays for overshoot-then-collapse, which is the FORCED phenotype the
        # campaign exists to reject. Forced-vs-grown is discriminated by Q, not by an inflated
        # peak. The threshold is NECESSARY, not sufficient (a spike also clears it) -- Q and
        # n_tubes carry the rest. `calibrate_success.py --check` holds this file to the geometry.
        "protr_peak": ">= 1.37",             # it is a tube (aspect >= 2) at some point
        "protr_final": ">= 1.37",            # and is still a tube at the end
        "Q_protr_after_relax": ">= 1.37",    # and STILL a tube with the drive off (absolute, not a ratio)
        "n_tubes": ">= 1",
        "no_extrude": True,                  # achieved WITHOUT the forcing node
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

    # ----------------------------------------------------------------- persistence
    def to_dict(self):
        return {"radius": self.radius,
                "centroids": [[e.tolist(), cid] for e, cid in self.centroids],
                "clusters": {cid: {**{k: v for k, v in c.items() if k != "members"},
                                   "members": sorted(c["members"]),
                                   "best": (None if c["best"] == -np.inf else c["best"])}
                             for cid, c in self.clusters.items()}}

    @classmethod
    def from_dict(cls, d):
        p = cls(radius=d.get("radius", 2.0))
        p.centroids = [(np.asarray(e, dtype=float), cid) for e, cid in d.get("centroids", [])]
        for cid, c in d.get("clusters", {}).items():
            p.clusters[cid] = {**c, "members": set(c.get("members", [])),
                               "best": -np.inf if c.get("best") is None else float(c["best"])}
        return p


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
    # ONLY gate-admissible metrics (instrument_gate.json). retention and the tube_len-derived
    # metrics are excluded by measurement: retention is anti-correlated (tau=-1.00) and
    # ta_aspect_len_over_diam scored 9.30 on a bud.
    pk = float(summary.get("protr_peak", 0.0))
    fin = float(summary.get("protr_final", 0.0))
    q = summary.get("Q_protr_after_relax")
    q = float(q) if q is not None else fin      # absolute survival; falls back to the final value
    ntb = float(summary.get("ta_n_tubes_final", 0.0))
    # elongation that PERSISTS, mildly rewarded for being a single coherent tube
    return 0.5 * pk + 1.0 * q + 0.25 * min(ntb, 2.0)


def _bar(cfg: CampaignConfig, key):
    """The numeric bar for `key`, READ FROM cfg.success rather than repeated here.

    The defect this prevents: this function used to hardcode 3.0 / 2.0 / 2.0 while
    `cfg.success` carried its own copy of the same three numbers. Two copies of a threshold is
    one copy too many -- editing the config would have moved the printed criterion and left the
    enforced one untouched, so the campaign could have reported a criterion it was not applying.
    The derived values (see CampaignConfig.success) now live in exactly one place.
    """
    return float(str(cfg.success[key]).split(">=")[1])


def meets_success(summary, cfg: CampaignConfig, has_extrude: bool):
    """Is this run a tube that survived? Every bar is the DERIVED geometric threshold; see
    CampaignConfig.success and `calibrate_success.py`. Nothing here is a tuned constant."""
    ok = (float(summary.get("protr_peak", 0)) >= _bar(cfg, "protr_peak")
          and float(summary.get("protr_final", 0)) >= _bar(cfg, "protr_final")
          and float(summary.get("Q_protr_after_relax",
                                summary.get("protr_final", 0))) >= _bar(cfg, "Q_protr_after_relax")
          and float(summary.get("ta_n_tubes_final", 0)) >= _bar(cfg, "n_tubes"))
    if cfg.success.get("no_extrude") and has_extrude:
        ok = False
    return ok


# ============================================================================ supervisor
class Supervisor:
    """Holds the objective, the budget and the terminal-state decision. Never does the work.

    THE SUPERVISOR MUST SURVIVE A RESTART.
    ---------------------------------------------------------------------------------------
    Every field below used to be initialised fresh in `__init__` and never read back from disk.
    `supervisor.jsonl` was append-only OUTPUT. In one long-lived process that is invisible; the
    campaign is specified to run for WEEKS, across crashes, cluster maintenance and session
    boundaries, and each restart silently reset every control mechanism that keeps the loop from
    becoming the very rabbit hole it was built to escape:

      self.round -> 0   every round is "round 1": config namespaces collide, the per-round purge
                        deletes the previous round's configs, and hypothesis ids collide. This
                        had ALREADY happened -- `hypotheses.jsonl` contained five duplicated
                        hids (R1.0.5e3159 twice, ...) before anyone looked.
      self.spent -> 0   `terminal()` compares spent against budget_runs, so THE CAMPAIGN COULD
                        NEVER TERMINATE ON BUDGET.
      self.dry   -> 0   dry rounds never accumulate, so `dry_rounds_to_escalate` NEVER FIRES --
                        the escalation path was dead at the trigger, not merely unbuilt at the
                        consumer.
      self.prox  -> {}  proximity clustering starts blank, so no cluster is ever frozen and
                        budget is never reallocated. That is the Co-Scientist mechanism which is
                        the entire antidote to "thirty rounds, four ideas".
      self.best  -> -inf no best-score history, so "no improvement" cannot be detected.

    So: state is checkpointed to `state.json` after every round and every escalation, and loaded
    here. `supervisor.jsonl` stays as the append-only audit trail -- the two have different jobs
    and neither replaces the other.
    """

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
        self.state_path = os.path.join(root, "state.json")
        self._load()
        cfgp = os.path.join(root, "campaign.json")
        if not os.path.exists(cfgp):
            json.dump(cfg.to_dict(), open(cfgp, "w"), indent=1)

    # ----------------------------------------------------------------- checkpoint
    def _load(self):
        if not os.path.exists(self.state_path):
            return
        try:
            s = json.load(open(self.state_path))
        except Exception as e:
            # Never silently start a weeks-long campaign from a blank slate because a checkpoint
            # would not parse. Losing the round counter corrupts the config namespace.
            raise RuntimeError(
                f"{self.state_path} exists but could not be read ({type(e).__name__}: {e}). "
                f"Refusing to start from a blank state -- that would restart the round counter "
                f"and overwrite this campaign's configs and hypothesis ids. Repair or move it.")
        self.round = int(s.get("round", 0))
        self.spent = int(s.get("spent", 0))
        self.dry = int(s.get("dry", 0))
        b = s.get("best")
        self.best = -math.inf if b is None else float(b)
        if s.get("prox"):
            self.prox = ProximityIndex.from_dict(s["prox"])
        # stage_gate is advanced by escalate(); it is campaign state, not a launch option
        if s.get("stage_gate") is not None:
            self.cfg.stage_gate = int(s["stage_gate"])
        print(f"[supervisor] resumed: round {self.round}, {self.spent} runs spent, "
              f"{len(self.prox.clusters)} clusters ({len(self.prox.active())} active), "
              f"dry {self.dry}, stage_gate {self.cfg.stage_gate}")

    def _save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"round": self.round, "spent": self.spent, "dry": self.dry,
                       "best": None if self.best == -math.inf else self.best,
                       "stage_gate": self.cfg.stage_gate,
                       "prox": self.prox.to_dict(), "t": time.time()}, f, indent=1)
        os.replace(tmp, self.state_path)        # atomic: a killed write must not truncate state

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
        self._save()             # an opened stage gate must survive the restart that follows it
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
        self._save()             # the round counter, the budget and the clusters must persist
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
        # ALWAYS assign: the guarded form skipped assign() on the very first graph, so no
        # centroid was ever created and EVERY graph fell into one cluster -- silently throttling
        # the batch to the per-cluster cap.
        cid = prox.assign(g)
        if prox.clusters.get(cid, {}).get("frozen"):
            continue
        if per_cluster.get(cid, 0) >= max(2, cfg.batch // 4):
            continue
        edits = g.legal_edits(cfg.stage_gate)
        rng.shuffle(edits)
        # take SEVERAL edits per parent when the frontier is small -- otherwise the batch size is
        # silently capped at |frontier| and early rounds are starved. The per-cluster cap still
        # prevents any one idea monopolising the round.
        cap = max(2, cfg.batch // 4)
        taken = 0
        for e, lbl in edits:
            if taken >= cap or per_cluster.get(cid, 0) >= cap:
                break
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
            taken += 1
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

        # ---------------------------------------------------------------- PERSISTENCE
        # The defect this guards: every field below was reset to zero on each process start, so a
        # weeks-long restartable campaign silently re-ran "round 1" forever, never spent its
        # budget, never accumulated a dry round, and never froze a cluster.
        want = (sup.round, sup.spent, round(sup.best, 6), sup.dry,
                len(sup.prox.clusters), sorted(sup.prox.clusters))
        sup.cfg.stage_gate = 2                    # pretend an escalation opened a gate
        sup._save()

        sup2 = Supervisor(CampaignConfig(), d)    # a FRESH process would build exactly this
        got = (sup2.round, sup2.spent, round(sup2.best, 6), sup2.dry,
               len(sup2.prox.clusters), sorted(sup2.prox.clusters))
        assert got == want, f"state did not survive a restart:\n  want {want}\n  got  {got}"
        assert sup2.cfg.stage_gate == 2, "an opened stage gate must survive a restart"
        for cid, c in sup2.prox.clusters.items():
            assert c["members"] == sup.prox.clusters[cid]["members"], f"{cid} members lost"
            assert c["evals"] == sup.prox.clusters[cid]["evals"], f"{cid} evals lost"
        print(f"[persist] round/spent/best/dry/clusters/stage_gate survive a restart: {got}")

        # a round after the restart must ADVANCE the counter, not repeat it
        rep2 = sup2.observe(results)
        assert rep2["round"] == sup.round + 1, (rep2["round"], sup.round)
        assert rep2["spent"] == sup.spent + len(results)
        print(f"[persist] next round after restart is {rep2['round']} "
              f"(was repeating {sup.round} forever), spent {rep2['spent']}")

        # ------------------------------------------------------- GEOMETRIC SUCCESS THRESHOLD
        # The defect this guards: the success bars were an IMPRESSION (3.0 / 2.0 / 2.0), never
        # checked against a shape of known aspect ratio. They sat ABOVE 1.90, which is the value
        # protrusion tends to for an infinitely long capsule -- so no tube of any aspect ratio
        # could ever have met them. The bars are now read back and re-derived from geometry on
        # every run of this file, so they cannot drift back into being taste.
        import calibrate_success
        print("\n[success] geometric acceptance test (capsules of known aspect ratio):")
        calibrate_success.check()

        # a corrupt checkpoint must REFUSE to start, not silently reset the counter
        open(os.path.join(d, "state.json"), "w").write("{not json")
        try:
            Supervisor(CampaignConfig(), d)
            raise AssertionError("a corrupt state.json must not start from a blank slate")
        except RuntimeError as e:
            print(f"[persist] corrupt checkpoint refused: {str(e)[:88]}...")

        print("\ncontrol law OK")
