"""escalation -- what the loop does when the composition space runs out of ideas.

WHY THIS IS THE MAIN PATH, NOT AN EDGE CASE
------------------------------------------------------------------------------------------------
`Supervisor.escalate()` already existed and could do two things: open the next stage gate, or
record an "operator request". But nothing ever GENERATED a request and nothing ever CONSUMED one,
and `round.py` never even looked at the `ESCALATE:` verdict `terminal()` returns -- so in practice
escalation could only ever bump the stage gate twice and then print
`"exhausted: no further stage, no operator request filed"`.

That was survivable while the objective was "find a composition that makes a tube": you would
either find one or not. It stopped being survivable when the objective became **a causal lever-map
of the mechanism space**. A map is finished when the space is COVERED, and a bounded operator set
covers a bounded space -- so a multi-week campaign spends most of its life at the boundary, asking
for mechanisms the language cannot yet express. Over weeks, escalation is where most of the
campaign's value is produced, not a rare failure branch.

An operator request is therefore a FIRST-CLASS DELIVERABLE, exactly like an impossibility result:

    "the proposer wanted to test X; the type system cannot express X; here is the contract X would
     need, the evidence that motivated it, and the test that would decide it"

That is one entry of atlas growth, one talk bullet, and -- unlike a failed run -- it does not
expire when the parameters change.

THE THREE ESCALATION ACTIONS, in the order they are tried
------------------------------------------------------------------------------------------------
  1. OPEN A STAGE GATE   cheap and reversible: admit the next stage's operators into the legal
                         move set. Bounded -- there are only 3 stages.
  2. FILE AN OPERATOR REQUEST  the language is the limit. Ask what is missing, record it, and
                         report it. Does not unblock THIS round; it is the deliverable.
  3. DECLARE THE REGION EXHAUSTED  every stage open, every cluster frozen, requests already filed
                         for what is missing. Say so plainly rather than looping.

A request is never invented by this module. It is written by the agent that hit the wall, because
only that agent knows what it was trying to express.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

STATUSES = ("open", "drafted", "implemented", "rejected", "superseded")


@dataclass
class OperatorRequest:
    """A mechanism the proposer wanted and the language could not express."""
    rid: str                        # stable id, e.g. "OR003"
    round_id: int
    mechanism: str                  # plain language: the biology being asked for
    why_inexpressible: str          # WHICH type-system limit blocks it -- the load-bearing field
    wanted_for: str                 # the question it would answer / the map cell it would fill
    proposed_contract: dict = field(default_factory=dict)   # set/kind/family/EMIT/params
    acceptance_test: str = ""       # how we would know the new operator works
    evidence: list = field(default_factory=list)            # comp hashes / run ids that motivated it
    status: str = "open"
    note: str = ""
    t: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")
        # The one field that makes this a request rather than a wish. Without a stated limit it
        # is not actionable and it is not evidence of anything about the language.
        if not self.why_inexpressible.strip():
            raise ValueError(
                "an operator request MUST say which limit blocks it. 'I would like a better "
                "growth operator' is a wish; 'no operator can write per-cell target volume from "
                "a vertex-set field because EMIT=velocity carries no cell index' is a request.")

    def to_dict(self):
        return asdict(self)


class Backlog:
    """Append-only store of operator requests. The campaign's contribution to the atlas."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._items = {}
        if os.path.exists(path):
            for line in open(path):
                if line.strip():
                    d = json.loads(line)
                    self._items[d["rid"]] = OperatorRequest(**d)

    def next_rid(self):
        return f"OR{len(self._items) + 1:03d}"

    def file(self, req: OperatorRequest):
        if req.rid in self._items:
            raise ValueError(f"{req.rid} already filed")
        self._items[req.rid] = req
        self._append(req)
        return req

    def set_status(self, rid, status, note=""):
        r = self._items[rid]
        if status not in STATUSES:
            raise ValueError(f"status {status!r} not in {STATUSES}")
        r.status, r.note = status, note or r.note
        self._append(r)
        return r

    def _append(self, r):
        with open(self.path, "a") as f:
            f.write(json.dumps(r.to_dict()) + "\n")

    def all(self):
        return list(self._items.values())

    def open_requests(self):
        return [r for r in self._items.values() if r.status == "open"]

    def duplicate_of(self, mechanism):
        """Crude but sufficient: the same mechanism must not be filed every dry round.

        Without this the loop files one request per escalation forever, and the backlog -- which
        is supposed to BE the deliverable -- becomes noise.
        """
        key = _norm(mechanism)
        for r in self._items.values():
            if _norm(r.mechanism) == key:
                return r
        return None

    # ----------------------------------------------------------------- the deliverable
    def render(self, path):
        rows = sorted(self._items.values(), key=lambda r: (r.status != "open", r.rid))
        lines = ["# Operator backlog", "",
                 "_Mechanisms the search wanted and the language could not express._ Each entry is",
                 "one unit of atlas growth: it survives parameter changes, and it is actionable",
                 "without re-running anything.", "",
                 f"_{len(self.open_requests())} open of {len(rows)} filed._", ""]
        if not rows:
            lines += ["_(none yet -- the composition space has not been exhausted)_", ""]
        for r in rows:
            lines += [f"## {r.rid} — {r.mechanism}",
                      f"- **status**: {r.status}   ·   filed round {r.round_id}",
                      f"- **why the language cannot express it**: {r.why_inexpressible}",
                      f"- **what it would answer**: {r.wanted_for}"]
            if r.proposed_contract:
                c = r.proposed_contract
                sig = "  ".join(f"{k}={v}" for k, v in c.items() if k != "params")
                lines.append(f"- **proposed contract**: `{sig}`")
                if c.get("params"):
                    lines.append(f"  - params: `{json.dumps(c['params'])}`")
            if r.acceptance_test:
                lines.append(f"- **acceptance test**: {r.acceptance_test}")
            if r.evidence:
                lines.append(f"- **motivated by**: {', '.join(map(str, r.evidence))}")
            if r.note:
                lines.append(f"- _{r.note}_")
            lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path


def _norm(s):
    return " ".join(str(s).lower().split())[:120]


# --------------------------------------------------------------------------- the decision
def decide(cfg, sup, backlog, n_legal_edits):
    """What should escalation DO right now? Returns (action, detail).

    Pure and testable: no LLM, no side effects. `round.py` executes the action.
    """
    if cfg.stage_gate < 3:
        return "open_stage_gate", (f"stage {cfg.stage_gate} -> {cfg.stage_gate + 1}: admit the "
                                   f"next stage's operators into the legal move set")
    # THE BACKLOG CHECK MUST COME FIRST. It used to sit BELOW `frozen_all`, and `frozen_all` is
    # the state a spent campaign settles into permanently -- so once the last cluster froze,
    # `decide()` returned `request_operator` on EVERY subsequent round, forever, and `exhausted`
    # was unreachable. That is not escalation, it is a loop filing duplicate wishes against a
    # language nobody is extending. If a request is already open, the answer is "stop and build",
    # whatever the cluster state.
    # AN OPEN REQUEST IS NOT AN EXHAUSTED SPACE. This branch exists to stop `request_operator`
    # firing every round forever -- a real bug, fixed here -- but it created the mirror image:
    # with one request open and NOTHING in the loop able to close it (set_status is called only
    # from this module's own __main__), every future escalation returned "exhausted" regardless
    # of how many moves remained. Measured: 96 legal edits across the frontier, and thirteen
    # consecutive rounds told there was no reachable region.
    if backlog.open_requests() and n_legal_edits <= 0:
        return "exhausted", (f"{len(backlog.open_requests())} operator request(s) already open "
                             f"and no new region is reachable; waiting on the language, not on "
                             f"compute")
    if n_legal_edits == 0:
        return "request_operator", ("every stage is open and the composition has NO legal edit "
                                    "left -- the language is the limit")
    frozen_all = bool(sup.prox.clusters) and not sup.prox.active()
    if frozen_all:
        return "request_operator", ("every stage is open and every proximity cluster is frozen: "
                                    "the reachable space is explored and none of it improved")
    return "request_operator", "dry rounds with every stage open"


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        bl = Backlog(os.path.join(d, "operator_requests.jsonl"))
        assert bl.next_rid() == "OR001"

        # a wish is refused; a request is accepted
        try:
            OperatorRequest(rid="OR000", round_id=1, mechanism="better growth",
                            why_inexpressible="   ", wanted_for="tubes")
            raise AssertionError("a request with no stated limit must be refused")
        except ValueError as e:
            print("refused as designed:", str(e)[:78], "...")

        r = bl.file(OperatorRequest(
            rid=bl.next_rid(), round_id=4,
            mechanism="anisotropic line tension along a morphogen gradient",
            why_inexpressible=("no registered operator EMITs a per-EDGE tension; `shape_energy_3d` "
                               "reads a single scalar Lambda for the whole mesh, and the cell set "
                               "carries no edge-indexed state block to route into it"),
            wanted_for=("whether Okuda's tube needs oriented tension or only oriented growth -- "
                        "the one lever the map cannot currently vary"),
            proposed_contract={"contract": "edge_tension", "set": "vertex", "kind": "lateral",
                               "family": "mechanics", "EMIT": "force",
                               "params": {"lambda_par": "float", "lambda_perp": "float"}},
            acceptance_test=("on a fixed cylinder, lambda_par > lambda_perp must shorten it along "
                             "the axis and leave the radius within 2%"),
            evidence=["C5e315998af4", "r002c_01_cba5fe"]))
        assert bl.duplicate_of("Anisotropic line tension along a morphogen gradient  ") is r
        assert bl.duplicate_of("something else") is None
        print(f"filed {r.rid}, duplicate detection OK")

        bl.set_status(r.rid, "drafted", note="contract sketched in plexus2_discovery §7")
        bl2 = Backlog(bl.path)                      # a fresh process must see the same state
        assert bl2.all()[0].status == "drafted"
        assert len(bl2.open_requests()) == 0
        print("append-only reload OK; status survives")

        # the decision table
        class _C:
            stage_gate = 1
        class _P:
            clusters = {"K000": {"frozen": True}}
            def active(self): return []
        class _S:
            prox = _P()
        cfg, sup = _C(), _S()
        assert decide(cfg, sup, bl2, 12)[0] == "open_stage_gate"
        cfg.stage_gate = 3
        assert decide(cfg, sup, bl2, 0)[0] == "request_operator"
        assert decide(cfg, sup, bl2, 12)[0] == "request_operator"   # all clusters frozen
        bl2.set_status(r.rid, "open")
        # THE REGRESSION THIS ORDERING PREVENTS: an open request AND every cluster frozen is the
        # state a spent campaign settles into permanently. With the backlog check below the freeze
        # check, `decide` returned `request_operator` here on every round forever and `exhausted`
        # was unreachable -- the loop filing duplicate wishes instead of stopping to build.
        assert sup.prox.clusters and not sup.prox.active()            # frozen, as in the live state
        assert decide(cfg, sup, bl2, 12)[0] == "exhausted", \
            "an open request must win over a frozen cluster, or escalation never terminates"
        sup.prox.clusters = {}                                       # nothing frozen, none active
        assert decide(cfg, sup, bl2, 12)[0] == "exhausted"
        print("decision table OK: gate -> request -> exhausted (open request wins over freeze)")

        p = bl2.render(os.path.join(d, "operator_backlog.md"))
        body = open(p).read()
        assert "OR001" in body and "edge_tension" in body and "acceptance test" in body
        print("\n--- operator_backlog.md ---")
        print(body[:700])
        print("escalation OK")
