"""composition_space -- the space the Okuda campaign searches: TYPED COMPOSITION GRAPHS.

A candidate mechanism is a GRAPH, not a Boolean feature vector:
  - operator NODES, each pinned to one IMPLEMENTATION,
  - typed CONNECTIONS from an output port to an input slot.

Routing the morphogen to growth's `gate`, to division's `axis`, or to the extrusion `site` gives
three DIFFERENT MECHANISMS, not three parameter settings.

--------------------------------------------------------------------------------------------
IDENTITY RULE (the line that keeps the campaign honest)
--------------------------------------------------------------------------------------------
    structure = {which operators, which IMPLEMENTATIONS, which typed connections}
    theta     = the numbers
    comp_hash(structure)  -- theta is EXCLUDED

So a change of numbers PROVABLY cannot register as a new hypothesis. That is the structural fix
for what happened over rounds 01-30 of the hand-run campaign.

*Why implementations are part of identity here.*  plexus2 says implementations of one contract
"differ only in numerics".  That holds for e.g. finite-difference vs spectral diffusion.  It does
NOT hold for the three reaction kinetics: the record shows Brusselator decays or reorganises a
seed, Gray-Scott holds it, and Gierer-Meinhardt amplifies it into a stable gradient peak -- three
qualitatively different phenomenologies.  Same for shape_energy_3d default (mid-surface wedge
volume) vs monolayer (true A*h volume, emergent bending).  Since the campaign must be able to ASK
"which kinetics", implementation is part of composition identity and is flagged
`impl_structural=True` on those operators.  This is a deliberate, recorded departure.

--------------------------------------------------------------------------------------------
EXTRUSION IS A NODE, NOT A PARAMETER
--------------------------------------------------------------------------------------------
The campaign's central question is round 41's finding: our tube is held open by an explicit
outward force, where Okuda's is a growth-driven quasi-static equilibrium.  If the forcing lives
in a config file it can only be turned down.  As a graph node, the standard necessity protocol
ABLATES IT AUTOMATICALLY on every composition -- so the central question becomes a routine
measurement rather than a special experiment.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np

# ============================================================================ port types
#   morphogen  -- a per-cell chemical amount/concentration (the RD field)
#   adjacency  -- the cell-cell neighbour relation read off the half-edge mesh
#   geometry   -- per-cell centroid / area / volume, aggregated from the vertex set
PORT_TYPES = ("morphogen", "adjacency", "geometry")

# ============================================================================ clock re-anchoring
#
# ⚠ THE PER-CALL / PER-FRAME TRAP  (raised 2026-07-30 after FINDING 8)
#
# `divide_3d` counts `min_cycle` / `max_cycle` in DIVISION-CALLS (its own docstring: "a cell may
# not divide before this many division-calls since birth"; `age` is "per-cell age in
# division-calls"), and `max_div_frac` is a PER-CALL throttle. The archived configs passed
# `every: 2`, which the ENGINE gated and the operator's private `self._k` ALSO gated -- product 4.
# So in the archived runs `divide_3d.forward` executed once every FOUR frames:
#
#     min_cycle = 8 calls      ==  32 frames        max_div_frac = 0.03/call  ==  0.0075/frame
#
# With the clock fixed (`every: 1`, engine owns it) the same numbers mean:
#
#     min_cycle = 8 calls      ==   8 frames        max_div_frac = 0.03/call  ==  0.03  /frame
#
# i.e. FOUR TIMES the proliferation. Leaving the hand-tuned values as vocabulary defaults would
# silently start every generated composition at a theta tuned for the wrong clock -- which is
# exactly what FINDING 8 measured (aspect 7.5 -> 3.2, cells 2700 -> 3335).
#
# Because the factor is exact and only `divide_3d` carried `every: 2`, we do NOT need a re-tuning
# sweep: the archived working point is recoverable analytically by rescaling the per-call
# quantities. That makes the D1 fix BEHAVIOUR-PRESERVING BY CONSTRUCTION, so any later change in
# phenotype is attributable to the change that caused it rather than to the clock.
DIVIDE_CALL_PERIOD_BEFORE_D1 = 4        # engine every=2  x  private self.every=2
CLOCK_COUPLED = {                       # param -> how to convert an archived value to per-frame
    "min_cycle":     "multiply by 4  (calls -> frames)",
    "max_cycle":     "multiply by 4  (calls -> frames)",
    "max_div_frac":  "divide by 4    (per-call -> per-frame)",
    "max_div":       "divide by 4    (per-call FLOOR; cap_div = max(max_div, frac*nF), so this "
                     "DOMINATES at realistic nF -- rescaling frac alone is entirely masked)",
}
# NOT clock-coupled (evaluated every frame either way): p0, Gamma, Lambda, h0, relax_iters, l_th.
# PARTIALLY coupled and therefore still provisional: K_V was raised to 6.0 specifically to crush
# the cell-size CV produced by the division wave; vcap is a per-call bypass. Both are flagged in
# validate_space (V10) and must be confirmed against the re-anchored baseline.
# vcap: CLAIM RETRACTED 2026-07-30 (Metrologist RET001). I asserted it was "not rate-coupled --
#   the same cells divide, only sooner". That is true of ONE cell and false of the POPULATION:
#   vcap divisions bypass the throttle entirely (`cap_div = max(cap_div, len(over))`), so
#   checking 4x more often makes oversized cells divide sooner, their daughters begin growing
#   sooner, and the total division count over a run differs. It is weakly rate-coupled and the
#   correct rescaling is NOT yet established.
#
#   CORRECTION (same day): the evidence I cited for this was a FALSE DISCREPANCY OF MY OWN
#   MAKING. I compared our r95/median metric -- which I had misnamed `aspect` -- against the
#   report's "aspect ~7.5" for round_40_mc8, which is tube_len/tube_diam. Two different
#   quantities. tube_analysis.py:89 calls r95/median `protr`, and ours is now named `protr` too.
#   The re-anchored replay recovered the archived CELL COUNT (2927 vs ~2700); whether it recovers
#   the archived tube_len/tube_diam is being measured with the archive's OWN metric bank.
#   vcap's rate-coupling therefore remains UNTESTED -- neither proven nor disproven. Which is
#   still the honest state, but for a different reason than I first recorded.
#   cycle_cv NOT clock-coupled. A dimensionless Gaussian CV on the per-cell threshold multiplier.
#   K_V      Its MEANING was never clock-coupled (a per-frame mechanical stiffness). Its
#            OPTIMALITY was stale only because it was tuned against the division wave -- and the
#            re-anchoring restores exactly that wave, so K_V = 6.0 is valid again.
PROVISIONAL_THETA = ("vcap",)   # see the retraction above -- do not treat as settled

# ============================================================================ vocabulary
# stage           -- the gate; the search opens stages in order
# role            -- for post-hoc naming and proximity clustering
# outputs         -- port types this operator produces
# slots           -- input ports another operator's output may connect to
# impls           -- available implementations; impls[0] is the default
# impl_structural -- True => the implementation choice changes the phenomenology, so it is part
#                    of composition identity (see module docstring)
# needs           -- PRECONDITIONS (defect D4): port types that must be produced by SOME node in
#                    the graph, else this operator silently no-ops. The Critic rejects for free.
# params          -- (lo, hi, default); theta only, never part of identity
OPERATORS = {
    # ---------------------------------------------------------------- Stage 1: substrate
    "seed_mesh_3d": dict(
        stage=1, role="substrate", outputs=[], slots=[], needs=[],
        impls=["fibonacci_sphere", "checkpoint"], impl_structural=False,
        params={"n_cells": (150, 2000, 500), "vseed_cv": (0.0, 0.5, 0.15)}),
    "shape_energy_3d": dict(
        stage=1, role="mechanics", outputs=["geometry"], slots=[], needs=[],
        impls=["default", "monolayer"], impl_structural=True,       # mid-surface vs true 3D volume
        params={"K_V": (1.0, 8.0, 6.0), "kappa_s": (0.05, 0.6, 0.2),
                "Gamma": (0.0, 0.4, 0.05), "Lambda": (0.0, 0.3, 0.20),
                "p0": (3.4, 4.2, 3.90), "h0": (0.05, 0.4, 0.40), "mono_gamma": (0.0, 0.3, 0.06),
                "relax_iters": (10, 90, 30)}),
    "reconnect_t1_3d": dict(
        stage=1, role="topology", outputs=[], slots=[], needs=[],
        impls=["length_threshold"], impl_structural=False,
        params={"l_th": (0.01, 0.12, 0.04)}),

    # ---------------------------------------------------------------- Stage 2: growth & topology
    "vesicle_growth": dict(                                  # uniform, body-wide inflation
        stage=2, role="growth", outputs=[], slots=[], needs=[],
        impls=["uniform_ramp"], impl_structural=False,
        params={"rate": (0.002, 0.03, 0.006)}),
    "morphogen_growth_3d": dict(                             # LOCAL growth, gated by the activator
        stage=2, role="growth", outputs=[], slots=["gate"], needs=["morphogen"],
        impls=["hill_conserve_amount", "hill_no_conserve"], impl_structural=True,
        params={"rate": (0.002, 0.03, 0.010), "a_sw": (0.2, 6.0, 1.5),
                "alpha": (1.0, 8.0, 4.0), "rho": (0.0, 1.0, 0.0)}),
    "divide_3d": dict(
        # `hertwig` splits normal to the cell's OWN longest axis -> needs no morphogen input.
        # `orient_iface` stacks daughters along the bud axis -> needs the activator routed in.
        # Slots are therefore PER IMPLEMENTATION; declaring `axis` unconditionally would make
        # every hertwig composition look like it had a dangling (inert) slot.
        stage=2, role="topology", outputs=[], slots=[], impl_slots={"orient_iface": ["axis"]},
        needs=[],
        impls=["hertwig", "orient_iface"], impl_structural=True,   # long-axis vs bud-axis septum
        params={"cycle_cv": (0.05, 0.5, 0.40), "min_cycle": (2, 64, 16),   # 4 calls x 4
                "max_cycle": (6, 10**9, 10**9), "vcap": (0.0, 3.0, 1.5),   # vcap: PROVISIONAL
                "max_div_frac": (0.00125, 0.20, 0.0075),   # 0.03/call / 4 = per-frame
                "max_div": (4, 480, 30),                   # 120/call / 4 = per-frame FLOOR
                "orient_asw": (0.2, 6.0, 1.0)}),
    "extrude": dict(                                          # THE FORCING TERM -- ablatable
        stage=2, role="forcing", outputs=[], slots=["site"], needs=["morphogen"],
        impls=["radial_push"], impl_structural=False,
        params={"K_extrude": (0.0, 14.0, 4.0), "a_sw": (0.2, 6.0, 0.5)}),

    # ---------------------------------------------------------------- Stage 3: patterning
    "cell_geometry_3d": dict(
        stage=3, role="readout", outputs=["geometry"], slots=[], needs=[],
        impls=["scatter_add"], impl_structural=False, params={}),
    "cell_adjacency": dict(
        stage=3, role="readout", outputs=["adjacency"], slots=[], needs=[],
        impls=["shared_edge"], impl_structural=False, params={}),
    "cell_diffuse": dict(
        stage=3, role="patterning", outputs=[], slots=[], needs=["adjacency"],
        impls=["graph_laplacian"], impl_structural=False,
        params={"d_a": (0.005, 0.2, 0.02), "d_h": (0.1, 2.0, 0.7),
                "chi": (1.0, 10.0, 4.0)}),
    "cell_react": dict(
        stage=3, role="patterning", outputs=["morphogen"], slots=[], needs=["adjacency"],
        impls=["gierer_meinhardt", "gray_scott", "brusselator"], impl_structural=True,
        params={"gamma": (0.1, 100.0, 0.3), "a0": (0.0, 0.05, 0.01),
                "rd_rate": (0.2, 3.0, 1.0), "F": (0.02, 0.06, 0.055), "kk": (0.05, 0.07, 0.062),
                "mu_h": (0.2, 2.0, 1.0)}),
    "cell_rd_seed": dict(                                     # the prescribed activation driver
        stage=3, role="driver", outputs=["morphogen"], slots=[], needs=[],
        impls=["tip", "cone", "spot"], impl_structural=True,
        params={"tip_radius": (0.6, 3.0, 2.0), "cone_deg": (4.0, 30.0, 8.0),
                "amp": (0.5, 5.0, 2.0), "n_spots": (1, 8, 1)}),
    # NOTE: there is deliberately no separate `rd_interface_tension` node. In the engine that op
    # carries BOTH K_purse and K_extrude; the mechanism we need to ablate is the outward forcing,
    # so it is exposed once, as `extrude`. A second node would be the same engine operator under
    # two names -- which would let one mechanism occupy two points of the search space.
}

# Slots may depend on the chosen implementation (see divide_3d).
def slots_of(op: str, impl: str):
    spec = OPERATORS[op]
    return list(spec.get("impl_slots", {}).get(impl, spec["slots"]))

STAGES = {s: [k for k, v in OPERATORS.items() if v["stage"] == s] for s in (1, 2, 3)}

# which (output port type -> input slot) connections are TYPE-LEGAL
LEGAL_LINKS = {
    ("morphogen", "gate"),    # activator drives local growth      -- the Okuda coupling
    ("morphogen", "axis"),    # activator orients the division plane
    ("morphogen", "site"),    # activator selects where to push     -- the forcing route
    ("morphogen", "field"),   # activator raises interface tension
}

# operators that must be present for the run to be meaningful at all
REQUIRED_ROLES = {"substrate", "mechanics"}


# ============================================================================ the graph
class CompositionGraph:
    def __init__(self, ops=None, conns=None, params=None):
        # ops:   [{"id": str, "op": name, "impl": str}]
        # conns: [{"src": id, "dst": id, "slot": str}]
        self.ops = [dict(o) for o in (ops or [])]
        self.conns = [dict(c) for c in (conns or [])]
        self.params = dict(params or {})

    # ---------------------------------------------------------------- identity
    def structure(self):
        """Operators (+ structural implementations) and typed connections. NO theta."""
        ops = sorted(
            ({"id": o["id"], "op": o["op"],
              **({"impl": o.get("impl", OPERATORS[o["op"]]["impls"][0])}
                 if OPERATORS[o["op"]]["impl_structural"] else {})}
             for o in self.ops),
            key=lambda o: (o["op"], o.get("impl", ""), o["id"]))
        conns = sorted(
            ({"src_op": self._op_of(c["src"]), "dst_op": self._op_of(c["dst"]), "slot": c["slot"]}
             for c in self.conns),
            key=lambda c: (str(c["src_op"]), str(c["dst_op"]), c["slot"]))
        return {"operators": ops, "connections": conns}

    def _op_of(self, node_id):
        return next((o["op"] for o in self.ops if o["id"] == node_id), None)

    def _node(self, node_id):
        return next((o for o in self.ops if o["id"] == node_id), None)

    def op_names(self):
        return [o["op"] for o in self.ops]

    def impl_of(self, node):
        return node.get("impl", OPERATORS[node["op"]]["impls"][0])

    def roles(self):
        return {OPERATORS[o["op"]]["role"] for o in self.ops}

    def produced_ports(self):
        out = set()
        for o in self.ops:
            out.update(OPERATORS[o["op"]]["outputs"])
        return out

    def copy(self):
        g = CompositionGraph(self.ops, self.conns, self.params)
        return g

    # ---------------------------------------------------------------- theta
    def default_params(self):
        p = {}
        for o in self.ops:
            for pn, (lo, hi, d) in OPERATORS[o["op"]]["params"].items():
                p[f"{o['id']}.{pn}"] = d
        return p

    def sample_params(self, rng, scale=0.15):
        """Perturb around the defaults -- the PARAMETER BASIN a robustness claim is made over."""
        p = self.default_params()
        for o in self.ops:
            for pn, (lo, hi, d) in OPERATORS[o["op"]]["params"].items():
                k = f"{o['id']}.{pn}"
                p[k] = float(np.clip(d + rng.normal(0, scale * (hi - lo)), lo, hi))
        return p

    def with_params(self, params):
        g = self.copy()
        g.params = dict(params)
        return g

    # ---------------------------------------------------------------- D4 preconditions
    def unmet_preconditions(self):
        """[(node_id, op, missing_port)] -- operators that would silently no-op.

        The Critic rejects these for FREE, before any cluster time is spent. This is the guard
        against recording 'this mechanism cannot make tubes' when the mechanism never ran.
        """
        have = self.produced_ports()
        bad = []
        for o in self.ops:
            for need in OPERATORS[o["op"]]["needs"]:
                if need not in have:
                    bad.append((o["id"], o["op"], need))
        return bad

    def unrouted_slots(self):
        """Operators with a slot that nothing feeds -- present but disconnected, so inert."""
        fed = {(c["dst"], c["slot"]) for c in self.conns}
        out = []
        for o in self.ops:
            for slot in slots_of(o["op"], self.impl_of(o)):
                if (o["id"], slot) not in fed:
                    out.append((o["id"], o["op"], slot))
        return out

    def is_runnable(self):
        """(ok, reason). A graph must have a substrate + mechanics, no unmet precondition, and
        no dangling slot."""
        if not REQUIRED_ROLES.issubset(self.roles()):
            return False, f"missing required role(s): {sorted(REQUIRED_ROLES - self.roles())}"
        if self.unmet_preconditions():
            return False, f"unmet precondition: {self.unmet_preconditions()}"
        if self.unrouted_slots():
            return False, f"dangling slot: {self.unrouted_slots()}"
        return True, "ok"

    # ---------------------------------------------------------------- one-edit API
    def _new_id(self, op):
        n = sum(1 for o in self.ops if o["op"] == op)
        return f"{op}{n}"

    def legal_edits(self, max_stage=3):
        """Every legal ONE-edit move from here, gated to <= max_stage. [(edit, label)]."""
        edits = []
        present = self.op_names()
        for stage in range(1, max_stage + 1):
            for op in STAGES[stage]:
                spec = OPERATORS[op]
                if op in present and not spec["impl_structural"]:
                    continue                                   # one copy is enough
                for impl in spec["impls"]:
                    if any(o["op"] == op and self.impl_of(o) == impl for o in self.ops):
                        continue
                    if op in present and not spec["impl_structural"]:
                        continue
                    edits.append((("add_op", op, impl), f"+{op}:{impl}"))
        for o in self.ops:                                     # removals
            role = OPERATORS[o["op"]]["role"]
            same_role = sum(1 for x in self.ops if OPERATORS[x["op"]]["role"] == role)
            if role in REQUIRED_ROLES and same_role == 1:
                continue                                       # never remove the last substrate
            edits.append((("remove_op", o["id"]), f"-{o['op']}"))
        for src, dst, slot in self._candidate_links():
            edits.append((("connect", src, dst, slot),
                          f"~{self._op_of(src)}->{self._op_of(dst)}.{slot}"))
        for c in self.conns:
            edits.append((("disconnect", c["src"], c["dst"], c["slot"]),
                          f"x{self._op_of(c['src'])}->{self._op_of(c['dst'])}.{c['slot']}"))
        # implementation swaps on structural operators are genuine mechanism edits
        for o in self.ops:
            spec = OPERATORS[o["op"]]
            if not spec["impl_structural"]:
                continue
            for impl in spec["impls"]:
                if impl != self.impl_of(o):
                    edits.append((("set_impl", o["id"], impl), f"={o['op']}:{impl}"))
        return edits

    def _candidate_links(self):
        out = []
        for s in self.ops:
            for otype in OPERATORS[s["op"]]["outputs"]:
                for d in self.ops:
                    if d["id"] == s["id"]:
                        continue
                    for slot in slots_of(d["op"], self.impl_of(d)):
                        if (otype, slot) not in LEGAL_LINKS:
                            continue
                        if any(c["src"] == s["id"] and c["dst"] == d["id"] and c["slot"] == slot
                               for c in self.conns):
                            continue
                        out.append((s["id"], d["id"], slot))
        return out

    def apply(self, edit):
        """Return (new_graph, edit) after ONE legal move."""
        g = self.copy()
        kind = edit[0]
        if kind == "add_op":
            op, impl = edit[1], edit[2]
            nid = self._new_id(op)
            g.ops.append({"id": nid, "op": op, "impl": impl})
            for pn, (lo, hi, d) in OPERATORS[op]["params"].items():
                g.params[f"{nid}.{pn}"] = d
        elif kind == "remove_op":
            nid = edit[1]
            g.ops = [o for o in g.ops if o["id"] != nid]
            g.conns = [c for c in g.conns if c["src"] != nid and c["dst"] != nid]
            g.params = {k: v for k, v in g.params.items() if not k.startswith(nid + ".")}
        elif kind == "connect":
            g.conns.append({"src": edit[1], "dst": edit[2], "slot": edit[3]})
        elif kind == "disconnect":
            g.conns = [c for c in g.conns if not (c["src"] == edit[1] and c["dst"] == edit[2]
                                                  and c["slot"] == edit[3])]
        elif kind == "set_impl":
            for o in g.ops:
                if o["id"] == edit[1]:
                    o["impl"] = edit[2]
            # an implementation with fewer slots cannot keep connections into the ones it lost
            keep = set(slots_of(g._op_of(edit[1]), edit[2]))
            g.conns = [c for c in g.conns
                       if c["dst"] != edit[1] or c["slot"] in keep]
        else:
            raise ValueError(f"unknown edit {edit!r}")
        return g, edit

    # ---------------------------------------------------------------- proximity encoding
    def encode(self):
        """Fixed-length structural feature vector, for clustering near-duplicates.

        Near-duplicate proliferation is pathology #3: thirty rounds explored perhaps four
        genuinely distinct ideas. Members of one cluster compete WITHIN the cluster so a family
        of near-identical ideas exhausts one budget, not twenty.
        """
        feat = []
        for op in OPERATORS:                                    # presence, per implementation
            spec = OPERATORS[op]
            if spec["impl_structural"]:
                for impl in spec["impls"]:
                    feat.append(float(any(o["op"] == op and self.impl_of(o) == impl
                                          for o in self.ops)))
            else:
                feat.append(float(any(o["op"] == op for o in self.ops)))
        for (otype, slot) in sorted(LEGAL_LINKS):               # routing flags
            feat.append(float(any(
                c["slot"] == slot and otype in OPERATORS[self._op_of(c["src"])]["outputs"]
                for c in self.conns)))
        return np.array(feat, np.float32)

    def distance(self, other):
        return float(np.abs(self.encode() - other.encode()).sum())

    # ---------------------------------------------------------------- post-hoc naming
    def name_region(self):
        """Label a DISCOVERED composition against the literature, never chosen a priori."""
        ops = set(self.op_names())
        impls = {o["op"]: self.impl_of(o) for o in self.ops}
        forced = "extrude" in ops
        local = "morphogen_growth_3d" in ops
        emergent = "cell_react" in ops
        driven = "cell_rd_seed" in ops
        mono = impls.get("shape_energy_3d") == "monolayer"

        if not (local or "vesicle_growth" in ops):
            return "mechanics-only (no growth)"
        if forced and driven and not emergent:
            return "driven + forced (round-33 recipe)"
        if forced and emergent:
            return "emergent RD + forced extrusion"
        if local and mono and not forced:
            return "growth-driven monolayer (Okuda route)"
        if local and not forced and emergent:
            return "growth-driven emergent (target mechanism)"
        if local and not forced:
            return "growth-driven mid-surface"
        if "vesicle_growth" in ops and not local:
            return "uniform inflation (no patterning)"
        return "unnamed"


# ============================================================================ seeds
def seed(kind="substrate"):
    """The seed the campaign grows from: a relaxing vesicle that does nothing."""
    if kind == "substrate":
        return CompositionGraph(ops=[
            {"id": "seed_mesh_3d0", "op": "seed_mesh_3d", "impl": "fibonacci_sphere"},
            {"id": "shape_energy_3d0", "op": "shape_energy_3d", "impl": "default"},
            {"id": "reconnect_t1_3d0", "op": "reconnect_t1_3d", "impl": "length_threshold"},
        ])
    if kind == "empty":
        return CompositionGraph()
    raise ValueError(kind)


def reference_recipes():
    """The two compositions the campaign must reproduce and discriminate.

    `round40_mc8` is our best hand-found tube -- DRIVEN activation + FORCED extrusion.
    `okuda_route` is the target: local growth on a monolayer, no forcing term at all.
    Both are constructed by legal edits from the seed, so they live in the searched space.
    """
    out = {}

    g = seed("substrate")
    for op, impl in [("cell_geometry_3d", "scatter_add"), ("cell_adjacency", "shared_edge"),
                     ("cell_rd_seed", "tip"), ("morphogen_growth_3d", "hill_conserve_amount"),
                     ("divide_3d", "orient_iface"), ("extrude", "radial_push")]:
        g, _ = g.apply(("add_op", op, impl))
    src = next(o["id"] for o in g.ops if o["op"] == "cell_rd_seed")
    g, _ = g.apply(("connect", src, next(o["id"] for o in g.ops
                                         if o["op"] == "morphogen_growth_3d"), "gate"))
    g, _ = g.apply(("connect", src, next(o["id"] for o in g.ops if o["op"] == "extrude"), "site"))
    g, _ = g.apply(("connect", src, next(o["id"] for o in g.ops if o["op"] == "divide_3d"), "axis"))
    out["round40_mc8"] = g

    h = seed("substrate")
    h, _ = h.apply(("set_impl", "shape_energy_3d0", "monolayer"))
    for op, impl in [("cell_geometry_3d", "scatter_add"), ("cell_adjacency", "shared_edge"),
                     ("cell_react", "gierer_meinhardt"), ("cell_diffuse", "graph_laplacian"),
                     ("morphogen_growth_3d", "hill_conserve_amount"), ("divide_3d", "hertwig")]:
        h, _ = h.apply(("add_op", op, impl))
    rsrc = next(o["id"] for o in h.ops if o["op"] == "cell_react")
    h, _ = h.apply(("connect", rsrc, next(o["id"] for o in h.ops
                                          if o["op"] == "morphogen_growth_3d"), "gate"))
    h, _ = h.apply(("connect", rsrc, next(o["id"] for o in h.ops if o["op"] == "divide_3d"), "axis"))
    out["okuda_route"] = h

    # the degenerate control the search must visit on its way: uniform inflation, no patterning.
    # It is the "grows but cannot subdivide" corner -- where impossibility results come from.
    u = seed("substrate")
    for op, impl in [("vesicle_growth", "uniform_ramp"), ("divide_3d", "hertwig")]:
        u, _ = u.apply(("add_op", op, impl))
    out["uniform_inflation"] = u
    return out


# ============================================================================ smoke test
if __name__ == "__main__":
    from run_record import comp_hash

    g = seed("substrate")
    print(f"seed: {'+'.join(g.op_names())}\n  hash={comp_hash(g)}  region={g.name_region()!r}")
    ok, why = g.is_runnable()
    print(f"  runnable={ok} ({why})")

    # --- the identity rule: theta must NOT change identity ---------------------------------
    rng = np.random.default_rng(0)
    assert comp_hash(g.with_params(g.sample_params(rng))) == comp_hash(g), \
        "theta must not change composition identity"
    print("\n[OK] theta does not change identity -- a retune cannot pose as a new hypothesis")

    # --- an implementation swap IS a mechanism edit ------------------------------------------
    g_mono, _ = g.apply(("set_impl", "shape_energy_3d0", "monolayer"))
    assert comp_hash(g_mono) != comp_hash(g)
    print("[OK] shape_energy_3d default->monolayer changes identity (mid-surface vs true 3D volume)")

    # --- D4: preconditions are caught for FREE, before any cluster time ----------------------
    bad, _ = g.apply(("add_op", "cell_diffuse", "graph_laplacian"))
    print(f"\n[D4] cell_diffuse without cell_adjacency -> unmet={bad.unmet_preconditions()}")
    assert not bad.is_runnable()[0]
    fixed, _ = bad.apply(("add_op", "cell_adjacency", "shared_edge"))
    assert fixed.is_runnable()[0]
    print("[D4] + cell_adjacency -> runnable. This is the false-impossibility guard.")

    # --- the two reference recipes ------------------------------------------------------------
    print("\nreference recipes (both reachable by legal edits from the seed):")
    refs = reference_recipes()
    for name, r in refs.items():
        ok, why = r.is_runnable()
        print(f"  {name:14} {comp_hash(r)}  {r.name_region():34} runnable={ok}")
        if not ok:
            print(f"      -> {why}")
    d = refs["round40_mc8"].distance(refs["okuda_route"])
    print(f"  structural distance between them = {d:.0f}  (proximity clustering keeps them apart)")

    # --- the campaign's central ablation is a legal one-edit move ----------------------------
    r40 = refs["round40_mc8"]
    ex = next(o["id"] for o in r40.ops if o["op"] == "extrude")
    ablated, _ = r40.apply(("remove_op", ex))
    print(f"\n[central test] ablate `extrude` -> {ablated.name_region()!r}  "
          f"hash {comp_hash(r40)} -> {comp_hash(ablated)}")
    print("  Round 41 by hand; here it is one automatic necessity test on every composition.")

    n_edits = len(g.legal_edits(3))
    print(f"\nlegal one-edit moves from the seed (stage<=3): {n_edits}")
    print("composition_space OK")
