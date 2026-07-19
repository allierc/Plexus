"""composition_space -- the space Loop I explores: TYPED COMPOSITION GRAPHS of operators, and a
one-edit mutation API. Substrate-agnostic (this file names operators and their typed ports; a backend
-- pf/ phase field today, vertex/MPM tomorrow -- interprets the graph).

A composition is NOT a Boolean feature vector. It is a graph:
  - operator NODES  (an operator may appear more than once -> unique node ids),
  - typed CONNECTIONS between output ports and input ports (e.g. a reaction-diffusion morphogen can be
    routed to cleft placement OR to gate growth OR to drive chemotaxis -- different graphs, different
    mechanisms).
Parameter VALUES (theta) are held separately and are NOT part of composition identity (comp_hash);
retuning theta is a different run of the SAME composition, handled by Loop II.

Loop I edits a composition by ONE legal move at a time (causality): add_op / remove_op / connect /
disconnect. Moves are STAGE-GATED so each stage exposes only a few legal edits (interpretable, not a
2^N enumeration). Named mechanisms are LABELLED post-hoc (name_region), never chosen a priori.
"""
from __future__ import annotations
import copy
import numpy as np

# ------------------------------------------------------------------ operator vocabulary (typed ports)
# ports carry a field TYPE: tissue (phi), morphogen (u,v RD field), cleft (F source), orientation.
# `slots` = the input ports an operator exposes that another operator's output can connect to.
OPERATORS = {
    # --- substrate (Stage 1) ---
    "interface_relax": dict(stage=1, role="substrate", outputs=[], slots=[],
        params={"kappa": (0.5, 2.4, 1.3), "w0": (0.8, 1.2, 1.0)}),                 # defaults = validated regime
    "tissue_grow": dict(stage=1, role="substrate", outputs=[], slots=["gate"],   # gate <- morphogen
        params={"growth_frac": (1.05, 2.2, 1.35), "beta": (2.0, 7.0, 4.0)}),
    # --- cleft mechanisms (Stage 2) ---
    "cleft_induce": dict(stage=2, role="cleft", outputs=["cleft"], slots=["source"],  # source <- morphogen
        params={"s": (0.5, 1.8, 1.0), "lam": (0.6, 1.6, 1.0), "kappa_gate": (0.03, 0.07, 0.045),
                "thick_gate": (0.45, 0.70, 0.60)}),
    "confine": dict(stage=2, role="boundary", outputs=[], slots=[],
        params={"conf_strength": (0.1, 0.8, 0.4), "conf_aspect": (1.0, 2.5, 1.6)}),
    # --- patterning / directed / anisotropy (Stage 3) ---
    "react_rd": dict(stage=3, role="morphogen", outputs=["morphogen"], slots=[],
        params={"feed": (0.028, 0.045, 0.030), "kill": (0.058, 0.066, 0.062), "Dv": (0.06, 0.10, 0.08),
                "v_thr": (0.10, 0.25, 0.18)}),
    "chemotax": dict(stage=3, role="directed", outputs=[], slots=["field"],       # field <- morphogen
        params={"gain": (0.1, 1.0, 0.4)}),
    "oriented_growth": dict(stage=3, role="anisotropy", outputs=[], slots=["axis"],
        params={"aniso": (0.1, 1.0, 0.5)}),
    "adhere": dict(stage=2, role="adhesion", outputs=[], slots=[],
        params={"adhesion": (0.1, 1.0, 0.5)}),
}
STAGES = {1: ["interface_relax", "tissue_grow"],
          2: ["cleft_induce", "confine", "adhere"],
          3: ["react_rd", "chemotax", "oriented_growth"]}

# which (output_type -> slot) connections are type-legal
LEGAL_LINKS = {("morphogen", "source"), ("morphogen", "gate"),
               ("morphogen", "field"), ("morphogen", "axis")}


class CompositionGraph:
    def __init__(self, ops=None, conns=None, params=None):
        # ops: list of {"id": str, "op": name}; conns: list of {"src": id, "dst": id, "slot": str}
        self.ops = [dict(o) for o in (ops or [])]
        self.conns = [dict(c) for c in (conns or [])]
        self.params = dict(params or {})                      # "<node_id>.<param>" -> value (theta)

    # -- identity: STRUCTURE only (operators + typed connections), no param values ------------------
    def structure(self):
        ops = sorted(({"id": o["id"], "op": o["op"]} for o in self.ops), key=lambda o: (o["op"], o["id"]))
        conns = sorted(({"src_op": self._op_of(c["src"]), "dst_op": self._op_of(c["dst"]),
                         "slot": c["slot"]} for c in self.conns),
                       key=lambda c: (c["src_op"], c["dst_op"], c["slot"]))
        return {"operators": ops, "connections": conns}

    def _op_of(self, node_id):
        return next((o["op"] for o in self.ops if o["id"] == node_id), None)

    def op_names(self):
        return [o["op"] for o in self.ops]

    def copy(self):
        return CompositionGraph(self.ops, self.conns, self.params)

    # -- parameter assignment (theta) ---------------------------------------------------------------
    def default_params(self):
        p = {}
        for o in self.ops:
            for pname, (lo, hi, dflt) in OPERATORS[o["op"]]["params"].items():
                p[f"{o['id']}.{pname}"] = dflt
        return p

    def sample_params(self, rng):
        p = {}
        for o in self.ops:
            for pname, (lo, hi, _d) in OPERATORS[o["op"]]["params"].items():
                p[f"{o['id']}.{pname}"] = float(rng.uniform(lo, hi))
        return p

    def with_params(self, params):
        g = self.copy(); g.params = dict(params); return g

    # -- one-edit mutation API (each returns a NEW graph; the edit is recorded) ----------------------
    def _new_id(self, op):
        n = sum(1 for o in self.ops if o["op"] == op)
        return f"{op}{n}"

    def legal_edits(self, max_stage=3):
        """The legal ONE-edit moves from here, gated to <= max_stage. Returns (edit_tuple, label)."""
        edits = []
        present = self.op_names()
        for stage in range(1, max_stage + 1):
            for op in STAGES[stage]:
                # allow a second copy only for morphogen producers (react_rd) -- the "appear twice" case
                if op not in present or op == "react_rd":
                    edits.append((("add_op", op), f"+{op}"))
        for o in self.ops:                                    # removals (keep >=1 substrate op)
            if not (OPERATORS[o["op"]]["role"] == "substrate" and
                    sum(1 for x in self.ops if OPERATORS[x["op"]]["role"] == "substrate") == 1):
                edits.append((("remove_op", o["id"]), f"-{o['op']}"))
        for src, dst, slot in self._candidate_links():        # typed rewires
            edits.append((("connect", src, dst, slot), f"~{self._op_of(src)}->{self._op_of(dst)}.{slot}"))
        for c in self.conns:
            edits.append((("disconnect", c["src"], c["dst"], c["slot"]),
                          f"x{self._op_of(c['src'])}->{self._op_of(c['dst'])}.{c['slot']}"))
        return edits

    def _candidate_links(self):
        out = []
        for s in self.ops:
            for otype in OPERATORS[s["op"]]["outputs"]:
                for d in self.ops:
                    if d["id"] == s["id"]:
                        continue
                    for slot in OPERATORS[d["op"]]["slots"]:
                        if (otype, slot) in LEGAL_LINKS and \
                           not any(c["src"] == s["id"] and c["dst"] == d["id"] and c["slot"] == slot
                                   for c in self.conns):
                            out.append((s["id"], d["id"], slot))
        return out

    def apply(self, edit):
        """Return (new_graph, edit) after one legal move. new_graph gets fresh default params for any
        added node (theta is Loop II's job); comp_hash changes iff the STRUCTURE changed."""
        g = self.copy(); kind = edit[0]
        if kind == "add_op":
            nid = self._new_id(edit[1]); g.ops.append({"id": nid, "op": edit[1]})
            for pname, (lo, hi, dflt) in OPERATORS[edit[1]]["params"].items():
                g.params[f"{nid}.{pname}"] = dflt
        elif kind == "remove_op":
            nid = edit[1]
            g.ops = [o for o in g.ops if o["id"] != nid]
            g.conns = [c for c in g.conns if c["src"] != nid and c["dst"] != nid]
            g.params = {k: v for k, v in g.params.items() if not k.startswith(nid + ".")}
        elif kind == "connect":
            g.conns.append({"src": edit[1], "dst": edit[2], "slot": edit[3]})
        elif kind == "disconnect":
            g.conns = [c for c in g.conns
                       if not (c["src"] == edit[1] and c["dst"] == edit[2] and c["slot"] == edit[3])]
        return g, edit

    # -- surrogate encoding + post-hoc naming -------------------------------------------------------
    def encode(self):
        """Fixed-length structural feature vector for a surrogate (presence counts + link flags)."""
        feat = [sum(1 for o in self.ops if o["op"] == op) for op in OPERATORS]
        for (otype, slot) in sorted(LEGAL_LINKS):
            feat.append(1.0 if any(c["slot"] == slot and
                        otype in OPERATORS[self._op_of(c["src"])]["outputs"] for c in self.conns) else 0.0)
        return np.array(feat, np.float32)

    def name_region(self):
        """Label a DISCOVERED composition post-hoc (never chosen a priori)."""
        ops = set(self.op_names())
        rd_to_cleft = any(c["slot"] == "source" for c in self.conns)
        if "react_rd" in ops and (rd_to_cleft or "cleft_induce" not in ops):
            return "turing-like (Menshykau-Iber)"
        if "cleft_induce" in ops and "confine" in ops and "react_rd" not in ops:
            return "focal-ECM under confinement"
        if "cleft_induce" in ops and "react_rd" not in ops:
            return "focal-ECM (Yamada)"
        if "cleft_induce" in ops and "react_rd" in ops:
            return "ecm+turing mixture"
        if "confine" in ops and "cleft_induce" not in ops:
            return "confined-growth buckling (Varner-Nelson)"
        if "adhere" in ops and "cleft_induce" not in ops:
            return "differential-adhesion (Steinberg)"
        return "unnamed"


# ------------------------------------------------------------------ seed compositions
def seed(kind="substrate"):
    """Minimal seed compositions Loop I grows from."""
    if kind == "substrate":                                    # bare dense tissue (Stage 1 only)
        return CompositionGraph(ops=[{"id": "interface_relax0", "op": "interface_relax"},
                                     {"id": "tissue_grow0", "op": "tissue_grow"}])
    if kind == "empty":
        return CompositionGraph()
    raise ValueError(kind)


if __name__ == "__main__":
    from run_record import comp_hash
    g = seed("substrate")
    print("seed ops:", g.op_names(), "hash:", comp_hash(g), "region:", g.name_region())
    print("legal edits (stage<=2):", [lbl for _, lbl in g.legal_edits(2)])
    g2, e = g.apply(("add_op", "cleft_induce"))
    print("after +cleft_induce:", g2.op_names(), "hash:", comp_hash(g2), "region:", g2.name_region())
    g3, _ = g2.apply(("add_op", "react_rd"))
    src = [o["id"] for o in g3.ops if o["op"] == "react_rd"][0]
    dst = [o["id"] for o in g3.ops if o["op"] == "cleft_induce"][0]
    g4, _ = g3.apply(("connect", src, dst, "source"))
    print("react_rd->cleft.source:", g4.name_region(), "hash:", comp_hash(g4), "enc:", g4.encode().astype(int).tolist())
    assert comp_hash(g2) != comp_hash(g4)                       # topology change -> identity change
    assert comp_hash(g2.with_params(g2.sample_params(np.random.default_rng(0)))) == comp_hash(g2)  # theta != identity
    print("composition graph + one-edit API OK")
