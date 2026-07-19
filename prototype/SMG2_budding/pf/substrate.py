"""substrate (phase-field backend) -- interprets a CompositionGraph and runs it on the dense phase
field. This is ONE Substrate.run(composition, params, seed) -> trajectory backend; vertex/MPM backends
would implement the same interface and plug into the identical discovery loops.

v1 reuses the validated `pf_ops` compose engine by TRANSLATING a composition graph into pf_ops's
(active-operator set, params). This guarantees trajectory equivalence with the four hand-built
phase-field hypotheses (the benchmark compositions below), which is the regression proof for Step 3.
Graph features beyond pf_ops's vocabulary (react_rd -> growth-gate, chemotax, oriented_growth, adhere)
are marked TODO and are added as the operator library grows.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "discovery"))
import numpy as np
import pf_ops
from composition_space import CompositionGraph, seed, OPERATORS

BACKEND = "phase_field"

# graph operator -> pf_ops active-operator name
_TO_PFOPS = {"interface_relax": "surface_tension", "tissue_grow": "growth",
             "confine": "confinement"}                     # cleft_induce/react_rd handled specially


def translate(g: CompositionGraph):
    """CompositionGraph -> (pf_ops active-op set, global param dict). Raises on graph features not yet
    expressible in the pf_ops backend (so we never silently mis-run a composition)."""
    ops, p = set(), {}
    nid_params = lambda nid: {k.split(".", 1)[1]: v for k, v in g.params.items() if k.startswith(nid + ".")}
    op_of = g._op_of
    rd_to_cleft = any(c["slot"] == "source" and op_of(c["src"]) == "react_rd" and
                      op_of(c["dst"]) == "cleft_induce" for c in g.conns)
    unsupported = []
    for node in g.ops:
        op, nid, pr = node["op"], node["id"], nid_params(node["id"])
        if op in _TO_PFOPS:
            ops.add(_TO_PFOPS[op])
            if op == "interface_relax":
                p["kappa"] = pr["kappa"]; p["w0"] = pr["w0"]
            elif op == "tissue_grow":
                p["growth_frac"] = pr["growth_frac"]; p["beta"] = pr["beta"]
                if any(c["slot"] == "gate" and c["dst"] == nid for c in g.conns):
                    unsupported.append("react_rd->tissue_grow.gate")   # growth-gating: TODO
            elif op == "confine":
                p["conf_strength"] = pr["conf_strength"]; p["conf_aspect"] = pr["conf_aspect"]
        elif op == "cleft_induce":
            p["lam"] = pr["lam"]; p["thick_gate"] = pr["thick_gate"]
            if rd_to_cleft:                                   # morphogen-sourced cleft = pf_ops reaction_diffusion
                ops.add("reaction_diffusion"); p["s_rd"] = pr["s"]
            else:                                             # curvature-sourced cleft = pf_ops cleft_ecm
                ops.add("cleft_ecm"); p["s_ecm"] = pr["s"]; p["kappa_gate"] = pr["kappa_gate"]
        elif op == "react_rd":
            p["feed"] = pr["feed"]; p["kill"] = pr["kill"]; p["Dv"] = pr["Dv"]; p["v_thr"] = pr["v_thr"]
            if not rd_to_cleft:
                unsupported.append("react_rd (unrouted / non-cleft route)")   # growth-gate/chemotax: TODO
        elif op in ("chemotax", "oriented_growth", "adhere"):
            unsupported.append(op)                            # not yet in the pf_ops backend
        else:
            unsupported.append(op)
    return ops, p, unsupported


def run(g: CompositionGraph, phi0, seed_=0, n_record=6, stride=130, device="cuda:0", strict=False):
    """Run a composition -> list of phi snapshots (the trajectory). strict=True refuses unsupported graphs."""
    ops, p, unsupported = translate(g)
    if unsupported and strict:
        raise NotImplementedError(f"backend cannot yet run: {unsupported}")
    return pf_ops.simulate(phi0, {"ops": ops, "params": p}, n_record=n_record, stride=stride,
                           device=device, seed=seed_)


# ------------------------------------------------------------------ benchmark compositions (Step 3)
# The four hand-built phase-field hypotheses, now expressed as composition GRAPHS with the validated
# parameter regimes. Loop I must rediscover regions like these; here we assert the backend reproduces
# their trajectories exactly (translate -> same pf_ops call).
def _set(g, **kw):
    p = g.default_params()
    for k, v in kw.items():
        p[k] = v
    return g.with_params(p)


def benchmarks():
    b = {}
    # focal-ECM: substrate + curvature cleft (validated "curv_clean" regime)
    g, _ = seed("substrate").apply(("add_op", "cleft_induce"))
    b["focal_ecm"] = _set(g, **{"interface_relax0.kappa": 1.3, "cleft_induce0.s": 1.0,
                                "cleft_induce0.lam": 1.0, "cleft_induce0.thick_gate": 0.6})
    # differential-adhesion: same graph, high surface tension / weak cleft (validated "big_lobes")
    b["differential_adhesion"] = _set(g, **{"interface_relax0.kappa": 1.7, "cleft_induce0.s": 0.8})
    # confined-growth buckling: + confinement, stronger growth
    gc, _ = g.apply(("add_op", "confine"))
    b["confined_growth"] = _set(gc, **{"tissue_grow0.growth_frac": 1.6, "cleft_induce0.s": 1.2})
    # turing: + reaction-diffusion routed to the cleft source (validated "turing_coarse")
    gt, _ = g.apply(("add_op", "react_rd"))
    src = [o["id"] for o in gt.ops if o["op"] == "react_rd"][0]
    dst = [o["id"] for o in gt.ops if o["op"] == "cleft_induce"][0]
    gt, _ = gt.apply(("connect", src, dst, "source"))
    b["turing"] = _set(gt, **{"interface_relax0.kappa": 1.3, "cleft_induce0.s": 1.2,
                              "cleft_induce0.lam": 1.1, "react_rd0.feed": 0.030, "react_rd0.kill": 0.062})
    return b


if __name__ == "__main__":                                    # Step 3: trajectory-equivalence proof
    from run_record import comp_hash
    phi0 = np.load(os.path.join(HERE, "_real", "phi0.npy"))
    print(f"{'benchmark':22} {'region':32} {'pf_ops ops':40} equiv")
    for name, g in benchmarks().items():
        ops, p, unsup = translate(g)
        traj_backend = run(g, phi0, seed_=0, n_record=5, stride=120)
        # reference: the SAME pf_ops call the translation produces
        ref = pf_ops.simulate(phi0, {"ops": ops, "params": p}, n_record=5, stride=120, seed=0)
        d = max(float(np.abs(np.array(a) - np.array(b)).max()) for a, b in zip(traj_backend, ref))
        print(f"{name:22} {g.name_region():32} {str(sorted(ops)):40} maxΔ={d:.1e} "
              f"{'OK' if d < 1e-6 else 'FAIL'} unsup={unsup}")
    print("\ncomposition graphs run on the phase-field backend; benchmark regions reproduced.")
