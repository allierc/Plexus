"""translate -- CompositionGraph  <->  runnable Plexus spec (the Tyssue 3D-AVM backend).

The composition space is substrate-agnostic; this file is the ONLY place that knows how an
abstract mechanism graph becomes a spec the engine can run. Two directions:

    to_spec(graph, ...)   -> cfg dict  (config/okuda/<name>.yaml)
    from_preset(name)     -> CompositionGraph   -- proves a known-good hand recipe is EXPRESSIBLE

`from_preset` is the validation gate. Before the campaign may run, the space must be able to
express the recipes we already trust; if it cannot, the search is exploring a different space
from the one our evidence is about.

--------------------------------------------------------------------------------------------
DEFECT FIXES APPLIED AT TRANSLATION -- every generated config is born correct
--------------------------------------------------------------------------------------------
D1  clock double-gating.  The engine gates operators AND several operators kept a private
    `every`, so the effective period was every^2 (a `divide_3d every=2` fired every 4). We emit
    every=1 and let the engine own the clock. NOTE: the operator-side private counters must ALSO
    be deleted in prototype/Tyssue -- this fix is necessary but not sufficient on its own.

D2  composition-dependent dt.  run_tyssue_round.make() sets
        dt = 1.0 if (cones and not rd) else 0.02
    i.e. adding/removing the RD operators -- the campaign's single most important edit -- ALSO
    rescaled chemical:mechanical time by 50x. Every signalling verdict would be confounded.
    We emit ONE global dt for the whole campaign.

D3  recording-stride mismatch.  Vertex positions and mesh topology were recorded on different
    strides; the analysis then paired one frame's coordinates with another frame's connectivity
    and reported phantom inverted cells. We pin topo_snapshot_3d every=1 and record_cap so both
    series have equal length, and the runner asserts it.
"""
from __future__ import annotations

import copy
import os

import yaml

from composition_space import OPERATORS, CompositionGraph, seed

HERE = os.path.dirname(os.path.abspath(__file__))
TYSSUE = os.path.abspath(os.path.join(HERE, "..", "prototype", "Tyssue"))
CKPT = os.path.join(TYSSUE, "archive", "smoke_hom", "ckpt.npz")

VBUF, CBUF = 30000, 16000
DT_GLOBAL = 0.02                      # D2: ONE dt for the whole campaign, never composition-dependent

# The engine executes the schedule in this order regardless of graph insertion order: readouts
# first, then patterning, then growth, then mechanics, then topology, then recording.
SCHEDULE_ORDER = [
    "load_mesh_3d", "seed_mesh_3d", "cell_geometry_3d",
    "cell_rd_seed", "cell_adjacency", "cell_diffuse", "cell_react",
    "vesicle_growth", "morphogen_growth_3d", "shape_energy_3d", "rd_interface_tension",
    "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d",
]

# composition-space operator -> engine operator name
ENGINE_NAME = {
    "seed_mesh_3d": None,             # resolved by implementation (see _emit_seed)
    "shape_energy_3d": "shape_energy_3d",
    "reconnect_t1_3d": "reconnect_t1_3d",
    "vesicle_growth": "vesicle_growth",
    "morphogen_growth_3d": "morphogen_growth_3d",
    "divide_3d": "divide_3d",
    "extrude": "rd_interface_tension",   # the forcing term lives inside this engine op
    "cell_geometry_3d": "cell_geometry_3d",
    "cell_adjacency": "cell_adjacency",
    "cell_diffuse": "cell_diffuse",
    "cell_react": "cell_react",
    "cell_rd_seed": "cell_rd_seed",
    "rd_interface_tension": "rd_interface_tension",
}


def _p(graph, node_id, name, default=None):
    """theta lookup: '<node_id>.<param>' with the vocabulary default as fallback."""
    key = f"{node_id}.{name}"
    if key in graph.params:
        return graph.params[key]
    op = next(o["op"] for o in graph.ops if o["id"] == node_id)
    spec = OPERATORS[op]["params"].get(name)
    return spec[2] if spec else default


# --------------------------------------------------------------------------- per-operator emit
def _emit_seed(g, n, ga):
    impl = g.impl_of(n)
    if impl == "checkpoint":
        return {"op": "load_mesh_3d", "at": "vertex", "cell_set": "cell",
                "ckpt": CKPT, "before_frame": 1}
    # `before_frame: 1` is MANDATORY: without it seed_mesh_3d rebuilds the sphere every tick,
    # wiping `_mesh` -- and `hist` lives inside `_mesh`, so the topology history is destroyed and
    # the D3 alignment assertion fires. (Found by that assertion on the first real run.)
    return {"op": "seed_mesh_3d", "at": "vertex", "cell_set": "cell", "before_frame": 1,
            "n_cells": int(_p(g, n["id"], "n_cells")),
            "vseed_cv": float(_p(g, n["id"], "vseed_cv"))}


def _emit_shape_energy(g, n, ga):
    i = n["id"]
    if g.impl_of(n) == "monolayer":
        return {"op": "shape_energy_3d", "implementation": "monolayer", "at": "vertex",
                "k_v": float(_p(g, i, "K_V")), "kappa_s": float(_p(g, i, "kappa_s")),
                "h0": float(_p(g, i, "h0")), "gamma": float(_p(g, i, "Gamma")),
                "mu": 1.0, "dt": DT_GLOBAL, "relax_iters": int(_p(g, i, "relax_iters")),
                "eta": 0.08, "cap_frac": 0.12}
    return {"op": "shape_energy_3d", "at": "vertex",
            "p0": float(_p(g, i, "p0")), "K_A": 1.0, "K_P": 1.0,
            "Gamma": float(_p(g, i, "Gamma")), "Lambda": float(_p(g, i, "Lambda")),
            "K_V": float(_p(g, i, "K_V")), "K_R": 0.02, "K_bend": 0.0, "antiinv": 0.0,
            "mu": 1.0, "dt": DT_GLOBAL, "relax_iters": int(_p(g, i, "relax_iters")),
            "eta": 0.08, "cap_frac": 0.12}


def _emit_rd_seed(g, n, ga):
    i, impl = n["id"], g.impl_of(n)
    d = {"op": "cell_rd_seed", "at": "cell", "mode": impl, "n_spots": 3,
         "amp": float(_p(g, i, "amp"))}
    if impl == "tip":
        d["tip_radius"] = float(_p(g, i, "tip_radius"))       # re-seeds EVERY frame: tip-tracking
    elif impl == "cone":
        d["cone_deg"] = float(_p(g, i, "cone_deg"))
    else:                                                     # a frozen spot -- the DOME control
        d["cone_deg"] = float(_p(g, i, "cone_deg"))
        d["before_frame"] = 3
    return d


def _emit_react(g, n, ga):
    i, impl = n["id"], g.impl_of(n)
    base = {"op": "cell_react", "at": "cell", "implementation": impl,
            "rate": float(_p(g, i, "rd_rate"))}
    if impl == "gierer_meinhardt":
        base.update({"gm_rho": 1.0, "mu_a": 1.0, "mu_h": 1.0, "a0": float(_p(g, i, "a0"))})
    elif impl == "gray_scott":
        base.update({"F": 0.03, "kk": 0.062})
    else:
        base.update({"gamma": float(_p(g, i, "gamma")), "A": 1.0, "B": 3.0})
    return base


def _emit_growth(g, n, ga):
    i = n["id"]
    return {"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell",
            "rate": float(_p(g, i, "rate")), "a_sw": float(_p(g, i, "a_sw")),
            "hill": float(_p(g, i, "alpha")), "rho": float(_p(g, i, "rho")),
            "vth_frac": 1.5, "after_frame": ga, "dt": DT_GLOBAL,
            "conserve_amount": g.impl_of(n) == "hill_conserve_amount"}


def _emit_divide(g, n, ga):
    i = n["id"]
    return {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12,
            "cycle_cv": float(_p(g, i, "cycle_cv")), "p0": 3.90,
            "every": 1,                                        # D1: the ENGINE owns the clock
            "max_div": 120, "max_div_frac": 0.03,
            "vcap": float(_p(g, i, "vcap")), "cell_set": "cell",
            "min_cycle": int(_p(g, i, "min_cycle")), "max_cycle": int(_p(g, i, "max_cycle")),
            "after_frame": ga, "orient_iface": g.impl_of(n) == "orient_iface",
            "orient_asw": 1.2, "g1_ramp": False}


def _emit_extrude(g, n, ga):
    i = n["id"]
    return {"op": "rd_interface_tension", "at": "vertex", "cell_set": "cell",
            "K_purse": 0.0, "K_extrude": float(_p(g, i, "K_extrude")),
            "a_sw": 1.2, "eta": 0.05, "iters": 4, "after_frame": ga}


EMIT = {
    "seed_mesh_3d": _emit_seed,
    "shape_energy_3d": _emit_shape_energy,
    "cell_rd_seed": _emit_rd_seed,
    "cell_react": _emit_react,
    "morphogen_growth_3d": _emit_growth,
    "divide_3d": _emit_divide,
    "extrude": _emit_extrude,
    "reconnect_t1_3d": lambda g, n, ga: {
        "op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": float(_p(g, n["id"], "l_th")) * 7.0,
        "every": 1, "max_flips": 300},                          # D1
    "cell_geometry_3d": lambda g, n, ga: {"op": "cell_geometry_3d", "at": "cell"},
    "cell_adjacency": lambda g, n, ga: {"op": "cell_adjacency", "at": "cell"},
    "cell_diffuse": lambda g, n, ga: {
        "op": "cell_diffuse", "at": "cell", "d_a": float(_p(g, n["id"], "d_a")),
        "d_h": float(_p(g, n["id"], "d_h")), "chi": float(_p(g, n["id"], "chi"))},
    "vesicle_growth": lambda g, n, ga: {
        "op": "vesicle_growth", "at": "vertex", "cell_set": "cell",
        "rate": float(_p(g, n["id"], "rate")), "dt": DT_GLOBAL},
    "rd_interface_tension": _emit_extrude,
}


# --------------------------------------------------------------------------- graph -> spec
def to_spec(graph: CompositionGraph, *, name="okuda", frames=350, seed_=0, grow_after=100,
            record_every=1):
    """Compile a CompositionGraph into a runnable Plexus spec dict.

    Raises if the graph is not runnable -- a composition with an unmet precondition or a
    dangling slot must never reach the cluster (D4: it would silently no-op and its metrics
    would be recorded as evidence that the mechanism cannot work).
    """
    ok, why = graph.is_runnable()
    if not ok:
        raise ValueError(f"refusing to compile a non-runnable composition: {why}")

    ops = []
    for node in graph.ops:
        emit = EMIT.get(node["op"])
        if emit is None:
            raise ValueError(f"no backend emitter for operator {node['op']!r}")
        ops.append(emit(graph, node, grow_after))

    # D3: topology must be recorded on the SAME stride as positions, and this is asserted
    # downstream. This is the fix for the phantom "97% hollow" result.
    ops.append({"op": "topo_snapshot_3d", "at": "vertex", "every": record_every})

    unordered = sorted({o["op"] for o in ops} - set(SCHEDULE_ORDER))
    if unordered:
        raise ValueError(
            f"operators missing from SCHEDULE_ORDER: {unordered}. They would sort to the END of "
            f"the schedule -- e.g. a growth operator running AFTER the recorder. Add them.")
    order = {n: i for i, n in enumerate(SCHEDULE_ORDER)}
    ops.sort(key=lambda o: order.get(o["op"], 999))
    sched = [o["op"] for o in ops]

    cfg = {
        "general": {"name": name, "seed": int(seed_), "n_frames": int(frames),
                    "dt": DT_GLOBAL, "record_cap": int(frames) + 2,
                    "record_every": int(record_every),
                    "boundary": "free", "dim": 3, "world": [16 * 5.0] * 3},
        "sets": {"vertex": {"n": VBUF},
                 "cell": {"n": CBUF, "state": {"chem": {"width": 2, "integration": "first_order"},
                                               "cen": {"width": 3}, "area": {"width": 1}}}},
        "fields": {},
        "operators": ops,
        "schedule": sched,
        # provenance: what this spec IS, so a config file alone identifies its hypothesis
        "_discovery": {
            "comp_hash": None,                                 # filled by write_config
            "structure": graph.structure(),
            "region": graph.name_region(),
            "defect_fixes": ["D1 engine owns the clock (every=1)",
                             f"D2 one global dt={DT_GLOBAL}",
                             f"D3 topo_snapshot every={record_every} == position stride"],
        },
    }
    return cfg


def write_config(graph, name, out_dir=None, **kw):
    """Write config/okuda/<name>.yaml -- the Plexus contract path."""
    from run_record import comp_hash
    out_dir = out_dir or os.path.abspath(os.path.join(HERE, "..", "config", "okuda"))
    os.makedirs(out_dir, exist_ok=True)
    cfg = to_spec(graph, name=name, **kw)
    cfg["_discovery"]["comp_hash"] = comp_hash(graph)
    path = os.path.join(out_dir, f"{name}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path, cfg


# --------------------------------------------------------------------------- preset -> graph
def from_preset(p: dict) -> CompositionGraph:
    """Express a hand-written run_tyssue_round preset as a CompositionGraph.

    THE VALIDATION GATE. If a recipe we already trust cannot be built from legal one-edit moves,
    the campaign is searching a space that does not contain our own evidence.
    """
    g = CompositionGraph(ops=[
        {"id": "seed_mesh_3d0", "op": "seed_mesh_3d", "impl": "checkpoint"},
        {"id": "cell_geometry_3d0", "op": "cell_geometry_3d", "impl": "scatter_add"},
    ])
    add = lambda gg, op, impl: gg.apply(("add_op", op, impl))[0]

    cones = "spots" in p
    rd = bool(p.get("rd", False))

    if cones:
        g = add(g, "cell_rd_seed", p.get("seed_mode", "cones").replace("cones", "cone"))
    if rd or not cones:
        g = add(g, "cell_adjacency", "shared_edge")
        g = add(g, "cell_diffuse", "graph_laplacian")
        g = add(g, "cell_react", p.get("rd_impl", "brusselator"))
        if not cones:
            g = add(g, "cell_rd_seed", "spot")

    g = add(g, "morphogen_growth_3d",
            "hill_conserve_amount" if p.get("conserve_amount", True) else "hill_no_conserve")
    g = add(g, "shape_energy_3d", "monolayer" if p.get("monolayer") else "default")
    if p.get("K_extrude", 0.0) > 0 or p.get("K_purse", 0.0) > 0:
        g = add(g, "extrude", "radial_push")
    g = add(g, "reconnect_t1_3d", "length_threshold")
    g = add(g, "divide_3d", "orient_iface" if p.get("orient_iface") else "hertwig")

    # route the morphogen: whichever node produces it feeds growth (+ axis / site if present)
    src = next((o["id"] for o in g.ops if "morphogen" in OPERATORS[o["op"]]["outputs"]), None)
    if src is None:
        return g
    for dst_op, slot in (("morphogen_growth_3d", "gate"), ("divide_3d", "axis"),
                         ("extrude", "site")):
        dst = next((o["id"] for o in g.ops if o["op"] == dst_op), None)
        if dst is None:
            continue
        if dst_op == "divide_3d" and not p.get("orient_iface"):
            continue                                            # hertwig uses the cell's own axis
        g = g.apply(("connect", src, dst, slot))[0]

    # carry theta across so the compiled spec is numerically the preset, not the defaults
    pm = g.default_params()
    node = lambda op: next((o["id"] for o in g.ops if o["op"] == op), None)
    mapping = [
        ("shape_energy_3d", "K_V", p.get("K_V")), ("shape_energy_3d", "relax_iters", p.get("relax")),
        ("shape_energy_3d", "kappa_s", p.get("kappa_s")), ("shape_energy_3d", "h0", p.get("h0")),
        ("morphogen_growth_3d", "rate", p.get("rate")), ("morphogen_growth_3d", "a_sw", p.get("a_sw")),
        ("morphogen_growth_3d", "alpha", p.get("hill")), ("morphogen_growth_3d", "rho", p.get("rho")),
        ("divide_3d", "cycle_cv", p.get("cycle_cv")), ("divide_3d", "min_cycle", p.get("min_cycle")),
        ("divide_3d", "max_cycle", p.get("max_cycle")), ("divide_3d", "vcap", p.get("vcap")),
        ("extrude", "K_extrude", p.get("K_extrude")),
        ("cell_diffuse", "chi", p.get("chi")), ("cell_diffuse", "d_a", p.get("d_a")),
        ("cell_diffuse", "d_h", p.get("d_h")),
        ("cell_react", "rd_rate", p.get("rd_rate")), ("cell_react", "a0", p.get("a0")),
        ("cell_react", "gamma", p.get("gamma")),
        ("cell_rd_seed", "cone_deg", p.get("cone_deg")),
        ("cell_rd_seed", "tip_radius", p.get("tip_radius")),
    ]
    for op, pname, val in mapping:
        nid = node(op)
        if nid is not None and val is not None:
            pm[f"{nid}.{pname}"] = val
    return g.with_params(pm)


def load_presets():
    """Import run_tyssue_round.PRESETS without running anything."""
    import importlib.util
    import sys
    sys.path.insert(0, TYSSUE)
    spec = importlib.util.spec_from_file_location(
        "run_tyssue_round", os.path.join(TYSSUE, "run_tyssue_round.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # heavy imports may fail; the presets are what we want
        print(f"  (preset module partially loaded: {type(e).__name__}: {str(e)[:70]})")
    return getattr(mod, "PRESETS", {})
