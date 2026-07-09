"""mechanism_tree -- the search space as a tree over BIOLOGICAL HYPOTHESES (not operator lists).

Each branch activates ONE hypothesis family (differential adhesion / chemotaxis / ECM guidance /
growth instability / reaction-diffusion / mechanical buckling) so the search is scientifically
interpretable, not merely combinatorial. UCB decides which hypothesis deserves compute; specs are
sampled WITHIN a branch. Every spec is built on the SMG substrate INITIALIZED FROM THE REAL DATA
(general.init = real_smg -> the worker seeds cell positions from x_list frame 0). A structured
ENCODER turns any (branch, params) into a fixed feature vector for the surrogate.

  BRANCHES                     -> {hypothesis, operators, param ranges, needs_operator}
  sample_params(branch, rng)   -> a params dict
  build_spec(branch, params..) -> a Plexus spec dict (dump with write_spec)
  encode(branch, params)       -> (feature vector, feature names)  for the surrogate
"""
from __future__ import annotations
import os, copy, hashlib
import numpy as np
import yaml

# ------------------------------------------------------------------ base substrate (real-init, 2D)
BASE = {
    "general": {"name": "smg", "seed": 0, "n_frames": 1200, "dt": 0.002,
                "boundary": "wall", "init": "real_smg"},   # init -> worker seeds from x_list[0]
    "sets": {
        "agent": {"n": 600, "buffer": 6000, "spawn": "real", "spawn_radius": 0.20,
                  "types": {"epi": {"fraction": 1.0, "move_speed": 0.30}}},
        "cell": {"n": 1, "start": [[0.5, 0.5]], "types": {"body": {"fraction": 1.0, "youngs": 80}}},
        "mpm_particle": {"parent": "cell", "per_parent": 12000, "radius": 0.30, "density": 1.0},
    },
    "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 64}},
    "operators": [
        {"op": "radius_graph", "at": "agent", "radius": 0.03},
        {"op": "polar_align", "at": "agent", "gamma": 40.0, "noise": 3.0},
        {"op": "repel", "at": "agent", "strength": 6.0, "r0": 0.020},
        {"op": "glide", "at": "agent"},
        {"op": "cell_divide", "at": "agent", "rate": 0.0, "max_occ": 0.95, "offset": 0.006},
        {"op": "mpm_anchor", "at": "mpm_particle", "mode": "substrate", "k": 40.0},
        {"op": "mpm_strain", "at": "mpm_particle"},
        {"op": "p2g", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.3, "a_max": 200},
        {"op": "agent_to_mpm", "at": "agent", "to": "mpm_grid", "agent_mass": 1.5e-5, "k": 1.0},
        {"op": "mpm_grid_update", "at": "mpm_grid", "surface_tension": 0.0, "wall_damp": 0.7},
        {"op": "g2p", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 0.7,
         "wall_contact": 0.04, "vmax": 1.0e9},
        {"op": "mpm_to_agent", "at": "agent", "from": "mpm_grid", "k": 0.7, "confine": 2.0},
        {"op": "flow_align", "at": "agent", "from": "mpm_grid", "gain": 100.0},
    ],
    "schedule": ["radius_graph", "polar_align", "repel", "glide", "cell_divide", "mpm_anchor",
                 {"substep_dt": 0.0002, "steps": ["mpm_strain", "p2g", "agent_to_mpm",
                                                  "mpm_grid_update", "g2p"]},
                 "mpm_to_agent", "flow_align"],
    "plotting": {"colors": {"epi": [1.0, 0.55, 0.2]}, "background": "black",
                 "marker": "dot", "point_size": 0.008},
}

# ------------------------------------------------------------------ hypothesis branches
# param range = (lo, hi) numeric OR [choices]. Each branch = ONE biological hypothesis.
BRANCHES = {
    "baseline_migration": dict(hypothesis="null: collective migration + exclusion, NO growth (control)",
        parent="root", needs_operator=[], field="none", growth="none",
        params={"repel.strength": (2, 12), "repel.r0": (0.012, 0.028), "polar_align.gamma": (0, 80),
                "move_speed": (0.10, 0.60), "pairwise_law": ["repel", "attraction_repulsion"]}),
    "differential_adhesion": dict(hypothesis="cell-cell vs cell-matrix adhesion differential drives budding (Wang-Yamada)",
        parent="baseline_migration", needs_operator=[], field="none", growth="grow",
        params={"pairwise_law": ["attraction_repulsion"], "ecm_boundary.strength": (10, 80),
                "ecm_boundary.adhesion": (0.1, 1.0), "cell_grow.rate": (0.1, 0.6)}),
    "chemotaxis": dict(hypothesis="cells migrate up a morphogen gradient, shaping ducts",
        parent="baseline_migration", needs_operator=[], field="static", growth="grow",
        params={"chemotax_field.gain": (0.1, 1.0), "chemotax_field.mode": ["gradient", "patch", "ring"],
                "cell_grow.rate": (0.1, 0.6)}),
    "ecm_guidance": dict(hypothesis="a deformable ECM boundary GUIDES bud/duct shape (constraint, not the growth source)",
        parent="baseline_migration", needs_operator=[], field="none", growth="grow",
        params={"ecm_boundary.strength": (10, 80), "ecm_boundary.aspect": (1.0, 3.0),
                "cell_grow.rate": (0.1, 0.6), "cell_grow.prestretch": (0.3, 0.9)}),
    "growth_instability": dict(hypothesis="localized/differential growth BUCKLES the tissue into buds",
        parent="baseline_migration", needs_operator=[], field="static", growth="grow",
        params={"growth_field.gain": (0.2, 1.5), "growth_field.mode": ["patch", "gradient", "ring"],
                "cell_grow.rate": (0.1, 0.6), "cell_grow.mode": ["isotropic", "tip"]}),
    "reaction_diffusion": dict(hypothesis="a Turing morphogen prepattern GATES budding (wavelength-set)",
        parent="baseline_migration", needs_operator=[], field="turing", growth="grow",  # gray_scott Turing = TODO
        params={"growth_gate.gain": (0.2, 1.5), "growth_gate.mode": ["patch", "gradient", "ring"],
                "cell_grow.rate": (0.1, 0.6)}),
    "mechanical_buckling": dict(hypothesis="stiffness heterogeneity (soft ducts) BUCKLES the growing tissue",
        parent="baseline_migration", needs_operator=[], field="static", growth="grow",
        params={"stiffness_field.lo": (30, 80), "stiffness_field.hi": (100, 300),
                "stiffness_field.axis": ["x", "y", "radial"], "cell_grow.rate": (0.1, 0.6)}),
}


def sample_params(branch, rng):
    out = {}
    for k, rng_ in BRANCHES[branch]["params"].items():
        out[k] = str(rng.choice(rng_)) if isinstance(rng_, list) else float(rng.uniform(*rng_))
    return out


# ------------------------------------------------------------------ spec builder
_OP_AT = {"stiffness_field": "mpm_particle"}                    # ops that live on a non-agent level


def _set_op(spec, opname, **kw):
    for o in spec["operators"]:
        if o["op"] == opname:
            o.update(kw); return
    spec["operators"].append({"op": opname, "at": _OP_AT.get(opname, "agent"), **kw})


def build_spec(branch, params, seed=0, frames=1200):
    """Compose the branch's hypothesis onto the real-init base. Returns a spec dict."""
    s = copy.deepcopy(BASE)
    s["general"]["seed"] = int(seed); s["general"]["n_frames"] = int(frames)
    s["general"]["name"] = branch
    for k, v in params.items():
        if k == "move_speed":
            s["sets"]["agent"]["types"]["epi"]["move_speed"] = v
        elif k == "pairwise_law" and v == "attraction_repulsion":
            for t in s["sets"]["agent"]["types"].values():
                t["p"] = [0.6, 0.05, 6.0, 0.02]                          # pull, pull_range, push, push_range
            _set_op(s, "attraction_repulsion", sigma=0.02, aggr="sum")   # cell-cell adhesion+repulsion
            if "attraction_repulsion" not in [o for o in s["schedule"] if isinstance(o, str)]:
                s["schedule"].insert(3, "attraction_repulsion")
        elif "." in k:
            op, param = k.split(".", 1)
            _set_op(s, op, **{param: v})
            if op not in [o for o in s["schedule"] if isinstance(o, str)] and op != "attraction_repulsion":
                s["schedule"].insert(5, op)
    return s


def write_spec(spec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    h = hashlib.md5(yaml.safe_dump(spec, sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(out_dir, f"{spec['general']['name']}_{h}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return path


# ------------------------------------------------------------------ structured encoder (surrogate input)
OPERATOR_VOCAB = ["repel", "attraction_repulsion", "polar_align", "glide", "flow_align",
                  "cell_divide", "cell_grow", "ecm_boundary", "growth_field", "slow_field",
                  "growth_gate", "chemotax_field", "stiffness_field", "gray_scott"]
SCALAR_FEATS = ["move_speed", "repel.strength", "repel.r0", "polar_align.gamma", "cell_grow.rate",
                "cell_divide.rate", "cell_grow.prestretch", "ecm_boundary.strength",
                "ecm_boundary.aspect", "ecm_boundary.adhesion", "growth_field.gain", "slow_field.gain",
                "slow_field.omega", "chemotax_field.gain", "growth_gate.gain", "stiffness_field.lo",
                "stiffness_field.hi"]
SCALAR_RANGE = {"move_speed": (0.1, 0.6), "repel.strength": (2, 12), "repel.r0": (0.012, 0.028),
                "polar_align.gamma": (0, 80), "cell_grow.rate": (0, 0.6), "cell_divide.rate": (0, 0.6),
                "cell_grow.prestretch": (0.3, 0.9), "ecm_boundary.strength": (10, 80),
                "ecm_boundary.aspect": (1.0, 3.0), "ecm_boundary.adhesion": (0.0, 1.0),
                "growth_field.gain": (0.2, 1.5), "slow_field.gain": (0.2, 1.5),
                "slow_field.omega": (0.1, 2.0), "chemotax_field.gain": (0.1, 1.0),
                "growth_gate.gain": (0.2, 1.5), "stiffness_field.lo": (30, 80),
                "stiffness_field.hi": (100, 300)}
FIELD_TYPES = ["none", "static", "slow", "turing", "siren"]
GROWTH_LAWS = ["none", "divide", "grow", "both"]


def encode(branch, params):
    b = BRANCHES[branch]
    ops = set(b.get("needs_operator", []))
    if b["growth"] in ("grow", "both"):
        ops.add("cell_grow")
    if b["growth"] in ("divide", "both") or params.get("cell_divide.rate", 0):
        ops.add("cell_divide")
    if params.get("pairwise_law") == "attraction_repulsion":
        ops.add("attraction_repulsion")
    for op in ("ecm_boundary", "growth_field", "slow_field", "growth_gate", "chemotax_field",
               "stiffness_field"):
        if any(k.startswith(op + ".") for k in params):
            ops.add(op)
    ops |= {"repel", "polar_align", "glide", "flow_align"}
    feat, names = [], []
    for op in OPERATOR_VOCAB:
        feat.append(1.0 if op in ops else 0.0); names.append(f"has_{op}")
    for k in SCALAR_FEATS:
        lo, hi = SCALAR_RANGE[k]; v = params.get(k, lo)
        feat.append(float((v - lo) / (hi - lo + 1e-9))); names.append(k)
    for ft in FIELD_TYPES:
        feat.append(1.0 if b["field"] == ft else 0.0); names.append(f"field_{ft}")
    for gl in GROWTH_LAWS:
        feat.append(1.0 if b["growth"] == gl else 0.0); names.append(f"growth_{gl}")
    for br in BRANCHES:
        feat.append(1.0 if br == branch else 0.0); names.append(f"branch_{br}")
    return np.array(feat, np.float32), names


# ------------------------------------------------------------------ audit
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'branch':22} {'hypothesis'}")
    for b in BRANCHES:
        print(f"{b:22} {BRANCHES[b]['hypothesis']}")
        p = sample_params(b, rng)
        vec, names = encode(b, p)
    print(f"\nfeature-vector length = {len(vec)}")
    outd = os.path.join(os.path.dirname(__file__), "_specs")
    for b in BRANCHES:
        write_spec(build_spec(b, sample_params(b, rng)), outd)
    print(f"wrote {len(BRANCHES)} sample specs to {outd}")
