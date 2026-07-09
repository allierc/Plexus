"""mechanism_tree -- the search space as a TREE OVER MECHANISMS (not raw YAML).

Each branch = a hypothesis family that OWNS its allowed operators + parameter ranges. UCB decides
which family deserves compute; specs are sampled WITHIN a branch. Every spec is built on the SMG
substrate INITIALIZED FROM THE REAL DATA (general.init = real_smg -> the worker seeds cell positions
from x_list frame 0). A structured ENCODER turns any (branch, params) into a fixed feature vector so
the surrogate can predict the metric vector.

  BRANCHES                     -> {operators, param ranges, needs_operator (TODO palette gaps)}
  sample_params(branch, rng)   -> a params dict
  build_spec(branch, params..) -> a Plexus spec dict (dump with write_spec)
  encode(branch, params)       -> (feature vector, feature names)  for the surrogate

`needs_operator` marks the operators the branch REQUIRES that are not yet in the palette -> build
those in operators_smg.py BEFORE the bootstrap, else that branch can only make clusters.
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

# ------------------------------------------------------------------ the mechanism tree
# param range = (lo, hi) numeric  OR  [choices]. needs_operator = palette gaps to build first.
BRANCHES = {
    "pairwise_only": dict(parent="root", needs_operator=[], field="none", growth="none",
        params={"repel.strength": (2, 12), "repel.r0": (0.012, 0.028),
                "polar_align.gamma": (0, 80), "move_speed": (0.10, 0.60),
                "pairwise_law": ["repel", "attraction_repulsion"]}),
    "migration_plus_growth": dict(parent="pairwise_only", needs_operator=[], field="none", growth="both",
        params={"polar_align.gamma": (10, 80), "move_speed": (0.15, 0.50),
                "cell_divide.rate": (0.0, 0.6), "cell_grow.rate": (0.0, 0.6)}),
    "static_growth_field": dict(parent="migration_plus_growth", needs_operator=["growth_field"],
        field="static", growth="grow",
        params={"cell_grow.rate": (0.1, 1.0), "growth_field.scale": (0.2, 1.5),
                "growth_field.mode": ["patch", "gradient", "ring"], "cell_grow.prestretch": (0.3, 0.9)}),
    "slow_growth_field": dict(parent="static_growth_field", needs_operator=["growth_field", "slow_field"],
        field="slow", growth="grow",
        params={"cell_grow.rate": (0.1, 1.0), "growth_field.scale": (0.2, 1.5),
                "slow_field.omega": (0.1, 2.0)}),
    "boundary_guided_growth": dict(parent="migration_plus_growth", needs_operator=["ecm_boundary"],
        field="none", growth="grow",
        params={"cell_grow.rate": (0.1, 1.0), "ecm_boundary.stiffness": (10, 120),
                "ecm_boundary.gap": (0.0, 0.06), "cell_grow.prestretch": (0.3, 0.9)}),
    "tip_localized_growth": dict(parent="migration_plus_growth", needs_operator=[],  # cell_grow mode=tip EXISTS
        field="none", growth="grow",
        params={"cell_grow.rate": (0.1, 1.0), "cell_grow.mode": ["tip"],
                "cell_grow.tip_sharpness": (4, 20), "cell_grow.prestretch": (0.3, 0.9)}),
    "duct_stiffness_gradient": dict(parent="migration_plus_growth", needs_operator=["stiffness_field"],
        field="static", growth="grow",
        params={"cell_grow.rate": (0.1, 0.8), "stiffness_field.lo": (30, 80),
                "stiffness_field.hi": (100, 300), "stiffness_field.axis": ["x", "y", "radial"]}),
    "signaling_like_field": dict(parent="migration_plus_growth", needs_operator=["growth_gate"],  # gray_scott EXISTS
        field="turing", growth="grow",
        params={"gray_scott.DA": (0.08, 0.24), "gray_scott.DB": (0.04, 0.12),
                "gray_scott.f": (0.030, 0.060), "gray_scott.k": (0.055, 0.065),
                "growth_gate.gain": (0.0, 1.0)}),
}


def sample_params(branch, rng):
    out = {}
    for k, rng_ in BRANCHES[branch]["params"].items():
        out[k] = str(rng.choice(rng_)) if isinstance(rng_, list) else float(rng.uniform(*rng_))
    return out


# ------------------------------------------------------------------ spec builder
def _set_op(spec, opname, **kw):
    for o in spec["operators"]:
        if o["op"] == opname:
            o.update(kw); return
    spec["operators"].append({"op": opname, "at": "agent", **kw})


def build_spec(branch, params, seed=0, frames=1200):
    """Compose the branch's mechanism onto the real-init base. Returns a spec dict.
    (NEW operators named in needs_operator are emitted as ops here; operators_smg.py must register them.)"""
    s = copy.deepcopy(BASE)
    s["general"]["seed"] = int(seed); s["general"]["n_frames"] = int(frames)
    s["general"]["name"] = branch
    for k, v in params.items():
        if k == "move_speed":
            s["sets"]["agent"]["types"]["epi"]["move_speed"] = v
        elif k == "pairwise_law" and v == "attraction_repulsion":
            _set_op(s, "attraction_repulsion", p=[0.6, 0.05, 6.0, 0.02])
        elif "." in k:
            op, param = k.split(".", 1)
            _set_op(s, op, **{param: v})
            if op not in [o for o in s["schedule"] if isinstance(o, str)] and op != "attraction_repulsion":
                s["schedule"].insert(5, op)          # run new mechanism after glide/divide
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
                  "cell_divide", "cell_grow", "growth_field", "slow_field", "ecm_boundary",
                  "stiffness_field", "gray_scott", "growth_gate"]
SCALAR_FEATS = ["move_speed", "repel.strength", "repel.r0", "polar_align.gamma", "cell_grow.rate",
                "cell_divide.rate", "cell_grow.prestretch", "growth_field.scale", "slow_field.omega",
                "ecm_boundary.stiffness", "cell_grow.tip_sharpness", "stiffness_field.lo",
                "stiffness_field.hi", "gray_scott.DA", "gray_scott.DB", "gray_scott.f",
                "gray_scott.k", "growth_gate.gain"]
SCALAR_RANGE = {"move_speed": (0.1, 0.6), "repel.strength": (2, 12), "repel.r0": (0.012, 0.028),
                "polar_align.gamma": (0, 80), "cell_grow.rate": (0, 1), "cell_divide.rate": (0, 0.6),
                "cell_grow.prestretch": (0.3, 0.9), "growth_field.scale": (0.2, 1.5),
                "slow_field.omega": (0.1, 2.0), "ecm_boundary.stiffness": (10, 120),
                "cell_grow.tip_sharpness": (4, 20), "stiffness_field.lo": (30, 80),
                "stiffness_field.hi": (100, 300), "gray_scott.DA": (0.08, 0.24),
                "gray_scott.DB": (0.04, 0.12), "gray_scott.f": (0.03, 0.06),
                "gray_scott.k": (0.055, 0.065), "growth_gate.gain": (0, 1)}
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
    ops |= {"repel", "polar_align", "glide", "flow_align"}
    feat, names = [], []
    for op in OPERATOR_VOCAB:                          # operators-present multi-hot
        feat.append(1.0 if op in ops else 0.0); names.append(f"has_{op}")
    for k in SCALAR_FEATS:                             # normalized scalar params
        lo, hi = SCALAR_RANGE[k]; v = params.get(k, lo)
        feat.append(float((v - lo) / (hi - lo + 1e-9))); names.append(k)
    for ft in FIELD_TYPES:                             # field type one-hot
        feat.append(1.0 if b["field"] == ft else 0.0); names.append(f"field_{ft}")
    for gl in GROWTH_LAWS:                             # growth law one-hot
        feat.append(1.0 if b["growth"] == gl else 0.0); names.append(f"growth_{gl}")
    for br in BRANCHES:                                # branch one-hot
        feat.append(1.0 if br == branch else 0.0); names.append(f"branch_{br}")
    return np.array(feat, np.float32), names


# ------------------------------------------------------------------ audit
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    todo = {}
    print(f"{'branch':24} {'#ops-needed':>11}  needs_operator")
    for b in BRANCHES:
        need = BRANCHES[b]["needs_operator"]
        for op in need:
            todo[op] = todo.get(op, 0) + 1
        p = sample_params(b, rng)
        vec, names = encode(b, p)
        print(f"{b:24} {len(need):>11}  {need}")
    print(f"\nfeature-vector length = {len(vec)}")
    print(f"\nPALETTE GAPS to build in operators_smg.py BEFORE bootstrap: {sorted(todo)}")
    # sanity: build+write one spec per branch
    outd = os.path.join(os.path.dirname(__file__), "_specs")
    for b in BRANCHES:
        sp = write_spec(build_spec(b, sample_params(b, rng)), outd)
    print(f"wrote {len(BRANCHES)} sample specs to {outd}")
