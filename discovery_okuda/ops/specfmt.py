"""specfmt -- write a plexus2 spec.yaml with a plain-language comment on EVERY line, so the
biology of a run is legible from the spec alone. Use write_spec(cfg, path) in place of yaml.dump;
regenerate existing specs with recomment_dir()."""
import os
import glob
import yaml

# operator purpose, keyed by the `op:` value
OP = {
    "seed_mesh":     "build the honeycomb half-edge mesh on the vertex set (frame 0)",
    "seed_cell":     "init the cell set from the mesh: one live cell/face, a0=base, type=0",
    "cell_geometry": "AGGREGATE vertices -> per-cell area / perimeter / centroid (readouts)",
    "cell_paint":    "one-shot: mark a central clone (radius r) as type 1 with a0 x gain -> it bulges",
    "shape_energy":  "AVM shape-energy force on the vertices (bounded Euler); reads target area a0 from the cell set",
    "t1_transition": "reversible T1 neighbour swaps: reconnect edges shorter than l_th",
    "face_divide":   "cell division: split cells by a septum (through-vertex)",
    "face_growth":   "growth: inflate each cell's target area a0 each tick",
    "face_extrude":  "cell elimination: collapse a face to a single vertex",
    "topo_snapshot": "record the per-tick topology so a movie can be rendered",
}

# per-key meaning (state fields, params, structure)
KEY = {
    "general": "run-level settings",
    "name": "run name (-> archive/<name>/)",
    "seed": "RNG seed",
    "n_frames": "number of simulation frames",
    "dt": "time step (a label: the bounded-Euler relaxation is per-frame)",
    "boundary": "domain boundary (free = unbounded; border vertices are pinned)",
    "dim": "spatial dimension",
    "world": "world box size",
    "sets": "the biological SETS (Levels) of the model",
    "vertex": "VERTEX set: mesh-vertex positions -- the mechanical DOF",
    "cell": "CELL set: per-cell biological state (the plexus2 two-level hierarchy)",
    "n": "buffer size (max elements)",
    "state": "per-element state blocks (name: {width, integration})",
    "a0": "target cell area (biological state; drives shape_energy)",
    "ctype": "cell type / fate (0 = wild-type, 1 = clone)",
    "area": "cell area (aggregate readout from vertices)",
    "perim": "cell perimeter (aggregate readout)",
    "cen": "cell centroid (aggregate readout)",
    "width": "number of scalar columns in this state block",
    "fields": "continuous fields",
    "operators": "the operators to instantiate (each realises one mechanism)",
    "schedule": "the order the operators run each frame",
    "at": "the set this operator acts on",
    "before_frame": "run this operator once, before the given frame",
    "every": "run this operator every N frames",
    # geometry / seed params
    "nx": "lattice columns (cell centres)",
    "ny": "lattice rows",
    "a": "cell spacing (sets the base cell size)",
    "border": "rings of border cells to drop (their vertices are pinned)",
    "jitter": "initial positional disorder",
    # mechanics params
    "p0": "target shape index -- the rigidity knob (< ~3.81 solid, > ~3.81 fluid)",
    "K_A": "area-elasticity stiffness",
    "K_P": "perimeter-elasticity stiffness",
    "mu": "mobility (overdamped)",
    "relax_iters": "inner bounded-Euler relaxation steps per frame",
    "eta": "relaxation step size",
    "cap_frac": "max per-step move as a fraction of mean edge (the stability cap)",
    "v0": "active self-propulsion speed (0 = pure relaxation)",
    "Dr": "rotational diffusion of the self-propulsion direction",
    # topology params
    "l_th": "T1 threshold: an edge shorter than this reconnects",
    "ratio": "divide a cell when its target area exceeds ratio x base",
    "frac": "fraction of cells to divide (one-shot clonal division)",
    "a0_base": "base target area (the reset value after division)",
    "rate": "growth rate of the target area per frame",
    # cell-set demo params
    "gain": "clone target-area multiplier (a0 x gain)",
    "r": "clone radius (cells within r of the centre form the clone)",
}


def _annotate(txt):
    out = []
    for line in txt.rstrip("\n").split("\n"):
        s = line.strip()
        com = None
        core = s[2:].strip() if s.startswith("- ") else s
        if core.startswith("op:"):
            com = OP.get(core.split(":", 1)[1].strip())
        elif ":" in core:
            com = KEY.get(core.split(":", 1)[0].strip())
        out.append(line + (f"   # {com}" if com else ""))
    return "\n".join(out) + "\n"


def write_spec(cfg, path):
    """Dump cfg to `path` as YAML with a comment on every recognised line."""
    txt = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    with open(path, "w") as f:
        f.write(_annotate(txt))


def recomment_dir(root):
    """Retroactively re-write every archive/*/spec.yaml with comments (parses, re-dumps)."""
    n = 0
    for p in glob.glob(os.path.join(root, "*", "spec.yaml")):
        try:
            cfg = yaml.safe_load(open(p))
            write_spec(cfg, p); n += 1
        except Exception as e:
            print("  skip", p, e)
    return n


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("recommented", recomment_dir(os.path.join(here, "archive")), "spec.yaml files")
