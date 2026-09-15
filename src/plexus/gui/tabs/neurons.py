"""The neurons tab: assemblies of continuous-time neurons with a generated connectivity.

The form is what `config/neural/ctrnn_assemblies.yaml` says it was generated with -- assemblies,
neurons per assembly, the within- and cross-assembly connection probabilities, the within
boost, the seed -- plus the four neuron types' parameters and the field pulse. `build_spec`
writes that spec's shape: brain -> assembly -> neuron, a `synapse` edge-set with inline edges
and weights ~ N(0, 1/sqrt N), the Omega field with its travelling pulse, and the three operators
`neuron_field_input`, `neuron_update`, `neuron_signal` (`operators/neural.py`).
"""
from __future__ import annotations

import os

import numpy as np

from plexus.gui import studio

NAME, TITLE = "neurons", "Plexus neurons"
PICK_DIR = os.path.join(studio.REPO, "config", "neural")
CORPUS_MODE = "neurons"
REFERENCE = os.path.join(studio.REPO, "config", "neural", "ctrnn_gui.yaml")

# THE FOUR CTRNN TYPES OF THE REFERENCE, as (a decay, b offset, g gain, s self-coupling, w psi
# width, h psi threshold); with `dale: true` the form splits them into an excitatory and an
# inhibitory population instead, and the SIGN of every synapse is the presynaptic neuron's.
TYPES = {
    "slow_quiet": [0.5, 0.0, 4.0, 0.5, 1.00, 0.0],
    "slow_bursty": [0.5, 0.0, 4.0, 2.0, 0.50, 0.0],
    "fast_quiet": [2.0, 0.0, 4.0, 0.5, 2.00, 0.0],
    "fast_bursty": [2.0, 0.0, 4.0, 2.0, 0.25, 0.0],
}
DEFAULT_FORM = {"name": "ctrnn_gui", "n_assemblies": 3, "per_assembly": 16, "p_within": 0.45, "p_cross": 0.10,
                "boost": 1.8, "frac_exc": 0.8, "dale": 1, "afferent": 1, "seed": 7, "noise": 0.02, "gain": 1.5,
                "pulse_period": 120, "pulse_duration": 60, "pulse_radius": 0.35, "drive": 1.0,
                "n_frames": 600, "dt": 0.1, "movie_frames": 300, "stills": 10}


def _ring(n: int, radius: float):
    import math
    return [[round(0.5 + radius * math.cos(2 * math.pi * i / n), 6), round(0.5 + radius * math.sin(2 * math.pi * i / n), 6)]
            for i in range(n)]


def _connectivity(n_a: int, per: int, p_in: float, p_x: float, boost: float, seed: int, sign_of=None,
                  balance: float = 1.0):
    """(edges [[pre, post]], weights): weights ~ N(0, 1/sqrt N), within-assembly ones times `boost`;
    with `sign_of` (a +1/-1 per neuron) every synapse takes its PRESYNAPTIC neuron's sign -- Dale's
    law, the one constraint a connectome imposes on a weight before any training does -- and the
    inhibitory ones are multiplied by `balance` (n_E / n_I, so the total E and I weight a cell
    receives are equal on average: the balanced-network convention, without which an E-dominated
    Dale network runs away to saturation)."""
    rng = np.random.RandomState(seed)
    n = n_a * per
    edges, weights = [], []
    sd = 1.0 / np.sqrt(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same = (i // per) == (j // per)
            if rng.rand() < (p_in if same else p_x):
                w = float(rng.randn() * sd * (boost if same else 1.0))
                if sign_of is not None:
                    w = float(abs(w) * sign_of[i] * (balance if sign_of[i] < 0 else 1.0))
                edges.append([i, j])
                weights.append(round(w, 6))
    return edges, weights


def _region_dir(name: str):
    """`graphs_data/neural_regions/<name>`, the tree `plexus.io.neuprint` froze."""
    from plexus.paths import graphs_data_path
    return os.path.join(graphs_data_path(), "neural_regions", str(name))


def regions() -> list:
    """Every frozen region on this host, with what it carries: neurons, edges, meshes, skeletons."""
    import json
    from plexus.paths import graphs_data_path
    root = os.path.join(graphs_data_path(), "neural_regions")
    out = []
    if not os.path.isdir(root):
        return out
    for nm in sorted(os.listdir(root)):
        d = os.path.join(root, nm)
        idx = os.path.join(d, "morphology_index.json")
        if not os.path.exists(os.path.join(d, "neurons.npz")):
            continue
        n_mesh = n_skel = 0
        if os.path.exists(idx):
            try:
                m = json.load(open(idx))
                n_mesh, n_skel = len(m.get("meshes") or {}), len(m.get("skeletons") or {})
            except Exception:                                    # noqa: BLE001
                pass
        try:
            import numpy as _np
            with _np.load(os.path.join(d, "neurons.npz"), allow_pickle=True) as z:
                n = int(z["body_id"].shape[0])
        except Exception:                                        # noqa: BLE001
            n = 0
        out.append({"name": nm, "neurons": n, "meshes": n_mesh, "skeletons": n_skel,
                    "edges": os.path.exists(os.path.join(d, "connectome.npz"))})
    return out


def _region_spec(form: dict) -> dict:
    """THE NEUPRINT SPEC THE GUI WRITES: real somata, that region's connectome, its own dynamics,
    and -- when the region carries morphology -- the points of each neuron's own file, painted with
    its voltage. The shape is `config/neural/hemibrain_cube_1000.yaml`, which was written by hand;
    what the form changes is the region, the count, the dynamics and whether the morphology is
    drawn."""
    import json
    import numpy as np
    name = str(form.get("name") or "region_scene").strip()
    reg = str(form.get("region") or "hemibrain_cube_1000")
    d = _region_dir(reg)
    with np.load(os.path.join(d, "neurons.npz"), allow_pickle=True) as z:
        n_neuron = int(z["body_id"].shape[0])
        n_type = int(np.asarray(z["type_names"], dtype=object).shape[0]) if "type_names" in z else 20
    # THE SIDE IN MICROMETRES COMES FROM THE MANIFEST, which states it: `neurons.npz` holds the
    # bounds in whatever the importer of the day used (nanometres for the zebrafish trees,
    # VOXELS for the first hemibrain cube), and `neural_seed` refuses a spec whose `length_um`
    # disagrees with the region -- rightly, since every micrometre the run reports rides on it.
    side_um = float(((json.load(open(os.path.join(d, "manifest.json"))).get("region") or {})
                     .get("side_um")) or 100.0)
    idx = {}
    if os.path.exists(os.path.join(d, "morphology_index.json")):
        idx = json.load(open(os.path.join(d, "morphology_index.json")))
    has_mesh, has_skel = bool(idx.get("meshes")), bool(idx.get("skeletons"))
    # THREE PICTURES OF ONE REGION, and they are different questions. `somata` is one coloured dot
    # per cell -- where the activity IS, and the only one a region without morphology can draw.
    # `skeleton` swells each neuron's own .swc to its radius; `mesh` fills its .obj. Lumping the
    # last two under one "morphology" entry meant a region with both could not be asked for the
    # cheap one, and a region with neither silently fell back to dots with no word said.
    _rq = str(form.get("render", "somata")).lower()
    if _rq == "morphology":                          # the old name: whichever the region carries
        _rq = "mesh" if has_mesh else ("skeleton" if has_skel else "somata")
    if _rq == "mesh" and not has_mesh:
        raise ValueError(f"region {reg!r} carries no meshes; ask for `skeleton` or `somata`")
    if _rq == "soma_skeleton" and not has_skel:
        raise ValueError(f"region {reg!r} carries no skeletons; ask for `somata`")
    if _rq == "skeleton" and not has_skel:
        raise ValueError(f"region {reg!r} carries no skeletons -- for the zebrafish oculomotor "
                         f"circuit they need figures/zebrafish/fetch_zebrafish_anatomy_OCULO.py "
                         f"and a neuprint-fish2 token. Ask for `somata`.")
    morph = _rq in ("mesh", "skeleton", "soma_skeleton")
    spec = {
        "general": {"name": name, "seed": int(form.get("seed", 0)), "n_frames": int(form.get("n_frames", 400)),
                    "dt": float(form.get("dt", 0.05)), "dim": 3, "world": [1.0, 1.0, 1.0],
                    "boundary": "wall", "record_cap": int(form.get("n_frames", 400)) + 1,
                    "units": {"length_um": round(side_um, 6), "time_s": 1.0}},
        "sets": {
            "brain": {"n": 1},
            "assembly": {"parent": "brain", "per_parent": 1},
            "neuron": {"parent": "assembly", "per_parent": n_neuron,
                       "types": {"slow": {"fraction": 0.5, "p": [1.0, 0.0, float(form.get("gain", 1.2)), 0.5, 1.0, 0.0]},
                                 "fast": {"fraction": 0.5, "p": [2.5, 0.0, float(form.get("gain", 1.2)), 1.6, 0.5, 0.0]}}},
            "synapse": {"parent": "brain", "edge_set": True, "pre": "neuron", "post": "neuron",
                        "edges_file": f"neural_regions/{reg}/connectome.npz"},
        },
        "seed": [{"op": "neural_seed", "at": "neuron", "region": reg, "v0_mean": 0.0,
                  "v0_sd": float(form.get("v0_sd", 0.5))}],
        "operators": [
            {"op": "neuron_update", "at": "neuron", "model": "leaky_tanh", "noise": float(form.get("noise", 0.01))},
            {"op": "neuron_signal", "at": "neuron", "model": "type_pairwise", "edge_set": "synapse",
             "activation": "tanh"},
        ],
        "schedule": ["neuron_update", "neuron_signal"],
        # `dot_size`, NOT `point_size`. The renderer reads `plotting.dot_size` (live_movie.py); the
        # key written here was read by nothing, so a region of a few hundred somata was drawn at the
        # 2 px default in a full box -- picked from the menu, it looked like no 3D at all. Lit
        # sphere sprites at 9 px, and the box frame, so a sparse circuit reads as a circuit.
        "fields": {},                      # the schema requires the block; a region scene has none
        "plotting": {"renderer": "vtk_points", "background": "black",
                     "dot_size": 9.0, "dot_shading": True, "box_frame": True, "up_axis": 2,
                     "color_field": "voltage", "color_range": [-1.5, 1.5],
                     "max_frames": int(form.get("movie_frames", 300)), "stills": int(form.get("stills", 10)),
                     "keep_stills": True},
    }
    if morph:
        # THE MATTER OF EACH NEURON, from its own file: a mesh where the region has one, else the
        # skeleton swollen to its radius. `paint_children` carries the soma's voltage onto them,
        # which is what makes the picture the activity rather than the anatomy.
        # THE POINTS OF EACH NEURON'S OWN FILE. `per_parent` is a plain count here: a budget split
        # by span needs every file measured first, which is a second pass over 1,002 files for a
        # number the picture does not depend on.
        _per = max(int(form.get("per_neuron", 400)), 8)
        spec["sets"]["morphology"] = {"entity": "points", "parent": "neuron", "per_parent": _per}
        spec["seed"].append({"op": "morphology_seed", "at": "morphology", "parent": "neuron",
                             "region": reg, "kind": _rq, "seed": int(form.get("seed", 0))})
        spec["plotting"].update(subject="morphology", color_field="paint", dot_size=1.6,
                                dot_shading=False, field_cmap="tab20",
                                color_range=[0.0, float(max(n_type - 1, 1))])
        # PAINT THE POINTS WITH THE PARENT'S VOLTAGE, every frame: the picture is the ACTIVITY
        # drawn on the anatomy, not the anatomy. Without it a skeleton render is a grey tree.
        spec["operators"].append({"op": "paint_children", "at": "morphology", "parent": "neuron",
                                  "block": "paint", "parent_block": "voltage"})
        spec["schedule"].append("paint_children")
    return spec


def build_spec(form: dict) -> dict:
    if str((form or {}).get("source", "assemblies")) == "region":
        return _region_spec({k: v for k, v in (form or {}).items() if v is not None})
    # A FORM READ BACK OFF A SPEC HAS HOLES: `p_within`, `p_cross` and `boost` cannot be recovered
    # from a written connectome (the edges are there, the probabilities that drew them are not), so
    # the reader returns None and a patch of one unrelated field used to die on float(None).
    form = {k: v for k, v in (form or {}).items() if v is not None}
    name = str(form.get("name") or "ctrnn").strip()
    n_a = int(form.get("n_assemblies", 3)); per = int(form.get("per_assembly", 16))
    if n_a < 1 or per < 1:
        raise ValueError("at least one assembly of one neuron")
    seed = int(form.get("seed", 7))
    gain = float(form.get("gain", 4.0))
    dale = bool(int(form.get("dale", 1) or 0))
    frac_exc = min(1.0, max(0.0, float(form.get("frac_exc", 0.8))))
    afferent = bool(int(form.get("afferent", 1) or 0))
    n = n_a * per
    if dale:
        # E and I populations, Dale-signed. Counts on a contained set are PER PARENT, so each
        # assembly holds the same E_aff / E / I split, laid out in that order inside its block
        # (`type_layout: ordered`); the sign vector below follows that layout. The afferent E
        # sub-population is the one the drive lands on (`at: neuron[type=E_aff]`).
        n_e = int(round(frac_exc * per)); n_i = per - n_e
        n_aff = max(1, n_e // 4) if afferent and n_e > 1 else 0
        types = {}
        if n_aff:
            types["E_aff"] = {"count": n_aff, "sign": "E", "p": [0.5, 0.0, gain, 0.5, 1.0, 0.0]}
        if n_e - n_aff:
            types["E"] = {"count": n_e - n_aff, "sign": "E", "p": [0.5, 0.0, gain, 0.5, 1.0, 0.0]}
        if n_i:
            types["I"] = {"count": n_i, "sign": "I", "p": [2.0, 0.0, gain, 0.5, 2.0, 0.0]}
        sign_of = np.array([1.0 if (i % per) < n_e else -1.0 for i in range(n)])
        balance = (n_e / n_i) if n_i else 1.0
        neuron_extra = {"type_layout": "ordered"}
    else:
        types = {nm: {"fraction": 0.25, "p": [p[0], p[1], gain, p[3], p[4], p[5]]} for nm, p in TYPES.items()}
        sign_of = None
        balance = 1.0
        neuron_extra = {}
    edges, weights = _connectivity(n_a, per, float(form.get("p_within", 0.45)), float(form.get("p_cross", 0.10)),
                                   float(form.get("boost", 1.8)), seed, sign_of, balance)
    if not edges:
        raise ValueError("the connection probabilities produced no synapse; raise p_within or p_cross")
    ops = [
        # THE CLOCK THE PULSE READS. `activation_pulse` paints p(t) * gaussian, and p(t) is what a
        # `pacemaker` publishes; without one the field stays at zero (the reference ctrnn spec
        # had that defect: its neurons ran on noise alone).
        {"op": "pacemaker", "at": "omega", "period": int(form.get("pulse_period", 120)),
         "duration": int(form.get("pulse_duration", 60))},
        {"op": "activation_pulse", "at": "omega", "period": int(form.get("pulse_period", 120)),
         "duration": int(form.get("pulse_duration", 60)), "radius": float(form.get("pulse_radius", 0.35)),
         "center": [0.3, 0.5]},
        {"op": "neuron_field_input", "at": "neuron", "from": "omega", "offset": 1.0},
        {"op": "neuron_update", "at": "neuron", "model": "leaky_tanh", "noise": float(form.get("noise", 0.02))},
        {"op": "neuron_signal", "at": "neuron", "model": "type_pairwise", "edge_set": "connectivity",
         "activation": "tanh", "field": "omega"},
    ]
    sched = ["pacemaker", "activation_pulse", "neuron_field_input", "neuron_update", "neuron_signal"]
    drive = float(form.get("drive", 0.0))
    if dale and "E_aff" in types and drive > 0:
        ops.insert(3, {"op": "neuron_drive", "at": "neuron[type=E_aff]", "from": "omega", "gain": drive})
        sched.insert(3, "neuron_drive")
    panel = {"input": ["E_aff"] if "E_aff" in types else None, "output": list(types), "kino_frames": 300}
    return {
        "general": {"name": name, "seed": seed, "n_frames": int(form.get("n_frames", 600)),
                    "dt": float(form.get("dt", 0.1)), "dim": 2, "world": 1.0, "boundary": "wall"},
        "sets": {
            "brain": {"n": 1},
            # ASSEMBLIES ON A RING, NAMED BY `start:`, so every neuron is inside the box: a
            # contained assembly is placed at random and a whole group could sit against the wall.
            "assembly": {"n": n_a, "start": _ring(n_a, 0.30 if n_a > 1 else 0.0), "radius": 0.12},
            "neuron": {"parent": "assembly", "per_parent": per, "radius": 0.10, "types": types, **neuron_extra},
            # `connectivity`, NOT `synapse`. These weights are N(0, 1/sqrt N) draws, not
            # measurements; a synapse is a measured thing and the word should stay available
            # for when the weights actually came from tissue (`_region_spec` above, which does
            # call its set `synapse`). `entity: connection` gives it the same `w` block.
            "connectivity": {"parent": "brain", "edge_set": True, "entity": "connection",
                             "pre": "neuron", "post": "neuron",
                             "edges": edges, "weights": weights},
        },
        "fields": {"omega": {"frame": "grid", "res": 64, "components": 1}},
        "operators": ops,
        "schedule": sched,
        "plotting": {"renderer": ("vtk_points" if str(form.get("render", "connectivity")) == "morphology"
                                  else "neural_panel"),
                     "color_field": ("x" if str(form.get("render", "connectivity")) == "morphology" else None),
                     "background": "black",
                     "panel": {k: v for k, v in panel.items() if v is not None},
                     "max_frames": int(form.get("movie_frames", 300)), "stills": int(form.get("stills", 10)),
                     "keep_stills": True},
    }


def form_from_spec(spec: dict) -> dict:
    g = spec.get("general") or {}
    # READ BACK WHAT THE SPEC IS, not just how it is drawn. A region scene carries a
    # `neural_seed`, and the presence of a `morphology` set -- and its form -- says whether the
    # picture is the somata, the skeletons or the meshes. Reporting "morphology" for anything
    # drawn with vtk_points put the menu on a value the builder no longer accepts.
    _seed = {o.get("op"): o for o in (spec.get("seed") or [])}
    _sets = spec.get("sets") or {}
    _mf = str(((_sets.get("morphology") or {}).get("form") or ""))
    _ms = _seed.get("morphology_seed") or {}
    if "neural_seed" in _seed:
        _src, _reg = "region", str(_seed["neural_seed"].get("region") or "")
        _r = (str(_ms.get("kind") or ("mesh" if _mf.startswith("mesh:") else "skeleton"))
              if "morphology" in _sets else "somata")
    else:
        _src, _reg = "assemblies", ""
        _r = ("morphology" if str(((spec.get("plotting") or {}).get("renderer") or "")) == "vtk_points"
              else "connectivity")
    sets = spec.get("sets") or {}
    ops = {o.get("op"): o for o in (spec.get("operators") or [])}
    pl = spec.get("plotting") or {}
    ntypes = (sets.get("neuron") or {}).get("types") or {}
    gain = next((t.get("p", [0, 0, 4.0])[2] for t in ntypes.values()), 4.0)
    dale = any(str((t or {}).get("sign", "")) for t in ntypes.values())
    n = sum(int((t or {}).get("count", 0)) for t in ntypes.values()) or 1
    n_e = sum(int((t or {}).get("count", 0)) for t in ntypes.values() if str((t or {}).get("sign", "")).upper().startswith("E"))
    pulse = ops.get("activation_pulse") or {}
    return {"name": g.get("name", ""), "render": _r, "source": _src, "region": _reg, "n_assemblies": (sets.get("assembly") or {}).get("n", 3),
            "per_assembly": (sets.get("neuron") or {}).get("per_parent", 16), "p_within": None, "p_cross": None,
            "boost": None, "frac_exc": (n_e / n) if dale else 0.8, "dale": int(dale),
            "afferent": int("E_aff" in ntypes), "drive": (ops.get("neuron_drive") or {}).get("gain", 0.0),
            "seed": g.get("seed", 7), "noise": (ops.get("neuron_update") or {}).get("noise", 0.02),
            "gain": gain, "pulse_period": pulse.get("period", 120), "pulse_duration": pulse.get("duration", 60),
            "pulse_radius": pulse.get("radius", 0.35), "n_frames": g.get("n_frames", 600), "dt": g.get("dt", 0.1),
            "movie_frames": pl.get("max_frames", 300), "stills": pl.get("stills", 10)}


FORM_HTML = r'''

 <div class="row"><label>name</label><input id="name" value="ctrnn_gui"></div>
 <div class="row"><label title="where the neurons come from: assemblies drawn here, or a region frozen from neuprint">source</label><select id="source" style="width:150px" onchange="srcChanged()"><option value="assemblies">synthetic assemblies</option><option value="region">neuprint region</option></select> <select id="region" style="width:170px" onchange="build()"></select></div>
 <div class="row"><label title="what the picture is: the circuit panel, or the neurons' own morphology coloured by their activity">render</label><select id="render" style="width:150px" onchange="setRender()"><option value="somata">somata (3D dots)</option><option value="soma_skeleton">somata + skeletons</option><option value="skeleton">skeletons</option><option value="mesh">meshes</option><option value="connectivity">connectivity panel</option></select> <span style="color:#778;font-size:11px">morphology needs a spec whose neurons carry a region</span></div>
 <div class="row"><label>assemblies</label><input id="n_assemblies" class="short" value="3"> <label style="width:70px">neurons each</label><input id="per_assembly" class="short" value="16"></div>
 <div class="row"><label title="share of excitatory neurons; the rest are inhibitory">excitatory</label><input id="frac_exc" class="short" value="0.8"> <label style="width:70px" title="Dale's law: every synapse takes the sign of its presynaptic neuron"><input type="checkbox" id="dale" checked style="width:auto"> Dale</label> <label style="width:80px" title="an afferent E sub-population receives the drive"><input type="checkbox" id="afferent" checked style="width:auto"> afferent</label></div>
 <div class="row"><label title="current into the afferent neurons from the field pulse (0 = none)">drive</label><input id="drive" class="short" value="1.0"></div>
 <div class="row"><label title="connection probability inside an assembly">p within</label><input id="p_within" class="short" value="0.45"> <label style="width:70px" title="connection probability across assemblies">p cross</label><input id="p_cross" class="short" value="0.10"> <label style="width:40px" title="within-assembly weights are multiplied by this">boost</label><input id="boost" class="short" value="1.8"></div>
 <div class="row"><label title="coupling gain g of every type">gain</label><input id="gain" class="short" value="1.5"> <label style="width:70px" title="sd of the per-step process noise">noise</label><input id="noise" class="short" value="0.02"> <label style="width:40px">seed</label><input id="seed" class="short" value="7"></div>
 <div class="row"><label title="the travelling Omega pulse: period and duration in frames, radius in the unit box">pulse</label><input id="pulse_period" class="short" value="120"> <input id="pulse_duration" class="short" value="60"> <input id="pulse_radius" class="short" value="0.35"></div>
 <div class="row"><label>frames</label><input id="n_frames" class="short" value="600"> <label style="width:70px">dt</label><input id="dt" class="short" value="0.1"></div>
 <div class="row"><label title="frames the movie keeps">movie frames</label><input id="movie_frames" class="short" value="300"> <label style="width:70px" title="PNG stills through the run, also the live pictures">stills</label><input id="stills" class="short" value="10"></div>
 <div style="color:#778;font-size:11px">dx_i/dt = -a x_i + s tanh(x_i) + I_i + g Omega_i sum_j W_ij psi(x_j); weights ~ N(0, 1/sqrt N), signed by the presynaptic neuron under Dale. The picture is the circuit panel: the message |W r| on the post x pre matrix (blue E, red I), the input and rate vectors beside it, the output per population, the kinograph below. OPEN a connectome spec (config/neural/zebrafish_om_285.yaml) to see a real circuit.</div>
'''

FORM_JS = r'''
const SEL=['render','source','region'];
// THE REGIONS ON THIS HOST, from the server: one entry per frozen neuprint tree, with what it
// carries. A region with meshes can be drawn as morphology; one with only skeletons still can.
async function fillRegions(){const j=await (await fetch('/api/neurons/regions')).json();const sel=$('region');sel.innerHTML='';
 for(const r of (j.regions||[])){const o=document.createElement('option');o.value=r.name;
  o.textContent=`${r.name}  (${r.neurons} neurons, ${r.meshes?r.meshes+' meshes':r.skeletons+' skeletons'})`;sel.appendChild(o);}
 srcChanged();}
window.srcChanged=function(){const on=$('source').value==='region';$('region').style.display=on?'':'none';
 for(const id of ['n_assemblies','per_assembly','p_within','p_cross','boost','frac_exc'])if($(id))$(id).disabled=on;};
fillRegions();
// THE THREE MENUS THAT DECIDE WHAT THE SPEC IS must ride in the form, and they did not. `source`,
// `region` and `render` were read from the page only when a menu fired, while `tabForm()` -- what
// every rebuild actually sends -- carried none of them. So picking a region rebuilt the synthetic
// assemblies, and switching somata/skeletons/meshes patched a key that the next rebuild dropped:
// the picture never changed, which is exactly what was reported. A render change also has to
// REBUILD, not patch, because the three modes are different SETS, not a different style.
window.setRender=async function(){await build();};
const NUM=['n_assemblies','per_assembly','p_within','p_cross','boost','frac_exc','drive','gain','noise','seed','pulse_period','pulse_duration','pulse_radius','n_frames','dt','movie_frames','stills'];
const CHK=['dale','afferent'];
window.tabForm=function(){const f={name:$('name').value,source:$('source').value,
 region:$('region').value,render:$('render').value};
 for(const k of NUM)f[k]=+$(k).value;for(const k of CHK)f[k]=$(k).checked?1:0;return f;};
window.tabFill=function(f){$('name').value=f.name;
 for(const k of ['source','region','render'])if(f[k])$(k).value=f[k];for(const k of NUM)if(f[k]!==undefined&&f[k]!==null)$(k).value=f[k];for(const k of CHK)if(f[k]!==undefined&&f[k]!==null)$(k).checked=!!f[k];};
window.tabInit=function(){};
'''

BRIEF = """You are driving the Plexus neurons page, a UI for DEFINING a circuit of continuous-time neurons
in assemblies with a generated connectivity, and running it. A person is watching the page: it
follows every change you make. Say, in ONE short plain-English line before each call, what you are
about to do and why; say what you found after it. No headers, no markdown.

Your ONLY tool is `curl` against the local server at http://127.0.0.1:{port} . The routes:

You have curl, sleep and jq ONLY: no python, no ls, no files. Put the JSON body inline in `curl -d '...'`
(one line, however long); a body you cannot write inline you cannot send.

  POST /api/tab/neurons/build  JSON form -> writes and seeds a spec. Fields: name, n_assemblies,
                          per_assembly, p_within, p_cross, boost, frac_exc (share excitatory), dale (0|1),
                          afferent (0|1), drive, gain, noise, seed, pulse_period, pulse_duration,
                          pulse_radius, n_frames, dt, movie_frames, stills.
  POST /api/scene/patch   {form: {...}, bodies: {"*"|<name>: {...}}} -> change a FEW fields of the
                          scene on screen and rebuild. THE FIRST THING TO REACH FOR: one short call,
                          where re-sending the whole form is thousands of characters.
  POST /api/scene/run     {device} -> generate the spec (the same run as Plexus_Main.py -o generate)
  GET  /api/scene/run     -> progress {running, frame, n_frames, seconds}
  POST /api/scene/run     {stop: true}
  GET  /api/scene/state   -> {name, version, ...}: what the page shows now
  GET  /api/scene/view?azim=&elev=&zoom=&message=   -> turns the viewer's camera, shows `message`
"""
