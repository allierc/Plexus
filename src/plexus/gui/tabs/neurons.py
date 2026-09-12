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


def build_spec(form: dict) -> dict:
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
        {"op": "neuron_signal", "at": "neuron", "model": "type_pairwise", "edge_set": "synapse",
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
            "synapse": {"parent": "brain", "edge_set": True, "pre": "neuron", "post": "neuron",
                        "edges": edges, "weights": weights},
        },
        "fields": {"omega": {"frame": "grid", "res": 64, "components": 1}},
        "operators": ops,
        "schedule": sched,
        "plotting": {"renderer": "neural_panel", "background": "black",
                     "panel": {k: v for k, v in panel.items() if v is not None},
                     "max_frames": int(form.get("movie_frames", 300)), "stills": int(form.get("stills", 10)),
                     "keep_stills": True},
    }


def form_from_spec(spec: dict) -> dict:
    g = spec.get("general") or {}
    sets = spec.get("sets") or {}
    ops = {o.get("op"): o for o in (spec.get("operators") or [])}
    pl = spec.get("plotting") or {}
    ntypes = (sets.get("neuron") or {}).get("types") or {}
    gain = next((t.get("p", [0, 0, 4.0])[2] for t in ntypes.values()), 4.0)
    dale = any(str((t or {}).get("sign", "")) for t in ntypes.values())
    n = sum(int((t or {}).get("count", 0)) for t in ntypes.values()) or 1
    n_e = sum(int((t or {}).get("count", 0)) for t in ntypes.values() if str((t or {}).get("sign", "")).upper().startswith("E"))
    pulse = ops.get("activation_pulse") or {}
    return {"name": g.get("name", ""), "n_assemblies": (sets.get("assembly") or {}).get("n", 3),
            "per_assembly": (sets.get("neuron") or {}).get("per_parent", 16), "p_within": None, "p_cross": None,
            "boost": None, "frac_exc": (n_e / n) if dale else 0.8, "dale": int(dale),
            "afferent": int("E_aff" in ntypes), "drive": (ops.get("neuron_drive") or {}).get("gain", 0.0),
            "seed": g.get("seed", 7), "noise": (ops.get("neuron_update") or {}).get("noise", 0.02),
            "gain": gain, "pulse_period": pulse.get("period", 120), "pulse_duration": pulse.get("duration", 60),
            "pulse_radius": pulse.get("radius", 0.35), "n_frames": g.get("n_frames", 600), "dt": g.get("dt", 0.1),
            "movie_frames": pl.get("max_frames", 300), "stills": pl.get("stills", 10)}


FORM_HTML = r'''

 <div class="row"><label>name</label><input id="name" value="ctrnn_gui"></div>
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
const NUM=['n_assemblies','per_assembly','p_within','p_cross','boost','frac_exc','drive','gain','noise','seed','pulse_period','pulse_duration','pulse_radius','n_frames','dt','movie_frames','stills'];
const CHK=['dale','afferent'];
window.tabForm=function(){const f={name:$('name').value};for(const k of NUM)f[k]=+$(k).value;for(const k of CHK)f[k]=$(k).checked?1:0;return f;};
window.tabFill=function(f){$('name').value=f.name;for(const k of NUM)if(f[k]!==undefined&&f[k]!==null)$(k).value=f[k];for(const k of CHK)if(f[k]!==undefined&&f[k]!==null)$(k).checked=!!f[k];};
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
  POST /api/scene/run     {device} -> generate the spec (the same run as Plexus_Main.py -o generate)
  GET  /api/scene/run     -> progress {running, frame, n_frames, seconds}
  POST /api/scene/run     {stop: true}
  GET  /api/scene/state   -> {name, version, ...}: what the page shows now
  GET  /api/scene/view?azim=&elev=&zoom=&message=   -> turns the viewer's camera, shows `message`
"""
