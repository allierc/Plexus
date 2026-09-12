"""The metabolism tab: a random mass-action network, written as a Plexus spec.

The form is `MetabolismGraph`'s `simulation:` block -- how many metabolites and reactions, how
many species a reaction may touch, the autocatalytic-cycle share, homeostasis, the circadian
modulation, the seed -- and `build_spec` generates the stoichiometry the way
`generators/utils.py::init_reaction` does (substrates and products disjoint, coefficients in {1,
2}, cycles A+B -> 2B, B+C -> 2C, ...), inline in the spec, because the language has no
`edges_file:` for a hypergraph yet. The dynamics are `operators/metabolism.py`.

Metabolites sit on a circle and reactions on an inner circle, so the picture is the bipartite
graph itself, coloured by concentration.
"""
from __future__ import annotations

import math
import os

import numpy as np

from plexus.gui import studio

NAME, TITLE = "metabolism", "Plexus metabolism"
PICK_DIR = os.path.join(studio.REPO, "config", "metabolism")
CORPUS_MODE = "metabolism"
REFERENCE = os.path.join(studio.REPO, "config", "metabolism", "massaction_toy.yaml")

DEFAULT_FORM = {"name": "massaction_toy", "n_metabolites": 40, "n_reactions": 80, "max_per_reaction": 3,
                "cycle_fraction": 0.4, "k_min": 0.001, "k_max": 0.1, "c_min": 2.5, "c_max": 7.5,
                "homeostasis": 0.02, "circadian_amplitude": 0.0, "circadian_period": 720,
                "n_frames": 1440, "dt": 0.05, "seed": 7, "movie_frames": 300, "stills": 10}


def _network(n_met: int, n_rxn: int, max_per: int, cycle_fraction: float, seed: int, cycle_length: int = 4):
    """(edges [[met, rxn]], weights [signed S_ij]) -- MetabolismGraph's init_reaction, in short."""
    rng = np.random.RandomState(seed)
    edges, weights = [], []
    n_cycle = int(n_rxn * cycle_fraction)
    r = 0
    used = set()
    while r + cycle_length <= n_cycle:
        avail = [m for m in range(n_met) if m not in used] or list(range(n_met))
        mets = rng.choice(avail, size=min(cycle_length, len(avail)), replace=False)
        used.update(int(m) for m in mets)
        for i in range(len(mets)):                       # A + B -> 2B: A and B in, 2 B out (net +1 B)
            a, b = int(mets[i]), int(mets[(i + 1) % len(mets)])
            edges += [[a, r], [b, r]]; weights += [-1.0, 1.0]
            r += 1
    while r < n_rxn:
        ns = int(rng.randint(1, max_per + 1)); npd = int(rng.randint(1, max_per + 1))
        picks = rng.choice(n_met, size=min(ns + npd, n_met), replace=False)
        subs, prods = picks[:ns], picks[ns:]
        for m in subs:
            edges.append([int(m), r]); weights.append(-float(rng.randint(1, 3)))
        for m in prods:
            edges.append([int(m), r]); weights.append(float(rng.randint(1, 3)))
        r += 1
    return edges, weights


def _circle(n: int, radius: float, phase: float = 0.0):
    return [[round(0.5 + radius * math.cos(phase + 2 * math.pi * i / n), 6),
             round(0.5 + radius * math.sin(phase + 2 * math.pi * i / n), 6)] for i in range(n)]


def build_spec(form: dict) -> dict:
    name = str(form.get("name") or "massaction").strip()
    n_met = int(form.get("n_metabolites", 40)); n_rxn = int(form.get("n_reactions", 80))
    if n_met < 2 or n_rxn < 1:
        raise ValueError("at least 2 metabolites and 1 reaction")
    max_per = max(1, min(int(form.get("max_per_reaction", 3)), n_met // 2))
    seed = int(form.get("seed", 7))
    edges, weights = _network(n_met, n_rxn, max_per, float(form.get("cycle_fraction", 0.0)), seed)
    frames = int(form.get("n_frames", 1440)); dt = float(form.get("dt", 0.05))
    lam = float(form.get("homeostasis", 0.0))
    ops = [
        {"op": "reaction_rate", "at": "reaction", "edge_set": "stoich", "c_floor": 1e-6, "flux_limit_dt": dt},
        {"op": "metabolite_flux", "at": "metabolite", "edge_set": "stoich"},
    ]
    sched = ["reaction_rate", "metabolite_flux"]
    if lam > 0 or float(form.get("circadian_amplitude", 0.0)) > 0:
        ops.append({"op": "metabolite_homeostasis", "at": "metabolite", "strength": lam,
                    "circadian_amplitude": float(form.get("circadian_amplitude", 0.0)),
                    "circadian_period": float(form.get("circadian_period", 720))})
        sched.append("metabolite_homeostasis")
    return {
        "general": {"name": name, "seed": seed, "n_frames": frames, "dt": dt, "dim": 2, "world": 1.0,
                    "boundary": "wall"},
        "sets": {
            "cell": {"n": 1},
            "metabolite": {"n": n_met, "start": _circle(n_met, 0.42),
                           "state": {"pos": {"width": 2, "role": "geometry", "integration": "none", "boundary": "world"},
                                     "conc": {"width": 1, "integration": "first_order", "boundary": "free"},
                                     "c0": {"width": 1, "integration": "none", "boundary": "free", "record": False}}},
            "reaction": {"n": n_rxn, "start": _circle(n_rxn, 0.22, 0.3),
                         "state": {"pos": {"width": 2, "role": "geometry", "integration": "none", "boundary": "world"},
                                   "v": {"width": 1, "integration": "none", "boundary": "free"},
                                   "k": {"width": 1, "integration": "none", "boundary": "free", "record": False}}},
            "stoich": {"parent": "cell", "edge_set": True, "pre": "metabolite", "post": "reaction",
                       "entity": "synapse", "edges": edges, "weights": weights},
        },
        "fields": {},
        "seed": [{"op": "metabolite_seed", "at": "metabolite", "c_min": float(form.get("c_min", 2.5)),
                  "c_max": float(form.get("c_max", 7.5)), "seed": seed},
                 {"op": "reaction_seed", "at": "reaction", "k_min": float(form.get("k_min", 0.05)),
                  "k_max": float(form.get("k_max", 0.5)), "seed": seed}],
        "operators": ops,
        "schedule": sched,
        "plotting": {"renderer": "vtk_points", "background": "black", "subject": "metabolite",
                     "color_field": "conc", "field_cmap": "viridis", "point_size": 14.0, "dot_size": 14.0,
                     "graph_overlay": {"sets": ["stoich"], "always": True, "opacity": 0.35, "line_width": 1.0},
                     "colors": {"stoich": [0.5, 0.5, 0.6]},
                     "max_frames": int(form.get("movie_frames", 300)), "stills": int(form.get("stills", 10)),
                     "keep_stills": True, "box_frame": False},
    }


def form_from_spec(spec: dict) -> dict:
    g = spec.get("general") or {}
    sets = spec.get("sets") or {}
    seeds = {o.get("op"): o for o in (spec.get("seed") or [])}
    ops = {o.get("op"): o for o in (spec.get("operators") or [])}
    hs = ops.get("metabolite_homeostasis") or {}
    pl = spec.get("plotting") or {}
    w = (sets.get("stoich") or {}).get("weights") or []
    e = (sets.get("stoich") or {}).get("edges") or []
    per = {}
    for (m, r), s in zip(e, w):
        per[r] = per.get(r, 0) + (1 if s < 0 else 0)
    return {"name": g.get("name", ""), "n_metabolites": (sets.get("metabolite") or {}).get("n", 40),
            "n_reactions": (sets.get("reaction") or {}).get("n", 80),
            "max_per_reaction": max(per.values()) if per else 3, "cycle_fraction": None,
            "k_min": (seeds.get("reaction_seed") or {}).get("k_min", 0.001), "k_max": (seeds.get("reaction_seed") or {}).get("k_max", 0.1),
            "c_min": (seeds.get("metabolite_seed") or {}).get("c_min", 2.5), "c_max": (seeds.get("metabolite_seed") or {}).get("c_max", 7.5),
            "homeostasis": hs.get("strength", 0.0), "circadian_amplitude": hs.get("circadian_amplitude", 0.0),
            "circadian_period": hs.get("circadian_period", 720), "n_frames": g.get("n_frames", 1440), "dt": g.get("dt", 0.05),
            "seed": g.get("seed", 7), "movie_frames": pl.get("max_frames", 300), "stills": pl.get("stills", 10)}


FORM_HTML = r'''
 <h2>Network</h2>
 <div class="row"><label>name</label><input id="name" value="massaction_toy"></div>
 <div class="row"><label>metabolites</label><input id="n_metabolites" class="short" value="40"> <label style="width:70px">reactions</label><input id="n_reactions" class="short" value="80"></div>
 <div class="row"><label title="substrates and products a reaction may have, each">per reaction</label><input id="max_per_reaction" class="short" value="3"> <label style="width:70px" title="share of the reactions that form autocatalytic cycles A+B -> 2B (oscillations)">cycles</label><input id="cycle_fraction" class="short" value="0.4"></div>
 <div class="row"><label title="rate constants k_j, log-uniform between these">k range</label><input id="k_min" class="short" value="0.001"> <input id="k_max" class="short" value="0.1"> <label style="width:70px" title="initial concentrations, uniform between these">c range</label><input id="c_min" class="short" value="2.5"> <input id="c_max" class="short" value="7.5"></div>
 <div class="row"><label title="pull towards the initial concentration, per unit time (0 = none)">homeostasis</label><input id="homeostasis" class="short" value="0.02"> <label style="width:70px" title="circadian modulation of the target: amplitude (0 = none) and period in frames">circadian</label><input id="circadian_amplitude" class="short" value="0"> <input id="circadian_period" class="short" value="720"></div>
 <div class="row"><label>frames</label><input id="n_frames" class="short" value="1440"> <label style="width:70px">dt</label><input id="dt" class="short" value="0.05"> <label style="width:40px">seed</label><input id="seed" class="short" value="7"></div>
 <div class="row"><label title="frames the movie keeps">movie frames</label><input id="movie_frames" class="short" value="300"> <label style="width:70px" title="PNG stills through the run, also the live pictures">stills</label><input id="stills" class="short" value="10"></div>
 <div style="color:#778;font-size:11px">mass action: v_j = k_j prod c_i^|S_ij| over the substrates; dc_i/dt = sum_j S_ij v_j - homeostasis (c_i - c_i(0)). Metabolites on the outer ring coloured by concentration, reactions on the inner ring, the stoichiometry as lines; a flux limiter keeps concentrations non-negative at the step.</div>
'''

FORM_JS = r'''
const NUM=['n_metabolites','n_reactions','max_per_reaction','cycle_fraction','k_min','k_max','c_min','c_max','homeostasis','circadian_amplitude','circadian_period','n_frames','dt','seed','movie_frames','stills'];
window.tabForm=function(){const f={name:$('name').value};for(const k of NUM)f[k]=+$(k).value;return f;};
window.tabFill=function(f){$('name').value=f.name;for(const k of NUM)if(f[k]!==undefined&&f[k]!==null)$(k).value=f[k];};
window.tabInit=function(){};
'''

BRIEF = """You are driving the Plexus metabolism page, a UI for DEFINING a random mass-action metabolic
network (metabolites, reactions, a stoichiometric relation) and running it. A person is watching
the page: it follows every change you make. Say, in ONE short plain-English line before each call,
what you are about to do and why; say what you found after it. No headers, no markdown.

Your ONLY tool is `curl` against the local server at http://127.0.0.1:{port} . The routes:

You have curl, sleep and jq ONLY: no python, no ls, no files. Put the JSON body inline in `curl -d '...'`
(one line, however long); a body you cannot write inline you cannot send.

  POST /api/tab/metabolism/build  JSON form -> writes and seeds a spec. Fields: name,
                          n_metabolites, n_reactions, max_per_reaction, cycle_fraction (0-1, share of
                          autocatalytic reactions), k_min, k_max (rate constants), c_min, c_max
                          (initial concentrations), homeostasis (per unit time), circadian_amplitude,
                          circadian_period (frames), n_frames, dt, seed, movie_frames, stills.
  POST /api/scene/run     {device} -> generate the spec (the same run as Plexus_Main.py -o generate)
  GET  /api/scene/run     -> progress {running, frame, n_frames, seconds}
  POST /api/scene/run     {stop: true}
  GET  /api/scene/state   -> {name, version, ...}: what the page shows now
  GET  /api/scene/view?azim=&elev=&zoom=&message=   -> turns the viewer's camera, shows `message`
"""
