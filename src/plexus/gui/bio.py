"""`plexus.gui.bio` -- define the biological objects of a Plexus scene and look at them, before any run.

    python Plexus_gui.py --bio            # http://127.0.0.1:8765/bio

WHAT IT IS. One page for DEFINING objects, not for making movies. A coarse form on the left says
what the tissue is (shape, how many cells, radius, thickness, which side is apical) and which
proteins ride it (species, region, density, rates, colour); BUILD writes a spec from that and
validates it with `plexus.schema.load`, the gatekeeper the engine trusts. The scene on the right
is the spec BUILT AND SEEDED in-process, exactly as `engine.run` would start it, drawn with
three.js so it can be turned, zoomed and clicked: a click on a cluster, a cell face or a vertex
shows what the object is, what it belongs to and what it carries, and the tree above lists the
sets of the spec with their containment and their relations. The prompt under the form is the
studio's own edit call: an instruction in English is applied by Claude to the spec on screen,
validated the same way, and the scene is re-seeded.

WHAT IT DELIBERATELY DOES NOT DO: render a movie, run frames, or validate on its own. The scene is
`engine.build` + `engine.seed` on the CPU and nothing else, so what is shown is the state a run of
the saved spec starts from; when the picture is right, `Plexus_Main.py -o generate studio/<name>`
runs it as it is.
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import yaml

from plexus.gui import studio

REPO = studio.REPO
CONFIG_DIR = studio.CONFIG_DIR
TEMPLATE = os.path.join(REPO, "config", "tissue", "spheroid_proteins.yaml")

DEFAULT_COLORS = [[0.95, 0.15, 0.15], [0.25, 0.60, 1.00], [0.45, 0.95, 0.55], [1.00, 0.85, 0.30],
                  [0.85, 0.40, 0.95], [0.30, 0.90, 0.95]]
SHAPES = ("sphere", "disc", "plane")

# THE SESSION, SHARED BETWEEN THE PAGE AND WHOEVER DRIVES THE API. The page polls it and applies
# what changed: a new spec version reloads and re-seeds; a new camera version turns the view; a
# pick shows that object's panel. Two counters, so a driver moving the camera does not force a
# re-seed and a re-seed does not snap the viewer's camera.
STATE = {"name": None, "version": 0, "cam_version": 0, "azim": 30.0, "elev": 20.0, "zoom": 1.0,
         "pick": None, "message": ""}


def bump(name: str | None = None, message: str = "") -> dict:
    if name:
        STATE["name"] = name
    STATE["version"] += 1
    STATE["message"] = message
    return dict(STATE)


def set_view(azim=None, elev=None, zoom=None, pick=None, message=None) -> dict:
    if azim is not None: STATE["azim"] = float(azim)
    if elev is not None: STATE["elev"] = float(elev)
    if zoom is not None: STATE["zoom"] = float(zoom)
    if pick is not None: STATE["pick"] = (pick or None)
    if message is not None: STATE["message"] = str(message)
    STATE["cam_version"] += 1
    return dict(STATE)
REGIONS = ("basal", "apical", "mid", "interior")
SCENE_MAX = 60_000
ORG_REGIONS = ("interior", "apical_side", "basal_side")
ORG_COLORS = {"nucleus": [0.55, 0.75, 1.0], "mitochondria": [0.95, 0.35, 0.2], "golgi": [1.0, 0.8, 0.25],
              "centrosome": [0.8, 0.95, 0.3], "lysosome": [0.85, 0.4, 0.95]}


# ---------------------------------------------------------------------------------------------
# the coarse form -> a spec
# ---------------------------------------------------------------------------------------------
def build_spec(form: dict) -> dict:
    """A spec from the coarse form. The mechanics, growth and division are the spheroid_proteins
    template's (the cvd2_adder_tension working point); the form decides geometry, polarity and
    the protein species. Buffers are sized from the cell count so a 40-cell sheet is not carried
    on a 12,800-cell buffer."""
    t = yaml.safe_load(open(TEMPLATE))
    s = copy.deepcopy(t)
    name = str(form.get("name") or "bio_scene").strip()
    n_cells = int(form.get("n_cells", 200))
    radius = float(form.get("radius", 5.0))
    h0 = float(form.get("h0", 0.88))
    shape = str(form.get("shape", "sphere")).lower()
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    apical = str(form.get("apical", "in")).lower()
    if apical not in ("in", "out"):
        raise ValueError("apical must be in|out")
    world = float(form.get("world", 50.0))
    frames = int(form.get("n_frames", 801))
    species = form.get("species") or []
    organelles = form.get("organelles") or []
    if not species and not organelles:
        raise ValueError("declare at least one protein species or one organelle")

    s["general"].update(name=name, n_frames=frames, record_cap=frames + 2, world=[world] * 3)
    cell_n = max(512, 8 * n_cells)
    vert_n = max(2048, 40 * n_cells)
    s["sets"]["cell"]["n"] = cell_n
    s["sets"]["vertex"]["n"] = vert_n
    s["sets"]["half_edge"]["n"] = max(8192, 8 * cell_n)
    ns = len(species)
    types = {}
    colors = {}
    dot_radius = {}
    if not species:                                          # a tissue with organelles only: no protein set
        s["sets"].pop("protein", None)
        for k in ("protein_s", "protein_tau", "n_protein"):
            s["sets"]["cell"]["state"].pop(k, None)
        s["operators"] = [o for o in s["operators"] if o.get("at") != "protein"]
        s["schedule"] = [x for x in s["schedule"] if x not in ("radius_graph", "attraction_repulsion", "protein_project", "protein_express")]
        s["seed"] = [o for o in s["seed"] if o.get("op") != "protein_seed"]
    else:
        for k in ("protein_s", "protein_tau", "n_protein"):
            s["sets"]["cell"]["state"][k] = {"width": ns}
    for i, sp in enumerate(species):
        nm = str(sp.get("name") or f"species_{i}").strip()
        region = str(sp.get("region", "basal")).lower()
        if region not in REGIONS:
            raise ValueError(f"species {nm!r}: region must be one of {REGIONS}")
        # the fractions must sum to exactly 1.0 for the schema; the last species takes the remainder
        frac = round(1.0 / ns, 6) if i < ns - 1 else round(1.0 - round(1.0 / ns, 6) * (ns - 1), 6)
        types[nm] = {"fraction": frac, "region": region,
                     "density": float(sp.get("density", 3.0)), "s": float(sp.get("s", 0.02)),
                     "tau": float(sp.get("tau", 300.0)), "p": [0.0, 1.0, 0.6, 0.35]}
        colors[nm] = [float(v) for v in (sp.get("color") or DEFAULT_COLORS[i % len(DEFAULT_COLORS)])]
        dot_radius[nm] = float(sp.get("radius", 0.05))
    if species:
        dens = max(float(t_["density"]) for t_ in types.values())
        s["sets"]["protein"]["types"] = types
        s["sets"]["protein"]["per_parent"] = int(max(6, round(2.0 * dens * ns)))
        s["sets"]["protein"].pop("grow_reserve", None)
    # organelles: one contained set of pieces with a radius, species as types with a count per cell
    otypes = {}
    for i, og in enumerate(organelles):
        nm = str(og.get("name") or f"organelle_{i}").strip()
        region = str(og.get("region", "interior")).lower()
        if region not in ORG_REGIONS:
            raise ValueError(f"organelle {nm!r}: region must be one of {ORG_REGIONS}")
        rule = str(og.get("on_divide", "duplicate")).lower()
        if rule not in ("duplicate", "halve", "none"):
            raise ValueError(f"organelle {nm!r}: on_divide must be duplicate|halve|none")
        t_ = {"count": int(og.get("count", 1)), "radius": float(og.get("radius", 0.3)), "region": region, "on_divide": rule}
        if og.get("tau") not in (None, "", 0):
            t_["tau"] = float(og["tau"])
        otypes[nm] = t_
        colors[nm] = [float(v) for v in (og.get("color") or ORG_COLORS.get(nm) or DEFAULT_COLORS[(ns + i) % len(DEFAULT_COLORS)])]
        dot_radius[nm] = t_["radius"]
    if otypes:
        s["sets"]["organelle"] = {"entity": "organelle", "parent": "cell", "parent_pos": "centroid",
                                  "per_parent": sum(t_["count"] for t_ in otypes.values()),
                                  "state": {"pos": {"width": 3}, "vel": {"width": 3}, "age": {"width": 1}},
                                  "types": otypes}
        s["sets"]["cell"]["state"]["n_organelle"] = {"width": len(otypes)}
        s["operators"].append({"op": "organelle_project", "at": "organelle", "tissue": "vertex", "seed": 0})
        s["schedule"].append("organelle_project")
        if any("tau" in t_ for t_ in otypes.values()):
            s["operators"].append({"op": "organelle_express", "at": "organelle", "tissue": "vertex", "seed": 0})
            s["schedule"].append("organelle_express")
        s["seed"].append({"op": "organelle_seed", "at": "organelle", "tissue": "vertex", "seed": 0})
    for o in s["seed"]:
        if o["op"] in ("seed_mesh", "mesh_seed"):
            o.update(n_cells=n_cells, radius=radius, h0=h0, apical=apical, shape=shape, cell_set="cell")
            if shape == "sphere":
                o.pop("width", None)
    for o in s["operators"]:
        if o["op"] == "cell_mechanics":
            o["surface"] = "basal" if apical == "in" else "apical"
    p = s["plotting"]
    p["colors"] = colors
    p["dot_radius"] = dot_radius
    p["mesh_surface"] = "basal" if apical == "in" else "apical"
    p["mesh_opacity"] = 0.3
    cs = p.get("cross_section") or {}
    cs.update(points=False, spheres=True, span=max(6.0, 3.6 * radius)); p["cross_section"] = cs
    p["cross_section_height"] = 0.34
    p["curve"] = [{"quantity": "cells", "xlabel": "frame", "ylabel": "cells", "ticks": 4, "ymin": 0, "ymax": max(1000, 5 * n_cells)}]
    for nm in types:
        p["curve"].append({"quantity": f"count:protein:{nm}", "xlabel": "frame", "ylabel": nm, "ticks": 4,
                           "ymin": 0, "ymax": 10000})
    for nm, t_ in otypes.items():
        p["curve"].append({"quantity": f"count:organelle:{nm}", "xlabel": "frame", "ylabel": nm, "ticks": 4,
                           "ymin": 0, "ymax": max(1000, 5 * n_cells * t_["count"])})
    p["curve"] = p["curve"][:3]
    return s


def normalise(spec: dict) -> dict:
    """The consequences of the species list, derived rather than typed: the cell's per-species
    blocks (`protein_s`, `protein_tau`, `n_protein`) are as wide as there are species, every
    species has a fraction and the fractions sum to 1, and every species has a colour. A prompt
    edit that adds a species and leaves a width at 2 would seed-fail; this is the mechanical part
    Claude should not have to remember."""
    sets = spec.get("sets") or {}
    prot = sets.get("protein") or {}
    types = prot.get("types") or {}
    ns = len(types)
    if ns:
        cst = (sets.get("cell") or {}).setdefault("state", {})
        for k in ("protein_s", "protein_tau", "n_protein"):
            cst[k] = {"width": ns}
        for i, (nm, t_) in enumerate(types.items()):
            t_ = t_ or {}
            t_["fraction"] = round(1.0 / ns, 6) if i < ns - 1 else round(1.0 - round(1.0 / ns, 6) * (ns - 1), 6)
            t_.setdefault("p", [0.0, 1.0, 0.6, 0.35])
            types[nm] = t_
        colors = (spec.setdefault("plotting", {}) or {}).setdefault("colors", {})
        for i, nm in enumerate(types):
            colors.setdefault(nm, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    org = sets.get("organelle") or {}
    otypes = org.get("types") or {}
    if otypes:
        org["per_parent"] = sum(int((t_ or {}).get("count", 1)) for t_ in otypes.values())
        org.setdefault("entity", "organelle"); org.setdefault("parent", "cell"); org.setdefault("parent_pos", "centroid")
        org.setdefault("state", {"pos": {"width": 3}, "vel": {"width": 3}, "age": {"width": 1}})
        (sets.get("cell") or {}).setdefault("state", {})["n_organelle"] = {"width": len(otypes)}
        pl = spec.setdefault("plotting", {}) or {}
        colors = pl.setdefault("colors", {}); rad = pl.setdefault("dot_radius", {})
        for i, (nm, t_) in enumerate(otypes.items()):
            t_ = t_ or {}
            t_.setdefault("count", 1); t_.setdefault("radius", 0.3); t_.setdefault("region", "interior"); t_.setdefault("on_divide", "duplicate")
            otypes[nm] = t_
            colors.setdefault(nm, ORG_COLORS.get(nm, DEFAULT_COLORS[(ns + i) % len(DEFAULT_COLORS)]))
            rad[nm] = float(t_["radius"])
        ops = [o.get("op") for o in spec.get("operators") or []]
        sched = spec.setdefault("schedule", [])
        if "organelle_project" not in ops:
            spec.setdefault("operators", []).append({"op": "organelle_project", "at": "organelle", "tissue": "vertex", "seed": 0})
        if "organelle_project" not in sched:
            sched.append("organelle_project")
        if any("tau" in t_ for t_ in otypes.values()) and "organelle_express" not in ops:
            spec["operators"].append({"op": "organelle_express", "at": "organelle", "tissue": "vertex", "seed": 0})
            if "organelle_express" not in sched:
                sched.append("organelle_express")
        if not any(o.get("op") == "organelle_seed" for o in spec.get("seed") or []):
            spec.setdefault("seed", []).append({"op": "organelle_seed", "at": "organelle", "tissue": "vertex", "seed": 0})
    return spec


def form_from_spec(spec: dict) -> dict:
    """The coarse form read back off a spec, so a spec edited by the prompt refills the form."""
    seed = next((o for o in spec.get("seed", []) if o.get("op") in ("seed_mesh", "mesh_seed")), {})
    types = (spec.get("sets", {}).get("protein", {}) or {}).get("types", {}) or {}
    colors = (spec.get("plotting", {}) or {}).get("colors", {}) or {}
    return {
        "name": spec.get("general", {}).get("name", ""),
        "shape": seed.get("shape", "sphere"), "n_cells": seed.get("n_cells", 200),
        "radius": seed.get("radius", 5.0), "h0": seed.get("h0", 0.88), "apical": seed.get("apical", "out"),
        "world": (spec.get("general", {}).get("world") or [50.0])[0],
        "n_frames": spec.get("general", {}).get("n_frames", 801),
        "species": [{"name": n, "region": t_.get("region", "basal"), "density": t_.get("density", 3.0),
                     "s": t_.get("s", 0.02), "tau": t_.get("tau", 300.0), "color": colors.get(n)}
                    for n, t_ in types.items()],
        "organelles": [{"name": n, "count": t_.get("count", 1), "radius": t_.get("radius", 0.3),
                        "region": t_.get("region", "interior"), "on_divide": t_.get("on_divide", "duplicate"),
                        "tau": t_.get("tau"), "color": colors.get(n)}
                       for n, t_ in ((spec.get("sets", {}).get("organelle", {}) or {}).get("types", {}) or {}).items()],
    }


# ---------------------------------------------------------------------------------------------
# the spec, built and seeded -> the scene
# ---------------------------------------------------------------------------------------------
def seed_scene(spec_path: str, device: str = "cpu") -> dict:
    """Build and seed the spec exactly as a run would, and return every live object: positions per
    set, species and parent per cluster, the tissue's caps as triangles with their cell ids, the
    cell set's scalar blocks, and the hierarchy (containment and relations) read off the spec."""
    from plexus import schema, engine
    sim = schema.load(spec_path)
    H = engine.build(sim, device)
    engine.seed(H, sim, device)
    return scene_from(H, sim, spec_path)


def scene_from(H, sim, spec_path: str) -> dict:
    """The scene dict off an already built and seeded hierarchy (the view keeps H for rendering)."""
    t0 = time.time()
    spec = yaml.safe_load(open(spec_path))
    out = {"name": sim.name, "world": [float(v) for v in sim.world_size], "sets": {}, "tissue": None,
           "hierarchy": hierarchy(spec), "colors": (spec.get("plotting") or {}).get("colors") or {}}

    def _np(v):
        return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

    # AT SEED TIME A CELL IS LIVE IF THE MESH HAS ITS FACE: the cell set's occupancy buffer is only
    # maintained once the population operators run, so the mesh table's `nF` is the count here.
    _nF_of = {}
    for name, lvl in H.levels.items():
        m = getattr(lvl, "_mesh", None)
        if m is not None and int(m.get("Nv", 0) or 0):
            _nF_of[getattr(lvl, "mesh_cell_set", None) or "cell"] = int(m["nF"])
    for name, lvl in H.levels.items():
        # the live mask the recorder uses: `active` (occupancy AND the mesh's own liveness) when the
        # level has it, `occ` otherwise -- the cell set's occupancy buffer reads every slot live
        _mask = getattr(lvl, "active", None)
        if _mask is None:
            _mask = getattr(lvl, "occ", None)
        occ = _np(_mask).astype(bool) if _mask is not None else None
        if name in _nF_of and occ is not None:
            occ = np.zeros_like(occ); occ[: _nF_of[name]] = True
        entry = {"n_buffer": int(lvl.n), "n_live": int(occ.sum()) if occ is not None else int(lvl.n),
                 "parent_name": getattr(lvl, "parent_name", None),
                 "blocks": list(getattr(lvl.state_schema, "blocks", []) and [b.name for b in lvl.state_schema.blocks]),
                 "type_names": list(getattr(lvl, "type_names", []) or [])}
        sd = (spec.get("sets") or {}).get(name) or {}
        entry["maps"] = sd.get("maps")
        entry["mesh"] = sd.get("mesh")
        entry["entity"] = sd.get("entity")
        if "pos" in lvl.state_schema:
            live = np.nonzero(occ)[0] if occ is not None else np.arange(lvl.n)
            # A MILLION MATERIAL POINTS DO NOT GO INTO A JSON. The scene dict serves picking and
            # the info panel; past SCENE_MAX live rows a set is subsampled by a stride, `idx`
            # keeping the buffer index of every row kept so a pick still names the real row.
            if len(live) > SCENE_MAX:
                live = live[:: len(live) // SCENE_MAX + 1]
                entry["subsampled"] = True
            P = _np(lvl.get("pos"))[live]
            entry["idx"] = live.tolist()
            entry["pos"] = np.round(P, 4).tolist()
            nt = getattr(lvl, "node_type", None)
            if nt is not None and int(np.asarray(_np(nt)).shape[0]) == int(lvl.n):
                entry["node_type"] = _np(nt)[live].astype(int).tolist()
            par = getattr(lvl, "parent", None)
            if par is not None and int(np.asarray(_np(par)).shape[0]) == int(lvl.n):
                entry["parent"] = _np(par)[live].astype(int).tolist()
        # scalar blocks of a non-spatial set (the cell set): one row per live element
        elif occ is not None:
            live = np.nonzero(occ)[0]
            entry["idx"] = live.tolist()
            entry["rows"] = {b.name: np.round(_np(lvl.get(b.name))[live], 4).tolist()
                             for b in lvl.state_schema.blocks if b.width <= 8}
        out["sets"][name] = entry
        # the tissue: caps as triangle fans, one fan per cell
        m = getattr(lvl, "_mesh", None)
        if m is not None and int(m.get("Nv", 0) or 0):
            Nv, nF = int(m["Nv"]), int(m["nF"])
            pos = _np(lvl.get("pos"))[:Nv]
            sep = _np(lvl.get("sep"))[:Nv] if "sep" in lvl.state_schema else np.zeros_like(pos)
            es, et, ef = (_np(m[k]).astype(int) for k in ("E_srce", "E_trgt", "E_face"))
            keep = (ef >= 0) & (ef < nF)
            es, et, ef = es[keep], et[keep], ef[keep]
            cnt = np.bincount(ef, minlength=nF).clip(1)
            caps = {}
            for cap, k in (("apical", 1.0), ("basal", -1.0), ("mid", 0.0)):
                X = pos + k * sep
                cen = np.zeros((nF, 3)); np.add.at(cen, ef, X[es]); cen /= cnt[:, None]
                verts = np.concatenate([X, cen], 0)                   # ring vertices then centroids
                tri = np.stack([Nv + ef, es, et], 1)                  # (centroid, source, target)
                caps[cap] = {"verts": np.round(verts, 4).tolist(), "tri": tri.tolist(), "face": ef.tolist()}
            # which cells share each vertex, for the vertex info panel
            cells_of_vertex = [[] for _ in range(Nv)]
            for a, f in zip(es.tolist(), ef.tolist()):
                if f not in cells_of_vertex[a]:
                    cells_of_vertex[a].append(f)
            out["tissue"] = {"set": name, "cell_set": getattr(lvl, "mesh_cell_set", None) or "cell",
                             "Nv": Nv, "nF": nF, "caps": caps, "cells_of_vertex": cells_of_vertex,
                             "apical": next((o.get("apical", "out") for o in spec.get("seed", [])
                                             if o.get("op") in ("seed_mesh", "mesh_seed")), "out")}
    out["seconds"] = round(time.time() - t0, 2)
    return out


def hierarchy(spec: dict) -> dict:
    """The sets and how they hang together, read off the spec: containment (`parent`), relations
    (`maps`), and which set carries a mesh."""
    sets = spec.get("sets") or {}
    nodes = []
    for name, sd in sets.items():
        sd = sd or {}
        nodes.append({"name": name, "n": sd.get("n"), "parent": sd.get("parent"), "maps": sd.get("maps"),
                      "mesh": sd.get("mesh"), "entity": sd.get("entity"), "per_parent": sd.get("per_parent"),
                      "types": list((sd.get("types") or {}).keys()), "blocks": list((sd.get("state") or {}).keys())})
    ops = [{"op": o.get("op"), "at": o.get("at"), "region": None} for o in (spec.get("operators") or [])]
    return {"sets": nodes, "operators": ops, "schedule": spec.get("schedule") or [], "seed": spec.get("seed") or []}


# ---------------------------------------------------------------------------------------------
# a pick resolved by id (the picture itself is the movie renderer's: gui/bio_view.py)
# ---------------------------------------------------------------------------------------------
def resolve_pick(scene: dict, pick: str) -> dict | None:
    """`<set>:<live index>` (a cluster or a vertex), `cell:<face id>` -- the same facts the page's
    info panel shows, as a dict."""
    try:
        sname, idx = pick.split(":", 1); idx = int(idx)
    except ValueError:
        return None
    T = scene.get("tissue")
    out = {"pick": pick}
    if sname == "cell" or (T and sname == T["cell_set"]):
        cs = T["cell_set"] if T else "cell"
        c = scene["sets"].get(cs, {})
        out.update(kind="cell", cell=idx, blocks={})
        if c.get("rows") and idx in c.get("idx", []):
            k = c["idx"].index(idx)
            out["blocks"] = {b: v[k] for b, v in c["rows"].items()}
        if T:
            verts = [i for i, cl in enumerate(T["cells_of_vertex"]) if idx in cl]
            out["vertices"] = verts
            V = np.asarray(T["caps"]["mid"]["verts"], float)
            out["position"] = V[T["Nv"] + idx].round(3).tolist() if len(V) > T["Nv"] + idx else None
        for name, st in scene["sets"].items():
            if st.get("parent") is None:
                continue
            per = {}
            for i, par in enumerate(st["parent"]):
                if par == idx:
                    sp = (st.get("type_names") or [name])[(st.get("node_type") or [0] * len(st["parent"]))[i]]
                    per[sp] = per.get(sp, 0) + 1
            if per:
                out.setdefault("contains", {})[name] = per
        return out
    st = scene["sets"].get(sname)
    if st is None or not st.get("pos") or idx >= len(st["pos"]):
        if T and sname == T["set"] and idx < T["Nv"]:
            out.update(kind="vertex", vertex=idx, position=T["caps"]["mid"]["verts"][idx],
                       shared_by_cells=T["cells_of_vertex"][idx])
            return out
        return None
    if T and sname == T["set"]:
        out.update(kind="vertex", vertex=idx, position=st["pos"][idx], shared_by_cells=T["cells_of_vertex"][idx])
        return out
    sp = (st.get("type_names") or [sname])[(st.get("node_type") or [0] * len(st["pos"]))[idx]]
    par = (st.get("parent") or [None])[idx] if st.get("parent") else None
    out.update(kind="cluster", set=sname, species=sp, index=st.get("idx", [idx])[idx], position=st["pos"][idx], parent_cell=par)
    if par is not None:
        out["cell"] = resolve_pick(scene, f"cell:{par}")
    return out


# ---------------------------------------------------------------------------------------------
# CLAUDE TAKES OVER THE PAGE: the CLI, with the page's own routes as its only tool
# ---------------------------------------------------------------------------------------------
CLAUDE: dict = {"running": False, "task": "", "lines": [], "seconds": 0.0, "error": None, "started": 0.0,
                "session": None, "turns": 0, "notes": []}


def claude_note(text: str) -> None:
    """Something the page did without Claude (a spec opened, built, refined or seeded): queued and
    handed to the session at its next task, so it never works from a stale picture of the scene."""
    CLAUDE["notes"].append(f"{time.strftime('%H:%M:%S')} {text}")
    CLAUDE["notes"] = CLAUDE["notes"][-20:]


def claude_new_session() -> dict:
    CLAUDE.update(session=None, turns=0, notes=[])
    CLAUDE["lines"].append("[new session]")
    return {"session": None}

BIO_BRIEF = """You are driving the Plexus bio-objects page, a UI for DEFINING biological objects
(an epithelial tissue and the protein clusters on it) before any simulation. A person is watching
the page: it follows every change you make through the session state, and your words appear in a
transcript beside the 3D scene. Say, in ONE short plain-English line before each call, what you
are about to do and why; say what you found after it. No headers, no markdown.

Your ONLY tool is `curl` against the local server at http://127.0.0.1:{port} . The routes:

  POST /api/bio/build     JSON form -> writes and seeds a spec. Fields: name, shape (sphere|disc|plane),
                          n_cells, radius, h0 (cell thickness), apical (in|out; `in` puts the basal
                          cap outside, where a matrix would be), world (box), n_frames,
                          species: [{name, region (basal|apical|mid|interior), density, s, tau}]
                          (may be empty), organelles: [...] (see below; may be empty)
  POST /api/bio/refine    {name, prompt} -> an English edit of the current spec (another Claude
                          applies it; 20-40 s); use it for anything the form cannot say
  GET  /api/bio/counts?name= -> live count per set, per species, and per cell per species
                          (use this for any counting; it is small)
  GET  /api/bio/seed?name=  -> the whole seeded scene as JSON (large; pipe through jq)
  GET  /api/bio/info?name=&pick=<set>:<index>   -> what one object is (protein:12, cell:7, vertex:5):
                          species, parent cell, what the cell contains, blocks, shared cells
  GET  /api/bio/view?azim=&elev=&zoom=&pick=&message=   -> turns the viewer's camera (degrees,
                          zoom 0.05-60), highlights a pick, and shows `message` on the page. Move in
                          steps of 30 degrees or less with `sleep 1` between them so the viewer can
                          follow; zoom no faster than x1.5 per step.
  POST /api/bio/visible   {species, on} -> hide or show one species in the picture
  POST /api/bio/run       {frames, device} -> simulate the spec from its seed, the picture following
                          each frame; {stop: true} aborts.  GET /api/bio/run -> progress and the
                          live counts per set and species (poll it with sleep 2 between calls)
  GET  /api/bio/open?path=<spec.yaml or run folder> -> import an existing spec into the session
  GET  /api/bio/state     -> the session state
The build form also takes organelles: [{name, count (per cell), radius, region
(interior|apical_side|basal_side), on_divide (duplicate|halve|none), tau (optional, frames to
recover the count after a division)}]; the picture is the movie renderer's, spheres at their radius.

Rules: use only these routes (curl -s, JSON bodies with -H 'Content-Type: application/json'), with
jq and sleep as the only other commands;
never touch files; keep the spec name given in the task unless told otherwise; finish with two
or three lines summarising what the scene now is and the counts per species.
"""


def counts(scene: dict) -> dict:
    """Small numbers for a driver: live count per set, per species, and per (cell, species)."""
    out = {"sets": {k: v.get("n_live") for k, v in scene["sets"].items()}, "species": {}, "per_cell": {}}
    for name, s in scene["sets"].items():
        if not s.get("parent"):
            continue
        names = s.get("type_names") or [name]
        for i, par in enumerate(s["parent"]):
            sp = names[(s.get("node_type") or [0] * len(s["parent"]))[i]] if names else name
            out["species"][sp] = out["species"].get(sp, 0) + 1
            out["per_cell"].setdefault(str(par), {})
            out["per_cell"][str(par)][sp] = out["per_cell"][str(par)].get(sp, 0) + 1
    return out


def _ev_lines(ev: dict) -> list:
    """One transcript line per event of interest: Claude's words, its calls, their results."""
    out = []
    t = ev.get("type")
    if t == "assistant":
        for c in (ev.get("message") or {}).get("content") or []:
            if c.get("type") == "text" and c.get("text", "").strip():
                out.append("Claude: " + c["text"].strip())
            elif c.get("type") == "tool_use":
                cmd = (c.get("input") or {}).get("command") or json.dumps(c.get("input") or {})[:200]
                out.append("$ " + str(cmd).strip()[:400])
    elif t == "user":
        for c in (ev.get("message") or {}).get("content") or []:
            if c.get("type") == "tool_result":
                body = c.get("content")
                if isinstance(body, list):
                    body = " ".join(str(b.get("text", "")) for b in body if isinstance(b, dict))
                body = str(body or "").strip().replace("\n", " ")
                out.append("  -> " + (body[:300] + ("..." if len(body) > 300 else "")))
    elif t == "result":
        out.append(f"[done in {float(ev.get('duration_ms', 0)) / 1000:.0f}s, {ev.get('num_turns', '?')} turns]")
    return out


def claude_start(task: str, port: int, model: str = "sonnet", timeout: int = 900, brief: str | None = None) -> dict:
    """Launch the CLI on the task in a thread; the transcript fills `CLAUDE['lines']` as it runs."""
    import subprocess
    import threading
    if CLAUDE["running"]:
        return {"error": "Claude is already driving; STOP it first"}
    # ONE SESSION ACROSS PRESSES. The first task opens a session under a fresh id; every later
    # task resumes it, so Claude keeps what it learned about the scene and skips re-reading it
    # (faster, and it can refer to "the nucleus you added"). Anything the page did meanwhile is
    # prepended as notes. "New session" on the page starts over.
    import uuid
    fresh = CLAUDE["session"] is None
    if fresh:
        CLAUDE["session"] = str(uuid.uuid4())
    notes = CLAUDE["notes"]; CLAUDE["notes"] = []
    prompt = task
    if notes:
        prompt = "Since your last task, the page did this without you:\n- " + "\n- ".join(notes) + \
                 "\nRe-read /api/bio/state before assuming anything about the scene.\n\nTask: " + task
    CLAUDE.update(running=True, task=task, lines=[f"task: {task}" + ("" if fresh else f"  (session turn {CLAUDE['turns'] + 1})")],
                  seconds=0.0, error=None, started=time.time())

    def _go():
        sess = ["--session-id", CLAUDE["session"]] if fresh else ["--resume", CLAUDE["session"]]
        cmd = [studio._claude_bin(), "-p", prompt, *sess,
               "--append-system-prompt", (brief or BIO_BRIEF).replace("{port}", str(port)),
               "--allowedTools", "Bash(curl:*)", "Bash(sleep:*)", "Bash(jq:*)",
               "--disallowedTools", "Write", "Edit", "NotebookEdit", "Read", "Glob", "Grep", "WebFetch", "WebSearch", "Task",
               "--model", model, "--effort", "low",
               "--output-format", "stream-json", "--verbose"]
        t0 = time.time()
        try:
            pr = subprocess.Popen(cmd, cwd=studio.REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            CLAUDE["proc"] = pr
            for line in pr.stdout:                                   # type: ignore[union-attr]
                if time.time() - t0 > timeout:
                    pr.kill(); CLAUDE["error"] = f"timed out after {timeout}s"; break
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except Exception:                                    # noqa: BLE001
                    continue
                CLAUDE["lines"].extend(_ev_lines(ev))
            pr.wait(timeout=5)
            err = (pr.stderr.read() or "").strip() if pr.stderr else ""
            if pr.returncode not in (0, None) and err:
                CLAUDE["error"] = err[-800:]
        except Exception as e:                                       # noqa: BLE001
            CLAUDE["error"] = f"{type(e).__name__}: {e}"
        finally:
            CLAUDE["seconds"] = round(time.time() - t0, 1)
            CLAUDE["running"] = False
            CLAUDE["turns"] += 1
            if CLAUDE["error"] and "resume" in (CLAUDE["error"] or "").lower():
                CLAUDE["session"] = None                    # a session the CLI cannot find: start over next time
            CLAUDE.pop("proc", None)
    threading.Thread(target=_go, daemon=True).start()
    return {"started": True}


def claude_stop() -> dict:
    pr = CLAUDE.get("proc")
    if pr is not None:
        try:
            pr.kill()
        except Exception:                                            # noqa: BLE001
            pass
    CLAUDE["lines"].append("[stopped by the viewer]")
    return {"stopped": True}


# ---------------------------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Plexus bio objects</title>
<style>
 html,body{margin:0;height:100%;background:#0b0b0d;color:#ddd;font:13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
 #left{position:absolute;left:0;top:0;bottom:0;width:400px;overflow:auto;background:#141418;border-right:1px solid #2a2a30;padding:10px 12px;box-sizing:border-box}
 #right{position:absolute;left:400px;top:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden}
 #view{max-width:100%;max-height:100%;cursor:grab;user-select:none;-webkit-user-drag:none}
 h1{font-size:15px;margin:2px 0 8px;color:#fff} h2{font-size:12px;margin:14px 0 4px;color:#9ab;letter-spacing:.06em;text-transform:uppercase}
 label{display:inline-block;width:92px;color:#aab} input,select{background:#0e0e12;color:#eee;border:1px solid #333;border-radius:3px;padding:2px 5px;width:110px;margin:1px 0}
 input.short{width:56px} select{width:118px}
 button{background:#2b5f9e;color:#fff;border:0;border-radius:3px;padding:5px 10px;margin:3px 3px 3px 0;cursor:pointer} button.dim{background:#3a3a44}
 button:disabled{opacity:.5;cursor:default}
 table.sp{border-collapse:collapse;width:100%} table.sp td{padding:1px 2px} table.sp th{font-weight:normal;color:#889;font-size:11px} table.sp input,table.sp select{width:100%;box-sizing:border-box}
 #status{color:#8c8;min-height:16px;margin:4px 0;white-space:pre-wrap;font-size:12px} #status.err{color:#f88}
 #tree div{padding:1px 0 1px 8px;cursor:pointer} #tree div:hover{color:#fff} #tree .n{color:#7fb3ff} #tree .rel{color:#9ac} #tree .cont{color:#c9a}
 #info{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:12px;white-space:pre-wrap;min-height:60px;max-height:260px;overflow:auto}
 textarea{width:100%;height:120px;background:#0e0e12;color:#ddd;border:1px solid #333;font:11px monospace;box-sizing:border-box}
 #yaml{display:none} .row{margin:2px 0}
 #hint{position:absolute;right:12px;top:8px;color:#778;font-size:12px}
 #vis label{width:auto;color:#ccd;margin-right:10px;cursor:pointer} #vis input[type=checkbox]{width:auto;margin:0 3px 0 0}
 button.claude{background:#000;border:1px solid #555;display:inline-flex;align-items:center;gap:6px} button.claude.on{background:#1f8f3f;border-color:#2fbf5f}
 button.claude svg{width:14px;height:14px;fill:#d97757} button.claude.on svg{fill:#fff}
 #claude{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:11px;white-space:pre-wrap;height:190px;overflow:auto;margin:4px 0}
</style></head><body>
<div id="left">
 <h1>Plexus bio objects</h1>
 <h2>Open a spec</h2>
 <div class="row"><button class="dim" onclick="pickOpen()">OPEN...</button> <span id="openlab" style="color:#778;font-size:11px">a spec.yaml or a run folder; copied into config/studio and seeded as is</span></div>
 <div id="picker" style="display:none;position:fixed;left:60px;top:40px;width:560px;max-height:80vh;background:#1a1a20;border:1px solid #556;border-radius:6px;padding:10px;z-index:10;box-shadow:0 0 30px #000">
  <div class="row"><b>Open a spec</b> <span style="float:right;cursor:pointer" onclick="$('picker').style.display='none'">&#10005;</span></div>
  <div class="row" id="pickroots"></div>
  <div class="row" id="pickpath" style="color:#9ab;font-family:monospace;font-size:11px;word-break:break-all"></div>
  <div id="picklist" style="max-height:55vh;overflow:auto;background:#0e0e12;border:1px solid #2a2a30;padding:4px;font-size:12px"></div>
 </div>
 <h2>Tissue</h2>
 <div class="row"><label>name</label><input id="name" value="bio_scene"></div>
 <div class="row"><label>shape</label><select id="shape"><option>sphere</option><option>disc</option><option>plane</option></select></div>
 <div class="row"><label>cells</label><input id="n_cells" class="short" value="200"> <label style="width:60px">radius</label><input id="radius" class="short" value="5.0"></div>
 <div class="row"><label>thickness h0</label><input id="h0" class="short" value="1.2"> <label style="width:60px">apical</label><select id="apical" style="width:64px"><option value="in">in</option><option value="out">out</option></select></div>
 <div class="row"><label>box</label><input id="world" class="short" value="50"> <label style="width:60px">frames</label><input id="n_frames" class="short" value="801"></div>
 <h2>Organelles <button class="dim" onclick="addOrganelle()">+ organelle</button></h2>
 <table class="sp" id="organelles"><tr><th>name</th><th>per cell</th><th>radius</th><th>region</th><th>on divide</th><th>tau</th><th></th></tr></table>
 <h2>Proteins <button class="dim" onclick="addSpecies()">+ species</button></h2>
 <table class="sp" id="species"><tr><th>name</th><th>region</th><th>density</th><th>s</th><th>tau</th><th></th></tr></table>
 <div class="row"><button onclick="build()">BUILD + SEED</button><button class="dim" onclick="toggleYaml()">YAML</button><button class="dim" onclick="reseed()">RE-SEED</button></div>
 <div id="status">form a scene, then BUILD</div>
 <h2>Run the engine</h2>
 <div class="row"><label>frames</label><input id="run_frames" class="short" value="200"> <label style="width:60px">device</label><select id="run_device" style="width:80px"><option>cuda:0</option><option>cuda:1</option><option>cpu</option></select> <label style="width:70px" title="frames kept for PLAY, spread over the run">keep</label><input id="run_keep" class="short" value="20"></div>
 <div class="row"><button onclick="runGo()" id="runbtn">RUN</button><button class="dim" onclick="runStop()">STOP</button> <span id="runstat" style="color:#8c8"></span></div>
 <div id="runcounts" style="color:#9ab;font-size:12px;min-height:14px"></div>
 <div class="row"><button class="dim" onclick="playGo()" id="playbtn">PLAY</button><button class="dim" onclick="playStop()">PAUSE</button> <input type="range" id="frame" min="0" max="0" value="0" style="width:170px" oninput="showFrame(+this.value)"> <span id="framelab" style="color:#9ab"></span></div>
 <h2>Claude takes over <span style="color:#778;font-weight:normal;text-transform:none">drives this page through its own routes</span></h2>
 <div class="row"><input id="task" style="width:100%" placeholder="e.g. build a 120-cell cyst with one nucleus per cell and integrins outside, then show me one cell" onkeydown="if(event.key==='Enter')claudeGo()"></div>
 <div class="row"><button onclick="claudeGo()" id="cbtn" class="claude"><svg viewBox="0 0 24 24"><path d="M12 1.5l1.6 6.4 5.6-3.6-3.6 5.6 6.4 1.6-6.4 1.6 3.6 5.6-5.6-3.6L12 22.5l-1.6-6.4-5.6 3.6 3.6-5.6L1.5 12l6.9-1.6-3.6-5.6 5.6 3.6z"/></svg>CLAUDE</button><button class="dim" onclick="claudeStop()">STOP</button><button class="dim" onclick="claudeNew()" title="forget the conversation so far">NEW SESSION</button> <span id="cstat" style="color:#8c8"></span></div>
 <pre id="claude"></pre>
 <div id="rstat" style="color:#9ab;min-height:14px"></div>
 <div id="yaml"><textarea id="yamltext"></textarea><div><button onclick="saveYaml()">SAVE YAML</button></div></div>
 <h2>Visibility</h2><div id="vis">(seed a scene first)</div>
 <h2>Hierarchy</h2><div id="tree">(none)</div>
 <h2>Selected object</h2><div id="info">click a piece, a cluster, a cell or a vertex</div>
</div>
<div id="right"><img id="view" draggable="false"><div id="hint">drag to orbit, wheel to zoom, click to select -- rendered by the movie renderer</div></div>
<script>
const $=id=>document.getElementById(id);
let SCENE=null, specName=null;
const CAM={azim:30,elev:20,zoom:1};
function status(t,err){const s=$('status');s.textContent=t;s.className=err?'err':'';}
window.addSpecies=function(sp){sp=sp||{};const tb=$('species');const tr=tb.insertRow(-1);
 tr.innerHTML=`<td><input value="${sp.name||''}"></td><td><select><option>basal</option><option>apical</option><option>mid</option><option>interior</option></select></td><td><input value="${sp.density??3}"></td><td><input value="${sp.s??0.02}"></td><td><input value="${sp.tau??300}"></td><td><button class="dim" onclick="this.closest('tr').remove()">x</button></td>`;
 tr.cells[1].firstChild.value=sp.region||'basal';};
window.addOrganelle=function(og){og=og||{};const tb=$('organelles');const tr=tb.insertRow(-1);
 tr.innerHTML=`<td><input value="${og.name||''}"></td><td><input value="${og.count??1}"></td><td><input value="${og.radius??0.3}"></td><td><select><option>interior</option><option>apical_side</option><option>basal_side</option></select></td><td><select><option>duplicate</option><option>halve</option><option>none</option></select></td><td><input value="${og.tau??''}" placeholder="-"></td><td><button class="dim" onclick="this.closest('tr').remove()">x</button></td>`;
 tr.cells[3].firstChild.value=og.region||'interior';tr.cells[4].firstChild.value=og.on_divide||'duplicate';};
function species(){const out=[];for(const tr of $('species').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 out.push({name,region:c[1].firstChild.value,density:+c[2].firstChild.value,s:+c[3].firstChild.value,tau:+c[4].firstChild.value});}return out;}
function organelles(){const out=[];for(const tr of $('organelles').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 const tau=c[5].firstChild.value.trim();out.push({name,count:+c[1].firstChild.value,radius:+c[2].firstChild.value,region:c[3].firstChild.value,on_divide:c[4].firstChild.value,tau:tau?+tau:null});}return out;}
function form(){return {name:$('name').value,shape:$('shape').value,n_cells:+$('n_cells').value,radius:+$('radius').value,h0:+$('h0').value,apical:$('apical').value,world:+$('world').value,n_frames:+$('n_frames').value,species:species(),organelles:organelles()};}
function fillForm(f){$('name').value=f.name;$('shape').value=f.shape;$('n_cells').value=f.n_cells;$('radius').value=f.radius;$('h0').value=f.h0;$('apical').value=f.apical;$('world').value=f.world;$('n_frames').value=f.n_frames;
 let tb=$('species');while(tb.rows.length>1)tb.deleteRow(-1);(f.species||[]).forEach(addSpecies);
 tb=$('organelles');while(tb.rows.length>1)tb.deleteRow(-1);(f.organelles||[]).forEach(addOrganelle);}
// THE PICKER IS THE SERVER'S LISTING: a browser file dialog hands the page bytes, never a path, and
// the specs live where the server runs. Folders that hold a spec.yaml (run archives) open as one.
window.pickOpen=async function(path){$('picker').style.display='block';const j=await (await fetch('/api/bio/ls?path='+encodeURIComponent(path||'/workspace/Plexus/config/tissue'))).json();if(j.error){$('picklist').textContent=j.error;return;}
 $('pickroots').innerHTML=Object.entries(j.roots).map(([k,v])=>`<button class="dim" onclick="pickOpen('${v}')">${k}</button>`).join('');$('pickpath').textContent=j.path;
 let h=`<div style="cursor:pointer;color:#9ac" onclick="pickOpen('${j.parent}')">.. (up)</div>`;
 for(const d of j.dirs)h+=`<div style="cursor:pointer;padding:1px 0"><span style="color:#7fb3ff" onclick="pickOpen('${j.path}/${d.name}')">&#128193; ${d.name}/</span>${d.spec?` <button class="dim" style="padding:1px 6px;font-size:11px" onclick="openSpec('${j.path}/${d.name}')">open run</button>`:''}</div>`;
 for(const f of j.files)h+=`<div style="cursor:pointer;padding:1px 0;color:#dde" onclick="openSpec('${j.path}/${f}')">&#128196; ${f}</div>`;
 $('picklist').innerHTML=h||'(empty)';};
window.openSpec=async function(pth){$('picker').style.display='none';status('opening '+pth+' ...');const j=await (await fetch('/api/bio/open?path='+encodeURIComponent(pth))).json();if(j.error){status(j.error,true);return;}$('openlab').textContent=pth;status('opened '+j.name+' -- seeding...');};
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
window.build=async function(){status('building the spec...');const j=await post('/api/bio/build',form());if(j.error){status(j.error+(j.detail?'\n'+j.detail:''),true);return;}specName=j.name;$('yamltext').value=j.raw;status(`spec saved: config/studio/${j.name}.yaml -- seeding and rendering...`);await reseed();};
window.reseed=async function(){playStop();FRAME=null;if(!specName){status('no spec yet',true);return;}status('seeding and building the renderer...');const r=await fetch('/api/bio/seed?name='+encodeURIComponent(specName));const j=await r.json();if(j.error){status(j.error,true);return;}SCENE=j;visPanel(j);tree(j);status(`seeded in ${j.seconds}s: `+Object.entries(j.sets).map(([k,v])=>`${k} ${v.n_live}`).join(', '));render(true);};
window.toggleYaml=function(){const y=$('yaml');y.style.display=y.style.display==='none'?'block':'none';};
window.saveYaml=async function(){const j=await post('/api/bio/save',{name:specName,raw:$('yamltext').value});if(j.error){status(j.error+(j.detail?'\n'+j.detail:''),true);return;}if(j.form)fillForm(j.form);status('saved; seeding...');await reseed();};
// THE PICTURE IS THE MOVIE RENDERER'S. Every camera change asks the server for a fresh screenshot;
// at most one request is in flight and the newest camera wins, so dragging never queues up.
let inflight=false, dirty=false;
async function render(force){if(inflight){dirty=true;return;}inflight=true;try{const r=await fetch(`/api/bio/render?azim=${CAM.azim}&elev=${CAM.elev}&zoom=${CAM.zoom}${FRAME===null?'':'&frame='+FRAME}&t=${Date.now()}`);if(r.ok){const b=await r.blob();const u=URL.createObjectURL(b);const im=$('view');const old=im.src;im.src=u;if(old.startsWith('blob:'))URL.revokeObjectURL(old);}else if(force){status((await r.json()).error||'render failed',true);}}catch(e){}finally{inflight=false;if(dirty){dirty=false;render();}}}
const im=$('view');let drag=null;
im.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,moved:false};im.style.cursor='grabbing';});
window.addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>2)drag.moved=true;CAM.azim-=dx*0.4;CAM.elev=Math.max(-89,Math.min(89,CAM.elev+dy*0.4));drag.x=e.clientX;drag.y=e.clientY;render();});
window.addEventListener('mouseup',async e=>{if(!drag)return;const moved=drag.moved;drag=null;im.style.cursor='grab';if(moved)return;
 const r=im.getBoundingClientRect();const fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;if(fx<0||fx>1||fy<0||fy>1)return;
 const j=await (await fetch(`/api/bio/pick?x=${fx.toFixed(4)}&y=${fy.toFixed(4)}`)).json();if(j.pick){showInfo(j.info);render();}else{$('info').textContent='nothing under the click';}});
im.addEventListener('wheel',e=>{e.preventDefault();CAM.zoom=Math.max(0.05,Math.min(60,CAM.zoom*(e.deltaY>0?1/1.08:1.08)));render();},{passive:false});
function showInfo(i){if(!i){$('info').textContent='(no object)';return;}let t='';
 if(i.kind==='cell'){t+=`cell #${i.cell}\n`;for(const [b,v] of Object.entries(i.blocks||{}))t+=`  ${b}: ${JSON.stringify(v)}\n`;for(const [s,per] of Object.entries(i.contains||{}))t+=`  contains ${s}: ${Object.entries(per).map(([a,b])=>b+' '+a).join(', ')}\n`;if(i.vertices)t+=`  vertices: ${i.vertices.length}`;}
 else if(i.kind==='vertex'){t+=`vertex #${i.vertex}\n  position: ${i.position.join(', ')}\n  shared by cells: ${(i.shared_by_cells||[]).join(', ')}`;}
 else{t+=`${i.species} #${i.index} (set ${i.set})\n  position: ${i.position.join(', ')}\n  parent: cell #${i.parent_cell}\n`;if(i.cell){const c=i.cell;for(const [s,per] of Object.entries(c.contains||{}))t+=`  the cell contains ${s}: ${Object.entries(per).map(([a,b])=>b+' '+a).join(', ')}\n`;}}
 $('info').textContent=t;}
function visPanel(j){const sps=[];for(const s of Object.values(j.sets))for(const n of (s.type_names||[]))if(!sps.includes(n))sps.push(n);
 if(!sps.length){$('vis').textContent='(no typed set)';return;}
 $('vis').innerHTML='<div class="row">'+sps.map(n=>{const c=(j.colors[n]||[1,1,1]).map(x=>Math.round(x*255));return `<label><input type="checkbox" checked onchange="setVisible('${n}',this.checked)"><span style="color:rgb(${c})">&#9679;</span> ${n}</label>`;}).join('')+'</div><div style="color:#778;font-size:11px">a species is one glyph actor of the renderer; unticking hides it</div>';}
window.setVisible=async function(n,on){await post('/api/bio/visible',{species:n,on});render();};
function tree(j){const h=j.hierarchy;let out='';for(const n of h.sets){const cnt=j.sets[n.name]?`${j.sets[n.name].n_live} live / ${j.sets[n.name].n_buffer}`:'';
 let rel='';if(n.parent)rel+=` <span class="cont">contained in ${n.parent}</span>`;if(n.maps)rel+=` <span class="rel">relation: ${Object.entries(n.maps).map(([k,v])=>k+'->'+v).join(', ')}</span>`;if(n.mesh)rel+=` <span class="rel">mesh: ${n.mesh}</span>`;
 out+=`<div onclick="window.setInfo('${n.name}')"><span class="n">${n.name}</span> ${cnt}${rel}${n.types.length?' <span class="cont">species: '+n.types.join(', ')+'</span>':''}</div>`;}
 out+=`<div style="color:#778;margin-top:4px">schedule: ${h.schedule.map(x=>typeof x==='string'?x:'{substeps}').join(' > ')}</div>`;$('tree').innerHTML=out;}
window.setInfo=function(name){const s=SCENE.sets[name];const n=SCENE.hierarchy.sets.find(x=>x.name===name);
 $('info').textContent=`set ${name}\n  entity: ${n.entity||'(by name)'}\n  buffer ${s.n_buffer}, live ${s.n_live}\n  blocks: ${s.blocks.join(', ')}\n`+(n.parent?`  contained in: ${n.parent} (${n.per_parent??'?'} per parent)\n`:'')+(n.maps?`  relation maps: ${JSON.stringify(n.maps)}\n`:'')+(n.types.length?`  species: ${n.types.join(', ')}\n`:'')+`  operators on it: ${SCENE.hierarchy.operators.filter(o=>o.at===name).map(o=>o.op).join(', ')||'-'}`;};
addOrganelle({name:'nucleus',count:1,radius:0.3,region:'basal_side',on_divide:'duplicate'});addSpecies({name:'integrin',region:'basal'});
// FOLLOW THE SERVER'S SESSION: whoever drives the API (a person, or Claude) is seen here.
let seen={version:-1,cam_version:-1};
async function poll(){try{const st=await (await fetch('/api/bio/state')).json();
 if(st.name&&st.version!==seen.version){seen.version=st.version;specName=st.name;$('name').value=st.name;const j=await (await fetch('/api/studio/spec?name='+encodeURIComponent(st.name))).json();if(j.raw)$('yamltext').value=j.raw;if(j.form)fillForm(j.form);await reseed();}
 if(st.cam_version!==seen.cam_version){seen.cam_version=st.cam_version;CAM.azim=st.azim;CAM.elev=st.elev;CAM.zoom=st.zoom;if(st.pick){const j=await (await fetch('/api/bio/info?pick='+encodeURIComponent(st.pick))).json();if(!j.error)showInfo(j);}render();}
 if(st.message)$('rstat').textContent=st.message;}catch(e){}finally{setTimeout(poll,1500);}}
poll();
// RUN: the engine simulates the spec from its seed; the picture follows every frame through the
// same renderer a generate uses, so the page shows the movie's frames as they are computed.
let running=false, playing=null, nframes=0;
// REPLAY AT ANY CAMERA: a kept frame is the levels' state, re-drawn by the renderer at the
// camera the page holds NOW, so orbit and zoom work while it plays (one frame in flight at a time).
let FRAME=null;
async function showFrame(i){FRAME=i;$('frame').value=i;$('framelab').textContent=`frame ${i}/${Math.max(nframes-1,0)}`;await render();}
window.playGo=async function(){const j=await (await fetch('/api/bio/frames')).json();nframes=j.n||0;if(!nframes){$('framelab').textContent='no frames yet: RUN first';return;}$('frame').max=nframes-1;playing=true;let i=0;
 while(playing){await showFrame(i);i=(i+1)%nframes;await new Promise(r=>setTimeout(r,30));}};
window.playStop=function(){playing=null;};
window.runGo=async function(){playStop();FRAME=null;const j=await post('/api/bio/run',{frames:+$('run_frames').value,device:$('run_device').value,keep:+$('run_keep').value});if(j.error){$('runstat').textContent=j.error;return;}running=true;$('runbtn').disabled=true;$('runstat').textContent=`running ${j.frames} frames on ${j.device}...`;rpoll();};
window.runStop=async function(){await post('/api/bio/run',{stop:true});};
async function rpoll(){try{const j=await (await fetch('/api/bio/run')).json();if(j.error&&!j.running){$('runstat').textContent='error: '+j.error;}
 else $('runstat').textContent=(j.running?'running: ':'done: ')+`frame ${j.frame}/${j.n_frames}, ${j.seconds}s`+(j.frame&&j.seconds?` (${(j.seconds/j.frame*1000).toFixed(0)} ms/frame)`:'');
 if(j.counts&&j.counts.sets)$('runcounts').textContent=Object.entries(j.counts.sets).filter(([k])=>k!=='half_edge').map(([k,v])=>`${k} ${v}`).join('  ')+(Object.keys(j.counts.species||{}).length?'  |  '+Object.entries(j.counts.species).map(([k,v])=>`${k} ${v}`).join('  '):'');
 render();if(j.running){setTimeout(rpoll,700);}else{running=false;$('runbtn').disabled=false;nframes=j.frames_kept||0;$('frame').max=Math.max(nframes-1,0);$('framelab').textContent=nframes?`${nframes} frames kept: PLAY (orbit and zoom while it plays)`:'';}}catch(e){setTimeout(rpoll,1500);}}
let cseen=0;
window.claudeGo=async function(){const t=$('task').value.trim();if(!t)return;$('claude').textContent='';cseen=0;const j=await post('/api/bio/claude',{task:t});if(j.error){$('cstat').textContent=j.error;return;}$('cstat').textContent='running...';$('cbtn').disabled=true;};
window.claudeStop=async function(){await post('/api/bio/claude',{stop:true});};
window.claudeNew=async function(){await post('/api/bio/claude',{new_session:true});$('claude').textContent+='[new session]\n';};
async function cpoll(){try{const j=await (await fetch('/api/bio/claude?since='+cseen)).json();if(j.lines&&j.lines.length){const el=$('claude');el.textContent+=j.lines.join('\n')+'\n';el.scrollTop=el.scrollHeight;cseen=j.n;}
 $('cstat').textContent=j.running?'running... '+j.seconds+'s':(j.error?'error: '+j.error.slice(0,200):(j.n?'done in '+j.seconds+'s':''));$('cbtn').disabled=!!j.running;$('cbtn').classList.toggle('on',!!j.running);}catch(e){}finally{setTimeout(cpoll,1200);}}
cpoll();
const q=new URLSearchParams(location.search);if(q.get('name')){specName=q.get('name');fetch('/api/studio/spec?name='+encodeURIComponent(specName)).then(r=>r.json()).then(j=>{if(j.raw){$('yamltext').value=j.raw;}if(j.form)fillForm(j.form);reseed();});}
</script></body></html>
"""


def page() -> str:
    return PAGE
