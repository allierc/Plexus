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

You have curl, sleep and jq ONLY: no python, no ls, no files. Put the JSON body inline in `curl -d '...'`
(one line, however long); a body you cannot write inline you cannot send.

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


def claude_start(task: str, port: int, model: str = "sonnet", timeout: int = 900, brief: str | None = None,
                 mode: str = "bio") -> dict:
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
    if fresh:
        # PRIMED ONCE PER SESSION: the framework digest, the operator atlas, the entities, the
        # design notes and two reference specs of this page's domain go in front of the first
        # task; every later task resumes the session and has them for free.
        from plexus.gui import corpus as _corpus
        ref = _corpus.corpus(mode)
        prompt = ("REFERENCE -- the Plexus framework you are working in. Read it once and keep it "
                  "for every task of this session.\n\n" + ref + "\n\n=== END OF REFERENCE ===\n\n" + prompt)
        primed_line = f"[session primed with {len(ref):,} chars: framework, operator atlas, entities, references]"
    CLAUDE.update(running=True, task=task, lines=([primed_line] if fresh else []) + [f"task: {task}" + ("" if fresh else f"  (session turn {CLAUDE['turns'] + 1})")],
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
