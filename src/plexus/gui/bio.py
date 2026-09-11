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
    if not species:
        raise ValueError("declare at least one protein species")

    s["general"].update(name=name, n_frames=frames, record_cap=frames + 2, world=[world] * 3)
    cell_n = max(512, 8 * n_cells)
    vert_n = max(2048, 40 * n_cells)
    s["sets"]["cell"]["n"] = cell_n
    s["sets"]["vertex"]["n"] = vert_n
    s["sets"]["half_edge"]["n"] = max(8192, 8 * cell_n)
    ns = len(species)
    for k in ("protein_s", "protein_tau", "n_protein"):
        s["sets"]["cell"]["state"][k] = {"width": ns}
    types = {}
    colors = {}
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
    dens = max(float(t_["density"]) for t_ in types.values())
    s["sets"]["protein"]["types"] = types
    s["sets"]["protein"]["per_parent"] = int(max(6, round(2.0 * dens * ns)))
    s["sets"]["protein"].pop("grow_reserve", None)
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
    p["mesh_surface"] = "basal" if apical == "in" else "apical"
    p["curve"] = [{"quantity": "cells", "xlabel": "frame", "ylabel": "cells", "ticks": 4, "ymin": 0, "ymax": max(1000, 5 * n_cells)}]
    for nm in types:
        p["curve"].append({"quantity": f"count:protein:{nm}", "xlabel": "frame", "ylabel": nm, "ticks": 4,
                           "ymin": 0, "ymax": 10000})
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
    }


# ---------------------------------------------------------------------------------------------
# the spec, built and seeded -> the scene
# ---------------------------------------------------------------------------------------------
def seed_scene(spec_path: str, device: str = "cpu") -> dict:
    """Build and seed the spec exactly as a run would, and return every live object: positions per
    set, species and parent per cluster, the tissue's caps as triangles with their cell ids, the
    cell set's scalar blocks, and the hierarchy (containment and relations) read off the spec."""
    from plexus import schema, engine
    import torch
    t0 = time.time()
    sim = schema.load(spec_path)
    H = engine.build(sim, device)
    engine.seed(H, sim, device)
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
# a server-side eye: the same scene, rendered offscreen, and a pick resolved by id
# ---------------------------------------------------------------------------------------------
def snapshot(scene: dict, azim: float = 30.0, elev: float = 20.0, zoom: float = 1.0,
             pick: str | None = None, size: tuple[int, int] = (900, 700)) -> bytes:
    """PNG of the seeded scene: the tissue's caps (the cap the proteins face in light grey, the
    other faint), every cluster as a dot in its species' colour, and the picked object, if any,
    ringed in yellow. Offscreen pyvista, so it works with no browser and no network."""
    import io
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background("black")
    T = scene.get("tissue")
    centre = np.zeros(3); radius = float(scene["world"][0]) * 0.25
    if T:
        front = "basal" if T["apical"] == "in" else "apical"
        back = "apical" if front == "basal" else "basal"
        for cap, col, op in ((front, "#cfd8e3", 0.45), (back, "#8090a0", 0.18)):
            c = T["caps"][cap]
            V = np.asarray(c["verts"], float); tri = np.asarray(c["tri"], int)
            faces = np.concatenate([np.full((tri.shape[0], 1), 3), tri], 1).ravel()
            pl.add_mesh(pv.PolyData(V, faces), color=col, opacity=op, show_edges=True, edge_color="#2b2b2b", smooth_shading=False)
        mid = np.asarray(T["caps"]["mid"]["verts"][: T["Nv"]], float)
        centre = mid.mean(0); radius = float(np.linalg.norm(mid - centre, axis=1).max()) * 1.6
    for name, st in scene["sets"].items():
        if not st.get("pos") or (T and name == T["set"]):
            continue
        P = np.asarray(st["pos"], float)
        names = st.get("type_names") or []
        nt = np.asarray(st.get("node_type") or [0] * len(P), int)
        for i, sp in enumerate(names or [name]):
            sel = nt == i if names else np.ones(len(P), bool)
            if not sel.any():
                continue
            col = scene["colors"].get(sp) or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            pl.add_points(P[sel], color=[float(v) for v in col], point_size=5, render_points_as_spheres=True)
    if pick:
        p = resolve_pick(scene, pick)
        if p and p.get("position") is not None:
            pl.add_points(np.asarray([p["position"]], float), color="yellow", point_size=18, render_points_as_spheres=True)
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    pl.camera.position = tuple(centre + d * radius * 4.0 / max(zoom, 1e-3))
    pl.camera.focal_point = tuple(centre)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = radius / max(zoom, 1e-3)
    img = pl.screenshot(return_img=True)
    pl.close()
    import imageio.v3 as iio
    buf = io.BytesIO(); iio.imwrite(buf, img, extension=".png")
    return buf.getvalue()


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
CLAUDE: dict = {"running": False, "task": "", "lines": [], "seconds": 0.0, "error": None, "started": 0.0}

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
  POST /api/bio/refine    {name, prompt} -> an English edit of the current spec (another Claude
                          applies it; 20-40 s); use it for anything the form cannot say
  GET  /api/bio/counts?name= -> live count per set, per species, and per cell per species
                          (use this for any counting; it is small)
  GET  /api/bio/seed?name=  -> the whole seeded scene as JSON (large; pipe through jq)
  GET  /api/bio/info?name=&pick=<set>:<index>   -> what one object is (protein:12, cell:7, vertex:5):
                          species, parent cell, what the cell contains, blocks, shared cells
  GET  /api/bio/view?azim=&elev=&zoom=&pick=&message=   -> turns the viewer's camera (degrees,
                          zoom 0.5-4), highlights a pick, and shows `message` on the page. Move in
                          steps of 30 degrees or less with `sleep 1` between them so the viewer can
                          follow; zoom no faster than x1.5 per step.
  GET  /api/bio/state     -> the session state

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


def claude_start(task: str, port: int, model: str = "sonnet", timeout: int = 900) -> dict:
    """Launch the CLI on the task in a thread; the transcript fills `CLAUDE['lines']` as it runs."""
    import subprocess
    import threading
    if CLAUDE["running"]:
        return {"error": "Claude is already driving; STOP it first"}
    CLAUDE.update(running=True, task=task, lines=[f"task: {task}"], seconds=0.0, error=None, started=time.time())

    def _go():
        cmd = [studio._claude_bin(), "-p", task,
               "--append-system-prompt", BIO_BRIEF.replace("{port}", str(port)),
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
 #left{position:absolute;left:0;top:0;bottom:0;width:380px;overflow:auto;background:#141418;border-right:1px solid #2a2a30;padding:10px 12px;box-sizing:border-box}
 #right{position:absolute;left:380px;top:0;right:0;bottom:0}
 canvas{display:block}
 h1{font-size:15px;margin:2px 0 8px;color:#fff} h2{font-size:12px;margin:14px 0 4px;color:#9ab;letter-spacing:.06em;text-transform:uppercase}
 label{display:inline-block;width:92px;color:#aab} input,select{background:#0e0e12;color:#eee;border:1px solid #333;border-radius:3px;padding:2px 5px;width:110px;margin:1px 0}
 input.short{width:56px} select{width:118px}
 button{background:#2b5f9e;color:#fff;border:0;border-radius:3px;padding:5px 10px;margin:3px 3px 3px 0;cursor:pointer} button.dim{background:#3a3a44}
 button:disabled{opacity:.5;cursor:default}
 table.sp{border-collapse:collapse;width:100%} table.sp td{padding:1px 2px} table.sp input,table.sp select{width:100%;box-sizing:border-box}
 #status{color:#8c8;min-height:16px;margin:4px 0;white-space:pre-wrap;font-size:12px} #status.err{color:#f88}
 #tree div{padding:1px 0 1px 8px;cursor:pointer} #tree div:hover{color:#fff} #tree .n{color:#7fb3ff} #tree .rel{color:#9ac} #tree .cont{color:#c9a}
 #info{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:12px;white-space:pre-wrap;min-height:60px;max-height:260px;overflow:auto}
 textarea{width:100%;height:120px;background:#0e0e12;color:#ddd;border:1px solid #333;font:11px monospace;box-sizing:border-box}
 #yaml{display:none} .row{margin:2px 0}
 #hint{position:absolute;right:12px;top:8px;color:#778;font-size:12px}
 #vis label{width:auto;color:#ccd;margin-right:10px;cursor:pointer} #vis input[type=checkbox]{width:auto;margin:0 3px 0 0}
 #cellgrid{display:flex;flex-wrap:wrap;gap:1px;max-height:110px;overflow:auto;background:#0e0e12;border:1px solid #2a2a30;padding:3px}
 #cellgrid span{width:26px;font-size:10px;text-align:center;cursor:pointer;color:#8ab;border:1px solid #223;border-radius:2px} #cellgrid span.on{background:#2b5f9e;color:#fff}
 button.claude{background:#000;border:1px solid #555;display:inline-flex;align-items:center;gap:6px} button.claude.on{background:#1f8f3f;border-color:#2fbf5f}
 button.claude svg{width:14px;height:14px;fill:#d97757} button.claude.on svg{fill:#fff}
 #claude{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:11px;white-space:pre-wrap;height:190px;overflow:auto;margin:4px 0}
</style></head><body>
<div id="left">
 <h1>Plexus bio objects</h1>
 <h2>Tissue</h2>
 <div class="row"><label>name</label><input id="name" value="bio_scene"></div>
 <div class="row"><label>shape</label><select id="shape"><option>sphere</option><option>disc</option><option>plane</option></select></div>
 <div class="row"><label>cells</label><input id="n_cells" class="short" value="200"> <label style="width:60px">radius</label><input id="radius" class="short" value="5.0"></div>
 <div class="row"><label>thickness h0</label><input id="h0" class="short" value="0.88"> <label style="width:60px">apical</label><select id="apical" style="width:64px"><option value="in">in</option><option value="out">out</option></select></div>
 <div class="row"><label>box</label><input id="world" class="short" value="50"> <label style="width:60px">frames</label><input id="n_frames" class="short" value="801"></div>
 <h2>Proteins <button class="dim" onclick="addSpecies()">+ species</button></h2>
 <table class="sp" id="species"><tr><th>name</th><th>region</th><th>density</th><th>s</th><th>tau</th><th></th></tr></table>
 <div class="row"><button onclick="build()">BUILD + SEED</button><button class="dim" onclick="toggleYaml()">YAML</button><button class="dim" onclick="reseed()">RE-SEED</button></div>
 <div id="status">form a scene, then BUILD</div>
 <h2>Claude takes over <span style="color:#778;font-weight:normal;text-transform:none">drives this page through its own routes</span></h2>
 <div class="row"><input id="task" style="width:100%" placeholder="e.g. build a 120-cell cyst with integrins outside and myosin inside, then show me one cell" onkeydown="if(event.key==='Enter')claudeGo()"></div>
 <div class="row"><button onclick="claudeGo()" id="cbtn" class="claude"><svg viewBox="0 0 24 24"><path d="M12 1.5l1.6 6.4 5.6-3.6-3.6 5.6 6.4 1.6-6.4 1.6 3.6 5.6-5.6-3.6L12 22.5l-1.6-6.4-5.6 3.6 3.6-5.6L1.5 12l6.9-1.6-3.6-5.6 5.6 3.6z"/></svg>CLAUDE</button><button class="dim" onclick="claudeStop()">STOP</button> <span id="cstat" style="color:#8c8"></span></div>
 <pre id="claude"></pre>
 <div id="rstat" style="color:#9ab;min-height:14px"></div>
 <div id="yaml"><textarea id="yamltext"></textarea><div><button onclick="saveYaml()">SAVE YAML</button></div></div>
 <h2>Visibility</h2><div id="vis">(seed a scene first)</div>
 <h2>Hierarchy</h2><div id="tree">(none)</div>
 <h2>Selected object</h2><div id="info">click a cluster, a cell face or a vertex</div>
</div>
<div id="right"><div id="hint">drag to orbit, wheel to zoom (8% per notch), click to select</div></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
// what is shown, Blender-outliner style: caps, edges, vertices, each species, and a per-cell set (null = all)
const VIS=window.VIS={apical:true,basal:true,edges:true,lateral:true,vertices:true,species:{},cells:null,width:2.5,dot:0.2,lmats:[],pmats:[]};
const $=id=>document.getElementById(id);
let SCENE=null, renderer, scene, camera, controls, pick=[], raycaster=new THREE.Raycaster(), mouse=new THREE.Vector2(), specName=null;
const DEFC=[[0.95,0.15,0.15],[0.25,0.6,1.0],[0.45,0.95,0.55],[1,0.85,0.3],[0.85,0.4,0.95],[0.3,0.9,0.95]];
function status(t,err){const s=$('status');s.textContent=t;s.className=err?'err':'';}
window.addSpecies=function(sp){sp=sp||{};const tb=$('species');const tr=tb.insertRow(-1);
 tr.innerHTML=`<td><input value="${sp.name||''}"></td><td><select><option>basal</option><option>apical</option><option>mid</option><option>interior</option></select></td><td><input value="${sp.density??3}"></td><td><input value="${sp.s??0.02}"></td><td><input value="${sp.tau??300}"></td><td><button class="dim" onclick="this.closest('tr').remove()">x</button></td>`;
 tr.cells[1].firstChild.value=sp.region||'basal';};
function species(){const out=[];for(const tr of $('species').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 out.push({name,region:c[1].firstChild.value,density:+c[2].firstChild.value,s:+c[3].firstChild.value,tau:+c[4].firstChild.value});}return out;}
function form(){return {name:$('name').value,shape:$('shape').value,n_cells:+$('n_cells').value,radius:+$('radius').value,h0:+$('h0').value,apical:$('apical').value,world:+$('world').value,n_frames:+$('n_frames').value,species:species()};}
function fillForm(f){$('name').value=f.name;$('shape').value=f.shape;$('n_cells').value=f.n_cells;$('radius').value=f.radius;$('h0').value=f.h0;$('apical').value=f.apical;$('world').value=f.world;$('n_frames').value=f.n_frames;
 const tb=$('species');while(tb.rows.length>1)tb.deleteRow(-1);(f.species||[]).forEach(addSpecies);}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
window.build=async function(){status('building the spec...');const j=await post('/api/bio/build',form());if(j.error){status(j.error+(j.detail?'\n'+j.detail:''),true);return;}specName=j.name;$('yamltext').value=j.raw;status(`spec saved: config/studio/${j.name}.yaml -- seeding...`);await reseed();};
window.reseed=async function(){if(!specName){status('no spec yet',true);return;}status('seeding on the CPU...');const r=await fetch('/api/bio/seed?name='+encodeURIComponent(specName));const j=await r.json();if(j.error){status(j.error,true);return;}SCENE=j;VIS.cells=null;draw(j);visPanel(j);tree(j);status(`seeded in ${j.seconds}s: `+Object.entries(j.sets).map(([k,v])=>`${k} ${v.n_live}`).join(', '));};
window.toggleYaml=function(){const y=$('yaml');y.style.display=y.style.display==='none'?'block':'none';};
window.saveYaml=async function(){const j=await post('/api/bio/save',{name:specName,raw:$('yamltext').value});if(j.error){status(j.error+(j.detail?'\n'+j.detail:''),true);return;}if(j.form)fillForm(j.form);status('saved; seeding...');await reseed();};
function init3d(){const R=$('right');renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(R.clientWidth,R.clientHeight);R.appendChild(renderer.domElement);
 scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0b0d);camera=new THREE.PerspectiveCamera(40,R.clientWidth/R.clientHeight,0.01,5000);camera.position.set(0,0,30);
 controls=new OrbitControls(camera,renderer.domElement);controls.enableZoom=false;controls.enableDamping=true;controls.dampingFactor=0.08;
 renderer.domElement.addEventListener('wheel',ev=>{ev.preventDefault();const f=ev.deltaY>0?1.08:1/1.08;const d=camera.position.clone().sub(controls.target);const L=d.length()*f;const fit=fitDist();
  d.setLength(Math.min(Math.max(L,0.15*fit),6*fit));camera.position.copy(controls.target).add(d);controls.update();},{passive:false});
 scene.add(new THREE.AmbientLight(0xffffff,0.6));const dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(1,1,1);scene.add(dl);
 renderer.domElement.addEventListener('click',onClick);window.addEventListener('resize',()=>{renderer.setSize(R.clientWidth,R.clientHeight);camera.aspect=R.clientWidth/R.clientHeight;camera.updateProjectionMatrix();for(const m of VIS.lmats)m.resolution.set(R.clientWidth,R.clientHeight);});
 (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();}
function fitDist(){return Math.max(12,((SCENE&&SCENE.world&&SCENE.world[0])||50)*0.45);}
function clear(){for(const o of pick)scene.remove(o);pick=[];if(hlc){scene.remove(hlc);hlc=null;}}
function cellOn(f){return VIS.cells===null||VIS.cells.has(f);}
function draw(j){clear();VIS.lmats=[];VIS.pmats=[];const T=j.tissue;let center=new THREE.Vector3();const R=$('right');
 if(T){const outer=T.apical==='in'?'basal':'apical';
  for(const [cap,mat] of [[outer,{color:0xcfd8e3,opacity:0.35}],[T.apical==='in'?'apical':'basal',{color:0x8090a0,opacity:0.18}]]){if(!VIS[cap])continue;const c=T.caps[cap];
   const idx=[],face=[];c.tri.forEach((t,k)=>{if(cellOn(c.face[k])){idx.push(...t);face.push(c.face[k]);}});if(!idx.length)continue;
   const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(c.verts.flat(),3));g.setIndex(idx);g.computeVertexNormals();
   const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:mat.color,transparent:true,opacity:mat.opacity,side:THREE.DoubleSide,flatShading:true}));m.userData={kind:'cap',cap,face};scene.add(m);pick.push(m);
   if(VIS.edges){const lg=new LineSegmentsGeometry().fromWireframeGeometry(new THREE.WireframeGeometry(g));const lm=new LineMaterial({color:cap===outer?0x9fb4cc:0x5a6a80,linewidth:VIS.width,transparent:true,opacity:0.9});
    lm.resolution.set(R.clientWidth,R.clientHeight);VIS.lmats.push(lm);const w=new LineSegments2(lg,lm);scene.add(w);pick.push(w);}}
  const allv=T.caps.mid.verts.slice(0,T.Nv);const keep=[];allv.forEach((v,i)=>{if(T.cells_of_vertex[i].some(cellOn))keep.push(i);});
  if(VIS.lateral&&keep.length){const A=T.caps.apical.verts,B=T.caps.basal.verts;const lg=new LineSegmentsGeometry();lg.setPositions(keep.flatMap(i=>[...A[i],...B[i]]));
   const lm=new LineMaterial({color:0x7f93ad,linewidth:VIS.width,transparent:true,opacity:0.9});lm.resolution.set(R.clientWidth,R.clientHeight);VIS.lmats.push(lm);const w=new LineSegments2(lg,lm);scene.add(w);pick.push(w);}
  if(VIS.vertices&&keep.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(keep.flatMap(i=>allv[i]),3));
   const p=new THREE.Points(g,new THREE.PointsMaterial({color:0xffffff,size:0.14}));p.userData={kind:'vertex',map:keep};scene.add(p);pick.push(p);}
  center=new THREE.Box3().setFromPoints(allv.map(a=>new THREE.Vector3(...a))).getCenter(new THREE.Vector3());}
 for(const [name,s] of Object.entries(j.sets)){if(!s.pos||(T&&name===T.set))continue;const names=s.type_names||[];const map=[],cols=[],pos=[];
  for(let i=0;i<s.pos.length;i++){const t=(s.node_type||[])[i]??0;const sp=names[t]||name;if(VIS.species[sp]===false)continue;if(s.parent&&!cellOn(s.parent[i]))continue;
   map.push(i);pos.push(...s.pos[i]);cols.push(...(j.colors[sp]||DEFC[t%DEFC.length]));}
  if(!map.length)continue;const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
  const pm=new THREE.PointsMaterial({size:VIS.dot,vertexColors:true});VIS.pmats.push(pm);const p=new THREE.Points(g,pm);p.userData={kind:'cluster',set:name,map};scene.add(p);pick.push(p);}
 if(!draw.keepCam){controls.target.copy(center);camera.position.copy(center.clone().add(new THREE.Vector3(0,0,Math.max(12,(j.world[0]||50)*0.45))));controls.update();}}
window.redraw=function(){if(SCENE){draw.keepCam=true;draw(SCENE);draw.keepCam=false;}};
// the Visibility panel: eye toggles per cap/edges/vertices, per species, and a clickable cell grid
function visPanel(j){const T=j.tissue;let h='';
 if(T){h+=`<div class="row"><label><input type="checkbox" ${VIS.apical?'checked':''} onchange="VIS.apical=this.checked;redraw()">apical cap</label><label><input type="checkbox" ${VIS.basal?'checked':''} onchange="VIS.basal=this.checked;redraw()">basal cap</label><label><input type="checkbox" ${VIS.edges?'checked':''} onchange="VIS.edges=this.checked;redraw()">cap edges</label><label><input type="checkbox" ${VIS.lateral?'checked':''} onchange="VIS.lateral=this.checked;redraw()">lateral edges</label><label><input type="checkbox" ${VIS.vertices?'checked':''} onchange="VIS.vertices=this.checked;redraw()">vertices</label></div>`;
  h+=`<div class="row"><label style="color:#aab">edge width px</label><input type="range" min="1" max="6" step="0.5" value="${VIS.width}" style="width:120px" oninput="VIS.width=+this.value;for(const m of VIS.lmats)m.linewidth=VIS.width"></div>`;}
 h+=`<div class="row"><label style="color:#aab">protein dot size</label><input type="range" min="0.05" max="1.0" step="0.05" value="${VIS.dot}" style="width:120px" oninput="VIS.dot=+this.value;for(const m of VIS.pmats)m.size=VIS.dot"></div>`;
 const sps=[];for(const s of Object.values(j.sets))for(const n of (s.type_names||[]))if(!sps.includes(n))sps.push(n);
 if(sps.length){h+='<div class="row">'+sps.map(n=>{if(VIS.species[n]===undefined)VIS.species[n]=true;const c=(j.colors[n]||[1,1,1]).map(x=>Math.round(x*255));return `<label><input type="checkbox" ${VIS.species[n]?'checked':''} onchange="VIS.species['${n}']=this.checked;redraw()"><span style="color:rgb(${c})">&#9679;</span> ${n}</label>`;}).join('')+'</div>';}
 if(T){const cells=j.sets[T.cell_set];const ids=cells&&cells.idx?cells.idx:[...new Set(T.caps.apical.face)].sort((a,b)=>a-b);
  h+=`<div class="row"><span style="color:#aab">cells</span> <button class="dim" onclick="VIS.cells=null;visPanel(SCENE);redraw()">all</button><button class="dim" onclick="VIS.cells=new Set();visPanel(SCENE);redraw()">none</button> <span style="color:#778">click a cell to toggle it; shift-click to solo</span></div>`;
  h+='<div id="cellgrid">'+ids.map(f=>`<span class="${cellOn(f)?'on':''}" onclick="toggleCell(${f},event.shiftKey)">${f}</span>`).join('')+'</div>';}
 $('vis').innerHTML=h||'(nothing to show)';}
window.visPanel=visPanel;
window.toggleCell=function(f,solo){const ids=SCENE.sets[SCENE.tissue.cell_set].idx;if(solo){VIS.cells=new Set([f]);}else{if(VIS.cells===null)VIS.cells=new Set(ids);if(VIS.cells.has(f))VIS.cells.delete(f);else VIS.cells.add(f);if(VIS.cells.size===ids.length)VIS.cells=null;}visPanel(SCENE);redraw();};
function tree(j){const h=j.hierarchy;let out='';for(const n of h.sets){const cnt=j.sets[n.name]?`${j.sets[n.name].n_live} live / ${j.sets[n.name].n_buffer}`:'';
 let rel='';if(n.parent)rel+=` <span class="cont">contained in ${n.parent}</span>`;if(n.maps)rel+=` <span class="rel">relation: ${Object.entries(n.maps).map(([k,v])=>k+'->'+v).join(', ')}</span>`;if(n.mesh)rel+=` <span class="rel">mesh: ${n.mesh}</span>`;
 out+=`<div onclick="window.setInfo('${n.name}')"><span class="n">${n.name}</span> ${cnt}${rel}${n.types.length?' <span class="cont">species: '+n.types.join(', ')+'</span>':''}</div>`;}
 out+=`<div style="color:#778;margin-top:4px">schedule: ${h.schedule.map(x=>typeof x==='string'?x:'{substeps}').join(' > ')}</div>`;$('tree').innerHTML=out;}
window.setInfo=function(name){const s=SCENE.sets[name];const n=SCENE.hierarchy.sets.find(x=>x.name===name);
 $('info').textContent=`set ${name}\n  entity: ${n.entity||'(by name)'}\n  buffer ${s.n_buffer}, live ${s.n_live}\n  blocks: ${s.blocks.join(', ')}\n`+(n.parent?`  contained in: ${n.parent} (${n.per_parent??'?'} per parent)\n`:'')+(n.maps?`  relation maps: ${JSON.stringify(n.maps)}\n`:'')+(n.types.length?`  species: ${n.types.join(', ')}\n`:'')+`  operators on it: ${SCENE.hierarchy.operators.filter(o=>o.at===name).map(o=>o.op).join(', ')||'-'}`;};
function cellInfo(f){const cs=SCENE.tissue.cell_set;const c=SCENE.sets[cs];let t=`cell #${f} (set ${cs})\n`;if(c&&c.rows){const k=c.idx.indexOf(f);if(k>=0)for(const [b,v] of Object.entries(c.rows))t+=`  ${b}: ${JSON.stringify(v[k])}\n`;}
 for(const [name,s] of Object.entries(SCENE.sets)){if(!s.parent)continue;const per={};s.parent.forEach((p,i)=>{if(p===f){const sp=(s.type_names||[])[(s.node_type||[])[i]??0]||name;per[sp]=(per[sp]||0)+1;}});if(Object.keys(per).length)t+=`  contains ${name}: ${Object.entries(per).map(([a,b])=>b+' '+a).join(', ')}\n`;}
 const vs=[];SCENE.tissue.cells_of_vertex.forEach((cl,i)=>{if(cl.includes(f))vs.push(i);});t+=`  vertices: ${vs.length} (${vs.slice(0,12).join(', ')}${vs.length>12?'...':''})`;return t;}
function onClick(ev){const r=renderer.domElement.getBoundingClientRect();mouse.x=((ev.clientX-r.left)/r.width)*2-1;mouse.y=-((ev.clientY-r.top)/r.height)*2+1;raycaster.setFromCamera(mouse,camera);raycaster.params.Points.threshold=0.25;
 const hits=raycaster.intersectObjects(pick.filter(o=>o.userData.kind),false);if(!hits.length)return;const h=hits[0];const u=h.object.userData;
 if(u.kind==='cluster'){const s=SCENE.sets[u.set];const i=u.map[h.index];const sp=(s.type_names||[])[(s.node_type||[])[i]??0]||u.set;const par=(s.parent||[])[i];
  $('info').textContent=`${sp} cluster #${s.idx[i]} (set ${u.set})\n  position: ${s.pos[i].join(', ')}\n  parent: cell #${par}\n\n`+(par!==undefined?cellInfo(par):'');highlight(s.pos[i]);highlightCell(par);}
 else if(u.kind==='cap'){const f=u.face[h.faceIndex];$('info').textContent=`${u.cap} face of `+cellInfo(f);highlight(null);highlightCell(f);}
 else if(u.kind==='vertex'){const i=u.map[h.index];const cl=SCENE.tissue.cells_of_vertex[i];$('info').textContent=`vertex #${i} (set ${SCENE.tissue.set})\n  position: ${SCENE.tissue.caps.mid.verts[i].join(', ')}\n  shared by cells: ${cl.join(', ')}\n  (a vertex belongs to ${cl.length} cells; the half_edge relation records each side)`;}}
init3d();addSpecies({name:'integrin',region:'basal'});addSpecies({name:'myosin',region:'apical'});
// FOLLOW THE SERVER'S SESSION: whoever drives the API (a person, or Claude from the terminal) is
// seen here. Spec version -> reload + reseed; camera version -> turn the view; pick -> select.
let seen={version:-1,cam_version:-1};
function applyCamera(st){if(!SCENE)return;const c=controls.target.clone();const e=st.elev*Math.PI/180,a=st.azim*Math.PI/180;const d=new THREE.Vector3(Math.cos(e)*Math.cos(a),Math.cos(e)*Math.sin(a),Math.sin(e));
 const dist=fitDist()/Math.min(Math.max(st.zoom,0.15),6);camera.position.copy(c.clone().add(d.multiplyScalar(dist)));camera.up.set(0,0,1);controls.update();}
window.selectById=function(pk){if(!SCENE||!pk)return;const [set,idx]=pk.split(':');const i=+idx;
 if(set==='cell'){$('info').textContent='cell '+cellInfo(i);highlight(null);highlightCell(i);return;}
 const s=SCENE.sets[set];if(s&&s.pos&&i<s.pos.length){const sp=(s.type_names||[])[(s.node_type||[])[i]??0]||set;const par=(s.parent||[])[i];$('info').textContent=`${sp} cluster #${s.idx[i]} (set ${set})\n  position: ${s.pos[i].join(', ')}\n  parent: cell #${par}\n\n`+(par!==undefined?cellInfo(par):'');highlight(s.pos[i]);highlightCell(par);}
 else if(SCENE.tissue&&set===SCENE.tissue.set){const cl=SCENE.tissue.cells_of_vertex[i];$('info').textContent=`vertex #${i}\n  shared by cells: ${cl.join(', ')}`;highlight(SCENE.tissue.caps.mid.verts[i]);}};
let hlc=null;window.highlightCell=function(f){if(hlc){scene.remove(hlc);hlc=null;}if(f===null||f===undefined||!SCENE||!SCENE.tissue)return;const T=SCENE.tissue;const seg=[];const ring=new Set();
 for(const cap of ['apical','basal']){const c=T.caps[cap];c.tri.forEach((t,k)=>{if(c.face[k]===f){seg.push(...c.verts[t[1]],...c.verts[t[2]]);ring.add(t[1]);}});}
 for(const i of ring)seg.push(...T.caps.apical.verts[i],...T.caps.basal.verts[i]);if(!seg.length)return;const lg=new LineSegmentsGeometry();lg.setPositions(seg);
 const lm=new LineMaterial({color:0xffee33,linewidth:VIS.width+1.5});lm.resolution.set($('right').clientWidth,$('right').clientHeight);VIS.lmats.push(lm);hlc=new LineSegments2(lg,lm);scene.add(hlc);};
let hl=null;function highlight(p){if(hl){scene.remove(hl);hl=null;}if(!p)return;const g=new THREE.SphereGeometry(0.22,12,12);hl=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:0xffee33}));hl.position.set(...p);scene.add(hl);}
async function poll(){try{const st=await (await fetch('/api/bio/state')).json();
 if(st.name&&st.version!==seen.version){seen.version=st.version;specName=st.name;$('name').value=st.name;const j=await (await fetch('/api/studio/spec?name='+encodeURIComponent(st.name))).json();if(j.raw)$('yamltext').value=j.raw;if(j.form)fillForm(j.form);await reseed();}
 if(st.cam_version!==seen.cam_version){seen.cam_version=st.cam_version;applyCamera(st);if(st.pick)selectById(st.pick);}
 if(st.message)$('rstat').textContent=st.message;}catch(e){}finally{setTimeout(poll,1500);}}
poll();
let cseen=0;
window.claudeGo=async function(){const t=$('task').value.trim();if(!t)return;$('claude').textContent='';cseen=0;const j=await post('/api/bio/claude',{task:t});if(j.error){$('cstat').textContent=j.error;return;}$('cstat').textContent='running...';$('cbtn').disabled=true;};
window.claudeStop=async function(){await post('/api/bio/claude',{stop:true});};
async function cpoll(){try{const j=await (await fetch('/api/bio/claude?since='+cseen)).json();if(j.lines&&j.lines.length){const el=$('claude');el.textContent+=j.lines.join('\n')+'\n';el.scrollTop=el.scrollHeight;cseen=j.n;}
 $('cstat').textContent=j.running?'running... '+j.seconds+'s':(j.error?'error: '+j.error.slice(0,200):(j.n?'done in '+j.seconds+'s':''));$('cbtn').disabled=!!j.running;$('cbtn').classList.toggle('on',!!j.running);}catch(e){}finally{setTimeout(cpoll,1200);}}
cpoll();
const q=new URLSearchParams(location.search);if(q.get('name')){specName=q.get('name');fetch('/api/studio/spec?name='+encodeURIComponent(specName)).then(r=>r.json()).then(j=>{if(j.raw){$('yamltext').value=j.raw;}reseed();});}
</script></body></html>
"""


def page() -> str:
    return PAGE
