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
 <h2>Refine with a prompt</h2>
 <div class="row"><input id="prompt" style="width:100%" placeholder="e.g. make the cells twice as thick, or add a species named laminin on the basal cap"></div>
 <div class="row"><button onclick="refine()">APPLY</button> <span id="rstat" style="color:#8c8"></span></div>
 <div id="yaml"><textarea id="yamltext"></textarea><div><button onclick="saveYaml()">SAVE YAML</button></div></div>
 <h2>Hierarchy</h2><div id="tree">(none)</div>
 <h2>Selected object</h2><div id="info">click a cluster, a cell face or a vertex</div>
</div>
<div id="right"><div id="hint">drag to orbit, wheel to zoom, click to select</div></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
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
window.reseed=async function(){if(!specName){status('no spec yet',true);return;}status('seeding on the CPU...');const r=await fetch('/api/bio/seed?name='+encodeURIComponent(specName));const j=await r.json();if(j.error){status(j.error,true);return;}SCENE=j;draw(j);tree(j);status(`seeded in ${j.seconds}s: `+Object.entries(j.sets).map(([k,v])=>`${k} ${v.n_live}`).join(', '));};
window.refine=async function(){const p=$('prompt').value.trim();if(!p||!specName)return;$('rstat').textContent='Claude is editing the spec...';const j=await post('/api/bio/refine',{name:specName,prompt:p});if(j.error){$('rstat').textContent='';status(j.error+(j.detail?'\n'+j.detail:''),true);return;}$('rstat').textContent=`applied in ${j.seconds}s`;$('yamltext').value=j.raw;if(j.form)fillForm(j.form);await reseed();};
window.toggleYaml=function(){const y=$('yaml');y.style.display=y.style.display==='none'?'block':'none';};
window.saveYaml=async function(){const j=await post('/api/bio/save',{name:specName,raw:$('yamltext').value});if(j.error){status(j.error+(j.detail?'\n'+j.detail:''),true);return;}if(j.form)fillForm(j.form);status('saved; seeding...');await reseed();};
function init3d(){const R=$('right');renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(R.clientWidth,R.clientHeight);R.appendChild(renderer.domElement);
 scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0b0d);camera=new THREE.PerspectiveCamera(40,R.clientWidth/R.clientHeight,0.01,5000);camera.position.set(0,0,30);
 controls=new OrbitControls(camera,renderer.domElement);
 scene.add(new THREE.AmbientLight(0xffffff,0.6));const dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(1,1,1);scene.add(dl);
 renderer.domElement.addEventListener('click',onClick);window.addEventListener('resize',()=>{renderer.setSize(R.clientWidth,R.clientHeight);camera.aspect=R.clientWidth/R.clientHeight;camera.updateProjectionMatrix();});
 (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();}
function clear(){for(const o of pick)scene.remove(o);pick=[];}
function draw(j){clear();const T=j.tissue;let center=new THREE.Vector3();
 if(T){const capName=T.apical==='in'?'basal':'apical';for(const [cap,mat] of [[capName,{color:0xcfd8e3,opacity:0.35}],[T.apical==='in'?'apical':'basal',{color:0x8090a0,opacity:0.18}]]){const c=T.caps[cap];const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(c.verts.flat(),3));g.setIndex(c.tri.flat());g.computeVertexNormals();
  const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:mat.color,transparent:true,opacity:mat.opacity,side:THREE.DoubleSide,flatShading:true}));m.userData={kind:'cap',cap,face:c.face};scene.add(m);pick.push(m);
  const w=new THREE.LineSegments(new THREE.WireframeGeometry(g),new THREE.LineBasicMaterial({color:0x445566,transparent:true,opacity:0.5}));scene.add(w);pick.push(w);}
  const v=T.caps.mid.verts.slice(0,T.Nv);const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(v.flat(),3));
  const p=new THREE.Points(g,new THREE.PointsMaterial({color:0xffffff,size:0.12}));p.userData={kind:'vertex'};scene.add(p);pick.push(p);
  center=new THREE.Box3().setFromPoints(v.map(a=>new THREE.Vector3(...a))).getCenter(new THREE.Vector3());}
 for(const [name,s] of Object.entries(j.sets)){if(!s.pos||(T&&name===T.set))continue;const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(s.pos.flat(),3));
  const cols=[];const names=s.type_names||[];for(let i=0;i<s.pos.length;i++){const t=(s.node_type||[])[i]??0;const c=(j.colors[names[t]]||DEFC[t%DEFC.length]);cols.push(...c);}g.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
  const p=new THREE.Points(g,new THREE.PointsMaterial({size:0.16,vertexColors:true}));p.userData={kind:'cluster',set:name};scene.add(p);pick.push(p);}
 controls.target.copy(center);camera.position.copy(center.clone().add(new THREE.Vector3(0,0,Math.max(12,(j.world[0]||50)*0.45))));controls.update();}
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
 if(u.kind==='cluster'){const s=SCENE.sets[u.set];const i=h.index;const sp=(s.type_names||[])[(s.node_type||[])[i]??0]||u.set;const par=(s.parent||[])[i];
  $('info').textContent=`${sp} cluster #${s.idx[i]} (set ${u.set})\n  position: ${s.pos[i].join(', ')}\n  region: ${SCENE.hierarchy.sets.find(x=>x.name===u.set)?'per species (see spec types)':''}\n  parent: cell #${par}\n\n`+(par!==undefined?cellInfo(par):'');}
 else if(u.kind==='cap'){const f=u.face[h.faceIndex];$('info').textContent=`${u.cap} face of `+cellInfo(f);}
 else if(u.kind==='vertex'){const i=h.index;const cl=SCENE.tissue.cells_of_vertex[i];$('info').textContent=`vertex #${i} (set ${SCENE.tissue.set})\n  position: ${SCENE.tissue.caps.mid.verts[i].join(', ')}\n  shared by cells: ${cl.join(', ')}\n  (a vertex belongs to ${cl.length} cells; the half_edge relation records each side)`;}}
init3d();addSpecies({name:'integrin',region:'basal'});addSpecies({name:'myosin',region:'apical'});
const q=new URLSearchParams(location.search);if(q.get('name')){specName=q.get('name');fetch('/api/studio/spec?name='+encodeURIComponent(specName)).then(r=>r.json()).then(j=>{if(j.raw){$('yamltext').value=j.raw;}reseed();});}
</script></body></html>
"""


def page() -> str:
    return PAGE
