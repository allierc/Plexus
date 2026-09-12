"""The material page: define MPM bodies in a box, seed them, look, run.

The twin of `gui/bio.py` for the material side of Plexus. A scene is a list of BODIES (a ball
or a block of one material: elastic, liquid or snow, with its stiffness and density) in a walled
box with gravity. The form WRITES THE SPEC A PERSON WOULD WRITE -- `general / sets / fields /
operators / schedule / plotting`, the shape of `config/si_material/si_three_balls.yaml`, which is
itself this form's default scene written out -- rather than patching a template: a template
carries its own numbers (its substep, its wall model) and those were what made the page run a
different simulation from the file with the same name. The picture, the picking, the engine run
and the Claude drive are the bio page's (`gui/bio_view.py`, `/api/bio/*`).

WHAT THE FORM CAN AND CANNOT SAY, and why. Bodies are typed on the `cell` set, one type per body
in FORM ORDER with `type_layout: ordered`, so the first body is at the first centre. Every ball
shares ONE radius (`sets.mpm_particle.radius`) because the engine sizes a body's points from
that radius and the body's density (`p_vol = V / per_parent`, mass = `p_vol * density`); a
per-body radius would need a per-body point count, which `particle_mass` refuses and a shared
radius makes unnecessary. A block spans its six numbers and is sized by its own box.
"""
from __future__ import annotations

import math
import os

import yaml

from plexus.gui import studio

REPO = studio.REPO
MATERIALS = ("elastic", "liquid", "snow")
DEFAULT_COLORS = [[0.95, 0.25, 0.20], [0.30, 0.55, 1.00], [0.95, 0.95, 0.95], [1.00, 0.85, 0.30],
                  [0.85, 0.40, 0.95], [0.30, 0.90, 0.95]]
# THE PAGE'S DEFAULT SCENE IS A FILE IN THE REPO. `config/si_material/si_three_balls.yaml` is this
# form written out (tests/test_gui_parity.py holds the two equal), so the benchmark everyone quotes
# and the scene the page opens on are one spec, not two that happen to agree today.
REFERENCE = os.path.join(REPO, "config", "si_material", "si_three_balls.yaml")
DEFAULT_FORM = {
    "name": "si_three_balls", "world": 0.5, "n_grid": 96, "n_frames": 800, "dt": 0.0008333333333333334,
    "gravity": 9.81, "particles": 100000, "radius": 0.05, "wall_damp": 0.5, "friction": 0.4,
    "launch": 0.4, "movie_frames": 400, "stills": 10, "seed": 1,
    "bodies": [
        {"name": "elastic", "shape": "ball", "centre": [0.15, 0.31, 0.25], "material": "elastic",
         "youngs": 2000000.0, "density": 1000.0},
        {"name": "liquid", "shape": "ball", "centre": [0.25, 0.37, 0.25], "material": "liquid",
         "bulk_modulus": 981000.0, "density": 1000.0},
        {"name": "snow", "shape": "ball", "centre": [0.35, 0.29, 0.25], "material": "snow",
         "youngs": 500000.0, "density": 350.0},
    ],
}


def build_spec(form: dict) -> dict:
    """The spec from the form: box, grid, gravity, frames, the wall, the movie's shape, the bodies."""
    name = str(form.get("name") or "material_scene").strip()
    world = float(form.get("world", 0.5))
    n_grid = int(form.get("n_grid", 96))
    frames = int(form.get("n_frames", 800))
    dt = float(form.get("dt", 1.0 / 1200.0))
    g = float(form.get("gravity", 9.81))
    per = int(form.get("particles", 100000))
    radius = float(form.get("radius", 0.1 * world))
    wall_damp = float(form.get("wall_damp", 0.5))
    friction = float(form.get("friction", 0.4))
    launch = float(form.get("launch", 0.0))
    seed = int(form.get("seed", 1))
    bodies = form.get("bodies") or []
    if not bodies:
        raise ValueError("declare at least one body")
    types, start, colors = {}, [], {}
    for i, b in enumerate(bodies):
        nm = str(b.get("name") or f"body_{i}").strip()
        if nm in types:
            raise ValueError(f"two bodies are called {nm!r}; names are the type names and must differ")
        mat = str(b.get("material", "elastic")).lower()
        if mat not in MATERIALS:
            raise ValueError(f"body {nm!r}: material must be one of {MATERIALS}")
        shape = str(b.get("shape", "ball")).lower()
        t = {"count": 1, "material": mat}
        if mat == "liquid":
            t["bulk_modulus"] = float(b.get("bulk_modulus", b.get("youngs", 1.0e5)))
        else:
            t["youngs"] = float(b.get("youngs", 2.0e6))
        t["density"] = float(b.get("density", 1000.0))
        if shape == "ball":
            c = [float(v) for v in (b.get("centre") or [world / 2, world * 0.75, world / 2])]
            if len(c) != 3:
                raise ValueError(f"body {nm!r}: a ball needs a centre x y z")
            t["shape"] = "ball"
            start.append(c)
        elif shape == "block":
            blk = [float(v) for v in (b.get("block") or [0, 0, 0, world, 0.25 * world, world])]
            if len(blk) != 6:
                raise ValueError(f"body {nm!r}: block needs 6 numbers x0 y0 z0 x1 y1 z1")
            t["block"] = blk
            start.append([0.5 * (blk[k] + blk[k + 3]) for k in range(3)])
        else:
            raise ValueError(f"body {nm!r}: shape must be ball|block")
        types[nm] = t
        colors[nm] = [float(v) for v in (b.get("color") or DEFAULT_COLORS[i % len(DEFAULT_COLORS)])]
    # THE SUBSTEP IS THE SCENE'S. Explicit MPM is stable for a substep below dx / c with
    # c = sqrt(stiffness / density) the sound speed of the stiffest body (Young's modulus for
    # elastic and snow, bulk modulus for liquid); 0.4 of that, rounded to a whole number of
    # substeps per frame so the frame clock stays exact. The pipeline's own CFL guard
    # (`mpm_cfl.Courant_Friedrichs_Lewy_condition`) then has the last word on the written file.
    dx = world / n_grid
    c_max = max(math.sqrt(float(t.get("youngs", t.get("bulk_modulus", 1.0e5))) / float(t["density"]))
                for t in types.values())
    n_sub = max(1, math.ceil(dt / (0.4 * dx / c_max)))
    substep = float(f"{dt / n_sub:.6e}")
    mp = {"parent": "cell", "per_parent": per, "density": float(bodies[0].get("density", 1000.0)),
          "radius": radius}
    if launch > 0:
        mp["vel_init"] = launch
    return {
        "general": {"name": name, "seed": seed, "n_frames": frames, "dt": dt, "boundary": "wall",
                    "dim": 3, "world": [world, world, world], "save_data": False,
                    "units": {"length_um": 1000000.0, "time_s": 1.0, "force_nN": 1000000000.0}},
        "sets": {"cell": {"n": len(bodies), "start": start, "type_layout": "ordered", "types": types},
                 "mpm_particle": mp},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": n_grid}},
        "operators": [
            {"op": "gravity", "at": "cell", "g": g},
            {"op": "mpm_strain", "at": "mpm_particle", "implementation": "warp"},
            {"op": "mpm_viscosity", "at": "mpm_particle", "eta": 0.001},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0, "a_max": 200.0,
             "implementation": "warp", "polar": "higham"},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": wall_damp, "wall_friction": friction,
             "surface_tension": 0.072, "csf_rho": 1000.0, "csf_band": 0.2},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1000000000.0, "implementation": "warp"},
        ],
        "schedule": ["gravity", {"substep_dt": substep,
                                 "steps": ["mpm_strain", "mpm_viscosity", "mpm_scatter", "mpm_grid_update", "mpm_gather"]}],
        "plotting": {"renderer": "vtk_points", "background": "black", "up_axis": 1, "camera_elev": 1.18,
                     "camera_turns": 0.0, "camera_zoom": 0.0, "render_3d": "dots", "dot_size": 1.6,
                     "dot_shading": "body", "dot_specular": 0.4, "dot_specular_power": 12,
                     "max_frames": int(form.get("movie_frames", 400)), "stills": int(form.get("stills", 10)),
                     "keep_stills": True, "splat_res": 600, "box_frame": True, "hide_sets": ["cell"],
                     "colors": colors, "slow_motion": 4},
    }


def write_spec(spec: dict, path: str) -> str:
    """Write the spec and let the pipeline's CFL guard have its say, exactly as `-o generate` would
    before running it -- so the file on disk is the file that runs, whichever button wrote it.
    Returns the text now in the file."""
    from plexus.gui.server import _dump_yaml
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_dump_yaml(spec))
    Courant_Friedrichs_Lewy_condition(path)
    return open(path).read()


def form_from_spec(spec: dict) -> dict:
    """The coarse form read back off a material spec."""
    cell = (spec.get("sets") or {}).get("cell") or {}
    mp = (spec.get("sets") or {}).get("mpm_particle") or {}
    colors = (spec.get("plotting") or {}).get("colors") or {}
    starts = cell.get("start") or []
    bodies = []
    for i, (nm, t) in enumerate((cell.get("types") or {}).items()):
        t = t or {}
        b = {"name": nm, "material": t.get("material", "elastic"), "density": t.get("density", mp.get("density", 1000.0)),
             "youngs": t.get("youngs", t.get("bulk_modulus")), "color": colors.get(nm)}
        if t.get("block"):
            b.update(shape="block", block=t["block"])
        else:
            b.update(shape="ball", centre=(starts[i] if i < len(starts) else None))
        bodies.append(b)
    gen = spec.get("general") or {}
    ops = spec.get("operators") or []
    g = next((o.get("g", 9.81) for o in ops if o.get("op") == "gravity"), 9.81)
    gu = next((o for o in ops if o.get("op") == "mpm_grid_update"), {})
    pl = spec.get("plotting") or {}
    return {"name": gen.get("name", ""), "world": (gen.get("world") or [0.5])[0], "n_frames": gen.get("n_frames", 800),
            "dt": gen.get("dt", 1.0 / 1200.0), "gravity": g, "wall_damp": gu.get("wall_damp", 0.5),
            "friction": gu.get("wall_friction", 0.0), "launch": mp.get("vel_init", 0.0), "seed": gen.get("seed", 1),
            "n_grid": ((spec.get("fields") or {}).get("mpm_grid") or {}).get("n_grid", 96),
            "particles": mp.get("per_parent", 100000), "radius": mp.get("radius", 0.05),
            "movie_frames": pl.get("max_frames", 400), "stills": pl.get("stills", 10), "bodies": bodies}


MATERIAL_BRIEF = """You are driving the Plexus material page, a UI for DEFINING material bodies (balls and
slabs of elastic, liquid or snow material in a walled box under gravity) and running them with the
material point method. A person is watching the page: it follows every change you make, and your
words appear in a transcript beside the 3D scene. Say, in ONE short plain-English line before
each call, what you are about to do and why; say what you found after it. No headers, no markdown.

Your ONLY tool is `curl` against the local server at http://127.0.0.1:{port} . The routes:

  POST /api/material/build  JSON form -> writes and seeds a spec. Fields: name, world (box side,
                          metres), n_grid, n_frames, dt, gravity, particles (per body), radius
                          (every ball's radius, metres), wall_damp (0-1, wall-normal velocity kept
                          at the grid), friction (Coulomb, 0 = slippery), launch (initial speed
                          per body, m/s), movie_frames, stills, seed,
                          bodies: [{name, shape (ball|block), centre [x,y,z] for a ball,
                          block [x0,y0,z0,x1,y1,z1] for a slab, material (elastic|liquid|snow),
                          youngs (elastic/snow) or bulk_modulus (liquid), density}].
                          Bodies keep their order: the first body is at the first centre.
                          A ball deforms visibly below ~30,000 Pa; 1,000,000 is rigid.
  POST /api/bio/refine    {name, prompt} -> an English edit of the current spec (another Claude
                          applies it; 20-40 s)
  GET  /api/bio/counts?name= -> live count per set and per body type
  GET  /api/bio/info?name=&pick=<set>:<index>   -> what one object is
  GET  /api/bio/view?azim=&elev=&zoom=&pick=&message=   -> turns the viewer's camera (degrees,
                          zoom 0.05-60), highlights a pick, shows `message`. Move in steps of 30
                          degrees or less with `sleep 1` between them.
  POST /api/bio/run       {frames, device} -> simulate from the seed, the picture following each
                          frame; {stop: true} aborts.  GET /api/bio/run -> progress and live counts
  GET  /api/bio/state     -> the session state

Rules: use only these routes (curl -s, JSON bodies with -H 'Content-Type: application/json'), with
jq and sleep as the only other commands; never touch files; keep the spec name given in the task
unless told otherwise; finish with two or three lines summarising the scene and the bodies.
"""


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Plexus material</title>
<style>
 html,body{margin:0;height:100%;background:#0b0b0d;color:#ddd;font:13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
 #left{position:absolute;left:0;top:0;bottom:0;width:430px;overflow:auto;background:#141418;border-right:1px solid #2a2a30;padding:10px 12px;box-sizing:border-box}
 #right{position:absolute;left:430px;top:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden}
 #view{max-width:100%;max-height:100%;cursor:grab;user-select:none;-webkit-user-drag:none}
 h1{font-size:15px;margin:2px 0 8px;color:#fff} h2{font-size:12px;margin:14px 0 4px;color:#9ab;letter-spacing:.06em;text-transform:uppercase}
 label{display:inline-block;width:92px;color:#aab} input,select{background:#0e0e12;color:#eee;border:1px solid #333;border-radius:3px;padding:2px 5px;width:110px;margin:1px 0}
 input.short{width:56px} select{width:118px}
 button{background:#2b5f9e;color:#fff;border:0;border-radius:3px;padding:5px 10px;margin:3px 3px 3px 0;cursor:pointer} button.dim{background:#3a3a44}
 button:disabled{opacity:.5;cursor:default}
 table.sp{border-collapse:collapse;width:100%} table.sp td{padding:1px 2px} table.sp th{font-weight:normal;color:#889;font-size:11px} table.sp input,table.sp select{width:100%;box-sizing:border-box;font-size:11px}
 #status{color:#8c8;min-height:16px;margin:4px 0;white-space:pre-wrap;font-size:12px} #status.err{color:#f88}
 #tree div{padding:1px 0 1px 8px;cursor:pointer} #tree div:hover{color:#fff} #tree .n{color:#7fb3ff} #tree .rel{color:#9ac} #tree .cont{color:#c9a}
 #info{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:12px;white-space:pre-wrap;min-height:60px;max-height:260px;overflow:auto}
 textarea{width:100%;height:120px;background:#0e0e12;color:#ddd;border:1px solid #333;font:11px monospace;box-sizing:border-box}
 #yaml{display:none} .row{margin:2px 0}
 #hint{position:absolute;right:12px;top:8px;color:#778;font-size:12px}
 button.claude{background:#000;border:1px solid #555;display:inline-flex;align-items:center;gap:6px} button.claude.on{background:#1f8f3f;border-color:#2fbf5f}
 button.claude svg{width:14px;height:14px;fill:#d97757} button.claude.on svg{fill:#fff}
 #claude{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:11px;white-space:pre-wrap;height:190px;overflow:auto;margin:4px 0}
</style></head><body>
<div id="left">
 <h1>Plexus material</h1>
 <h2>Open a spec</h2>
 <div class="row"><button class="dim" onclick="pickOpen()">OPEN...</button> <span id="openlab" style="color:#778;font-size:11px">a spec.yaml or a run folder; copied into config/studio and seeded as is</span></div>
 <div id="picker" style="display:none;position:fixed;left:60px;top:40px;width:560px;max-height:80vh;background:#1a1a20;border:1px solid #556;border-radius:6px;padding:10px;z-index:10;box-shadow:0 0 30px #000">
  <div class="row"><b>Open a spec</b> <span style="float:right;cursor:pointer" onclick="$('picker').style.display='none'">&#10005;</span></div>
  <div class="row" id="pickroots"></div>
  <div class="row" id="pickpath" style="color:#9ab;font-family:monospace;font-size:11px;word-break:break-all"></div>
  <div id="picklist" style="max-height:55vh;overflow:auto;background:#0e0e12;border:1px solid #2a2a30;padding:4px;font-size:12px"></div>
 </div>
 <h2>Box</h2>
 <div class="row"><label>name</label><input id="name" value="si_three_balls"></div>
 <div class="row"><label>box side (m)</label><input id="world" class="short" value="0.5"> <label style="width:60px">grid</label><input id="n_grid" class="short" value="96"></div>
 <div class="row"><label>frames</label><input id="n_frames" class="short" value="800"> <label style="width:60px">dt (s)</label><input id="dt" class="short" value="0.00083"></div>
 <div class="row"><label>gravity</label><input id="gravity" class="short" value="9.81"> <label style="width:60px">particles</label><input id="particles" class="short" value="100000" title="material points per body"></div>
 <div class="row"><label>ball radius (m)</label><input id="radius" class="short" value="0.05" title="every ball's radius; a block is sized by its own box"> <label style="width:60px">launch</label><input id="launch" class="short" value="0.4" title="initial speed given to each body, m/s (0 = dropped from rest)"></div>
 <div class="row"><label>wall damp</label><input id="wall_damp" class="short" value="0.5" title="share of the wall-normal velocity kept at the grid: 1 = elastic wall, 0 = dead"> <label style="width:60px">friction</label><input id="friction" class="short" value="0.4" title="Coulomb friction on the walls, 0 = slippery"></div>
 <div class="row"><label title="frames the movie keeps: every frame up to this many, then every 2nd, 4th...">movie frames</label><input id="movie_frames" class="short" value="400"> <label style="width:60px" title="PNG stills dropped through the run, also the live pictures on this page">stills</label><input id="stills" class="short" value="10"> <label style="width:40px">seed</label><input id="seed" class="short" value="1"></div>
 <h2>Bodies <button class="dim" onclick="addBody()">+ body</button></h2>
 <table class="sp" id="bodies"><tr><th>name</th><th>shape</th><th>centre x y z | block x0 y0 z0 x1 y1 z1</th><th>material</th><th>stiffness</th><th>density</th><th></th></tr></table>
 <div style="color:#778;font-size:11px">a ball is placed at its centre with the shared radius above; a block spans its six numbers (metres). Bodies keep their order: the first body is at the first centre. stiffness = Young's modulus (elastic, snow) or bulk modulus (liquid), Pa.</div>
 <div class="row"><button onclick="build()">BUILD + SEED</button><button class="dim" onclick="toggleYaml()">YAML</button><button class="dim" onclick="reseed()">RE-SEED</button></div>
 <div id="status">building the default scene...</div>
 <h2>Run the engine <span style="color:#778;font-weight:normal;text-transform:none">the same -o generate as Plexus_Main.py: movie and stills land in graphs_data/studio/&lt;name&gt;</span></h2>
 <div class="row"><label>device</label><select id="run_device" style="width:80px"><option>cuda:0</option><option>cuda:1</option><option>cpu</option></select> <span style="color:#778;font-size:11px">frames, movie frames and stills are the spec's (BUILD writes them)</span></div>
 <div class="row"><button onclick="runGo()" id="runbtn">RUN</button><button class="dim" onclick="runStop()">STOP</button> <span id="runstat" style="color:#8c8"></span></div>
 <div id="runcounts" style="color:#9ab;font-size:12px;min-height:14px"></div>
 <div class="row"><button class="dim" onclick="playGo()" id="playbtn">PLAY</button><button class="dim" onclick="playStop()">PAUSE</button><button class="dim" onclick="movieGo()" title="the movie.mp4 the run wrote, as a file">MOVIE</button><button class="dim" onclick="playStop();showMovie(null);FRAME=null;render(true)" title="back to the live view">LIVE</button> <input type="range" id="frame" min="0" max="0" value="0" style="width:150px" oninput="playing=null;showMovie(null);showFrame(+this.value)"> <span id="framelab" style="color:#9ab"></span></div>
 <h2>Claude takes over <span style="color:#778;font-weight:normal;text-transform:none">drives this page through its own routes</span></h2>
 <div class="row"><input id="task" style="width:100%" placeholder="e.g. drop a snow ball onto a water slab, then run 100 frames" onkeydown="if(event.key==='Enter')claudeGo()"></div>
 <div class="row"><button onclick="claudeGo()" id="cbtn" class="claude"><svg viewBox="0 0 24 24"><path d="M12 1.5l1.6 6.4 5.6-3.6-3.6 5.6 6.4 1.6-6.4 1.6 3.6 5.6-5.6-3.6L12 22.5l-1.6-6.4-5.6 3.6 3.6-5.6L1.5 12l6.9-1.6-3.6-5.6 5.6 3.6z"/></svg>CLAUDE</button><button class="dim" onclick="claudeStop()">STOP</button><button class="dim" onclick="claudeNew()" title="forget the conversation so far">NEW SESSION</button> <span id="cstat" style="color:#8c8"></span></div>
 <pre id="claude"></pre>
 <div id="rstat" style="color:#9ab;min-height:14px"></div>
 <div id="yaml"><textarea id="yamltext"></textarea><div><button onclick="saveYaml()">SAVE YAML</button></div></div>
 <h2>Hierarchy</h2><div id="tree">(none)</div>
 <h2>Selected object</h2><div id="info">click a particle</div>
</div>
<div id="right"><img id="view" draggable="false"><video id="movie" style="display:none;max-width:100%;max-height:100%" controls muted></video><div id="hint">drag to orbit, wheel to zoom, click to select -- rendered by the movie renderer, also while a run is going</div></div>
<script>
const $=id=>document.getElementById(id);
let SCENE=null, specName=null;
const CAM={azim:30,elev:20,zoom:1};
function status(t,err){const s=$('status');s.textContent=t;s.className=err?'err':'';}
window.addBody=function(b){b=b||{};const tb=$('bodies');const tr=tb.insertRow(-1);const geo=b.shape==='block'?(b.block||[0,0,0,0.5,0.125,0.5]).join(' '):(b.centre||[0.25,0.375,0.25]).join(' ');
 tr.innerHTML=`<td><input value="${b.name||''}"></td><td><select><option>ball</option><option>block</option></select></td><td><input value="${geo}"></td><td><select><option>elastic</option><option>liquid</option><option>snow</option></select></td><td><input value="${b.youngs??2000000}"></td><td><input value="${b.density??1000}"></td><td><button class="dim" onclick="this.closest('tr').remove()">x</button></td>`;
 tr.cells[1].firstChild.value=b.shape||'ball';tr.cells[3].firstChild.value=b.material||'elastic';};
function bodies(){const out=[];for(const tr of $('bodies').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 const nums=c[2].firstChild.value.trim().split(/[\s,]+/).map(Number);const shape=c[1].firstChild.value;const mat=c[3].firstChild.value;
 const b={name,shape,material:mat,density:+c[5].firstChild.value};if(shape==='ball'){b.centre=nums.slice(0,3);}else{b.block=nums.slice(0,6);}
 if(mat==='liquid')b.bulk_modulus=+c[4].firstChild.value;else b.youngs=+c[4].firstChild.value;out.push(b);}return out;}
const NUM=['world','n_grid','n_frames','dt','gravity','particles','radius','launch','wall_damp','friction','movie_frames','stills','seed'];
function form(){const f={name:$('name').value,bodies:bodies()};for(const k of NUM)f[k]=+$(k).value;return f;}
function fillForm(f){$('name').value=f.name;for(const k of NUM)if(f[k]!==undefined&&f[k]!==null)$(k).value=f[k];
 const tb=$('bodies');while(tb.rows.length>1)tb.deleteRow(-1);(f.bodies||[]).forEach(addBody);}
// THE PICKER IS THE SERVER'S LISTING: a browser file dialog hands the page bytes, never a path, and
// the specs live where the server runs. Folders that hold a spec.yaml (run archives) open as one.
window.pickOpen=async function(path){$('picker').style.display='block';const j=await (await fetch('/api/bio/ls?path='+encodeURIComponent(path||'/workspace/Plexus/config/si_material'))).json();if(j.error){$('picklist').textContent=j.error;return;}
 $('pickroots').innerHTML=Object.entries(j.roots).map(([k,v])=>`<button class="dim" onclick="pickOpen('${v}')">${k}</button>`).join('');$('pickpath').textContent=j.path;
 let h=`<div style="cursor:pointer;color:#9ac" onclick="pickOpen('${j.parent}')">.. (up)</div>`;
 for(const d of j.dirs)h+=`<div style="cursor:pointer;padding:1px 0"><span style="color:#7fb3ff" onclick="pickOpen('${j.path}/${d.name}')">&#128193; ${d.name}/</span>${d.spec?` <button class="dim" style="padding:1px 6px;font-size:11px" onclick="openSpec('${j.path}/${d.name}')">open run</button>`:''}</div>`;
 for(const f of j.files)h+=`<div style="cursor:pointer;padding:1px 0;color:#dde" onclick="openSpec('${j.path}/${f}')">&#128196; ${f}</div>`;
 $('picklist').innerHTML=h||'(empty)';};
window.openSpec=async function(pth){$('picker').style.display='none';status('opening '+pth+' ...');const j=await (await fetch('/api/bio/open?path='+encodeURIComponent(pth))).json();if(j.error){status(j.error,true);return;}$('openlab').textContent=pth;status('opened '+j.name+' -- seeding...');};
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
window.build=async function(){status('building the spec...');const j=await post('/api/material/build',form());if(j.error){status(j.error+(j.detail?'\n'+j.detail:''),true);return;}specName=j.name;if(j.version!==undefined)seen.version=j.version;$('yamltext').value=j.raw;status(`spec saved: config/studio/${j.name}.yaml -- seeding and rendering...`);await reseed();};
window.reseed=async function(){playStop();showMovie(null);FRAME=null;if(!specName){status('no spec yet',true);return;}status('seeding and building the renderer...');const r=await fetch('/api/bio/seed?name='+encodeURIComponent(specName));const j=await r.json();if(j.error){status(j.error,true);return;}SCENE=j;tree(j);status(`seeded in ${j.seconds}s: `+Object.entries(j.sets).map(([k,v])=>`${k} ${v.n_live}`).join(', '));render(true);};
window.toggleYaml=function(){const y=$('yaml');y.style.display=y.style.display==='none'?'block':'none';};
window.saveYaml=async function(){const j=await post('/api/bio/save',{name:specName,raw:$('yamltext').value});if(j.error){status(j.error+(j.detail?'\n'+j.detail:''),true);return;}status('saved; seeding...');await reseed();};
let inflight=false, dirty=false;
async function render(force){if(inflight){dirty=true;return;}inflight=true;try{const r=await fetch(`/api/bio/render?azim=${CAM.azim}&elev=${CAM.elev}&zoom=${CAM.zoom}${FRAME===null?'':'&frame='+FRAME}&t=${Date.now()}`);if(r.ok){const b=await r.blob();const u=URL.createObjectURL(b);const im=$('view');const old=im.src;im.src=u;if(old.startsWith('blob:'))URL.revokeObjectURL(old);}else if(force){status((await r.json()).error||'render failed',true);}}catch(e){}finally{inflight=false;if(dirty){dirty=false;render();}}}
const im=$('view');let drag=null;
im.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,moved:false};im.style.cursor='grabbing';});
window.addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>2)drag.moved=true;CAM.azim-=dx*0.4;CAM.elev=Math.max(-89,Math.min(89,CAM.elev+dy*0.4));drag.x=e.clientX;drag.y=e.clientY;render();});
window.addEventListener('mouseup',async e=>{if(!drag)return;const moved=drag.moved;drag=null;im.style.cursor='grab';if(moved)return;
 const r=im.getBoundingClientRect();const fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;if(fx<0||fx>1||fy<0||fy>1)return;
 const j=await (await fetch(`/api/bio/pick?x=${fx.toFixed(4)}&y=${fy.toFixed(4)}`)).json();if(j.pick){showInfo(j.info);render();}else{$('info').textContent='nothing under the click';}});
im.addEventListener('wheel',e=>{e.preventDefault();CAM.zoom=Math.max(0.05,Math.min(60,CAM.zoom*(e.deltaY>0?1/1.08:1.08)));render();},{passive:false});
function showInfo(i){if(!i){$('info').textContent='(no object)';return;}
 $('info').textContent=`${i.species||i.set} particle #${i.index} (set ${i.set})\n  position: ${(i.position||[]).map(v=>v.toFixed ? v.toFixed(4) : v).join(', ')}\n  body: #${i.parent_cell}`;}
function tree(j){const h=j.hierarchy;let out='';for(const n of h.sets){const cnt=j.sets[n.name]?`${j.sets[n.name].n_live} live / ${j.sets[n.name].n_buffer}`:'';
 let rel='';if(n.parent)rel+=` <span class="cont">contained in ${n.parent}</span>`;
 out+=`<div onclick="window.setInfo('${n.name}')"><span class="n">${n.name}</span> ${cnt}${rel}${n.types.length?' <span class="cont">bodies: '+n.types.join(', ')+'</span>':''}</div>`;}
 out+=`<div style="color:#778;margin-top:4px">schedule: ${h.schedule.map(x=>typeof x==='string'?x:'{substeps: '+(x.steps||[]).join(' > ')+'}').join(' > ')}</div>`;$('tree').innerHTML=out;}
window.setInfo=function(name){const s=SCENE.sets[name];const n=SCENE.hierarchy.sets.find(x=>x.name===name);
 $('info').textContent=`set ${name}\n  entity: ${n.entity||'(by name)'}\n  buffer ${s.n_buffer}, live ${s.n_live}\n  blocks: ${s.blocks.join(', ')}\n`+(n.parent?`  contained in: ${n.parent} (${n.per_parent??'?'} per parent)\n`:'')+(n.types.length?`  bodies: ${n.types.join(', ')}\n`:'')+`  operators on it: ${SCENE.hierarchy.operators.filter(o=>o.at===name).map(o=>o.op).join(', ')||'-'}`;};
addBody({name:'elastic',shape:'ball',centre:[0.15,0.31,0.25],material:'elastic',youngs:2000000,density:1000});addBody({name:'liquid',shape:'ball',centre:[0.25,0.37,0.25],material:'liquid',youngs:981000,density:1000});addBody({name:'snow',shape:'ball',centre:[0.35,0.29,0.25],material:'snow',youngs:500000,density:350});
let running=false;
window.runGo=async function(){playStop();showMovie(null);FRAME=null;if(!specName){$('runstat').textContent='build a scene first';return;}const j=await post('/api/bio/run',{device:$('run_device').value});if(j.error){$('runstat').textContent=j.error;return;}running=true;$('runbtn').disabled=true;$('runstat').textContent=`generate ${specName} on ${j.device}...`;MOVIE=null;rpoll();};
window.runStop=async function(){await post('/api/bio/run',{stop:true});};
// THE RUN IS plexus.pipeline.generate -- the body of Plexus_Main.py -o generate -- on the server's
// VTK thread; its per-frame hook feeds this page's renderer and answers camera moves between
// frames, so orbit and zoom work WHILE the movie is written to graphs_data/studio/<name>/.
// Every movie frame is also kept as a level state, so PLAY replays the run at any camera.
function showMovie(url){const v=$('movie');if(url){v.src=url;v.style.display='block';$('view').style.display='none';}else{v.pause();v.style.display='none';$('view').style.display='';}}
async function rpoll(){try{const j=await (await fetch('/api/bio/run')).json();if(j.error&&!j.running){$('runstat').textContent='error: '+j.error;}
 else $('runstat').textContent=(j.running?'running: ':(j.stopped?'stopped: ':'done: '))+`frame ${j.frame}/${j.n_frames}, ${j.seconds}s`+(j.ms_per_frame?` (${j.ms_per_frame.toFixed(0)} ms/frame, the engine's own clock)`:(j.frame&&j.seconds?` (${(j.seconds/j.frame*1000).toFixed(0)} ms/frame incl. the movie)`:''));
 if(j.counts&&j.counts.sets)$('runcounts').textContent=Object.entries(j.counts.sets).map(([k,v])=>`${k} ${v}`).join('  ');
 render();if(j.running){setTimeout(rpoll,700);}else{running=false;$('runbtn').disabled=false;nframes=j.frames_kept||0;$('frame').max=Math.max(nframes-1,0);
  const a=await (await fetch('/api/material/run?name='+encodeURIComponent(specName))).json();MOVIE=a.mp4||null;
  $('framelab').textContent=(nframes?`${nframes} frames kept (every ${j.keep_every||1}): PLAY replays at any camera`:'')+(MOVIE?`; MOVIE plays ${a.dir}/movie.mp4`:'');}}catch(e){setTimeout(rpoll,1500);}}
let MOVIE=null, playing=null, nframes=0, FRAME=null;
async function showFrame(i){FRAME=i;$('frame').value=i;$('framelab').textContent=`frame ${i}/${Math.max(nframes-1,0)}`;await render();}
window.playGo=async function(){showMovie(null);const j=await (await fetch('/api/bio/frames')).json();nframes=j.n||0;if(!nframes){$('framelab').textContent='no frames yet: RUN first';return;}$('frame').max=nframes-1;playing=true;let i=0;
 while(playing){await showFrame(i);i=(i+1)%nframes;await new Promise(r=>setTimeout(r,30));}};
window.playStop=function(){playing=null;const v=$('movie');if(v.style.display!=='none')v.pause();};
window.movieGo=function(){if(!MOVIE){$('framelab').textContent='no movie yet: RUN first';return;}playing=null;showMovie(MOVIE);const v=$('movie');v.loop=true;v.play();};
let seen={version:-1,cam_version:-1};
async function poll(){try{const st=await (await fetch('/api/bio/state')).json();
 if(st.name&&st.version!==seen.version){seen.version=st.version;specName=st.name;$('name').value=st.name;const j=await (await fetch('/api/material/spec?name='+encodeURIComponent(st.name))).json();if(j.raw)$('yamltext').value=j.raw;if(j.form)fillForm(j.form);await reseed();}
 if(st.cam_version!==seen.cam_version){seen.cam_version=st.cam_version;CAM.azim=st.azim;CAM.elev=st.elev;CAM.zoom=st.zoom;if(st.pick){const j=await (await fetch('/api/bio/info?pick='+encodeURIComponent(st.pick))).json();if(!j.error)showInfo(j);}render();}
 if(st.message)$('rstat').textContent=st.message;}catch(e){}finally{setTimeout(poll,1500);}}
poll();
let cseen=0;
window.claudeGo=async function(){const t=$('task').value.trim();if(!t)return;$('claude').textContent='';cseen=0;const j=await post('/api/bio/claude',{task:t,mode:'material'});if(j.error){$('cstat').textContent=j.error;return;}$('cstat').textContent='running...';$('cbtn').disabled=true;};
window.claudeStop=async function(){await post('/api/bio/claude',{stop:true});};
window.claudeNew=async function(){await post('/api/bio/claude',{new_session:true});$('claude').textContent+='[new session]\n';};
async function cpoll(){try{const j=await (await fetch('/api/bio/claude?since='+cseen)).json();if(j.lines&&j.lines.length){const el=$('claude');el.textContent+=j.lines.join('\n')+'\n';el.scrollTop=el.scrollHeight;cseen=j.n;}
 $('cstat').textContent=j.running?'running... '+j.seconds+'s':(j.error?'error: '+j.error.slice(0,200):(j.n?'done in '+j.seconds+'s':''));$('cbtn').disabled=!!j.running;$('cbtn').classList.toggle('on',!!j.running);}catch(e){}finally{setTimeout(cpoll,1200);}}
cpoll();
const q=new URLSearchParams(location.search);if(q.get('name')){specName=q.get('name');fetch('/api/material/spec?name='+encodeURIComponent(specName)).then(r=>r.json()).then(j=>{if(j.raw){$('yamltext').value=j.raw;}if(j.form)fillForm(j.form);reseed();});}
// THE SCENE IS THERE WHEN THE PAGE OPENS: a fresh server has no spec, so the default form is built and
// seeded at once; a server that already holds one is picked up by poll() instead.
else{fetch('/api/bio/state').then(r=>r.json()).then(st=>{if(!st.name)build();}).catch(()=>{});}
</script></body></html>
"""


def page() -> str:
    return PAGE
