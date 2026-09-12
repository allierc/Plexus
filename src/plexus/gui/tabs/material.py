"""The material tab: MPM bodies in a box, as a form that writes the spec a person would write.

The twin of `gui/bio.py` for the material side of Plexus. A scene is a list of BODIES (a ball
or a block of one material: elastic, liquid or snow, with its stiffness and density) in a walled
box with gravity. The form WRITES THE SPEC A PERSON WOULD WRITE -- `general / sets / fields /
operators / schedule / plotting`, the shape of `config/si_material/si_three_balls.yaml`, which is
itself this form's default scene written out -- rather than patching a template: a template
carries its own numbers (its substep, its wall model) and those were what made the page run a
different simulation from the file with the same name. The picture, the picking, the engine run
and the Claude drive are the shared panel's (`gui/app.py`, `gui/bio_view.py`, `/api/scene/*`).

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
REFERENCE = os.path.join(REPO, "config", "si_material", "si_twenty_balls.yaml")

NAME, TITLE = "material", "Plexus material"
PICK_DIR = os.path.join(REPO, "config", "si_material")
CORPUS_MODE = "material"


def _twenty_balls():
    """Twenty balls on a 5 x 4 grid, the three materials in turn, one colour each (tab20), the
    viscosity climbing across the grid so neighbouring balls of one material differ in it."""
    import matplotlib
    cm = matplotlib.colormaps["tab20"]
    mats = ({"material": "elastic", "youngs": 2000000.0, "density": 1000.0},
            {"material": "liquid", "bulk_modulus": 981000.0, "density": 1000.0},
            {"material": "snow", "youngs": 500000.0, "density": 350.0})
    out = []
    for i in range(20):
        c, r = i % 5, i // 5
        m = dict(mats[i % 3])
        # 0.001 (water) up to 0.1 Pa s, log-spaced over the twenty, so a liquid's neighbours
        # in the grid are the same material at a different viscosity
        m["eta"] = round(float(10 ** (-3 + 2 * i / 19)), 5)
        out.append({"name": f"b{i:02d}", "shape": "ball", "centre": [round(0.09 + 0.08 * c, 3), round(0.40 - 0.08 * r, 3), 0.25],
                    "color": [round(float(v), 3) for v in cm(i % 20)[:3]], **m})
    return out


DEFAULT_FORM = {
    "name": "si_twenty_balls", "world": 0.5, "n_grid": 96, "n_frames": 800, "dt": 0.0008333333333333334,
    "gravity": 9.81, "particles": 500, "radius": 0.03, "wall_damp": 0.5, "friction": 0.4,
    "launch": 0.4, "movie_frames": 400, "stills": 10, "seed": 1, "render": "dots", "specular": 0,
    "bodies": _twenty_balls(),
}

# THE RENDER MODES OF THE POINT RENDERER, as the plotting keys each one sets (`live_movie.py`).
# `dots` is flat pixels; `shaded` lights each dot by its depth in the body (`dot_shading: body`);
# `lit` is VTK's own sphere lighting (`dot_shading: true`); `surface` reconstructs an iso-surface
# per body each frame (`render_3d: contour`); `glassy` is the same surface as a dielectric under a
# white sky. `specular` adds the highlight to whichever of them has one.
RENDER_KEYS = ("render_3d", "dot_shading", "dot_specular", "dot_specular_power", "contour_by_type",
               "surface_env", "surface_env_color", "surface_opacity", "surface_roughness")
RENDER_MODES = ("dots", "shaded", "lit", "surface", "glassy")


def render_style(mode: str = "dots", specular: bool = False) -> dict:
    mode = str(mode or "dots").lower()
    if mode not in RENDER_MODES:
        raise ValueError(f"render must be one of {RENDER_MODES}")
    st = {"render_3d": "dots"}
    if mode == "shaded":
        st["dot_shading"] = "body"
    elif mode == "lit":
        st["dot_shading"] = True
    elif mode in ("surface", "glassy"):
        st.update(render_3d="contour", contour_by_type=True, surface_roughness=(0.07 if specular else 0.5))
        if mode == "glassy":
            st.update(surface_env=True, surface_env_color="white", surface_opacity=0.5)
        else:
            st.update(surface_env=False, surface_opacity=1.0)
    if specular and mode in ("shaded", "lit"):
        st.update(dot_specular=0.4, dot_specular_power=12)
    return st


def apply_render(plotting: dict, mode: str, specular: bool) -> dict:
    """The plotting block with its render keys replaced by `mode`'s -- the other keys untouched."""
    out = {k: v for k, v in (plotting or {}).items() if k not in RENDER_KEYS}
    out.update(render_style(mode, specular))
    return out


def render_of(plotting: dict) -> tuple:
    """(mode, specular) read back off a plotting block."""
    pl = plotting or {}
    if str(pl.get("render_3d", "dots")).lower() == "contour":
        return ("glassy" if pl.get("surface_env") else "surface"), float(pl.get("surface_roughness", 0.5)) < 0.2
    sh = pl.get("dot_shading", None)
    mode = "lit" if sh is True else ("shaded" if str(sh).lower() == "body" else "dots")
    return mode, float(pl.get("dot_specular", 0.0) or 0.0) > 0


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
        if b.get("eta") is not None:
            t["eta"] = float(b["eta"])                       # per-body viscosity, Pa s (overrides mpm_viscosity's)
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
        if b.get("color"):
            colors[nm] = [float(v) for v in b["color"]]
        elif len(bodies) <= len(DEFAULT_COLORS):
            colors[nm] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        else:                                                # more bodies than named colours: tab20
            import matplotlib
            colors[nm] = [round(float(v), 3) for v in matplotlib.colormaps["tab20"](i % 20)[:3]]
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
        "plotting": apply_render({"renderer": "vtk_points", "background": "black", "up_axis": 1, "camera_elev": 1.18,
                                  "camera_turns": 0.0, "camera_zoom": 0.0, "dot_size": 1.6,
                                  "max_frames": int(form.get("movie_frames", 400)), "stills": int(form.get("stills", 10)),
                                  "keep_stills": True, "splat_res": 600, "box_frame": True, "hide_sets": ["cell"],
                                  "colors": colors, "slow_motion": 4},
                                 str(form.get("render", "dots")), bool(int(form.get("specular", 0) or 0))),
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
        if t.get("eta") is not None:
            b["eta"] = t["eta"]
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
    mode, spec_ = render_of(pl)
    return {"name": gen.get("name", ""), "world": (gen.get("world") or [0.5])[0], "n_frames": gen.get("n_frames", 800),
            "render": mode, "specular": int(spec_),
            "dt": gen.get("dt", 1.0 / 1200.0), "gravity": g, "wall_damp": gu.get("wall_damp", 0.5),
            "friction": gu.get("wall_friction", 0.0), "launch": mp.get("vel_init", 0.0), "seed": gen.get("seed", 1),
            "n_grid": ((spec.get("fields") or {}).get("mpm_grid") or {}).get("n_grid", 96),
            "particles": mp.get("per_parent", 100000), "radius": mp.get("radius", 0.05),
            "movie_frames": pl.get("max_frames", 400), "stills": pl.get("stills", 10), "bodies": bodies}


BRIEF = """You are driving the Plexus material page, a UI for DEFINING material bodies (balls and
slabs of elastic, liquid or snow material in a walled box under gravity) and running them with the
material point method. A person is watching the page: it follows every change you make, and your
words appear in a transcript beside the 3D scene. Say, in ONE short plain-English line before
each call, what you are about to do and why; say what you found after it. No headers, no markdown.

Your ONLY tool is `curl` against the local server at http://127.0.0.1:{port} . The routes:

You have curl, sleep and jq ONLY: no python, no ls, no files. Put the JSON body inline in `curl -d '...'`
(one line, however long); a body you cannot write inline you cannot send.

  POST /api/tab/material/build  JSON form -> writes and seeds a spec. Fields: name, world (box side,
                          metres), n_grid, n_frames, dt, gravity, particles (per body), radius
                          (every ball's radius, metres), wall_damp (0-1, wall-normal velocity kept
                          at the grid), friction (Coulomb, 0 = slippery), launch (initial speed
                          per body, m/s), movie_frames, stills, seed,
                          bodies: [{name, shape (ball|block), centre [x,y,z] for a ball,
                          block [x0,y0,z0,x1,y1,z1] for a slab, material (elastic|liquid|snow),
                          youngs (elastic/snow) or bulk_modulus (liquid), density}].
                          Bodies keep their order: the first body is at the first centre.
                          render (dots|shaded|lit|surface|glassy), specular (0|1).
                          A ball deforms visibly below ~30,000 Pa; 1,000,000 is rigid.
  POST /api/scene/refine    {name, prompt} -> an English edit of the current spec (another Claude
                          applies it; 20-40 s)
  GET  /api/scene/counts?name= -> live count per set and per body type
  GET  /api/scene/info?name=&pick=<set>:<index>   -> what one object is
  GET  /api/scene/view?azim=&elev=&zoom=&pick=&message=   -> turns the viewer's camera (degrees,
                          zoom 0.05-60), highlights a pick, shows `message`. Move in steps of 30
                          degrees or less with `sleep 1` between them.
  POST /api/scene/run       {frames, device} -> simulate from the seed, the picture following each
                          frame; {stop: true} aborts.  GET /api/scene/run -> progress and live counts
  GET  /api/scene/state     -> the session state

Rules: use only these routes (curl -s, JSON bodies with -H 'Content-Type: application/json'), with
jq and sleep as the only other commands; never touch files; keep the spec name given in the task
unless told otherwise; finish with two or three lines summarising the scene and the bodies.
"""




FORM_HTML = r'''
 <h2>Box</h2>
 <div class="row"><label>name</label><input id="name" value="si_twenty_balls"></div>
 <div class="row"><label>box side (m)</label><input id="world" class="short" value="0.5"> <label style="width:60px">grid</label><input id="n_grid" class="short" value="96"></div>
 <div class="row"><label>frames</label><input id="n_frames" class="short" value="800"> <label style="width:60px">dt (s)</label><input id="dt" class="short" value="0.00083"></div>
 <div class="row"><label>gravity</label><input id="gravity" class="short" value="9.81"> <label style="width:60px">particles</label><input id="particles" class="short" value="500" title="material points per body"></div>
 <div class="row"><label>ball radius (m)</label><input id="radius" class="short" value="0.03" title="every ball's radius; a block is sized by its own box"> <label style="width:60px">launch</label><input id="launch" class="short" value="0.4" title="initial speed given to each body, m/s (0 = dropped from rest)"></div>
 <div class="row"><label>wall damp</label><input id="wall_damp" class="short" value="0.5" title="share of the wall-normal velocity kept at the grid: 1 = elastic wall, 0 = dead"> <label style="width:60px">friction</label><input id="friction" class="short" value="0.4" title="Coulomb friction on the walls, 0 = slippery"></div>
 <div class="row"><label title="frames the movie keeps: every frame up to this many, then every 2nd, 4th...">movie frames</label><input id="movie_frames" class="short" value="400"> <label style="width:60px" title="PNG stills dropped through the run, also the live pictures on this page">stills</label><input id="stills" class="short" value="10"> <label style="width:40px">seed</label><input id="seed" class="short" value="1"></div>
 <h2>Bodies <button class="dim" onclick="toggleBodies()" id="bodiesbtn">show</button> <button class="dim" onclick="addBody()">+ body</button> <span id="bodycount" style="color:#9ab;font-weight:normal;text-transform:none"></span></h2>
 <div id="bodieswrap" style="display:none">
 <table class="sp" id="bodies"><tr><th>name</th><th>shape</th><th>centre x y z | block x0 y0 z0 x1 y1 z1</th><th>material</th><th>stiffness</th><th>density</th><th title="viscosity, Pa s">eta</th><th title="r g b in 0-1, blank = automatic">colour</th><th></th></tr></table>
 <div style="color:#778;font-size:11px">a ball is placed at its centre with the shared radius above; a block spans its six numbers (metres). Bodies keep their order: the first body is at the first centre. stiffness = Young's modulus (elastic, snow) or bulk modulus (liquid), Pa; eta the viscosity, Pa s; colour r g b in 0-1 (blank = automatic).</div>
 </div>
 <h2>Render</h2>
 <div class="row"><label>points</label><select id="render" style="width:100px" onchange="setStyle()"><option value="dots">dots (flat)</option><option value="shaded">dots, shaded</option><option value="lit">dots, lit</option><option value="surface">surface</option><option value="glassy">glassy</option></select> <label style="width:80px" title="a highlight on the shaded or lit dots; a smooth coat on a surface"><input type="checkbox" id="specular" style="width:auto" onchange="setStyle()"> specular</label> <span style="color:#778;font-size:11px">applies to the scene now and to the movie</span></div>
'''

FORM_JS = r'''
window.addBody=function(b){b=b||{};const tb=$('bodies');const tr=tb.insertRow(-1);const geo=b.shape==='block'?(b.block||[0,0,0,0.5,0.125,0.5]).join(' '):(b.centre||[0.25,0.375,0.25]).join(' ');
 tr.innerHTML=`<td><input value="${b.name||''}"></td><td><select><option>ball</option><option>block</option></select></td><td><input value="${geo}"></td><td><select><option>elastic</option><option>liquid</option><option>snow</option></select></td><td><input value="${b.youngs??b.bulk_modulus??2000000}"></td><td><input value="${b.density??1000}"></td><td><input value="${b.eta??''}" placeholder="-"></td><td><input value="${(b.color||[]).join(' ')}" placeholder="auto"></td><td><button class="dim" onclick="this.closest('tr').remove();bodyCount()">x</button></td>`;
 tr.cells[1].firstChild.value=b.shape||'ball';tr.cells[3].firstChild.value=b.material||'elastic';bodyCount();};
function bodies(){const out=[];for(const tr of $('bodies').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 const nums=c[2].firstChild.value.trim().split(/[\s,]+/).map(Number);const shape=c[1].firstChild.value;const mat=c[3].firstChild.value;
 const b={name,shape,material:mat,density:+c[5].firstChild.value};if(shape==='ball'){b.centre=nums.slice(0,3);}else{b.block=nums.slice(0,6);}
 if(mat==='liquid')b.bulk_modulus=+c[4].firstChild.value;else b.youngs=+c[4].firstChild.value;
 const eta=c[6].firstChild.value.trim();if(eta)b.eta=+eta;const col=c[7].firstChild.value.trim();if(col)b.color=col.split(/[\s,]+/).map(Number);out.push(b);}return out;}
// THE LIST IS FOLDED: twenty bodies are a page of rows and a hundred are a scroll; the count says
// what is there and SHOW opens the table when a row needs editing.
window.bodyCount=function(){const bs=bodies();const per={};for(const b of bs)per[b.material]=(per[b.material]||0)+1;$('bodycount').textContent=bs.length?`${bs.length} bodies: `+Object.entries(per).map(([k,v])=>`${v} ${k}`).join(', '):'(none)';};
window.toggleBodies=function(){const w=$('bodieswrap');const on=w.style.display==='none';w.style.display=on?'block':'none';$('bodiesbtn').textContent=on?'hide':'show';};
const NUM=['world','n_grid','n_frames','dt','gravity','particles','radius','launch','wall_damp','friction','movie_frames','stills','seed'];
window.tabForm=function(){const f={name:$('name').value,bodies:bodies(),render:$('render').value,specular:$('specular').checked?1:0};for(const k of NUM)f[k]=+$(k).value;return f;};
window.tabFill=function(f){$('name').value=f.name;for(const k of NUM)if(f[k]!==undefined&&f[k]!==null)$(k).value=f[k];if(f.render)$('render').value=f.render;if(f.specular!==undefined)$('specular').checked=!!f.specular;
 const tb=$('bodies');while(tb.rows.length>1)tb.deleteRow(-1);(f.bodies||[]).forEach(addBody);bodyCount();};
// THE RENDER SELECTOR WRITES THE SPEC'S plotting AND RE-SEEDS, so what the page shows is what the
// movie will draw; a BUILD keeps the choice because the form carries it.
window.setStyle=async function(){if(!specName)return;status('changing the render...');const j=await post('/api/scene/style',{name:specName,render:$('render').value,specular:$('specular').checked?1:0});if(j.error){status(j.error,true);return;}if(j.version!==undefined)seen.version=j.version;if(j.raw)$('yamltext').value=j.raw;await reseed();};
window.tabInit=function(){for(const b of DEFAULT_BODIES)addBody(b);};
'''
