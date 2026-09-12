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

import numpy as np
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


def _twenty_balls(seed: int = 1, radius: float = 0.03, world: float = 0.5):
    """Twenty balls thrown into the box at random -- positions drawn without overlap (centres at
    least 2.4 radii apart, clear of the walls, in the upper two thirds so they fall), the three
    materials in turn, one colour each (tab20), the viscosity log-spaced from 0.001 to 0.1 Pa s.
    The launch speed of the form gives each its own random velocity (`vel_init`)."""
    import matplotlib
    cm = matplotlib.colormaps["tab20"]
    rng = np.random.RandomState(seed)
    mats = ({"material": "elastic", "youngs": 2000000.0, "density": 1000.0},
            {"material": "liquid", "bulk_modulus": 981000.0, "density": 1000.0},
            {"material": "snow", "youngs": 500000.0, "density": 350.0})
    lo, hi = radius + 0.02, world - radius - 0.02
    centres = []
    while len(centres) < 20:
        c = [rng.uniform(lo, hi), rng.uniform(0.35 * world, hi), rng.uniform(lo, hi)]
        if all(np.linalg.norm(np.subtract(c, o)) >= 2.4 * radius for o in centres):
            centres.append([round(float(v), 3) for v in c])
    out = []
    for i in range(20):
        m = dict(mats[i % 3])
        m["eta"] = round(float(10 ** (-3 + 2 * i / 19)), 5)
        out.append({"name": f"b{i:02d}", "shape": "ball", "centre": centres[i],
                    "color": [round(float(v), 3) for v in cm(i % 20)[:3]], **m})
    return out


def _multimaterial_27(world: float = 0.5, side: float = 0.04):
    """The twin of MPM_pytorch's `config/multimaterial/multimaterial_1_3D.yaml`: 27 cubes on a
    3 x 3 x 3 lattice falling and bouncing in a box, one colour per cube, the points of each cube
    on a lattice (`fill: lattice`) so a cube reads as a solid. The reference is 35,937 points over 27 cubes in a unit box under g = 20; here the
    box is 0.5 m, g = 9.81 and the form's `particles` sets the points per cube."""
    import matplotlib
    # BOUNCING CUBES, as the reference shows them: every cube elastic (the reference's jelly),
    # E = 50 kPa, one colour each; the material menu of the form can still change any of them.
    mats = ({"material": "elastic", "youngs": 50000.0, "density": 1000.0},)   # 50 kPa: a soft jelly, visibly squashed by its own landing
    # ONE COLOUR PER CUBE, as the reference draws them (tab10 per object): 27 distinct hues from
    # tab20 and tab20b, so neighbouring cubes of one material are still telling apart.
    cm_a, cm_b = matplotlib.colormaps["tab20"], matplotlib.colormaps["tab20b"]
    pitch = 0.5 * world / 3.0                         # the lattice spans the middle half of the box
    x0 = 0.5 * world - 1.5 * pitch + 0.5 * (pitch - side)
    out = []
    i = 0
    for iy in range(3):
        for iz in range(3):
            for ix in range(3):
                a = [round(x0 + ix * pitch, 4), round(0.40 * world + iy * pitch, 4), round(x0 + iz * pitch, 4)]
                col = (cm_a(i) if i < 20 else cm_b(i - 20))[:3]
                out.append({"name": f"c{i:02d}", "shape": "block", "block": [a[0], a[1], a[2], round(a[0] + side, 4),
                                                                              round(a[1] + side, 4), round(a[2] + side, 4)],
                            "color": [round(float(v), 3) for v in col], "fill": "lattice", **dict(mats[i % len(mats)])})
                i += 1
    return out


DEFAULT_FORM = {
    # dt 2 ms: 800 frames are 1.6 s, enough for the cubes to fall, bounce and settle; the reference
    # runs 10,000 frames of 0.1 ms in a unit box under g = 20. `launch` 1.0 is its `dpos_init`,
    # a random initial velocity per cube.
    "name": "si_multimaterial_27", "world": 0.5, "n_grid": 96, "n_frames": 800, "dt": 0.002,
    "gravity": 9.81, "particles": 1331, "radius": 0.03, "wall_damp": 0.9, "friction": 0.2,
    "launch": 1.0, "movie_frames": 400, "stills": 10, "seed": 1, "render": "middle_splats", "light": "default",
    "color": "particles", "bodies": _multimaterial_27(),
}
REFERENCE = os.path.join(REPO, "config", "si_material", "si_multimaterial_27.yaml")

# THE RENDER MENU, as the plotting keys each entry sets (`live_movie.py`). Three dot sizes, flat
# pixels; `surface` reconstructs an iso-surface per body each frame (`render_3d: contour`) with a
# diffuse, matte coat -- shading only; `surface_specular` is the same surface with a low
# roughness, the white-sky reflection and VTK's shadow pass; `glassy` is the dielectric under
# the white sky at half opacity.
RENDER_KEYS = ("render_3d", "dot_size", "dot_shading", "dot_specular", "dot_specular_power", "contour_by_type",
               "surface_env", "surface_env_color", "surface_opacity", "surface_roughness", "surface_metallic",
               "surface_pbr", "shadows", "light", "edl", "contour_ngrid", "contour_smooth")
RENDER_MODES = ("small_splats", "middle_splats", "large_splats", "surface", "surface_specular", "glassy")
LIGHT_MODES = ("default", "headlight", "sun", "studio", "flat")
# FLAT PIXELS ARE STILL A RENDER, just not a menu entry: `render_3d: dots` with a `dot_size` is
# what a hand-written spec uses and what `render_of` reads back when one is opened. The menu lists
# splats only -- a lit sphere imposter at the same cost, which the three dot sizes were the poor
# version of.
DOT_PX = {"small_dots": 1.0, "middle_dots": 2.0, "large_dots": 4.0}
# THE SPLAT SIZES, in pixels: a lit sphere imposter per point, so the size is the ball's diameter
# on screen and the three sizes are the three densities a body is worth drawing at.
SPLAT_PX = {"small_splats": 4.0, "middle_splats": 7.0, "large_splats": 11.0}


def render_style(mode: str = "small_dots", light: str = "default") -> dict:
    mode = str(mode or "small_dots").lower()
    light = str(light or "default").lower()
    if mode not in RENDER_MODES:
        raise ValueError(f"render must be one of {RENDER_MODES}")
    if light not in LIGHT_MODES:
        raise ValueError(f"light must be one of {LIGHT_MODES}")
    if mode in SPLAT_PX:
        # THE ONE IN BETWEEN: big dots drawn as lit sphere imposters (VTK's point sprites,
        # `render_points_as_spheres` + `dot_shading: true`) -- each point a shaded ball at the cost
        # of a point, so a body reads as a solid without a contour's density grid. (The gaussian
        # splat mapper and eye-dome lighting were tried first: the mapper drew nothing through
        # pyvista's rgb path and EDL darkened the whole off-screen frame.)
        st = {"render_3d": "dots", "dot_size": SPLAT_PX[mode], "dot_shading": True}
    elif mode in DOT_PX:
        st = {"render_3d": "dots", "dot_size": DOT_PX[mode]}
    else:
        # THE SURFACE, AT PAGE SPEED: a 96^3 density grid and 8 smoothing passes where the movie's
        # default is 176^3 and 35 -- a coarser skin, a few times faster to rebuild per frame.
        st = {"render_3d": "contour", "contour_by_type": True, "surface_metallic": 0.0,
              "contour_ngrid": 96, "contour_smooth": 8}
    if light != "default":
        st["light"] = light
    if mode in DOT_PX or mode in SPLAT_PX:
        # A LIGHT ON DOTS: flat pixels take no light, so any light but the default (and `flat`)
        # draws the dots as lit spheres (`dot_shading: true`), which is what a light can act on.
        if light not in ("default", "flat") and mode in DOT_PX:
            st["dot_shading"] = True
        return st
    if mode == "surface":
        st.update(surface_env=False, surface_pbr=False, surface_opacity=1.0, surface_roughness=1.0, shadows=False)
    elif mode == "surface_specular":
        st.update(surface_env=True, surface_env_color="white", surface_opacity=1.0, surface_roughness=0.1, shadows=True)
    else:
        st.update(surface_env=True, surface_env_color="white", surface_opacity=0.5, surface_roughness=0.07)
    return st


# THE COLOUR MENU: what a point is coloured by. `particles` is the body's own colour; the others
# are the renderer's per-particle fields (`plotting.color_field`, live_movie.py: `deformation` is
# the equivalent deviatoric strain off F, `pressure` is K(1 - J), `speed` is |v|), each on a fixed
# colour range settled on the first frame so two frames are comparable.
COLOR_KEYS = ("color_field", "field_cmap", "field_log")
COLOR_MODES = {"particles": None, "deformation": "deformation", "stress": "pressure", "velocities": "speed"}


def apply_color(plotting: dict, color: str = "particles") -> dict:
    color = str(color or "particles").lower()
    if color not in COLOR_MODES:
        raise ValueError(f"color must be one of {tuple(COLOR_MODES)}")
    out = {k: v for k, v in (plotting or {}).items() if k not in COLOR_KEYS}
    if COLOR_MODES[color]:
        # BLUE - WHITE - RED for a mechanical field (`coolwarm`): the eye reads white as the
        # resting value and the two ends as the two ways to leave it, which is what deformation
        # and pressure are. Speed has no such middle, so it keeps a sequential map.
        out.update(color_field=COLOR_MODES[color],
                   field_cmap=("turbo" if color == "velocities" else "coolwarm"))
    return out


def color_of(plotting: dict) -> str:
    cf = str((plotting or {}).get("color_field", "") or "").lower()
    return next((k for k, v in COLOR_MODES.items() if v == cf), "particles")


def apply_render(plotting: dict, mode: str, light: str = "default", color: str | None = None) -> dict:
    """The plotting block with its render keys replaced by `mode`'s -- the other keys untouched."""
    out = {k: v for k, v in (plotting or {}).items() if k not in RENDER_KEYS}
    out.update(render_style(mode, light))
    if color is not None:
        out = apply_color(out, color)
    return out


def render_of(plotting: dict) -> str:
    """The menu entry read back off a plotting block."""
    pl = plotting or {}
    if str(pl.get("render_3d", "dots")).lower() == "contour":
        if float(pl.get("surface_opacity", 1.0)) < 1.0:
            return "glassy"
        return "surface_specular" if pl.get("surface_env") else "surface"
    px = float(pl.get("dot_size", 1.0) or 1.0)
    table = SPLAT_PX if pl.get("dot_shading") is True else DOT_PX
    return min(table, key=lambda k: abs(table[k] - px))


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
        if shape == "block" and str(b.get("fill", "")).lower() == "lattice":
            t["fill"] = "lattice"
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
                                  "camera_turns": 0.0, "camera_zoom": 0.0,
                                  "max_frames": int(form.get("movie_frames", 400)), "stills": int(form.get("stills", 10)),
                                  "keep_stills": True, "splat_res": 600, "box_frame": True, "hide_sets": ["cell"],
                                  "floor": "#0b1a3a", "floor_opacity": 0.25, "colors": colors, "slow_motion": 4},
                                 str(form.get("render", "small_dots")), str(form.get("light", "default")),
                                 str(form.get("color", "particles"))),
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
            if t.get("fill"):
                b["fill"] = t["fill"]
        else:
            b.update(shape="ball", centre=(starts[i] if i < len(starts) else None))
        bodies.append(b)
    gen = spec.get("general") or {}
    ops = spec.get("operators") or []
    g = next((o.get("g", 9.81) for o in ops if o.get("op") == "gravity"), 9.81)
    gu = next((o for o in ops if o.get("op") == "mpm_grid_update"), {})
    pl = spec.get("plotting") or {}
    return {"name": gen.get("name", ""), "world": (gen.get("world") or [0.5])[0], "n_frames": gen.get("n_frames", 800),
            "render": render_of(pl), "light": str(pl.get("light", "default")), "color": color_of(pl),
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
                          render (small_splats|middle_splats|large_splats|surface|surface_specular|glassy),
                          light (default|headlight|sun|studio|flat), color (particles|deformation|stress|velocities).
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

 <div class="row"><label>name</label><input id="name" value="si_multimaterial_27"></div>
 <div class="row"><label>box side (m)</label><input id="world" class="short" value="0.5"> <label style="width:60px">grid</label><input id="n_grid" class="short" value="96"></div>
 <div class="row"><label>frames</label><input id="n_frames" class="short" value="800"> <label style="width:60px">dt (s)</label><input id="dt" class="short" value="0.002"></div>
 <div class="row"><label>gravity</label><input id="gravity" class="short" value="9.81"> <label style="width:60px">particles</label><input id="particles" class="short" value="1331" title="material points per body"></div>
 <div class="row"><label>ball radius (m)</label><input id="radius" class="short" value="0.03" title="every ball's radius; a block is sized by its own box"> <label style="width:60px">launch</label><input id="launch" class="short" value="1.0" title="initial speed given to each body, m/s (0 = dropped from rest)"></div>
 <div class="row"><label>wall damp</label><input id="wall_damp" class="short" value="0.9" title="share of the wall-normal velocity kept at the grid: 1 = elastic wall, 0 = dead"> <label style="width:60px">friction</label><input id="friction" class="short" value="0.2" title="Coulomb friction on the walls, 0 = slippery"></div>
 <div class="row"><label title="frames the movie keeps: every frame up to this many, then every 2nd, 4th...">movie frames</label><input id="movie_frames" class="short" value="400"> <label style="width:60px" title="PNG stills dropped through the run, also the live pictures on this page">stills</label><input id="stills" class="short" value="10"> <label style="width:40px">seed</label><input id="seed" class="short" value="1"></div>
 <div class="row"><button class="dim" onclick="toggleBodies()" id="bodiesbtn">show bodies</button><button class="dim" onclick="addBody()">+ body</button> <span id="bodycount" style="color:#9ab"></span></div>
 <div id="bodieswrap" style="display:none">
 <table class="sp" id="bodies"><tr><th>name</th><th>shape</th><th>centre x y z | block x0 y0 z0 x1 y1 z1</th><th>material</th><th>stiffness</th><th>density</th><th title="viscosity, Pa s">eta</th><th title="r g b in 0-1, blank = automatic">colour</th><th></th></tr></table>
 <div style="color:#778;font-size:11px">a ball is placed at its centre with the shared radius above; a block spans its six numbers (metres). Bodies keep their order: the first body is at the first centre. stiffness = Young's modulus (elastic, snow) or bulk modulus (liquid), Pa; eta the viscosity, Pa s; colour r g b in 0-1 (blank = automatic).</div>
 </div>

 <div class="row"><label>render</label><select id="render" style="width:130px" onchange="setStyle()"><option value="small_splats">small splats</option><option value="middle_splats">middle splats</option><option value="large_splats">large splats</option><option value="surface">surface</option><option value="surface_specular">surface specular</option><option value="glassy">glassy</option></select> <label style="width:40px">light</label><select id="light" style="width:100px" onchange="setStyle()"><option value="default">default</option><option value="headlight">headlight</option><option value="sun">sun</option><option value="studio">studio</option><option value="flat">flat</option></select></div>
 <div class="row"><label>colour</label><select id="color" style="width:130px" onchange="setStyle()"><option value="particles">particles</option><option value="deformation">deformation</option><option value="stress">stress</option><option value="velocities">velocities</option></select> <span style="color:#778;font-size:11px">render, light and colour apply now and to the movie</span></div>
'''

FORM_JS = r'''
window.addBody=function(b){b=b||{};const tb=$('bodies');const tr=tb.insertRow(-1);const geo=b.shape==='block'?(b.block||[0,0,0,0.5,0.125,0.5]).join(' '):(b.centre||[0.25,0.375,0.25]).join(' ');
 tr.innerHTML=`<td><input value="${b.name||''}"></td><td><select><option>ball</option><option>block</option></select></td><td><input value="${geo}"></td><td><select><option>elastic</option><option>liquid</option><option>snow</option></select></td><td><input value="${b.youngs??b.bulk_modulus??2000000}"></td><td><input value="${b.density??1000}"></td><td><input value="${b.eta??''}" placeholder="-"></td><td><input value="${(b.color||[]).join(' ')}" placeholder="auto"></td><td><button class="dim" onclick="this.closest('tr').remove();bodyCount()">x</button></td>`;
 tr.cells[1].firstChild.value=b.shape||'ball';tr.cells[3].firstChild.value=b.material||'elastic';bodyCount();};
function bodies(){const out=[];for(const tr of $('bodies').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 const nums=c[2].firstChild.value.trim().split(/[\s,]+/).map(Number);const shape=c[1].firstChild.value;const mat=c[3].firstChild.value;
 const b={name,shape,material:mat,density:+c[5].firstChild.value};if(shape==='ball'){b.centre=nums.slice(0,3);}else{b.block=nums.slice(0,6);b.fill='lattice';}
 if(mat==='liquid')b.bulk_modulus=+c[4].firstChild.value;else b.youngs=+c[4].firstChild.value;
 const eta=c[6].firstChild.value.trim();if(eta)b.eta=+eta;const col=c[7].firstChild.value.trim();if(col)b.color=col.split(/[\s,]+/).map(Number);out.push(b);}return out;}
// THE LIST IS FOLDED: twenty bodies are a page of rows and a hundred are a scroll; the count says
// what is there and SHOW opens the table when a row needs editing.
window.bodyCount=function(){const bs=bodies();const per={};for(const b of bs)per[b.material]=(per[b.material]||0)+1;$('bodycount').textContent=bs.length?`${bs.length} bodies: `+Object.entries(per).map(([k,v])=>`${v} ${k}`).join(', '):'(none)';};
window.toggleBodies=function(){const w=$('bodieswrap');const on=w.style.display==='none';w.style.display=on?'block':'none';$('bodiesbtn').textContent=on?'hide bodies':'show bodies';};
const NUM=['world','n_grid','n_frames','dt','gravity','particles','radius','launch','wall_damp','friction','movie_frames','stills','seed'];
window.tabForm=function(){const f={name:$('name').value,bodies:bodies(),render:$('render').value,light:$('light').value,color:$('color').value};for(const k of NUM)f[k]=+$(k).value;return f;};
window.tabFill=function(f){$('name').value=f.name;for(const k of NUM)if(f[k]!==undefined&&f[k]!==null)$(k).value=Number(Number(f[k]).toPrecision(4));if(f.render)$('render').value=f.render;if(f.light)$('light').value=f.light;if(f.color)$('color').value=f.color;
 const tb=$('bodies');while(tb.rows.length>1)tb.deleteRow(-1);(f.bodies||[]).forEach(addBody);bodyCount();};
// THE RENDER SELECTOR WRITES THE SPEC'S plotting AND RE-SEEDS, so what the page shows is what the
// movie will draw; a BUILD keeps the choice because the form carries it.
window.setStyle=async function(){if(!specName)return;status('changing the render...');const j=await post('/api/scene/style',{name:specName,render:$('render').value,light:$('light').value,color:$('color').value});if(j.error){status(j.error,true);return;}if(j.version!==undefined)seen.version=j.version;if(j.raw)$('yamltext').value=j.raw;await reseed();};
window.tabInit=function(){for(const b of DEFAULT_BODIES)addBody(b);};
'''
