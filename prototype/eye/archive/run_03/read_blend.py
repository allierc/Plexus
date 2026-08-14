"""read_blend -- open the Blender eye/muscle model and cut it into named parts.

`260802_s2_EYE_MUSCLES_MODEL.blend` is a 96 hpf zebrafish head: 30 cartilage plates,
the notochord, the CNS, one eye (cornea + lens + retina) and the six extraocular
muscles (`Cylinder.001..006`, one flat colour each -- the blue/violet bands).

TWO THINGS ABOUT THE FILE decide how it has to be read.

  1. ONLY THE LEFT HALF IS MODELLED. Every eye part and every muscle carries a
     MIRROR modifier across x = 0 (object origins are all at the world origin, so
     the mirror plane really is the midsagittal plane, and no vertex is merged
     across it because no part touches x = 0). The right eye and its six muscles
     therefore exist ONLY in the evaluated mesh -- reading `object.data` gives you
     six muscles, not twelve. We evaluate the depsgraph, then split each evaluated
     mesh into loose parts and label the part by the sign of its centroid's x.
  2. EVERYTHING ELSE CARRIES A SUBSURF (viewport level 1, render 2). We turn it off
     by default so the vertices you get are the artist's cage -- 1120 per muscle
     rather than ~4500 -- and back on with `--subsurf`.

THE FRAME. The blend is in head coordinates: +x is to the animal's right-hand side
(the modelled half is x < 0), +y is CAUDAL (the CNS runs from y = 1.1 at the
forebrain to y = 19.2 in the tail), +z is DORSAL. `fish_anatomy` uses a per-eye
frame instead -- (+x caudal, +y dorsal, +z LATERAL along the optic axis) -- so for
each side we build that frame here and report every muscle measurement in it:

    lateral  = unit(cornea centroid - globe centre)      # out of the head
    dorsal   = +z of the head, orthogonalised against lateral
    caudal   = +y of the head, orthogonalised against both

The two eyes are enantiomorphs, so this anatomical triad is right-handed on one
side and left-handed on the other; `frame_det` in the manifest says which, and any
TORSION sign taken from it flips between the sides.

NAMING THE SIX MUSCLES. The blend calls them `Cylinder.00N`; the model calls them
LR/SR/MR/IR/SO/IO (`fish_anatomy.MUSCLE_KEYS`). We do not guess from the names or
the material colours -- we measure, per muscle: which end is on the globe (the
INSERTION, the end nearest the globe surface) and which is on cartilage (the
ORIGIN), then, following Tulenko & Currie's description that `fish_anatomy` is
built on:

    the two muscles whose ORIGIN is the most ROSTRAL are the OBLIQUES (they arise
    together from the anterior ethmoid plate); of those, the one inserting on the
    dorsal face is SO and the ventral one is IO. The remaining four are the recti,
    assigned to caudal/rostral/dorsal/ventral insertion stations -- LR/MR/SR/IR --
    by a greedy best-match on the insertion direction in the eye frame.

The evidence for each call is printed and written to the manifest, and the mapping
is written to `blend_parts/muscle_names.json`. THAT FILE WINS on the next run: edit
it and the geometry is re-cut under your names, no code change.

RUNNING IT. Reading a .blend needs Blender's own loader, so this script runs under
a `bpy` build (Blender 5.2 LTS, wheel `bpy==5.2.0`, cp313) in a dedicated venv:

    python read_blend.py                       # re-execs itself into that venv
    /workspace/.conda_envs/bpy-env/bin/python read_blend.py --figure --ply

The venv was made with `python3 -m venv /workspace/.conda_envs/bpy-env` +
`pip install bpy==5.2.0 numpy matplotlib`. `bpy` links against X libraries that are
not installed system-wide in this container, so we point LD_LIBRARY_PATH at a conda
env that has them (any of /workspace/.conda_envs/*/lib does).

OUTPUTS, all under `blend_parts/`:

    parts.npz          <part>__v float32 [n,3] world verts, <part>__f int32 [m,3] tris
    parts.json         one record per part: group, side, counts, centroid, bbox,
                       volume + the per-side eye frames and per-muscle geometry
    muscle_names.json  Cylinder.00N -> muscle key, re-read on the next run
    meshes/*.ply       one ASCII PLY per part                        (--ply)
    blend_parts.png    3-view check figure, parts drawn in their own colours (--figure)

Part names are `L_LR`, `R_SO`, `L_retina`, `R_cornea`, `L_lens`, `cns`,
`bone_basal_plate_029_L`, ... -- side prefix for the paired organs, side suffix for
the cartilage.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                # experiment/run_03
EYE_DIR = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
BLEND = os.path.join(EYE_DIR, "260802_s2_EYE_MUSCLES_MODEL.blend")
OUT_DIR = os.path.join(HERE, "blend_parts")

BPY_PYTHON = "/workspace/.conda_envs/bpy-env/bin/python"      # the venv holding bpy==5.2.0
_REEXEC_FLAG = "READ_BLEND_REEXEC"


def _x_library_dir() -> str | None:
    """A directory holding libXfixes.so.3 -- bpy needs it and the container has none."""
    for cand in sorted(glob.glob("/workspace/.conda_envs/*/lib")):
        if os.path.exists(os.path.join(cand, "libXfixes.so.3")):
            return cand
    return None


def _reexec_under_bpy() -> None:
    """If `bpy` is missing, restart this same script under the bpy venv."""
    try:
        import bpy  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get(_REEXEC_FLAG):                       # already tried -- don't loop
        raise SystemExit("bpy is still not importable under " + sys.executable)
    if not os.path.exists(BPY_PYTHON):
        raise SystemExit(
            "no bpy interpreter at " + BPY_PYTHON + "\n"
            "make one with:\n"
            "  python3 -m venv /workspace/.conda_envs/bpy-env\n"
            "  /workspace/.conda_envs/bpy-env/bin/pip install bpy==5.2.0 numpy matplotlib")
    env = dict(os.environ, **{_REEXEC_FLAG: "1"})
    xlib = _x_library_dir()
    if xlib:
        env["LD_LIBRARY_PATH"] = xlib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    os.execve(BPY_PYTHON, [BPY_PYTHON, os.path.abspath(__file__)] + sys.argv[1:], env)


_reexec_under_bpy()

import numpy as np                                                     # noqa: E402
import bpy                                                             # noqa: E402

# --------------------------------------------------------------------------- #
#  what the blend calls things
# --------------------------------------------------------------------------- #
MUSCLE_COLLECTION = "eye muscles"
BONE_COLLECTION = "cartilage"
EYE_PART_NAMES = ("cornea", "lens", "retina")       # retina = the globe shell
CNS_NAMES = ("central nervous system",)

MUSCLE_KEYS = ["LR", "SR", "MR", "IR", "SO", "IO"]  # fish_anatomy's order
LONG_NAME = {"LR": "lateral rectus", "SR": "superior rectus", "MR": "medial rectus",
             "IR": "inferior rectus", "SO": "superior oblique", "IO": "inferior oblique"}
COLOR = {"LR": "#4da3ff", "SR": "#ff5c5c", "MR": "#ffd24d",
         "IR": "#7ee081", "SO": "#c58cff", "IO": "#ff9c42"}

N_CENTRELINE = 24          # bins along a muscle's long axis -> centreline polyline


# --------------------------------------------------------------------------- #
#  1. pulling meshes out of Blender
# --------------------------------------------------------------------------- #
def open_blend(path: str, subsurf: bool = False) -> None:
    bpy.ops.wm.open_mainfile(filepath=path)
    for ob in bpy.data.objects:
        for mod in ob.modifiers:
            if mod.type == 'SUBSURF':
                mod.show_viewport = subsurf


def evaluated_mesh(ob, depsgraph):
    """World-space vertices [n,3] and triangles [m,3] with all modifiers applied.

    MIRROR is one of those modifiers, so a mirrored object comes back with BOTH
    halves in one array -- that is exactly what `split_sides` then separates.
    """
    ob_eval = ob.evaluated_get(depsgraph)
    me = ob_eval.to_mesh()
    try:
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)

        try:
            me.calc_loop_triangles()                 # 4.x; 5.x computes on access
        except AttributeError:
            pass
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        tri = tri.reshape(-1, 3)

        edg = np.empty(len(me.edges) * 2, dtype=np.int32)
        me.edges.foreach_get("vertices", edg)
        edg = edg.reshape(-1, 2)
    finally:
        ob_eval.to_mesh_clear()

    M = np.array(ob.matrix_world, dtype=np.float64)
    world = co @ M[:3, :3].T + M[:3, 3]
    return world.astype(np.float32), tri, edg


def loose_parts(n_vert: int, edges: np.ndarray) -> list[np.ndarray]:
    """Connected components over the edge graph, as arrays of vertex indices."""
    parent = np.arange(n_vert, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    root = np.array([find(i) for i in range(n_vert)], dtype=np.int64)
    return [np.flatnonzero(root == r) for r in np.unique(root)]


def split_sides(verts, tris, edges, merge_parts=True):
    """Cut a mirrored mesh into its left (x < 0) and right (x > 0) halves.

    Returns {'L': (v, f), 'R': (v, f)}; a side is absent if it holds no geometry.
    With `merge_parts` the several loose parts of one side (a bone drawn as four
    shells, say) are welded back into a single part -- they are the same organ.
    """
    out: dict[str, list] = {}
    for idx in loose_parts(len(verts), edges):
        side = 'L' if verts[idx, 0].mean() < 0.0 else 'R'
        out.setdefault(side, []).append(idx)

    result = {}
    for side, groups in out.items():
        if merge_parts:
            groups = [np.concatenate(groups)]
        for k, idx in enumerate(groups):
            keep = np.zeros(len(verts), dtype=bool)
            keep[idx] = True
            remap = -np.ones(len(verts), dtype=np.int64)
            remap[idx] = np.arange(len(idx))
            f = tris[keep[tris].all(axis=1)]
            key = side if len(groups) == 1 else f"{side}{k}"
            result[key] = (verts[idx].copy(), remap[f].astype(np.int32))
    return result


def signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Divergence-theorem volume; meaningful only for a CLOSED mesh (muscles, lens)."""
    if len(f) == 0:
        return 0.0
    a, b, c = v[f[:, 0]].astype(np.float64), v[f[:, 1]].astype(np.float64), v[f[:, 2]].astype(np.float64)
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


# --------------------------------------------------------------------------- #
#  2. the eye frame, per side
# --------------------------------------------------------------------------- #
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def globe_fit(retina_v: np.ndarray) -> dict:
    """Centre and principal semi-axes of the globe, from the retina shell."""
    c = retina_v.mean(axis=0).astype(np.float64)
    d = retina_v.astype(np.float64) - c
    cov = d.T @ d / len(d)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return dict(center=c, axes=V[:, order].T, semi_axes=np.sqrt(np.maximum(w[order], 0.0)) * np.sqrt(3.0),
                radius_mean=float(np.linalg.norm(d, axis=1).mean()))


def eye_frame(globe_center: np.ndarray, cornea_v: np.ndarray) -> dict:
    """(caudal, dorsal, lateral) for one eye, in fish_anatomy's sense.

    lateral is the OPTIC AXIS, pointing out of the head -- measured as centre ->
    cornea, not assumed from the side. dorsal and caudal are the head's +z and +y
    orthogonalised against it, so the triad stays orthonormal even when the optic
    axis is not exactly horizontal (here it is tilted ~10 deg rostrally).
    """
    lateral = _unit(cornea_v.mean(axis=0).astype(np.float64) - globe_center)
    dorsal = np.array([0.0, 0.0, 1.0])
    dorsal = _unit(dorsal - np.dot(dorsal, lateral) * lateral)
    caudal = np.array([0.0, 1.0, 0.0])
    caudal = _unit(caudal - np.dot(caudal, lateral) * lateral - np.dot(caudal, dorsal) * dorsal)
    R = np.stack([caudal, dorsal, lateral])            # rows: world -> eye frame
    return dict(caudal=caudal, dorsal=dorsal, lateral=lateral, R=R, det=float(np.linalg.det(R)))


# --------------------------------------------------------------------------- #
#  3. muscle geometry: which end is on the globe, and what it pulls on
# --------------------------------------------------------------------------- #
def muscle_geometry(v: np.ndarray, globe: dict, frame: dict, shell: np.ndarray,
                    n_bins: int = N_CENTRELINE) -> dict:
    """Centreline, origin, insertion, length, line of action and rotation axis.

    The long axis is the first principal component of the muscle's own vertices;
    points are binned along it and averaged to a polyline. Which END is the
    INSERTION is then decided by CONTACT WITH THE GLOBE: the distance from each end
    to the nearest retina vertex. The separation is unambiguous -- an insertion end
    sits 0.04-0.10 world units off the shell, an origin end 0.27-0.59 -- whereas
    "nearest the sphere of mean radius" is not, because the caudal origin plate that
    SR/IR/MR/LR share happens to lie about one globe radius from the globe centre
    too, and it flips LR and IR. The call is made from where the tissue is, never
    from the muscle's name.
    """
    v = v.astype(np.float64)
    c = v.mean(axis=0)
    d = v - c
    _, _, Vt = np.linalg.svd(d, full_matrices=False)
    axis = Vt[0]
    s = d @ axis
    edges = np.linspace(s.min(), s.max(), n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    line = np.stack([v[idx == b].mean(axis=0) for b in range(n_bins) if (idx == b).any()])

    R, C = globe["radius_mean"], globe["center"]
    S = shell.astype(np.float64)
    contact = [float(np.linalg.norm(S - line[0], axis=1).min()),
               float(np.linalg.norm(S - line[-1], axis=1).min())]
    if contact[0] > contact[1]:
        line = line[::-1]                              # index 0 = insertion end
        contact = contact[::-1]
    insertion, origin = line[0], line[-1]

    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    pull = _unit(line[1] - line[0])                    # insertion -> origin: the pull
    r_hat = _unit(insertion - C)
    rot = _unit(np.cross(r_hat, pull))                 # moment axis of that pull

    E = frame["R"]                                     # world -> (caudal, dorsal, lateral)
    return dict(centreline=line, insertion=insertion, origin=origin,
                length=float(seg.sum()), chord=float(np.linalg.norm(origin - insertion)),
                pull_dir=pull, insertion_dir=r_hat, rot_axis=rot,
                insertion_eye=E @ r_hat, pull_eye=E @ pull, rot_eye=E @ rot,
                insertion_depth=float(np.linalg.norm(insertion - C) / R),
                contact_insertion=contact[0], contact_origin=contact[1],
                origin_caudal=float(origin[1]))        # head +y: small = rostral


def name_muscles(geo: dict) -> tuple[dict, list]:
    """Cylinder.00N -> LR/SR/MR/IR/SO/IO, measured (see the module docstring).

    Returns the mapping and the evidence rows that justify it.
    """
    objs = sorted(geo)
    # obliques: the two most ROSTRAL origins (smallest head +y), split by insertion height
    by_rostral = sorted(objs, key=lambda k: geo[k]["origin_caudal"])
    obliques = by_rostral[:2]
    ob_sorted = sorted(obliques, key=lambda k: geo[k]["insertion_eye"][1])   # dorsal component
    names = {ob_sorted[1]: "SO", ob_sorted[0]: "IO"}

    # recti: greedy best match of insertion direction to the four stations
    recti = [k for k in objs if k not in names]
    station = {"LR": np.array([1.0, 0.0, 0.0]), "MR": np.array([-1.0, 0.0, 0.0]),
               "SR": np.array([0.0, 1.0, 0.0]), "IR": np.array([0.0, -1.0, 0.0])}
    score = {(k, t): float(geo[k]["insertion_eye"] @ u) for k in recti for t, u in station.items()}
    left_k, left_t = set(recti), set(station)
    for k, t in sorted(score, key=lambda kt: -score[kt]):
        if k in left_k and t in left_t:
            names[k] = t
            left_k.discard(k)
            left_t.discard(t)

    rows = []
    for k in objs:
        g = geo[k]
        rows.append(dict(object=k, key=names[k], long_name=LONG_NAME[names[k]],
                         origin_caudal=round(g["origin_caudal"], 3),
                         insertion_eye=[round(float(x), 3) for x in g["insertion_eye"]],
                         rot_eye=[round(float(x), 3) for x in g["rot_eye"]],
                         contact_insertion=round(g["contact_insertion"], 4),
                         contact_origin=round(g["contact_origin"], 4),
                         length=round(g["length"], 4)))
    return names, rows


# --------------------------------------------------------------------------- #
#  4. the whole cut, end to end
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(".", "_")


def read_model(blend=BLEND, subsurf=False, name_map=None):
    """Cut the blend into named parts. Returns (parts, manifest)."""
    open_blend(blend, subsurf=subsurf)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    raw = {}                       # object name -> {side: (v, f)}
    groups = {}                    # object name -> group
    for ob in bpy.data.objects:
        if ob.type != 'MESH' or ob.data is None:
            continue
        colls = {c.name for c in ob.users_collection}
        if MUSCLE_COLLECTION in colls:
            grp = "muscle"
        elif ob.name in EYE_PART_NAMES:
            grp = "eye"
        elif ob.name in CNS_NAMES:
            grp = "cns"
        elif BONE_COLLECTION in colls:
            grp = "bone"
        else:
            grp = "other"
        v, f, e = evaluated_mesh(ob, depsgraph)
        raw[ob.name] = split_sides(v, f, e)
        groups[ob.name] = grp

    # --- the eye frames, one per side, before anything is named ---------------
    frames, globes = {}, {}
    for side in ("L", "R"):
        retina = raw["retina"].get(side)
        cornea = raw["cornea"].get(side)
        if retina is None or cornea is None:
            continue
        globes[side] = globe_fit(retina[0])
        frames[side] = eye_frame(globes[side]["center"], cornea[0])

    # --- muscles: measure on the LEFT, name once, apply to both sides ---------
    geo = {}
    for obname, grp in groups.items():
        if grp != "muscle" or "L" not in raw[obname]:
            continue
        geo[obname] = muscle_geometry(raw[obname]["L"][0], globes["L"], frames["L"],
                                      shell=raw["retina"]["L"][0])
    measured_names, evidence = name_muscles(geo)
    names = dict(measured_names)
    if name_map:                                       # muscle_names.json wins
        names.update({k: v for k, v in name_map.items() if k in names})

    # --- assemble the parts ---------------------------------------------------
    parts, records = {}, []
    for obname, grp in groups.items():
        for side, (v, f) in sorted(raw[obname].items()):
            if grp == "muscle":
                part = f"{side}_{names[obname]}"
            elif grp == "eye":
                part = f"{side}_{obname}"
            elif grp == "cns":
                part = _slug(obname)
            else:
                part = f"bone_{_slug(obname)}_{side}"
            parts[part] = (v, f)
            rec = dict(part=part, group=grp, side=side, blender_object=obname,
                       n_vert=int(len(v)), n_tri=int(len(f)),
                       centroid=[round(float(x), 4) for x in v.mean(axis=0)],
                       bbox_min=[round(float(x), 4) for x in v.min(axis=0)],
                       bbox_max=[round(float(x), 4) for x in v.max(axis=0)],
                       volume=round(signed_volume(v, f), 6))
            if grp == "muscle" and side in frames:
                g = muscle_geometry(v, globes[side], frames[side], shell=raw["retina"][side][0])
                rec.update(muscle_key=names[obname], long_name=LONG_NAME[names[obname]],
                           length=round(g["length"], 4), chord=round(g["chord"], 4),
                           insertion=[round(float(x), 4) for x in g["insertion"]],
                           origin=[round(float(x), 4) for x in g["origin"]],
                           insertion_eye=[round(float(x), 3) for x in g["insertion_eye"]],
                           pull_eye=[round(float(x), 3) for x in g["pull_eye"]],
                           rot_axis_eye=[round(float(x), 3) for x in g["rot_eye"]],
                           insertion_depth=round(g["insertion_depth"], 3),
                           contact_insertion=round(g["contact_insertion"], 4),
                           contact_origin=round(g["contact_origin"], 4))
                parts[part + "__centreline"] = (g["centreline"].astype(np.float32), np.zeros((0, 3), np.int32))
            records.append(rec)

    manifest = dict(
        blend=os.path.abspath(blend), subsurf=subsurf,
        frame_note="head axes: +x animal's right, +y caudal, +z dorsal; "
                   "eye frame rows = (caudal, dorsal, lateral)",
        eyes={side: dict(globe_center=[round(float(x), 4) for x in globes[side]["center"]],
                         globe_semi_axes=[round(float(x), 4) for x in globes[side]["semi_axes"]],
                         globe_radius_mean=round(globes[side]["radius_mean"], 4),
                         frame=[[round(float(x), 4) for x in row] for row in frames[side]["R"]],
                         frame_det=round(frames[side]["det"], 3)) for side in globes},
        muscle_names={k: names[k] for k in sorted(names)},
        muscle_evidence=evidence,
        parts=sorted(records, key=lambda r: (r["group"], r["part"])),
    )
    return parts, manifest


# --------------------------------------------------------------------------- #
#  5. writing it out
# --------------------------------------------------------------------------- #
def write_ply(path: str, v: np.ndarray, f: np.ndarray) -> None:
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n")
        fh.write(f"element face {len(f)}\nproperty list uchar int vertex_index\nend_header\n")
        for p in v:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in f:
            fh.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def figure(parts: dict, manifest: dict, path: str) -> None:
    """Three orthogonal views: is the cut right? Muscles in fish_anatomy's colours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = [("A", "lateral view (caudal-dorsal)", 1, 2),
             ("B", "dorsal view (caudal-right)", 1, 0),
             ("C", "frontal view (right-dorsal)", 0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), facecolor="black")
    key_of = {r["part"]: r for r in manifest["parts"]}

    # frame on the ORBITS: the notochord and CNS run to y = 19, and drawn whole they
    # squash the head -- the part of the animal this file is about -- to a smear.
    orbit = np.concatenate([v for p, (v, f) in parts.items()
                            if not p.endswith("__centreline")
                            and key_of[p]["group"] in ("muscle", "eye")])
    lo, hi = orbit.min(axis=0), orbit.max(axis=0)
    mid, half = (lo + hi) / 2.0, (hi - lo).max() / 2.0 * 1.45

    for ax, (lab, sub, i, j) in zip(axes, views):
        ax.set_facecolor("black")
        for part, (v, f) in sorted(parts.items()):
            if part.endswith("__centreline"):
                continue
            rec = key_of[part]
            grp = rec["group"]
            if grp == "bone":
                col, alpha, size = "#8e8494", 0.35, 0.4
            elif grp == "cns":
                col, alpha, size = "#3fcc55", 0.25, 0.4
            elif grp == "muscle":
                col, alpha, size = COLOR[rec["muscle_key"]], 0.95, 1.4
            else:
                col, alpha, size = {"cornea": "#9fd8ff", "lens": "#ffffff", "retina": "#c9c2cc"}[
                    rec["blender_object"]], 0.7, 1.0
            ax.scatter(v[:, i], v[:, j], s=size, c=col, alpha=alpha, linewidths=0, marker=".")
        for part, (v, f) in sorted(parts.items()):
            if part.endswith("__centreline"):
                k = key_of[part[:-len("__centreline")]]["muscle_key"]
                ax.plot(v[:, i], v[:, j], "-", color=COLOR[k], lw=1.6, alpha=1.0)
                ax.plot(v[0, i], v[0, j], "o", color=COLOR[k], ms=4)      # insertion
        ax.set_xlim(mid[i] - half, mid[i] + half)
        ax.set_ylim(mid[j] - half, mid[j] + half)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, color="white", fontsize=13,
                fontweight="bold", va="top", ha="left")
        ax.text(0.02, 0.92, sub, transform=ax.transAxes, color="white", fontsize=9,
                va="top", ha="left", alpha=0.75)

    handles = [plt.Line2D([], [], color=COLOR[k], lw=3,
                          label=f"{k}  {LONG_NAME[k]}") for k in MUSCLE_KEYS]
    leg = axes[1].legend(handles=handles, loc="lower center", frameon=False,
                         fontsize=8, ncol=3, labelcolor="white")
    for t in leg.get_texts():
        t.set_color("white")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="black")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blend", default=BLEND)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--subsurf", action="store_true", help="keep the subdivision surfaces")
    ap.add_argument("--ply", action="store_true", help="also write one PLY per part")
    ap.add_argument("--figure", action="store_true", help="write the 3-view check figure")
    ap.add_argument("--list", action="store_true", help="print the blend's inventory and stop")
    args = ap.parse_args()

    if args.list:
        open_blend(args.blend, subsurf=args.subsurf)
        for coll in bpy.data.collections:
            print(f"[{coll.name}]  {len(coll.objects)} objects")
            for ob in sorted(coll.objects, key=lambda o: o.name):
                n = len(ob.data.vertices) if ob.type == 'MESH' else 0
                mods = ",".join(m.type for m in ob.modifiers)
                print(f"    {ob.name:28s} {ob.type:7s} verts={n:7d} mods={mods}")
        return

    os.makedirs(args.out, exist_ok=True)
    name_file = os.path.join(args.out, "muscle_names.json")
    name_map = json.load(open(name_file)) if os.path.exists(name_file) else None

    parts, manifest = read_model(args.blend, subsurf=args.subsurf, name_map=name_map)

    np.savez_compressed(os.path.join(args.out, "parts.npz"),
                        **{f"{k}__v": v for k, (v, f) in parts.items()},
                        **{f"{k}__f": f for k, (v, f) in parts.items()})
    with open(os.path.join(args.out, "parts.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(name_file, "w") as fh:
        json.dump(manifest["muscle_names"], fh, indent=2)
    if args.ply:
        os.makedirs(os.path.join(args.out, "meshes"), exist_ok=True)
        for k, (v, f) in parts.items():
            if not k.endswith("__centreline"):
                write_ply(os.path.join(args.out, "meshes", k + ".ply"), v, f)
    if args.figure:
        figure(parts, manifest, os.path.join(args.out, "blend_parts.png"))

    # --- the report ----------------------------------------------------------
    print(f"\n{args.blend}  ->  {args.out}   (subsurf {'on' if args.subsurf else 'off'})\n")
    for side, e in manifest["eyes"].items():
        print(f"  {side} globe  centre {e['globe_center']}  semi-axes {e['globe_semi_axes']}  "
              f"mean radius {e['globe_radius_mean']}  frame det {e['frame_det']:+.0f}")
    print("\n  the six muscles, named from geometry"
          "  (contact = distance from that end to the globe shell):")
    print(f"    {'blender':14s} {'key':4s} {'name':18s} {'origin y':>9s} "
          f"{'insertion (caud,dors,lat)':>28s} {'rot axis (caud,dors,lat)':>28s} "
          f"{'contact ins':>11s} {'ori':>6s} {'length':>7s}")
    for r in manifest["muscle_evidence"]:
        print(f"    {r['object']:14s} {r['key']:4s} {r['long_name']:18s} {r['origin_caudal']:9.3f} "
              f"{str(r['insertion_eye']):>28s} {str(r['rot_eye']):>28s} "
              f"{r['contact_insertion']:11.4f} {r['contact_origin']:6.3f} {r['length']:7.4f}")
    n_by_group: dict[str, int] = {}
    for r in manifest["parts"]:
        n_by_group[r["group"]] = n_by_group.get(r["group"], 0) + 1
    print("\n  parts written: " + ", ".join(f"{v} {k}" for k, v in sorted(n_by_group.items()))
          + f"  ({sum(len(v) for v, _ in parts.values())} vertices total)")
    for r in manifest["parts"]:
        if r["group"] in ("muscle", "eye"):
            print(f"    {r['part']:14s} {r['group']:7s} verts {r['n_vert']:6d}  tris {r['n_tri']:6d}  "
                  f"centroid {r['centroid']}")


if __name__ == "__main__":
    main()
