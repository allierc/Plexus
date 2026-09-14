"""Cut a `.blend` into the shape library's parts -- the one step that needs Blender itself.

WHY A SEPARATE FILE AND A SUBPROCESS. `bpy` ships its own Python and its own numpy, so the
interpreter that runs Plexus cannot import it. This script therefore RE-EXECUTES ITSELF under the
bpy interpreter (`PLEXUS_BPY_PYTHON`, else a `bpy-env` beside the other conda envs) and is called
by `plexus.shapes._build_cache` as a subprocess, once, when `parts.npz` is missing or older than
the `.blend`. Nothing above it imports bpy, and a shape whose cache is already built needs no
Blender at all -- which is what lets a cluster job read a scanned body without it.

WHAT IT PROMISES, and it is deliberately less than the eye prototype's own reader
(`prototype/eye/archive/run_03/read_blend.py`, which this takes its mechanics from): named
watertight parts in WORLD coordinates, with every modifier applied. It measures no anatomical
frame, names no muscle and fits no globe -- those are the eye's questions, asked in the eye's
own file. The library's contract is `points(name, n, volume)` and the only thing it needs from
Blender is the geometry.

THREE THINGS ABOUT READING A .blend that decide what comes out:

  1. MODIFIERS ARE THE MODEL. An artist's MIRROR modifier means half the geometry exists only in
     the evaluated mesh: reading `object.data` gives one eye where the file shows two. The
     depsgraph is evaluated first, so what is cut is what is seen.
  2. SUBSURF MULTIPLIES VERTICES for a surface that is already dense enough to enclose a volume.
     It is turned OFF by default (`--subsurf` restores it): 1,120 vertices per muscle rather than
     ~4,500, and the rejection sampler in `shapes.points` cares only that the mesh be closed.
  3. ONE OBJECT CAN BE SEVERAL BODIES. The evaluated mesh is split into connected components:
     one component keeps the object's own name, two that separate cleanly across x = 0 become
     `<name>_l` and `<name>_r` (the mirror case), and anything else is numbered `<name>_00`.

    python -m plexus.shapes_blend --blend <file.blend> --out <parts.npz> [--subsurf]
"""
from __future__ import annotations

import glob
import os
import sys

_REEXEC = "PLEXUS_BLEND_REEXEC"
BPY_PYTHON = os.environ.get("PLEXUS_BPY_PYTHON", "/workspace/.conda_envs/bpy-env/bin/python")


def _x_library_dir():
    """A directory holding libXfixes.so.3 -- bpy links it and the container ships none."""
    for cand in sorted(glob.glob("/workspace/.conda_envs/*/lib")):
        if os.path.exists(os.path.join(cand, "libXfixes.so.3")):
            return cand
    return None


def _reexec_under_bpy():
    try:
        import bpy                                                    # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get(_REEXEC):                                       # already tried; do not loop
        raise SystemExit(f"bpy is still not importable under {sys.executable}")
    if not os.path.exists(BPY_PYTHON):
        raise SystemExit(
            f"no bpy interpreter at {BPY_PYTHON}. Set PLEXUS_BPY_PYTHON, or make one:\n"
            f"  python3 -m venv /workspace/.conda_envs/bpy-env\n"
            f"  /workspace/.conda_envs/bpy-env/bin/pip install bpy==5.2.0 numpy")
    env = dict(os.environ, **{_REEXEC: "1"})
    xlib = _x_library_dir()
    if xlib:
        env["LD_LIBRARY_PATH"] = xlib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    os.execve(BPY_PYTHON, [BPY_PYTHON, os.path.abspath(__file__)] + sys.argv[1:], env)


if __name__ == "__main__":
    _reexec_under_bpy()

import numpy as np                                                    # noqa: E402


def _slug(name):
    out = "".join(c.lower() if (c.isalnum() or c in "._") else "_" for c in str(name))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "part"


def _evaluated_mesh(ob, depsgraph):
    """World-space vertices [n,3], triangles [m,3] and edges [k,2], modifiers applied."""
    ob_eval = ob.evaluated_get(depsgraph)
    me = ob_eval.to_mesh()
    try:
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        try:
            me.calc_loop_triangles()                                  # 4.x; 5.x computes on access
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
    return (co @ M[:3, :3].T + M[:3, 3]).astype(np.float32), tri, edg


def _components(n_vert, edges):
    """Connected components over the edge graph, as arrays of vertex indices (union-find)."""
    parent = np.arange(n_vert, dtype=np.int64)

    def find(a):
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


def _sub(verts, tris, idx):
    keep = np.zeros(len(verts), dtype=bool)
    keep[idx] = True
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    f = tris[keep[tris].all(axis=1)]
    return verts[idx].copy(), remap[f].astype(np.int64)


def cut(blend, subsurf=False):
    """{part name: (V, F)} for every mesh object in the file, in world coordinates."""
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend))
    if not subsurf:
        for ob in bpy.data.objects:
            for m in list(getattr(ob, "modifiers", []) or []):
                if m.type == "SUBSURF":
                    m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not len(getattr(ob.data, "vertices", [])):
            continue
        V, F, E = _evaluated_mesh(ob, dg)
        if not len(F):
            continue
        base = _slug(ob.name)
        comps = _components(len(V), E)
        if len(comps) == 1:
            names = [base]
        elif len(comps) == 2 and (V[comps[0], 0].mean() < 0) != (V[comps[1], 0].mean() < 0):
            # THE MIRROR CASE, NAMED FOR WHAT IT IS. Two components that fall on opposite sides of
            # x = 0 are one organ drawn once and mirrored, so they are `_l` and `_r` rather than
            # `_00` and `_01` -- a spec asking for `mesh:eye/retina_r` should not have to know
            # which index the cutter happened to give it.
            order = sorted(range(2), key=lambda i: V[comps[i], 0].mean())
            names = [None, None]
            names[order[0]], names[order[1]] = f"{base}_l", f"{base}_r"
        else:
            names = [f"{base}_{i:02d}" for i in range(len(comps))]
        for nm, idx in zip(names, comps):
            out[nm] = _sub(V, F, idx)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True, help="the parts.npz to write")
    ap.add_argument("--subsurf", action="store_true", help="keep the subdivision modifiers on")
    a = ap.parse_args()
    parts = cut(a.blend, subsurf=a.subsurf)
    if not parts:
        raise SystemExit(f"{a.blend}: no mesh object with faces")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(a.out, **{f"{n}__{k}": v for n, (V, F) in parts.items()
                                  for k, v in (("V", V.astype(np.float64)), ("F", F))})
    print(f"[shapes-blend] {os.path.basename(a.blend)} -> {len(parts)} parts: "
          + ", ".join(f"{n} ({len(F):,} faces)" for n, (V, F) in sorted(parts.items())), flush=True)


if __name__ == "__main__":
    main()
