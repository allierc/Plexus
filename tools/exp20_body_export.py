"""exp20: the WHOLISTIC whole-body larval zebrafish model (Ruetten et al. 2026; wholebodymodel.blend, fetched by Cedric
2026-10-03 into graphs_data/zebrafish/wholistic/) exported to an npz the renderer reads -- bpy and pyvista live in two
environments (memory: bpy venv /workspace/.conda_envs/bpy-env, LD_LIBRARY_PATH to a conda env's lib for X).

    LD_LIBRARY_PATH=/workspace/.conda_envs/MPM-pytorch/lib /workspace/.conda_envs/bpy-env/bin/python tools/exp20_body_export.py

Writes experiments/exp20_gutbrain_graphcast/data/anatomy/wholebody.npz: per organ its world-space vertices and
triangles (`<organ>__v`, `<organ>__f`), in the model's own units (head at x ~ 0, tail at x ~ 21; z dorsal up).
"""
import os

import bpy
import bmesh
import numpy as np

SRC = "/groups/saalfeld/home/allierc/GraphData/graphs_data/zebrafish/wholistic/wholebodymodel.blend"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments",
                   "exp20_gutbrain_graphcast", "data", "anatomy", "wholebody.npz")
ORGANS = {"skin": ["skin"], "gut": ["gastrointestinal tract"], "liver": ["liver"], "pancreas": ["exocrine pancreas"],
          "swim_bladder": ["swim bladder"], "heart": ["atrium", "ventricle"], "eye": ["retina"],
          "kidney": ["nephric system"], "notochord": ["notochord.002"]}


def tri(o):
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.transform(o.matrix_world)
    v = np.array([x.co[:] for x in bm.verts], np.float32)
    f = np.array([[x.index for x in fc.verts] for fc in bm.faces], np.int32)
    bm.free()
    return v, f


bpy.ops.wm.open_mainfile(filepath=SRC)
out = {}
for k, names in ORGANS.items():
    V, F, n = [], [], 0
    for nm in names:
        o = bpy.data.objects.get(nm)
        if o is None or o.type != "MESH":
            print("missing", nm)
            continue
        v, f = tri(o)
        V.append(v); F.append(f + n); n += len(v)
    if V:
        out[f"{k}__v"], out[f"{k}__f"] = np.concatenate(V), np.concatenate(F)
ns = bpy.data.collections.get("nervous system")
if ns:
    V, F, n = [], [], 0
    for o in ns.objects:
        if o.type == "MESH" and "spinal" not in o.name.lower() and "nerve" not in o.name.lower():
            v, f = tri(o)
            V.append(v); F.append(f + n); n += len(v)
            print("brain part", o.name, len(v))
    if V:
        out["brain__v"], out["brain__f"] = np.concatenate(V), np.concatenate(F)
np.savez_compressed(OUT, **out)
print("->", OUT, sorted({k.split("__")[0] for k in out}))
