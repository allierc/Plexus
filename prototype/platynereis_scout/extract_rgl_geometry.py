"""Extract 3D geometry from the Platynereis whole-body connectome cell-type compendium.

Source: https://github.com/JekelyLab/Platynereis_3D_connectome_2024
        supplements/celltype_compendium_website/celltype_compendium/celltype_RGLs/*.html

Each HTML is an R `rgl` WebGL widget whose scene is a glTF-style buffer
(base64 in x.buffer.buffers[0].bytes, indexed by accessors -> bufferViews).
Per file:
  - `spheres` objects  -> one soma centre per cell of that cell type (nm)
  - `linestrip` objects -> traced skeleton arbors, NaN-separated (nm)

This bypasses the lab's CATMAID server (catmaid.jekelylab.ex.ac.uk), which is
unreachable from this machine. Verified 2026-09-18: 291 of 294 compendium files
decode, giving 4,117 soma spheres (4,116 unique positions) and 5,368,512
skeleton nodes.

Usage:
    python extract_rgl_geometry.py <dir_of_html> <out.csv>
"""

import base64
import glob
import json
import os
import re
import sys

import numpy as np

# glTF componentType -> numpy dtype
COMPONENT_DTYPE = {5126: "<f4", 5121: "<u1", 5125: "<u4", 5123: "<u2"}
# glTF element type -> number of components
TYPE_NCOMP = {"VEC3": 3, "VEC4": 4, "SCALAR": 1, "MAT4": 16}

_SCENE_RE = re.compile(r'<script type="application/json"[^>]*>(.*?)</script>', re.S)


def load_scene(path):
    """Return (objects_dict, accessor_getter) for one rgl HTML file."""
    text = open(path, encoding="utf8", errors="replace").read()
    match = _SCENE_RE.search(text)
    if match is None:
        raise ValueError(f"no rgl scene JSON in {path}")
    scene = json.loads(match.group(1))["x"]
    buf = scene["buffer"]
    raw = base64.b64decode(re.sub(r"\s", "", buf["buffers"][0]["bytes"]))
    views, accessors = buf["bufferViews"], buf["accessors"]

    def get(accessor_index):
        acc = accessors[int(accessor_index)]
        view = views[acc["bufferView"]]
        ncomp = TYPE_NCOMP[acc["type"]]
        return np.frombuffer(
            raw,
            dtype=np.dtype(COMPONENT_DTYPE[acc["componentType"]]),
            count=acc["count"] * ncomp,
            offset=view["byteOffset"],
        ).reshape(acc["count"], ncomp)

    return scene["objects"], get


def soma_positions(path):
    """(n_cells, 3) array of soma centres in nm for the cell type in `path`."""
    objects, get = load_scene(path)
    spheres = [o for o in objects.values() if o["type"] == "spheres"]
    if not spheres:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack([get(o["vertices"]) for o in spheres])


def skeleton_nodes(path):
    """List of (n_i, 3) arrays, one per traced arbor, in nm. Rows may be NaN separators."""
    objects, get = load_scene(path)
    return [
        get(o["vertices"])
        for o in objects.values()
        if o["type"] in ("linestrip", "lines")
    ]


def main(html_dir, out_csv):
    import csv

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["celltype_annotation", "idx_in_type", "x_nm", "y_nm", "z_nm"])
        total = 0
        for path in sorted(glob.glob(os.path.join(html_dir, "celltype*.html"))):
            annotation = os.path.basename(path)[:-5]
            try:
                pts = soma_positions(path)
            except ValueError:
                continue
            for i, p in enumerate(pts):
                writer.writerow([annotation, i, f"{p[0]:.1f}", f"{p[1]:.1f}", f"{p[2]:.1f}"])
                total += 1
    print(f"wrote {total} soma positions to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
