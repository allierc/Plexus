"""The shape library: a NAME resolves to a folder, a folder holds parts, a part fills a body.

WHY A LIBRARY AND NOT A PATH. `shape: obj` took a path to an `.obj` under `papers/morph_models`
and knew nothing else; the eye prototype read a `.blend` through a `bpy` subprocess into a cache
beside itself; a neuprint region holds one file per neuron under `graphs_data/neural_regions`.
Three readers, three layouts, and a spec that wants a scanned body has to know which it is. Here
a spec names a SHAPE, and the library answers with points:

    form: mesh:bunny                 -> graphs_data/shapes/bunny/
    form: mesh:eye/lateral_rectus    -> the named PART of a multi-part shape
    form: mesh:hemibrain/1004436788  -> a body of a population (a tree of one file per row)

A shape folder holds whatever a person supplied plus a cache the library builds:

    <root>/shapes/<name>/
        source.obj | source.blend | source.tif      what was supplied
        parts.npz                                   V, F and an offset per part, in metres
        parts.json                                  per part: volume, centroid, extent, watertight
        PROVENANCE.md                               where it came from, and its licence

`<root>` is `GNN_OUTPUT_ROOT/graphs_data` when that is set, else the repo's own `graphs_data/`,
and the repo's `papers/morph_models` is searched last so the five meshes that shipped there keep
working while they are moved.

WHAT THE LIBRARY GUARANTEES, and it is the whole contract a seeder needs:
  * `points(name, n, volume)` returns `n` points uniform INSIDE the part, centred on its own
    centroid, scaled so the part's volume is exactly `volume` -- so a bunny and a cow of the same
    mass weigh the same and displace the same, whatever their extent.
  * the sample is deterministic: a Halton sequence, rejected against the surface. No RNG, so a
    run is reproducible, and no voxelisation, so the density carries no grid pattern into the
    first frame's stress.
  * an open mesh is closed first (`fill_holes`), because a ray from inside escapes through a hole
    and parity then reports half the body outside: the bunny's base and the teapot's spout.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

_CACHE: dict = {}
_LOCK = threading.RLock()
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def roots() -> list:
    """Where shapes are looked for, in order: the data tree, the repo, the old morph_models."""
    out = []
    try:
        from plexus.paths import graphs_data_path
        out.append(os.path.join(graphs_data_path(), "shapes"))
    except Exception:                                            # noqa: BLE001
        pass
    out.append(os.path.join(_REPO, "graphs_data", "shapes"))
    out.append(os.path.join(_REPO, "papers", "morph_models"))    # the five that shipped here
    return out


def resolve(name: str) -> tuple:
    """`<shape>` or `<shape>/<part>` -> (folder, part or None). Raises with the roots it tried."""
    shape, _, part = str(name).strip().partition("/")
    tried = []
    for r in roots():
        d = os.path.join(r, shape)
        if os.path.isdir(d):
            return d, (part or None)
        tried.append(d)
        for ext in (".obj", ".blend", ".stl", ".ply"):           # a bare file in an old folder
            f = os.path.join(r, shape + ext)
            if os.path.exists(f):
                return f, (part or None)
            tried.append(f)
    raise FileNotFoundError(f"shape {name!r} is not in the library; looked for " + ", ".join(tried[:6]))


def _parts_from_obj(path: str) -> dict:
    """One part per connected mesh file: vertices and faces, holes filled, normals consistent."""
    import pyvista as pv
    m = pv.read(path)
    m = m.triangulate()
    if m.n_open_edges:
        m = m.fill_holes(m.length)                               # the bunny's base, the teapot's spout
    m = m.compute_normals(auto_orient_normals=True, consistent_normals=True)
    V = np.asarray(m.points, np.float64)
    F = np.asarray(m.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    return {os.path.splitext(os.path.basename(path))[0]: (V, F)}


def _build_cache(folder: str) -> dict:
    """`parts.npz` + `parts.json` for a folder, built from its source when missing or stale."""
    if os.path.isfile(folder):                                   # a bare file, the old layout
        return _parts_from_obj(folder)
    npz = os.path.join(folder, "parts.npz")
    src = None
    for nm in sorted(os.listdir(folder)):
        if nm.startswith("source."):
            src = os.path.join(folder, nm)
            break
    if os.path.exists(npz) and (src is None or os.path.getmtime(npz) >= os.path.getmtime(src)):
        z = np.load(npz, allow_pickle=False)
        names = [k[: -len("__V")] for k in z.files if k.endswith("__V")]
        return {n: (z[f"{n}__V"], z[f"{n}__F"]) for n in names}
    if src is None:
        raise FileNotFoundError(f"shape folder {folder} holds no `source.*` and no parts.npz")
    if src.endswith(".blend"):
        raise NotImplementedError(
            f"{src}: a .blend is cut into parts by Blender's own loader; that step is the eye "
            f"prototype's `read_blend.py`, not yet promoted (plan S2)")
    parts = _parts_from_obj(src)
    np.savez_compressed(npz, **{f"{n}__{k}": v for n, (V, F) in parts.items()
                                for k, v in (("V", V), ("F", F))})
    meta = {n: {"volume": float(_volume(V, F)), "centroid": V.mean(0).tolist(),
                "extent": (V.max(0) - V.min(0)).tolist(), "n_vertices": int(len(V)), "n_faces": int(len(F))}
            for n, (V, F) in parts.items()}
    json.dump(meta, open(os.path.join(folder, "parts.json"), "w"), indent=1)
    print(f"[shapes] built {os.path.basename(folder)}: " +
          ", ".join(f"{n} ({m['n_faces']:,} faces, volume {m['volume']:.4g})" for n, m in meta.items()), flush=True)
    return parts


def _volume(V, F) -> float:
    """The enclosed volume by the divergence theorem: sum of signed tetrahedra on the origin."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def parts(name: str) -> dict:
    """Every part of a shape, cached per folder for the life of the process."""
    folder, _ = resolve(name)
    with _LOCK:
        if folder not in _CACHE:
            _CACHE[folder] = _build_cache(folder)
        return _CACHE[folder]


def _halton(n: int, bases=(2, 3, 5)) -> np.ndarray:
    """A deterministic low-discrepancy sequence: no RNG, and it fills a volume more evenly than a
    uniform draw, which is what MLS-MPM resolves a material by."""
    out = np.empty((n, len(bases)))
    for j, b in enumerate(bases):
        i = np.arange(1, n + 1, dtype=np.float64)
        f, r = 1.0 / b, np.zeros(n)
        while (i > 0).any():
            r += f * (i % b)
            i = np.floor(i / b)
            f /= b
        out[:, j] = r
    return out


def points(name: str, n: int, volume: float, oversample: float = 3.0) -> np.ndarray:
    """`n` points inside the named shape (or part), centred on its centroid, scaled so the part's
    volume equals `volume`. Rejection against the surface; the batch is grown until `n` are kept."""
    import pyvista as pv
    P = parts(name)
    _, part = resolve(name)
    if part is None:
        part = sorted(P)[0] if len(P) == 1 else None
        if part is None:
            raise ValueError(f"shape {name!r} has {len(P)} parts ({', '.join(sorted(P))}); "
                             f"name one as `mesh:{name}/<part>`")
    if part not in P:
        raise KeyError(f"shape {name!r} has no part {part!r} (it has {', '.join(sorted(P))})")
    V, F = P[part]
    V = V - V.mean(0)
    s = (float(volume) / max(_volume(V, F), 1e-30)) ** (1.0 / 3.0)
    V = V * s
    surf = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    lo, hi = V.min(0), V.max(0)
    kept = np.zeros((0, 3))
    k = 0
    while len(kept) < n and k < 12:
        m = int(max(n * oversample, 1024))
        q = lo + _halton(m + k * m)[k * m:] * (hi - lo) if k else lo + _halton(m) * (hi - lo)
        sel = pv.PolyData(q).select_enclosed_points(surf, tolerance=0.0, check_surface=False)
        inside = q[np.asarray(sel["SelectedPoints"]).astype(bool)]
        kept = np.concatenate([kept, inside], 0)
        k += 1
    if len(kept) < n:
        raise RuntimeError(f"shape {name!r}: only {len(kept)} of {n} points landed inside; is the "
                           f"mesh closed?")
    return kept[:n] - kept[:n].mean(0)
