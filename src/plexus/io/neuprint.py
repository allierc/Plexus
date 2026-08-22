"""neuprint -- a connectome region, frozen into a manifest a Plexus seed can read.

WHAT IT DOES, in one line: pick a CUBE of the fly brain holding about a target number of
neurons, and write down exactly which neurons those are, where they are, and what they look
like -- once, offline, so that every later simulation of that region is reproducible without
touching the network.

    NeuPrint server
        |  soma locations of every traced neuron            (one Cypher query)
        v
    cube selection      bisect the side L about a centre until N(L) ~= target
        |
        v
    region manifest     bounds + neuron ids + xyz + types  (+ skeletons, + meshes)
        |
        v
    `neural_seed`       establishes x_0 from the manifest -- see operators/neural.py

WHY THE SELECTION IS FROZEN AND NOT REPEATED. The spec that runs the simulation names a
manifest, not a query. Repeating the selection every run would make the identity of the
1,000 neurons a function of the server's contents on the day, so two runs of "the same" spec
could silently be two different circuits -- and a connectome server IS versioned but is not
frozen (`hemibrain:v1.2.1` coexists with `v1.1`, `male-cns`, `manc`, ...). The manifest
records the dataset string, the query, the bounds and the ids, so the selection is auditable
after the fact.

WHAT IS MEASURED, AND WHAT IT COST (hemibrain:v1.2.1, this devcontainer, 2026-08-22):

    traced neurons carrying a soma          22,212
    soma bounding box (8 nm voxels)         x 9..34,412   y 6,527..37,576   z 2,856..41,360
                                            = 275 x 249 x 308 um
    densest cube of side 6,000 vox (48 um)  998 somas          <- the ~1000 target, almost exactly
    densest cube of side 4,000 vox (32 um)  496 somas
    densest cube of side 8,000 vox (64 um)  1,624 somas

    skeleton fetch          ~0.2 s / neuron                    -> ~4 min for 1,000
    mesh fetch (lod 2)      ~0.12 s / neuron at 8 threads      -> ~2 min for 1,000, warm
    mesh fetch (lod 2)      28.7 s for the FIRST neuron        -> the cold-start is the whole
                                                                  cost of a small run; batch.
    LOD range               -1 .. 3 only; lod=4+ is rejected by the server.

COORDINATES ARE HEMIBRAIN VOXELS, 8 nm each, and this module never converts them. The
manifest records `voxel_size_nm` so that a spec declaring `general.units:` can, and so that a
run which declares no units is visibly dimensionless rather than accidentally in micrometres.

WHERE A REGION MUST NOT LIVE, learned the hard way. A region is an INPUT. Writing it to
`graphs_data/<type>/<name>/` puts it inside the directory a spec of that name writes its
trajectory to -- and `Plexus_Main.py -o generate --force` begins by `shutil.rmtree`-ing that
directory. The first run of this importer put a manifest, a connectome, 1,002 skeletons and
950 meshes there, and the first `--force` generate deleted all of it before building. So a
bare `--out NAME` resolves to `graphs_data/neural_regions/NAME`, a tree no spec writes into;
pass an absolute path to override.

Usage (the token is read from the environment; it is a credential and is never written to
the manifest):

    export NEUPRINT_TOKEN=...
    python -m plexus.io.neuprint --out hemibrain_cube_1000 --target 1000 --skeletons --meshes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

DEFAULT_SERVER = "https://neuprint.janelia.org"
DEFAULT_DATASET = "hemibrain:v1.2.1"

# --------------------------------------------------------------------------------------- #
#  THE DATASETS, and the two things that differ between them that a cube cannot ignore.
#
#  (1) VOXELS ARE NOT ISOTROPIC AND NOT ALL THE SAME SIZE. hemibrain is a clean 8 nm cube;
#      fish2 is 16 nm in x and y and 15 nm in z, with an origin offset. So a cube measured in
#      VOXELS is not a cube in space -- on fish2 it would be 6.7% shorter along z than along
#      x, and the "44.5 um cube" a manifest claimed would be a cuboid. Every selection in this
#      module therefore happens in NANOMETRES, after the transform, and the transform is
#      recorded so the voxel coordinates remain recoverable.
#
#  (2) "WHICH BODIES ARE NEURONS" IS DATASET-SPECIFIC. On hemibrain, `status = 'Traced'` is
#      the right filter and 22,212 bodies pass it. On fish2 almost nothing is Traced -- the
#      bodies carrying somas are `Anchor` and `Orphan` (soma anchors), 177,513 of them -- so
#      the same filter would return a near-empty region. What generalises is "has a soma and
#      participates in at least one synapse": 150,764 on fish2.
#
#  Both are recorded in the manifest, because a region is only reproducible if the rule that
#  produced it travels with it.
# --------------------------------------------------------------------------------------- #
DATASETS = {
    # key is matched as a PREFIX of the dataset string, so `hemibrain:v1.2.1` finds `hemibrain`
    "hemibrain": {
        "server": "https://neuprint.janelia.org",
        "scale_nm": (8.0, 8.0, 8.0), "offset_nm": (0.0, 0.0, 0.0),
        "status": "Traced", "min_synapses": 0, "meshes": True,
        "note": "Scheffer et al. 2020. Isotropic 8 nm. 22,212 traced bodies carry a soma.",
    },
    "optic-lobe": {
        "server": "https://neuprint.janelia.org",
        "scale_nm": (8.0, 8.0, 8.0), "offset_nm": (0.0, 0.0, 0.0),
        "status": "Traced", "min_synapses": 0, "meshes": True,
        "note": "Janelia optic lobe. Isotropic 8 nm.",
    },
    "fish2": {
        "server": "https://neuprint-fish2.janelia.org",
        # papers/fishFuncEM/fishfuncem/utils/coords.py, as quoted by
        # connectome-gnn/figures/zebrafish/fetch_zebrafish_anatomy_HD.py:13 --
        #     x_nm = x_vox * 16 - 21120 * 8 ;  y_nm = y_vox * 16 - 19200 * 8 ;  z_nm = z_vox * 15
        "scale_nm": (16.0, 16.0, 15.0),
        "offset_nm": (-21120.0 * 8, -19200.0 * 8, 0.0),
        "status": None, "min_synapses": 1,
        # MEASURED: fish2 serves no usable neuron meshes. navis warns "This dataset does not
        # support LODs", and a 3-neuron request with lod=None had not returned after 180 s,
        # against 0.12 s/neuron on hemibrain. Skeletons ARE served, densely (14,105 nodes for
        # the first body against 3,054 on hemibrain), and skeletons are what this repo's own
        # zebrafish renderer draws (connectome-gnn plot_anatomy_voltage.py). So morphology
        # here means SWC, and the manifest says n_meshes = 0 rather than leaving a reader to
        # wonder whether the fetch was simply forgotten.
        "meshes": False,
        "note": ("Larval zebrafish (Petrucco et al. 2023 lineage). ANISOTROPIC: 16/16/15 nm. "
                 "Somas are Anchor/Orphan, not Traced -- filter on synapse count instead. "
                 "No neuron meshes served; morphology is SWC skeletons."),
    },
}


def dataset_profile(dataset: str) -> dict:
    """The transform and selection rule for a dataset string, by longest prefix match."""
    hits = [k for k in DATASETS if dataset.startswith(k)]
    if not hits:
        raise ValueError(
            f"dataset {dataset!r} is not in the known table ({', '.join(sorted(DATASETS))}). "
            f"Add it with its voxel->nm transform: cropping a cube in voxels on an anisotropic "
            f"dataset silently produces a cuboid, and nothing downstream can tell.")
    return DATASETS[max(hits, key=len)]


def to_nm(xyz_vox: np.ndarray, prof: dict) -> np.ndarray:
    """Dataset voxels -> nanometres. The only place the transform is applied."""
    return xyz_vox * np.asarray(prof["scale_nm"], float) + np.asarray(prof["offset_nm"], float)

# The INPUT tree. Kept apart from `graphs_data/<type>/<name>/`, which is where a spec writes
# its trajectory and which `-o generate --force` erases before every run. See the module
# docstring: a region cache put there once, and one --force deleted it.
REGION_ROOT = "neural_regions"


def region_path(name: str) -> str:
    """An absolute path for a region: `<graphs_data>/neural_regions/<name>`, or `name` if it
    is already absolute. One key, one path -- the same rule the rest of the repo follows."""
    if os.path.isabs(name):
        return name
    from plexus.paths import graphs_data_path
    return graphs_data_path(REGION_ROOT, name)

def soma_query(prof: dict) -> str:
    """The Cypher that defines "a neuron with a place", for this dataset.

    A body with no soma has no cell body to put anywhere, so it can never be a point in a
    cube. Beyond that the rule is dataset-specific (see the DATASETS table): a `status` filter
    where the dataset traces bodies, a synapse-count floor where it does not."""
    where = ["n.somaLocation IS NOT NULL"]
    if prof.get("status"):
        where.append(f"n.status = '{prof['status']}'")
    if int(prof.get("min_synapses", 0)) > 0:
        k = int(prof["min_synapses"])
        where.append(f"(coalesce(n.pre, 0) + coalesce(n.post, 0)) >= {k}")
    return ("MATCH (n:Neuron)\nWHERE " + " AND ".join(where) + "\n"
            "RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance,\n"
            "       n.somaLocation.x AS x, n.somaLocation.y AS y, n.somaLocation.z AS z,\n"
            "       n.somaRadius AS somaRadius, n.pre AS pre, n.post AS post")


# --------------------------------------------------------------------------- #
#  the cube
# --------------------------------------------------------------------------- #
def count_in_cube(xyz: np.ndarray, lo: np.ndarray, L: float) -> np.ndarray:
    """Boolean mask of the points inside the half-open cube `[lo, lo + L)`."""
    return np.all((xyz >= lo) & (xyz < lo + L), axis=1)


def densest_corner(xyz: np.ndarray, L: float, steps: int = 24) -> np.ndarray:
    """The lower corner of the cube of side `L` holding the most points, on a coarse grid.

    A COARSE SCAN AND NOT AN OPTIMUM, deliberately. Somas in the fly brain sit in a rind at
    the surface of the neuropil, so the density field is strongly non-uniform and the very
    densest cube sits flush against the edge of the imaged volume -- a cube that is half
    outside the data. `steps` grid offsets per axis is enough to land in the rind while
    leaving the result insensitive to a one-voxel change in the query.
    """
    best_n, best_lo = -1, None
    axes = [np.linspace(xyz[:, k].min(), max(xyz[:, k].max() - L, xyz[:, k].min()), steps)
            for k in range(3)]
    for x0 in axes[0]:
        mx = (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + L)
        if mx.sum() <= best_n:
            continue                                    # cannot beat the record on x alone
        for y0 in axes[1]:
            my = mx & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + L)
            if my.sum() <= best_n:
                continue
            for z0 in axes[2]:
                n = int((my & (xyz[:, 2] >= z0) & (xyz[:, 2] < z0 + L)).sum())
                if n > best_n:
                    best_n, best_lo = n, np.array([x0, y0, z0], float)
    return best_lo


def select_cube(xyz: np.ndarray, target: int = 1000, center=None,
                lo_side: float = 200.0, hi_side: float = 40000.0,
                tol: float = 0.02, max_iter: int = 40, verbose: bool = True):
    """Bisect the cube side `L` until it holds about `target` points. Returns (lo, L, mask).

    THE COUNT IS A STEP FUNCTION OF L, so an exact hit is not generally available: between
    two neighbouring soma positions the count does not change at all, and at one it jumps.
    Bisection therefore converges to a bracket, and this returns the side whose count is
    CLOSEST to the target rather than pretending to have solved N(L) = target. `tol` is a
    relative band on the count within which the search stops early.

    `center` fixes the cube's centre (a chosen anatomical location); without one, the centre
    is re-chosen at every trial side by `densest_corner`, because the densest place for a
    32 um cube is not the densest place for a 96 um one.
    """
    best = None
    for it in range(max_iter):
        L = 0.5 * (lo_side + hi_side)
        lo = (np.asarray(center, float) - 0.5 * L) if center is not None else densest_corner(xyz, L)
        m = count_in_cube(xyz, lo, L)
        n = int(m.sum())
        if best is None or abs(n - target) < abs(best[3] - target):
            best = (lo, L, m, n)
        if verbose:
            print(f"  [{it:2d}] L = {L / 1000:8.2f} um  ->  {n:5d} neurons")
        if abs(n - target) <= tol * target:
            break
        if n > target:
            hi_side = L
        else:
            lo_side = L
    lo, L, m, n = best
    if verbose:
        print(f"  chosen: L = {L / 1000:.3f} um, {n} neurons, "
              f"lo = {np.round(lo).astype(int).tolist()} nm")
    return lo, L, m


# --------------------------------------------------------------------------- #
#  the fetch
# --------------------------------------------------------------------------- #
def fetch_somas(client, prof: dict):
    """Every neuron with a soma that passes this dataset's rule. One query."""
    t = time.time()
    q = soma_query(prof)
    df = client.fetch_custom(q)
    print(f"[neuprint] {len(df)} neurons with a soma pass the rule  ({time.time() - t:.1f}s)")
    return df, q


def fetch_skeletons(client, body_ids, out_dir) -> dict:
    """`skeletons/<bodyId>.swc` for each id. Returns {bodyId: relative path}."""
    from neuprint import fetch_skeleton
    os.makedirs(out_dir, exist_ok=True)
    paths, t0 = {}, time.time()
    for k, bid in enumerate(body_ids):
        try:
            swc = fetch_skeleton(int(bid), format="swc", client=client)
        except Exception as e:                          # a body without a skeleton is skipped,
            print(f"  skeleton {bid}: {type(e).__name__}")      # not fatal -- it is recorded as
            continue                                    # absent in the manifest
        rel = os.path.join("skeletons", f"{int(bid)}.swc")
        with open(os.path.join(os.path.dirname(out_dir), rel), "w") as f:
            f.write(swc)
        paths[int(bid)] = rel
        if (k + 1) % 100 == 0:
            print(f"  skeletons {k + 1}/{len(body_ids)}  ({time.time() - t0:.0f}s)", flush=True)
    return paths


def fetch_meshes(client, body_ids, out_dir, lod: int = 2, threads: int = 8,
                 batch: int = 50) -> dict:
    """`meshes/<bodyId>.obj` for each id. Returns {bodyId: relative path}.

    BATCHED AND THREADED because the cost is dominated by a cold start: the first mesh in a
    process took 28.7 s while a warm batch of eight ran at ~0.12 s each. A serial loop over
    1,000 neurons would therefore be an eight-hour job for a two-minute amount of data.
    `lod` is capped at 3 by the server (a request for 4 is refused); 2 is the coarsest level
    that still resolves a neurite.
    """
    from navis.interfaces import neuprint as nvnp
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.dirname(out_dir)
    paths, t0 = {}, time.time()
    ids = [int(b) for b in body_ids]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        try:
            nl = nvnp.fetch_mesh_neuron(chunk, lod=lod, client=client, parallel=True,
                                        max_threads=threads)
        except Exception as e:
            print(f"  mesh batch {i}: {type(e).__name__} {str(e)[:120]}")
            continue
        for m in nl:
            bid = int(m.id)
            rel = os.path.join("meshes", f"{bid}.obj")
            m.trimesh.export(os.path.join(root, rel))
            paths[bid] = rel
        print(f"  meshes {min(i + batch, len(ids))}/{len(ids)}  ({time.time() - t0:.0f}s)",
              flush=True)
    return paths


# --------------------------------------------------------------------------- #
#  the manifest
# --------------------------------------------------------------------------- #
def write_manifest(out: str, dataset: str, server: str, prof: dict, query: str,
                   lo_nm, L_nm, df, skels, meshes) -> str:
    """The frozen region: `manifest.json` (what and where) + `neurons.npz` (the arrays).

    TWO FILES AND NOT ONE. The manifest is meant to be READ BY A HUMAN deciding whether a
    spec points at the region they think it does -- so it holds the bounds, the counts, the
    dataset string and the query, and none of the 1,000-row arrays. The arrays go in an npz
    that a seed operator loads with `np.load` and nothing else.

    THE TOKEN IS NEVER WRITTEN. It is a credential with a lifetime, it identifies its owner,
    and a manifest is exactly the sort of file that gets committed.
    """
    os.makedirs(out, exist_ok=True)
    xyz_vox = df[["x", "y", "z"]].to_numpy(np.float64)
    xyz_nm = to_nm(xyz_vox, prof)
    body = df["bodyId"].to_numpy(np.int64)
    types = df["type"].fillna("").astype(str).to_numpy()
    uniq = sorted(set(types))
    type_id = np.array([uniq.index(t) for t in types], np.int64)
    # BOTH COORDINATE SYSTEMS ARE STORED. `xyz_nm` is what the seed places neurons with and
    # what makes a distance meaningful; `xyz_vox` is what a NeuPrint query, a neuroglancer
    # link or a skeleton file speaks, so dropping it would make the region un-cross-checkable
    # against its own source.
    np.savez_compressed(
        os.path.join(out, "neurons.npz"),
        body_id=body, xyz_nm=xyz_nm, xyz_vox=xyz_vox,
        type_id=type_id, type_names=np.asarray(uniq, dtype=object),
        soma_radius=df["somaRadius"].fillna(0).to_numpy(np.float32),
        n_pre=df["pre"].fillna(0).to_numpy(np.int64),
        n_post=df["post"].fillna(0).to_numpy(np.int64),
        bounds_lo_nm=np.asarray(lo_nm, np.float64), bounds_side_nm=np.float64(L_nm),
    )
    man = {
        "source": {"server": server, "dataset": dataset, "query": query.strip(),
                   "selection_rule": {"status": prof.get("status"),
                                      "min_synapses": int(prof.get("min_synapses", 0))},
                   "voxel_to_nm": {"scale_nm": list(prof["scale_nm"]),
                                   "offset_nm": list(prof["offset_nm"]),
                                   "isotropic": len(set(prof["scale_nm"])) == 1},
                   "note": prof.get("note"),
                   "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "region": {
            # THE CUBE IS DEFINED IN NANOMETRES. On an anisotropic dataset the voxel bounds
            # below are a CUBOID -- they are recorded for provenance, and `side_um` is the one
            # number that describes the region's actual size.
            "shape": "cube",
            "bounds_lo_nm": [float(v) for v in np.asarray(lo_nm)],
            "side_nm": float(L_nm),
            "side_um": float(L_nm) / 1000.0,
            "bounds_lo_vox": [float(v) for v in
                              (np.asarray(lo_nm) - np.asarray(prof["offset_nm"]))
                              / np.asarray(prof["scale_nm"])],
            "side_vox": [float(L_nm / sc) for sc in prof["scale_nm"]],
            "n_neurons": int(len(df)),
        },
        "morphology": {
            "n_skeletons": len(skels), "n_meshes": len(meshes),
            "skeleton_dir": "skeletons" if skels else None,
            "mesh_dir": "meshes" if meshes else None,
        },
        "types": {"n_types": int(len(set(types))), "top": _top_types(types, 12)},
        "arrays": ("neurons.npz  (body_id, xyz_nm, xyz_vox, type_id, type_names, "
                   "soma_radius, n_pre, n_post, bounds_lo_nm, bounds_side_nm)"),
    }
    path = os.path.join(out, "manifest.json")
    with open(path, "w") as f:
        json.dump(man, f, indent=2)
    # the per-neuron morphology index, kept out of the manifest so it stays readable
    with open(os.path.join(out, "morphology_index.json"), "w") as f:
        json.dump({"skeletons": skels, "meshes": meshes}, f, indent=1)
    return path


def _top_types(types, k):
    vals, counts = np.unique(types, return_counts=True)
    order = np.argsort(-counts)[:k]
    return {str(vals[i]): int(counts[i]) for i in order}


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True,
                   help="region name -> graphs_data/neural_regions/<name>, or an absolute path")
    p.add_argument("--target", type=int, default=1000, help="neurons the cube should hold")
    p.add_argument("--server", default=None,
                   help="default: the server in the DATASETS table for --dataset")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--center", default=None,
                   help="fix the cube centre, 'x,y,z' in dataset voxels (default: densest)")
    p.add_argument("--skeletons", action="store_true", help="also fetch SWC skeletons")
    p.add_argument("--meshes", action="store_true", help="also fetch segmentation meshes")
    p.add_argument("--lod", type=int, default=2, help="mesh level of detail, 0 (finest) .. 3")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN")
                   or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS"))
    a = p.parse_args(argv)
    if not a.token:
        sys.exit("need a NeuPrint token: export NEUPRINT_TOKEN=... (never pass it in a file)")

    out = region_path(a.out)
    prof = dataset_profile(a.dataset)
    server = a.server or prof["server"]
    from neuprint import Client, set_default_client
    client = Client(server, dataset=a.dataset, token=a.token)
    set_default_client(client)
    iso = "isotropic" if len(set(prof["scale_nm"])) == 1 else "ANISOTROPIC"
    print(f"[neuprint] {server}  dataset={a.dataset}")
    print(f"[neuprint] voxel -> nm: scale {prof['scale_nm']} ({iso})  offset {prof['offset_nm']}")
    print(f"[neuprint] selection: status={prof.get('status')}  "
          f"min_synapses={prof.get('min_synapses', 0)}")

    df, query = fetch_somas(client, prof)
    xyz_nm = to_nm(df[["x", "y", "z"]].to_numpy(float), prof)
    ext = xyz_nm.max(0) - xyz_nm.min(0)
    print(f"[neuprint] soma extent: {ext[0] / 1000:.0f} x {ext[1] / 1000:.0f} x "
          f"{ext[2] / 1000:.0f} um")
    # THE CENTRE IS GIVEN IN VOXELS (what a neuroglancer link speaks) and converted here, so
    # the user never has to do the anisotropic arithmetic by hand.
    center = to_nm(np.asarray([[float(v) for v in a.center.split(",")]], float), prof)[0] \
        if a.center else None
    print(f"[neuprint] selecting a cube holding ~{a.target} neurons (in nm)")
    lo_nm, L_nm, mask = select_cube(xyz_nm, target=a.target, center=center,
                                    lo_side=1e3, hi_side=1e6)
    sub = df[mask].reset_index(drop=True)

    os.makedirs(out, exist_ok=True)
    skels = fetch_skeletons(client, sub.bodyId, os.path.join(out, "skeletons")) \
        if a.skeletons else {}
    if a.meshes and not prof.get("meshes", True):
        print(f"[neuprint] --meshes ignored: {a.dataset} serves no usable neuron meshes "
              f"(see the DATASETS table). Morphology for this dataset is SWC skeletons.")
    meshes = fetch_meshes(client, sub.bodyId, os.path.join(out, "meshes"),
                          lod=a.lod, threads=a.threads) \
        if (a.meshes and prof.get("meshes", True)) else {}
    path = write_manifest(out, a.dataset, server, prof, query, lo_nm, L_nm, sub, skels, meshes)
    print(f"[neuprint] wrote {path}")
    print(json.dumps(json.load(open(path))["region"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
