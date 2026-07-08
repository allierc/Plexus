"""Topological analysis of SMG2 branching morphogenesis from cell point clouds.

Goal: from per-frame 3D nuclei centroids, count the epithelial topology
      * main tube(s)  -> connected components of the gland body
      * branching      -> bifurcations of the coarse ductal skeleton
      * budding        -> lobules / terminal end-buds (fine-scale protrusions)
and track these across the 553-frame movie.

Why not a "standard image library" one-liner: centroids carry no connectivity,
and budding vs branching live at DIFFERENT SPATIAL SCALES, so we build an
explicit shape model and use topological persistence (implemented by hand,
no gudhi/ripser needed) to get counts that are stable to threshold/noise.

Pipeline (voxel + persistence; no mesh needed for the counts):
  points --KDE--> density field
    * coarse density (large sigma, lobules merged) --> occupancy solid
        --> #components = main tubes
        --> 3D skeleton --> junction graph --> persistence(length)-pruned
        --> #(degree>=3 nodes) = branching
    * fine density (small sigma) --> superlevel-set H0 persistence of peaks
        --> #(peaks with persistence > tau) = buds (lobules)

See README.md for the SOTA discussion (mesh / L1-skeleton / Reeb-graph
alternatives) and why this route was chosen.
"""
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import networkx as nx

PT_DEFAULT = "/workspace/ParticleGraph/graphs_data/cell/cell_gland_SMG2_smooth2/x_list_0.pt"


# --------------------------------------------------------------------------- data
def load_frame(pts_or_xl, frame):
    """Accept either a preloaded x_list or a path; return (N,3) points."""
    xl = pts_or_xl
    return xl[frame].numpy()[:, 1:4].astype(np.float64)


# --------------------------------------------------------------------------- density
def density_grid(pts, vox=4.0, sigma_um=9.0, pad_um=24.0, bounds=None):
    """Rasterize points to an isotropic grid and Gaussian-smooth -> density."""
    if bounds is None:
        lo = pts.min(0) - pad_um
        hi = pts.max(0) + pad_um
    else:
        lo, hi = bounds
        lo = np.asarray(lo) - pad_um
        hi = np.asarray(hi) + pad_um
    dims = np.ceil((hi - lo) / vox).astype(int)
    idx = np.clip(np.floor((pts - lo) / vox).astype(int), 0, dims - 1)
    g = np.zeros(dims, np.float32)
    np.add.at(g, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    g = ndi.gaussian_filter(g, sigma_um / vox)
    return g, lo


# --------------------------------------------------------------------------- main tubes / occupancy
def occupancy(density, rel_thresh=0.15, min_frac=0.05):
    """Threshold density -> solid; return (largest_component, n_main_tubes)."""
    occ = density > rel_thresh * density.max()
    occ = ndi.binary_fill_holes(occ)
    lbl, n = ndi.label(occ)
    if n == 0:
        return occ, 0
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    n_main = int((sizes >= min_frac * sizes.max()).sum())
    largest = lbl == int(sizes.argmax())
    return largest, n_main


# --------------------------------------------------------------------------- buds: superlevel-set H0 persistence
def persistence_peaks(field, floor_rel=0.05):
    """H0 persistence of superlevel sets of `field`.

    Sweep voxels high->low, union-find components; when two peaks merge at a
    saddle, the lower-born one dies with persistence = birth - saddle.
    Returns persistences (descending), normalized to field.max().
    """
    fmax = float(field.max())
    if fmax <= 0:
        return np.array([])
    floor = floor_rel * fmax
    shape = field.shape
    flat = field.ravel()
    sel = np.where(flat > floor)[0]
    order = sel[np.argsort(flat[sel])[::-1]]        # descending value
    parent = np.full(flat.size, -1, np.int64)
    birth = np.zeros(flat.size, np.float32)
    strides = np.array([shape[1] * shape[2], shape[2], 1], np.int64)
    nb = np.array([strides[0], -strides[0], strides[1], -strides[1],
                   strides[2], -strides[2]])
    # precompute coords for boundary checks
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    persist = []
    S0, S1, S2 = shape
    for v in order:
        z = v % S2
        y = (v // S2) % S1
        x = v // (S1 * S2)
        parent[v] = v
        birth[v] = flat[v]
        roots = set()
        # 6-neighbours, with bounds
        if x > 0 and parent[v - strides[0]] >= 0: roots.add(find(v - strides[0]))
        if x < S0 - 1 and parent[v + strides[0]] >= 0: roots.add(find(v + strides[0]))
        if y > 0 and parent[v - strides[1]] >= 0: roots.add(find(v - strides[1]))
        if y < S1 - 1 and parent[v + strides[1]] >= 0: roots.add(find(v + strides[1]))
        if z > 0 and parent[v - strides[2]] >= 0: roots.add(find(v - strides[2]))
        if z < S2 - 1 and parent[v + strides[2]] >= 0: roots.add(find(v + strides[2]))
        if not roots:
            continue                                # new peak born
        roots = list(roots)
        heights = [birth[r] for r in roots]
        win = roots[int(np.argmax(heights))]
        parent[v] = win
        for r in roots:
            if r != win:
                parent[r] = win
                persist.append(birth[win] if False else birth[r] - flat[v])
    persist = np.sort(np.array(persist))[::-1] / fmax
    return persist


def count_buds(density, tau_rel=0.12, floor_rel=0.05):
    """#buds = #(persistent peaks) = features with persistence>tau, +1 global."""
    p = persistence_peaks(density, floor_rel=floor_rel)
    return int((p > tau_rel).sum()) + 1, p


# --------------------------------------------------------------------------- branching: skeleton -> pruned junction graph
def _skeleton_graph(occ):
    skel = skeletonize(occ)
    vox = np.argwhere(skel)
    if len(vox) == 0:
        return nx.Graph(), vox
    index = {tuple(p): i for i, p in enumerate(vox)}
    G = nx.Graph()
    G.add_nodes_from(range(len(vox)))
    for i, p in enumerate(vox):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    j = index.get((p[0] + dx, p[1] + dy, p[2] + dz))
                    if j is not None and j > i:
                        G.add_edge(i, j)
    return G, vox


def _junction_graph(G, vox, vox_um):
    """Contract degree-2 chains -> multigraph of endpoints/junctions w/ length."""
    special = [n for n in G if G.degree(n) != 2]
    H = nx.MultiGraph()
    H.add_nodes_from(special)
    seen = set()
    for s in special:
        for nb in list(G[s]):
            if (s, nb) in seen:
                continue
            prev, cur = s, nb
            length = np.linalg.norm((vox[s] - vox[nb]) * vox_um)
            chain = [s, nb]
            while G.degree(cur) == 2:
                nxt = [x for x in G[cur] if x != prev][0]
                length += np.linalg.norm((vox[cur] - vox[nxt]) * vox_um)
                prev, cur = cur, nxt
                chain.append(cur)
            seen.add((s, nb))
            seen.add((cur, chain[-2]))
            if s != cur:
                H.add_edge(s, cur, length=length)
    return H


def _dissolve_deg2(H):
    """Contract degree-2 nodes, merging their two incident edge lengths."""
    for n in list(H.nodes):
        if n in H and H.degree(n) == 2:
            (_, a, ka), (_, b, kb) = list(H.edges(n, keys=True))  # (n, nbr, key)
            L = H[n][a][ka]["length"] + H[n][b][kb]["length"]
            H.remove_node(n)
            if a != b:
                H.add_edge(a, b, length=L)


def count_branches(occ, vox_um, prune_um=45.0):
    """Persistence(length)-prune leaf spurs, then count degree>=3 junctions."""
    G, vox = _skeleton_graph(occ)
    if len(vox) == 0:
        return 0, 0, None
    H = _junction_graph(G, vox, vox_um)
    _dissolve_deg2(H)
    changed = True
    while changed:
        changed = False
        for n in list(H.nodes):
            if n in H and H.degree(n) == 1:
                _, nbr, k = list(H.edges(n, keys=True))[0]
                if H[n][nbr][k]["length"] < prune_um:
                    H.remove_node(n)
                    changed = True
        _dissolve_deg2(H)
    n_branch = sum(1 for n in H if H.degree(n) >= 3)
    n_tips = sum(1 for n in H if H.degree(n) == 1)
    return n_branch, n_tips, H


# --------------------------------------------------------------------------- buds: inscribed-ellipsoid (medial radius) peaks
def _graph_persistence(G, value):
    """H0 persistence of the superlevel sets of a node function on a graph.

    Sweep nodes high->low, union neighbours already added; when two peaks meet
    the lower dies with persistence = its peak - the merge (saddle) value.
    Returns persistences (descending); the surviving global peak is excluded.
    """
    order = sorted(G.nodes, key=lambda n: value[n], reverse=True)
    parent, peak = {}, {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    persist = []
    for n in order:
        parent[n] = n
        peak[n] = value[n]
        roots = {find(m) for m in G[n] if m in parent}
        if not roots:
            continue
        roots = list(roots)
        win = max(roots, key=lambda r: peak[r])
        parent[n] = win
        for r in roots:
            if r != win:
                parent[r] = win
                persist.append(peak[r] - value[n])   # saddle at value[n]
    return sorted(persist, reverse=True)


def count_buds_medial(occ, vox_um, prom_um=12.0, min_radius_um=6.0):
    """Buds = prominent inscribed-radius maxima on the medial axis.

    radius(node) = EDT distance-to-surface at that skeleton voxel (the maximal
    inscribed sphere = the local ellipsoid primitive). A bulbous lobule is a
    persistent radius peak; thin ducts are valleys. Prominence (graph
    persistence of radius) filters noise.
    """
    edt = ndi.distance_transform_edt(occ) * vox_um
    G, vox = _skeleton_graph(occ)
    if len(vox) == 0:
        return 0, None
    radius = {i: float(edt[tuple(vox[i])]) for i in G.nodes}
    pers = _graph_persistence(G, radius)
    # count persistent peaks + the surviving global peak (if it is a real bulb)
    n = int(np.sum(np.array(pers) > prom_um))
    if max(radius.values()) > min_radius_um:
        n += 1
    return n, radius


# --------------------------------------------------------------------------- one frame (primitive model)
def analyze_frame(pts, vox=4.0, sigma=14.0, thr=0.15,
                  prune_um=55.0, prom_um=12.0, bounds=None):
    """Single medial-axis (union-of-inscribed-ellipsoids) model gives all counts.

    * main tube = # connected components of the solid
    * branching = # degree>=3 centerline junctions (Y-tubes), spur-pruned
    * budding   = # prominent inscribed-radius peaks (fat ellipsoids / lobules)
    """
    d, _ = density_grid(pts, vox=vox, sigma_um=sigma, bounds=bounds)
    occ, n_main = occupancy(d, rel_thresh=thr)
    n_branch, n_tips, _ = count_branches(occ, vox_um=vox, prune_um=prune_um)
    n_buds, _ = count_buds_medial(occ, vox_um=vox, prom_um=prom_um)
    return dict(n_buds=n_buds, n_branch=n_branch, n_main=n_main, n_tips=n_tips)
