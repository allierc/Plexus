"""smg_reward -- the VALUE VECTOR, FAILURE TAXONOMY and stage rewards for the SMG search.

The landscape is deceptive: many specs make organized CLUSTERS, but clusters are OFF the causal
path to buds/ducts/branches. So the reward must penalize clusters even when they look organized,
and the surrogate must learn WHICH KIND of failure a spec produces. Both come from a single locked
morphology readout (a cluster vs a nascent duct differ in branch-genealogy + elongation + tip-growth,
not in density).

Public API:
  obs_2d(points, W)         -> morphology observable dict from a 2D point cloud
  value_vector(obs)         -> the metric VECTOR (cluster/bud/duct/branch/... in ~[0,1])
  classify(obs)             -> one failure/success label from the taxonomy
  reward(obs, stage)        -> scalar reward for a stage (early stages penalize clusters hard)

Validate (real SMG must be branch-like, low cluster):  python search/smg_reward.py
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
import networkx as nx

FAILURE_CLASSES = ["unstable", "no-growth", "overgrowth", "fragment", "blob", "sheet",
                   "cluster", "branch-like"]

# duct_score/cluster_score are NORMALIZED against the REAL SMG (score anchor: real -> duct~1,
# cluster~0), NOT absolute heuristics. calibrate() writes _calib.json from real frames.
_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_calib.json")


def _load_calib():
    import json
    d = {"BLR_REF": 2.0, "GEN_REF": 3.0}                 # defaults from the real-SMG scale sweep
    if os.path.exists(_CALIB_PATH):
        try:
            d.update(json.load(open(_CALIB_PATH)))
        except Exception:
            pass
    return d


_CALIB = _load_calib()


# ------------------------------------------------------------------ morphology readout (2D)
def obs_2d(points, W=1.0, vox=0.008, sigma_vox=4.5, thr=0.10, prune=14):
    """Locked morphology observables from a 2D point cloud in [0,W]x[0,1]."""
    nx_, ny_ = int(W / vox) + 1, int(1.0 / vox) + 1
    ix = np.clip((points[:, 0] / vox).astype(int), 0, nx_ - 1)
    iy = np.clip((points[:, 1] / vox).astype(int), 0, ny_ - 1)
    g = np.zeros((nx_, ny_), np.float32); np.add.at(g, (ix, iy), 1.0)
    dens = ndi.gaussian_filter(g, sigma_vox)
    occ = ndi.binary_fill_holes(dens > thr * max(dens.max(), 1e-9))
    lbl, ncomp = ndi.label(occ)
    o = dict(n_tube=0, n_bud=0, n_branch=0, n_generations=0, skel_len=0.0,
             body_diam=1e-6, body_aniso=1.0, sheetness=0.0, area=0.0, ncomp=ncomp)
    if ncomp == 0:
        return o
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    o["n_tube"] = int((sizes > 0.05 * sizes.max()).sum())
    body = lbl == int(sizes.argmax())
    o["area"] = float(body.sum()) * vox * vox
    ys, xs = np.nonzero(body)
    pts = np.c_[xs, ys].astype(float) * vox
    c = pts.mean(0); cov = np.cov((pts - c).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1] + 1e-12
    o["body_aniso"] = float(np.sqrt(ev[0] / ev[1]))               # elongation
    o["body_diam"] = float(4 * np.sqrt(ev[0]))
    edt = ndi.distance_transform_edt(body)
    o["sheetness"] = float(np.clip(1.0 - 2.0 * edt.max() * vox / (o["body_diam"] + 1e-9), 0, 1))
    buds = peak_local_max(edt, min_distance=5, labels=body, threshold_abs=2.0)
    o["n_bud"] = len(buds)
    skel = skeletonize(body)
    o["skel_len"] = float(skel.sum()) * vox
    # skeleton graph -> prune spurs -> branch points + generations
    o["n_branch"], o["n_generations"] = _skel_topology(skel, vox, prune * vox)
    return o


def _skel_topology(skel, vox, prune_len):
    v = np.argwhere(skel)
    if len(v) < 3:
        return 0, 0
    idx = {tuple(p): i for i, p in enumerate(v)}
    G = nx.Graph()
    G.add_nodes_from(range(len(v)))
    for i, p in enumerate(v):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    j = idx.get((p[0] + dx, p[1] + dy))
                    if j is not None and j > i:
                        G.add_edge(i, j)
    special = [n for n in G if G.degree(n) != 2]
    H = nx.MultiGraph(); H.add_nodes_from(special); seen = set()
    for s in special:
        for nb in list(G[s]):
            if (s, nb) in seen:
                continue
            prev, cur = s, nb; length = np.linalg.norm((v[s] - v[nb]) * vox); chain = [s, nb]
            while G.degree(cur) == 2:
                nxt = [x for x in G[cur] if x != prev][0]
                length += np.linalg.norm((v[cur] - v[nxt]) * vox); prev, cur = cur, nxt; chain.append(cur)
            seen.add((s, nb)); seen.add((cur, chain[-2]))
            if s != cur:
                H.add_edge(s, cur, length=length)
    changed = True
    while changed:
        changed = False
        for n in list(H.nodes):
            if n in H and H.degree(n) == 1:
                _, nb, k = list(H.edges(n, keys=True))[0]
                if H[n][nb][k]["length"] < prune_len:
                    H.remove_node(n); changed = True
        for n in list(H.nodes):
            if n in H and H.degree(n) == 2:
                (_, a, ka), (_, b, kb) = list(H.edges(n, keys=True))
                L = H[n][a][ka]["length"] + H[n][b][kb]["length"]; H.remove_node(n)
                if a != b:
                    H.add_edge(a, b, length=L)
    n_branch = sum(1 for n in H if H.degree(n) >= 3)
    # generations = max #(deg>=3) on a root->leaf path
    n_gen = 0
    if H.number_of_nodes():
        comp = max(nx.connected_components(H), key=len); T = H.subgraph(comp)
        leaves = [n for n in T if T.degree(n) == 1]
        root = leaves[0] if leaves else next(iter(comp)); gen = {root: 0}
        from collections import deque
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for w in set(T.neighbors(u)):
                if w not in gen:
                    gen[w] = gen[u] + (1 if T.degree(w) >= 3 else 0); dq.append(w)
        n_gen = max(gen.values()) if gen else 0
    return n_branch, n_gen


# ------------------------------------------------------------------ value vector
def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def value_vector(obs, migration_coherence=0.0, growth_ratio=1.0,
                 tip_growth_localization=0.0, target_distance=1.0):
    """The metric VECTOR (surrogate target + UCB value + reward inputs)."""
    conn = 1.0 if obs["n_tube"] <= 1 else 1.0 / obs["n_tube"]
    elong = _clip((obs["body_aniso"] - 1.0) / 2.0)                 # aniso 1..3 -> 0..1
    blr = obs["skel_len"] / max(obs["body_diam"], 1e-6)            # branch_length_ratio
    gens = _clip(obs["n_generations"] / _CALIB["GEN_REF"])
    dref = _clip(blr / _CALIB["BLR_REF"])                          # length NORMALIZED to real SMG
    duct = conn * dref * (0.4 + 0.6 * gens)                        # real anchors ~1 (conn=1,dref=1,gens=1)
    bud = conn * _clip(obs["n_bud"] / 8.0) * (0.4 + 0.6 * elong)   # lobules on a connected body
    # cluster = fragmented OR (connected but compact, short, no tree, several lumps)
    cluster = 0.7 * (1 - conn) + conn * (1 - gens) * (1 - dref) * _clip(obs["n_bud"] / 3.0)
    return dict(
        cluster_score=round(_clip(cluster), 3),
        bud_score=round(_clip(bud), 3),
        duct_score=round(_clip(duct), 3),
        branch_count=int(obs["n_branch"]),
        branch_length_ratio=round(float(blr), 3),
        tip_growth_localization=round(float(tip_growth_localization), 3),
        migration_coherence=round(float(migration_coherence), 3),
        target_distance=round(float(target_distance), 3),
        elongation=round(elong, 3), connectedness=round(conn, 3), generations=int(obs["n_generations"]),
    )


# ------------------------------------------------------------------ failure taxonomy
def classify(obs, growth_ratio=1.0, area_ratio=1.0, unstable=False):
    """Rule-based label on the observable vector (the surrogate learns to predict this)."""
    if unstable or area_ratio < 0.3:
        return "unstable"
    if growth_ratio > 4.0 and area_ratio > 3.0:
        return "overgrowth"
    if obs["n_tube"] >= 4:
        return "fragment"
    if obs["n_generations"] >= 2 and (obs["skel_len"] / max(obs["body_diam"], 1e-6)) > 1.5:
        return "branch-like"
    if obs["sheetness"] > 0.7:
        return "sheet"
    if growth_ratio < 1.05 and area_ratio < 1.1:
        return "no-growth"
    if obs["n_bud"] >= 3:
        return "cluster"                                          # several lumps, no tree
    return "blob"                                                 # single round mass


# ------------------------------------------------------------------ stage rewards
STAGE_WEIGHTS = {
    # early: get OFF clusters onto connected elongated ducts before anything else
    "connect":  dict(duct=1.0, bud=0.2, branch=0.0, cluster=-1.5, migr=0.3),
    "bud":      dict(duct=0.6, bud=1.0, branch=0.2, cluster=-1.0, migr=0.2),
    "branch":   dict(duct=0.6, bud=0.6, branch=1.0, cluster=-0.8, migr=0.1),
    "match":    dict(duct=0.4, bud=0.5, branch=0.6, cluster=-0.5, migr=0.2),  # + target_distance
}


def reward(obs, stage="connect", **kw):
    v = value_vector(obs, **kw)
    w = STAGE_WEIGHTS[stage]
    r = (w["duct"] * v["duct_score"] + w["bud"] * v["bud_score"]
         + w["branch"] * _clip(v["branch_count"] / 4.0)
         + w["cluster"] * v["cluster_score"] + w["migr"] * v["migration_coherence"])
    if stage == "match":
        r -= 0.8 * v["target_distance"]
    return round(float(r), 3), v, classify(obs, growth_ratio=kw.get("growth_ratio", 1.0))


# ------------------------------------------------------------------ calibration to real SMG
def _real_obs(frame=552):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch, smg_topo as st
    xl = torch.load(st.PT_DEFAULT, map_location="cpu", weights_only=False)
    p = st.load_frame(xl, frame)[:, :2]
    p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-9)
    return obs_2d(p, W=1.0)


def calibrate(frames=(184, 276, 368, 460, 552)):
    """Anchor the reward to real SMG (score anchor: real -> duct~1, cluster~0). BLR_REF/GEN_REF =
    median real branch-length-ratio / generations. Writes _calib.json + updates the module global."""
    import json
    blrs, gens = [], []
    for f in frames:
        o = _real_obs(f)
        blrs.append(o["skel_len"] / max(o["body_diam"], 1e-6)); gens.append(o["n_generations"])
    calib = {"BLR_REF": round(float(np.median(blrs)), 3), "GEN_REF": float(max(2.0, np.median(gens)))}
    json.dump(calib, open(_CALIB_PATH, "w"), indent=2)
    global _CALIB; _CALIB = _load_calib()
    return calib, blrs, gens


def _synthetic(rng):
    C = rng.uniform(0.15, 0.85, (6, 2))                                        # cluster: 6 blobs
    cluster = np.clip(np.concatenate([c + 0.02 * rng.standard_normal((400, 2)) for c in C]), 0, 1)
    blob = np.clip([0.5, 0.5] + 0.10 * rng.standard_normal((1500, 2)), 0, 1)   # one round mass
    t = np.linspace(0, 1, 500)[:, None]                                        # thick Y branch
    ybr = np.concatenate([np.c_[0.5 + 0 * t, 0.12 + 0.42 * t],
                          np.c_[0.5 - 0.32 * t, 0.54 + 0.32 * t],
                          np.c_[0.5 + 0.32 * t, 0.54 + 0.32 * t]])
    branch = np.clip(ybr + 0.025 * rng.standard_normal(ybr.shape), 0, 1)
    return dict(cluster=obs_2d(cluster, W=1.0), blob=obs_2d(blob, W=1.0), branch=obs_2d(branch, W=1.0))


def calibration_gate():
    """MUST pass before bootstrap/UCB: real SMG strongly separates from cluster/blob/fragment in the
    VALUE VECTOR, not just the class label."""
    calib, blrs, gens = calibrate()
    print(f"calibrated to real SMG: BLR_REF={calib['BLR_REF']} GEN_REF={calib['GEN_REF']}  "
          f"(real blr {np.round(blrs, 2).tolist()}, gen {gens})\n")
    ro = _real_obs(552); real = value_vector(ro); real_cl = classify(ro)
    syn = _synthetic(np.random.default_rng(0))
    print(f"{'case':10} {'cluster':>7} {'bud':>5} {'duct':>5} {'branch':>6}  class")
    print(f"{'REAL':10} {real['cluster_score']:>7} {real['bud_score']:>5} {real['duct_score']:>5} "
          f"{real['branch_count']:>6}  {real_cl}")
    vs = {}
    for name, o in syn.items():
        vs[name] = value_vector(o)
        print(f"{name:10} {vs[name]['cluster_score']:>7} {vs[name]['bud_score']:>5} "
              f"{vs[name]['duct_score']:>5} {vs[name]['branch_count']:>6}  {classify(o)}")
    checks = {
        "real duct > 0.6": real["duct_score"] > 0.6,
        "real cluster < 0.2": real["cluster_score"] < 0.2,
        "real class = branch-like": real_cl == "branch-like",
        "real bud > cluster bud + 0.05": real["bud_score"] > vs["cluster"]["bud_score"] + 0.05,
        "real duct > cluster duct + 0.5": real["duct_score"] > vs["cluster"]["duct_score"] + 0.5,
        "real duct > blob duct + 0.4": real["duct_score"] > vs["blob"]["duct_score"] + 0.4,
    }
    print()
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    passed = all(checks.values())
    print(f"\n=== CALIBRATION GATE: {'PASS — reward approved, search may start'
                                   if passed else 'FAIL — do NOT start search'} ===")
    return passed


if __name__ == "__main__":
    calibration_gate()
