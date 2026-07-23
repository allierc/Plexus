"""build_flat_mesh -- an OPEN planar half-edge patch (a flat 2D epithelium) for the monolayer demos.
Jittered k x k grid -> planar Voronoi (guard ring so interior regions are bounded); keep the bounded
cells well inside [0,L]^2. Faces reoriented CCW (normal +z). Returns verts[Nv,3] (z=0), es/et/ef, nF,
and a boundary-vertex mask (a vertex on an edge used by only one face). The monolayer operator works on
this open patch because each cell's prism (apical+basal cap + lateral quads) is self-contained."""
from __future__ import annotations
import numpy as np
from scipy.spatial import Voronoi


def build_flat_mesh(k=12, L=10.0, jitter=0.55, seed=1):
    g = np.random.default_rng(seed)
    xs = np.linspace(0.0, L, k)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1).astype(np.float64)
    h = L / (k - 1)
    pts = pts + jitter * h * (g.random(pts.shape) - 0.5)
    guard = []                                              # far guard ring -> interior regions are bounded
    for gx in np.linspace(-L, 2 * L, 3 * k):
        guard += [[gx, -L], [gx, 2 * L], [-L, gx], [2 * L, gx]]
    vor = Voronoi(np.vstack([pts, np.array(guard, np.float64)]))
    faces = []
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if len(reg) < 3 or -1 in reg:
            continue
        P = vor.vertices[reg]
        if P[:, 0].min() < -0.01 or P[:, 0].max() > L + 0.01 or P[:, 1].min() < -0.01 or P[:, 1].max() > L + 0.01:
            continue                                        # drop border cells that leak past the patch edge
        faces.append(np.asarray(reg, np.int64))
    used = sorted({v for rr in faces for v in rr})          # compact the vertex set the kept cells use
    remap = {v: i for i, v in enumerate(used)}
    V = vor.vertices[used]
    faces = [np.array([remap[v] for v in rr], np.int64) for rr in faces]
    for idx, rr in enumerate(faces):                        # reorient CCW (signed area > 0, normal +z)
        P = V[rr]
        a2 = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        if a2 < 0:
            faces[idx] = rr[::-1]
    es, et, ef = [], [], []
    for f, rr in enumerate(faces):
        kk = len(rr)
        for j in range(kk):
            es.append(int(rr[j])); et.append(int(rr[(j + 1) % kk])); ef.append(f)
    es, et, ef = np.array(es, np.int64), np.array(et, np.int64), np.array(ef, np.int64)
    verts = np.concatenate([V, np.zeros((len(V), 1))], 1)   # lift to z=0
    # boundary vertex = endpoint of an undirected edge that only one face uses
    key = np.minimum(es, et) * (len(V) + 1) + np.maximum(es, et)
    uniq, cnt = np.unique(key, return_counts=True)
    bnd_key = set(uniq[cnt == 1].tolist())
    bmask = np.zeros(len(V), bool)
    for a, b in zip(es, et):
        if (min(a, b) * (len(V) + 1) + max(a, b)) in bnd_key:
            bmask[a] = True; bmask[b] = True
    return verts, es, et, ef, len(faces), bmask


if __name__ == "__main__":
    import torch
    from tyssue_monolayer import monolayer_geometry_3d
    from tyssue_ops3d import face_geometry_3d
    verts, es, et, ef, nF, bmask = build_flat_mesh()
    print(f"flat patch: {nF} cells, {len(verts)} verts, {int(bmask.sum())} boundary verts")
    pos = torch.as_tensor(verts); H0 = 0.4
    hc = torch.full((nF,), H0)
    v_f, s_f, A_ap, A_ba = monolayer_geometry_3d(pos, torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF, hc)
    area, _, _, _ = face_geometry_3d(pos, torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
    print(f" all v_f>0: {bool((v_f>0).all())}  sum(v_f)={float(v_f.sum()):.3f}  A_tot*h0={float(area.sum())*H0:.3f}  ratio={float(v_f.sum())/(float(area.sum())*H0):.4f}")
    print(f" FLAT: A_apical/A_basal median={float((A_ap/A_ba.clamp(min=1e-9)).median()):.4f} (expect 1.0, sheet is flat)")
