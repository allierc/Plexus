"""Contact sheet of the 3D scene data fetched for experiments 6-15 (2026-09-26): one panel per source."""
import glob, os, struct, numpy as np, zarr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
E = "/workspace/Plexus/experiments"
import sys; sys.path.insert(0, '/workspace/Plexus/tools'); from ngmesh import legacy as legacy_mesh, multilod_draco
def vtp(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(path); r.Update(); pd = r.GetOutput()
    v = vtk_to_numpy(pd.GetPoints().GetData()); c = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    lab = None
    if pd.GetPointData().GetNumberOfArrays(): lab = vtk_to_numpy(pd.GetPointData().GetArray(0))
    return v, c, lab
fig = plt.figure(figsize=(20, 10), facecolor="white")
# 1 salivary EM slice
z = zarr.open(f"{E}/exp11_bm_hole_budding/data/openorganelle_jrc_mus-salivary-1/em_fibsem-uint8.zarr", mode="r")["s4"]
ax = fig.add_subplot(2, 4, 1); ax.imshow(z[465], cmap="gray"); ax.set_axis_off(); ax.text(0, -20, "exp11  mouse salivary gland, FIB-SEM, 128 nm voxels, mid z-slice", fontsize=9)
# 2,3 LSTree budding organoid cells + nuclei
B = f"{E}/exp10_organoid_crypt/data/LSTree_fmi-basel/example/data/002-Budding"
for k, (sub, title) in enumerate([("cell_mesh", "cell meshes"), ("nuclei_mesh", "nucleus meshes")]):
    v, f, lab = vtp(sorted(glob.glob(f"{B}/{sub}/*.vtp"))[0])
    ax = fig.add_subplot(2, 4, 2 + k, projection="3d"); tri = v[f[::3]]
    col = plt.cm.tab20((lab[f[::3, 0]] % 20) / 20) if lab is not None else "tan"
    ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none", alpha=0.9))
    lo, hi = v.min(0), v.max(0); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_axis_off()
    ax.set_title(f"exp10  LSTree budding organoid, {title}, T0301 ({len(v)} vertices)", fontsize=9)
# 4 McDole embryo, the full predicted tracks (linajea), one late frame
import zipfile, io, pandas as pd
zf = zipfile.ZipFile(f"{E}/exp10_organoid_crypt/data/McDole_2018_mouse_embryo_tracks/predicted_tracks.zip")
t = pd.read_csv(zf.open("predicted_tracks/linajea/mouse_full_tracks_071621.txt"), sep=", ", engine="python", usecols=["t", "z", "y", "x"])
T = int(t.t.max() * 0.9); f = t[t.t == T]
ax = fig.add_subplot(2, 4, 4, projection="3d"); ax.scatter(f.x, f.y, f.z, s=0.3, c=f.z, cmap="viridis"); ax.set_axis_off()
ax.set_title(f"exp10/07  McDole 2018 mouse embryo, linajea tracks, frame {T} ({len(f)} cells)", fontsize=9)
# 5-8 OpenOrganelle meshes
for k, (exp, ds, title) in enumerate([("exp06_excitation_wave", "jrc_zf-cardiac-1", "zebrafish heart, labelled crop"),
                                      ("exp06_excitation_wave", "jrc_mus-heart-1", "mouse heart, nuclei"),
                                      ("exp12_tumor_invasion", "jrc_sum159-1", "SUM159 breast cancer cell, labelled crop"),
                                      ("exp14_tissue_maintenance", "jrc_mus-skin-1", "mouse skin, nuclei")]):
    root = f"{E}/{exp}/data/openorganelle_{ds}/neuroglancer/mesh"
    frags = [p for p in glob.glob(f"{root}/**/*:0", recursive=True)][:60]
    if not frags:
        d = f"{root}/nucleus_seg"; segs = sorted({int(x.split('.')[0]) for x in os.listdir(d) if x.split('.')[0].isdigit()})[:80]
        frags = [(d, sg) for sg in segs]
    ax = fig.add_subplot(2, 4, 5 + k, projection="3d"); allv = []
    for i, p in enumerate(frags):
        v, f = multilod_draco(p[0], p[1], lod=2) if isinstance(p, tuple) else legacy_mesh(p)
        if len(f) == 0: continue
        ax.add_collection3d(Poly3DCollection(v[f[::max(1, len(f) // 4000)]], facecolors=plt.cm.tab20(i % 20), edgecolor="none", alpha=0.8)); allv.append(v)
    V = np.concatenate(allv); lo, hi = V.min(0), V.max(0); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_axis_off()
    ax.set_title(f"{exp[:5]}  {title} ({len(frags)} largest meshes)", fontsize=9)
plt.tight_layout(); out = f"{E}/figs/data3d_contact_sheet.png"; os.makedirs(os.path.dirname(out), exist_ok=True); plt.savefig(out, dpi=80); print(out)
