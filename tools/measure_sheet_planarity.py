import sys, numpy as np
from plexus.models.topology import rings_from_flat_3d, _check_surface
def rings_at(z, t):
    off = z["vertex__mesh_offsets"]; a, b = int(off[t]), int(off[t + 1]); nF = int(z["vertex__mesh_nF"][t])
    return rings_from_flat_3d(z["vertex__mesh_E_srce"][a:b], z["vertex__mesh_E_trgt"][a:b], z["vertex__mesh_E_face"][a:b], nF)
for run in sys.argv[1:]:
    z = np.load(f"{run}/trajectory.npz", allow_pickle=True)
    pos = z["vertex__pos"]; T = pos.shape[0] - 1
    rows = []
    for t in (0, 150, 200, 250, 300, T):
        r = rings_at(z, t); F = sum(1 for rg in r if rg is not None and len(rg) >= 3)
        used = sorted({int(u) for rg in r if rg is not None for u in rg})
        p = pos[t][used]; zc = p[:, 2] - p[:, 2].mean()
        el = np.mean([np.linalg.norm(pos[t][rg[i]] - pos[t][rg[(i + 1) % len(rg)]]) for rg in r if rg is not None for i in range(len(rg))])
        tri = sum(1 for rg in r if rg is not None and len(rg) == 3)
        ok, V, E, Fc, eu, loops = _check_surface(r)
        rows.append(f"  t={t:3d} live cells={F:3d} triangles={tri:2d} out-of-plane rms={np.sqrt((zc**2).mean())/el:.3f} max={np.abs(zc).max()/el:.2f} edges  euler={eu} rim loops={loops} ok={ok}")
    print(run); print("\n".join(rows))
