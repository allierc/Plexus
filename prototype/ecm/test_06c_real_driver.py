"""
06c_real_driver -- 05b's RIG, ONE VARIABLE CHANGED: the driver.

06_spheroid_bm works. Its epithelium is a driven icosphere: 642 vertices, dilating uniformly, no
cells, no division, no T1, vertex count constant. This run holds EVERY OTHER VALUE of that rig --
E, thickness, nu, kn, xi, l0, zeta, s_target, the plaque law, the substep rule -- and replaces the
icosphere with the replayed vertex model. Nothing is tuned to make it work. The movie is what comes
out as is, and it is the evidence for the claim that the real tissue is a different problem.

WHAT THE DRIVER NOW IS. `cellfix_B_new_f401`: 396 vertices at frame 0, 12,756 at the last frame,
apical radius 4.66 -> 17.58 tissue units. The mesh is triangulated from its half-edges the way
`mesh_contact` does it -- one triangle per half-edge, (face centroid, srce, trgt) -- at the FRAME-0
topology, which stays a valid triangulation of vertices 0..395 because the vertex list is only ever
appended to. So a plaque holds the same material point of the tissue for the whole run; what changes
under it is where that point goes, not which point it is.

READ THE OUTPUT AS A COMPARISON, NOT AS A RESULT. The number to put beside each line is
06_spheroid_bm's: lam_geo 3.331 against a drive of 3.398 (2% under), momentum 1.5e-16, bound 100%.
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402

CACHE = os.path.join(B.LOG, "_tissue", "cellfix_B_new_f401_x4_c4a5698982.npz")


def tri_of(pos, srce, trgt, face, nF):
    """(vertices, triangles) of the half-edge mesh: one triangle per half-edge, centroid first.

    The centroids are appended after the real vertices, so triangle indices below Nv are material
    points of the tissue and indices above it are derived. `_epi_anchor` rebuilds the centroids from
    the replayed positions every frame, which is what keeps the two consistent.
    """
    nv = pos.shape[0]
    cen = torch.zeros((nF, 3), device=pos.device, dtype=pos.dtype)
    cnt = torch.zeros((nF,), device=pos.device, dtype=pos.dtype)
    cen.index_add_(0, face, pos[srce])
    cnt.index_add_(0, face, torch.ones_like(cnt[face]))
    cen = cen / cnt.clamp(min=1.0)[:, None]
    V = torch.cat([pos, cen], dim=0)
    F = torch.stack([nv + face, srce, trgt], dim=1)
    return V, F


class Rig06c(B.Rig05b):
    """05b's rig with `_epi_anchor` reading the tissue instead of a dilating sphere."""

    def __init__(self, **P):
        super().__init__(**P)
        dev, dt = self.dev, self.dtype
        z = np.load(CACHE)
        self.z = z
        self.mesh_frames = np.asarray(z["mesh_frames"])
        self.n_mesh = len(self.mesh_frames)
        # the same scale 04 uses: put the LAST apical radius at the box radius 05b's sphere reaches,
        # so the two runs cover the same swept volume and lam_geo is comparable.
        r0, r1 = float(z["r_apical"][0]), float(z["r_apical"][-1])
        R0_box = float(self.rep_e.R(0)[0]) if self.rep_e.R(0).ndim else float(self.rep_e.R(0))
        self.scale = R0_box / r0
        self.srce = torch.as_tensor(z["m0_E_srce"].astype(np.int64), device=dev)
        self.trgt = torch.as_tensor(z["m0_E_trgt"].astype(np.int64), device=dev)
        self.face = torch.as_tensor(z["m0_E_face"].astype(np.int64), device=dev)
        self.nF = int(z["m0_nF"])
        self.nv0 = int(z["m0_pos"].shape[0])
        X0 = self._pos(0)
        V, F = tri_of(X0, self.srce, self.trgt, self.face, self.nF)
        self.x_epi, self.F_epi = V, F
        self.v_epi = torch.zeros_like(V)
        self.u_epi = (V - self.c) / (V - self.c).norm(dim=1, keepdim=True).clamp(min=1e-12)
        # re-seed the plaques against the tissue's own mesh; the sheet is untouched
        node, face, w, missed = B.seed_plaques(self.u_bm, self.u_epi, self.F_epi,
                                               fraction=P.get("fraction", 1.0))
        # `break_load` IS CARRIED OVER FROM THE PLAQUES THIS REPLACES, not left to the default.
        # Re-seeding against the tissue's mesh builds a NEW `Plaques`, and the first version of this
        # line passed l0, kn and xi and stopped -- so the rupture law silently became None. It did not
        # fail loudly: a run asking for plaques that break at 5e-3 reported 100% bound with the median
        # load at 9.7e-3, which reads as "the adhesion held" and means "the adhesion had no strength".
        self.plq = B.Plaques(node, face, w, P["l0"], P["kn"], P["xi"],
                             break_load=self.plq.break_load)
        self.n_plaque, self.missed = node.shape[0], missed
        self.n_sub = self._nsub()
        print(f"[06c] driver: tissue {self.nv0} vertices at frame 0 -> "
              f"{int(z[f'm{self.n_mesh-1}_pos'].shape[0])} at the last, apical radius "
              f"{r0:.2f} -> {r1:.2f} ({r1/r0:.2f}x); {self.F_epi.shape[0]} triangles, "
              f"{self.n_plaque} plaques ({missed} directions hit no face)", flush=True)

    def _pos(self, t):
        """Tissue vertices 0..nv0-1 in box units at pass-2 frame `t`, linearly interpolated."""
        f = min(int(self.mesh_frames[-1]), max(0, int(t)))
        j = int(np.searchsorted(self.mesh_frames, f, side="right") - 1)
        j = min(max(j, 0), self.n_mesh - 1)
        a = torch.as_tensor(self.z[f"m{j}_pos"][:self.nv0], device=self.dev, dtype=self.dtype)
        if j + 1 < self.n_mesh:
            b = torch.as_tensor(self.z[f"m{j+1}_pos"][:self.nv0], device=self.dev, dtype=self.dtype)
            span = float(self.mesh_frames[j + 1] - self.mesh_frames[j])
            al = (f - float(self.mesh_frames[j])) / span if span > 0 else 0.0
            a = (1.0 - al) * a + al * b
        return self.c + a * self.scale

    def _epi_anchor(self, t):
        V, _ = tri_of(self._pos(t), self.srce, self.trgt, self.face, self.nF)
        return V


def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    name = arg("--name", str, "06c_real_driver")
    d = os.path.join(B.LOG, name)
    os.makedirs(d, exist_ok=True)

    T = 2.0e-3
    P = dict(subdiv=4, subdiv_epi=3, E=400.0, thickness=T, nu=0.3, kn=5.0, xi=0.0,
             l0=0.3 * T, zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    rig = Rig06c(**P)
    kept, reached = B.run(rig, frames, keep=keep, label=name)
    print(f"[{name}] reached frame {reached} of {frames}"
          f"{'  -- DIVERGED' if reached != frames else ''}", flush=True)
    if len(kept) < 2:
        print(f"[{name}] nothing kept -- no movie", flush=True)
        return
    s_hi = float(np.percentile(np.concatenate([k[2] for k in kept[::4]]), 99))
    if not np.isfinite(s_hi) or s_hi <= 1.0:
        s_hi = 2.0
    B.render(kept, rig.sheet.Fc.cpu().numpy(), rig.F_epi.cpu().numpy(), d, name, s_hi)
    print(f"[{name}] movie -> {os.path.join(d, 'movie.mp4')}  ({len(kept)} frames drawn, "
          f"of the {frames} asked for)", flush=True)


if __name__ == "__main__":
    main()
