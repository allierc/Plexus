"""
05l -- G43 and G44 on the REAL epithelium: `bm_secrete` and `bm_refine` under the tissue driver.

THE SAME ONE-VARIABLE MOVE AS 06c, APPLIED ONE RIG UP. 06c took 05b's certified rig and swapped the
driven icosphere for the replayed vertex model, changing nothing else, and the sheet tracked. 05f's
rig is 05b's plus refinement, the mass balance and the supply-driven tear, all certified against the
same icosphere. So the swap is the same swap: keep every value 05f published, replace `_epi_anchor`.

WHY THESE TWO GATES NEEDED IT. 05k measured the sheet on the real tissue with supply and refinement
OFF, and the two failures it found are exactly the two operators that were off:
  G44  mean edge reached 3.63x its seeded length at a 3.77x dilation -- no refinement, so the mesh
       just stretches. The number is the demand: how much `bm_refine` a real surface asks for.
  G43  rho was not even reported, because with no secretion mass is fixed and rho = m/A simply
       falls as 1/J. There is nothing to test until supply is on.
Both thresholds are 05f's own, unchanged: rho within 10% of seeded, mean edge in [0.8, 1.7]x.

WHAT WOULD MAKE THIS A FAILURE RATHER THAN A NUMBER. A real epithelium is not a dilating sphere: it
divides, so its area grows unevenly, and `bm_secrete`'s homeostatic rate s* = rho_target(1/tau + Adot/A)
reads Adot/A globally. If growth is local, a global rate over-supplies the slow regions and starves
the fast ones, and rho spreads even though its mean is held. So rho's SPREAD is reported next to its
mean, and a passing mean with a widening p05-p95 is not a pass.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402
import test_05f_secrete as F5                                            # noqa: E402
from test_06c_real_driver import CACHE, tri_of                           # noqa: E402


class RealDriver:
    """MIXIN: replace whatever epithelium the rig below built with the replayed vertex model.

    Written once and applied at two levels -- 05f's rig here, 05h1's protease rig in 05m -- because
    the swap is the same swap every time: keep every published value, change `_epi_anchor`. Anything
    the rig sized to the OLD epithelium (the contact set's face indices, the per-cell receptor pool)
    is re-pointed or refilled here, and asserted, since a stale index into a mesh that no longer
    exists surfaces as a device-side assert a long way from its cause.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        dev, dt = self.dev, self.dtype
        z = np.load(CACHE)
        self.z = z
        self.mesh_frames = np.asarray(z["mesh_frames"])
        self.n_mesh = len(self.mesh_frames)
        r0 = float(z["r_apical"][0])
        R0 = self.rep_e.R(0)
        self.scale = float(R0.mean() if R0.ndim else R0) / r0
        self.srce = torch.as_tensor(z["m0_E_srce"].astype(np.int64), device=dev)
        self.trgt = torch.as_tensor(z["m0_E_trgt"].astype(np.int64), device=dev)
        self.face = torch.as_tensor(z["m0_E_face"].astype(np.int64), device=dev)
        self.nF, self.nv0 = int(z["m0_nF"]), int(z["m0_pos"].shape[0])
        V, F = tri_of(self._pos(0), self.srce, self.trgt, self.face, self.nF)
        self.x_epi, self.F_epi, self.v_epi = V, F, torch.zeros_like(V)
        self.u_epi = (V - self.c) / (V - self.c).norm(dim=1, keepdim=True).clamp(min=1e-12)
        # RE-POINT THE EXISTING PLAQUES, do not rebuild them. 05f's adhesion is not 05b's -- it carries
        # turnover, k_on/k_off and a bound flag -- and replacing the object would silently drop the
        # kinetics this rig's own gates were certified with. Only the anchor changes: each plaque keeps
        # its node and finds the face of the NEW epithelium its node's direction points at.
        d0 = self.sheet.x - self.c
        u_bm = d0 / d0.norm(dim=1, keepdim=True).clamp(min=1e-12)
        node_all, face_all, w_all, missed = B.seed_plaques(u_bm, self.u_epi, self.F_epi, fraction=1.0)
        lf = torch.zeros(u_bm.shape[0], dtype=face_all.dtype, device=dev)
        lw = torch.zeros((u_bm.shape[0], 3), dtype=w_all.dtype, device=dev)
        lf[node_all], lw[node_all] = face_all, w_all
        # 05d/05f hold the contact set as ct_node/ct_face/ct_w on the rig; 05b holds it as rig.plq.
        # Re-point whichever exists, and assert the result indexes the NEW mesh -- a stale face index
        # into a 5,120-triangle icosphere is a device-side assert 1,188 triangles later, which is a
        # long way from the line that caused it.
        if hasattr(self, "ct_node"):
            self.ct_face, self.ct_w = lf[self.ct_node], lw[self.ct_node]
            n_plq, fmax = int(self.ct_node.shape[0]), int(self.ct_face.max())
        else:
            self.plq.face, self.plq.w = lf[self.plq.node], lw[self.plq.node]
            n_plq, fmax = int(self.plq.node.shape[0]), int(self.plq.face.max())
        assert fmax < self.F_epi.shape[0], (fmax, self.F_epi.shape)
        # THE RECEPTOR POOL IS PER EPITHELIAL CELL, so swapping the epithelium resizes it: 1,280
        # icosphere faces become 1,188 tissue triangles. Refill at the seeded per-cell amount rather
        # than scattering the old pool, because the two meshes have no cell-to-cell correspondence and
        # inventing one would put G30's conservation claim on a mapping nobody certified.
        cl = getattr(self, "clutch", None)
        if cl is not None and cl.Nf is not None and cl.Nf.shape[0] != self.F_epi.shape[0]:
            nf0 = float(cl.Nf.mean())
            cl.Nf = torch.full((self.F_epi.shape[0],), nf0, device=cl.Nf.device, dtype=cl.Nf.dtype)
            print(f"[{self.__class__.__name__}] receptor pool resized to {self.F_epi.shape[0]} cells at N_f {nf0:.4g} each",
                  flush=True)
        self.n_plaque, self.missed = n_plq, missed
        self.n_sub = self._nsub()
        print(f"[{self.__class__.__name__}] real driver: {self.nv0} tissue vertices -> {self.F_epi.shape[0]} triangles, "
              f"{self.n_plaque} plaques, max_refine {getattr(self, 'max_refine', 0)}", flush=True)

    def _pos(self, t):
        f = min(int(self.mesh_frames[-1]), max(0, int(t)))
        j = min(max(int(np.searchsorted(self.mesh_frames, f, side="right") - 1), 0), self.n_mesh - 1)
        a = torch.as_tensor(self.z[f"m{j}_pos"][:self.nv0], device=self.dev, dtype=self.dtype)
        if j + 1 < self.n_mesh:
            b = torch.as_tensor(self.z[f"m{j+1}_pos"][:self.nv0], device=self.dev, dtype=self.dtype)
            span = float(self.mesh_frames[j + 1] - self.mesh_frames[j])
            al = (f - float(self.mesh_frames[j])) / span if span > 0 else 0.0
            a = (1.0 - al) * a + al * b
        return self.c + a * self.scale

    def _epi_anchor(self, t):
        return tri_of(self._pos(t), self.srce, self.trgt, self.face, self.nF)[0]

class Rig05l(RealDriver, F5.Rig05f):
    """05f's rig -- refinement, mass balance, tear -- on the real tissue."""


def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)

    # 05f's published values, unchanged
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=7.0, zeta=20.0,
             s_target=1.0, k_drive=50.0, dev=dev)
    Q = dict(max_refine=2, edge_trigger=1.45, reseed=True, tau_bm=40.0, rho_crit=0.0)

    rig = Rig05l(**P, **Q)
    e0 = float((rig.sheet.x[rig.sheet.Ed[:, 1]] - rig.sheet.x[rig.sheet.Ed[:, 0]]).norm(dim=1).mean())
    S = {k: [] for k in ("t", "rho", "rho_p05", "rho_p95", "edge", "n_face", "lam")}
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[05l] DIVERGED at {t}", flush=True)
            break
        rho = (rig.sheet.areal_density() / rig.sheet.rho0).float().cpu().numpy()
        X = rig.sheet.x
        l1, _ = rig.sheet.stretch_geo()
        S["t"].append(t)
        S["rho"].append(float(np.mean(rho)))
        S["rho_p05"].append(float(np.percentile(rho, 5)))
        S["rho_p95"].append(float(np.percentile(rho, 95)))
        S["edge"].append(float((X[rig.sheet.Ed[:, 1]] - X[rig.sheet.Ed[:, 0]]).norm(dim=1).mean()) / e0)
        S["n_face"].append(int(rig.sheet.Fc.shape[0]))
        S["lam"].append(float(l1.mean()))

    g43 = abs(S["rho"][-1] - 1.0)
    spread = S["rho_p95"][-1] - S["rho_p05"][-1]
    g44 = S["edge"][-1]
    res = {
        "G43 rho within 10% of seeded, bm_secrete ON, real surface": {
            "value": g43, "threshold": 0.10, "pass": bool(g43 < 0.10),
            "rho_p05_p95_spread": spread},
        "G44 mean edge in [0.8, 1.7] x seeded, bm_refine ON, real surface": {
            "value": g44, "threshold": [0.8, 1.7], "pass": bool(0.8 <= g44 <= 1.7),
            "faces": S["n_face"][-1], "faces_seeded": S["n_face"][0]},
    }
    for k, v in res.items():
        print(f"[05l] {'PASS' if v['pass'] else 'FAIL'}  {k}: {v['value']:.4f}", flush=True)

    for folder, key, letter in (("05l_G43_secrete", "G43", "a"), ("05l_G44_refine", "G44", "b")):
        d = os.path.join(B.LOG, folder)
        os.makedirs(d, exist_ok=True)
        v = [x for kk, x in res.items() if kk.startswith(key)][0]
        fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor="white")
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("frame")
        if key == "G43":
            ax.plot(S["t"], S["rho"], color="black", lw=1.6)
            ax.fill_between(S["t"], S["rho_p05"], S["rho_p95"], color="black", alpha=0.12)
            ax.axhspan(0.9, 1.1, color="green", alpha=0.12)
            ax.set_ylabel(r"$\rho / \rho_0$ (mean, p05--p95)")
        else:
            ax.plot(S["t"], S["edge"], color="black", lw=1.6)
            ax.axhspan(0.8, 1.7, color="green", alpha=0.12)
            ax.set_ylabel("mean edge / seeded")
        ax.text(-0.17, 1.05, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.text(0.98, 1.05, ("PASS " if v["pass"] else "FAIL ") + f"{v['value']:.3f}",
                transform=ax.transAxes, fontsize=10, ha="right",
                color=("green" if v["pass"] else "red"))
        fig.tight_layout()
        fig.savefig(os.path.join(d, "gate.png"), dpi=150, facecolor="white")
        plt.close(fig)
        json.dump({"gate": key, **v, "series": S}, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        print(f"[05l] {folder} written", flush=True)


if __name__ == "__main__":
    main()
