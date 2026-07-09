"""operators_smg -- the mechanism-tree operators the stock palette is missing.

Built so the random bootstrap can REACH every branch (else it only makes clusters).
Grounded in the vendored repos (see search/README.md §7). Import this module BEFORE S.load so
the operators register.

  ecm_boundary   Chaste immersed-boundary + LinearDifferentialAdhesionForce (Wang-Yamada):
                 a deformable basement membrane R(theta) with bending stiffness that cells
                 PUSH OUT and ADHERE TO (cell-matrix adhesion) -> buds distend it, clefts form
                 between them. The confine that RE-ROUNDS is replaced by a boundary that DEFORMS.
  (growth_gate / stiffness_field / slow_field: TODO, same file.)
"""
from __future__ import annotations
import math
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("ecm_boundary", level="cell", kind="lateral")
class EcmBoundary(Lateral):
    """Deformable basement membrane as virtual nodes (immersed-boundary, radial form).

    State (lazily allocated on the agent level): `ecm_R[M]` = membrane radius per angular bin,
    `ecm_c[2]` = membrane centre (tracks the tissue). Each step:
      * cells beyond R feel an INWARD containment force; cells just inside (within `gap`) feel an
        OUTWARD cell-matrix ADHESION force (`adhesion`) -> they cling to the membrane;
      * the membrane is pushed OUT where cells pile up (`push`) and SMOOTHED by bending/tension
        (`stiffness`, a ring Laplacian) -> a stiff membrane clefts between buds, a soft one lobes.
    2D (radial, star-shaped wrt the tissue centroid); clefts = local minima of R(theta).
    """
    PREDICTION = "velocity"
    SUPPORTED_DIMS = [2]
    PARAM_ROLES = {"stiffness": "membrane_bending", "adhesion": "cell_matrix_adhesion",
                   "push": "cell_membrane_coupling", "gap": "adhesion_band"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.M = int(params.get("nodes", 64))
        self.stiffness = float(params.get("stiffness", 40.0))
        self.adhesion = float(params.get("adhesion", 0.5))
        self.push = float(params.get("push", 30.0))
        self.gap = float(params.get("gap", 0.02))
        self.at = params.get("_at", "cell")

    def _ensure(self, lvl, pos, occ):
        if not hasattr(lvl, "ecm_R"):
            live = occ > 0
            c = pos[live].mean(0) if live.any() else pos.mean(0)
            r = (pos[live] - c).norm(dim=-1) if live.any() else torch.tensor([0.2], device=pos.device)
            R0 = float(r.mean() + r.std())
            lvl.register_buffer("ecm_R", torch.full((self.M,), max(R0, 0.05), device=pos.device))
            lvl.register_buffer("ecm_c", c.clone())

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos, occ = lvl.get("pos"), lvl.occ
        if pos.shape[-1] != 2:
            return {}
        self._ensure(lvl, pos, occ)
        dev = pos.device
        live = occ > 0
        if live.any():
            lvl.ecm_c = pos[live].mean(0).detach()               # membrane tracks the tissue
        c, R, M = lvl.ecm_c, lvl.ecm_R, self.M
        rel = pos - c
        r = rel.norm(dim=-1).clamp(min=1e-6)
        outward = rel / r[:, None]
        ang = torch.atan2(rel[:, 1], rel[:, 0])
        mbin = ((ang + math.pi) / (2 * math.pi) * M).long().clamp(0, M - 1)
        Rm = R[mbin]
        beyond = (r - Rm).clamp(min=0.0)                          # outside the membrane
        band = ((r < Rm) & ((Rm - r) < self.gap)).float()        # just inside -> adhere
        # agent velocity: inward containment (deformable) + outward cell-matrix adhesion
        vel = (-self.push * beyond[:, None] + self.adhesion * band[:, None]) * outward
        # membrane update: cell pressure pushes R out; bending/tension (ring Laplacian) smooths
        pressure = torch.zeros(M, device=dev).index_add_(0, mbin, beyond * live.float())
        cnt = torch.zeros(M, device=dev).index_add_(0, mbin, live.float())
        pressure = pressure / cnt.clamp(min=1.0)
        lap = 0.5 * (torch.roll(R, 1) + torch.roll(R, -1)) - R    # bending/tension
        dt = float(getattr(H, "dt", 0.002))
        lvl.ecm_R = (R + dt * (self.push * pressure + self.stiffness * lap)).clamp(min=0.03).detach()
        mlive = (mask.float() if mask is not None else torch.ones_like(r)) * live.float()
        return {self.at: vel * mlive[:, None]}


# ------------------------------------------------------------------ unit test (register + run)
def _unit_test():
    import os, sys, copy, tempfile
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
    sys.path.insert(0, "/workspace/Plexus/src")
    sys.path.insert(0, os.path.join(HERE, "..", "..", "active_matter2"))
    sys.path.insert(0, HERE)
    import yaml
    import plexus.operators  # noqa
    import am2_ops           # noqa
    import plexus.schema as S
    from plexus.engine import run
    import mechanism_tree as mt

    spec = copy.deepcopy(mt.BASE)
    spec["general"].pop("init", None)                    # disc substrate (real-init not needed to test op)
    spec["sets"]["agent"]["spawn"] = "disc"
    spec["general"]["n_frames"] = 60
    spec["operators"].append({"op": "ecm_boundary", "at": "agent", "stiffness": 40.0,
                              "adhesion": 0.5, "push": 30.0, "gap": 0.02})
    spec["schedule"].insert(5, "ecm_boundary")
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f); f.close()
    sim = S.load(f.name)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    box = {}

    def hook(H, frame):
        lvl = H.level("agent")
        if hasattr(lvl, "ecm_R"):
            box["R"] = lvl.ecm_R.detach().cpu().numpy().copy()

    run(sim, out_path=None, device=dev, on_frame=hook)
    assert "R" in box, "ecm_R was never allocated -> operator did not run"
    R = box["R"]
    print(f"[ecm_boundary] OK: ran 60 frames on {dev}; membrane R[{len(R)}] "
          f"min/mean/max = {R.min():.3f}/{R.mean():.3f}/{R.max():.3f}  "
          f"(deformed: std={R.std():.4f})")


if __name__ == "__main__":
    _unit_test()
