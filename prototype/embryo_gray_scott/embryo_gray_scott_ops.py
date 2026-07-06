"""embryo3_ops -- Plexus operators for **Turing / Gray-Scott reaction-diffusion morphogenesis**.

A strict-Plexus reproduction of the reaction-diffusion morphogenesis model -- **A. M. Turing,
"The Chemical Basis of Morphogenesis" (Phil. Trans. R. Soc. B, 1952)**, in the two-species
**Gray-Scott** form popularised by **J. E. Pearson, "Complex Patterns in a Simple System"
(Science, 1993)**. Reference implementation vendored at `papers/reaction-diffusion/`
(B. Maier, MIT). Two morphogens A (substrate) and B (autocatalyst) diffuse and react:

    dA/dt = D_A ∇²A − A·B²  + f·(1 − A)          (A is fed in at rate f, consumed by the reaction)
    dB/dt = D_B ∇²B + A·B²  − (f + k)·B           (B is produced by the reaction, removed at f+k)

The single reaction A + 2B → 3B (autocatalysis) plus differential diffusion (D_A > D_B) is a
Turing instability: a homogeneous state destabilises into a stationary/【dynamic】 PATTERN whose
CLASS -- spots, stripes, mazes, self-replicating "mitosis" spots, moving solitons -- is set
entirely by the two rates (f, k) (the Pearson map). This is morphogenesis from pure local
chemistry: no cells, no mechanics, no learned rule -- the developmental pattern self-organises.

In Plexus the two morphogens are a 2-channel `grid` **field** and the PDE is a registered
**field operator** stepped by the engine (fields persist across frames):

`gray_scott` -- one reaction-diffusion tick (periodic Laplacian + reaction + feed/kill),
                with `substeps` inner Euler steps per frame; `kind=field`.
`rd_seed`    -- frame-0 initial condition (`before_frame: 1`): A≈1, B≈0 with a small central
                square of the reaction seeded (A=0.5, B=0.25) + a little noise.
"""
from __future__ import annotations

import torch

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator


def _laplacian(x):
    """5-point discrete Laplacian with periodic (wrap) boundaries -- matches the reference
    `apply_laplacian` (np.roll). x: [nx, ny]."""
    return (-4.0 * x
            + torch.roll(x, 1, 0) + torch.roll(x, -1, 0)
            + torch.roll(x, 1, 1) + torch.roll(x, -1, 1))


@register_operator("gray_scott", level="field", kind="field")
class GrayScott(FieldUpdate):
    """One Gray-Scott reaction-diffusion step on a 2-channel `grid` field (ch0=A, ch1=B).
    `substeps` inner Euler steps let many PDE steps happen per recorded frame."""
    SUPPORTED_DIMS = [2]
    MECHANISM_TAGS = ["reaction_diffusion", "turing_instability", "autocatalysis",
                      "morphogenesis", "pattern_formation"]
    PARAM_ROLES = {"DA": "substrate_diffusion", "DB": "autocatalyst_diffusion",
                   "f": "feed_rate", "k": "kill_rate"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.DA = float(params.get("DA", 0.16))
        self.DB = float(params.get("DB", 0.08))
        self.f = float(params.get("f", 0.060))
        self.k = float(params.get("k", 0.062))
        self.dt = float(params.get("dt", 1.0))
        self.substeps = int(params.get("substeps", 1))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                            # [2, nx, ny]
        A, B = g[0].clone(), g[1].clone()
        DA, DB, f, k, dt = self.DA, self.DB, self.f, self.k, self.dt
        for _ in range(self.substeps):
            reaction = A * B * B
            A = A + (DA * _laplacian(A) - reaction + f * (1.0 - A)) * dt
            B = B + (DB * _laplacian(B) + reaction - (f + k) * B) * dt
        g[0].copy_(A); g[1].copy_(B)
        return {}


@register_operator("rd_seed", level="field", kind="field")
class RDSeed(FieldUpdate):
    """Frame-0 initial condition: substrate A≈1, autocatalyst B≈0, with a small central
    square where the reaction is seeded (A=0.5, B=0.25) + a little uniform noise (the
    symmetry-breaking perturbation). Gate with `before_frame: 1`."""
    SUPPORTED_DIMS = [2]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.influence = float(params.get("influence", 0.05))    # noise amplitude
        self.seed_frac = float(params.get("seed_frac", 0.08))    # half-width of central seed (fraction of grid)

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                            # [2, nx, ny]
        nx, ny = g.shape[1], g.shape[2]
        dev = g.device
        rng = getattr(H, "rng", None)
        infl = self.influence
        A = (1.0 - infl) + infl * torch.rand(nx, ny, generator=rng, device=dev)
        B = infl * torch.rand(nx, ny, generator=rng, device=dev)
        r = max(1, int(self.seed_frac * nx))
        cx, cy = nx // 2, ny // 2
        A[cx - r:cx + r, cy - r:cy + r] = 0.50
        B[cx - r:cx + r, cy - r:cy + r] = 0.25
        g[0].copy_(A); g[1].copy_(B)
        return {}
