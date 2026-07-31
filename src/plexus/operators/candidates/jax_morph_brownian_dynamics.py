r"""agitate -- cell (lateral). A temperature-controlled thermal (Brownian) random walk
of cell positions: the constitutive core of one Euler-Maruyama step of overdamped
Langevin dynamics, with everything the promoted language already owns stripped away.

BIOLOGY. Cells in a tissue jiggle. In the differential-adhesion picture the source names
(SI, p. 19, "high-temperature Brownian relaxation") this jiggle is what lets an
adhesion-sorted arrangement anneal past kinetic traps into its energetic minimum -- the
thermal bath that turns a frozen jumble into a demixed pattern. This operator IS that bath:
a zero-drift, isotropic, temperature-parameterized Gaussian kick per alive cell per step.

WHAT THIS OPERATOR OWNS, AND WHAT IT DOES NOT. The reference `BrownianDynamics` step is a
composite; the Plexus decomposition keeps only the piece the frozen language lacks:

    dx = -(grad U / gamma) * dt   +   sqrt(2 kT dt / gamma) * xi
         \___ deterministic drift ___/   \____ THIS operator: the thermal bath ____/

  * the DRIFT F/gamma = -(grad U)/gamma is a SEPARATE pluggable pair-potential contract
    (soft_sphere / hertzian / the Morse family / attraction_repulsion), each already a
    registered `velocity`-emitting operator integrated by the engine's overdamped mobility.
    `agitate` does NOT read a potential and adds no drift; schedule the drift operator
    alongside it and the two velocities sum. With no drift operator present, `agitate` alone
    is the free Brownian gas (the reference's `potential=None`).
  * the Euler-Maruyama scheme, dt-scaling, dynamic-delta accumulation, and the whole
    StochasticStep trace/replay/logp machinery are ENGINE plumbing (out of scope here).

What is left as this step's own constitutive content is the temperature-driven random walk:
amplitude std = sqrt(2 kT dt / gamma), diffusion D = kT/gamma (the Einstein /
fluctuation-dissipation relation), Wiener sqrt(dt) scaling.

THE SPLIT dt-SCALING (the key surprise, faithfully preserved). The drift grows LINEARLY in
dt but the noise grows as sqrt(dt) (a Wiener increment) -- so the thermal DISPLACEMENT is
sqrt(2 kT dt / gamma) * xi. Plexus integrates a `velocity` delta as `pos += dt * v`, i.e. it
multiplies whatever we emit by dt. To land a displacement that scales as sqrt(dt) THROUGH an
engine that multiplies by dt, we must emit a velocity that scales as 1/sqrt(dt):

    v = sqrt(2 kT / (gamma dt)) * xi     ->     dt * v = sqrt(2 kT dt / gamma) * xi   (correct)

Emitting a dt-independent noise velocity (or scaling the noise by dt like the drift) gets the
diffusion constant WRONG (displacement variance would scale as dt^2, not dt). This 1/sqrt(dt)
velocity is exactly what makes the composite reproduce Euler-Maruyama: the engine sums the
drift operator's velocity F/gamma with this thermal velocity and integrates once,
`pos += dt*(F/gamma) + sqrt(2 kT dt/gamma)*xi`.

WHY A NEW CONTRACT, NOT AN ALIAS. Three registered operators already bolt an isotropic
`noise * randn` term onto a primary deterministic force -- drag (role `thermal_noise`), glide
(role `translational_noise`), attraction_repulsion (role `exploration_noise`). But those are
all the SAME uncalibrated modifier: a bare amplitude times a standard normal, off by default,
riding on another force. None carries a temperature; none obeys the Einstein relation
(amplitude uncoupled from friction); none applies the Wiener sqrt(dt) scaling (the engine
integrates them deterministically, so their diffusion constant is dt-scaling-wrong); and none
can stand ALONE as a bath (drag needs a velocity, glide a heading, attraction_repulsion
neighbours). `agitate` is the first-class thermostat those three keep re-inventing:
constitutive, temperature-parameterized, FDT-calibrated, Wiener-scaled, and self-standing.

SOURCE vs PAPER (rule 5, SOURCE WINS). The paper's Methods (Deshpande, Mottes et al. 2025,
p. 14) write mechanical relaxation as DETERMINISTIC "gradient descent energy minimization of
the Morse potential for a fixed number of steps" -- no kT, no gamma, no noise term. The SOURCE
implements a full overdamped Langevin step with explicit gamma, kT, and a sqrt(2 kT dt/gamma)
noise. The source's kT = 0 case reduces exactly to the paper's gradient descent (this operator
then emits zero and only the drift operator moves cells); kT > 0 is the noisy
high-temperature regime the SI names for adhesion sorting.

Translated from papers/jax-morph/jax_morph/physics/mechanics/dynamics.py:L36
(BrownianDynamics; noise scale std = sqrt(2 kT dt/gamma) at :L133, dx = mean + std*xi at
:L172). Torch, not JAX.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("agitate", family="motion", set="cell", kind="lateral")
class Agitate(Lateral):
    EMIT = "velocity"                            # a thermal velocity; the ENGINE integrates pos (x += dt*v)
    # typed signature (Plexus2 sec. 2.1): a single-body cell -> cell morphism. Reads pos (to
    # size/mask the kicks) and alive (occ, to mask dead/padded slots); writes the pos delta. No
    # gather map -- the thermal bath is uncoupled (unlike the drift potential it composes with).
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["pos", "alive"]
    WRITES = ["pos"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]                      # isotropic kick is dimension-generic (reads D = pos.shape[-1])
    REQUIRES_PARAMS = []                         # every knob optional (kT/gamma default to the source's 0.1 / 1.0)
    MECHANISM_TAGS = ["thermal_noise", "brownian_motion", "langevin_bath",
                      "fluctuation_dissipation", "wiener_process", "overdamped_diffusion"]
    PARAM_ROLES = {"kT": "thermal_energy", "gamma": "translational_drag",
                   "n_space_dim": "spatial_dimension_assert"}
    REFERENCE = (
        "Deshpande, Mottes, Vidal Saez, Kicheva & Hiscock (2025), 'Engineering morphogenesis "
        "of cell clusters with differentiable programming', Nat Comput Sci (deterministic Morse "
        "gradient-descent relaxation, Methods p. 14; 'high-temperature Brownian relaxation', SI "
        "p. 19). Translated from papers/jax-morph/jax_morph/physics/mechanics/dynamics.py:L36 "
        "(BrownianDynamics: std = sqrt(2 kT dt/gamma) at :L133, dx = mean + std*xi at :L172); "
        "the paper writes no Langevin equation, kT=0 recovers its gradient descent -- SOURCE WINS. "
        "Physics: Einstein, A. (1905). Ann. Phys. 322:549-560; Langevin, P. (1908). C. R. Acad. "
        "Sci. 146:530-533 (D = kT/gamma, fluctuation-dissipation)."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")                      # the set this acts on (engine-injected)
        self.kT = float(params.get("kT", 0.1))                   # thermal energy (temperature); 0 -> no bath
        self.gamma = float(params.get("gamma", 1.0))             # translational drag; mobility 1/gamma, and under the sqrt
        if self.kT < 0.0:
            raise ValueError(f"agitate: kT must be >= 0 (a temperature), got {self.kT}.")
        if self.gamma <= 0.0:
            raise ValueError(f"agitate: gamma must be > 0 (a drag coefficient), got {self.gamma}.")
        # optional static dim assert (the source RAISES on n_space_dim mismatch at trace time):
        # keep that surprise reproducible without forcing the caller to declare a dimension.
        nsd = params.get("n_space_dim", None)
        self.n_space_dim = None if nsd is None else int(nsd)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")                                     # [N, D]
        N, D = pos.shape[0], pos.shape[-1]
        dev = pos.device
        if self.n_space_dim is not None and self.n_space_dim != D:
            raise ValueError(
                f"agitate was built with n_space_dim={self.n_space_dim} but the set has spatial "
                f"dimension {D}; they must match (n_space_dim sizes the per-cell thermal kick).")

        # dead / masked slots draw a zero kick (the reference alive-masks the displacement).
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ

        dt = float(getattr(H.config, "dt", 1.0))
        # kT = 0 (deterministic gradient-descent relaxation limit) or a degenerate dt: no bath.
        if self.kT == 0.0 or dt <= 0.0:
            return {self.at: torch.zeros(N, D, device=dev)}

        # Wiener kick as a velocity. The DISPLACEMENT std is sqrt(2 kT dt / gamma) (scales as
        # sqrt(dt)); the engine multiplies our velocity by dt, so the velocity amplitude is
        # displacement_std / dt = sqrt(2 kT / (gamma dt)) (scales as 1/sqrt(dt)). This split
        # dt-scaling is Euler-Maruyama, NOT explicit Euler -- see the module docstring.
        vel_std = math.sqrt(2.0 * self.kT / (self.gamma * dt))
        xi = torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)   # standard normal
        vel = vel_std * xi
        return {self.at: vel * m[:, None]}
