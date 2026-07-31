r"""mechanosense -- cell (lateral). A per-cell MECHANICAL OBSERVABLE: the Irving-Kirkwood
virial pressure, reduced from the same conservative pair law that moves the cells, written into
a transient ``stress`` field for a downstream controller to sense. It MOVES NOTHING.

BIOLOGY. A cell in a crowded tissue feels a local mechanical load -- it is compressed by its
neighbours or stretched by them. Mechanotransduction is the cell reading that load and gating a
decision (here, proliferation) on it: in the paper's "Mechanical Control of Cell Proliferation"
(Fig. 4, p. 7-8) the trained gene circuit takes the local stress as an input and stress INHIBITS
division. This operator is the SENSOR that produces that input. For a receiver cell i, summed over
its live neighbours j, the per-cell Irving-Kirkwood virial pressure is

    p_i = -(1 / (2 d V_i)) * sum_{j != i, j alive}  r_ij * (dU/dr)(r_ij)

where r_ij = ||x_i - x_j|| is the pair separation, dU/dr the radial derivative of the SAME pair
energy the mechanics uses, d the spatial dimension, and V_i the cell's own d-ball volume from its
radius R_i (2 R_i in 1D, pi R_i^2 in 2D, 4/3 pi R_i^3 in 3D). Three conventions carry biology, not
cosmetics:
* the MINUS SIGN makes repulsion (dU/dr < 0, compression) read p_i > 0 and tension p_i < 0
  (compression-positive). Drop it and the mechanosensing signal inverts -- a trained "stress
  inhibits division" rule would become "stress promotes division";
* the ONE-HALF is the Irving-Kirkwood bond split -- each pair's virial is shared between its two
  cells; omit it and every cell's pressure double-counts by 2x;
* V_i is the cell's OWN d-ball volume (dimension-branched), NOT a Voronoi area nor the contact
  distance sigma, so the readout is size-consistent -- comparable across cells of different radius.
Cells beyond the potential's cutoff (dU/dr = 0 there) contribute 0.

ROUTING. ``kind=lateral, family=mechanics, set=cell`` -- a WITHIN-SET pairwise-neighbour reduction
to a per-cell scalar, the same taxonomy slot the frozen language gives ``gravity`` / ``mpm_anchor``
/ ``mpm_spin`` (body/continuum mechanics on a set). It is PURE SENSING, quasistatic: unlike every
delta-emitting operator it returns no motion delta (``EMIT=None``) -- it WRITES the transient
per-cell ``stress`` field in place and returns ``{}``. That write is a DERIVED-STATE READOUT (the
category the engine's frame-0 integration-invariant guard exempts via
``MAY_MUTATE_INTEGRATED_STATE``): ``stress`` is recomputed from the live configuration every
macro-step and is transient (a daughter is born at the default, not the mother's value), so it is
not integrated dynamics and not heritable. Schedule it upstream of the consumer (a gene network /
``cell_divide`` gate); no operator INSIDE this library reads ``stress`` -- the consumer is an
external control input, exactly as in the paper.

THE PAIR LAW IS A PLUG-IN (why this is a mechanoSENSOR, not a force). The source computes p_i in
``PairwisePotential.virial_pressure`` -- it reuses the SAME ``pair_energy`` the mechanics
(``adhere``) uses, and only the REDUCTION differs: the force is the rank-1 vector -grad_i U that
MOVES cells; the virial pressure is a rank-0 scalar (the radial force r_ij . dU/dr projected on the
bond and summed) that is SENSED. So ``mechanosense`` carries a ``potential:`` selector naming which
pair law supplies dU/dr (the paper's ``morse`` by default; also ``soft_sphere`` / ``hertzian`` /
``harmonic`` / ``lennard_jones``, the same library ``adhere`` draws from), with the law's own knobs
(``epsilon``/``alpha``/cutoff fractions) -- exactly the numbers the co-scheduled ``adhere`` operator
uses. dU/dr is taken by AUTODIFF of the FULL pair energy (INCLUDING the multiplicative smooth
cutoff), elementwise -- the source's ``jax.grad(pair_energy)``; differentiating only the bare
Morse/LJ core would drop the cutoff's switch-function term inside the transition window.
DIFFERENTIABLE readout: the written ``stress`` stays differentiable through the potential's
coupling (a traced per-cell ``epsilon``), so a morphology objective can optimize the interaction
through the sensed stress -- the source's "optimizable through the written stress".

DENSE, LIVE, MASKED, NaN-SAFE. The reduction runs over all live non-self pairs (dense N x N -- the
source's ``neighbor_sum`` seam): the self-diagonal (r = 0) and any dead-cell source j are dropped,
and a trailing ``* alive`` zeroes dead receivers i. ``_safe_norm`` keeps the r = 0 diagonal finite
in value AND gradient; ``_safe_divide`` gives a dead cell (radius 0 -> V_i = 0) a stress of 0 rather
than an inf. A naive reduction NaNs on the diagonal or on a padded dead slot.

WHY ``new`` (contract ``mechanosense``). No frozen contract exposes a per-cell MECHANICAL
OBSERVABLE as a pure-sensing quantity for downstream mechanotransduction. It is NOT a refinement of
``attraction_repulsion`` / ``adhere`` (the nearest kin, and literally the same pair law): those emit
a conservative FORCE that MOVES cells (EMIT=velocity, writes pos); fusing "the force that drives
motion" with "the load a cell feels" into one signature would force their OUTPUTS/WRITES/EMIT to
change and break every caller relying on them being pure motion laws -- and the derived quantity is
different (a rank-0 sensed scalar vs a rank-1 integrated vector). NOT an alias of ``sense`` (that
reads an ambient diffusible FIELD on a sensor fan and STEERS heading; this reduces over pairwise
neighbours and writes a scalar state field -- different inputs, output, map, kind). NOT a refinement
of ``cell_grow``'s ``stress_gain`` mechano-inhibition: that reads the MPM continuum DEFORMATION
GRADIENT F and FUSES sense+respond inside one growth op; ``mechanosense`` is the DECOUPLED sensor
producing a first-class, reusable ``stress`` field ANY consumer can read. Plexus expresses the
mechano-RESPONSE but has no promoted mechano-SENSE contract; that absence is the new vocabulary.

SOURCE vs PAPER (rule 5, SOURCE WINS). The CODE ships the standard Irving-Kirkwood virial pressure
above. The PAPER (p. 16, "Mechanical Stress") writes a DIFFERENT, 2D-only scalar
``sigma_i = sum_j [ F_ij,x * sgn(dx) + F_ij,y * sgn(dy) ]`` -- each force COMPONENT times the SIGN of
that displacement component (a taxicab weighting, NOT the Euclidean radial projection), with NO
volume/dimension normalization and NO 1/2 bond split; a reimplementer following the prose would get
an unnormalized, differently-weighted number off by roughly a per-cell factor of 2 d V_i. SOURCE
WINS: the shipped quantity is the virial pressure. (The two are two IMPLEMENTATIONS of the one
``mechanosense`` contract -- a per-cell scalar mechanical load reduced from the pair forces --
differing only in weighting/normalization, which is itself evidence the contract is genuine.)

Reference: jax-morph ``VirialStress``, papers/jax-morph/jax_morph/physics/mechanics/stress.py:L15-L58
(delegates to ``PairwisePotential.virial_pressure``, potentials.py:L242-L266; pair laws :L269-L459;
``_smooth_cutoff`` :L42, ``_compact_repulsion`` :L30; ``neighbor_sum``/``safe_*`` seams). Paper:
Deshpande, Mottes, Vidal Saez, Kicheva & Hiscock (2025), "Engineering morphogenesis of cell clusters
with differentiable programming", Nat. Comput. Sci. -- Methods "Mechanical Stress" (p. 16, the
sigma_i eq.), used in "Mechanical Control of Cell Proliferation" (p. 7-8, Fig. 4, Fig. S6); the
paper's sigma_i DIFFERS from the shipped virial pressure (SOURCE WINS). Physics: Irving, J. H. &
Kirkwood, J. G. (1950), J. Chem. Phys. 18:817-829 (the statistical-mechanical pressure tensor).
Torch, not JAX.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a / b`` where ``b != 0``, else 0 -- with a finite gradient at ``b == 0`` (the double-``where``
    trick: the division only ever sees a nonzero denominator). Mirrors jax-morph's ``safe_divide``,
    which guards a dead-dead padded pair (sigma = 0) and a dead cell's zero volume (V_i = 0) that
    would otherwise NaN / inf a masked entry in the backward pass."""
    safe_b = torch.where(b != 0.0, b, torch.ones_like(b))
    return torch.where(b != 0.0, a / safe_b, torch.zeros_like(a))


def _safe_norm(d2: torch.Tensor) -> torch.Tensor:
    """``sqrt(d2)`` where ``d2 > 0``, else 0 -- finite gradient at ``d2 == 0`` (the self-diagonal). The
    torch analogue of jax-morph's ``safe_norm``: ``sqrt`` at 0 has an infinite derivative, so the
    zero-separation diagonal must be routed around it."""
    safe = torch.where(d2 > 0.0, d2, torch.ones_like(d2))
    return torch.where(d2 > 0.0, torch.sqrt(safe), torch.zeros_like(d2))


def _smooth_cutoff(r: torch.Tensor, r_on: torch.Tensor, r_off: torch.Tensor) -> torch.Tensor:
    """Multiplicative isotropic cutoff (jax-md form): a C1 switch from 1 (r <= r_on) to 0 (r >= r_off).

    ``S(r) = (r_off^2 - r^2)^2 (r_off^2 + 2 r^2 - 3 r_on^2) / (r_off^2 - r_on^2)^3`` on the transition
    window. A verbatim port of jax-morph's ``_smooth_cutoff`` (potentials.py:L42); ``_safe_divide``
    guards the denominator for a dead-dead padded pair (sigma = 0 -> r_on = r_off = 0), which would
    otherwise be 0/0. The cutoff is part of the pair energy, so its derivative rides in dU/dr."""
    r2, ron2, roff2 = r * r, r_on * r_on, r_off * r_off
    s = _safe_divide((roff2 - r2) ** 2 * (roff2 + 2.0 * r2 - 3.0 * ron2), (roff2 - ron2) ** 3)
    return torch.where(r < r_on, torch.ones_like(s), torch.where(r < r_off, s, torch.zeros_like(s)))


def _compact_repulsion(r, sigma, eps, exponent: float, prefactor: float):
    """``prefactor * eps * (1 - r/sigma)**exponent`` for ``r < sigma``, else 0 (value AND grad safe).

    Verbatim port of jax-morph's ``_compact_repulsion`` (potentials.py:L30). ``_safe_divide`` handles a
    dead-dead padded pair (sigma = 0); the double ``where`` evaluates the (possibly fractional) power
    only on a strictly positive base, so the gradient is finite for ``r >= sigma`` too."""
    base = 1.0 - _safe_divide(r, sigma)
    safe = torch.where(base > 0.0, base, torch.ones_like(base))       # the power only sees a positive base
    return torch.where(base > 0.0, prefactor * eps * safe ** exponent, torch.zeros_like(base))


@register_operator("mechanosense", family="mechanics", set="cell", kind="lateral")
class Mechanosense(Lateral):
    r"""Per-cell Irving-Kirkwood virial pressure written into a transient ``stress`` field -- a pure
    (quasistatic) mechanosensor that reduces the SAME conservative pair law the mechanics uses to a
    per-cell scalar and MOVES NOTHING. Translated from jax-morph ``VirialStress`` /
    ``PairwisePotential.virial_pressure`` (stress.py:L15, potentials.py:L242)."""

    EMIT = None                                  # pure sensing: writes the `stress` field in place, returns {} — no integrable delta
    # A DERIVED-STATE READOUT: it legitimately writes the (non-integrated, transient) `stress`
    # block, so it opts out of the engine's frame-0 integration-invariant guard (same exemption as
    # a structural op / an aggregate centroid). It never touches pos/vel.
    MAY_MUTATE_INTEGRATED_STATE = True
    # typed signature (Plexus2 sec. 2.1): a within-set pairwise cell -> cell readout. Reads pos +
    # the physical radius (for V_i and the contact-distance sigma of the pair law) + alive (occ);
    # writes the per-cell `stress` scalar. No gather MAP: the reduction is a dense N x N over the
    # set itself, not a traversal of a named map (like `adhere`).
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["pos", "radius", "alive"]           # + the potential's coupling (epsilon), when field-driven
    WRITES = ["stress"]                          # the per-cell virial pressure (a transient sensing scalar)
    MAPS = []
    SUPPORTED_DIMS = [2, 3]                        # dimension-generic: reads d = pos.shape[-1]; V_i branches on d (also d=1)
    DIFFERENTIABLE = True                         # dU/dr by autodiff; `stress` stays differentiable through the coupling epsilon
    REQUIRES_PARAMS = []                          # every knob optional -- defaults to the paper's Morse mechanics
    MECHANISM_TAGS = ["mechanotransduction", "virial_pressure", "mechanical_stress",
                      "pairwise_reduction", "compression_positive", "quasistatic_sensing"]
    PARAM_ROLES = {"potential": "pair_law_supplying_dU_dr", "epsilon": "interaction_coupling",
                   "epsilon_field": "per_cell_coupling_field", "alpha": "morse_well_steepness",
                   "r_onset_frac": "cutoff_onset_fraction", "r_cutoff_frac": "cutoff_end_fraction",
                   "radius": "fallback_cell_radius", "stress": "output_stress_field"}
    REFERENCE = (
        "jax-morph VirialStress, physics/mechanics/stress.py:L15-L58 (delegates to "
        "PairwisePotential.virial_pressure, potentials.py:L242-L266; pair laws :L269-L459; "
        "_smooth_cutoff :L42, _compact_repulsion :L30); "
        "Irving, J. H. & Kirkwood, J. G. (1950) J. Chem. Phys. 18:817-829; "
        "Deshpande, Mottes et al. (2025) Nat. Comput. Sci., Methods 'Mechanical Stress' (p. 16) -- "
        "the paper's taxicab sigma_i DIFFERS from the shipped virial pressure (SOURCE WINS, rule 5)."
    )

    # the pair laws this sensor can reduce, and each law's non-coupling defaults (matching
    # potentials.py). The coupling (epsilon / k) default is law-dependent: Morse's well depth is 3.0.
    _LAWS = ("morse", "soft_sphere", "hertzian", "harmonic", "lennard_jones")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.potential = str(params.get("potential", "morse"))       # which pair law supplies dU/dr (paper: morse)
        if self.potential not in self._LAWS:
            raise ValueError(
                f"mechanosense: unknown potential {self.potential!r}; choose one of {self._LAWS}.")
        default_eps = 3.0 if self.potential == "morse" else 1.0       # Morse well depth default is 3.0 (potentials.py)
        self.epsilon = float(params.get("epsilon", default_eps))     # shared coupling (well depth / stiffness k)
        self.eps_field = params.get("epsilon_field", None)           # optional per-cell coupling field (block or buffer)
        self.alpha = float(params.get("alpha", 2.8))                 # Morse well steepness (unused by the other laws)
        self.r_onset_frac = float(params.get("r_onset_frac", 1.5))   # smooth-cutoff onset (morse / lennard_jones)
        self.r_cutoff_frac = float(params.get("r_cutoff_frac", 2.5)) # cutoff end (morse / lennard_jones / harmonic)
        self.radius0 = float(params.get("radius", 0.5))              # uniform fallback radius if the set carries no `radius`
        self.radius_block = str(params.get("radius_block", "radius"))  # per-cell size block/buffer (V_i and sigma)
        self.stress_block = str(params.get("stress", "stress"))      # the per-cell scalar this readout writes
        if self.potential in ("morse", "lennard_jones") and not self.r_onset_frac < self.r_cutoff_frac:
            raise ValueError(
                f"mechanosense:{self.potential} needs r_onset_frac < r_cutoff_frac for a smooth "
                f"cutoff window, got r_onset_frac={self.r_onset_frac}, r_cutoff_frac={self.r_cutoff_frac}.")
        if self.potential == "harmonic" and not self.r_cutoff_frac > 1.0:
            raise ValueError(
                f"mechanosense:harmonic needs r_cutoff_frac > 1 (the cutoff must sit beyond contact), "
                f"got r_cutoff_frac={self.r_cutoff_frac}.")

    # --- state I/O (the grow_radius recipe: a state block if the schema declares one, else a buffer) ---
    def _read_scalar(self, lvl, name):
        """A per-cell [N] view of a named quantity: a state block if the schema declares one, else a
        registered per-node buffer of that name, else None."""
        if name is None:
            return None
        if name in lvl.state_schema:
            v = lvl.get(name)
        else:
            v = getattr(lvl, name, None)
        if v is None or not torch.is_tensor(v):
            return None
        return v.reshape(v.shape[0], -1)[:, 0]                        # [N] (first component if width > 1)

    def _write_scalar(self, lvl, name, vals, dev, mask=None):
        """Write a per-cell scalar into the `stress` field IN PLACE: a state block if the schema
        declares one, else a lazily-provisioned per-node buffer. `mask` (from an `at:` subset
        selector) restricts the WRITE to the selected cells, preserving the others' prior stress --
        a subset selector senses a subset, it does not erase the rest."""
        vals = vals.reshape(-1)
        if mask is not None:
            m = mask.to(vals.dtype)
            cur = self._read_scalar(lvl, name)
            cur = torch.zeros_like(vals) if cur is None else cur.to(vals.dtype)
            vals = m * vals + (1.0 - m) * cur
        if name in lvl.state_schema:
            a, b = lvl.state_schema[name]
            lvl.state[:, a:b] = vals[:, None]                        # mutates `state` (guard-exempt: derived readout)
        else:
            buf = getattr(lvl, name, None)
            if buf is None or not torch.is_tensor(buf) or buf.shape[0] != lvl.n:
                lvl.register_buffer(name, torch.zeros(lvl.n, device=dev, dtype=vals.dtype))
                buf = getattr(lvl, name)
            buf[:] = vals

    def _read_coupling(self, lvl, n, dev, dtype):
        """The per-pair coupling (epsilon / k): a per-cell field mixed by the arithmetic mean
        0.5*(c_i + c_j) into an [N, N] matrix, else the shared scalar (broadcast). The additive
        contact distance keeps its OWN rule (sigma = r_i + r_j) -- do not mix sigma with the mean."""
        if self.eps_field is None:
            return self.epsilon                                      # scalar broadcasts over the [N, N] energy
        v = self._read_scalar(lvl, self.eps_field)
        if v is None:
            return self.epsilon
        v = v.to(dtype)
        return 0.5 * (v[:, None] + v[None, :])                       # arithmetic-mean mix (finite grad at zeros)

    def _pair_energy(self, r, sigma, c):
        """Elementwise pair energy U(r) for the selected law (the SAME shapes as potentials.py); dU/dr
        is taken by autodiff of this, so the multiplicative smooth cutoff's derivative is included."""
        p = self.potential
        if p == "soft_sphere":                                       # 0.5 eps (1 - r/sigma)^2, compact at contact
            return _compact_repulsion(r, sigma, c, 2.0, 0.5)
        if p == "hertzian":                                          # 0.4 eps (1 - r/sigma)^(5/2), compact
            return _compact_repulsion(r, sigma, c, 2.5, 0.4)
        if p == "harmonic":                                          # shifted spring, hard-truncated at r_c (C0)
            r_c = self.r_cutoff_frac * sigma
            u = 0.5 * c * ((r - sigma) ** 2 - (r_c - sigma) ** 2)
            return torch.where(r < r_c, u, torch.zeros_like(u))
        if p == "lennard_jones":                                     # r_min form: min -eps at contact, smooth cutoff
            x = _safe_divide(sigma, r)
            u = c * (x ** 12 - 2.0 * x ** 6)
            return u * _smooth_cutoff(r, self.r_onset_frac * sigma, self.r_cutoff_frac * sigma)
        # morse (the paper's mechanics): eps[(1 - exp(-alpha(r - sigma)))^2 - 1] * smooth cutoff
        e = 1.0 - torch.exp(-self.alpha * (r - sigma))
        u = c * (e * e - 1.0)
        return u * _smooth_cutoff(r, self.r_onset_frac * sigma, self.r_cutoff_frac * sigma)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        n = lvl.n
        pos = lvl.get("pos")                                         # [N, D]
        D = pos.shape[-1]

        periodic = getattr(H, "periodic", False)
        world = getattr(H, "world_size", getattr(H, "world_width", 1.0))

        # per-cell radius: V_i (the d-ball volume) and the pair law's contact distance sigma = r_i + r_j.
        radius = self._read_scalar(lvl, self.radius_block)
        if radius is None:
            radius = torch.full((n,), self.radius0, device=dev, dtype=pos.dtype)
        radius = radius.reshape(n).to(dev).to(pos.dtype)
        sigma = radius[:, None] + radius[None, :]                    # [N, N] additive contact rule

        c_ij = self._read_coupling(lvl, n, dev, pos.dtype)          # scalar or [N, N]

        # the neighbor_sum seam: live non-self pairs only (drop the r=0 diagonal AND dead sources j).
        alive = (lvl.occ > 0)
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        pair_mask = alive[:, None] & alive[None, :] & ~eye           # [N, N]

        disp = minimum_image(pos[:, None, :] - pos[None, :, :], periodic, world)   # [N, N, D]
        d2 = (disp * disp).sum(-1)                                   # [N, N]
        r = _safe_norm(d2)                                           # [N, N], 0 on the diagonal (masked out)

        # dU/dr, ELEMENTWISE, by autodiff of the full pair energy (the source's jax.grad(pair_energy)).
        # Differentiate against a leaf clone of r; under a grad-enabled (differentiable) rollout
        # create_graph keeps `stress` connected to the coupling epsilon (the source's optimizable path).
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            rr = r.detach().requires_grad_(True)
            u = self._pair_energy(rr, sigma, c_ij)                   # [N, N]
            (du_dr,) = torch.autograd.grad(u.sum(), rr, create_graph=outer_grad)   # d(sum U)/dr_ij = dU_ij/dr_ij

        # virial_i = sum_{j live, j != i} r_ij (dU/dr)(r_ij); then Irving-Kirkwood normalise.
        virial = torch.where(pair_mask, r * du_dr, torch.zeros_like(r)).sum(dim=1)  # [N]
        if D == 1:
            vol = 2.0 * radius                                       # a 1-ball is an interval of length 2r
        elif D == 2:
            vol = math.pi * radius ** 2                              # disk area
        else:                                                       # D == 3
            vol = (4.0 / 3.0) * math.pi * radius ** 3               # sphere volume
        # minus sign: repulsion (dU/dr < 0) -> p_i > 0 (compression positive). safe_divide: V_i=0 -> 0.
        p = _safe_divide(-virial, 2.0 * D * vol) * alive.to(r.dtype)  # [N]

        self._write_scalar(lvl, self.stress_block, p, dev, mask=mask)
        return {}                                                    # pure sensing: no set delta
