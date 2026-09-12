"""Mass-action metabolism as three operators over a metabolite set, a reaction set and the
stoichiometric relation between them -- the forward model of `MetabolismGraph` (PDE_M1 with
`use_mass_action`) written in the language.

THE HIERARCHY IS THE BIPARTITE GRAPH, and that is the point of running this model here at all:

    cell -> metabolite      one row per species, `conc` its concentration (integrated, first order)
    cell -> reaction        one row per reaction, `v` its rate (a readout, written each tick)
    stoich                  an EDGE-SET: one row per non-zero stoichiometric coefficient S_ij,
                            `pre: metabolite`, `post: reaction`, `w` = S_ij (negative: substrate,
                            positive: product)

The stoichiometric matrix is not a dense tensor inside an operator; it is the `w` state of the
relation, exactly as the connectivity matrix is the `w` of the synapse set (`operators/neural.py`),
and the same two incidence maps carry everything:

    reaction_rate      log v_j = log k_j + SUM_{i : S_ij < 0} |S_ij| log c_i     (Aggregate along `post`)
    metabolite_flux    dc_i/dt = SUM_j S_ij v_j                                    (Aggregate along `pre`)
    metabolite_homeostasis   dc_i/dt += -lambda (c_i - c0_i)                       (Lateral, no map)

Reference: Allier, C. et al., MetabolismGraph (`generators/PDE_M1.py`): v = k prod c^|s| in log
space, dx/dt = S v, minus lambda (c - c_baseline).

WHAT IS LEFT OUT, SAID ONCE. PDE_M1's flux limiter (scale v so no substrate is over-consumed in a
step) is a fix for explicit Euler at a coarse step; here `conc` is clamped at a floor inside the
rate, which keeps the log finite, and a spec that wants stiffness resolved takes a smaller dt.
The MLP kinetics (the inverse model's own substrate function) are not a forward model.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Aggregate, Lateral, Seed
from plexus.models.registry import register_operator


@register_operator("metabolite_seed", family="seed", set="metabolite", kind="seed")
class MetaboliteSeed(Seed):
    """x_0 for the metabolites: every concentration drawn uniformly in [c_min, c_max], once.

    metabolite -> metabolite: writes `conc`, and remembers the draw as `c0` (the homeostatic
    baseline, `baseline_mode: initial` in the reference) when the set carries that block."""

    EMIT = None
    INPUTS = ["metabolite"]
    OUTPUTS = ["metabolite"]
    READS = []
    WRITES = ["conc", "c0"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["initial_condition", "metabolism"]
    PARAM_ROLES = {"c_min": "initial_concentration_low", "c_max": "initial_concentration_high",
                   "seed": "random_seed"}
    REFERENCE = "MetabolismGraph generators/utils.py::init_concentration (mode uniform)"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "metabolite")
        self.c_min = float(params.get("c_min", 2.5))
        self.c_max = float(params.get("c_max", 7.5))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        c = self.c_min + (self.c_max - self.c_min) * torch.rand(lvl.n, 1, generator=g)
        c = c.to(lvl.state.device, lvl.state.dtype)
        st = lvl.state.clone()
        a, b = lvl.state_schema["conc"]
        st[:, a:b] = c
        if "c0" in lvl.state_schema:
            a0, b0 = lvl.state_schema["c0"]
            st[:, a0:b0] = c
        lvl.state = st
        return {}


@register_operator("reaction_rate", family="metabolism", set="reaction", kind="aggregate")
class ReactionRate(Aggregate):
    """The mass-action rate of every reaction, from its substrates' concentrations.

    (metabolite, stoich) -> reaction: lifts `conc` onto the substrate edges along `pre`, sums
    |S_ij| log c_i onto the reaction along `post`, writes `v` in place (a readout, not integrated).

        v_j = k_j  prod_{i : S_ij < 0}  c_i ^ |S_ij|        computed as  exp( log k_j + SUM |S_ij| log c_i )

    k_j comes from the reaction's own `k` block when the set declares one (a per-reaction rate
    constant, the inverse model's target), else from the operator's `k`. `c_floor` is the
    concentration below which log c is held, so a species driven to zero stops a reaction rather
    than producing -inf. `rate_noise` fluctuates each k_j by that relative sd per tick
    (enzyme-activity noise; S still conserves mass every step).
    """

    EMIT = None
    INPUTS = ["metabolite", "stoich"]
    OUTPUTS = ["reaction"]
    READS = ["conc", "w", "k"]
    WRITES = ["v"]
    MAPS = ["pre", "post"]
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True   # `v` is a derived readout written in place, as `omega` is
    REQUIRES_PARAMS = ["edge_set"]
    MECHANISM_TAGS = ["mass_action", "metabolism", "kinetics"]
    PARAM_ROLES = {"edge_set": "stoichiometry_as_edge_set", "k": "rate_constant_default",
                   "c_floor": "concentration_floor_in_log", "rate_noise": "relative_rate_noise_sd",
                   "flux_limit_dt": "step_for_the_flux_limiter"}
    REFERENCE = "MetabolismGraph generators/PDE_M1.py::forward (use_mass_action)"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "reaction")
        self.edge_set = params["edge_set"]
        self.k = float(params.get("k", 1.0))
        self.c_floor = float(params.get("c_floor", 1e-8))
        self.rate_noise = float(params.get("rate_noise", 0.0))
        # THE REFERENCE'S FLUX LIMITER, opt-in by naming the step. Explicit Euler over-consumes a
        # substrate whose reactions would take more than it holds in one step and drives it
        # negative; scaling every reaction so that no substrate is consumed past its concentration
        # in `flux_limit_dt` keeps c >= 0 at that step. The operator does not know the engine's
        # dt, so the spec states it (the form writes general.dt here).
        self.flux_limit_dt = float(params.get("flux_limit_dt", 0.0))

    def forward(self, H, mask=None):
        rxn = H.level(self.at)
        es = H.level(self.edge_set)
        c_e = H.gather(self.edge_set, "pre", "conc")                # [E, 1] the metabolite's conc on its edge
        s_e = es.get("w")                                           # [E, 1] signed S_ij
        sub = (s_e < 0).to(c_e.dtype)                               # substrates only
        log_term = sub * s_e.abs() * torch.log(c_e.clamp(min=self.c_floor))
        log_prod = H.scatter_along(self.edge_set, "post", log_term)  # [R, 1] SUM |S| log c per reaction
        k = rxn.get("k") if "k" in rxn.state_schema else torch.full_like(log_prod, self.k)
        if self.rate_noise > 0:
            k = k * (1.0 + self.rate_noise * torch.randn_like(k)).clamp(min=0.0)
        v = k * torch.exp(log_prod) * rxn.occ[:, None]
        if self.flux_limit_dt > 0:
            v_e = v[es.post]                                        # [E, 1] each edge's reaction rate
            consumed = sub * s_e.abs() * v_e * self.flux_limit_dt   # per edge, what the step takes
            total = H.scatter_along(self.edge_set, "pre", consumed)  # [M, 1] per metabolite
            c_m = H.level(es.pre_name).get("conc")
            scale_m = torch.where(total > 1e-12, (c_m / total.clamp(min=1e-12)).clamp(max=1.0),
                                  torch.ones_like(total))
            edge_scale = scale_m[es.pre] * sub + (1.0 - sub)         # products do not limit
            rxn_scale = torch.ones_like(v).scatter_reduce(0, es.post[:, None], edge_scale, reduce="amin",
                                                          include_self=True)
            v = v * rxn_scale
        if mask is not None:
            v = torch.where(mask[:, None], v, rxn.get("v"))
        st = rxn.state.clone()                                      # clone-and-reassign: autograd-safe
        a, b = rxn.state_schema["v"]
        st[:, a:b] = v
        rxn.state = st
        return {}


@register_operator("metabolite_flux", family="metabolism", set="metabolite", kind="aggregate")
class MetaboliteFlux(Aggregate):
    """What the reactions do to every concentration: the stoichiometric sum of their rates.

    (reaction, stoich) -> metabolite: lifts `v` onto the edges along `post`, weights by S_ij,
    sums onto the metabolite along `pre`, emits dc/dt.

        dc_i/dt = SUM_j S_ij v_j
    """

    EMIT = "velocity"                  # first-order: the engine integrates `conc`
    INPUTS = ["reaction", "stoich"]
    OUTPUTS = ["metabolite"]
    READS = ["v", "w"]
    WRITES = ["conc"]
    MAPS = ["pre", "post"]
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["edge_set"]
    MECHANISM_TAGS = ["stoichiometry", "metabolism", "mass_balance"]
    PARAM_ROLES = {"edge_set": "stoichiometry_as_edge_set"}
    REFERENCE = "MetabolismGraph generators/PDE_M1.py::forward (dx/dt = S v)"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "metabolite")
        self.edge_set = params["edge_set"]

    def forward(self, H, mask=None):
        met = H.level(self.at)
        es = H.level(self.edge_set)
        v_e = H.gather(self.edge_set, "post", "v")                  # [E, 1] the reaction's rate on its edge
        flux = H.scatter_along(self.edge_set, "pre", es.get("w") * v_e)   # [M, 1] SUM_j S_ij v_j
        dc = flux * met.occ[:, None]
        if mask is not None:
            dc = dc * mask[:, None].to(dc.dtype)
        return {self.at: dc}


@register_operator("metabolite_homeostasis", family="metabolism", set="metabolite", kind="lateral")
class MetaboliteHomeostasis(Lateral):
    """The homeostatic pull of every concentration towards its baseline.

    metabolite -> metabolite: reads `conc` and `c0`, emits dc/dt. No map.

        dc_i/dt += -lambda (c_i - c0_i (1 + A sin(2 pi t / T)))

    lambda is `strength` in inverse time; c0_i is the `c0` block the seed wrote (else `baseline`);
    A and T (`circadian_amplitude`, `circadian_period`, in ticks) modulate the target, 0 = none.
    """

    EMIT = "velocity"
    INPUTS = ["metabolite"]
    OUTPUTS = ["metabolite"]
    READS = ["conc", "c0"]
    WRITES = ["conc"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["homeostasis", "metabolism", "relaxation"]
    PARAM_ROLES = {"strength": "homeostatic_rate_inverse_time", "baseline": "target_concentration",
                   "circadian_amplitude": "target_modulation_amplitude",
                   "circadian_period": "target_modulation_period_ticks"}
    REFERENCE = "MetabolismGraph generators/PDE_M1.py::forward (homeostatic term)"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "metabolite")
        self.strength = float(params.get("strength", 0.0))
        self.baseline = float(params.get("baseline", 5.0))
        self.amp = float(params.get("circadian_amplitude", 0.0))
        self.period = float(params.get("circadian_period", 1440.0))
        self._tick = 0

    def forward(self, H, mask=None):
        met = H.level(self.at)
        c = met.get("conc")
        c0 = met.get("c0") if "c0" in met.state_schema else torch.full_like(c, self.baseline)
        if self.amp > 0:
            c0 = c0 * (1.0 + self.amp * math.sin(2.0 * math.pi * self._tick / self.period))
        self._tick += 1
        dc = -self.strength * (c - c0) * met.occ[:, None]
        if mask is not None:
            dc = dc * mask[:, None].to(dc.dtype)
        return {self.at: dc}


@register_operator("reaction_seed", family="seed", set="reaction", kind="seed")
class ReactionSeed(Seed):
    """x_0 for the reactions: every rate constant k_j drawn log-uniformly in [k_min, k_max], once.

    reaction -> reaction: writes `k` (and zeroes `v`). The rate constants are the inverse
    model's target, so they are state of the reaction set, not a number inside an operator."""

    EMIT = None
    INPUTS = ["reaction"]
    OUTPUTS = ["reaction"]
    READS = []
    WRITES = ["k", "v"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["initial_condition", "metabolism", "rate_constants"]
    PARAM_ROLES = {"k_min": "rate_constant_low", "k_max": "rate_constant_high", "seed": "random_seed"}
    REFERENCE = "MetabolismGraph generators/PDE_M1.py (log_k, per reaction)"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "reaction")
        self.k_min = float(params.get("k_min", 0.1))
        self.k_max = float(params.get("k_max", 1.0))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        u = torch.rand(lvl.n, 1, generator=g)
        k = (math.log(self.k_min) + u * (math.log(self.k_max) - math.log(self.k_min))).exp()
        st = lvl.state.clone()
        a, b = lvl.state_schema["k"]
        st[:, a:b] = k.to(st.device, st.dtype)
        if "v" in lvl.state_schema:
            a1, b1 = lvl.state_schema["v"]
            st[:, a1:b1] = 0.0
        lvl.state = st
        return {}
