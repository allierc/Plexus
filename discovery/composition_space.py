"""composition_space -- the space the Okuda campaign searches: TYPED COMPOSITION GRAPHS.

A candidate mechanism is a GRAPH, not a Boolean feature vector:
  - operator NODES, each pinned to one IMPLEMENTATION,
  - typed CONNECTIONS from an output port to an input slot.

Routing the morphogen to growth's `gate`, to division's `axis`, or to the extrusion `site` gives
three DIFFERENT MECHANISMS, not three parameter settings.

--------------------------------------------------------------------------------------------
IDENTITY RULE (the line that keeps the campaign honest)
--------------------------------------------------------------------------------------------
    structure = {which operators, which IMPLEMENTATIONS, which typed connections}
    theta     = the numbers
    comp_hash(structure)  -- theta is EXCLUDED

So a change of numbers PROVABLY cannot register as a new hypothesis. That is the structural fix
for what happened over rounds 01-30 of the hand-run campaign.

*Why implementations are part of identity here.*  plexus2 says implementations of one contract
"differ only in numerics".  That holds for e.g. finite-difference vs spectral diffusion.  It does
NOT hold for the three reaction kinetics: the record shows Brusselator decays or reorganises a
seed, Gray-Scott holds it, and Gierer-Meinhardt amplifies it into a stable gradient peak -- three
qualitatively different phenomenologies.  Same for shape_energy_3d default (mid-surface wedge
volume) vs monolayer (true A*h volume, emergent bending).  Since the campaign must be able to ASK
"which kinetics", implementation is part of composition identity and is flagged
`impl_structural=True` on those operators.  This is a deliberate, recorded departure.

--------------------------------------------------------------------------------------------
EXTRUSION IS A NODE, NOT A PARAMETER
--------------------------------------------------------------------------------------------
The campaign's central question is round 41's finding: our tube is held open by an explicit
outward force, where Okuda's is a growth-driven quasi-static equilibrium.  If the forcing lives
in a config file it can only be turned down.  As a graph node, the standard necessity protocol
ABLATES IT AUTOMATICALLY on every composition -- so the central question becomes a routine
measurement rather than a special experiment.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np

# ============================================================================ port types
#   morphogen  -- a per-cell chemical amount/concentration (the RD field)
#   adjacency  -- the cell-cell neighbour relation read off the half-edge mesh
#   geometry   -- per-cell centroid / area / volume, aggregated from the vertex set
PORT_TYPES = ("morphogen", "adjacency", "geometry")

# ============================================================================ clock re-anchoring
#
# ⚠ THE PER-CALL / PER-FRAME TRAP  (raised 2026-07-30 after FINDING 8)
#
# `divide_3d` counts `min_cycle` / `max_cycle` in DIVISION-CALLS (its own docstring: "a cell may
# not divide before this many division-calls since birth"; `age` is "per-cell age in
# division-calls"), and `max_div_frac` is a PER-CALL throttle. The archived configs passed
# `every: 2`, which the ENGINE gated and the operator's private `self._k` ALSO gated -- product 4.
# So in the archived runs `divide_3d.forward` executed once every FOUR frames:
#
#     min_cycle = 8 calls      ==  32 frames        max_div_frac = 0.03/call  ==  0.0075/frame
#
# With the clock fixed (`every: 1`, engine owns it) the same numbers mean:
#
#     min_cycle = 8 calls      ==   8 frames        max_div_frac = 0.03/call  ==  0.03  /frame
#
# i.e. FOUR TIMES the proliferation. Leaving the hand-tuned values as vocabulary defaults would
# silently start every generated composition at a theta tuned for the wrong clock -- which is
# exactly what FINDING 8 measured (aspect 7.5 -> 3.2, cells 2700 -> 3335).
#
# Because the factor is exact and only `divide_3d` carried `every: 2`, we do NOT need a re-tuning
# sweep: the archived working point is recoverable analytically by rescaling the per-call
# quantities. That makes the D1 fix BEHAVIOUR-PRESERVING BY CONSTRUCTION, so any later change in
# phenotype is attributable to the change that caused it rather than to the clock.
DIVIDE_CALL_PERIOD_BEFORE_D1 = 4        # engine every=2  x  private self.every=2
CLOCK_COUPLED = {                       # param -> how to convert an archived value to per-frame
    "min_cycle":     "multiply by 4  (calls -> frames)",
    "max_cycle":     "multiply by 4  (calls -> frames)",
    "max_div_frac":  "divide by 4    (per-call -> per-frame)",
    "max_div":       "divide by 4    (per-call FLOOR; cap_div = max(max_div, frac*nF), so this "
                     "DOMINATES at realistic nF -- rescaling frac alone is entirely masked)",
}
# NOT clock-coupled (evaluated every frame either way): p0, Gamma, Lambda, h0, relax_iters, l_th.
# PARTIALLY coupled and therefore still provisional: K_V was raised to 6.0 specifically to crush
# the cell-size CV produced by the division wave; vcap is a per-call bypass. Both are flagged in
# validate_space (V10) and must be confirmed against the re-anchored baseline.
# vcap: CLAIM RETRACTED 2026-07-30 (Metrologist RET001). I asserted it was "not rate-coupled --
#   the same cells divide, only sooner". That is true of ONE cell and false of the POPULATION:
#   vcap divisions bypass the throttle entirely (`cap_div = max(cap_div, len(over))`), so
#   checking 4x more often makes oversized cells divide sooner, their daughters begin growing
#   sooner, and the total division count over a run differs. It is weakly rate-coupled and the
#   correct rescaling is NOT yet established.
#
#   CORRECTION (same day): the evidence I cited for this was a FALSE DISCREPANCY OF MY OWN
#   MAKING. I compared our r95/median metric -- which I had misnamed `aspect` -- against the
#   report's "aspect ~7.5" for round_40_mc8, which is tube_len/tube_diam. Two different
#   quantities. tube_analysis.py:89 calls r95/median `protr`, and ours is now named `protr` too.
#   The re-anchored replay recovered the archived CELL COUNT (2927 vs ~2700); whether it recovers
#   the archived tube_len/tube_diam is being measured with the archive's OWN metric bank.
#   vcap's rate-coupling therefore remains UNTESTED -- neither proven nor disproven. Which is
#   still the honest state, but for a different reason than I first recorded.
#   cycle_cv NOT clock-coupled. A dimensionless Gaussian CV on the per-cell threshold multiplier.
#   K_V      Its MEANING was never clock-coupled (a per-frame mechanical stiffness). Its
#            OPTIMALITY was stale only because it was tuned against the division wave -- and the
#            re-anchoring restores exactly that wave, so K_V = 6.0 is valid again.
PROVISIONAL_THETA = ("vcap",)   # NOT settled. Metrologist D1d: the clock re-anchoring restores
#   the CELL COUNT (2927 vs ~2700) but NOT the phenotype -- archived is a long thin tube, the
#   replay is a small bud. vcap force-divides oversized cells bypassing the throttle, so checking
#   4x more often splits tip cells the moment they cross instead of letting them ramp while
#   queued -- and the report attributes the tube tip specifically to that backlog. Sweep required.

# ============================================================================ derived limits
#
# ⚠ THE SILENT CAP (raised 2026-07-31)
#
# `params` used to hold hand-written (lo, hi, default) triples and `sample_params` did
#
#       p[k] = float(np.clip(d + rng.normal(0, scale*(hi-lo)), lo, hi))
#
# a SILENT hard clamp. Measured on `okuda_route` at the shipped scale=0.15: 904/6200 = 14.6% of
# draws (over 200 seeds) left the box and were pushed back onto the bound WITHOUT a warning, a
# flag, or a return value the caller could see. Four of the 31 values in the seed-0 draw sat
# EXACTLY on a bound. Meanwhile `with_params` -- the path a HUMAN sweep takes (round.py:167) --
# validated nothing at all. The box bound the agent, did not bind us, and informed neither.
#
# This is the buffer-ceiling defect again (critic P2_BUFFER_SATURATED: "evidence about a buffer,
# not a mechanism"): an invisible limit that converts "we did not look there" into "there is
# nothing there". For a campaign whose headline findings are IMPOSSIBILITY claims that is the
# most dangerous bug available. Three of Okuda's own published values were outside the box:
# alpha=10 (box 1..8), d_h=10 (box 0.1..2.0), and rd_rate over four decades (box 0.2..3.0).
#
# THE RULE ADOPTED HERE
#   A bound is either PHYSICAL AND DERIVED, or it is absent.
#   Two different things had been conflated into one pair of numbers, which is WHY the box was
#   kept artificially tight -- widening it also widened the robustness basin:
#       BOX   (lo, hi)      the ADMISSIBLE region. Derived below, never hand-written.
#       BASIN PARAM_BASIN   the sampling width a robustness claim is made over. Explicit, and
#                           deliberately UNCHANGED by the widening.
#   Nothing is clipped. Leaving the admissible region is a named, visible condition (THETA_RULES).
#
# WHERE THE NUMBERS COME FROM -- all three limits were re-derived from the operator source and
# then CONFIRMED by running the operator's own arithmetic (see the smoke test at the bottom).
ENGINE_DT = 0.02
# ^ MUST equal translate.DT_GLOBAL (D2: one dt for the whole campaign). It cannot be imported at
#   module scope -- translate imports THIS module -- so it is duplicated and the duplication is
#   converted into a CHECKED invariant by check_dt_agreement(), called from theta_conditions().
#   A silent disagreement would make every derived bound below wrong by the ratio of the two dts.

FLOAT32_MAX = 3.4028234663852886e38   # the state dtype the operators run in
HILL_EPS = 1e-12          # the additive regulariser in MorphogenGrowth3D.forward's Hill function
GM_MU_A = 1.0             # hard-wired by translate._emit_react (gm_rho=1.0, mu_a=1.0)
BRUSSELATOR_B = 3.0       # hard-wired by translate._emit_react (A=1.0, B=3.0)

DIFFUSION_CFL_LIMIT = 1.0     # see diffusion_cfl
REACTION_EULER_LIMIT = 2.0    # |1 - dt*k| <= 1 for a linear decay k


def check_dt_agreement():
    """Fail LOUDLY if ENGINE_DT has drifted from the dt the campaign actually runs at.

    Every limit in this section is a function of dt. If the two copies disagree, the bounds are
    silently wrong -- the exact failure mode this whole section exists to remove.
    """
    from translate import DT_GLOBAL          # lazy: translate imports composition_space
    if abs(float(DT_GLOBAL) - ENGINE_DT) > 1e-15:
        raise RuntimeError(
            f"ENGINE_DT={ENGINE_DT} disagrees with translate.DT_GLOBAL={DT_GLOBAL}. Every derived "
            f"stability bound in composition_space is a function of dt and is now wrong.")
    return True


# ---------------------------------------------------------------- explicit diffusion (cell_diffuse)
# tyssue_rd_ops.CellDiffuse.forward returns   coef * lap,  coef = [d_a, d_h] * chi
#   norm=True (the default, and the only mode translate emits):  lap = mean(neighbours) - self
# and the engine integrates it explicitly:  engine.py  `new = new + dt * d`.
# The degree-normalised Laplacian has eigenvalues in [-2, 0] AT ANY CELL DEGREE, so the per-step
# amplification is 1 + dt*chi*d*lambda and stability needs |1 + dt*chi*d*lambda| <= 1, i.e.
#
#       dt * chi * max(d_a, d_h) * stencil_gain  <=  1
#
# The same statement in physical terms: the fraction of its own content a cell hands to its
# neighbours in one step cannot exceed what it holds. `stencil_gain` is the connectivity factor --
# 1 for the DEGREE-NORMALISED stencil (the normalisation is exactly what removes the degree
# dependence), max cell degree for a stencil that does not normalise.
# CONFIRMED numerically on the operator's own arithmetic: dt*chi*d = 1.000 is stable over 400
# steps, 1.010 diverges (K2 -> 1.4e3, 64-ring -> 5.6e1, odd 63-ring -> 7.3e14).
DIFFUSION_STENCIL_GAIN = {
    "graph_laplacian": 1.0,   # DERIVED from source: `agg / deg - chem` is degree-normalised.
    "interface_weighted": None,   # NOT YET DERIVED -- that implementation is being written in
    #   tyssue_rd_ops.py concurrently and its stencil has not been read. `None` means "we have not
    #   derived this", which is NOT the same as 1.0 and must never be silently rounded to it. The
    #   CFL is then evaluated at the most permissive gain (1.0) so we never BLOCK a composition on
    #   a number we did not derive, and T5_STENCIL_GAIN_UNDERIVED says so out loud.
}


def diffusion_cfl(d_a, d_h, chi, dt=None, stencil_gain=1.0):
    """The explicit-diffusion CFL number. Stable iff <= DIFFUSION_CFL_LIMIT (== 1)."""
    dt = ENGINE_DT if dt is None else dt
    return float(dt) * float(chi) * float(stencil_gain) * max(float(d_a), float(d_h))


def diffusivity_ceiling(chi, dt=None, stencil_gain=1.0):
    """Largest d_a/d_h an explicit step can carry at this chi. Computed, never hard-coded."""
    dt = ENGINE_DT if dt is None else dt
    return DIFFUSION_CFL_LIMIT / (float(dt) * float(chi) * float(stencil_gain))


def chi_ceiling(d, dt=None, stencil_gain=1.0):
    """Largest chi an explicit step can carry at this diffusivity."""
    dt = ENGINE_DT if dt is None else dt
    return DIFFUSION_CFL_LIMIT / (float(dt) * float(d) * float(stencil_gain))


# ---------------------------------------------------------------- explicit reaction (cell_react)
# Same integrator, same argument, applied to the stiffest LINEAR term of each kinetics (read off
# tyssue_rd_ops): |1 - dt*k| <= 1  =>  dt*k <= 2.
def reaction_stiffness(impl, rd_rate=1.0, mu_h=1.0, F=0.055, kk=0.062, gamma=0.3):
    """Stiffest linear decay rate of a kinetics implementation, in engine time units."""
    if impl == "gierer_meinhardt":       # da = ... - mu_a*a ;  dh = ... - mu_h*h ; scaled by rate
        return float(rd_rate) * max(GM_MU_A, float(mu_h))
    if impl == "gray_scott":             # da = ... - (F+kk)*a ; scaled by rate
        return float(rd_rate) * (float(F) + float(kk))
    if impl == "brusselator":            # da = gamma*(... - (B+1)*a ...); `rate` is NOT read
        return float(gamma) * (BRUSSELATOR_B + 1.0)
    return 0.0


def rd_rate_ceiling(impl, mu_h=1.0, F=0.055, kk=0.062, dt=None):
    """Largest rd_rate an explicit step can carry for this kinetics. `brusselator` ignores
    rd_rate entirely (it is driven by `gamma`), so it imposes no ceiling on it."""
    dt = ENGINE_DT if dt is None else dt
    k = reaction_stiffness(impl, rd_rate=1.0, mu_h=mu_h, F=F, kk=kk)
    return float("inf") if k <= 0 else REACTION_EULER_LIMIT / (float(dt) * k)


# ---------------------------------------------------------------- the growth switch (Hill)
# MorphogenGrowth3D.forward:  hillv = a**alpha / (a_sw**alpha + a**alpha + HILL_EPS)
# `a_sw**alpha` must stay REPRESENTABLE and must stay DISTINGUISHABLE from the regulariser:
#   a_sw > 1 : alpha*ln(a_sw) <  ln(FLOAT32_MAX)  else it overflows -> hillv == 0, growth silently
#              dies with no error
#   a_sw < 1 : alpha*ln(a_sw) >  ln(HILL_EPS)     else the threshold sinks below HILL_EPS and
#              hillv == 1 everywhere -> the switch silently becomes ALWAYS-ON, i.e. the operator
#              stops being morphogen-gated at all while still appearing to run
# Both failures are silent and both would be recorded as evidence about the mechanism.
def hill_alpha_ceiling(a_sw):
    """Largest Hill exponent at which the switch is still a switch, at this a_sw."""
    a_sw = float(a_sw)
    if a_sw == 1.0:
        return float("inf")
    if a_sw > 1.0:
        return np.log(FLOAT32_MAX) / np.log(a_sw)
    return np.log(HILL_EPS) / np.log(a_sw)


# ---------------------------------------------------------------- the boxes, computed
# HOW A BOX CEILING IS SET: the derived limit evaluated with the parameter's COMPANIONS AT THEIR
# DEFAULTS. The constraints are JOINT (chi x d, alpha x a_sw), so no per-parameter box can be
# sufficient on its own -- a value inside the box can still be unstable in combination. That is
# what the joint conditions in theta_conditions() are for. The box is the reachability envelope;
# the joint condition is the wall.
CHI_DEFAULT, D_A_DEFAULT, D_H_DEFAULT = 4.0, 0.02, 0.7
MU_H_DEFAULT, F_DEFAULT, KK_DEFAULT = 1.0, 0.055, 0.062
A_SW_MIN, A_SW_MAX, A_SW_DEFAULT = 0.2, 6.0, 1.5

D_CEIL = diffusivity_ceiling(CHI_DEFAULT)          # 12.5   -- reaches Okuda's d_h = 10
CHI_CEIL = chi_ceiling(D_H_DEFAULT)                # 71.4
RD_RATE_CEIL = min(rd_rate_ceiling("gierer_meinhardt", mu_h=MU_H_DEFAULT),
                   rd_rate_ceiling("gray_scott", F=F_DEFAULT, kk=KK_DEFAULT))   # 100 (GM binds)
GAMMA_CEIL = REACTION_EULER_LIMIT / (ENGINE_DT * (BRUSSELATOR_B + 1.0))         # 25
# alpha is the ONE place the default-companion rule is deliberately not used. The overflow term is
# `a**alpha` in the ACTIVATOR, whose magnitude is state-dependent and cannot be bounded statically
# at all; so the static box takes the conservative branch instead -- the largest alpha that is
# safe for EVERY admissible a_sw. That is 17.2, which still reaches Okuda's alpha = 10.
ALPHA_CEIL = min(hill_alpha_ceiling(A_SW_MIN), hill_alpha_ceiling(A_SW_MAX))    # 17.2

# ============================================================================ vocabulary
# stage           -- the gate; the search opens stages in order
# role            -- for post-hoc naming and proximity clustering
# outputs         -- port types this operator produces
# slots           -- input ports another operator's output may connect to
# impls           -- available implementations; impls[0] is the default
# impl_structural -- True => the implementation choice changes the phenomenology, so it is part
#                    of composition identity (see module docstring)
# needs           -- PRECONDITIONS (defect D4): port types that must be produced by SOME node in
#                    the graph, else this operator silently no-ops. The Critic rejects for free.
# params          -- (lo, hi, default); theta only, never part of identity
OPERATORS = {
    # ---------------------------------------------------------------- Stage 1: substrate
    "seed_mesh_3d": dict(
        stage=1, role="substrate", outputs=[], slots=[], needs=[],
        impls=["fibonacci_sphere", "checkpoint"], impl_structural=False,
        params={"n_cells": (150, 2000, 500), "vseed_cv": (0.0, 0.5, 0.15)}),
    "shape_energy_3d": dict(
        stage=1, role="mechanics", outputs=["geometry"], slots=[], needs=[],
        impls=["default", "monolayer"], impl_structural=True,       # mid-surface vs true 3D volume
        params={"K_V": (1.0, 8.0, 6.0), "kappa_s": (0.05, 0.6, 0.2),
                "Gamma": (0.0, 0.4, 0.05), "Lambda": (0.0, 0.3, 0.20),
                "p0": (3.4, 4.2, 3.90), "h0": (0.05, 0.4, 0.40), "mono_gamma": (0.0, 0.3, 0.06),
                "relax_iters": (10, 90, 30)}),
    "reconnect_t1_3d": dict(
        stage=1, role="topology", outputs=[], slots=[], needs=[],
        impls=["length_threshold"], impl_structural=False,
        params={"l_th": (0.01, 0.12, 0.04)}),

    # ---------------------------------------------------------------- Stage 2: growth & topology
    "vesicle_growth": dict(                                  # uniform, body-wide inflation
        stage=2, role="growth", outputs=[], slots=[], needs=[],
        impls=["uniform_ramp"], impl_structural=False,
        params={"rate": (0.002, 0.03, 0.006)}),
    "morphogen_growth_3d": dict(                             # LOCAL growth, gated by the activator
        stage=2, role="growth", outputs=[], slots=["gate"], needs=["morphogen"],
        impls=["hill_conserve_amount", "hill_no_conserve"], impl_structural=True,
        # alpha: was (1.0, 8.0) -- a hand-written cap that put OKUDA'S OWN alpha = 10 outside the
        # searchable space. Now ALPHA_CEIL, derived from the Hill function's float32/HILL_EPS
        # degeneracy (see hill_alpha_ceiling). lo = 1.0 stays: below 1 the Hill switch has
        # infinite slope at the origin and stops being a switch.
        params={"rate": (0.002, 0.03, 0.010), "a_sw": (A_SW_MIN, A_SW_MAX, A_SW_DEFAULT),
                "alpha": (1.0, ALPHA_CEIL, 4.0), "rho": (0.0, 1.0, 0.0)}),
    "divide_3d": dict(
        # `hertwig` splits normal to the cell's OWN longest axis -> needs no morphogen input.
        # `orient_iface` stacks daughters along the bud axis -> needs the activator routed in.
        # Slots are therefore PER IMPLEMENTATION; declaring `axis` unconditionally would make
        # every hertwig composition look like it had a dangling (inert) slot.
        stage=2, role="topology", outputs=[], slots=[], impl_slots={"orient_iface": ["axis"]},
        needs=[],
        impls=["hertwig", "orient_iface"], impl_structural=True,   # long-axis vs bud-axis septum
        params={"cycle_cv": (0.05, 0.5, 0.40), "min_cycle": (2, 64, 16),   # 4 calls x 4
                "max_cycle": (6, 10**9, 10**9), "vcap": (0.0, 3.0, 1.5),   # vcap: PROVISIONAL
                "max_div_frac": (0.00125, 0.20, 0.0075),   # 0.03/call / 4 = per-frame
                "max_div": (4, 480, 30),                   # 120/call / 4 = per-frame FLOOR
                "orient_asw": (0.2, 6.0, 1.0)}),
    "extrude": dict(                                          # THE FORCING TERM -- ablatable
        stage=2, role="forcing", outputs=[], slots=["site"], needs=["morphogen"],
        impls=["radial_push"], impl_structural=False,
        params={"K_extrude": (0.0, 14.0, 4.0), "a_sw": (0.2, 6.0, 0.5)}),

    # ---------------------------------------------------------------- Stage 3: patterning
    "cell_geometry_3d": dict(
        stage=3, role="readout", outputs=["geometry"], slots=[], needs=[],
        impls=["scatter_add"], impl_structural=False, params={}),
    "cell_adjacency": dict(
        stage=3, role="readout", outputs=["adjacency"], slots=[], needs=[],
        impls=["shared_edge"], impl_structural=False, params={}),
    "cell_diffuse": dict(
        stage=3, role="patterning", outputs=[], slots=[], needs=["adjacency"],
        # TWO implementations, and the choice is STRUCTURAL. `graph_laplacian` couples every
        # neighbour equally; `interface_weighted` couples through the shared-interface area, so
        # the coupling follows the geometry the mechanics is deforming. Those are two different
        # claims about how the morphogen moves, not two numerics -- the same reason cell_react's
        # kinetics are structural (module docstring). Registering both with impl_structural=True
        # makes ABLATING THE COUPLING a legal ONE-EDIT move (`=cell_diffuse:interface_weighted`),
        # so the loop can run that ablation itself instead of waiting for a human to hand-write it.
        impls=["graph_laplacian", "interface_weighted"], impl_structural=True,
        # d_a/d_h/chi: was (0.005,0.2)/(0.1,2.0)/(1.0,10.0) -- hand-written, and it put OKUDA'S
        # OWN inhibitor spread d_h = 10 outside the space. The real limit is the explicit-diffusion
        # CFL and it is JOINT in (chi, d): see diffusion_cfl / T1_DIFFUSION_UNSTABLE.
        params={"d_a": (0.0, D_CEIL, D_A_DEFAULT), "d_h": (0.0, D_CEIL, D_H_DEFAULT),
                "chi": (0.0, CHI_CEIL, CHI_DEFAULT)}),
    "cell_react": dict(
        stage=3, role="patterning", outputs=["morphogen"], slots=[], needs=["adjacency"],
        impls=["gierer_meinhardt", "gray_scott", "brusselator"], impl_structural=True,
        # rd_rate: was (0.2, 3.0) -- a factor of 15, where Okuda spans FOUR DECADES. lo = 0.0 is
        # the physical bound (a rate cannot be negative); hi = RD_RATE_CEIL = 100 is the derived
        # explicit-Euler ceiling of the stiffest kinetics at its defaults. 0.01 .. 100 is now
        # inside the box, so all four decades are reachable.
        # gamma (brusselator): was (0.1, 100.0). Not a cap at all -- 100 is four times the
        # integrator's own limit, i.e. the box positively invited a guaranteed divergence. Now the
        # derived GAMMA_CEIL. This is the same defect with the sign flipped: an undrived bound.
        params={"gamma": (0.0, GAMMA_CEIL, 0.3), "a0": (0.0, 0.05, 0.01),
                "rd_rate": (0.0, RD_RATE_CEIL, 1.0),
                "F": (0.02, 0.06, F_DEFAULT), "kk": (0.05, 0.07, KK_DEFAULT),
                "mu_h": (0.2, 2.0, MU_H_DEFAULT)}),
    "cell_rd_seed": dict(                                     # the prescribed activation driver
        stage=3, role="driver", outputs=["morphogen"], slots=[], needs=[],
        impls=["tip", "cone", "spot"], impl_structural=True,
        params={"tip_radius": (0.6, 3.0, 2.0), "cone_deg": (4.0, 30.0, 8.0),
                "amp": (0.5, 5.0, 2.0), "n_spots": (1, 8, 1)}),
    # NOTE: there is deliberately no separate `rd_interface_tension` node. In the engine that op
    # carries BOTH K_purse and K_extrude; the mechanism we need to ablate is the outward forcing,
    # so it is exposed once, as `extrude`. A second node would be the same engine operator under
    # two names -- which would let one mechanism occupy two points of the search space.
}

# Slots may depend on the chosen implementation (see divide_3d).
def slots_of(op: str, impl: str):
    spec = OPERATORS[op]
    return list(spec.get("impl_slots", {}).get(impl, spec["slots"]))


# ============================================================================ the sampling basin
# The BOX is the admissible region (derived above). The BASIN is how far `sample_params` wanders
# around the default -- the width a ROBUSTNESS claim is made over. They were the same numbers,
# which is why widening the box was expensive: it silently widened the basin too, so any claim
# of the form "this result survives a 15% perturbation" would have changed meaning underneath us.
#
# Default rule (unlisted parameters): sigma = scale * (hi - lo), i.e. EXACTLY the old behaviour.
# Listed here: sigma pinned to the width of the OLD, pre-widening box, so this change is
# basin-preserving by construction and no existing robustness claim changes meaning.
PARAM_BASIN = {
    ("morphogen_growth_3d", "alpha"): 8.0 - 1.0,        # old box (1.0, 8.0)
    ("cell_diffuse", "d_a"): 0.2 - 0.005,               # old box (0.005, 0.2)
    ("cell_diffuse", "d_h"): 2.0 - 0.1,                 # old box (0.1, 2.0)
    ("cell_diffuse", "chi"): 10.0 - 1.0,                # old box (1.0, 10.0)
    ("cell_react", "rd_rate"): 3.0 - 0.2,               # old box (0.2, 3.0)
    ("cell_react", "gamma"): GAMMA_CEIL,                # box NARROWED to the derived limit; the
    #   old width (99.9) sampled gamma far past the integrator's ceiling, so preserving it would
    #   preserve a basin most of which cannot be integrated.
    ("divide_3d", "max_cycle"): 0.0,                    # 1e9 is a SENTINEL for "no maximum", not a
    #   tunable. The old rule gave it sigma = 1.5e8 and the clip then pinned half the draws back
    #   onto 1e9 -- noise that looked like sampling. Fixed = never perturbed.
}


def basin_sigma(op, pname, scale=0.15):
    """Sampling width for one parameter. Explicit if listed, else the old scale*(hi-lo) rule."""
    if (op, pname) in PARAM_BASIN:
        return float(PARAM_BASIN[(op, pname)]) * float(scale)
    lo, hi, _ = OPERATORS[op]["params"][pname]
    return float(scale) * (float(hi) - float(lo))


# ============================================================================ theta conditions
# NAMED, VISIBLE conditions on theta -- the replacement for the silent clip.
#
# Deliberately shaped like critic.Rejection (code / rule / detail) so the Critic can adopt them
# with a one-line rule, and phrased in the same "this run is NOT evidence" idiom as
# P2_BUFFER_SATURATED. A run whose chemistry diverged is evidence about an integrator, not a
# mechanism, in exactly the way a saturated run is evidence about a buffer.
#
# BLOCKING vs not:
#   blocking=True  we DERIVED the limit and it is breached -> is_runnable() is False and to_spec
#                  refuses to compile. The composition cannot produce evidence.
#   blocking=False visible but not a wall. Used for (a) leaving the declared box, which is a
#                  prior and not a physical fact, and (b) any limit we have NOT derived. We never
#                  block on a number we did not derive -- that is how the original cap happened.
THETA_RULES = ["T1_DIFFUSION_UNSTABLE", "T2_REACTION_UNSTABLE", "T3_HILL_SWITCH_DEGENERATE",
               "T4_OUTSIDE_DECLARED_BOX", "T5_STENCIL_GAIN_UNDERIVED"]


class ThetaCondition:
    __slots__ = ("code", "rule", "detail", "blocking")

    def __init__(self, code, rule, detail, blocking):
        self.code, self.rule, self.detail, self.blocking = code, rule, detail, blocking

    def __repr__(self):
        return f"<{'!' if self.blocking else '?'}{self.code}: {self.detail}>"

    def line(self):
        """One loud line, in run_one.py's idiom."""
        mark = "\U0001f534" if self.blocking else "⚠"
        return f"{mark} {self.code} -- {self.rule}: {self.detail}"

STAGES = {s: [k for k, v in OPERATORS.items() if v["stage"] == s] for s in (1, 2, 3)}

# which (output port type -> input slot) connections are TYPE-LEGAL
LEGAL_LINKS = {
    ("morphogen", "gate"),    # activator drives local growth      -- the Okuda coupling
    ("morphogen", "axis"),    # activator orients the division plane
    ("morphogen", "site"),    # activator selects where to push     -- the forcing route
    ("morphogen", "field"),   # activator raises interface tension
}

# operators that must be present for the run to be meaningful at all
REQUIRED_ROLES = {"substrate", "mechanics"}


# ============================================================================ the graph
class CompositionGraph:
    def __init__(self, ops=None, conns=None, params=None):
        # ops:   [{"id": str, "op": name, "impl": str}]
        # conns: [{"src": id, "dst": id, "slot": str}]
        self.ops = [dict(o) for o in (ops or [])]
        self.conns = [dict(c) for c in (conns or [])]
        self.params = dict(params or {})

    # ---------------------------------------------------------------- identity
    def structure(self):
        """Operators (+ structural implementations) and typed connections. NO theta."""
        ops = sorted(
            ({"id": o["id"], "op": o["op"],
              **({"impl": o.get("impl", OPERATORS[o["op"]]["impls"][0])}
                 if OPERATORS[o["op"]]["impl_structural"] else {})}
             for o in self.ops),
            key=lambda o: (o["op"], o.get("impl", ""), o["id"]))
        conns = sorted(
            ({"src_op": self._op_of(c["src"]), "dst_op": self._op_of(c["dst"]), "slot": c["slot"]}
             for c in self.conns),
            key=lambda c: (str(c["src_op"]), str(c["dst_op"]), c["slot"]))
        return {"operators": ops, "connections": conns}

    def _op_of(self, node_id):
        return next((o["op"] for o in self.ops if o["id"] == node_id), None)

    def _node(self, node_id):
        return next((o for o in self.ops if o["id"] == node_id), None)

    def op_names(self):
        return [o["op"] for o in self.ops]

    def impl_of(self, node):
        return node.get("impl", OPERATORS[node["op"]]["impls"][0])

    def roles(self):
        return {OPERATORS[o["op"]]["role"] for o in self.ops}

    def produced_ports(self):
        out = set()
        for o in self.ops:
            out.update(OPERATORS[o["op"]]["outputs"])
        return out

    def copy(self):
        g = CompositionGraph(self.ops, self.conns, self.params)
        return g

    # ---------------------------------------------------------------- theta
    def default_params(self):
        p = {}
        for o in self.ops:
            for pn, (lo, hi, d) in OPERATORS[o["op"]]["params"].items():
                p[f"{o['id']}.{pn}"] = d
        return p

    def theta(self, node_id, pname):
        """theta lookup with the vocabulary default as fallback (same rule as translate._p)."""
        k = f"{node_id}.{pname}"
        if k in self.params:
            return self.params[k]
        return OPERATORS[self._op_of(node_id)]["params"][pname][2]

    def sample_params(self, rng, scale=0.15, verbose=True):
        """Perturb around the defaults -- the PARAMETER BASIN a robustness claim is made over.

        NEVER CLIPS. The previous implementation ended in `np.clip(..., lo, hi)`, which silently
        moved 14.6% of draws onto a bound: the sampler reported a value it had not drawn, the
        basin developed spikes of probability mass exactly at the edges, and nothing anywhere
        said so. A draw that leaves the admissible region is now either re-drawn (a proper
        truncated sample, no mass piled on the bound) or, if the basin cannot fit inside the box
        at all, RETURNED AS DRAWN and reported as a named condition. Out-of-box is a fact about
        the run; it is not something the sampler is allowed to quietly edit away.
        """
        p, cond = self.sample_params_report(rng, scale)
        if verbose:
            for c in cond:
                print(c.line())
        return p

    def sample_params_report(self, rng, scale=0.15, max_tries=64):
        """(params, [ThetaCondition]) -- the sampler with its excursions made explicit."""
        p = self.default_params()
        cond = []
        for o in self.ops:
            for pn, (lo, hi, d) in OPERATORS[o["op"]]["params"].items():
                k = f"{o['id']}.{pn}"
                sigma = basin_sigma(o["op"], pn, scale)
                if sigma <= 0:
                    p[k] = float(d)
                    continue
                v = float(d)
                for _ in range(max_tries):        # truncated draw: RE-draw, never clamp
                    v = float(d + rng.normal(0, sigma))
                    if lo <= v <= hi:
                        break
                else:
                    # the basin does not fit inside the box. Report the draw as it fell; the
                    # alternative (clamping) is precisely the defect.
                    cond.append(ThetaCondition(
                        "T4_OUTSIDE_DECLARED_BOX",
                        "a sampled value left the declared box and was NOT clipped",
                        f"{k}={v:.6g} not in [{lo:.6g}, {hi:.6g}] after {max_tries} draws at "
                        f"sigma={sigma:.6g} -- widen the box or narrow the basin", False))
                p[k] = v
        cond.extend(self.with_params(p, quiet=True).theta_conditions())
        return p, cond

    def with_params(self, params, quiet=False):
        """Attach theta. VALIDATES -- and says so out loud.

        This used to do no checking whatsoever, so the hand-written sweep path (round.py) was not
        bound by the ranges that bound the agent, and neither side was told. The box binding one
        of the two parties and informing neither is how a limit becomes invisible.
        """
        g = self.copy()
        g.params = dict(params)
        if not quiet:
            for c in g.theta_conditions():
                print(c.line())
        return g

    # ---------------------------------------------------------------- derived theta limits
    def theta_conditions(self):
        """[ThetaCondition] -- every named condition this theta triggers. See THETA_RULES."""
        check_dt_agreement()      # the bounds below are all functions of dt; refuse to guess
        out = []

        for k, v in self.params.items():
            if k.startswith("_run.") or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            nid, _, pname = k.partition(".")
            node = self._node(nid)
            if node is None:
                continue
            spec = OPERATORS[node["op"]]["params"].get(pname)
            if spec and not (spec[0] <= v <= spec[1]):
                out.append(ThetaCondition(
                    "T4_OUTSIDE_DECLARED_BOX",
                    "theta outside the declared box (a prior, not a physical wall)",
                    f"{k}={v} not in [{spec[0]:.6g}, {spec[1]:.6g}]", False))

        for o in self.ops:
            nid, op, impl = o["id"], o["op"], self.impl_of(o)

            if op == "cell_diffuse":
                gain = DIFFUSION_STENCIL_GAIN.get(impl)
                derived = gain is not None
                if not derived:
                    gain = 1.0        # most permissive; never block on an underived number
                    out.append(ThetaCondition(
                        "T5_STENCIL_GAIN_UNDERIVED",
                        "the CFL for this diffusion implementation has not been derived from "
                        "its source, so the stability bound below is provisional",
                        f"{op}:{impl} -- evaluated at the most permissive stencil_gain=1.0", False))
                d_a, d_h = self.theta(nid, "d_a"), self.theta(nid, "d_h")
                chi = self.theta(nid, "chi")
                cfl = diffusion_cfl(d_a, d_h, chi, stencil_gain=gain)
                if cfl > DIFFUSION_CFL_LIMIT:
                    out.append(ThetaCondition(
                        "T1_DIFFUSION_UNSTABLE",
                        "explicit diffusion past its CFL limit -- chem diverges, so the run is "
                        "evidence about an integrator, not a mechanism",
                        f"dt*chi*max(d_a,d_h)*gain = {ENGINE_DT}*{chi:g}*{max(d_a, d_h):g}"
                        f"*{gain:g} = {cfl:.4g} > {DIFFUSION_CFL_LIMIT} "
                        f"(max diffusivity here is {diffusivity_ceiling(chi, stencil_gain=gain):.4g})",
                        derived))

            elif op == "cell_react":
                k = reaction_stiffness(impl, rd_rate=self.theta(nid, "rd_rate"),
                                       mu_h=self.theta(nid, "mu_h"), F=self.theta(nid, "F"),
                                       kk=self.theta(nid, "kk"), gamma=self.theta(nid, "gamma"))
                if ENGINE_DT * k > REACTION_EULER_LIMIT:
                    out.append(ThetaCondition(
                        "T2_REACTION_UNSTABLE",
                        "explicit reaction past its Euler limit -- chem diverges, so the run is "
                        "evidence about an integrator, not a mechanism",
                        f"{impl}: dt*k = {ENGINE_DT}*{k:g} = {ENGINE_DT * k:.4g} > "
                        f"{REACTION_EULER_LIMIT}", True))

            elif op == "morphogen_growth_3d":
                a_sw, alpha = self.theta(nid, "a_sw"), self.theta(nid, "alpha")
                ceil = hill_alpha_ceiling(a_sw)
                if alpha > ceil:
                    why = ("overflows float32 -> hillv == 0, growth silently stops"
                           if a_sw > 1.0 else
                           f"sinks below the operator's own {HILL_EPS:g} regulariser -> hillv == 1 "
                           f"everywhere, the switch silently becomes ALWAYS-ON and the operator "
                           f"stops being morphogen-gated while still appearing to run")
                    out.append(ThetaCondition(
                        "T3_HILL_SWITCH_DEGENERATE",
                        "the growth switch stops being a switch, silently",
                        f"a_sw**alpha = {a_sw:g}**{alpha:g} {why}; alpha ceiling here is "
                        f"{ceil:.4g}", True))
        return out

    def blocking_theta_conditions(self):
        return [c for c in self.theta_conditions() if c.blocking]

    # ---------------------------------------------------------------- D4 preconditions
    def unmet_preconditions(self):
        """[(node_id, op, missing_port)] -- operators that would silently no-op.

        The Critic rejects these for FREE, before any cluster time is spent. This is the guard
        against recording 'this mechanism cannot make tubes' when the mechanism never ran.
        """
        have = self.produced_ports()
        bad = []
        for o in self.ops:
            for need in OPERATORS[o["op"]]["needs"]:
                if need not in have:
                    bad.append((o["id"], o["op"], need))
        return bad

    def unrouted_slots(self):
        """Operators with a slot that nothing feeds -- present but disconnected, so inert."""
        fed = {(c["dst"], c["slot"]) for c in self.conns}
        out = []
        for o in self.ops:
            for slot in slots_of(o["op"], self.impl_of(o)):
                if (o["id"], slot) not in fed:
                    out.append((o["id"], o["op"], slot))
        return out

    def is_runnable(self):
        """(ok, reason). A graph must have a substrate + mechanics, no unmet precondition, no
        dangling slot, and a theta the integrator can actually carry."""
        if not REQUIRED_ROLES.issubset(self.roles()):
            return False, f"missing required role(s): {sorted(REQUIRED_ROLES - self.roles())}"
        if self.unmet_preconditions():
            return False, f"unmet precondition: {self.unmet_preconditions()}"
        if self.unrouted_slots():
            return False, f"dangling slot: {self.unrouted_slots()}"
        for c in self.conns:                       # over-routed: a slot the impl does not expose
            dn = self._node(c["dst"])
            if dn is not None and c["slot"] not in slots_of(dn["op"], self.impl_of(dn)):
                return False, (f"connection into a slot the implementation does not expose: "
                               f"{dn['op']}:{self.impl_of(dn)} has no `{c['slot']}`")
        # A theta past a DERIVED integrator limit cannot produce evidence about the mechanism, so
        # it must never reach the cluster -- the same standing as an unmet precondition (D4).
        # to_spec() calls this and refuses to compile, which is where the condition becomes loud.
        blocking = self.blocking_theta_conditions()
        if blocking:
            return False, "; ".join(c.line() for c in blocking)
        return True, "ok"

    # ---------------------------------------------------------------- one-edit API
    def _new_id(self, op):
        n = sum(1 for o in self.ops if o["op"] == op)
        return f"{op}{n}"

    def legal_edits(self, max_stage=3):
        """Every legal ONE-edit move from here, gated to <= max_stage. [(edit, label)]."""
        edits = []
        present = self.op_names()
        for stage in range(1, max_stage + 1):
            for op in STAGES[stage]:
                spec = OPERATORS[op]
                if op in present and not spec["impl_structural"]:
                    continue                                   # one copy is enough
                for impl in spec["impls"]:
                    if any(o["op"] == op and self.impl_of(o) == impl for o in self.ops):
                        continue
                    if op in present and not spec["impl_structural"]:
                        continue
                    edits.append((("add_op", op, impl), f"+{op}:{impl}"))
        for o in self.ops:                                     # removals
            role = OPERATORS[o["op"]]["role"]
            same_role = sum(1 for x in self.ops if OPERATORS[x["op"]]["role"] == role)
            if role in REQUIRED_ROLES and same_role == 1:
                continue                                       # never remove the last substrate
            edits.append((("remove_op", o["id"]), f"-{o['op']}"))
        for src, dst, slot in self._candidate_links():
            edits.append((("connect", src, dst, slot),
                          f"~{self._op_of(src)}->{self._op_of(dst)}.{slot}"))
        for c in self.conns:
            edits.append((("disconnect", c["src"], c["dst"], c["slot"]),
                          f"x{self._op_of(c['src'])}->{self._op_of(c['dst'])}.{c['slot']}"))
        # implementation swaps on structural operators are genuine mechanism edits
        for o in self.ops:
            spec = OPERATORS[o["op"]]
            if not spec["impl_structural"]:
                continue
            for impl in spec["impls"]:
                if impl != self.impl_of(o):
                    edits.append((("set_impl", o["id"], impl), f"={o['op']}:{impl}"))
        return edits

    def _candidate_links(self):
        out = []
        for s in self.ops:
            for otype in OPERATORS[s["op"]]["outputs"]:
                for d in self.ops:
                    if d["id"] == s["id"]:
                        continue
                    for slot in slots_of(d["op"], self.impl_of(d)):
                        if (otype, slot) not in LEGAL_LINKS:
                            continue
                        if any(c["src"] == s["id"] and c["dst"] == d["id"] and c["slot"] == slot
                               for c in self.conns):
                            continue
                        out.append((s["id"], d["id"], slot))
        return out

    def apply(self, edit):
        """Return (new_graph, edit) after ONE legal move."""
        g = self.copy()
        kind = edit[0]
        if kind == "add_op":
            op, impl = edit[1], edit[2]
            nid = self._new_id(op)
            g.ops.append({"id": nid, "op": op, "impl": impl})
            for pn, (lo, hi, d) in OPERATORS[op]["params"].items():
                g.params[f"{nid}.{pn}"] = d
        elif kind == "remove_op":
            nid = edit[1]
            g.ops = [o for o in g.ops if o["id"] != nid]
            g.conns = [c for c in g.conns if c["src"] != nid and c["dst"] != nid]
            g.params = {k: v for k, v in g.params.items() if not k.startswith(nid + ".")}
        elif kind == "connect":
            g.conns.append({"src": edit[1], "dst": edit[2], "slot": edit[3]})
        elif kind == "disconnect":
            g.conns = [c for c in g.conns if not (c["src"] == edit[1] and c["dst"] == edit[2]
                                                  and c["slot"] == edit[3])]
        elif kind == "set_impl":
            for o in g.ops:
                if o["id"] == edit[1]:
                    o["impl"] = edit[2]
            # an implementation with fewer slots cannot keep connections into the ones it lost
            keep = set(slots_of(g._op_of(edit[1]), edit[2]))
            g.conns = [c for c in g.conns
                       if c["dst"] != edit[1] or c["slot"] in keep]
        else:
            raise ValueError(f"unknown edit {edit!r}")
        return g, edit

    # ---------------------------------------------------------------- proximity encoding
    def encode(self):
        """Fixed-length structural feature vector, for clustering near-duplicates.

        Near-duplicate proliferation is pathology #3: thirty rounds explored perhaps four
        genuinely distinct ideas. Members of one cluster compete WITHIN the cluster so a family
        of near-identical ideas exhausts one budget, not twenty.
        """
        feat = []
        for op in OPERATORS:                                    # presence, per implementation
            spec = OPERATORS[op]
            if spec["impl_structural"]:
                for impl in spec["impls"]:
                    feat.append(float(any(o["op"] == op and self.impl_of(o) == impl
                                          for o in self.ops)))
            else:
                feat.append(float(any(o["op"] == op for o in self.ops)))
        for (otype, slot) in sorted(LEGAL_LINKS):               # routing flags
            feat.append(float(any(
                c["slot"] == slot and otype in OPERATORS[self._op_of(c["src"])]["outputs"]
                for c in self.conns)))
        return np.array(feat, np.float32)

    def distance(self, other):
        return float(np.abs(self.encode() - other.encode()).sum())

    # ---------------------------------------------------------------- post-hoc naming
    def name_region(self):
        """Label a DISCOVERED composition against the literature, never chosen a priori."""
        ops = set(self.op_names())
        impls = {o["op"]: self.impl_of(o) for o in self.ops}
        forced = "extrude" in ops
        local = "morphogen_growth_3d" in ops
        emergent = "cell_react" in ops
        driven = "cell_rd_seed" in ops
        mono = impls.get("shape_energy_3d") == "monolayer"

        if not (local or "vesicle_growth" in ops):
            return "mechanics-only (no growth)"
        if forced and driven and not emergent:
            return "driven + forced (round-33 recipe)"
        if forced and emergent:
            return "emergent RD + forced extrusion"
        if local and mono and not forced:
            return "growth-driven monolayer (Okuda route)"
        if local and not forced and emergent:
            return "growth-driven emergent (target mechanism)"
        if local and not forced:
            return "growth-driven mid-surface"
        if "vesicle_growth" in ops and not local:
            return "uniform inflation (no patterning)"
        return "unnamed"


# ============================================================================ seeds
def seed(kind="substrate"):
    """The seed the campaign grows from: a relaxing vesicle that does nothing."""
    if kind == "substrate":
        return CompositionGraph(ops=[
            {"id": "seed_mesh_3d0", "op": "seed_mesh_3d", "impl": "fibonacci_sphere"},
            {"id": "shape_energy_3d0", "op": "shape_energy_3d", "impl": "default"},
            {"id": "reconnect_t1_3d0", "op": "reconnect_t1_3d", "impl": "length_threshold"},
        ])
    if kind == "empty":
        return CompositionGraph()
    raise ValueError(kind)


def reference_recipes():
    """The two compositions the campaign must reproduce and discriminate.

    `round40_mc8` is our best hand-found tube -- DRIVEN activation + FORCED extrusion.
    `okuda_route` is the target: local growth on a monolayer, no forcing term at all.
    Both are constructed by legal edits from the seed, so they live in the searched space.
    """
    out = {}

    g = seed("substrate")
    for op, impl in [("cell_geometry_3d", "scatter_add"), ("cell_adjacency", "shared_edge"),
                     ("cell_rd_seed", "tip"), ("morphogen_growth_3d", "hill_conserve_amount"),
                     ("divide_3d", "orient_iface"), ("extrude", "radial_push")]:
        g, _ = g.apply(("add_op", op, impl))
    src = next(o["id"] for o in g.ops if o["op"] == "cell_rd_seed")
    g, _ = g.apply(("connect", src, next(o["id"] for o in g.ops
                                         if o["op"] == "morphogen_growth_3d"), "gate"))
    g, _ = g.apply(("connect", src, next(o["id"] for o in g.ops if o["op"] == "extrude"), "site"))
    g, _ = g.apply(("connect", src, next(o["id"] for o in g.ops if o["op"] == "divide_3d"), "axis"))
    out["round40_mc8"] = g

    h = seed("substrate")
    h, _ = h.apply(("set_impl", "shape_energy_3d0", "monolayer"))
    for op, impl in [("cell_geometry_3d", "scatter_add"), ("cell_adjacency", "shared_edge"),
                     ("cell_react", "gierer_meinhardt"), ("cell_diffuse", "graph_laplacian"),
                     ("morphogen_growth_3d", "hill_conserve_amount"), ("divide_3d", "hertwig")]:
        h, _ = h.apply(("add_op", op, impl))
    rsrc = next(o["id"] for o in h.ops if o["op"] == "cell_react")
    h, _ = h.apply(("connect", rsrc, next(o["id"] for o in h.ops
                                          if o["op"] == "morphogen_growth_3d"), "gate"))
    # NO morphogen -> divide_3d.axis here: `hertwig` splits on the cell's OWN longest axis and
    # exposes no `axis` slot. The earlier version made that connection anyway; it compiled and
    # was SILENTLY IGNORED. Caught by the consolidated Critic rule R4_SLOT_NOT_ON_IMPL.
    out["okuda_route"] = h

    # the degenerate control the search must visit on its way: uniform inflation, no patterning.
    # It is the "grows but cannot subdivide" corner -- where impossibility results come from.
    u = seed("substrate")
    for op, impl in [("vesicle_growth", "uniform_ramp"), ("divide_3d", "hertwig")]:
        u, _ = u.apply(("add_op", op, impl))
    out["uniform_inflation"] = u
    return out


# ============================================================================ smoke test
if __name__ == "__main__":
    from run_record import comp_hash

    g = seed("substrate")
    print(f"seed: {'+'.join(g.op_names())}\n  hash={comp_hash(g)}  region={g.name_region()!r}")
    ok, why = g.is_runnable()
    print(f"  runnable={ok} ({why})")

    # --- the identity rule: theta must NOT change identity ---------------------------------
    rng = np.random.default_rng(0)
    assert comp_hash(g.with_params(g.sample_params(rng))) == comp_hash(g), \
        "theta must not change composition identity"
    print("\n[OK] theta does not change identity -- a retune cannot pose as a new hypothesis")

    # --- an implementation swap IS a mechanism edit ------------------------------------------
    g_mono, _ = g.apply(("set_impl", "shape_energy_3d0", "monolayer"))
    assert comp_hash(g_mono) != comp_hash(g)
    print("[OK] shape_energy_3d default->monolayer changes identity (mid-surface vs true 3D volume)")

    # --- D4: preconditions are caught for FREE, before any cluster time ----------------------
    bad, _ = g.apply(("add_op", "cell_diffuse", "graph_laplacian"))
    print(f"\n[D4] cell_diffuse without cell_adjacency -> unmet={bad.unmet_preconditions()}")
    assert not bad.is_runnable()[0]
    fixed, _ = bad.apply(("add_op", "cell_adjacency", "shared_edge"))
    assert fixed.is_runnable()[0]
    print("[D4] + cell_adjacency -> runnable. This is the false-impossibility guard.")

    # --- the two reference recipes ------------------------------------------------------------
    print("\nreference recipes (both reachable by legal edits from the seed):")
    refs = reference_recipes()
    for name, r in refs.items():
        ok, why = r.is_runnable()
        print(f"  {name:14} {comp_hash(r)}  {r.name_region():34} runnable={ok}")
        if not ok:
            print(f"      -> {why}")
    d = refs["round40_mc8"].distance(refs["okuda_route"])
    print(f"  structural distance between them = {d:.0f}  (proximity clustering keeps them apart)")

    # --- the campaign's central ablation is a legal one-edit move ----------------------------
    r40 = refs["round40_mc8"]
    ex = next(o["id"] for o in r40.ops if o["op"] == "extrude")
    ablated, _ = r40.apply(("remove_op", ex))
    print(f"\n[central test] ablate `extrude` -> {ablated.name_region()!r}  "
          f"hash {comp_hash(r40)} -> {comp_hash(ablated)}")
    print("  Round 41 by hand; here it is one automatic necessity test on every composition.")

    n_edits = len(g.legal_edits(3))
    print(f"\nlegal one-edit moves from the seed (stage<=3): {n_edits}")

    # --- THE SILENT CAP: the sampler must not clip -------------------------------------------
    print("\n" + "-" * 88)
    print("SILENT CAP -- sample_params no longer clips")
    print("-" * 88)
    ok_ref = refs["okuda_route"]
    r = np.random.default_rng(0)
    p, cond = ok_ref.sample_params_report(r)
    onbound = []
    for k, v in p.items():
        nid, _, pn = k.partition(".")
        op = ok_ref._node(nid)["op"]
        lo, hi, _d = OPERATORS[op]["params"][pn]
        if basin_sigma(op, pn) <= 0:
            continue          # held at its default on purpose (a sentinel), not clipped
        if abs(v - lo) < 1e-12 or abs(v - hi) < 1e-12:
            onbound.append(k)
    print(f"  seed-0 draw: {len(onbound)} of {len(p)} perturbed values sit exactly on a bound "
          f"(was 4/31 under the clip)  {onbound}")
    assert not onbound, "a clipped draw is a value the sampler did not draw"
    print(f"  excursions reported instead of hidden: {cond if cond else 'none'}")

    # --- Okuda's three published values must be REACHABLE --------------------------------------
    print("\n" + "-" * 88)
    print("REACHABILITY -- Okuda's published values are inside the boxes")
    print("-" * 88)
    for op, pn, val, what in [("morphogen_growth_3d", "alpha", 10.0, "growth-switch sharpness"),
                              ("cell_diffuse", "d_h", 10.0, "inhibitor spread"),
                              ("cell_react", "rd_rate", 0.01, "chemistry speed, bottom decade"),
                              ("cell_react", "rd_rate", 100.0, "chemistry speed, top decade")]:
        lo, hi, d = OPERATORS[op]["params"][pn]
        assert lo <= val <= hi, f"{op}.{pn}={val} still unreachable"
        print(f"  {op}.{pn:8} = {val:<8g} in [{lo:g}, {hi:g}]  default still {d:g}   [{what}]")

    # --- the bound that REMAINS is derived, and breaching it is loud ---------------------------
    print("\n" + "-" * 88)
    print("DERIVED BOUND -- explicit diffusion, computed not hard-coded")
    print("-" * 88)
    dif = next(o["id"] for o in ok_ref.ops if o["op"] == "cell_diffuse")
    print(f"  at chi={CHI_DEFAULT:g}, dt={ENGINE_DT:g}: max diffusivity = "
          f"{diffusivity_ceiling(CHI_DEFAULT):g}   (1 / (dt*chi))")
    stable = ok_ref.with_params({**ok_ref.default_params(),
                                 f"{dif}.d_h": 10.0, f"{dif}.chi": CHI_DEFAULT}, quiet=True)
    print(f"  d_h=10 chi=4  -> CFL={diffusion_cfl(0.02, 10.0, 4.0):.3g}  runnable={stable.is_runnable()[0]}")
    blown = ok_ref.with_params({**ok_ref.default_params(),
                                f"{dif}.d_h": 10.0, f"{dif}.chi": 20.0}, quiet=True)
    print(f"  d_h=10 chi=20 -> CFL={diffusion_cfl(0.02, 10.0, 20.0):.3g}")
    for c in blown.theta_conditions():
        print("   ", c.line())
    assert not blown.is_runnable()[0], "an unintegrable theta must not reach the cluster"

    # --- the coupling ablation is now a legal ONE-EDIT move ------------------------------------
    print("\n" + "-" * 88)
    print("ONE-EDIT SWAP -- the loop can ablate the coupling by itself")
    print("-" * 88)
    labels = [l for _, l in ok_ref.legal_edits(3)]
    assert "=cell_diffuse:interface_weighted" in labels
    swapped, _ = ok_ref.apply(("set_impl", dif, "interface_weighted"))
    print(f"  '=cell_diffuse:interface_weighted' in legal_edits: True")
    print(f"  hash {comp_hash(ok_ref)} -> {comp_hash(swapped)}  (the swap IS a new hypothesis)")
    assert comp_hash(swapped) != comp_hash(ok_ref)
    for c in swapped.theta_conditions():
        print("   ", c.line())

    print("\ncomposition_space OK")
