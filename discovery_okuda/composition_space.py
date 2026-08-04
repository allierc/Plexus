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
import math

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
# 1.0, matching translate.DT_GLOBAL. It was 0.02, and no run that ever produced a pattern used
# that value -- coral_fixed_ball and wk_null_s0 both carry dt 1.0. Changed together with
# DT_GLOBAL; the assertion below is what forces that.
ENGINE_DT = 1.0
# ^ MUST equal translate.DT_GLOBAL (D2: one dt for the whole campaign). It cannot be imported at
#   module scope -- translate imports THIS module -- so it is duplicated and the duplication is
#   converted into a CHECKED invariant by check_dt_agreement(), called from theta_conditions().
#   A silent disagreement would make every derived bound below wrong by the ratio of the two dts.

FLOAT32_MAX = 3.4028234663852886e38   # the state dtype the operators run in
HILL_EPS = 1e-12          # the additive regulariser in MorphogenGrowth3D.forward's Hill function
GM_MU_A = 1.0             # hard-wired by translate._emit_react (gm_rho=1.0, mu_a=1.0)
BRUSSELATOR_B = 3.0       # hard-wired by translate._emit_react (A=1.0, B=3.0)

DIFFUSION_CFL_LIMIT = 1.0     # see diffusion_cfl. The ONLY validated limit in this module.


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
    """The explicit-diffusion CFL number. Stable iff <= DIFFUSION_CFL_LIMIT.

    THE dt CANCELS, and leaving it in made this bound 50x too permissive.

    `translate` emits chi scaled by RD_PER_FRAME = 1/dt -- that is the D5a clock fix, which
    exists because cell_react and cell_diffuse EMIT=velocity into `chem` and were therefore
    being integrated on the MECHANICS substep. So the engine receives chi/dt, and the step it
    actually takes is dt * (chi/dt) * d = chi * d. The dt divides out.

    This function multiplied by dt anyway, against the UNSCALED chi, and so reported 0.056 for
    a composition whose real number is 2.8. Every ceiling derived from it -- the d_a/d_h box, the
    chi box -- inherited the same factor of 50 and declared a reachability envelope four times
    wider than the integrator can carry. `okuda_route`, the recipe named for the target, sat at
    2.8 and passed. Its chemistry went non-finite within a hundred frames and the Biologist
    refused every run.

    Measured, not reasoned: coral_fixed_ball is a HAND-written spec at dt=1.0 with chi unscaled,
    so its true number is 1.0*1.3*0.16 = 0.208, and it runs finite. okuda_route compiles to
    chi=200, d_h=0.7 and is 2.8. The two agree with this formula and disagree with the old one.
    """
    return float(chi) * float(stencil_gain) * max(float(d_a), float(d_h))


def diffusivity_ceiling(chi, dt=None, stencil_gain=1.0):
    """Largest d_a/d_h an explicit step can carry at this chi. Computed, never hard-coded."""
    return DIFFUSION_CFL_LIMIT / (float(chi) * float(stencil_gain))


def chi_ceiling(d, dt=None, stencil_gain=1.0):
    """Largest chi an explicit step can carry at this diffusivity."""
    return DIFFUSION_CFL_LIMIT / (float(d) * float(stencil_gain))


# ---------------------------------------------------------------- explicit reaction (cell_react)
# THERE IS NO REACTION BOUND HERE, AND THAT IS A MEASURED RESULT, NOT AN OVERSIGHT.
#
# The obvious move is to reuse the diffusion argument on the stiffest LINEAR term of each
# kinetics (|1 - dt*k| <= 1  =>  dt*k <= 2, with k = rate*max(mu_a,mu_h) for Gierer-Meinhardt and
# rate*(F+kk) for Gray-Scott). It was written, and then it was RUN against the real operator
# classes, and it does not hold:
#
#   Gierer-Meinhardt, rate=50  -> dt*k = 1.0, well inside the bound, and |chem| reaches 1.9e12.
#     Re-integrating the SAME reaction time at 1/50 the step (rate=1, 15000 steps) stays at 0.68,
#     so it is not a step-size failure at all: unsaturated GM (sat=0, the a^2/h autocatalysis) has
#     finite-time blow-up in the ODE, and `rate` merely buys more reaction time per frame.
#   Gray-Scott -> bounded at dt*k = 0.94, NaN by dt*k = 1.87, i.e. it diverges INSIDE the bound;
#     (F+kk) is not the largest Jacobian eigenvalue once the u*a^2 term is included.
#
# So dt*k <= 2 predicts neither, and shipping it would have put a confident, wrong, BLOCKING
# number in the search's way -- a hand-written cap wearing a derivation. Per the rule (a bound is
# PHYSICAL AND DERIVED, or ABSENT) the reaction boxes are therefore open above: rd_rate and gamma
# have a physical lower bound of 0 (a rate cannot be negative) and NO upper bound.
#
# What this costs: rd_rate = 100 with Gierer-Meinhardt does diverge, and nothing here stops it.
# That is deliberate. The guard for a run that diverged belongs where the divergence is OBSERVED
# (critic.check_posthoc, alongside P1_INERT_OPERATOR / P2_BUFFER_SATURATED -- "not evidence"),
# not in an invisible box that also removes four decades of Okuda's published span from view.
#
# `reaction_stiffness` is kept because it is the correct linear-decay rate and is useful for
# reporting; it is deliberately NOT wired to a blocking condition.
def reaction_stiffness(impl, rd_rate=1.0, mu_h=1.0, F=0.055, kk=0.062, gamma=0.3):
    """Stiffest LINEAR decay rate of a kinetics implementation, in engine time units.

    NOT a stability criterion -- see the measurements above. Reporting only.
    """
    if impl == "gierer_meinhardt":       # da = ... - mu_a*a ;  dh = ... - mu_h*h ; scaled by rate
        return float(rd_rate) * max(GM_MU_A, float(mu_h))
    if impl == "gray_scott":             # da = ... - (F+kk)*a ; scaled by rate
        return float(rd_rate) * (float(F) + float(kk))
    if impl == "brusselator":            # da = gamma*(... - (B+1)*a ...); `rate` is NOT read
        return float(gamma) * (BRUSSELATOR_B + 1.0)
    return 0.0


# ---------------------------------------------------------------- the growth switch (Hill)
# MorphogenGrowth3D.forward:  hillv = a**alpha / (a_sw**alpha + a**alpha + HILL_EPS)
#
# ONE branch of this is statically derivable and it is the a_sw > 1 branch: `a_sw**alpha` depends
# only on PARAMETERS, and once it overflows float32 the denominator is inf, hillv is identically
# 0, and growth silently stops with no error. Confirmed: a_sw=6, alpha=49.5 -> 3.3e38 (finite,
# hillv fine); alpha=50 -> inf, hillv == 0 at every activator value.
#
# The a_sw < 1 branch is NOT statically derivable and no bound is asserted for it. The failure
# there is `a**alpha` sinking below the HILL_EPS regulariser, which suppresses cells that should
# activate (measured: a=0.5, a_sw=0.2, alpha=40 -> hillv=0.48 where it should be ~1; alpha=60 ->
# 8.7e-07). That depends on the ACTIVATOR MAGNITUDE, which is state, not theta. An earlier version
# of this function asserted a ceiling of ln(HILL_EPS)/ln(a_sw) = 17.2 for a_sw=0.2 and described
# it as "the switch becomes always-on"; running it showed hillv = 9.4e-15 there, i.e. the switch
# was working fine and the stated mechanism was wrong. The bound is therefore absent.
def hill_alpha_ceiling(a_sw):
    """Largest Hill exponent at which a_sw**alpha is still representable in float32.

    Returns inf for a_sw <= 1: no STATIC bound exists there (the failure mode is activator-
    dependent -- see above). Absent, rather than guessed.
    """
    a_sw = float(a_sw)
    if a_sw <= 1.0:
        return float("inf")
    return np.log(FLOAT32_MAX) / np.log(a_sw)


# ---------------------------------------------------------------- the boxes, computed
# HOW A BOX CEILING IS SET: the derived limit evaluated with the parameter's COMPANIONS AT THEIR
# DEFAULTS. The constraints are JOINT (chi x d, alpha x a_sw), so no per-parameter box can be
# sufficient on its own -- a value inside the box can still be unstable in combination. That is
# what the joint conditions in theta_conditions() are for. The box is the reachability envelope;
# the joint condition is the wall.
# The proven point: coral_fixed_ball's chemistry, which runs finite and patterns.
# Chosen because it is MEASURED to work, not because it is inside a box.
CHI_DEFAULT, D_A_DEFAULT, D_H_DEFAULT = 1.3, 0.08, 0.16
# F_DEFAULT IS THE CALIBRATION (finding F012), not a guess. 0.046 with kk 0.062 gives three
# STABLE spots on a 2000-cell ball, reproducible across three seeds -- stability tested by
# comparing the count at step 1500 and step 3000, because a pattern still coarsening is not a spot
# pattern whatever it looks like at the moment you stop. The previous 0.055 sits in Gray-Scott's
# LABYRINTH regime and coarsens to a single bicontinuous domain at 53% coverage at any run length
# (F011) -- which is what this campaign has unknowingly been running all along.
MU_H_DEFAULT, F_DEFAULT, KK_DEFAULT = 1.0, 0.046, 0.062
A_SW_MIN, A_SW_MAX, A_SW_DEFAULT = 0.2, 6.0, 1.5

# MEASURED CEILINGS, from the 2026-08-01 certification sweep on L4 (cfl_certify + 14 cluster
# runs at 300 frames). The divergence wall sits below the formula's 1.0 -- points at CFL 0.50 and
# 0.65 went non-finite -- so the limit is set where the engine was actually observed to hold.
# SET BELOW THE FIRST OBSERVED FAILURE, and deliberately conservative. The measured points do
# not admit a single clean threshold: 0.48 and 0.55 were stable, 0.50 was not. The 0.50 failure
# is the d_h = 10 case, so something beyond the product matters -- large d_h fails at a CFL that
# small d_h survives. Until that term is derived, the bound sits under the lowest failure and
# refuses two points known to be fine. Refusing a good composition costs a proposal; admitting a
# divergent one costs a batch and, twice already, a conclusion.
DIFFUSION_CFL_LIMIT = 0.45
CHI_PATTERN_CEIL = 3.0          # measured: patterns at chi <= 3, finite-but-dead at 4 and 6

D_CEIL = diffusivity_ceiling(CHI_DEFAULT)          # 12.5   -- reaches Okuda's d_h = 10
CHI_CEIL = chi_ceiling(D_H_DEFAULT)                # 71.4
RD_RATE_CEIL = float("inf")     # ABSENT: no reaction bound survived measurement (see above)
GAMMA_CEIL = float("inf")       # ABSENT: same
# alpha is the one place the default-companion rule is deliberately not used: the ceiling FALLS as
# a_sw rises, so the box takes the tightest value over the admissible a_sw range rather than the
# value at a_sw's default. 49.5, which reaches Okuda's alpha = 10 with room to spare.
ALPHA_CEIL = hill_alpha_ceiling(A_SW_MAX)          # 49.5

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
        params={"n_cells": (150, 4000, 500),          # 4000 = Okuda's largest case (grounder.SETUP)
                "vseed_cv": (0.0, 0.5, 0.15)}),
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
        # searchable space. Now ALPHA_CEIL, derived from the float32 overflow of a_sw**alpha in
        # the Hill function (see hill_alpha_ceiling). lo = 1.0 stays: below 1 the Hill switch has
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
    # THE MISSING ARROW. Chemistry patterns the shell and the shell deforms, but the shape it takes
    # never reaches back -- half of Okuda's loop, and the mechanism behind every branching
    # morphology, was simply absent from the space. FOUR IMPLEMENTATIONS, structural, because WHICH
    # shape feature the chemistry listens to is the hypothesis and not a number: curvature-sensing
    # and tension-sensing are different biology and predict different things. impl_structural=True
    # makes `=shape_to_chem:tension` a legal one-edit move, so the loop can run that comparison
    # itself rather than waiting for a human to hand-write four configs.
    # `force` and `size` are deliberately NOT implementations -- see the operator's docstring.
    "shape_to_chem": dict(
        stage=3, role="patterning", outputs=[], slots=[], needs=["adjacency"],
        impls=["curvature", "tension", "apical_area", "pressure"], impl_structural=True,
        # beta spans BOTH SIGNS and includes zero. Zero is the null and must stay reachable: without
        # it, "shape feeds back" is asserted rather than tested. The sign is a real hypothesis --
        # do deformed cells signal more, or less? -- so a one-sided box would silently answer it.
        params={"beta": (-3.0, 3.0, 0.0), "F0": (0.0, 0.12, 0.055)}),
    "cell_react": dict(
        stage=3, role="patterning", outputs=["morphogen"], slots=[], needs=["adjacency"],
        impls=["gierer_meinhardt", "gray_scott", "brusselator"], impl_structural=True,
        # rd_rate: was (0.2, 3.0) -- a factor of 15, where Okuda spans FOUR DECADES. lo = 0.0 is
        # the physical bound (a rate cannot be negative); the upper bound is ABSENT because the
        # candidate derivation was measured and did not hold (see "explicit reaction" above).
        # An absent bound is the honest state; 3.0 was a hand-written one that hid three decades.
        # gamma (brusselator): was (0.1, 100.0), equally hand-written. Also open above.
        params={"gamma": (0.0, GAMMA_CEIL, 0.3), "a0": (0.0, 0.05, 0.01),
                "rd_rate": (0.0, RD_RATE_CEIL, 1.0),
                "F": (0.02, 0.06, F_DEFAULT), "kk": (0.05, 0.07, KK_DEFAULT),
                "mu_h": (0.2, 2.0, MU_H_DEFAULT)}),
    "cell_rd_seed": dict(                                     # the prescribed activation driver
        stage=3, role="driver", outputs=["morphogen"], slots=[], needs=[],
        # `scatter` added 2026-07-31: it is the ONLY Gray-Scott seeding validated on this
        # substrate -- the minisite coral movie uses mode="scatter", seed_frac=0.06, and that is
        # the configuration measured to give a live pattern (act_max 0.43). The search space
        # exposed only cones and tip, so the one seeding known to work could not be expressed.
        # Scattered seeds are also the physically natural initial condition for a Turing system:
        # a pattern should emerge from noise, not from foci we placed by hand.
        impls=["tip", "cone", "spot", "scatter"], impl_structural=True,
        params={"tip_radius": (0.6, 3.0, 2.0), "cone_deg": (4.0, 30.0, 8.0),
                "seed_frac": (0.01, 0.30, 0.06),
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
    ("cell_react", "gamma"): 100.0 - 0.1,               # old box (0.1, 100.0). Inherited, not
    #   endorsed -- the box above it is now open, so a basin MUST be stated explicitly, and
    #   restating the old one is the only choice that changes nothing else.
    ("divide_3d", "max_cycle"): 0.0,                    # 1e9 is a SENTINEL for "no maximum", not a
    #   tunable. The old rule gave it sigma = 1.5e8 and the clip then pinned half the draws back
    #   onto 1e9 -- noise that looked like sampling. Fixed = never perturbed.
}


def basin_sigma(op, pname, scale=0.15):
    """Sampling width for one parameter. Explicit if listed, else the old scale*(hi-lo) rule."""
    if (op, pname) in PARAM_BASIN:
        return float(PARAM_BASIN[(op, pname)]) * float(scale)
    lo, hi, _ = OPERATORS[op]["params"][pname]
    w = float(hi) - float(lo)
    if not np.isfinite(w):
        # An OPEN box (a bound we could not derive, so did not invent) carries no implied basin.
        # Falling back to scale*(hi-lo) would silently hand the sampler sigma=inf.
        raise ValueError(
            f"{op}.{pname} has an open box ({lo}, {hi}) and no PARAM_BASIN entry. The admissible "
            f"region and the sampling basin are separate; state the basin explicitly.")
    return float(scale) * w


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
# T2 is deliberately absent: the reaction-stability rule that would have gone here was written,
# measured against the real operators, refuted, and removed rather than shipped. The gap in the
# numbering is the record of that.
THETA_RULES = ["T1_DIFFUSION_UNSTABLE", "T3_HILL_SWITCH_DEGENERATE",
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
                # THE SECOND CLAUSE, and it is not in any formula. Certified on the cluster
                # 2026-08-01, 300 frames, 14 points around the predicted wall: the chemistry
                # survives to CFL 0.55 at chi = 1.3 and dies at 0.65, but at chi = 4 and chi = 6
                # the run stays perfectly FINITE while the activator goes to zero. A CFL bound
                # cannot see that -- it forbids divergence, and this is extinction. Two failure
                # modes, one of which the Biologist catches after the fact (P4) and the other
                # before. Only the second was ever checked.
                #
                # chi multiplies BOTH diffusivities, so raising it flattens the activator's own
                # gradient as fast as the inhibitor's: past ~3 the local peak cannot outrun its
                # own spreading and the pattern decays wherever it starts. Measured, not derived.
                if chi > CHI_PATTERN_CEIL:
                    out.append(ThetaCondition(
                        "T1b_PATTERN_EXTINGUISHED",
                        "the chemistry stays finite and goes DEAD -- above this chi the activator "
                        "diffuses away faster than it can autocatalyse, so the run is stable and "
                        "measures nothing",
                        f"chi = {chi:g} > {CHI_PATTERN_CEIL:g}; measured act_max 0.0 at chi 4 and "
                        f"chi 6 (both finite, both P4), against 0.40-0.55 at chi <= 3", True))
                if cfl > DIFFUSION_CFL_LIMIT:
                    out.append(ThetaCondition(
                        "T1_DIFFUSION_UNSTABLE",
                        "explicit diffusion past its CFL limit -- chem diverges, so the run is "
                        "evidence about an integrator, not a mechanism",
                        f"chi*max(d_a,d_h)*gain = {chi:g}*{max(d_a, d_h):g}"
                        f"*{gain:g} = {cfl:.4g} > {DIFFUSION_CFL_LIMIT} "
                        f"(max diffusivity here is {diffusivity_ceiling(chi, stencil_gain=gain):.4g})",
                        derived))

            # cell_react has NO stability condition on purpose: the candidate bound was measured
            # against the real operators and refuted. See "explicit reaction" above.

            elif op == "morphogen_growth_3d":
                a_sw, alpha = self.theta(nid, "a_sw"), self.theta(nid, "alpha")
                ceil = hill_alpha_ceiling(a_sw)
                if alpha > ceil:
                    out.append(ThetaCondition(
                        "T3_HILL_SWITCH_DEGENERATE",
                        "the growth switch stops being a switch, silently",
                        f"a_sw**alpha = {a_sw:g}**{alpha:g} overflows float32 -> the Hill "
                        f"denominator is inf, hillv == 0 at every activator value, and growth "
                        f"stops with no error; alpha ceiling at this a_sw is {ceil:.4g}", True))
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
                # ONE COPY IS ENOUGH -- FOR EVERY OPERATOR, structural implementations included.
                #
                # `impl_structural` was doing two jobs at once: it marks an operator whose
                # IMPLEMENTATION is a real mechanism choice (so `set_impl` is offered below), and
                # it was also being read as permission to hold TWO of them. Those are different
                # claims, and conflating them put a second whole mechanics solver into a spec.
                #
                # MEASURED on round 1: the menu offered both `+shape_energy_3d:default` (add) and
                # `=shape_energy_3d:default` (swap). The Proposer chose `add` and wrote the claim
                # for `swap` -- "swapping the monolayer shape energy for the default releases the
                # in-plane constraint" -- and the spec came out with shape_energy_3d TWICE, two
                # independent relaxation loops of 30 iterations each driving the same vertices.
                # That composition is not the one the hypothesis is about, so the run could not
                # have tested it. The previous batch had the same fault in three more slots, on
                # `shape_to_chem`.
                #
                # The way to change an implementation is `set_impl`. If a second instance is ever
                # genuinely meaningful, that needs its own explicit flag; no operator declares one.
                if op in present:
                    continue
                for impl in spec["impls"]:
                    edits.append((("add_op", op, impl), f"+{op}:{impl}"))
        for o in self.ops:                                     # removals
            role = OPERATORS[o["op"]]["role"]
            same_role = sum(1 for x in self.ops if OPERATORS[x["op"]]["role"] == role)
            if role in REQUIRED_ROLES and same_role == 1:
                continue                                       # never remove the last substrate
            edits.append((("remove_op", o["id"]), f"-{o['op']}"))
        # PARAMETER MOVES. Two points per parameter -- one below the standing value, one above --
        # taken from the operator's own declared (lo, hi, default) triple, so a sweep can never
        # leave the space the Critic will admit. Two rather than a grid because a menu must stay
        # readable: the Proposer chooses WHICH lever to lean on, and a theta round refines it.
        #
        # This is what makes "sweep the parameters" expressible at all. `--mode theta` has existed
        # since the first draft and the planner never emitted it, so the question "what does this
        # mechanism do as you turn it up" has never once been asked in a live round.
        for o in self.ops:
            # THE SPECIMEN IS NOT AN AXIS OF THE SEARCH. round.py grounds the starting conditions
            # on every slot for a reason it learned the hard way: two slots once ran 500 cells
            # against a control at 2000, so any difference between them confounded the EDIT with a
            # fourfold change in specimen size. Offering `n_cells` as a sweepable parameter would
            # hand that same confound back through the front door. Seeds likewise: they are forced
            # by the pipeline so a comparison cannot be accidentally confounded.
            if OPERATORS[o["op"]].get("role") == "substrate":
                continue
            for pname, tri in (OPERATORS[o["op"]].get("params") or {}).items():
                if pname in ("seed", "n_cells", "before_frame", "after_frame"):
                    continue
                if not (isinstance(tri, (tuple, list)) and len(tri) == 3):
                    continue
                lo, hi, dflt = (float(x) for x in tri)
                cur = float(self.params.get(f"{o['id']}.{pname}", dflt))
                # AN OPEN BOX IS NOT AN INFINITE STEP. Several ceilings are float("inf") -- an
                # honest way to say "no derived limit is known" -- and the midpoint rule turned
                # that into a literal menu entry at rd_rate = inf, which the Critic ADMITS on a
                # composition without division. Halving toward an open ceiling is meaningless;
                # doubling from the current value is the step a person would take.
                _up = cur * 2.0 if not math.isfinite(hi) else cur + (hi - cur) * 0.5
                for v in (cur - (cur - lo) * 0.5, _up):
                    v = round(v, 6)
                    if math.isfinite(v) and lo <= v <= hi and abs(v - cur) > 1e-9:
                        edits.append((("set_param", f"{o['id']}.{pname}", v),
                                      f"@{o['op']}.{pname}={v:g}"))
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
            # AN OPERATOR IS BORN WIRED, when there is only one way to wire it.
            #
            # THIS IS THE DEFECT THAT COST THE CAMPAIGN. add_op created the node and never the
            # connection, is_runnable() is false for a dangling slot, and the Proposer's menu
            # filters on is_runnable() -- so NO OPERATOR THAT DECLARES A SLOT COULD EVER BE ADDED
            # BY A ONE-EDIT MOVE, from any parent, ever. An external review proved it by breadth
            # first search: 9,760 reachable compositions, ZERO containing a single connection.
            #
            # The three slotted operators are the entire morphogen -> mechanics arrow:
            #     morphogen_growth_3d.gate   -- Okuda's own mechanism
            #     extrude.site               -- the forcing term
            #     divide_3d:orient_iface.axis
            # so the search could destroy the coupling and never build it, and six rounds of
            # "chemistry is inert for shape" were a measurement of a disconnected subsystem.
            #
            # UNIQUE SOURCE ONLY -- the same rule already used to rebuild parents from specs. One
            # candidate is not a guess, it is the only wiring the composition admits. Two or more
            # is a real choice and stays dangling, so the Critic still says R3 and a `connect`
            # edit remains available to make it deliberately.
            # LEGAL_LINKS is the authority on what may feed what -- reuse it rather than invent
            # a second rule, because a slot is named `gate` while the port it accepts is called
            # `morphogen`, and matching the two by name silently wires nothing.
            for _sl in slots_of(op, impl):
                _src = [o["id"] for o in g.ops if o["id"] != nid
                        and any((_ot, _sl) in LEGAL_LINKS
                                for _ot in (OPERATORS[o["op"]].get("outputs") or []))]
                if len(_src) == 1:
                    g.conns.append({"src": _src[0], "dst": nid, "slot": _sl})
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
        elif kind == "set_param":
            # A PARAMETER MOVE, AND IT IS NOT A MECHANISM. Track A asks two questions about a
            # mechanism -- does it matter, and what does it do as you turn it up -- and only the
            # first has ever been askable. `("set_param", "morphogen_growth_3d0.rate", 0.02)` is
            # the second, expressed as an ordinary edit so it can be chosen from the same menu and
            # MIXED with structural moves in one batch, instead of needing a whole round of its own.
            #
            # The discipline survives because comp_hash EXCLUDES params: a retune yields the SAME
            # composition identity, so it can never register as a new mechanism -- "a change of
            # numbers can never masquerade as a new idea" holds because the identity function says
            # so, not because an agent was asked to be careful.
            g.params = dict(g.params)
            g.params[edit[1]] = edit[2]
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
    #
    # THE SECOND HALF OF OKUDA'S COUPLING. Chemistry deforms the tissue and the tissue's shape
    # feeds back into the chemistry -- his flux is weighted by shared wall area and cell volume.
    # This recipe carried only the first arrow, because until Phase 2 no operator expressed the
    # second: a recipe named for his route was missing the half that makes it his. `curvature` is
    # the implementation his own framing implies; the other three are the alternative hypotheses,
    # and swapping them is a legal one-edit move rather than a dial.
    h, _ = h.apply(("add_op", "shape_to_chem", "curvature"))
    # AND IT MUST BE INTEGRABLE, or the recipe named for the target cannot be run. As written it
    # carried rd_rate = 1.0, which with divide_3d present advances dt*rate = 1.00 per step against
    # R1d's derived limit of 0.5 -- so the Critic refused the one composition in this file that
    # holds Okuda's coupling, and it could not go on the frontier. Halving the rate satisfies the
    # limit without touching the mechanism: it is the same chemistry, integrated in steps the
    # solver can carry. R1d stays exactly as it is; it was right.
    # MERGE, do not replace: with_params() overwrites the whole dict, so passing one key shipped
    # this recipe with 1 of its 22 parameters. The physics survived (the emitter falls back to the
    # declared defaults and the emitted spec is byte-identical), but a recipe named "the target"
    # carrying no explicit operating point is what made the theta hash non-canonical below.
    h = h.with_params({**h.params, "cell_react0.rd_rate": 0.4})
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


# ============================================================ REACTION STABILITY (the second bound)
# THERE WAS A CFL BOUND ON DIFFUSION AND NONE ON THE REACTION, and round 1 of the rebuilt loop died
# of the second while satisfying the first. Measured, not reasoned:
#
#   r001c_00_f4907e   chi_spec 65   act 0.01 -> 12.1 -> 1.41e6 -> NaN by frame 115, SPATIALLY
#                                   UNIFORM the whole way (max spread 3.4e-05 against a mean of
#                                   1.4e6). A diffusion instability makes a checkerboard; a
#                                   uniform blow-up is the reaction ODE exploding.
#   coral_fixed_ball  chi_spec 1.3  patterns, spatial spread 0.78, stable to the last frame.
#
# Same d_a, same d_h, same ratio. The one thing that differs is chi in the spec, and it differs by
# exactly RD_PER_FRAME.
#
# WHY IT IS 50x. `translate` emits chi * RD_PER_FRAME = chi/dt, which is the D5a clock fix -- it
# exists because cell_react EMITs a velocity and is therefore integrated on the MECHANICS substep
# rather than once per frame. But there are 1/dt substeps in a frame, so the substep clock ALREADY
# supplies the factor the fix adds. Applied on top, the reaction advances chi/dt per frame instead
# of chi. At dt=1.0 the two are equal and nothing was visible; at dt=0.02 the reaction runs 50x
# too fast and explicit Euler on an autocatalytic system does what it must.
#
# The bound is therefore on what the engine actually advances per frame, which is the SPEC value.
REACTION_PER_FRAME_LIMIT = 2.0     # 1.3 patterns and is stable; 65 explodes. Calibrated, not chosen.


def reaction_advance(chi, rd_per_frame=None):
    """How far the reaction advances in one FRAME, as the engine will actually run it.

    The engine applies the reaction once per mechanics substep and there are 1/dt substeps of
    length dt in a frame, so the substep clock alone advances `chi` per frame. Whatever
    `translate.RD_PER_FRAME` multiplies on top is pure excess -- which is why it is now 1.0.
    Stable iff <= REACTION_PER_FRAME_LIMIT.
    """
    if not chi:
        return 0.0
    if rd_per_frame is None:
        try:
            from translate import RD_PER_FRAME
            rd_per_frame = RD_PER_FRAME
        except Exception:
            rd_per_frame = 1.0
    return float(chi) * float(rd_per_frame)
