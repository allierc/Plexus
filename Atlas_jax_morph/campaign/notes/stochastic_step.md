<!-- StochasticStep -- append below; the driver merges this into campaign/analysis.md -->

# StochasticStep (core/step.py:L148)

Read the whole `StochasticStep` mixin, its base `SimulationStep`, the `Model` machinery that
drives it (`_reset_traces`, `_run`, `__call__`, `_replay_transition`, `_resolve_score`), the
`check_stochastic_step` round-trip guard, and two concrete subclasses (`physics/division.py`
`Division`, and the `MaybeDivide`/`Kick` toy steps in `guides/extending.md`). Paper anchors:
Methods p. 10 "Gradient calculation" (the `sum_t L_t * grad log pi(a_t|s_t)` REINFORCE estimator)
and the appendix policy-gradient derivation pp. 16-18.

This is an ABSTRACT mixin, not a physics step: it writes nothing itself, it fixes a contract
(trace_writes / _dist / sample_trace / replay / logp; `__call__` is DERIVED = sample_trace then
replay(pathwise=True)). The "update to the state" is that composition plus the scoring path.

Surprises worth flagging: (1) `replay` must co-emit every trace field or the whole record/score
round-trip silently produces garbage -- `check_stochastic_step` is the only guard, and it needs
TWO sentinel pre-fills because a reset-to-default single run can't distinguish "wrote it" from
"forgot it". (2) Reparameterizability is NOT a knob -- it is intrinsic to `replay` and selected by
the model via `pathwise = not scored`; `score_by_default` is only a scoring-inclusion flag. (3)
Trace fields are ephemeral (reset each macro-step), validated to not shadow real fields and to
default 0 when dynamic/additive.

PAPER-vs-CODE contradiction (recorded in `surprises:`): the paper presents ONLY the discrete
score-function/REINFORCE estimator ("stochastic operations have no differentiation rule") and a
uniformly-random division plane; the code additionally supports a reparameterized PATHWISE branch
and oriented placement -- a lower-variance gradient path the paper never describes. Source wins.

NOT established: I did not run the oracle or exercise `trajectory_logp`/`transition_logp` end to
end, so the claim that the forward record round-trips through scoring is read from the source and
`check_stochastic_step`'s docstring, not observed. I also did not enumerate every subclass in the
repo (checked Division + the two guide toys), so whether some step overrides `trace_from_state` or
routes `_dist` unusually is unverified beyond the documented "bespoke layout" escape hatch. Left
verdict/contract null for the normalizer.

--- normalizer pass ---
VERDICT: out_of_scope. StochasticStep is jax-morph's "hardened core" differentiability contract
(AGENTS.md: "stochastic trace replay/scoring", on which "physics and control steps compose"): an
abstract mixin whose own state_writes() and trace_writes() both default to (), carrying one
non-physical knob (score_by_default). Its entire content is the gradient-estimator machinery --
sample exogenous noise, replay pathwise, score logp -- REINFORCE for discrete events plus a
reparameterized pathwise branch for continuous ones. That is autodiff/optimization plumbing; in
Plexus differentiability is an engine concern (torch autograd), not an operator in the forward
algebra. The biology lives entirely in the subclasses (Division->cell_divide, Death->apoptosis,
Brownian->motion), each already its own atlas entry. No registered contract models gradient
estimation, and widening one to admit trace/replay/score would graft an optimization concern onto
a forward-dynamics signature -- so not new, not alias/refinement. The contract block is
schema-satisfaction only (R6/R7 force a typed contract at status=normalized); the ledger scores
this entry by its out_of_scope verdict and ignores those fields. Flagged the OVERLAP: the sibling
entry stochastic_trace___replay___score (order 24) is the SAME step.py:148 class and should get
the same verdict.

STRONGEST ARGUMENT AGAINST (the skeptic's line -- "find the biology hiding in the plumbing"): one
could argue the trace/replay/score contract is itself a genuinely NEW capability the promoted
language lacks -- Plexus has no notion of a differentiable stochastic step, and if the atlas is
meant to measure whether the vocabulary can express what the target does, then "an operator whose
effect is a scored random draw" is a real gap, arguably a `new` contract (call it `sample` or
`stochastic_event`) rather than out_of_scope. I reject this because it conflates two layers: the
FORWARD effect of every stochastic step already has a biological home in its subclass's own
contract (a Bernoulli division IS cell_divide), and what the mixin adds on top is purely the
gradient path -- REINFORCE vs pathwise -- which is an optimization/AD concern, not a forward
operation on state. Counting it `new` would inflate the operator yield with framework machinery,
exactly the failure the loop's out_of_scope verdict exists to prevent. The one thing that would
genuinely reopen this: if Plexus's engine could NOT recover the same optimization behaviour
through torch autograd on the subclass operators (i.e. if trace/replay/score encoded a forward
semantic the subclass contracts drop), then it would be a real gap -- I did not run that check
(jax is walled off here), so it stays an open question for the verifier, not a blocker on the
verdict.
