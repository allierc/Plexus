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
