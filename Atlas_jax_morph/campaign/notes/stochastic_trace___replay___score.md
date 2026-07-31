<!-- stochastic trace / replay / score -- append below; the driver merges this into campaign/analysis.md -->

# stochastic trace / replay / score (core/step.py:148 + simulate.py + ad_utils.py)

Read: `StochasticStep` (step.py:148) and its base `SimulationStep`; the driver side --
`Model._reset_traces`, `_run`, `_replay_transition` (step.py:550), `_resolve_score`; the top-level
scorers `trajectory_logp` / `transition_logp` (simulate.py:92,137); the density kernels
`bernoulli_logp` / `gaussian_logp` / `sample_bernoulli_st` (ad_utils.py); and two concrete
subclasses spanning both estimator regimes -- `Division` (discrete, score-only) and
`BrownianDynamics` / `ActiveBrownianDynamics2D` (dynamic, reparameterized = pathwise AND scorable).
The stale `code_path` (`core/logp.py:L1`) does not exist; fixed to step.py:148.

The contract as an update-to-state: forward = `sample_trace` then `replay(pathwise=True)`; scoring
= read the recorded trace back out of the post-step state, then per selected step add
`logp(s_live, trace)` while replaying `pathwise=False`. The model sets `pathwise = not scored`, so
the SAME trace drives either estimator but never both for one choice. `trajectory_logp` returns
one term per macro-step (shape (T,)) = the paper's per-step `log pi(a_t|s_t)`; the return-weight
`G_t` and baseline are deliberately left to user code (no trainer in the library).

Surprised by: (1) the co-emission trap -- a `replay` that forgets a trace field scores garbage
silently, caught only by `check_stochastic_step`'s two-sentinel round-trip. (2) `sample_bernoulli_st`
is forward-EXACT (true draw, temperature-independent) with an identity backward surrogate. (3)
scoring masks by the recorded `divide_eligible` (alive at decision time), not current `alive`.
(4) division uses the competing-risks hazard `p = 1 - exp(-lambda dt)`, not `lambda dt`.

PAPER-vs-CODE (recorded in `surprises:`): the paper formalizes ONLY the score-function/REINFORCE
half for the discrete division event and backprops pathwise only where there is no division; the
code unifies both into one trace/replay/logp contract that also admits reparameterized stochastic
steps. Source wins.

OVERLAP FLAG for the normalizer: this entry and the sibling `stochastic_step` entry cover the SAME
`StochasticStep` mixin from two angles (this one emphasizes the trace/replay/logp PROTOCOL and the
scoring DRIVERS; the other the mixin CLASS). They may collapse to one contract -- see
`stochastic_step.md`. See also `division.md`, `brownian_dynamics.md`, `active_brownian_dynamics2_d.md`.

NOT established: I did NOT run the oracle or exercise `trajectory_logp`/`transition_logp` end to
end -- the round-trip and gradient claims are read from source + docstrings, not observed. I did
not enumerate every StochasticStep subclass (checked Division + the two Brownian steps), so a
subclass with a bespoke `trace_from_state` or an unusual `_dist` routing is unverified beyond the
documented escape hatch. Left verdict/contract null for the normalizer.

--- re-inspection addendum ---
Sharpened one point for `equations`/`surprises`: the SI's clean `grad log P(tau) -> grad log
pi_theta(a|s)` simplification is valid only because it treats the transition kernel `P(s'|s,a)` as
theta-independent. The CODE does NOT: `trajectory_logp` keeps the reconstructed state carry LIVE
(never detached in `_replay_transition`), so the environment's theta-dependence (secretion ->
diffusion -> future division propensity) flows PATHWISE, and reparameterized transitions get their
own Gaussian `logp`. So what the code differentiates is a SUM of a discrete score term and a
pathwise environment term -- broader than the SI's policy-only score. The SI (p. 16) concedes this
("propagated through the environment updates till the very beginning"). Prior working copy was
reverted for a YAML parse error (unquoted colon in prose); this pass moved every colon-bearing
prose value into block scalars. `status: inspected`; evidence still null (no oracle run).

--- normalizer ---
VERDICT: `out_of_scope` (contract stub `sample_score`, kept identical to the sibling `stochastic_step`
entry, which is the SAME core/step.py:148 class and was already normalized out_of_scope with an
explicit directive that this entry match). `StochasticStep` is an abstract differentiability mixin:
base `state_writes()` and `trace_writes()` both default to `()`, it writes no physical state and adds
exactly one non-physical knob (`score_by_default`); everything it fixes is gradient-estimator plumbing
(sample parameter-free noise, replay, score `logp`; the model's `pathwise = not scored` inversion).
In Plexus differentiability is an engine concern, not a forward operator over sets and fields, and the
forward biology already lives in the concrete subclasses (`Division` = `cell_divide`, `BrownianDynamics`
= motion), each its own entry -- so counting this as `new` would inflate the yield with framework
machinery, exactly what out_of_scope guards against.

STRONGEST ARGUMENT AGAINST: the ATLAS exists to find MISSING vocabulary, and one could argue the
sample/record/replay/score protocol is itself a genuinely new STRUCTURAL contract the promoted algebra
lacks -- a first-class "stochastic step" obligation (declare an ephemeral trace, own a `replay`, expose
`logp`) that any operator emitting a sampled action must satisfy. If Plexus cannot express how a
stochastic operator records and scores its own noise, that is arguably a real gap in its capacity for
differentiable stochastic morphogenesis, and calling it out_of_scope discards a contract as "mere
plumbing" that the language may in fact need. I reject this because the protocol has NO forward effect
of its own to normalize -- strip the subclass biology and only the gradient PATH remains (which
estimator, how the trace round-trips, where `stop_gradient` falls), and the typed signature it would
register is empty (no set, no maps, reads nothing, writes only the ephemeral `trace`). An operator with
an empty forward signature is the tell that it is autodiff bookkeeping, not an operation on state; the
biological content is fully accounted for by `cell_divide` and the motion contracts, with nothing left
over for the vocabulary to gain. `status: normalized`; evidence still null (no oracle run).
