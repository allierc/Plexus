# Analysis log — Atlas: jax-morph

Append-only. Newest at the bottom. One heading per mechanism id per call.

---

## Phase 0 — instruments, before any agent ran

- Baseline frozen: **52 registered contracts** (42 canonical + 10 aliases) across 10 families.
  That set is the whole comparison — the promoted language only; unreviewed code in `prototype/`
  and `operators/candidates/` does not enter the measurement.
- Oracle built and verified: jax 0.11.0, jax-morph 0.4.0 (clone @ `ace08b8`), CPU.
- Oracle determinism checked at a fixed PRNG key: two runs of the authors' proliferation model
  are bit-identical in position and alive. A differential test against it is therefore measuring
  us, not the reference's own noise.
- First reference artefact: `_oracle/runs/smoke/` — 4 → 82 cells over 40 macro-steps,
  gyration 0.65 → 3.23, `division_overflow = 0` (no array bound was hit).
- Record seeded mechanically from the clone's AST: **24 candidate mechanisms**, every one at
  status `candidate` — named and located, nothing inspected, nothing believed.

---

## division

Read the whole `Division(StochasticStep)` class (`physics/division.py:L38-L229`), its base
`StochasticStep`/`SimulationStep` and the `Model` macro-step in `core/step.py`, the AD primitives it
calls (`sample_bernoulli_st`, `bernoulli_logp`, `safe_norm` in `core/ad_utils.py`), and the field
contract in `core/state.py`. Fixed `code_path` to the class range and set `status: inspected`.

The event itself is a plain per-cell Bernoulli hazard `p = 1 - exp(-rate*dt)`; the interesting
physics is all in `replay`. What surprised me: (1) volume conservation is dimension-dependent —
`m = 2^(-1/d)` shrinks the radius so the *d-volume* halves, so `n_space_dim` is a physics knob, not
just a shape knob; (2) the daughter offset uses the NEW radius `r*m`, giving exactly-touching
daughters; (3) the mother side is soft/differentiable (straight-through, `d/dp=1`) while the daughter
slot is a hard scatter — the two daughters are not symmetric under autodiff; (4) capacity overflow is
silently capped into a global running `division_overflow` counter (never raises); (5) `replay`
implicitly reads EVERY cell-scope field to fill the daughter (heritable → inherit, else reset to
default), though only `division_rate`/`division_axis` are declared reads.

Could NOT establish: the paper PDF would not render here (no poppler/pdftotext; text streams
compressed), so I anchored `paper_section` to the installed library guides and did NOT verify against
a specific figure/equation whether the authors' paper describes the same `2^(-1/d)` conservation and
oriented placement — that source-vs-paper check is still open. Did not run the oracle for this entry
(Phase 0 smoke already showed `division_overflow=0`, i.e. the cap was never hit at that scale). Did
not confirm whether any shipped example ever sets `division_axis` or a non-zero `orientation_snr`, so
whether oriented division is exercised in practice or is effectively dormant is unverified. Verdict
(alias/refinement/new) left null for the normalizer.
