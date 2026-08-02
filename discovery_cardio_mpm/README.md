# discovery_cardio_mpm

The third agentic loop, and the first pointed at a real measurement rather than a paper: a
differentiable MPM model of a beating cardiomyocyte sheet, fitted to microscope tracking data.

**Read `cardio_note.pdf` first.** It is the reader-facing document, organised by phase, and phases
are where work stops and Cedric decides. Build it with `pdflatex cardio_note.tex` twice.

## The standing rule

**No conclusion of the previous campaign (`prototype/cardio_mpm`, 60 batches) is inherited.** Its
claims are transcribed into `HYPOTHESES.md` as open questions marked *untested*; `BELIEFS.md` starts
empty. Apparatus may be reused only after it re-passes a gate here. Defaults count as beliefs.

## The loop, in four steps

    propose  ->  gate  ->  measure (on data it never saw)  ->  interpret
       ^                                                            |
       +-------- the part of the residual nothing explained --------+

## Status

**Phase 0 CLOSED (2026-08-02).** The gate passes 18/18, the canaries catch 6/6.

    PYTHONPATH=/workspace/Plexus/src python certify_apparatus.py            # the gate
    PYTHONPATH=/workspace/Plexus/src python certify_apparatus.py --canary   # break it 6 ways
    PYTHONPATH=/workspace/Plexus/src python certify_apparatus.py --fit      # + real fits (slow)

Phase 1 (freeze the recording, seal the diseased specimen) is next and has not started.

**Recorded, not certified:** determinism is complete on CPU and partial on GPU —
`grid_sampler_2d_backward_cuda` has no deterministic implementation and is reached from
`plexus.models.base.Field.sample` via `active_stress`. The same-seed spread is measured
(`_metrology/gpu_repeat.json`, 3.0e-6 at 2 iterations) rather than assumed to be zero.
`--allow_nondeterministic_ops` is off by default.

The Plexus operator path is still not differentiable end-to-end (measured — see the note §2 and
Phase 3); the trainer hand-rolls its own step, as the inherited one did.

## Related

- `../discovery_okuda/` — the Okuda loop this forks in spirit; `ROLES.md` there is the roster discipline.
- `../atlas_jax_morph/`, `../atlas_cc3d/` — the sibling atlases; same note template and phase ladder.
- `../prototype/cardio_mpm/` — the previous campaign. Apparatus and defect patterns only.
