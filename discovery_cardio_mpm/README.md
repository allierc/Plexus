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

Phase 0 not started. Nothing runs yet: the inherited trainer crashes on every invocation, and the
Plexus operator path is not differentiable (measured — see the note, §2 and Phase 3).

## Related

- `../discovery_okuda/` — the Okuda loop this forks in spirit; `ROLES.md` there is the roster discipline.
- `../atlas_jax_morph/`, `../atlas_cc3d/` — the sibling atlases; same note template and phase ladder.
- `../prototype/cardio_mpm/` — the previous campaign. Apparatus and defect patterns only.
