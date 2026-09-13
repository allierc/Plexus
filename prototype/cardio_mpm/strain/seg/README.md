# seg -- the five modules that turn the beat into an instance segmentation

Copied verbatim from `prototype/cardio_cells` when that folder was deleted (2026-09-13). They are
what `../segmentation.py` calls, and therefore the only part of that campaign the strain prototype
still depends on:

    beat.py      the mean beat, and the per-point contraction axis by PCA of the trajectory
    seeded.py    nuclei on the tracking lattice, the axis-discontinuity map, the seeded watershed
    validate.py  the weighted circular smoothing the boundary map uses
    segment.py   the axis field and the unseeded watershed (imported by seeded)
    strain.py    the strain-axis field (imported by validate)

The rest of `cardio_cells` -- the figures, the `algebraic/` and `crash/` campaigns of the discovery
loop -- is in the repository's history and is recoverable with `git log --diff-filter=D`.
