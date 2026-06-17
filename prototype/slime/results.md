# Slime test suite (Plexus port of SebLague/Slime-Simulation)

name | verdict | coverage | contrast | reinforce | overlap | notes
---|---|---|---|---|---|---
slime_default | PASS | 0.663 | 0.55 | 1.00 | - | baseline Physarum network: a disc reticulates into a web
slime_fine | PASS | 0.620 | 0.50 | 1.00 | - | short reach + low diffuse -> fine capillary mesh
slime_coarse | PASS | 0.589 | 0.64 | 1.00 | - | long reach -> few large transport cells
slime_filaments | PASS | 0.998 | 0.26 | 1.00 | - | narrow cone + low turn -> long straight filaments
slime_curly | PASS | 0.388 | 0.51 | 1.00 | - | wide cone + high turn -> curly foam texture
slime_sparse | PASS | 0.517 | 0.53 | 1.00 | - | high decay -> faint, transient sparse trails
slime_dense | PASS | 0.887 | 0.50 | 1.00 | - | low decay + strong deposit -> thick space-filling web
slime_smear | PASS | 0.398 | 0.54 | 1.00 | - | very high diffuse -> trails smear into blobs
slime_ring_in | PASS | 0.795 | 0.40 | 1.00 | - | ring spawn pointing inward (Lague signature)
slime_point | PASS | 0.994 | 0.47 | 1.01 | - | single point source -> radiating burst
slime_random_full | PASS | 0.991 | 0.33 | 1.00 | - | uniform seed -> global space-filling network
slime_torus | PASS | 0.992 | 0.33 | 1.00 | - | periodic boundary -> seamless endless network
slime_two_repel | PASS | 0.703 | 0.65 | 1.20 | 0.030 | 2 species, cross=-1 -> territorial segregation
slime_two_attract | PASS | 0.422 | 0.73 | 1.00 | 0.934 | 2 species, cross=+1 -> shared co-aligned network
slime_two_asym | PASS | 0.798 | 0.67 | 1.32 | 0.056 | 2 species, different settings, repelling
slime_three | PASS | 0.749 | 0.70 | 1.59 | 0.036 | 3 mutually-repelling species -> triple junctions
slime_four | PASS | 0.756 | 0.74 | 1.91 | 0.022 | 4 mutually-repelling species (full RGBA showcase)
sweep_angle_10 | PASS | 0.824 | 0.55 | 1.00 | - | sweep: angle 10
sweep_angle_18 | PASS | 0.712 | 0.59 | 1.00 | - | sweep: angle 18
sweep_angle_28 | PASS | 0.674 | 0.58 | 1.00 | - | sweep: angle 28
sweep_angle_38 | PASS | 0.663 | 0.59 | 1.00 | - | sweep: angle 38
sweep_angle_50 | PASS | 0.665 | 0.61 | 1.00 | - | sweep: angle 50
sweep_decay_0p006 | PASS | 0.753 | 0.61 | 1.00 | - | sweep: decay 0p006
sweep_decay_0p012 | PASS | 0.674 | 0.58 | 1.00 | - | sweep: decay 0p012
sweep_decay_0p025 | PASS | 0.615 | 0.56 | 1.00 | - | sweep: decay 0p025
sweep_decay_0p04 | PASS | 0.555 | 0.53 | 1.00 | - | sweep: decay 0p04
sweep_decay_0p06 | PASS | 0.517 | 0.55 | 1.00 | - | sweep: decay 0p06
sweep_deposit_0p4 | PASS | 0.544 | 0.67 | 1.00 | - | sweep: deposit 0p4
sweep_deposit_0p7 | PASS | 0.644 | 0.60 | 1.00 | - | sweep: deposit 0p7
sweep_deposit_1p0 | PASS | 0.696 | 0.56 | 1.00 | - | sweep: deposit 1p0
sweep_deposit_1p4 | PASS | 0.762 | 0.53 | 1.00 | - | sweep: deposit 1p4
sweep_diffuse_0p12 | PASS | 0.682 | 0.57 | 1.00 | - | sweep: diffuse 0p12
sweep_diffuse_0p25 | PASS | 0.691 | 0.58 | 1.00 | - | sweep: diffuse 0p25
sweep_diffuse_0p4 | PASS | 0.696 | 0.59 | 1.00 | - | sweep: diffuse 0p4
sweep_diffuse_0p6 | PASS | 0.695 | 0.58 | 1.00 | - | sweep: diffuse 0p6
sweep_diffuse_0p85 | PASS | 0.685 | 0.60 | 1.00 | - | sweep: diffuse 0p85
sweep_dist_0p012 | PASS | 0.890 | 0.52 | 1.00 | - | sweep: dist 0p012
sweep_dist_0p02 | PASS | 0.733 | 0.55 | 1.00 | - | sweep: dist 0p02
sweep_dist_0p03 | PASS | 0.630 | 0.61 | 1.00 | - | sweep: dist 0p03
sweep_dist_0p045 | PASS | 0.536 | 0.65 | 1.00 | - | sweep: dist 0p045
sweep_dist_0p06 | PASS | 0.468 | 0.69 | 1.00 | - | sweep: dist 0p06
sweep_size_1 | PASS | 0.674 | 0.58 | 1.00 | - | sweep: size 1
sweep_size_2 | PASS | 0.462 | 0.57 | 1.00 | - | sweep: size 2
sweep_size_3 | PASS | 0.329 | 0.50 | 1.00 | - | sweep: size 3
sweep_turn_0p15 | WEAK | 1.000 | 0.16 | 1.00 | - | sweep: turn 0p15
sweep_turn_0p3 | PASS | 0.880 | 0.53 | 1.00 | - | sweep: turn 0p3
sweep_turn_0p45 | PASS | 0.674 | 0.58 | 1.00 | - | sweep: turn 0p45
sweep_turn_0p7 | PASS | 0.498 | 0.63 | 1.00 | - | sweep: turn 0p7
sweep_turn_1p0 | PASS | 0.421 | 0.62 | 1.00 | - | sweep: turn 1p0
slime_six | PASS | 0.803 | 0.80 | 3.54 | 0.020 | 6 mutually-repelling species -> six colour territories
slime_eight | PASS | 0.710 | 0.93 | 3.55 | 0.009 | 8 mutually-repelling species -> eight colour territories
