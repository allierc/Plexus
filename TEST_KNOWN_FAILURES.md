# The 8 tests that were already red before the plexus2 alignment campaign

Recorded 2026-09-06, on branch `plexus2-algebra-alignment`, and confirmed red at `20eb3d06` --
the working point the branch was cut from -- so none of them belongs to this campaign. They are
listed here so that "8 failed, 113 passed" can be read as unchanged rather than re-derived every
time, and so a NINTH failure is visible immediately.

    PYTHONPATH=src:tools python -m pytest tests -q     ->  8 failed, 113 passed, 1 xfailed

| # | test | what it is |
|---|---|---|
| 1 | `test_gate_freeze.py::test_every_frozen_gate_is_clean_today` | `02_ecm_block`'s frozen `_gate:` block moved (78ff4878e7afe8f7 -> 76274650777dba35) and was never re-frozen. `run_gates` refuses to grade that gate for the same reason. From commit `49c14ad2`. |
| 2 | `test_mpm_decomposition.py::test_decomposition_reproduces_the_oracle[liq_grav]` | the decomposed MPM substeps no longer reproduce the monolithic oracle |
| 3 | `test_mpm_decomposition.py::test_decomposition_reproduces_the_oracle[snow]` | same |
| 4 | `test_mpm_decomposition.py::test_decomposition_reproduces_the_oracle[elastic]` | same |
| 5 | `test_mpm_decomposition.py::test_decomposition_reproduces_the_oracle[obstacle]` | same |
| 6 | `test_neural.py::test_assembly_activity_is_the_mean_voltage_of_its_neurons` | `aggregate` KeyError |
| 7 | `test_neural.py::test_aggregate_defaults_to_the_centroid_it_was_named_for` | `aggregate` KeyError |
| 8 | `test_neural.py::test_aggregate_refuses_a_block_the_child_does_not_have` | `aggregate` KeyError |

Rows 6-8 are worth a second look during THIS campaign rather than after it: `aggregate` is one of
the two families plexus2 says may cross the containment map, it has exactly one live registration
in the whole tree (`cell_geometry`), and three of its tests are red. A family with one
implementation and no working tests is a family the code does not really have.

Rows 2-5 and row 1 are independent of the operator algebra.
