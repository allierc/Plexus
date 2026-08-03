# Pre-clean archive, 3 August 2026

Everything that was in `log/okuda` and `config/okuda` before the tree was cut to twelve
starting points. Nothing here is deleted — the live tree is a *selection*, not a purge.

- `config_okuda.tar.gz` — all 92 configs as they were
- `run_records/` — 75 runs, records only (`diag.json`, `metrics.json`, `spec_run.yaml`,
  `description.txt`). Movies and `.npz` were left out: 224 MB, regenerable from the spec.
- `removed_runs/` — the 63 run directories taken out of the live tree, complete
- `removed_configs/` — the 80 configs taken out

## The twelve kept, and why

Chosen to span **pattern wavelength** — the variable the loop only learned to measure
today, and the one the three-strip comparison says governs budding — plus the best
protrusions and the growth family's own control.

| run | protr | spots | spacing | why |
|---|---|---|---|---|
| `refute_coral_nocons` | 1.169 | – | – | best protrusion on disk |
| `mini_coral_nodilute` | 1.131 | – | – | second, same family |
| `wk_pressure_pos_s0` | 1.110 | 59 | 2.5 | best growth+division |
| `wk_curvature_pos_s0` | 1.085 | 7 | 7.8 | growth+division, coarse pattern |
| `wk_null_s0` | 1.079 | 3 | 10.4 | the wk family's CONTROL, coarsest |
| `p1_ph_coral_fixed_ball` | 1.076 | – | – | coral at 2000 cells, phase-1 reference |
| `wk_tension_neg_s0` | 1.075 | 4 | 9.4 | growth+division, negative feedback sign |
| `wk_apical_area_pos_s0` | 1.074 | 26 | 4.5 | growth+division, mid pattern |
| `cellfix_B_new` | 1.073 | 0 | – | the cell-fix regression pair |
| `cfl_c000p080_d002p000` | 1.006 | **101** | 2.5 | chemistry only, FINE end |
| `cfl_c001p300_d000p160` | 1.006 | **9** | 8.4 | chemistry only, COARSE end |
| `coral_fixed_ball` | 1.006 | 14 | 6.9 | fastest run there is, 27 s |

**Spot counts run 3 → 101** across the set, spacing 2.5 → 10.4 cells. That ladder is the
point of the selection: it is the axis nothing in this project had ever recorded.

## Two things the selection makes visible

**They are FAST.** 1420 s of GPU serially, about two minutes wall for a 12-slot batch. The
hour-long rounds of 3 August were not these specs — they were these specs with their
reservoirs enlarged to 20–69k cells. The originals sit at 1778–6114 and finish in minutes.

**None carries a `composition.json`.** Verified: zero in `config/`, zero in any run dir. So
the Archivist still cannot rebuild any of them as a graph, and a cold start will fall back
to the reference recipes — which is what Cedric chose. These twelve are recon material and
a measured baseline, not a frontier.

## One thing worth arguing with

The obvious hypothesis from the strips — *finer pattern, less shape change* — is **not**
supported by this table. `wk_pressure_pos_s0` has the most spots of the growth family (59)
**and** the highest protrusion (1.110), while `wk_null_s0` has the fewest (3) and less
protrusion (1.079). If wavelength governs budding, it does not do so monotonically, and the
sign may be the opposite of what the three-image comparison suggested. That comparison
crossed families; this table does not. Worth testing before anyone writes it down.
