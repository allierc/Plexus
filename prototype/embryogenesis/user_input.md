# User directives (read + acknowledge each batch; apply going forward)

Overall the runs so far look good but three ranges should change:

1. **Cell movement is too slow — double it.** The `move_speed` baseline is now 0.12 (was 0.06).
   You may explore up to ~2x beyond that (≈0.24) when a stage needs faster flow/migration.
2. **Allow the cell population to grow up to ~4x via `cell_divide`.** Use division (`div_rate`,
   `max_occ`, `buffer` already 3000) to reach up to ~4x the starting count when a stage (1C/1D)
   calls for it — do not cap proliferation prematurely.
3. **Double the simulation length — use ~12000 frames (was 6000)) so slow dynamics have time to
   develop. Keep each job within the L4 wall (30 min); raise `stride` if render time grows.
