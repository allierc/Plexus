# User directives (read + acknowledge each batch; apply going forward)

## ⚠️ URGENT (2026-07-05) — CELLS ARE COLLAPSING in the recent GRO jobs. FIX BEFORE STUDYING GROWTH.
Quantitative check over ALL 16 GRO slots (b51 + b52) — every one HARD-FAILS the 1A gate:
- **collapsed: mean 0.89 (range 0.77–1.00), ALL 16 > 0.**
- **nn_min: 0.0002 on every slot — 99% BELOW r0=0.02** (cells crushed to ~100× closer than the exclusion distance).
- **escape: 13/16 slots > 0** (b52 pureball & noshell fully escape, escape=1.0).
- **clean slots (collapsed=0 & escape=0 & nn_min≥r0): 0 / 16.**

CAUSE (confirmed): every GRO spec regressed to **`mpm_to_agent.confine: 3.0`** — the exact high inward
press that 1A PROVED causes collapse. The ESTABLISHED non-collapsing operating point is
**`mpm_to_agent.confine: 0.03` + `repel.strength: 150`** (knowledge 1A/1B). **Restore that low-press
point in EVERY `embryo_GRO_*` spec.** The blastula MUST hold the 1A hard-failure gate (collapsed=0,
nn_min≥r0, escape=0) *while* `cell_grow` expands it — growth measured on a collapsed/escaping blastula
is invalid. Re-baseline GRO on the confine-0.03 operating point (rate=0 no-op must be gate-clean) FIRST,
then resume the growth-law sweep.

## Earlier directives (still apply):

Overall the runs so far look good but three ranges should change:

1. **Cell movement is too slow — double it.** The `move_speed` baseline is now 0.12 (was 0.06).
   You may explore up to ~2x beyond that (≈0.24) when a stage needs faster flow/migration.
2. **Allow the cell population to grow up to ~4x via `cell_divide`.** Use division (`div_rate`,
   `max_occ`, `buffer` already 3000) to reach up to ~4x the starting count when a stage (1C/1D)
   calls for it — do not cap proliferation prematurely.
3. **Double the simulation length — use ~12000 frames (was 6000)) so slow dynamics have time to
   develop. Keep each job within the L4 wall (30 min); raise `stride` if render time grows.
