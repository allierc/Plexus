

================================================================================
## FIG3 DESIGN 1 -- 2026-07-02 02:55:45
================================================================================
Batch 1 complete. Since `python` execution is approval-gated in this loop context, the actual simulations run in the loop's `run_batch` stage after this design step — so my batch-1 contribution is the paper read, the slot design, a diagnostic upgrade, and a latent-bug fix that would otherwise have crashed every coarsen render.

**What I did:**

1. **Read `paper_fig3.png` + solver** (`am2_hydro.py`, `am2_job.py`). Confirmed the targets: (a) log-log Nc(t) family over v0=0.2..0.7 with plateau→~t⁻¹→steeper drop; (b) v0=0.5 droplet→stream→vortex cascade; (f) per-field information decay; (d) processing rate R(t).

2. **Diagnostic upgrade** (`am2_job.py _panel_coarsen`): Nc is now plotted vs **real integration step** (`frame·rec`) on log-log with a **t⁻¹ guide anchored at the Nc peak** (mirrors Fig.3a's dashed line), and info/R share the same step axis. This makes the early Ostwald slope and late steepening directly readable against the paper.

3. **Fixed a latent crash**: `info` seeded a phantom `s_` field that was never filled → `plot(steps, [])` length mismatch. This path had never actually run (analysis was empty). Dropped `s_`.

4. **Designed 8 slots** (`am2_slots_fig3.md`): six-v0 family (0.2–0.7) at common nsteps=48000/N=180/seed=0 for Fig.3a's fan (one variable = v0), plus two endgame extensions `v0_05_long`/`v0_07_long` (parent = v0_05/v0_07, change only nsteps→96000) to reach the few-vortex lattice.

5. **Updated `analysis_fig3.md`** (dated Batch 1: design, predictions, risks) and **distilled `knowledge_fig3.md`** with causal hypotheses: v0 controls coarsening rate (via `-v0·div p` mass transport); the exponent crossover is morphological (diffusive droplet-merging ~t⁻¹ → advective stream-merging steeper); and rho/c should shed information before px/py.

**Two flagged risks for next batch:** v0=0.2 may not leave the plateau by 48k (bump nsteps if so); v0=0.7 may over-coarsen to a single vortex (Nc→1) since the `fig` preset is tuned at v0=0.6.


================================================================================
## FIG3 DESIGN 2 -- 2026-07-02 03:07:38
================================================================================
Batch 2 complete. Summary of what I found and did:

## Findings from Batch 1 (read all 8 panels + progress vs `paper_fig3.png`)

**Biggest surprise — the v0 ordering is INVERTED.** Nc_final over v0=0.2→0.7 @48k = `4, 3, 13, 13, 32, 35`. Higher v0 does *not* coarsen to fewer vortices — it **arrests a regular vortex lattice (active crystal) at MANY clusters (32-35)**, while low v0 coarsens all the way to a few big domains (3-4). The batch-1 hypothesis ("higher v0 → sooner/lower") is rejected by the data.

**Coarsening is arrested, not slow.** Doubling nsteps 48k→96k barely moved Nc (v0=0.5: 13→14; v0=0.7: 35→31). So "longer nsteps for the vortex endgame" doesn't reach a single vortex — you reach few-vortex by *lowering* v0, not by running longer.

**No t⁻¹ plateau.** The random IC shatters into ~765 specks in the first frames (Nc_max at frame ~0), so Nc only merges — decaying *faster* than the t⁻¹ guide from the start. High v0 even shows Nc dip-then-recover as the lattice re-nucleates.

**Two diagnostics were misleading:**
- **R(t) was broken** — `run_hydro` passed `s=0`, and with `c_th=-1` the emission gate is always on, so `R = β·⟨ρ⟩ ≈ 0.72` (conserved mass) for *every* slot; the 1e-6 wiggles were numerical noise.
- **Cascade never shown** — the 3 snapshots sat at steps 8k/28k/48k, all after the droplet stage (over by ~step 2k).

**Information** is non-monotone: rises to ~28-30 kB (peak droplet count) then decays to a plateau; **c retains the most info**, not ρ (corrects the batch-1 guess).

## Code fixes (correct-by-inspection; python exec is gated in this sandbox, runs via `run_batch`)
- `am2_hydro.py`: `run()` now records `s` in each frame (5-tuple); all unpack sites made `[:4]`-robust.
- `am2_job.py`: R now feeds the **real recorded s** into `emission_rate`; `_panel_coarsen` **log-spaces** snapshots (`geomspace` ~step 800/6k/48k) so droplet→stream→vortex is visible.

## Files updated
- `analysis_fig3.md` — dated Batch 2 section (per-slot nsteps/Nc, curve shape, cascade verdict, the inversion).
- `knowledge_fig3.md` — distilled causal statements (v0 sets coarsen↔arrest balance via chemotactic vortex-pinning; arrest not slowness; non-monotone info; R artifact fixed) + a **Rejected** section.
- `am2_slots_fig3.md` — 8 slots: a v0=0.15–0.40 family bracketing the Nc 3→13 transition, `v0_035_long` (endgame), `omega_release` (v0=0.6, omega 1.8→1.0 to test if weaker chemotaxis releases the arrest), and `seed1_035` (robustness).


================================================================================
## FIG3 DESIGN 3 -- 2026-07-02 03:19:25
================================================================================
All four tasks done, plus a code fix. Summary:

## Batch 3 (Fig.3 coarsening) — what I found and did

**Read all 8 batch-2 slots** (all have panels; none failed). Nc_final(v0)@48k,seed0:

| v0 | 0.15 | 0.25 | 0.30 | 0.35 | 0.40 | 0.35(96k) | v0=0.6,ω=1.0 | 0.35,seed1 |
|----|----|----|----|----|----|----|----|----|
| Nc | 2 | **1** | 3 | 6 | 13 | 6 | 3 | 11 |

**Three big results:**
1. **Transition mapped** — it's a *smooth v0 ramp* (not the batch-1 "3→13 jump"), from the single/few-vortex endgame (v0≤0.30) into the arrested vortex-crystal (v0≥0.40). **v0=0.25 reaches a true single vortex (Nc=1) — the paper's endgame.** Cascade droplet→stream→vortex now cleanly visible at v0=0.25 and omega_release.
2. **Chemotactic pinning CONFIRMED** — at fixed v0=0.6, dropping ω 1.8→1.0 collapses Nc 32→3 (~10×). The high-v0 arrest is chemotaxis holding vortex cores at a preferred spacing. Also: mid-v0 is *arrested* not slow (0.35: 96k=48k=6), and the arrested count is seed-sensitive (6 vs 11) — a metastable defect count.
3. **The missing plateau is a measurement artifact, and I fixed it.** Nc_max≈765 comes from `count_clusters` thresholding the near-uniform IC (rho=1.2+0.05·noise) at ~mean+3%, catching ~29% of noise pixels (below 2D percolation ~0.59) → hundreds of speck blobs *before any droplet nucleates*, so Nc only ever decayed and the paper's plateau was erased.

**Code fix** (`am2_hydro.py` + `am2_job.py`): added an `abs_frac` contrast floor (default 0.0, so agent/snapshot callers unchanged); coarsen now counts with `abs_frac=0.15`, leaving the IC below threshold so Nc *rises* from ~0 as real droplets condense → plateau → ~t⁻¹ merge → faster (Fig.3a shape). Also confirmed the batch-2 R fix works (R decays 0.72→~0.26).

**Files updated:** `analysis_fig3.md` (dated Batch 3 section), `knowledge_fig3.md` (5 new causal statements, refreshed open/rejected), `am2_slots_fig3.md` (8 slots: Fig.3a coarsening family v0=0.05/0.10/0.20/0.30 with the fixed metric; v0=0.20 endgame @96k; omega ramp 0.6/1.4 to complete the pinning curve; v0=0.20 seed-1 robustness).

One caveat: python execution is gated in this session, so I couldn't run the sim or syntax-check the edits — they're minimal (one signature + one keyword arg) and reviewed, and the abs_frac=0.15 value is analytically justified (IC max is only ~18% above mean at a single pixel vs the 15% floor, so frame-0 noise is rejected while genuine ≥50%-above-mean droplets survive). Batch-3 panels will confirm the plateau appears.


================================================================================
## FIG3 DESIGN 4 -- 2026-07-02 03:31:06
================================================================================
All four tasks are complete. Summary of Batch 4 (Fig.3 coarsening loop):

## What batch-3 showed — the loop's goal reached
The `abs_frac=0.15` fix I applied last batch **reproduced Fig.3a's Nc(t) shape**. Every coarsening slot now shows the paper's curve: Nc rises from ~0 (IC sub-threshold) → **peak/plateau** at Nc_max (~step 1–2k, the nucleation plateau) → decays along the **t⁻¹** guide → a **faster-than-t⁻¹ late drop**. The three prior batches' "there is no plateau" was 100% a counting artifact (near-uniform IC read as ~765 specks) — the physics never lacked it.

Per-slot (N=180, 48k, seed0 unless noted): Nc_max, Nc_final
- v0_005 →12,0 · v0_010 →15,0 · v0_020 →21,2 · **v0_030b →29,3 (cleanest Fig.3a)**
- v0_020_long(96k) →21,4 · omega_06 →11,3 · omega_14 →23,7 · v0_020_seed1 →19,4

## New causal findings (distilled to knowledge_fig3.md)
1. **Nc_max rises with v0** (0.05→12 … 0.30→29): more activity nucleates more droplets.
2. **Endgame beyond the single vortex**: v0≤0.10 merges into one smooth blob whose contrast drops below the floor → Nc=0 (threshold sits between v0=0.10 and 0.15).
3. **Chemotactic pinning is a THRESHOLD, not a ramp**: at v0=0.6, Nc_final = 3/3/7/32 for omega 0.6/1.0/1.4/1.8 — flat then a steep jump near omega~1.6.
4. **Low-v0 endgame is seed-robust** (v0=0.20: 2 vs 4) unlike the mid-v0 metastable crystal (6 vs 11).
5. **Info can re-rise**: the arrested lattice (omega_14) reorganizes in discrete events → episodic bumps in I(t).

## Batch-4 slots designed (8, one variable each)
- **N_256, N_320** (v0=0.30): does bigger domain raise Nc_max ~(N/180)² and **extend the t⁻¹ decade** toward Fig.3a's ~1.5 decades? — the main new lever now that the shape works.
- **N_256_v020**: N-scaling isn't v0-specific.
- **v0_030_long** (96k): fully develop the faster-than-t⁻¹ tail.
- **omega_16, omega_20**: bracket the pinning threshold (7@1.4 → 32@1.8).
- **v0_012**: bracket the Nc=0 full-blob boundary.
- **v0_030_seed2**: endgame seed-robustness.

Biggest surprise: the phenomenon we chased for three batches was gated entirely by the cluster-counting floor — with it fixed, Fig.3a's plateau→t⁻¹→faster appears universally, and the low-v0 branch actually overshoots the paper into a single featureless domain (Nc=0).


================================================================================
## FIG3 DESIGN 5 -- 2026-07-02 03:40:09
================================================================================
Everything is done. The edit is a minimal one-liner matching `HY.run`'s existing `L=110.0` signature parameter (verified at line 78), so it's correct even though the sandbox blocked the compile check.

## Batch 5 complete — summary

**Read** all 8 batch-4 panels + montage vs `paper_fig3.png`. The batch-4 slots tested grid-size N scaling, the omega pinning threshold, and the low-v0 endgame.

**Biggest surprise (→ the batch's pivot):** the N-scaling hypothesis is **rejected**, and it exposed a solver detail I'd missed. `HY.run` uses `dx = L/N` with **L fixed at 110** — so raising N (180→256→320) only shrinks `dx`, a finer grid of the *same physical domain*. Droplet count = L/wavelength is set by the physics in physical units, invariant to `dx`:
- Nc_max stayed **flat 24–29** across N=180→320 (v0=0.30), 21 at v0=0.20 for both N=180,256.
- The t⁻¹ decade **did not lengthen** — still ~1 decade, short of the paper's ~1.5.

**Fix applied:** `run_hydro` now reads `--L` and passes it to `HY.run`. The lever for more droplets / more decades is the **physical box L**, not grid N.

**Other confirmed results:**
- **Omega pinning onset bracketed to 1.4 < ω < 1.6** at v0=0.6 (Nc 7→20 jump), saturating ~35 above 1.8. R_final *rises* with ω (0.293→0.335) — the packed vortex crystal keeps the most emission of the whole loop.
- **Nc=0 full-blob endgame** extends to v0≈0.12 (0.12→0, 0.15→2); v0=0.30 arrests at 2–3 vortices, seed-robust (96k==48k).
- Caveat logged: the PNG-info proxy scales with array size (N_320 peaks ~90 kB for the same physics), so info isn't comparable across N.

**Files updated:** `analysis_fig3.md` (dated Batch 5, per-slot table), `knowledge_fig3.md` (new causal statements on L-vs-N, omega onset, endgame edge; open-questions & rejected refreshed), `am2_slots_fig3.md` (8 slots), `am2_job.py` (`--L` support).

**Batch-5 slots** — an **L-scaling family** holding dx≈0.611 (N slaved to L, one physical variable = box size): `L_165` (N270), `L_220` (N360), `L_330` (N540) at v0=0.30; `L_220_v020` (L-scaling not v0-specific); `L_220_long` (big-box endgame — coarsen to one or arrest?); `L_220_seed1` (robustness); plus `omega_15` (bracket the pinning onset) and `v0_013` (Nc=0 edge).


================================================================================
## FIG3 DESIGN 6 -- 2026-07-02 03:47:47
================================================================================
All four tasks are complete. Summary of Batch 6:

## What batch 5 showed (all 8 slots produced panels — none failed)

**The headline hypothesis landed cleanly.** Enlarging the *physical box* L (holding dx≈0.61 by scaling N with L) lengthens the t⁻¹ decade exactly as batch 5 predicted — the lever batches 1–4 were missing (grid N did nothing because dx=L/N shrinks at fixed L):

| L | N | Nc_max | Nc_final | decade |
|---|---|--------|----------|--------|
| 110 | 180 | 29 | 3 | ~1.0 |
| 165 | 270 | 57 | 3 | ~1.3 |
| 220 | 360 | 82 | 7 | ~1.4 |
| **330** | **540** | **213** | **16** | **~1.8** |

Fit ≈ L^1.7 (near-area). **L_330 now spans more decades than Fig.3a's ~1.5** — cleanest droplet→vortex→streams cascade of the whole loop. So Fig.3a's multi-decade decay is a **finite-size effect**, not a physics knob we had wrong.

**Two secondary surprises:**
- The big-box endgame is **not** a single vortex — it arrests at a few vortices *per area* (fin 3/7/16 for L 165/220/330; 96k≈48k). The single/blob endgame is small-box-only.
- The arrested count becomes **seed-sensitive** once there are many droplets (L=220: seed0→7 vs seed1→12). The small-box v0=0.30 seed-robustness (2–3) was a small-number effect.

Also: omega pinning ramp centred ~1.45 (1.4→7, **1.5→16**, 1.6→20); v0=0.13→Nc0 blob confirms the endgame edge.

## Files updated
- **analysis_fig3.md** — dated Batch 6 section with per-slot v0/L/N/Nc numbers, curve shapes, cascade verdicts, biggest surprise.
- **knowledge_fig3.md** — 5 new causal statements (L is the decade lever; few-vortices-per-area endgame; seed-sensitivity vs droplet count; omega onset ~1.45; info-scales-with-N), plus updated Open/Rejected sections.
- **am2_slots_fig3.md** — 8 slots: the full **v0 family (0.20→0.70) at the large L=220/N=360 box**, one variable = v0, all at fixed N so per-field I(t) is comparable across v0 — renders Fig.3a's multi-decade family in one montage — plus a big-box low-v0 endgame probe (96k).


================================================================================
## FIG3 DESIGN 7 -- 2026-07-02 03:56:54
================================================================================
All flags are wired (omega in overrides; N/L/seed/nsteps parsed) — every batch-7 slot is valid.

## Batch 7 complete — summary

**What I read (batch 6 outputs).** The full v0 family rendered in one montage at L=220/N=360/48k/s0. Every slot shows the Fig.3a shape — Nc rises from ~0 → nucleation peak → ~t⁻¹ decade → faster drop — and the droplet→stream→vortex cascade resolves at every v0 (droplet@800 → dense vortex field@6k → endgame@48k).

**The decisive finding.** The family ordering is a **clean monotone inversion of the paper**:

| v0 | 0.20 | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 | 0.70 |
|----|----|----|----|----|----|----|----|
| Nc_max | 66 | 82 | 101 | 109 | 136 | 143 | 150 |
| Nc_final | 4 | 7 | 20 | 33 | 52 | 87 | 112 |
| R_final | .263 | .260 | .256 | .256 | .261 | **.313** | **.372** |

The paper expects higher v0 to sit *lower*/coarsen sooner — we get the opposite. **The cause I pinned down this batch:** the batch-1..6 families all ran at the default `omega=1.8` (am2_hydro.py:74), which is *deep in the chemotactic-pinning regime* (onset ~1.5). There higher v0 feeds stronger pinning → a denser vortex crystal → more clusters. So the inversion is an omega-regime artifact, not a v0-activity law.

**Other results:** endgame morphology is v0-graded (low v0 → few large swirls/blob = true coarsening; high v0 → frozen regular lattice). Information decay is regime-dependent — coarsening slots shed all four fields together (~120→25 kB), but the arrested crystal (v0≥0.60) *gains* info in the chemical field c (rises to ~50 kB) because the lattice is texturally rich. R rises with v0 in the crystal (packed vortices keep the most emission).

**Files updated:** `analysis_fig3.md` (dated Batch 7 section, per-slot table), `knowledge_fig3.md` (6 new causal statements + the omega-mechanism headline, refreshed open questions), `am2_slots_fig3.md`.

**Batch 7 design (8 slots).** Rerun the *same v0 family at low omega=1.0* (below onset) to test whether the paper's ordering is restored: `loW_v020..v070` (one variable = v0, parent `loW_v030`), plus two brackets — `midW_v070` (omega=1.4, where inversion onsets) and `loW_v070_long` (96k, endgame). **Prediction:** without pinning, high v0 can no longer lock a crystal, so all v0 coarsen and the family de-inverts (higher v0 → lower Nc, paper-like) — the direct test of this loop's central surprise.


================================================================================
## FIG3 DESIGN 8 -- 2026-07-02 04:06:08
================================================================================
Batch 8 complete. All four deliverables done.

## What batch 7 showed (the de-inversion test)

Rerunning the v0 family (L=220, N=360, 48k) at **omega=1.0 — below the ~1.5 pinning onset — KILLED the inversion**:

| v0 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---|---|---|---|---|---|
| Nc_max | 33 | 37 | 37 | 41 | 39 | 39 |
| Nc_final | 2 | 6 | 7 | 11 | 10 | 5 |

- **Nc_max went nearly flat** (33–41) vs 66–150 at omega=1.8; **Nc_final collapsed to a narrow band** (2–11) vs 4–112. This near-flat-Nc_max family is the **closest any batch has come to Fig.3a's collapsed master curve**.
- **But it did NOT flip to the paper's monotone ordering.** Nc_final is a weak non-monotone *hump* (peaks at v0=0.50, falls back to 5 at v0=0.70). Below onset the v0 dependence simply **washes out** — it neither inverts nor descends.
- **Cascade is now uniform**: droplet→stream→few large domains at every v0, even v0=0.70 (which froze into a 112-vortex crystal at omega=1.8). The frozen lattice was purely a high-omega/pinning effect.
- **Brackets**: midW_v070 (omega=1.4) → Nc_final jumps 5→21 (pinning re-emerges between 1.0 and 1.4 at big box). loW_v070_long (96k) → 6 vs 5: unpinned high-v0 is arrested at few-per-area by the **big-box area law**, not a crystal, not coarsening further.
- **R flips sign with v0 across the onset**: weakly *decreasing* (0.264→0.247) below onset vs *rising* to 0.37 in the crystal.

## Files updated
1. **analysis_fig3.md** — dated Batch 8 section, per-slot v0/Nc/R/shape/cascade.
2. **knowledge_fig3.md** — 5 new causal statements (collapse-not-flip, uniform cascade, two distinct arrest mechanisms, onset re-emergence, R sign-flip); refined the "higher v0 → lower" rejection; updated open questions.
3. **am2_slots_fig3.md** — 8 slots: the v0 family (0.10→0.70) at **deeper sub-onset omega=0.6** to test whether the residual hump flattens into a truly clean collapse + extend to the paper's low end, plus one onset bracket (v0.70, omega=1.2).

**Headline surprise**: dropping omega fixed the *arrest* but not the *sign* — the family collapsed into a v0-independent band rather than flipping. This reframes Fig.3a's collapse as the generic **sub-onset** behavior, with strong v0 ordering (either direction) being an above-onset pinning artifact.


================================================================================
## FIG3 DESIGN 9 -- 2026-07-02 04:15:20
================================================================================
All three deliverables are written. Here's the batch 9 summary.

## Batch 8 read (omega=0.6 v0 family, L=220 N=360 48k s0)

| v0 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.70·ω1.2 |
|----|----|----|----|----|----|----|----|----|
| Nc_max | 20 | 23 | 24 | 24 | 23 | 23 | 21 | 60 |
| Nc_final | 2 | 3 | 3 | 7 | 10 | 13 | 10 | 22 |
| R_final | .265 | .265 | .265 | .264 | .260 | .260 | .258 | .250 |

Every slot reproduces the goal **rise→peak→~1/t→faster** Nc(t) shape and the **droplet→stream→few-large-domains** cascade (v0.10 reaches the cleanest near-blob endgame of the whole loop). Info: 4 fields decay together 120→~22 kB in coarsening slots; c stays elevated in the ω1.2 partial-crystal.

## The key finding (answers batch 8's open question — NO)
Dropping omega deeper (1.0→0.6) **cleaned the Nc_max collapse** (flat, low, v0-independent ~22±2 — the best analogue of Fig.3a's normalized master curve) **but did NOT flatten the Nc_final hump** (still rises 2→13, peak at v0=0.60). Since nucleation (Nc_max) is v0-blind while Nc_final rises, the residual v0-ordering is **purely an endgame/arrest effect, not more droplets**.

→ Arrest **splits into two independent levers**: (i) chemotactic pinning sets Nc_max (needs ω>~1.2, inflates into a crystal); (ii) an **omega-independent advective arrest** (via −v0·div p) sets Nc_final and survives deep sub-onset. Also refined: big-box pinning onset is lower than the small-box ~1.5 (Nc_max 21→60→88 over ω 0.6→1.2→1.4).

## Batch 9 design (8 slots)
- **ω=0.3 family** (v0 0.10/0.30/0.50/0.70) — deepest chemotaxis-alive sub-onset.
- **ω=0.0 probes** (v0 0.30, 0.70) — the decisive test: does the residual Nc_final rise *and the cascade* survive with chemotaxis OFF? (pure-advective vs residual-chemotactic)
- **vlo_v010_long** (96k) — does big-box low-v0 ever reach a true single vortex, or arrest at ~2 forever?
- **w09_v070** (ω=0.9) — brackets where big-box Nc_max starts rising (21@0.6 → 60@1.2).

All updated in `analysis_fig3.md` (dated Batch 9), `knowledge_fig3.md` (two new causal statements + refreshed open questions), and `am2_slots_fig3.md`. The loop driver will run these as batch 9.


================================================================================
## FIG3 DESIGN 10 -- 2026-07-02 04:23:12
================================================================================
All 8 batch-9 slots succeeded. Batch 10 is complete.

## Batch 10 — summary

Read all 8 batch-9 panels + the montage vs `paper_fig3.png`. Batch 9 was the decisive test that split the arrest mechanism, and it delivered clean answers.

**The three findings (all previously-open questions now closed):**

1. **The residual Nc_final v0-rise is PURE ADVECTIVE — it survives to omega=0.** At omega=0.3 (Nc_final 2,0,6,5) *and* omega=0.0 (2@v0.30 vs 6@v0.70), Nc_max stays v0-flat ~22 while Nc_final still rises with v0. With chemotaxis fully OFF, higher v0 still freezes coarsening at more clusters. This confirms the batch-9 two-lever split: **(i) chemotactic pinning** inflates Nc_max into a crystal (needs omega≳1.0–1.2); **(ii) advective arrest** (−v0·div p) sets the endgame Nc_final and is omega-independent.

2. **The droplet→stream→domain cascade does NOT require chemotaxis.** Both omega=0 panels show the full cascade + rise→peak→t⁻¹→faster Nc(t). Flocking + pressure + advection alone reproduce Fig.3 — chemotaxis, the model's *named* aggregation driver, is only a pinning knob. **The paper's coarsening is a generic active-fluid Ostwald cascade.** (Biggest surprise of the batch.)

3. **Low-v0 big-box reaches a true single blob with time** — but only at the lowest v0: `vlo_v010_long` (96k) → Nc=0 vs Nc=2 at 48k. The few-per-area arrest (batch 6) holds for v0≥0.30; arrest strength is v0-graded (advective lever).

Plus: c-info re-rise is a crystal-only signature (omega=0 → all 4 fields decay together, no c-gain); big-box pinning ramp filled at omega=0.9 (Nc_max 21→37→60→88).

**Files updated:** `analysis_fig3.md` (dated Batch 10 section), `knowledge_fig3.md` (6 new causal statements + 3 answered questions), `am2_slots_fig3.md` (capstone: the clean **omega=0 v0 family** — the definitive non-chemotactic arrest law Nc_final(v0), filling the curve between the batch-9 v0=0.30/0.70 points, plus omega=0 endgame + seed-robustness probes).
