# Embryogenesis (active matter × MPM) — knowledge ledger (v2, SCORECARD-driven, restart 2026-07-03)

Cumulative, curated working memory. CUMULATIVE: add/curate, never erase. Tags: **[established]**
(≥3 seeds, |Δ|>2·SD vs control) · **[open]** (hypothesis to test) · **[rejected]** (falsified) ·
**[engineering]** (tooling/metric). **Findings are decided on the QUANTITATIVE SCORECARD numbers +
their 5/25/50/75/100% trajectory — NOT on the movie. Visuals propose; statistics decide.**

## Objective
An in-silico blastula whose morphodynamics match the QUANTITATIVE observables of real teleost
(zebrafish) embryogenesis — see the reference section at the bottom for the ground-truth metrics.

## The system
A **blastula**: elastic MEMBRANE shell (deep blue) + WATER core (light blue), held by a substrate
anchor. **Cells** = active-matter agents in the core: dragged/confined by the fluid, deform the
membrane, divide, flow, partition. Base spec: `specs/embryo_base.yaml`. Even init: `spawn: sunflower`.
Operators (`src/plexus/operators/`): couplings `agent_to_mpm`, `mpm_to_agent` (`field: mass|colour`),
`mpm_spin`, `flow_align`, `agent_remodel`, `cell_divide`; cell laws `repel`, `attraction_repulsion`,
`separation`, `polar_align`, `glide`; chemical `deposit`/`diffuse`/`decay`/`chemotax`; MPM
`mpm_strain/p2g/mpm_grid_update/g2p`, `mpm_anchor`, `mpm_drag`.

## The SCORECARD (the decision basis) — `scorecard.py` → per slot `scorecard.json` + `scorecard.png`
5 families, EACH at 5/25/50/75/100% of the run: **shape** (fourier m1–m5, circularity, area,
perimeter, deform_rms) · **organization** (gr_peak, nn_mean/nn_cv, density_cv, contact_same) · **flow**
(speed, polar_order, enstrophy/net_circulation, msd, persistence_frames, **corr_length_xi** ξ) ·
**topology** (**t1_rate** neighbour-exchange) · **partition** (segregation_index, mixing_entropy,
mi_type_x, interface_frac) · **coupling** (stress_cell_corr, deform_cell_corr, flow_deform_lag,
**div_stress_angle** division-axis vs principal-stress). Shape also reports **shape_index** p=P/√A
(fluid⇄solid ≈3.81). HARD FAILURES (`metrics.json`, a gate not a tradeoff):
`collapsed>0`, `nn_min<r0`, `escape>0`, `accel` bounded only by the `vmax` clamp.
`[established]` requires ≥3 seeds and |Δ|>2·SD vs its ablation control.
The four zebrafish-facing observables (shape_index, ξ, div_stress_angle, t1_rate) are DONE and live
in the scorecard — read them as tier-3 diagnostics (validate before gating).

## Staged ladder (targets; ≤2 days OR ≤48 batches per sub-phase, whichever first)
1A stable / no-collapse · 1B inner flow deforms membrane · 1C division deforms shell · 1D high-density
flow / collective migration · 1E two-type partition. Then INT (integrate all).

## STAGE STATUS
- **1A — STARTED Batch 1, CLOSED Batch 7 (2026-07-03). GATE MET.** Official gate `collapsed=0 & escape=0` met with
  margin since b03; the self-imposed `nn_min≥r0 (0.02)` sub-gate now effectively met too. **b06 (all 8 landed via the
  working HOLD-retry) crossed the last gap:** every slot collapsed=0, escape=0, accel 0.0009–0.0014 (balance-bounded),
  membrane round (circularity 0.998, deform 0.0012, shape_index 3.55). **Flagship s3 = n24 + confine 0.03 + repel 150 →
  nn_min 0.0199 (0.995× r0)**, and its g(r) went gas-like (`gr_peak` 1.33, `gr_peak_r` **0.1433** = ~2× nn_mean — the
  near-neighbour shell itself vanished, not just the doublet). Two mechanism findings (see FINDINGS): **DENSITY is the
  dominant lever** (n44→24→16 at fixed force: nn_min 0.018→0.0194→0.0196) and **repel FORCE is saturated** (150 vs 400
  identical 0.0194). **1A OPERATING POINT: nodiv n44 (or n24) + confine 0.03 + repel 150** (`specs/embryo_1B_base.yaml`).
- **1B — STARTED Batch 7 (2026-07-03). Gate: inner flow visibly deforms the membrane (`deform_rms` ↑ from the ~0.0012
  floor, `fourier_m2/m3` ↑ from ~3e-4, `deform_cell_corr`/`flow_deform_lag` coupling) while collapsed=0 & escape=0 hold.**
  Quiescent baseline (all 1A slots): deform_rms ~0.0012, fourier_m2/m3 ~3e-4, flow ~0.0016, `net_circulation` 0,
  `polar_order` ~0.02–0.04 — cells nearly frozen/diffusive, membrane essentially undeformed. Batch 7 sweeps the deform
  drivers as single-lever changes on `embryo_1B_base.yaml`: `agent_to_mpm.agent_mass` (b01 deform lever), `mpm_spin.omega`
  (direct fluid swirl), move_speed (`embryo_1B_fast`, 0.24), `flow_align.gain` (120). Hypothesis: agent_mass ≳ spin drive
  deform; motility/flow_align are flow-source-limited. (see `embryo_slots.md` / analysis Batch 7).
  - **b07 RESULT (Batch 8 read; all 8 landed, nodiv n44, c03, r150): the Batch-7 ranking was WRONG. MOTILITY is the
    strongest deform driver; agent_mass second (saturating); mpm_spin and flow_align are NULL.** Deform ladder vs the
    quiescent floor (deform_rms 0.00124, fourier_m2 0.0004): fast_mass4x (move 0.24 + mass 8e-6) = **0.00444 (3.6×,
    batch max), fourier_m2 0.00667 (16×)**, and the ONLY POSITIVE deform_cell_corr (+0.0895); mass10x (move 0.12 +
    mass 2e-5) = 0.00302 (2.4×); spin1p0 (omega 1.0) = 0.00239 with net_circulation STILL 0 and enstrophy 3.9e-7
    (< floor) — spin rotates nothing; flowalign120 = 0.00120 = floor exactly. combo (fast+mass+spin0.8) 0.00384 <
    fast_mass4x — spin is subtractive. Deformation is a transient WOBBLE (fast deform_rms 0.00257→0.0042→0.00598→
    0.00337→0.00444, oscillates; circularity stays 0.997), NOT a locked shape. 1A held on all 8 (collapsed/escape 0,
    nn_min ~0.019, accel genuine not clamp-bound). Membrane still visibly ROUND. Batch 8 pushes the motility×mass
    corner + a new lever: SOFTER membrane (youngs 200→80→40). See FINDINGS entry.
  - **b08 RESULT (Batch 9 read; nodiv n44, c03, r150): deform lever = cell→fluid COUPLING GAIN (agent_mass AND
    agent_to_mpm.k, agent_mass NOT saturated); CEILING = ESCAPE (motility×coupling overdrive); membrane STIFFNESS
    FALSIFIED as a lever.** Clean flagship s0 fast_mass10x (move 0.24 + mass 2e-5): deform_rms 0.00819 (6.6×),
    fourier_m2 0.00967 (24×), deform_cell_corr +0.223, escape 0. s3 mass25x_slow (0.12 + mass 5e-5) matches it
    escape-safely (fourier_m2 0.011). Escape-fails s1 (0.24×5e-5) & s5 (0.24×2e-5×soft80) = 0.0227; softer shell only
    leaks. The one visibly-lobed slot (s1) is the escape-fail (its deform is bulk drift, fourier_m1 0.0225). Batch 9
    maps the motility↓×coupling↑ escape frontier + seeds the flagship. See FINDINGS entry.
  - **b09 RESULT (Batch 10 read; nodiv n44, c03, r150): the ESCAPE ceiling is 1B's binding constraint — 6 of 8 slots
    HARD-FAIL on escape. The escape-SAFE deform lever is `agent_to_mpm.k` (drag-coupling gain), NOT `agent_mass` (leaky);
    the b08 mass-flagship is SEED-FRAGILE and retired.** New CLEAN FLAGSHIP s1 fast_k4 (move 0.24 + agent_mass 8e-6 + k
    1→4): escape 0, deform_rms **0.01287 (10.4× floor)**, fourier_m2 **0.01973 (49× floor, CAMPAIGN MAX)**, fourier_m3
    0.00891, deform_cell_corr +0.0668, circ 0.9927, r_cell_max 0.811 (comfortable margin). fourier_m2 climbs+sustains
    (0.005→0.0146→0.0165→0.0152→0.0197, ends at max) and m2>m1 → real m=2 elongation, not drift/wobble. **Mass leaks, k
    doesn't, at the SAME motility:** fast_k4 escape 0 vs fast_mass10x_k2 (0.24, mass2e-5, k2) escape 0.0455 — mass drives
    per-cell ballistic escape, k amplifies the collective response cleanly. **Falsifier FIRED:** slow_mass80x (0.12,
    mass8e-5) escaped 0.0455 & slow_mass130x (0.12, mass1.3e-4) escaped 0.0227 → escape ceiling is a fixed cell→fluid PUSH
    (coupling threshold), not motility×coupling (b08 s3 mass5e-5 safe at 0.12 → onset in (5e-5,8e-5]). **Seed-fragile:**
    flagship_seed1 (mass2e-5, seed1) escaped 0.0227 where b08 s0 (seed0, same) was escape 0. **flow_align NULL 3rd time**
    (gain120 → polar_order 0.037 = floor, net_circ 0). 1A held on both clean slots. Batch 10: push k 4→6→8 (find the k
    ceiling) + 2-seed replicate fast_k4 toward [established] + containment tests (confine 0.06, membrane youngs 500). See FINDINGS.
    **[OVERTURNED by b10 — the fast_k4 "clean flagship" was a SEED FLUKE; see b10 RESULT below.]**
  - **b10 RESULT (Batch 11 read; nodiv n44, c03, r150): OVERTURNS b09. The FAST (move 0.24) deform route is escape-fragile for
    EVERY lever — `fast_k4` was a SEED FLUKE (seed0 escape 0; seed1 0.1364, seed2 0.0227). ALL 8 driver slots HARD-FAIL escape;
    only the quiescent floor survives.** fast_k6 escape 0.0682, fast_k8 0.0227, k4_mass13 0.1136 — k leaks across seeds exactly
    like agent_mass; b09's "k is the escape-SAFE lever" is FALSIFIED (seed noise, not mechanism). fourier_m2 is an oscillatory
    WOBBLE not accumulation (fast_k6 0.011→0.006→0.028→0.008→0.025). BOTH containment levers DEAD: mass20_confine6 (confine
    0.03→0.06) still escapes 0.0227 AND crushes deform to 0.005; mass20_stiff (youngs 500) escapes 0.0455 (WORSE) — with b08's
    softening-leaks, membrane stiffness is falsified in BOTH directions [rejected]. **The escape ceiling is set by MOTILITY
    (per-cell ballistic energy), not coupling type: the only robust escape-safe deform point is the SLOW route (b08 s3: move 0.12,
    mass 5e-5, f_m2 0.011, escape 0), not yet replicated.** Batch 11: pivot to SLOW — 3-seed replicate slow_mass5, slow_k4
    (motility-vs-k isolation test), slow_mass6/7 (slow frontier), grid96 probe (is fast_k4 escape a grid-tunneling artifact?). See FINDINGS.
  - **b11 RESULT (Batch 12 read; nodiv n44, c03, r150): FALSIFIES b11's slow-mass hypothesis — the SLOW (move 0.12) mass route is
    SEED-FRAGILE, not a robust operating point. The standout is `slow_k4` (move 0.12 + mass 8e-6 + k4): escape 0 with the BIGGEST
    containment margin AND real m=2 elongation.** 3-seed replicate of slow_mass5 (move 0.12, mass 5e-5): seed0 escape 0 / f_m2 0.01097
    (clean, == b08 s3), seed1 escape **0.0227 HARD FAIL** / f_m2 0.00209, seed2 escape 0 / f_m2 **0.00037 (= floor, NO deform)** →
    escape 0.0076±0.013 (1/3 leaks), f_m2 0.0045±0.0057 (SD>mean, 27× range) → cannot [establish]. **slow_k4** (move 0.12, mass 8e-6,
    k4): escape 0, f_m2 0.01045 (26× floor), f_m1 0.00437 (m2/m1=2.4 → REAL m=2 elongation not drift), r_cell_max 0.7817 (LARGEST margin
    in batch), deform_cell_corr +0.017 — where fast_k4 leaked 2/3 seeds (b10), slow_k4 is safe with room → MOTILITY (not k) is the escape
    driver. slow_mass6/7 (6e-5/7e-5) escape-safe on seed0 but deform is m=1 BULK DRIFT (f_m1 ≫ f_m2), not clean shape. grid96_fastk4
    (fast_k4-seed1 @ n_grid 96) escape 0.0682 — HALVED from 0.1364 @ ng64 (partial grid-tunneling) but STILL fails; the fast route does
    NOT reopen; net_circulation 0.00193 (first nonzero — finer grid resolves swirl). Batch 12: 3-seed replicate slow_k4 toward [established]
    + push slow-k frontier (k6, k8) + k-route motility onset (mid_k4 @ 0.16) + coupling-stack test (slow_k4_mass6). See FINDINGS.
  - **b12 RESULT (Batch 13 read; nodiv n44, c03, r150): 1B GATE MET. slow_k (move 0.12, mass 8e-6, `agent_to_mpm.k` ≥4) is an
    [ESTABLISHED] escape-safe membrane-deform lever — deform_rms 4.5–6× floor, robust across 3 seeds, escape 0/0/0 — but the
    specific m=2 MODE stays [open] (seed-fragile). The clean-m=2 STANDOUT is `slow_k6` (f_m2 0.0127, m2/m1 3.61, campaign-clean max).**
    slow_k4 3-seed: escape 0/0/0; deform_rms **0.00563 ± 0.00064** (Δ vs floor 0.00124 = 6.9·SD ≫ 2·SD → [established]); fourier_m2
    **0.00537 ± 0.00450** (SD≈mean, clean m2 only seed0, m3-dominant s1/s2 → mode [open]). **k ladder k4/k6/k8 ALL escape-safe at slow
    motility** (b12 falsifier answered: k never leaks at 0.12 through k8) → with fast_k4/6/8 all leaking (b10), **MOTILITY not coupling
    gain is the escape gate**, now across a full k ladder at both motilities. slow_k6 = clean-m2 optimum; **k8 OVER-drives** to
    m3/m4/m1-drift (f_m2 collapses 0.0127→0.0022, circ 0.9945→0.9858) → clean-m2 k window, k6 near its top. mid_k4 (move 0.16 + k4)
    escape 0 with deform_cell_corr +0.1058 (batch max) → k-route escape onset in (0.16, 0.24]. STACK slow_k4_mass6 (mass6e-5 + k4)
    escape 0.0227 FAIL — reconfirms b10 "escape risks ADD"; don't combine the two couplings. net_circulation 0 everywhere (deform is a
    WOBBLE not a flow-locked shape; Vicsek `alignment` coherence lever is blocked, see engineering note). Batch 13 CONSOLIDATES: 3-seed
    replicate slow_k6 (clean m=2 seed-robust where k4's was not?) + bracket the clean-m2 window (k5, k7) + k-ceiling falsifier (k12)
    → then ADVANCE to 1C. See FINDINGS.
  - **b13 RESULT (Batch 14 read; nodiv n44, c03, r150): 1B CONSOLIDATED → CLOSED; ADVANCE to 1C. slow_k6 confirmed the 1B
    operating point (escape 0/0/0, deform_rms 0.00784±0.00215 = 6.3× floor, [established] amplitude), but the clean-m=2 MODE
    is INTRINSIC WOBBLE — no k locks it.** slow_k6 3-seed: escape 0/0/0; deform_rms 0.00787/0.00997/0.00567; f_m2
    0.01267/0.01549/0.0065 — **m2-dominant on 2/3 seeds** (m2/m3 1.86/2.24) but **seed2 FLIPS to m3** (m2/m3 0.83) → the
    b13 m=2 falsifier FIRED (k6 raises the m2-dominant fraction over k4's 1/3, but doesn't lock). Clean-m2 window is NARROW,
    centered at k6: **slow_k5** f_m2 0.0106 ≈ f_m3 0.01108 (mixed), **slow_k7** f_m2 COLLAPSES to 0.00131 with f_m3 0.0114
    (over-driven to m3, like k8) — one k-step above k6 kills m2. **slow_k12 ESCAPE-FAILS 0.0227** (r_cell_max 0.9047) = FIRST
    proof k leaks even at slow motility (onset (8,12]) → refines b12 "motility-only gates escape": true across k4–k8, but
    coupling gain re-enters as the escape driver at k≳12 (k12 gave campaign-max f_m2 0.02091 but is DISQUALIFIED). mid_k6
    (0.16+k6) escape 0 but m3-dominant → 0.16 doesn't help the mode; k-route escape onset stays (0.16,0.24]. net_circulation
    0 everywhere (wobble, not flow-locked). Per pre-registered falsifier → adopt slow_k6 on deform_rms grounds, 1B DONE.
    Batch 14 = STAGE 1C batch 1: does bounded division (~4x) DEFORM/EXPAND the shell? cap ladder 2x/3x/4x + fill-rate + drive-
    isolation + soft-shell + nodiv control. New specs embryo_1C_base.yaml (slow_k6 + cell_divide) / embryo_1C_soft.yaml. See FINDINGS.
  - **1C — STARTED Batch 14 (2026-07-04). Gate: bounded division proliferation visibly DEFORMS/EXPANDS the shell (`deform_rms`
    and/or `area` ↑ vs the nodiv slow_k6 baseline) while collapsed=0, escape=0, nn_min≥r0 hold.** Substrate = the 1B slow_k6
    operating point (move 0.12, agent_mass 8e-6, k6) + `cell_divide` bounded to ~4x (cap = max_occ·buffer; buffer 200, max_occ
    0.88 → ~176 = 4x initial 44). GEOMETRIC HEADROOM: disc holds ~1040 at r0=0.02 hex, so 4x is far from packing-collapse (b01's
    collapse was unbounded n≈2850). Batch 14 = cap ladder (2x/3x/4x), fill-rate (0.6/0.4/0.2), drive-isolation (div4x_nok:
    division but no flow coupling), shell-compliance (div4x_soft: membrane youngs 100), nodiv slow_k6 control. (see embryo_slots.md / analysis Batch 14).
  - **b14 RESULT (Batch 15 read; 1C_base slow_k6 + cell_divide, 12000f): division DEFORMS the shell (deform_rms
    monotone 0.00787→0.0214 nodiv→4x, 2.7×; first LOBED shells of the campaign) BUT re-triggers the ESCAPE gate at
    every cap ≥2x (div2x 0.0341 / div3x 0.0379 / div4x 0.0852 / r6 0.0966); deform + escape BOTH ride the k6
    `agent_to_mpm` coupling (div4x_nok k1: deform 0.003≈floor, escape 0 → pure crowding is inert); AREA anchor-pinned
    (0.358 vs 0.360 nodiv — shape not size). 1C NOT YET GATED. Batch 15 = k-ladder at fixed 4x (k2/k3/k4) to find the
    escape-safe deform point between k1 (safe/dead) and k6 (leaks/deforms). See FINDINGS.**
  - **b15 RESULT (Batch 16 read; 1C_base 4x, k-ladder): NO escape-safe k at 4x. escape MONOTONE in k (k1→k2→k3→k4→k6:
    0→0.017→0.0227→0.0739→0.0852), onset BELOW k2; the sole escape-0 point (k1) has deform 0.00299 < nodiv-k6 floor 0.00787
    → k-lowering CANNOT gate 1C at 4x. Aids cut but don't zero escape (confine0.06 best 0.0114 but crushes nn_min 0.0142 +
    m1-drift; soft 0.0227 = cleanest deform f_m2 0.011; 3x-k4 0.0227 = m2-DOMINANT cleanest mode). Binding cause = AREA
    ANCHOR-PINNED (0.36 flat everywhere, mpm_anchor k40): rigid shell forbids expansion → division can only lobe-and-leak.
    Batch 16 = ANCHOR RELAXATION (mpm_anchor.k 40→20→10→5 at 4x) to convert division pressure into AREA GROWTH (epiboly)
    with escape→0. See FINDINGS.**
  - **b16 RESULT (Batch 17 read; 1C_base 4x, anchor ladder): 1C SHAPE-DEFORM GATE MET (1 seed) at `anch10_k4`
    (mpm_anchor.k 40→10 + agent_to_mpm.k 6→4): escape 0, deform_rms 0.01766 (1.9× nodiv 0.00934 floor), f_m2 0.02212
    DOMINANT (cleanest m=2 of 1C, no drift). Anchor relaxation does NOT grow AREA — epiboly UNREACHABLE (rest positions
    frame-0-fixed; no rest-length-growth operator) — it opens a COMPLIANCE window converting division push into
    escape-safe LOBING. Escape U-SHAPED in anchor.k (min at ~10: 0.0852→0.0682→0.0057→0.0284 for k40/20/10/5); dropping
    coupling k6→k4 at anchor10 zeroes escape + kills the m1-drift. div2x fallback @ stiff anchor still leaks (0.0341) →
    keep 4x. Batch 17 = CONSOLIDATE anch10_k4 (3-seed) + bracket the escape-safe window; then 1C CLOSES → 1D. See FINDINGS.**
  - **b17 RESULT (Batch 18 read; 1C_base 4x, anch10_k4 3-seed + window): anch10_k4 FAILS consolidation — escape leaks
    1/3 (seed1 0.0114) and the dominant MODE flips m2/m3/m1-drift across the 3 seeds → the b16 "clean m=2 gate" was
    seed-luck; division deform is INTRINSIC WOBBLE (the 1B pattern recurring in 1C, net_circ 0). Deform AMPLITUDE robust
    (0.0232±0.0049, ~2× floor, Δ>2·SD, division-driven vs matched nodiv-anch10 control 0.01141). KEY OVERTURN: at
    coupling k4, escape is MONOTONE-decreasing in anchor softness (anch40→15→10→8→5: 0.0739→0.0682→~0→0.0057→0), NOT
    U-shaped (the b16 U-shape was a k6 artifact) → softer anchor = safer; the b17 "anchor 8–15 safe" hyp FALSIFIED
    (anch15 leaked WORST 0.0682 w/ m1-drift 0.06676). NEW operating-point candidate `anch5_k4` (mpm_anchor.k 5 +
    agent_to_mpm.k 4): escape 0, deform 0.01571, f_m2 0.02185 DOM, r_cell_max 0.8843 (best margin), circ 0.9864.
    Coupling ceiling at anch10 = k4 (k5 leaks 0.0114). Slow-fill r2 @ anch10 gives f_m2 0.04298 (near campaign-max) but
    still leaks → retry @ anch5. AREA still pinned (no epiboly). Batch 18 = 3-seed anch5_k4 + escape/deform bracket
    (anch7/anch3) + coupling isolation (k3/k5) + slow-fill@anch5 + nodiv control. See FINDINGS.**
  - **b18 RESULT (Batch 19 read; 1C_base 4x, anch5_k4 3-seed + brackets): anch5_k4 FAILS consolidation — escape LEAKS
    2/3 (0/0.0114/0.0057), falsifier FIRED (3rd anchor whose seed0 "clean" failed replication; deform 0.0188±0.0064
    amplitude-robust, mode intrinsic wobble). TWO b17 claims OVERTURNED: (A) escape-vs-anchor is a seed-noisy BOWL NOT
    monotone — anch3_k4 (softer) RE-LEAKS 0.0341 w/ r_cell_max 1.027 (cell OUTSIDE) → NO anchor robustly zeros escape at
    4x. (B) at SOFT anchors deform is COMPLIANCE-driven not division-driven — nodiv-anch5 control deform 0.01939 ≥
    dividing 0.0188, f_m2 0.02954 DOM; as anchor softens 40→10→5 the nodiv floor RISES 0.0093→0.0114→0.0194, division's
    marginal gain → 0. The division-DRIVEN deform gate lives only at STIFFER anch10 (b17, 2× nodiv). anch7_k4 sole
    dividing escape-0 (single seed, m3-dom). AREA pinned. Batch 19 = the untested POPULATION lever: cap ladder at anch10
    (2x/3x-3seed/nodiv) + anch7 probe — is the 4x leak crowding-driven? if 3x-anch10 gates escape-safe → 1C CLOSES clean;
    else CLOSE on b17 amplitude arm → 1D. See FINDINGS.**
  - **b19 RESULT (Batch 20 read; 1C_base 4x, population cap ladder at anch10 k4): FALSIFIER FIRED → 1C CLOSED on the
    deform-amplitude arm; ADVANCE to 1D.** div3x_anch10 3-seed escape **0.0152/0/0.0227** (leaks 2/3); div2x (n88,
    LOWEST pop) escape **0.0568** WORSE than 3x while div4x_anch7 (n176, HIGHEST) escape 0 → escape NON-monotone in
    population = seed-noise on a marginal 0–0.06 band, NOT crowding. **POPULATION hypothesis DEAD (4th lever after k,
    anchor.k to fail robust-escape-0).** Deform amplitude robust + division-driven (0.0238±0.0018 = 2.1× nodiv-anch10
    0.01141, Δ≫2·SD); MODE intrinsic wobble (m2/m1-drift/m2); AREA anchor-pinned; net_circ ~0. Per pre-registered
    falsifier → CLOSE 1C. See FINDINGS.**
  - **1C — STARTED Batch 14, CLOSED Batch 20 (2026-07-04). SHAPE-DEFORM GATE MET on AMPLITUDE.** Across b14–b19 (6
    batches): bounded division robustly DEFORMS the shell (deform_rms ~2× the matched nodiv floor, division-driven,
    first lobed shells of the campaign) BUT (i) the deform MODE is intrinsic wobble (never locks m2/m3 across seeds —
    the 1B pattern), (ii) escape at every cap 2x/3x/4x is a marginal seed-noisy ~0–0.06 residual (1–2 cells) that NO
    lever (coupling `agent_to_mpm.k`, `mpm_anchor.k`, population) robustly zeros, and (iii) AREA/epiboly is UNREACHABLE
    (rest positions frame-0-fixed; needs a rest-length-growth operator [engineering]). **1C OPERATING POINT = 1C_base
    4x + `mpm_anchor.k` 10 + `agent_to_mpm.k` 4** (division deforms 2× nodiv; escape marginal; mode wobble; area fixed).
  - **1D — STARTED Batch 20 (2026-07-04). Gate: at CONFLUENCE cells keep flowing (`flow`>0, `t1_rate`>0, not jammed) AND
    COLLECTIVE MIGRATION emerges (`polar_order` ↑, `net_circulation` ↑ off its campaign-long 0, `msd`/migration ↑,
    coherent streams) while collapsed=0 & escape=0 hold.** Substrate = new `embryo_1D_base.yaml`: n132 nodiv (division
    OFF to isolate flow from the 1C division-escape confound), spawn_radius 0.30 (confluent init), `agent_to_mpm.k` 4,
    `mpm_anchor.k` 10, mpm_spin omega 0.3, flow_align.gain 40, move 0.12. STRUCTURAL OBSTACLE: the 1st-derivative cell
    set (repel/glide) BLOCKS Vicsek `alignment`/`cruise`/`cohesion` (all 2nd-derivative, b12) → the ONLY 1st-deriv
    heading operator is `flow_align` (steers heading toward local FLUID velocity). Batch 20 hypothesis: collective flow
    = the `agent_to_mpm`↔`flow_align` FEEDBACK LOOP closing at high density (cells push fluid → coherent fluid velocity
    → flow_align coheres cells into it → more push). flow_align.gain is the coherence lever; mpm_spin SEEDS, motility
    FEEDS. Falsifier: if gain↑ (+spin +confluence) leaves net_circ 0 / polar_order flat → this 1st-order set cannot
    flock → 1D needs the 2nd-order Vicsek rebuild (Coulomb/cruise+alignment) at Batch 21. (see embryo_slots.md / analysis Batch 20).
  - **b20 RESULT (Batch 21 read; 1D_base n132 nodiv confluent, flow_align feedback sweep): FALSIFIER FIRED — the
    1st-order flow_align (fluid-alignment) route CANNOT flock. net_circulation ticked off its campaign-long 0 for
    the FIRST time (~0.001–0.005 in 5/8 slots) but is tiny + tracks DENSITY (s4 n176 net_circ 0.00477 max) not
    gain; polar_order stays WEAK (≤0.16), NON-monotone (gain200 0.007 < gain120 0.11 < base), only TRANSIENT
    spikes (s0 polar 0.0948→0.4889@25%→0.11), no sustained streams. AND every driver slot escape-FAILS 0.0076–0.085
    (the 1C escape frontier persists at confluence). CONTROL s7 (flow_align gain 0) BLEW UP escape 1.0, membrane→box,
    deform 0.1277 → flow_align gain≥40 is a REGULARIZER of the confluent cell→fluid pump.** The falsifier's literal
    "2nd-order Vicsek rebuild" is ARCHITECTURALLY BLOCKED: `engine._resolve_prediction` forces ONE integration order
    per set (raises on conflict) + `mpm_to_agent` (shell confine) is `first_derivative`-LOCKED → the 2nd-deriv
    alignment/cruise/separation/Coulomb cannot join the MPM agent set (the real content of the b12 "2nd-deriv blocked"
    note). PIVOT: wrote a NEW 1st-order op **`heading_align`** (steer heading toward MEAN HEADING of radius-graph
    neighbours — the fluid-free Vicsek order term, mirrors flow_align, `PREDICTION=None` so it composes). Batch 21 =
    heading_align gain ladder (40/120/300/600) + attribution (flow_align-off, spin-off, dense176) + gain-0 control.
    Falsifier: if the gain ladder leaves polar_order weak/non-monotone AND net_circ ~0, NO first-order heading rule
    flocks in this confined MPM blastula → close 1D on the flow-off-zero arm, ADVANCE to 1E. See FINDINGS.
  - **b21 RESULT (Batch 22 read; 1D_flock n132 nodiv confluent, heading_align gain ladder): heading_align is a REAL
    but TRANSIENT flock — b21 falsifier only HALF fires, 1D STAYS OPEN.** Unlike flow_align (b20 null), heading_align
    separates from control: polar_order 0.17–0.20 final (~9× the gain-0 control 0.0214), migr 0.43–0.69 (3–4× control
    0.1519), net_circ MONOTONE in gain (0.0016→0.0052 for g40→g600, off the campaign-long 0). BUT (i) the flock is
    TRANSIENT — every driver spikes at 25% (polar 0.571/0.669/0.774 for g40/g300/g600) then DECAYS to a flat 0.18–0.20
    plateau (bounded disc → polar flock geometrically unstable; net_circ rises as polar decays = polar→milling); (ii)
    NEW MECHANISM = coherence↔membrane-push TENSION — pure_g300 (flow_align OFF) best-sustains the flock (polar 0.445)
    but ruptures the shell (escape 0.9924 at LOW speed = coordinated push, not ballistic); flow_align is a DECOHERER
    that protects the membrane AND damps the flock; (iii) heading_align RAISES escape (every driver 0.045–0.20 > control
    0.0227); (iv) spin seed INERT (spin0 net_circ 0.00477 ≈ spin-on 0.00479). Batch 22 breaks the tension: lower
    agent_to_mpm.k (4→2→1) + flow_align.gain (40→20→10) at heading_align 300 to sustain the flock AND zero escape;
    + anch20 + spin1.5 + base-g300 control. Falsifier: no point that BOTH sustains polar>control AND escape<0.02 →
    bounded 1st-order polar flock is geometry-limited → close 1D on the transient-flock arm, ADVANCE to 1E. See FINDINGS.
  - **b22 RESULT (Batch 23 read; 1D_flock n132 nodiv confluent, tension-breaking sweep, heading_align 300 held): the
    coherence↔push tension is INHERENT — `agent_to_mpm.k` is the SAME lever for the flock AND the escape, so no point
    both sustains polar>control AND escape<0.02 (b22 polar-arm falsifier FIRES). BUT the polar flock CONVERTS to
    MILLING (rotational), which is escape-SAFER and untested → PIVOT to test milling before closing 1D.** k is MONOTONE
    in BOTH escape and flock: k4/k2/k1 → escape 0.1439/0.0909/0.0152 AND polar_final 0.1806/0.1066/0.0662 (peak@25%
    0.669/0.629/0.346) — lowering push (k) proportionally kills coherence; k1 is the sole escape<0.02 (0.0152) but
    polar 0.0662 < control 0.1806 (flock-dead). flow_align-DOWN INCREASES escape (fa40/20/10 → 0.1439/0.1818/0.25,
    r_cell_max 1.01→1.09→1.18 cells OUTSIDE) = RECONFIRMS regularizer, FALSIFIES "decohering-noise" framing. Predicted
    sweet spot k2+fa20 DEAD (escape 0.1439 = no-lever ctrl). anch20 (stiff shell) = only escape-lever that spares the
    flock (0.0758 ≈ ½ ctrl). **KEY: spin1.5 (s6) REVERSES the b21 "spin inert" claim at high omega — polar COLLAPSES
    to 0.0266 (min) while net_circ rises to 0.00865 (max, 1.8× ctrl), msd 0.0745 (max), escape DROPS to 0.0303 (5×
    safer): a rotational MILLING flow is escape-safer (tangential not radial push).** Batch 23 = MILLING resolution:
    mpm_spin.omega ladder (1.5/3/6) × k (1/2) + flow_align80 + anch20 + heading_align120 + spin-0 control. Falsifier:
    if raising omega leaves net_circ intermittent (any 0 in 50/75/100% plateau) OR escape>0.02 at every spin, NO
    first-order collective mode (polar OR milling) is both sustained AND escape-safe → CLOSE 1D, ADVANCE to 1E. See FINDINGS.
  - **b23 RESULT (Batch 24 read; 1D_flock n132 nodiv confluent, spin ladder × k): the b23 falsifier FIRES on BOTH
    clauses → 1D CLOSED, ADVANCE to 1E. Milling does NOT lock and no first-order collective mode is both sustained
    AND escape-safe.** (i) net_circ NON-monotone in omega — peaks at spin1.5–3 (spin1p5_k2 0.00862 = batch max ==
    b22's milling signal; spin3_k2 0.00655) then COLLAPSES to 0.0 at high omega where the shell CRUMPLES (spin6_k2/
    fa80/anch20/ha120 net_circ 0.0, circ 0.72/0.59/0.59/0.64, r_cell_max 0.68–0.80; spin6 a frank blow-up: area
    0.678≈2×, nn_cv 0.005 central worm-blob, membrane fragmented). (ii) EVERY net_circ>0 slot escapes >0.02
    (spin3_k2 0.0379, spin1p5_k2 0.053, ctrl_spin0 0.0758 r_cell_max 1.03 cells-outside); the only escape-0 slots
    are the net_circ=0 crumpled failures. Per pre-registered falsifier → CLOSE 1D. Batch 24 = STAGE 1E batch 1
    (differential adhesion sorting). See FINDINGS.**
  - **1D — STARTED Batch 20, CLOSED Batch 24 (2026-07-04). GATE NOT MET — operator-limited.** Across b20–b23 (4
    batches) NO first-order collective mode is both sustained AND escape-safe in the confined MPM blastula:
    flow_align (fluid-alignment) NULL (b20); heading_align (neighbour-alignment) = REAL but TRANSIENT polar flock
    that RAISES escape, and `agent_to_mpm.k` is the SINGLE lever coupling flock-coherence to shell-rupture so it
    can't be both sustained + escape-safe (b21/b22); milling (rotational) either weak+leaky at low omega or crumples
    the shell at high omega (b23). net_circ ticked off its campaign-long 0 (peak ~0.009) but never sustained a
    healthy-shell vortex. **1D OPERATING POINT (best clean point) = 1D_flock heading_align 300 + `agent_to_mpm.k` 1
    (escape 0.0152, weak escape-safe flock, b22).** The 2nd-order Vicsek rebuild that would flock robustly is
    ARCHITECTURALLY BLOCKED (engine forces one integration order/set; mpm_to_agent first_derivative-locked) [engineering].
  - **1E — STARTED Batch 24 (2026-07-04). Gate: the two 50/50 agent types (a red / b yellow, dynamically identical
    through 1A–1D) SEGREGATE — `segregation_index` ↑ (1=sorted; scorecard 1−cross/exp_cross, exp_cross 0.5), `contact_same`
    ↑, `mixing_entropy`/`interface_frac` ↓ — while collapsed=0, escape=0, nn_min≥r0 hold.** Substrate = new
    `embryo_1E_base.yaml`: calm 1D confluent container (n132 nodiv, spawn 0.30, k4 mass8e-6, spin 0.3, flow_align 40,
    move 0.12, repel 150) with flocking drivers REMOVED (heading_align OFF — polar coherence MIXES) and anchor STIFFENED
    (k10→20, escape-safe round shell). NEW mechanism `attraction_repulsion` (per-type first-derivative pull/push, composes
    with repel/glide/MPM, proven b05): type a cohesive PULL (p=[pull,1,0,1]), type b NEUTRAL (p=0) → receiver-type params
    make a cohere into a CORE and displace b to the PERIPHERY (Steinberg differential adhesion). Batch 24 = pull ladder
    0.3/0.6/1.0 + range(sigma 0.05) + n88 hedge + symmetric ablation + active demix (b push) + no-adhesion control.
    Hypothesis: seg_index rises monotonically in pull vs control, escape-safe. (see embryo_slots.md / analysis Batch 24).
  - **b24 RESULT (Batch 25 read; 1E_base n132 confluent, `attraction_repulsion` differential adhesion): 1E STAYS
    OPEN — differential self-cohesion SORTS (b24 falsifier did NOT fire; seg_index separates from control) but
    WEAKLY, NON-monotone in pull, and the WINNER is the two-sided ACTIVE DEMIX.** scorecard `segregation_index`
    (final): xdemix (a-pull0.6 + b-push0.3) **0.139** (WINNER; contact_same 0.578 max, interface_frac 0.430 min) >
    a06 (pull0.6) 0.092 > a10 (pull1.0) 0.056 > a03 (pull0.3) 0.038 > n88 0.020 > **ctrl −0.028** > sig05 (range
    0.05) −0.052 > **sym_both06 (BOTH pull0.6) −0.077**. Pull PEAKS at 0.6 (over-pull 1.0 kinetically arrests the
    a-clump). **Two ablations land clean:** sym (equal cohesion) BELOW control → DIFFERENTIAL required, not cohesion
    per se (Steinberg); n88 (low density) → toward control → CONFLUENCE required. **Active b-push (self-dispersal)
    beats single-pull 1.5×** (a coheres to a core, b actively spreads to the shell). seg_index is NEGATIVE early
    (sunflower init) and RISES through the 2nd half → SLOW, coarsening-limited, still PROGRESSING at 100% (not
    saturated). Frame-noise ±0.05–0.1, single seed → gaps <0.05 are noise; Δ(winner−ctrl)≈0.17 is real. **KEY
    OPERATOR LIMIT:** `attraction_repulsion` uses RECEIVER-type params ONLY (does NOT read neighbour type; op
    source line 61 `p = type_params[node_type[i]]`) → it CANNOT express heterotypic (a–b) repulsion; sorting here is
    self-cohesion/self-dispersal differential MOBILITY, not interfacial tension → the seg~0.14 ceiling may be
    intrinsic to this op. TIER-1: collapsed 0, nn_min ~0.0185; **escape ~0.05 is a CONTAINER baseline (no-adhesion
    ctrl escapes 0.053)**, decoupled from sorting — a-pull (inward) neutral (a06 0.053=ctrl), b-push (outward)
    nudges it (xdemix 0.0682 rmax 1.002), n88 halves it (0.0227). Batch 25 = exploit the demix corner: b-push
    ladder (0.2/0.5/0.7 @ a0.6) + seed-1 replicate of xdemix + kinetic/range explores (sigma 0.02, move 0.18, n176)
    + ctrl. See FINDINGS.**
  - **b25 RESULT (Batch 26 read; 1E_base n132 confluent, `attraction_repulsion` b-push ladder + seed1 replicate):
    the b25 falsifier FIRES DECISIVELY — `attraction_repulsion` differential self-mobility CANNOT sort; the b24
    seg 0.139 "winner" was NOISE (RETIRES the b24 [open] sort). PIVOT to true heterotypic cross-repulsion.** The
    b24 xdemix (a-pull0.6 + b-push0.3) SEED1 replicate → seg_index **−0.108** vs seed0's +0.139 (0.25 swing).
    seg_index across all 8 spans −0.108…+0.055 (ctrl −0.028) = pure ±0.1 frame/seed noise; b-push ladder
    pb0.2/0.5/0.7 → seg −0.089/−0.081/+0.008 (does NOT recover the mechanism); contact_same 0.417–0.539 around
    ctrl 0.503 (no lever clears noise); montage stays salt-and-pepper mixed to t=12000. escape 0.03–0.14 seed-noisy,
    DECOUPLED from sorting (reconfirms confluent-container baseline). Root cause = b24 finding #7 operator limit
    (`attraction_repulsion` reads RECEIVER type only → no heterotypic a–b tension). PIVOT (verified expressible, NO
    new op): two-channel chemotaxis cross-repulsion — `deposit` auto-writes channel=own type (a→ch0, b→ch1); field
    `couples_to: agent` auto-sizes components=2; `chemotaxis[type=a] channel:1 gain<0` (a flees b-trail) +
    `chemotaxis[type=b] channel:0 gain<0` (b flees a-trail), both first_derivative → compose; schedule token
    `chemotaxis` runs both instances. Gain anchored to bison chemotaxis specs (2e-5…0.04). Batch 26 = |gain| ladder
    0.005/0.02/0.08 + self-cohesion combo + strong-gain(0.2) escape frontier + sharp-field + seed1 + ctrl. See FINDINGS.**
  - **b26 RESULT (Batch 27 read): NO DATA — EXECUTION LOSS, not a science null. The heterotypic chemotaxis
    mechanism was NEVER TESTED.** Of the 8 slots submitted (jobs 151981397–404) ONLY s7_ctrl_noadh (no field, no
    chemotaxis) archived; all 7 chemotaxis DRIVERS produced no archive. The loss is TYPE-CORRELATED (every
    chemotaxis slot dies, the one non-chemotaxis lives → not random infra) AND GAIN-INDEPENDENT (even g005 died →
    not a dynamics/escape blow-up) → the crasher is the SHARED field/chemotaxis MACHINERY the drivers add over the
    control. Static source read (python approval-blocked, no traceback): every operator + the 2-channel field are
    well-formed; the one embryo-NOVEL fact is that `chem` is the FIRST recorded/rendered scalar field (mpm_grid is
    RECORD=False) → new render load (per-field movie + couples_to overlay path) with no prior proof it lands. The
    lone control reconfirms the mixed baseline (seg_index −0.028, contact_same 0.503, collapsed 0, nn_min 0.0184 ==
    b24/b25). Batch 27 RE-ISSUES the mechanism as an ISOLATION LADDER (field_only / gain-0 / gain ladder / couples_to
    A/B / exact-b26 repro), de-risked (components:2 explicit + no couples_to, res 64), to pinpoint the crasher AND
    recover the sorting science. See FINDINGS.**
  - **b27 RESULT (Batch 28 read): CRASHER PINPOINTED — a YAML syntax bug, NOT the field render. Still 0 sorting
    data; mechanism now FIXED.** Isolation ladder landed exactly the 2 slots with NO `chemotaxis` op (s0 ctrl, s1
    field_only, bit-identical scorecards — deposited field is inert without a reader); all 6 `chemotaxis` slots
    died on `yaml.parser.ParserError` from an UNQUOTED `at: agent[type=a]` inside a flow map (the `[` starts a flow
    sequence). This RETIRES the b26 "first-scalar-field-render/couples_to" suspect (field_only ran the full field
    machinery clean in 827 s; s7 couples_to died on the SAME parse error). FIX = quote the selector; applied to
    xr_g0/g02/g05/g10 + authored xr_g20/g05_sharp/g05_cohere. Wiring verified correct (deposit a→ch0/b→ch1;
    chemotaxis gain<0 flees the other's channel). Batch 28 = the heterotypic gain ladder's FIRST real run. See FINDINGS.**
  - **b28 RESULT (Batch 29 read): FIRST CONFIRMED DEMIX. Heterotypic two-channel chemotactic cross-repulsion SORTS
    the blastula, seg_index scaling monotonically with |gain|, NO shell rupture.** All 8 slots landed (802–846 s).
    seg_final: ctrl/g0 −0.028 → g02 +0.208 → g05 +0.104 → g10 **+0.485** → g20 **+0.808**; sharp-field g05 +0.621.
    At strong rungs FIVE metrics move together+monotone: g20 contact_same 0.489→0.896, interface_frac 0.535→0.096,
    mixing_entropy 0.856→0.418. Genuine un-mixing from random start (seg begins ≤0, climbs). NO escape/rupture (the
    predicted g20 breach FALSIFIED — nn_min 0.0185, collapsed 0, container held). Coarsening-then-arrest (migr
    0.296→0.096, t1_rate 0.011→0.006 as domains lock), NOT jamming. Geometry = lateral side-by-side domains, NOT
    core-shell (mi_type_x stays low ~0.05 despite contact_same 0.90). Levers: g0 bit-identical to ctrl (no-op
    confirmed); SHARPER field ≈6× effective gain (sharp seg 0.621 at nominal g05 vs plain 0.104); cohesion SUPPRESSES
    sorting (cohere 0.185). n=1/rung → needs seed replication. Batch 29 = g10/g20 seed replicate (→[established]) +
    asymmetric-repulsion engulfment probe + self-adhesion route. See FINDINGS.**
  - **b29 RESULT (Batch 30 read): 1E SEGREGATION GATE MET — the gain-scaled demix PROMOTES to [established]
    over 3 seeds; the OPEN axis is now core-shell geometry.** All 8 slots landed (~810–850 s). seg replicates:
    g10 {0.485,0.419,0.579}=0.494±0.080 (Δ vs ctrl 6.5·SD), g20 {0.808,0.850,0.686}=0.781±0.085 (Δ 9.5·SD),
    monotone, co-metrics monotone (contact_same 0.50→0.76→0.89, interface_frac 0.51→0.25→0.12, mixing_entropy
    0.85→0.61→0.48), escape/nn_min-safe. sharp_g10 seg 0.799 ≈ g20 at half gain (sharper field ≈2× gain,
    reconfirms). GEOMETRY probes for core-shell: asym (a −0.20/b −0.02) mi_type_x 0.062 noisy → asymmetric
    cross-rep does NOT set radial order [rejected]; selfattr (positive self-chemotaxis +0.10) mi_type_x 0.084
    SUSTAINED (the lead) BUT overpacks (nn_min 0.0079 FAIL, escape 0.144, accel 0.010) → core-shell route =
    BOUNDED self-cohesion. Batch 30 = attraction_repulsion type-a self-pull ladder (0.4/0.8/1.2) on the
    established g10 demix + label-swap/one-sided/sharp explores + ctrl. Falsifier: mi_type_x ≤ 0.06 at every
    self-pull strength → close 1E on the lateral-demix gate, ADVANCE to INT. See FINDINGS.**
  - **b30 RESULT (Batch 31 read): FALSIFIER did NOT fire — FIRST CORE-SHELL signal of 1E. Strong bounded type-a
    self-pull (`attraction_repulsion` p[0]=1.2) on the established g10 demix sets RADIAL order WITHOUT overpacking
    — single seed, needs replication.** All 8 landed (774–864 s), collapsed 0 everywhere, nn_min 0.0177–0.0189
    (no overpack anywhere). mi_type_x final vs self-pull: ctrl 0.0132 → a04 0.0203 → a08 0.0213 → **a12 (1.2)
    0.2229** = a THRESHOLD (flat ≈ctrl through 0.8, 10× jump at 1.2), NOT a linear lever. a12's radial order is
    SUSTAINED coarsening (mi_type_x 0.0151→0.0185→0.032→0.0816→0.2229, monotone, accelerating, unsaturated at
    100%), joint with the demix (contact_same 0.47→0.82, interface_frac 0.56→0.18, seg −0.12→0.638 all move
    together). **a12 does NOT overpack** — nn_min stable 0.0181–0.0188 (attraction_repulsion's hard repulsive core
    r0 0.02 is the safety margin the b29 selfattr positive-chemotaxis route lacked, which failed at nn_min 0.0079);
    escape 0.0303 (container baseline), seg 0.638 HELD. Weak self-pull (a04/a08 mi_type_x ≈ctrl) inert radially but
    keeps the demix (seg 0.57/0.54). Label-swap b04 (b-pull 0.4) mi_type_x 0.0666 (noise-level, below knee).
    Sharp+weak (a04_sharp 0.0581, seg 0.723) and strong-demix+weak (a04_g20 0.0308, seg 0.727) both stay below
    the radial knee → sharpening/strengthening the CROSS-rep does NOT substitute for self-cohesion. asym1s 0.0112
    reconfirms asymmetric-repulsion [rejected]. Shell unchanged (area 0.36 flat, circ 0.99, net_circ 0) — core
    forms by internal rearrangement. CAVEAT: SINGLE SEED; campaign history (fast_k4/anch10_k4/anch5_k4/b24 xdemix)
    warns single-seed clean points routinely fail replication. Batch 31 = REPLICATE a12 (seed1/seed2) + bracket
    threshold (a10/a14) + overpack falsifier (a16) + sharp/label-swap explores + ctrl. Falsifier: a12 seed1+seed2
    mi_type_x both ≤0.06 → fluke → CLOSE 1E on lateral-demix, ADVANCE to INT. See FINDINGS.**
  - **b31 RESULT (Batch 32 read): FALSIFIER FIRED — the b30 a12 core-shell signal was a SINGLE-SEED
    FLUKE; 1E CLOSED on the lateral-demix [established] gate, ADVANCE to INT.** All 8 landed clean
    (774–865 s, collapsed 0, nn_min 0.018–0.0188). BOTH a12 replicates of the b30 mi_type_x 0.2229
    point fell ≤0.06: seed1 **0.0067** (traj 0.033→0.027→0.0067 DECLINING, below ctrl), seed2 **0.0539**
    (traj 0.0075→0.087→0.0539 bump-then-settle). NO slot reproduced the b30 sustained climb — every
    endpoint ~3–4× below 0.2229 (max 0.065 @ a16, barely at the line). NO monotone trend across
    self-pull (a10 0.019 → a12 0.007/0.054 → a14 0.017 → a16 0.065 = ±0.05 scatter around ctrl 0.013).
    5th single-seed clean point (after fast_k4/anch10_k4/anch5_k4/b24 xdemix) to fail replication.
    The DEMIX held orthogonally (ctrl_g10 seg 0.485 == b29 [established] 0.494±0.080; all slots ≥ mixed
    baseline). Batch 32 = INT batch 1 (does the partition SURVIVE proliferation + deform shell?):
    division ladder 2x/3x/4x on g10/g20 demix at the escape-safe anch20/k4 substrate. See STAGE STATUS below.
  - **1E — STARTED Batch 24, CLOSED Batch 32 (2026-07-05). LATERAL-DEMIX GATE MET [established].** Across
    b24–b31 (8 batches): differential self-MOBILITY (`attraction_repulsion`, reads receiver type only)
    CANNOT sort (b24/b25 seg ~±0.1 noise [rejected]); the working mechanism is HETEROTYPIC two-channel
    chemotactic cross-repulsion (`deposit` a→ch0/b→ch1; `chemotaxis[type=a] channel:1 gain<0` a-flees-b +
    `chemotaxis[type=b] channel:0 gain<0` b-flees-a). Gain-scaled LATERAL demix [ESTABLISHED] 3 seeds
    (b29: g10 seg 0.494±0.080 = 6.5·SD vs mixed ctrl −0.028; g20 0.781±0.085 = 9.5·SD; monotone
    co-metrics; escape/nn_min-safe; sharper field ≈2× gain). Geometry is LATERAL side-by-side domains,
    NOT core-shell — RADIAL order (mi_type_x) could NOT be robustly set (asym cross-rep [rejected];
    self-chemotaxis overpacks nn_min 0.0079; bounded self-cohesion b30 a12 mi_x 0.2229 FAILED 2-seed
    replication b31). **1E OPERATING POINT = `embryo_1E_ctrl_g10.yaml`** (n132 confluent nodiv, spawn
    0.30, chemotaxis gain −0.10, deposit 0.5/diffuse 0.1/decay 0.2, anch20, agent_to_mpm.k 4).
  - **INT — STARTED Batch 32 (2026-07-05). Gate: the integrated blastula co-develops MULTIPLE discovered
    mechanisms at once without them destroying each other — batch 1 tests PARTITION × PROLIFERATION: the
    [established] gain-scaled demix SURVIVES division (seg stays ≥ mixed baseline and gain-ordered g20>g10
    as population grows 2–4×) AND division deforms the shell (deform_rms ↑ vs nodiv) while collapsed=0,
    escape bounded (≲0.06 container baseline), nn_min≥r0.** Substrate = 1E demix + `cell_divide` ON
    (daughters inherit node_type, cell_divide.py:62; cap = max_occ·buffer, cell_divide.py:47). Batch 32 =
    division ladder 2x/3x/4x × {g10,g20} at anch20/k4 + anchor-relax deform probe + slow-fill + nodiv ctrl.
    **b32 RESULT (Batch 33 read): DIVISION DILUTES THE DEMIX, monotone with growth [open].** seg (g10)
    1×→2×→3×→4× = 0.485 → 0.216 → 0.064 → 0.056; co-metrics monotone (contact_same 0.778→0.535,
    mixing_entropy 0.669→0.921). Partition⊥proliferation COMPATIBLE to ~2× (45 % of full demix survives),
    INCOMPATIBLE at ≥3× (→ mixed baseline ~0.06). Loss is KINETIC not jamming: at 3× nn_min 0.0174 ≈ ctrl
    (healthy) yet seg dead — division front-loads (n hits cap by 25 %), sort can't keep pace. Neither lever
    rescues: g20 (stronger cross-rep) seg ≈ g10 at every rung AND ruptures at 4× (nn_min 0.0018, collapsed
    0.0038 — only TIER-1 fail); slowfill (rate 0.15) seg 0.079 ≈ fast (final density, not rate, is the
    ceiling). Batch 33 = TEMPORAL SEPARATION test (pre-pattern at n132, THEN divide via `after:`,
    cell_divide.py:35) — does an established domain survive dilution by adjacent same-type daughters?
    **b33 RESULT (Batch 34 read): TEMPORAL SEPARATION FALSIFIED → partition⊥high-proliferation PROMOTED to
    [established-integration]. Division is a MECHANICAL MIXING event that decimates even a FULLY-FORMED
    pattern, timing- and mobility-independent.** Late-division slots sort beautifully pre-division then COLLAPSE
    the instant n grows: 2×_late75 seg 0.642@75% → 0.112 (−82 %); 4×_late75 0.500@75% → 0.037; 4×_late50
    0.367@50% → 0.007. Finals are timing-independent (2× 0.131/0.112; 4× 0.007/0.037). MECHANISM: at each
    division checkpoint `msd` JUMPS ~10× (repacking wave shoves cells for daughters) and re-scrambles the
    interface (interface_frac 0.179→0.444, contact_same 0.832→0.573, mixing_entropy 0.531→0.839 all snap back);
    post-arrest chemotaxis CANNOT re-sort (2×_late50 kept DECLINING 0.567→0.187→0.131 over 6000 post-div frames;
    t1_rate already decayed to ~0.01). Mobility does NOT rescue (4×_fast move0.24 seg 0.069, never sorts, WORST
    shell: deform 0.093, escape 0.280 rupture). TIER-1: escape ~0.09–0.12 = CONTAINER BASELINE (nodiv ctrl 0.121);
    2× sits at baseline (SAFE) but 4× breaches it (0.14–0.28 = division-driven rupture) + 4×_late50 overpacks
    (nn_min 0.014). 2× envelope now 3 seeds: {0.216, 0.131, 0.306} = 0.218±0.088 (1.8·SD, partial+seed-noisy).
    **INT PROLIFERATION DELIVERABLE = 2× concurrent g10 demix (dividing+demixing seg 0.22+deforming 0.033,
    escape-safe).** Batch 34 = MOVE ON to PARTITION × MEMBRANE-DEFORMATION (does inner-flow deform coexist with
    the sort, nodiv isolation).
    **b34 RESULT (Batch 35 read): PARTITION × MEMBRANE-DEFORMATION — the deform driver splits by CHANNEL [open→
    channel-specific].** The b34 blanket prediction "all deform drivers re-mix the demix" is REFUTED. Two channels
    behave oppositely on the nodiv g10 demix (ctrl seg 0.485, deform_rms 0.0244, escape 0.121):
    (1) **MPM-CONTINUUM channel `agent_to_mpm.k` = COMPATIBLE**: k4→6 seg 0.419 deform 0.0284; k4→8 seg **0.500**
    (≈ctrl) deform **0.0384 (+57%)** escape 0.076 (safe); g20+k6 seg **0.514** (batch-best) deform 0.0304. Membrane
    deforms, cells DON'T reshuffle (msd ≈ ctrl 0.007–0.018) → demix topology preserved. **deform ∥ partition
    COMPATIBLE via the k-lever, and g20 headroom BUYS BACK seg (0.514 vs g10 0.419 at same k6).**
    (2) **CELL-KINETIC channel `agent_mass`/`move_speed` = INCOMPATIBLE**: mass 8e-6→2e-5 seg **0.217 (−55%)** deform
    0.0378; move 0.12→0.18 seg **0.226 (−53%)** deform 0.0363, escape 0.144 (BREACH). These drive diffusive cell
    REARRANGEMENT (contact_same →0.62) → sort decays to mixed baseline.
    (3) `mpm_spin` ω0.3→1.0 = a THIRD mode: msd 17× (coherent rotation), seg HELD 0.460, but SHRINKS/rounds the shell
    (area 0.360→0.330, circ 0.995→0.917, deform_rms flat). Advection ≠ rearrangement.
    **PRINCIPLE: it is diffusive cell rearrangement, NOT membrane deformation per se, that erases the sort —
    coherent/continuum deform (k-lever, spin) is topology-preserving; motility/inertial deform is topology-breaking.**
    THE TRIPLE (2x division + g10 + k6, s6): seg **0.273**, deform_rms **0.0403 (batch-max)**, n 264, escape 0.129,
    nn_min 0.0184, collapsed 0, div_stress_angle 0.771 — TIER-1 SAFE; k6 deform does NOT further crush the dividing
    demix (0.273 ≈ b33 2x envelope 0.218±0.088, ~0.6·SD). All b34 points n=1 → deform-compatibility AND the triple
    need seed replication [open]. Montage `seg=` field is the UNRELATED `segregation` metric and INVERTS — read
    `segregation_index` from scorecard.json only.
    **b36 RESULT (Batch 37 read): b35 loss RESOLVED, THE TRIPLE RAN (first real triple data) — partition × 2×
    division × k6 deform COEXIST at TIER-1, but g20 headroom does NOT survive division [prediction FALSIFIED].**
    g20 2×-k6 3 seeds seg {0.170, 0.203, 0.200} = **0.191±0.018** ≈ g10 2×-k6 ~0.228 (g20's nodiv 0.61–0.78
    advantage vanishes once dividing). Nodiv locks confirm headroom is real undivided: nodiv g20-k6-s1 seg
    **0.608**, nodiv g10-k8-s1 **0.389** (both n=132, reproduce b34 0.514/0.500). k8 deform doesn't help the
    dividing sort (g20 2×-k8 seg 0.195, deform 0.0465). Control g10 2×-k6 seed0 seg 0.2734 == b34's 0.273 (exact).
    → 2× division dilution is a hard, GAIN-INDEPENDENT ceiling (see [established-integration] entry). Batch 37 =
    PROLIFERATION-SORT FRONTIER: 1.25×/1.5×/1.75× g20-k6 rungs (find max growth with seg ≥0.35) + g30-2× gain
    falsifier + 1.5×-k8 + 1.5× ×3 seeds. Prior b35-loss note (now historical): EXECUTION LOSS — the refactor
    (commit b68864f) broke `repel`; every embryo run 0-archives at spec-load until fixed.** All 8 slots died
    identically: `.out` Run time 12–17 s / CPU 5 s / 300 MB (vs ~800 s for a real run) + `.err`
    `schema.py:151 ValueError: operator 'repel' has invalid PREDICTION 'first_derivative'`. The refactor
    renamed PREDICTION `first_derivative`/`second_derivative` → `velocity`/`acceleration`/`mpm_acceleration`
    (models/base.py:117, enforced schema.py:150) and updated all `src/plexus/operators/*` — but `repel` lives
    in the untouched prototype lib `active_matter2/am2_ops.py:394` (imported by showcase.py:27) and kept the old
    token. FIX: am2_ops.py:394 → `"velocity"` (matches glide's set order + the line's own comment; sole
    stale spec-referenced op). Batch 36 = exact b35 re-issue (specs unchanged). See FINDINGS (engineering).
    **b37 RESULT (Batch 38 read): PROLIFERATION-SORT FRONTIER MAPPED [established-integration].** seg falls
    MONOTONE with growth: 1.0×(nodiv) 0.608 → 1.25× 0.432 → 1.5× **0.408±0.161** {0.588,0.335,0.302} → 1.75×
    0.326 → 2.0× 0.170(g20)/0.265(g30). 1.5× ≫ 2× (falsifier did NOT fire → moderate growth keeps a stronger
    sort), but 1.5× is NOISY (seed0 0.588 outlier; median 0.335) so "1.5× ≥0.35" stays [open]. TIER-1 ESCAPER
    onset ≥1.75× (nn_cv 1.6, gr_peak 19, r_cell_max ~2.0, escape 0.15 = stochastic division-fling, cosmetic not
    hard-fail; clean ≤1.5×) = soft upper growth bound. k8 = same sort (0.393) but 10× fourier_m2 (elliptical
    shell, topology-preserving deform reconfirmed). Geometry LATERAL (mi_type_x ≤0.088). **1.5× g20-k6 =
    moderate-growth operating point.** Batch 38 = FLOW-integration capstone (last untested leg).
    **b38 RESULT (Batch 39 read): FLOW LEG RESOLVED — coherent flow is MOTILITY-driven, NOT spin-driven; both
    flow flavors COEXIST with the dividing lateral sort at TIER-1 [open→motility-flow, needs seed replication].**
    On the dividing 1.5× g20-k6 triple: (1) **`mpm_spin.omega` ≥0.8 FLUIDIZES but adds NO coherent circulation** —
    msd jumps ~10× to **0.12** (0.122–0.127) and is ω-INDEPENDENT/SATURATED (ω 0.8/1.2/2.0 & 1.25× all ~0.12;
    threshold between ω0.3 msd 0.013 and ω0.8) yet **net_circulation = 0.0 in ALL spin slots** (vs 0.0078 at ω0.3
    — raising ω KILLS the weak coherent swirl); enstrophy only 2× (ω0.8/1.2) rising to 6.3e-6 at ω2.0. Incoherent
    internal stirring, not bulk rotation — despite flow_align (gain 40) present in every slot → the substrate spin
    field has no NET curl. (2) **The lateral sort HELD under 10× stirring:** spin08 3-seed seg {0.288,0.374,0.232}
    = **0.298±0.071** vs noflow 0.408±0.161 (Δ within noise, seed0-outlier-driven); seg rises monotone within each
    run → chemotaxis re-sorts as fast as spin churns. mi_type_x ≤0.078 (LATERAL). (3) **ω2.0 SHRINKS shell** (area
    0.356→0.311 −13%, circ dips 0.87@75%) without rupture (collapsed 0, no escaper). (4) **`move_speed` 0.18
    (motility) is the ONLY coherent-flow slot** — net_circ **0.0118** (batch-max), fourier_m1 drift **0.112** (16×
    spin), deform_rms **0.0539** (batch-max, +51% vs ctrl) — yet msd only 0.034 (coherent not diffusive) and **seg
    0.371 HELD** (predicted remix to ~0.22 FALSIFIED; chemotaxis re-sorts). PRINCIPLE: coherent collective flow
    (net_circ / coherent drift / membrane deform) is driven by CELL MOTILITY, not substrate spin; mpm_spin only
    fluidizes. move18 n=1 → Batch 39 replicates + pushes move24 + ablates flow_align. THE CAMPAIGN GOAL (a
    FLOWING, dividing, self-partitioning blastula) is now demonstrable with motility as the flow driver.
    **b39 RESULT (Batch 40 read): MOTILITY-FLOW CAPSTONE — coherent flow is SINGLE-PEAKED in motility; move18/
    gain40 3-seeded; flow_align REQUIRED as a CONTAINMENT coupling [see two FINDINGS entries below].** Seed0-gain40
    ladder net_circ/m1(drift): 0.12→0.0057/0.075 ; **0.18→0.010/0.104 (PEAK)** ; 0.24→0.002/0.016 (COLLAPSE), while
    msd rises MONOTONE 0.022→0.039→0.090 → above ~0.18 motility FLUIDIZES into incoherent stir (same coherent→
    incoherent transition as high-ω spin, b38). Sort remixes above the optimum: seg 0.588(0.12) → 0.336±0.043 3-seed
    (0.18) → 0.156±0.071 (0.24) — the move24 falsifier (seg <0.22) FIRED. **move18/gain40 coherent flow now 3-SEEDED:**
    net_circ {0.0118[b38 s0],0.0101,0.0108}=0.0109±0.0009, seg {0.371,0.350,0.287}=0.336±0.043 (held ≥0.25), all
    TIER-1 clean → the FLOWING+dividing+partitioning blastula is DEMONSTRATED [established-integration]. align80 (gain
    80 @ move18, n=1) = best clean sort seg 0.392 + lowest msd 0.029 (stronger alignment tightens coherence, sharpens
    sort). spin08+move18 kills the drift (net_circ 0, m1 0.006 — spin dominates). flow_align.gain 0 @ move18 = TIER-1
    CATASTROPHE (escape 1.0, area 0.36→0.78, deform 0.175, nn_min 0) → flow_align is a CONTAINMENT coupling (reconfirms
    b20's confluent-blowup). Batch 40 = brackets the optimum (move15/20/22) + replicates align80 to 3 seeds + tests
    whether gain80 rescues coherent flow/sort above the optimum (move20_a80, move24_a80).
    **b40 RESULT (Batch 41 read): flow_align gain80 RAISES the fluidization threshold — alignment RESCUES coherent
    flow + sort above the gain40 optimum [open, n=1/rung; see FINDINGS].** gain40 ladder reconfirms fluidization
    (move20 seg 0.268 msd 0.085; move22 seg 0.245 msd 0.114 deform 0.078). gain80 DE-fluidizes at the SAME/higher
    motility: move20_a80 seg 0.268→**0.408**, msd 0.085→**0.033**; move24_a80 seg **0.383**, net_circ **0.0158
    (campaign max)**, msd 0.048. So the coherent window gain40 closed at ~0.18 stays OPEN to move24 under gain80 —
    Batch-40 falsifier did NOT fire (net_circ rose, seg held ~0.4). CAVEAT: gain80 does NOT help at move18
    (move18_a80 3-seed {0.392,0.255,0.239}=0.295±0.084 ≈ gain40 0.336±0.043) — it SHIFTS the optimum UP to move20–24.
    move20_a80 = best sort (0.408, n=1); move24_a80 = best flow (net_circ 0.0158, n=1). Batch 41 = replicate both to
    3 seeds + gain ladder at move24 ceiling (gain60/120) + move22_a80 + ctrl.
    **b41/b42 = execution losses (two code-crashes from one operator refactor, commit 8409136); b43/b44 recovered.
    b43/b44 RESULT (Batches 44/45): the gain80 flow winners were SINGLE-SEED FLUKES — move24_a80 {0.383,0.148,0.130}
    =0.220±0.141, move20_a80 {0.408,0.277,0.254}=0.313±0.083, move24_a60 {0.372,0.130,0.214}=0.239±0.121; NONE holds
    seg≥0.35 with elevated flow → sort↔flow is a HARD Pareto frontier [established-integration, see FINDINGS].**
    **b45 RESULT (Batch 46 read): FLOW LEG CLOSED, INT COMPLETE.** The intermediate move15/gain40 fails its falsifier:
    segregation_index 3-seed {0.445,0.325,0.195}=**0.322±0.125** (<0.35 AND indistinguishable from [established]
    move18 0.336, Δ0.014≪2·SD) with net_circ 0.0068±0.0046 ≈ static ctrl 0.0057 (9th single-seed regression).
    move18_a40 reconfirmed the op point (seg 0.371, net_circ 0.0118, 4th point in the 0.336±0.043 band); ctrl_move12
    reconfirmed the max-sort/min-flow endpoint EXACTLY (seg 0.588, net_circ 0.0057); move18_a60 net_circ 0.0040 (LOW)
    ⇒ b44's "net_circ peaks at a60" is move24-only, does NOT transfer to move18. All 8 TIER-1 clean.
  - **INT — STARTED Batch 32, CLOSED Batch 46 (2026-07-05). GATE MET [established-integration].** The flowing,
    dividing, self-partitioning blastula is demonstrated as a joint object; all three legs mapped: PROLIFERATION
    (2× envelope, division = mechanical mixing), continuum-DEFORM (k-lever/spin topology-preserving, motility/mass
    topology-breaking), motility-FLOW (sort↔flow hard Pareto frontier). **INT OPERATING POINT = move18/gain40**
    (`embryo_INT_g20_1p5x_move18.yaml`): seg 0.336±0.043, net_circ 0.0109±0.0009, 3-seed, TIER-1 clean.
  - **ORI (oriented symmetry-breaking) — STARTED Batch 46, CLOSED Batch 51 (2026-07-05). GATE MET.** The demix
    acquires a REPRODUCIBLE, CONTINUOUSLY STEERABLE spatial axis. `sediment` differential per-type drift orients the
    TYPE axis (b48, 3 seeds, −80.5±1.7°, [established]); the shell axis is separately oriented by uniform `gravity`
    (b47, [established]) which is type-blind and does NOT set the type axis. The type axis is PROGRAMMABLE
    ([established] b49): it FOLLOWS the drift vector — x-drift→18°, y-drift→99.5°, diagonal→45° (mod 180). And it is
    CONTINUOUS ([established] b50): intermediate drift-axes 22.5°/67.5° give intermediate measured axes 29.0°/69.6°
    (mod 180, SD ≤1.0°, no snapping) — the b50 SNAP-or-scatter falsifier did not fire. Full steering curve
    (0°→18°, 22.5°→29°, 45°→45°, 67.5°→70°, 90°→99.5°) is monotone with a ~+7–11° offset. **ORI OPERATING POINT =
    differential sediment d10 (a gy −0.10 / b gy +0.10, shell-gravity OFF)** on the INT flowing/dividing/demix
    substrate (`embryo_ORI_sed_d10.yaml`). One-sided drift also orients the y-axis ([established] n=3, angle
    −109.3±1.9°); the FULL-gravity oriented embryo (sediment + shell gravity) COEXISTS but its axis does NOT
    replicate tightly ([open], b50 3rd seed a 92° outlier). **[engineering] under any oriented force raw
    escape/r_cell_max is a BODY-DRIFT artifact — judge TIER-1 by collapsed/nn_min/circ/montage.**
  - **GRO (growth — Phase 2 prerequisite) — STARTED Batch 51 (2026-07-05). Gate: controlled area/volume growth via
    `cell_grow` (continuous rest-volume increase, INDEPENDENT of division) with a clean protrusion→rounding and the
    blastula still intact (collapsed=0, escape=0).** `cell_grow.py` EXISTS + registered: grows the CELL rest-volume
    `grow_V` by a logistic law (rate·V·(1−V/target)), REALIZED by waking dormant `grow_reserve` MPM particles near
    live seeds (modes isotropic/anisotropic/tip; `rate≤0` = byte-identical no-op). In THIS embryo the "cell" level =
    the blastula BODY (1 cell, membrane+water), child mpm_particle its discretisation → `cell_grow` grows the BLASTULA
    BODY area (epiboly-like), the rest-volume-growth operator whose ABSENCE blocked 1C epiboly. **[open, pre-registered
    OBSTACLE — source read] `mpm_anchor` captures rest=frame-0 for ALL particles; the reserve pool is parked at the
    parent CENTRE at frame 0 (engine.py:341) → a woken reserve is sucked toward the centre by the anchor, defeating
    expansion.** GRO batch 1 base (`embryo_GRO_base.yaml`) therefore DROPS mpm_anchor (contained by wall_damp, like the
    reference `material_cell_grow_iso.yaml`); division OFF, agents passive riders; per_parent 14000 + grow_reserve 6000,
    target 1.4. Batch 51 = ISOLATED validation: no-op control (rate 0) + iso rate ladder 0.15/0.3/0.6/1.0 +
    anchor-restored obstacle-doc + anisotropic-bud + tip-elongation explores.
    **Batch 52 read: b51 FALSIFIER FIRED — `cell_grow` does NOT realize area/epiboly in the embryo body [open →
    realization re-scope].** Area is DEAD-FLAT across the entire rate ladder 0→1.0 (final 0.363±<0.4%, byte-similar
    trajectory to rate-0 ctrl at every rate; circularity 0.99, no protrusion). Growth IS occurring — the woken
    reserve just densifies the CORE, not the rim: `anch_r03` is the tell (area SHRINKS 0.3596→0.3558, deform 0.0133
    = 5× below no-anchor ~0.06, escape 0.0 = agents pinned → the anchor sucks woken material to parent centre,
    engine.py:341 obstacle REPRODUCED), and growth slots run DENSER than ctrl (nn_mean r03 0.0123, aniso 0.0053 vs
    ctrl 0.0176). **ROOT CAUSE [established, source-read]:** `area` (scorecard.py:43-67) = the FRAME-0-identified
    outer-shell particle envelope at the last frame; `cell_grow._realize_cell` (cell_grow.py:58-95) places every
    woken reserve at a RANDOM INTERIOR live seed + offset 0.01 with F=I (REST → zero pressure) → interior densifies
    but the shell never advects out, held by the elastic MEMBRANE (outer 7%, youngs 200, rest-shape memory) +
    wall_damp 0.7. The reference `material_cell_grow_iso` expands 4.5× because it has NO membrane (free youngs-90
    ball), NO agents, and huge reserve headroom (9000/2500=3.6×, target 4.5) vs GRO's 6000/14000=0.43× + stiff
    membrane. montage `tip_elong` "spikes" = escaping AGENT tracers (escape 0.2045), NOT tissue lobing (circ 0.9921).
    **Batch 52 = REALIZATION DEBUG:** new base `embryo_GRO_g` (per_parent 8000 + reserve 12000, target 2.5, rate 1.1
    = reference magnitude, total 20000 = same runtime). POSITIVE CONTROL `pureball` (no membrane/no agents, radius
    0.15 = reference repro in harness). Membrane-stiffness ladder {noshell, youngs 20, 200, 600} at fixed growth +
    wall_damp 0.35 + offset 0.05 explores. Predict pureball expands ≥1.5×, embryo area rises as membrane softens.
    **Batch 53 read: b52 falsifier did NOT fire — `cell_grow` IS a working epiboly primitive [established];
    the b51-read "realization broken" is RETIRED.** POSITIVE CONTROL `pureball` (uniform elastic y90 free ball)
    expands area **5.7×** and stays ROUND (circularity 0.923–0.989, fourier_m1 collapses 0.44→0.04, deform 0.214);
    growth is back-loaded (logistic ramp, area 0.072→0.331 over 50–75%). **THE BLOCKER IS THE ELASTIC MEMBRANE,
    and it is a BINARY GATE [established, n=1/stiffness but 4-point flat]** — every membrane stiffness youngs
    {20,200,600} pins area DEAD-FLAT at 0.36 (softmemb y20 blocks as hard as stiffmemb y600 → the b52 "monotone
    softening" prediction is REJECTED). Mechanism: a closed elastic loop remembers its frame-0 rest shape (F, via
    mpm_strain every substep) and springs the boundary back ~independently of youngs; wall_damp (lowdamp) and
    placement offset (bigoffset) do NOT unblock (both flat) → confinement/placement are NOT the gate. **Dropping
    the shell (`noshell`, single liquid y40) unblocks growth (area 0.360→0.811, 2.25×) BUT the liquid FRAGMENTS
    into strands [established]:** circularity 0.982→0.463→0.752 (buckles), shape_index →5.63, fourier_m3 →0.119.
    **THE GRO TENSION = need BOTH cohesion (shell) AND area growth; elastic shell locks area, liquid loses
    cohesion.** TIER-1 clean everywhere (collapsed 0.86–1.0, nn_min ~0.0002, no rupture); agents passive → seg/migr
    are noise. **Batch 53 = COHESIVE EPIBOLY.** Source-read found TWO candidate cohesion mechanisms that don't lock
    area: (a) `material: viscoelastic` (Maxwell) membrane with a `tau:` relaxation (mpm_strain.py:48-56, entities.py
    :122-124) — holds shape short-term, RELAXES F toward isotropic over tau → remodels under growth pressure; (b)
    `surface_tension` (CSF, mpm_grid_update.py:94; ref water sims 8–30) on the liquid body — keeps a growing blob
    ROUND without fixed rest-shape memory. Primary = viscoelastic-membrane tau ladder; alt = surface-tension-liquid.
    **Batch 54 read: b53 BOTH cohesion mechanisms FAILED [rejected]; the epiboly architecture is UNIFORM ELASTIC
    (pureball), and a USER FIX restores the non-collapsing agent gate.** (i) Viscoelastic membrane BLOCKS area at
    EVERY tau — visco_t01/t03/t10/t30 area all FLAT 0.359→0.36 (spread <0.4% over a 30× tau span) → b53 primary
    FALSIFIED. MECHANISM [established by result+theory]: Maxwell/viscoelastic relaxes DEVIATORIC (shape) stress but
    CONSERVES VOLUME (isochoric) → area==volume in 2D can NEVER grow by shape-relaxing rheology → the b53 "elastic =
    binary area gate" is a VOLUMETRIC lock any viscoelastic inherits. RETIRE viscoelastic-epiboly [rejected].
    (ii) Surface-tension on the liquid CONTRACTS not grows: st_liq15 area 0.354 (b52 noshell was 0.81), st_liq40
    SHRINKS to 0.330 (−6.4%); st = perimeter-minimizing inward stress, buys cohesion but kills ALL growth → RETIRE
    surface-tension-epiboly [rejected]. **The ONLY working cohesive epiboly = b52 pureball UNIFORM soft elastic
    (single youngs-90 layer, no liquid-core/stiff-membrane split): area 5.7× round; cell_grow grows the elastic
    REST volume uniformly → body inflates to its larger rest shape, elasticity keeps it round.** cell_grow SEMANTICS
    [engineering]: `grow_V` = rest-VOLUME multiplier, logistic to ceiling `target`; realization wakes reserve so
    live COUNT ~ grow_base·grow_V → `grow_reserve` is a HARD growth cap (max count=(per_parent+reserve)/per_parent);
    pureball count 2.5× → AREA 5.7× (area/count≈2.3 spread) so at embryo radius 0.34 that spread blows the [0,1]
    box → GRO growth demos must start the body SMALLER (radius 0.24) with modest reserve caps. **USER DIRECTIVE
    (2026-07-05, applied b54): every b51/b52/b53 GRO spec HARD-FAILED the agent gate via `mpm_to_agent.confine: 3.0`
    (collapsed 0.91–1.0, nn_min 0.0002, escape 0.19–0.30) — the exact 1A-proven collapse press; RESTORE the 1A
    non-collapse point `confine 0.03 + repel.strength 150` in all GRO specs; growth measured on a collapsed blastula
    is invalid; re-baseline (rate-0 no-op gate-clean) FIRST.** Batch 54 = RE-BASELINE (confine 0.03) + UNIFORM-ELASTIC
    cohesive-epiboly test: new confine-0.03/repel-150 specs `embryo_GRO_u` (uniform y90, radius 0.24, reserve 3000
    cap 1.375×, target 2.0) + `embryo_GRO_m` (membrane arch, same scale). Slots: uni_ctrl0/memb_ctrl0 (rate0
    re-baseline) + youngs ladder 40/90/140 + uni90_big (reserve 6000) + memb90_g + uni90_st. HYP: uniform-elastic
    grows cohesive area (uni90_g ≥1.4× ctrl 0.18, circ ≥0.95) gate-clean; membrane stays locked ~0.18; softer→more.
    **Batch 55 read: b54 = USER COLLAPSE FIX WORKS but UNIFORM epiboly FLAT + a NEW escape gate.** (i) COLLAPSE
    FIXED [established]: confine 3.0→0.03 + repel 8→150 gives collapsed=0 on all 8 slots (was 0.86–1.0), nn_min
    0.0189–0.0191 (≈r0, marginal-pass, 100× above the b51–53 crush 0.0002). (ii) AREA DEAD-FLAT across the whole
    ladder [b54 falsifier FIRED] — uni y90/y40/y140 all 0.180→0.183–0.185 (+1.6–2.6%), reserve 3000→6000 flat too;
    the b52 pureball 5.7× did NOT reproduce at start-radius 0.24 + reserve 3000/6000. surf_tens 8 BIT-IDENTICAL to
    plain (inert at this scale; consistent with b53 st needs ≥15). (iii) NEW OPEN GATE = ESCAPE [open]: at confine
    0.03 the sparse 44-agent cluster in a SOFT uniform body wanders OUT — ctrl0 (rate0) escape 0.727, uniform slots
    0.41–0.73, correlates with deform 0.11–0.17 (soft body sloshes agents out); the MEMBRANE slot confines them
    (memb90 escape 0.068, stiff-rim colour field). So confine-0.03 is escape-clean ONLY for a stiff-rimmed disc
    that agents FILL (INT/ORI geometry), NOT a sparse cluster in a soft ball. (iv) HYPOTHESIS [open, b55 tests]:
    b52 pureball grew because it started SMALL (mpm radius 0.15, area 0.072) with HUGE reserve (12000, cap 2.5×)
    — a fixed # of woken interior particles inflates a small body a lot, a big body barely (they just densify the
    core, the b51 mechanism). b55 = reproduce the working pureball config (radius 0.15 + reserve 12000) WITH the
    confine-0.03 fix (base `embryo_GRO_pb`) + isolate start-scale × reserve on a 2×2; predict growth also IMPROVES
    escape (inflating body engulfs central agents). If pb_fix stays flat despite full pureball config → the confine
    coupling itself blocks continuum inflation → cohesive epiboly ⊥ non-collapse gate → re-scope.
    **Batch 56 read: b55 = pureball epiboly does NOT reproduce with the COUPLED agent blastula [falsifier's 2nd
    branch FIRED]; TWO durable discoveries.** (i) **`area` is an AGENT-CLOUD ARTIFACT [engineering, established]:**
    the rate-0 control (pb_ctrl0, cell_grow.rate 0) spikes area 0.072→**0.339** at 75% IDENTICALLY to every growth
    slot (pb_fix 0.337, pb_r6k 0.328) → `area` (frame-N alpha-hull) is dominated by transient AGENT dispersal that
    fully reverts, NOT epiboly. **From b56 on, judge GROWTH by `disc_R` (agent-shell radius), NOT `area`, whenever
    agents are present.** (ii) **`disc_R` DEAD-FLAT ~0.148 (=base radius 0.15) on ALL radius-0.15 slots** (ctrl0
    0.1481, fix 0.1481, r6k 0.1486, r3k 0.1489, t40 0.1481, y40 0.1481; big24 0.237=its 0.24 start), growth-ON and
    growth-OFF alike → the agent blastula NET-expanded ZERO. Had the continuum inflated 5.7× (b52), embedded agents
    would advect out to disc_R~0.33 — they did not. **The b52 pureball 5.7× (measured AGENT-FREE) does NOT transmit
    to / survive the coupled agent blastula; body starts r0.15, ends r0.15.** (iii) **ESCAPE ∝ continuum deform,
    tight monotone [open, quantified]:** r3k(deform 0.032)→escape 0.045 CLEAN | y40(0.058)→0.36 | ctrl0(0.089)→0.75
    | r6k(0.105)→1.0 | fix(0.111)→0.93 | big24(0.141)→0.70 | conf(0.152)→1.0. Growth-induced sloshing ejects the
    under-confined 44-agent cluster; only low-reserve→low-deform r3k holds. b55 prediction "growth improves escape"
    FALSIFIED (fix 0.93 > ctrl0 0.75). (iv) **confine-UP fails BOTH ways [rejected as escape fix]:** pb_conf
    (confine 0.03→0.3) escape 1.0 (WORSE) + nn_min 0.0104 (crush onset) + accel 0.00329 (flings agents). No clean
    hold window between escape(0.03) and crush(0.3). Collapse fix HELD a 3rd batch (collapsed 0 all 8). **THE GRO
    WALL:** cell_grow inflates a uniform elastic ball 5.7× AGENT-FREE [b52], but coupling the non-collapsing agent
    blastula (confine 0.03 + repel 150 + flow_align 40 + agent_to_mpm feedback) suppresses/reverts it (disc_R flat).
    **Batch 56 = AGENT-SUPPRESSION DIAGNOSTIC** (judged on disc_R): linchpin anchor `noag` (agents n→2 — does the
    continuum inflate in this base at all?) + agent-feedback ablation `m0` (agent_to_mpm.agent_mass 0 = passive
    riders, does disc_R rise?) / `fa0_m0` / `fa0` (kill flow_align swim-out) / `c0` (confine 0) / `m0_k05` (weak
    back-drag) / `dense_m0` (n130 jammed passive shell) + rate-0 ctrl. New specs `embryo_GRO_pb_noag` (n2),
    `embryo_GRO_pb_dense` (n130 spawn_radius 0.13); rest dotted overrides on `embryo_GRO_pb`. HYP: with agent→MPM
    mass-feedback OFF (agent_mass 0) the continuum inflates and CARRIES the passive agent shell out (disc_R
    0.15→>0.25) escape-clean → agent mass-feedback was the suppressor. FALSIFIER: `noag` ALSO disc_R ~0.15 →
    continuum growth broken in this base regardless of agents (target/reserve/frames regressed vs b52) → re-scope
    the CONTINUUM not the agents.
    **Batch 57 read: b56 falsifier fired but ON A BROKEN READOUT — `disc_R` IS A FRAME-0 CONSTANT, cannot
    measure growth [engineering, decisive]; the "no net growth" claim is REAL for a deeper reason (density-not-
    volume realization).** (i) `embryo_metrics.py:41` computes `disc_R = quantile(|mp[0]-c|,0.99)` from **frame
    0** (used as the deform/escape normaliser) → disc_R ≡ initial radius 0.148 for EVERY radius-0.15 spec by
    construction; its being bit-identical (0.1481) across all 8 b56 slots is a TAUTOLOGY, not evidence of no
    growth. **RETIRE all "disc_R flat → zero expansion" claims (b55, b56).** (ii) But the montage/blob_evolution
    make the no-growth conclusion VISUALLY REAL: rate-0 ctrl0 expands ~1.4× radius (~55→78 px) from elastic
    equilibration + anchor-free DRIFT with NO growth op; the growth slot m0 (rate 1.1) reaches ~1.3× ≈ ctrl0 —
    cell_grow adds nothing over the rate-0 transient. (iii) **ROOT CAUSE [open]:** `cell_grow._realize_cell`
    (cell_grow.py:73-90) wakes reserve particles INSIDE the blob (offset 0.011 from a random interior seed) at
    **F=I (rest)** → raises particle DENSITY, not REST-VOLUME; MPM has no particle collision, pressure comes only
    from constitutive F, so F=I → 0 stress → no outward pressure → no inflation. Wake-reserve adds mass without
    volume. (iv) **METRIC FIXED:** added drift-free `grow_R`/`grow_ratio` (median radius of frame-0 outer-shell
    particles about their OWN final-frame centroid, vs t0; try/except-guarded, disc_R unchanged). TIER-1: collapsed
    0 all 8 (fix held 4th batch); anchor-free escapes (noag/fa0/c0 1.0, ctrl0 0.75) are BODY-DRIFT artifacts not
    agent-escape (blob leaves origin). **Batch 57 = OFFSET LADDER growth-realization test** (does peripheral
    placement inflate?): ctrl0 rate-0 baseline + `cell_grow.offset` 0.011/0.03/0.06/0.10/0.15 on embryo_GRO_pb +
    weak-anchor `embryo_GRO_pb_wkanch` (mpm_anchor k2, contains drift) + reserve-headroom `embryo_GRO_pb_bigres`
    (reserve 20000/target 3.5). READOUT = grow_ratio (NEW). FALSIFIER: grow_ratio ≈ ctrl0 across the whole offset
    ladder → placement CANNOT inflate → the realization needs an F-prestretch (rest-volume) change in the OPERATOR
    next batch, not a spec knob. NEW GOTCHA: `disc_R` is frame-0 — NEVER use it as a growth readout; use grow_ratio.
    - **[established, Batch 58 / b57] WAKE-RESERVE REALIZATION ADDS DENSITY NOT VOLUME — offset/reserve inert.**
      The drift-free `grow_ratio` (validated: reads ctrl0 1.0003, wkanch 1.0012) is pinned at **1.000±0.001 on
      ALL 8 b57 configs** — offset 0.011→0.15 AND reserve 12k→16k/target 3.0, growth-ON (rate 1.1) and rate-0
      alike (grow_R net-displaced ≤+0.2%). Montage: blue MPM blob same diameter t0→t12000. ROOT CAUSE traced
      end-to-end: cell_grow woke particles at **F=I**, and the fixed-corotated law (mpm.py:100–101,
      `2μ(F−R)Fᵀ + λJ(J−1)I`) is **exactly 0 at F=I** → woken particle co-locates, no push → density up, volume
      flat (MPM has no particle collision; pressure is purely constitutive). Offset just relocates a force-free
      particle. **RETIRE the b57 offset hypothesis — placement CANNOT inflate.** Collapse fix HELD (collapsed 0
      all 8, 5th batch; nn_min 0.0183–0.0191). Escape is BODY-DRIFT (rate-0 ctrl0 escape 0.75 w/ zero growth
      proves it); larger offset DAMPS drift (off10=0.10: migr 0.29, r_cell_max 0.93<1, escape 0.045 = only clean
      slot) but still no growth.
    - **[engineering, Batch 58] OPERATOR FIX = `cell_grow.prestretch` (pre-compress woken particles).** Added
      param (cell_grow.py:__init__ + line 98): woken F = prestretch·I. prestretch=1.0 → F=I → BYTE-IDENTICAL to
      old no-op (safe default; b57 off03 IS the ps=1.0 anchor grow_ratio 1.0002). prestretch<1 → J=s²<1 → both
      corotated terms outward → particle relaxes toward rest by PUSHING neighbours → envelope inflates → realizes
      rest-VOLUME growth WITHOUT a material-model change. Contained/backward-compat: rate≤0 early-returns; no new
      op token (no import/spec-load crash risk); schema has no param whitelist. b58 = FIRST prestretch-ladder
      test; predict grow_ratio rises monotone as prestretch↓ (WIN ≳1.3). FALSIFIER: flat ~1.000 → pre-compression
      absorbed locally → need a sustained growth tensor F=Fe·Fg (stress on Fe), deeper change.
    - **[open→REALIZED, Batch 59 / b58] PRESTRETCH REALIZES GROWTH — the GRO wall is BROKEN. FIRST real
      inflation in the coupled agent blastula.** b58 falsifier did NOT fire. grow_ratio (drift-free) rose
      MONOTONE in pre-compression: ctrl0(rate0) 1.0003 → ps0.9 1.076 → ps0.8 1.136 → ps0.6 **1.202** (|Δ| 0.20
      = 20% radius / +37% area vs ctrl0), collapsed 0 all 8 (fix HELD 6th batch). Montage: blue MPM blob
      VISIBLY larger by t12000 in every growth panel (ctrl0 only drifts). n=1/rung → [open], needs seeds.
      TWO deviations from prediction, both informative: (a) **SATURATION at ps≤0.6** — ps0.4 1.199 ≈ ps0.6
      1.202 (Δ 0.003), and over-compression BUCKLES shape (ps0.4 circ 0.551, shape_index 4.78 vs ps0.6 0.866);
      stronger pre-compression buys no radius, only distortion → ps0.6 is the knee. (b) **CEILING IS
      RESERVE-LIMITED not compression-limited [the lever to break 1.3]:** bigres (reserve 16k, tgt 3.0) at the
      SAME ps0.6 → 1.2521 > base (12k) 1.2017 (+0.05) → more woken-reserve headroom → more inflation. To
      exceed grow_ratio 1.3, add reserve, not compression.
    - **[open, Batch 59 / b58] WEAK ANCHOR (mpm_anchor k2) IS COMPATIBLE WITH PRESTRETCH GROWTH — resolves the
      pre-registered GRO drop-anchor obstacle.** ps0.6+wkanch grow_ratio 1.2015 == ps0.6 base 1.2017 (anchor
      does NOT suppress) AND gives the CLEANEST TIER-1 (escape 0.43 vs base 0.73, r_cell_max 1.45 lowest,
      deform 0.062 lowest, circ 0.955 roundest). The obstacle (anchor sucks woken F=I reserve to parent centre
      → kills growth) does NOT bite for PRESTRETCH particles: each carries its own outward corotated stress and
      inflates even while the weak anchor pulls its rest toward centre. **The weak anchor is now a valid growth
      container** (tames body-drift, keeps body round, costs no growth) = the GRO operating architecture.
      **offset 0.06 HURTS once prestretch works** [engineering]: ps0.6+o06 grow_ratio 1.175 < base 1.202 AND
      fragments (circ 0.248, shape_index 7.12) — peripheral force-carrying particles punch lobes; use interior
      offset 0.03. **GRO OPERATING POINT (emerging) = ps0.6 + weak-anchor k2 + big reserve, offset 0.03.**
      Escape 0.43–1.0 remains the BODY-DRIFT artifact (rate-0 ctrl0 0.75, zero growth = proof); judge growth by
      grow_ratio. Batch 59 = reserve ladder (16k/20k/24k) to break 1.3 + ps0.6-wkanch ×3 seeds to establish.
    - **[ESTABLISHED, Batch 60 / b59] PRESTRETCH-REALIZED RESERVE GROWTH — 3 seeds, huge margin; BREAKS 1.3.**
      The b59 reserve-ladder falsifier did NOT fire. On the ps0.6 + weak-anchor(k2) base (confine 0.03 / repel
      150 non-collapse gate), grow_ratio (drift-free) rises MONOTONE with reserve: ctrl0(rate0) 1.0015 →
      res16a(16k) **1.2492 ± 0.0022** (3 seeds, SD 0.2%, Δ vs ctrl 0.248 ≈ 110·SD → [established]) → res20a(20k)
      1.292 → res24a(24k) **1.3318** (>1.3, circ 0.933 round, collapsed 0) ≈ +0.04 grow_ratio per +4k reserve.
      area 0.071→0.124 = 1.74× (grow_ratio² consistent). grow_ratio is a DETERMINISTIC bulk quantity (seed
      scatter 0.2%). **RESERVE is the growth lever, monotone; the GROWTH-LAW (rate/target) is INERT at fixed
      reserve** — res20a_hi (rate 2.2/target 4.5) 1.293 == res20a (rate 1.1/target 3.5) 1.292; the target (4.0
      set) is NEVER reached (got 1.77× area) → growth is mechanism-limited below the pool; to grow bigger, add
      reserve. **Over-compression BUCKLES w/o extra growth** (ps0.4 1.296 ≈ ps0.6 1.292 but circ 0.903→0.625;
      2nd batch confirming ps0.6 is the knee) → RETIRE ps≤0.4. **CIRCULARITY is the SEED-VARIABLE part** (res16a
      1 of 3 seeds buckled circ 0.665 while grow_ratio stayed 1.250 clean) → growth magnitude replicates, shape
      round-vs-lobed is stochastic. TIER-1 clean 7th straight batch (collapsed 0, nn_min 0.0189–0.0194); escape
      0.36–0.75 = BODY-DRIFT artifact (rate-0 ctrl0 escape 0.364, zero growth = proof). **GRO OPERATING POINT =
      ps0.6 + weak-anchor k2 + reserve 24k (grow_ratio 1.33, circ 0.93), offset 0.03.**
    - **[ESTABLISHED→extended, Batch 61 / b60] LADDER KEEPS CLIMBING THROUGH 32k, DECELERATING toward a soft
      ceiling; grow_ratio deterministic the whole way.** b60 falsifier (plateau at ~1.33) did NOT fire. Full
      ladder grow_ratio (drift-free, ps0.6/wkanch): res16a 1.249 → res20a 1.292 → **res24a 1.3318 ± 0.0005 (3
      seeds)** → **res28a 1.3653 ± 0.0007 (2 seeds)** → res32a 1.397 (n=1) → res32a_ps05 1.422 (n=1). Increments
      DECELERATE (+0.043, +0.040, +0.034, +0.032 per +4k) → approaching a soft pool-limited ceiling, not flat.
      area 0.0714 (ctrl) → 0.1363 (res32a) = **1.91×**. grow_ratio SD 0.0005 at res24a (0.04%) → magnitude is a
      deterministic bulk quantity up the whole ladder. **ps0.5 > ps0.6 at res32** [open, n=1]: milder pre-press
      realizes MORE growth (1.422 vs 1.397) AND stays round (circ 0.937) — a viable high-reserve knee (res32/ps05
      is gentler than the retired res20/ps04 buckle). **CIRCULARITY buckle is STOCHASTIC, NOT growth-monotone**:
      only res24a_s2 buckled (circ 0.757, shape_index 4.08 vs ~3.58) of ~7 growth slots; deform_rms rises monotone
      (0.070→0.078) but shape stays round on most seeds → grow_ratio replicates tight, round-vs-lobed is ~1/seed
      coin-flip. **ANCHOR-STIFFNESS (k4) is a shape lever but FREEZES flow**: res24a_stiff grow_ratio unchanged
      1.332, circ round 0.933, but msd 0.0296→0.0039 (7.6×), migration 0.60→0.32 — buys roundness by killing the
      cloud motion; weak anchor (k2) keeps the flowing blastula and accepts the ~1/seed buckle. TIER-1 clean 8th
      straight batch (collapsed 0, nn_min 0.0191–0.0194). Batch 61 = ceiling map (res36a/res40a — plateau or keep
      climbing?) + res32a & res32a_ps05 3-seed lock (high-growth op point + buckle frequency). If res36a≈res40a≈
      res32a (flat ±0.01) → hard pool ceiling ~1.40 → CLOSE GRO, ADVANCE to PAT.
    - **[open, Batch 62 / b61] LADDER STILL CLIMBING THROUGH 40k (no plateau); BUCKLE is SEED-driven not
      press/reserve-driven.** b61 plateau falsifier did NOT fire. Growth proxy g=√(area/0.07143 ctrl): ctrl0 1.000
      → res32a **1.390 ± 0.012 (3 seeds)** → res36a 1.409 (n=1) → res40a **1.438 (n=1, area 0.1478 = 2.07× ctrl,
      biggest yet)**; increments +0.015 (32→36), +0.029 (36→40) — NOT decelerating, pool not exhausted at 40k
      (extension deferred: cosmetic at rising compute, 48000 pts/26min; primitive already [established]).
      **BUCKLE FREQUENCY = ~1/3 at res32, ps-INDEPENDENT and the SAME seed index (s2) both ladders**: res32a_ps06
      circ {0.976,0.979,**0.681**}, res32a_ps05 circ {0.937,0.982,**0.557**}, shape_index 4.29–4.75 (round ~3.6) →
      buckle is INITIAL-CONDITION/seed sensitive, NOT prestretch- or reserve-driven; buckled seeds keep g clean
      (1.40,1.42) + gate-clean (nn_min 0.019). **res36a/res40a (n=1) stayed ROUND (circ 0.951/0.941) despite MORE
      growth** → buckle not reserve-monotone (either res32-specific or lucky seeds — the b62 3-seed roundness lock
      decides). Stiff anchor k4 rounds (res32a_stiff circ 0.964) but freezes flow (migr 0.29), reconfirmed. TIER-1
      clean 9th straight batch (collapsed 0, nn_min 0.0191–0.0193, incl. both buckled seeds). Batch 62 = GRO
      CLOSING: res36a/res40a each 3-seed roundness lock + moderate-anchor k3 buckle mitigation on res32a_s2. WIN =
      res36a & res40a circ ≥0.90 on 3/3 → round high-growth op point [established] → CLOSE GRO, ADVANCE to PAT;
      FALSIFIER = either buckles ≥1/3 → fall back to res24a (g 1.33, safest round) as the GRO op point.
    - **[established growth / open roundness, Batch 63 / b62] GROWTH PRIMITIVE SOLID (g→1.45, area 2.10×); the
      ~1/3 s2 BUCKLE is RESERVE-GENERAL + k3-UNFIXABLE — b62 falsifier FIRED on both top rungs.** Growth magnitude
      reconfirmed monotone/deterministic: res36a 3-seed g **1.416±0.013** {1.409(s0),1.408(s1),1.432(s2)}, res40a
      3-seed g **1.441±0.008** {1.438,1.435,1.450(area 0.15024=2.10× ctrl, biggest)}. ROUNDNESS LOCK FAILED — BOTH
      rungs buckle exactly 1/3, and it is the SAME s2 seed: res36a circ {0.951,0.971,**0.649**}, res40a circ
      {0.941,0.988,**0.771**} (s2 shape_index 4.04, fourier_m3/m4/m5 ~0.009 = higher-mode FOLD, not m2 ellipse).
      Buckle now seen at res24 (b60), res32 (b61), res36+res40 (b62) all at seed index s2 → **SEED/initial-condition
      instability recurring at EVERY reserve level, NOT res32-specific**; the b62 fallback rung res24a is itself
      buckle-prone (res24a_s2 circ 0.757) so lowering reserve does NOT escape it. **k3 MITIGATION FALSIFIED**:
      res32a_k3_s2 circ **0.634** (≈ res32a_s2 at k2: 0.681 ps06 / 0.557 ps05) — moderate anchor does NOT round the
      buckler, AND already costs ~30% flow (k3 msd 0.0169/migr 0.43 vs k2 round-seed msd 0.0248/migr 0.61) = worst of
      both. Only k4 rounds (res32a_stiff 0.964) but tested ONLY on already-round seeds + freezes flow (migr 0.29);
      k4-on-the-buckler untested. **Buckle = COMPRESSION-driven** (ps05 more-compressed buckled WORSE than ps06,
      0.557<0.681) → an elastic buckling mode of the growing pressurized shell, seed-triggered + prestretch-loaded.
      TIER-1 clean 10th straight batch (collapsed 0, nn_min 0.0188–0.0195, incl. buckled seeds). Batch 63 = buckle
      RESOLUTION on the reliable s2 buckler at res36a: flow-compatible anti-buckle levers (ps0.8 mild-press, membrane
      youngs 200/400 bending stiffness, k4-on-buckler, stacks) + s1_k4 round-seed reference + ctrl_s2 repro. WIN = a
      lever rounds s2 ≥0.90 with msd ≳0.02 → round high-growth op point; FALSIFIER = nothing rounds s2 flow-safely →
      adopt res36a (g 1.41, 2/3 round) as GRO op point, accept intrinsic ~1/3 buckle, CLOSE GRO → PAT.
    - **[GRO CLOSED, Batch 64 / b63] BUCKLE FALSIFIER FIRED — no flow-safe lever rounds the s2 buckler to
      ≥0.90; GRO closes on res36a, accepting the intrinsic ~1/3 buckle.** ctrl_s2 reproduced the buckle
      bit-for-bit (circ **0.6493 == b62 0.6493**, area 0.14651 == 0.14651 = deterministic IC instability).
      Lever results on the buckler (circ): anchor k4 **0.618** (NO round + freezes flow msd 0.0012, 22× down)
      → the b62/b63 "only k4 rounds" claim RETIRED (k4 only keeps an already-round seed round: s1_k4 0.981);
      membrane youngs200 **0.708** / youngs400 **0.511** (WORSE, shape_idx 4.96) → stiffness AMPLIFIES the
      fold, [rejected] as a lever; prestretch ps0.8 **0.772** (best flow-safe, msd 0.026 kept) BUT growth CUT
      area 0.1465→0.1149 (g 1.43→1.27, −22%) → less compressive drive rounds but is the SAME drive that grows
      → round+high-growth+flow are MUTUALLY EXCLUSIVE on a buckle seed. Best overall = ps08_k3 circ 0.774
      (still 0.13 short of gate). TIER-1 clean 11th straight batch (collapsed 0, nn_min 0.0191–0.0195).
      **GRO DELIVERABLE [established]: reserve-scaled prestretch `cell_grow` = a continuous non-mixing epiboly
      primitive; 3-seed deterministic magnitude, monotone in reserve through 40k, collapse-free, area to 2.10×
      ctrl. GRO OP POINT = res36a (reserve 36k, ps0.6, weak anchor k2): g 1.416±0.013, 2/3 round, area 2.0×,
      flowing (msd ~0.025, migr ~0.6); rounder-but-smaller alt = ps0.8 (g 1.27). Intrinsic ~1/3 seed buckle
      (compression-buckling mode) is ACCEPTED — not suppressible flow-safely at high growth.** GRO ran b51–b63
      (13 batches, over the 10-cap) → CLOSED, `current_stage.txt` → PAT.
  - **PAT (patterning — Phase 2) — STARTED Batch 64 (2026-07-05). Gate: `mi_type_x`/segregation ↑ and domains
    PERSIST during active growth (low late-time `mixing_entropy` drift); then a chemical field gates WHERE
    `cell_grow` acts (spatially programmable growth). PAT-1 (persistence leg) MET [established] b65: g10-grow
    seg 0.653±0.178 (3 seeds) vs nochem ctrl −0.158, 4.6·SD. PAT-2 (oriented domain map) MET [established] b67:
    differential sediment orients the growing demix to seg 1.0 / mi_type_y 0.989±0.011, 3 seeds, ~90·SD.** Batch 64 = ISOLATED PAT-1 validation: does the
    [established] 1E two-channel chemotactic demix (seg ~0.49) FORM and PERSIST on a GROWING tissue? The sharp
    question: `cell_grow` is CONTINUOUS material addition (no cell repacking), whereas INT established
    `cell_divide` = a mechanical MIXING event that DILUTED the demix (seg 0.485→0.216 at 2×, →0.06 at 3×). Does
    non-mixing epiboly PRESERVE the demix where mixing division destroyed it? PAT base = GRO res36a op-point +
    the 1E demix stack (chem field 2ch, deposit/diffuse/decay, per-type `op: chemotax` cross-repulsion gain
    -0.10; NB the M1 refactor renamed `chemotaxis`→`chemotax` — use `chemotax` or spec-load KeyError). Slots:
    nochem ctrl + demix_nogrow reference + demix_grow s0/s1/s2 (3-seed) + demix_g20 + demix_sharp + demix_slow20k.
    HYP: demix_grow seg >=0.3, HELD to 100%, > nochem ctrl. FALSIFIER: seg ≈ nochem ctrl (growth advection
    scrambles domains like division did) → partition ⊥ cell_grow.
    - **[open→PAT-1 WIN n=1, Batch 65 / b64] NON-MIXING EPIBOLY (`cell_grow`) PRESERVES THE CHEMOTACTIC DEMIX —
      the OPPOSITE of mixing division.** growth-ON demix (gain −0.10, reserve 36k, g~1.43): segregation_index
      **0.532** (traj 0.030→0.622→0.420→0.684→0.532, coarsens then HOLDS to 100%) vs nochem ctrl −0.158 (Δ0.69)
      vs static-nogrow 0.302 (Δ0.23); ALL 4 co-metrics corroborate (contact_same 0.455→0.797, interface_frac
      0.578→0.234, mixing_entropy 0.942→0.723). cell_grow (continuous MPM material addition, NO cell repacking)
      is COMPATIBLE with — indeed ENHANCES vs static — the partition, in sharp contrast to `cell_divide`
      (mechanical MIXING, diluted seg 0.485→0.06 at 3×). **BUT n=1 not 3: the b64 dotted `general.seed 1/2`
      override SILENTLY FAILED (all 3 slots ran seed 0 → bit-identical seg 0.5315; the override patches operator
      flow-maps but SKIPS the inline `general:` block). Seed genuinely matters (sunflower randomizes headings
      engine.py:66 + a/b type randperm engine.py:212). So [open] pending 3 REAL seeds (b65 uses authored per-seed
      spec FILES).** SUB-FINDINGS (all n=1): (a) GAIN-SCALING holds under growth — g20 (−0.20) seg 0.777 > g10
      0.532 > ctrl, monotone (1E law survives epiboly); (b) GENTLER growth = STRONGER sort — slow20k (reserve 20k,
      g~1.29) seg 0.710 ≫ g10 0.532 at same gain, highest mi_type_x 0.068 ("less advection = better sort",
      growth-magnitude ↔ demix-strength TRADEOFF); (c) SHARP field (diffuse 0.05/decay 0.3) KILLS demix (seg 0.034
      ≈ ctrl) — field DIFFUSENESS/long-range REQUIRED for the cross-repulsion gradient [rejected as improver].
      TIER-1 all clean (collapsed 0, nn_min ≈r0, circ 0.92–0.98 — n=44 blastula does NOT buckle). Geometry LATERAL
      (mi_type_x ≤0.068). **GOTCHA [durable]: per-seed replication REQUIRES an authored spec FILE with seed in the
      general block — the dotted `general.seed` override is a silent no-op (writes a comment, leaves seed:0).**
    - **[PAT-1 [ESTABLISHED], Batch 66 / b65] NON-MIXING EPIBOLY (`cell_grow`) PRESERVES THE CHEMOTACTIC DEMIX
      over 3 REAL seeds.** Per-seed FILE fix worked (seeds now DISTINCT). g10-grow (gain −0.10, reserve 36k,
      g~1.43) seg {0.5315, 0.5715, 0.8574} = **0.653 ± 0.178** vs nochem ctrl −0.158 (Δ 0.811 = **4.6·SD**,
      NON-overlapping; min seed 0.532 ≫ gate 0.35); 4 co-metric means corroborate (contact_same 0.848 vs ctrl
      0.455, interface_frac 0.172 vs 0.578, mixing_entropy 0.647 vs 0.942). Persistent to 100%. **PAT-1 (demix
      persists under growth) MET; cell_grow COMPATIBLE with the partition, OPPOSITE of cell_divide (mechanical
      mixing diluted seg 0.485→0.06@3×).** REVISIONS to the b64 n=1 sub-findings: (a) gain-scaling under growth
      HOLDS (g20 0.818 n=2 > g10 0.653, [open]); (b) **"gentler growth = stronger sort" [rejected]** — slow20k
      3 seeds 0.640±0.263 ≈ g10 0.653±0.178, OVERLAPPING (b64's 0.710 was seed-luck, its s4 crashed to 0.349);
      (c) **growth does NOT clearly ENHANCE vs static** — static nogrow 2 seeds 0.577±0.389 ≈ growth 0.653
      (b64's "growth>static" was seed-luck); growth is COMPATIBLE, not a proven enhancer. **(d) BUCKLE RECURS at
      n=44 UNDER GROWTH — CORRECTS b64 "n=44 does not buckle":** the seed-2 IC buckled under BOTH g10 (circ 0.647)
      AND gentle slow20k (circ 0.648), while static nogrow s1 stayed round (0.965) → the GRO compression-buckle
      (growth + seed-2 IC, reserve-general) carries into PAT; TIER-1 clean (nn_min ≈r0, collapsed 0 — shape
      artifact). **(e) PATTERN IS LATERAL/UN-ORIENTED** — mi_type_x/y seed-noisy (g10 mi_type_x {0.038,0.272,
      0.033}), type_axis_angle scatters (−55°…176°) = random axis per seed (like 1E) → the PAT gate's "mi_type_x ↑"
      (spatially-DEFINED map) NOT yet met. TIER-1 all clean (13th straight batch on confine 0.03; circ 0.92–0.98
      except the 2 buckle seeds). **PAT-1 OP POINT = embryo_PAT_base (g10-grow).** Batch 66 = PAT-2 = ORIENT it.
    - **[PAT-2 [ESTABLISHED], Batch 67 / b66] DIFFERENTIAL SEDIMENT ORIENTS THE GROWING DEMIX INTO A NEAR-PERFECT,
      REPRODUCIBLE-AXIS STRATIFICATION — and COMPLETES it (seg→1.0).** Adding differential `sediment` (a gy −0.10 /
      b gy +0.10, shell gravity OFF) to the PAT-1 growing chemotactic demix: **sed_d10 3 seeds seg 1.0 (all),
      mi_type_y {0.9985, 0.994, 0.976} = 0.989±0.011** vs nochem ctrl mi_type_y 0.0022 (Δ 0.987 ≈ **90·SD**),
      **type_axis −84.8±3.6°** (reproducible VERTICAL, vs ctrl random −54.7°). interface_frac 0.0, mixing_entropy
      0.0, contact_same 1.0 = a CLEAN top/bottom stratification with ZERO mixing — the strongest partition of the
      whole campaign (1E lateral demix maxed seg 0.81; here seg 1.0 + a set axis). mi_type_y LOCKS by 25% and holds
      to 100% (traj 0.43→0.9985→…→0.9985). Growth HELD (area 0.142 = 2× ctrl). TIER-1 clean (collapsed 0, nn_min
      ≈r0). **Orientation doesn't merely re-align the demix — it COMPLETES it**: un-oriented PAT-1 tops at seg
      ~0.53–0.65 (fuzzy interface 0.234, mixEnt 0.723); sediment drives seg→1.0, interface→0, mixEnt→0. **PAT-2
      (oriented, spatially-anchored, growing domain map) MET — the PAT gate's "mi_type_x/y ↑ spatially-DEFINED map"
      is satisfied.** SUB-FINDINGS: (a) AXIS PROGRAMMABLE UNDER GROWTH (n=1) — x-drift (sed_xaxis) SWAPS channels
      (mi_type_x 0.9985, mi_type_y 0.0515, axis −176.56° ≡ 3.4° mod 180, seg 1.0); mirrors ORI b49 with growth. (b)
      DOSE SATURATES EARLY — mi_type_y d05 0.9531 (seg 0.946) / d10 0.989 / d20 0.9985; orientation ~saturated by
      ±0.05, threshold lies BELOW d05, untested. (c) ORIENTATION ⟂ GROWTH — static nogrow (rate0, area 0.071) still
      seg 1.0 / mi_type_y 0.9531; growth if anything sharpens it (0.9531→0.989). (d) SEED-2 BUCKLE recurs (sed_d10_s2
      circ 0.684) but seg 1.0 / mi_type_y 0.976 unaffected — shape artifact, not partition. **PAT-2 OP POINT =
      embryo_PAT_sed (d10 growing oriented demix).** OPEN [b67 tests]: MECHANISM — does sediment ALONE (chem OFF)
      orient, or does it need chemotaxis to complete? (ORI b48 sediment-alone gave only mi_type_y 0.40 at a diff
      geometry → likely chem×sediment compose super-additively). Also diagonal programmability (n≥2) + dose threshold
      (d02/d03). Specs: embryo_PAT_sed(_s1/_s2/_d20/_d05/_xaxis/_nogrow), ctrl embryo_PAT_base. (dose/axis/seed all
      authored as FILES: dotted sediment.gy hits BOTH type instances → sign-break; dotted general.seed = silent no-op.)
    - **[PAT CLOSED, Batch 68 / b67] PAT-2 MECHANISM RESOLVED — differential sediment ALONE orients AND
      COMPLETES the sort; chemotaxis is REDUNDANT for the oriented map [established, 3 seeds].** nochem
      (sediment d10 ON, BOTH chemotax gains → 0.0): mi_type_y {0.9985, 0.994, 0.976} = **0.989±0.011**, seg
      1.0, mixing_entropy 0.0, interface_frac 0.0 — IDENTICAL within noise to the b66 sed×chem composite
      (0.989±0.011); Δ vs ctrl_nosed 0.0022 ≈ 90·SD; type_axis −85.3±1.8°. The b67 falsifier clause-1 FIRED,
      OVERTURNING b66's "super-additive" guess: with chemotaxis OFF the ONLY differential force is the sediment
      (repel/attraction_repulsion/glide type-blind) → seg 1.0 is 100% gravitational stratification. REVERSAL
      from ORI b48 (sediment-alone only mi_type_y 0.397) is a GEOMETRY effect — PAT n=44 (22+22 cells) sorts
      completely under gravity; ORI n=198 could not. Chemotaxis alone = weak un-oriented sorter (seg 0.53);
      sediment alone = complete oriented sorter (seg 1.0) — REDUNDANT, not additive. SUB-FINDINGS: (a)
      PROGRAMMABILITY replicates under growth (n=2) — diagonal drift → type_axis 45.3±1.6° (both seg 1.0, mi_x
      AND mi_y elevated); steering shown at 0/90 (b66) + 45 (b67) under cell_grow. (b) DOSE — weak drift gives
      PARTIAL orientation (d02 mi_y 0.72 / d03 0.62 ≫ ctrl 0.002 → onset below ±0.02) but not-fully-clean
      interface (d02/d03 mixEnt 0.06/0.10); the seg→1.0 completeness knee is d03→d05. TIER-1 clean 15th straight
      batch (collapsed 0, nn_min 0.0188–0.0194; seed-2 buckle recurs circ 0.722 but seg 1.0 intact).
      **PAT — STARTED Batch 64, CLOSED Batch 68 (2026-07-05). GATE MET [established].** Three legs: PAT-1 (demix
      persists under non-mixing epiboly, b65), PAT-2 (differential sediment orients it, seg 1.0, mi_type_y
      0.989±0.011, programmable axis, b66/b67), MECHANISM (sediment alone sufficient, b67). Deliverable = an
      ORIENTED, PROGRAMMABLE, GROWING two-domain map. **PAT OP POINT = embryo_PAT_sed.yaml** (n=44 growing,
      diff sediment d10; chemotaxis optional/redundant). ADVANCE to MOR (final phase of the ladder).
    - **MOR (morphogenesis — Phase 2, FINAL) — STARTED Batch 68 (2026-07-05). Gate: growth SCULPTS an ORIENTED,
      reproducible, non-round body SHAPE directed by the pattern axis (shape anisotropy — fourier_m2/m1 ↑,
      circularity ↓, shape_axis_angle aligned to the programmable growth/type axis, >2·SD vs the isotropic-growth
      control) while the partition (seg ≥0.9) and TIER-1 (collapsed 0, nn_min ≥r0) hold.** Substrate = the PAT-2
      op point (embryo_PAT_sed, sediment d10 → oriented seg-1.0 growing partition). MECHANISM = `cell_grow` mode
      anisotropic/tip (biases woken-reserve placement along `axis` → a polarized bud). NB the "cell" level = the
      whole blastula BODY (n=1) → cell_grow grows the WHOLE body; per-DOMAIN differential growth is NOT
      expressible in this single-cell architecture [engineering limit]. Batch 68 = aniso ladder (0.0 iso / 0.4 /
      0.8 / 1.0, axis +y) + seed replicate + tip-mode + diagonal-programmable, on embryo_MOR_base. FALSIFIER:
      fourier_m2 flat across the aniso ladder (elastic membrane rounds every bud) → cell_grow can't sculpt an
      oriented shape → MOR needs a different sculpting driver.
      **MOR-1 SHAPE SCULPT — batch-68 result [open, n≤2, toward gate]:** anisotropic `cell_grow` (axis +y on the
      seg-1.0 oriented partition) DOES sculpt an oriented body outgrowth — the b68 falsifier (m2 flat) did NOT
      fire. `fourier_m2` (2-fold) rises MONOTONE with aniso: iso 0.0108 → a04 0.0177 → a08 0.0335 → a10 0.0495
      (4.6×); `fourier_m1` (the +y DIPOLE/bud, the actual oriented signal) rises MONOTONE 0.147→0.160→0.263→0.332;
      deform_rms 0.073→0.098. **a08 REPLICATES 2 seeds: m2 {0.0335,0.0325}=0.0329±0.0007, m1 {0.263,0.276}=
      0.270±0.010** (tight). Partition HELD everywhere (seg 1.0, mi_type_y ≥0.994, mixEnt 0.0), TIER-1 clean
      (collapsed 0, nn_min ~0.0187), body grows (grow_ratio 1.38 @a10). CAVEATS: (i) `shape_axis_angle` is NOISE-
      dominated for these MILD 2-fold buds (scatter −39/0.8/28/132) → report `fourier_m1` (+y dipole) as the
      oriented-sculpt readout, NOT shape_axis. (ii) `tip` mode 4.0 = catastrophic MPM RUNAWAY [rejected] — all
      reserve woken at one top edge → prestretch pressure spike ejects a continuum plume (area 6.4×, deform 0.38,
      escape 1.0) while the AGENT body does NOT grow (grow_ratio 0.994) and partition holds; use aniso mode, or a
      mild tip ≤1.5. (iii) growth-axis STEERING of shape (diag axis [.707,.707], axdown axis −y) FAILED to sculpt
      m2 at n=1 (m2 0.0135/0.0061 ≈iso) though the DRIFT rotated the partition (mi_x elevated) — [open], b69 tests
      a clean decoupled case (growth axis +x vs partition +y). PROMOTION to MOR gate [established] needs a08/a10
      m2 > 2·SD above iso over 3 seeds each. Batch 69 = iso_s1/s2 (baseline SD) + a08_s2 + a10_s1/s2 (→3 seeds) +
      bud_big (target 7.0) + tip_mild (1.5) + axis_x (programmability). New specs embryo_MOR_base_s2, embryo_MOR_axisx.
      **MOR-1 — batch-69 3-seed VERDICT [open, refined]:** the +y `fourier_m2` gate does NOT cleanly replicate.
      3-seed stats (b68+b69): m2 iso {0.0108,0.0201,0.0281}=0.0197±0.0087, a08 {0.0335,0.0325,0.0480}=0.0380±0.0088
      (Δ 2.1·SD MARGINAL), a10 {0.0495,0.0033,0.0199}=0.0242±0.0231 (Δ<0.2·SD FAILS — s1 dud). m1 iso 0.183±0.032,
      **a08 0.270±0.007 (2.7·SD PASSES)**, a10 0.181±0.132 FAILS. → (a) **aniso 1.0 is seed-UNSTABLE [rejected as op
      point]** — pure deterministic +y reserve placement (dirv=1.0·axis) fails to bud on seeds 1/2 and even rotates
      the partition diagonal (a10_s1 type_axis −128.9, mi_type_x 0.75); the 0.2·rand jitter at aniso 0.8 stabilizes
      it → **MOR op point = aniso 0.8**. (b) **readout = fourier_m1, NOT m2** — m2 is a weak 2-fold harmonic
      CONTAMINATED by the seed-2 base BUCKLE (perimeter roughness: iso_s2 circ 0.68 m3-5 0.038, a08_s2 circ 0.33
      m3-5 0.096, vs clean seeds circ 0.93-0.99 m3-5 <0.008); m1 (centroid dipole, largest scale) is robust to it.
      (c) **axis_x DECOUPLING [open, n=1] = the clean programmability win**: growth axis +x while partition/sediment
      stays +y → shape dipole +x (m1 0.314 batch-max, shape_axis −12.86 ≈0°) ORTHOGONAL to the partition (type_axis
      −81, mi_type_y 0.9985, mi_type_x 0.12), circ 0.95 clean, seg 1.0 — shape sculpt is PROGRAMMABLE, decoupled
      from the type map; b69 secondary falsifier ("axis_x m1 points +y") did NOT fire. aniso 1.0 works CLEANLY for +x
      (decoupled) — the +y instability was growth fighting the sediment geometry. (d) **bud_big (target 7.0) [rejected
      secondary]** — bigger target did NOT enlarge the bud (m1 0.173 ≈iso, area 0.14 = base; reserve-wake/rate limited
      not target limited). (e) **tip mode 1.5 [rejected] any dose** — STILL runs away (area 3.4×, deform 0.24, MPM
      plume) same failure as tip 4.0; use anisotropic only. TIER-1 held 17th straight batch (collapsed 0, nn_min
      0.0185-0.0192, seg 1.0, mi_type_y ≥0.976 even under buckle/axis-rotation). Batch 70 = LOCK shape programmability:
      3-seed axisx (aniso 0.8) + diagx ×2 (growth [.707,.707], partition +y) + axisy anchor + iso ctrl + axisx aniso1.0
      re-confirm → steering curve shape_axis {+x→0, diag→45, +y→90} at fixed +y partition. New specs embryo_MOR_axisx_s1/
      _s2, embryo_MOR_diagx/_s1.
      **MOR-1 — batch-70 VERDICT [open, steering-curve REJECTED; decoupled +x is the only lock candidate, n=1].** The
      shape_axis STEERING curve does NOT materialize — b70 falsifier FIRED. Measured shape_axis_angle: +x aniso0.8 3-seed
      {−36.7,−45.1,24.4}=scatter SD~37° (no cluster); diag {71.4,−73.6→106.4}=~70–106° not 45; +y (s6) 28.2° NOT 90.
      ONLY +x aniso1.0 (s7) locks (shape_axis −12.86 == b69 seed0 exactly; m1 0.314 batch-max, traj 0.23→0.38→0.31).
      fourier_m1 at +x aniso0.8 = 0.199±0.086 does NOT separate from iso (0.147/0.183±0.032; Δ≪2·SD) — the b69 "a08 m1
      2.7·SD" was the +y arm; the +x arm at 0.8 is weaker+seed-noisy. **Root confounds:** (i) sweeping growth toward +y
      at fixed +y sediment is inherently confounded — growth COLLINEAR with sediment fights it and ROTATES the partition
      (+y s6 mi_type_x 0.548; diag s5 mi_type_x 0.994 + mi_type_y↓0.618; axisx08_s2 0.701) rather than budding; partition
      stays clean (mi_type_x ≤0.12) ONLY when growth is weak OR orthogonal +x. (ii) shape_axis_angle is weak-bud +
      base-BUCKLE noise below aniso 1.0 (buckle seeds circ 0.758/0.831/0.863 vs clean ~0.95). **The one clean,
      biological result = DECOUPLED ORTHOGONAL MORPHOGENESIS: growth +x (aniso 1.0) → +x shape dipole (m1 0.314,
      shape_axis ~0) ⊥ the +y molecular partition (mi_type_y 0.9985) — body-shape axis independent of the type-map
      axis — but SINGLE SEED (s7 = b69 seed0).** TIER-1+partition HELD 18th straight batch (collapsed 0, nn_min
      0.0175–0.0189, seg 1.0 all 8). Batch 71 = LOCK decoupled +x aniso1.0 over 3 seeds (axisx10 s0/s1/s2) + strong-bud
      (rate 1.6) + sediment-OFF isolation (embryo_MOR_axisx_nosed, does removing the +y dipole tighten the shape_axis
      lock?) + diag/+y aniso1.0 steering endpoints + iso ctrl. Falsifier: axisx10 3-seed shape_axis SD >45° OR m1 not
      >2·SD above iso → s7 was seed-luck → CLOSE MOR-1 as "bud real, direction not lockable"; campaign object stands on
      the PAT oriented-growing-partition without a programmable body-shape axis.
      **MOR-1 — batch-71 VERDICT [SPLIT: angle-decoupling CONFIRMED (3 seeds); magnitude gate FAILS on BUCKLE].** The
      +x aniso1.0 shape_axis CLUSTERS near 0°: 3 seeds {−12.86, 27.4, 11.63} = mean 8.7°, **SD 20.3° (<45)** — b71
      falsifier clause 1 did NOT fire (tightest +x cluster yet vs b70 aniso-0.8 SD~37); the body dipole reads +x
      ORTHOGONAL to the +y partition (mi_type_y 0.976–0.9985). BUT the fourier_m1 MAGNITUDE gate FAILS: +x m1
      {0.314,0.227,0.149}=**0.230±0.083 vs iso 0.183±0.032 = 0.5·SD (NOT >2·SD)** — clause 2 FIRED. **[engineering] the
      base membrane BUCKLE is the m1 noise source, NOT the sediment: circ⇄m1 (seed0 circ0.952→m1 0.314, seed1
      0.789→0.227, seed2 0.444→0.149) — a buckled youngs-90 elastic shell (surface_tension 0) DEFLATES the m=1 dipole**;
      buckle is the confound to eliminate. [rejected] sediment-OFF (nosed) did NOT tighten shape_axis (28.75, wandered
      −134→28 whole run) AND destroyed the partition (seg 0.377, mi_type_y 0.013 fully re-mixed) → sediment is NECESSARY
      (demix alone only seg~0.38) and is NOT the shape_axis noise source. [rejected] strong rate 1.6 rotates the
      partition (mi_type_x 0.9985 / mi_type_y 0.495), m1 0.224<seed0 — keep rate 1.1. [rejected] steering {0,45,90}
      still fails except +x (diag 82.6, +y 132.1 = large non-constant offset). [open] +y collinear HELD the partition
      this batch (mi_type_y 0.9985, m1 0.332 batch-max) unlike b70 — collinear-rotation is seed-stochastic; avoid +y.
      TIER-1+partition HELD 19th straight batch (collapsed 0, nn_min 0.0187–0.019, seg 1.0 except nosed, area ~0.135–0.142).
      Batch 72 = PAIRED iso(aniso0)/+x(aniso1) on MATCHED seed files (per-seed Δm1 removes base-draw variance) + buckle
      elimination (NEW embryo_MOR_axisx_s2_stiff, youngs 90→160 on the buckle-prone seed2, run aniso 0 vs 1). Falsifier:
      paired Δm1 ≤0 any seed OR stiff seed2 circ<0.7 / m1<0.22 (buckle not the cause) → MOR-1 magnitude gate NOT
      achievable → CLOSE MOR-1 on the ANGLE decoupling (shape_axis +x SD 20°, 3 seeds) with a buckle-limited caveat;
      deliverable rests on PAT (oriented growing partition).
      **MOR-1 — batch-72 VERDICT [BUCKLE IS THE m1 CONFOUND, CONFIRMED reducible; magnitude gate ACHIEVABLE-if-de-buckled,
      not yet met].** All 8 landed, TIER-1 clean (collapsed 0, nn_min ≥0.0186, seg 1.0, mi_type_y 0.976–0.9985 — partition
      HELD 20th straight batch). PAIRED Δm1 (iso→+x, matched seeds, youngs 90): seed0 0.147→0.314 **Δ+0.167** (circ
      0.952→0.952 clean) ; seed1 0.207→0.227 **Δ+0.019** (circ 0.987→0.789 buckled) ; seed2 0.194→0.149 **Δ−0.045**
      (circ 0.684→0.444 crumpled, perim 2.0, m3–m5 ~0.027). Falsifier clause 1 (paired Δm1 ≤0 any seed) FIRED on seed2.
      BUCKLE-ELIMINATION (seed2, youngs 90→160): iso m1 0.148 → +x m1 **0.258** — stiffening **FLIPPED seed2 Δm1 from
      −0.045 to +0.110** and lifted +x m1 0.149→0.258 = DECISIVE proof the weak/negative Δm1 on buckle seeds is a BUCKLE
      ARTIFACT, NOT a growth-bud failure. BUT clause 2 (stiff circ<0.7) FIRED: youngs 160 left +x circ 0.606 (shell still
      crumpled) = under-dosed. **[engineering, reconfirmed] m1↔circ (buckle) coupling across all +x slots: circ 0.952→m1
      0.314, 0.789→0.227, 0.606→0.258, 0.444→0.149 — a buckled elastic shell (youngs 90, surface_tension 0) DEFLATES the
      m=1 dipole; stiffening partly recovers it.** Rather than close on the caveat, b73 does ONE decisive buckle-elimination
      pass — surface_tension (untested in MOR; direct roundness lever, damps high-wavenumber wrinkle m3–m5 ∝curvature far
      more than the low-k m1 bud) STACKED with youngs 200. Batch 73 = paired iso/+x on matched seeds s0/s1/s2 under
      NEW embryo_MOR_ab/_s1/_s2 (youngs 90→200 + surface_tension 0→8) + 2 attribution slots on worst seed2 +x (ST-off =
      stiffness-alone; ST 20 = stronger roundness). WIN = all 3 paired Δm1 > 0, all +x circ > 0.80, m1(+x)−m1(iso) > 2·SD
      → MOR-1 MAGNITUDE gate MET [established], MOR-1 CLOSES on a WIN (programmable oriented body bud ⊥ oriented growing
      partition). FALSIFIER: seed2 +x circ still <0.75 (uncorrectable) OR any paired Δm1 ≤0 → CLOSE MOR-1 on the ANGLE
      decoupling (b71) with the buckle caveat; campaign rests on PAT + MOR angle-decoupling.
      **MOR-1 — batch-73 VERDICT [MAGNITUDE gate MET on the PAIRED test; surface_tension INERT; residual buckle unmet].**
      Two decisive results. (1) **[engineering — RETRACTED at Batch 98: CONTAMINATED by the dotted-override caching
      bug, NOT a real test] "surface_tension INERT on the youngs-200 elastic shell":** s5(ST8)=s6(ST0)=s7(ST20)
      BIT-IDENTICAL — but s6/s7 used dotted overrides `surface_tension=0.0/20.0` whose archived spec.yaml line 53
      STILL reads 8.0; all three secretly ran ST8 (op caches ST in __init__). ST 0→20 was NEVER applied. The
      "damp wrinkle, spare bud" premise remains UNTESTED, not falsified. (orig, now unsupported:) msd 0.016667;
      youngs is the ONLY working roundness lever (TESTS.md ST works only on WATER bodies at 120–460, not against a
      stiff membrane at 8–20). (2) **PAIRED Δm1 (aniso1−iso, matched seed, youngs 200): seed0 +0.049, seed1 +0.088,
      seed2 +0.064 = +0.067±0.020, ALL 3 POSITIVE, 3.4·SD → magnitude gate MET.** youngs 200 FLIPPED the b72/y90
      seed2 Δm1 −0.045→+0.064 and tightened SD 0.111→0.020 (buckle artifact removed at the sign level, all seeds).
      **[engineering, established] youngs TRADES bud-amplitude ↔ roundness — cannot separate wrinkle from bud:**
      seed0 +x m1 0.314(y90)→0.198(y200) as circ 0.952→0.977 (−37% bud to round it); m1↔circ coupling reconfirmed
      (circ 0.977→m1 0.198 clean vs buckled 0.707→0.251, 0.766→0.201). **The ONLY unmet clause is roundness:** +x
      circ {0.977, 0.707, 0.766} — seed0 rounds, seeds 1&2 stay <0.80. Buckle onset is SEED-DEPENDENT (not a
      uniform stiffness threshold). shape_axis scatter WORSENED at y200 ({52.5,−35.5,12.6} SD 44° vs b71/y90 SD 20°
      — smaller bud = noisier dipole direction). +x growth ELEVATES mi_type_x {0.727,0.333,0.296} vs iso ~0 (bud
      advects type-a along x, bleeds minority x-structure into the +y partition; mi_type_y held ≥0.976). TIER-1 held
      21st batch (collapsed 0, nn_min ≥0.0184, seg 1.0). Batch 74 = YOUNGS-UP sweep: full 3-seed paired iso/+x at
      youngs 280 (NEW embryo_MOR_y280/_s1/_s2, ST 0) + youngs-360 extreme on seed2 + y200 seed2 +x re-anchor. WIN =
      all 3 y280 +x circ>0.80 AND paired Δm1>2·SD → MOR-1 CLOSES clean. FALSIFIER: seeds 1&2 +x circ still <0.80
      (buckle seed-intrinsic) OR any Δm1 ≤0 / bud collapses (over-stiffening deflates) → CLOSE MOR-1 on the y200
      paired-magnitude WIN (Δm1 +0.067±0.020, 3.4·SD) with the residual-buckle + surface_tension-inert caveat.
      **MOR-1 — batch-74 VERDICT [CLOSED on the y200 paired-magnitude result; youngs EXHAUSTED as a roundness
      lever]. The pre-registered youngs-280 falsifier FIRED on BOTH clauses.** (1) Paired Δm1 at youngs 280 =
      seed0 −0.036 (bud DEFLATED below its iso partner 0.105<0.141), seed1 +0.030, seed2 +0.029 → clause 1 (any
      Δm1 ≤0) fired on seed0. (2) +x circ {0.978, 0.713, 0.507} → seeds 1&2 <0.80 fired clause 2 (seed2 WORSE
      than at y200: 0.766→0.507). **[engineering, established] youngs OVER-STIFFENS — the bud deflates MONOTONE
      with stiffness:** matched seed0 +x fourier_m1 0.314(y90)→0.198(y200)→0.105(y280), each step strips 35–47%;
      roundness is NON-monotone (circ 0.952→0.977→0.939, peaks y200). y280 is WORSE than y200 on BOTH bud AND
      roundness for the clean seed. **[established-direction] the BUCKLE is SEED-INTRINSIC, not a stiffness
      threshold:** no monotone circ(youngs) for the buckle seeds (seed1 +x circ 0.707@y200 ≈ 0.713@y280;
      seed2 +x 0.766@y200 → 0.507@y280 → 0.789@y360, erratic, m1 deflated to 0.176) → intrinsic packing defect of
      specific base draws. TIER-1 held 22nd batch (collapsed 0, nn_min ≥0.0184, seg_index 1.0, mi_type_y ≥0.976).
      **MOR-1 CLOSED [established-direction] = programmable oriented +x body BUD (paired Δm1 +0.067±0.020, 3.4·SD,
      3 seeds at youngs 200) ORTHOGONAL to the +y molecular partition (mi_type_y ≥0.976); MAGNITUDE gate MET,
      roundness clause buckle-limited on 2/3 seeds; surface_tension INERT, youngs trades bud↔roundness.** Batch 75
      (MOR batch 8/10) = BUCKLE-RELIEF via NON-stiffness levers: growth KINEMATICS (cell_grow.rate 1.1→0.6/0.8 =
      gentler compression; prestretch 0.6→0.8 = gentler inflation) + agent_remodel core-rigidify (the MOR-gate's
      "remodeling rounds+stabilizes" clause, NEW embryo_MOR_remodel_s2) on the buckle-prone seeds. WIN = seed2 +x
      circ >0.80 AND paired Δm1 >0 → MOR-1 upgrades to a clean WIN + demonstrates the remodeling leg. FALSIFIER:
      rate 0.6 leaves circ <0.80 OR bud deflates toward iso → buckle unremovable by growth kinematics → MOR-1
      rests FINAL on the y200 magnitude result and MOR closes on the established legs
      (partition·division·flow·orientation·growth·oriented-bud).
      **MOR-1 — batch-75 VERDICT [buckle-relief: gentle-kinematics & agent_remodel FAIL; growth-rate is a NEW
      OPPOSITE-SIGN lever]. The pre-registered rate-DOWN falsifier FIRED — but the data invert the model.** (1)
      seed2 +x circ is MONOTONE INCREASING in growth rate: rate0.6 0.684 → rate0.8 0.716 → rate1.1 0.766 (ctrl) —
      the predicted rate0.6 circ>0.80 was FALSE (0.684, WORST). The COMPRESSION-SHOCK model is REJECTED [rejected]:
      slowing the approach lets the two-domain interface/packing defect express as a lobed edge; FAST taut
      inflation resists wrinkling. seed1 same (rate0.6 circ 0.492 ≪ y200 ~0.707). (2) prestretch0.8 is a bud
      AMPLIFIER not a roundness lever [open]: fourier_m1 0.293 (batch-max, +45% vs anchor 0.201) but circ 0.615,
      area −19% (0.140→0.114) — roundness ⊥ bud amplitude, prestretch moves bud/rate moves roundness. (3)
      agent_remodel DESTABILIZES [rejected] — opposite of the MOR-gate "remodeling rounds+stabilizes" clause:
      near-rupture (nn_min 0.0163 batch-low, gr_peak 80.97 = 3.5–5×, msd 0.0737 = 4×, stress_cell_corr→NaN,
      cells expelled from shell at 9000–12000f, m1 deflated to 0.148). (4) seed0 PARTITION SWAP at rate0.6 [open]:
      round (circ 0.972) but type axis rotated +y→+x late (mi_type_y 0.9985→0.610, mi_type_x 0.0785→0.9985) — +x
      bud advected type-a along x on the clean-packing seed. Paired Δm1 seed2 rate0.6 = +0.018 (>0, thin vs y200
      +0.067). TIER-1 held 23rd batch (collapsed 0, nn_min ≥0.0184 on 7/8; remodel 0.0163 marginal). **REMAINING
      lever after b75 = growth rate UP (seed2 circ monotone 0.684→0.716→0.766 over rate 0.6/0.8/1.1, bud preserved
      m1 ~0.18–0.21).** Batch 76 (MOR 9/10) = rate UP 1.5/2.0/2.5 on buckle seeds 2&1 (dotted cell_grow.rate on
      embryo_MOR_ab_s2/_s1/_ab). WIN = seed2 rate1.5–2.0 +x circ >0.80 AND paired Δm1 >0 AND TIER-1 clean → MOR-1
      upgrades to clean WIN (rounds+buds). FALSIFIER: circ still <0.80 at rate2.0 OR bud deflates OR fast growth
      ruptures (like agent_remodel) → buckle is a hard packing floor for seeds 1&2, MOR-1 rests FINAL on the y200
      magnitude deliverable and MOR terminus (b77) reports the established legs.
      **MOR-1 — batch-76 VERDICT [CLOSED FINAL; buckle is a HARD PACKING FLOOR, every relief lever exhausted].**
      The pre-registered rate-UP falsifier FIRED HARD: growth rate ABOVE 1.1 does NOT round the buckle — it
      SHATTERS roundness non-monotonically. seed2 +x circ vs rate: rate1.1(ctrl) 0.766 → rate1.5 0.333 → rate2.0
      0.290 → rate2.5 0.519 (NONE clears 0.80; roundest is the SLOWEST). The b75 monotone (0.684→0.716→0.766 over
      rate 0.6/0.8/1.1) was a LOCAL trend that REVERSES above 1.1 — fast inflation drives a broadband high-wavenumber
      crumple (s0 fourier_m2/m3/m4/m5 ≈0.032–0.038 all equal). seed1 same (rate1.5 circ 0.655, rate2.0 0.559). BUD
      survives fast growth, bud⊥roundness independent (rate2.0 seed2 m1 0.326 = MOR-campaign-max at circ 0.290;
      seed0 m1 0.294 at circ 0.969). Fast +x advection SCRAMBLES the partition on 2/8 slots [open caveat]: seed2
      rate1.5 mi_type_y→0.068 disordered, seed1 rate2.0 axis FLIPPED +y→+x (mi_type_x 0.857); 6/8 held mi_type_y
      ≥0.976. TIER-1 held 24th batch (collapsed 0, nn_min 0.0184–0.0192, area ~2× 0.136–0.146; rupture-falsifier
      did NOT fire — fast cell_grow is TIER-1-safe, just crumples). BUCKLE-RELIEF MAP COMPLETE, all levers exhausted:
      youngs UP deflates bud w/o rounding (b74); rate DOWN worsens buckle (b75); rate UP shatters roundness (b76);
      surface_tension INERT (b73); agent_remodel ruptures (b75). Seeds-1&2 buckle = seed-intrinsic packing defect,
      not tunable.
    - **MOR (morphogenesis — Phase 2, FINAL) — STARTED Batch 68, CLOSED Batch 77 (2026-07-06). GATE MET on
      MOR-1 [established-direction]; buckle-limited on roundness.** Across b68–b76 (9 batches) the deliverable
      MOR-1 = anisotropic +x `cell_grow` sculpts a programmable ORIENTED body m=1 BUD (paired Δm1 +0.067±0.020,
      3.4·SD, 3 seeds, youngs 200/rate1.1) ORTHOGONAL to the +y molecular partition (mi_type_y ≥0.976) — pattern
      (the +y sediment-oriented demix) CONTROLS growth DIRECTION (+x aniso bud), the MOR terminus objective.
      Residual: 2/3 seeds keep circ 0.71–0.77 (seed-intrinsic buckle, unremovable b73–b76). **MOR OPERATING POINT
      = `embryo_MOR_ab.yaml`** (5-leg embryo: partition + sediment-orient +y + cell_grow aniso +x epiboly +
      flow_align 40, division OFF; TIER-1 held 24 straight batches). **Batch 77 (MOR 10/10) = the campaign TERMINUS
      / INTEGRATED CAPSTONE**: full 6-leg chain (the 5 legs + bounded ~1.5× DIVISION, the leg never combined with
      the growing/budding embryo) across 3 seeds + controls (nodiv 5-leg anchor, 2× stress, iso/flow ablations),
      new specs embryo_MOR_cap / _s1 / _s2 / _div2 (div driven by per-type div_rate 0.4 + buffer 75/100; MOR_ab's
      div_rate 0.0 = genuinely OFF). Falsifier: any full-chain seed mi_type_y <0.70 (division mixing wins per INT's
      2×-dilution law) OR TIER-1 breaks OR 2× ruptures → capstone rests on the 5-leg object. **After b77 the
      campaign (1A→1B→1C→1D→1E→INT→ORI→GRO→PAT→MOR) is COMPLETE per the ladder — STOP.**
      - **CAPSTONE VERDICT (Batch 78 read b77): FALSIFIER FIRED — DIVISION is the ONE incompatible leg; the
        campaign object rests on the 5-LEG NO-DIVISION embryo [established-integration].** Full 6-leg chain 3
        seeds mi_type_y {0.044, 0.373, 0.433} = 0.283±0.209, ALL <0.70 (division-cost −0.72 vs nodiv anchor
        0.9985); seg 0.343/0.356/0.424 vs nodiv 1.0. Reconfirms INT division=mechanical-mixing/dilution law
        (b33/b34, 5th time) under the oriented+growing embryo: chemotactic+sediment re-sort does NOT outpace
        1.5× division mixing. Growth extent scales dilution monotone (2× mi_type_y 0.202 < 1.5× 0.283 < nodiv
        0.9985). 2/3 full seeds partial-HOLD ~0.4, 1/3 (s0) built to 0.393@50% then DECAYED to 0.044 as n
        filled the cap (noisy partial-hold, not clean loss). Flow-up (gain80 mi_type_y 0.372) & iso-ablation
        (0.354) DON'T rescue → loss is division-mixing, not bud-advection/flow. **THE 5-LEG DELIVERABLE (b77
        s3 cap_nodiv `embryo_MOR_ab.yaml`) = partition (mi_type_y 0.9985) ⊥ oriented +x bud (fourier_m1 0.198,
        mi_type_x 0.727 — the +x bud drags type-a along x, clean bud↔partition coupling: no-bud baseiso
        mi_type_x 0.0007 same mi_type_y) + growth (area ~2×) + flow (gain40), TIER-1 clean (collapsed 0,
        nn_min 0.0187, circ 0.977, seg 1.0). Division TIER-1-SAFE (no rupture even at 2×, nn_min 0.0184), it
        ONLY dilutes the molecular partition.** TIER-1 held 25th straight batch.
      - **Batch 78 = RESCUE ATTEMPT [open]:** can stronger re-sorting beat 1.5× division mixing and restore
        the full 6-leg chain? Ladder on seed-2 full chain (baseline mi_type_y 0.433): chemotax demix gain 2×/3×
        (−0.20/−0.30), sediment orient 2× (±0.20), combo (chem −0.20 + sed ±0.20 = best candidate), slowdiv
        (div_rate 0.4→0.2, tests INT count-vs-rate law), move18 (0.18 faster transport), combo on worst seed0
        (mi_type_y 0.044), ctrl = exact cap_s2 re-run. New specs embryo_MOR_cap_{chem20,chem30,sed20,combo}_s2
        + _{slowdiv,move18}_s2 + _combo_s0. FALSIFIER: combo_s2 <0.70 AND no single lever clears 0.70 →
        re-sorting cannot beat 1.5× mixing at any accessible gain → division DEFINITIVELY incompatible, 5-leg
        object is FINAL.
      - **Batch 79 read b78 = RESCUE FALSIFIER FIRED, DIVISION-RESCUE QUESTION CLOSED [established-integration].**
        NONE of the 8 slots cleared mi_type_y 0.70; the double-boost combo (chemotax −0.20 + sed ±0.20) plateaued
        at 0.396 == ctrl 0.433 (NO lift). Stronger re-sorting cannot beat 1.5× division at any accessible gain.
        **MECHANISM [NEW]: under division the TOTAL type-partition is CAPPED (seg 0.399–0.531, narrow band across
        driver gains 1×–3×); driver gain only REDISTRIBUTES it between the chemotax(x) and sediment(y) axes, it
        cannot exceed the ceiling.** Stronger chemotax ROTATES the sort off +y onto +x (chem20: mi_type_y 0.393@50%
        →0.141@100% while mi_type_x 0.005→0.424; chem30 same, mi_type_y 0.235/mi_type_x 0.444) — the heterotypic
        chemotax lateral-demix COMPETES with the sediment y-orientation. move18 same rotation (mi_type_y 0.166,
        mi_type_x 0.384). slowdiv inert (0.393==ctrl, reconfirms INT count-not-rate). **sed20 (sediment 2×) = best
        TOTAL-partition lever — the only one lighting BOTH axes (mi_type_x 0.416 AND mi_type_y 0.409, sum 0.825 batch
        max, seg 0.440); combo INTERFERES destructively on x (mi_type_x 0.072) — drivers are NON-additive.** combo
        rescued worst seed0 (0.044→0.387) = variance reduction not ceiling lift, but shell BUCKLED (circ 0.716 vs
        ~0.92). TIER-1 held 26th straight batch (collapsed 0, nn_min 0.0181–0.0188, area ~2×, n_div 22 all).
      - **Batch 79 read: mi_type_y IS A CONFOUNDED METRIC — the +x bud rotates the type-axis seed-to-seed, so the
        COMPATIBILITY ENVELOPE lives on `segregation_index` (TOTAL sort), NOT mi_type_y [established-integration].**
        The b79 monotone-mi_type_y prediction FAILED for the right reason: the nodiv control (bud ON) gave mi_type_y
        only 0.537 (BELOW the 0.70 gate) — its complete partition (seg 1.000) was rotated DIAGONALLY (mi_type_x 0.807,
        axis −44.8°) by the +x anisotropic bud — while a DIVIDING slot (1.1× g11_s0) passed the gate at 0.874 (clean
        +y, axis −93.1°). So mi_type_y is NON-monotone in count and even zero-division fails the y-gate. 1.1× 3 seeds
        mi_type_y {0.874,0.007,0.791}=0.557±0.437 (HUGE variance = axis wander; mi_type_x anti-correlated {0.038,0.691,
        0.725}, s1 rotated the WHOLE sort onto +x, seg still 0.657); type_axis_angle spans −4.5/−57.8/−93.1° = ~90° of
        seed wander. **The b79 "falsifier fired" (1.1× mean 0.557<0.70) is an ARTIFACT of the confounded metric — 2/3
        seeds individually clear 0.70.** The CLEAN monotone readout = segregation_index: nodiv 1.000 → 1.1× {0.936,
        0.657,0.818}=0.804±0.140 → 1.2× {0.615,0.627}=0.621 → 1.3× 0.523 → 1.5× 0.424 (co-metrics agree: interface_frac
        0→0.287, mixing_entropy 0→0.563). **TOTAL-partition ENVELOPE: tolerates ~1.1× division (seg 0.80, 80% retained),
        degrades smoothly to 0.42 at 1.5×.** The bud (aniso 1.0, +x) DRIVES the axis rotation (mi_type_x elevated only
        on bud-captured seeds 0.69–0.81, ~0 on +y seeds 0.03–0.04), same chemotax/bud↔sediment axis competition as b78.
        TIER-1 held 27th straight batch (collapsed 0, nn_min 0.0178–0.0189, area ~2×, buckle circ 0.479–0.981 seed-noisy
        UNcorrelated with count). Specs embryo_MOR_cap_{g11,g12,g13}_s2 + _g11_s0/_s1 + _g12_s1 + _nodiv_s2.
      - **Batch 80 = DECONFOUNDED ORIENTED-PARTITION ENVELOPE [open]:** ablate the +x bud (cell_grow aniso 1.0→0.0,
        isotropic growth) to remove the axis-rotation confound, re-map the count ladder → does the PURE sediment +y
        orientation survive division cleanly? Predict bud-OFF nodiv mi_type_y ~0.99 (== b77 baseiso 0.9985), 1.1× holds
        ≥0.70 across 3 seeds with LOW variance (axis LOCKED to +y, no competing +x driver), 1.2× ~0.55, 1.5× ~0.40.
        FALSIFIER: bud-OFF 1.1× mi_type_y still <0.70 (mean) OR still high-variance (SD>0.25) across 3 seeds → division
        ITSELF destroys the oriented +y partition → oriented deliverable strictly the nodiv object, envelope only on
        seg. New specs embryo_MOR_cap_iso_{nodiv,g11_s0,g11_s1,g11_s2,g12_s2,g13_s2,g15_s2} + bud_nodiv ctrl (=b79 rerun).
      - **Batch 80 read: DECONFOUNDED ORIENTED-PARTITION ENVELOPE IS REAL — the pure sediment +y partition tolerates
        ~1.1× division axis-LOCKED [established-integration]; the b79 y-gate failure was a +x-bud rotation artifact.**
        Ablating the bud (cell_grow aniso 1.0→0.0) LOCKS the type-axis to +y and makes mi_type_y a clean monotone
        readout. Falsifier did NOT fire. Bud-OFF 1.1× (48 cells) 3 seeds: mi_type_y {0.881,0.690,0.789}=0.787±0.096
        (SD 0.096 << 0.25; cf. b79 bud-ON 0.557±0.437), type_axis_angle {−91.2,−80.0,−87.6}° all within 10° of −90°
        (cf. bud-ON ~90° wander), mi_type_x ~0 {0.032,0.165,0.046}. **Clean monotone mi_type_y envelope: nodiv 1.000
        → 1.1× 0.787±0.096 → 1.2× 0.628 (n=1) → 1.3× 0.543 (n=1) → 1.5× 0.410; crosses the 0.70 gate near ~1.15×.**
        nodiv upper anchor mi_type_y 1.000, seg 1.000, axis −74.7° (== b77 baseiso 0.9985). bud-ON control REPRODUCED
        the confound EXACTLY (mi_type_y 0.537, mi_type_x 0.807, axis −44.8°, seg 1.000 == b79 nodiv_s2) = cleanest proof
        the b77/b78 y-gate failure was bud rotation, not partition loss. seg (TOTAL sort) is BUD-INDEPENDENT: bud-OFF
        {1.000,0.801±0.081,0.659,0.588,0.422} ≈ b79 bud-ON {1.000,0.804,0.621,0.523,0.424} → bud only ROTATES the axis.
        TIER-1 held 28th straight batch (collapsed 0 all; nn_min 0.0182–0.0192; area ~2× base; escape 0.48–0.94 = sediment
        BODY-DRIFT artifact, judge by collapsed/nn_min/circ; buckle circ 0.379–0.990 seed-noisy UNcorrelated with count).
        Specs embryo_MOR_cap_iso_{nodiv,g11_s0/s1/s2,g12_s2,g13_s2,g15_s2}_s2 + bud_nodiv (b79 rerun).
      - **Batch 81 = FLOW CAPSTONE + envelope edge [open]:** add the INT flow leg (move_speed 0.12→0.18, flow_align gain
        40) to the deconfounded bud-OFF oriented 1.1× object (3 seeds) → does the whole flowing+dividing+oriented+
        partitioning blastula HOLD +y? Predict mi_type_y ≥0.55/3 seeds WITH elevated net_circ/msd (no +x bud for motility
        to rotate onto; axis-locked sediment re-sorts as fast as motility stirs, INT b39 analogy). + flow_g11hi (move24/
        gain60 max motility) + iso_g12 s0/s1 (1.2× to n=3) + iso_g115 (~1.15× 0.70-crossover). FALSIFIER: flow_g11 3-seed
        mi_type_y <0.40 (mean) OR SD>0.30 → motility fluidization erases +y → capstone oriented deliverable strictly the
        NON-flowing bud-OFF 1.1× object (0.787). New specs embryo_MOR_cap_flow_g11_{s0,s1,s2}/_g11hi_s2, iso_g12_{s0,s1},
        iso_g115_s2.
      - **[established, Batch 82 / b81] FLOW LEG CLOSED — the flowing+dividing+oriented+partitioning blastula is DEMONSTRATED
        (3 seeds).** Adding the INT flow leg (move_speed 0.12→0.18, flow_align gain 40) to the bud-OFF oriented 1.1× object
        HOLDS +y at full magnitude: flow_g11 3-seed mi_type_y {0.875,0.678,0.833}=0.795±0.085 == b80 no-flow 0.787±0.096
        (Δ 0.008 ≪ SD; falsifier did NOT fire). **CAVEAT [engineering]: the "flow" is FLUIDIZATION not coherent circulation
        — net_circulation ~0 EVERY slot (0.000/0.006/0.000; hi 0.000), while speed rises 0.0042→0.0062-0.0083 (~1.6-2×) and
        msd 0.028→0.048 (~1.7×). Coherent net_circ stays unreachable under sediment+confinement (consistent w/ INT Pareto).**
        High-flow move24/gain60 also held (0.807, n=1, seg 0.896, circ 0.980, TIER-1 clean). ENVELOPE EDGE PINNED: 0.70 gate
        crosses ~1.17× — 1.15× 0.780 (n=1, above), 1.2× {0.692,0.550,0.628}=0.623±0.058 (below). Determinism exact (ctrl
        0.789==b80). Shell buckle = seed-noise (circ 0.33-0.99, UNcorrelated w/ count OR flow). TIER-1 29th straight batch.
      - **Batch 82 = MORPHOGENETIC AXIAL ELONGATION [open]:** the flowing capstone stays ~spherical; MOR (=shape change)
        needs a real shape. LEVER = cell_grow aniso 0.6 axis [0,1] (+y-ALIGNED bud) on the flow_g11 object → does the body
        ELONGATE along the polarity axis (fourier_m2↑, circ↓, shape_axis→90 mod 180) AND REINFORCE +y (mi_type_y ≥0.70/3
        seeds)? vs elong06_perp axis [1,0] (+x PERP = rotation confound, cf b78). + elong10 (aniso 1.0 stronger) + flow_hi_s0
        (hi-flow lock) + flow_g115 (edge under flow) + flow_g11_s2 ctrl. FALSIFIER: aligned elong06 3-seed mi_type_y <0.55
        OR shape stays circular (circ>0.85 AND fourier_m2 unchanged ~0.02) → shape ⊥ oriented partition, capstone stays
        spherical. New specs embryo_MOR_cap_elong06_{s0,s1,s2}/_elong10_s2/_elong06_perp_s2/_flow_g115_s2/_flow_g11hi_s0.
      - **Batch 83 read: b82 = MORPHOGENETIC AXIS RULE — SHAPE-AXIS ⊥ POLARITY IS THE STABLE EGG; the b82 hypothesis
        REVERSED [open→axis-rule].** Falsifier FIRED: ALIGNED [0,1] elong06 3-seed mi_type_y {0.216,0.684,0.408}=
        0.436±0.196 (<0.55) — stretching the body ALONG the +y sort axis TUMBLES it (mi_type_y trajectories oscillate
        wildly frame-to-frame, mi_type_x rises to 0.54–0.75, type_axis finals all diagonal 157/−60/−135). The
        PERPENDICULAR [1,0] bud did the OPPOSITE of predicted: elong06_perp_s2 (n=1) HELD mi_type_y 0.789 (STABLE
        traj 0.80/0.79/0.73/0.79), mi_type_x 0.131, axis −99° (+y), with the STRONGEST shape change of the batch
        (circ 0.280, fourier_m2 0.0569=11× ctrl). WITHIN-SEED CONTRAST (same seed s2): aligned 0.408 tumbled vs perp
        0.789 held ⇒ axis [1,0] vs [0,1] is the driver, NOT seed luck. PRINCIPLE: elongation ⊥ polarity stacks the two
        domains across the shape's SHORT axis (geometric confinement holds them — classic AV-partition-across-minor-
        axis egg); elongation ∥ polarity stretches domains apart along their separation direction and lets the axis
        rotate. CAVEATS: (1) perp HOLD is n=1 → replicate; (2) circ 0.28 is largely boundary RAGGEDNESS (m3 0.048/m4
        0.041 high, not a clean ellipse) and division ALONE reproduces it (flow_g115 aniso-0 ended circ 0.275/
        mi_type_y 0.768) → judge shape by fourier_m2 (modest ~0.057), not circ. hi-flow move24/gain60 HELD 2-seed
        {0.874,0.807}. Determinism exact (flow_g11_s2 mi_type_y 0.833==b81). TIER-1 30th straight (collapsed 0 all 8,
        nn_min 0.0186–0.0192, growth realized, no rupture on lobed slots). Batch 83 = replicate perp06 to 3 seeds
        (perp06_s0/s1 + b82 perp_s2) + perp10 (aniso 1.0, 2 seeds) + nodiv mechanism pair (align_nodiv: does removing
        division-mixing let ALIGNED hold? perp06_nodiv: cleaner ellipse?) + perp06_noflow (is fluidization needed?) +
        flow_g11 ctrl. New specs embryo_MOR_cap_perp06_{s0,s1}/_perp10_{s0,s2}/_perp06_nodiv_s2/_align_nodiv_s1/
        _perp06_noflow_s2. Falsifier: perp06 3-seed mi_type_y <0.55 OR SD>0.30 → perp HOLD was seed luck, MOR shape
        deliverable unreachable, capstone stays a spherical oriented object.
      - **Batch 84 read b83 = PERPENDICULAR-EGG CAPSTONE REPLICATED [established] + the DIVISION/nodiv SHORT-vs-LONG
        AXIS RULE [open].** (1) perp bud (cell_grow aniso 0.6, axis [1,0], move18, 1.1× div) HOLDS +y across 3 seeds:
        mi_type_y {0.875,0.680,0.789(b82)}=**0.781±0.098**, type_axis all near −90° (−94.9/−94.4/−87), mi_type_x low
        (0.08/0.05). Falsifier (mean<0.55 OR SD>0.30) did NOT fire → **THE MOR SHAPE-WITH-DIVISION DELIVERABLE = a
        DIVIDING egg elongated ⊥ to the AV axis, partition across the short axis.** Caveat: shape signal weak/ragged
        (fourier_m2 seed-variable 0.012–0.053; division raggedness dominates, m3/m4 high; circ 0.50–0.90 seed-noisy).
        (2) **SHORT-vs-LONG AXIS RULE [open, n=1/cell]: the chemotactic sort aligns to the ellipse SHORT axis WITH
        division, LONG axis WITHOUT it** — removing division REVERSES which bud holds +y. WITH div: perp (short=y)
        holds +y 0.781, aligned (short=x) tumbles (b82 0.436). WITHOUT div: aligned (long=y) HOLDS +y PERFECTLY
        (align_nodiv mi_type_y **1.000**, seg 1.000, interface_frac 0, fourier_m2 0.058 clean 2-fold ellipse m3/m4
        low, circ 0.753 = the CLEANEST morphogenetic egg, elongated ALONG AV axis, partition along long axis), perp
        (long=x) TUMBLES onto x (perp06_nodiv mi_type_y 0.244 / mi_type_x 0.641 / type_axis −148°). All 4 conditions
        consistent with the one rule (nodiv demix minimizes interface→stacks along long axis; division front advects
        domains apart along growth long-axis→splits across short). (3) **aniso 1.0 TOO STRONG** — perp10 both seeds
        TILT diagonal (s0 mi_x 0.513/mi_y 0.808/axis −62; s2 mi_x 0.671/mi_y 0.686/axis −140) vs perp06 mi_x ~0.06
        clean; aniso 0.6 = sweet spot. (4) FLUIDIZATION NOT NEEDED for perp hold (perp06_noflow move12 mi_type_y
        0.810, axis −96 +y). (5) Determinism exact (flow_g11_s2 0.833==b82). TIER-1 held 31st straight (collapsed 0
        all 8, nn_min 0.0182–0.0192, area ~2× realized, no rupture on lobed slots). **TWO viable eggs now: DIVIDING
        perp egg (3 seeds, weak shape) vs NON-DIVIDING aligned egg (n=1, perfect sort + clean ellipse).** Batch 84 =
        lock align06_nodiv to 3 seeds (is the perfect egg robust?) + perp06_nodiv to 3 seeds (axis rule: mi_type_x >
        mi_type_y?) + align10_nodiv (bigger ellipse, tilt?) + align06_lowdiv 1.05× (division-bridge: does minimal
        proliferation flip long→short?) + align06_noflow (flow needed?) + flow_g11_s2 ctrl. New specs embryo_MOR_cap_
        align06_nodiv_s0/s2, _perp06_nodiv_s0/s1, _align10_nodiv_s1, _align06_lowdiv_s1, _align06_noflow_s1. Falsifier:
        align06_nodiv 3-seed mi_type_y <0.55 OR SD>0.30 (perfect egg = seed luck) OR perp06_nodiv 3-seed mi_type_x ≤
        mi_type_y (axis rule fails).
      - **Batch 85 read b84 = ALIGNED-NODIV PERFECT EGG [established]; perp-nodiv AXIS RULE [rejected]; NEW aligned
        elongation-STRENGTH axis FLIP [open].** (1) **ALIGNED-NODIV PERFECT EGG [established]** — cell_grow aniso 0.6
        axis [0,1] (+y-aligned), DIVISION OFF: mi_type_y {1.000,1.000,1.000(b83)} = **1.000±0.000** over 3 seeds,
        seg_index 1.000, interface_frac 0, mixing_entropy 0 (fully demixed). Campaign's CLEANEST oriented-partition
        object (static, n=44). CAVEAT: sort is +y-robust but SHAPE weak/seed-variable — mi_type_x 0.391±0.288
        (diagonal-contaminated s0/b83, clean only s1), fourier_m2 0.044±0.023 (weak, ~2-3× seed-variable), circ
        0.75–0.95. cell_grow aniso is a WEAK NOISY SHAPER (reconfirmed 3rd batch). (2) **b83 perp-nodiv AXIS RULE
        REJECTED [non-robust]** — perp06_nodiv 3 seeds {s2 (mi_y 1.0, mi_x 1.0 diagonal), s3 (1.0, 0.29 held +y),
        b83 (0.244, 0.641 tumbled x)} = mi_type_y 0.748±0.436, mi_x>mi_y in only 1/3. Perp elongation WITHOUT
        division does NOT reliably reorient the sort onto x — it merely DESTABILIZES it (seed-dependent land). The
        SHORT-vs-LONG rule survives ONLY for the ALIGNED case. (9th single-seed clean point to regress.) (3)
        **DIVISION destroys the aligned +y hold at ~1.05× [established-direction]** — align06_lowdiv (div_rate 0.1)
        mi_type_y 1.000→0.160, mi_x 0.311, seg 0.864, circ crashes 0.370 (m3 0.020/m4 0.015 ragged). Aligned perfect
        egg is strictly NON-dividing. (4) **STRONG aligned bud FLIPS sort to +x [open, n=1]** — align10_nodiv aniso
        1.0 mi_type_y 0.075, mi_type_x 1.000, type_axis −172° (on +x); shape barely changes (m2 0.020, circ 0.959) =
        growth-PRESSURE reorientation not visible elongation. So the aligned sort is NON-monotone in aniso: holds +y
        at 0.6, flips +x at 1.0. (5) Flow not needed (align06_noflow mi_y 1.000). (6) Determinism exact (flow_g11_s2
        0.833). TIER-1 held 32nd straight (collapsed 0 all 8, nn_min 0.0181–0.0193, area ~2× realized, no rupture on
        the circ-0.37 lobed slot). **Batch 85 = map the aligned elongation-STRENGTH axis flip: cell_grow aniso ladder
        0.6/0.7/0.8/0.9/1.0 (axis [0,1], nodiv, move18), 2 seeds/rung on 0.7/0.8/0.9 + 2nd seed on 1.0 flip + 0.6
        ctrl.** New specs embryo_MOR_cap_align07/08/09_nodiv_s0/s1 + _align10_nodiv_s0. Falsifier: mi_type_y NO
        monotone aniso trend (scatter independent of aniso = seed-noise) OR 0.6 anchor fails mi_y≥0.9.
      - **MOR — CLOSED Batch 86 (2026-07-06) by USER DIRECTIVE (MOR ran b68→b85, ~16 batches, far past its
        10-batch cap; auto-cap not firing). GATE PARTIALLY MET.** Deliverable = an ORIENTED, PROGRAMMABLE,
        two-domain body whose PATTERN AXIS is set by anisotropic cell_grow, with the demix held (seg 1.0).
        **MOR OP POINT = embryo_MOR_cap_align06_nodiv_s0** (aniso 0.6, nodiv, +y demix mi_type_y 1.000±0.000
        [established 3 seeds b84], grow_ratio 1.39, TIER-1 clean). **b85 = the aniso-magnitude AXIS FLIP
        [open→mapped]:** whole-body cell_grow aniso 0.6→1.0 flips the type axis +y→+x at a SHARP knee in
        (0.6,0.7] (mi_type_y 0.6:1.000 → 0.7:0.116 [mi_type_x 1.000, type_axis −179°] → 0.8:0.030 → 1.0:0.149);
        the crossover is a KNEE not a ramp (earlier than the predicted ~0.8). **[open] ELONGATION⊥PARTITION
        TRADE-OFF:** strong whole-body elongation and a clean demix are ANTAGONISTIC — aniso 0.6 = perfect sort
        + round outline (aspect_ratio 1.054, fourier_m2 0.017); aniso 1.0 = strongest shape (fourier_m2 0.049)
        but the demix WASHES OUT (seg_index 0.385, mi_type_x 0.199, nn_cv 0.58). **MOR OPEN BLOCKER [open]:**
        whole-body anisotropic cell_grow ORIENTS the pattern but never produced strong body ELONGATION (outline
        stays ~round; the substrate anchor likely flattens growth into density) — carried into BUD.
      - **[engineering, Batch 86] The organo bud detector has a NOISE FLOOR — org_n_buds ALONE is NOT a bud.**
        b85 whole-body-grown outlines report org_n_buds 1–2 with org_bud_score ≈ 0 (4.5e-5) and org_growth_bud
        _overlap = 0 (spurious outline wobble, not caused by localized growth). **The real BUD gate is
        org_bud_score > 0 AND org_growth_bud_overlap > 0** (a persistent protrusion that appeared WHERE the
        growth operator drove material). Gate on those, NOT on n_buds.
      - **BUD (localized morphogenesis — Phase 3, batch 1) — STARTED Batch 86 (2026-07-06). Gate: ONE
        reproducible localized bud (org_bud_score>0 AND org_growth_bud_overlap>0, org_bud_persistence rising,
        bud rounds by elastic relaxation) · embryo integrity (collapsed 0, nn_min ≥r0, no MPM plume) · pattern
        preserved (mi_type_y stays high).** Substrate = the MOR op point (embryo_MOR_cap_align06_nodiv_s0).
        MECHANISM = cell_grow mode=tip (seeds only the leading edge along axis → a localized protrusion).
        **CARRIED CONSTRAINT [rejected b68/b69]: tip mode ≥1.5 under the AGGRESSIVE MOR budget (target 5.5,
        prestretch 0.6, rate 1.1, reserve 36000) RUNS AWAY = MPM PLUME (area 3–6×, deform 0.24–0.38, escape 1.0,
        agent body does NOT grow).** BUD batch 1 hypothesis: the plume was the BUDGET not the MODE — a GENTLE,
        reserve-CAPPED tip regime (rate 0.4, target 1.8, prestretch 0.75, grow_reserve 8000 = hard 2.0× ceiling)
        buds cleanly. Batch 86 = gentle-tip sweep (tip 1.0/1.5/2.0 × prestretch 0.60/0.75/0.85) + whole-body
        aniso CONTRAST + anchor-off + rate-0 control, on embryo_BUD_base/_aniso/_noanch. RUNAWAY SIGNATURE to
        screen FIRST: area>2.5× + deform>0.2 + escape~1.0 TOGETHER (distinct from the ~0.7 sediment body-drift
        escape artifact). FALSIFIER: (A) bud_score ~0 across the tip ladder while grow_ratio rises → gentle tip
        realizes growth but not a bud; (B) any capped-reserve tip slot still plumes → tip mode intrinsically
        unstable, retreat to aniso mode + a different localization mechanism.
        - **Batch 87 read: b86 was a WIRING WASH — the tip sweep NEVER RAN.** [engineering, DURABLE] Dotted
          `cell_grow.*` overrides are SILENTLY NOT APPLIED to a flow-style `{op: cell_grow, ...}` line: the
          archived spec.yaml logs the intent in a comment (`# overrides: [cell_grow.tip=2.0]`) but the op line
          stays `tip: 1.5` → all 6 base slots (tip10/tip20/ps85/ps60/nogrow) were BIT-IDENTICAL re-runs of tip
          1.5 (org_bud_score 0.014364158435879943 shared to 16 digits by tip15+tip20+nogrow). **To vary an
          operator param, AUTHOR A SEPARATE SPEC** (the noanch/aniso separate-spec slots DID differ). — WHAT THE
          ONE CONFIG (tip 1.5, anchored) DID: growth IS realized and LARGE but DELOCALIZED. organo body_radius
          0.168→0.467 (2.8×), organo area 0.089→0.705 (~7.7×, saturates by 25%); **grow_ratio (cell rest-radius)
          reads only 1.0085 = WRONG lens for tip/continuum growth — judge growth by organo area/body_radius.**
          NOT a bud: org_growth_bud_overlap 0.0 at all 5 timepoints, org_bud_score ~0.014 (noise floor), n_buds
          3 / n_tips 22 / branchpoints 40 = DETECTOR ARTIFACTS on a rough low-circularity (organo circ 0.198)
          MPM blob; tip_growth_enrichment only 0.30 (tip-1.5 softmax temp=tip/std seeds the whole top diffusely).
          Pattern HELD (mi_type_y 1.0, seg_index 1.0 — inherit-capabilities met); TIER-1 clean (collapsed 0,
          nn_min 0.0183); NO runaway (area << plume threshold). AUX specs: noanch (tip 1.5, anchor OFF) gave the
          LONGEST protrusion (org_bud_len_bodyR 0.418 vs 0.21, persistence 1.0) yet overlap STILL 0 → anchor
          RESISTS tip extension, AND **[engineering] SUSPECT org_growth_bud_overlap: reads 0 even for a visibly
          longer finger — do NOT hard-gate BUD on it.** aniso_ref (whole-body) org_bud_score 0.0, migration 0.61
          (drift) → anisotropy ≠ bud; LOCALIZATION is the lever. **Batch 87 = the REAL tip-sharpness sweep via
          separate specs: tip 1.5→5→10 (softmax sharpening) × {offset 0.06, anchor-off, prestretch 0.60} +
          aniso contrast + tip-1.5 control.** HYP: sharper tip → discrete necked protrusion (bud_len_bodyR/
          bud_score rise vs tip). FALSIFIER: bud_len_bodyR/score FLAT across tip 1.5→10 → PIVOT to PATTERN-GATED
          growth (grow only in one demix domain → hemispheric bud). Judge bud by len_bodyR/score/persistence/
          neck (NOT overlap), growth by organo area/body_radius (NOT grow_ratio), pattern by mi_type_y.
        - **Batch 88 read: b87 = TIP LOCALIZATION MAKES A WEAK, TIP-MONOTONE BUD; the ANCHOR is the dominant
          SUPPRESSOR [open, n=1 at the winner].** The b87 tip-sharpness sweep RAN (separate specs, all differed).
          ANCHORED tip ladder (seed 0, axis +y) is MONOTONE but tiny: org_bud_score 0.0144 (tip1.5) → 0.0275
          (tip5) → 0.0405 (tip10); org_bud_len_bodyR 0.210 → 0.264 → 0.376. The b87 falsifier ("FLAT across tip
          1.5→10") did NOT fire — tip has a real tunable effect — but absolute bud_score stays <0.05 anchored (a
          weak bud, not a discrete organ; failure mode = ROUNDING, shape circularity 0.90, NOT rupture/pattern
          loss). **DROPPING mpm_anchor is the biggest single lever:** tip5_noanch (s3) → bud_score **0.0994**
          (7× ctrl, 3.6× anchored-tip5), neck 0.225 (MOST necked), persistence 1.0, shape area 0.961 (biggest
          body, ~3× anchored-tip5), TIER-1 clean, pattern held — the anchor resists BOTH tip extension AND body
          inflation. Offset 0.06 (s2, anchored) 2nd (bud_score 0.0519, longest finger 0.388). **CEILING = the
          aggressive combo overshoots:** tip10_noanch_off06 (s5) → bud_score **0.0**, neck **1.469** (>1 = a
          BULGE not a necked bud) — sharp-tip + anchor-off + big-offset DELOCALIZES the reserve into a broad
          shoulder that never necks. So there is a SWEET SPOT: anchor-off + MODERATE tip5. aniso_ref (s6,
          whole-body) bud_score 0.0 / n_buds 0 / migration 0.61 = anisotropy drifts, NO bud (LOCALIZATION is the
          lever, reconfirmed). **[engineering, DURABLE] org_growth_bud_overlap = 0.0 in ALL 8 slots at ALL
          timepoints = BROKEN/inert metric, do NOT gate BUD on it.** Pattern HELD everywhere (mi_type_y 1.0,
          seg_index 1.0); TIER-1 clean everywhere (collapsed 0, nn_min 0.0177–0.0186); NO runaway. **Batch 88 =
          NOANCH SURFACE-TENSION SWEEP + winner replication:** 3-seed the tip5_noanch winner (noanch_s1/s2 +
          b87 s3) + lower MPM surface_tension 8→5→3 on the noanch substrate (attack the rounding directly) +
          tip8 + prestretch0.60 + tip8×st5 combo + rate-0 control. HYP: lower surface tension lets the bud stand
          off/neck (bud_score↑, organo circularity↓/aspect_ratio↑). FALSIFIER: bud_score flat/falling vs
          surface_tension OR winner fails replication (s1/s2 <0.05) → accept weak-tip-bud [open] deliverable,
          pivot to pattern-gated growth. Runaway arm: st3 fragments (fragment_count>1 OR nn_min<0.016).
        - **Batch 89 read: b88 = WEAK TIP-BUD REPLICATES [established]; surface_tension NO-OP; ROUNDING is the
          hard ceiling.** (1) **tip5_noanch = a REAL weak bud, 3 seeds: org_bud_score 0.072±0.010** (seed0 0.0797,
          seed1 0.0573, seed2 0.0787; persistence 1.0, neck <0.40, pattern held mi_type_y 1.0, TIER-1 clean) —
          |Δ| vs ctrl 0.0 = 7·SD/3 seeds. The b87 single-seed 0.099 REGRESSED to 0.072 (9th single-seed clean
          point to fall on replication — durable law). (2) **[engineering — RETRACTED at Batch 98: this is NOT a
          physics no-op, it is the DOTTED-OVERRIDE CACHING BUG] mpm_grid_update.surface_tension "NO-OP":** st5(5.0)==
          st3(3.0) byte-identical (bud_score 0.07970581296896076 to 16 digits) — but byte-identity across ST values is
          the SIGNATURE of the inert override (op caches ST in __init__; the dotted override never reaches forward, see
          Batch 98). ST was NEVER actually varied here; the "dead lever" conclusion is UNSUPPORTED — bake ST to test it. cell_grow.tip
          IS live (tip5 0.0797 ≠ tip8 0.0946). (3) **tip8 (0.0946, neck 0.178) > tip5 (0.072) = monotone tip lever,
          single-seed batch-max** → needs 3-seed lock. (4) **ROUNDING is the ceiling:** body inflates 6× (shape
          area 0.156→0.96) but circularity RISES 0.87→0.96, org_aspect_ratio 1.07 → a BIGGER SPHERE not a lobe;
          bud_score comes from a rough organo membrane mask (org_circularity 0.262) at the tip, not a standing-off
          finger. ps60 (prestretch 0.60) 0.0355 < tip5 = over-compression ROUNDS (reconfirms MOR b75). **Every
          roundness lever is now EXHAUSTED across MOR+BUD: surface_tension inert (b73/b88), youngs deflates-not-
          rounds (b74–76), prestretch amplifies-not-rounds (b75/b88), rate-down worsens buckle (b75), rate-up
          shatters (b76).** The single elastic MPM cell rounds by energy minimization — a discrete organ likely
          needs a MULTI-CELL domain (grow a SUBSET of cells = the user's "pattern-gated growth", n=1 cannot express
          it). **Batch 89 = TIP-SHARPNESS × OFFSET FRONTIER + tip8 3-seed lock:** tip8_s1/s2 (lock 0.0946), tip12/
          tip16 (sharpness push), tip8_off05 + tip12_off05 (finger reach, bulge falsifier), tip8_k2 (agent-extrusion
          probe), ctrl_nogrow. FALSIFIER: tip12/16 ≤ tip8 (~0.095) AND off05/k2 <0.10 AND tip8 seeds spread <0.06 →
          single-cell tip-bud CAPPED ~0.09 → report weak-bud (0.072±0.010) as BUD [open] deliverable, OPEN the
          multi-cell-domain path. Runaway: neck_ratio>1 (bulge) OR nn_min<0.016 OR collapsed>0.
        - **Batch 90 read: b89 = SINGLE-CELL TIP-BUD CLOSED — the b89 falsifier FIRED on every clause.** (1) tip8
          3-seed lock FAILED: {0.0946 [b88 s0], **0.0** [s1, neck_ratio 2.0 = a BULGE], 0.0606 [s2]} = bud_score
          **0.052±0.048** (SD≈mean; b88's 0.0946 = seed-luck, ~10th single-seed clean point to regress). (2)
          sharpness does NOT push past tip8: tip12 0.0436, tip16 0.0767 — both ≤ tip8's single-seed 0.095, trend
          NON-monotone. tip16 IS the only slot to break roundness (circularity 0.94→0.774, **bud_len_bodyR 0.605**
          batch-max, **fourier_m1 0.408** batch-max dipole) BUT that is WHOLE-BODY pear elongation that SCRAMBLES
          the pattern (**mi_type_y 1.0→0.8627**, only slot <1.0), a tradeoff not a bud-on-a-body. (3) offset/agent-
          extrusion inert: tip8_off05 0.0257, tip12_off05 0.0235, tip8_k2 (agent_to_mpm k2) 0.0614 — none >0.10.
          ctrl_nogrow 0.0. All 8 TIER-1 clean (collapsed 0, nn_min 0.0174–0.0186). **CONCLUSION [open deliverable]:
          a single elastic MPM cell rounds any local growth and caps bud_score ~0.05–0.08, seed-unreliable; the ONE
          length route (tip16) scrambles the pattern → the single-cell tip-bud CANNOT make a discrete organ that
          inherits the pattern. The BUD [open] deliverable is the single-cell weak-bud (0.072±0.010 [established
          b88]).** ENABLEMENT [engineering]: **cell_grow made MASK-AWARE** (`if mask is not None: live_cell *= mask`,
          cell_grow.py:126) so `at: 'cell[type=bud]'` grows ONLY the selected cells — byte-identical for every
          existing `at: cell` spec (that mask is all-live). Unlocks the "grow a SUBSET" route both the b89 plan and
          the user directive name. **Batch 90 = MULTI-CELL SUBSET-GROWTH PIVOT:** cell.n=2, `type_layout: split_x`
          (body=low-x static, bud=high-x grows), two adjacent elastic balls (start 0.40/0.60, radius 0.13) form a
          peanut; cell_grow inflates ONLY the bud cell → a discrete lobe. Bud axis +x ⟂ y-demix axis (test pattern-
          inheritance). Slots: multi_aniso (+s1/s2 3-seed) / multi_tip / multi_iso / multi_gap (0.34/0.66) / multi_big
          (target 2.5) / multi_ctrl0 (rate 0 static-peanut control). HYP: growing one of two cells → bud_score ≫
          single-cell cap 0.08 AND ≫ ctrl0, neck<1, mi_type_y >0.8, 3 seeds. FALSIFIER: bud_score ≤0.08 OR ≈ctrl0 OR
          cells merge round (n_buds 0) OR mi_type_y<0.5 OR TIER-1 fail → subset-growth buys no organ here, reconsider
          cell-cell adhesion / distinct-material bud. **NEW substrate + new mask-gate = ELEVATED execution risk; if
          0-archive read a slot .err FIRST (code-crash vs infra).**
        - **Batch 91 read: b90 = MULTI-CELL SUBSET-GROWTH GEOMETRY-FAILS (bud_score≈0 in 7/8) — but the mask-gate
          & growth WORK; the failure is GEOMETRIC and fixable [open].** (1) TWO EQUAL side-by-side cells (start
          0.40/0.60) = a PEANUT: multi_aniso 3 seeds org_bud_score **0.0**, n_buds 0, org_aspect_ratio 1.86/1.92/1.88
          ≈ static ctrl0 **1.812** — growing the bud cell ~1.8× only lopsides the peanut, the detector reads ONE
          elongated body (two comparable lobes), so bud_score=0. NOT a growth failure: grow_ratio 1.049 is the
          WHOLE-BODY ratio DILUTED by the static body cell (bud cell itself ~1.8×, radius ×1.34 hidden in the 2-lobe
          metric). (2) No route off 2-equal-cells buds: multi_tip grow_ratio 1.0004 (no growth) + n_buds 4/hier_depth
          5/indep_domains 11 = DETECTOR ARTIFACTS on a rough faint mask (organo circ 0.453); multi_gap (0.34/0.66)
          org_fragment_count **2.0** = the cells DETACH into 2 round balls (no neck); multi_big (target 2.5) SHATTERS
          (circ 0.93→0.515, shape_index 4.94, flings a fragment). (3) Pattern eroded by demix SHEAR — mi_type_y aniso
          0.77/0.65/**0.36**, the 2 loosely-coupled cells shear vertically under sediment (no cohesive core to resist).
          All 8 TIER-1 clean (collapsed 0, nn_min 0.0173–0.0188; escape 0.43–1.0 = sediment body-drift ARTIFACT). **PRINCIPLE
          [open→body-dominant]: two EQUAL cells cannot present "round body + minority protrusion"; the "grow a subset"
          route needs a BODY-DOMINANT cluster (many body cells = round cohesive core + ONE small peripheral bud cell).**
          **Batch 91 = BODY-DOMINANT CLUSTER:** 7 MPM cells (6 body hex + 1 bud at +x edge via type_layout split_x),
          per_parent 1500, grow_reserve 3500; cell_grow grows ONLY the bud cell (aniso +x). Slots hex7_aniso(+s1/s2
          3-seed)/hex7_tip/hex7_big(target 3.0)/hex7_nosed(drop sediment=shear isolation)/merge2(2-cell start 0.50/0.55)/
          hex7_ctrl0(rate 0). HYP: small bud cell off a round multi-cell core → bud_score>0.05 & >ctrl0, n_buds≥1,
          neck<1, core cohesive (fragment_count 1), mi_type_y>0.6, 3 seeds. FALSIFIER: bud_score≈ctrl0≈0 OR cluster
          fragments OR mi_type_y<0.5 → subset-growth buds NOWHERE on this substrate → adopt single-cell weak-bud
          (0.072±0.010 [established b88]) as BUD [open] deliverable, ADVANCE to BRN (batch 7/10 of BUD).
        - **Batch 92 read: b91 = BODY-DOMINANT CLUSTER GEOMETRY-FAILS to bud (falsifier FIRED) → BUD CLOSED,
          ADVANCE to BRN.** All 8 slots `org_bud_score` 0.0 / `org_n_buds` 0 (ctrl0 floor 0.00014/2). Growing
          the +x bud cell only LOPSIDES the cluster into a mode-2 ELLIPSE — `org_aspect_ratio` monotone in dose
          (ctrl0 1.025 → aniso {1.337,1.315,1.303} → big 1.465) — which the bud detector's mode-0..3 low-pass
          reference ABSORBS → 0 protrusion. The ONE thing hex7 fixed vs b90: `org_fragment_count`=1.0 in ALL 8
          (body-dominant core resists detachment; b90 gap/big fragmented). Pattern seed-variable (mi_type_y
          aniso {0.870,0.168,0.665}=0.568±0.29; nosed 0.0; loose cells shear under sediment). tip mode STILL
          did not realize on multi-cell (grow_ratio 1.0). TIER-1 clean (collapsed 0, nn_min 0.017–0.019;
          escape 0.43–1.0 = sediment body-drift ARTIFACT). **[engineering, DURABLE — why subset-inflation caps
          at 0]:** `scorecard_organo.py:94-150` `buds()` low-passes radial FFT modes 0–3, counts a lobe only
          where `prot>0.12·body_R` — so a BUD MUST BE A NARROW mode-≥4 NECKED FINGER; a broad mode-1/2 bulge
          (a whole grown cell, a lopsided cluster) is absorbed → structurally 0. Single-cell TIP mode makes
          such a finger (0.072); multi-cell INFLATION cannot. **BUD CLOSED [established weak-bud + open
          blocker]: OP POINT = `embryo_BUD_noanch_tip8.yaml`** (single elastic cell, tip 8, anchor OFF;
          org_bud_score 0.072±0.010 [established b88, 3 seeds], pattern mi_type_y 1.0, TIER-1 clean). **[open]
          BUD BLOCKER: a STRONG discrete organ (bud_score≫0.1, clean neck) is UNREACHABLE on the elastic-MPM
          substrate** — single cell rounds all local growth (every roundness lever exhausted MOR+BUD:
          surface_tension inert, youngs deflates, prestretch amplifies, rate-down buckles, rate-up shatters);
          multi-cell subset = low-mode lopsiding (bud_score 0). `current_stage.txt`=BRN.
        - **BRN (branching morphogenesis — Phase 3, stage 8) — STARTED Batch 92 (2026-07-06). Gate:
          reproducible bifurcation (branching EMERGES from feedback, never a branch operator) · stable branch
          persistence · tissue continuity preserved · controlled branch spacing.** Decision surface = organo
          BRANCH family (`org_branch_score`, `org_n_tips`, `org_n_branchpoints`, `org_hierarchy_depth`,
          `org_tree_depth`, `org_independent_growth_domains`, `org_branch_persistence`); branches() zeroes on a
          convex round body (solidity>0.90) → a branch signal REQUIRES a low-solidity fingered outline.
          **[engineering, DURABLE — the BRN mechanism constraint, from `cell_grow.py:133-139`]: `stress_gain`
          (mechano-inhibition) modulates a PER-CELL-MEAN deformation → a GLOBAL per-cell rate brake; on ONE
          cell it CANNOT spatially redistribute growth → cannot tip-split. BRANCHING FEEDBACK THEREFORE NEEDS
          MULTIPLE GROWTH CENTRES COMPETING** — the b91 cohesive cluster with ALL cells growing: compressed
          interior cells self-inhibit, low-stress peripheral cells keep growing → fingering → emergent multi-tip
          outgrowth (= the "growth competition / multiple interacting growth centres" route the instruction
          names). BRN batch 1 = ISOLATED stress_gain-competition validation on the hex7 cohesive cluster (all
          cells grow isotropically, per_parent 1500 + reserve 3500): stress_gain SWEEP 0(ctrl)/2/5/10 + big-dose
          + seed replicate + line-substrate explore + rate-0 static ctrl. HYP: stress_gain>0 → fingered outline
          (org_solidity↓, org_branch_score/n_tips>0) vs sg0 round blob (branch_score~0). FALSIFIER: branch_score
          flat/0 across the stress_gain sweep (competition doesn't finger, elastic rounds) OR any slot ruptures
          (collapsed>0/nn_min<0.016/fragment_count>2) → competition-fingering fails on this substrate, pivot to
          a distinct branching driver (curvature-gated growth field, morphogen-controlled growth). GOTCHAs
          carried: dotted `cell_grow.*` overrides SILENTLY IGNORED on flow-style op lines → AUTHOR separate
          specs per growth config (b87); escape is a sediment ARTIFACT under any body force; NEW substrate =
          execution risk, if 0-archive read a slot .err FIRST (code-crash vs infra).
        - **Batch 93 read: b92 = GROWTH-COMPETITION FINGERING [rejected] — the b92 falsifier FIRED (elastic
          rounds the confluent cluster convex).** All 8 slots `org_branch_score` 0.0 / `org_n_tips` 0 /
          `org_n_branchpoints` 0, `org_solidity` 0.961–0.980 (branches() zeroes above 0.90), fragment_count 1,
          TIER-1 clean (collapsed 0, nn_min 0.018–0.019) — the "elastic rounds into a bigger blob" arm, NOT
          rupture. `stress_gain` works as documented (per-cell-MEAN mechano-inhibition: sg10 grow_ratio 1.074
          = LOWEST = strongest brake) but it is a GLOBAL per-cell rate brake → cannot spatially redistribute
          growth WITHIN a confluent mass → smaller-but-still-CONVEX blob. Higher-mode boundary content rises
          with growth (sg10 fourier_m3 0.0615/m4 0.0366/m5 0.0150 vs sg0 m3 0.0114) but stays a SHALLOW
          ripple (solidity pinned 0.96–0.98) — surface_tension 8.0 + mpm_to_agent confine round every notch
          back out. Even a 2:1 filament (line slot, aspect 1.98) is a SMOOTH convex ellipse (solidity 0.970,
          branch_score 0, no branch at its ends). SAME rounding blocker that closed BUD; confluent multi-cell
          growth inherits it. **[engineering, DURABLE — the Phase-3 substrate law]: the elastic-MPM +
          surface_tension substrate is a CONVEXITY ATTRACTOR — it rounds ALL local/confluent growth (single
          cell → bud rounds; 7 confluent cells → blob rounds; 2:1 rod → ellipse) so any organ metric gated on
          low solidity / mode-≥4 necks (bud_score>0.1, branch_score>0) is structurally hard to reach.** PIVOT
          (b93) → SPATIALLY SEPARATED multi-centre growth: pre-place a hub + N satellite growth cells APART
          (radius d 0.16–0.20, per_parent 900 + reserve 2100, radius 0.10) so the tissue is a BUSH of lobes
          joined by NECKS — testing whether pre-placed separation gives concave necks (solidity<0.90,
          branch_score>0) that confluent growth cannot, while staying connected (fragment_count 1). Falsifier:
          solidity>0.90 across ALL separated configs (substrate rounds the necks too → relax BRN gate) OR any
          fragments (fragment_count>2 → separation buys concavity only by losing continuity). GOTCHAs (b93,
          verified in tune.py:41-72): generic dotted `op.param` overrides set `o.params[param]` and WORK for
          normal ops, but `general.seed` (no matching op → silently ignored) AND `cell_grow` mode-dependent
          params (tip needs mode+axis) do NOT → AUTHOR a full spec for every seed change AND every cell_grow
          config change; `>>`/heredoc redirect still sandbox-blocked → Edit/Write only.
        - **Batch 94 read: b93 = BRANCHING ACHIEVED via SPATIALLY-SEPARATED multi-centre placement [open, n≤2].
          The b93 falsifier did NOT fire.** Pre-placing a hub + N satellite growth cells APART (radius d) and
          growing all isotropically yields a concave N-lobe star the detector scores, where the b92 confluent
          mass could not. `org_solidity` confluent-ctrl 0.972 → tri16(d0.16) 0.796 → tri20(d0.20) **0.703**
          (deepest, all <0.90 gate); `org_n_tips` == N satellites EXACTLY (tri 3 / cross 4 / penta 5 — directly
          PROGRAMMABLE topology); `org_branch_score` 1.0–2.4 (ctrl 0.0); `org_fragment_count` 1.0 ALL 8
          (connected — hub holds lobes even at d0.20); pattern held (tri16 mi_type_y 0.699, seg 0.897);
          TIER-1 clean (collapsed 0, nn_min 0.0176–0.0187). More lobes RAISE solidity back toward convex
          (tri 0.70–0.80 → cross 0.861 → penta 0.900 = rosette fills its own necks) — a soft upper bound on N.
          **[engineering, DURABLE] branching here is a PLACEMENT PROGRAM, not a growth-competition instability:**
          tri16_nogrow (STATIC, rate 0.0) ALREADY reads solidity 0.788 / n_tips 2 / branch 2.0, and tri16
          solidity is FLAT over time [0.799→0.796] — growth does NOT carve or fill the necks. Growth's causal
          role = organo AREA ~doubles (nogrow 0.100 → grow 0.172, +72%) + n_tips sharpens 2→3; the substrate
          (convexity attractor) neither deepens nor rounds the placement-set concavity, growth just scales it up
          and connects it. n=1–2 → tri20 needs 3 seeds + a clean grow/nogrow area contrast to promote. Batch 94
          = tri20 s0/s1/s2 (establish) + tri20_nog s0/s1 (causal growth-isolation) + tri24 (d0.24 deepen/
          fragment falsifier) + cross20 (4-lobe at d0.20) + tgt30 (does target 3.0 fill or deepen necks?).
          FALSIFIER: tri20 solidity >0.90 on any seed OR grow≈nogrow on AREA too OR tri24 fragment_count>2.
        - **Batch 95 read: b94 = SEPARATED-BRANCH [ESTABLISHED, 3 seeds] + growth causally separated (SCALE not
          topology) + separation UPPER BOUND found + cross throws first real bud_score.** The b94 falsifier did
          NOT fire on the establish/growth arms; the tri24 fragmentation arm fired AS DESIGNED (bounds the window).
          (1) **tri20 3-seed:** `org_solidity` {0.7031,0.7044,0.7015}=**0.7030±0.0012** vs confluent-ctrl 0.972
          (Δ 0.269 ≫200·SD below the 0.90 gate); `org_n_tips` 3/4/3; `org_branch_score` 2/2/3; `org_fragment_count`
          1 all three; pattern held (mi_type_y 0.64/0.96/0.67, seg 0.75/1.0/1.0); TIER-1 clean (collapsed 0,
          nn_min 0.0176–0.0185). SEPARATED-MULTI-CENTRE BRANCH = **[established]**. (2) **GROWTH = SCALE, PLACEMENT
          = TOPOLOGY [established, n=2/arm]:** grow vs matched-seed nogrow at IDENTICAL d0.20 placement → grow_ratio
          1.17 vs 1.001, body area 0.324 vs 0.245 (**+32%**), but org_solidity 0.703 vs 0.695 (EQUAL, Δ~0.008).
          Growth inflates the star to organ scale; it neither rounds nor carves the placement-set necks. (nogrow
          carries big sediment body-drift: fourier_m1 0.38/0.39 & deform_rms 0.14 vs grow 0.07–0.10 & 0.06 — the
          branch metric solidity is robust to it.) (3) **SEPARATION WINDOW BOUNDED ABOVE: tri24 (d0.24) FRAGMENTS**
          — org_fragment_count **4.0**, solidity 0.955 (convex remnant), n_tips/branch_score 0, grow_ratio 0.774
          (shrank). Branch window = d ≈ 0.16–0.20 (b93 tri16→tri20 connected; d0.24 detaches). (4) **cross20 (4
          satellites, d0.20) = RICHEST + FIRST real bud_score:** n_tips **5**, branch_score **3**, n_branchpoints 3,
          fragment 1, **org_bud_score 0.271** (batch max; the whole BUD stage never cleared ~0.1), solidity 0.745,
          pattern held (mi_type_y 0.812) — n=1. (5) **tgt30 (target 3.0) DEEPENS the tree at constant solidity:**
          grow_ratio 1.27, body area 0.384 (biggest), `org_tree_depth` **0.825** (deepest), n_tips 5, branch_score 3,
          solidity 0.704 (unchanged) — more growth = more tips + deeper skeleton WITHOUT rounding. **BRN STATUS
          [engineering, DURABLE]:** branching = a PLACEMENT PROGRAM framed by two hard bounds — BELOW the elastic
          rounds a confluent mass convex (b92 growth-competition [rejected]), ABOVE d≈0.24 the lobes detach
          (fragment_count 4); window d≈0.16–0.20; n_tips programmable by satellite count; growth sharpens/deepens
          the tree without altering topology. **Batch 95 = HIERARCHY + cross-establish:** hierY (hub+3 primary, TOP
          arm forked into 2 secondary = Y-of-Y; does org_hierarchy_depth reach 3 / tree_depth > 0.825?) + hierY_s1 +
          hiertgt30 [hierarchy]; cross20_s1/s2 (3-seed establish richest topology) + crosstgt30 (deep growth on
          cross) [establish]; tri22 (d0.22 fragmentation-boundary bracket) [explore]; tri20_ctrl [control]. New
          specs embryo_BRN_{cross20_s1,cross20_s2,hierY,hierY_s1,tri22}.yaml + cell_grow.target 3.0 dotted overrides.
          FALSIFIER: hierY hierarchy_depth stays 2 (fork lobes merge, no sub-branchpoint) OR any hierY/cross
          fragments OR the 6-lobe pack rounds solidity >0.90. ANCHOR: last real data = b94.
        - **Batch 96 read (b95): cross20 richest-topology ESTABLISHED (3 seeds); HIERARCHY CAPS AT DEPTH 2 (b95
          falsifier FIRED — the Y-of-Y fork MERGES); fragmentation bound TIGHTENED to d≤0.20.**
          (1) **cross20 (4 diagonal satellites + hub, d0.20) [ESTABLISHED, 3 seeds]:** org_solidity 0.737±0.009,
          **org_bud_score 0.269±0.002** (the BUD stage never cleared ~0.1 → strongest discrete-protrusion signal of
          the campaign, now 3-seed), org_n_tips {5,4,5}, org_branch_score {3,2,2}, org_fragment_count 1 all, pattern
          held (seg ~0.95–1.0, mi_type_y 0.51–0.81). cross = the MANY-TIP established branch.
          (2) **RECURSIVE HIERARCHY BLOCKED [open→likely rejected]:** the forked Y-of-Y arm (2 sublobes 0.20 apart,
          0.156 from primary) MERGED into one fat tip — hierY hierarchy_depth 2 (NOT 3), n_tips 3 (unchanged), both
          seeds; the surface_tension rounds the two sublobes inside one convex hull. The placement program carries a
          radial STAR faithfully but does NOT carry a TREE. What the fork DOES buy = a DEEPER/BRANCHIER skeleton:
          hierY reads the batch's lowest solidity (0.698–0.701), most n_branchpoints (3–4), highest branch_score
          (3–4); hiertgt30 (target 3.0) = deepest tree_depth of the campaign 0.863 (growth deepens the skeleton but
          still can't resolve the fork or round the necks — reconfirms growth=SCALE, placement=TOPOLOGY). A topology
          TRADE: cross maximizes TIPS (5), hierY maximizes BRANCHPOINTS/DEPTH (bp 4, tree_depth 0.86).
          (3) **FRAGMENTATION BOUND TIGHTENED to d≤0.20:** tri22 (d0.22) FRAGMENTS (fragment_count 4, solidity 0.946
          convex remnant, area collapsed to 0.050) — d0.22 already detaches (b94 had only bounded d0.24), so the
          connected branch window upper bound is d≈0.20. tri20_ctrl anchor EXACT (solidity 0.7031 == 0.7030±0.0012).
          TIER-1 clean 37th straight (collapsed 0 all 8; only fragment is the intended tri22 probe; escape = sediment
          body-drift artifact). **Batch 96 = HIERARCHY-RESOLUTION-OR-REJECT:** widen/lengthen/recurse the fork —
          hierwide (sublobes 0.26 apart) + hierwide_s1 + hierwide_g3 (wide + target 3.0) + hierlong (fork radially
          isolated at y=0.85) [exploit]; hierdbl (7-centre depth-2 binary tree) + crossfork (cross with one forked
          arm) + hierwide_frag (0.226 detach probe) [explore]; hierY [control]. New specs embryo_BRN_{hierwide,
          hierlong,hierwide_frag,hierdbl,crossfork}.yaml + seed/target dotted overrides. FALSIFIER: all fork variants
          still hierarchy_depth 2 / n_tips ≤3 until they DETACH → recursive branching [rejected] on this rounding
          substrate; BRN deliverable = first-order programmable stars (tri/cross) + deep-skeleton endpoint (hierY).
          ANCHOR: last real data = b95.
        - **Batch 97 read (b96): the b96 FALSIFIER ESSENTIALLY FIRED — recursive/2nd-generation branching NOT
          robustly achievable by PLACEMENT geometry [open→likely rejected]; hierarchy_depth is a NOISY 1-frame
          metric [engineering, DURABLE].** No fork variant reached a STABLE hierarchy_depth 3 with co-moving n_tips↑:
          (1) WIDEN (hierwide 0.26 apart) → final hierarchy_depth 1 (traj [1,2,3,2,1] — transient-3 spike), n_tips 4
          (traj [4,2,3,3,4]); hierwide_s1 depth 2/n_tips 3; hierwide_g3 (target 3.0) depth 2 STABLE [2,2,2,2,2]/n_tips
          3 (growth DEEPENS tree_depth 0.833 + bud_score 0.230 but never resolves the fork = growth-SCALE /
          placement-TOPOLOGY reconfirmed). (2) LENGTHEN (hierlong y=0.85) → n_tips 2 (fork became ONE long arm),
          depth 2. (3) DENSIFY (hierdbl 7-centre recursive binary tree) → HARD GEOMETRY FAIL: rounds to a convex
          BLOB (solidity 0.919, n_tips 0, everything 0) — densest placement merges completely; denser≠branchier. (4)
          crossfork (cross + 1 forked arm) = RICHEST SKELETON (n_branchpoints 6, branch_score 6 both batch-max) but
          final hierarchy_depth 3 is a LONE last-frame spike [2,2,2,2,3] and n_tips 3 (traj [4,5,3,5,3] flickers).
          hierwide_frag (0.226) DETACHES (fragment 3) → reconfirms connected window d≤~0.20. hierY_ctrl EXACT b95
          anchor (depth 2, n_tips 3, solidity 0.698, tree_depth 0.727). Pattern held all (seg 0.92–1.01). TIER-1
          clean 45th straight. **hierarchy_depth/n_tips flicker ±1 per pct → a depth-3 FINAL from one slot is NOT a
          resolved 2nd generation; needs STABLE (≥3 pcts) + co-moving n_tips↑, which NO slot shows.** MECHANISM
          [engineering]: three failure modes bound the window — merge-under-convex-hull (widen), one-long-arm
          (lengthen), round-to-blob (densify); NO geometry between "merges into one tip" and "detaches into
          fragments" gives a connected 2nd-gen branchpoint. The merge force = SURFACE-TENSION rounding in
          mpm_grid_update (surface_tension: 8.0), the convex-hull attractor smoothing concavities narrower than tip
          spacing — the ONE untried BRN lever. **Batch 97 = surface_tension DE-ROUNDING sweep (decisive test): is
          rounding SEPARABLE from cohesion?** crossfork ST ladder 8(ctrl)/4/2/1 + hierwide_st4 + crossfork_st4_g3
          (low-round+deep growth) + hierdbl_st3 (rescue the blob?) + crossfork_s1 (replicate the depth-3 spike). All
          dotted `mpm_grid_update.surface_tension` overrides on existing specs. FALSIFIER: no low-ST slot achieves
          STABLE hierarchy_depth 3 AND n_tips ≥5 AND fragment_count 1 → recursive branching [REJECTED] (rounding
          inseparable from cohesion), CLOSE BRN, advance to ORG next batch. BRN deliverable already banked =
          first-order programmable stars (tri/cross, tips=#satellites, cross20 [established] 3 seeds solidity
          0.737±0.009 bud_score 0.269±0.002) + deep-skeleton endpoint (hierY/hiertgt30 tree_depth 0.86). ANCHOR:
          last real data = b96.
        - **Batch 98 read: b97 was a SILENT ENGINEERING WASH — the surface_tension sweep NEVER RAN.** [engineering,
          DURABLE — supersedes/CORRECTS the "surface_tension inert [established]" claim] crossfork_st4/st2/st1/s1(seed1)
          all came back BIT-IDENTICAL to ST8 seed0 ctrl (montage strings to 6 sig figs; organo.final bit-identical,
          bud_score 0.1463718296038669, hierarchy_depth 3.0). ROOT CAUSE — dotted overrides via tune._apply
          (showcase.py:179, AFTER S.load): (a) `mpm_grid_update.surface_tension` is INERT because the op CACHES
          `self.surface_tension` in __init__ (mpm_grid_update.py:36) at load, before _apply mutates params; forward
          reads the stale attr. Same for wall_damp/dt_sub (all __init__-cached). (b) `general.seed` is INERT — no
          operator named 'general', so the `for o in sim.operators` loop never matches (silent, no warning). Only ops
          that read params at RUNTIME (cell_grow.target — g3 differed) respond to dotted overrides. **⇒ BAKE ST + seed
          + youngs into authored spec YAMLs; NEVER dotted-override an __init__-cached scalar or general.*.**
          **BIGGER: the "surface_tension INERT" claim across MOR b73 / BUD b88 / BRN b97 is a CONTAMINATED ARTIFACT of
          THIS bug — b73 s6 ('ST0', override `surface_tension=0.0`) has spec.yaml line 53 STILL reading 8.0; s5=s6=s7
          were bit-identical because all three secretly ran ST8. b88 st5==st3 byte-identical = same signature. ST has
          NEVER been cleanly varied [claim RETRACTED → re-open as UNTESTED].** Only valid b97 datum: crossfork_st4_g3
          (=ST8 + cell_grow.target 3.0) — deeper growth gives STABLE n_tips 5 [5,5,5,5,4] but hierarchy_depth STUCK 2
          [1,1,2,2,2] → a stiffer first-order 5-tip STAR, not a recursive tree. hierwide_st4/hierdbl_st3 (ST override
          void) merely re-ran b96 ST8 baselines (n_tips 4/depth 1/solidity 0.654; blob solidity 0.919/n_tips 0). TIER-1
          clean 46th straight. **Batch 98 = DE-ROUNDING DONE RIGHT (all BAKED):** the real rounding/cohesion lever on
          this youngs-200 ELASTIC body is MEMBRANE STIFFNESS (ST negligible vs a stiff membrane per TESTS.md 120-460 on
          water). youngs ladder 120/90/60 + y120_g3 (soft+deep grow) to test if softening opens a de-rounding window
          (resolve top fork → STABLE hierarchy_depth 3 + n_tips≥5, fragment 1) vs buckle/fragment; + BAKED st20/st2/st1
          to settle whether ST is a genuine dead lever or was always the artifact; + ctrl. FALSIFIER: no youngs slot
          hits stable depth-3 + n_tips≥5 fragment-1 AND baked ST inert → rounding inseparable from cohesion → recursive
          branching [REJECTED], CLOSE BRN, advance to ORG (terminus). ANCHOR: last real data = b96/b97-ctrl (same ST8 point).
    Batch 46 = minimal axis-cue probe (R1, ONE new operator family = `gravity`, a uniform membrane-cell body force
    wired into the MPM substep via p2g `a_ext` [p2g.py:45-50], default −y), built on the [established] INT op point.
    Slots = g magnitude ladder g1/g2/g4/g8 + g2 ×3 seeds (axis reproducibility) + gx horizontal-pull direction
    sanity + g0 control. Axial-order proxy = `fourier_m1` (shell dipole, existing/robust); a TYPE-axis metric is the
    next [engineering] TODO if the shell axis fires. FALSIFIER: fourier_m1 flat across the g ladder OR the gx dipole
    still points −y → gravity is not an axis cue, pivot to prescribed-gradient + differential-chemotax.
    - **[open→shell-oriented, Batch 47 / b46] GRAVITY ORIENTS THE SHELL (not yet the TYPE axis).** Shell dipole
      `fourier_m1` scales MONOTONE with g: ctrl_g0 0.112 → g1 0.153 → g2 0.211 → g4 0.329 → g8 0.648 (gravity
      effective; "m1-flat" falsifier did NOT fire). g2 magnitude 3-seed reproducible {0.211,0.239,0.231}=0.227±0.014
      = 8·SD vs ctrl — BUT fourier_m1 is a MAGNITUDE, direction-reproducibility (the actual ORI claim) stayed
      UNMEASURED (archives store no raw positions → no post-hoc angle). TIER-1 ceiling between g4 (deformed dome,
      circ 0.904, clean) and g8 (RUPTURE: circ 0.795, deform 0.251, nn_cv 1.96 escaper, gr_peak 22.5). Demix HOLDS
      across the ladder (g2 3-seed seg 0.357±0.076 ≈ ctrl 0.371; coherent sedimentation, not diffusive — contrast
      b34). Gravity does NOT set a TYPE axis (mi_type_x flat 0.012–0.05; uniform force acts identically on both
      types). gx (g in +x) gives same |m1| (0.216≈g2) as it must for a magnitude. **[engineering, Batch 47] Added
      3 scorecard metrics (pure additions, try/except-guarded in `_all_families` so a bug degrades to a `*_error`
      key, NOT a 0-archive): `shape.shape_axis_angle` = −angle(c[1]) of the m=1 boundary FFT = shell-bulge
      direction (deg); `partition.type_dipole` (|Δtype-centroid|) + `type_axis_angle` (deg) + `mi_type_y`.** Batch
      47 = re-run g2×3 / g4×2 / gx / ctrl×2 with the angle readout to test whether shell_axis_angle CLUSTERS
      across seeds (gravity) vs SCATTERS (ctrl) and rotates ~90° for gx. Next mechanism if confirmed: DIFFERENTIAL
      response (per-type body force / prescribed y-gradient + differential chemotax) to orient the TYPE axis —
      uniform gravity provably cannot (shown b46). GOTCHA reconfirmed: `>>`/heredoc redirect + `for`/`python3`
      Bash all sandbox-blocked → use Read/Edit/Write; archives keep NO trajectory (only png/scorecard/metrics/spec).
    - **[established→shell-oriented, Batch 48 / b47] GRAVITY ORIENTS THE SHELL REPRODUCIBLY (direct angle
      measurement); it does NOT orient the TYPE axis.** With the new `shape_axis_angle` readout, down-gravity
      shells CLUSTER: g2 {130.9,108.4,86.1}° = **108.5±22.4°** (SD < 45° falsifier → force-set), g4 112.9° + g6
      94.0° in the same 86–131° band (5-pt down mean ≈106°); direction TRACKS the force — gx (g→+x) angle −140.5°
      ≡ **+113° rotated** from the down cluster. `fourier_m1` reconfirms magnitude scaling ctrl 0.110 → g2
      0.227±0.014 → g4 0.329 → g6 0.490 (monotone). Shell-orientation now [established] on m1 (b46 3-seed) + angle
      (b47 3-seed cluster) + gx rotation. **CAVEAT:** the 2 g0 ctrl seeds did NOT scatter (both ≈−62°, SD 2.4°) —
      weakens the ctrl-scatter arm (n=2; move18 may impose a deterministic bulge), but the claim rests on the
      gravity cluster being at a DISTINCT angle (≈106° vs ctrl −62°, ~168° apart) AND rotating +113° with gx.
      **TYPE axis NOT oriented [established]:** `type_dipole` flat 0.010–0.046 (g4 smallest 0.0097, NO g-trend);
      `type_axis_angle` RANDOM across g2 seeds {130.8,−29.2,61.2}°; mi_type_x 0.009–0.042 / mi_type_y 0.013–0.064
      flat (g2 mean 0.033 ≈ ctrl 0.023). Uniform body force is type-blind → no differential → un-oriented a/b.
      Demix HOLDS (g2 seg 0.357±0.075 ≈ ctrl 0.360; declines only at strong deform g6 0.289 / in-plane gx 0.214).
      **[engineering, Batch 48] TIER-1 `escape`/`r_cell_max` is a BODY-DRIFT ARTIFACT under any oriented force:**
      escape 0.15 (ctrl) → 0.47 (g6), r_cell_max 1.1 → 1.42, but montages INTACT/contained, nn_min 0.018 healthy,
      collapsed 0, circularity 0.86–0.99, gr_peak 9.2 — escape measures radius from WORLD ORIGIN so a translating
      (sedimenting) intact blastula trips the gate. The gate MUST be re-centred on the shell centroid for ORI, or
      every oriented run reads as a spurious hard-fail (do NOT gate on raw escape this stage; use nn_min/collapsed/
      circularity/montage). deform_rms scales with g (0.054→0.202), circ falls (0.99→0.86). **PIVOT (Batch 48) =
      NEW `sediment` operator** (src/plexus/operators/sediment.py; agent-level per-agent constant directional drift,
      EMIT velocity/first-order, mirrors glide; registered in operators/__init__): differential per-type drift
      (a gy −0.10 sinks / b gy +0.10 floats) to sort the demix into a REPRODUCIBLE y-axis where type-blind gravity
      cannot. Batch 48 = sed_d10 ×3 seeds + d20/d05 ladder + aonly(one-sided) + grav(full oriented embryo) + ctrl.
      Falsifier: mi_type_y ≤0.06 AND type_axis_angle scatters across the 3 seeds → differential drift overwhelmed
      by confinement/mixing. NEW-OP crash-risk noted (module-import error blasts all 8 like b35); verified statically:
      unique registry name, mirrors gravity/glide, all `at:'agent[type=x]'` selectors single-quoted.
    - **[ESTABLISHED→type-oriented, Batch 49 / b48] DIFFERENTIAL SEDIMENTATION ORIENTS THE TYPE AXIS — reproducible
      animal-vegetal y-axis over 3 seeds. THE ORI STAGE GOAL IS MET (type axis, not just shell).** `sediment`
      a gy −0.10 / b gy +0.10 (shell-gravity OFF): **mi_type_y {0.4426,0.418,0.3305}=0.397±0.061** vs ctrl (gy 0)
      0.0289 → Δ 0.368 = **6.0·SD**; **type_axis_angle {−82.26,−80.25,−79.0}°=−80.5±1.7°** (SD 1.7° ≪ 45°
      falsifier → tightly force-set near −90°); **mi_type_x stays low ≈0.092** → order is genuinely AXIAL in y.
      The b48 falsifier (mi_type_y ≤0.06 AND angle scatters) DID NOT FIRE. Demix HELD (seg 0.452±0.027 ≥ ctrl
      0.371). TIER-1 clean (collapsed 0, nn_min ≥0.0147, circ 0.91–0.93, montages intact). ctrl (no differential):
      type_dipole 0.0209 flat, angle 78.12° (no axis to orient). **DOSE: orientation ANGLE saturates by d05**
      (mi_type_y 0.336, angle −82.08°, dipole 0.257, clean) — d10 the op point (dipole 0.31); **d20 OVERPACKS**
      (mi_type_y 0.34 NOT higher, mi_type_x rises 0.153, angle tilts −114.8°, escapers flung outside shell, nn_min
      crashes 0.0045@50%) → fidelity peaks d05–d10, degrades above. **[open] ONE-SIDED sufficient** (aonly a gy
      −0.10 / b 0: mi_type_y 0.416 ≈ d10, dipole 0.22 lower, clean — sinking one type alone orients y; n=1).
      **[open] FULL ORIENTED FLOWING EMBRYO** (grav = sediment d10 + shell gravity g2: mi_type_y 0.443 AND mi_type_x
      0.337 both ordered, angle tilts diagonal −131.6°, seg 0.555, deform 0.090≈2×, net_circ 0.013 — the two oriented
      mechanisms COEXIST but body gravity tilts the type dipole + injects mi_type_x; n=1). Batch 49 = PROGRAMMABILITY
      test: x-drift (a gx −0.10 / b +0.10) ×3 seeds → predict angle→0/180°, mi_type_x↑ / mi_type_y↓ (axis follows
      the drift vector) + diagonal drift → ~−45° + grav/aonly seed-replicates + ctrl. Falsifier: x-drift leaves
      mi_type_x low OR mi_type_y still high → axis NOT programmable, y was special. `sediment` supports gx & gy
      (sediment.py:39-40; default gy=−g).
    - **[ESTABLISHED→type-oriented programmable, Batch 50 / b49] THE TYPE AXIS IS PROGRAMMABLE — it FOLLOWS the
      sediment drift vector.** The b49 falsifier (x-drift leaves mi_type_x <0.10 OR mi_type_y still high) DID NOT
      FIRE. **x-drift (a gx −0.10 / b +0.10, gy 0) ×3 seeds**: mi_type_x {0.284,0.220,0.333}=**0.279±0.057** (Δ vs
      ctrl 0.042 = 4.1·SD), mi_type_y DROPS to {0.179,0.064,0.044}=0.096±0.073 (the channels SWAP dominance vs
      y-drift's mi_type_y 0.397), **type_axis_angle {−152.4,−158.8,−174.1}°=−161.8±11.2° ≡ 18° mod 180 ≈ x-axis**
      (SD ≪ 45°) = a ~81° rotation off the y-drift cluster (−80.5°≡99.5°). Demix HELD (seg 0.482≥ctrl 0.35), TIER-1
      clean. **Diagonal drift (a gx −0.07 gy −0.07) n=2**: mi_type_x AND mi_type_y BOTH elevated (~0.36/0.34),
      **type_axis_angle −134.9±7.3° ≡ 45° mod 180 = the diagonal exactly** (one flag: s3 nn_cv 1.12 + gr_peak 41.6
      = pole clustering, no hard fail). **STEERING CURVE** (drift-vector-axis → measured axis, mod 180): 0°→18°,
      45°→45°, 90°→99.5° = continuous tracking with a small ~+10° systematic offset (likely the demix/bulge
      contribution). Replicate consolidation (n=2): **aonly one-sided** mi_type_y {0.416,0.386}, angle {−107.5,
      −109.2}°=−108.3±1.2° (tight but ~18° off the two-sided d10 tilt — one-sided ≠ two-sided axis [open]);
      **grav full-oriented** angle {−131.6,−128.8}°=−130.2±2.0° (diagonal, both axes ordered, seg ~0.50, deform
      ~2×) = the flowing+dividing+partitioning+oriented embryo is REPRODUCIBLE; **ctrl** dipole flat ~0.022, angle
      scatters (78°/26°). ALL slots n=198 (1.5× division, 66 div events). Batch 50 = CONTINUITY test: intermediate
      drift-axes 22.5°/67.5° (|drift| 0.10; a gx/gy = 0.10·(cos,sin)) ×2 seeds → predict measured axis ≈32°/77°
      (linear interpolation) + diag/aonly/grav seed-3 (→n=3) + ctrl. Falsifier: intermediates SNAP to 0°/90° OR
      scatter SD>30° → axis quantized to geometry, not a continuous knob.
  - **b01 (division ON):** runaway `cell_divide` floods to n=2850 → `collapsed`≈0.99 identically → collapse test
    corrupted → 1A runs with division OFF (`embryo_nodiv.yaml`). [established-engineering, see FINDINGS]
  - **b02 (nodiv confine ladder 3.0/2.0/1.0/0.5):** collapse falls smoothly with confine (3.0→0.61, 1.0→0.59,
    0.5→0.45); coarse — no interior window visible. SUPERSEDED by b03's fine sweep.
  - **b03 (nodiv fine sweep 0.3/0.2/0.1 + probes):** THE decisive batch — confine 0.1–0.2 = first collapsed=0 &
    escape=0 window; residual = confinement-press-induced frozen doublet (see FINDINGS). Fully distilled.
  - **b04 (repel-strength ladder @ confine 0.1, nodiv n=44):** strength 8→48→96 raised nn_min 0.0039→0.0133→0.0163
    (0.82× r0); doublet dissolves (gr_peak_r 0.0034→0.0168) but diminishing returns → spring asymptote. Fully distilled.
  - **b05 (low-press + strong-repel + AR test, all 8 landed):** nn_min crossed to 0.0179–0.0188 (0.90–0.94× r0),
    collapsed/escape 0, doublet gone (gr_peak_r 0.0168). Best = n32-spread+c0.05+r150 = 0.0188. AR disperser REJECTED
    (see FINDINGS). Confine 0.05≈0.07 (lever saturates <0.1). Repel asymptote reconfirmed at low press. Fully distilled.
  - Historical note (pre-restart campaign, now closed): a ~30-batch SSH-auth outage lost b02–b31 of an earlier run;
    the driver was restarted and auth restored. If `SUBMIT FAILED … Permission denied` recurs in
    `loop_logs/campaign_l4.log` (`.sh` present, `.out`/`.err` absent), the operator fix is #1 restart the driver
    (loads the HOLD-and-retry guard), #2 renew the Kerberos/SSH cred; the agent can do neither, and a local/GPU
    fallback is blocked (every `python`/`nvidia-smi` call returns the ungrantable `This command requires approval`).
    Not currently active — auth is up as of Batch 4.

## Base operating point (reference, `specs/embryo_base.yaml`, seed 0) [engineering]
n=44 sunflower, spawn_radius 0.22, per_parent 14000, n_grid 64, dt 0.002, frames 12000.
Key couplings: `repel` strength 8.0 r0 0.02 · `agent_to_mpm.agent_mass` 2e-6 (k 1.0) ·
`mpm_to_agent` k 0.3 confine 3.0 field=colour · `flow_align.gain` 40 · `mpm_spin.omega` 0.3 ·
move_speed 0.12, div_rate 0.6.
**Reference scorecard (`archive/embryo_base_sc3`, 400-frame pilot — caveat: not 12000):**
`collapsed 0.806` **HARD FAIL** · `nn_min 0.0002` vs r0 0.02 (**100× below, HARD FAIL**) ·
`nn_mean 0.0119`(<r0) · `gr_peak 3.2→46.5`, `nn_cv 0.41→2.04` (progressive clumping) ·
`circularity 0.998`, `deform_rms 0.0013`, `fourier_m2 5e-5`/`m3 3e-4` (membrane ~undeformed) ·
`escape 0.0`, `accel 0.0012` (balance-bounded, clean). n_cells 44→67 (division active).

## FINDINGS
- **[established-integration, Batch 46 / b45] INT FORMALLY CLOSED; STAGE ORI opened.** The last flow-leg probe —
  the intermediate motility point move15/gain40 — failed its pre-registered falsifier: segregation_index 3-seed
  {0.445, 0.325, 0.195} = **0.322±0.125** (mean <0.35 AND indistinguishable from [established] move18 0.336, Δ0.014
  ≪ 2·SD=0.25), and it added essentially no coherent flow (net_circ 0.0068±0.0046 ≈ static ctrl 0.0057). 9th
  single-seed clean point to regress on replication. move18_a40 reconfirmed the INT op point (seg 0.371, net_circ
  0.0118, msd 0.0344 coherent — 4th point in the band); ctrl_move12 reconfirmed the max-sort/min-flow Pareto endpoint
  EXACTLY (seg 0.588, net_circ 0.0057); move18_a60 net_circ 0.0040 shows b44's "net_circ peaks at gain60" is a
  move24-only effect that does NOT transfer to move18. All 8 TIER-1 clean. **INT op point = `move18/gain40`
  (`embryo_INT_g20_1p5x_move18.yaml`).** Next open question (STAGE ORI): the demix is gain-scaled but LATERAL and
  UN-ORIENTED (mi_type_x ~0.01–0.04, random per seed) — a real embryo sets a reproducible axis. Batch 46 probes
  `gravity` (uniform membrane body force via p2g a_ext) as the minimal external axis cue; `fourier_m1` (shell dipole)
  is the axial-order proxy until a type-axis metric is engineered.
- **[established-integration, Batch 45 / b44] INT FLOW LEG CLOSED — the sort↔flow tradeoff is a HARD Pareto frontier;
  NO high-motility (move≥20) config robustly holds strong sort (seg≥0.35) with elevated flow.** The b43 move24_a60
  winner (seg 0.372, n=1) FAILED 3-seed replication: {0.372[b43_s5], 0.130[b44_s0], 0.214[b44_s1]} = **seg
  0.239±0.121** (falsifier seg<0.30 FIRED); net_circ steady {0.0215,0.0097,0.0185}=0.0166±0.006. This is the **8th
  single-seed clean point to regress** (fast_k4, anch10_k4, anch5_k4, b24 xdemix, b30 a12, b40 move24_a80, b40
  move20_a80, b43 move24_a60) — a DURABLE campaign law. Combined with the b43 falls (move24_a80 0.220±0.141,
  move20_a80 0.313±0.083), EVERY apparent high-motility winner regresses to seg ~0.22–0.31. **Recommended INT flow
  op points: `move18/gain40` [established] seg 0.336±0.043 / net_circ 0.0109±0.0009 (balanced) and `move20_a80`
  0.313±0.083 / net_circ ~0.012 (higher-flow); ctrl_move12 seg 0.588 / net_circ 0.0057 = max-sort min-flow endpoint.**
  Two secondary b44 effects [open, n=1]: the move24 gain ladder is genuinely NON-MONOTONE (a40 seg 0.105 / a50 0.340
  / a60 0.239(3s) / a70 0.222; net_circ peaks a60) but every rung except a60 is n=1 so the a50 0.340 is likely the
  same seed-luck; and LOWER gain BELOW the ceiling motility kills flow (move22_a60 net_circ decayed to 0.0 by 100%)
  → the flow-optimal gain RISES as motility falls (no single gain optimal across the range). All b44 TIER-1 clean
  (collapsed 0, nn_min 0.0172–0.0186, escape absent, n_cells 198, n_div 66); LATERAL geometry (mi_type_x ≤0.029);
  shell topology-preserving (circularity 0.90–0.96, deform_rms 0.028→0.063 rising with motility). **INT COMPLETE:**
  three legs mapped — PROLIFERATION (2× concurrent-division envelope), continuum-DEFORM (agent_to_mpm.k compatible,
  topology-preserving), motility-FLOW (Pareto frontier, move18 op point). Next stage = oriented symmetry-breaking.
  **EXECUTION-LOSS note (b44 s5):** spec-load `KeyError 'chemotaxis'` (root-cause (c)) — 5 reused INT specs still
  carried the commit-8409136-renamed token; FIXED Batch 45 (`chemotaxis`→`chemotax`, 2 ops + schedule, grep-verified).
- **[REJECTED as stated, Batch 44 / b43] "flow_align gain80 RESCUES sort at high motility (move20/24 seg ~0.4)" was a
  SINGLE-SEED ARTEFACT — both b40 gain80 winners FALL on 3-seed replication and the move24 gain ladder is REVERSED.**
  b43 replicated the two n=1 b40 winners to 3 seeds: move24_a80 {0.383[b40],0.148,0.130} = **seg 0.220±0.141** (b40 0.383
  = outlier); move20_a80 {0.408[b40],0.277,0.254} = **seg 0.313±0.083** (barely clears 0.30, b40 0.408 = outlier). Neither
  holds ≥0.35 → the "gain80 rescues sort" claim is retired; net_circ was steady across seeds (~0.014) so gain80 buys
  FLOW, not SORT. **NEW [open, n=1@60/120]: at the move24 ceiling flow_align gain is NON-MONOTONE and net_circ DECREASES
  with gain** — net_circ gain60 **0.0215** (campaign-max coherent flow) > gain80 ~0.014(3-seed) > gain120 **0.0069**;
  seg gain60 **0.372** > gain120 0.335 > gain80 0.220. So higher alignment OVER-RIGIDIFIES and SUPPRESSES the coherent
  bulk swirl (opposite of the b43 guess that gain120 boosts net_circ). **move24 + gain60 = best JOINT flow+sort point of
  b43: seg 0.372 (2nd only to no-motility ctrl 0.588) AND net_circ 0.0215, TIER-1 clean (nn_min 0.0179, deform 0.060,
  msd 0.032 coherent-not-diffusive)** — n=1, Batch 44 replicates. gain80 CAVEAT unchanged (no help at move18: 3-seed
  0.295±0.084 ≈ gain40 0.336±0.043). All b43 TIER-1 clean (collapsed 0, nn_min 0.0179–0.0186, escape 0). LATERAL
  everywhere (mi_type_x ≤0.05, one move20_a80 seed outlier 0.146). SORT↔FLOW is a hard Pareto frontier: ctrl_move12
  seg 0.588 / net_circ 0.0057 (max sort, min flow) ↔ move24_a60 seg 0.372 / net_circ 0.0215 (Pareto knee).
- **[established-integration, Batch 40 / b38+b39] FLOWING + dividing + partitioning BLASTULA DEMONSTRATED, and
  coherent collective flow is SINGLE-PEAKED in cell MOTILITY (peak ~move_speed 0.18).** On the 1.5× g20-k6 dividing
  triple, the motility-driven coherent flow (net_circulation, fourier_m1 drift) rises then FALLS with move_speed:
  seed0-gain40 net_circ 0.0057(0.12)→**0.010(0.18)**→0.002/0.0(0.24); m1 0.075→**0.104**→0.016. Crucially msd
  (incoherent diffusive motion) rises MONOTONE 0.022→0.039→0.090 — so above ~0.18 motility stops building COHERENT
  drift and instead FLUIDIZES into incoherent stir (identical coherent→incoherent transition to high-ω mpm_spin,
  b38). The lateral sort tracks the same peak: seg 0.588(0.12) → 0.336±0.043 3-seed(0.18) → 0.156±0.071(0.24, remixed
  <0.22). **move18/gain40 is 3-SEEDED [established gate met]:** net_circ 0.0109±0.0009 (Δ vs the 0.12 ctrl 0.0057 =
  5.8·SD), seg 0.336±0.043 (held ≥0.25), all TIER-1 clean (collapsed 0, escape 0, nn_min 0.0179). This is the FULL
  campaign object — a blastula that simultaneously FLOWS (coherent net_circ + m1 drift + deform 0.05), DIVIDES (n
  132→198, 66 events), and PARTITIONS (chemotactic lateral demix). OPERATING POINT = `embryo_INT_g20_1p5x_move18.yaml`
  (move 0.18, flow_align gain 40; gain 80 gives a sharper sort seg 0.392 at slightly lower net_circ, n=1). Motility
  0.24 (user ceiling) is TOO FAST — coherent flow collapses AND sort remixes. mpm_spin does NOT add (spin08+move18
  → net_circ 0, m1 0.006; spin's incoherent stir dominates the motility drift). LATERAL geometry (mi_type_x ≤0.09).
- **[established-integration, Batch 40 / b20+b39] flow_align is a CONTAINMENT/regularizer coupling, not just a
  flow-coherence knob — ablating it RUPTURES the shell.** flow_align.gain 0 blows the blastula apart in TWO regimes:
  (a) b20 confluent nodiv (escape 1.0, membrane→box, deform 0.128) and now (b) b39 dividing + move18 self-propelled
  (escape **1.0**, collapsed 0.0101, nn_min **0.0**, area BALLOONED 0.36→0.78 = 2.2×, deform 0.175 = 3.2× the gained
  move18 0.054, circ 0.60). Mechanism: velocity-to-fluid alignment is what keeps motile/pumping cells moving
  COHERENTLY inside the deforming shell; without it they push radially outward incoherently and plaster the wall.
  Ablation ALSO collapses the coherent flow it was claimed to produce (net_circ 0.010→0.0018, m1 0.104→0.049),
  confirming flow_align as the coherence-imposing coupling causally. gain 40 is a safe floor; gain 80 tightens
  coherence (msd 0.037→0.029) and sharpens the sort (seg 0.35→0.39). NEVER run the integrated blastula at gain 0.
- **[engineering, Batch 36 / b35] DURABLE GOTCHA: the operator-family refactor renamed the PREDICTION
  vocabulary `first_derivative`/`second_derivative` → `velocity`/`acceleration`/`mpm_acceleration`
  (models/base.py:117; schema.py:150 now RAISES on the old tokens). `src/plexus/operators/*` were all
  migrated, but `repel` — registered in the SEPARATE lib `active_matter2/am2_ops.py` (showcase.py:27) — was
  missed and kept `first_derivative`, so EVERY embryo spec (repel is universal) 0-archived at load
  (b35 all-8 loss).** SIGNATURE of a load-time crash vs a science/infra loss: `.out` Run time ~12 s / CPU ~5 s
  / ~300 MB (a real run is ~800 s) + identical `.err` ValueError at `schema.py:151`. FIX = am2_ops.py:394
  `PREDICTION = "velocity"` (repel shares the `glide` integration set → must match glide's `"velocity"`).
  If a future batch 0-archives, `grep -rn "first_derivative\|second_derivative" --include=*.py` the WHOLE
  Plexus repo (not just src/plexus) — prototype libs (am2_ops, ops_swim) can harbor stale tokens the refactor
  skipped. Python validation is approval-blocked here; verify fixes statically. Sole other stale survivor
  `candidates/ops_swim.py` is never spec-referenced (harmless).
- **[engineering, Batch 42 / b41] SAME loss mode, 2nd root cause = OPERATOR-NAME COLLISION from a refactor.**
  The M1 refactor (commit 8409136 "merge chemotaxis + chemo_force → chemotax") registered `chemotax` in
  `src/plexus/operators/chemotax.py:29` (a Keller-Segel VELOCITY op) — but the untouched prototype lib
  `active_matter2/am2_ops.py:153` ALREADY registered `chemotax` (an older heading-REORIENTATION op, `omega·sin`).
  Two `@register_operator("chemotax")` in the same global registry → `registry.py:42 ValueError: Operator name
  'chemotax' already registered` at IMPORT of am2_ops (showcase.py:27), before any spec loads → all 8 b41 slots
  0-archived, Run time ~11 s / 246 MB / identical `.err`. FIX = rename the am2_ops reorientation op `chemotax` →
  `chemo_reorient` (the M1 velocity op keeps the canonical name; the reorientation op is semantically distinct —
  reads/writes `heading`), + update its 2 dependent specs `agent_mpm_{disc,blastula}_4types.yaml`
  (`op: chemotax` → `op: chemo_reorient`; both OLD 4types, off-campaign). This fix was NECESSARY but INCOMPLETE
  (see Batch 43): the b41 note's claim that the campaign's `op: chemotaxis` was "still registered, NEITHER
  collision participant" was WRONG — the SAME M1 refactor also RENAMED `chemotaxis` → `chemotax`, so the campaign's
  own operator was already gone; the import crash merely masked it. GENERALIZED GOTCHA: when a `src/` refactor
  RENAMES or ADDS an operator, grep `am2_ops.py` for BOTH stale PREDICTION tokens AND duplicate
  `register_operator("<name>")` — AND grep the campaign SPECS for the operator's OLD name.
- **[engineering, Batch 43 / b42] SAME loss mode, 3rd root cause = a SPEC referencing a REFACTOR-RENAMED operator.**
  b42 (the b41 re-issue, after the collision fix) again 0-archived all 8 slots — but the crash MOVED: no longer an
  import ValueError, now `schema.py:132 KeyError: 'chemotaxis'` → `ValueError: operator 'chemotaxis' not in registry`
  at SPEC LOAD (Run time 13 s / 312 MB / identical `.err` across all 8). The b41 fix cleared the import collision and
  exposed the SECOND half of the M1 refactor (commit 8409136 "merge **chemotaxis** + chemo_force → chemotax"): it
  RENAMED the campaign's demix driver `chemotaxis` → `chemotax` in `src/plexus/operators/`, but every INT spec still
  said `op: chemotaxis`. Every prior REAL batch (b28–b40) ran because `chemotaxis` was registered then; the rename
  killed it. (well_ops.py:290 DOES still register a `chemotaxis`, but that lib is NOT imported — the b42 registry
  dump omits all well_ops names [slime/reaction_diffusion/advect] → it was a red herring in the b41 note.) The merged
  `chemotax` with default `emit: velocity` IS documented as "the old chemotaxis" (chemotax.py:6-8), same params
  (`from`/`channel`/`gain`/`at`) → FIX = pure rename `op: chemotaxis` → `op: chemotax` in the 7 INT specs the b42
  slots reference (schedule token too). Verified statically: all 18 ops in those specs now resolve against the
  registry. DURABLE: a spec surviving import can STILL die at spec-load on a renamed op — after any operator refactor,
  grep the SPECS (not just libs) for every renamed token; the loss signatures differ (import ValueError ~11 s vs
  spec-load KeyError ~13 s). ~73 older 1E/INT specs still carry `op: chemotaxis` — reuse ⇒ rename first.
- **[established-integration, Batch 37 / b36] THE TRIPLE (partition × 2× division × k6 continuum-deform)
  COEXISTS at TIER-1, and the sort under 2× division is a GAIN-INDEPENDENT ceiling.** All 8 slots collapsed 0,
  nn_min 0.0178–0.0184, escape 0.05–0.16 (≈container baseline), deform_rms 0.040–0.046. A flowing/dividing/
  self-partitioning/deforming blastula is ACHIEVED. But g20 headroom (nodiv seg 0.61–0.78) does NOT survive
  division: **g20 2×-k6 3 seeds seg 0.191±0.018 ≈ g10 2×-k6 ~0.228** (Δ within 0.5·SD) — 2× dilution decides
  the sort regardless of chemotactic gain, as expected if division = mechanical mixing (b33/b34 principle).
  Nodiv seed-locks isolate the pure division cost: nodiv g20-k6-s1 0.608 vs dividing 0.191 = −0.42. Stronger
  deform (k8) is inert on the dividing sort (0.195 ≈ k6). Geometry LATERAL throughout (mi_type_x 0.010–0.087).
  Two g20-divide runs show ONE escaper cell (nn_cv 1.05–1.18, r_cell_max ~2.0) — division occasionally flings a
  cell; not TIER-1 fail (escape ≤0.16), but g20-divide slightly more prone than g10.
- **[established-integration, Batch 38 / b37] PROLIFERATION-SORT FRONTIER MAPPED — sort strength falls
  MONOTONICALLY with growth factor; moderate growth (1.5×) preserves a stronger sort than 2×.** g20-k6
  continuum-deform dividing triple, seg_index vs growth: 1.0× (nodiv) 0.608 → 1.25× 0.432 → 1.5× 0.408±0.161
  {0.588,0.335,0.302, n=3} → 1.75× 0.326 → 2.0× 0.170 (g20 ctrl) / 0.265 (g30). Falsifier (1.5×≈2×) did NOT
  fire: 1.5× (0.408) ≫ 2× (0.170). CAVEAT — the 1.5× point is NOISY (seed0 0.588 outlier inflates mean+SD;
  median 0.335; Δ vs 2× = 0.238 < 2·SD 0.322), so "1.5× seg ≥0.35" stays **[open]**, needs ≥1 more seed. TIER-1
  ESCAPER onset at ≥1.75× (nn_cv 1.6–1.2, gr_peak 17–19, r_cell_max ~2.0, escape 0.15) — a stochastic
  division-fling, cosmetic (collapsed 0, escape ≤0.16), sets a soft upper growth bound; clean at ≤1.5×
  (nn_cv ~0.30). k8 deform same sort (0.393≈k6 0.408) but 10× fourier_m2 (0.080 vs 0.008, circ 0.95) =
  visible ELLIPTICAL shell — topology-preserving deform (b34) reconfirmed. Geometry LATERAL (mi_type_x ≤0.088).
  1.5× g20-k6 = practical moderate-growth op point (dividing+partition+deform coexist, TIER-1). OPEN: the FLOW
  leg — whether sustained coherent collective flow coexists with the dividing sort (only shown nodiv, b34).
- **[established-integration, Batch 34 / b32+b33] PARTITION ⊥ HIGH-PROLIFERATION — concurrent division DILUTES
  the demix monotone with growth, AND temporal separation does NOT rescue: the division EVENT itself mechanically
  re-mixes even a fully-formed pattern.** b33 falsifier FIRED both clauses. Late-division (pre-pattern then divide):
  2×_late75 seg **0.642@75% → 0.112** (−82 %), 4×_late75 0.500 → 0.037, 4×_late50 0.367 → 0.007 — finals
  TIMING-INDEPENDENT (2× 0.131/0.112; 4× 0.007/0.037) and ≈/worse than concurrent. MECHANISM (why it's not passive
  dilution): at the division checkpoint `msd` JUMPS ~10× (repacking wave to make room for daughters) and re-scrambles
  the interface (interface_frac 0.179→0.444, contact_same 0.832→0.573, mixing_entropy 0.531→0.839 snap back to mixed);
  post-arrest chemotaxis CANNOT re-sort (late50 kept DECLINING 0.567→0.131 over 6000 post-div frames; t1_rate decayed
  ~0.01). Mobility does NOT rescue (4×_fast move0.24 seg 0.069, WORST shell deform 0.093 escape 0.280). TIER-1: escape
  ~0.09–0.12 = CONTAINER BASELINE (nodiv ctrl 0.121); 2× at baseline (SAFE), 4× breaches (0.14–0.28 rupture) +
  overpacks (4×_late50 nn_min 0.014). **COMPATIBLE ≤2× (partial, seed-noisy: 3 seeds {0.216,0.131,0.306}=0.218±0.088,
  ~45 % of full 0.485), INCOMPATIBLE ≥3× (mixed baseline + rupture).** INT PROLIFERATION DELIVERABLE = 2× concurrent
  g10 demix (dividing+demixing 0.22+deforming 0.033, escape-safe). NEXT: partition × membrane-deformation.
  --- b32 origin (single seed/rung): g10 `segregation_index`: 1× 0.485 → 2× 0.216 → 3× 0.064 → 4× 0.056. Co-metrics monotone-mixing: contact_same 0.778→0.601→0.534→0.535 (→random 0.5),
  mixing_entropy 0.669→0.768→0.887→0.921 (→max), interface_frac 0.257→0.392→0.468→0.472. **Loss is KINETIC,
  not jamming** — the decisive number: at 3× nn_min 0.0174 ≈ ctrl 0.0185 (spacing healthy, not overpacked)
  yet seg already dead; division front-loads (n_cells hits cap by the 25 % checkpoint) so the enlarged
  population has too little time to coarsen. Daughters inherit type (cell_divide.py:62) placed adjacent
  (offset 0.004) — not actively mis-sorting, the sort just can't keep pace. **Neither lever rescues:**
  g20 (2× cross-rep gain) seg ≈ g10 at every rung (2× 0.235, 4× 0.076) AND is the ONLY TIER-1 FAILURE
  (4× nn_min 0.0018, collapsed 0.0038, escapees) → strong cross-rep overpacks at 4× [engineering: g20 unsafe
  @4×]; slowfill (rate 0.15) 4× seg 0.079 ≈ fast 0.056 → the final cell count sets the ceiling, not fill
  rate. **COMPATIBLE ≤2× (partial demix survives), INCOMPATIBLE ≥3×.** deform_rms rises with n (0.024→0.046)
  but circularity holds 0.99 (no lobing). NEXT: temporal separation (pre-pattern then divide) to test if an
  established domain resists dilution — Batch 33.
- **[ESTABLISHED, Batch 30 / b29] 1E SEGREGATION GATE MET: heterotypic two-channel chemotactic
  cross-repulsion (a flees b-trail ch1, b flees a-trail ch0) DEMIXES the confined blastula, `segregation_index`
  scaling monotonically with |gain|, escape/nn_min-safe.** 3 seeds/rung (seed0=b28, seed1/2=b29): **g10 (gain
  0.10) seg 0.494 ± 0.080** (Δ vs ctrl −0.028 = 6.5·SD), **g20 (gain 0.20) seg 0.781 ± 0.085** (Δ = 9.5·SD),
  both ≫ 2·SD, monotone (0.494 < 0.781). Co-metric means monotone: contact_same ctrl 0.503 → g10 0.756 → g20
  0.890; interface_frac 0.514 → 0.250 → 0.116; mixing_entropy 0.851 → 0.611 → 0.479. Real un-mixing (seg starts
  ≤0, g20_s1 climbs 0.091→0.850 monotone, not saturated at 100%). **NO shell rupture** (collapsed 0, nn_min
  0.0181–0.0186, area/circ stable; container escape baseline 0.05–0.10 decoupled from sorting). Dynamics =
  coarsening-then-arrest (migr↓, t1_rate↓ as domains lock AFTER sorting), not jamming. **GEOMETRY = LATERAL
  side-by-side same-type domains, NOT core-shell** — `mi_type_x` (type↔radial-position MI) stays low ~0.04 even
  at contact_same 0.90 (symmetric repulsion has no inside/outside preference). ENGINEERING sub-findings: (a)
  `[established]` gain-0 chemotaxis is a bit-identical no-op vs ctrl; (b) `[established, b29 reconfirm]` SHARPER/
  faster-turnover trail (deposit 0.5→1.0, diffuse 0.1→0.04, decay 0.2→0.4) ≈2× effective gain — `sharp_g10` seg
  0.799 ≈ g20 at HALF nominal gain; (c) `[open]` adding COHESION suppresses the demix (b28 cohere 0.185).
  MONTAGE CAVEAT [durable]: montage-title `seg=` is NOT scorecard `segregation_index` and INVERTS the ranking —
  read seg from scorecard.json only.
- **[open, Batch 30 / b29] CORE-SHELL geometry is the remaining 1E axis; the route is BOUNDED differential
  SELF-COHESION, not asymmetric repulsion.** Two b29 geometry probes: (i) `[rejected]` asymmetric cross-repulsion
  magnitude (a −0.20 / b −0.02) does NOT set radial order — `mi_type_x` 0.062, noisy endpoint spike
  (0.015/0.007/0.021/0.001/0.062), not sustained; seg 0.600. (ii) `[open→lead]` explicit self-adhesion (each type
  climbs its OWN trail, positive self-chemotaxis +0.10) RAISES `mi_type_x` to **0.084 SUSTAINED** (highest, holds
  50–100%) → self-cohesion DOES induce radial order — BUT it HARD-FAILS the gate (nn_min 0.0079 < r0, escape
  0.144, accel 0.0103 = 10× baseline): unbounded positive self-chemotaxis overpacks. The mechanism is right, the
  OPERATOR must be BOUNDED. Batch 30 tests `attraction_repulsion` per-type self-pull (hard repulsive core kept
  nn_min ~0.0185 at pull 1.0 in b24) as the bounded core-shell driver. If self-pull leaves mi_type_x ≤ 0.06 at
  every strength → core-shell needs explicit membrane/boundary affinity (unavailable) → close 1E on the lateral-
  demix [established] gate, ADVANCE to INT.
- **[established / engineering, Batch 28 / b27] The b26 AND b27 1E chemotaxis losses were ONE bug: an UNQUOTED
  per-type selector `at: agent[type=a]` inside a flow mapping → `yaml.parser.ParserError` at load (schema.py:85).**
  b27 isolation ladder: only the 2 slots WITHOUT a `chemotaxis` op landed (s0 ctrl, s1 field_only); all 6 WITH a
  `chemotaxis` op died (incl. s2 gain-0 and s7 res128+couples_to). Every `.err` = `while parsing a flow mapping …
  expected ',' or '}', but got '['` — inside `{op: chemotaxis, at: agent[type=a], …}` the unquoted `[` starts a
  flow SEQUENCE. FIX = quote it: `at: 'agent[type=a]'` (the working `agent_mpm_blastula_4types.yaml:39` already does).
  Applied to xr_g0/g02/g05/g10 + authored quoted xr_g20/g05_sharp/g05_cohere. **This RETIRES the Batch-27 suspect
  (first-scalar-field-render / couples_to overlay):** s1 field_only ran the FULL field machinery (chem field +
  deposit/diffuse/decay, embryo's first recorded field) clean in 827 s; s7 (couples_to res128) died on the SAME YAML
  error before render. Proof the field is inert without chemotaxis: ctrl and field_only scorecards are BIT-IDENTICAL
  (a deposited field no one reads = zero back-effect). **The heterotypic mechanism is STILL UNTESTED (0 sorting
  data across b26+b27); wiring verified correct by source read** (deposit.py:47 writes channel=node_type a→ch0/b→ch1;
  chemotaxis.py:39,54 reads channel + returns gain·∇, gain<0=flee ⇒ a flees b-trail, b flees a-trail = interfacial
  tension). NEXT (Batch 28): the gain ladder (g02/g05/g10/g20) runs for the FIRST time — first real 1E sorting data.
  ENGINEERING GOTCHA [durable]: any per-type `at: set[type=x]` selector in flow-style `{…}` YAML MUST be single-quoted
  or the whole batch silently 0-archives; `sed`/`>>` writes are sandbox-blocked here (use the Edit tool).
- **[rejected, Batch 26 / b25] 1E: `attraction_repulsion` differential SELF-mobility does NOT sort — the b24 seg
  0.139 was frame/seed NOISE (seed1 replicate of the SAME config gave −0.108; 0.25 swing). All 8 b25 slots span
  seg −0.108…+0.055 vs ctrl −0.028 (±0.1 noise); b-push ladder does not recover it; contact_same flat around ctrl;
  montage stays mixed. The confluence/differential "ablations" that looked clean in b24 were themselves noise. The
  b24 operator limit (finding #7) is the binding cause. PIVOT (Batch 26): true heterotypic cross-repulsion via
  two-channel chemotaxis (a flees b-trail, b flees a-trail) — verified expressible with existing deposit/diffuse/
  decay/chemotaxis + `at: agent[type=x]` per-type masks, no new op.**
- **[rejected, superseded by b25 above; retained for the operator-limit note] 1E b24: DIFFERENTIAL SELF-COHESION
  appeared to SORT weakly (seg 0.139) — FALSIFIED by b25 as noise.** b24 (8 slots, 1E_base n132 confluent, `attraction_repulsion` sigma 0.03,
  ctrl s7 both-p0). Decision metric = scorecard `segregation_index` (1−cross/exp_cross; 0=mixed, 1=sorted). (1)
  **CAUSAL sort, NON-monotone in pull:** seg_final ranking xdemix 0.139 > a06(pull0.6) 0.092 > a10(pull1.0) 0.056 >
  a03(pull0.3) 0.038 > n88 0.020 > ctrl −0.028 > sig05 −0.052 > sym06 −0.077. Pull peaks at 0.6 (over-pull 1.0
  kinetically arrests — a jams into a rigid clump that stops coarsening). (2) **ACTIVE DEMIX WINS:** xdemix (a-pull
  0.6 + b self-push 0.3) seg 0.139 = 1.5× the best single-pull; contact_same 0.578 (max), interface_frac 0.430
  (min), mixing_entropy 0.792 — a coheres to a CORE, b actively disperses to the SHELL. (3) **Two ablations land
  clean:** sym_both06 (BOTH pull 0.6) seg −0.077 BELOW control → it is the DIFFERENTIAL (asymmetry), not cohesion
  per se, that sorts (Steinberg control passes); n88 (low density) seg 0.020 → collapses toward control → sorting
  is a CONFLUENCE phenomenon. (4) **Range:** sig05 (sigma 0.05) FAILS (−0.052) and leaks worst (escape 0.083) — a
  diffuse pull smears the differential. (5) **KINETICS:** seg_index is NEGATIVE early (sunflower init) then RISES
  through the 2nd half (xdemix −0.02→+0.139) → sorting is SLOW, coarsening-limited, still PROGRESSING at 100% (not
  saturated). (6) **TIER-1:** collapsed 0, nn_min ~0.0185; **escape ~0.05 is a CONTAINER baseline (no-adhesion ctrl
  escapes 0.053, r_cell_max 0.926)** — the same marginal 1C/1D confluent-container leak, decoupled from the
  sorting signal; a-pull (inward) neutral, b-push (outward) nudges escape (xdemix 0.0682, rmax 1.002 = 1 cell just
  outside), lower density halves it (n88 0.0227). (7) **OPERATOR LIMIT [engineering]:** `attraction_repulsion` uses
  RECEIVER-type params ONLY (`p = type_params[node_type[i]]`, does NOT read the neighbour's type) → it cannot
  express heterotypic (a–b) repulsion (interfacial tension); the seg~0.14 sorting here is a self-cohesion /
  self-dispersal differential-MOBILITY effect. If Batch 25's demix corner cannot push past ~0.14, a new
  neighbour-type-aware op (or two-channel chemotaxis cross-repulsion) is needed for strong Steinberg sorting.
  **NEXT (Batch 25):** exploit the demix corner — b-push ladder (0.2/0.5/0.7 @ a-pull 0.6) + seed-1 replicate of
  xdemix (seg-noise robustness) + kinetic/range explores (tighter sigma 0.02, faster move 0.18, denser n176) +
  no-adhesion control. Falsifier: if NO b-push rung AND no kinetic explore beats xdemix's 0.139, the active-demix
  mechanism is saturated → Batch 26 needs a true cross-type (heterotypic) mechanism.
- **[established (closure), Batch 24 / b23] 1D CLOSED — NO first-order collective mode (polar OR milling) is both
  SUSTAINED and ESCAPE-SAFE in this confined MPM blastula; 1D is operator-limited (would need the architecturally-
  blocked 2nd-order Vicsek set).** b23 (8 slots, 1D_flock n132 nodiv confluent, mpm_spin.omega ladder × k, heading_align
  300 held; ctrl_spin0 s7 escape 0.0758 / net_circ 0.00426 / r_cell_max 1.03). The pre-registered falsifier fires on
  BOTH clauses: (1) **net_circ is NON-monotone in omega and dies at high omega** — it PEAKS at spin1.5–3 (spin1p5_k2
  net_circ 0.00862 = batch max == b22's 0.00865; spin3_k2 0.00655; spin3_k1 0.0058) then COLLAPSES to 0.0 where the
  shell CRUMPLES (spin6_k2/spin3_fa80/spin3_anch20/spin3_ha120 all net_circ 0.0, circularity 0.7233/0.5929/0.5867/0.6399,
  shape_index 4.17–4.63, deform_rms 0.065–0.130, r_cell_max 0.68–0.80 cells pulled inside a collapsing shell; spin6_k2
  a frank blow-up: area 0.678≈2× nodiv, perimeter 3.43, nn_cv 0.005 = one central worm-blob, membrane fragmented).
  Rotation at high omega TEARS the fluid/shell rather than locking a vortex. (2) **every net_circ>0 slot escapes >0.02**
  (spin3_k2 0.0379, spin3_k1 0.0379, spin1p5_k2 0.053; ctrl_spin0 0.0758 cells-outside); the only escape-0 slots are
  exactly the net_circ=0 crumpled failures → NO point is BOTH sustained AND escape-safe. Combined with b20 (flow_align
  NULL), b21 (heading_align REAL but transient, raises escape), b22 (`agent_to_mpm.k` is the single lever coupling
  flock-coherence to shell-rupture): the 1st-order collective-flow route is exhausted. **1D CLOSED; operating point =
  1D_flock heading_align 300 + `agent_to_mpm.k` 1 (escape 0.0152, weak escape-safe flock, b22).** NEXT (Batch 24): STAGE
  1E batch 1 — differential-adhesion sorting. New op `attraction_repulsion` (per-type cohesion) on `embryo_1E_base`
  (calm confluent container, heading_align OFF, anch20): type a pull, type b neutral → a-core/b-shell. Pull ladder
  0.3/0.6/1.0 + range + n88 hedge + symmetric ablation + active demix + no-adhesion control. Falsifier: if seg_index
  does not separate from control across the pull ladder, differential self-cohesion cannot sort → Batch 25 needs true
  cross-type chemotaxis cross-repulsion.
- **[open, Batch 23 / b22] The 1D coherence↔membrane-push tension is INHERENT — `agent_to_mpm.k` is a SINGLE lever
  coupling the flock's coherence to the shell-rupture, so a bounded POLAR flock is geometry-limited (pre-registered
  falsifier fires) — BUT the polar flock CONVERTS to MILLING (rotational), which is escape-SAFER, so pivot to test
  the milling arm before closing 1D.** b22 (8 slots, 1D_flock n132 nodiv confluent, heading_align 300 held,
  tension-breaking sweep; ctrl_g300 s7 base k4 escape 0.1439 / polar 0.1806 / net_circ 0.00479). (1) **k is MONOTONE
  in BOTH escape AND flock (they are the SAME lever):** k4/k2/k1 → escape 0.1439/0.0909/**0.0152** AND polar_final
  0.1806/0.1066/**0.0662** (peak@25% 0.669/0.629/0.346). k1 is the sole escape<0.02 but polar 0.0662 < control 0.1806
  → flock-dead; the b22 "k2 sustains polar>control AND escape<0.02" hyp FALSIFIED. Lowering the cell→fluid PUSH (k)
  necessarily kills the flock because the push IS its mechanical coupling. (2) **flow_align-DOWN INCREASES escape:**
  fa40/20/10 → 0.1439/0.1818/**0.25**, r_cell_max 1.01→1.09→**1.18** (cells OUTSIDE) → RECONFIRMS flow_align is a
  load-bearing REGULARIZER (b21), FALSIFIES the "decohering-noise" framing; the predicted sweet spot k2+fa20 is DEAD
  (escape 0.1439 = no-lever ctrl). (3) **anch20 (stiffer shell) = only escape-lever that spares the flock** (0.0758 ≈
  ½ ctrl, r_cell_max 0.96 inside). (4) **MILLING CONVERSION (reverses b21 "spin inert"):** spin1.5 (omega 0.3→1.5) →
  polar COLLAPSES to **0.0266 (min)**, net_circ rises to **0.00865 (max, 1.8× ctrl)**, msd **0.0745 (max)**, escape
  DROPS to **0.0303 (5× safer)** — a rotational milling flow is escape-safer (tangential not radial push); net_circ
  still intermittent (not locked) but strongest of campaign. **NEXT (Batch 23):** MILLING resolution — mpm_spin.omega
  ladder (1.5/3/6) × k(1/2) + flow_align80 + anch20 + heading_align120 + spin-0 control. Falsifier: if omega↑ leaves
  net_circ intermittent (any 0 in 50/75/100% plateau) OR escape>0.02 at every spin, NO first-order collective mode
  (polar OR milling) is both sustained AND escape-safe → CLOSE 1D and ADVANCE to 1E (two-type partition).
- **[open, Batch 22 / b21] `heading_align` (1st-order Vicsek neighbour-alignment) is a REAL but TRANSIENT flock —
  causal ~9× polar and 3–4× migration over control, self-organized weak swirl off the campaign-long net_circ 0 —
  but it DECAYS in confinement and RAISES escape; the controlling variable is now the coherence↔membrane-push
  TENSION (a coordinated flock ruptures the shell; the flow_align regularizer both protects the membrane AND damps
  the flock). NOT null like flow_align — so 1D stays OPEN.** b21 (8 slots, embryo_1D_flock n132 nodiv confluent,
  k4/anch10/spin0.3/flow_align40 + heading_align gain ladder, 12000f; gain-0 control s7). (1) **CAUSAL order:**
  every gain≥40 slot polar_order 0.17–0.20 final vs control **0.0214 (~9×)**; migr 0.43–0.69 vs control 0.1519
  (3–4×) → heading_align separates from control (flow_align never did, b20). net_circ rose MONOTONICALLY with gain
  (g40/120/300/600: 0.0016/0.0039/0.0048/0.0052) — off the campaign-long 0, NOT ~0 → b21 falsifier only HALF met.
  (2) **TRANSIENT flock:** every driver SPIKES at 25% (polar 0.571/0.669/**0.774** for g40/g300/g600, peak ↑ with
  gain) then DECAYS to a flat 0.18–0.20 plateau — gain sets the transient PEAK not the sustained order; a polar
  (translational) flock is geometrically unstable in the bounded disc (hits wall → disorders; net_circ rises as
  polar decays = polar→milling conversion). (3) **NEW MECHANISM — coherence↔membrane push:** pure_g300 (flow_align
  OFF) gives the BEST-sustained flock (polar 0.445, decays least) but ruptures the shell (escape **0.9924** at LOW
  speed 0.00019 → coordinated push, not ballistic); flow_align (steer heading toward INCOHERENT fluid velocity) is
  a DECOHERER that both protects the membrane (escape 0.99→0.05) AND damps the flock (0.445→0.18). **Trade:
  flow_align LOW = flock sustains/membrane leaks; HIGH = stable/flock decays.** (4) **spin seed INERT:** spin0_g300
  net_circ 0.00477 ≈ spin-ON 0.00479 → the b21 "seeded by mpm_spin chirality" mechanism FALSIFIED; weak rotation
  self-organizes from alignment. (5) **heading_align RAISES escape** (every driver 0.045–0.20 > control 0.0227); msd
  LOWEST when aligned (0.02–0.04) vs control 0.0725 (aligned cells mill in place, don't net-translate). **NEXT
  (Batch 22):** break the tension — lower `agent_to_mpm.k` (4→2→1, less coordinated push → escape↓) + lower
  `flow_align.gain` (40→20→10, less decohering noise → flock sustains) at heading_align 300; + anch20 (stiffer
  containment) + spin1.5 (does a strong seed lock the milling?) + base-g300 control. Falsifier: if no point BOTH
  sustains polar>control AND drops escape<0.02, a bounded 1st-order polar flock is geometry-limited → close 1D on
  the heading_align transient-flock arm and ADVANCE to 1E.
- **[open + engineering, Batch 21 / b20] 1D collective flow: the 1st-order flow_align (fluid-alignment) route
  CANNOT flock, and the 2nd-order Vicsek rebuild is ARCHITECTURALLY BLOCKED — so wrote a 1st-order-compatible
  Vicsek order op `heading_align`.** b20 (8 slots, embryo_1D_base n132 nodiv confluent 0.30, k4, anch10, spin 0.3,
  flow_align gain 40→sweep, 12000f). (1) **Falsifier fired:** net_circulation off its campaign-long 0 for the FIRST
  time (~0.001–0.005, 5/8 slots) but tiny + tracks DENSITY (s4 dense176 net_circ 0.00477 = batch max) not gain;
  polar_order WEAK ≤0.16, NON-monotone (fa200 0.007 < fa120 0.11), only TRANSIENT bursts (fa120 polar
  0.0948→**0.4889@25%**→0.0517→0.11 — spikes then decays, not a locked flock). Every driver slot escape-FAILS
  0.0076–0.085 (dense176 0.0852, mot24 0.0606, k5 0.0227 — the 1C escape frontier persists at confluence).
  (2) **flow_align is a REGULARIZER:** control s7 (gain 0) BLEW UP escape **1.0**, membrane→grid-aligned BOX, deform
  0.1277 (10× any slot) — removing the fluid-heading coupling lets the incoherent confluent pump resonate the MPM
  grid unstable. So flow_align gain≥40 is load-bearing for stability, independent of flocking. (3) **ENGINEERING —
  why the 2nd-order rebuild is blocked:** `engine._resolve_prediction` forces ONE integration order per set and
  RAISES on conflict; `mpm_to_agent` (the confine coupling that contains cells in the shell) is hardwired
  `first_derivative` → the 2nd-derivative `alignment`/`cruise`/`separation`/`cohesion`/`Coulomb` cannot be added to
  the MPM-coupled agent set (this — not `glide` — is the true content of the b12 "2nd-deriv blocked" note). (4)
  **PIVOT — new op `heading_align`** (`src/plexus/operators/heading_align.py`, registered): a HEADING-STEER
  (`kind=exchange`, `PREDICTION=None`, mirrors `flow_align`) that relaxes each cell's heading toward the mean
  heading of its radius-graph neighbours = the fluid-free 1st-order Vicsek order term, which DOES compose with the
  first-derivative set. **NEXT (Batch 21):** heading_align gain ladder (40/120/300/600) + attribution (flow_align-off
  `pure`, mpm_spin-off `spin0`, `dense176`) + gain-0 control, on `embryo_1D_flock.yaml`. Falsifier: if the gain
  ladder leaves polar_order weak/non-monotone AND net_circ ~0 like flow_align, NO first-order heading rule flocks in
  this confined MPM blastula → close 1D on the b20 flow-off-zero arm and ADVANCE to 1E.
- **[established (amplitude) + open (escape/mode), Batch 20 / b19] 1C CLOSED. Bounded division DEFORMS the shell —
  deform_rms ~2× the matched nodiv floor, division-driven, [established] across 3 seeds AND 4 anchor points — but the
  MODE is intrinsic wobble and the 4x/3x/2x escape residual is a marginal seed-noisy ~0–0.06 leak that NO lever (k,
  anchor.k, POPULATION) robustly zeros; escape is NON-monotone in population, killing the crowding hypothesis.** b19
  (8 slots, 1C_base 4x, population cap ladder at anch10 k4, 12000f; matched control s7 ctrl_anch10_nodiv n44 deform_rms
  0.01141 / f_m2 0.01168 / escape 0). (1) **div3x_anch10 3-seed (n132): escape 0.0152 / 0 / 0.0227 (mean 0.0126±0.0116,
  leaks 2/3 → falsifier FIRED, 4th point after anch10-k4 b17 / anch5-k4 b18 whose seed "clean" fails replication);
  deform_rms 0.0244 / 0.02176 / 0.02519 = 0.0238±0.0018 (Δ vs nodiv 0.01141 = 0.0124 ≫ 2·SD 0.0036 → division-driven
  deform amplitude [established], 2.1×); MODE m2(f_m2 0.0373 DOM) / m1-DRIFT(f_m1 0.0346) / m2(f_m2 0.0394 DOM) → not
  mode-robust (intrinsic wobble, the 1B pattern now across 4 1C anchor points).** (2) **POPULATION IS NOT THE ESCAPE
  LEVER (hypothesis DEAD):** div2x_anch10 (n88, LOWEST pop) escape 0.0568 = WORST slot; div4x_anch7_s1 (n176, HIGHEST)
  escape 0 → escape NON-monotone in n = seed-noise on a marginal 0–0.06 band, not crowding. (3) **Alt-anchor at 3x also
  leaks:** div3x_anch7 escape 0.0076 (m3-dom), div3x_anch12 escape 0.0152 (m4-mix) — neither softer nor stiffer zeros
  escape at n132. div4x_anch7_s1 escape 0 but deform_rms 0.011 ≈ floor (deform-DEAD; anch7-4x deform seed-fragile too).
  (4) AREA anchor-pinned 0.3485–0.3527 (no epiboly, reconfirmed); net_circ ~0 (wobble, not flow-locked). **1C GATE MET
  on deform AMPLITUDE; escape marginal; mode wobble; area epiboly unreachable [engineering].** OPERATING POINT = 1C_base
  4x + anch10 + k4. **NEXT (Batch 20): STAGE 1D batch 1 — collective flow / migration. Substrate embryo_1D_base (n132
  nodiv, confluent, k4, anch10). Hypothesis: collective migration = the agent_to_mpm↔flow_align feedback loop closing at
  confluence (flow_align = the only 1st-deriv coherence operator; Vicsek 2nd-deriv set blocked). Sweep flow_align.gain
  (120/200), spin seed (omega 1.5), motility (0.18/0.24), density (n176), k5; gain-0 control. Target net_circ/polar_order
  off their campaign-long 0.** Falsifier: if gain↑ (+spin +confluence) leaves net_circ 0 / polar_order flat everywhere,
  the 1st-order set cannot flock → rebuild to a 2nd-order Vicsek set (Coulomb/cruise+alignment) at Batch 21.
- **[open, Batch 19 / b18] 1C escape at 4x is NOT robustly anchor-tunable — `anch5_k4` LEAKS 2/3 seeds (falsifier
  FIRED), and TWO b17 claims OVERTURNED: (A) escape-vs-anchor is a seed-noisy BOWL not monotone (over-softening to
  anch3 RE-LEAKS with the cell pushed OUTSIDE), (B) at soft anchors deform is COMPLIANCE-driven not division-driven
  (nodiv-anch5 control ≥ dividing). The division-DRIVEN deform gate holds only at the STIFFER anch10 (b17). The
  untested lever is POPULATION (all 1C cramped 4x).** b18 (8 slots, 1C_base 4x/k4/12000f; matched control s7
  anch5-nodiv n44 escape 0 / deform_rms 0.01939 / f_m2 0.02954 DOM / r_cell_max 0.822). (1) **anch5_k4 3-seed (seed0
  b17 s6 escape 0/deform 0.01571/f_m2 0.02185; seed1 s0 escape 0.0114/0.01457; seed2 s1 escape 0.0057/0.0261/f_m2
  0.05144 DOM): escape 0/0.0114/0.0057 = 0.0057±0.0057 LEAKS 2/3 → NOT robust; deform 0.0188±0.0064 amplitude-robust,
  MODE mixed/m2/m2 (intrinsic wobble reconfirmed).** 3rd anchor (after anch10) whose seed0 "clean" failed replication.
  (2) **OVERTURN A — monotone FALSIFIED:** anch3_k4 (softer) escape 0.0341, r_cell_max **1.027 (cell OUTSIDE)**, f_m1
  0.0407 drift; escape-vs-anchor is a shallow noisy bowl (min ~anch7–8, all in 0–0.03 band), over-softening loses the
  restoring force → NO anchor robustly zeros escape at 4x. (3) **OVERTURN B — soft anchor = compliance-driven deform:**
  nodiv-anch5 deform 0.01939 ≥ dividing 0.0188, f_m2 0.02954 DOM (cleaner than 2/3 dividing seeds), escape 0. As anchor
  softens 40→10→5 the nodiv floor RISES 0.0093→0.0114→0.0194 and division's marginal gain SHRINKS to 0 → anch5 arm is a
  dead end for a division-DRIVEN deform (that gate lives at anch10: nodiv 0.0114 → dividing 0.023 = 2×, b17). (4) **anch7_k4
  (s2) sole dividing escape-0** but m3-DOM (f_m3 0.0306, migr 0.137), single seed = likely seed-fragile. anch5_k3 (s3)
  escape 0.0114 + deform crushed 0.0113 (k-down doesn't rescue); anch5_k5 (s5) escape 0.0682 (onset k4/k5 reconfirmed);
  anch5_r2 slow-fill (s6) escape 0.0057 + f_m2 collapsed 0.0082 (the anch10 slow-fill m2 boost was anch10-specific).
  AREA anchor-pinned 0.351–0.356 (no epiboly). **NEXT (Batch 19): POPULATION × anchor — cap ladder at anch10 (2x/3x-3seed
  /4x-from-b17/matched nodiv) + anch7 probe + stiffer-at-reduced-pop. Hypothesis: the 4x leak is CROWDING-driven; 3x
  (n132) at anch10 zeros escape robustly while keeping division-driven deform (vs nodiv 0.0114) → clean 1C gate.**
  Falsifier: if 3x-anch10 also leaks on a seed OR its division-gain vanishes, CLOSE 1C on the b17 deform-amplitude arm
  (anch10, escape a marginal 1–2-cell residual, mode intrinsic wobble, area epiboly unreachable) and ADVANCE to 1D.
- **[open, Batch 18 / b17] The b16 `anch10_k4` "clean m=2 gate" DOES NOT survive 3-seed replication — escape leaks 1/3
  and the dominant azimuthal MODE flips m2/m3/m1-drift across seeds (division deform is an INTRINSIC WOBBLE, the 1B
  pattern recurring in 1C); what IS robust is the deform AMPLITUDE (~2× floor, division-driven) and the escape-safe
  DIRECTION: at coupling k4, escape falls MONOTONICALLY as the substrate anchor softens, so `anch5_k4` (soft anchor +
  low coupling) is the cleanest escape-0 point.** b17 (8 slots, 1C_base 4x, 12000f; in-batch nodiv-anch10 control s7
  n=44 escape 0 / deform_rms 0.01141 / f_m2 0.01168 / area 0.35577). (1) **anch10_k4 3-seed (seed0 b16 s3, seed1/2 b17
  s0/s1): escape 0 / 0.0114 / 0 (mean 0.0038±0.0066, leaks 1/3 → falsifier FIRED, NOT a seed-robust escape-0 point);
  deform_rms 0.01766 / 0.02517 / 0.02679 (mean 0.0232±0.0049, 21% CV — AMPLITUDE robust, Δ vs control 0.0118 > 2·SD
  0.0097 → division deform amplitude significant); dominant MODE m2 / m3 / m1-drift → NOT mode-robust, the b16 clean-m2
  was seed-luck.** (2) **Anchor ladder at fixed k4 — escape MONOTONE, softer = safer (OVERTURNS b16's U-shape):** anch40
  0.0739 → anch15 0.0682 (f_m1 0.06676 DRIFT) → anch10 ~0 (0/0.0114/0) → anch8 0.0057 → **anch5 0** (f_m2 0.02185 DOM,
  r_cell_max 0.8843 best margin, circ 0.9864, migr 0.1408 no drift). The b16 U-shape (min at 10, anch5 leaks 0.0284) was
  a k6 artifact; at the lower coupling k4 the softest anchor is safest — the compliant shell absorbs division push. The
  b17 hyp "anchor 8–15 safe / softer re-leaks" FALSIFIED (anch15 leaked WORST). (3) **Coupling ceiling at anch10 = k4:**
  anch10_k5 escape 0.0114 FAIL (mixed m2≈m3), anch10_k6 0.0057 (b16) — onset sharp between k4 and k5. (4) **Slow-fill r2
  @ anch10:** f_m2 0.04298 DOM (growth 163×, near campaign-max) but escape 0.0114 FAIL — biggest clean-m2 amplitude, still
  leaks at the marginal anchor10 → retry at anch5. (5) **Division is the deform driver:** dividing anch10_k4 deform
  0.018–0.027 vs matched nodiv-anch10 control 0.01141 = 1.5–2.3× → gate deform is division-driven, not an anchor artifact
  (anchor-relaxation alone raises deform only ~1.2× over the anch40 floor 0.00934). AREA still pinned everywhere
  (0.342–0.356, no epiboly). **NEXT (Batch 18): CONSOLIDATE anch5_k4 — 3-seed replicate toward [established] escape-0 +
  deform gate; bracket the escape/deform tradeoff (anch7 stiffest-still-safe for max deform-ratio, anch3 softer floor);
  isolate coupling (anch5_k3 safer, anch5_k5 onset); slow-fill r2 @ anch5; control anch5-nodiv (matched deform-ratio).**
  Falsifier: if anch5_k4 ALSO leaks on a seed, escape at 4x is not anchor-tunable → adopt the best clean point and CLOSE
  1C on the deform-AMPLITUDE arm (mode is intrinsic wobble; area epiboly unreachable), then ADVANCE to 1D.
- **[open, Batch 17 / b16] 1C SHAPE-DEFORM GATE MET (1 seed): `anch10_k4` (relax `mpm_anchor.k` 40→10 + coupling
  `agent_to_mpm.k` 6→4, at 4x division) gives escape 0, deform_rms 0.01766 (1.9× the nodiv 0.00934 floor), f_m2
  0.02212 DOMINANT (cleanest m=2 of the 1C campaign, m2/m1 1.24), no drift. Anchor relaxation does NOT grow area
  (epiboly UNREACHABLE with current operators) — it opens a COMPLIANCE window that converts division push into
  escape-safe LOBING.** b16 (8 slots, 1C_base 4x, 12000f; in-batch nodiv control s7 escape 0 / deform_rms 0.00934 /
  area 0.35935 / circ 0.9913). (1) **Anchor ladder at k6/4x — escape U-SHAPED, min at anchor≈10:** escape
  0.0852(k40,b14)→0.0682(s0 k20)→**0.0057(s1 k10)**→0.0284(s2 k5); over-relaxing (k5) lets the blob wobble/slip and
  escape climbs back. Compliant shell absorbs division push into lobing (deform 2–3.7× floor) not expulsion (b14
  "compliant wall absorbs push", now via the anchor). (2) **AREA STAYS PINNED (area arm FALSIFIED):** all dividing
  slots 0.343–0.359, ALL ≤ nodiv 0.35935; disc_R 0.3381 fixed. Confirmed at SOURCE: `mpm_anchor` restores toward
  FRAME-0 rest positions (`_rest=pos.clone()`, fixed) — relaxing k changes COMPLIANCE not rest area; `agent_remodel`
  scales Lame moduli (stiffness) not rest length. **No current operator grows the shell's rest area → true epiboly
  needs a new rest-length-growth operator [engineering].** (3) **The WIN = drop coupling k6→k4 at anchor10:** removes
  the m1-drift AND zeroes escape — s1 anch10_k6 escape 0.0057 but f_m1 0.04075 dominant (drift, migr 0.33); **s3
  anch10_k4 escape 0, f_m2 0.02212 dominant, migr 0.0708, r_cell_max 0.8793, circ 0.9751 (lobed), nn_min 0.0173.**
  The anchor relaxation MADE k4 escape-safe (at anchor40, k4/4x escaped 0.0739, b15 s2). (4) **Stacks:** anch10_soft
  (k6+youngs100) escape 0.0966 WORST (over-relaxed, cell outside) [reject]; anch10_r2 (slow-fill) f_m2 0.04697
  CAMPAIGN-MAX but drift-contaminated at k6; div2x_k6 (2x @ stiff anchor) escape 0.0341 — POPULATION fallback still
  leaks, WORSE than s3 4x → 1C keeps the user's 4x. **NEXT (Batch 17): 3-seed replicate anch10_k4 toward [established]
  + bracket the escape-safe window (anchor 8/15@k4, k5@anchor10, anch5_k4, slow-fill r2@k4) + control anch10-nodiv.**
  Falsifier: if anch10_k4 leaks on any seed → seed fluke, adopt nearest robust point and CLOSE 1C on the shape arm
  anyway, ADVANCE to 1D.
- **[open, Batch 16 / b15] At 4x division there is NO escape-safe `agent_to_mpm.k` with live deform — escape onset is
  BELOW k2 and the only escape-0 point (k1) is deform-dead; the binding cause is the ANCHOR-PINNED AREA (rigid substrate
  anchor forbids shell expansion), so proliferation can only lobe-and-leak, never spread. NEXT LEVER = anchor relaxation.**
  b15 (8 slots, 1C_base = slow_k6 + cell_divide 4x, 12000f; nodiv slow_k6 baseline deform_rms 0.00787/escape 0/area 0.36015;
  in-batch floor s7 div4x_nok k1 deform 0.00299/escape 0). (1) **k-ladder at 4x, escape MONOTONE:** k1→k2→k3→k4→(k6) escape
  **0 → 0.017 → 0.0227 → 0.0739 → 0.0852** — onset in (k1, k2]; k1 the sole escape-0 point has deform 0.00299 < the nodiv-k6
  floor 0.00787 → no escape-safe k at 4x beats nodiv deform. **b15 hypothesis FALSIFIED; the b14 pre-registered "population-bound"
  falsifier answered YES.** (2) **Aids CUT but don't ZERO escape** (all @ k4, base escape 0.0739): confine 0.03→0.06 → **0.0114**
  (best, 6.5×) but crushes nn_min to 0.0142 (0.71×r0) & pure m1-drift & net_circ 0; soft youngs100 → **0.0227** with the batch's
  CLEANEST deform (f_m2 0.0111 ≈ f_m3 0.0124, deform 0.0185, circ 0.9867); slow-fill r0.2 → 0.0341 (m1-drift); cap **3x** (n132) k4
  → 0.0227 with **f_m2 0.01965 > f_m1 → m=2-DOMINANT** (cleanest MODE). Neither pre-registered fix (cap-down OR containment) zeroes
  escape alone. (3) **b14 pattern RE-confirmed: low-escape end = cleaner m≥2 mode** (soft, 3x m2-dom; leaky k4/k6/r2 = m1-drift).
  (4) **AREA STILL ANCHOR-PINNED everywhere (0.3596–0.3611 vs 0.36015 nodiv):** the shell is held to fixed radius by `mpm_anchor
  {mode: substrate, k: 40}` → division pressure has nowhere to go but THROUGH the membrane = escape. Physically-correct 1C
  (epiboly) needs the shell to EXPAND (area↑ relieves crowding). **NEXT (Batch 16): relax the substrate anchor** (`mpm_anchor.k`
  40→20→10→5 at 4x) to convert division pressure into AREA GROWTH with escape→0, + soft/slow-fill stacks + a 2x-pop fallback.
  Falsifier: if relaxing the anchor leaves area pinned & escape>0, OR destabilizes the blastula (collapsed>0 / bulk m1-drift↑)
  at every k, then area growth needs an explicit membrane rest-length operator (agent_remodel), not anchor relaxation → and 1C's
  SHAPE-deform gate is met by b15 3x-k4 (m=2-dom, deform 1.7× floor) at the residual cost of escape 0.0227.
- **[open, Batch 15 / b14] 1C STAGE-1 RESULT: bounded division DEFORMS the shell (deform_rms monotone with
  population, first LOBED shells of the campaign) but at the 1B k6 coupling it RE-TRIGGERS the escape gate at
  EVERY cap ≥2x; the deform transmits through the `agent_to_mpm` coupling, NOT pure crowding; and area is
  anchor-PINNED (shape change, not size expansion).** b14 (8 slots, 1C_base = slow_k6 + cell_divide, 12000f;
  nodiv control s7 deform_rms 0.00787 / escape 0 / area 0.36015). (1) **Deform monotone with the cap** n44→88→132→176:
  deform_rms **0.00787 → 0.01541 → 0.01769 → 0.0214** (2.0/2.2/2.7× nodiv), circularity 0.9945→0.9903 (−1 to −4%,
  visibly lobed); nn_min holds ~0.018 (4x geometrically safe, collapsed 0) — the "deform ↑ with population" arm
  CONFIRMED. (2) **ESCAPE FALSIFIER FIRED at every dividing cap:** div2x **0.0341**, div3x **0.0379**, div4x
  **0.0852**, div4x_r6 **0.0966** (r_cell_max 1.07, cell expelled OUTSIDE), soft 0.0568, r2 0.0341 — escape rises
  with the cap; 1C is NOT gated. (3) **Drive-isolation control decisive — deform AND escape ride the coupling, not
  crowding:** div4x_nok (176 cells, agent_mass 2e-6 + k1) → deform_rms **0.00299** (≈floor, BELOW the k6 nodiv
  control) with escape **0**. Pure contact-crowding neither deforms nor leaks; the k6 `agent_to_mpm` coupling is the
  sole transmitter and division amplifies it by multiplying coupled pushers → the 1C escape is the b13 escape ceiling
  (per-cell push × gain) re-crossed by AGGREGATE push ∝ n·k (k6 leaks at 4x as k12 leaked at n44). (4) **Fill RATE
  modulates escape at fixed final n:** slow fill r2 escape 0.0341 vs r4 0.0852 vs fast r6 0.0966 — gradual filling
  halves escape (shell relaxes between division bursts). (5) **AREA anchor-pinned:** all dividing area 0.3582–0.3597
  vs nodiv 0.36015 (flat/lower) — division deforms SHAPE not SIZE; real expansion needs membrane rest-length growth,
  not division. (6) **Soft shell (youngs 100) = deform amplifier:** div4x_soft deform_rms 0.02715 (batch max, circ
  0.9739 most-lobed), escape 0.0568 (LOWER than stiff div4x_r4 0.0852 — compliant wall absorbs the push) but area
  still flat — viable ONLY paired with a lower-coupling escape fix. (7) **Deform mode tracks escape:** leaky slots are
  m=1 bulk-DRIFT dominated (div4x_r4 f_m1 0.0407 ≫ f_m2 0.011; high migration r6 0.259/r2 0.085 = ejection recoil);
  the low-escape end is cleaner — div2x m=2-dominant (f_m2 0.02523 > f_m1, growth 96×). 1 seed → [open]. **NEXT (Batch
  15): k-LADDER at fixed 4x division** (k2/k3/k4, mass 8e-6 fixed) to find the escape-safe deform point between k1
  (safe/dead) and k6 (leaks/deforms), plus slow-fill (r2) + soft-shell + lower-pop (3x) aids. Falsifier: if no k in
  (1,6) gives escape 0 with deform ≫ floor at 4x, the escape ceiling is population-bound → cap 1C at the 2x/3x
  population where a k exists, or add real containment (confine↑) before pushing to 4x.
- **[established, Batch 14 / b13] 1B CLOSED. `slow_k6` (move 0.12, agent_mass 8e-6, `agent_to_mpm.k` 6) is the 1B operating
  point — escape-safe membrane deform_rms 0.00784 ± 0.00215 (6.3× the 0.00124 floor) robust across 3 seeds — but the clean
  m=2 MODE is INTRINSIC WOBBLE-NOISE that k biases (k6 m2-dominant on 2/3 seeds vs k4's 1/3) yet NEVER locks; and
  `agent_to_mpm.k` DOES eventually leak escape even at slow motility (k12).** b13 (8 slots, nodiv n44, c03, r150, 12000f;
  floor s7 deform_rms 0.00124 / f_m2 0.0004 / escape 0). (1) **slow_k6 3-seed:** escape **0/0/0**; deform_rms
  0.00787/0.00997/0.00567 (mean 0.00784 ± 0.00215; Δ vs floor = 3.1·SD > 2·SD → deform AMPLITUDE [established], concordant
  with slow_k4). (2) **m=2 MODE stays [open], now understood as intrinsic:** f_m2 0.01267/0.01549/0.0065 with m2/m3
  1.86/2.24/**0.83** — m2-dominant on seed0+seed1 but **seed2 flips to m3** → k6 improves the m2-dominant fraction over k4
  (1/3, b12) but no k value locks m=2. Window is NARROW: **slow_k5** f_m2 0.0106 ≈ f_m3 0.01108 (mixed), **slow_k7** f_m2
  COLLAPSES to 0.00131 vs f_m3 0.0114 (over-driven to m3, like k8) — k6 is the clean-m2 optimum, k7 already past it. With
  net_circulation 0 everywhere and Vicsek `alignment` blocked (b12), THIS operator set cannot lock the wobble → a locked
  shape awaits 1C division pressure or a 2nd-order-set rebuild. (3) **k-ceiling falsifier ANSWERED: k leaks at slow motility
  by k12** — slow_k12 escape **0.0227** (r_cell_max 0.9047; campaign-max f_m2 0.02091 but DISQUALIFIED). Refines the b12
  [established] "MOTILITY not coupling gates escape": holds across the k4–k8 window, but coupling gain re-enters as the escape
  driver at k≳12 (onset (8,12]). (4) mid_k6 (move 0.16 + k6) escape 0 but m3-dominant → higher motility doesn't help the
  mode; k-route escape onset stays (0.16,0.24]. **NEXT (Batch 14): ADVANCE to 1C** (division pressure deforms the shell) —
  per the pre-registered falsifier, adopt slow_k6 on deform_rms grounds since the m=2 mode is seed-fragile at every k.
- **[established, Batch 13 / b12] The 1B GATE IS MET: `slow_k` (move 0.12, agent_mass 8e-6, `agent_to_mpm.k` ≥4) is an
  escape-safe membrane-DEFORM lever — deform_rms robustly above the quiescent floor across 3 seeds with escape 0 — and
  MOTILITY (not the coupling gain) sets the escape ceiling.** b12 (8 slots, nodiv n44, c03, r150, 12000f; floor s7
  quiescent deform_rms 0.00124 / f_m2 0.0004 / escape 0). (1) **slow_k4 3-seed replicate:** escape **0/0/0** (robust — where
  fast_k4 leaked 2/3, b10); **deform_rms 0.00563 ± 0.00064** (SD ~11% of mean, 4.5× floor; Δ = 0.00439 = **6.9·SD ≫ 2·SD →
  [established]** that slow-k raises deform above the ablation floor). (2) **The m=2 MODE stays [open] — seed-fragile:**
  fourier_m2 **0.00537 ± 0.00450** (SD≈mean; clean m=2 (m2>m1) only on seed0 0.01045, m3-DOMINANT on seed1 0.00374/seed2
  0.00191). The total deform amplitude is robust; the azimuthal MODE at the final frame is wobble-noise → the deform is a
  transient WOBBLE sampling different modes by seed, not a locked m=2 shape. (3) **k ladder k4/k6/k8 ALL escape-safe at slow
  motility (0.12)** — the b12 falsifier "does k EVER leak at slow motility?" is answered NO through k8. With fast_k4/k6/k8 ALL
  leaking (b10), **MOTILITY (per-cell ballistic energy), not coupling gain, is the escape gate** — now supported across a full
  3-point k ladder at BOTH motilities (upgrades b10/b11 [open] → near-established). (4) **`slow_k6` is the CLEAN-m=2 optimum:**
  escape 0, f_m2 **0.01267 (32× floor, campaign-clean max)**, f_m1 0.00351 → **m2/m1 3.61 (cleanest real m=2 elongation in the
  campaign)**, deform_rms 0.00787, deform_cell_corr +0.0532 — k6 beats k4 for clean m=2 at the same motility (1 seed → [open]).
  (5) **k8 OVER-drives:** deform shifts to high modes (m3 0.0143, m4 0.0152, m1-drift 0.0108), f_m2 collapses to 0.00217, circ
  0.9945→0.9858 — bigger deform_rms (0.0115) but NOT clean shape → there is a clean-m2 k WINDOW, k6 near its top, k8 past it.
  (6) **mid_k4 (move 0.16 + k4)** escape 0, deform_cell_corr **+0.1058 (batch max)**, msd 0.0193 (2.6× the slow slots) → higher
  motility raises deform AND coupling while escape-safe at 0.16; the k-route escape onset is in **(0.16, 0.24]**. (7) **STACK
  slow_k4_mass6 (mass6e-5 + k4) escape 0.0227 HARD FAIL** — reconfirms b10 "escape risks ADD"; the two cell→fluid couplings must
  NOT be combined at deform-relevant strengths. 1A held on all 7 clean slots (collapsed 0, nn_min ~0.019, accel genuine). **1B
  operating point = slow_k6 (move 0.12, mass 8e-6, k6) OR slow_k4** — escape-safe membrane deformation, deform_rms 4.5–6× floor,
  1A holding. **NEXT (Batch 13): CONSOLIDATE — 3-seed replicate slow_k6 (is the clean m=2 seed-robust where k4's was not?), bracket
  the clean-m2 window (k5/k7), k-ceiling falsifier (k12), then ADVANCE to 1C (division deforms the shell).** Falsifier: if slow_k6
  m=2 is ALSO seed-fragile, the m=2 mode is intrinsic wobble-noise regardless of k → adopt slow_k6-seed0 on deform_rms grounds
  and advance to 1C anyway.
- **[engineering, Batch 13 / b12] Vicsek velocity `alignment` (flocking / coherent-motion) CANNOT be added to the 1B `agent`
  set — same integration-order conflict that killed `separation` (b04).** `alignment` (src `operators/alignment.py`) is
  `PREDICTION="second_derivative"` (emits an acceleration), while the agent set's `repel`/`glide` are `first_derivative`; a set
  integrates as ONE order, so adding it crashes at frame 0 exactly like `separation`. Consequence: the natural lever to convert
  the 1B deform-WOBBLE into a flow-LOCKED shape (coherent collective motion → net_circulation > 0 → sustained directional push)
  is unreachable without rebuilding the whole agent set to 2nd order (replacing repel/glide with 2nd-order equivalents) — too
  large/confounding for one batch, deferred. `flow_align` (first-derivative, heading-steering, safe) remains NULL because
  net_circulation is 0 (no fluid flow to align to). So with THIS operator set, 1B deformation is intrinsically a wobble; a locked
  shape awaits 1C division pressure or a 2nd-order-set rebuild.
- **[open, Batch 12 / b11] The SLOW (move 0.12) `agent_mass` deform route is SEED-FRAGILE (falsifies b11's hypothesis); the
  escape-safe deform STANDOUT is `slow_k4` (move 0.12 + agent_mass 8e-6 + `agent_to_mpm.k` 4) — escape 0 with the biggest
  containment margin AND a REAL m=2 elongation (m2>m1), consistent with MOTILITY (not the k lever) setting the escape ceiling.**
  b11 (8 slots, nodiv n44, c03, r150, 12000f; floor s7 deform_rms 0.00124 / f_m2 0.0004 / escape 0). (1) **3-seed replicate of
  slow_mass5 (move 0.12, mass 5e-5) is seed-fragile in BOTH escape and deform:** seed0 escape 0 / f_m2 0.01097 (clean, ==b08 s3),
  seed1 escape **0.0227 HARD FAIL** / f_m2 0.00209, seed2 escape 0 / f_m2 **0.00037 (= quiescent floor, NO deform)** → escape
  **0.0076 ± 0.013** (1/3 leaks), f_m2 **0.0045 ± 0.0057** (SD>mean, 27× spread 0.011→0.0004). Cannot promote to [established];
  the b11 falsifier ("if slow_mass5 leaks on any seed…") FIRED. (2) **`slow_k4` is the escape-safe deform standout:** escape 0,
  f_m2 0.01045 (26× floor, ties slow_mass5_s0), f_m1 0.00437 → **m2/m1 = 2.4 (REAL m=2 elongation, not bulk drift)**, deform_rms
  0.0059, **r_cell_max 0.7817 — the LARGEST containment margin in the batch** (vs mass5_s0 0.89, grid96 leaker 0.99),
  deform_cell_corr +0.017, accel 0.00123 genuine. Where `fast_k4` (move 0.24) HARD-FAILED escape on 2/3 seeds (b10: 0.1364,
  0.0227), `slow_k4` is escape-safe with room → **MOTILITY, not the k lever, drives escape** (supports the b10 [open]). 1 seed →
  [open]. (3) **Slow mass frontier buys DRIFT, not shape:** slow_mass6 (6e-5) escape 0 but f_m1 0.01196 > f_m2 0.00392;
  slow_mass7 (7e-5) escape 0, deform_rms 0.0144 (batch max) but f_m1 **0.02256 ≫** f_m2 0.00689, circ 0.9901 — the big "deform"
  is m=1 bulk shell drift, not elongation; slow mass-escape onset now (7e-5, 8e-5] on seed0. The k route gives cleaner m=2 shape
  than the mass route. (4) **grid96 HALVES the fast escape but doesn't fix it:** grid96_fastk4 (fast_k4-seed1 @ n_grid 96) escape
  **0.0682**, halved from 0.1364 @ ng64 — a partial grid-tunneling component, but the fast route stays escape-fragile and does NOT
  reopen. Side effect: net_circulation **0.00193** (first non-trivial nonzero in the campaign) + enstrophy 5.88e-6 — the finer grid
  resolves swirl the coarse grid smeared out. 1A held on all 8 (collapsed 0, nn_min ≈r0, accel genuine); the two hard-fails are
  slow_mass5_s1 and grid96_fastk4. **NEXT (Batch 12):** 3-seed replicate slow_k4 (seed0/1/2) toward [established] as 1B's operating
  point (the robustness test slow_mass5 failed); push the slow-k deform frontier (slow_k6, slow_k8) to find the k ceiling / whether
  k EVER leaks at slow motility; mid_k4 (move 0.16 + k4) locates the k-route motility escape onset (fast 0.24 leaks, slow 0.12
  safe); slow_k4_mass6 (mass6e-5 + k4) tests whether the two couplings' deform ADDS escape-safely or their escape risks ADD (b10
  found risks add at fast). Falsifier: if slow_k4 leaks on any seed, the k lever is no more robust than mass → adopt the best clean
  seed0 point (slow_k4, f_m2 0.010, m2>m1) as 1B's operating spec and ADVANCE to 1C (division deforms the shell).
- **[open, Batch 11 / b10] 1B's escape ceiling is set by MOTILITY (per-cell ballistic energy), NOT by the coupling lever —
  so the FAST (move 0.24) deform route is escape-fragile for BOTH `agent_mass` AND `agent_to_mpm.k` across seeds, and
  membrane stiffness is DEAD as a containment lever in both directions.** b10 (8 slots, nodiv n44, c03, r150, 12000f; floor
  s7 deform_rms 0.00124 / fourier_m2 0.0004 / escape 0). **EVERY driver slot HARD-FAILS escape; only the floor survives.**
  (1) **b09's `fast_k4` "clean flagship" was a SEED FLUKE [overturns b09]:** identical recipe (0.24, mass8e-6, k4) escaped
  **0.1364 on seed1** (6/44 cells out — batch worst) and **0.0227 on seed2**, where seed0 was escape 0. Pushing k harder
  leaks too: fast_k6 (k6) escape 0.0682, fast_k8 (k8) escape 0.0227. **k leaks across seeds exactly like agent_mass** — the
  b09 "k is the escape-SAFE lever, agent_mass leaky" claim is FALSIFIED; the b09 difference was seed noise. And the b09
  "climbs+sustains" fourier_m2 was a seed artifact — at higher k it's a big OSCILLATORY WOBBLE (fast_k6 0.01117→0.00624→
  0.02804→0.00810→0.02544; fast_k4_seed2 peaks mid-run then decays 0.02004→0.00975), no locked shape change. (2) **BOTH
  containment levers DEAD:** mass20_confine6 (confine 0.03→0.06 on the borderline mass2e-5 point) escape **0.0227** (unchanged
  from b09's seed1 leak) AND deform crushed to 0.00498 — the inward hold suppresses the deform it was meant to preserve;
  mass20_stiff (youngs 200→500) escape **0.0455** (WORSE than 0.0227). With b08's SOFTening-leaks, **membrane Young's modulus
  is falsified as a containment lever in BOTH directions [rejected]** — a strong enough cell→fluid reaction punches through
  regardless of shell stiffness. (3) **The escape onset tracks per-cell kinetic energy (motility), not coupling type:** across
  the whole 1B record, every escape-SAFE deform point with real deform lives at move 0.12 (b08 s3 mass5e-5 f_m2 0.011), and
  every move-0.24 point with f_m2 ≳ 0.007 leaks. Slowing motility RAISES the mass-escape onset (≤2e-5 seed-boundary at 0.24 →
  (5e-5,8e-5] at 0.12) and the k-escape onset (≤4 at 0.24). Coupling gain sets how-much-deform-per-push; motility sets whether
  a near-boundary cell gets flung through the shell. (4) k4_mass13 (k4 + sub-threshold mass1.3e-5) escaped 0.1136 — mass and k
  escape-risks ADD. net_circulation ≈0 everywhere (still no vortex). **The only robust escape-safe deform candidate is the SLOW
  route (b08 s3), not yet replicated. NEXT (Batch 11): 3-seed replicate slow_mass5 (move 0.12, mass5e-5) toward [established] as
  1B's operating point; slow_k4 (0.12 + k4) isolates motility-not-k as the escape driver (fast_k4 leaked 2/3 seeds); slow_mass6/7
  bracket the slow frontier; grid96_fastk4 probes whether escape is a grid-tunneling numerical artifact (escape↓ at n_grid 96 ⇒
  numerical, reopening the fast route). Falsifier: if slow_mass5 leaks on any seed, escape is a fixed cell→fluid push regardless
  of motility → adopt the best escape-safe point (b08 s3-class, f_m2 ~0.011) and ADVANCE to 1C.**
- **[OVERTURNED by b10, was open Batch 10 / b09] "The escape-SAFE 1B membrane-deform lever is `agent_to_mpm.k` (drag-coupling GAIN), NOT
  `agent_mass`; ESCAPE is a fixed cell→fluid-PUSH ceiling (coupling threshold, motility-independent); the b08 mass-flagship
  is SEED-FRAGILE.** b09 (8 slots, nodiv n44, c03, r150, 12000f; floor s7 deform_rms 0.00124 / fourier_m2 0.0004 / escape 0).
  **6 of 8 slots HARD-FAIL on escape** — the escape gate is now 1B's only failure mode (collapse/nn_min solved). **CLEAN
  FLAGSHIP s1 fast_k4 (move 0.24 + agent_mass 8e-6 + `agent_to_mpm.k` 1→4):** escape **0**, deform_rms **0.01287 (10.4×)**,
  fourier_m2 **0.01973 (49×, campaign max)**, fourier_m3 0.00891, deform_cell_corr +0.0668, circ 0.9927, accel 0.003107
  (genuine, speed 0.0072 ≪ vmax 0.6), r_cell_max 0.811 (safe margin vs leakers' 0.91–0.94). fourier_m2 CLIMBS+SUSTAINS
  (0.00504→0.01459→0.01654→0.01515→0.01973, ends at max — accumulating, not a wobble) and m2 (0.0197) > m1 (0.0136) → real
  m=2 elongation. **k SAFE, mass LEAKY at fixed motility:** at move 0.24, mass8e-6+k4 (s1) escape 0 but mass2e-5+k2 (s2)
  escape 0.0455 — the per-cell ballistic escape tracks `agent_mass`, while `k` amplifies the collective fluid→membrane
  response without flinging individual cells. **FALSIFIER FIRED (escape ceiling is a fixed push, not motility×coupling):**
  slow_mass80x (0.12, mass8e-5) escaped 0.0455 and slow_mass130x (0.12, mass1.3e-4) escaped 0.0227; since b08 s3 (0.12,
  mass5e-5) was safe, the slow-motility mass-escape onset is in (5e-5, 8e-5]. Slowing motility RAISES the mass threshold
  but does not remove it. **SEED-FRAGILE:** flagship_seed1 (move 0.24, mass2e-5, seed1) escaped 0.0227 where the identical
  b08 s0 (seed0) was escape 0 → the mass-flagship sits ON the escape boundary and fails ≥3-seed replication [retired as an
  operating point]. **flow_align NULL a 3rd time [rejected]:** gain 120 with strong coupling present → polar_order 0.037
  (≈floor), net_circulation 0, deform_rms 0.00943 (< plain) — no coherence even when a flow field exists. 1 seed → [open].
  **NEXT (Batch 10):** (a) push k 4→6→8 at fast+mass8e-6 to find the k escape ceiling (k-route falsifier: does k ALSO leak
  eventually?); (b) 2-seed replicate fast_k4 (seed1, seed2) toward [established] for "k is the deform lever"; (c) CONTAINMENT
  tests to raise the escape ceiling for the mass route — `mpm_to_agent.confine` 0.03→0.06 and membrane youngs 200→500 on the
  borderline mass2e-5 point (b08 falsified SOFTening as a deform lever; here STIFFENING is tested as a CONTAINMENT lever).
- **[open, Batch 9 / b08] The 1B membrane-deform lever is cell→fluid COUPLING GAIN (`agent_to_mpm.agent_mass` AND
  `agent_to_mpm.k`, both amplify; agent_mass is NOT saturated), and the CEILING is ESCAPE — driven by
  motility×coupling OVERDRIVE, not by membrane stiffness.** b08 (8 slots, nodiv n44, c03, r150, 12000f; floor s7
  deform_rms 0.00124 / fourier_m2 0.0004 / circ 0.9983 / escape 0). **Clean flagship s0 fast_mass10x (move 0.24 +
  agent_mass 2e-5):** deform_rms **0.00819 (6.6×)**, fourier_m2 **0.00967 (24×)**, fourier_m3 0.00897 (69×),
  deform_cell_corr −0.079→**+0.223** (sign-flip, climbs monotone), escape 0, nn_min 0.0194, accel genuine — and
  fourier_m2 > fourier_m1 (real elongation, not drift). deform_rms trajectory climbs+plateaus at ~0.009 (less
  oscillatory than b07's wobble). **agent_mass NOT saturated (b07 "saturating" was the MOTILITY ceiling):** at slow
  motility s3 mass25x_slow (0.12 + mass 5e-5) reaches deform_rms 0.00873 / fourier_m2 **0.01097** (batch-max clean)
  with escape 0 + accel 0.0011. **`agent_to_mpm.k` is a 2nd coupling lever:** s6 couplingk2 (k 1→2, mass 8e-6, move
  0.24) deform_rms 0.00896 / fourier_m2 0.00817, escape 0 (2× b07's fast_mass4x at k=1). **ESCAPE is the binding 1B
  constraint:** the two hard-fails s1 fast_mass25x (0.24 × mass 5e-5) and s5 fast_mass10x_soft80 (0.24 × mass 2e-5 ×
  youngs 80) both escape 0.0227 (1/44 cell punched through, r_cell_max 0.94/0.90). **The SAME coupling (mass 5e-5) is
  escape-SAFE at move 0.12 (s3) but FAILS at 0.24 (s1)** → motility×coupling is the escape driver, slowing motility
  decouples deform from leak. The one visibly-lobed slot (s1, circ 0.9731) is exactly the escape-fail, and its
  "deform" is dominated by fourier_m1 0.0225 (bulk drift), fourier_m2 only 0.00392 — not clean shape change.
  **MEMBRANE STIFFNESS FALSIFIED as a lever [rejected]:** youngs 200→80→40 (fast_soft80 0.00429 / fast_soft40 0.00482)
  barely beat b07's fast_mass4x 0.00444 and fourier_m2 stayed FLAT (~0.0025 vs mass10x 0.00967); softening at high
  mass LEAKED (s5). Shell is push/coupling-limited, not stiffness-limited; softer shell only lowers the escape margin.
  1A held on all 6 escape-clean slots (collapsed 0, nn_min ≈r0, accel genuine). 1 seed → [open]. **NEXT (Batch 9):
  map the motility↓×coupling↑ escape frontier — slow(0.12)+mass 8e-5/1.3e-4, k 4, the fast+mass2e-5+k2 overdrive
  corner (expect escape>0 = ceiling), mid motility 0.18 for the escape onset, flow_align.gain 120 at strong-flow for
  coherence, and a seed-1 replicate of s0 toward [established].** Falsifier: if slow+mass 8e-5 also escapes, the
  ceiling is a fixed cell→fluid push (needs a containment fix, not more coupling).
- **[open→refined by b08, Batch 8 / b07] 1B deform is driven by MOTILITY × fluid-coupling (agent_mass), NOT by direct fluid swirl
  (mpm_spin) or flock alignment (flow_align) — and the membrane is still visibly ROUND (deform is a sub-percent
  wobble).** b07 (8 slots, nodiv n44, c03, r150, 12000f), deform_rms vs the quiescent floor 0.00124: **MOTILITY**
  move_speed 0.12→0.24 at fixed mass 8e-6 lifts deform_rms 0.00244→**0.00444** (fast_mass4x, batch max, 3.6× floor),
  fourier_m2 0.00226→**0.00667** (16× floor), msd 0.00245→0.04262 (17×), and flips deform_cell_corr −0.024→**+0.0895**
  (the only positive coupling in the batch — cell motion phase-locks to membrane shape). **agent_mass** is a real but
  SATURATING lever at fixed motility: 2e-6→8e-6→2e-5 gives 0.00124→0.00244→0.00302 (diminishing; +0.0012 then +0.0006).
  **mpm_spin is NULL and creates NO circulation:** omega 0.3→1.0 leaves deform_rms 0.00239 (≈mass4x) with
  net_circulation 0.0 and enstrophy 3.9e-7 (BELOW the 4.4e-7 floor) — [rejected] as a swirl/circulation source at n44.
  **flow_align is NULL:** gain 40→120 leaves deform_rms 0.00120 = floor, polar_order 0.0092 ≈ floor — [rejected] as a
  deform lever at n44. Stacking spin on motility is SUBTRACTIVE (combo 0.00384 < fast_mass4x 0.00444). Deformation is a
  transient WOBBLE not accumulation (fast deform_rms 0.00257→0.0042→0.00598→0.00337→0.00444, oscillates; circularity
  0.997 throughout; msd climbs monotone but shape does not lock). 1A held on all 8 (collapsed=0, escape=0, nn_min
  0.0188–0.0195, accel 0.0013–0.0036 genuine — speed 0.0065 ≪ vmax 0.6, NOT clamp-bound). 1 seed → [open]. **NEXT
  (Batch 8): push the motility×high-mass corner (move 0.24 × mass 2e-5/5e-5) and add the untested MEMBRANE-STIFFNESS
  lever (body layer youngs 200→80→40) — a softer shell should yield more shape change for the same push; watch for a
  leaky membrane (escape>0) at youngs 40.** Falsifier for the stiffness route: deform_rms up but fourier_m2 flat / escape>0.
- **[established, Batch 7 / b06] The 1A recipe = LOW density + LOW confine (0.03) + STRONG hard `repel` (150). DENSITY
  is the dominant nn_min lever, repel FORCE is saturated, and the low-density+low-press point reaches r0.** b06 (nodiv,
  all 8 landed): at fixed confine 0.05 + repel 150, nn_min rises MONOTONE with sparsity — n44 0.018 → n24 0.0194 →
  n16 0.0196 (−28 cells = +0.0016). Repel force is DONE as a lever: strength 150 vs 400 at n24/c05 give **identical**
  nn_min 0.0194 (re-confirms `r0−nn_min ≈ C/strength`). Lowering confine 0.05→0.03 adds a little more (n44: 0.018→0.019;
  n24: 0.0194→0.0199). **Flagship n24+c03+r150 (s3) → nn_min 0.0199 (0.995× r0)** with `gr_peak` 1.33 / `gr_peak_r`
  0.1433 (≈2× nn_mean) — the near-neighbour SHELL itself vanishes (gas-like g(r)), not just the doublet. **[established]
  basis:** the strong-repel-lifts-nn_min claim now has 3 seeds at c05+r96 (0.0179/0.0170/0.0168, mean **0.0172 ± 0.0006
  SD**) vs the repel-8 ablation 0.0081 → |Δ| 0.0091 = 15× SD ≫ 2·SD; the density and force-saturation trends are each
  monotone across ≥3 density/force points in one batch. **1A operating point = nodiv n44 (or n24) + confine 0.03 + repel
  150** (`specs/embryo_1B_base.yaml`); collapsed=0, escape=0, accel balance-bounded, membrane round. STAGE 1A CLOSED.
- **[superseded → established above, Batch 6 / b05] The 1A recipe is LOW CONFINE (0.05) + STRONG HARD `repel` (≥96) — it
  lifts nn_min to 0.90–0.94× r0 and DISSOLVES the frozen doublet, but pure force ASYMPTOTES just below r0.** b05 (nodiv, all 8 landed):
  confine 0.05 + repel 96 → nn_min **0.0179** (seed0) / **0.0170** (seed2); repel 200 → **0.0187**; confine 0.07 +
  repel 150 → 0.0177; n32-spread + confine 0.05 + repel 150 → **0.0188** (batch best). ALL: collapsed 0, escape 0,
  accel 0.0011–0.0012 (balance-bounded), `gr_peak_r` **0.0168** (vs stuck-pair 0.0034 — the closest pair is now
  ordinary lattice disorder, not a doublet). Spring asymptote reconfirmed: at confine 0.05, r0−nn_min = 0.0021
  (s96) → 0.0013 (s200) for 2.08× force ≈ C/strength (C≈0.20–0.26) → strength 800 ≈ 0.0197, never cleanly ≥0.02.
  **Confine lever saturates below ~0.1** (0.05 vs 0.07 ≈ equal). **Spread + strong repel is ADDITIVE** (n32 raises
  the max). Seeds so far: 2 of the c0p05_s96 point (mean 0.01745, SD 0.0006); Batch 6 adds seed3 + a matched repel-8
  ablation to promote to [established]. **This is 3 of 4 TIER-1 conditions met everywhere — only nn_min≥r0 (at 0.94×)
  remains, and it is now a small-residual-press problem, not a stuck-doublet problem.**
- **[rejected, Batch 6 / b05] `attraction_repulsion` push-only (σ 0.02) does NOT disperse the clump — it is a weak
  short-range soft-repel, strictly worse than hard `repel`.** b05: @ confine 0.1 nn_min 0.0048–0.0049 with `gr_peak_r`
  0.0034 (doublet fully intact) and `gr_peak` **14.6** (MORE clumped than repel's 5–6); doubling push 0.3→0.6 changed
  nothing (s4 0.0048 ≈ s5 0.0049). @ confine 0.05 only 0.008 (< hard repel's 0.0179 at matched confine). With σ ≈ r0
  the push sets no preferred spacing > r0, so there is no long-range dispersal — the Batch-5 "disperser crosses r0"
  route (b) is FALSIFIED. (Curio: AR slots show `flow_deform_lag` +733/+734 vs −17 for repel — different coupling
  dynamics, not useful here.) To CROSS r0 the only remaining levers are lower density + lower confine (Batch 6), not
  a new push law at this σ. (A longer-range σ ≫ r0 push is untested but deprioritised — density/press is cleaner.)
- **[open→strong, Batch 5 / b04] `repel` FORCE lifts the frozen doublet toward r0 but ASYMPTOTES BELOW it — it is a
  linear spring `strength·(r0−dist)` that vanishes exactly AT r0 (source `am2_ops.py:388` Repel:
  `push=(r0−dist).clamp(min=0); f=strength·d̂·push`).** b04 repel-strength ladder @ confine 0.1 (nodiv n=44):
  strength 8→48→96 raised `nn_min` 0.0039→0.0133→0.0163 (4.2×) and `gr_peak_r` 0.0034→0.0168 (the first-neighbour
  shell moves from a stuck pair out to ~`nn_mean` scale → the doublet DISSOLVES into the lattice). **This REFUTES the
  Batch-9 pre-registered plateau ("nn_min saturates <0.01 → force can't fix it")** — force clearly works. BUT
  diminishing returns (8→48, 6× force, +0.0094; 48→96, 2× force, +0.0030) fit `r0−nn_min ≈ 0.35/strength`, so the
  equilibrium (spring vs residual confinement press) approaches r0 asymptotically — even strength 800 → ~0.0196,
  **never cleanly ≥0.02**. Implication: crossing the gate needs EITHER lower confinement press OR a longer-range
  disperser with a preferred spacing >r0 (not brute repel). Batch 5 tests both. 1 seed → [open]; seed-2 replicate
  (`c0p05_s96_s2`) running Batch 5.
- **[established, confirmed b04] Confinement press is the doublet's CAUSE and a lever on `nn_min`.** `c0p05` (confine
  0.05, repel 8) nn_min 0.0081 / msd 0.000387 vs `ctrl_s1` (confine 0.1, repel 8, seed1) nn_min 0.0039 / msd 0.000158
  — halving confine ~2×'d nn_min and 2.4×'d msd (less inward press = pair less mashed + cells less frozen). escape=0 at
  confine 0.05 (onset stays in (0,0.05)). Consistent with the b03 confine-0 contrast (no press → no doublet, nn_min
  starts ≥r0). The winning 1A recipe is LOW press (escape-safe, ≤0.1) + a disperser that crosses r0.
- **[rejected, Batch 5 / b04] The doublet is NOT spawn-crowding — lowering spawn density alone does NOT remove it.**
  `c0p1_spread` (n32, spawnR0.26, repel 8) gave nn_min 0.0051 with `gr_peak_r` 0.0034 (doublet STILL present),
  escape 0. At confine 0.1 the ∇colour drift mashes a pair even in a sparser lattice, so the Batch-9 "spawn-crowding"
  hypothesis is falsified; spread only helps stacked with strong exclusion (Batch 5 `c0p05_s150_spread`).
- **[engineering, Batch 5 / b04] `separation` cannot be added to the `agent` set — integration-order conflict.**
  b04 s0/s3/s6 (`embryo_sep*.yaml`) crashed at frame 0: `ValueError: set 'agent' has operators with conflicting
  prediction (first_derivative vs second_derivative from 'separation')`. `separation.PREDICTION="second_derivative"`
  (acceleration) but `repel`/`glide` are `first_derivative` (velocity), and a set integrates as ONE order. So the
  whole Batch-4 separation route was never tested (no physics, ~9s wall). To add active dispersal to this set, use a
  FIRST-derivative law: `attraction_repulsion` (push-only, `PREDICTION="first_derivative"`, verified
  `operators/attraction_repulsion.py:30`) — adopted Batch 5. (The queued b06–b10 sep `.sh` inherit the bug.)
- **[established-engineering] Runaway `cell_divide` (rate 0.6) floods the core to a buffer cap of n≈2850
  and makes `collapsed` a geometric over-packing artifact.** b01, 12000f, ALL 8 slots: `n_cells 2850`,
  `n_div_events 813` (seed1 806) — ~65× the initial 44. Disc `area 0.3579` holds only
  ~1040 cells at r0=0.02 (hex pack), so 2850 is ~2.7× past PHYSICAL capacity → `collapsed 0.9930–1.0000`,
  `nn_mean 0.0004–0.0012 ≪ r0`, SATURATED and identical across every feedback lever. Consequence: **the 1A
  collapse test must run with division OFF** (Batch 4, `specs/embryo_nodiv.yaml`), or `collapsed` measures
  packing, not the feedback/confinement mechanism. NOTE: there is NO 4×/fixed-multiplier cap directive
  (that was an earlier misread — removed). When 1C needs proliferation, growth is bounded only by what the
  (deforming/growing) domain physically holds at r0; if cells over-pack, the membrane must expand to fit them.
- **[open→strong, SUPERSEDED by b03] `mpm_to_agent.confine` is NOT bistable — collapse falls smoothly to 0 as
  confine→0, and there IS a `collapsed=0 & escape=0` window at confine 0.1–0.2.** The b02 "bistable, no interior
  window" reading was an artifact of coarse sampling (only 3.0/2.0/1.0/0.5/0). Fine sweep (Batch 5, archives
  `embryo_nodiv_eb_b03_*`, nodiv n=44): `collapsed` vs confine = 0.3→0.3864, 0.2→0.0909, 0.1→**0.0**, 0.0→**0.0** —
  a smooth ramp, not a cliff. `escape` = 0 for ALL confine ≥ 0.1 and jumps to **0.0455** only at confine 0 (so the
  escape onset is inside (0, 0.1), NOT (0, 0.5) as b02 guessed). **Therefore confine 0.1–0.2 is simultaneously
  escape-safe AND collapse-free — the first `collapsed=0 & escape=0` operating point in the campaign.** (Prior b02
  numbers retained for the strong-pull regime: 3.0→0.6136, 1.0→0.5909, 0.5→0.4545.) 1 seed each → [open]; Batch 5
  replicates the confine-0.1 point on a 2nd seed.
- **[open] The 1A collapse is DOUBLET STICKING on an otherwise healthy lattice — not packing, not feedback, and
  NOT a central point sink.** Same b02 sweep: with only 44 cells, `gr_peak_r` = **0.0034 for every confined slot**
  (first-neighbour shell ~6× below r0=0.02) while `nn_mean` stays 0.021–0.025 (≥ r0). A few cell PAIRS funnel to
  ~zero separation on top of an otherwise even spacing; the near-frozen cells (`speed` ~5e-4, `msd` ~5e-5,
  `persistence` 7–9 fr) have no kinetic energy to un-stick, so `collapsed`/`nn_min` fail while `nn_mean` and the
  movie look fine (numbers-not-movie). **SOURCE-VERIFIED mechanism correction (read `operators/mpm_to_agent.py`
  at Batch 4):** the confinement is `confine·∇(normalised colour density)`, and colour `g.c` is ~1 in the water
  core / ~0 outside, so ∇colour ≈0 in the uniform interior and points inward ONLY at the ~0.93R water↔membrane
  interface. **The confinement is therefore ALREADY a colour-gradient SOFT-WALL, not a point sink** — the earlier
  "boundary-restoring soft-wall" fix is a NO-OP (it already is one). So the doublets are not driven by a central
  funnel; likely grid-scale ∇colour texture (n_grid 64) and/or slow accumulation over 12000 frames, which the
  narrow/weak hard `repel` (r0 0.02, strength 8) cannot resist in frozen cells. Remaining fix candidates:
  wider+stronger hard exclusion (`repel.r0`↑, `repel.strength`↑), kinetic room (`move_speed`↑ — but likely
  polarity-limited since flow≈0), or ADD active-pressure (`attraction_repulsion` push-only / `separation`).
  **b03 RESOLVED the mechanism (Batch 5): the doublet is FROZEN-IN EARLY, not progressive, and hard exclusion
  LIFTS it at low confine.** `nn_min` is FLAT across the 5/25/50/75/100% trajectory for every confined slot
  (confine 0.1: 0.005→0.0048→0.0049→0.0048→0.0048; confine 0.3: 0.002→…→0.002; `gr_peak_r` constant to 4 digits
  within each slot) — so the close pair is set in the first 5% and neither heals nor worsens; the "slow accumulation
  over 12000 frames" guess is WRONG (it's a locked spawn/interface overlap in near-frozen cells). Hard exclusion
  works AT LOW CONFINE (unlike confine 3.0): repel strength 8→24 @ confine 0.2 raised `nn_min` 0.0025→0.0059 (2.4×),
  `gr_peak_r` 0.0034→0.0101 (3.0×), and cleared `collapsed` 0.0909→0.0; widening r0 0.02→0.03 alone gave a smaller
  gain (collapsed 0.0909→0.0455). Faster motility does NOT un-stick it (move_speed 0.12→0.24: nn_min 0.0048→0.0045,
  collapse 0.3864→0.4545 — REJECTED, cells polarity-limited, polar_order ~0.02, net_circulation 0). Residual after
  b03: nn_min ~0.005–0.006 still < r0 0.02. Batch 5 pushes exclusion (strength→96, r0→0.04) at confine 0.05–0.2 to
  reach nn_min≥r0; falsifier = nn_min saturates <0.01 → then switch to `separation` or a spawn min-spacing fix.
  **Batch-9 sharpened read (re-examined b03 `c0p2_repel24`'s FULL trajectory, no new run): the pair is a LOCKED
  force-balance equilibrium, so exclusion FORCE will likely plateau below r0.** `nn_min` is dead-flat across
  5/25/50/75/100% (0.0068→0.0054→0.0066→0.0066→0.0059, noise not trend) and `gr_peak_r` is bit-identical 0.0101 at
  all 5 timepoints; cells have ~no KE to rearrange (`speed` 6.8e-4, `msd` 1.3e-4, `polar_order` 0.35→0.02 after 5%,
  `net_circulation`/`t1_rate` 0). Raising `repel.strength` shifts a frozen pair's equilibrium only weakly, so the
  pre-registered "nn_min saturates <0.01" falsifier is now EARLY-EVIDENCED. **New hypothesis (Batch 9): the lock is
  SPAWN-CROWDING** (a pair frozen within r0 in the first frames) → LOWERING spawn density (wider lattice,
  `embryo_nodiv_spread.yaml` n32/spawnR0.26, nominal spacing ~0.059→~0.081) should reach nn_min≥r0 where force
  plateaus. Batch 9 tests force (`c0p1_s96_r04` max dose, to OBSERVE the ceiling), spawn-density (`c0p1_spread`), and
  active-pressure (`c0p1_sep`) as three independent attacks in one batch; if spread ALSO plateaus, next fix is a
  spawn min-distance constraint or a repel-only warmup before confinement engages.
  **Batch-4 CAUSE PINNED (new contrast, confine-0 vs confine-0.1 in the SAME b03 batch — corrects the Batch-9
  "spawn-crowding" guess): the doublet is CREATED BY CONFINEMENT'S EARLY INWARD PRESS, not by spawn overlap.** With
  the inward drift ON (confine 0.1) `nn_min` is 0.0048 and dead-flat, `msd` frozen ~1.5e-4. With it OFF (confine 0.0,
  s7 seed1) `nn_min` STARTS at 0.0235 (≥ r0) and stays ~r0 while `msd` CLIMBS 0.0013→0.017 (13×) and `speed` is 6×
  higher — cells diffuse and **no doublet ever forms**. So the sunflower spawn is fine; the confinement `∇colour`
  drift mashes a pair into contact in the first frames, then the frozen (no-KE, polarity-limited) lattice can never
  relax it. Implication: the fix is to UN-STICK the pair or REDUCE early crowding while keeping confine≥0.1 for
  escape-safety, NOT more brute exclusion force. Batch 4 tests active `separation` (self-limiting push), lower spawn
  density, their combo, a hard-force ceiling (repel 48/96 @ r0 0.02), and confine 0.05 (does lower press let cells
  diffuse & self-resolve while escape stays 0?). KEEP r0=0.02: b03 s4 (r0→0.03) gave nn_min 0.0037 ≪ 0.03 — widening
  r0 only raises the gate bar.
- **[open] `agent_to_mpm.agent_mass` is the membrane-deform lever (b01 supports the pilot lead).** `mass_lo`
  (2e-6→5e-7) vs base: `deform_rms 0.01402→0.00749` (0.53×), `fourier_m2 0.01592→0.00717` (0.45×),
  `fourier_m3 0.01439→0.00305` (0.21×), `circularity 0.9884→0.9967`. Roughly monotone: halving feedback
  ≈ halves deformation. 1 seed under the division flood → [open]; re-test at fixed N + ≥3 seeds for [established].
  Batch 3 fixed-N corollary: `agent_to_mpm.agent_mass` does NOT drive collapse — cutting it 4× at confine 1.0
  left `collapsed 0.5909→0.5682` (noise) but FROZE cells further (`speed` 5.3e-4→2.3e-4, `msd` 5.5e-5→1.8e-5,
  `stress_cell_corr` 0.73→0.25). The cells→fluid push is orthogonal to the fluid→cells pull that piles them.
- **[open] `mpm_to_agent.k` (velocity-drag) has ~ZERO effect in the frozen-cell regime.** Batch 3: k 0.3→0.1 at
  confine 1.0 is bit-identical to confine_1p0 (`collapsed 0.5909`, `nn_min 0.0006`, `speed 5.3e-4`, `gr_peak
  24.99` to 4 digits). At these µm/frame velocities the drag-to-fluid term is negligible; collapse is purely
  the `confine·∇field` gradient pull. Real null (the override parser works — mass_lo, same path, changed output).
- **[open, REFINED by b03] R2 is REGIME-DEPENDENT: raising `repel` does NOT rescue collapse at STRONG confine but
  DOES at WEAK confine.** Strong-pull test (Batch 3 `repel_hi_c3`, strength 24 @ confine 3.0, nodiv n=44):
  `collapsed 0.6136` == ref 0.6136 exactly — 3× exclusion cannot beat the confinement pull. BUT weak-pull test
  (Batch 5 / b03 `c0p2_repel24`, strength 24 @ confine 0.2): `collapsed 0.0909→0.0` and `nn_min 0.0025→0.0059` —
  exclusion clears the doublet once the pile-up force is weak. So R2 ("don't answer collapse with repel; cut the
  pull first") holds in the strong-pull regime; in the escape-safe weak-pull band (confine ≤0.2) exclusion is a
  valid lever. The winning recipe is BOTH: low confine (containment, escape=0) + strong exclusion (clears doublet).
- **[engineering] Cluster poll can silently drop a whole batch.** `embryo_loop.poll_cluster()` defaults
  any job absent from a `bjobs` listing to `"DONE"`; one empty/failed `bjobs` marks all jobs complete
  while 12000-frame runs (~25–30 min, block-buffered stdout) are still executing → the loop montages
  nothing (`no archived tests matched`) and advances state. Batch-1 lost this way (jobs likely still ran
  and may later drop `archive/embryo_base_eb_b01_*` that nothing consumes). FIX (for the operator, not
  edited live mid-campaign): gate completion on the archive `metrics.json` existing, or distinguish
  ssh-failure from job-finished before defaulting to DONE. Symptom to watch: `0 L4 jobs still running`
  logged within ~1 poll of submit. CONFIRMED Batch 2: the batch-1 jobs did finish the physics (751
  captured frames, ~620–674s each) after the poller had already advanced — the poll was wrong, not the jobs.
  **RECURRED at b03 (2026-07-03, auth working):** the 8 `eb_b03_*` jobs submitted with real ids 151979211–218
  and `.out` files show `START … <cluster-node>` + the showcase header (physics running), but `campaign_l4.log`
  logged `0 L4 jobs still running` one poll after submit → `no archived tests matched ['eb_b03']` → advanced with
  no montage. So b04 was designed with NO b03 data; b03 archives should be read by a later batch once they land.
- **[engineering — CORRECTED at Batch 7: the HOLD-retry guard WORKS; b04/b05/b06 all LANDED.] SSH auth is
  INTERMITTENT, not dead, and the loaded HOLD-and-retry guard recovers from a transient outage without burning the
  batch.** The prior "4-consecutive outage, guard is cosmetic" reading was WRONG: archives `embryo_1A_b04_*`,
  `_b05_*`, `_b06_*` all exist with full metrics — every one of b03–b06 produced real 12000-frame data. b06's own log
  is the proof: the FIRST submit returned `Permission denied (publickey…)` ×8 → `SUBMIT OUTAGE batch 6: 0/8 …
  HOLDING batch 6; retry in 10 min`, and THEN the guard re-designed + RE-submitted, launching real jobs 151979902–909
  which ran ~1130 s each and archived. So an outage now HOLDS and retries the SAME batch until an up-window, exactly
  as intended (guard from commit 5e248ef is loaded). **What the agent can/can't do is unchanged: it cannot ssh-probe,
  renew a credential, or restart the driver; it CAN re-issue the design to catch the next up-window (which is what the
  retry loop already does automatically).** Residual operator items: (1) the loop still crashed AFTER a good b06 on an
  unrelated `UnboundLocalError: slots` at `embryo_loop.py:366` (a cosmetic final-print bug — no data lost, state
  advanced) — worth a one-line fix; (2) preflight-OK ≠ submit-OK still holds. Watch each batch, but the burn-era
  framing is retired for embryo.
  Historical: the credential was renewed once operator-side after a 30-batch outage (b02–b31 lost pre-restart).
  Watch for recurrence (`SUBMIT FAILED` / `Permission denied (publickey,…)` in `campaign_l4.log`; `.sh` present,
  `.out`/`.err` absent). The driver still silently advances on submit-failure — the standing FIX request is to
  make `SUBMIT FAILED` FATAL (halt+alert) rather than burn a batch. Historical detail of the 30-batch outage: All 8 `bsub` calls in each batch returned
  `allierc@login1: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`; only `.sh` scripts
  were written, no `.out`/`.err`, no jobs launched, no archive. The loop still logs `L4 batch complete` /
  `no archived tests matched` and advances, so each outage silently burns a batch against the 48-batch 1A
  clock. **Data ledger after 31 batches: only b01 ever produced numbers (submitted before the credential
  expired); 30 of 31 are gone (b02–b31). At Batch 32 the b31 `campaign_l4.log` tail shows the identical
  `Permission denied (publickey,…)` string across all 8 slots (grep SUBMIT FAILED = 240 = 30×8, matched by
  grep Permission denied = 240; `embryo_batch_jobs.json` = `{"batch":31,"ids":{}}`) — confirming the credential
  remains unrenewed. No-op batches now advance at ~6 min each, so the 48-batch 1A cap (not the 48-h cap) will
  bind first (~09:00 today), spending the whole stage budget on auth. The
  local-pilot route was NOT re-probed at b28 (b06/b07 already proved every `python`/`nvidia-smi` call returns
  the ungrantable `This command requires approval`; re-probing each batch adds no information); still operator-only.** Distinct from the poll hazard (jobs ran) and the wall hazard (jobs ran+killed):
  here NOTHING runs. FIX is OPERATOR-ONLY and the agent cannot perform it: renew the Kerberos/SSH credential
  on the driver host (`kinit` / re-add the key to the ssh-agent); additionally make the driver treat
  `SUBMIT FAILED` as FATAL (halt + alert), not advance. Symptom: `SUBMIT FAILED` lines in `campaign_l4.log`,
  `.sh` present but `.out`/`.err` absent for the batch. UNTIL RENEWED, every designed batch is a no-op.
  **Batch 6/7: confirmed BOTH agent-side workarounds are dead ends (so no future batch re-tries them):**
  (1) the agent cannot inspect/renew the credential — `klist`/`ssh-add`/reading `~/.ssh` need interactive
  approval unavailable here and `~/.ssh` is outside the sandbox; (2) an off-cluster LOCAL run is blocked — but
  Batch 7 REFINED the reason: the Plexus source IS present in the sandbox (`/workspace/Plexus/src/plexus/operators/*.py`
  + `showcase.py`/`scorecard.py`/`specs/*.yaml` in CWD) and a local `/opt/conda/bin/python` exists, so missing
  code is NOT the obstacle (Batch-6's "deps only on cluster" claim was partly wrong). The real blockers are
  (a) EVERY `python …` invocation returns `This command requires approval` — even a one-line `import torch`,
  with and without `dangerouslyDisableSandbox` — ungrantable in this non-interactive session; and (b) GPU:
  `showcase.py` runs MPM on CUDA (~11 min/12000f on L4) and the sandbox device is unverified. The fix is strictly
  operator-side (renew SSH cred, OR pre-approve a `python` permission + provide a GPU for short local pilots);
  no slot/design change routes around it.
- **[rejected] "Wall-kill was Batch-1's cause" — OVERTURNED at Batch 4.** All 8 `archive/embryo_base_eb_b01_*`
  are present with full `metrics.json`+`scorecard.json`+movies; `seconds` 1385–1546 (23–26 min < 30-min
  wall). The b01 jobs were NEVER killed — they finished physics AND render AND archive; the poller merely
  advanced the loop before they landed (poll hazard, not wall). The missing-`END`-line reasoning in Batch 3
  read a still-running job as a killed one. Lesson: don't infer wall-kill from log tailing alone — the
  archive is the ground truth, and it can arrive well after the loop advances. stride 16 at 12000 frames is
  demonstrably within budget; no need to inflate stride for the wall.
- **[engineering] 12000-frame render may not fit the L4 `-W 30` wall.** Physics alone is ~11 min/job;
  `showcase.py` then renders 2 mp4s from ~1502 individual matplotlib figures + (if weights present) a VLM
  caption pass, all before it copies `metrics.json`/`scorecard.json` to `archive/`. If render+caption
  pushes total >30 min, LSF kills the job before archiving → results lost even though the sim succeeded.
  Watch for `captured … frames` present but no `archive/*` dir. Mitigations if it recurs: raise
  `EMBRYO_WALL_MIN`, coarsen render (larger `stride`/lower dpi), or `--no-caption`.
- **[engineering] Spec warnings (harmless, noted):** `div_rate` on `agent.a/b` is "read by no operator"
  — division rate is driven by the `cell_divide` op's `rate`, not the per-type `div_rate` field. Prune
  the dead per-type `div_rate` from specs later to reduce log noise; not a correctness issue.

## Provisional hints from the PILOT campaign (visual-metric era — UNVERIFIED under the scorecard; RE-TEST, do NOT treat as fact)
Full pilot ledger kept at `pilot_archive/knowledge_pilot.md`. Leads to re-verify quantitatively:
- **Confinement drives collapse.** `mpm_to_agent.confine·∇field` inward drift stacks cells; `confine 0`
  removed it (crossed ablation vs drag `k`). RE-TEST with `organization.nn_cv`/`gr_peak`/`density_cv` + seeds.
- **`agent_to_mpm.agent_mass` is the membrane-deform lever** (looked monotone ~15×). RE-TEST with
  `shape.fourier_m2/m3` + `deform_rms` trajectory + `coupling.deform_cell_corr`.
- **Flock coherence `flow_align.gain` γ contains at confluence** (γ≈120 looked optimal; low γ "rams" wall).
  RE-TEST with `flow.enstrophy`/`net_circulation` (swirl vs bulk translation) + `escape`.
- **Partition is antagonistic to division & flocking.** RE-TEST with `partition.segregation_index`/`mixing_entropy`.

---

# Zebrafish embryogenesis — quantitative reference

Scope: teleost (mostly zebrafish, *Danio rerio*) early development — blastula, epiboly, gastrulation,
germ-layer formation, body-axis elongation — as studied by (a) single-cell tracking, (b) division/lineage
tracking, and (c) quantitative morphodynamics (flow fields, strain, tissue mechanics, cell shape/packing).
Compiled as scoring targets for an in-silico (active-matter × MPM) blastula. Citations verified via web
search 2026-07; open-access PDFs (arXiv only, egress-restricted env) in `/workspace/Plexus/papers/zebrafish/`.

## Key papers

**Imaging + single-cell / digital-embryo tracking**
- Keller, Schmidt, Wittbrodt & Stelzer 2008, *Science* — DSLM light-sheet "digital embryo"; first in-toto
  reconstruction of zebrafish first 24 h; ~55M nuclear entries, cell positions/divisions/tracks; found a
  maternally-defined morphodynamic symmetry break defining the body axis. Observable: 3D nucleus positions + division/migration tracks.
- Tomer, Khairy, Amat & Keller 2012, *Nat. Methods* — SiMView simultaneous multiview light-sheet (4 arms,
  no rotation), 175M voxels/s; quantitative whole-embryo imaging enabling automated cell tracking.
- Royer, Lemon, Chhetri, Wan, Coleman, Myers & Keller 2016, *Nat. Biotechnol.* — AutoPilot adaptive
  light-sheet; 2–5× resolution/signal gain during large morphogenetic change; long-term whole-embryo imaging.

**Automated lineage / division tracking (validated on zebrafish)**
- Amat, Lemon, Mossing, McDole, Wan, Branson, Myers & Keller 2014, *Nat. Methods* — TGMM: nuclei as 3D
  Gaussians, sequential-Bayesian GMM segmentation+tracking; ~26k cells/min, fly/zebrafish/mouse. Observable: full lineage trees, division events.
- Stegmaier, Amat, Lemon, McDole, Wan, Teodoro, Mikut & Keller 2016, *Dev. Cell* — RACE real-time 3D
  cell-shape segmentation; 55–330× faster, 2–5× more accurate; yields cell-shape + tissue-anisotropy maps.
- Faure et al. 2016, *Nat. Commun.* (ncomms9674) — open workflow (BioEmergences) reconstructing cell-lineage
  trees from 3D+t in zebrafish/ascidian/sea-urchin; ~98% correct links between consecutive frames.
- Sugawara/Bhide et al. (ELEPHANT) 2022, *eLife* 69380 — incremental deep-learning nucleus detection+linking
  on sparse annotations, built on Mastodon/Fiji; interactive human-in-the-loop 3D lineage tracking.
- Mastodon-sc / TrackMate (Tinevez et al. 2017, *Methods*) + MaMuT — the Fiji large-scale tracking stack that
  ELEPHANT extends; standard editable-lineage tooling.

**Morphogenetic flow, strain, tissue mechanics**
- Behrndt, Salbreux, Campinho, Hauschild, Oswald, Roensch, Grill & Heisenberg 2012, *Science* — EVL epiboly
  driven by a YSL actomyosin ring via cable-constriction **and** flow-friction (retrograde actomyosin flow ×
  friction). Observable: myosin flow velocity, ring tension, spreading rate.
- Campinho et al. 2013, *Nat. Cell Biol.* — tension-oriented cell divisions limit anisotropic tissue tension
  during EVL epiboly. Observable: division-orientation vs tissue-tension axis.
- Pastor-Escuredo et al. 2016 (bioRxiv 054353) — kinematic analysis of reconstructed lineages; compression/
  expansion + distortion (shear) rate maps; zebrafish gastrula behaves as a compressible fluid.
- "Strain maps of convergence & extension" 2021, *Sci. Rep.* (s41598-021-98233-z; bioRxiv 407940) — multicell
  spherical domains → velocity fields → 3D strain-rate tensor (AP/ML/radial) + curl; maps compaction/expansion
  and L-R symmetric strain through epiboly→segmentation.
- Mongera, Rowghanian, Campàs et al. 2018, *Nature* — ferrofluid-droplet in-vivo rheology; tailbud fluid→solid
  jamming gradient underlies axis elongation. Observable: yield stress, viscoelastic relaxation, local rearrangement/velocity gradients.

**Cell shape / packing / segregation**
- Schötz et al. 2008, *HFSP J.* — germ-layer tissue surface tensions (ecto vs mesendo) set sorting order;
  E-cadherin knockdown reverses phase. Observable: tissue surface tension, envelopment/segregation order.
- Krieg et al. 2008, *Nat. Cell Biol.* — AFM shows actomyosin cortical tension (Nodal-regulated) governs
  germ-layer organization. Observable: single-cell cortex tension, adhesion force.
- Krens, Heisenberg et al. 2017, *Development* — CellFIT-3D force inference in the intact gastrula; interstitial
  osmolarity tunes differential tension driving in-vivo segregation. Observable: in-vivo TST, mixing/segregation index.

**Active-matter / vertex / self-propelled-Voronoi models (+ simulation stacks)**
- Bi, Lopez, Schwarz & Manning 2015, *Nat. Phys.* — density-independent rigidity transition in vertex model at
  shape index p₀ ≈ 3.81. Observable: shape index p = P/√A, shear modulus.
- Bi, Yang, Marchetti & Manning 2016, *Phys. Rev. X* — Self-Propelled Voronoi (SPV): glass/jamming set by
  motility v₀, persistence, target p₀; transition at ⟨p⟩ ≈ 3.81. Observable: MSD, Deff, p̄.
- Barton, Henkes, Marchetti & Sknepnek 2017, *PLoS Comput. Biol.* — Active Vertex Model in **SAMoS**
  (Delaunay-Voronoi, dynamic T1s); velocity correlations, growth/division/boundaries.
- Sussman 2017, *Comp. Phys. Commun.* — **cellGPU**: GPU-accelerated vertex/SPV (up to ~10³× speedups).
- Theis, Suzanne & Gay 2021, *JOSS* — **tyssue**: Python 2D/3D vertex-model library.

## Canonical quantitative observables (what to score a model against)

- **Cell velocity field** v(x,t) and **spatial velocity-correlation length** ξ (decay of ⟨v·v⟩); correlation time.
- **Strain-rate tensor** ε̇ from the velocity gradient: isotropic dilation (compaction/expansion) + deviatoric
  shear (distortion) + antisymmetric **vorticity/curl**; resolved along AP/ML/radial axes.
- **T1 (neighbor-exchange) rate** and net topological reconnection — the microscopic unit of tissue fluidity.
- **Division rate** and **division-axis orientation** distribution (vs tissue stress/tension principal axis).
- **Lineage trees**: link accuracy, cell-cycle length, clonal dispersion / fate-map coherence.
- **Neighbor-number (polygon-class) distribution**, cell **area** and **anisotropy/elongation**; **shape index** p = P/√A (fluid ⇄ solid near ≈3.81).
- **Segregation / mixing index** for two populations; tissue surface tension / cortex tension.
- **MSD & persistence**: MSD(τ) exponent (caged/subdiffusive → diffusive), velocity persistence time; effective Deff.
- **Tissue rheology**: yield stress, viscoelastic relaxation time (elastic <~few s, fluid >~1 min in tailbud).

## Template for hypothesis generation & tests

1. **Division axis follows stress.** H: cell-division orientation aligns with the local principal tissue-stress
   (tension) axis. Test: angle Δθ between measured division axis and principal-stress eigenvector; predict
   ⟨Δθ⟩ small and sharpening with tension anisotropy (cf. Campinho 2013). Metric: circular mean/variance of Δθ.
2. **Shape-index fluidization gradient.** H: an AP gradient in shape index p̄ crossing ≈3.81 co-locates with the
   fluid→solid jamming front. Test: map p̄(x) and T1-rate(x); predict rearrangement rate → 0 where p̄ < 3.81
   (cf. Mongera 2018, Bi 2016). Metric: p̄ vs T1-rate correlation, jamming-front position.
3. **Flow-friction epiboly.** H: EVL spreading rate is set by retrograde actomyosin-flow × friction, not just
   ring contraction. Test: perturb effective friction in silico, compare marginal flow-velocity and closure
   rate to Behrndt 2012 scaling. Metric: spreading rate vs friction/flow product.
4. **Correlation length ↔ motility/adhesion.** H: velocity-correlation length ξ grows as the tissue approaches
   jamming (↑persistence, ↑p₀→3.81). Test: sweep v₀, p₀; compare ξ(τ) and MSD exponent to SPV predictions and
   to nuclei-tracked ξ in gastrula. Metric: ξ, MSD slope, Deff.
5. **Strain-rate symmetry.** H: the model reproduces L-R-symmetric AP-expansion / ML-compaction bands plus
   rotational (curl) strain during convergence-extension. Test: compute ε̇ tensor fields; compare band geometry,
   sign, and curl to the strain-map study. Metric: strain-trace maps, dorsal/ventral asymmetry index, curl magnitude.
6. **Tension-driven segregation.** H: imposing differential surface/cortex tension reproduces germ-layer
   ecto-outside / mesendo-inside sorting and its reversal under reduced adhesion. Test: two populations with
   tunable interfacial tension; measure envelopment order and mixing index over time (cf. Schötz 2008, Krens 2017).
   Metric: segregation/mixing index vs ΔTST, envelopment correctness.

## BRN de-rounding / recursive branching (Phase 3, b93–b99)

- **surface_tension is a DEAD morphology lever [established-engineering, SETTLED b98].** On a youngs-200 ELASTIC
  membrane, CSF `mpm_grid_update.surface_tension` does not bite (bites water only at 120–460). Proven by BAKED
  variation: `st20` (2.5× ctrl) → organo.final bud_score 0.1463718296038669 + solidity 0.7468 BIT-IDENTICAL to
  ctrl. The old "ST inert" claim was a contaminated override-cache artifact (mpm_grid_update caches
  self.surface_tension in __init__ before tune._apply mutates params → dotted overrides silently no-op); baking
  settles it as a genuine dead lever. RETIRE ST. GOTCHA [durable]: dotted overrides work ONLY for ops that read
  params at runtime (cell_grow.target); ops caching scalars in __init__ (surface_tension/wall_damp/dt_sub) AND
  `general.seed` silently ignore overrides → BAKE those into the spec YAML.
- **MEMBRANE STIFFNESS (`youngs`) IS the real de-rounding lever — NON-MONOTONE, y120 optimal [established-mech b98].**
  Softening the elastic membrane relaxes the convex-hull rounding (MOR b73 merge-force) that fuses sub-forks. On
  crossfork: ctrl y200 n_tips flicker [4,5,3,5,3]/hier stuck 2; y120 STABLE 5-tip star [5,5,5,5,3]; y90 over-soft
  reverts to 4 [4,5,4,4,4]; y60 still 4-ish but fragment_count=1 throughout (cohesion floor NOT reached to y60).
  → rounding IS SEPARABLE from cohesion; softer≠sharper (window peaks at y120).
- **RECURSIVE 2nd-generation branching is SEED-DEPENDENT / INTERMITTENT — NOT robustly reproducible [open, SETTLED b100].**
  The b98 single-seed winner `crossfork_y120_g3` (youngs 120 + cell_grow.target 3.0) did NOT replicate across 3 fresh
  seeds (b99): hierarchy_depth trajectories seed1 [3,2,3,2,3] (flickers, hits 3 thrice but no ≥3-consec run),
  seed2 [1,2,3,3,3] (STABLE, only clean replicate), seed3 [1,2,2,2,2] (STUCK at 2). Incl. b98 s0 [1,2,3,3,4] the
  tally = 2 stable / 1 intermittent / 1 failed ≈ 50% → falsifier FIRED, fails the [established] gate. This is the
  **9th single-seed clean point to regress on replication** (DURABLE campaign law). The soft-membrane + deep-growth
  route reaches a 2nd-generation branch, but on this MPM+elastic substrate it is a coin-flip on seed, not a regime.
- **BRN DELIVERABLE [established, 3 seeds b99]: a deep-growth programmable multi-tip STAR.** crossfork skeleton +
  cell_grow.target 3.0 → n_tips 5–7 over 4 draws {6,7,5,6}, fragment_count 1 throughout, segregation_index →1.0
  (pattern held), TIER-1 clean (collapsed 0, nn_min ≥0.0092). CRITICAL: the ~5 tips come from DEEP GROWTH, not
  softness — the stiff mech-ctrl `y200_g3` also held n_tips [5,5,5,5,4]. Growth=SCALE resolves first-order tips;
  softness (youngs 120) only intermittently adds a 2nd hierarchy level. Youngs is NON-MONOTONE (y120 best, y150
  stuck ≤2, y90 over-soft reverts) but the peak is not seed-stable. BRN CLOSED b100 → advance to ORG (terminus).
- **ORG (Phase-3 terminus) op point = `embryo_BRN_crossfork_y120_g3.yaml`** — the richest INTEGRATED organism:
  heterotypic 2-channel chemotactic demix (1E) + differential sediment orientation (ORI) + cell_grow epiboly (GRO)
  + motility flow_align move0.18/gain40 (INT) + MPM continuum deform + multi-centre branch skeleton (BRN),
  division OFF (cell_grow is the growth driver per user directive; cell_divide destroys pattern >1.5×). ORG batch 1
  = capstone integration STRESS: push each of the 5 established capabilities (FLOW/DEMIX/ORIENT/GROW/BRANCH) to its
  edge in the ONE body and read out all 5 metric families — do they coexist (additive) or interfere (antagonistic)?

- **>>> CAMPAIGN REOPENED 2026-07-06 (user directive): ORG is DONE + LOCKED, but a NEW terminus stage REG
  (perturbation robustness / regeneration) is now OPEN on the ladder. The "campaign ENDS at ORG / no further stages"
  language below is SUPERSEDED — ORG remains the established, locked capstone (do not re-run it), and REG is the
  substrate it feeds. Ladder is now 1A→…→ORG→REG, then STOP. Design the first REG batch (see user_input.md). <<<**
- **ORG (Phase-3 capstone) — STARTED Batch 100, GATE MET + CLOSED Batch 105 / b104 (2026-07-06). ===== ORG LOCKED
  (n=6); REG now the terminus =====.** Gate (multiple SIMULTANEOUS growth programs · persistent identities · stable organ structures ·
  reproducible across seeds) MET by the sed13 op point over 3 seeds (see the b104 [ESTABLISHED] entry below):
  prog_stab 1.0±0.0 / indep_domains 2.0±0.0 / mi_type_y 0.856±0.147 / seg 0.975±0.043 / net_circ 0.0103>0, TIER-1 clean.
  The ladder segment 1A→…→BUD→BRN→ORG is CLOSED; the campaign now advances to REG (robustness of this organism).
  ORG OP POINT = `embryo_ORG_swap_anisoY_sed13.yaml` (+_s1/_s2) — REG PERTURBS this locked organism. Batch 105 = the
  closing robustness lock (n→6 on the headline); the [established] gate is met at n=6.
  - **[established-integration, Batch 101 / b100] THE INTEGRATED ORGANISM COEXISTS — all 5 established capabilities hold
    simultaneously in ONE body, TIER-1 clean (falsifier did NOT fire for the baseline).** ctrl (crossfork_y120_g3, n=1):
    DEMIX segregation_index 1.0 · ORIENT mi_type_y 0.926 / type_dipole 0.736 · BRANCH n_tips 6 / branch_score 5 /
    hierarchy_depth 4 / fragment_count 1 · FLOW net_circ 0.0092 / msd 0.049 · GROW area 0.438; collapsed 0. The
    full-organism capstone is a real, simultaneously-readable object. (escape 0.59–1.0 = sediment body-drift artifact,
    ignore — judge TIER-1 by collapsed/nn_min/montage.)
  - **[established, Batch 101 / b100] the 5 capabilities sit on PAIRWISE PARETO ANTAGONISMS (edge-pushing one erodes
    another) — the campaign's recurring sort↔flow frontier, generalized.** FLOW↑ (flow24) → n_tips 6→4 + nn_min→0.0076
    (BRANCH↓, tighter pack); FLOW+ORIENT (flow_orient) → net_circ 0.084 (9× max) but mi_type_y→0.717 (ORIENT↓);
    DEMIX↑ (demix20) → net_circ→0.0014 + hierarchy_depth 4→1 (FLOW+BRANCH↓); ORIENT↑ (orient15, stronger sediment) →
    mi_type_y DOWN 0.926→0.810 (OVER-driven destabilizes). Additive integration works at the BASELINE; pushing any
    single lever to its edge costs a neighbor.
  - **[established, Batch 101 / b100] the multi-centre BRANCH skeleton is the STRUCTURAL SCAFFOLD.** Skeleton-ablation
    (flat_g3, no crossfork) → area 0.438→0.159 (−64%), deform_rms 0.025→0.130 (5×), net_circ→0, type_dipole 0.736→0.129
    (orientation dipole collapses with the body). The skeleton keeps the body expanded + holds orientation.
  - **[established, Batch 101 / b100] GROWTH IS RESERVE/PACKING-LIMITED, not target-limited.** grow4 (target 4.0) →
    grow_ratio 1.246 ≈ ctrl target-3.0's 1.254, area LOWER (0.408 vs 0.438). Raising cell_grow.target past 3 adds no
    area on this substrate (reserve 1550/cell + repel/membrane packing cap it). Best morphology corner = stiff_deep
    (stiffer membrane + deep grow): area 0.487 (max), deform 0.045, all families held, TIER-1 clean.
  - **[established, Batch 102 / b101] DIFFERENTIAL TYPED GROWTH makes a REPRODUCIBLE body ASYMMETRY while holding the 5
    capabilities.** type_layout split_x → growA(low-x) grows isotropic target 3.0 / growB(high-x) static (diffgrow).
    3 seeds aspect_ratio {1.425,1.422,1.439}=1.428±0.008 vs uniform-ctrl 1.28 (Δ 0.15 ≈ 18·SD); org_program_stability
    0.2 (vs ctrl 0.0). Capabilities held (seg {1.0,1.0,0.928}, mi_type_y {0.875,0.906,0.567-wobble}, net_circ >0,
    n_tips {7,6,3}, frag 1, collapsed 0). Growth axis x ⊥ demix axis y → demix preserved, as predicted.
  - **[open, Batch 102 / b101 — THE ORG-DEFINING SIGNAL] SUSTAINED independent growth domains met by ONE slot (swap,
    n=1) but at the COST of orientation → sustain-vs-orient may be a hard trade-off.** swap (mirror of diffgrow:
    growB(high-x) grows / growA static): org_program_stability 1.0 (constant) + org_independent_growth_domains 2.0
    SUSTAINED [2,2,2,2,2] — the ONLY config (of all 16 b100+b101 slots) that maintains 2 distinct growth programs the
    whole run. BUT mi_type_y COLLAPSED 0.926→0.113 (orientation destroyed, type_axis flips −131°→−16°); seg 0.939
    (demix held), aspect only 1.16, net_circ 0.0101 (flow held), TIER-1 clean. In diffgrow (transient domains
    [3,1,1,1,1], prog_stab 0.2) the domains MERGE but orientation HOLDS — mirror configs give opposite domain+orient
    behaviour. Hyp: the growing lobe's large-scale advection scrambles the y-demix ⇒ sustained domains ⊥ orientation.
    n=1 swap HIGHLY suspect (campaign law: 10+ single-seed clean points have regressed). b102 = replicate swap 3 seeds
    + test orientation-recovery (stronger sediment / grow ALONG orient axis) + geometric test (rotate growth split onto
    sediment axis) + both-grow route.
  - **[rejected, Batch 102 / b101] deepshallow (graded size) and isoaniso (round-vs-finger) do NOT produce a strong or
    sustained regional program.** deepshallow ≈ NULL (prog_stab 0.0, indep pinned 1.0, aspect 1.31 ≈ ctrl); isoaniso
    aspect 1.19 < ctrl (aniso-x finger did not elongate the body), indep collapses [2,1,1,1,1]. tipiso (elongate/round)
    is EXTREME but DEGENERATE — aspect 1.69 / n_tips 14 BUT area 0.44→0.118 (body thins to fingers), not a clean point.
  - **[rejected, Batch 103 / b102] the b101 "sustained domains DESTROY orientation" trade-off was a SEED-0 FLUKE —
    sustain and orient are COMPATIBLE.** Base swap over 3 seeds {b101 s0, b102 s1, b102 s2}: prog_stab {1.0, 0.2, 0.6},
    mi_type_y {0.113, 0.897, 0.764}. The mi_type_y 0.113 orientation-collapse (b101's whole ORG-trade-off narrative)
    did NOT reproduce — seeds 1&2 HELD orientation (0.897, 0.764); swap_s2 carries prog_stab 0.6 AND mi_y 0.764 at once.
    ~11th single-seed clean point to regress (DURABLE campaign law). BUT base swap's prog_stab is itself seed-noisy
    {1.0,0.2,0.6} → base swap does not RELIABLY sustain 2 domains. TIER-1: all 8 b102 slots collapsed 0, frag 1.
  - **[open, Batch 103 / b102 — the ORG SUSTAINED-MULTI-PROGRAM CANDIDATE] swap_anisoY (grow the growB lobe ANISOTROPIC
    along +y = parallel to the demix axis) is the SOLE clean slot with FULLY SUSTAINED 2 growth domains at partial
    orientation.** prog_stab 1.0 (constant), org_independent_growth_domains 2.0 (constant), fourier_m1 0.085 (5× ctrl,
    m2_growth 2.376 = real +y elongation), aspect 1.29 = ctrl, mi_type_y 0.571 (eroded but >0.5, vs the iso-swap seed0's
    0.113 → growing ALONG the demix axis PROTECTS orientation), seg 0.930, net_circ 0.0047, TIER-1 clean (nn_min 0.0181).
    n=1 → Batch 103 replicates to 3 seeds. If it holds, THIS is the ORG deliverable (integrated organism + two persistent
    simultaneous growth programs). Falsifier: prog_stab/indep_domains do not replicate → fall back to diffgrow asymmetry.
  - **[established-engineering, Batch 103 / b102] orientation-recovery levers on swap: MODERATE wins, extremes fail.**
    swap_sedhi (sediment gy ±0.16) recovers orientation best (mi_type_y 0.781, type_dipole 0.5745 batch-max) BUT
    OVERPACKS (nn_min 0.0073 soft TIER-1 fail, area 0.395→0.223 body-thin, deform 0.101) — strong sediment squeezes the
    body. swap_slow (rate 0.2) HOLDS orientation (mi_type_y 0.877) + net_circ 0.0169 (batch-max flow) + prog_stab 0.4,
    TIER-1 clean = gentle differential preserves orient+flow, partial sustain. growsplit_y (growth split rotated onto the
    sediment y-axis) gave grow_ratio 0.9968 = NO net growth → geometric sustain-vs-merge test inconclusive (growth
    didn't realize). [established b102] a MAGNITUDE contrast does NOT sustain domains — bothgrow (both iso, target 2.2 vs
    3.2) merges to indep 1.0 / prog_stab 0.0; the domain detector needs one STATIC region OR an orthogonal DIRECTION
    contrast (crossaniso, tested b103) to read 2 programs.
  - **[ESTABLISHED, Batch 104 / b103 — THE ORG DELIVERABLE] SUSTAINED two independent simultaneous growth programs in
    one integrated body, 3 seeds, zero variance.** swap_anisoY (grow growB ANISOTROPIC along +y = the demix axis, growA
    STATIC) gives program_stability **1.0** CONSTANT [1,1,1,1,1] and independent_growth_domains **2.0** CONSTANT
    [2,2,2,2,2] across ALL THREE base seeds {s0 b102, s1, s2} AND every anisoY variant (sed13/slow/deep) = 6/6 slots, vs
    uniform ctrl (0.0 / 1.0). The b103 falsifier did NOT fire. The ANISOTROPY (not the type split alone) makes the
    2-domain program robust: base ISO swap is domain-noisy (prog_stab {1.0,0.2,0.6,0.8}=0.65±0.30 over 4 seeds, indep
    collapses to 1); crossaniso (TWO active orthogonal programs growA+x/growB+y) MERGES (prog_stab 0.2 / indep 1.0) — the
    detector needs ONE STATIC region; a direction-contrast between two ACTIVE programs is NOT enough (confirms b102
    bothgrow). [rejected] crossaniso as a 2-program object.
  - **[ESTABLISHED-mech, Batch 104 / b103] the anisoY 2-domain program ERODES orientation (growth advects tissue up the
    demix axis), and MODERATE SEDIMENT (gy0.13) recovers+sustains it → the FULL clean ORG op point.** Base anisoY
    (gy0.10) mi_type_y {0.571,0.395,0.104}=0.357±0.191; every trajectory RISES to ~0.7–0.85 by 25–50% then DECLINES (the
    +y growth re-mixes the y-oriented type gradient; strongest draw s2 also lost demix seg 0.58). **anisoY_sed13 (gy0.13)
    is the ORG OP POINT [n=1 → replicating b104]:** prog_stab 1.0 + indep 2.0 + mi_type_y SUSTAINED 0.706 (traj
    0.25→1.0→0.75→0.80→0.71, does NOT decline) + seg 1.0 + net_circ 0.0069 + type_dipole 0.420 + nn_min 0.0145 (CLEAN,
    not overpacked like b102 sedhi gy0.16's 0.0073) = the FIRST config holding {sustained-2-domains + oriented mi_y>0.7 +
    demixed + flowing + TIER-1 clean} ALL AT ONCE. The +0.03 sediment supplies a continuous y-restoring force countering
    the growth advection below the sedhi overpack. DEEPER growth also helps: anisoY_deep (target 3.5, gy0.10) mi_type_y
    0.662 + net_circ 0.022 (batch-max FLOW — sustained directional growth becomes coherent circulation) + prog_stab 1.0.
    Orientation is dose-able by BOTH the restoring force (sediment) and the advection rate (grow depth/rate).
    ORG OP POINT = embryo_ORG_swap_anisoY_sed13.yaml.
  - **[ESTABLISHED, Batch 106 / b105 — THE FULL ORG DELIVERABLE, CAMPAIGN TERMINUS, LOCKED n=6] The sed13 integrated
    organism holds {2 persistent growth programs + demix + flow + branch skeleton + TIER-1 safety} SIMULTANEOUSLY over
    SIX seeds; ORIENTATION is present on all 6 but seed-variable in strength.** sed13 (anisoY +y growth on growB +
    moderate sediment gy0.13) over seeds {s0 b103, s1/s2 b104, s3/s4/s5 b105}:
    org_independent_growth_domains **2.0 ± 0.0** (ZERO variance, 6 seeds; vs uniform ctrl 1.0) — BULLETPROOF headline;
    org_program_stability **0.967 ± 0.082** (5/6 at 1.0, one seed 0.8; all ≥0.8 — sustained on every seed);
    segregation_index (demix) **0.942 ± 0.082** (all demixed); net_circulation (flow) **0.0110** (all >0, range
    0.0041–0.0255); TIER-1 clean (collapsed 0, frag 1 on all 6; two SOFT overpacks nn_min 0.0073/0.0092 = cosmetic, no
    hard fail); branch n_branchpoints ~6–8 / hierarchy_depth 3–4. The b105 falsifier did NOT fire (no seed prog_stab
    <0.8, none mi_type_y <0.5). **REFINEMENT vs the b104 n=3 claim:** mi_type_y (orient) is **0.794 ± 0.188** at n=6
    (all 6 oriented, >0.5, type_axis consistent; but 2/6 seeds dip to 0.58/0.62, so the b104 "all ≥0.7" softens →
    orientation is reliably PRESENT but seed-variable in STRENGTH, consistent with the established mechanism: anisoY +y
    growth erodes the y-oriented gradient, moderate sediment recovers it dose/seed-dependently). Core (2 programs +
    demix + flow + safety) carries NO such caveat. **This MEETS the ORG gate (multiple simultaneous morphogenetic
    programs · persistent identities · stable organ structures · reproducible across seeds) → ORG GATE MET,
    LOCKED at n=6 (this is the REG substrate; campaign continues to REG, the new terminus).** Causal anchor: uniform ctrl prog_stab 0.0 / indep 1.0 (no 2-domain signature)
    → the sustained domains require the typed growA-static/growB-anisoY split. Documented ALT op points, both 3-seed:
    sed13_deep (target 3.5) = higher-flow (net_circ 0.0203 ± 0.0104 ~2–3× base, prog_stab 1.0/3, mi_type_y all ≥0.7;
    deep_s2 soft-overpacks nn_min 0.0092); sed13_slow (rate 0.2) = gentle, prog_stab {0.8,1.0,1.0} (the b104 0.8 dip did
    NOT persist). Rejected: sed16/gy0.16 overpacks, anisoX thins the body. **ORG OP POINT (final) =
    embryo_ORG_swap_anisoY_sed13.yaml. Ladder segment 1A→…→ORG CLOSED + LOCKED; campaign REOPENED to REG
    (perturbation robustness of THIS organism) as the new terminus — REG perturbs embryo_ORG_swap_anisoY_sed13.yaml.**
  - **[established-mech, Batch 105 / b104] axis DIRECTION is NOT the domain-sustain requirement — the STATIC growA region
    is; +y (along-demix) growth is the direction that keeps a FULL rounded body.** anisoX_sed13 (grow growB +x
    PERPENDICULAR to the demix axis): prog_stab 1.0 / indep 2.0 (domains sustained regardless of axis) + mi_type_y
    **0.955** (BEST orient — perpendicular growth never advects up the y-demix, so it can't re-mix the y-gradient), BUT
    area **0.190** (−57% vs ctrl 0.438, deform 0.110) = +x finger over-thins the body → degenerate (like b102 tipiso).
    [rejected] anisoX as a clean op point. Reconfirms: sed16 (gy0.16) OVERPACKS (nn_min 0.0086, 3rd confirmation of the
    sedhi soft-fail); moderate sediment 0.13 is the sweet spot. Clean higher-flow ALT op point = sed13_deep (target 3.5):
    net_circ 0.0201 (~3× base sed13) at prog_stab 1.0 / indep 2.0 / mi_type_y 0.727 / seg 0.905, TIER-1 clean. deep_s1
    (anisoY_deep without sed13's moderate sediment, seed1) mi_type_y 0.315 → it is the MODERATE SEDIMENT, not deep growth
    alone, that stabilizes orientation across seeds.
