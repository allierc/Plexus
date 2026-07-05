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
