# Embryogenesis (active matter × MPM) — knowledge ledger

Cumulative, curated working memory for the agentic loop. CUMULATIVE: add/curate, never erase.
Tags: **[established]** (causal, reproduced) · **[open]** (hypothesis to test) · **[rejected]**
(falsified) · **[engineering]** (about the tooling/metric, not the biology).

---

## Current objective (Phase 1, one line)
Inner-core cell **flow deforms the outer membrane**; cells **never collapse** (a hard minimum
cell-cell distance holds); motion stays bounded by **parameter balance, not the velocity clamp**;
**division** progressively deforms the blastula; cells **keep flowing at high density** (collective
migration emerges); two types **partition** the blastula (e.g. left/right).

## STAGE STATUS (updated Batch 29, 2026-07-03)
- **INTEGRATION (INT-7) — Batch 28 FOUND THE KNEE and essentially DELIVERED the integrated blastula; Batch 29 pushes the last weak
  leg (deform) via the untouched mass lever.** b28 broke the b27 escape↔migration antagonism: **cf0.09 (not cf0.10) is the
  containment knee — it closes escape to EXACTLY 0.000 while KEEPING migration 0.29** (cf0.10 had killed migr to 0.14). And the
  SPEED lever is even better: **move0.11 preserves migration 2–3× better than cf-up** (move0.11 migr 0.38–0.50 vs cf0.10 migr 0.14 at
  comparable escape; move0.10 over-tames to 0.093). TWO STRICTLY-CLEAN (escape 0.000, coll 0, r_max<0.9) full 5-phenomenon points:
  **(A) s1 cf09** = γ120/div0.02/move0.12/cf0.09/mass5e-5/n182 → seg **0.411** (batch-best partition), migr 0.289, deform 0.014;
  **(B) s6 div025_m11** = γ120/div0.025/move0.11/cf0.08/mass5e-5/n200 → seg 0.307, migr **0.380**, deform 0.018, division active —
  **the BEST ALL-ROUNDER and candidate FINAL integrated spec.** The flock's role at n182 REINTERPRETED by the γ0 control (s7): γ0
  gives HIGHER migration (0.374) but LEAKS (escape 0.055) and lower partition (0.223) → **the γ120 flock buys CONTAINMENT +
  partition-preservation at n182, NOT migration** (flow/division advection already supplies polar order). div0.025 lifts deform
  (0.0275 batch-max) but HARD-FAILS containment at move0.12/cf0.09 (escape 0.090, r_max 1.046); clean only at move0.11 (deform falls
  to 0.018) — **deform/containment trade through move-speed.** γ140 damps migration to 0.139 (falsified align-up; γ120 the sharp
  optimum). **The one weak leg left is DEFORM (~0.018 clean); the prime deform lever `agent_to_mpm.agent_mass` has stayed 5e-5 all
  through INT.** Batch 29 sweeps mass 5e-5→7e-5→1e-4 inside the s6 clean envelope (cf0.08/0.09/0.10) to lift deform toward ≥0.025 at
  escape≈0, + a seed2 replicate of s6. `current_stage.txt` = `INT`. INT started Batch 23.

### (superseded) STAGE STATUS (updated Batch 28, 2026-07-03)
- **INTEGRATION (INT-6) — Batch 28 maps the escape↔migration KNEE of the CONFIRMED 5-phenomenon blastula.** b27 REPLICATED the
  b26 s5 point across 3 seeds and settled the integration deliverable: **{stability + partition + division + collective migration}
  COEXIST robustly at n182 (γ120/div0.02/cf0.08/mass5e-5)** — every γ120 seed held seg 0.18–0.28 (≫ n182 floor 0.06), escape≈0,
  coll 0. The **γ120 flock does NOT preserve each seed's seg — it REGRESSES seg toward a ~0.25 attractor** (sharpened seed2
  0.22→0.28, eroded seed3 0.38→0.18; 3-seed γ120 mean 0.246 vs γ0 mean 0.297) — a mild partition-eroder, not neutral, but never
  destroys the partition. TWO clean escape-closers found (cf0.10 → escape 0.0055 no ram; move0.09 → escape 0.000) BUT **both TAME
  the flock → migration collapses to ~0.13** (the NEW open frontier: escape≈0 and strong migration are ANTAGONISTIC at n182 — the
  strong-migration point seed2 s1 migr 0.42 grazes at escape 0.031, r_max 0.97). Division confirmed as the deform source under the
  flock (div0.03 → deform 0.029 batch-max, but re-mixes seg to 0.167; div-threshold ∈(0.02,0.03) HOLDS under the flock). γ90 at
  n182 HARD-FAILED (escape 0.154, r_max 1.05 — over-translates into wall) → γ120 is the containment optimum, low side leaks even at
  n182. Batch 28 sweeps the cf {0.085,0.09} and move {0.10,0.11} frontiers on seed1 to find a knee (escape≤0.01 AND migr≳0.25),
  plus a div0.025 deform-bump arm, a γ140 align-up probe, and a γ0 flock-ablation control. `current_stage.txt` = `INT`. INT started Batch 23.

### (superseded) STAGE STATUS (updated Batch 27, 2026-07-03)
- **INTEGRATION (INT-5) — Batch 27 ROBUSTIFIES the b26 s5 candidate FULL 5-phenomenon integration point.** b26 FALSIFIED its
  monotone-γ hypothesis and settled two things: (1) **flock containment is NON-MONOTONE in γ with an OPTIMUM at γ≈120** (n503/cf0.10
  escape: γ60 0.242 → γ120 **0.032** → γ160 0.107 → γ200 0.165 → γ240 0.107) — a CONTAINMENT WINDOW: γ60 too disordered (shear into
  wall), γ≥160 over-aligns into a coherent TRANSLATING stream (migr 0.28→0.49) that marches into the boundary. (2) The **clean deform
  ceiling is a HARD ~0.03** — both remaining density levers are dead: higher-γ flow adds no grid momentum (all deform≥0.04 escape-FAIL)
  and mass7e-5 under γ120 re-leaks (escape 0.21) → "deform ≥0.04 CLEAN at n503" is UNREACHABLE; ADOPT s1 (γ120/n503/cf0.10, deform 0.031
  @ escape 0.032) as the density INT point. **THE NEW LEAD (b26 s5): a STRONG flock γ120 did NOT re-mix the minimal-division partition**
  — div0.02/n182/cf0.08/γ120 → seg **0.274** (≈ frozen 0.294, PRESERVED) at escape 0.0275, r_max 0.982 (no cell outside), collapsed 0,
  migr 0.24: the FIRST candidate FULL 5-phenomenon blastula {stability + partition + division + migration}, only deform low (0.022).
  A coherent γ120 recirculation preserves the L/R seed where a disordered γ60 shear (b24) re-mixed it. Batch 27 REPLICATES it on seeds
  2/3 (paired vs γ0 controls), tries to close its grazing escape (cf0.10, move0.09), maps the div-threshold under the flock (div0.03),
  and brackets the containment window low side (γ90). `current_stage.txt` = `INT`. INT started Batch 23.

### (superseded) STAGE STATUS (updated Batch 26, 2026-07-03)
- **INTEGRATION (INT-4) — Batch 26 pushes the FLOCK-CONTAINMENT axis (coherence γ) to lift the clean deform ceiling at n503.**
  b25 INVERTED its own premise and settled the containment mechanism: **STRONG flocking (γ120) CONTAINS at confluence, MILD (γ60)
  RAMS.** At n503/cf0.10, γ120 escape **0.032** (rescued the no-flock div06 0.092) vs γ60 escape **0.242** (worse than no-flock);
  γ120 migr 0.28-0.50 (HIGH, no jam) vs γ60 migr 0.096 (dead). A COHERENT flock advects as an organized recirculating stream off
  the boundary; a half-ordered γ60 flock is a disordered shear that drifts cells into the shell. The confluent flocking peak sits
  at γ≥120 (inverts sparse-n γ60>γ120). BUT the **clean deform ceiling stayed ~0.03** — the two deform≥0.045 slots (γ60 s3, cf0.08
  γ120 s7) BOTH escape-FAIL; cleanest flock deform caps at 0.031 (s1). Flock contains but scatters no extra grid momentum. Also:
  **cf0.10 REQUIRED at n503+flock** (cf0.08 s7 leaks 0.119 vs cf0.10 s1 0.032); **div-rate re-mix threshold ∈ (0.02, 0.03)** —
  div0.02 (s5) is a CLEAN dividing PARTITIONED blastula (seg 0.294, escape 0, coll 0, n182), and div0.03+move0.09+cf0.12 (s6)
  preserves seg 0.311 + rescues containment (escape 0.004) at minor collapse cost (0.0085, cf0.12 rams). Batch 26 sweeps
  γ120/160/200/240 at n503/cf0.10 (does more coherence lift clean deform past 0.03?) + a mass-7e-5 probe under γ120 (the other
  deform lever) + the partition-vs-flock explore (div0.02+γ120). `current_stage.txt` = `INT`. INT started Batch 23.

### (superseded) STAGE STATUS (updated Batch 25, 2026-07-03)
- **INTEGRATION (INT-3) — Batch 25 pushes the DIVISION+FLOCKING clean-coexistence point (b24 s4) to its deform ceiling.**
  b24 (division done right via `agent.div_rate`) settled the core integration question: **division IS a deform source inside
  the partitioned blastula** (deform 0.016→0.049 as n grows 120→503, monotone, 1C law holds), **but division RE-MIXES the
  partition** (seg 0.339→~0.11-0.15 across all div slots, seed2 →0.073 = floor) because proliferation generates outward
  flow/stir (migr rose 0.12→0.22). So the 1E partition is antagonistic to EVERY active process (flocking b23 + division b24).
  The STANDOUT: **s4 = division + mild flocking γ60 → {stability + deform 0.028 + division n235 + migration 0.35} coexist
  CLEANLY (escape 0.013, collapsed 0)** — the integration deliverable minus partition. Also: cf0.08 beat cf0.10 for deform at
  n374 (s3: 0.027 vs 0.021, still contained); div06 n503 no-flock escape-FAILS (0.092) — the 1C cf0.10 flux ceiling; move0.09
  PRESERVED seg (0.345) but escape-FAILED (0.128, inverting the move/escape rule — slow dividing cells pile up locally).
  Batch 25 uses FLOCKING AS CONTAINMENT (1D: flock coherence contains at density) to rescue the div06 escape-fail and reach
  deform ≥0.04 CLEAN; probes the div-rate re-mix threshold (div0.02) + one cf-up rescue of the dividing-partition (s7).
  `current_stage.txt` = `INT`. INT started Batch 23.

### (superseded) STAGE STATUS (updated Batch 24, 2026-07-03)
- **INTEGRATION (INT-2) — Batch 24 re-runs the division arm that b23 SILENTLY SKIPPED.** b23 (reading it now) found the
  central integration test never ran: `cell_divide.rate` is an INERT override when the spec carries a per-type `div_rate`
  buffer (the operator reads that buffer first — cell_divide.py:50), so all b23 division slots ran with div OFF (n stayed
  120; s0≡s3 and s2≡s4 byte-identical). The correct key is **`agent.div_rate`** (tune.py:48 broadcasts to every type).
  What b23 DID measure (flocking, true n120): flocking lifts migration at n120 (γ60 migr 0.33 > γ120 0.26 > frozen base
  0.12 — γ60>γ120, the non-monotone/bistable migration of 1D); flocking-stir re-mixes the partition SEED-DEPENDENTLY (γ60
  seed1 held seg 0.48, seed2 re-mixed to the mixed floor 0.046); cross-rep −0.5 still adds nothing to partition vs flocking
  (seg 0.249 < γ120-alone 0.277 — chemical route dead a 4th way) BUT is a clean CONTAINMENT lever as a side-effect (escape
  0.050→0.000). Batch 24 tests the real integration claim: does DIVISION (agent.div_rate 0.03-0.06, n→250-480, cf0.10) add
  deform while PRESERVING the partition (daughters inherit type + spawn local)? `current_stage.txt` = `INT`. INT started Batch 23.
- **1E — CLOSED (Batch 23). Delivered as a FROZEN (kinetically-maintained) seeded partition; chemical maintenance route DEAD.**
  Batch 22 (reading b22) ran the corrected STIR / re-mixing test and settled 1E: raising move_speed FINALLY re-mixed the no-force
  control (seed1 seg m12 0.339 -> m18 0.063 -> m24 0.078, seed2 m24 0.053 = n120 mixed floor ~0.07) — the headroom absent all
  campaign is real, stir erases the seed. BUT the -0.5 cross-rep STILL did not hold above the re-mixed control: paired Delta-seg =
  +0.004 (m18 s1) / -0.011 (m24 s1) / +0.060 (m24 s2), mean ~0, sign-flips; -0.75 over-drove seg to 0.032 < control. Even the
  cleanest test (m18, control re-mixed to floor at escape 0.008) showed the force adds nothing. AND move0.24 broke containment
  (3 of 5 slots escape >=0.10) -> move0.18 is the clean-stir ceiling at n120/cf0.08. DECISION (pre-committed): control re-mixed but
  force did NOT hold -> ADOPT the frozen seeded split (`embryo_1E_split_hin`, n120, cf0.08, move0.12, no force -> seg ~0.34 @ escape
  0.042, r_max 0.96, collapsed 0) as the 1E operating point and ADVANCE TO INTEGRATION. 1E started Batch 13, closed Batch 23 (11 batches).
- **INTEGRATION (INT) — STARTED Batch 23.** Goal: combine the five delivered rung operating points (1A stability, 1B membrane
  deform, 1C division deform, 1D flocking migration, 1E seeded partition) into ONE blastula and characterize which phenomena
  COEXIST at escape=0. Core tension (from the campaign): the seeded partition is FROZEN (holds only because interdiffusion is slow
  at move0.12); any active STIR (flocking, higher move_speed) re-mixes it, and NO force maintains it (chemical route dead). Division
  may be the one ingredient that adds deform WITHOUT re-mixing (daughters inherit parent type + spawn local -> preserve/sharpen each
  side). Batch 23 maps: frozen-partition base + {flocking gamma60/120, division 0.08, mid-mass, cross-rep vs flocking-stir}.
  `current_stage.txt` = `INT` (archive dirs `embryo_INT_b<NN>_<slot>`).

### (superseded) STAGE STATUS (updated Batch 22, 2026-07-03)
- **1E — two-type partitioning: THE CHEMICAL CROSS-REP ROUTE IS FALSIFIED AS A ROBUST MAINTENANCE FORCE (Batch 22 reading b21).**
  The b21 3-seed paired test at n120/cf0.08 flipped sign AGAIN: Delta-seg = seg(-0.5)-seg(gain0) = **-0.062 / +0.182 / -0.200**
  (seeds 1/2/3), mean **-0.027** — no consistent maintenance (over n44+n120 -> 6 realizations, mean ~ 0). CRUCIALLY the b20 premise
  also died: the n120 no-force **controls did NOT re-mix** (seg 0.339 / 0.250 / 0.550, all >> mixed floor ~0.07); seed-0's 0.036 was
  the outlier. So the "genuine force test" (a re-mixing control) was NEVER met — the root cause of ALL 1E ambiguity (n44 AND n120) is
  that **interdiffusion is too slow -> the seed persists passively -> NO headroom for any force to demonstrate maintenance.** All 8
  b21 slots escape EXACTLY 0 (cf0.08 containment robust); one minor blip (xrep_s3 collapsed 0.0167). **NEW positive [established]:
  a seeded L/R split is intrinsically STABLE — with NO force it stays partitioned over 12000f (seg 0.25-0.55) — a passive/frozen
  partition.** DECISION: chemical route CLOSED. Batch 22 runs ONE corrected test — raise STIR (move_speed 0.12->0.24) to force the
  control to re-mix, then ask if -0.5 cross-rep holds above it (paired, 2 seeds). If a stirred control re-mixes AND the force holds
  (Delta-seg>0) -> 1E delivered as active-re-mixing-resistant maintenance; ELSE adopt the frozen-seeded-split (`embryo_1E_split_hin`,
  cf0.08, no force) as the 1E operating point and ADVANCE to INTEGRATION. Stage started Batch 13; 9 batches in — LAST 1E batch either way.

### (superseded) STAGE STATUS (updated Batch 21, 2026-07-02)
- **1E — two-type partitioning: THE n44 −0.5 "PEAK" IS FALSIFIED AS ROBUST (Batch 21 reading b20) — it was a LUCKY SEED-0 draw.**
  The b20 paired 3-seed test flipped the sign on ALL THREE new seeds: n44 Δseg = seg(−0.5)−seg(gain0) = **−0.31 / −0.05 / −0.08**
  (seeds 1/2/3), vs seed-0's **+0.32**. Mean of the new seeds −0.146 → the −0.5 cross-rep does NOT maintain the n44 partition; it
  slightly HURTS it. Cause: at n44 interdiffusion is too slow — the no-force control retains 0.42–0.64 of the seed over 12000f, so
  there is no headroom for the force, and seed-0's control was just anomalously low (0.29). The "clean unimodal bell curve" is a
  single-realization artifact. **The ONE survivor: at n120 + cf0.08 the no-force control genuinely RE-MIXES (seg 0.036 < floor
  ~0.07 = mixed) while the −0.5 force HOLDS a partial partition (0.20, escape 0 — cf0.08 fixed the b19 leak) → Δ +0.165.** That is
  a GENUINE force test (control erases the seed), but it is a SINGLE seed-0 draw — the exact trap that just misled us at n44.
  **Batch 21 is the CLOSER: replicate n120/cf0.08 across seeds 1/2/3 (paired) + map the density dose-response (−0.35/−0.5/−0.75 at
  seed1). DECISION: if Δseg > 0 consistently (mean ≳ +0.1, controls re-mixing to near floor) → 1E DELIVERED as a DENSITY-dependent
  MAINTAINED partition (chemical cross-rep resists active re-mixing where the control cannot hold the seed); adopt
  `embryo_1E_split_hin` + cf0.08 + gain −0.5; begin INTEGRATION next batch. If Δseg is inconsistent/≤0 → chemical 1E route
  FALSIFIED (n44 lucky + n120 lucky); log 1E [open], adopt best-clean (the seeded split is itself clean/stable), advance off 1E.
  Stage started Batch 13; 8 batches in, ~40 headroom — but this is the LAST 1E batch either way (breadth rule).**

### (superseded) STAGE STATUS (Batch 19, 2026-07-02)
- **1E — two-type partitioning: FIRST CLEAN SIGNAL. The seed fix worked (b18) and the chemical route DOES maintain a
  seeded partition. MILD cross-rep (−0.5) HOLDS the L/R seed (seg 0.61 @ escape 0, collapsed 0) — 2× the no-force
  diffusion control (0.29) and ≫ the mixed-start floor (0.094). STRONG cross-rep (−1.0/−2.0) OVER-DRIVES and
  homogenizes to the floor (0.13–0.16, BELOW control). Self-agg from a seed still HARD-collapses (0.80). Differential
  adhesion (self+0.1/cross−1.0) is clean and holds seg 0.47. So: (i) cross-rep can MAINTAIN but not CREATE a partition
  (mix_xrep inert reconfirms b14–16 — symmetry-break is the missing ingredient, and the seed supplies it); (ii) the
  effect is NON-MONOTONE with a peak at MILD gain. Batch 19 MAPS the peak (n44 sweep 0/−0.25/−0.5/−0.75/−1.0,
  reproduce both peak 0.61 and over-drive 0.16) and tests DENSITY (mild −0.5 at n120 vs its own diffusion control) +
  self-cohesion+mild combo. If the peak reproduces and holds at density → ADOPT `seed_xrep_lo` (−0.5) as the 1E
  operating point (a maintained, seed-dependent partition), log the "cannot spontaneously create" limit as [open], and
  Stage 1E is DELIVERED (breadth rule — advance). Stage started Batch 13; ~34 batches of 1E headroom remain.**

### (superseded) STAGE STATUS (Batch 18, 2026-07-02)
- **1E — two-type partitioning: THE SEEDED-SPLIT EXPERIMENT HAS STILL NOT RUN — `type_layout: split_x` was BROKEN in b17
  (SECOND whole-batch engine bug, cf. b13 YAML). All b17 seeded slots (s0–s6) put EVERY live cell into ONE type
  → seg EXACTLY 0.0, s0–s3 byte-identical (cross-rep inert, no type b), one render colour, s4 self-agg collapsed 0.86 (single
  type Keller-Segel). ROOT CAUSE: `_assign_types` split the type fractions over `lvl.n` = BUFFER size (3000), not the live
  count (44); for `split_x`, argsort over the buffer sorts the 2956 dead slots (x=0) first, so they swallow the low-x types and
  all 44 live cells (disc x∈[0.2,0.8]) land in the last type. FIXED this batch (engineering note below): sort+split over LIVE
  indices/count only. Batch 18 RE-ISSUES the b17 design unchanged (specs were correct). The decisive test — does a chemical
  force MAINTAIN/SHARPEN a genuinely seeded split vs a re-mixing no-force control? — is finally testable. Only the mixed-start
  control mix_xrep had two real types in b17 (seg 0.094, mixed — re-confirms cross-rep inert from a mixed start, b14–16).
  Stage started Batch 13; ~35 batches of 1E headroom. If even the fixed seeded split re-mixes under the force → abandon the
  chemical route, adopt best-clean point, log 1E [open], advance.**

### (superseded) STAGE STATUS (Batch 17, 2026-07-02)
- **1E — two-type partitioning: CHEMOTAXIS-FROM-A-MIXED-START ROUTE EXHAUSTED (Batch 17, reading b16). THREE independent
  falsifications (b14 cross-rep inert · b15 self-agg collapse + cross-rep inert · b16 COMBINED = collapse-or-inert). The
  combined "differential adhesion" wiring is SQUEEZED between "too weak self → stays mixed" and "strong enough self →
  Keller-Segel collapse": at n44 the primary (self+0.1/cross−1.0) HARD-FAILED (collapsed 0.114, escape 0.114, nn_min 0.0009 —
  the self term crosses the stacking floor once cross co-locates a local excess); the clean combos (lo/hi/sharp) all pinned seg
  at the n44 noise floor (0.11–0.16, montage mixed); at n120 the self term drove CATASTROPHIC collapse (0.492, seg 0.020). The
  two seg>2σ readings were artefacts (s2 = collapse; s7 pure-xrep 0.25 = noise on a mixed montage). The blocker is definitively
  SYMMETRY BREAKING, not force/field/wiring. Batch 17 makes the pre-authorized engine move: `type_layout: split_x` SEEDS a
  left/right split (a=left, b=right) + `mpm_spin omega 0` (no rotation smear), and tests whether pure cross-rep / self-agg /
  combined MAINTAIN or SHARPEN the seed vs a re-mixing no-force control (seed_ctrl) and the same force from a mixed start
  (mix_xrep). This decouples the force-test from the symmetry-break problem. Stage started Batch 13; ~40 batches of 1E headroom.
  If even a seeded split re-mixes under the force → abandon the chemical route, adopt best-clean point, log 1E [open], advance.**

### (superseded) STAGE STATUS (Batch 16, 2026-07-02)
- **1E — two-type partitioning: BOTH PURE chemotaxis wirings FALSIFIED (Batch 16, reading b15) — same root cause (mixed
  co-located start). Combined self-agg+cross-rep (differential adhesion) is the Batch-16 primary.** With the field now ACTIVE
  (the b14 inert-field bug is fixed — self-agg produced strong clumping), b15 showed NEITHER pure wiring demixes: (a)
  SELF-AGGREGATION (gain>0) → **Keller-Segel COLLAPSE** (collapsed 0.75 @ n44 gain 0.3/1.0, **0.99 @ n120** — both types climb
  their own trail from a co-located start → collapse to ONE shared central knot; chemotactic self-attraction BEATS repel,
  nn_min→0; a HARD FAIL, NOT a demix); (b) CROSS-REPULSION (gain<0) → **still INERT** (seg 0.025–0.12 within noise, montage
  mixed at every |gain| incl −3.0 — from a mixed start the two channels are co-extensive so "flee the other" has no gradient;
  symmetry-break problem, now doubly confirmed b14+b15). Containment clean (n44 escape 0; n120 escape 0.0167). Both established
  below. **The missing ingredient is a symmetry-BREAKING term: within-type cohesion AND between-type repulsion together (chemical
  differential adhesion) — the last untested chemotaxis wiring.** Batch 16 runs both (weak self +0.1 below the collapse floor +
  strong cross −1.0) — a two-species cross-repulsive Keller-Segel/Cahn-Hilliard demixing instability that self-amplifies a local
  type excess where each pure wiring could not. Stage started Batch 13; still NO confirmed partition (~44 batches of 1E headroom).
  If combined ALSO stays mixed → next either SEED a split (needs engine `type_layout: split_x`; types are currently assigned by
  random perm so no seeded split is possible from a spec — see engineering note) to test whether the force can MAINTAIN a
  partition, or switch to a type-pair-aware differential-adhesion operator; else adopt best-clean point, log 1E [open], advance.

### (superseded) STAGE STATUS (Batch 14, 2026-07-02)
- **1D — high-density flow / collective migration: DELIVERED + CLEAN, Batch 13 (reading b12). ADVANCED to 1E.** TWO adopted
  operating points: **(A) `polar120`** = embryo_1D + `polar_align.gamma 120` (n557, noise0.1, move0.12, cf0.10) → migr **0.49**
  @ escape **0.020**, deform 0.047, collapsed 0.011, r_max 1.06 (strongest migration, near-clean — escape 0.020 REPRODUCED
  across b10 AND b12, so NOT a lucky draw; the b12 "escape plateaus 0.07" caveat is OVERTURNED). **(B) `polar120_m09_cf12`** =
  + `move0.09 mpm_to_agent.confine 0.12` → migr **0.35** @ escape **0.004** (STRICTLY CLEAN, r_max 0.906) — the residual-escape
  [open] is now CLOSED. Migration is non-monotone in γ with a PEAK at γ≈120 (γ100 0.43, γ120 0.49, γ140 0.25). Stage started
  Batch 10, delivered Batch 13.
- **1E — two-type partitioning: STARTED Batch 13, but Batch 13 PRODUCED ZERO DATA (spec YAML bug) — Batch 14 is the real
  first attempt.** All 8 b13 jobs died at load with `yaml.parser.ParserError` on the UNQUOTED per-type selector
  `at: agent[type=a]` inside a `{...}` flow mapping (see engineering note below); no sim ran, no b13 montage/archive/metrics
  exist. Fixed both specs (`'agent[type=a]'`) and re-issued the identical design as Batch 14. Mechanism under test (unchanged):
  per-type chemical cross-repulsion (`deposit` per-type channel + `diffuse`+`decay` + two per-type-selector `chemotaxis` each
  fleeing the OTHER type's channel). Base `embryo_1E.yaml` = clean sparse 1B point (n44, div OFF, cf0.05, mass 5e-5, polar_align
  γ0 OFF) + chemical field. Target: seg = |⟨x⟩_a−⟨x⟩_b|/R ↑ (left/right Janus split) at escape≈0. STILL no physics results.
- **[engineering, Batch 14] BRIGHT LINE — QUOTE per-type selectors (`at: 'agent[type=a]'`) in every spec.** In a YAML flow
  mapping `{op: ..., at: agent[type=a], ...}` the `[` is a flow-sequence indicator, so the bare selector is a HARD parse error
  that kills the whole batch at load (0 data — the most expensive failure mode, an entire L4 batch wasted). ALL working
  multi-type specs quote it (`agent_mpm_disc_4types.yaml`, `agent_mpm_blastula_4types.yaml`). Cost me all of Batch 13. Engine
  plumbing (verified b14): engine sets `_at`=base set name (`agent`) and applies the `agent[type=a]` mask separately
  (`engine.py:430,457`, `_selector_mask` = live & node_type==idx), so quoting fixes it with no other change. `chemotaxis` op
  (`plexus/operators/chemotaxis.py`) reads `from`/`gain`/`channel`/`noise`, emits velocity `gain·grad` (gain<0 flees); `deposit`
  writes channel=`node_type` (per-type trail real). Add a `python -c "import yaml; yaml.safe_load(open(spec))"` check to any new
  spec before submitting a batch.

### (superseded) STAGE STATUS (Batch 12, 2026-07-02)
- **1A — stable, no collapse: MET.** Recipe = `mpm_to_agent.confine 0` (see established block). collapsed=0,
  escape=0, nn_min≥r0, accel bounded by balance (not vmax), at n=44 AND confluent n=265. Base = `embryo_1A.yaml`.
  *Caveat (Batch 2):* at RUNAWAY division to n=1600 the disc is OVER-confluent — natural spacing ~0.015 < r0=0.02,
  so nn_min pins ~0.002 (repel can't hold packed cells apart). collapsed stays ~0 (those are daughters); but
  keep n bounded (cap `agent.div_rate`) to stay a true Stage-1A tiling with nn_min≥r0.
- **1B — inner flow deforms membrane: MET (CLEAN, Batch 9). Operating spec = `embryo_1B.yaml`.** `mpm_to_agent
  {confine 0.03, field: colour}` + `agent_to_mpm.agent_mass 5e-4` at n=44/move 0.12/12000f/div OFF gives **deform 0.0304
  (≫0.02), escape 0.000, r_cell_max 0.850 (all cells well inside the shell), collapsed 0.** cf0.07 is equally clean
  (deform 0.0332, escape 0). The clean confine window is a PLATEAU across [0.03, 0.07] (not the U-min I once read — the
  b07 cf0.05 escape 0.0227 was a one-cell blip); the ram onsets at 0.10 (escape 0.386). This is a ROBUST (12000-frame)
  clean Stage-1B, not a finite-time illusion. *nn_min caveat:* the confine catch compresses the closest pair to
  ~0.004–0.007 < r0=0.02 (still > the collapse floor 0.003, collapsed 0) — near-clean tiling, close-approach not stacking;
  cf0.03 keeps nn_min highest (0.0073). *Mechanism established below.* Stage started Batch 2, MET Batch 9.
- **1C — division pressure deforms shell: MET (Batch 10). Operating spec = `embryo_1C.yaml`.** Division proliferation reshapes
  the shell — deform rises monotonically with n at fixed low mass: 0.011 (n44) → 0.027 (n442) → 0.063–0.079 (n2700); montage
  goes round → lobed → amoeboid. **Two adopted operating points:** `div_ref` (n442, cf0.10, m5e-5, div0.10, ω0.3) = deform-max
  clean-ish (**deform 0.0268 @ escape 0.0158, r_max 0.954 <1.0** grazing not blowout, collapsed 0), and `div_spin06_slow`-style
  (n311, div0.08, +ω0.6) = **STRICTLY CLEAN (escape EXACTLY 0.000, r_max 0.867, deform 0.017)**. **Clean deform ceiling at density
  ≈0.027** — a real physical ceiling (fixed-capacity colour-confine catch vs n-scaling boundary flux), NOT a tuning miss: every
  lever to lift it failed (Batch-9 spin INVERTED at density — see established block; mid-mass uncontainable; confine-up rams).
  Negatives (all established): confine CANNOT be scaled up to contain density (cf0.10 is the ceiling at n442; cf0.15→ram-collapse);
  big per-cell push uncontainable (cf0.20/m5e-4 → escape 0.523); cap div_rate ≤0.10 (div0.20 → n2700, leaks). Stage started
  Batch 8, MET Batch 10. ADVANCED to 1D (breadth rule; s6 gate escape 0).
- **1D — high-density flow / collective migration: PHENOMENOLOGY DELIVERED, escape NOT strictly closed (Batch 12, reading b11).
  Adopted operating point = `polar120` (embryo_1D + polar_align.gamma 120, n557, noise0.1, move0.12) → migration ~0.49, flow
  ~0.0058 (flowing, NOT jammed), escape ~0.05–0.07 (grazing, r_max ~1.06), collapsed ~0.01.** EMERGENT flocking gives strong
  collective migration at confluence and is far cleaner than imposed spin (escape ~0.07 vs 0.26). **REVISED by b11: the b10
  "monotone γ↑ → migration↑ AND escape↓, escape→0" story is NOT robust** — migration is a noisy near-bistable order parameter
  (γ80→0.47, γ200→0.37, γ300→0.52; scatter, not monotone) and escape at every strong-migration n557 point PLATEAUS at
  0.066–0.077 (the b10 γ120 escape 0.020 was a lucky single draw, not reproduced). The only escape<0.005 points (move0.06,
  γ200_n311) have DEAD migration (~0.06–0.09). See the three revision blocks below (noise~0.1 required; migration NOT
  speed-independent; containment DENSITY-DEPENDENT). Batch 12 tests the last lever — a small boundary-confine bump (0.10→0.12)
  a coherent flock may tolerate — else ADOPT polar120 as-is (residual escape [open]) and ADVANCE to 1E. Stage started Batch 10.
  1E partition — not yet attempted (do not chase early; seg≈0 with no partition mechanism, expected).
  *(Stage-1B blocker history, condensed — for the record:)* Batches 5–6 found escape was a slow time-accumulating
  ballistic leak (r_max up to 2.09) with NO existing agent-force to contain it: drag `k` INVERTED (raising it worsened
  escape 0.136→1.000, see established block), `g2p.wall_contact` inert (acts on material not agents), move_speed the
  dominant co-driver (0.12→0.06 cut escape 6×). The Batch-4 "sparse-n escape shield / anti-diagonal frontier" was a
  FINITE-TIME (3000f) artefact — the same regime leaks at 6000–12000f. The FIX (Batch 7→9) was the boundary-only
  colour-confine below, none of those. Spin BREAKS a marginal catch (b07: cf0.05+ω0.6 → escape 0.023→0.273) — do not
  stack spin on a margin.

## The system (what exists)
A **blastula**: a thin elastic **membrane** shell (deep blue) enclosing a **water core** (light
blue), held by a **substrate anchor** so the shell contains the fluid and nothing drifts. **Cells**
= active-matter agents living in the core; they are dragged by the fluid, confined to the core, and
deform the membrane. Rendering: cells = coloured dots by type; material = blue (two blues for
membrane vs core); black background. Every run also emits the **2×2 mp4** (cells+material / stress /
deformation / cell tracks) and a VLM caption.

## Operators (the agent's action set — compose freely from the whole codebase)
NEW couplings (src/plexus/operators/):
- `agent_to_mpm` — cells scatter momentum onto the MPM grid → **deform** the material. [established]
- `mpm_to_agent` — grid velocity **drags** cells (`k`) + **confines** them up a density/colour
  gradient (`confine`, `field: mass|colour`). [established]
- `mpm_spin` — drive the disc toward slow solid-body **rotation** (`omega`). [established]
- `flow_align` — cell polarity relaxes toward the local **flow** (SPV polarity–velocity rule). [established]
- `agent_remodel` — cells **soften/rigidify** the tissue (μ,λ) via per-type `remodel_rate`. [open — untested in blastula]
- `cell_divide` — **proliferation** on a fixed `buffer` via occupancy; per-type `div_rate`. [established]
Reused active-matter ops: `repel` (hard-core min distance `r0`), `attraction_repulsion`
(equilibrium spacing; `p=[pull,pull_range,push,push_range]`), `polar_align`, `chemotax`/`deposit`/
`diffuse`/`decay`/`relay`/`adapt` (chemical signalling), `separation` (boids even_spacing),
`glide`. MPM: `mpm_strain/p2g/mpm_grid_update/g2p`, `mpm_anchor`, `mpm_drag`.
Even initialisation: `spawn: sunflower` (Vogel golden-angle lattice — cells start equidistant).

## Established mechanisms
- **[established]** Two-way agent↔MPM coupling routes through the shared grid: cells push the grid
  (agent_to_mpm) and are dragged/confined by it (mpm_to_agent); this gives genuine mutual
  deformation. Reused by every spec.
- **[established]** A **bounded substrate** = elastic body + `mpm_anchor mode:substrate` + density/
  colour confinement → 0 escape, no drift over 1500 frames (disc_growth≈0, aniso≈0.001). A free
  liquid disc instead slowly volume-drifts and sprays its skin — so anchor the body.
- **[established]** **Even coverage recipe**: `spawn: sunflower` + hard `repel` with `r0 ≈` the
  confluent spacing keeps a stable, uniform, non-clustering tiling (min NN held at `r0`, 0 stacking).
- **[established, 2026-07-02, Batch 2] CONFINEMENT IS THE COLLAPSE DRIVER — H5 confirmed by crossed
  ablation.** `mpm_to_agent {confine, field: colour}` adds `+confine·grad(colour)` pushing every cell
  inward up the material-colour gradient; that centripetal drift, not any force, stacks cells. Crossed
  ablation at n=44: `drag0` (k=0, confine 3.0) → collapsed=0.568 (unchanged from base); `confine0`
  (confine=0, k on) → collapsed=**0.000**, nn_min=**0.0291 > r0**, escape=0. Collapse tracks `confine`,
  NOT drag `k`. Dose-response is a THRESHOLD not a line: 3.0→1.0→0.5 barely moves it (0.568→0.523→0.477),
  0.5→0 crashes to 0 — critical confine sits in (0, 0.5). **Stage-1A recipe: set `confine 0`** (substrate
  anchor + elastic membrane + density confinement already retain cells — escape stayed 0, r_cell_max even
  dropped). This is the `specs/embryo_1A.yaml` operating point.
- **[rejected→corrected, 2026-07-02, Batch 2] "COLLAPSE ∝ DENSITY" was a CONFOUND.** The Batch-1
  density law (n=44→0.568 … n=265→0.98) held ONLY because every one of those runs had confine=3.0.
  With `confine 0`, n=265 division-ON → collapsed=**0.000**, escape=0 (`confine0_dense`). Density does
  NOT cause collapse. Corrected law: **confinement causes collapse; density AMPLIFIES confine-driven
  collapse.** Remove confine and confluence is collapse-free. (Turning division off still lowers n and
  was the biggest Batch-1 lever only because it reduced the confine-amplifying density.)
- **[rejected, 2026-07-02] H4 — active ops (`flow_align`/`glide`) cause the collapse.** Batch-1 s0
  (flow_align.gain 0), s2 (move_speed 0) and s3 (mass 0, k 0) EACH left collapsed at 0.96–0.985,
  indistinguishable from the 0.977 baseline. None of the active operators nor the passive coupling
  drives collapse. *Caveat: these three ran with division ON (n=265), so density could mask the
  effect; Batch 2 re-tests flow_align/glide at fixed n=44 (s5/s6) to close this cleanly.*
- **[rejected, 2026-07-02] Passive agent↔MPM drag causes collapse.** Both low_k (Batch-0) and s3
  no_couple (mass=k=0, Batch-1) left collapse untouched. The drag `k` is exonerated. *(The historical
  "hydrodynamic self-attraction / coupling-MIPS" story is fully rejected for the DRAG channel.)*
- **[resolved→established, H5, Batch 2] Confinement WAS the collapse driver — see "CONFINEMENT IS THE
  COLLAPSE DRIVER" above.** Crossed ablation settled it: collapse tracks `confine`, not drag `k`; confine 0
  → collapsed 0, nn_min>r0, escape 0, at n=44 AND n=265. Escape-trade risk did not materialise. Open tail:
  the exact critical `confine` in (0, 0.5) is unprobed (transition is a near-switch, not a ramp) — low
  priority now that `confine 0` is a clean Stage-1A point.
- **[established, 2026-07-02, Batch 2] `agent_to_mpm.agent_mass` IS the prime membrane-deform lever — monotone,
  ~15×.** On the confine-0 (Stage-1A) base, deform rises 0.0043→0.0067→0.0526→0.0625 as agent_mass goes
  2e-6→1e-5→5e-5→2e-4 (migration tracks it, 0.22→0.42→0.73). This is the cells→grid push channel: heavier cells
  scatter more momentum onto the grid, displacing the shell. Confirmed the Batch-2 hypothesis direction.
- **[established, 2026-07-02, Batch 2] DEFORM and ESCAPE are CONFOUNDED through agent_mass — big deform at
  mass≥5e-5 is BLOWOUT, not clean reshaping.** escape climbs in lockstep with deform (0.014→0.022→0.146→0.213)
  and r_cell_max exceeds 1.0 (1.22 at 5e-5, 1.27 at 2e-4) — cells are pushed OUTSIDE/through the membrane. So
  raising agent_mass alone cannot pass Stage-1B: the same knob that deforms the shell ejects the cells. Root
  cause (hypothesised, Batch 3 tests it): with `confine 0` there is NO inward force holding cells off the shell,
  and at runaway n=1600 the over-packed core presses cells against the wall; the push then punches them through.
  Stage-1B requires DECOUPLING deform from escape (contain cells while deforming the shell), not a bigger push.
- **[rejected, 2026-07-02, Batch 2] "Softer membrane deforms more" (youngs-limited deform).** youngs 200→80 left
  deform byte-identical at base mass (0.0043) and slightly LOWER at 5e-5 (0.0489 vs 0.0526), same escape. Shell
  stiffness in [80,200] is NOT the deform bottleneck — containment/coupling is. Do not chase membrane softening.
- **[established, 2026-07-02, Batch 2] `mpm_spin` deforms by CLEAN internal flow — best deform-per-escape.**
  omega 0.3→0.6 raised deform 0.0043→0.0069 (1.6×) at the LOWEST escape of the batch (0.011), cells staying
  inside (r_cell_max 0.97). Circulation reshapes the shell from within without ejecting cells — the right *kind*
  of deform (just small). `agent.move_speed` is a poor lever by contrast (raises escape/r_cell_max faster than
  deform). Keep spin as a clean deform amplifier to stack with a contained high-mass regime.
- **[OVERTURNED, Batch 6] "Sparse n is an escape SHIELD" / "escape = push × density (both needed)" — FALSE at
  long time; it was a FINITE-TIME (3000-frame) artefact.** Batch 4 (3000f, move_speed 0.06) read n=44 mass 5e-4 →
  escape 0 and inferred sparse-n shields ejection. At 6000f + move_speed 0.12 the SAME sparse regime ejects hard:
  n=44 mass 5e-4+spin → escape **0.273**; n=44 mass 1e-3 → escape **0.182, r_max 2.09** (ballistic, ~6× shell
  radius); even n=44 mass 2e-4 leaks (escape 0.023). **Escape is a SLOW ACCUMULATING LEAK, not a density-gated
  packing pressure:** given enough time (and faster cells), any cell pushed to the shell eventually punches
  through, regardless of n. So neither "sparse-n shield" nor "needs high density too" survives. Escape scales with
  per-cell push (mass), with observation TIME (frames), and with move_speed — density is NOT required. Corrected
  picture: at `confine 0` there is no boundary force on agents, so escape is gated only by how well drag keeps
  agents on the material flow (see Batch-6 containment hypothesis). *The narrow true residue: capping runaway
  n=1600→95 did cut escape — but because it cut TOTAL push and over-confluent wall pressure, not because sparse-n
  is intrinsically safe.*
- **[established→REVISED, Batch 5] DEFORM grows with BOTH agent_mass AND n — mass IS a lever at fixed n.** Batch 3
  claimed "deform ∝ n×mass, NOT set by mass at fixed n"; that was an artefact of a NARROW mass window (2e-5 vs
  5e-5). Over a wide range at FIXED n=95, deform rises 0.0027→0.0052→0.0105→0.0199→**0.0346** as mass
  2e-6→1e-4→2e-4→5e-4→1e-3 (500× mass → 13× deform, ≈ floor + slope·mass). Deform ALSO grows with n at fixed mass
  (Batch 3: 0.0067@n=95 → 0.0526@n=1600 at 5e-5). So deform is driven by TOTAL scattered momentum, and either
  per-cell mass OR cell number supplies it. Consequence: the decoupling axis WORKS — buy deform from mass at fixed
  n. But deform and escape stay coupled through mass at fixed n (both rise together), so the clean route is sparse
  n (escape shield) + high mass + spin, or dense n + low mass. Same clean deform ceiling ≈0.015 either way so far.
- **[OVERTURNED, Batch 6] "TWO CLEAN Stage-1B routes (anti-diagonal frontier)."** The sparse-n+high-mass route
  (a) is dead — it escaped 0.27 at 6000f (see OVERTURNED block above). The dense-n+low-mass route (b) is UNTESTED
  at 6000f and is re-checked in Batch 6 (dense_6k) — expect it to leak too if escape is purely time-accumulating.
  What SURVIVES: (i) deform grows with total scattered momentum (mass×n) — still holds; (ii) `mpm_spin` amplifies
  an EXISTING mass-driven flow (+40% deform, migration 0.29→0.69 at m2e4) but **does NOT create deform on its own**
  — Batch-6 s6 (ω1.2, floor mass) gave deform 0.0042 = floor. Spin is an amplifier, not a source. The open
  Stage-1B target (clean deform≥0.02) now requires solving CONTAINMENT first, not tuning the mass/n/spin mix.
- **[established, Batch 6] ESCAPE is BALLISTIC EJECTION past the material edge — a slow leak that grows with
  TIME, per-cell push, and move_speed.** r_cell_max reaches **2.09** (≈6× the shell radius ~0.34) on the
  worst slots — cells are not grazing the shell, they fly free of the material entirely. Mechanism: with
  `confine 0` the only inward force on an agent is grid-flow drag `mpm_to_agent.k`, which acts ONLY where grid
  velocity is non-trivial, i.e. inside the material. A big `agent_to_mpm.agent_mass` push flings an agent past the
  material boundary; out there grid-v≈0, so nothing decelerates or returns it → ballistic escape. This is why
  escape accumulates with frames (more chances to be flung out) and with move_speed (faster overshoot), and needs
  no density. Consequence: escape is a CONTAINMENT problem, not a packing problem.
- **[REJECTED & INVERTED, Batch 7, H6] `mpm_to_agent.k` (drag) as a CONTAINMENT lever — it does the OPPOSITE.**
  Raising drag k 0.3→0.6→1.0 at mass 5e-4/n=44/12000f made escape WORSE, monotonically: 0.136→0.523→0.705, and
  k=1.0+mass1e-3 → escape **1.000** (every cell out, deform collapses to 0.012 with no cells left to push).
  Mechanism (operator re-read): drag relaxes each agent's velocity toward the LOCAL grid velocity `k·v_fluid`, and
  at high agent_mass the cell's OWN `agent_to_mpm` push makes grid-v point OUTWARD right where the cell sits (it
  scattered that outward momentum onto g.mv itself). So higher k slaves the agent MORE tightly to the outward jet
  it created → carried out faster. Drag is a coupling to the self-generated outflow, not a restoring force. H6 dead.
- **[established, Batch 7] `confine field:colour` is a BOUNDARY-ONLY force; `field:mass` is a per-cell ATTRACTION —
  the field choice, not the magnitude alone, sets whether confine collapses.** `agent_to_mpm` scatters agent
  momentum onto g.m (mass) and g.mv (momentum) but NEVER onto g.c (colour). So `mpm_to_agent.confine field:mass`
  reads a density field that INCLUDES each cell's own scattered mass bump → cells drift up each other's peaks →
  mutual attraction → collapse (explains Batch-3 s6 confine 0.2 field:mass → collapsed 0.58, and why confine-collapse
  scaled with agent_mass). `field:colour` reads g.c which agents do NOT write: ~1 across the fluid disc, →0 outside,
  so grad(c)≠0 ONLY at the shell interface and ≈0 in the uniform core → a boundary-localized inward catch with NO
  per-cell attraction. Consequence: the collapse at LARGE colour-confine (0.5–3.0, Batch 2) is a boundary RAM (cells
  shoved into a dense inner clump), a magnitude effect independent of mass; SMALL colour-confine (0.02–0.15) should
  catch escaping cells without ramming — the natural boundary containment we lacked. This is the Batch-7 hypothesis.
- **[established, Batch 7] move_speed is the DOMINANT escape co-driver.** Halving move_speed 0.12→0.06 (same
  mass 5e-4/k0.3/n44/12000f) cut escape 6× (0.136→0.023) and r_max 2.06→1.12. Faster cells random-walk to the shell
  and overshoot it more; the user's doubled move_speed (0.12) is a primary reason escape is bad at 12000f. Because the
  directive wants ≥0.12, the fix must be a boundary FORCE (containment), not slowing cells — but 0.06 is a near-clean
  fallback if containment fails (deform 0.0144 @ escape 0.023 at move 0.06).
- **[established, Batch 7] dense+low-mass survives long time FAR better than sparse+high-mass, but still marginally
  leaks.** At 12000f, dense (n→442, div 0.10) + low mass 5e-5 gave deform 0.0312 (>0.02) at escape 0.054 — the best
  deform/escape trade of Batch 6 — but r_max 2.09 (one ballistic cell) makes it a marginal HARD FAIL. Adding small
  boundary confine to plug this leak is a Batch-7 explore route.
- **[engineering, Batch 7] `g2p.wall_contact` does NOT contain agents (reconfirmed at high mass).** wall_contact 1.0
  at mass 5e-4/n44 → escape 0.477 (WORSE than the confine-0 control 0.136). It acts on MATERIAL points, not agents;
  raising it does not help containment. Dead as an agent-containment lever.
- **[established, Batch 9] CLEAN STAGE-1B: colour-confine in [0.03, 0.07] fully contains at escape 0 while deform ≫0.02.**
  At n=44/mass 5e-4/move 0.12/12000f, `mpm_to_agent {confine 0.03, field:colour}` → deform 0.0304, **escape 0.000**,
  r_max 0.850, collapsed 0; confine 0.07 → deform 0.0332, **escape 0.000**, r_max 0.854. The clean window is a PLATEAU
  (0.03–0.07), NOT a U-minimum — the Batch-7 cf0.05 escape 0.0227 was a single-cell blip. The ram onsets at confine 0.10
  (escape 0.386, boundary over-confine). This is the FIRST fully clean Stage-1B and it holds at 12000 frames (robust, not
  finite-time). Operating spec `embryo_1B.yaml` (confine 0.03). *Residual:* nn_min ~0.005–0.007 < r0 (confine mildly
  compresses the closest pair, but collapsed 0, above the 0.003 stacking floor) — cf0.03 keeps nn_min highest.
- **[rejected, Batch 9] "Trimming the deform push lowers the boundary flux → escape→0."** FALSE and INVERTED: at
  confine ~0.03–0.05, dropping mass 5e-4→3e-4 RAISED escape (0.000→0.068) and 4e-4 (with cf0.04) gave escape 0.318.
  Lower per-cell push → weaker/less-organised internal circulation → cells drift to the shell in a way the catch handles
  WORSE, not better. **Keep mass 5e-4 at sparse n for clean 1B; do not trim the push.**
- **[established, Batch 9] `field:mass` is dirtier than `field:colour` even at small confine (0.05) — the field choice,
  not just magnitude, sets cleanliness.** cf0.05 field:mass → escape 0.182 (vs colour's 0.023), r_max 1.265, nn_mean
  0.0091 (batch MIN — cells clustered tightest). It did NOT fully collapse at 0.05 (collapsed 0; the full-collapse
  threshold is ≳0.2, Batch 3), but the per-cell attraction signature (tighter clustering + worse containment) is
  present exactly as the boundary-only vs per-cell-attraction mechanism predicts. Confirms `field:colour` as the clean
  containment field.
- **[established, Batch 9] DIVISION LIFTS DEFORM (density adds aggregate momentum) but BREAKS a fixed containment —
  Stage-1C's core trade.** Turning division on at the clean sparse recipe (cf0.05/m5e-4/div0.10) grew n 44→442 and raised
  deform to 0.0614 (highest seen) — density scatters more total momentum onto the grid — but escape RETURNED to 0.278
  (the n=44-calibrated 0.05 catch cannot hold the n=442 boundary flux). The fix is to scale confine UP and per-cell mass
  DOWN as n grows: the dense route cf0.10/mass 5e-5/div0.10 → n=442, deform 0.0268, **escape 0.016**, r_max 0.954,
  collapsed 0 (near-clean at density). So at confluence the recipe is LOW per-cell push (density does the deforming) +
  STRONGER boundary catch. This is `embryo_1C.yaml`; closing the residual 0.016 and maximizing division-driven deform at
  escape≈0 is Stage-1C.
- **[established, Batch 9] DIVISION DEFORMS THE SHELL — deform grows monotonically with n at fixed low mass (Stage-1C central
  claim).** On the dense base (cf0.10, m5e-5), deform = 0.0109 (n44, div OFF, = floor) → 0.0268 (n442, div0.10) → 0.063–0.079
  (n2700, div0.20). The R4 control (div OFF, n44) sits at floor, so the extra deform is proliferation/density-driven, not the
  base push. Montages show progressive reshaping (n442 lobed → n2700 grossly amoeboid/torn). Density adds aggregate scattered
  momentum onto the grid → shell deforms. Confirmed.
- **[established, Batch 9] CONFINE CANNOT BE SCALED UP TO CONTAIN DENSITY — the containment ceiling narrows with n (falsifies
  "confine-up closes escape").** At n=442, raising `mpm_to_agent.confine` 0.10→0.15 RAISED escape (0.0158→0.0226) AND induced
  ram-collapse (collapsed 0→**0.1448**, nn_min 0.002). This is the boundary RAM (Batch-2/7) reappearing at density: a stronger
  colour-gradient catch shoves the larger boundary population into a dense inner clump. The clean confine window NARROWS as n
  grows — n44 ram onset was cf~0.10; n442 collapse onset sits in (0.10, 0.15). So **cf0.10 is already the containment ceiling at
  n=442.** The colour-confine catch has a FIXED capacity (grad(c) localized at the interface); the boundary flux grows with n and
  per-cell mass, so at density containment and division-driven deform pull against each other and the clean window is narrow
  (n≈442, cf≈0.10, m≈5e-5, deform≈0.027, escape≈0.016). To lift clean deform, need a source that adds NO boundary flux (spin).
- **[established, Batch 9] BIG PER-CELL PUSH AT DENSITY IS UNCONTAINABLE BY CONFINE AT ANY SANE MAGNITUDE — mass, not confine,
  is the density-escape driver.** cf0.20/m5e-4/div0.10 (n442) → escape **0.5226**, collapsed **0.3552**, r_max 1.284 (shell tears
  into a teardrop, cells spill). Raising confine to 0.20 to "rescue" the big push made collapse WORSE (ram + push stack), did NOT
  contain it. Mid mass is also dirty at density: m1e-4 (cf0.15, n442) → deform 0.0522 (≈2×) but escape 0.086 + collapse 0.093;
  m1e-4 (div0.05, cf0.12, n139) → escape 0.0647. **At density keep mass 5e-5; there is no confine that holds mass ≥1e-4.**
- **[engineering, Batch 9] CAP `div_rate` ≤~0.10 — div0.20 runs away to n≈2700 (~60× growth) and leaks regardless of push.**
  div0.20 (self-limited by max_occ 0.9) reached n=2700 in 12000f: at m5e-5 escape 0.169, at m3e-5 escape 0.264. At n=2700 the
  boundary flux (n × per-cell drift) overwhelms any containment even at the tiniest per-cell mass. This is also FAR beyond the
  ~4× population directive. Keep div_rate ≤0.10 (n≤~442) to stay containable and near-budget.
- **[rejected, 2026-07-02, Batch 3] Sub-threshold boundary confine (`confine 0.2, field: mass`) contains cells
  without collapse.** FALSE: confine 0.2 (field:mass) → collapsed=**0.579**, nn_min 0.0021≪r0. Confinement drives
  collapse regardless of the gradient field (mass or colour); critical confine sits <0.2 (Batch-2 threshold was in
  (0,0.5); now tightened to <0.2). Do not use confine for boundary containment.
- **[rejected/inert, 2026-07-02, Batch 3] Stiffer `g2p.wall_contact` (0.04→0.12) as a containment lever at low n.**
  At n=95 the run is BYTE-IDENTICAL to the 0.04 default (deform 0.0067, escape 0, r_max 0.818) — no cell reaches the
  wall (r_max 0.82 < shell ~0.9), so wall_contact has nothing to gate. It can only matter under wall pressure (high
  n), which is exactly the escape regime; untested there. Not a low-n lever.
- **[engineering, 2026-07-02, Batch 2] At n=1600 the disc is OVER-CONFLUENT (nn_min<r0 is packing, not collapse).**
  Runaway division (div_rate 0.6, 3000 frames) fills to n=1600; the confluent spacing (~0.015) is below r0=0.02,
  so nn_min pins ~0.002 on every slot even at collapsed≈0 (the sub-r0 pairs are freshly-divided daughters, H2).
  Treat nn_min<r0 at high n as expected packing, not a hard fail — but cap `agent.div_rate` to keep a true 1A
  tiling (nn_min≥r0) and to avoid the pressure-eject that drives escape.
- **[engineering]** `vmax` (per-set speed clamp) exists as a safety net, but the goal is a parameter
  balance where accelerations stay bounded WITHOUT hitting it — treat a run that leans on `vmax` as
  not-yet-balanced.
- **[engineering, 2026-07-02, Batch 5] RUN LENGTH comes from `EMBRYO_FRAMES` / the per-slot `frames` token, NOT
  from the spec's `n_frames`.** `embryo_loop.py` reads `FRAMES=env("EMBRYO_FRAMES","3000")` and passes
  `frames={FRAMES}` to `showcase.py`, which does `sim.n_frames = frames` — so `n_frames: 6000` in the YAML is
  DEAD. That silently ran Batches 1–4 (and the first Batch-5 submit) at 3000, violating the mandatory 6000-frame
  directive. FIX: append `frames 6000 stride 8` to each slot line — showcase builds its arg dict left-to-right,
  the slot's `frames=6000` lands after the loop's `frames=3000`, so the later value wins and 6000 is guaranteed
  regardless of the env var. Always pin `frames`/`stride` per-slot; don't rely on the spec or the env default.
- **[engineering, 2026-07-02] `escape` was never measured — now added.** `embryo_metrics.py` reported
  no `escape` field even though it is a HARD FAILURE in the instruction. Added `escape` = frac of live
  cells with radius > 0.9·Rd (out of the water core, into/through the membrane) + `r_cell_max`. This is
  essential to read the confine ablation: `confine` in `mpm_to_agent` both (a) squeezes cells inward up
  the colour gradient (suspected collapse driver, H5) AND (b) keeps cells inside the core. `confine 0`
  removes BOTH, so a drop in `collapsed` there is only a real Stage-1A win if `escape≈0`. Without this
  metric the confine ablation is uninterpretable. Backward-compatible; does not touch in-flight jobs.

- **[established, Batch 10] `mpm_spin` is NOT a clean deform amplifier at density — INVERTED from its sparse-n role.** At n=442
  (cf0.10, m5e-5, div0.10) raising omega 0.3→0.6 left deform FLAT (0.0268→0.0246) and RAISED escape 5× (0.0158→0.0747, r_max
  crossing 1.0); ω0.9 → deform only 0.0303 at escape 0.0317 (2×). This overturns the sparse-n "spin = best deform-per-escape"
  claim AT CONFLUENCE. Mechanism: solid-body rotation adds a centrifugal outward velocity (∝ radius) to every cell; at density
  the many boundary cells are flung outward → boundary flux ↑ → escape ↑, with no extra shell reshaping. **Spin at density buys
  MIGRATION (coherent net swirl), not deform** — see the 1D lead. *Density-dependent sign:* at n=311 spin ω0.6 IMPROVED
  containment (escape 0.0096→0.000, r_max 0.940→0.867); at n=442 it worsened it. The centrifugal push only dominates once the
  boundary population is large. So do NOT use spin as a deform lever at confluence.
- **[established, Batch 10] The clean-frontier tension is FUNDAMENTAL: at fixed low mass, division-driven deform AND escape both
  scale with boundary flux (∝ n) — you cannot lower one without the other.** div0.08 (n311) vs div0.10 (n442) at cf0.10/m5e-5:
  escape 0.0096→0.0158 AND deform 0.0172→0.0268 move together. The colour-confine catch has fixed capacity; the flux grows with
  n; so the clean deform ceiling at density is ~0.027 (@ escape ~0.016 grazing) and is a real physical limit, not a tuning miss.
  Every attempt to break it (spin, mid-mass 1e-4, confine-up to 0.12/0.15) failed — spin/mass raise escape, confine-up rams.
- **[engineering, Batch 10] override plumbing confirmed.** `showcase.py` builds `ov = dict(kv.split('='))` from `key=value`
  tokens and calls `tune._apply(sim,k,v)`. `_apply` special-cases `agent.move_speed`/`agent.div_rate`/`agent.n`/`agent.p`/
  `cell.youngs`/`n_grid`/`per_parent`/`spawn_radius` (broadcast to all types), and for any other `opname.param` sets that param
  on EVERY operator whose `op==opname`. **Consequence: to make a NEW operator (e.g. `polar_align`) overridable per-slot it MUST
  already exist in the base-spec `operators` + `schedule`** — `_apply` only mutates existing ops, it does not add them. So the
  1D base `embryo_1D.yaml` carries `polar_align` (base gamma 0, inert) and slots override `polar_align.gamma`. `polar_align`
  params are `gamma` (alignment rate) + `noise` (angular diffusion); it reads the `radius_graph` edges (radius 0.05) and mutates
  `heading`, which `glide` turns into displacement. `migration` metric = ‖mean(v̂)‖ = global velocity polar order (net directed
  drift; solid-body rotation about centre ≈0, so a high value = coherent translational stream or off-centre swirl).

- **[established, Batch 11] EMERGENT flocking (`polar_align`, Vicsek) BEATS IMPOSED rotation (`mpm_spin`) for CLEAN collective
  migration at confluence.** At n=557, `polar_align.gamma 120` → migration **0.4929** at escape **0.0197** (r_max 1.06); imposed
  `mpm_spin ω0.6` → comparable migration 0.4285 but escape **0.2603** (r_max 1.24) — 13× dirtier for the same polar order.
  Mechanism: Vicsek alignment redirects each cell's EXISTING move_speed into a SHARED translational heading (the flock circulates
  as one body inside the shell); spin adds a CENTRIFUGAL outward drift (∝radius) that flings the large boundary population past the
  shell. So at confluence, choose emergent flocking, not imposed spin, to drive migration. Do NOT stack spin on a flock (b10 s7:
  γ40+ω0.6 → migr 0.54 but escape 0.38 — the two cooperate for polar order but the spin escape penalty dominates).
- **[established, Batch 11] FLOCKING COHERENCE IS THE CONTAINMENT — stronger `polar_align.gamma` RAISES migration AND LOWERS
  escape TOGETHER (inverts the campaign's deform/migration-vs-escape tension).** γ sweep at n=557: γ40 → migr 0.250 / escape
  0.108; γ120 → migr 0.493 / escape 0.020. 3× the gain → 2× migration AND 5× LESS escape. Mechanism: escape at confluence is a
  DISORGANIZATION problem — semi-independent cells (low γ) random-walk to the boundary and pile up; a strongly-locked flock (high
  γ) moves as a coherent stream that circulates rather than depositing cells on the wall. The R4-control PROVES this: with ALL
  organizing motion off (γ0, flow_align0, ω0) at n=557 the shell TEARS OPEN (escape **1.000**, r_max 2.09, collapsed 0.27, deform
  0.13 pathological). So with spin more circulation → more escape (they fight); with emergent flocking more circulation → LESS
  escape (they align). *Untested tail (Batch 11):* whether the γ↑→cleaner trend is monotone beyond γ120 or reverses when the flock
  crystallizes/jams (flow→0) at very high γ. NOTE the b10 confluence was n=557 (div0.10 in the 1D base grew HIGHER than b09's 442).
- **[established, Batch 11] `flow_align` (SPV, heading→local MPM flow) is a WEAK migration driver — Vicsek neighbour-heading
  (`polar_align`) ≫ flow-coupling.** flow_align gain 120 (γ0) at n=557 → migration only 0.1396 (barely above the diffusive floor
  ~0.10) and the lowest deform of the batch (0.0246). Relaxing heading toward the local fluid flow does not organize a coherent
  stream nearly as well as aligning directly to neighbour headings. `polar_align` is the confluence-migration operator; keep
  flow_align at its low base (40) as a minor SPV coupling, not a migration lever.
- **[established, Batch 11] move_speed 0.24 breaks containment even with flocking.** polar40 + move 0.24 (n557) → escape 0.221
  (vs γ40/move0.12 escape 0.108) — faster cells overshoot the boundary. Reconfirms move_speed as the dominant escape co-driver
  (Batch 7). At confluence keep move ≤0.12.

- **[OVERTURNED, Batch 12] The b10 "monotone γ↑ → migration↑ AND escape↓ (escape→0)" trend is a MIS-READ of a NOISY,
  near-BISTABLE order parameter.** Re-sweeping γ at n557/noise0.1/move0.12/12000f: γ80→migr 0.470, γ200→0.367, γ300→0.519 —
  NOT monotone; migration scatters 0.37–0.52 with no order in γ. And escape at EVERY strong-migration (migr>0.4) n557 point
  PLATEAUS at **0.066–0.077**, never the b10 γ120 0.020 (that was a favourable single realization). Mechanism: migration
  (velocity polar order) has two attractors — a single global translating stream (high migr) vs several locked/rotational
  domains whose drifts cancel (migr→floor ~0.05–0.1). Which one a run lands in is realization/parameter-sensitive (e.g. γ120_n311
  → migr 0.65 but γ200_n311 → 0.09 at the SAME density). So treat migration as a stochastic order parameter, not a smooth γ
  response; and the residual ~0.07 escape at a strong flock is NOT closable by more γ.
- **[established, Batch 12] ANGULAR NOISE ~0.1 IS REQUIRED for global flock coherence — LOWERING it fragments the flock and
  KILLS net migration (opposite of "tighter = cleaner").** noise 0.1→0.05 at γ120/n557 → migration CRASHED 0.49→**0.0379**
  (montage shows multiple competing stream directions whose net polar order cancels). A small angular diffusion lets the flock
  anneal into ONE shared heading; too little noise freezes it into competing domains. Keep polar_align.noise ≈0.1 (do not lower).
- **[established, Batch 12] MIGRATION IS NOT SPEED-INDEPENDENT — move0.06 KILLS the flock (falsifies b11's "direction order,
  speed-independent").** move 0.12→0.06 at γ120/n557 → migration 0.49→**0.0654** (dead) even though it gave the batch-cleanest
  escape (0.0036, r_max 0.902). A coherent translating stream needs enough per-cell speed to build up; at 0.06 it never
  organizes. So the clean containment at move0.06 is empty — there is no migration to drive cells to the wall. Migration needs
  BOTH noise~0.1 AND move~0.12. (Intermediate move0.09 untested — Batch 12.)
- **[established, Batch 12] FLOCKING-IS-CONTAINMENT IS DENSITY-DEPENDENT — it INVERTS at moderate density.** At n557 (jammed
  interior) strong γ contains (escape ~0.07). At n311, raising γ 40→120 raised migration 0.244→0.652 AND raised escape
  0.036→**0.157** — with fewer cells packing the core, the coherent flock translates as a BODY into the shell and punches
  through (the original "collective march into the wall"). So coherence contains ONLY when the interior is jammed enough to
  force circulation instead of translation; at moderate density a strong flock is a LEAK, not a shield. Corrects the Batch-11
  "coherence IS containment" claim: it is a HIGH-density effect.

- **[established, Batch 13] γ120/n557 is a ROBUST clean strong-flock point — the "lucky-draw" caveat is OVERTURNED.** Escape
  0.020 at migr 0.49 has now reproduced across TWO independent realizations (b10 AND b12). The b12-going-in claim that
  strong-migration escape "plateaus at 0.066–0.077 and 0.020 was a single lucky draw" is FALSIFIED — γ120 genuinely gives
  escape ~0.020. Adopt `polar120` (γ120, n557, move0.12, cf0.10) as the strong-migration 1D operating point (migr 0.49 @
  escape 0.020). Migration is non-monotone in γ with a PEAK at γ≈120 (γ100 0.433, γ120 0.493, γ140 0.248 → γ140 falls back to
  the diffusive floor); there is an OPTIMAL alignment strength, not "more is more."
- **[established, Batch 13] RESIDUAL ESCAPE AT THE STRONG FLOCK IS CLOSABLE by move-down + confine-up TOGETHER (single-lever
  confine-up FAILS).** At full move0.12, cf0.10→0.12 at γ120 made escape 10× WORSE (0.020→0.201, no ram — collapsed 0.022,
  nn_min fine — the stronger catch just flings boundary cells out faster). But at move0.09 the SAME cf0.12 is tolerated:
  escape **0.004** (migr 0.348, r_max 0.906, all cells inside). So the strictly-contained confluence recipe is `polar120_m09_cf12`
  (γ120, move0.09, cf0.12) → migr 0.35 @ escape 0.004. Mechanism: a slower flock approaches the boundary gently enough that the
  stronger confine absorbs it rather than shoving it through; at full speed the confine's inward push + the flock's outward
  translation stack into a faster ejection. move0.09 costs ~30% migration (0.49→0.35) — the trade for a strictly-clean shell.
- **[established, Batch 13] noise~0.1 is the flock-coherence PEAK (both sides fall off).** polar_align.noise 0.05 KILLED migration
  (b12: 0.038), 0.15 also dropped it (0.202); 0.10 is the optimum for a single global translating stream. Do not move noise off 0.1.
- **[engineering, Batch 13] PER-TYPE CROSS-REPULSION plumbing (Stage-1E mechanism).** `deposit` (level cell, `at: agent`,
  `to: chemical`) writes each agent's amount into channel = its `node_type` (type a→ch0, type b→ch1); the `chemical` field is
  `{frame: grid, res: N, couples_to: agent}` → C = #agent-types = 2 channels. `diffuse`/`decay` (`at: chemical`) shape it.
  `chemotaxis` (`at: agent[type=X]`, `from: chemical`, `channel`, `gain`) emits velocity `gain·grad(channel)` (first_derivative,
  ADDS to glide/repel/drag). Two instances — `agent[type=a]` reads channel 1 (b's trail), `agent[type=b]` reads channel 0 —
  both NEGATIVE gain → each type flees the other → demixing. KEY: a SINGLE schedule token `chemotaxis` runs BOTH instances
  (engine `_run_token` runs every op-instance of a name), and a SINGLE `chemotaxis.gain` override sets gain on both (tune
  `_apply` sets `opname.param` on every op named `opname`) — so one override sweeps the symmetric cross-repulsion strength.
  Type selectors (`agent[type=a]`) resolve to `active & node_type==index` each tick (engine `_selector_mask`). deposit dt-scales
  as `amount·dt`; diffuse/decay as `rate·dt` (dt=0.002 here, so use LARGE amount/rate ~O(1–8) to build a field in ≤12000 frames).

- **[established, Batch 15] THE b13/b14 CHEMICAL FIELD WAS INERT — deposit/decay saturates + diffuse flattens ⇒ grad≈0 ⇒
  chemotaxis force≈0.** Field arithmetic (dt=0.002): `deposit` adds `amount·dt` (4.0·0.002=0.008/frame) and `decay` removes
  `rate·dt·c` (2.0·0.002=0.004·c) ⇒ equilibrium pixel c = deposit/decay = **2.0, CLAMPED to 1.0 by `deposit`'s `clamp_(max=1)`**
  → every occupied pixel SATURATES. `diffuse` 6.0·dt=0.012/frame then smears the saturated pixels into a flat plateau across the
  disc → grad(channel)≈0 everywhere but the rim → chemotaxis velocity `gain·grad`≈0. Independently, from a MIXED start type-a's
  channel (ch0) and type-b's channel (ch1) are spatially CO-EXTENSIVE (both ~uniform over the disc) → cross-repulsion (`flee the
  other channel`) has NO gradient to act on until some OTHER process breaks symmetry (chicken-and-egg). b14 result: seg
  non-monotone/random in |gain|, montage stays mixed at every gain. **FIX (Batch 15): keep the field UNSATURATED + LOCALLY
  CONTRASTED** — deposit 0.8 (equilibrium c=deposit/decay=0.8<1, responsive), diffuse 2.0 (trail a few px wide, real gradient),
  decay 1.0 (field tracks where each type IS now). And prefer **SELF-AGGREGATION** (each type climbs its OWN trail, gain>0 — a
  Keller-Segel instability that self-amplifies any local excess) over cross-repulsion, which needs a pre-existing gradient.
- **[established, Batch 15] `segregation` HAS A SAMPLING NOISE FLOOR ≈0.12 AT n=44 — the metric cannot resolve sorting at sparse n.**
  seg = |⟨x⟩_a−⟨x⟩_b|/R is the gap between two 22-cell x-means over a disc of R≈0.34: std_x≈R/2, SE≈(R/2)/√22≈0.036, the |Δ| of two
  independent means is half-normal with E≈0.041 → **E[seg]≈0.12, ~2σ≈0.24.** The entire b14 sweep (0.045–0.23) lies WITHIN ~2σ of
  this floor — so NONE of those numbers is evidence of partitioning (the 0.23 at gain −0.02 is a ~2σ noise draw, not a dose
  response). Floor scales as 1/√(n/2): ≈0.07 at n=120, ≈0.05 at n=250. **Consequence: to detect a real demix, either run n≥~120
  (Batch-15 selfagg_hin/ctrl_hin) OR require seg≫0.25 at n44 — and ALWAYS read the cells panel (seg only sees the x-projection; a
  top/bottom or radial demix reads at the floor).** Also: differential MOTILITY alone (a 0.20 / b 0.05) gave seg 0.104 (= n44 floor)
  and montage mixed → a scalar speed difference does NOT sort AND raised escape 0.046 (fast type overshoots); sorting needs a
  type-PAIR interaction, not a per-type scalar (R1-minimal result).
- **[engineering, Batch 15] `attraction_repulsion` is per-RECEIVER-TYPE, NOT neighbour-type-aware — it CANNOT do differential
  adhesion (attract-same / repel-cross) as written.** Its force law reads `p = type_params[node_type[i]]` (the receiver's params)
  and sums `f(r_ij)·(pos_j−pos_i)` over ALL neighbours regardless of j's type (`attraction_repulsion.py:61-68`). So it gives each
  type its own radial law applied to every neighbour equally; it has no j-type term. If a future 1E batch wants Steinberg
  differential adhesion (the classic robust sorter), it needs a type-PAIR-aware operator or the chemical route — not this op.

- **[established, Batch 16] CHEMOTACTIC SELF-AGGREGATION (gain>0, each type climbs its OWN trail) is a KELLER-SEGEL COLLAPSE
  DRIVER, NOT a demixer — it beats hard-core repel exactly like confine.** b15: self-agg gain 0.3 (n44) → collapsed **0.75**,
  nn_min 0.0002 ≪ r0; gain 1.0 (n44) → collapsed 0.75, nn_min 0; gain 1.0 (n120) → collapsed **0.9917**, nn_min 0, flow 6e-5
  (frozen). Both types climb their own trail from a MIXED co-located start → they collapse to ONE shared central knot together,
  not two separated blobs (montage: single packed clump; seg stays 0.04–0.10). The chemotactic self-attraction OVERWHELMS repel
  (r0 0.02) — same signature as confine-collapse (chemotaxis velocity `gain·grad` adds an inward drift repel cannot beat).
  **Consequence: gain>0 chemotaxis alone can never partition — it aggregates to a point. Collapse worsens at density (n120
  catastrophic). The self-agg collapse floor at n44 sits below gain 0.3.** For any demix use, keep self-gain WEAK (<0.3).
- **[established, Batch 16] CHEMOTACTIC CROSS-REPULSION (gain<0) is INERT FROM A MIXED START even with an ACTIVE sharp field —
  the blocker is SYMMETRY BREAKING, not field inertness (b14's diagnosis is now separable from b15's).** With the sharp
  unsaturated field verified live (self-agg clumped strongly on it), cross-rep still did not sort: seg 0.049 (gain 0) → 0.059
  (−0.3) → 0.121 (−1.0) → 0.026 (−3.0), no order in |gain|, montage salt-and-pepper mixed at every gain (2nd batch confirming).
  From a mixed start the two type-channels are spatially CO-EXTENSIVE, so grad(other-channel) ≈ 0 → "flee the other" has no
  gradient to act on (chicken-and-egg). Cross-rep alone is CLEAN (never collapses — pushes types apart, escape 0 even at −3.0)
  but powerless without a pre-existing gradient. **So pure cross-rep needs a symmetry-breaking partner or a seeded gradient.**
- **[rejected, Batch 17 reading b16] CHEMICAL DIFFERENTIAL ADHESION (self-agg + cross-rep TOGETHER) does NOT demix from a MIXED
  start — it is SQUEEZED between "too weak → mixed" and "strong enough self → Keller-Segel collapse".** The b16 hypothesis was
  a two-species cross-repulsive instability that self-amplifies a local excess. Result: (a) the clean combos (self+0.05/cross−0.5,
  +0.15/−2.0, +0.1/−1.0 with sharper diffuse1.0) all pinned seg at the n44 noise floor 0.11–0.16, montage salt-and-pepper mixed —
  the weak self term never nucleates a symmetry break; (b) the primary self+0.1/cross−1.0 HARD-FAILED (collapsed 0.114, escape
  0.114, nn_min 0.0009 ≪ r0) — once the cross term co-locates a local type excess, the self +0.1 term crosses the stacking floor
  and collapses it (the "weak self stays sub-collapse" assumption is FALSE when cross-rep first pulls cells together); (c) at n120
  the self term drove CATASTROPHIC collapse (0.492, seg 0.020) — density amplifies the K-S self-attraction, collapsing the two
  types TOGETHER (co-located start) not apart. **The self-gain window for "breaks symmetry but doesn't collapse" from a mixed
  start is EMPTY in this system.** This is the THIRD falsification of chemotaxis-from-a-mixed-start (b14/15/16). The blocker is
  SYMMETRY BREAKING, not force strength/field sharpness/wiring — confirmed by exhausting the pure-spec action set over 3 batches.
- **[established, Batch 17 reading b16] SEEDED-SPLIT MAINTENANCE is the correct next experiment (engine `type_layout: split_x`
  IMPLEMENTED).** Because self-agg-alone collapses (no between-type term to hold types apart) and cross-rep/combined are inert-or-
  collapse from a mixed start (no pre-existing gradient / self term trips collapse), the force-test must be DECOUPLED from the
  symmetry-break problem: seed a partition and ask only whether a force MAINTAINS/SHARPENS it. `_assign_types` (engine.py:191) now
  reads `type_layout` (default `random`; `split_x` → `torch.argsort(state[:,0])` puts type a = LEFT half, b = RIGHT half).
  Positions are set (build line 253) before `_assign_types` (line 268); schema passes set keys through (schema.py:231), so it is
  opt-in and backward-compatible (no existing spec affected). Batch-17 seeded specs also set `mpm_spin omega 0` so bulk rotation
  cannot smear the x-projection seg — the only thing that can change the seed is interdiffusion (re-mix) vs sorting (maintain).
  KEY prediction to settle: from a seeded split the two type-channels are spatially SEPARATED, so cross-rep finally HAS a gradient
  — if seed_xrep holds seg high while mix_xrep (same force, mixed start) and seed_ctrl (no force, re-mixes) fall to the floor,
  the missing ingredient in b14–16 was SOLELY the symmetry break. If even the seeded split re-mixes under the force → the chemical
  route cannot even maintain a partition → abandon it, adopt best-clean point, log 1E [open], advance.

- **[engineering, Batch 18 reading b17] BRIGHT LINE — `type_layout: split_x` (as first shipped) was BROKEN: it split the type
  fractions over the BUFFER, not the live cells, so ALL live cells fell into ONE type. This wasted b17 (SECOND whole-batch
  engine bug in 1E, after the b13 YAML parse bug).** `Level.n` returns `state.shape[0]` = the BUFFER size (3000 here), NOT the
  live count (`models/base.py:162`). The original `_assign_types` did `perm = argsort(lvl.state[:lvl.n, 0])` then split by
  `round(fraction·lvl.n)` — an argsort over all 3000 rows. Dead slots (never spawned) have x=0; the live sunflower disc sits at
  x∈[~0.2,0.8] (all positive). Ascending sort ⇒ 2956 zeros first, 44 live cells last ⇒ type a = first 1500 = all dead (0 live),
  type b = all 44 live. Signature in metrics: **seg EXACTLY 0.0** (empty group), **byte-identical slots across a gain sweep**
  (force inert, one type), **one render colour**. The RANDOM layout is immune (randperm scatters live indices across the buffer,
  ~half land in each type). FIX (this batch): restrict the split_x sort+split to LIVE slots and the LIVE count —
  `perm = nonzero(occ>0).flatten(); perm = perm[argsort(state[perm,0])]; total = perm.numel()` (dead slots stay type 0,
  harmless). Now 44 live cells sort by x and split 22/22 → a=LEFT, b=RIGHT, a real seed. Backward-compatible (random path
  untouched). **General lesson: any per-type assignment / count / metric that indexes `lvl.n` must mask to live `occ>0` — the
  buffer is ~68× the live population here, so a buffer-wide operation is dominated by dead slots.** Diagnose a NULL 1E batch by
  reading seg (exactly 0.0 = empty group) and colour count in the montage BEFORE theorising physics.

- **[established, Batch 19 reading b18] THE SEED IS REAL + CROSS-REP MAINTAINS (BUT CANNOT CREATE) A SEEDED PARTITION —
  first clean 1E signal after 3 batches of mixed-start falsification.** The `_assign_types` live-mask fix works: every
  seeded slot's t=0 montage shows a clean red-LEFT / yellow-RIGHT split (both colours present; a uniform half/half disc
  seed has t=0 seg ≈ 0.85 = each half-centroid at ±4R/3π). Reading FINAL-frame seg = seed survival over 12000f, at
  n44/move0.12/cf0.05/mass5e-5: no-force control (gain 0) decays 0.85→**0.2935** (slow diffusion at n44, retains
  memory); MILD cross-rep −0.5 → **0.6147** (escape 0, collapsed 0) — the force MORE THAN DOUBLES retained seg vs the
  control, actively HOLDING the partition; the MIXED-start same force (−1.0) stays at the floor **0.0939** (no seed →
  nothing to maintain → inert, reconfirming b14–16). **So the missing ingredient in b14–16 was SOLELY the symmetry
  break — a chemical cross-repulsion MAINTAINS a partition it cannot spontaneously CREATE.** This is the adopted-candidate
  Stage-1E mechanism.
- **[established, Batch 19 reading b18] SEED-MAINTENANCE IS NON-MONOTONE IN CROSS-REP GAIN — a MILD peak (~−0.5)
  maintains, STRONG (−1.0/−2.0) OVER-DRIVES and homogenizes to the floor.** n44 gain series: 0 → seg 0.29 (control),
  −0.5 → **0.61** (peak, holds), −1.0 → **0.16**, −2.0 → **0.13** (both BELOW control — worse than no force at all).
  The strong-gain montages show red+yellow re-intermingled centrally by t=12000 (not wall-segregated). Mechanism
  (hypothesized, Batch 19 tests reproduction): a mild flee is a laminar drift to each type's own side (sorting); a
  strong flee overshoots each type to the far wall, where it spreads along the rim, wraps around top/bottom and
  interpenetrates the other type on the opposite side → over-mixing. Consequence: for seeded-partition maintenance use
  WEAK cross-rep (~−0.5); do NOT crank the gain. (Single realization each — Batch 19 reproduces the peak 0.61 and the
  over-drive 0.16 and maps −0.25/−0.75 between.)
- **[established, Batch 19 reading b18] SELF-AGGREGATION STILL HARD-COLLAPSES EVEN FROM A SEPARATED SEED; DIFFERENTIAL
  ADHESION (self+cross) buffers the cross over-drive.** seed_selfagg (+0.2 self, seeded) → collapsed **0.7955**, nn_min
  0.0003 — Keller-Segel collapse persists from a seed (each type climbs its own side-localized hill to a knot; the high
  seg 0.53 is two collapsed points, not a clean partition). DEAD as a partition mechanism (reconfirms b15/b16). BUT
  differential adhesion seed_combo (self+0.1 / cross−1.0, seeded) is CLEAN (collapsed 0, escape 0) and holds seg
  **0.4664** — adding weak self-cohesion RESCUES the over-driven −1.0 cross case (0.16→0.47): the self term keeps each
  type clustered so the strong cross-rep disperses/mixes them less. So self-cohesion is a stabilizer, not a partitioner,
  in this regime. Batch 19 tests self+0.1/cross−0.5 (self-cohesion on the MILD peak).

- **[engineering, Batch 20 reading b19] BRIGHT LINE — RUNS ARE FULLY DETERMINISTIC; re-running a spec is NOT a reproduction.**
  `general.seed` is fixed per spec (0 in every 1E spec) and the engine builds ALL randomness from it
  (`H.rng = torch.Generator(device).manual_seed(sim.seed)`, engine.py:235 — sunflower jitter, type assignment, noise). So
  b18 xrep_05 = **0.6147** and b19 xrep_05 = **0.6147** to 4 digits are the SAME trajectory recomputed. The two b19 slots that
  "reproduced the peak/over-drive" were wasted — they confirmed determinism, not robustness. **To test whether an effect is
  real (not a lucky seed), author spec COPIES with different `general.seed` and compare a PAIRED same-seed control.** `seed` is
  NOT overridable via a slot token — tune `_apply` handles only the special keys (`agent.move_speed`, `spawn_radius`, `agent.n`,
  `n_grid`, `per_parent`, `cell.youngs`, `agent.div_rate`, `agent.p`) and any `opname.param` on an existing operator; `seed` is
  none of these, so it must live in the spec file. General lesson: any "reproduce" slot must change the seed or a nuisance param,
  else it is a no-op recompute.
- **[established→REJECTED by Batch 21, see the b20 3-seed block below] THE −0.5 SEED-MAINTENANCE PEAK "BELL CURVE" WAS SINGLE-SEED
  AND DID NOT REPLICATE — the ~2× effect was a seed-0 coincidence (b20 seeds 1/2/3 gave Δseg −0.31/−0.05/−0.08).** n44 seeded
  split, seg vs cross-rep gain (SEED 0 ONLY): 0→0.29, −0.25→0.33, **−0.5→0.61 (seed-0 peak)**,
  −0.75→0.42, −1.0→0.16 (over-drive, below control); all escape 0, collapsed 0. The dose-response is smooth and unimodal with a
  well-defined optimum at −0.5, consistent with the b18 mechanism (mild flee = laminar sort to own side; strong flee = overshoot
  + rim-wrap + re-interpenetration). Because the sim is deterministic, this is ONE realization — the ~2× effect is real for seed 0
  but not yet shown to be seed-independent (Batch 20 tests seeds 1/2/3 paired).
- **[established, Batch 20 reading b19] SELF-COHESION HURTS AT THE MILD PEAK — it is a stabilizer ONLY when cross-rep OVER-drives,
  a liability otherwise (revises the b18 "self-cohesion stabilizes" reading).** combo (self+0.1/cross−0.5, seeded, n44) → seg
  **0.4826** (< the pure −0.5 peak 0.61) AND collapsed **0.0455**, nn_min 0.0025 (Keller-Segel onset). Adding weak self-cohesion
  ON TOP of the already-optimal mild cross-rep both (a) LOWERS seg (the self term pulls each type toward its own centroid, undoing
  some of the clean laminar sort) and (b) trips a mild K-S collapse (the self-attraction crosses the stacking floor once cross-rep
  co-locates a local excess). Contrast b18, where self+0.1 RESCUED the OVER-driven −1.0 case (0.16→0.47) by keeping each type
  clustered against the over-dispersal. **So: self-cohesion helps when the cross force is too strong (buffers over-mixing) but
  hurts when the cross force is already tuned (adds collapse, dilutes the sort). For the −0.5 operating point, use PURE cross-rep.**
- **[established, Batch 20 reading b19] THE n44 MAINTAINED PARTITION DOES NOT TRANSFER TO DENSITY (n120) AT THE SAME cf — force
  advantage collapses AND containment fails.** At n120/cf0.05: xrep_05 seg 0.477 vs ctrl 0.431 → the force adds only **+0.046**
  (vs **+0.32** at n44), and BOTH HARD-FAIL escape (0.075 xrep / 0.0167 ctrl, r_max 0.94–0.97). Two things happen at density:
  (i) the no-force control retains MORE seed memory passively (0.43 vs n44's 0.29 — slower interdiffusion when crowded), shrinking
  the headroom for the force to add; (ii) the cf0.05 colour-confine catch calibrated at n44 cannot hold the larger n120 boundary
  flux, so cells leak. **Whether the force>control gap returns once the boundary is restored (cf0.05→0.08) is the Batch-20 density
  question.** Provisional: the clean, strong (2×) maintained partition is so far a SPARSE-n (n44) effect.

- **[rejected, Batch 21 reading b20] THE n44 −0.5 SEED-MAINTENANCE "PEAK" IS A LUCKY SEED-0 ARTIFACT — cross-rep does NOT
  robustly maintain a partition at sparse n.** The b20 paired 3-seed test (each seed run gain0-vs-−0.5, Δseg per realization):
  n44 Δseg = **−0.312** (seed1), **−0.049** (seed2), **−0.076** (seed3) — the sign FLIPPED on all three new seeds vs seed-0's
  **+0.32**; mean of the new seeds is −0.146 (force slightly HURTS). Root cause: **at n44 interdiffusion is too slow to erase the
  seed** — the no-force controls retained seg **0.42–0.64** over 12000f (seeds 1/2/3), i.e. there is essentially no headroom for
  a force to add, and the laminar cross-rep drift just perturbs/erodes the passive decay. Seed-0's control happened to be an
  anomalously LOW draw (0.29), which manufactured the illusory doubling. **The b18/b19 "clean unimodal bell curve, peak −0.5, ~2×
  control" (now demoted) was a single-realization coincidence.** General lesson reinforced: a deterministic sim gives ONE draw per
  spec; never promote a single-seed effect — the b20 determinism warning was exactly right, and seed-0 was the outlier.
- **[open→promising, Batch 21 reading b20] THE GENUINE FORCE TEST LIVES AT DENSITY, NOT SPARSE n — because only there does the
  no-force control RE-MIX.** At n120 + cf0.08 (seed 0): the gain-0 control interdiffuses to seg **0.036** (< the n120 floor ~0.07 =
  fully mixed), while the −0.5 cross-rep HOLDS seg **0.20** (~3× floor), both **escape 0** (cf0.08 killed the b19 cf0.05 escape
  0.075 — the containment is the fix). Δseg **+0.165**. Unlike n44 (control retains the seed passively → no headroom), at n120 the
  control genuinely erases the seed, so the force's mixing-resistance is a REAL, measurable advantage. deform also rises at density
  (0.012–0.013 vs 0.004–0.008 at n44). **BUT this is a single seed-0 draw — untested, the exact trap that just fell at n44.** Batch
  21 replicates it across seeds 1/2/3 (paired) before any adoption. The mechanistic reading (if it holds): chemical cross-rep is a
  MAINTENANCE force that can only be SEEN to work where an active re-mixing process would otherwise destroy the seed — a
  density/stirring-dependent effect, not a property of the force alone. It still CANNOT spontaneously create a partition (b14–16).
- **[engineering, Batch 21] cf0.08 is the n120 containment fix — confine 0.05→0.08 killed escape (0.075→0.0) at n120 WITHOUT
  ram-collapse (collapsed 0, r_max 0.83–0.89, nn_min ~0.0043).** At sparse n cf-up rams (n44 ram onset ~0.10), but at n120 the seed
  boundary flux needs the stronger catch and 0.08 sits in the clean window (below the n120 ram). Contrast 1C where cf-up at n442
  rammed (0.10→0.15 → collapsed 0.14) — the clean confine ceiling is density-dependent; at n120 it is ≥0.08. Use cf0.08 for the
  n120 seeded-split family.

- **[rejected, Batch 22 reading b21] CHEMICAL CROSS-REPULSION DOES NOT ROBUSTLY MAINTAIN A SEEDED PARTITION — the whole 1E chemical
  route is dead.** Paired 3-seed test at n120/cf0.08 (each seed run gain0 vs -0.5): Delta-seg = -0.062 (s1) / +0.182 (s2) / -0.200 (s3),
  mean -0.027 — sign-flips, no maintenance. Combined with the b20 n44 3-seed test (Delta -0.31/-0.05/-0.08) that is SIX realizations
  averaging ~0. The b18/b19 n44 "peak" (seed0 +0.32) and the b20 n120 "genuine force" (seed0 +0.165) were BOTH lucky seed-0 draws.
  The density dose-response (-0.35/-0.5/-0.75 @ n120/seed1) is flat within noise (seg 0.278/0.277/0.373) — the "-0.75 over-drive"
  does not reproduce. Cross-rep is CLEAN (escape 0 always) but powerless. Route CLOSED after b14-b21 (mixed-start inert x3, seeded-
  maintenance sign-flips x2). [rejected]
- **[established, Batch 22 reading b21] THE ROOT CAUSE OF ALL 1E AMBIGUITY IS SLOW INTERDIFFUSION -> NO HEADROOM — the seed persists
  passively, so no force can be SEEN to maintain it.** At n120/move0.12 the no-force controls retain seg 0.25-0.55 over 12000f (>>
  mixed floor ~0.07) across seeds 1/2/3; only seed-0 happened to re-mix (0.036), which manufactured every prior "force works"
  reading. A maintenance force can only be demonstrated where an active process would otherwise erase the seed; at these
  densities/speeds nothing does. **Consequence: to test maintenance you must first make the control re-mix (raise stir /
  move_speed / add active mixing) — else the force test is inconclusive-by-no-headroom, not a win or a loss.** This reframes the
  whole 1E campaign: the missing ingredient was never the force, it was a re-mixing baseline.
- **[established, Batch 22 reading b21] A SEEDED L/R TWO-TYPE SPLIT IS A STABLE (FROZEN) PARTITION with NO force — the passive 1E
  operating point.** With chemotaxis off, a seeded split holds seg 0.25-0.55 over 12000f at n120 (escape 0, collapsed 0, deform
  0.013-0.018), i.e. the two types stay regionally partitioned purely because interdiffusion is slow. This is the best-clean 1E
  point to adopt if the stir test fails: a partitioned blastula, kinetically maintained, not force-maintained. (Not a demix
  MECHANISM — it cannot spontaneously CREATE a partition, b14-16 — but it DELIVERS the partition phenomenology from a seed.)
- **[established, Batch 22 reading b21] cf0.08 CONTAINS THE n120 SEEDED SPLIT ROBUSTLY ACROSS SEEDS — escape EXACTLY 0 on all 8
  b21 slots (3 seeds x gains 0/-0.35/-0.5/-0.75).** Confirms the b21 [engineering] cf0.08 fix is seed-independent, not a seed-0
  fluke. deform 0.013-0.019 at n120 (mild). One -0.5 slot (seed3) showed a minor K-S collapse onset (collapsed 0.0167, nn_min
  0.0028) — the cross-rep can nucleate a small local clump at some seeds, another mark against it as a clean partitioner.

- **[established, Batch 23 reading b22] STIR (move_speed) DOES re-mix the frozen seeded partition — the "no-headroom" excuse is
  RESOLVED, and the chemical force STILL fails against it (1E chemical route dead, FINAL).** At n120/cf0.08 the no-force control
  seg falls move0.12 0.339 -> move0.18 0.063 -> move0.24 0.078 (seed2 m24 0.053), i.e. move>=0.18 interdiffuses the seed to the
  n120 mixed floor (~0.07) within 12000f — the active re-mixing baseline the whole 1E campaign lacked. With that baseline in hand,
  the -0.5 cross-rep does NOT hold above it: paired Delta-seg = +0.004 (m18 s1) / -0.011 (m24 s1) / +0.060 (m24 s2), mean ~+0.018,
  sign-flips; -0.75 over-drives to seg 0.032 < control. Even at the cleanest point (m18, control re-mixed to floor at escape 0.008)
  the force adds +0.004 = nothing. So the chemical cross-rep cannot maintain a partition even against a genuine, measured re-mixing
  process. Route CLOSED after b14-b22 (mixed-start inert x3; seeded-maintenance sign-flips x2 with control frozen; and now fails
  x1 with control genuinely re-mixing). [rejected]
- **[established, Batch 23 reading b22] move0.18 IS THE CLEAN-STIR CEILING AT n120/cf0.08; move0.24 BREAKS CONTAINMENT.** At
  move0.24 three of five slots hard-fail escape (0.100/0.175/0.192), r_max crosses 1.0; at move0.18 the same specs are clean
  (control escape 0.008 r_max 0.90; xrep escape 0.000 r_max 0.88) AND the control already re-mixes to floor. So to STIR the system
  hard while staying contained, use move ~0.18, not 0.24 — reconfirms Batch 7 (move_speed is the dominant escape co-driver) and the
  b13 rule that cf cannot simply be scaled with speed. Consequence for INTEGRATION: keep move ~0.12 (frozen partition) to 0.18
  (clean stir) at n120/cf0.08; 0.24 is out of the clean window.
- **[established, Batch 23] 1E FINAL PICTURE — a two-type partition is DELIVERED as a FROZEN seed, NOT as an actively-maintained
  demix. Partition and stir/migration are ANTAGONISTIC in this system.** The seeded L/R split holds (seg ~0.34) ONLY at low stir
  (move0.12, no flocking); it is not created by any force (b14-16) and not maintained against re-mixing by any force (b18-22). The
  one process that erases it is stir (move>=0.18 / flocking). This is the central constraint carried into INTEGRATION: adding 1D
  flocking migration should re-mix the partition, so a simultaneously-migrating-AND-partitioned blastula is expected to require
  MILD flocking (partition survives) or to trade one phenomenon for the other. Division (daughters inherit type, spawn local) is
  the candidate deform source that does NOT stir — test whether it preserves/sharpens the partition.

- **[engineering, Batch 24 reading b23] BRIGHT LINE — TO ENABLE DIVISION USE `agent.div_rate`, NOT `cell_divide.rate`. The latter
  is INERT whenever the spec sets a per-type `div_rate` (all 1E/split specs do, = 0.0).** `cell_divide.forward` takes its rate
  from `getattr(lvl,"div_rate",None)` (the per-type buffer) and only falls back to the operator `rate` param if that buffer is
  ABSENT (cell_divide.py:50-51). The split specs declare `types: {a:{div_rate:0.0}, b:{div_rate:0.0}}`, so the buffer exists and
  overriding `cell_divide.rate` sets the ignored fallback → division stays OFF. `tune._apply` special-cases `agent.div_rate` and
  broadcasts it to every type's `div_rate` (tune.py:48-50) → that IS the working key. Signature of the bug in metrics: n stays at
  the start count for all frames AND the div-on slot is byte-identical to its div-off sibling (b23 s0≡s3, s2≡s4). This VOIDED the
  entire b23 division arm — the central integration test. General lesson: an override on an operator param only works if the
  operator actually READS that param at runtime; per-type buffers (div_rate, move_speed, p) shadow the op-level scalar, so use the
  `agent.<field>` broadcast keys for anything that has a per-type form. div_rate DOSE from n120 (growth ≈ exp(rate·dt·nframes) =
  exp(rate·24)): 0.03→~2.05× (n246), 0.05→~3.3× (n400), 0.06→~4× (n480, the directive max), 0.08→~6.8× (n820, OVERSHOOTS at n120).
- **[established, Batch 24 reading b23] FLOCKING LIFTS MIGRATION AT n120 (γ60 0.33 > γ120 0.26 > frozen base 0.12) — migration is
  achievable in the partitioned blastula, and γ is non-monotone (γ60 beat γ120 here).** Reconfirms 1D's near-bistable, noisy
  migration order parameter (b12): the coherence a realization lands in is not monotone in γ, so γ60 can out-migrate γ120. Escape
  is marginal though (γ60 0.075 FAIL, γ120 0.050) at n120/cf0.08 — a sparse-n flock grazes/punches the wall (b12 density-dependent
  containment). So migration coexists with the partition only marginally on the guardrail; the clean route needs cf-up or slower move.
- **[established, Batch 24 reading b23] FLOCKING-STIR RE-MIXES THE SEED SEED-DEPENDENTLY — partition survival under stir is a
  kinetic lottery, NOT robust.** γ60 kept seg 0.479 on seed1 but re-mixed to 0.046 (= n120 mixed floor) on seed2. Same γ, opposite
  outcome by seed → consistent with the whole-1E finding that the seed persists or erases by chance of interdiffusion, and stir
  tips it toward erasure. Any "flocking preserves the partition" claim must be shown across seeds; a single seed proves nothing.
- **[established, Batch 24 reading b23] CHEMICAL CROSS-REP −0.5 IS A CLEAN CONTAINMENT LEVER (side-effect), though still USELESS
  for the partition.** s5 (γ120 + xrep−0.5) held seg 0.249 < γ120-alone 0.277 (force adds nothing to partition — chemical route
  dead, 4th confirmation) BUT gave the batch's cleanest containment: escape 0.050→**0.000**, r_max 0.974→0.893. The mutual −0.5
  cross-rep pushes both types off the boundary (a symmetric inward spread) → fewer wall contacts → less escape. Parked as a
  containment option if cf-up alone can't close escape at density; NOT a partition mechanism.

- **[established, Batch 25 reading b24] DIVISION IS A DEFORM SOURCE INSIDE THE PARTITIONED BLASTULA — the 1C law (deform↑ with
  n) holds in the INT base.** With division now firing correctly (`agent.div_rate`), deform rises monotonically with the
  proliferation-driven density: 0.016 (n120, div OFF control = floor) → 0.020 (n235, div0.03) → 0.021 (n374, div0.05) → **0.049
  (n503, div0.06)**, collapsed 0 throughout. Montages go round → lobed → amoeboid. The R4 div-ablation control sits at floor, so
  the extra deform is proliferation/density-driven, not the base push. Division delivers the membrane deform the integration
  wanted (deform ≈ 3× the frozen-partition floor at 4× population).
- **[established, Batch 25 reading b24] DIVISION RE-MIXES THE SEEDED PARTITION — proliferation GENERATES STIR, so division is NOT
  a non-stir deform source (the INT-2 preservation hypothesis is FALSIFIED).** Every division slot at move0.12 collapsed seg from
  the frozen 0.339 toward the mixed floor: div0.03 → 0.115, div0.05 → 0.153, div0.06 → 0.126, and a 2nd SEED (seed2, div0.03) →
  **0.073 = the n230 mixed floor** (re-mix is seed-general, not a single draw). Daughters inherit type + spawn local (offset
  0.004), yet the partition still erased because proliferation drives an outward FLOW that stirs (migration rose 0.12→0.22 as
  division turned on). **Consequence: the 1E seeded partition is antagonistic to EVERY active deform/flow process tested — stir
  via flocking (b23) AND stir via division (b24) both erase it. A simultaneously-partitioned-AND-actively-deforming blastula is
  not achievable in this operator set; partition holds only in a frozen (no-stir) blastula.**
- **[established, Batch 25 reading b24] THE CLEAN 4-PHENOMENON INTEGRATION POINT = DIVISION + MILD FLOCKING (b24 s4, γ60/div0.03/
  cf0.10/move0.12).** deform 0.028, migration 0.35 (batch max), escape **0.013** (batch min/cleanest), n235 dividing, collapsed 0,
  nn_min 0.0039 — {stability + membrane-deform + division + collective-migration} coexist CLEANLY in one blastula. Only 1E
  partition is sacrificed (seg 0.064 = mixed). This is the current INTEGRATION deliverable (4 of 5 rungs, escape-clean). Batch 25
  tests whether pushing density/flock strength lifts its deform toward the 0.049 ceiling while staying clean.
- **[established, Batch 25 reading b24] cf0.08 BEATS cf0.10 FOR DEFORM AT n374 — a weaker boundary catch lets the shell displace
  MORE while still containing at this density.** At n374/div0.05, cf0.08 → deform 0.027, escape 0.029, r_max 0.95 vs cf0.10 →
  deform 0.021, escape 0.019: the weaker catch buys +30% deform for a small escape cost, still clean. So the cf/deform trade is
  live at density — carry cf0.08 into the flock+division combos to maximize clean deform. (Contrast div06 n503, where cf0.10 is
  already ceiling — escape 0.092 — so cf0.08 there would need flock-containment help.)
- **[established, Batch 25 reading b24] NO-FLOCK div06 (n503) ESCAPE-FAILS AT cf0.10 (escape 0.092, r_max 1.05) — the 1C boundary-
  flux ceiling (~n442) reappears under division.** The div06 deform max (0.049) is real but NOT clean without a containment
  helper. This is why Batch 25's flagship adds strong flocking (1D: flock coherence contains at density) to hold the n503
  boundary flux — the direct test of whether flock-containment converts the div06 deform-max into a clean point.
- **[established, Batch 25 reading b24] move0.09 PRESERVES THE PARTITION UNDER DIVISION (seg 0.345 = control) BUT INVERTS THE
  MOVE/ESCAPE RULE — slower dividing cells escape MORE, not less (escape 0.128 vs the move0.12 sibling's 0.017).** Slower cells
  stir the two sides less (seg held) but the newly-divided cells pile up LOCALLY instead of spreading, and that local packing
  presses on the boundary → leak (r_max 1.00). So "move_speed is the dominant escape driver, lower=cleaner" (Batch 7) does NOT
  hold once division adds cells: at fixed division, lower move → worse containment. The one config that keeps the partition
  dividing (slow + dividing) hard-fails escape. Batch 25 s6 tests whether cf0.12 rescues its containment (the only shot at a
  dividing PARTITIONED blastula).
- **[established, Batch 26 reading b25] FLOCK CONTAINMENT AT CONFLUENCE STRENGTHENS WITH COHERENCE γ — STRONG (γ120) contains,
  MILD (γ60) RAMS (INVERTS the sparse-n peak).** At n503/cf0.10/div06: γ120 escape **0.032** (rescued the no-flock 0.092),
  γ60 escape **0.242** (WORSE than no-flock). Migration tracks it: γ120 migr 0.28-0.50 (HIGH), γ60 0.096 (dead). A COHERENT
  flock advects as an organized recirculating stream that stays off the boundary; a half-ordered γ60 flock is a disordered
  shear that drifts cells into the shell. So at CONFLUENCE the flocking non-monotone migration/containment peak sits at γ≥120
  — the OPPOSITE of the sparse-n b23/b24 reading (γ60>γ120 at n120). γ120 does NOT jam the dense flock (migr high) — the
  "over-align jams" worry is dead up to γ120. This makes strong flocking the confluent containment lever (use γ120 at n503).
- **[established, Batch 26 reading b25] THE CLEAN DIVISION+FLOCK DEFORM CEILING IS ~0.03 — flock CONTAINS but scatters no
  extra grid momentum.** The two deform≥0.045 slots (γ60 s3 0.045, cf0.08 γ120 s7 0.046) BOTH escape-FAIL (0.242, 0.119);
  the cleanest flock deform caps at 0.031 (s1, γ120 cf0.10 esc 0.032). Coherent circulation reorganizes the same momentum, it
  does not add push. So the clean deform ceiling under density+flock (~0.03) equals the 1C density ceiling (~0.027) — to lift
  it the remaining lever is per-cell mass (Batch 26 s4 probes m7e-5 under the γ120 container), not more flow. Deliverable
  "deform ≥0.04 CLEAN" NOT yet reached.
- **[established, Batch 26 reading b25] cf0.10 IS REQUIRED AT n503+FLOCK; cf0.08 LEAKS.** γ120 cf0.10 (s1) escape 0.032 vs
  γ120 cf0.08 (s7) escape 0.119 — same else. cf0.08 only holds at n374 (s4, escape 0.019). The clean-flock operating point at
  max density is n503/cf0.10/γ120. (cf-density coupling: n120→cf0.05, n374→cf0.08, n503→cf0.10.)
- **[established, Batch 26 reading b25] THE DIV-RATE RE-MIX THRESHOLD ∈ (0.02, 0.03) — MINIMAL DIVISION IS A CLEAN PARTITIONED
  BLASTULA.** div0.02 (s5, γ0, cf0.08): seg **0.294** (≈ frozen ctrl 0.339, PRESERVED), escape **0.000**, collapsed 0, grew
  n120→182. div0.03+ re-mixes (b24 div03 seg 0.115). So gentle proliferation (n×1.5) below the stir threshold KEEPS the
  L/R seed — the one route to a dividing partitioned blastula found clean so far. Its cost: low deform (0.017) / migration
  (0.07). ALSO: div0.03+move0.09+cf0.12 (s6) preserves seg 0.311 AND rescues containment (escape 0.128→0.004) — but cf0.12
  induces minor collapse (0.0085, nn_min 0.0028), the 1C "confine-up rams at density" law reappearing at n235. So cf0.12 is
  above the ram threshold at n235; keep cf≤0.10 to stay collapse-free.

- **[established, Batch 27 reading b26] FLOCK CONTAINMENT AT CONFLUENCE IS NON-MONOTONE IN γ — a BELL peaked at γ≈120, NOT a
  monotone strengthening (b26 hypothesis FALSIFIED).** n503/cf0.10/div06 escape vs γ: γ60 **0.242** (b25) → γ120 **0.032** → γ160
  **0.107** → γ200 **0.165** → γ240 **0.107**. There is a CONTAINMENT WINDOW around γ120: below it (γ60) the flock is a disordered
  SHEAR that drifts cells into the wall; above it (γ≥160) the flock OVER-ALIGNS into a single coherent TRANSLATING stream — migration
  jumps 0.28→0.48–0.49 (the order parameter climbs monotonically with γ) — that marches into the boundary as a body (r_max 1.07–1.10,
  escape ↑). So more coherence buys MORE migration but WORSE containment past γ120; γ120 is the unique point coherent enough to
  circulate (contain) yet not so coherent it translates ballistically. This SHARPENS the b26-going-in "γ120 contains" reading:
  γ120 is an OPTIMUM, not a floor — do NOT raise γ past 120 at confluence. (Mechanistically unifies b12 "collective march into the
  wall" at moderate density with the confluent container: a translating flock always rams; only a circulating one contains.)
- **[established, Batch 27 reading b26] THE CLEAN DIVISION+FLOCK DEFORM CEILING IS A HARD ~0.03 — BOTH remaining density levers
  (more γ-flow, more per-cell mass) are DEAD.** Every deform≥0.04 slot escape-FAILS (ctrl γ0 0.049@esc0.092, γ160 0.043@0.107, mass7e-5
  0.041@0.207); the cleanest deform is γ120 0.031@esc0.032. Coherent circulation REORGANIZES the same scattered momentum, it does not
  add push (flow-lever dead). And mass7e-5 under the γ120 container re-leaks HARD (escape 0.207, r_max 1.16 — the container cannot hold
  more per-cell push at n503; mass-lever dead, reconfirms 1C "no confine holds mass≥1e-4 at density"). **Consequence: "deform ≥0.04
  CLEAN at n503" is UNREACHABLE in this operator set — ADOPT s1 (γ120/n503/cf0.10, deform 0.031 @ escape 0.032, collapsed 0) as the
  density INTEGRATION point; the clean-deform ceiling equals the 1C density ceiling ~0.03.**
- **[established→to-replicate, Batch 27 reading b26] A STRONG COHERENT FLOCK (γ120) PRESERVES A MINIMAL-DIVISION PARTITION — the
  FIRST candidate FULL 5-phenomenon integrated blastula (SINGLE SEED, replicating in b27).** div0.02/n182/cf0.08/γ120 (b26 s5) →
  seg **0.274** (≈ the γ0 frozen partition 0.294, PRESERVED — NOT re-mixed) at escape 0.0275, r_max **0.982 (no cell outside the
  membrane)**, collapsed 0, migr 0.24, deform 0.022. This {stability + partition seg 0.27 + division n182 + collective migration
  0.24} coexistence is the integration deliverable minus only high deform. It APPARENTLY CONTRADICTS the campaign law "partition is
  antagonistic to every active stir" (b23/b24: flocking-stir re-mixes the seed) — the resolution: (i) minimal division (n182) stays
  BELOW the proliferation-stir threshold (div-rate re-mix threshold ∈(0.02,0.03), b26), and (ii) a COHERENT γ120 recirculation
  advects the two halves as organized streams that preserve the large-scale L/R structure, whereas the b24 half-ordered γ60 shear
  interdiffused them. So "coherent circulation ≠ interdiffusion" — a well-locked flock can migrate WITHOUT mixing. CAVEAT: single
  seed (seed1); b26 s5's held seg could be a kinetic-lottery draw (b24 showed γ60 partition survival was seed-dependent). Batch 27
  replicates on seeds 2/3 (paired vs γ0) before adoption — if it holds, this is the Phase-1 INTEGRATION deliverable (5 of 5 rungs).
- **[established, Batch 28 reading b27] THE 5-PHENOMENON INTEGRATED BLASTULA IS CONFIRMED + REPRODUCIBLE ACROSS 3 SEEDS.** The b26
  s5 point (γ120/div0.02/cf0.08/mass5e-5/n182) replicated: {stability (coll 0) + L/R partition + division + collective migration}
  COEXIST on all three seeds — every γ120 slot held seg 0.18–0.28 (seed1 0.274, seed2 0.280, seed3 0.184), all ≫ the n182 mixed
  floor (~0.06), at escape≈0, coll 0, migr 0.29–0.42, deform ~0.02. This is the Phase-1 integration deliverable (5 of 5 rungs
  present; only deform is a weak leg at ~0.02). The lone hard requirement: DIVISION MUST STAY MINIMAL (div0.02, n182) — the flock
  coexists with the partition only below the proliferation-stir threshold.
- **[established, Batch 28 reading b27] THE γ120 FLOCK is a MILD PARTITION-ERODER, NOT NEUTRAL — it regresses seg toward a ~0.25
  attractor (overturns b26 "circulation preserves the seed").** Paired γ0→γ120 across 3 seeds: seed2 0.220→0.280 (Δ+0.060), seed3
  0.378→0.184 (Δ−0.194), seed1 ~0.294→0.274 (Δ−0.020); 3-seed γ120 mean 0.246 < γ0 mean 0.297. The flock SHARPENS a weak seed and
  ERODES a strong one, converging every realization onto seg~0.25 — so b26's "coherent circulation preserves the L/R seed" was a
  seed1 kinetic-lottery read; the true effect is partial homogenization that never destroys the partition (all γ120 ≥0.18). Net:
  the partition robustly SURVIVES the flock (that is the integration win), but the flock does mildly mix it — do not claim the flock
  preserves seg.
- **[established, Batch 28 reading b27] ESCAPE≈0 and STRONG MIGRATION are ANTAGONISTIC at n182 — the two clean escape-closers BOTH
  tame the flock.** cf0.08→0.10 (seed1) closed escape 0.0275→**0.0055** with NO ram (coll 0, r_max 0.907 — the ram threshold is
  above cf0.10 at n182, unlike the n442 ceiling of cf0.10) BUT migr collapsed 0.24→0.136. move0.12→0.09 closed escape→**0.000**
  (r_max 0.859, seg 0.399) BUT migr 0.130. Both routes suppress the coherent stream (cf damps translation; move0.09 starves the
  flock, per the b12 speed-law). The strong-migration point (seed2 γ120/cf0.08 migr 0.42) grazes at escape 0.031 (r_max 0.97, no
  cell OUTSIDE). So at n182 there is (so far) NO single point with migr≳0.3 AND escape≈0 — Batch 28 maps the cf/move knee.
- **[established, Batch 28 reading b27] DIVISION IS THE DEFORM SOURCE INSIDE THE INTEGRATED BLASTULA, but re-mixes the partition —
  div-threshold ∈(0.02,0.03) HOLDS under the γ120 flock.** div0.03 (seed1, n235, cf0.10) → deform **0.029** (batch-max clean, 1C
  law holds: proliferation reshapes the shell) but seg dropped to **0.167** (re-mix toward floor) — the flock does NOT shift the
  re-mix threshold. div0.02 is the partition-preserving division ceiling; buying deform via more division costs the partition.
- **[established, Batch 28 reading b27] γ120 IS THE CONTAINMENT OPTIMUM AT n182 TOO — the window LOW side leaks (γ90 HARD-FAILS).**
  γ90 (seed1, n182) → escape **0.154**, r_max **1.049** (cells punch THROUGH the shell), migr 0.457 — a below-optimum flock is a
  coherent stream that over-TRANSLATES into the wall (same failure mode as b26's γ≥160 at n503, mirrored on the low side). So the
  γ≈120 containment optimum is density-robust (held at n503 AND n182); milder alignment does NOT contain the low-density case
  better (falsifies the b27 explore premise).
- **[established, Batch 29 reading b28] THE CONTAINMENT KNEE IS cf0.09 (NOT cf0.10) — it closes escape to EXACTLY 0.000 WHILE
  KEEPING migration 0.29; the b27 escape↔migration antagonism is BROKEN.** At seed1/γ120/div0.02/move0.12/n182: cf0.085 → escape
  0.0165 migr 0.280; **cf0.09 → escape 0.000, migr 0.289, seg 0.411 (batch-best partition), r_max 0.892, coll 0** — vs cf0.10
  (b27) which had escaped 0.006 but killed migr to 0.14. So the antagonism was a cf-over-confinement artefact: cf0.09 is the knee
  that contains fully without the migration tax. cf0.09 REPLACES cf0.10 as the containment operating point at n182.
- **[established, Batch 29 reading b28] THE SPEED LEVER (move) PRESERVES MIGRATION FAR BETTER THAN cf-up — move0.11 is the
  migration-preserving containment knob; move0.10 is past the cliff.** At γ120/cf0.08/n182: move0.11 → migr **0.497** (escape 0.011,
  grazing, r_max 0.926); move0.10 → migr collapses to **0.093** (escape 0.000, over-tamed). So there is a sharp speed cliff between
  0.11 and 0.10: move0.11 keeps migration 2–5× higher than either move0.10 or the cf-up route at comparable escape. Use move0.11 (not
  cf-up, not move0.09) when strong migration must coexist with tight containment.
- **[established, Batch 29 reading b28] TWO STRICTLY-CLEAN (escape 0.000) FULL 5-PHENOMENON INTEGRATED BLASTULAE — integration is
  essentially DELIVERED.** (A) **s1 cf09** = γ120/div0.02/move0.12/cf0.09/mass5e-5/n182 → seg 0.411, migr 0.289, deform 0.014,
  r_max 0.892, coll 0 (strongest partition). (B) **s6 div025_m11** = γ120/div0.025/move0.11/cf0.08/mass5e-5/n200 → seg 0.307, migr
  0.380, deform 0.018, r_max 0.875, coll 0, division active — the BEST ALL-ROUNDER (strong on all 5 axes, strictly clean) and the
  candidate FINAL integrated operating spec. The ONE weak leg is deform (~0.018); mass has stayed 5e-5 all through INT.
- **[established, Batch 29 reading b28] AT n182 THE γ120 FLOCK BUYS CONTAINMENT + PARTITION-PRESERVATION, NOT MIGRATION (γ0 control
  overturns the "flock lifts migration" framing at this density).** γ0/cf0.09 (s7) → migr **0.374 (HIGHER than γ120's 0.289)** but
  escape 0.055 (LEAKS) and seg 0.223 (vs γ120's 0.411). So the flow/division advection already supplies the polar order at n182; the
  flock's causal contribution is that a coherent recirculation stays OFF the wall (escape 0.055→0.000) and preserves the seed. (The
  "flock lifts migration" law still holds at CONFLUENT n503, b26 — it is density-specific.)
- **[established, Batch 29 reading b28] DEFORM AND CONTAINMENT TRADE THROUGH move-SPEED under division.** div0.025 at move0.12/cf0.09
  (s4) HARD-FAILED (escape 0.090, r_max 1.046) but gave batch-max deform 0.0275 (fast cells scatter more momentum); the SAME div0.025
  at move0.11/cf0.08 (s6) is strictly clean but deform falls to 0.018. So the deform a dividing flock can scatter is limited by the
  speed the containment tolerates — a fixed knot the mass lever (untested at n182) may loosen.
- **[rejected, Batch 29 reading b28] γ140 ALIGN-UP DOES NOT HELP — it damps migration to 0.139.** γ140/cf0.09 (s5) → migr 0.139
  (over-alignment freezes net motion), escape 0.022 grazing, seg 0.309. γ120 is the sharp single optimum; no room to trade align-up
  for migration or containment (falsifies b28 explore c).

## Compelling results (the early, vivid runs — keep these as touchstones)
- **water disc** (`agent_mpm_disc_water_v3`): two cell types swim in a cohesive rotating water
  blob, form aligned streams, gently lobe the boundary; 0 escape.
- **elastic disc** (`agent_mpm_disc_elastic_v3`): a coherent stream of cells migrates across an
  elastic tissue (**polar order ≈ 0.46** — strong collective migration); rounder, 0.25% escape.
- **4-type showcase** (`agent_mpm_disc_4types_show`): four cell types with distinct behaviour
  (flocker / disperser / aggregator / noisy) in one disc; deforms strongly.
- **blastula + 4 types** (`agent_mpm_blastula_4types_v1`): the four types inside the two-blue
  membrane+core; collective push deforms the shell into an egg/teardrop; 0 escape.
These are the phenomenology to recover and understand — vivid flow, migration, and shape change.

## Open questions (Phase 1 — design experiments around these)
- **[open]** Which coupling strength (`agent_to_mpm.agent_mass`, `mpm_to_agent.k`) lets inner flow
  visibly **deform the membrane** while keeping cells non-collapsed and still flowing? (the central
  balance)
- **[open]** Does `flow_align` + gentle `mpm_spin` produce **collective migration** at high density
  without jamming? What sustains continuous flow at confluence?
- **[open]** How do two types **partition** (left/right)? Which mechanism drives it — differential
  `move_speed`/`div_rate`, chemical cross-repulsion (`deposit`+`chemotax` with opposite channels),
  or `agent_remodel` making each type stiffen its territory?
- **[open]** Does `cell_divide` (proliferation pressure) deform the blastula, and how does the
  membrane thickness/stiffness gate that?

## GLOBAL open theme — the stress ↔ deformation ↔ active-cell relationship
The central scientific object of this project is the **three-way coupling between the material
STRESS field, the material DEFORMATION field, and the ACTIVE CELLS** — where cells put stress,
how that stress maps to deformation, and how the resulting flow feeds back on the cells. This is to
be **understood empirically through observation + test/validation/falsification in the loop**, NOT
asserted from theory. Every batch should read the 2×2 (cells / stress / deformation / tracks) as
one coupled system and ask a falsifiable question about a link in that chain (e.g. "do cells raise
stress locally where they push?", "does membrane deformation lag cell flow?", "is deformation
localised to shear-stiff material only?"). Promote a link to [established] only after a run confirms
it; keep the rest [open]. The specific claims below are instances of this theme.

## Hypotheses to TEST in-loop (asserted from theory — validate or falsify; do not treat as fact)
- **[open, H1]** *Stress & deformation concentrate in the outer ring because only the elastic
  MEMBRANE is shear-stiff (μ>0); the liquid core (μ=0) bears no shear stress/deformation.* Predicted
  from the fixed-corotated law `2μ(F−R)Fᵀ + λJ(J−1)I` + liquid `mu=0`. **Falsify:** (a) make the core
  elastic (`youngs>0`, no liquid layer) → the stress/deform field should FILL the interior, not just
  the ring; (b) make the membrane liquid → the ring should VANISH. If either fails, our reading of
  where load lives is wrong. *(A general rule: whenever an observation is explained only from theory,
  log it here as [open] with a falsification test and let a batch settle it — do not promote to
  [established] until a run confirms it.)*
- **[open, H2]** *The `collapsed` metric double-counts freshly-divided daughters (spawn offset ≈
  0.2·r0).* Threshold now 0.15·r0 to exclude them; verify a division-only run reports collapsed≈0.
- **[open, H3] Collapse is DENSITY-DEPENDENT** (observed 2026-07-02, 400f, base coupling): 12 cells
  → collapsed 0.0 (+ segregation 0.29); 60 cells → 0.67 (pairs); 265 cells → 0.98. I.e. the
  hydrodynamic self-attraction grows with cell NUMBER; raising `repel.r0` alone does not fix it
  (12-cell r0=0.16 worked because SPARSE, not because r0 was big). **Falsify/validate:** sweep `n` at
  fixed coupling and fixed r0 → expect collapse onset above a critical density; then lower
  `mpm_to_agent.k`/`agent_to_mpm.agent_mass` and expect the critical density to rise. Consequence for
  Phase 1: reach confluence by keeping coupling weak, not by cranking exclusion.
  *Update 2026-07-02 (Batch 1):* density-dependence is now **[established]** (n↑→collapsed↑, near
  deterministic — see Established mechanisms). The per-cell clustering force is NOT active-driven
  (flow_align/glide rejected) nor drag-driven; lead suspect is now the **confinement** term (H5).
- **[rejected, H4]** Active `flow_align`/`glide` cause the clumping — see Established mechanisms
  (Batch-1 s0/s2 unchanged at 0.977/0.985). Superseded by **H5 (confinement)**.

## Rejected / dead ends (one line)
- **[rejected]** Beating the collapse by cranking `repel` gain/`r0` — exclusion cannot win against
  the collapse force. Reconfirmed Batch-1 s4: r0 0.02→0.04 & strength 8→20 moved nn_min only
  0.0002→0.0007 (still ≪ r0), collapsed 0.959, and DOUBLED accel (0.0025→0.0052). The fix is to
  remove the squeeze + control density, not stronger repulsion.
- **[rejected]** Passive agent↔MPM drag (`mpm_to_agent.k`, `agent_to_mpm.agent_mass`) as the collapse
  cause — low_k (Batch-0) and no_couple (Batch-1 s3) both inert.
- **[rejected]** `flow_align`/`glide` (active MIPS) as the collapse cause — Batch-1 s0/s2 inert.
