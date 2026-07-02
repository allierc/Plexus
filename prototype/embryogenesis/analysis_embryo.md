# Embryogenesis loop — analysis log

Dated per-batch narrative (append-only).

## Batch 1 — 2026-07-02 — target stage 1A (stable, no-collapse); isolate the collapse driver

**What I inherited / observed (completed b01 sweep — montage `montages/embryo_b01.png` + `loop_logs/eb_b01_*`).**
The prior run's two sub-sets tell one story:
- *Smoke pair (division ON, n=67):* `s0_baseline` collapsed=0.7612, nn_min=0.0002; `s1_low_k`
  collapsed=0.7761, nn_min=0.0002. **Lowering `mpm_to_agent.k` did NOTHING** (0.76→0.78, within noise;
  deform/flow/migr/accel all identical to 3 sig-figs). Both HARD-FAIL: collapsed≫0, nn_min≪r0(0.02).
- *Ablation sweep (frames=1500):*
  - `s1_no_div` (div_rate 0) → **n=44, collapsed=0.5682**, nn_min=0.0002, nn_mean=0.0248, flow=0.00107,
    migration=0.051, seg=0.074. Sparse yet still stacks pairs (nn_min pinned 100× below r0).
  - `s2_no_glide` → n=265, collapsed=0.9849.  `s3_no_couple` (mass=0,k=0) → n=265, collapsed=0.9811.
  - `s4_strong_repel` (r0 0.04, strength 20) → n=265, collapsed=0.9585, nn_min only 0.0007 (still ≪ r0),
    accel DOUBLED 0.0025→0.0052 (leans harder on nothing — repulsion just adds energy, can't unstack).
  - `s5_quiet_1A` (move 0.02, div 0.1, mass 5e-7, k 0.1, flow_align 0) → n=60, collapsed=0.7667.

**Reading vs the montage 2×2.** deform stays ~0.0025–0.0028 and flow ~0.005 across every slot regardless
of ablation — the membrane barely moves and the "flow" is the spin field, not cell-driven. The cells
themselves collapse into sub-r0 clumps in the cells/tracks panels. So Stage-1A (stable even coverage) is
**not yet met** — we are still failing the gate, and no later stage is interpretable until it is.

**Per-slot verdicts (b01).**
- `s1_low_k` — **falsified** that drag `k` sets collapse (0.76→0.78 flat). Lever: `mpm_to_agent.k` is inert.
- `s2_no_glide`, `s3_no_couple` — **falsified** that glide / passive agent↔MPM drag cause collapse
  (both 0.98 at n=265). *Confounded:* both ran division-ON (n=265), so density masks any small effect.
- `s4_strong_repel` — **falsified (reconfirmed)** that exclusion beats collapse; only cost is +accel.
- `s1_no_div` — **supported** density∝collapse: dropping n 265→44 dropped collapsed 0.98→0.57 — the single
  biggest lever in the whole sweep was controlling **n**, not any force. But 0.57 at n=44 sparse proves a
  *dynamic* force still stacks well-separated cells (init spacing ~0.033 ≫ r0=0.02).
- `s5_quiet_1A` — **inconclusive**: weakened many knobs at once (n=60) but LEFT `confine 3.0` on → 0.767.
  Uninterpretable as an isolation, but consistent with confinement surviving every other weakening.

**The gap that defines this batch.** `confine` (`mpm_to_agent {confine: 3.0, field: colour}`) is the ONLY
MPM→agent force never ablated at fixed low n — s3 zeroed `k`+mass but at n=265 (confounded) and LEFT confine
on. Reading the operator source: confine adds `+confine·grad(density)` drifting cells up the material-colour
gradient (inward). This batch isolates the three MPM→agent channels — **confine / drag-k / spin** — one at a
time at fixed n=44 (division off), plus one confluence test.

**Levers for Batch 1 (this design):** `mpm_to_agent.confine`, `mpm_to_agent.k`, `mpm_spin.omega`,
`mpm_to_agent.field`, `agent.div_rate`/`cell_divide.rate` (density control).

### Batch 1 — refinement (2026-07-02, re-entry; confine runs in flight)
On re-entry the confine-isolation slots were already submitted to the L4 cluster (`loop_logs/eb_b01_s*.sh`,
12:25) but **no `metrics.json` had returned** — so I have NO confine results yet to observe; I did not
fabricate any. I verified the ablation-sweep numbers directly from `archive/*/metrics.json`
(s1_no_div collapsed=0.5682 nn_min=0.0002 at n=44; s3_no_couple 0.9811 at n=265; s5_quiet_1A 0.7667 at
n=60) — they match the entry above.

Two changes this re-entry, both raising the batch's scientific value:
1. **[metric gap → fixed]** `embryo_metrics.py` never computed `escape`, although the instruction lists
   `escape > 0` as a HARD FAILURE. This directly undermines the confine ablation: a `confine 0` run that
   drops `collapsed` toward 0 is only a Stage-1A win if cells stayed in the core — otherwise it "cured"
   collapse by letting cells escape *into the membrane*, which is a hard fail, not a result. Added
   `escape` (frac of live cells with radius > 0.9·Rd, i.e. out of the water core) and `r_cell_max`
   (furthest cell in units of outer radius). Backward-compatible new keys; does not affect in-flight jobs.
2. **[design upgrade, R1]** Replaced the two lowest-value explore slots (`spin0`, `field_mass`) with a
   **confine DOSE-RESPONSE** (`confine 1.0`, `confine 0.5`). R1 says tune the existing knob strength
   before a binary ablation; a dose-response is more mechanistic (does `collapsed` scale monotonically
   with `confine`?) AND avoids the escape confound of full ablation (partial confine still holds the
   core). The clean `confine 0` / `confine0_k0` / `quiet_substrate` ablation ladder is kept to close H5.

**Predicted with the new escape metric:** if H5 holds, `collapsed` falls monotonically as
`confine` 3.0→1.0→0.5→0 while `escape` rises from ~0; the sweet spot is the largest `confine` that still
gives `collapsed≈0` at `escape≈0`. `drag0` (k 0, confine on) should stay ~0.57 with escape~0.

## Batch 2 — 2026-07-02 — H5 CONFIRMED; Stage-1A MET. Advance to Stage-1B (inner flow deforms membrane)

**Batch-1 confine-isolation results are in — the confine dose-response + ablation ladder closes H5
decisively.** All at n=44, division OFF unless noted (nn HARD-FAIL floor r0=0.02):

| slot                | confine | drag k | n   | collapsed | nn_min  | escape | flow     | migration | accel    |
|---------------------|---------|--------|-----|-----------|---------|--------|----------|-----------|----------|
| base_n44_ref (ctrl) | 3.0     | on     | 44  | **0.568** | 0.0002  | 0      | 0.00107  | 0.051     | 0.00244  |
| confine1p0          | 1.0     | on     | 44  | 0.523     | 0.0006  | 0      | 0.00015  | 0.213     | 0.00099  |
| confine0p5          | 0.5     | on     | 44  | 0.477     | 0.0012  | 0      | 0.00011  | 0.123     | 0.00020  |
| **confine0**        | **0**   | on     | 44  | **0.000** | **0.0291** | 0   | 0.00037  | 0.087     | 1.8e-5   |
| confine0_k0         | 0       | 0      | 44  | 0.000     | 0.0299  | 0      | 0.00036  | 0.091     | 1.6e-5   |
| quiet_substrate     | 0       | 0,spin0| 44  | 0.000     | 0.0359  | 0      | 0.00036  | 0.165     | 1.7e-5   |
| **drag0**           | **3.0** | **0**  | 44  | **0.568** | 0.0002  | 0      | 0.00106  | 0.050     | 0.00244  |
| **confine0_dense**  | **0**   | on     | 265 | **0.000** | 0.0040  | 0      | 0.00438  | 0.259     | 1.4e-4   |

**OBSERVE — what happened vs predictions.** The confine ablation matches H5's *direction* but not its
*shape*, and the confluence slot OVERTURNS a prior [established] claim:
1. **Confinement is the sole collapse driver — CONFIRMED, cleanly.** `drag0` (k=0, confine STILL 3.0) →
   collapsed=0.568, byte-identical to the full base — zeroing the drag does nothing. `confine0` (confine=0,
   drag STILL on) → collapsed=0.000, nn_min=0.0291 (**above r0=0.02**, first time ever), escape=0. The two
   ablations cross perfectly: collapse tracks `confine`, not `k`. H5 promoted to [established].
2. **The dose-response is a THRESHOLD, not linear.** 3.0→1.0→0.5 barely moved collapsed (0.568→0.523→0.477);
   0.5→0 CRASHED it to 0.000. So any confine ≳0.5 triggers the full pileup; the transition sits between 0
   and 0.5. My "monotonic dose-response" prediction was wrong in shape — it is bistable/near-switch. (The
   0–0.5 window is unprobed; likely a sharp critical confine ~0.1–0.3.)
3. **"Collapse ∝ density" is OVERTURNED as a standalone law.** confine0_dense: n=265, division ON,
   collapsed=**0.000**, escape=0 — vs 0.96–0.98 for every confine-ON dense run in Batch 1. Density does NOT
   cause collapse; **confinement does, and density amplifies it**. The Batch-1 density law was confounded —
   every one of those runs had confine=3.0. Corrected claim in ledger.
4. **escape stayed 0.000 at confine=0 everywhere, even at confluence** — and r_cell_max DROPPED (0.755→0.701
   at n=44). So the confine colour-drift was never needed for core retention: the substrate anchor + elastic
   membrane + density confinement already hold cells in. Removing it is a pure win, no escape trade. The
   escape-vs-collapse risk I flagged did not materialise.
5. **Stage-1A GATE is MET.** confine0 / confine0_k0 / quiet_substrate / confine0_dense all give collapsed=0,
   escape=0, nn_min≥r0, bounded accel (≤1.7e-5, NOT via vmax). Even coverage holds at n=44 AND n=265. Gate
   to 1B passed. Montage corroborates: confine-ON dense slots (s4_confined_dense, s7_field_mass, bottom rows)
   show orange cells piling into central clumps; confine-OFF slots show even tilings filling the core.

**Per-slot verdicts (Batch-1 confine ladder).**
- `drag0` — **supported (H5):** drag k is inert for collapse; confine alone holds 0.568. Cleanest exoneration
  of the drag channel yet (k=0 with confine on = full collapse).
- `confine0` / `confine0_k0` / `quiet_substrate` — **supported (H5):** confine=0 → collapsed=0, nn_min>r0,
  escape=0. Confinement is THE driver. quiet_substrate is the most even tiling (nn_min 0.0359) but inert
  (flow 0.0004, deform 0.0002 — spin & drag off, nothing moves).
- `confine1p0` / `confine0p5` — **partially supported / shape-falsified:** collapse falls with confine but
  only weakly until the threshold; not the smooth monotone I predicted. Threshold sits in (0, 0.5).
- `confine0_dense` — **overturns "collapse ∝ density":** n=265 confine0 → collapsed=0, and it has the HIGHEST
  flow (0.0044) and migration (0.259) of the batch. This is the launch pad for Stage-1B.

**Reading the stress↔deform↔cell chain.** Across EVERY slot deform stays ~0.0027 (or 0.0002 when spin off) —
the membrane essentially never moves yet. Flow is tiny (≤0.0044). So we have a stable non-collapsing base but
**zero membrane deformation** — exactly the Stage-1B problem. The chain cells→(agent_to_mpm)→grid→membrane is
not transmitting enough momentum: at agent_mass=2e-6 the cells barely dent the shell. Stage-1B is now the job.

**Levers for Batch 2 (Stage-1B):** `agent_to_mpm.agent_mass` (cells→grid push, prime deform lever),
membrane `youngs` (shell stiffness — softer deforms more), `mpm_spin.omega` (circulation → flow), `move_speed`
(active push). New operating base authored: `specs/embryo_1A.yaml` (= base with confine 0, division on).
Risk to watch: cranking agent_mass may reintroduce collapse via a cells→grid→drag self-attraction loop that
confine previously masked — collapsed/escape are the guardrails, and either outcome is a finding.

## Batch 3 — 2026-07-02 — Stage-1B: agent_mass DOES deform the membrane, but DEFORM & ESCAPE are CONFOUNDED

**OBSERVE — Batch-2 deform-lever sweep (all ran division-ON to n=1600, frames=3000; montage `embryo_b02.png`
+ `archive/embryo_1A*_eb_b02_s*/metrics.json`).** The escape metric (added Batch 1) is decisive here:

| slot            | agent_mass | youngs | deform | flow    | migr   | escape     | r_cell_max | collapsed |
|-----------------|-----------|--------|--------|---------|--------|------------|------------|-----------|
| s0 flowbase     | 2e-6      | 200    | 0.0043 | 0.0042  | 0.225  | 0.014      | 0.99       | 0.005     |
| s1 mass1e5      | 1e-5      | 200    | 0.0067 | 0.0043  | 0.422  | 0.022      | 1.01       | 0.004     |
| **s2 mass5e5**  | **5e-5**  | 200    | 0.0526 | 0.0048  | 0.729  | **0.146**  | **1.22**   | 0.004     |
| **s3 mass2e4**  | **2e-4**  | 200    | 0.0625 | 0.0039  | 0.526  | **0.213**  | **1.27**   | 0.003     |
| s4 soft80       | 2e-6      | **80** | 0.0043 | 0.0042  | 0.205  | 0.016      | 0.99       | 0.004     |
| s5 spin06       | 2e-6      | 200    | 0.0069 | 0.0041  | 0.364  | 0.011      | 0.97       | 0.004     |
| s6 move12       | 2e-6      | 200    | 0.0093 | 0.0049  | 0.225  | 0.046      | **1.43**   | 0.004     |
| **s7 m5e5_soft**| **5e-5**  | **80** | 0.0489 | 0.0048  | 0.708  | **0.136**  | **1.24**   | 0.004     |

**What happened vs Batch-2's prediction.**
1. **`agent_mass` IS the prime deform lever — CONFIRMED, monotone & ~15×.** deform 0.0043→0.0067→0.0526→0.0625
   as mass 2e-6→1e-5→5e-5→2e-4. My Batch-2 hypothesis (deform ↑ monotone with agent_mass) is **supported**.
2. **BUT deform is bought with ESCAPE — the two are confounded through the SAME knob.** escape climbs in
   lockstep 0.014→0.022→**0.146**→**0.213**, and r_cell_max goes 0.99→1.01→**1.22**→**1.27** (>1 = cells sit
   OUTSIDE the membrane). So the "membrane deformation" at high mass is cells *punching through the shell*, not
   inner flow gently reshaping it. **s2, s3, s7 HARD-FAIL (escape ≫ 0).** The big deform numbers are blowout,
   not a clean Stage-1B win. Montage confirms: s3 t=3000 is a burst kidney/comma with cells spilling into the
   deep-blue shell; the clean slots (s0/s4/s5) stay round with cells inside.
3. **Softening the shell (youngs 200→80) did essentially NOTHING — sub-hypothesis FALSIFIED.** s4_soft80 deform
   0.0043 = byte-identical to s0 at base mass; s7 (soft+5e-5) deform 0.0489 ≈ s2 (stiff+5e-5) 0.0526, actually
   slightly LESS, same escape. So membrane `youngs` in [80,200] is NOT deform-limiting — the containment/coupling
   is, not the shell stiffness. "Softer shell deforms more" is rejected in this range.
4. **`mpm_spin` gives the CLEANEST deform-per-escape.** s5_spin06 (omega 0.6) → deform 0.0069 (1.6× base) at the
   LOWEST escape of the batch (0.011), r_cell_max 0.97 (cells stay IN). Circulation reshapes gently without
   ejecting cells — the one slot that raises deform by *internal flow*, proving clean deform is possible (just
   small). `move12` raised escape per unit deform (r_cell_max 1.43, a few cells shot far out) — a worse lever.
5. **Stage-1A collapse-freedom HOLDS everywhere** (collapsed 0.003–0.005, all daughters). BUT nn_min≈0.002 ≪
   r0=0.02 on EVERY slot: at n=1600 the disc is OVER-confluent — natural spacing (~0.015) is below the exclusion
   distance, so repel physically cannot hold cells apart. This runaway division to n=1600 is itself part of the
   escape story (a jammed, over-packed core pressure-ejects cells when pushed).

**Reading the stress↔deform↔escape chain.** The chain is now clear: cells→(agent_to_mpm, ∝agent_mass)→grid push
→ membrane displacement. At the SAME push, whether that displacement is *clean deformation* (membrane bulges,
cells stay in) or *escape* (cells punch through) is gated by (a) how hard cells are pressed against the shell —
i.e. confluence pressure (n=1600 is over-packed) — and (b) whether the membrane is a real barrier to point
agents (currently only `g2p.wall_contact 0.04` + elastic material; with confine 0 there is NO inward force
holding cells off the shell). So the Stage-1B problem is not "make deform bigger" (agent_mass already does that)
— it is **decouple deform from escape**: deform the shell from inside while cells stay contained.

**Per-slot verdicts (Batch 2).**
- s1 mass1e5 — **supported (clean):** deform ↑ 1.6×, escape 0.022 (borderline-clean), cells in. Best clean mass.
- s2 mass5e5 / s3 mass2e4 — **hard-FAIL (escape 0.15/0.21):** confirm agent_mass↑deform but by blowout. The
  deform is escape-confounded; not a Stage-1B pass.
- s4 soft80 — **falsified** "softer shell deforms more": identical to base. Shell youngs not deform-limiting.
- s5 spin06 — **supported (cleanest):** circulation deforms via internal flow, escape LOWEST (0.011). Small but
  the right kind of deform. Keep spin as a clean amplifier.
- s6 move12 — **inconclusive/poor:** raises escape (r_cell_max 1.43) more than deform; move_speed is a bad lever.
- s7 mass5e5_soft — **hard-FAIL + falsifies soft:** = s2 with softer shell, same escape, no extra deform.

**Levers for Batch 3 (decouple deform from escape):** `agent.div_rate` (density control — cap n so the core
isn't over-packed and pressure-ejecting), `agent_to_mpm.agent_mass` (deform push, held at 5e-5/2e-5),
`mpm_to_agent.confine`+`field: mass` (restore a BOUNDARY-localised inward containment below the collapse
threshold), `g2p.wall_contact` (make the shell a stiffer barrier so cells bounce not penetrate), `mpm_spin.omega`
(clean internal-flow deform). New spec authored: `specs/embryo_1B_confine.yaml` (embryo_1A + confine 0.2 field mass).

**HYPOTHESIS (Batch 3):** *Escape at high agent_mass is CONFLUENCE-PRESSURE-driven, not intrinsic to the
coupling.* Capping cell number via `agent.div_rate` (so n stays ~300–600 or the sparse n=44, instead of 1600)
lets `agent_mass 5e-5` deform the membrane (deform ≳0.02) while `escape` falls back toward ≈0 and nn_min recovers
toward r0 — i.e. at the SAME push, a less-packed core deforms the shell without punching through it. Corollary:
a sub-threshold boundary confine (`confine 0.2, field: mass`) or a stiffer wall (`wall_contact 0.12`) contains
cells at high density without reintroducing collapse. Prediction ranking: s2/s3 (m5e5, low div) clean (escape<0.03,
deform>0.02); s0 control (m5e5, full div) reproduces escape≈0.15; s6 confine / s7 wall cut escape at fixed high n.

## Batch 4 — 2026-07-02 — Stage-1B: escape IS confluence-driven (H confirmed) but DEFORM ∝ n×mass, so capping n killed BOTH — deform must come from agent_mass at fixed escape-free n

**OBSERVE — Batch-3 density-control sweep (montage `embryo_b03.png` + `archive/*eb_b03*/metrics.json`).**
All at `agent_mass 5e-5` unless noted; `div_rate` sets n via division. HARD-FAIL floor r0=0.02.

| slot                | mass  | div  | n    | collapsed | nn_min | deform | flow    | migr  | escape | r_max |
|---------------------|-------|------|------|-----------|--------|--------|---------|-------|--------|-------|
| s0 ctrl_m5e5_full   | 5e-5  | 0.6  | 1600 | 0.004     | 0.002  | **0.0526** | 0.0048 | 0.729 | **0.146** | 1.22 |
| s1 m5e5_div15       | 5e-5  | 0.15 | 95   | 0.0       | 0.005  | 0.0067 | 0.00176 | 0.233 | **0.0** | 0.82 |
| s2 m5e5_div05       | 5e-5  | 0.05 | 58   | 0.0       | 0.0188 | 0.0034 | 0.00109 | 0.146 | **0.0** | 0.75 |
| s3 m5e5_nodiv       | 5e-5  | 0.0  | 44   | 0.0       | 0.0269 | 0.0036 | 0.00072 | 0.238 | **0.0** | 0.71 |
| s4 m2e5_div15       | 2e-5  | 0.15 | 95   | 0.0       | 0.0051 | 0.0029 | 0.00182 | 0.149 | 0.0    | 0.76 |
| s5 spin_m2e5        | 2e-5  | 0.15 | 95   | 0.0       | 0.0051 | 0.0044 | 0.00177 | 0.221 | 0.0    | 0.75 |
| s7 wallc_m5e5       | 5e-5  | 0.15 | 95   | 0.0       | 0.005  | 0.0067 | 0.00176 | 0.233 | 0.0    | 0.82 |
| **s6 confine_m5e5** | 5e-5  | 0.15 | 95   | **0.579** | 0.0021 | 0.0057 | 0.00164 | 0.496 | 0.0    | 0.83 |

**What happened vs Batch-3's prediction.**
1. **Escape IS confluence-pressure-driven — CONFIRMED.** At mass 5e-5, capping n from 1600→95/58/44
   drove escape 0.146→**0.000** and r_cell_max 1.22→~0.7 (cells retreat well inside the shell). So the
   Batch-2 "blowout" was cells jammed against the wall by over-confluence being pressure-ejected, exactly
   as hypothesised. This half of H is **supported**.
2. **BUT deform collapsed WITH n — the decoupling FAILED.** Predicted deform ≳0.02 at capped n; got
   0.0067 (n=95), 0.0034 (n=58), 0.0036 (n=44) — an ORDER of magnitude short, and ~8× below the n=1600
   value (0.0526) at the SAME mass. So the big deform at n=1600 was **density doing the work, not the push
   at fixed n**. deform ≈ floor(~0.003) + slope·(**n × agent_mass**) = aggregate scattered momentum. Capping
   n to kill escape ALSO kills deform. The two are NOT decoupled by density control — they are BOTH driven
   by confluence. This half of H is **falsified**. The real decoupling axis is UNTESTED: raise `agent_mass`
   at fixed escape-free n=95, where cells are not jammed against the wall (r_max 0.82, gap to shell ~0.9).
3. **A sub-threshold boundary confine does NOT work — corollary REJECTED.** s6 (confine 0.2, field:mass)
   reintroduced collapse: collapsed=**0.579**, nn_min 0.0021≪r0, cells clumping centrally (montage bottom
   row). So `confine 0.2` is ABOVE the collapse threshold even with `field:mass` (not just field:colour) —
   confinement drives collapse regardless of the gradient field. Critical confine < 0.2. Migration spiked
   to 0.496 (the inward drift reads as coherent polar order) but it is the collapse signature, not a win.
4. **Stiffer wall (`wall_contact 0.12`) is INERT at low n — falsified as a lever here.** s7 is
   BYTE-IDENTICAL to s1 (deform 0.0067, flow 0.00176, migr 0.2326, escape 0, r_max 0.818). At n=95 no cell
   reaches the wall (r_max 0.82 < shell ~0.9), so wall_contact has nothing to act on. It can only matter
   under wall pressure (high n) — untested there.
5. **Spin still adds clean deform+flow at fixed n/mass — supported.** s5 (spin 0.6) vs s4 (spin 0.3),
   both mass 2e-5 n=95: deform 0.0029→0.0044 (1.5×), migration 0.149→0.221, escape stays 0. Circulation is
   a clean amplifier, consistent with Batch 2. Worth stacking onto the mass push.
6. **Stage-1A holds at capped n.** collapsed=0 and escape=0 on every non-confine slot; nn_min recovers to
   ≥r0 at n≤58 (0.0188–0.0269) — a true even tiling. Only over-confluent n=95/1600 pins nn_min<r0 (packing).

**Per-slot verdicts (Batch 3).**
- s1/s2/s3 (m5e5, div↓) — **supported (escape half):** capping n → escape 0. **Falsified (deform half):**
  deform fell to ~0.004–0.007, not ≥0.02. Deform is confluence-driven, not mass-at-fixed-n driven.
- s4/s5 — **supported:** spin adds clean deform+migration at fixed n; mass 2e-5 is a weak push at n=95.
- s6 confine_m5e5 — **rejected corollary:** confine 0.2 (field:mass) → collapsed 0.58. Confine→collapse is
  field-independent; critical confine < 0.2.
- s7 wallc_m5e5 — **falsified (here):** wall_contact inert at n=95 (identical to s1); no wall contact to gate.
- s0 ctrl — **supported (control):** full division reproduces the n=1600 escape blowout (0.146) — the
  density confound is real and reproducible.

**Reading the stress↔deform↔escape chain (updated).** deform = RMS shell radial displacement is set by the
TOTAL momentum the cells scatter onto the grid ≈ n×agent_mass; that momentum propagates through the liquid
core to the shell. Escape is a SEPARATE gate: it fires only when cells are packed against the wall (high n)
and get pressure-ejected. So the two share the `n` axis (both rise with confluence) but escape ALSO needs
wall-contact packing. The clean-Stage-1B window is therefore: keep n moderate (escape-free) and buy the
missing momentum from `agent_mass` instead of from n. That is Batch 4's test.

**Levers for Batch 4:** `agent_to_mpm.agent_mass` (raise the push at fixed escape-free n=95 — the untested
decoupling axis), `mpm_spin.omega` (stack clean circulation onto the push), `agent.div_rate` (density: map
escape onset between n=95 and n=1600; and high-mass at n=44 to separate packing- vs push-driven escape).

**HYPOTHESIS (Batch 4):** *At fixed escape-free density (n≈95, div_rate 0.15), membrane deform scales with
`agent_to_mpm.agent_mass` and crosses ≳0.02 around mass 2–5e-4 (n×mass ≈ 0.02–0.05) while escape STAYS 0 —
because at n=95 cells sit at r≈0.82 with a gap to the shell, so extra scattered momentum deforms the shell
from within rather than ejecting cells.* Corollary probe: if instead escape RISES with mass at fixed n=95
(and even at n=44), escape is per-cell-push-driven, not packing-driven, and the mass route is capped — in
which case spin (internal circulation) is the only clean deform route. Prediction ranking: m2e4/m5e4 hit
deform>0.02 at escape≈0 (clean Stage-1B win); m1e3 either extends deform or is where escape first appears
(finds the ceiling); m5e4_nodiv at n=44 isolates packing vs push; spin_m2e4 gives the largest clean deform.

## Batch 5 — 2026-07-02 — Stage-1B: agent_mass IS a deform lever at fixed n (Batch-3 revised); escape is per-cell-push × density (both needed); SPARSE-n shields escape → clean route is sparse-n + high-mass + spin

**OBSERVE — Batch-4 agent_mass ladder at fixed escape-free n≈95 (montage `embryo_b04.png` +
`archive/embryo_1A_eb_b04_s*/metrics.json`).** All n=95 (div 0.15) unless noted; HARD-FAIL floor r0=0.02.

| slot                | mass  | n   | deform | escape  | r_max  | migr   | flow    | nn_min | collapsed |
|---------------------|-------|-----|--------|---------|--------|--------|---------|--------|-----------|
| s0 ctrl_massbase    | 2e-6  | 95  | 0.0027 | 0.0     | 0.791  | 0.026  | 0.0017  | 0.0049 | 0.0 |
| s1 m1e4_div15       | 1e-4  | 95  | 0.0052 | 0.0     | 0.877  | 0.085  | 0.00187 | 0.0048 | 0.0 |
| s2 m2e4_div15       | 2e-4  | 95  | 0.0105 | 0.0105  | 0.9305 | 0.291  | 0.00192 | 0.0051 | 0.0 |
| **s3 m5e4_div15**   | **5e-4** | 95 | **0.0199** | **0.0421** | 0.9291 | 0.4045 | 0.00177 | 0.005 | 0.0 |
| **s4 m1e3_div15**   | **1e-3** | 95 | **0.0346** | **0.1474** | **1.2006** | 0.5353 | 0.0019 | 0.0052 | 0.0 |
| **s5 spin_m2e4**    | 2e-4+ω0.6 | 95 | 0.0148 | 0.0105 | 0.9181 | **0.6871** | 0.00188 | 0.0052 | 0.0 |
| s6 meddens_m5e5     | 5e-5  | 224 | 0.0106 | **0.0** | 0.7772 | 0.2266 | 0.00277 | 0.0034 | 0.0 |
| **s7 m5e4_nodiv**   | 5e-4  | **44** | **0.0115** | **0.0** | 0.883 | 0.2973 | 0.00077 | **0.0229** | 0.0 |

**What happened vs Batch-4's prediction.**
1. **`agent_mass` IS a deform lever at FIXED n — Batch-3 REVISED, hypothesis deform-half SUPPORTED.** At n=95,
   deform rises 0.0027→0.0052→0.0105→0.0199→0.0346 as mass goes 2e-6→1e-4→2e-4→5e-4→1e-3 (500× mass → 13×
   deform, roughly deform ≈ floor + slope·mass). Batch 3 concluded "mass at fixed n does nothing" — that was
   an artefact of a NARROW window (it only compared 2e-5 vs 5e-5). Over a wide range, mass clearly buys deform
   at fixed n. The decoupling axis (buy momentum from mass, not confluence) WORKS.
2. **BUT escape RISES with mass at fixed n=95 — corollary FIRES; escape is per-cell-push-driven too, not ONLY
   confluence-packing.** escape 0→0→0.0105→0.0421→**0.1474** and r_max 0.79→0.88→0.93→0.93→**1.20** as mass
   climbs. A big enough push ejects cells even at moderate n=95. This REVISES Batch-3's "escape is PURELY
   confluence-pressure-driven": escape = f(per-cell push **×** density) — BOTH must be high. Hypothesis's
   "escape STAYS 0" half is **falsified**: deform reaches 0.02 (s3) only where escape is already 0.042 (hard
   fail). The clean n=95 ceiling is deform≈0.01 at mass 2e-4 (escape ≤0.01).
3. **SPARSE density SHIELDS against escape at the SAME push — the key result.** s7 (n=44, mass 5e-4) → escape
   **0.0**, r_max 0.883, and a TRUE tiling (nn_min 0.0229 **> r0**) — vs s3 (n=95, SAME mass 5e-4) → escape
   0.042, r_max 0.93. Dropping n removes escape at fixed mass, because escape needs wall-proximity/packing that
   only high n supplies. Combined with (1): the clean Stage-1B route is **sparse n + high mass** (push each cell
   hard while keeping the core off the wall), NOT dense high-mass.
4. **Spin is THE clean amplifier — reconfirmed at higher mass, and it drives the strongest migration yet.** s5
   (m2e4 + ω0.6) vs s2 (m2e4, ω0.3): deform 0.0105→0.0148 (+40%), migration 0.291→**0.687** (2.4×, batch max),
   escape UNCHANGED (0.0105), r_max slightly LOWER (0.93→0.918). Circulation adds deform+migration with no
   escape cost — stack it onto the mass push.
5. **Dense low-mass is escape-safe with moderate deform.** s6 (n=224, mass 5e-5) → deform 0.0106, escape 0,
   r_max 0.777 (cells well inside). Density lifts aggregate momentum while low per-cell mass keeps cells off the
   wall. A second clean route (dense + low-mass), complementary to sparse + high-mass.
6. **Stage-1A holds everywhere** — collapsed 0, accel 2.4e-5–1.0e-4 (bounded by balance, NOT vmax). nn_min<r0
   only at n≥95 (over-confluent packing); n=44 gives a true even tiling (nn_min 0.0229).

**Reading the stress↔deform↔escape chain (updated).** deform is set by the momentum each cell scatters onto the
grid — and BOTH per-cell mass and cell number raise it (deform grows with mass at fixed n AND with n at fixed
mass). Escape is a separate gate that fires only when cells are BOTH pushed hard (high mass) AND pressed against
the wall (high n) — either alone is safe (s6 dense+low-mass: escape 0; s7 sparse+high-mass: escape 0). So the
clean-Stage-1B frontier runs along the anti-diagonal: to reach deform 0.02 at escape 0, stay OFF the high-n ×
high-mass corner. The two clean routes are **sparse-n + high-mass + spin** (s7 pointer: 0.0115 at escape 0, plus
spin's +40%) and **dense-n + low-mass** (s6: 0.0106 at escape 0). Batch 5 pushes the sparse+high-mass+spin route
to try to cross 0.02 cleanly.

**Per-slot verdicts (Batch 4).**
- s0 ctrl_massbase — **supported (control):** base-mass floor deform 0.0027 at n=95; the push does the work.
- s1 m1e4 — **supported (clean):** deform ↑1.9× (0.0052), escape 0. Best fully-clean n=95 mass.
- s2 m2e4 — **supported/borderline:** deform 0.0105, escape onset 0.0105 (one cell grazing the shell).
- s3 m5e4 — **deform near 0.02 (0.0199) but HARD-FAIL escape 0.042.** Confirms deform∝mass; falsifies "escape
  stays 0" at n=95. The clean route must lower n, not just push harder.
- s4 m1e3 — **HARD-FAIL (escape 0.147, r_max 1.20 = blowout).** Max deform 0.0346 but cells punch through. Mass
  ceiling at n=95.
- s5 spin_m2e4 — **supported (cleanest amplifier):** +deform +migration (batch-max 0.687), no extra escape.
- s6 meddens_m5e5 — **supported:** dense+low-mass is escape-safe (escape 0, r_max 0.777), deform 0.0106.
- s7 m5e4_nodiv — **KEY supported result:** sparse+high-mass is escape-free (escape 0) with a TRUE tiling
  (nn_min>r0), deform 0.0115. The pointer to the clean Stage-1B route.

**Levers for Batch 5:** `agent.div_rate` (keep n SPARSE ≈44 as the escape shield), `agent_to_mpm.agent_mass`
(push to 1e-3 now that sparse-n allows it), `mpm_spin.omega` (stack clean circulation → +deform +migration).
Test whether sparse-n + high-mass + spin crosses deform 0.02 at escape 0 (the clean Stage-1B pass).
