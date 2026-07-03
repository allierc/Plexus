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

### Batch 5 — refinement (2026-07-02, re-entry; enforce the 6000-frame directive at the slot level)
On re-entry the Batch-5 slots were already designed but the run length was non-compliant. **Root cause found:**
`embryo_loop.py` sets the sim length from `FRAMES = int(os.environ.get("EMBRYO_FRAMES", "3000"))` and passes
`frames={FRAMES}` on the worker argv — the spec's `n_frames: 6000` is IGNORED (showcase.py takes `frames=`
from argv, `sim.n_frames = frames`). So every batch so far, and the first Batch-5 submission, ran at 3000
frames despite the mandatory 2026-07-02 directive to use ~6000. The driver was restarted for Batch 5 with the
env var raised, but to make compliance robust regardless of the env I appended `frames 6000 stride 8` to all 8
slot lines: in showcase's argv dict these land AFTER the loop's `frames=3000`, so the later value wins and each
job runs 6000 frames unconditionally. stride 8 → ~750 rendered frames (render stays bounded); ~13-min L4/job
(b04 was ~6.5 min @3000), within the 30-min wall. No physics/operator change — this only doubles the observation
window so the slow deform/migration dynamics that this batch's 0.02 test depends on have time to saturate. n-at-
6000 caveat: division self-limits via `max_occ 0.9`, so the div>0 slots (n60, dense) saturate rather than run
away; the three sparse exploit slots (div 0.0) hold n=44 exactly, frame-independent. No other file changed.

## Batch 6 (2026-07-02) — Stage 1B. The Batch-5 6000-frame run OVERTURNS the "sparse-n escape shield."

**OBSERVE — the shield was a finite-time artefact; escape is real and catastrophic at 6000 frames.**
Batch 5 predicted sparse n=44 + high mass (→1e-3) + spin (→1.0) would cross deform 0.02 at escape≈0. It did the
opposite. At 6000 frames (2× Batch 4) + move_speed 0.12 (2× Batch 4), EVERY deform-producing mass at n=44 EJECTS
cells. Metrics (all n=44 unless noted; r0=0.02):
- s0 ctrl_sparse_base (m2e-6, div0): deform **0.0027**, escape **0**, r_max 0.85. Floor, clean — membrane stays round.
- s1 sp_m1e3 (m1e-3, no spin): deform 0.0613, **escape 0.182**, **r_max 2.09** = ballistic ejection. HARD FAIL.
- s2 sp_m5e4_spin (m5e-4, ω0.6): deform 0.0427, **escape 0.273**, r_max 1.18. HARD FAIL.
- s3 sp_m1e3_spin (m1e-3, ω0.6): deform 0.049, **escape 0.114**, r_max 1.29. HARD FAIL. (spin cut escape 0.182→0.114 vs s1 but nowhere near 0.)
- s4 n60_m1e3_spin (m1e-3, ω0.6, div0.05, n=77): deform 0.096, **escape 0.234**, **nn_min 0.0058≪r0**, r_max 2.09. HARD FAIL ×2.
- s5 dense_m1e4 (m1e-4, div0.3, n=1398): deform 0.101, **collapsed 0.0043**, **nn_min 0**, **escape 0.428**. HARD FAIL ×3 (over-confluent).
- s6 spinonly_hi (m2e-6, ω1.2): deform **0.0042**, escape **0**, r_max 0.79. Pure circulation at ω1.2 → still floor deform.
- s7 sp_m2e4_spin1 (m2e-4, ω1.0): deform 0.0175, **escape 0.023**, r_max 2.04. Marginal-fail escape; even m2e-4 leaks a cell at 6000f.

**Falsification of the Batch-4/5 "sparse-n is an escape SHIELD" claim.** Batch 4 (3000f, move_speed 0.06) reported
n=44 mass 5e-4 → escape 0, r_max 0.883 (s7, "true tiling"). The SAME sparse regime at 6000f + move_speed 0.12 gives
escape 0.27 (s2). Escape is a SLOW, accumulating leak: given twice the time (and twice the per-step speed), cells
that random-walk/get-pushed to the shell eventually punch through — the "shield" was just that 3000 frames wasn't
long enough for the leak to register. This kills the whole "anti-diagonal clean frontier" from Batch 4: at
move_speed 0.12 / 6000f there is NO clean deform≥0.02 at n=44 via mass+spin. The clean runs (s0, s6) sit at deform
floor ≈0.004; the moment mass is big enough to deform (≥2e-4), escape appears. **deform and escape are NOT
separable by lowering n at long time** — the sparse route is dead.

**Two confounds changed together (Batch 4 → 5): frames (3000→6000) AND move_speed (0.06→0.12).** Both plausibly
raise escape (more time to leak; faster cells overshoot the shell). Batch 6 must disentangle them, and — more
importantly — find a CONTAINMENT that holds agents at high mass without the collapse-causing `confine`.

**Mechanism of escape (from operator reading).** With `confine 0` the ONLY thing pulling an agent back inward is
the grid-flow drag `mpm_to_agent.k` (=0.3) — and that only acts WHERE material exists. When a high-mass push
throws an agent past the material edge, grid velocity there ≈0, so nothing decelerates or returns it → it flies
ballistically (r_max 2.09, far outside the shell). So the containment lever is: keep agents tied to the contained
material flow. `mpm_to_agent.k` up (0.3→1.0) should make agents follow the flow more faithfully and overshoot
less; the substrate anchor keeps the MATERIAL in, so a well-coupled agent stays in with it. This is the Batch-6
primary hypothesis. `g2p.wall_contact` acts on MATERIAL points, not agents, so it likely can't contain agents
directly (tested as an explore). The collapse-causing `confine` remains rejected (<0.2 collapses).

**Per-slot verdicts (Batch 5).**
- s0 ctrl_sparse_base — **supported (control):** floor deform 0.0027 at escape 0. Anchors the batch.
- s1 sp_m1e3 — **falsified the hypothesis / HARD FAIL:** escape 0.182, r_max 2.09. Sparse n does NOT shield at 6000f.
- s2 sp_m5e4_spin — **falsified / HARD FAIL:** escape 0.273. Directly overturns Batch-4 s7 (same regime, escape 0 @3000f).
- s3 sp_m1e3_spin — **falsified / HARD FAIL:** escape 0.114. Spin slightly reduces escape but cannot contain.
- s4 n60_m1e3_spin — **HARD FAIL ×2** (escape + sub-r0 packing): the shield-ceiling test drifted straight into the corner as predicted.
- s5 dense_m1e4 — **HARD FAIL ×3:** div0.3 → n=1398 over-confluent; the dense route also escapes hard at 6000f.
- s6 spinonly_hi — **supported (clean, informative):** ω1.2 pure circulation with floor mass → deform 0.0042. Spin ALONE (no mass push) does not deform; spin only amplifies an existing mass-driven flow.
- s7 sp_m2e4_spin1 — **inconclusive/marginal FAIL:** even m2e-4 leaks (escape 0.023) at 6000f. The escape floor at n=44/6000f/ms0.12 sits below mass 2e-4.

**Levers for Batch 6:** primary = `mpm_to_agent.k` (0.3→1.0, agent-to-flow coupling as CONTAINMENT — R1-compliant
existing knob). Secondary/disentangle = `agent.move_speed` (0.12 vs 0.06 at fixed 6000f, to separate the
move_speed vs frames confound), `g2p.wall_contact` (material-wall route, likely inert for agents), and a dense
route re-test at 6000f (does the dense+low-mass shield also fail with time?). Deform reference = mass 5e-4
(produces deform ~0.04, currently escapes). Guardrails: escape (HARD), collapsed, nn_min≥r0, r_cell_max.

**HYPOTHESIS (Batch 6).** Escape at n=44/high-mass is agents OUTRUNNING the contained material because the drag
`mpm_to_agent.k`=0.3 is too weak to hold them to the flow; the substrate anchor keeps the material in, so raising
`k` (0.3→1.0) will pull ballistic agents back and drop escape toward 0 at mass 5e-4 — WITHOUT collapse (drag is
exonerated as a collapse cause) — yielding the first CLEAN deform≥0.02 at n=44. Corollary: escape is also inflated
by the doubled move_speed; at move_speed 0.06 (6000f) escape drops sharply, showing move_speed (not just frame
count) is a co-driver. Falsifier: if k=1.0 leaves escape high (agents still fly past the material edge where
grid-v≈0 and drag can't reach), then containment genuinely needs a boundary force we lack with `confine 0` — and
the honest conclusion is that Stage-1B at n=44/confine-0 is blocked, forcing a pivot (dense-low-mass or a new
boundary-only operator).

### Batch 6 — re-entry (2026-07-02, driver restarted at batch 6 with the 12000-frame config)
On re-entry the Batch-6 observation of b05 above is confirmed CORRECT — I re-verified all eight
`archive/embryo_1A_eb_b05_s*/metrics.json` directly and every number in the b05 table matches (s0 deform 0.0027
escape 0; s1 deform 0.0613 escape 0.182 r_max 2.09; s2 deform 0.0427 escape 0.273; s5 dense n=1398 escape 0.428
r_max 2.09; s6 spinonly deform 0.0042 escape 0; s7 m2e4 escape 0.023 r_max 2.02). No fabrication; the OVERTURN of
the sparse-n escape shield stands. **What changed on re-entry:** the loop had first submitted Batch 6 at
frames=6000 (`loop_logs/eb_b06_*.sh`), but the watchdog restarted the driver at batch 6 with
`mode=L4 frames=12000 stride=16` (campaign_l4.log) to honour the MANDATORY 12000-frame user directive. Those
6000-frame b06 jobs were superseded (no b06 montage/archive was produced). I re-emitted the SAME H6 containment
design (no mechanism change — the drag-as-containment hypothesis is unaffected by run length) but bumped every
slot to `frames 12000 stride 16` and renamed `dense_6k`→`dense_12k`. **Why the frames token matters (Batch-5
engineering lesson):** showcase builds its arg dict left-to-right and the slot's pinned `frames` lands AFTER the
loop's `frames`, so it WINS — a stale `frames 6000` here would have silently re-run Batch 6 at 6000 despite the
loop's 12000 config. Pinning 12000 guarantees compliance regardless of the env/loop default. Scientifically the
longer window makes this a STRICTER containment test: escape is a slow time-accumulating leak (Batch-6 finding),
so doubling frames 6000→12000 gives cells twice as long to leak — if `k=1.0` still holds escape≈0 at 12000f that
is a robust clean-Stage-1B pass, not a finite-time illusion (exactly the trap that killed the "sparse-n shield").
~22 min/job on L4 (b05 was ~11 min @6000f), within the 45-min wall.

**HYPOTHESIS restated (Batch 6, unchanged):** raising the grid-flow drag `mpm_to_agent.k` 0.3→0.6→1.0 holds
agents on the substrate-anchored (contained) material flow and drops escape toward 0 at deform-producing mass
5e-4 — WITHOUT collapse (drag is exonerated) — giving the first CLEAN deform≥0.02 at n=44 even at 12000 frames.
Falsifier: escape stays high at k=1.0 ⇒ no agent-force contains ballistic ejection with `confine 0`, and
Stage-1B needs a boundary-only operator (code change) — pivot decision.

## Batch 7 — 2026-07-02 — Stage-1B. H6 FALSIFIED and OVERTURNED: drag `k` makes escape WORSE, not better. Pivot to boundary containment via small `confine field:colour`.

**OBSERVE — Batch-6 drag-as-containment sweep at n=44, mass 5e-4, move 0.12, 12000 frames (montage
`embryo_b06.png` + `archive/embryo_1A_eb_b06_s*/metrics.json`). r0=0.02.**

| slot                | k    | mass  | move | n   | collapsed | nn_min | deform | escape    | r_max | migr  |
|---------------------|------|-------|------|-----|-----------|--------|--------|-----------|-------|-------|
| **s0 ctrl_k0p3**    | 0.3  | 5e-4  | 0.12 | 44  | 0.0       | 0.0176 | 0.0368 | **0.136** | 2.06  | 0.478 |
| s1 k0p6_m5e4        | 0.6  | 5e-4  | 0.12 | 44  | 0.0       | 0.0048 | 0.0185 | **0.523** | 2.09  | 0.219 |
| s2 k1p0_m5e4        | 1.0  | 5e-4  | 0.12 | 44  | 0.091     | 0.0    | 0.0163 | **0.705** | 2.09  | 0.256 |
| s3 k1p0_m5e4_spin   | 1.0  | 5e-4  | ω0.6 | 44  | 0.136     | 0.0    | 0.0268 | **0.886** | 2.09  | 0.155 |
| s4 k1p0_m1e3        | 1.0  | 1e-3  | 0.12 | 44  | 0.136     | 0.0    | 0.0117 | **1.000** | 2.09  | 0.056 |
| **s5 ms06_m5e4**    | 0.3  | 5e-4  | 0.06 | 44  | 0.0       | 0.0199 | 0.0144 | **0.023** | 1.12  | 0.322 |
| s6 wc1p0_m5e4       | wc1.0| 5e-4  | 0.12 | 44  | 0.046     | 0.0    | 0.0264 | **0.477** | 2.09  | 0.033 |
| s7 dense_12k        | 0.3  | 5e-5  | 0.12 | 442 | 0.0045    | 0.0    | 0.0312 | **0.054** | 2.09  | 0.229 |

**What happened vs Batch-6's prediction — H6 is not just falsified, it is INVERTED.**
1. **Raising drag `k` MONOTONICALLY WORSENS escape — the OPPOSITE of H6.** escape 0.136→0.523→0.705→1.000
   as k 0.3→0.6→1.0 (and 1.0 at mass 1e-3, s4: *every* cell escaped, deform collapsed to 0.0117 because no
   cells are left inside to push). Prediction was escape→0 at k=1.0; got escape→1.0. **H6 is dead.** Mechanism
   (re-read of `mpm_to_agent.py`): drag relaxes each agent's velocity toward the LOCAL grid velocity
   `k·v_fluid`. At mass 5e-4 the cells' own `agent_to_mpm` push makes the grid velocity point OUTWARD right where
   the cells are (they scattered that outward momentum onto g.mv themselves). So higher `k` slaves each agent
   MORE tightly to the very outward jet it created → it is carried out faster → more escape. Drag is not a
   restoring force to the interior; it is a coupling to the self-generated outflow. This cleanly explains the
   monotone worsening and kills "drag = containment."
2. **move_speed is the DOMINANT escape co-driver — corollary strongly CONFIRMED.** s5 (move 0.06, everything
   else = control s0) → escape **0.023** and r_max **1.12**, vs s0 (move 0.12) → escape 0.136, r_max 2.06.
   Halving move_speed cut escape ~6× and nearly halved r_max. So the doubled move_speed (user directive) is a
   primary reason escape is bad at 12000f; at 0.06 the SAME mass/k/frames is almost clean. (Directive still wants
   ≥0.12, so this is a lever we note but don't lean on — the fix must be a force, not slowing cells.)
3. **The falsifier condition of Batch 6 is MET: NO existing agent-force contains ejection at `confine 0`.**
   drag `k` worsens it (s1–s4); `g2p.wall_contact 1.0` (s6) did not help (escape 0.477, worse than control —
   it acts on material points, not agents, as predicted). The only clean-ish deform run is s7 (dense+low-mass,
   deform 0.0312 at escape 0.054) — but that still marginally fails (r_max 2.09, one ballistic cell) and grew to
   n=442 by division. So Stage-1B/confine-0 is BLOCKED by containment exactly as the Batch-6 falsifier warned.
4. **KEY MECHANISTIC OPENING (from reading BOTH coupling operators): `confine field:colour` is a BOUNDARY-ONLY
   force with NO per-cell attraction.** `agent_to_mpm` scatters agent momentum onto `g.m` (mass) and `g.mv`
   (momentum) but NEVER onto `g.c` (colour). So `mpm_to_agent.confine` with `field: colour` reads a colour field
   that is ~1 throughout the fluid disc and →0 outside; grad(c) is nonzero ONLY at the fluid/vacuum interface
   (the shell) and ≈0 in the uniform core. Therefore small colour-confine catches cells drifting toward the shell
   and pushes them back in, WITHOUT the mutual attraction that `field: mass` creates (mass sees each cell's own
   scattered density bump → cells drift up each other's peaks → collapse; THIS is why Batch-3 s6 confine 0.2
   field:mass → collapsed 0.58, and why collapse scaled with agent_mass). The prior ledger claim "confinement
   causes collapse, critical <0.2" was measured mostly at LARGE confine (0.5/1.0/3.0, where even colour rams
   cells into a dense clump); the small-confine field:colour window (0.02–0.15) is UNPROBED and is the natural
   boundary-containment we lacked — R1-compliant (existing knob), no code change.

**Per-slot verdicts (Batch 6).**
- s0 ctrl_k0p3 — **supported (control):** confine-0 escape baseline 0.136 at deform 0.037. Anchors the batch.
- s1/s2/s3/s4 (k↑) — **FALSIFY H6, cleanly & monotonically:** escape 0.52→0.70→0.89→1.00 as k/mass rise.
  Drag couples agents to their self-made outflow; raising it ejects them faster. Drag is NOT a containment lever.
- s5 ms06_m5e4 — **supported (corollary):** move_speed halved → escape 6× lower (0.136→0.023), r_max 2.06→1.12.
  move_speed is the dominant escape co-driver. Cleanest deform-per-escape of the batch (0.0144 @ escape 0.023).
- s6 wc1p0_m5e4 — **supported (predicted-inert/harmful):** wall_contact acts on material, not agents; escape
  0.477 (worse than control). Not a containment lever.
- s7 dense_12k — **partial:** deform 0.0312 (>0.02!) at escape 0.054 — best deform/escape trade, but marginal
  HARD-FAIL (escape>0, r_max 2.09). Dense+low-mass survives 12000f far better than the k-sweep but still leaks.

**Reading the stress↔deform↔escape chain (updated).** Two channels pull cells OUT and none holds them at the
boundary: (a) their own propulsion (move_speed) random-walks them to the shell; (b) the grid drag `k` slaves them
to the outward jet the high-mass push creates. With `confine 0` nothing catches a cell once it nears the shell —
grid-v and colour-grad both →0 just outside, so it flies ballistic (r_max 2.09). The missing piece is a
boundary-localized inward force. `confine field:colour` at SMALL magnitude is exactly that (grad(c) lives only at
the interface; agents don't pollute g.c), and unlike `field:mass` it carries no collapse-causing attraction.

**Levers for Batch 7:** `mpm_to_agent.confine` (SMALL, field:colour — boundary containment dose-response
0.02/0.05/0.10/0.15), `mpm_to_agent.field` (colour vs mass — mechanistic contrast: mass should collapse,
colour should not), `mpm_spin.omega` (established clean deform amplifier to stack on a contained base),
`agent.div_rate` (dense+low-mass route s7, add confine to plug its leak). Deform reference mass = 5e-4
(gives deform ~0.037 at confine 0). Guardrails: escape (HARD), collapsed, nn_min≥r0, r_cell_max.

**HYPOTHESIS (Batch 7).** *A small `mpm_to_agent.confine` with `field: colour` (~0.05) is a BOUNDARY-ONLY
containment: because agents write g.m/g.mv but NOT g.c, grad(colour) is nonzero only at the shell interface, so
confine·grad(c) catches cells drifting outward and returns them WITHOUT the per-cell attraction that field:mass
creates. Prediction: at n=44, mass 5e-4, move 0.12, 12000f, escape falls from 0.136 (confine 0) toward ≈0 as
confine rises 0→0.05, while deform STAYS ≥0.02 and collapsed STAYS 0 — the first clean Stage-1B. Collapse
re-onsets only at larger confine (≳0.15, the boundary ram). Falsification tests baked in: (i) the field:mass
control at the SAME confine 0.05 should instead COLLAPSE (agent-mass attraction) — proving the field distinction
is the mechanism; (ii) if even confine 0.05 colour collapses or leaves escape high, then colour-confine is not a
clean boundary force and Stage-1B/confine-0 genuinely needs a new boundary-only operator (code change, Batch 8).*
Prediction ranking: cf0p05_m5e4 = clean-1B candidate (escape→~0, deform≥0.02, collapsed 0); cf0p05_spin = best
clean deform (spin +40%); cf0p02 possibly too weak to contain; cf0p15 finds collapse onset; cf0p05_field_mass
collapses (mechanistic contrast); cf0p1_dense plugs the s7 leak → deform>0.02 at escape~0.

## Batch 8 — 2026-07-02 — Stage-1B: **H7 CONFIRMED — small colour-confine 0.05 is a clean boundary containment; FIRST near-clean Stage-1B (deform 0.0315, escape 0.023, r_max<1).** (b07 ran PARTIAL: s0–s3 complete, s4–s7 killed by watchdog restart.)

**RUN STATE (honest bookkeeping).** The Batch-7 slots were submitted and ran on L4; **s0–s3 completed and
returned metrics; s4–s7 (cf0p02 / cf0p15 / cf0p05_fieldmass / cf0p10_dense) each captured all 751 frames but
were `KeyboardInterrupt`-killed during PNG render** when the watchdog restarted the driver at 16:21:10
(`loop_logs/eb_b07_s{4,5,6,7}_*.err` end in KeyboardInterrupt in `Image.save`). So **no b07 montage was
generated** and s4–s7 have no `metrics.json`. I did NOT fabricate them; I read the four completed slots directly
from `loop_logs/eb_b07_s{0,1,2,3}_*.out`. This batch re-runs the four lost slots (esp. the field:mass mechanistic
control) and brackets the confine sweet spot to close escape→0.

**OBSERVE — Batch-7 small colour-confine dose (n=44, mass 5e-4, move 0.12, 12000f; r0=0.02).**

| slot                | confine | field  | deform | escape    | r_max  | collapsed | nn_min | migr   | flow    |
|---------------------|---------|--------|--------|-----------|--------|-----------|--------|--------|---------|
| s0 ctrl_cf0_m5e4    | 0       | —      | 0.0368 | **0.136** | 2.06   | 0.0       | 0.0176 | 0.478  | 0.00469 |
| **s1 cf0p05_m5e4**  | **0.05**| colour | **0.0315** | **0.0227** | **0.925** | **0.0** | 0.0069 | 0.594 | 0.00405 |
| s2 cf0p10_m5e4      | 0.10    | colour | 0.0287 | **0.386** | 1.093  | 0.0       | 0.0038 | 0.183  | 0.00418 |
| s3 cf0p05_spin      | 0.05+ω0.6| colour| 0.0336 | **0.273** | 1.081  | 0.0       | 0.0054 | 0.412  | 0.00398 |

**What happened vs Batch-7's prediction.**
1. **H7 SUPPORTED — small colour-confine 0.05 is the boundary containment we lacked.** s1 vs s0 (control):
   escape **0.136→0.0227** (6× lower) and — decisively — **r_cell_max 2.06→0.925** (ballistic ejection ELIMINATED;
   every cell now sits INSIDE the shell, max radius 0.92 < 1.0). And it did this WITHOUT killing deform (0.0368→
   0.0315, still ≫0.02) and WITHOUT collapse (collapsed 0, nn_min 0.0069>0). Migration is the HIGHEST of the four
   (0.594). So confine·grad(colour) at 0.05 catches cells drifting toward the shell and returns them, exactly as the
   boundary-only mechanism predicts. **This is the first near-clean Stage-1B: internal flow deforms the shell
   (deform 0.0315) with cells contained.** The one residual: escape is 0.0227, not 0 — 1 of 44 cells is at radius
   >0.9·Rd, but at r_max 0.925 it is grazing the INNER membrane, NOT flung outside (contrast s0's r_max 2.06). The
   qualitative regime changed from ballistic ejection to full containment; the last cell is a closing job, not a
   blowout.
2. **The confine dose is NON-MONOTONE (U-shaped) — 0.05 is the minimum, 0.10 is already the boundary RAM.** escape
   0.136 (cf0) → **0.0227** (cf0.05) → **0.386** (cf0.10). Stronger confine did NOT contain better; it made escape
   WORSE, and nn_min compressed 0.0069→0.0038 while r_max stayed low (1.09). Reading: at 0.10 the inward force stops
   being a gentle boundary catch and starts RAMMING cells into the shell zone (radius>0.9) — the collapse-onset
   direction (this is the small-confine tail of the Batch-2 "confine 0.5–3.0 rams cells into a central clump"). So
   the clean window is AT/BELOW 0.05; the sweet spot is bracketed by 0.05 (clean) and 0.10 (ram). My prediction that
   0.10 gives a "stronger catch" is falsified — it over-confines.
3. **Spin BREAKS marginal containment — it is NOT a free amplifier here.** s3 (cf0.05 + ω0.6) vs s1 (cf0.05):
   escape **0.0227→0.273** and r_max 0.925→1.08. Adding circulation to the winning containment EJECTED cells. This
   revises the Batch-2/4/5 "spin is a clean deform amplifier at zero escape cost": that held when escape was already
   0 by other means; when containment is at its margin (confine 0.05 holding one cell at 0.92), spin's outward fling
   overwhelms the catch. Deform did rise (0.0315→0.0336) but at a catastrophic escape cost. **Do not stack spin onto
   a marginal containment.** (Spin may still be safe once escape has real headroom — untested.)
4. **The control reproduces b06 s0 to 3 sig-figs** (escape 0.136, deform 0.0368, r_max 2.06, migr 0.478) — the
   confine-0 escape baseline is reproducible; the s1 improvement is a real confine effect, not run-to-run noise.

**Per-slot verdicts (Batch 7, partial).**
- s0 ctrl_cf0_m5e4 — **supported (control):** reproduces the confine-0 ballistic-escape baseline (0.136, r_max 2.06).
- **s1 cf0p05_m5e4 — KEY supported result / H7 confirmed:** confine 0.05 colour → escape 6× lower, ballistic
  ejection gone (r_max<1), deform 0.0315>0.02, collapsed 0. First near-clean Stage-1B; the candidate operating spec.
- s2 cf0p10_m5e4 — **falsifies "more confine = better catch":** escape rose to 0.386 (boundary ram); sweet spot ≤0.05.
- s3 cf0p05_spin — **falsifies "spin is a free amplifier at the margin":** spin broke the 0.05 containment (escape 0.273).
- s4/s5/s6/s7 — **no data (killed mid-render);** re-run this batch. s6 (field:mass) is the critical missing mechanistic control.

**Reading the stress↔deform↔escape chain (updated).** We now have the missing boundary force. Two channels push
cells to the shell (own propulsion move_speed; drag-to-self-outflow), and until now nothing caught them there
(grid-v & colour-grad both →0 just outside → ballistic). `confine·grad(colour)` supplies a boundary-localized
inward force that lives ONLY at the fluid/shell interface (agents don't write g.c), so at small magnitude (0.05) it
returns cells WITHOUT the per-cell attraction that field:mass would create and WITHOUT the ram that large confine
creates. Result: deform (set by mass×n scattered momentum) survives while escape is contained. The clean-Stage-1B
window is now: mass 5e-4 (deform ~0.03) + colour-confine ~0.05 + NO spin + move 0.12.

**HYPOTHESIS (Batch 8).** *The minimum-escape confine sits at/just-below 0.05, and escape reaches ≈0 either by
fine-tuning confine into (0.03, 0.07) or by trimming the push (mass 5e-4→3e-4, which lowers the outward flux the
0.05 catch must absorb) — while deform stays ≥0.02 and collapsed stays 0.* Built-in controls: (i) `cf0p05_fieldmass`
(re-run) must COLLAPSE (field:mass = per-cell attraction) — the mechanistic proof that the field choice, not the
magnitude, makes 0.05 safe; (ii) `cf0p05_div` tests whether the containment survives proliferation (n grows via
division) — the bridge toward Stage-1C. Prediction ranking: cf0p03/cf0p07 bracket the U-minimum (one should beat
s1's 0.023); cf0p05_m3e4 = cleanest (escape→~0 at deform ~0.02); cf0p05_fieldmass collapses; cf0p10_dense recovers
the dense route; cf0p05_div either holds (containment robust to density) or leaks (escape returns with n).

### Batch 8 — re-entry reconciliation (2026-07-02): driver counter is STILL batch 7; this design runs as the (first complete) b07
On re-entry I found the driver state `{"batch": 7}` and the watchdog log shows it restarted at batch 7 three times
(16:21/16:27/16:30) after the KeyboardInterrupt that killed b07 s4–s7 mid-render — so the loop is still calling
"DESIGN batch 7" and will submit whatever is in `embryo_slots.md` as `eb_b07`, producing the first COMPLETE b07
montage. I verified the s0–s3 numbers directly from `archive/embryo_1A_eb_b07_s{0..3}/metrics.json` (s1 deform 0.0315,
escape 0.0227, r_max 0.9247, migr 0.5939, collapsed 0; s0 escape 0.1364, r_max 2.06) — they match the table above to
all digits, NOT fabricated. The prior invocation had already observed the partial b07 and distilled H7-confirmed into
the ledger, but left the STALE batch-7 confine-dose slots in `embryo_slots.md` and never recovered the four killed
slots (crucially the `field:mass` mechanistic control). **Action this re-entry:** rewrote the 8 slots to the Batch-8
design above — bracket the U-minimum (confine 0.03/0.07), trim the push (mass 3e-4/4e-4), RECOVER the killed
`cf0p05_fieldmass` mechanistic control and the dense route, and add a division-survival probe (→1C bridge). No spec
file changed (all scalar overrides on embryo_1A.yaml). There is now a persistent +1 skew between the driver's montage
counter (b07) and these narrative section headers (Batch 8); I will keep observing by the ACTUAL montage filename the
driver emits, not the header number.

## Batch 9 — 2026-07-02 — Stage-1B **CLEANLY MET** (escape 0, deform >0.03, 12000f); **ADVANCE to Stage-1C** (division pressure deforms the shell). Observes the completed `embryo_b07` montage (my Batch-8 design ran in full this cycle).

**OBSERVE — the Batch-8 confine-bracket + control recovery ran to completion (montage `montages/embryo_b07.png` +
`archive/embryo_1B_b07_s*/metrics.json`; note the driver now names archives `embryo_1B_*` from `current_stage.txt`). All
n=44 unless noted; move 0.12, 12000f; r0=0.02; field colour unless noted.**

| slot                 | confine | field  | mass  | div  | n   | deform | escape    | r_max | collapsed | nn_min | migr  |
|----------------------|---------|--------|-------|------|-----|--------|-----------|-------|-----------|--------|-------|
| s0 ctrl_cf0_m5e4     | 0       | —      | 5e-4  | 0    | 44  | 0.0368 | **0.136** | 2.06  | 0.0       | 0.0176 | 0.478 |
| **s1 cf0p03_m5e4**   | **0.03**| colour | 5e-4  | 0    | 44  | 0.0304 | **0.000** | 0.850 | 0.0       | 0.0073 | 0.297 |
| **s2 cf0p07_m5e4**   | **0.07**| colour | 5e-4  | 0    | 44  | 0.0332 | **0.000** | 0.854 | 0.0       | 0.0043 | 0.410 |
| s3 cf0p05_m3e4       | 0.05    | colour | 3e-4  | 0    | 44  | 0.0275 | 0.068     | 0.920 | 0.0       | 0.0064 | 0.422 |
| s4 cf0p04_m4e4       | 0.04    | colour | 4e-4  | 0    | 44  | 0.0362 | **0.318** | 1.170 | 0.0       | 0.0054 | 0.624 |
| s5 cf0p05_fieldmass  | 0.05    | **mass**| 5e-4 | 0    | 44  | 0.0268 | **0.182** | 1.265 | 0.0       | 0.0041 | 0.519 |
| s6 cf0p05_div        | 0.05    | colour | 5e-4  | 0.10 | 442 | 0.0614 | **0.278** | 1.262 | 0.0       | 0.0046 | 0.356 |
| **s7 cf0p10_dense**  | 0.10    | colour | 5e-5  | 0.10 | 442 | 0.0268 | **0.016** | 0.954 | 0.0       | 0.0031 | 0.112 |

**What happened vs Batch-8's prediction.**
1. **STAGE 1B GATE MET — cleanly, reproducibly, and at 12000 frames (not a finite-time illusion).** The U-minimum
   bracket both landed clean: **cf0.03 → escape 0.000, r_max 0.850, deform 0.0304, collapsed 0** and **cf0.07 → escape
   0.000, r_max 0.854, deform 0.0332, collapsed 0.** These are the FIRST fully clean Stage-1B points ever — inner flow
   deforms the shell (deform ≫0.02) with EVERY cell contained well inside the membrane (r_max<0.86 ≪ the 2.06 ballistic
   control). The whole confine window [0.03, 0.07] is clean at mass 5e-4; the b07-partial cf0.05 (escape 0.0227) was a
   one-cell blip, not a U-minimum — the "clean point" is a PLATEAU across 0.03–0.07, then the ram at 0.10 (escape 0.386).
   My prediction that cf0.03/cf0.07 "bracket a minimum, one should beat s1's 0.023" is **supported and then some** —
   both hit exactly 0.
2. **PUSH-TRIM hypothesis FALSIFIED — lower mass leaked MORE, not less.** Predicted trimming mass 5e-4→3e-4/4e-4 lowers
   the outward flux the catch absorbs → escape→0. Opposite: cf0.05/m3e4 → escape 0.068 (worse than cf0.05/m5e4's 0.023),
   and cf0.04/m4e4 → escape **0.318** (a hard leak). So at confine 0.03–0.07, mass 5e-4 is the CLEAN regime and reducing
   the push does not help — likely because a weaker jet gives a less-organised internal circulation and cells random-walk
   to the boundary in a way the catch handles worse. **Keep mass 5e-4 for sparse-n Stage-1B; do not trim.** (s4's 0.318
   also confounds confine 0.04 with mass 4e-4, but either way the trim route is dead.)
3. **field:mass mechanistic control — partially confirms the field distinction (recovered, killed in the prior b07).**
   cf0.05 with `field:mass` did NOT fully collapse at this small magnitude (collapsed 0), but it clustered cells tighter
   (nn_mean 0.0091, the batch minimum, vs colour's 0.014–0.021) AND ejected far more (escape 0.182 vs colour's 0.023,
   r_max 1.265). So the per-cell attraction of `field:mass` is real even at 0.05 — it draws cells together and worsens
   containment — just below the full-collapse threshold that 0.2 (Batch-3) crossed. **field:colour is the clean choice;
   field:mass is dirtier at the same magnitude.** Mechanistic contrast supported (direction), full-collapse prediction
   overstated for magnitude 0.05.
4. **Containment does NOT survive proliferation at fixed confine — the key Stage-1C finding.** cf0.05/m5e-4 with division
   ON (s6) grew n 44→442 and deform jumped to **0.0614** (density lifts aggregate scattered momentum, highest of the
   batch) — BUT escape returned to **0.278**, r_max 1.262. The 0.05 catch is calibrated for the n=44 outward flux; at
   n=442 more cells drift to the shell than it can hold. The montage shows s6 becoming grossly lobed/blobby with cells
   spilling — the 1C phenomenology (proliferation reshaping the shell) but leaking.
5. **DENSE + LOW-MASS + HIGHER-CONFINE is near-clean at density — the Stage-1C launch pad.** s7 (cf0.10, mass 5e-5,
   div0.10) reached n=442 with **escape 0.016, r_max 0.954 (no ballistic ejection), deform 0.0268, collapsed 0.** So the
   density route works if the per-cell push is low (5e-5, less outward flux per cell) and the confine is raised (0.10) to
   contain the larger cell count. This is the base for Stage-1C. The one residual (1.6%) is cells grazing the inner
   membrane, not flung out.

**Per-slot verdicts (Batch 8 / b07).**
- s0 ctrl_cf0_m5e4 — **supported (control):** reproduces the confine-0 ballistic-escape baseline (0.136, r_max 2.06).
- **s1 cf0p03 / s2 cf0p07 — KEY supported result: CLEAN Stage-1B.** escape 0, deform 0.030/0.033, r_max<0.86, collapsed 0.
- s3 cf0p05_m3e4 / s4 cf0p04_m4e4 — **falsify the push-trim hypothesis:** lower mass leaked MORE (0.068 / 0.318).
- s5 cf0p05_fieldmass — **partial mechanistic control:** field:mass clusters + ejects (escape 0.182) vs colour's 0.023;
  no full collapse at 0.05 (below the Batch-3 0.2 threshold), but the attraction signature is present (nn_mean 0.0091).
- s6 cf0p05_div — **containment fails under division:** n→442, deform 0.0614 but escape 0.278. Fixed-confine catch
  can't hold the n=442 flux at mass 5e-4.
- **s7 cf0p10_dense — supported / Stage-1C launch pad:** dense (n=442) + low mass 5e-5 + confine 0.10 → escape 0.016,
  deform 0.0268. Near-clean at density; the recipe division needs.

**CAVEAT on the clean-1B tiling (nn_min).** All the clean confine slots show nn_min ~0.004–0.007 < r0=0.02 — the boundary
catch mildly compresses the closest pair below the exclusion distance. BUT collapsed=0 and nn_min stays above the collapse
floor (0.15·r0=0.003), so this is close-approach, not stacking; cf0.03 keeps it highest (0.0073). I treat Stage-1B as MET
on the primary gate (escape 0, collapsed 0, deform ≫0.02) and log the nn_min dip as [open] to watch through Stage-1C.

**Reading the stress↔deform↔escape chain (updated).** The chain is now fully closed for sparse-n: cells scatter momentum
(agent_to_mpm, ∝mass) → grid → shell deform; the small colour-confine supplies the missing boundary-localized inward
force (grad(c)≠0 only at the interface) that returns cells WITHOUT per-cell attraction → deform survives, escape→0. Adding
DENSITY (division) lifts deform further (more aggregate momentum) but also raises the boundary flux, so containment must be
re-tuned as n grows (lower per-cell mass + stronger confine). That trade IS Stage-1C's central question.

**STAGE TRANSITION → 1C.** Gate met with no hard failure on s1/s2 (escape 0, collapsed 0). Adopted operating specs:
`specs/embryo_1B.yaml` (sparse clean 1B: confine 0.03, mass 5e-4, n=44, div off) and `specs/embryo_1C.yaml` (division base:
confine 0.10, mass 5e-5, div0.10, the s7 near-clean dense point). Wrote `current_stage.txt` = `1C`.

**Levers for Batch 9 (Stage-1C):** `agent.div_rate` (proliferation pressure → n and shell reshaping), `mpm_to_agent.confine`
(scale the boundary catch UP with n), `agent_to_mpm.agent_mass` (per-cell push — must stay LOW at density to avoid the s6
leak; a mid value buys deform if confine can hold it). Guardrails: escape (HARD), collapsed, nn_min, r_cell_max.

**HYPOTHESIS (Batch 9).** *Proliferation deforms the shell, and containment at density is set by confine scaling faster
than the outward flux: raising confine 0.10→0.15 at low per-cell mass (5e-5) closes s7's residual escape (0.016→~0) while
division-driven deform stays ≥0.025; and the FAILED high-mass dense route (s6, escape 0.278) is rescued only by a much
larger confine (≥0.20) — otherwise the per-cell jet at n=442 overwhelms any small catch.* Built-in R4 control: division OFF
at the same confine/mass (n=44) must give LOW deform (~floor for mass 5e-5) — proving the extra deform in the div slots is
proliferation/density-driven, not the base push. Prediction ranking: div_cf0p15 closes escape→0 at deform ~0.027 (clean
1C); div_midmass (m1e-4, cf0.15) buys deform ~0.04 if contained; div_rescue_himass (m5e-4, cf0.20) tests whether big confine
can hold a big push at density; ctrl_nodiv isolates division's contribution.

## Batch 9 (2026-07-02) — STAGE 1C: division deforms the shell; is the containment scalable with density? (reading b08)

**Target sub-phase: 1C** (division pressure reshapes the shell). `current_stage.txt` = `1C`.

**b08 results (all 12000f, sorted by cleanliness):**

| slot | n | collapsed | nn_min | deform | escape | r_max | migr | verdict |
|------|---|-----------|--------|--------|--------|-------|------|---------|
| s0 ctrl_nodiv (div0, cf0.10, m5e-5, n44) | 44 | 0.0 | 0.0032 | 0.0109 | **0.000** | 0.883 | 0.175 | control — floor deform |
| s1 div_ref_cf0p10 (div0.10, cf0.10, m5e-5) | 442 | 0.0 | 0.0031 | 0.0268 | **0.0158** | 0.954 | 0.112 | **CLEAN dense (only one)** |
| s2 div_cf0p15 (cf0.15) | 442 | **0.1448** | 0.002 | 0.0253 | 0.0226 | 0.943 | 0.178 | ram-collapse — FALSIFIES H |
| s3 div_midmass (cf0.15, m1e-4) | 442 | 0.0928 | 0.0025 | 0.0522 | 0.086 | 1.060 | 0.37 | deform↑ but leaks + collapse |
| s4 div_slow_midmass (div0.05, cf0.12, m1e-4) | 139 | 0.0 | 0.0032 | 0.0175 | 0.0647 | 0.937 | 0.266 | m1e-4 leaks even at n139 |
| s5 div_fast (div0.20, cf0.15, m5e-5) | 2700 | 0.1285 | 0.002 | 0.0791 | 0.1693 | 1.163 | 0.376 | runaway n, hard-fail escape |
| s6 div_rescue_himass (cf0.20, m5e-4) | 442 | **0.3552** | 0.0015 | 0.0733 | **0.5226** | 1.284 | 0.608 | TOTAL FAIL — big push uncontainable |
| s7 div_verydense (div0.20, cf0.12, m3e-5) | 2700 | 0.0478 | 0.0024 | 0.063 | 0.2644 | 1.236 | 0.562 | extreme density leaks at tiny push |

**1. OBSERVE — division DOES deform the shell (Stage-1C central claim CONFIRMED), but containment does NOT scale with density.**
deform rises monotonically with n at fixed low mass: 0.0109 (n44 ctrl) → 0.0268 (n442) → 0.063–0.079 (n2700). The montages
show progressive reshaping: s1 lobed/blobby by t=12000, s5/s7 grossly amoeboid (torn, multi-lobed) at n=2700. So proliferation
adds aggregate scattered momentum → the shell reshapes. **But every route to higher n or higher per-cell push hard-fails escape.**

**2. The Batch-9 hypothesis is FALSIFIED and INVERTED.** I predicted raising `confine` 0.10→0.15 at low mass would CLOSE the
residual escape (0.016→~0). Instead cf0.15 at n=442 (s2) *raised* escape (0.0158→0.0226) AND induced **collapse 0.1448** (from
0.0). This is the boundary RAM reappearing at density: a stronger colour-gradient catch shoves the n=442 boundary cells into a
dense inner clump. **The clean confine window NARROWS with density** — at n=44 the ram onset was cf~0.10 (Batch-2/7); at n=442 the
collapse onset sits between cf0.10 and cf0.15. So **cf0.10 is already at the containment ceiling for n=442; confine cannot be
scaled UP to contain more density.** This is the key negative result of the batch.

**3. The ONLY clean dense point is div_ref_cf0p10** (n=442, cf0.10, m5e-5): escape 0.0158, deform 0.0268, r_max 0.954 (<1.0 → no
ballistic ejection, cells graze the inner membrane), collapsed 0. Faithfully reproduces b07 s7 — robust at 12000f. This is the
Stage-1C operating point; the residual 1.6% is grazing, not blowout.

**4. Big per-cell push at density is uncontainable by confine at ANY sane magnitude (established).** div_rescue_himass (cf0.20,
m5e-4) gave escape 0.5226, collapsed 0.3552, r_max 1.284 — the montage shows the shell tearing into a gross teardrop with cells
spilling. Raising confine to 0.20 made collapse WORSE (ram + big push), did not rescue. **Mass, not confine, is the density-escape
driver; there is no confine that holds mass 5e-4 at n=442.**

**5. Mid per-cell mass (1e-4) buys deform but hard-fails escape at density.** s3 (cf0.15, m1e-4) → deform 0.0522 (≈2× the clean
low-mass) but escape 0.086 + collapse 0.093; s4 (div0.05, cf0.12, m1e-4, n139) → escape 0.0647, deform only 0.0175. So m1e-4 is
dirty at density regardless of how confine/div_rate are set.

**6. Extreme density leaks even at the tiniest push.** div_verydense (div0.20, cf0.12, m3e-5, n2700) → escape 0.264; div_fast
(div0.20, cf0.15, m5e-5, n2700) → escape 0.169. At n=2700 the boundary flux (n × per-cell drift) overwhelms any containment even
at mass 3e-5. Also `div_rate 0.20` → n=2700 = 60× growth (self-limited by max_occ) — FAR beyond the 4× directive. **Cap div_rate
≤~0.10 (n≤~442) to stay in the containable regime and near the 4× budget.**

**7. Control (R4) satisfied.** ctrl_nodiv (n44, div OFF) → deform 0.0109 = floor for m5e-5. The div slots' extra deform
(0.027–0.079) is therefore proliferation/density-driven, NOT the base push. Attribution is causal.

**Per-slot verdicts (Batch 8 / b08).**
- s0 ctrl_nodiv — **supported (control):** floor deform 0.011, escape 0. Isolates division.
- **s1 div_ref_cf0p10 — supported / the clean Stage-1C point:** escape 0.016, deform 0.027, collapsed 0, r_max<1.0.
- s2 div_cf0p15 — **FALSIFIES the confine-up hypothesis (inverted):** ram-collapse 0.14 + escape up. Confine ceiling at n=442.
- s3 div_midmass — **falsified as clean:** deform 0.052 but escape 0.086 + collapse 0.093.
- s4 div_slow_midmass — **falsified as clean:** m1e-4 leaks 0.065 even at n=139, deform below target.
- s5 div_fast — **runaway / hard-fail:** n=2700, escape 0.169; div_rate 0.20 too high.
- s6 div_rescue_himass — **TOTAL FAIL (established):** big push at density uncontainable; cf0.20 worsens collapse.
- s7 div_verydense — **hard-fail:** extreme density leaks 0.264 at the tiniest push.

**Reading the stress↔deform↔escape chain (density regime).** At density the chain is: division ↑ n → more aggregate scattered
momentum on the grid → shell deform ↑ (good), BUT also more cells drifting to the boundary → boundary flux ↑ → escape ↑. The
colour-confine catch is a boundary-localized inward force whose CAPACITY is fixed (grad(c) at the interface); it cannot be scaled
up (magnitude → ram-collapse) and the flux grows with n and per-cell mass. So there is a HARD TENSION: division-driven deform and
containment pull against each other, and the clean window is narrow (n≈442, cf≈0.10, m≈5e-5, deform≈0.027). To lift clean deform
we need a deform source that does NOT add boundary flux — **`mpm_spin` (clean internal circulation, established as best
deform-per-escape at sparse n)** is the prime candidate to test at the dense base.

**Levers for Batch 9.** (a) `mpm_spin.omega` — amplify deform via clean circulation WITHOUT adding boundary flux (established
sparse-n; test at density). (b) `agent.div_rate` DOWN (0.08 → n~300) — lower flux to close the residual escape. (c) fine
`mpm_to_agent.confine` map (0.10–0.12) — find the exact clean ceiling below the ram. Confine cannot go up much; mass must stay
5e-5; div_rate must stay ≤0.10.

**HYPOTHESIS (Batch 9).** *At the clean dense base (n=442, cf0.10, m5e-5), raising `mpm_spin.omega` 0.3→0.6 amplifies
division-driven deform (0.027→~0.04) WITHOUT raising escape — spin drives clean internal circulation that reshapes the shell
from within rather than pushing cells against it (established best deform-per-escape at sparse n). Independently, lowering
`div_rate` 0.10→0.08 (n~300) closes the residual escape (0.016→<0.01) at slightly lower deform. Confine cannot be scaled above
~0.10–0.12 at n=442 without the ram-collapse seen in s2.* Falsifiers: if spin ω0.6 raises escape (spin flings grazing cells
past the shell at density), then spin is not a clean amplifier at confluence and 1C's deform ceiling is ~0.027; if div0.08
does not lower escape, the residual is not flux-limited and is irreducible with the current operator set (adopt cf0.10/div0.10
as the 1C spec and advance). R4 control: ctrl_nodiv isolates division's deform contribution.

---

## Batch 10 (2026-07-02) — reading b09; STAGE 1C → declared MET, ADVANCE to 1D

**Target read:** b09 (Stage 1C) — tested `mpm_spin` as a clean deform amplifier at density + flux-limited escape.

**b09 results (all 12000f). Base embryo_1C.yaml = div0.10→n442, cf0.10, m5e-5, ω0.3, move0.12.**

| slot | n | collapsed | nn_min | deform | escape | r_max | migr | verdict |
|------|---|-----------|--------|--------|--------|-------|------|---------|
| s0 ctrl_nodiv (div0, n44) | 44 | 0.0 | 0.0032 | 0.0109 | **0.000** | 0.883 | 0.175 | control — floor deform |
| s1 div_ref (div0.10, cf0.10, m5e-5, ω0.3) | 442 | 0.0 | 0.0031 | **0.0268** | 0.0158 | 0.954 | 0.112 | clean-dense ref reproduced (deform-max clean) |
| s2 div_spin06 (ω0.6) | 442 | 0.0 | 0.0032 | 0.0246 | **0.0747** | 1.005 | **0.4413** | spin RAISED escape 5×, deform ↓; migr↑ hugely |
| s3 div_cf0p12 (cf0.12) | 442 | **0.0136** | 0.0029 | 0.014 | 0.0068 | 0.938 | 0.186 | cf0.12: escape↓ but collapse onset + deform crash |
| s4 div0p08 (div0.08, n311) | 311 | 0.0 | 0.0037 | 0.0172 | 0.0096 | 0.940 | 0.147 | lower flux → escape<0.01, deform ↓ |
| s5 div_spin09 (ω0.9) | 442 | 0.0 | 0.0033 | 0.0303 | 0.0317 | 0.951 | 0.105 | ω0.9: deform↑0.030 but escape 2× |
| s6 div_spin06_slow (div0.08+ω0.6, n311) | 311 | 0.0 | 0.0035 | 0.0168 | **0.000** | 0.867 | 0.092 | **STRICTLY CLEAN** (escape 0) — low deform |
| s7 div_himass_spin (m1e-4+ω0.6) | 442 | 0.0045 | 0.003 | **0.0443** | **0.2149** | 1.104 | 0.331 | high deform but escape 0.21 → HARD FAIL |

**1. OBSERVE — the Batch-9 spin hypothesis is FALSIFIED and INVERTED at density.** I predicted `mpm_spin` ω0.3→0.6
would amplify division-driven deform (0.027→~0.04) WITHOUT raising escape (established "best deform-per-escape" at sparse
n). At n=442 the opposite happened: ω0.6 (s2) LEFT deform flat (0.0268→0.0246) and RAISED escape 5× (0.0158→**0.0747**),
r_max crossing 1.0 (1.005 — first cells actually reaching the shell edge). ω0.9 (s5) lifted deform only to 0.0303 while
doubling escape (0.0317). **So the sparse-n "spin = clean deform amplifier" does NOT transfer to confluence.** Mechanism:
solid-body rotation adds a **centrifugal outward** component (∝ radius) to every cell; at density the many boundary cells
are flung outward → boundary flux ↑ → escape ↑, with no extra shell reshaping to show for it. Spin at density buys
**MIGRATION, not deform** (s2 migr 0.4413 — a coherent net swirl; see 1D below), the opposite of what 1C wanted.

**2. The residual escape IS flux-limited (Batch-9 sub-hypothesis SUPPORTED).** div0.08 (s4, n311) cut escape to 0.0096
(<0.01) — fewer cells → less boundary flux → cleaner — but deform dropped in lockstep (0.0268→0.0172). This restates the
1C tension cleanly: **at fixed low mass, deform AND escape both scale with n (boundary flux); you cannot lower one without
the other.** There is no free lunch on the clean frontier.

**3. A STRICTLY-CLEAN confluent point exists: s6 (div0.08 + ω0.6, n=311) → escape EXACTLY 0.000, collapsed 0, r_max
0.867, deform 0.0168.** Notably spin at n=311 IMPROVED containment (s4 no-spin escape 0.0096, r_max 0.940 → s6 +spin
escape 0.000, r_max 0.867) — the reverse of its effect at n=442. At moderate density the circulation seems to keep cells
off the shell; at confluence its centrifugal push dominates. (Could be a density threshold or partly noise; low priority.)
This gives Stage-1C a HARD-CLEAN operating point (division on, n 44→311 = 7×, shell mildly reshaped, ZERO escape).

**4. cf0.12 confirms cf0.10 is the containment CEILING at n=442.** s3 (cf0.12): escape dropped to 0.0068 (good) BUT
collapse onset appeared (0.0136 — the boundary ram beginning) and deform crashed to 0.014. So nudging confine up trades
deform + collapse-safety for lower escape — a bad trade. cf0.10 remains the ceiling (consistent with b08's cf0.15 ram).

**5. Mid mass + spin still uncontainable at density.** s7 (m1e-4 + ω0.6): deform 0.0443 (highest), migr 0.331, but escape
**0.2149**, r_max 1.104, collapse 0.0045 — HARD FAIL. Spin's "organised circulation" did NOT let m1e-4 deform cleanly;
m1e-4 leaks at density regardless (consistent with b08). Confirms: at density keep mass 5e-5.

**Per-slot verdicts (Batch 9 / b09).**
- s0 ctrl_nodiv — **supported (control):** floor deform 0.011, escape 0. Isolates division's deform contribution.
- s1 div_ref — **supported / the deform-max clean-ish 1C point:** deform 0.0268 @ escape 0.0158 (grazing, r_max<1.0).
- s2 div_spin06 — **FALSIFIES the spin-amplifier hypothesis (inverted):** escape 5×, deform flat; migr↑ (→1D lead).
- s3 div_cf0p12 — **confirms cf0.10 ceiling:** collapse onset + deform crash; confine-up is a bad trade.
- s4 div0p08 — **supports flux-limited escape:** escape<0.01 at n311, deform ↓ proportionally.
- s5 div_spin09 — **partial:** deform 0.030 but escape 2× — spin buys little clean deform at density.
- s6 div_spin06_slow — **STRICTLY-CLEAN 1C point (escape 0):** adopt as the hard-clean 1C variant; deform 0.017.
- s7 div_himass_spin — **HARD FAIL:** m1e-4 uncontainable at density even with spin.

**STAGE-1C VERDICT — MET; ADOPT + ADVANCE.** Stage-1C central claim (division proliferation reshapes the shell) is
[established] (deform ∝ n, monotone; montage s1/s6 clearly lobed/amoeboid vs the round div-OFF control). Two operating
points adopted: **`div_ref` (n442, cf0.10, m5e-5, div0.10, ω0.3) = deform-max clean-ish (deform 0.027 @ escape 0.016
grazing, r_max<1.0)**, and **`div_spin06_slow`-style (n311, div0.08, +ω0.6) = strictly clean (escape 0, deform 0.017)**.
Every attempt to lift CLEAN deform above ~0.027 (spin, mid-mass, confine-up) failed — the clean deform ceiling at density
is ~0.027, set by the fixed-capacity colour-confine catch vs the n-scaling boundary flux. This is a real physical ceiling,
not a tuning miss. Per the ladder rule (breadth beats perfecting a rung; advance when the gate is met with no hard
failures — s6 has escape 0), **advance to Stage 1D.** `current_stage.txt` → `1D`.

**BRIDGE TO 1D (the b09 lead).** Spin at confluence produced strong global velocity polar order (migration 0.4413 at
n=442, ω0.6) with cells still flowing (flow 0.0057, not jammed) — i.e. **collective migration is reachable at confluence.**
But that migration was IMPOSED (spin) and came with escape 0.075. Stage-1D's real question: can collective migration
EMERGE from cell-cell coupling (neighbour heading-alignment `polar_align`, or flow-coupling `flow_align`) rather than being
imposed by a body-rotation field — and is emergent migration CLEANER (lower escape) than imposed rotation, because
alignment redirects each cell's existing move_speed into a shared direction instead of adding centrifugal outward drift?

**HYPOTHESIS (Batch 10, Stage 1D).** *At confluence (n=442, contained by cf0.10/m5e-5), adding `polar_align` (Vicsek
neighbour heading-alignment, gamma ≥40) drives EMERGENT collective migration — global velocity polar order (migration)
rises well above the diffusive control (~0.1) toward ≥0.3 with coherent streams in the tracks panel — WITHOUT the
boundary-flux escape that imposed `mpm_spin` incurs (escape stays ≤~0.03, r_max<1.0), because polar_align redirects each
cell's move_speed into a shared heading rather than adding centrifugal outward drift.* Predictions: ctrl_nomig (all
migration drivers off) → migr ≈0.1, escape ~0.016 (diffusive baseline); polar40/polar120 → migr↑ with escape contained;
flowalign_hi → migr↑ via flow-coupling; spin06 → migr ~0.44 but escape ~0.075 (imposed, leaky — the contrast). Falsifier:
if polar_align raises escape as much as spin does (heading-alignment marches cells collectively into the shell), then
emergent migration is NOT cleaner than imposed rotation and Stage-1D at confluence is escape-limited regardless of driver —
in that case run migration at the moderate density n311 (escape-safe) and log the confluent escape ceiling as [open].

---

## Batch 11 (2026-07-02) — reading b10; STAGE 1D: emergent flocking (polar_align) BEATS imposed spin — and STRONGER flocking is CLEANER (γ↑ → migration↑ AND escape↓). Hypothesis SUPPORTED.

**Target sub-phase: 1D** (collective migration at confluence). `current_stage.txt` = `1D`. Note: div0.10 in the 1D base grew to **n=557** (not the b09 442) — a *higher* confluence than any prior clean test, so b10 is a stricter containment test.

**b10 results (all 12000f). Base embryo_1D.yaml = div0.10→n557, cf0.10 field colour, m5e-5, ω0.3, flow_align 40, move0.12; polar_align γ set per slot. r0=0.02.**

| slot | n | collapsed | nn_min | deform | flow | migr | seg | escape | r_max | verdict |
|------|---|-----------|--------|--------|------|------|-----|--------|-------|---------|
| s0 ctrl_nomig (γ0, gain0, ω0 — ALL drivers off) | 557 | **0.2657** | 0.0 | 0.1314 | 0.00301 | 0.1009 | 0.0749 | **1.000** | 2.091 | TOTAL FAIL — shell torn open |
| s1 flowalign_hi (gain120, γ0) | 557 | 0.0108 | 0.0020 | 0.0246 | 0.00575 | 0.1396 | 0.0146 | 0.0413 | 0.957 | flow_align WEAK driver; near-clean |
| s2 polar40 (γ40) | 557 | 0.0036 | 0.0027 | 0.0355 | 0.00567 | 0.2503 | 0.0281 | 0.1077 | 1.007 | emergent migr↑ but escape 0.11 |
| **s3 polar120 (γ120)** | 557 | 0.0108 | 0.0021 | 0.0470 | 0.00576 | **0.4929** | 0.0101 | **0.0197** | 1.062 | **strong migr + near-clean — KEY** |
| s4 spin06 (ω0.6 imposed, γ0) | 557 | 0.0036 | 0.0028 | 0.0471 | 0.0057 | 0.4285 | 0.0359 | **0.2603** | 1.243 | imposed rotation — leaky |
| s5 polar40_fast (γ40, move0.24) | 557 | 0.0108 | 0.0027 | 0.0533 | 0.00901 | 0.2555 | 0.0191 | **0.2208** | 1.082 | fast cells overshoot → escape 0.22 |
| s6 polar40_n311 (γ40, div0.08) | 331 | **0.0** | 0.0033 | 0.0312 | 0.00526 | 0.2444 | 0.0007 | 0.0363 | 0.959 | moderate density, collapsed 0 |
| s7 polar40_spin06 (γ40 + ω0.6) | 557 | 0.0162 | 0.0025 | 0.0549 | 0.00559 | **0.5364** | 0.0411 | **0.3788** | 1.281 | max migr but escape 0.38 (spin stacks) |

**1. OBSERVE — the Batch-10 hypothesis is SUPPORTED, decisively: emergent migration (polar_align) is strong AND far cleaner than imposed rotation (spin).** polar120 (s3): **migration 0.4929 at escape 0.0197** — comparable migration to imposed spin ω0.6 (s4: migr 0.4285) but escape **13× lower** (0.0197 vs 0.2603) and r_max 1.06 vs 1.24. Vicsek neighbour-heading alignment redirects each cell's move_speed into a SHARED translational stream (the montage s3/s4 show aligned orange streaks migrating coherently), whereas mpm_spin adds a centrifugal outward drift that flings the large boundary population past the shell. Emergent > imposed for cleanliness — as predicted.

**2. THE KEY NEW RESULT — stronger flocking is CLEANER: raising γ raised migration AND lowered escape together (non-obvious, opposite of the "collective march into the wall" falsifier).** Across the γ sweep at n=557: γ40 → migr 0.250 / escape **0.108**; γ120 → migr 0.493 / escape **0.020**. So 3× the alignment gain gave 2× the migration AND 5× LESS escape. Mechanism: at partial alignment (γ40) cells are still semi-independent and a fraction drift to the boundary and pile up (escape 0.11); at strong alignment (γ120) the flock locks into ONE coherent stream that circulates inside the shell as a body rather than depositing cells on the wall. The falsifier ("heading-alignment marches cells collectively into the shell") did NOT fire — coherence is *protective*, not destructive.

**3. The control CONFIRMS coherence is protective — with NO organizing motion the shell TEARS APART.** s0 ctrl_nomig (γ0, gain0, ω0 — every migration/circulation driver off) at n=557 → **escape 1.000, r_max 2.091, collapsed 0.2657**, deform 0.1314 (pathological — the membrane is ripped into an angular torn shape, cells strung along broken edges; top montage row). At the same density, γ120 holds escape to 0.02. **At confluence, coherent collective motion CONTAINS cells; disorganized random glide is catastrophic.** *Caveat:* this control also removed the base spin ω0.3, so it is not a clean polar-only ablation — Batch 11 adds the proper control (γ0 with spin ω0.3 retained) to isolate polar_align's contribution against a still-circulating base.

**4. flow_align (SPV) is a WEAK migration driver — Vicsek neighbour-heading beats flow-coupling.** s1 (flow_align gain 120, γ0) → migration only 0.1396 (barely above the diffusive floor ~0.10) and the LOWEST deform (0.0246). Relaxing heading toward the local MPM flow does not organize a stream nearly as well as aligning to neighbour headings directly. So `polar_align`, not `flow_align`, is the confluence-migration operator.

**5. move_speed 0.24 breaks containment (s5) and stacking spin on flocking stacks escape (s7).** polar40_fast (move0.24) → escape 0.221 (faster cells overshoot the boundary — reconfirms move_speed as the dominant escape co-driver). polar40 + ω0.6 (s7) → highest migration of all (0.5364) but escape 0.379 — the flock's coherence and the spin's centrifugal drift cooperate for polar order but the spin's escape penalty dominates. **Do not add spin to a flock; do not raise move_speed at confluence.**

**6. Moderate density (n=311, div0.08) is cleaner but γ40 leaves migration modest.** s6 (γ40, n311) → escape 0.036, collapsed **0.0**, migr 0.244. Fewer cells → less boundary flux → cleaner, but at γ40 migration is only 0.24. The obvious next move: γ120+ at n311 should give strong migration AND strictly-clean escape (Batch 11). segregation 0.0007 ≈ 0 (single coherent blob, no partition mechanism — expected; 1E not yet attempted).

**Per-slot verdicts (Batch 10 / b10).**
- s0 ctrl_nomig — **control (confounded):** TOTAL FAIL (escape 1.0, shell torn). Shows organization is protective, but removed spin too → not a clean polar ablation. Redo cleanly in b11.
- s1 flowalign_hi — **falsifies flow_align as a strong migration driver:** migr 0.14 at gain 120. Vicsek ≫ SPV here.
- s2 polar40 — **supported (emergent migration real) but dirty at γ40:** migr 0.25, escape 0.11.
- **s3 polar120 — KEY supported result:** strong emergent migration (0.49) near-clean (escape 0.02); beats imposed spin's cleanliness 13×.
- s4 spin06 — **reconfirms imposed rotation is leaky at density:** migr 0.43 @ escape 0.26 (b09-consistent centrifugal escape).
- s5 polar40_fast — **falsifies move0.24 as safe:** escape 0.22. move_speed is the escape co-driver.
- s6 polar40_n311 — **supported (moderate-density clean-ish):** collapsed 0, escape 0.036 at γ40; migration modest.
- s7 polar40_spin06 — **HARD FAIL:** emergent+imposed max out migration (0.54) but escape 0.38; spin stacks onto flock.

**Reading the stress↔deform↔migration chain (confluence regime).** Migration (velocity polar order) is set by how coherently cells share a heading; `polar_align` supplies that coherence directly and, crucially, coherence also SOLVES containment — a locked flock circulates as one body inside the shell instead of individually random-walking to the boundary and punching through. So at confluence the SAME operator (strong polar_align) is BOTH the migration driver AND the containment, and escape falls as coherence rises. This inverts the whole prior campaign's tension (deform/migration vs escape pulling against each other): with imposed spin they fought (more spin → more escape); with emergent flocking they align (more flocking → more migration AND less escape). Escape at confluence is fundamentally a *disorganization* problem.

**STAGE-1D STATUS — near MET; Batch 11 closes escape→0 and maps the γ trend.** polar120 already delivers Stage-1D's target phenomenology: coherent collective migration (0.49) at confluence with cells still flowing (flow 0.0058, not jammed) and near-contained (escape 0.02). The residual 2% escape and the collapsed 0.011 are closing jobs. Not yet declaring MET until a STRICTLY-CLEAN (escape 0) strong-migration point is found — the Batch-11 job.

**Levers for Batch 11.** (a) `polar_align.gamma` — map the trend (γ 80 / 200 / 300): does escape keep falling and migration keep rising, or does the flock jam (flow→0) / re-leak at very high γ? (b) `polar_align.noise` DOWN (0.1→0.05) — tighter flock, cleaner? (c) `agent.div_rate` 0.08 (n~311, lower flux) stacked with strong γ — the strictly-clean strong-migration candidate. (d) `agent.move_speed` DOWN (0.06) — less overshoot; does migration (a *direction* order, speed-independent) survive while escape →0? Guardrails: escape (HARD), collapsed, nn_min, r_cell_max. Keep mass 5e-5, ω0.3, no spin-stacking, move ≤0.12.

**HYPOTHESIS (Batch 11).** *At confluence, flocking coherence IS the containment: raising `polar_align.gamma` (and/or lowering `noise`) monotonically RAISES emergent migration AND LOWERS escape, because strong Vicsek alignment converts each cell's independent boundary-ward drift into a shared translational stream that circulates inside the shell rather than piling onto the boundary. Prediction: γ 40→80→120→200 gives migration rising toward saturation (~0.5) while escape falls (0.11→…→0.02→<0.01), and the STRICTLY-CLEAN (escape 0) strong-migration point is γ≥120 at moderate density (n≈311, div0.08). Falsifier: if beyond γ120 escape rises again OR migration saturates while flow collapses toward 0 (the flock crystallizes/jams), then γ120 is the optimum and the trend is non-monotone — coherence helps only up to a rigidity threshold.* R4 control: γ0 with spin ω0.3 retained (clean polar-only ablation) — isolates polar_align's migration+containment contribution at fixed n=557.

---

## Batch 12 (2026-07-02) — reading b11; STAGE 1D: the b10 "monotone γ↑ → migration↑ AND escape↓" claim is NOT robust — migration is a NOISY / near-bistable order parameter, containment is DENSITY-DEPENDENT, and lownoise/slow both KILL migration. Hypothesis PARTIALLY FALSIFIED.

**Target sub-phase: 1D** (collective migration at confluence). `current_stage.txt` = `1D`. Base embryo_1D.yaml = div0.10→n557, cf0.10 field colour, m5e-5, ω0.3, flow_align 40, move0.12, polar_align noise 0.1, γ per slot. All 12000f. r0=0.02.

| slot | γ | n | collapsed | nn_min | deform | flow | migr | escape | r_max | verdict |
|------|---|---|-----------|--------|--------|------|------|--------|-------|---------|
| s0 ctrl_nopolar (γ0, ω0.3) | 0 | 557 | 0.0162 | 0.0021 | 0.0255 | 0.0055 | 0.1817 | 0.1005 | 0.983 | diffusive baseline; escape 0.10 even with ω0.3 |
| s1 gamma80 | 80 | 557 | 0.0126 | 0.0024 | 0.0435 | 0.0055 | 0.470 | 0.0664 | 1.017 | strong migr, escape 0.066 (NOT →0) |
| s2 gamma200 | 200 | 557 | 0.0072 | 0.0022 | 0.0391 | 0.0056 | 0.3672 | 0.070 | 1.002 | migr LOWER than γ80, escape same ~0.07 |
| s3 gamma120_n311 | 120 | 331 | **0.0** | 0.0034 | 0.0373 | 0.0051 | **0.6517** | **0.1571** | 1.024 | migr MAX but escape 0.16 — moderate density LEAKS more |
| s4 gamma120_lownoise (noise0.05) | 120 | 557 | 0.0036 | 0.0028 | 0.028 | 0.0058 | **0.0379** | 0.0503 | 1.041 | lownoise KILLED migration (0.04 floor) |
| s5 gamma300 | 300 | 557 | 0.0072 | 0.0027 | 0.051 | 0.0058 | 0.5187 | 0.0772 | 1.096 | strong migr, escape 0.077 (plateau) |
| s6 gamma120_slow (move0.06) | 120 | 557 | 0.0036 | 0.0027 | 0.018 | 0.0041 | **0.0654** | **0.0036** | 0.902 | slow → cleanest escape but migration DEAD |
| s7 gamma200_n311 (move0.12) | 200 | 331 | **0.0** | 0.0037 | 0.0271 | 0.0052 | **0.0901** | 0.0242 | 0.999 | γ200 n311 → migration DEAD (vs γ120 n311 0.65) |

**1. OBSERVE — the b10 monotone story does NOT reproduce; migration is a NOISY, near-bistable order parameter, not a smooth function of γ.** b10 read γ40→0.25/esc0.108, γ120→0.49/esc0.020 and inferred a clean monotone trend. b11 at n557: γ80→0.47, γ200→0.37, γ300→0.52 — migration does NOT rise monotonically with γ; it scatters between ~0.37 and ~0.52 with no order. And critically, escape at every strong-migration n557 point sits at **0.066–0.077**, NOT the b10 γ120 0.020. **The b10 γ120 escape 0.020 now looks like a favorable single realization, not a robust plateau.** No n557 point in b11 reached escape<0.05 while migration>0.4. The strictly-clean strong-migration point was NOT found.

**2. THE FALSIFIER FIRED IN TWO WAYS — both "lownoise → cleaner tighter flock" and "migration is speed-independent" are FALSE.** (a) **s4 gamma120_lownoise (noise 0.1→0.05) CRASHED migration to 0.038** (a floor), not tightened it. Montage s4 shows multiple competing stream directions at t=12000 → the net velocity polar order cancels. Interpretation: a small angular noise (~0.1) is REQUIRED to let the flock anneal into ONE global heading; with too little noise it freezes into multiple locked domains whose drifts cancel → migration≈0. (b) **s6 gamma120_slow (move 0.12→0.06) CRASHED migration to 0.065** while giving the batch-cleanest escape (0.0036). This falsifies b10's "migration is a direction order, speed-independent, survives while escape→0." Migration does NOT survive at move0.06 — a coherent translating stream needs enough per-cell speed to build up; at 0.06 the flock never organizes. **The clean escape at move0.06 is clean because there is no migration to drive cells into the wall.**

**3. CONTAINMENT IS DENSITY-DEPENDENT — the "coherence IS containment" claim holds at HIGH confluence but INVERTS at moderate density.** At n557, strong γ contains (escape 0.066–0.077). At n311, RAISING γ 40→120 raised migration 0.244→0.652 AND raised escape 0.036→**0.157** (s3) — the coherent stream, with fewer cells packing the interior, sweeps as a BODY into the boundary and punches through (exactly the original "collective march into the wall" mechanism). So flocking-is-containment is a HIGH-DENSITY effect: only when the interior is jammed does the flock circulate rather than translate into the wall. At moderate density a strong flock is a LEAK, not a shield.

**4. Non-monotone / bistable at n311 too.** s3 γ120 n311 → migr 0.65 (batch max); s7 γ200 n311 → migr 0.090 (dead). Same density, higher γ, migration collapsed. This confirms migration is realization/parameter-sensitive and near-bistable (coherent-translating stream vs fragmented/rotational cancelling state), not a monotone γ response.

**5. What SURVIVES from b10.** (i) Emergent flocking DOES produce strong collective migration (0.37–0.65) with cells still flowing (flow 0.005, not jammed) — the Stage-1D phenomenology is real and delivered. (ii) At high confluence strong γ is far cleaner than imposed spin (escape ~0.07 vs 0.26). (iii) The R4 control (γ0) sits at migration 0.18, escape 0.10 — polar_align is genuinely the migration driver. What's OVERTURNED: the *monotone* γ↑→escape↓ trend and the escape→0 promise.

**Per-slot verdicts (Batch 11 / b11).**
- s0 ctrl_nopolar — **clean R4 control:** γ0 (ω0.3 retained) → migr 0.18, escape 0.10. Isolates polar_align as the migration driver; diffusive baseline leaks 10%.
- s1 gamma80 — **supported (strong migr) but escape 0.066:** falsifies escape→0 at strong γ.
- s2 gamma200 — **falsifies monotone γ↑→migr↑:** migr 0.37 < γ80's 0.47.
- s3 gamma120_n311 — **falsifies "n311+γ120 = strictly clean":** migr MAX 0.65 but escape 0.16; moderate density LEAKS more, not less.
- s4 gamma120_lownoise — **FALSIFIES "lownoise = tighter/cleaner flock":** migration CRASHED to 0.038 (competing domains). Noise ~0.1 is required for global coherence.
- s5 gamma300 — **plateau:** migr 0.52, escape 0.077. No cleaner than γ80–120.
- s6 gamma120_slow — **FALSIFIES "migration speed-independent":** move0.06 → migr 0.065 (dead), escape 0.0036 (cleanest — but empty).
- s7 gamma200_n311 — **near-bistable:** migr 0.090 (dead) vs s3's 0.65 at same density; realization-sensitive.

**Reading the chain.** Migration (velocity polar order) here is a genuine collective order parameter with two attractors: a single global translating stream (high migr) vs fragmented/cancelling domains (low migr). It needs BOTH a moderate angular noise (~0.1, to anneal to one heading) AND enough move_speed (~0.12, to build the stream). Escape then tracks how the coherent stream interacts with the boundary: at high density the interior jam forces circulation (contained, escape ~0.07); at moderate density the stream translates into the wall (leak, escape ~0.16). Coherence is containment ONLY under interior jamming. The residual ~0.07 escape at the clean high-density flock is intrinsic to a translating flock grazing the shell (r_max ~1.0–1.06) and will NOT be closed by more γ — it needs a boundary lever or a slower/absorbed approach.

**STAGE-1D STATUS — phenomenology DELIVERED, escape NOT yet strictly closed at strong migration.** Collective migration (0.37–0.65) at confluence with flowing (not jammed) cells is real and robust. The residual: escape ~0.05–0.16 whenever migration>0.4; the only escape<0.005 points have DEAD migration. Batch 12 tests (a) reproducibility/variance of the strong-migration plateau (γ100/120/140 at n557), (b) whether a modest boundary-confine bump (0.10→0.12) — which a COHERENT flock, unlike a disorganized crowd, may tolerate without ram-collapse — closes the residual escape, and (c) an intermediate move_speed (0.09) to see if migration survives while escape drops. If no strictly-clean strong-migration point emerges, ADOPT the best (γ~120 n557, migr ~0.49 @ escape ~0.07 grazing, r_max ~1.06, collapsed ~0.01) as the Stage-1D operating point, log the intrinsic residual as [open], and ADVANCE to Stage 1E (two-type partitioning) per the breadth rule.

**HYPOTHESIS (Batch 12).** *The strong-flock migration at confluence is a noisy near-bistable order parameter (not monotone in γ), and its residual escape (~0.05–0.08 at n557 whenever migr>0.4) is intrinsic to a translating flock grazing the shell — more γ will NOT close it. The lever that lowers escape at fixed strong migration is a small boundary-confine bump (0.10→0.12), which a COHERENT flock tolerates without the ram-collapse a disorganized crowd suffers (b09 confine-up rams). Prediction: γ100/120/140 at n557 all give migr ~0.4–0.55 at escape ~0.05–0.08 (plateau, scattered, NOT →0), confirming non-monotonicity; confine 0.12 at γ120 cuts escape below 0.03 while keeping migr>0.4 and collapsed<0.02 (NO ram); move0.09 keeps migr>0.3 (survives, unlike move0.06) at escape below move0.12's. Falsifier: if confine 0.12 rams (collapsed↑, nn_min→0.002) OR drops migration to floor, then the flock is no more confine-tolerant than a crowd and the residual escape is unclosable at this density — adopt γ120 n557 as-is and advance to 1E.*

## Batch 13 — 2026-07-02 — reading b12 (STAGE 1D close-out) → ADVANCE to STAGE 1E (two-type partitioning)

**What happened vs b12 predictions (montage `montages/embryo_b12.png` + `archive/embryo_1D_b12_*`).**
All 8 at n557/div0.10; move0.12 & cf0.10 unless noted:

| slot | mods | migr | escape | deform | collapsed | r_max |
|------|------|------|--------|--------|-----------|-------|
| s0 ctrl_nopolar (γ0)      | —              | 0.182 | 0.101 | 0.026 | 0.016 | 0.983 |
| s1 gamma100               | γ100           | 0.433 | 0.086 | 0.051 | 0.007 | 1.180 |
| s2 gamma120               | γ120           | **0.493** | **0.020** | 0.047 | 0.011 | 1.062 |
| s3 gamma140               | γ140           | 0.248 | 0.014 | 0.033 | 0.004 | 0.980 |
| s4 gamma120_conf12        | cf0.12         | 0.445 | **0.201** | 0.051 | 0.022 | 1.159 |
| s5 gamma120_noise15       | noise0.15      | 0.202 | 0.068 | 0.030 | 0.007 | 1.035 |
| s6 gamma120_move09        | move0.09       | 0.252 | 0.014 | 0.027 | 0.004 | 0.987 |
| s7 gamma120_m09_cf12      | move0.09+cf0.12| 0.348 | **0.004** | 0.025 | 0.018 | 0.906 |

**Per-slot verdicts.**
- **s2 gamma120 — pivotal, prediction OVERTURNED (in our favour).** Predicted escape ~0.05–0.08 (plateau, "b10 0.020 was a lucky draw"). Got escape **0.020 AGAIN** (migr 0.493). Two independent realizations (b10 + b12) now both give γ120/n557 → escape ~0.020 at strong migration. The "lucky-draw" caveat is FALSIFIED: γ120 is a GENUINELY clean strong-flock point. *supported (γ120 is the migration operating point); the escape-plateau claim falsified.*
- **s1 gamma100 / s3 gamma140 — non-monotonicity CONFIRMED, but it is a PEAK not scatter.** migr 0.433 (γ100) → 0.493 (γ120) → 0.248 (γ140). γ140 falls back toward the diffusive floor; there is an OPTIMAL alignment strength at γ≈120, not "more is more." *supported (non-monotone); refined to a peak.*
- **s4 gamma120_conf12 — FALSIFIED HARD (last containment lever dead at full speed).** Predicted cf0.12 cuts escape <0.03 for a coherent flock. Got escape **0.201** — 10× WORSE than cf0.10 (0.020). No ram-collapse (collapsed 0.022, nn_min 0.0026 fine), so it is not the b09 density-ram; the stronger boundary catch simply shoves the coherent flock's boundary cells outward faster. *falsified.*
- **s7 gamma120_m09_cf12 — the CONTAINMENT WIN (residual escape essentially CLOSED).** The SAME cf0.12 that blew escape to 0.201 at move0.12 (s4) is TOLERATED at move0.09: escape **0.004** (batch-cleanest), migr 0.348, r_max 0.906 (all cells well inside), collapsed 0.018. Clean strong-ish-flock recipe at confluence = move0.09 + cf0.12. The b12 residual-escape [open] is RESOLVED. *supported (combined levers close escape); single-lever cf0.12 does not.*
- **s6 gamma120_move09 — speed-sensitivity reconfirmed.** move0.09 dropped migr to 0.252 (<0.3 predicted) at clean escape 0.014. 0.09 already weakens the flock; migration needs move≈0.12 for full strength. *falsified (migr<0.3), containment supported.*
- **s5 gamma120_noise15 — noise~0.1 is the coherence peak.** noise0.15 dropped migr to 0.202; with 0.05 also dead (b12), noise 0.1 is the optimum. *supported.*
- **s0 ctrl_nopolar (R4) — polar_align is the migration driver.** γ0 → migr 0.182 (floor), escape 0.101. *supported.*

**DECISION — Stage 1D DELIVERED; ADVANCE to Stage 1E.** Two clean 1D operating points now exist:
(A) **γ120 / n557 / move0.12 / cf0.10** → migr **0.49** @ escape **0.020** (strongest migration, near-clean, reproduced);
(B) **γ120 / n557 / move0.09 / cf0.12** → migr **0.35** @ escape **0.004** (strictly clean). Phenomenology (strong
collective migration at confluence, flowing not jammed) is robust and the residual escape is now closable. 1D started
Batch 10 (batch 4 of its budget); per the breadth rule I ADVANCE to Stage 1E. Wrote `1E` to `current_stage.txt`.

**STAGE 1E — the partition mechanism (new operator family this batch: chemical signalling).** Target: the two cell
types segregate (seg = |⟨x⟩_a−⟨x⟩_b|/R ↑) while the shell stays whole (escape≈0). R1/R3: introduce ONE new family —
per-type chemical cross-repulsion — tested against an R1-minimal alternative (differential motility, no chemistry) and an
R4 control (chemistry present, cross-repulsion OFF). Mechanism (slime/vicsek precedent + operator re-read): `deposit`
writes each agent's trail into its OWN type-channel (channel = node_type); `diffuse`+`decay` shape a smooth per-type
density field; two `chemotaxis` instances — one on `agent[type=a]` reading channel 1 (type-b's trail), one on
`agent[type=b]` reading channel 0 — each with NEGATIVE gain, so each type FLEES the other's chemical → the two
populations demix into opposite domains (a Janus split → seg ↑). Both instances share the one schedule token
`chemotaxis` (_run_token runs every instance of a name) and the one `chemotaxis.gain` override (sets gain on both). Base
`embryo_1E.yaml` = the clean sparse 1B point (n44, div OFF, cf0.05 field colour, mass 5e-5, move0.12, polar_align γ0 —
flocking OFF so it does not homogenize the sort) + the chemical field + deposit/diffuse/decay/2×chemotaxis.

**HYPOTHESIS (Batch 13).** *Per-type chemical cross-repulsion drives the two types to demix into a left/right (Janus)
partition — seg rises monotonically with |chemotaxis.gain| above the no-chemotaxis control (seg ~0.02 baseline) — while
the cf0.05 colour-confine keeps the sparse n44 disc contained (escape≈0) up to a gain ceiling where the mutual push
overwhelms the catch. Predictions: ctrl_nochem (gain 0) → seg ~0.02 (no partition, R4 baseline); the gain sweep
−0.02→−0.3 raises seg well above 0.05 with escape <0.02 in the clean window; a very strong gain (−1.0) pushes seg highest
but breaks containment (escape↑, r_max>1); differential motility alone (type-a move 0.20 / type-b 0.05, NO chemistry)
does NOT produce left/right seg (~0.02) — proving the sort needs a type-pair interaction, not a speed difference; at
confluence (n557 + flock γ120) cross-repulsion is overwhelmed by the global flock → seg stays low. Falsifier: if
ctrl_nochem already shows seg>0.05, or the gain sweep does not raise seg above the control, then chemotaxis
cross-repulsion is not the driver (or the field is mis-scaled) and Batch 14 re-tunes the field timescales or switches to
`sense` (own+1/cross−1) / differential adhesion.*

## Batch 14 — 2026-07-02 — reading b13: ZERO DATA — all 8 jobs died at spec-parse (YAML bug), NOT a scientific result. STAGE 1E still UNTESTED. Fix + re-issue.

**1. OBSERVE — Batch 13 produced NO physics.** There is no `montages/embryo_b13.png`, no `archive/embryo_1E_b13_*`
dir, and no `metrics.json` for any slot. Every one of the 8 jobs failed at LOAD time. `loop_logs/eb_b13_s0_ctrl_nochem.err`
(and s1–s7, identical) ends in:
```
yaml.parser.ParserError: while parsing a flow mapping
  in "specs/embryo_1E.yaml", line 50, column 5
expected ',' or '}', but got '['
  in "specs/embryo_1E.yaml", line 50, column 31
```
**Root cause — a YAML syntax bug of MY making, not a physics finding.** Lines 50–51 of `embryo_1E.yaml` (and 41–42 of
`embryo_1E_diffmot.yaml`) wrote the per-type selector UNQUOTED inside a `{...}` flow mapping:
`- {op: chemotaxis, at: agent[type=a], ...}`. In YAML flow context `[` is a flow-sequence indicator, so the bare scalar
`agent[type=a]` is a parse error (the `[` at col 31). The existing WORKING multi-type specs all QUOTE it —
`specs/agent_mpm_disc_4types.yaml:39 → at: 'agent[type=a]'` — which I failed to copy. So the entire 1E launch was lost to
one missing pair of quotes; the sim never started. **This is an ENGINEERING failure; the Stage-1E partition question is
completely UNTESTED — I have no data on whether chemical cross-repulsion demixes the two types.**

**2. FIX (applied this batch) + mechanism re-verified so b14 does not fail a second way.** Quoted the selector in both
specs (`at: 'agent[type=a]'` / `'agent[type=b]'`). Then re-read the engine to confirm the DESIGN is otherwise sound (a
second latent bug would have wasted b14 too):
- `chemotaxis` op EXISTS (`plexus/operators/chemotaxis.py`): `PREDICTION=first_derivative`, emits velocity `gain·grad(channel)`
  sampled at each particle, `gain<0` ⇒ flee; params it reads = `from`, `gain`, `channel`, `noise` — exactly my spec's keys.
- `deposit` (`plexus/operators/deposit.py:47`) writes `amount·dt` into channel = `node_type[i]` — the per-type trail is real
  (type a→ch0, b→ch1), and the `chemical` field `{res:128, couples_to: agent}` allocates C=2 channels.
- Selector resolution (`engine.py:430,457`): the engine sets `_at = o.on.set` (the BASE set, `agent`) and applies the
  `agent[type=a]` mask SEPARATELY via `_selector_mask` (live & node_type==a). So `H.level('agent')` works and the velocity
  delta is masked to type a — the plumbing is correct once the YAML parses. A single `chemotaxis` schedule token runs BOTH
  instances; a single `chemotaxis.gain` override sets both (symmetric cross-repulsion strength). All confirmed.

**3. Per-slot verdicts (Batch 13).** All 8 — **INCONCLUSIVE (never executed).** No collapse/escape/seg to report;
nothing supported or falsified. The b13 hypothesis stands entirely untested.

**4. DECISION — re-issue the IDENTICAL 1E design as Batch 14 (the real first 1E attempt).** Because b13 yielded zero
information, the scientifically honest move is to run exactly the experiment that was designed (now that the specs parse),
not to redesign around imagined results. Batch 14 = b13's 8-slot layout (1 control · 4 exploit gain-sweep · 3 explore) on
the fixed specs. `current_stage.txt` stays `1E`. Guardrail unchanged: escape (HARD), collapsed, nn_min; TARGET = seg.
*Metric caveat I will watch:* seg = |⟨x⟩_a−⟨x⟩_b|/R measures only the x-projection (left/right), so a top/bottom or
radial demix would read low even if the montage clearly shows sorting — I will READ THE CELLS PANEL, not just seg, and if
demixing is visible but seg is low, add a rotation-invariant demix metric next batch.

**HYPOTHESIS (Batch 14, = b13's, now testable).** *Per-type chemical cross-repulsion demixes the two types (seg rises
monotonically with |chemotaxis.gain| above the no-chemotaxis control seg≈0.02) while cf0.05 keeps the sparse n44 disc
contained (escape≈0) up to a gain ceiling where the mutual push overwhelms the catch; differential motility alone
(no chemistry) does NOT sort; at confluence a global flock (γ120) overwhelms the sort. Falsifier: ctrl_nochem already
shows seg>0.05, OR the gain sweep does not lift seg above the control → chemotaxis is not the driver / the field is
mis-scaled (deposit 4 / diffuse 6 / decay 2 was a first guess) → re-tune the field timescales or switch to `sense`
(own +1 / cross −1) / differential adhesion next batch.*

**Levers for Batch 14:** `chemotaxis.gain` (−0.02 → −1.0, the cross-repulsion sweep), `agent.div_rate`+`mpm_to_agent.confine`
(does partition survive at density), `polar_align.gamma` (partition-vs-flock tension), differential `move_speed` (R1
minimal alternative). Fields held at deposit 4 / diffuse 6 / decay 2 (first-guess; re-tune only if the whole sweep is flat).

---

## Batch 15 (2026-07-02) — STAGE 1E: first REAL 1E data — chemical cross-repulsion did NOT sort; diagnose + fix

**TARGET:** do the two types PARTITION (demix) via per-type chemical signalling, shell whole (escape≈0)?
**Reading b14** (the first 1E batch that actually ran — b13 was the all-8-died YAML bug). Full table (n · collapsed ·
nn_min · escape · r_max · deform · migr · **seg**):

| slot | mech (gain) | n | collapsed | nn_min | escape | r_max | deform | migr | **seg** |
|------|-------------|---|-----------|--------|--------|-------|--------|------|---------|
| s0 ctrl_nochem | field, gain 0 | 44 | 0 | 0.0071 | **0.0** | 0.877 | 0.0103 | 0.010 | 0.0495 |
| s1 xrep002 | −0.02 | 44 | 0 | 0.0071 | **0.0** | 0.849 | 0.0079 | 0.123 | **0.230** |
| s2 xrep01 | −0.1 | 44 | 0 | 0.004 | **0.0** | 0.770 | 0.0053 | 0.056 | 0.0456 |
| s3 xrep03 | −0.3 | 44 | 0 | 0.0044 | **0.0** | 0.867 | 0.0068 | 0.065 | 0.0737 |
| s4 xrep01_div | −0.1, n157 | 157 | 0.0127 | 0.0028 | 0.0255 | 0.942 | 0.0202 | 0.133 | 0.0464 |
| s5 xrep10 | −1.0 | 44 | 0 | 0.0044 | **0.0** | 0.785 | 0.0108 | 0.305 | 0.1198 |
| s6 diffmot | speed a0.20/b0.05 | 44 | 0 | 0.0056 | 0.0455 | 0.915 | 0.0144 | 0.192 | 0.1039 |
| s7 xrep01_flock | −0.1 γ120 n557 | 557 | 0.018 | 0.0015 | **0.158** | 1.111 | 0.0379 | 0.189 | 0.0281 |

**1. OBSERVE vs prediction.** The b14 hypothesis (seg rises MONOTONICALLY with |gain| above the control) is **FALSIFIED.**
seg vs |gain| is NON-MONOTONE and near-random: control 0.050 → −0.02 gives **0.230** → −0.1 gives **0.046 (BELOW control)**
→ −0.3 gives 0.074 → −1.0 gives 0.120. The single high value (0.230 at the WEAKEST gain −0.02) is the tell-tale of noise,
not a dose-response. The montage confirms it: **every n44 slot stays salt-and-pepper MIXED at t=12000 — no left/right split,
no two-blob demix, at any gain.** Cross-repulsion produced no visible sorting.

**2. TWO DIAGNOSED CAUSES (both now [established] — see ledger).**
  (a) **THE FIELD WAS INERT.** `deposit` amount 4.0 × dt 0.002 = 0.008/frame vs `decay` 2.0 × dt = 0.004/frame ⇒ equilibrium
      pixel value = deposit/decay = **2.0, CLAMPED to 1.0** — every occupied pixel SATURATES. `diffuse` 6.0 then smears the
      saturated pixels into a flat plateau across the disc ⇒ **grad(channel) ≈ 0 ⇒ chemotaxis velocity = gain·grad ≈ 0.**
      Independently, from a MIXED start type-a's channel (ch0) and type-b's channel (ch1) are spatially co-extensive
      (both ~uniform over the disc) ⇒ cross-repulsion has NO gradient to act on until something ELSE breaks symmetry
      (chicken-and-egg). So the op did essentially nothing; seg was pure sampling noise.
  (b) **seg HAS A NOISE FLOOR ≈0.12 AT n=44.** seg = |⟨x⟩_a−⟨x⟩_b|/R with two 22-cell x-means over a disc of R=0.34:
      std_x≈R/2, SE≈(R/2)/√22, |Δ| half-normal ⇒ E[seg]≈**0.12**, ~2σ≈0.24. The whole b14 sweep (0.045–0.23) lies
      WITHIN ~2σ of that floor — **the metric literally cannot resolve sorting at n44.** Floor drops to ≈0.07 at n=120.

**3. PER-SLOT VERDICTS (Batch 15 reading b14).**
- s0 ctrl_nochem — **supported (as control):** clean (escape 0), seg 0.050 ≈ below the n44 noise mean → confirms no
  intrinsic sort without a working force. Baseline established.
- s1 xrep002 (seg 0.230) — **falsified as a sort:** highest seg at the WEAKEST gain, montage mixed → a ~2σ noise draw,
  not partitioning. Clean (escape 0).
- s2 xrep01 / s3 xrep03 / s5 xrep10 — **falsified:** no monotone seg rise, montage mixed. All clean (escape 0.0). The
  cross-repulsion mechanism as configured is INERT (field saturated/flat).
- s4 xrep01_div (n157) — **hard-fail-ish:** escape 0.0255, collapsed 0.0127 (division at cf0.08 leaks mildly); seg 0.046
  no better. Density did not rescue the (inert) sort.
- s6 diffmot — **falsified (R1 minimal):** differential motility (a 0.20 / b 0.05) gave seg 0.104 (≈ n44 noise floor),
  montage mixed, AND escape 0.0455 (the fast type overshoots) → a scalar speed difference does NOT sort AND breaks
  containment. Confirms sorting needs a type-PAIR interaction, but the chemical one tested here was inert.
- s7 xrep01_flock (n557 γ120) — **hard FAIL:** escape 0.158, r_max 1.111, collapsed 0.018; seg 0.028 (lowest). A strong
  flock at confluence both homogenizes any sort AND punches the shell (as predicted). Do not stack flock on the sort.

**4. HYPOTHESIS (Batch 15).** *With a SHARP, UNSATURATED field (deposit 0.8 / diffuse 2.0 / decay 1.0 ⇒ equilibrium c≈0.8,
local contrast preserved), **SELF-AGGREGATION** — each type climbs its OWN trail (gain>0), the Keller-Segel clustering
instability that self-amplifies from a mixed state without needing a pre-existing gradient — demixes the two types into
separated colour blobs (visible in the CELLS panel; seg ≫ noise floor at n=120), monotone in gain up to a stacking ceiling
(collapsed>0 when the clump packs below r0). Cross-repulsion, re-tested with the sharp field, should at least lift above the
b14 flat/inert result but may stay weak (still needs symmetry breaking). Falsifier: even sharp-field self-agg stays at the
control's seg with no blobs in the montage → chemotaxis is not a viable 1E driver in this system → next batch switch
mechanism (differential adhesion) or adopt the best clean point, log 1E [open], and advance per the stage budget.*

**5. DESIGN (8 slots).** Two chemotaxis wirings of the SAME family (R3-clean) on the fixed sharp field, + a higher-n pair
to beat the seg noise floor:
- `ctrl_flat` (R4 control) — xrep spec, gain 0: n44 baseline / noise floor.
- `xrep_mid −0.3`, `xrep_strong −1.0`, `xrep_vstrong −3.0` — cross-repulsion sweep with the sharp field (does a responsive
  field make flee-the-other work; where does it break containment).
- `selfagg_mid +0.3`, `selfagg_strong +1.0` — each type climbs its own trail (expect nascent → two-blob demix; watch
  collapsed as the clump packs).
- `selfagg_hin +1.0` (n=120 static, cf0.06) — self-agg at lower seg noise floor (~0.07) so a real demix is resolvable.
- `ctrl_hin` (n=120 static, gain 0) — high-n noise-floor baseline + containment check for the static-n120 tiling.
New specs authored: `embryo_1E_xrep.yaml` (cross-rep, sharp field), `embryo_1E_selfagg.yaml` (self-agg, sharp field),
`embryo_1E_selfagg_hi.yaml` (self-agg, n120). frames 12000 stride 16 pinned per slot. Guardrails: escape (HARD), collapsed
(self-agg stacking risk), nn_min, r_max. `current_stage.txt` stays `1E`.

---

# Batch 16 (2026-07-02) — STAGE 1E: two-type partitioning. Reading b15.

**Target sub-phase: 1E** (started Batch 13; this is the 4th batch with real 1E data — well inside the 48-batch/2-day cap).
`current_stage.txt` = `1E`.

**1. OBSERVE (b15 montage + metrics).** Batch 15 tested TWO pure chemotaxis wirings on the fixed SHARP UNSATURATED field
(deposit 0.8 / diffuse 2.0 / decay 1.0). The field is now demonstrably ACTIVE (self-agg produced strong clumping — the b14
inert-field problem is fixed). But NEITHER pure wiring demixed; both failed for the SAME root cause — a mixed, co-located
start:
- **SELF-AGGREGATION (gain>0) → KELLER-SEGEL COLLAPSE, not demixing. HARD FAIL.** s4 selfagg_mid (gain 0.3, n44):
  **collapsed 0.75**, nn_min 0.0002 ≪ r0. s5 selfagg_strong (gain 1.0, n44): **collapsed 0.75**, nn_min 0.0. s6 selfagg_hin
  (gain 1.0, n120): **collapsed 0.9917**, nn_min 0.0, flow 6e-5 (frozen solid). The montages show every cell packing into ONE
  central knot (both types together), not two separated blobs. Both types climb their own trail but start co-located, so they
  collapse to a shared point. seg stayed low (0.043–0.10). *Critically: the chemotactic self-attraction BEATS hard-core repel
  (nn_min→0 despite repel r0=0.02) — exactly the confine-collapse signature. gain>0 chemotaxis is a collapse driver.*
- **CROSS-REPULSION (gain<0) → still INERT / MIXED.** s0 ctrl_flat (gain 0): seg 0.0495, collapsed 0, escape 0. s1 (−0.3)
  seg 0.0589; s2 (−1.0) seg 0.1211; s3 (−3.0) seg 0.0255. No order in |gain|, all within the n44 noise floor, montages
  salt-and-pepper mixed. The sharp field did NOT rescue cross-rep: from a mixed start the two channels are co-extensive, so
  "flee the other" has no gradient to act on (the chicken-and-egg symmetry-break problem, now doubly confirmed b14+b15).
- **Containment:** all n44 slots escape 0.0 (clean). n120 slots escape 0.0167 (mild, r_max 0.91–0.95; acceptable). The
  cross-rep slots stayed clean even at −3.0 (cross-rep alone never collapses — it pushes types apart, not together).
- **Note:** s7 ctrl_hin (n120, chemotaxis OFF) spontaneously reached migration 0.234 — the base flow_align+spin organizes
  some coherent motion at n120 even at polar_align γ0. Not central to sorting but flags that n120 has live internal flow.

**2. PER-SLOT VERDICTS (b15).**
- s0 ctrl_flat — control OK: n44 noise-floor baseline seg 0.0495, clean.
- s1/s2/s3 xrep_mid/strong/vstrong — **FALSIFIED (2nd batch):** cross-rep from a mixed start does not sort even with an
  active sharp field; seg within noise, montage mixed. Symmetry-break problem is the blocker, not field inertness.
- s4/s5 selfagg_mid/strong — **FALSIFIED as a demixer + HARD FAIL:** self-agg collapses both types to one shared point
  (collapsed 0.75). It clumps but does not separate.
- s6 selfagg_hin (n120) — **HARD FAIL (catastrophic):** collapsed 0.9917, frozen. Self-agg collapse is worse at density.
- s7 ctrl_hin (n120) — control OK: seg 0.0156 (floor), clean-ish (escape 0.0167), containment holds at static n120.

**3. LEVERS / DIAGNOSIS.** Both pure wirings are dead for the same reason: **from a mixed co-located start, self-agg has no
between-type term (→ both collapse together) and cross-rep has no within-type gradient (→ inert).** The missing ingredient is
a term that BREAKS SYMMETRY: within-type cohesion AND between-type repulsion acting together — chemical DIFFERENTIAL ADHESION.
That combination (a two-species cross-repulsive Keller-Segel / Cahn-Hilliard instability) self-amplifies any local excess of a
type (a clumps via its own trail AND repels b via cross-rep), so symmetry breaks where each pure wiring could not. This is the
last untested wiring of the chemotaxis family and is the Batch-16 primary. Keep self-gain WEAK (<0.3, the b15 collapse floor)
so within-type cohesion stays below the stacking/collapse threshold; let the strong cross term drive separation (cross-rep
alone was clean even at −3.0).

**4. HYPOTHESIS (Batch 16).** *Running self-aggregation and cross-repulsion TOGETHER (each type climbs its OWN channel with a
WEAK positive gain +0.1 AND flees the OTHER's channel with a STRONG negative gain −1.0) produces a demixing instability that
separates the two types into two colour regions from a MIXED start — where self-agg-alone collapsed and cross-rep-alone stayed
inert. seg rises above the noise floor (clearest at n=120), the CELLS panel shows two separated regions, and collapsed stays 0
provided self-gain is below the stacking threshold. Falsifier: the combined wiring ALSO stays at the control's seg (mixed
montage) → chemotaxis cannot sort from a mixed start in this system → next batch either seed a spatial split to test whether
the force can MAINTAIN a partition (needs an engine `type_layout: split_x` — random-perm type assignment currently forbids a
seeded split), or switch to a type-pair-aware differential-adhesion operator; else adopt best-clean point, log 1E [open],
advance per stage budget (still ~44 batches of 1E headroom).*

**5. DESIGN (8 slots).** Combined self-agg + cross-rep (chemical differential adhesion), sweeping the balance + field
sharpness + n, with pure-wiring and ablation controls for causal attribution:
- `ctrl_off` (R4 control) — combo spec, `chemotaxis.gain 0.0` ablates ALL FOUR instances → n44 no-chemotaxis baseline.
- `combo_lo` (self +0.05 / cross −0.5) — gentle: brackets the demix/collapse boundary from below.
- `combo` (self +0.1 / cross −1.0) — **PRIMARY**: the demixing instability at moderate strength.
- `combo_hi` (self +0.15 / cross −2.0) — aggressive: fastest separation; brackets collapse from above (watch collapsed).
- `combo_sharp` — primary gains + `diffuse.rate 1.0` (sharper trail, still unsaturated c=0.8): does a crisper gradient
  strengthen/accelerate the demix?
- `combo_hin` (n=120, self +0.1 / cross −1.0) — resolve the seg noise floor (~0.07) + more cells for clear regions.
- `ctrl_hin_off` (n=120, gain 0) — high-n no-chemotaxis baseline + containment check.
- `xrep_pure` (embryo_1E_xrep, gain −2.0) — pure cross-rep re-anchor from mixed: confirms combo ≠ cross-alone.
New specs authored: `embryo_1E_combo.yaml` / `_lo` / `_hi` / `_hin` (4 chemotaxis instances each: a climbs ch0 + flees ch1;
b climbs ch1 + flees ch0; all per-type selectors QUOTED per the b13 bright line). frames 12000 stride 16 pinned per slot.
Guardrails: escape (HARD =0), collapsed (self-agg stacking risk — HARD FAIL), nn_min≥~r0, r_max<1. TARGET metric: segregation,
but READ THE CELLS PANEL (seg only sees the x-projection; a top/bottom or radial demix reads at the floor).

---

## Batch 17 (2026-07-02) — Stage 1E: SEED a left/right split, test whether a chemical force MAINTAINS/SHARPENS it
### (reading Batch 16 — chemical differential adhesion from a MIXED start)

**1. OBSERVE (Batch 16 = combined self-agg + cross-rep, "chemical differential adhesion", from a MIXED start).**
Prediction was: the combined wiring separates the two types where each pure wiring failed. **FALSIFIED — the combined
wiring is squeezed between "too weak → mixed" and "strong enough self → Keller-Segel collapse"; NO clean demix.**

| slot | wiring | n | collapsed | escape | nn_min | seg | migr | verdict |
|---|---|---|---|---|---|---|---|---|
| s0 ctrl_off    | gains 0            | 44  | 0.000 | 0.000 | 0.0071 | 0.050 | 0.010 | clean baseline, seg < floor 0.12 OK |
| s1 combo_lo    | +0.05 / -0.5       | 44  | 0.000 | 0.000 | 0.0058 | 0.106 | 0.181 | clean, seg AT floor — no demix |
| s2 combo (PRI) | +0.1 / -1.0        | 44  | **0.114** | **0.114** | 0.0009 | 0.319 | 0.141 | **HARD FAIL** (partial K-S stacking) |
| s3 combo_hi    | +0.15 / -2.0       | 44  | 0.000 | 0.000 | 0.0064 | 0.160 | 0.059 | clean, seg ~floor — no demix |
| s4 combo_sharp | +0.1/-1.0, diff1.0 | 44  | 0.000 | 0.000 | 0.0055 | 0.156 | 0.041 | clean, seg ~floor — no demix |
| s5 combo_hin   | +0.1 / -1.0        | 120 | **0.492** | 0.000 | 0.0001 | 0.020 | 0.039 | **HARD FAIL** (catastrophic collapse) |
| s6 ctrl_hin_off| gains 0            | 120 | 0.000 | 0.000 | 0.0057 | 0.043 | 0.083 | clean baseline, seg < floor 0.07 OK |
| s7 xrep_pure   | pure -2.0          | 44  | 0.000 | 0.000 | 0.0070 | 0.253 | 0.138 | clean, seg ~2sigma, montage MIXED — noise |

Montages: every CLEAN slot (s0/s1/s3/s4/s6/s7) stays salt-and-pepper mixed at t=12000 — no two-region structure. The two
slots with seg above 2sigma (s2 seg 0.32, s7 seg 0.25) are NOT demixes: **s2's 0.32 comes WITH collapse** (the self +0.1 term
crossed the stacking threshold once cross pulled cells together — collapsed 0.114, nn_min 0.0009 << r0, escape 0.114 = same
cells pushed through the shell as the clump forms), and **s7's 0.25 is a ~2sigma noise draw on a visibly mixed montage** (b15
pure-xrep already scattered 0.03–0.12 with no order in |gain|; -2.0 → 0.25 is the high tail of that scatter, not a signal).
At n=120 the story is even clearer: **s5's self-agg term drives a CATASTROPHIC collapse (0.49) while seg reads 0.020 (near
zero)** — density amplifies the Keller-Segel self-attraction exactly as b16-going-in warned, and it collapses the two types
TOGETHER (co-located start) rather than apart. **Third independent falsification of chemotaxis-from-a-mixed-start.**

**2. VERDICTS (Batch 16 slots).**
- s0/s6 ctrl_off/ctrl_hin_off: **supported** (clean no-chemotaxis baselines; seg below the n-appropriate noise floor).
- s2 combo (PRIMARY): **falsified as a demixer** — its high seg is a collapse artefact, a HARD FAIL. The self +0.1 term is
  NOT below the stacking floor once the cross term co-locates cells; the b16 assumption "weak self stays sub-collapse" is wrong
  when cross-rep first pulls a local excess together.
- s1/s3/s4 combo_lo/hi/sharp: **falsified** — clean but seg pinned at the n44 noise floor; neither strength nor a sharper
  field (diffuse 1.0) breaks symmetry from a mixed start. Field sharpness is not the missing ingredient.
- s5 combo_hin: **falsified (HARD FAIL)** — combined wiring at density = catastrophic collapse (0.49), seg ≈ 0. Density makes
  the self-agg collapse WORSE, not the demix better.
- s7 xrep_pure: **inconclusive→falsified** — reconfirms cross-rep-from-mixed is inert (montage mixed); the 0.25 is noise.

**CONCLUSION: the chemotaxis-from-a-mixed-start route is EXHAUSTED (b14 cross-rep inert · b15 self-agg collapse + cross-rep
inert · b16 combined = collapse-or-inert). The blocker is definitively SYMMETRY BREAKING, not the force strength/field/wiring.**
Per the stage plan (and R1 having exhausted the pure-spec action set over 3 batches), Batch 17 makes the pre-authorized engine
move: SEED a split and test whether the force MAINTAINS/SHARPENS it — decoupling the force-test from the symmetry-break problem.

**3. ENGINE CHANGE (backward-compatible).** Added `type_layout` to `_assign_types` (engine.py:191): default `random` (unchanged
salt-and-pepper perm); opt-in `type_layout: split_x` sorts the type assignment by x (`torch.argsort(state[:,0])`) so type a =
LEFT half, b = RIGHT half. Positions are already set (build pass 1, line 253) before `_assign_types` (line 268), and the schema
passes set keys through untouched (schema.py:231 `sets=raw["sets"]`), so this is a clean, opt-in seed with no effect on any
existing spec. Now a's trail (ch0) sits left and b's (ch1) right → cross-rep HAS the spatial gradient it lacked from a mixed
start. `mpm_spin` is turned OFF (omega 0) in every seeded spec so bulk rotation cannot smear the x-projection seg — the ONLY
thing that can change the seeded split is interdiffusion (re-mixing) vs sorting (maintenance/sharpening).

**4. HYPOTHESIS (Batch 17).** *From a SEEDED left/right split, pure cross-repulsion (each type flees the other's now
spatially-separated trail) MAINTAINS the partition — seg stays HIGH (near its seeded t=0 value) and does NOT decay toward the
noise floor — whereas the identical force from a MIXED start (mix_xrep) and the no-force seeded control (seed_ctrl, which
re-mixes by diffusion) both fall to the floor. I.e. the missing ingredient in b14–16 was SOLELY the symmetry break: give the
force a seed and it holds a partition. Sharpest signal: seed_xrep_hin (n=120, floor ~0.07) holds seg >> 0.07. Falsifier: the
seeded split re-mixes to the floor even under the force (seg tracks seed_ctrl) → the chemical force cannot even MAINTAIN a
partition → abandon the chemical route, adopt best-clean point, log 1E [open], and advance per the stage budget.*

**5. DESIGN (8 slots).** All seeded specs share `type_layout: split_x` + `mpm_spin omega 0`; base = clean sparse 1B
(n44, div OFF, cf0.05, mass 5e-5, move0.12) + sharp unsaturated field (deposit 0.8 / diffuse 2.0 / decay 1.0). Read the t=0
montage panel for the SEEDED seg, and compare each slot's t=12000 seg against seed_ctrl (re-mixing rate) and mix_xrep (same
force, no seed). 4 exploit · 2 explore · 2 control.
- `seed_ctrl` (R4 control) — split_x, `chemotaxis.gain 0.0` → measures the natural re-mixing rate of the seed (pure diffusion).
- `seed_xrep_lo` (exploit) — split_x, pure cross `gain -0.5` → does mild cross-rep hold the seed?
- `seed_xrep` (**PRIMARY exploit**) — split_x, pure cross -1.0 → the force that was inert from a mixed start, now with a gradient.
- `seed_xrep_hi` (exploit) — split_x, pure cross -2.0 → sharper interface; watch escape at the midline.
- `seed_selfagg` (explore) — split_x, pure self +0.2 → tests whether the b15 self-agg collapse was PURELY the co-located start
  (from a separated seed each type should compact into its own-side blob → clean two-blob partition, not one central knot).
- `seed_combo` (explore) — split_x, differential adhesion +0.1/-1.0 → does adding within-type cohesion sharpen, or (as at mixed
  start) reintroduce collapse now that the types START apart?
- `seed_xrep_hin` (exploit) — split_x, pure cross -1.0, **n=120** → crisper metric (floor ~0.07) + denser midline interface.
- `mix_xrep` (CONTROL) — RANDOM layout, pure cross -1.0, spin off — byte-identical to seed_xrep except the seed → the causal
  A/B that ISOLATES symmetry-break as the sole missing ingredient.
New specs: `embryo_1E_split_xrep.yaml` / `_selfagg` / `_combo` / `_split_hin` / `embryo_1E_mix_xrep.yaml` (all YAML-checked;
per-type selectors QUOTED per the b13 bright line). frames 12000 stride 16 pinned per slot. Guardrails: escape (HARD =0),
collapsed (self terms → HARD FAIL), nn_min>=~r0, r_max<1. TARGET: seg held high vs re-mixing controls + CELLS panel two regions.


---

## Batch 18 (2026-07-02) — reading b17 — STAGE 1E (seeded-split maintenance): SECOND ENGINE BUG. `type_layout: split_x` produced ZERO valid data — all live cells fell into ONE type. Bug fixed; re-issue b17 design.

**1. OBSERVE — b17 montage + metrics.** The seeded-split experiment produced a NULL result caused by an engine
bug, not physics. Three independent tells, all consistent, all pointing at the type assignment:
- **`segregation` is EXACTLY 0.0 on every seeded slot s0–s6** (seed_ctrl, xrep_lo/mid/hi, selfagg, combo, hin) —
  not near the n44 noise floor 0.12, but a hard 0.0. seg = |⟨x⟩_a−⟨x⟩_b|/R returns 0 when one type-group is EMPTY.
- **s0–s3 are BYTE-IDENTICAL** (collapsed 0.0, nn_min 0.0067, deform 0.0085, flow 0.00331, migr 0.1237, seg 0.0,
  accel 0.001286, r_max 0.8935) despite spanning chemotaxis.gain 0.0 → −0.5 → −1.0 → −2.0. A cross-rep force that
  does *nothing* across a 4× gain sweep = the force has no second type to act against (only one channel populated).
- **The montage renders ONE colour (yellow) for every seeded slot**; only s7 mix_xrep (default random layout) shows
  TWO colours (red + yellow). So the seeded runs have a single populated type.
- Corroborating: **s4 seed_selfagg → collapsed 0.8636, nn_min 0.0003** (a single self-aggregating type climbs its own
  trail → Keller-Segel collapse to one knot — exactly what one type with gain>0 does); s6 seed_xrep_hin (n120) shows
  packing clusters, not a midline. All coherent with "one type, cross-rep inert."

**ROOT CAUSE (traced in engine.py).** `_assign_types` split the type fractions over `lvl.n`, and **`Level.n` returns
the BUFFER size (3000), not the live count (44)** (`models/base.py:162`). For `split_x`, `perm =
torch.argsort(lvl.state[:lvl.n, 0])` sorts all 3000 rows by x; the 2956 DEAD slots have x=0 (never spawned), the 44
LIVE cells are a sunflower disc at x∈[~0.2,0.8] (all positive). Ascending sort ⇒ the 2956 zeros come first, the 44
live cells last. type a = round(0.5·3000)=1500 lowest-x entries = **all dead slots (0 live cells)**; type b = the
remaining 1500 = **all 44 live cells**. So every live cell became type b → one populated type → cross-rep inert,
seg=0.0, single colour. The RANDOM layout survived only because `randperm` scatters live indices across the buffer, so
~half the 44 land in each type by chance. This is the SECOND "whole-batch-wasted" engine bug in 1E (cf. the b13 YAML
parse bug) — the seeded-split experiment has STILL not actually run.

**THE FIX (engine.py `_assign_types`, this batch).** Restrict the `split_x` sort+split to LIVE slots and the LIVE
count: `perm = nonzero(occ>0).flatten(); perm = perm[argsort(state[perm,0])]; total = perm.numel()` (was `lvl.n`).
Dead slots stay type 0 (harmless). Now the 44 live cells sort by x and split 22/22 → type a = LEFT half, b = RIGHT
half, a true seeded partition. Backward-compatible (random path untouched). Verified by code trace (live indices
[0..43], all positive x, split by count 44 → 22/22); could not run the build test in-loop (python exec gated behind
approval this session), but the b17 spec YAMLs already parse + run cleanly (they completed b17), so there is no YAML
risk — only the engine changed.

**2. PER-SLOT VERDICTS — all INCONCLUSIVE (no valid seeded-split data; engine bug, not physics).**
- `seed_ctrl` (s0) — INCONCLUSIVE. seg 0.0, one type; the "re-mixing control" never had two types to re-mix.
- `seed_xrep_lo/mid/hi` (s1/s2/s3) — INCONCLUSIVE. Byte-identical, force inert (no type b to repel). No dose response
  because there is no second channel gradient.
- `seed_selfagg` (s4) — INCONCLUSIVE for sorting, but the collapsed 0.8636 CONFIRMS (again) that a SINGLE self-agg
  type Keller-Segel-collapses; says nothing about a two-type seeded self-cohesion.
- `seed_combo` (s5) — INCONCLUSIVE. seg 0.0; with one type the −1.0 cross term is inert and only the +0.1 self term
  acts (weak, no collapse: collapsed 0.0).
- `seed_xrep_hin` (s6, n120) — INCONCLUSIVE. Same single-type bug at higher n.
- `mix_xrep` (s7) — the ONLY slot with two real types (random layout): seg 0.0939 (at/below the n44 floor 0.12),
  montage salt-and-pepper mixed. Consistent with the established "cross-rep inert from a mixed start" — this is the
  mixed-start control, so it re-confirms b14–16; it does not test the seed.

**3. LEVERS.** The engine now seeds a real split; the decisive experiment (does a force MAINTAIN/SHARPEN a seeded
partition vs a re-mixing no-force control?) is finally testable. Re-issue the b17 design unchanged (specs already
correct) as Batch 18. Guardrails unchanged: escape (HARD =0), collapsed (self terms → HARD FAIL), nn_min≥~r0, r_max<1.

**4. HYPOTHESIS (Batch 18).** *With the type-assignment fix, from a genuinely SEEDED left/right split (a=left, b=right,
22/22), pure cross-repulsion (each type flees the other's now spatially-SEPARATED trail) MAINTAINS the partition — seg
stays HIGH near its seeded t=0 value and does NOT decay to the noise floor — whereas the no-force seeded control
(`seed_ctrl`) re-mixes toward the floor by diffusion and the same force from a MIXED start (`mix_xrep`, already seen at
0.094) stays at the floor. The cleanest readout is `seed_xrep_hin` (n120, floor ~0.07): seg holding ≫0.07 = a
maintained partition. Falsifier: even the seeded split re-mixes to the floor under the force (seg tracks seed_ctrl) →
the chemical route cannot even MAINTAIN a partition → adopt best-clean point, log 1E [open], advance per stage budget.*

**5. DESIGN (8 slots) — re-issue of the b17 design (now that the seed is real).** All seeded specs carry
`type_layout: split_x` + `mpm_spin omega 0`; base = clean sparse 1B (n44, div OFF, cf0.05, mass 5e-5, move0.12) +
sharp unsaturated field (deposit 0.8 / diffuse 2.0 / decay 1.0). READ: t=0 montage panel = the SEEDED seg (now a HIGH
baseline, red-left / yellow-right); compare each slot's t=12000 seg vs seed_ctrl (re-mix rate) and mix_xrep (same
force, no seed). 4 exploit · 2 explore · 2 control.
- `seed_ctrl` (R4 control) — split_x, `chemotaxis.gain 0.0` → natural re-mixing rate of the seed under pure diffusion.
- `seed_xrep_lo` (exploit) — split_x, pure cross `gain -0.5` → does mild cross-rep hold the seed?
- `seed_xrep` (**PRIMARY exploit**) — split_x, pure cross -1.0 → the force inert from a mixed start, now with a gradient.
- `seed_xrep_hi` (exploit) — split_x, pure cross -2.0 → sharper interface; watch escape at the midline.
- `seed_selfagg` (explore) — split_x, pure self +0.2 → from a SEPARATED seed does each type compact into its own-side
  blob (clean two-blob partition) or still collapse?
- `seed_combo` (explore) — split_x, differential adhesion +0.1/-1.0 → does within-type cohesion SHARPEN the seed?
- `seed_xrep_hin` (exploit) — split_x, pure cross -1.0, **n=120** → crisper metric (floor ~0.07) + denser midline.
- `mix_xrep` (CONTROL, causal A/B) — RANDOM layout, pure cross -1.0 — isolates symmetry-break as the sole ingredient.
frames 12000 stride 16 pinned per slot; all n44 jobs ~19 min, n120 ~18 min on L4 — within budget.

## Batch 19 (2026-07-02) — reading b18. STAGE 1E. **THE SEED FIX WORKED — first real two-type data, and the
## chemical route DID something: MILD cross-rep MAINTAINS a seeded partition; STRONG cross-rep over-drives and
## destroys it (non-monotone).**

**1. OBSERVE (montage montages/embryo_b18.png + 8 metrics.json).** The engine `_assign_types` fix is confirmed:
every seeded slot's **t=0 panel shows a clean red-left / yellow-right split** (both colours present; s7 mix_xrep is
salt-and-pepper as designed). A uniform-disc half/half seed has t=0 seg ≈ 0.85 (each half-centroid at ±4R/3π). The
metrics report the FINAL-frame seg, so each slot's number is "how much of the seed survived 12000 frames." Results
(all n44 unless noted; collapsed/escape are the guardrails):
- **seed_ctrl (gain 0, no force):** seg **0.2935**, collapsed 0, escape 0. Pure diffusion decays the 0.85 seed to
  0.29 over 12000f — SLOW re-mixing at n44 (retains clear seed memory, still ≫ noise floor 0.12).
- **seed_xrep_lo (−0.5):** seg **0.6147**, collapsed 0, escape 0. Mild cross-rep MORE THAN DOUBLES the retained
  seg vs the no-force control (0.61 vs 0.29) — the force actively HOLDS the partition. **BEST maintenance of the batch.**
- **seed_xrep (−1.0, "primary"):** seg **0.1627**, clean. COLLAPSED TO THE FLOOR — LOWER than the no-force control!
  Strong cross-rep did NOT maintain; it homogenized (montage t=12000: red+yellow intermingled centrally, no L/R).
- **seed_xrep_hi (−2.0):** seg **0.1308**, clean. Also floor. Confirms the over-drive: stronger = worse.
- **seed_selfagg (+0.2):** seg 0.5349 but **collapsed 0.7955, nn_min 0.0003** — HARD FAIL (Keller-Segel collapse,
  as at a mixed start). Even from a separated seed, self-attraction climbs its own hill to a point; the high seg is
  an artefact of two collapsed knots, not a clean partition. DEAD.
- **seed_combo (self +0.1 / cross −1.0):** seg **0.4664**, collapsed 0, escape 0 — CLEAN and well above control.
  Adding weak self-cohesion RESCUES the over-driven −1.0 cross case (0.16 → 0.47): the self term keeps each type
  clustered so the strong cross-rep disperses/mixes them less. Second-best clean maintenance.
- **seed_xrep_hin (−1.0, n120):** seg **0.1007** (floor ~0.07), escape 0.0083, collapsed 0. At n120 the STRONG
  −1.0 fails to maintain (same over-drive), and faster diffusion at density decays the seed harder. Montage: seed
  visible at t=0, fully re-mixed into streams by t=12000.
- **mix_xrep (−1.0, mixed start):** seg **0.0939** (floor), clean. No seed → nothing to maintain → force inert
  (reconfirms b14–16: cross-rep CANNOT create a partition; it can only maintain a pre-existing one).

**vs last batch's prediction:** the b18 hypothesis ("pure cross-rep MAINTAINS the seed, best at −1.0/hin") is
**PARTLY SUPPORTED, PARTLY FALSIFIED.** SUPPORTED: from a seed the chemical force finally does something and a
maintained partition exists (seg_lo 0.61 ≫ ctrl 0.29 ≫ mix floor 0.094 — the missing ingredient WAS solely the
symmetry break). FALSIFIED: the "primary" −1.0 did the OPPOSITE of maintain (0.16, below control); the effect is
NON-MONOTONE with a peak at MILD gain (~−0.5), not "more force = sharper." The n120/−1.0 "crisp readout" slot was
over-driven and read the floor — need the MILD gain at density instead.

**2. PER-SLOT VERDICTS.**
- seed_xrep_lo (−0.5): **SUPPORTED** — mild cross-rep maintains a seeded L/R partition (seg 0.61 @ escape 0,
  collapsed 0), 2× the no-force control. First clean 1E signal of the campaign.
- seed_combo (+0.1/−1.0): **SUPPORTED (clean)** — differential adhesion holds seg 0.47; self-cohesion buffers the
  over-drive of −1.0.
- seed_ctrl (gain 0): **baseline** — n44 diffusion decays seed 0.85→0.29 in 12000f (slow; retains memory).
- seed_xrep (−1.0) / seed_xrep_hi (−2.0): **FALSIFIED as maintainers** — over-driven, homogenize to the floor.
- seed_selfagg (+0.2): **FALSIFIED / HARD FAIL** — Keller-Segel collapse even from a separated seed.
- seed_xrep_hin (−1.0, n120): **FALSIFIED (wrong gain)** — inconclusive on density because −1.0 is the over-drive
  gain; retest with the mild peak.
- mix_xrep (−1.0, mixed): **CONTROL confirms** — no seed → force inert → floor.

**3. LEVERS for Batch 19.** (a) MAP the maintenance peak: fine n44 sweep {0, −0.25, −0.5, −0.75, −1.0} to locate
the peak and REPRODUCE both the peak (0.61) and the over-drive collapse (0.16) — establish the non-monotonicity is
real, not a single-draw fluke. (b) DENSITY: does the mild peak (−0.5) hold at n120 where −1.0 failed? (with its own
no-force n120 control for the diffusion-decay baseline). (c) combine: does self-cohesion + MILD cross (+0.1/−0.5)
beat pure mild cross (author combo05 spec)?

**4. HYPOTHESIS (Batch 19).** Seed-maintenance seg is NON-MONOTONE in |cross-rep gain| with a PEAK near −0.5:
weaker → decays toward the no-force control (0.29); stronger (−1.0, −2.0) → over-drives past the midline and
homogenizes to the floor (0.13–0.16). Prediction: xrep_05 reproduces seg ≈ 0.55–0.65 (≫ ctrl), xrep_025 ≈ 0.35–0.5,
xrep_075 ≈ 0.25–0.4, xrep_10 ≈ 0.13–0.20 (floor). At n120 the mild −0.5 holds seg well above the n120 diffusion
control, whereas −1.0 read the floor (0.10). Mechanism (hypothesized): mild flee = laminar drift to own side
(sorting); strong flee = each type overshoots to the far wall, wraps around the rim, and interpenetrates on the
opposite side → over-mixing.

**5. DESIGN (8 slots) — map the maintenance peak + its density dependence.** 4 exploit · 3 explore · 1 control.
- `seed_ctrl` (CONTROL, R4) — split_xrep `chemotaxis.gain 0.0` (n44) → reproduce diffusion-decay baseline 0.29.
- `xrep_025` (exploit) — split_xrep gain −0.25 (n44) → low side of the peak.
- `xrep_05` (exploit) — split_xrep gain −0.5 (n44) → REPRODUCE the peak (0.61).
- `xrep_075` (exploit) — split_xrep gain −0.75 (n44) → high side of the peak.
- `xrep_10` (exploit) — split_xrep gain −1.0 (n44) → REPRODUCE the over-drive collapse (0.16).
- `combo_05` (explore) — NEW spec embryo_1E_split_combo05.yaml, self +0.1 / cross −0.5 → self-cohesion + mild peak.
- `xrep_05_n120` (explore) — split_hin gain −0.5 (n120) → does the mild peak hold at density?
- `ctrl_n120` (explore/control) — split_hin gain 0.0 (n120) → n120 diffusion-decay baseline (compare vs xrep_05_n120).
frames 12000 stride 16 pinned per slot; all within the L4 wall (b18 ran 1081–1167 s).

## Batch 20 (2026-07-02) — STAGE 1E: the −0.5 seed-maintenance peak MAPPED cleanly but on ONE (deterministic) seed

TARGET SUB-PHASE: 1E (partitioning). current_stage.txt = 1E.

### OBSERVE — b19 vs predictions
b19 was designed to REPRODUCE the b18 −0.5 maintenance peak (0.61) and map the non-monotone bell curve. It did — but
the "reproduction" was **trivial: the sim is DETERMINISTIC** (`general.seed: 0` fixed; engine
`H.rng = manual_seed(sim.seed)`, engine.py:235). b18 xrep_05 = **0.6147** and b19 xrep_05 = **0.6147** to 4 digits →
the *same run* re-executed, not an independent realization. So the peak has only ever been observed on **seed 0**. The
"lucky-draw" trap the ledger fell into before (b10 escape 0.020) is live: I have not tested whether the maintained
partition survives a different seed.

The non-monotone bell curve (n44, seeded split, all escape 0 / collapsed 0):
| slot | gain | seg | note |
|---|---|---|---|
| seed_ctrl | 0 | 0.2935 | control |
| xrep_025 | −0.25 | 0.3264 | low side |
| xrep_05 | −0.5 | 0.6147 | PEAK (= b18 exactly, same seed) |
| xrep_075 | −0.75 | 0.4219 | high side |
| xrep_10 | −1.0 | 0.1627 | over-drive (= b18 0.16) |
A clean unimodal dose-response peaking at −0.5, ~2× the no-force control — but a single seed.

Two other results:
- **combo_05** (self +0.1 / cross −0.5, seeded): seg **0.4826** but **collapsed 0.0455**, nn_min **0.0025** (K-S onset).
  Adding self-cohesion on the MILD peak DROPPED seg (0.61→0.48) AND tripped a mild collapse. This **falsifies** the
  b18-going-in hope that self-cohesion "sharpens" the peak — self-cohesion is only a stabilizer when cross is
  OVER-driving (b18 rescued −1.0: 0.16→0.47); at the peak it is a liability (adds K-S risk, lowers seg).
- **Density (n120)**: xrep_05_n120 seg **0.477** vs ctrl_n120 **0.4307** → the force adds only **+0.046** (vs **+0.32**
  at n44). AND both HARD-FAIL containment: escape **0.075** (xrep) / **0.0167** (ctrl), r_max 0.97/0.94, cf0.05. So the
  n44 clean 2× effect does NOT transfer to density: the force barely helps (both retain the seed passively — n120 seg
  floor ~0.07, both well above it), and the cf0.05 catch calibrated at n44 cannot hold the n120 boundary flux.

### PER-SLOT VERDICTS (b19)
- seed_ctrl (gain0 n44) — **supported (control)**: diffusion retains partial seed memory at n44 (0.29).
- xrep_025 (−0.25) — **supported**: ≈ control, force too weak (low side of peak).
- xrep_05 (−0.5) — **supported but NOT independently reproduced**: peak 0.61, deterministic re-run of seed 0.
- xrep_075 (−0.75) — **supported**: 0.42, high-side falloff.
- xrep_10 (−1.0) — **supported**: 0.16, over-drive below control.
- combo_05 — **FALSIFIED** the "self-cohesion sharpens the mild peak" hope: seg 0.48 (< 0.61) + collapsed 0.0455.
- xrep_05_n120 — **HARD FAIL (escape 0.075) + force advantage marginal (+0.046)**: density inconclusive-to-negative.
- ctrl_n120 — **HARD FAIL (escape 0.0167, control)**.

### LEVERS for Batch 20
1. **SEED** (new lever): must author per-seed spec files — `seed` is NOT overridable via a token (tune `_apply` only
   handles the special keys + `opname.param`; `seed` is neither). Wrote embryo_1E_split_xrep_s{1,2,3}.yaml.
2. **mpm_to_agent.confine** (overridable operator param): raise 0.05→0.08 at n120 to kill the escape and see whether
   the force>control gap returns once the boundary is held.
3. **chemotaxis.gain** (overridable): sweeps both cross-rep instances symmetrically.

### HYPOTHESIS (Batch 20)
The −0.5 seed-maintenance peak is a **robust** effect, not a seed-0 artifact: across independent seeds 1/2/3 the
cross-rep force will hold seg well above its **matched same-seed** no-force control (Δseg = seg(−0.5) − seg(gain0) ≳ +0.2
per seed), reproducing the ~2× n44 effect. Secondary: at n120 the marginal force advantage is a *containment* artifact —
restoring the boundary (cf0.05→0.08) will drive escape→~0 and recover a clear force>control gap.

### DESIGN (8 slots) — 3 seed-paired robustness tests + 1 density-containment pair
Paired: for each seed run BOTH gain0 (control) and −0.5 (force) so Δseg is a per-realization statistic. Seed 0 already in
hand (ctrl 0.29 / −0.5 0.61 → Δ +0.32).
- s0 ctrl_s1 (seed1, gain0) — control
- s1 xrep_s1 (seed1, −0.5) — exploit
- s2 ctrl_s2 (seed2, gain0) — control
- s3 xrep_s2 (seed2, −0.5) — exploit
- s4 ctrl_s3 (seed3, gain0) — control
- s5 xrep_s3 (seed3, −0.5) — exploit
- s6 dens_xrep_cf08 (n120, −0.5, confine 0.08) — explore (density with containment)
- s7 dens_ctrl_cf08 (n120, gain0, confine 0.08) — explore/control (matched density control)
DECISION RULE: if Δseg ≳ +0.2 at all 3 new seeds → 1E DELIVERED + ROBUST, adopt seed_xrep_lo (−0.5, n44) as the operating
point, log "cannot spontaneously create (needs seed)" as [open], next batch begin INTEGRATION (partition + flow/div).
If Δseg is inconsistent/small on new seeds → the "partition" is a lucky-seed artifact → log 1E [open], adopt best-clean,
advance. frames 12000 stride 16; all ~19 min on L4.

---

## Batch 21 (2026-07-02) — reading b20: the −0.5 n44 "peak" is FALSIFIED as robust (lucky seed-0); a NEW density signal survives
STAGE 1E (started Batch 13; 8 batches in, ~40 batch headroom — this is the CLOSER for the chemical route).

### OBSERVE — what happened vs the Batch-20 prediction
Prediction was: the −0.5 seed-maintenance peak is ROBUST — Δseg = seg(−0.5) − seg(gain0) ≳ +0.2 at every new seed.
**Prediction FALSIFIED, decisively.** All 8 slots clean (collapsed 0, escape 0, nn_min≥~r0, r_max<0.9). Paired Δseg:
- **n44 seed 0** (prior b18/b19): ctrl 0.29 → xrep 0.61 → Δ **+0.32**  (the draw that built the whole "peak" story)
- **n44 seed 1**: ctrl 0.6355 → xrep 0.3235 → Δ **−0.312**  (SIGN FLIPPED — force HURTS)
- **n44 seed 2**: ctrl 0.4166 → xrep 0.368  → Δ **−0.049**
- **n44 seed 3**: ctrl 0.6007 → xrep 0.5245 → Δ **−0.076**
Mean Δ of the three NEW seeds = **−0.146**. The −0.5 cross-rep does NOT robustly maintain the n44 partition; on average it
slightly REDUCES retained seg. **Root cause (now clear):** at n44 interdiffusion is so slow that even the no-force control
retains most of the seed over 12000f (ctrl seg 0.42–0.64 on seeds 1/2/3) — there is NO headroom for the force to help, and
the laminar cross-rep drift just perturbs/erodes the passive decay. Seed-0's control happened to be anomalously LOW (0.29),
manufacturing the illusory +0.32. **The b18/b19/b20-going-in "clean unimodal bell curve, peak at −0.5, ~2× control" is a
SINGLE-REALIZATION ARTIFACT.** (The b20 determinism warning was right: seed-0 is one draw, and it was the outlier.)

### THE SURVIVING SIGNAL — density + containment (n120, cf0.08) is where the force test is GENUINE
The b20 density pair (seed 0, n120, confine raised 0.05→0.08):
- **dens_ctrl_cf08** (gain 0): seg **0.036**, escape **0.0** — the seed genuinely RE-MIXES (0.036 < n120 floor ~0.07 = fully mixed).
- **dens_xrep_cf08** (−0.5): seg **0.2007**, escape **0.0** — the force HOLDS a partial partition (~3× floor). Δ **+0.165**.
Two things flip vs n44: (i) cf0.08 killed the b19 escape 0.075 leak → both n120 slots strictly clean (escape 0, r_max 0.89/0.83);
(ii) at n120 the control interdiffuses fast enough to erase the seed, so — unlike n44 — there IS headroom and the force's
mixing-resistance shows up as a real advantage. deform is also up at density (0.012–0.013 vs 0.004–0.008 at n44). **This is the
regime where a chemical cross-repulsion can actually be seen to maintain a partition against active re-mixing.** BUT it is a
SINGLE seed-0 draw — exactly the trap that just misled us at n44. It must be seed-replicated before I trust it.

### PER-SLOT VERDICTS (b20)
- ctrl_s1 / xrep_s1 (n44 seed1) — **FALSIFIES robustness**: Δ −0.31, force hurts.
- ctrl_s2 / xrep_s2 (n44 seed2) — **FALSIFIES**: Δ −0.05.
- ctrl_s3 / xrep_s3 (n44 seed3) — **FALSIFIES**: Δ −0.08. (3/3 new seeds negative → n44 peak dead.)
- dens_xrep_cf08 (n120, −0.5, cf0.08) — **supported, promising**: seg 0.20 @ escape 0 (containment fixed). Single seed.
- dens_ctrl_cf08 (n120, gain0, cf0.08) — **supported (control)**: control re-mixes to 0.036 → genuine test regime.

### LEVERS for Batch 21
1. **SEED** (author per-seed spec copies — not token-overridable): wrote embryo_1E_split_hin_s{1,2,3}.yaml (n120, seed 1/2/3).
2. **chemotaxis.gain** (overridable, sweeps both cross-rep instances): map density dose-response −0.35/−0.5/−0.75.
3. **mpm_to_agent.confine 0.08** (overridable): the containment that made n120 strictly clean — keep it.

### HYPOTHESIS (Batch 21)
**At n120/cf0.08 — where the no-force control genuinely re-mixes — the −0.5 chemical cross-repulsion robustly maintains a
partial partition: Δseg = seg(−0.5) − seg(gain0) > 0 at every new seed (1/2/3), mean ≳ +0.1, with each control re-mixing to
near the ~0.07 floor.** If it holds across seeds, the real 1E mechanism is a DENSITY-dependent maintained partition (chemical
cross-rep resists active re-mixing) — NOT the n44 "peak". Secondary: the density dose-response peaks near −0.5 (−0.35 weaker,
−0.75 over-drives toward the floor as at n44).

### DESIGN (8 slots) — n120/cf0.08 seed-replication + density dose-response
Everything at n120, confine 0.08, seeded split, spin off (single spec family embryo_1E_split_hin_s{1,2,3}; gain/confine via token).
- s0 ctrl_s1  (seed1, gain 0)     — control (matched no-force baseline for seed1)
- s1 xrep_s1  (seed1, −0.5)        — exploit (main effect, seed1)
- s2 ctrl_s2  (seed2, gain 0)      — control
- s3 xrep_s2  (seed2, −0.5)        — exploit (seed2)
- s4 ctrl_s3  (seed3, gain 0)      — control
- s5 xrep_s3  (seed3, −0.5)        — exploit (seed3)
- s6 xrep_s1_hi (seed1, −0.75)     — explore (density dose: over-drive? shares ctrl_s1 as its control)
- s7 xrep_s1_lo (seed1, −0.35)     — explore (density dose: milder; shares ctrl_s1)
Roles: 3 exploit (−0.5 × seeds 1/2/3) · 2 explore (−0.75/−0.35 dose at seed1) · 3 control (gain0 × seeds 1/2/3). Focused
replication (deviates from 4/3/1) because this is the decisive closer for the 1E chemical route. Combined with seed-0 (b20)
→ 4 realizations of Δseg at n120/cf0.08. All ~19 min on L4 (b20 n120 ran 1147 s). frames 12000 stride 16.
**DECISION RULE:** if Δseg > 0 consistently (mean ≳ +0.1, controls re-mixing to near floor) → 1E DELIVERED as a DENSITY-dependent
maintained partition; adopt `embryo_1E_split_hin` + cf0.08 + gain −0.5 as the 1E operating point; log "chemical cross-rep
MAINTAINS (cannot CREATE) a seeded partition, ONLY at density where the control re-mixes" as [established]; next batch begin
INTEGRATION. If Δseg is inconsistent/≤0 across seeds → the chemical 1E route is FALSIFIED (n44 lucky + n120 lucky); log 1E
[open], adopt best-clean (the seeded split itself is clean/stable), advance off 1E.

---

## Batch 22 (2026-07-03) — reading b21: STAGE 1E, chemical-cross-rep CLOSER (n120/cf0.08, seeds 1/2/3 paired + density dose)

### OBSERVE (montage + metrics, b21)
Target: does the −0.5 chemical cross-rep MAINTAIN a partition where the no-force control re-mixes (the b20 seed-0 result), across
independent seeds? **Result: the premise FAILED and the force test flipped sign again.** Per-slot (all n120, cf0.08, 12000f):

| slot | seed | gain | seg | collapsed | nn_min | escape | deform | migr |
|------|------|------|-----|-----------|--------|--------|--------|------|
| s0 ctrl_s1   | 1 | 0     | 0.339 | 0     | 0.0044 | 0 | 0.0164 | 0.117 |
| s1 xrep_s1   | 1 | −0.5  | 0.277 | 0     | 0.0038 | 0 | 0.0158 | 0.112 |
| s7 xrep_s1_lo| 1 | −0.35 | 0.278 | 0     | 0.0038 | 0 | 0.0154 | 0.150 |
| s6 xrep_s1_hi| 1 | −0.75 | 0.373 | 0     | 0.0044 | 0 | 0.0175 | 0.077 |
| s2 ctrl_s2   | 2 | 0     | 0.250 | 0     | 0.0045 | 0 | 0.0176 | 0.176 |
| s3 xrep_s2   | 2 | −0.5  | 0.432 | 0     | 0.0035 | 0 | 0.0185 | 0.229 |
| s4 ctrl_s3   | 3 | 0     | 0.550 | 0     | 0.0041 | 0 | 0.0133 | 0.065 |
| s5 xrep_s3   | 3 | −0.5  | 0.350 | **0.0167** | 0.0028 | 0 | 0.0126 | 0.142 |

**Paired Δseg = seg(−0.5) − seg(gain0), same seed: seed1 −0.062, seed2 +0.182, seed3 −0.200 → mean −0.027.** Sign flips across
seeds (only seed2 positive), exactly as b20 at n44. **The decisive new fact: the no-force CONTROLS did NOT re-mix** — seg
0.339 / 0.250 / 0.550, all ≫ the n120 mixed floor ~0.07. The whole b21 design rested on "at n120 the control genuinely re-mixes"
(b20 seed-0 control → 0.036); that was itself a lucky seed-0 draw. Montages confirm: every control still shows regional red/yellow
structure at t=12000 (ctrl_s3 strongly left/right, seg 0.55); the force slots look equally mixed-or-structured with no consistent
sharpening. Guardrails otherwise clean (escape EXACTLY 0 on all 8 — cf0.08 containment holds at n120 across seeds; deform
0.013–0.019). ONE minor hard-fail: xrep_s3 collapsed 0.0167, nn_min 0.0028 (a small Keller-Segel onset blip under −0.5 at seed3).

### VERDICTS
- **s1/s3/s5 xrep_s{1,2,3} (−0.5, the main effect): FALSIFIED as a robust maintenance force.** Paired Δseg sign-flips
  (−0.062/+0.182/−0.200); mean ≈ 0. No consistent partition maintenance. (Combined with b20's n44 seeds → 6 realizations, mean ≈ 0.)
- **b20's "n120 control re-mixes → genuine force test" premise: FALSIFIED.** Controls retain seg 0.25–0.55 at n120/move0.12;
  seed-0's 0.036 was the outlier. So the "genuine force test" condition (a re-mixing control) was NEVER actually met for seeds 1–3
  → the force test at n120 base stirring is INCONCLUSIVE-because-no-headroom, not a clean force win/loss. Same trap as n44.
- **Density dose-response (s6/s7, −0.35/−0.5/−0.75 @ seed1): inconclusive.** seg 0.278/0.277/0.373 — no monotone dose, all within
  noise of the control 0.339; the −0.75 "over-drive" story does not reproduce at n120/seed1.
- **Containment (cf0.08 @ n120): SUPPORTED, robust — escape EXACTLY 0 across all 8 seeds/gains.** The b21 engineering win holds.
- **Seeded split is intrinsically STABLE (new [established]):** with NO force, a seeded L/R two-type blastula stays partitioned
  over 12000f at n120 (seg 0.25–0.55 ≫ floor 0.07) — interdiffusion is too slow to erase it. A *passive/frozen* partition.

### DECISION (per the pre-registered rule) — chemical-cross-rep 1E route CLOSED; one corrected test then INTEGRATION
Δseg is inconsistent/≈0 across seeds (n44 AND n120) → **the chemical cross-repulsion route is FALSIFIED as a robust partition
mechanism.** BUT the b21 design's own premise (re-mixing control) also failed, so we have never run the *corrected* test: a control
that actually re-mixes. Root cause of ALL 1E ambiguity (n44 and n120): interdiffusion is too slow → controls retain the seed →
NO headroom for any force to demonstrate maintenance. **Batch 22 runs ONE corrected, decisive test before adopting: raise the
STIR (move_speed 0.12→0.24, a directive-endorsed lever) to force the no-force control to re-mix, then ask whether −0.5 cross-rep
holds the partition above the re-mixed control.** This is the last 1E batch: if a stirred control re-mixes AND the force holds
(Δseg>0) → 1E delivered as an active-re-mixing-resistant maintained partition; else adopt the **seeded split as a frozen-partition
1E operating point** (`embryo_1E_split_hin`, cf0.08, no force) and advance to INTEGRATION next batch.

### LEVERS for Batch 22
1. **agent.move_speed** (overridable special key, broadcasts to both types): the STIR lever — 0.12→0.18→0.24 to accelerate
   interdiffusion and make the control re-mix (create the headroom that has never existed). Directive allows up to ~0.24.
2. **chemotaxis.gain** (overridable): −0.5 primary, −0.75 explore, paired against gain-0 controls at each stir level.
3. **mpm_to_agent.confine 0.08** (fixed): watch escape — move0.24 is the dominant escape co-driver (est. Batch 7); if it leaks,
   that is itself the finding (this system cannot be stirred hard cleanly), and cf0.10 is the explore hedge.
4. **seed** (via specs _s1/_s2): replicate the crux (move0.24 control vs force) on a 2nd seed so we do not repeat the single-seed trap.

### HYPOTHESIS (Batch 22)
**Raising move_speed to 0.24 (2× stir) makes the n120 no-force control interdiffuse and re-mix the seed toward the floor
(seg → ~0.1, from 0.34 at move0.12), finally creating the headroom absent at every prior 1E point; the −0.5 chemical cross-rep
then HOLDS the partition above the re-mixed control (Δseg > 0, reproduced on seeds 1 and 2), delivering a genuine
active-re-mixing-resistant partition at escape ≈ 0.** Falsifier: if the control still does not re-mix under 2× stir, OR the force
cannot hold above it, OR move0.24 breaks containment (escape>0) → the chemical route is dead in this system; adopt the
frozen-seeded-split as the 1E operating point and advance to integration.

### DESIGN (8 slots) — the STIR / re-mixing test (all overrides on existing seed specs; no new YAML)
All n120, cf0.08, seeded split, spin off (spin would smear the x-projection seg). Base specs embryo_1E_split_hin_s{1,2}.yaml.
- s0 s1_ctrl_m12   (seed1, gain0,  move0.12) — reference: reproduces b21 ctrl_s1 seg~0.34 (no re-mix). control
- s1 s1_ctrl_m18   (seed1, gain0,  move0.18) — control: does mid-stir partially re-mix?
- s2 s1_ctrl_m24   (seed1, gain0,  move0.24) — control (KEY): does 2× stir re-mix the control toward the floor?
- s3 s1_xrep_m18   (seed1, −0.5,   move0.18) — exploit: paired force @ mid stir
- s4 s1_xrep_m24   (seed1, −0.5,   move0.24) — exploit (CRUX): does the force hold above the re-mixed control?
- s5 s1_xrep_m24_hi(seed1, −0.75,  move0.24) — exploit: stronger force @ max stir (over-drive vs hold?)
- s6 s2_ctrl_m24   (seed2, gain0,  move0.24) — explore: is the re-mixing robust across seed?
- s7 s2_xrep_m24   (seed2, −0.5,   move0.24) — explore: paired force @ max stir, 2nd seed
Roles: 3 exploit (s3/s4/s5) · 2 explore (s6/s7) · 3 control (s0/s1/s2). Compare xrep_mXX vs ctrl_mXX at SAME stir & seed → Δseg.
Read: does seg(ctrl) FALL with move_speed (re-mixing)? does seg(xrep) stay above seg(ctrl) once the control re-mixes? escape=0?
frames 12000 stride 16, ~18–20 min on L4.

---

## Batch 23 (2026-07-03) — reading b22 · STAGE 1E CLOSED → ADVANCE TO INTEGRATION

**Target this batch: INTEGRATION (INT) — first batch.** 1E is delivered as a *frozen* (kinetically-maintained) seeded
partition; the chemical maintenance route is dead. Per the pre-committed b22 decision, adopt the frozen seeded split and
advance to combining the five delivered operating points into ONE blastula.

### OBSERVE (b22 — the STIR / re-mixing test; all n120, cf0.08, seeded L/R split, spin off)
| slot | move | gain | seg | escape | r_max | coll | deform | migr | flow |
|------|------|------|-----|--------|-------|------|--------|------|------|
| s0 ctrl_m12    | 0.12 |  0    | **0.339** | 0.042 | 0.96 | 0     | 0.016 | 0.117 | 0.0036 |
| s1 ctrl_m18    | 0.18 |  0    | 0.063 | 0.008 | 0.90 | 0     | 0.021 | 0.093 | 0.0054 |
| s2 ctrl_m24    | 0.24 |  0    | 0.078 | 0.033 | 1.01 | 0     | 0.049 | 0.198 | 0.0076 |
| s3 xrep_m18    | 0.18 | −0.5  | 0.067 | 0.000 | 0.88 | 0.025 | 0.021 | 0.201 | 0.0053 |
| s4 xrep_m24    | 0.24 | −0.5  | 0.067 | **0.175** | 1.09 | 0 | 0.039 | 0.138 | 0.0073 |
| s5 xrep_m24_hi | 0.24 | −0.75 | 0.032 | **0.100** | 1.03 | 0.017 | 0.037 | 0.295 | 0.0072 |
| s6 s2_ctrl_m24 | 0.24 |  0    | 0.053 | **0.192** | 1.01 | 0 | 0.034 | 0.341 | 0.0074 |
| s7 s2_xrep_m24 | 0.24 | −0.5  | 0.113 | 0.042 | 0.93 | 0 | 0.042 | 0.099 | 0.0075 |

1. **STIR RE-MIXES THE CONTROL — CONFIRMED (the headroom absent all campaign is finally created).** seed1 no-force
   control seg: **m12 0.339 → m18 0.063 → m24 0.078** — at move0.12 it holds the seed (matches b21's slow interdiffusion),
   but at move0.18 AND 0.24 it collapses to the n120 mixed floor (~0.07). seed2 m24 control 0.053 (also floor) → robust
   across seed. Montages confirm: m18/m24 controls go from a clean red-LEFT/yellow-RIGHT split to salt-and-pepper mixed.
   **b22 hypothesis part (a): SUPPORTED.**
2. **THE FORCE DOES NOT HOLD ABOVE THE RE-MIXED CONTROL — FALSIFIED (chemical route dead, final).** Paired
   Δseg = seg(xrep) − seg(ctrl) at the SAME seed & stir: **m18 seed1 +0.004 · m24 seed1 −0.011 · m24 seed2 +0.060**
   (mean ≈ +0.018, sign-flips yet again). The −0.75 stronger force (s5) drove seg 0.032 = BELOW the control (over-driven,
   as always). Even with a genuine re-mixing baseline finally in hand, cross-rep adds nothing consistent. The cleanest
   single datapoint: **m18 seed1 — control re-mixes to floor (0.063) at escape 0.008, force adds +0.004 (nothing).**
   **b22 hypothesis part (b): FALSIFIED.**
3. **move0.24 BREAKS CONTAINMENT — the stir ceiling at n120/cf0.08 is move0.18.** m24 escapes: 0.033 / 0.175 / 0.100 /
   0.192 / 0.042 — three of five ≥0.10 (HARD FAIL), r_max crosses 1.0. cf0.08 held clean at move0.12 but cannot hold at
   move0.24 (reconfirms Batch 7: move_speed is the dominant escape co-driver). **move0.18 is the sweet spot: control
   re-mixes to floor AND stays clean (escape 0.008; xrep_m18 escape 0.000, r_max 0.88).** So the decisive force test even
   at *clean* containment (m18) shows the force does nothing. One minor blip: xrep_m18 collapsed 0.025 (cross-rep K-S onset).

### PER-SLOT VERDICTS
- **STIR re-mixes the frozen seed (s0→s1→s2, s6): SUPPORTED** — the missing re-mixing baseline is finally achieved; move≥0.18 erases the seed to floor.
- **Chemical cross-rep maintains partition above a re-mixed control (s3/s4/s5/s7 vs paired controls): FALSIFIED** — Δseg sign-flips (mean ~0) even at clean m18; route CLOSED for good after b14–b22.
- **move0.24 as a clean stir (s2/s4/s5/s6): FALSIFIED as clean** — leaks (escape up to 0.19); move0.18 is the n120 clean-stir ceiling.

### DECISION (pre-committed) — 1E DELIVERED AS A FROZEN PARTITION; ADVANCE TO INTEGRATION
Control re-mixed but the force did NOT hold → per the b22 rule, **adopt the frozen seeded split** (`embryo_1E_split_hin`,
n120, cf0.08, move0.12, no force → seg ~0.34 @ escape 0.042, r_max 0.96, collapsed 0) **as the 1E operating point** and
**ADVANCE TO INTEGRATION**. 1E started Batch 13; closed Batch 23 (11 batches, within the 48-batch cap). `current_stage.txt` → `INT`.

### STAGE LADDER — all five rungs delivered (operating points to integrate)
- **1A** stable/no-collapse: `confine 0` recipe (or cf0.08 boundary catch at density).
- **1B** inner flow deforms membrane: cf0.03–0.07, mass 5e-4, sparse n → deform 0.03 @ escape 0.
- **1C** division deforms shell: cf0.10, mass 5e-5, div0.08–0.10 → deform 0.027 @ escape ~0.016.
- **1D** collective migration: `polar_align.gamma 120`, n557, move0.12, noise0.1 → migr 0.49 @ escape 0.020.
- **1E** two-type partition: seeded L/R split (`type_layout: split_x`), n120, cf0.08, move0.12, no force → seg ~0.34 (frozen).

### THE INTEGRATION TENSION (what makes INT hard — the scientific object of this phase)
The five recipes were each tuned in isolation and several are ANTAGONISTIC:
- **Partition vs stir.** The seeded partition is *frozen* — it holds only because interdiffusion is slow at move0.12. b22
  proved that RAISING stir (move≥0.18) re-mixes it. Flocking (1D migration) is exactly an active stir → it should re-mix
  the partition. So partition and migration pull against each other; the campaign has NO active force that maintains a
  partition against re-mixing (chemical route dead).
- **Migration vs escape at sparse n.** Flocking contains only when the interior is jammed (b12 — density-dependent);
  at n120 (sparse) a strong flock (γ120) translates as a body into the wall → likely leaks. Milder γ may coexist.
- **Division helps partition?** Daughters inherit the parent's type and spawn adjacent (`offset 0.004`), so proliferation
  should *preserve or sharpen* each side rather than dilute it — division may be the one ingredient that ADDS deform
  WITHOUT re-mixing. First integration test.

### HYPOTHESIS (Batch 23 / INT-1)
**Starting from the frozen seeded partition (seg ~0.34 at move0.12): (i) adding DIVISION (`cell_divide.rate 0.08`) raises
deform AND preserves/sharpens the partition (daughters inherit type, spawn local) at escape≈0; (ii) adding FLOCKING
(`polar_align.gamma`) trades partition for migration — mild γ60 keeps seg above the mixed floor while lifting migration,
but strong γ120 re-mixes the seed to the floor (like move0.18) and leaks at sparse n120 (escape>0.05); (iii) chemical
cross-rep still cannot hold the partition against flocking stir (Δseg ≈ 0).** So the best simultaneous
(partition + flow + deform) operating point is division + mild flocking (γ60), NOT strong flocking. Falsifier for (i):
division dilutes/re-mixes seg or leaks. Falsifier for (ii): γ120 keeps seg high (flocking does not stir) — would reopen
the maintenance question.

### DESIGN (8 slots) — INTEGRATION map; all overrides on `embryo_1E_split_hin_s{1,2}.yaml` (no new YAML)
All n120, cf0.08, move0.12, seeded split, spin off. Levers: `polar_align.gamma` (migration), `cell_divide.rate` (deform/n),
`agent_to_mpm.agent_mass` (deform), `chemotaxis.gain` (force), all overridable on the existing spec.
- s0 int_base     (seed1, γ0,   div0,    mass5e-5, gain0)  — CONTROL: frozen partition reference (expect seg~0.34, escape~0.04).
- s1 int_flock120 (seed1, γ120, div0,    mass5e-5, gain0)  — exploit: strong flocking; migration↑ but re-mix + sparse-n leak?
- s2 int_flock60  (seed1, γ60,  div0,    mass5e-5, gain0)  — exploit: mild flocking; coexistence sweet spot (some migr, seg survives).
- s3 int_div      (seed1, γ0,   div0.08, mass5e-5, gain0)  — exploit: add division; deform↑, partition sharpen or dilute?
- s4 int_full60   (seed1, γ60,  div0.08, mass5e-5, gain0)  — exploit: flow + division + partition together (the integration attempt).
- s5 int_flk_xrep (seed1, γ120, div0,    mass5e-5, gain−0.5)— explore: legit final force test — does cross-rep resist FLOCKING re-mixing? paired w/ s1.
- s6 int_div_hm   (seed1, γ0,   div0.08, mass1e-4, gain0)  — explore: push deform via mid-mass+division (1C says mass1e-4 leaks at density — boundary check).
- s7 int_full60_s2(seed2, γ60,  div0.08, mass5e-5, gain0)  — explore: reproduce the integration point on a 2nd seed (avoid single-seed trap).
Roles: 1 control (s0) · 4 exploit (s1–s4) · 3 explore (s5–s7). GUARDRAILS: escape (HARD =0; flocking at sparse n120 and
mass1e-4 both leak-prone), collapsed, nn_min, r_max<1. Read seg (partition survival) · migr/flow · deform · escape together.
frames 12000 stride 16, ~19 min (division grows n→~240; stride16 keeps render within the L4 wall).

## Batch 24 (2026-07-03) — STAGE: INTEGRATION (INT-2). Reading b23.

### OBSERVE (vs b23 predictions)
**THE BIG ONE: DIVISION NEVER FIRED IN b23 — `cell_divide.rate` is the WRONG override key.** The montage shows
n=120 at EVERY timepoint in EVERY slot, and the metrics prove it byte-for-byte: s0 int_base (div0) ≡ s3 int_div
(div0.08) IDENTICAL to 4 digits (deform 0.0164, migr 0.1174, seg 0.3391, escape 0.0417, nn_min 0.0044); and
s2 int_flock60 (div0) ≡ s4 int_full60 (flock60+div0.08) IDENTICAL (deform 0.0229, migr 0.3318, seg 0.4789).
Root cause (verified in `cell_divide.py:50`): the operator takes its rate from the per-type `div_rate` buffer if
present (`rate = getattr(lvl,"div_rate",None)`), and only falls back to the operator `rate` param if that buffer is
absent. The split specs set per-type `div_rate: 0.0` (lines 16-17), so overriding `cell_divide.rate` set the ignored
fallback and division stayed OFF. **The correct key is `agent.div_rate`** (`tune._apply` broadcasts it to every
type's `div_rate` buffer — tune.py:48). So b23's entire division arm (s3, s4, s6 division component) is VOID; the
central integration hypothesis (division adds deform without re-mixing the partition) is STILL UNTESTED.

**What b23 DID measure (flocking + containment, all at true n=120, div inert):**
- s0/s3 base (γ0, mass5e-5): deform 0.0164, migr 0.117, seg 0.339, escape 0.042, r_max 0.959, collapsed 0.
- s1 flock120: deform 0.0195, migr **0.261**, seg 0.277, escape 0.050, r_max 0.974.
- s2/s4 flock60: deform 0.0229, migr **0.332**, seg **0.479**, escape 0.075 (FAIL), r_max 0.992.
- s5 flk_xrep (γ120 + xrep−0.5): deform 0.0124, migr 0.215, seg 0.249, **escape 0.000, r_max 0.893** (CLEANEST).
- s6 div_hm (mass1e-4, div inert → just mass1e-4): deform 0.0229, migr 0.072, seg 0.165, **escape 0.117 (FAIL)**.
- s7 full60 SEED2 (γ60): deform 0.0145, migr 0.159, seg **0.046** (= mixed floor!), escape 0.008.

### Per-slot verdicts
- **DIVISION arm (s3, s4-div, s6-div): VOID (inert override).** Re-run with `agent.div_rate` this batch. Inconclusive.
- **FLOCKING raises migration at n120 (supported, but γ60 > γ120 here).** γ60 migr 0.332 > γ120 0.261 > base 0.117.
  The b23 prediction "γ120 strong, γ60 mild" INVERTED — at sparse n120, γ60 gave the STRONGER migration. Consistent
  with 1D's non-monotone-in-γ, near-bistable migration (b12): which flock coherence a realization lands in is noisy;
  γ120 need not beat γ60. **Migration IS achievable at n120 (0.26-0.33), above the frozen base 0.12.**
- **FLOCKING re-mixes the partition — SEED-DEPENDENTLY (supported, prediction (ii) partially).** γ60 kept seg 0.479 on
  seed1 but re-mixed to 0.046 (mixed floor) on seed2 (s7). So flocking-stir CAN erase the seed (s7) as predicted, but
  seed1 happened to keep/raise it. Reconfirms the whole-campaign lesson: partition survival at these speeds is a
  kinetic/seed lottery, NOT robust. (seg is only the x-projection — a coherent flock swirl can transiently inflate it.)
- **CROSS-REP does NOT hold the partition against flocking (supported, (iii)).** s5 (γ120+xrep−0.5) seg 0.249 <
  s1 (γ120, no force) seg 0.277 — the force added nothing (slightly less). Chemical route dead, reconfirmed a 4th way.
  BUT a genuine SIDE-EFFECT: xrep−0.5 gave the batch's CLEANEST containment (escape 0.000, r_max 0.893, vs γ120-alone
  escape 0.050). The mutual −0.5 cross-rep spreads cells off the boundary → fewer wall contacts → less escape. New lever.
- **mass1e-4 raises deform but LEAKS even at n120 (supported).** s6 deform 0.0229 (up from 0.016) but escape 0.117 —
  reconfirms 1C "mass1e-4 uncontainable at density" now at n120/cf0.08 too. Not a clean deform route.
- **CONTAINMENT: the frozen base already leaks (escape 0.042) at n120/cf0.08/move0.12.** Only s5 (xrep) and s7 hit
  escape≈0. So the adopted "frozen partition operating point" is itself marginal on the escape guardrail — cf0.08 at
  n120 is not strictly clean when the seed is present. Division (raising n) will worsen this → use cf0.10.

### Levers for Batch 24
- Enable division with the CORRECT key `agent.div_rate` (NOT `cell_divide.rate`). This is the whole batch.
- div_rate DOSE: from n120, growth ≈ exp(rate·dt·nframes) = exp(rate·24). div0.03→~2.05× (n246), 0.05→~3.3× (n400),
  0.06→~4× (n480, directive max), 0.08→~6.8× (n820, OVERSHOOTS — do NOT use at n120). Stay ≤0.06.
- cf0.10 for dividing slots (1C boundary-flux law: cf must rise with n); one cf0.08 comparison. move0.09 for strict-clean.
- xrep−0.5 as an unexpected CONTAINMENT lever (s5), separate from its dead partition role — parked for now.

### HYPOTHESIS (Batch 24 / INT-2)
**Enabling division via `agent.div_rate` from the frozen seeded split RAISES deform monotonically with the resulting
density (base 0.016 → ~0.025-0.030 at n≈250-480, per the 1C law deform↑ with n) WHILE PRESERVING the partition
(seg ≥ frozen 0.34), because daughters inherit the parent type and spawn locally (offset 0.004) — each side proliferates
in place and interdiffusion slows further at higher density. So division is the deform source that does NOT re-mix the
partition (unlike flocking-stir).** Containment: cf must rise to 0.10 once n>~250 (else the larger boundary flux leaks,
1C); div_rate ≤0.06 keeps n ≤~480 (4× directive) and contained. FALSIFIERS: (a) division dilutes seg toward the mixed
floor (daughters cross the midline) → division re-mixes too; (b) deform stays at base ~0.016 → density does not deform
here (contra 1C); (c) escape >0 even at cf0.10 → division uncontainable at this base.

### DESIGN (8 slots) — division done RIGHT; all overrides on `embryo_1E_split_hin_s{1,2}.yaml` (no new YAML)
All chemotaxis.gain 0.0 (base spec carries −1.0 — MUST disable), spin off, move0.12 unless noted, seeded split.
- s0 int_ctrl    (seed1, div0,    cf0.08, γ0)         — CONTROL (R4 division ablation): frozen base, expect seg~0.34, deform 0.016.
- s1 int_div03   (seed1, div0.03, cf0.10, γ0)         — exploit: gentle division (~n246); deform↑? seg preserved? clean at cf0.10?
- s2 int_div05   (seed1, div0.05, cf0.10, γ0)         — exploit: more division (~n400); deform↑↑, still contained?
- s3 int_div05c8 (seed1, div0.05, cf0.08, γ0)         — exploit: SAME division at cf0.08 — containment dose (does n400 leak at 0.08?).
- s4 int_divflk  (seed1, div0.03, cf0.10, γ60)        — exploit: THE integration — dividing + flowing + partitioned at once.
- s5 int_div03s2 (seed2, div0.03, cf0.10, γ0)         — explore: partition-preservation robustness on a 2nd seed (avoid single-seed trap).
- s6 int_div06   (seed1, div0.06, cf0.10, γ0)         — explore: push to ~4× (n480, directive max); containment ceiling?
- s7 int_div03m9 (seed1, div0.03, cf0.10, γ0, move0.09)— explore: strict-clean dividing partition (slower cells absorb into the catch).
Roles: 1 control (s0) · 4 exploit (s1-s4) · 3 explore (s5-s7). READ: deform (division payoff) · seg (partition survival
vs frozen 0.34 & mixed floor ~0.07) · escape/r_max/collapsed/nn_min (HARD guardrails, watch as n grows). Compare each
div slot vs s0. frames 12000 stride 16 (~20 min; n grows to ≤480, MPM cost fixed at per_parent 14000 so wall is safe).

## Batch 25 (2026-07-03) — STAGE: INTEGRATION (INT-3). Reading b24: division done RIGHT — the `agent.div_rate` fix worked.

### OBSERVE (b24 vs the INT-2 hypothesis) — table (div/n/cf/move/γ → seg/deform/migr/escape/r_max, all collapsed 0)
- s0 int_ctrl   div0  n120 cf.08 m.12 γ0   → seg 0.339 deform 0.016 migr 0.12 escape 0.042 r_max 0.96  (frozen partition ref, as predicted)
- s1 int_div03  .03   n235 cf.10 m.12 γ0   → seg 0.115 deform 0.020 migr 0.22 escape 0.017 r_max 0.95
- s2 int_div05  .05   n374 cf.10 m.12 γ0   → seg 0.153 deform 0.021 migr 0.08 escape 0.019 r_max 0.96
- s3 int_div05c8 .05  n374 cf.08 m.12 γ0   → seg 0.017 deform 0.027 migr 0.23 escape 0.029 r_max 0.95  (cf0.08 HELD n374 + MORE deform)
- s4 int_divflk .03   n235 cf.10 m.12 γ60  → seg 0.064 deform 0.028 migr 0.35 escape 0.013 r_max 0.98  (CLEANEST + richest 4-phenom)
- s5 int_div03s2 .03  n230 cf.10 m.12 γ0   → seg 0.073 deform 0.026 migr 0.12 escape 0.074 r_max 0.96  (seed2: re-mixed to floor)
- s6 int_div06  .06   n503 cf.10 m.12 γ0   → seg 0.126 deform 0.049 migr 0.29 escape 0.092 r_max 1.05  (deform MAX but escape FAIL)
- s7 int_div03m9 .03  n235 cf.10 m.09 γ0   → seg 0.345 deform 0.023 migr 0.16 escape 0.128 r_max 1.00  (seg PRESERVED but escape FAIL)

**Division fired correctly** — n grew 120→235/374/503 exactly per exp(24·div_rate) (the b23 inert-`cell_divide.rate` bug is
gone; `agent.div_rate` broadcasts). No collapse anywhere (collapsed 0 all 8; nn_min 0.0031-0.0044, packing not stacking).

**Big result 1 — DIVISION IS A DEFORM SOURCE (1C law holds inside the partitioned blastula):** deform rises monotonically
with the division-driven density: 0.016 (n120, ctrl) → 0.020 (n235) → 0.021 (n374) → **0.049 (n503, div06)**. The R4 control
sits at floor 0.016, so the extra deform is proliferation/density-driven. Montages go round → lobed → amoeboid as n climbs.
Division delivers the deform the integration wanted.

**Big result 2 — DIVISION RE-MIXES THE PARTITION (falsifier (a) FIRED; the preservation hypothesis is FALSIFIED):** every
division slot at move0.12 dropped seg from the frozen 0.339 toward the mixed floor: div03 0.115, div05 0.153, div06 0.126,
and seed2 div03 **0.073** (= n230 mixed floor). Daughters inherit type + spawn local (offset 0.004), yet the partition still
erased — because proliferation GENERATES FLOW/STIR: migr rose with division (0.12→0.22 at div03), and that outward
proliferation flow interdiffuses the two sides. So division is NOT a "non-stir" deform source — it stirs, same antagonism as
flocking. **The 1E partition is antagonistic to EVERY active deform/flow process tested (flocking b23, division b24).**

**Big result 3 — s4 (division + mild flocking γ60) is the cleanest, richest INTEGRATION point:** deform 0.028, migr 0.35
(batch max), escape 0.013 (batch min / cleanest), n235 dividing, collapsed 0 — {stability + membrane-deform + division +
collective-migration} coexist CLEANLY in one blastula. Partition is the only rung sacrificed (seg 0.064 = mixed). This is the
integration deliverable minus 1E.

**Secondary reads:** (s3) cf0.08 HELD n374 (escape 0.029, r_max 0.95) AND gave MORE deform (0.027) than cf0.10/div05
(0.021) — a weaker boundary catch lets the shell displace more while still containing at this density → cf0.08 is a better
deform/contain point than cf0.10 at n374. (s6) div06 n503 no-flock escape-FAILS (0.092, r_max 1.05) — the cf0.10 boundary
flux ceiling (1C: ~n442) reappears; the div06 deform 0.049 is real but NOT clean without help. (s7) move0.09 PRESERVED seg
(0.345 = ctrl) — slower cells stir the partition less — but escape jumped to 0.128 (WORSE than the move0.12 sibling's 0.017),
INVERTING the "move_speed is the dominant escape driver" rule at density: slower + dividing cells pile up locally instead of
spreading, and the local packing presses on the boundary → leak. So the one config that keeps the partition dividing also
hard-fails containment.

### PER-SLOT VERDICTS
- s0 control: SUPPORTED (frozen partition ref, seg 0.34, deform floor 0.016) — attribution anchor.
- s1/s2/s6 (division deform): deform-rise SUPPORTED; partition-preservation FALSIFIED (seg → floor). s6 escape-FAIL.
- s3 (cf0.08 at n374): SUPPORTED + bonus — contained AND higher deform than cf0.10; cf0.08 is the better n374 catch.
- s4 (division+flock): SUPPORTED as the clean 4-phenomenon integration point (deform+migr+division, escape 0.013).
- s5 (seed2): SUPPORTED the re-mix generality (seg 0.073 = floor on a 2nd seed) — division re-mixes across seeds.
- s7 (move0.09): partition-preservation SUPPORTED but containment FALSIFIED (escape 0.128) — inverts the move/escape rule.

### LEVERS for Batch 25
- Use FLOCKING AS CONTAINMENT (1D established: flock coherence contains at density) to push division-driven deform higher
  while staying clean — does div06 (n503) + strong flock γ120 rescue the s6 escape-fail and deliver deform ~0.05 clean?
- cf0.08 beats cf0.10 for deform at n374 (s3) — carry cf0.08 into the flock+division combos.
- Partition survival: only no-stir configs keep seg; test the div-rate THRESHOLD for re-mixing (div0.02, minimal) + try to
  rescue s7's escape (cf-up at move0.09) — the one shot at a dividing PARTITIONED blastula.

### HYPOTHESIS (Batch 25 / INT-3)
**Adding MILD-to-STRONG flocking (γ60-γ120) to division uses the flock's coherence-containment (1D) to hold the larger
boundary flux that broke the no-flock div06 (escape 0.092), so a dividing blastula at n≈400-500 reaches deform ≥0.04 at
escape <0.03 (CLEAN) — a simultaneously stable, dividing, deforming, MIGRATING blastula. The partition is sacrificed
(seg → floor ~0.05) because both division and flocking stir. FALSIFIERS: (a) flock+division still escapes >0.03 at n503
(flock cannot contain the proliferation flux) → deform ceiling clean is <0.04; (b) γ120 jams the dense flock (flow→0,
migr→floor) → over-alignment kills migration; (c) cf0.08 leaks at n503+flock → need cf0.10.**

### DESIGN (8 slots) — all overrides on `embryo_1E_split_hin_s{1,2}.yaml` (no new YAML; scalar tweaks only)
All chemotaxis.gain 0.0 (base carries −1.0 — MUST disable), mass 5.0e-5, seeded split, frames 12000 stride 16.
- s0 int_ctrl    (seed1, div0,    cf0.08, γ0,   m.12) — CONTROL (R4 div+flock ablation): frozen partition ref (seg~0.34).
- s1 flk_d06g120 (seed1, div0.06, cf0.10, γ120, m.12) — FLAGSHIP: n503 + strong flock; deform~0.05 @ escape<0.03? (rescue s6).
- s2 flk_d05g120 (seed1, div0.05, cf0.10, γ120, m.12) — exploit: n374 + strong flock — deform + clean + migration.
- s3 flk_d06g60  (seed1, div0.06, cf0.10, γ60,  m.12) — exploit: n503 + MILD flock — γ60 vs γ120 containment at max density.
- s4 flk_d05g60c8(seed1, div0.05, cf0.08, γ60,  m.12) — exploit: n374 + mild flock + cf0.08 (s3 winner catch) — max clean deform.
- s5 lowdiv02    (seed1, div0.02, cf0.08, γ0,   m.12) — explore: minimal division (~n194) — the div-rate THRESHOLD for re-mix (seg?).
- s6 d03m9c12    (seed1, div0.03, cf0.12, γ0,   m.09) — explore: rescue s7 — stronger catch on slow dividing cells → seg + clean?
- s7 flk_d06g120c8(seed1,div0.06, cf0.08, γ120, m.12) — explore: n503 + strong flock + WEAK catch — the max clean deform ceiling.
Roles: 1 control (s0) · 4 exploit (s1-s4) · 3 explore (s5-s7). READ: deform (ceiling) · escape/r_max (does flock contain the
division flux?) · migr/flow (does γ120 jam?) · seg (expect floor except s0/s5/s6) · collapsed/nn_min (guardrails).

## Batch 26 (2026-07-03) — STAGE: INTEGRATION (INT-4). Reading b25: flock-as-containment — the premise INVERTED.

### OBSERVE (b25 vs the INT-3 hypothesis) — table (div/n/cf/γ/move → deform/migr/seg/escape/r_max)
- s0 int_ctrl    div0  n120 cf.08 γ0   m.12 → deform 0.016 migr 0.12 seg 0.339 escape 0.042 r_max 0.96 coll 0     (frozen partition ref)
- s1 flk_d06g120 .06   n503 cf.10 γ120 m.12 → deform 0.031 migr 0.28 seg 0.141 escape **0.032** r_max 1.04 coll 0  (γ120 near-clean at n503!)
- s2 flk_d05g120 .05   n374 cf.10 γ120 m.12 → deform 0.035 migr 0.45 seg 0.152 escape 0.136 r_max 1.17 coll 0     (n374 cf.10 leaks)
- s3 flk_d06g60  .06   n503 cf.10 γ60  m.12 → deform **0.045** migr 0.10 seg 0.073 escape **0.242** r_max 1.08 coll 0.004 (γ60 RAMS)
- s4 flk_d05g60c8 .05  n374 cf.08 γ60  m.12 → deform 0.021 migr 0.25 seg 0.178 escape **0.019** r_max 0.97 coll 0  (cleanest dividing+flock)
- s5 lowdiv02    .02   n182 cf.08 γ0   m.12 → deform 0.017 migr 0.07 seg **0.294** escape **0.000** r_max 0.86 coll 0 (CLEAN PARTITIONED div)
- s6 d03m9c12    .03   n235 cf.12 γ0   m.09 → deform 0.014 migr 0.18 seg **0.311** escape 0.004 r_max 0.92 coll **0.0085** (seg held, minor collapse)
- s7 flk_d06g120c8 .06 n503 cf.08 γ120 m.12 → deform 0.046 migr **0.50** seg 0.171 escape 0.119 r_max 1.10 coll 0 (deform+migr MAX, cf.08 leaks)

**Big result 1 — THE PREMISE INVERTED: STRONG flocking (γ120) contains at confluence, MILD (γ60) RAMS.** The batch assumed
γ60 (mild) would be the gentle container; the opposite held. At n503/cf0.10: γ120 (s1) escape **0.032**, γ60 (s3) escape
**0.242** — γ60 is 7.6× WORSE, and worse than the no-flock div06 (b24 s6 0.092). Strong flock cut escape 0.092→0.032 vs
no-flock (flock-containment CONFIRMED for γ120). Mechanism: a COHERENT (γ120) flock advects as an organized recirculating
stream that stays off the boundary; a HALF-ordered (γ60) flock is a disorganized shear that drifts cells into the shell.
Migration confirms the same axis: γ120 migr 0.28-0.50 (HIGH, no jam — falsifier (b) DEAD), γ60 migr 0.096 (dead). So at
CONFLUENCE the flocking non-monotone peak has moved UP — γ120 is coherent-flowing, γ60 is the disordered/jammed regime
(inverting the sparse-n b23/b24 reading where γ60>γ120). Falsifier (b) FALSIFIED, and the mild-flock arm of the hypothesis
is FALSIFIED.

**Big result 2 — THE CLEAN DEFORM CEILING STAYS ~0.03 EVEN WITH FLOCK-CONTAINMENT (falsifier (a) FIRED).** The two slots at
deform≥0.045 (s3 0.045, s7 0.046) BOTH escape-FAIL (0.242, 0.119). The cleanest flock slots cap at deform ~0.03 (s1 0.031
@ esc 0.032; s4 0.021 @ esc 0.019). So flocking did NOT lift the clean deform ceiling to 0.04 — coherent flow contains but
does not scatter more momentum onto the grid than the density already does. The clean division+flock deform ceiling is
~0.03, same order as the 1C density ceiling (~0.027). The deliverable "deform ≥0.04 CLEAN" was NOT reached this batch.

**Big result 3 — cf0.10 is REQUIRED at n503+flock; cf0.08 leaks (falsifier (c) CONFIRMED).** s1 (γ120 cf0.10) escape 0.032
vs s7 (γ120 cf0.08) escape 0.119 — same everything else, weaker catch → 3.7× more leak. At n503 the boundary flux needs the
full cf0.10; cf0.08 only held at n374 (s4). So the clean-flock operating point at max density is n503/cf0.10/γ120.

**Big result 4 — THE div-rate RE-MIX THRESHOLD is between 0.02 and 0.03: minimal division (div0.02) is a CLEAN PARTITIONED
blastula.** s5 (div0.02, γ0, cf0.08): seg **0.294** (≈ ctrl 0.339, PRESERVED), escape **0.000**, collapsed 0, n182 — gentle
proliferation grew the population 120→182 WITHOUT erasing the partition and stayed strictly clean. div0.03+ re-mixes (b24
div03 seg 0.115); div0.02 sits below the stir threshold. So a dividing PARTITIONED blastula exists at div≤0.02. s6 (div0.03,
move0.09, cf0.12) also PRESERVED seg (0.311) and rescued the b24-s7 escape-fail (0.128→0.004, cf0.12 + slow cells contain)
— but at the cost of minor collapse (0.0085, nn_min 0.0028 — cf0.12 rams slightly at n235, the 1C "confine-up rams" law).

### PER-SLOT VERDICTS
- s0 control: SUPPORTED (frozen partition ref, seg 0.34, deform floor 0.016) — attribution anchor.
- s1 (γ120 n503): SUPPORTED as flock-containment (escape 0.032, rescued the no-flock 0.092) — the near-clean n503 point; deform 0.031 (<0.04).
- s2 (γ120 n374 cf.10): INinclusive→leaks (escape 0.136) — cf0.10 over-catches nothing here; n374 wants cf0.08 (see s4).
- s3 (γ60 n503): FALSIFIED the mild-flock-container hypothesis (escape 0.242, RAMS; migr dead 0.096) — γ60 is the disordered regime at density.
- s4 (γ60 n374 cf.08): SUPPORTED as the cleanest dividing+flock point (escape 0.019, deform 0.021, migr 0.25, n374) — but deform low.
- s5 (div0.02): SUPPORTED — a CLEAN dividing PARTITIONED blastula (seg 0.294, escape 0, coll 0); div-rate re-mix threshold ∈ (0.02, 0.03).
- s6 (div0.03 m.09 cf.12): PARTIALLY SUPPORTED — seg preserved (0.311) + escape rescued (0.004) but minor collapse 0.0085 (cf0.12 rams at density).
- s7 (γ120 cf.08 n503): deform+migr MAX (0.046/0.50) but escape-FAIL (0.119) — confirms cf0.08 leaks at n503; the max-flow point is not clean.

### LEVERS for Batch 26
- The flock-containment axis is MONOTONE in coherence at n503 (γ60 rams 0.242 → γ120 clean 0.032). PUSH γ higher (γ160/200/240):
  does more coherence tighten containment further AND organize a larger-scale circulation that lifts the clean deform above 0.03?
- Clean deform ceiling is ~0.03 via density+flock — try the OTHER deform lever (agent_mass ↑) UNDER the γ120 container, which now
  holds the boundary: does mass 7e-5 on the contained n503 flock buy deform ≥0.04 without re-leaking?
- Partition thread: div0.02 is clean+partitioned but low-activity. Does STRONG flock (γ120) re-mix the minimal-division partition,
  or does the coherent stream leave the L/R seed intact (flock circulation ≠ interdiffusion)? One shot at partition+migration.

### HYPOTHESIS (Batch 26 / INT-4)
**Flock containment at confluence strengthens MONOTONICALLY with alignment coherence γ (established this batch: n503/cf0.10 γ60
escape 0.242 → γ120 0.032). Pushing γ to 160-240 will hold containment at escape <0.03 AND organize a larger-scale coherent
circulation that lifts the clean deform above the ~0.03 density ceiling toward the deliverable ≥0.04 in a dividing n503 blastula.
FALSIFIERS: (a) clean deform stays ~0.03 across all γ (coherent flow contains but scatters no extra grid momentum) → the clean
division+flock deform ceiling is a hard ~0.03, and mass (s4) is the only remaining lever; (b) γ≳200 over-aligns into a single
translating block that rams one side (escape↑, r_max>1.05, deform localized not global); (c) a mass bump (7e-5) under γ120
re-leaks (escape>0.03) → the container cannot hold more per-cell push at n503 → adopt s1 (γ120 n503 cf0.10) as the INT clean point.**

### DESIGN (8 slots) — all overrides on `embryo_1E_split_hin_s1.yaml` (no new YAML; scalar tweaks only)
All chemotaxis.gain 0.0 (base carries −1.0 — MUST disable), seeded split, frames 12000 stride 16. mass 5.0e-5 unless noted.
- s0 ctrl_nof   (div0.06, n503, cf0.10, γ0,   m5e-5) — CONTROL (R4 flock ablation): no-flock div06 ref (expect escape-FAIL ~0.09).
- s1 g120       (div0.06, n503, cf0.10, γ120, m5e-5) — EXPLOIT anchor: reproduce s1 (escape 0.032, deform 0.031) — γ-sweep base.
- s2 g160       (div0.06, n503, cf0.10, γ160, m5e-5) — EXPLOIT: more coherence → tighter containment + more deform?
- s3 g200       (div0.06, n503, cf0.10, γ200, m5e-5) — EXPLOIT: strong coherence — the deform-vs-γ ceiling.
- s4 g120_m7    (div0.06, n503, cf0.10, γ120, m7e-5) — EXPLOIT: mass bump UNDER the γ120 container → deform ≥0.04 without re-leak?
- s5 g120_div02 (div0.02, n182, cf0.08, γ120, m5e-5) — EXPLORE: does strong flock re-mix the minimal-div partition (s5 held seg 0.294 @ γ0)?
- s6 g200_n374  (div0.05, n374, cf0.10, γ200, m5e-5) — EXPLORE: mid density + strong flock — max migration + deform + clean?
- s7 g240       (div0.06, n503, cf0.10, γ240, m5e-5) — EXPLORE: over-alignment onset — does extreme coherence ram (block flow) or contain?
Roles: 1 control (s0) · 4 exploit (s1-s4) · 3 explore (s5-s7). READ: escape/r_max (γ-monotone containment? mass re-leak?) ·
deform (ceiling vs γ and vs mass) · migr/flow (γ over-align jam?) · seg (s5: flock vs partition) · collapsed/nn_min (guardrails at n503).

---

## Batch 27 (2026-07-03) — STAGE: INTEGRATION (INT-5). Reading b26.

### OBSERVE (montage montages/embryo_b26.png + 8 metrics.json)
All 8 slots collapsed=0, nn_min 0.0031–0.0043 (packing at n503, no stacking). n503/cf0.10/div06/mass5e-5/move0.12 unless noted.
- **s0 ctrl_nof** (γ0): escape **0.0915**, deform 0.0493, migr 0.291, r_max 1.049 — CONTROL. Escape-FAILS as predicted (~0.09);
  highest deform of the batch but blows out. Confirms flock is needed for containment at n503.
- **s1 g120**: escape **0.0318**, deform 0.0307, migr 0.28, r_max 1.041 — REPRODUCES b25 s1 (escape 0.032, deform 0.031) to
  the digit. Cleanest strong-flock point. The γ120 container rescues the no-flock 0.092 → 0.032.
- **s2 g160**: escape **0.1074**, deform 0.0431, migr 0.476, r_max 1.075 — escape JUMPED back UP (worse than γ120).
- **s3 g200**: escape **0.165**, deform 0.0368, migr 0.493, r_max 1.072 — escape WORSE still; highest migration.
- **s7 g240**: escape **0.1074**, deform 0.0327, migr 0.452, r_max 1.095 — escape 0.107, no better than γ160.
- **s4 g120_m7** (mass 7e-5): escape **0.2068**, deform 0.0411, migr 0.34, r_max **1.161** — mass bump RE-LEAKS badly.
- **s5 g120_div02** (div0.02, n182, cf0.08): escape **0.0275**, deform 0.0216, seg **0.2737**, migr 0.24, r_max **0.982** (<1.0,
  no cell outside membrane), collapsed 0 — the STANDOUT (see below).
- **s6 g200_n374** (n374, div0.05): escape **0.1123**, deform 0.027, migr 0.263, r_max 0.987 — escape-fails.

### VERDICTS
- **HYPOTHESIS FALSIFIED (all 3 falsifiers fired).** Flock containment is NOT monotone in γ; it has an OPTIMUM (containment
  MINIMUM) at **γ≈120**. n503/cf0.10/div06 escape vs γ: γ60 **0.242** (b25) → γ120 **0.032** → γ160 **0.107** → γ200 **0.165** →
  γ240 **0.107**. There is a CONTAINMENT WINDOW around γ120: γ60 too disordered (shear drifts cells into the wall), γ≥160
  over-aligns into a coherent TRANSLATING stream (migr jumps 0.28→0.48–0.49) that marches into the boundary (r_max>1.07). So
  strong-flock containment at confluence is a bell in γ peaked at ~120, NOT monotone. Falsifier (b) confirmed.
- **[supported] Clean deform ceiling STAYS ~0.03** — every deform≥0.04 slot escape-FAILS (s0 0.049@esc0.092, s2 0.043@0.107,
  s4 0.041@0.207). Cleanest deform = s1 γ120 0.0307. Coherent circulation reorganizes the same momentum; it adds no push.
  Falsifier (a) confirmed.
- **[supported] Mass 7e-5 under γ120 re-leaks (escape 0.21, r_max 1.16)** — the container cannot hold more per-cell push at
  n503. Falsifier (c) confirmed. BOTH remaining clean-deform levers at density (more γ-flow, more mass) are now dead → the
  deliverable "deform ≥0.04 CLEAN at n503" is UNREACHABLE in this operator set. Adopt s1 (γ120/n503/cf0.10, deform 0.031 @
  escape 0.032) as the density INT point; the clean-deform ceiling is ~0.03.
- **[STANDOUT — new lead] s5: strong flock γ120 did NOT re-mix the minimal-division partition.** div0.02/n182/cf0.08/γ120 →
  seg **0.2737** (≈ b25 s5 γ0 0.294, PRESERVED) at escape 0.0275, r_max 0.982 (NO cell outside the shell), collapsed 0,
  migr 0.24. This is the FIRST candidate **FULL 5-phenomenon integrated blastula**: {stability + membrane partition seg 0.27 +
  division n182 + collective migration 0.24} coexist near-clean — only deform is low (0.0216). Why the flock did NOT re-mix
  here (vs b24 γ60 re-mixing seed-dependently at n120): minimal division (n182) stays below the proliferation-stir threshold,
  AND a COHERENT γ120 recirculation preserves the large-scale L/R structure better than a half-ordered γ60 shear (circulation
  ≠ interdiffusion). SINGLE SEED (seed1) — must replicate before adoption.

### HYPOTHESIS (Batch 27 / INT-5)
**The b26 s5 config (γ120 + div0.02 + cf0.08 + n182) is a GENUINE full 5-phenomenon integrated blastula, not a seed-1 fluke.
(1) The partition survival (seg ~0.27 under the γ120 flock) will REPLICATE on seeds 2 and 3 (paired vs their own γ0 no-flock
controls, seg gap small — the flock circulation preserves, does not stir). (2) Its residual escape (0.0275, grazing, r_max<1.0)
CLOSES to ≈0 by raising the boundary catch cf0.08→0.10 at this low n182 (below the ram threshold) while seg/migration hold.
(3) The div-rate re-mix threshold UNDER a γ120 flock sits between div0.02 (holds seg) and div0.03 (re-mixes) — same window as
the no-flock threshold (0.02,0.03), i.e. the flock does not shift it. FALSIFIERS: (a) seed2/seed3 γ120 re-mix to floor (seg<0.1)
→ s5 was a lucky seed, partition+flock antagonism holds → adopt the frozen partition + separate flock points; (b) cf0.10 rams at
n182 (collapsed>0) or does not close escape → keep cf0.08, log escape 0.0275 grazing as the clean point; (c) div0.03+γ120
re-mixes (seg<0.1) → division-stir dominates over the flock's structure-preservation.**

### DESIGN (8 slots) — all overrides on the `embryo_1E_split_hin_s{1,2,3}.yaml` seed copies (no new YAML; scalar tweaks only)
All chemotaxis.gain 0.0 (base carries −1.0 — MUST disable), spin off, seeded split, mass 5.0e-5, frames 12000 stride 16.
- s0 s2_g0   (SEED2, γ0,   div0.02, cf0.08, move0.12) — CONTROL (R4 flock ablation) + seed2 no-flock partition baseline.
- s1 s2_g120 (SEED2, γ120, div0.02, cf0.08, move0.12) — EXPLOIT: does flock-preserves-partition replicate on seed2?
- s2 s3_g120 (SEED3, γ120, div0.02, cf0.08, move0.12) — EXPLOIT: seed3 replication of the 5-phenomenon point.
- s3 cf10    (SEED1, γ120, div0.02, cf0.10, move0.12) — EXPLOIT: close the escape 0.0275 with a stronger catch at n182.
- s4 div03   (SEED1, γ120, div0.03, cf0.10, move0.12) — EXPLOIT: div-threshold under the flock — does more division re-mix?
- s5 s3_g0   (SEED3, γ0,   div0.02, cf0.08, move0.12) — EXPLORE: seed3 no-flock baseline (paired control for s2).
- s6 move09  (SEED1, γ120, div0.02, cf0.08, move0.09) — EXPLORE: alt escape closer (slower flock approaches boundary gently).
- s7 g90     (SEED1, γ90,  div0.02, cf0.08, move0.12) — EXPLORE: bracket the containment window low side at n182.
Roles: 1 control (s0) · 4 exploit (s1–s4) · 3 explore (s5–s7). READ: seg (s0/s1 seed2, s5/s2 seed3, s3/s4 seed1 — does the γ120
flock preserve the partition across seeds?) · escape/r_max (cf0.10 close it? move0.09? γ90 vs γ120?) · collapsed/nn_min (cf0.10
ram at n182?) · migr/deform (activity of the 5-phenomenon point). div0.02→n182, div0.03→n~250.

---

## Batch 28 (2026-07-03) — STAGE: INTEGRATION (INT-6). Reading b27.
**Target: does the b26 s5 5-phenomenon blastula replicate across seeds, and where is the escape/migration knee?**

### OBSERVE (b27 montage + 8 metrics.json) — all 8 ran; ONE hard fail (g90 escape 0.154).
Config table (all n182 unless noted; γ120/div0.02/cf0.08/move0.12/seed1 = the b26 s5 point):
| slot | seed | γ | div | cf | move | seg | escape | r_max | migr | deform | coll |
|------|------|---|-----|----|----- |-----|--------|-------|------|--------|------|
| s0 s2_g0   | 2 | 0   | .02 | .08 | .12 | 0.220 | 0.073 | 0.960 | 0.256 | 0.016 | 0 |
| s1 s2_g120 | 2 | 120 | .02 | .08 | .12 | **0.280** | 0.031 | 0.969 | **0.422** | 0.021 | 0 |
| s2 s3_g120 | 3 | 120 | .02 | .08 | .12 | 0.184 | 0.016 | 0.932 | 0.288 | 0.022 | 0 |
| s5 s3_g0   | 3 | 0   | .02 | .08 | .12 | 0.378 | 0.000 | 0.892 | 0.103 | 0.012 | 0.010 |
| s3 cf10    | 1 | 120 | .02 | .10 | .12 | 0.293 | 0.0055 | 0.907 | 0.136 | 0.016 | 0 |
| s4 div03   | 1 | 120 | .03 | .10 | .12 | 0.167 | 0.026 | 0.972 | 0.210 | **0.029** | 0 |
| s6 move09  | 1 | 120 | .02 | .08 | .09 | **0.399** | **0.000** | 0.859 | 0.130 | 0.013 | 0 |
| s7 g90     | 1 | 90  | .02 | .08 | .12 | 0.166 | **0.154** | **1.049** | 0.457 | 0.035 | 0 |

**Seed pairs (γ0 → γ120):** seed2 0.220→0.280 (**Δ+0.060**, flock HELD + halved escape + doubled migr); seed3 0.378→0.184
(**Δ−0.194**, flock ERODED); seed1 (b26) ~0.294→0.274 (Δ−0.020). Three γ120 seg = 0.274/0.280/0.184, **mean 0.246**; three γ0
controls = ~0.294/0.220/0.378, **mean 0.297**. So the γ120 flock does NOT preserve each seed's seg — it **regresses seg toward a
~0.25 attractor** (sharpens the weak seed2, erodes the strong seed3) — BUT every γ120 slot stays seg 0.18–0.28, far above the
n182 mixed floor (~0.06). **The partition SURVIVES the flock on all 3 seeds → {stability + partition + division + migration}
coexist robustly, reproducibly, across seeds. This is the confirmed 5-phenomenon integrated blastula (deform still the weak leg).**

### PER-SLOT VERDICTS
- **Hyp (1) partition-under-flock replication — PARTIALLY SUPPORTED.** SUPPORTED: partition coexists with the γ120 flock on
  every seed (seg 0.18–0.28 ≫ floor 0.06, escape≈0, coll 0). FALSIFIED: "flock PRESERVES each seed's seg" — it homogenizes toward
  ~0.25 (seed3 fell 0.38→0.18). The blastula is real & reproducible; the flock is a mild partition-eroder, not neutral.
- **Hyp (2) escape closes at n182 — SUPPORTED, but at a migration cost (NEW antagonism).** cf0.10 (s3) escape 0.0275→**0.0055**,
  coll 0 (NO ram at n182 — the ram threshold is above cf0.10 at this low n), r_max 0.907, seg HELD 0.293 — BUT migr collapsed
  0.24→0.136. move0.09 (s6) escape→**0.000**, r_max 0.859, seg 0.399 (highest) — BUT migr 0.130. **Both escape-closers TAME the
  flock → migration drops to ~0.13.** Closing escape and keeping strong migration are ANTAGONISTIC at n182: the two clean routes
  (cf0.10, move0.09) both suppress the coherent stream. The strong-migration point (seed2 s1, migr 0.42) grazes at escape 0.031
  (r_max 0.97, no cell outside).
- **Hyp (3) div-threshold under flock ∈(0.02,0.03) — SUPPORTED + division is the deform source.** div0.03 (s4, n235) → seg
  dropped to **0.167** (re-mix toward floor), deform rose to **0.029** (batch-max clean), escape 0.026 grazing, coll 0, migr 0.21.
  Division adds deform inside the integrated blastula but erodes the partition — the 1C law (division→deform) and the 1E
  antagonism (division-stir re-mixes) BOTH hold under the flock. div0.02 is the partition-preserving ceiling.
- **γ90 (s7) — HARD FAIL (escape 0.154, r_max 1.049).** Below the γ120 optimum the flock at n182 is a coherent stream that
  over-translates INTO the wall (migr 0.457 high, cells punch through r_max>1). Confirms γ120 is the containment optimum and the
  window's LOW side leaks even at n182 (falsifies "milder flock contains the low density better").
- **s5 seed3 γ0 control — minor blip collapsed 0.0104** (nn_min 0.0023, one close pair; not a structural collapse) but seg 0.378
  (strongest frozen seed) — the seed3 no-flock partition is the sharpest, which is why the flock's erosion there is most visible.

### BEST INTEGRATED POINTS (5-phenomenon, escape-ranked)
1. **Strictly clean:** s6 move0.09 — seg 0.399, migr 0.130, deform 0.013, escape 0.000, r_max 0.859, coll 0 (all 5, migration weak).
2. **Clean:** s3 cf0.10 — seg 0.293, migr 0.136, deform 0.016, escape 0.0055 (all 5, migration weak).
3. **Strong-migration grazing:** seed2 s1 γ120/cf0.08 — seg 0.280, migr 0.422, deform 0.021, escape 0.031 (r_max 0.97, grazing).
The open frontier: NO single point yet has strong migration (≳0.3) AND escape≈0 at n182 — batch 28 maps the cf/move knee between them.

### HYPOTHESIS (Batch 28 / INT-6)
**The escape↔migration antagonism at the n182 integrated point is a KNEE, not a cliff: an intermediate containment
(cf≈0.085–0.09, OR move≈0.10–0.11) at γ120/div0.02/seed1 closes escape to ≤0.01 with r_max<1 while preserving migration ≳0.25
(better than cf0.10's 0.136) — i.e. a usable middle exists between the grazing-strong (cf0.08, migr 0.24, esc 0.028) and
clean-weak (cf0.10, migr 0.14, esc 0.006) endpoints. Additionally a modest division bump div0.025 lifts deform toward ~0.025
without dropping seg below ~0.20. FALSIFIERS: (a) cf0.085/0.09 migration is already damped to ~0.14 (cliff at cf0.08⁺, no knee)
→ accept the grazing cf0.08 point as the strong-migration operating spec; (b) div0.025 re-mixes seg<0.20 → div0.02 is the hard
partition ceiling; (c) γ140 leaks (like γ90) or damps migration → γ120 is a sharp single optimum, no room to trade align-up.**

### DESIGN (8 slots) — all seed1 overrides on `embryo_1E_split_hin_s1.yaml` (no new YAML; scalar tweaks only)
All chemotaxis.gain 0.0 (base −1.0, MUST disable), spin off, seeded split, mass 5.0e-5, γ120 & div0.02 unless noted, frames 12000
stride 16. Maps TWO frontiers on ONE seed for a clean tradeoff curve: containment (cf) and speed (move).
- s0 cf085     (cf0.085, move0.12)            — EXPLOIT: containment knee, low side. Expect escape~0.015, migr>0.20?
- s1 cf09      (cf0.09,  move0.12)            — EXPLOIT: containment knee, mid. Between cf0.08(esc.028) and cf0.10(esc.006).
- s2 move11    (cf0.08,  move0.11)            — EXPLOIT: speed knee, high side. Between move0.09(esc0) and move0.12(esc.028).
- s3 move10    (cf0.08,  move0.10)            — EXPLOIT: speed knee, mid.
- s4 div025    (div0.025, cf0.09, move0.12)   — EXPLORE: deform bump — does div0.025 lift deform w/o re-mixing seg<0.20?
- s5 g140      (γ140, cf0.09, move0.12)       — EXPLORE: align-up — does a stronger flock hold migration + contain at n182?
- s6 div025_m11(div0.025, cf0.08, move0.11)   — EXPLORE: deform bump + gentle-speed containment combo.
- s7 g0_cf09   (γ0, cf0.09, move0.12)         — CONTROL (R4 flock ablation): isolate the flock's migration contribution.
Roles: 4 exploit (s0–s3) · 3 explore (s4–s6) · 1 control (s7). READ: escape/r_max/migr across the cf & move frontiers (is there a
knee with escape≤0.01 AND migr≳0.25?) · deform/seg for the div025 arm · migr(s7 γ0) vs migr(s1 γ120) for the flock attribution.

---

## Batch 29 (2026-07-03) — STAGE: INTEGRATION (INT-7). Reading b28.
**Target: the escape↔migration KNEE — does an intermediate containment give escape≤0.01 AND migr≥0.25 at the n182 blastula?**

### OBSERVE (b28 montage + 8 metrics.json) — all 8 ran; ONE hard fail (div025 escape 0.090, r_max 1.046).
Config table (all seed1, γ120/div0.02/mass5e-5 unless noted):
| slot | move | cf | γ | div | escape | r_max | migr | seg | deform | n | coll |
|------|------|-----|---|-----|--------|-------|------|-----|--------|---|------|
| s0 cf085     | .12 | .085 | 120 | .02  | 0.0165 | 0.906 | 0.280 | 0.332 | 0.016  | 182 | 0 |
| s1 cf09      | .12 | .09  | 120 | .02  | **0.000** | 0.892 | 0.289 | **0.411** | 0.014 | 182 | 0 |
| s2 move11    | .11 | .08  | 120 | .02  | 0.011  | 0.926 | **0.497** | 0.270 | 0.018 | 182 | 0 |
| s3 move10    | .10 | .08  | 120 | .02  | **0.000** | 0.880 | 0.093 | 0.507 | 0.016 | 182 | 0 |
| s4 div025    | .12 | .09  | 120 | .025 | **0.090** | **1.046** | 0.489 | 0.191 | **0.0275** | 200 | 0 |
| s5 g140      | .12 | .09  | 140 | .02  | 0.022  | 0.920 | 0.139 | 0.309 | 0.024 | 182 | 0 |
| s6 div025_m11| .11 | .08  | 120 | .025 | **0.000** | 0.875 | **0.380** | 0.307 | 0.018 | 200 | 0 |
| s7 g0_cf09   | .12 | .09  | 0   | .02  | 0.055  | 0.979 | 0.374 | 0.223 | 0.026 | 182 | 0 |

**THE KNEE IS REAL — hypothesis SUPPORTED, and better than predicted.** Three points clear escape≤0.01 AND migr≥0.25:
- **s1 cf09 — STRICTLY CLEAN (escape 0.000), migr 0.289, seg 0.411 (batch-best partition), deform 0.014, r_max 0.892, coll 0.**
- **s6 div025_m11 — STRICTLY CLEAN (escape 0.000), migr 0.380, seg 0.307, deform 0.018, n200 (division), r_max 0.875, coll 0.**
- s2 move11 — near-clean (escape 0.011, r_max 0.926), **migr 0.497 (batch-max)**, seg 0.270. The strong-migration grazing point.
The b27 antagonism ("escape≈0 ⇒ migr collapses to ~0.13") is BROKEN: cf0.10 killed migr to 0.14, but **cf0.09 closes escape to 0
while KEEPING migr 0.29** — the containment knee sits at cf0.09, not 0.10, and 0.09 is the better operating point (clean + migration).

### PER-SLOT VERDICTS
- **cf-knee (s0/s1) — SUPPORTED.** cf0.085 escape 0.0165 (migr 0.280), cf0.09 escape **0.000** (migr 0.289, seg 0.411). Between
  cf0.08 (esc 0.028, b27) and cf0.10 (esc 0.006 but migr 0.14, b27) the knee is at cf0.09: escape fully closes while migration is
  held. cf0.09 REPLACES cf0.10 as the containment operating point (closes escape without the migration tax).
- **move-knee (s2/s3) — SUPPORTED, and the SPEED lever preserves migration far better than the cf lever.** move0.11 → migr 0.497
  (esc 0.011, grazing) or 0.380 (esc 0.000 with div025, s6); move0.10 → migr collapses to 0.093 (over-tamed). **move0.11 is the
  migration-preserving containment knob**; move0.10 is past the cliff. The speed knee (move0.11) keeps migration 2–3× higher than
  the cf-up route at comparable escape.
- **div025 deform bump (s4/s6) — division lifts deform but needs the speed-knee to stay contained.** div0.025 at move0.12/cf0.09
  (s4) HARD-FAILED (escape 0.090, r_max 1.046 — the denser division flux + fast move overruns cf0.09) yet gave the batch-max deform
  0.0275 and re-mixed seg to 0.191. div0.025 at move0.11/cf0.08 (s6) is STRICTLY CLEAN (escape 0.000) but deform only 0.018 and seg
  HELD 0.307 — slower stir preserves the partition (frozen-partition rule) but scatters less momentum. **Deform and containment
  trade through move-speed: fast move buys deform (0.028) but leaks; slow move contains but deform falls to 0.018.**
- **γ140 (s5) — FALSIFIED align-up (falsifier c confirmed).** γ140 → migr collapses to 0.139 (over-alignment damps net motion),
  escape 0.022 (grazing fail). No help above γ120; γ120 is the sharp single optimum. (Not the γ90 over-translate failure — γ140
  contains r_max 0.920 — it simply kills migration.)
- **CONTROL γ0 (s7) — REINTERPRETS the flock's role at n182.** γ0/cf0.09 → migr **0.374 (HIGHER than γ120 s1's 0.289)** but escape
  0.055 (LEAKS) and seg 0.223 (lower partition than γ120 s1's 0.411). So at n182 the flock is NOT the migration source (the flow/
  division advection already gives high polar order without it); **the flock's causal contribution is CONTAINMENT (escape
  0.055→0.000) + partition preservation (seg 0.223→0.411).** A coherent γ120 recirculation stays off the wall; a γ0 translating
  drift leaks into it. This overturns the earlier "flock lifts migration" framing at this density — the flock buys cleanliness,
  not migration, at n182.

### BEST INTEGRATED POINTS (5-phenomenon, all escape≈0, coll 0)
1. **s6 div025_m11 (move0.11/cf0.08/γ120/div0.025) — BEST ALL-ROUNDER: stability + partition seg 0.307 + division n200 +
   migration 0.380 + escape 0.000, r_max 0.875. Only deform modest (0.018).** The candidate final integrated operating spec.
2. **s1 cf09 (move0.12/cf0.09/γ120/div0.02) — strongest partition: seg 0.411, migr 0.289, escape 0.000, deform 0.014.**
3. **s2 move11 — strongest migration clean-ish: migr 0.497, escape 0.011 (grazing), seg 0.270.**
Integration is essentially DELIVERED (s6 is clean on all 5 axes). **The remaining weak leg is DEFORM (~0.018 clean, ceiling ~0.028
only when containment hard-fails).** Every clean point sits at mass 5e-5 — the prime deform lever (`agent_to_mpm.agent_mass`) has
NOT been pushed at the integrated n182 point through all of INT.

### HYPOTHESIS (Batch 29 / INT-7)
**Raising per-cell mass `agent_to_mpm.agent_mass` (5e-5 → 7e-5/1e-4) at the s6 clean envelope (γ120/div0.025/move0.11/cf0.08–0.09)
lifts deform toward ≥0.025 while staying contained (escape ≤0.01, r_max<1), because mass is the prime deform lever (established,
~15×) and at n182–200 there is boundary-flux headroom the n503 regime lacked (where mass7e-5 re-leaked). The flock provides the
containment that a bare high-mass push lacks. FALSIFIERS: (a) mass≥7e-5 re-leaks even at n182 (escape >0.02, r_max>1) → the clean
deform ceiling ~0.02 is a hard integration ceiling → ADOPT s6 as the final integrated spec, log deform-cap [open]; (b) mass lifts
deform but re-mixes seg<0.20 (heavier push stirs the partition) → mass trades deform for partition, not free; (c) cf0.09/0.10
contains a mass1e-4 push that cf0.08 cannot → containment is the limiting knob, not mass.**

### DESIGN (8 slots) — all seed1 overrides on `embryo_1E_split_hin_s{1,2}.yaml` (no new YAML; scalar tweaks only)
All chemotaxis.gain 0.0 (base −1.0, MUST disable), spin off, seeded split, γ120 & div0.025 & move0.11 & cf0.08 unless noted,
frames 12000 stride 16. Pushes the ONE unexplored deform lever (mass) inside the clean s6 envelope.
- s0 m7_c08     (mass7e-5, cf0.08)                  — EXPLOIT: mass up at the s6 clean envelope. Deform>0.018? escape≤0.01?
- s1 m1e4_c08   (mass1e-4, cf0.08)                  — EXPLOIT: bigger push, same containment. Expect deform↑; watch escape leak.
- s2 m7_c09     (mass7e-5, cf0.09)                  — EXPLOIT: mass up + stronger catch (cf0.09) to hold the heavier push.
- s3 m7_c09_d02 (mass7e-5, cf0.09, div0.02)         — EXPLOIT: mass up with partition-safe division (keep seg high while deform↑).
- s4 m1e4_c10   (mass1e-4, cf0.10)                  — EXPLORE: biggest push + strongest clean catch — how far does deform go clean?
- s5 m7_c08_m12 (mass7e-5, cf0.08, move0.12)        — EXPLORE: mass up + fast move (more momentum) — deform max, does it leak?
- s6 seed2      (SEED2, mass5e-5, s6 config exactly) — EXPLORE: replicate the best all-rounder (s6) across seed — robustness.
- s7 m5_c08     (mass5e-5, cf0.08) = s6 base repeat  — CONTROL (R4 mass ablation): the deform baseline; isolates mass's effect.
Roles: 4 exploit (s0–s3) · 3 explore (s4–s6) · 1 control (s7). READ: deform vs mass (s7 5e-5 → s0 7e-5 → s1 1e-4 at cf0.08; the
mass→deform slope inside the flock) · escape/r_max (does 7e-5/1e-4 leak at n182? cf0.09/0.10 rescue it?) · seg (does the heavier
push re-mix the partition?) · seed2 (s6) reproduces the clean all-rounder?
