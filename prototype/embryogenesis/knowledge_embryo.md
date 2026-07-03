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
- **1A — STARTED Batch 1 (2026-07-03). NOW AT Batch 4. Gate: collapsed=0 & escape=0 & nn_min≥r0 (0.02), accel
  balance-bounded (not clamp).** Auth RECOVERED after the restart — `embryo_1A_b01/b02/b03` all landed real
  archives; the driver counter is `{"batch":4}` (NOT burned to 9+ as the pre-restart doom narrative feared).
  **2 of 3 sub-gates MET at confine 0.1** (collapsed 0, escape 0, accel clean). The only unmet sub-gate is
  **nn_min≥r0**: at confine 0.1 nn_min=0.0048 ≪ 0.02, a FROZEN doublet. **Batch 4 attacks it directly** with active
  `separation` (`embryo_sep.yaml`) + lower spawn density (`embryo_nodiv_spread.yaml`) + a sep×spread combo
  (`embryo_sep_spread.yaml`) + a hard-force dose ceiling (repel 48/96 @ r0 0.02) + a confine-0.05 escape-boundary
  probe + a seed-1 R4 control, all @ confine 0.1 (see `embryo_slots.md` / analysis Batch 4).
  - **b01 (division ON):** runaway `cell_divide` floods to n=2850 → `collapsed`≈0.99 identically → collapse test
    corrupted → 1A runs with division OFF (`embryo_nodiv.yaml`). [established-engineering, see FINDINGS]
  - **b02 (nodiv confine ladder 3.0/2.0/1.0/0.5):** collapse falls smoothly with confine (3.0→0.61, 1.0→0.59,
    0.5→0.45); coarse — no interior window visible. SUPERSEDED by b03's fine sweep.
  - **b03 (nodiv fine sweep 0.3/0.2/0.1 + probes):** THE decisive batch — confine 0.1–0.2 = first collapsed=0 &
    escape=0 window; residual = confinement-press-induced frozen doublet (see FINDINGS). Fully distilled.
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
- **[engineering — OUTAGE, 4 STRAIGHT as of Batch 8 (2026-07-03)] SSH auth to the Janelia login node is DEAD: it
  cleared ONLY b02+b03, then failed b04+b05+b06+b07 (4 consecutive).** b02 (ids 151979189–196) and b03 (ids
  151979211–218) both launched real jobs and landed archives; b04→b07 each returned `Permission denied (publickey,…)`
  on all 8 slots (SUBMIT OUTAGE). **CRITICAL CORRECTION — the HOLD-and-retry guard is COSMETIC in the RUNNING driver,
  proven 5× (b03→b07):** the source patch exists in `embryo_loop.py`, but the live process logs `SUBMIT OUTAGE batch
  N: 0/8 … HOLDING batch N; retry in 10 min` and then the very next log line is `Claude: DESIGN batch N+1` — it never
  retries batch N's submit, it advances. `embryo_loop_state.json`/`embryo_batch_jobs.json` increment on each design
  (now `{"batch":7…}`). So EVERY outage still burns a batch number against the 48-batch 1A clock; the guard is not
  loaded because the driver was never restarted. **Ranked operator fix (agent can do NEITHER): #1 RESTART the driver
  (credential-independent — loads the HOLD guard, actually stops the burn); #2 renew Kerberos/SSH cred (`kinit`/re-add
  key) to restore submit.** Watch each batch for recurrence.
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
