# Working Memory: cardio model-discovery

The DELIVERABLE of this loop is this ledger — defensible claims about **what mechanics are
necessary and sufficient** to reproduce the real cardiomyocyte trajectories (open, superposing
per-node loops that stay periodic across cycles). Fit-RMSE is an *instrument* that adjudicates
hypotheses; it is never the goal. A clean falsification is a success.

Read + update this file EVERY batch. Keep it the single source of truth; the human-readable
per-batch narrative lives in `analysis.md`. Seeded 2026-06-22 from the 08_07 → 08_08 → 08_09
investigation (see `LOGBOOK.md`).

## Paper Summary (update at every theme boundary)

- **THE METRIC (read first):** `fitRMSE` is **boundary-diluted and is NOT a goodness measure** — a do-nothing
  model (predict zero interior motion) BEATS every run to date (user_input 2026-06-23, verified offline on the
  former "best" b04.s2: interior render error 0.00321 vs do-nothing 0.00064 → **R2≈−13**). RANK ON **render-R2**
  (motion-normalised, interior+moving nodes, FREE render): R2→1 good, R2≤0 = worse than no motion = FAILED fit;
  R2=nan (frozen) = FAILED. The honest R2-logging code lands from **batch 6** (batch 5 ran the old script, so
  batch 5 is ranked on **morphology** = the overshoot/streak amplitude in true_vs_learned, which is the direct
  R2 proxy). Every "best fitRMSE" claim below is RETAINED ONLY as a relative damping/streak trend — the absolute
  fits are POOR (R2<0 to date).
- **Open the loops:** SOLVED qualitatively — per-node signed transverse-`w` force opens 2-D loops (Principles 3–5);
  every run opens loops. BUT the open loops **OVERSHOOT the real motion ~5×** and are mis-directed (R2<0). The
  program goal has REFRAMED from "open the loops" to **"CALIBRATE the open loops DOWN to the tiny real motion
  (R2→positive)"** (user_input). Consequence: **TRANS_SCALE is an AMPLITUDE/calibration lever (Principle 15)** —
  lowering it shrinks the overshoot; **raising openness (LAM_OPEN) is COUNTERPRODUCTIVE (F8)** because the loops are
  already too big, not too small. (Its old role as the "streak master knob" stays dead — P11-revised.)
- **Keep them stable across cycles:** **DAMPING (γ-floor magnitude) is the streak knob (Principle 13).** Batch 4
  dose 0.35→0.45→0.55 monotonically reduced streaks + fitRMSE at flat trans|max| (step dt/γ shrinking). Batch 5
  extended the dose to 0.65/0.75: damping keeps helping monotonically (no over-damp collapse by 0.75, fitRMSE still
  falling) but with **DIMINISHING returns and is INSUFFICIENT ALONE** — every pure-damping render still streaks
  (P13 refined). The WINNER is **trans0.06 + floor0.55 (b05.s5)**: low-trans and high-damping **STACK** to the
  lowest-overshoot render yet (Principle 15). γ-floor cures the NaN-freeze (Principle 12, 6/6 stable b04+b05). Two
  levers are DEAD: periodicity (LAM_DRIFT, P14 — no render effect) and TRANS_SCALE-as-streak-knob (P11-revised).
- **Is the tissue structure real (A_ij / phase):** OPEN — A_ij **saturates to +1 EVERYWHERE** (all 6 slots every
  batch incl. b05: range≈[−1,+1] but mean-coupling map uniformly +1 with speckle) → carries ~no structure; φ has
  real blob structure but its causal role is unprobed (Q3).

## Knowledge Base

### Comparison Table

One row per slot worth remembering (best/representative slots; not every run). Metrics from
`run.log`; morphology from the dashboard.

| Batch.slot | Config summary | fitRMSE | drift | trans\|max\| | loops open? | streaks? | Verdict |
| ---------- | -------------- | ------- | ----- | ------------ | ----------- | -------- | ------- |
| 08_08.ref  | N_BEATS=1, scale0.12, open2, lr4e3 (val3) | 0.0064 | n/a | 0.105 | yes (41% nodes) | yes, centre | baseline |
| b01.s0 parent | lr4e3 drift2 scale.12 (NaN-frozen@it250) | 0.0088 | 0.00085 | 0.115 | yes, SUPERPOSE | faint vertical | best render but early-stop artefact |
| b01.s1 lr2e3 | lr2e3 drift2 scale.12 (COMPLETE@1450) | 0.0048 | 0.00033 | 0.114 | yes, not superp. | MASSIVE | new parent (only valid full drift>0 run) |
| b01.s2 lr8e3 | lr8e3 drift2 (NaN-frozen@it300) | 0.0128 | 0.00109 | 0.120 | yes | dense diagonal (worst) | LR too high → NaN |
| b01.s5 drift0 | lr4e3 LAM_DRIFT=0 (COMPLETE@1450) | 0.0043 | 0.00363 | 0.120 | yes, not superp. | YES | drift0 ablation; 10× higher drift metric |
| b02.s0 parent | lr2e3 drift2 scale.12 (FROZE@700, γ0.183) | 0.0065 | 0.00046 | 0.107 | yes, superpose | diagonal moderate | control; low γ drove the freeze |
| b02.s1 scale008 | TRANS_SCALE .12→.08 (FROZE@1000, γ0.165) | 0.0048 | 0.00042 | 0.075 | yes, superpose | vertical reduced | NEW PARENT (openness↔stability) |
| b02.s2 scale006 | TRANS_SCALE .12→.06 (FROZE@1250, γ0.179) | 0.0033 | 0.00025 | 0.056 | SHRINKING | faint (BEST streaks) | best numbers, loops collapse |
| b02.s3 lamtrans1 | LAM_TRANS .3→1.0 (COMPLETE@1450, γ0.275) | 0.0056 | 0.00041 | 0.113 | spaghetti | EXPLODED (worst) | hinge lever falsified |
| b02.s5 drift0 | lr2e3 LAM_DRIFT=0 (COMPLETE@1450, γ0.372) | 0.0045 | 0.00127 | 0.113 | spaghetti | EXPLODED | clean drift0; metric 3× higher |
| b03.s0 parent | GAMMA_FLOOR=0 trans.075 (FROZE@908, γ0.181) | 0.0046 | 0.0014 | 0.076 | open+superpose | faint diagonal | control; clean look = early-stop ARTEFACT |
| b03.s1 gfloor025 | +GAMMA_FLOOR0.25 (COMPLETE@1500, γ0.216) | 0.0061 | 0.00051 | 0.076 | buried | EXPLODED (heavy) | floor stopped freeze; full-train explodes |
| b03.s2 gfloor035 | +GAMMA_FLOOR0.35 (COMPLETE@1500, γ0.300) | 0.0044 | 0.00037 | 0.076 | many open+superpose | moderate (least of completers) | NEW PARENT; floor works, damping reduces streaks |
| b03.s3 scale006 | TRANS_SCALE .08→.06, floor0 (FROZE@871, γ0.163) | 0.0038 | 0.00025 | 0.054 | collapsing | moderate | frozen ARTEFACT (no floor) |
| b03.s5 earlystop | N_ITER 1500→600, floor0 (COMPLETE@600, γ0.185) | 0.0059 | 0.00064 | 0.066 | buried | EXPLODED | clean early-stop still explodes → F7 |
| b04.s0 parent | FLOOR0.35 trans.08 drift2 (COMPLETE@1450) | 0.00361 | 0.00031 | 0.073 | open+superpose | HEAVY (worst damping arm) | control; low-damping baseline |
| b04.s1 gfloor045 | FLOOR 0.35→0.45 (COMPLETE@1450) | 0.00351 | 0.00033 | 0.072 | open+superpose | moderate | damping dose mid → P13 |
| b04.s2 gfloor055 | FLOOR 0.35→0.55 (COMPLETE@1450) | 0.00221 | 0.00040 | 0.070 | open+superpose | LEAST (cleanest ever) | NEW PARENT; damping wins (P13) |
| b04.s3 trans006 | TRANS_SCALE .08→.06, floor0.35 (COMPLETE@1450) | 0.00321 | 0.00028 | 0.058 | open+superpose | HEAVY (still streaks) | trans drop ≠ streak cure → P11 revised |
| b04.s4 drift6 | LAM_DRIFT 2→6, floor0.35 (COMPLETE@1450) | 0.00363 | 0.00023 | 0.078 | open+superpose | HEAVY (lowest metric, no render gain) | F5→P14 (periodicity dead) |
| b04.s5 drift0 | LAM_DRIFT 2→0, floor0.35 (COMPLETE@1450) | 0.00244 | 0.00060 | 0.074 | open+superpose | moderate (≈drift6, not worse) | drift0 ablation; render ≥ drift6 |
| b05.s0 parent | floor0.55 trans0.08 (COMPLETE@1450) | 0.00471† | 0.00057 | 0.077 | open | HEAVY (overshoots) | control; pure-damping still streaks |
| b05.s2 gfloor075 | FLOOR 0.55→0.75 (COMPLETE@1450) | 0.00374† | 0.00053 | 0.071 | open (not collapsed) | moderate (best pure-damping) | dose still paying off, no collapse → P13 refined |
| b05.s3 gfloor035 | FLOOR 0.55→0.35 ABLATION (COMPLETE@1450) | ~0.0046† | ~0.0006 | 0.075 | open | HEAVY (re-streaks) | low-damping anchor; confirms dose |
| b05.s4 lamopen4 | LAM_OPEN 2→4 (COMPLETE@1450) | ~0.0050† | ~0.0006 | 0.074 | open+ENLARGED | WORST (bigger overshoot) | F8: openness-up worsens fit |
| b05.s5 trans006 | TRANS_SCALE 0.08→0.06, floor0.55 (COMPLETE@1450) | ~0.0040† | ~0.0005 | 0.052 | open (smaller) | LEAST of batch (tamest) | NEW PARENT; trans×damping stack → P15 |
†fitRMSE is boundary-diluted — NOT a goodness measure (user_input); batch 5 R2 not logged (old script). Ranked on morphology.

### Established Principles

1. **The loops collapse to lines because the mechanics is reversible.** `forward_beat` is a
   first-order **overdamped** elastic relaxation driven by a single scalar gate `gate(u(t))`:
   `pos += (dt/γ)·F(pos, act)`, no inertia, no rate-dependent force → time-reversible → each node
   tracks a moving equilibrium out and back along the **same path** → enclosed area ≈ 0 → a line.

2. **No openness penalty can open a loop the mechanics cannot represent.** Raising `lam_open` in
   08_07 did nothing — no mechanical DOF for enclosed area, so the gradient had nowhere to go.

3. **The FHN already carries a loop; use its `w`.** The `(u,w)` limit cycle is a closed orbit.
   A per-node **transverse (cross-fibre) force from the recovery variable `w`** (out of phase with
   `u`), added to the axial `u`-contraction, makes each node trace a real 2-D ellipse oriented by
   the fibre angle (`forward_beat_loop` + `LoopParams`, 08_08). **Loops open.**

4. **The transverse gain must be a per-node SIGNED field, not a global scalar.** A global `trans≥0`
   is driven to 0 by RMSE (a node with opposite-chirality GT loop is hurt by a one-signed force). A
   per-node signed tanh-bounded field lets each node pick chirality/magnitude → `l_fit` & `l_open`
   cooperate. (08_08: field grows to ~0.1–0.25 and sticks; 41–61 % of nodes active.)

5. **The openness loss must be scale-free and masked to fail.** Openness is a tiny fraction of total
   motion, so absolute-minor-axis averaged over all N nodes is negligible vs the major-axis fit →
   no teeth. Working form: `mean over looping nodes of (1 − openness_pred/openness_real)₊`.

6. **Opening the loops introduces cross-cycle instability (streaks).** With the transverse force on,
   the continuous multi-cycle render drifts on high-gain (central) nodes, though single-beat
   fit-RMSE is excellent. Cause: single-beat training never sees / penalises cycle-to-cycle drift.
   Amplitude taming (`TRANS_SCALE` 0.3→0.12 + hinge) cleans the bulk but not the centre.

7. **`u` was too wide / not AP-like under the original FHN.** `eps=0.3` → slow sawtooth on most of
   the cycle. 08_08 widened `eps`→0.9 range (init 0.4) and lowered gate `eta` floor to 0.05.
   (Effect on the fit not yet isolated — Q4.)

8. **Multi-cycle + LAM_DRIFT training is forward-NaN-unstable — REFINED by Principle 10 (it is a
   γ-threshold, not an LR-threshold).** (Batch 1, two orthogonal axes.) LR sweep at drift2: lr2e3 trained
   clean, lr4e3 AND lr8e3 hit a **forward-pass** NaN ~it300 and then skipped every step (frozen). Drift
   sweep at lr4e3: drift0 clean, drift2 AND drift5 NaN'd. The NaN is in `forward_beat`'s explicit
   overdamped relaxation (`pos += (dt/γ)·F`, line 194), NOT in the gradient — gradient clipping
   (`GRAD_CLIP=1.0`) was ON and did not help; the `isfinite(loss)` guard only prevents poisoning, leaving
   the run permanently frozen. Batch 1's reading "LR≤2e-3 is the safe point" was an artefact of which
   runs happened to keep γ high — see Principle 10 for the corrected, causal account.

10. **The forward NaN-freeze is a γ-threshold instability of the explicit integrator: it fires whenever
    training drives γ below ~0.2, at ANY LR.** (Batch 2, decisive.) At LR=2e-3, 4/6 runs still NaN-froze;
    the discriminator was γ, not the knob varied: every frozen run had γ≈0.165–0.183 (s0@700, s1@1000,
    s2@1250, s4@1250), both completers had γ≥0.275 (s3 0.275, s5 0.372). The step size is `dt/γ`
    (line 194, clamped only at min=1e-2), so as the optimiser shrinks γ the explicit overdamped sub-step
    overshoots and blows up. This (a) PARTIALLY FALSIFIES Principle 8's LR framing and (b) makes the
    freeze an *uncontrolled early-stop* that confounds every render comparison (the only fully-trained
    runs are the high-γ ones, which then explode via Q7). Predicted cure (Batch 3, code edit): floor the
    effective γ in the integrator (`CARDIO_GAMMA_FLOOR`, line 194 `clamp(min=max(1e-2,FLOOR))`; default
    0 = exact reproduction) so the step stays bounded regardless of the learned γ (Q8).

11. **Transverse MAGNITUDE (`TRANS_SCALE`) is the master knob for the openness↔stability trade-off.**
    (Batch 2, clean monotone sweep.) 0.12→0.08→0.06 monotonically lowered trans|max| (0.107→0.075→0.056),
    drift (0.00046→0.00042→0.00025), fitRMSE (0.00646→0.00480→0.00326) AND render-streak amplitude (0.06
    faintest). The streaks are therefore transverse-magnitude-driven, not periodicity-driven: s2 (trans
    0.056) trained to it1250 yet stayed faint, while the trans≈0.113 runs streak heavily — so streak
    growth is gated by magnitude, not iteration count alone. The COST is openness: at 0.06 the loops
    shrink toward collapse. → a real Pareto front (Q2); trans≈0.075 is the current open-and-reduced-streak
    compromise. By contrast LAM_DRIFT and LAM_TRANS do NOT move the streaks (F5-revised UPHELD; F6 new).
    **REVISED (Batch 4): the STREAK half of this Principle was a NaN-freeze artefact.** The batch-2 monotone
    "lower trans → fewer streaks" was measured on runs that NaN-FROZE at different iterations (uncontrolled
    early-stops). With the γ-floor making all runs fully train, b04.s3 (trans0.06, floor0.35) STILL streaks
    heavily — lowering TRANS_SCALE 0.08→0.06 does NOT clean the render, while raising the γ-floor does
    (Principle 13). So TRANS_SCALE is NOT the dominant streak lever under full training. What SURVIVES: the
    openness↔magnitude trade-off for loop SIZE (too-low trans collapses the loops) and that trans|max| tracks
    TRANS_SCALE. What is RETRACTED: the claim that TRANS_SCALE is the master *streak* knob — damping is.

12. **A γ-floor in the explicit integrator CURES the NaN-freeze.** (Batch 3, decisive — direct test of
    Principle 10's predicted cure.) Flooring the effective γ in the step `dt/gamma.clamp(min=max(1e-2,
    GAMMA_FLOOR))` (cardio_train08_09.py:78,196) at 0.25 and 0.35 let BOTH runs train to it1500 with ZERO
    non-finite steps, while all three GAMMA_FLOOR=0 runs NaN-froze (s0@908 γ0.181, s3@871 γ0.163, s4@900
    γ0.163). This confirms Principle 10's causal account (the freeze is the explicit overdamped step
    overshooting as γ→<0.2) and supplies the fix: bound the step, don't constrain LR. The floor is now a
    STANDARD part of the parent config. COROLLARY (the sting): the cure unmasks that fully-trained runs
    then EXPLODE (Q7 below) — the freeze had been silently early-stopping every run into a clean-looking
    artefact. So Principle 12 does not by itself produce a stable render; it makes the real instability
    *visible and trainable* so it can be attacked (damping Q9, periodicity Q1-revisited).

13. **DAMPING (the γ-floor MAGNITUDE) is the lever that reduces the free-rollout streaks.** (Batches 3→4 —
    clean monotone dose-response, Q9 resolved positive.) Raising the floor 0.35→0.45→0.55 (b04.s0→s1→s2)
    monotonically reduced render-streak amplitude (s0 heavy spaghetti → s2 cleanest render to date, loops open
    + superposing) AND lowered fitRMSE (0.00361→0.00351→0.00221), at FLAT trans|max|≈0.07 — so it is NOT a
    transverse-magnitude effect. Mechanism: the explicit overdamped step is `dt/γ` (cardio_train08_09.py:196);
    a larger floor → smaller step → gentler free rollout → less cross-cycle overshoot. NOTE the learned γ scalar
    sits at its init (0.300) in all 6 runs and every floor exceeds it, so the floor is the SOLE damping DOF (the
    learned γ's gradient vanishes below the floor). This is the FIRST lever to improve the render AND the fit at
    once (TRANS_SCALE traded openness for streaks; LAM_DRIFT did nothing — P11-revised, P14). CONFOUND noted: the
    higher floor also shifts the FHN basin (s2 eps0.166/eta0.05 vs parent eps0.745) — a 3-point monotone dose is
    strong but basin-vs-damping is not 100% separated; batch 5 extends the dose (0.65, 0.75) + a floor0.35 anchor
    to test monotonicity vs an over-damp/under-fit turnover.
    **REFINED (Batch 5): damping is monotone but DIMINISHING and INSUFFICIENT ALONE.** The dose continued 0.55→
    0.65→0.75 (b05.s0/s1/s2): s2 (0.75) is the cleanest of the three and its fitRMSE keeps falling
    (0.00471→0.00374) with loops still OPEN (no collapse), so there is no over-damp turnover by 0.75 — BUT s1 (0.65)
    is barely better than s0 (0.55) and NO pure-damping render is clean (all still streak). The floor0.35 anchor
    (b05.s3) re-streaks heavily, re-confirming the dose. So damping is necessary, monotone, and has not saturated
    into collapse, yet it cannot calibrate the render on its own — it must be STACKED with low transverse magnitude
    (Principle 15). The damping-vs-basin confound is unresolved but secondary now that the live lever is amplitude.

14. **The LAM_DRIFT periodicity loss does NOT reduce the free-rollout streaks (it only lowers the per-step
    drift METRIC).** (Batch 4, decisive — F5 promoted, Q1 closed-negative.) The 08_09 premise was that a
    cycle-to-cycle periodicity penalty would kill the streaks. On the FIRST drift sweep where every run fully
    trains (γ-floor on, no freeze confound), drift {0,2,6} (b04.s5/s0/s4) cut the drift metric monotonically
    (0.00060→0.00031→0.00023) yet the RENDER did not improve at all — drift6 (lowest metric) streaks as hard as
    drift0, and drift0 is if anything tamer than drift6. The metric and the free rollout are decoupled: pinning
    per-step periodicity does not stabilise the multi-cycle trajectory. Earlier batches could not settle this
    (the drift>0 runs NaN-FROZE — the metric-vs-render comparison was confounded by uncontrolled early-stops);
    the γ-floor removed that confound and the verdict is now clean. Periodicity is NOT a route to a stable rollout.

9. **Low per-beat fitRMSE does NOT imply a stable free rollout.** The training fit pins the boundary
   band to the real displacement (Dirichlet), so fitRMSE stays ~0.004–0.009 even when the *free*
   multi-cycle render (`true_vs_learned`) explodes into streaks (batch 1 slot1: fitRMSE 0.0048 yet
   whole-field streaks). Judge stability from the render, not the scalar.

15. **Low TRANSVERSE MAGNITUDE and HIGH DAMPING STACK to calibrate the loop amplitude; TRANS_SCALE is an
    AMPLITUDE lever (not a streak lever).** (Batch 5, decisive — the winner b05.s5.) trans0.06 at floor0.55 gives
    the LOWEST-overshoot render the program has produced (smallest red excursions, most red ellipses sit on the
    green GT), beating every pure-damping run. Crucially this is NOT a contradiction of P11-revised: in batch 4,
    trans0.06 at floor0.35 (b04.s3) still streaked, so trans ALONE does nothing — but trans0.06 WITH high damping
    (b05.s5) is the tamest run. Mechanism: damping (P13) holds the rollout stable; lowering trans then shrinks the
    transverse EXCURSION (the overshoot amplitude, which is what drives R2<0). The two stabilisers are
    complementary — damping for stability, low-trans for amplitude. This RE-INSTATES TRANS_SCALE as a calibration
    knob distinct from its dead streak role. Open: how far can trans drop (0.045, 0.03) before the loops collapse
    (the openness floor) — i.e. the amplitude↔collapse trade-off (batch 6).

16. **`fitRMSE` is boundary-diluted and is NOT a goodness measure; render-R2 is the metric.** (user_input
    2026-06-23, verified offline.) ~30% of nodes are the Dirichlet-pinned boundary band (pred==GT → exactly 0
    error) and the real motion is tiny (~6e-4), so averaging absolute error over boundary+static-interior nodes
    makes fitRMSE ~0.002–0.009 even for a model whose free render is ~5× too large. A do-nothing model (predict
    zero interior motion) scores BETTER than every run to date: on the former "best" b04.s2, interior render error
    0.00321 vs do-nothing 0.00064 → motion-**R2 ≈ −13**. RANK ON **render-R2** (motion-normalised SS, interior AND
    moving nodes, computed on the FREE render in run.log as `render_fit interior_moving: R2=…`): R2→1 good, R2≤0 =
    worse than predicting no motion = FAILED, R2=nan (frozen) = FAILED. Code change: `l_fit` now interior-only and
    a `render_fit R2/NRMSE` line is logged — EFFECTIVE FROM BATCH 6 (batches 1–5 ran the old script; their fitRMSE
    rankings hold only as RELATIVE damping/streak trends, never as absolute goodness — absolute fits are R2<0).

### Falsified Hypotheses

- **F1.** "Increasing `lam_open` opens the loops." — FALSE without a mechanism (Principle 2).
- **F2.** "A single global transverse gain suffices." — FALSE; decays to 0 (Principle 4).
- **F3.** "Per-cycle centring of the transverse drive removes the render streaks." — FALSE; tried,
  no effect (streaks are dynamical instability, not net per-cycle impulse).
- **F4.** "Lowering transverse amplitude alone removes the streaks." — PARTIAL/FALSE; 0.25× cleans
  the bulk but the high-gain central region still streaks (Principle 6).
- **F5.** "Multi-cycle training with a LAM_DRIFT periodicity loss removes the cross-cycle streaks."
  (the 08_09 premise / Q1.) — PARTIAL FALSIFICATION (batch 1), now STRENGTHENED (batch 2). drift2 cut the
  drift *metric* ~3–10× vs drift0 (b01: 0.00033 vs 0.00363; b02 clean-LR: s0 0.00046 vs s5 0.00127) but
  the renders never improve with drift. The loss lowers measured per-step drift without stabilising the
  free rollout. Revised hypothesis (batch 2): streaks are driven by transverse *magnitude*, so TRANS_SCALE
  (not LAM_DRIFT, not LAM_TRANS) is the effective lever — **CONFIRMED, now Principle 11.**
  **FINAL (Batch 4): the LAM_DRIFT half is now fully ESTABLISHED → Principle 14.** The drift sweep on
  fully-training runs (drift{0,2,6}, b04.s5/s0/s4) lowered the drift metric monotonically yet left the render
  streaks unchanged (drift6 ≈ drift0) — the freeze-artefact escape hatch is closed; periodicity gives no render
  benefit. SEPARATELY the batch-2 "TRANS_SCALE is the effective lever" half is itself REVISED (see P11): under
  full training trans0.06 still streaks; the real streak lever is DAMPING (Principle 13), trans only set loop size.

- **F6 (batch 2).** "Strengthening the LAM_TRANS runaway-tail hinge (0.3→1.0) reduces the streaks."
  — FALSE. Slot 3 (lamtrans1) gave the WORST render of the batch: it landed in a different basin
  (high γ=0.275, distinct fibre/gain maps), trained to completion, and exploded into spaghetti
  (trans|max| still 0.113). The hinge does not cap the *effective* transverse excursion in the free
  rollout. Learned: tame the magnitude at the source (TRANS_SCALE, Principle 11), not via a soft penalty.

- **F7 (batch 3).** "Once the γ-freeze is removed (or via an explicit early-stop), a fully-/cleanly-trained
  run renders without streaks." — FALSE. (1) Original hope: the clean early-frozen renders of batches 1–2
  were the real model; removing the freeze (γ-floor) would reproduce them at full training. (2) Evidence:
  the two γ-floored completers (b03.s1 floor0.25, b03.s2 floor0.35) BOTH streak in the free render (s1
  heavy spaghetti), and the explicit clean early-stop (b03.s5, N_ITER600, trans0.066) ALSO explodes; only
  the NaN-frozen runs (s0,s3,s4 — params pinned pre-blowup) look clean. (3) Learned: the clean look was the
  FREEZE itself (a pinned pre-instability state), not iteration count and not a trainable optimum — the
  fit-vs-rollout divergence is genuine and grows with training (Q7 resolved). Early-stop is NOT a usable
  stability route. (4) Revised: attack the divergence directly via damping (γ-floor magnitude, Q9) and a
  now-valid periodicity test (Q1-revisited), not by stopping training.

- **F8 (batch 5).** "Once damping controls the streaks, raising openness (LAM_OPEN 2→4) re-enlarges the loops
  WITHOUT re-streaking — a free win on openness." — FALSE, and instructively so. (1) Original hope (batch-4 plan):
  openness and stability were in tension only via transverse magnitude, so with damping holding stability the
  openness budget could be spent. (2) Evidence: b05.s4 (lamopen4) gave the WORST render of the batch — the loops
  grew LARGER and overshot the green GT even more; combined with user_input's offline finding that the loops
  already overshoot the real motion ~5× (R2≈−13), bigger loops = worse fit. (3) Learned: the loops are too BIG,
  not too small — the openness loss was solving a problem that no longer exists, so pushing it is counterproductive.
  The fit error is now an OVERSHOOT/calibration problem, not an openness deficit. (4) Revised: REDUCE loop amplitude
  (low trans + high damping, Principle 15) and consider REMOVING/lowering LAM_OPEN; add a motion-normalised l_fit
  (CARDIO_FIT_NORM) that penalises overshoot directly (batch 6).

### Open Questions

- **Q1 (08_09 target).** CLOSED-NEGATIVE (batch 4) → Principle 14. The clean LAM_DRIFT {0,2,6} sweep on
  fully-training runs (b04.s5/s0/s4) lowered the per-step drift METRIC monotonically but left the render
  streaks unchanged (drift6 streaks as hard as drift0). The earlier "no render effect" reading was NOT a
  freeze artefact — it holds under full training. Periodicity is not the route to a stable rollout; the
  effective lever is DAMPING (Principle 13). No further drift sweeps planned.
- **Q2 (batch 4 → RESOLVED batch 5) → Principle 15.** "Does combining high damping with low trans help or just
  collapse loops?" — it HELPS: trans0.06 at floor0.55 (b05.s5) is the tamest/lowest-overshoot render yet (the two
  stabilisers stack — damping for stability, low-trans for amplitude → Principle 15). Loops did NOT collapse at
  trans0.06+floor0.55. Damping dose continued to floor0.75 without over-damp collapse but with diminishing returns
  (P13 refined). RESIDUAL (→Q10): how far can trans drop (0.045, 0.03) before the loops collapse — the amplitude↔
  collapse floor.
- **Q3.** Do `A_ij` and the phase map carry real tissue structure, or is the transverse field doing
  all the loop work? (08_07: A_ij ≈ init 0.96; φ ≈ flat.)
- **Q4.** Does the sharper FHN (`eps`, `eta`) improve the fit, or just the look of `u`?
- **Q5.** Is overdamped + transverse enough, or is a small **inertia / viscoelastic** (second-order)
  term a cleaner route to stable open loops?
- **Q6 (LR).** SUPERSEDED by Principle 10: the instability is a γ-threshold, not an LR-threshold
  (LR=2e-3 still froze 4/6 in batch 2). LR is no longer the control variable for stability; γ-flooring is
  (Q8). The per-category-LR sub-question is parked until the γ-freeze is removed.
- **Q7 (batch 1 → RESOLVED batch 3).** "Do streaks grow with training, i.e. is there a genuine
  fit-vs-rollout divergence beyond transverse magnitude?" — YES (decisive). With the γ-freeze removed
  (Principle 12), the fully-trained floored runs (b03.s1, s2) AND the explicit clean early-stop (b03.s5,
  600 iters) ALL explode in the free render; only NaN-frozen runs (params pinned pre-blowup) look clean.
  So the clean look was the freeze, not iteration count and not magnitude alone — the divergence is real
  and grows with training (F7). Answer to the batch-3 sub-question: an explicit early-stop does NOT render
  cleaner — it explodes too. The divergence, not the freeze, is now the central target.
- **Q8 (batch 1 → RESOLVED POSITIVE batch 3).** Does a γ-floor stop the NaN-freeze? YES — see Principle
  12. Both floored runs (0.25, 0.35) trained to it1500 with zero non-finite steps; all three unfloored
  runs froze at γ≈0.16–0.18. Follow-up (a)/(b): they EXPLODE (a), not stay clean (b) — the modest trans
  did NOT save them (F7), which is why the residual instability (Q7) and the damping lever (Q9) are now the
  active questions. The γ-floor is henceforth a standard part of the parent config.
- **Q9 (batch 3 → RESOLVED POSITIVE batch 4) → Principle 13.** Does DAMPING (γ-floor MAGNITUDE) reduce the
  free-rollout streaks? YES, monotonically. The batch-4 dose sweep (floor 0.35/0.45/0.55, b04.s0/s1/s2)
  reduced render-streak amplitude monotonically AND improved fitRMSE (0.00361→0.00351→0.00221) at flat
  trans|max|≈0.07 — confirming the batch-3 hint as a clean 3-point trend. Promoted to Principle 13. RESIDUAL
  sub-questions (Q9b, batch 5): (i) does the dose keep paying off (floor 0.65, 0.75) or saturate / turn over
  into over-damp (under-fit, collapsed loops)? (ii) the higher floor shifts the FHN basin (s2 eps0.166 vs parent
  0.745) — a floor0.35 anchor + extended dose tests damping-vs-basin separation.
  **Q9b RESOLVED (batch 5):** the dose KEEPS paying off to floor0.75 (b05.s2 cleanest of the damping arm, loops
  not collapsed, fitRMSE still falling) — NO over-damp turnover by 0.75 — but with DIMINISHING returns and never
  clean alone (P13 refined). The floor0.35 anchor (b05.s3) re-streaked, re-confirming the dose. Basin-vs-damping
  still not separated, but de-prioritised now that amplitude (Q10) is the live lever.
- **Q10 (batch 5 → live). THE central question per the metric reframe (P16) and the overshoot finding (F8/P15):
  can the over-large open loops be CALIBRATED DOWN to the tiny real motion so render-R2 goes POSITIVE?** Batch 5
  established the amplitude levers (low trans + high damping, P15; openness-up is counterproductive, F8). Batch 6
  attacks it directly: (a) a motion-normalised l_fit (`CARDIO_FIT_NORM=1`) that penalises interior OVERSHOOT during
  training — the user-proposed calibration mechanism; (b) push trans lower (0.045) toward the collapse floor;
  (c) remove/lower the now-counterproductive LAM_OPEN (2→0, 2→1); (d) stack max damping (floor0.75) with low trans.
  Success = render-R2 > 0 (first positive fit). This SUBSUMES the old "lowest-fitRMSE" goal (P16: fitRMSE is not goodness).
- **Q11 (batch 6, methodological).** Does `CARDIO_FIT_NORM` (normalising l_fit by GT-motion-RMS on moving nodes)
  actually CALIBRATE the loop amplitude (R2↑), or does it just shrink everything toward the do-nothing baseline
  (loops collapse, R2→0 from below)? The FIT_NORM=0 parent is the ablation/control.

**RULE: keep summaries for the last 4 completed batches, oldest→newest. This section MUST appear
before `## Current Batch`.**

- **Batch 2 (2026-06-23) — MIXED, decisive (4/6 NaN-frozen again, but at LR=2e-3).** Streak-attack at
  the "safe" LR. All 6 produced full data. Findings: (i) **Principle 11** → TRANS_SCALE is the master
  openness↔stability knob: 0.12→0.08→0.06 monotonically dropped trans|max|, drift, fitRMSE AND streaks;
  0.06 near-streak-free but loops collapse; 0.075 = compromise (new parent). (ii) **Principle 10** →
  the NaN-freeze is a γ-threshold (fires when γ<~0.2), NOT an LR-threshold — 4/6 froze at LR=2e-3, every
  frozen run γ≈0.165–0.183, both completers γ≥0.275. Partially falsifies Principle 8's LR framing.
  (iii) **F6** → LAM_TRANS 0.3→1.0 (s3) FALSE: worst render, different basin. (iv) Q1 RESOLVED-negative
  (clean drift sweep: drift only moves the metric). (v) Q7 sharpened: streak growth is magnitude-gated
  (s2 trained long yet faint). (vi) Q3 still open: A_ij saturates to +1 everywhere in all 6. The freeze
  is now the #1 confound (it is an uncontrolled early-stop). Next: a code-level γ-floor (Q8) to remove it.

- **Batch 3 (2026-06-23) — GOOD, decisive (γ-floor cures the freeze; full training then explodes).**
  Tested the γ-floor (code edit, GAMMA_FLOOR at :78/:196) vs the unfloored parent + a trans Pareto re-walk
  + an early-stop probe. Findings: (i) **Principle 12 / Q8 RESOLVED+** → the γ-floor ELIMINATES the
  NaN-freeze: both floored runs (0.25, 0.35) trained clean to it1500; all three unfloored runs froze at
  γ≈0.16–0.18 — confirming Principle 10's dt/γ mechanism and supplying the cure. (ii) **Q7 RESOLVED / F7**
  → removing the freeze UNMASKS that fully-trained runs EXPLODE (s1 heavy, s2 moderate); an explicit clean
  early-stop (s5@600) ALSO explodes; only frozen runs look clean. The fit-vs-rollout divergence is genuine,
  grows with training, and is NOT cured by removing the freeze or by modest trans. (iii) **Q9 new** →
  damping helps: s2 (floor0.35) streaks less than s1 (floor0.25), but confounded by a basin shift → dose
  sweep next. (iv) Q2 trans Pareto STILL un-walked (s3/s4 ran floor0 → froze). (v) Q3 unchanged (A_ij still
  +1 everywhere). Winner/new parent = s2 (gfloor035): only clean-trained, low-streak, low-RMSE run. Next:
  attack the divergence via a γ-floor dose sweep + the first VALID LAM_DRIFT render test (runs now complete).

- **Batch 4 (2026-06-23) — EXCELLENT, decisive (DAMPING found; periodicity falsified clean; 6/6 trained).**
  First fully-clean batch — all 6 trained to it1450 with ZERO NaN-freeze (γ-floor≥0.35 everywhere → P12
  re-confirmed 6/6). Findings: (i) **Principle 13 / Q9 RESOLVED+** → DAMPING (γ-floor magnitude) is THE streak
  lever: floor 0.35→0.45→0.55 (s0→s1→s2) monotonically reduced render streaks AND improved fitRMSE
  (0.00361→0.00351→0.00221) at flat trans|max|≈0.07 (step dt/γ shrinking, not a magnitude effect). s2 (floor0.55)
  is the cleanest render to date — loops open + superpose, streaks largely tamed → NEW PARENT. (ii) **Principle 14
  / Q1 CLOSED-NEG** → LAM_DRIFT {0,2,6} cut the drift METRIC monotonically (0.00060→0.00031→0.00023) but did
  NOTHING for the render (drift6 streaks as hard as drift0) — first valid test on fully-training runs, freeze
  escape hatch closed → periodicity is dead for streaks. (iii) **Principle 11 REVISED** → trans0.06+floor (s3)
  still streaks heavily; TRANS_SCALE is NOT the streak lever under full training (batch-2 reading was a freeze
  artefact); it only sets loop size. (iv) Learned γ scalar = init 0.300 in all 6, masked by the floor → the
  floor is the sole damping DOF. (v) Q3 unchanged (A_ij +1 everywhere; φ blob structure). Caveat: floor0.55 also
  shifts the FHN basin (eps0.166) — damping-vs-basin not 100% separated. Next: extend the dose (floor 0.65/0.75)
  + floor0.35 anchor (monotonicity/over-damp turnover), trans0.06 at floor0.55, and a LAM_OPEN push under damping.

- **Batch 5 (2026-06-23) — GOOD, decisive REFRAME (loops OVERSHOOT; calibrate-DOWN beats open-UP; 6/6 trained).**
  Extended the damping dose + opened the openness/trans frontier. 6/6 trained to it1450, zero NaN (P12 6/6 again).
  CRITICAL: user_input revealed fitRMSE is boundary-diluted and NOT a goodness measure (→ Principle 16); the honest
  render-R2 code lands only from batch 6, so batch 5 ranked on MORPHOLOGY (overshoot amplitude = R2 proxy). Findings:
  (i) **Principle 13 REFINED / Q9b RESOLVED** → damping dose 0.55→0.65→0.75 keeps helping monotonically (b05.s2@0.75
  cleanest of the damping arm, loops NOT collapsed, fitRMSE still falling) — NO over-damp turnover by 0.75 — but with
  DIMINISHING returns and INSUFFICIENT ALONE (every pure-damping render still streaks). floor0.35 anchor (s3)
  re-streaked, re-confirming the dose. (ii) **Principle 15 / Q2 RESOLVED** → trans0.06 + floor0.55 (s5) is the
  tamest/lowest-overshoot render yet — low-trans and high-damping STACK (damping=stability, low-trans=amplitude);
  TRANS_SCALE re-instated as a CALIBRATION/amplitude lever (distinct from its dead streak role, P11-revised). →
  WINNER/new parent. (iii) **F8** → LAM_OPEN 2→4 (s4) gave the WORST render (loops grow, overshoot worse) — the
  loops are too BIG not too small, so openness-up is counterproductive. (iv) **Principle 16** → fitRMSE is boundary-
  diluted; rank on render-R2; absolute fits R2<0 to date. (v) Q3 unchanged (A_ij +1 in all 6; φ blob). Program GOAL
  REFRAMED: from "open the loops" to "CALIBRATE them DOWN to the real motion (R2→positive)". Next: add CARDIO_FIT_NORM
  (motion-normalised l_fit, penalise overshoot), push trans lower, remove/lower LAM_OPEN, stack max damping; rank R2.

---

## Current Batch

### Batch info
Batch 6 (2026-06-23). Theme: **CALIBRATE the over-large loops DOWN to the tiny real motion → push render-R2 > 0
(Q10).** This is the reframe forced by Principle 16 (fitRMSE is not goodness; current best R2≈−13) and F8/P15 (the
loops OVERSHOOT ~5×; reduce amplitude, don't add openness). **CODE EDIT:** added `CARDIO_FIT_NORM` (cardio_train08_09.py
:78 + l_fit block) — when =1, l_fit is motion-normalised over the moving-node mask (residual-RMS / GT-motion-RMS, an
NRMSE) so interior OVERSHOOT is penalised DURING training; default 0 = the absolute interior RMSE used in batches 1–5
(so the parent is its own ablation). The honest render-R2 line is now logged (effective this batch). **RANK ON
render-R2** (run.log `render_fit interior_moving: R2=…`); treat R2≤0 as a failed fit regardless of fitRMSE, skip
R2=nan (frozen) as FAILED. Parent/control = batch-5 slot5 = trans006+floor055 (the lowest-overshoot render).
Lines, each one-knob from parent: (A) the new calibration mechanism (FIT_NORM=1); (B) amplitude push (trans 0.06→
0.045 toward the collapse floor, Q10); (C) the openness arm — LAM_OPEN 2→0 (ABLATION, remove the counterproductive
loop-enlarging pressure, F8) and 2→1 (dose); (D) damping stack (floor 0.55→0.75).

### Current hypothesis
"The open loops fit poorly (R2<0) because they OVERSHOOT the real motion ~5×, not because they are too small
(F8/P15/P16). So CALIBRATING the amplitude down should raise R2: (a) a motion-normalised l_fit (FIT_NORM=1)
penalises interior overshoot directly and should be the single biggest R2 gain — unless it just shrinks motion
toward the do-nothing baseline (Q11, the parent is the control); (b) lowering trans 0.06→0.045 shrinks the
transverse excursion and should keep raising R2 until the loops start to collapse (the amplitude floor); (c)
removing the now-counterproductive openness pressure (LAM_OPEN 2→0) should HELP, or at least not hurt — if it
helps, openness has become a liability; (d) stacking max damping (floor0.75) with low trans should give the
cleanest pure-mechanics render. Success = the FIRST run with render-R2 > 0."

### Slots this batch (one-knob from parent trans006+floor055)
- s0 parent     — trans006 floor055 open2 drift2 FIT_NORM=0      (control / current best render; FIT_NORM ablation)
- s1 fitnorm    — CARDIO_FIT_NORM 0→1            (NEW MECHANISM — motion-normalised l_fit penalises overshoot, Q10/Q11)
- s2 trans0045  — CARDIO_TRANS_SCALE 0.06→0.045  (amplitude push DOWN toward the collapse floor, Q10)
- s3 lamopen0   — CARDIO_LAM_OPEN 2→0            (ABLATION — remove counterproductive openness pressure, F8)
- s4 lamopen1   — CARDIO_LAM_OPEN 2→1            (openness dose mid — brackets s3 vs parent)
- s5 floor075   — CARDIO_GAMMA_FLOOR 0.55→0.75   (stack MAX damping with low trans — cleanest pure-mechanics render)

### Emerging observations

**CRITICAL: this section must ALWAYS be at the END of the file.**

_(pre-launch) Batch 5 forced the program's biggest reframe. The damping lever (P13) was extended to floor0.75 and
keeps paying off (no collapse) but with diminishing returns and never clean alone; the real win was trans0.06 +
floor0.55 (s5), the lowest-overshoot render yet — low-trans and high-damping STACK (P15). But user_input's offline
audit is the headline: fitRMSE is boundary-diluted and meaningless as goodness (P16) — every run to date has
render-R2 < 0 (worse than predicting NO motion), because the open loops OVERSHOOT the real motion ~5× (F8: pushing
LAM_OPEN up made it worse). So the goal flips from "open the loops" to "CALIBRATE them down" (Q10). Batch 6 attacks
this with the user-requested FIT_NORM mechanism + amplitude/openness/damping one-knob probes, and — for the first
time — RANKS ON render-R2 (now logged). Sanity-check on landing: does run.log now carry `render_fit ... R2=`? Does
ANY slot reach R2 > 0 (the first real fit)? Does FIT_NORM (s1) calibrate (R2↑) or just collapse motion toward
do-nothing (Q11)? Does trans0.045 (s2) still hold open loops or start to collapse? Does removing LAM_OPEN (s3) help
or hurt R2? Watch for the trade where amplitude-reduction overshoots into collapse (R2→0 from below = no motion)._
