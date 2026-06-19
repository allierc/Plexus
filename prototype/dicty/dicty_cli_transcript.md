

================================================================================
## DICTY START — 2026-06-17 15:54:59
================================================================================
`user_input.md` had no pending items, so no acknowledgement to move.

**Batch 1 set up.** I read the instruction, ledger, base spec and user input; the ledger flagged the central open question (is a long-range cAMP relay-wave necessary/sufficient?), and parameter tuning of {AR + chemotaxis + influx} alone is already falsified at ~0.32 inner-mass vs real 0.61.

Changes:
- **Added a `relay` operator** in `dicty_ops.py:177` — a FitzHugh-Nagumo-style excitable update on the existing `camp` grid with a hidden refractory grid `_relay_v`. When `gain=0` it's a no-op (clean ablation).
- **Updated `specs/dicty_loop_base.yaml`** to the prior best (`spring.kadh=120, sense.gain=40, inflow.rate=2.0`) and inserted `relay` into the schedule between `camp.diffuse` and `sense` with `gain=0.0` — parent ≡ prior best.
- **Updated `dicty_knowledge.md`** with 10 explicit hypotheses for Batch 1 (H1 necessity, H2 sufficiency, H3 saturation of relay; H4–H10 for the cAMP machinery, kadh extrapolation, new `spring.delta` sweep, inflow trade-off, friction).
- **Rewrote `sweep_plan.json`** with 16 sweeps × 16 values. The **central experiment is sweep 1**: `relay.gain ∈ [0 … 120]` with 0 as the ablation — necessity and sufficiency in one curve. Sweeps 2–16 refine the response surface around the new parent (extending `sense.gain` and `spring.kadh` past prior upper edges, and introducing a `spring.delta` sweep that has never been mapped).
- **Initialized `dicty_loop_log.md`** with the Batch-1 design block and per-slot predictions.


================================================================================
## BATCH 1 — 2026-06-17 16:03:44
================================================================================
Batch 1 read-back complete. Summary:

**Best result of the batch:** `relay.gain=120` → loss=**0.5474**, inner_mass=**0.404** (down from prior parent's 1.55 / 0.27). Relay-gain response is monotone-up to the top of range with no saturation seen — extended in Batch 2 to [40, 800].

**Biggest scientific finding (breakthrough):** `spring.r_on=0.40` reaches **inner_mass=0.586 ≈ REAL 0.606** with visibly few-compact-mound morphology. This **partially retracts** the ledger's prior claim that "{AR+chemotaxis+influx+adhesion} cannot reach the real morphology" — prior sweeps had not explored r_on>0.22. Adhesion *range* (not amplitude `kadh`) appears to be the missing lever. Batch 2 sweep 2 retests this at the new relay-on parent.

**Clean falsifications recorded:** H6 (sense.gain has no hidden high optimum — flat across [30, 500]); H8 (spring.delta has no morphological effect).

**Established (new):** inflow.rate has an interior optimum near 0.5–1.0 (was 2.0). Confirmed by clean monotone-degradation above 1.5.

**Dropped from Batch 2:** spring.r0 (flat), spring.delta (falsified). Slots replaced by **relay.thr** and **relay.eps** to characterise the excitable mechanism we just confirmed is the strongest single lever.

Files updated: `dicty_loop_log.md` (16 sweep entries + summary), `dicty_knowledge.md` (Established / Falsified / Open updated, Batch-2 hypotheses listed), `specs/dicty_loop_base.yaml` (relay.gain=120 new parent), `sweep_plan.json` (16 sweeps for Batch 2 with hypothesis per sweep), `user_input.md` (acknowledged).


================================================================================
## BATCH 2 — 2026-06-17 16:13:58
================================================================================
Batch 2 read-back complete. Key results:

**Headline finding (and the new central question):** the global best of the batch is `inflow.rate=0.0` → loss=**0.312**, inner_mass=**0.561 ≈ REAL 0.606** — *but it satisfies the radial metric only by NOT receiving fresh cells* (n stays at 767 vs REAL's 1413). The model lacks any mechanism to retain newcomers in existing aggregates; uniform inflow scatters them.

**Falsifications (key updates to ledger):**
- H1: relay.gain does not improve monotonically — interior optimum near 150–250; gain≥500 degrades.
- H5: the Batch-1 inflow.rate=0.5–0.8 interior optimum does NOT persist at relay-on parent; the new optimum is 0.
- H7: high camp.diffusion does NOT enlarge catchment — it homogenises and destroys gradients.
- H10: sense.gain is NOT a saturated plateau — gain=10 is best.

**Supported:** spring.r_on remains the dominant coalescence control (2nd confirmation — promoted to **Established**); relay.thr interior optimum at 0.10–0.18; relay-on shifts kadh ceiling up to ~550–640.

**Actions taken:**
1. Appended 16 sweep entries + batch summary to `dicty_loop_log.md`.
2. Rewrote `dicty_knowledge.md` (new Established #3, new Falsified entries, new central Open Question, Batch-3 plan).
3. New parent in `specs/dicty_loop_base.yaml` = Batch-2 parent + `inflow.rate=0.0` (the actually-tested-best config).
4. **New mechanism added** to `dicty_ops.py`: `Inflow.bias_to_camp` — when >0, new cells are sampled from the cAMP grid weighted by `grid**bias` instead of uniformly. bias=0 = ablation.
5. Rewrote `sweep_plan.json` for Batch 3. The central experiment is Sweep 0 (bias_to_camp ∈ [0,16] at inflow.rate=1.5) testing whether biased inflow rescues n-growth without destroying inner_mass. Sweep 1 is a paired inflow.rate sweep at bias=5. Slots 2–15 refine the other params around the new parent.
6. Acknowledged `user_input.md` (no pending instructions).


================================================================================
## BATCH 3 — 2026-06-17 16:25:58
================================================================================
Batch 3 read-back complete. Summary of what I did and what changed:

**Critical finding — methodology fault.** `eval_sweeps.py` was ignoring the `"fixed"` per-sweep overrides in `sweep_plan.json`. This made Batch 3's central biased-inflow experiment (sweeps 0 and 1) **void** — `inflow.rate` stayed at the parent's 0, so the new `bias_to_camp` mechanism had no cells to bias. Patched at `eval_sweeps.py:81-86` (overrides now merged with the swept param).

**Batch 3 results.**
- **Global best (single-param-from-parent):** `camp.diffusion=0.02→0.0008` → loss 0.239, inner 0.510. Promoted "low diffusion preferred under relay" to Established Principle (3rd-batch confirmation).
- **Other wins beating parent's 0.312:** sense.gain=22, k_rep=40, relay.eps=0.07, mu_f=0.12, random_walk=0.018. None stacked into parent (untested in combination); all retested at new parent in Batch 4.
- **Falsified at new parent:** relay.gain interior optimum near 200 (now 120); kadh ceiling 200-700 (now 120); sense.gain optimum below 10 (now 22); secrete=15 (now 8). The sense.gain finding established a new tentative principle: **chemotaxis-gain optimum is regime-dependent**.
- **Established:** Relay necessity (gain=0 → loss 1.6, 5× worse), r_on dominance (3rd batch), low-diffusion preference (3rd batch), and the n-growth-gaming caveat for the loss metric.

**Batch 4 plan.** New parent = old parent with `camp.diffusion=0.0008`. Sweeps 0–2 re-attempt the biased-inflow mechanism (now that `"fixed"` works) at rate=1.5, rate=3.0, and rate-sweep-at-bias=5. Sweeps 4–16 narrow refines around Batch-3 wins. Dropped `cell.n` (saturated).

Files updated: `dicty_loop_log.md`, `dicty_knowledge.md`, `user_input.md`, `specs/dicty_loop_base.yaml`, `sweep_plan.json`, `eval_sweeps.py`.


================================================================================
## BATCH 4 — 2026-06-17 16:37:24
================================================================================
Batch 4 read-back complete. Key result: **the central scientific question of this arc is now answered negatively** — biased inflow is INSUFFICIENT to close the n-growth gap while preserving morphology (H1/H2/H3 all FALSIFIED across three sweeps with the patched harness). 11/16 sweeps had the parent value inside the range and the parent won them all; the four Batch-3 "candidate wins" (k_rep=40, mu_f=0.12, eps=0.07, random_walk=0.018) were single-seed noise and are RETRACTED. The parameter surface is exhausted.

**Updates made:**
- `dicty_loop_log.md`: appended 16 per-sweep entries + Batch 4 summary.
- `dicty_knowledge.md`: two new Established Principles (#7 biased-inflow falsified as sufficient; #8 parameter surface exhausted); five new Falsified entries; new central Open Question; Batch 5 plan written.
- `dicty_ops.py`: added two new mechanisms. `Inflow.edge_band` restricts newcomer positions to a boundary band of width `edge_band * world_width`; `Relay.{omega, amplitude}` adds a sinusoidal pacemaker drive to the FN activator. Both default to ablation (0) so parent is bit-identical to Batch 4 parent.
- `specs/dicty_loop_base.yaml`: parent UNCHANGED (loss=0.239 still the global best); only added the new params with ablation defaults.
- `sweep_plan.json`: 16 sweeps for Batch 5 — 4 testing the new mechanisms (necessity + sufficiency, each with a clear ablation in the range), 2 joint sweeps (r_on × boundary inflow; rate × boundary inflow), 1 dt sweep to test the vmax×dt aliasing hypothesis, 1 fine vmax valley scan, plus 8 narrow refinements at the new parent. Dropped: `bias_to_camp` (falsified), `relay.eps`/`relay.gain`/`k_rep`/`secrete.rate` (parent confirmed twice).
- `user_input.md`: acknowledged.

Two harmless temp files (`_b4_log.md`, `append_log.py`) remain in the dicty dir — bash `rm` was sandbox-blocked even within the allowed directory. Safe to delete manually.


================================================================================
## BATCH 5 — 2026-06-17 16:56:36
================================================================================
All 16 sweeps in place. Here's the summary:

**Batch 5 verdict: BOTH new mechanisms FALSIFIED; parent UNCHANGED for the third consecutive batch (loss=0.239, inner_mass=0.510).**

Key findings logged:
- **`inflow.edge_band`** (boundary-source inflow): falsified across 3 sweeps. Spatial restriction actively hurts; uniform-band ablation wins; rate=0 always best — strengthens Established #7 to mechanism-class statement.
- **`relay.{omega, amplitude}`** (pulsatile pacemaker): falsified at both low and high amplitudes. Pure-FN ablation wins.
- **`r_on=0.24`** pinned as exact inner_mass=REAL crossover (inner=0.611), but at +30% loss — confirms the new **Established #10**: single-attractor failure mode is the bottleneck.
- **`dt × vmax aliasing`** promoted from Open Question to **Established #9** (sweeps 8+9 + Batch 4 sw 12).

**Batch 6 design:**
- Added new mechanism `relay.{nucleate_rate, nucleate_amp}` to `dicty_ops.py` — Poisson-random activator pulses at random grid points, to break the single-attractor convergence and seed multi-mound morphology. Ablation = `nucleate_rate=0` = parent.
- Sweep_plan.json rewritten: 4 sweeps probe the new mechanism (rate at three amplitudes + amp curve), 3 sweeps re-enter the multi-knot `relay.thr` regime (with/without nucleation and with r_on joint), 1 sweep re-tests relay-necessity under nucleation, 1 sweep camp.decay × nucleation, 1 vmax iso-product walk to test the aliasing principle, 1 noise-floor seed sweep (H14) to quantify single-seed variance, plus joint tests with cell.n, camp.res, secrete.rate, random_walk.

Files updated: `dicty_ops.py`, `specs/dicty_loop_base.yaml`, `dicty_knowledge.md`, `dicty_loop_log.md`, `sweep_plan.json`, `user_input.md`.


================================================================================
## BATCH 6 — 2026-06-17 17:13:27
================================================================================
Batch 6 complete. Summary:

**Verdict:** Parent unchanged for 5th consecutive batch (loss=0.239, inner_mass=0.510, n_final=767).

**Key results:**
- `relay.nucleate_{rate,amp}` mechanism **fully FALSIFIED** across 6 joint sweeps (B6 sweeps 0,1,2,3,7,15). Field-side stochastic perturbation does not break the single-attractor — pulses get absorbed by the dominant self-organising central field.
- Sweep 4 (`relay.thr` alone, 0.16–0.34) is the **key positive observation**: morphology strip shows 3–6 discrete mounds at thr ∈ {0.20, 0.21, 0.25, 0.28} — qualitatively the closest to REAL across the entire batch. inner_mass spikes to 0.67–0.90, but radial-profile loss climbs to 0.5–2.4. Promoted to **Established #11**: *the model has the morphological capacity for multi-mound; the configured loss cannot reward it.*
- Sweep 13 (`seed`) failed (NaN — bad spec path); re-queued for B7.
- 11 new entries added to Falsified Hypotheses; 2 new Established Principles (#11 multi-knot regime exists; #12 field-side perturbation cannot break the attractor).

**B7 design:** new mechanism on the **cell side**: `sense_adapt` operator (per-cell chemotactic desensitization, `s_i ∈ [0,1]` decays in high cAMP, recovers in low). Replaces plain `sense` in parent schedule; ablation = `adapt_rate=0` recovers parent exactly. 16 sweeps: 4 map the new 3-param mechanism + saturation; 2 re-probe relay.thr (alone and × adapt); 4 joint tests with sense.gain, r_on, kadh, camp.decay; 1 corrected seed sweep (noise floor); diagnostic sweeps over random_walk and relay.eps (field-side analog); 1 multi-knot × cell.n. Test of sanity import was blocked — operator code follows the same pattern as `Relay` and uses verified `fld.sample`/`fld.gather_grad` APIs.


================================================================================
## BATCH 7 — 2026-06-17 17:30:00
================================================================================
Batch 7 verdict and Batch 8 plan committed.

**Batch 7 result (sense_adapt mechanism):** FULLY FALSIFIED across all 11 sweeps testing the mechanism. Every adapt_rate>0 caused cells to disperse into a sparse scattered field — the OPPOSITE of multi-mound. Parent (adapt_rate=0 ablation) wins decisively (loss=0.2388). Promoted to **Established Principle #13**.

**Positive findings:**
- Sweep 4 re-confirmed multi-knot regime at relay.thr ∈ [0.18, 0.30] (consistent 2-5 discrete mounds).
- Sweep 15 found new morphologically-best config: **relay.thr=0.25 × cell.n=1500 → loss=0.452, clean 3-6 mound morphology** (5× better than B7sw4 at same thr=0.25 because more cells per mound). Promoted to **Established Principle #14**.
- Sweep 13 (random_walk) and sweep 14 (relay.eps) re-confirmed parent values.

**Negative:** seed sweep (sw10) BROKEN AGAIN — eval_sweeps.py doesn't write root scalar params; noise floor still unmeasured. Workaround in B8 uses `cell.seed`.

**Batch 8 strategy:**
- Parent REVERTED: drop `sense_adapt` from schedule, restore plain `sense.gain=40`. Parent unchanged for the 7th consecutive batch.
- New mechanism added: **`align`** (per-cell polarity, Vicsek-style local velocity alignment + chemotactic bias) — tests whether STREAMING (a real Dicty observable absent from current model) restructures recruitment around line-shaped stream-merge events instead of radial pull. Implemented in `dicty_ops.py:425+`. Ablation = `strength=0` recovers parent exactly.
- 16 sweeps: 4 mapping the new align mechanism, 1 multi-knot re-confirm, 1 align×multi-knot joint, 5 diagnostic probes around the new multi-knot best point (thr=0.25, n=1500), 2 align×multi-knot joints, 1 cell.seed noise-floor workaround, 1 wide align saturation.

Files updated: `dicty_loop_log.md`, `dicty_knowledge.md` (added Est #13, #14; added 11 falsified hypotheses for B7; rewrote Open Questions around polarity/streaming; added Batch 8 planned hypotheses), `specs/dicty_loop_base.yaml`, `dicty_ops.py` (new `Align` op), `sweep_plan.json`, `user_input.md`.


================================================================================
## BATCH 8 — 2026-06-17 17:47:33
================================================================================
Batch 8 adjudication complete.

**Key findings**
- **align (Vicsek polarity) FULLY FALSIFIED** across 6 of 7 sweeps (0,1,2,3,5,12,13,15). Marginal 0.4% loss "win" at strength=0.22 is within noise and morphologically identical to parent (no streams, no flock, no multi-mound). Third successive cell-side mechanism falsification (after nucleation B6, sense_adapt B7).
- **NEW multi-knot best**: thr=0.25, n=1450, kadh=60, gain=140 → loss=0.343, inner=0.6 with clean 2-3 mounds (25% loss reduction vs B7 sw15). Parameter refinement inside the multi-knot regime IS paying off.
- **cell.seed sweep broke in a new way** (init_npz overrides initial scatter) — eval_sweeps.set_param patched in B9 to fix the root `seed` sweep properly.
- Parent **unchanged for the 7th consecutive batch** (loss=0.239).

**Batch 9 design** (4 files written)
- `dicty_loop_log.md` — 16 per-sweep entries + B8 summary.
- `dicty_knowledge.md` — 2 new Established Principles (#15 align falsified, #16 multi-knot refinement progress), 6 new Falsified Hypotheses (H1-H4-B8, H6-B8, H13-H14-B8, H16-B8), Open Questions re-reframed, Batch 9 hypotheses written.
- `specs/dicty_loop_base.yaml` — align removed from schedule (kept in code); new `inhib` field + `inhib_op` operator added with ablation values (gain=0, rate=0 → parent recovered exactly).
- `sweep_plan.json` — 16 sweeps: 4 map the new Gierer-Meinhardt activator-inhibitor mechanism (inhib_gain, inhib_rate, inhib.diffusion, inhib.decay), 1 inhib × multi-knot joint, 7 refine the multi-knot best point neighbourhood (r_on, kadh, gain, n, thr, camp.diff, RW), 1 inhib×multi-knot diffusion joint, 1 root seed sweep (noise floor), 1 inhib-without-relay clean Turing test, 1 strong-Turing regime.
- `dicty_ops.py` — added `InhibOp` (deposits inhib, anti-chemotaxis from inhib gradient).
- `eval_sweeps.py` — patched `set_param` to handle root `seed`, `n_frames`, and the new `inhib.*` field path; also routes op `seed` params as ints.


================================================================================
## BATCH 9 — 2026-06-17 18:07:20
================================================================================
Batch 9 closed. Summary:

**Headline result — first regime change in 8 batches.** The multi-knot regime with `spring.r_on=0.224` (sweep 5) gives loss=0.2594 with clean 2-3 mound morphology — statistically tied with the legacy single-blob parent (0.2388) under the seed-noise floor measured in sweep 13 (0.357-1.20, median 0.46). For the first time the model matches parent loss WHILE producing REAL-like multi-mound morphology.

**Inhibitor (Gierer-Meinhardt) FALSIFIED across 8 sweeps** (0, 1, 2, 3, 4, 12, 14, 15) — every parameter, every regime, with/without relay. Cells deposit faster than the field can pattern, so anti-chemotaxis = global dispersion, never Turing peaks. Mechanism removed from base spec (kept in code as ablation). This is the 4th falsified "missing-mechanism" candidate after nucleation, sense_adapt, align. **The morphology gap is closed by parameter joint refinement of EXISTING operators, not by a new mechanism.**

**Batch 10 setup written:**
- New parent (`specs/dicty_loop_base.yaml`): `thr=0.23, n=1400, kadh=75, r_on=0.224, gain=140, camp.diffusion=0.0004, random_walk=0.009`.
- New operator `persistence` added in `dicty_ops.py:561` — per-cell self-only motion memory, distinct from falsified `align` (no neighbour coupling, no chemotactic bias). Ablation = strength=0.
- 16 sweeps in `sweep_plan.json`: persistence necessity/sufficiency (sw 0, 1), fine joint refinements of r_on/kadh/thr/n/gain/D/rw at new parent (sw 2-9), seed-noise floor at new parent (sw 10), persistence joints (sw 11, 12, 15: inflow rescue, higher-thr, slow-memory), camp.decay / sense.gain / vmax re-tests (sw 13-15).
- Knowledge ledger updated with new Established #17 (multi-mound at parent loss), #18 (seed-noise floor quantified), #19 (inhibitor falsified), and 9 new Falsified-hypothesis entries.


================================================================================
## BATCH 10 — 2026-06-17 18:23:49
================================================================================
## Batch 10 summary

**256 sims run, 16 sweep figures inspected. Two structural findings:**

1. **Persistence FALSIFIED** as morphologically silent (sw 0, 11, 15 — three independent (strength, rho, thr) joints; strips visually identical across strength=0 → 0.12). **5th cell-side mechanism falsified** after nucleation, sense_adapt, align, inhibitor.

2. **Inflow rehabilitated** (sw 10: persistence×inflow.rate=2.4 → loss=0.2771, n=1410→1985 — the batch winner). First inflow win in 9 batches; partially retracts old Est #6/#7.

**Mid-batch metric change** (acknowledged from user_input.md, confirmed in `eval_sweeps.py`): loss is now `(1−SSIM_full-FOV) + 0.2·g(r) + 0.3·mound-count-pen + 0.5·velocity`, with real n_mounds≈8. Sim currently produces 2–3 mounds → the morphology gap is now **bigger, not smaller**, under the new metric. Old radial-profile-based principles (#2, #6, #7) marked superseded; mechanism findings retained.

**Updates made:**
- `dicty_loop_log.md`: 16 per-sweep entries + batch summary appended
- `dicty_knowledge.md`: Est #20 (persistence falsified), #21 (inflow rehabilitated), #22 (seed=0 unfavorable at new parent); 4 new falsified hypotheses; metric-change addendum prepended; Batch 11 plan
- `specs/dicty_loop_base.yaml`: new parent = B10 sw 10 winner config (multi-knot + inflow=2.4 + persistence; all axis-sweep refinements adopted)
- `sweep_plan.json`: B11 designed around (a) persistence decoupling, (b) inflow regime characterisation, (c) **mound multiplication probes** targeting n_mounds=8 (low kadh, low r_on, very low D, more cells, wider thr×kadh joint), (d) bias_to_camp / edge_band re-tested under new metric
- `user_input.md`: metric-change instruction acknowledged

**Strategic frontier for B11:** the cell-side mechanism well is exhausted (5 falsified); the new explicit n_mounds=8 target makes the morphology gap unambiguous. B11 probes whether weakening adhesion/catchment + boosting cell count can push mound count from 2–3 toward 8, before committing to a structural model change.


================================================================================
## BATCH 11 — 2026-06-17 18:40:58
================================================================================
Batch 11 complete. Summary:

**Adjudication (16 sweeps under new SSIM metric):**
- **Best loss: cell.n=800 → 0.9001** (also strong: inflow=7.5 → 0.9144; r_on=0.24 → 0.9341 with inner_mass=0.614 *exact* REAL match; sense.gain=80 → 0.9229). ALL within the seed-noise floor (sw 9: Δ=0.13, σ≈0.04).
- **The new SSIM-based loss is saturated** at ~0.92–1.03 across nearly every parameter axis — the model's 2-3-mound morphology vs REAL's ~8 mounds is the dominant residual and has no single-parameter lever.
- **Decoupling test resolved**: persistence FALSIFIED (sw 0 flat ablation); inflow rate FLAT-WITH-NOISE under new metric (sw 1, sw 14) — **Est #21 retracted**.
- **Two new boundaries**: secrete.rate ∈ [6, 10] (sw 10: catastrophic dispersion at 4-5); camp.decay ≤ 1.0 (sw 11); cell.n ≤ 3200 (engine NaN ≥ 3500).
- **Re-falsified at new metric**: inflow.bias_to_camp (sw 7), inflow.edge_band (sw 8).
- **Strongest signal**: spring.r_on monotone (sw 3) — r_on=0.24 best morphology, strengthens Est #3.

**Deliverables updated:**
- `dicty_loop_log.md`: 16 per-sweep entries + batch summary appended
- `dicty_knowledge.md`: added Est #23–#28 (flat surface, Est #21 retraction, persistence falsified, working-band boundaries, r_on lever, sense.gain plateau), Falsified entries for H1-B11–H7-B11 and H10-B11–H16-B11, central Open Question reframed (structural ceiling), Batch 12 plan added
- `specs/dicty_loop_base.yaml`: new B12 parent (r_on=0.24, n=800, no persistence, inflow=4, sense.gain=80, secrete=7, kadh=40)
- `sweep_plan.json`: 16 new B12 sweeps (seed noise floor; r_on × n joint; secrete dispersion mode probe; relay ablation; refractory eps; extended n_frames)
- `user_input.md`: B11 acknowledged


================================================================================
## BATCH 12 — 2026-06-17 18:56:23
================================================================================
Batch 12 (12/40) complete.

**What I did**
- Read all 16 sweep figures + best_montage + sweep_results.json + ledger + base spec + user input.
- Per-sweep entries appended to `dicty_loop_log.md` with response shape, best value, and a one-line morphological observation (every sweep produces a single central blob at this parent — the 2-knot morphology of B11 is lost at inflow=4).
- Ledger updated: added Established #29-33 (5 sharp failure-mode boundaries; retraction of Est #14 at sharp-r_on; n_frames not a lever; inflow×morphology coupling; high-thr multi-spot promising). Added 10 new falsified hypotheses (H3-B12..H16-B12). Added B13 plan + open questions about `sense_sat` and densifying high-thr multi-spot.
- `dicty_ops.py`: added `sense_sat` operator (Hill-saturated chemotaxis, instantaneous — distinct from falsified `sense_adapt`). Ablation = `c_sat=1e6` → identical to plain `sense`.
- `specs/dicty_loop_base.yaml`: new B13 parent — r_on=0.245, n=1000, vmax=0.062, inflow.rate=3.0 (B12 improvements within noise but on favourable edge); `sense_sat` replaces `sense` in schedule with c_sat=1e6 ablation.
- `sweep_plan.json`: 16 sweeps designed for B13 — seed recalibration; 3 sense_sat probes (c_sat wide, sat_n Hill exponent, c_sat × r_on=0.20 joint, c_sat × n=2000 joint); 2 multi-spot densification probes (r_on × thr=0.40 joint, thr × n=2000 joint); refinements (r_on, inflow, n, vmax); 4 saturation-on joints (diffusion, secrete, decay, relay.gain); 1 never-mapped (spring.r0).
- `user_input.md`: B12 read-back acknowledged.

**Best of B12**: r_on=0.245 → loss=0.8997, inner=0.485, n_final=1764. Within seed-noise σ≈0.18, no axis broke the single-blob ceiling — sharper falsification of "parameter tuning alone can close the morphology gap" (now 12 batches confirming).


================================================================================
## BATCH 13 — 2026-06-17 19:13:19
================================================================================
## Batch 13 — done

**Headline finding (the science):** `sense_sat` (Hill-saturated chemotaxis) — the sixth structural mechanism added — is the **first in 13 batches** to break the single-blob/2-knot ceiling under the SSIM metric. At `c_sat=0.20` the simulation produces **5-6 distinct compact mounds with inner_mass=0.662 vs REAL=0.606** (sw 1) — closest visual match to REAL the model has ever produced. The mechanism is biology-faithful: cells in established mounds saturate cAMP receptors → become non-responsive → outside cells form new mounds instead of joining.

**Surprising side-effects (sweeps 11, 13, 14):**
- Saturation **eliminates** the secrete.rate∈[3.6, 6.0] explosive-dispersion catastrophe (Est #29).
- Saturation **broadens** camp.decay's working band from [0.05, 0.40] to [0.05, 0.80].
- Under saturation, **relay.gain=0 still gives multi-mound** — partial retraction of Est #4 (relay necessity).

**Ledger updates:** added Est #34 (sense_sat breaks ceiling), #35 (sat_n densification axis), #36 (saturation regularizes failures), #37 (relay non-necessary under sense_sat). Falsified at B13: looser-adhesion-densifies (sw 3), cell.n-densifies (sw 4), and r0/inflow/cell.n single-axis refinements.

**New parent (B14):** `sense_sat.c_sat=0.20, sat_n=2, r_on=0.225, vmax=0.061, r0=0.018` — the first morphologically credible parent in 13 batches.

**B14 strategy:** densify the multi-mound spots toward REAL per-mound density via (kadh, r_on, sat_n) joints with c_sat=0.20, plus a clean relay ablation and a 2D c_sat × sat_n densification grid.


================================================================================
## BATCH 14 — 2026-06-17 19:30:41
================================================================================
Batch 14 complete. Key findings:

- **SSIM bias quantified (Est #42):** loss-winner (sw 2 sat_n=0.5, c_sat=0.20, loss=0.924, inner=0.453) is a washed-out diffuse cloud; morphology-winner (sw 2 sat_n=1.25, c_sat=0.20, loss=1.162, inner=0.777) is 5-7 dense compact mounds — closest visual match to REAL in 14 batches. Loss/morphology divergence is now 25% loss-penalty for the right answer.
- **Two multi-mound regimes (Est #38):** SPARSE at c_sat≤0.20 (B13's "breakthrough"); DENSE at c_sat∈[0.4, 1.0] (new). Monotone-up loss as c_sat decreases (sw 1, sw 15).
- **Densification lever found (Est #39):** sense_sat.gain monotone-densifies up to 240 (sw 12); B15 will extend to 600 to find saturation.
- **Sense_sat regularizes the field-side surface (Est #41):** camp.diffusion (sw 9), vmax (sw 14), random_walk (sw 13), relay.thr (sw 7) all flat under saturation — partial retraction of Est #5 + Est #9.
- **Est #37 tightened:** relay still NECESSARY at c_sat=0.20 (sw 6 ablation collapses to 1.27); sense_sat sufficient alone only at c_sat≤0.10.
- **New B15 parent (science-over-loss):** c_sat=0.20, sat_n=1.25, gain=240, kadh=15, r_on=0.20, decay=0.07, inflow=4. Flagged the SSIM bias to user as a metric question.

Files updated: `dicty_loop_log.md` (16 sweep entries + summary), `dicty_knowledge.md` (Est #38-42, falsifieds, B15 plan, Open Qs), `specs/dicty_loop_base.yaml`, `sweep_plan.json`, `user_input.md`.


================================================================================
## BATCH 15 — 2026-06-17 19:48:42
================================================================================
Batch 15 read-back complete. Summary:

**Catastrophic parent failure**: The B14 "science-over-loss" parent (sat_n=1.25 + joint adoption of B14 single-axis winners kadh=15, r_on=0.20, decay=0.07, inflow=4, gain=240) collapsed to dispersion at loss=11.78 — 10× worse than B14 parent. 14 of 16 sweeps could not adjudicate at this broken parent.

**Methodological lesson (NEW Est #43)**: Single-axis morphology peaks are NOT TRANSITIVE across joint changes. B14's "winners" recombined into a broken regime.

**Genuine breakthrough (sw 7 → NEW Est #45)**: At c_sat=0.50 + sat_n=3.0 (otherwise B15 parent), the model produces CLEAN 5-6 DENSE compact mounds, loss=1.10. This is the actual dense-multi regime and the B16 parent.

**New Established Principles**: #43 (peaks not transitive); #44 (sat_n is master regime switch — sat_n≥1.9 mandatory); #45 (dense-multi at sat_n=3.0 c_sat=0.50); #46 (kadh-sat_n coupling); #47 (relay was destructive at broken parent — re-test pending).

**B16 plan**: 16 sweeps re-testing every adopted axis at the new credible parent (c_sat=0.50, sat_n=3.0). Decisive experiments: sw 2 = relay.gain ablation (Est #4 vs #47), sw 3 = kadh (Est #46), sw 1 = noise floor at new parent. Plus (sat_n × c_sat) joint grids at c_sat={0.30, 1.0}.

Files updated: `dicty_loop_log.md` (B15 entries + summary), `dicty_knowledge.md` (Est #43–47, B15 falsifications, B16 plan), `specs/dicty_loop_base.yaml` (new parent), `sweep_plan.json` (B16), `user_input.md` (acknowledged with methodology flag).


================================================================================
## BATCH 16 — 2026-06-17 20:07:07
================================================================================
Batch 16 read-back done.

## Summary

**B16 BREAKTHROUGH: first morphologically robust parent + first genuine best-loss under SSIM.**

- **sw 0 (seed):** all 16 seeds produce clean 5-7 dense compact mounds at the B16 parent (sat_n=3.0, c_sat=0.50); σ=0.04 — lowest noise floor in 16 batches. Est #43 lesson (adopt whole-config morphology winner) validated.
- **sw 5 (sense_sat.gain):** monotone-down 40→500 then reversal 600→1000. **gain=500 → loss=0.9802 (best of project under SSIM)**. New Est #50.
- **sw 1 (relay.gain):** Est #4 (relay necessary) RE-CONFIRMED at the dense parent; Est #47 (relay destructive) FALSIFIED — it was a broken-parent artifact. Ringing band [30, 60] re-confirmed.
- **sw 4/12/13 (c_sat × sat_n):** trace a trade-off ridge — sat_n≥2.25 at c_sat=0.30, ≥2.75 at 0.50, ≥3.5 at 1.0; all yield loss ≈1.03–1.10. NEW Est #49.
- **sense_sat broadly regularizes** secrete/diffusion/decay/inflow failure modes (Est #29 dismantled) — **except vmax aliasing**, which returns at vmax≥0.072 (Est #51 nuance).
- **cell.n remains NOT a densifier** (sw 9, re-confirmed under strongest dense regime).

**B17 parent (conservative multi-axis adoption):** sense_sat.gain 240→500, secrete.rate 4→11, spring.kadh 15→10. All else unchanged.

**B17 sweep_plan.json:** 16 sweeps — seed validation (sw 1 = Est #43 transitivity check), peak refinements (gain, kadh, sat_n, c_sat, r_on), extended failure-mode boundaries (inflow to 40, decay to 0.85), the (c_sat, sat_n) ridge mapped via two columns (sw 12, 13), and two **densification joints** (gain × c_sat=0.30 sw 14; gain × kadh=6 sw 15) to test whether per-mound density can be pushed toward REAL.

Files updated: `dicty_loop_log.md`, `dicty_knowledge.md`, `specs/dicty_loop_base.yaml`, `sweep_plan.json`, `user_input.md`.


================================================================================
## BATCH 17 — 2026-06-17 20:29:59
================================================================================
Batch 17 readout complete.

**Headline findings (B17):**
- **NEW PROJECT BEST LOSS** under SSIM: sw 0 seed=8 → **0.9268** (multi-mound preserved), beating B16's 0.9802.
- **Est #53 (new) — DENSIFICATION AXIS FOUND.** sw 13 (gain × c_sat=0.30) is MONOTONE-DOWN from gain=100 (1.13) → gain=1000 (0.9823); the sparse multi-mound regime densifies under high gain, with apparent mound count rising toward 7-8. First axis-level mechanism signal below parent loss since B16 sw 5.
- **Est #50 RETRACTED → Est #54.** B16's gain=500 peak with reversal at 600+ was regime-specific (sw 1 flat under joint parent).
- **Est #55-56 (new) — sense_sat regularization extends to relay** (ringing band [30,60] dismantled) and to most field-side failure modes (inflow flat to 40, decay flat to 0.85). Only **vmax aliasing** persists (sw 15 reconfirms Est #9/#51).
- **Est #57 (new) — (c_sat, sat_n) ridge quantified** through {(0.30, 2.0), (0.50, 3.0), (1.0, 3.5), (2.5, 4.0)}.
- Structural ceiling: mound count still 5-7 across all 16 sweeps; REAL=8 still unreached.

**Files updated:**
- `dicty_knowledge.md` — added Est #53-57, B17 H1-H16 falsified/supported entries, two new Open Questions, full B18 plan.
- `_b17_log.md` — 16 per-sweep entries + batch summary (staged for append to `dicty_loop_log.md`; the append script needs one user approval — `python3 append_log.py`).
- `specs/dicty_loop_base.yaml` — single conservative change: cell.n 1000 → 1800.
- `sweep_plan.json` — 16 new B18 sweeps focused on pushing the densification axis (gain × c_sat ∈ {0.20, 0.30, 0.40} up to gain=7000), alternative-parent grid (sw 3), joint re-tests inside the densification regime, extended failure modes, and a direct mound-wavelength probe via camp.diffusion (sw 16).
- `user_input.md` — B17 acknowledgment with flag about Est #42 status (SSIM/morphology divergence reduced; structural ceiling now the only gap).


================================================================================
## BATCH 18 — 2026-06-17 20:50:49
================================================================================
B18 batch complete. 

**Headlines:**
- **256 sims, no 8-mound breakthrough** — the 5-7 mound ceiling holds across all 16 sweeps × 16 values.
- **Best loss = 0.9167 at cell.n=3200 (sw 9)**, marginal over B17 best=0.9268 within seed noise (σ=0.04).
- **Failure-mode dismantling continued dramatically (Est #59)**: camp.decay plateau [0.04, 2.0+], inflow [0, 150], kadh [6, 280], c_sat [0.20, 1.2] — the dense regime is extraordinarily robust.
- **New wall discovered (Est #60)**: camp.diffusion CATASTROPHIC at D≥0.05 (sw 15) — the B16 sw 11 "flat to 0.08" missed this.
- **Est #61**: sat_n=3.0 sits at the upper edge, not the center, of the productive plateau ([1.9, 2.5]).
- **Est #62**: secrete.rate degrades morphology monotonically past rate=14; safe band [8, 14].
- **Est #58**: parameter densification at c_sat=0.50 confirmed EXHAUSTED — extreme gain pushes (sw 1/13/14 to 7000) saturate at 2000-3000 and DEGRADE to sparse-tiny past 3500.

**B19 mechanism added:** `secrete_het` operator in `dicty_ops.py:666+` — per-cell heterogeneous secretion via log-normal multiplier (E[m_i]=1, shape `het_std`). Ablation `het_std=0` ≡ plain secrete. Distinct family from all 5 falsified cell-side mechanisms (gain, polarity, persistence, etc.) — this is per-cell SOURCE strength. Biological rationale: Dicty cells are not synchronous, spaced seed nuclei from per-cell variability may break the 7-mound ceiling.

**B19 sweep plan**: 16 sweeps — seed validation (sw 0), het_std necessity+sufficiency with ablation (sw 1), three DECISIVE het_std joints with each densification axis (c_sat=0.30 / gain=1500 / cell.n=3000), two more het joints (kadh, r_on), het_seed sweep at het_std=0.5, high-het seed sweep, het × relay.gain=300, plus refinements: sat_n FINE in productive band, camp.diffusion FINE around D=0.05 wall, secrete.rate FINE, vmax FINE, cell.n FINE, relay.gain FINE.

Files updated: `dicty_loop_log.md` (16 sweep entries + summary), `dicty_knowledge.md` (Est #58-#62, B18 falsified hypotheses, B19 planned section), `specs/dicty_loop_base.yaml` (added `secrete_het` op, het_std=0 ablation), `sweep_plan.json` (16 new sweeps), `dicty_ops.py` (new `SecreteHet` class), `user_input.md` (B18 acknowledgement).


================================================================================
## BATCH 19 — 2026-06-17 21:14:29
================================================================================
Batch 19 read-back complete.

**Key findings (B19 / 256 sims):**

- **NEW project best: sw 5 `sense_sat.sat_n=2.1` → loss=0.9126** (was B18 0.9167). Refines Est #61 productive plateau center.
- **`secrete_het` FALSIFIED as a mound-multiplier** across SEVEN sweeps (1, 2, 3, 4, 10, 11, 15). At every joint (parent, `c_sat=0.30`, `gain=1500`, `cell.n=3000`, `kadh=20`, `r_on=0.215`, `relay=300`), ablation `het_std=0` wins or ties; non-zero `het_std≥0.7` monotone-up loss and disperses per-mound density. Sw 14 seed sweep at `het_std=1.0` confirms degraded regime across all 16 seeds. **Sixth cell-side mechanism falsified** — the cell-side family (per-cell GAIN/POLARITY/SOURCE STRENGTH) is now exhausted.
- **Wall refinements:** Est #60 `camp.diffusion` wall is at `D~0.07` (not 0.05); `vmax` wall is at `>=0.075` (not 0.072); `relay.gain` plateau tightened to `[100, 280]` with monotone-up degradation past 320.
- **Engine no longer NaN at cell.n=3400** (B18 buffer fix took effect).

**B20 design:**

- **Parent:** single conservative change `sat_n: 3.0 → 2.1`. Reverted `secrete_het → plain secrete` (no reason to keep a falsified noise source).
- **NEW operator `decay_dens`** added to `dicty_ops.py` — per-mound density-coupled cAMP decay (`c -= dens_coeff · ρ_norm · c · dt`), ablation at `dens_coeff=0`. First FIELD-SIDE structural mechanism after the cell-side family was exhausted. Distinct from inhibitor / pulsatile / nucleation (all falsified): local, deterministic, modulates existing cAMP turnover.
- **16 sweeps:** sw 0 seed/transitivity validation; sw 1 `dens_coeff` necessity+sufficiency `[0, 40]`; sw 2/3/6 `dens_coeff` joints with the three densification axes (`c_sat=0.30`, `gain=1500`, `cell.n=2500`); sw 5 `sat_n` FINE; sw 14 high-`dens_coeff=1.0` seed sweep (analogue of B19 sw 14); sw 7/8/9/10 plateau re-pins (relay/vmax/secrete/D); sw 11/12 dens_coeff×decay & ×kadh; sw 13 cell.n re-test; sw 12 c_sat ridge; sw 15 `sat_n × dens_coeff=1.0`.

Files updated: `dicty_knowledge.md` (Est #63-67, B19 falsified entries, B20 planned), `specs/dicty_loop_base.yaml`, `dicty_ops.py` (added `DecayDens`), `sweep_plan.json`, `_b19_log.md`, `user_input.md`. The append-script `append_log.py` is configured to fold `_b19_log.md` into `dicty_loop_log.md` (execution blocked here, but the per-batch convention `_bN_log.md` is preserved).


================================================================================
## BATCH 20 — 2026-06-17 21:37:37
================================================================================
Batch 20 / 40 complete.

**Result — DECAY_DENS FALSIFIED.** The B20 256-sim batch produced the seventh project mechanism falsification (third field-side) and a marginal new project best.

**Verdicts (16 sweeps):**
- sw 1, 2, 3, 5, 10, 11, 14, 15 → `decay_dens` **falsified** universally. dens_coeff ≥ 1.2 annihilates the cAMP field at every joint (parent, c_sat=0.30, gain=1500, cell.n=2500, camp.decay=0.20, kadh=20); ablation always wins or ties.
- sw 0, 4, 6, 9, 12, 13 → plateau re-pins (sat_n=2.1, relay.gain=140, D=0.0012, n=1800, c_sat=0.50) all confirmed within seed noise.
- sw 7 → vmax aliasing landscape is **broader** under sat_n=2.1 (Est #69 new).
- sw 8 → **new project best loss = 0.9111** at secrete.rate=9 (marginal, within seed noise).

**Knowledge ledger:** added Est #68 (decay_dens falsified), #69 (vmax landscape), #70 (sat_n=2.1 plateau width), #71 (cell.n no capacity wall); B20 falsifications + open-Q close-out written.

**B21 design:**
- Parent updated: secrete.rate 11 → 9; `decay_dens` removed from schedule; NEW `diff_dens` operator added at ablation (kappa=0) — density-modulated diffusion (D_eff = D0/(1+κρ)), the opposite failure-mode profile from decay_dens (preserves field mass, sharpens gradients).
- 16 sweeps: kappa necessity+sufficiency + 4 joints (c_sat=0.30, gain=1500, cell.n=2500, relay.thr=0.30, kadh=20, camp.diffusion=0.005) + sat_n FINE + secrete FINE + vmax FINE in B20 dips + seed sweeps at kappa=0 and kappa=2.0 (decisive analogue of B20 sw 14).
- `dicty_ops.py` gained DiffDens class; spec schedule reads `[spring, secrete, diff_dens, camp.diffuse, relay, sense_sat, inflow, random_walk, integrate]`.

**Note:** `dicty_loop_log.md` append via `python3 append_log.py` was blocked at runtime; B20 entries are in `_b20_log.md` (canonical, matches `_b18_log.md`/`_b19_log.md` convention) with a one-line marker in the main log.

**Flag for user:** if `diff_dens` also fails, only one field-side candidate (density-triggered pulse) remains before the structural-ceiling diagnosis becomes load-bearing and the next escalation is engine-change (3D / finer grid / soft particles) or metric augmentation.


================================================================================
## BATCH 21 — 2026-06-17 22:01:23
================================================================================
All 16 sweeps in place. **Summary of B21 batch 21/40:**

**Headline finding — B21 INVALID for diff_dens.** All 6 diff_dens.kappa sweeps (sw 1, 2, 3, 5, 10, 11, 15) returned BIT-IDENTICAL losses across all 16 kappa values [0, 120]. Root cause: `DiffDens.forward` in `dicty_ops.py:918` read `getattr(fld, "diffusion", 0.0)` to get D0, but `GridField` stores diffusion as `self.D` (`grid_field.py:31`) — so D0=0 → silent no-op.

**Actions taken:**
1. **Fixed** `dicty_ops.py:918` — fallback `getattr(fld, "D", getattr(fld, "diffusion", 0.0))`.
2. **`_b21_log.md`** with per-sweep entries + summary; marker appended to main log.
3. **6 new Est** (#72-#77) added to ledger: bug record, secrete=9 noise floor σ≈0.11 (~2.5× wider than secrete=11), relay.gain plateau extends to 320, sat_n plateau broadens to [1.7, 3.2], camp.D wall transition starts at D=0.045, secrete catastrophe edge at rate=7.5.
4. **`user_input.md`** acknowledged with full B21 readback.
5. **`specs/dicty_loop_base.yaml`** preserved (only one change is the operator patch); B22 parent loss 0.9111.
6. **`sweep_plan.json` rewritten** for B22: 11 slots on the FIXED diff_dens × maximally-diverse joints (c_sat=0.30, gain=1500, kadh=20 [B21 morphology winner], relay.thr=0.30, cell.n=2500, camp.D=0.005, inflow=1.5, sat_n=2.5 + 2 high-kappa seed sweeps) + 5 plateau refinements (sat_n FINE, secrete FINE low-end, vmax avoid-spike, camp.D wall ladder). Duplicate seed slot dropped.

**SIDE-FINDING worth noting:** sw 11 (kadh=20, kappa silent) produced visually the BEST 5-6 crisp multi-mound morphology of the entire batch at loss=1.25 — SSIM-penalised per Est #42. Promoted to a B22 kappa-joint as the morphology probe.


================================================================================
## BATCH 22 — 2026-06-17 22:24:13
================================================================================
## Batch 22 — DECISIVE FALSIFICATION of `diff_dens`

**Result:** `diff_dens` (density-modulated diffusion) is the **8th project mechanism falsified / 4th field-side**, joining pulsatile relay (B5), inhibitor (B9), and `decay_dens` (B20). With the `dicty_ops.py:918` bug patched (D0 now reads `fld.D`), every kappa>0 is catastrophic at the parent and at 7 productive joints — field annihilation, same failure mode as `decay_dens`. Two seed sweeps (kappa=2.0, kappa=20) confirm deterministic, not seed-luck: 32 seeds, distributions disjoint from ablation.

- One exception (sw 7, kappa × camp.D=0.005): kappa>0 partially rescues an over-diffusive baseline back to 5-7 mounds at loss=1.03, but never reaches parent 0.911. Recorded as Est #79 — self-cancelling pair, not a mechanism.
- Plateau refinements re-confirm parent: sat_n=2.1, secrete.rate=9≡11 (tied at 0.911 within noise), vmax=0.060, camp.D working band [0.0001, 0.036] with ringy transition [0.038, 0.055].
- **No new project best.** Parent loss = 0.9111 (5 ties).

**B23 parent / plan:**
- Revert `secrete.rate` 9→11 (Est #73: recovers σ≈0.04 noise floor vs σ≈0.11 at rate=9; B22 sw 13 confirms ties).
- NEW operator `pulse_dens` added to `dicty_ops.py` — density-triggered local cAMP burst (SOURCE-SIDE + DETERMINISTIC + LOCAL + DENSITY-CONDITIONED), the unexplored quadrant after `decay_dens` (sink, falsified), `diff_dens` (transport, falsified), nucleation B6 (random), and pacemaker B5 (global).
- 16 sweeps: validation seed + amplitude necessity + threshold + 5 joints + high-amp seed sweep + plateau re-pins.
- **Last untested structural candidate in the 2D point-cell engine.** If pulse_dens also fails, B24 escalates to engine change or metric augmentation (Est #42 flag has been live since B14).

Files updated: `_b22_log.md` (16 per-sweep entries + summary), `dicty_loop_log.md` (one-line marker), `dicty_knowledge.md` (Est #78/#79 added; 5 H-B22 hypothesis verdicts in Falsified; new Open Question central B23; new B23 planned section), `specs/dicty_loop_base.yaml`, `sweep_plan.json`, `user_input.md`, `dicty_ops.py` (PulseDens class).


================================================================================
## BATCH 23 — 2026-06-17 22:47:04
================================================================================
B23 adjudication complete. **pulse_dens decisively falsified** (Est #80) — ninth project mechanism, fifth field-side. Across all 7 amplitude sweeps + threshold sweep + high-amplitude seed sweep, ablation wins; every amp≥0.05 monotone-up loss with cells dispersing rather than nucleating new mounds (same failure family as decay_dens and diff_dens). The 16 seeds at amp=2.0 lie disjointly above ablation [1.14, 1.20] vs [0.91, 1.00] — catastrophe is deterministic, not seed-luck. Plateau sweeps (sw 9-15) reconfirm B23 parent under the recovered σ≈0.04 noise floor; no new project best.

**Operator-side mechanism family is definitively exhausted** — 10 mechanisms falsified across 23 batches. B24 forks strategy: METRIC AUGMENTATION first (`eval_sweeps.py` to report `morph_score` alongside loss), with sweeps focused on the c_sat=0.30 sparse densification column + four engine-resolution probes (`camp.res`, `n_frames`, `dt`) that have never been systematically tested. If the augmented metric is also flat across the parameter cube, B25+ forks to the `dicty_engine_mpm` engine prototype.

Files updated: `_b23_log.md` (per-sweep entries + summary), `dicty_loop_log.md` (B23 marker), `dicty_knowledge.md` (Est #80 + falsified H2-B23..H9-B23 + Batch 24 planned), `specs/dicty_loop_base.yaml` (pulse_dens/diff_dens dropped from schedule, buffer extended to 6500), `sweep_plan.json` (16 B24 sweeps), `user_input.md` (B23 acknowledged with metric-augmentation flag for user).


================================================================================
## BATCH 24 — 2026-06-17 23:15:13
================================================================================
Batch 24 read-back complete. Two decisive findings + one meta-failure, all set up for B25.

**B24 verdict:**
- **Est #81 (NEW):** Engine resolution does NOT bind the ceiling. TWO independent `camp.res` sweeps (sw 5, 15) + `dt` sweep (sw 7) show mound count INVARIANT in working bands. The 5–7 mound ceiling is mechanism-bound, not grid-bound.
- **Est #82 (NEW — smoking gun):** The point-cell engine has NO STABLE MULTI-MOUND ATTRACTOR. n_frames sweeps at TWO regimes (sw 6 parent, sw 14 densification) show monotone-UP loss + monotone-UP inner_mass→0.83 as n_frames extends to 1600; morphology collapses to a single tight point by n_frames=1000+. Multi-mound is a TRANSIENT.
- **Est #83/84/85:** sat_n plateau at c_sat=0.30 = [1.5, 2.7]; n>=3800 introduces unrecruited halo; r_on=0.23 at densification joint produces visible 4-mound morphology (candidate morph_score winner).
- **META-FAILURE:** the morph_score metric augmentation promised by the B24 plan was NOT implemented; central B24 fork question UNADJUDICATED.

**B25 setup (committed):**
- **(α) `eval_sweeps.py`** — IMPLEMENTED `morph_score`, `n_mounds`, `peak_frac` reporting per value, including `best_morph_value`/`best_morph_score` per sweep. Adjudicates Est #42 flag.
- **(β) `dicty_ops.py`** — NEW operator `density_repel` (density-saturating short-range repulsion via tanh gate; pushes cells down-gradient when local ρ>thr). Biologically: finite cell volume. Ablation `strength=0`.
- **`specs/dicty_loop_base.yaml`** — B25 parent, unchanged parameters + `density_repel` ablation in schedule.
- **`sweep_plan.json`** — 16 B25 sweeps: sw 1 necessity, sw 3 thr sweep, sw 4–6/16 joints, **sw 7 DECISIVE Est #82 break test** (density_repel × n_frames=1200), sw 8/14 seed sweeps, sw 9/10/11/12/13/15 plateau re-pins under morph_score (refining r_on=0.23 4-mound candidate).
- **`dicty_loop_log.md`** + **`_b24_log.md`** + **`dicty_knowledge.md`** + **`user_input.md`** updated.

**Next decision point:** if `density_repel` fails (11th mechanism falsified), B26 escalates to the full engine fork (`dicty_engine_mpm.py`).


================================================================================
## BATCH 25 — 2026-06-17 23:42:52
================================================================================
**Batch 25 read-back complete. The breakthrough is on the metric, not the mechanism.**

**What B25 found:**
- `morph_score` got implemented (carried over from B24's missed deliverable) and **immediately found 8-mound configurations in NINE of 16 sweeps** with morph≤0.005. The Est #42 SSIM/morphology divergence flag — live since B14 — is finally adjudicated: the parameter-flatness was metric-induced. **Est #58 5-7 mound STRUCTURAL CEILING — RETRACTED as metric artefact (Est #87).**
- `density_repel`: **SUFFICIENT but NOT NECESSARY**. Productive band [0.02, 3.5] (sw 1, 3); ablation reaches nm=8 at multiple alternative parents (sw 4, 8, 10, 15).
- `density_repel` **RESCUES the high-cell.n single-blob collapse** (sw 5: nm=1 → nm=9 at cell.n=3000) — first clean productive role (Est #89).
- `density_repel` **FAILS the Est #82 break test** at n_frames=1200 (sw 6): runaway compaction completes regardless of strength (Est #90). MPM engine fork remains live.
- **New project-best morph_score ≈ 0** at sw 15 (kadh=20, strength=3.5, nm=8, loss=0.9381).

**Files updated:**
- `_b25_log.md` — full per-sweep entries + summary.
- `dicty_loop_log.md` — B25 marker appended.
- `dicty_knowledge.md` — Est #86-91 added; B25 metric change log entry; Falsified Hypotheses extended with H1-H16-B25 entries; Open Questions rewritten for B26; B26 planned hypotheses block prepended.
- `specs/dicty_loop_base.yaml` — new B26 parent (`spring.r_on` 0.20 → 0.19).
- `sweep_plan.json` — 16 B26 sweeps: (α) 8-mound manifold mapping (sw 1-4, 13-15), (β) re-evaluation of 4 falsified mechanisms under morph_score (sw 9-12), (γ) Est #82 re-test at new parent (sw 7-8) + density_repel productive-band refinement (sw 5-6) + camp.D wall under morph_score (sw 16).
- `user_input.md` — B25 acknowledged; user flagged for the B26 "re-litigation" framing and the MPM-fork escalation contingency.


================================================================================
## BATCH 26 — 2026-06-18 00:10:59
================================================================================
B26 complete. Summary:

**Decisive findings (Est #92-96):**
- **Prong α (8-mound manifold)** — STRONGLY SUPPORTED. Every informative sweep produced nm=8 with morph_score≤0.001 at multiple values. Joint transitivity holds across (r_on × kadh), (r_on × c_sat), (r_on × kadh × gain). The B15 Est #43 joint-collapse concern is dissolved under morph_score. Cleanest 8-mound corner: (r_on=0.19, kadh=20, gain in [750, 2500]).
- **Prong γ (Est #82 runaway compaction)** — RECONFIRMED. sw 7 (n_frames at r_on=0.19) reproduces the collapse trajectory; sw 6 (density_repel × n_frames=1200, thr=0.5) shows NO strength halts it. density_repel is NOT a finite-volume fix in any form.
- **Est #93** — B25's density_repel rescue at high cell.n narrows to n≤3000.

**Critical bug (Est #95):** Prong β invalid. 4 of 16 sweeps (sw 8/9/10/11) returned **bit-identical** results because secrete_het/decay_dens/pulse_dens/diff_dens weren't in the spec's operators/schedule (kept "in code as ablations" only). `eval_sweeps.set_param` is a silent no-op when the named operator isn't scheduled. SECOND silent-op bug after B21 Est #72.

**Project-best loss UNCHANGED at 0.9126** (sw 1 r_on=0.20).

**B27 plan:**
- **Spec fix**: secrete_het/decay_dens/pulse_dens/diff_dens now scheduled at ablation defaults so prong-β can finally run (sw 3-6).
- **Parent reverts** r_on 0.19→0.20 (Est #91 reclassified as plateau across [0.195, 0.20]; loss-tied with project best).
- **Drill the 8-mound corner** at (r_on=0.19, kadh=20) via sw 9, 10, 16.
- **adh_cap** (mass-cap adhesion, next Est #82 candidate) deferred to B28 pending operator code.

Files written: `_b26_log.md` (canonical per-sweep entries), `_b26_marker.txt` (one-line summary; main-log append blocked at runtime as in prior batches), `dicty_knowledge.md` (Est #92-96 + B27 plan section), `specs/dicty_loop_base.yaml` (B27 parent + 4 newly scheduled ops), `sweep_plan.json` (16 B27 sweeps), `user_input.md` (acknowledged with 2026-06-18 timestamp).


================================================================================
## BATCH 27 — 2026-06-18 00:32:41
================================================================================
B27 batch complete.

## Summary

**Project-best loss UNCHANGED at 0.9126.** The decisive findings:

- **Est #97 NEW** (strongest finding): sw 2 `secrete_het.het_std=0.20` produced `nm=8 morph=0.0001 loss=0.9152` — within seed-noise of parent. Est #63 (B19 falsification by SSIM-loss) **partially retracts**. Pending 16-seed verification.
- **Prong (β-redo) otherwise holds:** decay_dens, pulse_dens, diff_dens all reconfirmed falsified under morph_score with the Est #95 spec fix.
- **Est #98 NEW:** (r_on=0.19, kadh=20, gain ∈ [750, 3500]) is a 4-point morph_score=0 corridor — the cleanest 8-mound manifold subregion of the project.
- **Est #100 NEW (decisive):** Est #82 runaway compaction **worsens** at the 8-mound corner — collapse by `n_frames=750` vs `1050` at parent. The 8-mound morphology is a transient, never an equilibrium.

**B28 plan (16 sweeps):** sw 0 is the DECISIVE 16-seed verification of Est #97 at `het_std=0.20`; sw 1–6 narrow het FINE + densification joints + het × n_frames=1200 (decisive Est #82 break test under het-on); sw 7–9 drill the 8-mound corner FINER; sw 10–12 plateau re-pins; sw 13–15 break tests; sw 16 is the adh_cap PILOT (kadh × cell.n=3000 × n_frames=1200) — if low kadh produces stable multi-mound, the mass-cap-adhesion mechanism is on track for B29 implementation.

**B28 parent UNCHANGED** pending verification (B26 Est #95 caution applied). adh_cap operator implementation deferred to B29.

Files updated: `_b27_log.md` (new), `dicty_loop_log.md` (B27 marker), `dicty_knowledge.md` (Est #97–100 + B28 planned hypotheses), `specs/dicty_loop_base.yaml` (B28 header), `sweep_plan.json` (B28 16 sweeps), `user_input.md` (B27 ack).


================================================================================
## BATCH 28 — 2026-06-18 01:02:45
================================================================================
B28 batch processed. Summary:

**Two decisive negatives:**
- **Est #97 RETRACTED (Est #101 NEW):** 16-seed verification at `secrete_het.het_std=0.20` (sw 0) failed — distribution matches parent noise (σ≈0.036), but morph_score WILD: only 4/16 seeds at morph≤0.05. The B27 sw 2 nm=8 morph=0.0001 was 1/16 seed-luck. Across sw 1-4, ablation wins or ties on loss at every joint. B19 Est #63 (secrete_het falsified) stands reinstated under BOTH metrics.
- **Est #102 NEW (adh_cap pilot FALSIFIED):** sw 15 spring.kadh × cell.n=3000 × n_frames=1200 — kadh=0 explodes, kadh ∈ [1, 200] universally collapses to nm=1. Low kadh does NOT halt Est #82 because chemotactic pull dominates regardless of adhesion. The adh_cap design (spring-side attenuation) cannot succeed.

**Reconfirmations:** Est #98 corner broadened (gain [600, 3500], kadh [10, 22], NEW r_on candidate [0.220, 0.225]); Est #100 generalized (corner collapse is gain-invariant ~750 frames); pulse_dens (Est #80) failure is amplitude-driven not threshold-driven (sw 13).

**Project-best loss = 0.909 at sw 0 seed=5 het=0.20** — within parent noise; marginal, not adopted.

**Files updated:**
- `_b28_log.md` (canonical per-sweep entries + summary)
- `dicty_loop_log.md` (B28 marker appended)
- `dicty_knowledge.md` (Est #97 marked RETRACTED, Est #101 + #102 added, Est #100 generalized; B28 planned section moved to ADJUDICATED; B29 planned section added)
- `specs/dicty_loop_base.yaml` (B29 parent: secrete_het.het_std → 0)
- `sweep_plan.json` (B29 16 sweeps — final point-cell closing batch)
- `user_input.md` (B28 acknowledgment with B29 strategic frame)

**B29 frame:** final point-cell closing batch — sense_sat.gain attenuation × n_frames=1200 is the last operator-side Est #82 candidate (chemotaxis SOURCE attenuation since Est #102 ruled out sink attenuation); new r_on=0.222 corner replication; Est #99 D=0.0042 replication. MPM engine fork queued for B30 if B29 closes flat.


================================================================================
## BATCH 29 — 2026-06-18 01:50:09
================================================================================
Batch 29 adjudication complete.

**Summary — B29 SURPRISE: FIRST POINT-CELL Est #82 PARTIAL RESCUE**

- **NEW Est #104** (sw 4): the 8-mound corner (r_on=0.19, kadh=20, gain=1500) + camp.decay=1.4 sustains nm=6, loss=0.9863 at n_frames=1200 — first point-cell config to break the runaway compaction since B24. Single-seed; requires 16-seed verification in B30. This DEFERS the MPM engine fork.
- **NEW Est #105** (sw 6): camp.diffusion=0.0042 verified across 16 seeds (4/16 morph=0). **B30 PARENT adopts D=0.0012→0.0042** (single-axis change).
- **Four new Est #82 falsifications** (Est #106-108, #110): sense_sat.gain attenuation (12th project mechanism), random_walk, vmax, stacked density_repel × corner — all fail at n_frames=1200. Operator-side rescue family is closed; Est #104 is the sole remaining mitigation candidate.
- **Est #103**: new r_on=[0.220, 0.225] corner falsified as corridor; **Est #111**: Est #100 generalized (all 8-mound corners are kinematic transients); **Est #109**: low-decay morph niche at decay=0.02 (single-seed, needs verification).
- Project-best loss 0.9066 (sw 5 seed=7 r_on=0.222) — marginal, within seed noise, NOT adopted.
- **B30 plan**: 16 sweeps across three prongs — (μ) Est #104 16-seed verification + mechanism-edge probes (sw 0-4); (ν) D=0.0042 new parent refinement (sw 5-10); (ξ) Est #82 adjudication probes including Est #109 verification (sw 11-15). Decisive sw 0: if ≥3/16 seeds replicate at Est #104, point-cell engine has a real mitigation; if 0-2/16, B31 commits to MPM fork.

Files updated: `dicty_knowledge.md`, `dicty_loop_log.md` (+ `_b29_log.md` canonical), `user_input.md`, `specs/dicty_loop_base.yaml`, `sweep_plan.json`.


================================================================================
## BATCH 30 — 2026-06-18 02:44:26
================================================================================
B30 complete. Summary:

**TWO DECISIVE FINDINGS:**

1. **Est #104 RETRACTED as METRIC ARTEFACT (NEW Est #112).** sw 0 (16-seed verification at the Est #104 corner+decay=1.4+n_frames=1200) numerically passed (12/16 nm≥4) but morphology strip showed SINGLE CENTRAL BLOB + halo speckle at every seed — peak detector counting halo as "mounds". sw 2 (n_frames at Est #104) revealed the configuration is just a DELAYED TRANSIENT — collapse trajectory identical to Est #82 just delayed ~400 frames. The MPM fork is no longer "deferred by Est #104".

2. **D=0.0042 NEW PARENT IS THE BREAKTHROUGH (NEW Est #113).** sw 5 (16-seed at bare new parent) shows GENUINE multi-mound morphology (4-8 dense compact mounds at every seed) — the FIRST point-cell parent in 30 batches to do so. Tightest noise floor of project (σ≈0.026). Est #109 promoted to Est #114 (decay=0.02 verified). Est #98 r_on corner DISSOLVES under new D (sw 7 broadly productive). PROJECT-BEST loss 0.9055 at kadh=9 (single-seed; needs verification).

**B31 PARENT: unchanged** (no single-axis change verified across 16 seeds with morphology support).

**B31 CENTRAL TEST:** n_frames sweep at the BARE NEW PARENT (sw 0) — does the multi-mound regime SURVIVE to n_frames=1200, or is it (like Est #104) just a delayed transient? This single test adjudicates the engine-level question: if it survives, Est #82 is mitigated by structural D shift and MPM fork is DEFERRED INDEFINITELY; if it collapses, MPM fork is the principled escalation.


================================================================================
## BATCH 31 — 2026-06-18 03:30:47
================================================================================
B31 read-back complete. All deliverables written.

**Summary of B31 outcome and B32 plan:**

- **DECISIVE CLOSURE: Est #113 RETRACTED.** sw 0 n_frames sweep at the bare new parent (D=0.0042) shows the exact Est #82 collapse trajectory (nm 10→6→3→1 by n_frames=480, inner_mass monotone 0.351→0.895). The D=0.0042 multi-mound regime was a transient evaluated at n_frames=400, not a structural mitigation. Est #114 (decay=0.02 niche) RETRACTED on the same grounds.
- **FOUR INDEPENDENT n_frames=1200 STRESS TESTS UNIVERSALLY COLLAPSE** (sw 8 cell.n, sw 9 kadh, sw 11 D, sw 15 r_on) — Est #82 confirmed parameter-invariant. Est #124 decisively closes the "D shift mitigates Est #82" hypothesis that motivated the last 6 batches' parent.
- **B30 SINGLE-SEED WINS NOT REPLICATED** (sw 2 kadh=9 median≈0.95 σ≈0.022; sw 3 D=0.002 indistinguishable from D=0.0042). Third case of single-seed retraction.
- **Parameter refinements universally flat** — broad plateaus, no sharp optima.
- **PROJECT-BEST LOSS UNCHANGED at 0.9126** (6th tie at this floor since B17, confirming metric-bound floor).
- **NEW Est #115–#126** added to ledger; Est #126 records the strategic-exhaustion conclusion.

**B32 = MPM ENGINE FORK.** `specs/dicty_loop_base.yaml` now contains the MPM base spec (each cell = 8 MLS-MPM particles with Young's modulus 60, true volume exclusion via the grid). `sweep_plan.json` rewritten as 16 MPM-native sweeps: H1 (n_frames at MPM parent — the structural Est #82 break test at engine level), H2-H14 MPM parameter map, H15 16-seed verification, H16 DECISIVE cell.youngs × n_frames=1200 (the structural-stiffness Est #82 mitigator test). **RUN REQUIREMENT: set `DICTY_ENGINE=dicty_engine_mpm` env var** (documented in spec header).

Files updated: `dicty_loop_log.md` (B31 summary), `_b31_log.md` (per-sweep entries), `dicty_knowledge.md` (Est #115–#126 + B32 Open Questions + B32 Planned Hypotheses), `user_input.md` (B31 acknowledged), `specs/dicty_loop_base.yaml` (MPM spec), `sweep_plan.json` (B32 16 sweeps).


================================================================================
## BATCH 32 — 2026-06-18 03:45:05
================================================================================
Verified the patches. Final summary:

## B32 outcome — infrastructure failure, no scientific evidence

**What happened.** All 256 sims aborted at the first `set_param` call with `operator 'inflow_mpm' not in registry`. `sweep_results.json` is all-`NaN`; the sweep figures listed in the prompt are stale from prior batches (only the few timestamped today contain empty morphology strips). I read the instruction, the B31 ledger, and the loop log; the figures contain no morphology — there was nothing to adjudicate. **I did not fabricate per-sweep observations.**

**Root cause.** `dicty_loop.py:60` (`eval_sweeps()`) launched the subprocess without `DICTY_ENGINE`. `eval_sweeps.py:26` defaulted to the point-cell engine, which has no `inflow_mpm`/`mpm` ops. The MPM base spec authored last batch was correct; only the launcher was broken. Same family as B21 Est #72 and B26 Est #95.

**Fixes applied this batch (code, not science).**
- `dicty_loop.py` — `_engine_from_spec()` sniffs the base spec for MPM tokens and sets `DICTY_ENGINE=dicty_engine_mpm` for the subprocess; explicit env override still wins.
- `eval_sweeps.py` — `_preflight(base)` aborts before the 256-sim loop if any scheduled op is missing from `_OPERATOR_REGISTRY`, with a diagnostic naming the missing op and the loaded engine.

**Files updated.**
- `dicty_loop_log.md` — B32 batch entry: infrastructure failure, two patches, no ledger change.
- `dicty_knowledge.md` — added Est #127 (engine/spec mismatch must fail at preflight; methodological lesson).
- `user_input.md` — B32 acknowledged with full context + flag for the user about the schedule slip.
- `specs/dicty_loop_base.yaml` — UNCHANGED (still the MPM spec).
- `sweep_plan.json` — UNCHANGED (the plan was well-designed; re-runs cleanly under the patched launcher as B33).

**B33 = re-run the unchanged B32 plan.** The structural Est #82 break test at MPM engine level — the project's primary structural deliverable — slips one batch. The scientific question is unchanged.


================================================================================
## BATCH 33 — 2026-06-18 03:52:22
================================================================================
## B33 — second consecutive infrastructure failure, no scientific evidence

**What happened.** All 256 sims aborted at `set_param` with `operator 'inflow_mpm' not in registry`. `sweep_results.json` is all-NaN; every sweep figure has an empty morphology strip; `best_montage.png` is stale from B30. There was nothing to read morphologically — I did not fabricate per-sweep observations.

**Root cause (timestamps).** `sweep_results.json` mtime **03:45:04**, `dicty_loop.py` **03:47:54**, `eval_sweeps.py` **03:48:26**. The B32 agent's two patches were written to disk *after* B33's subprocess had already finished NaN-ing every sim. End of `dicty_loop_run.log` shows the newly-live `_preflight()` raising a clean `ValueError` — confirming the patches **are now on disk for B34**, but two consecutive batches were lost to the same launcher/spec mismatch.

**No code changes this batch** (the B32 patches now exist on disk and are working).

**Files updated.**
- `dicty_loop_log.md` — B33 batch entry referencing `_b33_log.md`.
- `_b33_log.md` — full batch entry (no per-sweep entries; nothing to adjudicate).
- `dicty_knowledge.md` — added **Est #128** (save-before-launch enforcement methodological lesson) inside the new "B33 — second consecutive infrastructure failure" section. Est #115–#126 (B31 point-cell closure) and Est #127 (B32 preflight) all stand.
- `user_input.md` — B33 acknowledged with full context and a flag to the user about a hardened launcher (canary + edit-fsync) since the current behavior relies on `Edit`/`Write` being immediately durable, which empirically failed.
- `specs/dicty_loop_base.yaml` — **UNCHANGED** (still the MPM base spec).
- `sweep_plan.json` — **UNCHANGED**.

**B34 = re-run of the unchanged B32/B33 plan.** The structural Est #82 break test at MPM engine level — the project's primary structural deliverable — is now two batches behind, but the scientific question is unchanged.


================================================================================
## BATCH 34 — 2026-06-18 03:58:44
================================================================================
All updates applied. Summary:

**B34 = third consecutive infrastructure failure — no scientific evidence.** 256 sims aborted at the on-disk `_preflight()` with `operator 'inflow_mpm' not in registry`. `sweep_results.json` is all-NaN, sweep figures are empty, `best_montage.png` is stale from B30.

**Root cause:** the long-running `dicty_loop.py` parent process holds a stale in-memory `eval_sweeps()` from before the B32 patches. Python does not re-import a running module from disk, so the autodetect that B32 put in the launcher never executed despite being on disk since 03:47.

**Fix applied:** moved the engine autodetect from `dicty_loop.py:eval_sweeps()` into `eval_sweeps.py:_autodetect_engine()` at module top. Because `eval_sweeps.py` is launched as a fresh subprocess each batch, the on-disk version is always loaded — defeating the entire stale-launcher class. Env override `DICTY_ENGINE` still wins.

**Files updated:**
- `eval_sweeps.py` — `_autodetect_engine()` + `[eval_sweeps] DICTY_ENGINE=<name>` startup print
- `dicty_knowledge.md` — new B34 section + **Est #129** (engine autodetect must live in the spawned subprocess)
- `dicty_loop_log.md` — B34 batch entry appended
- `_b34_log.md` — full batch entry (no per-sweep entries; nothing to adjudicate)
- `user_input.md` — B34 acknowledged with full context

**Unchanged (per the "no science = no parent change" principle):**
- `specs/dicty_loop_base.yaml` — still the MPM base spec
- `sweep_plan.json` — still the B32 plan

**B35 = unchanged re-run.** The MPM Est #82 structural break test is now three batches behind schedule. Maximum-certainty option for the user: kill the running `dicty_loop.py` and restart with `--resume` so both launcher and subprocess load the patched paths cleanly.


================================================================================
## BATCH 35 — 2026-06-18 04:35:31
================================================================================
B35 complete. Headline result: **MPM engine fork produces a clean structural NEGATIVE.**

**Findings (across 16 sweeps × 16 values = 256 sims):**
- **Est #130 DECISIVE NEGATIVE:** the bare `{secrete, sense, inflow_mpm, mpm}` stack is **NON-AGGREGATING** at every parameter. inner_mass FLAT at ~0.19 (REAL=0.61) across n_frames ∈ [120, 2400]; morphology is identical sparse-scatter at every panel of every sweep; loss MONOTONE-UP with n_frames. The finite-volume hypothesis is falsified in the **opposite** direction (under-aggregation, not over-aggregation).
- **Est #136 + #144:** cell.youngs across 4 orders of magnitude produces NO morphological change — the B31 central candidate ("structural stiffness mitigates Est #82") is dead.
- **Est #139 methodological:** engine outer `dt` is structurally inert in MPM (16 pixel-identical results); root-caused in `mpm.py:184` — MPM uses its own dt_sub. dt sweeps dropped from future MPM batches.
- **Suspected artefacts (Est #140 / #144 provisional):** project-best loss 1.141 at substeps=32 and 1.17 at youngs=5 × n_frames=1200 are consistent with **particle depletion** masquerading as low loss — morphology strips show progressively sparser fields, not aggregation.

**B36 design — adjudication batch, no code edits:**
- 5 sweeps adjudicate the suspected wins (substeps×n_frames=1200, youngs×n_frames=1200, substeps fine, 16-seed at substeps=32, 16-seed at the joint corner).
- 5 joint sweeps rule out narrow productive corners (sense.gain × {camp.D, secrete, mpm.drag, mpm.a_max}, cell.n × n_frames=1200).
- 6 refinements: mpm.dt_sub (the live resolution knob, never swept), camp.{D, decay} fine, inflow_mpm.rate fine, particle.per_parent fine, n_frames at new parent.

**B36 parent:** conservative metric-ripple adoptions only — `per_parent=24, a_max=1800, vmax=1.5, drag=10, inflow=0.5, secrete=6, gain=50`. Suspected artefact wins NOT adopted.

**B37 commitment:** inter-cell cohesion mechanism added to MPM (enable existing `surface_tension` via `is_liquid` flag in `dicty_engine_mpm.build()`, OR dedicated MPM-native adhesion operator). Deferred from B36 to keep this batch's infra surface stable after the B32/33/34 losses.


================================================================================
## BATCH 36 — 2026-06-18 07:33:47
================================================================================
