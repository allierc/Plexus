# Cardio-MPM Loop — Running Analysis

Per-batch analysis written by the agent-in-the-loop. Each batch: the parallel `cardio_mpm_train.py`
jobs, their final interior **R²**, the dashboard read (red-on-green superposition, learned stiffness +
direction dx/dy structure), the winner, and the reasoning for the next `cardio_mpm_plan.json`. Durable
claims distilled here live in `knowledge_cardio_mpm.md`.

Seed batch 1 (parent `material_directional_cardio`, lr 2e-3, substeps 5, amplitude 15, dur0 30,
w_amp 0.3) probes the **anti-collapse term**: w_amp {0 (ablation), 0.3, 1.0}, plus one-knob amplitude
{10,25} and lr 5e-3 — to answer Q6 (does w_amp keep GD out of the zero-motion basin?) and to see
whether the UNet begins structuring the stiffness + direction fields.

---

<!-- agent appends dated batch sections below -->

## Batch 1 — mixed (all R²<0, but 1700× better than init; clean separation on amplitude) — 2026-06-23
Parent: slot 0 = directional_cardio, force, lr2e-3, substeps5, amp15, dur0 30, w_amp0.3 (400 it)
Hypothesis: "w_amp defends GD against the zero-motion collapse basin (Q6), and amplitude is the dominant overshoot knob (Q3); meanwhile watch whether the UNet structures stiffness/direction (Q1/Q2)."

Slot 0 [parent_wamp03]   cfg=amp15 w_amp0.3          R2=-1.552  ampL=0.105 dur→52.0  red-on-green=OFF  stiffness=uniform-low (bright anchored frame only)  dir=coherent smooth domains
Slot 1 [wamp0_ablation]  cfg=w_amp0.0 (ablation)     R2=-1.639  ampL=0.075 dur→47.1  red-on-green=OFF  stiffness=uniform-low+frame  dir=coherent domains (slightly sharper)
Slot 2 [wamp10]          cfg=w_amp1.0                R2=-1.516  ampL=0.112 dur→53.2  red-on-green=OFF  stiffness=uniform-low+frame  dir=coherent domains
Slot 3 [wamp03_amp10]    cfg=amp10                   R2=-1.433  ampL=0.150 dur→52.2  red-on-green=OFF (best aligned)  stiffness=uniform-low, faint interior texture  dir=coherent domains  ← WINNER
Slot 4 [wamp03_amp25]    cfg=amp25                   R2=-2.122  ampL=0.010 dur→51.4  red-on-green=OFF (worst)  stiffness=uniform-low+strong frame  dir=coherent domains
Slot 5 [wamp03_lr5e3]    cfg=lr5e-3                  R2=-1.672  ampL=0.077 dur→111.4 (RAN AWAY)  red-on-green=OFF  stiffness=uniform-low+frame  dir=coherent domains

Ranking (R²→1): s3(-1.433) > s2(-1.516) > s0(-1.552) > s1(-1.639) > s5(-1.672) > s4(-2.122).
Winner: slot 3 (amp10) — best R² AND visibly the least-misaligned red, by reducing wrong-direction overshoot.
Verdict:
 - Q3 SUPPORTED: amplitude is the dominant overshoot knob and the effect is monotonic — amp10 > amp15 > amp25 in R². Lower amplitude → less wrong-direction overshoot → better R².
 - Q6 INCONCLUSIVE/weak: no slot collapsed to zero motion (all kept ampL>0, dur→~period), so the collapse basin was NOT the active failure mode at amp15. w_amp helped R² only marginally (−1.64→−1.55→−1.52 over 0/0.3/1.0). Collapse is not the bottleneck here; directional misalignment is.
 - Q1 ANSWERED (yes): a coherent direction field emerges in every slot — smooth low-frequency domains, NOT salt-and-pepper noise. But the domains do not reproduce the real per-node beat directions.
 - Q2 ANSWERED (no): learned stiffness stays ~uniform-low interior with only a bright anchored-boundary frame. The UNet is not using stiffness to fit; all the work is in the direction field.
 - KEY CROSS-CUT: motion MAGNITUDE matching ≠ good fit. amp25 matched real motion energy best (ampL=0.010) yet had the worst R² (−2.122); amp10 under-shot energy (ampL=0.150) but fit best. The bottleneck is DIRECTIONAL alignment, not magnitude. lr5e-3 destabilised the learnable duration (ran to 111 frames ≫ period 52) and hurt R².
Failures: none (all 6 done, R² reported). lr5e-3 is a soft failure (duration runaway).
Next: parent = slot3 config (amp10, force, lr2e-3, substeps5, dur0 30, w_amp0.3). Batch 2 probes the MECHANISM (Q7: force body-force vs active-stress −A·a·nnᵀ, which may give the coordinated shear the directional force can't), continues the amplitude trend down (amp5) with an amp0 ablation as the no-active-contraction floor, re-tests w_amp0 collapse at the better amp10 regime, and tests lr1e-3 for duration stability.

## Batch 2 — mixed→breakthrough (phase falsified; STRESS mechanism jumps to best R² + best superposition) — 2026-06-23
Parent: slot 0 = directional_cardio, FORCE, md0 (no phase), amp10, lr2e-3, sub5, dur0 30, w_amp0.3 (400 it) = batch-1 winner with τ OFF
Hypothesis: "Making activation a TRAVELLING wave a(x,y,t)=pulse(t−τ(x,y)), with a learnable phase-delay τ∈[0,max_delay], lets neighbouring regions fire in sequence — the substrate for the real curved/rotary loops — so a nonzero max_delay should bend red onto green and beat the max_delay=0 control."
(NB the actual launched slots differ from the seed plan: md60 ran as an EXTRA, and the seed's dropped md40+stress slot was KEPT as s4 — so the batch also got an unplanned, confounded mechanism probe. Authoritative R² from job logs `done ->`.)

Slot 0 [parent_nophase]    cfg=md0 (control)            R2=-1.141  red-on-green=OFF (overshoot/misaligned)  stiffness=uniform-low+frame (inert)  dir=coherent domains  | no τ panel
Slot 1 [phase_md20]        cfg=--max_delay 20           R2=-1.217  red-on-green=OFF  stiffness=uniform-low+frame  dir=coherent  τ=smooth, used[0,9.9] of 20 (central low-delay blob)
Slot 2 [phase_md40]        cfg=--max_delay 40           R2=-1.195  red-on-green=OFF  stiffness=uniform-low+frame  dir=coherent  τ=smooth, used[0,18.7] of 40
Slot 3 [phase_md80]        cfg=--max_delay 80           R2=-1.253 (worst)  red-on-green=OFF  stiffness=uniform-low+frame  dir=coherent  τ=mottled, used only[0.3,8.9] of 80
Slot 4 [phase_md40_stress] cfg=mech STRESS + md40       R2=-0.845 (BEST)  red-on-green=BEST interior superposition seen yet  | dur=NaN, τ=NaN, field panels BLANK (numerically broken)  ← WINNER (signal, not clean)
Slot 5 [phase_md40_lr1e3]  cfg=md40 + lr1e-3            R2=-1.138  red-on-green=OFF  stiffness=uniform-low+frame  dir=coherent (sharper)  τ=smooth, used[0,9.9] of 40
(extra) [phase_md60]       cfg=--max_delay 60           R2=-1.164  (fills the sweep; force)

Ranking (R²→1): s4 stress(-0.845) ≫ s5 md40_lr1e3(-1.138) ≈ s0 md0(-1.141) > md60(-1.164) > s2 md40(-1.195) > s1 md20(-1.217) > s3 md80(-1.253).
Winner: slot 4 (stress) — best R² AND the first dashboard where interior red loops visibly sit ON the green loops. BUT confounded (mechanism+phase changed together) and numerically broken (dur=NaN, blank field panels), so it is a strong SIGNAL, not a clean result.
Verdict:
 - PHASE (force mechanism) FALSIFIED: the no-phase md0 control (-1.141) beats EVERY nonzero max_delay (md20 -1.217, md40 -1.195, md60 -1.164, md80 -1.253). Adding the travelling wave mildly HURT. The learnable τ converged to a SMALL delay (max used ~9–19 f, far under both the allowed max and the period ≈50) with coherent low-frequency structure (central low-delay region) — so τ DOES self-organise, but it does NOT bend red onto green. Travelling-wave activation is not the lever for the directional-misalignment bottleneck under the force mechanism.
 - lr: lr1e-3 (s5, -1.138) > lr2e-3 (s2, -1.195) at md40 — re-confirms Est.#9 (lower lr is better). Even the best-tuned phase run only MATCHES the md0 control (-1.138 ≈ -1.141), never beats it.
 - MECHANISM (Q7) — the headline: active stress M1 (s4) leapt to R2=-0.845 and the best interior superposition yet, DESPITE amp10 (which the ledger feared would be near-collapse for stress). This CONTRADICTS that fear: amp10-stress moves plenty and aligns better. M1 (active stress, contraction along axis n via stress divergence) appears to give the coordinated motion the directional body force cannot. Confounded + NaN, so batch 3 must run a CLEAN, NaN-guarded stress sweep with an md0 control.
 - Q2 (stiffness) re-confirmed inert: uniform-low interior + bright anchored frame in every renderable slot.
 - Q1 re-confirmed: coherent direction domains everywhere.
Failures: s4 numerically broke (NaN log_dur/τ → blank field panels) — cause: the stress×phase forward spiked a non-finite gradient that corrupted log_dur and the UNet (no grad guard). FIXED in cardio_mpm_train.py: skip opt.step() when the clipped grad norm is non-finite. s5's progress.txt was stale at it=350 (external refresh) but dashboard_00399 exists and the job log reports the final R2=-1.138.
Next: parent = STRESS, md0, amp10, lr1e-3, w_amp0.3, sub5, dur0 30 (NaN-guarded). Batch 3 = clean STRESS calibration: s1 = force md0 (direct M0-vs-M1, one knob = mechanism), s2 amp5, s3 amp20 (stress-gain calibration), s4 amp0 (ablation/motion floor), s5 stress+md40 (re-test phase under stress, NaN-guarded — clean redo of the s4 breakthrough).

## Batch 3 — clean falsification (the STRESS "breakthrough" does NOT replicate; force wins decisively) — 2026-06-23
Parent: slot 0 = directional_cardio, STRESS, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3 (400 it), NaN-guard active. The clean redo of the batch-2 s4 stress winner.
Hypothesis: "The MECHANISM is the lever (b02): a CLEAN NaN-guarded active-stress md0 run beats force-md0 at matched amp10/lr1e-3 (Q7); amp10 is near the stress-gain sweet spot (Q8), amp0 is the motion floor, and phase τ behaves differently under stress (Q9)."
NB each slot's job log holds TWO `done ->` lines — the driver submitted every slot TWICE (~50 s apart, same config, same archive dir). The 2nd run finishes last and is what dashboard_00399 shows; both values are reported (2nd | 1st) and the spread is itself evidence (stress is run-to-run chaotic; force is reproducible). Authoritative R² = the dashboard-matching 2nd run.

Slot 0 [parent_stress_md0] cfg=STRESS amp10 md0            R2=-117.038 (|-14.437)  dur→35.9  amp10  red-on-green=OFF (RED SPAGHETTI — long wild overshoot streaks, green buried)  stiffness=uniform-low+frame (inert)  dir=coherent blocky domains
Slot 1 [force_md0]         cfg=--mechanism force (1 knob)   R2=-1.045  (|-1.027)    dur→41.3  amp10  red-on-green=OFF but small & modestly aligned (least overshoot)  stiffness=uniform-low+frame, faint interior texture  dir=coherent smooth domains  ← WINNER
Slot 2 [stress_amp5]       cfg=--amplitude 5 (1 knob)       R2=-8.452  (|-8.877)    dur→42.8  amp5   red-on-green=OFF (small loops, still overshoot/misaligned — best stress slot)  stiffness=low+frame+faint blobs  dir=coherent
Slot 3 [stress_amp20]      cfg=--amplitude 20 (1 knob)      R2=-208.409 (|-979.657) dur→30.8  amp20  red-on-green=OFF (wild streaks)  stiffness=green mottled (more structure, high)  dir=coherent
Slot 4 [stress_amp0 ABL]   cfg=--amplitude 0 (ablation)     R2=-1081.124 (|-2445.120) dur→31.1 amp=25.0(!)  red-on-green=OFF (worst, big streaks)  — ABLATION INVALID: see Failures  stiffness=green structured  dir=coherent
Slot 5 [stress_md40]       cfg=--max_delay 40 (1 knob)      R2=-407.627 (|-2727.361) dur→31.8  amp10  τ used[0.25,2.3]/40 (tiny)  red-on-green=OFF (streaks)  stiffness=green structured  dir=coherent

Ranking (R²→1): s1 force(-1.045) ≫ s2 stress_amp5(-8.45) > s0 stress_amp10(-117) > s3 stress_amp20(-208) > s5 stress_md40(-408) > s4 stress_amp0-INVALID(-1081).
Winner: slot 1 (force_md0) — best R² by ~8× over the best stress slot AND the only one whose red isn't a spaghetti of overshoot streaks. This is the best HONEST (non-NaN) R² across all batches (force md0 lr1e-3 -1.045 < the batch-2 force md0 lr2e-3 -1.141; re-confirms lr1e-3 > lr2e-3, Est.#9).
Verdict:
 - Q7 (mechanism) — FLIPPED, the headline FALSIFICATION: the batch-2 "active stress M1 ≫ force M0" breakthrough does NOT replicate when cleaned. The b02.s4 −0.845 was an ARTIFACT of the NaN-corrupted run (blank field panels = the forward had already diverged; the metric was computed on a degenerate state). With the NaN-guard ON and identical amp10/lr1e-3, clean stress is catastrophic (−117) and force md0 (−1.045) decisively wins. M0 force ≫ M1 stress for this fit. Active stress drives wild overshoot streaks (ampL huge: s4 ampL=537), NOT coordinated superposition.
 - Q8 (stress-gain): monotonic like force but offset to catastrophic — amp5(−8.45) > amp10(−117) > amp20(−208). Even amp5 is ~8× worse than force amp10. The stress gain is wildly over-scaled at every amplitude tried; there is no stress sweet spot in [5,20].
 - NEW finding — stress is NUMERICALLY CHAOTIC: the duplicate submissions diverge wildly for stress (s3 −980 vs −208; s5 −2727 vs −408; s0 −14 vs −117) but are nearly identical for force (−1.027 vs −1.045). Force is reproducible; stress overshoot is unstable/seed-sensitive. Another mark against the stress mechanism.
 - Q9 (phase under stress): stress_md40 (−408) ≪ stress_md0 (−117). Phase does NOT rescue stress; τ self-organised to a TINY delay (used [0.25,2.3] of 40) — same "small-τ" signature as under force. Phase remains a non-lever in both mechanisms.
 - Q2 (stiffness) under the WINNER (force): re-confirmed inert (uniform-low + bright anchored frame, only faint interior texture). The stress slots show MORE stiffness structure but it is in service of overshoot, not fit. Direction stays coherent everywhere (Q1).
Failures: s4 amp0 ABLATION INVALID — `cardio_mpm_train.py:202` read `amp = args.amplitude or spec_default`, so `--amplitude 0` (falsy) fell through to the spec default 25.0 (progress.txt/dashboard show amp=25.0, ampL=537). The slot ran amp25-stress (the worst overshoot), NOT a zero floor. The motion floor is STILL UNMEASURED. FIXED in cardio_mpm_train.py: `--amplitude` default → −1 sentinel ("<0=spec; 0=true zero"); amp selection now `spec if args.amplitude<0 else args.amplitude`. Batch 4 re-runs a TRUE amp0 ablation under force.
Next: parent = slot 1 (FORCE, md0, amp10, lr1e-3, w_amp0.3, sub5, dur0 30) — revert to the force track, now the validated best. Batch 4 abandons stress and attacks the OVERSHOOT/directional-misalignment bottleneck with the two unexplored physical overshoot levers: amplitude further down (amp5) and DRAG (overdamping, never swept) {drag_k 30,60}, plus higher substeps (sub8, fidelity) and the now-FIXED true amp0 ablation (motion floor, Q6/Q8).

## Batch 4 — GOOD (DRAG breaks the −1.0 force plateau; motion floor measured; new best honest R²) — 2026-06-23
Parent: slot 0 = directional_cardio, FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag=spec-default (400 it). The batch-3 force winner. (`--drag_k` default 0 is falsy → spec default; drag30/drag60 override. Spec default drag_k is bracketed to (30,60) by the ranking below.)
Hypothesis: "R² is gated by wrong-direction OVERSHOOT, not magnitude/mechanism (Est.#8/#10). Two untried physical levers suppress it: (a) lower amplitude (amp5 continues monotonic Est.#6), (b) higher DRAG (overdamping kills momentum overshoot, Q11). Higher substeps tests numerical-fidelity (Q5). The now-FIXED amp0 ablation finally measures the motion floor (Q6/Q8)."

Slot 0 [parent_force_md0] cfg=amp10 drag=spec       R2=-0.978  ampL=0.336 dur→39.8 amp10  red-on-green=OFF (small, modest align)  stiffness=uniform-low+frame (inert)  dir=coherent smooth domains
Slot 1 [force_amp5]       cfg=--amplitude 5          R2=-0.799  ampL=0.410 dur→44.9 amp5   red-on-green=OFF but small & best-aligned of the standard slots  stiffness=uniform-low+bright frame  dir=coherent domains
Slot 2 [force_drag30]     cfg=--drag_k 30            R2=-1.045  ampL=0.262 dur→42.2 amp10  red-on-green=OFF (same as parent, slightly worse)  stiffness=uniform-low+frame  dir=coherent
Slot 3 [force_drag60]     cfg=--drag_k 60            R2=-0.602  ampL=0.369 dur→43.3 amp10  red-on-green=OFF but SMALLEST/most-contained loops, least overshoot  stiffness=uniform-low+frame + MOST interior texture yet  dir=coherent domains  ← WINNER
Slot 4 [force_amp0 ABL]   cfg=--amplitude 0 (TRUE 0) R2=-0.845  ampL=1.000 dur→30.0(unchanged) amp0  red-on-green=OFF (red=tiny passive stubs, no loops)  stiffness=SALT-AND-PEPPER NOISE (untrained)  dir=SALT-AND-PEPPER NOISE (untrained)  ← MOTION FLOOR
Slot 5 [force_sub8]       cfg=--substeps 8           R2=-0.737  ampL=0.293 dur→39.8 amp10  red-on-green=OFF but better-aligned than parent  stiffness=uniform-low+frame  dir=coherent domains

Ranking (R²→1): s3 drag60(-0.602) > s5 sub8(-0.737) > s1 amp5(-0.799) > s4 amp0-floor(-0.845) > s0 parent(-0.978) > s2 drag30(-1.045).
Winner: slot 3 (drag60) — best R² by a clear margin AND the smallest, most-contained red loops (least wrong-direction overshoot). This is the BEST HONEST R² across ALL batches (−0.602 < the batch-3 force md0 −1.045) and the first time the ~−1.0 force plateau (held b01→b03) is broken.
Verdict:
 - Q11 (DRAG) SUPPORTED — the headline: overdamping is the lever. Monotonic in drag: drag30 (−1.045, less than spec default) < spec-default parent (−0.978) < drag60 (−0.602, more). More drag → less momentum-driven overshoot → red pulled onto green. drag30 being WORSE than the parent places the spec default in (30,60); mild damping below default hurts, strong damping above it helps. Push drag higher next batch.
 - Q6/Q8 (MOTION FLOOR) ANSWERED + REFRAMING: the TRUE amp0 ablation (sentinel fix verified — progress.txt amp=0.0) gives R²=−0.845, and CRITICALLY it BEATS the active parent (−0.978). Passive boundary-anchoring alone explains most of the interior motion; active force at amp10/default-drag injects NET-HARMFUL overshoot (it makes the fit WORSE than doing nothing). Only amp5 (−0.799), sub8 (−0.737), and drag60 (−0.602) clear the passive floor. The real bar to beat is −0.845, not 0 — and most settings fail it. (Partial answer to Q4: a large share of interior R² is just the anchored boundary, not the active model.)
 - amp0 FIELDS = UNTRAINED NOISE: with no active force there is no field gradient, so dx/dy AND stiffness stay at salt-and-pepper init and dur stays at 30.0. This confirms the coherent direction DOMAINS seen everywhere else are a LEARNED product of the active-force gradient, not an init artifact.
 - Est.#6 (amplitude) re-confirmed: amp5 (−0.799) > amp10 (−0.978), monotonic-down still holds. amp5 just clears the floor — i.e. at amp5 the active force finally adds slightly-useful aligned motion instead of net overshoot.
 - Q5 (substeps) ANSWERED — a real, new lever: sub8 (−0.737) > sub5 parent (−0.978). Some of the residual misalignment WAS a numerical-fidelity / integration artifact; finer substeps reduce it. Not just a stability knob.
 - UNIFYING THEME: the three improvements (drag↑, substeps↑, amplitude↓) are ALL overshoot-suppression. This nails Est.#8 — OVERSHOOT, not magnitude or mechanism, is the bottleneck — and identifies DRAG as the strongest single brake found.
 - Q2 (stiffness): still mostly uniform-low + anchored frame, BUT drag60 shows the MOST interior texture seen yet. Hint that once overshoot is damped the UNet begins to use stiffness; not yet load-bearing. Q1 (coherent direction) re-confirmed in every ACTIVE slot.
Failures: none (all 6 done, R² reported, all dashboards at 00399). The amp0 ablation is now VALID (true zero confirmed).
Next: parent = slot 3 (drag60: FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k60) — drag is promoted to a parent knob. Batch 5 pushes drag HIGHER (drag90, drag120: is the monotonic gain sustained, where's the optimum/over-damp collapse?), stacks the other two winning levers onto the drag60 base one-at-a-time (sub8, amp5), and the ABLATION is amp0+drag60 — the crucial control: does drag60 shift the PASSIVE floor (≠ −0.845) or does it win purely by taming ACTIVE overshoot (≈ −0.845 passive)?

## Batch 5 — GOOD (drag monotone sustained → new best R²; brakes don't cleanly stack; passive floor confirmed drag-invariant) — 2026-06-23
Parent: slot 0 = directional_cardio, FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k60 (400 it). The batch-4 winner. (Archive dir prefix is `mpm_b04_*` but this is the 5th batch in the ledger narrative — dir prefix ≠ ledger count, as before.)
Hypothesis: "DRAG is the overshoot lever (Q11): pushing drag HIGHER (90,120) keeps lifting R² until over-damping starves motion — find the optimum (Q12). The other confirmed brakes (sub8, amp5) each beat the OLD parent alone; stacking them onto the drag60 base tests whether brakes STACK below −0.6 or saturate (Q13). The ablation amp0+drag60 is the attribution control: ≈ −0.845 (b04 passive floor) ⇒ drag wins purely by taming ACTIVE overshoot; ≠ ⇒ drag also reshapes passive motion (Q14)."
NB each slot ran TWICE (dup submission); authoritative R² = the 2nd `done ->` (the one dashboard_00399 shows). Parent drag60 reproduced at −0.592 here vs −0.602 in b04 (force is reproducible, Est.#11).

Slot 0 [parent_drag60]  cfg=drag_k60 (control)       R2=-0.592  ampL=0.355 dur→41.0 amp10  red-on-green=OFF but small/contained (least overshoot of the standard base)  stiffness=uniform-low+frame, faint interior teal blobs  dir=coherent smooth domains
Slot 1 [drag90]         cfg=--drag_k 90              R2=-0.519  ampL=0.355 dur→41.2 amp10  red-on-green=OFF but smaller/tighter than parent  stiffness=dark interior + yellow blobs + bright frame (more texture)  dir=coherent domains
Slot 2 [drag120]        cfg=--drag_k 120             R2=-0.502  ampL=0.387 dur→44.2 amp10  red-on-green=OFF, smallest/most-contained loops  stiffness=MOST interior texture yet (scattered green/yellow blobs through dark interior + bright frame)  dir=coherent domains  ← WINNER
Slot 3 [drag60_sub8]    cfg=--substeps 8             R2=-0.544  ampL=0.308 dur→41.2 amp10  red-on-green=OFF, slightly better-aligned than drag60 parent  stiffness=interior teal texture + frame  dir=coherent domains
Slot 4 [drag60_amp5]    cfg=--amplitude 5            R2=-0.620  ampL=0.438 dur→41.9 amp5   red-on-green=OFF, WORSE than drag60 parent (−0.592)  stiffness=dark interior + scattered teal blobs (youngs scale to ~160)  dir=coherent domains
Slot 5 [drag60_amp0 ABL] cfg=--amplitude 0 (TRUE 0)  R2=-0.845  ampL=1.000 dur→30.0(unchanged) amp0  red-on-green=OFF (red=tiny passive stubs, no loops)  stiffness=SALT-AND-PEPPER NOISE (untrained)  dir=SALT-AND-PEPPER NOISE (untrained)  ← PASSIVE FLOOR (= b03/b04 −0.845 exactly)

Ranking (R²→1): s2 drag120(-0.502) > s1 drag90(-0.519) > s3 sub8(-0.544) > s0 parent_drag60(-0.592) > s4 amp5(-0.620) > s5 amp0-floor(-0.845).
Winner: slot 2 (drag120) — best HONEST R² across ALL batches (−0.502 < the b04 drag60 −0.602) and the smallest, most-contained red loops. The drag monotone is sustained but the marginal gains are SHRINKING.
Verdict:
 - Q12 (push drag) SUPPORTED but DIMINISHING: drag60 (−0.592) → drag90 (−0.519, Δ+0.073) → drag120 (−0.502, Δ+0.017). The gain is monotone and NOT yet turned over (ampL even RISES slightly at drag120: 0.387 vs 0.355 — motion is not yet starved), but the curve is flattening hard. The optimum is near; one more push (drag150/180) should bracket the turnover and CLOSE Q12.
 - Q13 (do brakes stack?) PARTLY FALSIFIED — the headline nuance: the brakes are NOT independent. sub8 stacks WEAKLY onto drag60 (−0.544 vs −0.592, small gain). But amp5 onto drag60 (−0.620) is WORSE than the drag60 parent (−0.592) — even though amp5 ALONE beat the OLD parent (b04: −0.799 vs −0.978). Once drag has drained the overshoot reservoir, cutting amplitude further just STARVES useful aligned motion (ampL rose to 0.438 — motion present but less contained/aligned). So the three brakes tap the SAME overshoot reservoir; they don't add. Drag is the dominant, near-sufficient brake.
 - Q14 (attribution) ANSWERED CLEAN: amp0+drag60 = −0.845, IDENTICAL to the b03/b04 amp0 passive floor (−0.845). Drag60 does NOT move the passive boundary-driven motion — drag wins PURELY by taming ACTIVE overshoot. The floor is drag-invariant at this level. (amp0 fields are untrained salt-and-pepper noise again, re-confirming Est.#16: no field gradient without active force.)
 - Q2 (stiffness) WEAKENING further: the higher-drag slots show the MOST interior stiffness structure seen in any batch — drag120 has scattered green/yellow blobs through the dark interior, not just the anchored frame. As overshoot is damped, the UNet increasingly recruits stiffness. Still secondary to direction (which carries the fit) but no longer purely inert. Q1 (coherent direction) re-confirmed in every active slot — still coherent-but-not-yet-correct.
Failures: none (all 6 done at 00399, R² reported). Ablation valid (amp0 true-zero confirmed, amp=0.0).
Next: parent = slot 2 (drag120: FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k120). Batch 6 brackets the drag turnover (drag150, drag180: where does over-damping finally starve motion and reverse the gain? CLOSE Q12), keeps the ONE brake that stacked (drag120+sub8), probes whether the now-overshoot-free fit is OPTIMIZATION-limited on the DIRECTIONAL bottleneck (drag120 + n_iter 600 — can the coherent-but-wrong direction field refine with more iters? Q15/Q1), and the ablation is amp0+drag120 (does even higher drag move the passive floor? Q14 extended). The overshoot reservoir is nearly drained — the NEXT frontier is directional misalignment (Q1), not overshoot.

## Batch 6 — GOOD/PIVOTAL (drag still climbing past 180 — NO turnover; iter600 FLAT ⇒ misfit is NOT optimization-limited) — 2026-06-23
Parent: slot 0 = directional_cardio, FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k120 (400 it). The batch-5 winner. (Archive dir prefix is `mpm_b05_*`; ledger-batch 6.) Each slot ran TWICE (dup submission); authoritative R² = the 2nd `done ->` (the one dashboard_00399 shows). Parent drag120 reproduced at −0.488 here vs −0.502/−0.488 in b05.

Slot 0 [parent_drag120]   cfg=drag_k120 (control)   R2=-0.488  ampL=0.386 dur→43.3 amp10  red-on-green=OFF, small/contained out-and-back loops  stiffness=purple interior + scattered green/yellow blobs + bright frame (interior texture)  dir=coherent domains
Slot 1 [drag150]          cfg=--drag_k 150 (1 knob)  R2=-0.475  ampL=0.382 dur→42.4 amp10  red-on-green=OFF, slightly smaller loops than parent  stiffness=more-washed purple, FEWER interior blobs + frame (texture fading)  dir=coherent domains
Slot 2 [drag180]          cfg=--drag_k 180 (1 knob)  R2=-0.464  ampL=0.386 dur→42.6 amp10  red-on-green=OFF, SMALLEST/most-contained loops  stiffness=SMOOTHEST (mostly purple, least interior texture, faint frame)  dir=coherent (dx one big red domain)  ← WINNER
Slot 3 [drag120_sub8]     cfg=--substeps 8 (1 knob)  R2=-0.490  ampL=0.324 dur→41.9 amp10  red-on-green=OFF, ≈parent (slightly WORSE)  stiffness=interior teal/green texture + frame  dir=coherent domains
Slot 4 [drag120_iter600]  cfg=--n_iter 600 (1 knob)  R2=-0.482  ampL=0.361 dur→52.9 amp10  red-on-green=OFF, ≈parent (Δ+0.006 ≈ noise; dur drifted UP to 52.9)  stiffness=washed purple + faint frame (less texture)  dir=coherent, slightly smoother
Slot 5 [drag120_amp0 ABL] cfg=--amplitude 0 (TRUE 0)  R2=-0.845  ampL=1.000 dur→30.0(unchanged) amp0  red-on-green=OFF (tiny passive stubs, no loops)  stiffness=SALT-AND-PEPPER NOISE (untrained)  dir=SALT-AND-PEPPER NOISE (untrained)  ← PASSIVE FLOOR (= b03/b04/b05 −0.845 exactly)

Ranking (R²→1): s2 drag180(-0.464) > s1 drag150(-0.475) > s4 iter600(-0.482) > s0 parent_drag120(-0.488) > s3 sub8(-0.490) > s5 amp0-floor(-0.845).
Winner: slot 2 (drag180) — best HONEST R² across ALL batches (−0.464). But the gain over drag120 is TINY (drag120 −0.488 → drag150 −0.475 → drag180 −0.464; Δ+0.013 then +0.011) and the red loops are barely smaller.
Verdict:
 - Q12 (drag turnover) — the prediction is FALSIFIED: there is NO turnover out to drag180. The monotone CONTINUES (drag120 −0.488 → drag150 −0.475 → drag180 −0.464), the gains stay positive and roughly CONSTANT (~+0.012/+30, no longer collapsing like b05's +0.073→+0.017), and ampL stays ~0.386 (motion is NOT starved — the predicted over-damp collapse to dots did not happen). Drag is ASYMPTOTICALLY helpful, not optimal-then-reversing. The optimum is a shallow plateau, not a peak; chasing it further buys ~0.01/step. Drag is effectively SATURATED as a lever.
 - Q15/Q1 (is the misfit OPTIMIZATION-limited?) — the PIVOTAL result, answered NO: 50% more iterations (iter600 −0.482) do NOT meaningfully beat the parent (−0.488, Δ+0.006 ≈ run-to-run noise), and the learnable duration even DRIFTED up to 52.9 (from 43.3) without helping R². The coherent-but-wrong direction field does NOT refine onto green with more training. The residual misfit is ARCHITECTURE/LOSS-limited, not training-time-limited. ⇒ training longer / lower lr is NOT the path; a richer DIRECTIONAL MECHANISM is.
 - Q13/Q5 (sub8 stack) CLOSED: sub8 on the drag120 base (−0.490) is now slightly WORSE than the parent (−0.488) — the one brake that helped weakly at drag60 (b05) no longer helps once drag has fully drained the overshoot reservoir. Confirms the brakes tap one (now-empty) reservoir.
 - Q14 (floor) CLOSED-EXTENDED: amp0+drag120 = −0.845, IDENTICAL to the b03/b04/b05 floor — the passive floor is drag-invariant even at drag120. Drag wins purely by taming ACTIVE overshoot at every level.
 - Q2 (stiffness) NUANCE — the texture trend REVERSES at very high drag: interior stiffness texture PEAKS around drag120 and WASHES OUT by drag180 (drag180 is the smoothest active field). So stiffness recruitment is not monotone in drag; it is a transient of the mid-drag regime, not a load-bearing fit component. Direction still carries everything (Q1: coherent-but-wrong).
 - STRUCTURAL DIAGNOSIS (new): the force forward is F=amplitude·a(t)·d with a SCALAR envelope a(t) and a STATIC direction d — so every node can only move OUT-AND-BACK along a line. The real green loops are area-enclosing ELLIPSES. The model is structurally incapable of tracing an ellipse with one static direction + scalar envelope. This is WHY the coherent direction field is "coherent-but-wrong" and why neither drag, substeps, nor more iters close the gap — they all attack overshoot/optimization, not the missing rotational DOF.
Failures: none (all 6 done at 00399/00599, R² reported). Ablation valid (amp0 true-zero confirmed). s3 final progress.txt + dashboard_00399 both settle at R2=-0.490 (the earlier -0.497 was a stale-read estimate; the +0.007 correction does NOT change the ranking — s3 is still just below the parent).
Next: parent = slot 2 (drag180: FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180). The overshoot/optimization levers are EXHAUSTED → batch 7 OPENS the directional frontier with a NEW MECHANISM, `--rotary`: the active-force direction ROTATES through an angle over the beat (d(x,y,t)=R(rotary·(phase−0.5))·d) so each node traces an ELLIPSE/curve instead of a line — directly supplying the missing rotational DOF. Sweep rotary {+90°,+180°,−90°} (handedness probe), keep ONE last drag bracket (drag240: confirm asymptote vs late turnover, CLOSE Q12), and the ablation is amp0+drag180 (rotary=0 is itself the rotary ablation = the parent). If rotary bends red onto green, the line→ellipse diagnosis is confirmed and the directional frontier is unlocked.

## Batch 7 — EXCELLENT/BREAKTHROUGH (the rotary mechanism WORKS — first improvement via SHAPE, not overshoot; line→ellipse diagnosis CONFIRMED) — 2026-06-24
Parent: slot 0 = directional_cardio, FORCE, md0, **rotary 0**, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180 (400 it). The batch-6 winner. (Archive dir prefix is `mpm_b06_*`; ledger-batch 7.) Each of s1–s5 is ONE knob from s0. rotary=0 ⇒ s0 is itself the clean rotary ABLATION.
Hypothesis: "F=a(t)·d (scalar envelope × static direction) can only make degenerate OUT-AND-BACK LINE loops (Est.#23); the real per-node beat is an area-enclosing ELLIPSE. A ROTARY force (d sweeps R radians over the beat → each node ORBITS) should bend red onto green and beat drag180 (−0.464) via SHAPE; handedness (+ vs −) may matter."

Slot 0 [parent_drag180]  cfg=rotary 0 (control/ablation)   R2=-0.472  ampL=0.383 dur→42.7 amp10  red-on-green=OFF, red = short radial OUT-AND-BACK stubs (no loop area)  stiffness=purple interior + scattered green/yellow blobs + frame  dir=coherent domains (reproduces b06 drag180 −0.464; force reproducible, Est.#11)
Slot 1 [rotary_p90]      cfg=--rotary 1.5708 (+90°)        R2=-0.440  ampL=0.338 dur→44.2 amp10  red-on-green=PARTIAL, red excursions now CURVE (opening into arcs/small loops)  stiffness=purple + scattered yellow blobs + bright frame  dir=coherent domains
Slot 2 [rotary_p180]     cfg=--rotary 3.1416 (+180°)       R2=-0.394  ampL=0.275 dur→44.2 amp10  red-on-green=BEST yet — several red traces are now closed LOOPS/ovals sitting inside the green loops (area-enclosing, not lines)  stiffness=MOST interior texture (scattered green/yellow blobs through purple + frame)  dir=coherent domains  ← WINNER
Slot 3 [rotary_n90]      cfg=--rotary -1.5708 (−90°)       R2=-0.468  ampL=0.333 dur→44.8 amp10  red-on-green=OFF, ≈parent (barely better, Δ+0.004 ≈ noise) — wrong handedness gives ~no benefit  stiffness=purple + scattered yellow blobs + frame  dir=coherent domains
Slot 4 [drag240]         cfg=--drag_k 240 (1 knob)         R2=-0.449  ampL=0.394 dur→43.1 amp10  red-on-green=OFF, small/contained out-and-back loops (still lines)  stiffness=purple + scattered teal/yellow + frame  dir=coherent domains
Slot 5 [drag180_amp0 ABL] cfg=--amplitude 0 (TRUE 0)       R2=-0.845  ampL=1.000 dur→30.0(unchanged) amp0  red-on-green=OFF (tiny passive stubs)  stiffness=SALT-AND-PEPPER NOISE (untrained)  dir=SALT-AND-PEPPER NOISE (untrained)  ← PASSIVE FLOOR (= b03/b04/b05/b06 −0.845 exactly)

Ranking (R²→1): s2 rotary_p180(-0.394) > s1 rotary_p90(-0.440) > s4 drag240(-0.449) > s3 rotary_n90(-0.468) > s0 parent_rotary0(-0.472) > s5 amp0-floor(-0.845).
Winner: slot 2 (rotary_p180) — best HONEST R² across ALL batches (−0.394, beats the b06 drag180 −0.464 by +0.07) AND, for the FIRST time, the red traces form area-enclosing LOOPS that sit inside the green loops rather than radial lines. This is improvement by SHAPE (the missing rotational DOF), not by overshoot suppression.
Verdict:
 - Q16 (does rotary work?) SUPPORTED — DECISIVELY. Adding a rotational DOF to the force direction is the single biggest one-knob R² jump since drag was discovered (rotary0 −0.472 → +180 −0.394, Δ+0.078), and it is the FIRST gain not attributable to overshoot. Est.#23 (line→ellipse) is CONFIRMED: with a static d the node loops are zero-area lines; rotating d over the beat opens them into ellipses that overlap green.
 - rotary MAGNITUDE is MONOTONE up to +180°: rotary0 (−0.472) < +90 (−0.440) < +180 (−0.394). No turnover yet — more sweep is better. Push +270/+360 to bracket the optimum.
 - HANDEDNESS MATTERS (chirality is real): +90 (−0.440) clearly beats −90 (−0.468); −90 is statistically indistinguishable from the rotary0 parent (−0.468 vs −0.472). The real beat has a DEFINITE positive (CCW in this convention) sense of rotation; rotating the wrong way buys nothing. Strong evidence the benefit is genuine geometric alignment, not just "any extra motion."
 - Q12 (drag asymptote) CONFIRMED-CLOSED: drag240 (−0.449) continues the shallow monotone above drag180 (~+0.015/+60, ampL=0.394 not starved) — no turnover, asymptotic plateau as predicted. BUT rotary_p180 (−0.394) now BEATS drag240 (−0.449): the directional/shape lever has overtaken the exhausted overshoot lever. Drag is retired as the frontier.
 - Q14 (floor) CLOSED-EXTENDED again: amp0+drag180 = −0.845, IDENTICAL to every prior floor — drag-invariant; amp0 fields stay untrained salt-and-pepper noise (re-confirms Est.#16).
 - Q2 (stiffness) the winner (rotary_p180) shows the most interior stiffness texture of the batch — as the rotary mechanism lets motion align, the UNet recruits a little more stiffness structure, but it remains secondary; direction (now elliptical) carries the fit.
 - NB ampL DROPS with rotary magnitude (rotary0 0.383 → +90 0.338 → +180 0.275): rotary trades some motion-energy match for shape/direction alignment yet R² IMPROVES — re-confirming Est.#8 (R² is gated by directional/shape alignment, not by motion magnitude).
Failures: none (all 6 done at 00399, R² reported). Ablations valid (rotary0 = byte-identical parent; amp0 true-zero confirmed amp=0.0).
Next: parent = slot 2 (rotary_p180: FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180, **rotary 3.1416**). Batch 8 pins the GLOBAL rotary optimum + chirality before making it spatial: push magnitude (rotary +270°, +360° — find the turnover the +90→+180 monotone hasn't reached), test chirality at large magnitude (−180°, −270° — does magnitude help on the WRONG-handed side, or is it purely handedness?), and the ablation is rotary0 (mechanism OFF = b06 drag180 parent −0.472). Once the best global rotary is known, batch 9 makes the rotary angle a LEARNABLE per-pixel field (UNet 5th channel) so chirality/magnitude vary spatially.

## Batch 8 — EXCELLENT (rotary optimum = positive PLATEAU at +270/+360°; chirality decisively confirmed — wrong handedness flat-then-HARMFUL; stiffness now LOAD-BEARING) — 2026-06-24
Parent: slot 0 = directional_cardio, FORCE, md0, **rotary +180° (3.1416)**, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180 (400 it). The batch-7 winner. (Archive dir prefix `mpm_b07_*`; ledger-batch 8.) Each of s1–s5 is ONE knob — the rotary value — from s0. s0/s1 ran ONCE; s2/s3/s4/s5 ran TWICE (dup submission) with near-identical R² (force reproducible, Est.#11) — authoritative = the dashboard_00399-matching run.
Hypothesis: "Rotary is the directional lever (b07): positive monotone to +180° (best −0.394) and handedness is real (−90 ≈ parent). The +90→+180 sweep hasn't turned over → the magnitude optimum is at/beyond π. Pushing +270/+360 either keeps helping (optimum >π) or overshoots into crossed/too-large loops (turnover); on the WRONG-handed side −180/−270 stay ≈ rotary0 if the gain is PURELY chirality, or improve with magnitude if wrong-handed curvature still beats a line. rotary0 = mechanism ablation."

Slot 0 [parent_rp180]  cfg=--rotary +180° (control)   R2=-0.400  ampL=0.272 dur→44.4 amp10  red-on-green=curved arcs/some closed loops on green (reproduces b07 −0.394)  stiffness=purple + scattered green/yellow blobs + frame  dir=coherent domains
Slot 1 [rotary_p270]   cfg=--rotary +270° (4.7124)     R2=-0.357  ampL=0.282 dur→44.1 amp10  red-on-green=BETTER — more curved arcs/loops on green  stiffness=MORE coherent bright-yellow domains (youngs→200)  dir=coherent domains
Slot 2 [rotary_p360]   cfg=--rotary +360° (6.2832)     R2=-0.351  ampL=0.313 dur→44.0 amp10  red-on-green=BEST — curved area-enclosing loops on green  stiffness=MOST coherent of ANY batch (large connected yellow network, youngs→180)  dir=coherent domains  ← WINNER
Slot 3 [rotary_n180]   cfg=--rotary -180° (-3.1416)    R2=-0.450  ampL=0.295 dur→43.9 amp10  red-on-green=OFF, more line-stub-ish than +180 (wrong sense) ≈ rotary0  stiffness=purple + scattered blobs (less coherent than +)  dir=coherent domains
Slot 4 [rotary_n270]   cfg=--rotary -270° (-4.7124)    R2=-0.494  ampL=0.381 dur→44.6 amp10  red-on-green=OFF, biggest/most-wasted motion (WORST, below rotary0)  stiffness=purple + scattered green/yellow blobs  dir=coherent domains
Slot 5 [rotary0 ABL]   cfg=--rotary 0 (ablation)       R2=-0.459  ampL=0.381 dur→43.7 amp10  red-on-green=OFF, out-and-back LINE stubs (zero-area)  stiffness=INERT purple interior + bright frame (the old pattern)  dir=coherent domains (= b06 drag180 −0.464 / b07 rotary0 −0.472)

Ranking (R²→1): s2 +360(-0.351) > s1 +270(-0.357) > s0 +180(-0.400) > s3 -180(-0.450) > s5 rotary0(-0.459) > s4 -270(-0.494).
Winner: slot 2 (rotary_p360) — best HONEST R² across ALL batches (−0.351), nominally edging +270 (−0.357) within noise. The positive-rotary loops are the most area-enclosing yet AND its stiffness field is the most coherent of any batch.
Verdict:
 - Q17 (magnitude optimum) ANSWERED — a POSITIVE-HANDED PLATEAU, no turnover: the positive monotone CONTINUES past π but SATURATES — +180 (−0.400) → +270 (−0.357, Δ+0.043) → +360 (−0.351, Δ+0.006). The +270→+360 step is noise-level, so the optimum is a shallow PLATEAU spanning ≈+270° to +360° (≈ one full turn of the force direction over the beat). No degradation/turnover by +360 — bigger-than-needed positive sweep does NOT crash. The scalar rotary is now bracketed; pushing further buys nothing.
 - CHIRALITY decisively confirmed AND wrong handedness is flat-then-HARMFUL — the cleanest result: at matched |magnitude| positive ≫ negative (+180 −0.400 vs −180 −0.450, Δ0.050; +270 −0.357 vs −270 −0.494, Δ0.137 — the handedness gap WIDENS with magnitude). The wrong-handed side is non-monotone and damaging: −180 (−0.450) ≈ rotary0 (−0.459) but −270 (−0.494) drops BELOW the rotary-off ablation. So over-rotating the WRONG way is WORSE than not rotating at all. This is the strongest evidence yet that the gain is genuine CORRECT-SENSE elliptical tracing (not "any rotation/extra motion"): correct sense helps monotonically to a plateau; wrong sense is flat then actively harmful.
 - ampL reinforces (Est.#8): positive-rotary slots are efficient (ampL +180 0.272, +270 0.282, +360 0.313), while rotary0 AND −270 BOTH have the HIGHEST ampL (0.381) yet are among the WORST R² — wasted, anti-aligned motion. ampL RISES slightly with positive magnitude (bigger correct arcs) yet R² still IMPROVES → R² is gated by shape/direction, not motion energy.
 - Q2 (stiffness) — NOW LOAD-BEARING (Falsified#2 REVISED): the positive-rotary winners (+270, +360) show the MOST coherent stiffness of ANY batch — large CONNECTED bright-yellow domains (youngs→180–200) forming a network pattern, qualitatively unlike the inert purple+frame. The rotary0 ablation still shows the OLD inert purple-interior+bright-frame. So once the correct rotational DOF exists at sufficient magnitude, the UNet recruits genuine coherent stiffness structure — stiffness has flipped from non-load-bearing transient to a real fit component. (Direction still coherent everywhere, Q1.)
 - rotary0 ablation reproduces the line-stub morphology and −0.459 (≈ b06 drag180 −0.464 / b07 −0.472; force reproducible). Clean mechanism-off anchor; isolates rotary's whole contribution (+0.108 from rotary0 −0.459 to +360 −0.351).
Failures: none (all 6 done at 00399, R² reported, dup runs consistent). No amp0 floor this batch (−0.845 is pinned 5× over; rotary0 is the more informative ablation now).
Next: parent = slot 2 (rotary +360: FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180, **rotary 6.2832**). The GLOBAL scalar rotary is now bracketed (positive plateau, +270–360°) → batch 9 GOES SPATIAL with a NEW mechanism added to `cardio_mpm_train.py`: `--rotary_field` adds a UNet channel → a LEARNABLE per-pixel rotary DEVIATION map R(x,y)∈[−rotary_spread,+rotary_spread] rad, so the effective per-pixel sweep is (rotary + R(x,y)) — chirality/magnitude vary spatially (real myocardial fibers rotate spatially). The scalar path stays byte-identical (parent/ablation safe; field slots exercise the new code; runtime smoke-test was blocked by the sandbox so the loop driver is the first real exercise). Slots: s0 = scalar +360 control (the bar, ≈−0.35); s1 field around +360 base (spread ±π) — does spatial variation beat the global scalar?; s2/s3 sweep spread (±90°, ±360°); s4 PURE field from a 0 base (does the UNet rediscover the +positive global sense from scratch?); s5 rotary0 ablation (mechanism off, −0.459).

## Batch 9 — MIXED (spatial rotary field = MARGINAL: tight-spread ±90 edges scalar by Δ+0.012 (new best −0.341) but wider spread HURTS; field saturates to a uniform + nudge, cannot bootstrap chirality from 0, does NOT amplify stiffness → rotary lever EXHAUSTED) — 2026-06-24
Parent: slot 0 = directional_cardio, FORCE, md0, **rotary +360° (6.2832)**, **rotary_field 0** (scalar control), amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180 (400 it). The batch-8 winner. (Archive dir prefix `mpm_b08_*`; ledger-batch 9.) NEW mechanism this batch: `--rotary_field>0` adds a UNet channel → a learnable per-pixel rotary DEVIATION R(x,y)=`rotary_spread`·tanh(o)∈[−spread,+spread] rad; effective per-pixel sweep = (`--rotary`+R(x,y)). s0/s5 are byte-identical to the b08 scalar path (no field channel); s1–s4 exercise the new code. ALL 6 dup-submitted (consistent, force reproducible) — authoritative = the dashboard_00399-matching 2nd run.
Hypothesis: "The global scalar rotary is bracketed (b08, +270–360° plateau, −0.351) and stiffness is now load-bearing under it. Real myocardium has spatially-varying fiber rotation, so a LEARNABLE per-pixel rotary deviation around the +360 base should BEAT the best global scalar: rfield_b360 (s1) > scalar (s0); a moderate spread helps while a very wide ±360 spread that can flip local chirality may add noise; a PURE field from a 0 base (s4) tests whether the UNet rediscovers the +positive global sense from scratch; does the spatial field push stiffness even more coherent (Q2)?"

Slot 0 [parent_rp360]      cfg=rotary_field 0 (scalar +360 control)        R2=-0.353  ampL=0.312 dur→44.1 amp10  red-on-green=small curved arcs partly on green (= b08 +360 winner, reproduced)  stiffness=MOST coherent of the batch — LARGE connected yellow network (youngs→180+)  dir=coherent domains
Slot 1 [rfield_b360_s180]  cfg=--rotary_field 1 --rotary_spread π (±180°)   R2=-0.388  ampL=0.280 dur→45.8 amp10  red-on-green=small arcs, WORSE than control  stiffness=mostly purple + few small green/yellow blobs (LESS coherent than s0)  dir=coherent domains; FIELD saturates to +180° (red) w/ a few blue islands
Slot 2 [rfield_b360_s90]   cfg=--rotary_field 1 --rotary_spread π/2 (±90°)  R2=-0.341  ampL=0.278 dur→44.8 amp10  red-on-green=small curved arcs on green (≈ control)  stiffness=scattered small green/yellow blobs (less coherent than s0)  dir=coherent domains; FIELD saturates to +90° (red) ALMOST EVERYWHERE w/ a few small blue islands → near-uniform + boost (eff ~+450°)  ← WINNER
Slot 3 [rfield_b360_s360]  cfg=--rotary_field 1 --rotary_spread 2π (±360°)  R2=-0.462  ampL=0.355 dur→41.5 amp10  red-on-green=OFF, ≈ ablation level  stiffness=INERT yellow-frame + purple interior  dir=noisier; FIELD red(+) w/ noise + small blue spots
Slot 4 [rfield_b0_s360]    cfg=--rotary 0 --rotary_field 1 --rotary_spread 2π (PURE field, 0 base)  R2=-0.457  ampL=0.343 dur→45.6 amp10  red-on-green=OFF, ≈ ablation  stiffness=INERT yellow-frame + purple  dir=coherent-ish; FIELD is BALANCED/ZERO-MEAN (red+blue mix, NO global + bias) → did NOT find the + sense
Slot 5 [rotary0 ABL]       cfg=--rotary 0 --rotary_field 0 (mechanism OFF)  R2=-0.464  ampL=0.382 dur→40.7 amp10  red-on-green=OFF, out-and-back LINE stubs (zero-area)  stiffness=INERT purple interior + bright frame  dir=coherent domains (= b06 drag180 −0.464 / b07 −0.472 / b08 −0.459)

Ranking (R²→1): s2 ±90(-0.341) > s0 scalar(-0.353) > s1 ±180(-0.388) > s4 pure-0base(-0.457) > s3 ±360(-0.462) > s5 rotary0(-0.464).
Winner: slot 2 (rfield_b360_s90) — best HONEST R² across ALL batches (−0.341), but only Δ+0.012 over the scalar control (−0.353) ≈ noise/marginal. The field bought a small, NOT a structural, gain.
Verdict:
 - Q18 (does spatial rotary beat the global scalar?) ANSWERED — MARGINALLY, and ONLY at TIGHT spread; the field is a near-DEAD lever. R² is MONOTONE in spread: ±90 (−0.341) > scalar/0 (−0.353) > ±180 (−0.388) > ±360 (−0.462 ≈ ablation). Only the SMALLEST deviation (±90) edges the scalar; every wider spread HURTS, and ±360 (which can flip local chirality) collapses ALL the way to the rotary0 ablation level — the locally-flipped (wrong-handed) islands cancel enclosed area, re-confirming the chirality result (b08). Spatial rotary helps only as a SMALL perturbation, not as a rich spatial pattern.
 - The field does NOT learn "spatial fiber rotation" — it SATURATES to +spread (a near-uniform POSITIVE magnitude nudge). In s1/s2/s3 (all +360 base) the learned R(x,y) is red (=+spread) almost everywhere with only a few small blue islands → the UNet uses the channel to push effective magnitude past +360 (e.g. ~+450° at ±90), NOT to carve spatially-varying chirality. This is consistent with the +positive plateau being shallowly still-rising just past +360 (b08 Q17): ±90 lands near the plateau top (tiny gain); ±180/±360 overshoot into the saturated/noisy regime + flipped-island cancellation (worse). So the marginal win is just a hair more positive magnitude, not spatial structure.
 - The +360 scalar base is a NEEDED PRIOR — the UNet does NOT bootstrap global chirality from zero (clean falsification). s4 (0 base, ±360 spread): the learned field stays BALANCED/zero-mean (red+blue mix, no global + bias), UNLIKE the saturated-red fields of the +360-base slots, and R²=−0.457 ≈ rotary0 ablation (−0.464). Without a scalar seed the field cannot discover the +positive sense on its own; chirality must be supplied globally and the field only refines around it.
 - Q2 (stiffness) — the spatial field did NOT amplify the load-bearing stiffness; if anything it FRAGMENTS it (the b09 stiffness hypothesis is FALSIFIED). The SCALAR control s0 shows the MOST coherent stiffness of the batch (large connected yellow network, youngs→180+) — Est.#25's load-bearing structure is a property of the +360 MAGNITUDE regime, faithfully reproduced by s0. The field winner s2 shows smaller scattered blobs (less coherent), and the wide/0-base slots (s3/s4) revert to the inert yellow-frame+purple pattern. Adding the spatial DOF did not recruit MORE stiffness structure.
 - ampL reinforces (Est.#8): the two BEST R² slots (s2 0.278, s0 0.312, low ampL = efficient) vs the two WORST (s5 0.382, s3 0.355, high ampL = wasted) — efficient curved motion still tracks best; the field win is not from extra motion.
 - No NaN/blank field panels — the new `--rotary_field` code ran CLEAN (all 4 field slots rendered finite RdBu maps). The runtime-unverified mechanism is now runtime-VALIDATED (no b02-style artifact); its scientific verdict is just that it is a marginal lever.
Failures: none (all 6 done at 00399, dup runs consistent). Ablations valid (s0/s5 byte-identical scalar path; s4 pure-field is the informative chirality-bootstrap test, not a failure).
Next: parent = slot 2 (rfield_b360_s90: FORCE, md0, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180, **rotary 6.2832, rotary_field 1, rotary_spread 1.5708**). The ROTARY lever (scalar AND spatial) is now EXHAUSTED — fit is stuck at ≈−0.34, still negative (above the −0.845 floor but the curved loops don't yet superpose). Batch 10 PIVOTS to NEW levers, one knob each from parent: (s1) PHASE-ON-ROTARY `--max_delay 40` — re-test the b02-falsified travelling-wave phase NOW that the ellipse exists (rotation + propagation = a spiral wave, the real cardiac mechanic; phase τ and rotary are independent UNet/forward paths so they coexist); (s2) amp15 + (s3) amp7 — RE-OPEN amplitude (overshoot was the pre-rotary cap; now curvature is efficient and the red arcs look UNDER-sized vs green — is the optimum amplitude higher?); (s4) `--rotary_spread 0.7854` (±45°, tighter than the ±90 winner — does the field → scalar as spread→0, confirming it's just a magnitude nudge?); (s5) rotary0 ablation (mechanism off, −0.464).

## Batch 10 — EXCELLENT (AMPLITUDE RE-OPEN is the breakthrough — the amp optimum FLIPS UP in the rotary regime; new best R²; phase marginally helps but τ stays tiny = no spiral) — 2026-06-24
Parent: slot 0 = directional_cardio, FORCE, md0, **rotary +360° (6.2832), rotary_field 1, rotary_spread π/2 (±90°)**, amp10, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180 (400 it). The batch-9 winner (rfield_b360_s90). (Archive dir prefix `mpm_b09_*`; ledger-batch 10.) Each of s1–s5 is ONE knob from s0. All 6 ran once (single submission this batch). Authoritative R² = progress.txt it=399 (= dashboard_00399).

Slot 0 [parent_rfield_s90] cfg=control (rfield ±90)            R2=-0.354  ampL=0.278 dur→45.5 amp10  red-on-green=small curved arcs partly on green (reproduces b09 winner −0.341; force reproducible within ~0.01)  stiffness=coherent yellow blobs/network (youngs→200)  dir=coherent domains; FIELD saturates +90° (red almost everywhere + blue islands)
Slot 1 [spiral_md40]      cfg=--max_delay 40 (1 knob)          R2=-0.325  ampL=0.313 dur→18.6(!) amp10  red-on-green=arcs on green, ≈ parent shape  stiffness=coherent green/yellow blobs  dir=coherent domains; **τ panel used [0.31,9.9] of 40 — TINY, same small-τ signature as b02** (NOT a travelling/spiral wave); dur COLLAPSED to 18.6
Slot 2 [amp15]            cfg=--amplitude 15 (1 knob)          R2=-0.261  ampL=0.182 dur→45.6 amp15  red-on-green=BEST yet — bigger curved arcs sitting on green (loops no longer under-sized)  stiffness=MOST coherent (connected yellow network, youngs→200)  dir=coherent domains; FIELD saturates +90°  ← WINNER
Slot 3 [amp7]             cfg=--amplitude 7 (1 knob)           R2=-0.437  ampL=0.396 dur→45.3 amp7   red-on-green=OFF, under-driven small line-ish stubs (worse than parent)  stiffness=yellow blobs (less than winner)  dir=coherent domains; FIELD saturates +90°
Slot 4 [spread45]         cfg=--rotary_spread 0.7854 (±45°)    R2=-0.345  ampL=0.306 dur→45.1 amp10  red-on-green=small curved arcs on green (≈ parent)  stiffness=coherent yellow blobs  dir=coherent domains; FIELD saturates +45° (red + blue islands) → near-scalar nudge
Slot 5 [rotary0 ABL]      cfg=--rotary 0 --rotary_field 0      R2=-0.461  ampL=0.377 dur→44.2 amp10  red-on-green=OFF, out-and-back LINE stubs (zero-area)  stiffness=INERT purple interior + bright frame  dir=coherent domains (= b06/b07/b08/b09 rotary0 ≈ −0.46)

Ranking (R²→1): s2 amp15(-0.261) > s1 spiral_md40(-0.325) > s4 spread45(-0.345) > s0 parent(-0.354) > s3 amp7(-0.437) > s5 rotary0(-0.461).
Winner: slot 2 (amp15) — best HONEST R² across ALL batches (−0.261, beats the b09 ±90 winner −0.341 by Δ+0.08) AND the bigger curved arcs now sit ON green (the b09 loops looked under-sized) with the LOWEST ampL (0.182 = best motion-energy match). Improvement by SIZE, in the curvature-efficient regime.
Verdict:
 - Q20 (RE-OPEN amplitude) ANSWERED-YES — the headline, and an OVERTURN of Est.#6: amplitude is MONOTONE UP in the rotary regime — amp7 (−0.437) < amp10 (−0.354) < amp15 (−0.261). The pre-rotary "amplitude monotone DOWN" (b01, Est.#6) was an OVERSHOOT-regime property: with a static direction every extra unit of amplitude was wrong-direction overshoot. Once rotary makes the excursion CURVE (efficient, area-enclosing), MORE amplitude grows the under-sized loops toward green instead of overshooting — so the optimum FLIPS. ampL CONFIRMS: amp15 has the LOWEST ampL (0.182, best energy match) AND best R², amp7 the highest (0.396) and worst → at amp10/amp7 the curved loops were UNDER-driven. Biggest one-knob jump since rotary itself (Δ+0.093 from amp7). Push amp higher (b11) to find the new turnover.
 - Q19 (PHASE-ON-ROTARY) NUANCED — sign-flips vs b02 but does NOT engage as a spiral: spiral_md40 (−0.325) beats the parent (−0.354) by Δ+0.029 — phase no longer HURTS (it hurt every nonzero md in b02, Falsified#3). BUT the τ panel shows τ used only [0.31,9.9] of 40 — the SAME tiny-τ signature as b02 (used [0,18.7]/40), so it is NOT forming a travelling/spiral wave. The dur COLLAPSED to 18.6 (from 45.5): the marginal gain is a SHORTER-PULSE effect (a ~10-frame phase offset + an 18-frame pulse tiling the beat differently), not coordinated propagation. So the "spiral wave" hypothesis is NOT supported (τ stays tiny), but the regime change is real (phase flipped from harmful to mildly helpful). Worth re-testing on the amp15 parent (b11 s3) — does it help MORE at the better amplitude, or is it just absorbing the same shorter-pulse slack?
 - Field = near-scalar magnitude nudge CONFIRMED (Est.#26): spread45 (−0.345) lands BETWEEN scalar/0 (−0.353) and ±90 (−0.341), monotone in spread, and the field saturates to +45° (red almost everywhere). As spread→0 the field → scalar exactly as predicted — it carries no spatial structure, just a uniform + magnitude nudge whose size scales with the spread cap. Clean confirmation.
 - Q2 (stiffness) LOAD-BEARING re-confirmed under the winner: amp15 shows the MOST coherent stiffness of the batch (connected yellow network, youngs→200), consistent with Est.#25 (load-bearing in the high-effective-magnitude rotary regime). The rotary0 ablation stays INERT purple+frame (mechanism-off anchor).
 - rotary0 ablation (−0.461) reproduces the line-stub morphology and ≈ the pinned floor (b06 −0.464 / b07 −0.472 / b08 −0.459 / b09 −0.464); isolates rotary's whole contribution at amp10 (+0.107 to the amp15 winner).
Failures: none (all 6 done at it=399, R² reported). Ablation valid (rotary0 = mechanism off). No NaN/blank panels — spiral τ panel rendered finite (no b02-style artifact).
Next: parent = slot 2 (amp15: FORCE, md0, rotary 6.2832, rotary_field 1, rotary_spread 1.5708, **amp15**, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180). AMPLITUDE is the re-opened frontier → batch 11 brackets the new amp turnover (amp20, amp25 — does the monotone-up continue or peak?), re-tests phase on the amp15 parent (spiral_md40 — does phase help MORE at the better amplitude, or still just absorb pulse slack? Q19), re-tests whether higher amplitude RE-OPENS the drag overshoot lever (drag240 — more amp may re-introduce overshoot that drag can tame), and the ablation is rotary0 at amp15 (isolates rotary's contribution at the new amplitude).

## Batch 11 — GOOD (amp monotone-up CONTINUES to amp25 = new best; phase-on-amp15 helps as much as amp but via shorter-pulse not spiral; drag stays saturated; rotary still essential) — 2026-06-24
Parent: slot 0 = directional_cardio, FORCE, md0, rotary +360° (6.2832), rotary_field 1, rotary_spread π/2 (±90°), **amp15**, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180 (400 it). The batch-10 winner. (Archive dir prefix `mpm_b10_*`; ledger-batch 11.) Each of s1–s5 is ONE knob from s0. DUP submission this batch — s0/s1/s2 ran twice in the same dir (two `done ->` in the .out), s3/s4/s5 had a renamed duplicate dir (`spiral_amp15`/`drag240`/`rotary0_amp15`); authoritative R² = the in-dir final run (= dashboard_00399), with the pair as a free variance estimate (Δ≈0.02–0.06, force reproducible).

Slot 0 [parent_amp15]   cfg=control (amp15)              R2=-0.242 [pair -0.291; avg -0.267]  ampL=0.191 dur→45.1 amp15  red-on-green=small curved arcs partly on green (reproduces b10 −0.261)  stiffness=coherent yellow network (youngs→200)  dir=coherent domains; FIELD saturates +90°
Slot 1 [amp20]          cfg=--amplitude 20 (1 knob)      R2=-0.279 [pair -0.255; avg -0.267]  ampL=0.147 dur→43.3 amp20  red-on-green=curved arcs ≈ parent  stiffness=teal-washed connected network (purple holes)  dir=coherent domains; FIELD saturates +
Slot 2 [amp25]          cfg=--amplitude 25 (1 knob)      R2=-0.189 [pair -0.249; avg -0.219]  ampL=0.115 dur→45.0 amp25  red-on-green=slightly BIGGER arcs, more on green (best energy match)  stiffness=MOST coherent (connected yellow network, youngs→200)  dir=coherent domains; FIELD saturates +  ← WINNER (best single + clean amp axis)
Slot 3 [spiral_md40]    cfg=--max_delay 40 (1 knob)      R2=-0.211 [pair -0.205; avg -0.208]  ampL=0.215 dur→17.3(!) amp15  red-on-green=arcs on green ≈ parent  stiffness=purple + green/yellow blobs  dir=coherent; **τ panel used [0.32,9.9] of 40 — TINY (b02/b10 signature, NOT a spiral)**; dur COLLAPSED to 17.3
Slot 4 [drag240]        cfg=--drag_k 240 (1 knob)        R2=-0.261 [pair -0.273; avg -0.267]  ampL=0.220 dur→44.9 amp15  red-on-green=arcs ≈ parent  stiffness=purple + green/yellow blobs (coherent)  dir=coherent; FIELD saturates +  (= parent → drag STILL saturated)
Slot 5 [rotary0_abl]    cfg=--rotary 0 --rotary_field 0  R2=-0.418 [pair -0.416; avg -0.417]  ampL=0.327 dur→42.1 amp15  red-on-green=OFF, out-and-back LINE stubs + tiny loops (zero-area)  stiffness=INERT purple interior + bright frame  dir=coherent domains

Ranking (R²→1, by avg): s3 spiral(-0.208) ≈ s2 amp25(-0.219) > [s0 amp15(-0.267) ≈ s1 amp20(-0.267) ≈ s4 drag240(-0.267)] > s5 rotary0(-0.417). By best-single: s2 amp25(-0.189) > s3 spiral(-0.205) > s0 amp15(-0.242) > …
Winner: slot 2 (amp25) — best single R² of ALL batches (−0.189) and the LOWEST ampL (0.115, best motion-energy match), the clean continuation of the established amplitude lever. spiral_md40 ties it by average (−0.208) but its gain is a confounded shorter-pulse effect (dur→17.3), so amp25 is the cleaner parent.
Verdict:
 - Q20 (amp bracket) — monotone-up CONTINUES to amp25, NO turnover yet: amp25 (−0.189/avg −0.219) is the new best; ampL is cleanly MONOTONE-DOWN with amplitude (amp15 0.191 > amp20 0.147 > amp25 0.115) → the curved loops are STILL under-sized even at amp25, so more amplitude keeps improving the energy match. The R² is FLAT amp15≈amp20 (both avg −0.267) but amp25 breaks lower → the size lever is real with diminishing R²/Δamp; the turnover (where extra size becomes overshoot) is NOT reached. Push amp30/amp35 (b12).
 - Q19 (phase on amp15) — phase HELPS more clearly now (spiral −0.208 vs parent −0.267, Δ+0.06, ≈ the amp25 gain) BUT it is NOT a spiral: τ STAYS TINY (used [0.32,9.9]/40, the SAME b02/b10 signature) and dur COLLAPSED to 17.3 with ampL UP (0.215). The mechanism is again SHORTER-PULSE, not coordinated propagation (Falsified#3 re-confirmed). So two INDEPENDENT levers — amplitude (size) and phase/short-pulse (timing) — each reach ≈−0.21 from the −0.267 parent. OPEN (Q21): do they STACK on amp25?
 - Q12 (drag re-test at amp15) — CLEAN falsification of "higher amplitude re-opens drag": drag240 (avg −0.267) = parent exactly, ampL even rises to 0.220 (slight motion starve). Drag is STILL saturated at amp15; the higher-amplitude overshoot it might tame does not exist (curvature makes the extra motion useful, not wasteful). Q12 stays CLOSED.
 - rotary0 ABL at amp15 (−0.417, ampL 0.327, the highest = most wasted out-and-back motion) — rotary contributes ≈+0.15–0.20 R² at amp15, still ESSENTIAL. NOTE the prediction "amp15 rotary0 worse than the amp10 floor (−0.461)" is FALSIFIED: −0.417 is slightly BETTER than the amp10 rotary0 floor (−0.461) — more amplitude marginally helps even the line-stub regime (drag180 tames the overshoot), it does NOT make it worse. The line-stub floor itself rises a hair with amplitude.
 - Q2 (stiffness) load-bearing re-confirmed: the amp25 winner shows the MOST coherent stiffness (connected yellow network, youngs→200); the rotary0 ablation stays INERT purple+frame.
Failures: none (all 6 done at it=399, R² reported, both submissions). No NaN/blank panels — spiral τ panel finite. The duplicate-dir slot names (`spiral_amp15`/`drag240`/`rotary0_amp15`) are the first-submission renames of s3/s4/s5; same configs, consistent R².
Next: parent = slot 2 (amp25: FORCE, md0, rotary 6.2832, rotary_field 1, rotary_spread 1.5708, **amp25**, lr1e-3, sub5, dur0 30, w_amp0.3, drag_k180). Batch 12 pushes the amp bracket higher (amp30, amp35 — find the turnover), tests whether the phase/short-pulse lever STACKS on amp25 (spiral_amp25, Q21), decouples phase from pulse-duration (dur0_18 — does forcing a short pulse WITHOUT phase reproduce the spiral gain, or does duration just self-tune back to ~period?), and ablates rotary0 at amp25.

---

## PIVOT — Batch 11 (MORPHOLOGY ATLAS BATCH 1 / Phase 1, forward active-stress atlas) — 2026-06-24

**OBJECTIVE PIVOT (2026-06-24).** The prior 11 batches fit the inverse model (UNet learning force/direction fields) and made it better at matching real trajectories (R²→−0.189). But the 2×2 test (Est.#28, Falsified#7) showed loops are GENERIC in active-stress MPM — both isotropic and structured patterns loop with non-affine openness ~0.2–0.3, so **structure is not REQUIRED for loops; loops are inertial/available dynamics**. The new objective: **learn which anisotropic ACTIVE-STRESS PATTERN PARAMETERS generate the REAL loop MORPHOLOGY (size, axis angle, openness, chirality, spatial coherence).**

**Phase 1 = forward SHAPE ATLAS** (forward-sweep `cardio_mpm_atlas.py`, NO inverse training, NO rotary/phase). Launched 6 slots, each ONE pattern knob from `material_aniso_cardio` base: stiff_wl 8, gain_wl 26, fibre_wl 16, fibre_angle 0.6, amplitude 10, drag_k 30. Archive prefix: `mpm_b11_s*`.

Parent (base): aniso_atlas base config
Hypothesis: "Pattern params separate along morphology axes: fibre ANGLE → major-axis ORIENTATION; fibre/stiffness WAVELENGTH → spatial period/ellipticity; amplitude → SIZE; drag → OPENNESS. The atlas reveals the param→morphology families; Phase 2 inverse-tunes the best family to real beat."

**Results (morphology metrics from progress.txt, reported as: openness / aspect / major-axis-angle / size / chirality-agreement):**

Slot 0 [base]          config=base (stiff_wl 8, gain_wl 26, fibre_wl 16, fibre_angle 0.6, amp 10, drag 30)
  - open=0.258, aspect=0.23, angle=1.54 rad, size=5.32e-03, chir+=0.47
  - Trajectory: symmetric radial beat, moderate openness

Slot 1 [fibre_angle0]  config=fibre_angle 0.0 (remove fibre rotation)
  - open=0.303↑, aspect=0.24, angle=1.70, size=4.94e-03, chir+=0.42↓
  - **Finding: removing fibre angle → INCREASES openness, DECREASES chirality** (rotation couples the two)

Slot 2 [fibre_wl32]    config=fibre_wl 32 (coarsen fibre wavelength 16→32)
  - open=0.276, aspect=0.34↑↑, angle=2.29↑↑, size=5.26e-03, chir+=0.51↑
  - **WINNER: coarser fibre → highest aspect (ellipticity), largest angle shift, max chirality; most elliptical/rotated loops**

Slot 3 [stiff_wl24]    config=stiff_wl 24 (coarsen stiffness 8→24)
  - open=0.253, aspect=0.25, angle=1.61, size=5.38e-03↑, chir+=0.45
  - **Finding: stiffness wavelength has MINIMAL morphological effect** on the base pattern

Slot 4 [amplitude25]   config=amplitude 25 (3× amplification 10→25, naive forward)
  - open=0.170, aspect=0.02↓↓, angle=1.58, size=1.09e-03↓↓↓, chir+=0.07↓↓↓
  - **FAILED: complete morphological collapse** — no inverse training to guide the high amplitude; loops vanish (open drops to raw 0.013)

Slot 5 [drag300]       config=drag_k 300 (10× overdamping 30→300)
  - open=0.306↑↑, aspect=0.31, angle=2.77↑↑↑, size=1.95e-03↓, chir+=0.50
  - **Finding: extreme drag → HIGHEST openness/angle-shift but COLLAPSED size** (quasi-static regime, long thin loops)

**Ranking on combined MORPHOLOGY (openness + aspect + angle + chirality, penalizing collapse):**
1. **s2 (fibre_wl32) — WINNER** — best aspect (0.34), highest angle shift (2.29), good openness, coherent size
2. **s1 (fibre_angle0)** — highest openness (0.303), retains size, shows angle-openness coupling
3. **s5 (drag300)** — tied-highest openness (0.306), highest angle (2.77), but severely collapsed size
4. **s0 (base)** — balanced baseline
5. **s3 (stiff_wl24)** — minimal morphology variation from base
6. **s4 (amplitude25)** — FAILED, complete collapse (overshoot artifact in naive forward)

**Verdict:**
- Q22 (PARTIALLY SUPPORTED): fibre WAVELENGTH (coarser 16→32) INCREASES ellipticity (aspect 0.23→0.34) and major-axis rotation (angle 1.54→2.29 rad), suggesting it controls loop **SHAPE**, not just size. Fibre ANGLE affects the openness-vs-chirality trade-off: removing angle (0.6→0.0) opens loops but kills handedness coupling.
- **STIFFNESS wavelength (8→24) has NO visible morphology effect** on the pattern — the gain/fibre patterns dominate.
- **AMPLITUDE in the forward atlas (no inverse):** high amplitude (25×) causes collapse without inverse guidance. The amplitude=10 base is stable; amp25 is over-driven in a naive forward. This differs from the inverse track where amp25 was the best (Est.#27) — in inverse context the UNet learns to harness the extra motion; in forward-only it drives overshoot.
- **DRAG (30→300, 10×):** extreme overdamping MAXIMIZES openness (inertial→quasi-static) but SHRINKS absolute loop size (only 1.95e-03 vs base 5.32e-03) — trade-off is open/thin vs closed/fat. The best morphology balance is mid-drag, not extreme.
- **WINNER = s2 (fibre_wl32):** the coarser fibre wavelength yields the most elliptical (aspect 0.34) and most-rotated (angle 2.29) loops — a key parameter for matching real myocardium's anisotropic morphology.

**Failures:**
- **s4 (amplitude25) FAILED** — complete morphological collapse (open raw 0.013, size 1.09e-03, chir 0.07) in the forward atlas without inverse optimization. This shows amplitude is a SIZE lever ONLY in the inverse context (where the model learns structure around it); in a naive forward it is catastrophic overshoot.

**Next:** The 2×2 test + this atlas batch establish the Phase-1 evidence base. Batch 12 continues the MORPHOLOGY ATLAS with denser parameter sampling to build the families map: fibre wavelength sweep (16, 24, 32, 40 — charting the trend), fibre angle sweep (0, 0.3, 0.9 — full rotation effect), a safe amplitude test (amp15 instead of the collapsed amp25), and drag intermediate (60, 100 instead of extreme 300) — all under active-stress + uniform pulse. Parent = fibre_wl32 (the morphology winner); 6 one-knob variants from that base.

## Batch 12 — good (Phase 1 morphology atlas denser sampling; fibre_wl40 is the morphology leader) — 2026-06-24

Parent: aniso_atlas base (fibre_wl=32, from b11 s2 winner; the morphology pivot batch)
Hypothesis: "Phase 1 atlas (forward): probe fibre_wl wavelength sweep (coarser vs finer than parent 32) and amplitude scaling to build the morphology family map. Fibre_wl40 should increase ellipticity/angle further (coarser → more complex); amp15 tests size scaling; fibre_angle0 confirms angle-openness coupling."

Slot 0 [fibre_wl32]      parent fibre_wl=32                open=0.276 aspect=0.34 angle=2.29 size=5.26e-03 chir+=0.51  loops: moderate-open ellipses, radial beat
Slot 1 [fibre_wl24]      fibre_wl=24 (finer)                open=0.218 aspect=0.27 angle=1.74 size=5.22e-03 chir+=0.48  loops: tighter, less elliptical (fine→constrain)
Slot 2 [fibre_wl40]      fibre_wl=40 (coarser) ← WINNER     open=0.262 aspect=0.35↑ angle=3.06↑↑ size=5.42e-03 chir+=0.46  loops: most elliptical, most-rotated (beyond π/2)
Slot 3 [fibre_angle0]    fibre_angle=0.0 (no rotation)      open=0.322↑↑ aspect=0.32 angle=1.92 size=5.19e-03 chir+=0.45↓  loops: most open, less chirality (angle decouples)
Slot 4 [amplitude15]     amplitude=15 (3× amp10)            open=0.225 aspect=0.24 angle=0.00 size=5.10e-02↑↑↑ chir+=0.50  loops: 10× LARGER but closed/inertial (no learned structure)
Slot 5 [amplitude0_abl]  amplitude=0 (true ablation)        open=0.000 size=0.00  — FAILED, no motion (confirms active-stress required)

Ranking (combined morphology for Phase 2):
1. **s2 fibre_wl40 — WINNER** (aspect 0.35, angle 3.06 = most complex/elliptical; closest to real cardiomyocyte anisotropy)
2. s0 fibre_wl32 parent (balanced baseline; good aspect 0.34, good angle 2.29)
3. s4 amplitude15 (largest absolute size but collapsed openness & weird angle → inertial overshoot in forward)
4. s3 fibre_angle0 (most open but loses chirality/rotation structure)
5. s1 fibre_wl24 (intermediate; fine fibre constrains morphology)
6. s5 amplitude0_abl (FAILED)

Verdict:
- **Fibre wavelength is the PRIMARY MORPHOLOGY lever:** coarser wavelength monotonically increases both ellipticity (0.27→0.34→0.35) and major-axis angle (1.74→2.29→3.06 rad). Fibre_wl40 reaches the morphology extreme — it produces the most elliptical (aspect 0.35), most-rotated loops (angle approaching π). This is the target for Phase 2 inverse (likely better matched to real beat's rich anisotropy than the underdrive inertial loops of smaller fibre_wl or high-amplitude isotropic).
- **Fibre angle controls the openness-chirality TRADE:** removing angle (0.6→0.0) OPENS loops by +0.046 openness but REDUCES chirality from 0.51→0.45, showing rotation is load-bearing for handedness but inhibits openness. The coupling is real and decoupled.
- **Amplitude in forward-atlas has OPPOSITE EFFECT than inverse regime:** amp15 yields 10× bigger loops (size jumps from 5.3e-03 to 5.1e-02) BUT with collapsed openness (0.276→0.225) and angle=0 (weird alignment). The large amplitude drives isotropic overshoot WITHOUT learned structure (unlike inverse where UNet harnesses amp25 → −0.189 best R²). Forward morphology is inertial/ineluctible; inverse context has degrees of freedom to compensate.
- **Stiffness wavelength remains INERT** — batch 11 s3 (stiff_wl24) showed no visible morphological change, and this is confirmed by the coherent stiffness/direction patterns being insensitive to fibre_wl changes (the fibre GAIN/direction patterns dominate).
- **Ablation valid** — amplitude0 confirmed zero motion (true ablation).

Failures: s5 (amplitude0 ablation, expected zero motion — not a failure, expected result).

Next: Batch 13 = Phase 2 INVERSE begins. Parent = fibre_wl40 (morphology leader from b12 s2). Shift from forward atlas → inverse-train the wl40 pattern family on real beat: inverse-fit real trajectory + loop-morphology (openness/aspect/angle/size/chirality match), with UCB tree over ONE-knob variants (amp, fibre_angle, drag) on the wl40 base. Each slot changes ONE knob from the wl40 parent; the objective is REAL R² + morphology loss, not forward-atlas metrics. This tests whether the morphology-rich fibre_wl40 family can be tuned to match the real per-node beat distribution.

---

## Batch 13 — FAILED — 2026-06-24

**CRITICAL: ALL 6 SLOTS FAILED** due to trainer/config mismatch.

Intended config: Phase 1b forward MORPHOLOGY ATLAS (parametric variants on fibre_wl40).
Actual config: Plan specified `cardio_mpm_train.py` (inverse trainer) but included pattern args (`--stiff_wl`, `--gain_wl`, `--fibre_wl`, `--fibre_angle`) meant for `cardio_mpm_atlas.py` (forward atlas).

Exit codes: all 6 jobs exited with code 2 (argument parsing error).
Error example (s0): `cardio_mpm_train.py: error: unrecognized arguments: --stiff_wl 8 --gain_wl 26 --fibre_wl 40 --fibre_angle 0.6`

**ROOT CAUSE:** `cardio_mpm_plan.json` has `"train_script": "cardio_mpm_train.py"` but the configs contain forward-atlas pattern parameters. The trainer script does NOT accept these args directly (they are embedded in the spec files).

**Slots (all done=NO, R2=NA):**
- s0 [fibre_wl40_parent] — FAILED
- s1 [amplitude15] — FAILED
- s2 [amplitude20] — FAILED
- s3 [fibre_angle0.3] — FAILED
- s4 [drag60] — FAILED
- s5 [amplitude0_abl] — FAILED

**Failure diagnosis:** The batch was designed to run Phase-1b forward morphology atlas (sweeping amplitude/angle/drag on the fibre_wl40 base) but the plan incorrectly specified the inverse trainer. The config itself is VALID for `cardio_mpm_atlas.py` — all the args (`--stiff_wl 8`, `--gain_wl 26`, `--fibre_wl 40`, `--fibre_angle {0.3,0.6}`, `--amplitude {10,15,20,0}`, `--drag_k {30,60}`) are recognized by the atlas script.

**Fix:** Corrected `cardio_mpm_plan.json` to use `cardio_mpm_atlas.py` instead of `cardio_mpm_train.py`. Batch 14 (retried batch 13 with corrected trainer) will complete the Phase-1 morphology atlas by sweeping amplitude/angle/drag variants of the fibre_wl40 pattern family.

## Batch 14 — good (Phase 1 forward atlas; fibre_wl40 morphology landscape clarified) — 2026-06-24

**Runner:** `cardio_mpm_atlas.py` (Phase 1 forward atlas, NOT inverse).

**Parent config:** Base = `material_aniso_cardio`: stiff_wl 8, gain_wl 26, fibre_wl 40, fibre_angle 0.6, amplitude 10, drag_k 30.
Each slot changes ONE knob from the parent. Spec `material_aniso_cardio` (parametric active-stress patterns with learnable gain, no inverse training).

**Hypothesis:** "The fibre_wl40 pattern family produces the richest anisotropic morphology. Sweeping amplitude/angle/drag on this family maps the morphology landscape: how size/rotation/openness couple to these parameters. Phase 2 inverse will then target the real beat's morphology distribution by selecting the best family point(s)."

**Morphology metrics per slot (from progress.txt final line):**

| Slot | Config | open | aspect | angle | size | chir | Assessment |
|------|--------|------|--------|-------|------|------|------------|
| **s0** | **fibre_wl40_parent** (control) | **0.262** | **0.35** (elliptical) | **3.06** rad (highly rotated) | **5.42e-03** | **0.46** | **BALANCED WINNER** — best ellipticity & rotation, rich morphology |
| s1 | amplitude15 | 0.201 | 0.24 (flat) | **0.11** (lost rotation) | **3.94e-02** (7× EXPLODED) | 0.48 | **COLLAPSED INERTIAL** — high amp drives unstructured overshoot; aspect down, angle down, size exploded |
| s2 | amplitude20 | 0.270 | **0.06** (LINE) | 1.59 | **3.29e-03** (tiny) | **0.12** (lost chirality) | **DEGENERATE** — even worse; loops flatten & shrink, handedness lost |
| s3 | fibre_angle0.3 | 0.243 | 0.32 | 1.91 (half parent rotation) | 5.17e-03 | **0.58** (BEST chirality!) | CHIRALITY REVERSAL — angle down → handedness UP (opposite of Est.#29) |
| s4 | drag60 | **0.276** (highest open) | **0.36** (most elliptical) | **0.06** (LOST rotation) | 4.57e-03 | 0.51 | **QUASI-STATIC TRADE** — drag opens loops but kills transient rotation |
| s5 | amplitude0_abl | 0.000 | 0.00 | — | 0.00 | 0.00 | **ABLATION FAILS** — confirms active-stress required (Est.#16 re-confirmed) |

**Ranking by morphology quality:**
1. **s0 parent** (fibre_wl40): open 0.262, aspect 0.35, angle 3.06 — WINNER, balanced
2. **s4** (drag60): open 0.276, aspect 0.36, angle 0.06 — most open/elliptical, but lost rotation
3. **s3** (fibre_angle0.3): open 0.243, aspect 0.32, angle 1.91, chir 0.58 (best handedness) — rotation reduced, chirality flipped UP
4. **s1** (amplitude15): open 0.201, aspect 0.24, angle 0.11 — inertially collapsed, under-curved
5. **s2** (amplitude20): open 0.270, aspect 0.06, size 3.29e-03 — degenerate tiny lines, lost handedness
6. **s5** (amplitude0_abl): all zero — ablation

**Verdict:**
- **AMPLITUDE INVERSE EFFECT (Q24 FALSIFIED/REFINED):** The forward atlas contradicts the inverse finding where amp15/amp25 were winners. Here, amp15 immediately collapses (aspect ↓0.24, angle ↓0.11, size ↑3.94e-02 inertial overshoot) and amp20 is degenerate (aspect ↓0.06 flattened, size ↓3.29e-03 tiny, chir ↓0.12 lost). **The divergence:** forward atlas is unstructured random init, while inverse fits to real data with learned gain/direction structure that constrains inertial growth. ⇒ **Phase 2 should RETURN to the amp10–15 bracket, NOT amp20+.** In inverse, learned structure harnesses the amplitude; in forward random field, high amplitude drives pure inertial overshoot.
- **FIBRE ANGLE REVERSAL (Est.#29 FALSIFIED):** angle0.3 (vs parent 0.6) prediction was "angle=0 opens but kills handedness." Here angle 0.3 shows aspect 0.32 (similar to parent 0.35), angle 1.91 (half the rotation), BUT chirality UP to 0.58 (the BEST — opposite sign). The real data's chirality axis is orthogonal to fibre rotation; this parameter decouples handedness from rotation direction.
- **DRAG OPENS BUT KILLS ROTATION:** drag60 shows highest openness (0.276) and most elliptical (aspect 0.36), but angle 0.06 (lost transient rotation). This is the inertial→quasi-static trade: damping suppresses the transient force direction sweep that curves the loops.
- **ACTIVE-STRESS REQUIRED (Est.#16 RE-CONFIRMED):** amplitude0_abl shows all metrics zero — passive deformation alone cannot generate the loop morphology; active contraction is essential.
- **PHASE 1 ATLAS CONCLUSION:** The parent s0 (fibre_wl40) is the morphology leader for Phase 2 inverse. Next batch (b15) should switch back to **inverse fitting on the REAL beat**, using the fibre_wl40 base with learned gain/direction variants (constraints: amplitude 10–15, fibre_angle {0.3,0.6}, drag {30,60}) to match the real per-node morphology distribution on R² + loop-shape metrics.

**Next:** parent = s0 (fibre_wl40, the Phase-1 atlas winner); **PIVOT TO PHASE 2 INVERSE:** train on the real beat with the fibre_wl40 pattern family (material_aniso_cardio base), learning the per-particle gain + direction fields (no phase, no rotary, uniform pulse). Objective = interior R² + loop-morphology loss. Sweep gain_wl/fibre_angle/amplitude variants within the family to find the best real-trajectory fit.

---

## Phase2 Batch 1 [learn=fibre] — 2026-06-24

**Runner:** `cardio_mpm_train2.py` (Phase 2 PARAMETRIC inverse fit on the REAL beat).

**Parent config (s0):** Base = `material_aniso_cardio`: fibre_wl 40, fibre_angle 0.6, fibre_amp 1.0, fibre_phase 0.7, gain0 1.0, stiff [50,150], amplitude 10, drag_k 30, dur0 8, learn=fibre. 300 iterations. Outer band Dirichlet-anchored; interior free; metric = interior R² (motion-normalised, boundary excluded).

**Hypothesis:** "Sweeping fibre init params (angle, wavelength, amplitude) with --learn fibre isolates which fibre family best matches the real beat's trajectory shape. Fibre_wl40 (the Phase-1 atlas winner) is the expected leader; lower angle and finer wavelength may help by increasing spatial resolution for gradient-based tuning."

**Results per slot (from progress.txt final line + dashboard header converged params):**

| Slot | Config (ONE knob from parent) | R2 | ampL | open | chir | size | Converged fibre (wl/angle/amp/phase) | red-on-green |
|------|------|-----|------|------|------|------|------|------|
| **s5** | **fibre_amp=1.5** (init fibre amplitude UP) | **-5.448** | **0.315** | 0.258 | 0.61 | 1.46e-03 | 40.2 / 0.72 / **1.52** / 1.53 | smallest red loops, best contained; still OFF green (overshoot direction wrong) but LEAST overshoot |
| s3 | fibre_wl=28 (finer wavelength) | -8.267 | 1.132 | 0.295 | 0.53 | 1.81e-03 | 28.1 / 0.69 / **0.34** / 0.71 | red loops smaller, better contained than parent; closer to green; finer fibre stripes |
| s1 | fibre_angle=0.3 (less rotation) | -10.005 | 1.452 | 0.265 | 0.67 | 1.97e-03 | 40.3 / 0.35 / 1.00 / 1.01 | red loops overshoot green moderately; best chirality (0.67) |
| s0 | **fibre_parent** (control) | -14.451 | 2.907 | 0.255 | 0.59 | 2.35e-03 | 40.3 / 0.30 / **0.01**(!) / 1.50 | red MUCH bigger than green; large overshoot arcs; fibre_amp COLLAPSED to ~0 |
| s2 | fibre_angle=0.9 (more rotation) | -17.029 | 3.838 | 0.264 | 0.64 | 2.55e-03 | 40.4 / 0.80 / 0.84 / 0.75 | red very large; significant overshoot; more vertical stripes in fibre pattern |
| s4 | fibre_wl=52 (coarser wavelength) | -21.213 | 5.071 | 0.320 | 0.70 | 3.05e-03 | 51.8 / 0.59 / 0.90 / 1.50 | red MASSIVE (5× energy), worst; coarse fibre pattern can't constrain motion |

**Ranking by interior R² (higher = better):**
1. **s5 fibre_amp_1.5** (R²=-5.448, WINNER)
2. s3 fibre_wl_28 (R²=-8.267)
3. s1 fibre_angle_0.3 (R²=-10.005)
4. s0 fibre_parent (R²=-14.451)
5. s2 fibre_angle_0.9 (R²=-17.029)
6. s4 fibre_wl_52 (R²=-21.213)

**Winner: s5 (fibre_amp=1.5, R²=-5.448)**. Best R² AND best morphology containment (smallest loops, lowest ampL=0.315). BUT all R² are deeply negative — fibre alone (4 scalars) at amp=10/drag=30/dur=8 is far from fitting the real beat. The other levers (stiff UNet, gain, duration) are needed.

**Key findings:**

1. **fibre_amp is THE critical fibre parameter.** In s0 (parent), the optimizer COLLAPSED fibre_amp from 1.0→0.01, effectively KILLING fibre structure — the anisotropic pattern at default settings HURTS the fit (generates overshoot in wrong directions). Only s5 (init 1.5→converged 1.52) KEPT amp high AND won. High fibre_amp creates a denser interference pattern in the active stress that naturally CONSTRAINS motion → less overshoot → better R². The dashboard shows s5's fibre dx/dy as a tight diamond lattice with more nodes of destructive interference per unit area.

2. **fibre_wl is a SLOW gradient lever** — it barely moves from init in all slots (±0.4 in 300 iterations). Init placement matters: wl=28 (s3, finer) beats wl=40 (s0, parent) beats wl=52 (s4, coarser). The forward-atlas winner (wl=40, most elliptical) is NOT the inverse winner — more spatial resolution helps the parametric fit. BUT wl=28 with default fibre_amp=1.0 still partially collapses amp to 0.34.

3. **Fibre angle is anti-correlated with R²**: angle=0.3 (s1, -10.005) > angle=0.6 (s0, -14.451) > angle=0.9 (s2, -17.029). Lower rotation → less overshoot. BUT the s0 result is confounded by fibre_amp collapse (0.01); s1 maintained fibre_amp=1.0. The angle→R² relationship may partly reflect amp survival.

4. **ampL inversely correlated with R²**: s5 (ampL=0.315, under-driven) → best R². s4 (ampL=5.071, 5× over-driven) → worst R². At these settings (no rotary, amp=10, drag=30, dur=8), the active stress generates too much motion in wrong directions → LESS motion = BETTER. This echoes the Phase-1 finding that overshoot direction (not magnitude) gates R².

5. **dur=8 is FROZEN (learn=fibre only) and far below the period (~50).** The old inverse (Phase 1) found dur converges to ~47-53. At dur=8, the pulse is very short → sharp contraction spike → inertial ringing. This likely explains the poor R² regime: a 8-frame pulse at period 50 is a 16% duty cycle → violent kick → ring → overshoot.

**Verdict:** SUPPORTED — fibre params DO differentiate R² (4× range, -5 to -21), confirming the parametric inverse is sensitive to fibre init. FALSIFIED — the Phase-1 forward-atlas morphology winner (wl=40, angle=0.6, fibre_amp=1.0) is NOT the inverse winner; the optimizer kills its fibre_amp. The live finding is that HIGH fibre_amp (1.5) is needed for the fibre pattern to SURVIVE gradient descent and contribute positively.

**Next:** Batch 2 = `--learn stiff`. Freeze fibre at s5's converged values (wl=40.2, angle=0.72, amp=1.52, phase=1.53). Sweep stiffness range variants (stiff_lo/stiff_hi) and test whether the UNet-derived microscope stiffness pattern improves R² from the fibre-only baseline. Include one slot with s3's converged fibre (wl=28) to test whether finer fibre + stiff learning helps, and an amplitude=12 slot to address the under-driving (ampL=0.315). Ablation = uniform stiffness (stiff_lo=stiff_hi=100) to test whether SPATIAL stiffness variation matters.
