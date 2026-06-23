# Cardio Loop — Running Analysis

Per-batch analysis written by the agent-in-the-loop. Each batch: the 6 parallel cluster jobs, their
numbers (fit-RMSE / drift / `trans|max|`), the trajectory-morphology read of each dashboard, the
winner, and the reasoning for the next `cardio_plan.json`. The durable claims distilled from here
live in `knowledge.md`.

Seed plan (batch 1) crosses a **3-point learning-rate sweep** (`CARDIO_LR` = 2e-3 / 4e-3 / 8e-3 at
the parent config) with the 08_09 limit-cycle stability probes (strong drift, lower transverse
scale + higher openness, and a `LAM_DRIFT=0` ablation that should reproduce the 08_08 central
streaks). LR is a routine axis because multi-cycle gradients are noisier (Q1/Q2/Q6 in `knowledge.md`).

---

<!-- agent appends dated batch sections below -->

## Batch 1 — plan review (pre-launch, 2026-06-22)

Reviewed the seed `cardio_plan.json` against `knowledge.md` (Q1/Q2/Q6) and the 08_09 train
script. Kept it unchanged. Rationale:

- All env knobs are valid (`CARDIO_LR` line 80; `N_BEATS`, `LAM_DRIFT`, `TRANS_SCALE`,
  `LAM_OPEN`, `N_ITER`, `CKPT_EVERY`, `MAX_SECONDS`). `PERIOD=T_FIT` is auto-set in code
  (line 66), so Q1's train/render-period match needs no knob.
- The 6 configs form **two clean orthogonal sweeps crossing at the parent** (`parent_lr4e3`):
  - **LR sweep {2e-3, 4e-3, 8e-3}** at fixed `(drift2, scale0.12, open2, beats2)` → answers
    Q6 (multi-cycle gradients are noisier; isolate divergence/stall as LR artefact vs model).
  - **LAM_DRIFT sweep {0, 2, 5}** at fixed `LR4e3` → answers Q1; `ablation_drift0` is the
    Principle-6 control that should reproduce the 08_08 central streaks.
  - `scale08_open3_lr4e3` adds one Q2 openness-vs-drift probe.
- Trade-off accepted: dropped an explicit `N_BEATS=3` slot to fit the LR sweep in. Q6 asks
  for ≥2 LR points per batch and N_BEATS can be probed next batch — good ordering.

Read targets when results land: per job `dashboard_<last>.png` (red-on-green superposition?
loops open or collapsed? central streaks?), `true_vs_learned.mp4` (cross-cycle drift), and
`run.log` (fit-RMSE / drift / `trans|max|`). Adjudicate Q6 from the LR sweep first (discard
LR-diverged runs before reading the science), then Q1 from the drift sweep, Q2 from the
scale/open probe.

## Batch 1 — FAILED (batch-wide external kill, no science) — 2026-06-22
Parent/control: slot 0 = LR4e3, beats2, drift2, scale0.12, open2, N_ITER1500, ckpt300
Hypothesis tested: "Multi-cycle (N_BEATS=2)+LAM_DRIFT removes central streaks while keeping loops
open; the LR sweep isolates divergence as an LR artefact" (Q1/Q2/Q6) — **NOT TESTED**, see below.

ALL SIX JOBS FAILED before completing a single training iteration. No `run.log`, no
`true_vs_learned.png`, no trained dashboard exists for any slot — so there is **zero** trajectory
evidence and NO hypothesis can be adjudicated. Verdicts on Q1/Q2/Q6 remain exactly as before.

Evidence captured before the loop driver wiped the result dirs:
- Slots 0,1,2,3: only `config.json` + `checkpoints/dashboard_00000.png` + `model_00000.pt`
  (the iteration-0 init checkpoint — written at setup, BEFORE any optimisation step).
- Slots 4 (scale08), 5 (ablation_drift0): only `config.json`; their `.sh` existed but no `.out`/
  `.err` were ever produced → they never started executing (queued/launched then killed).
- `loop_logs/*.err` (slots 0–3): `CondaError: KeyboardInterrupt`.
- `loop_logs/*.out` (LSF summaries, slots 0–3): `TERM_OWNER: job killed by owner. Exited with exit
  code 1.` Every job started 21:32:5x and **all terminated at the same instant, 21:40:55** (~8 min
  wall). CPU time ~460 s, Max Memory ~1.8 GB (no substantial GPU tensor work — consistent with
  setup/imports + the iter-0 dashboard render only).

Diagnosis: a **single external termination of the whole batch ~8 min in** (simultaneous kill across
all hosts + `KeyboardInterrupt` propagated through conda = the launcher/controller was interrupted
and `bkill`ed its children). This is NOT a config bug (the six `config.json` match the plan with
clean one-knob deltas) and NOT a code bug (the script reached data-load, model-init, and wrote the
iter-0 dashboard for the 4 jobs that got scheduled). The science knobs were never exercised.

Winner: none (no trained job).
Verdict: inconclusive — batch invalid; re-run required.
Failures: all 6. Cause = external batch kill at 21:40:55, not LR/config/model.
Next: RE-RUN the identical, well-formed scientific plan (it is the right experiment), with one
non-science robustness change applied uniformly to all 6 slots: `CARDIO_CKPT_EVERY` 300→100 so that
if an early kill recurs we still capture a *trained* dashboard (evidence) rather than only iter-0.
Causality preserved — the one-knob deltas between slots are unchanged; only a uniform diagnostic
cadence differs. Parent unchanged otherwise.

## Batch 1 (valid re-run) — mixed (rich science, 4/6 NaN-frozen) — 2026-06-23
Parent/control: slot 0 = LR4e3, beats2, drift2, scale0.12, open2, lam_trans0.3, N_ITER1500
Hypothesis tested: "Multi-cycle (N_BEATS=2)+LAM_DRIFT removes the central streaks while keeping
loops open; the LR sweep isolates divergence as an LR artefact" (Q1/Q2/Q6).

All 6 jobs ran (LSF "Successfully completed", no external kill). BUT four diverged to a
**forward-pass NaN** mid-training and then skipped every remaining step (`isfinite(loss)` guard at
line 455), so their "final" model is FROZEN at the last good iter. Gradient clipping was already ON
(`CARDIO_GRAD_CLIP=1.0`, line 459) and did NOT help — the NaN is in the forward rollout, not the
gradient. NaN onset: slot0 it298, slot2 it331, slot3 ~it360, slot4 ~it450.

Slot 0 [parent lr4e3]   cfg=parent            fitRMSE=0.0088 drift=0.00085 trans|max|=0.115 γ=0.20  NaN-frozen@it250  loops=open+SUPERPOSE (best) streaks=faint vertical
Slot 1 [lr2e3]          cfg=LR 4e3→2e3        fitRMSE=0.0048 drift=0.00033 trans|max|=0.114 γ=0.23  COMPLETE@1450     loops=open NOT-superposed streaks=MASSIVE (whole field)
Slot 2 [lr8e3]          cfg=LR 4e3→8e3        fitRMSE=0.0128 drift=0.00109 trans|max|=0.120 γ=0.218 NaN-frozen@it300  loops=open streaks=DENSE diagonal (worst)
Slot 3 [drift5_lr4e3]   cfg=LAM_DRIFT 2→5     fitRMSE=0.0085 drift=0.00060 trans|max|=0.118 γ=0.179 NaN-frozen@it350  loops=open+superpose streaks=vertical
Slot 4 [scale08]        cfg=TRANS_SCALE .12→.08 fitRMSE=0.0057 drift=0.00050 trans|max|=0.078 γ=0.18 NaN-frozen@it430 loops=open+superpose streaks=diagonal
Slot 5 [ablation_drift0] cfg=LAM_DRIFT 2→0    fitRMSE=0.0043 drift=0.00363 trans|max|=0.120 γ=0.70  COMPLETE@1450     loops=open NOT-superposed streaks=YES (textured fields)

Winner: slot 1 (lr2e3) — the only numerically VALID *fully-trained* drift>0 run; lowest fitRMSE
(0.0048) and lowest drift metric (0.00033) among complete jobs → new parent/control. BUT its free
render streaks badly: low per-beat fitRMSE (boundary is Dirichlet-pinned) does NOT imply a stable
free rollout. Morphologically the BEST render is slot 0 (frozen at it250) — small red loops
superposing on green node-by-node — but that is an early-stop artefact of the NaN, not a trained
optimum.

Verdict:
 - Q6 — SUPPORTED→ESTABLISHED. Two orthogonal axes agree: in the LR sweep at drift2, lr2e3 trained
   clean while lr4e3 AND lr8e3 NaN'd ~it300; in the drift sweep at lr4e3, drift0 trained clean while
   drift2 AND drift5 NaN'd. The destabiliser is the LAM_DRIFT(>0)×LR(≥4e-3) interaction driving the
   overdamped integrator (small γ≈0.18–0.20, growing β/A_ij) into a forward blow-up. Stable operating
   point for multi-cycle+drift training: LR≤2e-3.
 - Q1 — PARTIAL FALSIFICATION. LAM_DRIFT=2 cut the drift METRIC 10× vs drift0 (slot1 0.00033 vs
   slot5 0.00363) but BOTH fully-trained renders streak. So the periodicity loss lowers the measured
   per-step drift yet does NOT remove the free-rollout streaks. Caveat: slot1-vs-slot5 is confounded
   by LR (2e3 vs 4e3); a clean drift0-at-lr2e3 ablation is required (next batch).
 - NEW (Q7) — streaks GROW with training: every model frozen at it~250–450 superposes small loops
   cleanly; both models trained to it1450 explode into streaks. The per-beat (pinned) fit objective
   and free-rollout stability DIVERGE with continued training.
Failures: slots 0,2,3,4 = forward-pass NaN (not external kill, not config error). Root cause is the
LAM_DRIFT×LR instability above; the CKPT_EVERY=100 robustness change from the failed attempt worked
(we caught trained dashboards + frozen renders, so the runs are scientifically usable).
Next (Batch 2): parent = lr2e3 drift2 scale0.12 (proven-stable). Attack the STREAKS (the real target)
at the safe LR via the transverse-magnitude mechanism (TRANS_SCALE 0.08/0.06 + LAM_TRANS hinge) and a
CLEAN drift0 ablation (lr2e3, one-knob — fixing this batch's LR confound), plus a drift5 point.

## Batch 2 — mixed (rich science; TRANS_SCALE lever confirmed, γ-instability now the root blocker) — 2026-06-23
Parent/control: slot 0 = lr2e3, beats2, drift2, scale0.12, open2, lamtrans0.3, N_ITER1500 (batch-1 slot1)
Hypothesis tested: "At the stable LR=2e-3, the free-rollout streaks are driven by transverse-field
MAGNITUDE growth, so lowering TRANS_SCALE (0.08, 0.06) and/or strengthening LAM_TRANS reduces streaks
more than LAM_DRIFT does; a clean drift0-vs-drift2-vs-drift5 ablation at fixed LR isolates LAM_DRIFT."
(Q1 reframed / Q2 / Q7; F5-revised.)

Slot 0 [parent scale0.12] cfg=control                 fitRMSE=0.00646 drift=0.00046 trans|max|=0.107 γ=0.183  it700/1500  loops=open+mostly-superpose streaks=diagonal(moderate)  NaN-FROZE@700
Slot 1 [scale008]         cfg=TRANS_SCALE .12→.08      fitRMSE=0.00480 drift=0.00042 trans|max|=0.075 γ=0.165  it1000/1500 loops=open+superpose         streaks=vertical(reduced)   NaN-FROZE@1000
Slot 2 [scale006]         cfg=TRANS_SCALE .12→.06      fitRMSE=0.00326 drift=0.00025 trans|max|=0.056 γ=0.179  it1250/1500 loops=SHRINKING/near-collapse streaks=faint hatch(BEST)    NaN-FROZE@1250
Slot 3 [lamtrans1]        cfg=LAM_TRANS .3→1.0         fitRMSE=0.00562 drift=0.00041 trans|max|=0.113 γ=0.275  it1450/1500 loops=buried in spaghetti     streaks=EXPLODED(worst)      COMPLETE
Slot 4 [drift5]           cfg=LAM_DRIFT 2→5            fitRMSE=0.00607 drift=0.00039 trans|max|=0.113 γ=0.183  it1250/1500 loops=open+superpose         streaks=diagonal(moderate)   NaN-FROZE@1250
Slot 5 [ablation_drift0]  cfg=LAM_DRIFT 2→0            fitRMSE=0.00451 drift=0.00127 trans|max|=0.113 γ=0.372  it1450/1500 loops=open but spaghetti      streaks=EXPLODED             COMPLETE

Morphology (the primary evidence; A_ij saturates to +1 EVERYWHERE in all 6 → carries ~no structure, Q3):
- The two runs that TRAINED TO COMPLETION (s3, s5 — both happened to keep γ HIGH, 0.275/0.372, so the
  integrator stayed stable) both EXPLODE into red spaghetti in true_vs_learned. Every run that NaN-FROZE
  early (s0@700, s1@1000, s2@1250, s4@1250) keeps open, mostly-superposing loops with only modest streaks.
- TRANS_SCALE sweep is a clean monotone lever: 0.12→0.08→0.06 lowered trans|max| (0.107→0.075→0.056),
  drift (0.00046→0.00042→0.00025), fitRMSE (0.00646→0.00480→0.00326), AND render streak amplitude (s2
  faintest). COST: at 0.06 the loops shrink toward collapse (s2 dashboard: small loops, dim gain panel).
  Crucially s2 trained to it1250 yet stayed faint — so low trans suppresses streaks even at long training,
  decoupling streak-growth (Q7) from pure iteration count: it is transverse-MAGNITUDE-gated.
- LAM_TRANS=1.0 (s3) did NOT tame the tail; it landed in a different basin (high γ, distinct fibre/gain
  maps) and gave the WORST render. Hinge lever falsified.
- Drift sweep: drift>0 lowers the drift METRIC ~3× (s0/s4 ~0.0004 vs s5 drift0 0.00127), reconfirming
  batch 1. But the render comparison is again confounded by Q7 — s5 (drift0) trained to 1450 and exploded
  while s0 (drift2) froze at 700 and looks clean. LAM_DRIFT still shows no render benefit.

Winner: numerically slot 2 (scale006) — lowest fitRMSE 0.00326, lowest drift 0.00025, faintest streaks.
BUT its loops are collapsing, failing the openness objective. Best openness↔stability COMPROMISE =
slot 1 (scale008, trans0.075): loops stay open AND streaks drop → new parent/control.

Verdict:
 - Q2 / batch-2 hypothesis — SUPPORTED. TRANS_SCALE is THE effective streak lever (monotone in trans,
   drift, fitRMSE AND streak amplitude); LAM_DRIFT and LAM_TRANS are not. A clear openness↔stability
   Pareto front: trans≈0.075 keeps loops open with reduced streaks; trans≈0.056 is near-streak-free but
   collapses loops. F5-revised (streaks are transverse-magnitude-driven) UPHELD.
 - Q7 — RE-CONFIRMED and SHARPENED. The only fully-trained runs explode; early-frozen runs look clean.
   But s2 shows the explosion is gated by transverse MAGNITUDE, not iteration count alone.
 - Principle 8 — REFINED / PARTIALLY FALSIFIED. The NaN-freeze RECURS at LR=2e-3 (4/6 froze), so
   "LR≤2e-3 is safe" is wrong as stated. The true determinant is γ: every frozen run had γ≈0.165–0.183;
   both completers had γ≥0.275. The explicit step dt/γ (line 194, clamp floor only 1e-2) blows up once
   training drives γ below ~0.2. → New Principle 10; the cure is a γ-floor in the integrator (Q8), not LR.
Failures: none missing data — all 6 produced run.log + dashboards + renders. But 4/6 NaN-froze
(uncontrolled early-stop), which now confounds EVERY render comparison via Q7. This is the batch's main
methodological problem and the next batch's target.
Next (Batch 3): EDITED CODE — added env-gated γ-floor in the integrator (CARDIO_GAMMA_FLOOR, line 194;
default 0 = exact reproduction). New parent = scale008 (trans0.075, no floor, control). Slots: γ-floor
0.25 & 0.35 (Q8 — does flooring γ stop the NaN-freeze so runs train to completion WITHOUT exploding?),
trans 0.06 & 0.12 (re-walk the Pareto without the freeze confound), and an N_ITER=600 early-stop slot
(direct Q7 isolation). Dropped the saturated LAM_TRANS (falsified) and drift5 levers.

## Batch 3 — good (decisive: γ-floor CURES the freeze; full training then EXPLODES) — 2026-06-23
Parent/control: slot 0 = lr2e3, beats2, drift2, scale0.08 (trans0.075), open2, lamtrans0.3, GAMMA_FLOOR=0, N_ITER1500 (batch-2 slot1)
Hypothesis tested: "The NaN-freeze is a γ-threshold instability (Principle 10), so flooring the effective
γ (0.25, 0.35) lets ALL runs train to completion without NaN. Once the freeze confound is gone, streaks
are governed purely by transverse magnitude (Principle 11): a γ-floored run at trans0.075 should stay
clean to it1500. An N_ITER=600 early-stop isolates any fit-vs-rollout divergence beyond magnitude (Q7)."
(Q8 / Principle 10 / Q7.)
CODE confirmed: GAMMA_FLOOR at cardio_train08_09.py:78, applied at :196 `dt/gamma.clamp(min=max(1e-2,FLOOR))`.

NOTE on metrics: run.log this batch omits the fit/drift line; numbers below are from progress.txt
(last logged iter) + the .out trace. "Frozen" = forward NaN then every later step skipped (params pinned).

Slot 0 [parent gfloor0]   cfg=control GAMMA_FLOOR=0      fitRMSE=0.00456 drift=0.00140 trans|max|=0.076 γ=0.181  NaN-FROZE@908  loops=open+superpose streaks=faint diagonal (clean — early-stop ARTEFACT)
Slot 1 [gfloor025]        cfg=GAMMA_FLOOR 0→0.25         fitRMSE=0.00611 drift=0.00051 trans|max|=0.076 γ=0.216  COMPLETE@1500  loops=buried streaks=EXPLODED (heavy spaghetti)
Slot 2 [gfloor035]        cfg=GAMMA_FLOOR 0→0.35         fitRMSE=0.00443 drift=0.00037 trans|max|=0.076 γ=0.300  COMPLETE@1500  loops=many open+superpose streaks=moderate (LEAST-exploded completer)
Slot 3 [scale006]         cfg=TRANS_SCALE 0.08→0.06      fitRMSE=0.00381 drift=0.00025 trans|max|=0.054 γ=0.163  NaN-FROZE@871  loops=small/collapsing streaks=moderate (frozen ARTEFACT)
Slot 4 [scale012]         cfg=TRANS_SCALE 0.08→0.12      fitRMSE=0.00580 drift=0.00040 trans|max|=0.108 γ=0.163  NaN-FROZE@900  loops=small+superpose streaks=faint (frozen ARTEFACT)
Slot 5 [earlystop600]     cfg=N_ITER 1500→600           fitRMSE=0.00590 drift=0.00064 trans|max|=0.066 γ=0.185  COMPLETE@600   loops=buried streaks=EXPLODED (heavy spaghetti)

Morphology (primary evidence; A_ij STILL saturates to +1 everywhere in all 6 → ~no structure, Q3; φ keeps blob structure):
- THE result: the two γ-floored runs (s1, s2) trained to it1500 with ZERO non-finite steps. All three
  GAMMA_FLOOR=0 runs NaN-froze, every one at γ≈0.16–0.18 (s0@908 γ0.181, s3@871 γ0.163, s4@900 γ0.163).
  This is the clean confirmation of Principle 10 (freeze fires when γ<~0.2) AND validates the cure. → Q8 +.
- BUT the cure UNMASKS the real problem: the fully-trained floored runs EXPLODE in the free render. s1
  (floor0.25) is heavy red spaghetti; s2 (floor0.35) is the least-exploded completer but still streaks.
  So every "clean" early-frozen render in batches 1–2 was an uncontrolled-early-stop artefact — confirmed
  here by s0 (froze@908, looks clean) vs s5 (explicit clean 600-iter run, EXPLODES). Removing the freeze
  did NOT yield a clean trained optimum. The fit-vs-rollout divergence is genuine and grows with training.
- NEW LEVER: damping. s2 (floor0.35) explodes visibly LESS than s1 (floor0.25): larger effective γ →
  smaller integrator step dt/γ → gentler free rollout. Distinct from TRANS_SCALE. Caveat: the higher floor
  also shifted the learned basin (s2 eps0.79/eta0.45/aniso0.80 vs s1 eps0.67/eta0.24/aniso0.70), so
  damping-vs-basin must be disentangled by a dedicated dose sweep (Batch 4).
- s5 (explicit early-stop @600, trans0.066) EXPLODES → direct Q7 isolation: a cleanly-trained 600-iter run
  already streaks. The clean look of frozen runs is the FREEZE (params pinned pre-blowup), NOT iteration
  count. Early-stop is not a usable stability route. Falsifies "an explicit early-stop renders clean" (F7).
- s3/s4 Pareto re-walk STILL confounded: both froze (GAMMA_FLOOR=0). Lesson: TRANS_SCALE must be swept
  WITH the floor on. The frozen s3 (trans0.054) shows collapsing loops, s4 (trans0.108) faint streaks —
  but as frozen artefacts these don't calibrate the Pareto.

Winner: slot 2 (gfloor035) — the best fully-trained run: floor stopped the freeze, lowest completer
fitRMSE (0.00443) and drift (0.00037), and the least-exploded render of the completers (loops open and
many superpose). New parent/control. (s3 has lower fitRMSE 0.00381 but is a frozen artefact — disqualified.)

Verdict:
 - Q8 / Principle 10 — RESOLVED POSITIVE (decisive). The γ-floor eliminates the NaN-freeze: both floored
   runs completed 1500 with no non-finite step; all three unfloored runs froze at γ≈0.16–0.18. The
   dt/γ explicit-step mechanism and the γ<~0.2 threshold are confirmed; flooring γ is the cure. → Principle 12.
 - Q7 — RESOLVED. The freeze WAS masking instability. Fully-trained floored runs (s1,s2) AND a clean
   early-stop (s5) all explode; only frozen runs look clean. Fit-vs-rollout divergence is real, persists
   beyond transverse magnitude, and intensifies with training. "Remove the freeze → clean trained optimum"
   is FALSE. The remaining target is this divergence itself (next batch: damping + a now-valid drift test).
 - NEW (damping lever) — γ-floor magnitude modulates streak amplitude (s2 floor0.35 < s1 floor0.25). One
   batch only → Open Question (Q9); a 0.35/0.45/0.55 dose sweep next batch to promote/falsify.
Failures: none missing data — all 6 produced run.log + dashboards + renders. 3/6 NaN-froze (the unfloored
slots), exactly as the hypothesis predicted; this is now an expected, understood, and CURABLE event.
Next (Batch 4): parent = gfloor035. The freeze is solved — attack the genuine streak divergence. Slots
(one-knob from parent): γ-floor 0.45 & 0.55 (damping dose-response, Q9), TRANS_SCALE 0.06 WITH floor (clean
Pareto re-walk, Q2), LAM_DRIFT 6 and LAM_DRIFT 0 (Q1 REVISITED — now that runs fully train, does the
periodicity loss FINALLY cut the render streaks? this was un-testable while runs froze). Dropped the
N_ITER early-stop probe (Q7 resolved) and the unfloored TRANS slots (must carry the floor now).

## Batch 4 — excellent (decisive: DAMPING is the streak lever; periodicity falsified clean) — 2026-06-23
Parent/control: slot 0 = lr2e3, beats2, drift2, scale0.08(trans0.075), open2, lamtrans0.3, GAMMA_FLOOR=0.35, N_ITER1500 (batch-3 slot2)
Hypothesis tested: "With the γ-floor standard (P12), the residual streaks (F7/Q7) are an integrator-step
instability, so RAISING the floor (0.45, 0.55) monotonically reduces render-streak amplitude (Q9).
Independently — now that ALL runs fully train (no freeze confound) — a LAM_DRIFT {0,2,6} sweep can finally
ask on the RENDER (not the metric) whether periodicity helps (Q1 revisited / F5). Plus a clean trans0.06
Pareto re-walk WITH the floor (Q2)." (Q9 / Q1 / Q2.)
metrics from progress.txt (final it1450); morphology from dashboard_01400 + true_vs_learned.png. ALL 6 trained
to it1450 with ZERO NaN-freeze (floor≥0.35 on every slot) → P12 re-confirmed across 6/6.

Slot 0 [parent gfloor035] cfg=control FLOOR0.35       fitRMSE=0.00361 drift=0.00031 trans|max|=0.073  loops=open+superpose streaks=HEAVY (worst of damping arm; long red spans across centre)
Slot 1 [gfloor045]        cfg=GAMMA_FLOOR 0.35→0.45    fitRMSE=0.00351 drift=0.00033 trans|max|=0.072  loops=open+superpose streaks=MODERATE (fewer/shorter spans than s0)
Slot 2 [gfloor055]        cfg=GAMMA_FLOOR 0.35→0.55    fitRMSE=0.00221 drift=0.00040 trans|max|=0.070  loops=open+superpose streaks=LEAST (cleanest render of batch; loops contained) — WINNER
Slot 3 [trans006]         cfg=TRANS_SCALE 0.08→0.06    fitRMSE=0.00321 drift=0.00028 trans|max|=0.058  loops=open+superpose streaks=HEAVY (still long spans — trans drop did NOT clean render)
Slot 4 [drift6]           cfg=LAM_DRIFT 2→6            fitRMSE=0.00363 drift=0.00023 trans|max|=0.078  loops=open+superpose streaks=HEAVY (no better than drift2; lowest drift METRIC yet)
Slot 5 [ablation_drift0]  cfg=LAM_DRIFT 2→0            fitRMSE=0.00244 drift=0.00060 trans|max|=0.074  loops=open+superpose streaks=MODERATE (≈drift6, NOT worse than it)

Morphology (the primary evidence; A_ij STILL saturates to +1 EVERYWHERE in all 6 → ~no structure, Q3; φ keeps blob structure in all 6):
- DAMPING DOSE-RESPONSE (the result): floor 0.35→0.45→0.55 (s0→s1→s2) gives a MONOTONE reduction in
  render-streak amplitude — s0 is heavy red spaghetti spanning the centre, s1 noticeably tamer, s2 the
  cleanest render produced by the loop so far (loops bounded, many open+superposing on green). fitRMSE also
  IMPROVES with damping (0.00361→0.00351→0.00221); trans|max| is flat (~0.07) so this is NOT a magnitude
  effect — it is the integrator step dt/γ shrinking as the floor rises. CONFIRMS the batch-3 hint (s2<s1),
  now as a clean 3-point trend → Q9 promoted (Principle 13). Caveat: s2 also sits in a shifted FHN basin
  (eps0.166/eta0.05/a0.948 vs parent eps0.745) and smoother stiffness/gain maps — basin-vs-damping is not
  100% separated, but a monotone 3-point dose curve is far stronger evidence than batch-3's confounded pair.
- LEARNED γ IS MASKED BY THE FLOOR: run.log reports the learned gamma scalar = 0.300 (=init) in ALL 6 slots.
  Every floor (0.35/0.45/0.55) exceeds it, so the integrator damping is set ENTIRELY by the floor and the
  learned γ is decoupled (its gradient vanishes below the floor). i.e. GAMMA_FLOOR is now the sole damping DOF.
- PERIODICITY (Q1/F5) — the clean test, finally valid: drift {0,2,6} (s5/s0/s4) cut the drift METRIC
  monotonically (0.00060→0.00031→0.00023) but did NOTHING for the RENDER — drift6 (s4, lowest metric ever)
  is heavily streaked, and drift0 (s5) is NOT worse than drift6 (if anything s5 is tamer). This is the FIRST
  drift sweep where every run fully trained, so the F5 "no render benefit" verdict can no longer be dismissed
  as a freeze artefact (the batch-1/2 escape hatch). → F5 promoted to Established (Principle 14); Q1 closed-negative.
- PARETO (Q2) — surprising: trans0.06 WITH the floor (s3) still streaks HEAVILY, despite the lowest-but-one
  trans|max| (0.058) and a good fitRMSE (0.00321). So with full training, lowering TRANS_SCALE 0.08→0.06 does
  NOT clean the render — yet RAISING the floor does. This PARTIALLY REVISES Principle 11: batch-2's "TRANS_SCALE
  is the master streak knob" was measured on NaN-FROZEN runs (early-stop artefacts); under full training the
  dominant streak lever is DAMPING, not transverse magnitude. (Principle 11's openness↔magnitude trade-off for
  loop SIZE still stands; its claim over STREAKS is downgraded.)

Winner: slot 2 (gfloor055) — cleanest render in the program to date, BEST fitRMSE (0.00221) and low drift
(0.00040), loops open and many superpose. New parent/control. (s5 drift0 has comparable fitRMSE 0.00244 but
its drift metric is 2× and its render is no cleaner — and it ablates a mechanism we keep, so not the parent.)

Verdict:
 - Q9 / damping hypothesis — SUPPORTED → ESTABLISHED (Principle 13). Clean monotone 3-point dose-response
   (floor 0.35>0.45>0.55 in streak amplitude) across two batches (b3 hint + b4 dose); mechanism understood
   (larger floor → smaller explicit step dt/γ → gentler free rollout). Damping is the first lever that
   reduces RENDER streaks AND improves fitRMSE simultaneously.
 - Q1 / F5 — RESOLVED NEGATIVE → ESTABLISHED (Principle 14). On fully-training runs the LAM_DRIFT periodicity
   loss lowers the per-step drift METRIC monotonically but has ZERO beneficial effect on the free-rollout
   streaks (drift6 streaks as badly as drift0). The metric and the render are decoupled. Q1 (the 08_09 premise)
   is closed: periodicity is not the route to a stable rollout.
 - Q2 / Principle 11 — PARTIALLY REVISED. Under full training trans0.06 still streaks; TRANS_SCALE is NOT the
   dominant streak lever once the freeze confound is gone — damping is. P11's streak claim downgraded to a
   freeze artefact; its loop-SIZE (openness) trade-off retained.
 - Q3 — unchanged. A_ij saturates to +1 in all 6; φ carries blob structure but its causal role is still unprobed.
Failures: none — all 6 produced run.log + progress.txt + dashboards + renders, all trained to it1450, zero
NaN. The γ-floor (P12) has made the freeze a non-event: this is the first fully-clean 6/6 batch.
Next (Batch 5): parent = gfloor055. Damping is the live lever — continue the dose (floor 0.65, 0.75: does
streak reduction keep going or saturate / over-damp into under-fit + collapsed loops?), anchor it with a
floor0.35 low-damping ABLATION (should re-streak → confirms monotonicity, isolates damping vs basin), test
whether low-trans helps ONCE damping is high (trans0.06 at floor0.55), and probe whether the freed stability
budget lets us push openness back up (LAM_OPEN 2→4). Drop the drift sweep (P14 settled) and LR (P10/Q6: floor,
not LR, governs stability — 6/6 stable at lr2e-3 this batch).

## Batch 5 — good (decisive REFRAME: loops OVERSHOOT; calibrate-DOWN beats open-UP) — 2026-06-23
Parent/control: slot 0 = lr2e3, beats2, drift2, scale0.08(trans0.075), open2, lamtrans0.3, GAMMA_FLOOR=0.55, N_ITER1500 (batch-4 slot2)
Hypothesis tested: "Damping (γ-floor) reduces streaks by shrinking dt/γ (P13), so RAISING it (0.65,0.75) keeps
reducing streaks until an over-damp turnover (fitRMSE rises, loops collapse); a floor0.35 anchor re-streaks
(isolates damping vs basin); with streaks now damping-controlled, LAM_OPEN 2→4 re-enlarges loops WITHOUT
re-streaking; trans0.06 stacks with high damping." (Q9b / Q2-reframed.)
ALL 6 trained to it1450, ZERO NaN-freeze (floor≥0.35 everywhere → P12 re-confirmed 6/6). gamma scalar = 0.300
(=init) in ALL 6 → floor is the sole damping DOF (P13 corollary re-confirmed). A_ij saturates to +1 EVERYWHERE
in all 6 (Q3 unchanged); φ keeps blob structure in all 6.

CRITICAL CAVEAT (user_input.md, this batch): fitRMSE is boundary-diluted and is NOT a goodness measure — the
honest metric is render-R2 on the free rollout (interior+moving nodes), and the new R2-logging code takes effect
only from the NEXT batch (batch 6). Batch 5 ran the OLD script: NO R2 line in run.log/progress.txt. So this batch
is RANKED ON MORPHOLOGY (overshoot/streak amplitude in true_vs_learned.png = the direct R2 proxy: red overshooting
green ⇒ R2<0). The boundary-diluted fitRMSE numbers are reported for the record ONLY, not used for ranking. (An
offline R2 script `_compute_r2_b05.py` is staged but its python run is sandbox-gated/unrun this session.)

Slot 0 [parent gfloor055] cfg=control FLOOR0.55      fitRMSE=0.00471 drift=0.00057 trans|max|=0.077  loops=open streaks=HEAVY (long red spans across centre; red overshoots green badly)
Slot 1 [gfloor065]        cfg=GAMMA_FLOOR 0.55→0.65   fitRMSE≈0.0044  drift≈0.0005  trans|max|=0.073  loops=open streaks=HEAVY (spaghetti; ≈ s0, not visibly cleaner)
Slot 2 [gfloor075]        cfg=GAMMA_FLOOR 0.55→0.75   fitRMSE=0.00374 drift=0.00053 trans|max|=0.071  loops=open (NOT collapsed) streaks=MODERATE (cleaner than s0/s1; fitRMSE still falling → no over-damp turnover yet)
Slot 3 [ablation_gfloor035] cfg=GAMMA_FLOOR 0.55→0.35 fitRMSE≈0.0046  drift≈0.0006  trans|max|=0.075  loops=open streaks=HEAVY (big sweeping arcs; re-streaks → CONFIRMS the damping dose)
Slot 4 [lamopen4]         cfg=LAM_OPEN 2→4            fitRMSE≈0.0050  drift≈0.0006  trans|max|=0.074  loops=open+ENLARGED streaks=WORST (bigger loops = bigger overshoot)
Slot 5 [trans006]         cfg=TRANS_SCALE 0.08→0.06   fitRMSE≈0.0040  drift≈0.0005  trans|max|=0.052  loops=open (smaller) streaks=LEAST of batch (tamest excursions; most red ellipses sit on green) — WINNER
(s1/s3/s4 fitRMSE read off progress.txt cadence; s0/s2 exact. Numbers are boundary-diluted — see caveat.)

Morphology (the primary evidence, = R2 proxy):
- DAMPING DOSE CONTINUATION (s0/s1/s2: floor 0.55→0.65→0.75): s2 (0.75) is the cleanest of the three and its
  fitRMSE keeps falling (0.00471→0.00374); the loops are still OPEN, NOT collapsed. So damping has NOT hit an
  over-damp turnover by 0.75 — it keeps helping with DIMINISHING returns. BUT no pure-damping run is clean; all
  three still streak. ⇒ damping is necessary, monotone, but INSUFFICIENT ALONE to calibrate the render (refines P13).
  (s1@0.65 is barely better than s0@0.55 — the dose curve is flattening.)
- ABLATION s3 (floor0.35): re-streaks heavily (worst of the damping family) ⇒ re-confirms the monotone damping
  dose and isolates the floor (low floor → more streaks), supporting P13.
- LAM_OPEN 2→4 (s4): the WORST render of the batch — loops grow LARGER and overshoot green even more. Combined with
  user_input's finding that the loops already overshoot ~5× (R2≈−13), this FALSIFIES "openness can be safely pushed
  up under damping": the loops are too BIG, not too small, so openness pressure is COUNTERPRODUCTIVE for the fit.
- trans0.06 + floor0.55 (s5): the CLEANEST render of the batch — smallest red excursions, most red ellipses sit on
  the green GT. This is the decisive surprise: in batch 4, trans0.06 at floor0.35 (b04.s3) still streaked heavily,
  but trans0.06 at floor0.55 (HIGH damping) is the tamest run yet. ⇒ low-trans and high-damping STACK; lowering
  trans reduces overshoot AMPLITUDE (the R2-killer) only once damping holds the rollout. This partially RESTORES a
  role for TRANS_SCALE — not as a streak knob (P11-revised stands) but as an AMPLITUDE/calibration knob.

Winner: slot 5 (trans006 at floor055) — tamest render, least overshoot, best R2 proxy → new parent/control.
(s2 gfloor075 is the runner-up and the best pure-damping run, but s5's lower amplitude wins on calibration.)

Verdict:
 - Q9b / damping dose — RESOLVED: damping keeps helping monotonically to floor 0.75 (no collapse, fitRMSE still
   falling) but with DIMINISHING returns and is INSUFFICIENT ALONE — every pure-damping render still streaks.
   Refines P13: damping is necessary+monotone, not sufficient. Over-damp turnover NOT reached by 0.75.
 - LAM_OPEN-up hypothesis — FALSIFIED (s4 worst). The loops OVERSHOOT the real motion, so pushing openness UP
   worsens calibration. → new F8; the program goal flips from "open the loops" to "CALIBRATE them DOWN" (matches
   user_input's R2 reframe).
 - trans×damping stacking — SUPPORTED (s5 cleanest). Low trans + high damping stack to the lowest-overshoot render
   yet. TRANS_SCALE re-instated as an AMPLITUDE/calibration lever (distinct from its dead streak role). → new P15.
 - Q3 — unchanged (A_ij +1 in all 6; φ blob structure, causal role still unprobed).
Failures: none — all 6 produced run.log + progress.txt + dashboards + renders, all it1450, zero NaN. Methodological
gap: the honest R2 was not logged (old script); ranking rests on morphology this batch (R2 lands from batch 6).
Next (Batch 6): parent = trans006+floor055 (s5). The reframe drives the plan — CALIBRATE the over-large loops down
to the tiny real motion (push R2>0). Add the user-requested CARDIO_FIT_NORM mechanism (motion-normalised l_fit on
moving nodes → penalises overshoot during training; default off = current behaviour, the ablation/parent). Slots:
FIT_NORM=1 (the new calibration mechanism), TRANS_SCALE 0.06→0.045 (push amplitude down), LAM_OPEN 2→0 (ABLATION —
remove the loop-enlarging pressure that F8 shows is counterproductive) and 2→1 (openness dose), GAMMA_FLOOR 0.55→0.75
(stack max damping with low trans). RANK ON render-R2 (now logged); treat R2≤0 as a failed fit regardless of fitRMSE.
