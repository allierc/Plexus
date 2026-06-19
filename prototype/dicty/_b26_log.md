# Batch 26 log — per-sweep entries

## Batch 26 Sweep 0 — seed [0..15]  [inconclusive — within noise]
Hypothesis (H1-B26): seed sweep at B26 parent (spring.r_on=0.19; density_repel.strength=0). Validates new parent reproducibility under morph_score; expects sigma_loss<=0.04.
Response: loss range [0.9261, 0.9960] (Δ=0.07, σ≈0.024); inner_mass [0.22, 0.42]; nm range [5, 12]; morph_score range [0.0, 0.50]. seed=11 best loss 0.9261 (inner=0.421); seed=4 best morph 0.0 (nm=8); seed=12 morph 0.0001 (nm=8).
Morphology (from strip): all 16 seeds produce 5-12 mounds; visually multi-mound everywhere. seeds 4, 9, 12 visibly closest to REAL 8-mound layout.
Verdict: SUPPORTED — parent r_on=0.19 is reproducible; sigma in [Est #73] floor; morph_score saturates (multiple seeds reach morph≤0.0001).
Knowledge update: r_on=0.19 parent confirmed multi-mound robustly across seeds; sigma_loss ~ 0.02-0.03 tightest yet.

## Batch 26 Sweep 1 — spring.r_on [0.175..0.260]  [supported — peak]
Hypothesis (H2-B26): r_on FINE around 0.19; tests whether r_on=0.19 is a true peak or plateau edge.
Response: loss U-shape; best loss r_on=0.20 = 0.9126 (PROJECT-BEST TIE), worst r_on=0.26 = 1.024; inner_mass monotone-UP 0.27→0.55 toward over-compaction at high r_on. Best morph r_on=0.195 morph=0.0001 nm=8 loss=0.9373.
Morphology (from strip): r_on=0.175-0.188 sparse-multi (small mounds); r_on=0.19-0.21 dense multi-mound (5-9 mounds, visible morph match); r_on>=0.23 mound-COUNT drops to 4-6 and per-mound mass spikes (over-compaction). REAL-like at r_on=0.195-0.20.
Verdict: SUPPORTED — r_on=0.195-0.20 is a PLATEAU peak for both loss and morph. Est #91 RECLASSIFIED from "r_on=0.19 single morph winner" to "r_on=0.195-0.20 morph PLATEAU"; r_on=0.20 is the best-loss parent.
Knowledge update: project-best loss UNCHANGED at 0.9126 (sw 1, r_on=0.20). Adopt r_on=0.20 as B27 parent.

## Batch 26 Sweep 2 — spring.kadh [3..400]  [supported — joint transitivity, productive interior]
Hypothesis (H3-B26): r_on=0.19 × kadh joint transitivity. Tests whether Est #91 (r_on) + Est #59 (kadh=20 morph winner) compose without B15-style joint collapse.
Response: loss noisy [0.92, 1.14]; best loss kadh=45 = 0.9456 (nm=8 morph=0.0029); rises monotonically at kadh>=150 (loss>=1.10, nm drops to 2-3). Best morph kadh=45 morph=0.0029.
Morphology (from strip): kadh=3-80 = 5-7 distinct dense mounds; kadh>=150 = consolidates to 2-3 large dense blobs (over-adhesion compaction). Multi-mound preserved over a 50x range.
Verdict: SUPPORTED — joint transitivity HOLDS. Joint corner (r_on=0.19, kadh=45) reaches 8-mound morph; Est #43 joint-collapse concern dissolved under morph_score.
Knowledge update: r_on x kadh joint = transitive on the 8-mound manifold; kadh productive band [3, 80] at r_on=0.19.

## Batch 26 Sweep 3 — sense_sat.c_sat [0.2..1.6]  [supported — flat, multi-mound everywhere]
Hypothesis (H4-B26): r_on=0.19 × c_sat ridge transitivity; tests c_sat=0.8/1.2 B25 morph winners at new r_on.
Response: loss largely flat [0.93, 1.04]; anomalous spike at c_sat=1.2 (1.04, nm=13); best loss c_sat=0.8 = 0.9343 (nm=7). Best morph c_sat=0.9 morph=0.0 nm=8 loss=0.9454.
Morphology (from strip): every c_sat in [0.2, 1.6] produces 5-13 dense multi-mound; the (c_sat, sat_n) ridge (Est #49/#57) extends cleanly to the new r_on parent. Visible 8-mound at c_sat=0.6, 0.9.
Verdict: SUPPORTED — c_sat ridge transitive; multi-mound robust across two orders of magnitude in c_sat at r_on=0.19.
Knowledge update: ridge plateau confirmed at r_on=0.19; c_sat=0.5 parent value remains canonical (no improvement).

## Batch 26 Sweep 4 — density_repel.strength [0..2.0]  [supported — productive interior at parent]
Hypothesis (H5-B26): density_repel FINE at r_on=0.19; resolves the B25 sw 1 interior morph peak at new parent.
Response: loss flat-noisy [0.92, 0.99]; best loss strength=0.14 = 0.9253; best morph strength=0.1 morph=0.0 nm=8 loss=0.9491. Slight monotone-down to strength~0.4, flat above.
Morphology (from strip): strength=0 baseline 5 mounds; strength=0.02-0.20 → 6-11 distinct mounds (nm=8 at strength=0.02/0.1/0.2); strength>=0.4 = 7-11 mounds, dense; strength=2.0 still multi-mound. No catastrophe in [0, 2.0].
Verdict: SUPPORTED — density_repel productive at parent under morph_score; band [0.02, 0.4] yields nm=8 at multiple values; not a loss improvement but a clear morph contribution.
Knowledge update: density_repel productive band TRANSFERS to r_on=0.19 parent.

## Batch 26 Sweep 5 — cell.n [400..6500]  [supported — U-shape; collapse persists at high n]
Hypothesis (H6-B26): cell.n WIDE at density_repel.strength=2.0 (Est #89 rescue). Tests whether density_repel halts high-n single-blob collapse.
Response: loss U-shape; best n=2700 loss=0.9267 (nm=7, inner=0.367); rises at both ends (n=400 loss=1.10; n=6500 loss=1.09). Best morph n=2200 morph=0.0002 nm=8 loss=0.9389.
Morphology (from strip): n=400 sparse multi-mound; n=800-3000 dense multi-mound (5-12); n>=4000 collapses to **2 distinct mounds** (over-compaction); n=6000-6500 = 2 dense mounds — density_repel.strength=2 did NOT prevent the high-n collapse. Est #89 rescue PARTIALLY RETRACTED: strength=2 was sufficient at n=3000 in B25 sw 5 but not at n>=4000 here.
Verdict: PARTIALLY FALSIFIED — density_repel strength=2 rescues mid-range cell.n (up to ~3000) but the high-n single-blob collapse REAPPEARS past n=4000 even with the rescue operator on. Est #89 narrowed to "rescues n<=3000 only".
Knowledge update: high-n single-attractor regime is more robust than B25 evidence suggested; density_repel scaling to cell.n is sub-linear.

## Batch 26 Sweep 6 — density_repel.strength × n_frames=1200, thr=0.5  [falsified — collapse uninterrupted]
Hypothesis (H7-B26): Est #90 RE-TEST with thr=0.5 (always-on density spacer). If lower thr halts runaway compaction, op survives as finite-volume mechanism.
Response: BIMODAL; strength<=6 → single-blob collapse (loss 1.18-1.25, nm=1, inner=0.6-0.75); strength>=12 → explosive uniform dispersion (loss 15-20, nm=60+, inner=0.17). No productive interior anywhere in [0, 300].
Morphology (from strip): strength=0..6 = SINGLE TINY POINT (runaway compaction completes); strength=12..300 = uniform sand (cells flooded by repulsion). No regime preserves multi-mound at n_frames=1200.
Verdict: FALSIFIED — Est #90 reconfirmed. density_repel.thr=0.5 does NOT halt the runaway compaction; with high thr the collapse completes uninterrupted; with low thr the operator becomes destructive before it can stabilize. The finite-volume mechanism IN THIS FORM does not solve Est #82.
Knowledge update: density_repel does not yield a stable multi-mound attractor under long integration regardless of thr. The missing mechanism is NOT short-range cell-cell repulsion alone.

## Batch 26 Sweep 7 — n_frames [200..1600] at r_on=0.19  [falsified for "r_on rescues compaction"]
Hypothesis (H8-B26): Est #82 RE-TEST at r_on=0.19 (sparser adhesion reach). Does runaway compaction persist?
Response: monotone-UP loss (0.94 at n_frames=280 → 1.29 at n_frames=1600); inner_mass monotone-UP (0.27→0.84); nm collapses 11→1 by n_frames=560+. Best loss n_frames=280 = 0.9397 (nm=8).
Morphology (from strip): n_frames=200-360 multi-mound (5-11 mounds); n_frames=400-480 transition (2-5 mounds); n_frames>=560 SINGLE BLOB. Same collapse trajectory as B24 sw 6/14.
Verdict: FALSIFIED — r_on=0.19 does NOT resist runaway compaction. Est #82 (no stable multi-mound attractor) holds across r_on in [0.19, 0.245] and across operator additions.
Knowledge update: runaway compaction is invariant to r_on within the productive range; structural, not parameter-tunable.

## Batch 26 Sweep 8 — secrete_het.het_std [0..3.0]  [INVALID — SILENT OPERATOR]
Hypothesis (H9-B26): re-evaluate secrete_het under morph_score (Est #63 FALSIFIED B19 by SSIM).
Response: BIT-IDENTICAL across all 16 values (loss=0.98, inner=0.293, nm=5, morph=0.376). The varied parameter had ZERO effect.
Morphology (from strip): IDENTICAL strip; same image 16 times.
Verdict: INVALID — `secrete_het` operator instance is NOT in `specs/dicty_loop_base.yaml` operators/schedule; only kept "in code as ablation". `eval_sweeps.set_param("secrete_het.het_std", ...)` searches `sc.operators` for op.op=="secrete_het", finds none, returns silently. Sweep produced 16 identical parent runs.
Knowledge update: METHODOLOGICAL — to re-evaluate a code-only ablation operator, schedule it with parameter=0 default. PRONG (β) FAILS for secrete_het this batch; re-run in B27 after spec fix.

## Batch 26 Sweep 9 — decay_dens.dens_coeff [0..4.0]  [INVALID — SILENT OPERATOR]
Hypothesis (H10-B26): re-evaluate decay_dens (Est #68) under morph_score.
Response: BIT-IDENTICAL (loss=0.98, inner=0.293, nm=5, morph=0.376).
Morphology: IDENTICAL strip × 16.
Verdict: INVALID — same cause as sw 8; `decay_dens` not scheduled. PRONG (β) FAILS for decay_dens.
Knowledge update: same as sw 8.

## Batch 26 Sweep 10 — pulse_dens.amplitude [0..20]  [INVALID — SILENT OPERATOR]
Hypothesis (H11-B26): re-evaluate pulse_dens (Est #80) under morph_score.
Response: BIT-IDENTICAL (loss=0.98, inner=0.293, nm=5, morph=0.376).
Morphology: IDENTICAL strip × 16.
Verdict: INVALID — same cause as sw 8. PRONG (β) FAILS for pulse_dens.
Knowledge update: same as sw 8.

## Batch 26 Sweep 11 — diff_dens.kappa [0..5.0]  [INVALID — SILENT OPERATOR]
Hypothesis (H12-B26): re-evaluate diff_dens (Est #78) under morph_score.
Response: BIT-IDENTICAL (loss=0.98, inner=0.293, nm=5, morph=0.376).
Morphology: IDENTICAL strip × 16.
Verdict: INVALID — same cause as sw 8. PRONG (β) FAILS for diff_dens.
Knowledge update: same as sw 8. NONE of the four B19-B23 falsified mechanisms were validly re-evaluated under morph_score in B26.

## Batch 26 Sweep 12 — sense_sat.gain × c_sat=0.30 [200..9000]  [supported — densification axis transferred]
Hypothesis (H13-B26): sense_sat.gain × c_sat=0.30 at r_on=0.19; refines B25 sw 11 winner at new parent.
Response: loss flat-noisy [0.93, 1.18] with spike at gain=7500 (1.18); best loss gain=1500 = 0.9277; best morph gain=1100 morph=0.0002 nm=8 loss=0.9412.
Morphology (from strip): multi-mound (5-15) across the range; nm spikes to 15 at gain=7500 (over-attraction fragmentation); otherwise stable. Est #53 densification axis transfers cleanly to r_on=0.19.
Verdict: SUPPORTED — densification axis transitive; gain plateau around 1100-1800.
Knowledge update: B25 sw 11 winner (gain=2200) and B27 candidate parent gain=1500 reach 8-mound at r_on=0.19.

## Batch 26 Sweep 13 — relay.gain [60..500]  [supported — plateau reconfirmed]
Hypothesis (H14-B26): relay.gain FINE under morph_score at r_on=0.19. Test for hidden morph peak in flat plateau.
Response: catastrophe at gain<=80 (loss 1.61 at 60, 1.17 at 80; single-blob ringing failure); plateau [0.95, 1.08] for gain>=100; best gain=100 loss=0.9505 (nm=6, morph=0.25); best morph gain=180 morph=0.125 nm=9 loss=0.9557.
Morphology (from strip): gain=60-80 single-blob (relay ringing); gain=100-500 dense multi-mound (5-12 mounds). No hidden morph peak — plateau in both metrics.
Verdict: SUPPORTED — Est #67 / #74 relay.gain plateau holds. No morph_score-revealed productive direction.
Knowledge update: relay.gain=140 parent unchanged; plateau flat under both metrics.

## Batch 26 Sweep 14 — sense_sat.gain × kadh=20 [200..8000]  [supported — joint morph winner]
Hypothesis (H15-B26): joint of two B25 morph winners (kadh=20 + gain densification) at r_on=0.19.
Response: U-shape; best loss gain=750 = 0.9316 (nm=8, morph=0.0001); best morph gain=2500 morph=0.0 nm=8 loss=0.9403. Catastrophe at gain<=300 (loss 1.12 at 200) and gain=8000 (loss 1.11).
Morphology (from strip): gain=200-300 sparse; gain=400-5500 = 5-14 dense multi-mound (nm=8 reached at gain={750, 1100, 2500}); gain=8000 over-attraction to 14 small spots. Cleanest 8-mound corner of B26: (r_on=0.19, kadh=20, gain=750-2500).
Verdict: SUPPORTED — joint corner reaches 8-mound morph_score≈0 at multiple gain values; Est #43 joint-collapse concern dissolved here.
Knowledge update: (r_on=0.19, kadh=20, gain=750-2500) is the cleanest 8-mound corner of B26; candidate B27 sub-parent for ridge-mapping.

## Batch 26 Sweep 15 — camp.diffusion [0.0001..0.07]  [supported — Est #60/#65 wall reconfirmed]
Hypothesis (H16-B26): morph_score under camp.D at r_on=0.19; test if D wall is metric artefact like Est #58.
Response: loss flat [0.93, 1.03] for D in [0.0001, 0.035]; CATASTROPHE at D>=0.045 (loss 1.50 at D=0.045, 1.33 at 0.055, 1.64 at 0.07). Best loss D=0.001 = 0.9303; best morph D=0.008 morph=0.0 nm=8 loss=0.9549.
Morphology (from strip): D<=0.035 = 5-12 multi-mound; D>=0.045 = nm spikes to 17-20 (over-diffusion fragments cells into tiny scattered spots — loss collapses because cells DO fragment but per-spot density wrong); morph_score actually rises (1.13-1.50 at high D).
Verdict: SUPPORTED — camp.D wall is GENUINE (not a metric artefact); mound count past the wall is "wrong-fragmentation" (too many tiny clumps), not productive. Est #60/#65 hold under morph_score.
Knowledge update: camp.D working band [0.0001, 0.035] confirmed under both metrics.

## Batch 26 — summary

- **PROJECT-BEST LOSS UNCHANGED at 0.9126** (sw 1 r_on=0.20). Multiple ties at <=0.94 across sw 0, 1, 4, 5, 14.
- **PRONG (α) 8-MOUND MANIFOLD TRANSITIVITY — STRONGLY SUPPORTED.** Every joint sweep reaches morph_score≈0 nm=8 (sw 2 kadh=45, sw 3 c_sat=0.9, sw 5 cell.n=2200, sw 14 gain=2500 at kadh=20). The 8-mound manifold is broad and traversable; the B25 Est #43 joint-collapse concern is dissolved under morph_score.
- **PRONG (β) RE-EVALUATION INVALID — SPEC BUG.** 4 of 16 sweeps (sw 8/9/10/11) ran with SILENT operators because secrete_het/decay_dens/pulse_dens/diff_dens were not in `specs/dicty_loop_base.yaml`'s operators/schedule. All 16 values per sweep produced bit-identical results. The four B19-B23 falsifications stand untested under morph_score; re-run in B27 after spec fix.
- **PRONG (γ) RUNAWAY COMPACTION RECONFIRMED.** sw 7 (n_frames sweep at r_on=0.19) reproduces Est #82 collapse; sw 6 (density_repel × n_frames=1200, thr=0.5) confirms Est #90: density_repel does NOT halt compaction at low thr either — the operator either ignores the runaway (low strength) or destroys multi-mound entirely (high strength).
- **EST #89 PARTIALLY RETRACTED.** sw 5 (cell.n WIDE at density_repel.strength=2.0) shows the high-n single-blob REAPPEARS past n>=4000 even with the rescue operator on; the B25 rescue narrows to n<=3000.
- **EST #91 RECLASSIFIED.** r_on=0.19 morph "winner" is actually a plateau across r_on=[0.195, 0.20]; r_on=0.20 reclaims B27 parent on best-loss grounds.
- **B27 STRATEGIC FRAME:**
  - (i) FIX SPEC — add secrete_het/decay_dens/pulse_dens/diff_dens to the operators+schedule with ablation defaults so prong-β can finally be measured.
  - (ii) PROPER prong-β re-evaluation: 4 of the 16 slots for these mechanisms under morph_score.
  - (iii) Attack Est #82 from a NEW angle (cell-cell repulsion alone failed): try a per-mound MASS CAP (gate adhesion off above ρ threshold) or chemotactic GRADIENT SATURATION (sense effectively zero when local cAMP exceeds c_sat × N) as a new operator. Sense_sat already does the latter at the cell level; the engine-level analog would be field-side.
  - (iv) Adopt B27 parent r_on=0.20 (loss-winner) and add a sub-corner (r_on=0.19, kadh=20, gain=2500) sweep set for the cleanest 8-mound manifold.

> Methodological log: this is the SECOND batch (after B21 diff_dens silent-op bug, Est #72) where a silent-operator implementation defect invalidated multiple sweeps. Pattern: when re-introducing a previously-falsified operator for re-evaluation, ALWAYS confirm it is in `operators:` and `schedule:` before launching.
