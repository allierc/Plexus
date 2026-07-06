# Slots designed by the agent each batch.
# Format: name : SPEC specs/<file>.yaml [key val ...]  (dotted keys = op.param overrides)
#
# Batch 88 slots -- PHASE 3 / STAGE BUD, batch 3. NOANCH SURFACE-TENSION SWEEP + WINNER REPLICATION.
#
# WHY THIS BATCH: b87's REAL tip-sharpness sweep ran (separate specs, all differed). Findings:
#   (1) ANCHORED tip ladder is MONOTONE but tiny -- org_bud_score 0.0144(tip1.5)->0.0275(tip5)->0.0405(tip10),
#       bud_len_bodyR 0.210->0.264->0.376; b87 falsifier ("FLAT") did NOT fire, but bud_score stays <0.05
#       anchored = a WEAK bud, failure mode = ROUNDING (shape circularity 0.90), NOT rupture/pattern loss.
#   (2) DROPPING mpm_anchor is the DOMINANT lever: tip5_noanch (b87 s3) -> bud_score 0.0994 (7x ctrl, MOST
#       necked neck 0.225, persistence 1.0, biggest body shape-area 0.961) -- the anchor resists BOTH tip
#       extension AND body inflation. n=1.
#   (3) CEILING = the aggressive combo OVERSHOOTS: tip10_noanch_off06 (s5) -> bud_score 0.0, neck 1.469 (>1 =
#       a BULGE not a necked bud). Sweet spot = anchor-off + MODERATE tip.
#   Pattern HELD everywhere (mi_type_y 1.0, seg_index 1.0); TIER-1 clean everywhere (collapsed 0, nn_min
#   0.0177-0.0186); NO runaway. [engineering] org_growth_bud_overlap = 0.0 in ALL 8 slots = BROKEN metric.
#
# HYPOTHESIS (Batch 88): on the anchor-free substrate, LOWERING MPM surface_tension (8->5->3) is the
#   discreteness lever -- surface tension rounds the nascent bud back into the sphere; lowering it lets the
#   tip-localized growth STAND OFF and NECK. Predict org_bud_score UP + organo circularity DOWN / aspect_ratio
#   UP as surface_tension falls, while the tip5_noanch winner (bud_score 0.099) replicates across 3 seeds.
#   FALSIFIER: bud_score FLAT/FALLING vs surface_tension (rounding not the limiter) OR winner fails to
#   replicate (s1/s2 bud_score <0.05, seed luck) -> accept the weak monotone-tip bud as BUD [open] deliverable
#   and PIVOT to pattern-gated growth. RUNAWAY ARM: st3 fragments (fragment_count>1 sustained OR nn_min<0.016
#   OR collapsed>0) -> surface tension floor found, retreat to st5.
#
# READ ORDER (organo family from scorecard.json / org_* in metrics.json, NOT the movie): FIRST the WINNER
#   REPLICATION -- noanch_s1/s2 org_bud_score vs the b87 s3 0.099 (compute mean+/-SD over the 3 seeds; <0.05 =
#   seed luck). THEN the SURFACE-TENSION TREND at seed 0: st8 (b87 s3 = anchor) -> st5 -> st3, read org_bud_
#   score, organo circularity/aspect_ratio (stand-off), org_bud_neck_ratio (<1 = necked, >1 = bulge),
#   org_bud_persistence. SCREEN st3 FIRST for the fragment signature (org_fragment_count>1 sustained, nn_min<
#   0.016, collapsed>0). Judge bud by score/len/neck/persistence (NOT overlap=0), growth by organo area/
#   body_radius (NOT grow_ratio), pattern by mi_type_y (must stay 1.0). escape~1.0 = body-drift artifact ALONE.
#
# GOTCHAs (durable): dotted cell_grow.*/mpm_grid_update.* overrides DON'T apply to flow-style op lines ->
#   SEPARATE SPECS authored (this batch); general.seed override also risky on the flow map -> seed variants
#   are separate spec files (noanch_s1 seed 1, noanch_s2 seed 2). all `at:'agent[type=x]'` single-quoted;
#   cp/sed/>>/heredoc/python3/loop-for sandbox-blocked -> Write/Edit + Read; nn_min ~0.018 = clean floor.
#   12000f (~800-830 s) < 20-min L4.
#
# Roles: 4 exploit (noanch_s1 + noanch_s2 = 3-seed the winner / noanch_st5 = de-round / noanch_tip8 = sharper
#   tip) / 3 explore (noanch_st3 aggressive de-round+runaway probe / noanch_ps60 stronger local pressure /
#   noanch_st5_tip8 best-guess combo) / 1 control (ctrl_nogrow = noanch rate-0 no-op, pattern/shape baseline).

noanch_s1      : SPEC specs/embryo_BUD_noanch_s1.yaml
noanch_s2      : SPEC specs/embryo_BUD_noanch_s2.yaml
noanch_st5     : SPEC specs/embryo_BUD_noanch_st5.yaml
noanch_tip8    : SPEC specs/embryo_BUD_noanch_tip8.yaml
noanch_st3     : SPEC specs/embryo_BUD_noanch_st3.yaml
noanch_ps60    : SPEC specs/embryo_BUD_noanch_ps60.yaml
noanch_st5_tip8: SPEC specs/embryo_BUD_noanch_st5_tip8.yaml
ctrl_nogrow    : SPEC specs/embryo_BUD_noanch_nogrow.yaml
