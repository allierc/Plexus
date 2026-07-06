# Slots designed by the agent each batch.
# Format: name : SPEC specs/<file>.yaml [key val ...]  (dotted keys = op.param overrides)
#
# Batch 89 slots -- PHASE 3 / STAGE BUD, batch 4. TIP-SHARPNESS x OFFSET FRONTIER + tip8 3-SEED LOCK.
#
# WHY THIS BATCH (b88 read):
#   (1) tip5_noanch = a REAL WEAK BUD, 3 seeds: org_bud_score 0.072+/-0.010 (seed0 0.0797, seed1 0.0573,
#       seed2 0.0787; persistence 1.0, neck<0.40, pattern held mi_type_y 1.0, TIER-1 clean) = 7-SD over ctrl
#       -> [established]. The b87 single-seed 0.099 REGRESSED to 0.072 (9th single-seed clean point to fall).
#   (2) [engineering] mpm_grid_update.surface_tension is a NO-OP: st5(5.0)==st3(3.0) byte-identical bud
#       metrics; tip8(8.0)==st5_tip8(5.0) identical. The b88 de-rounding hypothesis was NEVER TESTED (dead
#       lever, reconfirms MOR b73). cell_grow.tip IS live (tip5 0.0797 != tip8 0.0946).
#   (3) tip8 (0.0946, neck 0.178) > tip5 = monotone tip lever, single-seed batch-max -> needs 3-seed lock.
#   (4) ROUNDING is the ceiling: body inflates 6x (area 0.156->0.96) but circularity RISES 0.87->0.96,
#       aspect 1.07 = a bigger SPHERE not a lobe. ps60 (prestretch 0.60) 0.0355 = over-compression rounds.
#       Every roundness lever now exhausted (surface_tension inert, youngs deflates, prestretch amplifies,
#       rate-down worsens, rate-up shatters). A discrete organ likely needs a MULTI-CELL domain (n=1 can't).
#
# HYPOTHESIS (Batch 89): on the anchor-free substrate org_bud_score keeps rising with tip SHARPNESS past tip8
#   and with a MODERATE placement offset, peaking before it overshoots into a non-necked bulge (neck>1):
#   tip12/tip16 and tip8_off05 exceed tip8's 0.0946 while holding neck<1, and the tip8 winner replicates.
#   FALSIFIER: tip12/tip16 <= tip8 (~0.095) AND off05/k2 <0.10 AND tip8 seeds spread <0.06 -> single-cell
#   tip-bud CAPPED ~0.09 -> report the weak reproducible tip-bud (0.072+/-0.010) as the BUD [open] deliverable
#   and OPEN the multi-cell-domain path (grow a SUBSET of cells) as the only route to a discrete organ.
#   Runaway arm: any slot neck_ratio>1 (bulge) OR nn_min<0.016 OR collapsed>0 -> overshoot/rupture, retreat.
#
# READ order: FIRST tip8 3-seed spread (s1/s2 vs b88 s3=seed0 0.0946); THEN sharpness trend (tip8->12->16:
#   bud_score, neck<1); THEN offset (off05, tip12_off05); screen EVERY slot for neck_ratio>1 / nn_min<0.016.
#   Judge bud by score/neck/persistence (NOT overlap=0, broken), pattern by mi_type_y, growth by organo area.
#   All 12000f (~800-830 s on L4) < 20-min wall.
#
# 4 exploit (tip8_s1, tip8_s2, tip12, tip8_off05) / 3 explore (tip16, tip8_k2, tip12_off05) / 1 control.

tip8_s1     : SPEC specs/embryo_BUD_noanch_tip8_s1.yaml
tip8_s2     : SPEC specs/embryo_BUD_noanch_tip8_s2.yaml
tip12       : SPEC specs/embryo_BUD_noanch_tip12.yaml
tip8_off05  : SPEC specs/embryo_BUD_noanch_tip8_off05.yaml
tip16       : SPEC specs/embryo_BUD_noanch_tip16.yaml
tip8_k2     : SPEC specs/embryo_BUD_noanch_tip8_k2.yaml
tip12_off05 : SPEC specs/embryo_BUD_noanch_tip12_off05.yaml
ctrl_nogrow : SPEC specs/embryo_BUD_noanch_nogrow.yaml
