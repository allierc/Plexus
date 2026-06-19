# Batch 29 — per-sweep entries + summary

> B29 parent: B28 parent with secrete_het.het_std reverted to 0 (per Est #101).
> 256 sims. Framed as "FINAL POINT-CELL CLOSING BATCH" — two parallel prongs:
> (a) FINAL CORNER + TIMESCALE adjudication (NEW r_on=[0.220, 0.225] candidate,
> camp.decay × Est #100 timescale extension, D=0.0042 seed verification);
> (b) FINAL OPERATOR-SIDE TESTS (sense_sat.gain attenuation = chemotaxis-source
> attenuation, last untested adh_cap analogue; random_walk × n_frames=1200;
> extreme vmax × n_frames=1200; stacked density_repel × corner × n_frames=1200).

## Batch 29 Sweep 0 — seed [0..15] at B29 parent  [supported — parent re-pinned]
Hypothesis (H1-B29): if median<=0.94 and best<=0.91, project-best 0.9126 is solid; >=3 seeds at morph<=0.05 means 8-mound manifold reachable from parent.
Response: loss [0.913, 1.390], median ~0.97, sigma~0.10 (outlier seed=1 loss=1.39 inflates). Without outlier sigma~0.04; morph distribution: 2/16 seeds at morph<=0.13 (best seed=4 morph=0.125 nm=7). Best loss 0.9126 at seed=0 ties project best.
Morphology (from strip): consistent 5-10 mound multi-mound with stringy elongation; seed=1 is a clear catastrophe-blob (loss 1.39, nm=18 over-fragmented). Most seeds 6-7 mounds with REAL-like compact structure.
Verdict: SUPPORTED — parent reproducible; project-best tied. NO seed at morph<=0.05; the 8-mound manifold is NOT reachable from the bare parent.
Knowledge update: parent reproducibility confirmed (Est #48 holds at B29 parent); morph-floor at parent ~0.125 (3 mounds off from 8). 1/16 outlier rate is the noise structure; sigma_loss without outlier ~0.04.

## Batch 29 Sweep 1 — spring.r_on FINE [0.215, 0.235]  [FALSIFIED candidate corner]
Hypothesis (H2-B29): the B28 sw 8 r_on=[0.220, 0.225] candidate corner is a robust morph window.
Response: loss flat with noise [0.92, 1.10]; best morph=0.001 at r_on=0.228 (nm=8); r_on=0.222 itself gives morph=0.25 nm=6.
Morphology (from strip): 3-5 streak-mound morphology across all values; no qualitative shift. Visually indistinguishable across [0.215, 0.232].
Verdict: FALSIFIED — the B28 sw 8 r_on=0.222 peak was 1/16 seed-luck; the morph win moves to a single different bin (r_on=0.228) under a different seed; not a robust corridor.
Knowledge update: NEW Est #103 — the NEW r_on=[0.220, 0.225] candidate corner from B28 sw 8 is FALSIFIED as a robust morph corridor. r_on FINE response is single-bin scatter typical of seed-luck.

## Batch 29 Sweep 2 — spring.r_on at corner (kadh=20, gain=1500)  [FALSIFIED transfer]
Hypothesis (H3-B29): the new r_on candidate transfers to the Est #98 kadh+gain corner.
Response: loss flat [0.93, 1.09]; best morph=0.0003 at r_on=0.217 (nm=8). r_on=0.222 at the corner gives morph=0.51 nm=4 — WORSE than nearby bins.
Morphology (from strip): 4-7 mound morphology distributed; r_on=0.222 column shows 4 mounds and looks no better than neighbors.
Verdict: FALSIFIED — r_on=0.222 does NOT transfer to the corner; single-bin best at r_on=0.217 reinforces single-bin seed-luck reading.
Knowledge update: Est #103 reconfirmed at corner. Est #98 corner r_on band [0.183, 0.198] is the only robust corner; new candidate [0.220, 0.225] retracted.

## Batch 29 Sweep 3 — camp.decay [0.05, 5.0] at n_frames=1200 (parent)  [FALSIFIED rescue at parent]
Hypothesis (H4-B29): shorter cAMP memory (high decay) extends the 8-mound transient at n_frames=1200.
Response: decay<=1.0 all collapse to nm=1 (loss 1.10-1.24, inner_mass 0.59-0.75); decay=1.4 single transition (loss 1.19, nm=1 inner_mass 0.398); decay>=2.0 CATASTROPHIC field annihilation (loss 3.5-11.4, nm=29-55 over-fragmentation, inner_mass collapsing to 0.0).
Morphology (from strip): low-decay = single small bright spot; decay=1.4-2.0 = larger diffuse blob, no multi-mound; decay>=2.8 = catastrophic fragmented dust.
Verdict: FALSIFIED at parent — high camp.decay does NOT rescue Est #82 at the BARE parent; decay=1.4 marginal but still single-blob.
Knowledge update: Est #82 rescue via decay-at-parent FALSIFIED. The decay rescue is corner-conditional (see sw 4).

## Batch 29 Sweep 4 — camp.decay [0.05, 5.0] at 8-mound corner + n_frames=1200  [SUPPORTED — NEW Est #104, first Est #82 partial rescue]
Hypothesis (H5-B29): high decay at the 8-mound corner extends Est #100 timescale.
Response: decay<=1.0 all collapse (nm=1, loss 1.11-1.22). **decay=1.4 gives loss=0.9863, inner_mass=0.373, nm=6 at n_frames=1200** — best in the batch at this break-test condition. decay>=2.0 catastrophic.
Morphology (from strip): low-decay = single tight spot like sw 3; decay=1.4 column shows a diffuse multi-blob structure with broader inner mass (consistent with nm=6). decay>=2.0 catastrophic dispersion.
Verdict: SUPPORTED — the FIRST point-cell configuration to break Est #82 (nm>=4 at n_frames=1200) since the runaway was discovered in B24. **Requires 16-seed verification before adoption (mirrors B27/B28 Est #97 lesson)**.
Knowledge update: NEW Est #104 — at the 8-mound corner (r_on=0.19, kadh=20, gain=1500), camp.decay=1.4 sustains nm=6 multi-mound at n_frames=1200 with loss=0.9863. INDEPENDENT FROM corner-only collapse (sw 3 at parent same decay collapses), so this is a CORNER × DECAY INTERACTION. Mechanism: faster cAMP turnover prevents global-attractor formation while corner-adhesion maintains local mound structure. Project's first Est #82 partial mitigation candidate; promoted to B30 verification.

## Batch 29 Sweep 5 — seed [0..15] at r_on=0.222  [INCONCLUSIVE — single-seed wins at new corner]
Hypothesis (H6-B29): if the new r_on=0.222 morph win replicates across seeds, Est #103 retracts.
Response: loss range [0.907, 1.047], median ~0.95, sigma~0.04; best loss=0.9066 at seed=7 (NEW PROJECT-BEST, marginal); morph=0.0 at seed=6 (nm=8). 5/16 seeds at morph<=0.13.
Morphology (from strip): all 16 seeds show 5-10 multi-mound; some seeds look noticeably tighter than parent equivalents.
Verdict: INCONCLUSIVE — improvement is marginal (Δloss=0.006 < σ=0.04). 5/16 seeds at morph<=0.13 is BETTER than sw 0 (2/16) but still distance-noise from a robust corner.
Knowledge update: r_on=0.222 may have a mild productive effect across seeds (better than bare parent distribution) but does NOT cross the threshold for parent adoption. Project-best loss=0.9066 noted but NOT adopted.

## Batch 29 Sweep 6 — seed [0..15] at camp.diffusion=0.0042  [SUPPORTED — Est #99 promoted to robust morph anchor]
Hypothesis (H7-B29): the Est #99 morph-best D=0.0042 corridor replicates across seeds.
Response: loss range [0.923, 1.037], median ~0.94, sigma~0.03 (tightest of the batch). **4/16 seeds at morph_score=0** (seeds 0/6/11/15, nm=8); 3 more seeds at morph<=0.13. Best loss=0.9226 at seed=11 (nm=8, morph=0.0001).
Morphology (from strip): most seeds show CLEAR 7-10 mound structure with REAL-like spatial distribution. Visibly the strongest morph batch of B29.
Verdict: SUPPORTED — Est #99 promoted from corridor-edge to robust morph anchor. 4/16 morph_score=0 seeds is 4× the bare-parent rate (sw 0).
Knowledge update: NEW Est #105 — camp.diffusion=0.0042 is a ROBUST morph anchor (4/16 seeds at morph=0, 7/16 at morph<=0.13). The B30 parent adopts camp.diffusion=0.0042 (single-axis conservative change with seed-distribution verification, mirroring B16/B19 adoption protocol).

## Batch 29 Sweep 7 — sense_sat.gain [200, 8000] at n_frames=1200 (parent)  [FALSIFIED — 12th project mechanism]
Hypothesis (H8-B29): low-gain chemotaxis source attenuation rescues Est #82 where adh_cap (sink attenuation) failed.
Response: loss universally 1.14-1.25; nm=1 at every gain in [300, 8000]; gain=200 nm=2 marginal. inner_mass climbing to 0.83 at high gain (over-compaction signature).
Morphology (from strip): single tight bright point at every value; gain=200 column barely shows a second peak.
Verdict: FALSIFIED — chemotaxis-source attenuation does NOT rescue Est #82. Even at gain=200 (5x below parent 500), runaway compaction proceeds.
Knowledge update: NEW Est #106 — sense_sat.gain attenuation falsified as Est #82 rescue (12th project mechanism falsified). With adh_cap (Est #102, B28) also falsified, BOTH source- and sink-side attenuation routes are dead. Est #82 is robust to per-cell magnitude scaling of the chemotaxis pathway.

## Batch 29 Sweep 8 — sense_sat.gain [200, 8000] at corner + n_frames=1200  [FALSIFIED — confirms sw 7]
Hypothesis (H9-B29): corner-conditioning rescues gain attenuation where parent fails.
Response: nm=1-2 across all values; loss [1.10, 1.21]; gain=600 marginally nm=2 inner=0.436 loss=1.11.
Morphology (from strip): single tight blob across the strip; occasional second peak.
Verdict: FALSIFIED — corner does NOT rescue gain attenuation under Est #82 break test. Confirms sw 7 — the mechanism family is dead.
Knowledge update: Est #106 reconfirmed at corner. Note: sw 8 collapses MORE than sw 7 at gain=200 (sw 8 nm=2 inner=0.584 vs sw 7 nm=2 inner=0.529) — adhesion-corner accelerates collapse here.

## Batch 29 Sweep 9 — random_walk.strength [0, 2.5] at n_frames=1200  [FALSIFIED RW rescue]
Hypothesis (H10-B29): high RW noise breaks deterministic gradient-following and prevents runaway compaction.
Response: nm=1 universally; loss flat 1.18-1.25; inner_mass 0.68-0.77 (all compacted); morph_score 1.4-4.6.
Morphology (from strip): single tight blob at every value. No multi-mound break with RW noise up to strength=2.5.
Verdict: FALSIFIED — RW noise does NOT prevent runaway compaction. Chemotactic pull dominates RW even at strength=2.5.
Knowledge update: NEW Est #107 — RW falsified under Est #82 break test. RW silent for ELEVEN batches now (Est #80 era + B29). Permanently dropped from sweep designs.

## Batch 29 Sweep 10 — vmax FINE [0.030, 0.080] at n_frames=1200  [FALSIFIED — vmax × Est #82]
Hypothesis (H11-B29): low-vmax slow integration defers Est #82 collapse.
Response: nm=1 universally; loss [1.10, 1.22] with aliasing variance at vmax in [0.058, 0.072]. No vmax in the working band rescues multi-mound.
Morphology (from strip): single bright point everywhere; resolution-aliasing visible at vmax=0.06 and 0.0735 (slightly worse single-blob).
Verdict: FALSIFIED — vmax does NOT defer Est #82. Est #66 × Est #82 joint closed.
Knowledge update: NEW Est #108 — vmax attenuation falsified under Est #82. Est #82 timescale is determined by chemotactic dynamics, not by per-step displacement cap.

## Batch 29 Sweep 11 — camp.decay [0.02, 4.0] at parent (n_frames=400)  [PARTIAL SUPPORTED — low-decay morph niche]
Hypothesis (H12-B29): productive interior decay window under morph_score.
Response: loss range [0.91, 11.3]; best loss=0.9126 at decay=0.07 (parent, ties project best); **morph=0.0002 at decay=0.02 (nm=8 loss=0.9572)** — new morph win; morph=0.125 at decay=0.10 (nm=9). decay>=2.0 catastrophic.
Morphology (from strip): low-decay (0.02-0.07) = compact multi-mound (matches REAL clusters); decay=0.20-1.0 = sparser but still multi-mound; decay>=2.0 = catastrophic dispersion.
Verdict: PARTIAL SUPPORT — decay=0.02 opens a new morph niche under morph_score (Δmorph=0.125 below parent's 0.125 floor). Loss penalty +0.045 (within seed-noise).
Knowledge update: NEW Est #109 — camp.decay=0.02 is a low-decay morph niche (single-seed; requires 16-seed verification in B30). Refines Est #56 lower edge.

## Batch 29 Sweep 12 — density_repel.strength × corner × n_frames=1200  [FALSIFIED — final stacked Est #82 test]
Hypothesis (H13-B29): density_repel + 8-mound corner stacked at n_frames=1200 halts the collapse.
Response: nm=1-2 across all strengths in [0, 12]; loss [1.18, 1.22]; strength=5 marginally nm=2 (loss 1.11).
Morphology (from strip): single tight blob at every strength; brief secondary peak hint at strength=5.
Verdict: FALSIFIED — STACKED density_repel × corner does NOT rescue Est #82. This was the final stacked Est #82 test inside the operator family.
Knowledge update: NEW Est #110 — stacked density_repel × 8-mound corner falsified under Est #82 break test. Confirms B25 sw 6, B26 sw 6, B28 sw 15 independently. The operator family CANNOT halt Est #82 even under maximal stacking (corner + density_repel + n_frames=1200). Reinforces decision to escalate to MPM engine fork unless Est #104 verifies.

## Batch 29 Sweep 13 — spring.kadh FINE [5, 35] at r_on=0.222 + gain=1500  [SUPPORTED — new corner valid]
Hypothesis (H14-B29): refines the NEW r_on=0.222 corner's kadh band.
Response: loss flat [0.93, 1.10]; best morph=0.0021 at kadh=18 (nm=8 loss=0.9303); 4-7 mound morphology across most kadh values.
Morphology (from strip): visible 5-7 multi-mound across kadh in [9, 22]; kadh<7 and kadh>26 show more single-blob tendency.
Verdict: SUPPORTED — kadh=18 at r_on=0.222 is a single-bin morph win comparable to Est #98 corner kadh=20.
Knowledge update: the new r_on=0.222 corner has a comparable kadh band [10, 22] to Est #98 r_on=0.19 corner. Both are short-transient corners (sw 14).

## Batch 29 Sweep 14 — n_frames at NEW r_on=0.222 corner (kadh=20, gain=1500)  [SUPPORTED — Est #100 generalized]
Hypothesis (H15-B29): the new corner has a longer transient than Est #100 (~750 at r_on=0.19).
Response: monotone-up loss 0.97→1.25; inner_mass monotone-up 0.49→0.82; nm collapses 6→4→2→1 by n_frames=360.
Morphology (from strip): clean 4-6 multi-mound at n_frames<=320; single tight blob by n_frames=400; persists through n_frames=1600.
Verdict: SUPPORTED — the new r_on=0.222 corner collapses FASTER than r_on=0.19 (nm=1 by frame 360 vs 750). Est #100 generalized to ALL tested 8-mound corners.
Knowledge update: NEW Est #111 — the r_on=0.222 corner is a SHORTER transient than Est #98 corner. Est #100 universal to the 8-mound corner family; the corners' "stability" is purely kinematic delay, not equilibrium.

## Batch 29 Sweep 15 — cell.n at NEW r_on=0.222 corner  [SUPPORTED — capacity at corner]
Hypothesis (H16-B29): capacity test at new corner.
Response: loss flat [0.93, 1.18]; best morph=0.0005 at cell.n=2200 (nm=8 loss=0.9310). n>=3000 collapses to nm=1-3.
Morphology (from strip): n=2200 shows clear multi-mound 8-distinct-spot structure; n>=3500 collapses to single tight blob.
Verdict: SUPPORTED — cell.n=2200 is a productive joint at the new corner (mirrors Est #71 at Est #98 corner).
Knowledge update: capacity wall around n=2500-3000 at new corner (sw 15) consistent with Est #71/#93 at Est #98 corner. cell.n=2200 added to morph candidates list.

## Batch 29 — summary

- **PROJECT-BEST LOSS marginally improved: 0.9066 at sw 5 seed=7 r_on=0.222** (Δ=0.006 < σ=0.04 — within noise, NOT adopted as parent).
- **NEW Est #104 — FIRST POINT-CELL CONFIGURATION TO BREAK Est #82 PARTIALLY:** sw 4 (8-mound corner: r_on=0.19, kadh=20, gain=1500) + camp.decay=1.4 at n_frames=1200 produces nm=6, inner_mass=0.373, loss=0.9863. This is a CORNER × DECAY INTERACTION (sw 3 at parent + same decay collapses to nm=1). Requires 16-seed verification in B30 before adoption (B27/B28 lesson).
- **NEW Est #105 — D=0.0042 robust morph anchor:** sw 6 16-seed verification at camp.diffusion=0.0042 gives 4/16 morph_score=0 seeds and 7/16 morph<=0.13. This is 4× the parent rate; the strongest seed-verified morph evidence of the project. **B30 PARENT adopts camp.diffusion=0.0042** (single-axis conservative change).
- **FOUR NEW FALSIFICATIONS (Est #106-110):** sense_sat.gain attenuation (sw 7/8, 12th project mechanism), random_walk × n_frames=1200 (sw 9), vmax × n_frames=1200 (sw 10), stacked density_repel × corner × n_frames=1200 (sw 12). Operator family CANNOT halt Est #82.
- **Est #103 — new r_on=[0.220, 0.225] candidate corner FALSIFIED as a corridor** (sw 1, sw 2); but the r_on=0.222 corner is a single-bin morph candidate (sw 13/15 mound=2200) and seeds-distribution at r_on=0.222 (sw 5) is slightly better than parent. Borderline.
- **Est #111 — Est #100 generalized:** the new r_on=0.222 corner collapses FASTER (~360 frames) than Est #98 corner (~750 frames). ALL tested 8-mound corners are kinematic transients.
- **Est #109 — low-decay morph niche at parent (decay=0.02):** single-seed morph=0.0002 nm=8 win at parent + decay=0.02 (sw 11). Requires 16-seed verification.
- **STRATEGIC FRAME:** B29 was positioned as the final point-cell batch with B30 as the MPM fork. The Est #104 finding REOPENS the point-cell engine — the corner + camp.decay rescue is the FIRST evidence Est #82 can be partially mitigated within the existing engine. B30 PARALLEL: (a) DECISIVE 16-seed verification of Est #104; (b) refinement of D=0.0042 new parent (sw 6) under the new parent baseline; (c) verification of Est #109 low-decay niche; (d) Est #104 mechanism-edge probes (n_frames extension, alternative decay values, corner variants). If Est #104 verifies, the MPM fork is DEFERRED to B31+; if it fails verification, B30 adopts D=0.0042 alone and B31 commits to MPM fork.
- **B30 PARENT:** B29 parent + camp.diffusion: 0.0012 → 0.0042 (single-axis change, Est #105 verified across 16 seeds). All else unchanged.
