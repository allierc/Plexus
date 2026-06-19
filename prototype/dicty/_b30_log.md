# Batch 30 log — Est #104 ADJUDICATION + D=0.0042 PARENT VINDICATION

## Batch 30 Sweep 0 — seed (Est #104 verification at r_on=0.19, kadh=20, gain=1500, decay=1.4, n_frames=1200)  [falsified-morphologically]
Hypothesis: 16-seed at Est #104 config — if median loss <=1.05 with >=3/16 seeds at nm>=4, Est #104 RECONFIRMED.
Response: Numerically the verification PASSES — 12/16 seeds at nm>=4, median loss ~1.00. BUT morphology strip shows SINGLE TIGHT CENTRAL BLOB with diffuse halo at EVERY seed. The peak detector counts halo scatter as "mounds".
Morphology (from strip): single central blob + sparse halo at all 16 seeds; no genuine multi-mound morphology anywhere; high nm counts are detector artefact on halo speckle.
Verdict: FALSIFIED MORPHOLOGICALLY despite numerical pass — Est #104 is a metric artefact (peak detection on diffuse halo). The corner+decay+long-n_frames combination only DELAYS collapse, it does not produce REAL multi-mound morphology.
Knowledge update: Est #104 RETRACTS as a morphological mechanism; reclassify as a peak-detector artefact on halo scatter. The visual evidence DECISIVELY overrules numerical verification. Lesson: 16-seed pass is necessary but not sufficient — morphology strip is the adjudicator.

## Batch 30 Sweep 1 — camp.decay [0.5, 3.0] at 8-mound corner + n_frames=1200  [falsified]
Hypothesis: refine the decay rescue band found in B29 sw 4.
Response: U-shape — loss [0.96, 1.18] at decay in [0.5, 1.6], catastrophic explosion >1.8 (loss 3.7-13.2, inner_mass falls to 0.21). Best at decay=1.4 loss=0.963.
Morphology (from strip): SINGLE TINY CENTRAL POINT at decay 0.5-1.5; transitions to DIFFUSE NO-AGGREGATION cloud at decay>=2.0 (field destroyed). No multi-mound morphology anywhere.
Verdict: FALSIFIED — the entire decay rescue corridor at the corner produces single-point morphology in the no-explosion band. Est #104 corner+decay is monomorphic; the only "win" is decay=1.4 sitting at the regime boundary.
Knowledge update: confirms Sweep 0 — Est #104 corner produces collapse irrespective of decay. The decay sweep refines the catastrophe wall (>=1.8 destroys field) but does not unlock multi-mound.

## Batch 30 Sweep 2 — n_frames at Est #104 (corner + decay=1.4)  [falsified — Est #82 holds]
Hypothesis: if Est #104 sustains nm>=4 to n_frames>=1600, it is a true attractor not just an extended transient.
Response: monotone-DOWN nm (17, 19, 17, 20, 23, 16, 25, 13, 7, 2, 1, 1, 1, 1, 1, 1) as n_frames extends 200 → 2400. inner_mass monotone-UP 0.245 → 0.438. Loss V-shaped with minimum at n_frames=1200 (0.96).
Morphology (from strip): sparse-scatter multi-spot at n_frames=200-600 → coalescence to fewer larger spots at 750-1050 → single tiny central point by 1500+. EXACTLY the same runaway-compaction trajectory as Est #82 — just delayed by ~300-500 frames vs parent collapse.
Verdict: FALSIFIED — Est #104 is a DELAYED TRANSIENT, not a stable attractor. Est #82 holds at Est #104 just as everywhere else. The n_frames=1200 "win" is the lucky measurement window where collapse is in mid-stride.
Knowledge update: NEW Est candidate — Est #104 morphology is an extended Est #100 transient; the corner+decay only buys ~400 frames of stability before collapse completes. Est #82 NOT mitigated by Est #104.

## Batch 30 Sweep 3 — camp.decay at r_on=0.222 corner + n_frames=1200  [falsified]
Hypothesis: does Est #104 decay rescue transfer to the alternative r_on=0.222 corner family?
Response: monotone-up loss (1.21 → 16.25 by decay=3.0) with same explosion at decay>=2.0; best at decay=1.6 (1.00).
Morphology (from strip): single tiny central point across decay [0.5, 1.6]; diffuse no-aggregation at decay>=1.8. Identical failure to sw 1 r_on=0.19 corner — NO transfer of multi-mound morphology.
Verdict: FALSIFIED — Est #104 mechanism does NOT transfer to the alternative corner family; the corner+decay combo is monomorphic in BOTH r_on candidates.
Knowledge update: dual confirmation Est #104 is not a productive rescue regime under EITHER candidate corner; the entire 8-mound-corner family fails the n_frames=1200 break test.

## Batch 30 Sweep 4 — cell.n [400, 6500] at Est #104  [falsified — single-blob persists]
Hypothesis: capacity test under Est #104 rescue.
Response: U-shaped — n=400-1400 high loss (1.43-2.34); n=1400-1800 valley (0.96); n>=2000 single-blob plateau (1.13-1.18); morph_score ~0.9 across high n with nm=1.
Morphology (from strip): scattered fragments at low n, multi-spot transient at n=1400-1800 (matching parent), then single-blob from n=2200 onwards. Identical to Est #93 high-n single-attractor collapse.
Verdict: FALSIFIED — Est #93 high-n capacity wall HOLDS at Est #104 config; cell.n>=2200 single-blobs regardless of corner+decay rescue.
Knowledge update: reconfirms Est #93 — the high-n collapse is structural, not corner-conditional. Est #104 does not generalize as a capacity-extending rescue.

## Batch 30 Sweep 5 — seed at NEW PARENT (D=0.0042)  [STRONGLY SUPPORTED — BIG WIN]
Hypothesis: re-pin noise floor at new D=0.0042 parent.
Response: loss [0.92, 1.04], median ~0.945, σ≈0.026 — tightest noise floor of the project. morph_score in [0, 0.625] with 7/16 at morph<=0.13, including 3/16 morph≈0; n_mounds widely 5-10.
Morphology (from strip): CLEAR GENUINE MULTI-MOUND at every seed — distinct dense compact mounds (visually 4-8) scattered across the FOV, resembling REAL morphology. The strongest morphological evidence in the project.
Verdict: STRONGLY SUPPORTED — the NEW D=0.0042 parent IS a morphologically credible multi-mound producer at frame 400. This is the central B30 finding.
Knowledge update: NEW Est candidate — D=0.0042 PARENT (without any corner/decay/rescue) is the first parent in 30 batches to REPLICATE genuine multi-mound morphology across 16 seeds at default n_frames=400. PROJECT-BEST LOSS = 0.9226 at sw 5 seed=11.

## Batch 30 Sweep 6 — camp.diffusion [0.001, 0.012] at new parent  [supported — broad working band]
Hypothesis: re-resolve productive D corridor under D=0.0042 baseline.
Response: loss flat-favorable in [0.92, 1.07] across D in [0.001, 0.012], best at D=0.002 (0.918). morph_score ~0.13 at most values with 3 settings at morph<=0.001.
Morphology (from strip): genuine multi-mound across full sweep range (cells form 3-9 distinct dense clusters); only D=0.009 onwards begins to lose definition (3 mounds, larger spots).
Verdict: SUPPORTED — productive D corridor refined to [0.001, 0.012] (visually [0.0015, 0.008] for best morphology). D=0.0042 sits near the corridor center; D=0.002 marginally better on loss.
Knowledge update: Est #99 D corridor BROADENED to [0.001, 0.012] under new parent; refines #99 from [0.0022, 0.0055].

## Batch 30 Sweep 7 — spring.r_on [0.17, 0.24] at new D parent  [supported — broad working band]
Hypothesis: re-map Est #98 corner under D=0.0042.
Response: loss broadly flat [0.92, 1.10], best r_on=0.205 (0.924); morph_score=0 at r_on=0.20 and r_on=0.228.
Morphology (from strip): genuine multi-mound at every r_on value; mound count varies slightly but morphology stable; no sharp r_on dependence at the new parent.
Verdict: SUPPORTED — at D=0.0042, the r_on lever is BROAD (the Est #98 sharp corner dissolves under the productive D regime). The new parent does not need the r_on corner trick.
Knowledge update: Est #98 corner sensitivity is D-conditional; under D=0.0042 r_on is broadly productive across [0.17, 0.24]. Refines/partially retracts the corner specificity.

## Batch 30 Sweep 8 — spring.kadh [5, 35] at new D × r_on=0.19  [PROJECT-BEST LOSS]
Hypothesis: does the Est #98 kadh band [10, 22] hold under D=0.0042 parent?
Response: loss flat [0.91, 1.06], best at kadh=9 → loss=0.9055 inner_mass=0.376 (NEW PROJECT BEST). morph_score=0 at kadh=35.
Morphology (from strip): genuine multi-mound across the full kadh range; mound spacing and count visually consistent — kadh is largely a regularizer at this regime.
Verdict: SUPPORTED — kadh broadly productive [5, 35] at new D, kadh=9 numerically best (single-seed; needs 16-seed verification).
Knowledge update: NEW PROJECT-BEST LOSS = 0.9055 at sw 8 kadh=9. Marginal over 0.9090 / 0.9126 history. kadh=9 candidate B31 axis (single-seed-deep — flag for verification).

## Batch 30 Sweep 9 — sense_sat.gain [200, 5000] at new D parent  [supported]
Hypothesis: re-pin Est #54 gain plateau under D=0.0042.
Response: loss flat [0.93, 1.01] across gain in [200, 5000], slight optimum at gain=4000 (0.935). morph_score=0 at gain=500.
Morphology (from strip): genuine multi-mound at every gain; no regime change across nearly 2 decades. gain saturation reconfirmed.
Verdict: SUPPORTED — Est #54 gain plateau [200, 7000] holds under new D parent; gain is D-invariant.
Knowledge update: gain plateau confirmed D-invariant; the lever is saturated as previously found.

## Batch 30 Sweep 10 — cell.n [400, 6500] at new D parent  [partial — Est #93 holds]
Hypothesis: capacity test at new parent baseline.
Response: loss decreases from 1.06 (n=400) → 0.94 (n=1800-2200) → climbs back to 1.10 (n=5000); morph_score=0 at n=1800 and n=2200; morph_score>0.5 at n>=3000.
Morphology (from strip): clean multi-mound at n=400-3000; degrades to fewer-larger spots at n=3500-4500; collapses to single small point at n>=5000.
Verdict: PARTIALLY SUPPORTED — Est #71 productive range narrows under new D to [1100, 3000]; Est #93 high-n collapse holds (n>=3500 begins single-attractor regime).
Knowledge update: Est #71 capacity refined under new D parent: productive n in [1100, 3000]; Est #93 high-n single-attractor holds at n>=3500.

## Batch 30 Sweep 11 — seed at decay=0.02, new D parent (Est #109 verification)  [supported]
Hypothesis: verify B29 sw 11 single-seed morph=0.0002 nm=8 finding.
Response: loss [0.91, 1.22], median ~0.96, σ≈0.07; morph_score in [0.001, 1.0] with 7/16 at morph<=0.25 and 2/16 morph<=0.013.
Morphology (from strip): genuine multi-mound at most seeds (4-12 dense mounds visible); seed 15 single-blob (1.22 loss); modest seed-spread but visually robust multi-mound morphology.
Verdict: SUPPORTED — Est #109 (low decay=0.02 at new D) replicates as a productive morph niche across seeds. Project-second-best at sw 11 seed=8 = 0.9090.
Knowledge update: Est #109 PROMOTED — decay=0.02 at D=0.0042 parent is a productive low-decay morph niche, second-best replicated regime in the project after the default new parent.

## Batch 30 Sweep 12 — density_repel.strength [0, 6] at Est #104  [falsified — does not rescue]
Hypothesis: stacked density_repel + Est #104 rescue improvement.
Response: noisy with intermittent catastrophes at strength 0.3-1.3 (1.44-1.77); all strengths >=1.6 plateau at parent-equivalent loss; morph_score >=0.125 mostly.
Morphology (from strip): single tiny central point at every strength; identical to sw 0 Est #104 morphology — no improvement.
Verdict: FALSIFIED — density_repel does not rescue Est #104 morphology; reconfirms Est #110.
Knowledge update: Est #110 reconfirmed; density_repel stacking with Est #104 yields no morphological gain.

## Batch 30 Sweep 13 — sense_sat.c_sat [0.1, 2.0] at Est #104  [partial]
Hypothesis: c_sat ridge under Est #104 protection.
Response: catastrophic at c_sat<=0.15 (loss 1.20, inner_mass 0.86 — single tight blob); flattens at c_sat>=0.5 in [0.96, 1.04]; morph_score=0 at c_sat=1.6.
Morphology (from strip): tiny tight point at c_sat<=0.20; diffuse cloud at c_sat=0.25-0.40; broad halo-blob at c_sat>=0.50. NO genuine multi-mound at any value — confirms Est #104 morphology is monomorphic-blob.
Verdict: PARTIALLY SUPPORTED on c_sat plateau (Est #49/#57 ridge [0.5, 2.0] holds at this joint) but FAILS to produce multi-mound morphology. Reconfirms Est #104 retraction.
Knowledge update: c_sat ridge holds; Est #104 morphology remains monomorphic under c_sat sweep.

## Batch 30 Sweep 14 — cell.n at Est #104 (independent variant of sw 4)  [reconfirms sw 4]
Hypothesis: independent capacity test at Est #104 with different seed sequence.
Response: U-shaped (loss 2.33 → 0.95 at n=700-1000 → 1.18 plateau at n>=2200); morph_score>=0.875 at n>=2200; nm=1 throughout high-n.
Morphology (from strip): scattered fragments at n=400; transient multi-spot at n=700-1300; single tiny central point at n>=1900. Identical failure mode to sw 4.
Verdict: SUPPORTED (cross-replicates sw 4) — Est #93 capacity wall holds; Est #104 fails to extend capacity.
Knowledge update: independent confirmation of sw 4 and Est #93.

## Batch 30 Sweep 15 — sense_sat.sat_n [1.0, 4.0] at Est #104  [partial]
Hypothesis: sat_n plateau under Est #104 protection.
Response: loss flat-tied at [0.95, 1.22] across sat_n in [1.0, 3.0], climbs at sat_n>=3.4 (1.20, 1.23). morph_score=0.0001 at sat_n=1.7.
Morphology (from strip): single tight blob across sat_n=1.0-3.0; tighter point + radial-explosion artefact at sat_n>=3.4 (likely sense_sat operator regime change). No multi-mound at any sat_n.
Verdict: PARTIALLY SUPPORTED on sat_n plateau (Est #70/#83 broad band [1.5, 2.7] holds) but FAILS to produce multi-mound under Est #104.
Knowledge update: sat_n plateau confirmed D-invariant; reinforces Est #104 retraction.

## Batch 30 — summary

- **PROJECT-BEST LOSS = 0.9055** at sw 8 spring.kadh=9 (single-seed; needs verification). Improves on B29's 0.9066 by Δ=0.001 (<σ=0.026 noise floor) — marginal.
- **DECISIVE B30 FINDING — Est #104 RETRACTS as METRIC ARTEFACT:** 16-seed verification (sw 0) numerically PASSES (12/16 seeds nm>=4, median loss ~1.00) BUT morphology strip shows SINGLE CENTRAL BLOB at every seed. The peak detector counts halo speckle as "mounds". Sw 2 (n_frames at Est #104) reveals Est #104 is a DELAYED TRANSIENT, NOT a true attractor — collapse trajectory identical to Est #82, just delayed by ~400 frames. NEW Est #112: Est #104 morphology is monomorphic single-blob with halo; the corner+decay+long-n_frames combination only buys time. **Est #82 NOT mitigated by Est #104.**
- **DECISIVE B30 FINDING #2 — D=0.0042 NEW PARENT IS THE BREAKTHROUGH:** Sw 5 (16-seed at new parent) shows genuine multi-mound morphology (4-8 dense compact mounds visible) across ALL 16 seeds with the tightest noise floor of the project (σ≈0.026). This is the FIRST point-cell parent in 30 batches to replicate REAL-like multi-mound morphology under SEED variation. NEW Est #113: D=0.0042 parent is a structurally credible multi-mound regime at n_frames=400.
- **Est #109 VERIFIED (NEW Est #114):** sw 11 16-seed at decay=0.02 reproduces multi-mound at most seeds (7/16 morph<=0.25). Low-decay productive niche confirmed.
- **PRODUCTIVE BANDS WIDEN UNDER NEW PARENT:** Est #99 D corridor [0.001, 0.012] (sw 6 broadens from [0.0022, 0.0055]); Est #98 r_on corner DISSOLVES — r_on broadly productive [0.17, 0.24] under new D (sw 7); Est #54 gain plateau D-invariant (sw 9); Est #71 capacity range refined to [1100, 3000] under new D (sw 10); Est #70/#83 sat_n plateau D-invariant (sw 15).
- **kadh=9 candidate B31 axis (single-seed; needs 16-seed verification);** Est #98 corner config kadh band narrows under new parent.
- **THE ENGINE-LEVEL Est #82 CRITICAL TEST STILL UNRUN AT NEW PARENT** — sw 2 (n_frames sweep) was at Est #104 corner, NOT at the bare new parent. The decisive question for B31: does the D=0.0042 multi-mound morphology SURVIVE to n_frames=1200, or is it (like everything else) a transient?
- **B31 PARENT: keep B30 parent (D=0.0042) — no single-axis change replicated with morph-supported robustness yet.** kadh=9 needs verification before adoption.
- **B31 STRATEGIC FRAME — THREE PRONGS:** (i) DECISIVE n_frames sweep at NEW PARENT itself (sw 0, 1) — is the D=0.0042 multi-mound regime ATTRACTOR or TRANSIENT? This is the single most important test left in the point-cell engine. (ii) 16-seed verification at kadh=9 (sw 2) and at sw 5 morph-best D=0.002 (sw 3); refine working bands under new D (sw 4-11). (iii) Test if multi-mound at new parent survives n_frames=1200 across cell.n, decay, kadh joints (sw 12-15). If the new parent SURVIVES n_frames=1200 with multi-mound preserved, Est #82 is FINALLY mitigated by a structural parameter change (D shift), not by mechanism addition. If it collapses, the MPM fork remains the principled escalation — but now with EVIDENCE that point-cell engine has at least an n_frames=400 multi-mound regime.
