# Batch 21 — per-sweep log (canonical artefact; mirrors _b20_log.md convention)

**HEADLINE:** B21 cannot adjudicate diff_dens — the operator is SILENT due to an
attribute-name bug. All 6 diff_dens sweeps (1, 2, 3, 5, 10, 11, 15) return
BIT-IDENTICAL losses across all kappa values [0, 120]. Root cause: `dicty_ops.py`
DiffDens.forward reads `getattr(fld, "diffusion", 0.0)` but `GridField` stores
the diffusion constant as `self.D` (set in `grid_field.py:31`). So D0 = 0 for
every kappa → operator subtracts zero → no-op. **FIXED in dicty_ops.py:918** by
falling back to `fld.D`. Mechanism not adjudicated; re-tested in B22 with the
fix as the FIRST priority. Best loss = 0.9111 (parent, unchanged from B20).

## Batch 21 Sweep 0 — seed  [supported (parent reproducible)]
Hypothesis: H1-B21 — secrete=9 + diff_dens(kappa=0) preserves Est #48 multi-mound morphology across seeds.
Response: noisy; loss range [0.911, 1.229], σ≈0.11; best seed=0 at 0.9111 (parent reference).
Morphology (from strip): all 16 seeds produce 4-6 visible dense compact mounds; seeds 2, 4, 6, 8 visibly looser per-mound.
Verdict: supported with caveat — morphology robust, but seed-noise floor WIDER than B20 sw 0 (σ≈0.04). secrete=9 noise floor is ~2.5× B20 secrete=11 noise floor.
Knowledge update: refines Est #48 — seed-noise width is regime-dependent. The B21 parent (secrete=9) has σ≈0.11 vs σ≈0.04 at secrete=11. Inflates the noise floor for B21 single-axis adjudications. Single-axis "wins" of Δ<0.05 inside noise.

## Batch 21 Sweep 1 — diff_dens.kappa [0, 120]  [INVALID — operator silent]
Hypothesis: H2-B21 — density-modulated diffusion breaks the 5-7 mound ceiling toward REAL=8.
Response: FLAT at loss=0.9111 inner=0.317 across ALL 16 values (kappa ∈ {0, 0.05, ..., 120}); bit-identical.
Morphology (from strip): bit-identical 4-mound morphology at every kappa value, indistinguishable from kappa=0 ablation.
Verdict: INVALID — cannot adjudicate. The DiffDens operator reads `fld.diffusion` (attribute does not exist on GridField; GridField stores `self.D`) so D0=0 → the operator subtracts 0 from camp.grid regardless of kappa.
Knowledge update: NO knowledge update for the mechanism; flagged as implementation bug. Fix applied to dicty_ops.py:918 (fallback `getattr(fld, "D", getattr(fld, "diffusion", 0.0))`). diff_dens necessity+sufficiency test moved to B22 sw 1.

## Batch 21 Sweep 2 — diff_dens.kappa × c_sat=0.30  [INVALID — operator silent]
Hypothesis: H3-B21 — kappa rescues the sparse densification-handle column at c_sat=0.30.
Response: FLAT at loss=0.9733 inner=0.278 across all 16 kappa values; bit-identical (joint fixed parameter c_sat=0.30 is non-parent baseline 0.9733).
Morphology (from strip): bit-identical clean 5-mound morphology, denser and more REAL-like than parent; produced by the c_sat=0.30 sat_n=2.0 joint, NOT by kappa.
Verdict: INVALID for the hypothesis. SIDE-FINDING: c_sat=0.30 + sat_n=2.0 at the B21 parent gives 5-mound morphology with inner=0.278; loss 0.9733 within seed-noise floor. The Est #53 densification column is still productive at the B21 parent (secrete=9, kadh=10).
Knowledge update: corroborates Est #53/#57 ridge column productive at the B21 parent (morphology, not loss).

## Batch 21 Sweep 3 — diff_dens.kappa × sense_sat.gain=1500  [INVALID — operator silent]
Hypothesis: H4-B21 — kappa rescues the densification regime at gain=1500.
Response: FLAT at loss=1.1015 inner=0.336; bit-identical.
Morphology (from strip): bit-identical 5-6 CRISP multi-mound morphology; the gain=1500 regime visibly densifies each mound; clearest multi-spot morphology of the batch.
Verdict: INVALID for the hypothesis. SIDE-FINDING: gain=1500 produces the SAME multi-mound morphology Est #53 documented, with mounds slightly looser (per-mound density lower) than parent. Loss 1.10 above parent because SSIM penalises sparser-per-mound (Est #42).
Knowledge update: re-confirms Est #53 morphological mound-multiplier at the B21 parent.

## Batch 21 Sweep 4 — sense_sat.sat_n [1.85, 2.5] FINE  [supported plateau]
Hypothesis: H5-B21 — sat_n=2.1 tied/peak at narrow refinement.
Response: noisy plateau loss=[0.911, 1.252]; best=2.1 at 0.9111 (parent); single-replica spikes at 2.06 (1.18), 2.16 (1.24), 2.25 (1.25), 2.20 (1.11); plateau dips at 2.1, 1.9 (0.93), 2.5 (0.93).
Morphology (from strip): visually flat 4-mound morphology at every sat_n; no transition; sat_n=1.85 marginally more diffuse, sat_n=2.5 marginally denser.
Verdict: supported — sat_n=2.1 is parent-tied within noise across a broad band. Est #70 (sat_n plateau [1.6, 2.7]) re-confirmed at the finer scale.
Knowledge update: tightens Est #70 — plateau is flat-noisy at the σ≈0.11 secrete=9 floor across [1.85, 2.5]. No sub-grid optimum.

## Batch 21 Sweep 5 — diff_dens.kappa × cell.n=2500  [INVALID — operator silent]
Hypothesis: H6-B21 — kappa × high-n densifies the high-cell-n regime.
Response: FLAT at loss=1.0516 inner=0.342; bit-identical.
Morphology (from strip): bit-identical 5-mound morphology, looser per-mound than sw 3 because more cells spread over same area.
Verdict: INVALID. SIDE-FINDING: cell.n=2500 at the B21 parent gives 5-mound with inner=0.342; reconfirms Est #71 (cell.n is not a count-densifier).
Knowledge update: none new.

## Batch 21 Sweep 6 — relay.gain [100, 320] FINE  [supported plateau]
Hypothesis: H7-B21 — relay.gain=140 plateau center under secrete=9 + diff_dens parent.
Response: bumpy plateau loss=[0.911, 1.489]; best=140 (parent) at 0.9111. Single-replica spikes at 130 (1.49), 190 (1.24), 100 (1.32), 110 (1.23); plateau [140, 320] mostly 0.93-1.10.
Morphology (from strip): consistent 4-mound morphology across the range; no qualitative transition; the spikes are seed-noise within the same mound count.
Verdict: supported — Est #67 (relay.gain plateau [100, 280]) holds at the B21 parent; extends to 320 at parent-tied loss.
Knowledge update: refines Est #67 — plateau extends to 320 (was [100, 280]); the upper edge of monotone-up degradation pushed back.

## Batch 21 Sweep 7 — vmax [0.057, 0.075] FINE-PINS  [supported aliasing landscape]
Hypothesis: H8-B21 — pin the four working dips around resonance spikes.
Response: highly bumpy; best=0.060 (parent) at 0.9111; secondary dip at 0.0615 (0.9344). Spikes at 0.058 (1.60), 0.062 (1.26), 0.0715 (1.33), 0.0743 (1.62), 0.0746 (1.53), 0.0733 (1.24); working dips at 0.060, 0.0615, 0.0728 (1.02), 0.0738 (0.982).
Morphology (from strip): morphology degrades at vmax≥0.062 (more diffuse, less compact); aliasing visible as one-blob/multi-blob alternation.
Verdict: supported — Est #69 aliasing landscape RE-CONFIRMED. Cleaner working dips at vmax=0.060/0.0615 (parent band); pseudo-working at 0.0738 with caveat (single-replica noise).
Knowledge update: refines Est #69 — vmax=0.060 + 0.0615 is the most reliable working pair; 0.0738 still a candidate but flanked by 1.5+ spikes (fragile).

## Batch 21 Sweep 8 — secrete.rate [7.5, 12] FINE  [supported — secrete=9 parent-tied]
Hypothesis: H9-B21 — rate=9 is a sharp dip vs noise-floor plateau.
Response: rate=7.5 catastrophic (1.40, low-end dispersion); rate=8 elevated (1.16); rate=8.5-8.75 elevated (1.22-1.33); rate=9 = 0.9111 sharp dip; rate=9.1-9.4 = 1.05-1.16; rate=9.6-11.5 plateau 0.93-1.05; rate=12 = 1.09.
Morphology (from strip): morphology degrades at rate<9 (fewer mounds, more diffuse); rate=9-11 = stable 4-mound; rate=11.5 marginally denser.
Verdict: supported with nuance — rate=9 is a sharp dip but the [9.6, 11.5] plateau is loss-tied with it within noise. rate=11 (B20 parent) survives as a near-tied alternative (0.9126). Est #62 morphology-safe band [8, 14] holds; rate<8 is dispersive.
Knowledge update: refines secrete safe band lower edge — rate=7.5 is the new catastrophe threshold; rate=8 marginal (1.16). rate=9 dip is real but within noise of [9.6, 11.5] plateau.

## Batch 21 Sweep 9 — camp.diffusion [0.0001, 0.07] WALL EDGE  [supported]
Hypothesis: H10-B21 — re-pin Est #65 catastrophe wall at the new parent.
Response: plateau loss=[0.91, 1.33] across D=[0.0001, 0.035] (working band); transition D=0.045 (2.30); wall D∈{0.05, 0.055, 0.06, 0.063, 0.066, 0.07} with loss 1.61, 2.82, 1.86, 1.85, 4.36, 2.63.
Morphology (from strip): D≤0.035 = stable 4-mound; D=0.045 onward = sparse-diffuse scatter (field smears out chemotaxis); D=0.066 = near-empty FOV.
Verdict: supported — Est #65 wall RE-CONFIRMED. Wall transition is at D~0.045; collapse complete by D=0.05. Working band slightly tighter than B20 estimate.
Knowledge update: refines Est #65 — wall transition starts at D=0.045 (was D~0.05). Working band [0.0001, 0.035] at the B21 parent.

## Batch 21 Sweep 10 — diff_dens.kappa × relay.thr=0.30  [INVALID — operator silent]
Hypothesis: H11-B21 — kappa rescues high-thr sparse-multi regime (Est #33 candidate).
Response: FLAT at loss=0.9668 inner=0.339; bit-identical.
Morphology (from strip): bit-identical clean 5-mound morphology at every kappa value; relay.thr=0.30 IS productive at the B21 parent — produces visibly cleaner multi-mound than parent (relay.thr=0.22).
Verdict: INVALID. SIDE-FINDING: relay.thr=0.30 + sat_n=2.1 + secrete=9 + gain=500 produces 5-mound at loss=0.9668 — very close to parent (within seed noise). Promising joint to recharacterize in B22.
Knowledge update: tentatively suggests Est #33's sparse-multi candidate may now sit closer to the parent loss under the B21 parent (vs B12's far-worse loss). Open question for B22.

## Batch 21 Sweep 11 — diff_dens.kappa × spring.kadh=20  [INVALID — operator silent]
Hypothesis: H12-B21 — kappa × kadh=20 adhesion-coupling joint.
Response: FLAT at loss=1.2459 inner=0.229; bit-identical (kadh=20 is OFF-parent-tied).
Morphology (from strip): bit-identical strikingly CRISP 5-6 distinct mounds — **visually the BEST multi-mound morphology of the batch**, but each mound is SPARSER (low inner=0.229), so SSIM penalises.
Verdict: INVALID for the hypothesis. SIDE-FINDING: kadh=20 at the B21 parent gives visually-best multi-mound morphology of the batch (5-6 crisp mounds) at loss=1.25 — confirms Est #42 (SSIM/morphology divergence) and Est #40 (low kadh = denser/multi at sense_sat). The kadh=20 joint may be worth re-considering as a candidate parent if SSIM is augmented.
Knowledge update: reaffirms Est #42 (SSIM bias against sparser-per-mound multi-mound) at the B21 parent.

## Batch 21 Sweep 12 — cell.n [800, 3400] RECONFIRM  [supported flat plateau]
Hypothesis: H13-B21 — Est #71 (no capacity wall) holds at secrete=9 + diff_dens(silent).
Response: flat-noisy plateau across full range; best=1800 (parent) at 0.9111; single-replica low at 1100 (0.96), 1400 (0.95), 3300 (0.99); spike at 1600 (1.24), 3000 (1.25).
Morphology (from strip): consistent 4-mound across n; n=800 mounds tighter; n=3400 mounds looser (more cells spread same area).
Verdict: supported — Est #52/#71 (cell.n not a count-densifier; no capacity wall) RE-CONFIRMED at the B21 parent.
Knowledge update: none new; reconfirms.

## Batch 21 Sweep 13 — seed (DUPLICATE OF SW 0)  [redundant]
Hypothesis: noise floor measurement.
Response: BIT-IDENTICAL to sw 0 (same param, same values, same outcomes).
Morphology (from strip): identical to sw 0.
Verdict: REDUNDANT — wasted slot. Plan misalignment: sw 13 was intended as a separate measurement but the plan slotted "seed" again.
Knowledge update: none; flag to NOT duplicate sweeps in B22 (lost 1/16 slots).

## Batch 21 Sweep 14 — sense_sat.sat_n [1.7, 3.2] BROADER  [supported broad plateau]
Hypothesis: H15-B21 — sat_n × kappa=2.0 joint (kappa silent → just sat_n breadth).
Response: noisy plateau loss=[0.911, 1.252]; best=2.1 (parent) at 0.9111; secondary lows at 2.8 (0.93), 3.2 (0.96), 1.95 (0.95), 2.4 (0.96); spikes at 2.05 (1.16), 2.2 (1.11), 2.25 (1.25).
Morphology (from strip): broad band — 4-5 mound morphology across full [1.7, 3.2] range; sat_n=1.7 slightly more diffuse; sat_n=3.0-3.2 slightly tighter.
Verdict: supported — Est #70 plateau extended further; sat_n=2.1 plateau actually extends to [1.7, 3.2] within noise (Est #70 was [1.6, 2.7]; B21 evidence pushes upper edge to 3.2).
Knowledge update: refines Est #70 — sat_n plateau is BROADER than estimated; at the B21 parent (secrete=9), [1.7, 3.2] is flat-within-noise.

## Batch 21 Sweep 15 — diff_dens.kappa × camp.diffusion=0.005  [INVALID — operator silent]
Hypothesis: H16-B21 — elevated base D makes kappa modulation more visible.
Response: FLAT at loss=1.2268 inner=0.304; bit-identical.
Morphology (from strip): bit-identical 4-mound morphology; camp.D=0.005 at parent is in Est #65 working band (loss elevated due to single-replica seed noise, not D itself).
Verdict: INVALID. SIDE-FINDING: camp.D=0.005 at the B21 parent gives loss=1.23 — slightly elevated vs D=0.0012 parent; consistent with the Est #65 working band [0.0001, 0.035] but with D=0.005 not parent-tied.
Knowledge update: none new.

## Batch 21 — SUMMARY

Parent loss: 0.9111 (unchanged from B20; secrete=9 + sat_n=2.1 + sense_sat.gain=500 + cell.n=1800 + r_on=0.20 + kadh=10 + relay.gain=140 + camp.D=0.0012 + camp.decay=0.07 + vmax=0.061 + inflow=4 + dt=0.5 + n_frames=400).

**Headline:** B21 invalid for adjudicating diff_dens — operator bug discovered
(typo `fld.diffusion` vs `fld.D`). 6 of 16 sweeps wasted on a silent operator;
1 of 16 wasted on a duplicate seed sweep (sw 13 == sw 0). NET: only 9 sweeps
informative. All 9 informative sweeps reconfirm parent at 0.9111 (sw 4, 6, 7,
8, 9, 12, 14) or extend Est-level scope (sw 8 lower secrete catastrophe edge,
sw 9 wall edge refined, sw 14 sat_n plateau broadened to [1.7, 3.2]).

**Key SIDE-FINDINGS from the silent diff_dens sweeps** (the fixed joints reveal
morphology even when kappa is no-op):
- sw 2 (c_sat=0.30): 5-mound morphology at loss=0.97 — densification column productive at B21 parent.
- sw 3 (gain=1500): 5-6 CRISP multi-mound at loss=1.10 — Est #53 densification axis re-confirmed.
- sw 10 (relay.thr=0.30): 5-mound morphology at loss=0.97 — Est #33 sparse-multi candidate closer to parent now.
- sw 11 (kadh=20): visually BEST 5-6 multi-mound of batch at loss=1.25 (SSIM-penalised by per-mound sparseness).

**Plan for B22:** Apply diff_dens.py fix and re-run the kappa necessity+sufficiency
test with maximally-diverse joint configurations. Keep 5 slots for plateau
refinement; drop the duplicate seed sweep; pull in the sw 10 (thr=0.30) and
sw 11 (kadh=20) JOINTS for joint × kappa probes since they showed cleanest
multi-mound morphology in B21.

Action items:
1. dicty_ops.py:918 fix applied (fallback `getattr(fld, "D", getattr(fld, "diffusion", 0.0))`).
2. Add log entry "B21 INVALID for diff_dens — implementation bug; re-test in B22" to dicty_loop_log.md.
3. New ledger entry: NOT a falsification; an "Open Question / Invalid Adjudication" record.
4. B22 plan: 11 slots for fixed diff_dens × diverse joints + 5 plateau refinements.
