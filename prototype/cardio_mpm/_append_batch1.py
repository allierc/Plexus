#!/usr/bin/env python3
"""Append Batch 1 section to analysis_cardio_mpm.md"""

content = """

---

## Batch 1 (LoopScore era) -- 2026-06-26

> **Phase 3 begins.** Objective = LoopScore. R2 is diagnostic only. All prior R2-era conclusions are provisional.

Parent: archive p2_b14_s1 = gain05_colearn (LS=0.589, LS_SD=0.080, R2=-1.649, amp10, drag30, dur->30, fibre_wl=28.8, gain0=0.5, stiff uniform [100,100], learn=fibre,gain,dur, 2400it).

**Surprise (from the archive baseline):** Three surprises from the 6-slot p2_b14 archive (the FIRST runs under LoopScore):
1. **Lower gain init (0.5) is the best LS** -- LS=0.589 vs 0.567-0.578 for gain0=0.854 variants, AND has the best uniformity (LS_SD=0.080 vs ~0.104). Under R2, gain0=0.854 was the reference; under LoopScore, 0.5 wins.
2. **SIREN fibre at omega=5 is CATASTROPHIC** (LS=0.079, ampL=9.5) -- the free fibre field created wild spatial variation that amplified overshoot. WORSE than inert -- actively harmful. R2-era closure of free fibre CONFIRMED under LoopScore.
3. **LS converges faster than R2** -- 3600it (LS=0.567) slightly WORSE than 2400it (LS=0.576 for same config). LS may plateau or degrade with excessive depth.

**Observation (systematic failure):** From the s1 dashboard, red sim loops approximate green GT loops but show: (a) wrong axis angle at several nodes, (b) insufficient openness (sim loops more collapsed/linear than GT), (c) some nodes with reversed chirality. The UNIFORM model cannot fix spatial variation in loop shape.

**Hypothesis:** Gain is the primary overshoot/shape lever under LoopScore: there exists a gain-init sweet spot in [0.3, 0.7]. Below ~0.3, loops collapse. Separately, coarse SIREN stiffness (rejected under R2) may help LS because regional loop-size variation requires a spatial lever.

**Slots designed (6):**

- Slot 0 [gain03] role=EXPLOIT -- gain0=0.3 (from parent 0.5). Push gain lower.
- Slot 1 [gain07] role=EXPLOIT -- gain0=0.7 (intermediate). Maps gain-LS curve.
- Slot 2 [deeper_3600] role=EXPLOIT -- n_iter=3600 (from parent 2400). Test convergence.
- Slot 3 [coarse_stiff] role=EXPLORE -- stiff_lo=50, stiff_hi=150, stiff_src=siren, siren_omega=5, learn=fibre,stiff,gain,dur.
- Slot 4 [short_dur] role=EXPLORE -- dur0=8, dur_hi=20. Shorter pulse -> different loop family?
- Slot 5 [parent_repro] role=CONTROL -- exact reproduction of parent.

Best optimizer slot: TBD (after training).
Best scientific slot: TBD (after training).
Verdict: TBD.
Batch outcome: TBD.
Next: TBD.
"""

with open("analysis_cardio_mpm.md", "a") as f:
    f.write(content)
print("Appended Batch 1 section.")
