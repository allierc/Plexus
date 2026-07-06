# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 33 — DOES A SIZE LEVER CONVERT TO LOOP SIZE IN THE ROTATING (enclosure-solved) REGIME?
#
# PARENT = dev18 (NEW op point, B32 best LS=0.492): rot1.0 + soft-floor stiff[30,300] ω5, drag40, amp10,
#   gain[0.2,1.5] gain0=0.5, SIREN fibre-ON dev0.18, dur_hi11, substeps10.
#
# SURPRISE (B32): with rotation, ENCLOSURE is SOLVED (loopiness_ratio 1.06-1.18, at/ABOVE real) — so the
#   dominant residual FLIPPED BACK to SIZE: clean real-referenced peak_ratio ~0.49 (sim peak = HALF real),
#   area_ratio ~0.35. Dashboard: red loops loopy + correctly-chiral but INSIDE green. SIZE was declared
#   "invariant to every lever" (facts #24/#25) — but ONLY at rot=0 (radial/time-reversible, drive->overshoot).
#   (Also: fibre_dev is a REAL monotone lever peaking ~0.18; fibre_wl SCALE inert.)
#
# HYPOTHESIS: in the rotating regime (which redistributes radial motion into circulation), extra
#   drive/gain/compliance may now CONVERT to loop SIZE (raise peak_ratio) instead of overshooting.
#   FALSIFIER: all size levers leave peak_ratio ~0.49 + lower LS -> size is capped INDEPENDENT of rotation
#   (structural: boundary --bwidth compliance / constitutive strain). OVERTURN: any lever raises peak_ratio
#   while holding LS+chir -> facts #24/#25 are regime-bound to rot=0.
#
#   Balance: 3 EXPLOIT (amp12, ghi20, dev20) . 2 EXPLORE (amp14, slo20) . 1 CONTROL (dev18 replicate).
#   Read peak_ratio + area_ratio from RESIDUAL_MORPHOLOGY, not LS alone (a bigger loop that overshoots LOWERS LS).
#
#   b33_dev18   [CONTROL/REPLICATE] : exact dev18 -- reproduce LS=0.492 (single draw), net the lottery in-batch.
#   b33_amp12   [EXPLOIT/size]      : amplitude 10->12 -- does more drive now GROW peak (revisit fact #25 @rot1.0)?
#   b33_ghi20   [EXPLOIT/size]      : gain_hi 1.5->2.0 -- raise the gain ceiling to push bigger loops (fact #28 @soft-floor+rot).
#   b33_dev20   [EXPLOIT/dose]      : fibre_dev 0.18->0.20 -- nail the fibre-dose peak / pin the operating point.
#   b33_amp14   [EXPLORE/size]      : amplitude 10->14 -- push magnitude; reveal the size-vs-overshoot boundary.
#   b33_slo20   [EXPLORE/size]      : stiff_lo 30->20 -- softer floor; clean peak_ratio read (revisit fact #26 in rotating regime).
#
b33_dev18 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b33_amp12 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 12 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b33_ghi20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b33_dev20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.20 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b33_amp14 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 14 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b33_slo20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 20 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
