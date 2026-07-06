# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 34 — CONFIRM + PUSH the GAIN CEILING (the newly-found size/uniformity lever @rot1.0)
#
# PARENT = ghi20 (NEW RECORD LS=0.509): dev18 with gain_hi 1.5->2.0. rot1.0 + soft-floor stiff[30,300] ω5,
#   drag40, amp10, gain[0.2,2.0] gain0=0.5, SIREN fibre-ON dev0.18, dur_hi11, substeps10.
#
# SURPRISE (B33): gain_hi 1.5->2.0 (ghi20) WON + set a RECORD (0.492->0.509), OVERTURNING fact #28
#   ("gain_hi>1.5 buys nothing"). In the enclosure-solved ROTATING regime the gain ceiling is a per-region
#   SIZE+UNIFORMITY lever: it rescues the small radial-stub nodes (dev18 {+0.01,-0.07,+0.11,+0.14} -> ghi20
#   {+0.20,+0.45,+0.36,+0.42}, ALL-POSITIVE), raising peak_ratio 0.50->0.53 + area 0.34->0.39. fibre_dev 0.20
#   reproduces it (dev20 0.505); base amplitude (amp12/amp14) + floor (slo20) do NOT (global drive overshoots).
#
# HYPOTHESIS: the gain ceiling is a real per-region size lever @rot1.0; peak_ratio keeps rising with gain_hi
#   until the old soft-floor runaway edge (~2.0-2.5, fact #28) is re-hit -- OR rotation raises that edge.
#   FALSIFIER: ghi20 replicate < ~0.48 (record was seed-luck, campaign law) OR ghi22/ghi25 diverge/LS drops
#   (2.0 is the edge even under rotation). OVERTURN-CONFIRM: ghi22 raises peak_ratio+LS -> monotone size knob.
#   Read peak_ratio + area_ratio + LS_SD from RESIDUAL_MORPHOLOGY, not LS alone.
#
#   Balance: 3 EXPLOIT (ghi20 replicate, ghi22, dev20-stack) . 2 EXPLORE (ghi25, glo30) . 1 CONTROL (ghi15).
#
#   b34_ghi20  [EXPLOIT/replicate] : gain_hi 2.0 -- reproduce the RECORD (0.509 single draw); net the lottery.
#   b34_ghi22  [EXPLOIT/push]      : gain_hi 2.0->2.2 -- does peak_ratio keep rising (monotone size knob)?
#   b34_dev20  [EXPLOIT/stack]     : fibre_dev 0.18->0.20 -- do the two per-region size levers ADD on the record?
#   b34_ghi25  [EXPLORE/edge]      : gain_hi 2.0->2.5 -- does rotation raise the runaway edge (was RUNAWAY @soft-floor,rot=0, #28)?
#   b34_glo30  [EXPLORE/floor]     : gain_lo 0.2->0.3 -- raise the gain FLOOR: attack the small nodes from below (uniformity).
#   b34_ghi15  [CONTROL/causal]    : gain_hi 2.0->1.5 -- =dev18 baseline; confirm the +0.035 IS the ceiling (paired contrast).
#
b34_ghi20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b34_ghi22 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.2 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b34_dev20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.20 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b34_ghi25 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.5 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b34_glo30 : --gain0 0.5 --gain_src siren --gain_lo 0.3 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b34_ghi15 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.18 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
