# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 37 — CLOSE THE SIZE AXIS (the last open Phase-2 axis, now near-✓).
#
# B36 RESULT (real data, dose ladder): fibre_dev is a DOSE-CONFIRMED SIZE lever @rot1.0 — peak_ratio rose MONOTONE
#   0.482(dev20)->0.500(dev22)->0.534(dev25), area 0.326->0.418, then ROLLED OFF at dev0.30 (peak 0.504). So fibre
#   heterogeneity CONVERTS to loop SIZE, peaking at dev~0.25, BUT peak_ratio still CAPS ~0.53 (sim ~half real).
#   SURPRISE: control ghi15 (gain_hi 1.5) TOPPED the batch (LS 0.516, best LS_SD 0.281) with SIZE flat -> gain_hi is
#   a UNIFORMITY lever, cleanly separate from the size channel. fwl40 (coarser fibre) INERT -> size = dev magnitude,
#   not spatial scale.
#
# PARENT = dev25 (new SIZE op point): fibre_dev 0.25, gain_hi 2.0, rot1.0 + soft-floor stiff[30,300] w5, drag40,
#   amp10, gain[0.2,2.0] g0=0.5, SIREN fibre-ON, dur_hi11, substeps10, 2400it.
#
# THE ONE QUESTION: is the ~0.53 peak_ratio cap a fibre-lever limit that a DIFFERENT mechanism breaks, or a
#   STRUCTURAL cap of the active-stress model? Read peak_ratio + area_ratio from RESIDUAL_MORPHOLOGY, not LS alone.
#   FALSIFIER for "size structurally capped": bwnar OR durhi13 raises peak_ratio ABOVE ~0.53 while holding LS+chir
#   -> a new size mechanism, keep SIZE open. If dev25 replicates ~0.53 AND both cap-tests stay <=0.53 -> SIZE is
#   dose-confirmed structurally-capped -> close SIZE ✓ -> Phase 2 COMPLETE next batch.
#
#   Balance: 3 EXPLOIT (dev25 replicate, dev27 dose, d25g15 combine) . 2 EXPLORE (bwnar, durhi13 cap-tests) . 1 CONTROL (dev20 anchor).
#
#   b37_dev25   [EXPLOIT/replicate] : fibre_dev 0.25 -- net the single-draw peak (is peak_ratio 0.534 / LS 0.512 real?)
#   b37_dev27   [EXPLOIT/dose]      : fibre_dev 0.25->0.27 -- locate the peak between dev25 and the dev30 roll-off
#   b37_d25g15  [EXPLOIT/combine]   : gain_hi 2.0->1.5 (at dev0.25) -- combine B36's two wins (dev25 size + ghi15 uniformity)
#   b37_bwnar   [EXPLORE/boundary]  : bwidth 0.06->0.03 -- does releasing the Dirichlet anchor break the ~0.53 peak cap?
#   b37_durhi13 [EXPLORE/duration]  : dur_hi 11->13 -- does longer contraction travel raise peak excursion @rot1.0? (untested size candidate)
#   b37_dev20   [CONTROL/anchor]    : fibre_dev 0.25->0.20 -- dose anchor; confirm dev25 > dev20 (net lottery both ends)
#
b37_dev25 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b37_dev27 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.27 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b37_d25g15 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b37_bwnar : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0 --bwidth 0.03
b37_durhi13 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 13 --rot_stress 1.0
b37_dev20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.20 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
