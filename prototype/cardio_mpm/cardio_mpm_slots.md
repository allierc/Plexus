# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep BOTH pressures: default ~3 exploit (improve LS) · 2 explore (new morphology family) · 1 control/ablation
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 15 — SPATIAL GAIN CONVERGENCE: IS THE amp=10+sgain RESULT (LS=0.323) STILL CLIMBING?
# Parent: B14-s4 sgain_amp10 (LS=0.323, gain_src=siren, amp=10, stiff [80,300], dur_hi=11→dur=10.0, ω=5, 2400it)
#
# Hypothesis: "Spatial gain at amp=10 is NOT converged at 2400it (B14 showed +0.059 from 2400→3600it
#   at amp=12). Deeper training (3600it) of the amp=10 best config will push LS further. Additionally,
#   the gain SIREN omega (ω=5, inherited from stiffness) may not be optimal for the gain field —
#   a coarser gain field (ω=3) may better match the tissue's amplitude variation scale."
#
# Code change: added --gain_omega flag to decouple gain SIREN frequency from stiffness SIREN
#
# EXPLOIT (3): push LS above 0.323 via depth and fine-tuning
b15_deep3600 : --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11
b15_deep4800 : --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 4800 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11
b15_amp11 : --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 11 --drag_k 30 --dur0 10 --dur_hi 11
# EXPLORE (2): test gain SIREN spatial frequency and gain init landscape
b15_gomega3 : --gain0 0.5 --gain_src siren --gain_omega 3 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11
b15_durhi13 : --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 13
# CONTROL (1): amp=10 WITHOUT spatial gain — confirms spatial gain is responsible
b15_ctrl_nosgain : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11
