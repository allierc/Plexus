# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep BOTH pressures: default ~3 exploit (improve LS) · 2 explore (new morphology family) · 1 control/ablation
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 1 — BASELINE + RE-TEST R²-ERA CLOSURES
# Parent: archive s1 (gain0=0.5, learn=fibre,gain,dur, 2400it, fibre wl=28.8/angle=0.17/amp=0.39/phase=0.41,
#   stiff=[100,100], amp=10, drag=30, dur0=14, dur_hi=30)
#
b1_control : --gain0 0.5 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 10 --drag_k 30 --dur0 14 --dur_hi 30
b1_frozen : --gain0 0.5 --learn gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 10 --drag_k 30 --dur0 14 --dur_hi 30
b1_depth3600 : --gain0 0.5 --learn fibre,gain,dur --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 10 --drag_k 30 --dur0 14 --dur_hi 30
b1_stiff_coarse : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 50 --stiff_hi 150 --amplitude 10 --drag_k 30 --dur0 14 --dur_hi 30
b1_gain03 : --gain0 0.3 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 10 --drag_k 30 --dur0 14 --dur_hi 30
b1_amp12_g05 : --gain0 0.5 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
