# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep BOTH pressures: default ~3 exploit (improve LS) · 2 explore (new morphology family) · 1 control/ablation
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 7 — ISOLATE SIREN FIBRE FROM STIFFNESS INTERACTION
# Parent A (stiff-active): B5-s4 stiff_hi300 (LS=0.149, stiff [80,300], gain0=0.5, amp=12, ω=5, 2400it)
# Parent B (fibre-only):   B5-s3 no_stiff   (LS=0.118, uniform stiff, gain0=0.5, amp=12, 2400it)
#
# Hypothesis: "The catastrophe redistribution in B6 is caused by SIREN fibre × SIREN stiffness
#   INTERACTION. SIREN fibre with UNIFORM stiffness (no stiffness SIREN) avoids this interaction
#   and should break the fibre-only ceiling (LS≈0.118). A coarser fibre SIREN (ω=3) may produce
#   more coherent corrections than ω=5's noisy patterns."
#
# EXPLOIT (3): SIREN fibre isolated from stiffness, and with stiffness at best config
b7_siren_fibre_nostiff : --gain0 0.5 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --siren_fibre 1 --fibre_dev 0.3 --siren_omega 5 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
b7_siren_fibre_nostiff_coarse : --gain0 0.5 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --siren_fibre 1 --fibre_dev 0.3 --siren_omega 3 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
b7_siren_fibre_stiff300 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --siren_fibre 1 --fibre_dev 0.3 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
# EXPLORE (2): amplitude probe + fibre-only ablation baseline
b7_amp10_stiff300 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 14 --dur_hi 30
b7_fibreonly_ctrl : --gain0 0.5 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
# CONTROL (1): reproduce B5 best (stiff [80,300])
b7_stiff300_ctrl : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
