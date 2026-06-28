# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep BOTH pressures: default ~3 exploit (improve LS) · 2 explore (new morphology family) · 1 control/ablation
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 8 — BREAK THE LS≈0.15 PLATEAU: UNTESTED PHYSICAL + OPTIMIZATION PARAMETERS
# Parent: B7-s5 stiff300_ctrl (LS=0.151, stiff [80,300], gain0=0.5, amp=12, ω=5, 2400it)
#
# Hypothesis: "The parametric model has plateaued at LS≈0.15 with all spatial-field levers exhausted.
#   Progress requires probing UNTESTED physical parameters (drag_k) that affect the dynamic response
#   (loop shape via damping timescale), or optimization strategies (deeper training, w_amp ablation)
#   that improve the global solution. Different drag may shift loop morphology toward the target."
#
# EXPLOIT (3): try to break the plateau via drag, deeper training, w_amp ablation
b8_drag20 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 20 --dur0 14 --dur_hi 30
b8_deep3600 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
b8_wamp0 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --w_amp 0
# EXPLORE (2): drag sweep to map morphology families + wider stiffness range
b8_drag50 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 50 --dur0 14 --dur_hi 30
b8_stiff400 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 400 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
# CONTROL (1): reproduce B7 best
b8_ctrl : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30
