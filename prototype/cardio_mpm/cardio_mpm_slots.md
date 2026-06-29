# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep BOTH pressures: default ~3 exploit (improve LS) · 2 explore (new morphology family) · 1 control/ablation
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 11 — EXPLORING THE VERY-SHORT-DURATION REGIME (dur≈11)
# Parent: B10-s4 durhi15 (LS=0.196, 2400it, dur_hi=15, dur→11.3, stiff [80,300], gain0=0.5, amp=12, omega=5, dur0=10)
#
# Hypothesis: "dur≈11 is NOT the floor — even shorter pulses (dur_hi=12 or 10) will continue to
#   tame the catastrophic node because the overshoot energy scales with pulse duration. Below some
#   threshold, pulse energy will be insufficient for any loop structure, creating a new
#   morphology family (degenerate stubs). The transition between productive short-duration
#   and degenerate no-motion is the next boundary to map."
#
# EXPLOIT (3): push the dur≈11 regime — shorter ceiling, deeper training, lower amplitude
b11_durhi12 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12
b11_durhi10 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 8 --dur_hi 10
b11_deep3600 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 15
# EXPLORE (2): test amplitude and stiffness in the new regime
b11_amp10 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 15
b11_uniform_stiff : --gain0 0.5 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 15
# CONTROL (1): reproduce B10-s4 parent exactly
b11_ctrl : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 15
