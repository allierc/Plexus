# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep BOTH pressures: default ~3 exploit (improve LS) · 2 explore (new morphology family) · 1 control/ablation
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 12 — THE GOLDILOCKS ZONE: HIGH LS + ZERO NEGATIVES
# Parent A (uniformity): B11-s0 durhi12 (LS=0.200, dur_hi=12→dur=10.0, ZERO negative nodes, stiff [80,300], gain0=0.5, amp=12, omega=5, dur0=10)
# Parent B (best mean):  B11-s1 durhi10 (LS=0.211, dur_hi=10→dur=8.5, 3 negatives)
#
# Hypothesis: "The dur=10 regime (durhi12) is a Goldilocks zone where pulse energy is too low
#   for overshoot at ANY node but sufficient for all to form loops. This zone depends on the
#   stiffness contrast: narrower stiffness range should EXTEND the Goldilocks zone to shorter
#   durations (eliminating durhi10's negatives), while deeper training should push durhi12's
#   mean LS above 0.211 while preserving the all-positive property."
#
# EXPLOIT (3): push LS in the all-positive durhi12 regime
b12_deep3600 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12
b12_durhi11 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11
b12_durhi10_narrow : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 100 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 8 --dur_hi 10
# EXPLORE (2): stiffness topology variations in the Goldilocks zone
b12_lo100 : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 100 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12
b12_narrow_stiff : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 100 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12
# CONTROL (1): reproduce the all-positive durhi12 config
b12_ctrl : --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12
