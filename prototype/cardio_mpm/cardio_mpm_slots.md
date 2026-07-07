# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 36 — EXACT RE-ISSUE of B35 (the FIBRE_DEV DOSE LADDER). B35 was a 0-archive CODE-CRASH loss, NOT science.
#
# LOSS ROOT CAUSE (fixed): the 948ff60 transfer-family refactor renamed p2g->mpm_scatter, g2p->mpm_gather in the
#   registry AND the cardio spec YAML, but cardio_mpm_train.py still hardcoded p2g/g2p (mpm_ops L590, p_op L515)
#   -> KeyError:'p2g' at step_frame, all 6 slots died ~10-19s. FIXED (rename both sites, grep-clean). Specs are
#   unchanged and well-formed; this batch re-runs them on the fixed trainer. Last REAL data = B34 (dev20 0.505).
#
# PARENT = dev20 (B34 WINNER, LS=0.505): fibre_dev 0.20, gain_hi 2.0, rot1.0 + soft-floor stiff[30,300] w5,
#   drag40, amp10, gain[0.2,2.0] g0=0.5, SIREN fibre-ON, dur_hi11, substeps10.
#
# SCIENCE (unchanged from B35): SIZE is the sole open Phase-2 axis (peak_ratio ~0.51 = sim HALF real). The per-
#   region SIZE lever @rot1.0 is FIBRE_DEV (gain ceiling RETRACTED inert, B34). Dosing fibre_dev decides SIZE:
#   either peak_ratio keeps RISING past dev0.20 (fibre SOLVES size -> close SIZE positive) OR it CAPS ~0.51 /
#   rolls off (structural size ceiling -> close SIZE as dose-confirmed structurally-limited). B32 rolled off at
#   dev0.25 on LS @soft-floor; test whether SIZE (peak_ratio) does the same or decouples from LS.
#   FALSIFIER for "fibre solves size": dev22/dev25 leave peak_ratio flat ~0.51 -> size is capped, NOT solvable.
#   Read peak_ratio + area_ratio + LS_SD from RESIDUAL_MORPHOLOGY, not LS alone.
#
#   Balance: 2 EXPLOIT (dev20 replicate, dev22) . 3 EXPLORE (dev25, dev30, fwl40) . 1 CONTROL (ghi15).
#
#   b36_dev20  [EXPLOIT/replicate] : fibre_dev 0.20 -- net the lottery on the B34 winner (0.505 single draw).
#   b36_dev22  [EXPLOIT/dose]      : fibre_dev 0.20->0.22 -- does peak_ratio rise past the dev20 point?
#   b36_dev25  [EXPLORE/dose]      : fibre_dev 0.20->0.25 -- roll-off point (B32 rolled off here on LS @soft-floor)?
#   b36_dev30  [EXPLORE/dose-far]  : fibre_dev 0.20->0.30 -- map the far dose: where does size/LS collapse?
#   b36_fwl40  [EXPLORE/scale]     : fibre_wl 28.8->40 -- COARSER heterogeneity: is size SCALE-sensitive (coarsen before concluding scale-inert)?
#   b36_ghi15  [CONTROL/causal]    : gain_hi 2.0->1.5 -- confirm gain ceiling still inert at the dev20 fibre point (paired contrast).
#
b36_dev20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.20 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b36_dev22 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.22 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b36_dev25 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b36_dev30 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.30 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b36_fwl40 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 2.0 --siren_fibre 1 --fibre_dev 0.20 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 40 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b36_ghi15 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.20 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
