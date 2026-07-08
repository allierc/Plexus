# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 41 = PHASE 3 batch 1 (prestress/viscoelastic operator discovery) — FIRST REAL RUN.
#
# ✅ DATA-LOSS BLOCKER RESOLVED. B38→B40 were execution losses from ONE `../cardio` deletion (B38 lost the
#   cosmetic cardio_unet import [fixed]; B39/B40 lost the gitignored REAL TARGET cardio_real.npz [hard]). A HUMAN
#   restored it 2026-07-08 03:52 at the resolver's preferred self-contained slot:
#     /workspace/Plexus/prototype/cardio_mpm/cardio_real.npz  (54 MB ≈ pos[~360, 137^2, 2] float32)
#   B40 ran 2026-07-07 17:12 (BEFORE the restore) so it 0-archived; B41 is the FIRST batch since B37 with data.
#   Pre-flight verified: both Phase-3 operators are wired in cardio_mpm_train.py (--residual_stress/_amp/_hidden/
#   _omega + `residual` in --learn @604-628; --tau -> lvl.is_visco/visco_tau @520/556) and default-OFF = exact
#   baseline. If this batch 0-archives AGAIN, read a slot .err FIRST: FileNotFoundError on the npz = the
#   self-contained copy was deleted (re-restore); import/spec-load traceback = a code/rename landmine.
#
# SCIENCE (unchanged since B37; SIZE ✓-capped ~0.52, Phase 2 CLOSED B38). PHASE-3 QUESTION: is the ~0.52
#   peak_ratio SIZE cap an active-stress AMPLITUDE limit, or a missing PRE-STRESS / residual-stress state?
#   Two engine-ready operators (both default-OFF = EXACT baseline): prestress (--residual_stress 1 --residual_amp a,
#   learned SIREN rest tensor F_res=I+a*tanh(dF), Fe=F@F_res^-1, add 'residual' to --learn) · viscoelastic (--tau t,
#   Maxwell relaxation exp(-dt/t), t=0=OFF). PARENT/ctrl = d25g15 (fibre_dev 0.25, gain_hi 1.5, gain[0.2,1.5] g0=0.5,
#   SIREN fibre-ON, rot1.0, soft-floor stiff[30,300] w5, drag40, amp10, dur_hi11, substeps10, 2400it).
#
# FREEZE RULE: a slot COUNTS only if it raises peak_ratio WHILE holding enclosure/chir/shape/uniformity (read the
#   full RESIDUAL_MORPHOLOGY, not LS alone). FALSIFIER: any slot pushes peak_ratio past ~0.53 with the ✓ axes intact
#   -> prestress/viscoelasticity is the missing operator, SIZE reopens as SOLVED. Clean dose-confirmed null
#   (neither exceeds 0.53, axes held) -> the cap is deeper (constitutive nonlinearity), operators join the rejected
#   record — still a ✓. No combo slot yet (only after one alone shows signal).
#
#   Balance: 1 CONTROL (ctrl anchor) . 3 EXPLOIT-dose (res_lo/mid/hi prestress ladder) . 2 EXPLORE (visco_mid/hi
#   viscoelastic dose — the emergent counterpart). Design well-formed & unchanged from B40 (which never ran).
#
#   ctrl       [CONTROL/anchor]   : both operators OFF -- the ~0.52 peak_ratio anchor (must reproduce d25g15).
#   res_lo     [EXPLOIT/dose]     : --residual_stress 1 --residual_amp 0.1 -- weak imposed prestress.
#   res_mid    [EXPLOIT/dose]     : --residual_amp 0.2 -- mid prestress (engine default alpha).
#   res_hi     [EXPLOIT/dose]     : --residual_amp 0.3 -- strong prestress; locate roll-off if any.
#   visco_mid  [EXPLORE/emergent] : --tau 0.05 -- moderate viscoelastic drift (emergent residual).
#   visco_hi   [EXPLORE/emergent] : --tau 0.02 -- more fluid (stronger drift); brackets the tau dose.
#
ctrl : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
res_lo : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff,residual --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0 --residual_stress 1 --residual_amp 0.1 --residual_hidden 128 --residual_omega 5
res_mid : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff,residual --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0 --residual_stress 1 --residual_amp 0.2 --residual_hidden 128 --residual_omega 5
res_hi : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff,residual --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0 --residual_stress 1 --residual_amp 0.3 --residual_hidden 128 --residual_omega 5
visco_mid : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0 --tau 0.05
visco_hi : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.25 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0 --tau 0.02
