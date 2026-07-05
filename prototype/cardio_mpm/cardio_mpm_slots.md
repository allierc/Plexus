# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss; set it only for an occasional r2 diagnostic)
#   - each slot changes EXACTLY ONE variable from the current best parent
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# BATCH 28 — ROTATING CONTRACTION AXIS (attack the AREA-ENCLOSURE residual, take 2).
#
# WHAT B27 SETTLED: the travelling-wave hypothesis (fact #29) is FALSIFIED. Over tw_amp 0->6->12->20 the
#   real-referenced enclosure_row moved the WRONG way, monotonically: area_ratio 0.130->0.085, loopiness_ratio
#   0.503->0.365, energy 0.94->0.66, ampL 0.004->0.114, LS 0.360->0.249. Staggered timing DECOHERES a still-uniaxial
#   contraction (Dirichlet-pinned interior regions fire out of phase and partly cancel) -- it does NOT rotate a
#   particle's FORCE DIRECTION, so loops get THINNER not fatter. Optimizer anchor reconfirmed: hi400 LS=0.369,
#   ampL=0.001 (lowest overshoot ever), zero negatives.
#
# THE QUESTION: enclosure requires the contraction AXIS to ROTATE DURING the beat. A single axis n(x,y) with a
#   near-symmetric temporal envelope is TIME-REVERSIBLE (contract along n -> release back along n -> retrace ->
#   ~zero area; the ~0.13 area we see is only inertial/damping lag = thin loops). If the axis rotates over the
#   beat, the contraction and release half-cycles push along DIFFERENT directions -> the trajectory opens into an
#   ellipse -> area rises.
#
# MECHANISM: --rot_stress (radians, FIXED knob): each frame the contraction axis = theta(x,y) +
#   rot_stress*sin(2*pi*(fr-onset)/period) -- a mean-zero, phase-locked axis SWING over the beat. 0=OFF (fixed
#   axis, byte-identical to the old path). Code (train.py, dir_at() + 4 frame-stepping sites), differentiable in
#   the fibre theta. READ the enclosure_row area_ratio/loopiness_ratio/minor_axis_ratio (want them to RISE with
#   |rot_stress|) and chir_match (sign test). rot= now logged in progress.txt.
#
#   PARENT = b27_ctrl record family: stiff[50,300] SIREN w5, drag40, amp10, gain[0.2,1.5], fibre-ON dev005,
#   dur_hi11, substeps10, tw0, rot0. Balance: 1 CONTROL . 4 EXPLORE (rot dose + sign) . 1 EXPLOIT.
#
#   b28_ctrl   [CONTROL]        : rot_stress 0 (exact parent) -- enclosure_row reference + lottery anchor
#   b28_rot03  [EXPLORE/loop]   : rot_stress 0.3  -- mild axis swing (~17 deg): does area_ratio/loopiness rise? (dose 1)
#   b28_rot06  [EXPLORE/loop]   : rot_stress 0.6  -- stronger swing (~34 deg) (dose 2)
#   b28_rot10  [EXPLORE/loop]   : rot_stress 1.0  -- strong swing (~57 deg); find the overshoot/instability edge (dose 3)
#   b28_rotneg [EXPLORE/chir]   : rot_stress -1.0 -- SIGN test (ref=rot10): does reversing the swing flip chirality/enclosure?
#   b28_hi400  [EXPLOIT]        : stiff_hi 300->400 (rot 0) -- reconfirm the 0.369 clean-overshoot optimizer anchor
#
b28_ctrl : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 50 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11
b28_rot03 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 50 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 0.3
b28_rot06 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 50 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 0.6
b28_rot10 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 50 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.0
b28_rotneg : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 50 --stiff_hi 300 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress -1.0
b28_hi400 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 50 --stiff_hi 400 --amplitude 10 --drag_k 40 --dur0 10 --dur_hi 11
