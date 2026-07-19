# Next-batch slots — the AGENT rewrites this file each iteration (<=6 non-comment lines).
# Format (one slot per line):   <slot_name> : <args>
#   - spec is ALWAYS material/material_aniso_cardio (do not repeat it)
#   - objective defaults to LoopScore (omit --loss)
#   - each slot changes EXACTLY ONE variable from the parent (causal inference)
#   - keep stiffness/direction COARSE (low --siren_omega, larger --fibre_wl); amplitude in [10,15]
#
# ============================================================================================
# BATCH 60 — is the corner SIZE ceiling a TEMPORAL-ASYMMETRY limit? pulse_skew dose ladder on the corner.
#
#   B59 BIGGEST SURPRISE — the DRAG hypothesis FALSIFIED and the ranking FLIPPED against it. LS rose WEAKLY with MORE drag
#     (drag25 0.488 -> 30 0.521 -> 35 0.527 -> 40 0.536 -> 55 0.539), the OPPOSITE of "lower drag lets the excursion travel further."
#     peak_ratio stayed FLAT across drag{25,30,35,40,55} (0.56-0.61, no trend), area_ratio FLAT (0.43-0.48). Lowering drag did NOT buy
#     size; at drag25 it RE-ADDED overshoot (ampL 0.018, minor 1.06) and dropped LS to batch-worst. -> the corner SIZE ceiling (peak ~0.60,
#     area ~0.47 at energy ~0.95 ~= real) is NOT set by overdamping; it is STRUCTURAL to the FS-regulated elastic active-stress language
#     [dose-confirmed ✓ null]. drag40/55 LS-tied (Δ0.003) -> drag40 op point replicates n>=3.
#   B59 second finding: MIRROR/chirality-flip family = NULL. rotneg (rot_stress -1.5) held chir 0.852 (NOT flipped) — the learned fibre
#     field ABSORBS the sign flip and re-derives correct handedness to match the fixed real data. rot_stress SIGN is chirality-inert
#     (generalizes #33 spin sign-blindness). rotneg drew batch-MAX area 0.528, peak 0.600, dur 6.2 = valid reversed-axis corner draw.
#   B59 rank by LS: drag55 0.539 (WINNER, noise) ~= drag40 0.536 > drag35 0.527 > rotneg 0.525 > drag30 0.521 > drag25 0.488.
#
#   The SIZE residual is now confirmed structural to EVERY mechanical/magnitude lever: orientation (fibre_dev B58, rot B58), spin, gain
#     (#28), amplitude (#34), drag (B59) — ALL inert on peak/area at the corner, at energy already ~0.95 ~= real. So the residual is NOT a
#     force/magnitude gap; it is TEMPORAL: real peak/energy is HIGHER than sim (real motion more temporally CONCENTRATED = a sharper
#     systolic excursion; sim spreads the same ~real work over a smaller loop). The one untested-on-the-corner axis is the pulse TIME
#     PROFILE: the campaign has run a SYMMETRIC activation pulse (--pulse_skew 1.0) throughout the corner era.
#
#   B60 HYPOTHESIS (one, predictive): the corner SIZE ceiling is a TEMPORAL-SYMMETRY limit. --pulse_skew >1 widens the RELEASE side of the
#     Gaussian while keeping contraction sharp (train.py:669-673) = fast systole / slow diastole = the physiological twitch. On the RADIAL
#     regime (B25/B26) skew=2.0 was a "shape/overshoot lever, not size" (fact #27) — but that was pre-composition (amp10, NO spin/FS/
#     over-rotation), where skew's recoil dissipated as overshoot (no FS to cap it) and its added openness had no rot to convert into
#     enclosure. Per the regime-transfer rule that null is a HYPOTHESIS on the composed corner, where FS now CAPS the recoil overshoot and
#     rot already supplies enclosure -> skew's openness may finally ENLARGE the loop (peak/area^) instead of dissipating.
#   CONFIRMER: peak_ratio rises MONOTONE with pulse_skew at chir>=0.83 AND loopiness>=0.95 AND ampL<=0.02 -> SIZE ceiling is
#     temporal-asymmetry-limited; skew is a composed-base size lever; SIZE REOPENS on the corner; re-pin the op point at the best skew.
#   FALSIFIER: peak/area FLAT across the skew ladder (skew inert) OR ampL re-explodes / loops decohere (fact #27 REPLICATES on the corner)
#     -> the peak ceiling is structural to the SYMMETRIC *and* asymmetric elastic active-stress language [dose-confirmed ✓ null; SIZE
#     stays ✓-capped ~0.60 at the corner, and the corner is the terminal Phase-3 deliverable].
#   SMALLEST distinguishing experiment: a 6-rung pulse_skew dose ladder {0.7,1.0,1.5,2.0,2.5,3.5} straddling the symmetric baseline settles
#     the temporal-size question in ONE batch; 1.0 anchors the corner (n=4); 0.7 is the reverse-asymmetry causal control; 2.0 directly
#     regime-transfers fact #27 (radial->corner); 3.5 maps the strong-asymmetry family / decoherence onset.
#
#   Parent = drag40 CORNER = rot_stress 1.5 + spin_omega 0.2 + spin_k 20 + amplitude 14 + stretch_activation 2.0 (FS beta2) + fibre_dev 0.45
#     + stiff[30,300] + gain[0.2,1.5] g0.5 + siren_omega 5 + drag40 + dur_hi 11 + fibre_wl 28.8 + pulse_skew 1.0. Amplitude fixed 14 in
#     [10,15]. n_iter 2400. Each slot changes EXACTLY ONE variable: --pulse_skew. Fields kept COARSE.
#   Balance = 3 EXPLOIT (skew15, skew20, skew25 = physiological asymmetry) . 2 EXPLORE (skew07 reverse-asymmetry control, skew35 strong
#     asymmetry / decoherence probe) . 1 CONTROL (ctrl = pulse_skew 1.0, corner anchor n=4).
#
skew07 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.45 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 14 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.5 --spin_omega 0.2 --spin_k 20 --stretch_activation 2.0 --pulse_skew 0.7
ctrl : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.45 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 14 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.5 --spin_omega 0.2 --spin_k 20 --stretch_activation 2.0 --pulse_skew 1.0
skew15 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.45 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 14 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.5 --spin_omega 0.2 --spin_k 20 --stretch_activation 2.0 --pulse_skew 1.5
skew20 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.45 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 14 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.5 --spin_omega 0.2 --spin_k 20 --stretch_activation 2.0 --pulse_skew 2.0
skew25 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.45 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 14 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.5 --spin_omega 0.2 --spin_k 20 --stretch_activation 2.0 --pulse_skew 2.5
skew35 : --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.45 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 30 --stiff_hi 300 --amplitude 14 --drag_k 40 --dur0 10 --dur_hi 11 --rot_stress 1.5 --spin_omega 0.2 --spin_k 20 --stretch_activation 2.0 --pulse_skew 3.5
