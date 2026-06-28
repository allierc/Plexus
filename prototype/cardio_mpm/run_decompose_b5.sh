#!/bin/bash
# Residual decomposition for all Batch 5 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

# s0: angle05
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.5 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b05_s0_b5_angle05/checkpoints/model_02399.pt --eval_decompose archive/p3_b05_s0_b5_angle05/residual.png --device cuda:0
echo "DONE s0"

# s1: angle_neg
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle -0.3 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b05_s1_b5_angle_neg/checkpoints/model_02399.pt --eval_decompose archive/p3_b05_s1_b5_angle_neg/residual.png --device cuda:0
echo "DONE s1"

# s2: phase_shift
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 1.2 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b05_s2_b5_phase_shift/checkpoints/model_02399.pt --eval_decompose archive/p3_b05_s2_b5_phase_shift/residual.png --device cuda:0
echo "DONE s2"

# s3: no_stiff
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b05_s3_b5_no_stiff/checkpoints/model_02399.pt --eval_decompose archive/p3_b05_s3_b5_no_stiff/residual.png --device cuda:0
echo "DONE s3"

# s4: stiff_hi300
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b05_s4_b5_stiff_hi300/checkpoints/model_02399.pt --eval_decompose archive/p3_b05_s4_b5_stiff_hi300/residual.png --device cuda:0
echo "DONE s4"

# s5: control
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b05_s5_b5_control/checkpoints/model_02399.pt --eval_decompose archive/p3_b05_s5_b5_control/residual.png --device cuda:0
echo "DONE s5"

echo "ALL DONE"
