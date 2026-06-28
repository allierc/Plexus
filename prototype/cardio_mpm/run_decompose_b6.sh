#!/bin/bash
# Run residual decomposition for all B6 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b06_s5_b6_control/checkpoints/model_02399.pt --eval_decompose archive/p3_b06_s5_b6_control/residual.png --device cuda:0 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.4 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b06_s4_b6_gain04/checkpoints/model_02399.pt --eval_decompose archive/p3_b06_s4_b6_gain04/residual.png --device cuda:1 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.8 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b06_s3_b6_fibre_amp08/checkpoints/model_02399.pt --eval_decompose archive/p3_b06_s3_b6_fibre_amp08/residual.png --device cuda:2 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --siren_fibre 1 --fibre_dev 0.15 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b06_s0_b6_siren_fibre_015/checkpoints/model_02399.pt --eval_decompose archive/p3_b06_s0_b6_siren_fibre_015/residual.png --device cuda:3 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --siren_fibre 1 --fibre_dev 0.3 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b06_s1_b6_siren_fibre_03/checkpoints/model_02399.pt --eval_decompose archive/p3_b06_s1_b6_siren_fibre_03/residual.png --device cuda:0 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --siren_fibre 1 --fibre_dev 0.5 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b06_s2_b6_siren_fibre_05/checkpoints/model_02399.pt --eval_decompose archive/p3_b06_s2_b6_siren_fibre_05/residual.png --device cuda:1 &

wait
echo "All decompositions done"
