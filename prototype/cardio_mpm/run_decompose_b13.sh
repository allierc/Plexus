#!/bin/bash
# Run residual decomposition for all B13 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

# s0: deep3600
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b13_s0_b13_deep3600/checkpoints/model_03599.pt --eval_decompose archive/p3_b13_s0_b13_deep3600/residual.png --device cuda:0 &

# s1: hidden384
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --siren_hidden 384 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b13_s1_b13_hidden384/checkpoints/model_02399.pt --eval_decompose archive/p3_b13_s1_b13_hidden384/residual.png --device cuda:1 &

# s2: layers4
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --siren_layers 4 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b13_s2_b13_layers4/checkpoints/model_02399.pt --eval_decompose archive/p3_b13_s2_b13_layers4/residual.png --device cuda:2 &

# s3: lr5e4
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --lr 5e-4 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b13_s3_b13_lr5e4/checkpoints/model_02399.pt --eval_decompose archive/p3_b13_s3_b13_lr5e4/residual.png --device cuda:3 &

wait

# s4: dur0_8
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 8 --dur_hi 11 --resume archive/p3_b13_s4_b13_dur0_8/checkpoints/model_02399.pt --eval_decompose archive/p3_b13_s4_b13_dur0_8/residual.png --device cuda:0 &

# s5: ctrl
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b13_s5_b13_ctrl/checkpoints/model_02399.pt --eval_decompose archive/p3_b13_s5_b13_ctrl/residual.png --device cuda:1 &

wait
echo "All B13 decompositions done"
