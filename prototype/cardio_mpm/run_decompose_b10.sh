#!/bin/bash
# Run residual decomposition for all B10 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

# s0: fibre_angle05
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.5 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 30 --resume archive/p3_b10_s0_b10_fibre_angle05/checkpoints/model_02399.pt --eval_decompose archive/p3_b10_s0_b10_fibre_angle05/residual.png --device cuda:0 &

# s1: fibre_phase12
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 1.2 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 30 --resume archive/p3_b10_s1_b10_fibre_phase12/checkpoints/model_02399.pt --eval_decompose archive/p3_b10_s1_b10_fibre_phase12/residual.png --device cuda:1 &

# s2: deep3600
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 30 --resume archive/p3_b10_s2_b10_deep3600/checkpoints/model_03599.pt --eval_decompose archive/p3_b10_s2_b10_deep3600/residual.png --device cuda:2 &

# s3: wl35
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 35 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 30 --resume archive/p3_b10_s3_b10_wl35/checkpoints/model_02399.pt --eval_decompose archive/p3_b10_s3_b10_wl35/residual.png --device cuda:3 &

wait

# s4: durhi15
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 15 --resume archive/p3_b10_s4_b10_durhi15/checkpoints/model_02399.pt --eval_decompose archive/p3_b10_s4_b10_durhi15/residual.png --device cuda:0 &

# s5: ctrl
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 30 --resume archive/p3_b10_s5_b10_ctrl/checkpoints/model_02399.pt --eval_decompose archive/p3_b10_s5_b10_ctrl/residual.png --device cuda:1 &

wait
echo "All B10 decompositions done"
