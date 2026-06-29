#!/bin/bash
# Run residual decomposition for all B9 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

# s0: deep4800 (partial at 4150it)
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 4800 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b09_s0_b9_deep4800/checkpoints/model_04150.pt --eval_decompose archive/p3_b09_s0_b9_deep4800/residual.png --device cuda:0 &

# s1: durhi25
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 25 --resume archive/p3_b09_s1_b9_durhi25/checkpoints/model_02399.pt --eval_decompose archive/p3_b09_s1_b9_durhi25/residual.png --device cuda:1 &

# s2: 3600_gain04
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.4 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b09_s2_b9_3600_gain04/checkpoints/model_03599.pt --eval_decompose archive/p3_b09_s2_b9_3600_gain04/residual.png --device cuda:2 &

# s3: dur0_10
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 30 --resume archive/p3_b09_s3_b9_dur0_10/checkpoints/model_02399.pt --eval_decompose archive/p3_b09_s3_b9_dur0_10/residual.png --device cuda:3 &

wait

# s4: durhi20
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 20 --resume archive/p3_b09_s4_b9_durhi20/checkpoints/model_02399.pt --eval_decompose archive/p3_b09_s4_b9_durhi20/residual.png --device cuda:0 &

# s5: ctrl3600
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b09_s5_b9_ctrl3600/checkpoints/model_03599.pt --eval_decompose archive/p3_b09_s5_b9_ctrl3600/residual.png --device cuda:1 &

wait
echo "All B9 decompositions done"
