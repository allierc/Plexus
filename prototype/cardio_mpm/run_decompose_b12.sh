#!/bin/bash
# Run residual decomposition for all B12 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

# s0: deep3600
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12 --resume archive/p3_b12_s0_b12_deep3600/checkpoints/model_03599.pt --eval_decompose archive/p3_b12_s0_b12_deep3600/residual.png --device cuda:0 &

# s1: durhi11
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b12_s1_b12_durhi11/checkpoints/model_02399.pt --eval_decompose archive/p3_b12_s1_b12_durhi11/residual.png --device cuda:1 &

# s2: durhi10_narrow
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 100 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 8 --dur_hi 10 --resume archive/p3_b12_s2_b12_durhi10_narrow/checkpoints/model_02399.pt --eval_decompose archive/p3_b12_s2_b12_durhi10_narrow/residual.png --device cuda:2 &

# s3: lo100
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 100 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12 --resume archive/p3_b12_s3_b12_lo100/checkpoints/model_02399.pt --eval_decompose archive/p3_b12_s3_b12_lo100/residual.png --device cuda:3 &

wait

# s4: narrow_stiff
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 100 --stiff_hi 200 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12 --resume archive/p3_b12_s4_b12_narrow_stiff/checkpoints/model_02399.pt --eval_decompose archive/p3_b12_s4_b12_narrow_stiff/residual.png --device cuda:0 &

# s5: ctrl
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 12 --resume archive/p3_b12_s5_b12_ctrl/checkpoints/model_02399.pt --eval_decompose archive/p3_b12_s5_b12_ctrl/residual.png --device cuda:1 &

wait
echo "All B12 decompositions done"
