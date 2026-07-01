#!/bin/bash
# Run residual decomposition for all B14 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

# s0: sgain (spatial gain, 2400it, amp=12)
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b14_s0_b14_sgain/checkpoints/model_02399.pt --eval_decompose archive/p3_b14_s0_b14_sgain/residual.png --device cuda:0 &

# s1: sgain_deep (spatial gain, 3600it, amp=12)
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b14_s1_b14_sgain_deep/checkpoints/model_03599.pt --eval_decompose archive/p3_b14_s1_b14_sgain_deep/residual.png --device cuda:1 &

# s2: sgain_nostiff (spatial gain, no stiffness, 2400it)
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --gain_src siren --learn fibre,gain,dur --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_lo 100 --stiff_hi 100 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b14_s2_b14_sgain_nostiff/checkpoints/model_02399.pt --eval_decompose archive/p3_b14_s2_b14_sgain_nostiff/residual.png --device cuda:2 &

# s3: sgain_dur0_8 (spatial gain, dur0=8, 2400it)
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 8 --dur_hi 11 --resume archive/p3_b14_s3_b14_sgain_dur0_8/checkpoints/model_02399.pt --eval_decompose archive/p3_b14_s3_b14_sgain_dur0_8/residual.png --device cuda:3 &

wait

# s4: sgain_amp10 (spatial gain, amp=10, 2400it)
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b14_s4_b14_sgain_amp10/checkpoints/model_02399.pt --eval_decompose archive/p3_b14_s4_b14_sgain_amp10/residual.png --device cuda:0 &

# s5: ctrl (no spatial gain, 2400it)
python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 10 --dur_hi 11 --resume archive/p3_b14_s5_b14_ctrl/checkpoints/model_02399.pt --eval_decompose archive/p3_b14_s5_b14_ctrl/residual.png --device cuda:1 &

wait
echo "All B14 decompositions done"
