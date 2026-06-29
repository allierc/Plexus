#!/bin/bash
# Run residual decomposition for all B8 slots
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 20 --dur0 14 --dur_hi 30 --resume archive/p3_b08_s0_b8_drag20/checkpoints/model_02399.pt --eval_decompose archive/p3_b08_s0_b8_drag20/residual.png --device cuda:0 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 3600 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b08_s1_b8_deep3600/checkpoints/model_03599.pt --eval_decompose archive/p3_b08_s1_b8_deep3600/residual.png --device cuda:1 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --w_amp 0 --resume archive/p3_b08_s2_b8_wamp0/checkpoints/model_02399.pt --eval_decompose archive/p3_b08_s2_b8_wamp0/residual.png --device cuda:2 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 50 --dur0 14 --dur_hi 30 --resume archive/p3_b08_s3_b8_drag50/checkpoints/model_02399.pt --eval_decompose archive/p3_b08_s3_b8_drag50/residual.png --device cuda:3 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 400 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b08_s4_b8_stiff400/checkpoints/model_02399.pt --eval_decompose archive/p3_b08_s4_b8_stiff400/residual.png --device cuda:0 &

python cardio_mpm_train.py material/material_aniso_cardio --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 --amplitude 12 --drag_k 30 --dur0 14 --dur_hi 30 --resume archive/p3_b08_s5_b8_ctrl/checkpoints/model_02399.pt --eval_decompose archive/p3_b08_s5_b8_ctrl/residual.png --device cuda:1 &

wait
echo "All B8 decompositions done"
